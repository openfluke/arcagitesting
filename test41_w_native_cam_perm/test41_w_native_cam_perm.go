package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/systems/tween"
	"github.com/openfluke/welvet/weights"
)

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 41-W NATIVE CAM PERM — dtype × quant × mode × native cameral arch @ SIMD
// ═══════════════════════════════════════════════════════════════════════════════
//
// Like test41_w_sine_ada_perm, but cameral arches use Hemispheres(n), not
// hand-wired Parallel. Measuring via welvet/lucy (+ thread-CPU duty clock).
//
// Resume: results/cell_*.json — skip passed cells on rerun.
//

const (
	InputSize  = 10
	HiddenSize = 32
	OutputSize = 1
	MeshWidth  = 32
	NumLayers  = 3

	LearningRate     = float64(0.01)
	InitScale        = float32(0.5)
	TweenBudgetFloor = float32(0.8)

	SinePoints     = 100
	SineResolution = 0.1
	AdaptWindows   = 4 // perm / lucy default (4×50ms = 200ms)
	TrainInterval  = 10 * time.Millisecond
)

type ArchKind int

const (
	ArchDense ArchKind = iota
	ArchBicameral
	ArchTricameral
	ArchQuadcameral
)

var archNames = map[ArchKind]string{
	ArchDense:       "Dense",
	ArchBicameral:   "Bicameral",
	ArchTricameral:  "Tricameral",
	ArchQuadcameral: "Quadcameral",
}

func archHemiCount(a ArchKind) int {
	switch a {
	case ArchBicameral:
		return 2
	case ArchTricameral:
		return 3
	case ArchQuadcameral:
		return 4
	default:
		return 0
	}
}

func isCameral(a ArchKind) bool { return archHemiCount(a) >= 2 }

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
	ModeMeshBP
	ModeMeshTween
	ModeMeshTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP: "NormalBP", ModeStepBP: "StepBP", ModeTween: "Tween", ModeTweenChain: "TweenChain",
	ModeStepTween: "StepTween", ModeStepTweenChain: "StepTweenChain",
	ModeMeshBP: "MeshBP", ModeMeshTween: "MeshTween", ModeMeshTweenChain: "MeshTweenChain",
}

func isMesh(m TrainingMode) bool {
	return m == ModeMeshBP || m == ModeMeshTween || m == ModeMeshTweenChain
}
func isTween(m TrainingMode) bool {
	switch m {
	case ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain, ModeMeshTween, ModeMeshTweenChain:
		return true
	}
	return false
}
func useChain(m TrainingMode) bool {
	return m == ModeTweenChain || m == ModeStepTweenChain || m == ModeMeshTweenChain
}

type CellSpec struct {
	Arch    ArchKind
	Mode    TrainingMode
	DType   core.DType
	Quant   quant.Format
	Backend core.Backend
}

func (c CellSpec) Key() string {
	return fmt.Sprintf("%s_%s_%s_%s_simd",
		archNames[c.Arch], modeNames[c.Mode], c.DType.String(), c.Quant.String())
}

func (c CellSpec) Label() string {
	return fmt.Sprintf("%s/%s/%s/%s", archNames[c.Arch], modeNames[c.Mode], c.DType.String(), c.Quant.String())
}

type TimeWindow struct {
	TimeMs       int     `json:"timeMs"`
	Outputs      int     `json:"outputs"`
	TotalAcc     float64 `json:"totalAccuracy"`
	Accuracy     float64 `json:"accuracy"`
	FreqSwitches int     `json:"freqSwitches"`
	InferMs      float64 `json:"inferMs"`
	TrainMs      float64 `json:"trainMs"`
	MaxLatencyMs float64 `json:"maxLatencyMs"`
}

type CellResult struct {
	Key              string             `json:"key"`
	Label            string             `json:"label"`
	Arch             string             `json:"arch"`
	Mode             string             `json:"mode"`
	DType            string             `json:"dtype"`
	Quant            string             `json:"quant"`
	Backend          string             `json:"backend"`
	Passed           bool               `json:"passed"`
	Error            string             `json:"error,omitempty"`
	Windows          []TimeWindow       `json:"windows,omitempty"`
	TotalOutputs     int                `json:"totalOutputs"`
	TotalAttempts    int                `json:"totalAttempts"`
	TotalFreqSwitch  int                `json:"totalFreqSwitches"`
	TrainTimeSec     float64            `json:"trainTimeSec"`
	InferMs          float64            `json:"inferMs"`
	TrainMs          float64            `json:"trainMs"`
	AvgTrainAccuracy float64            `json:"avgTrainAccuracy"`
	AdaptPct         float64            `json:"adaptPct"`
	Stability        float64            `json:"stability"`
	Consistency      float64            `json:"consistency"`
	Availability     float64            `json:"availability"`
	ZeroDowntime     float64            `json:"zeroDowntime"`
	ThroughputPerSec float64            `json:"throughputPerSec"`
	Score            float64            `json:"score"`
	WeightBytes      int64              `json:"weightBytes"`
	HeapBytes        int64              `json:"heapBytes"`
	WeightMiB        float64            `json:"weightMiB"`
	HeapMiB          float64            `json:"heapMiB"`
	MobileScore      float64            `json:"mobileScore"`
	Lucy             lucy.Snapshot      `json:"lucy,omitempty"`
}

type SweepResults struct {
	Timestamp    string       `json:"timestamp"`
	Duration     string       `json:"cellDuration"`
	Workers      int          `json:"workers"`
	Total        int          `json:"total"`
	Passed       int          `json:"passed"`
	Failed       int          `json:"failed"`
	Skipped      int          `json:"skipped"`
	ScoreFormula string       `json:"scoreFormula"`
	Engine       string       `json:"engine"`
	Results      []CellResult `json:"results"`
}

func main() {
	workers := flag.Int("workers", 1, "concurrent cells (1 = comparable Score; raise for faster smoke)")
	dur := flag.Duration("duration", 2*time.Second, "per-cell run duration")
	switchEvery := flag.Duration("switch", 500*time.Millisecond, "frequency switch interval")
	window := flag.Duration("window", 50*time.Millisecond, "accuracy window")
	adaptN := flag.Int("adapt-windows", AdaptWindows, "windows after switch for AdaptPct")
	resultsDir := flag.String("outdir", "results", "per-cell JSON cache dir")
	smoke := flag.Bool("smoke", false, "tiny subset instead of full matrix")
	summaryOut := flag.String("summary", "perm_summary.json", "aggregate summary path")
	flag.Parse()

	if *workers < 1 {
		*workers = 1
	}
	if *adaptN <= 0 {
		*adaptN = lucy.AdaptWindowsDefault
	}
	adaptWindows := *adaptN

	rand.Seed(time.Now().UnixNano())

	testDur := *dur
	switchIv := *switchEvery
	winDur := *window

	freqs := []float64{1, 2, 3, 4}
	allIn := make([][][]float32, len(freqs))
	allTg := make([][]float32, len(freqs))
	for i, f := range freqs {
		allIn[i], allTg[i] = createSamples(generateSine(f))
	}

	cells := buildMatrix(*smoke)
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  TEST 41-W NATIVE CAM PERM — dtype×quant×mode×arch @ SIMD                  ║")
	fmt.Println("║  Dense | Bicameral(n=2) | Tricameral(n=3) | Quadcameral(n=4) Hemispheres   ║")
	fmt.Println("║  Lucy: Score = T × Availability × SoftAcc / 10000  + WeightBytes           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Cells: %d | workers=%d | duration=%s | switch=%s | adapt=%d | smoke=%v | duty=%s\n",
		len(cells), *workers, testDur, switchIv, adaptWindows, *smoke, dutyClockName())
	fmt.Printf("Cache: %s/ | summary: %s\n\n", *resultsDir, *summaryOut)

	_ = os.MkdirAll(*resultsDir, 0o755)

	summary := &SweepResults{
		Timestamp:    time.Now().Format(time.RFC3339),
		Duration:     testDur.String(),
		Workers:      *workers,
		Total:        len(cells),
		ScoreFormula: "Throughput × Availability × SoftAcc / 10000",
		Engine:       "welvet-native-cam-perm/simd",
		Results:      make([]CellResult, 0, len(cells)),
	}

	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		sem   = make(chan struct{}, *workers)
		doneN atomic.Int64
		passN atomic.Int64
		failN atomic.Int64
		skipN atomic.Int64
	)

	for _, spec := range cells {
		wg.Add(1)
		go func(spec CellSpec) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			key := spec.Key()
			path := filepath.Join(*resultsDir, "cell_"+key+".json")
			if r, ok := loadCell(path); ok && r.Passed {
				skipN.Add(1)
				doneN.Add(1)
				mu.Lock()
				summary.Results = append(summary.Results, r)
				mu.Unlock()
				fmt.Printf("⏭️  [%d/%d] %s (cached score=%.0f w=%.1fKiB)\n",
					doneN.Load(), len(cells), r.Label, r.Score, float64(r.WeightBytes)/1024)
				return
			}

			r := runCell(spec, allIn, allTg, freqs, testDur, switchIv, winDur, adaptWindows)
			_ = saveCell(path, r)
			doneN.Add(1)
			if r.Passed {
				passN.Add(1)
			} else {
				failN.Add(1)
			}
			mu.Lock()
			summary.Results = append(summary.Results, r)
			mu.Unlock()

			status := "✅"
			if !r.Passed {
				status = "❌"
			}
			errBit := ""
			if r.Error != "" {
				errBit = " | " + truncate(r.Error, 80)
			}
			fmt.Printf("%s [%d/%d] %-52s Acc:%5.1f%% Avail:%5.1f%% Score:%7.0f RAM:%6.1fKiB%s\n",
				status, doneN.Load(), len(cells), r.Label,
				r.AvgTrainAccuracy, r.Availability, r.Score,
				float64(r.WeightBytes)/1024, errBit)
		}(spec)
	}
	wg.Wait()

	summary.Passed = int(passN.Load())
	summary.Failed = int(failN.Load())
	summary.Skipped = int(skipN.Load())
	writeSummary(*summaryOut, summary)
	printTop(summary)
}

func buildMatrix(smoke bool) []CellSpec {
	modesSeq := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	modesMesh := []TrainingMode{ModeMeshBP, ModeMeshTween, ModeMeshTweenChain}
	arches := []ArchKind{ArchDense, ArchBicameral, ArchTricameral, ArchQuadcameral}

	dtypes := append([]core.DType(nil), core.AllDTypes...)
	quants := append([]quant.Format(nil), quant.AllFormats...)

	if smoke {
		dtypes = []core.DType{core.DTypeFloat32, core.DTypeFloat16, core.DTypeInt8}
		quants = []quant.Format{quant.FormatNone, quant.FormatQ8_0, quant.FormatQ4_K}
		modesSeq = []TrainingMode{ModeStepBP, ModeStepTweenChain}
		modesMesh = []TrainingMode{ModeMeshBP}
		arches = []ArchKind{ArchDense, ArchBicameral, ArchTricameral}
	}

	var out []CellSpec
	for _, dt := range dtypes {
		for _, q := range quants {
			for _, arch := range arches {
				for _, m := range modesSeq {
					out = append(out, CellSpec{Arch: arch, Mode: m, DType: dt, Quant: q, Backend: core.BackendSIMD})
				}
			}
			for _, m := range modesMesh {
				out = append(out, CellSpec{Arch: ArchDense, Mode: m, DType: dt, Quant: q, Backend: core.BackendSIMD})
			}
		}
	}
	return out
}

func generateSine(freq float64) []float64 {
	d := make([]float64, SinePoints)
	for i := range d {
		d[i] = math.Sin(freq * float64(i) * SineResolution)
	}
	return d
}

func createSamples(data []float64) (ins [][]float32, tgs []float32) {
	n := len(data) - InputSize
	ins = make([][]float32, n)
	tgs = make([]float32, n)
	for i := 0; i < n; i++ {
		in := make([]float32, InputSize)
		for j := 0; j < InputSize; j++ {
			in[j] = float32((data[i+j] + 1) / 2)
		}
		ins[i] = in
		tgs[i] = float32((data[i+InputSize] + 1) / 2)
	}
	return ins, tgs
}

func randW(out, in int, scale float32) []float32 {
	w := make([]float32, out*in)
	for i := range w {
		w[i] = (rand.Float32()*2 - 1) * scale
	}
	return w
}

func floorBudgets[T core.Numeric](ts *tween.State[T]) {
	if ts == nil {
		return
	}
	for i := range ts.LinkBudgets {
		if ts.LinkBudgets[i] < TweenBudgetFloor {
			ts.LinkBudgets[i] = TweenBudgetFloor
		}
	}
}

func storeBytes(s *weights.Store) int64 {
	if s == nil {
		return 0
	}
	n := int64(len(s.Bias) * 8)
	if s.Packed != nil {
		n += int64(len(s.Packed.Raw))
		n += int64(len(s.Packed.Scales) * 4)
		n += int64(len(s.Packed.Mins) * 4)
		n += int64(len(s.Packed.Meta))
		n += int64(len(s.Packed.Q4Packed) * 4)
		n += int64(len(s.Packed.Int8QS))
		n += int64(len(s.Packed.F32Cache) * 4)
		return n
	}
	if len(s.Native) > 0 {
		return n + int64(len(s.Native))
	}
	bits := s.DType.Bits()
	if bits <= 0 {
		bits = 32
	}
	return n + int64((s.Rows*s.Cols*bits+7)/8)
}

var heapMu sync.Mutex

func modelWeightBytes(g *architecture.Grid) int64 {
	var n int64
	for i := range g.Cells {
		for _, s := range dna.CollectStores(g.Cells[i].Op) {
			n += storeBytes(s)
		}
	}
	return n
}

func heapNow() uint64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

func createNet(spec CellSpec) (*architecture.Grid, error) {
	if isMesh(spec.Mode) {
		if spec.Arch != ArchDense {
			return nil, fmt.Errorf("mesh modes require Dense arch")
		}
		return createMesh(spec)
	}
	if isCameral(spec.Arch) {
		return createNativeCameral(spec)
	}
	return createDense(spec)
}

func stampSIMD(g *architecture.Grid) {
	g.Exec.Backend = core.BackendSIMD
	g.Exec.MultiCore = true
	g.Exec.TileSize = 32
	for i := range g.Cells {
		if dl, ok := g.Cells[i].Op.(*dense.Layer); ok && dl != nil {
			dl.Exec.Backend = core.BackendSIMD
			dl.Exec.MultiCore = true
		}
		if pl, ok := g.Cells[i].Op.(*parallel.Layer); ok && pl != nil {
			pl.Exec.Backend = core.BackendSIMD
			pl.SyncBranchExec()
		}
	}
}

func placeDense(g *architecture.Grid, l, in, out int, act core.ActivationType, dt core.DType, qf quant.Format) error {
	layer, err := dense.NewConfigured[float32](in, out, act, dt, qf, randW(out, in, InitScale))
	if err != nil {
		return err
	}
	layer.Exec.Backend = core.BackendSIMD
	return dense.Place(g, 0, 0, 0, l, layer)
}

func createDense(spec CellSpec) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 3)
	stampSIMD(g)
	specs := []struct {
		in, out int
		act     core.ActivationType
	}{
		{InputSize, HiddenSize, core.ActivationLeakyReLU},
		{HiddenSize, HiddenSize, core.ActivationLeakyReLU},
		{HiddenSize, OutputSize, core.ActivationSigmoid},
	}
	for i, s := range specs {
		if err := placeDense(g, i, s.in, s.out, s.act, spec.DType, spec.Quant); err != nil {
			return nil, err
		}
	}
	stampSIMD(g)
	return g, nil
}

func createNativeCameral(spec CellSpec) (*architecture.Grid, error) {
	nHemi := archHemiCount(spec.Arch)
	if nHemi < 2 {
		return nil, fmt.Errorf("cameral needs n≥2, got %d", nHemi)
	}
	g := architecture.NewGrid(1, 1, 1, 3)
	stampSIMD(g)
	if err := placeDense(g, 0, InputSize, HiddenSize, core.ActivationLeakyReLU, spec.DType, spec.Quant); err != nil {
		return nil, err
	}

	hemi, err := parallel.Hemispheres(HiddenSize, HiddenSize, nHemi, parallel.CombineAdd,
		core.ActivationLeakyReLU, spec.DType, spec.Quant)
	if err != nil {
		return nil, err
	}
	for i := 0; i < nHemi; i++ {
		br, ok := hemi.DenseBranch(i)
		if !ok || br == nil || br.Weights == nil {
			return nil, fmt.Errorf("hemisphere %d not Dense", i)
		}
		if err := br.Weights.SetFromF32(randW(HiddenSize, HiddenSize, InitScale)); err != nil {
			return nil, err
		}
	}
	modes := parallel.AllConcreteTrainModes()
	branchModes := make([]parallel.TrainMode, nHemi)
	for i := 0; i < nHemi; i++ {
		branchModes[i] = modes[i%len(modes)]
	}
	hemi.SetBranchModes(branchModes...)
	hemi.Exec = g.Exec
	hemi.SyncBranchExec()
	if err := parallel.Place(g, 0, 0, 0, 1, hemi); err != nil {
		return nil, err
	}
	if err := placeDense(g, 2, HiddenSize, OutputSize, core.ActivationSigmoid, spec.DType, spec.Quant); err != nil {
		return nil, err
	}
	stampSIMD(g)
	return g, nil
}

func createMesh(spec CellSpec) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, NumLayers)
	stampSIMD(g)
	for i := 0; i < NumLayers; i++ {
		act := core.ActivationLeakyReLU
		if i == NumLayers-1 {
			act = core.ActivationSigmoid
		}
		if err := placeDense(g, i, MeshWidth, MeshWidth, act, spec.DType, spec.Quant); err != nil {
			return nil, err
		}
	}
	stampSIMD(g)
	return g, nil
}

func inputTensor(raw []float32, mesh bool) *core.Tensor[float32] {
	if !mesh {
		t := core.NewTensor[float32](1, len(raw))
		copy(t.Data, raw)
		return t
	}
	t := core.NewTensor[float32](1, MeshWidth)
	n := len(raw)
	if n > MeshWidth {
		n = MeshWidth
	}
	copy(t.Data[:n], raw[:n])
	return t
}

func targetTensor(v float32, mesh bool) *core.Tensor[float32] {
	if !mesh {
		t := core.NewTensor[float32](1, 1)
		t.Data[0] = v
		return t
	}
	t := core.NewTensor[float32](1, MeshWidth)
	t.Data[0] = v
	return t
}

func runCell(spec CellSpec, allIn [][][]float32, allTg [][]float32, freqs []float64,
	testDur, switchIv, winDur time.Duration, adaptWindows int) (out CellResult) {

	out = CellResult{
		Key:     spec.Key(),
		Label:   spec.Label(),
		Arch:    archNames[spec.Arch],
		Mode:    modeNames[spec.Mode],
		DType:   spec.DType.String(),
		Quant:   spec.Quant.String(),
		Backend: "simd",
	}

	defer func() {
		if rec := recover(); rec != nil {
			out.Passed = false
			out.Error = fmt.Sprintf("panic: %v", rec)
		}
	}()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	heapMu.Lock()
	before := heapNow()
	g, err := createNet(spec)
	after := heapNow()
	heapMu.Unlock()
	if err != nil {
		out.Error = err.Error()
		return out
	}
	if after >= before {
		out.HeapBytes = int64(after - before)
	}
	out.WeightBytes = modelWeightBytes(g)
	out.WeightMiB = float64(out.WeightBytes) / (1024 * 1024)
	out.HeapMiB = float64(out.HeapBytes) / (1024 * 1024)
	out.Lucy.WeightBytes = out.WeightBytes
	out.Lucy.HeapBytes = out.HeapBytes

	mesh := isMesh(spec.Mode)
	numWindows := int(testDur / winDur)
	if numWindows < 1 {
		numWindows = 1
	}
	out.Windows = make([]TimeWindow, numWindows)
	for i := range out.Windows {
		out.Windows[i].TimeMs = (i + 1) * int(winDur.Milliseconds())
	}

	numLayers := g.StackLayerCount()
	var meshState *step.State[float32]
	if mesh {
		meshState = step.New[float32](g)
	}
	var ts *tween.State[float32]
	if isTween(spec.Mode) && !mesh {
		cfg := tween.DefaultConfig()
		cfg.UseChainRule = useChain(spec.Mode)
		cfg.LearningRate = float32(LearningRate)
		ts = tween.NewState[float32](g, cfg)
	}

	type samp struct {
		In []float32
		Tg float32
	}
	batch := make([]samp, 0, 16)
	lastTrain := time.Now()
	start := time.Now()
	curWin, sampleIdx, freqIdx := 0, 0, 0
	lastSwitch := start
	lastOut := start
	var totalInfer, totalTrain time.Duration

	for time.Since(start) < testDur {
		elapsed := time.Since(start)
		nw := int(elapsed / winDur)
		if nw > curWin && nw < numWindows {
			curWin = nw
		}
		if time.Since(lastSwitch) >= switchIv && freqIdx < len(freqs)-1 {
			freqIdx++
			lastSwitch = time.Now()
			out.TotalFreqSwitch++
			if curWin < numWindows {
				out.Windows[curWin].FreqSwitches++
			}
		}

		in := allIn[freqIdx][sampleIdx%len(allIn[freqIdx])]
		tg := allTg[freqIdx][sampleIdx%len(allTg[freqIdx])]
		sampleIdx++
		out.TotalAttempts++

		x := inputTensor(in, mesh)
		y := targetTensor(tg, mesh)
		var output []float32
		var fwd *forward.Result[float32]

		t0 := startWork()
		switch spec.Mode {
		case ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain:
			fwd, err = forward.Forward(g, x)
			if err != nil {
				out.Error = err.Error()
				return out
			}
			output = fwd.Output.Data
		case ModeStepTween, ModeStepTweenChain:
			o, e := tween.Forward(g, ts, x)
			if e != nil {
				out.Error = e.Error()
				return out
			}
			output = o.Data
		default:
			meshState.HistoryIn = meshState.HistoryIn[:0]
			meshState.HistoryPre = meshState.HistoryPre[:0]
			meshState.SetInput(x)
			for t := 0; t < numLayers; t++ {
				if _, e := step.Forward(g, meshState, true); e != nil {
					out.Error = e.Error()
					return out
				}
			}
			if meshState.LayerData[numLayers-1] == nil {
				continue
			}
			output = meshState.LayerData[numLayers-1].Data
		}
		inf := t0.elapsed()
		totalInfer += inf

		pred := float32(0)
		if len(output) > 0 {
			pred = output[0]
		}
		acc := lucy.SoftAccOne(pred, tg)
		if curWin < numWindows {
			lat := time.Since(lastOut).Seconds() * 1000
			if lat > out.Windows[curWin].MaxLatencyMs {
				out.Windows[curWin].MaxLatencyMs = lat
			}
			lastOut = time.Now()
			out.Windows[curWin].Outputs++
			out.Windows[curWin].TotalAcc += acc
			out.Windows[curWin].InferMs += inf.Seconds() * 1000
			out.TotalOutputs++
			out.Lucy.TotalOutputs++
		}

		var trainDur time.Duration
		switch spec.Mode {
		case ModeNormalBP:
			batch = append(batch, samp{in, tg})
			if time.Since(lastTrain) > TrainInterval && len(batch) > 0 {
				t1 := startWork()
				for _, s := range batch {
					f, e := forward.Forward(g, inputTensor(s.In, false))
					if e != nil {
						continue
					}
					_, _ = training.Step(f, targetTensor(s.Tg, false), LearningRate)
				}
				trainDur = t1.elapsed()
				batch = batch[:0]
				lastTrain = time.Now()
			}
		case ModeStepBP:
			if fwd != nil {
				t1 := startWork()
				_, _ = training.Step(fwd, y, LearningRate)
				trainDur = t1.elapsed()
			}
		case ModeTween, ModeTweenChain:
			batch = append(batch, samp{in, tg})
			if time.Since(lastTrain) > TrainInterval && len(batch) > 0 {
				t1 := startWork()
				for _, s := range batch {
					sx := inputTensor(s.In, false)
					sy := targetTensor(s.Tg, false)
					f, e := forward.Forward(g, sx)
					if e != nil {
						continue
					}
					tween.CaptureFromForward(ts, f, sx)
					if e := tween.Backward(g, ts, sy); e != nil {
						continue
					}
					ts.CalculateLinkBudgets()
					floorBudgets(ts)
					_ = tween.ApplyGaps(g, ts, float32(LearningRate))
				}
				trainDur = t1.elapsed()
				batch = batch[:0]
				lastTrain = time.Now()
			}
		case ModeStepTween, ModeStepTweenChain:
			t1 := startWork()
			if e := tween.Backward(g, ts, y); e == nil {
				ts.CalculateLinkBudgets()
				floorBudgets(ts)
				_ = tween.ApplyGaps(g, ts, float32(LearningRate))
			}
			trainDur = t1.elapsed()
		case ModeMeshBP:
			outT := meshState.LayerData[numLayers-1]
			if outT != nil {
				t1 := startWork()
				if grad, ge := training.MSEGrad(outT, y); ge == nil {
					if _, lg, be := step.Backward(g, meshState, grad); be == nil {
						for idx, pair := range lg {
							if pair[1] == nil || idx >= len(g.Cells) {
								continue
							}
							if dl, ok := g.Cells[idx].Op.(*dense.Layer); ok {
								_ = dense.ApplyGradSGD(dl, pair[1], LearningRate)
							}
						}
					}
				}
				trainDur = t1.elapsed()
			}
		case ModeMeshTween, ModeMeshTweenChain:
			t1 := startWork()
			if meshState.TweenState == nil {
				cfg := tween.DefaultConfig()
				cfg.UseChainRule = useChain(spec.Mode)
				cfg.LearningRate = float32(LearningRate)
				meshState.TweenState = tween.NewState[float32](g, cfg)
			}
			st := meshState.TweenState
			st.ForwardActs[0] = x
			for i := 0; i < len(meshState.LayerData) && i+1 < len(st.ForwardActs); i++ {
				st.ForwardActs[i+1] = meshState.LayerData[i]
			}
			if e := tween.Backward(g, st, y); e == nil {
				st.CalculateLinkBudgets()
				floorBudgets(st)
				_ = tween.ApplyGaps(g, st, float32(LearningRate))
			}
			trainDur = t1.elapsed()
		}
		if trainDur > 0 {
			totalTrain += trainDur
			if curWin < numWindows {
				out.Windows[curWin].TrainMs += trainDur.Seconds() * 1000
			}
		}
	}

	for i := range out.Windows {
		w := &out.Windows[i]
		if w.Outputs > 0 {
			w.Accuracy = w.TotalAcc / float64(w.Outputs)
		}
	}
	out.TrainTimeSec = time.Since(start).Seconds()
	out.InferMs = totalInfer.Seconds() * 1000
	out.TrainMs = totalTrain.Seconds() * 1000
	finalizeLucy(&out, adaptWindows)
	out.Passed = out.Error == "" && out.TotalOutputs > 0
	return out
}

func finalizeLucy(r *CellResult, adaptWindows int) {
	r.Lucy.Duration = time.Duration(r.TrainTimeSec * float64(time.Second))
	r.Lucy.InferMs = r.InferMs
	r.Lucy.TrainMs = r.TrainMs
	r.Lucy.TotalOutputs = int64(r.TotalOutputs)
	r.Lucy.WeightBytes = r.WeightBytes
	r.Lucy.HeapBytes = r.HeapBytes
	r.Lucy.Windows = r.Lucy.Windows[:0]
	for _, w := range r.Windows {
		soft := w.Accuracy
		if math.IsNaN(soft) || math.IsInf(soft, 0) {
			soft = 0
		}
		r.Lucy.Windows = append(r.Lucy.Windows, lucy.Window{
			Outputs:       int64(w.Outputs),
			InferMs:       w.InferMs,
			TrainMs:       w.TrainMs,
			PhaseSwitches: w.FreqSwitches,
			SoftAcc:       soft,
			Accuracy:      soft,
		})
	}
	lucy.Finalize(&r.Lucy, lucy.Options{
		AdaptWindows:  adaptWindows,
		ConsThreshold: lucy.ConsThreshold,
	})
	r.AvgTrainAccuracy = r.Lucy.SoftAcc
	r.AdaptPct = r.Lucy.AdaptPct
	r.Stability = r.Lucy.Stability
	r.Consistency = r.Lucy.Consistency
	r.Availability = r.Lucy.Availability
	r.ZeroDowntime = r.Lucy.ZeroDowntime
	r.ThroughputPerSec = r.Lucy.Throughput
	r.Score = r.Lucy.Score
	r.WeightMiB = r.Lucy.WeightMiB
	r.HeapMiB = r.Lucy.HeapMiB
	r.MobileScore = r.Lucy.MobileScore
}

func loadCell(path string) (CellResult, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return CellResult{}, false
	}
	var r CellResult
	if json.Unmarshal(b, &r) != nil {
		return CellResult{}, false
	}
	return r, true
}

func saveCell(path string, r CellResult) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func writeSummary(path string, s *SweepResults) {
	b, _ := json.MarshalIndent(s, "", "  ")
	_ = os.WriteFile(path, b, 0o644)
	fmt.Printf("\n✅ Summary → %s  (pass=%d fail=%d skip=%d / %d)\n",
		path, s.Passed, s.Failed, s.Skipped, s.Total)
}

func printTop(s *SweepResults) {
	var pass []CellResult
	for _, r := range s.Results {
		if r.Passed {
			pass = append(pass, r)
		}
	}
	for i := 0; i < len(pass); i++ {
		for j := i + 1; j < len(pass); j++ {
			if pass[j].Score > pass[i].Score {
				pass[i], pass[j] = pass[j], pass[i]
			}
		}
	}
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║ TOP by Lucy Score                                                                                  ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	limit := 15
	if len(pass) < limit {
		limit = len(pass)
	}
	for i := 0; i < limit; i++ {
		r := pass[i]
		fmt.Printf("║ %2d. %-56s Score:%8.0f Acc:%5.1f Avail:%5.1f RAM:%6.1fKiB ║\n",
			i+1, r.Label, r.Score, r.AvgTrainAccuracy, r.Availability, float64(r.WeightBytes)/1024)
	}
	bestMob, bestMobL := -1.0, ""
	bestRAM, bestRAML := int64(1<<62), ""
	for _, r := range s.Results {
		if !r.Passed {
			continue
		}
		if r.MobileScore > bestMob {
			bestMob, bestMobL = r.MobileScore, r.Label
		}
		if r.WeightBytes > 0 && r.WeightBytes < bestRAM {
			bestRAM, bestRAML = r.WeightBytes, r.Label
		}
	}
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Best mobile Score/MiB: %-44s  %.0f              ║\n", bestMobL, bestMob)
	fmt.Printf("║ Smallest weights:      %-44s  %d bytes         ║\n", bestRAML, bestRAM)
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

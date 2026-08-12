package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
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
// TEST 41-W NATIVE CAM: SINE WAVE ADAPTATION — native cameral Stack/Parallel API
// ═══════════════════════════════════════════════════════════════════════════════
//
// Same Lucy protocol as test41_w_sine_ada, but hemispheres are built with the
// native welvet cameral API (Hemispheres + Place), not hand-wired Parallel.
//
// Architectures (seq modes only — mesh needs equal-width Dense ticks):
//   Bicameral  — Dense in → Hemispheres(n=2, add) → Dense out
//   Tricameral — Dense in → Hemispheres(n=3, add) → Dense out
//   Quadcameral— Dense in → Hemispheres(n=4, add) → Dense out
//
// Each Parallel stamps distinct per-hemisphere TrainModes (cycling the 9
// test41 modes via SetBranchModes).
//
// Protocol defaults match test41_w_sine_ada_perm (short SIMD race):
//   2s, switch every 500ms, AdaptWindows=4, BackendSIMD, workers=1.
// Measuring math is welvet/lucy (same equations as perm/tide/live_mnist).
// Override with -duration/-switch/-adapt-windows for the old 10s sine_ada race.
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

	// Perm-aligned defaults (was 10s / 2.5s / AdaptWindows=10 on CPU tiled).
	TestDuration   = 2 * time.Second
	WindowDuration = 50 * time.Millisecond
	SwitchInterval = 500 * time.Millisecond
	TrainInterval  = 10 * time.Millisecond
	AdaptWindows   = 4 // 4×50ms = 200ms post-switch (lucy / perm default)
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
	case ArchTricameral:
		return 3
	case ArchQuadcameral:
		return 4
	case ArchBicameral:
		return 2
	default:
		return 0
	}
}

func isCameralArch(a ArchKind) bool {
	return a == ArchBicameral || a == ArchTricameral || a == ArchQuadcameral
}

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
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
	ModeMeshBP:         "MeshBP",
	ModeMeshTween:      "MeshTween",
	ModeMeshTweenChain: "MeshTweenChain",
}

func isMeshMode(m TrainingMode) bool {
	return m == ModeMeshBP || m == ModeMeshTween || m == ModeMeshTweenChain
}

func isTweenMode(m TrainingMode) bool {
	switch m {
	case ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain, ModeMeshTween, ModeMeshTweenChain:
		return true
	default:
		return false
	}
}

func useChainRule(m TrainingMode) bool {
	switch m {
	case ModeTweenChain, ModeStepTweenChain, ModeMeshTweenChain:
		return true
	default:
		return false
	}
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

type ModeResult struct {
	Label           string             `json:"label"`
	Arch            string             `json:"arch"`
	Mode            string             `json:"mode"`
	Windows         []TimeWindow       `json:"windows"`
	TotalAttempts   int                `json:"totalAttempts"`
	TotalFreqSwitch int                `json:"totalFreqSwitches"`
	ZeroOutWindows  int                `json:"zeroOutputWindows"`
	AvgLatencyMs    float64            `json:"avgLatencyMs"`
	MaxLatencyMs    float64            `json:"maxLatencyMs"`
	Lucy            lucy.Snapshot      `json:"lucy"` // SoftAcc / Score / AdaptPct / …
}

type BenchmarkResults struct {
	Modes         []string               `json:"modes"`
	Results       map[string]*ModeResult `json:"results"`
	Timestamp     string                 `json:"timestamp"`
	Duration      string                 `json:"duration"`
	SwitchMs      int                    `json:"switchMs"`
	WindowMs      int                    `json:"windowMs"`
	AdaptWindows  int                    `json:"adaptWindows"`
	Workers       int                    `json:"workers"`
	Frequencies   []float64              `json:"frequencies"`
	Engine        string                 `json:"engine"`
	SoftAccScale  float64                `json:"softAccScale"`
	ScoreFormula  string                 `json:"scoreFormula"`
}

// Protocol knobs — set from flags in main (defaults match sine_ada long race).
var (
	testDuration   = TestDuration
	windowDuration = WindowDuration
	switchInterval = SwitchInterval
	adaptWindows   = AdaptWindows
	jobWorkers     = 1 // default serial — comparable wall-Tput/Score; raise for smoke only
)

type job struct {
	arch ArchKind
	mode TrainingMode
}

func labelOf(j job) string {
	return archNames[j.arch] + "/" + modeNames[j.mode]
}

func main() {
	workers := flag.Int("workers", 1, "concurrent jobs (1 = comparable Lucy Score; N>1 speeds up but wall-Tput/Score drop under CPU share)")
	dur := flag.Duration("duration", TestDuration, "per-job run duration")
	switchEvery := flag.Duration("switch", SwitchInterval, "frequency switch interval")
	window := flag.Duration("window", WindowDuration, "SoftAcc window")
	adaptN := flag.Int("adapt-windows", AdaptWindows, "pulse windows after switch folded into AdaptPct")
	flag.Parse()

	testDuration = *dur
	windowDuration = *window
	switchInterval = *switchEvery
	adaptWindows = *adaptN
	if adaptWindows <= 0 {
		adaptWindows = lucy.AdaptWindowsDefault
	}
	jobWorkers = *workers
	if jobWorkers < 1 {
		jobWorkers = 1
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🌊 TEST 41-W NATIVE CAM: sine adaptation — Dense + bi/tri/quad Hemispheres        ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   SoftAcc + AdaptPct | Availability = InferMs/(InferMs+TrainMs) | Lucy Score        ║")
	fmt.Println("║   Cost: WeightBytes + HeapBytes → MobileScore = Score / WeightMiB                   ║")
	fmt.Println("║   Arch: Dense | Bicameral(n=2) | Tricameral(n=3) | Quadcameral(n=4)                 ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	frequencies := []float64{1.0, 2.0, 3.0, 4.0}
	allInputs := make([][][]float32, len(frequencies))
	allTargets := make([][]float32, len(frequencies))
	for i, freq := range frequencies {
		allInputs[i], allTargets[i] = createSamples(generateSineWave(freq))
	}

	jobs := buildJobs()
	fmt.Printf("\n📊 %d samples/freq | %d jobs | SoftAccScale=%.2f\n", SinePoints, len(jobs), lucy.SoftAccScaleSine)
	fmt.Printf("⏱️  %s/job | switch every %s | adapt %dms (%d×%dms) | workers=%d | duty=%s\n\n",
		testDuration, switchInterval,
		adaptWindows*int(windowDuration.Milliseconds()), adaptWindows, windowDuration.Milliseconds(),
		jobWorkers, dutyClockName())

	results := &BenchmarkResults{
		Modes:        make([]string, len(jobs)),
		Results:      make(map[string]*ModeResult),
		Timestamp:    time.Now().Format(time.RFC3339),
		Duration:     testDuration.String(),
		SwitchMs:     int(switchInterval.Milliseconds()),
		WindowMs:     int(windowDuration.Milliseconds()),
		AdaptWindows: adaptWindows,
		Workers:      jobWorkers,
		Frequencies:  frequencies,
		Engine:       "welvet-native-cam/simd",
		SoftAccScale: lucy.SoftAccScaleSine,
		ScoreFormula: "Throughput × Availability × SoftAcc / 10000",
	}
	for i, j := range jobs {
		results.Modes[i] = labelOf(j)
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, jobWorkers)
	for _, j := range jobs {
		wg.Add(1)
		go func(j job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			name := labelOf(j)
			fmt.Printf("🚀 [%s] Starting...\n", name)
			r := runBenchmark(j, allInputs, allTargets, frequencies)
			mu.Lock()
			results.Results[name] = r
			mu.Unlock()
			fmt.Printf("✅ [%s] Acc:%.1f%% Adapt:%.1f%% Avail:%.1f%% Tput:%.0f Score:%.0f RAM:%.1fKiB\n",
				name, r.Lucy.SoftAcc, r.Lucy.AdaptPct, r.Lucy.Availability, r.Lucy.Throughput, r.Lucy.Score,
				float64(r.Lucy.WeightBytes)/1024)
		}(j)
	}
	wg.Wait()
	fmt.Println("\n✅ All benchmarks complete!")
	saveResults(results)
	printTimeline(results)
	printSummary(results)
}

func buildJobs() []job {
	seq := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	mesh := []TrainingMode{ModeMeshBP, ModeMeshTween, ModeMeshTweenChain}
	arches := []ArchKind{ArchDense, ArchBicameral, ArchTricameral, ArchQuadcameral}
	var out []job
	for _, a := range arches {
		for _, m := range seq {
			out = append(out, job{a, m})
		}
	}
	// Mesh needs equal-width Dense stack only.
	for _, m := range mesh {
		out = append(out, job{ArchDense, m})
	}
	return out
}

func generateSineWave(freq float64) []float64 {
	data := make([]float64, SinePoints)
	for i := range data {
		data[i] = math.Sin(freq * float64(i) * SineResolution)
	}
	return data
}

func createSamples(data []float64) (inputs [][]float32, targets []float32) {
	n := len(data) - InputSize
	inputs = make([][]float32, n)
	targets = make([]float32, n)
	for i := 0; i < n; i++ {
		in := make([]float32, InputSize)
		for j := 0; j < InputSize; j++ {
			in[j] = float32((data[i+j] + 1) / 2)
		}
		inputs[i] = in
		targets[i] = float32((data[i+InputSize] + 1) / 2)
	}
	return inputs, targets
}

func randWeights(out, in int, scale float32) []float32 {
	w := make([]float32, out*in)
	for i := range w {
		w[i] = (rand.Float32()*2 - 1) * scale
	}
	return w
}

func createNetwork(arch ArchKind, mesh bool) (*architecture.Grid, error) {
	var (
		g   *architecture.Grid
		err error
	)
	if mesh {
		if arch != ArchDense {
			return nil, fmt.Errorf("native_cam: mesh modes require Dense arch, got %s", archNames[arch])
		}
		g, err = createMeshNetwork()
	} else if isCameralArch(arch) {
		g, err = createNativeCameralNetwork(archHemiCount(arch))
	} else {
		g, err = createDenseNetwork()
	}
	if err != nil {
		return nil, err
	}
	stampSIMD(g) // same backend as test41_w_sine_ada_perm
	return g, nil
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

func createDenseNetwork() (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 3)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = false
	g.Exec.TileSize = 32
	specs := []struct {
		in, out int
		act     core.ActivationType
	}{
		{InputSize, HiddenSize, core.ActivationLeakyReLU},
		{HiddenSize, HiddenSize, core.ActivationLeakyReLU},
		{HiddenSize, OutputSize, core.ActivationSigmoid},
	}
	for i, s := range specs {
		l, err := dense.NewConfigured[float32](s.in, s.out, s.act, core.DTypeFloat32, quant.FormatNone, randWeights(s.out, s.in, InitScale))
		if err != nil {
			return nil, err
		}
		if err := dense.Place(g, 0, 0, 0, i, l); err != nil {
			return nil, err
		}
	}
	return g, nil
}

func createMeshNetwork() (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, NumLayers)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = false
	g.Exec.TileSize = 32
	for i := 0; i < NumLayers; i++ {
		act := core.ActivationLeakyReLU
		if i == NumLayers-1 {
			act = core.ActivationSigmoid
		}
		l, err := dense.NewConfigured[float32](MeshWidth, MeshWidth, act, core.DTypeFloat32, quant.FormatNone, randWeights(MeshWidth, MeshWidth, InitScale))
		if err != nil {
			return nil, err
		}
		if err := dense.Place(g, 0, 0, 0, i, l); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// createNativeCameralNetwork builds Dense in → Hemispheres(n, add) → Dense out
// via the native cameral API, with distinct per-hemisphere TrainModes.
func createNativeCameralNetwork(nHemi int) (*architecture.Grid, error) {
	if nHemi < 2 {
		return nil, fmt.Errorf("native_cam: need nHemi≥2, got %d", nHemi)
	}
	g := architecture.NewGrid(1, 1, 1, 3)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = false
	g.Exec.TileSize = 32

	in, err := dense.NewConfigured[float32](InputSize, HiddenSize, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, randWeights(HiddenSize, InputSize, InitScale))
	if err != nil {
		return nil, err
	}
	if err := dense.Place(g, 0, 0, 0, 0, in); err != nil {
		return nil, err
	}

	hemi, err := parallel.Hemispheres(HiddenSize, HiddenSize, nHemi, parallel.CombineAdd, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return nil, err
	}
	// Hemispheres starts at zeros — seed like the old hand-wired bicameral.
	for i := 0; i < nHemi; i++ {
		br, ok := hemi.DenseBranch(i)
		if !ok || br == nil || br.Weights == nil {
			return nil, fmt.Errorf("native_cam: hemisphere %d not Dense", i)
		}
		if err := br.Weights.SetFromF32(randWeights(HiddenSize, HiddenSize, InitScale)); err != nil {
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

	out, err := dense.NewConfigured[float32](HiddenSize, OutputSize, core.ActivationSigmoid, core.DTypeFloat32, quant.FormatNone, randWeights(OutputSize, HiddenSize, InitScale))
	if err != nil {
		return nil, err
	}
	if err := dense.Place(g, 0, 0, 0, 2, out); err != nil {
		return nil, err
	}
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

func predFrom(out []float32) float32 {
	if len(out) == 0 {
		return 0
	}
	return out[0]
}

func floorTweenBudgets[T core.Numeric](ts *tween.State[T]) {
	if ts == nil {
		return
	}
	for i := range ts.LinkBudgets {
		if ts.LinkBudgets[i] < TweenBudgetFloor {
			ts.LinkBudgets[i] = TweenBudgetFloor
		}
	}
}

// heapMu serializes GC+alloc sampling across concurrent jobs (WeightBytes is lock-free).
var heapMu sync.Mutex

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

func runBenchmark(j job, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) *ModeResult {
	// Pin work + duty-cycle measure to one OS thread (RUSAGE_THREAD needs this).
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	mesh := isMeshMode(j.mode)
	numWindows := int(testDuration / windowDuration)
	result := &ModeResult{
		Label:   labelOf(j),
		Arch:    archNames[j.arch],
		Mode:    modeNames[j.mode],
		Windows: make([]TimeWindow, numWindows),
	}
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(windowDuration.Milliseconds())
	}

	// Same cost sampling as perm / tide: weight stores + heap delta around build.
	heapMu.Lock()
	before := heapNow()
	g, err := createNetwork(j.arch, mesh)
	after := heapNow()
	heapMu.Unlock()
	if err != nil {
		fmt.Printf("❌ [%s] createNetwork: %v\n", result.Label, err)
		return result
	}
	result.Lucy.HeapBytes = int64(after - before)
	if result.Lucy.HeapBytes < 0 {
		result.Lucy.HeapBytes = 0
	}
	result.Lucy.WeightBytes = modelWeightBytes(g)
	numLayers := g.StackLayerCount()

	var meshState *step.State[float32]
	if mesh {
		meshState = step.New[float32](g)
	}
	var ts *tween.State[float32]
	if isTweenMode(j.mode) && !mesh {
		cfg := tween.DefaultConfig()
		cfg.UseChainRule = useChainRule(j.mode)
		cfg.LearningRate = float32(LearningRate)
		ts = tween.NewState[float32](g, cfg)
	}

	type sample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]sample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	currentFreqIdx := 0
	lastSwitchTime := start
	lastOutputTime := start
	var totalInfer, totalTrain time.Duration

	for time.Since(start) < testDuration {
		elapsed := time.Since(start)
		newWindow := int(elapsed / windowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}
		if time.Since(lastSwitchTime) >= switchInterval && currentFreqIdx < len(frequencies)-1 {
			currentFreqIdx++
			lastSwitchTime = time.Now()
			result.TotalFreqSwitch++
			if currentWindow < numWindows {
				result.Windows[currentWindow].FreqSwitches++
			}
		}

		input := allInputs[currentFreqIdx][sampleIdx%len(allInputs[currentFreqIdx])]
		target := allTargets[currentFreqIdx][sampleIdx%len(allTargets[currentFreqIdx])]
		sampleIdx++
		result.TotalAttempts++

		x := inputTensor(input, mesh)
		y := targetTensor(target, mesh)

		var output []float32
		var fwd *forward.Result[float32]

		tInf := startWork()
		switch j.mode {
		case ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain:
			fwd, err = forward.Forward(g, x)
			if err != nil {
				continue
			}
			output = fwd.Output.Data
		case ModeStepTween, ModeStepTweenChain:
			out, e := tween.Forward(g, ts, x)
			if e != nil {
				continue
			}
			output = out.Data
		case ModeMeshBP, ModeMeshTween, ModeMeshTweenChain:
			out, ok := meshForward(g, meshState, x, numLayers)
			if !ok {
				continue
			}
			output = out
		}
		inferDur := tInf.elapsed()
		totalInfer += inferDur

		sampleAcc := lucy.SoftAccOne(predFrom(output), target)
		if currentWindow < numWindows {
			lat := time.Since(lastOutputTime).Seconds() * 1000
			if lat > result.Windows[currentWindow].MaxLatencyMs {
				result.Windows[currentWindow].MaxLatencyMs = lat
			}
			lastOutputTime = time.Now()
			result.Windows[currentWindow].Outputs++
			result.Windows[currentWindow].TotalAcc += sampleAcc
			result.Windows[currentWindow].InferMs += inferDur.Seconds() * 1000
			result.Lucy.TotalOutputs++
		}

		var trainDur time.Duration
		switch j.mode {
		case ModeNormalBP:
			trainBatch = append(trainBatch, sample{input, target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				t0 := startWork()
				for _, s := range trainBatch {
					sx := inputTensor(s.Input, false)
					sy := targetTensor(s.Target, false)
					f, ferr := forward.Forward(g, sx)
					if ferr != nil {
						continue
					}
					_, _ = training.Step(f, sy, LearningRate)
				}
				trainDur = t0.elapsed()
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case ModeStepBP:
			if fwd != nil {
				t0 := startWork()
				_, _ = training.Step(fwd, y, LearningRate)
				trainDur = t0.elapsed()
			}
		case ModeTween, ModeTweenChain:
			trainBatch = append(trainBatch, sample{input, target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				t0 := startWork()
				for _, s := range trainBatch {
					sx := inputTensor(s.Input, false)
					sy := targetTensor(s.Target, false)
					f, ferr := forward.Forward(g, sx)
					if ferr != nil {
						continue
					}
					tween.CaptureFromForward(ts, f, sx)
					if e := tween.Backward(g, ts, sy); e != nil {
						continue
					}
					ts.CalculateLinkBudgets()
					floorTweenBudgets(ts)
					_ = tween.ApplyGaps(g, ts, float32(LearningRate))
				}
				trainDur = t0.elapsed()
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case ModeStepTween, ModeStepTweenChain:
			t0 := startWork()
			if e := tween.Backward(g, ts, y); e == nil {
				ts.CalculateLinkBudgets()
				floorTweenBudgets(ts)
				_ = tween.ApplyGaps(g, ts, float32(LearningRate))
			}
			trainDur = t0.elapsed()
		case ModeMeshBP:
			outT := meshState.LayerData[numLayers-1]
			if outT != nil {
				t0 := startWork()
				if grad, gerr := training.MSEGrad(outT, y); gerr == nil {
					if _, lg, berr := step.Backward(g, meshState, grad); berr == nil {
						_ = applyStepGrads(g, lg, LearningRate)
					}
				}
				trainDur = t0.elapsed()
			}
		case ModeMeshTween, ModeMeshTweenChain:
			if !math.IsNaN(float64(predFrom(output))) {
				t0 := startWork()
				_ = meshApplyTween(g, meshState, x, y, float32(LearningRate), j.mode == ModeMeshTweenChain)
				trainDur = t0.elapsed()
			}
		}
		if trainDur > 0 {
			totalTrain += trainDur
			if currentWindow < numWindows {
				result.Windows[currentWindow].TrainMs += trainDur.Seconds() * 1000
			}
		}
	}

	for i := range result.Windows {
		w := &result.Windows[i]
		if w.Outputs > 0 {
			acc := w.TotalAcc / float64(w.Outputs)
			if math.IsNaN(acc) || math.IsInf(acc, 0) {
				acc = 0
			}
			w.Accuracy = acc
		} else {
			result.ZeroOutWindows++
		}
	}

	result.Lucy.Duration = time.Since(start)
	result.Lucy.InferMs = totalInfer.Seconds() * 1000
	result.Lucy.TrainMs = totalTrain.Seconds() * 1000
	calculateSummaryMetrics(result)
	return result
}

func meshForward(g *architecture.Grid, mesh *step.State[float32], x *core.Tensor[float32], ticks int) ([]float32, bool) {
	mesh.HistoryIn = mesh.HistoryIn[:0]
	mesh.HistoryPre = mesh.HistoryPre[:0]
	mesh.SetInput(x)
	for t := 0; t < ticks; t++ {
		if _, err := step.Forward(g, mesh, true); err != nil {
			return nil, false
		}
	}
	out := mesh.LayerData[len(mesh.LayerData)-1]
	if out == nil {
		return nil, false
	}
	return out.Data, true
}

func meshApplyTween(g *architecture.Grid, s *step.State[float32], lastInput, target *core.Tensor[float32], lr float32, chain bool) error {
	if s.TweenState == nil {
		cfg := tween.DefaultConfig()
		cfg.UseChainRule = chain
		cfg.LearningRate = lr
		s.TweenState = tween.NewState[float32](g, cfg)
	}
	if len(s.LayerData) > 0 {
		s.TweenState.ForwardActs[0] = lastInput
		for i := 0; i < len(s.LayerData) && i+1 < len(s.TweenState.ForwardActs); i++ {
			s.TweenState.ForwardActs[i+1] = s.LayerData[i]
		}
	}
	if err := tween.Backward(g, s.TweenState, target); err != nil {
		return err
	}
	s.TweenState.CalculateLinkBudgets()
	floorTweenBudgets(s.TweenState)
	return tween.ApplyGaps(g, s.TweenState, lr)
}

func applyStepGrads(g *architecture.Grid, layerGradients [][2]*core.Tensor[float32], lr float64) error {
	for idx, pair := range layerGradients {
		gW := pair[1]
		if gW == nil || idx < 0 || idx >= len(g.Cells) {
			continue
		}
		dl, ok := g.Cells[idx].Op.(*dense.Layer)
		if !ok || dl == nil {
			continue
		}
		if err := dense.ApplyGradSGD(dl, gW, lr); err != nil {
			return err
		}
	}
	return nil
}

func calculateSummaryMetrics(result *ModeResult) {
	n := len(result.Windows)
	if n == 0 {
		return
	}
	// Fold time windows into Lucy.Windows, then Finalize fills Score/Adapt/…
	result.Lucy.Windows = result.Lucy.Windows[:0]
	var latSum float64
	for _, w := range result.Windows {
		soft := w.Accuracy
		if math.IsNaN(soft) || math.IsInf(soft, 0) {
			soft = 0
		}
		result.Lucy.Windows = append(result.Lucy.Windows, lucy.Window{
			Outputs:       int64(w.Outputs),
			InferMs:       w.InferMs,
			TrainMs:       w.TrainMs,
			PhaseSwitches: w.FreqSwitches,
			SoftAcc:       soft,
			Accuracy:      soft,
		})
		latSum += w.MaxLatencyMs
		if w.MaxLatencyMs > result.MaxLatencyMs {
			result.MaxLatencyMs = w.MaxLatencyMs
		}
	}
	lucy.Finalize(&result.Lucy, lucy.Options{
		AdaptWindows:  adaptWindows,
		ConsThreshold: lucy.ConsThreshold,
	})
	result.AvgLatencyMs = latSum / float64(n)
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	_ = os.WriteFile("test41_w_native_cam_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test41_w_native_cam_results.json")
}

func printTimeline(results *BenchmarkResults) {
	secs := int(testDuration / time.Second)
	if secs < 1 {
		secs = 1
	}
	winPerSec := int(time.Second / windowDuration)
	if winPerSec < 1 {
		winPerSec = 1
	}
	switchMarks := ""
	for t := switchInterval; t < testDuration; t += switchInterval {
		if switchMarks != "" {
			switchMarks += " / "
		}
		switchMarks += fmt.Sprintf("%.1fs", t.Seconds())
	}
	if switchMarks == "" {
		switchMarks = "(none)"
	}

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║  SoftAcc %% (1s blocks) — switches at %s                                                                                      ║\n", switchMarks)
	fmt.Println("╠════════════════════════════╦══════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════╣")
	fmt.Printf("║ Label                      ║")
	for i := 0; i < secs; i++ {
		fmt.Printf(" %ds ", i+1)
	}
	fmt.Printf("║ Avg   ║ Score  ║\n")
	fmt.Println("╠════════════════════════════╬══════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════╣")
	for _, name := range results.Modes {
		r := results.Results[name]
		fmt.Printf("║ %-26s ║", name)
		for sec := 0; sec < secs; sec++ {
			avg, c := 0.0, 0
			for w := sec * winPerSec; w < (sec+1)*winPerSec && w < len(r.Windows); w++ {
				avg += r.Windows[w].Accuracy
				c++
			}
			if c > 0 {
				avg /= float64(c)
			}
			fmt.Printf(" %2.0f%%", avg)
		}
		fmt.Printf(" ║ %3.0f%% ║ %6.0f ║\n", r.Lucy.SoftAcc, r.Lucy.Score)
	}
	fmt.Println("╚════════════════════════════╩══════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════╝")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  SUMMARY — Score = T × Availability × SoftAcc / 10_000   |   MobileScore = Score / WeightMiB                                                           ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Label                      │ Acc    │ Adapt  │ Avail  │ Stab   │ Cons   │ Tput     │ Score    │ RAM KiB │ Mobile  │ InferMs │ TrainMs ║")
	fmt.Println("║  ───────────────────────────┼────────┼────────┼────────┼────────┼────────┼──────────┼──────────┼─────────┼─────────┼─────────┼─────────║")

	best, bestName := -1.0, ""
	bestMob, bestMobName := -1.0, ""
	bestRAM, bestRAMName := int64(1<<62), ""
	for _, name := range results.Modes {
		r := results.Results[name]
		fmt.Printf("║  %-26s │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %8.0f │ %8.0f │ %7.1f │ %7.0f │ %7.0f │ %7.0f ║\n",
			name, r.Lucy.SoftAcc, r.Lucy.AdaptPct, r.Lucy.Availability, r.Lucy.Stability, r.Lucy.Consistency,
			r.Lucy.Throughput, r.Lucy.Score, float64(r.Lucy.WeightBytes)/1024, r.Lucy.MobileScore,
			r.Lucy.InferMs, r.Lucy.TrainMs)
		if r.Lucy.Score > best {
			best, bestName = r.Lucy.Score, name
		}
		if r.Lucy.MobileScore > bestMob {
			bestMob, bestMobName = r.Lucy.MobileScore, name
		}
		if r.Lucy.WeightBytes > 0 && r.Lucy.WeightBytes < bestRAM {
			bestRAM, bestRAMName = r.Lucy.WeightBytes, name
		}
	}
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  🏆 WINNER (Score):        %-26s  Score=%.0f                                                                          ║\n", bestName, best)
	fmt.Printf("║  📱 Best mobile Score/MiB: %-26s  %.0f                                                                                ║\n", bestMobName, bestMob)
	fmt.Printf("║  💾 Smallest weights:      %-26s  %.1f KiB                                                                            ║\n", bestRAMName, float64(bestRAM)/1024)
	fmt.Printf("║  Availability = InferMs/(InferMs+TrainMs)×100 — duty clock = %s; default -workers 1 for comparable Score.                    ║\n", dutyClockName())
	fmt.Printf("║  AdaptPct = mean SoftAcc in %dms after each frequency switch (%d×%dms windows).                                                                  ║\n",
		adaptWindows*int(windowDuration.Milliseconds()), adaptWindows, windowDuration.Milliseconds())
	fmt.Println("║  WeightBytes = dna.CollectStores (same as perm); MobileScore = Score / WeightMiB (same as tide / live_mnist).                                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}

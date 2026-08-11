package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/tween"
)

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 41-W: SINE WAVE ADAPTATION (welvet) — measuring from legacy all_sine_wave.go
// ═══════════════════════════════════════════════════════════════════════════════
//
// Protocol: 10s, freq switch every 2.5s Sin(1x)→(2x)→(3x)→(4x), 50ms windows.
// Target: adapt after each switch (<500ms ideal).
//
// Measuring (legacy all_sine_wave.go / Lucy — not fudged):
//   SoftAcc     = 100×(1 − |err|/SoftAccScale)   // continuous adaptation quality
//   Availability = InferMs / (InferMs+TrainMs) × 100
//                 True duty-cycle: time producing preds vs time training (ALL modes).
//   AdaptPct    = mean SoftAcc in 500ms after each freq switch
//   Score       = Throughput × Availability × SoftAcc / 10_000   // Lucy equation
//
// Architectures:
//   Dense      — 10→32→32→1 (mesh uses pad→32 equal-width)
//   Bicameral  — Dense in → Parallel(Dense∥Dense, add) → Dense out  (two-brain)
//

const (
	InputSize  = 10
	HiddenSize = 32
	OutputSize = 1
	MeshWidth  = 32
	NumLayers  = 3

	LearningRate     = float64(0.01)
	InitScale        = float32(0.5)
	SoftAccScale     = 0.10
	TweenBudgetFloor = float32(0.8)

	// Consistency gate used for reporting (Lucy score uses Acc directly).
	ConsistencyThreshold = 10.0 // legacy all_sine_wave.go with availability

	SinePoints     = 100
	SineResolution = 0.1

	TestDuration   = 10 * time.Second
	WindowDuration = 50 * time.Millisecond
	SwitchInterval = 2500 * time.Millisecond
	TrainInterval  = 10 * time.Millisecond

	// Post-switch adaptation window: 500ms = 10×50ms
	AdaptWindows = 10
)

type ArchKind int

const (
	ArchDense ArchKind = iota
	ArchBicameral
)

var archNames = map[ArchKind]string{
	ArchDense:     "Dense",
	ArchBicameral: "Bicameral",
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
	Label            string       `json:"label"`
	Arch             string       `json:"arch"`
	Mode             string       `json:"mode"`
	Windows          []TimeWindow `json:"windows"`
	TotalOutputs     int          `json:"totalOutputs"`
	TotalAttempts    int          `json:"totalAttempts"`
	TotalFreqSwitch  int          `json:"totalFreqSwitches"`
	TrainTimeSec     float64      `json:"trainTimeSec"`
	InferMs          float64      `json:"inferMs"`
	TrainMs          float64      `json:"trainMs"`
	AvgTrainAccuracy float64      `json:"avgTrainAccuracy"`
	AdaptPct         float64      `json:"adaptPct"` // post-switch soft accuracy
	Stability        float64      `json:"stability"`
	Consistency      float64      `json:"consistency"`
	Availability     float64      `json:"availability"` // infer/(infer+train)×100
	ZeroDowntime     float64      `json:"zeroDowntime"`
	AvgLatencyMs     float64      `json:"avgLatencyMs"`
	MaxLatencyMs     float64      `json:"maxLatencyMs"`
	ZeroOutWindows   int          `json:"zeroOutputWindows"`
	ThroughputPerSec float64      `json:"throughputPerSec"`
	Score            float64      `json:"score"`
}

type BenchmarkResults struct {
	Modes        []string               `json:"modes"`
	Results      map[string]*ModeResult `json:"results"`
	Timestamp    string                 `json:"timestamp"`
	Duration     string                 `json:"duration"`
	WindowMs     int                    `json:"windowMs"`
	Frequencies  []float64              `json:"frequencies"`
	Engine       string                 `json:"engine"`
	SoftAccScale float64                `json:"softAccScale"`
	ScoreFormula string                 `json:"scoreFormula"`
}

type job struct {
	arch ArchKind
	mode TrainingMode
}

func labelOf(j job) string {
	return archNames[j.arch] + "/" + modeNames[j.mode]
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🌊 TEST 41-W: SINE ADAPTATION (welvet) — Lucy measuring + Bicameral               ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   Sin(1x)→(2x)→(3x)→(4x) every 2.5s | SoftAcc + AdaptPct after switches            ║")
	fmt.Println("║   Availability = InferMs/(InferMs+TrainMs)×100  (true duty-cycle, all modes)       ║")
	fmt.Println("║   Score = Throughput × Availability × SoftAcc / 10_000  (legacy Lucy)              ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   Arch: Dense | Bicameral(Dense∥Dense)                                              ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	frequencies := []float64{1.0, 2.0, 3.0, 4.0}
	allInputs := make([][][]float32, len(frequencies))
	allTargets := make([][]float32, len(frequencies))
	for i, freq := range frequencies {
		allInputs[i], allTargets[i] = createSamples(generateSineWave(freq))
	}

	jobs := buildJobs()
	fmt.Printf("\n📊 %d samples/freq | %d jobs | SoftAccScale=%.2f\n", SinePoints, len(jobs), SoftAccScale)
	fmt.Printf("⏱️  %s run | switch every %s | adapt window %dms after switch\n\n",
		TestDuration, SwitchInterval, AdaptWindows*int(WindowDuration.Milliseconds()))

	results := &BenchmarkResults{
		Modes:        make([]string, len(jobs)),
		Results:      make(map[string]*ModeResult),
		Timestamp:    time.Now().Format(time.RFC3339),
		Duration:     TestDuration.String(),
		WindowMs:     int(WindowDuration.Milliseconds()),
		Frequencies:  frequencies,
		Engine:       "welvet",
		SoftAccScale: SoftAccScale,
		ScoreFormula: "Throughput × Availability × SoftAcc / 10000",
	}
	for i, j := range jobs {
		results.Modes[i] = labelOf(j)
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	for _, j := range jobs {
		wg.Add(1)
		go func(j job) {
			defer wg.Done()
			name := labelOf(j)
			fmt.Printf("🚀 [%s] Starting...\n", name)
			r := runBenchmark(j, allInputs, allTargets, frequencies)
			mu.Lock()
			results.Results[name] = r
			mu.Unlock()
			fmt.Printf("✅ [%s] Acc:%.1f%% Adapt:%.1f%% Avail:%.1f%% Tput:%.0f Score:%.0f\n",
				name, r.AvgTrainAccuracy, r.AdaptPct, r.Availability, r.ThroughputPerSec, r.Score)
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
	var out []job
	for _, m := range seq {
		out = append(out, job{ArchDense, m}, job{ArchBicameral, m})
	}
	for _, m := range mesh {
		out = append(out, job{ArchDense, m}) // mesh needs equal-width dense stack
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
	if mesh {
		return createMeshNetwork()
	}
	if arch == ArchBicameral {
		return createBicameralNetwork()
	}
	return createDenseNetwork()
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

// createBicameralNetwork: Dense in → Parallel(Dense∥Dense, add) → Dense out.
// Mirrors test27 “two brains” idea with Dense branches (sine-shaped, no seq MHA).
func createBicameralNetwork() (*architecture.Grid, error) {
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

	right, err := dense.NewConfigured[float32](HiddenSize, HiddenSize, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, randWeights(HiddenSize, HiddenSize, InitScale))
	if err != nil {
		return nil, err
	}
	left, err := dense.NewConfigured[float32](HiddenSize, HiddenSize, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, randWeights(HiddenSize, HiddenSize, InitScale))
	if err != nil {
		return nil, err
	}
	para, err := parallel.NewFromBranches(parallel.Config{
		Dim: HiddenSize, OutFeat: HiddenSize, Branches: 2, Combine: parallel.CombineAdd,
	}, []any{right, left}, nil)
	if err != nil {
		return nil, err
	}
	if err := parallel.Place(g, 0, 0, 0, 1, para); err != nil {
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

func softAccuracy(pred, target float32) float64 {
	if math.IsNaN(float64(pred)) || math.IsInf(float64(pred), 0) {
		return 0
	}
	err := math.Abs(float64(pred - target))
	acc := 100 * (1 - err/SoftAccScale)
	if acc < 0 {
		return 0
	}
	if acc > 100 {
		return 100
	}
	return acc
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

func runBenchmark(j job, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) *ModeResult {
	mesh := isMeshMode(j.mode)
	numWindows := int(TestDuration / WindowDuration)
	result := &ModeResult{
		Label:   labelOf(j),
		Arch:    archNames[j.arch],
		Mode:    modeNames[j.mode],
		Windows: make([]TimeWindow, numWindows),
	}
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	g, err := createNetwork(j.arch, mesh)
	if err != nil {
		fmt.Printf("❌ [%s] createNetwork: %v\n", result.Label, err)
		return result
	}
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

	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}
		if time.Since(lastSwitchTime) >= SwitchInterval && currentFreqIdx < len(frequencies)-1 {
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

		tInf := time.Now()
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
		inferDur := time.Since(tInf)
		totalInfer += inferDur

		sampleAcc := softAccuracy(predFrom(output), target)
		if currentWindow < numWindows {
			lat := time.Since(lastOutputTime).Seconds() * 1000
			if lat > result.Windows[currentWindow].MaxLatencyMs {
				result.Windows[currentWindow].MaxLatencyMs = lat
			}
			lastOutputTime = time.Now()
			result.Windows[currentWindow].Outputs++
			result.Windows[currentWindow].TotalAcc += sampleAcc
			result.Windows[currentWindow].InferMs += inferDur.Seconds() * 1000
			result.TotalOutputs++
		}

		var trainDur time.Duration
		switch j.mode {
		case ModeNormalBP:
			trainBatch = append(trainBatch, sample{input, target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				t0 := time.Now()
				for _, s := range trainBatch {
					sx := inputTensor(s.Input, false)
					sy := targetTensor(s.Target, false)
					f, ferr := forward.Forward(g, sx)
					if ferr != nil {
						continue
					}
					_, _ = training.Step(f, sy, LearningRate)
				}
				trainDur = time.Since(t0)
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case ModeStepBP:
			if fwd != nil {
				t0 := time.Now()
				_, _ = training.Step(fwd, y, LearningRate)
				trainDur = time.Since(t0)
			}
		case ModeTween, ModeTweenChain:
			trainBatch = append(trainBatch, sample{input, target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				t0 := time.Now()
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
				trainDur = time.Since(t0)
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case ModeStepTween, ModeStepTweenChain:
			t0 := time.Now()
			if e := tween.Backward(g, ts, y); e == nil {
				ts.CalculateLinkBudgets()
				floorTweenBudgets(ts)
				_ = tween.ApplyGaps(g, ts, float32(LearningRate))
			}
			trainDur = time.Since(t0)
		case ModeMeshBP:
			outT := meshState.LayerData[numLayers-1]
			if outT != nil {
				t0 := time.Now()
				if grad, gerr := training.MSEGrad(outT, y); gerr == nil {
					if _, lg, berr := step.Backward(g, meshState, grad); berr == nil {
						_ = applyStepGrads(g, lg, LearningRate)
					}
				}
				trainDur = time.Since(t0)
			}
		case ModeMeshTween, ModeMeshTweenChain:
			if !math.IsNaN(float64(predFrom(output))) {
				t0 := time.Now()
				_ = meshApplyTween(g, meshState, x, y, float32(LearningRate), j.mode == ModeMeshTweenChain)
				trainDur = time.Since(t0)
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

	result.TrainTimeSec = time.Since(start).Seconds()
	result.InferMs = totalInfer.Seconds() * 1000
	result.TrainMs = totalTrain.Seconds() * 1000
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
	sum := 0.0
	valid := 0
	latSum := 0.0
	for _, w := range result.Windows {
		if math.IsNaN(w.Accuracy) || math.IsInf(w.Accuracy, 0) {
			continue
		}
		sum += w.Accuracy
		valid++
		latSum += w.MaxLatencyMs
		if w.MaxLatencyMs > result.MaxLatencyMs {
			result.MaxLatencyMs = w.MaxLatencyMs
		}
	}
	if valid == 0 {
		return
	}
	result.AvgTrainAccuracy = sum / float64(valid)
	result.AvgLatencyMs = latSum / float64(n)

	variance := 0.0
	for _, w := range result.Windows {
		if math.IsNaN(w.Accuracy) {
			continue
		}
		d := w.Accuracy - result.AvgTrainAccuracy
		variance += d * d
	}
	variance /= float64(valid)
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	above := 0
	for _, w := range result.Windows {
		if w.Accuracy >= ConsistencyThreshold {
			above++
		}
	}
	result.Consistency = float64(above) / float64(n) * 100

	// Adaptation: softAcc in AdaptWindows after each switch marker.
	adaptSum := 0.0
	adaptN := 0
	for i, w := range result.Windows {
		if w.FreqSwitches == 0 {
			continue
		}
		for k := 0; k < AdaptWindows && i+k < n; k++ {
			adaptSum += result.Windows[i+k].Accuracy
			adaptN++
		}
	}
	if adaptN > 0 {
		result.AdaptPct = adaptSum / float64(adaptN)
	}

	dur := math.Max(result.TrainTimeSec, 1e-9)
	result.ThroughputPerSec = float64(result.TotalOutputs) / dur

	// True duty-cycle availability (sensitive for online AND batch).
	busy := result.InferMs + result.TrainMs
	if busy > 0 {
		result.Availability = 100 * result.InferMs / busy
	} else {
		result.Availability = 0
	}
	if result.Availability < 0 {
		result.Availability = 0
	}
	if result.Availability > 100 {
		result.Availability = 100
	}
	result.ZeroDowntime = result.AvgTrainAccuracy * result.Availability / 100

	// Lucy score from legacy all_sine_wave.go
	result.Score = (result.ThroughputPerSec * result.Availability * result.AvgTrainAccuracy) / 10000
	if math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
		result.Score = 0
	}
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	_ = os.WriteFile("test41_w_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test41_w_results.json")
}

func printTimeline(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  SoftAcc % (1s blocks) — switches at 2.5 / 5.0 / 7.5s                                                                                            ║")
	fmt.Println("╠════════════════════════════╦══════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════╣")
	fmt.Printf("║ Label                      ║")
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds ", i+1)
	}
	fmt.Printf("║ Avg   ║ Score  ║\n")
	fmt.Println("╠════════════════════════════╬══════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════╣")
	for _, name := range results.Modes {
		r := results.Results[name]
		fmt.Printf("║ %-26s ║", name)
		for sec := 0; sec < 10; sec++ {
			avg, c := 0.0, 0
			for w := sec * 20; w < (sec+1)*20 && w < len(r.Windows); w++ {
				avg += r.Windows[w].Accuracy
				c++
			}
			if c > 0 {
				avg /= float64(c)
			}
			fmt.Printf(" %2.0f%%", avg)
		}
		fmt.Printf(" ║ %3.0f%% ║ %6.0f ║\n", r.AvgTrainAccuracy, r.Score)
	}
	fmt.Println("╚════════════════════════════╩══════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════╝")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  SUMMARY — Score = T × Availability × SoftAcc / 10_000                                                                                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Label                      │ Acc    │ Adapt  │ Avail  │ Stab   │ Cons   │ Tput     │ Score    │ InferMs │ TrainMs ║")
	fmt.Println("║  ───────────────────────────┼────────┼────────┼────────┼────────┼────────┼──────────┼──────────┼─────────┼─────────║")

	best, bestName := -1.0, ""
	for _, name := range results.Modes {
		r := results.Results[name]
		fmt.Printf("║  %-26s │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %8.0f │ %8.0f │ %7.0f │ %7.0f ║\n",
			name, r.AvgTrainAccuracy, r.AdaptPct, r.Availability, r.Stability, r.Consistency,
			r.ThroughputPerSec, r.Score, r.InferMs, r.TrainMs)
		if r.Score > best {
			best, bestName = r.Score, name
		}
	}
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  🏆 WINNER: %-26s  Score=%.0f                                                                 ║\n", bestName, best)
	fmt.Println("║  Availability = InferMs/(InferMs+TrainMs)×100 — heavier trainers drop; batch pauses drop further.                                        ║")
	fmt.Println("║  AdaptPct = mean SoftAcc in 500ms after each frequency switch (adaptation speed).                                                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}

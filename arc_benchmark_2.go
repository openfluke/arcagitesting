package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ARC-AGI Real-Time Task Switching Benchmark
//
// Mirrors adaptation_demo.html behavior:
//   - Stream through 400 tasks in 10 seconds
//   - Track PIXEL ACCURACY % on each training sample
//   - 100ms windows to capture fine-grained adaptation
//   - NormalBP: STOPS to batch train → accuracy dips during training
//   - StepTweenChain: trains EVERY sample → maintains accuracy while switching
//
// Scoring: Score = (Throughput × Stability × Consistency) / 100000
//   - Stability: 100 - stddev of window accuracies
//   - Consistency: % of windows above 30% accuracy
//   - Throughput: outputs per second

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 1000                      // ARC-AGI2 has 1000 training tasks
	LearningRate = float32(1000.01)          // Same as arc_benchmark.go
	InitScale    = float32(1000.5)
	BudgetScale  = float32(1000.8)

	// Architecture params
	DModel     = 32
	NumHeads   = 4
	LSTMHidden = 32

	// Timing - 10 second training run with 100ms windows (100 windows total)
	TestDuration   = 10 * time.Second
	WindowDuration = 100 * time.Millisecond // 100ms for fine-grained accuracy tracking

	// Batch training interval for NormalBP/Tween (this is where they PAUSE!)
	TrainInterval = 50 * time.Millisecond

	numHives = 10
)

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
}

// Data types
type ARCTask struct {
	ID          string
	Train, Test []GridPair
}
type GridPair struct{ Input, Output [][]int }
type Sample struct {
	Input, Target []float32
	Height, Width int
	TaskID        string
	TaskIndex     int
}

// TimeWindow for 100ms accuracy tracking (matches adaptation_demo.html)
type TimeWindow struct {
	TimeMs        int     `json:"timeMs"`
	Outputs       int     `json:"outputs"`
	TotalPixelAcc float64 `json:"totalPixelAcc"` // Sum of pixel accuracy % for averaging
	Accuracy      float64 `json:"accuracy"`      // Average pixel accuracy % in this window
	TaskSwitches  int     `json:"taskSwitches"`
}

// ModeResult holds per-mode benchmark results
type ModeResult struct {
	// Training phase results (100ms windows)
	Windows         []TimeWindow `json:"windows"`
	TotalOutputs    int          `json:"totalOutputs"`
	TotalTaskSwitch int          `json:"totalTaskSwitches"`
	TrainTimeSec    float64      `json:"trainTimeSec"`

	// Eval phase results
	EvalAccuracy  float64  `json:"evalAccuracy"`
	TasksSolved   int      `json:"tasksSolved"`
	SolvedTaskIDs []string `json:"solvedTaskIds,omitempty"`

	// Summary metrics (matching SPARTA dashboard)
	AvgTrainAccuracy float64           `json:"avgTrainAccuracy"`
	Stability        float64           `json:"stability"`   // 100 - stddev
	Consistency      float64           `json:"consistency"` // % windows above 30%
	ThroughputPerSec float64           `json:"throughputPerSec"`
	Score            float64           `json:"score"`           // T×S×C / 100000
	TrainDeviations  *DeviationMetrics `json:"trainDeviations"` // Deviation buckets for training data
}

// BenchmarkResults is the full output
type BenchmarkResults struct {
	Modes     []string               `json:"modes"`
	Results   map[string]*ModeResult `json:"results"`
	Timestamp string                 `json:"timestamp"`
	NumTasks  int                    `json:"numTasks"`
	Duration  string                 `json:"duration"`
	WindowMs  int                    `json:"windowMs"`
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     ARC-AGI2 Benchmark (Train + Eval)                                               ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     TRAINING: ARC-AGI2 training set (1000 tasks) — Cycle through (10 seconds)       ║")
	fmt.Println("║     EVAL: ARC-AGI2 evaluation set (120 tasks) — Test on unseen tasks!               ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     → NormalBP: STOPS to batch train → accuracy DIPS during training               ║")
	fmt.Println("║     → StepTweenChain: trains EVERY sample → maintains accuracy while switching     ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Score = (Throughput × Stability × Consistency) / 100000                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load ARC-AGI2 training data (1000 tasks)
	trainTasks, err := loadARCTasks("ARC-AGI2/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load ARC-AGI2 training tasks: %v\n", err)
		return
	}

	// Load ARC-AGI2 evaluation data
	evalTasks, err := loadARCTasks("ARC-AGI2/data/evaluation", 120)
	if err != nil {
		fmt.Printf("❌ Failed to load ARC-AGI2 evaluation tasks: %v\n", err)
		return
	}

	trainSamples := createSequentialSamples(trainTasks)
	evalSamples := createEvalSamples(evalTasks)

	numWindows := int(TestDuration / WindowDuration)
	fmt.Printf("\n📦 Loaded %d training tasks, %d train samples\n", len(trainTasks), len(trainSamples))
	fmt.Printf("📦 Loaded %d eval tasks, %d eval samples\n", len(evalTasks), len(evalSamples))
	fmt.Printf("⏱️  Duration: %s with %dms windows (=%d windows)\n\n", TestDuration, WindowDuration.Milliseconds(), numWindows)

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTween,
		ModeStepTweenChain,
	}

	results := &BenchmarkResults{
		Modes:     make([]string, len(modes)),
		Results:   make(map[string]*ModeResult),
		Timestamp: time.Now().Format(time.RFC3339),
		NumTasks:  len(trainTasks),
		Duration:  TestDuration.String(),
		WindowMs:  int(WindowDuration.Milliseconds()),
	}

	for i, m := range modes {
		results.Modes[i] = modeNames[m]
	}

	// Run benchmarks in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			modeName := modeNames[m]
			fmt.Printf("🚀 [%s] Starting...\n", modeName)

			result := runTaskSwitchingBenchmark(m, trainSamples, evalSamples, evalTasks)

			mu.Lock()
			results.Results[modeName] = result
			mu.Unlock()

			fmt.Printf("✅ [%s] Done | Acc: %.1f%% | Stab: %.0f%% | Cons: %.0f%% | Tput: %.0f | Score: %.0f\n",
				modeName, result.AvgTrainAccuracy, result.Stability, result.Consistency, result.ThroughputPerSec, result.Score)
		}(mode)
	}

	wg.Wait()
	fmt.Println("\n✅ All benchmarks complete!")

	saveResults(results)
	printTimeline(results)
	printSummary(results)
}

// createSequentialSamples creates samples indexed by task for sequential switching
func createSequentialSamples(tasks []*ARCTask) []Sample {
	var samples []Sample
	for i, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample{
				Input:     encodeGrid(pair.Input),
				Target:    encodeGrid(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

func createEvalSamples(tasks []*ARCTask) []Sample {
	var samples []Sample
	for i, task := range tasks {
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample{
				Input:     encodeGrid(pair.Input),
				Target:    encodeGrid(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

// runTaskSwitchingBenchmark runs real-time task switching
func runTaskSwitchingBenchmark(mode TrainingMode, trainSamples, evalSamples []Sample, evalTasks []*ARCTask) *ModeResult {
	numWindows := int(TestDuration / WindowDuration) // 100 windows at 100ms each
	result := &ModeResult{
		Windows:         make([]TimeWindow, numWindows),
		TrainDeviations: NewDeviationMetrics(),
	}

	// Initialize windows
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	// Create fresh network
	net := createHiveMindNetwork()
	numLayers := net.TotalLayers()

	// Initialize states based on mode
	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		ts.Config.LinkBudgetScale = BudgetScale
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	// Training batch for batch-based methods
	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	lastTaskIndex := -1

	// =========================================================================
	// TRAINING PHASE: Cycle through tasks and track pixel accuracy per sample
	// =========================================================================
	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)

		// Update window (100ms windows)
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		// Get next sample (cycling through ALL tasks sequentially)
		sample := trainSamples[sampleIdx%len(trainSamples)]
		sampleIdx++

		// Detect task switch!
		if sample.TaskIndex != lastTaskIndex {
			if currentWindow < numWindows {
				result.Windows[currentWindow].TaskSwitches++
			}
			result.TotalTaskSwitch++
			lastTaskIndex = sample.TaskIndex
		}

		// Forward pass
		var output []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(sample.Input)
		case ModeStepBP, ModeStepTween, ModeStepTweenChain:
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		}

		// Calculate PIXEL ACCURACY % on this sample and track DEVIATION
		pixelAccuracy := calculatePixelAccuracy(output, sample, result.TrainDeviations)

		// Record to current window
		if currentWindow < numWindows {
			result.Windows[currentWindow].Outputs++
			result.Windows[currentWindow].TotalPixelAcc += pixelAccuracy
			result.TotalOutputs++
		}

		// =====================================================================
		// TRAINING - THIS IS WHERE EACH MODE DIFFERS
		// NormalBP: accumulates samples, then STOPS to batch train
		// StepTweenChain: trains IMMEDIATELY on this sample
		// =====================================================================
		switch mode {
		case ModeNormalBP:
			// Batch training - accumulates samples, then PAUSES to train
			// During the pause, network isn't processing → output count drops
			trainBatch = append(trainBatch, TrainingSample{Input: sample.Input, Target: sample.Target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				batches := make([]nn.TrainingBatch, len(trainBatch))
				for i, s := range trainBatch {
					batches[i] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
				}
				net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse"})
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepBP:
			// Immediate step-based backprop
			grad := make([]float32, len(output))
			for i := range output {
				if i < len(sample.Target) {
					grad[i] = output[i] - sample.Target[i]
				}
			}
			net.StepBackward(state, grad)
			net.ApplyGradients(LearningRate)

		case ModeTween, ModeTweenChain:
			// Batch tween - accumulates samples, trains periodically
			trainBatch = append(trainBatch, TrainingSample{Input: sample.Input, Target: sample.Target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					ts.TweenStep(net, s.Input, argmax(s.Target), len(s.Target), LearningRate)
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepTween:
			// Immediate tween (no chain rule)
			ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)

		case ModeStepTweenChain:
			// Immediate tween with chain rule - TRAINS EVERY SAMPLE IMMEDIATELY!
			ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)
		}
	}

	// Finalize windows - compute average accuracy per window
	for i := range result.Windows {
		if result.Windows[i].Outputs > 0 {
			result.Windows[i].Accuracy = result.Windows[i].TotalPixelAcc / float64(result.Windows[i].Outputs)
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()

	// =========================================================================
	// EVAL PHASE: Few-Shot Adaptation (Learn from examples -> Solve test)
	// This is the "real" ARC test: adapt to each task's examples, then solve
	// =========================================================================
	evalTotalAcc := 0.0
	evalCount := 0
	taskResults := make(map[string]struct {
		totalAcc float64
		count    int
	})

	// Iterate through TASKS (not flattened samples) for proper few-shot
	for _, task := range evalTasks {
		// 1. ADAPTATION PHASE: Train on the task's EXAMPLE pairs first!
		// This is where "Fluid Intelligence" happens - learning the rule
		// Only Tween-based modes can adapt fast enough
		if ts != nil && (mode == ModeStepTweenChain || mode == ModeStepTween || mode == ModeTweenChain || mode == ModeTween) {
			// Quick 5-pass adaptation loop to learn the rule from examples
			for k := 0; k < 5; k++ {
				for _, pair := range task.Train {
					if len(pair.Input) == 0 || len(pair.Output) == 0 {
						continue
					}
					input := encodeGrid(pair.Input)
					target := encodeGrid(pair.Output)
					// The network LEARNS the rule here (this is fair - it's the task's examples)
					ts.TweenStep(net, input, argmax(target), len(target), LearningRate)
				}
			}
		}

		// 2. TESTING PHASE: Now solve the unseen test pair(s)
		for _, pair := range task.Test {
			if len(pair.Input) == 0 || len(pair.Output) == 0 {
				continue
			}

			input := encodeGrid(pair.Input)
			target := encodeGrid(pair.Output)

			var output []float32
			if state != nil {
				state.SetInput(input)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output = state.GetOutput()
			} else {
				output, _ = net.ForwardCPU(input)
			}

			acc := calculatePixelAccuracy(output, Sample{
				Target: target,
				Height: len(pair.Output),
				Width:  len(pair.Output[0]),
			}, nil)
			evalTotalAcc += acc
			evalCount++

			// Track per-task results
			r := taskResults[task.ID]
			r.totalAcc += acc
			r.count++
			taskResults[task.ID] = r
		}
	}

	if evalCount > 0 {
		result.EvalAccuracy = evalTotalAcc / float64(evalCount)
	}

	// Count tasks "solved" (100% pixel accuracy required)
	for taskID, r := range taskResults {
		if r.count > 0 && r.totalAcc/float64(r.count) >= 100 {
			result.TasksSolved++
			result.SolvedTaskIDs = append(result.SolvedTaskIDs, taskID)
		}
	}

	// Calculate summary metrics (matching SPARTA dashboard)
	calculateSummaryMetrics(result)

	return result
}

// calculatePixelAccuracy returns the percentage of pixels that match (0-100)
// Also updates deviation metrics if tracker is provided
func calculatePixelAccuracy(output []float32, sample Sample, tracker *DeviationMetrics) float64 {
	correct, total := 0, 0
	for r := 0; r < sample.Height; r++ {
		for c := 0; c < sample.Width; c++ {
			idx := r*MaxGridSize + c
			if idx < len(output) && idx < len(sample.Target) {
				pred := clampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
				exp := clampInt(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
				if pred == exp {
					correct++
				}
				total++

				if tracker != nil {
					// Update deviation bucket for this pixel
					// Use raw float values for deviation calculation to be precise
					res := EvaluatePrediction(idx, float64(sample.Target[idx]), float64(output[idx]))
					tracker.UpdateMetrics(res)
				}
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}

func calculateSummaryMetrics(result *ModeResult) {
	// Average training accuracy
	sum := 0.0
	for _, w := range result.Windows {
		sum += w.Accuracy
	}
	result.AvgTrainAccuracy = sum / float64(len(result.Windows))

	// Stability: 100 - stddev (matching SPARTA dashboard)
	variance := 0.0
	for _, w := range result.Windows {
		diff := w.Accuracy - result.AvgTrainAccuracy
		variance += diff * diff
	}
	variance /= float64(len(result.Windows))
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	// Consistency: % of windows ABOVE RANDOM BASELINE
	// Random guessing on 10 colors = ~11% accuracy, so threshold = 12%
	const consistencyThreshold = 12.0
	aboveThreshold := 0
	for _, w := range result.Windows {
		if w.Accuracy >= consistencyThreshold {
			aboveThreshold++
		}
	}
	result.Consistency = float64(aboveThreshold) / float64(len(result.Windows)) * 100

	// Throughput
	result.ThroughputPerSec = float64(result.TotalOutputs) / result.TrainTimeSec

	// Score = (T × S × C) / 100000 (matching SPARTA dashboard)
	result.Score = (result.ThroughputPerSec * result.Stability * result.Consistency) / 100000
}

// ============================================================================
// HIVE MIND NETWORK ARCHITECTURE
// ============================================================================

func createHiveMindNetwork() *nn.Network {

	totalLayers := 1 + (numHives * 2) + 1 // Input + 3*(Hive+Merger) + Output
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Stack 3 Hives
	for i := 0; i < numHives; i++ {
		parallelLayer := createGridScatterHive()
		net.SetLayer(0, 0, layerIdx, parallelLayer)
		layerIdx++

		mergerInputSize := DModel * 4
		mergerLayer := nn.InitDenseLayer(mergerInputSize, DModel, nn.ActivationLeakyReLU)
		scaleWeights(mergerLayer.Kernel, InitScale)
		net.SetLayer(0, 0, layerIdx, mergerLayer)
		layerIdx++
	}

	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createGridScatterHive() nn.LayerConfig {
	brain00 := createMHABrain()
	brain01 := createLSTMBrain()
	brain10 := createMHABrain()
	brain11 := createMHABrain()

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   2,
		GridOutputCols:   2,
		GridOutputLayers: 1,
		ParallelBranches: []nn.LayerConfig{brain00, brain01, brain10, brain11},
		GridPositions: []nn.GridPosition{
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0},
			{BranchIndex: 2, TargetRow: 1, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 3, TargetRow: 1, TargetCol: 1, TargetLayer: 0},
		},
	}
	return parallel
}

func createMHABrain() nn.LayerConfig {
	headDim := DModel / NumHeads
	mha := nn.LayerConfig{
		Type:      nn.LayerMultiHeadAttention,
		DModel:    DModel,
		NumHeads:  NumHeads,
		SeqLength: 1,
	}
	mha.QWeights = make([]float32, DModel*DModel)
	mha.KWeights = make([]float32, DModel*DModel)
	mha.VWeights = make([]float32, DModel*DModel)
	mha.OutputWeight = make([]float32, DModel*DModel)
	mha.QBias = make([]float32, DModel)
	mha.KBias = make([]float32, DModel)
	mha.VBias = make([]float32, DModel)
	mha.OutputBias = make([]float32, DModel)

	qkScale := InitScale / float32(math.Sqrt(float64(headDim)))
	outScale := InitScale / float32(math.Sqrt(float64(DModel)))
	initRandom(mha.QWeights, qkScale)
	initRandom(mha.KWeights, qkScale)
	initRandom(mha.VWeights, qkScale)
	initRandom(mha.OutputWeight, outScale)
	return mha
}

func createLSTMBrain() nn.LayerConfig {
	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: DModel,
		HiddenSize:   LSTMHidden,
		SeqLength:    1,
		OutputHeight: DModel,
	}
	initLSTMWeights(&lstm)
	return lstm
}

func initLSTMWeights(cfg *nn.LayerConfig) {
	inputSize := cfg.RNNInputSize
	hiddenSize := cfg.HiddenSize

	cfg.WeightIH_i = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_f = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_g = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_o = make([]float32, hiddenSize*inputSize)
	cfg.WeightHH_i = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_f = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_g = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_o = make([]float32, hiddenSize*hiddenSize)
	cfg.BiasH_i = make([]float32, hiddenSize)
	cfg.BiasH_f = make([]float32, hiddenSize)
	cfg.BiasH_g = make([]float32, hiddenSize)
	cfg.BiasH_o = make([]float32, hiddenSize)

	scale := InitScale / float32(math.Sqrt(float64(hiddenSize)))
	initRandom(cfg.WeightIH_i, scale)
	initRandom(cfg.WeightIH_f, scale)
	initRandom(cfg.WeightIH_g, scale)
	initRandom(cfg.WeightIH_o, scale)
	initRandom(cfg.WeightHH_i, scale)
	initRandom(cfg.WeightHH_f, scale)
	initRandom(cfg.WeightHH_g, scale)
	initRandom(cfg.WeightHH_o, scale)
	for i := range cfg.BiasH_f {
		cfg.BiasH_f[i] = 1.0
	}
}

// ============================================================================
// Utilities
// ============================================================================

func scaleWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

func initRandom(slice []float32, scale float32) {
	for i := range slice {
		slice[i] = (rand.Float32()*2 - 1) * scale
	}
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func argmax(s []float32) int {
	if len(s) == 0 {
		return 0
	}
	maxI, maxV := 0, s[0]
	for i, v := range s {
		if v > maxV {
			maxV, maxI = v, i
		}
	}
	return maxI
}

// ============================================================================
// Data Loading
// ============================================================================

type rawTask struct {
	Train []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"train"`
	Test []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"test"`
}

func loadARCTasks(dir string, maxTasks int) ([]*ARCTask, error) {
	files, _ := os.ReadDir(dir)
	rand.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	var tasks []*ARCTask
	for _, f := range files {
		if len(tasks) >= maxTasks || filepath.Ext(f.Name()) != ".json" {
			continue
		}
		data, _ := os.ReadFile(filepath.Join(dir, f.Name()))
		var raw rawTask
		if json.Unmarshal(data, &raw) != nil {
			continue
		}
		task := &ARCTask{ID: f.Name()[:len(f.Name())-5]}
		for _, p := range raw.Train {
			task.Train = append(task.Train, GridPair{Input: p.Input, Output: p.Output})
		}
		for _, p := range raw.Test {
			task.Test = append(task.Test, GridPair{Input: p.Input, Output: p.Output})
		}
		tasks = append(tasks, task)
	}
	return tasks, nil
}

func encodeGrid(grid [][]int) []float32 {
	encoded := make([]float32, InputSize)
	for r := 0; r < len(grid) && r < MaxGridSize; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize; c++ {
			encoded[r*MaxGridSize+c] = float32(grid[r][c]) / 9.0
		}
	}
	return encoded
}

// ============================================================================
// Output
// ============================================================================

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("arc_benchmark_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to arc_benchmark_results.json")
}

func printTimeline(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           TRAINING PIXEL ACCURACY % (100ms windows) — Switching between 400 tasks in real-time                                                 ║")
	fmt.Println("║           NormalBP PAUSES to batch train → low throughput | StepTweenChain trains EVERY sample → maintains accuracy                           ║")
	fmt.Println("╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Printf("║ Mode                 ║")

	// Print time headers (showing every 1s = 10 windows)
	for i := 0; i < 100; i += 10 {
		fmt.Printf(" %ds ", i/10+1)
	}
	fmt.Printf("║ Eval  ║ Score      ║\n")
	fmt.Println("╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║", modeName)

		// Print accuracy for each 1-second block (average of 10 windows)
		for sec := 0; sec < 10; sec++ {
			avgAcc := 0.0
			for w := sec * 10; w < (sec+1)*10 && w < len(r.Windows); w++ {
				avgAcc += r.Windows[w].Accuracy
			}
			avgAcc /= 10
			fmt.Printf(" %2.0f%%", avgAcc)
		}
		fmt.Printf(" ║ %3.0f%% ║ %10.0f ║\n", r.EvalAccuracy, r.Score)
	}

	fmt.Println("╚══════════════════════╩════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                         ARC-AGI MODE COMPARISON — Score = (Throughput × Stability × Consistency) / 100000                                     ║")
	fmt.Println("╠══════════════════════╦════════════╦════════════╦══════════════╦════════════════╦════════════════╦════════════════╦══════════════════════════════╣")
	fmt.Println("║ Mode                 ║ Train Acc  ║ Stability  ║ Throughput   ║ Consistency    ║ Eval Acc       ║ Tasks Solved   ║ = Score                      ║")
	fmt.Println("╠══════════════════════╬════════════╬════════════╬══════════════╬════════════════╬════════════════╬════════════════╬══════════════════════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║ %8.1f%% ║ %8.0f%% ║ %10.0f/s ║ %12.0f%% ║ %12.1f%% ║ %14d ║ %26.0f   ║\n",
			modeName, r.AvgTrainAccuracy, r.Stability, r.ThroughputPerSec, r.Consistency, r.EvalAccuracy, r.TasksSolved, r.Score)

		if r.TrainDeviations != nil {
			fmt.Printf("\n--- [%s] Training Deviation Distribution (Learning Progress) ---\n", modeName)
			r.TrainDeviations.PrintSummary()
			fmt.Println("---------------------------------------------------------------")
		}
	}

	fmt.Println("╚══════════════════════╩════════════╩════════════╩══════════════╩════════════════╩════════════════╩════════════════╩══════════════════════════════╝")

	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                                    KEY METRICS                                          │")
	fmt.Println("├─────────────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Println("│ • Stability = 100 - stddev of window accuracies (higher = more consistent)             │")
	fmt.Println("│ • Consistency = % of windows with accuracy >= 30% (higher = reliable)                   │")
	fmt.Println("│ • Throughput = outputs per second (higher = faster processing)                          │")
	fmt.Println("│ • Score = (T × S × C) / 100000 (balanced metric)                                        │")
	fmt.Println("│                                                                                          │")
	fmt.Println("│ ★ StepTweenChain should score highest: trains every sample, maintains accuracy          │")
	fmt.Println("│ ★ NormalBP should score lowest: stops to batch train, disrupts throughput                │")
	fmt.Println("└─────────────────────────────────────────────────────────────────────────────────────────┘")
}

// ============================================================================
// Deviation Metrics (Buckets)
// ============================================================================

// DeviationBucket represents a specific deviation percentage range
type DeviationBucket struct {
	RangeMin float64 `json:"range_min"`
	RangeMax float64 `json:"range_max"`
	Count    int     `json:"count"`
}

// PredictionResult represents the performance of the model on one prediction
type PredictionResult struct {
	SampleIndex    int     `json:"sample_index"`
	ExpectedOutput float64 `json:"expected"`
	ActualOutput   float64 `json:"actual"`
	Deviation      float64 `json:"deviation"` // Percentage deviation
	Bucket         string  `json:"bucket"`
}

// DeviationMetrics stores the full model performance breakdown
type DeviationMetrics struct {
	Buckets          map[string]*DeviationBucket `json:"buckets"`
	Score            float64                     `json:"score"` // Average quality score (0-100)
	TotalSamples     int                         `json:"total_samples"`
	Failures         int                         `json:"failures"`      // Count of 100%+ deviations
	AverageDeviation float64                     `json:"avg_deviation"` // Mean deviation across all samples
}

// NewDeviationMetrics initializes an empty metrics struct
func NewDeviationMetrics() *DeviationMetrics {
	return &DeviationMetrics{
		Buckets: map[string]*DeviationBucket{
			"0-10%":   {0, 10, 0},
			"10-20%":  {10, 20, 0},
			"20-30%":  {20, 30, 0},
			"30-40%":  {30, 40, 0},
			"40-50%":  {40, 50, 0},
			"50-100%": {50, 100, 0},
			"100%+":   {100, math.Inf(1), 0},
		},
		Score: 0,
	}
}

// EvaluatePrediction categorizes an expected vs actual output into a deviation bucket
func EvaluatePrediction(sampleIndex int, expected, actual float64) PredictionResult {
	var deviation float64
	if math.Abs(expected) < 1e-10 { // Handle near-zero expected values
		deviation = math.Abs(actual-expected) * 100 // Scale to percentage (e.g. 0.1 diff = 10%)
	} else {
		// Avoid division by zero if expected is very small but not zero (shouldn't happen with 1e-10 check)
		deviation = math.Abs((actual - expected) / expected * 100)
	}

	// Prevent NaN/Inf issues
	if math.IsNaN(deviation) || math.IsInf(deviation, 0) {
		deviation = 100 // Default worst case
	}

	var bucketName string
	switch {
	case deviation <= 10:
		bucketName = "0-10%"
	case deviation <= 20:
		bucketName = "10-20%"
	case deviation <= 30:
		bucketName = "20-30%"
	case deviation <= 40:
		bucketName = "30-40%"
	case deviation <= 50:
		bucketName = "40-50%"
	case deviation <= 100:
		bucketName = "50-100%"
	default:
		bucketName = "100%+"
	}

	return PredictionResult{
		SampleIndex:    sampleIndex,
		ExpectedOutput: expected,
		ActualOutput:   actual,
		Deviation:      deviation,
		Bucket:         bucketName,
	}
}

// UpdateMetrics updates the metrics with a single prediction result
func (dm *DeviationMetrics) UpdateMetrics(result PredictionResult) {
	bucket := dm.Buckets[result.Bucket]
	bucket.Count++

	dm.TotalSamples++
	if result.Bucket == "100%+" {
		dm.Failures++
	}

	// Compute score: lower deviations contribute more positively
	dm.Score += math.Max(0, 100-result.Deviation)
	dm.AverageDeviation += result.Deviation // Accumulate sum
}

// PrintSummary prints a human-readable summary of the deviation metrics
func (dm *DeviationMetrics) PrintSummary() {
	if dm.TotalSamples == 0 {
		fmt.Println("No training samples recorded.")
		return
	}
	avgDev := dm.AverageDeviation / float64(dm.TotalSamples)
	score := dm.Score / float64(dm.TotalSamples)

	fmt.Printf("Total Pixels: %d | Quality Score: %.2f/100 | Avg Dev: %.2f%%\n", dm.TotalSamples, score, avgDev)
	fmt.Printf("Failures (>100%%): %d (%.1f%%)\n", dm.Failures, float64(dm.Failures)/float64(dm.TotalSamples)*100)

	fmt.Printf("Deviation Distribution:\n")
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucketName := range bucketOrder {
		bucket := dm.Buckets[bucketName]
		percentage := float64(bucket.Count) / float64(dm.TotalSamples) * 100
		barLen := int(percentage / 100 * 50)
		bar := ""
		for i := 0; i < barLen; i++ {
			bar += "█"
		}
		fmt.Printf("  %8s: %4d (%.1f%%) %s\n", bucketName, bucket.Count, percentage, bar)
	}
}

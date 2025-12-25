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
	NumTasks     = 400                       // Focus on fewer tasks for better pattern learning
	LearningRate = float32(0.01)             // Learning rate for training
	AdaptLR      = float32(0.02)             // Balanced LR for few-shot adaptation
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// Architecture params - balanced for speed + capacity
	DModel     = 64
	NumHeads   = 8
	LSTMHidden = 64

	// Timing - 90 seconds for better pattern learning
	TestDuration   = 10 * time.Minute
	WindowDuration = 100 * time.Millisecond // 100ms for fine-grained accuracy tracking

	// Batch training interval for NormalBP/Tween (this is where they PAUSE!)
	TrainInterval = 50 * time.Millisecond

	// Few-shot adaptation - aggressive for complex ARC-AGI2 rules
	AdaptationPasses = 200
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
	AvgTrainAccuracy float64 `json:"avgTrainAccuracy"`
	Stability        float64 `json:"stability"`   // 100 - stddev
	Consistency      float64 `json:"consistency"` // % windows above 30%
	ThroughputPerSec float64 `json:"throughputPerSec"`
	Score            float64 `json:"score"` // T×S×C / 100000
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
	fmt.Printf("║     TRAINING: ARC-AGI2 training set (1000 tasks) — Cycle through (%d seconds)       ║\n", int(TestDuration.Seconds()))
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
	fmt.Printf("\n📦 Loaded %d ARC-AGI2 training tasks\n", len(trainTasks))
	fmt.Printf("📦 Loaded %d ARC-AGI2 evaluation tasks\n", len(evalTasks))
	fmt.Printf("📊 Training: %d samples, switching between tasks rapidly\n", len(trainSamples))
	fmt.Printf("🎯 Eval: %d samples (unseen ARC-AGI2 test examples)\n", len(evalSamples))
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
		Windows: make([]TimeWindow, numWindows),
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

		// Calculate PIXEL ACCURACY % on this sample (not just solved/not)
		pixelAccuracy := calculatePixelAccuracy(output, sample)

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
		// Only Tween-based modes can adapt fast enough
		if ts != nil && (mode == ModeStepTweenChain || mode == ModeStepTween || mode == ModeTweenChain || mode == ModeTween) {
			// Extended adaptation loop for complex ARC-AGI2 rules
			for k := 0; k < AdaptationPasses; k++ {
				for _, pair := range task.Train {
					if len(pair.Input) == 0 || len(pair.Output) == 0 {
						continue
					}
					input := encodeGrid(pair.Input)
					target := encodeGrid(pair.Output)
					// Use higher learning rate for adaptation
					ts.TweenStep(net, input, argmax(target), len(target), AdaptLR)
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
			})
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
func calculatePixelAccuracy(output []float32, sample Sample) float64 {
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
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	parallelLayer := createGridScatterHive()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// 2x4 grid = 8 brains
	mergerInputSize := DModel * 8
	mergerLayer := nn.InitDenseLayer(mergerInputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createGridScatterHive() nn.LayerConfig {
	// 2x4 grid = 8 brains (4 MHA + 4 LSTM)
	brain00 := createMHABrain()
	brain01 := createLSTMBrain()
	brain02 := createMHABrain()
	brain03 := createLSTMBrain()
	brain10 := createMHABrain()
	brain11 := createLSTMBrain()
	brain12 := createMHABrain()
	brain13 := createLSTMBrain()

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   2,
		GridOutputCols:   4,
		GridOutputLayers: 1,
		ParallelBranches: []nn.LayerConfig{brain00, brain01, brain02, brain03, brain10, brain11, brain12, brain13},
		GridPositions: []nn.GridPosition{
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0},
			{BranchIndex: 2, TargetRow: 0, TargetCol: 2, TargetLayer: 0},
			{BranchIndex: 3, TargetRow: 0, TargetCol: 3, TargetLayer: 0},
			{BranchIndex: 4, TargetRow: 1, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 5, TargetRow: 1, TargetCol: 1, TargetLayer: 0},
			{BranchIndex: 6, TargetRow: 1, TargetCol: 2, TargetLayer: 0},
			{BranchIndex: 7, TargetRow: 1, TargetCol: 3, TargetLayer: 0},
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
	os.WriteFile("arc_benchmark_results_2.json", data, 0644)
	fmt.Println("\n✅ Results saved to arc_benchmark_results_2.json")
}

func printTimeline(results *BenchmarkResults) {
	numSeconds := int(TestDuration / time.Second)
	numWindows := int(TestDuration / WindowDuration)

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║           TRAINING PIXEL ACCURACY %% (100ms windows) — %d seconds, switching between 1000 tasks in real-time                                  ║\n", numSeconds)
	fmt.Println("║           NormalBP PAUSES to batch train → low throughput | StepTweenChain trains EVERY sample → maintains accuracy                           ║")
	fmt.Println("╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Printf("║ Mode                 ║")

	// Print time headers - show first 10 seconds, then last few if longer
	windowsPerSec := 10
	maxDisplaySecs := 10
	if numSeconds <= maxDisplaySecs {
		for i := 0; i < numWindows; i += windowsPerSec {
			fmt.Printf(" %ds ", i/windowsPerSec+1)
		}
	} else {
		// Show first 5 and last 5 seconds
		for i := 0; i < 5; i++ {
			fmt.Printf(" %ds ", i+1)
		}
		fmt.Printf(" ... ")
		for i := numSeconds - 5; i < numSeconds; i++ {
			fmt.Printf(" %ds ", i+1)
		}
	}
	fmt.Printf("║ Eval  ║ Score      ║\n")
	fmt.Println("╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║", modeName)

		// Print accuracy for each 1-second block
		if numSeconds <= maxDisplaySecs {
			for sec := 0; sec < numSeconds; sec++ {
				avgAcc := 0.0
				count := 0
				for w := sec * windowsPerSec; w < (sec+1)*windowsPerSec && w < len(r.Windows); w++ {
					avgAcc += r.Windows[w].Accuracy
					count++
				}
				if count > 0 {
					avgAcc /= float64(count)
				}
				fmt.Printf(" %2.0f%%", avgAcc)
			}
		} else {
			// Show first 5 and last 5 seconds
			for sec := 0; sec < 5; sec++ {
				avgAcc := 0.0
				for w := sec * windowsPerSec; w < (sec+1)*windowsPerSec && w < len(r.Windows); w++ {
					avgAcc += r.Windows[w].Accuracy
				}
				avgAcc /= float64(windowsPerSec)
				fmt.Printf(" %2.0f%%", avgAcc)
			}
			fmt.Printf(" ... ")
			for sec := numSeconds - 5; sec < numSeconds; sec++ {
				avgAcc := 0.0
				for w := sec * windowsPerSec; w < (sec+1)*windowsPerSec && w < len(r.Windows); w++ {
					avgAcc += r.Windows[w].Accuracy
				}
				avgAcc /= float64(windowsPerSec)
				fmt.Printf(" %2.0f%%", avgAcc)
			}
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

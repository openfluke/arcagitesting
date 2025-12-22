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

// Test 23: ARC-AGI Multi-Architecture Real-Time Benchmark
//
// Combines test22's architecture variety with test18's real-time display.
// Networks: Dense, Conv2D, Attn with 3, 5, 7 layers
// Modes: NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain
// Shows training progress per batch then eval benchmark

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400                       // Full ARC-AGI training set
	NumBatches   = 50                        // More batches for larger dataset
	BatchSize    = 100                       // Larger batches
	EvalRounds   = 3                         // Fewer rounds for speed
	LearningRate = float32(0.01)
)

type ArchType string

const (
	ArchDense  ArchType = "Dense"
	ArchConv2D ArchType = "Conv2D"
	ArchAttn   ArchType = "Attn"
)

var allArchitectures = []ArchType{ArchDense, ArchConv2D, ArchAttn}
var allDepths = []int{9, 11, 13}

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
	ModeStepTweenChain: "StepTweenChn",
}
var allModes = []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}

// Data types
type ARCTask struct {
	ID          string
	Train, Test []GridPair
}
type GridPair struct{ Input, Output [][]int }
type Sample struct {
	Input, Target []float32
	Height, Width int
}

type ConfigResult struct {
	TrainAccuracies []float64 // Per-batch train accuracy (cells)
	EvalAccuracies  []float64 // Per-batch eval accuracy (cells)
	FinalTrain      float64
	FinalEval       float64
	EvalBenchAvg    float64
	TasksSolved     int // Number of tasks 100% correct
	TotalTasks      int // Total tasks tested
	TrainTime       time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 23: ARC-AGI Multi-Architecture Real-Time Benchmark                ║")
	fmt.Println("║  Dense/Conv2D/Attn × 3/5/7 Layers × All Training Modes                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("Loaded %d tasks: %d train, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Results storage: config -> mode -> result
	allResults := make(map[string]map[TrainingMode]*ConfigResult)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// PHASE 1: Training section by section (6 modes in parallel per section)
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  PHASE 1: TRAINING (6 modes in parallel per architecture section)       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	for _, arch := range allArchitectures {
		for _, depth := range allDepths {
			configName := fmt.Sprintf("%s-%dL", arch, depth)
			allResults[configName] = make(map[TrainingMode]*ConfigResult)

			fmt.Printf("\n┌───────────────────────────────────────────────────────────────────────────┐\n")
			fmt.Printf("│ %-73s │\n", configName)
			fmt.Printf("└───────────────────────────────────────────────────────────────────────────┘\n")

			// Run all 6 modes in parallel FOR THIS SECTION
			for _, mode := range allModes {
				wg.Add(1)
				go func(a ArchType, d int, cn string, m TrainingMode) {
					defer wg.Done()
					net, result := runTraining(trainSamples, evalSamples, a, d, m)

					// Eval benchmark phase
					result.EvalBenchAvg, result.TasksSolved, result.TotalTasks = runEvalBenchmark(net, evalSamples, m)

					mu.Lock()
					allResults[cn][m] = result
					mu.Unlock()

					fmt.Printf("  [%-12s] %s Train: %4.0f%% | Eval: %4.0f%% | Tasks: %d/%d (%.1fs)\n",
						modeNames[m], cn, result.FinalTrain, result.FinalEval, result.TasksSolved, result.TotalTasks, result.TrainTime.Seconds())
				}(arch, depth, configName, mode)
			}
			// Wait for this section to complete before moving to next
			wg.Wait()
		}
	}
	fmt.Println("\n✓ All configurations complete!")

	// Print training timeline like test17
	printTrainingTimeline(allResults)

	// Print summary tables
	printArchitectureComparison(allResults)
	printModeComparison(allResults)
	printBestOverall(allResults)
	saveResults(allResults)
}

// ============================================================================
// Training
// ============================================================================

func runTraining(trainSamples, evalSamples []Sample, arch ArchType, depth int, mode TrainingMode) (*nn.Network, *ConfigResult) {
	start := time.Now()
	net := createNetwork(arch, depth)
	if net == nil {
		return nil, &ConfigResult{}
	}
	numLayers := net.TotalLayers()

	result := &ConfigResult{
		TrainAccuracies: make([]float64, NumBatches),
		EvalAccuracies:  make([]float64, NumBatches),
	}

	var state *nn.StepState
	usesStep := mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain
	if usesStep {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	usesTween := mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain
	if usesTween {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	sampleIdx := 0
	for batch := 0; batch < NumBatches; batch++ {
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++
			trainOneSample(net, sample, mode, numLayers, state, ts, LearningRate)
		}
		result.TrainAccuracies[batch] = measureAccuracy(net, trainSamples, mode, numLayers, state, ts)
		if len(evalSamples) > 0 {
			result.EvalAccuracies[batch] = measureAccuracy(net, evalSamples, mode, numLayers, state, ts)
		}
	}

	result.FinalTrain = result.TrainAccuracies[NumBatches-1]
	result.FinalEval = result.EvalAccuracies[NumBatches-1]
	result.TrainTime = time.Since(start)

	return net, result
}

func runEvalBenchmark(net *nn.Network, evalSamples []Sample, mode TrainingMode) (avgAcc float64, tasksSolved, totalTasks int) {
	if net == nil {
		return 0, 0, 0
	}
	numLayers := net.TotalLayers()

	var state *nn.StepState
	usesStep := mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain
	if usesStep {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	usesTween := mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain
	if usesTween {
		ts = nn.NewTweenState(net, nil)
	}

	// Measure cell accuracy
	totalAcc := 0.0
	for round := 0; round < EvalRounds; round++ {
		totalAcc += measureAccuracy(net, evalSamples, mode, numLayers, state, ts)
	}
	avgAcc = totalAcc / float64(EvalRounds)

	// Measure task-level accuracy (100% correct grids)
	totalTasks = len(evalSamples)
	for _, sample := range evalSamples {
		output := getOutput(net, sample.Input, mode, numLayers, state, ts)
		all_correct := true
		for r := 0; r < sample.Height && all_correct; r++ {
			for c := 0; c < sample.Width && all_correct; c++ {
				idx := r*MaxGridSize + c
				if idx < len(output) && idx < len(sample.Target) {
					pred := clampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					exp := clampInt(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
					if pred != exp {
						all_correct = false
					}
				}
			}
		}
		if all_correct {
			tasksSolved++
		}
	}
	return avgAcc, tasksSolved, totalTasks
}

func trainOneSample(net *nn.Network, sample Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState, lr float32) {
	switch mode {
	case ModeNormalBP:
		output, _ := net.ForwardCPU(sample.Input)
		grad := computeGradient(output, sample.Target)
		net.BackwardCPU(grad)
		net.ApplyGradients(lr)
	case ModeStepBP:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		grad := computeGradient(output, sample.Target)
		net.StepBackward(state, grad)
		net.ApplyGradients(lr)
	case ModeTween:
		ts.ForwardPass(net, sample.Input)
		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), lr)
	case ModeTweenChain:
		output := ts.ForwardPass(net, sample.Input)
		applyTweenUpdate(ts, net, sample, output, lr)
	case ModeStepTween:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), lr)
	case ModeStepTweenChain:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		ts.ForwardPass(net, sample.Input)
		applyTweenUpdate(ts, net, sample, output, lr)
	}
}

func computeGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	for i := range output {
		if i < len(target) {
			grad[i] = clipGrad(output[i]-target[i], 0.5)
		}
	}
	return grad
}

func applyTweenUpdate(ts *nn.TweenState, net *nn.Network, sample Sample, output []float32, lr float32) {
	outputGrad := make([]float32, len(output))
	for i := range output {
		if i < len(sample.Target) {
			outputGrad[i] = sample.Target[i] - output[i]
		}
	}
	ts.ChainGradients[net.TotalLayers()] = outputGrad
	ts.BackwardTargets[net.TotalLayers()] = sample.Target
	ts.TweenWeightsChainRule(net, lr)
}

func measureAccuracy(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		output := getOutput(net, sample.Input, mode, numLayers, state, ts)
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
	}
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}

func getOutput(net *nn.Network, input []float32, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) []float32 {
	switch mode {
	case ModeStepBP, ModeStepTween, ModeStepTweenChain:
		state.SetInput(input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		return state.GetOutput()
	case ModeTween, ModeTweenChain:
		return ts.ForwardPass(net, input)
	default:
		output, _ := net.ForwardCPU(input)
		return output
	}
}

// ============================================================================
// Network Factories
// ============================================================================

func createNetwork(arch ArchType, depth int) *nn.Network {
	switch arch {
	case ArchDense:
		return createDenseNet(depth)
	case ArchConv2D:
		return createConv2DNet(depth)
	case ArchAttn:
		return createAttnNet(depth)
	}
	return nil
}

func createDenseNet(depth int) *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	sizes := []int{128, 64, 48, 32}
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, sizes[0], nn.ActivationLeakyReLU))
	for i := 1; i < depth-1; i++ {
		inS := sizes[(i-1)%len(sizes)]
		outS := sizes[i%len(sizes)]
		net.SetLayer(0, 0, i, nn.InitDenseLayer(inS, outS, nn.ActivationLeakyReLU))
	}
	lastH := sizes[(depth-2)%len(sizes)]
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(lastH, InputSize, nn.ActivationSigmoid))
	return net
}

func createConv2DNet(depth int) *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	conv := nn.LayerConfig{
		Type: nn.LayerConv2D, InputHeight: 30, InputWidth: 30, InputChannels: 1,
		Filters: 8, KernelSize: 3, Stride: 1, Padding: 1,
		OutputHeight: 30, OutputWidth: 30, Activation: nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 8*1*3*3)
	conv.Bias = make([]float32, 8)
	initRandom(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)
	for i := 1; i < depth-1; i++ {
		if i == 1 {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(7200, 64, nn.ActivationLeakyReLU))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
		}
	}
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(64, InputSize, nn.ActivationSigmoid))
	return net
}

func createAttnNet(depth int) *nn.Network {
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU))
	for i := 1; i < depth-1; i++ {
		if i%2 == 1 {
			net.SetLayer(0, 0, i, createMHA(dModel))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU))
		}
	}
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid))
	return net
}

func createMHA(dModel int) nn.LayerConfig {
	headDim := dModel / 4
	mha := nn.LayerConfig{Type: nn.LayerMultiHeadAttention, DModel: dModel, NumHeads: 4}
	mha.QWeights = make([]float32, dModel*dModel)
	mha.KWeights = make([]float32, dModel*dModel)
	mha.VWeights = make([]float32, dModel*dModel)
	mha.OutputWeight = make([]float32, dModel*dModel)
	mha.QBias = make([]float32, dModel)
	mha.KBias = make([]float32, dModel)
	mha.VBias = make([]float32, dModel)
	mha.OutputBias = make([]float32, dModel)
	initRandom(mha.QWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandom(mha.KWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandom(mha.VWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandom(mha.OutputWeight, 0.1/float32(math.Sqrt(float64(dModel))))
	return mha
}

func initRandom(slice []float32, scale float32) {
	for i := range slice {
		slice[i] = (rand.Float32()*2 - 1) * scale
	}
}

// ============================================================================
// Printing
// ============================================================================

func printTrainingTimeline(results map[string]map[TrainingMode]*ConfigResult) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    TRAINING TIMELINE (Eval Accuracy % per batch)                                                                                                     ║")
	fmt.Println("╠═══════════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦═════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Mode              ║   B4      ║   B8      ║   B12     ║   B16     ║   B20     ║  Tasks    ║   Learning Curve                                                                            ║\n")
	fmt.Println("╠═══════════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════╬═════════════════════════════════════════════════════════════════════════════════════════════╣")

	// Get best config for each mode
	for _, mode := range allModes {
		// Find best config for this mode
		bestConfig := ""
		bestAcc := 0.0
		for config, modes := range results {
			if r, ok := modes[mode]; ok && r.EvalBenchAvg > bestAcc {
				bestAcc = r.EvalBenchAvg
				bestConfig = config
			}
		}
		if bestConfig == "" {
			continue
		}
		r := results[bestConfig][mode]

		// Generate simple curve
		curve := generateCurve(r.EvalAccuracies)

		taskInfo := fmt.Sprintf("%d/%d", r.TasksSolved, r.TotalTasks)
		fmt.Printf("║ %-17s ║   %4.0f%%   ║   %4.0f%%   ║   %4.0f%%   ║   %4.0f%%   ║   %4.0f%%   ║   %-5s   ║   %s\n",
			modeNames[mode]+"("+bestConfig+")",
			safeGet(r.EvalAccuracies, 3), safeGet(r.EvalAccuracies, 7),
			safeGet(r.EvalAccuracies, 11), safeGet(r.EvalAccuracies, 15), safeGet(r.EvalAccuracies, 19),
			taskInfo, curve)
	}
	fmt.Println("╚═══════════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩═════════════════════════════════════════════════════════════════════════════════════════════╝")
}

func generateCurve(accs []float64) string {
	if len(accs) == 0 {
		return ""
	}
	// Sample every 4 batches
	curve := ""
	for i := 0; i < len(accs); i += 2 {
		acc := accs[i]
		if acc < 15 {
			curve += "▁"
		} else if acc < 25 {
			curve += "▂"
		} else if acc < 35 {
			curve += "▃"
		} else if acc < 45 {
			curve += "▄"
		} else if acc < 55 {
			curve += "▅"
		} else if acc < 65 {
			curve += "▆"
		} else {
			curve += "█"
		}
	}
	return curve
}

func safeGet(slice []float64, idx int) float64 {
	if idx < len(slice) {
		return slice[idx]
	}
	return 0
}

func printArchitectureComparison(results map[string]map[TrainingMode]*ConfigResult) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ARCHITECTURE COMPARISON (Eval Accuracy %)                                                                                  ║")
	fmt.Println("╠═══════════════════╦═════════════╦═════════════╦═════════════╦═════════════╦═════════════╦═════════════╦═══════════════════════════════════════════════════════╣")
	fmt.Printf("║ Config            ║  NormalBP   ║   StepBP    ║    Tween    ║ TweenChain  ║  StepTween  ║ StepTwnChn  ║ Best                                                  ║\n")
	fmt.Println("╠═══════════════════╬═════════════╬═════════════╬═════════════╬═════════════╬═════════════╬═════════════╬═══════════════════════════════════════════════════════╣")

	configs := []string{}
	for _, arch := range allArchitectures {
		for _, d := range allDepths {
			configs = append(configs, fmt.Sprintf("%s-%dL", arch, d))
		}
	}

	for _, config := range configs {
		modes := results[config]
		best := ""
		bestVal := 0.0
		for _, m := range allModes {
			if r, ok := modes[m]; ok && r.EvalBenchAvg > bestVal {
				bestVal = r.EvalBenchAvg
				best = modeNames[m]
			}
		}
		fmt.Printf("║ %-17s ║   %5.0f%%    ║   %5.0f%%    ║   %5.0f%%    ║   %5.0f%%    ║   %5.0f%%    ║   %5.0f%%    ║ %-51s ║\n",
			config,
			modes[ModeNormalBP].EvalBenchAvg, modes[ModeStepBP].EvalBenchAvg, modes[ModeTween].EvalBenchAvg,
			modes[ModeTweenChain].EvalBenchAvg, modes[ModeStepTween].EvalBenchAvg, modes[ModeStepTweenChain].EvalBenchAvg,
			best)
	}
	fmt.Println("╚═══════════════════╩═════════════╩═════════════╩═════════════╩═════════════╩═════════════╩═════════════╩═══════════════════════════════════════════════════════╝")
}

func printModeComparison(results map[string]map[TrainingMode]*ConfigResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                            MODE SUMMARY (Average across all architectures)                        ║")
	fmt.Println("╠═══════════════════╦═══════════════╦═══════════════╦═══════════════╦══════════════════════════════╣")
	fmt.Println("║ Mode              ║ Avg Train     ║ Avg Eval      ║ Avg Time      ║ Notes                        ║")
	fmt.Println("╠═══════════════════╬═══════════════╬═══════════════╬═══════════════╬══════════════════════════════╣")

	for _, mode := range allModes {
		trainSum, evalSum, timeSum, count := 0.0, 0.0, 0.0, 0
		for _, modes := range results {
			if r, ok := modes[mode]; ok {
				trainSum += r.FinalTrain
				evalSum += r.EvalBenchAvg
				timeSum += r.TrainTime.Seconds()
				count++
			}
		}
		if count > 0 {
			note := ""
			if evalSum/float64(count) < 15 {
				note = "⚠ Not learning"
			}
			fmt.Printf("║ %-17s ║    %5.1f%%     ║    %5.1f%%     ║    %5.1fs     ║ %-28s ║\n",
				modeNames[mode], trainSum/float64(count), evalSum/float64(count), timeSum/float64(count), note)
		}
	}
	fmt.Println("╚═══════════════════╩═══════════════╩═══════════════╩═══════════════╩══════════════════════════════╝")
}

func printBestOverall(results map[string]map[TrainingMode]*ConfigResult) {
	bestConfig, bestMode, bestAcc := "", ModeNormalBP, 0.0
	for config, modes := range results {
		for mode, r := range modes {
			if r.EvalBenchAvg > bestAcc {
				bestAcc = r.EvalBenchAvg
				bestConfig = config
				bestMode = mode
			}
		}
	}
	fmt.Printf("\n★ BEST OVERALL: %s + %s = %.0f%% eval accuracy\n", bestConfig, modeNames[bestMode], bestAcc)
	fmt.Println("  Note: Random baseline ~11%, higher is better")
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

func splitTrainEval(tasks []*ARCTask) (trainSamples, evalSamples []Sample) {
	for _, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			trainSamples = append(trainSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]),
			})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]),
			})
		}
	}
	if len(evalSamples) == 0 && len(trainSamples) > 5 {
		holdout := len(trainSamples) / 5
		evalSamples = trainSamples[len(trainSamples)-holdout:]
		trainSamples = trainSamples[:len(trainSamples)-holdout]
	}
	return trainSamples, evalSamples
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
// Utility
// ============================================================================

func clipGrad(v, max float32) float32 {
	if v > max {
		return max
	}
	if v < -max {
		return -max
	}
	if math.IsNaN(float64(v)) {
		return 0
	}
	return v
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

func saveResults(results map[string]map[TrainingMode]*ConfigResult) {
	output := make(map[string]map[string]interface{})
	for config, modes := range results {
		output[config] = make(map[string]interface{})
		for mode, r := range modes {
			output[config][modeNames[mode]] = map[string]interface{}{
				"final_train": r.FinalTrain,
				"final_eval":  r.FinalEval,
				"eval_bench":  r.EvalBenchAvg,
				"train_time":  r.TrainTime.Seconds(),
			}
		}
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test23_results.json", data, 0644)
	fmt.Println("✓ Results saved to test23_results.json")
}

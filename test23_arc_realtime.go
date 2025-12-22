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

// Test 23: ARC-AGI Real-Time Benchmark
//
// Shows training progress live across all batches, then benchmarks
// trained models against eval data with mode switching.
//
// Phase 1: TRAINING - Shows accuracy per batch, all modes in parallel
// Phase 2: EVAL BENCHMARK - Tests trained models on eval data over multiple rounds

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 10
	NumBatches   = 30 // Training batches
	BatchSize    = 50 // Samples per batch
	EvalRounds   = 10 // Evaluation rounds switching between modes
	LearningRate = float32(0.01)
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

type TrainingProgress struct {
	BatchAccuracies []float64 // Accuracy after each batch
	EvalAccuracies  []float64 // Eval accuracy after each batch
	FinalTrainAcc   float64
	FinalEvalAcc    float64
	TrainTime       time.Duration
}

type EvalResult struct {
	RoundAccuracies []float64 // Accuracy per eval round
	AvgAccuracy     float64
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 23: ARC-AGI Real-Time Benchmark                                    ║")
	fmt.Println("║  Phase 1: Training with live progress                                   ║")
	fmt.Println("║  Phase 2: Eval benchmark with mode switching                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("Loaded %d tasks: %d train, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Phase 1: Training
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  PHASE 1: TRAINING                                                       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	allProgress := make(map[TrainingMode]*TrainingProgress)
	trainedNets := make(map[TrainingMode]*nn.Network)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, mode := range allModes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			net, progress := runTrainingWithProgress(trainSamples, evalSamples, m)
			mu.Lock()
			trainedNets[m] = net
			allProgress[m] = progress
			mu.Unlock()
		}(mode)
	}
	wg.Wait()

	// Print training timeline
	printTrainingTimeline(allProgress)

	// Phase 2: Eval Benchmark
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  PHASE 2: EVAL BENCHMARK (switching between modes each round)           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	evalResults := make(map[TrainingMode]*EvalResult)
	for _, mode := range allModes {
		evalResults[mode] = runEvalBenchmark(trainedNets[mode], evalSamples, mode)
	}

	printEvalTimeline(evalResults)
	printFinalSummary(allProgress, evalResults)
	saveResults(allProgress, evalResults)
}

// ============================================================================
// Training
// ============================================================================

func runTrainingWithProgress(trainSamples, evalSamples []Sample, mode TrainingMode) (*nn.Network, *TrainingProgress) {
	start := time.Now()
	net := createNetwork()
	numLayers := net.TotalLayers()

	progress := &TrainingProgress{
		BatchAccuracies: make([]float64, NumBatches),
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
		// Measure accuracy on both train and eval
		progress.BatchAccuracies[batch] = measureAccuracy(net, trainSamples, mode, numLayers, state, ts)
		if len(evalSamples) > 0 {
			progress.EvalAccuracies[batch] = measureAccuracy(net, evalSamples, mode, numLayers, state, ts)
		}
		fmt.Printf("  [%-12s] Batch %2d/%d: Train %4.0f%% | Eval %4.0f%%\n",
			modeNames[mode], batch+1, NumBatches, progress.BatchAccuracies[batch], progress.EvalAccuracies[batch])
	}

	progress.FinalTrainAcc = progress.BatchAccuracies[NumBatches-1]
	progress.FinalEvalAcc = progress.EvalAccuracies[NumBatches-1]
	progress.TrainTime = time.Since(start)

	return net, progress
}

func runEvalBenchmark(net *nn.Network, evalSamples []Sample, mode TrainingMode) *EvalResult {
	result := &EvalResult{RoundAccuracies: make([]float64, EvalRounds)}
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

	totalAcc := 0.0
	for round := 0; round < EvalRounds; round++ {
		acc := measureAccuracy(net, evalSamples, mode, numLayers, state, ts)
		result.RoundAccuracies[round] = acc
		totalAcc += acc
	}
	result.AvgAccuracy = totalAcc / float64(EvalRounds)
	return result
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
	totalLayers := net.TotalLayers()
	ts.ChainGradients[totalLayers] = outputGrad
	ts.BackwardTargets[totalLayers] = sample.Target
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
// Network
// ============================================================================

func createNetwork() *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, 5)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, 128, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(128, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 4, nn.InitDenseLayer(32, InputSize, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Printing
// ============================================================================

func printTrainingTimeline(progress map[TrainingMode]*TrainingProgress) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    TRAINING PROGRESS (Eval Accuracy per 5 batches)                                                                            ║")
	fmt.Println("╠═══════════════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╗")
	fmt.Printf("║ Mode              ║  B5     ║  B10    ║  B15    ║  B20    ║  B25    ║  B30    ║ Final   ║ Time    ║\n")
	fmt.Println("╠═══════════════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╣")

	for _, mode := range allModes {
		p := progress[mode]
		fmt.Printf("║ %-17s ║  %4.0f%%  ║  %4.0f%%  ║  %4.0f%%  ║  %4.0f%%  ║  %4.0f%%  ║  %4.0f%%  ║  %4.0f%%  ║ %5.1fs  ║\n",
			modeNames[mode],
			p.EvalAccuracies[4], p.EvalAccuracies[9], p.EvalAccuracies[14],
			p.EvalAccuracies[19], p.EvalAccuracies[24], p.EvalAccuracies[29],
			p.FinalEvalAcc, p.TrainTime.Seconds())
	}
	fmt.Println("╚═══════════════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝")
}

func printEvalTimeline(results map[TrainingMode]*EvalResult) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    EVAL BENCHMARK (Accuracy per round on held-out eval data)                                                              ║")
	fmt.Println("╠═══════════════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═════════╗")
	fmt.Printf("║ Mode              ║  R1     ║  R2     ║  R3     ║  R4     ║  R5     ║  R6     ║  R7     ║  R8     ║  R9     ║  R10    ║  Avg    ║\n")
	fmt.Println("╠═══════════════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╣")

	for _, mode := range allModes {
		r := results[mode]
		fmt.Printf("║ %-17s ║", modeNames[mode])
		for i := 0; i < 10; i++ {
			fmt.Printf("  %4.0f%%  ║", r.RoundAccuracies[i])
		}
		fmt.Printf("  %4.0f%%  ║\n", r.AvgAccuracy)
	}
	fmt.Println("╚═══════════════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝")
}

func printFinalSummary(progress map[TrainingMode]*TrainingProgress, evalResults map[TrainingMode]*EvalResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    FINAL SUMMARY                                                 ║")
	fmt.Println("╠═══════════════════╦═══════════════╦═══════════════╦═══════════════╦══════════════════════════════╣")
	fmt.Println("║ Mode              ║ Train Final   ║ Eval Final    ║ Generalization║ Winner?                      ║")
	fmt.Println("╠═══════════════════╬═══════════════╬═══════════════╬═══════════════╬══════════════════════════════╣")

	bestMode := allModes[0]
	bestEval := evalResults[bestMode].AvgAccuracy
	for _, mode := range allModes {
		if evalResults[mode].AvgAccuracy > bestEval {
			bestEval = evalResults[mode].AvgAccuracy
			bestMode = mode
		}
	}

	for _, mode := range allModes {
		p := progress[mode]
		e := evalResults[mode]
		gap := p.FinalTrainAcc - p.FinalEvalAcc
		winner := ""
		if mode == bestMode {
			winner = "★ BEST!"
		}
		fmt.Printf("║ %-17s ║     %5.1f%%    ║     %5.1f%%    ║    %+5.1f%%    ║ %-28s ║\n",
			modeNames[mode], p.FinalTrainAcc, e.AvgAccuracy, -gap, winner)
	}
	fmt.Println("╚═══════════════════╩═══════════════╩═══════════════╩═══════════════╩══════════════════════════════╝")

	fmt.Println("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                                         KEY INSIGHTS                                              │")
	fmt.Println("├────────────────────────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ ★ Best Overall: %s with %.0f%% eval accuracy\n", modeNames[bestMode], bestEval)
	fmt.Println("│ • Generalization gap = Train acc - Eval acc (smaller is better)")
	fmt.Println("│ • Eval benchmark tests trained models on held-out data")
	fmt.Println("│ • Random baseline ~11%, higher is better")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────────────────────────┘")
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

func saveResults(progress map[TrainingMode]*TrainingProgress, eval map[TrainingMode]*EvalResult) {
	output := make(map[string]interface{})
	for mode := range progress {
		output[modeNames[mode]] = map[string]interface{}{
			"final_train": progress[mode].FinalTrainAcc,
			"final_eval":  progress[mode].FinalEvalAcc,
			"avg_eval":    eval[mode].AvgAccuracy,
			"train_time":  progress[mode].TrainTime.Seconds(),
		}
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test23_results.json", data, 0644)
	fmt.Println("✓ Results saved to test23_results.json")
}

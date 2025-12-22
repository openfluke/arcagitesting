package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 25: BICAMERAL AI - Two Brains, One Mind
//
// Based on Test 24b findings:
//   - "RightBrain" (Heuristic/StepTween): Spatial/physics tasks (54% acc)
//   - "LeftBrain" (Math/StepBP): Logical/counting tasks (7% acc, but unique solves)
//
// The Corpus Callosum: Average predictions from both brains
// Goal: Maximize Task Completion through ensemble

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 400
	BatchSize   = 100
	NumEpochs   = 200
	InitScale   = float32(0.5)
)

// Brain-specific configs
const (
	RightBrainLR   = float32(0.001)
	RightBrainClip = float32(0.1)
	LeftBrainLR    = float32(0.001)
	LeftBrainClip  = float32(0.1)
)

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
}

type BrainResult struct {
	Name            string
	AccuracyHistory []float64
	BudgetHistory   []float32
	FinalAccuracy   float64
	FinalBudget     float32
	TasksSolved     int
	SolvedTaskIDs   []string
	TrainTime       time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 25: BICAMERAL AI - Two Brains, One Mind                                    ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🧠 RightBrain: Heuristic (StepTween) - Spatial/Physics Intelligence             ║")
	fmt.Println("║     🧮 LeftBrain:  Math (StepBP)         - Logical/Counting Intelligence            ║")
	fmt.Println("║     🔗 Corpus Callosum: Ensemble averaging of both predictions                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Architecture: Attn-21L × 2 (Dual Networks)                                      ║")
	fmt.Println("║     Epochs: 200 | Batch Size: 100                                                   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the two brains
	rightBrain := createMHAStabilized(21)
	leftBrain := createMHAStabilized(21)

	rightState := rightBrain.InitStepState(InputSize)
	leftState := leftBrain.InitStepState(InputSize)

	rightTS := nn.NewTweenState(rightBrain, nil)
	rightTS.Config.LinkBudgetScale = 0.8

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🧠 BICAMERAL TRAINING BEGINS 🧠")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	rightResult := &BrainResult{Name: "RightBrain", AccuracyHistory: make([]float64, NumEpochs), BudgetHistory: make([]float32, NumEpochs)}
	leftResult := &BrainResult{Name: "LeftBrain", AccuracyHistory: make([]float64, NumEpochs), BudgetHistory: make([]float32, NumEpochs)}
	combinedResult := &BrainResult{Name: "Combined", AccuracyHistory: make([]float64, NumEpochs), BudgetHistory: make([]float32, NumEpochs)}

	start := time.Now()
	numLayers := rightBrain.TotalLayers()

	// Training loop - train both brains in parallel
	sampleIdx := 0
	for epoch := 0; epoch < NumEpochs; epoch++ {
		var wg sync.WaitGroup

		// Train both brains on the same batch
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			wg.Add(2)

			// RightBrain (Heuristic)
			go func(s Sample) {
				defer wg.Done()
				trainRightBrain(rightBrain, s, numLayers, rightState, rightTS)
			}(sample)

			// LeftBrain (Math)
			go func(s Sample) {
				defer wg.Done()
				trainLeftBrain(leftBrain, s, numLayers, leftState)
			}(sample)

			wg.Wait()
		}

		// Measure metrics for all three: Right, Left, Combined
		rightAcc, rightBudget := measureBrainMetrics(rightBrain, evalSamples, numLayers, rightState, rightTS, true)
		leftAcc, leftBudget := measureBrainMetrics(leftBrain, evalSamples, numLayers, leftState, nil, false)
		combinedAcc := measureCombinedMetrics(rightBrain, leftBrain, evalSamples, numLayers, rightState, leftState)

		rightResult.AccuracyHistory[epoch] = rightAcc
		rightResult.BudgetHistory[epoch] = rightBudget
		leftResult.AccuracyHistory[epoch] = leftAcc
		leftResult.BudgetHistory[epoch] = leftBudget
		combinedResult.AccuracyHistory[epoch] = combinedAcc
		combinedResult.BudgetHistory[epoch] = (rightBudget + leftBudget) / 2

		if (epoch+1)%20 == 0 {
			fmt.Printf("  Epoch %3d/%d | 🧠 Right: %5.1f%% | 🧮 Left: %5.1f%% | 🔗 Combined: %5.1f%%\n",
				epoch+1, NumEpochs, rightAcc, leftAcc, combinedAcc)
		}
	}

	trainTime := time.Since(start)

	// Final evaluation with task tracking
	rightResult.FinalAccuracy = rightResult.AccuracyHistory[NumEpochs-1]
	rightResult.FinalBudget = rightResult.BudgetHistory[NumEpochs-1]
	leftResult.FinalAccuracy = leftResult.AccuracyHistory[NumEpochs-1]
	leftResult.FinalBudget = leftResult.BudgetHistory[NumEpochs-1]
	combinedResult.FinalAccuracy = combinedResult.AccuracyHistory[NumEpochs-1]
	combinedResult.FinalBudget = combinedResult.BudgetHistory[NumEpochs-1]

	rightResult.TasksSolved, rightResult.SolvedTaskIDs = measureSolvedTasks(rightBrain, evalSamples, numLayers, rightState, rightTS, true)
	leftResult.TasksSolved, leftResult.SolvedTaskIDs = measureSolvedTasks(leftBrain, evalSamples, numLayers, leftState, nil, false)
	combinedResult.TasksSolved, combinedResult.SolvedTaskIDs = measureCombinedSolvedTasks(rightBrain, leftBrain, evalSamples, numLayers, rightState, leftState)

	rightResult.TrainTime = trainTime
	leftResult.TrainTime = trainTime
	combinedResult.TrainTime = trainTime

	fmt.Printf("\n✅ Training complete in %.1fs\n", trainTime.Seconds())

	// Print results
	printBicameralResults(rightResult, leftResult, combinedResult)
	printVennDiagram(rightResult, leftResult, combinedResult)
	saveResults(rightResult, leftResult, combinedResult)
}

// ============================================================================
// Training Functions
// ============================================================================

func trainRightBrain(net *nn.Network, sample Sample, numLayers int, state *nn.StepState, ts *nn.TweenState) {
	state.SetInput(sample.Input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), RightBrainLR)
}

func trainLeftBrain(net *nn.Network, sample Sample, numLayers int, state *nn.StepState) {
	state.SetInput(sample.Input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	output := state.GetOutput()
	grad := computeGradient(output, sample.Target, LeftBrainClip)
	net.StepBackward(state, grad)
	net.ApplyGradients(LeftBrainLR)
}

func computeGradient(output, target []float32, clip float32) []float32 {
	grad := make([]float32, len(output))
	for i := range output {
		if i < len(target) {
			diff := output[i] - target[i]
			if diff > clip {
				diff = clip
			} else if diff < -clip {
				diff = -clip
			}
			grad[i] = diff
		}
	}
	return grad
}

// ============================================================================
// Metrics Functions
// ============================================================================

func measureBrainMetrics(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState, ts *nn.TweenState, useTween bool) (accuracy float64, budget float32) {
	correct, total := 0, 0

	for _, sample := range samples {
		output := getBrainOutput(net, sample.Input, numLayers, state)
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
		accuracy = 0
	} else {
		accuracy = float64(correct) / float64(total) * 100
	}

	if ts != nil && len(ts.LinkBudgets) > 0 {
		midIdx := len(ts.LinkBudgets) / 2
		budget = ts.LinkBudgets[midIdx]
	} else {
		budget = estimateBudget(net, samples[0].Input, state)
	}

	return accuracy, budget
}

func measureCombinedMetrics(rightNet, leftNet *nn.Network, samples []Sample, numLayers int, rightState, leftState *nn.StepState) float64 {
	correct, total := 0, 0

	for _, sample := range samples {
		rightOutput := getBrainOutput(rightNet, sample.Input, numLayers, rightState)
		leftOutput := getBrainOutput(leftNet, sample.Input, numLayers, leftState)

		// Corpus Callosum: Average predictions
		combined := make([]float32, len(rightOutput))
		for i := range combined {
			if i < len(leftOutput) {
				combined[i] = (rightOutput[i] + leftOutput[i]) / 2
			} else {
				combined[i] = rightOutput[i]
			}
		}

		for r := 0; r < sample.Height; r++ {
			for c := 0; c < sample.Width; c++ {
				idx := r*MaxGridSize + c
				if idx < len(combined) && idx < len(sample.Target) {
					pred := clampInt(int(math.Round(float64(combined[idx])*9.0)), 0, 9)
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

func measureSolvedTasks(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState, ts *nn.TweenState, useTween bool) (int, []string) {
	solved := 0
	solvedIDs := []string{}

	for _, sample := range samples {
		output := getBrainOutput(net, sample.Input, numLayers, state)
		if isTaskSolved(output, sample) {
			solved++
			solvedIDs = append(solvedIDs, sample.TaskID)
		}
	}

	return solved, solvedIDs
}

func measureCombinedSolvedTasks(rightNet, leftNet *nn.Network, samples []Sample, numLayers int, rightState, leftState *nn.StepState) (int, []string) {
	solved := 0
	solvedIDs := []string{}

	for _, sample := range samples {
		rightOutput := getBrainOutput(rightNet, sample.Input, numLayers, rightState)
		leftOutput := getBrainOutput(leftNet, sample.Input, numLayers, leftState)

		// Corpus Callosum: Average
		combined := make([]float32, len(rightOutput))
		for i := range combined {
			if i < len(leftOutput) {
				combined[i] = (rightOutput[i] + leftOutput[i]) / 2
			} else {
				combined[i] = rightOutput[i]
			}
		}

		if isTaskSolved(combined, sample) {
			solved++
			solvedIDs = append(solvedIDs, sample.TaskID)
		}
	}

	return solved, solvedIDs
}

func isTaskSolved(output []float32, sample Sample) bool {
	for r := 0; r < sample.Height; r++ {
		for c := 0; c < sample.Width; c++ {
			idx := r*MaxGridSize + c
			if idx < len(output) && idx < len(sample.Target) {
				pred := clampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
				exp := clampInt(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
				if pred != exp {
					return false
				}
			}
		}
	}
	return true
}

func getBrainOutput(net *nn.Network, input []float32, numLayers int, state *nn.StepState) []float32 {
	state.SetInput(input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	return state.GetOutput()
}

func estimateBudget(net *nn.Network, input []float32, state *nn.StepState) float32 {
	output := getBrainOutput(net, input, net.TotalLayers(), state)
	sum := float32(0)
	for _, v := range output {
		sum += float32(math.Abs(float64(v)))
	}
	if len(output) == 0 {
		return 0
	}
	avgMag := sum / float32(len(output))
	budget := avgMag * 2
	if budget > 1 {
		budget = 1
	}
	return budget
}

// ============================================================================
// Visualization
// ============================================================================

func printBicameralResults(right, left, combined *BrainResult) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                       🧠 BICAMERAL AI - FINAL RESULTS 🧠                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                      ║")
	fmt.Printf("║   🧠 RightBrain (Heuristic):  Accuracy = %5.1f%% | Tasks Solved = %d                 ║\n", right.FinalAccuracy, right.TasksSolved)
	fmt.Printf("║   🧮 LeftBrain (Math):        Accuracy = %5.1f%% | Tasks Solved = %d                  ║\n", left.FinalAccuracy, left.TasksSolved)
	fmt.Printf("║   🔗 Combined (Ensemble):     Accuracy = %5.1f%% | Tasks Solved = %d                 ║\n", combined.FinalAccuracy, combined.TasksSolved)
	fmt.Println("║                                                                                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	// Progress timeline
	fmt.Println("║                         ACCURACY TIMELINE                                            ║")
	fmt.Println("╠═══════════════════╦════════════╦════════════╦════════════╦════════════╦══════════════╣")
	fmt.Println("║ Brain             ║   Ep 40    ║   Ep 80    ║   Ep 120   ║   Ep 160   ║   Ep 200     ║")
	fmt.Println("╠═══════════════════╬════════════╬════════════╬════════════╬════════════╬══════════════╣")
	fmt.Printf("║ 🧠 RightBrain     ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%     ║\n",
		safeGet(right.AccuracyHistory, 39), safeGet(right.AccuracyHistory, 79),
		safeGet(right.AccuracyHistory, 119), safeGet(right.AccuracyHistory, 159), right.FinalAccuracy)
	fmt.Printf("║ 🧮 LeftBrain      ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%     ║\n",
		safeGet(left.AccuracyHistory, 39), safeGet(left.AccuracyHistory, 79),
		safeGet(left.AccuracyHistory, 119), safeGet(left.AccuracyHistory, 159), left.FinalAccuracy)
	fmt.Printf("║ 🔗 Combined       ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%   ║   %5.1f%%     ║\n",
		safeGet(combined.AccuracyHistory, 39), safeGet(combined.AccuracyHistory, 79),
		safeGet(combined.AccuracyHistory, 119), safeGet(combined.AccuracyHistory, 159), combined.FinalAccuracy)
	fmt.Println("╚═══════════════════╩════════════╩════════════╩════════════╩════════════╩══════════════╝")
}

func printVennDiagram(right, left, combined *BrainResult) {
	// Build sets
	rightSet := make(map[string]bool)
	leftSet := make(map[string]bool)
	combinedSet := make(map[string]bool)

	for _, id := range right.SolvedTaskIDs {
		rightSet[id] = true
	}
	for _, id := range left.SolvedTaskIDs {
		leftSet[id] = true
	}
	for _, id := range combined.SolvedTaskIDs {
		combinedSet[id] = true
	}

	// Calculate overlaps
	onlyRight := 0
	onlyLeft := 0
	onlyCombined := 0
	rightAndLeft := 0
	rightAndCombined := 0
	leftAndCombined := 0
	allThree := 0

	// Union of all
	allIDs := make(map[string]bool)
	for id := range rightSet {
		allIDs[id] = true
	}
	for id := range leftSet {
		allIDs[id] = true
	}
	for id := range combinedSet {
		allIDs[id] = true
	}

	for id := range allIDs {
		inRight := rightSet[id]
		inLeft := leftSet[id]
		inCombined := combinedSet[id]

		if inRight && inLeft && inCombined {
			allThree++
		} else if inRight && inLeft {
			rightAndLeft++
		} else if inRight && inCombined {
			rightAndCombined++
		} else if inLeft && inCombined {
			leftAndCombined++
		} else if inRight {
			onlyRight++
		} else if inLeft {
			onlyLeft++
		} else if inCombined {
			onlyCombined++
		}
	}

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🔗 CORPUS CALLOSUM - VENN DIAGRAM 🔗                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║                    ┌─────────────────┐     ┌─────────────────┐                       ║")
	fmt.Printf("║                    │   🧠 Right      │     │   🧮 Left       │                       ║\n")
	fmt.Printf("║                    │    Only: %-3d    │     │    Only: %-3d    │                       ║\n", onlyRight, onlyLeft)
	fmt.Println("║                    │                 │     │                 │                       ║")
	fmt.Println("║                    └────────┬────────┘     └────────┬────────┘                       ║")
	fmt.Println("║                             │                       │                                ║")
	fmt.Printf("║                             └───────┬───────────────┘                                ║\n")
	fmt.Printf("║                                     │ Both: %-3d                                      ║\n", rightAndLeft)
	fmt.Println("║                                     │                                                ║")
	fmt.Println("║                             ┌───────┴───────┐                                        ║")
	fmt.Printf("║                             │  🔗 Combined  │                                        ║\n")
	fmt.Printf("║                             │   Only: %-3d   │                                        ║\n", onlyCombined)
	fmt.Println("║                             └───────────────┘                                        ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  📊 TASK COVERAGE SUMMARY:                                                           ║\n")
	fmt.Printf("║     • RightBrain solved:    %3d tasks                                                ║\n", right.TasksSolved)
	fmt.Printf("║     • LeftBrain solved:     %3d tasks                                                ║\n", left.TasksSolved)
	fmt.Printf("║     • Combined solved:      %3d tasks                                                ║\n", combined.TasksSolved)
	fmt.Printf("║     • TOTAL UNION:          %3d unique tasks (solved by at least one method)        ║\n", len(allIDs))
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	// Determine winner
	best := "RightBrain"
	bestCount := right.TasksSolved
	if left.TasksSolved > bestCount {
		best = "LeftBrain"
		bestCount = left.TasksSolved
	}
	if combined.TasksSolved > bestCount {
		best = "Combined"
		bestCount = combined.TasksSolved
	}
	if len(allIDs) > bestCount {
		fmt.Println("║  ✨ INSIGHT: The UNION of all methods solves MORE tasks than any single method!     ║")
		fmt.Println("║     → Bicameral ensemble provides complementary intelligence!                       ║")
	} else {
		fmt.Printf("║  🏆 WINNER: %s with %d tasks solved                                             ║\n", best, bestCount)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Print sample task IDs
	if len(right.SolvedTaskIDs) > 0 {
		fmt.Println("\n📋 Sample Solved Task IDs:")
		fmt.Printf("   🧠 RightBrain: %v\n", truncateList(right.SolvedTaskIDs, 5))
		fmt.Printf("   🧮 LeftBrain:  %v\n", truncateList(left.SolvedTaskIDs, 5))
		fmt.Printf("   🔗 Combined:   %v\n", truncateList(combined.SolvedTaskIDs, 5))
	}
}

func truncateList(list []string, max int) []string {
	if len(list) <= max {
		return list
	}
	result := make([]string, max+1)
	copy(result, list[:max])
	result[max] = fmt.Sprintf("...+%d more", len(list)-max)
	return result
}

// ============================================================================
// Network Factory
// ============================================================================

func createMHAStabilized(depth int) *nn.Network {
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1

	inputLayer := nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, 0, inputLayer)

	for i := 1; i < depth-1; i++ {
		if i%2 == 1 {
			net.SetLayer(0, 0, i, createMHALayer(dModel))
		} else {
			denseLayer := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
			scaleWeights(denseLayer.Kernel, InitScale)
			net.SetLayer(0, 0, i, denseLayer)
		}
	}

	outputLayer := nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, depth-1, outputLayer)

	return net
}

func createMHALayer(dModel int) nn.LayerConfig {
	headDim := dModel / 4

	mha := nn.LayerConfig{
		Type:     nn.LayerMultiHeadAttention,
		DModel:   dModel,
		NumHeads: 4,
	}

	mha.QWeights = make([]float32, dModel*dModel)
	mha.KWeights = make([]float32, dModel*dModel)
	mha.VWeights = make([]float32, dModel*dModel)
	mha.OutputWeight = make([]float32, dModel*dModel)
	mha.QBias = make([]float32, dModel)
	mha.KBias = make([]float32, dModel)
	mha.VBias = make([]float32, dModel)
	mha.OutputBias = make([]float32, dModel)

	qkScale := InitScale / float32(math.Sqrt(float64(headDim)))
	outScale := InitScale / float32(math.Sqrt(float64(dModel)))

	initRandom(mha.QWeights, qkScale)
	initRandom(mha.KWeights, qkScale)
	initRandom(mha.VWeights, qkScale)
	initRandom(mha.OutputWeight, outScale)

	return mha
}

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
				TaskID: task.ID,
			})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]),
				TaskID: task.ID,
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

func safeGet(slice []float64, idx int) float64 {
	if idx < len(slice) && idx >= 0 {
		return slice[idx]
	}
	return 0
}

func saveResults(right, left, combined *BrainResult) {
	output := map[string]interface{}{
		"RightBrain": map[string]interface{}{
			"final_accuracy":   right.FinalAccuracy,
			"final_budget":     right.FinalBudget,
			"tasks_solved":     right.TasksSolved,
			"solved_task_ids":  right.SolvedTaskIDs,
			"accuracy_history": right.AccuracyHistory,
		},
		"LeftBrain": map[string]interface{}{
			"final_accuracy":   left.FinalAccuracy,
			"final_budget":     left.FinalBudget,
			"tasks_solved":     left.TasksSolved,
			"solved_task_ids":  left.SolvedTaskIDs,
			"accuracy_history": left.AccuracyHistory,
		},
		"Combined": map[string]interface{}{
			"final_accuracy":   combined.FinalAccuracy,
			"final_budget":     combined.FinalBudget,
			"tasks_solved":     combined.TasksSolved,
			"solved_task_ids":  combined.SolvedTaskIDs,
			"accuracy_history": combined.AccuracyHistory,
		},
		"meta": map[string]interface{}{
			"architecture":    "Attn-21L × 2 (Bicameral)",
			"epochs":          NumEpochs,
			"batch_size":      BatchSize,
			"right_brain_lr":  RightBrainLR,
			"left_brain_lr":   LeftBrainLR,
			"ensemble_method": "Average",
		},
	}

	// Calculate union
	allIDs := make(map[string]bool)
	for _, id := range right.SolvedTaskIDs {
		allIDs[id] = true
	}
	for _, id := range left.SolvedTaskIDs {
		allIDs[id] = true
	}
	for _, id := range combined.SolvedTaskIDs {
		allIDs[id] = true
	}

	unionIDs := make([]string, 0, len(allIDs))
	for id := range allIDs {
		unionIDs = append(unionIDs, id)
	}
	sort.Strings(unionIDs)

	output["union"] = map[string]interface{}{
		"total_unique_solved": len(allIDs),
		"task_ids":            unionIDs,
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test25_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test25_results.json")
}

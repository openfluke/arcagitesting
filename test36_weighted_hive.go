package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 36: WEIGHTED HIVE - Class-Weighted Heuristic Training
//
// Problem: Previous tests collapsed to "Predicting All Black" because the ARC
// dataset is dominated by background pixels (~90% are color 0).
//
// Solution: Weight the training signal by pixel importance:
//   - Background (0): Weight = 0.1 (Don't care much)
//   - Colors (1-9):   Weight = 10.0 (CARE A LOT)
//
// Expected Results:
//   - Accuracy will DROP (maybe 30-40%) - stops getting free background points
//   - Solved Tasks will RISE - forced to focus on actual object patterns

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize
	NumTasks    = 400
	BatchSize   = 100
	NumEpochs   = 1400

	LearningRate = float32(0.001) // Safe speed
	BudgetScale  = float32(0.8)

	// Weights for class imbalance
	WeightBackground = float32(0.1)  // Background pixels (color 0)
	WeightColor      = float32(10.0) // Colored pixels (colors 1-9)

	// Architecture - same as Test 35 Hive Mind
	DModel       = 64
	NumHeads     = 4
	LSTMHidden   = 64
	ConvGridSize = 8
	ConvFilters  = 8
	ConvKernel   = 3
	InitScale    = float32(0.5)
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

type Results struct {
	AccuracyHistory         []float64
	WeightedAccuracyHistory []float64 // New: tracks color-weighted accuracy
	BudgetHistory           []float32
	FinalAccuracy           float64
	FinalWeightedAccuracy   float64
	TasksSolved             int
	SolvedTaskIDs           []string
	TrainTime               time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 36: WEIGHTED HIVE - Class-Weighted Heuristic Training                      ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🎯 Background (0): Weight = 0.1 (Ignore)                                        ║")
	fmt.Println("║     🌈 Colors (1-9):   Weight = 10.0 (FOCUS!)                                       ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Goal: Drop accuracy, RAISE solved tasks by focusing on objects                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks.\n", len(tasks))

	// Create Hive Mind Network
	net := createHiveMindNetwork()
	numLayers := net.TotalLayers()
	state := net.InitStepState(InputSize)

	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = false // Heuristic mode
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory:         make([]float64, NumEpochs),
		WeightedAccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:           make([]float32, NumEpochs),
		SolvedTaskIDs:           []string{},
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     ⚖️ WEIGHTED HEURISTIC TRAINING ⚖️")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}

			// === WEIGHTED TRAINING LOGIC ===
			// Instead of uniform TweenStep, we apply weighted updates
			weightedTweenStep(ts, net, sample, LearningRate)
		}

		// Measure metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		wAcc := measureWeightedAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.WeightedAccuracyHistory[epoch] = wAcc
		results.BudgetHistory[epoch] = budget

		if (epoch+1)%20 == 0 {
			fmt.Printf("  Epoch %3d/%d | Acc: %5.1f%% | Weighted Acc: %5.1f%% | Budget: %.3f\n",
				epoch+1, NumEpochs, acc, wAcc, budget)
		}
	}

	results.TrainTime = time.Since(start)
	results.FinalAccuracy = results.AccuracyHistory[NumEpochs-1]
	results.FinalWeightedAccuracy = results.WeightedAccuracyHistory[NumEpochs-1]
	results.TasksSolved, results.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, numLayers, state)

	printResults(results)
	saveResults(results)
}

// weightedTweenStep applies TweenStep with class weights
// Colors get 10x more attention than background
func weightedTweenStep(ts *nn.TweenState, net *nn.Network, sample Sample, lr float32) {
	// For each pixel, calculate weighted learning rate based on target
	// We'll use multiple TweenStep calls with varying rates

	// Find the most important (non-zero) target index
	// This focuses the heuristic on colored pixels
	colorIndices := []int{}
	bgIndices := []int{}

	for i, t := range sample.Target {
		if t > 0.05 { // Color pixel (normalized > 0)
			colorIndices = append(colorIndices, i)
		} else {
			bgIndices = append(bgIndices, i)
		}
	}

	// If we have color pixels, focus on them with high weight
	if len(colorIndices) > 0 {
		// Pick a random color pixel to focus on
		idx := colorIndices[rand.Intn(len(colorIndices))]
		ts.TweenStep(net, sample.Input, idx, len(sample.Target), lr*WeightColor)
	}

	// Occasionally also train on background (low weight)
	if len(bgIndices) > 0 && rand.Float32() < 0.1 { // Only 10% of the time
		idx := bgIndices[rand.Intn(len(bgIndices))]
		ts.TweenStep(net, sample.Input, idx, len(sample.Target), lr*WeightBackground)
	}
}

// measureWeightedAccuracy measures accuracy on colored pixels only
func measureWeightedAccuracy(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		for i := range output {
			if i < len(sample.Target) {
				// Only count colored pixels
				exp := clampInt(int(math.Round(float64(sample.Target[i])*9.0)), 0, 9)
				if exp > 0 { // Skip background
					pred := clampInt(int(math.Round(float64(output[i])*9.0)), 0, 9)
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

// === NETWORK ARCHITECTURE (same as Test 35) ===

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

	mergerLayer := nn.InitDenseLayer(DModel*4, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createGridScatterHive() nn.LayerConfig {
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   2,
		GridOutputCols:   2,
		GridOutputLayers: 1,
		ParallelBranches: []nn.LayerConfig{
			createMHABrain(), createLSTMBrain(), createCNNBrain(), createMHABrain(),
		},
		GridPositions: []nn.GridPosition{
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0},
			{BranchIndex: 2, TargetRow: 1, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 3, TargetRow: 1, TargetCol: 1, TargetLayer: 0},
		},
	}
}

func createMHABrain() nn.LayerConfig {
	headDim := DModel / NumHeads
	mha := nn.LayerConfig{Type: nn.LayerMultiHeadAttention, DModel: DModel, NumHeads: NumHeads, SeqLength: 1}
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
	lstm := nn.LayerConfig{Type: nn.LayerLSTM, RNNInputSize: DModel, HiddenSize: LSTMHidden, SeqLength: 1, OutputHeight: DModel}
	initLSTMWeights(&lstm)
	return lstm
}

func createCNNBrain() nn.LayerConfig {
	cnn := nn.LayerConfig{
		Type: nn.LayerConv2D, InputHeight: ConvGridSize, InputWidth: ConvGridSize,
		InputChannels: 1, Filters: ConvFilters, KernelSize: ConvKernel,
		Stride: 1, Padding: 1, OutputHeight: ConvGridSize, OutputWidth: ConvGridSize,
		Activation: nn.ActivationLeakyReLU,
	}
	fanIn := ConvKernel * ConvKernel
	kernelSize := ConvFilters * ConvKernel * ConvKernel
	cnn.Kernel = make([]float32, kernelSize)
	cnn.Bias = make([]float32, ConvFilters)
	scale := InitScale / float32(math.Sqrt(float64(fanIn)))
	initRandom(cnn.Kernel, scale)
	return cnn
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

// === HELPERS ===

func measureAccuracy(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		for i := range output {
			if i < len(sample.Target) {
				pred := clampInt(int(math.Round(float64(output[i])*9.0)), 0, 9)
				exp := clampInt(int(math.Round(float64(sample.Target[i])*9.0)), 0, 9)
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

func measureSolvedTasks(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) (int, []string) {
	solved := 0
	solvedIDs := []string{}
	seen := make(map[string]bool)
	for _, sample := range samples {
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		isSolved := true
		for i := range output {
			if i < len(sample.Target) {
				pred := clampInt(int(math.Round(float64(output[i])*9.0)), 0, 9)
				exp := clampInt(int(math.Round(float64(sample.Target[i])*9.0)), 0, 9)
				if pred != exp {
					isSolved = false
					break
				}
			}
		}
		if isSolved && !seen[sample.TaskID] {
			solved++
			solvedIDs = append(solvedIDs, sample.TaskID)
			seen[sample.TaskID] = true
		}
	}
	return solved, solvedIDs
}

func getBudget(ts *nn.TweenState) float32 {
	if len(ts.LinkBudgets) > 0 {
		return ts.LinkBudgets[len(ts.LinkBudgets)/2]
	}
	return 0.5
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

type rawTask struct {
	Train []struct{ Input, Output [][]int }
	Test  []struct{ Input, Output [][]int }
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

func splitTrainEval(tasks []*ARCTask) (trainSamples, evalSamples []Sample) {
	for _, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 {
				continue
			}
			trainSamples = append(trainSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID,
			})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID,
			})
		}
	}
	holdout := len(trainSamples) / 5
	evalSamples = trainSamples[len(trainSamples)-holdout:]
	trainSamples = trainSamples[:len(trainSamples)-holdout]
	return trainSamples, evalSamples
}

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      ⚖️ WEIGHTED HIVE - FINAL RESULTS ⚖️                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Final Accuracy:           %5.1f%%                                                  ║\n", results.FinalAccuracy)
	fmt.Printf("║   Weighted Accuracy (Color): %5.1f%%                                                  ║\n", results.FinalWeightedAccuracy)
	fmt.Printf("║   Tasks Solved:             %d                                                        ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:            %.1fs                                                     ║\n", results.TrainTime.Seconds())
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	if len(results.SolvedTaskIDs) > 0 {
		fmt.Println("\n📋 Solved Task IDs:")
		for i, id := range results.SolvedTaskIDs {
			if i >= 10 {
				fmt.Printf("   ... and %d more\n", len(results.SolvedTaskIDs)-10)
				break
			}
			fmt.Printf("   - %s\n", id)
		}
	}
}

func saveResults(results *Results) {
	output := map[string]interface{}{
		"final_accuracy":            results.FinalAccuracy,
		"final_weighted_accuracy":   results.FinalWeightedAccuracy,
		"tasks_solved":              results.TasksSolved,
		"solved_task_ids":           results.SolvedTaskIDs,
		"train_time_sec":            results.TrainTime.Seconds(),
		"accuracy_history":          results.AccuracyHistory,
		"weighted_accuracy_history": results.WeightedAccuracyHistory,
		"meta": map[string]interface{}{
			"architecture":      "2x2 Grid Scatter Hive Mind",
			"training_mode":     "Weighted StepTween",
			"weight_background": WeightBackground,
			"weight_color":      WeightColor,
			"epochs":            NumEpochs,
			"learning_rate":     LearningRate,
		},
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test36_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test36_results.json")
}

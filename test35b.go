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

// Test 35: THE PHASE SHIFT - Hybrid Optimizer Strategy
//
// Concept:
//   Phase 1 (Epoch 0-200): "The Sprint"
//     - Mode: StepTween (Heuristic)
//     - Goal: Rapidly reach the 54% plateau. Establish global structure.
//
//   Phase 2 (Epoch 200-600): "The Refinement"
//     - Mode: StepTweenChain (Gradient)
//     - Goal: Use precise Chain Rule updates to break the 54% ceiling.
//     - Note: We lower LR significantly to prevent "forgetting" the heuristic structure.

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize
	NumTasks    = 400
	BatchSize   = 100
	NumEpochs   = 1600

	// Phase 1 Params (Heuristic)
	Phase1Epochs = 200
	LR_Heuristic = float32(0.1)
	Scale_Heur   = float32(0.8)

	// Phase 2 Params (Gradient)
	LR_Gradient = float32(0.00001) // 10x smaller for fine-tuning
	Scale_Grad  = float32(2.0)     // High scale to drive weak gradients

	// Architecture - DModel must be >= ConvGridSize^2 for CNN branch
	DModel       = 64 // Embedding dimension
	NumHeads     = 4
	LSTMHidden   = 64
	ConvGridSize = 8 // 8x8 = 64 pixels, matches DModel
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
	AccuracyHistory []float64
	BudgetHistory   []float32
	FinalAccuracy   float64
	TasksSolved     int
	SolvedTaskIDs   []string
	SwitchEpoch     int
	TrainTime       time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 35: THE PHASE SHIFT (Heuristic → Gradient Handover)                        ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Phase 1 (0-200):  StepTween (Heuristic) → Sprint to 54%                         ║")
	fmt.Println("║     Phase 2 (200-600): StepTweenChain (Gradient) → Fine-tune to break the ceiling    ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Architecture: 2x2 Grid Scatter Hive Mind (proven robust)                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks.\n", len(tasks))

	// Create 2x2 Hive Mind (Reuse architecture from Test 30/31)
	net := createHiveMindNetwork()
	numLayers := net.TotalLayers()
	state := net.InitStepState(InputSize)

	// Init TweenState (starts in Heuristic mode)
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true // Start Heuristic
	ts.Config.LinkBudgetScale = Scale_Heur

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
		SwitchEpoch:     Phase1Epochs,
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🚀 PHASE 1: HEURISTIC SPRINT 🚀")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {

		// === THE PHASE SWITCH ===
		if epoch == Phase1Epochs {
			fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
			fmt.Println("                     🔄 PHASE 2: GRADIENT HANDOVER 🔄")
			fmt.Println("           (Switching to Chain Rule, Lower LR, Higher Budget Scale)")
			fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
			ts.Config.UseChainRule = true
			ts.Config.LinkBudgetScale = Scale_Grad
		}

		currentLR := LR_Heuristic
		if epoch >= Phase1Epochs {
			currentLR = LR_Gradient
		}

		// Training loop
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}

			if ts.Config.UseChainRule {
				// Gradient Path
				output := state.GetOutput()
				ts.ForwardPass(net, sample.Input) // Refresh forward pass for chain rule state
				outputGrad := make([]float32, len(output))
				for k := range output {
					if k < len(sample.Target) {
						outputGrad[k] = sample.Target[k] - output[k]
					}
				}
				ts.ChainGradients[net.TotalLayers()] = outputGrad
				ts.TweenWeightsChainRule(net, currentLR)
			} else {
				// Heuristic Path
				ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), currentLR)
			}
		}

		// Metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)
		results.AccuracyHistory[epoch] = acc
		results.BudgetHistory[epoch] = budget

		if (epoch+1)%20 == 0 {
			mode := "Heur"
			if epoch >= Phase1Epochs {
				mode = "Grad"
			}
			fmt.Printf("  [%s] Epoch %3d/%d | Accuracy: %5.1f%% | Budget: %.3f\n",
				mode, epoch+1, NumEpochs, acc, budget)
		}
	}

	results.TrainTime = time.Since(start)
	results.FinalAccuracy = results.AccuracyHistory[NumEpochs-1]
	results.TasksSolved, results.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, numLayers, state)

	printResults(results)
	saveResults(results)
}

// ... [Insert standard HiveMind Network Architecture from Test 30 here] ...
// ... [Insert createHiveMindNetwork, createGridScatterHive, Brain creators] ...
// ... [Insert Helper functions: measureAccuracy, etc] ...

// --- Copied Helpers for context ---
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
			{0, 0, 0, 0}, {1, 0, 1, 0}, {2, 1, 0, 0}, {3, 1, 1, 0},
		},
	}
}

// (Rest of the architecture functions createMHABrain, createLSTMBrain, createCNNBrain
// are identical to Test 30/31. Paste them here to complete the file)

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
	cnn := nn.LayerConfig{Type: nn.LayerConv2D, InputHeight: ConvGridSize, InputWidth: ConvGridSize, InputChannels: 1, Filters: ConvFilters, KernelSize: ConvKernel, Stride: 1, Padding: 1, OutputHeight: ConvGridSize, OutputWidth: ConvGridSize, Activation: nn.ActivationLeakyReLU}
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
	Train []struct {
		Input  [][]int
		Output [][]int
	}
	Test []struct {
		Input  [][]int
		Output [][]int
	}
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
			trainSamples = append(trainSamples, Sample{Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output), Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output), Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID})
		}
	}
	holdout := len(trainSamples) / 5
	evalSamples = trainSamples[len(trainSamples)-holdout:]
	trainSamples = trainSamples[:len(trainSamples)-holdout]
	return trainSamples, evalSamples
}

func printResults(results *Results) {
	fmt.Printf("\nFinal Accuracy: %5.1f%%\nTasks Solved: %d\n", results.FinalAccuracy, results.TasksSolved)
	if len(results.SolvedTaskIDs) > 0 {
		fmt.Println("Solved IDs:", results.SolvedTaskIDs)
	}
}

func saveResults(results *Results) {
	output := map[string]interface{}{"final_accuracy": results.FinalAccuracy, "tasks_solved": results.TasksSolved, "solved_task_ids": results.SolvedTaskIDs}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test35b_results.json", data, 0644)
	fmt.Println("\n✅ Results saved.")
}

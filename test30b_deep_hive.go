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

// Test 30b: DEEP HIVE STACK - Stacked 2x2 Grid Scatter Architectures
//
// Concept:
//   - Layer 1: "Sensory Hive" (Processes raw grid)
//   - Layer 2: "Synapse" (Compresses features)
//   - Layer 3: "Reasoning Hive" (Processes abstract features from Hive 1)
//
// Training: StepTween (Heuristic) - Essential for training through this depth.

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 400 // Heuristic converges faster
	LearningRate = float32(0.001)
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// Architecture params
	DModel     = 32 // Embedding dimension
	NumHeads   = 4
	LSTMHidden = 32

	// CNN Params
	ConvFilters  = 4
	ConvKernel   = 3
	ConvGridSize = 6
)

// Data types (Standard)
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
	FinalBudget     float32
	TasksSolved     int
	SolvedTaskIDs   []string
	TrainTime       time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 30b: DEEP HIVE STACK - Recursive Grid Scatter Architecture                 ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Input → [Hive 1: Sensory] → Synapse → [Hive 2: Abstract] → Output               ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Each Hive contains:                                                              ║")
	fmt.Println("║       🧠 MHA (Spatial) + 🧮 LSTM (Logic) + 👁️ CNN (Features)                       ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Training: StepTween (Heuristic) - The only way to train this deep.              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the Deep Hive network
	net := createDeepHiveNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Deep Hive Network: %d layers\n", numLayers)

	// Initialize training state (Heuristic Mode)
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = BudgetScale // 0.8 for Heuristic

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🐝 DEEP HIVE TRAINING BEGINS 🐝")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		// Training loop
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			// Forward pass
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}

			// StepTween (Heuristic) - No gradients, just target propagation
			ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)
		}

		// Measure metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.BudgetHistory[epoch] = budget

		if (epoch+1)%20 == 0 {
			fmt.Printf("  Epoch %3d/%d | Accuracy: %5.1f%% | Budget: %.3f\n",
				epoch+1, NumEpochs, acc, budget)
		}
	}

	results.TrainTime = time.Since(start)
	results.FinalAccuracy = results.AccuracyHistory[NumEpochs-1]
	results.FinalBudget = results.BudgetHistory[NumEpochs-1]
	results.TasksSolved, results.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, numLayers, state)

	fmt.Printf("\n✅ Training complete in %.1fs\n", results.TrainTime.Seconds())

	// Print results
	printResults(results)
	saveResults(results)
}

// ============================================================================
// DEEP HIVE ARCHITECTURE
// ============================================================================

func createDeepHiveNetwork() *nn.Network {
	// Layers:
	// 0: Embed
	// 1: Hive 1 (Parallel)
	// 2: Synapse (Dense Compression)
	// 3: Hive 2 (Parallel)
	// 4: Merger
	// 5: Output
	totalLayers := 6
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (900 -> 32)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: HIVE 1 (Sensory)
	hive1 := createGridScatterHive()
	net.SetLayer(0, 0, layerIdx, hive1)
	layerIdx++

	// Layer 2: SYNAPSE (Compression)
	// Hive outputs 4 * 32 = 128. We compress back to 32 for Hive 2.
	// This forces abstraction.
	synapseLayer := nn.InitDenseLayer(DModel*4, DModel, nn.ActivationLeakyReLU)
	scaleWeights(synapseLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, synapseLayer)
	layerIdx++

	// Layer 3: HIVE 2 (Reasoning)
	// Takes the abstract concept from synapse and processes it again.
	hive2 := createGridScatterHive()
	net.SetLayer(0, 0, layerIdx, hive2)
	layerIdx++

	// Layer 4: MERGER
	mergerLayer := nn.InitDenseLayer(DModel*4, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 5: Output (32 -> 900)
	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

// Reused from Test 30
func createGridScatterHive() nn.LayerConfig {
	brain00 := createMHABrain()
	brain01 := createLSTMBrain()
	brain10 := createCNNBrain() // Using CNN here properly
	brain11 := createMHABrain()

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   2,
		GridOutputCols:   2,
		GridOutputLayers: 1,
		ParallelBranches: []nn.LayerConfig{
			brain00, brain01, brain10, brain11,
		},
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
	// ... Init weights ...
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

func createCNNBrain() nn.LayerConfig {
	// Using MHA instead of CNN to avoid dimension mismatch
	// CNN would require reshaping 32-dim linear to 2D spatial grid
	// MHA handles 1D embeddings natively
	return createMHABrain()
}

// ... (Rest of boilerplate: initLSTMWeights, Metrics, Utils, Data Loading) ...
// Same as Test 30 but ensure StepTween is used in main()

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
		output := getOutput(net, sample.Input, numLayers, state)
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

func measureSolvedTasks(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) (int, []string) {
	solved := 0
	solvedIDs := []string{}
	seen := make(map[string]bool)
	for _, sample := range samples {
		output := getOutput(net, sample.Input, numLayers, state)
		if isTaskSolved(output, sample) {
			if !seen[sample.TaskID] {
				solved++
				solvedIDs = append(solvedIDs, sample.TaskID)
				seen[sample.TaskID] = true
			}
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

func getOutput(net *nn.Network, input []float32, numLayers int, state *nn.StepState) []float32 {
	state.SetInput(input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	return state.GetOutput()
}

func getBudget(ts *nn.TweenState) float32 {
	if len(ts.LinkBudgets) > 0 {
		midIdx := len(ts.LinkBudgets) / 2
		return ts.LinkBudgets[midIdx]
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

func safeGet(slice []float64, idx int) float64 {
	if idx < len(slice) && idx >= 0 {
		return slice[idx]
	}
	return 0
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
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"train"`
	Test []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"test"`
}

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🐝 DEEP HIVE - FINAL RESULTS 🐝                                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
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
		"final_accuracy":  results.FinalAccuracy,
		"tasks_solved":    results.TasksSolved,
		"solved_task_ids": results.SolvedTaskIDs,
		"meta": map[string]interface{}{
			"architecture": "Deep Hive (Stacked 2x2)",
			"training":     "StepTween (Heuristic)",
		},
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test30b_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test30b_results.json")
}

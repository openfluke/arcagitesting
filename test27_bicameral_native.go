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

// Test 27: BICAMERAL NATIVE - True Dual-Path Architecture using LayerParallel
//
// Architecture:
//   - Input: 30x30 Grid (900 floats)
//   - Layer 0: Dense -> dModel (shared embedding)
//   - Layer 1: LayerParallel (Split into two branches)
//       - Branch A (RightBrain): Stack of MHA layers -> Spatial/global patterns
//       - Branch B (LeftBrain): Stack of LSTM layers -> Sequential/counting
//   - Layer 2: Dense Merger (concatenated outputs -> decision)
//   - Layer 3: Output (Dense -> 900)
//
// Training: StepTween ONLY (proven robust in Test 24b)

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 400
	LearningRate = float32(0.001)
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// Architecture params
	DModel     = 64 // Embedding dimension
	NumHeads   = 4
	LSTMHidden = 64
	SeqLength  = 1 // Treat as single sequence for grid
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
	FinalBudget     float32
	TasksSolved     int
	SolvedTaskIDs   []string
	TrainTime       time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 27: BICAMERAL NATIVE - True Dual-Path Architecture                         ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🧠 RightBrain (MHA Branch):   Global patterns, spatial relationships            ║")
	fmt.Println("║     🧮 LeftBrain (LSTM Branch):   Sequential reasoning, counting                    ║")
	fmt.Println("║     🔗 Unified via LayerParallel: Single network, dual intelligence                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Training: StepTween (Heuristic) - proven robust at 54% on Attn-21L              ║")
	fmt.Println("║     Architecture: Parallel(MHA-6L + LSTM-6L) → Merge → Output                       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the bicameral network
	net := createBicameralNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Bicameral Network: %d layers\n", numLayers)

	// Initialize training state
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🧠 BICAMERAL TRAINING BEGINS 🧠")
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

			// TweenStep training
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
// BICAMERAL NETWORK ARCHITECTURE
// ============================================================================

func createBicameralNetwork() *nn.Network {
	// Total layers: Input Embed + Parallel + Merger + Output = 4 layers
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (900 -> dModel)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: PARALLEL SPLIT - The Bicameral Core
	parallelLayer := createParallelBicameralLayer()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Layer 2: Merger - Combine both branches (dModel*2 -> dModel)
	// After concat: RightBrain(dModel) + LeftBrain(dModel) = dModel*2
	mergerLayer := nn.InitDenseLayer(DModel*2, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 3: Output (dModel -> 900)
	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createParallelBicameralLayer() nn.LayerConfig {
	// Create the two branches
	rightBrain := createRightBrainBranch() // MHA chain
	leftBrain := createLeftBrainBranch()   // LSTM chain

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "concat", // Concatenate outputs from both brains
		ParallelBranches: []nn.LayerConfig{
			rightBrain,
			leftBrain,
		},
	}

	return parallel
}

// RightBrain: Stack of MHA layers for spatial/global pattern recognition
func createRightBrainBranch() nn.LayerConfig {
	// Use a single MHA layer (the branch handles one layer)
	// We could nest multiple but keeping it simple
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

// LeftBrain: LSTM for sequential/counting reasoning
func createLeftBrainBranch() nn.LayerConfig {
	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: DModel,
		HiddenSize:   LSTMHidden, // Same as DModel for easy concat
		SeqLength:    SeqLength,
		OutputHeight: DModel, // Output dimension for compatibility
	}

	// Initialize LSTM weights
	initLSTMWeights(&lstm)

	return lstm
}

func initLSTMWeights(cfg *nn.LayerConfig) {
	inputSize := cfg.RNNInputSize
	hiddenSize := cfg.HiddenSize

	// Input-to-hidden weights for all 4 gates
	cfg.WeightIH_i = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_f = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_g = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_o = make([]float32, hiddenSize*inputSize)

	// Hidden-to-hidden weights for all 4 gates
	cfg.WeightHH_i = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_f = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_g = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_o = make([]float32, hiddenSize*hiddenSize)

	// Biases for all 4 gates
	cfg.BiasH_i = make([]float32, hiddenSize)
	cfg.BiasH_f = make([]float32, hiddenSize)
	cfg.BiasH_g = make([]float32, hiddenSize)
	cfg.BiasH_o = make([]float32, hiddenSize)

	// Initialize with conservative scale
	scale := InitScale / float32(math.Sqrt(float64(hiddenSize)))
	initRandom(cfg.WeightIH_i, scale)
	initRandom(cfg.WeightIH_f, scale)
	initRandom(cfg.WeightIH_g, scale)
	initRandom(cfg.WeightIH_o, scale)
	initRandom(cfg.WeightHH_i, scale)
	initRandom(cfg.WeightHH_f, scale)
	initRandom(cfg.WeightHH_g, scale)
	initRandom(cfg.WeightHH_o, scale)

	// Initialize forget gate bias to 1 (helps with gradient flow)
	for i := range cfg.BiasH_f {
		cfg.BiasH_f[i] = 1.0
	}
}

// ============================================================================
// Metrics
// ============================================================================

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
	for _, sample := range samples {
		output := getOutput(net, sample.Input, numLayers, state)
		if isTaskSolved(output, sample) {
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

// ============================================================================
// Visualization
// ============================================================================

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🧠 BICAMERAL NATIVE - FINAL RESULTS 🧠                          ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║                                                                                      ║\n")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
	fmt.Printf("║                                                                                      ║\n")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           ACCURACY TIMELINE                                          ║")
	fmt.Println("╠════════════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═════════════════╣")
	fmt.Println("║     Epoch          ║    40     ║    80     ║   120     ║   160     ║   200           ║")
	fmt.Println("╠════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═════════════════╣")
	fmt.Printf("║ Bicameral (MHA+LSTM) ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%         ║\n",
		safeGet(results.AccuracyHistory, 39), safeGet(results.AccuracyHistory, 79),
		safeGet(results.AccuracyHistory, 119), safeGet(results.AccuracyHistory, 159),
		results.FinalAccuracy)
	fmt.Println("╚════════════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═════════════════╝")

	// Comparison with baseline
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           COMPARISON WITH BASELINE                                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Baseline (Attn-21L StepTween from Test 24b):  ~54% accuracy, 1 task solved          ║")
	fmt.Println("║  ─────────────────────────────────────────────────────────────────────────────────── ║")

	if results.FinalAccuracy > 54 {
		fmt.Println("║  ✨ BREAKTHROUGH: Bicameral architecture BEATS the Attn-21L baseline!               ║")
	} else if results.FinalAccuracy > 45 {
		fmt.Println("║  ⚡ COMPETITIVE: Bicameral is close to baseline performance.                        ║")
	} else if results.FinalAccuracy > 20 {
		fmt.Println("║  📊 LEARNING: Bicameral is training but not yet matching baseline.                  ║")
	} else {
		fmt.Println("║  ⚠️  STRUGGLING: Bicameral architecture needs adjustment.                           ║")
	}

	if results.TasksSolved > 1 {
		fmt.Printf("║  🎯 Tasks Solved: %d (Target was >1, ACHIEVED!)                                      ║\n", results.TasksSolved)
	} else {
		fmt.Printf("║  Tasks Solved: %d (Target was >1)                                                    ║\n", results.TasksSolved)
	}

	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Sample solved tasks
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

func safeGet(slice []float64, idx int) float64 {
	if idx < len(slice) && idx >= 0 {
		return slice[idx]
	}
	return 0
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

func saveResults(results *Results) {
	output := map[string]interface{}{
		"final_accuracy":   results.FinalAccuracy,
		"final_budget":     results.FinalBudget,
		"tasks_solved":     results.TasksSolved,
		"solved_task_ids":  results.SolvedTaskIDs,
		"train_time_sec":   results.TrainTime.Seconds(),
		"accuracy_history": results.AccuracyHistory,
		"budget_history":   results.BudgetHistory,
		"meta": map[string]interface{}{
			"architecture":  "Bicameral (LayerParallel: MHA + LSTM)",
			"epochs":        NumEpochs,
			"batch_size":    BatchSize,
			"learning_rate": LearningRate,
			"budget_scale":  BudgetScale,
			"dmodel":        DModel,
			"lstm_hidden":   LSTMHidden,
			"training_mode": "StepTween",
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test27_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test27_results.json")
}

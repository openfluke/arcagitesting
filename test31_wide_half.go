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

// Test 31 WIDE HALF: HEURISTIC HIVE MIND - WIDE Grid Scatter (HALF SIZE)
//
// Extension of Test 31 HALF with a WIDER hive mind:
//   4x4 Grid (16 brains) instead of 2x2 (4 brains)
//
// Architecture:
//   - Input -> Embedding (16 dim)
//   - Layer 1: LayerParallel (Grid Scatter 4x4) - 16 brains!
//       - Mix of MHA and LSTM brains
//   - Layer 2: Dense Merger (16 * 16 = 256 -> 16)
//   - Layer 3: Output (16 -> 900)
//
// Training: StepTween + ChainRule

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 400
	LearningRate = float32(0.001)
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// Architecture params (HALF SIZE)
	DModel     = 16
	NumHeads   = 2
	LSTMHidden = 16
	GridSize   = 10 // 4x4 grid of brains

	// Grokking Detection
	GrokThreshold = 20.0
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
	GrokEpoch       int
}

func main() {
	numBrains := GridSize * GridSize
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║     Test 31 WIDE HALF: %dx%d GRID HIVE (%d BRAINS) - HALF SIZE                      ║\n", GridSize, GridSize, numBrains)
	fmt.Println("║                                                                                      ║")
	fmt.Printf("║     🐝 %d Brains in a %dx%d Grid Scatter pattern                                    ║\n", numBrains, GridSize, GridSize)
	fmt.Println("║     🧠 Mix of MHA, LSTM, and CNN brains                                             ║")
	fmt.Println("║     📦 DModel=16 | NumHeads=2 | LSTMHidden=16 (HALF)                               ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Training: StepTween + ChainRule | 400 Epochs                                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the WIDE Hive Mind network
	net := createWideHiveMindNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created WIDE Hive Mind Network: %d layers (%d brains)\n", numLayers, numBrains)

	// Initialize training state - NO ChainRule (Pure Heuristic)
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = BudgetScale
	ts.Config.UseChainRule = true
	// Note: NOT setting UseChainRule = true

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
		GrokEpoch:       -1,
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🐝 HEURISTIC HIVE TRAINING BEGINS 🐝")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0
	prevAcc := 0.0
	hasGrokked := false

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

			// TweenStep training (Pure Heuristic - Gap Closing)
			ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)
		}

		// Measure metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.BudgetHistory[epoch] = budget

		// Grokking Detection
		if !hasGrokked && acc > GrokThreshold && prevAcc < GrokThreshold {
			hasGrokked = true
			results.GrokEpoch = epoch + 1
			fmt.Println()
			fmt.Println("  ╔═══════════════════════════════════════════════════════════════════════╗")
			fmt.Printf("  ║  🐝🐝🐝 GROKKING DETECTED 🐝🐝🐝  Epoch %d: %.1f%% → %.1f%%  ║\n", epoch+1, prevAcc, acc)
			fmt.Println("  ║      The Heuristic Hive has awakened!                                ║")
			fmt.Println("  ╚═══════════════════════════════════════════════════════════════════════╝")
			fmt.Println()
		}

		if (epoch+1)%20 == 0 {
			status := ""
			if hasGrokked && acc > 50 {
				status = " 🔥"
			} else if acc > 15 {
				status = " 👀"
			}
			fmt.Printf("  Epoch %3d/%d | Accuracy: %5.1f%% | Budget: %.3f%s\n",
				epoch+1, NumEpochs, acc, budget, status)
		}

		prevAcc = acc
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
// WIDE HIVE MIND NETWORK ARCHITECTURE (4x4 Grid Scatter)
// ============================================================================

func createWideHiveMindNetwork() *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (900 -> DModel=16)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: WIDE GRID SCATTER - 4x4 Hive of Brains (16 brains!)
	parallelLayer := createWideGridScatterHive()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Layer 2: Merger (16 brains * DModel = 256 -> DModel)
	numBrains := GridSize * GridSize
	mergerInputSize := DModel * numBrains
	mergerLayer := nn.InitDenseLayer(mergerInputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 3: Output (DModel -> 900)
	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createWideGridScatterHive() nn.LayerConfig {
	numBrains := GridSize * GridSize
	brains := make([]nn.LayerConfig, numBrains)
	positions := make([]nn.GridPosition, numBrains)

	// Create brains: mix of MHA, LSTM, and CNN
	// - LSTM at every 4th position (i%4 == 1)
	// - CNN at every 3rd position (i%3 == 2)
	// - MHA for the rest
	for i := 0; i < numBrains; i++ {
		row := i / GridSize
		col := i % GridSize

		if i%4 == 1 {
			// LSTM for temporal processing
			brains[i] = createLSTMBrain()
		} else if i%3 == 2 {
			// CNN for local pattern detection
			brains[i] = createCNNBrain()
		} else {
			// MHA for global attention
			brains[i] = createMHABrain()
		}

		positions[i] = nn.GridPosition{
			BranchIndex: i,
			TargetRow:   row,
			TargetCol:   col,
			TargetLayer: 0,
		}
	}

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   GridSize,
		GridOutputCols:   GridSize,
		GridOutputLayers: 1,
		ParallelBranches: brains,
		GridPositions:    positions,
	}

	return parallel
}

func createCNNBrain() nn.LayerConfig {
	// CNN for local pattern detection
	// Input: DModel (treated as 1x1xDModel), Output: DModel
	cnn := nn.LayerConfig{
		Type:          nn.LayerConv2D,
		InputWidth:    1,
		InputHeight:   1,
		InputChannels: DModel,
		KernelSize:    1, // 1x1 convolution
		Stride:        1,
		Padding:       0,
		Filters:       DModel,
		OutputHeight:  DModel,
		OutputWidth:   1,
		Activation:    nn.ActivationLeakyReLU,
	}

	// Initialize CNN weights: [filters][inChannels][kernelH][kernelW]
	kernelTotal := cnn.Filters * DModel * cnn.KernelSize * cnn.KernelSize
	cnn.Kernel = make([]float32, kernelTotal)
	cnn.Bias = make([]float32, cnn.Filters)

	scale := InitScale / float32(math.Sqrt(float64(DModel)))
	initRandom(cnn.Kernel, scale)

	return cnn
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
// Metrics & Utilities
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

// ============================================================================
// Visualization
// ============================================================================

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🐝 HEURISTIC HIVE - FINAL RESULTS 🐝                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║                                                                                      ║\n")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())

	if results.GrokEpoch > 0 {
		fmt.Printf("║   🐝 Grok Epoch:      %d                                                             ║\n", results.GrokEpoch)
	} else {
		fmt.Printf("║   😴 Grok Epoch:      NEVER                                                          ║\n")
	}

	fmt.Printf("║                                                                                      ║\n")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           ACCURACY TIMELINE                                          ║")
	fmt.Println("╠════════════════════╦═════════╦═════════╦═════════╦═════════╦═════════════════════════╣")
	fmt.Println("║     Epoch          ║   80    ║   160   ║   240   ║   320   ║   400                   ║")
	fmt.Println("╠════════════════════╬═════════╬═════════╬═════════╬═════════╬═════════════════════════╣")
	fmt.Printf("║ Heuristic Hive     ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%%                 ║\n",
		safeGet(results.AccuracyHistory, 79), safeGet(results.AccuracyHistory, 159),
		safeGet(results.AccuracyHistory, 239), safeGet(results.AccuracyHistory, 319),
		results.FinalAccuracy)
	fmt.Println("╚════════════════════╩═════════╩═════════╩═════════╩═════════╩═════════════════════════╝")

	// Comparison
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        HEURISTIC HIVE HYPOTHESIS VERDICT                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Baseline (Test 27 Bicameral): 53.2% | Test 30 (Grid+Chain): 6.8%                    ║")
	fmt.Println("║  ─────────────────────────────────────────────────────────────────────────────────── ║")

	if results.FinalAccuracy > 53.2 {
		fmt.Println("║  ✨ BREAKTHROUGH: Heuristic Hive BEATS all baselines!                               ║")
		fmt.Println("║     → Grid Scatter + StepTween is the winning combination!                          ║")
	} else if results.FinalAccuracy > 45 {
		fmt.Println("║  ⚡ COMPETITIVE: Heuristic Hive matches baseline performance.                       ║")
		fmt.Println("║     → Proves Heuristic works where Gradient fails with grid_scatter.                ║")
	} else if results.FinalAccuracy > 20 {
		fmt.Println("║  📊 LEARNING: Significant improvement over Test 30 (6.8%).                          ║")
		fmt.Println("║     → Confirms Heuristic is viable with grid_scatter.                               ║")
	} else {
		fmt.Println("║  ❌ UNEXPECTED: Heuristic also struggled with grid_scatter.                         ║")
		fmt.Println("║     → May indicate a fundamental issue with the architecture.                       ║")
	}

	if results.TasksSolved > 2 {
		fmt.Printf("║  🎯 Tasks Solved: %d (MORE than baseline 2!)                                         ║\n", results.TasksSolved)
	}

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
		"grok_epoch":       results.GrokEpoch,
		"accuracy_history": results.AccuracyHistory,
		"budget_history":   results.BudgetHistory,
		"meta": map[string]interface{}{
			"architecture":  fmt.Sprintf("Wide Hive HALF (%dx%d Grid Scatter, %d brains) DModel=16", GridSize, GridSize, GridSize*GridSize),
			"epochs":        NumEpochs,
			"batch_size":    BatchSize,
			"learning_rate": LearningRate,
			"budget_scale":  BudgetScale,
			"dmodel":        DModel,
			"training_mode": "StepTween (Heuristic)",
			"combine_mode":  "grid_scatter",
			"hypothesis":    "Heuristic works where Gradient fails",
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test31_wide_half_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test31_wide_half_results.json")
}

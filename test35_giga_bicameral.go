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

// Test 35: QUINTUPLE GRID SCATTER HIVE - 5 Sequential 2x2 Spatial Brain Layers
//
// Based on Test 30's Grid Scatter architecture, but stacking 5 parallel layers
// in sequence to create a deep hierarchical hive mind.
//
// Architecture:
//   - Input: 30x30 Grid (900 floats) -> Embedding (32 dim)
//   - Layer 1-5: 5x LayerParallel with 'grid_scatter' (2x2 Hive each)
//       - Each layer: Pos(0,0): MHA | Pos(0,1): LSTM | Pos(1,0): MHA | Pos(1,1): MHA
//   - Layer 6: Dense Merger
//   - Layer 7: Output (900)
//
// Training: StepTweenChain (Gradient-based)

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 1400
	LearningRate = float32(1000.5)
	InitScale    = float32(0.5)
	BudgetScale  = float32(1.2)

	// Architecture params (smaller per brain)
	DModel     = 32 // Smaller to fit 4 brains per layer
	NumHeads   = 4
	LSTMHidden = 32

	// Number of stacked parallel layers
	NumParallelLayers = 5

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
	GrokEpoch       int // -1 if no grokking detected
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 35: QUINTUPLE GRID SCATTER HIVE - 5 Sequential 2x2 Brain Layers            ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🧠 Each Layer has 4 Brains in a 2x2 Grid:                                       ║")
	fmt.Println("║        - Brain[0,0] (MHA):  Global Spatial Patterns                                 ║")
	fmt.Println("║        - Brain[0,1] (LSTM): Temporal/Sequential Logic                               ║")
	fmt.Println("║        - Brain[1,0] (MHA):  Spatial Backup                                          ║")
	fmt.Println("║        - Brain[1,1] (MHA):  Redundancy                                              ║")
	fmt.Println("║     🔗 CombineMode: grid_scatter (Spatial routing)                                  ║")
	fmt.Println("║     📚 Depth: 5 Parallel Layers Stacked Sequentially                                ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Training: StepTweenChain (Gradient-based) | 1400 Epochs                         ║")
	fmt.Println("║     Goal: Does stacking 5 grid_scatter layers create emergent behavior?             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the Quintuple Hive Mind network
	net := createQuintupleHiveMindNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Quintuple Hive Mind Network: %d layers (5 parallel hives)\n", numLayers)

	// Initialize training state with Chain Rule
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
		GrokEpoch:       -1,
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🐝🐝🐝🐝🐝 QUINTUPLE HIVE MIND TRAINING 🐝🐝🐝🐝🐝")
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
			output := state.GetOutput()

			// ChainRule training path
			ts.ForwardPass(net, sample.Input)
			applyChainRuleUpdate(ts, net, sample, output, LearningRate)
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
			fmt.Printf("  ║  🐝🐝🐝🐝🐝 GROKKING DETECTED 🐝🐝🐝🐝🐝  Epoch %d: %.1f%% → %.1f%%  ║\n", epoch+1, prevAcc, acc)
			fmt.Println("  ║      The Quintuple Hive Mind has awakened!                            ║")
			fmt.Println("  ╚═══════════════════════════════════════════════════════════════════════╝")
			fmt.Println()
		}

		if (epoch+1)%20 == 0 {
			status := ""
			if hasGrokked && acc > 40 {
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
// Chain Rule Training
// ============================================================================

func applyChainRuleUpdate(ts *nn.TweenState, net *nn.Network, sample Sample, output []float32, lr float32) {
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

// ============================================================================
// QUINTUPLE HIVE MIND NETWORK ARCHITECTURE (5x Grid Scatter 2x2)
// ============================================================================

func createQuintupleHiveMindNetwork() *nn.Network {
	// Total layers: Input Embed + 5x Parallel(GridScatter) + Merger + Output = 8 layers
	totalLayers := 2 + NumParallelLayers + 1 // input + parallels + merger + output
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (900 -> DModel=32)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layers 1-5: 5x GRID SCATTER - 2x2 Hive of Brains each
	for i := 0; i < NumParallelLayers; i++ {
		parallelLayer := createGridScatterHive(i)
		net.SetLayer(0, 0, layerIdx, parallelLayer)
		layerIdx++
	}

	// Layer 6: Merger
	// Each brain outputs DModel=32, and we have 4 brains in a 2x2 grid
	// Grid scatter reorganizes them spatially, but total output is still 4*DModel = 128
	mergerInputSize := DModel * 4 // 4 brains * 32 each
	mergerLayer := nn.InitDenseLayer(mergerInputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 7: Output (DModel -> 900)
	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createGridScatterHive(hiveIndex int) nn.LayerConfig {
	// Create the four brains for this hive
	brain00 := createMHABrain()  // Pos(0,0): Spatial
	brain01 := createLSTMBrain() // Pos(0,1): Temporal
	brain10 := createMHABrain()  // Pos(1,0): Spatial Backup
	brain11 := createMHABrain()  // Pos(1,1): Redundancy

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   2,
		GridOutputCols:   2,
		GridOutputLayers: 1,
		ParallelBranches: []nn.LayerConfig{
			brain00,
			brain01,
			brain10,
			brain11,
		},
		GridPositions: []nn.GridPosition{
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0}, // MHA -> (0,0)
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0}, // LSTM -> (0,1)
			{BranchIndex: 2, TargetRow: 1, TargetCol: 0, TargetLayer: 0}, // MHA2 -> (1,0)
			{BranchIndex: 3, TargetRow: 1, TargetCol: 1, TargetLayer: 0}, // MHA3 -> (1,1)
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
	fmt.Println("║                      🐝🐝🐝🐝🐝 QUINTUPLE HIVE MIND - FINAL RESULTS 🐝🐝🐝🐝🐝                ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║                                                                                      ║\n")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
	fmt.Printf("║   Parallel Layers:    %d (5x Grid Scatter 2x2)                                       ║\n", NumParallelLayers)

	if results.GrokEpoch > 0 {
		fmt.Printf("║   🐝 Grok Epoch:      %d (Deep hierarchy worked!)                                   ║\n", results.GrokEpoch)
	} else {
		fmt.Printf("║   😴 Grok Epoch:      NEVER                                                          ║\n")
	}

	fmt.Printf("║                                                                                      ║\n")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           ACCURACY TIMELINE                                          ║")
	fmt.Println("╠════════════════════╦═════════╦═════════╦═════════╦═════════╦═════════════════════════╣")
	fmt.Println("║     Epoch          ║   280   ║   560   ║   840   ║   1120  ║   1400                  ║")
	fmt.Println("╠════════════════════╬═════════╬═════════╬═════════╬═════════╬═════════════════════════╣")
	fmt.Printf("║ Quintuple Hive     ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%%                 ║\n",
		safeGet(results.AccuracyHistory, 279), safeGet(results.AccuracyHistory, 559),
		safeGet(results.AccuracyHistory, 839), safeGet(results.AccuracyHistory, 1119),
		results.FinalAccuracy)
	fmt.Println("╚════════════════════╩═════════╩═════════╩═════════╩═════════╩═════════════════════════╝")

	// Comparison
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        QUINTUPLE HIVE HYPOTHESIS VERDICT                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Baseline (Test 30 Single Grid Scatter): Compared to 1 parallel layer               ║")
	fmt.Println("║  ─────────────────────────────────────────────────────────────────────────────────── ║")

	if results.GrokEpoch > 0 && results.FinalAccuracy > 25 {
		fmt.Printf("║  ✨ HYPOTHESIS CONFIRMED: Deep hive stacking improved results!                      ║\n")
		fmt.Println("║     → 5 sequential grid_scatter layers enable deeper pattern learning!             ║")
	} else if results.GrokEpoch > 0 {
		fmt.Printf("║  ⚡ PARTIAL: Grokking occurred at epoch %d                                          ║\n", results.GrokEpoch)
	} else if results.FinalAccuracy > 15 {
		fmt.Println("║  📊 SLOW LEARNING: Some progress but no clear grok transition                       ║")
	} else {
		fmt.Println("║  ❌ HYPOTHESIS NEEDS TUNING: Deep stacking may need different hyperparams          ║")
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
			"architecture":    "Quintuple Hive Mind (5x 2x2 Grid Scatter: MHA+LSTM+MHA+MHA each)",
			"parallel_layers": NumParallelLayers,
			"epochs":          NumEpochs,
			"batch_size":      BatchSize,
			"learning_rate":   LearningRate,
			"budget_scale":    BudgetScale,
			"dmodel":          DModel,
			"training_mode":   "StepTweenChain (Gradient)",
			"combine_mode":    "grid_scatter",
			"hypothesis":      "Stacking 5 grid_scatter layers enables deeper hierarchical learning",
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test35_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test35_results.json")
}

var _ = argmax // suppress unused warning

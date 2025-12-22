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

// Test 28: LONG CHAIN - The "Sleeper" Gradient Theory
//
// Hypothesis: StepTweenChain (Gradient) needs 300+ epochs to "Grok"
// The Heuristic model required ~140 epochs of silence before learning;
// The Gradient model likely needs even more time for deep layer alignment.
//
// Architecture: Bicameral Native (proven to solve 2 tasks in Test 27)
// Training: StepTweenChain with high LinkBudgetScale
// Epochs: 600 (The Long Haul)

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 600 // THE LONG HAUL - Give gradients time to align
	LearningRate = float32(0.001)
	InitScale    = float32(0.5)
	BudgetScale  = float32(2.0) // HIGH - Drive tiny gradients through

	// Architecture params
	DModel     = 64
	NumHeads   = 4
	LSTMHidden = 64
	SeqLength  = 1

	// Wake detection threshold
	WakeThreshold = 20.0 // If acc jumps from ~8% to >20%, it's awake
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
	WakeUpEpoch     int // -1 if never woke up
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 28: LONG CHAIN - The \"Sleeper\" Gradient Theory                             ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🧪 HYPOTHESIS: Gradients need 300+ epochs to align deep layers                  ║")
	fmt.Println("║     🏗️  Architecture: Bicameral (MHA + LSTM) - proven at 53.2%                       ║")
	fmt.Println("║     ⚡ Training: StepTweenChain (Gradient path, not Heuristic!)                     ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Config:                                                                          ║")
	fmt.Println("║       • Epochs: 600 (The Long Haul)                                                  ║")
	fmt.Println("║       • LinkBudgetScale: 2.0 (High - drive gradients through)                       ║")
	fmt.Println("║       • UseChainRule: true                                                           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the bicameral network (same as Test 27)
	net := createBicameralNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Bicameral Network: %d layers\n", numLayers)

	// Initialize training state with CHAIN RULE CONFIG
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true           // CRITICAL: Enable gradient path
	ts.Config.LinkBudgetScale = BudgetScale // High scale to amplify weak gradients

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
		WakeUpEpoch:     -1,
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     ⚡ LONG CHAIN TRAINING BEGINS ⚡")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0
	prevAcc := 0.0
	hasWoken := false

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

		// Wake detection
		if !hasWoken && acc > WakeThreshold && prevAcc < WakeThreshold {
			hasWoken = true
			results.WakeUpEpoch = epoch + 1
			fmt.Println()
			fmt.Println("  ╔═══════════════════════════════════════════════════════════════════════╗")
			fmt.Printf("  ║  🚨🚨🚨 WAKE UP DETECTED 🚨🚨🚨  Epoch %d: %.1f%% → %.1f%%  ║\n", epoch+1, prevAcc, acc)
			fmt.Println("  ║      The Gradient model has awakened! Deep layers are aligning!      ║")
			fmt.Println("  ╚═══════════════════════════════════════════════════════════════════════╝")
			fmt.Println()
		}

		if (epoch+1)%20 == 0 {
			status := ""
			if acc > WakeThreshold && !hasWoken {
				status = " 👀"
			} else if hasWoken && acc > 50 {
				status = " 🔥"
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
// BICAMERAL NETWORK ARCHITECTURE (Same as Test 27)
// ============================================================================

func createBicameralNetwork() *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: PARALLEL SPLIT - The Bicameral Core
	parallelLayer := createParallelBicameralLayer()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Layer 2: Merger
	mergerLayer := nn.InitDenseLayer(DModel*2, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 3: Output
	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createParallelBicameralLayer() nn.LayerConfig {
	rightBrain := createRightBrainBranch()
	leftBrain := createLeftBrainBranch()

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "concat",
		ParallelBranches: []nn.LayerConfig{
			rightBrain,
			leftBrain,
		},
	}

	return parallel
}

func createRightBrainBranch() nn.LayerConfig {
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

func createLeftBrainBranch() nn.LayerConfig {
	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: DModel,
		HiddenSize:   LSTMHidden,
		SeqLength:    SeqLength,
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
	fmt.Println("║                      ⚡ LONG CHAIN - FINAL RESULTS ⚡                                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║                                                                                      ║\n")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
	fmt.Printf("║                                                                                      ║\n")

	if results.WakeUpEpoch > 0 {
		fmt.Printf("║   🚨 Wake-Up Epoch:   %d (Gradients started working!)                               ║\n", results.WakeUpEpoch)
	} else {
		fmt.Printf("║   😴 Wake-Up Epoch:   NEVER (Stayed asleep)                                          ║\n")
	}

	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           ACCURACY TIMELINE                                          ║")
	fmt.Println("╠════════════════════╦═════════╦═════════╦═════════╦═════════╦═════════╦═══════════════╣")
	fmt.Println("║     Epoch          ║   100   ║   200   ║   300   ║   400   ║   500   ║   600         ║")
	fmt.Println("╠════════════════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═══════════════╣")
	fmt.Printf("║ ChainRule (Grad)   ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%%       ║\n",
		safeGet(results.AccuracyHistory, 99), safeGet(results.AccuracyHistory, 199),
		safeGet(results.AccuracyHistory, 299), safeGet(results.AccuracyHistory, 399),
		safeGet(results.AccuracyHistory, 499), results.FinalAccuracy)
	fmt.Println("╚════════════════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═══════════════╝")

	// Analysis
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           SLEEPER THEORY VERDICT                                     ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	if results.WakeUpEpoch > 0 && results.FinalAccuracy > 40 {
		fmt.Println("║  ✅ THEORY CONFIRMED: Gradients CAN learn if given enough time!                     ║")
		fmt.Printf("║     The network woke up at epoch %d and reached %.1f%% accuracy.                    ║\n", results.WakeUpEpoch, results.FinalAccuracy)
		fmt.Println("║     → StepTweenChain is viable for deep networks with patience!                     ║")
	} else if results.WakeUpEpoch > 0 {
		fmt.Println("║  ⚡ PARTIAL: Gradients woke up but didn't reach heuristic performance.              ║")
		fmt.Printf("║     Woke at epoch %d, final acc %.1f%% (vs heuristic ~54%%)                          ║\n", results.WakeUpEpoch, results.FinalAccuracy)
	} else if results.FinalAccuracy > 15 {
		fmt.Println("║  😐 SLOW LEARNER: Some learning but no clear \"wake up\" moment.                     ║")
		fmt.Println("║     May need even more epochs or different hyperparameters.                         ║")
	} else {
		fmt.Println("║  ❌ THEORY REJECTED: 600 epochs wasn't enough. Gradients stayed asleep.             ║")
		fmt.Println("║     The Chain Rule path may have fundamental issues with this architecture.         ║")
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
		"wake_up_epoch":    results.WakeUpEpoch,
		"accuracy_history": results.AccuracyHistory,
		"budget_history":   results.BudgetHistory,
		"meta": map[string]interface{}{
			"architecture":   "Bicameral (LayerParallel: MHA + LSTM)",
			"epochs":         NumEpochs,
			"batch_size":     BatchSize,
			"learning_rate":  LearningRate,
			"budget_scale":   BudgetScale,
			"training_mode":  "StepTweenChain (Gradient)",
			"use_chain_rule": true,
			"hypothesis":     "Sleeper Theory - Gradients need 300+ epochs",
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test28_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test28_results.json")
}

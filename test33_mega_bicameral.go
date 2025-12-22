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

// Test 33: MEGA BICAMERAL - Double the Capacity
//
// Test 32 hit 52.6% with DModel=128 - tantalizingly close to 53.2%!
// Let's double everything to push past the barrier.
//
// Architecture:
//   - DModel = 256 (2x test32, 4x original)
//   - NumHeads = 16 (scaled proportionally)
//   - Dual MHA branches
//   - StepTween training (proven stable)
//   - 600 epochs

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 600
	LearningRate = float32(0.001)
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// MEGA Architecture - Double test32
	DModel   = 256 // 4x original (64), 2x test32 (128)
	NumHeads = 16  // Scaled proportionally
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
	PeakAccuracy    float64
	PeakEpoch       int
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 33: MEGA BICAMERAL - Double the Capacity                                   ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🎯 Goal: Break 53.2% with maximum capacity                                      ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Strategy:                                                                        ║")
	fmt.Println("║       • MEGA model: DModel=256 (2x test32, 4x original)                             ║")
	fmt.Println("║       • 16 Attention Heads (scaled proportionally)                                  ║")
	fmt.Println("║       • DUAL MHA: Both branches are attention                                       ║")
	fmt.Println("║       • StepTween: Proven stable heuristic training                                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Training: StepTween (Heuristic) | DModel=256 | NumHeads=16                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the Mega Bicameral network
	net := createMegaBicameralNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created MEGA Bicameral Network: %d layers, DModel=%d, Heads=%d\n", numLayers, DModel, NumHeads)

	// Initialize training state
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
		PeakAccuracy:    0,
		PeakEpoch:       0,
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🧠 MEGA BICAMERAL TRAINING BEGINS 🧠")
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

			// StepTween training (Pure Heuristic)
			ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)
		}

		// Measure metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.BudgetHistory[epoch] = budget

		// Track peak
		if acc > results.PeakAccuracy {
			results.PeakAccuracy = acc
			results.PeakEpoch = epoch + 1
		}

		if (epoch+1)%20 == 0 {
			status := ""
			if acc > 53.2 {
				status = " 🏆 NEW RECORD!"
			} else if acc > 52 {
				status = " 🔥🔥"
			} else if acc > 50 {
				status = " 🔥"
			} else if acc > 40 {
				status = " 👀"
			}
			fmt.Printf("  Epoch %3d/%d | Accuracy: %5.1f%% | Budget: %.3f%s\n",
				epoch+1, NumEpochs, acc, budget, status)
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
// MEGA BICAMERAL ARCHITECTURE
// ============================================================================

func createMegaBicameralNetwork() *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (900 -> 256)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: PARALLEL DUAL MHA
	parallelLayer := createDualMHALayer()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Layer 2: Merger (512 -> 256)
	mergerLayer := nn.InitDenseLayer(DModel*2, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 3: Output (256 -> 900)
	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createDualMHALayer() nn.LayerConfig {
	brain1 := createMHABrain()
	brain2 := createMHABrain()

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "concat",
		ParallelBranches: []nn.LayerConfig{
			brain1,
			brain2,
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
	fmt.Println("║                      🧠 MEGA BICAMERAL - FINAL RESULTS 🧠                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║                                                                                      ║\n")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Peak Accuracy:      %5.1f%% (Epoch %d)                                             ║\n", results.PeakAccuracy, results.PeakEpoch)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
	fmt.Printf("║                                                                                      ║\n")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                     SCALING COMPARISON                                               ║")
	fmt.Println("╠═════════════════════════╦═══════════════╦═══════════════╦════════════════════════════╣")
	fmt.Println("║ Model                   ║ DModel        ║ Peak Acc      ║ Status                     ║")
	fmt.Println("╠═════════════════════════╬═══════════════╬═══════════════╬════════════════════════════╣")
	fmt.Println("║ Test 27 (Original)      ║ 64            ║ 53.2%         ║ Baseline                   ║")
	fmt.Println("║ Test 32 (Wide)          ║ 128           ║ ~52.6%        ║ Close                      ║")
	fmt.Printf("║ Test 33 (MEGA)          ║ 256           ║ %5.1f%%        ║ ", results.PeakAccuracy)
	if results.PeakAccuracy > 53.2 {
		fmt.Println("🏆 NEW RECORD!           ║")
	} else if results.PeakAccuracy > 52 {
		fmt.Println("Close but no cigar       ║")
	} else {
		fmt.Println("Needs work               ║")
	}
	fmt.Println("╚═════════════════════════╩═══════════════╩═══════════════╩════════════════════════════╝")

	// Verdict
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              SCALING VERDICT                                         ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	if results.PeakAccuracy > 53.2 {
		fmt.Printf("║  🏆 SCALING WORKS! MEGA model reached %.1f%% (Peak at Epoch %d)!                    ║\n", results.PeakAccuracy, results.PeakEpoch)
		fmt.Println("║     → Bigger IS better for ARC-AGI! Need more capacity to learn patterns.           ║")
	} else if results.PeakAccuracy >= 52 {
		fmt.Println("║  📊 Diminishing returns: Doubling size didn't double performance.                   ║")
		fmt.Println("║     → The bottleneck might be architecture, not capacity.                           ║")
		fmt.Println("║     → Consider: deeper layers, different branch types, or training strategy.        ║")
	} else {
		fmt.Println("║  ⚠️  Larger model underperformed - possibly overfitting or underfitting.            ║")
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
		"peak_accuracy":    results.PeakAccuracy,
		"peak_epoch":       results.PeakEpoch,
		"final_budget":     results.FinalBudget,
		"tasks_solved":     results.TasksSolved,
		"solved_task_ids":  results.SolvedTaskIDs,
		"train_time_sec":   results.TrainTime.Seconds(),
		"accuracy_history": results.AccuracyHistory,
		"budget_history":   results.BudgetHistory,
		"meta": map[string]interface{}{
			"architecture":  "MEGA Bicameral (Dual MHA, DModel=256)",
			"epochs":        NumEpochs,
			"batch_size":    BatchSize,
			"learning_rate": LearningRate,
			"budget_scale":  BudgetScale,
			"dmodel":        DModel,
			"num_heads":     NumHeads,
			"training_mode": "StepTween (Heuristic)",
			"hypothesis":    "Scaling to 256 dim breaks the 53.2% barrier",
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test33_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test33_results.json")
}

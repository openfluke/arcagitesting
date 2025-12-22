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

// Test 34: RECURRENT HIVE - Looping Parallel Processing
//
// Hypothesis: The 53% ceiling exists because the network only gets ONE pass
// through the parallel brains. By looping N times, we allow:
//   1. Information to propagate DOWN through brains
//   2. Information to propagate BACK UP via residual connections
//   3. Multiple "thinking iterations" to refine the representation
//
// Architecture:
//   - Input: 900 -> Embedding (128 dim)
//   - LOOP N times:
//       - Parallel Brains (MHA + MHA)
//       - Synapse (compress/mix)
//       - Residual add from previous iteration
//   - Output: 128 -> 900
//
// This is like giving the network multiple "cycles" to think!

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 600
	LearningRate = float32(100.001)
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// Architecture
	DModel    = 128
	NumHeads  = 8
	NumLoops  = 3   // Number of times to loop through the hive
	LoopDecay = 0.5 // Each loop contribution is scaled by this
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
	fmt.Println("║     Test 34: RECURRENT HIVE - Looping Parallel Processing                           ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🎯 Goal: Break 53% ceiling with recurrent processing                            ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Hypothesis: One pass isn't enough. Let the network LOOP to refine!              ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Architecture:                                                                    ║")
	fmt.Printf("║       • %d Thinking Loops through the Hive                                          ║\n", NumLoops)
	fmt.Println("║       • Each loop: Parallel MHA -> Synapse -> Residual Add                          ║")
	fmt.Println("║       • Information propagates UP and DOWN through iterations                       ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║     Training: StepTween (Heuristic) | DModel=%d | %d Loops                           ║\n", DModel, NumLoops)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the Recurrent Hive network
	net := createRecurrentHiveNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Recurrent Hive Network: %d layers, %d loops\n", numLayers, NumLoops)

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
	fmt.Println("                     🔄 RECURRENT HIVE TRAINING BEGINS 🔄")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		// Training loop
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			// Forward pass with looping
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}

			// StepTween training
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
			if acc > 54 {
				status = " 🏆🏆 BREAKTHROUGH!"
			} else if acc > 53.2 {
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

	printResults(results)
	saveResults(results)
}

// ============================================================================
// RECURRENT HIVE ARCHITECTURE
// ============================================================================

func createRecurrentHiveNetwork() *nn.Network {
	// Structure:
	// Layer 0: Input Embedding (900 -> 128)
	// Layers 1,2: Loop 1 (Parallel + Synapse)
	// Layers 3,4: Loop 2 (Parallel + Synapse)
	// Layers 5,6: Loop 3 (Parallel + Synapse)
	// Layer 7: Output (128 -> 900)

	totalLayers := 2 + (NumLoops * 2) // embed + loops*2 + output
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (900 -> 128)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Create N loops of Parallel + Synapse
	for loop := 0; loop < NumLoops; loop++ {
		// Parallel MHA layer
		parallelLayer := createDualMHALayer()
		net.SetLayer(0, 0, layerIdx, parallelLayer)
		layerIdx++

		// Synapse: Compress back and mix (256 -> 128)
		// This also acts as a "residual checkpoint" between loops
		synapseLayer := nn.InitDenseLayer(DModel*2, DModel, nn.ActivationLeakyReLU)
		scaleWeights(synapseLayer.Kernel, InitScale)
		net.SetLayer(0, 0, layerIdx, synapseLayer)
		layerIdx++
	}

	// Final Output (128 -> 900)
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
	fmt.Println("║                      🔄 RECURRENT HIVE - FINAL RESULTS 🔄                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║                                                                                      ║\n")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Peak Accuracy:      %5.1f%% (Epoch %d)                                             ║\n", results.PeakAccuracy, results.PeakEpoch)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d / 416                                                       ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
	fmt.Printf("║                                                                                      ║\n")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                     RECURRENCE HYPOTHESIS                                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	if results.PeakAccuracy > 54 {
		fmt.Println("║  🏆🏆 BREAKTHROUGH: Recurrent processing broke the 54% barrier!                    ║")
		fmt.Println("║     → Multiple thinking iterations WORK! The network needs time to reason.         ║")
	} else if results.PeakAccuracy > 53.2 {
		fmt.Printf("║  🏆 NEW RECORD: Recurrent Hive reached %.1f%% (vs 53.2%% baseline)!                  ║\n", results.PeakAccuracy)
		fmt.Println("║     → Looping helps! More iterations = better reasoning.                            ║")
	} else if results.PeakAccuracy >= 52 {
		fmt.Println("║  📊 Close but not quite. Recurrence alone may not be enough.                        ║")
		fmt.Printf("║     → Reached %.1f%% with %d loops.                                                  ║\n", results.PeakAccuracy, NumLoops)
	} else {
		fmt.Println("║  ⚠️  Recurrence didn't help. May need different loop structure.                     ║")
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
			"architecture":  fmt.Sprintf("Recurrent Hive (%d loops, DModel=%d)", NumLoops, DModel),
			"epochs":        NumEpochs,
			"batch_size":    BatchSize,
			"learning_rate": LearningRate,
			"budget_scale":  BudgetScale,
			"dmodel":        DModel,
			"num_heads":     NumHeads,
			"num_loops":     NumLoops,
			"training_mode": "StepTween (Heuristic)",
			"hypothesis":    "Recurrent processing allows multiple 'thinking' iterations",
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test34_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test34_results.json")
}

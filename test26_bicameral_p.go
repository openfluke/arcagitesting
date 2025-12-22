package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 26: BICAMERAL-P - Split-Brain Training for ARC-AGI
//
// Two Networks, One Goal:
//   - netHeuristic (StepTween): Spatial/physics intelligence (54% acc)
//   - netMath (StepBP): Logical/counting intelligence (7% acc)
//
// Ensemble: Average predictions to get best of both worlds
// Goal: Union of Solved Tasks > Any Single Method

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 400
	BatchSize   = 100
	NumEpochs   = 200
	InitScale   = float32(0.5)

	// Brain-specific configs
	HeuristicLR       = float32(0.001)
	HeuristicGradClip = float32(0.1)
	HeuristicBudget   = float32(0.8)
	MathLR            = float32(0.001)
	MathGradClip      = float32(0.1)
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

type BrainStats struct {
	Name            string
	FinalAccuracy   float64
	FinalBudget     float32
	TasksSolved     int
	SolvedTaskIDs   []string
	AccuracyHistory []float64
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 26: BICAMERAL-P - Split-Brain Training                                     ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🧠 netHeuristic: StepTween  - Spatial/Physics Intelligence                      ║")
	fmt.Println("║     🧮 netMath:      StepBP     - Logical/Counting Intelligence                     ║")
	fmt.Println("║     🔗 Ensemble:     Average    - Best of Both Worlds                               ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Architecture: Attn-21L × 2 | Epochs: 200 | Batch: 100                           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Initialize the two brains
	netHeuristic := createMHAStabilized(21)
	netMath := createMHAStabilized(21)

	stateHeuristic := netHeuristic.InitStepState(InputSize)
	stateMath := netMath.InitStepState(InputSize)

	tsHeuristic := nn.NewTweenState(netHeuristic, nil)
	tsHeuristic.Config.LinkBudgetScale = HeuristicBudget

	numLayers := netHeuristic.TotalLayers()

	// Stats tracking
	heuristicHistory := make([]float64, NumEpochs)
	mathHistory := make([]float64, NumEpochs)
	ensembleHistory := make([]float64, NumEpochs)

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🧠 SPLIT-BRAIN TRAINING BEGINS 🧠")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		// Train both brains on the same batch
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			// Train Heuristic (StepTween)
			stateHeuristic.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				netHeuristic.StepForward(stateHeuristic)
			}
			tsHeuristic.TweenStep(netHeuristic, sample.Input, argmax(sample.Target), len(sample.Target), HeuristicLR)

			// Train Math (StepBP)
			stateMath.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				netMath.StepForward(stateMath)
			}
			output := stateMath.GetOutput()
			grad := computeGradient(output, sample.Target, MathGradClip)
			netMath.StepBackward(stateMath, grad)
			netMath.ApplyGradients(MathLR)
		}

		// Measure all three: Heuristic, Math, Ensemble
		heuristicAcc := measureAccuracy(netHeuristic, evalSamples, numLayers, stateHeuristic)
		mathAcc := measureAccuracy(netMath, evalSamples, numLayers, stateMath)
		ensembleAcc := measureEnsembleAccuracy(netHeuristic, netMath, evalSamples, numLayers, stateHeuristic, stateMath)

		heuristicHistory[epoch] = heuristicAcc
		mathHistory[epoch] = mathAcc
		ensembleHistory[epoch] = ensembleAcc

		if (epoch+1)%20 == 0 {
			fmt.Printf("  Epoch %3d/%d | 🧠 Heuristic: %5.1f%% | 🧮 Math: %5.1f%% | 🔗 Ensemble: %5.1f%%\n",
				epoch+1, NumEpochs, heuristicAcc, mathAcc, ensembleAcc)
		}
	}

	trainTime := time.Since(start)
	fmt.Printf("\n✅ Training complete in %.1fs\n", trainTime.Seconds())

	// Final evaluation with task tracking
	heuristicSolved, heuristicIDs := measureSolvedTasks(netHeuristic, evalSamples, numLayers, stateHeuristic)
	mathSolved, mathIDs := measureSolvedTasks(netMath, evalSamples, numLayers, stateMath)
	ensembleSolved, ensembleIDs := measureEnsembleSolvedTasks(netHeuristic, netMath, evalSamples, numLayers, stateHeuristic, stateMath)

	heuristic := &BrainStats{
		Name:            "Heuristic",
		FinalAccuracy:   heuristicHistory[NumEpochs-1],
		TasksSolved:     heuristicSolved,
		SolvedTaskIDs:   heuristicIDs,
		AccuracyHistory: heuristicHistory,
	}
	mathBrain := &BrainStats{
		Name:            "Math",
		FinalAccuracy:   mathHistory[NumEpochs-1],
		TasksSolved:     mathSolved,
		SolvedTaskIDs:   mathIDs,
		AccuracyHistory: mathHistory,
	}
	ensemble := &BrainStats{
		Name:            "Ensemble",
		FinalAccuracy:   ensembleHistory[NumEpochs-1],
		TasksSolved:     ensembleSolved,
		SolvedTaskIDs:   ensembleIDs,
		AccuracyHistory: ensembleHistory,
	}

	// Print results
	printFinalResults(heuristic, mathBrain, ensemble)
	printVennDiagram(heuristic, mathBrain, ensemble)
	saveResults(heuristic, mathBrain, ensemble, trainTime)
}

// ============================================================================
// Training Helpers
// ============================================================================

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
// Accuracy Measurement
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

func measureEnsembleAccuracy(netH, netM *nn.Network, samples []Sample, numLayers int, stateH, stateM *nn.StepState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		outH := getOutput(netH, sample.Input, numLayers, stateH)
		outM := getOutput(netM, sample.Input, numLayers, stateM)

		// Ensemble: average
		ensemble := make([]float32, len(outH))
		for i := range ensemble {
			if i < len(outM) {
				ensemble[i] = (outH[i] + outM[i]) / 2.0
			} else {
				ensemble[i] = outH[i]
			}
		}

		for r := 0; r < sample.Height; r++ {
			for c := 0; c < sample.Width; c++ {
				idx := r*MaxGridSize + c
				if idx < len(ensemble) && idx < len(sample.Target) {
					pred := clampInt(int(math.Round(float64(ensemble[idx])*9.0)), 0, 9)
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

func measureEnsembleSolvedTasks(netH, netM *nn.Network, samples []Sample, numLayers int, stateH, stateM *nn.StepState) (int, []string) {
	solved := 0
	solvedIDs := []string{}
	for _, sample := range samples {
		outH := getOutput(netH, sample.Input, numLayers, stateH)
		outM := getOutput(netM, sample.Input, numLayers, stateM)

		// Ensemble: average
		ensemble := make([]float32, len(outH))
		for i := range ensemble {
			if i < len(outM) {
				ensemble[i] = (outH[i] + outM[i]) / 2.0
			} else {
				ensemble[i] = outH[i]
			}
		}

		if isTaskSolved(ensemble, sample) {
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

// ============================================================================
// Visualization
// ============================================================================

func printFinalResults(heuristic, mathBrain, ensemble *BrainStats) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🧠 BICAMERAL-P FINAL RESULTS 🧠                                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                      ║")
	fmt.Printf("║   🧠 Heuristic (StepTween):  Accuracy = %5.1f%% | Tasks Solved = %-3d                ║\n", heuristic.FinalAccuracy, heuristic.TasksSolved)
	fmt.Printf("║   🧮 Math (StepBP):          Accuracy = %5.1f%% | Tasks Solved = %-3d                 ║\n", mathBrain.FinalAccuracy, mathBrain.TasksSolved)
	fmt.Printf("║   🔗 Ensemble (Average):     Accuracy = %5.1f%% | Tasks Solved = %-3d                ║\n", ensemble.FinalAccuracy, ensemble.TasksSolved)
	fmt.Println("║                                                                                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           ACCURACY TIMELINE                                          ║")
	fmt.Println("╠════════════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═════════════════╣")
	fmt.Println("║ Method             ║   Ep 40   ║   Ep 80   ║  Ep 120   ║  Ep 160   ║   Ep 200        ║")
	fmt.Println("╠════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═════════════════╣")
	fmt.Printf("║ 🧠 Heuristic       ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%         ║\n",
		safeGet(heuristic.AccuracyHistory, 39), safeGet(heuristic.AccuracyHistory, 79),
		safeGet(heuristic.AccuracyHistory, 119), safeGet(heuristic.AccuracyHistory, 159), heuristic.FinalAccuracy)
	fmt.Printf("║ 🧮 Math            ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%         ║\n",
		safeGet(mathBrain.AccuracyHistory, 39), safeGet(mathBrain.AccuracyHistory, 79),
		safeGet(mathBrain.AccuracyHistory, 119), safeGet(mathBrain.AccuracyHistory, 159), mathBrain.FinalAccuracy)
	fmt.Printf("║ 🔗 Ensemble        ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%         ║\n",
		safeGet(ensemble.AccuracyHistory, 39), safeGet(ensemble.AccuracyHistory, 79),
		safeGet(ensemble.AccuracyHistory, 119), safeGet(ensemble.AccuracyHistory, 159), ensemble.FinalAccuracy)
	fmt.Println("╚════════════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═════════════════╝")
}

func printVennDiagram(heuristic, mathBrain, ensemble *BrainStats) {
	// Build sets
	hSet := make(map[string]bool)
	mSet := make(map[string]bool)
	eSet := make(map[string]bool)

	for _, id := range heuristic.SolvedTaskIDs {
		hSet[id] = true
	}
	for _, id := range mathBrain.SolvedTaskIDs {
		mSet[id] = true
	}
	for _, id := range ensemble.SolvedTaskIDs {
		eSet[id] = true
	}

	// Calculate categories
	onlyHeuristic := []string{}
	onlyMath := []string{}
	bothHM := []string{}
	ensembleNewSolves := []string{} // Solved by ensemble but NOT by either brain alone

	// Union of all
	allIDs := make(map[string]bool)
	for id := range hSet {
		allIDs[id] = true
	}
	for id := range mSet {
		allIDs[id] = true
	}
	for id := range eSet {
		allIDs[id] = true
	}

	for id := range allIDs {
		inH := hSet[id]
		inM := mSet[id]
		inE := eSet[id]

		if inH && inM {
			bothHM = append(bothHM, id)
		} else if inH && !inM {
			onlyHeuristic = append(onlyHeuristic, id)
		} else if inM && !inH {
			onlyMath = append(onlyMath, id)
		}

		// New solves: ensemble solved it but neither brain did alone
		if inE && !inH && !inM {
			ensembleNewSolves = append(ensembleNewSolves, id)
		}
	}

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        🔗 VENN DIAGRAM ANALYSIS 🔗                                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║              ┌───────────────────┐         ┌───────────────────┐                     ║")
	fmt.Println("║              │    🧠 HEURISTIC   │         │     🧮 MATH       │                     ║")
	fmt.Printf("║              │    Only: %-3d      │         │    Only: %-3d      │                     ║\n", len(onlyHeuristic), len(onlyMath))
	fmt.Println("║              │                   │         │                   │                     ║")
	fmt.Println("║              └─────────┬─────────┘         └─────────┬─────────┘                     ║")
	fmt.Println("║                        │                             │                               ║")
	fmt.Println("║                        └──────────┬──────────────────┘                               ║")
	fmt.Printf("║                                   │ BOTH: %-3d                                        ║\n", len(bothHM))
	fmt.Println("║                                   │                                                  ║")
	fmt.Println("║                           ┌───────┴───────┐                                          ║")
	fmt.Println("║                           │  🔗 ENSEMBLE  │                                          ║")
	fmt.Printf("║                           │  NEW: %-3d     │                                          ║\n", len(ensembleNewSolves))
	fmt.Println("║                           └───────────────┘                                          ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  📊 TASK COVERAGE BREAKDOWN:                                                         ║\n")
	fmt.Printf("║     • Solved by Heuristic ONLY:   %-3d tasks                                         ║\n", len(onlyHeuristic))
	fmt.Printf("║     • Solved by Math ONLY:        %-3d tasks                                         ║\n", len(onlyMath))
	fmt.Printf("║     • Solved by BOTH:             %-3d tasks                                         ║\n", len(bothHM))
	fmt.Printf("║     • NEW Solves by Ensemble:     %-3d tasks (averaging unlocked these!)             ║\n", len(ensembleNewSolves))
	fmt.Printf("║     • TOTAL UNION:                %-3d unique tasks                                  ║\n", len(allIDs))
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	// Analysis
	if len(ensembleNewSolves) > 0 {
		fmt.Println("║  ✨ BREAKTHROUGH: Ensemble averaging solved tasks NEITHER brain could alone!        ║")
		fmt.Printf("║     New solves: %v\n", truncateList(ensembleNewSolves, 3))
	}

	if len(allIDs) > heuristic.TasksSolved && len(allIDs) > mathBrain.TasksSolved {
		fmt.Println("║  🎯 SUCCESS: Union of solved tasks > any single method!                             ║")
		fmt.Println("║     → Bicameral architecture provides complementary intelligence!                   ║")
	}

	if len(onlyHeuristic) > 0 && len(onlyMath) > 0 {
		fmt.Println("║  🧠🧮 CONFIRMED: Each brain solves DIFFERENT tasks!                                 ║")
		fmt.Println("║     → The two training methods have orthogonal strengths!                           ║")
	}

	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Print sample task IDs
	if len(onlyHeuristic) > 0 || len(onlyMath) > 0 {
		fmt.Println("\n📋 Sample Solved Task IDs:")
		if len(onlyHeuristic) > 0 {
			fmt.Printf("   🧠 Heuristic Only: %v\n", truncateList(onlyHeuristic, 3))
		}
		if len(onlyMath) > 0 {
			fmt.Printf("   🧮 Math Only:      %v\n", truncateList(onlyMath, 3))
		}
		if len(bothHM) > 0 {
			fmt.Printf("   ✓ Both:            %v\n", truncateList(bothHM, 3))
		}
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

func saveResults(heuristic, mathBrain, ensemble *BrainStats, trainTime time.Duration) {
	// Build sets for union analysis
	hSet := make(map[string]bool)
	mSet := make(map[string]bool)
	eSet := make(map[string]bool)
	for _, id := range heuristic.SolvedTaskIDs {
		hSet[id] = true
	}
	for _, id := range mathBrain.SolvedTaskIDs {
		mSet[id] = true
	}
	for _, id := range ensemble.SolvedTaskIDs {
		eSet[id] = true
	}

	// Union
	allIDs := make(map[string]bool)
	for id := range hSet {
		allIDs[id] = true
	}
	for id := range mSet {
		allIDs[id] = true
	}
	for id := range eSet {
		allIDs[id] = true
	}

	unionList := make([]string, 0, len(allIDs))
	for id := range allIDs {
		unionList = append(unionList, id)
	}
	sort.Strings(unionList)

	// New solves by ensemble
	ensembleNew := []string{}
	for id := range eSet {
		if !hSet[id] && !mSet[id] {
			ensembleNew = append(ensembleNew, id)
		}
	}

	output := map[string]interface{}{
		"heuristic": map[string]interface{}{
			"final_accuracy":   heuristic.FinalAccuracy,
			"tasks_solved":     heuristic.TasksSolved,
			"solved_task_ids":  heuristic.SolvedTaskIDs,
			"accuracy_history": heuristic.AccuracyHistory,
		},
		"math": map[string]interface{}{
			"final_accuracy":   mathBrain.FinalAccuracy,
			"tasks_solved":     mathBrain.TasksSolved,
			"solved_task_ids":  mathBrain.SolvedTaskIDs,
			"accuracy_history": mathBrain.AccuracyHistory,
		},
		"ensemble": map[string]interface{}{
			"final_accuracy":   ensemble.FinalAccuracy,
			"tasks_solved":     ensemble.TasksSolved,
			"solved_task_ids":  ensemble.SolvedTaskIDs,
			"accuracy_history": ensemble.AccuracyHistory,
		},
		"union": map[string]interface{}{
			"total_unique":        len(allIDs),
			"task_ids":            unionList,
			"ensemble_new_solves": ensembleNew,
		},
		"meta": map[string]interface{}{
			"architecture":   "Attn-21L × 2 (Bicameral-P)",
			"epochs":         NumEpochs,
			"batch_size":     BatchSize,
			"heuristic_lr":   HeuristicLR,
			"math_lr":        MathLR,
			"train_time_sec": trainTime.Seconds(),
		},
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test26_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test26_results.json")
}

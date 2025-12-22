package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 24b: THE ULTIMATE SHOWDOWN - STABILIZED
//
// Changes from Test 24:
//   - Learning Rate: 0.01 → 0.001 (10x reduction to prevent collapse)
//   - Gradient Clipping: 0.5 → 0.1 (tighter clipping for stability)
//   - Weight Init Scale: 1.0 → 0.5 (more conservative initialization)
//   - Hybrid LinkBudgetScale: 0.5 → 2.0 (force gradients through)
//
// Note: RMSNorm removed - causes StepBackward gradient propagation issues

const (
	MaxGridSize  = 30
	InputSize    = MaxGridSize * MaxGridSize // 900
	NumTasks     = 400                       // Full ARC-AGI training set
	BatchSize    = 100
	NumEpochs    = 200
	LearningRate = float32(0.001) // STABILIZED: 10x lower than test24
	GradClip     = float32(0.1)   // STABILIZED: Tighter gradient clipping
	InitScale    = float32(0.5)   // STABILIZED: More conservative init
)

type TrainingMode int

const (
	ModeStepTween TrainingMode = iota
	ModeStepTweenChain
	ModeStepBP
)

var modeNames = map[TrainingMode]string{
	ModeStepTween:      "Heuristic",
	ModeStepTweenChain: "Hybrid",
	ModeStepBP:         "Math",
}

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

type ContenderResult struct {
	Mode            TrainingMode
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
	fmt.Println("║     Test 24b: THE ULTIMATE SHOWDOWN - STABILIZED                                    ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Architecture: Attn-21L (Conservative Init + Tight Gradient Clipping)            ║")
	fmt.Println("║     Learning Rate: 0.001 (10x lower for stability)                                  ║")
	fmt.Println("║     Hybrid LinkBudgetScale: 2.0 (force gradients through)                           ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     CONTENDERS:                                                                      ║")
	fmt.Println("║       🏆 Heuristic: ModeStepTween      - Current Champion                            ║")
	fmt.Println("║       ⚡ Hybrid:    ModeStepTweenChain - PATCHED + High LinkBudgetScale              ║")
	fmt.Println("║       🧮 Math:      ModeStepBP         - Pure Backpropagation                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	results := make(map[TrainingMode]*ContenderResult)
	var mu sync.Mutex
	var wg sync.WaitGroup

	modes := []TrainingMode{ModeStepTween, ModeStepTweenChain, ModeStepBP}

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                            ⚔️  TRAINING BEGINS ⚔️")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			result := runContender(trainSamples, evalSamples, m)
			mu.Lock()
			results[m] = result
			mu.Unlock()

			emoji := map[TrainingMode]string{ModeStepTween: "🏆", ModeStepTweenChain: "⚡", ModeStepBP: "🧮"}
			fmt.Printf("\n%s [%-10s] COMPLETE: Accuracy=%.1f%% | Budget=%.3f | Tasks=%d/%d | Time=%.1fs\n",
				emoji[m], modeNames[m], result.FinalAccuracy, result.FinalBudget,
				result.TasksSolved, len(evalSamples), result.TrainTime.Seconds())
		}(mode)
	}

	wg.Wait()

	printProgressTimeline(results)
	printFinalShowdown(results)
	printSolvedTasksAnalysis(results, evalSamples)
	saveResults(results)
}

// ============================================================================
// Training Logic
// ============================================================================

func runContender(trainSamples, evalSamples []Sample, mode TrainingMode) *ContenderResult {
	start := time.Now()

	// Create stabilized Attn-21L (no RMSNorm - causes StepBackward issues)
	net := createMHAStabilized(21)
	numLayers := net.TotalLayers()

	result := &ContenderResult{
		Mode:            mode,
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
	}

	state := net.InitStepState(InputSize)

	var ts *nn.TweenState
	if mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
			ts.Config.LinkBudgetScale = 2.0 // CRITICAL: Force gradients through
		}
	}

	sampleIdx := 0
	for epoch := 0; epoch < NumEpochs; epoch++ {
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++
			trainOneSample(net, sample, mode, numLayers, state, ts, LearningRate)
		}

		acc, budget := measureMetrics(net, evalSamples, mode, numLayers, state, ts)
		result.AccuracyHistory[epoch] = acc
		result.BudgetHistory[epoch] = budget

		if (epoch+1)%20 == 0 {
			fmt.Printf("  [%-10s] Epoch %3d/%d: Acc=%.1f%% Budget=%.3f\n",
				modeNames[mode], epoch+1, NumEpochs, acc, budget)
		}
	}

	result.FinalAccuracy = result.AccuracyHistory[NumEpochs-1]
	result.FinalBudget = result.BudgetHistory[NumEpochs-1]
	result.TasksSolved, result.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, mode, numLayers, state, ts)
	result.TrainTime = time.Since(start)

	return result
}

func trainOneSample(net *nn.Network, sample Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState, lr float32) {
	switch mode {
	case ModeStepBP:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		grad := computeGradient(output, sample.Target)
		net.StepBackward(state, grad)
		net.ApplyGradients(lr)

	case ModeStepTween:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), lr)

	case ModeStepTweenChain:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		ts.ForwardPass(net, sample.Input)
		applyTweenUpdate(ts, net, sample, output, lr)
	}
}

func computeGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	for i := range output {
		if i < len(target) {
			grad[i] = clipGrad(output[i]-target[i], GradClip) // STABILIZED: Tighter clipping
		}
	}
	return grad
}

func applyTweenUpdate(ts *nn.TweenState, net *nn.Network, sample Sample, output []float32, lr float32) {
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

func measureMetrics(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) (accuracy float64, avgBudget float32) {
	correct, total := 0, 0

	for _, sample := range samples {
		output := getOutput(net, sample.Input, mode, numLayers, state, ts)
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
		accuracy = 0
	} else {
		accuracy = float64(correct) / float64(total) * 100
	}

	if ts != nil && len(ts.LinkBudgets) > 0 {
		midIdx := len(ts.LinkBudgets) / 2
		avgBudget = ts.LinkBudgets[midIdx]
	} else {
		avgBudget = estimateBudgetFromActivations(net, samples[0].Input, state)
	}

	return accuracy, avgBudget
}

func estimateBudgetFromActivations(net *nn.Network, input []float32, state *nn.StepState) float32 {
	state.SetInput(input)
	numLayers := net.TotalLayers()
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	output := state.GetOutput()

	sum := float32(0)
	for _, v := range output {
		sum += float32(math.Abs(float64(v)))
	}
	if len(output) == 0 {
		return 0
	}
	avgMag := sum / float32(len(output))
	budget := avgMag * 2
	if budget > 1 {
		budget = 1
	}
	return budget
}

func measureSolvedTasks(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) (int, []string) {
	solved := 0
	solvedIDs := []string{}

	for _, sample := range samples {
		output := getOutput(net, sample.Input, mode, numLayers, state, ts)
		allCorrect := true

		for r := 0; r < sample.Height && allCorrect; r++ {
			for c := 0; c < sample.Width && allCorrect; c++ {
				idx := r*MaxGridSize + c
				if idx < len(output) && idx < len(sample.Target) {
					pred := clampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					exp := clampInt(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
					if pred != exp {
						allCorrect = false
					}
				}
			}
		}

		if allCorrect {
			solved++
			solvedIDs = append(solvedIDs, sample.TaskID)
		}
	}

	return solved, solvedIDs
}

func getOutput(net *nn.Network, input []float32, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) []float32 {
	state.SetInput(input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	return state.GetOutput()
}

// ============================================================================
// Network Factory - STABILIZED: Attn-21L with conservative initialization
// ============================================================================

func createMHAStabilized(depth int) *nn.Network {
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1

	// Input projection with conservative init
	inputLayer := nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, 0, inputLayer)

	// Alternating MHA + Dense layers
	for i := 1; i < depth-1; i++ {
		if i%2 == 1 {
			net.SetLayer(0, 0, i, createMHAHeavyLayer(dModel))
		} else {
			denseLayer := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
			scaleWeights(denseLayer.Kernel, InitScale)
			net.SetLayer(0, 0, i, denseLayer)
		}
	}

	// Output projection
	outputLayer := nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, depth-1, outputLayer)

	return net
}

func createMHAHeavyLayer(dModel int) nn.LayerConfig {
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

	// STABILIZED: More conservative initialization
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
// Visualization
// ============================================================================

func printProgressTimeline(results map[TrainingMode]*ContenderResult) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      📊 TRAINING PROGRESS TIMELINE                                   ║")
	fmt.Println("╠═══════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════════════════╣")
	fmt.Println("║ Contender     ║  Ep 40    ║  Ep 80    ║  Ep 120   ║  Ep 160   ║  Final (Ep 200)       ║")
	fmt.Println("╠═══════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════════════════╣")

	modes := []TrainingMode{ModeStepTween, ModeStepTweenChain, ModeStepBP}
	emoji := map[TrainingMode]string{ModeStepTween: "🏆", ModeStepTweenChain: "⚡", ModeStepBP: "🧮"}

	for _, m := range modes {
		r := results[m]
		if r == nil {
			continue
		}

		fmt.Printf("║ %s %-10s ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%% (B:%.3f)      ║\n",
			emoji[m], modeNames[m],
			safeGet(r.AccuracyHistory, 39),
			safeGet(r.AccuracyHistory, 79),
			safeGet(r.AccuracyHistory, 119),
			safeGet(r.AccuracyHistory, 159),
			r.FinalAccuracy, r.FinalBudget)
	}
	fmt.Println("╚═══════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════════════════╝")

	// Budget Timeline
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🔋 MIDBUDGET TIMELINE (Network Health)                          ║")
	fmt.Println("╠═══════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════════════════╣")
	fmt.Println("║ Contender     ║  Ep 40    ║  Ep 80    ║  Ep 120   ║  Ep 160   ║  Final (Ep 200)       ║")
	fmt.Println("╠═══════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════════════════╣")

	for _, m := range modes {
		r := results[m]
		if r == nil {
			continue
		}

		status := "✓ HEALTHY"
		if r.FinalBudget < 0.5 {
			status = "⚠ DEAD"
		} else if r.FinalBudget < 0.7 {
			status = "⚡ WEAK"
		}

		fmt.Printf("║ %s %-10s ║   %.3f   ║   %.3f   ║   %.3f   ║   %.3f   ║  %.3f %s      ║\n",
			emoji[m], modeNames[m],
			safeGetF32(r.BudgetHistory, 39),
			safeGetF32(r.BudgetHistory, 79),
			safeGetF32(r.BudgetHistory, 119),
			safeGetF32(r.BudgetHistory, 159),
			r.FinalBudget, status)
	}
	fmt.Println("╚═══════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════════════════╝")
}

func printFinalShowdown(results map[TrainingMode]*ContenderResult) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           🏆 FINAL SHOWDOWN RESULTS 🏆                               ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════╣")

	modes := []TrainingMode{ModeStepTween, ModeStepTweenChain, ModeStepBP}
	emoji := map[TrainingMode]string{ModeStepTween: "🏆", ModeStepTweenChain: "⚡", ModeStepBP: "🧮"}

	bestMode := ModeStepTween
	bestAcc := 0.0
	for _, m := range modes {
		if results[m] != nil && results[m].FinalAccuracy > bestAcc {
			bestAcc = results[m].FinalAccuracy
			bestMode = m
		}
	}

	for _, m := range modes {
		r := results[m]
		if r == nil {
			continue
		}

		crown := "  "
		if m == bestMode {
			crown = "👑"
		}

		healthStatus := "✓ HEALTHY"
		if r.FinalBudget < 0.5 {
			healthStatus = "💀 DEAD"
		} else if r.FinalBudget < 0.7 {
			healthStatus = "⚡ WEAK"
		}

		fmt.Printf("║ %s %s %-12s │ Acc: %5.1f%% │ Budget: %.3f (%s) │ Tasks: %d │ %.1fs ║\n",
			crown, emoji[m], modeNames[m],
			r.FinalAccuracy, r.FinalBudget, healthStatus,
			r.TasksSolved, r.TrainTime.Seconds())
	}

	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════╣")

	hybrid := results[ModeStepTweenChain]
	heuristic := results[ModeStepTween]
	mathMode := results[ModeStepBP]

	if hybrid != nil && heuristic != nil {
		if hybrid.FinalBudget > 0.7 && hybrid.FinalAccuracy > heuristic.FinalAccuracy {
			fmt.Println("║  ✅ SUCCESS: Hybrid shows HEALTHY budget AND higher accuracy!                        ║")
			fmt.Println("║     → The Gradient Fragility problem has been SOLVED!                                ║")
		} else if hybrid.FinalBudget > 0.7 && hybrid.FinalAccuracy > 15 {
			fmt.Println("║  ⚡ PROGRESS: Hybrid has healthy budget and is learning (>15%)                       ║")
			fmt.Println("║     → Stabilization helped. May need more epochs.                                    ║")
		} else if hybrid.FinalBudget > 0.7 {
			fmt.Println("║  ⚠️  PARTIAL: Hybrid has healthy budget but accuracy is low.                         ║")
			fmt.Println("║     → Gradients flowing but not learning effectively.                                ║")
		} else if hybrid.FinalBudget < 0.5 {
			fmt.Println("║  ❌ FAILED: Hybrid budget collapsed despite stabilization.                           ║")
		}
	}

	if mathMode != nil && mathMode.FinalBudget > 0.5 {
		fmt.Println("║  🧮 Math (StepBP) survived with lower LR + conservative init!                        ║")
	} else if mathMode != nil {
		fmt.Println("║  🧮 Math (StepBP) still struggling - may need even lower LR.                         ║")
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════╝")
}

func printSolvedTasksAnalysis(results map[TrainingMode]*ContenderResult, samples []Sample) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        🔍 SOLVED TASKS ANALYSIS                                      ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════╣")

	modes := []TrainingMode{ModeStepTween, ModeStepTweenChain, ModeStepBP}
	allSolved := make(map[string][]string)

	for _, m := range modes {
		r := results[m]
		if r == nil {
			continue
		}
		for _, taskID := range r.SolvedTaskIDs {
			allSolved[taskID] = append(allSolved[taskID], modeNames[m])
		}
	}

	onlyHeuristic := 0
	onlyHybrid := 0
	onlyMath := 0
	sharedAll := 0
	sharedSome := 0

	for _, solvers := range allSolved {
		if len(solvers) == 3 {
			sharedAll++
		} else if len(solvers) > 1 {
			sharedSome++
		} else if solvers[0] == "Heuristic" {
			onlyHeuristic++
		} else if solvers[0] == "Hybrid" {
			onlyHybrid++
		} else if solvers[0] == "Math" {
			onlyMath++
		}
	}

	fmt.Printf("║  Total Unique Tasks Solved: %d                                                        ║\n", len(allSolved))
	fmt.Printf("║  ├─ Solved by ALL three:    %d                                                        ║\n", sharedAll)
	fmt.Printf("║  ├─ Solved by 2/3:          %d                                                        ║\n", sharedSome)
	fmt.Printf("║  ├─ ONLY Heuristic:         %d                                                        ║\n", onlyHeuristic)
	fmt.Printf("║  ├─ ONLY Hybrid:            %d                                                        ║\n", onlyHybrid)
	fmt.Printf("║  └─ ONLY Math:              %d                                                        ║\n", onlyMath)
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════╣")

	if onlyHybrid > 0 || onlyMath > 0 {
		fmt.Println("║  ✨ INSIGHT: Gradient methods solve DIFFERENT tasks than heuristic!                 ║")
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════╝")
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

func clipGrad(v, max float32) float32 {
	if v > max {
		return max
	}
	if v < -max {
		return -max
	}
	if math.IsNaN(float64(v)) {
		return 0
	}
	return v
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

func safeGetF32(slice []float32, idx int) float32 {
	if idx < len(slice) && idx >= 0 {
		return slice[idx]
	}
	return 0
}

func saveResults(results map[TrainingMode]*ContenderResult) {
	output := make(map[string]interface{})
	for mode, r := range results {
		if r == nil {
			continue
		}
		output[modeNames[mode]] = map[string]interface{}{
			"final_accuracy":   r.FinalAccuracy,
			"final_budget":     r.FinalBudget,
			"tasks_solved":     r.TasksSolved,
			"solved_task_ids":  r.SolvedTaskIDs,
			"train_time_sec":   r.TrainTime.Seconds(),
			"accuracy_history": r.AccuracyHistory,
			"budget_history":   r.BudgetHistory,
		}
	}

	output["meta"] = map[string]interface{}{
		"architecture":        "Attn-21L (Conservative Init)",
		"batch_size":          BatchSize,
		"epochs":              NumEpochs,
		"learning_rate":       LearningRate,
		"grad_clip":           GradClip,
		"init_scale":          InitScale,
		"hybrid_budget_scale": 2.0,
		"stabilization":       "Lower LR + Conservative Init + Tight Grad Clip",
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test24b_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test24b_results.json")
}

var _ = sort.Strings

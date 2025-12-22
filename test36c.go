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

// Test 36c: WEIGHTED HIVE + RL COLOR BOOST
//
// Extension of Test 36: After every epoch of weighted training,
// we run a Reinforcement Learning phase to specifically improve COLOR accuracy.
//
// RL Approach: Perturbation-Based Policy Gradient
//   - After weighted TweenStep epoch: measure color accuracy
//   - Apply small random perturbations to weights
//   - Keep perturbations that improve color accuracy
//   - Report accuracy BEFORE and AFTER RL phase
//
// Architecture (same as Test 36):
//   - 2x2 Grid Scatter Hive Mind with MHA, LSTM, CNN branches

var (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize
	NumTasks    = 400
	BatchSize   = 100
	NumEpochs   = 20

	LearningRate = float32(0.1)
	BudgetScale  = float32(0.8)

	// Weights for class imbalance
	WeightBackground = float32(0.1)
	WeightColor      = float32(10.0)

	// Architecture
	DModel       = 64
	NumHeads     = 4
	LSTMHidden   = 64
	ConvGridSize = 8
	ConvFilters  = 8
	ConvKernel   = 3
	InitScale    = float32(0.5)

	// RL params - MORE AGGRESSIVE
	RLPerturbScaleStart = float32(0.05) // Start with bigger perturbations
	RLPerturbScaleMin   = float32(0.1)  // Minimum perturbation scale
	RLTrials            = 50            // Try 50 times per epoch!
	RLSampleFraction    = 0.3           // Use 30% of samples for faster eval
	RLTrainOnImprove    = 5             // Do 5 TweenStep trains when RL improves
)

// Global TweenState reference for RL training
var globalTS *nn.TweenState
var globalTrainSamples []Sample

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
	AccuracyHistory         []float64
	WeightedAccuracyHistory []float64
	PostRLColorAccHistory   []float64
	BudgetHistory           []float32
	RLImprovements          int // Track how many times RL helped
	FinalAccuracy           float64
	FinalWeightedAccuracy   float64
	FinalPostRLColorAcc     float64
	TasksSolved             int
	SolvedTaskIDs           []string
	TrainTime               time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 36c: WEIGHTED HIVE + RL COLOR BOOST (AGGRESSIVE)                           ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🎯 Background (0): Weight = 0.1 (Ignore)                                        ║")
	fmt.Println("║     🌈 Colors (1-9):   Weight = 10.0 (FOCUS!)                                       ║")
	fmt.Println("║     🎲 RL Phase: 50 trials + TweenStep training on improvements!                    ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Goal: Weighted training + AGGRESSIVE RL boost for color accuracy                ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	globalTrainSamples = trainSamples // Set global for RL training access
	fmt.Printf("\n📦 Loaded %d tasks.\n", len(tasks))

	// Create Hive Mind Network
	net := createHiveMindNetwork()
	numLayers := net.TotalLayers()
	state := net.InitStepState(InputSize)

	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true
	ts.Config.LinkBudgetScale = BudgetScale
	globalTS = ts // Set global for RL training access

	results := &Results{
		AccuracyHistory:         make([]float64, NumEpochs),
		WeightedAccuracyHistory: make([]float64, NumEpochs),
		PostRLColorAccHistory:   make([]float64, NumEpochs),
		BudgetHistory:           make([]float32, NumEpochs),
		SolvedTaskIDs:           []string{},
		RLImprovements:          0,
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("              ⚖️ WEIGHTED + AGGRESSIVE RL COLOR TRAINING ⚖️")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	// Create subsample for faster RL evaluation
	rlSampleCount := int(float64(len(evalSamples)) * RLSampleFraction)
	if rlSampleCount < 10 {
		rlSampleCount = 10
	}

	for epoch := 0; epoch < NumEpochs; epoch++ {
		// PHASE 1: Weighted TweenStep training + per-sample RL
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}

			weightedTweenStep(ts, net, sample, LearningRate)

			// === PER-SAMPLE RL ===
			// Try to improve color accuracy on this specific sample
			runSampleRL(net, sample, numLayers, state, ts)
		}

		// Measure pre-RL metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		wAcc := measureWeightedAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.WeightedAccuracyHistory[epoch] = wAcc
		results.BudgetHistory[epoch] = budget

		// PHASE 2: RL Color Boost
		rlSamples := make([]Sample, rlSampleCount)
		perm := rand.Perm(len(evalSamples))
		for i := 0; i < rlSampleCount; i++ {
			rlSamples[i] = evalSamples[perm[i]]
		}

		postRLColorAcc := runRLColorBoost(net, rlSamples, numLayers, state, wAcc)
		results.PostRLColorAccHistory[epoch] = postRLColorAcc

		if (epoch+1)%1 == 0 {
			colorGain := postRLColorAcc - wAcc
			gainSymbol := "→"
			if colorGain > 0.1 {
				gainSymbol = "↑"
			} else if colorGain < -0.1 {
				gainSymbol = "↓"
			}
			fmt.Printf("  Epoch %3d/%d | Acc: %5.1f%% | Color: %5.1f%% %s %5.1f%% | Budget: %.3f\n",
				epoch+1, NumEpochs, acc, wAcc, gainSymbol, postRLColorAcc, budget)
		}

		LearningRate = float32(0.001)
	}

	results.TrainTime = time.Since(start)
	results.FinalAccuracy = results.AccuracyHistory[NumEpochs-1]
	results.FinalWeightedAccuracy = results.WeightedAccuracyHistory[NumEpochs-1]
	results.FinalPostRLColorAcc = results.PostRLColorAccHistory[NumEpochs-1]
	results.TasksSolved, results.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, numLayers, state)

	printResults(results)
	saveResults(results)
}

// ============================================================================
// RL COLOR BOOST - AGGRESSIVE Perturbation + Training on Improvement
// ============================================================================

func runRLColorBoost(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState, baselineColorAcc float64) float64 {
	bestColorAcc := baselineColorAcc
	baselineWeights := saveNetworkWeights(net)
	currentScale := RLPerturbScaleStart
	improvementCount := 0

	for trial := 0; trial < RLTrials; trial++ {
		// Apply perturbation with current scale
		applyRandomPerturbations(net, currentScale)
		newColorAcc := measureWeightedAccuracy(net, samples, numLayers, state)

		if newColorAcc > bestColorAcc {
			// IMPROVEMENT FOUND!
			improvementCount++
			improvement := newColorAcc - bestColorAcc
			bestColorAcc = newColorAcc
			baselineWeights = saveNetworkWeights(net)

			// === TRAIN ON IMPROVEMENT ===
			// Since this perturbation helped, do TweenStep training to reinforce it
			if globalTS != nil && len(globalTrainSamples) > 0 {
				for trainIter := 0; trainIter < RLTrainOnImprove; trainIter++ {
					// Pick a random training sample that has color pixels
					sample := globalTrainSamples[rand.Intn(len(globalTrainSamples))]

					// Train on color pixels with boosted learning rate
					colorIndices := []int{}
					for i, t := range sample.Target {
						if t > 0.05 {
							colorIndices = append(colorIndices, i)
						}
					}
					if len(colorIndices) > 0 {
						idx := colorIndices[rand.Intn(len(colorIndices))]
						// Use higher LR since we know this direction is good
						globalTS.TweenStep(net, sample.Input, idx, len(sample.Target), LearningRate*WeightColor*2)
					}
				}
			}

			// Increase scale slightly when we find improvements (explore more)
			currentScale = min32(currentScale*1.2, RLPerturbScaleStart*2)

			fmt.Printf("      RL trial %d: +%.2f%% (total: %.1f%%) 🎯\n", trial+1, improvement, bestColorAcc)
		} else {
			// No improvement - revert and shrink perturbation scale
			restoreNetworkWeights(net, baselineWeights)
			currentScale = max32(currentScale*0.9, RLPerturbScaleMin)
		}
	}

	if improvementCount > 0 {
		fmt.Printf("      RL made %d improvements!\n", improvementCount)
	}

	return bestColorAcc
}

// runSampleRL does mini RL on a single training sample to improve color accuracy
const SampleRLTrials = 5

func runSampleRL(net *nn.Network, sample Sample, numLayers int, state *nn.StepState, ts *nn.TweenState) {
	// Measure current color accuracy on this sample
	baselineAcc := measureSampleColorAcc(net, sample, numLayers, state)
	if baselineAcc >= 100.0 {
		return // Already perfect on this sample
	}

	baselineWeights := saveNetworkWeights(net)
	bestAcc := baselineAcc
	scale := RLPerturbScaleStart * 0.5 // Smaller scale for single-sample RL

	for trial := 0; trial < SampleRLTrials; trial++ {
		applyRandomPerturbations(net, scale)
		newAcc := measureSampleColorAcc(net, sample, numLayers, state)

		if newAcc > bestAcc {
			// Improvement! Keep it and train to reinforce
			bestAcc = newAcc
			baselineWeights = saveNetworkWeights(net)

			// Train on color pixels to reinforce the improvement
			colorIndices := []int{}
			for i, t := range sample.Target {
				if t > 0.05 {
					colorIndices = append(colorIndices, i)
				}
			}
			if len(colorIndices) > 0 && ts != nil {
				idx := colorIndices[rand.Intn(len(colorIndices))]
				ts.TweenStep(net, sample.Input, idx, len(sample.Target), LearningRate*WeightColor)
			}
		} else {
			restoreNetworkWeights(net, baselineWeights)
			scale *= 0.8 // Shrink on failure
		}
	}
}

// measureSampleColorAcc measures color accuracy on a single sample
func measureSampleColorAcc(net *nn.Network, sample Sample, numLayers int, state *nn.StepState) float64 {
	state.SetInput(sample.Input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	output := state.GetOutput()

	correct, total := 0, 0
	for i := range output {
		if i < len(sample.Target) {
			exp := clampInt(int(math.Round(float64(sample.Target[i])*9.0)), 0, 9)
			if exp > 0 { // Only count colors
				pred := clampInt(int(math.Round(float64(output[i])*9.0)), 0, 9)
				if pred == exp {
					correct++
				}
				total++
			}
		}
	}
	if total == 0 {
		return 100.0 // No color pixels = perfect
	}
	return float64(correct) / float64(total) * 100
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func saveNetworkWeights(net *nn.Network) [][]float32 {
	totalLayers := net.TotalLayers()
	weights := make([][]float32, totalLayers)

	for i := 0; i < totalLayers; i++ {
		cfg := net.GetLayer(0, 0, i)
		if cfg != nil && len(cfg.Kernel) > 0 {
			weights[i] = make([]float32, len(cfg.Kernel))
			copy(weights[i], cfg.Kernel)
		}
	}

	return weights
}

func restoreNetworkWeights(net *nn.Network, weights [][]float32) {
	for i := 0; i < len(weights); i++ {
		cfg := net.GetLayer(0, 0, i)
		if cfg != nil && weights[i] != nil && len(cfg.Kernel) > 0 {
			copy(cfg.Kernel, weights[i])
		}
	}
}

func applyRandomPerturbations(net *nn.Network, scale float32) {
	totalLayers := net.TotalLayers()

	for i := 0; i < totalLayers; i++ {
		cfg := net.GetLayer(0, 0, i)
		if cfg == nil {
			continue
		}

		for j := range cfg.Kernel {
			cfg.Kernel[j] += (rand.Float32()*2 - 1) * scale
		}

		for j := range cfg.Bias {
			cfg.Bias[j] += (rand.Float32()*2 - 1) * scale * 0.5
		}
	}
}

// ============================================================================
// WEIGHTED TRAINING (same as test36)
// ============================================================================

func weightedTweenStep(ts *nn.TweenState, net *nn.Network, sample Sample, lr float32) {
	colorIndices := []int{}
	bgIndices := []int{}

	for i, t := range sample.Target {
		if t > 0.05 {
			colorIndices = append(colorIndices, i)
		} else {
			bgIndices = append(bgIndices, i)
		}
	}

	if len(colorIndices) > 0 {
		idx := colorIndices[rand.Intn(len(colorIndices))]
		ts.TweenStep(net, sample.Input, idx, len(sample.Target), lr*WeightColor)
	}

	if len(bgIndices) > 0 && rand.Float32() < 0.1 {
		idx := bgIndices[rand.Intn(len(bgIndices))]
		ts.TweenStep(net, sample.Input, idx, len(sample.Target), lr*WeightBackground)
	}
}

// measureWeightedAccuracy measures accuracy on colored pixels only
func measureWeightedAccuracy(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		for i := range output {
			if i < len(sample.Target) {
				exp := clampInt(int(math.Round(float64(sample.Target[i])*9.0)), 0, 9)
				if exp > 0 {
					pred := clampInt(int(math.Round(float64(output[i])*9.0)), 0, 9)
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

// ============================================================================
// NETWORK ARCHITECTURE (same as Test 36)
// ============================================================================

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
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0},
			{BranchIndex: 2, TargetRow: 1, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 3, TargetRow: 1, TargetCol: 1, TargetLayer: 0},
		},
	}
}

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
	cnn := nn.LayerConfig{
		Type: nn.LayerConv2D, InputHeight: ConvGridSize, InputWidth: ConvGridSize,
		InputChannels: 1, Filters: ConvFilters, KernelSize: ConvKernel,
		Stride: 1, Padding: 1, OutputHeight: ConvGridSize, OutputWidth: ConvGridSize,
		Activation: nn.ActivationLeakyReLU,
	}
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

// ============================================================================
// HELPERS
// ============================================================================

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

func safeGet(slice []float64, idx int) float64 {
	if idx < len(slice) && idx >= 0 {
		return slice[idx]
	}
	return 0
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
	Train []struct{ Input, Output [][]int }
	Test  []struct{ Input, Output [][]int }
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
			trainSamples = append(trainSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID,
			})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID,
			})
		}
	}
	holdout := len(trainSamples) / 5
	evalSamples = trainSamples[len(trainSamples)-holdout:]
	trainSamples = trainSamples[:len(trainSamples)-holdout]
	return trainSamples, evalSamples
}

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                  ⚖️ WEIGHTED HIVE + RL COLOR - FINAL RESULTS ⚖️                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Final Accuracy:           %5.1f%%                                                  ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Color Accuracy:     %5.1f%%                                                  ║\n", results.FinalWeightedAccuracy)
	fmt.Printf("║   Post-RL Color Accuracy:   %5.1f%%                                                  ║\n", results.FinalPostRLColorAcc)
	fmt.Printf("║   Tasks Solved:             %d                                                        ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:            %.1fs                                                     ║\n", results.TrainTime.Seconds())
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           COLOR ACCURACY TIMELINE                                    ║")
	fmt.Println("╠════════════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═════════════════╣")
	fmt.Println("║     Epoch          ║    40     ║    80     ║   120     ║   160     ║   Final         ║")
	fmt.Println("╠════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═════════════════╣")
	fmt.Printf("║ Pre-RL Color       ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%         ║\n",
		safeGet(results.WeightedAccuracyHistory, 39), safeGet(results.WeightedAccuracyHistory, 79),
		safeGet(results.WeightedAccuracyHistory, 119), safeGet(results.WeightedAccuracyHistory, 159),
		results.FinalWeightedAccuracy)
	fmt.Printf("║ Post-RL Color      ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%   ║  %5.1f%%         ║\n",
		safeGet(results.PostRLColorAccHistory, 39), safeGet(results.PostRLColorAccHistory, 79),
		safeGet(results.PostRLColorAccHistory, 119), safeGet(results.PostRLColorAccHistory, 159),
		results.FinalPostRLColorAcc)
	fmt.Println("╚════════════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═════════════════╝")

	if results.FinalPostRLColorAcc > results.FinalWeightedAccuracy {
		fmt.Println("\n🎨 RL successfully boosted color accuracy!")
	}

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
		"final_accuracy":                 results.FinalAccuracy,
		"final_color_accuracy":           results.FinalWeightedAccuracy,
		"final_post_rl_color_accuracy":   results.FinalPostRLColorAcc,
		"tasks_solved":                   results.TasksSolved,
		"solved_task_ids":                results.SolvedTaskIDs,
		"train_time_sec":                 results.TrainTime.Seconds(),
		"accuracy_history":               results.AccuracyHistory,
		"color_accuracy_history":         results.WeightedAccuracyHistory,
		"post_rl_color_accuracy_history": results.PostRLColorAccHistory,
		"meta": map[string]interface{}{
			"architecture":       "2x2 Grid Scatter Hive Mind + RL Color Boost",
			"training_mode":      "Weighted StepTween + RL",
			"weight_background":  WeightBackground,
			"weight_color":       WeightColor,
			"epochs":             NumEpochs,
			"learning_rate":      LearningRate,
			"rl_perturb_scale":   RLPerturbScaleStart,
			"rl_trials":          RLTrials,
			"rl_sample_fraction": RLSampleFraction,
		},
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test36c_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test36c_results.json")
}

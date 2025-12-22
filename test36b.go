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

// Test 36b: MULTI-POINT HIVE - Granular Error Correction
//
// Change from Test 36:
// Instead of training on ONE random pixel per sample (which implies others are 0),
// we scan the entire grid and apply 'TweenStep' to EVERY pixel that has significant error.
//
// Logic:
//   1. Forward Pass -> Get Output.
//   2. For every pixel i in 0..900:
//      Diff = Abs(Target[i] - Output[i])
//      If Diff < 0.1: Continue (Good enough, don't touch)
//      Weight = (Target[i] > 0) ? 10.0 : 0.1
//      TweenStep(i, LR * Weight * Diff)
//
// Expected Result:
//   - No more "All Black" collapse.
//   - Color pixels get strong, consistent reinforcement.
//   - Background pixels are only suppressed if they are wrongly active.

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize
	NumTasks    = 400
	BatchSize   = 50 // Reduced batch size because we do more ops per sample
	NumEpochs   = 600

	LearningRate = float32(0.001)
	BudgetScale  = float32(0.8)

	// Weights
	WeightBackground = float32(0.1)  // Weak suppression
	WeightColor      = float32(10.0) // Strong excitation

	// Architecture
	DModel       = 64
	NumHeads     = 4
	LSTMHidden   = 64
	ConvGridSize = 8
	ConvFilters  = 8
	ConvKernel   = 3
	InitScale    = float32(0.5)
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
	AccuracyHistory         []float64
	WeightedAccuracyHistory []float64
	BudgetHistory           []float32
	FinalAccuracy           float64
	FinalWeightedAccuracy   float64
	TasksSolved             int
	SolvedTaskIDs           []string
	TrainTime               time.Duration
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 36b: MULTI-POINT HIVE (Granular Error Correction)                          ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Strategy: Iterate all pixels. Only train if Error > 0.1.                        ║")
	fmt.Println("║     Weights:  Color errors get 10x LR. Background errors get 0.1x LR.               ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Goal: Fix the 'All Black' collapse by consistently reinforcing color pixels.    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks.\n", len(tasks))

	// Create Hive Mind Network (Reuse Architecture)
	net := createHiveMindNetwork()
	numLayers := net.TotalLayers()
	state := net.InitStepState(InputSize)

	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = false // Heuristic
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory:         make([]float64, NumEpochs),
		WeightedAccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:           make([]float32, NumEpochs),
		SolvedTaskIDs:           []string{},
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🎯 MULTI-POINT TRAINING START 🎯")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		// Training loop
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			// 1. Forward Pass (To calculate errors)
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output := state.GetOutput()

			// 2. Multi-Point Tween Application
			applyMultiPointTween(ts, net, sample, output, LearningRate)
		}

		// Metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		wAcc := measureWeightedAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.WeightedAccuracyHistory[epoch] = wAcc
		results.BudgetHistory[epoch] = budget

		if (epoch+1)%1 == 0 {
			fmt.Printf("  Epoch %3d/%d | Acc: %5.1f%% | Color Acc: %5.1f%% | Budget: %.3f\n",
				epoch+1, NumEpochs, acc, wAcc, budget)
		}
	}

	results.TrainTime = time.Since(start)
	results.FinalAccuracy = results.AccuracyHistory[NumEpochs-1]
	results.FinalWeightedAccuracy = results.WeightedAccuracyHistory[NumEpochs-1]
	results.TasksSolved, results.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, numLayers, state)

	printResults(results)
	saveResults(results)
}

// === THE FIX: Multi-Point Tween Logic ===

func applyMultiPointTween(ts *nn.TweenState, net *nn.Network, sample Sample, output []float32, baseLR float32) {
	// Instead of random sampling, iterate valid pixels
	target := sample.Target

	// We scan the grid. If a pixel is WRONG, we apply a TweenStep fix.
	for i := 0; i < len(target); i++ {
		tVal := target[i]
		oVal := output[i]

		// Calculate Error magnitude
		diff := float32(math.Abs(float64(tVal - oVal)))

		// DEADBAND: If error is small, don't train. This prevents "jitter" and over-fitting background.
		if diff < 0.1 {
			continue
		}

		// Determine Weight
		// If Target is Color (>0.05), we care A LOT (10.0)
		// If Target is BG, we care a little (0.1)
		weight := WeightBackground
		if tVal > 0.05 {
			weight = WeightColor
		}

		// Calculate Effective LR for this specific pixel
		// LR * Weight * Error Magnitude (Bigger error = Bigger push)
		stepLR := baseLR * weight * diff

		// Apply Heuristic Update to this specific pixel index
		ts.TweenStep(net, sample.Input, i, len(target), stepLR)
	}
}

// === NETWORK ARCHITECTURE (Same as Test 35/36) ===

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
			{0, 0, 0, 0}, {1, 0, 1, 0}, {2, 1, 0, 0}, {3, 1, 1, 0},
		},
	}
}

// === BRAIN FACTORIES ===
func createMHABrain() nn.LayerConfig {
	mha := nn.LayerConfig{Type: nn.LayerMultiHeadAttention, DModel: DModel, NumHeads: NumHeads, SeqLength: 1}
	initMHAWeights(&mha)
	return mha
}
func createLSTMBrain() nn.LayerConfig {
	lstm := nn.LayerConfig{Type: nn.LayerLSTM, RNNInputSize: DModel, HiddenSize: LSTMHidden, SeqLength: 1, OutputHeight: DModel}
	initLSTMWeights(&lstm)
	return lstm
}
func createCNNBrain() nn.LayerConfig {
	cnn := nn.LayerConfig{Type: nn.LayerConv2D, InputHeight: ConvGridSize, InputWidth: ConvGridSize, InputChannels: 1, Filters: ConvFilters, KernelSize: ConvKernel, Stride: 1, Padding: 1, OutputHeight: ConvGridSize, OutputWidth: ConvGridSize, Activation: nn.ActivationLeakyReLU}
	initCNNWeights(&cnn)
	return cnn
}

// === WEIGHT INIT & UTILS ===
func initMHAWeights(cfg *nn.LayerConfig) {
	cfg.QWeights = make([]float32, DModel*DModel)
	cfg.KWeights = make([]float32, DModel*DModel)
	cfg.VWeights = make([]float32, DModel*DModel)
	cfg.OutputWeight = make([]float32, DModel*DModel)
	cfg.QBias = make([]float32, DModel)
	cfg.KBias = make([]float32, DModel)
	cfg.VBias = make([]float32, DModel)
	cfg.OutputBias = make([]float32, DModel)
	scale := InitScale / float32(math.Sqrt(float64(DModel/NumHeads)))
	initRandom(cfg.QWeights, scale)
	initRandom(cfg.KWeights, scale)
	initRandom(cfg.VWeights, scale)
	initRandom(cfg.OutputWeight, scale)
}
func initLSTMWeights(cfg *nn.LayerConfig) {
	h, i := cfg.HiddenSize, cfg.RNNInputSize
	cfg.WeightIH_i = make([]float32, h*i)
	cfg.WeightIH_f = make([]float32, h*i)
	cfg.WeightIH_g = make([]float32, h*i)
	cfg.WeightIH_o = make([]float32, h*i)
	cfg.WeightHH_i = make([]float32, h*h)
	cfg.WeightHH_f = make([]float32, h*h)
	cfg.WeightHH_g = make([]float32, h*h)
	cfg.WeightHH_o = make([]float32, h*h)
	cfg.BiasH_i = make([]float32, h)
	cfg.BiasH_f = make([]float32, h)
	cfg.BiasH_g = make([]float32, h)
	cfg.BiasH_o = make([]float32, h)
	scale := InitScale / float32(math.Sqrt(float64(h)))
	initRandom(cfg.WeightIH_i, scale)
	initRandom(cfg.WeightIH_f, scale)
	initRandom(cfg.WeightIH_g, scale)
	initRandom(cfg.WeightIH_o, scale)
	initRandom(cfg.WeightHH_i, scale)
	initRandom(cfg.WeightHH_f, scale)
	initRandom(cfg.WeightHH_g, scale)
	initRandom(cfg.WeightHH_o, scale)
	for j := range cfg.BiasH_f {
		cfg.BiasH_f[j] = 1.0
	}
}
func initCNNWeights(cfg *nn.LayerConfig) {
	fanIn := ConvKernel * ConvKernel
	kernelSize := ConvFilters * ConvKernel * ConvKernel
	cfg.Kernel = make([]float32, kernelSize)
	cfg.Bias = make([]float32, ConvFilters)
	scale := InitScale / float32(math.Sqrt(float64(fanIn)))
	initRandom(cfg.Kernel, scale)
}

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
				if exp > 0 { // Color only
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
			trainSamples = append(trainSamples, Sample{Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output), Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output), Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID})
		}
	}
	holdout := len(trainSamples) / 5
	evalSamples = trainSamples[len(trainSamples)-holdout:]
	trainSamples = trainSamples[:len(trainSamples)-holdout]
	return trainSamples, evalSamples
}

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║   Final Accuracy:       %5.1f%%                                                      ║\n", results.FinalAccuracy)
	fmt.Printf("║   Color Accuracy:       %5.1f%%                                                      ║\n", results.FinalWeightedAccuracy)
	fmt.Printf("║   Tasks Solved:         %d                                                            ║\n", results.TasksSolved)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")
	if len(results.SolvedTaskIDs) > 0 {
		fmt.Println("Solved IDs:", results.SolvedTaskIDs)
	}
}

func saveResults(results *Results) {
	output := map[string]interface{}{"final_accuracy": results.FinalAccuracy, "color_accuracy": results.FinalWeightedAccuracy, "tasks_solved": results.TasksSolved, "solved_ids": results.SolvedTaskIDs}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test36b_results.json", data, 0644)
	fmt.Println("\n✅ Results saved.")
}

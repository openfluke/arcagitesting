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

// Test 36c: NORMALIZED HIVE - LayerNorm after Grid Scatter
//
// Change from Test 36b:
// Add a LayerNorm layer after the Grid Scatter Hive to stabilize activations
// before the merger layer.
//
// Architecture:
//   Layer 0: Input Embedding (900 -> 64)
//   Layer 1: Grid Scatter Hive (2x2, 4 brains)
//   Layer 2: LayerNorm (Normalize the 256-dim concat output)  <-- NEW
//   Layer 3: Merger Dense (256 -> 64)
//   Layer 4: Output Dense (64 -> 900)

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize
	NumTasks    = 400
	BatchSize   = 50
	NumEpochs   = 600

	LearningRate = float32(0.001)
	BudgetScale  = float32(0.8)

	WeightBackground = float32(0.1)
	WeightColor      = float32(10.0)

	DModel       = 64
	NumHeads     = 4
	LSTMHidden   = 64
	ConvGridSize = 8
	ConvFilters  = 8
	ConvKernel   = 3
	InitScale    = float32(0.5)
)

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
	fmt.Println("║     Test 36c: NORMALIZED HIVE (LayerNorm after Grid Scatter)                        ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Architecture: Embed -> GridScatter -> LayerNorm -> Merger -> Output             ║")
	fmt.Println("║     Goal: Stabilize activations for better gradient flow                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks.\n", len(tasks))

	net := createNormalizedHiveNetwork()
	numLayers := net.TotalLayers()
	state := net.InitStepState(InputSize)
	fmt.Printf("🏗️  Created Normalized Hive: %d layers\n", numLayers)

	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = false
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory:         make([]float64, NumEpochs),
		WeightedAccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:           make([]float32, NumEpochs),
		SolvedTaskIDs:           []string{},
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     📊 NORMALIZED HIVE TRAINING 📊")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	start := time.Now()
	sampleIdx := 0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++

			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output := state.GetOutput()

			applyMultiPointTween(ts, net, sample, output, LearningRate)
		}

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

func applyMultiPointTween(ts *nn.TweenState, net *nn.Network, sample Sample, output []float32, baseLR float32) {
	target := sample.Target
	for i := 0; i < len(target); i++ {
		tVal := target[i]
		oVal := output[i]
		diff := float32(math.Abs(float64(tVal - oVal)))
		if diff < 0.1 {
			continue
		}
		weight := WeightBackground
		if tVal > 0.05 {
			weight = WeightColor
		}
		stepLR := baseLR * weight * diff
		ts.TweenStep(net, sample.Input, i, len(target), stepLR)
	}
}

// === NORMALIZED HIVE NETWORK (5 layers now) ===

func createNormalizedHiveNetwork() *nn.Network {
	totalLayers := 5 // Added LayerNorm
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1
	layerIdx := 0

	// Layer 0: Input Embedding
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: Grid Scatter Hive
	parallelLayer := createGridScatterHive()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Layer 2: LayerNorm (NEW) - normalizes the 256-dim concat output
	normSize := DModel * 4 // 64 * 4 = 256
	normLayer := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: normSize,
		Gamma:    make([]float32, normSize),
		Beta:     make([]float32, normSize),
		Epsilon:  1e-5,
	}
	// Initialize gamma to 1, beta to 0
	for i := range normLayer.Gamma {
		normLayer.Gamma[i] = 1.0
		normLayer.Beta[i] = 0.0
	}
	net.SetLayer(0, 0, layerIdx, normLayer)
	layerIdx++

	// Layer 3: Merger Dense
	mergerLayer := nn.InitDenseLayer(DModel*4, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 4: Output Dense
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

// === METRICS ===

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

// === HELPERS ===

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
	fmt.Println("║                      📊 NORMALIZED HIVE - FINAL RESULTS 📊                          ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Final Accuracy:       %5.1f%%                                                      ║\n", results.FinalAccuracy)
	fmt.Printf("║   Color Accuracy:       %5.1f%%                                                      ║\n", results.FinalWeightedAccuracy)
	fmt.Printf("║   Tasks Solved:         %d                                                            ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:        %.1fs                                                         ║\n", results.TrainTime.Seconds())
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")
	if len(results.SolvedTaskIDs) > 0 {
		fmt.Println("\n📋 Solved Task IDs:")
		for _, id := range results.SolvedTaskIDs {
			fmt.Printf("   - %s\n", id)
		}
	}
}

func saveResults(results *Results) {
	output := map[string]interface{}{
		"final_accuracy":            results.FinalAccuracy,
		"color_accuracy":            results.FinalWeightedAccuracy,
		"tasks_solved":              results.TasksSolved,
		"solved_ids":                results.SolvedTaskIDs,
		"train_time_sec":            results.TrainTime.Seconds(),
		"accuracy_history":          results.AccuracyHistory,
		"weighted_accuracy_history": results.WeightedAccuracyHistory,
		"meta": map[string]interface{}{
			"architecture": "Normalized Hive (GridScatter + LayerNorm)",
			"layers":       5,
		},
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test36c_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test36c_results.json")
}

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

// Test 30b: ADD HIVE - Parallel Branches with Element-wise Addition
//
// CombineMode: "add" - Adds all branch outputs element-wise
// All branches must have same output size

const (
	MaxGridSize   = 30
	InputSize     = MaxGridSize * MaxGridSize
	NumTasks      = 400
	BatchSize     = 100
	NumEpochs     = 1400
	LearningRate  = float32(0.5)
	InitScale     = float32(0.5)
	BudgetScale   = float32(1.2)
	DModel        = 32
	NumHeads      = 4
	GrokThreshold = 20.0
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
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 30b: ADD HIVE - Element-wise Addition Combine Mode                         ║")
	fmt.Println("║     ➕ CombineMode: add (Sum all branch outputs)                                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train, %d eval\n\n", len(tasks), len(trainSamples), len(evalSamples))

	net := createAddHiveNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Add Hive Network: %d layers\n", numLayers)

	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{AccuracyHistory: make([]float64, NumEpochs), BudgetHistory: make([]float32, NumEpochs), GrokEpoch: -1}

	fmt.Println("\n━━━ ADD HIVE TRAINING ━━━")
	start := time.Now()
	sampleIdx := 0
	prevAcc := 0.0

	for epoch := 0; epoch < NumEpochs; epoch++ {
		for i := 0; i < BatchSize; i++ {
			sample := trainSamples[sampleIdx%len(trainSamples)]
			sampleIdx++
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output := state.GetOutput()
			ts.ForwardPass(net, sample.Input)
			applyChainRuleUpdate(ts, net, sample, output, LearningRate)
		}

		acc := measureAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)
		results.AccuracyHistory[epoch] = acc
		results.BudgetHistory[epoch] = budget

		if results.GrokEpoch == -1 && acc > GrokThreshold && prevAcc < GrokThreshold {
			results.GrokEpoch = epoch + 1
			fmt.Printf("\n  🎯 GROK at Epoch %d: %.1f%% → %.1f%%\n", epoch+1, prevAcc, acc)
		}
		if (epoch+1)%10 == 0 {
			fmt.Printf("  Epoch %3d | Acc: %5.1f%% | Budget: %.3f\n", epoch+1, acc, budget)
		}
		prevAcc = acc
	}

	results.TrainTime = time.Since(start)
	results.FinalAccuracy = results.AccuracyHistory[NumEpochs-1]
	results.FinalBudget = results.BudgetHistory[NumEpochs-1]
	results.TasksSolved, results.SolvedTaskIDs = measureSolvedTasks(net, evalSamples, numLayers, state)

	fmt.Printf("\n✅ Final: %.1f%% | Grok: %d | Time: %.1fs\n", results.FinalAccuracy, results.GrokEpoch, results.TrainTime.Seconds())
	saveResults(results, "test30b_add_results.json", "add")
}

func createAddHiveNetwork() *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, 0, inputLayer)

	parallelLayer := createAddHive()
	net.SetLayer(0, 0, 1, parallelLayer)

	// Add mode: output is same size as each branch = DModel
	mergerLayer := nn.InitDenseLayer(DModel, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, 2, mergerLayer)

	outputLayer := nn.InitDenseLayer(DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, 3, outputLayer)

	return net
}

func createAddHive() nn.LayerConfig {
	// All branches must output same size for add mode
	return nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			createMHABrain(),
			createMHABrain(),
			createMHABrain(),
			createMHABrain(),
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

func measureAccuracy(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
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
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		allMatch := true
		for r := 0; r < sample.Height && allMatch; r++ {
			for c := 0; c < sample.Width && allMatch; c++ {
				idx := r*MaxGridSize + c
				if idx < len(output) && idx < len(sample.Target) {
					pred := clampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					exp := clampInt(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
					if pred != exp {
						allMatch = false
					}
				}
			}
		}
		if allMatch && !seen[sample.TaskID] {
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
				Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID,
			})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]), TaskID: task.ID,
			})
		}
	}
	if len(evalSamples) == 0 && len(trainSamples) > 5 {
		holdout := len(trainSamples) / 5
		evalSamples = trainSamples[len(trainSamples)-holdout:]
		trainSamples = trainSamples[:len(trainSamples)-holdout]
	}
	return
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

func saveResults(results *Results, filename, mode string) {
	output := map[string]interface{}{
		"final_accuracy": results.FinalAccuracy, "grok_epoch": results.GrokEpoch,
		"tasks_solved": results.TasksSolved, "train_time_sec": results.TrainTime.Seconds(),
		"accuracy_history": results.AccuracyHistory,
		"meta":             map[string]interface{}{"combine_mode": mode, "architecture": "4-brain hive (all MHA for same size)"},
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile(filename, data, 0644)
	fmt.Printf("✅ Saved to %s\n", filename)
}

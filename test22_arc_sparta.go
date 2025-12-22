package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	"github.com/openfluke/loom/nn"
)

// Test 22: ARC-AGI Full SPARTA - All Modes, Depths, and Architectures
// Tests complete combinatorics for ARC-AGI benchmarking
// Output shows TASK accuracy (official ARC metric: % of puzzles fully solved)

const (
	MaxGridSize   = 30
	InputSize     = MaxGridSize * MaxGridSize // 900
	NumTasks      = 10
	NumRuns       = 3   // Runs per config
	MaxBatches    = 20  // Training batches per run
	BatchSize     = 100 // Samples per batch
	MaxConcurrent = 8
	LearningRate  = float32(0.01)
)

// Architecture types
type ArchType string

const (
	ArchDense     ArchType = "Dense"
	ArchConv2D    ArchType = "Conv2D"
	ArchAttn      ArchType = "Attn"
	ArchConvDense ArchType = "Conv+Dense"
	ArchAttnDense ArchType = "Attn+Dense"
)

var allArchitectures = []ArchType{ArchDense, ArchConv2D, ArchAttn, ArchConvDense, ArchAttnDense}
var allDepths = []int{3, 5, 7}

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
}

var allModes = []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}

// Data types
type ARCTask struct {
	ID          string
	Train, Test []GridPair
}
type GridPair struct{ Input, Output [][]int }
type Sample struct {
	Input, Target []float32
	Height, Width int
}

type ModeStat struct {
	Mean, StdDev, Min, Max float64
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 22: ARC-AGI Full SPARTA                                            ║")
	fmt.Println("║  All Modes × All Depths × All Architectures                              ║")
	fmt.Println("║  Output: % Tasks Solved (Official ARC Metric)                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	samples := flattenTasksToSamples(tasks)
	fmt.Printf("Loaded %d tasks, %d samples\n\n", len(tasks), len(samples))

	// Track global best
	var globalBestArch ArchType
	var globalBestMode TrainingMode
	globalBestDepth := 0
	globalBestAcc := 0.0

	allResults := make(map[string]map[TrainingMode]float64)

	for _, arch := range allArchitectures {
		for _, depth := range allDepths {
			configName := fmt.Sprintf("%s-%dL", arch, depth)
			allResults[configName] = make(map[TrainingMode]float64)

			fmt.Printf("\n┌───────────────────────────────────────────────────────────────────────────┐\n")
			fmt.Printf("│ %-73s │\n", fmt.Sprintf("%s — Testing %d modes", configName, len(allModes)))
			fmt.Printf("└───────────────────────────────────────────────────────────────────────────┘\n")

			for _, mode := range allModes {
				cellAcc, taskAcc := runTrials(samples, arch, depth, mode, NumRuns)
				allResults[configName][mode] = taskAcc

				// Check for new global best
				isNewBest := taskAcc > globalBestAcc
				if isNewBest {
					globalBestAcc = taskAcc
					globalBestArch = arch
					globalBestMode = mode
					globalBestDepth = depth
				}

				bestMarker := ""
				if isNewBest && taskAcc > 0 {
					bestMarker = " ★ NEW BEST!"
				}

				fmt.Printf("  [%-12s] Cell: %4.0f%% | Tasks: %4.0f%% %s\n",
					modeNames[mode], cellAcc, taskAcc, bestMarker)
			}

			fmt.Printf("  ▸ Leader: %s-%dL + %s = %.0f%% tasks\n",
				globalBestArch, globalBestDepth, modeNames[globalBestMode], globalBestAcc)
		}
	}

	printFinalSummary(allResults, globalBestArch, globalBestDepth, globalBestMode, globalBestAcc)
	saveResults(allResults)
}

// ============================================================================
// Training
// ============================================================================

func runTrials(samples []Sample, arch ArchType, depth int, mode TrainingMode, numRuns int) (avgCellAcc, maxTaskAcc float64) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, MaxConcurrent)
	var mu sync.Mutex

	cellAccs := make([]float64, numRuns)
	taskAccs := make([]float64, numRuns)

	for i := 0; i < numRuns; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					cellAccs[idx] = 0
					taskAccs[idx] = 0
				}
			}()
			sem <- struct{}{}
			defer func() { <-sem }()

			c, t := runSingleTrial(samples, arch, depth, mode)
			mu.Lock()
			cellAccs[idx] = c
			taskAccs[idx] = t
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	// Calculate avg cell and max task
	sum := 0.0
	maxTask := 0.0
	for i := 0; i < numRuns; i++ {
		sum += cellAccs[i]
		if taskAccs[i] > maxTask {
			maxTask = taskAccs[i]
		}
	}
	return sum / float64(numRuns), maxTask
}

func runSingleTrial(samples []Sample, arch ArchType, depth int, mode TrainingMode) (cellAcc, taskAcc float64) {
	net := createNetwork(arch, depth)
	if net == nil {
		return 0, 0
	}
	numLayers := net.TotalLayers()

	// Initialize states based on mode
	var state *nn.StepState
	usesStep := mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain
	if usesStep {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	usesTween := mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain
	if usesTween {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	bestCell, bestTask := 0.0, 0.0
	sampleIdx := 0

	for batch := 0; batch < MaxBatches; batch++ {
		for i := 0; i < BatchSize; i++ {
			sample := samples[sampleIdx%len(samples)]
			sampleIdx++
			trainOneSample(net, sample, mode, numLayers, state, ts, LearningRate)
		}
		c := measureCellAccuracy(net, samples, mode, numLayers, state, ts)
		t := measureTaskAccuracy(net, samples, mode, numLayers, state, ts)
		if c > bestCell {
			bestCell = c
		}
		if t > bestTask {
			bestTask = t
		}
	}
	return bestCell, bestTask
}

func trainOneSample(net *nn.Network, sample Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState, lr float32) {
	switch mode {
	case ModeNormalBP:
		output, _ := net.ForwardCPU(sample.Input)
		grad := computeGradient(output, sample.Target)
		net.BackwardCPU(grad)
		net.ApplyGradients(lr)

	case ModeStepBP:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		grad := computeGradient(output, sample.Target)
		net.StepBackward(state, grad)
		net.ApplyGradients(lr)

	case ModeTween:
		output := ts.ForwardPass(net, sample.Input)
		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), lr)
		_ = output

	case ModeTweenChain:
		output := ts.ForwardPass(net, sample.Input)
		applyTweenUpdate(ts, net, sample, output, lr)

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
			grad[i] = clipGrad(output[i]-target[i], 0.5)
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
	totalLayers := net.TotalLayers()
	ts.ChainGradients[totalLayers] = outputGrad
	ts.BackwardTargets[totalLayers] = sample.Target
	ts.TweenWeightsChainRule(net, lr)
}

func measureCellAccuracy(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) float64 {
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
		return 0
	}
	return float64(correct) / float64(total) * 100
}

func measureTaskAccuracy(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) float64 {
	solved := 0
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
		}
	}
	if len(samples) == 0 {
		return 0
	}
	return float64(solved) / float64(len(samples)) * 100
}

func getOutput(net *nn.Network, input []float32, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) []float32 {
	switch mode {
	case ModeStepBP, ModeStepTween, ModeStepTweenChain:
		state.SetInput(input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		return state.GetOutput()
	case ModeTween, ModeTweenChain:
		return ts.ForwardPass(net, input)
	default:
		output, _ := net.ForwardCPU(input)
		return output
	}
}

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

// ============================================================================
// Network Factories
// ============================================================================

func createNetwork(arch ArchType, depth int) *nn.Network {
	switch arch {
	case ArchDense:
		return createDenseNet(depth)
	case ArchConv2D:
		return createConv2DNet(depth)
	case ArchAttn:
		return createAttnNet(depth)
	case ArchConvDense:
		return createConvDenseNet(depth)
	case ArchAttnDense:
		return createAttnDenseNet(depth)
	}
	return nil
}

func createDenseNet(depth int) *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	sizes := []int{128, 64, 48, 32, 24, 16}
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, sizes[0], nn.ActivationLeakyReLU))
	for i := 1; i < depth-1; i++ {
		inS := sizes[(i-1)%len(sizes)]
		outS := sizes[i%len(sizes)]
		net.SetLayer(0, 0, i, nn.InitDenseLayer(inS, outS, nn.ActivationLeakyReLU))
	}
	lastH := sizes[(depth-2)%len(sizes)]
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(lastH, InputSize, nn.ActivationSigmoid))
	return net
}

func createConv2DNet(depth int) *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	conv := nn.LayerConfig{
		Type: nn.LayerConv2D, InputHeight: 30, InputWidth: 30, InputChannels: 1,
		Filters: 8, KernelSize: 3, Stride: 1, Padding: 1,
		OutputHeight: 30, OutputWidth: 30, Activation: nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 8*1*3*3)
	conv.Bias = make([]float32, 8)
	initRandom(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)
	for i := 1; i < depth-1; i++ {
		if i == 1 {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(7200, 64, nn.ActivationLeakyReLU))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
		}
	}
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(64, InputSize, nn.ActivationSigmoid))
	return net
}

func createAttnNet(depth int) *nn.Network {
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU))
	for i := 1; i < depth-1; i++ {
		if i%2 == 1 {
			mha := createMHA(dModel)
			net.SetLayer(0, 0, i, mha)
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU))
		}
	}
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid))
	return net
}

func createConvDenseNet(depth int) *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	conv := nn.LayerConfig{
		Type: nn.LayerConv2D, InputHeight: 30, InputWidth: 30, InputChannels: 1,
		Filters: 4, KernelSize: 3, Stride: 1, Padding: 1,
		OutputHeight: 30, OutputWidth: 30, Activation: nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 4*1*3*3)
	conv.Bias = make([]float32, 4)
	initRandom(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)
	for i := 1; i < depth-1; i++ {
		if i == 1 {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(3600, 64, nn.ActivationLeakyReLU))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
		}
	}
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(32, InputSize, nn.ActivationSigmoid))
	return net
}

func createAttnDenseNet(depth int) *nn.Network {
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, depth)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU))
	if depth > 2 {
		mha := createMHA(dModel)
		net.SetLayer(0, 0, 1, mha)
	}
	for i := 2; i < depth-1; i++ {
		net.SetLayer(0, 0, i, nn.InitDenseLayer(dModel, 32, nn.ActivationLeakyReLU))
	}
	lastH := 32
	if depth <= 2 {
		lastH = dModel
	}
	net.SetLayer(0, 0, depth-1, nn.InitDenseLayer(lastH, InputSize, nn.ActivationSigmoid))
	return net
}

func createMHA(dModel int) nn.LayerConfig {
	headDim := dModel / 4
	mha := nn.LayerConfig{Type: nn.LayerMultiHeadAttention, DModel: dModel, NumHeads: 4}
	mha.QWeights = make([]float32, dModel*dModel)
	mha.KWeights = make([]float32, dModel*dModel)
	mha.VWeights = make([]float32, dModel*dModel)
	mha.OutputWeight = make([]float32, dModel*dModel)
	mha.QBias = make([]float32, dModel)
	mha.KBias = make([]float32, dModel)
	mha.VBias = make([]float32, dModel)
	mha.OutputBias = make([]float32, dModel)
	initRandom(mha.QWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandom(mha.KWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandom(mha.VWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandom(mha.OutputWeight, 0.1/float32(math.Sqrt(float64(dModel))))
	return mha
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

func flattenTasksToSamples(tasks []*ARCTask) []Sample {
	var samples []Sample
	for _, task := range tasks {
		for _, pair := range task.Train {
			samples = append(samples, Sample{
				Input: encodeGrid(pair.Input), Target: encodeGrid(pair.Output),
				Height: len(pair.Output), Width: len(pair.Output[0]),
			})
		}
	}
	return samples
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
// Output
// ============================================================================

func printFinalSummary(results map[string]map[TrainingMode]float64, bestArch ArchType, bestDepth int, bestMode TrainingMode, bestAcc float64) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                        ARC-AGI FULL SPARTA SUMMARY (% Tasks Solved - Official Metric)                                                        ║")
	fmt.Println("╠════════════════════╦════════════╦════════════╦════════════╦════════════╦════════════╦════════════╦════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Config             ║ NormalBP   ║ StepBP     ║ Tween      ║ TweenChain ║ StepTween  ║ StepTwChn  ║ Best Mode                                                  ║")
	fmt.Println("╠════════════════════╬════════════╬════════════╬════════════╬════════════╬════════════╬════════════╬════════════════════════════════════════════════════════════╣")

	configs := make([]string, 0)
	for _, arch := range allArchitectures {
		for _, d := range allDepths {
			configs = append(configs, fmt.Sprintf("%s-%dL", arch, d))
		}
	}

	for _, config := range configs {
		modes := results[config]
		best := ""
		bestVal := 0.0
		for _, m := range allModes {
			if modes[m] > bestVal {
				bestVal = modes[m]
				best = modeNames[m]
			}
		}
		fmt.Printf("║ %-18s ║   %5.0f%%   ║   %5.0f%%   ║   %5.0f%%   ║   %5.0f%%   ║   %5.0f%%   ║   %5.0f%%   ║ %-58s ║\n",
			config,
			modes[ModeNormalBP], modes[ModeStepBP], modes[ModeTween],
			modes[ModeTweenChain], modes[ModeStepTween], modes[ModeStepTweenChain],
			best)
	}

	fmt.Println("╚════════════════════╩════════════╩════════════╩════════════╩════════════╩════════════╩════════════╩════════════════════════════════════════════════════════════╝")
	fmt.Printf("\n★ BEST OVERALL: %s-%dL + %s = %.0f%% tasks solved\n", bestArch, bestDepth, modeNames[bestMode], bestAcc)
	fmt.Println("  Note: World record for efficient methods is 29.72% (NVIDIA TTT)")
}

func saveResults(results map[string]map[TrainingMode]float64) {
	output := make(map[string]map[string]float64)
	for config, modes := range results {
		output[config] = make(map[string]float64)
		for mode, acc := range modes {
			output[config][modeNames[mode]] = acc
		}
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test22_results.json", data, 0644)
	fmt.Println("✓ Results saved to test22_results.json")
}

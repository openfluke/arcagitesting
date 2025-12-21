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

// Test 21: ARC-AGI Architecture Comparison (SPARTA Style)
// Tests all layer types and blended architectures on ARC-AGI tasks
// with statistical parallel runs to find the best architecture.

const (
	MaxGridSize   = 30
	InputSize     = MaxGridSize * MaxGridSize // 900
	NumTasks      = 10
	NumRuns       = 5   // Runs per config (reduced for speed)
	MaxBatches    = 30  // Training batches per run
	BatchSize     = 100 // Samples per batch
	MaxConcurrent = 8   // Parallel goroutines
	LearningRate  = float32(0.01)
)

// Architecture types
type ArchType string

const (
	ArchDense     ArchType = "Dense"
	ArchConv2D    ArchType = "Conv2D"
	ArchLSTM      ArchType = "LSTM"
	ArchAttn      ArchType = "Attn"
	ArchConvDense ArchType = "Conv+Dense"
	ArchConvLSTM  ArchType = "Conv+LSTM"
	ArchAttnDense ArchType = "Attn+Dense"
)

var allArchitectures = []ArchType{
	ArchDense, ArchConv2D, ArchAttn,
	ArchConvDense, ArchAttnDense,
	// Note: LSTM architectures disabled due to dimension mismatch in lstm.go
}

type TrainingMode int

const (
	ModeStepBP TrainingMode = iota
	ModeTweenChain
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeStepBP:         "StepBP",
	ModeTweenChain:     "TweenChain",
	ModeStepTweenChain: "StepTweenChain",
}

type RunResult struct {
	Architecture string  `json:"architecture"`
	Mode         string  `json:"mode"`
	FinalAcc     float64 `json:"final_accuracy"`
	BestAcc      float64 `json:"best_accuracy"`
	Batches      int     `json:"batches_trained"`
}

type ArchSummary struct {
	Name        string              `json:"name"`
	Modes       map[string]ModeStat `json:"modes"`
	BestMode    string              `json:"best_mode"`
	BestAccMean float64             `json:"best_acc_mean"`
}

type ModeStat struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
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
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 21: ARC-AGI Architecture Comparison (SPARTA Style)                 ║")
	fmt.Println("║  Testing: Dense, Conv2D, LSTM, Attn + Blended Architectures              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	samples := flattenTasksToSamples(tasks)
	fmt.Printf("Loaded %d tasks, %d samples\n\n", len(tasks), len(samples))

	modes := []TrainingMode{ModeStepBP, ModeTweenChain, ModeStepTweenChain}
	allResults := make(map[ArchType]map[TrainingMode][]float64)
	var mu sync.Mutex

	// Track best overall as we go
	var bestArch ArchType
	var bestMode TrainingMode
	bestAcc := 0.0

	for _, arch := range allArchitectures {
		allResults[arch] = make(map[TrainingMode][]float64)

		fmt.Printf("\n┌─────────────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ %-67s │\n", fmt.Sprintf("%s — Running %d trials per mode", arch, NumRuns))
		fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

		for _, mode := range modes {
			results := runParallelTrials(samples, arch, mode, NumRuns)

			mu.Lock()
			allResults[arch][mode] = results
			mu.Unlock()

			stat := calcStats(results)

			// Check if this is new best
			isNewBest := stat.Max > bestAcc
			if isNewBest {
				bestAcc = stat.Max
				bestArch = arch
				bestMode = mode
			}

			// Include best indicator
			bestIndicator := ""
			if isNewBest {
				bestIndicator = " ★ NEW BEST!"
			}

			fmt.Printf("  [%-14s] [%-12s] Acc: %5.1f%% (±%4.1f%%) | Best: %.0f%%%s\n",
				arch, modeNames[mode], stat.Mean, stat.StdDev, stat.Max, bestIndicator)
		}

		// Show current leader after each architecture
		fmt.Printf("  ▸ Current Leader: %s + %s = %.1f%%\n", bestArch, modeNames[bestMode], bestAcc)
	}

	printComparisonTable(allResults)
	saveResults(allResults)
}

// ============================================================================
// Parallel Execution
// ============================================================================

func runParallelTrials(samples []Sample, arch ArchType, mode TrainingMode, numRuns int) []float64 {
	results := make([]float64, numRuns)
	var wg sync.WaitGroup
	sem := make(chan struct{}, MaxConcurrent)

	for i := 0; i < numRuns; i++ {
		wg.Add(1)
		go func(runIdx int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					results[runIdx] = 0 // Mark as failed
				}
			}()
			sem <- struct{}{}
			defer func() { <-sem }()

			acc := runSingleTrial(samples, arch, mode)
			results[runIdx] = acc
		}(i)
	}
	wg.Wait()
	return results
}

func runSingleTrial(samples []Sample, arch ArchType, mode TrainingMode) float64 {
	net := createNetwork(arch)
	if net == nil {
		return 0
	}
	numLayers := net.TotalLayers()

	var state *nn.StepState
	usesStep := mode == ModeStepBP || mode == ModeStepTweenChain
	if usesStep {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTweenChain || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		ts.Config.UseChainRule = true
	}

	bestAcc := 0.0
	sampleIdx := 0

	for batch := 0; batch < MaxBatches; batch++ {
		// Training phase
		for i := 0; i < BatchSize; i++ {
			sample := samples[sampleIdx%len(samples)]
			sampleIdx++
			trainOneSample(net, sample, mode, numLayers, state, ts, LearningRate)
		}

		// Measure accuracy
		acc := measureAccuracy(net, samples, mode, numLayers, state, ts)
		if acc > bestAcc {
			bestAcc = acc
		}
	}

	return bestAcc
}

func trainOneSample(net *nn.Network, sample Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState, lr float32) {
	switch mode {
	case ModeStepBP:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		grad := make([]float32, len(output))
		for j := range output {
			if j < len(sample.Target) {
				grad[j] = clipGrad(output[j]-sample.Target[j], 0.5)
			}
		}
		net.StepBackward(state, grad)
		net.ApplyGradients(lr)

	case ModeStepTweenChain:
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()
		ts.ForwardPass(net, sample.Input)
		applyTweenUpdate(ts, net, sample, output, lr)

	case ModeTweenChain:
		output := ts.ForwardPass(net, sample.Input)
		applyTweenUpdate(ts, net, sample, output, lr)
	}
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

func measureAccuracy(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		var output []float32
		if mode == ModeStepBP || mode == ModeStepTweenChain {
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		} else {
			output = ts.ForwardPass(net, sample.Input)
		}
		for r := 0; r < sample.Height; r++ {
			for c := 0; c < sample.Width; c++ {
				idx := r*MaxGridSize + c
				if idx < len(output) && idx < len(sample.Target) {
					pred := int(math.Round(float64(output[idx]) * 9.0))
					exp := int(math.Round(float64(sample.Target[idx]) * 9.0))
					if pred < 0 {
						pred = 0
					}
					if pred > 9 {
						pred = 9
					}
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

// ============================================================================
// Network Factories
// ============================================================================

func createNetwork(arch ArchType) *nn.Network {
	switch arch {
	case ArchDense:
		return createDenseNet()
	case ArchConv2D:
		return createConv2DNet()
	case ArchLSTM:
		return createLSTMNet()
	case ArchAttn:
		return createAttnNet()
	case ArchConvDense:
		return createConvDenseNet()
	case ArchConvLSTM:
		return createConvLSTMNet()
	case ArchAttnDense:
		return createAttnDenseNet()
	}
	return nil
}

func createDenseNet() *nn.Network {
	// Input(900) -> 64 -> 32 -> 900
	net := nn.NewNetwork(InputSize, 1, 1, 3)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, InputSize, nn.ActivationSigmoid))
	return net
}

func createConv2DNet() *nn.Network {
	// Conv2D(30x30) -> Flatten -> Dense -> Output(900)
	net := nn.NewNetwork(InputSize, 1, 1, 3)
	net.BatchSize = 1

	conv := nn.LayerConfig{
		Type:        nn.LayerConv2D,
		InputHeight: 30, InputWidth: 30, InputChannels: 1,
		Filters: 8, KernelSize: 3, Stride: 1, Padding: 1,
		OutputHeight: 30, OutputWidth: 30,
		Activation: nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 8*1*3*3)
	conv.Bias = make([]float32, 8)
	initRandomSlice(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)

	// Flatten: 8*30*30 = 7200 -> 64
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(7200, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, InputSize, nn.ActivationSigmoid))
	return net
}

func createLSTMNet() *nn.Network {
	// Dense -> LSTM -> Dense
	net := nn.NewNetwork(InputSize, 1, 1, 3)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, 64, nn.ActivationLeakyReLU))
	lstm := nn.InitLSTMLayer(8, 8, 1, 64) // seqLen=8, inputDim=8, batchSize=1, hiddenSize=64
	net.SetLayer(0, 0, 1, lstm)
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, InputSize, nn.ActivationSigmoid))
	return net
}

func createAttnNet() *nn.Network {
	// Attention -> Dense -> Output
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, 3)
	net.BatchSize = 1

	// First project to dModel size
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU))

	mha := nn.LayerConfig{
		Type:     nn.LayerMultiHeadAttention,
		DModel:   dModel,
		NumHeads: 4,
	}
	headDim := dModel / 4
	mha.QWeights = make([]float32, dModel*dModel)
	mha.KWeights = make([]float32, dModel*dModel)
	mha.VWeights = make([]float32, dModel*dModel)
	mha.OutputWeight = make([]float32, dModel*dModel)
	mha.QBias = make([]float32, dModel)
	mha.KBias = make([]float32, dModel)
	mha.VBias = make([]float32, dModel)
	mha.OutputBias = make([]float32, dModel)
	initRandomSlice(mha.QWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandomSlice(mha.KWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandomSlice(mha.VWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandomSlice(mha.OutputWeight, 0.1/float32(math.Sqrt(float64(dModel))))
	net.SetLayer(0, 0, 1, mha)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid))
	return net
}

func createConvDenseNet() *nn.Network {
	// Conv2D -> Dense -> Dense -> Output
	net := nn.NewNetwork(InputSize, 1, 1, 4)
	net.BatchSize = 1

	conv := nn.LayerConfig{
		Type:        nn.LayerConv2D,
		InputHeight: 30, InputWidth: 30, InputChannels: 1,
		Filters: 4, KernelSize: 3, Stride: 1, Padding: 1,
		OutputHeight: 30, OutputWidth: 30,
		Activation: nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 4*1*3*3)
	conv.Bias = make([]float32, 4)
	initRandomSlice(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(3600, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(32, InputSize, nn.ActivationSigmoid))
	return net
}

func createConvLSTMNet() *nn.Network {
	// Conv2D -> LSTM -> Dense
	net := nn.NewNetwork(InputSize, 1, 1, 4)
	net.BatchSize = 1

	conv := nn.LayerConfig{
		Type:        nn.LayerConv2D,
		InputHeight: 30, InputWidth: 30, InputChannels: 1,
		Filters: 4, KernelSize: 3, Stride: 1, Padding: 1,
		OutputHeight: 30, OutputWidth: 30,
		Activation: nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 4*1*3*3)
	conv.Bias = make([]float32, 4)
	initRandomSlice(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(3600, 64, nn.ActivationLeakyReLU))
	lstm := nn.InitLSTMLayer(8, 8, 1, 64)
	net.SetLayer(0, 0, 2, lstm)
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(64, InputSize, nn.ActivationSigmoid))
	return net
}

func createAttnDenseNet() *nn.Network {
	// Dense -> Attention -> Dense -> Output
	dModel := 64
	net := nn.NewNetwork(InputSize, 1, 1, 4)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, dModel, nn.ActivationLeakyReLU))

	mha := nn.LayerConfig{
		Type:     nn.LayerMultiHeadAttention,
		DModel:   dModel,
		NumHeads: 4,
	}
	headDim := dModel / 4
	mha.QWeights = make([]float32, dModel*dModel)
	mha.KWeights = make([]float32, dModel*dModel)
	mha.VWeights = make([]float32, dModel*dModel)
	mha.OutputWeight = make([]float32, dModel*dModel)
	mha.QBias = make([]float32, dModel)
	mha.KBias = make([]float32, dModel)
	mha.VBias = make([]float32, dModel)
	mha.OutputBias = make([]float32, dModel)
	initRandomSlice(mha.QWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandomSlice(mha.KWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandomSlice(mha.VWeights, 0.1/float32(math.Sqrt(float64(headDim))))
	initRandomSlice(mha.OutputWeight, 0.1/float32(math.Sqrt(float64(dModel))))
	net.SetLayer(0, 0, 1, mha)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(dModel, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(32, InputSize, nn.ActivationSigmoid))
	return net
}

func initRandomSlice(slice []float32, scale float32) {
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
				Input:  encodeGrid(pair.Input),
				Target: encodeGrid(pair.Output),
				Height: len(pair.Output),
				Width:  len(pair.Output[0]),
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
// Statistics & Output
// ============================================================================

func calcStats(values []float64) ModeStat {
	if len(values) == 0 {
		return ModeStat{}
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(values))

	minV, maxV := values[0], values[0]
	for _, v := range values {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}

	return ModeStat{Mean: mean, StdDev: math.Sqrt(variance), Min: minV, Max: maxV}
}

func printComparisonTable(results map[ArchType]map[TrainingMode][]float64) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ARC-AGI ARCHITECTURE COMPARISON (Best Accuracy %)                                          ║")
	fmt.Println("╠═══════════════════╦═════════════════════════╦═════════════════════════╦═════════════════════════╦════════════════════════════╣")
	fmt.Println("║ Architecture      ║ StepBP                  ║ TweenChain              ║ StepTweenChain          ║ Best                       ║")
	fmt.Println("╠═══════════════════╬═════════════════════════╬═════════════════════════╬═════════════════════════╬════════════════════════════╣")

	modes := []TrainingMode{ModeStepBP, ModeTweenChain, ModeStepTweenChain}

	for _, arch := range allArchitectures {
		bestMode := ""
		bestMean := 0.0

		stats := make([]ModeStat, 3)
		for i, mode := range modes {
			if vals, ok := results[arch][mode]; ok {
				stats[i] = calcStats(vals)
				if stats[i].Mean > bestMean {
					bestMean = stats[i].Mean
					bestMode = modeNames[mode]
				}
			}
		}

		fmt.Printf("║ %-17s ║ %5.1f%% (±%4.1f%%)         ║ %5.1f%% (±%4.1f%%)         ║ %5.1f%% (±%4.1f%%)         ║ %-26s ║\n",
			arch,
			stats[0].Mean, stats[0].StdDev,
			stats[1].Mean, stats[1].StdDev,
			stats[2].Mean, stats[2].StdDev,
			bestMode)
	}

	fmt.Println("╚═══════════════════╩═════════════════════════╩═════════════════════════╩═════════════════════════╩════════════════════════════╝")

	// Find overall best
	var bestArch ArchType
	var bestMode TrainingMode
	bestAcc := 0.0
	for arch, modes := range results {
		for mode, vals := range modes {
			stat := calcStats(vals)
			if stat.Mean > bestAcc {
				bestAcc = stat.Mean
				bestArch = arch
				bestMode = mode
			}
		}
	}

	fmt.Printf("\n★ Best Overall: %s + %s = %.1f%% accuracy\n", bestArch, modeNames[bestMode], bestAcc)
}

func saveResults(results map[ArchType]map[TrainingMode][]float64) {
	summaries := make(map[string]ArchSummary)

	for arch, modes := range results {
		summary := ArchSummary{
			Name:  string(arch),
			Modes: make(map[string]ModeStat),
		}

		bestMean := 0.0
		for mode, vals := range modes {
			stat := calcStats(vals)
			summary.Modes[modeNames[mode]] = stat
			if stat.Mean > bestMean {
				bestMean = stat.Mean
				summary.BestMode = modeNames[mode]
				summary.BestAccMean = stat.Mean
			}
		}
		summaries[string(arch)] = summary
	}

	data, _ := json.MarshalIndent(summaries, "", "  ")
	os.WriteFile("test21_results.json", data, 0644)
	fmt.Println("✓ Results saved to test21_results.json")
}

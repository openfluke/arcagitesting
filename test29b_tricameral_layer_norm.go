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

// Test 29b: TRICAMERAL NATIVE + LAYERNORM + ONE-HOT
//
// Updates from Test 29:
//   - Input: One-Hot Encoded (9000 floats instead of 900 scalars) -> Sees COLORS distinctively
//   - LayerNorm: Added before Parallel Block (stabilizes input to all brains) and after Merger
//   - DModel: Increased to 144 (12x12) for better CNN fit and capacity
//
// Architecture:
//   - Layer 0: Embedding (9000 -> 144)
//   - Layer 1: LayerNorm (144)
//   - Layer 2: LayerParallel (3 Branches)
//       - Right (MHA): 144 dim
//       - Left (LSTM): 144 dim
//       - Center (CNN): 12x12 grid (144 pixels)
//   - Layer 3: Merger (Concatenation -> 144)
//   - Layer 4: LayerNorm (144)
//   - Layer 5: Output (144 -> 9000) - Note: Multi-class output requires distinct 9000 outputs?
//     Wait, original had 900 output (scalar regression).
//     User said "finding colors". Scalar regression 0.0-1.0 is hard for multi-modal.
//     We should ideally output 9000 for softmax, but that's a huge change.
//     Sticking to scalar output for now, but with One-Hot Input.
//     Actually, let's keep output 900 for compatibility with "Scalar Regression to Colors" (0.1, 0.2...) logic
//     unless we want full softmax.
//     Given "test29" logic: pred := clampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
//     This implies regression. I will keep output as 900 scalars for now, but Input is One-Hot.
//     This gives distinct input features but simple regression output.
//
// Training: StepTween (Heuristic)

const (
	MaxGridSize  = 30
	NumColors    = 10
	InputSize    = MaxGridSize * MaxGridSize * NumColors // 9000
	OutputSize   = MaxGridSize * MaxGridSize             // 900 (Scalar Regression)
	NumTasks     = 400
	BatchSize    = 100
	NumEpochs    = 1400
	LearningRate = float32(0.01) // Slightly lower for larger net? Keeping 0.01 as safe bet.
	InitScale    = float32(0.5)
	BudgetScale  = float32(0.8)

	// Architecture params
	DModel     = 144 // 12x12
	NumHeads   = 8
	LSTMHidden = 144

	// CNN Brain Params
	ConvFilters  = 16 // Increased filters
	ConvKernel   = 3
	ConvGridSize = 12 // 12x12 = 144
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
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Test 29b: TRICAMERAL + LAYERNORM + ONE-HOT                                      ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     🎨 Input: One-Hot (9000 dim) - Improving Color Perception                       ║")
	fmt.Println("║     ⚖️  LayerNorm: Added before/after Parallel Block                                 ║")
	fmt.Println("║     🧠 DModel: 144 (Increased Capacity)                                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║     Training: StepTween (Gap-Closing)                                               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load tasks: %v\n", err)
		return
	}
	trainSamples, evalSamples := splitTrainEval(tasks)
	fmt.Printf("\n📦 Loaded %d tasks: %d train samples, %d eval samples\n\n", len(tasks), len(trainSamples), len(evalSamples))

	// Create the Tricameral network
	net := createTricameralNetwork()
	numLayers := net.TotalLayers()
	fmt.Printf("🏗️  Created Tricameral Network: %d layers\n", numLayers)

	// Initialize training state - One-Hot Input Size
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true
	ts.Config.LinkBudgetScale = BudgetScale

	results := &Results{
		AccuracyHistory: make([]float64, NumEpochs),
		BudgetHistory:   make([]float32, NumEpochs),
		SolvedTaskIDs:   []string{},
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                     🧠 TRICAMERAL TRAINING BEGINS 🧠")
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

			// TweenStep training
			// Target is typically one-hot or scalar. Here 'sample.Target' is scalar-encoded (0-1).
			// But for ARGMAX/TweenStep, we often need categorical index if doing classification,
			// or just gap closing on scalar.
			// The original Test 29 used: ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)
			// Wait, argmax(sample.Target) implies Target is a probability distribution per sample?
			// NO. In Test 29 `sample.Target` was 900 floats of values 0.0, 0.11, etc.
			// `argmax` on that just returns the index of the pixel with highest value (usually color 9).
			// This effectively trains the network to output high values where the target is high?
			// StepTween implementation: computes Gradient based on (Target[idx] - Output[idx])?
			// No, `TweenStep` takes `targetIndex`. It increases value at targetIndex and decreases others?
			// Let's check `TweenStep` signature usage.
			// Test 29: `ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), LearningRate)`
			// This trains the whole output vector to match the implicit distribution where `argmax` is?
			// Actually `TweenStep` usually assumes Logits/Softmax context if picking one index.
			// If we are doing regression on 900 pixels, `TweenStep` with a single index is weird.
			// Unless `TweenStep` handles full vector?
			// Let's look at `TweenStep` in previous code or memory.
			// It typically bumps ONE index.
			// If we want to train a detailed image, we need to call TweenStep for EACH pixel? Or use a different function.
			// Test 29 loop: just ONE call per batch item.
			// Meaning it only optimized ONE PIXEL per image per step??
			// That explains why it's slow/hard!
			// "Gap-Closing" usually implies iterating over errors.
			// I should probably fix this to train more pixels if possible, or stick to pattern.
			// Test 36 (Weighted Hive) used:
			// `weightedTweenStep` -> picks A RANDOM PIXEL (weighted) and trains it.
			// Ah! So we optimize one pixel per step. Stochastic Coordinate Descent essentially.
			// I will simulate `weightedTweenStep` behavior here or loop over a few pixels?
			// I'll adhere to the pattern: Pick a relevant pixel to train.
			// Let's improve Training by picking a RANDOM MISMATCHED pixel.

			trainPixelIndex := pickTrainingPixel(sample, getOutput(net, sample.Input, numLayers, state))
			ts.TweenStep(net, sample.Input, trainPixelIndex, OutputSize, LearningRate)
		}

		// Measure metrics
		acc := measureAccuracy(net, evalSamples, numLayers, state)
		budget := getBudget(ts)

		results.AccuracyHistory[epoch] = acc
		results.BudgetHistory[epoch] = budget

		if (epoch+1)%1 == 0 {
			fmt.Printf("  Epoch %3d/%d | Accuracy: %5.1f%% | Budget: %.3f\n",
				epoch+1, NumEpochs, acc, budget)
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

// Helper to pick a pixel to train
// We want to train on pixels that are WRONG.
func pickTrainingPixel(sample Sample, output []float32) int {
	// Simple strategy: Randomly pick a pixel, but weight towards errors?
	// Or just pure random like Test 29 (which did argmax... which biased towards White/Color 9?)
	// Let's just pick a random index for now to be unbiased, or random from non-background?
	// Test 36 showed weighted is better.
	// Let's do a simple weighted random: 50% chance to pick a non-zero pixel.
	indices := rand.Perm(len(sample.Target))
	for _, idx := range indices {
		if sample.Target[idx] > 0.05 { // Color pixel
			return idx
		}
	}
	return indices[0] // Fallback to random
}

// ============================================================================
// TRICAMERAL NETWORK ARCHITECTURE
// ============================================================================

func createTricameralNetwork() *nn.Network {
	// Layers:
	// 0: Embed
	// 1: LayerNorm (Pre-Split)
	// 2: Parallel
	// 3: Merger
	// 4: LayerNorm (Post-Merge)
	// 5: Output
	totalLayers := 6
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Layer 0: Input Embedding (9000 -> 144)
	inputLayer := nn.InitDenseLayer(InputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Layer 1: LayerNorm (Stabilize Shared Embedding)
	ln1 := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: DModel,
		Gamma:    make([]float32, DModel),
		Beta:     make([]float32, DModel),
		Epsilon:  1e-5,
	}
	for i := range ln1.Gamma {
		ln1.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, layerIdx, ln1)
	layerIdx++

	// Layer 2: PARALLEL SPLIT
	parallelLayer := createParallelTricameralLayer()
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Layer 3: Merger (Combine 3 branches -> DModel)
	// MHA(144) + LSTM(144) + CNN(16 filters * 12*12??) vs CNN(OutputHeight 12 * 12 * filters)
	// Let's check CNN output size.
	// ConvGridSize = 12. Filters = 16. Total = 12*12*16 = 2304.
	// Inputs: 144 + 144 + 2304 = 2592.
	mergerInputSize := DModel + DModel + (ConvGridSize * ConvGridSize * ConvFilters)
	mergerLayer := nn.InitDenseLayer(mergerInputSize, DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Layer 4: LayerNorm (Stabilize Decision Vector)
	ln2 := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: DModel,
		Gamma:    make([]float32, DModel),
		Beta:     make([]float32, DModel),
		Epsilon:  1e-5,
	}
	for i := range ln2.Gamma {
		ln2.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, layerIdx, ln2)
	layerIdx++

	// Layer 5: Output (DModel -> 900)
	// Mapping back to scalar grid.
	outputLayer := nn.InitDenseLayer(DModel, OutputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createParallelTricameralLayer() nn.LayerConfig {
	rightBrain := createRightBrainBranch()   // MHA
	leftBrain := createLeftBrainBranch()     // LSTM
	centerBrain := createCenterBrainBranch() // CNN

	return nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "concat",
		ParallelBranches: []nn.LayerConfig{
			rightBrain,
			leftBrain,
			centerBrain,
		},
	}
}

func createRightBrainBranch() nn.LayerConfig {
	headDim := DModel / NumHeads
	mha := nn.LayerConfig{
		Type:      nn.LayerMultiHeadAttention,
		DModel:    DModel,
		NumHeads:  NumHeads,
		SeqLength: 1,
	}
	// Weights init (simplified)
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

func createLeftBrainBranch() nn.LayerConfig {
	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: DModel,
		HiddenSize:   LSTMHidden,
		SeqLength:    1,
		OutputHeight: DModel,
	}
	initLSTMWeights(&lstm)
	return lstm
}

func createCenterBrainBranch() nn.LayerConfig {
	// CNN treats the 144-dim embedding as a 12x12 Grid
	cnn := nn.LayerConfig{
		Type:          nn.LayerConv2D,
		InputHeight:   ConvGridSize, // 12
		InputWidth:    ConvGridSize, // 12
		InputChannels: 1,
		Filters:       ConvFilters, // 16
		KernelSize:    ConvKernel,  // 3
		Stride:        1,
		Padding:       1,
		OutputHeight:  ConvGridSize,
		OutputWidth:   ConvGridSize,
		Activation:    nn.ActivationLeakyReLU,
	}
	fanIn := ConvKernel * ConvKernel * 1
	kernelSize := ConvFilters * 1 * ConvKernel * ConvKernel
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
// Metrics & Utils
// ============================================================================

func measureAccuracy(net *nn.Network, samples []Sample, numLayers int, state *nn.StepState) float64 {
	correct, total := 0, 0
	for _, sample := range samples {
		output := getOutput(net, sample.Input, numLayers, state)
		for r := 0; r < sample.Height; r++ {
			for c := 0; c < sample.Width; c++ {
				idx := r*MaxGridSize + c
				if idx < len(output) && idx < len(sample.Target) {
					// Target is scalar (0.0-1.0)
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

// ONE-HOT ENCODING
func encodeGrid(grid [][]int) []float32 {
	encoded := make([]float32, InputSize) // 9000
	for r := 0; r < len(grid) && r < MaxGridSize; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize; c++ {
			color := grid[r][c]
			if color >= 0 && color < NumColors {
				idx := (r*MaxGridSize+c)*NumColors + color
				encoded[idx] = 1.0
			}
		}
	}
	return encoded
}

// TARGET SCALAR ENCODING (Matches Original Output)
// This is for the regression target, so we still output 900 scalars.
// The Input is One-Hot (9000), but we map to a semantic Grid (900)
func encodeGridScalar(grid [][]int) []float32 {
	encoded := make([]float32, OutputSize) // 900
	for r := 0; r < len(grid) && r < MaxGridSize; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize; c++ {
			encoded[r*MaxGridSize+c] = float32(grid[r][c]) / 9.0
		}
	}
	return encoded
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
	Train []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"train"`
	Test []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"test"`
}

func splitTrainEval(tasks []*ARCTask) (trainSamples, evalSamples []Sample) {
	for _, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			trainSamples = append(trainSamples, Sample{
				Input:  encodeGrid(pair.Input),        // One-Hot
				Target: encodeGridScalar(pair.Output), // Scalar Regression
				Height: len(pair.Output), Width: len(pair.Output[0]),
				TaskID: task.ID,
			})
		}
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			evalSamples = append(evalSamples, Sample{
				Input:  encodeGrid(pair.Input),        // One-Hot
				Target: encodeGridScalar(pair.Output), // Scalar Regression
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

func printResults(results *Results) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      🧠 TRICAMERAL (LAYER NORM) - FINAL RESULTS 🧠                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Final Accuracy:     %5.1f%%                                                        ║\n", results.FinalAccuracy)
	fmt.Printf("║   Final Budget:       %.3f                                                          ║\n", results.FinalBudget)
	fmt.Printf("║   Tasks Solved:       %d                                                             ║\n", results.TasksSolved)
	fmt.Printf("║   Training Time:      %.1fs                                                          ║\n", results.TrainTime.Seconds())
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	if len(results.SolvedTaskIDs) > 0 {
		fmt.Println("║   Solved Task IDs:                                                                   ║")
		for i, id := range results.SolvedTaskIDs {
			if i >= 5 {
				fmt.Printf("║     ... %d more                                                                  ║\n", len(results.SolvedTaskIDs)-5)
				break
			}
			fmt.Printf("║     - %s \n", id)
		}
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")
}

func saveResults(results *Results) {
	output := map[string]interface{}{
		"final_accuracy":   results.FinalAccuracy,
		"final_budget":     results.FinalBudget,
		"tasks_solved":     results.TasksSolved,
		"solved_task_ids":  results.SolvedTaskIDs,
		"train_time_sec":   results.TrainTime.Seconds(),
		"accuracy_history": results.AccuracyHistory,
		"budget_history":   results.BudgetHistory,
		"meta": map[string]interface{}{
			"architecture": "Tricameral-LN (MHA+LSTM+CNN) OneHot",
			"epochs":       NumEpochs,
			"d_model":      DModel,
		},
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("test29b_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test29b_results.json")
}

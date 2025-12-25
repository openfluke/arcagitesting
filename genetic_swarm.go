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

// ═══════════════════════════════════════════════════════════════════════════════
// EVOLUTIONARY SWARM - Genetic Lottery with Nano-Hives
// ═══════════════════════════════════════════════════════════════════════════════
//
// Spawns 100 network variants with randomized architectures:
//   - Different brain types (MHA, LSTM, RNN, Dense)
//   - Different sizes (DModel, NumHeads, HiddenSize)
//   - Different grid configurations (1x1, 2x2, 3x3)
//   - Different learning rates
//
// Each network runs for 10s on the same task stream, then we pick the winners!

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 400
	InitScale   = float32(0.5)
	BudgetScale = float32(0.8)

	// Swarm settings
	SwarmSize      = 100
	TestDuration   = 10 * time.Second
	WindowDuration = 100 * time.Millisecond

	// Only use StepTweenChain for the swarm (best performer)
	TrainMode = ModeStepTweenChain
)

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
)

// BrainType defines what kind of layer a brain uses
type BrainType int

const (
	BrainMHA BrainType = iota
	BrainLSTM
	BrainRNN
	BrainDense
)

var brainNames = []string{"MHA", "LSTM", "RNN", "Dense"}

// NanoHiveConfig defines a randomized architecture
type NanoHiveConfig struct {
	ID           int         `json:"id"`
	Name         string      `json:"name"`
	GridSize     int         `json:"gridSize"`     // 1, 2, or 3
	DModel       int         `json:"dModel"`       // 16, 32, 64
	NumHeads     int         `json:"numHeads"`     // 2, 4, 8
	HiddenSize   int         `json:"hiddenSize"`   // 16, 32, 64
	LearningRate float32     `json:"learningRate"` // 0.001 - 0.1
	Brains       []BrainType `json:"-"`
	BrainNames   []string    `json:"brains"`
}

// SwarmResult tracks per-network performance
type SwarmResult struct {
	Config           NanoHiveConfig `json:"config"`
	Windows          []TimeWindow   `json:"windows"`
	TotalOutputs     int            `json:"totalOutputs"`
	AvgTrainAccuracy float64        `json:"avgTrainAccuracy"`
	Stability        float64        `json:"stability"`
	Consistency      float64        `json:"consistency"`
	ThroughputPerSec float64        `json:"throughputPerSec"`
	Score            float64        `json:"score"`
	TasksSolved      int            `json:"tasksSolved"`
	EvalAccuracy     float64        `json:"evalAccuracy"`
	TrainTimeSec     float64        `json:"trainTimeSec"`
}

type TimeWindow struct {
	TimeMs        int     `json:"timeMs"`
	Outputs       int     `json:"outputs"`
	TotalPixelAcc float64 `json:"totalPixelAcc"`
	Accuracy      float64 `json:"accuracy"`
}

// SwarmBenchmarkResults is the output
type SwarmBenchmarkResults struct {
	SwarmSize  int            `json:"swarmSize"`
	TopN       []SwarmResult  `json:"topN"`       // Top 10 performers
	AllResults []SwarmResult  `json:"allResults"` // All 100 for analysis
	BestConfig NanoHiveConfig `json:"bestConfig"`
	Timestamp  string         `json:"timestamp"`
	Duration   string         `json:"duration"`
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
	TaskIndex     int
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     🧬 EVOLUTIONARY SWARM - Genetic Lottery with Nano-Hives                         ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║     Spawning 100 randomized network architectures...                                 ║")
	fmt.Println("║     Each runs 10s on 400 ARC tasks, then we find the WINNER!                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	// Load ARC-AGI training data
	trainTasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load training tasks: %v\n", err)
		return
	}

	// Load ARC-AGI evaluation data (separate 400 tasks)
	evalTasks, err := loadARCTasks("ARC-AGI/data/evaluation", 400)
	if err != nil {
		fmt.Printf("❌ Failed to load eval tasks: %v\n", err)
		return
	}

	trainSamples := createSequentialSamples(trainTasks)
	evalSamples := createEvalSamples(evalTasks)

	fmt.Printf("\n📦 Loaded %d training tasks, %d train samples\n", len(trainTasks), len(trainSamples))
	fmt.Printf("📦 Loaded %d eval tasks, %d eval samples\n", len(evalTasks), len(evalSamples))
	fmt.Printf("🧬 Generating %d random network configurations...\n\n", SwarmSize)

	// Generate random configurations
	configs := generateRandomConfigs(SwarmSize)

	// Run swarm in parallel (with concurrency limit)
	results := make([]SwarmResult, SwarmSize)
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 10) // Limit to 10 concurrent networks

	startTime := time.Now()

	for i, cfg := range configs {
		wg.Add(1)
		go func(idx int, config NanoHiveConfig) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			result := runSwarmMember(config, trainSamples, evalSamples, evalTasks)
			results[idx] = result

			if (idx+1)%10 == 0 {
				fmt.Printf("🔄 Progress: %d/%d networks evaluated\n", idx+1, SwarmSize)
			}
		}(i, cfg)
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Build output
	topN := 10
	if len(results) < topN {
		topN = len(results)
	}

	output := &SwarmBenchmarkResults{
		SwarmSize:  SwarmSize,
		TopN:       results[:topN],
		AllResults: results,
		BestConfig: results[0].Config,
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   totalTime.String(),
	}

	// Save results
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("genetic_swarm_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to genetic_swarm_results.json")

	// Print leaderboard
	printLeaderboard(results)
}

func generateRandomConfigs(count int) []NanoHiveConfig {
	configs := make([]NanoHiveConfig, count)

	gridSizes := []int{1, 2, 3}
	dModels := []int{16, 32, 64}
	numHeads := []int{2, 4, 8}
	hiddenSizes := []int{16, 32, 64}

	for i := 0; i < count; i++ {
		gridSize := gridSizes[rand.Intn(len(gridSizes))]
		numBrains := gridSize * gridSize

		brains := make([]BrainType, numBrains)
		brainNamesList := make([]string, numBrains)
		for b := 0; b < numBrains; b++ {
			brainType := BrainType(rand.Intn(4))
			brains[b] = brainType
			brainNamesList[b] = brainNames[brainType]
		}

		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		// Ensure heads divides dModel
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		configs[i] = NanoHiveConfig{
			ID:           i,
			Name:         fmt.Sprintf("Nano-%d", i),
			GridSize:     gridSize,
			DModel:       dModel,
			NumHeads:     heads,
			HiddenSize:   hiddenSizes[rand.Intn(len(hiddenSizes))],
			LearningRate: float32(0.001 + rand.Float64()*0.099), // 0.001 to 0.1
			Brains:       brains,
			BrainNames:   brainNamesList,
		}
	}

	return configs
}

func runSwarmMember(config NanoHiveConfig, trainSamples, evalSamples []Sample, evalTasks []*ARCTask) SwarmResult {
	numWindows := int(TestDuration / WindowDuration)
	result := SwarmResult{
		Config:  config,
		Windows: make([]TimeWindow, numWindows),
	}

	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	// Create network with this config
	net := createNanoHiveNetwork(config)
	numLayers := net.TotalLayers()

	// Use StepTweenChain (best mode)
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = BudgetScale
	ts.Config.UseChainRule = true

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0

	// Training loop
	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		sample := trainSamples[sampleIdx%len(trainSamples)]
		sampleIdx++

		// Forward
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()

		// Calculate accuracy
		pixelAcc := calculatePixelAccuracy(output, sample)

		if currentWindow < numWindows {
			result.Windows[currentWindow].Outputs++
			result.Windows[currentWindow].TotalPixelAcc += pixelAcc
			result.TotalOutputs++
		}

		// Train with StepTweenChain
		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), config.LearningRate)
	}

	// Finalize windows
	for i := range result.Windows {
		if result.Windows[i].Outputs > 0 {
			result.Windows[i].Accuracy = result.Windows[i].TotalPixelAcc / float64(result.Windows[i].Outputs)
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()

	// =========================================================================
	// EVAL PHASE: Few-Shot Adaptation (Learn from examples -> Solve test)
	// =========================================================================
	taskResults := make(map[string]struct {
		totalAcc float64
		count    int
	})
	evalTotal := 0.0
	evalCount := 0

	// Iterate through TASKS for proper few-shot adaptation
	for _, task := range evalTasks {
		// 1. ADAPTATION PHASE: Learn from the task's example pairs
		for k := 0; k < 5; k++ {
			for _, pair := range task.Train {
				if len(pair.Input) == 0 || len(pair.Output) == 0 {
					continue
				}
				input := encodeGrid(pair.Input)
				target := encodeGrid(pair.Output)
				ts.TweenStep(net, input, argmax(target), len(target), config.LearningRate)
			}
		}

		// 2. TESTING PHASE: Solve the test pair(s)
		for _, pair := range task.Test {
			if len(pair.Input) == 0 || len(pair.Output) == 0 {
				continue
			}

			input := encodeGrid(pair.Input)
			target := encodeGrid(pair.Output)

			state.SetInput(input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output := state.GetOutput()

			acc := calculatePixelAccuracy(output, Sample{
				Target: target,
				Height: len(pair.Output),
				Width:  len(pair.Output[0]),
			})
			evalTotal += acc
			evalCount++

			r := taskResults[task.ID]
			r.totalAcc += acc
			r.count++
			taskResults[task.ID] = r
		}
	}

	if evalCount > 0 {
		result.EvalAccuracy = evalTotal / float64(evalCount)
	}

	// Count solved tasks (100% accuracy required)
	for _, r := range taskResults {
		if r.count > 0 && r.totalAcc/float64(r.count) >= 100 {
			result.TasksSolved++
		}
	}

	// Calculate metrics
	calculateSwarmMetrics(&result)

	return result
}

func createNanoHiveNetwork(config NanoHiveConfig) *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Input layer
	inputLayer := nn.InitDenseLayer(InputSize, config.DModel, nn.ActivationLeakyReLU)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Parallel hive layer
	parallelLayer := createNanoHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Merger layer
	mergerInputSize := config.DModel * config.GridSize * config.GridSize
	mergerLayer := nn.InitDenseLayer(mergerInputSize, config.DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Output layer
	outputLayer := nn.InitDenseLayer(config.DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createNanoHive(config NanoHiveConfig) nn.LayerConfig {
	numBrains := config.GridSize * config.GridSize
	branches := make([]nn.LayerConfig, numBrains)
	positions := make([]nn.GridPosition, numBrains)

	for i := 0; i < numBrains; i++ {
		brainType := config.Brains[i]
		switch brainType {
		case BrainMHA:
			branches[i] = createMHABrain(config.DModel, config.NumHeads)
		case BrainLSTM:
			branches[i] = createLSTMBrain(config.DModel)
		case BrainRNN:
			branches[i] = createRNNBrain(config.DModel)
		case BrainDense:
			branches[i] = createDenseBrain(config.DModel)
		}

		row := i / config.GridSize
		col := i % config.GridSize
		positions[i] = nn.GridPosition{
			BranchIndex: i,
			TargetRow:   row,
			TargetCol:   col,
			TargetLayer: 0,
		}
	}

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   config.GridSize,
		GridOutputCols:   config.GridSize,
		GridOutputLayers: 1,
		ParallelBranches: branches,
		GridPositions:    positions,
	}

	return parallel
}

func createMHABrain(dModel, numHeads int) nn.LayerConfig {
	headDim := dModel / numHeads
	mha := nn.LayerConfig{
		Type:      nn.LayerMultiHeadAttention,
		DModel:    dModel,
		NumHeads:  numHeads,
		SeqLength: 1,
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

func createLSTMBrain(dModel int) nn.LayerConfig {
	// Use dModel as hidden size to ensure output matches expected dimension
	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: dModel,
		HiddenSize:   dModel,
		SeqLength:    1,
		OutputHeight: dModel,
	}
	initLSTMWeights(&lstm)
	return lstm
}

func createRNNBrain(dModel int) nn.LayerConfig {
	// Use dModel as hidden size to ensure output matches expected dimension
	rnn := nn.LayerConfig{
		Type:         nn.LayerRNN,
		RNNInputSize: dModel,
		HiddenSize:   dModel,
		SeqLength:    1,
		OutputHeight: dModel,
	}
	initRNNWeights(&rnn)
	return rnn
}

func createDenseBrain(dModel int) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights(dense.Kernel, InitScale)
	return dense
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

func initRNNWeights(cfg *nn.LayerConfig) {
	inputSize := cfg.RNNInputSize
	hiddenSize := cfg.HiddenSize

	cfg.WeightIH = make([]float32, hiddenSize*inputSize)
	cfg.WeightHH = make([]float32, hiddenSize*hiddenSize)
	cfg.BiasH = make([]float32, hiddenSize)

	scale := InitScale / float32(math.Sqrt(float64(hiddenSize)))
	initRandom(cfg.WeightIH, scale)
	initRandom(cfg.WeightHH, scale)
}

func calculateSwarmMetrics(result *SwarmResult) {
	// Average training accuracy
	sum := 0.0
	for _, w := range result.Windows {
		sum += w.Accuracy
	}
	result.AvgTrainAccuracy = sum / float64(len(result.Windows))

	// Stability: 100 - stddev
	variance := 0.0
	for _, w := range result.Windows {
		diff := w.Accuracy - result.AvgTrainAccuracy
		variance += diff * diff
	}
	variance /= float64(len(result.Windows))
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	// Consistency: % windows above 12% (random baseline)
	const consistencyThreshold = 12.0
	aboveThreshold := 0
	for _, w := range result.Windows {
		if w.Accuracy >= consistencyThreshold {
			aboveThreshold++
		}
	}
	result.Consistency = float64(aboveThreshold) / float64(len(result.Windows)) * 100

	// Throughput
	result.ThroughputPerSec = float64(result.TotalOutputs) / result.TrainTimeSec

	// Score
	result.Score = (result.ThroughputPerSec * result.Stability * result.Consistency) / 100000
}

func calculatePixelAccuracy(output []float32, sample Sample) float64 {
	correct, total := 0, 0
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
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}

// Utilities
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

// Data loading
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

func encodeGrid(grid [][]int) []float32 {
	encoded := make([]float32, InputSize)
	for r := 0; r < len(grid) && r < MaxGridSize; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize; c++ {
			encoded[r*MaxGridSize+c] = float32(grid[r][c]) / 9.0
		}
	}
	return encoded
}

func createSequentialSamples(tasks []*ARCTask) []Sample {
	var samples []Sample
	for i, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample{
				Input:     encodeGrid(pair.Input),
				Target:    encodeGrid(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

func createEvalSamples(tasks []*ARCTask) []Sample {
	var samples []Sample
	for i, task := range tasks {
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample{
				Input:     encodeGrid(pair.Input),
				Target:    encodeGrid(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

func printLeaderboard(results []SwarmResult) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                            🏆 EVOLUTIONARY SWARM LEADERBOARD - Top 10                                                   ║")
	fmt.Println("╠═════╦══════════════════════════════════════════════════════════════════════╦══════════╦══════════╦════════╦══════════════╣")
	fmt.Println("║ Rank║ Architecture                                                         ║ Accuracy ║ Solved   ║ Score  ║ LR           ║")
	fmt.Println("╠═════╬══════════════════════════════════════════════════════════════════════╬══════════╬══════════╬════════╬══════════════╣")

	for i := 0; i < 10 && i < len(results); i++ {
		r := results[i]
		cfg := r.Config
		archDesc := fmt.Sprintf("%dx%d Grid | D=%d H=%d | %v",
			cfg.GridSize, cfg.GridSize, cfg.DModel, cfg.NumHeads, cfg.BrainNames)
		if len(archDesc) > 68 {
			archDesc = archDesc[:65] + "..."
		}
		fmt.Printf("║ %3d ║ %-68s ║ %6.1f%% ║ %8d ║ %6.0f ║ %.4f       ║\n",
			i+1, archDesc, r.AvgTrainAccuracy, r.TasksSolved, r.Score, cfg.LearningRate)
	}

	fmt.Println("╚═════╩══════════════════════════════════════════════════════════════════════╩══════════╩══════════╩════════╩══════════════╝")

	// Winner details
	if len(results) > 0 {
		winner := results[0]
		fmt.Println("\n🏆 WINNER ARCHITECTURE:")
		fmt.Printf("   Grid: %dx%d | DModel: %d | NumHeads: %d | HiddenSize: %d\n",
			winner.Config.GridSize, winner.Config.GridSize,
			winner.Config.DModel, winner.Config.NumHeads, winner.Config.HiddenSize)
		fmt.Printf("   Brains: %v\n", winner.Config.BrainNames)
		fmt.Printf("   Learning Rate: %.4f\n", winner.Config.LearningRate)
		fmt.Printf("   Score: %.0f | Accuracy: %.1f%% | Tasks Solved: %d\n",
			winner.Score, winner.AvgTrainAccuracy, winner.TasksSolved)
	}
}

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE EVOLUTIONARY ZOO - Deep Architectural Mutations for Task Discovery
// ═══════════════════════════════════════════════════════════════════════════════
//
// Test 38 proved that same-architecture clones saturate at ~11 tasks.
// The Zoo introduces SPECIATION - wildly different network topologies.
//
// Mutations include:
// - Grid shapes: 1x1, 2x2, 3x3, 4x1, 1x4, 2x3, 3x2
// - Brain types: MHA, LSTM, RNN, Dense (Conv2D needs image input)
// - Activations: ReLU, LeakyReLU, SiLU, Tanh
// - DModel: 16, 32, 64
// - Learning Rate: Log-uniform 0.0001 to 0.1

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 400
	InitScale   = float32(0.5)

	// Zoo settings
	ZooSize        = 2500
	TestDuration   = 10 * time.Second
	WindowDuration = 100 * time.Millisecond
)

// BrainType defines what kind of layer a brain uses
type BrainType int

const (
	BrainMHA BrainType = iota
	BrainLSTM
	BrainRNN
	BrainDense
)

var brainTypeNames = []string{"MHA", "LSTM", "RNN", "Dense"}

// ActivationType for layer activations
type ActivationType int

const (
	ActReLU ActivationType = iota
	ActLeakyReLU
	ActSiLU
	ActTanh
)

var activationNames = []string{"ScaledReLU", "LeakyReLU", "Tanh", "Softplus"}

// getActivation returns the nn activation constant for the given type
func getActivation(act ActivationType) nn.ActivationType {
	switch act {
	case ActReLU:
		return nn.ActivationScaledReLU
	case ActLeakyReLU:
		return nn.ActivationLeakyReLU
	case ActSiLU:
		return nn.ActivationSoftplus // SiLU not available, fallback to Softplus
	case ActTanh:
		return nn.ActivationTanh
	default:
		return nn.ActivationLeakyReLU
	}
}

// GridShape for different topologies
type GridShape struct {
	Rows int
	Cols int
	Name string
}

var gridShapes = []GridShape{
	{1, 1, "1x1 Monolith"},
	{2, 2, "2x2 Standard"},
	{3, 3, "3x3 Complex"},
	{4, 1, "4x1 Tall"},
	{1, 4, "1x4 Wide"},
	{2, 3, "2x3 Rectangle"},
	{3, 2, "3x2 Rectangle"},
}

// MutantConfig defines a randomized architecture
type MutantConfig struct {
	ID           int            `json:"id"`
	Name         string         `json:"name"`
	Species      string         `json:"species"`     // e.g., "2x2 Standard"
	MutationStr  string         `json:"mutationStr"` // e.g., "2x2_MHA-LSTM_LeakyReLU_D32_LR0.05"
	GridRows     int            `json:"gridRows"`
	GridCols     int            `json:"gridCols"`
	DModel       int            `json:"dModel"`
	NumHeads     int            `json:"numHeads"`
	LearningRate float32        `json:"learningRate"`
	BudgetScale  float32        `json:"budgetScale"`
	Activation   ActivationType `json:"-"`
	ActivationN  string         `json:"activation"`
	Brains       []BrainType    `json:"-"`
	BrainNames   []string       `json:"brains"`
}

// MutantResult tracks per-mutant performance
type MutantResult struct {
	Config           MutantConfig `json:"config"`
	AvgTrainAccuracy float64      `json:"avgTrainAccuracy"`
	TasksSolved      int          `json:"tasksSolved"`
	SolvedTaskIDs    []string     `json:"solvedTaskIds"`
	Score            float64      `json:"score"`
	ThroughputPerSec float64      `json:"throughputPerSec"`
}

// SpeciesStats tracks performance by species
type SpeciesStats struct {
	Species     string   `json:"species"`
	Count       int      `json:"count"`
	TotalSolved int      `json:"totalSolved"`
	UniqueTasks []string `json:"uniqueTasks"`
	UniqueCount int      `json:"uniqueCount"`
	BestMutant  string   `json:"bestMutant"`
	BestSolved  int      `json:"bestSolved"`
	AvgAccuracy float64  `json:"avgAccuracy"`
}

// DiscoveryPoint tracks cumulative unique tasks
type DiscoveryPoint struct {
	MutantNum        int `json:"mutantNum"`
	CumulativeUnique int `json:"cumulativeUnique"`
}

// ZooResults is the output
type ZooResults struct {
	ZooSize           int              `json:"zooSize"`
	TopMutants        []MutantResult   `json:"topMutants"`
	SpeciesBreakdown  []SpeciesStats   `json:"speciesBreakdown"`
	CollectiveTasks   []string         `json:"collectiveTasks"`
	CollectiveCount   int              `json:"collectiveCount"`
	DiscoveryTimeline []DiscoveryPoint `json:"discoveryTimeline"`
	Timestamp         string           `json:"timestamp"`
	Duration          string           `json:"duration"`
	WorkerCount       int              `json:"workerCount"`
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

type TimeWindow struct {
	TimeMs        int
	Outputs       int
	TotalPixelAcc float64
	Accuracy      float64
}

func main() {
	rand.Seed(time.Now().UnixNano())
	numWorkers := runtime.NumCPU()

	fmt.Println("╔════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🦎 THE EVOLUTIONARY ZOO - Deep Architectural Mutations                                       ║")
	fmt.Println("║                                                                                                ║")
	fmt.Printf("║   Spawning %d mutant architectures with %d parallel workers...                              ║\n", ZooSize, numWorkers)
	fmt.Println("║   Testing SPECIATION: Can different topologies break the 11-task ceiling?                     ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════╝")

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
	fmt.Printf("🦎 Generating %d mutant configurations with deep architectural diversity...\n\n", ZooSize)

	// Generate mutant configurations
	configs := generateMutantConfigs(ZooSize)

	// Track results
	var (
		results             = make([]MutantResult, ZooSize)
		collectiveTasksMu   sync.Mutex
		collectiveTasks     = make(map[string]bool)
		speciesTasks        = make(map[string]map[string]bool)
		speciesResults      = make(map[string][]MutantResult)
		discoveryTimeline   []DiscoveryPoint
		processedCount      int
		lastReportedPercent int
	)

	// Initialize species maps
	for _, shape := range gridShapes {
		speciesTasks[shape.Name] = make(map[string]bool)
		speciesResults[shape.Name] = []MutantResult{}
	}

	// Worker pool
	startTime := time.Now()
	jobs := make(chan int, ZooSize)
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				result := runMutant(configs[idx], trainSamples, evalSamples, evalTasks)
				results[idx] = result

				// Update tracking (thread-safe)
				collectiveTasksMu.Lock()
				for _, taskID := range result.SolvedTaskIDs {
					collectiveTasks[taskID] = true
					speciesTasks[result.Config.Species][taskID] = true
				}
				speciesResults[result.Config.Species] = append(speciesResults[result.Config.Species], result)
				processedCount++

				// Record discovery point every 25 mutants
				if processedCount%25 == 0 {
					discoveryTimeline = append(discoveryTimeline, DiscoveryPoint{
						MutantNum:        processedCount,
						CumulativeUnique: len(collectiveTasks),
					})
				}

				// Progress report
				percent := processedCount * 100 / ZooSize
				if percent >= lastReportedPercent+5 {
					lastReportedPercent = percent
					fmt.Printf("🔄 Progress: %d/%d (%d%%) | Unique tasks: %d | Species discovering...\n",
						processedCount, ZooSize, percent, len(collectiveTasks))
				}
				collectiveTasksMu.Unlock()
			}
		}()
	}

	// Send jobs
	for i := 0; i < ZooSize; i++ {
		jobs <- i
	}
	close(jobs)

	wg.Wait()
	totalTime := time.Since(startTime)

	// Sort by tasks solved
	sort.Slice(results, func(i, j int) bool {
		if results[i].TasksSolved != results[j].TasksSolved {
			return results[i].TasksSolved > results[j].TasksSolved
		}
		return results[i].Score > results[j].Score
	})

	// Build species stats
	var speciesBreakdown []SpeciesStats
	for _, shape := range gridShapes {
		species := shape.Name
		speciesRes := speciesResults[species]
		uniqueTasks := make([]string, 0, len(speciesTasks[species]))
		for t := range speciesTasks[species] {
			uniqueTasks = append(uniqueTasks, t)
		}
		sort.Strings(uniqueTasks)

		// Find best mutant in species
		bestMutant := ""
		bestSolved := 0
		totalAcc := 0.0
		totalSolved := 0
		for _, r := range speciesRes {
			totalAcc += r.AvgTrainAccuracy
			totalSolved += r.TasksSolved
			if r.TasksSolved > bestSolved {
				bestSolved = r.TasksSolved
				bestMutant = r.Config.MutationStr
			}
		}

		avgAcc := 0.0
		if len(speciesRes) > 0 {
			avgAcc = totalAcc / float64(len(speciesRes))
		}

		speciesBreakdown = append(speciesBreakdown, SpeciesStats{
			Species:     species,
			Count:       len(speciesRes),
			TotalSolved: totalSolved,
			UniqueTasks: uniqueTasks,
			UniqueCount: len(uniqueTasks),
			BestMutant:  bestMutant,
			BestSolved:  bestSolved,
			AvgAccuracy: avgAcc,
		})
	}

	// Sort species by unique count
	sort.Slice(speciesBreakdown, func(i, j int) bool {
		return speciesBreakdown[i].UniqueCount > speciesBreakdown[j].UniqueCount
	})

	// Build collective tasks list
	var collectiveTasksList []string
	for taskID := range collectiveTasks {
		collectiveTasksList = append(collectiveTasksList, taskID)
	}
	sort.Strings(collectiveTasksList)

	// Top 20 mutants
	topN := 20
	if len(results) < topN {
		topN = len(results)
	}

	// Build output
	output := &ZooResults{
		ZooSize:           ZooSize,
		TopMutants:        results[:topN],
		SpeciesBreakdown:  speciesBreakdown,
		CollectiveTasks:   collectiveTasksList,
		CollectiveCount:   len(collectiveTasksList),
		DiscoveryTimeline: discoveryTimeline,
		Timestamp:         time.Now().Format(time.RFC3339),
		Duration:          totalTime.String(),
		WorkerCount:       numWorkers,
	}

	// Save results
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("zoo_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to zoo_results.json")

	// Print results
	printZooResults(output)
}

func generateMutantConfigs(count int) []MutantConfig {
	configs := make([]MutantConfig, count)

	dModels := []int{16, 32, 64}
	numHeads := []int{2, 4, 8}

	for i := 0; i < count; i++ {
		// Random grid shape
		shape := gridShapes[rand.Intn(len(gridShapes))]
		numBrains := shape.Rows * shape.Cols

		// Random brains
		brains := make([]BrainType, numBrains)
		brainNames := make([]string, numBrains)
		for b := 0; b < numBrains; b++ {
			brainType := BrainType(rand.Intn(4))
			brains[b] = brainType
			brainNames[b] = brainTypeNames[brainType]
		}

		// Random DModel and heads
		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		// Log-uniform learning rate (0.0001 to 0.1)
		logMin := math.Log(0.0001)
		logMax := math.Log(0.1)
		lr := float32(math.Exp(logMin + rand.Float64()*(logMax-logMin)))

		// Random activation
		activation := ActivationType(rand.Intn(4))

		// Build mutation string
		brainStr := strings.Join(brainNames, "-")
		mutationStr := fmt.Sprintf("%dx%d_%s_%s_D%d_LR%.4f",
			shape.Rows, shape.Cols, brainStr, activationNames[activation], dModel, lr)

		configs[i] = MutantConfig{
			ID:           i,
			Name:         fmt.Sprintf("Mutant-%d", i),
			Species:      shape.Name,
			MutationStr:  mutationStr,
			GridRows:     shape.Rows,
			GridCols:     shape.Cols,
			DModel:       dModel,
			NumHeads:     heads,
			LearningRate: lr,
			BudgetScale:  float32(0.5 + rand.Float64()*0.5),
			Activation:   activation,
			ActivationN:  activationNames[activation],
			Brains:       brains,
			BrainNames:   brainNames,
		}
	}

	return configs
}

func runMutant(config MutantConfig, trainSamples, evalSamples []Sample, evalTasks []*ARCTask) MutantResult {
	numWindows := int(TestDuration / WindowDuration)
	windows := make([]TimeWindow, numWindows)

	for i := range windows {
		windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	// Create network
	net := createMutantNetwork(config)
	numLayers := net.TotalLayers()

	// Use StepTweenChain
	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = config.BudgetScale
	ts.Config.UseChainRule = true

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	totalOutputs := 0

	// Training loop
	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		sample := trainSamples[sampleIdx%len(trainSamples)]
		sampleIdx++

		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()

		pixelAcc := calculatePixelAccuracy(output, sample)

		if currentWindow < numWindows {
			windows[currentWindow].Outputs++
			windows[currentWindow].TotalPixelAcc += pixelAcc
			totalOutputs++
		}

		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), config.LearningRate)
	}

	trainTime := time.Since(start).Seconds()

	// Finalize windows
	for i := range windows {
		if windows[i].Outputs > 0 {
			windows[i].Accuracy = windows[i].TotalPixelAcc / float64(windows[i].Outputs)
		}
	}

	// Calculate average accuracy
	sum := 0.0
	for _, w := range windows {
		sum += w.Accuracy
	}
	avgAcc := sum / float64(len(windows))

	// =========================================================================
	// EVAL PHASE: Few-Shot Adaptation (Learn from examples -> Solve test)
	// =========================================================================
	taskResults := make(map[string]struct {
		totalAcc float64
		count    int
	})

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

			r := taskResults[task.ID]
			r.totalAcc += acc
			r.count++
			taskResults[task.ID] = r
		}
	}

	// Find solved tasks (100% accuracy required)
	var solvedIDs []string
	for taskID, r := range taskResults {
		if r.count > 0 && r.totalAcc/float64(r.count) >= 100 {
			solvedIDs = append(solvedIDs, taskID)
		}
	}

	return MutantResult{
		Config:           config,
		AvgTrainAccuracy: avgAcc,
		TasksSolved:      len(solvedIDs),
		SolvedTaskIDs:    solvedIDs,
		ThroughputPerSec: float64(totalOutputs) / trainTime,
		Score:            avgAcc * float64(len(solvedIDs)+1),
	}
}

func createMutantNetwork(config MutantConfig) *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	activation := getActivation(config.Activation)
	layerIdx := 0

	// Input layer
	inputLayer := nn.InitDenseLayer(InputSize, config.DModel, activation)
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Parallel hive layer
	parallelLayer := createMutantHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Merger layer
	mergerInputSize := config.DModel * config.GridRows * config.GridCols
	mergerLayer := nn.InitDenseLayer(mergerInputSize, config.DModel, activation)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Output layer
	outputLayer := nn.InitDenseLayer(config.DModel, InputSize, nn.ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createMutantHive(config MutantConfig) nn.LayerConfig {
	numBrains := config.GridRows * config.GridCols
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
			branches[i] = createDenseBrain(config.DModel, config.Activation)
		}

		row := i / config.GridCols
		col := i % config.GridCols
		positions[i] = nn.GridPosition{
			BranchIndex: i,
			TargetRow:   row,
			TargetCol:   col,
			TargetLayer: 0,
		}
	}

	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   config.GridRows,
		GridOutputCols:   config.GridCols,
		GridOutputLayers: 1,
		ParallelBranches: branches,
		GridPositions:    positions,
	}
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

func createDenseBrain(dModel int, activation ActivationType) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, getActivation(activation))
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

func printZooResults(output *ZooResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    🦎 THE EVOLUTIONARY ZOO - RESULTS                                                                 ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Zoo Size: %d | Unique Tasks Solved: %d | Duration: %s                                                                     ║\n",
		output.ZooSize, output.CollectiveCount, output.Duration)
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                         📊 SPECIES BREAKDOWN (Phylogenetic Tree)                                                     ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, s := range output.SpeciesBreakdown {
		bar := ""
		for j := 0; j < s.UniqueCount; j++ {
			bar += "█"
		}
		fmt.Printf("║ %-15s: %s %d unique (from %d mutants, best: %d solved)\n",
			s.Species, bar, s.UniqueCount, s.Count, s.BestSolved)
	}

	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                              🏆 HALL OF FAME - TOP 10 MUTANTS                                                        ║")
	fmt.Println("╠═════════════════╦═════════════════════════════════════════════════════════════════════════════════════════════════════════╦════════════╣")

	for i := 0; i < 10 && i < len(output.TopMutants); i++ {
		r := output.TopMutants[i]
		mutStr := r.Config.MutationStr
		if len(mutStr) > 85 {
			mutStr = mutStr[:82] + "..."
		}
		fmt.Printf("║ %-15s ║ %-91s ║ %3d tasks  ║\n",
			r.Config.Name, mutStr, r.TasksSolved)
	}

	fmt.Println("╚═════════════════╩═════════════════════════════════════════════════════════════════════════════════════════════════════════╩════════════╝")

	// Discovery curve
	fmt.Println("\n📈 DISCOVERY CURVE:")
	for _, dp := range output.DiscoveryTimeline {
		bar := ""
		for j := 0; j < dp.CumulativeUnique; j++ {
			bar += "█"
		}
		if dp.MutantNum%100 == 0 || dp.MutantNum == ZooSize {
			fmt.Printf("   Mutant %5d: %s (%d tasks)\n", dp.MutantNum, bar, dp.CumulativeUnique)
		}
	}

	fmt.Printf("\n🧬 COLLECTIVE WISDOM: %d unique tasks solved by the Zoo\n", output.CollectiveCount)
	fmt.Printf("   Tasks: %v\n", output.CollectiveTasks)
}

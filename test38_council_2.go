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
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE COUNCIL OF 1000 - Massive Scale Architecture Search
// ═══════════════════════════════════════════════════════════════════════════════
//
// Goal: Test statistical saturation - how many unique tasks can 1000 agents solve?
//
// Key insight: If the "Discovery Chart" flattens, we've hit the architecture ceiling.
// If it keeps rising, we should run 10,000!

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 1000                      // All ARC-AGI2 training tasks
	InitScale   = float32(0.5)

	// Council settings
	CouncilSize    = 1000
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

// AgentConfig defines a randomized architecture
type AgentConfig struct {
	ID           int         `json:"id"`
	Name         string      `json:"name"`
	GridSize     int         `json:"gridSize"`
	DModel       int         `json:"dModel"`
	NumHeads     int         `json:"numHeads"`
	LearningRate float32     `json:"learningRate"`
	BudgetScale  float32     `json:"budgetScale"`
	Brains       []BrainType `json:"-"`
	BrainNames   []string    `json:"brains"`
}

// AgentResult tracks per-agent performance
type AgentResult struct {
	Config           AgentConfig `json:"config"`
	AvgTrainAccuracy float64     `json:"avgTrainAccuracy"`
	TasksSolved      int         `json:"tasksSolved"`
	SolvedTaskIDs    []string    `json:"solvedTaskIds"`
	Score            float64     `json:"score"`
	ThroughputPerSec float64     `json:"throughputPerSec"`
}

// DiscoveryPoint tracks cumulative unique tasks over time
type DiscoveryPoint struct {
	AgentNum          int `json:"agentNum"`
	CumulativeUnique  int `json:"cumulativeUnique"`
	NewTasksThisAgent int `json:"newTasksThisAgent"`
}

// CouncilResults is the output
type CouncilResults struct {
	CouncilSize       int              `json:"councilSize"`
	TopExperts        []AgentResult    `json:"topExperts"`
	CollectiveTasks   []string         `json:"collectiveTasks"`
	CollectiveCount   int              `json:"collectiveCount"`
	DiscoveryTimeline []DiscoveryPoint `json:"discoveryTimeline"`
	CouncilEfficiency float64          `json:"councilEfficiency"` // Tasks per 100 agents
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

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   👑 THE COUNCIL OF 1000 - Massive Scale Architecture Search                             ║")
	fmt.Println("║                                                                                          ║")
	fmt.Printf("║   Spawning %d agents with %d parallel workers...                                       ║\n", CouncilSize, numWorkers)
	fmt.Println("║   Testing statistical saturation: How many unique tasks can 1000 minds solve?           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	// Load ARC-AGI2 training data
	trainTasks, err := loadARCTasks("ARC-AGI2/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load ARC-AGI2 training tasks: %v\n", err)
		return
	}

	// Load ARC-AGI2 evaluation data
	evalTasks, err := loadARCTasks("ARC-AGI2/data/evaluation", 120)
	if err != nil {
		fmt.Printf("❌ Failed to load ARC-AGI2 evaluation tasks: %v\n", err)
		return
	}

	trainSamples := createSequentialSamples(trainTasks)
	evalSamples := createEvalSamples(evalTasks)

	fmt.Printf("\n📦 Loaded %d ARC-AGI2 training tasks, %d train samples\n", len(trainTasks), len(trainSamples))
	fmt.Printf("📦 Loaded %d ARC-AGI2 evaluation tasks, %d eval samples\n", len(evalTasks), len(evalSamples))
	fmt.Printf("👑 Generating %d Council member configurations...\n\n", CouncilSize)

	// Generate random configurations
	configs := generateAgentConfigs(CouncilSize)

	// Track collective tasks with mutex for thread safety
	var (
		results             = make([]AgentResult, CouncilSize)
		collectiveTasksMu   sync.Mutex
		collectiveTasks     = make(map[string]bool)
		discoveryTimeline   []DiscoveryPoint
		processedCount      int
		lastReportedPercent int
	)

	// Worker pool
	startTime := time.Now()
	jobs := make(chan int, CouncilSize)
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				result := runCouncilMember(configs[idx], trainSamples, evalSamples)
				results[idx] = result

				// Update collective tasks (thread-safe)
				collectiveTasksMu.Lock()
				newTasks := 0
				for _, taskID := range result.SolvedTaskIDs {
					if !collectiveTasks[taskID] {
						collectiveTasks[taskID] = true
						newTasks++
					}
				}
				processedCount++

				// Record discovery point every 10 agents
				if processedCount%10 == 0 {
					discoveryTimeline = append(discoveryTimeline, DiscoveryPoint{
						AgentNum:          processedCount,
						CumulativeUnique:  len(collectiveTasks),
						NewTasksThisAgent: newTasks,
					})
				}

				// Progress report
				percent := processedCount * 100 / CouncilSize
				if percent >= lastReportedPercent+10 {
					lastReportedPercent = percent
					fmt.Printf("🔄 Progress: %d/%d (%.0f%%) | Unique tasks discovered: %d\n",
						processedCount, CouncilSize, float64(percent), len(collectiveTasks))
				}
				collectiveTasksMu.Unlock()
			}
		}()
	}

	// Send jobs
	for i := 0; i < CouncilSize; i++ {
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

	// Build collective tasks list
	var collectiveTasksList []string
	for taskID := range collectiveTasks {
		collectiveTasksList = append(collectiveTasksList, taskID)
	}
	sort.Strings(collectiveTasksList)

	// Top 20 experts
	topN := 20
	if len(results) < topN {
		topN = len(results)
	}

	// Build output
	output := &CouncilResults{
		CouncilSize:       CouncilSize,
		TopExperts:        results[:topN],
		CollectiveTasks:   collectiveTasksList,
		CollectiveCount:   len(collectiveTasksList),
		DiscoveryTimeline: discoveryTimeline,
		CouncilEfficiency: float64(len(collectiveTasksList)) / float64(CouncilSize) * 100,
		Timestamp:         time.Now().Format(time.RFC3339),
		Duration:          totalTime.String(),
		WorkerCount:       numWorkers,
	}

	// Save results
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("council_results_2.json", data, 0644)
	fmt.Println("\n✅ Results saved to council_results_2.json")

	// Print results
	printCouncilResults(output, results)
}

func generateAgentConfigs(count int) []AgentConfig {
	configs := make([]AgentConfig, count)

	gridSizes := []int{1, 2} // Focus on 1x1 and 2x2 (faster)
	dModels := []int{16, 32} // Smaller models work better
	numHeads := []int{2, 4, 8}

	for i := 0; i < count; i++ {
		gridSize := gridSizes[rand.Intn(len(gridSizes))]
		numBrains := gridSize * gridSize

		brains := make([]BrainType, numBrains)
		brainNames := make([]string, numBrains)
		for b := 0; b < numBrains; b++ {
			brainType := BrainType(rand.Intn(4))
			brains[b] = brainType
			brainNames[b] = brainTypeNames[brainType]
		}

		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		configs[i] = AgentConfig{
			ID:           i,
			Name:         fmt.Sprintf("Agent-%d", i),
			GridSize:     gridSize,
			DModel:       dModel,
			NumHeads:     heads,
			LearningRate: float32(0.001 + rand.Float64()*0.099),
			BudgetScale:  float32(0.5 + rand.Float64()*0.5),
			Brains:       brains,
			BrainNames:   brainNames,
		}
	}

	return configs
}

func runCouncilMember(config AgentConfig, trainSamples, evalSamples []Sample) AgentResult {
	numWindows := int(TestDuration / WindowDuration)
	windows := make([]TimeWindow, numWindows)

	for i := range windows {
		windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	// Create network
	net := createAgentNetwork(config)
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

	// Eval phase
	taskResults := make(map[string]struct {
		totalAcc float64
		count    int
	})

	for _, sample := range evalSamples {
		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()

		acc := calculatePixelAccuracy(output, sample)
		r := taskResults[sample.TaskID]
		r.totalAcc += acc
		r.count++
		taskResults[sample.TaskID] = r
	}

	// Find solved tasks (100% accuracy required)
	var solvedIDs []string
	for taskID, r := range taskResults {
		if r.count > 0 && r.totalAcc/float64(r.count) >= 100 {
			solvedIDs = append(solvedIDs, taskID)
		}
	}

	return AgentResult{
		Config:           config,
		AvgTrainAccuracy: avgAcc,
		TasksSolved:      len(solvedIDs),
		SolvedTaskIDs:    solvedIDs,
		ThroughputPerSec: float64(totalOutputs) / trainTime,
		Score:            avgAcc * float64(len(solvedIDs)+1),
	}
}

func createAgentNetwork(config AgentConfig) *nn.Network {
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
	parallelLayer := createAgentHive(config)
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

func createAgentHive(config AgentConfig) nn.LayerConfig {
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

	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   config.GridSize,
		GridOutputCols:   config.GridSize,
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

func printCouncilResults(output *CouncilResults, allResults []AgentResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    👑 THE COUNCIL OF 1000 - RESULTS                                                          ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Council Size: %d | Unique Tasks Solved: %d | Efficiency: %.2f tasks per 100 agents                                        ║\n",
		output.CouncilSize, output.CollectiveCount, output.CouncilEfficiency)
	fmt.Printf("║   Duration: %s | Workers: %d                                                                                             ║\n",
		output.Duration, output.WorkerCount)
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                              🏆 TOP 10 EXPERTS                                                               ║")
	fmt.Println("╠═════════════════╦═══════════════════════════════════════════════════════════════════════╦══════════╦══════════╦═══════════════╣")
	fmt.Println("║     Agent       ║ Architecture                                                         ║ Accuracy ║ Solved   ║ LR            ║")
	fmt.Println("╠═════════════════╬═══════════════════════════════════════════════════════════════════════╬══════════╬══════════╬═══════════════╣")

	for i := 0; i < 10 && i < len(output.TopExperts); i++ {
		r := output.TopExperts[i]
		cfg := r.Config
		archDesc := fmt.Sprintf("%dx%d | D=%d | %v", cfg.GridSize, cfg.GridSize, cfg.DModel, cfg.BrainNames)
		if len(archDesc) > 67 {
			archDesc = archDesc[:64] + "..."
		}
		fmt.Printf("║ %-15s ║ %-69s ║ %6.1f%% ║ %8d ║ %.4f        ║\n",
			r.Config.Name, archDesc, r.AvgTrainAccuracy, r.TasksSolved, cfg.LearningRate)
	}

	fmt.Println("╚═════════════════╩═══════════════════════════════════════════════════════════════════════╩══════════╩══════════╩═══════════════╝")

	// Discovery curve summary
	fmt.Println("\n📈 DISCOVERY CURVE:")
	for _, dp := range output.DiscoveryTimeline {
		bar := ""
		for j := 0; j < dp.CumulativeUnique; j++ {
			bar += "█"
		}
		fmt.Printf("   Agent %4d: %s (%d tasks)\n", dp.AgentNum, bar, dp.CumulativeUnique)
	}

	// Collective wisdom
	fmt.Printf("\n🧠 COLLECTIVE WISDOM: %d unique tasks solved by the Council\n", output.CollectiveCount)
	fmt.Printf("   Tasks: %v\n", output.CollectiveTasks)
}

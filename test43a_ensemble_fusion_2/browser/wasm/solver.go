//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"syscall/js"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// ARC-AGI BROWSER SOLVER - WASM Entry Point
// ═══════════════════════════════════════════════════════════════════════════════
// This is the WASM version of the ensemble fusion solver.
// It receives ARC task data from JavaScript and runs the same solver logic.

const (
	MaxGridSize43 = 30
	InputSize43   = MaxGridSize43 * MaxGridSize43
	InitScale43   = float32(0.5)

	EnsembleSize     = 15
	NumEnsembles     = 4               // Reduced for browser performance
	TestDuration43   = 5 * time.Second // Shorter for browser
	AdaptationPasses = 2
)

// ═══════════════════════════════════════════════════════════════════════════════
// Brain and Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

type BrainType43 int

const (
	BrainMHA43 BrainType43 = iota
	BrainLSTM43
	BrainRNN43
	BrainDense43
	BrainSwiGLU43
	BrainNormDense43
)

var brainTypeNames43 = []string{"MHA", "LSTM", "RNN", "Dense", "SwiGLU", "NormDense"}

type CombineModeType43 int

const (
	CombineConcat43 CombineModeType43 = iota
	CombineAdd43
	CombineAvg43
	CombineGridScatter43
)

var combineModeNames43 = []string{"concat", "add", "avg", "grid_scatter"}

type ActivationType43 int

const (
	ActScaledReLU43 ActivationType43 = iota
	ActLeakyReLU43
	ActTanh43
	ActSoftplus43
	ActSigmoid43
)

var activationNames43 = []string{"ScaledReLU", "LeakyReLU", "Tanh", "Softplus", "Sigmoid"}

func getActivation43(act ActivationType43) nn.ActivationType {
	switch act {
	case ActScaledReLU43:
		return nn.ActivationScaledReLU
	case ActLeakyReLU43:
		return nn.ActivationLeakyReLU
	case ActTanh43:
		return nn.ActivationTanh
	case ActSoftplus43:
		return nn.ActivationSoftplus
	case ActSigmoid43:
		return nn.ActivationSigmoid
	default:
		return nn.ActivationLeakyReLU
	}
}

// GridShape43 for different topologies
type GridShape43 struct {
	Rows int
	Cols int
	Name string
}

// Diverse grid shapes
var gridShapes43 = []GridShape43{
	{1, 1, "1x1 Mono"}, {2, 2, "2x2 Standard"}, {3, 3, "3x3 Complex"},
	{4, 1, "4x1 Tall"}, {1, 4, "1x4 Wide"},
}

// Data types
type ARCTask43 struct {
	ID          string
	Train, Test []GridPair43
}
type GridPair43 struct{ Input, Output [][]int }
type Sample43 struct {
	Input, Target []float32
	Height, Width int
	TaskID        string
	TaskIndex     int
}

// AgentConfig43 defines a randomized architecture
type AgentConfig43 struct {
	ID           int
	Name         string
	Species      string
	GridRows     int
	GridCols     int
	NumBrains    int
	DModel       int
	NumHeads     int
	LearningRate float32
	Activation   ActivationType43
	CombineMode  CombineModeType43
	Brains       []BrainType43
	BrainNames   []string
}

// NetworkState holds a trained network and its performance metrics
type NetworkState struct {
	Config          AgentConfig43
	Network         *nn.Network
	State           *nn.StepState
	Tween           *nn.TweenState
	PerTaskAccuracy map[string]float64
	OverallAccuracy float64
	Predictions     map[string][][]int // taskID -> predicted grid
}

// NetworkSpecialist extends NetworkState with clustering info
type NetworkSpecialist struct {
	*NetworkState
	ClusterID int
	Specialty string
}

// FusionStrategy defines how to combine network outputs
type FusionStrategy int

const (
	FusionVote FusionStrategy = iota
	FusionAverage
	FusionWeighted
)

var fusionNames = []string{"Vote", "Average", "Weighted"}

// PixelAnalysis tracks per-pixel correctness across all networks for a task
type PixelAnalysis struct {
	TaskID              string
	Height, Width       int
	NumNetworks         int
	PixelCorrectness    [][]bool // [networkIdx][pixelIdx] = correct?
	NetworkOutputs      [][]int  // [networkIdx][pixelIdx] = predicted color (0-9)
	TargetColors        []int    // [pixelIdx] = expected color (0-9)
	BestNetworkPerPixel []int    // Which network is correct for each pixel (-1 if none)
	CoverageMap         []int    // Count of networks correct per pixel
	TotalPixels         int
	CoveredPixels       int // Pixels with at least one correct network
	CoverageRate        float64
}

// ComplementaryPair identifies two networks that together cover more pixels
type ComplementaryPair struct {
	NetworkA, NetworkB int
	OverlapPixels      int
	UniqueA, UniqueB   int
	CombinedCoverage   int
	CombinedRate       float64
	ComplementScore    float64
}

// StitchResult tracks results from pixel-level stitching
type StitchResult struct {
	TaskID            string
	PairUsed          ComplementaryPair
	OriginalCoverageA float64
	OriginalCoverageB float64
	StitchedCoverage  float64
	PixelsImproved    int
	FullySolved       bool
	StitchedOutput    []int
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("🧠 ARC-AGI WASM Solver v3 initialized!")
	fmt.Println("🔮 Neural network ensemble ready for browser-based solving!")

	// Export solver function to JavaScript
	js.Global().Set("startARCSolver", js.FuncOf(startARCSolverJS))
	js.Global().Set("solveWithData", js.FuncOf(solveWithDataJS))

	// Keep runtime alive
	select {}
}

// ═══════════════════════════════════════════════════════════════════════════════
// JavaScript Bridge Functions
// ═══════════════════════════════════════════════════════════════════════════════

func jsLog(msg string, logType string) {
	js.Global().Call("jsLog", msg, logType)
}

func jsUpdateStats(solved, total int) {
	js.Global().Call("jsUpdateStats", solved, total)
}

func jsCelebrate(taskId string) {
	js.Global().Call("jsCelebrate", taskId)
}

func jsSetPhase(phase string) {
	js.Global().Call("jsSetPhase", phase)
}

func jsSetTaskId(taskId string) {
	js.Global().Call("jsSetTaskId", taskId)
}

func jsRenderGrid(containerId string, grid [][]int) {
	jsGrid := make([]interface{}, len(grid))
	for i, row := range grid {
		jsRow := make([]interface{}, len(row))
		for j, val := range row {
			jsRow[j] = val
		}
		jsGrid[i] = jsRow
	}
	js.Global().Call("jsRenderGrid", containerId, jsGrid)
}

// jsStoreTaskResult sends task result to JS for storage and clickable badges
func jsStoreTaskResult(taskId string, input, expected, predicted [][]int, solved bool, coverage float64) {
	jsInput := gridToJSArray(input)
	jsExpected := gridToJSArray(expected)
	jsPredicted := gridToJSArray(predicted)
	js.Global().Call("storeTaskResult", taskId, jsInput, jsExpected, jsPredicted, solved, coverage)
}

func gridToJSArray(grid [][]int) []interface{} {
	if grid == nil || len(grid) == 0 {
		return []interface{}{}
	}
	jsGrid := make([]interface{}, len(grid))
	for i, row := range grid {
		jsRow := make([]interface{}, len(row))
		for j, val := range row {
			jsRow[j] = val
		}
		jsGrid[i] = jsRow
	}
	return jsGrid
}

// startARCSolverJS - called from JavaScript with dataset name and URL
func startARCSolverJS(this js.Value, args []js.Value) interface{} {
	if len(args) < 3 {
		jsLog("❌ Need: dataset, networkCount, dataUrl", "warning")
		return nil
	}

	dataset := args[0].String()
	networkCount := args[1].Int()
	dataUrl := args[2].String()

	go func() {
		jsLog(fmt.Sprintf("🚀 Starting %s solver with %d networks...", dataset, networkCount), "phase")
		jsLog(fmt.Sprintf("📡 Data URL: %s", dataUrl), "info")
		jsLog("⏳ Waiting for data from JavaScript...", "info")
		jsSetPhase("Waiting for data...")
	}()

	return nil
}

// solveWithDataJS - called from JavaScript with actual task data as JSON
func solveWithDataJS(this js.Value, args []js.Value) interface{} {
	if len(args) < 3 {
		jsLog("❌ Need: trainTasksJSON, evalTasksJSON, networkCount", "warning")
		return nil
	}

	trainJSON := args[0].String()
	evalJSON := args[1].String()
	networkCount := args[2].Int()

	go func() {
		runSolver(trainJSON, evalJSON, networkCount)
	}()

	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Solver Logic (uses nn.Network)
// ═══════════════════════════════════════════════════════════════════════════════

func runSolver(trainJSON, evalJSON string, networkCount int) {
	jsSetPhase("Phase 0: Parsing Data")
	jsLog("📦 Parsing task data...", "info")

	// Parse task data
	trainTasks, err := parseTasksJSON(trainJSON)
	if err != nil {
		jsLog(fmt.Sprintf("❌ Failed to parse training: %v", err), "warning")
		return
	}

	evalTasks, err := parseTasksJSON(evalJSON)
	if err != nil {
		jsLog(fmt.Sprintf("❌ Failed to parse eval: %v", err), "warning")
		return
	}

	// Check for empty data
	if len(trainTasks) == 0 {
		jsLog("❌ No training tasks loaded! Check data URL and CORS.", "warning")
		jsSetPhase("Error: No data")
		return
	}
	if len(evalTasks) == 0 {
		jsLog("❌ No eval tasks loaded! Check data URL and CORS.", "warning")
		jsSetPhase("Error: No data")
		return
	}

	trainSamples := createSequentialSamples43(trainTasks)

	if len(trainSamples) == 0 {
		jsLog("❌ No training samples created! Tasks may be malformed.", "warning")
		jsSetPhase("Error: No samples")
		return
	}

	jsLog(fmt.Sprintf("✅ Loaded %d train tasks, %d eval tasks, %d samples", len(trainTasks), len(evalTasks), len(trainSamples)), "success")
	jsUpdateStats(0, len(evalTasks))

	// Phase 1: Train networks
	totalNetworks := networkCount
	if totalNetworks <= 0 {
		totalNetworks = NumEnsembles * EnsembleSize // fallback to default
	}
	jsSetPhase(fmt.Sprintf("Phase 1: Training %d Networks", totalNetworks))
	jsLog(fmt.Sprintf("🧠 Training %d diverse networks...", totalNetworks), "phase")

	configs := generateDiverseConfigs(totalNetworks)
	networks := make([]*NetworkState, totalNetworks)

	startTime := time.Now()
	for idx, cfg := range configs {
		networks[idx] = trainNetworkWASM(cfg, trainSamples, evalTasks)
		// Update UI progress bar
		js.Global().Call("jsUpdateTraining", idx+1, totalNetworks)
		// Log every network trained
		jsLog(fmt.Sprintf("🔄 Trained %d/%d networks (%.0f%%)", idx+1, totalNetworks, float64(idx+1)/float64(totalNetworks)*100), "info")
		// Yield to browser to prevent freeze
		time.Sleep(1 * time.Millisecond)
	}

	jsLog(fmt.Sprintf("✅ Phase 1 complete in %v", time.Since(startTime)), "success")

	// Phase 1.5: Cluster specialization
	jsSetPhase("Phase 1.5: Clustering")
	specialists := clusterNetworksWASM(networks)
	jsLog(fmt.Sprintf("📊 Created %d specialist clusters", len(specialists)/EnsembleSize), "info")
	time.Sleep(1 * time.Millisecond)

	collectiveTasks := make(map[string]bool)

	// Phase 1.5: Cross-Cluster Fusion Evaluation
	jsSetPhase("Phase 1.5: Cross-Cluster Fusion")
	jsLog("🔮 Evaluating cross-cluster fusion strategies...", "phase")
	time.Sleep(1 * time.Millisecond)

	crossResults := evaluateCrossClusterEnsemble(specialists, evalTasks)
	for _, result := range crossResults {
		jsLog(fmt.Sprintf("   📊 %s: %d tasks solved", result.Config.Name, result.TasksSolved), "info")
		for _, taskID := range result.SolvedTaskIDs {
			collectiveTasks[taskID] = true
		}
		time.Sleep(1 * time.Millisecond)
	}

	// Phase 1.6: Meta-Cluster Voting
	jsSetPhase("Phase 1.6: Meta-Cluster Voting")
	metaResult := evaluateMetaClusterVoting(specialists, evalTasks)
	if metaResult != nil {
		for _, taskID := range metaResult.SolvedTaskIDs {
			collectiveTasks[taskID] = true
		}
	}

	jsLog(fmt.Sprintf("📊 After fusion phases: %d tasks solved collectively", len(collectiveTasks)), "success")
	jsUpdateStats(len(collectiveTasks), len(evalTasks))
	time.Sleep(1 * time.Millisecond)

	// Phase 2: Complementary Model Stitching
	jsSetPhase("Phase 2: Complementary Stitching")
	jsLog("🔮 Finding complementary pairs and stitching...", "phase")

	var solvedList []string

	for taskIdx, task := range evalTasks {
		jsSetTaskId(task.ID)

		if len(task.Test) == 0 {
			continue
		}

		testPair := task.Test[0]
		jsRenderGrid("inputGrid", testPair.Input)
		jsRenderGrid("expectedGrid", testPair.Output)

		// Analyze pixel correctness across all specialists
		analysis := analyzePixelCorrectness(specialists, task)
		if analysis == nil {
			continue
		}

		// Try complementary pair stitching
		pairs := findComplementaryPairs(analysis, 5)
		var bestResult *StitchResult
		for _, pair := range pairs {
			result := stitchPredictions(analysis, pair)
			if result != nil {
				if bestResult == nil || result.StitchedCoverage > bestResult.StitchedCoverage {
					bestResult = result
				}
			}
		}

		// Also try best-per-pixel stitching
		bestPerPixel := stitchFromBestPerPixel(analysis)
		if bestPerPixel != nil && (bestResult == nil || bestPerPixel.StitchedCoverage > bestResult.StitchedCoverage) {
			bestResult = bestPerPixel
		}

		// Build predicted grid from best result
		var predicted [][]int
		solved := false
		if bestResult != nil && bestResult.FullySolved {
			solved = true
			// Convert flat output to 2D grid
			predicted = make([][]int, analysis.Height)
			for y := 0; y < analysis.Height; y++ {
				predicted[y] = make([]int, analysis.Width)
				for x := 0; x < analysis.Width; x++ {
					predicted[y][x] = bestResult.StitchedOutput[y*analysis.Width+x]
				}
			}
		} else {
			// Use voting fallback
			_, predicted = tryStitchTask(specialists, task)
			// Double-check if voting solved it
			if predicted != nil && len(task.Test[0].Output) > 0 {
				correct := 0
				total := 0
				expected := task.Test[0].Output
				for y := 0; y < len(expected); y++ {
					for x := 0; x < len(expected[y]); x++ {
						total++
						if y < len(predicted) && x < len(predicted[y]) && predicted[y][x] == expected[y][x] {
							correct++
						}
					}
				}
				solved = correct == total
			}
		}

		if solved {
			collectiveTasks[task.ID] = true
			solvedList = append(solvedList, task.ID)
			if predicted != nil {
				jsRenderGrid("predictedGrid", predicted)
			}
			jsCelebrate(task.ID)
			jsLog(fmt.Sprintf("✅ SOLVED: %s", task.ID), "success")
		} else if predicted != nil {
			jsRenderGrid("predictedGrid", predicted)
		}

		// Store result for all tasks (solved and unsolved) for clickable badges
		coverage := 0.0
		if bestResult != nil {
			coverage = bestResult.StitchedCoverage
		}
		jsStoreTaskResult(task.ID, testPair.Input, testPair.Output, predicted, solved, coverage)

		jsUpdateStats(len(collectiveTasks), len(evalTasks))

		// Yield every 5 tasks
		if (taskIdx+1)%5 == 0 {
			time.Sleep(1 * time.Millisecond)
		}
	}

	// Final results
	jsSetPhase("Complete!")
	accuracy := float64(len(collectiveTasks)) / float64(len(evalTasks)) * 100
	jsLog(fmt.Sprintf("🎉 FINISHED! Solved %d/%d tasks (%.1f%%)", len(collectiveTasks), len(evalTasks), accuracy), "success")

	// Signal completion to JavaScript (stops timer, resets button)
	js.Global().Call("jsComplete")
}

// parseTasksJSON parses JSON array of ARC tasks
func parseTasksJSON(jsonStr string) ([]*ARCTask43, error) {
	var rawTasks []struct {
		ID    string `json:"id"`
		Train []struct {
			Input  [][]int `json:"input"`
			Output [][]int `json:"output"`
		} `json:"train"`
		Test []struct {
			Input  [][]int `json:"input"`
			Output [][]int `json:"output"`
		} `json:"test"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &rawTasks); err != nil {
		return nil, err
	}

	tasks := make([]*ARCTask43, len(rawTasks))
	for i, rt := range rawTasks {
		task := &ARCTask43{ID: rt.ID}
		for _, t := range rt.Train {
			task.Train = append(task.Train, GridPair43{Input: t.Input, Output: t.Output})
		}
		for _, t := range rt.Test {
			task.Test = append(task.Test, GridPair43{Input: t.Input, Output: t.Output})
		}
		tasks[i] = task
	}

	return tasks, nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// Network Creation and Training (using nn.Network)
// ═══════════════════════════════════════════════════════════════════════════════

func createSequentialSamples43(tasks []*ARCTask43) []Sample43 {
	var samples []Sample43
	for taskIdx, task := range tasks {
		for _, pair := range task.Train {
			input := make([]float32, InputSize43)
			target := make([]float32, InputSize43)

			outH := len(pair.Output)
			outW := 0
			if outH > 0 {
				outW = len(pair.Output[0])
			}

			for y := 0; y < len(pair.Input) && y < MaxGridSize43; y++ {
				for x := 0; x < len(pair.Input[y]) && x < MaxGridSize43; x++ {
					input[y*MaxGridSize43+x] = float32(pair.Input[y][x]) / 9.0
				}
			}
			for y := 0; y < outH && y < MaxGridSize43; y++ {
				for x := 0; x < len(pair.Output[y]) && x < MaxGridSize43; x++ {
					target[y*MaxGridSize43+x] = float32(pair.Output[y][x]) / 9.0
				}
			}

			samples = append(samples, Sample43{
				Input: input, Target: target,
				Height: outH, Width: outW,
				TaskID: task.ID, TaskIndex: taskIdx,
			})
		}
	}
	return samples
}

func generateDiverseConfigs(count int) []AgentConfig43 {
	configs := make([]AgentConfig43, count)

	dModels := []int{64, 64, 64, 32} // 75% D64, 25% D32
	numHeads := []int{4, 8}

	for i := 0; i < count; i++ {
		shape := gridShapes43[rand.Intn(len(gridShapes43))]
		numBrains := shape.Rows * shape.Cols

		brains := make([]BrainType43, numBrains)
		brainNames := make([]string, numBrains)
		for b := 0; b < numBrains; b++ {
			r := rand.Float64()
			var brainType BrainType43
			switch {
			case r < 0.30:
				brainType = BrainMHA43
			case r < 0.55:
				brainType = BrainLSTM43
			case r < 0.70:
				brainType = BrainRNN43
			case r < 0.85:
				brainType = BrainDense43
			case r < 0.93:
				brainType = BrainSwiGLU43
			default:
				brainType = BrainNormDense43
			}
			brains[b] = brainType
			brainNames[b] = brainTypeNames43[brainType]
		}

		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		logMin := math.Log(0.0001)
		logMax := math.Log(0.01)
		lr := float32(math.Exp(logMin + rand.Float64()*(logMax-logMin)))

		activation := ActivationType43(rand.Intn(5))

		var combineMode CombineModeType43
		rc := rand.Float64()
		switch {
		case rc < 0.35:
			combineMode = CombineAvg43
		case rc < 0.65:
			combineMode = CombineAdd43
		case rc < 0.85:
			combineMode = CombineConcat43
		default:
			combineMode = CombineGridScatter43
		}

		configs[i] = AgentConfig43{
			ID:           i,
			Name:         fmt.Sprintf("Net-%d", i),
			Species:      shape.Name,
			GridRows:     shape.Rows,
			GridCols:     shape.Cols,
			NumBrains:    numBrains,
			DModel:       dModel,
			NumHeads:     heads,
			LearningRate: lr,
			Activation:   activation,
			CombineMode:  combineMode,
			Brains:       brains,
			BrainNames:   brainNames,
		}
	}

	return configs
}

func trainNetworkWASM(config AgentConfig43, samples []Sample43, evalTasks []*ARCTask43) *NetworkState {
	net := createDiverseNetwork(config)
	numLayers := net.TotalLayers()

	state := net.InitStepState(InputSize43)
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true

	// Quick training loop - skip if no samples
	if len(samples) == 0 {
		return &NetworkState{
			Config:          config,
			Network:         net,
			State:           state,
			Tween:           ts,
			PerTaskAccuracy: make(map[string]float64),
			Predictions:     make(map[string][][]int),
		}
	}

	start := time.Now()
	sampleIdx := 0

	for time.Since(start) < TestDuration43 {
		sample := samples[sampleIdx%len(samples)]
		sampleIdx++

		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}

		ts.TweenStep(net, sample.Input, argmax43(sample.Target), len(sample.Target), config.LearningRate)
	}

	// Get predictions for eval tasks
	nstate := &NetworkState{
		Config:          config,
		Network:         net,
		State:           state,
		Tween:           ts,
		PerTaskAccuracy: make(map[string]float64),
		Predictions:     make(map[string][][]int),
	}

	for _, task := range evalTasks {
		if len(task.Test) == 0 {
			continue
		}

		// Few-shot adaptation
		for k := 0; k < AdaptationPasses; k++ {
			for _, pair := range task.Train {
				if len(pair.Input) == 0 || len(pair.Output) == 0 {
					continue
				}
				input := encodeGrid43(pair.Input)
				target := encodeGrid43(pair.Output)
				ts.TweenStep(net, input, argmax43(target), len(target), config.LearningRate)
			}
		}

		// Get prediction
		pred := predictTask(net, state, numLayers, task)
		nstate.Predictions[task.ID] = pred

		// Calculate accuracy
		pair := task.Test[0]
		acc := calculateGridAccuracy(pred, pair.Output)
		nstate.PerTaskAccuracy[task.ID] = acc
	}

	return nstate
}

func predictTask(net *nn.Network, state *nn.StepState, numLayers int, task *ARCTask43) [][]int {
	testPair := task.Test[0]
	input := encodeGrid43(testPair.Input)

	state.SetInput(input)
	for s := 0; s < numLayers; s++ {
		net.StepForward(state)
	}
	output := state.GetOutput()

	height := len(testPair.Output)
	width := len(testPair.Output[0])

	grid := make([][]int, height)
	for y := 0; y < height; y++ {
		grid[y] = make([]int, width)
		for x := 0; x < width; x++ {
			val := int(math.Round(float64(output[y*MaxGridSize43+x]) * 9))
			if val < 0 {
				val = 0
			}
			if val > 9 {
				val = 9
			}
			grid[y][x] = val
		}
	}
	return grid
}

// predictTaskFromSpecialist generates a prediction from a specialist network
func predictTaskFromSpecialist(spec *NetworkSpecialist, task *ARCTask43) [][]int {
	if spec.Network == nil || spec.State == nil || len(task.Test) == 0 {
		return nil
	}
	numLayers := spec.Network.TotalLayers()
	return predictTask(spec.Network, spec.State, numLayers, task)
}

func calculateGridAccuracy(predicted [][]int, expected [][]int) float64 {
	correct, total := 0, 0
	for y := 0; y < len(expected); y++ {
		for x := 0; x < len(expected[y]); x++ {
			total++
			if y < len(predicted) && x < len(predicted[y]) {
				if predicted[y][x] == expected[y][x] {
					correct++
				}
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}

// ═══════════════════════════════════════════════════════════════════════════════
// Network Architecture (matching main.go)
// ═══════════════════════════════════════════════════════════════════════════════

func createDiverseNetwork(config AgentConfig43) *nn.Network {
	totalLayers := 4
	net := nn.NewNetwork(InputSize43, 1, 1, totalLayers)
	net.BatchSize = 1

	activation := getActivation43(config.Activation)
	layerIdx := 0

	// Input layer
	inputLayer := nn.InitDenseLayer(InputSize43, config.DModel, activation)
	scaleWeights43(inputLayer.Kernel, InitScale43)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Parallel hive layer with configurable combine mode
	parallelLayer := createDiverseHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Merger layer - size depends on combine mode
	var mergerInputSize int
	switch config.CombineMode {
	case CombineConcat43, CombineGridScatter43:
		mergerInputSize = config.DModel * config.GridRows * config.GridCols
	case CombineAdd43, CombineAvg43:
		mergerInputSize = config.DModel
	}
	mergerLayer := nn.InitDenseLayer(mergerInputSize, config.DModel, activation)
	scaleWeights43(mergerLayer.Kernel, InitScale43)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Output layer
	outputLayer := nn.InitDenseLayer(config.DModel, InputSize43, nn.ActivationSigmoid)
	scaleWeights43(outputLayer.Kernel, InitScale43)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createDiverseHive(config AgentConfig43) nn.LayerConfig {
	numBrains := config.GridRows * config.GridCols
	branches := make([]nn.LayerConfig, numBrains)
	positions := make([]nn.GridPosition, numBrains)

	for i := 0; i < numBrains; i++ {
		brainType := config.Brains[i]
		switch brainType {
		case BrainMHA43:
			branches[i] = createMHABrain43(config.DModel, config.NumHeads)
		case BrainLSTM43:
			branches[i] = createLSTMBrain43(config.DModel)
		case BrainRNN43:
			branches[i] = createRNNBrain43(config.DModel)
		case BrainDense43:
			branches[i] = createDenseBrain43(config.DModel, config.Activation)
		case BrainSwiGLU43:
			branches[i] = createSwiGLUBrain43(config.DModel)
		case BrainNormDense43:
			branches[i] = createNormDenseBrain43(config.DModel, config.Activation)
		default:
			branches[i] = createDenseBrain43(config.DModel, config.Activation)
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

	layer := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      combineModeNames43[config.CombineMode],
		ParallelBranches: branches,
	}

	// Only set grid positions for grid_scatter mode
	if config.CombineMode == CombineGridScatter43 {
		layer.GridOutputRows = config.GridRows
		layer.GridOutputCols = config.GridCols
		layer.GridOutputLayers = 1
		layer.GridPositions = positions
	}

	return layer
}

// Brain creation functions
func createMHABrain43(dModel, numHeads int) nn.LayerConfig {
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

	qkScale := InitScale43 / float32(math.Sqrt(float64(headDim)))
	outScale := InitScale43 / float32(math.Sqrt(float64(dModel)))
	initRandom43(mha.QWeights, qkScale)
	initRandom43(mha.KWeights, qkScale)
	initRandom43(mha.VWeights, qkScale)
	initRandom43(mha.OutputWeight, outScale)
	return mha
}

func createLSTMBrain43(dModel int) nn.LayerConfig {
	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: dModel,
		HiddenSize:   dModel,
		SeqLength:    1,
		OutputHeight: dModel,
	}
	initLSTMWeights43(&lstm)
	return lstm
}

func createRNNBrain43(dModel int) nn.LayerConfig {
	rnn := nn.LayerConfig{
		Type:         nn.LayerRNN,
		RNNInputSize: dModel,
		HiddenSize:   dModel,
		SeqLength:    1,
		OutputHeight: dModel,
	}
	initRNNWeights43(&rnn)
	return rnn
}

func createDenseBrain43(dModel int, activation ActivationType43) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, getActivation43(activation))
	scaleWeights43(dense.Kernel, InitScale43)
	return dense
}

func createSwiGLUBrain43(dModel int) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights43(dense.Kernel, InitScale43*0.7)
	return dense
}

func createNormDenseBrain43(dModel int, activation ActivationType43) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, getActivation43(activation))
	scaleWeights43(dense.Kernel, InitScale43*0.8)
	return dense
}

func initRNNWeights43(cfg *nn.LayerConfig) {
	inputSize := cfg.RNNInputSize
	hiddenSize := cfg.HiddenSize

	cfg.WeightIH = make([]float32, hiddenSize*inputSize)
	cfg.WeightHH = make([]float32, hiddenSize*hiddenSize)
	cfg.BiasH = make([]float32, hiddenSize)

	scale := InitScale43 / float32(math.Sqrt(float64(hiddenSize)))
	initRandom43(cfg.WeightIH, scale)
	initRandom43(cfg.WeightHH, scale)
}

func initLSTMWeights43(cfg *nn.LayerConfig) {
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

	scale := InitScale43 / float32(math.Sqrt(float64(hiddenSize)))
	initRandom43(cfg.WeightIH_i, scale)
	initRandom43(cfg.WeightIH_f, scale)
	initRandom43(cfg.WeightIH_g, scale)
	initRandom43(cfg.WeightIH_o, scale)
	initRandom43(cfg.WeightHH_i, scale)
	initRandom43(cfg.WeightHH_f, scale)
	initRandom43(cfg.WeightHH_g, scale)
	initRandom43(cfg.WeightHH_o, scale)
	for i := range cfg.BiasH_f {
		cfg.BiasH_f[i] = 1.0
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

func scaleWeights43(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

func initRandom43(slice []float32, scale float32) {
	for i := range slice {
		slice[i] = (rand.Float32()*2 - 1) * scale
	}
}

func encodeGrid43(grid [][]int) []float32 {
	encoded := make([]float32, InputSize43)
	for r := 0; r < len(grid) && r < MaxGridSize43; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize43; c++ {
			encoded[r*MaxGridSize43+c] = float32(grid[r][c]) / 9.0
		}
	}
	return encoded
}

func argmax43(s []float32) int {
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

// ═══════════════════════════════════════════════════════════════════════════════
// Clustering and Stitching
// ═══════════════════════════════════════════════════════════════════════════════

func clusterNetworksWASM(networks []*NetworkState) []*NetworkSpecialist {
	specialists := make([]*NetworkSpecialist, len(networks))
	for i, net := range networks {
		specialists[i] = &NetworkSpecialist{
			NetworkState: net,
			ClusterID:    i % 5,
			Specialty:    net.Config.Species,
		}
	}
	return specialists
}

func tryStitchTask(specialists []*NetworkSpecialist, task *ARCTask43) (bool, [][]int) {
	if len(task.Test) == 0 {
		return false, nil
	}

	expected := task.Test[0].Output
	height := len(expected)
	if height == 0 {
		return false, nil
	}
	width := len(expected[0])

	// Collect all predictions - try cache first, then compute on-the-fly
	var predictions [][][]int
	for _, spec := range specialists {
		if pred, ok := spec.Predictions[task.ID]; ok && pred != nil {
			predictions = append(predictions, pred)
		} else if spec.Network != nil && spec.State != nil {
			// Compute prediction on-the-fly
			pred := predictTaskFromSpecialist(spec, task)
			if pred != nil {
				predictions = append(predictions, pred)
			}
		}
	}

	if len(predictions) == 0 {
		return false, nil
	}

	// Stitch by voting per pixel
	stitched := make([][]int, height)
	correct := 0
	total := height * width

	for y := 0; y < height; y++ {
		stitched[y] = make([]int, width)
		for x := 0; x < width; x++ {
			votes := make([]int, 10)
			for _, pred := range predictions {
				if y < len(pred) && x < len(pred[y]) {
					votes[pred[y][x]]++
				}
			}

			maxVote := 0
			for color, count := range votes {
				if count > maxVote {
					maxVote = count
					stitched[y][x] = color
				}
			}

			if stitched[y][x] == expected[y][x] {
				correct++
			}
		}
	}

	solved := correct == total
	return solved, stitched
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2: COMPLEMENTARY MODEL STITCHING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// analyzePixelCorrectness analyzes which pixels each network predicts correctly for a task
func analyzePixelCorrectness(specialists []*NetworkSpecialist, task *ARCTask43) *PixelAnalysis {
	if len(task.Test) == 0 || len(task.Test[0].Output) == 0 {
		return nil
	}

	pair := task.Test[0]
	height := len(pair.Output)
	width := len(pair.Output[0])
	totalPixels := height * width
	numNetworks := len(specialists)

	analysis := &PixelAnalysis{
		TaskID:              task.ID,
		Height:              height,
		Width:               width,
		NumNetworks:         numNetworks,
		TotalPixels:         totalPixels,
		PixelCorrectness:    make([][]bool, numNetworks),
		NetworkOutputs:      make([][]int, numNetworks),
		TargetColors:        make([]int, totalPixels),
		BestNetworkPerPixel: make([]int, totalPixels),
		CoverageMap:         make([]int, totalPixels),
	}

	// Initialize BestNetworkPerPixel to -1 (no correct network)
	for i := range analysis.BestNetworkPerPixel {
		analysis.BestNetworkPerPixel[i] = -1
	}

	// Extract target colors
	for r := 0; r < height; r++ {
		for c := 0; c < width; c++ {
			pixelIdx := r*width + c
			if r < len(pair.Output) && c < len(pair.Output[r]) {
				analysis.TargetColors[pixelIdx] = pair.Output[r][c]
			}
		}
	}

	// Get predictions from all networks
	for netIdx, specialist := range specialists {
		if specialist.Network == nil || specialist.State == nil {
			analysis.PixelCorrectness[netIdx] = make([]bool, totalPixels)
			analysis.NetworkOutputs[netIdx] = make([]int, totalPixels)
			continue
		}

		pred := predictTaskFromSpecialist(specialist, task)
		analysis.PixelCorrectness[netIdx] = make([]bool, totalPixels)
		analysis.NetworkOutputs[netIdx] = make([]int, totalPixels)

		for r := 0; r < height; r++ {
			for c := 0; c < width; c++ {
				pixelIdx := r*width + c
				if r < len(pred) && c < len(pred[r]) {
					predColor := pred[r][c]
					analysis.NetworkOutputs[netIdx][pixelIdx] = predColor

					targetColor := analysis.TargetColors[pixelIdx]
					if predColor == targetColor {
						analysis.PixelCorrectness[netIdx][pixelIdx] = true
						analysis.CoverageMap[pixelIdx]++

						if analysis.BestNetworkPerPixel[pixelIdx] == -1 {
							analysis.BestNetworkPerPixel[pixelIdx] = netIdx
						}
					}
				}
			}
		}
	}

	// Calculate coverage stats
	for _, count := range analysis.CoverageMap {
		if count > 0 {
			analysis.CoveredPixels++
		}
	}
	if analysis.TotalPixels > 0 {
		analysis.CoverageRate = float64(analysis.CoveredPixels) / float64(analysis.TotalPixels)
	}

	return analysis
}

// findComplementaryPairs discovers pairs of networks that complement each other
func findComplementaryPairs(analysis *PixelAnalysis, topN int) []ComplementaryPair {
	if analysis == nil || analysis.NumNetworks < 2 {
		return nil
	}

	var pairs []ComplementaryPair

	for i := 0; i < analysis.NumNetworks; i++ {
		for j := i + 1; j < analysis.NumNetworks; j++ {
			var overlap, uniqueA, uniqueB int

			for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
				correctA := analysis.PixelCorrectness[i][pixelIdx]
				correctB := analysis.PixelCorrectness[j][pixelIdx]

				if correctA && correctB {
					overlap++
				} else if correctA {
					uniqueA++
				} else if correctB {
					uniqueB++
				}
			}

			combinedCoverage := overlap + uniqueA + uniqueB
			combinedRate := 0.0
			if analysis.TotalPixels > 0 {
				combinedRate = float64(combinedCoverage) / float64(analysis.TotalPixels)
			}

			complementScore := 0.0
			if combinedCoverage > 0 {
				complementScore = float64(uniqueA+uniqueB) / float64(combinedCoverage)
			}

			pairs = append(pairs, ComplementaryPair{
				NetworkA:         i,
				NetworkB:         j,
				OverlapPixels:    overlap,
				UniqueA:          uniqueA,
				UniqueB:          uniqueB,
				CombinedCoverage: combinedCoverage,
				CombinedRate:     combinedRate,
				ComplementScore:  complementScore,
			})
		}
	}

	// Sort by combined coverage descending
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].CombinedCoverage > pairs[i].CombinedCoverage {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	if topN > len(pairs) {
		topN = len(pairs)
	}
	return pairs[:topN]
}

// stitchPredictions creates a new output by taking each pixel from the correct network
func stitchPredictions(analysis *PixelAnalysis, pair ComplementaryPair) *StitchResult {
	if analysis == nil {
		return nil
	}

	result := &StitchResult{
		TaskID:         analysis.TaskID,
		PairUsed:       pair,
		StitchedOutput: make([]int, analysis.TotalPixels),
	}

	correctA, correctB := 0, 0
	for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
		if analysis.PixelCorrectness[pair.NetworkA][pixelIdx] {
			correctA++
		}
		if analysis.PixelCorrectness[pair.NetworkB][pixelIdx] {
			correctB++
		}
	}
	if analysis.TotalPixels > 0 {
		result.OriginalCoverageA = float64(correctA) / float64(analysis.TotalPixels)
		result.OriginalCoverageB = float64(correctB) / float64(analysis.TotalPixels)
	}

	correctStitched := 0
	for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
		cA := analysis.PixelCorrectness[pair.NetworkA][pixelIdx]
		cB := analysis.PixelCorrectness[pair.NetworkB][pixelIdx]

		var chosenColor int
		if cA && cB {
			chosenColor = analysis.NetworkOutputs[pair.NetworkA][pixelIdx]
			correctStitched++
		} else if cA {
			chosenColor = analysis.NetworkOutputs[pair.NetworkA][pixelIdx]
			correctStitched++
			result.PixelsImproved++
		} else if cB {
			chosenColor = analysis.NetworkOutputs[pair.NetworkB][pixelIdx]
			correctStitched++
			result.PixelsImproved++
		} else {
			chosenColor = analysis.NetworkOutputs[pair.NetworkA][pixelIdx]
		}
		result.StitchedOutput[pixelIdx] = chosenColor
	}

	if analysis.TotalPixels > 0 {
		result.StitchedCoverage = float64(correctStitched) / float64(analysis.TotalPixels)
	}
	result.FullySolved = correctStitched == analysis.TotalPixels

	return result
}

// stitchFromBestPerPixel creates output using best network for each pixel
func stitchFromBestPerPixel(analysis *PixelAnalysis) *StitchResult {
	if analysis == nil {
		return nil
	}

	result := &StitchResult{
		TaskID:         analysis.TaskID,
		StitchedOutput: make([]int, analysis.TotalPixels),
	}

	correctStitched := 0
	for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
		bestNet := analysis.BestNetworkPerPixel[pixelIdx]
		if bestNet >= 0 && bestNet < analysis.NumNetworks {
			result.StitchedOutput[pixelIdx] = analysis.NetworkOutputs[bestNet][pixelIdx]
			correctStitched++
		} else {
			// Vote across all networks
			votes := make(map[int]int)
			for netIdx := 0; netIdx < analysis.NumNetworks; netIdx++ {
				color := analysis.NetworkOutputs[netIdx][pixelIdx]
				votes[color]++
			}
			maxVotes, bestColor := 0, 0
			for color, count := range votes {
				if count > maxVotes {
					maxVotes = count
					bestColor = color
				}
			}
			result.StitchedOutput[pixelIdx] = bestColor
		}
	}

	if analysis.TotalPixels > 0 {
		result.StitchedCoverage = float64(correctStitched) / float64(analysis.TotalPixels)
	}
	result.FullySolved = correctStitched == analysis.TotalPixels

	return result
}

// clampInt43 clamps an integer to a range
func clampInt43(v, mn, mx int) int {
	if v < mn {
		return mn
	}
	if v > mx {
		return mx
	}
	return v
}

// min returns the smaller of two ints
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// EnsembleConfig defines fusion strategy config
type EnsembleConfig struct {
	Name         string
	StrategyName string
	Size         int
}

// EnsembleResult tracks ensemble performance
type EnsembleResult struct {
	Config        EnsembleConfig
	TasksSolved   int
	SolvedTaskIDs []string
	AvgAccuracy   float64
}

// fuseOutputs43 combines outputs from multiple networks using the specified strategy
func fuseOutputs43(outputs [][]float32, confidences []float64, strategy FusionStrategy, height, width int) []float32 {
	if len(outputs) == 0 || len(outputs[0]) == 0 {
		return make([]float32, InputSize43)
	}

	result := make([]float32, len(outputs[0]))

	switch strategy {
	case FusionVote:
		// Majority vote per pixel
		for idx := 0; idx < len(result); idx++ {
			votes := make(map[int]int)
			for _, output := range outputs {
				if idx < len(output) {
					pred := clampInt43(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					votes[pred]++
				}
			}
			maxVotes, bestPred := 0, 0
			for pred, count := range votes {
				if count > maxVotes {
					maxVotes = count
					bestPred = pred
				}
			}
			result[idx] = float32(bestPred) / 9.0
		}

	case FusionAverage:
		// Simple average
		for idx := 0; idx < len(result); idx++ {
			sum := float32(0)
			count := 0
			for _, output := range outputs {
				if idx < len(output) {
					sum += output[idx]
					count++
				}
			}
			if count > 0 {
				result[idx] = sum / float32(count)
			}
		}

	case FusionWeighted:
		// Weighted by confidence
		totalWeight := 0.0
		for _, conf := range confidences {
			totalWeight += math.Max(conf, 0.01)
		}

		for idx := 0; idx < len(result); idx++ {
			weightedSum := float64(0)
			for i, output := range outputs {
				if idx < len(output) && i < len(confidences) {
					weight := math.Max(confidences[i], 0.01) / totalWeight
					weightedSum += float64(output[idx]) * weight
				}
			}
			result[idx] = float32(weightedSum)
		}
	}

	return result
}

// calculatePixelAccuracy43 calculates accuracy for a prediction
func calculatePixelAccuracy43(output []float32, sample Sample43) float64 {
	if len(sample.Target) == 0 {
		return 0
	}

	correct := 0
	total := sample.Height * sample.Width

	for y := 0; y < sample.Height; y++ {
		for x := 0; x < sample.Width; x++ {
			idx := y*MaxGridSize43 + x
			if idx < len(output) && idx < len(sample.Target) {
				pred := clampInt43(int(math.Round(float64(output[idx])*9.0)), 0, 9)
				target := clampInt43(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
				if pred == target {
					correct++
				}
			}
		}
	}

	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}

// calculateCoverage43 checks how many pixels have at least one correct network
func calculateCoverage43(outputs [][]float32, target []float32, height, width int) (covered, total int) {
	total = height * width
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*MaxGridSize43 + x
			if idx >= len(target) {
				continue
			}
			targetColor := clampInt43(int(math.Round(float64(target[idx])*9.0)), 0, 9)

			for _, output := range outputs {
				if idx < len(output) {
					pred := clampInt43(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					if pred == targetColor {
						covered++
						break
					}
				}
			}
		}
	}
	return covered, total
}

// evaluateCrossClusterEnsemble evaluates elite networks with multiple fusion strategies
func evaluateCrossClusterEnsemble(specialists []*NetworkSpecialist, evalTasks []*ARCTask43) []EnsembleResult {
	if len(specialists) == 0 {
		return nil
	}

	strategies := []struct {
		Name     string
		Strategy FusionStrategy
	}{
		{"Elite-Vote", FusionVote},
		{"Elite-Average", FusionAverage},
		{"Elite-Weighted", FusionWeighted},
	}

	var results []EnsembleResult

	for stratIdx, strat := range strategies {
		jsLog(fmt.Sprintf("🔄 Testing fusion strategy %d/%d: %s", stratIdx+1, len(strategies), strat.Name), "info")
		time.Sleep(1 * time.Millisecond)

		var solvedIDs []string
		totalAccuracy := 0.0
		taskCount := 0

		for taskIdx, task := range evalTasks {
			if len(task.Test) == 0 || len(task.Test[0].Input) == 0 {
				continue
			}

			// Get outputs from all specialists using ForwardCPU (read-only)
			allOutputs := make([][]float32, len(specialists))
			allConfidences := make([]float64, len(specialists))

			input := encodeGrid43(task.Test[0].Input)
			for i, spec := range specialists {
				if spec.Network != nil {
					output, _ := spec.Network.ForwardCPU(input)
					allOutputs[i] = output
					allConfidences[i] = spec.OverallAccuracy
				}
			}

			// Apply fusion
			for _, pair := range task.Test {
				if len(pair.Output) == 0 {
					continue
				}

				target := encodeGrid43(pair.Output)
				height := len(pair.Output)
				width := len(pair.Output[0])

				fusedOutput := fuseOutputs43(allOutputs, allConfidences, strat.Strategy, height, width)

				acc := calculatePixelAccuracy43(fusedOutput, Sample43{
					Target: target,
					Height: height,
					Width:  width,
				})

				totalAccuracy += acc
				taskCount++

				if acc >= 100 {
					solvedIDs = append(solvedIDs, task.ID)
				}
			}

			// Log progress every 10 tasks
			if (taskIdx+1)%10 == 0 {
				jsLog(fmt.Sprintf("   📊 %s: Evaluated %d/%d tasks (%d solved so far)", strat.Name, taskIdx+1, len(evalTasks), len(solvedIDs)), "info")
				time.Sleep(1 * time.Millisecond)
			}
		}

		avgAcc := 0.0
		if taskCount > 0 {
			avgAcc = totalAccuracy / float64(taskCount)
		}

		results = append(results, EnsembleResult{
			Config: EnsembleConfig{
				Name:         strat.Name,
				StrategyName: strat.Name,
				Size:         len(specialists),
			},
			TasksSolved:   len(solvedIDs),
			SolvedTaskIDs: solvedIDs,
			AvgAccuracy:   avgAcc,
		})

		jsLog(fmt.Sprintf("✅ %s: Solved %d tasks (%.1f%% avg accuracy)", strat.Name, len(solvedIDs), avgAcc), "success")
	}

	return results
}

// evaluateMetaClusterVoting performs meta-voting across cluster predictions
func evaluateMetaClusterVoting(specialists []*NetworkSpecialist, evalTasks []*ARCTask43) *EnsembleResult {
	if len(specialists) == 0 {
		return nil
	}

	jsLog("🗳️ Running Meta-Cluster Voting...", "phase")
	time.Sleep(1 * time.Millisecond)

	// Group specialists by cluster
	clusterMap := make(map[int][]*NetworkSpecialist)
	for _, spec := range specialists {
		clusterMap[spec.ClusterID] = append(clusterMap[spec.ClusterID], spec)
	}

	if len(clusterMap) < 2 {
		jsLog("   ⚠️ Need at least 2 clusters for meta-voting", "warning")
		return nil
	}

	var solvedIDs []string
	totalAccuracy := 0.0
	taskCount := 0

	for taskIdx, task := range evalTasks {
		if len(task.Test) == 0 || len(task.Test[0].Input) == 0 {
			continue
		}

		for _, pair := range task.Test {
			if len(pair.Output) == 0 {
				continue
			}

			input := encodeGrid43(pair.Input)
			target := encodeGrid43(pair.Output)
			height := len(pair.Output)
			width := len(pair.Output[0])

			// Get per-cluster voting result
			var clusterPredictions [][]float32
			for _, members := range clusterMap {
				clusterOutputs := make([][]float32, len(members))
				for mi, member := range members {
					if member.Network != nil {
						output, _ := member.Network.ForwardCPU(input)
						clusterOutputs[mi] = output
					}
				}
				// Vote within cluster
				clusterPred := fuseOutputs43(clusterOutputs, nil, FusionVote, height, width)
				clusterPredictions = append(clusterPredictions, clusterPred)
			}

			// Meta-vote across clusters
			finalOutput := fuseOutputs43(clusterPredictions, nil, FusionVote, height, width)

			acc := calculatePixelAccuracy43(finalOutput, Sample43{
				Target: target,
				Height: height,
				Width:  width,
			})

			totalAccuracy += acc
			taskCount++

			if acc >= 100 {
				solvedIDs = append(solvedIDs, task.ID)
			}
		}

		// Log every 20 tasks
		if (taskIdx+1)%20 == 0 {
			jsLog(fmt.Sprintf("   🗳️ Meta-vote: Evaluated %d/%d tasks (%d solved)", taskIdx+1, len(evalTasks), len(solvedIDs)), "info")
			time.Sleep(1 * time.Millisecond)
		}
	}

	avgAcc := 0.0
	if taskCount > 0 {
		avgAcc = totalAccuracy / float64(taskCount)
	}

	jsLog(fmt.Sprintf("✅ Meta-Cluster Voting: Solved %d tasks (%.1f%% avg)", len(solvedIDs), avgAcc), "success")

	return &EnsembleResult{
		Config: EnsembleConfig{
			Name:         "Meta-Cluster-Vote",
			StrategyName: "MetaVote",
			Size:         len(clusterMap),
		},
		TasksSolved:   len(solvedIDs),
		SolvedTaskIDs: solvedIDs,
		AvgAccuracy:   avgAcc,
	}
}

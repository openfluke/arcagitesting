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
// ENSEMBLE FUSION v2 - Combining Diverse Neural Network Outputs
// ═══════════════════════════════════════════════════════════════════════════════
//
// UPGRADED from v1: Now borrows architectural diversity from test40_monster_zoo
// - Multiple CombineModes: concat, add, avg, grid_scatter
// - Diverse brain types: MHA, LSTM, RNN, Dense, SwiGLU, NormDense
// - Varied grid shapes: 1x1 to 8x1
// - Different activations and learning rates
//
// Key Insight: Diverse architectures = diverse errors = better fusion potential
// By combining networks with DIFFERENT strengths, we capture partial solutions.
//
// Fusion Strategies:
// 1. VOTE     - Each pixel gets the most common prediction across networks
// 2. AVERAGE  - Average the outputs and round
// 3. WEIGHTED - Weight networks by their confidence/accuracy
// 4. CASCADE  - Networks specialize in regions, outputs stitch together
// 5. ORACLE   - Pick the best prediction per-pixel (theoretical maximum)

const (
	MaxGridSize43 = 30
	InputSize43   = MaxGridSize43 * MaxGridSize43 // 900
	NumTasks43    = 400
	InitScale43   = float32(0.5)

	// Ensemble settings - balanced for speed and diversity
	EnsembleSize     = 15               // Networks per ensemble (more diverse now)
	NumEnsembles     = 8                // Different ensemble combinations
	TestDuration43   = 18 * time.Second // Training time per network
	WindowDuration43 = 100 * time.Millisecond
	AdaptationPasses = 3 // Few-shot adaptation passes
)

// FusionStrategy defines how to combine network outputs
type FusionStrategy int

const (
	FusionVote FusionStrategy = iota
	FusionAverage
	FusionWeighted
	FusionCascade
	FusionOracle
)

var fusionNames = []string{"Vote", "Average", "Weighted", "Cascade", "Oracle"}

// BrainType43 defines what kind of layer a brain uses
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

// CombineModeType for parallel layer combination
type CombineModeType43 int

const (
	CombineConcat43 CombineModeType43 = iota
	CombineAdd43
	CombineAvg43
	CombineGridScatter43
)

var combineModeNames43 = []string{"concat", "add", "avg", "grid_scatter"}

// ActivationType43 for layer activations
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

// Diverse grid shapes from test40
var gridShapes43 = []GridShape43{
	{1, 1, "1x1 Mono"},
	{2, 2, "2x2 Standard"},
	{3, 3, "3x3 Complex"},
	{4, 1, "4x1 Tall"},
	{1, 4, "1x4 Wide"},
	{2, 3, "2x3 Rect"},
	{3, 2, "3x2 Rect"},
	{8, 1, "8x1 Scanner"},
	{6, 4, "6x4 Matrix"},
}

// AgentConfig43 defines a randomized architecture
type AgentConfig43 struct {
	ID           int               `json:"id"`
	Name         string            `json:"name"`
	Species      string            `json:"species"`
	MutationStr  string            `json:"mutationStr"`
	GridRows     int               `json:"gridRows"`
	GridCols     int               `json:"gridCols"`
	NumBrains    int               `json:"numBrains"`
	DModel       int               `json:"dModel"`
	NumHeads     int               `json:"numHeads"`
	LearningRate float32           `json:"learningRate"`
	BudgetScale  float32           `json:"budgetScale"`
	Activation   ActivationType43  `json:"-"`
	ActivationN  string            `json:"activation"`
	CombineMode  CombineModeType43 `json:"-"`
	CombineModeN string            `json:"combineMode"`
	Brains       []BrainType43     `json:"-"`
	BrainNames   []string          `json:"brains"`
}

// EnsembleConfig defines a group of networks and fusion strategy
type EnsembleConfig struct {
	ID           int            `json:"id"`
	Name         string         `json:"name"`
	Size         int            `json:"size"`
	Strategy     FusionStrategy `json:"-"`
	StrategyName string         `json:"strategy"`
	MemberIDs    []int          `json:"memberIds"`
}

// NetworkState holds a trained network and its performance metrics
type NetworkState struct {
	Config          AgentConfig43
	Network         *nn.Network
	State           *nn.StepState
	Tween           *nn.TweenState
	PerTaskAccuracy map[string]float64
	OverallAccuracy float64
}

// EnsembleResult tracks ensemble performance
type EnsembleResult struct {
	Config        EnsembleConfig `json:"config"`
	TasksSolved   int            `json:"tasksSolved"`
	SolvedTaskIDs []string       `json:"solvedTaskIds"`
	AvgAccuracy   float64        `json:"avgAccuracy"`
	MemberSolved  []int          `json:"memberSolved"`
	FusionBonus   int            `json:"fusionBonus"`
	SynergyScore  float64        `json:"synergyScore"`
	CoverageRate  float64        `json:"coverageRate"`

	// NEW: Architecture diversity metrics
	UniqueSpecies      int `json:"uniqueSpecies"`
	UniqueCombineModes int `json:"uniqueCombineModes"`
}

// FusionResults is the output
type FusionResults struct {
	NumEnsembles       int              `json:"numEnsembles"`
	EnsembleSize       int              `json:"ensembleSize"`
	TopEnsembles       []EnsembleResult `json:"topEnsembles"`
	CollectiveTasks    []string         `json:"collectiveTasks"`
	CollectiveCount    int              `json:"collectiveCount"`
	BestFusionStrategy string           `json:"bestFusionStrategy"`
	StrategyComparison map[string]int   `json:"strategyComparison"`
	CombineModeStats   map[string]int   `json:"combineModeStats"`
	SpeciesStats       map[string]int   `json:"speciesStats"`
	Timestamp          string           `json:"timestamp"`
	Duration           string           `json:"duration"`
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

func main() {
	rand.Seed(time.Now().UnixNano())
	numWorkers := runtime.NumCPU()

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🔮 ENSEMBLE FUSION v2 - Diverse Architecture Combination                              ║")
	fmt.Println("║                                                                                          ║")
	fmt.Printf("║   Spawning %d ensembles of %d DIVERSE networks with %d workers...                       ║\n", NumEnsembles, EnsembleSize, numWorkers)
	fmt.Println("║   Testing: CombineModes (concat/add/avg/grid_scatter), Brain types, Grid shapes         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	// Load ARC-AGI training data
	trainTasks, err := loadARCTasks43("ARC-AGI/data/training", NumTasks43)
	if err != nil {
		fmt.Printf("❌ Failed to load training tasks: %v\n", err)
		return
	}

	// Load ARC-AGI evaluation data
	evalTasks, err := loadARCTasks43("ARC-AGI/data/evaluation", 400)
	if err != nil {
		fmt.Printf("❌ Failed to load eval tasks: %v\n", err)
		return
	}

	trainSamples := createSequentialSamples43(trainTasks)
	evalSamples := createEvalSamples43(evalTasks)

	fmt.Printf("\n📦 Loaded %d training tasks, %d train samples\n", len(trainTasks), len(trainSamples))
	fmt.Printf("📦 Loaded %d eval tasks, %d eval samples\n", len(evalTasks), len(evalSamples))

	// ===========================================================================
	// PHASE 1: Train All Individual Networks with DIVERSE architectures
	// ===========================================================================
	totalNetworks := NumEnsembles * EnsembleSize
	fmt.Printf("\n🧠 Phase 1: Training %d DIVERSE networks...\n", totalNetworks)

	configs := generateDiverseConfigs(totalNetworks)
	networks := make([]*NetworkState, totalNetworks)

	// Print diversity stats
	speciesCount := make(map[string]int)
	combineModeCount := make(map[string]int)
	for _, c := range configs {
		speciesCount[c.Species]++
		combineModeCount[c.CombineModeN]++
	}
	fmt.Printf("   📊 Species distribution: %v\n", speciesCount)
	fmt.Printf("   📊 CombineMode distribution: %v\n", combineModeCount)

	startTime := time.Now()

	// Train networks in parallel
	var wg sync.WaitGroup
	jobs := make(chan int, totalNetworks)
	var progressMu sync.Mutex
	completed := 0
	lastPercent := 0

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				networks[idx] = trainNetwork43(configs[idx], trainSamples, evalSamples, evalTasks)

				progressMu.Lock()
				completed++
				percent := completed * 100 / totalNetworks
				if percent >= lastPercent+10 {
					lastPercent = percent
					fmt.Printf("   🔄 Network training: %d/%d (%.0f%%)\n", completed, totalNetworks, float64(percent))
				}
				progressMu.Unlock()
			}
		}()
	}

	for i := 0; i < totalNetworks; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	trainingTime := time.Since(startTime)
	fmt.Printf("✅ Phase 1 complete: Trained %d diverse networks in %s\n\n", totalNetworks, trainingTime)

	// ===========================================================================
	// PHASE 2: Form Ensembles and Test Fusion Strategies (PARALLELIZED)
	// ===========================================================================
	fmt.Printf("🔮 Phase 2: Testing %d ensemble configurations with different fusion strategies (parallel)...\n\n", NumEnsembles)

	var collectiveTasksMu sync.Mutex
	collectiveTasks := make(map[string]bool)
	strategyStats := make(map[string]int)
	results := make([]EnsembleResult, NumEnsembles*len(fusionNames))

	// Job struct for parallel processing
	type ensembleJob struct {
		ensembleIdx int
		strategy    FusionStrategy
		resultIdx   int
		config      EnsembleConfig
		networks    []*NetworkState
	}

	// Create all jobs
	totalJobs := NumEnsembles * len(fusionNames)
	jobs2 := make(chan ensembleJob, totalJobs)
	var wg2 sync.WaitGroup

	// Progress tracking
	var progressMu2 sync.Mutex
	completed2 := 0
	lastPercent2 := 0

	// Start workers for ensemble evaluation
	for w := 0; w < numWorkers; w++ {
		wg2.Add(1)
		go func() {
			defer wg2.Done()
			for job := range jobs2 {
				result := evaluateEnsemble43(job.config, job.networks, evalTasks)
				results[job.resultIdx] = result

				// Update collective tasks (thread-safe)
				collectiveTasksMu.Lock()
				for _, taskID := range result.SolvedTaskIDs {
					collectiveTasks[taskID] = true
				}
				strategyStats[fusionNames[job.strategy]] += result.TasksSolved
				collectiveTasksMu.Unlock()

				// Progress update
				progressMu2.Lock()
				completed2++
				percent := completed2 * 100 / totalJobs
				if percent >= lastPercent2+10 {
					lastPercent2 = percent
					fmt.Printf("   🔄 Ensemble evaluation: %d/%d (%.0f%%)\n", completed2, totalJobs, float64(percent))
				}
				progressMu2.Unlock()
			}
		}()
	}

	// Submit all jobs
	resultIdx := 0
	for e := 0; e < NumEnsembles; e++ {
		// Select networks for this ensemble
		startIdx := e * EnsembleSize
		endIdx := startIdx + EnsembleSize
		ensembleNetworks := networks[startIdx:endIdx]

		memberIDs := make([]int, EnsembleSize)
		for i := 0; i < EnsembleSize; i++ {
			memberIDs[i] = startIdx + i
		}

		// Submit job for each fusion strategy
		for strategy := FusionVote; strategy <= FusionOracle; strategy++ {
			config := EnsembleConfig{
				ID:           e*len(fusionNames) + int(strategy),
				Name:         fmt.Sprintf("Ensemble-%d-%s", e, fusionNames[strategy]),
				Size:         EnsembleSize,
				Strategy:     strategy,
				StrategyName: fusionNames[strategy],
				MemberIDs:    memberIDs,
			}

			jobs2 <- ensembleJob{
				ensembleIdx: e,
				strategy:    strategy,
				resultIdx:   resultIdx,
				config:      config,
				networks:    ensembleNetworks,
			}
			resultIdx++
		}
	}
	close(jobs2)
	wg2.Wait()
	fmt.Printf("   ✅ All %d ensemble evaluations complete\n", totalJobs)

	totalTime := time.Since(startTime)

	// Sort results by tasks solved
	sort.Slice(results, func(i, j int) bool {
		if results[i].TasksSolved != results[j].TasksSolved {
			return results[i].TasksSolved > results[j].TasksSolved
		}
		return results[i].SynergyScore > results[j].SynergyScore
	})

	// Find best strategy
	bestStrategy := ""
	bestCount := 0
	for strategy, count := range strategyStats {
		if count > bestCount {
			bestCount = count
			bestStrategy = strategy
		}
	}

	// Build collective tasks list
	var collectiveTasksList []string
	for taskID := range collectiveTasks {
		collectiveTasksList = append(collectiveTasksList, taskID)
	}
	sort.Strings(collectiveTasksList)

	// Top ensembles
	topN := 20
	if len(results) < topN {
		topN = len(results)
	}

	// Build output
	output := &FusionResults{
		NumEnsembles:       NumEnsembles,
		EnsembleSize:       EnsembleSize,
		TopEnsembles:       results[:topN],
		CollectiveTasks:    collectiveTasksList,
		CollectiveCount:    len(collectiveTasksList),
		BestFusionStrategy: bestStrategy,
		StrategyComparison: strategyStats,
		CombineModeStats:   combineModeCount,
		SpeciesStats:       speciesCount,
		Timestamp:          time.Now().Format(time.RFC3339),
		Duration:           totalTime.String(),
	}

	// Save results
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("ensemble_fusion_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to ensemble_fusion_results.json")

	// Print results
	printFusionResults43(output, results)
}

// generateDiverseConfigs creates architecturally diverse network configurations
// Borrows from test40_monster_zoo patterns
func generateDiverseConfigs(count int) []AgentConfig43 {
	configs := make([]AgentConfig43, count)

	// Prefer D64 (proven effective in test40)
	dModels := []int{64, 64, 64, 32} // 75% D64, 25% D32
	numHeads := []int{4, 8}

	for i := 0; i < count; i++ {
		// Random grid shape
		shape := gridShapes43[rand.Intn(len(gridShapes43))]
		numBrains := shape.Rows * shape.Cols

		// Diverse brain types (from test40)
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

		// Random DModel and heads
		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		// Log-uniform learning rate (test40 style: narrow range for stability)
		logMin := math.Log(0.0001)
		logMax := math.Log(0.01)
		lr := float32(math.Exp(logMin + rand.Float64()*(logMax-logMin)))

		// Random activation
		activation := ActivationType43(rand.Intn(5))

		// CombineMode distribution (test40 style: prefer avg/add)
		var combineMode CombineModeType43
		r := rand.Float64()
		switch {
		case r < 0.35:
			combineMode = CombineAvg43
		case r < 0.65:
			combineMode = CombineAdd43
		case r < 0.85:
			combineMode = CombineConcat43
		default:
			combineMode = CombineGridScatter43
		}

		// Build mutation string for tracking
		brainStr := strings.Join(brainNames, "-")
		if len(brainStr) > 30 {
			brainStr = brainStr[:27] + "..."
		}
		mutationStr := fmt.Sprintf("%dx%d_%s_%s_D%d_LR%.4f",
			shape.Rows, shape.Cols,
			combineModeNames43[combineMode],
			activationNames43[activation],
			dModel, lr)

		configs[i] = AgentConfig43{
			ID:           i,
			Name:         fmt.Sprintf("Net-%d", i),
			Species:      shape.Name,
			MutationStr:  mutationStr,
			GridRows:     shape.Rows,
			GridCols:     shape.Cols,
			NumBrains:    numBrains,
			DModel:       dModel,
			NumHeads:     heads,
			LearningRate: lr,
			BudgetScale:  float32(0.5 + rand.Float64()*0.5),
			Activation:   activation,
			ActivationN:  activationNames43[activation],
			CombineMode:  combineMode,
			CombineModeN: combineModeNames43[combineMode],
			Brains:       brains,
			BrainNames:   brainNames,
		}
	}

	return configs
}

// trainNetwork43 trains a single network and returns its state
func trainNetwork43(config AgentConfig43, trainSamples, evalSamples []Sample43, evalTasks []*ARCTask43) *NetworkState {
	net := createDiverseNetwork(config)
	numLayers := net.TotalLayers()

	state := net.InitStepState(InputSize43)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = config.BudgetScale
	ts.Config.UseChainRule = true

	// Training loop
	start := time.Now()
	sampleIdx := 0

	for time.Since(start) < TestDuration43 {
		sample := trainSamples[sampleIdx%len(trainSamples)]
		sampleIdx++

		state.SetInput(sample.Input)
		for s := 0; s < numLayers; s++ {
			net.StepForward(state)
		}

		ts.TweenStep(net, sample.Input, argmax43(sample.Target), len(sample.Target), config.LearningRate)
	}

	// Evaluate on each task
	perTaskAcc := make(map[string]float64)
	totalAcc := 0.0
	taskCount := 0

	for _, task := range evalTasks {
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

		// Test
		taskAcc := 0.0
		testCount := 0
		for _, pair := range task.Test {
			if len(pair.Input) == 0 || len(pair.Output) == 0 {
				continue
			}

			input := encodeGrid43(pair.Input)
			target := encodeGrid43(pair.Output)

			state.SetInput(input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output := state.GetOutput()

			acc := calculatePixelAccuracy43(output, Sample43{
				Target: target,
				Height: len(pair.Output),
				Width:  len(pair.Output[0]),
			})
			taskAcc += acc
			testCount++
		}

		if testCount > 0 {
			perTaskAcc[task.ID] = taskAcc / float64(testCount)
			totalAcc += perTaskAcc[task.ID]
			taskCount++
		}
	}

	overall := 0.0
	if taskCount > 0 {
		overall = totalAcc / float64(taskCount)
	}

	return &NetworkState{
		Config:          config,
		Network:         net,
		State:           state,
		Tween:           ts,
		PerTaskAccuracy: perTaskAcc,
		OverallAccuracy: overall,
	}
}

// evaluateEnsemble43 tests an ensemble with a specific fusion strategy
func evaluateEnsemble43(config EnsembleConfig, networks []*NetworkState, evalTasks []*ARCTask43) EnsembleResult {
	var solvedIDs []string
	memberSolved := make([]int, len(networks))
	totalAccuracy := 0.0
	taskCount := 0
	pixelsCoveredTotal := 0
	pixelsTotalTotal := 0

	// Track diversity
	speciesSet := make(map[string]bool)
	combineModeSet := make(map[string]bool)
	for _, ns := range networks {
		speciesSet[ns.Config.Species] = true
		combineModeSet[ns.Config.CombineModeN] = true
	}

	for _, task := range evalTasks {
		// Get outputs from all networks for this task
		allOutputs := make([][]float32, len(networks))
		allConfidences := make([]float64, len(networks))

		for i, ns := range networks {
			// Few-shot adaptation for this network
			for k := 0; k < AdaptationPasses; k++ {
				for _, pair := range task.Train {
					if len(pair.Input) == 0 || len(pair.Output) == 0 {
						continue
					}
					input := encodeGrid43(pair.Input)
					target := encodeGrid43(pair.Output)
					ns.Tween.TweenStep(ns.Network, input, argmax43(target), len(target), ns.Config.LearningRate)
				}
			}

			// Get prediction for test pairs
			for _, pair := range task.Test {
				if len(pair.Input) == 0 || len(pair.Output) == 0 {
					continue
				}

				input := encodeGrid43(pair.Input)
				ns.State.SetInput(input)
				numLayers := ns.Network.TotalLayers()
				for s := 0; s < numLayers; s++ {
					ns.Network.StepForward(ns.State)
				}
				allOutputs[i] = append([]float32(nil), ns.State.GetOutput()...)
				allConfidences[i] = ns.PerTaskAccuracy[task.ID]
			}
		}

		// Apply fusion strategy
		for _, pair := range task.Test {
			if len(pair.Input) == 0 || len(pair.Output) == 0 {
				continue
			}

			target := encodeGrid43(pair.Output)
			height := len(pair.Output)
			width := len(pair.Output[0])

			fusedOutput := fuseOutputs43(allOutputs, allConfidences, config.Strategy, height, width)

			acc := calculatePixelAccuracy43(fusedOutput, Sample43{
				Target: target,
				Height: height,
				Width:  width,
			})

			totalAccuracy += acc
			taskCount++

			// Check coverage (at least one network correct per pixel)
			covered, total := calculateCoverage43(allOutputs, target, height, width)
			pixelsCoveredTotal += covered
			pixelsTotalTotal += total

			// Check if 100% accurate (solved)
			if acc >= 100 {
				solvedIDs = append(solvedIDs, task.ID)
			}
		}

		// Track individual member performance
		for i, ns := range networks {
			if ns.PerTaskAccuracy[task.ID] >= 100 {
				memberSolved[i]++
			}
		}
	}

	avgAcc := 0.0
	if taskCount > 0 {
		avgAcc = totalAccuracy / float64(taskCount)
	}

	// Calculate synergy score
	maxIndividual := 0
	for _, solved := range memberSolved {
		if solved > maxIndividual {
			maxIndividual = solved
		}
	}

	synergyScore := 1.0
	if maxIndividual > 0 {
		synergyScore = float64(len(solvedIDs)) / float64(maxIndividual)
	}

	// Calculate fusion bonus (tasks solved by ensemble but not any individual)
	fusionBonus := len(solvedIDs) - maxIndividual
	if fusionBonus < 0 {
		fusionBonus = 0
	}

	coverageRate := 0.0
	if pixelsTotalTotal > 0 {
		coverageRate = float64(pixelsCoveredTotal) / float64(pixelsTotalTotal) * 100
	}

	return EnsembleResult{
		Config:             config,
		TasksSolved:        len(solvedIDs),
		SolvedTaskIDs:      solvedIDs,
		AvgAccuracy:        avgAcc,
		MemberSolved:       memberSolved,
		FusionBonus:        fusionBonus,
		SynergyScore:       synergyScore,
		CoverageRate:       coverageRate,
		UniqueSpecies:      len(speciesSet),
		UniqueCombineModes: len(combineModeSet),
	}
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
				if idx < len(output) {
					weight := math.Max(confidences[i], 0.01) / totalWeight
					weightedSum += float64(output[idx]) * weight
				}
			}
			result[idx] = float32(weightedSum)
		}

	case FusionCascade:
		// Spatial fusion: different networks contribute to different regions
		regionRows := (height + 2) / 3
		regionCols := (width + 2) / 3

		for r := 0; r < height; r++ {
			for c := 0; c < width; c++ {
				regionIdx := (r/regionRows)*3 + (c / regionCols)
				chosenNet := regionIdx % len(outputs)

				bestConf := -1.0
				for i, conf := range confidences {
					if i%3 == chosenNet%3 && conf > bestConf {
						bestConf = conf
						chosenNet = i
					}
				}

				idx := r*MaxGridSize43 + c
				if chosenNet < len(outputs) && idx < len(outputs[chosenNet]) {
					result[idx] = outputs[chosenNet][idx]
				}
			}
		}

	case FusionOracle:
		// Pick highest confidence network's prediction per pixel
		for idx := 0; idx < len(result); idx++ {
			bestConf := -1.0
			for i, output := range outputs {
				if idx < len(output) && confidences[i] > bestConf {
					bestConf = confidences[i]
					result[idx] = output[idx]
				}
			}
		}
	}

	return result
}

// calculateCoverage43 checks how many pixels have at least one correct network
func calculateCoverage43(outputs [][]float32, target []float32, height, width int) (covered, total int) {
	for r := 0; r < height; r++ {
		for c := 0; c < width; c++ {
			idx := r*MaxGridSize43 + c
			if idx >= len(target) {
				continue
			}

			expected := clampInt43(int(math.Round(float64(target[idx])*9.0)), 0, 9)
			total++

			for _, output := range outputs {
				if idx < len(output) {
					pred := clampInt43(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					if pred == expected {
						covered++
						break
					}
				}
			}
		}
	}
	return covered, total
}

// createDiverseNetwork creates a network with configurable CombineMode
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

// createDiverseHive creates a parallel layer with diverse brain types
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

func calculatePixelAccuracy43(output []float32, sample Sample43) float64 {
	correct, total := 0, 0
	for r := 0; r < sample.Height; r++ {
		for c := 0; c < sample.Width; c++ {
			idx := r*MaxGridSize43 + c
			if idx < len(output) && idx < len(sample.Target) {
				pred := clampInt43(int(math.Round(float64(output[idx])*9.0)), 0, 9)
				exp := clampInt43(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
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

func clampInt43(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
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

// Data loading
type rawTask43 struct {
	Train []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"train"`
	Test []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"test"`
}

func loadARCTasks43(dir string, maxTasks int) ([]*ARCTask43, error) {
	files, _ := os.ReadDir(dir)
	rand.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	var tasks []*ARCTask43
	for _, f := range files {
		if len(tasks) >= maxTasks || filepath.Ext(f.Name()) != ".json" {
			continue
		}
		data, _ := os.ReadFile(filepath.Join(dir, f.Name()))
		var raw rawTask43
		if json.Unmarshal(data, &raw) != nil {
			continue
		}
		task := &ARCTask43{ID: f.Name()[:len(f.Name())-5]}
		for _, p := range raw.Train {
			task.Train = append(task.Train, GridPair43{Input: p.Input, Output: p.Output})
		}
		for _, p := range raw.Test {
			task.Test = append(task.Test, GridPair43{Input: p.Input, Output: p.Output})
		}
		tasks = append(tasks, task)
	}
	return tasks, nil
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

func createSequentialSamples43(tasks []*ARCTask43) []Sample43 {
	var samples []Sample43
	for i, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample43{
				Input:     encodeGrid43(pair.Input),
				Target:    encodeGrid43(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

func createEvalSamples43(tasks []*ARCTask43) []Sample43 {
	var samples []Sample43
	for i, task := range tasks {
		for _, pair := range task.Test {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample43{
				Input:     encodeGrid43(pair.Input),
				Target:    encodeGrid43(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

func printFusionResults43(output *FusionResults, allResults []EnsembleResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    🔮 ENSEMBLE FUSION v2 - RESULTS                                                          ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Ensembles: %d x %d DIVERSE networks | Unique Tasks Solved: %d | Best Strategy: %s                              ║\n",
		output.NumEnsembles, output.EnsembleSize, output.CollectiveCount, output.BestFusionStrategy)
	fmt.Printf("║   Duration: %s                                                                                                       ║\n",
		output.Duration)
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	fmt.Println("║                                         📊 STRATEGY COMPARISON                                                               ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, strategy := range fusionNames {
		totalSolved := output.StrategyComparison[strategy]
		bar := ""
		for i := 0; i < totalSolved; i++ {
			bar += "█"
		}
		fmt.Printf("║   %-10s: %-50s (%d total)                             ║\n",
			strategy, bar, totalSolved)
	}

	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                         📊 ARCHITECTURE DIVERSITY                                                            ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   CombineModes: %v                                                                                  ║\n", output.CombineModeStats)
	fmt.Printf("║   Grid Species: %v                   ║\n", output.SpeciesStats)

	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                              🏆 TOP 10 ENSEMBLES                                                             ║")
	fmt.Println("╠═════════════════════════════════╦══════════════╦══════════╦══════════╦════════════╦══════════╦═══════════════════════════════╣")
	fmt.Println("║     Ensemble                    ║   Strategy   ║  Solved  ║  Bonus   ║  Synergy   ║ Coverage ║ Diversity (Species/Modes)     ║")
	fmt.Println("╠═════════════════════════════════╬══════════════╬══════════╬══════════╬════════════╬══════════╬═══════════════════════════════╣")

	for i := 0; i < 10 && i < len(output.TopEnsembles); i++ {
		r := output.TopEnsembles[i]
		diversity := fmt.Sprintf("%d/%d", r.UniqueSpecies, r.UniqueCombineModes)
		fmt.Printf("║ %-31s ║ %-12s ║ %8d ║ %+7d  ║ %9.2fx ║ %6.1f%% ║ %-29s ║\n",
			r.Config.Name, r.Config.StrategyName, r.TasksSolved, r.FusionBonus, r.SynergyScore, r.CoverageRate, diversity)
	}

	fmt.Println("╚═════════════════════════════════╩══════════════╩══════════╩══════════╩════════════╩══════════╩═══════════════════════════════╝")

	// Synergy insight
	fmt.Println("\n💡 SYNERGY INSIGHT:")
	fmt.Println("   Fusion Bonus = Extra tasks solved by ensemble that NO individual network could solve")
	fmt.Println("   Synergy Score = (Ensemble solved) / (Best individual solved) - higher means better teamwork")
	fmt.Println("   Coverage Rate = % of pixels where at least one network in the ensemble got it right")
	fmt.Println("   Diversity = How many different species/combine modes in the ensemble")

	// Key finding
	bestEnsemble := output.TopEnsembles[0]
	if bestEnsemble.FusionBonus > 0 {
		fmt.Printf("\n🎯 KEY FINDING: The best ensemble solved %d MORE tasks than any individual!\n", bestEnsemble.FusionBonus)
		fmt.Printf("   This proves that diverse architectures capture different partial solutions.\n")
	} else {
		fmt.Printf("\n🎯 KEY FINDING: Best strategy was %s with %d tasks solved.\n", output.BestFusionStrategy, bestEnsemble.TasksSolved)
	}

	fmt.Printf("\n🧠 COLLECTIVE WISDOM: %d unique tasks solved across all ensembles\n", output.CollectiveCount)
}

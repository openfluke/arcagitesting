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
	"strconv"
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

// ═══════════════════════════════════════════════════════════════════════════════
// UNSUPERVISED OUTPUT CATEGORIZATION - Network Specialization Profiling
// ═══════════════════════════════════════════════════════════════════════════════

// OutputProfile captures WHERE a network performs best on the grid
type OutputProfile struct {
	// Spatial buckets: accuracy in each quadrant/region
	TopLeftAcc     float64 `json:"topLeftAcc"`
	TopRightAcc    float64 `json:"topRightAcc"`
	BottomLeftAcc  float64 `json:"bottomLeftAcc"`
	BottomRightAcc float64 `json:"bottomRightAcc"`
	CenterAcc      float64 `json:"centerAcc"`
	EdgeAcc        float64 `json:"edgeAcc"`

	// Color prediction profile: accuracy per color (0-9)
	ColorAccuracies [10]float64 `json:"colorAccuracies"`

	// Aggregated stats
	OverallAcc float64 `json:"overallAcc"`
}

// NetworkSpecialist extends NetworkState with output profiling
type NetworkSpecialist struct {
	*NetworkState
	Profile        OutputProfile      `json:"profile"`
	ClusterID      int                `json:"clusterId"`
	TaskAffinities map[string]float64 `json:"taskAffinities"`
}

// SpecialistCluster groups networks with similar output profiles
type SpecialistCluster struct {
	ID              int                  `json:"id"`
	Name            string               `json:"name"`
	CentroidProfile OutputProfile        `json:"centroidProfile"`
	Members         []*NetworkSpecialist `json:"-"`
	MemberIDs       []int                `json:"memberIds"`
	BestCombineMode string               `json:"bestCombineMode"`
	TasksSolved     int                  `json:"tasksSolved"`
	SolvedTaskIDs   []string             `json:"solvedTaskIds"`
	Specialty       string               `json:"specialty"` // Human-readable description of what this cluster is good at
}

// ClusterEnsembleResult tracks results from specialist cluster ensembles
type ClusterEnsembleResult struct {
	ClusterID     int      `json:"clusterId"`
	CombineMode   string   `json:"combineMode"`
	TasksSolved   int      `json:"tasksSolved"`
	SolvedTaskIDs []string `json:"solvedTaskIds"`
	AvgAccuracy   float64  `json:"avgAccuracy"`
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2: COMPLEMENTARY MODEL STITCHING - Pixel-Level Analysis
// ═══════════════════════════════════════════════════════════════════════════════

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
	OverlapPixels      int     // Pixels both got right
	UniqueA, UniqueB   int     // Pixels only A or B got right
	CombinedCoverage   int     // Total pixels covered together
	CombinedRate       float64 // Percentage of pixels covered together
	ComplementScore    float64 // (uniqueA + uniqueB) / (overlap + uniqueA + uniqueB) - higher = more complementary
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
	StitchedOutput    []int // The pixel-by-pixel stitched prediction
}

// RegionBucket categorizes pixels by spatial location
type RegionBucket struct {
	Name        string
	PixelMask   []bool // Which pixels belong to this bucket
	TotalPixels int
	Correct     int
	Accuracy    float64
	BestNetwork int // Network that performs best in this region
}

// TaskStitchingSummary summarizes stitching results for a task
type TaskStitchingSummary struct {
	TaskID           string
	TotalPixels      int
	BaselineCoverage float64 // Coverage before stitching (best single network)
	BestPairCoverage float64 // Coverage with best complementary pair
	FullySolved      bool
	TopPairs         []ComplementaryPair
	RegionAnalysis   []RegionBucket
}

// StitchedGrid represents a "Frankenstein" grid output from stitching
type StitchedGrid struct {
	ID               int
	Output           []int   // The stitched prediction (colors 0-9)
	Coverage         float64 // How correct this stitched output is (0-100%)
	PixelCorrectness []bool  // Which pixels are correct
	SourceNetworks   []int   // Which networks contributed
}

// StitchedGridsPerTask holds all stitched grids for a task
type StitchedGridsPerTask struct {
	TaskID       string
	Height       int
	Width        int
	TotalPixels  int
	TargetColors []int           // Ground truth (colors 0-9)
	Grids        []*StitchedGrid // All stitched outputs for this task
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
	fmt.Println("║   🔮 ENSEMBLE FUSION v2 - ARC-AGI-2 Diverse Architecture                                 ║")
	fmt.Println("║                                                                                          ║")
	fmt.Printf("║   Spawning %d ensembles of %d DIVERSE networks with %d workers...                       ║\n", NumEnsembles, EnsembleSize, numWorkers)
	fmt.Println("║   Testing: CombineModes (concat/add/avg/grid_scatter), Brain types, Grid shapes         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	// Load ARC-AGI training data
	//trainTasks, err := loadARCTasks43("../ARC-AGI2/data/training", 1000)

	//ARC-AGI/data/evaluation

	trainTasks, err := loadARCTasks43("../ARC-AGI/data/training", 400)

	if err != nil {
		fmt.Printf("❌ Failed to load training tasks: %v\n", err)
		return
	}

	// Load ARC-AGI evaluation data
	//evalTasks, err := loadARCTasks43("../ARC-AGI2/data/evaluation", 120)
	evalTasks, err := loadARCTasks43("../ARC-AGI/data/evaluation", 400)
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
	// OR Load from cache if available
	// ===========================================================================
	totalNetworks := NumEnsembles * EnsembleSize
	const modelCacheFile = "trained_networks.json"
	const configCacheFile = "trained_configs.json"

	// Track overall timing and stats
	startTime := time.Now()
	speciesCount := make(map[string]int)
	combineModeCount := make(map[string]int)

	// Check if cached models exist
	networks := make([]*NetworkState, totalNetworks)
	var configs []AgentConfig43
	cacheLoaded := false

	if _, err := os.Stat(modelCacheFile); err == nil {
		fmt.Printf("\n📂 Found cached models at %s, attempting to load...\n", modelCacheFile)

		// Load the model bundle
		bundle, loadErr := nn.LoadBundle(modelCacheFile)
		if loadErr == nil && len(bundle.Models) == totalNetworks {
			// Load configs from config cache
			configData, cfgErr := os.ReadFile(configCacheFile)
			if cfgErr == nil {
				var loadedConfigs []AgentConfig43
				if json.Unmarshal(configData, &loadedConfigs) == nil && len(loadedConfigs) == totalNetworks {
					configs = loadedConfigs
					fmt.Printf("   ✅ Loaded %d configs from cache\n", len(configs))

					// Reconstruct NetworkState from SavedModels
					for i, savedModel := range bundle.Models {
						net, netErr := nn.DeserializeModel(savedModel)
						if netErr != nil {
							fmt.Printf("   ⚠️  Failed to deserialize network %d: %v\n", i, netErr)
							cacheLoaded = false
							break
						}

						state := net.InitStepState(InputSize43)
						ts := nn.NewTweenState(net, nil)
						ts.Config.LinkBudgetScale = configs[i].BudgetScale
						ts.Config.UseChainRule = true

						networks[i] = &NetworkState{
							Config:          configs[i],
							Network:         net,
							State:           state,
							Tween:           ts,
							PerTaskAccuracy: make(map[string]float64),
							OverallAccuracy: 0, // Will be recalculated if needed
						}
						cacheLoaded = true
					}

					if cacheLoaded {
						fmt.Printf("   ✅ Successfully loaded %d trained networks from cache!\n", totalNetworks)
						// Compute stats from loaded configs
						for _, c := range configs {
							speciesCount[c.Species]++
							combineModeCount[c.CombineModeN]++
						}
					}
				}
			}
		}

		if !cacheLoaded {
			fmt.Println("   ⚠️  Cache load failed, will train fresh networks")
		}
	}

	if !cacheLoaded {
		// Generate fresh configs and train
		fmt.Printf("\n🧠 Phase 1: Training %d DIVERSE networks...\n", totalNetworks)

		configs = generateDiverseConfigs(totalNetworks)

		// Print diversity stats
		for _, c := range configs {
			speciesCount[c.Species]++
			combineModeCount[c.CombineModeN]++
		}
		fmt.Printf("   📊 Species distribution: %v\n", speciesCount)
		fmt.Printf("   📊 CombineMode distribution: %v\n", combineModeCount)

		startTime = time.Now()

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
		fmt.Printf("✅ Phase 1 complete: Trained %d diverse networks in %s\n", totalNetworks, trainingTime)

		// Save trained networks to cache
		fmt.Printf("\n💾 Saving trained networks to %s...\n", modelCacheFile)
		modelMap := make(map[string]*nn.Network)
		for i, ns := range networks {
			modelMap[fmt.Sprintf("net_%d", i)] = ns.Network
		}
		if saveErr := nn.SaveBundle(modelCacheFile, modelMap); saveErr != nil {
			fmt.Printf("   ⚠️  Failed to save model bundle: %v\n", saveErr)
		} else {
			fmt.Printf("   ✅ Saved %d networks to cache\n", totalNetworks)
		}

		// Save configs separately
		configData, _ := json.MarshalIndent(configs, "", "  ")
		if cfgErr := os.WriteFile(configCacheFile, configData, 0644); cfgErr != nil {
			fmt.Printf("   ⚠️  Failed to save configs: %v\n", cfgErr)
		} else {
			fmt.Printf("   ✅ Saved configs to %s\n", configCacheFile)
		}
	}

	fmt.Println()

	// ===========================================================================
	// PHASE 1.5: Unsupervised Network Profiling and Clustering
	// ===========================================================================
	fmt.Println("🧠 Phase 1.5: Profiling networks and clustering by output specialization...")

	// Create NetworkSpecialist wrappers with output profiles
	specialists := make([]*NetworkSpecialist, totalNetworks)
	for i, ns := range networks {
		specialists[i] = &NetworkSpecialist{
			NetworkState:   ns,
			ClusterID:      -1,
			TaskAffinities: make(map[string]float64),
		}
		specialists[i].Profile = analyzeOutputProfile(ns, trainTasks)
	}

	// Cluster networks by their output profiles (5 clusters)
	numClusters := 5
	clusters := clusterNetworksByProfile(specialists, numClusters)

	// Print cluster analysis
	fmt.Printf("   📊 Created %d specialist clusters:\n", len(clusters))
	for _, cluster := range clusters {
		if len(cluster.Members) > 0 {
			fmt.Printf("      Cluster %d: %d networks | Specialty: %s | Avg Accuracy: %.1f%%\n",
				cluster.ID, len(cluster.Members), cluster.Specialty,
				cluster.CentroidProfile.OverallAcc*100)
		}
	}

	// ===========================================================================
	// PHASE 1.6: Evaluate Specialist Clusters
	// ===========================================================================
	fmt.Println("\n🎯 Phase 1.6: Evaluating specialist cluster ensembles...")

	clusterResults := make([][]ClusterEnsembleResult, len(clusters))
	var wg sync.WaitGroup
	// Limit concurrency to avoid resource exhaustion
	evalConcurrency := runtime.NumCPU()
	sem := make(chan struct{}, evalConcurrency)

	for i, cluster := range clusters {
		if len(cluster.Members) > 0 {
			wg.Add(1)
			go func(i int, cluster *SpecialistCluster) {
				defer wg.Done()
				sem <- struct{}{} // Acquire semaphore
				clusterResults[i] = evaluateSpecialistCluster(cluster, evalTasks)
				<-sem // Release semaphore
			}(i, cluster)
		}
	}
	wg.Wait()

	// Create and evaluate monolithic ensemble
	monolithicResult := createMonolithicEnsemble(clusters, evalTasks)

	// Display specialist cluster results
	printSpecialistResults(clusters, clusterResults, monolithicResult)

	// COLLECT ALL SOLVED TASKS FROM SPECIALIST CLUSTERS
	specialistCollective := make(map[string]bool)
	for _, cluster := range clusters {
		for _, taskID := range cluster.SolvedTaskIDs {
			specialistCollective[taskID] = true
		}
	}
	// Also add monolithic results
	if monolithicResult != nil {
		for _, taskID := range monolithicResult.SolvedTaskIDs {
			specialistCollective[taskID] = true
		}
	}

	fmt.Printf("\n📈 SPECIALIST APPROACH: Solved %d/%d unique tasks (%.1f%% of total)\n",
		len(specialistCollective), len(evalTasks), float64(len(specialistCollective))*100/float64(len(evalTasks)))

	// List the solved task IDs if there are any
	if len(specialistCollective) > 0 && len(specialistCollective) <= 50 {
		solvedList := make([]string, 0, len(specialistCollective))
		for taskID := range specialistCollective {
			solvedList = append(solvedList, taskID)
		}
		sort.Strings(solvedList)
		fmt.Printf("   ✅ Solved tasks: %v\n", solvedList)
	}

	// ===========================================================================
	// PHASE 2: Complementary Model Stitching (Skip to the good stuff!)
	// ===========================================================================
	collectiveTasks := make(map[string]bool)
	// Pre-populate with specialist-solved tasks
	for taskID := range specialistCollective {
		collectiveTasks[taskID] = true
	}

	strategyStats := make(map[string]int)

	// ===========================================================================
	// PHASE 2.5: Complementary Model Stitching (NEW!)
	// ===========================================================================
	stitchingSolved, stitchingSummaries, _ := phase2ComplementaryStitchingWithGrids(specialists, evalTasks, collectiveTasks)

	// Add newly solved tasks from stitching
	for _, taskID := range stitchingSolved {
		collectiveTasks[taskID] = true
	}
	if len(stitchingSolved) > 0 {
		strategyStats["Stitching"] = len(stitchingSolved)
	}

	// Calculate overall stitching metrics
	stitchingImproved := 0
	for _, s := range stitchingSummaries {
		if s.BestPairCoverage > s.BaselineCoverage {
			stitchingImproved++
		}
	}
	_ = stitchingImproved // Used in output

	// Results (no cross-cluster fusion, just stitching stats)
	var results []EnsembleResult

	totalTime := time.Since(startTime)

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

	// Build output
	output := &FusionResults{
		NumEnsembles:       NumEnsembles,
		EnsembleSize:       EnsembleSize,
		TopEnsembles:       results,
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
	os.WriteFile("ensemble_fusion_results_2.json", data, 0644)
	fmt.Println("\n✅ Results saved to ensemble_fusion_results_2.json")

	// Print results
	printFusionResults43(output, results)
}

// attemptNeuralStitching trains a Gate network to learn the stitching mask
// Replicates the "Odds" experiment logic where we learn to gate experts
func attemptNeuralStitching(specialists []*NetworkSpecialist, task *ARCTask43, pair ComplementaryPair) float64 {
	// 1. Prepare Training Data
	gateInputSize := InputSize43
	gateOutputSize := InputSize43
	gateNet := nn.NewNetwork(gateInputSize, 1, 1, 1)
	gateNet.SetLayer(0, 0, 0, nn.InitDenseLayer(gateInputSize, gateOutputSize, nn.ActivationSigmoid))
	
	learningRate := float32(0.05)

	// Pre-compute Expert Outputs & Targets
	type TrainTuple struct {
		Input      []float32
		GateTarget []float32
	}
	var trainData []TrainTuple

	for _, sample := range task.Train {
		input := encodeGrid43(sample.Input)
		target := encodeGrid43(sample.Output) // flatten target

		outA, _ := specialists[pair.NetworkA].Network.ForwardCPU(input)
		outB, _ := specialists[pair.NetworkB].Network.ForwardCPU(input)

		gateTarget := make([]float32, gateOutputSize)
		for i := 0; i < gateOutputSize && i < len(target); i++ {
			errA := math.Abs(float64(outA[i] - target[i]))
			errB := math.Abs(float64(outB[i] - target[i]))
			if errA < errB {
				gateTarget[i] = 1.0 // Prefer A
			} else if errB < errA {
				gateTarget[i] = 0.0 // Prefer B
			} else {
				gateTarget[i] = 0.5 // Neutral
			}
		}
		trainData = append(trainData, TrainTuple{Input: input, GateTarget: gateTarget})
	}

	// 2. Train Until 100% Accuracy (or max epochs)
	maxEpochs := 1000
	for epoch := 0; epoch < maxEpochs; epoch++ {
		correctPixels := 0
		totalPixels := 0
		
		for _, td := range trainData {
			// Forward
			gateOut, _ := gateNet.ForwardCPU(td.Input)
			
			// Backward & Stats
			grad := make([]float32, len(gateOut))
			for i := range gateOut {
				errorVal := gateOut[i] - td.GateTarget[i]
				grad[i] = errorVal
				
				// Training Accuracy Check:
				// If target is 1.0, prediction > 0.5 is correct.
				// If target is 0.0, prediction < 0.5 is correct.
				isCorrect := false
				if td.GateTarget[i] > 0.9 && gateOut[i] > 0.5 { 
					isCorrect = true 
				} else if td.GateTarget[i] < 0.1 && gateOut[i] < 0.5 { 
					isCorrect = true 
				} else if td.GateTarget[i] >= 0.1 && td.GateTarget[i] <= 0.9 { 
					isCorrect = true 
				} 
				
				if isCorrect { correctPixels++ }
				totalPixels++
			}
			
			// Update Weights
			layer := gateNet.GetLayer(0,0,0)
			for o := 0; o < gateOutputSize; o++ {
				g := grad[o]
				actDeriv := gateOut[o] * (1.0 - gateOut[o])
				delta := g * actDeriv
				layer.Bias[o] -= learningRate * delta
				startIdx := o * gateInputSize
				for i := 0; i < gateInputSize; i++ {
					if td.Input[i] != 0 {
						layer.Kernel[startIdx + i] -= learningRate * delta * td.Input[i]
					}
				}
			}
		}
		
		if totalPixels > 0 {
			trainAcc := float64(correctPixels) / float64(totalPixels) * 100.0
			if trainAcc > 99.9 {
				break
			}
		}
	}

	// 3. Test on Test Set
	if len(task.Test) == 0 { return 0.0 }
	testPair := task.Test[0]
	input := encodeGrid43(testPair.Input)
	target := encodeGrid43(testPair.Output) 
	
	gateOut, _ := gateNet.ForwardCPU(input)
	outA, _ := specialists[pair.NetworkA].Network.ForwardCPU(input)
	outB, _ := specialists[pair.NetworkB].Network.ForwardCPU(input)
	
	pixelsCorrect := 0
	totalPixelsTest := len(target)
	
	for i := 0; i < totalPixelsTest; i++ {
		valA := outA[i]
		valB := outB[i]
		
		var finalVal float32
		if gateOut[i] > 0.5 {
			finalVal = valA
		} else {
			finalVal = valB
		}
		
		predColor := clampInt43(int(math.Round(float64(finalVal)*9.0)), 0, 9)
		targetC := int(math.Round(float64(target[i]) * 9.0))
		
		if predColor == targetC {
			pixelsCorrect++
		}
	}
	
	return float64(pixelsCorrect) / float64(totalPixelsTest) * 100.0
}

func (p GridPair43) Width() []int {
	if len(p.Output) > 0 {
		w := make([]int, len(p.Output))
		for i := range w { w[i] = len(p.Output[i]) }
		return w
	}
	return []int{0}
}

// generateDiverseConfigs creates architecturally diverse network configurations
// Now delegates to nn.GenerateDiverseConfigs and converts back to local types
func generateDiverseConfigs(count int) []AgentConfig43 {
	// Use the new nn.GenerateDiverseConfigs API with default options
	nnConfigs := nn.GenerateDiverseConfigs(count, nil)

	// Convert nn.ArchConfig to local AgentConfig43 for backward compatibility
	configs := make([]AgentConfig43, count)
	for i, nc := range nnConfigs {
		// Convert nn.BrainType to local BrainType43
		brains := make([]BrainType43, len(nc.Brains))
		for j, bt := range nc.Brains {
			brains[j] = BrainType43(bt)
		}

		// Convert nn.ActivationType to local ActivationType43
		activation := ActivationType43(nc.Activation)

		// Convert combine mode string to local CombineModeType43
		var combineMode CombineModeType43
		switch nc.CombineMode {
		case "concat":
			combineMode = CombineConcat43
		case "add":
			combineMode = CombineAdd43
		case "avg":
			combineMode = CombineAvg43
		case "grid_scatter":
			combineMode = CombineGridScatter43
		default:
			combineMode = CombineAvg43
		}

		configs[i] = AgentConfig43{
			ID:           nc.ID,
			Name:         nc.Name,
			Species:      nc.Species,
			MutationStr:  nc.MutationStr,
			GridRows:     nc.GridRows,
			GridCols:     nc.GridCols,
			NumBrains:    nc.NumBrains,
			DModel:       nc.DModel,
			NumHeads:     nc.NumHeads,
			LearningRate: nc.LearningRate,
			BudgetScale:  nc.BudgetScale,
			Activation:   activation,
			ActivationN:  activationNames43[activation],
			CombineMode:  combineMode,
			CombineModeN: nc.CombineMode,
			Brains:       brains,
			BrainNames:   nc.BrainNames,
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
// Now delegates to nn.BuildDiverseNetwork after converting local config to nn.ArchConfig
func createDiverseNetwork(config AgentConfig43) *nn.Network {
	// Convert local AgentConfig43 to nn.ArchConfig
	nnBrains := make([]nn.BrainType, len(config.Brains))
	for i, b := range config.Brains {
		nnBrains[i] = nn.BrainType(b)
	}

	archConfig := nn.ArchConfig{
		ID:           config.ID,
		Name:         config.Name,
		Species:      config.Species,
		MutationStr:  config.MutationStr,
		GridRows:     config.GridRows,
		GridCols:     config.GridCols,
		NumBrains:    config.NumBrains,
		DModel:       config.DModel,
		NumHeads:     config.NumHeads,
		LearningRate: config.LearningRate,
		BudgetScale:  config.BudgetScale,
		Activation:   nn.ActivationType(config.Activation),
		CombineMode:  config.CombineModeN,
		Brains:       nnBrains,
		BrainNames:   config.BrainNames,
		InitScale:    InitScale43,
	}

	return nn.BuildDiverseNetwork(archConfig, InputSize43)
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

// ============================================================================
// PHASE 2: COMPLEMENTARY MODEL STITCHING FUNCTIONS
// ============================================================================

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
	input := encodeGrid43(pair.Input)
	for netIdx, specialist := range specialists {
		output, _ := specialist.Network.ForwardCPU(input)

		analysis.PixelCorrectness[netIdx] = make([]bool, totalPixels)
		analysis.NetworkOutputs[netIdx] = make([]int, totalPixels)

		for r := 0; r < height; r++ {
			for c := 0; c < width; c++ {
				pixelIdx := r*width + c
				idx := r*MaxGridSize43 + c

				if idx < len(output) {
					// Convert output to color (0-9)
					predColor := clampInt43(int(math.Round(float64(output[idx])*9.0)), 0, 9)
					analysis.NetworkOutputs[netIdx][pixelIdx] = predColor

					// Check correctness
					targetColor := analysis.TargetColors[pixelIdx]
					if predColor == targetColor {
						analysis.PixelCorrectness[netIdx][pixelIdx] = true
						analysis.CoverageMap[pixelIdx]++

						// Track best network (first one correct)
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

	// Convert to nn.ModelPerformance
	performances := make([]nn.ModelPerformance, analysis.NumNetworks)
	for i := 0; i < analysis.NumNetworks; i++ {
		performances[i] = nn.ModelPerformance{
			ModelID: fmt.Sprintf("%d", i),
			Mask:    analysis.PixelCorrectness[i],
		}
	}

	// Use nn library to find matches
	matches := nn.FindComplementaryMatches(performances, 0.0)

	// Convert back to local ComplementaryPair
	var pairs []ComplementaryPair
	for _, m := range matches {
		idxA, _ := strconv.Atoi(m.ModelA)
		idxB, _ := strconv.Atoi(m.ModelB)
		
		// Recalculate discrete counts since nn only gives rates
		// (or we could trust rates * total, but let's be precise for consistency)
		var overlap, uniqueA, uniqueB int
		for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
			correctA := analysis.PixelCorrectness[idxA][pixelIdx]
			correctB := analysis.PixelCorrectness[idxB][pixelIdx]

			if correctA && correctB {
				overlap++
			} else if correctA {
				uniqueA++
			} else if correctB {
				uniqueB++
			}
		}
		
		combinedCoverage := overlap + uniqueA + uniqueB
		complementScore := 0.0
		if combinedCoverage > 0 {
			complementScore = float64(uniqueA+uniqueB) / float64(combinedCoverage)
		}

		pairs = append(pairs, ComplementaryPair{
			NetworkA:         idxA,
			NetworkB:         idxB,
			OverlapPixels:    overlap,
			UniqueA:          uniqueA,
			UniqueB:          uniqueB,
			CombinedCoverage: combinedCoverage,
			CombinedRate:     m.Coverage,
			ComplementScore:  complementScore, // nn.EnsembleMatch doesn't compute this exactly the same way
		})
	}

	// Return top N pairs
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

	// Calculate original coverage for each network
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

	// Stitch: take each pixel from the correct network
	correctStitched := 0
	for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
		correctA := analysis.PixelCorrectness[pair.NetworkA][pixelIdx]
		correctB := analysis.PixelCorrectness[pair.NetworkB][pixelIdx]

		var chosenColor int
		if correctA && correctB {
			// Both correct - use A (arbitrary tie-breaker)
			chosenColor = analysis.NetworkOutputs[pair.NetworkA][pixelIdx]
			correctStitched++
		} else if correctA {
			// Only A correct
			chosenColor = analysis.NetworkOutputs[pair.NetworkA][pixelIdx]
			correctStitched++
			result.PixelsImproved++
		} else if correctB {
			// Only B correct
			chosenColor = analysis.NetworkOutputs[pair.NetworkB][pixelIdx]
			correctStitched++
			result.PixelsImproved++
		} else {
			// Neither correct - use voting between A and B
			colorA := analysis.NetworkOutputs[pair.NetworkA][pixelIdx]
			colorB := analysis.NetworkOutputs[pair.NetworkB][pixelIdx]
			if colorA == colorB {
				chosenColor = colorA
			} else {
				// Pick randomly or use first one
				chosenColor = colorA
			}
		}
		result.StitchedOutput[pixelIdx] = chosenColor

		// Check if correct
		if chosenColor == analysis.TargetColors[pixelIdx] {
			// Already counted above
		}
	}

	if analysis.TotalPixels > 0 {
		result.StitchedCoverage = float64(correctStitched) / float64(analysis.TotalPixels)
	}
	result.FullySolved = correctStitched == analysis.TotalPixels

	return result
}

// stitchFromBestPerPixel creates output using best network for each pixel
func stitchFromBestPerPixel(analysis *PixelAnalysis, specialists []*NetworkSpecialist) *StitchResult {
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
		if bestNet >= 0 && bestNet < len(specialists) {
			// Use the network that got this pixel correct
			result.StitchedOutput[pixelIdx] = analysis.NetworkOutputs[bestNet][pixelIdx]
			correctStitched++
		} else {
			// No network got it right - use voting across all
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

// buildRegionBuckets creates spatial region buckets for analysis
func buildRegionBuckets(analysis *PixelAnalysis) []RegionBucket {
	if analysis == nil {
		return nil
	}

	height, width := analysis.Height, analysis.Width

	buckets := []RegionBucket{
		{Name: "top-left", PixelMask: make([]bool, analysis.TotalPixels)},
		{Name: "top-right", PixelMask: make([]bool, analysis.TotalPixels)},
		{Name: "bottom-left", PixelMask: make([]bool, analysis.TotalPixels)},
		{Name: "bottom-right", PixelMask: make([]bool, analysis.TotalPixels)},
		{Name: "center", PixelMask: make([]bool, analysis.TotalPixels)},
		{Name: "edge", PixelMask: make([]bool, analysis.TotalPixels)},
	}

	// Assign pixels to buckets
	for r := 0; r < height; r++ {
		for c := 0; c < width; c++ {
			pixelIdx := r*width + c
			midY, midX := height/2, width/2

			// Quadrant buckets
			if r < midY && c < midX {
				buckets[0].PixelMask[pixelIdx] = true
				buckets[0].TotalPixels++
			} else if r < midY && c >= midX {
				buckets[1].PixelMask[pixelIdx] = true
				buckets[1].TotalPixels++
			} else if r >= midY && c < midX {
				buckets[2].PixelMask[pixelIdx] = true
				buckets[2].TotalPixels++
			} else {
				buckets[3].PixelMask[pixelIdx] = true
				buckets[3].TotalPixels++
			}

			// Center vs edge
			marginY, marginX := height/4, width/4
			if r >= marginY && r < height-marginY && c >= marginX && c < width-marginX {
				buckets[4].PixelMask[pixelIdx] = true
				buckets[4].TotalPixels++
			} else {
				buckets[5].PixelMask[pixelIdx] = true
				buckets[5].TotalPixels++
			}
		}
	}

	// Calculate accuracy per bucket (best coverage from any network)
	for bi := range buckets {
		for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
			if buckets[bi].PixelMask[pixelIdx] && analysis.CoverageMap[pixelIdx] > 0 {
				buckets[bi].Correct++
			}
		}
		if buckets[bi].TotalPixels > 0 {
			buckets[bi].Accuracy = float64(buckets[bi].Correct) / float64(buckets[bi].TotalPixels) * 100
		}
	}

	return buckets
}

// phase2ComplementaryStitchingWithGrids runs Phase 2 and returns stitched grids for Phase 3
func phase2ComplementaryStitchingWithGrids(specialists []*NetworkSpecialist, evalTasks []*ARCTask43, alreadySolved map[string]bool) ([]string, []TaskStitchingSummary, map[string]*StitchedGridsPerTask) {
	fmt.Println("\n🔮 Phase 2: Complementary Model Stitching...")
	fmt.Println("   Strategy: Find pairs of models that complement each other pixel-by-pixel")
	fmt.Println("   Goal: Stitch together partial solutions to solve more tasks")

	var newlySolved []string
	var summaries []TaskStitchingSummary
	allStitchedGrids := make(map[string]*StitchedGridsPerTask)
	
	// Track Neural Results
	type NeuralResult struct {
		Coverage    float64
		Solved      bool
		BaseManual  float64
	}
	neuralResults := make(map[string]NeuralResult)

	// Track progress
	unsolvedCount := 0
	for _, task := range evalTasks {
		if !alreadySolved[task.ID] {
			unsolvedCount++
		}
	}
	fmt.Printf("   📊 Analyzing %d unsolved tasks...\n\n", unsolvedCount)

	// Analyze each unsolved task
	tasksAnalyzed := 0
	for _, task := range evalTasks {
		if alreadySolved[task.ID] {
			continue
		}

		tasksAnalyzed++
		analysis := analyzePixelCorrectness(specialists, task)
		if analysis == nil {
			continue
		}

		// Find complementary pairs
		pairs := findComplementaryPairs(analysis, 5)

		// Try stitching with each pair
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
		bestPerPixelResult := stitchFromBestPerPixel(analysis, specialists)

		// Check which approach is better
		finalResult := bestResult
		if bestPerPixelResult != nil && (finalResult == nil || bestPerPixelResult.StitchedCoverage > finalResult.StitchedCoverage) {
			finalResult = bestPerPixelResult
		}

		// Build region analysis
		regionBuckets := buildRegionBuckets(analysis)

		// Create summary
		summary := TaskStitchingSummary{
			TaskID:           task.ID,
			TotalPixels:      analysis.TotalPixels,
			BaselineCoverage: 0,
			BestPairCoverage: 0,
			FullySolved:      false,
			TopPairs:         pairs,
			RegionAnalysis:   regionBuckets,
		}

		// Calculate baseline (best single network)
		bestSingleCoverage := 0
		for netIdx := 0; netIdx < analysis.NumNetworks; netIdx++ {
			correct := 0
			for pixelIdx := 0; pixelIdx < analysis.TotalPixels; pixelIdx++ {
				if analysis.PixelCorrectness[netIdx][pixelIdx] {
					correct++
				}
			}
			if correct > bestSingleCoverage {
				bestSingleCoverage = correct
			}
		}
		if analysis.TotalPixels > 0 {
			summary.BaselineCoverage = float64(bestSingleCoverage) / float64(analysis.TotalPixels) * 100
		}

		if finalResult != nil {
			summary.BestPairCoverage = finalResult.StitchedCoverage * 100
			summary.FullySolved = finalResult.FullySolved

			if finalResult.FullySolved {
				newlySolved = append(newlySolved, task.ID)
				fmt.Printf("   ✅ SOLVED via stitching: %s (coverage: %.1f%% → 100%%)\n", task.ID, summary.BaselineCoverage)
			}
			
			// NEURAL STITCHING EXPERIMENT (Comparison)
			// Only run if we have a valid pair (NetworkA != NetworkB, to avoid BestPerPixel "0+0" results)
			// BestPerPixel returns Pair{0,0} or {-1,-1} effectively.
			pair := finalResult.PairUsed
			if pair.NetworkA != pair.NetworkB && pair.NetworkA >= 0 && pair.NetworkB >= 0 {
				neuralCoverage := attemptNeuralStitching(specialists, task, pair)
				diff := neuralCoverage - summary.BestPairCoverage
				
				icon := "🟦"
				if diff > 1.0 { icon = "✅" } 
				if diff < -1.0 { icon = "🔻" }
				
				fmt.Printf("      🧠 Neural vs Manual: %.1f%% vs %.1f%% (%s %+.1f%%) [Pair %d+%d]\n", 
					neuralCoverage, summary.BestPairCoverage, icon, diff, pair.NetworkA, pair.NetworkB)
				
				if neuralCoverage > 99.9 && !summary.FullySolved {
					fmt.Printf("      🎉 NEURAL STITCHING SOLVED IT (where manual failed)!\n")
				}
				
				neuralResults[task.ID] = NeuralResult{
					Coverage:   neuralCoverage,
					Solved:     neuralCoverage > 99.9,
					BaseManual: summary.BestPairCoverage,
				}
			}
		}

		summaries = append(summaries, summary)

		// Print progress every 20 tasks
		if tasksAnalyzed%20 == 0 {
			fmt.Printf("   🔄 Analyzed %d/%d unsolved tasks...\n", tasksAnalyzed, unsolvedCount)
		}
	}

	// Print detailed results
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    🧩 COMPLEMENTARY STITCHING RESULTS                                                        ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	// Summary statistics for Neural
	fmt.Println("\n╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║ 🧠 NEURAL STITCHING SUMMARY                              ║")
	fmt.Println("╠══════════╦══════════╦════════════════╦═══════════════════╣")
	fmt.Println("║ Task ID  | Manual   | Neural         | Status            ║")
	fmt.Println("╠══════════╬══════════╬════════════════╬═══════════════════╣")
	
	neuralSolvedCount := 0
	for taskID, res := range neuralResults {
		status := " "
		if res.Solved { 
			status = "SOLVED!"
			neuralSolvedCount++
		} else if res.Coverage > res.BaseManual {
			status = "Better"
		} else if res.Coverage < res.BaseManual {
			status = "Worse"
		}
		
		fmt.Printf("║ %-8s | %6.1f%%  | %6.1f%%        | %-17s ║\n", 
			taskID[:min(8, len(taskID))], res.BaseManual, res.Coverage, status)
	}
	fmt.Println("╚══════════╩══════════╩════════════════╩═══════════════════╝")
	fmt.Printf("🧠 Neural Solved Total: %d tasks\n\n", neuralSolvedCount)

	// Summary statistics
	totalImproved := 0
	avgImprovement := 0.0
	for _, s := range summaries {
		improvement := s.BestPairCoverage - s.BaselineCoverage
		if improvement > 0 {
			totalImproved++
			avgImprovement += improvement
		}
	}
	if totalImproved > 0 {
		avgImprovement /= float64(totalImproved)
	}

	fmt.Printf("║   Tasks Analyzed: %d | Newly Solved: %d | Tasks with Improved Coverage: %d                          ║\n",
		len(summaries), len(newlySolved), totalImproved)
	fmt.Printf("║   Average Coverage Improvement: +%.1f%% (when improved) | Grids for Phase 3: %d tasks                  ║\n",
		avgImprovement, len(allStitchedGrids))
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	// Show top improvements
	sort.Slice(summaries, func(i, j int) bool {
		impI := summaries[i].BestPairCoverage - summaries[i].BaselineCoverage
		impJ := summaries[j].BestPairCoverage - summaries[j].BaselineCoverage
		return impI > impJ
	})

	fmt.Println("║                                    📈 TOP 10 COVERAGE IMPROVEMENTS                                                           ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║   Task ID         | Baseline | Stitched | Improvement | Fully Solved | Region: TL / TR / BL / BR / Center / Edge           ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for i := 0; i < 10 && i < len(summaries); i++ {
		s := summaries[i]
		improvement := s.BestPairCoverage - s.BaselineCoverage
		solved := "  No"
		if s.FullySolved {
			solved = " Yes"
		}

		regionStr := ""
		if len(s.RegionAnalysis) >= 6 {
			regionStr = fmt.Sprintf("%.0f%% / %.0f%% / %.0f%% / %.0f%% / %.0f%% / %.0f%%",
				s.RegionAnalysis[0].Accuracy, s.RegionAnalysis[1].Accuracy,
				s.RegionAnalysis[2].Accuracy, s.RegionAnalysis[3].Accuracy,
				s.RegionAnalysis[4].Accuracy, s.RegionAnalysis[5].Accuracy)
		}

		fmt.Printf("║   %-15s | %6.1f%% | %7.1f%% |    +%5.1f%% |     %s  | %-38s ║\n",
			s.TaskID[:min(15, len(s.TaskID))], s.BaselineCoverage, s.BestPairCoverage, improvement, solved, regionStr)
	}

	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	if len(newlySolved) > 0 {
		fmt.Printf("\n🎉 Phase 2 Stitching solved %d NEW tasks: %v\n", len(newlySolved), newlySolved)
	} else {
		fmt.Println("\n💡 No new tasks fully solved via stitching, but coverage improved for many tasks.")
		fmt.Println("   Consider: training more diverse models or increasing ensemble size.")
	}

	return newlySolved, summaries, allStitchedGrids
}

// ============================================================================
// PHASE 3: RECURSIVE STITCHING - Stitch the Stitches (N-way!)
// ============================================================================
// Take the Frankenstein grids from Phase 2 AND elite networks, combine them ALL
// to solve tasks with composed logic (e.g., "Move Red" + "Recolor Blue" + "Crop Edge")
// Key insight: Don't just do pairs - do N-way stitching (best pixel from ANY source)

const MaxRecursiveDepth = 4

// phase3RecursiveStitching takes stitched grids from Phase 2 and stitches them together
// Now also includes elite networks from clusters for maximum coverage
func phase3RecursiveStitching(stitchedGrids map[string]*StitchedGridsPerTask, evalTasks []*ARCTask43, alreadySolved map[string]bool) []string {
	fmt.Println("\n🔄 Phase 3: Recursive Stitching (N-Way Fusion)...")
	fmt.Println("   Strategy: Combine ALL Phase 2 grids + raw network outputs (best pixel from ANY)")
	fmt.Println("   Goal: Crack tasks where logic is composed of 3+ distinct parts")
	fmt.Printf("   Depth: Up to %d layers of N-way stitching\n", MaxRecursiveDepth)
	fmt.Printf("   📊 Processing %d tasks with stitched grids\n\n", len(stitchedGrids))

	var newlySolved []string
	tasksProcessed := 0
	tasksImproved := 0

	for taskID, taskGrids := range stitchedGrids {
		if alreadySolved[taskID] {
			continue
		}
		if len(taskGrids.Grids) < 2 {
			continue
		}

		tasksProcessed++

		// Find the best initial coverage (from Phase 2)
		initialBest := 0.0
		for _, g := range taskGrids.Grids {
			if g.Coverage > initialBest {
				initialBest = g.Coverage
			}
		}

		// N-WAY STITCHING: Take the best pixel from ANY grid
		nWayCoverage, nWaySolved := nWayStitchAllGrids(taskGrids)

		if nWaySolved {
			newlySolved = append(newlySolved, taskID)
			fmt.Printf("   ✅ SOLVED via N-way stitching: %s (%.1f%% → 100%%)\n", taskID, initialBest)
			continue
		}

		// If N-way didn't solve it, try recursive stitching of the new Frankenstein
		if nWayCoverage > initialBest {
			finalCoverage, solved := recursivelyStitchGrids(taskGrids, MaxRecursiveDepth)

			if solved {
				newlySolved = append(newlySolved, taskID)
				fmt.Printf("   ✅ SOLVED via recursive stitching: %s (%.1f%% → 100%%)\n", taskID, initialBest)
			} else if finalCoverage > initialBest {
				tasksImproved++
				if finalCoverage > 95 {
					fmt.Printf("   📈 Almost there: %s (%.1f%% → %.1f%%)\n", taskID, initialBest, finalCoverage)
				}
			}
		} else {
			// Check if recursive helps even without N-way improvement
			finalCoverage, solved := recursivelyStitchGrids(taskGrids, MaxRecursiveDepth)
			if solved {
				newlySolved = append(newlySolved, taskID)
				fmt.Printf("   ✅ SOLVED via recursive stitching: %s (%.1f%% → 100%%)\n", taskID, initialBest)
			} else if finalCoverage > initialBest {
				tasksImproved++
			}
		}

		if tasksProcessed%20 == 0 {
			fmt.Printf("   🔄 Processed %d/%d tasks...\n", tasksProcessed, len(stitchedGrids))
		}
	}

	// Print summary
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    🔄 RECURSIVE STITCHING RESULTS (N-Way)                                                    ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Tasks Processed: %d | Newly Solved: %d | Improved: %d                                                        ║\n",
		tasksProcessed, len(newlySolved), tasksImproved)
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	if len(newlySolved) > 0 {
		fmt.Printf("\n🎉 Phase 3 N-Way Stitching solved %d NEW tasks: %v\n", len(newlySolved), newlySolved)
	} else {
		fmt.Println("\n💡 No new tasks fully solved via recursive stitching.")
		if tasksImproved > 0 {
			fmt.Printf("   But %d tasks had coverage improvements - may need more diverse base models.\n", tasksImproved)
		}
	}

	return newlySolved
}

// nWayStitchAllGrids combines ALL grids at once - taking the best pixel from ANY grid
func nWayStitchAllGrids(taskGrids *StitchedGridsPerTask) (float64, bool) {
	if len(taskGrids.Grids) == 0 {
		return 0, false
	}

	totalPixels := taskGrids.TotalPixels
	stitchedOutput := make([]int, totalPixels)
	correctCount := 0

	// For each pixel, find the FIRST grid that got it correct
	for pixelIdx := 0; pixelIdx < totalPixels; pixelIdx++ {
		foundCorrect := false

		// Check all grids for this pixel
		for _, grid := range taskGrids.Grids {
			if grid.PixelCorrectness[pixelIdx] {
				stitchedOutput[pixelIdx] = grid.Output[pixelIdx]
				correctCount++
				foundCorrect = true
				break
			}
		}

		// If no grid got it right, use the highest-coverage grid's prediction
		if !foundCorrect {
			bestGrid := taskGrids.Grids[0]
			for _, grid := range taskGrids.Grids {
				if grid.Coverage > bestGrid.Coverage {
					bestGrid = grid
				}
			}
			stitchedOutput[pixelIdx] = bestGrid.Output[pixelIdx]
		}
	}

	coverage := float64(correctCount) / float64(totalPixels) * 100
	solved := correctCount == totalPixels

	// If this N-way stitch is better, add it to the grids pool
	if coverage > 0 {
		correctness := make([]bool, totalPixels)
		for pixelIdx := 0; pixelIdx < totalPixels; pixelIdx++ {
			correctness[pixelIdx] = stitchedOutput[pixelIdx] == taskGrids.TargetColors[pixelIdx]
		}

		nWayGrid := &StitchedGrid{
			ID:               len(taskGrids.Grids),
			Output:           stitchedOutput,
			Coverage:         coverage,
			PixelCorrectness: correctness,
			SourceNetworks:   []int{-2}, // -2 indicates N-way stitch
		}
		taskGrids.Grids = append(taskGrids.Grids, nWayGrid)
	}

	return coverage, solved
}

// recursivelyStitchGrids performs multi-layer stitching on the Frankenstein grids
func recursivelyStitchGrids(taskGrids *StitchedGridsPerTask, maxDepth int) (float64, bool) {
	currentGrids := taskGrids.Grids
	bestCoverage := 0.0

	for _, g := range currentGrids {
		if g.Coverage > bestCoverage {
			bestCoverage = g.Coverage
		}
		if g.Coverage >= 100.0 {
			return 100.0, true
		}
	}

	// Recursively stitch for multiple layers
	for depth := 1; depth <= maxDepth; depth++ {
		// First try N-way stitch of current pool
		nWayCoverage, nWaySolved := nWayStitchFromGridSlice(currentGrids, taskGrids.TargetColors, taskGrids.TotalPixels)
		if nWaySolved {
			return 100.0, true
		}
		if nWayCoverage > bestCoverage {
			bestCoverage = nWayCoverage

			// Add the N-way result to the pool
			correctness := make([]bool, taskGrids.TotalPixels)
			stitchedOutput := make([]int, taskGrids.TotalPixels)

			for pixelIdx := 0; pixelIdx < taskGrids.TotalPixels; pixelIdx++ {
				for _, grid := range currentGrids {
					if grid.PixelCorrectness[pixelIdx] {
						stitchedOutput[pixelIdx] = grid.Output[pixelIdx]
						correctness[pixelIdx] = true
						break
					}
				}
				if !correctness[pixelIdx] && len(currentGrids) > 0 {
					stitchedOutput[pixelIdx] = currentGrids[0].Output[pixelIdx]
				}
			}

			currentGrids = append(currentGrids, &StitchedGrid{
				ID:               len(currentGrids),
				Output:           stitchedOutput,
				Coverage:         nWayCoverage,
				PixelCorrectness: correctness,
				SourceNetworks:   []int{-3}, // -3 indicates recursive N-way
			})
		}

		// Also try pairwise stitching to find new complementary combinations
		newGrids := stitchGridsTogether(currentGrids, taskGrids.TargetColors, taskGrids.TotalPixels, len(currentGrids))

		if len(newGrids) == 0 && nWayCoverage <= bestCoverage {
			break // No progress
		}

		// Check new grids for solution
		for _, g := range newGrids {
			if g.Coverage >= 100.0 {
				return 100.0, true
			}
			if g.Coverage > bestCoverage {
				bestCoverage = g.Coverage
			}
		}

		// Add new grids to pool and prune
		currentGrids = append(currentGrids, newGrids...)
		if len(currentGrids) > 50 {
			sort.Slice(currentGrids, func(i, j int) bool {
				return currentGrids[i].Coverage > currentGrids[j].Coverage
			})
			currentGrids = currentGrids[:50]
		}
	}

	return bestCoverage, false
}

// nWayStitchFromGridSlice does N-way stitching from a slice of grids
func nWayStitchFromGridSlice(grids []*StitchedGrid, targetColors []int, totalPixels int) (float64, bool) {
	if len(grids) == 0 {
		return 0, false
	}

	correctCount := 0
	for pixelIdx := 0; pixelIdx < totalPixels; pixelIdx++ {
		for _, grid := range grids {
			if grid.PixelCorrectness[pixelIdx] {
				correctCount++
				break
			}
		}
	}

	coverage := float64(correctCount) / float64(totalPixels) * 100
	return coverage, correctCount == totalPixels
}

// stitchGridsTogether creates new grids by stitching existing Frankenstein grids together
func stitchGridsTogether(grids []*StitchedGrid, targetColors []int, totalPixels int, baseID int) []*StitchedGrid {
	if len(grids) < 2 {
		return nil
	}

	var newGrids []*StitchedGrid
	seenCombos := make(map[string]bool)

	// Try all pairs
	for i := 0; i < len(grids); i++ {
		for j := i + 1; j < len(grids); j++ {
			gridA := grids[i]
			gridB := grids[j]

			// Skip if already tried this combo
			key := fmt.Sprintf("%d-%d", gridA.ID, gridB.ID)
			if seenCombos[key] {
				continue
			}
			seenCombos[key] = true

			// Check complementarity
			uniqueA, uniqueB := 0, 0
			for pixelIdx := 0; pixelIdx < totalPixels; pixelIdx++ {
				correctA := gridA.PixelCorrectness[pixelIdx]
				correctB := gridB.PixelCorrectness[pixelIdx]

				if correctA && !correctB {
					uniqueA++
				} else if correctB && !correctA {
					uniqueB++
				}
			}

			// Only stitch if there's complementarity
			if uniqueA == 0 && uniqueB == 0 {
				continue
			}

			// Stitch: take correct pixels from whichever grid got them right
			stitchedOutput := make([]int, totalPixels)
			stitchedCorrectness := make([]bool, totalPixels)
			correctCount := 0

			for pixelIdx := 0; pixelIdx < totalPixels; pixelIdx++ {
				correctA := gridA.PixelCorrectness[pixelIdx]
				correctB := gridB.PixelCorrectness[pixelIdx]

				var chosenColor int
				if correctA {
					chosenColor = gridA.Output[pixelIdx]
				} else if correctB {
					chosenColor = gridB.Output[pixelIdx]
				} else {
					// Neither correct - use higher coverage grid's prediction
					if gridA.Coverage > gridB.Coverage {
						chosenColor = gridA.Output[pixelIdx]
					} else {
						chosenColor = gridB.Output[pixelIdx]
					}
				}

				stitchedOutput[pixelIdx] = chosenColor
				if chosenColor == targetColors[pixelIdx] {
					stitchedCorrectness[pixelIdx] = true
					correctCount++
				}
			}

			coverage := float64(correctCount) / float64(totalPixels) * 100

			// Only keep if this improves over BOTH parents
			if coverage > gridA.Coverage && coverage > gridB.Coverage {
				newGrids = append(newGrids, &StitchedGrid{
					ID:               baseID + len(newGrids),
					Output:           stitchedOutput,
					Coverage:         coverage,
					PixelCorrectness: stitchedCorrectness,
					SourceNetworks:   append(gridA.SourceNetworks, gridB.SourceNetworks...),
				})
			}
		}
	}

	// Also try 3-way stitching for the top grids
	if len(grids) >= 3 {
		newGrids = append(newGrids, stitchThreeWay(grids, targetColors, totalPixels, baseID+len(newGrids))...)
	}

	return newGrids
}

// stitchThreeWay tries combining the top 3 grids together
func stitchThreeWay(grids []*StitchedGrid, targetColors []int, totalPixels int, baseID int) []*StitchedGrid {
	var newGrids []*StitchedGrid

	// Sort by coverage and take top N for 3-way combinations
	sorted := make([]*StitchedGrid, len(grids))
	copy(sorted, grids)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Coverage > sorted[j].Coverage
	})

	// Try 3-way combinations of top 10 grids
	topN := 10
	if len(sorted) < topN {
		topN = len(sorted)
	}

	for i := 0; i < topN; i++ {
		for j := i + 1; j < topN; j++ {
			for k := j + 1; k < topN; k++ {
				gridA := sorted[i]
				gridB := sorted[j]
				gridC := sorted[k]

				// 3-way stitch: take correct pixel from any of the three
				stitchedOutput := make([]int, totalPixels)
				stitchedCorrectness := make([]bool, totalPixels)
				correctCount := 0

				for pixelIdx := 0; pixelIdx < totalPixels; pixelIdx++ {
					var chosenColor int
					var isCorrect bool

					if gridA.PixelCorrectness[pixelIdx] {
						chosenColor = gridA.Output[pixelIdx]
						isCorrect = true
					} else if gridB.PixelCorrectness[pixelIdx] {
						chosenColor = gridB.Output[pixelIdx]
						isCorrect = true
					} else if gridC.PixelCorrectness[pixelIdx] {
						chosenColor = gridC.Output[pixelIdx]
						isCorrect = true
					} else {
						// None correct - use highest coverage prediction
						if gridA.Coverage >= gridB.Coverage && gridA.Coverage >= gridC.Coverage {
							chosenColor = gridA.Output[pixelIdx]
						} else if gridB.Coverage >= gridC.Coverage {
							chosenColor = gridB.Output[pixelIdx]
						} else {
							chosenColor = gridC.Output[pixelIdx]
						}
					}

					stitchedOutput[pixelIdx] = chosenColor
					stitchedCorrectness[pixelIdx] = isCorrect
					if isCorrect {
						correctCount++
					}
				}

				coverage := float64(correctCount) / float64(totalPixels) * 100

				// Only keep if better than all three parents
				if coverage > gridA.Coverage && coverage > gridB.Coverage && coverage > gridC.Coverage {
					newGrids = append(newGrids, &StitchedGrid{
						ID:               baseID + len(newGrids),
						Output:           stitchedOutput,
						Coverage:         coverage,
						PixelCorrectness: stitchedCorrectness,
						SourceNetworks:   []int{gridA.ID, gridB.ID, gridC.ID},
					})
				}
			}
		}
	}

	return newGrids
}

// ============================================================================

// analyzeOutputProfile analyzes a network's predictions on training data to populate its OutputProfile
func analyzeOutputProfile(ns *NetworkState, trainTasks []*ARCTask43) OutputProfile {
	profile := OutputProfile{}

	samples := createEvalSamples43(trainTasks)
	if len(samples) == 0 {
		return profile
	}

	var totalPixels, correctPixels int
	var centerCorrect, centerTotal int
	var edgeCorrect, edgeTotal int
	quadrantCorrect := make([]int, 4)
	quadrantTotal := make([]int, 4)
	colorCorrect := [10]int{}
	colorTotal := [10]int{}

	for _, sample := range samples {
		output, _ := ns.Network.ForwardCPU(sample.Input)

		height, width := sample.Height, sample.Width
		numColors := 10

		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				pixelIdx := y*width + x
				base := pixelIdx * numColors

				if base+numColors > len(output) || base+numColors > len(sample.Target) {
					continue
				}

				// Get predicted and target color
				predColor := argmax43(output[base : base+numColors])
				targetColor := argmax43(sample.Target[base : base+numColors])

				totalPixels++
				if predColor == targetColor {
					correctPixels++

					// Track quadrant accuracy
					quadrant := getQuadrant(y, x, height, width)
					quadrantCorrect[quadrant]++

					// Track center vs edge
					if isCenterPixel(y, x, height, width) {
						centerCorrect++
					} else {
						edgeCorrect++
					}

					// Track per-color accuracy
					if targetColor >= 0 && targetColor < 10 {
						colorCorrect[targetColor]++
					}
				}

				// Update totals
				quadrant := getQuadrant(y, x, height, width)
				quadrantTotal[quadrant]++

				if isCenterPixel(y, x, height, width) {
					centerTotal++
				} else {
					edgeTotal++
				}

				if targetColor >= 0 && targetColor < 10 {
					colorTotal[targetColor]++
				}
			}
		}
	}

	// Calculate profile metrics
	if totalPixels > 0 {
		profile.OverallAcc = float64(correctPixels) / float64(totalPixels)
	}

	if centerTotal > 0 {
		profile.CenterAcc = float64(centerCorrect) / float64(centerTotal)
	}

	if edgeTotal > 0 {
		profile.EdgeAcc = float64(edgeCorrect) / float64(edgeTotal)
	}

	// Quadrant accuracies
	if quadrantTotal[0] > 0 {
		profile.TopLeftAcc = float64(quadrantCorrect[0]) / float64(quadrantTotal[0])
	}
	if quadrantTotal[1] > 0 {
		profile.TopRightAcc = float64(quadrantCorrect[1]) / float64(quadrantTotal[1])
	}
	if quadrantTotal[2] > 0 {
		profile.BottomLeftAcc = float64(quadrantCorrect[2]) / float64(quadrantTotal[2])
	}
	if quadrantTotal[3] > 0 {
		profile.BottomRightAcc = float64(quadrantCorrect[3]) / float64(quadrantTotal[3])
	}

	// Color accuracies
	for i := 0; i < 10; i++ {
		if colorTotal[i] > 0 {
			profile.ColorAccuracies[i] = float64(colorCorrect[i]) / float64(colorTotal[i])
		}
	}

	return profile
}

// getQuadrant returns which quadrant (0-3) a pixel belongs to
func getQuadrant(y, x, height, width int) int {
	midY := height / 2
	midX := width / 2

	if y < midY {
		if x < midX {
			return 0 // Top-left
		}
		return 1 // Top-right
	}
	if x < midX {
		return 2 // Bottom-left
	}
	return 3 // Bottom-right
}

// isCenterPixel checks if a pixel is in the center region (inner 50%)
func isCenterPixel(y, x, height, width int) bool {
	marginY := height / 4
	marginX := width / 4
	return y >= marginY && y < height-marginY && x >= marginX && x < width-marginX
}

// getProfileVector converts an OutputProfile to a numeric vector for clustering
func getProfileVector(p OutputProfile) []float64 {
	vec := make([]float64, 0, 17)
	vec = append(vec, p.OverallAcc)
	vec = append(vec, p.CenterAcc)
	vec = append(vec, p.EdgeAcc)
	vec = append(vec, p.TopLeftAcc, p.TopRightAcc, p.BottomLeftAcc, p.BottomRightAcc)

	// Add color accuracies
	for i := 0; i < 10; i++ {
		vec = append(vec, p.ColorAccuracies[i])
	}

	return vec
}

// profileDistance calculates Euclidean distance between two profile vectors
func profileDistance(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return math.MaxFloat64
	}

	var sum float64
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// clusterNetworksByProfile groups networks into clusters based on their output profiles
func clusterNetworksByProfile(specialists []*NetworkSpecialist, numClusters int) []*SpecialistCluster {
	if len(specialists) == 0 || numClusters <= 0 {
		return nil
	}

	// Ensure we don't have more clusters than specialists
	if numClusters > len(specialists) {
		numClusters = len(specialists)
	}

	// Convert profile vectors to float32 for nn.KMeansCluster
	vectors := make([][]float32, len(specialists))
	for i, s := range specialists {
		v := getProfileVector(s.Profile)
		vectors[i] = make([]float32, len(v))
		for j, val := range v {
			vectors[i][j] = float32(val)
		}
	}

	// Use nn.KMeansCluster for clustering
	_, assignments := nn.KMeansCluster(vectors, numClusters, 50, true)

	// Initialize clusters
	clusters := make([]*SpecialistCluster, numClusters)
	for i := 0; i < numClusters; i++ {
		clusters[i] = &SpecialistCluster{
			ID:              i,
			Name:            fmt.Sprintf("Cluster-%d", i),
			Members:         make([]*NetworkSpecialist, 0),
			MemberIDs:       make([]int, 0),
			SolvedTaskIDs:   make([]string, 0),
			BestCombineMode: "avg",
		}
	}

	// Assign specialists to clusters based on assignments
	for i, clusterIdx := range assignments {
		specialists[i].ClusterID = clusterIdx
		clusters[clusterIdx].Members = append(clusters[clusterIdx].Members, specialists[i])
	}

	// Set centroid profiles and describe clusters
	for k, cluster := range clusters {
		if len(cluster.Members) == 0 {
			continue
		}

		// Calculate average profile for centroid
		for _, s := range cluster.Members {
			cluster.CentroidProfile.OverallAcc += s.Profile.OverallAcc
			cluster.CentroidProfile.CenterAcc += s.Profile.CenterAcc
			cluster.CentroidProfile.EdgeAcc += s.Profile.EdgeAcc
			cluster.CentroidProfile.TopLeftAcc += s.Profile.TopLeftAcc
			cluster.CentroidProfile.TopRightAcc += s.Profile.TopRightAcc
			cluster.CentroidProfile.BottomLeftAcc += s.Profile.BottomLeftAcc
			cluster.CentroidProfile.BottomRightAcc += s.Profile.BottomRightAcc

			for i := 0; i < 10; i++ {
				cluster.CentroidProfile.ColorAccuracies[i] += s.Profile.ColorAccuracies[i]
			}
		}

		n := float64(len(cluster.Members))
		cluster.CentroidProfile.OverallAcc /= n
		cluster.CentroidProfile.CenterAcc /= n
		cluster.CentroidProfile.EdgeAcc /= n
		cluster.CentroidProfile.TopLeftAcc /= n
		cluster.CentroidProfile.TopRightAcc /= n
		cluster.CentroidProfile.BottomLeftAcc /= n
		cluster.CentroidProfile.BottomRightAcc /= n

		for i := 0; i < 10; i++ {
			cluster.CentroidProfile.ColorAccuracies[i] /= n
		}

		cluster.Specialty = describeClusterSpecialty(cluster.CentroidProfile, k)
	}

	return clusters
}

// describeClusterSpecialty generates a human-readable description of what a cluster specializes in
func describeClusterSpecialty(profile OutputProfile, clusterID int) string {
	var strengths []string

	// Check spatial specialization
	if profile.CenterAcc > profile.EdgeAcc+0.1 {
		strengths = append(strengths, "center-focused")
	} else if profile.EdgeAcc > profile.CenterAcc+0.1 {
		strengths = append(strengths, "edge-focused")
	}

	// Check quadrant specialization
	quadrantScores := []float64{profile.TopLeftAcc, profile.TopRightAcc, profile.BottomLeftAcc, profile.BottomRightAcc}
	maxQ, maxQScore := 0, 0.0
	for i, score := range quadrantScores {
		if score > maxQScore {
			maxQ = i
			maxQScore = score
		}
	}
	quadrantNames := []string{"top-left", "top-right", "bottom-left", "bottom-right"}
	if maxQScore > 0.3 {
		strengths = append(strengths, quadrantNames[maxQ]+"-specialist")
	}

	// Check color specialization
	var bestColor int
	var bestColorScore float64
	for color := 0; color < 10; color++ {
		if profile.ColorAccuracies[color] > bestColorScore {
			bestColor = color
			bestColorScore = profile.ColorAccuracies[color]
		}
	}
	if bestColorScore > 0.5 {
		strengths = append(strengths, fmt.Sprintf("color-%d-expert", bestColor))
	}

	// Overall accuracy description
	if profile.OverallAcc > 0.8 {
		strengths = append(strengths, "high-accuracy")
	} else if profile.OverallAcc < 0.3 {
		strengths = append(strengths, "pattern-seeker")
	}

	if len(strengths) == 0 {
		return fmt.Sprintf("generalist-cluster-%d", clusterID)
	}

	return strings.Join(strengths, " | ")
}

// evaluateSpecialistCluster evaluates ensembles formed from a cluster using different combine modes
func evaluateSpecialistCluster(cluster *SpecialistCluster, evalTasks []*ARCTask43) []ClusterEnsembleResult {
	if len(cluster.Members) == 0 {
		return nil
	}

	combineModes := []string{"avg", "grid_scatter", "concat"}
	results := make([]ClusterEnsembleResult, 0, len(combineModes))

	for _, mode := range combineModes {
		result := ClusterEnsembleResult{
			ClusterID:   cluster.ID,
			CombineMode: mode,
		}

		// Collect networks from cluster members
		networks := make([]*NetworkState, len(cluster.Members))
		for i, member := range cluster.Members {
			networks[i] = member.NetworkState
		}

		// Evaluate using voting fusion (simple but effective)
		var solved, totalCorrect, totalPixels int
		for _, task := range evalTasks {
			samples := createEvalSamples43([]*ARCTask43{task})
			if len(samples) == 0 {
				continue
			}

			taskSolved := true
			for _, sample := range samples {
				// Get predictions from all networks
				allOutputs := make([][]float32, len(networks))
				for i, ns := range networks {
					output, _ := ns.Network.ForwardCPU(sample.Input)
					allOutputs[i] = output
				}

				// Fuse outputs using voting
				fusedOutput := fuseOutputs43(allOutputs, nil, FusionVote, sample.Height, sample.Width)

				// Calculate accuracy
				numColors := 10
				for y := 0; y < sample.Height; y++ {
					for x := 0; x < sample.Width; x++ {
						pixelIdx := y*sample.Width + x
						base := pixelIdx * numColors

						if base+numColors > len(fusedOutput) || base+numColors > len(sample.Target) {
							continue
						}

						predColor := argmax43(fusedOutput[base : base+numColors])
						targetColor := argmax43(sample.Target[base : base+numColors])

						totalPixels++
						if predColor == targetColor {
							totalCorrect++
						} else {
							taskSolved = false
						}
					}
				}
			}

			if taskSolved {
				solved++
				cluster.SolvedTaskIDs = append(cluster.SolvedTaskIDs, task.ID)
			}
		}

		result.TasksSolved = solved
		if totalPixels > 0 {
			result.AvgAccuracy = float64(totalCorrect) / float64(totalPixels)
		}

		results = append(results, result)
	}

	return results
}

// createMonolithicEnsemble creates a large ensemble combining best networks from each cluster
func createMonolithicEnsemble(clusters []*SpecialistCluster, evalTasks []*ARCTask43) *ClusterEnsembleResult {
	// Collect top networks from each cluster (by overall accuracy)
	var allNetworks []*NetworkState

	for _, cluster := range clusters {
		if len(cluster.Members) == 0 {
			continue
		}

		// Sort members by overall accuracy
		members := make([]*NetworkSpecialist, len(cluster.Members))
		copy(members, cluster.Members)
		sort.Slice(members, func(i, j int) bool {
			return members[i].Profile.OverallAcc > members[j].Profile.OverallAcc
		})

		// Take top 3 from each cluster (or all if fewer)
		take := 3
		if len(members) < take {
			take = len(members)
		}

		for i := 0; i < take; i++ {
			allNetworks = append(allNetworks, members[i].NetworkState)
		}
	}

	if len(allNetworks) == 0 {
		return nil
	}

	// Evaluate monolithic ensemble
	result := &ClusterEnsembleResult{
		ClusterID:   -1, // Special ID for monolithic
		CombineMode: "monolithic",
	}

	var solved, totalCorrect, totalPixels int
	var solvedTasks []string

	for _, task := range evalTasks {
		samples := createEvalSamples43([]*ARCTask43{task})
		if len(samples) == 0 {
			continue
		}

		taskSolved := true
		for _, sample := range samples {
			allOutputs := make([][]float32, len(allNetworks))
			allConfidences := make([]float64, len(allNetworks))

			for i, ns := range allNetworks {
				output, _ := ns.Network.ForwardCPU(sample.Input)
				allOutputs[i] = output
				allConfidences[i] = ns.OverallAccuracy
			}

			// Use weighted fusion for monolithic ensemble
			fusedOutput := fuseOutputs43(allOutputs, allConfidences, FusionWeighted, sample.Height, sample.Width)

			numColors := 10
			for y := 0; y < sample.Height; y++ {
				for x := 0; x < sample.Width; x++ {
					pixelIdx := y*sample.Width + x
					base := pixelIdx * numColors

					if base+numColors > len(fusedOutput) || base+numColors > len(sample.Target) {
						continue
					}

					predColor := argmax43(fusedOutput[base : base+numColors])
					targetColor := argmax43(sample.Target[base : base+numColors])

					totalPixels++
					if predColor == targetColor {
						totalCorrect++
					} else {
						taskSolved = false
					}
				}
			}
		}

		if taskSolved {
			solved++
			solvedTasks = append(solvedTasks, task.ID)
		}
	}

	result.TasksSolved = solved
	if totalPixels > 0 {
		result.AvgAccuracy = float64(totalCorrect) / float64(totalPixels)
	}
	result.SolvedTaskIDs = solvedTasks

	return result
}

// printSpecialistResults prints the results of specialist cluster evaluation
func printSpecialistResults(clusters []*SpecialistCluster, clusterResults [][]ClusterEnsembleResult, monolithic *ClusterEnsembleResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    🧠 SPECIALIST CLUSTER ANALYSIS                                                            ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for i, cluster := range clusters {
		if len(cluster.Members) == 0 {
			continue
		}

		fmt.Printf("║ Cluster %d: %-50s | Members: %d                                     ║\n",
			cluster.ID, cluster.Specialty, len(cluster.Members))

		if i < len(clusterResults) {
			for _, res := range clusterResults[i] {
				fmt.Printf("║   Mode: %-15s | Solved: %3d | Accuracy: %.2f%%                                                       ║\n",
					res.CombineMode, res.TasksSolved, res.AvgAccuracy*100)
			}
		}
		fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	}

	if monolithic != nil {
		fmt.Println("║                                    🌟 MONOLITHIC META-ENSEMBLE                                                               ║")
		fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
		fmt.Printf("║ Networks: %d | Tasks Solved: %d | Accuracy: %.2f%%                                                                ║\n",
			len(clusters)*3, monolithic.TasksSolved, monolithic.AvgAccuracy*100)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
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
	
	fmt.Println("║                                         📊 ARCHITECTURE DIVERSITY                                                            ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   CombineModes: %v                                                                                  ║\n", output.CombineModeStats)
	fmt.Printf("║   Grid Species: %v                   ║\n", output.SpeciesStats)
	
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Key finding
	if len(output.TopEnsembles) > 0 {
		bestEnsemble := output.TopEnsembles[0]
		if bestEnsemble.FusionBonus > 0 {
			fmt.Printf("\n🎯 KEY FINDING: The best ensemble solved %d MORE tasks than any individual!\n", bestEnsemble.FusionBonus)
			fmt.Printf("   This proves that diverse architectures capture different partial solutions.\n")
		} else {
			fmt.Printf("\n🎯 KEY FINDING: Best strategy was %s with %d tasks solved.\n", output.BestFusionStrategy, bestEnsemble.TasksSolved)
		}
	} else {
		fmt.Printf("\n🎯 KEY FINDING: Best strategy was %s - stitching approach worked best!\n", output.BestFusionStrategy)
	}

	fmt.Printf("\n🧠 COLLECTIVE WISDOM: %d/%d unique tasks solved across all approaches (%.1f%%)\n",
		output.CollectiveCount, 400, float64(output.CollectiveCount)*100/400)
}

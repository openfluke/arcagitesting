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
// THE ARCHITECTURAL DIVERSITY ZOO  - Exploring Layer & Combine Mode Variations
// ═══════════════════════════════════════════════════════════════════════════════
//
// Test 39 proved small models work. Test 40 v1 proved raw capacity doesn't help.
// This version explores ARCHITECTURAL DIVERSITY:
// - CombineModes: concat, add, avg, grid_scatter
// - SoftmaxTypes: Standard, Sparse, Gumbel, Entmax
// - BrainTypes: MHA, LSTM, RNN, Dense, SwiGLU, LayerNorm+Dense
// - Activations: All 5 types

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 400
	InitScale   = float32(0.5)

	// Zoo settings - GENERATION 4: OPTIMIZED based on results analysis
	ZooSize        = 250              // Reduced from 500 - plateau was at 250
	TestDuration   = 60 * time.Second // Increased from 10s for deeper training
	WindowDuration = 100 * time.Millisecond
)

// BrainType defines what kind of layer a brain uses
type BrainType int

const (
	BrainMHA BrainType = iota
	BrainLSTM
	BrainRNN
	BrainDense
	BrainSwiGLU         // Gated activation
	BrainNormDense      // LayerNorm + Dense combo
	BrainNestedParallel // Parallel layer INSIDE a parallel layer
	BrainSoftmaxGate    // Softmax as intermediate gating layer
	BrainConv1D         // NEW: 1D Convolution (treats input as 30x30 image)
	BrainRMSNorm        // NEW: RMS Normalization (Llama-style)
	BrainDeepStack      // NEW: Multi-layer branch (Dense->ReLU->Dense)
	BrainResidualMHA    // NEW: MHA with residual connection
	BrainSoftmax        // NEW: Actual softmax layer for attention-like gating
)

var brainTypeNames = []string{
	"MHA", "LSTM", "RNN", "Dense", "SwiGLU", "NormDense",
	"Nested", "SoftGate", "Conv1D", "RMSNorm", "DeepStack", "ResMHA", "Softmax",
}

// ArchPattern defines the overall architecture pattern
type ArchPattern int

const (
	ArchFlat      ArchPattern = iota // Simple flat parallel (current)
	ArchNested                       // Nested parallel branches
	ArchCascade                      // Softmax gates between stages
	ArchCrossGrid                    // Grid-to-grid routing
)

var archPatternNames = []string{"Flat", "Nested", "Cascade", "CrossGrid"}

// BranchPattern defines multi-layer composition within a single branch
type BranchPattern int

const (
	BranchSingle             BranchPattern = iota // Single layer (original)
	BranchDenseStack                              // Dense → Dense (2 activations)
	BranchTransformer                             // MHA + FFN
	BranchRecurrentAttention                      // LSTM + MHA hybrid
	BranchGatedNorm                               // SwiGLU + RMSNorm
	BranchTriple                                  // Dense → Norm → Dense
)

var branchPatternNames = []string{"Single", "DenseStack", "Transformer", "RecurrentAttn", "GatedNorm", "Triple"}

// CombineModeType for parallel layer combination
type CombineModeType int

const (
	CombineConcat CombineModeType = iota
	CombineAdd
	CombineAvg
	CombineGridScatter
)

var combineModeNames = []string{"concat", "add", "avg", "grid_scatter"}

// OutputType for final layer softmax variant
type OutputType int

const (
	OutputSigmoid   OutputType = iota // Standard (like Test 39)
	OutputSoftmax                     // Standard softmax
	OutputSparsemax                   // Can output exact zeros
	OutputGumbel                      // Adds exploration noise
	OutputEntmax                      // Generalization of softmax/sparsemax
)

var outputTypeNames = []string{"Sigmoid", "Softmax", "Sparsemax", "Gumbel", "Entmax"}

// ActivationType for layer activations
type ActivationType int

const (
	ActScaledReLU ActivationType = iota
	ActLeakyReLU
	ActTanh
	ActSoftplus
	ActSigmoid // NEW
)

var activationNames = []string{"ScaledReLU", "LeakyReLU", "Tanh", "Softplus", "Sigmoid"}

// getActivation returns the nn activation constant
func getActivation(act ActivationType) nn.ActivationType {
	switch act {
	case ActScaledReLU:
		return nn.ActivationScaledReLU
	case ActLeakyReLU:
		return nn.ActivationLeakyReLU
	case ActTanh:
		return nn.ActivationTanh
	case ActSoftplus:
		return nn.ActivationSoftplus
	case ActSigmoid:
		return nn.ActivationSigmoid
	default:
		return nn.ActivationLeakyReLU
	}
}

// Grid shapes from Test 39 (proven effective)
type GridShape struct {
	Rows int
	Cols int
	Name string
}

// Grid shapes - GENERATION 3: Cambrian Explosion (restore diversity + extreme shapes)
var gridShapes = []GridShape{
	{1, 1, "1x1 Mono"}, // Brought back - solved 11 unique in Test 39!
	{2, 2, "2x2 Standard"},
	{3, 3, "3x3 Complex"},
	{4, 1, "4x1 Tall"},
	{1, 4, "1x4 Wide"},
	{2, 3, "2x3 Rect"},
	{3, 2, "3x2 Rect"},
	{2, 4, "2x4 Wide"},
	{4, 2, "4x2 Tall"},
	{1, 8, "1x8 Pipeline"},           // The Pipeline (extreme depth)
	{8, 1, "8x1 Scanner"},            // The Parallel Scanner (extreme width)
	{10, 1, "10x1 ScannerWider"},     // The Parallel Scanner (extreme width)
	{15, 1, "15x1 ScannerWiderMore"}, // The Parallel Scanner (extreme width)
	{8, 2, "8x2 WideGrid"},           // NEW: 16 brains wide
	{8, 3, "8x3 MegaWide"},           // NEW: 24 brains wide
	{7, 3, "7x3 Prime"},              // NEW: 21 brains (prime-ish)
	{6, 4, "6x4 Matrix"},             // NEW: 24 brains rectangular
	{5, 4, "5x4 Squad"},              // NEW: 20 brains balanced
}

// MutantConfig defines architectural configuration
type MutantConfig struct {
	ID                 int             `json:"id"`
	Name               string          `json:"name"`
	Species            string          `json:"species"`
	MutationStr        string          `json:"mutationStr"`
	GridRows           int             `json:"gridRows"`
	GridCols           int             `json:"gridCols"`
	NumBrains          int             `json:"numBrains"`
	DModel             int             `json:"dModel"`
	NumHeads           int             `json:"numHeads"`
	LearningRate       float32         `json:"learningRate"`
	BudgetScale        float32         `json:"budgetScale"`
	Activation         ActivationType  `json:"-"`
	ActivationN        string          `json:"activation"`
	CombineMode        CombineModeType `json:"-"`
	CombineModeN       string          `json:"combineMode"`
	OutputType         OutputType      `json:"-"`
	OutputTypeN        string          `json:"outputType"`
	Brains             []BrainType     `json:"-"`
	BrainNames         []string        `json:"brains"`
	BranchPatterns     []BranchPattern `json:"-"`              // NEW: Multi-layer pattern per brain
	BranchPatternNames []string        `json:"branchPatterns"` // NEW: Human-readable
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

// CombineModeStats tracks performance by combine mode
type CombineModeStats struct {
	CombineMode string   `json:"combineMode"`
	Count       int      `json:"count"`
	TotalSolved int      `json:"totalSolved"`
	UniqueTasks []string `json:"uniqueTasks"`
	UniqueCount int      `json:"uniqueCount"`
	BestMutant  string   `json:"bestMutant"`
	BestSolved  int      `json:"bestSolved"`
}

// OutputTypeStats tracks performance by output type
type OutputTypeStats struct {
	OutputType  string   `json:"outputType"`
	Count       int      `json:"count"`
	TotalSolved int      `json:"totalSolved"`
	UniqueTasks []string `json:"uniqueTasks"`
	UniqueCount int      `json:"uniqueCount"`
	BestMutant  string   `json:"bestMutant"`
	BestSolved  int      `json:"bestSolved"`
}

// DiscoveryPoint tracks cumulative unique tasks
type DiscoveryPoint struct {
	MutantNum        int `json:"mutantNum"`
	CumulativeUnique int `json:"cumulativeUnique"`
}

// ZooResults is the output
type ZooResults struct {
	ZooSize              int                `json:"zooSize"`
	TopMutants           []MutantResult     `json:"topMutants"`
	SpeciesBreakdown     []SpeciesStats     `json:"speciesBreakdown"`
	CombineModeBreakdown []CombineModeStats `json:"combineModeBreakdown"`
	OutputTypeBreakdown  []OutputTypeStats  `json:"outputTypeBreakdown"`
	CollectiveTasks      []string           `json:"collectiveTasks"`
	CollectiveCount      int                `json:"collectiveCount"`
	DiscoveryTimeline    []DiscoveryPoint   `json:"discoveryTimeline"`
	Timestamp            string             `json:"timestamp"`
	Duration             string             `json:"duration"`
	WorkerCount          int                `json:"workerCount"`
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

	fmt.Println("╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🧬 ARCHITECTURAL DIVERSITY ZOO (ARC-AGI2) - Exploring Layer & Combine Mode Variations              ║")
	fmt.Println("║                                                                                                        ║")
	fmt.Printf("║   Spawning %d mutant architectures with %d parallel workers...                                        ║\n", ZooSize, numWorkers)
	fmt.Println("║   Testing: CombineModes (concat/add/avg), SoftmaxTypes (Sparse/Gumbel/Entmax), SwiGLU brains          ║")
	fmt.Println("║   Training Duration: 60 seconds per mutant | Dataset: ARC-AGI2 (harder tasks)                        ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Load ARC-AGI2 training data
	trainTasks, err := loadARCTasks("ARC-AGI2/data/training", NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load ARC-AGI2 training tasks: %v\n", err)
		return
	}

	// Load ARC-AGI2 evaluation data (120 tasks)
	evalTasks, err := loadARCTasks("ARC-AGI2/data/evaluation", 120)
	if err != nil {
		fmt.Printf("❌ Failed to load ARC-AGI2 eval tasks: %v\n", err)
		return
	}

	trainSamples := createSequentialSamples(trainTasks)
	evalSamples := createEvalSamples(evalTasks)

	fmt.Printf("\n📦 Loaded %d ARC-AGI2 training tasks, %d train samples\n", len(trainTasks), len(trainSamples))
	fmt.Printf("📦 Loaded %d ARC-AGI2 eval tasks, %d eval samples\n", len(evalTasks), len(evalSamples))
	fmt.Printf("🧬 Generating %d mutant configurations with architectural diversity...\n\n", ZooSize)

	// Generate mutant configurations
	configs := generateMutantConfigs(ZooSize)

	// Track results
	var (
		results             = make([]MutantResult, ZooSize)
		collectiveTasksMu   sync.Mutex
		collectiveTasks     = make(map[string]bool)
		speciesTasks        = make(map[string]map[string]bool)
		speciesResults      = make(map[string][]MutantResult)
		combineModeTasks    = make(map[string]map[string]bool)
		combineModeResults  = make(map[string][]MutantResult)
		outputTypeTasks     = make(map[string]map[string]bool)
		outputTypeResults   = make(map[string][]MutantResult)
		discoveryTimeline   []DiscoveryPoint
		processedCount      int
		lastReportedPercent int
	)

	// Initialize maps
	for _, shape := range gridShapes {
		speciesTasks[shape.Name] = make(map[string]bool)
		speciesResults[shape.Name] = []MutantResult{}
	}
	for _, cm := range combineModeNames {
		combineModeTasks[cm] = make(map[string]bool)
		combineModeResults[cm] = []MutantResult{}
	}
	for _, ot := range outputTypeNames {
		outputTypeTasks[ot] = make(map[string]bool)
		outputTypeResults[ot] = []MutantResult{}
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
					combineModeTasks[result.Config.CombineModeN][taskID] = true
					outputTypeTasks[result.Config.OutputTypeN][taskID] = true
				}
				speciesResults[result.Config.Species] = append(speciesResults[result.Config.Species], result)
				combineModeResults[result.Config.CombineModeN] = append(combineModeResults[result.Config.CombineModeN], result)
				outputTypeResults[result.Config.OutputTypeN] = append(outputTypeResults[result.Config.OutputTypeN], result)

				processedCount++

				// Record discovery point every 10 mutants
				if processedCount%10 == 0 {
					discoveryTimeline = append(discoveryTimeline, DiscoveryPoint{
						MutantNum:        processedCount,
						CumulativeUnique: len(collectiveTasks),
					})
				}

				// Progress report
				percent := processedCount * 100 / ZooSize
				if percent >= lastReportedPercent+5 {
					lastReportedPercent = percent
					fmt.Printf("🔄 Progress: %d/%d (%d%%) | Unique tasks: %d | Exploring diversity...\n",
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
	sort.Slice(speciesBreakdown, func(i, j int) bool {
		return speciesBreakdown[i].UniqueCount > speciesBreakdown[j].UniqueCount
	})

	// Build combine mode stats
	var combineModeBreakdown []CombineModeStats
	for _, cm := range combineModeNames {
		cmRes := combineModeResults[cm]
		uniqueTasks := make([]string, 0, len(combineModeTasks[cm]))
		for t := range combineModeTasks[cm] {
			uniqueTasks = append(uniqueTasks, t)
		}
		sort.Strings(uniqueTasks)

		bestMutant := ""
		bestSolved := 0
		totalSolved := 0
		for _, r := range cmRes {
			totalSolved += r.TasksSolved
			if r.TasksSolved > bestSolved {
				bestSolved = r.TasksSolved
				bestMutant = r.Config.MutationStr
			}
		}

		combineModeBreakdown = append(combineModeBreakdown, CombineModeStats{
			CombineMode: cm,
			Count:       len(cmRes),
			TotalSolved: totalSolved,
			UniqueTasks: uniqueTasks,
			UniqueCount: len(uniqueTasks),
			BestMutant:  bestMutant,
			BestSolved:  bestSolved,
		})
	}
	sort.Slice(combineModeBreakdown, func(i, j int) bool {
		return combineModeBreakdown[i].UniqueCount > combineModeBreakdown[j].UniqueCount
	})

	// Build output type stats
	var outputTypeBreakdown []OutputTypeStats
	for _, ot := range outputTypeNames {
		otRes := outputTypeResults[ot]
		uniqueTasks := make([]string, 0, len(outputTypeTasks[ot]))
		for t := range outputTypeTasks[ot] {
			uniqueTasks = append(uniqueTasks, t)
		}
		sort.Strings(uniqueTasks)

		bestMutant := ""
		bestSolved := 0
		totalSolved := 0
		for _, r := range otRes {
			totalSolved += r.TasksSolved
			if r.TasksSolved > bestSolved {
				bestSolved = r.TasksSolved
				bestMutant = r.Config.MutationStr
			}
		}

		outputTypeBreakdown = append(outputTypeBreakdown, OutputTypeStats{
			OutputType:  ot,
			Count:       len(otRes),
			TotalSolved: totalSolved,
			UniqueTasks: uniqueTasks,
			UniqueCount: len(uniqueTasks),
			BestMutant:  bestMutant,
			BestSolved:  bestSolved,
		})
	}
	sort.Slice(outputTypeBreakdown, func(i, j int) bool {
		return outputTypeBreakdown[i].UniqueCount > outputTypeBreakdown[j].UniqueCount
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
		ZooSize:              ZooSize,
		TopMutants:           results[:topN],
		SpeciesBreakdown:     speciesBreakdown,
		CombineModeBreakdown: combineModeBreakdown,
		OutputTypeBreakdown:  outputTypeBreakdown,
		CollectiveTasks:      collectiveTasksList,
		CollectiveCount:      len(collectiveTasksList),
		DiscoveryTimeline:    discoveryTimeline,
		Timestamp:            time.Now().Format(time.RFC3339),
		Duration:             totalTime.String(),
		WorkerCount:          numWorkers,
	}

	// Save results
	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("monster_results_2.json", data, 0644)
	fmt.Println("\n✅ Results saved to monster_results_2.json")

	// Print results
	printResults(output)
}

func generateMutantConfigs(count int) []MutantConfig {
	configs := make([]MutantConfig, count)

	// GENERATION 4: Focus on D64 (80% chance) - proven more effective
	dModels := []int{64, 64, 64, 64, 32} // 80% D64, 20% D32
	numHeads := []int{4, 8}

	for i := 0; i < count; i++ {
		// Random grid shape
		shape := gridShapes[rand.Intn(len(gridShapes))]
		numBrains := shape.Rows * shape.Cols

		// STABLE: Only use proven brain types that guarantee correct output sizes
		// 30% MHA, 30% LSTM, 15% RNN, 10% Dense, 8% SwiGLU, 7% NormDense
		brains := make([]BrainType, numBrains)
		brainNames := make([]string, numBrains)
		patterns := make([]BranchPattern, numBrains)
		patternNames := make([]string, numBrains)

		for b := 0; b < numBrains; b++ {
			// Select brain type
			r := rand.Float64()
			var brainType BrainType
			switch {
			case r < 0.30:
				brainType = BrainMHA
			case r < 0.60:
				brainType = BrainLSTM
			case r < 0.75:
				brainType = BrainRNN
			case r < 0.85:
				brainType = BrainDense
			case r < 0.93:
				brainType = BrainSwiGLU
			default:
				brainType = BrainNormDense
			}
			brains[b] = brainType
			brainNames[b] = brainTypeNames[brainType]

			// Select branch pattern: 50% Single, 10% each multi-layer
			r = rand.Float64()
			var pattern BranchPattern
			switch {
			case r < 0.50:
				pattern = BranchSingle // Original single-layer
			case r < 0.60:
				pattern = BranchDenseStack // Dense → Dense
			case r < 0.70:
				pattern = BranchTransformer // MHA + FFN
			case r < 0.80:
				pattern = BranchRecurrentAttention // LSTM + MHA
			case r < 0.90:
				pattern = BranchGatedNorm // SwiGLU + RMSNorm
			default:
				pattern = BranchTriple // Dense → Norm → Dense
			}
			patterns[b] = pattern
			patternNames[b] = branchPatternNames[pattern]
		}

		// Random DModel and heads
		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		// GENERATION 4: Narrowed learning rate range (0.0001-0.001) - winners had low LR
		logMin := math.Log(0.0001)
		logMax := math.Log(0.001) // Was 0.1 - now 100x smaller range
		lr := float32(math.Exp(logMin + rand.Float64()*(logMax-logMin)))

		// Random activation
		activation := ActivationType(rand.Intn(5))

		// GENERATION 4: Prefer avg/add (70% combined) - these performed best
		// 40% avg, 30% add, 15% concat, 15% grid_scatter
		var combineMode CombineModeType
		r := rand.Float64()
		switch {
		case r < 0.40:
			combineMode = CombineAvg
		case r < 0.70:
			combineMode = CombineAdd
		case r < 0.85:
			combineMode = CombineConcat
		default:
			combineMode = CombineGridScatter
		}

		// GENERATION 3: Balanced OutputType (30% Entmax, 30% Sparsemax, 40% others)
		var outputType OutputType
		r = rand.Float64()
		switch {
		case r < 0.30:
			outputType = OutputEntmax
		case r < 0.60:
			outputType = OutputSparsemax
		case r < 0.73:
			outputType = OutputSigmoid
		case r < 0.87:
			outputType = OutputSoftmax
		default:
			outputType = OutputGumbel
		}

		// Build mutation string
		brainStr := strings.Join(brainNames, "-")
		mutationStr := fmt.Sprintf("%dx%d_%s_%s_%s_%s_D%d_LR%.4f",
			shape.Rows, shape.Cols,
			combineModeNames[combineMode],
			brainStr,
			outputTypeNames[outputType],
			activationNames[activation],
			dModel, lr)

		configs[i] = MutantConfig{
			ID:                 i,
			Name:               fmt.Sprintf("Mutant-%d", i),
			Species:            shape.Name,
			MutationStr:        mutationStr,
			GridRows:           shape.Rows,
			GridCols:           shape.Cols,
			NumBrains:          numBrains,
			DModel:             dModel,
			NumHeads:           heads,
			LearningRate:       lr,
			BudgetScale:        float32(0.5 + rand.Float64()*0.5),
			Activation:         activation,
			ActivationN:        activationNames[activation],
			CombineMode:        combineMode,
			CombineModeN:       combineModeNames[combineMode],
			OutputType:         outputType,
			OutputTypeN:        outputTypeNames[outputType],
			Brains:             brains,
			BrainNames:         brainNames,
			BranchPatterns:     patterns,
			BranchPatternNames: patternNames,
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

	// Parallel hive layer with configurable combine mode
	parallelLayer := createHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Merger layer
	var mergerInputSize int
	switch config.CombineMode {
	case CombineConcat, CombineGridScatter:
		mergerInputSize = config.DModel * config.GridRows * config.GridCols
	case CombineAdd, CombineAvg:
		mergerInputSize = config.DModel // Add/Avg produce same size as single branch
	}
	mergerLayer := nn.InitDenseLayer(mergerInputSize, config.DModel, activation)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Output layer - use configured output type
	outputLayer := createOutputLayer(config.DModel, config.OutputType)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

func createHive(config MutantConfig) nn.LayerConfig {
	numBrains := config.GridRows * config.GridCols
	branches := make([]nn.LayerConfig, numBrains)
	positions := make([]nn.GridPosition, numBrains)

	for i := 0; i < numBrains; i++ {
		brainType := config.Brains[i]
		var baseBrain nn.LayerConfig

		// Create base brain
		switch brainType {
		case BrainMHA:
			baseBrain = createMHABrain(config.DModel, config.NumHeads)
		case BrainLSTM:
			baseBrain = createLSTMBrain(config.DModel)
		case BrainRNN:
			baseBrain = createRNNBrain(config.DModel)
		case BrainDense:
			baseBrain = createDenseBrain(config.DModel, config.Activation)
		case BrainSwiGLU:
			baseBrain = createSwiGLUBrain(config.DModel)
		case BrainNormDense:
			baseBrain = createNormDenseBrain(config.DModel, config.Activation)
		case BrainNestedParallel:
			baseBrain = createNestedParallelBrain(config.DModel, config.NumHeads)
		case BrainSoftmaxGate:
			baseBrain = createSoftmaxGateBrain(config.DModel)
		case BrainConv1D:
			baseBrain = createConv1DBrain(config.DModel)
		case BrainRMSNorm:
			baseBrain = createRMSNormBrain(config.DModel)
		case BrainDeepStack:
			baseBrain = createDeepStackBrain(config.DModel, config.Activation)
		case BrainResidualMHA:
			baseBrain = createResidualMHABrain(config.DModel, config.NumHeads)
		case BrainSoftmax:
			baseBrain = createSoftmaxBrain(config.DModel)
		default:
			baseBrain = createDenseBrain(config.DModel, config.Activation)
		}

		// Apply branch pattern (wrap base brain with multi-layer structure if needed)
		pattern := BranchSingle
		if len(config.BranchPatterns) > i {
			pattern = config.BranchPatterns[i]
		}
		branches[i] = createMultiLayerBranch(baseBrain, pattern, config.DModel, config.NumHeads, config.Activation)

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
		CombineMode:      combineModeNames[config.CombineMode],
		ParallelBranches: branches,
	}

	// Only set grid positions for grid_scatter mode
	if config.CombineMode == CombineGridScatter {
		layer.GridOutputRows = config.GridRows
		layer.GridOutputCols = config.GridCols
		layer.GridOutputLayers = 1
		layer.GridPositions = positions
	}

	return layer
}

func createOutputLayer(dModel int, outputType OutputType) nn.LayerConfig {
	switch outputType {
	case OutputSoftmax:
		// Dense + Softmax
		dense := nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid)
		scaleWeights(dense.Kernel, InitScale)
		return dense // For simplicity, use sigmoid (softmax layer separately would require different architecture)
	case OutputSparsemax:
		// Dense with sigmoid (sparsemax would need post-processing)
		dense := nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid)
		scaleWeights(dense.Kernel, InitScale)
		return dense
	case OutputGumbel:
		// Dense with sigmoid (gumbel would add noise)
		dense := nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid)
		scaleWeights(dense.Kernel, InitScale)
		return dense
	case OutputEntmax:
		// Dense with tanh (entmax-like behavior)
		dense := nn.InitDenseLayer(dModel, InputSize, nn.ActivationTanh)
		scaleWeights(dense.Kernel, InitScale)
		return dense
	default: // OutputSigmoid
		outputLayer := nn.InitDenseLayer(dModel, InputSize, nn.ActivationSigmoid)
		scaleWeights(outputLayer.Kernel, InitScale)
		return outputLayer
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

// NEW: SwiGLU brain (gated activation)
func createSwiGLUBrain(dModel int) nn.LayerConfig {
	// SwiGLU: gate_proj * silu(up_proj)
	// For now, implement as dense layer (true SwiGLU needs nn.LayerSwiGLU support with proper weights)
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights(dense.Kernel, InitScale*0.7) // Smaller init for stability
	return dense
}

// NEW: LayerNorm + Dense combo brain
func createNormDenseBrain(dModel int, activation ActivationType) nn.LayerConfig {
	// For simplicity, just use a dense layer with smaller weights (norm effect)
	dense := nn.InitDenseLayer(dModel, dModel, getActivation(activation))
	scaleWeights(dense.Kernel, InitScale*0.8)
	return dense
}

// NEW: Nested Parallel brain - Simulates hierarchical processing
// Use two Dense layers with different activations (guaranteed dModel output)
func createNestedParallelBrain(dModel, numHeads int) nn.LayerConfig {
	// Create a mini-parallel with 2 dense sub-branches (safe for all combine modes)
	branch1 := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights(branch1.Kernel, InitScale)

	branch2 := nn.InitDenseLayer(dModel, dModel, nn.ActivationTanh)
	scaleWeights(branch2.Kernel, InitScale*0.8)

	// Combine with avg (voting between two differently-activated dense layers)
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{branch1, branch2},
	}
}

// NEW: Softmax Gate brain - Uses softmax as intermediate attention/gating
// This creates attention-like behavior where softmax focuses on important features
func createSoftmaxGateBrain(dModel int) nn.LayerConfig {
	// For now, use a dense layer followed by a softmax-like behavior (tanh squashing)
	// True softmax gating would need: Dense -> Softmax -> ElementwiseMul with input
	// We approximate this with tanh activation which has similar [0,1] squashing behavior
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationTanh)
	scaleWeights(dense.Kernel, InitScale*0.6) // Smaller init for gating stability
	return dense
}

// NEW: Conv1D brain - Simulates local pattern detection with correct output size
// Note: Use Dense layer to guarantee dModel output for add/avg combine compatibility
func createConv1DBrain(dModel int) nn.LayerConfig {
	// Use a Dense layer with Tanh activation to simulate pattern-based processing
	// (Full Conv2D would output wrong shape for parallel combine modes)
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationTanh)
	scaleWeights(dense.Kernel, InitScale*0.7)
	return dense
}

// NEW: RMSNorm brain - RMS Normalization (Llama-style, no beta)
func createRMSNormBrain(dModel int) nn.LayerConfig {
	// RMSNorm layer
	return nn.LayerConfig{
		Type:     nn.LayerRMSNorm,
		NormSize: dModel,
		Gamma:    makeOnes(dModel), // Gamma = 1.0
	}
}

// NEW: DeepStack brain - Multi-layer branch using nested parallel!
// Creates: Dense -> ReLU -> Dense (2 layers in one branch)
func createDeepStackBrain(dModel int, activation ActivationType) nn.LayerConfig {
	// Create a nested parallel with 2 sequential dense layers
	// Note: We use nested parallel with "concat" mode where each branch processes differently
	// But for a true sequential stack, we need 2 dense layers

	// Layer 1: Dense with activation
	layer1 := nn.InitDenseLayer(dModel, dModel, getActivation(activation))
	scaleWeights(layer1.Kernel, InitScale)

	// Layer 2: Dense with different activation (projection)
	layer2 := nn.InitDenseLayer(dModel, dModel, nn.ActivationTanh)
	scaleWeights(layer2.Kernel, InitScale*0.8)

	// Wrap in nested parallel with avg (both layers vote on output)
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{layer1, layer2},
	}
}

// NEW: ResidualMHA brain - Simulates residual processing with Dense layers
// (Using pure Dense to guarantee correct dModel output)
func createResidualMHABrain(dModel, numHeads int) nn.LayerConfig {
	// Main path: Dense with strong nonlinearity
	mainPath := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights(mainPath.Kernel, InitScale)

	// Skip path: Dense passthrough (small contribution)
	skipPath := nn.InitDenseLayer(dModel, dModel, nn.ActivationSigmoid)
	scaleWeights(skipPath.Kernel, InitScale*0.3)

	// Use add combine mode for residual-like behavior
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "add",
		ParallelBranches: []nn.LayerConfig{mainPath, skipPath},
	}
}

// NEW: Softmax brain - Actual nn.LayerSoftmax for probability distributions
func createSoftmaxBrain(dModel int) nn.LayerConfig {
	// Initialize a standard softmax layer
	return nn.InitSoftmaxLayer()
}

// ============================================================================
// MULTI-LAYER BRANCH PATTERNS
// These wrap the base brain with additional layers for richer processing
// ============================================================================

// createMultiLayerBranch wraps a base brain with the specified pattern
func createMultiLayerBranch(baseBrain nn.LayerConfig, pattern BranchPattern, dModel, numHeads int, activation ActivationType) nn.LayerConfig {
	switch pattern {
	case BranchSingle:
		return baseBrain // No wrapping, use base brain as-is
	case BranchDenseStack:
		return createDenseStackBranch(baseBrain, dModel, activation)
	case BranchTransformer:
		return createTransformerBranch(baseBrain, dModel, numHeads)
	case BranchRecurrentAttention:
		return createRecurrentAttentionBranch(baseBrain, dModel, numHeads)
	case BranchGatedNorm:
		return createGatedNormBranch(baseBrain, dModel)
	case BranchTriple:
		return createTripleBranch(baseBrain, dModel, activation)
	default:
		return baseBrain
	}
}

// BranchDenseStack: Base + Dense (two perspectives combined with avg)
func createDenseStackBranch(baseBrain nn.LayerConfig, dModel int, activation ActivationType) nn.LayerConfig {
	extraDense := nn.InitDenseLayer(dModel, dModel, getActivation(activation))
	scaleWeights(extraDense.Kernel, InitScale*0.8)
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{baseBrain, extraDense},
	}
}

// BranchTransformer: Base + Dense FFN (mimics attention + FFN pattern)
func createTransformerBranch(baseBrain nn.LayerConfig, dModel, numHeads int) nn.LayerConfig {
	ffn := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights(ffn.Kernel, InitScale*0.7)
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{baseBrain, ffn},
	}
}

// BranchRecurrentAttention: Base + LSTM (combines recurrence with base)
func createRecurrentAttentionBranch(baseBrain nn.LayerConfig, dModel, numHeads int) nn.LayerConfig {
	lstm := createLSTMBrain(dModel)
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{baseBrain, lstm},
	}
}

// BranchGatedNorm: Base + RMSNorm layer (adds normalization perspective)
func createGatedNormBranch(baseBrain nn.LayerConfig, dModel int) nn.LayerConfig {
	rmsNorm := nn.LayerConfig{
		Type:     nn.LayerRMSNorm,
		NormSize: dModel,
		Gamma:    makeOnes(dModel),
	}
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{baseBrain, rmsNorm},
	}
}

// BranchTriple: Base + Dense + Dense (triple voting ensemble)
func createTripleBranch(baseBrain nn.LayerConfig, dModel int, activation ActivationType) nn.LayerConfig {
	dense1 := nn.InitDenseLayer(dModel, dModel, getActivation(activation))
	scaleWeights(dense1.Kernel, InitScale*0.7)
	dense2 := nn.InitDenseLayer(dModel, dModel, nn.ActivationTanh)
	scaleWeights(dense2.Kernel, InitScale*0.6)
	return nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{baseBrain, dense1, dense2},
	}
}

// Helper function to create a slice of ones
func makeOnes(size int) []float32 {
	ones := make([]float32, size)
	for i := range ones {
		ones[i] = 1.0
	}
	return ones
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

func printResults(output *ZooResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    🧬 ARCHITECTURAL DIVERSITY ZOO - RESULTS                                                         ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║   Zoo Size: %d | Unique Tasks Solved: %d | Duration: %s                                                                    ║\n",
		output.ZooSize, output.CollectiveCount, output.Duration)

	// Combine Mode breakdown
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                         📊 COMBINE MODE BREAKDOWN                                                                    ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, cm := range output.CombineModeBreakdown {
		bar := ""
		for j := 0; j < cm.UniqueCount; j++ {
			bar += "█"
		}
		fmt.Printf("║ %-15s: %s %d unique (from %d mutants, best: %d solved)\n",
			cm.CombineMode, bar, cm.UniqueCount, cm.Count, cm.BestSolved)
	}

	// Output Type breakdown
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                         📊 OUTPUT TYPE BREAKDOWN                                                                     ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, ot := range output.OutputTypeBreakdown {
		bar := ""
		for j := 0; j < ot.UniqueCount; j++ {
			bar += "█"
		}
		fmt.Printf("║ %-15s: %s %d unique (from %d mutants, best: %d solved)\n",
			ot.OutputType, bar, ot.UniqueCount, ot.Count, ot.BestSolved)
	}

	// Species breakdown
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                         📊 SPECIES BREAKDOWN                                                                         ║")
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
	fmt.Println("║                                              🏆 TOP 10 MUTANTS                                                                       ║")
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
		if dp.MutantNum%50 == 0 || dp.MutantNum == ZooSize {
			fmt.Printf("   Mutant %4d: %s (%d tasks)\n", dp.MutantNum, bar, dp.CumulativeUnique)
		}
	}

	fmt.Printf("\n🧬 COLLECTIVE WISDOM: %d unique tasks solved by the Zoo\n", output.CollectiveCount)
	fmt.Printf("   Tasks: %v\n", output.CollectiveTasks)
}

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

	"github.com/openfluke/loom/nn"
)

// Test 20: ARC-AGI Adaptive Training with Early Stopping
// Trains in batches, measures accuracy, stops when threshold reached or no improvement

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize
	NumTasks    = 10

	// Adaptive training params
	BatchSize         = 200            // Train more samples per batch
	MaxBatches        = 50             // Maximum training batches
	AccuracyTarget    = 90.0           // Stop if accuracy reaches this
	PatienceLimit     = 10             // More patience
	LearningRateTween = float32(0.02)  // Slightly lower
	LearningRateBP    = float32(0.005) // Higher for smaller net
)

// Smaller network - less capacity, faster to train
var networkConfig = NetworkConfig{Name: "Small", Layers: []int{32, 16}}

type NetworkConfig struct {
	Name   string
	Layers []int
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

type Result struct {
	Mode            string
	FinalAccuracy   float64
	BatchesTrained  int
	StoppedEarly    bool
	StopReason      string
	AccuracyHistory []float64
}

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
	fmt.Println("║  Test 20: ARC-AGI - Adaptive Early Stopping                              ║")
	fmt.Println("║  Train → Measure → Stop when good or stuck                               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasks)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	samples := flattenTasksToSamples(tasks)
	fmt.Printf("Loaded %d tasks, %d samples\n\n", len(tasks), len(samples))

	modes := []TrainingMode{ModeStepBP, ModeTweenChain, ModeStepTweenChain}
	results := make(map[TrainingMode]*Result)
	networks := make(map[TrainingMode]*nn.Network)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			result, net := trainWithEarlyStopping(samples, m)
			mu.Lock()
			results[m] = result
			networks[m] = net
			mu.Unlock()
		}(mode)
	}
	wg.Wait()

	printResults(results)

	// Show visual predictions for best mode
	var bestMode TrainingMode
	bestAcc := 0.0
	for mode, r := range results {
		if r.FinalAccuracy > bestAcc {
			bestAcc = r.FinalAccuracy
			bestMode = mode
		}
	}
	if net, ok := networks[bestMode]; ok {
		showVisualPredictions(net, samples, bestMode, len(networkConfig.Layers)+1, results[bestMode])
	}
}

func trainWithEarlyStopping(samples []Sample, mode TrainingMode) (*Result, *nn.Network) {
	net := createNetwork()
	numLayers := len(networkConfig.Layers) + 1

	// Choose LR based on mode (StepBP needs lower to avoid explosion)
	lr := LearningRateTween
	if mode == ModeStepBP {
		lr = LearningRateBP
	}

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

	result := &Result{Mode: modeNames[mode], AccuracyHistory: []float64{}}
	bestAccuracy := 0.0
	patienceCounter := 0
	sampleIdx := 0

	fmt.Printf("  [%-14s] Starting training...\n", modeNames[mode])

	for batch := 0; batch < MaxBatches; batch++ {
		// Training phase: train on BatchSize samples
		for i := 0; i < BatchSize; i++ {
			sample := samples[sampleIdx%len(samples)]
			sampleIdx++

			switch {
			case usesStep:
				state.SetInput(sample.Input)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output := state.GetOutput()

				if mode == ModeStepBP {
					grad := make([]float32, len(output))
					for j := range output {
						if j < len(sample.Target) {
							grad[j] = clipGrad(output[j]-sample.Target[j], 0.5)
						}
					}
					net.StepBackward(state, grad)
					net.ApplyGradients(lr)
				} else {
					ts.ForwardPass(net, sample.Input)
					applyTweenUpdate(ts, net, sample, output, lr)
				}

			default:
				output := ts.ForwardPass(net, sample.Input)
				applyTweenUpdate(ts, net, sample, output, lr)
			}
		}

		// Measurement phase: evaluate accuracy on all samples
		accuracy := measureAccuracy(net, samples, mode, numLayers, state, ts)
		result.AccuracyHistory = append(result.AccuracyHistory, accuracy)
		result.BatchesTrained = batch + 1

		// Check stopping conditions
		if accuracy >= AccuracyTarget {
			result.FinalAccuracy = accuracy
			result.StoppedEarly = true
			result.StopReason = fmt.Sprintf("Target %.0f%% reached", AccuracyTarget)
			fmt.Printf("  [%-14s] ✓ Batch %3d: %.1f%% - TARGET REACHED!\n", modeNames[mode], batch+1, accuracy)
			return result, net
		}

		if accuracy > bestAccuracy {
			bestAccuracy = accuracy
			patienceCounter = 0
		} else {
			patienceCounter++
			if patienceCounter >= PatienceLimit {
				result.FinalAccuracy = accuracy
				result.StoppedEarly = true
				result.StopReason = fmt.Sprintf("No improvement for %d batches", PatienceLimit)
				fmt.Printf("  [%-14s] ✗ Batch %3d: %.1f%% - NO IMPROVEMENT, stopping\n", modeNames[mode], batch+1, accuracy)
				return result, net
			}
		}

		if batch%10 == 9 {
			fmt.Printf("  [%-14s]   Batch %3d: %.1f%% (best: %.1f%%)\n", modeNames[mode], batch+1, accuracy, bestAccuracy)
		}
	}

	result.FinalAccuracy = measureAccuracy(net, samples, mode, numLayers, state, ts)
	result.StopReason = "Max batches reached"
	fmt.Printf("  [%-14s] ! Batch %3d: %.1f%% - MAX BATCHES\n", modeNames[mode], MaxBatches, result.FinalAccuracy)
	return result, net
}

// Show visual grid predictions for a few samples
func showVisualPredictions(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, res *Result) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║  VISUAL PREDICTIONS (%s - %.1f%% accuracy)                                               ║\n", modeNames[mode], res.FinalAccuracy)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var state *nn.StepState
	var ts *nn.TweenState
	if mode == ModeStepBP || mode == ModeStepTweenChain {
		state = net.InitStepState(InputSize)
	} else {
		ts = nn.NewTweenState(net, nil)
	}

	// Show first 3 samples
	for i, sample := range samples {
		if i >= 3 {
			break
		}

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

		fmt.Printf("\n  Sample %d (%dx%d):\n", i+1, sample.Height, sample.Width)

		// Side by side: Expected | Predicted
		fmt.Printf("  %s\n", "Expected              Predicted")
		for r := 0; r < sample.Height && r < 8; r++ {
			// Expected row
			fmt.Print("  ")
			for c := 0; c < sample.Width && c < 10; c++ {
				idx := r*MaxGridSize + c
				val := int(math.Round(float64(sample.Target[idx]) * 9.0))
				fmt.Printf("%d ", val)
			}
			fmt.Print("     ")
			// Predicted row
			for c := 0; c < sample.Width && c < 10; c++ {
				idx := r*MaxGridSize + c
				val := int(math.Round(float64(output[idx]) * 9.0))
				if val < 0 {
					val = 0
				}
				if val > 9 {
					val = 9
				}
				fmt.Printf("%d ", val)
			}
			fmt.Println()
		}

		// Count correct
		correct := 0
		for r := 0; r < sample.Height; r++ {
			for c := 0; c < sample.Width; c++ {
				idx := r*MaxGridSize + c
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
			}
		}
		fmt.Printf("  Correct: %d/%d (%.1f%%)\n", correct, sample.Height*sample.Width, float64(correct)/float64(sample.Height*sample.Width)*100)
	}
}

func measureAccuracy(net *nn.Network, samples []Sample, mode TrainingMode, numLayers int, state *nn.StepState, ts *nn.TweenState) float64 {
	correct := 0
	total := 0

	for _, sample := range samples {
		var output []float32
		switch {
		case mode == ModeStepBP || mode == ModeStepTweenChain:
			state.SetInput(sample.Input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		default:
			output = ts.ForwardPass(net, sample.Input)
		}

		// Count correct cells
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

func createNetwork() *nn.Network {
	numLayers := len(networkConfig.Layers) + 1
	net := nn.NewNetwork(InputSize, 1, 1, numLayers)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSize, networkConfig.Layers[0], nn.ActivationLeakyReLU))
	for i := 1; i < len(networkConfig.Layers); i++ {
		net.SetLayer(0, 0, i, nn.InitDenseLayer(networkConfig.Layers[i-1], networkConfig.Layers[i], nn.ActivationLeakyReLU))
	}
	net.SetLayer(0, 0, len(networkConfig.Layers), nn.InitDenseLayer(networkConfig.Layers[len(networkConfig.Layers)-1], InputSize, nn.ActivationSigmoid))
	return net
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
// Output
// ============================================================================

func printResults(results map[TrainingMode]*Result) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              ADAPTIVE EARLY STOPPING RESULTS                                                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Mode             │ Final Acc │ Batches │ Stopped │ Reason                                    ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	modes := []TrainingMode{ModeStepBP, ModeTweenChain, ModeStepTweenChain}
	for _, mode := range modes {
		if r, ok := results[mode]; ok {
			stoppedStr := "No"
			if r.StoppedEarly {
				stoppedStr = "Yes"
			}
			fmt.Printf("║ %-16s │   %5.1f%% │    %4d │   %-5s │ %-40s ║\n",
				r.Mode, r.FinalAccuracy, r.BatchesTrained, stoppedStr, r.StopReason)
		}
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Print training curves
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ACCURACY OVER BATCHES                                                      ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	// Sort modes
	var sortedModes []TrainingMode
	for m := range results {
		sortedModes = append(sortedModes, m)
	}
	sort.Slice(sortedModes, func(i, j int) bool { return sortedModes[i] < sortedModes[j] })

	for _, mode := range sortedModes {
		r := results[mode]
		fmt.Printf("║ %-14s │", r.Mode)
		maxShow := 20
		step := 1
		if len(r.AccuracyHistory) > maxShow {
			step = len(r.AccuracyHistory) / maxShow
		}
		for i := 0; i < len(r.AccuracyHistory) && i/step < maxShow; i += step {
			fmt.Printf(" %4.0f", r.AccuracyHistory[i])
		}
		fmt.Println(" ║")
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}

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

// Test 20: ARC-AGI Learning Rate Diagnosis
// Tests LR 0.001 (low), 0.01 (med), 0.05 (high) to find optimal training rate

const (
	MaxGridSize    = 30
	InputSize      = MaxGridSize * MaxGridSize
	NumTasksToRun  = 10
	TrainDuration  = 5 * time.Second
	WindowDuration = 500 * time.Millisecond
)

var networkConfig = NetworkConfig{Name: "Medium", Layers: []int{128, 64}}
var learningRates = []float32{0.001, 0.01, 0.05}

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
	Adaptation *nn.AdaptationResult
	Deviation  *nn.DeviationMetrics
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
	fmt.Println("║  Test 20: ARC-AGI - Learning Rate Diagnosis                              ║")
	fmt.Println("║  Testing LR: 0.001 (underfit?), 0.01 (med), 0.05 (overfit?)              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	tasks, err := loadARCTasks("ARC-AGI/data/training", NumTasksToRun)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	samples := flattenTasksToSamples(tasks)
	fmt.Printf("Loaded %d tasks, %d samples\n\n", len(tasks), len(samples))

	modes := []TrainingMode{ModeStepBP, ModeTweenChain, ModeStepTweenChain}
	allResults := make(map[float32]map[TrainingMode]*Result)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, lr := range learningRates {
		allResults[lr] = make(map[TrainingMode]*Result)
		fmt.Printf("\n╔═══════════════════════════════════════════╗\n")
		fmt.Printf("║  Learning Rate: %.4f                      ║\n", lr)
		fmt.Printf("╚═══════════════════════════════════════════╝\n")

		for _, mode := range modes {
			wg.Add(1)
			go func(rate float32, m TrainingMode) {
				defer wg.Done()
				result := runTraining(samples, m, rate)
				mu.Lock()
				allResults[rate][m] = result
				mu.Unlock()
				fmt.Printf("  [LR=%.4f] [%-12s] Acc:%5.1f%% | Dev:%5.1f%% | 0-10%%:%5.1f%% | 100%%+:%5.1f%%\n",
					rate, modeNames[m], result.Adaptation.AvgAccuracy, result.Deviation.AverageDeviation,
					getBucketPct(result, "0-10%"), getBucketPct(result, "100%+"))
			}(lr, mode)
		}
		wg.Wait()
	}
	printLRTable(allResults)
}

func getBucketPct(r *Result, bucket string) float64 {
	if r.Deviation.TotalSamples == 0 {
		return 0
	}
	if b, ok := r.Deviation.Buckets[bucket]; ok {
		return float64(b.Count) / float64(r.Deviation.TotalSamples) * 100
	}
	return 0
}

// ============================================================================
// Training
// ============================================================================

func runTraining(samples []Sample, mode TrainingMode, lr float32) *Result {
	net := createNetwork()
	numLayers := len(networkConfig.Layers) + 1
	evalCycles := 3

	tracker := nn.NewAdaptationTracker(WindowDuration, TrainDuration)
	tracker.SetModelInfo(networkConfig.Name, modeNames[mode])
	deviationMetrics := nn.NewDeviationMetrics()

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

	tracker.Start("LEARNING", 0)
	start := time.Now()
	sampleIdx := 0

	for time.Since(start) < TrainDuration {
		sample := samples[sampleIdx%len(samples)]
		sampleIdx++

		var bestOutput []float32
		bestDev := math.MaxFloat64

		switch {
		case usesStep:
			for cycle := 0; cycle < evalCycles; cycle++ {
				state.SetInput(sample.Input)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output := state.GetOutput()
				if dev := computeDeviation(sample, output); dev < bestDev {
					bestDev = dev
					bestOutput = append([]float32{}, output...)
				}
			}
			evaluateSample(sample, bestOutput, deviationMetrics, sampleIdx)
			tracker.RecordOutput(bestDev < 50)

			output := state.GetOutput()
			if mode == ModeStepBP {
				grad := make([]float32, len(output))
				for i := range output {
					if i < len(sample.Target) {
						grad[i] = clipGrad(output[i]-sample.Target[i], 1.0)
					}
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(lr)
			} else {
				ts.ForwardPass(net, sample.Input)
				applyTweenUpdate(ts, net, sample, output, lr)
			}

		default:
			for cycle := 0; cycle < evalCycles; cycle++ {
				output := ts.ForwardPass(net, sample.Input)
				if dev := computeDeviation(sample, output); dev < bestDev {
					bestDev = dev
					bestOutput = append([]float32{}, output...)
				}
			}
			evaluateSample(sample, bestOutput, deviationMetrics, sampleIdx)
			tracker.RecordOutput(bestDev < 50)

			output := ts.ForwardPass(net, sample.Input)
			applyTweenUpdate(ts, net, sample, output, lr)
		}
	}

	deviationMetrics.ComputeFinalMetrics()
	return &Result{Adaptation: tracker.Finalize(), Deviation: deviationMetrics}
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

func computeDeviation(sample Sample, output []float32) float64 {
	totalDev, count := 0.0, 0
	for r := 0; r < sample.Height; r++ {
		for c := 0; c < sample.Width; c++ {
			idx := r*MaxGridSize + c
			if idx < len(output) && idx < len(sample.Target) {
				exp, act := float64(sample.Target[idx]), float64(output[idx])
				if math.Abs(exp) < 1e-10 {
					totalDev += math.Abs(act-exp) * 100
				} else {
					totalDev += math.Abs((act-exp)/exp) * 100
				}
				count++
			}
		}
	}
	if count == 0 {
		return math.MaxFloat64
	}
	return totalDev / float64(count)
}

func evaluateSample(sample Sample, output []float32, metrics *nn.DeviationMetrics, sampleIdx int) {
	for r := 0; r < sample.Height; r++ {
		for c := 0; c < sample.Width; c++ {
			idx := r*MaxGridSize + c
			if idx < len(output) && idx < len(sample.Target) {
				result := nn.EvaluatePrediction(sampleIdx*1000+idx, float64(sample.Target[idx]), float64(output[idx]))
				metrics.UpdateMetrics(result)
			}
		}
	}
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

func printLRTable(results map[float32]map[TrainingMode]*Result) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           LEARNING RATE COMPARISON (Deviation Buckets)                                        ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║ LR / Mode        │   0-10% │  10-20% │  20-30% │  30-40% │  40-50% │ 50-100% │   100%+ │")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	buckets := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	modes := []TrainingMode{ModeStepBP, ModeTweenChain, ModeStepTweenChain}

	// Sort learning rates
	var lrs []float32
	for lr := range results {
		lrs = append(lrs, lr)
	}
	sort.Slice(lrs, func(i, j int) bool { return lrs[i] < lrs[j] })

	for _, lr := range lrs {
		for _, mode := range modes {
			if r, ok := results[lr][mode]; ok {
				fmt.Printf("║ %.3f/%-10s │", lr, modeNames[mode])
				for _, b := range buckets {
					fmt.Printf(" %5.1f%% │", getBucketPct(r, b))
				}
				fmt.Println()
			}
		}
	}
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  ↑ Higher 0-10% = better | ↑ Higher 100%+ = worse (model completely wrong)")
}

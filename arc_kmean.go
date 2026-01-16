package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ARC-AGI KMeans Benchmark
//
// Evaluates KMeansLayer performance on ARC-AGI task-switching and generalization.

const (
	ARCMaxGridSize  = 30
	ARCInputSize    = ARCMaxGridSize * ARCMaxGridSize // 900
	ARCNumTasks     = 400                             // All training tasks
	ARCLearningRate = float32(0.005)
	ARCInitScale    = float32(0.2)
	ARCBudgetScale  = float32(0.8)

	ARCDModel      = 128
	ARCNumClusters = 256

	ARCTestDuration   = 15 * time.Second
	ARCWindowDuration = 50 * time.Millisecond
)

type KMeansTrainingMode int

const (
	KModeNormalBP KMeansTrainingMode = iota
	KModeStepBP
	KModeTween
	KModeTweenChain
	KModeStepTween
	KModeStepTweenChain
)

var kmeansModeNames = map[KMeansTrainingMode]string{
	KModeNormalBP:       "NormalBP",
	KModeStepBP:         "StepBP",
	KModeTween:          "Tween",
	KModeTweenChain:     "TweenChain",
	KModeStepTween:      "StepTween",
	KModeStepTweenChain: "StepTweenChain",
}

type KMeansARCTask struct {
	ID          string
	Train, Test []KMeansGridPair
}
type KMeansGridPair struct{ Input, Output [][]int }
type KMeansSample struct {
	Input, Target []float32
	Height, Width int
	TaskID        string
	TaskIndex     int
}

type KMeansTimeWindow struct {
	TimeMs        int     `json:"timeMs"`
	Outputs       int     `json:"outputs"`
	TotalPixelAcc float64 `json:"totalPixelAcc"`
	Accuracy      float64 `json:"accuracy"`
	TaskSwitches  int     `json:"taskSwitches"`
}

type KMeansModeResult struct {
	Windows          []KMeansTimeWindow   `json:"windows"`
	TotalOutputs     int                  `json:"totalOutputs"`
	TotalTaskSwitch  int                  `json:"totalTaskSwitches"`
	TrainTimeSec     float64              `json:"trainTimeSec"`
	EvalAccuracy     float64              `json:"evalAccuracy"`
	TrainTasksSolved int                  `json:"trainTasksSolved"`
	EvalTasksSolved  int                  `json:"evalTasksSolved"`
	AvgTrainAccuracy float64              `json:"avgTrainAccuracy"`
	Stability        float64              `json:"stability"`
	Consistency      float64              `json:"consistency"`
	ThroughputPerSec float64              `json:"throughputPerSec"`
	Score            float64              `json:"score"`
	DevMetrics       *nn.DeviationMetrics `json:"devMetrics"`
}

func main() {
	fmt.Println("🚀 ARC-AGI KMeans Benchmark")

	trainTasks, err := loadARCTasksInternal("ARC-AGI/data/training", ARCNumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load training tasks: %v\n", err)
		return
	}

	evalTasks, err := loadARCTasksInternal("ARC-AGI/data/evaluation", 400)
	if err != nil {
		fmt.Printf("❌ Failed to load eval tasks: %v\n", err)
		return
	}

	trainSamples := createSequentialKMeansSamples(trainTasks)

	modes := []KMeansTrainingMode{
		KModeNormalBP,
		KModeStepBP,
		KModeTween,
		KModeTweenChain,
		KModeStepTween,
		KModeStepTweenChain,
	}

	results := make(map[string]*KMeansModeResult)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, mode := range modes {
		wg.Add(1)
		go func(m KMeansTrainingMode) {
			defer wg.Done()
			name := kmeansModeNames[m]
			fmt.Printf("🏃 Processing [%s]...\n", name)
			res := runKMeansARCBenchmarkInternal(m, trainSamples, evalTasks)
			mu.Lock()
			results[name] = res
			mu.Unlock()
			fmt.Printf("✅ [%s] | Train Acc: %.1f%% | Eval Acc: %.1f%% | Solved(Train/Eval): %d/%d | Score: %.1f\n",
				name, res.AvgTrainAccuracy, res.EvalAccuracy, res.TrainTasksSolved, res.EvalTasksSolved, res.Score)
			res.DevMetrics.PrintSummary()
		}(mode)
	}

	wg.Wait()
	fmt.Println("\n🏁 All benchmarks complete!")
}

func createKMeansARCNET() *nn.Network {
	net := nn.NewNetwork(ARCInputSize, 1, 1, 4)
	net.BatchSize = 1

	// Feature Extractor (Deeper)
	dense1 := nn.InitDenseLayer(ARCInputSize, ARCDModel*2, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, dense1)

	dense2 := nn.InitDenseLayer(ARCDModel*2, ARCDModel, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 1, dense2)

	// KMeans Layer
	kmeansSubNet := nn.InitDenseLayer(ARCDModel, ARCDModel, nn.ActivationLeakyReLU)
	kmeansLayer := nn.InitKMeansLayer(ARCNumClusters, kmeansSubNet, "probabilities")
	net.SetLayer(0, 0, 2, kmeansLayer)

	// Output Layer
	denseOut := nn.InitDenseLayer(ARCNumClusters, ARCInputSize, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 3, denseOut)

	net.InitializeWeights()
	return net
}

func runKMeansARCBenchmarkInternal(mode KMeansTrainingMode, trainSamples []KMeansSample, evalTasks []*KMeansARCTask) *KMeansModeResult {
	numWindows := int(ARCTestDuration / ARCWindowDuration)
	result := &KMeansModeResult{
		Windows:    make([]KMeansTimeWindow, numWindows),
		DevMetrics: nn.NewDeviationMetrics(),
	}
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(ARCWindowDuration.Milliseconds())
	}

	net := createKMeansARCNET()
	opt := nn.NewSGDOptimizer()
	state := net.InitStepState(ARCInputSize)

	var ts *nn.TweenState
	if mode >= KModeTween {
		ts = nn.NewTweenState(net, nil)
		ts.Config.LinkBudgetScale = ARCBudgetScale
		if mode == KModeTweenChain || mode == KModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	lastTaskIndex := -1
	solvedInTrain := make(map[int]bool)

	// Training Phase
	for time.Since(start) < ARCTestDuration {
		elapsed := time.Since(start)
		newWindow := int(elapsed / ARCWindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		sample := trainSamples[sampleIdx%len(trainSamples)]
		sampleIdx++

		if sample.TaskIndex != lastTaskIndex {
			if currentWindow < numWindows {
				result.Windows[currentWindow].TaskSwitches++
			}
			result.TotalTaskSwitch++
			lastTaskIndex = sample.TaskIndex
		}

		state.SetInput(sample.Input)
		net.StepForward(state)
		output := state.GetOutput()

		acc := calculateARCKMeansPixelAccuracy(output, sample)
		if currentWindow < numWindows {
			result.Windows[currentWindow].Outputs++
			result.Windows[currentWindow].TotalPixelAcc += acc
			result.TotalOutputs++
		}

		// Training
		switch mode {
		case KModeNormalBP, KModeStepBP:
			grad := make([]float32, len(output))
			for i := range grad {
				grad[i] = output[i] - sample.Target[i]
			}
			net.StepBackward(state, grad)
			opt.Step(net, ARCLearningRate)
		case KModeTween, KModeTweenChain, KModeStepTween, KModeStepTweenChain:
			ts.TweenStep(net, sample.Input, argmaxInternal(sample.Target), len(sample.Target), ARCLearningRate)
		}

		if acc >= 100 {
			solvedInTrain[sample.TaskIndex] = true
		}
	}
	result.TrainTasksSolved = len(solvedInTrain)

	result.TrainTimeSec = time.Since(start).Seconds()

	// Eval Phase
	evalTotalAcc := 0.0
	evalCount := 0
	for _, task := range evalTasks {
		// Adaptation
		if ts != nil {
			for k := 0; k < 3; k++ {
				for _, pair := range task.Train {
					input := encodeARCGrid(pair.Input)
					target := encodeARCGrid(pair.Output)
					ts.TweenStep(net, input, argmaxInternal(target), len(target), ARCLearningRate)
				}
			}
		}

		// Inference on Test
		taskSolved := true
		for _, pair := range task.Test {
			input := encodeARCGrid(pair.Input)
			target := encodeARCGrid(pair.Output)
			state.SetInput(input)
			net.StepForward(state)
			output := state.GetOutput()
			acc := calculateARCKMeansPixelAccuracy(output, KMeansSample{Target: target, Height: len(pair.Output), Width: len(pair.Output[0])})
			evalTotalAcc += acc
			evalCount++
			if acc < 100 {
				taskSolved = false
			}

			// Add to deviation metrics
			// For ARC, we can treat the grid as multiple values or a single quality metric.
			// The EvaluatePrediction function expects scalars. Let's use the average pixel accuracy for the grid as the "actual" vs 100 as "expected".
			// Or better: iterate over all pixels and add them to metrics.
			for r := 0; r < len(pair.Output); r++ {
				for c := 0; c < len(pair.Output[0]); c++ {
					idx := r*ARCMaxGridSize + c
					pred := float64(output[idx] * 9.0)
					targetVal := float64(target[idx] * 9.0)
					res := nn.EvaluatePrediction(evalCount, targetVal, pred)
					result.DevMetrics.UpdateMetrics(res)
				}
			}
		}
		if taskSolved {
			result.EvalTasksSolved++
		}
	}
	result.DevMetrics.ComputeFinalMetrics()
	if evalCount > 0 {
		result.EvalAccuracy = evalTotalAcc / float64(evalCount)
	}

	calculateKMeansSummaryMetrics(result)
	return result
}

// Utilities from arc_benchmark.go
func loadARCTasksInternal(dir string, limit int) ([]*KMeansARCTask, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var tasks []*KMeansARCTask
	for i, f := range files {
		if i >= limit {
			break
		}
		if filepath.Ext(f.Name()) != ".json" {
			continue
		}
		data, _ := os.ReadFile(filepath.Join(dir, f.Name()))
		var task KMeansARCTask
		json.Unmarshal(data, &task)
		task.ID = f.Name()
		tasks = append(tasks, &task)
	}
	return tasks, nil
}

func createSequentialKMeansSamples(tasks []*KMeansARCTask) []KMeansSample {
	var samples []KMeansSample
	for i, task := range tasks {
		for _, pair := range task.Train {
			samples = append(samples, KMeansSample{
				Input:     encodeARCGrid(pair.Input),
				Target:    encodeARCGrid(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskIndex: i,
			})
		}
	}
	return samples
}

func encodeARCGrid(grid [][]int) []float32 {
	flat := make([]float32, ARCInputSize)
	for r := range grid {
		for c, val := range grid[r] {
			if r < ARCMaxGridSize && c < ARCMaxGridSize {
				flat[r*ARCMaxGridSize+c] = float32(val) / 9.0
			}
		}
	}
	return flat
}

func calculateARCKMeansPixelAccuracy(output []float32, sample KMeansSample) float64 {
	correct, total := 0, 0
	for r := 0; r < sample.Height; r++ {
		for c := 0; c < sample.Width; c++ {
			idx := r*ARCMaxGridSize + c
			pred := int(math.Round(float64(output[idx] * 9.0)))
			target := int(math.Round(float64(sample.Target[idx] * 9.0)))
			if pred == target {
				correct++
			}
			total++
		}
	}
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}

func argmaxInternal(v []float32) int {
	max := float32(-1.0)
	idx := 0
	for i, x := range v {
		if x > max {
			max, idx = x, i
		}
	}
	return idx
}

func calculateKMeansSummaryMetrics(res *KMeansModeResult) {
	sumAcc := 0.0
	for _, w := range res.Windows {
		if w.Outputs > 0 {
			w.Accuracy = w.TotalPixelAcc / float64(w.Outputs)
		}
		sumAcc += w.Accuracy
	}
	res.AvgTrainAccuracy = sumAcc / float64(len(res.Windows))
	res.ThroughputPerSec = float64(res.TotalOutputs) / res.TrainTimeSec

	variance := 0.0
	aboveThreshold := 0
	for _, w := range res.Windows {
		diff := w.Accuracy - res.AvgTrainAccuracy
		variance += diff * diff
		if w.Accuracy >= 12.0 {
			aboveThreshold++
		}
	}
	res.Stability = math.Max(0, 100-math.Sqrt(variance/float64(len(res.Windows))))
	res.Consistency = float64(aboveThreshold) / float64(len(res.Windows)) * 100
	res.Score = (res.ThroughputPerSec * res.Stability * res.Consistency) / 100000
}

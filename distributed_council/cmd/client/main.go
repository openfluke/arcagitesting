package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"runtime"
	"sync"
	"time"

	. "distributed_council/shared"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	serverAddr := flag.String("server", "localhost:9000", "Server address (ip:port)")
	parallel := flag.Bool("parallel", false, "Use all CPU cores (run multiple agents concurrently)")
	workers := flag.Int("workers", 0, "Number of worker goroutines (0 = NumCPU in parallel mode, 1 in single mode)")
	flag.Parse()

	if flag.NArg() > 0 {
		*serverAddr = flag.Arg(0)
	}

	// Determine worker count
	numWorkers := 1
	if *parallel {
		numWorkers = runtime.NumCPU()
	}
	if *workers > 0 {
		numWorkers = *workers
	}

	hostname, _ := os.Hostname()
	clientID := fmt.Sprintf("%s-%d", hostname, time.Now().UnixNano()%10000)

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   👑 DISTRIBUTED COUNCIL CLIENT - Worker Node                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Printf("   Client ID: %s\n", clientID)
	fmt.Printf("   CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("   Workers: %d %s\n", numWorkers, map[bool]string{true: "(parallel mode)", false: "(single-threaded)"}[numWorkers > 1])
	fmt.Printf("   Connecting to: %s\n\n", *serverAddr)

	conn, err := net.Dial("tcp", *serverAddr)
	if err != nil {
		fmt.Printf("❌ Failed to connect to server: %v\n", err)
		fmt.Println("   Make sure the server is running and the address is correct.")
		return
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)
	var connMu sync.Mutex // Mutex for writing to connection

	SendMessage(conn, Message{
		Type: MsgRegister,
		Payload: MakePayload(RegisterPayload{
			ClientID: clientID,
			Hostname: hostname,
			NumCPU:   numWorkers,
		}),
	})

	fmt.Println("📡 Connected! Waiting for server to start the council...")

	started := false
	var jobsCompleted int
	var jobsMu sync.Mutex

	// Worker pool for parallel processing
	type Job struct {
		Config    AgentConfig
		TrainData []Sample
		EvalTasks []*ARCTask
	}

	jobs := make(chan Job, numWorkers*2)
	results := make(chan AgentResult, numWorkers*2)
	var wg sync.WaitGroup

	// Start worker goroutines
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for job := range jobs {
				result := runCouncilMember(job.Config, job.TrainData, job.EvalTasks)
				results <- result
			}
		}(w)
	}

	// Result sender goroutine
	go func() {
		for result := range results {
			connMu.Lock()
			SendMessage(conn, Message{
				Type: MsgResult,
				Payload: MakePayload(ResultPayload{
					ClientID: clientID,
					Result:   result,
				}),
			})
			connMu.Unlock()

			jobsMu.Lock()
			jobsCompleted++
			count := jobsCompleted
			jobsMu.Unlock()

			fmt.Printf("   ✅ %s: %d tasks solved, %.1f%% accuracy (total: %d done)\n",
				result.Config.Name, result.TasksSolved, result.AvgTrainAccuracy, count)
		}
	}()

	pendingJobs := 0

	for {
		// Request more jobs if we have spare worker capacity
		if started && pendingJobs < numWorkers {
			connMu.Lock()
			SendMessage(conn, Message{Type: MsgRequestJob})
			connMu.Unlock()
		}

		msg, err := ReadMessage(reader)
		if err != nil {
			fmt.Printf("❌ Connection lost: %v\n", err)
			break
		}

		switch msg.Type {
		case MsgStatus:
			payload, _ := ParsePayload[StatusPayload](msg)
			if payload.Message == "registered" {
				fmt.Println("✅ Registered with server")
			} else if payload.Message == "waiting_for_start" {
				time.Sleep(500 * time.Millisecond)
			}

		case MsgStart:
			fmt.Println("\n🚀 Council started! Beginning work...")
			started = true

		case MsgConfig:
			payload, _ := ParsePayload[ConfigPayload](msg)
			config := payload.Config

			// Reconstruct Brains from BrainNames (Brains has json:"-" so doesn't serialize)
			config.Brains = make([]BrainType, len(config.BrainNames))
			for i, name := range config.BrainNames {
				switch name {
				case "MHA":
					config.Brains[i] = BrainMHA
				case "LSTM":
					config.Brains[i] = BrainLSTM
				case "RNN":
					config.Brains[i] = BrainRNN
				case "Dense":
					config.Brains[i] = BrainDense
				}
			}

			fmt.Printf("\n🔧 Queued %s (D=%d, Grid=%dx%d)\n",
				config.Name, config.DModel, config.GridSize, config.GridSize)

			evalTasks := make([]*ARCTask, len(payload.EvalTasks))
			for i, td := range payload.EvalTasks {
				evalTasks[i] = &ARCTask{
					ID:    td.ID,
					Train: td.Train,
					Test:  td.Test,
				}
			}

			pendingJobs++
			jobs <- Job{Config: config, TrainData: payload.TrainData, EvalTasks: evalTasks}
			pendingJobs--

		case MsgNoMoreWork:
			fmt.Printf("\n⏳ No more work, waiting for %d pending jobs...\n", pendingJobs)
			close(jobs)
			wg.Wait()
			close(results)
			time.Sleep(500 * time.Millisecond) // Let results flush

			jobsMu.Lock()
			count := jobsCompleted
			jobsMu.Unlock()
			fmt.Printf("\n🎉 All work complete! Processed %d agents.\n", count)
			return

		case MsgShutdown:
			fmt.Println("\n👋 Server shutting down. Goodbye!")
			close(jobs)
			return
		}

		if !started {
			time.Sleep(500 * time.Millisecond)
		}
	}
}

func runCouncilMember(config AgentConfig, trainSamples []Sample, evalTasks []*ARCTask) AgentResult {
	numWindows := int(TestDuration / WindowDuration)
	windows := make([]TimeWindow, numWindows)

	for i := range windows {
		windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	net := createAgentNetwork(config)
	numLayers := net.TotalLayers()

	state := net.InitStepState(InputSize)
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = config.BudgetScale
	ts.Config.UseChainRule = true

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	totalOutputs := 0

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

		ts.TweenStep(net, sample.Input, Argmax(sample.Target), len(sample.Target), config.LearningRate)
	}

	trainTime := time.Since(start).Seconds()

	for i := range windows {
		if windows[i].Outputs > 0 {
			windows[i].Accuracy = windows[i].TotalPixelAcc / float64(windows[i].Outputs)
		}
	}

	sum := 0.0
	for _, w := range windows {
		sum += w.Accuracy
	}
	avgAcc := sum / float64(len(windows))

	taskResults := make(map[string]struct {
		totalAcc float64
		count    int
	})

	for _, task := range evalTasks {
		for k := 0; k < 5; k++ {
			for _, pair := range task.Train {
				if len(pair.Input) == 0 || len(pair.Output) == 0 {
					continue
				}
				input := EncodeGrid(pair.Input)
				target := EncodeGrid(pair.Output)
				ts.TweenStep(net, input, Argmax(target), len(target), config.LearningRate)
			}
		}

		for _, pair := range task.Test {
			if len(pair.Input) == 0 || len(pair.Output) == 0 {
				continue
			}

			input := EncodeGrid(pair.Input)
			target := EncodeGrid(pair.Output)

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

	inputLayer := nn.InitDenseLayer(InputSize, config.DModel, nn.ActivationLeakyReLU)
	ScaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	parallelLayer := createAgentHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	mergerInputSize := config.DModel * config.GridSize * config.GridSize
	mergerLayer := nn.InitDenseLayer(mergerInputSize, config.DModel, nn.ActivationLeakyReLU)
	ScaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	outputLayer := nn.InitDenseLayer(config.DModel, InputSize, nn.ActivationSigmoid)
	ScaleWeights(outputLayer.Kernel, InitScale)
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
	initRandomLocal(mha.QWeights, qkScale)
	initRandomLocal(mha.KWeights, qkScale)
	initRandomLocal(mha.VWeights, qkScale)
	initRandomLocal(mha.OutputWeight, outScale)
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
	initRandomLocal(cfg.WeightIH, scale)
	initRandomLocal(cfg.WeightHH, scale)
}

func createDenseBrain(dModel int) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	ScaleWeights(dense.Kernel, InitScale)
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
	initRandomLocal(cfg.WeightIH_i, scale)
	initRandomLocal(cfg.WeightIH_f, scale)
	initRandomLocal(cfg.WeightIH_g, scale)
	initRandomLocal(cfg.WeightIH_o, scale)
	initRandomLocal(cfg.WeightHH_i, scale)
	initRandomLocal(cfg.WeightHH_f, scale)
	initRandomLocal(cfg.WeightHH_g, scale)
	initRandomLocal(cfg.WeightHH_o, scale)
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
				pred := ClampInt(int(math.Round(float64(output[idx])*9.0)), 0, 9)
				exp := ClampInt(int(math.Round(float64(sample.Target[idx])*9.0)), 0, 9)
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

func initRandomLocal(slice []float32, scale float32) {
	for i := range slice {
		slice[i] = (rand.Float32()*2 - 1) * scale
	}
}

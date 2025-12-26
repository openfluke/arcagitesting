//go:build js && wasm

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"syscall/js"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS (matching shared package)
// ═══════════════════════════════════════════════════════════════════════════════

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	InitScale   = float32(0.5)

	TestDuration   = 10 * time.Second
	WindowDuration = 100 * time.Millisecond
)

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES (matching shared package)
// ═══════════════════════════════════════════════════════════════════════════════

type MsgType string

const (
	MsgRegister   MsgType = "register"
	MsgConfig     MsgType = "config"
	MsgResult     MsgType = "result"
	MsgStart      MsgType = "start"
	MsgNoMoreWork MsgType = "no_more_work"
	MsgStatus     MsgType = "status"
	MsgRequestJob MsgType = "request_job"
	MsgShutdown   MsgType = "shutdown"
)

type Message struct {
	Type    MsgType         `json:"type"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

type RegisterPayload struct {
	ClientID string `json:"clientId"`
	Hostname string `json:"hostname"`
	NumCPU   int    `json:"numCpu"`
}

type ConfigPayload struct {
	Config      AgentConfig `json:"config"`
	TrainData   []Sample    `json:"trainData"`
	EvalTasks   []TaskData  `json:"evalTasks"`
	TotalAgents int         `json:"totalAgents"`
}

type TaskData struct {
	ID    string     `json:"id"`
	Train []GridPair `json:"train"`
	Test  []GridPair `json:"test"`
}

type ResultPayload struct {
	ClientID string      `json:"clientId"`
	Result   AgentResult `json:"result"`
}

type StatusPayload struct {
	Message string `json:"message"`
	Ready   bool   `json:"ready,omitempty"`
}

type BrainType int

const (
	BrainMHA BrainType = iota
	BrainLSTM
	BrainRNN
	BrainDense
)

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

type AgentResult struct {
	Config           AgentConfig `json:"config"`
	AvgTrainAccuracy float64     `json:"avgTrainAccuracy"`
	TasksSolved      int         `json:"tasksSolved"`
	SolvedTaskIDs    []string    `json:"solvedTaskIds"`
	Score            float64     `json:"score"`
	ThroughputPerSec float64     `json:"throughputPerSec"`
}

type GridPair struct {
	Input  [][]int `json:"input"`
	Output [][]int `json:"output"`
}

type Sample struct {
	Input     []float32 `json:"input"`
	Target    []float32 `json:"target"`
	Height    int       `json:"height"`
	Width     int       `json:"width"`
	TaskID    string    `json:"taskId"`
	TaskIndex int       `json:"taskIndex"`
}

type ARCTask struct {
	ID          string
	Train, Test []GridPair
}

type TimeWindow struct {
	TimeMs        int
	Outputs       int
	TotalPixelAcc float64
	Accuracy      float64
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASM CLIENT STATE
// ═══════════════════════════════════════════════════════════════════════════════

var (
	ws             js.Value
	clientID       string
	running        bool
	runningMu      sync.Mutex
	jobsCompleted  int
	tasksSolved    int
	currentStatus  string
	statusCallback js.Value
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Expose JavaScript API
	js.Global().Set("councilConnect", js.FuncOf(jsConnect))
	js.Global().Set("councilDisconnect", js.FuncOf(jsDisconnect))
	js.Global().Set("councilGetStatus", js.FuncOf(jsGetStatus))
	js.Global().Set("councilSetStatusCallback", js.FuncOf(jsSetStatusCallback))

	fmt.Println("🧠 Council WASM Client loaded")
	updateStatus("ready", "Waiting for connection...")

	// Keep WASM running
	select {}
}

func jsConnect(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return "Error: server URL required"
	}
	serverURL := args[0].String()
	go connect(serverURL)
	return nil
}

func jsDisconnect(this js.Value, args []js.Value) interface{} {
	runningMu.Lock()
	running = false
	runningMu.Unlock()

	if !ws.IsUndefined() && !ws.IsNull() {
		ws.Call("close")
	}
	updateStatus("disconnected", "Disconnected")
	return nil
}

func jsGetStatus(this js.Value, args []js.Value) interface{} {
	return js.ValueOf(map[string]interface{}{
		"running":       running,
		"jobsCompleted": jobsCompleted,
		"tasksSolved":   tasksSolved,
		"status":        currentStatus,
	})
}

func jsSetStatusCallback(this js.Value, args []js.Value) interface{} {
	if len(args) > 0 && args[0].Type() == js.TypeFunction {
		statusCallback = args[0]
	}
	return nil
}

func updateStatus(state, message string) {
	currentStatus = message
	if !statusCallback.IsUndefined() && statusCallback.Type() == js.TypeFunction {
		statusCallback.Invoke(state, message, jobsCompleted, tasksSolved)
	}
	fmt.Printf("[%s] %s\n", state, message)
}

func connect(serverURL string) {
	runningMu.Lock()
	if running {
		runningMu.Unlock()
		return
	}
	running = true
	runningMu.Unlock()

	clientID = fmt.Sprintf("browser-%d-%d", time.Now().UnixNano(), rand.Intn(10000))
	updateStatus("connecting", "Connecting to "+serverURL+"...")

	// Create WebSocket
	ws = js.Global().Get("WebSocket").New(serverURL)

	ws.Set("onopen", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		fmt.Println("📡 WebSocket connected")
		updateStatus("connected", "Connected! Registering...")

		// Send registration
		regPayload, _ := json.Marshal(RegisterPayload{
			ClientID: clientID,
			Hostname: "browser",
			NumCPU:   1,
		})
		sendMessage(Message{
			Type:    MsgRegister,
			Payload: regPayload,
		})
		return nil
	}))

	ws.Set("onmessage", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		data := args[0].Get("data").String()
		go handleMessage(data)
		return nil
	}))

	ws.Set("onclose", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		runningMu.Lock()
		running = false
		runningMu.Unlock()
		updateStatus("disconnected", "Connection closed")
		return nil
	}))

	ws.Set("onerror", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		updateStatus("error", "WebSocket error")
		return nil
	}))
}

func sendMessage(msg Message) {
	data, err := json.Marshal(msg)
	if err != nil {
		fmt.Println("Marshal error:", err)
		return
	}
	if !ws.IsUndefined() && !ws.IsNull() {
		ws.Call("send", string(data))
	}
}

var started bool

func handleMessage(data string) {
	var msg Message
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		fmt.Println("Parse error:", err)
		return
	}

	switch msg.Type {
	case MsgStatus:
		var payload StatusPayload
		json.Unmarshal(msg.Payload, &payload)
		if payload.Message == "registered" {
			updateStatus("registered", "Registered! Waiting for council to start...")
		} else if payload.Message == "waiting_for_start" {
			// Request again after delay
			time.Sleep(500 * time.Millisecond)
			sendMessage(Message{Type: MsgRequestJob})
		}

	case MsgStart:
		started = true
		updateStatus("running", "Council started! Processing jobs...")
		sendMessage(Message{Type: MsgRequestJob})

	case MsgConfig:
		var payload ConfigPayload
		if err := json.Unmarshal(msg.Payload, &payload); err != nil {
			fmt.Println("Config parse error:", err)
			return
		}

		config := payload.Config
		// Reconstruct Brains from BrainNames
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

		updateStatus("working", fmt.Sprintf("Training %s (D=%d)...", config.Name, config.DModel))

		// Convert eval tasks
		evalTasks := make([]*ARCTask, len(payload.EvalTasks))
		for i, td := range payload.EvalTasks {
			evalTasks[i] = &ARCTask{
				ID:    td.ID,
				Train: td.Train,
				Test:  td.Test,
			}
		}

		// Run the agent
		result := runCouncilMember(config, payload.TrainData, evalTasks)

		// Send result
		resultPayload, _ := json.Marshal(ResultPayload{
			ClientID: clientID,
			Result:   result,
		})
		sendMessage(Message{
			Type:    MsgResult,
			Payload: resultPayload,
		})

		jobsCompleted++
		tasksSolved += result.TasksSolved
		updateStatus("running", fmt.Sprintf("Completed %d jobs, %d tasks solved", jobsCompleted, tasksSolved))

		// Request next job
		sendMessage(Message{Type: MsgRequestJob})

	case MsgNoMoreWork:
		updateStatus("complete", fmt.Sprintf("No more work! Completed %d jobs, %d tasks solved", jobsCompleted, tasksSolved))

	case MsgShutdown:
		runningMu.Lock()
		running = false
		runningMu.Unlock()
		updateStatus("shutdown", "Server shutting down")
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// AGENT LOGIC (from cmd/client/main.go)
// ═══════════════════════════════════════════════════════════════════════════════

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

		ts.TweenStep(net, sample.Input, argmax(sample.Target), len(sample.Target), config.LearningRate)
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
				input := encodeGrid(pair.Input)
				target := encodeGrid(pair.Output)
				ts.TweenStep(net, input, argmax(target), len(target), config.LearningRate)
			}
		}

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
	scaleWeights(inputLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	parallelLayer := createAgentHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	mergerInputSize := config.DModel * config.GridSize * config.GridSize
	mergerLayer := nn.InitDenseLayer(mergerInputSize, config.DModel, nn.ActivationLeakyReLU)
	scaleWeights(mergerLayer.Kernel, InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

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

func createDenseBrain(dModel int) nn.LayerConfig {
	dense := nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU)
	scaleWeights(dense.Kernel, InitScale)
	return dense
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

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

func encodeGrid(grid [][]int) []float32 {
	encoded := make([]float32, InputSize)
	for r := 0; r < len(grid) && r < MaxGridSize; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize; c++ {
			encoded[r*MaxGridSize+c] = float32(grid[r][c]) / 9.0
		}
	}
	return encoded
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

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

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

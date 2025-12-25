package shared

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const (
	MaxGridSize = 30
	InputSize   = MaxGridSize * MaxGridSize // 900
	NumTasks    = 400
	InitScale   = float32(0.5)

	// Council settings
	CouncilSize    = 1000
	TestDuration   = 10 * time.Second
	WindowDuration = 100 * time.Millisecond

	// Network ports
	TCPPort  = ":9000"
	HTTPPort = ":8080"
)

// ═══════════════════════════════════════════════════════════════════════════════
// PROTOCOL MESSAGE TYPES
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
	Message      string `json:"message"`
	Connected    int    `json:"connected,omitempty"`
	Assigned     int    `json:"assigned,omitempty"`
	Completed    int    `json:"completed,omitempty"`
	UniqueSolved int    `json:"uniqueSolved,omitempty"`
	Ready        bool   `json:"ready,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA TYPES
// ═══════════════════════════════════════════════════════════════════════════════

type BrainType int

const (
	BrainMHA BrainType = iota
	BrainLSTM
	BrainRNN
	BrainDense
)

var BrainTypeNames = []string{"MHA", "LSTM", "RNN", "Dense"}

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

type DiscoveryPoint struct {
	AgentNum          int `json:"agentNum"`
	CumulativeUnique  int `json:"cumulativeUnique"`
	NewTasksThisAgent int `json:"newTasksThisAgent"`
}

type CouncilResults struct {
	CouncilSize       int              `json:"councilSize"`
	TopExperts        []AgentResult    `json:"topExperts"`
	CollectiveTasks   []string         `json:"collectiveTasks"`
	CollectiveCount   int              `json:"collectiveCount"`
	DiscoveryTimeline []DiscoveryPoint `json:"discoveryTimeline"`
	CouncilEfficiency float64          `json:"councilEfficiency"`
	Timestamp         string           `json:"timestamp"`
	Duration          string           `json:"duration"`
	WorkerCount       int              `json:"workerCount"`
}

type ARCTask struct {
	ID          string
	Train, Test []GridPair
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

type TimeWindow struct {
	TimeMs        int
	Outputs       int
	TotalPixelAcc float64
	Accuracy      float64
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROTOCOL HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

func SendMessage(conn net.Conn, msg Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal error: %w", err)
	}
	_, err = conn.Write(append(data, '\n'))
	return err
}

func ReadMessage(reader *bufio.Reader) (*Message, error) {
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var msg Message
	if err := json.Unmarshal(line, &msg); err != nil {
		return nil, fmt.Errorf("unmarshal error: %w", err)
	}
	return &msg, nil
}

func MakePayload(v interface{}) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

func ParsePayload[T any](msg *Message) (*T, error) {
	var v T
	if err := json.Unmarshal(msg.Payload, &v); err != nil {
		return nil, err
	}
	return &v, nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

func EncodeGrid(grid [][]int) []float32 {
	encoded := make([]float32, InputSize)
	for r := 0; r < len(grid) && r < MaxGridSize; r++ {
		for c := 0; c < len(grid[r]) && c < MaxGridSize; c++ {
			encoded[r*MaxGridSize+c] = float32(grid[r][c]) / 9.0
		}
	}
	return encoded
}

func ClampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func Argmax(s []float32) int {
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

func ScaleWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

func InitRandom(slice []float32, scale float32) {
	for i := range slice {
		slice[i] = (float32(i%100)/50 - 1) * scale // Deterministic for shared state
	}
}

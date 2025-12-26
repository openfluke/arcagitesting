package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	. "distributed_council/shared"

	"golang.org/x/net/websocket"
)

var maxAgents int

type CouncilServer struct {
	mu sync.RWMutex

	clients     map[string]*ClientConnection
	wsClients   map[string]*WSClientConnection
	clientOrder []string

	configs       []AgentConfig
	trainSamples  []Sample
	evalTasks     []*ARCTask
	nextConfigIdx int
	started       bool
	stopped       bool

	results         []AgentResult
	collectiveTasks map[string]bool
	completedCount  int

	discoveryTimeline []DiscoveryPoint
	startTime         time.Time
}

type ClientConnection struct {
	ID       string
	Hostname string
	NumCPU   int
	Conn     net.Conn
	Reader   *bufio.Reader
	Assigned int
	Done     int
}

type WSClientConnection struct {
	ID       string
	Hostname string
	NumCPU   int
	Conn     *websocket.Conn
	Mu       sync.Mutex
	Assigned int
	Done     int
}

func NewCouncilServer() *CouncilServer {
	return &CouncilServer{
		clients:         make(map[string]*ClientConnection),
		wsClients:       make(map[string]*WSClientConnection),
		collectiveTasks: make(map[string]bool),
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	flag.IntVar(&maxAgents, "max-agents", CouncilSize, "Maximum agents (0=unlimited)")
	flag.Parse()

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   👑 DISTRIBUTED COUNCIL SERVER - Network Orchestrator                                   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	if maxAgents == 0 {
		fmt.Println("   Mode: UNLIMITED (continuous operation)")
	} else {
		fmt.Printf("   Mode: %d agents max\n", maxAgents)
	}

	server := NewCouncilServer()

	// Load ARC-AGI data - look in parent directory
	arcPath := "../ARC-AGI/data/training"
	if _, err := os.Stat(arcPath); os.IsNotExist(err) {
		arcPath = "../../ARC-AGI/data/training" // Also try two levels up
	}

	fmt.Println("\n📦 Loading ARC-AGI training data...")
	trainTasks, err := loadARCTasks(arcPath, NumTasks)
	if err != nil {
		fmt.Printf("❌ Failed to load training tasks: %v\n", err)
		fmt.Println("   Make sure ARC-AGI data is available in the parent directory")
		return
	}

	evalPath := "../ARC-AGI/data/evaluation"
	if _, err := os.Stat(evalPath); os.IsNotExist(err) {
		evalPath = "../../ARC-AGI/data/evaluation"
	}

	evalTasks, err := loadARCTasks(evalPath, 400)
	if err != nil {
		fmt.Printf("❌ Failed to load eval tasks: %v\n", err)
		return
	}

	server.trainSamples = createSequentialSamples(trainTasks)
	server.evalTasks = evalTasks
	fmt.Printf("✅ Loaded %d training samples, %d eval tasks\n", len(server.trainSamples), len(server.evalTasks))

	councilSize := maxAgents
	if councilSize == 0 {
		councilSize = 10000 // Pre-generate a large batch for unlimited mode
	}
	fmt.Printf("👑 Generating %d agent configurations...\n", councilSize)
	server.configs = generateAgentConfigs(councilSize)
	server.results = make([]AgentResult, councilSize)

	go server.runTCPServer()
	go server.runWSServer()

	server.runHTTPServer()
}

func (s *CouncilServer) runTCPServer() {
	listener, err := net.Listen("tcp", TCPPort)
	if err != nil {
		fmt.Printf("❌ Failed to start TCP server: %v\n", err)
		return
	}
	defer listener.Close()

	fmt.Printf("🌐 TCP Server listening on %s\n", TCPPort)

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}
		go s.handleClient(conn)
	}
}

// WebSocket server for browser clients
func (s *CouncilServer) runWSServer() {
	http.Handle("/ws", websocket.Handler(s.handleWSClient))

	fmt.Printf("🌐 WebSocket Server listening on %s\n", WSPort)

	if err := http.ListenAndServe(WSPort, nil); err != nil {
		fmt.Printf("❌ Failed to start WebSocket server: %v\n", err)
	}
}

func (s *CouncilServer) handleWSClient(ws *websocket.Conn) {
	clientID := ""

	defer func() {
		ws.Close()
		if clientID != "" {
			s.mu.Lock()
			delete(s.wsClients, clientID)
			s.mu.Unlock()
			fmt.Printf("📤 Browser client disconnected: %s\n", clientID)
		}
	}()

	for {
		var data string
		if err := websocket.Message.Receive(ws, &data); err != nil {
			if err != io.EOF {
				fmt.Printf("⚠️ WS read error from %s: %v\n", clientID, err)
			}
			return
		}

		var msg Message
		if err := json.Unmarshal([]byte(data), &msg); err != nil {
			fmt.Printf("⚠️ WS parse error: %v\n", err)
			continue
		}

		switch msg.Type {
		case MsgRegister:
			payload, _ := ParsePayload[RegisterPayload](&msg)
			clientID = payload.ClientID

			s.mu.Lock()
			s.wsClients[clientID] = &WSClientConnection{
				ID:       clientID,
				Hostname: payload.Hostname,
				NumCPU:   payload.NumCPU,
				Conn:     ws,
			}
			s.clientOrder = append(s.clientOrder, clientID)
			alreadyStarted := s.started
			s.mu.Unlock()

			fmt.Printf("📥 Browser client registered: %s (%s)\n", clientID, payload.Hostname)

			s.sendWSMessage(ws, Message{
				Type:    MsgStatus,
				Payload: MakePayload(StatusPayload{Message: "registered", Ready: !alreadyStarted}),
			})

			// If council already started, tell the new client to start working
			if alreadyStarted {
				s.sendWSMessage(ws, Message{Type: MsgStart})
			}

		case MsgRequestJob:
			s.mu.Lock()
			if !s.started {
				s.mu.Unlock()
				s.sendWSMessage(ws, Message{
					Type:    MsgStatus,
					Payload: MakePayload(StatusPayload{Message: "waiting_for_start"}),
				})
				continue
			}

			// Check if we need more configs (unlimited mode)
			if s.nextConfigIdx >= len(s.configs) {
				if maxAgents == 0 {
					// Generate more configs for unlimited mode
					newConfigs := generateAgentConfigs(1000)
					for i := range newConfigs {
						newConfigs[i].ID = len(s.configs) + i
						newConfigs[i].Name = fmt.Sprintf("Agent-%d", newConfigs[i].ID)
					}
					s.configs = append(s.configs, newConfigs...)
					s.results = append(s.results, make([]AgentResult, 1000)...)
				} else {
					s.mu.Unlock()
					s.sendWSMessage(ws, Message{Type: MsgNoMoreWork})
					continue
				}
			}

			configIdx := s.nextConfigIdx
			s.nextConfigIdx++

			if client, ok := s.wsClients[clientID]; ok {
				client.Assigned++
			}
			s.mu.Unlock()

			evalTaskData := make([]TaskData, len(s.evalTasks))
			for i, t := range s.evalTasks {
				evalTaskData[i] = TaskData{
					ID:    t.ID,
					Train: t.Train,
					Test:  t.Test,
				}
			}

			s.sendWSMessage(ws, Message{
				Type: MsgConfig,
				Payload: MakePayload(ConfigPayload{
					Config:      s.configs[configIdx],
					TrainData:   s.trainSamples,
					EvalTasks:   evalTaskData,
					TotalAgents: len(s.configs),
				}),
			})

			fmt.Printf("📋 Assigned Agent-%d to browser %s (%d/%d)\n", configIdx, clientID, s.nextConfigIdx, len(s.configs))

		case MsgResult:
			payload, _ := ParsePayload[ResultPayload](&msg)

			s.mu.Lock()
			idx := payload.Result.Config.ID
			if idx >= 0 && idx < len(s.results) {
				s.results[idx] = payload.Result
				s.completedCount++

				newTasks := 0
				for _, taskID := range payload.Result.SolvedTaskIDs {
					if !s.collectiveTasks[taskID] {
						s.collectiveTasks[taskID] = true
						newTasks++
					}
				}

				if s.completedCount%10 == 0 {
					s.discoveryTimeline = append(s.discoveryTimeline, DiscoveryPoint{
						AgentNum:          s.completedCount,
						CumulativeUnique:  len(s.collectiveTasks),
						NewTasksThisAgent: newTasks,
					})
				}

				if client, ok := s.wsClients[clientID]; ok {
					client.Done++
				}

				fmt.Printf("✅ Result from browser %s: Agent-%d solved %d tasks (%d complete)\n",
					clientID, idx, payload.Result.TasksSolved, s.completedCount)
			}

			allDone := maxAgents > 0 && s.completedCount >= maxAgents
			s.mu.Unlock()

			if allDone {
				s.finalizeResults()
			}
		}
	}
}

func (s *CouncilServer) sendWSMessage(ws *websocket.Conn, msg Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return websocket.Message.Send(ws, string(data))
}

func (s *CouncilServer) handleClient(conn net.Conn) {
	reader := bufio.NewReader(conn)
	clientID := ""

	defer func() {
		conn.Close()
		if clientID != "" {
			s.mu.Lock()
			delete(s.clients, clientID)
			s.mu.Unlock()
			fmt.Printf("📤 Client disconnected: %s\n", clientID)
		}
	}()

	for {
		msg, err := ReadMessage(reader)
		if err != nil {
			if err != io.EOF {
				fmt.Printf("⚠️ Read error from %s: %v\n", clientID, err)
			}
			return
		}

		switch msg.Type {
		case MsgRegister:
			payload, _ := ParsePayload[RegisterPayload](msg)
			clientID = payload.ClientID

			s.mu.Lock()
			s.clients[clientID] = &ClientConnection{
				ID:       clientID,
				Hostname: payload.Hostname,
				NumCPU:   payload.NumCPU,
				Conn:     conn,
				Reader:   reader,
			}
			s.clientOrder = append(s.clientOrder, clientID)
			alreadyStarted := s.started
			s.mu.Unlock()

			fmt.Printf("📥 Client registered: %s (%s, %d CPUs)\n", clientID, payload.Hostname, payload.NumCPU)

			SendMessage(conn, Message{
				Type:    MsgStatus,
				Payload: MakePayload(StatusPayload{Message: "registered", Ready: !alreadyStarted}),
			})

			// If council already started, tell the new client to start working
			if alreadyStarted {
				SendMessage(conn, Message{Type: MsgStart})
			}

		case MsgRequestJob:
			s.mu.Lock()
			if !s.started {
				s.mu.Unlock()
				SendMessage(conn, Message{
					Type:    MsgStatus,
					Payload: MakePayload(StatusPayload{Message: "waiting_for_start"}),
				})
				continue
			}

			if s.nextConfigIdx >= len(s.configs) {
				s.mu.Unlock()
				SendMessage(conn, Message{Type: MsgNoMoreWork})
				continue
			}

			configIdx := s.nextConfigIdx
			s.nextConfigIdx++

			if client, ok := s.clients[clientID]; ok {
				client.Assigned++
			}
			s.mu.Unlock()

			evalTaskData := make([]TaskData, len(s.evalTasks))
			for i, t := range s.evalTasks {
				evalTaskData[i] = TaskData{
					ID:    t.ID,
					Train: t.Train,
					Test:  t.Test,
				}
			}

			SendMessage(conn, Message{
				Type: MsgConfig,
				Payload: MakePayload(ConfigPayload{
					Config:      s.configs[configIdx],
					TrainData:   s.trainSamples,
					EvalTasks:   evalTaskData,
					TotalAgents: len(s.configs),
				}),
			})

			fmt.Printf("📋 Assigned Agent-%d to %s (%d/%d)\n", configIdx, clientID, s.nextConfigIdx, len(s.configs))

		case MsgResult:
			payload, _ := ParsePayload[ResultPayload](msg)

			s.mu.Lock()
			idx := payload.Result.Config.ID
			if idx >= 0 && idx < len(s.results) {
				s.results[idx] = payload.Result
				s.completedCount++

				newTasks := 0
				for _, taskID := range payload.Result.SolvedTaskIDs {
					if !s.collectiveTasks[taskID] {
						s.collectiveTasks[taskID] = true
						newTasks++
					}
				}

				if s.completedCount%10 == 0 {
					s.discoveryTimeline = append(s.discoveryTimeline, DiscoveryPoint{
						AgentNum:          s.completedCount,
						CumulativeUnique:  len(s.collectiveTasks),
						NewTasksThisAgent: newTasks,
					})
				}

				if client, ok := s.clients[clientID]; ok {
					client.Done++
				}

				fmt.Printf("✅ Result from %s: Agent-%d solved %d tasks (%d/%d complete)\n",
					clientID, idx, payload.Result.TasksSolved, s.completedCount, len(s.configs))
			}

			allDone := s.completedCount >= len(s.configs)
			s.mu.Unlock()

			if allDone {
				s.finalizeResults()
			}
		}
	}
}

func (s *CouncilServer) startCouncil() {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return
	}
	s.started = true
	s.startTime = time.Now()

	// Notify TCP clients
	for _, client := range s.clients {
		SendMessage(client.Conn, Message{Type: MsgStart})
	}
	// Notify WebSocket clients
	for _, client := range s.wsClients {
		s.sendWSMessage(client.Conn, Message{Type: MsgStart})
	}
	s.mu.Unlock()

	fmt.Println("\n🚀 COUNCIL STARTED - Distributing work to clients...")
}

func (s *CouncilServer) finalizeResults() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return
	}
	s.stopped = true

	totalTime := time.Since(s.startTime)

	sort.Slice(s.results, func(i, j int) bool {
		if s.results[i].TasksSolved != s.results[j].TasksSolved {
			return s.results[i].TasksSolved > s.results[j].TasksSolved
		}
		return s.results[i].Score > s.results[j].Score
	})

	var collectiveTasksList []string
	for taskID := range s.collectiveTasks {
		collectiveTasksList = append(collectiveTasksList, taskID)
	}
	sort.Strings(collectiveTasksList)

	topN := 20
	if len(s.results) < topN {
		topN = len(s.results)
	}

	output := &CouncilResults{
		CouncilSize:       CouncilSize,
		TopExperts:        s.results[:topN],
		CollectiveTasks:   collectiveTasksList,
		CollectiveCount:   len(collectiveTasksList),
		DiscoveryTimeline: s.discoveryTimeline,
		CouncilEfficiency: float64(len(collectiveTasksList)) / float64(CouncilSize) * 100,
		Timestamp:         time.Now().Format(time.RFC3339),
		Duration:          totalTime.String(),
		WorkerCount:       len(s.clients),
	}

	data, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile("distributed_council_results.json", data, 0644)

	fmt.Println("\n" + "═══════════════════════════════════════════════════════════════════")
	fmt.Println("✅ COUNCIL COMPLETE!")
	fmt.Printf("   Duration: %s\n", totalTime)
	fmt.Printf("   Clients: %d\n", len(s.clients))
	fmt.Printf("   Unique Tasks Solved: %d\n", len(collectiveTasksList))
	fmt.Println("   Results saved to distributed_council_results.json")
	fmt.Println("═══════════════════════════════════════════════════════════════════")

	for i := 0; i < 5 && i < len(s.results); i++ {
		r := s.results[i]
		fmt.Printf("   #%d: %s - %d tasks solved\n", i+1, r.Config.Name, r.TasksSolved)
	}
}

// HTTP Server
func (s *CouncilServer) runHTTPServer() {
	http.HandleFunc("/", s.handleHome)
	http.HandleFunc("/api/status", s.handleAPIStatus)
	http.HandleFunc("/api/start", s.handleAPIStart)
	http.HandleFunc("/download/client", s.handleDownloadClient)

	// Serve WASM client files
	http.HandleFunc("/wasm/", func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/wasm/")
		if path == "" {
			path = "index.html"
		}
		filePath := filepath.Join("wasm/static", path)
		if strings.HasSuffix(path, ".wasm") {
			w.Header().Set("Content-Type", "application/wasm")
		}
		http.ServeFile(w, r, filePath)
	})

	fmt.Printf("🌐 Web UI available at http://localhost%s\n", HTTPPort)
	fmt.Println("   → Download client binary: http://localhost" + HTTPPort + "/download/client")
	fmt.Printf("   → Browser client: http://localhost%s/wasm/\n", HTTPPort)
	fmt.Printf("   → WebSocket endpoint: ws://localhost%s\n", WSPort)
	fmt.Println("\n⏳ Waiting for clients to connect. Press 'Start' in the web UI when ready.\n")

	if err := http.ListenAndServe(HTTPPort, nil); err != nil {
		fmt.Printf("❌ HTTP server error: %v\n", err)
	}
}

func (s *CouncilServer) handleHome(w http.ResponseWriter, r *http.Request) {
	tmpl := template.Must(template.New("home").Parse(homeHTML))
	tmpl.Execute(w, nil)
}

func (s *CouncilServer) handleAPIStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	status := struct {
		Started      bool             `json:"started"`
		Stopped      bool             `json:"stopped"`
		Clients      []map[string]any `json:"clients"`
		TotalAgents  int              `json:"totalAgents"`
		Assigned     int              `json:"assigned"`
		Completed    int              `json:"completed"`
		UniqueSolved int              `json:"uniqueSolved"`
		TopResults   []AgentResult    `json:"topResults"`
		Timeline     []DiscoveryPoint `json:"timeline"`
	}{
		Started:      s.started,
		Stopped:      s.stopped,
		TotalAgents:  len(s.configs),
		Assigned:     s.nextConfigIdx,
		Completed:    s.completedCount,
		UniqueSolved: len(s.collectiveTasks),
		Timeline:     s.discoveryTimeline,
	}

	for _, c := range s.clients {
		status.Clients = append(status.Clients, map[string]any{
			"id":       c.ID,
			"hostname": c.Hostname,
			"numCpu":   c.NumCPU,
			"assigned": c.Assigned,
			"done":     c.Done,
		})
	}

	// Include WebSocket clients
	for _, c := range s.wsClients {
		status.Clients = append(status.Clients, map[string]any{
			"id":       c.ID,
			"hostname": c.Hostname + " (browser)",
			"numCpu":   c.NumCPU,
			"assigned": c.Assigned,
			"done":     c.Done,
		})
	}

	sorted := make([]AgentResult, 0)
	for _, r := range s.results {
		if r.Config.ID > 0 || r.TasksSolved > 0 {
			sorted = append(sorted, r)
		}
	}
	s.mu.RUnlock()

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].TasksSolved > sorted[j].TasksSolved
	})
	if len(sorted) > 5 {
		sorted = sorted[:5]
	}
	status.TopResults = sorted

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (s *CouncilServer) handleAPIStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	s.mu.RLock()
	numClients := len(s.clients) + len(s.wsClients)
	s.mu.RUnlock()

	if numClients == 0 {
		http.Error(w, "No clients connected", 400)
		return
	}

	s.startCouncil()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "started"})
}

func (s *CouncilServer) handleDownloadClient(w http.ResponseWriter, r *http.Request) {
	data, err := os.ReadFile("council_client_linux")
	if err != nil {
		http.Error(w, "Client binary not found. Run 'make client' first.", 404)
		return
	}

	w.Header().Set("Content-Disposition", "attachment; filename=council_client_linux")
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(data)
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
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
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

func createSequentialSamples(tasks []*ARCTask) []Sample {
	var samples []Sample
	for i, task := range tasks {
		for _, pair := range task.Train {
			if len(pair.Output) == 0 || len(pair.Output[0]) == 0 {
				continue
			}
			samples = append(samples, Sample{
				Input:     EncodeGrid(pair.Input),
				Target:    EncodeGrid(pair.Output),
				Height:    len(pair.Output),
				Width:     len(pair.Output[0]),
				TaskID:    task.ID,
				TaskIndex: i,
			})
		}
	}
	return samples
}

func generateAgentConfigs(count int) []AgentConfig {
	configs := make([]AgentConfig, count)

	gridSizes := []int{1, 2}
	dModels := []int{16, 32}
	numHeads := []int{2, 4, 8}

	for i := 0; i < count; i++ {
		gridSize := gridSizes[rand.Intn(len(gridSizes))]
		numBrains := gridSize * gridSize

		brains := make([]BrainType, numBrains)
		brainNames := make([]string, numBrains)
		for b := 0; b < numBrains; b++ {
			brainType := BrainType(rand.Intn(4))
			brains[b] = brainType
			brainNames[b] = BrainTypeNames[brainType]
		}

		dModel := dModels[rand.Intn(len(dModels))]
		heads := numHeads[rand.Intn(len(numHeads))]
		for dModel%heads != 0 {
			heads = numHeads[rand.Intn(len(numHeads))]
		}

		configs[i] = AgentConfig{
			ID:           i,
			Name:         fmt.Sprintf("Agent-%d", i),
			GridSize:     gridSize,
			DModel:       dModel,
			NumHeads:     heads,
			LearningRate: float32(0.001 + rand.Float64()*0.099),
			BudgetScale:  float32(0.5 + rand.Float64()*0.5),
			Brains:       brains,
			BrainNames:   brainNames,
		}
	}

	return configs
}

const homeHTML = `<!DOCTYPE html>
<html>
<head>
    <title>Council Server - Distributed Neural Network Training</title>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #f39c12, #e74c3c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { color: #f39c12; margin-bottom: 16px; font-size: 1.3em; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .stat:last-child { border: none; }
        .stat-value { font-weight: bold; color: #3498db; }
        .btn {
            background: linear-gradient(45deg, #f39c12, #e74c3c);
            color: white;
            border: none;
            padding: 16px 48px;
            font-size: 1.2em;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 20px rgba(243, 156, 18, 0.4);
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 30px rgba(243, 156, 18, 0.6); }
        .btn:disabled { background: #666; cursor: not-allowed; transform: none; box-shadow: none; }
        .client-list { max-height: 200px; overflow-y: auto; }
        .client {
            background: rgba(52, 152, 219, 0.2);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .progress-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            background: linear-gradient(90deg, #2ecc71, #27ae60);
            height: 100%;
            transition: width 0.3s;
        }
        .download-link {
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 10px;
            margin-right: 10px;
        }
        .browser-link {
            display: inline-block;
            background: linear-gradient(45deg, #9b59b6, #8e44ad);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 10px;
        }
        .browser-link:hover { opacity: 0.9; }
        .results-table { width: 100%; border-collapse: collapse; }
        .results-table th, .results-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .results-table th { color: #f39c12; }
        #status { text-align: center; font-size: 1.2em; padding: 10px; }
        .running { color: #2ecc71; }
        .waiting { color: #f39c12; }
        .complete { color: #3498db; }
    </style>
</head>
<body>
    <div class="container">
        <h1>👑 Distributed Council Server</h1>
        <div id="status" class="waiting">⏳ Waiting for clients...</div>
        <button id="startBtn" class="btn" disabled>🚀 Start Council</button>
        <div class="grid">
            <div class="card">
                <h2>📊 Progress</h2>
                <div class="stat"><span>Total Agents:</span><span class="stat-value" id="totalAgents">0</span></div>
                <div class="stat"><span>Assigned:</span><span class="stat-value" id="assigned">0</span></div>
                <div class="stat"><span>Completed:</span><span class="stat-value" id="completed">0</span></div>
                <div class="stat"><span>Unique Tasks Solved:</span><span class="stat-value" id="uniqueSolved">0</span></div>
                <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
            </div>
            <div class="card">
                <h2>🖥️ Connected Clients</h2>
                <div class="client-list" id="clientList"><p style="color:#888">No clients connected yet</p></div>
                <a href="/download/client" class="download-link">⬇️ Download Client (Linux)</a>
                <a href="/wasm/" class="browser-link">🌐 Join via Browser</a>
            </div>
        </div>
        <div class="card">
            <h2>🏆 Top Results</h2>
            <table class="results-table">
                <thead><tr><th>Rank</th><th>Agent</th><th>Architecture</th><th>Tasks Solved</th></tr></thead>
                <tbody id="resultsBody"><tr><td colspan="4" style="color:#888">Waiting for results...</td></tr></tbody>
            </table>
        </div>
    </div>
    <script>
        const startBtn = document.getElementById('startBtn');
        let started = false;
        startBtn.addEventListener('click', async () => {
            if (started) return;
            const res = await fetch('/api/start', { method: 'POST' });
            if (res.ok) { started = true; startBtn.disabled = true; startBtn.textContent = '🔄 Running...'; }
        });
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('totalAgents').textContent = data.totalAgents;
                document.getElementById('assigned').textContent = data.assigned;
                document.getElementById('completed').textContent = data.completed;
                document.getElementById('uniqueSolved').textContent = data.uniqueSolved;
                const pct = data.totalAgents > 0 ? (data.completed / data.totalAgents * 100) : 0;
                document.getElementById('progressFill').style.width = pct + '%';
                const statusEl = document.getElementById('status');
                if (data.stopped) {
                    statusEl.className = 'complete';
                    statusEl.textContent = '✅ Council Complete! Results saved.';
                    startBtn.textContent = '✅ Complete';
                } else if (data.started) {
                    statusEl.className = 'running';
                    statusEl.textContent = '🔄 Running... ' + data.completed + '/' + data.totalAgents;
                    started = true; startBtn.disabled = true; startBtn.textContent = '🔄 Running...';
                } else {
                    statusEl.className = 'waiting';
                    statusEl.textContent = data.clients && data.clients.length > 0
                        ? '✅ ' + data.clients.length + ' client(s) connected. Ready to start!'
                        : '⏳ Waiting for clients...';
                    startBtn.disabled = !data.clients || data.clients.length === 0;
                }
                const clientList = document.getElementById('clientList');
                if (data.clients && data.clients.length > 0) {
                    clientList.innerHTML = data.clients.map(c =>
                        '<div class="client"><strong>' + c.hostname + '</strong> (' + c.numCpu + ' CPUs)<br>Assigned: ' + c.assigned + ' | Done: ' + c.done + '</div>'
                    ).join('');
                } else { clientList.innerHTML = '<p style="color:#888">No clients connected yet</p>'; }
                const tbody = document.getElementById('resultsBody');
                if (data.topResults && data.topResults.length > 0) {
                    tbody.innerHTML = data.topResults.map((r, i) =>
                        '<tr><td>#' + (i+1) + '</td><td>' + r.config.name + '</td><td>' + r.config.gridSize + 'x' + r.config.gridSize + ' D=' + r.config.dModel + '</td><td><strong>' + r.tasksSolved + '</strong></td></tr>'
                    ).join('');
                }
            } catch (e) { console.error('Status update error:', e); }
        }
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>`

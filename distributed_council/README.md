# Distributed Council 👑

Distribute the "Council of 1000" ARC-AGI benchmark across multiple computers. Now with browser-based workers!

## How It Works

This is a distributed version of `test38_council.go` and follows the same methodology as `arc_benchmark.go`:

1. **Training Phase** (10 seconds): Network cycles through ARC-AGI **training** data samples, learning patterns via StepTweenChain
2. **Evaluation Phase**: For each of the 400 **evaluation** tasks:
   - 5-pass adaptation on the task's train pairs (learning the rule)
   - Test prediction on the task's test pairs
   - "Solved" = 100% pixel accuracy on test

**Data Used:**
- `ARC-AGI/data/training/` → Training samples (for the 10-second training loop)
- `ARC-AGI/data/evaluation/` → Evaluation tasks (for few-shot testing)

## Quick Start

```bash
# Build everything (server + TCP client + WASM)
make all

# Start server (on main machine)
./council_server

# Start client (on remote machines)
./council_client_linux <server-ip>:9000

# Or open browser to contribute compute
# http://<server-ip>:8080/wasm/
```

## Commands

| Command | Description |
|---------|-------------|
| `make all` | Build server, client, and WASM |
| `make server` | Build server only |
| `make client` | Build Linux client (cross-compile) |
| `make wasm` | Build WASM client for browser |
| `make run-server-unlimited` | Run server in unlimited mode |
| `make clean` | Remove binaries |

## Usage

### 1. Start Server
```bash
# Fixed mode (default: 1000 agents)
./council_server

# Unlimited mode (runs forever, late-join supported)
./council_server --max-agents=0

# Custom limit
./council_server --max-agents=5000
```

### 2. Connect Clients

**Option A: Native Client (TCP)**
```bash
wget http://<server-ip>:8080/download/client -O council_client
chmod +x council_client

# Single-threaded
./council_client <server-ip>:9000

# Parallel mode (use all CPUs)
./council_client -parallel <server-ip>:9000
```

**Option B: Browser Client (WebSocket)**
1. Open `http://<server-ip>:8080/wasm/`
2. WebSocket URL auto-detected from current host
3. Click "Start Contributing"

### 3. Start Council
Open `http://<server-ip>:8080` and click **Start Council**

### 4. Results
Results saved to `distributed_council_results.json`

## Architecture

```
distributed_council/
├── cmd/
│   ├── server/main.go   # Server (TCP + WebSocket + HTTP)
│   └── client/main.go   # TCP Client
├── wasm/
│   ├── main.go          # WASM Client (WebSocket)
│   └── static/
│       ├── index.html   # Browser UI
│       ├── client.wasm  # Compiled WASM
│       └── wasm_exec.js # Go WASM runtime
├── shared/types.go      # Protocol & data types
├── Makefile
└── README.md
```

## Network Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 9000 | TCP | Native client communication |
| 8080 | HTTP | Web UI + WASM client + downloads |
| 8081 | WebSocket | Browser client communication |

## Features

- ✅ Configurable max agents (`--max-agents=0` for unlimited)
- ✅ Late-join support (clients can connect after council starts)
- ✅ Browser + native client support simultaneously
- ✅ Real-time progress monitoring via web UI

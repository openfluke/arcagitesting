# Distributed Council 👑

Distribute the "Council of 1000" ARC-AGI benchmark across multiple computers.

## Quick Start

```bash
# Build both binaries
make all

# Start server (on main machine)
./council_server

# Start client (on remote machines)
./council_client_linux <server-ip>:9000
```

## Commands

| Command | Description |
|---------|-------------|
| `make all` | Build server and Linux client |
| `make server` | Build server only |
| `make client` | Build Linux client (cross-compile) |
| `make client-local` | Build client for current platform |
| `make clean` | Remove binaries |
| `make help` | Show all targets |

## Usage

### 1. Start Server
```bash
./council_server
```
- TCP server on `:9000`
- Web UI on `:8080`

### 2. Connect Clients
On each remote machine:
```bash
wget http://<server-ip>:8080/download/client -O council_client
chmod +x council_client

# Single-threaded (default)
./council_client <server-ip>:9000

# Parallel mode (use all CPU cores)
./council_client -parallel <server-ip>:9000

# Custom worker count
./council_client -workers 4 <server-ip>:9000
```

### 3. Start Council
Open `http://<server-ip>:8080` in browser and click **Start Council**

### 4. Results
Results saved to `distributed_council_results.json`

## Architecture

```
distributed_council/
├── cmd/
│   ├── server/main.go   # Server (TCP + HTTP)
│   └── client/main.go   # Client (runs agents)
├── shared/types.go      # Protocol & data types
├── Makefile
└── README.md
```

## Network Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 9000 | TCP | Client-server communication |
| 8080 | HTTP | Web UI + client download |

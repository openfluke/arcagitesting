.PHONY: all server client wasm clean help

all: client server wasm

# Build server (for current platform)
server:
	@echo "🔧 Building council_server..."
	cd cmd/server && go build -o ../../council_server .
	@echo "✅ Server binary: council_server"

# Build Linux client (cross-compile for Linux amd64)
client:
	@echo "🔧 Building council_client_linux..."
	cd cmd/client && GOOS=linux GOARCH=amd64 go build -o ../../council_client_linux .
	@echo "✅ Client binary: council_client_linux"

# Build client for current platform (for testing)
client-local:
	@echo "🔧 Building council_client (local)..."
	cd cmd/client && go build -o ../../council_client .
	@echo "✅ Client binary: council_client"

# Build WASM client for browser
wasm:
	@echo "🔧 Building WASM client..."
	cd wasm && GOOS=js GOARCH=wasm go build -o static/client.wasm .
	@echo "✅ WASM binary: wasm/static/client.wasm"

clean:
	rm -f council_server council_client council_client_linux
	rm -f wasm/static/client.wasm

# Run server
run-server: server
	./council_server

# Run server in unlimited mode
run-server-unlimited: server
	./council_server --max-agents=0

# Run client (local testing)
run-client: client-local
	./council_client localhost:9000

# Show help
help:
	@echo "╔═════════════════════════════════════════════════════════════════╗"
	@echo "║  Distributed Council - Build Targets                            ║"
	@echo "╠═════════════════════════════════════════════════════════════════╣"
	@echo "║  make all              - Build server, client, and WASM         ║"
	@echo "║  make server           - Build server binary                    ║"
	@echo "║  make client           - Build Linux client (cross-compile)     ║"
	@echo "║  make client-local     - Build client for current platform      ║"
	@echo "║  make wasm             - Build WASM client for browser          ║"
	@echo "║  make run-server       - Build and run server                   ║"
	@echo "║  make run-server-unlimited - Run in unlimited mode              ║"
	@echo "║  make run-client       - Build and run client (localhost:9000)  ║"
	@echo "║  make clean            - Remove built binaries                  ║"
	@echo "╠═════════════════════════════════════════════════════════════════╣"
	@echo "║  Usage:                                                         ║"
	@echo "║  1. Run 'make all' to build everything                          ║"
	@echo "║  2. Start server: ./council_server                              ║"
	@echo "║  3. On remote machines: ./council_client_linux <ip>:9000        ║"
	@echo "║  4. In browser: http://localhost:8080/wasm/                     ║"
	@echo "║  5. Open http://localhost:8080 and click 'Start'                ║"
	@echo "╚═════════════════════════════════════════════════════════════════╝"

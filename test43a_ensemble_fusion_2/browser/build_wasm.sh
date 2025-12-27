#!/bin/bash
# Build script for ARC-AGI Browser Solver

echo "🔨 Building ARC-AGI WASM Browser Solver..."
echo ""

cd "$(dirname "$0")"

# Initialize go module if needed
if [ ! -f go.mod ]; then
    echo "📦 Initializing Go module..."
    go mod init arc-browser
    go mod tidy
fi

# Build the WASM binary
echo "📦 Compiling Go to WASM..."
GOOS=js GOARCH=wasm go build -o main.wasm .

if [ $? -eq 0 ]; then
    echo "✅ WASM binary built: main.wasm ($(du -h main.wasm | cut -f1))"
else
    echo "❌ WASM build failed!"
    exit 1
fi

echo ""
echo "🎉 Build complete!"
echo ""
echo "To run the server:"
echo "  go run server.go"
echo ""
echo "Then open: http://localhost:8043"

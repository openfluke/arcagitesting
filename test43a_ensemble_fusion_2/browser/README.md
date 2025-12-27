# 🔮 ARC-AGI Browser Solver

A fully in-browser neural network ensemble solver for ARC-AGI tasks, powered by WebAssembly.

## Overview

This browser application runs the Ensemble Fusion v2 solver entirely client-side using WASM compiled from Go. It trains diverse neural networks and uses complementary stitching to solve ARC-AGI tasks.

## Features

- **🧠 60 Diverse Networks** - MHA, LSTM, RNN, Dense, SwiGLU, NormDense brains
- **🔗 Multiple CombineModes** - concat, add, avg, grid_scatter
- **📐 Varied Grid Shapes** - 1x1 to 3x3 topologies
- **🎯 Pixel-Level Stitching** - Voting-based fusion across predictions
- **✨ Live Visualization** - Real-time grid rendering and progress tracking

## Quick Start

```bash
# Start the static file server
go run server.go

# Open in browser
# http://localhost:8043
```

## Project Structure

```
browser/
├── index.html      # Main UI with controls and visualization
├── server.go       # Simple static file server (port 8043)
├── build_wasm.sh   # Build script for WASM binary
└── wasm/           # WASM solver module
    ├── solver.go   # Go source (compiles to WASM)
    ├── main.wasm   # Compiled WASM binary (~5MB)
    └── README.md   # WASM module documentation
```

## Requirements

- Modern browser with WebAssembly support
- `wasm_exec.js` served at configured URL
- ARC-AGI data accessible via HTTP

## Configuration

The `index.html` expects:
- `wasm_exec.js` from `http://192.168.0.228:4000/wasm_exec.js`
- ARC data from `http://192.168.0.228:4000/data/ARC-AGI/data`

Update these URLs in `index.html` for your setup.

## How It Works

1. **Phase 1**: Trains 60 diverse neural networks on training tasks
2. **Phase 1.5**: Clusters networks by output specialization
3. **Phase 2**: Stitches predictions using pixel-level voting
4. **Result**: Displays solved tasks with confetti 🎉

## License

Part of the Loom neural network framework.

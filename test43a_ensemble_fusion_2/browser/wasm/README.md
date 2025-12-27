# 🧠 ARC-AGI WASM Solver Module

WebAssembly neural network solver for ARC-AGI tasks.

## Overview

This module compiles the Ensemble Fusion v2 solver to WASM, allowing it to run entirely in the browser without a backend server.

## Architecture

The solver uses `nn.Network` from the Loom framework with:

### Brain Types
| Type | Description |
|------|-------------|
| MHA | Multi-Head Attention (30% probability) |
| LSTM | Long Short-Term Memory (25%) |
| RNN | Recurrent Neural Network (15%) |
| Dense | Dense/MLP layers (15%) |
| SwiGLU | Gated Linear Unit variant (8%) |
| NormDense | Normalized Dense (7%) |

### Grid Shapes
- `1x1 Mono` - Single brain
- `2x2 Standard` - 4 brains in grid
- `3x3 Complex` - 9 brains in grid
- `4x1 Tall` - Vertical arrangement
- `1x4 Wide` - Horizontal arrangement

### CombineModes
- `avg` (35%) - Average outputs
- `add` (30%) - Sum outputs
- `concat` (20%) - Concatenate outputs
- `grid_scatter` (15%) - Scatter to grid positions

## Building

```bash
# From this directory
GOOS=js GOARCH=wasm go build -o main.wasm .

# Or use the parent build script
cd .. && ./build_wasm.sh
```

## Output

- `main.wasm` (~5.2 MB) - Compiled WASM binary

## Exported JavaScript Functions

```javascript
// Initialize with dataset parameters
startARCSolver(dataset, networkCount, dataUrl)

// Run solver with task data
solveWithData(trainTasksJSON, evalTasksJSON)
```

## Callbacks (called from Go)

```javascript
window.jsLog(message, type)       // Log messages
window.jsUpdateStats(solved, total)  // Update progress
window.jsCelebrate(taskId)        // Trigger celebration
window.jsRenderGrid(containerId, grid)  // Render ARC grid
window.jsSetPhase(phase)          // Update phase display
window.jsSetTaskId(taskId)        // Update current task
```

## Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| MaxGridSize | 30 | Maximum grid dimension |
| InputSize | 900 | 30×30 flattened |
| EnsembleSize | 15 | Networks per ensemble |
| NumEnsembles | 4 | Number of ensembles |
| TestDuration | 5s | Training time per network |
| AdaptationPasses | 2 | Few-shot adaptation iterations |

## Dependencies

- `github.com/openfluke/loom/nn` - Neural network framework

## License

Part of the Loom neural network framework.

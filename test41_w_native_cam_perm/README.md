# Test 41-W Native Cam Perm — sine adaptation matrix @ SIMD

Full **dtype × quant × mode × arch** sweep using the **native**
`Hemispheres(n)` cameral API (not hand-wired Parallel).

Sibling of [`test41_w_sine_ada_perm`](../test41_w_sine_ada_perm) (Dense +
hand-wired Bicameral) and [`test41_w_native_cam`](../test41_w_native_cam)
(float32-only spot check).

Measuring: `welvet/lucy` (SoftAcc / Availability / AdaptPct / Score /
WeightBytes / MobileScore) + thread-CPU duty clock.

---

## Architectures

| Arch | Stack |
|------|--------|
| Dense | `10→32→32→1` |
| Bicameral | Dense → `Hemispheres(n=2, add)` → Dense |
| Tricameral | Dense → `Hemispheres(n=3, add)` → Dense |
| Quadcameral | Dense → `Hemispheres(n=4, add)` → Dense |
| Mesh (Dense only) | equal-width `32→32×3` |

Cameral Parallel stamps distinct per-hemisphere `TrainMode`s via `SetBranchModes`.

## Modes

| Kind | Modes |
|------|--------|
| Sequential | NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain |
| Mesh | MeshBP, MeshTween, MeshTweenChain |

## Matrix

| Axis | Values |
|------|--------|
| Backend | SIMD |
| DType | `core.AllDTypes` |
| Quant | `quant.AllFormats` |
| Arch | Dense, Bi, Tri, Quad (+ mesh on Dense) |

`--smoke` shrinks dtype/quant/mode/arch for a quick pass.

## Protocol

| Knob | Default | Flag |
|------|---------|------|
| Duration | 2s | `-duration` |
| Switch | 500ms | `-switch` |
| Window | 50ms | `-window` |
| AdaptWindows | 4 | `-adapt-windows` |
| Workers | **1** (comparable Score) | `-workers` |

## Run

```bash
cd loom/arcagitesting/test41_w_native_cam_perm

# quick sanity
go run . -smoke -workers 4 -duration 1s

# full matrix (resume-friendly)
go run . -workers 1 -duration 2s

# faster / noisier Score
go run . -workers 8
```

Writes `results/cell_*.json` (skip passed on rerun) and `perm_summary.json`.

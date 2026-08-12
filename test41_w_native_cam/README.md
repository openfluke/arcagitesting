# Test 41-W Native Cam — sine adaptation × Dense + bi / tri / quad

Lucy-style sine adaptation bench. Cameral arches use the **native**
`Hemispheres(n)` API; Dense is the non-cameral baseline (+ mesh modes).

Measuring matches **test41_w_sine_ada_perm** race knobs + equations
(`welvet/lucy`): SoftAcc, Availability, AdaptPct, Score, WeightBytes,
MobileScore. Sine data stays in this app.

| Arch | What |
|------|------|
| Dense | `10→32→32→1` (mesh: equal-width `32→32×3`) |
| Bicameral | Dense → `Hemispheres(n=2)` → Dense |
| Tricameral | Dense → `Hemispheres(n=3)` → Dense |
| Quadcameral | Dense → `Hemispheres(n=4)` → Dense |

Cameral Parallel stamps distinct per-hemisphere `TrainMode`s via `SetBranchModes`.

## What you are measuring

Same three axes as perm / tide / live_mnist:

| Axis | Metrics | Meaning |
|------|---------|---------|
| Adaptation quality | SoftAcc, AdaptPct, Stability, Consistency | Track after each frequency switch |
| Duty-cycle availability | Availability, ZeroDowntime | InferMs / (InferMs+TrainMs) |
| Cost | WeightBytes, HeapBytes, MobileScore | Size + Score/MiB |

| Symbol | Formula |
|--------|---------|
| SoftAcc | `100 × (1 − \|pred − target\| / 0.10)`, clamped `[0,100]` |
| Availability | `InferMs / (InferMs + TrainMs) × 100` |
| AdaptPct | Mean SoftAcc in AdaptWindows after each switch |
| Score | `T × Availability × SoftAcc / 10_000` |
| MobileScore | `Score / WeightMiB` |

## Modes

| Kind | Modes |
|------|--------|
| Sequential (all arches) | NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain |
| Mesh (Dense only) | MeshBP, MeshTween, MeshTweenChain |

→ **27 jobs** (4 arches × 6 seq + 3 mesh).

## Protocol (aligned with perm)

| Knob | Default | Flag | Perm |
|------|---------|------|------|
| Duration | **2s** | `-duration` | 2s |
| Freq switch | **500ms** | `-switch` | 500ms |
| Window | 50ms | `-window` | 50ms |
| AdaptWindows | **4** (=200ms) | `-adapt-windows` | 4 |
| Workers | **1** | `-workers` | often 8 (noisier Score) |
| Backend | **SIMD** | — | SIMD |
| Duty clock | thread-CPU | — | wall (≈same at workers=1) |

```bash
# perm-aligned measuring (default)
go run .

# old long sine_ada race
go run . -duration 10s -switch 2.5s -adapt-windows 10

# faster smoke (Score not comparable)
go run . -workers 8
```

Writes `test41_w_native_cam_results.json` (Lucy summary + RAM / MobileScore).

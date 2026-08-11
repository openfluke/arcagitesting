# Test 41-W Perm — sine adaptation matrix @ SIMD

Welvet port of the Lucy-style sine adaptation bench, swept across **every**
dtype × quant format × training mode × architecture on the **SIMD** backend,
with **RAM / weight size** recorded per cell.

This is not “who gets the best static sine fit.” It asks which combinations
still **serve answers while learning** when the target frequency keeps
switching — and how expensive that is in storage.

---

## What you are measuring

Three axes at once:

| Axis | Metrics | Meaning |
|------|---------|---------|
| Adaptation quality | SoftAcc, AdaptPct, Stability, Consistency | How well / how fast the net tracks after each frequency switch |
| Duty-cycle availability | Availability, ZeroDowntime | Share of busy time spent inferring vs training |
| Cost | WeightBytes, HeapBytes, MobileScore | How small the model is, and Score per MiB |

### Pareto front (why the full matrix)

A **Pareto front** (a “surface” when you have more than two goals) is the set of
options where improving one goal forces you to hurt another.

Example tradeoffs in this sweep:

- Higher Score ↔ larger WeightBytes  
- Higher SoftAcc / AdaptPct ↔ lower Availability (more train time)  
- Aggressive k-quants ↔ cheaper RAM but maybe worse Acc / train stability  

Any cell that is worse on **all** axes than another is *dominated* and falls
off the front. The interesting winners are the undominated edge — e.g. “best
Score at ≤2 KiB” or “best Availability among Q4_K cells.” The top-N score
table at the end is a convenience ranking; the real story is that tradeoff
surface.

---

## Lucy score formulas

Inherited from the legacy `all_sine_wave.go` measuring style:

| Symbol | Formula |
|--------|---------|
| SoftAcc | `100 × (1 − \|pred − target\| / 0.10)`, clamped to `[0, 100]` |
| Availability | `InferMs / (InferMs + TrainMs) × 100` |
| AdaptPct | Mean SoftAcc in the first few windows after each frequency switch |
| Throughput (T) | `TotalOutputs / duration_seconds` |
| Score | `T × Availability × SoftAcc / 10_000` |
| ZeroDowntime | `SoftAcc × Availability / 100` |
| MobileScore | `Score / WeightMiB` |

Task: predict the next sine sample from a sliding window of 10 points. Every
`--switch` interval the sine frequency changes (`1 → 2 → 3 → 4`), so the net
must re-adapt online.

---

## Matrix

Default full sweep (~10k+ cells):

| Dimension | Values |
|-----------|--------|
| Backend | SIMD only (`core.BackendSIMD`) |
| DType | `core.AllDTypes` (~34) |
| Quant | `quant.AllFormats` (none, Q*, K-quants, IQ*, ternary/binary/affine, …) |
| Modes (sequential) | NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain |
| Modes (mesh) | MeshBP, MeshTween, MeshTweenChain |
| Arch | Dense (`10→32→32→1`), Bicameral (Dense → Parallel add → Dense) |

Mesh modes use equal-width Dense (`32→32×3`) only — volumetric step ticks need
matching dims. Bicameral is sequential only (not mesh).

`--smoke` shrinks this to a tiny dtype/quant/mode subset for a quick sanity pass.

### Expected hard fails

Some formats have packing constraints this tiny net violates. Those cells fail
at build time (❌) and are still useful as “unsupported for this shape”:

- **AffinePacked** — first Dense layer has `cols=10` (not multiple of 8); mesh
  has `cols=32` (not multiple of group 64).

Other formats should generally build and run; failures there are real bugs or
backend gaps worth chasing.

---

## Architectures & modes (short)

**Dense** — classic MLP.  
**Bicameral** — two parallel Dense branches combined with add (welvet
`parallel`), sandwiched between Dense in/out.

| Mode | Idea |
|------|------|
| NormalBP | Batched SGD every ~10 ms |
| StepBP | Online SGD every sample |
| Tween / TweenChain | Gap/tween updates (chain rule on/off), batched |
| StepTween / StepTweenChain | Online tween every sample |
| MeshBP / MeshTween / MeshTweenChain | Volumetric step ticks through equal-width stack |

Tween link budgets are floored to `0.8` before `ApplyGaps` so updates are not
skipped by welvet’s low-budget gate (legacy LinkBudgetScale ≈ 0.8).

---

## Status markers in the log

| Mark | Meaning |
|------|---------|
| ✅ | Cell ran and passed |
| ❌ | Cell failed (build/pack/runtime); error truncated after `\|` |
| ⏭️ | Skipped — passed result already in `results/cell_*.json` (resume) |

On rerun, **passed** cells are skipped; **failed** cells are retried. When
everything that can pass is cached, a rerun mostly prints ⏭️ lines then the
summary / top table.

---

## RAM metrics

Per cell, after network construction:

| Field | Source |
|-------|--------|
| `weightBytes` / `weightMiB` | Sum of stores from `dna.CollectStores` (Native / Packed raw+scales+mins+caches, Bias) |
| `heapBytes` / `heapMiB` | `runtime.MemStats.HeapAlloc` delta around create (serialized GC sample) |
| `mobileScore` | `Score / WeightMiB` |

`weightBytes` is the meaningful model-size number; heap is noisier under
concurrent workers but useful as a rough alloc signal.

---

## How to run

From this directory:

```bash
# quick subset
go run . -smoke -workers 4 -duration 1s

# full matrix (resume-friendly)
go run . -workers 8 -duration 2s

# custom
go run . -workers 16 -duration 2s -switch 500ms -window 50ms \
  -outdir results -summary perm_summary.json
```

Flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `-workers` | `8` | Concurrent cells |
| `-duration` | `2s` | Per-cell wall time |
| `-switch` | `500ms` | Frequency switch interval |
| `-window` | `50ms` | SoftAcc window size |
| `-outdir` | `results` | Per-cell JSON cache |
| `-summary` | `perm_summary.json` | Aggregate summary |
| `-smoke` | off | Tiny matrix |

Outputs:

- `results/cell_<Arch>_<Mode>_<dtype>_<quant>_simd.json` — one file per cell  
- `perm_summary.json` — full aggregate + all results  
- stdout — live progress + top-15 by Lucy Score + best MobileScore / smallest weights  

Rough full-run time: on the order of tens of minutes at `-duration 2s` with
8 workers (many AffinePacked cells fail instantly).

---

## Reading results

After a run, sort / filter `perm_summary.json` (or the per-cell files) for the
Pareto story you care about, e.g.:

1. Max `score` among cells with `weightBytes ≤ N`  
2. Max `availability` among cells with `avgTrainAccuracy ≥ 20`  
3. Max `mobileScore` (Score per MiB)  
4. Group by `quant` or `mode` and compare distributions  

The printed top table is Score-ordered only — use the JSON for multi-objective
cuts.

---

## Related

- Sibling single-matrix bench: `../test41_w_sine_ada/`  
- Legacy loom sine adaptation: `../test41_sine_adaptation.go`  
- Measuring lineage: Lucy / `all_sine_wave.go` SoftAcc + duty-cycle Availability  

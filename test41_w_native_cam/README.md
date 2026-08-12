# Test 41-W Native Cam — sine adaptation × Dense + bi / tri / quad

Lucy-style sine adaptation bench. Cameral arches use the **native**
`Hemispheres(n)` API; Dense is the non-cameral baseline (+ mesh modes).

Measuring: `welvet/lucy` (SoftAcc, Availability, AdaptPct, Score, WeightBytes,
MobileScore). Full dtype×quant matrix → [`test41_w_native_cam_perm`](../test41_w_native_cam_perm).

| Arch | What |
|------|------|
| Dense | `10→32→32→1` (mesh: equal-width `32→32×3`) |
| Bicameral | Dense → `Hemispheres(n=2)` → Dense |
| Tricameral | Dense → `Hemispheres(n=3)` → Dense |
| Quadcameral | Dense → `Hemispheres(n=4)` → Dense |
| Mix (hex) | Dense → `Hemispheres(n=6)` → Dense, one of each seq mode |

## Modes

| Kind | Modes |
|------|--------|
| Sequential (all arches) | NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain |
| Mesh (Dense only) | MeshBP, MeshTween, MeshTweenChain |

→ **27 uniform** + Mix distinct-mode perms (float32 / none @ SIMD).

### Mix — distinct-mode permutations on Bi / Tri / Quad

Uniform Bi/Tri/Quad stay as-is (same mode on every hemi).

**Mix** assigns **different** seq modes to each hemisphere (`TrainStackMSE`).
Order matters: `NormalBP∥StepBP` ≠ `StepBP∥NormalBP`.

| Scope (`-mix`) | Jobs added | Formula |
|----------------|-----------:|---------|
| `off` | 0 | uniform only (27) |
| `bi` | **30** | P(6,2) |
| `tri` | **120** | P(6,3) |
| `quad` | **360** | P(6,4) |
| `hex` | **1** | all 6 modes once |
| `all` (default) | **511** | bi+tri+quad+hex → **538** total |

Examples: `Bicameral/Mix(NormalBP∥StepBP)`, `Bicameral/Mix(Tween∥StepBP)`, …

```bash
go run .              # full Mix perms (538)
go run . -mix bi      # Bi pairs only (57)
go run . -mix off     # uniform only
```


## Protocol (long race — default)

| Knob | Default | Flag |
|------|---------|------|
| Duration | **10s** | `-duration` |
| Freq switch | **2.5s** (`1→2→3→4`) | `-switch` |
| Window | 50ms | `-window` |
| AdaptWindows | **10** (=500ms) | `-adapt-windows` |
| Workers | **1** | `-workers` |
| Mix | **all** (511 Mix + 27) | `-mix` |
| Backend | SIMD | — |
| Duty clock | thread-CPU | — |

```bash
cd loom/arcagitesting/test41_w_native_cam
go run .

# short / perm-like race
go run . -duration 2s -switch 500ms -adapt-windows 4
```

Writes `test41_w_native_cam_results.json`.

---

## Snapshot results (float32 / none @ SIMD)

Run: `-workers 1 -duration 10s -switch 2.5s -adapt-windows 10` · duty =
thread-CPU · **27/27** passed.

### SoftAcc timeline (1s blocks)

Switches at 2.5 / 5.0 / 7.5s.

| Label | 1s | 2s | 3s | 4s | 5s | 6s | 7s | 8s | 9s | 10s | Avg | Score |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|----:|----:|------:|
| Dense/NormalBP | 47 | 59 | 49 | 74 | 65 | 63 | 73 | 66 | 73 | 89 | 66 | 2827 |
| Dense/StepBP | 59 | 70 | 65 | 81 | 83 | 77 | 85 | 82 | 87 | 89 | 78 | **4492** |
| Dense/Tween | 12 | 12 | 12 | 12 | 11 | 10 | 9 | 10 | 10 | 10 | 11 | 679 |
| Dense/TweenChain | 15 | 25 | 28 | 39 | 38 | 24 | 37 | 38 | 37 | 49 | 33 | 486 |
| Dense/StepTween | 11 | 11 | 14 | 14 | 15 | 9 | 9 | 10 | 10 | 10 | 11 | 1339 |
| Dense/StepTweenChain | 34 | 50 | 57 | 78 | 80 | 63 | 73 | 69 | 75 | 79 | 66 | 1414 |
| Bicameral/NormalBP | 34 | 44 | 43 | 58 | 56 | 47 | 51 | 51 | 55 | 62 | 50 | 991 |
| Bicameral/StepBP | 51 | 66 | 62 | 80 | 82 | 73 | 83 | 82 | 87 | 89 | 75 | 1961 |
| Bicameral/Tween | 12 | 12 | 12 | 10 | 10 | 9 | 9 | 10 | 9 | 10 | 10 | 704 |
| Bicameral/TweenChain | 16 | 19 | 24 | 29 | 32 | 19 | 32 | 29 | 31 | 30 | 26 | 210 |
| Bicameral/StepTween | 12 | 12 | 11 | 11 | 11 | 9 | 9 | 10 | 10 | 10 | 10 | 1407 |
| Bicameral/StepTweenChain | 40 | 58 | 58 | 75 | 82 | 62 | 75 | 71 | 74 | 78 | 67 | 746 |
| Tricameral/NormalBP | 27 | 50 | 29 | 38 | 48 | 47 | 52 | 37 | 55 | 69 | 45 | 649 |
| Tricameral/StepBP | 53 | 68 | 66 | 79 | 85 | 75 | 87 | 80 | 86 | 89 | 77 | 1360 |
| Tricameral/Tween | 12 | 12 | 12 | 11 | 11 | 9 | 9 | 10 | 10 | 10 | 11 | 592 |
| Tricameral/TweenChain | 19 | 23 | 21 | 22 | 35 | 19 | 28 | 23 | 26 | 38 | 26 | 140 |
| Tricameral/StepTween | 12 | 12 | 12 | 11 | 11 | 9 | 9 | 10 | 10 | 10 | 11 | 886 |
| Tricameral/StepTweenChain | 39 | 58 | 56 | 76 | 81 | 58 | 70 | 69 | 75 | 79 | 66 | 449 |
| Quadcameral/NormalBP | 30 | 35 | 30 | 48 | 67 | 48 | 42 | 37 | 47 | 58 | 44 | 441 |
| Quadcameral/StepBP | 54 | 69 | 64 | 79 | 83 | 75 | 85 | 79 | 84 | 87 | 76 | 932 |
| Quadcameral/Tween | 11 | 11 | 12 | 11 | 11 | 11 | 10 | 11 | 10 | 10 | 11 | 413 |
| Quadcameral/TweenChain | 12 | 21 | 27 | 23 | 23 | 20 | 34 | 25 | 27 | 32 | 24 | 100 |
| Quadcameral/StepTween | 12 | 12 | 11 | 10 | 10 | 9 | 9 | 10 | 11 | 11 | 11 | 628 |
| Quadcameral/StepTweenChain | 43 | 56 | 51 | 57 | 72 | 59 | 75 | 67 | 71 | 76 | 63 | 271 |
| Dense/MeshBP | 13 | 12 | 16 | 24 | 31 | 38 | 44 | 32 | 16 | 16 | 24 | 2328 |
| Dense/MeshTween | 9 | 14 | 12 | 21 | 14 | 0 | 0 | 0 | 0 | 0 | 7 | 1127 |
| Dense/MeshTweenChain | 16 | 18 | 28 | 33 | 36 | 29 | 28 | 33 | 42 | 40 | 30 | 2009 |

### Summary table

| Label | Acc | Adapt | Avail | Stab | Cons | Tput | Score | RAM KiB | Mobile |
|-------|----:|------:|------:|-----:|-----:|-----:|------:|--------:|-------:|
| Dense/NormalBP | 65.9 | 53.3 | 12.5 | 72.6 | 88.0 | 34273 | 2827 | 5.4 | 538565 |
| **Dense/StepBP** | **77.6** | **68.2** | 15.3 | 88.0 | 100 | **37913** | **4492** | 5.4 | **855873** |
| Dense/Tween | 10.8 | 11.1 | 15.5 | 97.8 | 69.5 | 40827 | 679 | 5.4 | 129409 |
| Dense/TweenChain | 33.0 | 25.3 | 7.5 | 67.3 | 54.5 | 19667 | 486 | 5.4 | 92592 |
| Dense/StepTween | 11.5 | 11.5 | 21.7 | 97.7 | 74.0 | 53789 | 1339 | 5.4 | 255185 |
| Dense/StepTweenChain | 65.8 | 58.3 | 9.1 | 84.2 | 99.5 | 23685 | 1414 | 5.4 | 269344 |
| Bicameral/NormalBP | 49.9 | 43.2 | 10.6 | 64.4 | 68.5 | 18729 | 991 | 9.4 | 108238 |
| Bicameral/StepBP | 75.4 | 66.3 | 12.3 | 86.2 | 100 | 21081 | 1961 | 9.4 | 214221 |
| Bicameral/Tween | 10.5 | 10.0 | 19.7 | 98.6 | 50.0 | 34229 | 704 | 9.4 | 76915 |
| Bicameral/TweenChain | 26.0 | 23.1 | 6.7 | 67.6 | 42.0 | 12136 | 210 | 9.4 | 22952 |
| Bicameral/StepTween | 10.5 | 10.1 | 30.0 | 99.0 | 70.5 | 44818 | 1407 | 9.4 | 153666 |
| Bicameral/StepTweenChain | 67.3 | 57.6 | 8.5 | 86.1 | 99.5 | 13078 | 746 | 9.4 | 81479 |
| Tricameral/NormalBP | 45.2 | 38.2 | 9.7 | 63.2 | 63.0 | 14736 | 649 | 13.4 | 49686 |
| Tricameral/StepBP | 76.7 | 66.5 | 11.1 | 87.1 | 100 | 15927 | 1360 | 13.4 | 104131 |
| Tricameral/Tween | 10.7 | 10.6 | 19.1 | 98.8 | 74.0 | 29055 | 592 | 13.4 | 45358 |
| Tricameral/TweenChain | 25.6 | 19.9 | 6.0 | 66.9 | 40.0 | 9140 | 140 | 13.4 | 10702 |
| Tricameral/StepTween | 10.6 | 10.1 | 26.5 | 98.7 | 75.0 | 31625 | 886 | 13.4 | 67803 |
| Tricameral/StepTweenChain | 66.0 | 55.5 | 7.0 | 85.7 | 99.5 | 9699 | 449 | 13.4 | 34396 |
| Quadcameral/NormalBP | 44.2 | 30.2 | 9.2 | 61.5 | 59.5 | 10841 | 441 | 17.4 | 25981 |
| Quadcameral/StepBP | 75.9 | 65.5 | 9.8 | 87.5 | 100 | 12477 | 932 | 17.4 | 54937 |
| Quadcameral/Tween | 10.8 | 11.0 | 17.8 | 98.5 | 93.5 | 21386 | 413 | 17.4 | 24364 |
| Quadcameral/TweenChain | 24.4 | 20.4 | 5.7 | 67.1 | 38.0 | 7123 | 100 | 17.4 | 5887 |
| Quadcameral/StepTween | 10.5 | 10.2 | 23.0 | 99.1 | 74.5 | 25850 | 628 | 17.4 | 37000 |
| Quadcameral/StepTweenChain | 62.7 | 49.4 | 6.4 | 86.2 | 99.5 | 6721 | 271 | 17.4 | 15992 |
| Dense/MeshBP | 24.1 | 23.8 | 94.7 | 87.8 | 100 | 10199 | 2328 | 12.0 | 198630 |
| Dense/MeshTween | 7.0 | 4.0 | 96.7 | 91.6 | 32.5 | 16670 | 1127 | 12.0 | 96162 |
| Dense/MeshTweenChain | 30.1 | 33.0 | 92.0 | 91.2 | 99.5 | 7247 | 2009 | 12.0 | 171432 |

### Highlights

| | Cell | Value |
|--|------|------:|
| Winner (Score) | Dense/StepBP | 4492 |
| Best mobile Score/MiB | Dense/StepBP | ~856k |
| Smallest weights | Dense/* | 5.4 KiB |
| Best cameral Score | Bicameral/StepBP | 1961 |
| Best Availability | Dense/MeshTween | 96.7% |

### Reading

- **StepBP owns the long race.** Acc ~76–78% across Dense→Quad, AdaptPct
  ~65–68%, and Dense still has the best Tput → Score 4492.
- **More hemispheres ≠ better Score here.** Acc stays high on Tri/Quad StepBP,
  but Tput/Avail fall with width (9.4 → 17.4 KiB), so Score drops
  Dense 4492 → Bi 1961 → Tri 1360 → Quad 932.
- **Chain helps tween learn** (StepTweenChain Acc ~63–67% vs StepTween ~11%)
  but costs Availability → loses Score to StepBP on this 10s protocol.
- **Mesh** is the Availability story (~92–97%) with weak Acc; MeshBP/MeshTweenChain
  land mid-board on Score via duty-cycle, not fit quality.
- **Plain Tween / StepTween** barely track the sine (~11% Acc) — fast but not
  adapting under SoftAcc scale 0.10.
- Timeline dips at switch seconds (esp. ~3s / 6s) then recover — StepBP recovers
  cleanest; MeshTween collapses to 0% after mid-run (unstable).

Contrast short/noisy `test41_w_sine_ada_perm` boards (workers=8, 2s): those
favored Bicameral/StepTweenChain on Availability. This clean 10s serial run
favors **Dense/StepBP**.

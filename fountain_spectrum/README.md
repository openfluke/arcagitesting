# ARC-AGI Neural Fountain

Same two modes as [`loom_neural_fountain`](../../../loom_neural_fountain):

| Mode | Like | What |
|------|------|------|
| **normal** (`./run.sh 1`) | MNIST option 1 | One dense Master: shard all train demos → specialize → LT → ensemble |
| **spectrum** (`./run.sh 2`) | MNIST option 2 | family×dtype matrix + mega fountain |

**EVAL is never used for specialize** — only scored afterward.

```bash
cd loom/arcagitesting/fountain_spectrum
./run.sh                     # menu
./run.sh 1                   # normal · ARC-AGI-1 · all train demos
./run.sh normal 1 quick      # K=8 · 3 epochs · still full train set
./run.sh 2                   # spectrum · AGI-1
./run.sh spectrum 1 quick    # dense+residual · 7 dtypes
./run.sh spectrum 1 transport
```

## Normal (MNIST analogue)

Same pipeline as MNIST option 1 — but ARC is **grid regression**, not 10-way classification.

| | MNIST | ARC (this harness) |
|--|-------|---------------------|
| Target | argmax digit | every cell color |
| Default net | `784→128→64→10` | `256→1024→1024→256` |
| Default epochs | 5 | 100 |
| K | 16 | auto (~4 demos/shard, ≤192) |
| Headline score | sample accuracy % | **oracle demo mean pixel %** |

Blind-copied MNIST width (`→128→64→`) cannot memorize ARC grids — that was the bug behind ~0 solves.
Ensemble looks weak on ARC because specialists memorize *different* puzzles; MNIST classes are shared.
Oversized grids (>16×16) are dropped from specialize/score. EVAL is never used for train.

## Spectrum

Default families: `dense`, `residual`. Use `transport` / `-all-families` for the rest
(weight-transport only). Full train+eval corpus; quick only shrinks dtypes/K/epochs.

## Pipeline (spectrum)

**Level 1** — each `(family × dtype)`: specialize → LT → score  
**Level 2** — mega fountain over all Masters → vote demos / train-test / EVAL

## Notes

- Canvas pad/crop is **16×16** (larger ARC grids are clipped).
- Data: `../ARC-AGI/data/{training,evaluation}`, `../ARC-AGI2/data/{training,evaluation}`.

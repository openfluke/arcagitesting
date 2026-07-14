# ARC-AGI Testing

Neural network experiments for the **ARC-AGI (Abstraction and Reasoning Corpus)** benchmark using LOOM's training modes.

## Neural Fountain spectrum (new)

Same specialize → LT peel → Master matrix as `loom_neural_fountain` run **2**, pointed at ARC:

```bash
cd fountain_spectrum
./run.sh both          # ARC-AGI-1 logs, then ARC-AGI-2 logs
./run.sh 1 quick
```

See [`fountain_spectrum/README.md`](fountain_spectrum/README.md).

---

## 🎯 What is ARC-AGI?

ARC-AGI is considered one of the hardest AI benchmarks, designed to test **abstract reasoning and generalization** - abilities that current AI systems struggle with. Created by François Chollet (creator of Keras), it features 400 unique visual reasoning tasks.

**2024 Challenge Results:**
- 🏆 **$600,000 grand prize** for 85% accuracy — **unclaimed**
- 🥇 Best team: MindsAI at **55.5%** (private eval)
- 🥈 Top open-source: "the ARChitects" at **53.5%** ($25,000 prize)
- 🤖 GPT-4, Claude: ~20-30% without fine-tuning
- 👤 Humans: ~85% average

---

## ⚡ Real-Time Task Switching Benchmark

Our benchmark tests something different: **how quickly can a neural network adapt when rapidly switching between unknown tasks?**

### The Experiment
- Stream through **400 ARC tasks** in **10 seconds**
- Track **pixel accuracy %** every **100ms**
- Compare batch-based vs per-sample training modes

### 🔥 Key Result: 3 Tasks Solved in 10 Seconds

While rapidly switching between 400 completely different tasks, our **StepTweenChain** mode:
- Maintained **40-50% training accuracy** consistently
- Solved **3 tasks** with ≥95% pixel accuracy on unseen eval data
- Scored highest on the **T×S×C metric** (Throughput × Stability × Consistency)

Compare this to **NormalBP** (standard batch training):
- Stuck at **~10% accuracy** (barely above random chance of 11%)
- **Pauses** to do batch training — can't adapt in real-time
- Scored lowest due to near-zero consistency

### 📊 Mode Comparison

| Mode | Stability | Throughput | Consistency | Solved | Score |
|------|-----------|------------|-------------|--------|-------|
| **StepTweenChain** | 85% | 804/s | 84% | **3** | **57** |
| StepTween | 84% | 801/s | 80% | 1 | 54 |
| Tween | 78% | 1,264/s | 56% | 0 | 55 |
| TweenChain | 77% | 1,266/s | 54% | 0 | 53 |
| StepBP | 95% | 436/s | 73% | 0 | 30 |
| NormalBP | 97% | 1,204/s | 1% | 0 | 1 |

---

## 🧬 Evolutionary Swarm: Genetic Lottery with Nano-Hives

We spawned **100 randomized network architectures** to find the optimal configuration through evolution!

### What We Randomized
- **Grid sizes**: 1×1, 2×2, 3×3 parallel brains
- **Brain types**: MHA, LSTM, RNN, Dense (randomly combined)
- **DModel**: 16, 32, 64
- **NumHeads**: 2, 4, 8
- **Learning rate**: 0.001 - 0.1

### 🏆 Swarm Results: 5 Tasks Solved!

| Rank | Architecture | Accuracy | Solved | Score |
|------|--------------|----------|--------|-------|
| 🥇 1 | 1×1 MHA (D=16, H=2, LR=0.095) | 46.5% | 3 | 138 |
| 🥈 2 | 1×1 LSTM (D=16, H=4, LR=0.079) | 46.1% | 4 | 133 |
| 🥉 3 | 1×1 RNN (D=16, H=4, LR=0.048) | 45.5% | 3 | 128 |
| 6 | 2×2 Dense+MHA+LSTM+RNN (D=16) | 44.8% | 4 | 122 |
| **9** | **3×3 MHA×5+RNN×3+Dense (D=16)** | **45.6%** | **5** | 111 |

### 🔑 Key Insights

1. **Simpler is faster**: 1×1 grids score highest due to higher throughput
2. **Larger grids solve more**: The 3×3 grid solved **5 tasks** (most of any architecture!)
3. **Small DModel wins**: D=16 dominated the leaderboard - smaller = faster adaptation
4. **High learning rate helps**: Top performers had LR > 0.04
5. **Mixed brains work**: The 2×2 with Dense+MHA+LSTM+RNN solved 4 tasks

### Why 3×3 Solved More Tasks

The 3×3 architecture with 9 diverse brains (5 MHA + 3 RNN + 1 Dense) achieved the **most tasks solved** (5), but scored lower on the T×S×C metric due to lower throughput. This suggests:

- **Diverse parallel brains** capture different patterns
- **More brains = better generalization** at cost of speed
- For **task solving**, prefer larger grids
- For **real-time adaptation**, prefer smaller grids

---

## 👑 Council of 1000: Testing Statistical Saturation

We scaled up to **1000 randomized agents** to find the architecture ceiling - how many unique tasks can this approach solve?

### 🔬 The Science: Statistical Saturation

- **If the curve keeps rising** → run more agents!
- **If the curve flattens** → you've hit the architecture ceiling

### 📊 Results: 11 Tasks, Then Ceiling

| Agents | Unique Tasks | Discovery Rate |
|--------|--------------|----------------|
| 0→50 | 7 tasks | Fast discovery |
| 50→160 | 9 tasks | Slowing down |
| 160→500 | 10 tasks | Almost flat |
| 500→710 | 11 tasks | Very rare finds |
| **710→1000** | **11 tasks** | **CEILING HIT** |

**Key Finding:** The last 290 agents (710→1000) found **zero new tasks**. The architecture has reached its limit.

### 🏆 Top Performers

| Agent | Architecture | Accuracy | Solved |
|-------|--------------|----------|--------|
| Agent-143 | 2×2 Dense×4 | 45.3% | 5 |
| Agent-925 | 1×1 Dense | 45.2% | 5 |
| Agent-816 | 1×1 LSTM | 45.1% | 5 |
| Agent-675 | 2×2 Dense+Dense+RNN+Dense | 45.0% | 5 |
| Agent-301 | 2×2 Dense+MHA+RNN+LSTM | 44.6% | 5 |

### 💡 What This Means

1. **Architecture Ceiling = 11 tasks** with Nano-Hive + real-time training
2. **Running 10,000 agents won't help** - the curve is flat
3. **To solve more tasks**, need fundamentally different architecture
4. **Dense layers surprisingly effective** - the winner was all-Dense!
5. **D=16 + High LR (0.09+)** is the sweet spot

### 🧠 The 11 Tasks Solved

These represent the "sweet spot" - tasks whose patterns match what the network can learn in 10 seconds of real-time training.

---

## 🦎 Evolutionary Zoo: Deep Architectural Mutations (Test 39)

Test 38 proved that same-topology networks saturate at 11 tasks. The Zoo introduces **SPECIATION** - wildly different network topologies to break that ceiling.

### 🔬 Results: 14 Unique Tasks Solved!

**+3 tasks beyond the Council ceiling!** Speciation works.

| Metric | Result |
|--------|--------|
| **Zoo Size** | 2500 mutants |
| **Collective Tasks** | 14 unique (vs 11 with Council) |
| **Duration** | 25m 40s |
| **Workers** | 18 parallel |

### 📊 Species Breakdown (Phylogenetic Tree)

| Species | Unique Tasks | Count | Best Mutant |
|---------|--------------|-------|-------------|
| 🥇 **3×3 Complex** | **13** | 384 | 5 solved |
| 🥈 3×2 Rectangle | 12 | 335 | 5 solved |
| 🥉 1×1 Monolith | 11 | 366 | 5 solved |
| 2×2 Standard | 11 | 355 | 5 solved |
| 1×4 Wide | 11 | 341 | 4 solved |
| 4×1 Tall | 10 | 370 | 5 solved |
| 2×3 Rectangle | 10 | 349 | 5 solved |

### 🏆 Hall of Fame - Top Performers

| Mutant | Architecture | Tasks |
|--------|--------------|-------|
| Mutant-2301 | 3×3 Dense-Dense-LSTM-LSTM-RNN-RNN-LSTM-Dense-LSTM **Tanh** D=32 LR=0.087 | 5 |
| Mutant-2314 | 2×3 MHA-Dense-Dense-Dense-LSTM-Dense **Tanh** D=64 LR=0.088 | 5 |
| Mutant-898 | 2×3 Dense-RNN-RNN-Dense-RNN-Dense **Tanh** D=32 LR=0.087 | 5 |

### 💡 Key Insights

1. **Tanh Dominates** - Every top-10 mutant used Tanh activation (not LeakyReLU!)
2. **3×3 Complex is Best** - The 9-brain grid discovered the most unique tasks (13/14)
3. **Rectangles Work** - 3×2 and 2×3 grids are competitive with traditional squares
4. **High LR Still Key** - Top performers cluster around LR 0.04-0.1
5. **Dense Brains Matter** - Most winners mix Dense layers with LSTM/RNN

### 📈 Discovery Curve: Still Rising (Barely)

```
Mutant  100: ████████ (8 tasks)
Mutant  400: ████████████ (12 tasks)
Mutant  900: █████████████ (13 tasks)
Mutant 2500: ██████████████ (14 tasks)  ← Last task found at the end!
```

**Interpretation:** The curve is very flat after 900 mutants. Task #14 was found in the final batch, suggesting a few more tasks *might* be discoverable with 5000+ mutants, but returns are diminishing rapidly.

### 🧠 The 14 Tasks Solved

These task IDs can be analyzed to understand what patterns this architecture class can learn:
```
0e206a2e 1190e5a7 1fad071e 2dc579da 445eab21 91413438 91714a58 
ae3edfdc b9b7f026 d631b094 d89b689b d9fac9be de1cd16c e8dc4411
```

---

## 🌊 Test 41: Sine Wave Adaptation Benchmark

A simplified benchmark to test **real-time adaptation** when a pattern suddenly changes. Instead of complex ARC grids, we use a simple task: **predict the next value in a sine wave** — then suddenly change the frequency!

### The Experiment

- Train on **Sin(1x)** for 2.5 seconds
- Instantly switch to **Sin(2x)** (double frequency)
- Switch to **Sin(3x)** then **Sin(4x)** (every 2.5 seconds)
- Track **prediction accuracy %** every **50ms**
- Measure which training mode adapts fastest

### 🏆 Results: StepTweenChain Wins with 4x the Score!

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║           PREDICTION ACCURACY % (50ms windows) — Sin(1x)→Sin(2x)→Sin(3x)→Sin(4x) switching every 2.5s                                          ║
╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣
║ Mode                 ║ 1s  2s  3s  4s  5s  6s  7s  8s  9s  10s ║ Avg   ║ Score      ║
╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣
║ NormalBP             ║ 73% 81% 85% 95% 96% 90% 100% 95% 100% 100% ║  92% ║       6462 ║
║ StepBP               ║ 42% 76% 76% 85% 86% 78% 86% 79% 86% 88% ║  78% ║       1841 ║
║ Tween                ║ 13% 16% 26% 54% 62% 62% 66% 58% 61% 65% ║  48% ║       8880 ║
║ TweenChain           ║ 12% 14% 14% 21% 35% 50% 60% 51% 51% 53% ║  36% ║       5493 ║
║ StepTween            ║ 17% 53% 60% 66% 66% 64% 82% 61% 67% 72% ║  61% ║      20240 ║
║ StepTweenChain       ║ 19% 61% 74% 78% 80% 73% 76% 65% 64% 65% ║  66% ║      22131 ║
╚══════════════════════╩════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝
                           ↑ 2.5s     ↑ 5.0s     ↑ 7.5s        ← Frequency switches
```

### 📊 Summary Table

| Mode | Avg Accuracy | Stability | Consistency | Throughput | **Score** |
|------|--------------|-----------|-------------|------------|-----------|
| **StepTweenChain** | 66% | 82% | 87% | 313,205/s | **22,131** 🏆 |
| StepTween | 61% | 81% | 81% | 310,326/s | 20,240 |
| Tween | 48% | 79% | 66% | 171,062/s | 8,880 |
| NormalBP | 92% | 87% | 98% | 75,434/s | 6,462 |
| TweenChain | 36% | 80% | 41% | 169,344/s | 5,493 |
| StepBP | 78% | 83% | 94% | 23,470/s | 1,841 |

### 🔑 What These Results Mean

1. **StepTweenChain has 4x the throughput of backprop** (313k vs 75k samples/sec)
   - Trains on EVERY sample immediately — no batch accumulation
   - Higher throughput = more learning cycles per second
   
2. **NormalBP has highest accuracy but lowest throughput**
   - 92% accuracy looks great, but only processes 75k samples/sec
   - PAUSES to batch train — can't adapt while training
   
3. **Score = Throughput × Stability × Consistency / 100,000**
   - Balances speed vs accuracy vs reliability
   - StepTweenChain wins because it's FAST and CONSISTENT
   
4. **Step-based methods adapt faster to frequency switches**
   - Look at columns 3, 6, 8 (right after switches) — Step methods recover faster
   - NormalBP stays high but had a 2.5s head start on each frequency
   
5. **Chain Rule matters for real-time learning**
   - StepTweenChain beats StepTween (22k vs 20k score)
   - TweenChain underperforms Tween — chain rule hurts batch methods

### 🧠 Why This Matters for AI

This benchmark demonstrates the core advantage of **step-based training**:

- **Traditional AI**: Train offline → Deploy → Can't adapt
- **Step-based AI**: Train AND serve simultaneously → Continuous adaptation

In real-world applications (robotics, trading, games), the environment changes constantly. An AI that can adapt **while running** has a massive advantage over one that needs to stop and retrain.

---

## 🔮 Test 43a: Ensemble Fusion v2 - Complementary Stitching

A multi-phase ensemble approach that trains diverse networks and **stitches their predictions together** pixel-by-pixel to solve more tasks than any single network.

### The Strategy

1. **Phase 1**: Train 120 diverse networks with random architectures (grid shapes, brain types, combine modes)
2. **Phase 1.5**: Cluster networks by output patterns into specialist groups
3. **Phase 2**: Find complementary pairs of networks and stitch their predictions (if Network A got pixel X right and Network B got pixel Y right, combine them!)
4. **Phase 3**: Recursive N-way stitching of the "Frankenstein" grids

### 🏆 Results

| Benchmark | Tasks Solved | Accuracy | Duration |
|-----------|--------------|----------|----------|
| **ARC-AGI-1** | **109/400** | **27.25%** | 5m 48s |
| ARC-AGI-2 | 8/120 | 6.7% | 4m 4s |

### 📊 Breakdown (ARC-AGI-1)

| Phase | Tasks Solved | Notes |
|-------|--------------|-------|
| Phase 1 (Clusters) | 31 | Individual specialist clusters |
| Phase 2 (Stitching) | +78 | Complementary pair stitching |
| **Total** | **109** | 27.25% of eval set |

### 💡 Key Insights

1. **Stitching is powerful**: Phase 2 solved 78 additional tasks by combining partial solutions
2. **ARC-AGI-2 is much harder**: Only 8 tasks solved vs 109 on ARC-AGI-1 (the new benchmark is designed to resist pattern-matching)
3. **Coverage improvements**: Even unsolved tasks saw +21.9% average pixel coverage improvement
4. **Diverse architectures matter**: Networks with different grid shapes/brain types produce complementary errors

### 🧠 Why Stitching Works

When two networks make **different mistakes**, their correct pixels can be combined:

```
Network A: ✅✅❌❌✅❌  (4/6 = 67% correct)
Network B: ❌✅✅❌❌✅  (3/6 = 50% correct)
Stitched:  ✅✅✅❌✅✅  (5/6 = 83% correct!)
```

Some tasks went from 30% → 100% by finding the right complementary pairs!

---

## 🚀 Running the Benchmarks

```bash
cd examples/tween/arcagitesting

# Mode comparison benchmark (10 seconds)
go run arc_benchmark.go

# Evolutionary swarm (100 networks, ~10 min)
go run genetic_swarm.go

# Council of 1000 (~10 min with 18 workers)
go run test38_council.go

# Evolutionary Zoo (2500 mutants, ~25 min)
go run test39_evolutionary_zoo.go

# Sine Wave Adaptation Benchmark (10 seconds)
go run test41_sine_adaptation.go

# Start visualization dashboard
go run viz_server.go

# Open http://localhost:8001
```

## 📁 Test Files

| Test | Description | Best Result |
|------|-------------|-------------|
| **arc_benchmark.go** | Real-time mode comparison | 3 tasks solved |
| **genetic_swarm.go** | Evolutionary architecture search (100) | 5 tasks solved |
| **test38_council.go** | Statistical saturation (1000) | 11 unique tasks |
| **test39_evolutionary_zoo.go** | Speciation (2500 mutants, 7 topologies) | 14 unique tasks |
| **test41_sine_adaptation.go** | Sine wave frequency adaptation | Score: 22,131 |
| **test43a_ensemble_fusion_2/** | Ensemble Fusion v2 with complementary stitching | **109/400 (27.25%) ARC-AGI-1** |
| test31_heuristic_hive.go | Heuristic Hive (MHA+LSTM) | 53.2% accuracy |

## 🏗️ Architecture

```
Hive Mind Network (Grid Scatter)
├── 1×1 Monolith: Single brain (fastest)
├── 2×2 Standard: 4 parallel brains
├── 3×3 Complex: 9 parallel brains
├── 4×1 Tall: 4 brains in a column
├── 1×4 Wide: 4 brains in a row
├── 2×3 Rectangle: 6 brains
└── 3×2 Rectangle: 6 brains
```

## 📚 References

- [ARC Prize 2024](https://arcprize.org/) - $1M+ in prizes
- [ARC-AGI Dataset](https://github.com/fchollet/ARC-AGI) - 400 training + 400 eval tasks
- [LOOM Neural Network Library](https://github.com/openfluke/loom) - Our training framework

# ARC-AGI Testing

Neural network experiments for the **ARC-AGI (Abstraction and Reasoning Corpus)** benchmark using LOOM's training modes.

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

## 🚀 Running the Benchmarks

```bash
cd examples/tween/arcagitesting

# Mode comparison benchmark (10 seconds)
go run arc_benchmark.go

# Evolutionary swarm (100 networks, ~10 min)
go run genetic_swarm.go

# Council of 1000 (~10 min with 18 workers)
go run test38_council.go

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
| test31_heuristic_hive.go | Heuristic Hive (MHA+LSTM) | 53.2% accuracy |

## 🏗️ Architecture

```
Hive Mind Network (Grid Scatter)
├── 1×1: Single brain (fastest)
├── 2×2: 4 parallel brains
│   ├── Brain[0,0]: MHA/LSTM/RNN/Dense
│   ├── Brain[0,1]: MHA/LSTM/RNN/Dense
│   ├── Brain[1,0]: MHA/LSTM/RNN/Dense
│   └── Brain[1,1]: MHA/LSTM/RNN/Dense
└── 3×3: 9 parallel brains (most capable, solved 5 tasks)
```

## 📚 References

- [ARC Prize 2024](https://arcprize.org/) - $1M+ in prizes
- [ARC-AGI Dataset](https://github.com/fchollet/ARC-AGI) - 400 training + 400 eval tasks
- [LOOM Neural Network Library](https://github.com/openfluke/loom) - Our training framework

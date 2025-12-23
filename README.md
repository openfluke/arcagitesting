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

### Why This Matters

Traditional ML training **stops** to process batches. In real-time scenarios (robotics, games, autonomous systems), you can't pause the world while your network trains.

**StepTweenChain** trains on **every single sample immediately** using heuristic updates + chain rule, enabling:
- Continuous adaptation without pausing
- Stable performance during rapid task switches  
- Higher throughput and consistency

## 📊 Mode Comparison

| Mode | Stability | Throughput | Consistency | Solved | Score |
|------|-----------|------------|-------------|--------|-------|
| **StepTweenChain** | 85% | 804/s | 84% | **3** | **57** |
| StepTween | 84% | 801/s | 80% | 1 | 54 |
| Tween | 78% | 1,264/s | 56% | 0 | 55 |
| TweenChain | 77% | 1,266/s | 54% | 0 | 53 |
| StepBP | 95% | 436/s | 73% | 0 | 30 |
| NormalBP | 97% | 1,204/s | 1% | 0 | 1 |

**Scoring Formula:** `Score = (Throughput × Stability × Consistency) / 100000`

## 🚀 Running the Benchmark

```bash
cd examples/tween/arcagitesting

# Run the 10-second real-time benchmark
go run arc_benchmark.go

# Start visualization server
go run viz_server.go

# Open http://localhost:8001 in browser
```

## 📁 Test Files

| Test | Description | Best Accuracy |
|------|-------------|---------------|
| **arc_benchmark.go** | Real-time 10s task switching | 3 tasks solved |
| test31_heuristic_hive.go | Heuristic Hive (MHA+LSTM) | 53.2% |
| test29 | Tricameral Native | 53.2% |
| test36c | Weighted Hive + RL | varies |

## 🏗️ Architecture

```
Hive Mind Network (Grid Scatter 2×2)
├── Brain[0,0]: Multi-Head Attention (Spatial Patterns)
├── Brain[0,1]: LSTM (Sequential Logic)
├── Brain[1,0]: MHA (Spatial Backup)
└── Brain[1,1]: MHA (Redundancy)
```

## 📚 References

- [ARC Prize 2024](https://arcprize.org/) - $1M+ in prizes
- [ARC-AGI Dataset](https://github.com/fchollet/ARC-AGI) - 400 training + 400 eval tasks
- [LOOM Neural Network Library](https://github.com/openfluke/loom) - Our training framework

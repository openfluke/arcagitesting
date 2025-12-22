# ARC-AGI Testing

Neural network experiments for the ARC-AGI (Abstraction and Reasoning Corpus) benchmark.

## 🏆 Best Result: Test 31 - Heuristic Hive Mind

**53.2% accuracy | 2 tasks solved | 139.8s training time**

```
Architecture: 2x2 Grid Scatter Hive Mind
├── Brain[0,0]: MHA (Spatial Patterns)
├── Brain[0,1]: LSTM (Temporal Logic)
├── Brain[1,0]: MHA (Spatial Backup)
└── Brain[1,1]: MHA (Redundancy)

Training: StepTween (Heuristic) | 400 Epochs
Grokking: Epoch 157 (19.2% → 20.4%)
```

### Key Findings
- **Grid Scatter + StepTween** is the winning combination
- Heuristic training works where gradient fails with parallel layers
- Grokking phenomenon observed around epoch 157

### Solved Tasks
- `27a28665`
- `de1cd16c`

## Test Files

| Test | Description | Accuracy | Tasks Solved |
|------|-------------|----------|--------------|
| **test31** | Heuristic Hive (MHA+LSTM) | **53.2%** | **2** |
| test31_rl | Heuristic Hive + RL Color Boost | TBD | TBD |
| test29 | Tricameral Native | 53.2% | 3 |
| test36 | Weighted Hive | varies | 0 |
| test36c | Weighted Hive + RL | TBD | TBD |

## Running Tests

```bash
cd examples/tween/arcagitesting
go run test31_heuristic_hive.go
```

Results are saved to `*_results.json` files.

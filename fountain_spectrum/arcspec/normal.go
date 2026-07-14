package arcspec

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

// NormalConfig mirrors loom_neural_fountain MNIST "option 1" knobs,
// but ARC grid *regression* needs a wider net + more epochs than digit classification.
type NormalConfig struct {
	Quick    bool
	LogPath  string
	K        int // 0 = auto (~8 demos/shard, capped)
	Epochs   int
	LR       float32
	LossRate float64
	MaxTasks int // 0 = all training tasks
	MaxEval  int // 0 = all evaluation (score only; never specialize)
	Sizes    []int
}

// Default dense for ARC: wide enough to memorize grids.
// MNIST used 784→128→64→10 (classification). Blind-copy 256→128→64→256 cannot
// even fit a single ARC demo — see diagnostics.
func DefaultNormalConfig() NormalConfig {
	return NormalConfig{
		K:        0, // auto from demo count
		Epochs:   100,
		LR:       0.18,
		LossRate: 0.30,
		MaxTasks: 0,
		MaxEval:  0,
		// 512 bottleneck underfits; 1024 matches diagnostics that hit ~85% oracle pixel.
		Sizes: []int{GridDim, 1024, 1024, GridDim},
	}
}

func defaultARCSizes() []int {
	return []int{GridDim, 1024, 1024, GridDim}
}

// RunNormalDataset is the MNIST-style Neural Fountain on one ARC corpus:
// specialize dense experts on training demos → LT → Master,
// then score demos / train-test / EVAL (eval never used for train).
func RunNormalDataset(ds Dataset, cfg NormalConfig) bool {
	if len(cfg.Sizes) < 2 {
		cfg.Sizes = defaultARCSizes()
	}
	if cfg.Quick {
		if cfg.Epochs == 100 {
			cfg.Epochs = 80
		}
	}
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}

	logPath := cfg.LogPath
	if logPath == "" {
		_ = os.MkdirAll("logs", 0o755)
		safe := strings.ReplaceAll(ds.Name, " ", "_")
		logPath = filepath.Join("logs", fmt.Sprintf("%s_fountain_normal_%s.log",
			safe, time.Now().Format("20060102_150405")))
	} else if dir := filepath.Dir(logPath); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("FAIL log: %v\n", err)
		return false
	}
	defer f.Close()
	w := io.MultiWriter(os.Stdout, f)
	logf := func(format string, args ...any) { fmt.Fprintf(w, format+"\n", args...) }

	logf("╔══════════════════════════════════════════════════════════╗")
	logf("║  ARC Neural Fountain — NORMAL (MNIST-style Master)       ║")
	logf("╚══════════════════════════════════════════════════════════╝")
	logf("")
	logf("Paradigm (same as loom_neural_fountain ./run.sh 1):")
	logf("  shard train demos → specialize dense experts → LT spray/peel → Master")
	logf("  EVAL tasks are scored only — never used for specialize.")
	logf("")
	logf("Why ARC looks different from MNIST 90%%:")
	logf("  MNIST = 10-way classification (shared classes → ensemble ≈ oracle).")
	logf("  ARC   = per-demo grid regression. Specialists memorize DIFFERENT puzzles,")
	logf("  so ensemble average looks weak; the MNIST-analogue is ORACLE demo pixel%%.")
	logf("  Exact full-grid / task solves are much harder than mean cell color %%.")
	logf("")
	logf("dataset: %s", ds.Name)
	logf("train:   %s", ds.TrainDir)
	logf("eval:    %s  (holdout)", ds.EvalDir)
	logf("log:     %s", logPath)

	trainAll, err := LoadTasks(ds.TrainDir, cfg.MaxTasks)
	if err != nil {
		logf("FAIL load train: %v", err)
		return false
	}
	trainTasks, droppedTrain := FilterCanvasTasks(trainAll)
	batches := demoBatches(trainTasks)

	// Auto K: target ~4 demos per shard so specialists can memorize (unlike MNIST digits).
	if cfg.K <= 0 {
		cfg.K = len(batches) / 4
		if cfg.K < 16 {
			cfg.K = 16
		}
		if cfg.K > 192 {
			cfg.K = 192
		}
		if cfg.Quick && cfg.K > 48 {
			cfg.K = 48
		}
	}
	if cfg.K < 2 {
		cfg.K = 2
	}
	if len(batches) < cfg.K {
		cfg.K = len(batches)
		if cfg.K < 2 {
			logf("FAIL: need ≥2 canvas-fit demos, got %d (dropped %d oversized)", len(batches), droppedTrain)
			return false
		}
	}

	logf("config:  K=%d (~%.1f demos/shard) epochs=%d sizes=%v loss=%.2f lr=%.3f quick=%v canvas=%dx%d",
		cfg.K, float64(len(batches))/float64(cfg.K), cfg.Epochs, cfg.Sizes, cfg.LossRate, cfg.LR, cfg.Quick, MaxCanvas, MaxCanvas)

	logf("")
	logf("── ARC data ──")
	logf("  TRAIN tasks loaded=%d  canvas-fit=%d  dropped_oversized=%d", len(trainAll), len(trainTasks), droppedTrain)
	logf("  TRAIN demos(batches)=%d  (specialize on these)", len(batches))

	var evalTasks []*Task
	var droppedEval int
	if ds.EvalDir != "" {
		evalAll, err := LoadTasks(ds.EvalDir, cfg.MaxEval)
		if err != nil {
			logf("  WARN eval: %v", err)
		} else {
			evalTasks, droppedEval = FilterCanvasTasks(evalAll)
			logf("  EVAL  tasks loaded=%d  canvas-fit=%d  dropped_oversized=%d  (holdout)",
				len(evalAll), len(evalTasks), droppedEval)
		}
	}

	logf("")
	logf("── specialize · LT · Master ──")
	pcfg := poly.NeuralFountainConfig{
		K:           cfg.K,
		Epochs:      cfg.Epochs,
		LR:          cfg.LR,
		LossType:    "mse",
		Mode:        poly.TrainingModeCPUMC,
		LossRate:    cfg.LossRate,
		MaxOverhead: 5.0,
		Verbose:     true,
		Seed:        poly.SeedFrom("arc-nf-normal", ds.Name),
	}
	factory := poly.DenseSpecialistFactory("arc-nf-normal-dense", cfg.Sizes, nil)
	start := time.Now()
	master, err := poly.NeuralFountain(factory, batches, pcfg)
	if err != nil {
		logf("FAIL assemble: %v", err)
		return false
	}
	wall := time.Since(start)

	logf("")
	logf("── Master scores (after Neural Fountain) ──")
	logf("  recovered specialists = %d/%d (fountain byte-exact)", master.Recovered, master.K)
	logf("  specialize=%.1fms  fountain=%.1fms  wall=%s",
		float64(master.SpecializeUs)/1000, float64(master.FountainUs)/1000, wall.Round(time.Millisecond))
	logf("")
	logf("  How to read:")
	logf("    tasks finished  = ALL pairs in a JSON task exact  (the ARC unit)")
	logf("    pairs exact     = one input→output grid fully correct")
	logf("    mean pixel%%    ≈ MNIST sample%% analogue (cells with right color)")
	logf("    task-routed     = pick specialist by demo fit, then score tests")
	logf("                      (ensemble average of 192 nets ruins grids — ignore it for ARC)")
	logf("")

	oracleDemo := evalOracleDemos(master, trainTasks, batches)
	routedDemo := EvalTaskRoutedDemos(master, trainTasks)
	routedTrainTest := EvalTaskRoutedHoldout(master, trainTasks)
	routedEval := EvalTaskRoutedHoldout(master, evalTasks)
	ensTrainTest := EvalARCHoldout(master, trainTasks) // cross-shard avg (reference only)
	ensEval := EvalARCHoldout(master, evalTasks)

	logf("── TASKS FINISHED (primary ARC metric) ──")
	logf("  oracle demos:              %d / %d  (%.2f%%)   ← shard expert per demo",
		oracleDemo.TasksSolved, oracleDemo.NTasks, oracleDemo.TaskExact)
	logf("  task-routed demos:         %d / %d  (%.2f%%)   ← best specialist on that task's demos",
		routedDemo.TasksSolved, routedDemo.NTasks, routedDemo.TaskExact)
	logf("  task-routed TRAIN-TEST:    %d / %d  (%.2f%%)   ← route on demos → predict train/*.json tests",
		routedTrainTest.TasksSolved, routedTrainTest.NTasks, routedTrainTest.TaskExact)
	logf("  task-routed EVAL:          %d / %d  (%.2f%%)   ← route on demos → predict evaluation/*.json",
		routedEval.TasksSolved, routedEval.NTasks, routedEval.TaskExact)
	logf("")

	logf("── detail · oracle demos (MNIST-style shard coverage) ──")
	for _, line := range FormatScore("", oracleDemo) {
		logf("%s", line)
	}
	logf("")
	logf("── detail · task-routed demos ──")
	for _, line := range FormatScore("", routedDemo) {
		logf("%s", line)
	}
	logf("")
	logf("── detail · task-routed TRAIN-TEST ──")
	for _, line := range FormatScore("", routedTrainTest) {
		logf("%s", line)
	}
	logf("")
	logf("── detail · task-routed EVAL (held out tasks) ──")
	for _, line := range FormatScore("", routedEval) {
		logf("%s", line)
	}
	logf("")
	logf("── reference · naive ensemble average (usually ~0 exact on ARC) ──")
	logf("  train-test tasks finished %d/%d   EVAL tasks finished %d/%d",
		ensTrainTest.TasksSolved, ensTrainTest.NTasks, ensEval.TasksSolved, ensEval.NTasks)

	logf("")
	logf("══════════════════════════════════════════════════════════")
	logf("SUMMARY · %s normal Master", ds.Name)
	logf("  ★ TASKS FINISHED")
	logf("      demos (oracle):     %d / %d  (%.2f%%)",
		oracleDemo.TasksSolved, oracleDemo.NTasks, oracleDemo.TaskExact)
	logf("      demos (routed):     %d / %d  (%.2f%%)",
		routedDemo.TasksSolved, routedDemo.NTasks, routedDemo.TaskExact)
	logf("      TRAIN-TEST routed:  %d / %d  (%.2f%%)",
		routedTrainTest.TasksSolved, routedTrainTest.NTasks, routedTrainTest.TaskExact)
	logf("      EVAL routed:        %d / %d  (%.2f%%)",
		routedEval.TasksSolved, routedEval.NTasks, routedEval.TaskExact)
	logf("  ★ pairs / pixel (oracle demos) = %d/%d (%.2f%%) exact · %.2f%% pixel",
		oracleDemo.PairsSolved, oracleDemo.NPairs, oracleDemo.PairExact, oracleDemo.PixelPct)
	logf("══════════════════════════════════════════════════════════")

	if master.Recovered != cfg.K {
		logf("\n  FAIL: recovered %d/%d", master.Recovered, cfg.K)
		return false
	}
	if oracleDemo.TasksSolved == 0 && oracleDemo.PixelPct < 70 {
		logf("\n  note: weak demo coverage — raise -epochs or -k")
	} else {
		logf("\n✓ Fountain recovered specialists; demo task finishes above are real memorize coverage.")
	}
	logf("✓ Master assembled without layer_seed search (same story as MNIST normal).")
	logf("")
	logf("finished: %s", time.Now().Format(time.RFC3339))
	logf("full log: %s", logPath)
	return true
}

// evalOracleDemos scores each demo with the specialist that owns its batch index.
func evalOracleDemos(master *poly.FountainMaster, tasks []*Task, batches []poly.TrainingBatch[float32]) ARCScore {
	var s ARCScore
	if master == nil || len(tasks) == 0 {
		return s
	}
	bi := 0
	var pixSum float64
	for _, t := range tasks {
		if len(t.Train) == 0 {
			continue
		}
		s.NTasks++
		taskOK := true
		hits := 0
		for _, p := range t.Train {
			s.NPairs++
			var pred [][]int
			if bi < len(batches) && bi < len(master.ShardOf) {
				out, err := master.OracleForward(bi, batches[bi].Input)
				if err == nil && out != nil {
					pred = DecodeGrid(out.Data)
				} else {
					pred = DecodeGrid(nil)
				}
			} else {
				pred = predictPair(master, p)
			}
			acc := PixelAccuracy(pred, p.Output)
			pixSum += acc
			if ExactMatch(pred, p.Output) {
				s.PairsSolved++
				hits++
			} else {
				taskOK = false
			}
			bi++
		}
		if taskOK {
			s.TasksSolved++
		}
		if hits > 0 {
			s.TasksPartial++
		}
	}
	return s.finalize(pixSum)
}

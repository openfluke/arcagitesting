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

type Dataset struct {
	Name     string // "ARC-AGI-1" | "ARC-AGI-2"
	TrainDir string // specialize + demos + train-test scoring
	EvalDir  string // official evaluation split (holdout tasks)
}

type SpectrumConfig struct {
	Quick      bool
	LogPath    string
	K          int
	Epochs     int
	MaxTasks   int // 0 = all training tasks
	MaxEval    int // 0 = all evaluation tasks
	LossRate   float64
	MinPixel   float64 // soft quality gate on train-test pixel% (grid families only)
	Strict     bool
	Families   []string
	DTypes     []poly.DType
}

func DefaultSpectrumConfig() SpectrumConfig {
	return SpectrumConfig{
		K:        8,
		Epochs:   25,
		MaxTasks: 0, // full training set
		MaxEval:  0, // full evaluation set
		LossRate: 0.25,
		MinPixel: 40,
		// ARC bake-off default: only families that decode real grids.
		// Use -all-families for the weight-transport spectrum (swiglu/cnn/…).
		Families: []string{"dense", "residual"},
	}
}

type caseResult struct {
	Family    string
	DType     string
	OK        bool
	QualityOK bool
	GridAble  bool // dense/residual — real ARC grid scores
	Err       string
	BlobBytes int
	Recovered int
	Received  int
	Sprayed   int
	K         int
	SpecializeUs int64
	FountainUs   int64
	TotalUs      int64
	MinGate   float64

	// ARC bake-off (grid families only). Non-grid = fountain transport only.
	Demo      ARCScore // training demos (memorization)
	TrainTest ARCScore // training/*.json test pairs (within-task holdout)
	Eval      ARCScore // evaluation/*.json test pairs (OOD tasks)
}

// RunDatasetSpectrum: Neural Fountain spectrum on one ARC corpus.
func RunDatasetSpectrum(ds Dataset, cfg SpectrumConfig) bool {
	if cfg.Quick {
		// Quick = shorter train + fewer dtypes. Never shrink the ARC corpus —
		// train/eval stay full unless the user passes -tasks / -eval-tasks.
		if cfg.K == 8 {
			cfg.K = 4
		}
		if cfg.Epochs == 25 {
			cfg.Epochs = 10
		}
		if cfg.MinPixel == 40 {
			cfg.MinPixel = 25
		}
	}
	if cfg.K < 2 {
		cfg.K = 2
	}
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}

	logPath := cfg.LogPath
	if logPath == "" {
		_ = os.MkdirAll("logs", 0o755)
		safe := strings.ReplaceAll(ds.Name, " ", "_")
		logPath = filepath.Join("logs", fmt.Sprintf("%s_fountain_spectrum_%s.log",
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

	logf("╔══════════════════════════════════════════════════════════════╗")
	logf("║  ARC Neural Fountain SPECTRUM — %s", ds.Name)
	logf("╚══════════════════════════════════════════════════════════════╝")
	logf("started: %s", time.Now().Format(time.RFC3339))
	logf("train: %s", ds.TrainDir)
	logf("eval:  %s", ds.EvalDir)
	logf("log: %s", logPath)
	logf("pipeline: specialize(train) → L1 LT Master → L2 MEGA fountain → train+eval bake-off")
	logf("honesty: only dense/residual score ARC grids; other families = weight-transport probes")
	tasksLabel := "all"
	if cfg.MaxTasks > 0 {
		tasksLabel = fmt.Sprintf("≤%d", cfg.MaxTasks)
	}
	evalLabel := "all"
	if cfg.MaxEval > 0 {
		evalLabel = fmt.Sprintf("≤%d", cfg.MaxEval)
	}
	logf("config: K=%d epochs=%d train_tasks=%s eval_tasks=%s canvas=%dx%d loss=%.2f min_pixel=%.0f%% quick=%v",
		cfg.K, cfg.Epochs, tasksLabel, evalLabel, MaxCanvas, MaxCanvas, cfg.LossRate, cfg.MinPixel, cfg.Quick)

	trainTasks, err := LoadTasks(ds.TrainDir, cfg.MaxTasks)
	if err != nil {
		logf("FAIL load train: %v", err)
		return false
	}
	logf("loaded TRAIN tasks: %d  (specialize on demos only)", len(trainTasks))

	var evalTasks []*Task
	if ds.EvalDir != "" {
		evalTasks, err = LoadTasks(ds.EvalDir, cfg.MaxEval)
		if err != nil {
			logf("WARN load eval: %v — eval bake-off will be empty", err)
			evalTasks = nil
		} else {
			logf("loaded EVAL  tasks: %d  (holdout — not used for specialize)", len(evalTasks))
		}
	} else {
		logf("WARN: no EvalDir — eval bake-off skipped")
	}

	families := SpectrumFamilies()
	dtypes := poly.SeedDTypesAll()
	if len(cfg.Families) > 0 {
		want := map[string]bool{}
		for _, x := range cfg.Families {
			want[strings.ToLower(x)] = true
		}
		filtered := families[:0]
		for _, fam := range families {
			if want[fam.name] {
				filtered = append(filtered, fam)
			}
		}
		families = filtered
		if len(families) == 0 {
			logf("FAIL: no families matched %v", cfg.Families)
			return false
		}
	}
	if len(cfg.DTypes) > 0 {
		dtypes = cfg.DTypes
	} else if cfg.Quick {
		dtypes = quickDTypes()
		logf("QUICK dtypes: %d", len(dtypes))
	}

	type specCase struct {
		Family string
		DType  poly.DType
	}
	var cases []specCase
	for _, fam := range families {
		for _, dt := range dtypes {
			cases = append(cases, specCase{Family: fam.name, DType: dt})
		}
	}
	logf("total cases: %d (%d families × %d dtypes)", len(cases), len(families), len(dtypes))
	logf("")

	results := make([]caseResult, 0, len(cases))
	cargos := make([]masterCargo, 0, len(cases))
	start := time.Now()
	pipeOK, qualOK, weak, fail, transportOK := 0, 0, 0, 0, 0

	for i, c := range cases {
		logf("── case %d/%d  family=%s  dtype=%s ──", i+1, len(cases), c.Family, c.DType.String())
		res, cargo := runCase(c.Family, c.DType, trainTasks, evalTasks, cfg, logf)
		results = append(results, res)
		if cargo != nil {
			cargos = append(cargos, *cargo)
		}
		if !res.OK {
			fail++
			logf("  FAIL  err=%s  total=%.1fms", res.Err, float64(res.TotalUs)/1000)
		} else if !res.GridAble {
			pipeOK++
			transportOK++
			logf("  CONSOLIDATED  recovered=%d/%d  recv=%d sprayed=%d  blob=%dB",
				res.Recovered, res.K, res.Received, res.Sprayed, res.BlobBytes)
			logf("  TRANSPORT  (not ARC grid — weight fountain only)  specialize=%.1fms fountain=%.1fms",
				float64(res.SpecializeUs)/1000, float64(res.FountainUs)/1000)
		} else if res.QualityOK {
			pipeOK++
			qualOK++
			logf("  CONSOLIDATED  recovered=%d/%d  recv=%d sprayed=%d  blob=%dB",
				res.Recovered, res.K, res.Received, res.Sprayed, res.BlobBytes)
			logf("  PASS  %s", formatARCTrio(res.Demo, res.TrainTest, res.Eval))
			logf("        specialize=%.1fms fountain=%.1fms",
				float64(res.SpecializeUs)/1000, float64(res.FountainUs)/1000)
		} else {
			pipeOK++
			weak++
			logf("  CONSOLIDATED  recovered=%d/%d  recv=%d sprayed=%d  blob=%dB",
				res.Recovered, res.K, res.Received, res.Sprayed, res.BlobBytes)
			logf("  WEAK  train-test pixel < %.0f%% — weights recovered; ARC still soft", res.MinGate)
			logf("        %s", formatARCTrio(res.Demo, res.TrainTest, res.Eval))
		}
		logf("")
	}

	elapsed := time.Since(start)
	logf("══════════════════════════════════════════════════════════════")
	logf("SUMMARY · %s L1", ds.Name)
	logf("  pipeline_ok=%d  grid_pass=%d  grid_weak=%d  transport=%d  fail=%d  wall=%s",
		pipeOK, qualOK, weak, transportOK, fail, elapsed.Round(time.Millisecond))
	logf("")
	logf("%-12s %-12s %-9s %10s %10s %10s %8s %8s %s",
		"FAMILY", "DTYPE", "STATUS", "DEMO_T", "TRN_TEST", "EVAL_T", "SPEC_ms", "BLOB", "NOTE")
	for _, r := range results {
		status, note := "PASS", ""
		if !r.OK {
			status = "FAIL"
			note = r.Err
			if len(note) > 36 {
				note = note[:33] + "..."
			}
		} else if !r.GridAble {
			status = "TRANSPORT"
			note = "no ARC score"
		} else if !r.QualityOK {
			status = "WEAK"
			note = fmt.Sprintf("<%.0f%% trn-test px", r.MinGate)
		}
		demoT, trainT, evalT := "—", "—", "—"
		if r.GridAble && r.OK {
			demoT = fmt.Sprintf("%d/%d", r.Demo.TasksSolved, r.Demo.NTasks)
			trainT = fmt.Sprintf("%d/%d", r.TrainTest.TasksSolved, r.TrainTest.NTasks)
			evalT = fmt.Sprintf("%d/%d", r.Eval.TasksSolved, r.Eval.NTasks)
		}
		logf("%-12s %-12s %-9s %10s %10s %10s %8.1f %8d %s",
			r.Family, r.DType, status, demoT, trainT, evalT,
			float64(r.SpecializeUs)/1000, r.BlobBytes, note)
	}

	// Best grid Masters by eval task solves (then train-test).
	logBestGridBakeoff(logf, results)

	// Level-2: fountain the Masters themselves.
	megaOK := RunMegaConsolidate(logf, cargos, trainTasks, evalTasks, cfg)

	logf("")
	logf("finished: %s", time.Now().Format(time.RFC3339))
	logf("full log: %s", logPath)

	csvPath := strings.TrimSuffix(logPath, filepath.Ext(logPath)) + ".csv"
	_ = writeCSV(csvPath, results)
	logf("csv: %s", csvPath)

	if fail > 0 {
		return false
	}
	if cfg.Strict && weak > 0 {
		return false
	}
	if !megaOK {
		return false
	}
	return true
}

func runCase(family string, dt poly.DType, trainTasks, evalTasks []*Task, cfg SpectrumConfig, logf func(string, ...any)) (res caseResult, cargo *masterCargo) {
	gridAble := family == "dense" || family == "residual"
	res = caseResult{
		Family:   family,
		DType:    dt.String(),
		MinGate:  cfg.MinPixel,
		K:        cfg.K,
		GridAble: gridAble,
	}
	start := time.Now()
	defer func() { res.TotalUs = time.Since(start).Microseconds() }()

	task, err := BuildFactoryAndBatches(family, dt, trainTasks, cfg.K)
	if err != nil {
		res.Err = "setup: " + err.Error()
		return
	}
	if len(task.TrainBatches) < cfg.K {
		res.Err = fmt.Sprintf("not enough demos (%d) for K=%d", len(task.TrainBatches), cfg.K)
		return
	}

	pcfg := poly.DefaultNeuralFountainConfig()
	pcfg.K = cfg.K
	pcfg.Epochs = cfg.Epochs
	pcfg.LossRate = cfg.LossRate
	pcfg.LossType = task.LossType
	pcfg.LR = 0.12
	pcfg.UseExactDType = false
	pcfg.UniformDType = 0
	pcfg.Verbose = false
	pcfg.MaxOverhead = 8
	pcfg.Seed = poly.SeedFrom("arc-nf", family, dt.String())

	master, err := poly.NeuralFountain(task.Factory, task.TrainBatches, pcfg)
	if err != nil {
		res.Err = err.Error()
		return
	}
	res.Recovered = master.Recovered
	res.Received = master.Received
	res.Sprayed = master.Sprayed
	res.SpecializeUs = master.SpecializeUs
	res.FountainUs = master.FountainUs

	if packed, err := packMasterCargo(master); err == nil {
		res.BlobBytes = len(packed)
		cargo = &masterCargo{
			Family:   family,
			DType:    dt,
			Blob:     packed,
			GridAble: gridAble,
		}
	} else if len(master.Experts) > 0 && master.Experts[0] != nil {
		if blob, e := poly.PackNetworkWeights(master.Experts[0]); e == nil {
			res.BlobBytes = len(blob)
		}
	}

	for _, e := range master.Experts {
		if e == nil {
			continue
		}
		poly.ApplyUniformDType(e, dt)
		poly.MorphNetworkToLayerDTypes(e)
	}

	res.OK = master.Recovered == cfg.K
	if !res.OK {
		res.Err = fmt.Sprintf("recovered %d/%d", master.Recovered, cfg.K)
		cargo = nil
		_ = logf
		return
	}

	if gridAble {
		res.Demo = EvalARCDemos(master, trainTasks)
		res.TrainTest = EvalARCHoldout(master, trainTasks)
		res.Eval = EvalARCHoldout(master, evalTasks)
		res.QualityOK = res.TrainTest.PixelPct >= cfg.MinPixel
		if cargo != nil {
			cargo.TrainTest = res.TrainTest
			cargo.Eval = res.Eval
			cargo.Demo = res.Demo
		}
	} else {
		// Fountain recovered — transport probe only. Do NOT invent ARC "solves".
		res.QualityOK = true
	}
	_ = logf
	return
}

func logBestGridBakeoff(logf func(string, ...any), results []caseResult) {
	bestEval, bestTrain := -1, -1
	for i, r := range results {
		if !r.OK || !r.GridAble {
			continue
		}
		if bestEval < 0 ||
			r.Eval.TasksSolved > results[bestEval].Eval.TasksSolved ||
			(r.Eval.TasksSolved == results[bestEval].Eval.TasksSolved &&
				r.Eval.PixelPct > results[bestEval].Eval.PixelPct) {
			bestEval = i
		}
		if bestTrain < 0 ||
			r.TrainTest.TasksSolved > results[bestTrain].TrainTest.TasksSolved ||
			(r.TrainTest.TasksSolved == results[bestTrain].TrainTest.TasksSolved &&
				r.TrainTest.PixelPct > results[bestTrain].TrainTest.PixelPct) {
			bestTrain = i
		}
	}
	logf("")
	logf("── L1 BAKE-OFF (best single Master after specialize+consolidation) ──")
	if bestTrain < 0 {
		logf("  (no grid Masters)")
		return
	}
	bt := results[bestTrain]
	logf("  best on TRAIN-TEST: %s/%s  tasks=%d/%d  pixel=%.1f%%  pair_exact=%.1f%%",
		bt.Family, bt.DType, bt.TrainTest.TasksSolved, bt.TrainTest.NTasks,
		bt.TrainTest.PixelPct, bt.TrainTest.PairExact)
	if bestEval >= 0 {
		be := results[bestEval]
		logf("  best on EVAL:       %s/%s  tasks=%d/%d  pixel=%.1f%%  pair_exact=%.1f%%",
			be.Family, be.DType, be.Eval.TasksSolved, be.Eval.NTasks,
			be.Eval.PixelPct, be.Eval.PairExact)
	}
}

func formatARCTrio(demo, trainTest, evalS ARCScore) string {
	return fmt.Sprintf("demos=%d/%d tasks (%.0f%%px) | train-test=%d/%d (%.0f%%px/%.0f%%exact) | EVAL=%d/%d (%.0f%%px/%.0f%%exact)",
		demo.TasksSolved, demo.NTasks, demo.PixelPct,
		trainTest.TasksSolved, trainTest.NTasks, trainTest.PixelPct, trainTest.PairExact,
		evalS.TasksSolved, evalS.NTasks, evalS.PixelPct, evalS.PairExact)
}

func writeCSV(path string, results []caseResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintln(f, "family,dtype,ok,quality_ok,grid_able,err,"+
		"demo_tasks_solved,demo_n_tasks,demo_pixel,demo_pair_exact,"+
		"train_tasks_solved,train_n_tasks,train_pixel,train_pair_exact,"+
		"eval_tasks_solved,eval_n_tasks,eval_pixel,eval_pair_exact,"+
		"blob,recovered,received,sprayed,specialize_us,fountain_us,total_us")
	for _, r := range results {
		errEsc := strings.ReplaceAll(r.Err, `"`, `'`)
		fmt.Fprintf(f, "%s,%s,%t,%t,%t,\"%s\",%d,%d,%.3f,%.3f,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d\n",
			r.Family, r.DType, r.OK, r.QualityOK, r.GridAble, errEsc,
			r.Demo.TasksSolved, r.Demo.NTasks, r.Demo.PixelPct, r.Demo.PairExact,
			r.TrainTest.TasksSolved, r.TrainTest.NTasks, r.TrainTest.PixelPct, r.TrainTest.PairExact,
			r.Eval.TasksSolved, r.Eval.NTasks, r.Eval.PixelPct, r.Eval.PairExact,
			r.BlobBytes, r.Recovered, r.Received, r.Sprayed, r.SpecializeUs, r.FountainUs, r.TotalUs)
	}
	return nil
}

func quickDTypes() []poly.DType {
	return []poly.DType{
		poly.DTypeFloat32,
		poly.DTypeFloat16,
		poly.DTypeBFloat16,
		poly.DTypeInt8,
		poly.DTypeInt4,
		poly.DTypeTernary,
		poly.DTypeBinary,
	}
}

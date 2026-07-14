package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/openfluke/loom/arcagitesting/fountain_spectrum/arcspec"
	"github.com/openfluke/loom/poly"
)

func main() {
	root, _ := os.Getwd()
	agi1Train := filepath.Join(root, "..", "ARC-AGI", "data", "training")
	agi1Eval := filepath.Join(root, "..", "ARC-AGI", "data", "evaluation")
	agi2Train := filepath.Join(root, "..", "ARC-AGI2", "data", "training")
	agi2Eval := filepath.Join(root, "..", "ARC-AGI2", "data", "evaluation")

	mode := "" // "normal" | "spectrum"
	only := "" // "" = both, "1", "2"
	quick := false
	allFam := false
	logPath := ""
	var families []string
	var dtypes []poly.DType

	normal := arcspec.DefaultNormalConfig()
	spec := arcspec.DefaultSpectrumConfig()

	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "normal" || a == "mnist" || a == "master":
			mode = "normal"
		case a == "spectrum" || a == "layers" || a == "showcase":
			mode = "spectrum"
		case a == "1" || a == "arc1" || a == "agi1":
			only = "1"
		case a == "2" || a == "arc2" || a == "agi2":
			only = "2"
		case a == "both" || a == "all":
			only = ""
		case a == "-quick" || a == "--quick" || a == "quick":
			quick = true
		case a == "-strict" || a == "--strict":
			spec.Strict = true
		case a == "-all-families" || a == "--all-families" || a == "transport":
			allFam = true
		case a == "-k" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 1 {
				normal.K = v
				spec.K = v
			}
		case a == "-epochs" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 0 {
				normal.Epochs = v
				spec.Epochs = v
			}
		case a == "-loss" && i+1 < len(args):
			i++
			if v, err := strconv.ParseFloat(args[i], 64); err == nil {
				normal.LossRate = v
				spec.LossRate = v
			}
		case a == "-tasks" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 0 {
				normal.MaxTasks = v
				spec.MaxTasks = v
			}
		case a == "-eval-tasks" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 0 {
				normal.MaxEval = v
				spec.MaxEval = v
			}
		case a == "-min-pixel" && i+1 < len(args):
			i++
			if v, err := strconv.ParseFloat(args[i], 64); err == nil {
				spec.MinPixel = v
			}
		case a == "-family" && i+1 < len(args):
			i++
			families = splitCSV(args[i])
		case a == "-dtype" && i+1 < len(args):
			i++
			dts, err := parseDTypes(args[i])
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(2)
			}
			dtypes = dts
		case a == "-log" && i+1 < len(args):
			i++
			logPath = args[i]
		case a == "-h" || a == "--help" || a == "help":
			printHelp()
			return
		default:
			fmt.Fprintf(os.Stderr, "unknown arg %s\n", a)
			printHelp()
			os.Exit(2)
		}
	}

	if mode == "" {
		mode = "normal" // default like loom_neural_fountain
	}

	ok := true
	runNormal := func(name, trainDir, evalDir string) {
		ds := arcspec.Dataset{Name: name, TrainDir: trainDir, EvalDir: evalDir}
		c := normal
		c.Quick = quick
		c.LogPath = logPath
		fmt.Println()
		if !arcspec.RunNormalDataset(ds, c) {
			ok = false
		}
	}
	runSpec := func(name, trainDir, evalDir string) {
		ds := arcspec.Dataset{Name: name, TrainDir: trainDir, EvalDir: evalDir}
		c := spec
		c.Quick = quick
		c.LogPath = logPath
		if allFam {
			c.Families = nil
		} else if len(families) > 0 {
			c.Families = families
		}
		if len(dtypes) > 0 {
			c.DTypes = dtypes
		}
		fmt.Println()
		if !arcspec.RunDatasetSpectrum(ds, c) {
			ok = false
		}
	}

	runOne := runNormal
	if mode == "spectrum" {
		runOne = runSpec
	}

	switch only {
	case "1":
		runOne("ARC-AGI-1", agi1Train, agi1Eval)
	case "2":
		runOne("ARC-AGI-2", agi2Train, agi2Eval)
	default:
		fmt.Println("═══ Phase A: ARC-AGI-1 ═══")
		runOne("ARC-AGI-1", agi1Train, agi1Eval)
		fmt.Println()
		fmt.Println("═══ Phase B: ARC-AGI-2 ═══")
		runOne("ARC-AGI-2", agi2Train, agi2Eval)
	}

	if !ok {
		os.Exit(1)
	}
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(strings.ToLower(p))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseDTypes(s string) ([]poly.DType, error) {
	names := splitCSV(s)
	out := make([]poly.DType, 0, len(names))
	for _, name := range names {
		dt := poly.ParseDType(name)
		matched := false
		for _, known := range poly.SeedDTypesAll() {
			if known == dt || strings.EqualFold(known.String(), name) {
				out = append(out, known)
				matched = true
				break
			}
		}
		if !matched {
			return nil, fmt.Errorf("unknown dtype %q", name)
		}
	}
	return out, nil
}

func printHelp() {
	fmt.Fprintf(os.Stderr, `ARC-AGI Neural Fountain

Same two modes as loom_neural_fountain:

  normal   — one dense Master (K specialists → LT → ensemble), like ./run.sh 1 there
  spectrum — family×dtype bake-off + mega fountain

Specialize on ALL training demos. EVAL is scored only — never trained on.

Usage:
  go run . normal 1              # AGI-1 MNIST-style Master (default mode)
  go run . normal 1 quick        # K=8 · 3 epochs · still all 400+400
  go run . spectrum 1 quick      # dense+residual × 7 dtypes · full corpus
  go run . spectrum 1 transport  # all families

Flags:
  normal | spectrum     run mode (default: normal)
  1 | 2 | both          which ARC corpus
  -quick / quick        normal: K=8/3ep · spectrum: 7 dtypes / less K·ep
  -k N                  specialists (normal default 16)
  -epochs N             epochs / specialist (normal default 5)
  -loss R               fountain erase rate (normal default 0.30)
  -tasks N / -eval-tasks N   cap corpus (default 0 = all)
  -all-families         spectrum: include transport probes
  -family / -dtype      spectrum filters
  -log PATH
`)
}

package arcspec

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
)

// masterCargo is one spectrum Master (all K specialist packs) for level-2 fountain.
type masterCargo struct {
	Family    string
	DType     poly.DType
	Blob      []byte // packed K experts
	GridAble  bool   // dense/residual — can score ARC grids
	Demo      ARCScore
	TrainTest ARCScore
	Eval      ARCScore
}

func packMasterCargo(m *poly.FountainMaster) ([]byte, error) {
	if m == nil || len(m.Experts) == 0 {
		return nil, fmt.Errorf("empty master")
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(m.Experts))); err != nil {
		return nil, err
	}
	for i, e := range m.Experts {
		if e == nil {
			return nil, fmt.Errorf("nil expert %d", i)
		}
		b, err := poly.PackNetworkWeights(e)
		if err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(b))); err != nil {
			return nil, err
		}
		if _, err := buf.Write(b); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func unpackMasterCargo(blob []byte) ([][]byte, error) {
	r := bytes.NewReader(blob)
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	out := make([][]byte, 0, n)
	for i := uint32(0); i < n; i++ {
		var ln uint32
		if err := binary.Read(r, binary.LittleEndian, &ln); err != nil {
			return nil, err
		}
		b := make([]byte, ln)
		if _, err := r.Read(b); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

func padBlobsEqual(blobs [][]byte) [][]byte {
	max := 0
	for _, b := range blobs {
		if len(b) > max {
			max = len(b)
		}
	}
	out := make([][]byte, len(blobs))
	for i, b := range blobs {
		p := make([]byte, max)
		copy(p, b)
		out[i] = p
	}
	return out
}

// RunMegaConsolidate: Level-2 fountain — consolidate all Level-1 Masters into one zoo,
// then score ARC demos / train-test / eval with dense/residual voters.
func RunMegaConsolidate(logf func(string, ...any), cargos []masterCargo, trainTasks, evalTasks []*Task, cfg SpectrumConfig) bool {
	if len(cargos) < 2 {
		logf("")
		logf("── MEGA consolidate skipped (need ≥2 Masters, got %d) ──", len(cargos))
		return true
	}

	logf("")
	logf("╔══════════════════════════════════════════════════════════════╗")
	logf("║  MEGA FOUNTAIN — consolidate the consolidated Masters        ║")
	logf("╚══════════════════════════════════════════════════════════════╝")
	logf("Level-2: every L1 Master blob → LT spray/peel → rebuild zoo → train/eval bake-off")
	logf("masters in: %d  (families×dtypes that recovered at Level-1)", len(cargos))

	raw := make([][]byte, len(cargos))
	for i, c := range cargos {
		raw[i] = c.Blob
		if c.GridAble {
			logf("  cargo[%d] %s/%s  %dB  GRID  L1 eval tasks=%d/%d (%.1f%%px)",
				i, c.Family, c.DType.String(), len(c.Blob),
				c.Eval.TasksSolved, c.Eval.NTasks, c.Eval.PixelPct)
		} else {
			logf("  cargo[%d] %s/%s  %dB  transport-only",
				i, c.Family, c.DType.String(), len(c.Blob))
		}
	}
	padded := padBlobsEqual(raw)
	logf("padded block size: %dB × K=%d", len(padded[0]), len(padded))

	seed := poly.SeedFrom("arc-mega-fountain", dsNameHint(cargos), uint64(len(padded)), uint64(len(padded[0])))
	start := time.Now()
	recovered, recv, sprayed, err := poly.RecoverWeightBlobs(padded, seed, cfg.LossRate, 8.0)
	megaMs := time.Since(start)
	if err != nil {
		logf("  FAIL mega LT: %v", err)
		return false
	}
	logf("  MEGA CONSOLIDATED  recovered=%d/%d  recv=%d sprayed=%d  wall=%s",
		len(recovered), len(padded), recv, sprayed, megaMs.Round(time.Millisecond))

	exactBlobs := 0
	for i := range recovered {
		if bytes.Equal(recovered[i], padded[i]) {
			exactBlobs++
		}
	}
	logf("  mega blob byte-exact: %d/%d", exactBlobs, len(recovered))

	var gridMasters []*poly.FountainMaster
	var gridTags []string
	rebuildOK := 0
	for i, c := range cargos {
		experts, err := unpackMasterCargo(trimPad(recovered[i], len(c.Blob)))
		if err != nil {
			logf("  cargo[%d] unpack fail: %v", i, err)
			continue
		}
		task, err := BuildFactoryAndBatches(c.Family, c.DType, trainTasks, cfg.K)
		if err != nil {
			logf("  cargo[%d] factory fail: %v", i, err)
			continue
		}
		nets := make([]*poly.VolumetricNetwork, len(experts))
		ok := true
		for j, eb := range experts {
			net, err := task.Factory(j)
			if err != nil {
				ok = false
				break
			}
			poly.WireNetworkLayers(net)
			net.ReleaseFP32MasterWhenIdle = false
			_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
			net.EnsureTrainingWeights()
			if err := poly.UnpackNetworkWeights(net, eb); err != nil {
				ok = false
				break
			}
			nets[j] = net
		}
		if !ok {
			logf("  cargo[%d] %s/%s rebuild FAIL", i, c.Family, c.DType.String())
			continue
		}
		rebuildOK++
		m := &poly.FountainMaster{Experts: nets, K: len(nets), Recovered: len(nets)}
		if c.GridAble {
			gridMasters = append(gridMasters, m)
			gridTags = append(gridTags, c.Family+"/"+c.DType.String())
		}
	}
	logf("  zoo rebuilt: %d/%d Masters  grid-voters: %d", rebuildOK, len(cargos), len(gridMasters))

	if len(gridMasters) == 0 {
		logf("  MEGA ARC bake-off skipped (no dense/residual Master for grids)")
		logf("  (Mega still proved Level-2 LT can ship every Level-1 Master blob)")
		return exactBlobs == len(recovered) && rebuildOK == len(cargos)
	}

	demoVote, _ := EvalMegaVoteDemos(gridMasters, gridTags, trainTasks)
	trainVote, bestTrain := EvalMegaVoteHoldout(gridMasters, gridTags, trainTasks)
	evalVote, bestEval := EvalMegaVoteHoldout(gridMasters, gridTags, evalTasks)

	logf("")
	logf("── L2 MEGA BAKE-OFF (dense/residual zoo majority vote) ──")
	logf("  demos (train signal):")
	for _, line := range FormatScore("", demoVote) {
		logf("%s", line)
	}
	logf("  train-test (train/*.json):")
	for _, line := range FormatScore("", trainVote) {
		logf("%s", line)
	}
	if bestTrain != "" {
		logf("    best single voter on train-test: %s", bestTrain)
	}
	logf("  EVAL (evaluation/*.json):")
	for _, line := range FormatScore("", evalVote) {
		logf("%s", line)
	}
	if bestEval != "" {
		logf("    best single voter on EVAL: %s", bestEval)
	}
	logf("  hive size: %d layer×dtype Masters in mega zoo (%d grid voters)", rebuildOK, len(gridMasters))

	return exactBlobs == len(recovered)
}

func trimPad(b []byte, n int) []byte {
	if n <= 0 || n > len(b) {
		return b
	}
	return b[:n]
}

func dsNameHint(cargos []masterCargo) string {
	if len(cargos) == 0 {
		return "empty"
	}
	return cargos[0].Family
}

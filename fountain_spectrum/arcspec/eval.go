package arcspec

import (
	"fmt"

	"github.com/openfluke/loom/poly"
)

// ARCScore is one split evaluation (demos or holdout tests).
type ARCScore struct {
	PixelPct     float64 // mean pixel accuracy over pairs
	PairExact    float64 // % of I/O pairs exactly correct
	TaskExact    float64 // % of tasks with ALL pairs exact
	PartialExact float64 // % of tasks with ≥1 exact pair
	PairsSolved  int
	NPairs       int
	TasksSolved  int // all pairs exact
	TasksPartial int // ≥1 pair exact
	NTasks       int
}

func (s ARCScore) finalize(pixSum float64) ARCScore {
	if s.NPairs > 0 {
		s.PixelPct = pixSum / float64(s.NPairs)
		s.PairExact = 100 * float64(s.PairsSolved) / float64(s.NPairs)
	}
	if s.NTasks > 0 {
		s.TaskExact = 100 * float64(s.TasksSolved) / float64(s.NTasks)
		s.PartialExact = 100 * float64(s.TasksPartial) / float64(s.NTasks)
	}
	return s
}

// FormatScore lines for logs — pair% is the MNIST-analogue accuracy.
func FormatScore(title string, s ARCScore) []string {
	return []string{
		fmt.Sprintf("  %-22s mean pixel        = %6.2f%%", title, s.PixelPct),
		fmt.Sprintf("  %-22s pairs exact       = %d/%d  (%.2f%%)   ← grid I/O fully correct",
			"", s.PairsSolved, s.NPairs, s.PairExact),
		fmt.Sprintf("  %-22s tasks finished    = %d/%d  (%.2f%%)   ← ALL pairs in task exact",
			"", s.TasksSolved, s.NTasks, s.TaskExact),
		fmt.Sprintf("  %-22s tasks any-hit     = %d/%d  (%.2f%%)   ← ≥1 exact pair in task",
			"", s.TasksPartial, s.NTasks, s.PartialExact),
	}
}

// EvalARCHoldout scores every task's Test pairs (official-style holdout within a task file).
func EvalARCHoldout(master *poly.FountainMaster, tasks []*Task) ARCScore {
	return evalARC(master, tasks, false)
}

// EvalARCDemos scores every task's Train demonstration pairs (memorization / coverage).
func EvalARCDemos(master *poly.FountainMaster, tasks []*Task) ARCScore {
	return evalARC(master, tasks, true)
}

func evalARC(master *poly.FountainMaster, tasks []*Task, demos bool) ARCScore {
	var s ARCScore
	if master == nil || len(tasks) == 0 {
		return s
	}
	var pixSum float64
	for _, t := range tasks {
		pairs := t.Test
		if demos {
			pairs = t.Train
		}
		if len(pairs) == 0 {
			continue
		}
		s.NTasks++
		taskOK := true
		hits := 0
		for _, p := range pairs {
			s.NPairs++
			pred := predictPair(master, p)
			acc := PixelAccuracy(pred, p.Output)
			pixSum += acc
			if ExactMatch(pred, p.Output) {
				s.PairsSolved++
				hits++
			} else {
				taskOK = false
			}
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

func predictPair(master *poly.FountainMaster, p GridPair) [][]int {
	in := poly.NewTensorFromSlice(EncodeGrid(p.Input), 1, GridDim)
	out, err := master.Forward(in)
	if err != nil || out == nil {
		return DecodeGrid(nil)
	}
	return DecodeGrid(out.Data)
}

func predictExpert(e *poly.VolumetricNetwork, p GridPair) [][]int {
	if e == nil {
		return DecodeGrid(nil)
	}
	in := poly.NewTensorFromSlice(EncodeGrid(p.Input), 1, GridDim)
	out, _, _ := poly.ForwardPolymorphic(e, in)
	if out == nil {
		return DecodeGrid(nil)
	}
	return DecodeGrid(out.Data)
}

// pickExpertForTask chooses the specialist that best fits this task's demos
// (exact pair count, then mean pixel). ARC-style: use demos to route, then predict tests.
func pickExpertForTask(master *poly.FountainMaster, demos []GridPair) int {
	if master == nil || len(master.Experts) == 0 || len(demos) == 0 {
		return 0
	}
	bestI, bestExact, bestPix := 0, -1, -1.0
	for i, e := range master.Experts {
		if e == nil {
			continue
		}
		exact := 0
		pix := 0.0
		for _, p := range demos {
			pred := predictExpert(e, p)
			pix += PixelAccuracy(pred, p.Output)
			if ExactMatch(pred, p.Output) {
				exact++
			}
		}
		pix /= float64(len(demos))
		if exact > bestExact || (exact == bestExact && pix > bestPix) {
			bestExact, bestPix, bestI = exact, pix, i
		}
	}
	return bestI
}

// EvalTaskRoutedHoldout: for each task, pick specialist by demo fit, score Test pairs.
// This is the deployable ARC-style Master (not cross-shard average).
func EvalTaskRoutedHoldout(master *poly.FountainMaster, tasks []*Task) ARCScore {
	return evalTaskRouted(master, tasks, false)
}

// EvalTaskRoutedDemos: same routing, score the demos themselves (should be strong).
func EvalTaskRoutedDemos(master *poly.FountainMaster, tasks []*Task) ARCScore {
	return evalTaskRouted(master, tasks, true)
}

func evalTaskRouted(master *poly.FountainMaster, tasks []*Task, scoreDemos bool) ARCScore {
	var s ARCScore
	if master == nil || len(tasks) == 0 {
		return s
	}
	var pixSum float64
	for _, t := range tasks {
		if len(t.Train) == 0 {
			continue
		}
		pairs := t.Test
		if scoreDemos {
			pairs = t.Train
		}
		if len(pairs) == 0 {
			continue
		}
		ei := pickExpertForTask(master, t.Train)
		e := master.Experts[ei]
		s.NTasks++
		taskOK := true
		hits := 0
		for _, p := range pairs {
			s.NPairs++
			pred := predictExpert(e, p)
			acc := PixelAccuracy(pred, p.Output)
			pixSum += acc
			if ExactMatch(pred, p.Output) {
				s.PairsSolved++
				hits++
			} else {
				taskOK = false
			}
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

// EvalMegaVoteHoldout majority-votes grid Masters on task Test pairs.
func EvalMegaVoteHoldout(masters []*poly.FountainMaster, tags []string, tasks []*Task) (ARCScore, string) {
	return evalMegaVote(masters, tags, tasks, false)
}

// EvalMegaVoteDemos majority-votes on Train demos.
func EvalMegaVoteDemos(masters []*poly.FountainMaster, tags []string, tasks []*Task) (ARCScore, string) {
	return evalMegaVote(masters, tags, tasks, true)
}

func evalMegaVote(masters []*poly.FountainMaster, tags []string, tasks []*Task, demos bool) (ARCScore, string) {
	var s ARCScore
	bestTag := ""
	if len(masters) == 0 || len(tasks) == 0 {
		return s, bestTag
	}
	bestPix := -1.0
	bestI := -1
	perPix := make([]float64, len(masters))
	perN := make([]int, len(masters))
	var pixSum float64

	for _, t := range tasks {
		pairs := t.Test
		if demos {
			pairs = t.Train
		}
		if len(pairs) == 0 {
			continue
		}
		s.NTasks++
		taskOK := true
		hits := 0
		for _, p := range pairs {
			s.NPairs++
			votes := make([][]map[int]int, MaxCanvas)
			for r := 0; r < MaxCanvas; r++ {
				votes[r] = make([]map[int]int, MaxCanvas)
				for c := 0; c < MaxCanvas; c++ {
					votes[r][c] = map[int]int{}
				}
			}
			for mi, m := range masters {
				pred := predictPair(m, p)
				acc := PixelAccuracy(pred, p.Output)
				perPix[mi] += acc
				perN[mi]++
				th := len(p.Output)
				if th > MaxCanvas {
					th = MaxCanvas
				}
				for r := 0; r < th; r++ {
					tw := len(p.Output[r])
					if tw > MaxCanvas {
						tw = MaxCanvas
					}
					for c := 0; c < tw; c++ {
						if r < len(pred) && c < len(pred[r]) {
							votes[r][c][pred[r][c]]++
						}
					}
				}
			}
			maj := make([][]int, MaxCanvas)
			for r := 0; r < MaxCanvas; r++ {
				maj[r] = make([]int, MaxCanvas)
				for c := 0; c < MaxCanvas; c++ {
					bestC, bestN := 0, -1
					for col, cnt := range votes[r][c] {
						if cnt > bestN {
							bestN = cnt
							bestC = col
						}
					}
					maj[r][c] = bestC
				}
			}
			acc := PixelAccuracy(maj, p.Output)
			pixSum += acc
			if ExactMatch(maj, p.Output) {
				s.PairsSolved++
				hits++
			} else {
				taskOK = false
			}
		}
		if taskOK {
			s.TasksSolved++
		}
		if hits > 0 {
			s.TasksPartial++
		}
	}
	for i := range masters {
		if perN[i] == 0 {
			continue
		}
		avg := perPix[i] / float64(perN[i])
		if avg > bestPix {
			bestPix = avg
			bestI = i
		}
	}
	if bestI >= 0 && bestI < len(tags) {
		bestTag = fmt.Sprintf("%s (%.1f%% px)", tags[bestI], bestPix)
	}
	s = s.finalize(pixSum)
	return s, bestTag
}

package arcspec

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

const (
	MaxCanvas = 16 // pad/crop side — tractable full spectrum; many ARC tasks fit
	GridDim   = MaxCanvas * MaxCanvas
	NumColors = 10
)

type GridPair struct {
	Input  [][]int
	Output [][]int
}

type Task struct {
	ID    string
	Train []GridPair
	Test  []GridPair
}

type rawTask struct {
	Train []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"train"`
	Test []struct {
		Input  [][]int `json:"input"`
		Output [][]int `json:"output"`
	} `json:"test"`
}

// LoadTasks loads up to maxTasks JSON tasks from dir (stable sorted order).
func LoadTasks(dir string, maxTasks int) ([]*Task, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if maxTasks > 0 && len(names) > maxTasks {
		names = names[:maxTasks]
	}
	out := make([]*Task, 0, len(names))
	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		var raw rawTask
		if json.Unmarshal(data, &raw) != nil {
			continue
		}
		t := &Task{ID: name[:len(name)-len(filepath.Ext(name))]}
		for _, p := range raw.Train {
			t.Train = append(t.Train, GridPair{Input: p.Input, Output: p.Output})
		}
		for _, p := range raw.Test {
			t.Test = append(t.Test, GridPair{Input: p.Input, Output: p.Output})
		}
		if len(t.Train) == 0 || len(t.Test) == 0 {
			continue
		}
		out = append(out, t)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no ARC tasks in %s", dir)
	}
	return out, nil
}

// EncodeGrid pads/crops to MaxCanvas×MaxCanvas, colors scaled to [0,1].
func EncodeGrid(grid [][]int) []float32 {
	enc := make([]float32, GridDim)
	for r := 0; r < MaxCanvas; r++ {
		for c := 0; c < MaxCanvas; c++ {
			if r < len(grid) && c < len(grid[r]) {
				v := grid[r][c]
				if v < 0 {
					v = 0
				}
				if v > 9 {
					v = 9
				}
				enc[r*MaxCanvas+c] = float32(v) / 9.0
			}
		}
	}
	return enc
}

// DecodeGrid rounds network output back to colors; returns MaxCanvas square.
func DecodeGrid(out []float32) [][]int {
	g := make([][]int, MaxCanvas)
	for r := 0; r < MaxCanvas; r++ {
		g[r] = make([]int, MaxCanvas)
		for c := 0; c < MaxCanvas; c++ {
			idx := r*MaxCanvas + c
			v := 0.0
			if idx < len(out) {
				v = float64(out[idx]) * 9.0
			}
			if v < 0 {
				v = 0
			}
			if v > 9 {
				v = 9
			}
			g[r][c] = int(v + 0.5)
			if g[r][c] > 9 {
				g[r][c] = 9
			}
		}
	}
	return g
}

// PixelAccuracy compares pred vs target on the true target bounding box (clipped to canvas).
func PixelAccuracy(pred, target [][]int) float64 {
	th := len(target)
	if th == 0 {
		return 0
	}
	tw := len(target[0])
	if th > MaxCanvas {
		th = MaxCanvas
	}
	if tw > MaxCanvas {
		tw = MaxCanvas
	}
	ok, n := 0, 0
	for r := 0; r < th; r++ {
		rowW := tw
		if rowW > len(target[r]) {
			rowW = len(target[r])
		}
		for c := 0; c < rowW; c++ {
			n++
			pv := 0
			if r < len(pred) && c < len(pred[r]) {
				pv = pred[r][c]
			}
			if pv == target[r][c] {
				ok++
			}
		}
	}
	if n == 0 {
		return 0
	}
	return 100 * float64(ok) / float64(n)
}

// ExactMatch is true when every cell in the target bbox matches.
func ExactMatch(pred, target [][]int) bool {
	return PixelAccuracy(pred, target) >= 99.999
}

// FitsCanvas is true when the grid is fully representable on MaxCanvas (no crop).
func FitsCanvas(g [][]int) bool {
	if len(g) == 0 || len(g) > MaxCanvas {
		return false
	}
	for _, row := range g {
		if len(row) > MaxCanvas {
			return false
		}
	}
	return true
}

// FilterCanvasTasks keeps tasks whose train+test grids all fit MaxCanvas.
func FilterCanvasTasks(tasks []*Task) (kept []*Task, dropped int) {
	for _, t := range tasks {
		ok := true
		for _, p := range t.Train {
			if !FitsCanvas(p.Input) || !FitsCanvas(p.Output) {
				ok = false
				break
			}
		}
		if ok {
			for _, p := range t.Test {
				if !FitsCanvas(p.Input) || !FitsCanvas(p.Output) {
					ok = false
					break
				}
			}
		}
		if ok {
			kept = append(kept, t)
		} else {
			dropped++
		}
	}
	return kept, dropped
}

package arcspec

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

type layerFamily struct {
	name string
}

func SpectrumFamilies() []layerFamily {
	return []layerFamily{
		{name: "dense"},
		{name: "swiglu"},
		{name: "mha"},
		{name: "cnn1"},
		{name: "cnn2"},
		{name: "cnn3"},
		{name: "rnn"},
		{name: "lstm"},
		{name: "embedding"},
		{name: "residual"},
	}
}

type arcNetTask struct {
	Factory poly.NetworkFactory
	// TrainBatches from demonstration pairs across the task pool.
	TrainBatches []poly.TrainingBatch[float32]
	// EvalPairs keep original grids for scoring (same order as train demos + tests tracked separately).
	EvalTasks []*Task
	LossType  string
	InKind    string // "vector" | "cnn" | "tokens"
}

// BuildFactoryAndBatches builds a NeuralFountain factory + ARC demo batches for one family.
func BuildFactoryAndBatches(family string, dt poly.DType, tasks []*Task, k int) (arcNetTask, error) {
	dtName := dt.String()
	trainBatches := demoBatches(tasks)
	if len(trainBatches) < k*2 {
		// Repeat demos so every shard has samples.
		for len(trainBatches) < k*4 {
			trainBatches = append(trainBatches, trainBatches...)
		}
	}

	switch family {
	case "dense":
		sizes := []int{GridDim, 1024, 1024, GridDim}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.DenseTopologySeed("arc-nf-dense", sizes)
			topo ^= uint64(idx+1) * 0x9e3779b97f4a7c15
			dtypes := repeatDtype(dtName, len(sizes)-1)
			m, err := poly.BuildDenseManifest(topo, sizes, dtypes)
			if err != nil {
				return nil, err
			}
			net, err := poly.BuildDenseVolumetricFromManifest(m)
			if err != nil {
				return nil, err
			}
			last := net.GetLayer(0, 0, 0, len(m.Layers)-1)
			last.Activation = poly.ActivationLinear
			return net, nil
		}
		return arcNetTask{Factory: factory, TrainBatches: trainBatches, EvalTasks: tasks, LossType: "mse", InKind: "vector"}, nil

	case "swiglu":
		// Project via first-GridDim: use Hidden=GridDim through a SwiGLU that expects GridDim.
		// SwiGLUSpec Hidden is the width; Intermediate expands. Input must be Hidden.
		specs := []poly.SwiGLUSpec{{Hidden: GridDim, Intermediate: GridDim * 2}}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.SwiGLUTopologySeed("arc-nf-swiglu", specs)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildSwiGLUManifest(topo, specs, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildSwiGLUVolumetricFromManifest(m)
		}
		batches, err := remapBatchesToProbeIO(factory, trainBatches)
		return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "vector"}, err

	case "mha":
		// Compress input to DModel via truncate/pad of encoded grid head.
		const dModel = 32
		specs := []poly.MHASpec{{DModel: dModel, NumHeads: 4, NumKVHeads: 4, HeadDim: 8, QueryDim: dModel}}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.MHATopologySeed("arc-nf-mha", specs)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildMHAManifest(topo, specs, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildMHAVolumetricFromManifest(m)
		}
		comp := compressBatches(trainBatches, dModel)
		batches, err := remapBatchesToProbeIO(factory, comp)
		return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "vector"}, err

	case "cnn1":
		return cnnFamily(1, MaxCanvas, dtName, trainBatches, tasks)
	case "cnn2":
		return cnnFamily(2, MaxCanvas, dtName, trainBatches, tasks)
	case "cnn3":
		return cnnFamily(3, 4, dtName, trainBatches, tasks)

	case "rnn":
		sizes := []int{MaxCanvas, MaxCanvas}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.RNNTopologySeed("arc-nf-rnn", sizes)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildRNNManifest(topo, sizes, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildRNNVolumetricFromManifest(m)
		}
		rowBatches := rowBatches(trainBatches, MaxCanvas)
		batches, err := remapBatchesToProbeIO(factory, rowBatches)
		return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "vector"}, err

	case "lstm":
		sizes := []int{MaxCanvas, MaxCanvas}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.LSTMTopologySeed("arc-nf-lstm", sizes)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildLSTMManifest(topo, sizes, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildLSTMVolumetricFromManifest(m)
		}
		rowBatches := rowBatches(trainBatches, MaxCanvas)
		batches, err := remapBatchesToProbeIO(factory, rowBatches)
		return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "vector"}, err

	case "embedding":
		spec := poly.EmbeddingSpec{VocabSize: NumColors, EmbeddingDim: 8, SeqLen: MaxCanvas}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.EmbeddingTopologySeed("arc-nf-emb", []poly.EmbeddingSpec{spec})
			topo ^= uint64(idx + 1)
			m, err := poly.BuildEmbeddingManifest(topo, []poly.EmbeddingSpec{spec}, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildEmbeddingVolumetricFromManifest(m)
		}
		batches := tokenBatches(tasks, spec)
		return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "tokens"}, nil

	case "residual":
		spec := poly.ResidualSpec{In: GridDim, Out: GridDim}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.ResidualTopologySeed("arc-nf-res", spec)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildResidualManifest(topo, spec, dtName)
			if err != nil {
				return nil, err
			}
			return poly.BuildResidualVolumetricFromManifest(m)
		}
		batches, err := remapBatchesToProbeIO(factory, trainBatches)
		return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "vector"}, err
	}

	return arcNetTask{}, fmt.Errorf("unknown family %q", family)
}

func cnnFamily(dim, spatial int, dtName string, train []poly.TrainingBatch[float32], tasks []*Task) (arcNetTask, error) {
	spec := poly.CNNSpec{Dim: dim, InputChannels: 1, Filters: 4, Spatial: spatial, KernelSize: 3}
	factory := func(idx int) (*poly.VolumetricNetwork, error) {
		topo := poly.CNNTopologySeed(fmt.Sprintf("arc-nf-cnn%d", dim), []poly.CNNSpec{spec})
		topo ^= uint64(idx + 1)
		m, err := poly.BuildCNNManifest(topo, []poly.CNNSpec{spec}, []string{dtName})
		if err != nil {
			return nil, err
		}
		return poly.BuildCNNVolumetricFromManifest(m)
	}
	// Build CNN-shaped batches from encoded grids.
	base := poly.CNNDemoInput(spec)
	if base == nil {
		return arcNetTask{}, fmt.Errorf("cnn%d nil demo", dim)
	}
	net, err := factory(0)
	if err != nil {
		return arcNetTask{}, err
	}
	poly.WireNetworkLayers(net)
	net.ReleaseFP32MasterWhenIdle = false
	_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
	net.EnsureTrainingWeights()
	out, _, _ := poly.ForwardPolymorphic(net, base)
	if out == nil {
		return arcNetTask{}, fmt.Errorf("cnn%d probe nil", dim)
	}
	batches := make([]poly.TrainingBatch[float32], 0, len(train))
	for _, b := range train {
		in := reshapeToCNN(b.Input.Data, spec, base.Shape)
		tgt := make([]float32, len(out.Data))
		// Project target grid energy into CNN output space.
		src := b.Target.Data
		for j := range tgt {
			tgt[j] = src[j%len(src)] * 0.5
		}
		batches = append(batches, poly.TrainingBatch[float32]{
			Input:  in,
			Target: poly.NewTensorFromSlice(tgt, out.Shape...),
		})
	}
	return arcNetTask{Factory: factory, TrainBatches: batches, EvalTasks: tasks, LossType: "mse", InKind: "cnn"}, nil
}

func demoBatches(tasks []*Task) []poly.TrainingBatch[float32] {
	var batches []poly.TrainingBatch[float32]
	for _, t := range tasks {
		for _, p := range t.Train {
			in := EncodeGrid(p.Input)
			out := EncodeGrid(p.Output)
			batches = append(batches, poly.TrainingBatch[float32]{
				Input:  poly.NewTensorFromSlice(in, 1, GridDim),
				Target: poly.NewTensorFromSlice(out, 1, GridDim),
			})
		}
	}
	return batches
}

func compressBatches(src []poly.TrainingBatch[float32], dim int) []poly.TrainingBatch[float32] {
	out := make([]poly.TrainingBatch[float32], len(src))
	for i, b := range src {
		in := make([]float32, dim)
		tg := make([]float32, dim)
		copy(in, b.Input.Data)
		copy(tg, b.Target.Data)
		// Mix remaining energy into bins.
		for j := dim; j < len(b.Input.Data); j++ {
			in[j%dim] += b.Input.Data[j] * 0.1
		}
		for j := dim; j < len(b.Target.Data); j++ {
			tg[j%dim] += b.Target.Data[j] * 0.1
		}
		out[i] = poly.TrainingBatch[float32]{
			Input:  poly.NewTensorFromSlice(in, 1, dim),
			Target: poly.NewTensorFromSlice(tg, 1, dim),
		}
	}
	return out
}

func rowBatches(src []poly.TrainingBatch[float32], row int) []poly.TrainingBatch[float32] {
	out := make([]poly.TrainingBatch[float32], len(src))
	for i, b := range src {
		in := make([]float32, row)
		tg := make([]float32, row)
		for r := 0; r < row && r*MaxCanvas < len(b.Input.Data); r++ {
			in[r] = b.Input.Data[r*MaxCanvas]
			if r*MaxCanvas < len(b.Target.Data) {
				tg[r] = b.Target.Data[r*MaxCanvas]
			}
		}
		out[i] = poly.TrainingBatch[float32]{
			Input:  poly.NewTensorFromSlice(in, 1, row),
			Target: poly.NewTensorFromSlice(tg, 1, row),
		}
	}
	return out
}

func tokenBatches(tasks []*Task, spec poly.EmbeddingSpec) []poly.TrainingBatch[float32] {
	outDim := spec.SeqLen * spec.EmbeddingDim
	var batches []poly.TrainingBatch[float32]
	for _, t := range tasks {
		for _, p := range t.Train {
			tok := make([]float32, spec.SeqLen)
			flat := EncodeGrid(p.Input)
			for i := 0; i < spec.SeqLen; i++ {
				// Quantize color from pad cell.
				c := int(flat[i]*9 + 0.5)
				if c < 0 {
					c = 0
				}
				if c > 9 {
					c = 9
				}
				tok[i] = float32(c)
			}
			tgtFlat := EncodeGrid(p.Output)
			tgt := make([]float32, outDim)
			for j := range tgt {
				tgt[j] = tgtFlat[j%len(tgtFlat)]
			}
			batches = append(batches, poly.TrainingBatch[float32]{
				Input:  poly.NewTensorFromSlice(tok, 1, spec.SeqLen),
				Target: poly.NewTensorFromSlice(tgt, 1, outDim),
			})
		}
	}
	return batches
}

func remapBatchesToProbeIO(factory poly.NetworkFactory, src []poly.TrainingBatch[float32]) ([]poly.TrainingBatch[float32], error) {
	if len(src) == 0 {
		return nil, fmt.Errorf("empty batches")
	}
	net, err := factory(0)
	if err != nil {
		return nil, err
	}
	poly.WireNetworkLayers(net)
	net.ReleaseFP32MasterWhenIdle = false
	_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
	net.EnsureTrainingWeights()
	out, _, _ := poly.ForwardPolymorphic(net, src[0].Input)
	if out == nil || len(out.Data) == 0 {
		return nil, fmt.Errorf("probe forward nil")
	}
	batches := make([]poly.TrainingBatch[float32], len(src))
	for i, b := range src {
		tgt := make([]float32, len(out.Data))
		for j := range tgt {
			if j < len(b.Target.Data) {
				tgt[j] = b.Target.Data[j]
			} else if len(b.Target.Data) > 0 {
				tgt[j] = b.Target.Data[j%len(b.Target.Data)] * float32(math.Sin(float64(j)*0.07+1))
			}
		}
		batches[i] = poly.TrainingBatch[float32]{
			Input:  b.Input.Clone(),
			Target: poly.NewTensorFromSlice(tgt, out.Shape...),
		}
	}
	return batches, nil
}

func reshapeToCNN(flat []float32, spec poly.CNNSpec, shape []int) *poly.Tensor[float32] {
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := make([]float32, n)
	for i := range data {
		if i < len(flat) {
			data[i] = flat[i]
		}
	}
	return poly.NewTensorFromSlice(data, shape...)
}

func repeatDtype(name string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = name
	}
	return out
}

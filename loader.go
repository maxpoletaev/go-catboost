package catboost

import (
	"fmt"
	"math"

	"github.com/maxpoletaev/go-catboost/internal/cbm"
)

// deserializeModel loads a Model from FlatBuffers
// Ref: catboost/libs/model/model.cpp
func deserializeModel(fbTrees *cbm.TModelTrees) (*Model, error) {
	approxDimension := int(fbTrees.ApproxDimension())
	if approxDimension < 1 {
		approxDimension = 1
	}

	floatFeatures := deserializeFloatFeatures(fbTrees)
	catFeatures := deserializeCatFeatures(fbTrees)

	oneHotFeatures, err := deserializeOneHotFeatures(fbTrees, catFeatures)
	if err != nil {
		return nil, err
	}

	treeSizes, treeStartOffsets, repackedBins := deserializeTreeSplits(fbTrees)
	stepNodes, nodeIdToLeafId := deserializeStepNodes(fbTrees)
	leafValues := deserializeLeafValues(fbTrees)
	scale, bias := deserializeScaleBias(fbTrees, approxDimension)

	m := &Model{
		approxDimension:  approxDimension,
		floatFeatures:    floatFeatures,
		catFeatures:      catFeatures,
		oneHotFeatures:   oneHotFeatures,
		treeSizes:        treeSizes,
		treeStartOffsets: treeStartOffsets,
		repackedBins:     repackedBins,
		leafValues:       leafValues,
		stepNodes:        stepNodes,
		nodeIdToLeafId:   nodeIdToLeafId,
		scale:            scale,
		bias:             bias,
		modelInfo:        make(map[string]string),
	}

	m.updateRuntimeData()

	return m, nil
}

// updateRuntimeData computes fields derived from the raw deserialized data.
// Ref: catboost/libs/model/model.cpp (UpdateRuntimeData)
func (m *Model) updateRuntimeData() {
	m.isOblivious = len(m.stepNodes) == 0
	m.needXorMask = len(m.oneHotFeatures) > 0
	if m.isOblivious {
		m.treeFirstLeafOffset = calcObliviousFirstLeafOffsets(m.treeSizes, m.approxDimension)
	} else {
		m.treeFirstLeafOffset = calcNonSymmetricFirstLeafOffsets(
			m.treeSizes, m.treeStartOffsets, m.stepNodes, m.nodeIdToLeafId, m.approxDimension,
		)
	}
	m.effectiveBinCount = computeEffectiveBinCount(m.floatFeatures, m.oneHotFeatures)

	// Ref: catboost/libs/model/model.cpp (UpdateApplyData)
	// MinimalSufficientFloatFeaturesVectorSize = max(Position.Index) + 1
	for _, ff := range m.floatFeatures {
		if ff.index+1 > m.minFloatFeatures {
			m.minFloatFeatures = ff.index + 1
		}
	}
	for _, cf := range m.catFeatures {
		if cf.index+1 > m.minCatFeatures {
			m.minCatFeatures = cf.index + 1
		}
	}
}

func deserializeFloatFeatures(fbTrees *cbm.TModelTrees) []floatFeatureMeta {
	features := make([]floatFeatureMeta, fbTrees.FloatFeaturesLength())
	for i := range features {
		var ff cbm.TFloatFeature
		fbTrees.FloatFeatures(&ff, i)
		borders := make([]float32, ff.BordersLength())
		for j := range borders {
			borders[j] = ff.Borders(j)
		}
		features[i] = floatFeatureMeta{
			index:   int(ff.Index()),
			borders: borders,
			nanMode: nanValueTreatment(ff.NanValueTreatment()),
		}
	}
	return features
}

func deserializeCatFeatures(fbTrees *cbm.TModelTrees) []catFeatureMeta {
	var features []catFeatureMeta
	for i := 0; i < fbTrees.CatFeaturesLength(); i++ {
		var cf cbm.TCatFeature
		fbTrees.CatFeatures(&cf, i)
		if !cf.UsedInModel() {
			continue
		}
		features = append(features, catFeatureMeta{index: int(cf.Index())})
	}
	return features
}

func deserializeOneHotFeatures(fbTrees *cbm.TModelTrees, catFeatures []catFeatureMeta) ([]oneHotFeatureMeta, error) {
	features := make([]oneHotFeatureMeta, fbTrees.OneHotFeaturesLength())
	for i := range features {
		var ohe cbm.TOneHotFeature
		fbTrees.OneHotFeatures(&ohe, i)

		// Find the packed index of the referenced cat feature.
		catIdx := int(ohe.Index())
		packedIdx := -1
		for j, cf := range catFeatures {
			if cf.index == catIdx {
				packedIdx = j
				break
			}
		}
		if packedIdx < 0 {
			return nil, fmt.Errorf("one-hot feature references cat feature %d which is not used in model", catIdx)
		}

		values := make([]int32, ohe.ValuesLength())
		for j := range values {
			values[j] = ohe.Values(j)
		}
		features[i] = oneHotFeatureMeta{catFeatureIndex: packedIdx, values: values}
	}
	return features, nil
}

func deserializeTreeSplits(fbTrees *cbm.TModelTrees) ([]int, []int, []repackedBin) {
	nTrees := fbTrees.TreeSizesLength()
	treeSizes := make([]int, nTrees)
	treeStartOffsets := make([]int, nTrees)
	for i := range treeSizes {
		treeSizes[i] = int(fbTrees.TreeSizes(i))
		treeStartOffsets[i] = int(fbTrees.TreeStartOffsets(i))
	}

	bins := make([]repackedBin, fbTrees.RepackedBinsLength())
	var rb cbm.TRepackedBin
	for i := range bins {
		fbTrees.RepackedBins(&rb, i)
		bins[i] = repackedBin{
			featureIndex: rb.FeatureIndex(),
			xorMask:      rb.XorMask(),
			splitIdx:     rb.SplitIdx(),
		}
	}
	return treeSizes, treeStartOffsets, bins
}

func deserializeStepNodes(fbTrees *cbm.TModelTrees) ([]stepNode, []uint32) {
	nodes := make([]stepNode, fbTrees.NonSymmetricStepNodesLength())
	var sn cbm.TNonSymmetricTreeStepNode
	for i := range nodes {
		fbTrees.NonSymmetricStepNodes(&sn, i)
		nodes[i] = stepNode{
			leftDiff:  sn.LeftSubtreeDiff(),
			rightDiff: sn.RightSubtreeDiff(),
		}
	}

	nodeIdToLeafId := make([]uint32, fbTrees.NonSymmetricNodeIdToLeafIdLength())
	for i := range nodeIdToLeafId {
		nodeIdToLeafId[i] = fbTrees.NonSymmetricNodeIdToLeafId(i)
	}
	return nodes, nodeIdToLeafId
}

func deserializeLeafValues(fbTrees *cbm.TModelTrees) []float64 {
	leafValues := make([]float64, fbTrees.LeafValuesLength())
	for i := range leafValues {
		leafValues[i] = fbTrees.LeafValues(i)
	}
	return leafValues
}

func deserializeScaleBias(fbTrees *cbm.TModelTrees, approxDimension int) (float64, []float64) {
	scale := fbTrees.Scale()
	if scale == 0 {
		scale = 1.0 // default in schema
	}

	var bias []float64
	if fbTrees.MultiBiasLength() > 0 {
		bias = make([]float64, fbTrees.MultiBiasLength())
		for i := range bias {
			bias[i] = fbTrees.MultiBias(i)
		}
	} else {
		bias = make([]float64, approxDimension)
		b := fbTrees.Bias()
		for i := range bias {
			bias[i] = b
		}
	}
	return scale, bias
}

// Ref: catboost/libs/model/model.cpp (CalcFirstLeafOffsets)
func calcObliviousFirstLeafOffsets(treeSizes []int, approxDimension int) []int {
	offsets := make([]int, len(treeSizes))
	offset := 0
	for i, size := range treeSizes {
		offsets[i] = offset
		offset += (1 << size) * approxDimension
	}
	return offsets
}

// Ref: catboost/libs/model/model.cpp (CalcFirstLeafOffsets, non-symmetric branch)
func calcNonSymmetricFirstLeafOffsets(
	treeSizes []int,
	treeStartOffsets []int,
	stepNodes []stepNode,
	nodeIdToLeafId []uint32,
	approxDimension int,
) []int {
	offsets := make([]int, len(treeSizes))
	for treeId, size := range treeSizes {
		start := treeStartOffsets[treeId]
		end := start + size
		minIdx := ^uint32(0)
		for nodeIdx := start; nodeIdx < end; nodeIdx++ {
			sn := stepNodes[nodeIdx]
			if sn.leftDiff == 0 || sn.rightDiff == 0 {
				leafIdx := nodeIdToLeafId[nodeIdx]
				if leafIdx < minIdx {
					minIdx = leafIdx
				}
			}
		}
		offsets[treeId] = int(minIdx)
	}
	return offsets
}

// Ref: catboost/libs/model/model.cpp (CalcBinFeatures)
func computeEffectiveBinCount(floatFeatures []floatFeatureMeta, oneHotFeatures []oneHotFeatureMeta) int {
	count := 0
	for i := range floatFeatures {
		count += (len(floatFeatures[i].borders) + maxValuesPerBin - 1) / maxValuesPerBin
	}
	for i := range oneHotFeatures {
		count += (len(oneHotFeatures[i].values) + maxValuesPerBin - 1) / maxValuesPerBin
	}
	return count
}

// Ref: catboost/libs/model/cpu/quantization.h (BinarizeFeatures, NaN branch)
func nanSubstitution(mode nanValueTreatment) (float32, bool) {
	switch mode {
	case nanAsFalse:
		return float32(math.Inf(-1)), true
	case nanAsTrue:
		return float32(math.Inf(1)), true
	default:
		return 0, false
	}
}

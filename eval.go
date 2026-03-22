package catboost

import (
	"math"
)

func (m *Model) evalSplit(s *treeSplit, floatVals []float32, catHashes []int32) bool {
	switch s.kind {
	case splitKindOneHot:
		return catHashes[s.featureIndex] == s.hashValue
	case splitKindCTR:
		return m.computeCTR(s.ctrIdx, floatVals, catHashes) > float64(s.border)
	default: // splitKindFloat
		val := floatVals[s.featureIndex]
		if math.IsNaN(float64(val)) {
			if subst, ok := nanSubstitution(s.nanMode); ok {
				val = subst
			}
		}
		return val > s.border
	}
}

func (m *Model) evalBatch(floatFeatures [][]float32, catHashes [][]int32) [][]float64 {
	n := len(floatFeatures)
	dim := m.approxDimension
	flat := make([]float64, n*dim)
	results := make([][]float64, n)
	for i := range results {
		results[i] = flat[i*dim : (i+1)*dim]
	}

	if m.isOblivious {
		m.evalObliviousBatch(floatFeatures, catHashes, flat)
	} else {
		m.evalNonSymmetricBatch(floatFeatures, catHashes, results)
	}

	for i := range results {
		m.applyScaleBias(results[i])
		m.applyPredictionType(results[i])
	}

	return results
}

func (m *Model) evalObliviousSingle(floatVals []float32, cats []int32, result []float64) {
	for treeIdx := range m.treeSizes {
		depth := m.treeSizes[treeIdx]
		start := m.treeStartOffsets[treeIdx]
		base := m.treeFirstLeafOffset[treeIdx]
		splits := m.splits[start : start+depth]

		leafIdx := 0
		for d := range splits {
			s := &splits[d]
			var taken bool
			switch s.kind {
			case splitKindFloat:
				val := floatVals[s.featureIndex]
				if val != val { // NaN check without float64 conversion
					if s.nanMode == nanAsTrue {
						taken = true
					}
				} else {
					taken = val > s.border
				}
			case splitKindOneHot:
				taken = cats[s.featureIndex] == s.hashValue
			case splitKindCTR:
				taken = m.computeCTR(s.ctrIdx, floatVals, cats) > float64(s.border)
			}
			if taken {
				leafIdx |= 1 << d
			}
		}

		leafStart := base + leafIdx*m.approxDimension
		for dim := range result {
			result[dim] += m.leafValues[leafStart+dim]
		}
	}

}

func transpose[T any](rows [][]T, nCols, nRows int) [][]T {
	if nCols == 0 || rows == nil {
		return nil
	}
	buf := make([]T, nCols*nRows)
	cols := make([][]T, nCols)
	for f := range cols {
		col := buf[f*nRows : (f+1)*nRows]
		cols[f] = col
		for i, row := range rows {
			col[i] = row[f]
		}
	}
	return cols
}

func (m *Model) evalObliviousBatch(floatFeatures [][]float32, catHashes [][]int32, flat []float64) {
	n := len(floatFeatures)

	// Single-document fast path: skip transposition overhead.
	if n == 1 {
		var cats []int32
		if catHashes != nil {
			cats = catHashes[0]
		}
		m.evalObliviousSingle(floatFeatures[0], cats, flat[:m.approxDimension])
		return
	}

	catCols := transpose(catHashes, m.catFeatureCount, n)
	leafIndices := make([]uint32, n)
	leafValues := m.leafValues

	evalTreesScalar(m, floatFeatures, catHashes, catCols, leafIndices, leafValues, flat, n)
}

func (m *Model) evalNonSymmetricBatch(floatFeatures [][]float32, catHashes [][]int32, results [][]float64) {
	n := len(floatFeatures)
	for treeIdx := range m.treeSizes {
		treeStart := m.treeStartOffsets[treeIdx]
		for i := 0; i < n; i++ {
			var cats []int32
			if catHashes != nil {
				cats = catHashes[i]
			}

			nodeIdx := treeStart
			for {
				var diff uint16
				sn := m.stepNodes[nodeIdx]
				if m.evalSplit(&m.splits[nodeIdx], floatFeatures[i], cats) {
					diff = sn.rightDiff
				} else {
					diff = sn.leftDiff
				}
				if diff == 0 {
					break
				}
				nodeIdx += int(diff)
			}

			leafValueIdx := int(m.nodeIdToLeafId[nodeIdx])
			for dim := range results[i] {
				results[i][dim] += m.leafValues[leafValueIdx+dim]
			}
		}
	}

}

func (m *Model) applyScaleBias(result []float64) {
	for d := range result {
		result[d] = m.scale*result[d] + m.bias[d]
	}
}

func (m *Model) applyPredictionType(result []float64) {
	if m.approxDimension == 1 {
		switch m.predictionType {
		case PredictionTypeRawFormulaVal:
			// no transformation
		case PredictionTypeProbability:
			sigmoid(result)
		case PredictionTypeClass:
			binclass(result, m.binclassRawValBorder)
		case PredictionTypeExponent:
			exponent(result)
		case PredictionTypeLogProbability:
			logSigmoid(result)
		}
	} else {
		switch m.predictionType {
		case PredictionTypeRawFormulaVal:
			// no transformation
		case PredictionTypeProbability:
			softmax(result)
		case PredictionTypeClass:
			argmax(result)
		case PredictionTypeExponent:
			exponent(result)
		case PredictionTypeLogProbability:
			logSoftmax(result)
		}
	}
}

func sigmoid(result []float64) {
	result[0] = 1.0 / (1.0 + math.Exp(-result[0]))
}

func softmax(result []float64) {
	max := result[0]
	for _, v := range result[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for i, v := range result {
		result[i] = math.Exp(v - max)
		sum += result[i]
	}
	for i := range result {
		result[i] /= sum
	}
}

func logSigmoid(result []float64) {
	result[0] = -math.Log(1.0 + math.Exp(-result[0]))
}

func logSoftmax(result []float64) {
	max := result[0]
	for _, v := range result[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for _, v := range result {
		sum += math.Exp(v - max)
	}
	logSum := math.Log(sum)
	for i, v := range result {
		result[i] = v - max - logSum
	}
}

func exponent(result []float64) {
	for i, v := range result {
		result[i] = math.Exp(v)
	}
}

func binclass(result []float64, border float64) {
	if result[0] > border {
		result[0] = 1
	} else {
		result[0] = 0
	}
}

func argmax(result []float64) {
	best := 0
	for i := range result[1:] {
		if result[i+1] > result[best] {
			best = i + 1
		}
	}
	for i := range result {
		result[i] = 0
	}
	result[0] = float64(best)
}

func evalTreesScalar(m *Model, floatFeatures [][]float32, catHashes [][]int32, catCols [][]int32, leafIndices []uint32, leafValues []float64, flat []float64, n int) {
	dim := m.approxDimension
	featCols := transpose(floatFeatures, m.floatFeatureCount, n)

	for treeIdx := range m.treeSizes {
		clear(leafIndices[:n])

		depth := m.treeSizes[treeIdx]
		start := m.treeStartOffsets[treeIdx]
		base := m.treeFirstLeafOffset[treeIdx]
		splits := m.splits[start : start+depth]

		for d := range splits {
			split := &splits[d]
			bit := uint32(1) << d

			switch split.kind {
			case splitKindFloat:
				applySplitFloat(
					featCols[split.featureIndex], split.border,
					split.nanMode == nanAsTrue, leafIndices, bit, n,
				)

			case splitKindOneHot:
				applySplitOneHot(
					catCols[split.featureIndex], split.hashValue,
					leafIndices, bit, n,
				)

			default:
				for i := 0; i < n; i++ {
					var cats []int32
					if catHashes != nil {
						cats = catHashes[i]
					}
					if m.evalSplit(split, floatFeatures[i], cats) {
						leafIndices[i] |= bit
					}
				}
			}
		}

		accumulateLeaves(leafIndices[:n], leafValues, flat, base, dim, n)
	}
}

func btou(b bool) (u uint32) {
	if b {
		u = 1
	} else {
		u = 0
	}
	return u
}

func applySplitFloat(col []float32, border float32, nanTrue bool, leafIndices []uint32, bit uint32, n int) {
	// Branchless split: -btou(cond) is 0x00000000 or 0xffffffff,
	// so "bit & -btou(cond)" is bit or 0 without a branch.
	// Relies on the compiler inlining btou as CSET/SETcc.
	if nanTrue {
		i := 0
		for ; i <= n-4; i += 4 {
			v0, v1, v2, v3 := col[i], col[i+1], col[i+2], col[i+3]
			leafIndices[i] |= bit & -btou(v0 > border || v0 != v0)
			leafIndices[i+1] |= bit & -btou(v1 > border || v1 != v1)
			leafIndices[i+2] |= bit & -btou(v2 > border || v2 != v2)
			leafIndices[i+3] |= bit & -btou(v3 > border || v3 != v3)
		}
		for ; i < n; i++ {
			v := col[i]
			leafIndices[i] |= bit & -btou(v > border || v != v)
		}
	} else {
		i := 0
		for ; i <= n-4; i += 4 {
			v0, v1, v2, v3 := col[i], col[i+1], col[i+2], col[i+3]
			leafIndices[i] |= bit & -btou(v0 > border)
			leafIndices[i+1] |= bit & -btou(v1 > border)
			leafIndices[i+2] |= bit & -btou(v2 > border)
			leafIndices[i+3] |= bit & -btou(v3 > border)
		}
		for ; i < n; i++ {
			leafIndices[i] |= bit & -btou(col[i] > border)
		}
	}
}

func applySplitOneHot(col []int32, hashVal int32, leafIndices []uint32, bit uint32, n int) {
	i := 0
	for ; i <= n-4; i += 4 {
		leafIndices[i] |= bit & -btou(col[i] == hashVal)
		leafIndices[i+1] |= bit & -btou(col[i+1] == hashVal)
		leafIndices[i+2] |= bit & -btou(col[i+2] == hashVal)
		leafIndices[i+3] |= bit & -btou(col[i+3] == hashVal)
	}
	for ; i < n; i++ {
		leafIndices[i] |= bit & -btou(col[i] == hashVal)
	}
}

func accumulateLeaves(leafIndices []uint32, leafValues []float64, flat []float64, base, dim, n int) {
	for i := 0; i < n; i++ {
		leafStart := base + int(leafIndices[i])*dim
		off := i * dim

		for d := 0; d < dim; d++ {
			flat[off+d] += leafValues[leafStart+d]
		}
	}
}

package catboost

import "math"

func (m *Model) eval(bins []uint8) []float64 {
	var result []float64
	if m.isOblivious {
		result = m.evalOblivious(bins)
	} else {
		result = m.evalNonSymmetric(bins)
	}
	m.applyPredictionType(result)
	return result
}

func (m *Model) applyPredictionType(result []float64) {
	switch m.predictionType {
	case Probability:
		if m.approxDimension == 1 {
			sigmoid(result)
		} else {
			softmax(result)
		}
	case Class:
		if m.approxDimension == 1 {
			binclass(result)
		} else {
			argmax(result)
		}
	case Exponent:
		exponent(result)
	case LogProbability:
		if m.approxDimension == 1 {
			logSigmoid(result)
		} else {
			logSoftmax(result)
		}
	}
}

// Ref: catboost/libs/model/cpu/evaluator_impl.cpp (CalcTreesSingleDocImpl)
func (m *Model) evalOblivious(bins []uint8) []float64 {
	result := make([]float64, m.approxDimension)

	for treeIdx := range m.treeSizes {
		depth := m.treeSizes[treeIdx]
		startOffset := m.treeStartOffsets[treeIdx]
		splits := m.repackedBins[startOffset : startOffset+depth]

		leafIdx := 0
		if m.needXorMask {
			for d, s := range splits {
				if bins[s.featureIndex]^s.xorMask >= s.splitIdx {
					leafIdx |= 1 << d
				}
			}
		} else {
			for d, s := range splits {
				if bins[s.featureIndex] >= s.splitIdx {
					leafIdx |= 1 << d
				}
			}
		}

		base := m.treeFirstLeafOffset[treeIdx]
		if m.approxDimension == 1 {
			result[0] += m.leafValues[base+leafIdx]
		} else {
			leafStart := base + leafIdx*m.approxDimension
			for dim := 0; dim < m.approxDimension; dim++ {
				result[dim] += m.leafValues[leafStart+dim]
			}
		}
	}

	m.applyScaleBias(result)
	return result
}

// Ref: catboost/libs/model/cpu/evaluator_impl.cpp (CalcNonSymmetricTreesSingle)
func (m *Model) evalNonSymmetric(bins []uint8) []float64 {
	result := make([]float64, m.approxDimension)

	skipWork := len(bins) == 0
	for treeIdx := range m.treeSizes {
		nodeIdx := m.treeStartOffsets[treeIdx]
		for !skipWork {
			sn := m.stepNodes[nodeIdx]
			s := m.repackedBins[nodeIdx]
			val := bins[s.featureIndex] ^ s.xorMask
			var diff uint16
			if val >= s.splitIdx {
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
		if m.approxDimension == 1 {
			result[0] += m.leafValues[leafValueIdx]
		} else {
			for dim := 0; dim < m.approxDimension; dim++ {
				result[dim] += m.leafValues[leafValueIdx+dim]
			}
		}
	}

	m.applyScaleBias(result)
	return result
}

func (m *Model) applyScaleBias(result []float64) {
	for dim := range result {
		result[dim] = m.scale*result[dim] + m.bias[dim]
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

func binclass(result []float64) {
	if result[0] > 0 {
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
	result[best] = 1
}

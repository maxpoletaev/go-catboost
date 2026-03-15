package catboost

import "math"

// Ref: catboost/libs/model/cpu/quantization.h (BinarizeFeatures)
func (m *Model) quantize(floatVals []float32, catHashes []int32) []uint8 {
	bins := make([]uint8, m.effectiveBinCount)
	pos := 0

	for i := range m.floatFeatures {
		ff := &m.floatFeatures[i]
		val := float32(math.NaN())
		if ff.index < len(floatVals) {
			val = floatVals[ff.index]
		}
		if math.IsNaN(float64(val)) {
			if subst, ok := nanSubstitution(ff.nanMode); ok {
				val = subst
			}
		}
		pos = binarizeFloats(val, ff.borders, bins, pos)
	}

	for i := range m.oneHotFeatures {
		ohe := &m.oneHotFeatures[i]
		var hash uint32
		flatIdx := m.catFeatures[ohe.catFeatureIndex].index
		if flatIdx < len(catHashes) {
			hash = uint32(catHashes[flatIdx])
		}
		pos = oneHotBins(hash, ohe.values, bins, pos)
	}

	return bins
}

// Ref: catboost/libs/model/cpu/quantization.h (BinarizeFloatsNonSse)
func binarizeFloats(val float32, borders []float32, bins []uint8, pos int) int {
	for blockStart := 0; blockStart < len(borders); blockStart += maxValuesPerBin {
		blockEnd := blockStart + maxValuesPerBin
		if blockEnd > len(borders) {
			blockEnd = len(borders)
		}
		var count uint8
		for j := blockStart; j < blockEnd; j++ {
			if val > borders[j] {
				count++
			}
		}
		bins[pos] = count
		pos++
	}
	return pos
}

// Ref: catboost/libs/model/cpu/quantization.h (OneHotBinsFromTransposedCatFeatures)
func oneHotBins(hash uint32, values []int32, bins []uint8, pos int) int {
	for blockStart := 0; blockStart < len(values); blockStart += maxValuesPerBin {
		blockEnd := blockStart + maxValuesPerBin
		if blockEnd > len(values) {
			blockEnd = len(values)
		}
		var bin uint8
		for j := blockStart; j < blockEnd; j++ {
			if int32(hash) == values[j] {
				bin = uint8(j-blockStart) + 1
				break
			}
		}
		bins[pos] = bin
		pos++
	}
	return pos
}

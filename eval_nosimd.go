//go:build !(goexperiment.simd && amd64)

package catboost

func applySplitFloat(col []float32, border float32, nanTrue bool, leafIndices []uint32, bit uint32, n int) {
	applySplitFloatScalar(col, border, nanTrue, leafIndices, bit, n)
}

func applySplitOneHot(col []int32, hashVal int32, leafIndices []uint32, bit uint32, n int) {
	applySplitOneHotScalar(col, hashVal, leafIndices, bit, n)
}

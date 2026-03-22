//go:build goexperiment.simd && amd64

package catboost

import "simd/archsimd"

var hasAVX2 = archsimd.X86.AVX2()

func applySplitFloat(col []float32, border float32, nanTrue bool, leafIndices []uint32, bit uint32, n int) {
	if !hasAVX2 || n < 8 {
		applySplitFloatScalar(col, border, nanTrue, leafIndices, bit, n)
		return
	}

	borderVec := archsimd.BroadcastFloat32x8(border)
	bitVec := archsimd.BroadcastUint32x8(bit)

	i := 0
	if nanTrue {
		for ; i <= n-8; i += 8 {
			vals := archsimd.LoadFloat32x8Slice(col[i:])
			cmp := vals.Greater(borderVec)
			nan := vals.IsNaN()
			mask := cmp.Or(nan).ToInt32x8().AsUint32x8()
			idx := archsimd.LoadUint32x8Slice(leafIndices[i:])
			idx.Or(mask.And(bitVec)).StoreSlice(leafIndices[i:])
		}
	} else {
		for ; i <= n-8; i += 8 {
			vals := archsimd.LoadFloat32x8Slice(col[i:])
			mask := vals.Greater(borderVec).ToInt32x8().AsUint32x8()
			idx := archsimd.LoadUint32x8Slice(leafIndices[i:])
			idx.Or(mask.And(bitVec)).StoreSlice(leafIndices[i:])
		}
	}

	applySplitFloatScalar(col[i:], border, nanTrue, leafIndices[i:], bit, n-i)
}

func applySplitOneHot(col []int32, hashVal int32, leafIndices []uint32, bit uint32, n int) {
	if !hasAVX2 || n < 8 {
		applySplitOneHotScalar(col, hashVal, leafIndices, bit, n)
		return
	}

	hashVec := archsimd.BroadcastInt32x8(hashVal)
	bitVec := archsimd.BroadcastUint32x8(bit)

	i := 0
	for ; i <= n-8; i += 8 {
		cats := archsimd.LoadInt32x8Slice(col[i:])
		mask := cats.Equal(hashVec).ToInt32x8().AsUint32x8()
		idx := archsimd.LoadUint32x8Slice(leafIndices[i:])
		idx.Or(mask.And(bitVec)).StoreSlice(leafIndices[i:])
	}

	applySplitOneHotScalar(col[i:], hashVal, leafIndices[i:], bit, n-i)
}

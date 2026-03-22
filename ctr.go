package catboost

type ctrValueTable struct {
	lookup       map[uint64]int // hash -> row index in data
	data         []int32        // counts, (stride) ints per row
	stride       int            // values per bucket: 1 for Counter/FeatureFreq, 2+ for Borders/Buckets
	counterDenom int32          // denominator for Counter/FeatureFreq type
}

type ctrElemType int

const (
	ctrElemCat      ctrElemType = iota // cat_feature_value: fold catHashes[catIdx]
	ctrElemFloat                       // float_feature: fold (floatVals[floatIdx] > border) ? 1 : 0
	ctrElemExactCat                    // cat_feature_exact_value: fold (catHashes[catIdx] == hashValue) ? 1 : 0
)

type ctrProjElem struct {
	elemType  ctrElemType
	catIdx    int     // ctrElemCat, ctrElemExactCat
	floatIdx  int     // ctrElemFloat
	border    float32 // ctrElemFloat
	hashValue int32   // ctrElemExactCat
}

type ctrType int8

const (
	ctrTypeBorders ctrType = iota
	ctrTypeBuckets
	ctrTypeCounter
	ctrTypeFeqFreq
)

type ctrFeature struct {
	projection      []ctrProjElem
	ctrType         ctrType
	targetBorderIdx int
	priorNum        float64
	priorDenom      float64
	scale           float64
	shift           float64
	table           *ctrValueTable
}

const ctrHashMult = uint64(0x4906ba494954cb65)

func ctrCombineHash(current uint64, v int32) uint64 {
	return ctrHashMult * (current + ctrHashMult*uint64(int64(v)))
}

func (m *Model) computeCTR(ctrIdx int, floatVals []float32, catHashes []int32) float64 {
	ctr := &m.ctrFeatures[ctrIdx]

	var h uint64
	for _, elem := range ctr.projection {
		switch elem.elemType {
		case ctrElemCat:
			h = ctrCombineHash(h, catHashes[elem.catIdx])
		case ctrElemFloat:
			var bit int32
			if floatVals[elem.floatIdx] > elem.border {
				bit = 1
			}
			h = ctrCombineHash(h, bit)
		case ctrElemExactCat:
			var bit int32
			if catHashes[elem.catIdx] == elem.hashValue {
				bit = 1
			}
			h = ctrCombineHash(h, bit)
		}
	}

	var goodCount, totalCount float64

	table := ctr.table
	if idx, ok := table.lookup[h]; ok {
		switch ctr.ctrType {
		case ctrTypeCounter, ctrTypeFeqFreq:
			goodCount = float64(table.data[idx])
			totalCount = float64(table.counterDenom)

		case ctrTypeBuckets:
			base := idx * table.stride
			goodCount = float64(table.data[base+ctr.targetBorderIdx])
			for j := 0; j < table.stride; j++ {
				totalCount += float64(table.data[base+j])
			}

		default: // ctrTypeBorders and other target-mean types
			base := idx * table.stride
			if table.stride == 2 {
				goodCount = float64(table.data[base+1])
				totalCount = float64(table.data[base+0]) + float64(table.data[base+1])
			} else {
				for j := 0; j <= ctr.targetBorderIdx; j++ {
					totalCount += float64(table.data[base+j])
				}
				for j := ctr.targetBorderIdx + 1; j < table.stride; j++ {
					goodCount += float64(table.data[base+j])
				}
				totalCount += goodCount
			}
		}
	}

	ctrValue := (goodCount + ctr.priorNum) / (totalCount + ctr.priorDenom)
	return (ctrValue + ctr.shift) * ctr.scale
}

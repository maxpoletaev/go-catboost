package catboost

import (
	"encoding/json"
	"fmt"
	"strconv"
)

type jsonModelFile struct {
	FeaturesInfo   jsonFeaturesInfo         `json:"features_info"`
	ObliviousTrees []jsonOblTree            `json:"oblivious_trees"`
	Trees          []json.RawMessage        `json:"trees"`
	ScaleAndBias   jsonScaleBias            `json:"scale_and_bias"`
	CtrData        map[string]*jsonCtrTable `json:"ctr_data"`
}

type jsonFeaturesInfo struct {
	FloatFeatures []jsonFloatFeature `json:"float_features"`
	CatFeatures   []jsonCatFeature   `json:"categorical_features"`
	CtrFeatures   []jsonCtrFeature   `json:"ctrs"`
}

type jsonCtrFeature struct {
	Identifier      string            `json:"identifier"`
	Elements        []jsonCombElement `json:"elements"`
	CtrType         string            `json:"ctr_type"`
	PriorNum        float64           `json:"prior_numerator"`
	PriorDenom      float64           `json:"prior_denomerator"` // typo in CatBoost source
	Shift           float64           `json:"shift"`
	Scale           float64           `json:"scale"`
	TargetBorderIdx int               `json:"target_border_idx"`
	Borders         []float32         `json:"borders"`
}

type jsonCombElement struct {
	CombElement     string  `json:"combination_element"`
	CatFeatureIndex int     `json:"cat_feature_index"`   // cat_feature_value, cat_feature_exact_value
	FloatFeatIndex  int     `json:"float_feature_index"` // float_feature
	Border          float32 `json:"border"`              // float_feature
	Value           int32   `json:"value"`               // cat_feature_exact_value
}

type jsonCtrTable struct {
	HashMap            []json.RawMessage `json:"hash_map"`
	HashStride         int               `json:"hash_stride"`
	CounterDenominator int32             `json:"counter_denominator"`
}

type jsonFloatFeature struct {
	FeatureIndex int       `json:"feature_index"`
	Borders      []float32 `json:"borders"`
	NanTreatment string    `json:"nan_value_treatment"`
}

type jsonCatFeature struct {
	FeatureIndex int     `json:"feature_index"`
	Values       []int32 `json:"values"` // non-empty for one-hot features
}

type jsonOblTree struct {
	Splits     []jsonSplit `json:"splits"`
	LeafValues []float64   `json:"leaf_values"`
}

type jsonSplit struct {
	SplitType    string  `json:"split_type"`
	FloatFeatIdx int     `json:"float_feature_index"`
	Border       float32 `json:"border"`
	CatFeatIdx   int     `json:"cat_feature_index"`
	Value        int32   `json:"value"`
	SplitIndex   int     `json:"split_index"`
}

type jsonScaleBias struct {
	Scale float64
	Bias  []float64
}

func (sb *jsonScaleBias) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if len(raw) != 2 {
		return fmt.Errorf("scale_and_bias: expected 2 elements, got %d", len(raw))
	}
	if err := json.Unmarshal(raw[0], &sb.Scale); err != nil {
		return fmt.Errorf("scale_and_bias: scale: %w", err)
	}
	return json.Unmarshal(raw[1], &sb.Bias)
}

func buildModel(jm *jsonModelFile) (*Model, error) {
	approxDimension := len(jm.ScaleAndBias.Bias)
	if approxDimension < 1 {
		approxDimension = 1
	}

	// Build CTR features and a split-index-to-CTR-index mapping.
	ctrFeats, splitIdxToCTR, err := buildCtrFeatures(jm)
	if err != nil {
		return nil, err
	}

	var (
		treeSizes        []int
		treeStartOffsets []int
		splits           []treeSplit
		leafValues       []float64
		stepNodes        []stepNode
		nodeIdToLeafId   []uint32
	)

	if len(jm.ObliviousTrees) > 0 {
		for _, tree := range jm.ObliviousTrees {
			treeStartOffsets = append(treeStartOffsets, len(splits))
			treeSizes = append(treeSizes, len(tree.Splits))

			for _, s := range tree.Splits {
				sp, err := convertSplit(s, jm.FeaturesInfo, splitIdxToCTR)
				if err != nil {
					return nil, err
				}
				splits = append(splits, sp)
			}

			leafValues = append(leafValues, tree.LeafValues...)
		}
	} else {
		b := &treeBuilder{featuresInfo: jm.FeaturesInfo, splitIdxToCtr: splitIdxToCTR}
		for _, rawTree := range jm.Trees {
			var root jsonTreeNode
			if err := json.Unmarshal(rawTree, &root); err != nil {
				return nil, fmt.Errorf("parsing tree: %w", err)
			}

			treeStartOffsets = append(treeStartOffsets, len(b.splits))
			size, err := b.flatten(&root)
			if err != nil {
				return nil, err
			}

			treeSizes = append(treeSizes, size)
		}

		splits = b.splits
		stepNodes = b.stepNodes
		nodeIdToLeafId = b.nodeIdToLeafId
		leafValues = b.leafValues
	}

	scale := jm.ScaleAndBias.Scale
	if scale == 0 {
		scale = 1.0
	}
	bias := jm.ScaleAndBias.Bias
	if len(bias) == 0 {
		bias = make([]float64, approxDimension)
	}

	// Compute minimum required feature slice lengths from the feature descriptors.
	minFloat, minCat := 0, 0
	for _, ff := range jm.FeaturesInfo.FloatFeatures {
		if ff.FeatureIndex+1 > minFloat {
			minFloat = ff.FeatureIndex + 1
		}
	}
	for _, cf := range jm.FeaturesInfo.CatFeatures {
		if cf.FeatureIndex+1 > minCat {
			minCat = cf.FeatureIndex + 1
		}
	}

	isOblivious := len(stepNodes) == 0
	var treeFirstLeafOffset []int
	if isOblivious {
		treeFirstLeafOffset = calcObliviousLeafOffsets(treeSizes, approxDimension)
	}

	m := &Model{
		approxDimension:     approxDimension,
		treeSizes:           treeSizes,
		treeStartOffsets:    treeStartOffsets,
		splits:              splits,
		leafValues:          leafValues,
		stepNodes:           stepNodes,
		nodeIdToLeafId:      nodeIdToLeafId,
		scale:               scale,
		bias:                bias,
		floatFeatureCount:   minFloat,
		catFeatureCount:     minCat,
		modelInfo:           make(map[string]string),
		ctrFeatures:         ctrFeats,
		isOblivious:         isOblivious,
		treeFirstLeafOffset: treeFirstLeafOffset,
	}

	return m, nil
}

func calcObliviousLeafOffsets(treeSizes []int, approxDimension int) []int {
	offsets := make([]int, len(treeSizes))
	offset := 0
	for i, size := range treeSizes {
		offsets[i] = offset
		offset += (1 << size) * approxDimension
	}
	return offsets
}

type jsonTreeNode struct {
	Split  *jsonSplit      `json:"split"`
	Left   *jsonTreeNode   `json:"left"`
	Right  *jsonTreeNode   `json:"right"`
	Value  json.RawMessage `json:"value"`
	Weight float64         `json:"weight"`
}

type treeBuilder struct {
	splits         []treeSplit
	stepNodes      []stepNode
	nodeIdToLeafId []uint32
	leafValues     []float64
	featuresInfo   jsonFeaturesInfo
	splitIdxToCtr  map[int]int
}

func (b *treeBuilder) flatten(node *jsonTreeNode) (int, error) {
	myIdx := len(b.splits)

	if node.Split == nil {
		// Leaf node
		b.splits = append(b.splits, treeSplit{})
		b.stepNodes = append(b.stepNodes, stepNode{0, 0})
		b.nodeIdToLeafId = append(b.nodeIdToLeafId, uint32(len(b.leafValues)))

		var vals []float64
		if err := json.Unmarshal(node.Value, &vals); err != nil {
			// scalar (regression / binary)
			var v float64
			if err2 := json.Unmarshal(node.Value, &v); err2 != nil {
				return 0, fmt.Errorf("parsing leaf value: %w", err2)
			}
			vals = []float64{v}
		}
		b.leafValues = append(b.leafValues, vals...)
		return 1, nil
	}

	sp, err := convertSplit(*node.Split, b.featuresInfo, b.splitIdxToCtr)
	if err != nil {
		return 0, err
	}

	b.splits = append(b.splits, sp)
	b.stepNodes = append(b.stepNodes, stepNode{}) // filled after recursion
	b.nodeIdToLeafId = append(b.nodeIdToLeafId, 0)

	leftSize, err := b.flatten(node.Left)
	if err != nil {
		return 0, err
	}
	rightSize, err := b.flatten(node.Right)
	if err != nil {
		return 0, err
	}

	b.stepNodes[myIdx] = stepNode{leftDiff: 1, rightDiff: uint16(1 + leftSize)}
	return 1 + leftSize + rightSize, nil
}

func convertSplit(s jsonSplit, fi jsonFeaturesInfo, splitIdxToCtr map[int]int) (treeSplit, error) {
	switch s.SplitType {
	case "FloatFeature":
		idx := s.FloatFeatIdx
		if idx < 0 || idx >= len(fi.FloatFeatures) {
			return treeSplit{}, fmt.Errorf("float_feature_index %d out of range", idx)
		}
		ff := fi.FloatFeatures[idx]
		return treeSplit{
			kind:         splitKindFloat,
			featureIndex: ff.FeatureIndex,
			border:       s.Border,
			nanMode:      parseNanTreatment(ff.NanTreatment),
		}, nil

	case "OneHotFeature":
		idx := s.CatFeatIdx
		if idx < 0 || idx >= len(fi.CatFeatures) {
			return treeSplit{}, fmt.Errorf("cat_feature_index %d out of range", idx)
		}
		return treeSplit{
			kind:         splitKindOneHot,
			featureIndex: fi.CatFeatures[idx].FeatureIndex,
			hashValue:    s.Value,
		}, nil

	case "OnlineCtr":
		ctrIdx, ok := splitIdxToCtr[s.SplitIndex]
		if !ok {
			return treeSplit{}, fmt.Errorf("OnlineCtr split_index %d not in CTR range", s.SplitIndex)
		}
		return treeSplit{
			kind:   splitKindCTR,
			ctrIdx: ctrIdx,
			border: s.Border,
		}, nil

	default:
		return treeSplit{}, fmt.Errorf("unsupported split type %q", s.SplitType)
	}
}

func parseCtrType(s string) (ctrType, error) {
	switch s {
	case "Borders", "BinarizedTargetMeanValue", "FloatTargetMeanValue":
		return ctrTypeBorders, nil
	case "Buckets":
		return ctrTypeBuckets, nil
	case "Counter":
		return ctrTypeCounter, nil
	case "FeatureFreq":
		return ctrTypeFeqFreq, nil
	default:
		return 0, fmt.Errorf("unknown CTR type %q", s)
	}
}

func parseNanTreatment(s string) nanMode {
	switch s {
	case "AsFalse":
		return nanAsFalse
	case "AsTrue":
		return nanAsTrue
	default:
		return nanAsIs
	}
}

func buildCtrFeatures(jm *jsonModelFile) ([]ctrFeature, map[int]int, error) {
	if len(jm.FeaturesInfo.CtrFeatures) == 0 {
		return nil, nil, nil
	}

	// Count how many binary feature slots come before CTR features.
	ctrSplitStart := 0
	for _, ff := range jm.FeaturesInfo.FloatFeatures {
		ctrSplitStart += len(ff.Borders)
	}
	for _, cf := range jm.FeaturesInfo.CatFeatures {
		ctrSplitStart += len(cf.Values)
	}

	// Build per-identifier CTR value tables from ctr_data.
	tables := make(map[string]*ctrValueTable, len(jm.CtrData))
	for identifier, jt := range jm.CtrData {
		tbl, err := parseCtrTable(jt)
		if err != nil {
			return nil, nil, fmt.Errorf("CTR table %q: %w", identifier, err)
		}
		tables[identifier] = tbl
	}

	feats := make([]ctrFeature, 0, len(jm.FeaturesInfo.CtrFeatures))
	splitIdxToCtr := make(map[int]int)
	offset := ctrSplitStart

	for _, jcf := range jm.FeaturesInfo.CtrFeatures {
		tbl, ok := tables[jcf.Identifier]
		if !ok {
			return nil, nil, fmt.Errorf("ctr_data missing entry for identifier %q", jcf.Identifier)
		}

		proj, err := parseCtrProjection(jcf.Elements)
		if err != nil {
			return nil, nil, fmt.Errorf("CTR feature %q: %w", jcf.Identifier, err)
		}
		ct, err := parseCtrType(jcf.CtrType)
		if err != nil {
			return nil, nil, fmt.Errorf("CTR feature %q: %w", jcf.Identifier, err)
		}

		if ct != ctrTypeCounter && ct != ctrTypeFeqFreq && jcf.TargetBorderIdx >= tbl.stride {
			return nil, nil, fmt.Errorf("CTR %q: targetBorderIdx %d out of range [0, %d)", jcf.Identifier, jcf.TargetBorderIdx, tbl.stride)
		}

		ctrIdx := len(feats)
		feats = append(feats, ctrFeature{
			projection:      proj,
			ctrType:         ct,
			targetBorderIdx: jcf.TargetBorderIdx,
			priorNum:        jcf.PriorNum,
			priorDenom:      jcf.PriorDenom,
			scale:           jcf.Scale,
			shift:           jcf.Shift,
			table:           tbl,
		})

		for range jcf.Borders {
			splitIdxToCtr[offset] = ctrIdx
			offset++
		}
	}

	return feats, splitIdxToCtr, nil
}

func parseCtrProjection(elements []jsonCombElement) ([]ctrProjElem, error) {
	proj := make([]ctrProjElem, 0, len(elements))
	for _, elem := range elements {
		switch elem.CombElement {
		case "cat_feature_value":
			proj = append(proj, ctrProjElem{elemType: ctrElemCat, catIdx: elem.CatFeatureIndex})
		case "float_feature":
			proj = append(proj, ctrProjElem{elemType: ctrElemFloat, floatIdx: elem.FloatFeatIndex, border: elem.Border})
		case "cat_feature_exact_value":
			proj = append(proj, ctrProjElem{elemType: ctrElemExactCat, catIdx: elem.CatFeatureIndex, hashValue: elem.Value})
		default:
			return nil, fmt.Errorf("unsupported combination element type %q", elem.CombElement)
		}
	}
	return proj, nil
}

func parseCtrTable(jt *jsonCtrTable) (*ctrValueTable, error) {
	stride := jt.HashStride - 1 // data values per bucket
	if jt.HashStride < 1 {
		return nil, fmt.Errorf("invalid hash_stride %d", jt.HashStride)
	}
	if len(jt.HashMap)%jt.HashStride != 0 {
		return nil, fmt.Errorf("hash_map length %d not divisible by hash_stride %d", len(jt.HashMap), jt.HashStride)
	}

	n := len(jt.HashMap) / jt.HashStride
	lookup := make(map[uint64]int, n)
	data := make([]int32, n*stride)

	for i := 0; i < len(jt.HashMap); i += jt.HashStride {
		rowIdx := i / jt.HashStride

		var hashStr string
		if err := json.Unmarshal(jt.HashMap[i], &hashStr); err != nil {
			return nil, fmt.Errorf("hash key at position %d: %w", i, err)
		}

		h, err := strconv.ParseUint(hashStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parsing hash %q: %w", hashStr, err)
		}

		lookup[h] = rowIdx

		for j := 0; j < stride; j++ {
			var v int32
			if err := json.Unmarshal(jt.HashMap[i+1+j], &v); err != nil {
				return nil, fmt.Errorf("hash_map data at position %d: %w", i+1+j, err)
			}
			data[rowIdx*stride+j] = v
		}
	}

	return &ctrValueTable{
		lookup:       lookup,
		data:         data,
		stride:       stride,
		counterDenom: jt.CounterDenominator,
	}, nil
}

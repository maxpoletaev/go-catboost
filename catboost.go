package catboost

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"
)

type PredictionType int

const (
	PredictionTypeRawFormulaVal PredictionType = iota
	PredictionTypeProbability
	PredictionTypeClass
	PredictionTypeExponent
	PredictionTypeLogProbability
)

type nanMode int8

const (
	nanAsIs    nanMode = 0
	nanAsFalse nanMode = 1
	nanAsTrue  nanMode = 2
)

type splitKind int8

const (
	splitKindFloat  splitKind = iota // float threshold on floatVals[featureIndex]
	splitKindOneHot                  // equality check: catHashes[featureIndex] == hashValue
	splitKindCTR                     // CTR value threshold: computeCTR(ctrIdx) > border
)

type treeSplit struct {
	featureIndex int
	border       float32
	hashValue    int32
	nanMode      nanMode
	kind         splitKind
	ctrIdx       int
}

type stepNode struct {
	leftDiff  uint16
	rightDiff uint16
}

// Model holds a deserialized CatBoost model ready for inference.
type Model struct {
	approxDimension int
	predictionType  PredictionType

	treeSizes           []int
	treeStartOffsets    []int
	splits              []treeSplit
	leafValues          []float64
	treeFirstLeafOffset []int

	stepNodes      []stepNode
	nodeIdToLeafId []uint32
	isOblivious    bool

	scale                float64
	bias                 []float64
	binclassRawValBorder float64

	floatFeatureCount int
	catFeatureCount   int
	modelInfo         map[string]string
	ctrFeatures       []ctrFeature
}

func nanSubstitution(mode nanMode) (float32, bool) {
	switch mode {
	case nanAsFalse:
		return float32(math.Inf(-1)), true
	case nanAsTrue:
		return float32(math.Inf(1)), true
	default:
		return 0, false
	}
}

// LoadFromFile loads a CatBoost model from the given file path.
// The file is expected to be in json format, optionally gzipped.
func LoadFromFile(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("reading model file: %w", err)
	}
	defer f.Close()
	return LoadFromReader(f)
}

// LoadFromReader loads a CatBoost model from the given reader.
// The contents should be in json format.
func LoadFromReader(r io.Reader) (*Model, error) {
	var jm jsonModelFile

	var magic [2]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("reading model: %w", err)
	}

	r = io.MultiReader(bytes.NewReader(magic[:]), r)
	if magic == [2]byte{0x1f, 0x8b} {
		gzReader, err := gzip.NewReader(r)
		if err != nil {
			return nil, fmt.Errorf("opening gzip model: %w", err)
		}
		defer gzReader.Close()
		r = gzReader
	}

	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&jm); err != nil {
		return nil, fmt.Errorf("parsing JSON model: %w", err)
	}

	return buildModel(&jm)
}

// CalcSingle calculates a prediction for a single object.
// Pass float32(math.NaN()) for missing float values.
// Pass nil for catFeatures if the model has no categorical features.
func (m *Model) CalcSingle(floatFeatures []float32, catFeatures []string) ([]float64, error) {
	if err := validateInputs(m, floatFeatures, catFeatures); err != nil {
		return nil, err
	}
	var catHashes [][]int32
	if len(catFeatures) > 0 {
		hashes := make([]int32, len(catFeatures))
		for i, s := range catFeatures {
			hashes[i] = int32(CatFeatureHash(s))
		}
		catHashes = [][]int32{hashes}
	}
	return m.evalBatch([][]float32{floatFeatures}, catHashes)[0], nil
}

// CalcHashedSingle calculates a prediction using pre-computed cat feature hashes.
// Use CatFeatureHash to compute the hashes, which avoids re-hashing on every call.
func (m *Model) CalcHashedSingle(floatFeatures []float32, catHashes []int32) ([]float64, error) {
	if err := validateInputs(m, floatFeatures, catHashes); err != nil {
		return nil, err
	}
	var allHashes [][]int32
	if len(catHashes) > 0 {
		allHashes = [][]int32{catHashes}
	}
	return m.evalBatch([][]float32{floatFeatures}, allHashes)[0], nil
}

// Calc calculates predictions for multiple objects.
// Pass nil for catFeatures if the model has no categorical features.
func (m *Model) Calc(floatFeatures [][]float32, catFeatures [][]string) ([][]float64, error) {
	n := len(floatFeatures)
	if catFeatures != nil && len(catFeatures) != n {
		return nil, fmt.Errorf("floatFeatures and catFeatures have different lengths (%d vs %d)", n, len(catFeatures))
	}

	for i := range floatFeatures {
		var cats []string
		if catFeatures != nil {
			cats = catFeatures[i]
		}
		if err := validateInputs(m, floatFeatures[i], cats); err != nil {
			return nil, fmt.Errorf("document %d: %w", i, err)
		}
	}

	var allHashes [][]int32
	if len(catFeatures) != 0 {
		nCat := len(catFeatures[0])
		flat := make([]int32, n*nCat)
		allHashes = make([][]int32, n)

		for i, cats := range catFeatures {
			row := flat[i*nCat : (i+1)*nCat]
			for j, s := range cats {
				row[j] = int32(CatFeatureHash(s))
			}
			allHashes[i] = row
		}
	}

	return m.evalBatch(floatFeatures, allHashes), nil
}

// CalcHashed calculates predictions for multiple objects using pre-computed cat hashes.
func (m *Model) CalcHashed(floatFeatures [][]float32, catHashes [][]int32) ([][]float64, error) {
	n := len(floatFeatures)
	if catHashes != nil && len(catHashes) != n {
		return nil, fmt.Errorf("floatFeatures and catHashes have different lengths (%d vs %d)", n, len(catHashes))
	}

	for i := range floatFeatures {
		var hashes []int32
		if len(catHashes) != 0 {
			hashes = catHashes[i]
		}
		if err := validateInputs(m, floatFeatures[i], hashes); err != nil {
			return nil, fmt.Errorf("document %d: %w", i, err)
		}
	}

	return m.evalBatch(floatFeatures, catHashes), nil
}

// DimensionsCount returns 1 for regression/binary classification, N for multiclass.
func (m *Model) DimensionsCount() int {
	return m.approxDimension
}

// FloatFeaturesCount returns the number of float features the model expects.
func (m *Model) FloatFeaturesCount() int {
	return m.floatFeatureCount
}

// CatFeaturesCount returns the number of categorical features the model expects.
func (m *Model) CatFeaturesCount() int {
	return m.catFeatureCount
}

// InfoValue returns model metainfo for the given key.
func (m *Model) InfoValue(key string) (string, bool) {
	v, ok := m.modelInfo[key]
	return v, ok
}

// SetPredictionType sets the prediction type. Default is RawFormulaVal.
func (m *Model) SetPredictionType(pt PredictionType) {
	m.predictionType = pt
}

// SetProbabilityBorder sets the probability threshold for binary Class predictions.
// The value must be in (0, 1). Default is 0.5 (equivalent to raw value border of 0).
func (m *Model) SetProbabilityBorder(p float64) {
	if p <= 0 || p >= 1 {
		panic(fmt.Sprintf("probability border must be in (0, 1), got %v", p))
	}
	m.binclassRawValBorder = -math.Log(1/p - 1)
}

// CatFeatureHash returns the CatBoost hash for a categorical feature value.
func CatFeatureHash(s string) uint32 {
	sd := unsafe.Slice(unsafe.StringData(s), len(s))
	return uint32(cityHash64(sd) & 0xffffffff)
}

func validateInputs[T any](m *Model, floatFeatures []float32, catFeatures []T) error {
	if len(floatFeatures) < m.floatFeatureCount {
		return fmt.Errorf("expected %d float features, got %d", m.floatFeatureCount, len(floatFeatures))
	}
	if len(catFeatures) < m.catFeatureCount {
		return fmt.Errorf("expected %d cat features, got %d", m.catFeatureCount, len(catFeatures))
	}
	return nil
}

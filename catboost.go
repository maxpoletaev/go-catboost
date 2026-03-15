package catboost

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/maxpoletaev/go-catboost/internal/cbm"
)

const (
	cbmMagic        = "CBM1"
	formatVersion   = "FlabuffersModel_v1"
	maxValuesPerBin = 254
)

type PredictionType int

const (
	PredictionTypeRawFormulaVal PredictionType = iota
	PredictionTypeProbability
	PredictionTypeClass
	PredictionTypeExponent
	PredictionTypeLogProbability
)

type nanValueTreatment int8

const (
	nanAsIs    nanValueTreatment = 0
	nanAsFalse nanValueTreatment = 1
	nanAsTrue  nanValueTreatment = 2
)

type floatFeatureMeta struct {
	index   int // position in the caller's float feature slice
	borders []float32
	nanMode nanValueTreatment
}

type catFeatureMeta struct {
	index int // position in the caller's cat feature slice
}

type oneHotFeatureMeta struct {
	catFeatureIndex int // packed index among used cat features
	values          []int32
}

type repackedBin struct {
	featureIndex uint16
	xorMask      byte
	splitIdx     byte
}

type stepNode struct {
	leftDiff  uint16
	rightDiff uint16
}

// Model holds a deserialized CatBoost model.
type Model struct {
	approxDimension int
	predictionType  PredictionType

	floatFeatures  []floatFeatureMeta
	catFeatures    []catFeatureMeta
	oneHotFeatures []oneHotFeatureMeta
	needXorMask    bool

	treeSizes        []int
	treeStartOffsets []int
	repackedBins     []repackedBin

	leafValues          []float64
	treeFirstLeafOffset []int

	stepNodes      []stepNode
	nodeIdToLeafId []uint32
	isOblivious    bool

	scale             float64
	bias              []float64
	effectiveBinCount int

	minFloatFeatures int
	minCatFeatures   int

	modelInfo map[string]string
}

// LoadFromFile loads a model from a .cbm file.
func LoadFromFile(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading model file: %w", err)
	}
	return LoadFromBuffer(data)
}

// LoadFromBuffer loads a model from an in-memory buffer.
func LoadFromBuffer(data []byte) (*Model, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("model data too short")
	}

	// Verify magic bytes "CBM1".
	if string(data[:4]) != cbmMagic {
		return nil, fmt.Errorf("invalid model magic bytes, expected %q", cbmMagic)
	}

	// Read the FlatBuffers core size (little-endian uint32 at offset 4).
	coreSize := binary.LittleEndian.Uint32(data[4:8])

	// Handle large models: if coreSize == 0xffffffff, the actual size is in the
	// next 8 bytes as a uint64 (matching the C++ LoadSize logic).
	offset := 8
	var fbSize int
	if coreSize == 0xffffffff {
		if len(data) < 16 {
			return nil, fmt.Errorf("model data too short for extended size")
		}
		fbSize = int(binary.LittleEndian.Uint64(data[8:16]))
		offset = 16
	} else {
		fbSize = int(coreSize)
	}

	if offset+fbSize > len(data) {
		return nil, fmt.Errorf("model data truncated: need %d bytes, have %d", offset+fbSize, len(data))
	}

	fbData := data[offset : offset+fbSize]

	// Verify FlatBuffers identifier, flatc sets it to "CBMC".
	// We use GetRootAsTModelCore which doesn't verify, so parse manually.
	core := cbm.GetRootAsTModelCore(fbData, 0)

	if string(core.FormatVersion()) != formatVersion {
		return nil, fmt.Errorf("unsupported format version %q (expected %q)",
			core.FormatVersion(), formatVersion)
	}

	var fbTrees cbm.TModelTrees
	if core.ModelTrees(&fbTrees) == nil {
		return nil, fmt.Errorf("model has no trees")
	}

	// Check for unsupported features.
	if fbTrees.CtrFeaturesLength() > 0 {
		return nil, fmt.Errorf("model uses CTR features which are not yet supported")
	}
	if fbTrees.TextFeaturesLength() > 0 {
		return nil, fmt.Errorf("model uses text features which are not yet supported")
	}
	if fbTrees.EmbeddingFeaturesLength() > 0 {
		return nil, fmt.Errorf("model uses embedding features which are not yet supported")
	}

	m, err := deserializeModel(&fbTrees)
	if err != nil {
		return nil, fmt.Errorf("deserializing model: %w", err)
	}

	// Collect model info.
	for i := 0; i < core.InfoMapLength(); i++ {
		var kv cbm.TKeyValue
		if core.InfoMap(&kv, i) {
			m.modelInfo[string(kv.Key())] = string(kv.Value())
		}
	}

	return m, nil
}

func hashCatFeatures(catFeatures []string) []int32 {
	hashes := make([]int32, len(catFeatures))
	for i, s := range catFeatures {
		hashes[i] = int32(CatFeatureHash(s))
	}
	return hashes
}

// CalcSingle calculates model prediction on float features and string categorical
// feature values for a single object. Pass float32(math.NaN()) for missing float
// values. Pass nil for catFeatures if the model has no categorical features.
func (m *Model) CalcSingle(floatFeatures []float32, catFeatures []string) ([]float64, error) {
	if err := m.validateInputs(floatFeatures, catFeatures); err != nil {
		return nil, err
	}
	bins := m.quantize(floatFeatures, hashCatFeatures(catFeatures))
	return m.eval(bins), nil
}

// CalcHashedSingle calculates model prediction on float features and hashed
// categorical feature values for a single object. Use CatFeatureHash to compute
// the hashes, which avoids re-hashing the same strings on every call.
func (m *Model) CalcHashedSingle(floatFeatures []float32, catHashes []int32) ([]float64, error) {
	if err := m.validateHashedInputs(floatFeatures, catHashes); err != nil {
		return nil, err
	}
	bins := m.quantize(floatFeatures, catHashes)
	return m.eval(bins), nil
}

// Calc calculates model predictions on float features and string categorical
// feature values for multiple objects. Pass nil for catFeatures if the model
// has no categorical features.
func (m *Model) Calc(floatFeatures [][]float32, catFeatures [][]string) ([][]float64, error) {
	n := len(floatFeatures)
	if catFeatures != nil && len(catFeatures) != n {
		return nil, fmt.Errorf("floatFeatures and catFeatures have different lengths (%d vs %d)", n, len(catFeatures))
	}

	results := make([][]float64, n)
	for i := range results {
		var cats []string
		if catFeatures != nil {
			cats = catFeatures[i]
		}
		if err := m.validateInputs(floatFeatures[i], cats); err != nil {
			return nil, fmt.Errorf("document %d: %w", i, err)
		}
		bins := m.quantize(floatFeatures[i], hashCatFeatures(cats))
		results[i] = m.eval(bins)
	}
	return results, nil
}

// CalcHashed calculates model predictions on float features and hashed categorical
// feature values for multiple objects. Use CatFeatureHash to compute the hashes,
// which avoids re-hashing the same strings on every call.
func (m *Model) CalcHashed(floatFeatures [][]float32, catHashes [][]int32) ([][]float64, error) {
	n := len(floatFeatures)
	if catHashes != nil && len(catHashes) != n {
		return nil, fmt.Errorf("floatFeatures and catHashes have different lengths (%d vs %d)", n, len(catHashes))
	}
	results := make([][]float64, n)
	for i := range results {
		var hashes []int32
		if catHashes != nil {
			hashes = catHashes[i]
		}
		if err := m.validateHashedInputs(floatFeatures[i], hashes); err != nil {
			return nil, fmt.Errorf("document %d: %w", i, err)
		}
		bins := m.quantize(floatFeatures[i], hashes)
		results[i] = m.eval(bins)
	}
	return results, nil
}

// DimensionsCount returns the number of dimensions in the model:
// 1 for regression and binary classification, N for N-class multiclass.
func (m *Model) DimensionsCount() int {
	return m.approxDimension
}

// FloatFeaturesCount returns the minimum float feature slice length required by the model.
// Ref: catboost/libs/model/model.h (GetNumFloatFeatures)
func (m *Model) FloatFeaturesCount() int {
	return m.minFloatFeatures
}

// CatFeaturesCount returns the minimum cat feature slice length required by the model.
// Ref: catboost/libs/model/model.h (GetNumCatFeatures)
func (m *Model) CatFeaturesCount() int {
	return m.minCatFeatures
}

// InfoValue returns model metainfo for the given key. If the key is missing,
// the second return value is false.
func (m *Model) InfoValue(key string) (string, bool) {
	v, ok := m.modelInfo[key]
	return v, ok
}

// SetPredictionType sets the prediction type for model evaluation.
// The default is RawFormulaVal.
func (m *Model) SetPredictionType(pt PredictionType) {
	m.predictionType = pt
}

func (m *Model) validateInputs(floatFeatures []float32, catFeatures []string) error {
	if len(floatFeatures) < m.minFloatFeatures {
		return fmt.Errorf("need at least %d float features, got %d", m.minFloatFeatures, len(floatFeatures))
	}
	if len(catFeatures) < m.minCatFeatures {
		return fmt.Errorf("need at least %d cat features, got %d", m.minCatFeatures, len(catFeatures))
	}
	return nil
}

func (m *Model) validateHashedInputs(floatFeatures []float32, catHashes []int32) error {
	if len(floatFeatures) < m.minFloatFeatures {
		return fmt.Errorf("need at least %d float features, got %d", m.minFloatFeatures, len(floatFeatures))
	}
	if len(catHashes) < m.minCatFeatures {
		return fmt.Errorf("need at least %d cat features, got %d", m.minCatFeatures, len(catHashes))
	}
	return nil
}

// CatFeatureHash returns the hash for the given categorical feature string value.
// NOTE: CatBoost uses a fork of CityHash 1.0 that differs from the standard Google
// CityHash implementation.
func CatFeatureHash(s string) uint32 {
	return uint32(cityHash64([]byte(s)) & 0xffffffff)
}

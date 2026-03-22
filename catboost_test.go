package catboost_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/maxpoletaev/go-catboost"
)

var fixtures = []string{
	"float_regression",
	"float_binary",
	"multiclass_3",
	"multiclass_5",
	"cat_onehot_small",
	"multi_cat",
	"unknown_cat",
	"nan_as_false",
	"nan_as_true",
	"many_borders",
	"many_cat_values",
	"deep_trees",
	"shallow_trees",
	"single_tree",
	"large_features",
	"extreme_values",
	"mixed_multiclass",
	"nonsym_regression",
	"nonsym_binary",
	"nonsym_multiclass",
	"multiclass_class",
	"ctr_binary",
}

type fixture struct {
	Description       string      `json:"description"`
	PredictionType    string      `json:"prediction_type"` // omitted → "RawFormulaVal"
	FloatFeatureCount int         `json:"float_feature_count"`
	CatFeatureCount   int         `json:"cat_feature_count"`
	OutputDimension   int         `json:"output_dimension"`
	Samples           []sample    `json:"samples"`
	Predictions       [][]float64 `json:"predictions"`
}

type sample struct {
	Floats []nullableFloat `json:"floats"`
	Cats   []string        `json:"cats"`
}

type nullableFloat struct{ v float32 }

func (n *nullableFloat) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		n.v = float32(math.NaN())
		return nil
	}

	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return err
	}

	n.v = float32(f)
	return nil
}

func predictionTypeFromString(s string) (catboost.PredictionType, bool) {
	switch s {
	case "", "RawFormulaVal":
		return catboost.PredictionTypeRawFormulaVal, true
	case "Probability":
		return catboost.PredictionTypeProbability, true
	case "Class":
		return catboost.PredictionTypeClass, true
	case "Exponent":
		return catboost.PredictionTypeExponent, true
	case "LogProbability":
		return catboost.PredictionTypeLogProbability, true
	default:
		return 0, false
	}
}

func toFloats(ns []nullableFloat) []float32 {
	out := make([]float32, len(ns))
	for i, n := range ns {
		out[i] = n.v
	}
	return out
}

// ulpDiff returns how many floats apart a and b are. This works because IEEE 754
// bit patterns for same-sign floats are ordered the same way as integers, so we
// can just subtract them. Unlike a fixed epsilon, this scales with the magnitude
// of the values being compared.
func ulpDiff(a, b float64) uint64 {
	ai := math.Float64bits(a)
	bi := math.Float64bits(b)
	if ai > bi {
		return ai - bi
	}
	return bi - ai
}

func nearEqual(a, b float64) bool {
	const maxULPs = 2
	switch {
	case a == b:
		return true
	case math.IsNaN(a) || math.IsNaN(b):
		return false
	case math.Signbit(a) != math.Signbit(b):
		return false
	default:
		return ulpDiff(a, b) <= maxULPs
	}
}

func loadFixture(t testing.TB, name string) (fixture, *catboost.Model) {
	t.Helper()

	jsonData, err := os.ReadFile("testdata/" + name + "_test.json")
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}

	var fix fixture
	if err := json.Unmarshal(jsonData, &fix); err != nil {
		t.Fatalf("parsing fixture: %v", err)
	}

	m, err := catboost.LoadFromFile("testdata/" + name + "_model.json.gz")
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}

	pt, ok := predictionTypeFromString(fix.PredictionType)
	if !ok {
		t.Fatalf("unknown prediction_type %q", fix.PredictionType)
	}
	m.SetPredictionType(pt)

	return fix, m
}

func TestMetadata(t *testing.T) {
	for _, name := range fixtures {
		name := name
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			fix, m := loadFixture(t, name)
			t.Logf("%s", fix.Description)

			if got := m.DimensionsCount(); got != fix.OutputDimension {
				t.Errorf("DimensionsCount = %d, want %d", got, fix.OutputDimension)
			}
			if got := m.FloatFeaturesCount(); got != fix.FloatFeatureCount {
				t.Errorf("FloatFeaturesCount = %d, want %d", got, fix.FloatFeatureCount)
			}
			if got := m.CatFeaturesCount(); got != fix.CatFeatureCount {
				t.Errorf("CatFeaturesCount = %d, want %d", got, fix.CatFeatureCount)
			}
		})
	}
}

func TestCalcSingle(t *testing.T) {
	for _, name := range fixtures {
		name := name
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			fix, m := loadFixture(t, name)

			for i, s := range fix.Samples {
				got, err := m.CalcSingle(toFloats(s.Floats), s.Cats)
				if err != nil {
					t.Fatalf("sample %d: %v", i, err)
				}

				want := fix.Predictions[i]
				if len(got) != len(want) {
					t.Fatalf("sample %d: output length %d, want %d", i, len(got), len(want))
				}

				for dim := range want {
					if !nearEqual(got[dim], want[dim]) {
						t.Errorf("sample %d dim %d: got %.17g, want %.17g", i, dim, got[dim], want[dim])
					}
				}
			}
		})
	}
}

func TestCalc(t *testing.T) {
	for _, name := range fixtures {
		name := name
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			fix, m := loadFixture(t, name)

			allFloats := make([][]float32, len(fix.Samples))
			var allCats [][]string
			if fix.CatFeatureCount > 0 {
				allCats = make([][]string, len(fix.Samples))
			}
			for i, s := range fix.Samples {
				allFloats[i] = toFloats(s.Floats)
				if allCats != nil {
					allCats[i] = s.Cats
				}
			}

			batch, err := m.Calc(allFloats, allCats)
			if err != nil {
				t.Fatalf("Calc: %v", err)
			}

			for i, want := range fix.Predictions {
				for dim := range want {
					if !nearEqual(batch[i][dim], want[dim]) {
						t.Errorf("sample %d dim %d: got %.17g, want %.17g", i, dim, batch[i][dim], want[dim])
					}
				}
			}
		})
	}
}

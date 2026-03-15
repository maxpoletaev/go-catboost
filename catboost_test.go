package catboost_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	catboost "github.com/maxpoletaev/go-catboost"
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

	// Non-symmetric trees (grow_policy=Lossguide / Depthwise)
	"nonsym_regression",
	"nonsym_binary",
	"nonsym_multiclass",
}

// --------------------------------------------------------------------------
// JSON fixture type
// --------------------------------------------------------------------------

type fixture struct {
	Description       string      `json:"description"`
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

// nullableFloat unmarshals a JSON number or null into a float32.
// null becomes NaN (used to represent missing values).
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

func toFloats(ns []nullableFloat) []float32 {
	out := make([]float32, len(ns))
	for i, n := range ns {
		out[i] = n.v
	}
	return out
}

// --------------------------------------------------------------------------
// Table-driven test
// --------------------------------------------------------------------------

func TestFixtures(t *testing.T) {
	for _, name := range fixtures {
		name := name
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Load fixture JSON.
			jsonData, err := os.ReadFile("testdata/" + name + ".json")
			if err != nil {
				t.Fatalf("reading fixture: %v", err)
			}
			var fix fixture
			if err := json.Unmarshal(jsonData, &fix); err != nil {
				t.Fatalf("parsing fixture: %v", err)
			}
			t.Logf("%s", fix.Description)

			// Load model.
			m, err := catboost.LoadFromFile("testdata/" + name + ".cbm")
			if err != nil {
				t.Fatalf("LoadFromFile: %v", err)
			}

			// Verify model metadata matches fixture.
			if got := m.DimensionsCount(); got != fix.OutputDimension {
				t.Errorf("DimensionsCount = %d, want %d", got, fix.OutputDimension)
			}
			if got := m.FloatFeaturesCount(); got != fix.FloatFeatureCount {
				t.Errorf("FloatFeaturesCount = %d, want %d", got, fix.FloatFeatureCount)
			}
			if got := m.CatFeaturesCount(); got != fix.CatFeatureCount {
				t.Errorf("CatFeaturesCount = %d, want %d", got, fix.CatFeatureCount)
			}

			// Single-doc predictions.
			for i, s := range fix.Samples {
				got, err := m.CalcSingle(toFloats(s.Floats), s.Cats)
				if err != nil {
					t.Fatalf("sample %d CalcSingle: %v", i, err)
				}
				want := fix.Predictions[i]
				if len(got) != len(want) {
					t.Fatalf("sample %d: output length %d, want %d", i, len(got), len(want))
				}
				for dim := range want {
					if got[dim] != want[dim] {
						t.Errorf("sample %d dim %d: got %.17g, want %.17g",
							i, dim, got[dim], want[dim])
					}
				}
			}

			// Batch predictions must match single-doc results.
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
					if batch[i][dim] != want[dim] {
						t.Errorf("batch sample %d dim %d: got %.17g, want %.17g",
							i, dim, batch[i][dim], want[dim])
					}
				}
			}
		})
	}
}

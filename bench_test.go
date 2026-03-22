package catboost_test

import (
	"math/rand/v2"
	"testing"
)

var benchCases = []string{
	"float_regression",
	"deep_trees",
	"large_features",
	"multiclass_3",
	"cat_onehot_small",
}

func catVocab(fix fixture) []string {
	seen := make(map[string]struct{})
	for _, s := range fix.Samples {
		for _, v := range s.Cats {
			seen[v] = struct{}{}
		}
	}
	vocab := make([]string, 0, len(seen))
	for v := range seen {
		vocab = append(vocab, v)
	}
	return vocab
}

func randBenchInputs(rng *rand.Rand, nFloat, nCat int, vocab []string) ([]float32, []string) {
	floats := make([]float32, nFloat)
	for i := range floats {
		floats[i] = float32(rng.NormFloat64())
	}
	if nCat == 0 {
		return floats, nil
	}
	cats := make([]string, nCat)
	for i := range cats {
		cats[i] = vocab[rng.IntN(len(vocab))]
	}
	return floats, cats
}

func makeBatch(fix fixture, vocab []string, size int) ([][]float32, [][]string) {
	rng := rand.New(rand.NewPCG(0, 0))
	allFloats := make([][]float32, size)
	var allCats [][]string
	if fix.CatFeatureCount > 0 {
		allCats = make([][]string, size)
	}
	for i := range allFloats {
		floats, cats := randBenchInputs(rng, fix.FloatFeatureCount, fix.CatFeatureCount, vocab)
		allFloats[i] = floats
		if allCats != nil {
			allCats[i] = cats
		}
	}
	return allFloats, allCats
}

func BenchmarkSingle(b *testing.B) {
	for _, name := range benchCases {
		b.Run(name, func(b *testing.B) {
			fix, m := loadFixture(b, name)
			vocab := catVocab(fix)
			rng := rand.New(rand.NewPCG(0, 0))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				floats, cats := randBenchInputs(rng, fix.FloatFeatureCount, fix.CatFeatureCount, vocab)
				if _, err := m.CalcSingle(floats, cats); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBatch100(b *testing.B) {
	for _, name := range benchCases {
		b.Run(name, func(b *testing.B) {
			fix, m := loadFixture(b, name)
			allFloats, allCats := makeBatch(fix, catVocab(fix), 100)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := m.Calc(allFloats, allCats); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBatch1000(b *testing.B) {
	for _, name := range benchCases {
		b.Run(name, func(b *testing.B) {
			fix, m := loadFixture(b, name)
			allFloats, allCats := makeBatch(fix, catVocab(fix), 1000)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := m.Calc(allFloats, allCats); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

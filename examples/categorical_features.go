package main

import (
	"fmt"
	"log"

	catboost "github.com/maxpoletaev/go-catboost"
)

func runCategoricalFeatures() {
	m, err := catboost.LoadFromFile("../testdata/cat_onehot_small.cbm")
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	fmt.Println("=== Categorical Features ===")
	fmt.Printf("float features: %d, cat features: %d\n", m.FloatFeaturesCount(), m.CatFeaturesCount())

	m.SetPredictionType(catboost.PredictionTypeProbability)

	// Cat features are passed as raw strings; the model hashes them internally.
	// An unseen value is silently treated as "no match" for all one-hot splits.
	samples := []struct {
		floats []float32
		cats   []string
	}{
		{[]float32{-0.626, 1.007, -0.795}, []string{"cat"}},
		{[]float32{0.071, 0.085, -0.456}, []string{"dog"}},
		{[]float32{0.300, -0.500, 0.100}, []string{"unknown_value"}},
	}

	for _, s := range samples {
		result, err := m.CalcSingle(s.floats, s.cats)
		if err != nil {
			log.Fatalf("predict: %v", err)
		}
		fmt.Printf("cat=%q  prob=%.4f\n", s.cats[0], result[0])
	}
}

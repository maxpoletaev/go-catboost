package main

import (
	"fmt"
	"log"

	catboost "github.com/maxpoletaev/go-catboost"
)

func runMulticlass() {
	m, err := catboost.LoadFromFile("../testdata/multiclass_3_model.json.gz")
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	fmt.Println("=== Multiclass Classification ===")
	fmt.Printf("float features: %d, classes: %d\n", m.FloatFeaturesCount(), m.DimensionsCount())

	m.SetPredictionType(catboost.PredictionTypeProbability)

	floats := []float32{1.276, 0.091, 1.174, -1.932}
	probs, err := m.CalcSingle(floats, nil)
	if err != nil {
		log.Fatalf("predict: %v", err)
	}

	best := 0
	fmt.Println("probabilities:")
	for i, p := range probs {
		fmt.Printf("  class %d: %.4f\n", i, p)
		if p > probs[best] {
			best = i
		}
	}
	fmt.Printf("predicted class: %d\n", best)
}

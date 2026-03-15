package main

import (
	"fmt"
	"log"

	catboost "github.com/maxpoletaev/go-catboost"
)

func runBinaryClassification() {
	m, err := catboost.LoadFromFile("../testdata/float_binary.cbm")
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	fmt.Println("=== Binary Classification ===")
	fmt.Printf("float features: %d\n", m.FloatFeaturesCount())

	m.SetPredictionType(catboost.PredictionTypeProbability)

	floats := []float32{0.572, 0.100, -1.567, -1.818, 1.509}
	result, err := m.CalcSingle(floats, nil)
	if err != nil {
		log.Fatalf("predict: %v", err)
	}

	fmt.Printf("probability: %.6f\n", result[0])
	if result[0] >= 0.5 {
		fmt.Println("predicted class: 1 (positive)")
	} else {
		fmt.Println("predicted class: 0 (negative)")
	}
}

package main

import (
	"fmt"
	"log"
	"math"

	catboost "github.com/maxpoletaev/go-catboost"
)

func runRegression() {
	m, err := catboost.LoadFromFile("../testdata/float_regression.cbm")
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	fmt.Println("=== Regression ===")
	fmt.Printf("float features: %d, output dimension: %d\n", m.FloatFeaturesCount(), m.DimensionsCount())

	// Single prediction.
	floats := []float32{-0.182, 1.070, 2.228, 0.180, -0.007, 0.256}
	result, err := m.CalcSingle(floats, nil)
	if err != nil {
		log.Fatalf("predict: %v", err)
	}
	fmt.Printf("prediction: %.6f\n", result[0])

	// Missing values: pass NaN for unknown features.
	floatsWithNaN := []float32{float32(math.NaN()), 1.070, 2.228, 0.180, -0.007, 0.256}
	result, err = m.CalcSingle(floatsWithNaN, nil)
	if err != nil {
		log.Fatalf("predict with NaN: %v", err)
	}
	fmt.Printf("prediction with NaN: %.6f\n", result[0])

	// Batch prediction.
	batch := [][]float32{
		{-0.182, 1.070, 2.228, 0.180, -0.007, 0.256},
		{-0.658, 0.765, 0.441, -0.420, 0.902, 0.019},
		{-0.970, 0.352, 0.100, 0.500, -0.200, 0.800},
	}
	results, err := m.Calc(batch, nil)
	if err != nil {
		log.Fatalf("predict batch: %v", err)
	}
	fmt.Println("batch predictions:")
	for i, r := range results {
		fmt.Printf("  [%d] %.6f\n", i, r[0])
	}
}

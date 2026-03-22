package main

import (
	"fmt"
	"log"

	catboost "github.com/maxpoletaev/go-catboost"
)

func runPrehashedCats() {
	m, err := catboost.LoadFromFile("../testdata/cat_onehot_small_model.json.gz")
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	m.SetPredictionType(catboost.PredictionTypeProbability)

	fmt.Println("=== Pre-hashed Categorical Features ===")

	// Pre-compute hashes once, then reuse across many predictions.
	// Useful when the same categorical values appear in many documents.
	catValues := []string{"cat", "dog", "unknown_value"}
	hashes := make([]int32, len(catValues))
	for i, v := range catValues {
		hashes[i] = int32(catboost.CatFeatureHash(v))
	}

	floats := [][]float32{
		{-0.626, 1.007, -0.795},
		{0.071, 0.085, -0.456},
		{0.300, -0.500, 0.100},
	}

	for i, h := range hashes {
		result, err := m.CalcHashedSingle(floats[i], []int32{h})
		if err != nil {
			log.Fatalf("predict: %v", err)
		}
		fmt.Printf("cat=%q  hash=%d  prob=%.4f\n", catValues[i], h, result[0])
	}
}

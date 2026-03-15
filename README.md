# go-catboost

Pure-Go implementation of CatBoost model evaluation library for cgo-restricted environments.

Unofficial research project. Not proven to be production-ready.

Check out [examples/](examples/).

## Caveats

* Only float and categorical features are supported. Text and embeddings require way more knowledge.
* Likely slower (not benchmarked), as the original C++ lib does a lot of SIMD.

# go-catboost

Pure-Go implementation of CatBoost model evaluation library for cgo-restricted environments.

Unofficial research project. Not proven to be production-ready.

Check out [examples/](examples).

## Caveats

* Only float and one-hot categorical features are supported for now.
* Likely slower (not benchmarked), as the original C++ lib does a lot of SIMD.

# go-catboost

Pure-Go zero-deps implementation of CatBoost model evaluation library.

Check out [examples](examples) and [godoc.md](godoc.md).

## Status

Proof-of-concept. Not proven to be production-ready.

## Caveats

* Only json (and json.gz) model formats are supported. Export with `save_model(..., format="json")` or use [cbm2json.py](cbm2json.py) to convert.
* Only float and categorical (one-hot and ctr) features are supported.
* Likely noticable slower (especially on large batches) than C++ core due lack of vectorization.

## How this was made

The work is largely based on [Explanation of Json model format of CatBoost][1] article by Paras Malik with heavy reference to the CatBoost C++ source code. The project is either a direct port or a reimplementation of the code that can be found in:

- `catboost/libs/model/cpu/evaluator_impl.cpp` - oblivious & non-symmetric tree evaluation
- `catboost/libs/model/eval_processing.h` - prediction types handling (sigmoid, softmax, class, argmax)
- `catboost/libs/model/model.h` - model and tree structs
- `catboost/libs/model/model_export/resources/ctr_calcer.cpp` - ctr value computation
- `catboost/libs/model/online_ctr.h` - ctr formula
- `catboost/libs/cat_feature/cat_feature.cpp` - cat feature hashing
- `util/digest/city.cpp` - CityHash implementation

[1]: https://parasmalik.blogspot.com/2020/07/explanation-of-json-model-format-of.html

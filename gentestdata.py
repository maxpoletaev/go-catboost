#!/usr/bin/env python3
"""
Generate test models and expected predictions for the go-catboost test suite.

Each fixture produces:
  {name}.cbm   - CatBoost binary model
  {name}.json  - feature info + samples + expected RawFormulaVal predictions

Run from the repo root:
  python3 gentestdata.py
"""

import json
import math
import os
import sys

import numpy as np

try:
    import catboost as cb
except ImportError:
    print("catboost not found. Install with: pip install catboost", file=sys.stderr)
    sys.exit(1)

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "testdata")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rng_floats(n_samples, n_features, nan_frac=0.0):
    """Generate float matrix, optionally injecting NaNs."""
    X = RNG.standard_normal((n_samples, n_features)).astype(np.float32)
    if nan_frac > 0:
        mask = RNG.random(X.shape) < nan_frac
        X[mask] = np.nan
    return X


def rng_cats(n_samples, vocabulary, n_features=1):
    """Generate categorical string columns."""
    cols = [RNG.choice(vocabulary, n_samples) for _ in range(n_features)]
    return np.column_stack(cols) if n_features > 1 else cols[0].reshape(-1, 1)


def build_pool(X_float, X_cat=None, y=None, cat_col_offset=0):
    """Build a catboost Pool from float + optional cat columns."""
    if X_cat is not None:
        X = np.column_stack([X_float, X_cat])
        cat_features = list(range(cat_col_offset, cat_col_offset + X_cat.shape[1]))
    else:
        X = X_float
        cat_features = []
    return cb.Pool(X, y, cat_features=cat_features if cat_features else None)


def predict_raw(model, X_float, X_cat=None, cat_col_offset=0):
    """Return RawFormulaVal predictions as list-of-lists."""
    pool = build_pool(X_float, X_cat, cat_col_offset=cat_col_offset)
    preds = model.predict(pool, prediction_type="RawFormulaVal")
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds.tolist()


def samples_json(X_float, X_cat=None):
    """Build the 'samples' list for the expected JSON."""
    samples = []
    for i in range(len(X_float)):
        s = {"floats": [None if math.isnan(v) else float(v) for v in X_float[i]]}
        if X_cat is not None:
            s["cats"] = X_cat[i].tolist()
        samples.append(s)
    return samples


def save(name, model, X_test_float, X_test_cat, float_count, cat_count, output_dim,
         cat_col_offset=0, description=""):
    model.save_model(os.path.join(OUT, f"{name}.cbm"), format="CatboostBinary")

    preds = predict_raw(model, X_test_float, X_test_cat, cat_col_offset=cat_col_offset)
    fixture = {
        "description": description,
        "float_feature_count": float_count,
        "cat_feature_count": cat_count,
        "output_dimension": output_dim,
        "samples": samples_json(X_test_float, X_test_cat),
        "predictions": preds,
    }
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"  [{name}] dim={output_dim} float={float_count} cat={cat_count} samples={len(preds)}")


# ---------------------------------------------------------------------------
# 1. Float regression
# ---------------------------------------------------------------------------
def gen_float_regression():
    n = 2000
    X = rng_floats(n, 6)
    y = X[:, 0] * 2 - X[:, 1] + X[:, 2] * 0.5 + RNG.standard_normal(n).astype(np.float32) * 0.1

    model = cb.CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05,
        verbose=False, random_seed=1,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 6)
    save("float_regression", model, X_test, None, 6, 0, 1,
         description="Regression on 6 float features")


# ---------------------------------------------------------------------------
# 2. Float binary classification
# ---------------------------------------------------------------------------
def gen_float_binary():
    n = 2000
    X = rng_floats(n, 5)
    y = (X[:, 0] + X[:, 1] * 2 - X[:, 2] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=150, depth=6, learning_rate=0.1,
        verbose=False, random_seed=2,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 5)
    save("float_binary", model, X_test, None, 5, 0, 1,
         description="Binary classification on 5 float features")


# ---------------------------------------------------------------------------
# 3. Multiclass — 3 classes
# ---------------------------------------------------------------------------
def gen_multiclass_3():
    n = 2000
    X = rng_floats(n, 4)
    y = np.argmax(X[:, :3], axis=1)

    model = cb.CatBoostClassifier(
        iterations=150, depth=5, learning_rate=0.1,
        verbose=False, random_seed=3, classes_count=3,
        loss_function="MultiClass",
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 4)
    save("multiclass_3", model, X_test, None, 4, 0, 3,
         description="3-class classification on 4 float features")


# ---------------------------------------------------------------------------
# 4. Multiclass — 5 classes
# ---------------------------------------------------------------------------
def gen_multiclass_5():
    n = 3000
    X = rng_floats(n, 6)
    y = np.argmax(X[:, :5], axis=1)

    model = cb.CatBoostClassifier(
        iterations=200, depth=5, learning_rate=0.1,
        verbose=False, random_seed=4, classes_count=5,
        loss_function="MultiClass",
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 6)
    save("multiclass_5", model, X_test, None, 6, 0, 5,
         description="5-class classification on 6 float features")


# ---------------------------------------------------------------------------
# 5. One-hot categorical + float (4 values)
# ---------------------------------------------------------------------------
def gen_cat_onehot_small():
    n = 2000
    vocab = ["cat", "dog", "bird", "fish"]
    X_float = rng_floats(n, 3)
    X_cat = rng_cats(n, vocab)
    y = (X_float[:, 0] + (X_cat[:, 0] == "cat").astype(float) > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.1,
        verbose=False, random_seed=5, one_hot_max_size=10,
    )
    model.fit(build_pool(X_float, X_cat, y, cat_col_offset=3))

    X_test_f = rng_floats(20, 3)
    X_test_c = rng_cats(20, vocab)
    save("cat_onehot_small", model, X_test_f, X_test_c, 3, 1, 1,
         cat_col_offset=3,
         description="Binary classification, 4-value one-hot cat + 3 floats")


# ---------------------------------------------------------------------------
# 6. Multiple categorical features
# ---------------------------------------------------------------------------
def gen_multi_cat():
    n = 2000
    vocab_a = ["red", "green", "blue"]
    vocab_b = ["small", "medium", "large", "xlarge"]
    vocab_c = ["yes", "no"]
    X_float = rng_floats(n, 2)
    X_cat = np.column_stack([
        rng_cats(n, vocab_a),
        rng_cats(n, vocab_b),
        rng_cats(n, vocab_c),
    ])
    y = (
        X_float[:, 0]
        + (X_cat[:, 0] == "red").astype(float)
        - (X_cat[:, 2] == "no").astype(float)
        > 0
    ).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.1,
        verbose=False, random_seed=6, one_hot_max_size=10,
    )
    model.fit(build_pool(X_float, X_cat, y, cat_col_offset=2))

    X_test_f = rng_floats(20, 2)
    X_test_c = np.column_stack([
        rng_cats(20, vocab_a),
        rng_cats(20, vocab_b),
        rng_cats(20, vocab_c),
    ])
    save("multi_cat", model, X_test_f, X_test_c, 2, 3, 1,
         cat_col_offset=2,
         description="3 one-hot cat features + 2 floats, binary classification")


# ---------------------------------------------------------------------------
# 7. Unknown (unseen) categorical value at prediction time
# ---------------------------------------------------------------------------
def gen_unknown_cat():
    """Test that unseen cat values at predict time don't crash and match
    catboost's behaviour (hash not found → bin 0 for all one-hot blocks)."""
    n = 2000
    vocab_train = ["alpha", "beta", "gamma", "delta"]
    X_float = rng_floats(n, 2)
    X_cat = rng_cats(n, vocab_train)
    y = (X_float[:, 0] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=80, depth=4, learning_rate=0.1,
        verbose=False, random_seed=7, one_hot_max_size=10,
    )
    model.fit(build_pool(X_float, X_cat, y, cat_col_offset=2))

    # Test samples include known and unknown values.
    X_test_f = rng_floats(10, 2)
    known = [["alpha"], ["beta"], ["gamma"], ["delta"]] * 2 + [["alpha"], ["beta"]]
    unknown = [["zzzunknown"], ["NEW_VALUE"]]
    X_test_c = np.array(known[:8] + unknown)
    save("unknown_cat", model, X_test_f, X_test_c, 2, 1, 1,
         cat_col_offset=2,
         description="Includes unseen cat values at predict time (bin 0)")


# ---------------------------------------------------------------------------
# 8. NaN values — nan_mode=Min (→ AsFalse in model)
# ---------------------------------------------------------------------------
def gen_nan_as_false():
    """nan_mode='Min' makes CatBoost treat NaN as less than all borders (AsFalse)."""
    n = 2000
    X = rng_floats(n, 4, nan_frac=0.15)
    y = (np.nansum(X[:, :2], axis=1) > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.1,
        verbose=False, random_seed=8, nan_mode="Min",
    )
    model.fit(cb.Pool(X, y))

    # Test samples contain NaN values.
    X_test = rng_floats(20, 4, nan_frac=0.25)
    save("nan_as_false", model, X_test, None, 4, 0, 1,
         description="Float features with NaN treated as Min (AsFalse)")


# ---------------------------------------------------------------------------
# 9. NaN values — nan_mode=Max (→ AsTrue in model)
# ---------------------------------------------------------------------------
def gen_nan_as_true():
    n = 2000
    X = rng_floats(n, 4, nan_frac=0.15)
    y = (np.nansum(X[:, :2], axis=1) > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.1,
        verbose=False, random_seed=9, nan_mode="Max",
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 4, nan_frac=0.25)
    save("nan_as_true", model, X_test, None, 4, 0, 1,
         description="Float features with NaN treated as Max (AsTrue)")


# ---------------------------------------------------------------------------
# 10. Many borders (>254 per feature) — tests multi-block float quantization
# ---------------------------------------------------------------------------
def gen_many_borders():
    """border_count=500 forces >254 borders on high-variance features."""
    n = 5000
    X = rng_floats(n, 3)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        verbose=False, random_seed=10,
        border_count=500,  # forces >254 borders → multi-block quantization
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 3)
    save("many_borders", model, X_test, None, 3, 0, 1,
         description="Float features with >254 borders (multi-block quantization)")


# ---------------------------------------------------------------------------
# 11. Many categorical values (>254) — tests multi-block one-hot
# ---------------------------------------------------------------------------
def gen_many_cat_values():
    """one_hot_max_size=300 with 300 distinct values → multi-block one-hot."""
    n = 5000
    # 300 distinct cat values
    vocab = [f"val_{i:03d}" for i in range(300)]
    X_float = rng_floats(n, 2)
    X_cat = rng_cats(n, vocab)
    # Make the label correlated with the float features so the tree has something to learn.
    y = (X_float[:, 0] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=50, depth=4, learning_rate=0.1,
        verbose=False, random_seed=11,
        one_hot_max_size=300,
    )
    model.fit(build_pool(X_float, X_cat, y, cat_col_offset=2))

    X_test_f = rng_floats(20, 2)
    # Include some known values + a few unseen ones
    known = RNG.choice(vocab, 18).reshape(-1, 1).tolist()
    unseen = [["val_NEW_A"], ["val_NEW_B"]]
    X_test_c = np.array(known + unseen)
    save("many_cat_values", model, X_test_f, X_test_c, 2, 1, 1,
         cat_col_offset=2,
         description="300-value one-hot (multi-block), includes unseen values")


# ---------------------------------------------------------------------------
# 12. Deep trees (depth 8)
# ---------------------------------------------------------------------------
def gen_deep_trees():
    n = 3000
    X = rng_floats(n, 8)
    y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=8, learning_rate=0.05,
        verbose=False, random_seed=12,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 8)
    save("deep_trees", model, X_test, None, 8, 0, 1,
         description="Trees of depth 8")


# ---------------------------------------------------------------------------
# 13. Shallow trees (depth 1 — decision stumps)
# ---------------------------------------------------------------------------
def gen_shallow_trees():
    n = 1000
    X = rng_floats(n, 4)
    y = (X[:, 0] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=200, depth=1, learning_rate=0.1,
        verbose=False, random_seed=13,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 4)
    save("shallow_trees", model, X_test, None, 4, 0, 1,
         description="Decision stumps (depth 1)")


# ---------------------------------------------------------------------------
# 14. Single tree
# ---------------------------------------------------------------------------
def gen_single_tree():
    n = 1000
    X = rng_floats(n, 3)
    y = (X[:, 0] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=1, depth=4, learning_rate=1.0,
        verbose=False, random_seed=14,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 3)
    save("single_tree", model, X_test, None, 3, 0, 1,
         description="Model with a single tree")


# ---------------------------------------------------------------------------
# 15. Large feature set (50 float features)
# ---------------------------------------------------------------------------
def gen_large_features():
    n = 3000
    X = rng_floats(n, 50)
    y = (X[:, :5].sum(axis=1) > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        verbose=False, random_seed=15,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 50)
    save("large_features", model, X_test, None, 50, 0, 1,
         description="50 float features")


# ---------------------------------------------------------------------------
# 16. Float regression with extreme values
# ---------------------------------------------------------------------------
def gen_extreme_values():
    """Very large, very small, and near-zero float values."""
    n = 1000
    X = rng_floats(n, 3)
    # Scale to extreme ranges
    X[:, 0] *= 1e6
    X[:, 1] *= 1e-6
    X[:, 2] = X[:, 2]
    y = (X[:, 0] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.1,
        verbose=False, random_seed=16,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 3)
    X_test[:, 0] *= 1e6
    X_test[:, 1] *= 1e-6
    save("extreme_values", model, X_test, None, 3, 0, 1,
         description="Float features with extreme magnitudes")


# ---------------------------------------------------------------------------
# 17. Mixed: float + multiple cat features + multiclass
# ---------------------------------------------------------------------------
def gen_mixed_multiclass():
    n = 3000
    vocab_a = ["A", "B", "C", "D"]
    vocab_b = ["x", "y", "z"]
    X_float = rng_floats(n, 3)
    X_cat = np.column_stack([rng_cats(n, vocab_a), rng_cats(n, vocab_b)])
    # 3-class label
    raw = X_float[:, 0] + (X_cat[:, 0] == "A").astype(float) - (X_cat[:, 1] == "z").astype(float)
    y = np.clip(np.floor(raw + 1.5).astype(int), 0, 2)

    model = cb.CatBoostClassifier(
        iterations=150, depth=5, learning_rate=0.1,
        verbose=False, random_seed=17,
        one_hot_max_size=10, classes_count=3, loss_function="MultiClass",
    )
    model.fit(build_pool(X_float, X_cat, y, cat_col_offset=3))

    X_test_f = rng_floats(20, 3)
    X_test_c = np.column_stack([rng_cats(20, vocab_a), rng_cats(20, vocab_b)])
    save("mixed_multiclass", model, X_test_f, X_test_c, 3, 2, 3,
         cat_col_offset=3,
         description="3-class, 3 floats + 2 one-hot cat features")


# ---------------------------------------------------------------------------
# 18. Non-symmetric regression (Lossguide)
# ---------------------------------------------------------------------------
def gen_nonsym_regression():
    n = 2000
    X = rng_floats(n, 6)
    y = X[:, 0] * 2 - X[:, 1] + X[:, 2] * 0.5

    model = cb.CatBoostRegressor(
        iterations=200, learning_rate=0.05, verbose=False,
        random_seed=20, grow_policy="Lossguide", max_leaves=16,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 6)
    save("nonsym_regression", model, X_test, None, 6, 0, 1,
         description="Non-symmetric regression (Lossguide, max_leaves=16)")


# ---------------------------------------------------------------------------
# 19. Non-symmetric binary classification (Lossguide)
# ---------------------------------------------------------------------------
def gen_nonsym_binary():
    n = 2000
    X = rng_floats(n, 5)
    y = (X[:, 0] + X[:, 1] * 2 - X[:, 2] > 0).astype(int)

    model = cb.CatBoostClassifier(
        iterations=150, learning_rate=0.1, verbose=False,
        random_seed=21, grow_policy="Lossguide", max_leaves=16,
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 5)
    save("nonsym_binary", model, X_test, None, 5, 0, 1,
         description="Non-symmetric binary classification (Lossguide, max_leaves=16)")


# ---------------------------------------------------------------------------
# 20. Non-symmetric multiclass (Depthwise)
# ---------------------------------------------------------------------------
def gen_nonsym_multiclass():
    n = 2000
    X = rng_floats(n, 4)
    y = np.argmax(X[:, :3], axis=1)

    model = cb.CatBoostClassifier(
        iterations=150, learning_rate=0.1, verbose=False,
        random_seed=22, grow_policy="Depthwise", depth=5,
        classes_count=3, loss_function="MultiClass",
    )
    model.fit(cb.Pool(X, y))

    X_test = rng_floats(20, 4)
    save("nonsym_multiclass", model, X_test, None, 4, 0, 3,
         description="Non-symmetric 3-class classification (Depthwise)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = [
    gen_float_regression,
    gen_float_binary,
    gen_multiclass_3,
    gen_multiclass_5,
    gen_cat_onehot_small,
    gen_multi_cat,
    gen_unknown_cat,
    gen_nan_as_false,
    gen_nan_as_true,
    gen_many_borders,
    gen_many_cat_values,
    gen_deep_trees,
    gen_shallow_trees,
    gen_single_tree,
    gen_large_features,
    gen_extreme_values,
    gen_mixed_multiclass,
    # Non-symmetric trees (grow_policy=Lossguide / Depthwise)
    gen_nonsym_regression,
    gen_nonsym_binary,
    gen_nonsym_multiclass,
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*",
        help="Names of fixtures to regenerate (default: all)",
    )
    args = parser.parse_args()

    selected = set(args.names) if args.names else set()

    print(f"Generating test fixtures → {OUT}/")
    for gen in GENERATORS:
        # Derive fixture name from function name: gen_foo_bar → foo_bar
        fixture_name = gen.__name__[len("gen_"):]
        if selected and fixture_name not in selected:
            continue
        gen()

    print(f"\nDone. {len(GENERATORS) if not selected else len(selected)} fixture(s) written to {OUT}/")

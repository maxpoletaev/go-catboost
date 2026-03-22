#!/usr/bin/env python3
"""
Benchmark CatBoost Python inference and compare with Go results.

Run from the repo root:
  .venv/bin/python3 bench.py

Adjust BATCH_SIZE and ROUNDS to taste.
"""

import gzip
import json
import os
import tempfile
import time

import numpy as np

try:
    import catboost as cb
except ImportError:
    raise SystemExit("catboost not found — activate venv: source .venv/bin/activate")

TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")
BATCH_SIZES = [100, 1000]
ROUNDS = 1000  # how many single-predict calls to time
BATCH_ROUNDS = 1000  # how many batch-predict calls to time
RNG = np.random.default_rng(0)

CASES = [
    "float_regression",
    "deep_trees",
    "large_features",
    "multiclass_3",
    "cat_onehot_small",
]


def load_fixture(name: str) -> dict:
    path = os.path.join(TESTDATA, f"{name}_test.json")
    with open(path) as f:
        data = json.load(f)
    n_float = data["float_feature_count"]
    n_cat = data["cat_feature_count"]
    vocab = None
    if n_cat > 0:
        vocab = list({v for s in data["samples"] for v in s["cats"]})
    return {"n_float": n_float, "n_cat": n_cat, "cat_vocab": vocab}


def load_model(name: str) -> cb.CatBoost:
    gz_path = os.path.join(TESTDATA, f"{name}_model.json.gz")
    with gzip.open(gz_path, "rb") as f:
        blob = f.read()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        model = cb.CatBoost()
        model.load_model(tmp_path, format="json")
    finally:
        os.unlink(tmp_path)
    return model


def rand_float_row(n_float: int) -> np.ndarray:
    return RNG.standard_normal(n_float).astype(np.float32)


def rand_cat_row(n_cat: int, vocab: list[str]) -> list[str]:
    return [vocab[RNG.integers(len(vocab))] for _ in range(n_cat)]


def make_pool_single(n_float, n_cat, vocab):
    floats = rand_float_row(n_float).reshape(1, -1)
    if n_cat == 0:
        return cb.Pool(floats)
    cats = [rand_cat_row(n_cat, vocab)]
    data = np.hstack([floats, np.array(cats)])
    cat_features = list(range(n_float, n_float + n_cat))
    return cb.Pool(data, cat_features=cat_features)


def make_pool_batch(n_float, n_cat, vocab, size):
    floats = RNG.standard_normal((size, n_float)).astype(np.float32)
    if n_cat == 0:
        return cb.Pool(floats)
    cats = np.array([[v for v in rand_cat_row(n_cat, vocab)] for _ in range(size)])
    data = np.hstack([floats, cats])
    cat_features = list(range(n_float, n_float + n_cat))
    return cb.Pool(data, cat_features=cat_features)


def bench(label: str, fn, rounds: int) -> float:
    """Run fn() `rounds` times; return throughput in calls/sec."""
    # Warmup
    for _ in range(min(10, rounds // 10)):
        fn()
    t0 = time.perf_counter()
    for _ in range(rounds):
        fn()
    elapsed = time.perf_counter() - t0
    return rounds / elapsed


def fmt_ns(calls_per_sec: float) -> str:
    ns = 1e9 / calls_per_sec
    if ns >= 1_000_000:
        return f"{ns/1_000_000:.1f} ms/op"
    if ns >= 1_000:
        return f"{ns/1_000:.1f} µs/op"
    return f"{ns:.1f} ns/op"


def fmt_batch(ns_per_op: float, size: int) -> str:
    ns_per_sample = ns_per_op / size
    op_str = (
        f"{ns_per_op/1_000:.1f} µs/op"
        if ns_per_op >= 1_000
        else f"{ns_per_op:.1f} ns/op"
    )
    return f"{op_str} ({ns_per_sample:.1f} ns/sample)"


def main():
    batch_headers = "  ".join(f"{'batch-' + str(s):>30}" for s in BATCH_SIZES)
    print(f"{'Model':<22}  {'single':>14}  {batch_headers}")
    print("-" * 100)

    for name in CASES:
        fixture = load_fixture(name)
        n_float = fixture["n_float"]
        n_cat = fixture["n_cat"]
        vocab = fixture["cat_vocab"]

        model = load_model(name)

        single_pool = make_pool_single(n_float, n_cat, vocab)
        single_tps = bench(name, lambda: model.predict(single_pool), ROUNDS)
        single_str = fmt_ns(single_tps)

        batch_strs = []
        for size in BATCH_SIZES:
            pool = make_pool_batch(n_float, n_cat, vocab, size)
            tps = bench(name, lambda: model.predict(pool), BATCH_ROUNDS)
            batch_strs.append(fmt_batch(1e9 / tps, size))

        batch_cols = "  ".join(f"{s:>30}" for s in batch_strs)
        print(f"{name:<22}  {single_str:>14}  {batch_cols}")

    print()
    print(
        f"Batch sizes: {BATCH_SIZES} | Single rounds: {ROUNDS} | Batch rounds: {BATCH_ROUNDS}"
    )


if __name__ == "__main__":
    main()

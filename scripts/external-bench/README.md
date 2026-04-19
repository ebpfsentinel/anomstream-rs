# External bench — rcf-rs vs Python / Java baselines

Reproducible speed + AUC comparison between `rcf-rs` and three
published reference implementations:

- [`rrcf`](https://github.com/kLabUM/rrcf) 0.4.4 — Python + NumPy,
  the original open-source RCF port.
- [`scikit-learn` `IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
  — not RCF, but the canonical streaming-friendly tree-isolation
  baseline every comparison pins against.
- AWS's [`randomcutforest-java`](https://github.com/aws/random-cut-forest-by-aws)
  4.4.0 — JVM reference. See `README-aws-java.md`.

Scripts live outside the Rust crate by design — they pull in
Python / JVM toolchains, aren't CI-green, and produce numbers
that are only meaningful on the dev box they run on.

## Layout

- `gen_points.py` — deterministic CSV generator shared by every
  runner. First `n_normal` rows are clean (warm-up), last
  `n_outliers` rows are the anomaly probes.
- `bench_rrcf.py` — rrcf warm + `codisp` score loop.
- `bench_sklearn_iforest.py` — sklearn `IsolationForest` fit +
  `decision_function`.
- `java-driver/RcfBench.java` — AWS Java driver; see
  `README-aws-java.md` for the Maven Central jar path.

The rcf-rs side is an ordinary crate example:
`examples/external_bench_driver.rs` — invoked via `cargo run`.

## Running

From the `rcf-rs` crate root, JDK 26 + Python 3.13 on a machine
with `rrcf` and `scikit-learn` installed:

```bash
# Shared dataset: 10 000 pts, D=16, seed 2026, 1 % outliers.
python3 scripts/external-bench/gen_points.py \
    --n 10000 --dim 16 --seed 2026 > data.csv

# Python baselines.
pip install --user rrcf scikit-learn numpy
python3 scripts/external-bench/bench_rrcf.py \
    --input data.csv --trees 100 --sample 256
python3 scripts/external-bench/bench_sklearn_iforest.py \
    --input data.csv --trees 100 --train-frac 0.3

# rcf-rs.
cargo run --release --example external_bench_driver -- \
    data.csv 100 256

# AWS Java (see README-aws-java.md for the jar).
JAR=/tmp/aws-rcf/randomcutforest-core-4.4.0.jar
javac -cp "$JAR" scripts/external-bench/java-driver/RcfBench.java
java -cp "scripts/external-bench/java-driver:$JAR" RcfBench \
    data.csv 100 256
```

## Measured numbers (i7-1370P, synthetic 10k × D=16, 1 % outliers)

| Impl | Backend | Updates / s | Scores / s | AUC |
|---|---|---|---|---|
| `rcf-rs` 0.0.0-dev | native Rust, rayon-parallel | **32.5k** | **203k** | 1.000 |
| `randomcutforest-java` 4.4.0 | AWS reference, JVM 26 | 3.9k | 21k | 1.000 |
| `rrcf` 0.4.4 | Python + NumPy | 0.15k | 184k | 0.992 |
| `sklearn.IsolationForest` | NumPy + Cython, batch-fit | batch ≈ 48k | 234k | 1.000 |

- `rcf-rs` inserts ~10× faster than AWS Java and ~220× faster
  than `rrcf`.
- Score throughput is within 15 % across all four once each
  impl runs on its idiomatic fast path.
- `rcf-rs` score path is pinned by `score_many` (rayon);
  sklearn by NumPy/Cython SIMD; rrcf by single-process NumPy.

## Caveats

The Python scripts are best-effort one-shot harnesses — they
don't pin NumPy BLAS threads, don't warm up, don't stabilise
CPU frequency. Treat the numbers as "same hardware,
order-of-magnitude" only.

## Why the Python runners are single-process

- **`rrcf`** — `codisp` scoring mutates the tree on every probe
  (`insert_point(index=-1)` → `codisp(-1)` → `forget_point(-1)`),
  so threads collide on the shared `-1` slot and trip
  `AssertionError: index in leaves`. Multiprocessing fails at
  pickle time — tree objects hold module references that
  `pickle` rejects (`cannot pickle 'module' object`). The
  measured 184k scores/s is `rrcf`'s single-process ceiling;
  NumPy SIMD inside `codisp` already saturates.
- **sklearn `IsolationForest`** — `n_jobs=-1` was tested and
  regresses at 100 trees × 10k points (joblib task-spawn
  overhead exceeds the split-tree win). The default
  single-threaded BLAS SIMD path is the faster one for this
  batch size.
- **rcf-rs** uses rayon-parallel `score_many`; the table above
  compares each impl on its respective maximum-throughput
  entry point.

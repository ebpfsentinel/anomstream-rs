# External bench — compare rcf-rs against Python / Java baselines

Reproducible speed + AUC comparison between `rcf-rs` and published
reference implementations:

- [`rrcf`](https://github.com/kLabUM/rrcf) — Python, NumPy-backed,
  the original open-source RCF port.
- [`scikit-learn` `IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
  — not RCF, but the canonical streaming-friendly tree-isolation
  baseline every comparison pins against.
- AWS's [`randomcutforest-java`](https://github.com/aws/random-cut-forest-by-aws)
  — JVM reference. Heavy setup (Maven + JDK); bench script is
  provided as an outline rather than a one-shot runner.

These scripts are **deliberately out of the Rust crate** — they pull
in Python / JVM toolchains, are not CI-green, and produce
measurements that are only meaningful on the dev box they run on.
The expected flow is "run once, paste the numbers into
`docs/performance.md`".

## Layout

- `gen_points.py` — emit a deterministic dataset of D-dim f64
  points (CSV on stdout) that every runner consumes — ensures
  identical inputs across implementations.
- `bench_rrcf.py` — warm `rrcf.RCTree` + score loop; reports
  `updates/s` and `scores/s`.
- `bench_sklearn_iforest.py` — `IsolationForest` fit + `decision_function`;
  streaming emulated via incremental fit.
- `run_rcf_rs.sh` — build the rcf-rs release binary from the
  `scripts/external-bench/rcf_rs_driver` example and time it on
  the same dataset.
- `README-aws-java.md` — hand-written outline for running the
  AWS Java port; too many moving parts for a self-contained
  script.

## Running

```bash
# Generate a shared dataset (10 000 points, D=16, seed 2026).
python3 gen_points.py --n 10000 --dim 16 --seed 2026 > data.csv

# Python baselines.
pip install --user rrcf scikit-learn numpy
python3 bench_rrcf.py --input data.csv --trees 100 --sample 256
python3 bench_sklearn_iforest.py --input data.csv --trees 100

# rcf-rs baseline.
./run_rcf_rs.sh data.csv 100 256
```

Report the resulting `µs / op` side-by-side in
`docs/performance.md § External baselines`.

## Interpretation

- **Update throughput**: expect `rcf-rs` > AWS Java > `rrcf` ≫
  sklearn on the same `(trees, samples, D)` config. Ballpark:
  rcf-rs in tens of µs, rrcf in hundreds of µs, sklearn refit
  cost hides the comparison.
- **Score throughput**: similar picture, sklearn shines on batch
  score because its trees are dense numpy arrays.
- **AUC**: all four should land within 1–2 % of each other on
  separable synthetic data — if rcf-rs drops materially, the
  score aggregation or cut sampling has regressed.

## Caveats

The Python scripts are best-effort one-shot harnesses, not
benchmark-grade — they do not pin NumPy BLAS threads, warm up,
or stabilise CPU frequency. Treat the numbers as "order of
magnitude, same hardware" only.

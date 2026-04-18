# Performance

Criterion benches (`cargo bench --features parallel`), times reported
as the mean point estimate on a 4-core box.

## Core ops

| Workload | `(trees, samples, D)` | Time |
|---|---|---|
| `forest_update` | `(50, 128, 16)` | 35.91 µs |
| `forest_update` | `(100, 256, 4)` | 31.89 µs |
| `forest_update` | `(100, 256, 16)` | 47.98 µs |
| `forest_update` | `(100, 256, 64)` | 104.93 µs |
| `forest_update` | `(200, 512, 16)` | 84.91 µs |
| `forest_score` | `(50, 128, 16)` | 26.60 µs |
| `forest_score` | `(100, 256, 4)` | 37.08 µs |
| `forest_score` | `(100, 256, 16)` | 38.88 µs |
| `forest_score` | `(100, 256, 64)` | 46.62 µs |
| `forest_score` | `(200, 512, 16)` | 67.05 µs |
| `forest_attribution` | `(100, 256, 4)` | 72.21 µs |
| `forest_attribution` | `(100, 256, 16)` | 131.26 µs |
| `forest_attribution` | `(100, 256, 64)` | 150.39 µs |

At `(100, 256, 16)`: ~21k inserts/s and ~26k scores/s
single-thread-equivalent.

## Tuning sweep at `D = 16`

The `forest_tuning_dim16` bench group sweeps `(num_trees, sample_size)`:

| `(num_trees, sample_size)` | `update` | `score` |
|---|---|---|
| `(50, 64)` | 32.44 µs | 27.71 µs |
| `(50, 128)` | 35.98 µs | 27.97 µs |
| `(50, 256)` | 43.30 µs | 30.41 µs |
| `(100, 64)` | 36.85 µs | 35.13 µs |
| `(100, 128)` | 41.78 µs | 37.41 µs |
| `(100, 256)` | 50.75 µs | 37.61 µs |

## Bulk / early-termination speedups

Illustrative (not criterion-measured) from the example binaries:

| Scenario | Baseline | Optimised | Speedup |
|---|---|---|---|
| `score_many` vs serial for-loop, 4096 probes | 141 ms | 18.6 ms | 7.6× |
| `score_early_term` vs `score`, baseline-heavy | 43 ms | 14 ms | 3.1× |

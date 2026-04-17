# rcf-rs

Pure Rust Random Cut Forest for streaming anomaly detection.

`rcf-rs` implements the Random Cut Forest algorithm from Guha et al.
(ICML 2016) and is conformant with the
[AWS SageMaker RCF specification](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html):
reservoir sampling without replacement, random cuts weighted by per-dimension
range, anomaly score averaged across trees, hyperparameter bounds matching
the AWS reference (`feature_dim`, `num_trees`, `num_samples_per_tree`).

> **Status**: under active development — APIs are unstable until v0.1.0.

## Quickstart

```rust,ignore
use rcf_rs::{ForestBuilder, AnomalyScore};

// Build a forest with the AWS-default hyperparameters. Per-point
// dimensionality is pinned at the type level via the const-generic
// `D` parameter — `4` here.
let mut forest = ForestBuilder::<4>::new()
    .num_trees(100)
    .sample_size(256)
    .seed(42)
    .build()?;

// Stream points through the forest.
for point in stream_of_points {
    forest.update(point)?;
    let score: AnomalyScore = forest.score(&point)?;
    if f64::from(score) > 1.5 {
        println!("anomaly: score={score}");
    }
}
# Ok::<(), rcf_rs::RcfError>(())
```

## Algorithm

`rcf-rs` follows the original paper:

> Sudipto Guha, Nina Mishra, Gourav Roy, Okke Schrijvers.
> "Robust Random Cut Forest Based Anomaly Detection on Streams."
> *International Conference on Machine Learning*, pp. 2712–2721. 2016.

Reservoir sampling without replacement is from:

> Byung-Hoon Park, George Ostrouchov, Nagiza F. Samatova, Al Geist.
> "Reservoir-based random sampling with replacement from data stream."
> *SIAM International Conference on Data Mining*, pp. 492–496. 2004.

## AWS SageMaker conformance

| AWS specification | `rcf-rs` mapping |
|---|---|
| `feature_dim ∈ [1, 10000]` | const-generic `D`, validated by `ForestBuilder::build` |
| `num_trees ∈ [50, 1000]`, default `100` | enforced by `ForestBuilder` |
| `num_samples_per_tree ∈ [1, 2048]`, default `256` | enforced by `ForestBuilder` |
| Reservoir sampling without replacement | `sampler::ReservoirSampler` |
| Score = average across trees | `forest::RandomCutForest::score` |
| Anomaly threshold `≥ 3σ` from mean | caller responsibility |

Out of scope for v0.1.0:

- Shingling for 1D time series (consumer can pre-shingle the input)
- AWS `eval_metrics` (accuracy / precision-recall) — caller owns labels

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `std` | ✅ | Standard library support (future `no_std` planned) |
| `parallel` | ❌ | Per-tree parallel insert/score/attribution via `rayon` |
| `serde` | ❌ | Forest state serialisation |
| `bincode` | ❌ | Versioned binary persistence helpers (implies `serde`) |
| `serde_json` | ❌ | JSON helpers (implies `serde`) |

### `parallel` and the dedicated thread pool

Enable the `parallel` feature to run the per-tree work across rayon
workers. By default the global rayon pool is used; pin a dedicated
pool (and isolate this forest from the rest of the application's
rayon workload) via `ForestBuilder::num_threads`:

```rust,ignore
let forest = ForestBuilder::<16>::new()
    .num_trees(100)
    .sample_size(256)
    .num_threads(4)              // dedicated 4-worker pool
    .build()?;
```

`num_threads` is only honoured with `--features parallel`; without it
the field is recorded in the config but ignored at runtime.

## Performance

### Bench matrix (`forest_throughput`)

Latest run (`cargo bench --features parallel`), times reported as the
mean point estimate:

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

At `(100, 256, 16)` this is **~21k inserts/sec**, **~26k scores/sec**
single-thread-equivalent on a 4-core box, with both metrics scaling
sub-linearly down to single-core because each operation already
parallelises across trees.

A `forest_tuning_dim16` group sweeps `(num_trees, sample_size)` at the
AWS-default `D = 16` so callers can pick a precision/latency tradeoff:

| `(num_trees, sample_size)` | `update` | `score` |
|---|---|---|
| `(50, 64)` | 32.44 µs | 27.71 µs |
| `(50, 128)` | 35.98 µs | 27.97 µs |
| `(50, 256)` | 43.30 µs | 30.41 µs |
| `(100, 64)` | 36.85 µs | 35.13 µs |
| `(100, 128)` | 41.78 µs | 37.41 µs |
| `(100, 256)` | 50.75 µs | 37.61 µs |

## Minimum Supported Rust Version

`rcf-rs` requires Rust **1.93** or later, edition 2024.

## License

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE).

Contributions submitted to this repository are licensed under the same terms,
without any additional terms or conditions.

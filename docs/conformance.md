# AWS SageMaker Conformance

`rcf-rs` enforces the documented AWS `SageMaker` hyperparameter
bounds at build time. Beyond these invariants the library does not
aim for bit-exact parity with
[aws/random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws) —
feature evolution is driven by eBPFsentinel Enterprise needs.

| AWS specification | `rcf-rs` mapping |
|---|---|
| `feature_dim ∈ [1, 10000]` | const-generic `D`, validated by `ForestBuilder::build` |
| `num_trees ∈ [50, 1000]`, default `100` | enforced by `ForestBuilder` |
| `num_samples_per_tree ∈ [1, 2048]`, default `256` | enforced by `ForestBuilder` |
| `time_decay = 0.1 / sample_size` | resolved by `ForestBuilder`; pass `.time_decay(0.0)` to disable |
| Reservoir sampling without replacement | `sampler::ReservoirSampler` |
| Score = average across trees | `forest::RandomCutForest::score` |
| Anomaly threshold `≥ 3σ` from mean | caller responsibility (or `ThresholdedForest`) |

Deliberately absent from `rcf-rs` (out of scope for streaming
network anomaly detection):

- Density estimation (AWS `density()`)
- Imputation (AWS `imputevisitor`)
- Forecasting (AWS `RCFCaster`)
- Near-neighbor list (AWS `near_neighbor_list()`)
- Internal shingling + rotation
- GLAD locally adaptive variant
- Label / Attribute generics (`AugmentedRCF`)
- RCF 3.0 alternate score formula

## `parallel` and dedicated thread pool

Enable `parallel` to run per-tree work on rayon workers. Pin a
dedicated pool via `ForestBuilder::num_threads` to isolate the
forest from the rest of the application's rayon workload:

```rust,ignore
let forest = ForestBuilder::<16>::new()
    .num_trees(100)
    .sample_size(256)
    .num_threads(4)
    .build()?;
```

`num_threads` is only honoured with `--features parallel`.

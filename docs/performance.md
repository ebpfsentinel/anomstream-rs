# Performance

Criterion benches (`cargo bench`), wall-clock mean point estimate
on `x86_64` with `mimalloc` pinned globally. Two bench files:

- `benches/forest_throughput.rs` — core ops (insert, score,
  attribution) across the `(trees, samples, D)` matrix.
- `benches/extended.rs` — value-add APIs: bulk, early-term,
  forensic, tenant.

Quick run with smaller sample: `cargo bench -- --sample-size 10
--warm-up-time 1 --measurement-time 2`. Full run (default
criterion config): `cargo bench`.

## Reference hardware

The numbers below were captured on:

- **CPU**: Intel Core i7-1370P (13th gen) —
  14 cores / 20 threads, L3 = 24 MiB
- **Memory**: 32 GB DDR5
- **Kernel**: Linux 6.17
- **Allocator**: mimalloc 0.1 pinned globally in the bench harness
- **Compiler**: rustc 1.95 stable

Absolute numbers scale with CPU generation / frequency /
memory-bandwidth — the *ratios* between ops (parallel speedup,
early-term savings, tenant-count scaling) are the portable signal.
Re-run on target hardware before committing SLO budgets.

## Measurement methodology caveats

- **Cross-group variance**: do not compare absolute numbers across
  benches that run at different points of the `cargo bench` run.
  Each bench function mutates a persistent forest through its
  `b.iter()` body, and criterion chooses batch sizes based on
  per-op cost — so the reservoir state + per-iter overhead drift
  between groups. Trust *ratios* inside a group; suspect
  cross-group comparisons.
- **Parallel ceiling**: `score_many` plateaus at ~6× speedup on
  a 14-core host. Per-point work is memory-bandwidth-bound once
  the cache working set exceeds L3; more cores do not help past
  that point. Known target for future arena-layout work.
- **No external comparison yet**: no side-by-side vs AWS's
  `randomcutforest-java`, `rrcf` (Python), or Isolation Forest
  baselines. Tracked under future work.
- **No external-dataset detection-quality measurement here**: this
  file measures speed. Detection quality on public corpora
  (`NAB` / Yahoo S5 / Numenta) is not covered; see Future work.
  `tests/detection_quality.rs` does report **AUC**, **score
  separation ratio**, and **precision / recall at top-K** on
  synthetic ground-truth streams (cluster + outliers, transition
  anomalies) — regression-guards the core quality claim and pins
  AUC > 0.95 on separable data, > 0.90 on transition data.

## Core ops (`forest_throughput`)

| Workload | `(trees, samples, D)` | Time |
|---|---|---|
| `forest_update` | `(50, 128, 16)` | 23.59 µs |
| `forest_update` | `(100, 256, 4)` | 19.91 µs |
| `forest_update` | `(100, 256, 16)` | 32.36 µs |
| `forest_update` | `(100, 256, 64)` | 82.27 µs |
| `forest_update` | `(200, 512, 16)` | 68.67 µs |
| `forest_score` | `(50, 128, 16)` | 19.33 µs |
| `forest_score` | `(100, 256, 4)` | 23.69 µs |
| `forest_score` | `(100, 256, 16)` | 25.56 µs |
| `forest_score` | `(100, 256, 64)` | 34.22 µs |
| `forest_score` | `(200, 512, 16)` | 41.24 µs |
| `forest_attribution` | `(100, 256, 4)` | 35.17 µs |
| `forest_attribution` | `(100, 256, 16)` | 49.59 µs |
| `forest_attribution` | `(100, 256, 64)` | 98.78 µs |

At `(100, 256, 16)`: ~31k inserts/s and ~39k scores/s
single-thread-equivalent.

## Bulk batch scoring

`bulk_scoring` bench group, `D=16`, forest `(100, 256)`, batches
of random probes:

| Batch size | `score_many` (par) | Serial for-loop | Speedup |
|---|---|---|---|
| 64 | 439.64 µs | 2.19 ms | 5.0× |
| 512 | 3.17 ms | 19.48 ms | 6.1× |
| 4096 | 24.14 ms | 145.81 ms | 6.0× |

Speedup saturates around 6× as rayon task-scheduling amortises
then the ceiling is set by per-probe memory bandwidth.

## Early-termination scoring

`early_term` bench group, `D=16`, forest `(100, 256)`, single
probe:

| Path | Time |
|---|---|
| `score` (full parallel ensemble) | 36.21 µs |
| `score_early_term`, `threshold=0.02` (tight, rarely stops) | 58.73 µs |
| `score_early_term`, `threshold=0.20` (loose, stops ~20 trees) | 8.41 µs |

Tight threshold is slower than plain `score` because it walks
trees sequentially and rarely short-circuits — the parallel
ensemble wins when ambiguity forces a full traversal. Loose
threshold gives a **~4.3× speedup** on baseline-dominated traffic
where most points stop early.

## Forensic baseline

`forensic_baseline` bench group, `D` and `sample_size` swept:

| `(trees, samples, D)` | Time |
|---|---|
| `(100, 256, 4)` | 68.30 µs |
| `(100, 256, 16)` | 78.55 µs |
| `(100, 1024, 16)` | 315.07 µs |

Cost is dominated by the `O(live_points × D)` Welford sweep over
the union of tenant reservoirs. Quadrupling `sample_size` → ~4×
slower. Per-dim cost is marginal vs. the iteration overhead.

## Tenant pool at scale

`tenant_pool` bench group, each tenant `D=4` / `(50, 64)`, warmed
with 128 samples:

| N tenants | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
|---|---|---|---|
| 32 | 48.16 µs | 135.61 µs | 698.78 ns |
| 128 | 131.26 µs | 455.59 µs | 2.24 µs |
| 512 | 1.48 ms | 6.69 ms | 9.06 µs |

Observations:
- `similarity_matrix` is `O(N²)` on EMA-stat pairs, parallelised
  via rayon — N=32→512 gives ~31× (not 256×) because the parallel
  fan-out hides the quadratic cost up to core-count saturation.
- `score_across_tenants` is `O(N)` — one `score_only` per tenant,
  parallelised; N=32→512 gives ~49× for 16× more tenants (the
  extra ~3× beyond linear is rayon scheduling overhead at larger
  fan-outs).
- `most_similar_top5` is `O(N · log top_n)` via bounded
  `BinaryHeap`; N=32→512 gives ~13× for 16× more tenants —
  sub-linear because the fixed-size heap caps per-iter work.

## Future work

- **External baselines** — run the same matrix against AWS's
  `randomcutforest-java` on identical points, `rrcf` Python
  (Numpy-backed), and scikit-learn's `IsolationForest` to pin
  where rcf-rs sits in absolute throughput terms.
- **Detection-quality benchmarks** — integrate the Numenta NAB
  dataset, Yahoo S5, and Wikipedia pageviews; report per-corpus
  AUC + precision/recall at fixed operating points. A speed-only
  number says nothing about whether the forest actually catches
  the anomalies.
- **Arena-layout hot-path work** — per-tree node arenas are
  currently `Vec<Node>` dispatched via `NodeRef` indices. A
  DFS-packed layout (parent-before-children, `u16` deltas when
  the subtree fits) would halve the memory bandwidth required by
  `score` / `attribution` and lift the `~6×` parallel ceiling.
- **No-alloc scoring** — `score_many` builds one intermediate
  `Vec<AnomalyScore>`; a callback-based variant would cut the
  allocation on tight hot paths.
- **AVX-512 for `D=64`** — the bounding-box `range_sum` already
  uses `wide::f64x4`; `f64x8` on AVX-512 hosts would help at
  large `D`.

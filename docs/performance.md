# Performance

Criterion (`cargo bench`) — wall-clock mean point estimate,
`mimalloc` pinned globally. Two bench files:

- `benches/forest_throughput.rs` — core ops (insert, score,
  attribution).
- `benches/extended.rs` — bulk, early-term, forensic, tenant.

```bash
cargo bench                                            # full
cargo bench -- --sample-size 10 --measurement-time 2   # quick
```

## Reference hardware

| | |
|---|---|
| CPU | Intel Core i7-1370P (13th gen), 14C/20T, L3 24 MiB |
| Memory | 32 GB DDR5 |
| Kernel | Linux 6.17 |
| Allocator | mimalloc 0.1 (pinned in bench harness) |
| Compiler | rustc 1.95 stable |

Absolute values scale with CPU / memory bandwidth; *ratios*
(parallel speedup, early-term savings, tenant scaling) are the
portable signal.

## Caveats

- **Cross-group variance**: `b.iter()` mutates a persistent
  forest and criterion chooses batch sizes per-op, so reservoir
  state + per-iter overhead drift across bench groups. Trust
  ratios within a group, not absolute numbers across groups.
- **Parallel ceiling**: `score_many` plateaus at ~6× speedup on
  a 14-core host — memory-bandwidth-bound past L3 working set.

## Core ops

`(trees=100, sample=256, D=16)` after split-typed-arena refactor
(persistence v4):

| Workload | Time | Throughput |
|---|---|---|
| `forest_update` | ~23 µs | 43k/s |
| `forest_score` | ~23 µs | 43k/s |
| `forest_attribution` | ~31 µs | 32k/s |

Refactor delta vs pre-v4 at the same config: update −28 %,
score −10 %, attribution −37 %. Leaf-arena memory −90 %
(~320 B → ~40 B per slot).

Other `(trees, samples, D)` tuples: criterion HTML report
(`target/criterion/`).

## Bulk batch scoring

`D=16`, forest `(100, 256)`:

| Batch size | `score_many` (parallel) | Serial loop | Speedup |
|---|---|---|---|
| 64 | 440 µs | 2.19 ms | 5.0× |
| 512 | 3.17 ms | 19.5 ms | 6.1× |
| 4096 | 24.1 ms | 146 ms | 6.0× |

## Early-termination

`D=16`, forest `(100, 256)`, single probe:

| Path | Time |
|---|---|
| `score` (parallel ensemble) | 36 µs |
| `score_early_term` threshold=0.02 (tight) | 59 µs |
| `score_early_term` threshold=0.20 (loose, stops ~20 trees) | 8.4 µs |

Loose threshold → 4.3× speedup on baseline-dominated traffic;
tight threshold loses to parallel `score` (sequential walk
rarely short-circuits).

## Forensic baseline

`forensic_baseline`:

| `(trees, samples, D)` | Time |
|---|---|
| `(100, 256, 4)` | 68 µs |
| `(100, 256, 16)` | 79 µs |
| `(100, 1024, 16)` | 315 µs |

Cost ≈ `O(live_points × D)` Welford sweep — `sample_size` ×4
→ ×4 time, dim cost marginal.

## Tenant pool at scale

`tenant_pool`, each tenant `D=4` / `(50, 64)`, warmed 128 samples:

| N | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
|---|---|---|---|
| 32 | 48 µs | 136 µs | 0.70 µs |
| 128 | 131 µs | 456 µs | 2.2 µs |
| 512 | 1.48 ms | 6.69 ms | 9.1 µs |

Scaling `N=32→512` (16× tenants):
- `similarity_matrix` O(N²) parallelised: 31× (not 256× —
  rayon fan-out hides quadratic until core saturation).
- `score_across_tenants` O(N) parallelised: 49×.
- `most_similar_top5` O(N·log k) bounded heap: 13×.

## External baselines (synthetic)

Input: 10k points, `D=16`, 1 % outliers, 30 % warm / 70 % eval,
frozen baseline. Each impl on its idiomatic fast path
(rcf-rs rayon / rrcf single-process / sklearn NumPy-Cython SIMD
/ AWS Java cold JVM). **5-seed variance** (seeds 2026–2030),
mean ± stddev, coefficient of variation in parens.

| Impl | Backend | Updates/s | Scores/s | AUC |
|---|---|---|---|---|
| `rcf-rs` 0.0.0-dev, `score()` | Rust, rayon-parallel | **17 500 ± 1 240** (7 %) | 125 900 ± 1 840 (1.5 %) | 1.000 ± 0 |
| `rcf-rs` 0.0.0-dev, `score_codisp()` | Rust, serial per probe | — (uses insert/delete per probe) | ~1 140 (single seed) | 1.000 |
| `randomcutforest-java` 4.4.0 | JVM 26, cold | 2 090 ± 134 (6 %) | 8 870 ± 415 (5 %) | 1.000 ± 0 |
| `rrcf` 0.4.4 | Python + NumPy | 73 ± 3 (4 %) | 94 150 ± 4 840 (5 %) | 0.992 ± 0 |
| `sklearn.IsolationForest` | NumPy + Cython | batch-only | **136 300 ± 2 450** (2 %) | 1.000 ± 0 |

Ratios (mean/mean):

- **Updates**: rcf-rs is ~8.4× faster than AWS Java, ~240× faster
  than rrcf. CVs around 5-7 % on all impls; the ratios sit well
  outside the noise floor.
- **Scores (fast path)**: sklearn edges rcf-rs `score()` by 8 %
  (136k vs 126k) — real but small (stddevs combined ≈ 3k, so the
  10k delta is ~3σ significant). rrcf trails rcf-rs by ~25 %;
  AWS Java trails by ~14×.
- **Scores (codisp path)**: rcf-rs `score_codisp()` mutates the
  forest per probe (insert → walk leaf→root → delete) so it is
  two orders of magnitude slower than `score()` — ~1.1k probes/s
  at the same `(100, 256, D=16)` config. Matches AWS Java
  `getAnomalyScore` / rrcf `codisp()` semantic; use it for SOC
  triage / forensic replay, not the eBPF hot path.
- **AUC**: identical within measurement precision across every
  seed (0.992 for rrcf, 1.000 for the other three).

Noise sources documented: machine thermal state varies across
runs — single-seed numbers from an earlier cool-CPU session
landed at ~32k/203k for rcf-rs, dropping to ~17k/126k on this
run. The **ratios are portable, the absolute numbers aren't**.

Reproduce the sweep:

```bash
scripts/synthetic/variance_sweep.sh /tmp/aws-rcf/randomcutforest-core-4.4.0.jar
```

## Detection quality — NAB `realKnownCause`

Two scoring APIs, two use cases:

- **`RandomCutForest::score()`** — isolation-depth, never
  mutates the forest, rayon-parallel, eBPF-hot-path friendly.
  On NAB: **0.719** aggregate AUC after the lag=32 + zscore +
  smooth(0.02) pipeline.
- **`RandomCutForest::score_codisp()`** — probe-based (insert,
  walk leaf→root accumulating `max(sibling.mass /
  subtree.mass)`, remove). Matches rrcf / AWS Java scoring
  semantic. ~30× slower; intended for SOC triage / forensic
  replay. On NAB: **0.776** aggregate AUC, beats both rrcf
  and AWS Java.

Same embedding pipeline (32-lag → warm-phase z-score → EMA
α = 0.02), 15 % warm, 100 trees × 256 sample. `tests/nab.rs`
runs the 7-file corpus in parallel via rayon `par_iter` over
files — each file owns an independent forest. Full run
(both variants, parallel file iter) completes in ~12 s.

| File | rcf-rs `score()` | rcf-rs `score_codisp()` | rrcf | AWS Java |
|---|---|---|---|---|
| `ambient_temperature_system_failure` | **0.813** | **0.813** | 0.734 | 0.786 |
| `cpu_utilization_asg_misconfiguration` | 0.953 | **0.969** | 0.849 | 0.906 |
| `ec2_request_latency_system_failure` | 0.709 | 0.706 | 0.481 | 0.482 |
| `machine_temperature_system_failure` | 0.578 | 0.817 | 0.880 | **0.883** |
| `nyc_taxi` | **0.698** | 0.636 | 0.571 | 0.540 |
| `rogue_agent_key_hold` | 0.145 | 0.198 | 0.535 | **0.633** |
| `rogue_agent_key_updown` | **0.633** | 0.579 | 0.657 | 0.542 |
| **weighted aggregate** | 0.719 | **0.776** | 0.748 | 0.757 |

### Hyperparameter ablation

`examples/nab_ablation.rs` on the same corpus:

| Config | Aggregate AUC |
|---|---|
| baseline (lag=8, raw score) | 0.615 |
| lag=32 | 0.665 |
| lag=32 + diff | 0.640 |
| lag=32 + zscore | 0.683 |
| lag=32 + smooth(0.1) | 0.687 |
| lag=32 + zscore + smooth(0.05) | 0.718 |
| **lag=32 + zscore + smooth(0.02)** | **0.719** |
| lag=64 + zscore + smooth(0.05) | 0.672 |
| trcf-online D=32 | 0.320 |
| probe-score D=8 (naive hack) | 0.330 |
| **codisp D=32 + zscore + smooth(0.02)** | **0.776** |

- **Longer embedding** (lag=32) captures wider context — +0.050.
- **Z-score per warm-phase dim stddev** compensates NAB's scale
  variance (CPU % vs taxi counts vs temperatures) — RCF cuts are
  range-weighted, so un-normalised inputs let one dim dominate.
  +0.018 on top of lag=32.
- **EMA smoothing of raw scores** (α ≈ 0.02, half-life ~35 steps)
  cuts per-point noise without losing the wide-window shape.
  +0.036 on top of lag=32 + zscore.
- **Differencing** (first-diff of lag values) regresses — NAB's
  contextual signal lives in absolute values, not rate-of-change.
- **TRCF online** regresses catastrophically (0.72 → 0.32): the
  EMA threshold adapts UP during the multi-day anomaly windows
  and stops flagging them. Frozen baseline is the right paradigm
  for NAB's wide-window labels.
- **Probe-based naive hack** (`update_indexed → score → delete`)
  tanks AUC (0.330) — post-insert `score` ranks the freshly-
  inserted probe as seen. Proper codisp is `score_codisp()`:
  walks from inserted leaf → root accumulating
  `max(sibling.mass / subtree.mass)`, then deletes the probe.

### Two-API split

`score()` (isolation-depth, non-mutating, parallel) and
`score_codisp()` (probe-based, mutating, sequential per tree)
serve different use cases. `score()` is the eBPF hot-path
default; `score_codisp()` is for SOC triage / forensic replay
where the extra ~30× latency is acceptable for the +0.057 AUC
gain. On NAB, `score_codisp()` (0.776) leads both rrcf (0.748)
and AWS Java (0.757).

- `tests/detection_quality.rs` pins synthetic-corpus regression
  guards: AUC > 0.95 on separable clusters, > 0.90 on transition.
- `tests/nab.rs` pins NAB aggregate floor at 0.70.

Reproduce:

```bash
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
RCF_NAB_PATH=/opt/nab cargo test --test nab --all-features -- --ignored --nocapture
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
java -cp "scripts/nab:/tmp/aws-rcf/randomcutforest-core-4.4.0.jar" RcfBenchNab /opt/nab
```

## Detection quality — TSB-AD-M (multivariate)

TSB-AD-M (TheDatumOrg, 2024): 200 multivariate series across 16
source datasets, per-point binary labels, native multivariate
(no lag embedding). Pipeline: per-dim z-score on the upstream
`tr_<N>` train split, frozen-baseline scoring, EMA-smooth
α = 0.02. Forest `(100, 256)`, seed `2026`. Const-generic
whitelist `{2, 3, 7, 8, 9, 12, 16, 17, 18, 19, 25, 29, 31, 38, 51,
55, 66}` covers **192 / 200 files (96 %)**; the eight D=248 files
are skipped. `tests/tsb_ad_m.rs` runs the corpus in parallel via
rayon `par_iter` over files. Runtime on reference hardware:
~3 min for `score()`, ~6 min for `score_codisp()`.

Per-dataset ROC-AUC (weighted by positive count) against
`randomcutforest-java` 4.4.0 on the same corpus. rrcf 0.4.4 was
benched with the same protocol
(`scripts/tsb_ad/bench_rrcf_tsb_ad.py`, parallel across files)
but wall-time is prohibitive on the full corpus — ~3–4 h at
14 workers / `--max-eval 1500`. Numbers are left for the reader
to reproduce; the script is provided for reproducibility.

| Source dataset | Files | rcf-rs `score()` | rcf-rs `score_codisp()` | AWS Java `getAnomalyScore` |
|---|---|---|---|---|
| Genesis | 1 | 0.968 | **0.991** | 0.982 |
| SMAP | 27 | 0.803 | **0.823** | 0.805 |
| SMD | 22 | 0.618 | 0.760 | **0.806** |
| MSL | 16 | 0.705 | 0.746 | **0.762** |
| SVDB | 31 | 0.692 | 0.737 | **0.757** |
| LTDB | 5 | 0.601 | **0.755** | **0.755** |
| Exathlon | 27 | 0.491 | **0.894** | 0.865 |
| MITDB | 13 | 0.597 | **0.678** | 0.660 |
| PSM | 1 | 0.608 | 0.595 | **0.611** |
| CATSv2 | 6 | **0.580** | 0.547 | 0.547 |
| CreditCard | 1 | 0.589 | 0.679 | **0.693** |
| Daphnet | 1 | 0.309 | 0.885 | **0.944** |
| GECCO | 1 | 0.412 | 0.523 | **0.594** |
| GHL | 25 | 0.454 | **0.461** | 0.419 |
| OPPORTUNITY | 8 (skipped D=248) | — | — | 0.298 |
| SWaT | 2 | 0.282 | 0.825 | 0.825 |
| TAO | 13 | 0.451 | 0.453 | **0.471** |
| **aggregate weighted** | **192 / 200** | **0.583** | **0.768** | **0.753** |

- **rcf-rs `score()`** — isolation depth, rayon-parallel, full
  eval scan. Same fast API eBPFsentinel ships on the hot path.
  `tests/tsb_ad_m.rs::tsb_ad_m_aggregate_auc_above_floor` pins
  the aggregate floor at 0.55 — regression guard, not a quality
  claim.
- **rcf-rs `score_codisp()`** — probe-based codisp walk (leaf → root,
  `max(sibling.mass / subtree.mass)`), sequential per tree.
  Stride-subsampled to 50 000 eval rows per file (const
  `CODISP_MAX_EVAL`). Rayon-parallel across files so a
  14 C / 20 T host covers the 192-file corpus in ~5 min.
  Directly comparable to the AWS Java / rrcf semantic; leads
  aggregate **0.768** vs AWS Java 0.753.
  `tests/tsb_ad_m.rs::tsb_ad_m_codisp_aggregate_auc_above_floor`.
- **AWS Java `getAnomalyScore()`** — codisp-like, stride-
  subsampled to 50 000 eval rows per file (essentially full-scan
  for 95 % of the corpus). Covers all 200 files including the
  eight D=248 OPPORTUNITY series the const-generic whitelist
  skips.

Caveats:
- **Plain point-wise ROC-AUC**; the official TSB-AD leaderboard
  ranks on **VUS-PR** (Paparrizos et al. 2022) which integrates
  range-based precision / recall across a sliding window.
- **RCF is classical by design** — transformer-based SOTA
  (TimesNet, Anomaly Transformer) outscores every impl here on
  heavy-physics datasets (SWaT, Daphnet, GECCO) where the anomaly
  signature lives in higher-order cross-channel structure. RCF
  stays competitive on Genesis / SMAP / MSL / SVDB where per-dim
  statistical drift dominates — closer to eBPFsentinel's
  production feature mix (rate, ratio, entropy, cardinality).

Reproduce:

```bash
scripts/tsb_ad/fetch.sh /tmp/tsb-ad
RCF_TSB_AD_M_PATH=/tmp/tsb-ad/TSB-AD-M \
    cargo test --release --test tsb_ad_m --all-features -- --ignored --nocapture
python3 scripts/tsb_ad/bench_rrcf_tsb_ad.py \
    --dir /tmp/tsb-ad/TSB-AD-M --max-eval 1500 --workers "$(nproc)"
javac -cp /tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    scripts/tsb_ad/RcfBenchTsbAd.java
java -cp scripts/tsb_ad:/tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    RcfBenchTsbAd /tmp/tsb-ad/TSB-AD-M 50000
```

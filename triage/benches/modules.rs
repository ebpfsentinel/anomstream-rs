//! Criterion bench suite for the triage layer modules on top of
//! `anomstream-core`: LSH alert clustering, Platt calibration,
//! SAGE Shapley attribution.
//!
//! Run with `cargo bench -p anomstream-triage --bench modules`.

#![allow(clippy::cast_precision_loss)]

use anomstream_core::{AnomalyScore, DiVector, ForensicBaseline, ForestBuilder, RandomCutForest};
use anomstream_triage::{
    AlertRecord, LshAlertClusterer, PlattCalibrator, PlattFitConfig, SageEstimator,
};
use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// `LshAlertClusterer::hash_divector` + `observe` — quantise +
/// bucket lookup on a `DiVector<16>`.
fn bench_lsh_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_cluster");

    let mut di = DiVector::zeros(16);
    di.add_high(3, 2.5).expect("add_high");
    di.add_low(7, 1.5).expect("add_low");
    let rec: AlertRecord<u32, 16> = AlertRecord {
        version: anomstream_triage::ALERT_RECORD_VERSION,
        tenant: Some(1_u32),
        timestamp_ms: 0,
        point: [0.0; 16],
        score: AnomalyScore::new(1.0).expect("score"),
        grade: None,
        severity: None,
        attribution: di.clone(),
        baseline: ForensicBaseline::<16> {
            observed: [0.0; 16],
            expected: [0.0; 16],
            stddev: [0.0; 16],
            delta: [0.0; 16],
            zscore: [0.0; 16],
            live_points: 0,
        },
    };

    group.bench_function("hash_divector_d16", |b| {
        let clusterer = LshAlertClusterer::default_lsh();
        b.iter(|| {
            let h = clusterer.hash_divector(black_box(&di));
            black_box(h);
        });
    });

    group.bench_function("observe_d16", |b| {
        let mut clusterer = LshAlertClusterer::default_lsh();
        b.iter(|| {
            let (h, d) = clusterer.observe(black_box(&rec));
            black_box((h, d));
        });
    });

    group.finish();
}

/// `PlattCalibrator::fit` (Newton-Raphson) + `calibrate` (σ).
fn bench_calibrator(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibrator");

    group.bench_function("fit_2048_samples", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let data: Vec<(f64, bool)> = (0..2048)
            .map(|_| {
                let s: f64 = rng.random::<f64>() * 5.0;
                (s, s > 3.0)
            })
            .collect();
        b.iter(|| {
            let cal =
                PlattCalibrator::fit(black_box(&data), PlattFitConfig::default()).expect("fit");
            black_box(cal);
        });
    });

    group.bench_function("calibrate_single", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let data: Vec<(f64, bool)> = (0..1024)
            .map(|_| {
                let s: f64 = rng.random::<f64>() * 5.0;
                (s, s > 3.0)
            })
            .collect();
        let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).expect("fit");
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let s: f64 = rng_live.random::<f64>() * 5.0;
            let p = cal.calibrate(black_box(s));
            black_box(p);
        });
    });

    group.finish();
}

/// `SageEstimator::explain` — permutation Shapley with default
/// 64 permutations. Expensive: `K · D` forest scores per call.
fn bench_sage(c: &mut Criterion) {
    let mut group = c.benchmark_group("sage");
    group.sample_size(20);

    group.bench_function("explain_d16_k64", |b| {
        let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
            .num_trees(50)
            .sample_size(128)
            .seed(2026)
            .build()
            .expect("forest");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..512 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            forest.update(p).expect("update");
        }
        let sage: SageEstimator<16> = SageEstimator::default_anchor([0.0_f64; 16]).expect("sage");
        let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());
        b.iter(|| {
            let e = sage.explain(&forest, black_box(&probe)).expect("explain");
            black_box(e);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_lsh_cluster, bench_calibrator, bench_sage);
criterion_main!(benches);

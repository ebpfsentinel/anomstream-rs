//! `anomstream-triage` — SOC-opinionated triage layer on top of
//! [`anomstream-core`].
//!
//! Scaffold crate; populated in RCF-WS.3 with the six triage
//! modules currently living inside `anomstream-core`:
//!
//! - `calibrator` (Platt scaling, batch + online SGD)
//! - `sage` (SAGE Shapley attribution estimator)
//! - `alert_cluster` (cosine-similarity alert dedup)
//! - `lsh_cluster` (LSH-based alert dedup)
//! - `feedback` (SOC-label-driven score adjustment)
//! - `audit` (serialisable `AlertRecord<K, D>` envelope)
//!
//! Depends on [`anomstream-core`] for the output vocabulary
//! (`DiVector`, `AnomalyScore`, `AnomalyGrade`, `Severity`,
//! `MetricsSink`, `RandomCutForest`).

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

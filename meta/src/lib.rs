//! `anomstream` — streaming anomaly detection toolkit (facade).
//!
//! Umbrella crate that re-exports the three workspace members
//! under feature gates so consumers get a single import path
//! regardless of which layers they pull in:
//!
//! ```toml
//! [dependencies]
//! anomstream = { version = "0.2", default-features = false, features = ["core", "triage"] }
//! ```
//!
//! Feature matrix:
//!
//! | Feature | Pulls in | Purpose |
//! |---|---|---|
//! | `core` | [`anomstream-core`] | Detectors + streaming primitives + `MetricsSink` + `SeverityBands` |
//! | `triage` | [`anomstream-triage`] | Platt, SAGE, alert clustering, feedback, audit records |
//! | `hotpath` | [`anomstream-hotpath`] | eBPF-style ingress `UpdateSampler` / `PrefixRateCap` / `channel` |
//!
//! `triage` and `hotpath` both depend on `core`; enabling them
//! implies `core`. Default feature set enables all three plus
//! `std` / `parallel` / `serde` / `postcard` passthroughs.
//!
//! Populated fully in RCF-WS.5 with the per-member re-exports.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "core")]
pub use anomstream_core as core;

#[cfg(feature = "triage")]
pub use anomstream_triage as triage;

#[cfg(feature = "hotpath")]
pub use anomstream_hotpath as hotpath;

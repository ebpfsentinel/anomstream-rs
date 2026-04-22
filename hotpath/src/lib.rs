//! `anomstream-hotpath` — high-cadence ingress primitives on top
//! of [`anomstream-core`].
//!
//! Scaffold crate; populated in RCF-WS.4 with the three hot-path
//! ingress primitives currently living inside `anomstream-core`:
//!
//! - `UpdateSampler` — stride + hashed + keyed sampling gate
//! - `PrefixRateCap` — 256-bucket atomic rate sketch
//! - `channel<D>` — bounded MPSC for classifier/updater split
//!
//! Opt into this crate only if a deployment needs MPSC-backed
//! pre-forest sampling + per-prefix rate capping — typical
//! eBPF classifier/updater thread split. Most consumers should
//! talk to [`anomstream-core`] detectors directly.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

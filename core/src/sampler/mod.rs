//! Streaming reservoir sampling primitives.
//!
//! [`reservoir::ReservoirSampler`] holds at most `capacity` distinct
//! point indices, sampled without replacement (Park et al. 2004 per
//! the AWS `SageMaker` reference) with optional time decay (Guha et
//! al. 2016 §4) so the reservoir can bias toward recent points.

pub mod reservoir;

pub use reservoir::{ReservoirSampler, SamplerOp};

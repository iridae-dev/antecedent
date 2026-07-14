//! Probabilistic and structural causal models (DESIGN.md §15).
//!
//! Compiles DAGs to topological execution plans; sampling uses intervention
//! overlays rather than cloning models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod batch;
pub mod compile;
pub mod error;
pub mod overlay;

pub use batch::{
    MechanismWorkspace, NoiseBatch, NoiseBatchMut, ParentBatch, ValueBatch, ValueBatchMut,
};
pub use compile::{
    CompiledCausalModel, CompiledMechanismStore, InvertibleStructuralCausalModel, MechanismSlot,
    ModelOutputLayout, ParentGatherPlan, ProbabilisticCausalModel, StructuralCausalModel,
};
pub use error::ModelError;
pub use overlay::{InterventionOverlay, ModelView};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

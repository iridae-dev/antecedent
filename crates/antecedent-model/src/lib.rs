//! Probabilistic and structural causal models.
//!
//! Compiles DAGs to topological execution plans; sampling uses intervention
//! overlays rather than cloning models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod batch;
pub mod compile;
pub mod do_sampler;
pub mod error;
pub mod evaluate;
pub mod lgssm;
pub mod mechanism;
pub mod overlay;
pub mod registry;
pub mod sample;

pub use batch::{
    MechanismWorkspace, NoiseBatch, NoiseBatchMut, ParentBatch, ValueBatch, ValueBatchMut,
};
pub use compile::{
    CompiledCausalModel, CompiledMechanismStore, DynamicMechanism, InvertibleStructuralCausalModel,
    MechanismSlot, ModelOutputLayout, ParentGatherPlan, ProbabilisticCausalModel,
    StructuralCausalModel,
};
pub use do_sampler::{
    DoSampleResult, KdeDoSampler, McmcDoSampler, WeightingDoSampler, interventional_mean,
};
pub use error::ModelError;
pub use evaluate::{MechanismPredictiveCheck, ModelEvaluationReport, ModelEvaluator};
pub use lgssm::{
    infer_lgssm_innovations, kalman_filter, pack_innovations, rts_smooth, sample_lgssm_noise,
    unpack_innovations,
};
pub use mechanism::{
    NoiseInferenceMode, evaluate_batch_topo, evaluate_column, infer_noise_column,
    infer_noise_column_rng, log_prob_column, sample_column, sample_noise_batch,
    sample_noise_column,
};
pub use overlay::{InterventionOverlay, ModelView};
pub use registry::{
    MechanismAssignment, MechanismCandidate, MechanismFamily, MechanismRegistry, ModelCollection,
    SelectionPolicy,
};
pub use sample::{
    sample_conditional_interventional, sample_interventional, sample_observational,
    sample_posterior_predictive, sample_stochastic, sample_structural_with_overlay,
    sample_with_overlay, soft_to_slot,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

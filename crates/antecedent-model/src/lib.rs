//! Probabilistic and structural causal models.
//!
//! Compiles DAGs to topological execution plans; sampling uses intervention
//! overlays rather than cloning models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod basis;
pub mod batch;
pub mod compile;
pub mod compose;
pub mod cyclic_component;
pub mod discretized_scm;
pub mod discretized_scm_artifact;
pub mod do_sampler;
pub mod error;
pub mod evaluate;
pub mod falsify;
pub mod history_operators;
pub mod information_time;
pub mod lgssm;
pub mod linear_solution;
pub mod mechanism;
pub mod mechanism_transport;
pub mod model_collection;
pub mod noise_replay;
pub mod observation_query;
pub mod overlay;
pub mod program;
pub mod registry;
pub mod sample;
pub mod scc_compile;
pub mod temporal_rollout;
pub mod tree;

pub use basis::{BasisTerm, ParentBasis};
pub use batch::{
    MechanismWorkspace, NoiseBatch, NoiseBatchMut, ParentBatch, ValueBatch, ValueBatchMut,
};
pub use compile::{
    CompiledCausalModel, CompiledMechanismStore, DynamicMechanism, InvertibleStructuralCausalModel,
    MechanismSlot, ModelOutputLayout, ParentGatherPlan, ProbabilisticCausalModel,
    StructuralCausalModel,
};
pub use cyclic_component::{
    CYCLE_GATE_NEXT_ATTEMPT, ComponentDependency, ComponentOrder, CycleGateError, CyclicComponent,
    CyclicComponentError, FAMILY_BOUND_REFERENCE, ProvenanceCitation, ProvenanceOrder,
    SolveContractName, check_component_composition, check_provenance_chain,
};
pub use do_sampler::{
    DoSampleResult, KdeDoSampler, McmcDoSampler, WeightingDoSampler, interventional_mean,
};
pub use error::ModelError;
pub use evaluate::{ModelEvaluationReport, ModelEvaluator};
pub use falsify::{
    DEFAULT_MAX_CI_TESTS, GraphFalsificationConfig, GraphFalsificationOutcome,
    GraphFalsificationReport, GraphUninformativeReason, MAX_GRAPH_PERMUTATIONS,
    MarkovImplicationTest, falsify_graph,
};
pub use lgssm::{kalman_filter, rts_smooth};
pub use linear_solution::{
    Conditioning, ConsistencyWitness, FREE_BLOCK_EQUATION, FactorDescription, FamilyCertificate,
    HardClamp, IdentificationStatus, LinearSolutionError, LinearStructuralMap, NextAttempt,
    PhysicalStability, QueryBlock, SingularBlock, SolutionAssessment, SolverConvergence,
    Uniqueness, assess,
};
pub use mechanism::{
    NoiseInferenceMode, evaluate_batch_topo, evaluate_column, infer_noise_column,
    infer_noise_column_rng, log_prob_column, noise_inference_mode_of, reference_noise,
    sample_noise_column,
};
pub use model_collection::ModelCollection;
pub use noise_replay::{
    NOISE_REPLAY_STREAM_DOMAIN, NOISE_ROW_BLOCK, NoiseFix, NoiseOrigin, NoiseReplayer, NoiseSource,
    NoiseTable, ReplayOutcome,
};
pub use overlay::{InterventionOverlay, ModelView};
pub use program::{
    NodeProgram, PolicyRefusal, PolicyRoute, PolicySupport, ProgramKind, RouteDecision,
    apply_node_program, history_slot, policy_support,
};
pub use registry::{
    MechanismAssignment, MechanismCandidate, MechanismFamily, MechanismRegistry, MechanismTyping,
    NodeFamilyConstraint, SelectionOptions, SelectionPolicy, SelectionProvenance, SelectionRule,
    ValidationSplits,
};
pub use sample::{
    ConditionEvidence, PredictiveDraws, refuse_cross_family_soft,
    sample_conditional_interventional, sample_interventional, sample_observational,
    sample_observational_into, sample_posterior_predictive, sample_stochastic, sample_with_overlay,
    sample_with_overlay_into, soft_to_slot,
};
pub use sample::{NodeEvaluation, NodeNoise, evaluate_node};
pub use temporal_rollout::{TemporalNoise, TemporalPath, TemporalPlan, rollout_temporal};
pub use tree::TreeEnsemble;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

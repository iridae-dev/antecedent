//! The groups of declarations a [`super::StudyBuilder`] collects and the build pipeline carries.
//!
//! The builder, the build input and the pipeline's carried state hold the same
//! caller declarations. They are grouped once here by what they decide, so the
//! phases pass a group rather than re-listing its fields.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{PopulationRegistry, VariableId};
use antecedent_data::DiscoveryEstimationSplit;
use antecedent_estimate::{ContinuousResponseOptions, OverlapPolicy};
use antecedent_graph::TieredBackground;
use antecedent_validate::CustomEffectValidator;

use super::{InterferenceSpec, RdConfig, RefuteSuite, TransportTrialSpec};
use crate::estimator_spec::EstimatorSpec;
use crate::inference::InferenceMode;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::support::StructureSource;

use super::super::latency::{ComputeBudget, LatencyMode};
use super::super::stage::StageResultSink;

/// What the caller declared about the structure, the procedure and the population.
#[derive(Clone, Debug, Default)]
pub(super) struct Declarations {
    /// Caller-supplied mass over incomplete-temporal class members.
    pub(super) class_prior: Option<crate::ClassPrior>,
    /// Optional cap on `TemporalCpdag` / `TemporalPag` completion search.
    pub(super) max_completions: Option<usize>,
    /// Set by `graph`: `Accepted` vs `Explicit`. A posterior overrides it at build.
    pub(super) structure_source: Option<StructureSource>,
    pub(super) split: Option<DiscoveryEstimationSplit>,
    pub(super) identifier: Option<IdentifierId>,
    pub(super) estimator: Option<EstimatorId>,
    /// Caller-configured estimator (superset of `estimator`); `Some` only when
    /// `StudyBuilder::estimator` was called with a configured estimator rather
    /// than a bare [`EstimatorId`].
    pub(super) estimator_spec: Option<EstimatorSpec>,
    /// Numerical/inference options for response-family estimators.
    pub(super) response_options: Option<ContinuousResponseOptions>,
    pub(super) observation_options: antecedent_estimate::ObservationEstimatorOptions,
    /// Numerical controls on the Bayesian posterior draws behind the interval.
    pub(super) draw_controls: Option<crate::PosteriorDrawControls>,
    pub(super) observation_delayed_entry: Option<VariableId>,
    pub(super) rd: Option<RdConfig>,
    /// Optional override for propensity / AIPW overlap (clip/trim). `None` keeps estimator defaults.
    pub(super) overlap_policy: Option<OverlapPolicy>,
    /// Optional bindings for named predicates / custom target distributions.
    pub(super) population_registry: Option<PopulationRegistry>,
    /// Participant labels bound to the rows of the data (digests only).
    pub(super) participants: Option<antecedent_io::ParticipantBindingWire>,
    /// Declared handling of missing values; `None` leaves each route on its own default.
    pub(super) missing: Option<antecedent_data::MissingDeclaration>,
    /// Declared sampling design of the rows; `None` leaves each estimator on the design its
    /// row grouping implies.
    pub(super) sampling_design: Option<antecedent_core::DeclaredSamplingDesign>,
    /// Custom slow-path validators appended after the built-in refute suite.
    pub(super) custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
    /// Optional tier-rule background (fast-path generalized adjustment).
    pub(super) tiered: Option<TieredBackground>,
    /// Refused at build: coarsened continuous coordinate is not a point CDE.
    pub(super) continuous_cell: Option<(VariableId, Arc<[f64]>)>,
}

/// The inputs of a transport or an interference query.
#[derive(Clone, Debug, Default)]
pub(super) struct ExposureInputs {
    /// Selection-node targets for a transport query (empty = Direct).
    pub(super) selection_targets: Option<Arc<[VariableId]>>,
    /// Trial columns for a transport query.
    pub(super) transport_trial: Option<TransportTrialSpec>,
    /// Network + assignment for an interference query.
    pub(super) interference: Option<InterferenceSpec>,
}

/// How uncertainty is quantified and validated, with the explicit-versus-defaulted
/// markers that decide whether a latency tier may override a setting.
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the explicit-versus-defaulted markers and the joint-horizon retention flag are independent facts the builder records one for one"
)]
pub(super) struct InferenceSettings {
    pub(super) inference: InferenceMode,
    /// Whether Bayesian `n_draws` were set via [`ComputeBudget`] (mode draw map skipped).
    pub(super) n_draws_explicit: bool,
    pub(super) refute: RefuteSuite,
    /// Whether `refute` was set explicitly (wins over latency mode).
    pub(super) refute_explicit: bool,
    pub(super) bootstrap_replicates: u32,
    /// Whether `bootstrap_replicates` was set explicitly.
    pub(super) bootstrap_explicit: bool,
    /// Whether the temporal response retains its joint horizon replicates.
    pub(super) retain_joint_horizons: bool,
}

impl InferenceSettings {
    /// The settings of a builder nothing has configured: Frequentist, with the omitted
    /// validation suite and replicate count.
    pub(super) fn omitted(refute: RefuteSuite, bootstrap_replicates: u32) -> Self {
        Self {
            inference: InferenceMode::Frequentist,
            n_draws_explicit: false,
            refute,
            refute_explicit: false,
            bootstrap_replicates,
            bootstrap_explicit: false,
            retain_joint_horizons: false,
        }
    }
}

/// How and under what budget the study runs.
#[derive(Clone, Debug)]
pub(super) struct ExecutionSettings {
    /// Optional latency tier (maps to known-equivalent budgets unless overridden).
    pub(super) latency_mode: Option<LatencyMode>,
    /// Optional field-level compute budget overrides.
    pub(super) compute_budget: ComputeBudget,
    /// Optional progressive stage-result sink (Identify, Point, Uncertainty, Validate).
    pub(super) stage_sink: Option<Arc<dyn StageResultSink>>,
}

impl Default for ExecutionSettings {
    fn default() -> Self {
        Self { latency_mode: None, compute_budget: ComputeBudget::new(), stage_sink: None }
    }
}

//! Replayable artifact for an observation-conditioned scientific query.
//!
//! Format version 1, kind `observation_query_v1`. The artifact keeps every
//! premise the answer depends on: for a query on a composed causal model, the
//! canonical composition premises (the same wire the mechanism-composition
//! artifact uses); for a query on a declared finite or conjugate Bayes model,
//! the declared prior and likelihood inside the query itself. It also keeps
//! the query, the operation the query performs, how the value is to be read
//! (`exact_given_declared_model` or `point_only_sample_fitted`) and the stored
//! result as IEEE bit patterns.
//!
//! # Conditioning and intervention are different claims
//!
//! Every query variant names its operation, and the operation is part of the
//! identity: conditioning on `S = s`, intervening with `do(S = s)` and the
//! paired comparison of the two over one model have three different
//! identities and three differently-shaped results. A measured value is
//! carried with the measurement model that relates it to the construct; the
//! artifact has no way to say "this recorded value is the construct".
//!
//! A consumer trusts none of it. It recomputes the identity over the canonical
//! premises, query and reading, recompiles the composition under the coherence
//! checker, replays the query and accepts only the identical result bits.
//! What replay protects against: corruption, edits that are not resealed, and
//! resealed edits that change the answer. It does not protect against a
//! producer that seals wrong but coherent premises, or a bug shared by
//! producer and consumer; the independent oracles are in the test suites.
//!
//! No interval and no calibration is carried.
//!
//! # General queries
//!
//! Queries on continuous variables (conditioning a linear-Gaussian composition
//! on observed real values, with or without an intervention), observation after
//! an intervention on a finite model, and parameter uncertainty by retained-data
//! redraw are carried in format version 2 under the feature marker
//! `observation_query_general_v1`. A version-1 artifact keeps its bytes and its
//! identity; a version-2 artifact is refused by a version-1 reader before its
//! payload is interpreted. Real values are IEEE bit patterns, never strings.
//! Redrawn parameters are returned as aligned draws (one row is one joint
//! realization of every queried quantity) with the alignment identity of the
//! repo's distribution artifact, and carry no coverage statement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ExternalRefusal, InferenceBasis, MeasuredIndicator, MechanismOriginKind, ObservationContext,
    ObservedSlot, QuantityRole, ScientificQuantity, SlotProvenance, SlotValue,
    level_unreachable_refusal, measured_likelihood, mixed_conditioning_refusal,
    zero_probability_evidence_refusal,
};
use antecedent_core::measurement_model::{MeasurementKind, RepresentationVersion};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_model::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CompositionError, FiniteLaw,
    MAX_COMPOSED_MECHANISMS, MechanismOrigin,
};
use antecedent_model::observation_query::{
    BetaBernoulli, Direction, FiniteBayesModel, MAX_BAYES_HYPOTHESES, MAX_BAYES_OBSERVATION_LEVELS,
    MAX_BAYES_SEQUENCE, ObservationQueryError, UnreachableBound, condition, condition_gaussian,
    condition_on_measured, conditioning_and_intervention, intervene, intervened_law,
    observations_required_binary_signal, observations_required_conjugate, required_bayes_factor,
    threshold_crossing_observations,
};
use antecedent_model::observation_uncertainty::{
    GaussianSufficient, MAX_REDRAW_QUANTITIES, MAX_REDRAWS, MAX_RETAINED_REGRESSORS,
    ParameterDraws, RedrawPrior, RedrawTarget, RetainedMechanism, TableCountRow, TableCounts,
    redraw_conditional,
};
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use crate::measurement_model_wire::{MeasurementModelWire, model_from_wire};
use crate::mechanism_composition_artifact::{
    MECHANISM_COMPOSITION_MAX_INTERVENTIONS, MechanismCompositionPremisesWire,
    refusals_for_findings,
};
use crate::quantity_wire::DistributionMeaningWire;

/// The artifact format this reader writes and accepts.
pub const OBSERVATION_QUERY_ARTIFACT_VERSION: u32 = 1;
/// The feature marker (and artifact kind) of the accepted format.
pub const OBSERVATION_QUERY_ARTIFACT_FEATURE: &str = "observation_query_v1";
/// The format version that carries the general queries (continuous evidence,
/// observation after an intervention, retained-data redraw). A reader of
/// [`OBSERVATION_QUERY_ARTIFACT_VERSION`] alone refuses it before the payload
/// is interpreted.
pub const OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION: u32 = 2;
/// The feature marker of the general queries.
pub const OBSERVATION_QUERY_GENERAL_FEATURE: &str = "observation_query_general_v1";
/// Most retained mechanisms, and most parent-configuration rows of one retained
/// table, a redraw query may carry.
pub const OBSERVATION_QUERY_MAX_RETAINED_ROWS: usize = 4096;
/// The reading of the spread of redrawn parameters: a spread of exact posterior
/// draws with no coverage statement.
pub const PARAMETER_DRAWS_READING: &str = antecedent_contract::UNCALIBRATED_SPREAD_LABEL;
/// Most assignments (observed values, interventions, background values) one
/// query may carry.
pub const OBSERVATION_QUERY_MAX_ASSIGNMENTS: usize = MECHANISM_COMPOSITION_MAX_INTERVENTIONS;

/// Why an observation-query artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ObservationQueryArtifactError {
    /// The bytes do not decode as this format.
    #[error("observation-query artifact does not decode: {0}")]
    Decode(String),
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The identity does not match the stored premises, query and reading, or
    /// the caller's expected identity.
    #[error("identity mismatch")]
    IdentityMismatch,
    /// The artifact could not be encoded.
    #[error("observation-query artifact does not encode: {0}")]
    Encode(String),
    /// A stored field is not a well formed declaration.
    #[error("malformed premises: {0}")]
    Malformed(String),
    /// The stored mechanisms do not establish a coherent joint law.
    #[error("mechanisms do not compose: {} refusal(s)", .refusals.len())]
    Incoherent {
        /// Every defect, as structured refusals.
        refusals: Vec<ExternalRefusal>,
    },
    /// The request is refused: each refusal has a registered code, a
    /// namespaced `observation_query.*` detail and a remedy.
    #[error("observation query refused: {} refusal(s)", .refusals.len())]
    Refused {
        /// The refusals (never empty).
        refusals: Vec<ExternalRefusal>,
    },
    /// The replayed result differs from the stored one.
    #[error("result does not replay")]
    ResultMismatch,
    /// Cancellation was observed at a work boundary.
    #[error("observation query cancelled")]
    Cancelled,
    /// The replay itself failed.
    #[error("replay failed: {0}")]
    Replay(String),
}

impl From<IoError> for ObservationQueryArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn refused(refusal: ExternalRefusal) -> ObservationQueryArtifactError {
    ObservationQueryArtifactError::Refused { refusals: vec![refusal] }
}

fn registry_refusal(
    code: &'static str,
    detail: &str,
    offending: impl Into<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal::new(code, antecedent_contract::OBSERVATION_QUERY_STAGE, detail)
        .with_offending(offending)
        .with_remedy(remedy)
}

/// The structured refusal for a numerical-layer error.
///
/// [`ObservationQueryError::Cancelled`] is not a refusal of the request and
/// has none; callers handle it separately.
#[must_use]
pub fn query_error_refusals(error: &ObservationQueryError) -> Vec<ExternalRefusal> {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    let one = |r: ExternalRefusal| vec![r];
    match error {
        ObservationQueryError::UnknownVariable(name) => one(registry_refusal(
            antecedent_core::reason_code!("unknown_variable"),
            "observation_query.unknown_variable",
            name.clone(),
            "name a variable of the composed model",
        )),
        ObservationQueryError::UnknownLevel { variable, level } => one(registry_refusal(
            invalid,
            "observation_query.unknown_level",
            format!("{variable}={level}"),
            "name a declared level of the variable",
        )),
        ObservationQueryError::RepeatedVariable(name) => one(registry_refusal(
            invalid,
            "observation_query.repeated_slot",
            name.clone(),
            "assign each variable at most once",
        )),
        ObservationQueryError::OutcomeObserved(name) => one(registry_refusal(
            invalid,
            "observation_query.outcome_assigned",
            name.clone(),
            "ask about an outcome that is not itself conditioned on",
        )),
        ObservationQueryError::ConstructObserved(name) => one(registry_refusal(
            invalid,
            "observation_query.construct_observed_directly",
            name.clone(),
            "condition on the construct directly or on its measured indicator, not both",
        )),
        ObservationQueryError::ZeroProbabilityEvidence => one(zero_probability_evidence_refusal()),
        ObservationQueryError::ObservedAndIntervened(name) => one(mixed_conditioning_refusal(name)),
        ObservationQueryError::EvidenceSingular => one(registry_refusal(
            invalid,
            "observation_query.evidence_covariance_singular",
            "observed".to_owned(),
            "observe values whose joint covariance under the model is positive definite: drop an observed variable that is a deterministic function of the others, or give the mechanism a positive residual scale",
        )),
        ObservationQueryError::Invalid(what) => one(registry_refusal(
            invalid,
            "observation_query.invalid_declaration",
            (*what).to_owned(),
            "supply a well formed declaration: probabilities in range, distributions that sum to one, named levels",
        )),
        ObservationQueryError::LevelUnreachable { level, extremum, direction, bound } => {
            let bound = match bound {
                UnreachableBound::SignalDoesNotMove => "signal_does_not_move",
                UnreachableBound::ObservationBudget { .. } => "observation_budget",
                UnreachableBound::NoObservationCrosses => "no_observation_crosses",
                _ => "unreachable",
            };
            one(level_unreachable_refusal(*level, *extremum, direction.as_str(), bound))
        }
        ObservationQueryError::Composition(CompositionError::Incoherent { findings }) => {
            refusals_for_findings(findings)
        }
        ObservationQueryError::Composition(other) => one(registry_refusal(
            antecedent_core::reason_code!("route_not_supported"),
            "observation_query.request_unsupported",
            other.to_string(),
            "state the query within the bounded finite family",
        )),
        _ => one(registry_refusal(
            antecedent_core::reason_code!("route_not_supported"),
            "observation_query.request_unsupported",
            error.to_string(),
            "state the query within the bounded family",
        )),
    }
}

fn query_error(error: ObservationQueryError) -> ObservationQueryArtifactError {
    match error {
        ObservationQueryError::Cancelled => ObservationQueryArtifactError::Cancelled,
        ObservationQueryError::Composition(CompositionError::Incoherent { findings }) => {
            ObservationQueryArtifactError::Incoherent { refusals: refusals_for_findings(&findings) }
        }
        other => ObservationQueryArtifactError::Refused { refusals: query_error_refusals(&other) },
    }
}

fn composition_error(error: CompositionError) -> ObservationQueryArtifactError {
    match error {
        CompositionError::Incoherent { findings } => {
            ObservationQueryArtifactError::Incoherent { refusals: refusals_for_findings(&findings) }
        }
        CompositionError::Cancelled => ObservationQueryArtifactError::Cancelled,
        other => ObservationQueryArtifactError::Replay(other.to_string()),
    }
}

// ----------------------------------------------------------------- the wire

/// A recorded value of an indicator, with the measurement model that relates
/// the indicator to the construct it is offered for.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredSlotWire {
    /// The model variable that is the construct.
    pub construct: String,
    /// Stable identity of the recorded column.
    pub indicator_id: String,
    /// Kind of the recorded column (a `MeasurementKind` name).
    pub indicator_kind: String,
    /// The recorded level, in the construct's level coding.
    pub recorded_level: String,
    /// For a binary indicator, the positive level of the construct.
    pub positive_level: Option<String>,
    /// Coordinate system of the recorded column.
    pub representation_system: String,
    /// Version of that coordinate system.
    pub representation_version: u32,
    /// The measurement model.
    pub model: MeasurementModelWire,
}

/// The data one mechanism's numbers were fitted from, retained for redraw.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RetainedMechanismWire {
    /// Sufficient statistics of a regression on the mechanism's declared
    /// parents, intercept first.
    Gaussian {
        /// The variable the mechanism produces.
        variable: String,
        /// The regressors after the intercept, in the order of the statistics.
        parents: Vec<String>,
        /// Row-major Gram matrix `X'X`, as bit patterns.
        gram_bits: Vec<u64>,
        /// `X'y`, as bit patterns.
        xty_bits: Vec<u64>,
        /// `y'y`, as bits.
        yty_bits: u64,
        /// Rows the statistics were computed from.
        rows: u64,
    },
    /// Counts of each parent configuration of a finite mechanism.
    Table {
        /// The variable the mechanism produces.
        variable: String,
        /// `(parent configuration, (level, count))`, one entry per configuration.
        rows: Vec<(Vec<(String, String)>, Vec<(String, u64)>)>,
    },
}

impl RetainedMechanismWire {
    fn variable(&self) -> &str {
        match self {
            Self::Gaussian { variable, .. } | Self::Table { variable, .. } => variable,
        }
    }
}

/// The priors of a redraw, as bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedrawPriorWire {
    /// Dirichlet concentration added to every cell count.
    pub dirichlet_alpha_bits: u64,
    /// Prior precision of every regression coefficient relative to the error variance.
    pub coefficient_precision_bits: u64,
    /// Inverse-Gamma shape of the error variance.
    pub variance_shape_bits: u64,
    /// Inverse-Gamma scale of the error variance.
    pub variance_scale_bits: u64,
}

impl From<&RedrawPrior> for RedrawPriorWire {
    fn from(prior: &RedrawPrior) -> Self {
        Self {
            dirichlet_alpha_bits: prior.dirichlet_alpha.to_bits(),
            coefficient_precision_bits: prior.coefficient_precision.to_bits(),
            variance_shape_bits: prior.variance_shape.to_bits(),
            variance_scale_bits: prior.variance_scale.to_bits(),
        }
    }
}

impl RedrawPriorWire {
    fn to_prior(&self) -> RedrawPrior {
        RedrawPrior {
            dirichlet_alpha: f64::from_bits(self.dirichlet_alpha_bits),
            coefficient_precision: f64::from_bits(self.coefficient_precision_bits),
            variance_shape: f64::from_bits(self.variance_shape_bits),
            variance_scale: f64::from_bits(self.variance_scale_bits),
        }
    }
}

/// The conditional functional whose parameter uncertainty is redrawn.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RedrawTargetWire {
    /// Conditional mean and variance of a continuous outcome.
    Gaussian {
        /// Outcome variable.
        outcome: String,
        /// `(variable, value bits)` set by intervention.
        interventions: Vec<(String, u64)>,
        /// `(variable, value bits)` recorded.
        observed: Vec<(String, u64)>,
    },
    /// The conditional probability of every level of a finite outcome.
    Finite {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` set by intervention.
        interventions: Vec<(String, String)>,
        /// `(variable, level)` recorded.
        observed: Vec<(String, String)>,
    },
}

impl RedrawTargetWire {
    fn to_target(&self) -> RedrawTarget {
        match self {
            Self::Gaussian { outcome, interventions, observed } => RedrawTarget::Gaussian {
                outcome: outcome.clone(),
                interventions: real_pairs(interventions),
                observed: real_pairs(observed),
            },
            Self::Finite { outcome, interventions, observed } => RedrawTarget::Finite {
                outcome: outcome.clone(),
                interventions: interventions.clone(),
                observed: observed.clone(),
            },
        }
    }

    fn outcome(&self) -> &str {
        match self {
            Self::Gaussian { outcome, .. } | Self::Finite { outcome, .. } => outcome,
        }
    }

    fn has_interventions(&self) -> bool {
        match self {
            Self::Gaussian { interventions, .. } => !interventions.is_empty(),
            Self::Finite { interventions, .. } => !interventions.is_empty(),
        }
    }
}

fn real_pairs(pairs: &[(String, u64)]) -> Vec<(String, f64)> {
    pairs.iter().map(|(n, b)| (n.clone(), f64::from_bits(*b))).collect()
}

fn real_refs(pairs: &[(String, u64)]) -> Vec<(&str, f64)> {
    pairs.iter().map(|(n, b)| (n.as_str(), f64::from_bits(*b))).collect()
}

/// What is asked.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "wire shape: boxing would change the serialized form and the public variant shape"
)]
pub enum ObservationQueryWire {
    /// `P(outcome | observed values)` on a composed finite model, optionally
    /// with one measured signal inverted through its measurement model.
    ConditionalOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` recorded without error.
        observed: Vec<(String, String)>,
        /// One recorded indicator value, applied through its model.
        measured: Option<MeasuredSlotWire>,
    },
    /// `P(outcome | do(interventions))` on a composed finite model.
    InterventionalOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` set by intervention.
        interventions: Vec<(String, String)>,
    },
    /// Both laws of one assignment from one model, reported side by side.
    ConditioningVersusIntervention {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)`, read once as observed and once as set.
        assignment: Vec<(String, String)>,
    },
    /// Which observed levels of a signal move an outcome probability across a
    /// threshold.
    ThresholdCrossing {
        /// Outcome variable.
        outcome: String,
        /// Outcome level whose probability is tracked.
        outcome_level: String,
        /// The signal variable.
        signal: String,
        /// Recorded values held fixed.
        background: Vec<(String, String)>,
        /// Threshold, as a bit pattern.
        threshold_bits: u64,
        /// `raise` or `lower`.
        direction: String,
    },
    /// `P(outcome | do(interventions), observed values)` on a composed
    /// linear-Gaussian model: the interventional joint Gaussian conditioned on
    /// the observed real values by the Schur complement of the observed block.
    /// Either list may be empty. Format version 2.
    GaussianOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, value bits)` set by intervention.
        interventions: Vec<(String, u64)>,
        /// `(variable, value bits)` recorded.
        observed: Vec<(String, u64)>,
    },
    /// `P(outcome | do(interventions), observed values)` on a composed finite
    /// model: the truncated factorization conditioned on the recorded values,
    /// optionally with one measured signal inverted through its model. Format
    /// version 2.
    ConditionalInterventionalOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` set by intervention.
        interventions: Vec<(String, String)>,
        /// `(variable, level)` recorded without error.
        observed: Vec<(String, String)>,
        /// One recorded indicator value, applied through its model.
        measured: Option<MeasuredSlotWire>,
    },
    /// The parameter uncertainty of a conditional functional by retained-data
    /// redraw: aligned posterior draws of the queried quantities. Format
    /// version 2.
    ParameterRedraw {
        /// The functional redrawn.
        target: RedrawTargetWire,
        /// The data the redrawn mechanisms were fitted from.
        retained: Vec<RetainedMechanismWire>,
        /// The priors.
        prior: RedrawPriorWire,
        /// Number of posterior draws.
        draws: u32,
        /// Seed of the draw streams.
        seed: u64,
    },
    /// Posterior of a declared finite hypothesis space after observations.
    FinitePosterior {
        /// Hypothesis names.
        hypotheses: Vec<String>,
        /// Prior probabilities, as bit patterns.
        prior_bits: Vec<u64>,
        /// Observation level names.
        observation_levels: Vec<String>,
        /// Row-major `[hypothesis][observation]` likelihood, as bit patterns.
        likelihood_bits: Vec<u64>,
        /// The observations (exchangeable).
        observations: Vec<String>,
    },
    /// Posterior of a declared Beta-Bernoulli model.
    ConjugatePosterior {
        /// Prior first shape, as bits.
        alpha_bits: u64,
        /// Prior second shape, as bits.
        beta_bits: u64,
        /// Successes observed.
        successes: u64,
        /// Failures observed.
        failures: u64,
    },
    /// The Bayes factor that moves a probability to a level.
    RequiredBayesFactor {
        /// Starting probability, as bits.
        prior_bits: u64,
        /// Target level, as bits.
        level_bits: u64,
        /// `raise` or `lower`.
        direction: String,
    },
    /// Readings of a binary signal needed to move a probability to a level.
    RequiredObservationsBinarySignal {
        /// Starting probability of the hypothesis, as bits.
        prior_bits: u64,
        /// Target level, as bits.
        level_bits: u64,
        /// `raise` or `lower`.
        direction: String,
        /// `P(positive reading | hypothesis)`, as bits.
        rate_given_hypothesis_bits: u64,
        /// `P(positive reading | complement)`, as bits.
        rate_given_complement_bits: u64,
        /// Largest observation count the solve may return.
        max_observations: u64,
    },
    /// Observations needed to move a Beta-Bernoulli predictive probability to
    /// a level.
    RequiredObservationsConjugate {
        /// Prior first shape, as bits.
        alpha_bits: u64,
        /// Prior second shape, as bits.
        beta_bits: u64,
        /// Target level, as bits.
        level_bits: u64,
        /// `raise` or `lower`.
        direction: String,
        /// Largest observation count the solve may return.
        max_observations: u64,
    },
}

impl ObservationQueryWire {
    /// The operation this query performs; part of the identity.
    #[must_use]
    pub fn operation(&self) -> &'static str {
        match self {
            Self::GaussianOutcome { interventions, observed, .. } => {
                match (interventions.is_empty(), observed.is_empty()) {
                    (true, _) => "conditioning",
                    (false, true) => "intervention",
                    (false, false) => "conditional_intervention",
                }
            }
            Self::ConditionalInterventionalOutcome { .. } => "conditional_intervention",
            Self::ParameterRedraw { .. } => "parameter_redraw",
            Self::ConditionalOutcome { .. } => "conditioning",
            Self::InterventionalOutcome { .. } => "intervention",
            Self::ConditioningVersusIntervention { .. } => "conditioning_and_intervention",
            Self::ThresholdCrossing { .. } => "threshold_crossing",
            Self::FinitePosterior { .. } | Self::ConjugatePosterior { .. } => "posterior_update",
            Self::RequiredBayesFactor { .. }
            | Self::RequiredObservationsBinarySignal { .. }
            | Self::RequiredObservationsConjugate { .. } => "evidence_requirement",
        }
    }

    /// Whether the query runs on a composed causal model.
    #[must_use]
    pub const fn needs_composed_model(&self) -> bool {
        matches!(
            self,
            Self::ConditionalOutcome { .. }
                | Self::InterventionalOutcome { .. }
                | Self::ConditioningVersusIntervention { .. }
                | Self::ThresholdCrossing { .. }
                | Self::GaussianOutcome { .. }
                | Self::ConditionalInterventionalOutcome { .. }
                | Self::ParameterRedraw { .. }
        )
    }

    /// Whether the query is one of the general queries, carried in format
    /// version 2 under [`OBSERVATION_QUERY_GENERAL_FEATURE`].
    #[must_use]
    pub const fn is_general(&self) -> bool {
        matches!(
            self,
            Self::GaussianOutcome { .. }
                | Self::ConditionalInterventionalOutcome { .. }
                | Self::ParameterRedraw { .. }
        )
    }

    /// This query with every assignment list in canonical (variable, level)
    /// order, so two declarations that differ only in listing order have one
    /// wire form and one identity. Results do not depend on the order.
    #[must_use]
    pub fn canonical(mut self) -> Self {
        match &mut self {
            Self::ConditionalOutcome { observed, .. } => observed.sort(),
            Self::InterventionalOutcome { interventions, .. } => interventions.sort(),
            Self::ConditioningVersusIntervention { assignment, .. } => assignment.sort(),
            Self::ThresholdCrossing { background, .. } => background.sort(),
            Self::GaussianOutcome { interventions, observed, .. } => {
                interventions.sort();
                observed.sort();
            }
            Self::ConditionalInterventionalOutcome { interventions, observed, .. } => {
                interventions.sort();
                observed.sort();
            }
            Self::ParameterRedraw { target, retained, .. } => {
                match target {
                    RedrawTargetWire::Gaussian { interventions, observed, .. } => {
                        interventions.sort();
                        observed.sort();
                    }
                    RedrawTargetWire::Finite { interventions, observed, .. } => {
                        interventions.sort();
                        observed.sort();
                    }
                }
                for item in retained.iter_mut() {
                    if let RetainedMechanismWire::Table { rows, .. } = item {
                        for (configuration, counts) in rows.iter_mut() {
                            configuration.sort();
                            counts.sort();
                        }
                        rows.sort();
                    }
                }
                retained.sort_by(|a, b| a.variable().cmp(b.variable()));
            }
            _ => {}
        }
        self
    }
}

fn level_bits(levels: &[(String, f64)]) -> Vec<(String, u64)> {
    levels.iter().map(|(l, p)| (l.clone(), p.to_bits())).collect()
}

/// The stored result, as IEEE bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationResultWire {
    /// A conditional distribution.
    Conditional {
        /// Always `conditioning`.
        operation: String,
        /// `P(observed values)` (with the recorded indicator value, when one
        /// is applied), as bits.
        evidence_probability_bits: u64,
        /// `(level, probability bits)` of the outcome.
        levels: Vec<(String, u64)>,
        /// `(level, probability bits)` of the construct given a measured
        /// signal; `None` when no measured signal was applied.
        construct_posterior: Option<Vec<(String, u64)>>,
    },
    /// An interventional distribution.
    Interventional {
        /// Always `intervention`.
        operation: String,
        /// `(level, probability bits)` of the outcome.
        levels: Vec<(String, u64)>,
    },
    /// Both laws of one assignment.
    Paired {
        /// `P(assignment)` under the observational law, as bits.
        conditioning_evidence_bits: u64,
        /// Conditioning: `(level, probability bits)`.
        conditioning: Vec<(String, u64)>,
        /// Intervention: `(level, probability bits)`.
        intervention: Vec<(String, u64)>,
        /// Largest absolute difference over the outcome's levels, as bits.
        max_abs_difference_bits: u64,
    },
    /// The observed levels that cross a threshold.
    ThresholdCrossing {
        /// Probability of observing a crossing level, as bits.
        crossing_probability_bits: u64,
        /// `(signal level, P(level) bits, conditional bits, crosses)`.
        rows: Vec<(String, u64, u64, bool)>,
    },
    /// The Gaussian conditional law of one outcome.
    GaussianConditional {
        /// `conditioning`, `intervention` or `conditional_intervention`.
        operation: String,
        /// Conditional mean, as bits.
        mean_bits: u64,
        /// Conditional variance, as bits.
        variance_bits: u64,
        /// Log density of the observed values under their marginal law, as
        /// bits; `None` when nothing was observed.
        evidence_log_density_bits: Option<u64>,
    },
    /// Aligned posterior draws of the queried quantities.
    ParameterDraws {
        /// Functional ids of the quantities, in column order.
        quantities: Vec<String>,
        /// Number of draws.
        draws: u32,
        /// The quantities under the composition's own numbers, as bits.
        point_bits: Vec<u64>,
        /// Draw-major values, as bits.
        values_bits: Vec<u64>,
        /// Alignment identity of the draws (the distribution artifact's).
        alignment_identity: String,
        /// Always `uncalibrated_spread`: the draws are exact posterior draws and
        /// make no coverage statement.
        reading: String,
    },
    /// A finite posterior.
    FinitePosterior {
        /// `(hypothesis, posterior bits)`.
        posterior: Vec<(String, u64)>,
        /// Log evidence, as bits.
        log_evidence_bits: u64,
        /// Evidence probability, as bits.
        evidence_probability_bits: u64,
        /// `(observation level, predictive bits)`.
        predictive: Vec<(String, u64)>,
    },
    /// A conjugate posterior.
    ConjugatePosterior {
        /// Posterior first shape, as bits.
        alpha_bits: u64,
        /// Posterior second shape, as bits.
        beta_bits: u64,
        /// Predictive probability of the next success, as bits.
        predictive_success_bits: u64,
    },
    /// A required Bayes factor.
    RequiredBayesFactor {
        /// The factor, as bits.
        factor_bits: u64,
        /// No evidence is needed.
        already_satisfied: bool,
    },
    /// A required observation count.
    RequiredObservations {
        /// Smallest count that reaches the level.
        observations: u64,
        /// The probability at that count, as bits.
        value_bits: u64,
        /// No observation is needed.
        already_satisfied: bool,
        /// Bayes factor of one reading, as bits, for a likelihood-ratio route.
        bayes_factor_bits: Option<u64>,
    },
}

/// The versioned artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationQueryArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The composed model premises; present exactly when the query runs on a
    /// composed model.
    pub premises: Option<MechanismCompositionPremisesWire>,
    /// The question.
    pub query: ObservationQueryWire,
    /// How the value is to be read.
    pub inference_basis: String,
    /// The stored answer.
    pub result: ObservationResultWire,
    /// Digest of the premises, query and reading.
    pub identity: String,
}

/// A consumed artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedObservationQuery {
    /// The identity that was recomputed and verified.
    pub identity: String,
    /// The operation the query performed.
    pub operation: String,
    /// How the value is to be read.
    pub inference_basis: String,
    /// The replayed result, bit-identical to the stored one.
    pub result: ObservationResultWire,
}

// ------------------------------------------------------------------ identity

/// Identity of the premises, query and reading.
///
/// The digest is over the canonical wire encoding under a versioned tag. The
/// operation tag is bound in explicitly as well as through the query variant,
/// and the stored result is excluded: the result is an output replayed from
/// the premises, not a premise.
///
/// # Errors
/// Encoding failure.
pub fn observation_query_identity(
    premises: Option<&MechanismCompositionPremisesWire>,
    query: &ObservationQueryWire,
    inference_basis: &str,
) -> Result<String, IoError> {
    if query.is_general() {
        return crate::digest_hex(
            IdentityDomain::Program,
            &(
                OBSERVATION_QUERY_GENERAL_FEATURE,
                OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION,
                query.operation(),
                inference_basis,
                premises,
                query,
            ),
        );
    }
    crate::digest_hex(
        IdentityDomain::Program,
        &(
            "observation_query_v1",
            OBSERVATION_QUERY_ARTIFACT_VERSION,
            query.operation(),
            inference_basis,
            premises,
            query,
        ),
    )
}

// ----------------------------------------------------------------- execution

fn check_assignments(
    pairs: &[(String, String)],
    what: &'static str,
) -> Result<(), ObservationQueryArtifactError> {
    if pairs.len() > OBSERVATION_QUERY_MAX_ASSIGNMENTS {
        return Err(ObservationQueryArtifactError::LimitsExceeded(what));
    }
    Ok(())
}

fn check_query_bounds(query: &ObservationQueryWire) -> Result<(), ObservationQueryArtifactError> {
    match query {
        ObservationQueryWire::ConditionalOutcome { observed, .. } => {
            check_assignments(observed, "observed values")
        }
        ObservationQueryWire::InterventionalOutcome { interventions, .. } => {
            check_assignments(interventions, "interventions")
        }
        ObservationQueryWire::ConditioningVersusIntervention { assignment, .. } => {
            check_assignments(assignment, "assignment")
        }
        ObservationQueryWire::GaussianOutcome { interventions, observed, .. } => {
            if interventions.len() > OBSERVATION_QUERY_MAX_ASSIGNMENTS
                || observed.len() > OBSERVATION_QUERY_MAX_ASSIGNMENTS
            {
                return Err(ObservationQueryArtifactError::LimitsExceeded("assignments"));
            }
            Ok(())
        }
        ObservationQueryWire::ConditionalInterventionalOutcome {
            interventions, observed, ..
        } => {
            check_assignments(interventions, "interventions")?;
            check_assignments(observed, "observed values")
        }
        ObservationQueryWire::ParameterRedraw { target, retained, draws, .. } => {
            let (interventions, observed) = match target {
                RedrawTargetWire::Gaussian { interventions, observed, .. } => {
                    (interventions.len(), observed.len())
                }
                RedrawTargetWire::Finite { interventions, observed, .. } => {
                    (interventions.len(), observed.len())
                }
            };
            if interventions > OBSERVATION_QUERY_MAX_ASSIGNMENTS
                || observed > OBSERVATION_QUERY_MAX_ASSIGNMENTS
            {
                return Err(ObservationQueryArtifactError::LimitsExceeded("assignments"));
            }
            if retained.len() > MAX_COMPOSED_MECHANISMS || *draws > MAX_REDRAWS {
                return Err(ObservationQueryArtifactError::LimitsExceeded("redraw"));
            }
            for item in retained {
                match item {
                    RetainedMechanismWire::Gaussian { parents, gram_bits, xty_bits, .. } => {
                        if parents.len() >= MAX_RETAINED_REGRESSORS
                            || gram_bits.len() > MAX_RETAINED_REGRESSORS * MAX_RETAINED_REGRESSORS
                            || xty_bits.len() > MAX_RETAINED_REGRESSORS
                        {
                            return Err(ObservationQueryArtifactError::LimitsExceeded(
                                "retained regression",
                            ));
                        }
                    }
                    RetainedMechanismWire::Table { rows, .. } => {
                        if rows.len() > OBSERVATION_QUERY_MAX_RETAINED_ROWS
                            || rows.iter().any(|(config, counts)| {
                                config.len() > MAX_COMPOSED_MECHANISMS
                                    || counts.len()
                                        > antecedent_model::compose::MAX_LEVELS_PER_VARIABLE
                            })
                        {
                            return Err(ObservationQueryArtifactError::LimitsExceeded(
                                "retained table",
                            ));
                        }
                    }
                }
            }
            Ok(())
        }
        ObservationQueryWire::ThresholdCrossing { background, .. } => {
            check_assignments(background, "background values")
        }
        ObservationQueryWire::FinitePosterior {
            hypotheses,
            prior_bits,
            observation_levels,
            likelihood_bits,
            observations,
        } => {
            if hypotheses.len() > MAX_BAYES_HYPOTHESES
                || prior_bits.len() > MAX_BAYES_HYPOTHESES
                || observation_levels.len() > MAX_BAYES_OBSERVATION_LEVELS
                || likelihood_bits.len() > MAX_BAYES_HYPOTHESES * MAX_BAYES_OBSERVATION_LEVELS
            {
                return Err(ObservationQueryArtifactError::LimitsExceeded("bayes model"));
            }
            if observations.len() > MAX_BAYES_SEQUENCE {
                return Err(ObservationQueryArtifactError::LimitsExceeded("observations"));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_premises_bounds(
    premises: &MechanismCompositionPremisesWire,
) -> Result<(), ObservationQueryArtifactError> {
    if premises.mechanisms.is_empty() || premises.mechanisms.len() > MAX_COMPOSED_MECHANISMS {
        return Err(ObservationQueryArtifactError::LimitsExceeded("mechanisms"));
    }
    if premises.edges.len() > MAX_COMPOSED_MECHANISMS * MAX_COMPOSED_MECHANISMS {
        return Err(ObservationQueryArtifactError::LimitsExceeded("edges"));
    }
    Ok(())
}

fn refs(pairs: &[(String, String)]) -> Vec<(&str, &str)> {
    pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

fn direction(name: &str) -> Result<Direction, ObservationQueryArtifactError> {
    Direction::from_name(name)
        .ok_or_else(|| ObservationQueryArtifactError::Malformed(format!("direction `{name}`")))
}

fn bits(values: &[u64]) -> Vec<f64> {
    values.iter().map(|b| f64::from_bits(*b)).collect()
}

fn measured_slot_likelihood(
    measured: &MeasuredSlotWire,
    law: &FiniteLaw,
    spec: &ComposedModelSpec,
) -> Result<Vec<f64>, ObservationQueryArtifactError> {
    let Some(index) = law.variables().iter().position(|v| v == &measured.construct) else {
        return Err(refused(registry_refusal(
            antecedent_core::reason_code!("unknown_variable"),
            "observation_query.unknown_variable",
            measured.construct.clone(),
            "name a variable of the composed model",
        )));
    };
    let model = model_from_wire(&measured.model)
        .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?;
    let kind = MeasurementKind::from_name(&measured.indicator_kind).ok_or_else(|| {
        ObservationQueryArtifactError::Malformed(format!(
            "indicator kind `{}`",
            measured.indicator_kind
        ))
    })?;
    let slot = ObservedSlot {
        variable_id: measured.construct.clone(),
        level: measured.recorded_level.clone(),
        provenance: SlotProvenance::Recorded,
        value: SlotValue::Measured(MeasuredIndicator {
            indicator_id: measured.indicator_id.clone(),
            kind,
            model: Some(model),
            positive_level: measured.positive_level.clone(),
            representation: RepresentationVersion {
                coordinate_system_id: measured.representation_system.clone(),
                version: measured.representation_version,
            },
        }),
    };
    let context = ObservationContext {
        population_id: &spec.target.population,
        regime_id: &spec.target.regime,
        arm_id: None,
    };
    measured_likelihood(&slot, &law.levels()[index], &context).map_err(refused)
}

fn evaluate_composed(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    match query {
        ObservationQueryWire::GaussianOutcome { .. } => evaluate_gaussian(spec, query, ctx),
        ObservationQueryWire::ParameterRedraw { .. } => evaluate_redraw(spec, query, ctx),
        _ => evaluate_finite(spec, query, ctx),
    }
}

fn evaluate_gaussian(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let ObservationQueryWire::GaussianOutcome { outcome, interventions, observed } = query else {
        return Err(ObservationQueryArtifactError::Malformed(
            "a Gaussian evaluation was given another query".to_owned(),
        ));
    };
    let model = ComposedLinearGaussian::compile(spec).map_err(composition_error)?;
    let answer =
        condition_gaussian(&model, outcome, &real_refs(interventions), &real_refs(observed), ctx)
            .map_err(query_error)?;
    Ok(ObservationResultWire::GaussianConditional {
        operation: query.operation().to_owned(),
        mean_bits: answer.mean.to_bits(),
        variance_bits: answer.variance.to_bits(),
        evidence_log_density_bits: answer.evidence_log_density.map(f64::to_bits),
    })
}

fn retained_from_wire(item: &RetainedMechanismWire) -> RetainedMechanism {
    match item {
        RetainedMechanismWire::Gaussian {
            variable,
            parents,
            gram_bits,
            xty_bits,
            yty_bits,
            rows,
        } => RetainedMechanism::Gaussian(GaussianSufficient {
            variable: variable.clone(),
            parents: parents.clone(),
            gram: bits(gram_bits),
            xty: bits(xty_bits),
            yty: f64::from_bits(*yty_bits),
            rows: *rows,
        }),
        RetainedMechanismWire::Table { variable, rows } => RetainedMechanism::Table(TableCounts {
            variable: variable.clone(),
            rows: rows
                .iter()
                .map(|(parent_levels, counts)| TableCountRow {
                    parent_levels: parent_levels.clone(),
                    counts: counts.clone(),
                })
                .collect(),
        }),
    }
}

/// The quantity coordinates of redrawn draws: one per queried functional of the
/// outcome, at the composition's population, under the intervened regime, with
/// the recorded values as conditioning.
fn draw_quantities(
    spec: &ComposedModelSpec,
    target: &RedrawTargetWire,
    functionals: &[String],
) -> Vec<ScientificQuantity> {
    let outcome = target.outcome();
    let units = spec
        .mechanisms
        .iter()
        .find(|m| m.variable == outcome)
        .map(|m| m.output_units.clone())
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| "dimensionless".to_owned());
    let mut conditions: Vec<(String, String)> = match target {
        RedrawTargetWire::Gaussian { observed, .. } => {
            observed.iter().map(|(v, b)| (v.clone(), format!("{}", f64::from_bits(*b)))).collect()
        }
        RedrawTargetWire::Finite { observed, .. } => observed.clone(),
    };
    conditions.sort();
    let regime = if target.has_interventions() {
        let mut set: Vec<String> = match target {
            RedrawTargetWire::Gaussian { interventions, .. } => {
                interventions.iter().map(|(v, b)| format!("{v}={}", f64::from_bits(*b))).collect()
            }
            RedrawTargetWire::Finite { interventions, .. } => {
                interventions.iter().map(|(v, l)| format!("{v}={l}")).collect()
            }
        };
        set.sort();
        format!("do({})", set.join(","))
    } else {
        spec.target.regime.clone()
    };
    functionals
        .iter()
        .map(|functional| ScientificQuantity {
            variable_id: format!("{}:{outcome}", spec.target.population),
            variable_name: outcome.to_owned(),
            role: QuantityRole::Outcome,
            units: units.clone(),
            population_id: spec.target.population.clone(),
            regime_id: regime.clone(),
            horizon: 0,
            functional_id: functional.clone(),
            conditioning: conditions
                .iter()
                .map(|(variable, value)| antecedent_contract::QuantityCondition {
                    variable_id: variable.clone(),
                    value_id: value.clone(),
                })
                .collect(),
            transform_id: "identity".to_owned(),
        })
        .collect()
}

fn digest_error(error: IoError) -> ObservationQueryArtifactError {
    ObservationQueryArtifactError::Encode(error.to_string())
}

/// The aligned draws of a redraw query as the repo's distribution artifact.
///
/// Every row is one joint realization of the queried quantities (one parameter
/// draw produced them all), so the alignment is [`DrawAlignment::Joint`]. The
/// source is the identity of the query on these premises, the snapshot is the
/// digest of the retained data and priors, and the stream is the seed.
fn parameter_draws_distribution(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    draws: &ParameterDraws,
) -> Result<DistributionArtifact, ObservationQueryArtifactError> {
    let ObservationQueryWire::ParameterRedraw { target, retained, prior, draws: n, seed } = query
    else {
        return Err(ObservationQueryArtifactError::Malformed(
            "a distribution was requested for a query that redraws nothing".to_owned(),
        ));
    };
    let premises = MechanismCompositionPremisesWire::from_spec(spec);
    let basis = inference_basis_of(spec);
    let source =
        observation_query_identity(Some(&premises), query, basis.as_str()).map_err(digest_error)?;
    let snapshot = crate::digest_hex(
        IdentityDomain::Program,
        &("observation_query_retained_data_v1", retained, prior),
    )
    .map_err(digest_error)?;
    let contract =
        crate::digest_hex(IdentityDomain::Program, &("observation_query_premises_v1", &premises))
            .map_err(digest_error)?;
    let quantities = draw_quantities(spec, target, &draws.quantities);
    let semantic = if target.has_interventions() {
        DistributionMeaningWire::InterventionalPredictive
    } else {
        DistributionMeaningWire::PosteriorPredictive
    };
    let identity = DistributionIdentity::new(
        semantic,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: source,
            provider_id: "observation_query.parameter_redraw".to_owned(),
            rng_id: format!("splitmix64.derived_stream:seed={seed}:draws={n}"),
            snapshot_id: snapshot,
            causal_contract_id: contract,
        },
    )
    .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".to_owned(), "quantity".to_owned()],
            shape: [draws.draws, draws.width()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws.values.clone(),
    )
    .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))
}

fn evaluate_redraw(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let ObservationQueryWire::ParameterRedraw { target, retained, prior, draws, seed } = query
    else {
        return Err(ObservationQueryArtifactError::Malformed(
            "a redraw evaluation was given another query".to_owned(),
        ));
    };
    let retained: Vec<RetainedMechanism> = retained.iter().map(retained_from_wire).collect();
    let drawn = redraw_conditional(
        spec,
        &target.to_target(),
        &retained,
        &prior.to_prior(),
        *draws,
        *seed,
        ctx,
    )
    .map_err(query_error)?;
    if drawn.width() > MAX_REDRAW_QUANTITIES {
        return Err(ObservationQueryArtifactError::LimitsExceeded("quantities"));
    }
    let alignment_identity = parameter_draws_distribution(spec, query, &drawn)?
        .alignment_identity()
        .map_err(digest_error)?;
    Ok(ObservationResultWire::ParameterDraws {
        quantities: drawn.quantities.clone(),
        draws: *draws,
        point_bits: drawn.point.iter().map(|v| v.to_bits()).collect(),
        values_bits: drawn.values.iter().map(|v| v.to_bits()).collect(),
        alignment_identity,
        reading: PARAMETER_DRAWS_READING.to_owned(),
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per composed-model query, each a short call into the numerical layer"
)]
fn evaluate_finite(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let model = ComposedFinite::compile(spec).map_err(composition_error)?;
    match query {
        ObservationQueryWire::ConditionalInterventionalOutcome {
            outcome,
            interventions,
            observed,
            measured,
        } => {
            let observed = refs(observed);
            let interventions = refs(interventions);
            let law =
                intervened_law(&model, &interventions, &observed, ctx).map_err(query_error)?;
            if let Some(m) = measured {
                if interventions.iter().any(|(v, _)| *v == m.construct) {
                    return Err(refused(mixed_conditioning_refusal(&m.construct)));
                }
                let likelihood = measured_slot_likelihood(m, &law, spec)?;
                let answer =
                    condition_on_measured(&law, outcome, &m.construct, &observed, &likelihood, ctx)
                        .map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: query.operation().to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.outcome),
                    construct_posterior: Some(level_bits(&answer.construct_posterior)),
                })
            } else {
                let answer = condition(&law, outcome, &observed, ctx).map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: query.operation().to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.levels),
                    construct_posterior: None,
                })
            }
        }
        ObservationQueryWire::ConditionalOutcome { outcome, observed, measured } => {
            let law = model.law(&[], ctx).map_err(composition_error)?;
            let observed = refs(observed);
            if let Some(m) = measured {
                let likelihood = measured_slot_likelihood(m, &law, spec)?;
                let answer =
                    condition_on_measured(&law, outcome, &m.construct, &observed, &likelihood, ctx)
                        .map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: "conditioning".to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.outcome),
                    construct_posterior: Some(level_bits(&answer.construct_posterior)),
                })
            } else {
                let answer = condition(&law, outcome, &observed, ctx).map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: "conditioning".to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.levels),
                    construct_posterior: None,
                })
            }
        }
        ObservationQueryWire::InterventionalOutcome { outcome, interventions } => {
            let levels =
                intervene(&model, outcome, &refs(interventions), ctx).map_err(query_error)?;
            Ok(ObservationResultWire::Interventional {
                operation: "intervention".to_owned(),
                levels: level_bits(&levels),
            })
        }
        ObservationQueryWire::ConditioningVersusIntervention { outcome, assignment } => {
            let pair = conditioning_and_intervention(&model, outcome, &refs(assignment), ctx)
                .map_err(query_error)?;
            Ok(ObservationResultWire::Paired {
                conditioning_evidence_bits: pair.conditioning.evidence_probability.to_bits(),
                conditioning: level_bits(&pair.conditioning.levels),
                intervention: level_bits(&pair.intervention),
                max_abs_difference_bits: pair.max_abs_difference.to_bits(),
            })
        }
        ObservationQueryWire::ThresholdCrossing {
            outcome,
            outcome_level,
            signal,
            background,
            threshold_bits,
            direction: direction_name,
        } => {
            let law = model.law(&[], ctx).map_err(composition_error)?;
            let crossing = threshold_crossing_observations(
                &law,
                outcome,
                outcome_level,
                signal,
                &refs(background),
                f64::from_bits(*threshold_bits),
                direction(direction_name)?,
                ctx,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::ThresholdCrossing {
                crossing_probability_bits: crossing.crossing_probability.to_bits(),
                rows: crossing
                    .levels
                    .iter()
                    .map(|r| {
                        (
                            r.level.clone(),
                            r.probability.to_bits(),
                            r.conditional.to_bits(),
                            r.crosses,
                        )
                    })
                    .collect(),
            })
        }
        _ => Err(ObservationQueryArtifactError::Malformed(
            "a declared-model query was given composed premises".to_owned(),
        )),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per declared-model query, each a short call into the numerical layer"
)]
fn evaluate_declared(
    query: &ObservationQueryWire,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    match query {
        ObservationQueryWire::FinitePosterior {
            hypotheses,
            prior_bits,
            observation_levels,
            likelihood_bits,
            observations,
        } => {
            let model = FiniteBayesModel::new(
                hypotheses.clone(),
                bits(prior_bits),
                observation_levels.clone(),
                bits(likelihood_bits),
            )
            .map_err(query_error)?;
            let observations: Vec<&str> = observations.iter().map(String::as_str).collect();
            let posterior = model.posterior(&observations).map_err(query_error)?;
            Ok(ObservationResultWire::FinitePosterior {
                posterior: level_bits(&posterior.hypotheses),
                log_evidence_bits: posterior.log_evidence.to_bits(),
                evidence_probability_bits: posterior.evidence_probability.to_bits(),
                predictive: level_bits(&posterior.predictive),
            })
        }
        ObservationQueryWire::ConjugatePosterior { alpha_bits, beta_bits, successes, failures } => {
            let prior = BetaBernoulli::new(f64::from_bits(*alpha_bits), f64::from_bits(*beta_bits))
                .map_err(query_error)?;
            let posterior = prior.update(*successes, *failures).map_err(query_error)?;
            Ok(ObservationResultWire::ConjugatePosterior {
                alpha_bits: posterior.alpha().to_bits(),
                beta_bits: posterior.beta().to_bits(),
                predictive_success_bits: posterior.mean().to_bits(),
            })
        }
        ObservationQueryWire::RequiredBayesFactor {
            prior_bits,
            level_bits: target_bits,
            direction: direction_name,
        } => {
            let required = required_bayes_factor(
                f64::from_bits(*prior_bits),
                f64::from_bits(*target_bits),
                direction(direction_name)?,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::RequiredBayesFactor {
                factor_bits: required.factor.to_bits(),
                already_satisfied: required.already_satisfied,
            })
        }
        ObservationQueryWire::RequiredObservationsBinarySignal {
            prior_bits,
            level_bits: target_bits,
            direction: direction_name,
            rate_given_hypothesis_bits,
            rate_given_complement_bits,
            max_observations,
        } => {
            let required = observations_required_binary_signal(
                f64::from_bits(*prior_bits),
                f64::from_bits(*target_bits),
                direction(direction_name)?,
                f64::from_bits(*rate_given_hypothesis_bits),
                f64::from_bits(*rate_given_complement_bits),
                *max_observations,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::RequiredObservations {
                observations: required.observations,
                value_bits: required.value_at_requirement.to_bits(),
                already_satisfied: required.already_satisfied,
                bayes_factor_bits: required.bayes_factor_per_observation.map(f64::to_bits),
            })
        }
        ObservationQueryWire::RequiredObservationsConjugate {
            alpha_bits,
            beta_bits,
            level_bits: target_bits,
            direction: direction_name,
            max_observations,
        } => {
            let prior = BetaBernoulli::new(f64::from_bits(*alpha_bits), f64::from_bits(*beta_bits))
                .map_err(query_error)?;
            let required = observations_required_conjugate(
                &prior,
                f64::from_bits(*target_bits),
                direction(direction_name)?,
                *max_observations,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::RequiredObservations {
                observations: required.observations,
                value_bits: required.value_at_requirement.to_bits(),
                already_satisfied: required.already_satisfied,
                bayes_factor_bits: None,
            })
        }
        _ => Err(ObservationQueryArtifactError::Malformed(
            "a composed-model query was given no composed premises".to_owned(),
        )),
    }
}

/// How the value of a query on `spec` is to be read: declared only when every
/// mechanism was written down by the analyst.
#[must_use]
pub fn inference_basis_of(spec: &ComposedModelSpec) -> InferenceBasis {
    let origins: Vec<MechanismOriginKind> = spec
        .mechanisms
        .iter()
        .map(|m| match m.provenance.origin {
            MechanismOrigin::Declared => MechanismOriginKind::Declared,
            MechanismOrigin::Fitted => MechanismOriginKind::Fitted,
            MechanismOrigin::ExternalBinding => MechanismOriginKind::ExternalBinding,
        })
        .collect();
    InferenceBasis::from_origins(&origins)
}

fn basis_from_name(name: &str) -> Result<InferenceBasis, ObservationQueryArtifactError> {
    [InferenceBasis::DeclaredModel, InferenceBasis::SampleFittedPointOnly]
        .into_iter()
        .find(|b| b.as_str() == name)
        .ok_or_else(|| {
            ObservationQueryArtifactError::Malformed(format!("inference basis `{name}`"))
        })
}

/// Run `query`: against the composed model when it needs one, against its own
/// declared model otherwise.
///
/// # Errors
/// Incoherent mechanisms (structured refusals), a refused request (each with a
/// registered code and remedy, including an unreachable level and a measured
/// value without an applicable model), a shape that disagrees with the
/// premises, or cancellation.
pub fn evaluate_observation_query(
    spec: Option<&ComposedModelSpec>,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    check_query_bounds(query)?;
    if ctx.cancellation.is_cancelled() {
        return Err(ObservationQueryArtifactError::Cancelled);
    }
    match (query.needs_composed_model(), spec) {
        (true, Some(spec)) => evaluate_composed(spec, query, ctx),
        (false, None) => evaluate_declared(query),
        (true, None) => Err(ObservationQueryArtifactError::Malformed(
            "the query needs composed premises and none were supplied".to_owned(),
        )),
        (false, Some(_)) => Err(ObservationQueryArtifactError::Malformed(
            "a declared-model query was given composed premises".to_owned(),
        )),
    }
}

/// The format version and feature marker a query is carried under.
const fn format_of(query: &ObservationQueryWire) -> (u32, &'static str) {
    if query.is_general() {
        (OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION, OBSERVATION_QUERY_GENERAL_FEATURE)
    } else {
        (OBSERVATION_QUERY_ARTIFACT_VERSION, OBSERVATION_QUERY_ARTIFACT_FEATURE)
    }
}

impl ObservationQueryArtifactWire {
    /// The aligned draws of a parameter-redraw claim as the repo's distribution
    /// artifact, rebuilt from the stored result.
    ///
    /// # Errors
    /// A claim that is not a parameter redraw, premises that no longer compose,
    /// or draws that do not form a valid distribution.
    pub fn distribution(&self) -> Result<DistributionArtifact, ObservationQueryArtifactError> {
        let (
            Some(premises),
            ObservationResultWire::ParameterDraws {
                quantities,
                draws,
                values_bits,
                point_bits,
                ..
            },
        ) = (&self.premises, &self.result)
        else {
            return Err(ObservationQueryArtifactError::Malformed(
                "the claim carries no parameter draws".to_owned(),
            ));
        };
        let spec = premises
            .to_spec()
            .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?;
        parameter_draws_distribution(
            &spec,
            &self.query,
            &ParameterDraws {
                quantities: quantities.clone(),
                draws: *draws as usize,
                values: bits(values_bits),
                point: bits(point_bits),
            },
        )
    }

    /// Execute and seal an artifact for `query`.
    ///
    /// `spec` is the composed model for a query that runs on one and `None`
    /// otherwise. For a composed model the reading is derived from the
    /// mechanisms' origins and `declared_basis` is ignored; a declared-model
    /// query takes `declared_basis` from the caller, who states whether the
    /// prior and likelihood were written down or fitted to a sample.
    ///
    /// # Errors
    /// Incoherent mechanisms, a refused request, a failed execution or an
    /// encoding failure.
    pub fn seal(
        spec: Option<&ComposedModelSpec>,
        query: ObservationQueryWire,
        declared_basis: InferenceBasis,
        ctx: &ExecutionContext,
    ) -> Result<Self, ObservationQueryArtifactError> {
        let query = query.canonical();
        let premises = spec.map(MechanismCompositionPremisesWire::from_spec);
        if let Some(p) = &premises {
            check_premises_bounds(p)?;
        }
        let canonical = match &premises {
            Some(p) => Some(
                p.to_spec().map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?,
            ),
            None => None,
        };
        let result = evaluate_observation_query(canonical.as_ref(), &query, ctx)?;
        let basis = match &canonical {
            Some(spec) => inference_basis_of(spec),
            None => declared_basis,
        };
        let inference_basis = basis.as_str().to_owned();
        let identity = observation_query_identity(premises.as_ref(), &query, &inference_basis)
            .map_err(|e| ObservationQueryArtifactError::Encode(e.to_string()))?;
        let (version, feature) = format_of(&query);
        Ok(Self {
            version,
            required_features: vec![feature.to_owned()],
            premises,
            query,
            inference_basis,
            result,
            identity,
        })
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ObservationQueryArtifactError> {
        if let Some(p) = &self.premises {
            check_premises_bounds(p)?;
        }
        check_query_bounds(&self.query)?;
        crate::to_cbor(self).map_err(|e| ObservationQueryArtifactError::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature before the payload is
    /// interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, ObservationQueryArtifactError> {
        let wire: Self = crate::decode_with_version(bytes, |version: u32| {
            if version == OBSERVATION_QUERY_ARTIFACT_VERSION
                || version == OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION
            {
                Ok(())
            } else {
                Err(ObservationQueryArtifactError::UnsupportedVersion { version })
            }
        })?;
        wire.check_format()?;
        Ok(wire)
    }

    /// The version, the feature marker and the query variant must agree: a
    /// version-1 artifact holds only the version-1 queries and a version-2
    /// artifact only the general ones.
    fn check_format(&self) -> Result<(), ObservationQueryArtifactError> {
        let (version, feature) = format_of(&self.query);
        if self.version != version {
            return Err(ObservationQueryArtifactError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.required_features != [feature] {
            return Err(ObservationQueryArtifactError::UnsupportedSemantics("required features"));
        }
        Ok(())
    }

    /// Verify and replay.
    ///
    /// `expected_identity`, when given, is the identity the caller holds
    /// independently of the artifact.
    ///
    /// # Errors
    /// Any of the refusals of [`ObservationQueryArtifactError`].
    pub fn consume(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedObservationQuery, ObservationQueryArtifactError> {
        self.check_format()?;
        if let Some(p) = &self.premises {
            check_premises_bounds(p)?;
        }
        check_query_bounds(&self.query)?;
        let recomputed =
            observation_query_identity(self.premises.as_ref(), &self.query, &self.inference_basis)
                .map_err(|e| ObservationQueryArtifactError::Encode(e.to_string()))?;
        if recomputed != self.identity {
            return Err(ObservationQueryArtifactError::IdentityMismatch);
        }
        if let Some(expected) = expected_identity {
            if expected != recomputed {
                return Err(ObservationQueryArtifactError::IdentityMismatch);
            }
        }
        let stored_basis = basis_from_name(&self.inference_basis)?;
        let spec = match &self.premises {
            Some(p) => Some(
                p.to_spec().map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?,
            ),
            None => None,
        };
        if let Some(spec) = &spec {
            // The reading of a composed model is a function of its premises;
            // a stored reading that differs is an edit, not a declaration.
            if inference_basis_of(spec) != stored_basis {
                return Err(ObservationQueryArtifactError::IdentityMismatch);
            }
        }
        let result = evaluate_observation_query(spec.as_ref(), &self.query, ctx)?;
        if result != self.result {
            return Err(ObservationQueryArtifactError::ResultMismatch);
        }
        Ok(ConsumedObservationQuery {
            identity: recomputed,
            operation: self.query.operation().to_owned(),
            inference_basis: self.inference_basis.clone(),
            result,
        })
    }
}

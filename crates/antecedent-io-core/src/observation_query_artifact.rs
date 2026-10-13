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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ExternalRefusal, InferenceBasis, MeasuredIndicator, MechanismOriginKind, ObservationContext,
    ObservedSlot, SlotProvenance, SlotValue, level_unreachable_refusal, measured_likelihood,
    zero_probability_evidence_refusal,
};
use antecedent_core::measurement_model::{MeasurementKind, RepresentationVersion};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_model::compose::{
    ComposedFinite, ComposedModelSpec, CompositionError, FiniteLaw, MAX_COMPOSED_MECHANISMS,
    MechanismOrigin,
};
use antecedent_model::observation_query::{
    BetaBernoulli, Direction, FiniteBayesModel, MAX_BAYES_HYPOTHESES, MAX_BAYES_OBSERVATION_LEVELS,
    MAX_BAYES_SEQUENCE, ObservationQueryError, UnreachableBound, condition, condition_on_measured,
    conditioning_and_intervention, intervene, observations_required_binary_signal,
    observations_required_conjugate, required_bayes_factor, threshold_crossing_observations,
};
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::measurement_model_wire::{MeasurementModelWire, model_from_wire};
use crate::mechanism_composition_artifact::{
    MECHANISM_COMPOSITION_MAX_INTERVENTIONS, MechanismCompositionPremisesWire,
    refusals_for_findings,
};

/// The artifact format this reader writes and accepts.
pub const OBSERVATION_QUERY_ARTIFACT_VERSION: u32 = 1;
/// The feature marker (and artifact kind) of the accepted format.
pub const OBSERVATION_QUERY_ARTIFACT_FEATURE: &str = "observation_query_v1";
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

/// What is asked.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
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
    pub const fn operation(&self) -> &'static str {
        match self {
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

#[allow(
    clippy::too_many_lines,
    reason = "one arm per composed-model query, each a short call into the numerical layer"
)]
fn evaluate_composed(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let model = ComposedFinite::compile(spec).map_err(composition_error)?;
    match query {
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
            _ => MechanismOriginKind::ExternalBinding,
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

impl ObservationQueryArtifactWire {
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
        Ok(Self {
            version: OBSERVATION_QUERY_ARTIFACT_VERSION,
            required_features: vec![OBSERVATION_QUERY_ARTIFACT_FEATURE.to_owned()],
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
        let wire: Self = crate::decode_with_version(bytes, |version| {
            if version == OBSERVATION_QUERY_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(ObservationQueryArtifactError::UnsupportedVersion { version })
            }
        })?;
        if wire.required_features != [OBSERVATION_QUERY_ARTIFACT_FEATURE] {
            return Err(ObservationQueryArtifactError::UnsupportedSemantics("required features"));
        }
        Ok(wire)
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
        if self.version != OBSERVATION_QUERY_ARTIFACT_VERSION {
            return Err(ObservationQueryArtifactError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.required_features != [OBSERVATION_QUERY_ARTIFACT_FEATURE] {
            return Err(ObservationQueryArtifactError::UnsupportedSemantics("required features"));
        }
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

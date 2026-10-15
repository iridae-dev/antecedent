//! Observation-conditioned scientific queries: prepare, execute, export and
//! independently consume.
//!
//! Conditioning on an observation and intervening are different claims, also
//! when both are computed from the same causal model. This module routes a
//! request through the layers that own each rule and adds none of its own:
//!
//! 1. **Check** (`antecedent_contract::check_observation_slots`): a
//!    conditioning slot holds only recorded values; an assigned or
//!    hypothetical value, or an intervention supplied to a conditioning query,
//!    is refused as an intervention smuggled into a conditioning slot. A
//!    recorded value of an *indicator* offered about a construct is refused as
//!    `observation_query.measured_as_construct` unless a measurement model
//!    relating the two is declared and applies in the model's population,
//!    regime and arm.
//! 2. **Execute** (`antecedent_model::observation_query`) on the composed
//!    finite model of [`PreparedMechanismComposition`] (exact enumeration), or
//!    on a declared finite or conjugate Bayes model: outcome probability given
//!    a signal, posterior after new evidence, the observations that cross a
//!    threshold, and the evidence required to reach a level.
//! 3. **Export** seals the premises, the query, the operation and the reading
//!    into `observation_query_v1`; **consume** recomputes the identity,
//!    recompiles the premises and replays the query to identical result bits.
//!
//! # What a value means
//!
//! A value computed from mechanisms the analyst declared is exact given that
//! declaration (`exact_given_declared_model`). A value computed from a model
//! fitted to a sample, or supplied by a provider, is a point value
//! (`point_only_sample_fitted`): no interval and no calibration claim is
//! made, and [`ObservationClaim::interval`] and
//! [`ObservationClaim::calibrated_probability`] refuse with the stated remedy.
//!
//! # Measured signals
//!
//! Conditioning on a recorded indicator value `m` of a construct `C` inverts
//! the declared confusion through the model's own law:
//! `P(y | m) = sum_c P(y | c) P(c | m)` with `P(c | m)` proportional to
//! `P(m | c) P(c)`. The indicator is assumed conditionally independent of the
//! outcome and of every other observed variable given the construct
//! (non-differential error); error declared to differ by arm, population or
//! time is refused rather than pooled.
//!
//! # Lifecycle
//!
//! A claim holds the premises it was computed from. Refreshing the composition
//! with refitted mechanisms ([`PreparedMechanismComposition::refresh_mechanisms`])
//! returns a new composition; a query asked of it is a new claim with a new
//! identity, and the earlier claim is unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ExternalRefusal, InferenceBasis, MeasuredIndicator, ObservationOperation,
    ObservationSlotsRequest, ObservedSlot, SlotValue, calibration_refusal, check_observation_slots,
    interval_refusal,
};
use antecedent_core::ExecutionContext;
use antecedent_io::IoError;
use antecedent_io::measurement_model_wire::model_to_wire;
use antecedent_io::observation_query_artifact::{
    ObservationQueryArtifactError, ObservationQueryArtifactWire,
};

pub use super::mechanism_exports::model_crate::observation_query::Direction;
pub use antecedent_io::observation_query_artifact::{
    MeasuredSlotWire, ObservationQueryWire, ObservationResultWire,
};

use super::PreparedMechanismComposition;

/// The refusals of an observation query: each with a registered reason code, a
/// namespaced `observation_query.*` detail, the offending subject, expected
/// and supplied semantics and a remedy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationQueryRefusal {
    /// Every refusal, in the order the checkers found them. Never empty.
    pub refusals: Vec<ExternalRefusal>,
}

impl ObservationQueryRefusal {
    fn one(refusal: ExternalRefusal) -> Self {
        Self { refusals: vec![refusal] }
    }

    fn of(
        code: &'static str,
        stage: &'static str,
        detail: &str,
        subject: impl Into<String>,
    ) -> Self {
        Self::one(ExternalRefusal::new(code, stage, detail).with_offending(subject.into()))
    }

    /// The code of the first refusal.
    #[must_use]
    pub fn code(&self) -> &'static str {
        self.refusals.first().map_or(antecedent_core::reason_code!("invalid_argument"), |r| r.code)
    }

    /// The namespaced detail of every refusal.
    #[must_use]
    pub fn details(&self) -> Vec<&str> {
        self.refusals.iter().map(|r| r.detail.as_str()).collect()
    }
}

impl std::fmt::Display for ObservationQueryRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.refusals.first() {
            Some(first) => {
                write!(
                    f,
                    "{}{}: {}",
                    antecedent_core::reason_code::PREFIX,
                    first.code,
                    first.detail
                )?;
                if let Some(subject) = &first.offending {
                    write!(f, " ({subject})")?;
                }
                if self.refusals.len() > 1 {
                    write!(f, " and {} more", self.refusals.len() - 1)?;
                }
                Ok(())
            }
            None => f.write_str("observation query refused"),
        }
    }
}

impl std::error::Error for ObservationQueryRefusal {}

impl From<ObservationQueryRefusal> for IoError {
    fn from(refusal: ObservationQueryRefusal) -> Self {
        Self::Refused { code: refusal.code(), message: refusal.to_string() }
    }
}

impl From<ObservationQueryRefusal> for crate::CausalError {
    fn from(refusal: ObservationQueryRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

fn cancelled() -> ObservationQueryRefusal {
    ObservationQueryRefusal::of(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "observe",
        "observation_query.cancelled",
        "cancelled before a value was formed; this is not a verdict on the model",
    )
}

fn artifact_refusal(error: ObservationQueryArtifactError) -> ObservationQueryRefusal {
    match error {
        ObservationQueryArtifactError::Incoherent { refusals }
        | ObservationQueryArtifactError::Refused { refusals } => {
            ObservationQueryRefusal { refusals }
        }
        ObservationQueryArtifactError::Cancelled => cancelled(),
        ObservationQueryArtifactError::IdentityMismatch
        | ObservationQueryArtifactError::ResultMismatch => ObservationQueryRefusal::of(
            antecedent_core::reason_code!("invalid_argument"),
            "consume",
            "observation_query.artifact_refused",
            "the premises, query and reading do not hash to the stored or the retained identity, \
             or the stored result is not their replay",
        ),
        other => ObservationQueryRefusal::of(
            antecedent_core::reason_code!("invalid_argument"),
            "consume",
            "observation_query.artifact_invalid",
            other.to_string(),
        ),
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), ObservationQueryRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    Ok(())
}

fn levels_of(bits: &[(String, u64)]) -> Vec<(String, f64)> {
    bits.iter().map(|(l, p)| (l.clone(), f64::from_bits(*p))).collect()
}

/// What is asked of a composed causal model.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationRequest {
    /// Whether the request conditions or intervenes.
    pub operation: ObservationOperation,
    /// The outcome variable.
    pub outcome: String,
    /// Values placed in conditioning slots.
    pub conditioning: Vec<ObservedSlot>,
    /// `(variable, level)` assignments of `do()`.
    pub interventions: Vec<(String, String)>,
}

/// One answered observation query: the value, how it is to be read, and the
/// identity that binds the operation, the premises and the query.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationClaim {
    wire: ObservationQueryArtifactWire,
}

/// The two laws of a paired claim and their largest absolute difference.
pub type PairedLaws = (Vec<(String, f64)>, Vec<(String, f64)>, f64);

/// The rows of a threshold-crossing claim and the total crossing probability.
pub type ThresholdRows = (Vec<(String, f64, f64, bool)>, f64);

impl ObservationClaim {
    /// Identity of the claim (premises, query, operation and reading).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.wire.identity
    }

    /// The operation the claim performs: `conditioning`, `intervention`,
    /// `conditioning_and_intervention`, `threshold_crossing`,
    /// `posterior_update` or `evidence_requirement`.
    #[must_use]
    pub fn operation(&self) -> &'static str {
        self.wire.query.operation()
    }

    /// How the value is to be read: `exact_given_declared_model` or
    /// `point_only_sample_fitted`.
    #[must_use]
    pub fn inference_basis(&self) -> &str {
        &self.wire.inference_basis
    }

    /// The stored result, as bit patterns.
    #[must_use]
    pub const fn result(&self) -> &ObservationResultWire {
        &self.wire.result
    }

    /// The query that was answered.
    #[must_use]
    pub const fn query(&self) -> &ObservationQueryWire {
        &self.wire.query
    }

    /// Seal the claim as `observation_query_v1` bytes.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ObservationQueryRefusal> {
        self.wire.export().map_err(artifact_refusal)
    }

    /// An interval for any observation-conditioned value.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `observation_query.interval_route_closed`.
    pub fn interval(&self) -> Result<std::convert::Infallible, ObservationQueryRefusal> {
        Err(ObservationQueryRefusal::one(interval_refusal()))
    }

    /// A calibration claim for a posterior or decision probability.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with
    /// `observation_query.calibrated_probability_closed`: no measured
    /// whole-method record exists at any query's coordinate.
    pub fn calibrated_probability(
        &self,
    ) -> Result<std::convert::Infallible, ObservationQueryRefusal> {
        Err(ObservationQueryRefusal::one(calibration_refusal()))
    }

    /// `(level, probability)` of the outcome of a conditioning claim.
    #[must_use]
    pub fn conditional_levels(&self) -> Option<Vec<(String, f64)>> {
        match &self.wire.result {
            ObservationResultWire::Conditional { levels, .. } => Some(levels_of(levels)),
            _ => None,
        }
    }

    /// Probability of the evidence of a conditioning claim.
    #[must_use]
    pub fn evidence_probability(&self) -> Option<f64> {
        match &self.wire.result {
            ObservationResultWire::Conditional { evidence_probability_bits, .. }
            | ObservationResultWire::FinitePosterior { evidence_probability_bits, .. } => {
                Some(f64::from_bits(*evidence_probability_bits))
            }
            ObservationResultWire::Paired { conditioning_evidence_bits, .. } => {
                Some(f64::from_bits(*conditioning_evidence_bits))
            }
            _ => None,
        }
    }

    /// `(level, probability)` of the construct given a measured signal.
    #[must_use]
    pub fn construct_posterior(&self) -> Option<Vec<(String, f64)>> {
        match &self.wire.result {
            ObservationResultWire::Conditional { construct_posterior: Some(p), .. } => {
                Some(levels_of(p))
            }
            _ => None,
        }
    }

    /// `(level, probability)` of the outcome of an interventional claim.
    #[must_use]
    pub fn interventional_levels(&self) -> Option<Vec<(String, f64)>> {
        match &self.wire.result {
            ObservationResultWire::Interventional { levels, .. } => Some(levels_of(levels)),
            _ => None,
        }
    }

    /// The two laws of a paired claim: `(conditioning, intervention, largest
    /// absolute difference)`.
    #[must_use]
    pub fn paired(&self) -> Option<PairedLaws> {
        match &self.wire.result {
            ObservationResultWire::Paired {
                conditioning,
                intervention,
                max_abs_difference_bits,
                ..
            } => Some((
                levels_of(conditioning),
                levels_of(intervention),
                f64::from_bits(*max_abs_difference_bits),
            )),
            _ => None,
        }
    }

    /// The observations of a threshold-crossing claim:
    /// `(signal level, probability of observing it, conditional probability,
    /// crosses)`, and the total probability of a crossing observation.
    #[must_use]
    pub fn threshold_rows(&self) -> Option<ThresholdRows> {
        match &self.wire.result {
            ObservationResultWire::ThresholdCrossing { crossing_probability_bits, rows } => Some((
                rows.iter()
                    .map(|(l, p, c, x)| (l.clone(), f64::from_bits(*p), f64::from_bits(*c), *x))
                    .collect(),
                f64::from_bits(*crossing_probability_bits),
            )),
            _ => None,
        }
    }

    /// `(hypothesis, posterior probability)` of a finite-posterior claim.
    #[must_use]
    pub fn posterior(&self) -> Option<Vec<(String, f64)>> {
        match &self.wire.result {
            ObservationResultWire::FinitePosterior { posterior, .. } => Some(levels_of(posterior)),
            _ => None,
        }
    }

    /// `(alpha, beta, predictive success probability)` of a conjugate claim.
    #[must_use]
    pub fn conjugate_posterior(&self) -> Option<(f64, f64, f64)> {
        match &self.wire.result {
            ObservationResultWire::ConjugatePosterior {
                alpha_bits,
                beta_bits,
                predictive_success_bits,
            } => Some((
                f64::from_bits(*alpha_bits),
                f64::from_bits(*beta_bits),
                f64::from_bits(*predictive_success_bits),
            )),
            _ => None,
        }
    }

    /// The Bayes factor of a required-factor claim.
    #[must_use]
    pub fn required_factor(&self) -> Option<f64> {
        match &self.wire.result {
            ObservationResultWire::RequiredBayesFactor { factor_bits, .. } => {
                Some(f64::from_bits(*factor_bits))
            }
            _ => None,
        }
    }

    /// `(observations, probability at that count)` of a required-observations
    /// claim.
    #[must_use]
    pub fn required_observations(&self) -> Option<(u64, f64)> {
        match &self.wire.result {
            ObservationResultWire::RequiredObservations { observations, value_bits, .. } => {
                Some((*observations, f64::from_bits(*value_bits)))
            }
            _ => None,
        }
    }
}

fn measured_wire(
    slot: &ObservedSlot,
    indicator: &MeasuredIndicator,
) -> Result<MeasuredSlotWire, ObservationQueryRefusal> {
    let Some(model) = &indicator.model else {
        return Err(ObservationQueryRefusal::of(
            antecedent_core::reason_code!("proxy_offered_as_construct"),
            "observe",
            "observation_query.measured_as_construct",
            slot.variable_id.clone(),
        ));
    };
    Ok(MeasuredSlotWire {
        construct: slot.variable_id.clone(),
        indicator_id: indicator.indicator_id.clone(),
        indicator_kind: indicator.kind.as_str().to_owned(),
        recorded_level: slot.level.clone(),
        positive_level: indicator.positive_level.clone(),
        representation_system: indicator.representation.coordinate_system_id.clone(),
        representation_version: indicator.representation.version,
        model: model_to_wire(model),
    })
}

/// Seal a query that runs on a declared model (no composed premises).
///
/// Used by the posterior-update and evidence-requirement routes, which share
/// this module's claim, refusal and artifact types.
pub(crate) fn seal_declared_query(
    query: ObservationQueryWire,
    basis: InferenceBasis,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    seal(None, query, basis, ctx)
}

fn seal(
    spec: Option<&antecedent_model::compose::ComposedModelSpec>,
    query: ObservationQueryWire,
    basis: InferenceBasis,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    check_cancelled(ctx)?;
    let wire =
        ObservationQueryArtifactWire::seal(spec, query, basis, ctx).map_err(artifact_refusal)?;
    Ok(ObservationClaim { wire })
}

impl PreparedMechanismComposition {
    /// Answer one observation query on this composed finite model.
    ///
    /// The request is checked first: an assigned value in a conditioning slot,
    /// an intervention supplied to a conditioning query, observations mixed
    /// with an intervention, and a recorded indicator value without a
    /// measurement model are each refused with their own detail.
    ///
    /// # Errors
    /// Any refusal of the slot check; a linear-Gaussian composition (the
    /// queries are exact finite-law queries); an unknown variable or level;
    /// evidence of probability zero; incoherent mechanisms; cancellation.
    pub fn observe(
        &self,
        request: &ObservationRequest,
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        check_cancelled(ctx)?;
        check_observation_slots(&ObservationSlotsRequest {
            operation: request.operation,
            outcome_id: &request.outcome,
            conditioning: &request.conditioning,
            interventions: &request.interventions,
        })
        .map_err(ObservationQueryRefusal::one)?;
        let query = if request.operation == ObservationOperation::Intervention {
            ObservationQueryWire::InterventionalOutcome {
                outcome: request.outcome.clone(),
                interventions: request.interventions.clone(),
            }
        } else {
            {
                let mut observed = Vec::new();
                let mut measured = None;
                for slot in &request.conditioning {
                    match &slot.value {
                        SlotValue::Direct => {
                            observed.push((slot.variable_id.clone(), slot.level.clone()));
                        }
                        SlotValue::Measured(indicator) => {
                            measured = Some(measured_wire(slot, indicator)?);
                        }
                    }
                }
                ObservationQueryWire::ConditionalOutcome {
                    outcome: request.outcome.clone(),
                    observed,
                    measured,
                }
            }
        };
        seal(Some(self.spec()), query, InferenceBasis::DeclaredModel, ctx)
    }

    /// `P(outcome | observed values)`: the conditioning claim.
    ///
    /// # Errors
    /// As [`Self::observe`].
    pub fn condition_on(
        &self,
        outcome: &str,
        conditioning: &[ObservedSlot],
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        self.observe(
            &ObservationRequest {
                operation: ObservationOperation::Conditioning,
                outcome: outcome.to_owned(),
                conditioning: conditioning.to_vec(),
                interventions: Vec::new(),
            },
            ctx,
        )
    }

    /// `P(outcome | do(interventions))`: the interventional claim.
    ///
    /// # Errors
    /// As [`Self::observe`].
    pub fn intervene_on(
        &self,
        outcome: &str,
        interventions: &[(String, String)],
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        self.observe(
            &ObservationRequest {
                operation: ObservationOperation::Intervention,
                outcome: outcome.to_owned(),
                conditioning: Vec::new(),
                interventions: interventions.to_vec(),
            },
            ctx,
        )
    }

    /// Both laws of one assignment from this model, with the largest absolute
    /// difference. Where a common cause of the assigned variables and the
    /// outcome is in the model the two differ.
    ///
    /// # Errors
    /// An unknown variable or level, an assigned outcome, evidence of
    /// probability zero, a linear-Gaussian composition, or cancellation.
    pub fn condition_versus_intervene(
        &self,
        outcome: &str,
        assignment: &[(String, String)],
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        check_cancelled(ctx)?;
        let slots: Vec<ObservedSlot> = assignment
            .iter()
            .map(|(variable, level)| ObservedSlot {
                variable_id: variable.clone(),
                level: level.clone(),
                provenance: antecedent_contract::SlotProvenance::Recorded,
                value: SlotValue::Direct,
            })
            .collect();
        check_observation_slots(&ObservationSlotsRequest {
            operation: ObservationOperation::Conditioning,
            outcome_id: outcome,
            conditioning: &slots,
            interventions: &[],
        })
        .map_err(ObservationQueryRefusal::one)?;
        seal(
            Some(self.spec()),
            ObservationQueryWire::ConditioningVersusIntervention {
                outcome: outcome.to_owned(),
                assignment: assignment.to_vec(),
            },
            InferenceBasis::DeclaredModel,
            ctx,
        )
    }

    /// The observed levels of `signal` under which
    /// `P(outcome = outcome_level)` reaches `threshold`, with the probability
    /// of observing one. `background` is recorded values held fixed.
    ///
    /// # Errors
    /// `evidence_cannot_reach_level` (stating the most extreme conditional
    /// probability any observation attains) when no observed level crosses;
    /// the refusals of [`Self::observe`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the query names the outcome, its level, the signal, the background, the threshold and its direction as separate declared values"
    )]
    pub fn threshold_crossing(
        &self,
        outcome: &str,
        outcome_level: &str,
        signal: &str,
        background: &[ObservedSlot],
        threshold: f64,
        direction: Direction,
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        check_cancelled(ctx)?;
        check_observation_slots(&ObservationSlotsRequest {
            operation: ObservationOperation::Conditioning,
            outcome_id: outcome,
            conditioning: background,
            interventions: &[],
        })
        .map_err(ObservationQueryRefusal::one)?;
        let mut fixed = Vec::with_capacity(background.len());
        for slot in background {
            if matches!(slot.value, SlotValue::Measured(_)) {
                return Err(ObservationQueryRefusal::of(
                    antecedent_core::reason_code!("route_not_supported"),
                    "observe",
                    "observation_query.measured_background",
                    slot.variable_id.clone(),
                ));
            }
            fixed.push((slot.variable_id.clone(), slot.level.clone()));
        }
        seal(
            Some(self.spec()),
            ObservationQueryWire::ThresholdCrossing {
                outcome: outcome.to_owned(),
                outcome_level: outcome_level.to_owned(),
                signal: signal.to_owned(),
                background: fixed,
                threshold_bits: threshold.to_bits(),
                direction: direction.as_str().to_owned(),
            },
            InferenceBasis::DeclaredModel,
            ctx,
        )
    }
}

// ------------------------------------------------------------------ consume

/// A consumed `observation_query_v1` artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedObservationClaim {
    /// The identity that was recomputed and verified.
    pub identity: String,
    /// The operation the claim performed.
    pub operation: String,
    /// How the value is to be read.
    pub inference_basis: String,
    /// The replayed result, bit-identical to the stored one.
    pub result: ObservationResultWire,
}

/// Independently consume an observation-query artifact: recompute the
/// identity, recompile the premises under the coherence checker and replay
/// the query to identical result bits.
///
/// `expected_identity` is the identity the consumer holds independently of the
/// bytes.
///
/// # Errors
/// Corrupt or unsupported bytes, an identity that is not the stored or the
/// retained one, premises that no longer compose (with every structured
/// refusal), a query that is now refused, or a stored result that does not
/// replay.
pub fn consume_observation_query_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ConsumedObservationClaim, ObservationQueryRefusal> {
    check_cancelled(ctx)?;
    let wire = ObservationQueryArtifactWire::decode(bytes).map_err(artifact_refusal)?;
    let consumed = wire.consume(expected_identity, ctx).map_err(artifact_refusal)?;
    Ok(ConsumedObservationClaim {
        identity: consumed.identity,
        operation: consumed.operation,
        inference_basis: consumed.inference_basis,
        result: consumed.result,
    })
}

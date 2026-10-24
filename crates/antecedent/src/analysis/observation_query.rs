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
//!    recorded variable that is itself intervened on is refused (it is set, not
//!    observed); recorded values of other variables after an intervention are
//!    the interventional law conditioned on them. A
//!    recorded value of an *indicator* offered about a construct is refused as
//!    `observation_query.measured_as_construct` unless a measurement model
//!    relating the two is declared and applies in the model's population,
//!    regime and arm.
//! 2. **Execute** (`antecedent_model::observation_query`) on the composed
//!    model of [`PreparedMechanismComposition`]: exact enumeration for a finite
//!    model, the Schur complement of the observed block of the (possibly
//!    intervened) joint law for a linear-Gaussian model whose evidence is typed
//!    real values ([`ContinuousAssignments`]); or on a declared finite or
//!    conjugate Bayes model: outcome probability given a signal, posterior
//!    after new evidence, the observations that cross a threshold, and the
//!    evidence required to reach a level.
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
//! # Parameter uncertainty
//!
//! [`PreparedMechanismComposition::observe_with_parameter_uncertainty`] redraws
//! the parameters of mechanisms whose fitting data was retained (sufficient
//! statistics for a regression, parent-configuration counts for a table) from
//! their conjugate posterior and re-answers the query under each draw. The
//! result is aligned draws (one row is one joint realization of every queried
//! quantity) carried in the distribution artifact's alignment identity. Their
//! spread is labelled `uncalibrated_spread`; an interval is returned only
//! through the composed-interval gate and a currently attesting coverage record
//! ([`ObservationClaim::parameter_interval`]), and is otherwise refused.
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
    ComposedCoordinate, ContinuousSlotsRequest, CoverageRecord, ExternalRefusal, InferenceBasis,
    InputAlignment, InputRole, MeasuredIndicator, ObservationOperation, ObservationSlotsRequest,
    ObservedSlot, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading, SlotValue, calibration_refusal, check_composed_interval,
    check_continuous_assignments, check_observation_slots, check_propagation, interval_refusal,
};
use antecedent_core::ExecutionContext;
use antecedent_io::IoError;
use antecedent_io::measurement_model_wire::model_to_wire;
use antecedent_io::observation_query_artifact::{
    ObservationQueryArtifactError, ObservationQueryArtifactWire,
};

pub use super::mechanism_exports::model_crate::observation_query::Direction;
pub use super::mechanism_exports::model_crate::observation_uncertainty::RedrawPrior;
pub use antecedent_io::observation_query_artifact::{
    MeasuredSlotWire, ObservationQueryWire, ObservationResultWire, RedrawPriorWire,
    RedrawTargetWire, RetainedMechanismWire,
};

use super::PreparedMechanismComposition;
use super::joint_law_composed_basis::{Resolution, composed_basis, resolve};
use super::mechanism_exports::model_crate::compose::MechanismFamily;
use super::mechanism_exports::model_crate::observation_uncertainty::GaussianSufficient;

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

    pub(crate) fn of_cancelled() -> Self {
        cancelled()
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

/// Typed real values for a composition of continuous variables: never a level
/// string. Used with a linear-Gaussian composition in place of the conditioning
/// slots and the level assignments, which belong to a finite composition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContinuousAssignments {
    /// `(variable, value)` recorded without error.
    pub observed: Vec<(String, f64)>,
    /// `(variable, value)` assignments of `do()`.
    pub interventions: Vec<(String, f64)>,
}

/// What is asked of a composed causal model.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationRequest {
    /// Whether the request conditions or intervenes.
    pub operation: ObservationOperation,
    /// The outcome variable.
    pub outcome: String,
    /// Values placed in conditioning slots (a finite composition).
    pub conditioning: Vec<ObservedSlot>,
    /// `(variable, level)` assignments of `do()` (a finite composition).
    pub interventions: Vec<(String, String)>,
    /// Real-valued evidence and interventions (a linear-Gaussian composition).
    pub continuous: ContinuousAssignments,
}

/// A request together with the retained data its parameters are redrawn from.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterRedrawRequest {
    /// The conditional functional whose parameter uncertainty is asked.
    pub request: ObservationRequest,
    /// The data the redrawn mechanisms were fitted from.
    pub retained: Vec<RetainedMechanismWire>,
    /// The priors of the redraw.
    pub prior: RedrawPrior,
    /// Number of posterior draws.
    pub draws: u32,
    /// Seed of the draw streams.
    pub seed: u64,
}

/// Retain the sufficient statistics of `y` regressed on an intercept and the
/// named parent columns, for a parameter redraw of a linear-Gaussian mechanism.
///
/// # Errors
/// No rows, columns of unequal length, a non-finite value or too many regressors.
pub fn retain_regression(
    variable: &str,
    parents: &[(&str, &[f64])],
    y: &[f64],
) -> Result<RetainedMechanismWire, ObservationQueryRefusal> {
    let stats = GaussianSufficient::from_columns(variable, parents, y).map_err(|error| {
        ObservationQueryRefusal::of(
            antecedent_core::reason_code!("invalid_argument"),
            "observe",
            "observation_query.invalid_declaration",
            error.to_string(),
        )
    })?;
    Ok(RetainedMechanismWire::Gaussian {
        variable: stats.variable,
        parents: stats.parents,
        gram_bits: stats.gram.iter().map(|v| v.to_bits()).collect(),
        xty_bits: stats.xty.iter().map(|v| v.to_bits()).collect(),
        yty_bits: stats.yty.to_bits(),
        rows: stats.rows,
    })
}

/// Retain the parent-configuration counts a finite mechanism was fitted from.
///
/// `rows` is one `(parent configuration, (level, count))` entry per
/// configuration of the mechanism's table.
#[must_use]
pub fn retain_counts(
    variable: &str,
    rows: Vec<(Vec<(String, String)>, Vec<(String, u64)>)>,
) -> RetainedMechanismWire {
    RetainedMechanismWire::Table { variable: variable.to_owned(), rows }
}

/// The aligned posterior draws of a parameter-redraw claim.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorDraws {
    /// Functional ids of the quantities, in column order.
    pub quantities: Vec<String>,
    /// Number of draws.
    pub draws: usize,
    /// The quantities under the composition's own (fitted) numbers.
    pub point: Vec<f64>,
    /// Draw-major values: draw `i` occupies `i * quantities.len()..`.
    pub values: Vec<f64>,
    /// Alignment identity: rows are realizations of one joint law only under
    /// this identity.
    pub alignment_identity: String,
    /// The reading of the spread: always `uncalibrated_spread`.
    pub reading: String,
}

impl PosteriorDraws {
    /// The draws of one quantity.
    #[must_use]
    pub fn column(&self, quantity: usize) -> Vec<f64> {
        let width = self.quantities.len();
        (0..self.draws).map(|d| self.values[d * width + quantity]).collect()
    }

    /// Mean and sample standard deviation of one quantity over the draws. A
    /// spread of exact posterior draws; it makes no coverage statement.
    #[must_use]
    pub fn spread(&self, quantity: usize) -> (f64, f64) {
        let column = self.column(quantity);
        let n = column.len() as f64;
        let mean = column.iter().sum::<f64>() / n;
        let ss: f64 = column.iter().map(|v| (v - mean) * (v - mean)).sum();
        (mean, (ss / (n - 1.0)).sqrt())
    }
}

/// An interval of redrawn quantities that a currently attesting coverage record
/// licensed.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterInterval {
    /// The quantity.
    pub quantity: String,
    /// The composed coordinate the record was matched against.
    pub coordinate_id: String,
    /// The record that licensed the interval.
    pub record_id: String,
    /// The construction the interval was built with.
    pub construction: String,
    /// The nominal level.
    pub nominal_level: f64,
    /// The label of the result.
    pub label: &'static str,
    /// The equal-tailed bounds of the aligned draws; present for the draw
    /// constructions.
    pub bounds: Option<(f64, f64)>,
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

    /// Re-open a sealed claim from its artifact bytes, after the independent
    /// consumer has recomputed the identity, recompiled the premises and
    /// replayed the query to identical result bits.
    ///
    /// # Errors
    /// Any refusal of [`consume_observation_query_artifact`].
    pub fn from_artifact(
        bytes: &[u8],
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Self, ObservationQueryRefusal> {
        check_cancelled(ctx)?;
        let wire = ObservationQueryArtifactWire::decode(bytes).map_err(artifact_refusal)?;
        wire.consume(expected_identity, ctx).map_err(artifact_refusal)?;
        Ok(Self { wire })
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
    /// Always `cell_not_licensed` with `observation_query.declared_model_has_no_sampling_interval`.
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

    /// `(conditional mean, conditional variance, log density of the evidence)`
    /// of a Gaussian claim; the log density is `None` when nothing was observed.
    #[must_use]
    pub fn gaussian_conditional(&self) -> Option<(f64, f64, Option<f64>)> {
        match &self.wire.result {
            ObservationResultWire::GaussianConditional {
                mean_bits,
                variance_bits,
                evidence_log_density_bits,
                ..
            } => Some((
                f64::from_bits(*mean_bits),
                f64::from_bits(*variance_bits),
                evidence_log_density_bits.map(f64::from_bits),
            )),
            _ => None,
        }
    }

    /// The aligned posterior draws of a parameter-redraw claim.
    #[must_use]
    pub fn parameter_draws(&self) -> Option<PosteriorDraws> {
        match &self.wire.result {
            ObservationResultWire::ParameterDraws {
                quantities,
                draws,
                point_bits,
                values_bits,
                alignment_identity,
                reading,
            } => Some(PosteriorDraws {
                quantities: quantities.clone(),
                draws: *draws as usize,
                point: point_bits.iter().map(|b| f64::from_bits(*b)).collect(),
                values: values_bits.iter().map(|b| f64::from_bits(*b)).collect(),
                alignment_identity: alignment_identity.clone(),
                reading: reading.clone(),
            }),
            _ => None,
        }
    }

    /// The aligned draws of a parameter-redraw claim as the repo's
    /// distribution artifact (alignment `Joint`, calibration `Unmeasured`).
    ///
    /// # Errors
    /// A claim that redraws nothing, or draws that do not form a distribution.
    pub fn distribution(
        &self,
    ) -> Result<antecedent_io::distribution_artifact::DistributionArtifact, ObservationQueryRefusal>
    {
        self.wire.distribution().map_err(artifact_refusal)
    }

    /// The interval of one redrawn quantity, licensed only by a currently
    /// attesting coverage record the registry holds for exactly this composed
    /// coordinate, construction, nominal level and evidence size. The caller
    /// supplies no record. Without one the interval is closed with the typed
    /// refusal and the spread of the draws is the labelled
    /// `uncalibrated_spread` ([`PosteriorDraws::spread`]).
    ///
    /// `evidence_rows` is declared, not authenticated: the number of
    /// observations behind the least-supported retained mechanism.
    ///
    /// # Errors
    /// A claim without parameter draws, an unknown quantity, or the refusals of
    /// `antecedent_contract::check_composed_interval` (a missing or
    /// non-attesting record).
    pub fn parameter_interval(
        &self,
        quantity: &str,
        construction: &str,
        evidence_rows: u64,
    ) -> Result<ParameterInterval, ObservationQueryRefusal> {
        let Some(draws) = self.parameter_draws() else {
            return Err(ObservationQueryRefusal::of(
                antecedent_core::reason_code!("invalid_argument"),
                "observe",
                "observation_query.no_parameter_draws",
                "this claim carries no parameter draws; ask for parameter uncertainty with a retained-data redraw",
            ));
        };
        let Some(index) = draws.quantities.iter().position(|q| q == quantity) else {
            return Err(ObservationQueryRefusal::of(
                antecedent_core::reason_code!("invalid_argument"),
                "observe",
                "observation_query.unknown_quantity",
                quantity,
            ));
        };
        let request = PropagationRequest {
            operation: PropagationOperation::LinearTransform,
            inputs: vec![PropagationInput {
                id: format!("observation_query:{quantity}"),
                evidence_id: self.wire.identity.clone(),
                role: InputRole::Estimate,
                alignment: InputAlignment::Aligned {
                    alignment_id: draws.alignment_identity.clone(),
                },
            }],
            dependence: PropagationDependence::AlignedJointLaw,
            reading: RequestedReading::Interval,
        };
        let spread = PropagationRequest { reading: RequestedReading::Spread, ..request.clone() };
        let propagation = check_propagation(&spread).map_err(ObservationQueryRefusal::one)?;
        let coordinate = ComposedCoordinate::of(&spread, &propagation);
        let coordinate_id = coordinate.id();
        let basis = composed_basis(
            &coordinate_id,
            propagation.route.as_str(),
            &request.dependence,
            construction,
            evidence_rows,
        );
        let record = match resolve(&basis) {
            Resolution::NoRecord => None,
            Resolution::NotAttesting { record_id, .. } => Some(CoverageRecord {
                record_id,
                coordinate_id,
                construction: construction.to_owned(),
                nominal_level: super::joint_law_composed_basis::COMPOSED_LEVEL,
                measured: false,
            }),
            Resolution::Calibrated { record_id } => Some(CoverageRecord {
                record_id,
                coordinate_id,
                construction: construction.to_owned(),
                nominal_level: super::joint_law_composed_basis::COMPOSED_LEVEL,
                measured: true,
            }),
        };
        let plan = check_composed_interval(&request, record.as_ref(), construction)
            .map_err(ObservationQueryRefusal::one)?;
        let rule = if construction == antecedent_stats::DRAW_EXCHANGEABLE_RANK_CONSTRUCTION {
            Some(antecedent_stats::QuantileRule::ExchangeableRank)
        } else if construction == antecedent_stats::DRAW_INTERPOLATED_CONSTRUCTION {
            Some(antecedent_stats::QuantileRule::Interpolated)
        } else {
            None
        };
        let bounds = match rule {
            Some(rule) => {
                let candidate = antecedent_stats::draw_interval_candidate(
                    &draws.column(index),
                    plan.nominal_level,
                    rule,
                )
                .map_err(|error| {
                    ObservationQueryRefusal::of(
                        antecedent_core::reason_code!("invalid_argument"),
                        "observe",
                        "observation_query.invalid_interval",
                        error.to_string(),
                    )
                })?;
                Some((candidate.lower, candidate.upper))
            }
            None => None,
        };
        Ok(ParameterInterval {
            quantity: quantity.to_owned(),
            coordinate_id: plan.coordinate.id(),
            record_id: plan.record_id,
            construction: plan.construction,
            nominal_level: plan.nominal_level,
            label: plan.label,
            bounds,
        })
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

/// Split conditioning slots into direct recorded values and the one measured
/// signal, if any.
fn split_slots(
    slots: &[ObservedSlot],
) -> Result<(Vec<(String, String)>, Option<MeasuredSlotWire>), ObservationQueryRefusal> {
    let mut observed = Vec::new();
    let mut measured = None;
    for slot in slots {
        match &slot.value {
            SlotValue::Direct => observed.push((slot.variable_id.clone(), slot.level.clone())),
            SlotValue::Measured(indicator) => measured = Some(measured_wire(slot, indicator)?),
        }
    }
    Ok((observed, measured))
}

fn real_bits(pairs: &[(String, f64)]) -> Vec<(String, u64)> {
    pairs.iter().map(|(name, value)| (name.clone(), value.to_bits())).collect()
}

impl PreparedMechanismComposition {
    /// Answer one observation query on this composed model.
    ///
    /// On a finite composition the request is checked first: an assigned value
    /// in a conditioning slot, an intervention supplied to a conditioning
    /// query, a recorded variable that is also intervened on, and a recorded
    /// indicator value without a measurement model are each refused with their
    /// own detail. Recorded values of other variables together with an
    /// intervention are the interventional law conditioned on them,
    /// `P(y | do(s), z)`. On a linear-Gaussian composition the evidence and the
    /// interventions are the typed real values of
    /// [`ObservationRequest::continuous`], and the answer is the conditional
    /// mean and variance of the outcome (Schur complement of the observed
    /// block of the possibly intervened joint law).
    ///
    /// # Errors
    /// Any refusal of the slot check; level strings given to a linear-Gaussian
    /// composition or real values given to a finite one; an unknown variable
    /// or level; evidence of probability zero or of singular covariance;
    /// incoherent mechanisms; cancellation.
    pub fn observe(
        &self,
        request: &ObservationRequest,
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        check_cancelled(ctx)?;
        let query = self.query_of(request)?;
        seal(Some(self.spec()), query, InferenceBasis::DeclaredModel, ctx)
    }

    /// The wire query a request states, after the slot checks of its family.
    fn query_of(
        &self,
        request: &ObservationRequest,
    ) -> Result<ObservationQueryWire, ObservationQueryRefusal> {
        match self.family() {
            MechanismFamily::LinearGaussian => {
                if !request.conditioning.is_empty() || !request.interventions.is_empty() {
                    return Err(ObservationQueryRefusal::one(
                        ExternalRefusal::new(
                            antecedent_core::reason_code!("invalid_argument"),
                            "observe",
                            "observation_query.levels_for_continuous_model",
                        )
                        .with_offending(
                            request
                                .conditioning
                                .first()
                                .map(|s| s.variable_id.clone())
                                .or_else(|| request.interventions.first().map(|(v, _)| v.clone()))
                                .unwrap_or_default(),
                        )
                        .with_remedy(
                            "state the evidence and the interventions of a linear-Gaussian composition as typed real values (the `continuous` assignments), not as levels",
                        ),
                    ));
                }
                check_continuous_assignments(&ContinuousSlotsRequest {
                    operation: request.operation,
                    outcome_id: &request.outcome,
                    observed: &request.continuous.observed,
                    interventions: &request.continuous.interventions,
                })
                .map_err(ObservationQueryRefusal::one)?;
                Ok(ObservationQueryWire::GaussianOutcome {
                    outcome: request.outcome.clone(),
                    interventions: real_bits(&request.continuous.interventions),
                    observed: real_bits(&request.continuous.observed),
                })
            }
            MechanismFamily::Tabular => {
                if let Some((variable, _)) = request
                    .continuous
                    .observed
                    .first()
                    .or_else(|| request.continuous.interventions.first())
                {
                    return Err(ObservationQueryRefusal::one(
                        ExternalRefusal::new(
                            antecedent_core::reason_code!("invalid_argument"),
                            "observe",
                            "observation_query.real_values_for_finite_model",
                        )
                        .with_offending(variable.clone())
                        .with_remedy(
                            "state the evidence and the interventions of a finite composition as levels of the variable (conditioning slots and level assignments), not as real values",
                        ),
                    ));
                }
                check_observation_slots(&ObservationSlotsRequest {
                    operation: request.operation,
                    outcome_id: &request.outcome,
                    conditioning: &request.conditioning,
                    interventions: &request.interventions,
                })
                .map_err(ObservationQueryRefusal::one)?;
                let (observed, measured) = split_slots(&request.conditioning)?;
                Ok(match request.operation {
                    ObservationOperation::Intervention if request.conditioning.is_empty() => {
                        ObservationQueryWire::InterventionalOutcome {
                            outcome: request.outcome.clone(),
                            interventions: request.interventions.clone(),
                        }
                    }
                    ObservationOperation::Intervention => {
                        ObservationQueryWire::ConditionalInterventionalOutcome {
                            outcome: request.outcome.clone(),
                            interventions: request.interventions.clone(),
                            observed,
                            measured,
                        }
                    }
                    ObservationOperation::Conditioning => {
                        ObservationQueryWire::ConditionalOutcome {
                            outcome: request.outcome.clone(),
                            observed,
                            measured,
                        }
                    }
                })
            }
        }
    }

    /// The parameter uncertainty of a conditional query, by retained-data
    /// redraw: the parameters of every mechanism with retained data are drawn
    /// from their conjugate posterior (Normal-Inverse-Gamma for a regression,
    /// Dirichlet for a table), the query is re-answered under each draw and the
    /// quantities are returned as aligned draws.
    ///
    /// Intervals are not produced here: the spread is labelled
    /// `uncalibrated_spread`, and an interval is available only through
    /// [`ObservationClaim::parameter_interval`] and a currently attesting
    /// coverage record. A measured signal is not redrawn (its error rates are
    /// declared, not fitted from retained data).
    ///
    /// # Errors
    /// The refusals of [`Self::observe`]; a measured signal; retained data that
    /// does not match the composition; a prior outside its range; a draw count
    /// outside the supported range; evidence of probability zero under a draw.
    pub fn observe_with_parameter_uncertainty(
        &self,
        request: &ParameterRedrawRequest,
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        check_cancelled(ctx)?;
        let query = self.query_of(&request.request)?;
        let target = match query {
            ObservationQueryWire::GaussianOutcome { outcome, interventions, observed } => {
                RedrawTargetWire::Gaussian { outcome, interventions, observed }
            }
            ObservationQueryWire::ConditionalOutcome { outcome, observed, measured: None } => {
                RedrawTargetWire::Finite { outcome, interventions: Vec::new(), observed }
            }
            ObservationQueryWire::InterventionalOutcome { outcome, interventions } => {
                RedrawTargetWire::Finite { outcome, interventions, observed: Vec::new() }
            }
            ObservationQueryWire::ConditionalInterventionalOutcome {
                outcome,
                interventions,
                observed,
                measured: None,
            } => RedrawTargetWire::Finite { outcome, interventions, observed },
            other => {
                return Err(ObservationQueryRefusal::of(
                    antecedent_core::reason_code!("route_not_supported"),
                    "observe",
                    "observation_query.measured_signal_not_redrawn",
                    other.operation(),
                ));
            }
        };
        seal(
            Some(self.spec()),
            ObservationQueryWire::ParameterRedraw {
                target,
                retained: request.retained.clone(),
                prior: RedrawPriorWire::from(&request.prior),
                draws: request.draws,
                seed: request.seed,
            },
            InferenceBasis::DeclaredModel,
            ctx,
        )
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
                continuous: ContinuousAssignments::default(),
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
                continuous: ContinuousAssignments::default(),
            },
            ctx,
        )
    }

    /// `P(outcome | do(interventions), observed values)` on a linear-Gaussian
    /// composition: the conditional mean and variance of the outcome given
    /// typed real values. Either list may be empty; with no interventions this
    /// is conditioning and with no observed values it is intervention.
    ///
    /// # Errors
    /// As [`Self::observe`].
    pub fn observe_continuous(
        &self,
        outcome: &str,
        interventions: &[(String, f64)],
        observed: &[(String, f64)],
        ctx: &ExecutionContext,
    ) -> Result<ObservationClaim, ObservationQueryRefusal> {
        self.observe(
            &ObservationRequest {
                operation: if interventions.is_empty() {
                    ObservationOperation::Conditioning
                } else {
                    ObservationOperation::Intervention
                },
                outcome: outcome.to_owned(),
                conditioning: Vec::new(),
                interventions: Vec::new(),
                continuous: ContinuousAssignments {
                    observed: observed.to_vec(),
                    interventions: interventions.to_vec(),
                },
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

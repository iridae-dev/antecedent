//! The producer connection: how cohort-decision and sequential-evaluation outputs reach
//! the reliability carrier.
//!
//! Both routes are point-only on their own: a cohort decision is one affine mean
//! contrast, a rollout or off-policy estimate is a mean with a standard error and a
//! count. Neither holds a law, so neither can be handed to the carrier as it stands.
//! This module is the explicit seam:
//!
//! * a producer that **kept per-draw utilities from one declared joint law** hands them
//!   over as [`CohortUtilityDraws`], [`RolloutUtilityDraws`] or [`OffPolicyReturnDraws`]
//!   and receives a [`ProducerCarrier`], whose stream identity is built from the
//!   producer's own semantic coordinates (producing source or model, stream scheme, seed,
//!   purpose, replicate) and never from a position;
//! * a producer that holds only the point output hands over [`PointOnlyOutput`] and is
//!   refused with a typed reason: `joint_law_required` for a cohort mean (draws need an
//!   aligned law), `diagnostic_not_available` for a rollout or off-policy mean (the
//!   per-draw utilities were not retained).
//!
//! What a carrier may claim depends on where its randomness came from. Per-action
//! utilities that are functions of one posterior draw are a joint law, and ranking
//! statements over them are licensed to the carrier. Policies rolled out on
//! implementation-keyed uniforms share noise as a variance-reduction device, which makes
//! their means and mean differences valid but does not make pairwise ranking
//! probabilities a statement about any structural joint law; and importance-weighted
//! per-trajectory contributions are not outcome draws at all. Those carriers are
//! [`JointLawStanding::MarginalOnly`]: functional estimates and selection assessment are
//! allowed, ranking and the confidence-set and probability intervals that rest on a
//! joint law are refused `joint_law_required`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, is_blank};
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::draws::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, SourceParadigm,
    WithinStreamDependence,
};
use super::interval::{DerivedUncertainty, IntervalRequest, derive_uncertainty};
use super::mc_error::{BatchPlan, Functional, estimate};
use super::ranking::{RankingReport, RankingRequest, rank};
use super::report::{ReliabilityReport, ReliabilityRequest, assess};
use super::stream::{DrawSegment, StreamId, StreamPurpose};

/// The route that produced the outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerKind {
    /// A fitted effect model evaluated over a cohort and mapped to a decision.
    CohortMeanDecision,
    /// Policies rolled out on a model.
    SequentialRollout,
    /// Policies evaluated from logged trajectories by importance weighting.
    SequentialOffPolicy,
}

impl ProducerKind {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CohortMeanDecision => "cohort_decision",
            Self::SequentialRollout => "sequential_rollout",
            Self::SequentialOffPolicy => "sequential_off_policy",
        }
    }

    /// Producer label the carrier is stamped with.
    const fn carrier_label(self) -> &'static str {
        match self {
            Self::CohortMeanDecision => "cohort_decision",
            Self::SequentialRollout | Self::SequentialOffPolicy => "sequential_evaluation",
        }
    }
}

/// Whether per-action utilities come from one joint law.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointLawDeclaration {
    /// Draw `i` of every action is evaluated on one draw of the same law (for example
    /// one posterior draw of the model coefficients).
    OneSharedLaw,
    /// The actions' utilities were drawn from separate marginals: no pairing exists.
    SeparateMarginals,
    /// Not declared.
    Undeclared,
}

/// What a carrier built by a producer may be used for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointLawStanding {
    /// The utilities are functions of one declared joint law: ranking statements and the
    /// joint-law intervals are licensed to the carrier.
    OneSharedLaw,
    /// Only functional estimates and selection assessment are licensed; ranking is
    /// refused because the shared noise is not a structural joint law.
    MarginalOnly {
        /// Why no ranking is licensed.
        reason: &'static str,
    },
}

/// A producer output that holds no draws.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointOnlyOutput {
    /// The producing route.
    pub producer: ProducerKind,
    /// Identity of the source or estimate (for the refusal).
    pub identity: String,
}

/// Per-draw cohort utilities from an aligned law.
#[derive(Clone, Debug, PartialEq)]
pub struct CohortUtilityDraws {
    /// Identity of the retained cohort decision source the draws were mapped from.
    pub source_identity: String,
    /// Utility unit.
    pub utility_unit: String,
    /// Run seed of the draws' stream.
    pub run_seed: u64,
    /// What the draws are used for.
    pub purpose: StreamPurpose,
    /// Replicate coordinate.
    pub replicate: u64,
    /// Semantic label of the draws (posterior name, fold).
    pub label: String,
    /// Posterior or sampling draws; a fitted model's law is never declared known.
    pub source: SourceParadigm,
    /// Whether the actions' utilities share one law.
    pub joint_law: JointLawDeclaration,
    /// Serial dependence of the draws.
    pub within_stream: WithinStreamDependence,
    /// Per-action utilities of the cohort, one value per draw.
    pub actions: Vec<(String, Vec<f64>)>,
    /// The point-only values the cohort route reported per action; each must agree with
    /// the mean of its draws within `max_standard_errors`. Empty: no check.
    pub reported_means: Vec<(String, f64)>,
    /// Tolerance of the agreement check, in standard errors of the draws' mean.
    pub max_standard_errors: f64,
}

/// Whether a rollout's shared uniforms are the structural exogenous noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutNoise {
    /// The shared uniforms are the model's exogenous noise: policies are compared on
    /// the same simulated world.
    StructuralExogenous,
    /// The uniforms are keyed by `(seed, rollout, time, role)` of the implementation:
    /// policies share noise for variance reduction only.
    ImplementationKeyed,
}

/// Per-rollout utilities of policies rolled out on one model and seed.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutUtilityDraws {
    /// Identity of the model the policies were rolled out on.
    pub model_identity: String,
    /// The stream scheme the rollout draws were keyed by.
    pub stream_scheme: String,
    /// Master seed of the rollouts.
    pub seed: u64,
    /// Utility unit.
    pub utility_unit: String,
    /// What the draws are used for.
    pub purpose: StreamPurpose,
    /// Replicate coordinate.
    pub replicate: u64,
    /// Nature of the shared noise.
    pub noise: RolloutNoise,
    /// Total utility of rollout `r` of each policy, keyed by policy identity.
    pub policies: Vec<(String, Vec<f64>)>,
    /// The mean values the rollout reported per policy; empty: no check.
    pub reported_means: Vec<(String, f64)>,
    /// Tolerance of the agreement check, in standard errors of the draws' mean.
    pub max_standard_errors: f64,
}

/// Per-trajectory importance-weighted returns of target policies on one logged data set.
#[derive(Clone, Debug, PartialEq)]
pub struct OffPolicyReturnDraws {
    /// Identity of the model the returns were computed under.
    pub model_identity: String,
    /// Identity of the logged trajectories.
    pub log_identity: String,
    /// Run seed coordinate of the stream.
    pub run_seed: u64,
    /// Utility unit.
    pub utility_unit: String,
    /// What the draws are used for.
    pub purpose: StreamPurpose,
    /// Replicate coordinate.
    pub replicate: u64,
    /// `w_i G_i` of trajectory `i` for each target policy, keyed by policy identity.
    pub policies: Vec<(String, Vec<f64>)>,
    /// The ordinary importance-weighted estimates reported per policy; empty: no check.
    pub reported_means: Vec<(String, f64)>,
    /// Tolerance of the agreement check, in standard errors of the draws' mean.
    pub max_standard_errors: f64,
}

/// What a producer hands to the reliability route.
#[derive(Clone, Debug, PartialEq)]
pub enum ProducerOutput {
    /// Cohort utilities from an aligned law.
    Cohort(CohortUtilityDraws),
    /// Per-rollout utilities.
    Rollout(RolloutUtilityDraws),
    /// Per-trajectory weighted returns.
    OffPolicy(OffPolicyReturnDraws),
    /// Only the point output: refused.
    PointOnly(PointOnlyOutput),
}

/// A validated carrier together with what the producer allows it to claim.
#[derive(Clone, Debug, PartialEq)]
pub struct ProducerCarrier {
    draws: DecisionDraws,
    producer: ProducerKind,
    joint_law: JointLawStanding,
    expected_utility_only: bool,
}

impl ProducerCarrier {
    /// The carrier.
    #[must_use]
    pub fn draws(&self) -> &DecisionDraws {
        &self.draws
    }

    /// The producing route.
    #[must_use]
    pub fn producer(&self) -> ProducerKind {
        self.producer
    }

    /// What the producer allows the carrier to claim.
    #[must_use]
    pub fn joint_law(&self) -> JointLawStanding {
        self.joint_law
    }

    fn check_functional(&self, functional: &Functional) -> Result<(), ReliabilityError> {
        if self.expected_utility_only && *functional != Functional::ExpectedUtility {
            return Err(ReliabilityError::RouteNotSupported {
                message: format!(
                    "{}: importance-weighted contributions carry the expected utility only; \
                     a {} of contributions is not a quantity of any policy",
                    self.producer.as_str(),
                    functional.kind()
                ),
            });
        }
        Ok(())
    }

    fn check_ranking(&self, wants_ranking: bool) -> Result<(), ReliabilityError> {
        match (wants_ranking, self.joint_law) {
            (true, JointLawStanding::MarginalOnly { reason }) => {
                Err(ReliabilityError::JointLawRequired { reason })
            }
            _ => Ok(()),
        }
    }

    /// Assess the carrier's decision.
    ///
    /// # Errors
    /// A ranking request on a marginal-only carrier, a non-expected-utility functional on
    /// weighted contributions, or anything [`assess`] refuses.
    pub fn assess(
        &self,
        validation: Option<&DecisionDraws>,
        request: &ReliabilityRequest,
        ctx: &ExecutionContext,
    ) -> Result<ReliabilityReport, ReliabilityError> {
        self.check_functional(&request.functional)?;
        self.check_ranking(request.ranking.is_some())?;
        assess(&self.draws, validation, request, ctx)
    }

    /// Rank the carrier's candidates.
    ///
    /// # Errors
    /// A marginal-only carrier, or anything [`rank`] refuses.
    pub fn rank(
        &self,
        request: &RankingRequest,
        ctx: &ExecutionContext,
    ) -> Result<RankingReport, ReliabilityError> {
        self.check_ranking(true)?;
        rank(&self.draws, request, ctx)
    }

    /// Derive the three components of uncertainty for the carrier's decision.
    ///
    /// A marginal-only carrier may derive its numerical and selection components; the
    /// ranking probabilities and the pairwise confidence set rest on a joint law and are
    /// refused.
    ///
    /// # Errors
    /// A ranking request on a marginal-only carrier, or anything [`derive_uncertainty`]
    /// refuses.
    pub fn derive_uncertainty(
        &self,
        validation: Option<&DecisionDraws>,
        request: &ReliabilityRequest,
        interval: &IntervalRequest,
        ctx: &ExecutionContext,
    ) -> Result<DerivedUncertainty, ReliabilityError> {
        self.check_functional(&request.functional)?;
        self.check_ranking(request.ranking.is_some())?;
        derive_uncertainty(&self.draws, validation, request, interval, ctx)
    }
}

fn point_only_held(producer: ProducerKind) -> &'static str {
    match producer {
        ProducerKind::CohortMeanDecision => "an_affine_mean_contrast_without_a_law",
        ProducerKind::SequentialRollout => "a_mean_and_standard_error_without_rollout_utilities",
        ProducerKind::SequentialOffPolicy => {
            "an_importance_weighted_mean_without_trajectory_returns"
        }
    }
}

fn require_named(slot: &'static str, value: &str) -> Result<(), ReliabilityError> {
    if is_blank(value) { Err(ReliabilityError::invalid(slot, "must be named")) } else { Ok(()) }
}

fn check_reported(
    columns: &[(String, Vec<f64>)],
    reported: &[(String, f64)],
    within: WithinStreamDependence,
    max_standard_errors: f64,
) -> Result<(), ReliabilityError> {
    if reported.is_empty() {
        return Ok(());
    }
    if !(max_standard_errors.is_finite() && max_standard_errors > 0.0) {
        return Err(ReliabilityError::invalid(
            "max_standard_errors",
            "agreement tolerance must be finite and positive",
        ));
    }
    for (id, value) in reported {
        let (_, utilities) = columns.iter().find(|(name, _)| name == id).ok_or_else(|| {
            ReliabilityError::invalid("reported_means", "a reported value names no supplied id")
        })?;
        let mean =
            estimate(utilities, None, within, BatchPlan::SquareRoot, &Functional::ExpectedUtility)?
                .estimate;
        let tolerance = max_standard_errors * mean.standard_error + 1e-9 * (1.0 + value.abs());
        if !value.is_finite() || (mean.value - value).abs() > tolerance {
            return Err(ReliabilityError::ProducerMeanDisagrees {
                id: id.clone(),
                reported: *value,
                draws_mean: mean.value,
                standard_error: mean.standard_error,
            });
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "one constructor for the carrier every producer shares"
)]
fn build(
    producer: ProducerKind,
    stream: &StreamId,
    utility_unit: &str,
    source: SourceParadigm,
    within_stream: WithinStreamDependence,
    columns: Vec<(String, Vec<f64>)>,
    joint_law: JointLawStanding,
    expected_utility_only: bool,
) -> Result<ProducerCarrier, ReliabilityError> {
    let candidates = columns
        .into_iter()
        .map(|(id, utilities)| {
            let segment = DrawSegment::whole(stream.clone(), utilities.len());
            CandidateDraws { id, utilities, segment }
        })
        .collect();
    let draws = DecisionDraws {
        utility_unit: utility_unit.to_owned(),
        producer: producer.carrier_label().to_owned(),
        source,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates,
    };
    draws.validate()?;
    Ok(ProducerCarrier { draws, producer, joint_law, expected_utility_only })
}

fn cohort(input: CohortUtilityDraws) -> Result<ProducerCarrier, ReliabilityError> {
    require_named("source_identity", &input.source_identity)?;
    require_named("label", &input.label)?;
    match input.joint_law {
        JointLawDeclaration::OneSharedLaw => {}
        JointLawDeclaration::SeparateMarginals => {
            return Err(ReliabilityError::JointLawRequired {
                reason: "cohort_utilities_drawn_from_separate_marginals",
            });
        }
        JointLawDeclaration::Undeclared => {
            return Err(ReliabilityError::DependenceUnknown { what: "cohort_joint_law" });
        }
    }
    if input.source == SourceParadigm::KnownLaw {
        return Err(ReliabilityError::invalid(
            "source",
            "a fitted effect model's law is never known: declare posterior or sampling draws",
        ));
    }
    check_reported(
        &input.actions,
        &input.reported_means,
        input.within_stream,
        input.max_standard_errors,
    )?;
    let stream = StreamId::new(
        input.run_seed,
        input.purpose,
        input.replicate,
        format!("cohort_decision/{}/{}", input.source_identity, input.label),
    );
    build(
        ProducerKind::CohortMeanDecision,
        &stream,
        &input.utility_unit,
        input.source,
        input.within_stream,
        input.actions,
        JointLawStanding::OneSharedLaw,
        false,
    )
}

fn rollout(input: RolloutUtilityDraws) -> Result<ProducerCarrier, ReliabilityError> {
    require_named("model_identity", &input.model_identity)?;
    require_named("stream_scheme", &input.stream_scheme)?;
    check_reported(
        &input.policies,
        &input.reported_means,
        WithinStreamDependence::Independent,
        input.max_standard_errors,
    )?;
    let stream = StreamId::new(
        input.seed,
        input.purpose,
        input.replicate,
        format!("sequential_rollout/{}/{}", input.model_identity, input.stream_scheme),
    );
    let joint_law = match input.noise {
        RolloutNoise::StructuralExogenous => JointLawStanding::OneSharedLaw,
        RolloutNoise::ImplementationKeyed => JointLawStanding::MarginalOnly {
            reason: "implementation_keyed_noise_is_not_a_structural_joint_law",
        },
    };
    build(
        ProducerKind::SequentialRollout,
        &stream,
        &input.utility_unit,
        SourceParadigm::KnownLaw,
        WithinStreamDependence::Independent,
        input.policies,
        joint_law,
        false,
    )
}

fn off_policy(input: OffPolicyReturnDraws) -> Result<ProducerCarrier, ReliabilityError> {
    require_named("model_identity", &input.model_identity)?;
    require_named("log_identity", &input.log_identity)?;
    check_reported(
        &input.policies,
        &input.reported_means,
        WithinStreamDependence::Independent,
        input.max_standard_errors,
    )?;
    let stream = StreamId::new(
        input.run_seed,
        input.purpose,
        input.replicate,
        format!("sequential_off_policy/{}/{}", input.model_identity, input.log_identity),
    );
    build(
        ProducerKind::SequentialOffPolicy,
        &stream,
        &input.utility_unit,
        SourceParadigm::SamplingDistribution,
        WithinStreamDependence::Independent,
        input.policies,
        JointLawStanding::MarginalOnly {
            reason: "importance_weighted_contributions_are_not_outcome_draws",
        },
        true,
    )
}

/// Take a producer's output to the reliability carrier, or refuse with the typed reason
/// that names what the producer holds.
///
/// # Errors
/// [`ReliabilityError::ProducerPointOnly`] for a point-only output;
/// [`ReliabilityError::JointLawRequired`] or [`ReliabilityError::DependenceUnknown`] for
/// cohort utilities without a declared shared law; a reported value the draws disagree
/// with; or any carrier defect.
pub fn reach_carrier(output: ProducerOutput) -> Result<ProducerCarrier, ReliabilityError> {
    match output {
        ProducerOutput::Cohort(input) => cohort(input),
        ProducerOutput::Rollout(input) => rollout(input),
        ProducerOutput::OffPolicy(input) => off_policy(input),
        ProducerOutput::PointOnly(point) => Err(ReliabilityError::ProducerPointOnly {
            producer: point.producer,
            held: point_only_held(point.producer),
        }),
    }
}

//! Compile that starts from a decision or from a value-of-information request.
//!
//! [`Study::compile_for_decision`] and [`Study::compile_for_evsi`] return the plan
//! [`Study::compile`] returns, with layer four of the lowered program filled from
//! the decision: the carrier it demands of the statistical program, its actions,
//! constraints, chance threshold, utility form and unit, the decision contract's
//! identity and, for EVSI, the candidates, the observations the prior already
//! summarizes, the ranking schedule, the seed and the Monte Carlo tolerance.
//!
//! Layer four checks the demanded carrier against the statistical realization
//! ([`require_carrier`]). A confidence interval is not a parameter posterior,
//! parameter draws are not individual outcome risk, an identified set has no
//! point mean, and a mean carries only a utility affine in the state. Each
//! refusal is a typed [`DecisionCarrierError`] that names the next attempt.
//!
//! EVSI adds a Monte Carlo task dimension: one score per candidate and draw, and
//! one common-random-number seed per draw. The store is quoted with checked
//! arithmetic and refused before compile allocates anything when the admission
//! pool cannot hold it.
//!
//! [`score_decision`] applies the same carrier rules to an executed answer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use antecedent_contract::{DecisionRequest, EvidencePort, EvidencePortError};
use antecedent_core::{ExecutionContext, NextAttempt, is_blank};
use antecedent_design::DecisionProblem;
use antecedent_design::evsi::EvsiRequest;

use super::Study;
use crate::analysis::lowering::{CompileGoal, RouteCell};
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{CarrierDemand, DecisionProgram, EvsiProgram};
use crate::planner::statistical::Realization;

/// Namespaced detail of every carrier refusal.
const CARRIER_DETAIL: &str = "decision_compile.carrier";

/// Compile the same study under Bayesian inference.
pub const BAYESIAN_REALIZATION: NextAttempt = NextAttempt::new(
    "bayesian_realization",
    "compile the same study under Bayesian inference so the statistical program publishes a \
     parameter posterior; an affine utility can instead demand the mean",
);

/// Compile a query whose answer is an interventional law.
pub const INTERVENTIONAL_LAW: NextAttempt = NextAttempt::new(
    "interventional_law",
    "compile an interventional-distribution query for the outcome so the program publishes an \
     interventional law",
);

/// Evaluate the decision robustly over the identified set.
pub const ROBUST_OVER_IDENTIFIED_SET: NextAttempt = NextAttempt::new(
    "robust_over_identified_set",
    "evaluate the decision over the identified set with evaluate_identified_sets; a robust \
     verdict is not an expected utility",
);

/// Demand a distributional carrier, or declare the utility's affine form.
pub const DISTRIBUTIONAL_CARRIER: NextAttempt = NextAttempt::new(
    "distributional_carrier",
    "demand a parameter posterior or an outcome law, or declare the utility's affine form in the \
     state",
);

/// Every attempt a carrier refusal can name.
pub const DECISION_NEXT_ATTEMPTS: [NextAttempt; 4] =
    [BAYESIAN_REALIZATION, INTERVENTIONAL_LAW, ROBUST_OVER_IDENTIFIED_SET, DISTRIBUTIONAL_CARRIER];

/// What a decision asks a published answer to score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecisionDemand {
    /// Affine utility of the answer, scored from a parameter posterior.
    AffinePosterior {
        /// Utility per unit of the answer.
        slope: f64,
        /// Utility at zero.
        intercept: f64,
    },
    /// Affine utility of the answer, scored from the mean.
    AffineMean {
        /// Utility per unit of the answer.
        slope: f64,
        /// Utility at zero.
        intercept: f64,
    },
    /// Nonlinear utility. A mean cannot carry it.
    Nonlinear,
    /// Threshold or other individual outcome risk. A mean cannot carry it.
    Risk,
}

impl DecisionDemand {
    /// The carrier this demand needs.
    #[must_use]
    pub const fn carrier(self) -> CarrierDemand {
        match self {
            Self::AffinePosterior { .. } => CarrierDemand::ParameterPosterior,
            Self::AffineMean { .. } => CarrierDemand::Mean,
            Self::Nonlinear | Self::Risk => CarrierDemand::OutcomeLaw,
        }
    }
}

/// Carrier a decision still needs, or that an answer could not publish.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MissingCarrier {
    /// Parameter posterior. A confidence interval is not this carrier.
    ParameterPosterior,
    /// Predictive or interventional law. A mean is not this carrier.
    OutcomeLaw,
}

impl MissingCarrier {
    /// Stable name: `parameter_posterior` or `outcome_law`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ParameterPosterior => "parameter_posterior",
            Self::OutcomeLaw => "outcome_law",
        }
    }

    /// The missing carrier a refused demand names. A mean is not a missing
    /// carrier: an answer that cannot publish one refuses the decision instead.
    #[must_use]
    pub const fn of(carrier: CarrierDemand) -> Option<Self> {
        match carrier {
            CarrierDemand::Mean => None,
            CarrierDemand::ParameterPosterior => Some(Self::ParameterPosterior),
            CarrierDemand::OutcomeLaw => Some(Self::OutcomeLaw),
        }
    }
}

/// Affine score and the carrier that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "an affine score names the carrier that produced it"]
pub struct ScoredDecision {
    /// `slope * location + intercept`.
    pub value: f64,
    /// Posterior or mean. Never a confidence interval.
    pub carrier: EvidencePort,
}

/// What an executed answer published for a decision to read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedCarriers {
    /// Published point answer.
    pub mean: f64,
    /// Posterior mean of the answer, when the answer is a posterior.
    pub posterior_mean: Option<f64>,
}

/// Score `demand` from what an answer published.
///
/// An affine posterior demand reads the posterior mean; an affine mean demand
/// reads the published answer and does not need a posterior.
///
/// # Errors
///
/// A nonlinear utility or an outcome risk names [`MissingCarrier::OutcomeLaw`].
/// A posterior demand on an answer without a posterior names
/// [`MissingCarrier::ParameterPosterior`].
pub fn score_decision(
    demand: DecisionDemand,
    published: PublishedCarriers,
) -> Result<ScoredDecision, MissingCarrier> {
    match demand {
        DecisionDemand::AffinePosterior { slope, intercept } => {
            let Some(mean) = published.posterior_mean else {
                return Err(MissingCarrier::ParameterPosterior);
            };
            Ok(ScoredDecision {
                value: slope.mul_add(mean, intercept),
                carrier: EvidencePort::ParameterPosterior,
            })
        }
        DecisionDemand::AffineMean { slope, intercept } => Ok(ScoredDecision {
            value: slope.mul_add(published.mean, intercept),
            carrier: EvidencePort::Mean,
        }),
        DecisionDemand::Nonlinear | DecisionDemand::Risk => Err(MissingCarrier::OutcomeLaw),
    }
}

/// A decision demand the statistical program cannot supply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionCarrierError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: &'static str,
    /// What was refused.
    pub message: &'static str,
    /// The carrier the decision demanded.
    pub demanded: CarrierDemand,
    /// Statistical realization of the compiled route.
    pub realization: &'static str,
    /// The refused port offer, when the demand is an evidence-port request.
    pub port: Option<EvidencePortError>,
    /// What to try instead.
    pub next_attempt: NextAttempt,
}

impl DecisionCarrierError {
    /// The missing carrier, when the demand was a posterior or an outcome law.
    #[must_use]
    pub const fn missing_carrier(&self) -> Option<MissingCarrier> {
        MissingCarrier::of(self.demanded)
    }

    fn from_port(
        port: EvidencePortError,
        demanded: CarrierDemand,
        realization: &'static str,
        next_attempt: NextAttempt,
    ) -> Self {
        Self {
            code: port.code,
            detail: CARRIER_DETAIL,
            message: port.message,
            demanded,
            realization,
            port: Some(port),
            next_attempt,
        }
    }

    const fn contract(
        message: &'static str,
        demanded: CarrierDemand,
        realization: &'static str,
        next_attempt: NextAttempt,
    ) -> Self {
        Self {
            code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
            detail: CARRIER_DETAIL,
            message,
            demanded,
            realization,
            port: None,
            next_attempt,
        }
    }
}

impl fmt::Display for DecisionCarrierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}{}: {}: {}; next attempt: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message,
            self.next_attempt.sentence()
        )
    }
}

impl std::error::Error for DecisionCarrierError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.port.as_ref().map(|port| port as &(dyn std::error::Error + 'static))
    }
}

antecedent_core::impl_refused!(DecisionCarrierError, static_message);

/// Stable name of a statistical realization.
const fn realization_name(realization: &Realization) -> &'static str {
    match realization {
        Realization::Frequentist(_) => "frequentist",
        Realization::Bayesian(_) => "bayesian",
        Realization::GeneralizedBayes(_) => "generalized_bayes",
        Realization::Envelope(_) => "envelope",
        Realization::ExactPoint { .. } => "exact_point",
    }
}

/// The uncertainty port a realization publishes.
const fn offered_port(realization: &Realization) -> EvidencePort {
    match realization {
        Realization::Frequentist(_) => EvidencePort::ConfidenceRegion,
        Realization::Bayesian(_) | Realization::GeneralizedBayes(_) => {
            EvidencePort::ParameterPosterior
        }
        Realization::Envelope(_) => EvidencePort::IdentifiedSet,
        Realization::ExactPoint { .. } => EvidencePort::Mean,
    }
}

/// Whether the cell's answer is an interventional law of the outcome.
const fn publishes_interventional_law(cell: RouteCell) -> bool {
    matches!(cell, RouteCell::DistributionDag | RouteCell::DistributionMixed)
}

/// Check that `realization` on `cell` can publish what `program` demands.
///
/// # Errors
///
/// - a mean demanded of a utility that declares no affine form;
/// - a mean demanded of an identified set;
/// - a parameter posterior demanded of a sampling, exact or set realization
///   (a confidence interval is not a posterior);
/// - an outcome law demanded of a route whose answer is not an interventional law.
#[allow(
    clippy::result_large_err,
    reason = "the carrier refusal is built once at compile time and converted at once into the shared rejection"
)]
pub(crate) fn require_carrier(
    cell: RouteCell,
    realization: &Realization,
    program: &DecisionProgram,
) -> Result<(), DecisionCarrierError> {
    let demanded = program.carrier();
    let name = realization_name(realization);
    let envelope = matches!(realization, Realization::Envelope(_));
    let attempt = |otherwise: NextAttempt| {
        if envelope { ROBUST_OVER_IDENTIFIED_SET } else { otherwise }
    };
    match demanded {
        CarrierDemand::Mean if !program.utility().is_affine() => {
            Err(DecisionCarrierError::contract(
                "a mean carries only a utility affine in the state; this utility declares no \
                 affine form",
                demanded,
                name,
                DISTRIBUTIONAL_CARRIER,
            ))
        }
        CarrierDemand::Mean if envelope => Err(DecisionCarrierError::contract(
            "an identified set has no point mean for an affine utility to read",
            demanded,
            name,
            ROBUST_OVER_IDENTIFIED_SET,
        )),
        CarrierDemand::Mean => Ok(()),
        CarrierDemand::ParameterPosterior => offered_port(realization)
            .try_coerce(DecisionRequest::ParameterPosterior)
            .map(drop)
            .map_err(|port| {
                DecisionCarrierError::from_port(port, demanded, name, attempt(BAYESIAN_REALIZATION))
            }),
        CarrierDemand::OutcomeLaw if publishes_interventional_law(cell) && !envelope => Ok(()),
        CarrierDemand::OutcomeLaw => offered_port(realization)
            .try_coerce(DecisionRequest::IndividualOutcomeRisk)
            .map(drop)
            .map_err(|port| {
                DecisionCarrierError::from_port(port, demanded, name, attempt(INTERVENTIONAL_LAW))
            }),
    }
}

/// What a decision compile demands, beyond the decision problem itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionCompileRequest {
    carrier: CarrierDemand,
    contract_identity: Option<String>,
    utility_unit: Option<String>,
}

impl DecisionCompileRequest {
    /// Demand `carrier` of the statistical program.
    #[must_use]
    pub const fn new(carrier: CarrierDemand) -> Self {
        Self { carrier, contract_identity: None, utility_unit: None }
    }

    /// Name the decision contract the problem implements.
    #[must_use]
    pub fn with_contract_identity(mut self, identity: impl Into<String>) -> Self {
        self.contract_identity = Some(identity.into());
        self
    }

    /// Declare the utility's unit.
    #[must_use]
    pub fn with_utility_unit(mut self, unit: impl Into<String>) -> Self {
        self.utility_unit = Some(unit.into());
        self
    }

    /// Demanded carrier.
    #[must_use]
    pub const fn carrier(&self) -> CarrierDemand {
        self.carrier
    }

    /// Decision contract identity, when declared.
    #[must_use]
    pub fn contract_identity(&self) -> Option<&str> {
        self.contract_identity.as_deref()
    }

    /// Utility unit, when declared.
    #[must_use]
    pub fn utility_unit(&self) -> Option<&str> {
        self.utility_unit.as_deref()
    }
}

impl From<DecisionDemand> for DecisionCompileRequest {
    fn from(demand: DecisionDemand) -> Self {
        Self::new(demand.carrier())
    }
}

impl Study {
    /// Compile the study for a terminal decision over its answer.
    ///
    /// The plan is the one [`Self::compile`] returns; its lowered program's layer
    /// four carries the decision: the carrier `request` demands, the actions,
    /// constraints and chance threshold of `problem`, the utility's affine form
    /// when it declares one, the utility unit and the contract identity. A
    /// different utility, threshold or carrier gives a different lowered
    /// identity. Nothing is fitted.
    ///
    /// # Errors
    ///
    /// - `invalid_argument`: no actions, a chance threshold outside `[0, 1]`, or
    ///   a blank contract identity or utility unit;
    /// - a [`DecisionCarrierError`] (as [`CausalError::Rejected`]) when the
    ///   route's statistical realization cannot supply the demanded carrier;
    /// - any refusal of [`Self::compile`].
    pub fn compile_for_decision<A, O>(
        &self,
        problem: &DecisionProblem<A, O>,
        request: &DecisionCompileRequest,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        validate_problem(problem)?;
        let contract_identity = declared_text(request.contract_identity(), "contract identity")?;
        let utility_unit = declared_text(request.utility_unit(), "utility unit")?;
        let program = DecisionProgram {
            carrier: request.carrier(),
            utility: DecisionProgram::utility_form(affine_coefficients(problem)),
            actions: problem.actions.len(),
            constraints: problem.constraints.len(),
            chance_threshold: problem.chance_threshold,
            utility_unit,
            contract_identity,
            information_value: None,
        };
        self.compile_for_goal(&CompileGoal::Decision(program), ctx)
    }

    /// Compile the study as the state belief of a preposterior value-of-information
    /// request over `request`'s candidate studies.
    ///
    /// EVSI updates a belief about the decision state, so the demanded carrier is
    /// a parameter posterior: a sampling realization is refused and names
    /// [`BAYESIAN_REALIZATION`]. Layer four carries the decision of `problem` and
    /// the request's candidates (canonical order, bounded by `max_candidates`),
    /// prior observations, ranking schedule, cost map, seed and tolerances.
    ///
    /// Before anything is compiled, the Monte Carlo store (one score per
    /// candidate and draw at the schedule's cap, and one seed per draw) is quoted
    /// with checked arithmetic and refused when the admission pool cannot hold
    /// it. Nothing is fitted or simulated.
    ///
    /// # Errors
    ///
    /// - `invalid_argument`: action ids that do not align with `problem`'s
    ///   actions, no or repeated candidates, a non-positive search bound or
    ///   schedule, a non-finite or negative tolerance, a blank contract identity
    ///   or utility unit, or an invalid chance threshold;
    /// - [`CausalError::Resource`]: the Monte Carlo store overflows `u64` or
    ///   exceeds what the admission pool holds;
    /// - a [`DecisionCarrierError`] when the route publishes no parameter posterior;
    /// - any refusal of [`Self::compile`].
    pub fn compile_for_evsi<A, O>(
        &self,
        problem: &DecisionProblem<A, O>,
        request: &EvsiRequest,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        validate_problem(problem)?;
        let information_value = evsi_program(problem, request, ctx)?;
        let program = DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(affine_coefficients(problem)),
            actions: problem.actions.len(),
            constraints: problem.constraints.len(),
            chance_threshold: problem.chance_threshold,
            utility_unit: declared_text(Some(request.utility_unit.as_str()), "utility unit")?,
            contract_identity: declared_text(
                Some(request.decision_contract_identity.as_str()),
                "contract identity",
            )?,
            information_value: Some(information_value),
        };
        self.compile_for_goal(&CompileGoal::Evsi(program), ctx)
    }
}

/// Refuse a decision with no actions or an invalid chance threshold.
fn validate_problem<A, O>(problem: &DecisionProblem<A, O>) -> Result<(), CausalError> {
    if problem.actions.is_empty() {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the decision problem has no actions"
        ));
    }
    let threshold = problem.chance_threshold;
    if !(0.0..=1.0).contains(&threshold) {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "chance_threshold must lie in [0, 1] (got {threshold})"
        ));
    }
    Ok(())
}

/// A declared, non-blank text, or none.
fn declared_text(text: Option<&str>, what: &str) -> Result<Option<Arc<str>>, CausalError> {
    match text {
        Some(text) if is_blank(text) => {
            Err(crate::compile_reason!("invalid_argument", "the decision {what} is blank"))
        }
        Some(text) => Ok(Some(Arc::from(text))),
        None => Ok(None),
    }
}

/// Per-action `(intercept, slope)` pairs when the utility declares one finite
/// pair for every action.
fn affine_coefficients<A, O>(problem: &DecisionProblem<A, O>) -> Option<Vec<(f64, f64)>> {
    let pairs = problem.utility.affine_in_state(&problem.actions)?;
    let complete = pairs.len() == problem.actions.len()
        && pairs.iter().all(|(intercept, slope)| intercept.is_finite() && slope.is_finite());
    complete.then_some(pairs)
}

/// The value-of-information layer of `request`, with its Monte Carlo store
/// quoted and admitted against `ctx`'s pool.
fn evsi_program<A, O>(
    problem: &DecisionProblem<A, O>,
    request: &EvsiRequest,
    ctx: &ExecutionContext,
) -> Result<EvsiProgram, CausalError> {
    if request.action_ids.len() != problem.actions.len() {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the request names {} action ids for {} actions",
            request.action_ids.len(),
            problem.actions.len()
        ));
    }
    if request.max_candidates == 0 {
        return Err(crate::compile_reason!("invalid_argument", "max_candidates must be > 0"));
    }
    if !(request.mc_error_tolerance.is_finite() && request.mc_error_tolerance > 0.0) {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "mc_error_tolerance must be finite and > 0 (got {})",
            request.mc_error_tolerance
        ));
    }
    if !(request.tie_tolerance.is_finite() && request.tie_tolerance >= 0.0) {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "tie_tolerance must be finite and >= 0 (got {})",
            request.tie_tolerance
        ));
    }
    let config = &request.rank_config;
    if config.max_batches == 0 || config.batch_size == 0 {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "max_batches and batch_size must be > 0"
        ));
    }
    if !(config.rank_uncertainty_threshold.is_finite() && config.rank_uncertainty_threshold >= 0.0)
    {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "rank_uncertainty_threshold must be finite and >= 0 (got {})",
            config.rank_uncertainty_threshold
        ));
    }
    let candidates = canonical_candidates(request)?;
    let rank = EvsiProgram::schedule(
        config.min_batches,
        config.max_batches,
        config.batch_size,
        config.rank_uncertainty_threshold,
    );
    let draws = rank.draws_per_candidate();
    let monte_carlo =
        EvsiProgram::quote_tasks(candidates.len() as u64, draws).ok_or_else(|| {
            CausalError::Resource {
                message: format!(
                    "evsi Monte Carlo store of {} candidates x {draws} draws overflows u64",
                    candidates.len()
                ),
            }
        })?;
    let available = ctx.admission().available().bytes();
    if monte_carlo.total_bytes() > available {
        return Err(CausalError::Resource {
            message: format!(
                "evsi Monte Carlo store of {} candidates x {} draws needs {} bytes ({} score, {} \
                 seed) and the admission pool holds {available}; next attempt: lower max_batches \
                 or batch_size, or bound max_candidates",
                monte_carlo.candidates(),
                monte_carlo.draws(),
                monte_carlo.total_bytes(),
                monte_carlo.score_bytes(),
                monte_carlo.seed_bytes(),
            ),
        });
    }
    let prior_observation_ids = request
        .prior_observation_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(Arc::<str>::from)
        .collect::<Arc<[Arc<str>]>>();
    Ok(EvsiProgram {
        candidates,
        prior_observation_ids,
        rng_seed: request.rng_seed,
        mc_error_tolerance: request.mc_error_tolerance,
        require_net_value: request.require_net_value,
        tie_tolerance: request.tie_tolerance,
        max_candidates: request.max_candidates as u64,
        rank,
        cost_map: request.cost_map.as_ref().map(|map| {
            EvsiProgram::cost_mapping(&map.cost_unit, &map.utility_unit, map.utility_per_cost)
        }),
        monte_carlo,
    })
}

/// The candidates the evaluation scores: canonical semantic-id order, bounded by
/// the request's search bound.
fn canonical_candidates(request: &EvsiRequest) -> Result<Arc<[Arc<str>]>, CausalError> {
    if request.candidates.is_empty() {
        return Err(crate::compile_reason!("invalid_argument", "the request has no candidates"));
    }
    let mut ids = BTreeSet::new();
    for candidate in &request.candidates {
        if is_blank(&candidate.semantic_id) {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "a candidate has a blank semantic id"
            ));
        }
        if !ids.insert(candidate.semantic_id.as_str()) {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "candidate {} is declared twice",
                candidate.semantic_id
            ));
        }
    }
    Ok(ids.into_iter().take(request.max_candidates).map(Arc::<str>::from).collect())
}

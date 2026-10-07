//! Per-candidate expected value of sample information with declared integration,
//! cost mapping, rank uncertainty, provider trust and source-overlap diagnostics.
//!
//! Every candidate's signal comes from its [`SignalProvider`]. A provider that supplies
//! an observation law (native or external likelihood) is scored through the existing
//! [`DesignObjective::ReduceDecisionRegret`] path of [`DesignRanker`] (one
//! preposterior analysis per candidate; no parallel scorer), with EVPI and the current
//! Bayes action read from [`PreposteriorAnalysis`]. A provider that supplies externally
//! computed per-branch decision values is evaluated as
//! `EVSI = Σ_y P(y)·max_{a∈F} V_y(a) − max_{a∈F} E[U(a,θ)]`, after the supplied values
//! are checked against the prior (`Σ_y P(y)·V_y(a) = E[U(a,θ)]` for every action) and
//! its action set is checked against the decision's. The terminal action set `F` is the
//! same before and after the information; a changed action set is a different problem
//! and refuses.
//!
//! `NetValueOfInformation = EVSI − StudyCostUtility` is reported only when the caller
//! supplies a [`CostToUtilityMap`] whose utility unit equals the decision's and whose
//! cost unit equals every candidate's; otherwise value and cost are reported
//! separately and incompatible units refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CancellationToken, ExecutionContext, ExternalRefusal};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use crate::candidate::CandidateDesign;
use crate::decision::{DecisionProblem, DecisionProblemId, DecisionTable, evaluate_decision};
use crate::error::DesignError;
use crate::objective::DesignObjective;
use crate::preposterior::{DecisionPrior, DecisionSignal, PreposteriorAnalysis};
use crate::ranker::{DecisionRegistry, DesignEvaluationContext, DesignRankConfig, DesignRanker};
use crate::result::ScoreEvaluation;
use crate::signal::{
    ExternalDecisionValues, PreparedLaw, SignalError, SignalProvider, SignalReceipt,
    SignalTrustLabel, SignalUpdateMode, make_refusal, state_key,
};

/// Critical value for the unpaired rank-uncertainty comparison of two independently
/// evaluated candidates.
const RANK_Z: f64 = 1.96;

/// Positive linear map from the caller's cost unit to the decision's utility unit:
/// `StudyCostUtility = utility_per_cost · cost`.
#[derive(Clone, Debug, PartialEq)]
pub struct CostToUtilityMap {
    /// Unit the study costs are declared in.
    pub cost_unit: String,
    /// Utility unit of the decision this maps into.
    pub utility_unit: String,
    /// Utility per unit cost (finite, > 0).
    pub utility_per_cost: f64,
}

/// Declared cost of a candidate study.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyCostSpec {
    /// Amount (finite, >= 0).
    pub amount: f64,
    /// Unit of the amount.
    pub unit: String,
}

/// One candidate study and the provider of its signal.
#[derive(Clone)]
pub struct EvsiCandidate {
    /// Stable semantic identity of the candidate.
    pub semantic_id: String,
    /// Candidate design (its declared sample size must equal the signal request's).
    pub design: CandidateDesign,
    /// Exact signal request for this candidate.
    pub signal_request: crate::signal::SignalRequest,
    /// Provider answering the request.
    pub provider: Arc<dyn SignalProvider>,
    /// Declared study cost.
    pub cost: StudyCostSpec,
    /// Identities of existing observations the study would reuse.
    pub reused_observation_ids: Vec<String>,
}

/// One EVSI evaluation request over a finite candidate catalog.
#[derive(Clone)]
pub struct EvsiRequest {
    /// Identity of the decision contract the problem implements.
    pub decision_contract_identity: String,
    /// Utility unit of the decision.
    pub utility_unit: String,
    /// Action identities aligned with the problem's actions.
    pub action_ids: Vec<String>,
    /// Candidates (any order; evaluation is canonical by semantic id).
    pub candidates: Vec<EvsiCandidate>,
    /// Cost-to-utility mapping, if one is licensed.
    pub cost_map: Option<CostToUtilityMap>,
    /// Refuse (instead of reporting value and cost separately) when no map is supplied.
    pub require_net_value: bool,
    /// Identities of observations already summarized by the prior.
    pub prior_observation_ids: Vec<String>,
    /// Monte Carlo configuration of the ranker.
    pub rank_config: DesignRankConfig,
    /// RNG seed of the evaluation.
    pub rng_seed: u64,
    /// Standard error at or below which a Monte Carlo estimate is called converged.
    pub mc_error_tolerance: f64,
    /// Absolute gap at or below which two candidates are tied.
    pub tie_tolerance: f64,
    /// Search bound: at most this many candidates (canonical order) are evaluated.
    pub max_candidates: usize,
}

/// How the EVSI integral was evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegrationMethod {
    /// Finite enumeration or conjugate closed form.
    Exact,
    /// Monte Carlo over states and data.
    MonteCarlo,
    /// Computed by the provider; Antecedent only combined the supplied branches.
    ExternallyComputed,
}

impl IntegrationMethod {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::MonteCarlo => "monte_carlo",
            Self::ExternallyComputed => "externally_computed",
        }
    }
}

/// Integration error and convergence of one candidate's EVSI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrationReport {
    /// Method.
    pub method: IntegrationMethod,
    /// Monte Carlo replicates (zero when not Monte Carlo).
    pub replicates: u64,
    /// Standard error of the EVSI estimate (zero when not Monte Carlo).
    pub stderr: f64,
    /// Effective sample size; replicates are independent draws, so it equals the
    /// replicate count. `None` when not Monte Carlo.
    pub ess: Option<f64>,
    /// Exact methods always; Monte Carlo when `stderr` is within the tolerance.
    pub converged: bool,
    /// The ranker stopped before its batch cap.
    pub early_stopped: bool,
}

/// Overlap between the observations behind the prior and those a study reuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceOverlapDiagnostics {
    /// Distinct observation identities compared.
    pub observations_checked: usize,
    /// Shared identities; empty for every successful evaluation.
    pub overlapping: Vec<String>,
}

/// Report for one candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEvsi {
    /// Semantic identity.
    pub semantic_id: String,
    /// Expected value of sample information.
    pub evsi: f64,
    /// Integration report.
    pub integration: IntegrationReport,
    /// Expected value of perfect information (an upper bound on EVSI).
    pub evpi: f64,
    /// Sample size.
    pub sample_size: u64,
    /// Declared study cost.
    pub study_cost: StudyCostSpec,
    /// Study cost in utility units, only with a valid cost map.
    pub study_cost_utility: Option<f64>,
    /// `EVSI − StudyCostUtility`, only with a valid cost map.
    pub net_value: Option<f64>,
    /// Rank position (0 best); ties share the canonical semantic-id order.
    pub rank: usize,
    /// The gap to a neighbouring candidate is within integration error or the tie
    /// tolerance.
    pub rank_uncertain: bool,
    /// Provider trust label.
    pub provider_trust: SignalTrustLabel,
    /// Update mode.
    pub update_mode: SignalUpdateMode,
    /// Provider receipt for the exact request.
    pub signal_receipt: SignalReceipt,
    /// Source-overlap diagnostics.
    pub source_overlap: SourceOverlapDiagnostics,
    /// Assumptions the value rests on.
    pub assumptions: Vec<String>,
}

/// What the ranking is sorted by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankingBasis {
    /// EVSI (no cost mapping supplied).
    Evsi,
    /// Net value of information (valid cost mapping supplied).
    NetValue,
}

/// How much of the catalog was evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchReceipt {
    /// Candidates supplied.
    pub supplied: usize,
    /// Candidates evaluated.
    pub evaluated: usize,
    /// The bound cut the catalog; the ranking covers only the evaluated candidates.
    pub truncated: bool,
    /// Semantic ids not evaluated.
    pub unevaluated_ids: Vec<String>,
}

/// EVSI/preposterior receipt over the evaluated candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiReport {
    /// Decision contract identity.
    pub decision_contract_identity: String,
    /// Utility unit.
    pub utility_unit: String,
    /// Terminal action identities (sorted); identical before and after information.
    pub action_ids: Vec<String>,
    /// Index of the Bayes action under the current belief, in problem order.
    pub bayes_action: usize,
    /// Expected utility of the Bayes action under the current belief.
    pub prior_expected_utility: f64,
    /// Expected value of perfect information.
    pub evpi: f64,
    /// RNG seed.
    pub rng_seed: u64,
    /// Cost mapping used, if any.
    pub cost_map: Option<CostToUtilityMap>,
    /// What the candidates are sorted by.
    pub basis: RankingBasis,
    /// Candidates, best first.
    pub candidates: Vec<CandidateEvsi>,
    /// Adjacent pairs tied within the tie tolerance.
    pub ties: Vec<(String, String)>,
    /// Search receipt.
    pub search: SearchReceipt,
}

/// Why an EVSI evaluation refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvsiError {
    /// A signal request or provider answer was refused.
    Signal(SignalError),
    /// The underlying decision analysis failed.
    Design(DesignError),
    /// A request field is invalid.
    InvalidRequest(&'static str),
    /// Two candidates share a semantic id.
    DuplicateCandidate(String),
    /// A study reuses observations the prior already summarizes.
    SourceOverlap(Vec<String>),
    /// A candidate's declared sample size differs from its signal request's.
    SampleSizeMismatch(String),
    /// A provider's receipt answers a different request.
    ReceiptMismatch(String),
    /// An external law is defined on different prior draws than the decision's.
    PriorStatesMismatch(String),
    /// Externally computed decision values name a different terminal action set.
    ActionSetChanged,
    /// Externally computed decision values need a prior of draws.
    DecisionValuesNeedDraws,
    /// Externally computed decision values do not average to the prior values.
    DecisionValuesIncoherent(String),
    /// Net value was required without a cost-to-utility map.
    CostMapRequired,
    /// A cost unit is incompatible with the cost map or the map with the decision.
    CostUnitsMismatch(String),
    /// A declared study cost or map is not finite and nonnegative.
    InvalidCost(String),
    /// EVSI violates `0 <= EVSI <= EVPI` beyond its error.
    BoundViolation(&'static str),
    /// Cancellation was requested.
    Cancelled,
    /// The ranker produced no score for a candidate.
    Unscored(String),
}

impl From<DesignError> for EvsiError {
    fn from(error: DesignError) -> Self {
        Self::Design(error)
    }
}

impl From<SignalError> for EvsiError {
    fn from(error: SignalError) -> Self {
        Self::Signal(error)
    }
}

impl EvsiError {
    /// Structured refusal under the `evsi` namespace (signal refusals keep
    /// `signal_provider`).
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let units = antecedent_core::reason_code!("design_cost_units_mismatch");
        let at = |code, detail: &str| make_refusal(code, "evaluate", detail);
        let with = |code, detail: &str, offending: String| ExternalRefusal {
            offending: Some(offending),
            ..make_refusal(code, "evaluate", detail)
        };
        match self {
            Self::Signal(error) => error.to_refusal(),
            Self::Design(error) => ExternalRefusal {
                expected: Some(error.to_string()),
                ..at(signal, "evsi.design_error")
            },
            Self::InvalidRequest(what) => with(
                antecedent_core::reason_code!("invalid_argument"),
                "evsi.invalid_request",
                (*what).to_owned(),
            ),
            Self::DuplicateCandidate(id) => with(signal, "evsi.duplicate_candidate", id.clone()),
            Self::SourceOverlap(ids) => ExternalRefusal {
                remedy: Some("use only data not already summarized by the prior"),
                ..with(signal, "evsi.source_overlap", ids.join(","))
            },
            Self::SampleSizeMismatch(id) => with(signal, "evsi.sample_size_mismatch", id.clone()),
            Self::ReceiptMismatch(id) => with(signal, "evsi.receipt_mismatch", id.clone()),
            Self::PriorStatesMismatch(id) => with(signal, "evsi.prior_states_mismatch", id.clone()),
            Self::ActionSetChanged => ExternalRefusal {
                remedy: Some("an action-set change is a separate decision problem"),
                ..at(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "evsi.action_set_changed",
                )
            },
            Self::DecisionValuesNeedDraws => at(signal, "evsi.decision_values_need_draws"),
            Self::DecisionValuesIncoherent(id) => {
                with(signal, "evsi.decision_values_incoherent", id.clone())
            }
            Self::CostMapRequired => ExternalRefusal {
                remedy: Some(
                    "supply a cost-to-utility mapping or report value and cost separately",
                ),
                ..at(units, "evsi.cost_map_required")
            },
            Self::CostUnitsMismatch(what) => with(units, "evsi.cost_units_mismatch", what.clone()),
            Self::InvalidCost(what) => with(
                antecedent_core::reason_code!("invalid_argument"),
                "evsi.invalid_cost",
                what.clone(),
            ),
            Self::BoundViolation(which) => {
                with(signal, "evsi.bound_violation", (*which).to_owned())
            }
            Self::Cancelled => {
                at(antecedent_core::reason_code!("cancelled_no_claim"), "evsi.cancelled")
            }
            Self::Unscored(id) => with(signal, "evsi.unscored_candidate", id.clone()),
        }
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn validate_costs(request: &EvsiRequest) -> Result<(), EvsiError> {
    if let Some(map) = &request.cost_map {
        if blank(&map.cost_unit)
            || blank(&map.utility_unit)
            || !(map.utility_per_cost.is_finite() && map.utility_per_cost > 0.0)
        {
            return Err(EvsiError::InvalidCost("cost_map".into()));
        }
        if map.utility_unit != request.utility_unit {
            return Err(EvsiError::CostUnitsMismatch(format!(
                "map utility unit {} differs from decision utility unit {}",
                map.utility_unit, request.utility_unit
            )));
        }
    } else if request.require_net_value {
        return Err(EvsiError::CostMapRequired);
    }
    for candidate in &request.candidates {
        let cost = &candidate.cost;
        if !(cost.amount.is_finite() && cost.amount >= 0.0) || blank(&cost.unit) {
            return Err(EvsiError::InvalidCost(candidate.semantic_id.clone()));
        }
        if let Some(map) = &request.cost_map {
            if cost.unit != map.cost_unit {
                return Err(EvsiError::CostUnitsMismatch(format!(
                    "candidate {} cost unit {} differs from map cost unit {}",
                    candidate.semantic_id, cost.unit, map.cost_unit
                )));
            }
        }
    }
    Ok(())
}

fn validate_request<A>(
    problem: &DecisionProblem<A, f64>,
    request: &EvsiRequest,
) -> Result<(), EvsiError> {
    if blank(&request.decision_contract_identity) || blank(&request.utility_unit) {
        return Err(EvsiError::InvalidRequest("decision identity and utility unit"));
    }
    if request.action_ids.len() != problem.actions.len()
        || request.action_ids.iter().any(|id| blank(id))
    {
        return Err(EvsiError::InvalidRequest("action_ids must align with the problem actions"));
    }
    let mut ids: Vec<&String> = request.action_ids.iter().collect();
    ids.sort();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(EvsiError::InvalidRequest("duplicate action id"));
    }
    if request.candidates.is_empty() || request.max_candidates == 0 {
        return Err(EvsiError::InvalidRequest("candidates and max_candidates"));
    }
    if !(request.mc_error_tolerance.is_finite()
        && request.mc_error_tolerance >= 0.0
        && request.tie_tolerance.is_finite()
        && request.tie_tolerance >= 0.0)
    {
        return Err(EvsiError::InvalidRequest("tolerances"));
    }
    let mut seen: Vec<&String> = request.candidates.iter().map(|c| &c.semantic_id).collect();
    seen.sort();
    if let Some(pair) = seen.windows(2).find(|w| w[0] == w[1]) {
        return Err(EvsiError::DuplicateCandidate(pair[0].clone()));
    }
    if request.candidates.iter().any(|c| blank(&c.semantic_id)) {
        return Err(EvsiError::InvalidRequest("blank candidate id"));
    }
    validate_costs(request)
}

const fn declared_sample_size(design: &CandidateDesign) -> Option<u64> {
    match design {
        CandidateDesign::IncreaseSamplingRate(plan) => Some(plan.additional_samples),
        CandidateDesign::ObserveEnvironment(plan) => Some(plan.additional_rows),
        CandidateDesign::Measure(_) | CandidateDesign::Intervene(_) => None,
    }
}

fn overlap_of(request: &EvsiRequest, candidate: &EvsiCandidate) -> SourceOverlapDiagnostics {
    let prior: std::collections::BTreeSet<&str> =
        request.prior_observation_ids.iter().map(String::as_str).collect();
    let reused: std::collections::BTreeSet<&str> =
        candidate.reused_observation_ids.iter().map(String::as_str).collect();
    SourceOverlapDiagnostics {
        observations_checked: prior.union(&reused).count(),
        overlapping: prior.intersection(&reused).map(|s| (*s).to_owned()).collect(),
    }
}

/// Numbers one candidate's evaluation produced.
struct Evaluated {
    evsi: f64,
    integration: IntegrationReport,
    evpi: f64,
    prior_expected_utility: f64,
    bayes_action: usize,
    receipt: SignalReceipt,
}

fn likelihood_value<A: Clone>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    signal: &Arc<dyn DecisionSignal<f64>>,
    candidate: &EvsiCandidate,
    request: &EvsiRequest,
    ctx: &ExecutionContext,
) -> Result<(f64, IntegrationReport, f64, f64, usize), EvsiError> {
    let registry = DecisionRegistry {
        problems: vec![Some(DecisionProblem {
            actions: problem.actions.clone(),
            utility: Arc::clone(&problem.utility),
            constraints: problem.constraints.clone(),
            chance_threshold: problem.chance_threshold,
        })],
        prior: prior.clone(),
        signal: Arc::clone(signal),
    };
    let graphs =
        WeightedGraphSamples::new(vec![1.0], vec![GraphIdentFlag::Identified], vec![0_u64])
            .map_err(|e| DesignError::Prob(format!("{e:?}")))?;
    let eval = DesignEvaluationContext {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: Some(&registry),
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let ranking = DesignRanker::new().with_config(request.rank_config.clone()).rank(
        &DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) },
        std::slice::from_ref(&candidate.design),
        &eval,
        ctx,
    )?;
    let top =
        ranking.ranked.first().ok_or_else(|| EvsiError::Unscored(candidate.semantic_id.clone()))?;
    let analysis = PreposteriorAnalysis::new(problem, prior, signal.as_ref())?;
    let integration = match top.evaluation {
        ScoreEvaluation::Exact => IntegrationReport {
            method: IntegrationMethod::Exact,
            replicates: 0,
            stderr: 0.0,
            ess: None,
            converged: true,
            early_stopped: ranking.early_stopped,
        },
        ScoreEvaluation::MonteCarlo => IntegrationReport {
            method: IntegrationMethod::MonteCarlo,
            replicates: top.monte_carlo.samples,
            stderr: top.monte_carlo.stderr,
            ess: Some(top.monte_carlo.samples as f64),
            converged: top.monte_carlo.stderr <= request.mc_error_tolerance,
            early_stopped: ranking.early_stopped,
        },
    };
    Ok((
        top.score,
        integration,
        analysis.expected_value_of_perfect_information(),
        analysis.prior_expected_utility(),
        analysis.bayes_action(),
    ))
}

fn values_value<A>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    values: &ExternalDecisionValues,
    request: &EvsiRequest,
    id: &str,
) -> Result<(f64, IntegrationReport, f64, f64, usize), EvsiError> {
    let DecisionPrior::Draws(states) = prior else {
        return Err(EvsiError::DecisionValuesNeedDraws);
    };
    let columns: Vec<usize> = request
        .action_ids
        .iter()
        .map(|action| values.action_ids.iter().position(|v| v == action))
        .collect::<Option<Vec<_>>>()
        .ok_or(EvsiError::ActionSetChanged)?;
    if values.action_ids.len() != request.action_ids.len() {
        return Err(EvsiError::ActionSetChanged);
    }
    let table = DecisionTable::build(problem, states)?;
    let n_actions = problem.actions.len();
    let prior_eu: Vec<f64> = (0..n_actions)
        .map(|a| {
            (0..table.n_outcomes).map(|k| table.utility(a, k)).sum::<f64>()
                / table.n_outcomes as f64
        })
        .collect();
    let branch_value = |branch: usize, action: usize| values.values[branch][columns[action]];
    for (action, &prior) in prior_eu.iter().enumerate() {
        let averaged: f64 = values
            .branch_probabilities
            .iter()
            .enumerate()
            .map(|(y, p)| p * branch_value(y, action))
            .sum();
        if (averaged - prior).abs() > 1e-8 * (1.0 + prior.abs()) {
            return Err(EvsiError::DecisionValuesIncoherent(id.to_owned()));
        }
    }
    let after: f64 = values
        .branch_probabilities
        .iter()
        .enumerate()
        .map(|(y, p)| {
            let best = table
                .admissible
                .iter()
                .map(|&a| branch_value(y, a))
                .fold(f64::NEG_INFINITY, f64::max);
            p * best
        })
        .sum();
    let (bayes_action, prior_best) = table.bayes_action();
    let integration = IntegrationReport {
        method: IntegrationMethod::ExternallyComputed,
        replicates: 0,
        stderr: 0.0,
        ess: None,
        converged: true,
        early_stopped: false,
    };
    let evpi = evaluate_decision(problem, states)?.posterior_regret;
    Ok((after - prior_best, integration, evpi, prior_best, bayes_action))
}

fn evaluate_candidate<A: Clone>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    candidate: &EvsiCandidate,
    ctx: &ExecutionContext,
) -> Result<Evaluated, EvsiError> {
    let id = &candidate.semantic_id;
    if declared_sample_size(&candidate.design)
        .is_some_and(|n| n != candidate.signal_request.sample_size)
    {
        return Err(EvsiError::SampleSizeMismatch(id.clone()));
    }
    let overlap = overlap_of(request, candidate);
    if !overlap.overlapping.is_empty() {
        return Err(EvsiError::SourceOverlap(overlap.overlapping));
    }
    let prepared = candidate.provider.prepare(&candidate.signal_request)?;
    if prepared.receipt.request_fingerprint != candidate.signal_request.fingerprint() {
        return Err(EvsiError::ReceiptMismatch(id.clone()));
    }
    if let Some(bound) = &prepared.bound_states {
        let aligned = match prior {
            DecisionPrior::Draws(states) => {
                states.len() == bound.len()
                    && states.iter().zip(bound).all(|(a, b)| state_key(*a) == state_key(*b))
            }
            DecisionPrior::Normal { .. } => false,
        };
        if !aligned {
            return Err(EvsiError::PriorStatesMismatch(id.clone()));
        }
    }
    let (evsi, integration, evpi, prior_expected_utility, bayes_action) = match &prepared.law {
        PreparedLaw::Likelihood(signal) => {
            likelihood_value(problem, prior, signal, candidate, request, ctx)?
        }
        PreparedLaw::DecisionValues(values) => values_value(problem, prior, values, request, id)?,
    };
    let tolerance = 1e-9 + 4.0 * integration.stderr;
    if evsi < -tolerance {
        return Err(EvsiError::BoundViolation("negative_evsi"));
    }
    if evsi > evpi + tolerance {
        return Err(EvsiError::BoundViolation("exceeds_evpi"));
    }
    Ok(Evaluated {
        evsi,
        integration,
        evpi,
        prior_expected_utility,
        bayes_action,
        receipt: prepared.receipt,
    })
}

fn assumptions(
    request: &EvsiRequest,
    prior: &DecisionPrior<f64>,
    evaluated: &Evaluated,
) -> Vec<String> {
    let mut out = vec![
        format!("conditional_independence:{}", evaluated.receipt.conditional_independence),
        "terminal_action_set_identical_before_and_after_information".to_owned(),
        format!("utility_units:{}", request.utility_unit),
        format!("provider_trust:{}", evaluated.receipt.trust.as_str()),
        format!("update_mode:{}", evaluated.receipt.update_mode.as_str()),
        format!("integration:{}", evaluated.integration.method.as_str()),
        match prior {
            DecisionPrior::Draws(_) => "prior:equally_weighted_draws".to_owned(),
            DecisionPrior::Normal { .. } => "prior:normal_conjugate".to_owned(),
        },
    ];
    if evaluated.receipt.update_mode.externally_computed() {
        out.push("update_computed_externally_not_verified_by_antecedent".to_owned());
    }
    out
}

/// Evaluate EVSI for every candidate (canonical semantic-id order, bounded by
/// `max_candidates`) and rank them by net value when a valid cost mapping is
/// supplied, otherwise by EVSI.
///
/// The ranking depends only on the candidate set, not the order it was supplied in.
///
/// # Errors
///
/// Invalid requests, signal refusals, source overlap, sample-size or action-set
/// mismatches, incoherent external values, cost-unit mismatches, bound violations,
/// cancellation, and decision-analysis failures.
pub fn evaluate_evsi<A: Clone>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    cancellation: &CancellationToken,
) -> Result<EvsiReport, EvsiError> {
    validate_request(problem, request)?;
    let mut ordered: Vec<&EvsiCandidate> = request.candidates.iter().collect();
    ordered.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
    let supplied = ordered.len();
    let unevaluated_ids: Vec<String> =
        ordered.iter().skip(request.max_candidates).map(|c| c.semantic_id.clone()).collect();
    ordered.truncate(request.max_candidates);
    let mut ctx = ExecutionContext::production(request.rng_seed, 1);
    ctx.cancellation = cancellation.clone();
    let mut evaluated = Vec::with_capacity(ordered.len());
    for candidate in &ordered {
        if cancellation.is_cancelled() {
            return Err(EvsiError::Cancelled);
        }
        evaluated.push(evaluate_candidate(problem, prior, request, candidate, &ctx)?);
    }
    let search = SearchReceipt {
        supplied,
        evaluated: ordered.len(),
        truncated: !unevaluated_ids.is_empty(),
        unevaluated_ids,
    };
    Ok(assemble(prior, request, &ordered, &evaluated, search))
}

fn assemble(
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    ordered: &[&EvsiCandidate],
    evaluated: &[Evaluated],
    search: SearchReceipt,
) -> EvsiReport {
    let map = request.cost_map.as_ref();
    let mut rows: Vec<CandidateEvsi> = ordered
        .iter()
        .zip(evaluated)
        .map(|(candidate, e)| {
            let cost_utility = map.map(|m| m.utility_per_cost * candidate.cost.amount);
            CandidateEvsi {
                semantic_id: candidate.semantic_id.clone(),
                evsi: e.evsi,
                integration: e.integration,
                evpi: e.evpi,
                sample_size: candidate.signal_request.sample_size,
                study_cost: candidate.cost.clone(),
                study_cost_utility: cost_utility,
                net_value: cost_utility.map(|c| e.evsi - c),
                rank: 0,
                rank_uncertain: false,
                provider_trust: e.receipt.trust,
                update_mode: e.receipt.update_mode,
                signal_receipt: e.receipt.clone(),
                source_overlap: overlap_of(request, candidate),
                assumptions: assumptions(request, prior, e),
            }
        })
        .collect();
    let key = |row: &CandidateEvsi| row.net_value.unwrap_or(row.evsi);
    rows.sort_by(|a, b| key(b).total_cmp(&key(a)).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
    let mut ties = Vec::new();
    let mut close = vec![false; rows.len()];
    for i in 0..rows.len().saturating_sub(1) {
        let gap = (key(&rows[i]) - key(&rows[i + 1])).abs();
        let se = rows[i].integration.stderr.hypot(rows[i + 1].integration.stderr);
        if gap <= request.tie_tolerance {
            ties.push((rows[i].semantic_id.clone(), rows[i + 1].semantic_id.clone()));
        }
        if gap <= request.tie_tolerance.max(RANK_Z * se) {
            close[i] = true;
            close[i + 1] = true;
        }
    }
    for (i, row) in rows.iter_mut().enumerate() {
        row.rank = i;
        row.rank_uncertain = close[i];
    }
    let first = evaluated.first();
    EvsiReport {
        decision_contract_identity: request.decision_contract_identity.clone(),
        utility_unit: request.utility_unit.clone(),
        action_ids: {
            let mut ids = request.action_ids.clone();
            ids.sort();
            ids
        },
        bayes_action: first.map_or(0, |e| e.bayes_action),
        prior_expected_utility: first.map_or(0.0, |e| e.prior_expected_utility),
        evpi: first.map_or(0.0, |e| e.evpi),
        rng_seed: request.rng_seed,
        cost_map: request.cost_map.clone(),
        basis: if map.is_some() { RankingBasis::NetValue } else { RankingBasis::Evsi },
        candidates: rows,
        ties,
        search,
    }
}

//! Experiment and measurement design: which study, measurement or policy to run next.
//!
//! Use this module when the question is what to do next rather than what the current data
//! estimate: rank candidate designs by how much they would raise the probability that a query
//! is identified, plan experiments and sample sizes, search adaptive policy families, and
//! evaluate decisions with their reliability and robustness records. Ranking is advisory: it
//! runs no analysis.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ObligationKind, ObligationRecord, ObligationScope};
use antecedent_core::{
    AssumptionSource, AssumptionStatus, ExecutionContext, MonteCarloBudget, QueryId,
    SemanticDigest, VariableId,
};

use crate::analysis::CausalContract;
use crate::error::CausalError;
use crate::support::CellStatus;

/// Experiment planning types; [`crate::analysis::planning`] compiles and
/// executes a plan. A plan's `experiment_candidate` and `sampling_candidate`
/// feed [`rank_designs_bound`] with the planned sample size.
pub use crate::analysis::planning::{PlannedExperiment, PlanningPlan, STANDALONE_PLANNING};
/// Bounded search over history-dependent policy families (schedule, threshold, piecewise,
/// linear feedback, tree) against a response source, with path constraints, held-out evaluation
/// and a replayable record. A fitted causal model is a response source through
/// [`crate::gcm::ModelPolicySource`].
pub use antecedent_design::adaptive_policy::{
    AdaptivePolicyError, AdaptivePolicyFamily, ParameterBound, PolicyFeature, PolicyRule,
};
pub use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, AdaptivePolicySearchResult, ChanceBound, ConstraintReport,
    DrawSplit, FamilySearchResult, HeldOutDesign, HeldOutRequirement, PairedContrast,
    PolicyAttempt, PolicyConstraints, PolicyEvaluation, PolicyObjective, PolicyPath,
    PolicyRolloutSource, PolicySearchOptions, PolicyStanding, PreparedAdaptivePolicySearch,
    ResponseStanding, SearchStop, StandingRequest,
};
pub use antecedent_design::planning::{
    AnalyticCharacteristics, ArmComparison, FactorialContrast, FamilyCharacteristics, FamilyPower,
    FamilySimulation, InterferenceExposure, Multiplicity, NormalPrior, OutcomeModel,
    PLANNING_NEXT_ATTEMPTS, PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal,
    PlanningRefusal, PlanningRequest, PlanningResult, SequentialCharacteristics,
    SequentialSimulation, SimulatedCharacteristics, SimulationSpec, StratumSpec, SuccessCriterion,
    TestSidedness,
};
pub use antecedent_design::{
    AffineUtility, BinomialSignal, CandidateDesign, ConstraintViolation, DecisionConstraint,
    DecisionEvaluation, DecisionPrior, DecisionProblem, DecisionProblemId, DecisionRegistry,
    DecisionSignal, DesignConstraints, DesignCost, DesignError, DesignEvaluationContext,
    DesignObjective, DesignRankConfig, DesignRanker, EffectWidthContext, EnvironmentGramSpec,
    EnvironmentPlan, ExperimentPlan, GaussianMeanSignal, InterventionDesignEffect,
    MeasureColumnSpec, MeasurementPlan, ModelLoglikDraws, PreposteriorAnalysis, RankedCandidate,
    SamplingPlan, ScoreEvaluation, ScoredDesignRanking, Utility,
};
/// Candidate studies generated from verified obligations, the plan of options that
/// close an identification gap, and the derived structure of a graph posterior.
pub use antecedent_design::{
    CandidateGeneration, CandidateTemplate, ConfounderPriorDeclaration, ConfounderValueFrame,
    DurableStudyCandidate, EvidencePlan, EvidencePlanArtifact, EvidencePlanArtifactError,
    GraphPosteriorSample, GridUtility, PlanOption, PlanOutcome, PlanValue, PlanValueBasis,
    RepairReport, StudyDeclarations, StudyKind, UnmatchedObligation, UnverifiedStudy,
    VerifiedUnlocks, generate_candidates, mutilate, verified_unlocks,
};
use antecedent_design::{ObjectiveChannel, ObjectiveScorer};
/// Study planning over the mz-transport and mixed-source catalogs: declared
/// candidate studies, the bounded cost-ordered plan, arrival and the portable
/// `study_plan_v1` artifact.
pub use antecedent_design::{
    StudyArrival, StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlan,
    StudyPlanArtifactWire, StudyPlanConsumeLimits, StudyPlanError, StudyPlanLimits,
    StudyPlanProposal, StudyPlanResult, StudyPlanRoute, StudyPlanStop, StudyProposal,
    StudySubsetOutcome,
};
pub use antecedent_stats::adaptive_inference::group_sequential::SpendingFunction;

/// The posterior with each structure flagged by running the ID algorithm on it.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::structure_posterior`].
///
/// # Errors
/// An empty or invalid posterior (a non-finite, negative or all-zero weight), or
/// any refusal of `identifies`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled StructureIdentification route, which calls this kernel function with the same inputs"
)]
pub fn structure_posterior(
    samples: &[antecedent_design::GraphPosteriorSample],
    treatment: antecedent_core::ids::VariableId,
    outcome: antecedent_core::ids::VariableId,
) -> Result<
    antecedent_prob::graph_samples::WeightedGraphSamples,
    antecedent_design::StructureIdentificationError,
> {
    antecedent_design::structure_posterior(samples, treatment, outcome)
}

/// Compile the declared candidates and plan the cheapest subset of at most
/// three that would make the failed mz-transport or mixed-source decision identify, under one
/// shared search budget (see `antecedent_identify::plan_study_additions`).
/// The result is invariant to candidate and source declaration order.
///
/// The declared bounds (plan limits, at most 16 candidates, at most 8 level
/// combinations per candidate) are checked before any candidate is compiled.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::plan_studies`].
///
/// # Errors
/// Every `StudyPlanRefusal` of the planner, plus
/// `study_plan.invalid_candidate` for a declaration that does not compile
/// (empty or overlapping margin, a variable outside the graph, levels that do
/// not assign every intervention or leave the declared domain, no recruitment
/// declaration, or a level label `id#k` that a base regime already uses).
#[deprecated(
    since = "3.1.0",
    note = "use the compiled StudyAdditionPlanning route, which calls this kernel function with the same inputs"
)]
pub fn plan_studies(
    graph: &antecedent_graph::admg::Admg,
    route: &antecedent_identify::StudyPlanRoute,
    catalog: &antecedent_core::EvidenceCatalog,
    candidates: &[antecedent_design::StudyCandidate],
    limits: antecedent_identify::StudyPlanLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_design::StudyPlanResult, antecedent_design::StudyPlanError> {
    antecedent_design::plan_studies(graph, route, catalog, candidates, limits, ctx)
}

/// Per-structure identification after an experiment that sets `targets`: the ID
/// algorithm on each structure's `mutilate`d graph. This is the flag vector the
/// ranker's `identified_under_intervention` expects.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::identified_under_intervention`].
///
/// # Errors
/// As `structure_posterior`, and an unknown target.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled StructureIdentification route, which calls this kernel function with the same inputs"
)]
pub fn identified_under_intervention(
    samples: &[antecedent_design::GraphPosteriorSample],
    treatment: antecedent_core::ids::VariableId,
    outcome: antecedent_core::ids::VariableId,
    targets: &[antecedent_core::ids::VariableId],
) -> Result<
    Vec<antecedent_prob::graph_samples::GraphIdentFlag>,
    antecedent_design::StructureIdentificationError,
> {
    antecedent_design::identified_under_intervention(samples, treatment, outcome, targets)
}

/// Evaluate the Bayes action and its expected regret under equally weighted draws of
/// the decision state.
///
/// Constraints define the admissible actions (minimum satisfaction plus `1e-12` at
/// least `chance_threshold`); inadmissible actions are excluded from both the choice
/// and the perfect-information benchmark.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::evaluate_decision`].
///
/// # Errors
///
/// Utility callback failures, empty actions or draws, and
/// `DesignError::NoAdmissibleAction` when no action meets the threshold (there is
/// no decision to value, so no number is fabricated).
#[deprecated(
    since = "3.1.0",
    note = "use the compiled FiniteActionDecision route, which calls this kernel function with the same inputs"
)]
pub fn evaluate_decision<A, O>(
    problem: &antecedent_design::DecisionProblem<A, O>,
    outcomes: &[O],
) -> Result<antecedent_design::DecisionEvaluation, antecedent_design::DesignError> {
    antecedent_design::evaluate_decision(problem, outcomes)
}

/// Obligations inspected before the existing ranker runs.
///
/// This is not a second ranker. Unsupported candidates become typed violations,
/// not proxy scores.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignRankPreview {
    /// Required posterior / model / mapping / update-rule obligations.
    pub obligations: Arc<[ObligationRecord]>,
    /// Unidentified mass on the declared structural distribution, when present.
    pub unidentified_mass: Option<f64>,
    /// Prepared target identity the ranking is bound to.
    pub target: SemanticDigest,
    /// Candidates that must not be scored.
    pub violations: Arc<[ConstraintViolation]>,
    /// Required utility cannot be evaluated; ranking is incomplete.
    pub unresolved: bool,
}

/// Rank candidate designs under an objective, with no contract binding.
///
/// Crate-internal: the public entries are [`preview_design_rank`] and
/// [`rank_designs_bound`], which refuse an unsupported or unprepared contract and
/// report the candidates the objective cannot score before this ranker runs.
///
/// # Errors
///
/// Propagates design evaluation failures.
pub(crate) fn rank_designs<A, O>(
    ranker: &DesignRanker,
    objective: &DesignObjective,
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
) -> Result<ScoredDesignRanking, CausalError>
where
    A: Clone + Sync,
    O: Clone + Sync,
{
    ranker.rank(objective, candidates, eval, ctx).map_err(CausalError::from)
}

/// Inspect ranking obligations against a prepared contract. Does not score.
///
/// # Errors
///
/// Unsupported or unprepared contracts, or a width ranking across incomparable
/// estimands without a common decision utility.
pub fn preview_design_rank<A, O>(
    contract: &CausalContract,
    objective: &DesignObjective,
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
    decision_target: Option<SemanticDigest>,
) -> Result<DesignRankPreview, CausalError> {
    require_rankable_contract(contract)?;
    if width_targets_incomparable(objective, contract, decision_target) {
        return Err(CausalError::Unsupported {
            message: crate::error::RANK_DESIGNS_INCOMPARABLE_TARGETS,
        });
    }
    if id_mass_renormalized(objective, contract, eval) {
        return Err(CausalError::Unsupported {
            message: crate::error::RANK_DESIGNS_RENORMALIZED_MASS,
        });
    }
    Ok(inspect_rank_binding(contract, objective, candidates, eval))
}

/// Rank through a prepared contract after inspecting obligations.
///
/// # Errors
///
/// Unsupported/unprepared contracts, incomparable width targets, or ranker failure.
pub fn rank_designs_bound<A, O>(
    contract: &CausalContract,
    ranker: &DesignRanker,
    objective: &DesignObjective,
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
    decision_target: Option<SemanticDigest>,
) -> Result<ScoredDesignRanking, CausalError>
where
    A: Clone + Sync,
    O: Clone + Sync,
{
    let preview = preview_design_rank(contract, objective, candidates, eval, decision_target)?;
    if preview.unresolved {
        return Ok(incomplete_ranking(&preview.violations));
    }
    let (active, index_map) = active_candidates(candidates, &preview.violations);
    if active.is_empty() {
        return Ok(incomplete_ranking(&preview.violations));
    }
    let ranking = rank_designs(ranker, objective, &active, eval, ctx)?;
    Ok(remap_ranking(ranking, &index_map, &preview.violations))
}

fn require_rankable_contract(contract: &CausalContract) -> Result<(), CausalError> {
    if !matches!(contract.support_status, Some(CellStatus::Supported)) {
        return Err(CausalError::Unsupported {
            message: crate::error::RANK_DESIGNS_REQUIRES_SUPPORT,
        });
    }
    if contract.identities.identification_product.is_none() {
        return Err(CausalError::Unsupported {
            message: crate::error::RANK_DESIGNS_REQUIRES_PRODUCT,
        });
    }
    Ok(())
}

fn width_targets_incomparable(
    objective: &DesignObjective,
    contract: &CausalContract,
    decision_target: Option<SemanticDigest>,
) -> bool {
    matches!(objective.channel(), ObjectiveChannel::EffectWidth { .. })
        && decision_target.is_some_and(|target| target != contract.identities.target)
}

/// Unidentified share of the ensemble: graph weights need not be normalized, a mass is a share.
fn normalized_unidentified_mass(graphs: &antecedent_prob::WeightedGraphSamples) -> f64 {
    graphs.unidentified_mass() / graphs.total_weight()
}

fn id_mass_renormalized<A, O>(
    objective: &DesignObjective,
    contract: &CausalContract,
    eval: &DesignEvaluationContext<'_, A, O>,
) -> bool {
    if !matches!(objective.channel(), ObjectiveChannel::Identification { .. }) {
        return false;
    }
    let Some(slot) = contract.reasoning.identification.as_ref() else {
        return false;
    };
    slot.unidentified_mass > 0.0
        && normalized_unidentified_mass(eval.graphs) + 1e-12 < slot.unidentified_mass
}

fn inspect_rank_binding<A, O>(
    contract: &CausalContract,
    objective: &DesignObjective,
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
) -> DesignRankPreview {
    let mut obligations = vec![
        rank_obligation(
            "design.posterior_or_envelope",
            "declared_structural_distribution",
            "ranking requires a declared posterior or envelope; it is not implied by a point estimate",
        ),
        rank_obligation(
            "design.hypothetical_data_model",
            "predictive_or_gram_model",
            "a posterior interval is not a predictive model of what an experiment will teach",
        ),
        rank_obligation(
            "design.candidate_mapping",
            "candidate_to_design_map",
            "each candidate must map onto a supported add-units, extend-series, pulse-window, or measure-covariate contract",
        ),
        rank_obligation(
            "design.cost_constraints",
            "declared_costs",
            "costs and hard constraints are declared on the existing ranker, not invented here",
        ),
        rank_obligation(
            "design.update_rule",
            objective.implemented_functional(),
            "the scored functional is the ranker's implemented update, not the historical public name",
        ),
    ];
    let mut violations = Vec::new();
    let mut unresolved = false;
    match objective.channel() {
        ObjectiveChannel::EffectWidth { .. } => {
            if eval.effect_width.is_none() {
                unresolved = true;
                obligations.push(rank_obligation(
                    "design.effect_width_model",
                    "effect_width",
                    "width ranking is unresolved without a declared Gram / hypothetical-data model",
                ));
                push_all_unresolved(
                    candidates,
                    &mut violations,
                    "unresolved_utility",
                    "missing_effect_width_model",
                );
            } else {
                collect_width_violations(candidates, eval.effect_width, &mut violations);
            }
        }
        ObjectiveChannel::Identification { query } => {
            if eval.graphs.n_samples == 0 {
                unresolved = true;
                push_all_unresolved(
                    candidates,
                    &mut violations,
                    "unresolved_utility",
                    "missing_structural_distribution",
                );
            } else {
                collect_id_violations(candidates, eval, query, &mut violations);
            }
        }
        ObjectiveChannel::GraphEntropy => {
            if eval.graphs.n_samples == 0 {
                unresolved = true;
                push_all_unresolved(
                    candidates,
                    &mut violations,
                    "unresolved_utility",
                    "missing_structural_distribution",
                );
            }
        }
        ObjectiveChannel::ModelDistinction { .. } if eval.model_loglik.is_none() => {
            unresolved = true;
            push_all_unresolved(
                candidates,
                &mut violations,
                "unresolved_utility",
                "missing_model_loglik",
            );
        }
        ObjectiveChannel::Decision { .. } => {
            let signal = eval.decisions.map(|registry| registry.signal.as_ref());
            unresolved = collect_decision_violations(candidates, signal, &mut violations);
        }
        ObjectiveChannel::ModelDistinction { .. } => {}
    }
    DesignRankPreview {
        obligations: obligations.into(),
        unidentified_mass: (eval.graphs.n_samples > 0)
            .then(|| normalized_unidentified_mass(eval.graphs)),
        target: contract.identities.target,
        violations: violations.into(),
        unresolved,
    }
}

fn rank_obligation(
    id: &'static str,
    check: &'static str,
    message: &'static str,
) -> ObligationRecord {
    ObligationRecord::new(
        id,
        ObligationScope::Program,
        AssumptionSource::UserDeclared,
        ObligationKind::CheckNotRun,
        AssumptionStatus::Declared,
        message,
    )
    .with_required_check(check)
}

fn collect_width_violations(
    candidates: &[CandidateDesign],
    effect_width: Option<&EffectWidthContext>,
    violations: &mut Vec<ConstraintViolation>,
) {
    for (index, candidate) in candidates.iter().enumerate() {
        let detail = match candidate {
            CandidateDesign::IncreaseSamplingRate(_) => None,
            CandidateDesign::ObserveEnvironment(_)
                if effect_width.and_then(|ctx| ctx.environment_grams.as_ref()).is_some() =>
            {
                None
            }
            CandidateDesign::Measure(_)
                if effect_width.and_then(|ctx| ctx.measure_columns.as_ref()).is_some() =>
            {
                None
            }
            CandidateDesign::Intervene(_)
                if effect_width.and_then(|ctx| ctx.intervention_design.as_ref()).is_some() =>
            {
                None
            }
            CandidateDesign::Measure(_) => {
                Some("measure_covariate requires a declared Gram update")
            }
            CandidateDesign::ObserveEnvironment(_) => {
                Some("extend-series width requires environment grams")
            }
            CandidateDesign::Intervene(_) => {
                Some("pulse-window / intervention width requires a declared intervention design")
            }
        };
        if let Some(detail) = detail {
            violations.push(violation(index, "unsupported_candidate", detail));
        }
    }
}

fn collect_id_violations<A, O>(
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
    query: antecedent_core::QueryId,
    violations: &mut Vec<ConstraintViolation>,
) {
    let unlock = eval
        .query_id_unlock
        .and_then(|map| map.iter().find(|(id, _)| *id == query).map(|(_, vars)| vars.as_ref()))
        .unwrap_or(&[]);
    let env_unlock = eval
        .env_id_unlock
        .and_then(|map| map.iter().find(|(id, _)| *id == query).map(|(_, vars)| vars.as_ref()))
        .unwrap_or(&[]);
    let intervene = eval
        .identified_under_intervention
        .is_some_and(|flags| flags.len() == eval.graphs.n_samples);
    for (index, candidate) in candidates.iter().enumerate() {
        let detail = match candidate {
            CandidateDesign::IncreaseSamplingRate(_) => {
                Some("more observational units do not solve structural non-identification")
            }
            CandidateDesign::Measure(_) if unlock.is_empty() => {
                Some("measure_covariate requires a supported identification update")
            }
            CandidateDesign::Intervene(_) if unlock.is_empty() && !intervene => {
                Some("pulse-window / intervention requires a supported identification update")
            }
            CandidateDesign::ObserveEnvironment(_) if env_unlock.is_empty() => {
                Some("extend-series identification requires a declared environment unlock")
            }
            _ => None,
        };
        if let Some(detail) = detail {
            violations.push(violation(index, "unsupported_candidate", detail));
        }
    }
}

/// Without a decision registry the ranking is unresolved (returns `true`). With one,
/// a candidate the signal assigns no sample size has no data model for the
/// value-of-information update and is unsupported; the ranker would not score it.
fn collect_decision_violations<O>(
    candidates: &[CandidateDesign],
    signal: Option<&dyn DecisionSignal<O>>,
    violations: &mut Vec<ConstraintViolation>,
) -> bool {
    let Some(signal) = signal else {
        push_all_unresolved(
            candidates,
            violations,
            "unresolved_utility",
            "missing_decision_registry",
        );
        return true;
    };
    for (index, candidate) in candidates.iter().enumerate() {
        if signal.sample_size(candidate).is_none() {
            violations.push(violation(
                index,
                "unsupported_candidate",
                "decision signal declares no sample size for this candidate",
            ));
        }
    }
    false
}

fn push_all_unresolved(
    candidates: &[CandidateDesign],
    violations: &mut Vec<ConstraintViolation>,
    constraint: &'static str,
    detail: &'static str,
) {
    for index in 0..candidates.len() {
        violations.push(violation(index, constraint, detail));
    }
}

fn violation(index: usize, constraint: &'static str, detail: &'static str) -> ConstraintViolation {
    ConstraintViolation {
        candidate_index: index,
        constraint: Arc::from(constraint),
        detail: Arc::from(detail),
    }
}

fn incomplete_ranking(violations: &[ConstraintViolation]) -> ScoredDesignRanking {
    ScoredDesignRanking {
        ranked: Arc::from([]),
        violations: Arc::from(violations.to_vec()),
        budget: MonteCarloBudget::default(),
        early_stopped: false,
    }
}

fn active_candidates(
    candidates: &[CandidateDesign],
    violations: &[ConstraintViolation],
) -> (Vec<CandidateDesign>, Vec<usize>) {
    let mut active = Vec::new();
    let mut index_map = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if violations.iter().any(|item| item.candidate_index == index) {
            continue;
        }
        index_map.push(index);
        active.push(candidate.clone());
    }
    (active, index_map)
}

/// Rewrite each ranked candidate's index through `map`, in place when the slice is not shared.
fn remap_ranked(ranked: &mut Arc<[RankedCandidate]>, map: impl Fn(usize) -> usize) {
    let remap = |item: &mut RankedCandidate| item.candidate_index = map(item.candidate_index);
    if let Some(items) = Arc::get_mut(ranked) {
        items.iter_mut().for_each(remap);
    } else {
        let mut items = ranked.to_vec();
        items.iter_mut().for_each(remap);
        *ranked = items.into();
    }
}

/// `leading` followed by `trailing` with each trailing violation's candidate index mapped.
fn remap_violations(
    leading: Vec<ConstraintViolation>,
    trailing: &[ConstraintViolation],
    map: impl Fn(usize) -> usize,
) -> Arc<[ConstraintViolation]> {
    let mut violations = leading;
    violations.reserve(trailing.len());
    violations.extend(trailing.iter().cloned().map(|mut item| {
        item.candidate_index = map(item.candidate_index);
        item
    }));
    violations.into()
}

fn remap_ranking(
    mut ranking: ScoredDesignRanking,
    index_map: &[usize],
    preview_violations: &[ConstraintViolation],
) -> ScoredDesignRanking {
    remap_ranked(&mut ranking.ranked, |index| index_map[index]);
    let violations = remap_violations(preview_violations.to_vec(), &ranking.violations, |index| {
        index_map[index]
    });
    ScoredDesignRanking {
        ranked: ranking.ranked,
        violations,
        budget: ranking.budget,
        early_stopped: ranking.early_stopped,
    }
}

/// Candidate studies ranked with unlocks derived from a repair search.
#[derive(Clone, Debug)]
pub struct StudyDesignRanking {
    /// The ranking of the verified studies. Candidate indices are positions in the
    /// declared `studies` slice.
    pub ranking: ScoredDesignRanking,
    /// Indices of the studies the repair search verified sufficient on their own.
    pub verified: Vec<usize>,
    /// Every other declared study, with what the search found.
    pub unverified: Vec<UnverifiedStudy>,
    /// The variables the verified studies measure (outside the base law) or set:
    /// the unlock list the ranking used for the query.
    pub unlock: Vec<VariableId>,
}

/// Rank declared studies by identification gain, deriving the unlock lists from a
/// finished repair search instead of a manual list.
///
/// Only a study that `report` classifies verified sufficient on its own unlocks
/// identification: its measured variables outside `base_law` and its set
/// variables become the unlock list for `query`, and its
/// [`DurableStudyCandidate::design_action`] is what the ranker scores. Every other
/// study is reported in [`StudyDesignRanking::unverified`] and as an
/// `unsupported_candidate` violation, so a study that merely names a variable (or only
/// adds rows) is never credited. Any unlock entries already in `eval` are kept.
///
/// For studies that set variables, supply `eval.identified_under_intervention` from
/// [`identified_under_intervention`] on the graph posterior: one flag vector applies
/// to every intervention candidate, so rank experiments on different target sets
/// separately.
///
/// # Errors
///
/// As [`rank_designs_bound`].
#[allow(clippy::too_many_arguments, reason = "the bound ranking plus the repair evidence")]
pub fn rank_studies_bound<A, O>(
    contract: &CausalContract,
    ranker: &DesignRanker,
    objective: &DesignObjective,
    studies: &[DurableStudyCandidate],
    report: &RepairReport,
    base_law: &[VariableId],
    query: QueryId,
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
    decision_target: Option<SemanticDigest>,
) -> Result<StudyDesignRanking, CausalError>
where
    A: Clone + Sync,
    O: Clone + Sync,
{
    let unlocks = verified_unlocks(report, studies, base_law);
    let not_verified = unlocks
        .unverified
        .iter()
        .map(|study| {
            violation(
                study.index,
                "unsupported_candidate",
                "the repair search did not verify this study sufficient on its own",
            )
        })
        .collect::<Vec<_>>();
    let designs = unlocks.designs(studies);
    let (indices, candidates): (Vec<usize>, Vec<CandidateDesign>) = designs.into_iter().unzip();
    if candidates.is_empty() {
        return Ok(StudyDesignRanking {
            ranking: incomplete_ranking(&not_verified),
            verified: Vec::new(),
            unverified: unlocks.unverified,
            unlock: unlocks.variables,
        });
    }
    let mut merged: Vec<(QueryId, Arc<[VariableId]>)> =
        eval.query_id_unlock.map(<[_]>::to_vec).unwrap_or_default();
    if let Some((_, derived)) = unlocks.unlock_entry(query) {
        match merged.iter_mut().find(|(id, _)| *id == query) {
            Some((_, existing)) => {
                let mut union: Vec<VariableId> = existing.to_vec();
                union.extend(derived.iter().copied().filter(|v| !existing.contains(v)));
                *existing = Arc::from(union);
            }
            None => merged.push((query, derived)),
        }
    }
    let derived_eval = DesignEvaluationContext {
        graphs: eval.graphs,
        effect_width: eval.effect_width,
        model_loglik: eval.model_loglik,
        decisions: eval.decisions,
        query_id_unlock: Some(&merged),
        env_id_unlock: eval.env_id_unlock,
        identified_under_intervention: eval.identified_under_intervention,
        graph_features: eval.graph_features,
    };
    let preview =
        preview_design_rank(contract, objective, &candidates, &derived_eval, decision_target)?;
    let mut ranking = if preview.unresolved {
        incomplete_ranking(&preview.violations)
    } else {
        let (active, index_map) = active_candidates(&candidates, &preview.violations);
        if active.is_empty() {
            incomplete_ranking(&preview.violations)
        } else {
            let scored = rank_designs(ranker, objective, &active, &derived_eval, ctx)?;
            remap_ranking(scored, &index_map, &preview.violations)
        }
    };
    // Candidate indices are positions among the verified designs; report them as
    // positions in the declared slice, violations included.
    let to_declared = |index: usize| indices[index];
    remap_ranked(&mut ranking.ranked, to_declared);
    let violations = remap_violations(not_verified, &ranking.violations, to_declared);
    Ok(StudyDesignRanking {
        ranking: ScoredDesignRanking {
            ranked: ranking.ranked,
            violations,
            budget: ranking.budget,
            early_stopped: ranking.early_stopped,
        },
        verified: indices,
        unverified: unlocks.unverified,
        unlock: unlocks.variables,
    })
}

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Identification flags of a graph posterior, derived rather than declared.
pub use antecedent_design::StructureIdentificationError;
/// Candidate transport-evidence additions and their portable design records.
pub use antecedent_design::TransportEvidenceCandidate;
/// The theorem routes of a failed identification, as repair families.
pub use antecedent_design::{FailureRoutes, failure_routes};
/// Failure snapshots and evidence-arrival checks for bounded z-transport.
pub use antecedent_design::{
    ZTransportCandidateOutcome, ZTransportFailureSnapshot, ZTransportFailureSnapshotWire,
    ZTransportPlanSpec, ZTransportPlanningError, ZTransportProposal, ZTransportProposalWire,
};

/// Capture an explicit failed outcome from the currently supported identifier
/// under the caller's identification limits and execution context.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::snapshot_z_transport_failure`].
///
/// # Errors
/// The query binds to available evidence (there is no failure to snapshot), an
/// invalid catalog, or an identification error that is not a classified
/// refusal (an unsupported input or exhausted budget is a snapshot status, not
/// an error).
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ZTransportEvidencePlanning route, which calls this kernel function with the same inputs"
)]
pub fn snapshot_z_transport_failure(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ZTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_identify::sid::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_design::ZTransportFailureSnapshot, antecedent_design::ZTransportPlanningError>
{
    antecedent_design::snapshot_z_transport_failure(diagram, query, catalog, limits, ctx)
}

/// Identification repair: the items of `antecedent_design::repair`.
/// The run entry is a deprecated wrapper of the kernel function.
pub mod repair {
    pub use antecedent_design::repair::{
        BackdoorRepairFamily, FailedContract, FamilyVerdict, REPAIR_DEFAULT_DEPTH,
        REPAIR_MAX_MEMORY_BYTES, REPAIR_MAX_OPERATIONS, RepairCancelled, RepairClassification,
        RepairDerivation, RepairError, RepairFamily, RepairLimits, RepairObjective, RepairOutcome,
        RepairReceipt, RepairReport, TransportRepairFamily,
    };

    /// Apply hypothetical study evidence to a failed contract and report which
    /// candidates and bounded subsets repair it, under one declared budget.
    ///
    /// Candidates are validated first (an invalid or infeasible candidate is an
    /// `invalid` outcome and never searched); the rest are searched in semantic-id
    /// order, by subset size, so the result does not depend on declaration order.
    /// A subset is `verified_sufficient` only when the family's theorem checker
    /// identifies the hypothetical evidence and its derivation re-verifies, and no
    /// unresolved assumption obligation (never satisfiable by a study) remains.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::repair`](fn@antecedent_design::repair).
    ///
    /// # Errors
    /// `identification_repair.bounds_exceeded` for a request beyond the declared
    /// bounds, `identification_repair.cost_units_mismatch` when cost ranking
    /// compares different unit labels, and `identification_repair.budget` when the
    /// budget is stopped before the search is entered. A stop during the search is
    /// a report, never an error.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled IdentificationRepair route, which calls this kernel function with the same inputs"
    )]
    pub fn repair(
        family: &dyn antecedent_design::RepairFamily,
        candidates: &[antecedent_design::DurableStudyCandidate],
        objective: antecedent_design::RepairObjective,
        limits: antecedent_design::RepairLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<antecedent_design::RepairReport, antecedent_design::RepairError> {
        antecedent_design::repair(family, candidates, objective, limits, ctx)
    }
}

#[allow(deprecated, reason = "re-exports the deprecated wrapper at its earlier path")]
pub use repair::repair;

/// Evaluate up to `spec.max_evaluated` candidates against the frozen failure
/// and rank only checked proposals. The receipt names every candidate that was
/// and was not evaluated.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::plan_z_transport_evidence`].
///
/// # Errors
/// A zero evaluation budget, or cancellation before the plan completes.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ZTransportEvidencePlanning route, which calls this kernel function with the same inputs"
)]
pub fn plan_z_transport_evidence(
    snapshot: &std::sync::Arc<antecedent_design::ZTransportFailureSnapshot>,
    spec: &antecedent_design::ZTransportPlanSpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_design::ZTransportPlanResult, antecedent_design::ZTransportPlanningError> {
    antecedent_design::plan_z_transport_evidence(snapshot, spec, ctx)
}

/// The composition boundary: what a decision input carries and what may be combined with what.
pub mod composition_boundary {
    pub use antecedent_design::composition_boundary::{
        ActionDisposition, ActionStatus, AtomCombination, AtomPlan, BoundaryError,
        CalibrationStatus, CompositionOperation, CompositionReceipt, CoordinateIssue,
        DecisionInput, DependenceRoute, EvidenceRelation, InputProvenance, InputSource,
        NativeExecutionRecord, PairRelation, ProviderKind, ReceiptRef, RouteKind, ScalarClaim,
        SupportPolicy, SupportedDecision, SupportedVerdict, TrustEvidence, TrustRequirement,
        UnsupportedActionPolicy, check_atom_combination, check_composition, check_paired_draws,
    };

    /// Evaluate `contract` over `inputs`, deciding support per action.
    ///
    /// Each action is assigned to the first input whose support map covers every
    /// coordinate it reads at or above the policy's weakest status and whose
    /// representation can answer it (a mean or scalar answers only an affine utility
    /// of non-outcome inputs). The actions assigned to one input are evaluated
    /// together by the existing evaluator (`evaluate_contract` or
    /// `evaluate_contract_on_means`), with hard constraints narrowed to those
    /// actions. Actions no input can answer are reported with a reason and left out
    /// of the comparison under `UnsupportedActionPolicy::CompareSupported`. When
    /// actions come from several inputs, values are compared by the criterion,
    /// regret and perfect-information values are withheld, and a criterion that
    /// needs state-by-state alignment refuses.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::composition_boundary::evaluate_with_support`].
    ///
    /// # Errors
    /// An invalid contract or input set; any unevaluated action under
    /// `RequireAllActions`; a state-aligned criterion across several sources; and
    /// any refusal of the underlying evaluator (a mean source with a hard
    /// constraint, a non-joint draw alignment, a non-outcome meaning).
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SupportedInputDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_with_support(
        contract: &antecedent_design::DecisionContract,
        inputs: &[antecedent_design::composition_boundary::DecisionInput],
        policy: &antecedent_design::composition_boundary::SupportPolicy,
    ) -> Result<
        antecedent_design::composition_boundary::SupportedDecision,
        antecedent_design::composition_boundary::BoundaryError,
    > {
        antecedent_design::composition_boundary::evaluate_with_support(contract, inputs, policy)
    }

    /// Compute `functional` of one action's utility from one input.
    ///
    /// A joint law answers through `evaluate_functional`. A mean or scalar answers
    /// only the expectation of an affine utility over non-outcome inputs; a
    /// probability, quantile, variance, tail expectation, nonlinear utility or
    /// outcome-law input is refused, never approximated from a mean.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::composition_boundary::evaluate_functional_on_input`].
    ///
    /// # Errors
    /// An unknown action, a coordinate below `weakest`, a functional a mean cannot
    /// answer, and any refusal of the joint-law evaluator.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SupportedInputDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_functional_on_input(
        contract: &antecedent_design::DecisionContract,
        action_id: &str,
        functional: antecedent_design::DecisionFunctional,
        input: &antecedent_design::composition_boundary::DecisionInput,
        weakest: antecedent_core::response::SupportStatus,
    ) -> Result<
        antecedent_design::decision_eval::FunctionalValue,
        antecedent_design::composition_boundary::BoundaryError,
    > {
        antecedent_design::composition_boundary::evaluate_functional_on_input(
            contract, action_id, functional, input, weakest,
        )
    }
}
/// Adapters from claim types to decision inputs under a declared policy.
pub mod decision_adapters {
    pub use antecedent_design::decision_adapters::{
        AdaptedClaims, AdaptedDecision, ClaimProbability, IdentifiedSetDecision, IdentifiedUtility,
        IdentifiedVerdict, SuppliedClaim,
    };

    /// Interval of an action's utility given an identified set for each input.
    ///
    /// This is an outer enclosure by interval arithmetic: it contains every value
    /// the utility can take, and is sharp when each input appears once in the
    /// expression. An input that appears twice may widen it.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::utility_interval`].
    ///
    /// # Errors
    /// An unknown action, a count of intervals different from the action's inputs,
    /// an invalid contract, or an interval that is not finite.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled IdentifiedSetDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn utility_interval(
        contract: &antecedent_design::DecisionContract,
        action_id: &str,
        inputs: &[antecedent_core::response::IdentifiedSet<f64>],
    ) -> Result<
        antecedent_core::response::IdentifiedSet<f64>,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::utility_interval(contract, action_id, inputs)
    }

    /// Decide over per-action identified sets of utility.
    ///
    /// Intervals are read in the contract's units. `PosteriorExpectedLoss` and
    /// `MinimaxOverIdentifiedSet` read them as losses. Under
    /// `RequireInvariantBestAction` an action is chosen only if its interval beats
    /// every other eligible action's interval entirely (strictly); otherwise the
    /// possibly optimal actions are listed. Under `Maximin`, or for the
    /// `MaximinOverStructures` and `MinimaxOverIdentifiedSet` criteria, the best
    /// worst case wins; the `Regret` criterion under `Maximin` picks the smallest
    /// maximum regret. `ReportOnly` returns the intervals.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::evaluate_identified_sets`].
    ///
    /// # Errors
    /// An invalid contract, a utility interval for an unknown, repeated or
    /// missing action, a non-finite or inverted interval, `BayesOverStructures`
    /// (an identified set has no probability law) and criteria that need a
    /// distribution (`ThresholdProbability`, `Quantile`, `ExpectedRegret`).
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled IdentifiedSetDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_identified_sets(
        contract: &antecedent_design::decision_contract::AdmissibleDecisionContract,
        utilities: &[antecedent_design::decision_adapters::IdentifiedUtility],
    ) -> Result<
        antecedent_design::decision_adapters::IdentifiedSetDecision,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::evaluate_identified_sets(contract, utilities)
    }

    /// Evaluate adapted claims under the contract and assess their robustness.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::evaluate_adapted`].
    ///
    /// # Errors
    /// A policy that differs from the one the claims were adapted under, and any
    /// refusal of the structural evaluation or the robustness assessment.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SuppliedClaimDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_adapted(
        contract: &antecedent_design::decision_contract::AdmissibleDecisionContract,
        adapted: &antecedent_design::decision_adapters::AdaptedClaims,
    ) -> Result<
        antecedent_design::decision_adapters::AdaptedDecision,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::evaluate_adapted(contract, adapted)
    }

    /// Adapt weighted graph-posterior atoms. With genuine probabilities for every
    /// atom the claims are credible; without them they are a structural envelope.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::adapt_weighted_graph_atoms`].
    ///
    /// # Errors
    /// Empty, repeated or blank identities; probabilities for some atoms only;
    /// completion counts or missing probabilities under `BayesOverStructures`.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SuppliedClaimDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn adapt_weighted_graph_atoms(
        claims: Vec<antecedent_design::decision_adapters::SuppliedClaim>,
        policy: antecedent_design::StructuralPolicy,
    ) -> Result<
        antecedent_design::decision_adapters::AdaptedClaims,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::adapt_weighted_graph_atoms(claims, policy)
    }

    /// Adapt one point claim (a single exact law or answer).
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::adapt_point_claim`].
    ///
    /// # Errors
    /// A blank identity, or `BayesOverStructures`, which needs genuine structure
    /// probabilities a single claim does not supply.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SuppliedClaimDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn adapt_point_claim(
        id: &str,
        evidence: antecedent_design::decision_structural::AtomEvidence,
        support: antecedent_design::decision_robustness::AtomSupport,
        policy: antecedent_design::StructuralPolicy,
    ) -> Result<
        antecedent_design::decision_adapters::AdaptedClaims,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::adapt_point_claim(id, evidence, support, policy)
    }

    /// Adapt graph-dependent claims: one answer per graph, no probabilities.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::adapt_graph_dependent_claims`].
    ///
    /// # Errors
    /// Empty, repeated or blank identities; a genuine probability on any graph; and
    /// `BayesOverStructures`, which needs probabilities graph-dependent claims do not
    /// have (completion counts are named in the refusal).
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SuppliedClaimDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn adapt_graph_dependent_claims(
        claims: Vec<antecedent_design::decision_adapters::SuppliedClaim>,
        policy: antecedent_design::StructuralPolicy,
    ) -> Result<
        antecedent_design::decision_adapters::AdaptedClaims,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::adapt_graph_dependent_claims(claims, policy)
    }

    /// Adapt a finite set of supplied scenarios. Declared weights are weights the
    /// caller asserts as probabilities; scenarios are always read as a structural
    /// envelope, never as a posterior.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_adapters::adapt_finite_scenarios`].
    ///
    /// # Errors
    /// As `adapt_weighted_graph_atoms`.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SuppliedClaimDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn adapt_finite_scenarios(
        claims: Vec<antecedent_design::decision_adapters::SuppliedClaim>,
        policy: antecedent_design::StructuralPolicy,
    ) -> Result<
        antecedent_design::decision_adapters::AdaptedClaims,
        antecedent_design::decision_adapters::AdapterError,
    > {
        antecedent_design::decision_adapters::adapt_finite_scenarios(claims, policy)
    }
}
/// Portable decision contract and decision result artifacts.
pub mod decision_artifact {
    pub use antecedent_design::decision_artifact::{
        DecisionContractArtifact, DecisionResultArtifact, MAX_DECISION_ARTIFACT_BYTES,
        contract_from_json, contract_from_json_refusal, contract_to_json, mean_result_to_json,
        result_to_json, source_digest,
    };
}
/// Serde wires of decision contract parts: the bytes every decision artifact stores them as
/// and the JSON a host declares them in.
pub mod decision_wire {
    pub use antecedent_design::decision_wire::{
        ConstraintWire, CriterionWire, ExclusionWire, ExprWire, kind_name, kind_named,
        structural_name, structural_named,
    };
}
/// Durable decision contract and the source each functional needs.
pub mod decision_contract {
    pub use antecedent_design::decision_contract::{
        AdmissibleDecisionContract, ProgramObjective, ProgramSearchContract, ProgramSearchSpace,
        SourceRepresentation, SourceRequirement, UtilityExpr,
    };
}
/// Evaluate a `DecisionContract` on an aligned joint distribution artifact.
pub mod decision_eval {
    pub use antecedent_design::decision_eval::{
        ActionOutcome, DecisionEvalError, DecisionResult, MeanEstimate, Verdict,
    };

    /// Evaluate `contract` on a `MeanSource`.
    ///
    /// Exact only for affine utilities, whose expectation is the utility of the
    /// means. `DecisionContract::source_requirement` is the classification of that
    /// demand. The result's criterion label names the source's posterior mean or
    /// frequentist point estimate, so the same numbers do not share a label. No
    /// sampling error is claimed (`standard_error` is `None`), regret and the value
    /// of perfect information are unavailable, and the verdict is indistinguishable
    /// only on exactly equal values.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_eval::evaluate_contract_on_means`].
    ///
    /// # Errors
    /// Refuses an invalid contract, a demand a mean cannot answer, an unknown
    /// coordinate, a coordinate named twice (the first would silently win), an input
    /// whose functional is `outcome` (a mean grid is not an outcome law), and a
    /// non-finite utility.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled ContractLawDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_contract_on_means(
        contract: &antecedent_design::DecisionContract,
        source: &antecedent_design::decision_eval::MeanSource,
    ) -> Result<
        antecedent_design::decision_eval::DecisionResult,
        antecedent_design::decision_eval::DecisionEvalError,
    > {
        antecedent_design::decision_eval::evaluate_contract_on_means(contract, source)
    }

    /// Evaluate `contract` on `source`.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_eval::evaluate_contract`].
    ///
    /// # Errors
    /// Refuses an invalid contract, non-joint draws, a missing or masked
    /// coordinate, a source meaning that cannot answer an outcome-law input, and
    /// criteria that need structure inputs.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled ContractLawDecision route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_contract(
        contract: &antecedent_design::DecisionContract,
        source: &antecedent_io_core::distribution_artifact::DistributionArtifact,
    ) -> Result<
        antecedent_design::decision_eval::DecisionResult,
        antecedent_design::decision_eval::DecisionEvalError,
    > {
        antecedent_design::decision_eval::evaluate_contract(contract, source)
    }
}
/// Reliability of sampled and adaptively selected decisions.
pub mod decision_reliability {
    pub use antecedent_design::decision_reliability::{
        RankingRequest, ReliabilityError, support_route,
    };

    /// Rank the candidates of a carrier.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::decision_reliability::rank`].
    ///
    /// # Errors
    /// Anything `DecisionDraws::validate` refuses, fewer than two candidates, an unknown
    /// baseline, a `k` outside the candidate count, rank tables (two `u32` ranks per
    /// candidate-draw cell) the context's memory budget does not admit, or cancellation.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled DecisionDrawsRanking route, which calls this kernel function with the same inputs"
    )]
    pub fn rank(
        draws: &antecedent_design::decision_reliability::DecisionDraws,
        request: &antecedent_design::decision_reliability::RankingRequest,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_design::decision_reliability::RankingReport,
        antecedent_design::decision_reliability::ReliabilityError,
    > {
        antecedent_design::decision_reliability::rank(draws, request, ctx)
    }
}
/// Durable admissible decision contract and robust decision result artifacts.
pub mod decision_robust_artifact {
    pub use antecedent_design::decision_robust_artifact::{
        AdmissibleContractArtifact, ExternalCallbackReceipt, ExternalTrustLimit,
        RobustResultArtifact, admissible_contract_from_json, admissible_contract_from_json_refusal,
        admissible_contract_to_json, atom_digests_of,
    };
}
/// Robustness of a decision across structures, support and claims.
pub mod decision_robustness {
    pub use antecedent_design::decision_robustness::{AtomSupport, InputSupport, RobustVerdict};
}
/// Decisions under structural uncertainty.
pub mod decision_structural {
    pub use antecedent_design::decision_structural::{
        AtomStatus, AtomSummary, StructuralDecisionResult, StructuralVerdict,
    };
}
/// Durable design-ranking artifact (`design_ranking_v1`) and its independent consumer.
pub mod design_ranking_artifact {
    pub use antecedent_design::design_ranking_artifact::MAX_DESIGN_RANKING_ARTIFACT_BYTES;
}
/// Per-candidate expected value of sample information with declared integration.
pub mod evsi {
    pub use antecedent_design::evsi::{CostToUtilityMap, EvsiError, StudyCostSpec};
}
/// Original finite-law functional replay with complete source rows and scientific identity.
pub mod functional_source {
    pub use antecedent_design::functional_source::LawFunctionalArtifact;
}
/// Checked adapter from `PriorCatalog` posterior sources to a future candidate-signal.
pub mod prior_signal {
    pub use antecedent_design::prior_signal::{
        CandidateSignalPrior, CandidateSignalSource, NativeSignalFamily, PriorSignalRequest,
        PriorSourceInput, SourceToTarget, TransportPolicyDecl,
    };

    /// Adapt compatible prior sources and a future candidate signal into a prior and
    /// provider for preposterior analysis.
    ///
    /// Sources are processed in canonical artifact-id order, so the result does not
    /// depend on the order they were supplied in. Each source's draws are mapped to the
    /// target coordinate, tempered by its prior strength, its location moved toward the
    /// pooled weighted mean (`weight`) by its conflict shrinkage, and resampled by deterministic stratified quantiles
    /// into `resolution` equally weighted draws in proportion to the effective weights.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::prior_signal::adapt_prior_to_signal`].
    ///
    /// # Errors
    ///
    /// A signal that is only a coefficient prior or unlock list, an artifact missing
    /// from or incompatible with the catalog, invalid weights, strength, shrinkage or
    /// maps, missing lineage, a population change without transport policy, observation
    /// reuse between a source and the candidate or between sources, or an invalid
    /// resolution.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled PriorSignalAdaptation route, which calls this kernel function with the same inputs"
    )]
    pub fn adapt_prior_to_signal(
        catalog: &antecedent_io_core::prior_bank::PriorCatalog,
        target: &antecedent_io_core::prior_bank::TargetDesign,
        request: &antecedent_design::prior_signal::PriorSignalRequest,
    ) -> Result<
        antecedent_design::prior_signal::CandidateSignalPrior,
        antecedent_design::prior_signal::PriorSignalError,
    > {
        antecedent_design::prior_signal::adapt_prior_to_signal(catalog, target, request)
    }
}
/// Exhaustive bounded program search over checked response quantities.
pub mod program_search {
    pub use antecedent_design::program_search::{
        FiniteProgramDomain, ProgramControl, ProgramFeasibility, ProgramRestriction,
        ProgramSearchError,
    };
    /// Global bounds derived from expressions, with outward-rounded arithmetic.
    pub mod certified {
        pub use antecedent_design::program_search::certified::{
            CertificateOptions, CertificateStop,
        };
    }
    /// Bounded derivative-free search over actual compiled programs.
    pub mod continuous {
        pub use antecedent_design::program_search::continuous::{
            ContinuousSearchOptions, ContinuousSearchReliability, SelectedProgramReliability,
        };
    }
}
/// Per-proposal receipt linking identification repair and design ranking.
pub mod proposal_receipt {
    pub use antecedent_design::proposal_receipt::{
        Arrival, ArrivalVerdict, ArrivedEvidence, BaseFailureBinding, ChangedPremise, CostBinding,
        DecisionReceipt, HypotheticalBinding, LineageBinding, ObservedLaw, ProposalBundle,
        ProposalReceipt, ProposalReceiptError, ValuationBinding, on_arrival, verify,
    };
}
/// Finite source-law to terminal-decision and study-ranking binding.
pub mod rollout_artifact {
    pub use antecedent_design::rollout_artifact::{
        MAX_ROLLOUT_BYTES, RolloutArtifact, RolloutExpectation, consume, consume_expecting,
    };
}
/// Checked scientific standing carried from a compiled response into a decision.
pub mod scientific_source {
    pub use antecedent_design::scientific_source::{
        ScientificConsumptionScope, ScientificSourcePayload,
    };
}
/// `SequentialEvaluationArtifact` (`sequential_evaluation_v1`).
pub mod sequential_evaluation_artifact {
    pub use antecedent_design::sequential_evaluation_artifact::StepWire;
}
/// Source-backed projections for a portable composed workflow.
pub mod source_projection_artifact {
    pub use antecedent_design::source_projection_artifact::{
        MAX_SOURCE_PROJECTION_BYTES, ProjectionSource, SourceProjection, SourceProjectionArtifact,
    };
}
/// Identification flags of a graph posterior, derived rather than declared.
pub mod structure_identification {
    pub use antecedent_design::structure_identification::GraphIdentFlag;

    /// Whether the complete ID algorithm identifies `P(outcome | do(treatment))` in
    /// `graph`.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_design::structure_identification::identifies`].
    ///
    /// # Errors
    /// Unknown variables, or a budget stop of the identifier (never read as
    /// non-identification).
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled StructureIdentification route, which calls this kernel function with the same inputs"
    )]
    pub fn identifies(
        graph: &antecedent_graph::admg::Admg,
        treatment: antecedent_core::ids::VariableId,
        outcome: antecedent_core::ids::VariableId,
    ) -> Result<
        antecedent_prob::graph_samples::GraphIdentFlag,
        antecedent_design::StructureIdentificationError,
    > {
        antecedent_design::structure_identification::identifies(graph, treatment, outcome)
    }
}
/// Portable transported decision artifact (`transported_decision_v1`).
pub mod transported_decision_artifact {
    pub use antecedent_design::transported_decision_artifact::{
        ConsumedTransportedDecision, consume_transported_decision_expecting,
        export_transported_decision,
    };

    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`consume_transported_decision_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_design::transported_decision_artifact::consume_transported_decision` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_design::transported_decision_artifact::consume_transported_decision`.
    #[deprecated(
        since = "3.1.0",
        note = "use consume_transported_decision_expecting with a RetainedArtifactIdentity"
    )]
    pub fn consume_transported_decision(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<ConsumedTransportedDecision, antecedent_io_core::IoError> {
        antecedent_design::transported_decision_artifact::consume_transported_decision(
            bytes,
            expected_identity,
        )
    }
}

//! experiment / measurement design facade helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
    SamplingPlan, ScoreEvaluation, ScoredDesignRanking, Utility, evaluate_decision,
};
/// Candidate studies generated from verified obligations, the plan of options that
/// close an identification gap, and the derived structure of a graph posterior.
pub use antecedent_design::{
    CandidateGeneration, CandidateTemplate, ConfounderPriorDeclaration, ConfounderValueFrame,
    DurableStudyCandidate, EvidencePlan, EvidencePlanArtifact, EvidencePlanArtifactError,
    GraphPosteriorSample, GridUtility, PlanOption, PlanOutcome, PlanValue, PlanValueBasis,
    RepairReport, StudyDeclarations, StudyKind, UnmatchedObligation, UnverifiedStudy,
    VerifiedUnlocks, generate_candidates, identified_under_intervention, mutilate,
    structure_posterior, verified_unlocks,
};
use antecedent_design::{ObjectiveChannel, ObjectiveScorer};
/// Study planning over the mz-transport and mixed-source catalogs: declared
/// candidate studies, the bounded cost-ordered plan, arrival and the portable
/// `study_plan_v1` artifact.
pub use antecedent_design::{
    StudyArrival, StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlan,
    StudyPlanArtifactWire, StudyPlanConsumeLimits, StudyPlanError, StudyPlanLimits,
    StudyPlanProposal, StudyPlanResult, StudyPlanRoute, StudyPlanStop, StudyProposal,
    StudySubsetOutcome, plan_studies,
};
pub use antecedent_stats::adaptive_inference::group_sequential::SpendingFunction;

/// Obligations inspected before the existing ranker runs.
///
/// This is not a second ranker. Unlicensed candidates become typed violations,
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
/// [`rank_designs_bound`], which refuse an unlicensed or unprepared contract and
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
/// Unlicensed or unprepared contracts, or a width ranking across incomparable
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
/// Unlicensed/unprepared contracts, incomparable width targets, or ranker failure.
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
    if !matches!(contract.support_status, Some(CellStatus::Licensed)) {
        return Err(CausalError::Unsupported {
            message: crate::error::RANK_DESIGNS_REQUIRES_LICENSE,
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
    slot.unidentified_mass > 0.0 && eval.graphs.unidentified_mass() + 1e-12 < slot.unidentified_mass
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
            "each candidate must map onto a licensed add-units, extend-series, pulse-window, or measure-covariate contract",
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
        unidentified_mass: (eval.graphs.n_samples > 0).then(|| eval.graphs.unidentified_mass()),
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
            violations.push(violation(index, "unlicensed_candidate", detail));
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
                Some("measure_covariate requires a licensed identification update")
            }
            CandidateDesign::Intervene(_) if unlock.is_empty() && !intervene => {
                Some("pulse-window / intervention requires a licensed identification update")
            }
            CandidateDesign::ObserveEnvironment(_) if env_unlock.is_empty() => {
                Some("extend-series identification requires a declared environment unlock")
            }
            _ => None,
        };
        if let Some(detail) = detail {
            violations.push(violation(index, "unlicensed_candidate", detail));
        }
    }
}

/// Without a decision registry the ranking is unresolved (returns `true`). With one,
/// a candidate the signal assigns no sample size has no data model for the
/// value-of-information update and is unlicensed; the ranker would not score it.
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
                "unlicensed_candidate",
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

fn remap_ranking(
    ranking: ScoredDesignRanking,
    index_map: &[usize],
    preview_violations: &[ConstraintViolation],
) -> ScoredDesignRanking {
    let ranked: Vec<RankedCandidate> = ranking
        .ranked
        .iter()
        .cloned()
        .map(|mut item| {
            item.candidate_index = index_map[item.candidate_index];
            item
        })
        .collect();
    let mut violations = preview_violations.to_vec();
    violations.extend(ranking.violations.iter().cloned().map(|mut item| {
        item.candidate_index = index_map[item.candidate_index];
        item
    }));
    ScoredDesignRanking {
        ranked: ranked.into(),
        violations: violations.into(),
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
/// `unlicensed_candidate` violation, so a study that merely names a variable (or only
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
                "unlicensed_candidate",
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
    let ranking = if preview.unresolved {
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
    let ranked: Vec<RankedCandidate> = ranking
        .ranked
        .iter()
        .cloned()
        .map(|mut item| {
            item.candidate_index = to_declared(item.candidate_index);
            item
        })
        .collect();
    let mut violations = not_verified;
    violations.extend(ranking.violations.iter().cloned().map(|mut item| {
        item.candidate_index = to_declared(item.candidate_index);
        item
    }));
    Ok(StudyDesignRanking {
        ranking: ScoredDesignRanking {
            ranked: ranked.into(),
            violations: violations.into(),
            budget: ranking.budget,
            early_stopped: ranking.early_stopped,
        },
        verified: indices,
        unverified: unlocks.unverified,
        unlock: unlocks.variables,
    })
}

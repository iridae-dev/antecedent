//! experiment / measurement design facade helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSource, AssumptionStatus, ExecutionContext, MonteCarloBudget, ObligationKind,
    ObligationRecord, ObligationScope, SemanticDigest,
};

use crate::analysis::CausalContract;
use crate::error::CausalError;
use crate::support::CellStatus;

pub use antecedent_design::{
    CandidateDesign, ConstraintViolation, DecisionConstraint, DecisionEvaluation, DecisionProblem,
    DecisionProblemId, DesignConstraints, DesignCost, DesignError, DesignEvaluationContext,
    DesignObjective, DesignRankConfig, DesignRanker, DesignRanking, EffectWidthContext,
    EnvironmentGramSpec, EnvironmentPlan, ExperimentPlan, InterventionDesignEffect,
    MeasureColumnSpec, MeasurementPlan, ModelLoglikDraws, RankedCandidate, SamplingPlan, Utility,
    evaluate_decision,
};

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

/// Rank candidate designs under an objective.
///
/// # Errors
///
/// Propagates design evaluation failures.
pub fn rank_designs<A, O>(
    ranker: &DesignRanker,
    objective: &DesignObjective,
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
) -> Result<DesignRanking, CausalError>
where
    A: Clone,
    O: Clone,
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
) -> Result<DesignRanking, CausalError>
where
    A: Clone,
    O: Clone,
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
    matches!(objective, DesignObjective::ReduceEffectPosteriorWidth { .. })
        && decision_target.is_some_and(|target| target != contract.identities.target)
}

fn id_mass_renormalized<A, O>(
    objective: &DesignObjective,
    contract: &CausalContract,
    eval: &DesignEvaluationContext<'_, A, O>,
) -> bool {
    if !matches!(objective, DesignObjective::IncreaseIdentificationProbability { .. }) {
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
    match objective {
        DesignObjective::ReduceEffectPosteriorWidth { .. } => {
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
        DesignObjective::IncreaseIdentificationProbability { query } => {
            if eval.graphs.n_samples == 0 {
                unresolved = true;
                push_all_unresolved(
                    candidates,
                    &mut violations,
                    "unresolved_utility",
                    "missing_structural_distribution",
                );
            } else {
                collect_id_violations(candidates, eval, *query, &mut violations);
            }
        }
        DesignObjective::ReduceGraphEntropy => {
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
        DesignObjective::DistinguishModels { .. } if eval.model_loglik.is_none() => {
            unresolved = true;
            push_all_unresolved(
                candidates,
                &mut violations,
                "unresolved_utility",
                "missing_model_loglik",
            );
        }
        DesignObjective::ReduceDecisionRegret { .. } if eval.decisions.is_none() => {
            unresolved = true;
            push_all_unresolved(
                candidates,
                &mut violations,
                "unresolved_utility",
                "missing_decision_registry",
            );
        }
        _ => {}
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

fn incomplete_ranking(violations: &[ConstraintViolation]) -> DesignRanking {
    DesignRanking {
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
    ranking: DesignRanking,
    index_map: &[usize],
    preview_violations: &[ConstraintViolation],
) -> DesignRanking {
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
    DesignRanking {
        ranked: ranked.into(),
        violations: violations.into(),
        budget: ranking.budget,
        early_stopped: ranking.early_stopped,
    }
}

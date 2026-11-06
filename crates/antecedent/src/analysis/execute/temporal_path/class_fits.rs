//! Fitting and refutation of temporal class completions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BayesianConfig, BayesianGCompWorkspace, BayesianMediationFit, CausalError, CausalResponse,
    ClassBayesianPlan, ClassMediationInput, ClassResponseInput, CompletionFitMemo, EstimatorId,
    ExecutionContext, GraphIdentFlag, IdentificationResult, IdentificationStatus, MediationFitKey,
    ResponseFitKey, TemporalMediationEstimator, TemporalResponseEstimator, VariableId,
    bayesian_gcomp, identification_status_ok_for_case, mediation_mechanism_seed,
    resolve_envelope_prior_anchor, resolve_temporal_response_prior, select_estimand,
};

/// Classifies each completion of a class envelope: its weight (the class
/// prior's mass when supplied), whether it is identified, and which to fit.
pub(super) fn plan_class_bayesian_atoms(
    bundle_envelope: &antecedent_identify::TemporalClassEnvelope,
    class_masses: Option<&[f64]>,
) -> Result<ClassBayesianPlan, CausalError> {
    let mut plan = ClassBayesianPlan {
        weights: Vec::new(),
        flags: Vec::new(),
        keys: Vec::new(),
        fit_atoms: Vec::new(),
        primary_estimand: None,
    };
    for (i, (case, indexer)) in
        bundle_envelope.envelope.cases.iter().zip(bundle_envelope.indexers.iter()).enumerate()
    {
        let key = case.graph.fingerprint();
        plan.keys.push(key);
        let weight = class_masses.map_or(case.weight.0, |masses| masses[i]);
        plan.weights.push(weight);
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            plan.flags.push(GraphIdentFlag::Unidentified);
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        plan.primary_estimand.get_or_insert_with(|| estimand.clone());
        plan.flags.push(GraphIdentFlag::Identified);
        plan.fit_atoms.push((key, estimand, case.result.status, indexer.clone(), weight));
    }
    Ok(plan)
}

/// The composed Bayesian mediation posterior of one class completion, with the
/// prior-conflict summaries its mechanism fits resolved, served from `fits`
/// when an earlier completion (or the shared-design recompute) of the same
/// horizon fitted the same method, mediator, adjustment and status.
#[inline(never)]
pub(super) fn fit_class_bayesian_mediation(
    fits: &mut CompletionFitMemo<BayesianMediationFit, MediationFitKey>,
    cfg: &BayesianConfig,
    input: ClassMediationInput<'_>,
    horizon: u32,
    ctx: &ExecutionContext,
) -> Result<BayesianMediationFit, CausalError> {
    let ClassMediationInput { data, estimand, query, adjustment, status } = input;
    let key = MediationFitKey::new(estimand, adjustment, status);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let bayes = bayesian_gcomp(cfg, ctx);
    let preparations =
        antecedent_estimate::bayesian_mediation::prepare_temporal_mediation_adjusted(
            data, estimand, query, adjustment, ctx,
        )
        .map_err(CausalError::from)?;
    let mut conflicts = Vec::new();
    let mut mechanisms = Vec::with_capacity(preparations.len());
    for (index, prep) in preparations.iter().enumerate() {
        let mut mechanism = bayes.clone();
        mechanism.seed = mediation_mechanism_seed(mechanism.seed, horizon, index);
        let (prior, conflict) = resolve_envelope_prior_anchor(cfg, prep, ctx)?;
        conflicts.extend(conflict);
        mechanism.prior = prior;
        mechanisms.push(
            mechanism
                .fit(prep, status, &mut BayesianGCompWorkspace::default(), ctx)
                .map_err(CausalError::from)?,
        );
    }
    let mechanisms = antecedent_estimate::bayesian_mediation::couple_mediation_mechanisms(
        &preparations,
        mechanisms,
        horizon as usize,
        mediation_mechanism_seed(bayes.seed, horizon, preparations.len()),
        ctx,
    );
    let composed = antecedent_estimate::bayesian_mediation::compose_temporal_mediation_mechanisms(
        &mechanisms,
        query,
        status,
    )
    .map_err(CausalError::from)?;
    let fitted = (composed, conflicts);
    fits.insert(key, fitted.clone());
    Ok(fitted)
}

/// The point mediation estimate of one class completion, served from `fits`
/// when an earlier completion of the horizon fitted the same method, mediator
/// and adjustment.
#[inline(never)]
pub(super) fn fit_class_frequentist_mediation(
    fits: &mut CompletionFitMemo<antecedent_estimate::TemporalMediationEstimate, MediationFitKey>,
    estimator: &TemporalMediationEstimator,
    input: ClassMediationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<antecedent_estimate::TemporalMediationEstimate, CausalError> {
    let ClassMediationInput { data, estimand, query, adjustment, status } = input;
    let key = MediationFitKey::new(estimand, adjustment, status);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let estimate = estimator
        .estimate_with_adjustment(data, estimand, query, adjustment, &[], ctx)
        .map_err(CausalError::from)?;
    fits.insert(key, estimate.clone());
    Ok(estimate)
}

/// The refutation reports of one class completion's mediation estimate, served
/// from `fits` under the key of the estimate they check. The reports read the
/// estimand only through its mediator and the estimator's method check.
#[inline(never)]
pub(super) fn refute_class_mediation(
    fits: &mut CompletionFitMemo<Vec<antecedent_validate::RefutationReport>, MediationFitKey>,
    input: ClassMediationInput<'_>,
    estimate: &antecedent_estimate::TemporalMediationEstimate,
    (full, gaps): (bool, bool),
    ctx: &ExecutionContext,
) -> Result<Vec<antecedent_validate::RefutationReport>, CausalError> {
    let ClassMediationInput { data, estimand, query, adjustment, status } = input;
    let key = MediationFitKey::new(estimand, adjustment, status);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let reports = antecedent_validate::mediation::refute_temporal_mediation_with_gaps(
        data, estimand, query, estimate, full, adjustment, gaps, ctx,
    )
    .map_err(CausalError::from)?;
    fits.insert(key, reports.clone());
    Ok(reports)
}

pub(super) type BayesianResponseFit = (CausalResponse, Option<antecedent_prob::ConflictSummary>);

/// One class completion's Bayesian temporal response and the prior-conflict
/// summary its prior resolution produced, served from `fits` when an earlier
/// completion of the horizon was given the same fit inputs.
#[inline(never)]
pub(super) fn fit_class_bayesian_response(
    fits: &mut CompletionFitMemo<BayesianResponseFit, ResponseFitKey>,
    cfg: &BayesianConfig,
    input: ClassResponseInput<'_>,
    horizon: u32,
    identification: &IdentificationResult,
    (treatment, outcome): (VariableId, VariableId),
    ctx: &ExecutionContext,
) -> Result<BayesianResponseFit, CausalError> {
    let ClassResponseInput { series, query, estimand, indexer } = input;
    let status = identification.status;
    let assumptions = &identification.required_assumptions;
    let key = ResponseFitKey::new(estimand, indexer, status, assumptions);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let mut bayes = bayesian_gcomp(cfg, ctx);
    let entry = crate::analysis::prepared::CachedTemporalHorizonIdentification {
        horizon,
        identification: identification.clone(),
        estimand: estimand.clone(),
        indexer: indexer.clone(),
    };
    let (prior, conflict) = resolve_temporal_response_prior(
        cfg,
        series,
        query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class temporal response query carries no temporal spec".into(),
        })?,
        &[&entry],
        treatment,
        outcome,
        ctx,
    )?;
    bayes.prior = prior;
    let response = TemporalResponseEstimator::new()
        .estimate_bayesian(
            series,
            antecedent_estimate::TemporalIdentification {
                horizons: &[(estimand, indexer)],
                status,
                assumptions: assumptions.clone(),
            },
            query,
            &bayes,
            ctx,
        )
        .map_err(CausalError::from)?;
    let fitted = (response, conflict);
    fits.insert(key, fitted.clone());
    Ok(fitted)
}

/// One class completion's frequentist temporal response, served from `fits`
/// (when the caller supplies a memo) under the same condition.
#[inline(never)]
pub(super) fn fit_class_frequentist_response(
    fits: Option<&mut CompletionFitMemo<CausalResponse, ResponseFitKey>>,
    estimator: &TemporalResponseEstimator,
    input: ClassResponseInput<'_>,
    status: IdentificationStatus,
    assumptions: &antecedent_core::AssumptionSet,
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let ClassResponseInput { series, query, estimand, indexer } = input;
    let key = ResponseFitKey::new(estimand, indexer, status, assumptions);
    if let Some(hit) = fits.as_deref().and_then(|memo| memo.get(&key)) {
        return Ok(hit.clone());
    }
    let response = estimator
        .estimate(series, &[(estimand, indexer)], query, status, assumptions.clone(), ctx)
        .map_err(CausalError::from)?;
    if let Some(memo) = fits {
        memo.insert(key, response.clone());
    }
    Ok(response)
}

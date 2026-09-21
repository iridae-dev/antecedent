//! Shared learner-backed nuisance helpers for DML / [`crate::DrLearner`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value,
    clippy::needless_lifetimes,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_learn::{
    CrossFittedPrediction, DesignView, LearnerFactory, LearnerSpec, NuisanceDiagnostics,
    PredictionTask, RowSelection, TargetView, cross_fit_with_folds, diagnose, resolve_for,
};

use crate::error::EstimationError;

pub(crate) fn learn_err(err: antecedent_learn::LearnError) -> EstimationError {
    EstimationError::stats_msg(err.to_string())
}

pub(crate) fn intercept_is_constant(design: &[f64], nrows: usize) -> bool {
    if nrows == 0 {
        return false;
    }
    design[..nrows].iter().all(|&v| (v - 1.0).abs() <= 1e-12)
}

/// Drop a constant intercept column for tree learners; keep it for parametric ones.
pub(crate) fn design_for_spec<'a>(
    spec: LearnerSpec,
    design: &'a [f64],
    nrows: usize,
    ncols: usize,
) -> Result<DesignView<'a>, EstimationError> {
    let drop_intercept = matches!(
        spec,
        LearnerSpec::GradientBoostedTrees(_)
            | LearnerSpec::RandomForest(_)
            | LearnerSpec::NeuralNet(_)
    ) && ncols > 1
        && intercept_is_constant(design, nrows);
    if drop_intercept {
        DesignView::from_column_major(&design[nrows..], nrows, ncols - 1).map_err(learn_err)
    } else {
        DesignView::from_column_major(design, nrows, ncols).map_err(learn_err)
    }
}

pub(crate) fn resolve_nuisance(
    spec: LearnerSpec,
    task: PredictionTask,
    _design: &[f64],
    _nrows: usize,
    _ncols: usize,
    _y: &[f64],
    _ctx: &ExecutionContext,
) -> Result<(Box<dyn LearnerFactory>, NuisanceDiagnostics), EstimationError> {
    Ok((resolve_for(spec, task).map_err(learn_err)?, NuisanceDiagnostics::default()))
}

/// Seeded, stratified, unit-level cross-fit fold plan (one fold id per row).
///
/// `units[i]` names the physical unit behind row `i` (the original row index): every row of a
/// unit — a bootstrap resample's duplicates — shares one fold, so a unit is never both in a
/// training set and in its own validation fold. Within each stratum (`strata[i]`, e.g. the
/// treatment arm) the distinct units are permuted by a `seed`-keyed Fisher–Yates shuffle and
/// dealt round-robin, so every fold sees every stratum and the plan is a function of the seed
/// and the unit ids, not of any periodic structure in the file order.
///
/// # Errors
///
/// `folds < 2`, length mismatch, more distinct units than the fold count allows, or fewer
/// distinct units than folds.
pub(crate) fn crossfit_fold_plan(
    strata: &[u32],
    units: &[u32],
    folds: usize,
    seed: u64,
) -> Result<Vec<u32>, EstimationError> {
    if folds < 2 || u32::try_from(folds).is_err() {
        return Err(EstimationError::unsupported("cross-fitting requires at least two folds"));
    }
    if strata.len() != units.len() {
        return Err(EstimationError::data_msg("fold plan strata and unit ids must align"));
    }
    let mut stratum_of = std::collections::BTreeMap::new();
    for (&u, &s) in units.iter().zip(strata) {
        stratum_of.entry(u).or_insert(s);
    }
    if stratum_of.len() < folds {
        return Err(EstimationError::data_msg("cross-fitting folds cannot exceed distinct units"));
    }
    // Units grouped by stratum, each group sorted by unit id (determinism), then shuffled.
    let mut groups: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (&u, &s) in &stratum_of {
        groups.entry(s).or_default().push(u);
    }
    let mut fold_of_unit = std::collections::HashMap::with_capacity(stratum_of.len());
    let mut dealt = 0usize;
    for (&stratum, members) in &mut groups {
        let mut rng = CausalRng::from_seed(
            seed ^ u64::from(stratum).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03),
        );
        for i in (1..members.len()).rev() {
            let j = (rng.next_f64() * (i as f64 + 1.0)) as usize;
            members.swap(i, j.min(i));
        }
        for &u in members.iter() {
            fold_of_unit.insert(u, (dealt % folds) as u32);
            dealt += 1;
        }
    }
    Ok(units.iter().map(|u| fold_of_unit[u]).collect())
}

pub(crate) fn cross_fit_nuisance(
    spec: LearnerSpec,
    task: PredictionTask,
    design: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    folds: usize,
    ctx: &ExecutionContext,
) -> Result<CrossFittedPrediction, EstimationError> {
    let (factory, _) = resolve_nuisance(spec, task, design, nrows, ncols, y, ctx)?;
    let view = design_for_spec(spec, design, nrows, ncols)?;
    let rows: Vec<u32> = (0..nrows)
        .map(|i| {
            u32::try_from(i)
                .map_err(|_| EstimationError::data_msg("rows exceed u32 index capacity"))
        })
        .collect::<Result<_, _>>()?;
    let plan = crossfit_fold_plan(&vec![0; nrows], &rows, folds, ctx.rng.master_seed())?;
    let plan: Vec<u16> = plan.into_iter().map(|f| f as u16).collect();
    cross_fit_with_folds(factory.as_ref(), view, TargetView::new(y), plan, ctx, None)
        .map_err(learn_err)
}

/// Per-arm outcome models + propensity, trained on `train ∩ arm` and scored on the valid fold.
pub(crate) fn cross_fit_aipw_nuisances(
    outcome: LearnerSpec,
    treatment: LearnerSpec,
    design: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    t: &[f64],
    folds: usize,
    ctx: &ExecutionContext,
    fold_ids: Vec<u16>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, CrossFittedPrediction), EstimationError> {
    let (outcome_factory, _) =
        resolve_nuisance(outcome, PredictionTask::Regression, design, nrows, ncols, y, ctx)?;
    let (treat_factory, _) = resolve_nuisance(
        treatment,
        PredictionTask::BinaryProbability,
        design,
        nrows,
        ncols,
        t,
        ctx,
    )?;
    let x_outcome = design_for_spec(outcome, design, nrows, ncols)?;
    let x_treat = design_for_spec(treatment, design, nrows, ncols)?;
    let fold_parts = ctx
        .map_indexed(folds, |fold, inner| {
            fit_aipw_fold(
                outcome_factory.as_ref(),
                treat_factory.as_ref(),
                x_outcome,
                x_treat,
                y,
                t,
                &fold_ids,
                fold as u16,
                inner,
            )
        })
        .map_err(learn_err)?;

    let mut mu0 = vec![0.0; nrows];
    let mut mu1 = vec![0.0; nrows];
    let mut ehat = vec![0.0; nrows];
    let mut treat_prov = Vec::with_capacity(folds);
    for part in fold_parts {
        treat_prov.extend(part.provenance);
        for (i, pred) in part.valid.iter().copied().zip(part.mu0) {
            mu0[i as usize] = pred;
        }
        for (i, pred) in part.valid.iter().copied().zip(part.mu1) {
            mu1[i as usize] = pred;
        }
        for (i, pred) in part.valid.iter().copied().zip(part.ehat) {
            ehat[i as usize] = pred;
        }
    }
    let treat = CrossFittedPrediction {
        predictions: ehat.clone(),
        fold_assignment: fold_ids,
        model_provenance: treat_prov,
        validation: diagnose(PredictionTask::BinaryProbability, t, &ehat),
    };
    Ok((mu0, mu1, ehat, treat))
}

struct AipwFold {
    valid: Vec<u32>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
    ehat: Vec<f64>,
    provenance: [antecedent_learn::LearnerProvenance; 3],
}

fn fit_aipw_fold(
    outcome: &dyn LearnerFactory,
    treatment: &dyn LearnerFactory,
    x_outcome: DesignView<'_>,
    x_treat: DesignView<'_>,
    y: &[f64],
    t: &[f64],
    folds: &[u16],
    fold: u16,
    ctx: &ExecutionContext,
) -> Result<AipwFold, antecedent_learn::LearnError> {
    let mut train = Vec::new();
    let mut train0 = Vec::new();
    let mut train1 = Vec::new();
    let mut valid = Vec::new();
    for (i, &f) in folds.iter().enumerate() {
        let idx = i as u32;
        if f == fold {
            valid.push(idx);
            continue;
        }
        train.push(idx);
        if t[i] > 0.5 {
            train1.push(idx);
        } else {
            train0.push(idx);
        }
    }
    if train.is_empty() || valid.is_empty() || train0.is_empty() || train1.is_empty() {
        return Err(antecedent_learn::LearnError::Shape {
            message: "empty DML/AIPW train, valid, or treatment arm in a fold",
        });
    }
    let y_view = TargetView::new(y);
    let t_view = TargetView::new(t);
    let mu0 = outcome.fit(x_outcome.with_rows(RowSelection::new(&train0))?, y_view, None, ctx)?;
    let mu1 = outcome.fit(x_outcome.with_rows(RowSelection::new(&train1))?, y_view, None, ctx)?;
    let ehat = treatment.fit(x_treat.with_rows(RowSelection::new(&train))?, t_view, None, ctx)?;
    let valid_out = x_outcome.with_rows(RowSelection::new(&valid))?;
    let valid_t = x_treat.with_rows(RowSelection::new(&valid))?;
    let mut m0 = vec![0.0; valid.len()];
    let mut m1 = vec![0.0; valid.len()];
    let mut e = vec![0.0; valid.len()];
    mu0.predict(valid_out, &mut m0, ctx)?;
    mu1.predict(valid_out, &mut m1, ctx)?;
    ehat.predict(valid_t, &mut e, ctx)?;
    Ok(AipwFold {
        valid,
        mu0: m0,
        mu1: m1,
        ehat: e,
        provenance: [mu0.provenance(), mu1.provenance(), ehat.provenance()],
    })
}

pub(crate) fn clip_propensity(scores: &mut [f64], clip: Option<f64>) {
    let Some(c) = clip else {
        return;
    };
    for s in scores {
        *s = s.clamp(c, 1.0 - c);
    }
}

pub(crate) fn iid_se(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = values.iter().sum::<f64>() / nf;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    (var / nf).max(0.0).sqrt()
}

pub(crate) fn aipw_scores(t: &[f64], y: &[f64], e: &[f64], mu0: &[f64], mu1: &[f64]) -> Vec<f64> {
    t.iter()
        .zip(y)
        .zip(e)
        .zip(mu0)
        .zip(mu1)
        .map(|((((&ti, &yi), &ei), &m0), &m1)| {
            (m1 - m0) + ti * (yi - m1) / ei - (1.0 - ti) * (yi - m0) / (1.0 - ei)
        })
        .collect()
}

/// Immutable predictions shared across compatible estimators.
pub(crate) type AipwPredictions = (Vec<f64>, Vec<f64>, Vec<f64>, CrossFittedPrediction);

/// A prepared-input key owns its buffers so allocator reuse cannot create stale hits.
#[derive(Debug)]
pub(crate) struct AipwCacheEntry {
    design: std::sync::Arc<[f64]>,
    outcome_values: std::sync::Arc<[f64]>,
    treatment_values: std::sync::Arc<[f64]>,
    nrows: usize,
    ncols: usize,
    outcome: LearnerSpec,
    treatment: LearnerSpec,
    folds: usize,
    seed: u64,
    row_index: std::sync::Arc<[u32]>,
    fold_assignment: Option<std::sync::Arc<[u32]>>,
    result: std::sync::Mutex<Option<std::sync::Arc<AipwPredictions>>>,
}

pub(crate) fn cached_aipw_nuisances(
    problem: &crate::propensity::PreparedPropensityProblem,
    outcome: LearnerSpec,
    treatment: LearnerSpec,
    folds: usize,
    ctx: &ExecutionContext,
) -> Result<std::sync::Arc<AipwPredictions>, EstimationError> {
    use std::sync::Arc;
    let check_cancel = || {
        if ctx.cancellation.is_cancelled() {
            Err(EstimationError::stats_msg("nuisance fitting cancelled"))
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    let entry = {
        let mut cache = problem
            .learner_cache
            .lock()
            .map_err(|_| EstimationError::stats_msg("nuisance cache lock poisoned"))?;
        if let Some(entry) = cache.iter().find(|entry| {
            Arc::ptr_eq(&entry.design, &problem.design_matrix)
                && Arc::ptr_eq(&entry.outcome_values, &problem.outcome)
                && Arc::ptr_eq(&entry.treatment_values, &problem.treatment)
                && entry.nrows == problem.nrows
                && entry.ncols == problem.design_ncols
                && entry.outcome == outcome
                && entry.treatment == treatment
                && entry.folds == folds
                && entry.seed == ctx.rng.master_seed()
                && Arc::ptr_eq(&entry.row_index, &problem.row_index)
                && match (&entry.fold_assignment, &problem.fold_assignment) {
                    (None, None) => true,
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    _ => false,
                }
        }) {
            Arc::clone(entry)
        } else {
            // Bound idle retention without evicting an in-flight initialization.
            while cache.len() >= 8 {
                let Some(index) = cache.iter().position(|entry| Arc::strong_count(entry) == 1)
                else {
                    break;
                };
                cache.remove(index);
            }
            let entry = Arc::new(AipwCacheEntry {
                design: Arc::clone(&problem.design_matrix),
                outcome_values: Arc::clone(&problem.outcome),
                treatment_values: Arc::clone(&problem.treatment),
                nrows: problem.nrows,
                ncols: problem.design_ncols,
                outcome,
                treatment,
                folds,
                seed: ctx.rng.master_seed(),
                row_index: Arc::clone(&problem.row_index),
                fold_assignment: problem.fold_assignment.clone(),
                result: std::sync::Mutex::new(None),
            });
            cache.push(Arc::clone(&entry));
            entry
        }
    };
    // Only callers for this exact key wait on the fit. Failure leaves it retryable.
    let mut slot = loop {
        check_cancel()?;
        match entry.result.try_lock() {
            Ok(slot) => break slot,
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(std::time::Duration::from_millis(5))
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(EstimationError::stats_msg("nuisance fit lock poisoned"));
            }
        }
    };
    check_cancel()?;
    if let Some(result) = slot.as_ref() {
        return Ok(Arc::clone(result));
    }
    // Preserve physical unit folds, including duplicated bootstrap rows.
    if folds < 2
        || folds > usize::from(u16::MAX) + 1
        || problem.nrows < folds
        || problem.nrows > u32::MAX as usize
    {
        return Err(EstimationError::data_msg("invalid retained nuisance fold count"));
    }
    let fold_ids: Vec<u16> = match problem.fold_assignment.as_deref() {
        Some(raw) => {
            if raw.len() != problem.nrows || raw.iter().any(|f| *f as usize >= folds) {
                return Err(EstimationError::data_msg("invalid retained nuisance fold plan"));
            }
            raw.iter().map(|&f| f as u16).collect()
        }
        None => {
            let arms: Vec<u32> = problem.treatment.iter().map(|&t| u32::from(t > 0.5)).collect();
            crossfit_fold_plan(&arms, &problem.row_index, folds, ctx.rng.master_seed())?
                .into_iter()
                .map(|f| f as u16)
                .collect()
        }
    };
    let result = Arc::new(cross_fit_aipw_nuisances(
        outcome,
        treatment,
        &problem.design_matrix,
        problem.nrows,
        problem.design_ncols,
        &problem.outcome,
        &problem.treatment,
        folds,
        ctx,
        fold_ids,
    )?);
    check_cancel()?;
    *slot = Some(Arc::clone(&result));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_plan_is_seeded_stratified_and_balanced() {
        // 100 units, stratum = parity: each fold must hold exactly 10 units of each stratum,
        // and the plan must change with the seed and not follow `i % folds`.
        let units: Vec<u32> = (0..100).collect();
        let strata: Vec<u32> = units.iter().map(|u| u % 2).collect();
        let plan = crossfit_fold_plan(&strata, &units, 5, 11).unwrap();
        for fold in 0..5u32 {
            for stratum in 0..2u32 {
                let count = (0..100).filter(|&i| plan[i] == fold && strata[i] == stratum).count();
                assert_eq!(count, 10, "fold {fold} stratum {stratum}");
            }
        }
        let modulo: Vec<u32> = (0..100).map(|i| i % 5).collect();
        assert_ne!(plan, modulo);
        assert_ne!(plan, crossfit_fold_plan(&strata, &units, 5, 12).unwrap());
        assert_eq!(plan, crossfit_fold_plan(&strata, &units, 5, 11).unwrap());
    }

    #[test]
    fn fold_plan_keeps_duplicated_units_together_and_refuses_too_few_units() {
        // A bootstrap resample repeats unit ids; every copy must share its unit's fold.
        let units = [7u32, 7, 3, 3, 3, 9, 1, 2, 4, 5, 6, 8];
        let strata = [0u32; 12];
        let plan = crossfit_fold_plan(&strata, &units, 4, 5).unwrap();
        for (i, &u) in units.iter().enumerate() {
            for (j, &v) in units.iter().enumerate() {
                if u == v {
                    assert_eq!(plan[i], plan[j], "unit {u}");
                }
            }
        }
        assert!(plan.iter().all(|&f| f < 4));
        let err = crossfit_fold_plan(&[0, 0, 0], &[1, 1, 2], 3, 5).unwrap_err();
        assert!(err.to_string().contains("distinct units"), "{err}");
        assert!(crossfit_fold_plan(&[0, 0], &[1, 2], 1, 5).is_err());
    }
}

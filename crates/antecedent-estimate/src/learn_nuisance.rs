//! Shared learner-backed nuisance helpers for DML / [`crate::DrLearner`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::needless_lifetimes,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    CrossFittedPrediction, DesignView, LearnerFactory, LearnerSpec, NuisanceDiagnostics,
    PredictionTask, RowSelection, TargetView, cross_fit, diagnose, resolve_for,
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
    cross_fit(factory.as_ref(), view, TargetView::new(y), folds, ctx, None).map_err(learn_err)
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
    let raw = problem.fold_assignment.as_deref().unwrap_or(&problem.row_index);
    if raw.len() != problem.nrows
        || problem.fold_assignment.is_some() && raw.iter().any(|f| *f as usize >= folds)
    {
        return Err(EstimationError::data_msg("invalid retained nuisance fold plan"));
    }
    let fold_ids = raw.iter().map(|f| (*f as usize % folds) as u16).collect();
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

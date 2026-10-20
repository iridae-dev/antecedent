//! Cross-fitted binary / discrete-joint AIPW scores.
//!
//! Fold assignment is a seeded, arm-stratified permutation of the distinct units
//! (`crossfit_plan::crossfit_fold_plan`), so it depends on `fold_seed` and unit ids rather
//! than on file order, and duplicated (bootstrap) rows stay in one fold.
//! These are uncentered AIPW scores. Centering estimates the efficient influence
//! function only under consistency, positivity, and nuisance convergence/rate
//! conditions. Cross-fitting alone does not guarantee valid inference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::cell::RefCell;
use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, ExecutionContext, OutcomeFunctional, VariableId};
use antecedent_stats::{
    FaerBackend, GlmOptions, PropensityWorkspace, fit_propensity, predict_propensity,
};

use crate::aipw::{AipwWorkspace, fit_outcome_models_weighted, predict_colmajor};
use crate::crossfit_plan::{
    CrossfitPlan, FoldRows, FoldSource, SHARED_FOLD_ID_ERRORS, rows_colmajor,
};
use crate::error::EstimationError;
use crate::propensity::{
    FailedFit, FallbackRecord, PreparedPropensityProblem, RidgeFoldInput, RidgeFoldSelection,
    clamp_scores, clip_of, fallback_provenance, fit_penalized_fold, gather,
    require_interior_propensities, split_by_treatment,
};
use crate::scores::{ScoreColumn, ScoreTable};
use crate::util::stats_err;

/// Default fold count for licensed AIPW scores.
pub const DEFAULT_AIPW_FOLDS: usize = 5;

/// Provenance tag frozen on the score table.
pub(crate) const AIPW_CROSSFIT_PROVENANCE: &str = "aipw.crossfit.v1";

/// Build a score table from a prepared propensity problem.
///
/// One column per `(arm, threshold)`. `thresholds = [None]` is the mean
/// functional. Arms are `0` (control) and `1` (treated) for binary ATE.
///
/// # Errors
///
/// Empty folds, missing arms in a fold, or GLM/OLS failure.
pub fn crossfit_binary_scores(
    problem: &PreparedPropensityProblem,
    query: &AverageEffectQuery,
    folds: usize,
    glm_options: &GlmOptions,
    backend: FaerBackend,
) -> Result<ScoreTable, EstimationError> {
    let thresholds = if query.outcome_functional.quantile_level().is_some() {
        crate::quantile::empirical_threshold_grid(&problem.outcome, 19)?
            .into_iter()
            .map(Some)
            .collect()
    } else {
        thresholds_of(&query.outcome_functional)
    };
    build_binary_scores(problem, query.treatment, &thresholds, folds, glm_options, backend)
}

/// Build binary-arm scores for an explicit threshold list (`None` = mean).
///
/// # Errors
///
/// Empty folds or nuisance failure.
///
/// Inside a [`crate::CrossfitNuisanceCache::scope`], a propensity or outcome fit whose
/// every input matches an earlier fit in the same cache is reused instead of refit; the
/// table is bit-identical either way.
pub fn build_binary_scores(
    problem: &PreparedPropensityProblem,
    treatment: VariableId,
    thresholds: &[Option<f64>],
    folds: usize,
    glm_options: &GlmOptions,
    backend: FaerBackend,
) -> Result<ScoreTable, EstimationError> {
    let build = ScoreBuild {
        treatment,
        thresholds,
        folds,
        glm_options,
        backend,
        share: true,
        ctx: None,
        fold_seed: None,
    };
    build_binary_scores_in(problem, &build).map(|(table, _)| table)
}

/// Everything a score-table build is configured by besides the prepared problem.
#[derive(Clone, Copy)]
pub(crate) struct ScoreBuild<'a> {
    pub(crate) treatment: VariableId,
    /// Outcome thresholds, `None` being the mean.
    pub(crate) thresholds: &'a [Option<f64>],
    pub(crate) folds: usize,
    pub(crate) glm_options: &'a GlmOptions,
    pub(crate) backend: FaerBackend,
    /// Whether a batch's shared nuisance cache may serve this table.
    pub(crate) share: bool,
    /// Context carrying cancellation and parallelism; `None` is a serial context seeded
    /// from the problem's fold seed.
    pub(crate) ctx: Option<&'a ExecutionContext>,
    /// Fold-plan seed. `None` uses [`PreparedPropensityProblem::fold_seed`].
    pub(crate) fold_seed: Option<u64>,
}

/// What the nuisance stage of a score-table build recorded beside the table.
#[derive(Clone, Debug, Default)]
pub(crate) struct NuisanceFit {
    /// Penalty (and, for a lasso, support) selected on each fold's training rows.
    pub(crate) selections: Vec<RidgeFoldSelection>,
    /// The declared GLM fallback, when the GLM propensity fit failed and it ran.
    pub(crate) fallback: Option<FallbackRecord>,
}

/// The score-table builder behind every public entry point.
///
/// With a ridge- or lasso-penalized propensity declared on `problem`, each fold's penalty (and
/// support) is chosen on that fold's training rows (see
/// [`crate::propensity::PropensityNuisance`]) and returned beside the table; the fit is never
/// served from or stored in a shared nuisance cache. `ctx` carries cancellation and
/// parallelism into penalty selection (`None`: a serial context seeded from the problem's fold
/// seed).
///
/// With a GLM-to-penalized fallback declared, a GLM propensity fit that fails on any fold
/// discards the partial table and rebuilds the *whole* table with the declared destination
/// (no fold mixes the two nuisances); the failed fit and the destination are returned and the
/// table's provenance names both. Any other failure is returned unchanged.
pub(crate) fn build_binary_scores_in(
    problem: &PreparedPropensityProblem,
    build: &ScoreBuild<'_>,
) -> Result<(ScoreTable, NuisanceFit), EstimationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::ScoreConstruction,
        || {
            let mut failed: Option<FailedFit> = None;
            let first = build_scores_pass(problem, build, &mut failed);
            let error = match first {
                Ok((table, selections)) => {
                    return Ok((table, NuisanceFit { selections, fallback: None }));
                }
                Err(error) => error,
            };
            let Some(failed_fit) = failed else {
                return Err(error);
            };
            let destination = problem.propensity.resolve_failed_fit(error)?;
            let mut rerun = problem.clone();
            rerun.propensity = destination;
            let (mut table, selections) = build_scores_pass(&rerun, build, &mut None)?;
            table.nuisance_provenance = Arc::from(format!(
                "{}{}",
                table.nuisance_provenance,
                fallback_provenance(&problem.propensity, &failed_fit)
            ));
            let record =
                FallbackRecord { failed_fit, destination: rerun.propensity.canonical_key() };
            Ok((table, NuisanceFit { selections, fallback: Some(record) }))
        },
    )
}

/// A fold's fitted propensity (raw and clipped) with the gathered fold designs.
struct FoldNuisance {
    design_train: Vec<f64>,
    design_valid: Vec<f64>,
    t_train: Vec<f64>,
    raw_e: Vec<f64>,
    e_valid: Vec<f64>,
}

/// One pass of the score-table builder; `failed` records the GLM propensity fit that failed.
#[allow(
    clippy::too_many_lines,
    reason = "fold-by-fold nuisance fitting under the shared-slot leases, then the score table; the lease scope spans the whole fit and every stage reads the fold ids"
)]
fn build_scores_pass(
    problem: &PreparedPropensityProblem,
    build: &ScoreBuild<'_>,
    failed: &mut Option<FailedFit>,
) -> Result<(ScoreTable, Vec<RidgeFoldSelection>), EstimationError> {
    let ScoreBuild { treatment, thresholds, folds, glm_options, backend, share, ctx, fold_seed } =
        *build;
    let fold_seed = fold_seed.unwrap_or(problem.fold_seed);
    problem.propensity.validate_for_execution()?;
    let penalized = problem.propensity.fold_penalty();
    let unpenalized = penalized.is_none();
    // Nuisances fit under sampling weights are not the unweighted fits a shared cache holds.
    let share = share && unpenalized && problem.sampling_weights.is_none();
    let sampling = problem.sampling_weights.as_ref().map(|s| &*s.weights);
    let local_ctx =
        (penalized.is_some() && ctx.is_none()).then(|| ExecutionContext::production(fold_seed, 1));
    let penalized = penalized.zip(ctx.or(local_ctx.as_ref()));
    if folds < 2 {
        return Err(EstimationError::unsupported("AIPW cross-fitting requires at least two folds"));
    }
    if problem.nrows < folds {
        return Err(EstimationError::data_msg("cross-fitting folds cannot exceed complete rows"));
    }
    if thresholds.is_empty() {
        return Err(EstimationError::data_msg("score table requires at least one functional"));
    }

    let n = problem.nrows;
    let ncols = problem.design_ncols;
    let clip = clip_of(problem.overlap);
    let arms = problem.fold_strata();
    let source = match (problem.fold_assignment.as_deref(), problem.fold_units.as_deref()) {
        (Some(ids), _) => FoldSource::Shared(ids),
        (None, Some(units)) => FoldSource::Clustered { units, seed: fold_seed },
        (None, None) => {
            FoldSource::Stratified { strata: &arms, units: &problem.row_index, seed: fold_seed }
        }
    };
    let plan = CrossfitPlan::derive(n, folds, source, &SHARED_FOLD_ID_ERRORS)?;
    let fold_ids = plan.fold_ids();

    let mut columns = Vec::new();
    for &c in thresholds {
        columns.push(ScoreColumn { arm: 0, threshold: c });
        columns.push(ScoreColumn { arm: 1, threshold: c });
    }
    let n_cols = columns.len();
    let mut scores = vec![0.0; n * n_cols];
    let mut propensities = vec![0.0; n * n_cols];

    let mut prop_ws = PropensityWorkspace::default();
    let mut out_ws = AipwWorkspace::default();
    let mut selections: Vec<RidgeFoldSelection> = Vec::new();

    // Shared nuisance slots (batch scope only). The propensity lease is held until the
    // table is built, so a query with the same key waits and then reuses the finished fit;
    // outcome slots are only ever locked under their propensity slot, so leases cannot
    // deadlock. A failed build leaves every slot empty.
    let fold_ids: Arc<[u32]> = Arc::from(fold_ids);
    // Certified reuse sits beside [`crate::CrossfitNuisanceCache`]. An inactive scope, a
    // penalized or weighted fit, or more than the mean functional keeps the fold loop.
    let certify = crate::nuisance_cache::certification_requested()
        && unpenalized
        && sampling.is_none()
        && thresholds.len() == 1
        && thresholds[0].is_none();
    let restored = if certify {
        restore_certified_mean_table(problem, treatment, &fold_ids, folds, clip)?
    } else {
        None
    };
    if let Some(table) = restored {
        return Ok((table, Vec::new()));
    }
    let certified = certify.then(|| RefCell::new(CertifiedMean::empty(n)));
    let lease = if share {
        crate::crossfit_cache::PropensityLease::acquire(|| crate::crossfit_cache::PropensityKey {
            design: Arc::clone(&problem.design_matrix),
            nrows: n,
            ncols,
            treatment: Arc::clone(&problem.treatment),
            fold_ids: Arc::clone(&fold_ids),
            folds,
            glm_options: *glm_options,
        })
    } else {
        None
    };
    let mut prop_slot =
        lease.as_ref().map(crate::crossfit_cache::PropensityLease::lock).transpose()?;
    let shared_e: Option<Arc<[f64]>> = prop_slot.as_ref().and_then(|slot| slot.as_ref().cloned());
    let mut fitted_e: Option<Vec<f64>> =
        (lease.is_some() && shared_e.is_none()).then(|| vec![0.0; n]);
    let y_sources: Vec<Arc<[f64]>> =
        thresholds.iter().map(|&c| Arc::from(transform_outcome(&problem.outcome, c))).collect();
    let outcome_leases: Vec<Option<crate::crossfit_cache::OutcomeLease>> = y_sources
        .iter()
        .map(|y| lease.as_ref().and_then(|l| l.outcome(|| Arc::clone(y))))
        .collect();
    let mut outcome_slots = outcome_leases
        .iter()
        .map(|l| l.as_ref().map(crate::crossfit_cache::OutcomeLease::lock).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    let shared_mu: Vec<crate::crossfit_cache::OutcomeSlot> =
        outcome_slots.iter().map(|slot| slot.as_ref().and_then(|s| s.as_ref().cloned())).collect();
    let mut fitted_mu: Vec<Option<(Vec<f64>, Vec<f64>)>> = outcome_slots
        .iter()
        .zip(&shared_mu)
        .map(|(slot, shared)| {
            (slot.is_some() && shared.is_none()).then(|| (vec![0.0; n], vec![0.0; n]))
        })
        .collect();

    // An unpenalized fit observes cancellation before every fold (the fold is the unit of work;
    // a penalized fit polls once per penalty instead); a cancelled fit reports no table.
    // Polling does not touch the numbers of an uncancelled fit.
    plan.run(
        Some("AIPW cross-fit fold is empty"),
        |_fold| {
            if penalized.is_none() && ctx.is_some_and(|c| c.cancellation.is_cancelled()) {
                return Err(if problem.fold_units.is_some() {
                    crate::cluster_dml_aipw::cancelled()
                } else {
                    crate::util::refuse(
                        antecedent_core::reason_code!("cancelled_no_claim"),
                        "aipw.cancelled",
                        "the cross-fitted AIPW fit was cancelled before every fold was fit; no \
                         estimate is reported and the stop is not a verdict on the data",
                    )
                });
            }
            Ok(())
        },
        |FoldRows { fold, train, valid }| {
            let design_train = rows_colmajor(&problem.design_matrix, n, ncols, train);
            let t_train = gather(&problem.treatment, train);
            let (treated_rows, control_rows) = split_by_treatment(&t_train);
            if treated_rows.is_empty() || control_rows.is_empty() {
                return Err(EstimationError::data_msg(
                    "AIPW cross-fit fold is missing a treatment arm",
                ));
            }
            let design_valid = rows_colmajor(&problem.design_matrix, n, ncols, valid);
            let mut e_valid = if let Some(shared) = shared_e.as_deref() {
                gather(shared, valid)
            } else if let Some(((fold_penalty, tuning), ridge_ctx)) = penalized {
                let units: Vec<_> = train.iter().map(|&i| problem.row_index[i]).collect();
                let train_weights = sampling.map(|weights| gather(weights, train));
                let (predicted, selection) = fit_penalized_fold(
                    fold_penalty,
                    tuning,
                    &RidgeFoldInput {
                        design_train: &design_train,
                        n_train: train.len(),
                        design_valid: &design_valid,
                        n_valid: valid.len(),
                        ncols,
                        t_train: &t_train,
                        units: &units,
                        fold,
                        seed: fold_seed,
                        weights: train_weights.as_deref(),
                    },
                    ridge_ctx,
                )?;
                crate::crossfit_cache::record_completed_fold_fit();
                selections.push(selection);
                predicted
            } else {
                let fit = match sampling {
                    Some(weights) => antecedent_stats::fit_propensity_weighted(
                        &design_train,
                        train.len(),
                        ncols,
                        &t_train,
                        &gather(weights, train),
                        &backend,
                        &mut prop_ws,
                        glm_options,
                    ),
                    None => fit_propensity(
                        &design_train,
                        train.len(),
                        ncols,
                        &t_train,
                        &backend,
                        &mut prop_ws,
                        glm_options,
                    ),
                }
                .map_err(stats_err)
                .and_then(|fit| {
                    fit.glm.require_ok().map_err(stats_err)?;
                    Ok(fit)
                })
                .inspect_err(|error| *failed = Some(FailedFit::from_error(fold, error)))?;
                crate::crossfit_cache::record_completed_fold_fit();
                let mut e_valid = vec![0.0; valid.len()];
                predict_propensity(
                    &design_valid,
                    valid.len(),
                    ncols,
                    &fit.coefficients,
                    &mut e_valid,
                )
                .map_err(stats_err)?;
                if let Some(out) = fitted_e.as_mut() {
                    for (k, &i) in valid.iter().enumerate() {
                        out[i] = e_valid[k];
                    }
                }
                if let Some(buffers) = certified.as_ref() {
                    buffers.borrow_mut().fitted_e = true;
                }
                e_valid
            };
            let raw_e = e_valid.clone();
            if let Some(buffers) = certified.as_ref() {
                let mut buffers = buffers.borrow_mut();
                for (k, &i) in valid.iter().enumerate() {
                    buffers.e[i] = raw_e[k];
                }
            }
            if let Some(clip_at) = clip {
                clamp_scores(&mut e_valid, clip_at);
            }
            // No hidden floor: with a clip every score is interior; without one a propensity of
            // exactly 0 or 1 (infinite weight) is refused rather than silently floored.
            require_interior_propensities(&e_valid)?;

            Ok(FoldNuisance { design_train, design_valid, t_train, raw_e, e_valid })
        },
        |FoldRows { train, valid, .. }, nuisance| {
            let FoldNuisance { design_train, design_valid, t_train, raw_e, e_valid } = nuisance;
            for t_idx in 0..thresholds.len() {
                let y_source = &y_sources[t_idx];
                let y_valid = gather(y_source, valid);
                if let Some((mu0, mu1)) = shared_mu[t_idx].as_ref() {
                    out_ws.mu0 = gather(mu0, valid);
                    out_ws.mu1 = gather(mu1, valid);
                } else {
                    let y_train = gather(y_source, train);
                    let train_weights = sampling.map(|weights| gather(weights, train));
                    let (beta0, beta1) = fit_outcome_models_weighted(
                        &design_train,
                        train.len(),
                        ncols,
                        &t_train,
                        &y_train,
                        train_weights.as_deref(),
                        backend,
                        &mut out_ws,
                    )?;
                    crate::crossfit_cache::record_completed_fold_fit();
                    predict_colmajor(&design_valid, valid.len(), ncols, &beta0, &mut out_ws.mu0);
                    predict_colmajor(&design_valid, valid.len(), ncols, &beta1, &mut out_ws.mu1);
                    if let Some((mu0, mu1)) = fitted_mu[t_idx].as_mut() {
                        for (k, &i) in valid.iter().enumerate() {
                            mu0[i] = out_ws.mu0[k];
                            mu1[i] = out_ws.mu1[k];
                        }
                    }
                    if let Some(buffers) = certified.as_ref() {
                        buffers.borrow_mut().fitted_mu = true;
                    }
                }
                if let Some(buffers) = certified.as_ref() {
                    let mut buffers = buffers.borrow_mut();
                    for (k, &i) in valid.iter().enumerate() {
                        buffers.mu0[i] = out_ws.mu0[k];
                        buffers.mu1[i] = out_ws.mu1[k];
                    }
                }

                let col0 = t_idx * 2;
                let col1 = col0 + 1;
                for (k, &i) in valid.iter().enumerate() {
                    let t = problem.treatment[i];
                    let y = y_valid[k];
                    let e = e_valid[k];
                    let m0 = out_ws.mu0[k];
                    let m1 = out_ws.mu1[k];
                    propensities[col0 * n + i] = 1.0 - raw_e[k];
                    propensities[col1 * n + i] = raw_e[k];
                    scores[col0 * n + i] = m0 + ((1.0 - t) / (1.0 - e)) * (y - m0);
                    scores[col1 * n + i] = m1 + (t / e) * (y - m1);
                }
            }
            Ok(())
        },
    )?;

    // Every fold succeeded: publish fresh fits and count each lease once.
    if let (Some(lease), Some(slot)) = (lease.as_ref(), prop_slot.as_mut()) {
        lease.record(fitted_e.is_none());
        if let Some(fitted) = fitted_e.take() {
            **slot = Some(Arc::from(fitted));
        }
    }
    for ((lease, slot), fitted) in
        outcome_leases.iter().zip(outcome_slots.iter_mut()).zip(fitted_mu)
    {
        if let (Some(lease), Some(slot)) = (lease.as_ref(), slot.as_mut()) {
            lease.record(fitted.is_none());
            if let Some((mu0, mu1)) = fitted {
                **slot = Some((Arc::from(mu0), Arc::from(mu1)));
            }
        }
    }
    drop(outcome_slots);
    drop(prop_slot);

    if let Some(buffers) = certified {
        store_certified_mean(problem, &fold_ids, buffers.into_inner())?;
    }

    let provenance = format!(
        "{}{}",
        if problem.shared_design {
            "aipw.crossfit.v1;batch.shared_design"
        } else {
            AIPW_CROSSFIT_PROVENANCE
        },
        problem.propensity.provenance_suffix_with(&selections, &problem.adjustment_set)
    );
    // Whole-cluster folds change the dependence structure the scores support, so the table
    // names them (and `provenance_withholds_interval` keeps its iid summaries unpublished).
    let provenance = match (problem.fold_assignment.as_deref(), problem.fold_units.as_deref()) {
        (None, Some(units)) => {
            format!(
                "{provenance}{}",
                crate::cluster_dml_aipw::provenance_suffix(units, problem.fold_unit_kind)
            )
        }
        _ => provenance,
    };
    let table = ScoreTable {
        observed_arm: problem
            .treatment
            .iter()
            .map(|t| u32::from(*t > 0.5))
            .collect::<Vec<_>>()
            .into(),
        propensities: propensities.into(),
        observed_outcome: Arc::clone(&problem.outcome),
        n_rows: n,
        row_index: Arc::clone(&problem.row_index),
        fold_ids,
        n_folds: u32::try_from(folds).unwrap_or(u32::MAX),
        scores: Arc::from(scores),
        columns: Arc::from(columns),
        adjustment_set: Arc::clone(&problem.adjustment_set),
        nuisance_provenance: Arc::from(provenance),
        propensity_clip: clip,
        treatment,
        intervened: Arc::from([]),
        preparation: None,
    };
    Ok((table, selections))
}

struct CertifiedMean {
    e: Vec<f64>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
    fitted_e: bool,
    fitted_mu: bool,
}

impl CertifiedMean {
    fn empty(rows: usize) -> Self {
        Self {
            e: vec![f64::NAN; rows],
            mu0: vec![f64::NAN; rows],
            mu1: vec![f64::NAN; rows],
            fitted_e: false,
            fitted_mu: false,
        }
    }
}

#[allow(clippy::needless_pass_by_value, reason = "map_err passes the cache error by value")]
fn cache_failure(error: crate::nuisance_cache::NuisanceCacheError) -> EstimationError {
    EstimationError::data_msg(error.to_string())
}

fn row_ids(problem: &PreparedPropensityProblem) -> Vec<u64> {
    problem.row_index.iter().map(|id| id.get()).collect()
}

fn restore_certified_mean_table(
    problem: &PreparedPropensityProblem,
    treatment: VariableId,
    fold_ids: &[u32],
    folds: usize,
    clip: Option<f64>,
) -> Result<Option<ScoreTable>, EstimationError> {
    let ids = row_ids(problem);
    let mut e = vec![0.0; problem.nrows];
    let mut mu0 = vec![0.0; problem.nrows];
    let mut mu1 = vec![0.0; problem.nrows];
    for fold in 0..folds {
        let fold_id = u32::try_from(fold).unwrap_or(u32::MAX);
        let rows: Vec<usize> = fold_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == fold_id)
            .map(|(index, _)| index)
            .collect();
        if rows.is_empty() {
            return Ok(None);
        }
        let row_ids: Vec<u64> = rows.iter().map(|&i| ids[i]).collect();
        let Some(e_rows) = crate::nuisance_cache::restore_active(
            crate::nuisance_cache::NuisanceRole::Treatment,
            fold_id,
            &row_ids,
        )
        .map_err(cache_failure)?
        else {
            return Ok(None);
        };
        let Some(m0) = crate::nuisance_cache::restore_active(
            crate::nuisance_cache::NuisanceRole::Outcome { level: 0 },
            fold_id,
            &row_ids,
        )
        .map_err(cache_failure)?
        else {
            return Ok(None);
        };
        let Some(m1) = crate::nuisance_cache::restore_active(
            crate::nuisance_cache::NuisanceRole::Outcome { level: 1 },
            fold_id,
            &row_ids,
        )
        .map_err(cache_failure)?
        else {
            return Ok(None);
        };
        if e_rows.len() != rows.len() || m0.len() != rows.len() || m1.len() != rows.len() {
            return Err(EstimationError::data_msg(
                "certified AIPW restore length does not match the fold",
            ));
        }
        for (k, &i) in rows.iter().enumerate() {
            e[i] = e_rows[k];
            mu0[i] = m0[k];
            mu1[i] = m1[k];
        }
    }
    crate::nuisance_cache::record_active_hits(3);
    mean_score_table(problem, treatment, fold_ids, folds, clip, &e, &mu0, &mu1, &[]).map(Some)
}

fn store_certified_mean(
    problem: &PreparedPropensityProblem,
    fold_ids: &[u32],
    buffers: CertifiedMean,
) -> Result<(), EstimationError> {
    if buffers.e.iter().any(|value| !value.is_finite())
        || buffers.mu0.iter().any(|value| !value.is_finite())
        || buffers.mu1.iter().any(|value| !value.is_finite())
    {
        return Err(EstimationError::data_msg(
            "certified AIPW nuisance was not written for every row",
        ));
    }
    let ids = row_ids(problem);
    let roles = [
        (crate::nuisance_cache::NuisanceRole::Treatment, buffers.e),
        (crate::nuisance_cache::NuisanceRole::Outcome { level: 0 }, buffers.mu0),
        (crate::nuisance_cache::NuisanceRole::Outcome { level: 1 }, buffers.mu1),
    ];
    let mut stored = Vec::with_capacity(roles.len());
    for (role, values) in roles {
        let blocks = crate::nuisance_cache::blocks_by_fold(&ids, fold_ids, &values)
            .map_err(cache_failure)?;
        stored.push((role, blocks));
    }
    for (role, blocks) in &stored {
        crate::nuisance_cache::store_active(*role, blocks).map_err(cache_failure)?;
    }
    let fitted = u64::from(buffers.fitted_e) + u64::from(buffers.fitted_mu) * 2;
    crate::nuisance_cache::record_certified_fits(fitted);
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the mean score table takes the same aligned row inputs the fold loop writes"
)]
fn mean_score_table(
    problem: &PreparedPropensityProblem,
    treatment: VariableId,
    fold_ids: &[u32],
    folds: usize,
    clip: Option<f64>,
    raw_e: &[f64],
    mu0: &[f64],
    mu1: &[f64],
    selections: &[RidgeFoldSelection],
) -> Result<ScoreTable, EstimationError> {
    let n = problem.nrows;
    let mut e_used = raw_e.to_vec();
    if let Some(clip_at) = clip {
        clamp_scores(&mut e_used, clip_at);
    }
    require_interior_propensities(&e_used)?;
    let mut scores = vec![0.0; n * 2];
    let mut propensities = vec![0.0; n * 2];
    for i in 0..n {
        let t = problem.treatment[i];
        let y = problem.outcome[i];
        let e = e_used[i];
        let m0 = mu0[i];
        let m1 = mu1[i];
        propensities[i] = 1.0 - raw_e[i];
        propensities[n + i] = raw_e[i];
        scores[i] = m0 + ((1.0 - t) / (1.0 - e)) * (y - m0);
        scores[n + i] = m1 + (t / e) * (y - m1);
    }
    let provenance = format!(
        "{}{}",
        if problem.shared_design {
            "aipw.crossfit.v1;batch.shared_design"
        } else {
            AIPW_CROSSFIT_PROVENANCE
        },
        problem.propensity.provenance_suffix_with(selections, &problem.adjustment_set)
    );
    let provenance = match (problem.fold_assignment.as_deref(), problem.fold_units.as_deref()) {
        (None, Some(units)) => format!(
            "{provenance}{}",
            crate::cluster_dml_aipw::provenance_suffix(units, problem.fold_unit_kind)
        ),
        _ => provenance,
    };
    Ok(ScoreTable {
        observed_arm: problem
            .treatment
            .iter()
            .map(|t| u32::from(*t > 0.5))
            .collect::<Vec<_>>()
            .into(),
        propensities: propensities.into(),
        observed_outcome: Arc::clone(&problem.outcome),
        n_rows: n,
        row_index: Arc::clone(&problem.row_index),
        fold_ids: Arc::from(fold_ids),
        n_folds: u32::try_from(folds).unwrap_or(u32::MAX),
        scores: Arc::from(scores),
        columns: Arc::from([
            ScoreColumn { arm: 0, threshold: None },
            ScoreColumn { arm: 1, threshold: None },
        ]),
        adjustment_set: Arc::clone(&problem.adjustment_set),
        nuisance_provenance: Arc::from(provenance),
        propensity_clip: clip,
        treatment,
        intervened: Arc::from([]),
        preparation: None,
    })
}

/// Execute AIPW prepared requests through the certified nuisance cache.
///
/// # Errors
///
/// [`crate::nuisance_cache::NuisanceCacheError::WrongFamily`] when a request is not AIPW,
/// and the compilation and execution errors of the shared subplan.
#[must_use = "a refused AIPW subplan is not a certified fit"]
pub fn execute_certified_aipw(
    cache: &mut crate::nuisance_cache::NuisanceCache,
    requests: &[crate::nuisance_cache::PreparedFamilyRequest],
    produce: impl FnMut(
        &crate::nuisance_cache::NuisanceCacheKey,
        crate::nuisance_cache::NuisanceRole,
    ) -> Result<
        Vec<crate::nuisance_cache::BlockedOutOfFoldPredictions>,
        crate::nuisance_cache::NuisanceCacheError,
    >,
) -> Result<crate::nuisance_cache::BatchExecution, crate::nuisance_cache::NuisanceCacheError> {
    crate::nuisance_cache::execute_family_requests(
        crate::nuisance_cache::EstimatorFamily::Aipw,
        cache,
        requests,
        produce,
    )
}

/// Map an outcome functional to score-table thresholds (`None` = mean).
#[must_use]
pub(crate) fn thresholds_of(functional: &OutcomeFunctional) -> Vec<Option<f64>> {
    match functional.thresholds() {
        None => vec![None],
        Some(cs) => cs.into_iter().map(Some).collect(),
    }
}

fn transform_outcome(y: &[f64], threshold: Option<f64>) -> Vec<f64> {
    match threshold {
        None => y.to_vec(),
        Some(c) => y.iter().map(|&yi| if yi > c { 1.0 } else { 0.0 }).collect(),
    }
}

/// Weighted support diagnostics for a binary score table.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedSupport {
    /// Kish `n_eff` under `w`.
    pub n_eff: f64,
    /// Per-arm Kish `n_eff` using `w * 1{A=a}`.
    pub n_eff_by_arm: Vec<f64>,
    /// Min / max propensity among rows with `w > 0` (if supplied).
    pub propensity_range: Option<(f64, f64)>,
    /// Whether weighted overlap is usable.
    pub overlap_ok: bool,
}

/// Weighted overlap for binary treatment under target weights.
#[must_use]
pub(crate) fn weighted_support(
    treatment: &[f64],
    weights: &[f64],
    propensity: Option<&[f64]>,
    min_n_eff_arm: f64,
) -> WeightedSupport {
    let mut w0 = Vec::new();
    let mut w1 = Vec::new();
    let mut w_all = Vec::new();
    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;
    for (i, &t) in treatment.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(0.0);
        if w <= 0.0 {
            continue;
        }
        w_all.push(w);
        if t > 0.5 {
            w1.push(w);
        } else {
            w0.push(w);
        }
        if let Some(e) = propensity.and_then(|p| p.get(i)).copied() {
            p_min = p_min.min(e);
            p_max = p_max.max(e);
        }
    }
    let n_eff = crate::joint_if::kish_n_eff(&w_all);
    let n0 = crate::joint_if::kish_n_eff(&w0);
    let n1 = crate::joint_if::kish_n_eff(&w1);
    let range = if p_min.is_finite() { Some((p_min, p_max)) } else { None };
    // Same gate as the score-table support (retarget), including the propensity range and the
    // extreme-propensity share, whenever propensities are supplied.
    let extreme_share = propensity.map_or(0.0, |p| {
        crate::retarget::extreme_propensity_share(
            std::iter::once(p),
            weights,
            crate::overlap::DEFAULT_PROPENSITY_CLIP,
        )
    });
    let overlap_ok = crate::retarget::overlap_gate(
        &[n0, n1],
        min_n_eff_arm,
        range,
        propensity.is_some(),
        extreme_share,
    );
    WeightedSupport { n_eff, n_eff_by_arm: vec![n0, n1], propensity_range: range, overlap_ok }
}

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use super::*;
    use crate::aipw::AipwAte;
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;
    use std::sync::Arc;

    fn confounded(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x51);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let p = antecedent_kernels::sigmoid(zi - 0.5);
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + 0.25 * standard_normal(&mut rng);
        }
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols = [t, y, z]
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(v),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn crossfit_scores_recover_ate_two() {
        let (data, estimand) = confounded(1_200, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let table =
            crossfit_binary_scores(&prep, &query, 5, &est.glm_options, est.backend).unwrap();
        let summary = table.summarize(None).unwrap();
        let ate = summary.means[1] - summary.means[0];
        assert!((ate - 2.0).abs() < 0.35, "crossfit ate={ate}");
        assert_eq!(table.n_folds, 5);
        assert_eq!(table.columns.len(), 2);
    }

    fn table_bits(table: &ScoreTable) -> (Vec<u64>, Vec<u64>) {
        (
            table.scores.iter().map(|v| v.to_bits()).collect(),
            table.propensities.iter().map(|v| v.to_bits()).collect(),
        )
    }

    fn certified_aipw_key(
        problem: &PreparedPropensityProblem,
    ) -> crate::nuisance_cache::NuisanceCacheKey {
        use crate::nuisance_cache::{
            DesignDependency, FoldAssignment, FoldRow, NuisanceCacheKey, NumericalDependencies,
            ProviderDependency, TargetDependency, TreatmentDependency,
        };
        use antecedent_core::{DesignDependencyId, ProviderId, SemanticDigest};
        let arms = problem.fold_strata();
        let plan = CrossfitPlan::derive(
            problem.nrows,
            5,
            FoldSource::Stratified {
                strata: &arms,
                units: &problem.row_index,
                seed: problem.fold_seed,
            },
            &SHARED_FOLD_ID_ERRORS,
        )
        .unwrap();
        let fold_ids = plan.fold_ids();
        let rows: Vec<FoldRow> = (0..problem.nrows)
            .map(|row| FoldRow::new(problem.row_index[row].get(), fold_ids[row]))
            .collect();
        let mut bytes = [0u8; 32];
        bytes[0] = 1;
        let digest = SemanticDigest::from_bytes(bytes);
        NuisanceCacheKey::new(
            TargetDependency::new(digest),
            DesignDependency::new(DesignDependencyId::from_raw(7)),
            FoldAssignment::try_new(5, &rows, &[]).unwrap(),
            TreatmentDependency::try_new("arm", digest).unwrap(),
            ProviderDependency::new(ProviderId::from_raw(4), digest),
            NumericalDependencies::try_new(antecedent_core::NumericalPolicy::strict(), &[])
                .unwrap(),
        )
    }

    #[test]
    fn certified_aipw_second_request_reuses_the_fit() {
        use crate::crossfit_cache::CrossfitNuisanceCache;
        use crate::nuisance_cache::{NuisanceCache, NuisanceCacheKey, with_certified_cache};
        let (data, estimand) = confounded(400, 61);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte::new();
        let problem = est.prepare(&data, &estimand, &query).unwrap();
        let solo =
            crossfit_binary_scores(&problem, &query, 5, &est.glm_options, est.backend).unwrap();
        let key = certified_aipw_key(&problem);
        let mut cache = NuisanceCache::new();
        let score = |cache: &mut NuisanceCache, key: &NuisanceCacheKey| {
            let crossfit = CrossfitNuisanceCache::new();
            let ((table, receipt), _) = crossfit.scope(|| {
                with_certified_cache(cache, key, || {
                    crossfit_binary_scores(&problem, &query, 5, &est.glm_options, est.backend)
                        .unwrap()
                })
            });
            (table, receipt, crossfit.completed_fold_fits())
        };
        let (first, receipt, fold_fits) = score(&mut cache, &key);
        assert_eq!(table_bits(&first), table_bits(&solo));
        assert_eq!((receipt.scans, receipt.fits, receipt.hits), (1, 3, 0));
        assert_eq!(fold_fits, 10);
        let (again, receipt, fold_fits) = score(&mut cache, &key);
        assert_eq!(table_bits(&again), table_bits(&solo));
        assert_eq!((receipt.scans, receipt.fits, receipt.hits), (0, 0, 3));
        assert_eq!(fold_fits, 0, "the certified hit does not refit");

        let mut bytes = [0u8; 32];
        bytes[0] = 2;
        let changed = NuisanceCacheKey::new(
            crate::nuisance_cache::TargetDependency::new(
                antecedent_core::SemanticDigest::from_bytes(bytes),
            ),
            key.design(),
            key.folds().clone(),
            key.treatment().clone(),
            key.provider(),
            key.numerical().clone(),
        );
        let (_, receipt, fold_fits) = score(&mut cache, &changed);
        assert_eq!((receipt.scans, receipt.fits), (1, 3));
        assert_eq!(fold_fits, 10, "a different target misses");
    }

    #[test]
    fn aipw_family_executor_reuses_the_certified_fit() {
        use crate::nuisance_cache::{
            BlockedOutOfFoldPredictions, CertifiedSource, DesignDependency, EstimatorFamily,
            FoldAssignment, FoldRow, NuisanceCache, NuisanceCacheKey, NuisanceRole,
            NumericalDependencies, PredictedRow, PreparedFamilyRequest, ProviderDependency,
            RequestColumns, TargetDependency, TreatmentDependency,
        };
        use antecedent_core::{DesignDependencyId, ProviderId, SemanticDigest, SourceDependencyId};
        let mut bytes = [0u8; 32];
        bytes[0] = 4;
        let digest = SemanticDigest::from_bytes(bytes);
        let folds =
            FoldAssignment::try_new(2, &[FoldRow::new(10, 0), FoldRow::new(40, 1)], &[]).unwrap();
        let key = NuisanceCacheKey::new(
            TargetDependency::new(digest),
            DesignDependency::new(DesignDependencyId::from_raw(7)),
            folds,
            TreatmentDependency::try_new("arm", digest).unwrap(),
            ProviderDependency::new(ProviderId::from_raw(4), digest),
            NumericalDependencies::try_new(antecedent_core::NumericalPolicy::strict(), &[])
                .unwrap(),
        );
        let prepared = PreparedFamilyRequest::try_new(
            EstimatorFamily::Aipw,
            RequestColumns::try_new("outcome", "arm", &["z"]).unwrap(),
            CertifiedSource::new(SourceDependencyId::from_raw(3), digest),
            key,
            &[NuisanceRole::Treatment],
        )
        .unwrap();
        let mut cache = NuisanceCache::new();
        let calls = std::cell::Cell::new(0u64);
        let run = |cache: &mut NuisanceCache, requests: &[PreparedFamilyRequest]| {
            execute_certified_aipw(cache, requests, |_, _| {
                calls.set(calls.get().saturating_add(1));
                let rows = [
                    PredictedRow::try_new(10, 0.2).unwrap(),
                    PredictedRow::try_new(40, 0.4).unwrap(),
                ];
                Ok(vec![
                    BlockedOutOfFoldPredictions::try_new(0, &rows[..1]).unwrap(),
                    BlockedOutOfFoldPredictions::try_new(1, &rows[1..]).unwrap(),
                ])
            })
        };
        let first = run(&mut cache, &[prepared.clone(), prepared.clone()]).unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!((first.receipt().scans, first.receipt().fits), (1, 1));
        calls.set(0);
        let second = run(&mut cache, &[prepared]).unwrap();
        assert_eq!(calls.get(), 0);
        assert_eq!(
            (second.receipt().scans, second.receipt().fits, second.receipt().hits),
            (0, 0, 1)
        );
        assert_eq!(first.fits()[0], second.fits()[0]);
    }

    #[test]
    fn shared_nuisances_reuse_only_identical_fits_and_match_unshared_bits() {
        use crate::crossfit_cache::CrossfitNuisanceCache;
        let (data, estimand) = confounded(400, 61);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte::new();
        let base = est.prepare(&data, &estimand, &query).unwrap();
        let build = |p: &PreparedPropensityProblem, glm: &GlmOptions| {
            crossfit_binary_scores(p, &query, 5, glm, est.backend).unwrap()
        };
        let solo = build(&base, &est.glm_options);
        // Same treatment, design and folds but another outcome: shares only e(Z).
        let mut other_outcome = base.clone();
        other_outcome.outcome = base.outcome.iter().map(|y| 0.5 * y + 1.0).collect();
        let solo_other = build(&other_outcome, &est.glm_options);
        // Another fold seed (fold plan), and other GLM options: share nothing.
        let mut other_seed = base.clone();
        other_seed.fold_seed = base.fold_seed.wrapping_add(1);
        let solo_seed = build(&other_seed, &est.glm_options);
        let other_glm = GlmOptions { max_iter: est.glm_options.max_iter + 1, ..est.glm_options };
        let solo_glm = build(&base, &other_glm);

        let cache = CrossfitNuisanceCache::new();
        let (first, a) = cache.scope(|| build(&base, &est.glm_options));
        assert_eq!(cache.completed_fold_fits(), 10);
        let (again, b) = cache.scope(|| build(&base, &est.glm_options));
        assert_eq!(cache.completed_fold_fits(), 10, "cache hits do no fold-fit work");
        let (outcome, c) = cache.scope(|| build(&other_outcome, &est.glm_options));
        assert_eq!(cache.completed_fold_fits(), 15);
        let (seeded, d) = cache.scope(|| build(&other_seed, &est.glm_options));
        assert_eq!(cache.completed_fold_fits(), 25);
        let (glm, e) = cache.scope(|| build(&base, &other_glm));
        assert_eq!(cache.completed_fold_fits(), 35);
        assert_eq!(table_bits(&first), table_bits(&solo));
        assert_eq!(table_bits(&again), table_bits(&solo));
        assert_eq!(table_bits(&outcome), table_bits(&solo_other));
        assert_eq!(table_bits(&seeded), table_bits(&solo_seed));
        assert_eq!(table_bits(&glm), table_bits(&solo_glm));

        let stats = cache.stats();
        // base (fit), repeat (reuse both), other outcome (reuse e, fit mu),
        // other seed (fit both), other GLM options (fit both).
        assert_eq!((stats.propensity_fits, stats.propensity_reuses), (3, 2));
        assert_eq!((stats.outcome_fits, stats.outcome_reuses), (4, 1));
        assert_eq!(cache.shared_with_another_scope(&a), (true, true));
        assert_eq!(cache.shared_with_another_scope(&b), (true, true));
        assert_eq!(cache.shared_with_another_scope(&c), (true, false));
        assert_eq!(cache.shared_with_another_scope(&d), (false, false));
        assert_eq!(cache.shared_with_another_scope(&e), (false, false));

        // Outside a scope nothing is read or recorded.
        let unscoped = build(&base, &est.glm_options);
        assert_eq!(table_bits(&unscoped), table_bits(&solo));
        assert_eq!(cache.stats(), stats);
    }

    #[test]
    fn unshared_penalized_folds_are_counted_in_the_active_scope() {
        use crate::crossfit_cache::CrossfitNuisanceCache;
        use crate::propensity::{PropensityNuisance, RidgeTuning};

        let (data, estimand) = confounded(400, 62);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte::new();
        let mut problem = est.prepare(&data, &estimand, &query).unwrap();
        problem.propensity =
            PropensityNuisance::ridge_logistic(RidgeTuning::new(&[0.1, 1.0], 3).unwrap());

        let cache = CrossfitNuisanceCache::new();
        let (table, _) = cache.scope(|| {
            crossfit_binary_scores(&problem, &query, 5, &est.glm_options, est.backend).unwrap()
        });
        assert_eq!(table.n_folds, 5);
        assert_eq!(cache.completed_fold_fits(), 10);
        assert_eq!(cache.stats(), crate::crossfit_cache::CrossfitCacheStats::default());
    }

    #[test]
    fn fallback_counts_completed_prefix_and_successful_destination_fits() {
        use crate::crossfit_cache::CrossfitNuisanceCache;
        use crate::propensity::{NuisanceFallback, PropensityNuisance, RidgeTuning};

        let (data, estimand) = confounded(400, 63);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte::new();
        let mut problem = est.prepare(&data, &estimand, &query).unwrap();
        let n = problem.nrows;
        let mut treatment: Vec<_> =
            (0..n).map(|row| f64::from(u8::from(problem.design_matrix[n + row] > 0.0))).collect();
        // The intentionally discordant fold makes fold zero's training fit finite. It is
        // excluded from fold one's training rows, where the remaining data are separated.
        for row in (1..n).step_by(5) {
            treatment[row] = 1.0 - treatment[row];
        }
        problem.treatment = Arc::from(treatment);
        problem.fold_assignment =
            Some(Arc::from((0..n).map(|row| u32::try_from(row % 5).unwrap()).collect::<Vec<_>>()));
        problem.propensity = PropensityNuisance::default().with_fallback(
            NuisanceFallback::RidgeLogistic(RidgeTuning::new(&[1.0, 10.0, 100.0], 3).unwrap()),
        );

        let build = ScoreBuild {
            treatment: query.treatment,
            thresholds: &[None],
            folds: 5,
            glm_options: &est.glm_options,
            backend: est.backend,
            share: true,
            ctx: None,
            fold_seed: None,
        };
        let cache = CrossfitNuisanceCache::new();
        let (result, _) = cache.scope(|| build_binary_scores_in(&problem, &build));
        let (table, fitted) = result.unwrap();
        assert_eq!(table.n_folds, 5);
        assert_eq!(fitted.fallback.as_ref().unwrap().failed_fit.fold, 1);
        // The fold-zero GLM propensity and outcome pair completed before fold one's
        // failed GLM fit; the successful ridge fallback then refit both nuisances in all
        // five folds. The failed GLM attempt itself is not counted.
        assert_eq!(cache.completed_fold_fits(), 12);
    }

    #[test]
    fn invalid_shared_fold_cannot_leave_unscored_rows() {
        let (data, estimand) = confounded(200, 317);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte::new();
        let mut prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ids: Vec<_> = (0..prep.nrows).map(|i| u32::try_from(i % 5).unwrap()).collect();
        ids[0] = 5;
        prep.fold_assignment = Some(ids.into());
        let error =
            crossfit_binary_scores(&prep, &query, 5, &est.glm_options, est.backend).unwrap_err();
        assert!(error.to_string().contains("fold ids"));
    }

    #[test]
    fn custom_weights_shift_the_mean() {
        let (data, estimand) = confounded(800, 4);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::AllObserved);
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let table =
            crossfit_binary_scores(&prep, &query, 5, &est.glm_options, est.backend).unwrap();
        let w: Vec<f64> = (0..table.n_rows).map(|i| if i % 2 == 0 { 2.0 } else { 0.5 }).collect();
        let a = table.summarize(None).unwrap();
        let b = table.summarize(Some(&w)).unwrap();
        assert_eq!(a.means.len(), b.means.len());
    }
}

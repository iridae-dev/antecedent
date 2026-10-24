//! Shared learner-backed nuisance helpers for DML / [`crate::DrLearner`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    CrossFittedPrediction, DesignView, LearnerFactory, LearnerSpec, NuisanceDiagnostics,
    PredictionTask, RowSelection, TargetView, cross_fit_with_folds_weighted, diagnose, resolve_for,
};

use crate::crossfit_plan::{
    CrossfitPlan, FoldIdErrors, FoldRows, FoldSource, SHARED_FOLD_ID_ERRORS, derive_fold_ids,
};
use crate::error::EstimationError;

#[allow(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adapter: the error arrives by value at every call site"
)]
pub(crate) fn learn_err(err: antecedent_learn::LearnError) -> EstimationError {
    match err {
        antecedent_learn::LearnError::Cancelled => crate::util::refuse(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "cross_fit.cancelled",
            "a cross-fit fold was cancelled; no nuisance fit is reported and the stop is not a \
             verdict on the data",
        ),
        other => EstimationError::stats_msg(other.to_string()),
    }
}

/// The stop of a cancelled cross-fit: no nuisance bundle is returned or cached.
fn nuisance_cancelled() -> EstimationError {
    crate::util::refuse(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "dml.cancelled",
        "the cross-fitted nuisance fit was cancelled; no estimate or score table is reported \
         and the stop is not a verdict on the data",
    )
}

pub(crate) fn intercept_is_constant(design: &[f64], nrows: usize) -> bool {
    if nrows == 0 {
        return false;
    }
    design[..nrows].iter().all(|&v| (v - 1.0).abs() <= 1e-12)
}

/// Drop a constant intercept column for tree learners; keep it for parametric ones.
pub(crate) fn design_for_spec(
    spec: LearnerSpec,
    design: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<DesignView<'_>, EstimationError> {
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

/// A column-major `nrows × ncols` nuisance design borrowed from its owner.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NuisanceDesign<'a> {
    pub(crate) values: &'a [f64],
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<'a> NuisanceDesign<'a> {
    /// The retained design of a prepared propensity problem.
    pub(crate) fn of(problem: &'a crate::propensity::PreparedPropensityProblem) -> Self {
        Self {
            values: problem.design_matrix.as_ref(),
            nrows: problem.nrows,
            ncols: problem.design_ncols,
        }
    }
}

/// The learners of the outcome-regression and treatment-propensity nuisances.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AipwLearners {
    pub(crate) outcome: LearnerSpec,
    pub(crate) treatment: LearnerSpec,
}

/// Refuse prior weights for a learner that cannot fit under them, naming the learner.
pub(crate) fn require_weight_capable(
    factory: &dyn LearnerFactory,
    spec: LearnerSpec,
) -> Result<(), EstimationError> {
    if factory.capabilities().sample_weights {
        return Ok(());
    }
    Err(crate::sampling_weights::unsupported_route(&format!(
        "the {} nuisance learner cannot be fit under sampling weights; use a linear, ridge, \
         logistic or gradient-boosted learner, which carry prior weights through every fold",
        spec.identity()
    )))
}

pub(crate) fn cross_fit_nuisance(
    spec: LearnerSpec,
    task: PredictionTask,
    design: NuisanceDesign<'_>,
    y: &[f64],
    weights: Option<&[f64]>,
    folds: usize,
    ctx: &ExecutionContext,
) -> Result<CrossFittedPrediction, EstimationError> {
    let NuisanceDesign { values: design, nrows, ncols } = design;
    if ctx.cancellation.is_cancelled() {
        return Err(nuisance_cancelled());
    }
    let (factory, _) = resolve_nuisance(spec, task, design, nrows, ncols, y, ctx)?;
    if weights.is_some() {
        require_weight_capable(factory.as_ref(), spec)?;
    }
    let view = design_for_spec(spec, design, nrows, ncols)?;
    let rows: Vec<antecedent_data::GlobalId> = (0..nrows)
        .map(|i| {
            u64::try_from(i).map(antecedent_data::GlobalId::new).map_err(|_| {
                EstimationError::data_msg(
                    "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let plan = derive_fold_ids(
        nrows,
        folds,
        FoldSource::Stratified {
            strata: &vec![0; nrows],
            units: &rows,
            seed: ctx.rng.master_seed(),
        },
        &SHARED_FOLD_ID_ERRORS,
    )?;
    let plan: Vec<u16> = plan
        .into_iter()
        .map(|f| {
            u16::try_from(f).map_err(|_| {
                EstimationError::unsupported("cross-fitting supports at most 65536 folds")
            })
        })
        .collect::<Result<_, _>>()?;
    cross_fit_with_folds_weighted(
        factory.as_ref(),
        view,
        TargetView::new(y),
        weights,
        plan,
        ctx,
        None,
    )
    .map_err(learn_err)
}

/// Per-arm outcome models + propensity, trained on `train ∩ arm` and scored on the valid fold.
pub(crate) fn cross_fit_aipw_nuisances(
    learners: AipwLearners,
    design: NuisanceDesign<'_>,
    y: &[f64],
    t: &[f64],
    (folds, fold_ids): (usize, Vec<u16>),
    ctx: &ExecutionContext,
    weights: Option<&[f64]>,
) -> Result<AipwPredictions, EstimationError> {
    let AipwLearners { outcome, treatment } = learners;
    let NuisanceDesign { values: design, nrows, ncols } = design;
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
    if weights.is_some() {
        require_weight_capable(outcome_factory.as_ref(), outcome)?;
        require_weight_capable(treat_factory.as_ref(), treatment)?;
    }
    let x_outcome = design_for_spec(outcome, design, nrows, ncols)?;
    let x_treat = design_for_spec(treatment, design, nrows, ncols)?;
    let plan = CrossfitPlan::from_fold_ids(folds, fold_ids.iter().map(|&f| u32::from(f)).collect());
    let models = AipwFoldModels {
        outcome: outcome_factory.as_ref(),
        treatment: treat_factory.as_ref(),
        x_outcome,
        x_treat,
        y,
        t,
        weights,
    };
    let mut mu0 = vec![0.0; nrows];
    let mut mu1 = vec![0.0; nrows];
    let mut ehat = vec![0.0; nrows];
    let mut treat_prov = Vec::with_capacity(folds);
    // Folds are fitted ahead of the scatter by at most one result per thread, so the fold
    // outputs alive at once are bounded by the window, not by the fold count.
    let window = usize::try_from(ctx.parallelism.max_threads.get()).unwrap_or(1).max(1);
    plan.pipeline_folds(
        ctx,
        window,
        |rows, inner| fit_aipw_fold(&models, rows, inner),
        |_, part| {
            treat_prov.extend(part.provenance);
            for (i, pred) in part.valid.iter().copied().zip(part.mu0) {
                mu0[usize::try_from(i).expect("row id")] = pred;
            }
            for (i, pred) in part.valid.iter().copied().zip(part.mu1) {
                mu1[usize::try_from(i).expect("row id")] = pred;
            }
            for (i, pred) in part.valid.iter().copied().zip(part.ehat) {
                ehat[usize::try_from(i).expect("row id")] = pred;
            }
            Ok(())
        },
    )
    .map_err(|error| match error {
        antecedent_core::execution::MapExecutionError::Work(error)
            if !ctx.cancellation.is_cancelled() =>
        {
            learn_err(error)
        }
        _ => nuisance_cancelled(),
    })?;
    let treat = CrossFittedPrediction {
        predictions: ehat.clone(),
        fold_assignment: fold_ids,
        model_provenance: treat_prov,
        validation: diagnose(PredictionTask::BinaryProbability, t, &ehat),
    };
    Ok((mu0, mu1, ehat, treat))
}

/// What every fold of the AIPW cross-fit fits and scores: the two learner factories, their
/// designs and the targets.
#[derive(Clone, Copy)]
struct AipwFoldModels<'a> {
    outcome: &'a dyn LearnerFactory,
    treatment: &'a dyn LearnerFactory,
    x_outcome: DesignView<'a>,
    x_treat: DesignView<'a>,
    y: &'a [f64],
    t: &'a [f64],
    /// Prior weight of every row (mean one), or `None` for the unweighted fits.
    weights: Option<&'a [f64]>,
}

struct AipwFold {
    valid: Vec<u64>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
    ehat: Vec<f64>,
    provenance: [antecedent_learn::LearnerProvenance; 3],
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "i indexes the rows of a problem whose row count the caller checked to fit u32"
)]
fn fit_aipw_fold(
    models: &AipwFoldModels<'_>,
    rows: FoldRows<'_>,
    ctx: &ExecutionContext,
) -> Result<AipwFold, antecedent_learn::LearnError> {
    let AipwFoldModels { outcome, treatment, x_outcome, x_treat, y, t, weights } = *models;
    // The fold is the unit of work: cancellation is observed before its three fits.
    if ctx.cancellation.is_cancelled() {
        return Err(antecedent_learn::LearnError::Backend(
            "nuisance fitting cancelled".to_string(),
        ));
    }
    let mut train = Vec::with_capacity(rows.train.len());
    let mut train0 = Vec::new();
    let mut train1 = Vec::new();
    for &i in rows.train {
        let idx = u64::try_from(i).expect("row id");
        train.push(idx);
        if t[i] > 0.5 {
            train1.push(idx);
        } else {
            train0.push(idx);
        }
    }
    let valid: Vec<u64> = rows.valid.iter().map(|&i| u64::try_from(i).expect("row id")).collect();
    if train.is_empty() || valid.is_empty() || train0.is_empty() || train1.is_empty() {
        return Err(antecedent_learn::LearnError::Shape {
            message: "empty DML/AIPW train, valid, or treatment arm in a fold",
        });
    }
    let y_view = TargetView::new(y);
    let t_view = TargetView::new(t);
    let mu0 =
        outcome.fit(x_outcome.with_rows(RowSelection::new(&train0))?, y_view, weights, ctx)?;
    let mu1 =
        outcome.fit(x_outcome.with_rows(RowSelection::new(&train1))?, y_view, weights, ctx)?;
    let ehat =
        treatment.fit(x_treat.with_rows(RowSelection::new(&train))?, t_view, weights, ctx)?;
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
    if let Some(c) = clip {
        for score in scores.iter_mut() {
            *score = crate::overlap::clip_propensity(*score, c);
        }
    }
}

/// The clipped retained propensities, refusing an exact 0 or 1 (or a non-finite value).
///
/// With a clip every value is interior; without one such a propensity carries an infinite
/// inverse-probability weight, which is refused here as it is by the cross-fit AIPW score
/// table instead of surfacing later as non-finite scores.
///
/// # Errors
///
/// An overlap refusal naming the propensity-score fields when a clipped value is not in
/// the open unit interval.
pub(crate) fn guarded_propensity(
    raw: &[f64],
    clip: Option<f64>,
) -> Result<Vec<f64>, EstimationError> {
    let mut ehat = raw.to_vec();
    clip_propensity(&mut ehat, clip);
    crate::propensity::require_interior_propensities(&ehat)?;
    Ok(ehat)
}

pub(crate) fn iid_se(values: &[f64]) -> f64 {
    antecedent_stats::iid_se(values)
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

/// Per-row fold ids for the retained AIPW nuisances: the recorded assignment when the
/// problem carries one, otherwise a fresh seeded plan stratified by treatment arm.
#[allow(
    clippy::cast_possible_truncation,
    reason = "fold ids are below the fold count, which was checked above to be at most u16::MAX + 1"
)]
fn retained_fold_ids(
    problem: &crate::propensity::PreparedPropensityProblem,
    folds: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<u16>, EstimationError> {
    const RETAINED_PLAN_ERRORS: FoldIdErrors = FoldIdErrors {
        length: "invalid retained nuisance fold plan",
        units: "invalid retained nuisance fold plan",
        range: "invalid retained nuisance fold plan",
    };
    // Preserve physical unit folds, including duplicated bootstrap rows.
    if folds < 2
        || folds > usize::from(u16::MAX) + 1
        || problem.nrows < folds
        || problem.nrows > u32::MAX as usize
    {
        return Err(EstimationError::data_msg("invalid retained nuisance fold count"));
    }
    let arms = problem.fold_strata();
    // Whole-unit folds when a cluster-DML unit is declared: the nuisances scoring a unit are
    // then fit on other units only.
    let source = match (problem.fold_assignment.as_deref(), problem.fold_units.as_deref()) {
        (Some(raw), _) => FoldSource::Shared(raw),
        (None, Some(units)) => FoldSource::Clustered { units, seed: ctx.rng.master_seed() },
        (None, None) => FoldSource::Stratified {
            strata: &arms,
            units: &problem.row_index,
            seed: ctx.rng.master_seed(),
        },
    };
    Ok(derive_fold_ids(problem.nrows, folds, source, &RETAINED_PLAN_ERRORS)?
        .into_iter()
        .map(|f| f as u16)
        .collect())
}

fn require_populated_arms(treatment: &[f64]) -> Result<(), EstimationError> {
    // Preparation validates binary values with its numerical tolerance. Match the
    // executing fold classifier so accepted near-zero/near-one values retain their arms.
    if treatment.iter().any(|&value| value <= 0.5) && treatment.iter().any(|&value| value > 0.5) {
        return Ok(());
    }
    Err(crate::util::refuse(
        antecedent_core::reason_code!("arm_not_populated"),
        "dml.arm_not_populated",
        "both treatment arms must remain populated after complete-case selection",
    ))
}

/// The learner-backed cross-fitted nuisances of `problem`, fit once per key.
///
/// The key is derived from the design, outcome and treatment columns, the weights, the row
/// identities, the realized folds and the fold-plan inputs, both learner specifications and the
/// seed ([`crate::nuisance_keys::learner_bundle_key`]), so a changed input is a miss and an
/// identical request (even from another prepared problem of the same cache scope) reads the
/// retained bundle. Retention is charged to the pool of `ctx`, or to the pool of the cache the
/// problem shares; when it cannot be retained the bundle is fit for this caller alone.
pub(crate) fn cached_aipw_nuisances(
    problem: &crate::propensity::PreparedPropensityProblem,
    outcome: LearnerSpec,
    treatment: LearnerSpec,
    folds: usize,
    ctx: &ExecutionContext,
) -> Result<std::sync::Arc<AipwPredictions>, EstimationError> {
    let check_cancel = || {
        if ctx.cancellation.is_cancelled() { Err(nuisance_cancelled()) } else { Ok(()) }
    };
    check_cancel()?;
    require_populated_arms(&problem.treatment)?;
    let fold_ids = retained_fold_ids(problem, folds, ctx)?;
    // Four f64 vectors of one value per row, the fold ids, and the per-fold provenance.
    let payload = (problem.nrows as u64).saturating_mul(4 * 8 + 2).saturating_add(4096);
    let key = crate::nuisance_keys::learner_bundle_key(
        problem,
        outcome,
        treatment,
        folds,
        ctx.rng.master_seed(),
        &fold_ids,
    );
    let compute = || {
        check_cancel()?;
        cross_fit_aipw_nuisances(
            AipwLearners { outcome, treatment },
            NuisanceDesign::of(problem),
            &problem.outcome,
            &problem.treatment,
            (folds, fold_ids.clone()),
            ctx,
            problem.sampling_weights.as_ref().map(|s| &*s.weights),
        )
    };
    let Some(key) = key else {
        return compute().map(std::sync::Arc::new);
    };
    let (bundle, reused) =
        problem.learner_cache.get_or_compute(key, payload, Some(ctx), &check_cancel, compute)?;
    if reused {
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
            1,
        );
    }
    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_propensity_refuses_boundary_values_unless_a_clip_makes_them_interior() {
        let raw = [0.4, 0.0, 1.0, 0.7];
        let refused = guarded_propensity(&raw, None).unwrap_err();
        assert!(matches!(refused, EstimationError::OverlapWithFields { .. }), "{refused}");
        assert!(guarded_propensity(&[0.4, f64::NAN], None).is_err());
        let clipped = guarded_propensity(&raw, Some(0.05)).unwrap();
        assert_eq!(clipped, vec![0.4, 0.05, 0.95, 0.7]);
        assert_eq!(guarded_propensity(&[0.2, 1e-12], None).unwrap(), vec![0.2, 1e-12]);
    }

    #[test]
    fn populated_arm_guard_preserves_prepared_binary_tolerance() {
        assert!(require_populated_arms(&[1e-13, 1.0 - 1e-13]).is_ok());
        assert!(require_populated_arms(&[-1e-13, 1.0 + 1e-13]).is_ok());
        for absent in [&[1e-13, -1e-13][..], &[1.0 - 1e-13, 1.0 + 1e-13][..]] {
            assert!(matches!(
                require_populated_arms(absent),
                Err(EstimationError::Refused { code: "arm_not_populated", .. })
            ));
        }
    }
}

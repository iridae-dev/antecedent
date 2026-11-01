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
use antecedent_learn::aipw_crossfit::{AipwCrossFitError, ColumnMajorDesign};
use antecedent_learn::{
    CrossFittedPrediction, DesignView, LearnerFactory, LearnerSpec, NuisanceDiagnostics,
    PredictionTask, TargetView, cross_fit_with_folds_weighted, resolve_for,
};

use crate::crossfit_plan::{FoldIdErrors, FoldSource, derive_fold_ids};
use crate::error::EstimationError;

#[allow(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adapter: the error arrives by value at every call site"
)]
pub(crate) fn learn_err(err: antecedent_learn::LearnError) -> EstimationError {
    match err {
        antecedent_learn::LearnError::Cancelled
        | antecedent_learn::LearnError::Stats(
            antecedent_stats::StatsError::Cancelled
            | antecedent_stats::StatsError::Admission(
                antecedent_core::execution::AdmissionError::Cancelled,
            ),
        )
        | antecedent_learn::LearnError::Admission(
            antecedent_core::execution::AdmissionError::Cancelled,
        )
        | antecedent_learn::LearnError::Probability(
            antecedent_prob::ProbError::Admission(
                antecedent_core::execution::AdmissionError::Cancelled,
            )
            | antecedent_prob::ProbError::Cancelled { .. },
        ) => nuisance_cancelled(),
        antecedent_learn::LearnError::Admission(error)
        | antecedent_learn::LearnError::Stats(antecedent_stats::StatsError::Admission(error))
        | antecedent_learn::LearnError::Probability(antecedent_prob::ProbError::Admission(error)) => {
            EstimationError::Admission(error)
        }
        other => EstimationError::Learner(other),
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

/// The estimator-side error of an AIPW cross-fit stop: each case keeps the error it had
/// before the computation moved to the learner layer.
fn aipw_cross_fit_error(error: AipwCrossFitError) -> EstimationError {
    match error {
        AipwCrossFitError::Learner(error) => learn_err(error),
        AipwCrossFitError::Cancelled => nuisance_cancelled(),
        AipwCrossFitError::Admission(error) => EstimationError::Admission(error),
        AipwCrossFitError::WeightsUnsupported { message } => {
            crate::sampling_weights::unsupported_route(&message)
        }
    }
}

/// Drop a constant intercept column for tree learners; keep it for parametric ones.
pub(crate) fn design_for_spec(
    spec: LearnerSpec,
    design: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<DesignView<'_>, EstimationError> {
    antecedent_learn::aipw_crossfit::design_for_spec(
        spec,
        ColumnMajorDesign { values: design, nrows, ncols },
    )
    .map_err(learn_err)
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

    /// The same design as the learner layer's plain column-major view.
    fn plain(self) -> ColumnMajorDesign<'a> {
        ColumnMajorDesign { values: self.values, nrows: self.nrows, ncols: self.ncols }
    }
}

pub(crate) use antecedent_learn::aipw_crossfit::AipwLearners;
pub(crate) use antecedent_learn::aipw_crossfit::intercept_is_constant;

/// Refuse prior weights for a learner that cannot fit under them, naming the learner.
pub(crate) fn require_weight_capable(
    factory: &dyn LearnerFactory,
    spec: LearnerSpec,
) -> Result<(), EstimationError> {
    antecedent_learn::aipw_crossfit::require_weight_capable(factory, spec)
        .map_err(aipw_cross_fit_error)
}

pub(crate) fn cross_fit_nuisance(
    spec: LearnerSpec,
    task: PredictionTask,
    design: NuisanceDesign<'_>,
    y: &[f64],
    weights: Option<&[f64]>,
    fold_ids: Vec<u16>,
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
    cross_fit_with_folds_weighted(
        factory.as_ref(),
        view,
        TargetView::new(y),
        weights,
        fold_ids,
        ctx,
        None,
    )
    .map_err(learn_err)
}

/// Per-arm outcome models + propensity, trained on `train ∩ arm` and scored on the valid fold.
#[allow(clippy::too_many_arguments, reason = "the inputs describe one estimation procedure")]
pub(crate) fn cross_fit_aipw_nuisances(
    learners: AipwLearners,
    design: NuisanceDesign<'_>,
    y: &[f64],
    t: &[f64],
    (folds, fold_ids): (usize, Vec<u16>),
    ctx: &ExecutionContext,
    weights: Option<&[f64]>,
    admit_boundary_saturation: bool,
) -> Result<AipwPredictions, EstimationError> {
    antecedent_learn::aipw_crossfit::cross_fit_aipw_nuisances(
        learners,
        design.plain(),
        y,
        t,
        (folds, fold_ids),
        ctx,
        weights,
        admit_boundary_saturation,
    )
    .map_err(aipw_cross_fit_error)
}

pub(crate) fn clip_propensity(scores: &mut [f64], clip: Option<f64>) {
    antecedent_learn::aipw_crossfit::clip_propensities(scores, clip);
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

pub(crate) use antecedent_learn::aipw_crossfit::aipw_scores;

/// Immutable predictions shared across compatible estimators.
pub(crate) type AipwPredictions = (Vec<f64>, Vec<f64>, Vec<f64>, CrossFittedPrediction);

/// Per-row fold ids for the retained AIPW nuisances: the recorded assignment when the
/// problem carries one, otherwise a fresh seeded plan stratified by treatment arm.
#[allow(
    clippy::cast_possible_truncation,
    reason = "fold ids are below the fold count, which was checked above to be at most u16::MAX + 1"
)]
pub(crate) fn retained_fold_ids(
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
    if antecedent_learn::dr_replay::both_arms_populated(treatment) {
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
/// A boundary-saturated GLM refusal from an AIPW nuisance is the propensity's: the outcome
/// nuisance resolves for regression, where no logistic learner is admitted. It refuses as
/// `positivity_violated`; every other error is unchanged.
fn propensity_nuisance_error(error: EstimationError) -> EstimationError {
    match error {
        EstimationError::Learner(antecedent_learn::LearnError::Stats(
            stats @ antecedent_stats::StatsError::GlmRefused {
                kind: antecedent_stats::GlmRefusalKind::BoundarySaturated,
                ..
            },
        )) => crate::propensity::propensity_fit_error(stats),
        other => other,
    }
}

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
    // A declared overlap clip or trim governs propensities at the boundary; without one a
    // saturated propensity nuisance refuses as `positivity_violated`.
    let admit = crate::propensity::admits_boundary_saturation(problem.overlap);
    let key = crate::nuisance_keys::learner_bundle_key(
        problem,
        outcome,
        treatment,
        folds,
        ctx.rng.master_seed(),
        &fold_ids,
        admit,
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
            admit,
        )
        .map_err(propensity_nuisance_error)
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
    fn nuisance_cancellation_preserves_detail_and_non_cancel_admission() {
        use antecedent_core::execution::AdmissionError;
        use antecedent_learn::LearnError;
        for input in [
            LearnError::Cancelled,
            LearnError::Admission(AdmissionError::Cancelled),
            LearnError::Stats(antecedent_stats::StatsError::Cancelled),
            LearnError::Stats(antecedent_stats::StatsError::Admission(AdmissionError::Cancelled)),
            LearnError::Probability(antecedent_prob::ProbError::Admission(
                AdmissionError::Cancelled,
            )),
            LearnError::Probability(antecedent_prob::ProbError::Cancelled {
                stage: "hmc.cancelled",
            }),
        ] {
            let error = learn_err(input);
            assert!(
                matches!(error, EstimationError::Refused { code: "cancelled_no_claim", ref message }
                if message.detail() == Some("dml.cancelled"))
            );
        }
        assert!(matches!(
            learn_err(LearnError::Admission(AdmissionError::Overflow)),
            EstimationError::Admission(AdmissionError::Overflow)
        ));
    }

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

//! The cross-fitted nuisances of an augmented inverse-probability-weighted (AIPW) score.
//!
//! [`cross_fit_aipw_nuisances`] fits, for every fold of a [`CrossfitPlan`], one outcome model per
//! treatment arm and one propensity model on the fold's training rows, predicts the held-out
//! rows with them, and scatters the predictions back to row order. [`aipw_scores`] turns those
//! predictions into the doubly-robust score. Neither names an estimator type, a prepared
//! problem or a cache, so any layer that must replay a cross-fitted estimate runs the same
//! fits on the same rows and obtains the same numbers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, MapExecutionError};
use antecedent_core::treatment_indicator::is_treated_indicator;
use thiserror::Error;

use crate::crossfit::{CrossFittedPrediction, diagnose};
use crate::crossfit_plan::{CrossfitPlan, FoldRows};
use crate::design::{DesignView, RowSelection, TargetView};
use crate::error::LearnError;
use crate::learner::{LearnerFactory, LearnerProvenance, PredictionTask};
use crate::spec::{LearnerSpec, resolve_for, resolve_propensity_for};

/// Out-of-fold outcome predictions under control and treatment, the out-of-fold propensity, and
/// the propensity's cross-fit record.
pub type AipwPredictions = (Vec<f64>, Vec<f64>, Vec<f64>, CrossFittedPrediction);

/// Why the AIPW cross-fit stopped.
#[derive(Debug, Error)]
pub enum AipwCrossFitError {
    /// A learner could not be resolved, fitted or evaluated; also a malformed fold.
    #[error(transparent)]
    Learner(#[from] LearnError),
    /// The cancellation token of the execution context stopped the work, whichever way the
    /// stop surfaced; no partial result is reported.
    #[error("the cross-fitted nuisance fit was cancelled")]
    Cancelled,
    /// Worker or scratch admission refused the work (other than by cancellation).
    #[error(transparent)]
    Admission(AdmissionError),
    /// Prior weights were supplied but a nuisance learner cannot fit under them.
    #[error("{message}")]
    WeightsUnsupported {
        /// The refusal text, naming the learner.
        message: String,
    },
}

/// A plain column-major `nrows × ncols` design: `values[col * nrows + row]`.
#[derive(Clone, Copy, Debug)]
pub struct ColumnMajorDesign<'a> {
    /// The column-major values (at least `nrows * ncols` long).
    pub values: &'a [f64],
    /// The row count.
    pub nrows: usize,
    /// The column count.
    pub ncols: usize,
}

/// The learners of the outcome-regression and treatment-propensity nuisances.
#[derive(Clone, Copy, Debug)]
pub struct AipwLearners {
    /// The learner of the per-arm outcome regressions.
    pub outcome: LearnerSpec,
    /// The learner of the treatment propensity.
    pub treatment: LearnerSpec,
}

/// Whether the first `nrows` entries of `design` (its first column) are all one.
#[must_use]
pub fn intercept_is_constant(design: &[f64], nrows: usize) -> bool {
    if nrows == 0 {
        return false;
    }
    design[..nrows].iter().all(|&v| (v - 1.0).abs() <= 1e-12)
}

/// The design a learner of `spec` fits on: a constant intercept column is dropped for tree and
/// neural learners and kept for parametric ones.
///
/// # Errors
///
/// [`LearnError::Shape`] when the buffer is shorter than the matrix.
pub fn design_for_spec(
    spec: LearnerSpec,
    design: ColumnMajorDesign<'_>,
) -> Result<DesignView<'_>, LearnError> {
    let ColumnMajorDesign { values, nrows, ncols } = design;
    let drop_intercept = matches!(
        spec,
        LearnerSpec::GradientBoostedTrees(_)
            | LearnerSpec::RandomForest(_)
            | LearnerSpec::NeuralNet(_)
    ) && ncols > 1
        && intercept_is_constant(values, nrows);
    if drop_intercept {
        DesignView::from_column_major(&values[nrows..], nrows, ncols - 1)
    } else {
        DesignView::from_column_major(values, nrows, ncols)
    }
}

/// Refuse prior weights for a learner that cannot fit under them, naming the learner.
///
/// # Errors
///
/// [`AipwCrossFitError::WeightsUnsupported`] when the factory does not carry sample weights.
pub fn require_weight_capable(
    factory: &dyn LearnerFactory,
    spec: LearnerSpec,
) -> Result<(), AipwCrossFitError> {
    if factory.capabilities().sample_weights {
        return Ok(());
    }
    Err(AipwCrossFitError::WeightsUnsupported {
        message: format!(
            "the {} nuisance learner cannot be fit under sampling weights; use a linear, ridge, \
             logistic or gradient-boosted learner, which carry prior weights through every fold",
            spec.identity()
        ),
    })
}

/// Clamp one propensity into `[clip, 1 − clip]`.
///
/// This is the only propensity clip. `clip` is used as given (no `1e-6` floor). Callers that
/// hold an overlap policy have already required `clip ∈ (0, 0.5)`.
#[must_use]
pub fn clip_propensity(score: f64, clip: f64) -> f64 {
    debug_assert!(
        clip.is_finite() && clip > 0.0 && clip < 0.5,
        "propensity clip must lie in (0, 0.5)"
    );
    score.clamp(clip, 1.0 - clip)
}

/// Clip every propensity in place when a clip is declared; leave them as given otherwise.
pub fn clip_propensities(scores: &mut [f64], clip: Option<f64>) {
    if let Some(c) = clip {
        for score in scores.iter_mut() {
            *score = clip_propensity(*score, c);
        }
    }
}

/// The AIPW score of every row:
/// `(μ₁ − μ₀) + t (y − μ₁) / e − (1 − t) (y − μ₀) / (1 − e)`.
#[must_use]
pub fn aipw_scores(t: &[f64], y: &[f64], e: &[f64], mu0: &[f64], mu1: &[f64]) -> Vec<f64> {
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

/// Per-arm outcome models + propensity, trained on `train ∩ arm` and scored on the valid fold.
///
/// `fold_ids` assigns every row to one of `folds` folds. Folds are fitted on leased workers
/// ahead of the scatter by at most one result per thread, so the fold outputs alive at once are
/// bounded by the window, not by the fold count. The result is `(mu0, mu1, ehat, propensity
/// record)`, each vector in row order.
///
/// # Errors
///
/// [`AipwCrossFitError::Learner`] for an unresolvable learner, a fit failure, or a fold with an
/// empty training, validation or treatment-arm side;
/// [`AipwCrossFitError::WeightsUnsupported`] when `weights` are given and a learner cannot carry
/// them; [`AipwCrossFitError::Cancelled`] when the token of `ctx` stopped the work;
/// [`AipwCrossFitError::Admission`] when the shared scheduler refused it.
#[allow(
    clippy::too_many_arguments,
    reason = "the checked cross-fit inputs are independent parts of the operation"
)]
pub fn cross_fit_aipw_nuisances(
    learners: AipwLearners,
    design: ColumnMajorDesign<'_>,
    y: &[f64],
    t: &[f64],
    (folds, fold_ids): (usize, Vec<u16>),
    ctx: &ExecutionContext,
    weights: Option<&[f64]>,
    admit_boundary_saturation: bool,
) -> Result<AipwPredictions, AipwCrossFitError> {
    let AipwLearners { outcome, treatment } = learners;
    let nrows = design.nrows;
    let outcome_factory = resolve_for(outcome, PredictionTask::Regression)?;
    let treat_factory = resolve_propensity_for(
        treatment,
        PredictionTask::BinaryProbability,
        admit_boundary_saturation,
    )?;
    if weights.is_some() {
        require_weight_capable(outcome_factory.as_ref(), outcome)?;
        require_weight_capable(treat_factory.as_ref(), treatment)?;
    }
    let x_outcome = design_for_spec(outcome, design)?;
    let x_treat = design_for_spec(treatment, design)?;
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
    let window = usize::try_from(ctx.parallelism.max_threads.get()).unwrap_or(1).max(1);
    plan.pipeline_folds(
        ctx,
        window,
        |rows, inner| fit_aipw_fold(&models, rows, inner),
        |_, part| {
            treat_prov.extend(part.provenance);
            for (((&row, p0), p1), pe) in
                part.valid.iter().zip(part.mu0).zip(part.mu1).zip(part.ehat)
            {
                mu0[row] = p0;
                mu1[row] = p1;
                ehat[row] = pe;
            }
            Ok(())
        },
    )
    .map_err(|error| match error {
        MapExecutionError::Work(error) if !ctx.cancellation.is_cancelled() => {
            AipwCrossFitError::Learner(error)
        }
        MapExecutionError::Work(_) | MapExecutionError::Admission(AdmissionError::Cancelled) => {
            AipwCrossFitError::Cancelled
        }
        MapExecutionError::Admission(error) => AipwCrossFitError::Admission(error),
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
    /// Validation rows of the fold, in row order.
    valid: Vec<usize>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
    ehat: Vec<f64>,
    provenance: [LearnerProvenance; 3],
}

/// A row index as the `u64` the learner row selections take.
fn selection_row(row: usize) -> Result<u64, LearnError> {
    u64::try_from(row).map_err(|_| LearnError::Shape {
        message: "a row index exceeds the 64-bit selection range",
    })
}

fn fit_aipw_fold(
    models: &AipwFoldModels<'_>,
    rows: FoldRows<'_>,
    ctx: &ExecutionContext,
) -> Result<AipwFold, LearnError> {
    let AipwFoldModels { outcome, treatment, x_outcome, x_treat, y, t, weights } = *models;
    // The fold is the unit of work: cancellation is observed before its three fits.
    if ctx.cancellation.is_cancelled() {
        return Err(LearnError::Backend("nuisance fitting cancelled".to_string()));
    }
    let mut train = Vec::with_capacity(rows.train.len());
    let mut train0 = Vec::new();
    let mut train1 = Vec::new();
    for &i in rows.train {
        let idx = selection_row(i)?;
        train.push(idx);
        if is_treated_indicator(t[i]) {
            train1.push(idx);
        } else {
            train0.push(idx);
        }
    }
    let valid: Vec<u64> = rows.valid.iter().map(|&i| selection_row(i)).collect::<Result<_, _>>()?;
    if train.is_empty() || valid.is_empty() || train0.is_empty() || train1.is_empty() {
        return Err(LearnError::Shape {
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
        valid: rows.valid.to_vec(),
        mu0: m0,
        mu1: m1,
        ehat: e,
        provenance: [mu0.provenance(), mu1.provenance(), ehat.provenance()],
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "the controls build small synthetic designs from small indices and compare exact copies"
)]
mod tests {
    use super::*;
    use crate::spec::{LinearSpec, LogisticSpec};

    const OUTCOME_LINEAR: LearnerSpec = LearnerSpec::Linear(LinearSpec {});
    const UNPENALIZED_LOGISTIC: LearnerSpec =
        LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 });
    const LEARNERS: AipwLearners =
        AipwLearners { outcome: OUTCOME_LINEAR, treatment: UNPENALIZED_LOGISTIC };

    fn run(
        design: &[f64],
        ncols: usize,
        y: &[f64],
        t: &[f64],
        folds: usize,
        ids: &[u16],
        ctx: &ExecutionContext,
    ) -> Result<AipwPredictions, AipwCrossFitError> {
        cross_fit_aipw_nuisances(
            LEARNERS,
            ColumnMajorDesign { values: design, nrows: y.len(), ncols },
            y,
            t,
            (folds, ids.to_vec()),
            ctx,
            None,
            false,
        )
    }

    /// Mean of `values` over the rows `keep` accepts.
    fn mean_where(values: &[f64], keep: impl Fn(usize) -> bool) -> f64 {
        let picked: Vec<f64> =
            values.iter().enumerate().filter(|&(i, _)| keep(i)).map(|(_, &v)| v).collect();
        assert!(!picked.is_empty());
        picked.iter().sum::<f64>() / picked.len() as f64
    }

    #[test]
    fn an_intercept_only_design_predicts_the_training_arm_means_and_treated_fraction() {
        let n = 40;
        let folds = 4;
        let ids: Vec<u16> = (0..n).map(|i| (i % folds) as u16).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 5 < 2))).collect();
        let y: Vec<f64> =
            (0..n).map(|i| ((i * i) % 7) as f64 + 2.0 * t[i] + 0.25 * (i % 3) as f64).collect();
        let ones = vec![1.0; n];
        let ctx = ExecutionContext::for_tests(3);
        let (mu0, mu1, ehat, record) = run(&ones, 1, &y, &t, folds, &ids, &ctx).unwrap();
        assert_eq!((mu0.len(), mu1.len(), ehat.len()), (n, n, n));
        for i in 0..n {
            let fold = ids[i];
            let in_train = |j: usize| ids[j] != fold;
            let arm0 = mean_where(&y, |j| in_train(j) && t[j] < 0.5);
            let arm1 = mean_where(&y, |j| in_train(j) && t[j] > 0.5);
            let frac = mean_where(&t, in_train);
            assert!((mu0[i] - arm0).abs() <= 1e-10 * (1.0 + arm0.abs()), "mu0 row {i}");
            assert!((mu1[i] - arm1).abs() <= 1e-10 * (1.0 + arm1.abs()), "mu1 row {i}");
            // The unpenalized logistic MLE stops when the linear predictor moves by at most
            // `1e-8 (1 + |eta|)`; that bounds the probability error through the sigmoid slope.
            let eta = (frac / (1.0 - frac)).ln();
            let bound = 1e-8 * (1.0 + eta.abs()) * frac * (1.0 - frac) + 1e-12;
            assert!((ehat[i] - frac).abs() <= bound, "ehat row {i}: {} vs {frac}", ehat[i]);
        }
        assert_eq!(record.predictions, ehat);
        assert_eq!(record.fold_assignment, ids);
        assert_eq!(record.model_provenance.len(), 3 * folds);
    }

    /// `y = a + b t + c x` with the treatment overlapping `x` in every fold.
    fn noiseless_linear(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (a, b, c) = (1.5, 2.0, -0.75);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 13.0).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let y: Vec<f64> = (0..n).map(|i| a + b * t[i] + c * x[i]).collect();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&x);
        (design, x, t, y)
    }

    #[test]
    fn a_noiseless_linear_outcome_recovers_the_effect_and_the_scores_average_to_it() {
        let (a, b, c) = (1.5, 2.0, -0.75);
        let n = 52;
        let folds = 4;
        let ids: Vec<u16> = (0..n).map(|i| (i % folds) as u16).collect();
        let (design, x, t, y) = noiseless_linear(n);
        let ctx = ExecutionContext::for_tests(5);
        let (mu0, mu1, ehat, _) = run(&design, 2, &y, &t, folds, &ids, &ctx).unwrap();
        for i in 0..n {
            assert!((mu1[i] - mu0[i] - b).abs() <= 1e-9, "effect row {i}");
            assert!((mu0[i] - (a + c * x[i])).abs() <= 1e-9, "mu0 row {i}");
            assert!(ehat[i] > 0.0 && ehat[i] < 1.0, "propensity row {i}");
        }
        let scores = aipw_scores(&t, &y, &ehat, &mu0, &mu1);
        let mean = scores.iter().sum::<f64>() / n as f64;
        assert!((mean - b).abs() <= 1e-9, "{mean}");
    }

    #[test]
    fn the_score_formula_matches_a_hand_computation() {
        let scores = aipw_scores(&[1.0, 0.0], &[3.0, 1.0], &[0.5, 0.25], &[0.5, 0.5], &[2.0, 2.0]);
        // Treated row: 1.5 + (3 - 2) / 0.5. Control row: 1.5 - (1 - 0.5) / 0.75.
        assert!((scores[0] - 3.5).abs() <= 1e-15);
        assert!((scores[1] - (1.5 - 0.5 / 0.75)).abs() <= 1e-15);
    }

    #[test]
    fn the_clip_bounds_a_propensity_and_leaves_interior_values() {
        assert_eq!(clip_propensity(0.0, 0.05), 0.05);
        assert_eq!(clip_propensity(1.0, 0.05), 0.95);
        assert_eq!(clip_propensity(0.4, 0.05), 0.4);
        let mut scores = [0.0, 0.4, 1.0];
        clip_propensities(&mut scores, None);
        assert_eq!(scores, [0.0, 0.4, 1.0]);
        clip_propensities(&mut scores, Some(0.05));
        assert_eq!(scores, [0.05, 0.4, 0.95]);
    }

    #[test]
    fn a_constant_first_column_is_the_only_intercept() {
        assert!(intercept_is_constant(&[1.0, 1.0, 1.0, 4.0], 3));
        assert!(!intercept_is_constant(&[1.0, 0.0, 1.0], 3));
        assert!(!intercept_is_constant(&[], 0));
    }

    #[test]
    fn a_cancelled_token_returns_the_cancellation_case() {
        let (design, _, t, y) = noiseless_linear(52);
        let ids: Vec<u16> = (0_u16..52).map(|i| i % 4).collect();
        let ctx = ExecutionContext::for_tests(5);
        ctx.cancellation.cancel();
        let outcome = run(&design, 2, &y, &t, 4, &ids, &ctx);
        assert!(matches!(outcome, Err(AipwCrossFitError::Cancelled)), "{outcome:?}");
    }

    #[test]
    fn a_fold_whose_training_rows_miss_an_arm_is_a_shape_error() {
        let n = 12;
        // Fold 1 trains on rows 0..6, which are all treated, so its control arm is empty.
        let ids: Vec<u16> = (0..n).map(|i| u16::from(i >= 6)).collect();
        let t = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y: Vec<f64> = (0..n).map(|i| 0.5 * i as f64 + (i % 3) as f64).collect();
        let ones = vec![1.0; n];
        let ctx = ExecutionContext::for_tests(1);
        let outcome = run(&ones, 1, &y, &t, 2, &ids, &ctx);
        assert!(
            matches!(
                outcome,
                Err(AipwCrossFitError::Learner(LearnError::Shape {
                    message: "empty DML/AIPW train, valid, or treatment arm in a fold"
                }))
            ),
            "{outcome:?}"
        );
    }

    #[test]
    fn the_same_inputs_give_bit_identical_vectors() {
        let n = 52;
        let ids: Vec<u16> = (0..n).map(|i| (i % 4) as u16).collect();
        let (design, _, t, y) = noiseless_linear(n);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let first = run(&design, 2, &y, &t, 4, &ids, &ExecutionContext::for_tests(9)).unwrap();
        let second = run(&design, 2, &y, &t, 4, &ids, &ExecutionContext::for_tests(9)).unwrap();
        assert_eq!(bits(&first.0), bits(&second.0));
        assert_eq!(bits(&first.1), bits(&second.1));
        assert_eq!(bits(&first.2), bits(&second.2));
        assert_eq!(bits(&first.3.predictions), bits(&second.3.predictions));
    }
}

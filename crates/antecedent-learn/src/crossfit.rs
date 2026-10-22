//! Generic out-of-fold cross-fitting over [`DesignView`] row indices.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::device_reservation::{
    CpuAdmissionError, FoldAdmission, LearnerAdmission, admit_fold, nested_thread_request,
    place_physical_predictions,
};
use antecedent_core::{CausalRng, ExecutionContext, StreamDomain};
use antecedent_kernels::{ViewError, shuffled_fold_assignment};

use crate::design::{DesignView, RowSelection, TargetView};
use crate::error::LearnError;
use crate::learner::{LearnerFactory, LearnerProvenance, PredictionTask};
use crate::transform::TransformerFactory;

/// Out-of-fold predictions for one nuisance.
#[derive(Clone, Debug)]
pub struct CrossFittedPrediction {
    /// Predictions aligned with the design's **physical** rows.
    pub predictions: Vec<f64>,
    /// Fold id per physical row.
    pub fold_assignment: Vec<u16>,
    /// One provenance record per fold.
    pub model_provenance: Vec<LearnerProvenance>,
    /// Held-out loss diagnostics.
    pub validation: NuisanceDiagnostics,
}

/// Held-out nuisance diagnostics. Unused fields stay `None`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NuisanceDiagnostics {
    /// Task the diagnostics were computed for.
    pub task: Option<PredictionTask>,
    /// RMSE for regression.
    pub rmse: Option<f64>,
    /// Out-of-fold R² for regression.
    pub r2: Option<f64>,
    /// Mean log loss for binary probability.
    pub logloss: Option<f64>,
    /// Public spec name of the selected learner (Auto).
    pub winner: Option<&'static str>,
    /// OOF loss of the runner-up (Auto).
    pub challenger_loss: Option<f64>,
}

/// Assign each physical row to one of `folds` folds through a seeded shuffle.
///
/// Fold membership is *not* a function of row position: an `i % folds` rule makes any
/// periodic or sorted row order (alternating treated/control exports, weekly data with
/// seven folds) put whole classes into single folds, so a fold's training rows can miss
/// a class entirely. With `strata`, rows of each stratum are dealt round-robin across the
/// folds, so every training fold keeps each class that has at least `folds` members.
///
/// # Errors
///
/// Fewer than two folds, fewer rows than folds, or `strata` of the wrong length.
#[allow(
    clippy::cast_possible_truncation,
    reason = "folds is checked to be at most 2^16 above, so every fold index fits u16"
)]
pub fn assign_folds(
    n_rows: usize,
    folds: usize,
    rng: &mut CausalRng,
    strata: Option<&[u32]>,
) -> Result<Vec<u16>, LearnError> {
    if folds < 2 {
        return Err(LearnError::Shape { message: "cross-fitting requires at least two folds" });
    }
    if folds > usize::from(u16::MAX) + 1 || u64::try_from(n_rows).is_err() {
        return Err(LearnError::Shape {
            message: "cross-fitting exceeds row or fold index capacity",
        });
    }
    if n_rows < folds {
        return Err(LearnError::Shape { message: "cross-fitting folds cannot exceed rows" });
    }
    if strata.is_some_and(|s| s.len() != n_rows) {
        return Err(LearnError::Shape { message: "fold strata length != rows" });
    }
    let assignment =
        shuffled_fold_assignment(rng, n_rows, folds, strata).map_err(|error| match error {
            ViewError::InvalidShape { message } => LearnError::Shape { message },
            other => LearnError::View(other.to_string()),
        })?;
    Ok(assignment.into_iter().map(|f| f as u16).collect())
}

/// Class labels to stratify folds on: the 0/1 target of a probability task.
#[allow(clippy::float_cmp, reason = "labels are coded exactly 0/1")]
fn binary_strata(task: PredictionTask, y: &[f64]) -> Option<Vec<u32>> {
    (task == PredictionTask::BinaryProbability && y.iter().all(|v| *v == 0.0 || *v == 1.0))
        .then(|| y.iter().map(|&v| u32::from(v == 1.0)).collect())
}

/// Number of folds in a plan that uses every id in `0..folds` at least once.
fn complete_fold_count(assignment: &[u16], message: &'static str) -> Result<usize, LearnError> {
    let folds = usize::from(*assignment.iter().max().unwrap_or(&0)) + 1;
    if folds < 2 || (0..folds).any(|f| !assignment.iter().any(|v| usize::from(*v) == f)) {
        return Err(LearnError::Shape { message });
    }
    Ok(folds)
}

/// Accounted width of one `f64`.
const F64_BYTES: u64 = 8;

/// Scatter each fold's held-out predictions back onto physical rows.
///
/// Fold completion order does not change the vector: each value is written at
/// its physical row, and a row that is missing or repeated is refused.
fn scatter_fold_predictions(
    n: usize,
    parts: Vec<FoldFit>,
) -> Result<(Vec<f64>, Vec<LearnerProvenance>), LearnError> {
    let mut pairs = Vec::with_capacity(parts.len());
    for part in &parts {
        pairs.push((part.valid_phys.as_slice(), part.preds.as_slice()));
    }
    let predictions = place_physical_predictions(n, &pairs).map_err(|_| LearnError::Shape {
        message: "held-out predictions are not one value per physical row",
    })?;
    drop(pairs);
    let mut model_provenance = Vec::with_capacity(parts.len());
    for part in parts {
        model_provenance.push(part.provenance);
    }
    Ok((predictions, model_provenance))
}

fn accounted_bytes(rows: usize) -> Result<u64, LearnError> {
    let rows = u64::try_from(rows)
        .map_err(|_| LearnError::Shape { message: "cross-fit fold byte account overflowed" })?;
    rows.checked_mul(F64_BYTES)
        .ok_or(LearnError::Shape { message: "cross-fit fold byte account overflowed" })
}

fn admission_to_learn(error: CpuAdmissionError) -> LearnError {
    match error {
        CpuAdmissionError::Cancelled => LearnError::Cancelled,
        CpuAdmissionError::FoldExceedsLease { .. } => {
            LearnError::Unsupported { message: "cross-fit fold exceeds the shared admission lease" }
        }
        CpuAdmissionError::LearnerExceedsLease { .. } => {
            LearnError::Unsupported { message: "cross-fit learner exceeds the fold lease" }
        }
        CpuAdmissionError::Scratch { .. } => {
            LearnError::Unsupported { message: "cross-fit scratch is full" }
        }
        _ => LearnError::Unsupported { message: "cross-fit admission was refused" },
    }
}

fn lease_fold(
    ctx: &ExecutionContext,
    rows: usize,
) -> Result<(FoldAdmission, LearnerAdmission), LearnError> {
    let accounted = accounted_bytes(rows)?;
    let threads = nested_thread_request(ctx.admission().available().threads());
    let fold = admit_fold(ctx, threads, accounted, F64_BYTES).map_err(admission_to_learn)?;
    let learner_threads = nested_thread_request(fold.held_threads());
    let learner =
        fold.admit_learner(learner_threads, accounted, F64_BYTES).map_err(admission_to_learn)?;
    Ok((fold, learner))
}

fn stop_if_cancelled(
    learner: &mut LearnerAdmission,
    fold: &mut FoldAdmission,
    ctx: &ExecutionContext,
) -> Result<(), LearnError> {
    let learner_stopped = learner.observe_cancel(&ctx.cancellation);
    let fold_stopped = fold.observe_cancel(&ctx.cancellation);
    if learner_stopped || fold_stopped {
        return Err(LearnError::Cancelled);
    }
    Ok(())
}

/// Cross-fit `factory` on `x` / `y`. `y` is physical-aligned.
///
/// Train/valid sets are row-index views over `x`. Fold models are discarded.
///
/// # Errors
///
/// Shape, empty train/valid fold, transformer, or learner failure.
pub(crate) fn cross_fit(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
    folds: usize,
    ctx: &ExecutionContext,
    transformer: Option<&dyn TransformerFactory>,
) -> Result<CrossFittedPrediction, LearnError> {
    if x.row_selection().is_some() {
        return Err(LearnError::Unsupported {
            message: "cross_fit expects a physical design without a row selection",
        });
    }
    let strata = binary_strata(factory.task(), y.values());
    let mut fold_rng = ctx.rng.stream_for(StreamDomain::Learner, crate::streams::FOLD_SHUFFLE);
    let fold_assignment =
        assign_folds(x.physical_nrows(), folds, &mut fold_rng, strata.as_deref())?;
    cross_fit_with_folds(factory, x, y, fold_assignment, ctx, transformer)
}

/// [`cross_fit`] with a caller-supplied fold plan (one id in `0..folds` per physical row).
///
/// Every fold id in `0..max+1` must occur, so each fold has a validation set and a
/// non-empty training complement.
///
/// # Errors
///
/// Shape mismatch, fewer than two folds, an empty fold, a fold that exceeds the
/// shared CPU and RAM lease, cancellation, or learner failure.
#[allow(
    clippy::cast_possible_truncation,
    reason = "fold ids are u16 and at most max + 1 <= 2^16 of them exist, so the fold index fits u16"
)]
pub fn cross_fit_with_folds(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
    fold_assignment: Vec<u16>,
    ctx: &ExecutionContext,
    transformer: Option<&dyn TransformerFactory>,
) -> Result<CrossFittedPrediction, LearnError> {
    cross_fit_with_folds_weighted(factory, x, y, None, fold_assignment, ctx, transformer)
}

/// [`cross_fit_with_folds`] fitting every fold model with nonnegative prior (case) weights,
/// one per physical row. Predictions are out-of-fold as before; the diagnostics are the
/// unweighted ones. A learner that does not accept sample weights refuses at its fit.
///
/// # Errors
///
/// As [`cross_fit_with_folds`], or a weights length different from the row count.
#[allow(
    clippy::cast_possible_truncation,
    reason = "fold ids are u16 and at most max + 1 <= 2^16 of them exist, so the fold index fits u16"
)]
pub fn cross_fit_with_folds_weighted(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
    weights: Option<&[f64]>,
    fold_assignment: Vec<u16>,
    ctx: &ExecutionContext,
    transformer: Option<&dyn TransformerFactory>,
) -> Result<CrossFittedPrediction, LearnError> {
    if x.row_selection().is_some() {
        return Err(LearnError::Unsupported {
            message: "cross_fit expects a physical design without a row selection",
        });
    }
    let n = x.physical_nrows();
    if y.len() != n {
        return Err(LearnError::Shape { message: "target length != physical rows" });
    }
    if weights.is_some_and(|w| w.len() != n) {
        return Err(LearnError::Shape { message: "weights length != physical rows" });
    }
    if fold_assignment.len() != n || n == 0 || u64::try_from(n).is_err() {
        return Err(LearnError::Shape { message: "fold plan length != physical rows" });
    }
    let folds =
        complete_fold_count(&fold_assignment, "cross-fitting needs at least two non-empty folds")?;
    let fold_results = ctx.map_indexed(folds, |fold, inner| {
        fit_one_fold(factory, x, y, weights, (&fold_assignment, fold as u16), inner, transformer)
    })?;

    let (predictions, model_provenance) = scatter_fold_predictions(n, fold_results)?;
    let validation = diagnose(factory.task(), y.values(), &predictions);
    Ok(CrossFittedPrediction { predictions, fold_assignment, model_provenance, validation })
}

struct FoldFit {
    valid_phys: Vec<u64>,
    preds: Vec<f64>,
    provenance: LearnerProvenance,
}

fn fit_one_fold(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
    weights: Option<&[f64]>,
    (folds, fold): (&[u16], u16),
    ctx: &ExecutionContext,
    transformer: Option<&dyn TransformerFactory>,
) -> Result<FoldFit, LearnError> {
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (i, &f) in folds.iter().enumerate() {
        let idx = u64::try_from(i).expect("row id");
        if f == fold {
            valid.push(idx);
        } else {
            train.push(idx);
        }
    }
    if train.is_empty() || valid.is_empty() {
        return Err(LearnError::Shape { message: "empty cross-fit train or valid fold" });
    }
    let rows = train
        .len()
        .checked_add(valid.len())
        .ok_or(LearnError::Shape { message: "cross-fit fold byte account overflowed" })?;
    let (mut fold_hold, mut learner_hold) = lease_fold(ctx, rows)?;
    stop_if_cancelled(&mut learner_hold, &mut fold_hold, ctx)?;
    let train_view = x.with_rows(RowSelection::new(&train))?;
    let valid_view = x.with_rows(RowSelection::new(&valid))?;
    let (fitted, preds) = if let Some(tf) = transformer {
        let xf = tf.fit(x, RowSelection::new(&train), ctx)?;
        let (train_buf, tr, tc) = xf.transform(train_view, ctx)?;
        let (valid_buf, vr, vc) = xf.transform(valid_view, ctx)?;
        let train_x = DesignView::from_column_major(&train_buf, tr, tc)?;
        let valid_x = DesignView::from_column_major(&valid_buf, vr, vc)?;
        let train_y = gather_logical_target(y, &train)?;
        let train_w =
            weights.map(|w| gather_logical_target(TargetView::new(w), &train)).transpose()?;
        let model = factory.fit(train_x, TargetView::new(&train_y), train_w.as_deref(), ctx)?;
        let mut out = vec![0.0; vr];
        model.predict(valid_x, &mut out, ctx)?;
        (model, out)
    } else {
        let model = factory.fit(train_view, y, weights, ctx)?;
        let mut out = vec![0.0; valid_view.nrows()];
        model.predict(valid_view, &mut out, ctx)?;
        (model, out)
    };
    stop_if_cancelled(&mut learner_hold, &mut fold_hold, ctx)?;
    Ok(FoldFit { valid_phys: valid, preds, provenance: fitted.provenance() })
}

fn gather_logical_target(y: TargetView<'_>, rows: &[u64]) -> Result<Vec<f64>, LearnError> {
    let mut out = Vec::with_capacity(rows.len());
    for &i in rows {
        let idx = usize::try_from(i).expect("row id");
        if idx >= y.len() {
            return Err(LearnError::Shape { message: "fold index out of target" });
        }
        out.push(y.values()[idx]);
    }
    Ok(out)
}

/// Compute OOF diagnostics from physical-aligned predictions.
#[must_use]
pub fn diagnose(task: PredictionTask, y: &[f64], pred: &[f64]) -> NuisanceDiagnostics {
    let n = y.len().min(pred.len());
    if n == 0 {
        return NuisanceDiagnostics { task: Some(task), ..NuisanceDiagnostics::default() };
    }
    match task {
        PredictionTask::Regression => {
            let mut sse = 0.0;
            let mut mean = 0.0;
            for i in 0..n {
                let e = y[i] - pred[i];
                sse += e * e;
                mean += y[i];
            }
            mean /= n as f64;
            let mut sst = 0.0;
            for yi in y.iter().take(n) {
                let d = yi - mean;
                sst += d * d;
            }
            let rmse = (sse / n as f64).sqrt();
            let r2 = if sst > 0.0 { Some(1.0 - sse / sst) } else { None };
            NuisanceDiagnostics {
                task: Some(task),
                rmse: Some(rmse),
                r2,
                ..NuisanceDiagnostics::default()
            }
        }
        PredictionTask::BinaryProbability => {
            let mut logloss = 0.0;
            for i in 0..n {
                let p = crate::learner::clip_probability(pred[i]);
                let yi = y[i].clamp(0.0, 1.0);
                logloss -= yi * p.ln() + (1.0 - yi) * (1.0 - p).ln();
            }
            NuisanceDiagnostics {
                task: Some(task),
                logloss: Some(logloss / n as f64),
                ..NuisanceDiagnostics::default()
            }
        }
    }
}

/// Cross-fit a nuisance trained only on declared eligible rows, predicting all
/// held-out rows. Fold assignment is shared across nuisance roles.
/// # Errors
/// Misaligned inputs, invalid folds, empty training roles, a fold that exceeds
/// the shared CPU and RAM lease, cancellation, or provider failure.
#[allow(
    clippy::cast_possible_truncation,
    reason = "n is checked against u32::MAX above, so every row index fits u32"
)]
pub fn cross_fit_selected(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
    fold_assignment: &[u16],
    eligible: &[bool],
    ctx: &ExecutionContext,
) -> Result<CrossFittedPrediction, LearnError> {
    let n = x.physical_nrows();
    if x.row_selection().is_some()
        || n == 0
        || u64::try_from(n).is_err()
        || y.len() != n
        || fold_assignment.len() != n
        || eligible.len() != n
    {
        return Err(LearnError::Shape { message: "misaligned selected cross-fit inputs" });
    }
    let folds = complete_fold_count(fold_assignment, "invalid selected cross-fit folds")?;
    let parts = ctx.map_indexed(folds, |fold, inner| {
        let train: Vec<u64> = (0..n)
            .filter(|i| eligible[*i] && usize::from(fold_assignment[*i]) != fold)
            .map(|i| u64::try_from(i).expect("row id"))
            .collect();
        let valid: Vec<u64> = (0..n)
            .filter(|i| usize::from(fold_assignment[*i]) == fold)
            .map(|i| u64::try_from(i).expect("row id"))
            .collect();
        if train.is_empty() {
            return Err(LearnError::Shape { message: "empty training role in cross-fit fold" });
        }
        let rows = train
            .len()
            .checked_add(valid.len())
            .ok_or(LearnError::Shape { message: "cross-fit fold byte account overflowed" })?;
        let (mut fold_hold, mut learner_hold) = lease_fold(inner, rows)?;
        stop_if_cancelled(&mut learner_hold, &mut fold_hold, inner)?;
        let fitted = factory.fit(x.with_rows(RowSelection::new(&train))?, y, None, inner)?;
        let mut preds = vec![0.0; valid.len()];
        fitted.predict(x.with_rows(RowSelection::new(&valid))?, &mut preds, inner)?;
        stop_if_cancelled(&mut learner_hold, &mut fold_hold, inner)?;
        Ok(FoldFit { valid_phys: valid, preds, provenance: fitted.provenance() })
    })?;
    let (predictions, model_provenance) = scatter_fold_predictions(n, parts)?;
    let observed: Vec<_> =
        y.values().iter().zip(eligible).filter_map(|(v, use_row)| use_row.then_some(*v)).collect();
    let predicted: Vec<_> =
        predictions.iter().zip(eligible).filter_map(|(v, use_row)| use_row.then_some(*v)).collect();
    Ok(CrossFittedPrediction {
        validation: diagnose(factory.task(), &observed, &predicted),
        predictions,
        model_provenance,
        fold_assignment: fold_assignment.to_vec(),
    })
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "this unit-test module compares floats that are copied, clamped or hand-set without rounding, so exact equality is intended"
)]
mod tests {
    use antecedent_core::execution::AdmissionPool;

    use super::*;
    use crate::linear::LinearLearner;
    use crate::transform::Identity;

    #[test]
    fn fold_ids_cannot_wrap() {
        let mut rng = CausalRng::from_seed(1);
        assert!(assign_folds(65_537, 65_537, &mut rng, None).is_err());
        let folds = assign_folds(65_536, 65_536, &mut rng, None).unwrap();
        assert_eq!(folds.iter().copied().max(), Some(u16::MAX));
    }

    /// Alternating treated/control rows with two folds: under `i % 2` fold 0 is all
    /// treated and its propensity model trains on fold 1 = all controls (a single-class
    /// target). The stratified shuffle puts both classes in every fold.
    #[test]
    fn alternating_binary_rows_keep_both_classes_in_every_fold() {
        let n = 60usize;
        let y: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let strata = binary_strata(PredictionTask::BinaryProbability, &y).unwrap();
        let mut rng = CausalRng::from_seed(3);
        let folds = assign_folds(n, 2, &mut rng, Some(&strata)).unwrap();
        for fold in 0..2u16 {
            let treated = (0..n).filter(|&i| folds[i] == fold && y[i] == 1.0).count();
            let control = (0..n).filter(|&i| folds[i] == fold && y[i] == 0.0).count();
            assert_eq!((treated, control), (15, 15), "fold {fold}");
        }
        // A regression task is not stratified.
        assert!(binary_strata(PredictionTask::Regression, &y).is_none());
    }

    fn line(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 3.0 + 4.0 * (i as f64);
        }
        (x, y)
    }

    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn oof_line_recovers_without_copying_x() {
        let n = 12usize;
        let (x, y) = line(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let oof = cross_fit(&LinearLearner, view, TargetView::new(&y), 4, &ctx, None).unwrap();
        assert_eq!(oof.predictions.len(), n);
        assert_eq!(oof.fold_assignment.len(), n);
        for i in 0..n {
            assert!((oof.predictions[i] - y[i]).abs() < 1e-8);
        }
        assert_eq!(view.dense().values().as_ptr(), x.as_ptr());
        assert!(oof.validation.rmse.unwrap() < 1e-8);
    }

    #[test]
    fn identity_transformer_does_not_change_oof() {
        let n = 10usize;
        let (x, y) = line(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let a = cross_fit(&LinearLearner, view, TargetView::new(&y), 5, &ctx, None).unwrap();
        let b =
            cross_fit(&LinearLearner, view, TargetView::new(&y), 5, &ctx, Some(&Identity)).unwrap();
        for i in 0..n {
            assert!((a.predictions[i] - b.predictions[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn fold_exceeding_the_lease_is_refused() {
        let n = 12usize;
        let (x, y) = line(n);
        let pool = AdmissionPool::fixed(1, 8);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let error =
            cross_fit(&LinearLearner, view, TargetView::new(&y), 4, &ctx, None).unwrap_err();
        assert!(matches!(error, LearnError::Unsupported { message } if message.contains("lease")));
        assert_eq!(pool.available().bytes(), 8);
        assert_eq!(pool.available().threads(), 1);
    }

    #[test]
    fn cancelled_cross_fit_releases_the_cpu_lease() {
        let n = 12usize;
        let (x, y) = line(n);
        let pool = AdmissionPool::fixed(1, 1_000);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        ctx.cancellation.cancel();
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let error =
            cross_fit(&LinearLearner, view, TargetView::new(&y), 4, &ctx, None).unwrap_err();
        assert!(matches!(error, LearnError::Cancelled));
        assert_eq!(pool.available().bytes(), 1_000);
        assert_eq!(pool.available().threads(), 1);
    }
}

//! Generic out-of-fold cross-fitting over [`DesignView`] row indices.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]

use antecedent_core::ExecutionContext;

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

/// Assign `i % folds` on physical rows.
///
/// # Errors
///
/// Fewer than two folds, or fewer rows than folds.
pub fn assign_folds(n_rows: usize, folds: usize) -> Result<Vec<u16>, LearnError> {
    if folds < 2 {
        return Err(LearnError::Shape { message: "cross-fitting requires at least two folds" });
    }
    if folds > usize::from(u16::MAX) + 1 || n_rows > u32::MAX as usize {
        return Err(LearnError::Shape {
            message: "cross-fitting exceeds row or fold index capacity",
        });
    }
    if n_rows < folds {
        return Err(LearnError::Shape { message: "cross-fitting folds cannot exceed rows" });
    }
    Ok((0..n_rows).map(|i| (i % folds) as u16).collect())
}

/// Cross-fit `factory` on `x` / `y`. `y` is physical-aligned.
///
/// Train/valid sets are row-index views over `x`. Fold models are discarded.
///
/// # Errors
///
/// Shape, empty train/valid fold, transformer, or learner failure.
pub fn cross_fit(
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
    let fold_assignment = assign_folds(x.physical_nrows(), folds)?;
    cross_fit_with_folds(factory, x, y, fold_assignment, ctx, transformer)
}

/// [`cross_fit`] with a caller-supplied fold plan (one id in `0..folds` per physical row).
///
/// Every fold id in `0..max+1` must occur, so each fold has a validation set and a
/// non-empty training complement.
///
/// # Errors
///
/// Shape mismatch, fewer than two folds, an empty fold, or learner failure.
pub fn cross_fit_with_folds(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
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
    if fold_assignment.len() != n || n == 0 {
        return Err(LearnError::Shape { message: "fold plan length != physical rows" });
    }
    let folds = usize::from(*fold_assignment.iter().max().unwrap_or(&0)) + 1;
    if folds < 2 || (0..folds).any(|f| !fold_assignment.iter().any(|v| usize::from(*v) == f)) {
        return Err(LearnError::Shape {
            message: "cross-fitting needs at least two non-empty folds",
        });
    }
    let fold_results = ctx.map_indexed(folds, |fold, inner| {
        fit_one_fold(factory, x, y, &fold_assignment, fold as u16, inner, transformer)
    })?;

    let mut predictions = vec![0.0; n];
    let mut model_provenance = Vec::with_capacity(folds);
    for result in fold_results {
        model_provenance.push(result.provenance);
        for (phys, pred) in result.valid_phys.into_iter().zip(result.preds) {
            predictions[phys as usize] = pred;
        }
    }
    let validation = diagnose(factory.task(), y.values(), &predictions);
    Ok(CrossFittedPrediction { predictions, fold_assignment, model_provenance, validation })
}

struct FoldFit {
    valid_phys: Vec<u32>,
    preds: Vec<f64>,
    provenance: LearnerProvenance,
}

fn fit_one_fold(
    factory: &dyn LearnerFactory,
    x: DesignView<'_>,
    y: TargetView<'_>,
    folds: &[u16],
    fold: u16,
    ctx: &ExecutionContext,
    transformer: Option<&dyn TransformerFactory>,
) -> Result<FoldFit, LearnError> {
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (i, &f) in folds.iter().enumerate() {
        let idx = i as u32;
        if f == fold {
            valid.push(idx);
        } else {
            train.push(idx);
        }
    }
    if train.is_empty() || valid.is_empty() {
        return Err(LearnError::Shape { message: "empty cross-fit train or valid fold" });
    }
    let train_view = x.with_rows(RowSelection::new(&train))?;
    let valid_view = x.with_rows(RowSelection::new(&valid))?;
    let (fitted, preds) = if let Some(tf) = transformer {
        let xf = tf.fit(x, RowSelection::new(&train), ctx)?;
        let (train_buf, tr, tc) = xf.transform(train_view, ctx)?;
        let (valid_buf, vr, vc) = xf.transform(valid_view, ctx)?;
        let train_x = DesignView::from_column_major(&train_buf, tr, tc)?;
        let valid_x = DesignView::from_column_major(&valid_buf, vr, vc)?;
        let train_y = gather_logical_target(y, &train)?;
        let model = factory.fit(train_x, TargetView::new(&train_y), None, ctx)?;
        let mut out = vec![0.0; vr];
        model.predict(valid_x, &mut out, ctx)?;
        (model, out)
    } else {
        let model = factory.fit(train_view, y, None, ctx)?;
        let mut out = vec![0.0; valid_view.nrows()];
        model.predict(valid_view, &mut out, ctx)?;
        (model, out)
    };
    Ok(FoldFit { valid_phys: valid, preds, provenance: fitted.provenance() })
}

fn gather_logical_target(y: TargetView<'_>, rows: &[u32]) -> Result<Vec<f64>, LearnError> {
    let mut out = Vec::with_capacity(rows.len());
    for &i in rows {
        let idx = i as usize;
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
                let p = pred[i].clamp(1e-9, 1.0 - 1e-9);
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

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::linear::LinearLearner;
    use crate::transform::Identity;

    #[test]
    fn fold_ids_cannot_wrap() {
        assert!(assign_folds(65_537, 65_537).is_err());
        assert_eq!(assign_folds(65_536, 65_536).unwrap()[65_535], u16::MAX);
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
        match view.storage() {
            crate::DesignStorage::Dense(d) => assert_eq!(d.values().as_ptr(), x.as_ptr()),
            crate::DesignStorage::SparseCsr(_) => panic!("dense"),
        }
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
}

/// Cross-fit a nuisance trained only on declared eligible rows, predicting all
/// held-out rows. Fold assignment is shared across nuisance roles.
/// # Errors
/// Misaligned inputs, invalid folds, empty training roles, or provider failure.
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
        || n > u32::MAX as usize
        || y.len() != n
        || fold_assignment.len() != n
        || eligible.len() != n
    {
        return Err(LearnError::Shape { message: "misaligned selected cross-fit inputs" });
    }
    let folds = usize::from(*fold_assignment.iter().max().unwrap()) + 1;
    if folds < 2 || (0..folds).any(|f| !fold_assignment.iter().any(|v| usize::from(*v) == f)) {
        return Err(LearnError::Shape { message: "invalid selected cross-fit folds" });
    }
    let parts = ctx.map_indexed(folds, |fold, inner| {
        let train: Vec<u32> = (0..n)
            .filter(|i| eligible[*i] && usize::from(fold_assignment[*i]) != fold)
            .map(|i| i as u32)
            .collect();
        let valid: Vec<u32> =
            (0..n).filter(|i| usize::from(fold_assignment[*i]) == fold).map(|i| i as u32).collect();
        if train.is_empty() {
            return Err(LearnError::Shape { message: "empty training role in cross-fit fold" });
        }
        let fitted = factory.fit(x.with_rows(RowSelection::new(&train))?, y, None, inner)?;
        let mut preds = vec![0.0; valid.len()];
        fitted.predict(x.with_rows(RowSelection::new(&valid))?, &mut preds, inner)?;
        Ok(FoldFit { valid_phys: valid, preds, provenance: fitted.provenance() })
    })?;
    let mut predictions = vec![0.0; n];
    let mut model_provenance = Vec::new();
    for part in parts {
        model_provenance.push(part.provenance);
        for (row, value) in part.valid_phys.into_iter().zip(part.preds) {
            predictions[row as usize] = value;
        }
    }
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

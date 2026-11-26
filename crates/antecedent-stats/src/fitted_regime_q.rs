//! Two-decision fixed-schedule g-formula with an independently specified cell-mean Q model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use crate::{equal_tail_interval_sorted, QuantileRule, StatsError};
use antecedent_core::{ExecutionContext, StreamDomain};

const SUBJECT_BOOTSTRAP_STREAM: u64 = 0x5126_7740_0000_0000;

/// Binary pre-treatment stratum, two realized actions and endpoint for each independent subject.
pub struct FittedRegimeRows<'a> {
    /// A complete binary baseline marker recorded before the first action.
    pub baseline: &'a [f64],
    /// Subject-major realized actions.
    pub actions: &'a [bool],
    /// Observed terminal reward.
    pub outcomes: &'a [f64],
    /// Whole-subject training and evaluation ownership.
    pub folds: &'a [u32],
}

/// Value and whole-procedure subject-bootstrap interval of the fitted Q regime evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedRegimeQSummary {
    /// Mean of regime predictions over held-out evaluation subjects.
    pub value: f64,
    /// Bootstrap standard deviation of the whole refit procedure.
    pub standard_error: f64,
    /// Pointwise 95% percentile interval.
    pub interval: [f64; 2],
    /// Independent target subjects across the held-out folds.
    pub evaluation_subjects: usize,
    /// Complete subject-resample replicates.
    pub replicates_ok: usize,
}

#[derive(Clone, Copy, Default)]
struct Cell {
    subjects: f64,
    matched: f64,
    rewards: f64,
}

impl Cell {
    fn add(self, another: Self) -> Self {
        Self {
            subjects: self.subjects + another.subjects,
            matched: self.matched + another.matched,
            rewards: self.rewards + another.rewards,
        }
    }

    fn remove(self, another: Self) -> Self {
        Self {
            subjects: self.subjects - another.subjects,
            matched: self.matched - another.matched,
            rewards: self.rewards - another.rewards,
        }
    }
}

fn prediction_sum(evaluation: [Cell; 2], training: [Cell; 2]) -> Option<(f64, f64)> {
    let mut prediction_sum = 0.0;
    let mut subjects = 0.0;
    for (evaluated, trained) in evaluation.into_iter().zip(training) {
        if evaluated.subjects > 0.0 {
            if trained.matched < 1.0 {
                return None;
            }
            prediction_sum += evaluated.subjects * trained.rewards / trained.matched;
            subjects += evaluated.subjects;
        }
    }
    Some((prediction_sum, subjects))
}

fn estimate(
    rows: &FittedRegimeRows<'_>,
    folds: &[Vec<usize>],
    weights: &[f64],
    actions: [bool; 2],
    training_fold: Option<usize>,
    cells: &mut [[Cell; 2]],
) -> Option<f64> {
    cells.fill([Cell::default(); 2]);
    let mut pooled = [Cell::default(); 2];
    for (fold, members) in folds.iter().enumerate() {
        for &i in members {
            let stratum = usize::from(rows.baseline[i] > 0.0);
            let weight = weights[i];
            let follows = rows.actions[2 * i..2 * i + 2] == actions;
            let cell = &mut cells[fold][stratum];
            cell.subjects += weight;
            if follows {
                cell.matched += weight;
                cell.rewards += weight * rows.outcomes[i];
            }
        }
        for (total, own) in pooled.iter_mut().zip(cells[fold]) {
            *total = total.add(own);
        }
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (fold, &held_out) in cells.iter().enumerate() {
        if training_fold == Some(fold) {
            continue;
        }
        let training = if let Some(training_fold) = training_fold {
            cells[training_fold]
        } else {
            [pooled[0].remove(held_out[0]), pooled[1].remove(held_out[1])]
        };
        let (sum, subjects) = prediction_sum(held_out, training)?;
        numerator += sum;
        denominator += subjects;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

/// Refit the Q regression and evaluate the fixed schedule after each whole-subject draw.
///
/// Bootstrap strata preserve the declared fold owners. An external training fold is the only
/// source of Q fits when supplied, with the other fold providing the target baseline distribution.
/// A same-sample analysis instead refits outside every held-out fold, then pools its evaluations.
/// With binary pre-treatment stratum and no evolving covariate, backward g-computation under the
/// static schedule reduces to the terminal conditional cell mean, averaged over target strata.
#[allow(clippy::float_cmp, reason = "baseline values are categorical 0 and 1")]
pub fn estimate_fitted_regime_q(
    rows: &FittedRegimeRows<'_>,
    actions: [bool; 2],
    external_training_fold: Option<u32>,
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<FittedRegimeQSummary, StatsError> {
    let subjects = rows.outcomes.len();
    if subjects < 2
        || rows.baseline.len() != subjects
        || rows.folds.len() != subjects
        || rows.actions.len() != subjects.saturating_mul(2)
        || rows.baseline.iter().any(|&x| x != 0.0 && x != 1.0)
        || rows.outcomes.iter().any(|y| !y.is_finite())
    {
        return Err(StatsError::Shape {
            message: "fitted Q needs aligned binary-baseline histories",
        });
    }
    if replicates < 99 {
        return Err(StatsError::Shape {
            message: "fitted Q needs at least 99 whole-subject bootstrap replicates",
        });
    }
    let quoted_subjects = u64::try_from(subjects).map_err(|_| fitted_quote_overflow(ctx))?;
    let quoted_replicates = u64::try_from(replicates).map_err(|_| fitted_quote_overflow(ctx))?;
    let bytes = quoted_subjects
        .checked_mul(128)
        .and_then(|body| quoted_replicates.checked_mul(8).and_then(|draws| body.checked_add(draws)))
        .ok_or_else(|| fitted_quote_overflow(ctx))?;
    let _lease = ctx.acquire_serial_numerical_admission(bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => StatsError::Cancelled,
        other => StatsError::Admission(other),
    })?;
    let mut by_fold: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &fold) in rows.folds.iter().enumerate() {
        by_fold.entry(fold).or_default().push(i);
    }
    let training_fold =
        external_training_fold.and_then(|fold| by_fold.keys().position(|&id| id == fold));
    if by_fold.len() < 2
        || external_training_fold.is_some() && (by_fold.len() != 2 || training_fold.is_none())
    {
        return Err(StatsError::Shape {
            message: "fitted Q needs at least two exclusion folds or distinct external training and evaluation folds",
        });
    }
    let folds: Vec<_> = by_fold.into_values().collect();
    let evaluation_subjects = training_fold.map_or(subjects, |fold| subjects - folds[fold].len());
    let mut cells = vec![[Cell::default(); 2]; folds.len()];
    let mut weights = vec![1.0; subjects];
    let value = estimate(rows, &folds, &weights, actions, training_fold, &mut cells).ok_or_else(
        || StatsError::Unsupported {
            message:
                "the training split has no observed regime match for an evaluated baseline stratum",
        },
    )?;
    let mut fitted_values = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, SUBJECT_BOOTSTRAP_STREAM | replicate as u64);
        weights.fill(0.0);
        for members in &folds {
            for _ in members {
                weights[members[antecedent_kernels::unbiased_index(&mut rng, members.len())]] +=
                    1.0;
            }
        }
        if let Some(value) = estimate(rows, &folds, &weights, actions, training_fold, &mut cells) {
            fitted_values.push(value);
        }
    }
    if fitted_values.len() < 99 || (replicates - fitted_values.len()) * 20 > replicates {
        return Err(StatsError::Unsupported {
            message: "too many whole-subject fitted Q replicates lacked regime support",
        });
    }
    let standard_error = crate::sample_std(&fitted_values);
    fitted_values.sort_by(f64::total_cmp);
    let (lower, upper) =
        equal_tail_interval_sorted(&fitted_values, 0.95, QuantileRule::Interpolated);
    Ok(FittedRegimeQSummary {
        value,
        standard_error,
        interval: [lower, upper],
        evaluation_subjects,
        replicates_ok: fitted_values.len(),
    })
}

fn fitted_quote_overflow(ctx: &ExecutionContext) -> StatsError {
    StatsError::ResourceLimit {
        message: "fitted Q work quote overflowed",
        required_bytes: u64::MAX,
        limit_bytes: ctx.memory_limit_bytes().unwrap_or(u64::MAX),
    }
}

//! Paired policy scores for fixed two-decision schedules and honest schedule learning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_kernels::{standard_normal, unbiased_index};

use crate::{quantile_sorted, QuantileRule, StatsError, NORMAL_Q975};

const SIMULTANEOUS_STREAM: u64 = 0x5319_0000_0000_0000;
const POLICY_REFIT_STREAM: u64 = 0x5320_0000_0000_0000;

/// Prespecified schedule versus reference, with a common family critical value.
#[derive(Clone, Debug, PartialEq)]
pub struct LongitudinalScheduleContrast {
    /// Bits of the evaluated actions, in decision order.
    pub schedule: u8,
    /// Bits of the reference actions, in decision order.
    pub reference: u8,
    /// Evaluation mean of the paired contrast scores.
    pub difference: f64,
    /// Standard error of the paired scores.
    pub standard_error: f64,
    /// Pointwise entry of the simultaneous family interval.
    pub interval: [f64; 2],
}

/// Joint interval results for prespecified schedules or for all pairs used by policy learning.
#[derive(Clone, Debug, PartialEq)]
pub struct LongitudinalScheduleComparison {
    /// Evaluation regime values in schedule order, paired with their two-bit names.
    pub values: Vec<(u8, f64)>,
    /// Contrasts in declared order and the common maximum-t construction.
    pub contrasts: Vec<LongitudinalScheduleContrast>,
    /// Quantile of the maximum absolute centered multiplier statistic.
    pub critical_value: f64,
    /// Independent subjects whose paired scores enter the test.
    pub evaluation_subjects: usize,
}

/// Held-out learning result including the complete-class optimum confidence set.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedLongitudinalSchedule {
    /// Schedule selected by training subjects, with ties going to the smallest two-bit name.
    pub selected_schedule: u8,
    /// Held-out evaluation value of that schedule.
    pub selected_value: f64,
    /// Held-out independent-subject score standard error.
    pub selected_standard_error: f64,
    /// 95% independent evaluation-subject interval for the trained schedule's value.
    pub selected_interval: [f64; 2],
    /// Regimes compatible with being optimal after the simultaneous comparisons.
    pub optimum_set: Vec<u8>,
    /// All pairwise contrasts underlying the optimum confidence set.
    pub comparison: LongitudinalScheduleComparison,
    /// Count of training-refit selections by regime, in regime order.
    pub refit_selections: Vec<(u8, usize)>,
    /// Standard deviation of held-out values under whole-procedure bootstrap refits.
    pub refit_value_sd: f64,
}

fn shape(message: &'static str) -> StatsError {
    StatsError::Shape { message }
}

fn support(message: &'static str) -> StatsError {
    StatsError::Unsupported { message }
}

/// A schedule's packed actions, first action most significant.
#[must_use]
pub const fn longitudinal_schedule_code(first: bool, second: bool) -> u8 {
    (first as u8) * 2 + second as u8
}

fn schedules(mask: u8) -> Result<Vec<u8>, StatsError> {
    if mask > 0b1111 || mask.count_ones() < 2 {
        return Err(shape(
            "a longitudinal schedule family requires at least two distinct schedules",
        ));
    }
    Ok((0..4).filter(|&schedule| mask & (1 << schedule) != 0).collect())
}

fn prepare_scores(treatments: &[bool], outcomes: &[f64]) -> Result<Vec<(u8, f64)>, StatsError> {
    if treatments.len() != outcomes.len().saturating_mul(2) {
        return Err(shape("outcomes and two-decision histories must align by subject"));
    }
    treatments
        .chunks_exact(2)
        .zip(outcomes)
        .map(|(actions, &y)| {
            let score = 4.0 * y;
            if !score.is_finite() {
                return Err(shape(
                    "the endpoint must have a finite inverse-assignment-weighted score",
                ));
            }
            Ok((longitudinal_schedule_code(actions[0], actions[1]), score))
        })
        .collect()
}

fn ensure_matches(rows: &[(u8, f64)], family: &[u8]) -> Result<(), StatsError> {
    let mut counts = [0; 4];
    for &(schedule, _) in rows {
        counts[schedule as usize] += 1;
    }
    if rows.len() < 200 || family.iter().any(|&schedule| counts[schedule as usize] < 20) {
        return Err(support(
            "too few independent subjects or full matches for a declared schedule",
        ));
    }
    Ok(())
}

fn values(rows: &[(u8, f64)]) -> [f64; 4] {
    let mut totals = [0.0; 4];
    for &(schedule, score) in rows {
        totals[schedule as usize] += score;
    }
    totals.map(|total| total / rows.len() as f64)
}

fn measure_family(
    rows: &[(u8, f64)],
    family: &[u8],
    pairs: &[(u8, u8)],
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<LongitudinalScheduleComparison, StatsError> {
    ensure_matches(rows, family)?;
    let means = values(rows);
    let offsets: Vec<_> =
        pairs.iter().map(|&(a, b)| means[a as usize] - means[b as usize]).collect();
    let residuals: Vec<Vec<_>> = pairs
        .iter()
        .zip(&offsets)
        .map(|(&(a, b), &offset)| {
            rows.iter()
                .map(|&(schedule, score)| {
                    (f64::from(schedule == a) - f64::from(schedule == b)) * score - offset
                })
                .collect()
        })
        .collect();
    let norms: Vec<_> =
        residuals.iter().map(|column| column.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    if norms.iter().any(|norm| !norm.is_finite()) {
        return Err(support("paired schedule score variation exceeds finite arithmetic"));
    }
    let n = rows.len() as f64;
    let independent_scale = ((n - 1.0) * n).sqrt();
    let mut maxima = Vec::with_capacity(draws);
    let mut multipliers = vec![0.0; rows.len()];
    for replicate in 0..draws {
        if ctx.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, SIMULTANEOUS_STREAM | replicate as u64);
        for multiplier in &mut multipliers {
            *multiplier = standard_normal(&mut rng);
        }
        let maximum = residuals
            .iter()
            .zip(&norms)
            .filter(|&(_, &norm)| norm > 0.0)
            .map(|(column, &norm)| {
                (column
                    .iter()
                    .zip(&multipliers)
                    .map(|(value, multiplier)| value * multiplier)
                    .sum::<f64>()
                    / norm)
                    .abs()
            })
            .fold(0.0_f64, f64::max);
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    let critical_value = quantile_sorted(&maxima, 0.95, QuantileRule::ExchangeableRank);
    let contrasts = pairs
        .iter()
        .zip(offsets)
        .zip(norms)
        .map(|((&(schedule, reference), difference), norm)| {
            let standard_error = norm / independent_scale;
            let radius = critical_value * standard_error;
            LongitudinalScheduleContrast {
                schedule,
                reference,
                difference,
                standard_error,
                interval: [difference - radius, difference + radius],
            }
        })
        .collect();
    Ok(LongitudinalScheduleComparison {
        values: family.iter().map(|&schedule| (schedule, means[schedule as usize])).collect(),
        contrasts,
        critical_value,
        evaluation_subjects: rows.len(),
    })
}

fn check_work(
    ctx: &ExecutionContext,
    subjects: usize,
    draws: usize,
) -> Result<antecedent_core::execution::AdmissionLease, StatsError> {
    if draws < 199 {
        return Err(shape(
            "longitudinal schedule inference requires at least 199 multiplier draws",
        ));
    }
    let subjects = u64::try_from(subjects).map_err(|_| schedule_quote_overflow(ctx))?;
    let draws = u64::try_from(draws).map_err(|_| schedule_quote_overflow(ctx))?;
    let bytes = subjects
        .checked_mul(192)
        .and_then(|body| draws.checked_mul(16).and_then(|tail| body.checked_add(tail)))
        .ok_or_else(|| schedule_quote_overflow(ctx))?;
    ctx.acquire_serial_numerical_admission(bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => StatsError::Cancelled,
        other => StatsError::Admission(other),
    })
}

fn schedule_quote_overflow(ctx: &ExecutionContext) -> StatsError {
    StatsError::ResourceLimit {
        message: "longitudinal schedule work quote overflowed",
        required_bytes: u64::MAX,
        limit_bytes: ctx.memory_limit_bytes().unwrap_or(u64::MAX),
    }
}

/// Compare declared schedules in one paired, simultaneous analysis of all subjects.
pub fn compare_longitudinal_schedules(
    treatments: &[bool],
    outcomes: &[f64],
    family_mask: u8,
    reference: u8,
    contrast_mask: u8,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<LongitudinalScheduleComparison, StatsError> {
    let family = schedules(family_mask)?;
    if !family.contains(&reference)
        || contrast_mask == 0
        || contrast_mask & !family_mask != 0
        || contrast_mask & (1 << reference) != 0
    {
        return Err(shape(
            "the simultaneous contrast family must name distinct nonreference members",
        ));
    }
    let _lease = check_work(ctx, outcomes.len(), draws)?;
    let rows = prepare_scores(treatments, outcomes)?;
    let pairs: Vec<_> = family
        .iter()
        .copied()
        .filter(|&schedule| contrast_mask & (1 << schedule) != 0)
        .map(|schedule| (schedule, reference))
        .collect();
    measure_family(&rows, &family, &pairs, draws, ctx)
}

fn choose(family: &[u8], means: &[f64; 4]) -> u8 {
    let mut chosen = family[0];
    for &schedule in &family[1..] {
        if means[schedule as usize] > means[chosen as usize] {
            chosen = schedule;
        }
    }
    chosen
}

fn refit_values(
    training: &[(u8, f64)],
    evaluation: &[(u8, f64)],
    family: &[u8],
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<(Vec<(u8, usize)>, f64), StatsError> {
    let mut train_draw = Vec::with_capacity(training.len());
    let mut evaluation_draw = Vec::with_capacity(evaluation.len());
    let mut counts = [0; 4];
    let mut evaluated = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, POLICY_REFIT_STREAM | replicate as u64);
        train_draw.clear();
        evaluation_draw.clear();
        train_draw.extend(
            (0..training.len()).map(|_| training[unbiased_index(&mut rng, training.len())]),
        );
        evaluation_draw.extend(
            (0..evaluation.len()).map(|_| evaluation[unbiased_index(&mut rng, evaluation.len())]),
        );
        let selected = choose(family, &values(&train_draw));
        counts[selected as usize] += 1;
        evaluated.push(values(&evaluation_draw)[selected as usize]);
    }
    let selections = family.iter().map(|&schedule| (schedule, counts[schedule as usize])).collect();
    Ok((selections, crate::sample_std(&evaluated)))
}

/// Train on the declared subject fold and evaluate on the other fold.
///
/// The selected-value interval is conditional on the trained schedule. The optimum confidence
/// set uses simultaneous evaluation of every pair in the declared class, including tied policies.
/// Separate whole-procedure refits describe training variability, including changes in selection.
pub fn learn_longitudinal_schedule(
    treatments: &[bool],
    outcomes: &[f64],
    folds: &[u32],
    family_mask: u8,
    training_fold: u32,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<LearnedLongitudinalSchedule, StatsError> {
    let family = schedules(family_mask)?;
    if folds.len() != outcomes.len()
        || !folds.contains(&training_fold)
        || folds.iter().find(|&&fold| fold != training_fold).is_none_or(|evaluation| {
            folds.iter().any(|fold| *fold != training_fold && fold != evaluation)
        })
    {
        return Err(shape(
            "schedule learning requires distinct subject-owned training and evaluation folds",
        ));
    }
    let _lease = check_work(ctx, outcomes.len(), draws)?;
    let rows = prepare_scores(treatments, outcomes)?;
    let (training, evaluation): (Vec<_>, Vec<_>) =
        rows.into_iter().zip(folds).partition(|&(_, &fold)| fold == training_fold);
    let training: Vec<_> = training.into_iter().map(|(row, _)| row).collect();
    let evaluation: Vec<_> = evaluation.into_iter().map(|(row, _)| row).collect();
    ensure_matches(&training, &family)?;
    let selected_schedule = choose(&family, &values(&training));
    let pairs: Vec<_> = family
        .iter()
        .enumerate()
        .flat_map(|(i, &b)| family[i + 1..].iter().map(move |&a| (a, b)))
        .collect();
    let comparison = measure_family(&evaluation, &family, &pairs, draws, ctx)?;
    let optimum_set = family
        .iter()
        .copied()
        .filter(|&schedule| {
            !comparison.contrasts.iter().any(|contrast| {
                (contrast.schedule == schedule && contrast.interval[1] < 0.0)
                    || (contrast.reference == schedule && contrast.interval[0] > 0.0)
            })
        })
        .collect();
    let selected_scores: Vec<_> = evaluation
        .iter()
        .map(|&(schedule, score)| if schedule == selected_schedule { score } else { 0.0 })
        .collect();
    let (selected_value, variance) = crate::mean_and_sample_variance(&selected_scores)
        .ok_or_else(|| support("selected schedule evaluation has no finite variance"))?;
    let selected_standard_error = (variance / evaluation.len() as f64).sqrt();
    let radius = NORMAL_Q975 * selected_standard_error;
    let (refit_selections, refit_value_sd) =
        refit_values(&training, &evaluation, &family, draws, ctx)?;
    Ok(LearnedLongitudinalSchedule {
        selected_schedule,
        selected_value,
        selected_standard_error,
        selected_interval: [selected_value - radius, selected_value + radius],
        optimum_set,
        comparison,
        refit_selections,
        refit_value_sd,
    })
}

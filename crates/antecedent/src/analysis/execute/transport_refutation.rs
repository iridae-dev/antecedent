// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a trial-to-target transport estimate.
//!
//! The ATE-shaped contrast refuters do not apply to a transported contrast: the estimate is a
//! ratio of weighted means over a trial and a target sample, and its identifying inputs are
//! the supplied selection and treatment probabilities. Its `cheap` and `full` suites therefore
//! check those inputs against the rows they describe, and the estimate against its own
//! resampling:
//!
//! * `transport.selection_calibration` (cheap, full): the supplied selection probabilities say
//!   how many rows should have joined the trial; the observed trial count is tested against
//!   their sum (a Poisson-binomial z test, with batch-means variance on one dependent series).
//! * `transport.arm_calibration` (cheap, full): the supplied treatment probabilities say how
//!   many trial rows should have been treated; the observed count is tested against their sum.
//! * `transport.selection_overlap` and `transport.treatment_overlap` (cheap, full): the Kish
//!   effective sample size of the selection-odds weights and of the arm weights, as a share of
//!   the trial rows. A share under one tenth is a descriptive warning, not a calibrated test.
//! * `transport.bootstrap_range` (full): a whole-procedure bootstrap of the transported
//!   contrast (rows, or circular blocks on one dependent series); the published value must lie
//!   in the replicates' 95% percentile range. Failed replicates are kept as failed mass.
//!
//! The state executes under both inference modes and is unlicensed: its verdicts are checks, not
//! a calibration claim, and no coverage record names it.

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_estimate::{TransportOverlapDiagnostic, trial_to_target_effect};
use antecedent_identify::TransportIdentification;
use antecedent_kernels::erfc;
use antecedent_validate::RefutationReport;

use super::{CausalError, RefuteSuite};

/// Two-sided level of the calibration tests.
const SIGNIFICANCE: f64 = 0.05;
/// Share of the trial rows below which the effective sample size of a weight set is flagged.
const MIN_EFFECTIVE_SHARE: f64 = 0.1;
/// Bootstrap replicates when the study requested none.
const DEFAULT_BOOTSTRAP_REPLICATES: u32 = 49;

/// Borrowed inputs of one transport refutation.
pub(super) struct TransportRefutationInput<'a> {
    pub(super) transport: &'a TransportIdentification,
    pub(super) outcomes: &'a [f64],
    pub(super) treatment: &'a [bool],
    pub(super) trial: &'a [bool],
    pub(super) selection: &'a [f64],
    pub(super) propensity: &'a [f64],
    pub(super) overlap: antecedent_estimate::TransportOverlapReport,
    /// The published transported contrast (the posterior mean under Bayesian inference).
    pub(super) published: f64,
    pub(super) suite: RefuteSuite,
    /// Block length when the rows are the consecutive times of one dependent series.
    pub(super) series_block: Option<usize>,
    /// The study's replicate count (zero: the default budget).
    pub(super) replicates: u32,
}

/// The reports of the requested suite; empty for [`RefuteSuite::None`].
///
/// # Errors
///
/// A cancelled run, or a replicate budget the execution pool refuses.
pub(super) fn refute_transport(
    input: &TransportRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, CausalError> {
    if input.suite == RefuteSuite::None {
        return Ok(Vec::new());
    }
    let trial_rows = input.trial.iter().filter(|&&member| member).count();
    let mut reports = vec![
        selection_calibration(input.trial, input.selection, input.series_block),
        arm_calibration((input.trial, input.treatment, input.propensity), input.series_block),
        overlap_report(
            "transport.selection_overlap",
            input.overlap.selection,
            trial_rows,
            "selection-odds weights",
        ),
        overlap_report(
            "transport.treatment_overlap",
            input.overlap.treatment,
            trial_rows,
            "within-trial arm weights",
        ),
    ];
    if input.suite == RefuteSuite::Full {
        reports.push(bootstrap_range(input, ctx)?);
    }
    Ok(reports)
}

/// Two-sided normal p-value of a sum of centred residuals with the given variance. A variance
/// of zero means the sum is exactly determined: it either matches or it does not.
fn z_p_value(total: f64, variance: f64) -> f64 {
    if variance > 0.0 && variance.is_finite() {
        erfc((total / variance.sqrt()).abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
    } else if total.abs() <= 1e-9 {
        1.0
    } else {
        0.0
    }
}

/// Variance of the sum of `residuals`: the independent-rows value, or on one dependent series
/// the sum of squared block sums (the residuals have mean zero under the supplied
/// probabilities, so the blocks are not re-centred).
fn sum_variance(residuals: &[f64], independent: f64, block: Option<usize>) -> f64 {
    match block {
        None => independent,
        Some(length) => {
            let length = length.clamp(1, residuals.len().max(1));
            residuals
                .chunks(length)
                .map(|chunk| {
                    let sum: f64 = chunk.iter().sum();
                    sum * sum
                })
                .sum()
        }
    }
}

/// A calibration report: `observed` against `expected`, the sum of the supplied probabilities.
fn calibration_report(
    id: &str,
    (observed, expected): (f64, f64),
    variance: f64,
    what: &str,
) -> RefutationReport {
    let p_value = z_p_value(observed - expected, variance);
    let passed = p_value >= SIGNIFICANCE;
    RefutationReport::new(
        id,
        observed,
        expected,
        p_value,
        true,
        passed,
        (!passed).then(|| {
            std::sync::Arc::from(format!(
                "observed {observed} {what} where the supplied probabilities imply {expected:.2} \
                 (p = {p_value:.4}); the supplied probabilities do not describe these rows"
            ))
        }),
        1,
    )
}

fn selection_calibration(
    trial: &[bool],
    selection: &[f64],
    block: Option<usize>,
) -> RefutationReport {
    let residuals: Vec<f64> = trial
        .iter()
        .zip(selection)
        .map(|(&member, &probability)| f64::from(u8::from(member)) - probability)
        .collect();
    let independent: f64 =
        selection.iter().map(|probability| probability * (1.0 - probability)).sum();
    let observed = trial.iter().filter(|&&member| member).count() as f64;
    let expected: f64 = selection.iter().sum();
    calibration_report(
        "transport.selection_calibration",
        (observed, expected),
        sum_variance(&residuals, independent, block),
        "trial rows",
    )
}

fn arm_calibration(
    (trial, treatment, propensity): (&[bool], &[bool], &[f64]),
    block: Option<usize>,
) -> RefutationReport {
    let mut residuals = Vec::new();
    let (mut independent, mut observed, mut expected) = (0.0, 0.0, 0.0);
    for ((&member, &treated), &probability) in trial.iter().zip(treatment).zip(propensity) {
        if !member {
            continue;
        }
        let treated = f64::from(u8::from(treated));
        residuals.push(treated - probability);
        independent += probability * (1.0 - probability);
        observed += treated;
        expected += probability;
    }
    calibration_report(
        "transport.arm_calibration",
        (observed, expected),
        sum_variance(&residuals, independent, block),
        "treated trial rows",
    )
}

fn overlap_report(
    id: &str,
    diagnostic: TransportOverlapDiagnostic,
    trial_rows: usize,
    what: &str,
) -> RefutationReport {
    let share =
        if trial_rows == 0 { 0.0 } else { diagnostic.effective_sample_size / trial_rows as f64 };
    let passed = share.is_finite() && share >= MIN_EFFECTIVE_SHARE;
    RefutationReport::new(
        id,
        diagnostic.effective_sample_size,
        trial_rows as f64,
        if share.is_finite() { share } else { 0.0 },
        true,
        passed,
        (!passed).then(|| {
            std::sync::Arc::from(format!(
                "the {what} leave an effective sample size of {:.1} of {trial_rows} trial rows \
                 ({} extreme weights, probabilities from {:.4} to {:.4})",
                diagnostic.effective_sample_size,
                diagnostic.extreme_weight_count,
                diagnostic.probability_min,
                diagnostic.probability_max,
            ))
        }),
        1,
    )
}

/// `n` row indices drawn with replacement, or circular blocks of `block` consecutive rows.
fn resample_rows(
    n: usize,
    block: Option<usize>,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<usize> {
    let mut draw = || {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform in [0, 1) times n floors to a row index below n"
        )]
        let row = ((rng.next_f64() * n as f64) as usize).min(n.saturating_sub(1));
        row
    };
    match block {
        None => (0..n).map(|_| draw()).collect(),
        Some(length) => {
            let length = length.clamp(1, n.max(1));
            let mut rows = Vec::with_capacity(n + length);
            while rows.len() < n {
                let start = draw();
                rows.extend((0..length).map(|offset| (start + offset) % n));
            }
            rows.truncate(n);
            rows
        }
    }
}

fn gather<T: Copy>(values: &[T], rows: &[usize]) -> Vec<T> {
    rows.iter().map(|&row| values[row]).collect()
}

/// One transport certificate refitted by the whole-procedure bootstrap, with the report id and
/// the value the bootstrap range must contain.
struct BootstrapMember<'a> {
    id: String,
    transport: &'a TransportIdentification,
    published: f64,
}

fn bootstrap_range(
    input: &TransportRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, CausalError> {
    let member = BootstrapMember {
        id: "transport.bootstrap_range".to_owned(),
        transport: input.transport,
        published: input.published,
    };
    bootstrap_ranges(input, &[member], ctx)?.into_iter().next().ok_or_else(|| {
        CausalError::Compile { message: "the transport bootstrap produced no report".into() }
    })
}

/// Member-specific checks of a class: the members of a Cpdag or a Pag may certify different
/// formulas, and each is read by the trial-to-target weighting through its own certificate, so
/// its transported contrast is bootstrapped on its own. Every member is refitted on the same
/// resampled tables as the representative the base suite is about, and its published value
/// (`published`) must lie in its own 95% percentile range. Only the full suite bootstraps.
///
/// `members` pairs a case index with the certificate and the plug-in value of that case.
///
/// # Errors
///
/// A cancelled run, or a replicate budget the execution pool refuses.
pub(super) fn refute_transport_members(
    input: &TransportRefutationInput<'_>,
    members: &[(usize, &TransportIdentification, f64)],
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, CausalError> {
    if input.suite != RefuteSuite::Full || members.is_empty() {
        return Ok(Vec::new());
    }
    let members: Vec<BootstrapMember<'_>> = members
        .iter()
        .map(|&(index, transport, published)| BootstrapMember {
            id: format!("transport.bootstrap_range.case{index}"),
            transport,
            published,
        })
        .collect();
    bootstrap_ranges(input, &members, ctx)
}

fn bootstrap_ranges(
    input: &TransportRefutationInput<'_>,
    members: &[BootstrapMember<'_>],
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, CausalError> {
    let n = input.outcomes.len();
    let replicates =
        if input.replicates == 0 { DEFAULT_BOOTSTRAP_REPLICATES } else { input.replicates };
    let width = n.checked_mul(6).ok_or_else(|| CausalError::Resource {
        message: "transport bootstrap size overflow".to_owned(),
    })?;
    let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, members.len(), width)?;
    let mut values: Vec<Vec<f64>> = members
        .iter()
        .map(|_| Vec::with_capacity(usize::try_from(replicates).unwrap_or(0)))
        .collect();
    let mut failed = vec![0_u32; members.len()];
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let mut rng = ctx
            .rng
            .stream_for(StreamDomain::Resample, 0x7A17_B007_u64.wrapping_add(u64::from(replicate)));
        let rows = resample_rows(n, input.series_block, &mut rng);
        let (outcomes, treatment, trial, selection, propensity) = (
            gather(input.outcomes, &rows),
            gather(input.treatment, &rows),
            gather(input.trial, &rows),
            gather(input.selection, &rows),
            gather(input.propensity, &rows),
        );
        for (slot, member) in members.iter().enumerate() {
            let estimate = trial_to_target_effect(
                member.transport,
                &outcomes,
                &treatment,
                &trial,
                &selection,
                &propensity,
                None,
            );
            match estimate {
                Ok(estimate) if estimate.ipw.is_finite() => values[slot].push(estimate.ipw),
                _ => failed[slot] += 1,
            }
        }
    }
    Ok(members
        .iter()
        .zip(values)
        .zip(failed)
        .map(|((member, values), failed)| {
            range_report(&member.id, member.published, values, (failed, replicates))
        })
        .collect())
}

/// The report of one bootstrap: the published value against the 95% percentile range.
fn range_report(
    id: &str,
    published: f64,
    mut values: Vec<f64>,
    (failed, replicates): (u32, u32),
) -> RefutationReport {
    let ok = u32::try_from(values.len()).unwrap_or(u32::MAX);
    if values.len() < 2 {
        return RefutationReport::new(
            id,
            published,
            0.0,
            0.0,
            false,
            false,
            Some(std::sync::Arc::from(format!(
                "only {ok} of {replicates} bootstrap replicates produced a transported contrast \
                 ({failed} failed)"
            ))),
            ok,
        );
    }
    values.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "p in [0, 1] times the last index is a valid index"
        )]
        let index = (p * (values.len() - 1) as f64).round() as usize;
        values[index.min(values.len() - 1)]
    };
    let (lower, upper) = (quantile(0.025), quantile(0.975));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let passed = (lower..=upper).contains(&published);
    RefutationReport::new(
        id,
        published,
        mean,
        upper - lower,
        false,
        passed,
        (!passed).then(|| {
            std::sync::Arc::from(format!(
                "the published contrast {published:.4} lies outside the bootstrap 95% range \
                 [{lower:.4}, {upper:.4}] ({failed} of {replicates} replicates failed)"
            ))
        }),
        ok,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normal_tail_matches_the_closed_form_at_the_customary_quantiles() {
        // Two-sided tail of z = 1.96 is 0.0500 and of z = 2.5758 is 0.0100.
        assert!((z_p_value(1.96, 1.0) - 0.05).abs() < 1e-3);
        assert!((z_p_value(-2.5758, 1.0) - 0.01).abs() < 1e-4);
        assert!((z_p_value(0.0, 4.0) - 1.0).abs() < 1e-12);
        // No variance: an exact match passes and any gap fails.
        assert!((z_p_value(0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!(z_p_value(1.0, 0.0).abs() < 1e-12);
    }

    #[test]
    fn a_trial_count_far_from_the_supplied_probabilities_is_flagged() {
        // Ten rows with selection probability one half: five trial rows is the expectation.
        let selection = [0.5; 10];
        let balanced = selection_calibration(
            &[true, true, true, true, true, false, false, false, false, false],
            &selection,
            None,
        );
        assert!(balanced.passed);
        assert!((balanced.comparison - 1.0).abs() < 1e-12);
        // Nine trial rows: z = 4 / sqrt(2.5) = 2.5298, two-sided p = 0.01141 (normal tail).
        let skewed = selection_calibration(
            &[true, true, true, true, true, true, true, true, true, false],
            &selection,
            None,
        );
        assert!(!skewed.passed);
        assert!((skewed.comparison - 0.011_4).abs() < 1e-3, "{}", skewed.comparison);
        assert!((skewed.original_ate - 9.0).abs() < 1e-12);
        assert!((skewed.refuted_ate - 5.0).abs() < 1e-12);
    }

    #[test]
    fn the_arm_count_is_tested_on_trial_rows_only() {
        // Four trial rows at probability one half with three treated: z = 1 / 1 = 1, p = 0.3173.
        // The target rows (trial = false) carry arbitrary arms and probabilities and are skipped.
        let trial = [true, true, true, true, false, false];
        let treatment = [true, true, true, false, true, true];
        let propensity = [0.5, 0.5, 0.5, 0.5, 0.01, 0.99];
        let report = arm_calibration((&trial, &treatment, &propensity), None);
        assert!((report.original_ate - 3.0).abs() < 1e-12);
        assert!((report.refuted_ate - 2.0).abs() < 1e-12);
        assert!((report.comparison - 0.317_3).abs() < 1e-3, "{}", report.comparison);
        assert!(report.passed);
    }

    #[test]
    fn a_dependent_series_uses_the_squared_block_sums() {
        // Residuals +1, +1, -1, -1 in blocks of two sum to +2 and -2: variance 8, not the
        // independent value.
        let residuals = [1.0, 1.0, -1.0, -1.0];
        assert!((sum_variance(&residuals, 1.0, Some(2)) - 8.0).abs() < 1e-12);
        assert!((sum_variance(&residuals, 1.0, None) - 1.0).abs() < 1e-12);
        // A block longer than the series is one block of the whole series.
        assert!(sum_variance(&residuals, 1.0, Some(40)).abs() < 1e-12);
    }

    #[test]
    fn circular_blocks_are_runs_of_consecutive_rows() {
        let ctx = ExecutionContext::for_tests(5);
        let mut rng = ctx.rng.stream(1);
        let rows = resample_rows(7, Some(3), &mut rng);
        assert_eq!(rows.len(), 7);
        for chunk in rows.chunks(3).filter(|chunk| chunk.len() == 3) {
            assert_eq!(chunk[1], (chunk[0] + 1) % 7);
            assert_eq!(chunk[2], (chunk[0] + 2) % 7);
        }
        let independent = resample_rows(7, None, &mut ctx.rng.stream(2));
        assert_eq!(independent.len(), 7);
        assert!(independent.iter().all(|&row| row < 7));
    }

    #[test]
    fn a_published_contrast_is_judged_against_its_own_bootstrap_range() {
        // 1.0, 1.01, ..., 1.99: the 95% percentile range is [1.02, 1.97].
        let values: Vec<f64> = (0..100).map(|i| 1.0 + 0.01 * f64::from(i)).collect();
        let inside = range_report("transport.bootstrap_range.case3", 1.5, values.clone(), (0, 100));
        assert!(inside.passed);
        assert_eq!(inside.refuter.as_ref(), "transport.bootstrap_range.case3");
        assert!((inside.refuted_ate - 1.495).abs() < 1e-9);
        assert!((inside.comparison - 0.95).abs() < 1e-9, "{}", inside.comparison);
        let outside = range_report("transport.bootstrap_range.case3", 2.5, values, (0, 100));
        assert!(!outside.passed && outside.failure_condition.is_some());
        // Fewer than two usable replicates is a failed report that names the loss.
        let lost = range_report("transport.bootstrap_range", 1.0, vec![1.0], (48, 49));
        assert!(!lost.passed);
        assert_eq!(lost.replicates, 1);
    }

    #[test]
    fn an_overlap_share_under_a_tenth_of_the_trial_is_flagged() {
        let diagnostic = |effective| TransportOverlapDiagnostic {
            probability_min: 0.2,
            probability_max: 0.8,
            effective_sample_size: effective,
            extreme_weight_count: 0,
        };
        let healthy = overlap_report("transport.selection_overlap", diagnostic(60.0), 100, "w");
        assert!(healthy.passed && (healthy.comparison - 0.6).abs() < 1e-12);
        let thin = overlap_report("transport.selection_overlap", diagnostic(5.0), 100, "w");
        assert!(!thin.passed && (thin.comparison - 0.05).abs() < 1e-12);
        let empty = overlap_report("transport.selection_overlap", diagnostic(5.0), 0, "w");
        assert!(!empty.passed);
    }
}

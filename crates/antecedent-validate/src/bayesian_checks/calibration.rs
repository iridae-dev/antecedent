//! Monte Carlo summaries shared by the predictive checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::KernelPolicy;
use antecedent_kernels::{PosteriorReduceOp, reduce_posterior_draws};

use super::predictive::{PredictiveCheckKind, PredictiveCheckReport, PredictiveNoise};

pub(super) fn summarize_check(
    kind: PredictiveCheckKind,
    observed: f64,
    summaries: &[f64],
    n_sims: u32,
) -> PredictiveCheckReport {
    let policy = KernelPolicy::default_policy();
    let mean = reduce_posterior_draws(summaries, PosteriorReduceOp::Mean, &policy).unwrap_or(0.0);
    let sd = reduce_posterior_draws(summaries, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
    let n = summaries.len() as f64;
    // (1 + count) / (1 + n) form (Davison & Hinkley): an exact-zero Monte Carlo
    // p-value is never valid evidence with a finite sample, so both tails are
    // bounded below by 1/(n+1) and the two-sided p-value by 2/(n+1).
    let below = summaries.iter().filter(|&&x| x <= observed).count() as f64;
    let above = summaries.iter().filter(|&&x| x >= observed).count() as f64;
    let p_lower = (1.0 + below) / (1.0 + n);
    let p_upper = (1.0 + above) / (1.0 + n);
    let p = (2.0 * p_lower.min(p_upper)).min(1.0);
    PredictiveCheckReport {
        kind,
        observed,
        predictive_mean: mean,
        predictive_sd: sd,
        p_value: p,
        location_tails: [p_lower, p_upper],
        // Populated by [`summarize_predictive_check`], the two-axis wrapper around this
        // function; callers of `summarize_check` directly (unit tests exercising the
        // Monte Carlo p-value formula) only care about the location axis above.
        observed_dispersion: 0.0,
        predictive_dispersion_mean: 0.0,
        dispersion_p_value: 1.0,
        dispersion_tails: [1.0, 1.0],
        n_sims,
        serial: None,
        residual: Vec::new(),
        residual_p_value: 1.0,
        noise: PredictiveNoise::Likelihood,
    }
}

/// Push the per-simulation location (mean) and dispersion (cross-observation SD)
/// summary statistics for one replicated outcome vector `y_pred`.
///
/// Dispersion is the sample SD of `y_pred` *across observations within a single
/// simulation* — how spread the replicated outcomes are over the design — which is
/// exactly the axis a mean-only PPC statistic is blind to (see D3 / defect C:
/// a model with an unbiased predictive mean but a badly wrong predictive spread
/// must be caught here, not just on the mean axis).
pub(super) fn push_mean_and_dispersion(
    y_pred: &[f64],
    policy: KernelPolicy,
    mean_summaries: &mut Vec<f64>,
    disp_summaries: &mut Vec<f64>,
) {
    let n = y_pred.len().max(1) as f64;
    let mean_y = y_pred.iter().sum::<f64>() / n;
    let sd_y = reduce_posterior_draws(y_pred, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
    mean_summaries.push(mean_y);
    disp_summaries.push(sd_y);
}

/// Two-axis predictive check: combines a location (mean) Monte Carlo check with a
/// dispersion (cross-observation SD) Monte Carlo check via [`summarize_check`], and
/// merges both into a single [`PredictiveCheckReport`].
pub(super) fn summarize_predictive_check(
    kind: PredictiveCheckKind,
    outcome: &[f64],
    policy: KernelPolicy,
    mean_summaries: &[f64],
    disp_summaries: &[f64],
    n_sims: u32,
    noise: PredictiveNoise,
) -> PredictiveCheckReport {
    let n = outcome.len().max(1) as f64;
    let observed_mean = outcome.iter().sum::<f64>() / n;
    let observed_dispersion =
        reduce_posterior_draws(outcome, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
    let mean_report = summarize_check(kind, observed_mean, mean_summaries, n_sims);
    let disp_report = summarize_check(kind, observed_dispersion, disp_summaries, n_sims);
    let (location_tails, dispersion_tails) = match noise {
        PredictiveNoise::Likelihood => (mean_report.location_tails, disp_report.location_tails),
        // Unbounded noise reaches any larger spread.
        PredictiveNoise::MeanOnlyImproperScale => {
            (mean_report.location_tails, [disp_report.location_tails[0], 1.0])
        }
    };
    let two_sided = |tails: [f64; 2]| (2.0 * tails[0].min(tails[1])).min(1.0);
    PredictiveCheckReport {
        kind,
        observed: mean_report.observed,
        predictive_mean: mean_report.predictive_mean,
        predictive_sd: mean_report.predictive_sd,
        p_value: two_sided(location_tails),
        location_tails,
        observed_dispersion,
        predictive_dispersion_mean: disp_report.predictive_mean,
        dispersion_p_value: two_sided(dispersion_tails),
        dispersion_tails,
        n_sims,
        serial: None,
        residual: Vec::new(),
        residual_p_value: 1.0,
        noise,
    }
}

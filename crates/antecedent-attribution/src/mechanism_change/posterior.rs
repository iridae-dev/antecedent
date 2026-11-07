//! Posterior mechanism-change probabilities from residual comparisons.

use std::sync::Arc;

use antecedent_core::{ExecutionContext, MechanismChangeQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_model::CompiledCausalModel;

use crate::error::AttributionError;
use crate::result::MechanismChangeDetection;

use super::{
    PopulationRuns, charge_run_validation, gaussian_lr_statistic, poll_run_cancellation,
    residual_pairs, validate_runs,
};

/// The posterior probability that one tested mechanism changed.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismChangePosterior {
    /// Tested node.
    pub variable: VariableId,
    /// Gaussian likelihood-ratio statistic `2 log LR` of separate against common residual mean
    /// and variance.
    pub statistic: f64,
    /// Variance inflation of the residuals' serial dependence (at least 1): the likelihood is
    /// tempered by its reciprocal, the effective-sample deflation.
    pub inflation: f64,
    /// `log` Bayes factor of a change against no change from the conjugate
    /// normal-inverse-gamma marginal likelihoods (see [`mechanism_change_posterior`]).
    pub log_bayes_factor: f64,
    /// The declared prior probability of a change.
    pub prior: f64,
    /// The posterior probability of a change given the residuals.
    pub probability: f64,
}

/// Prior pseudo-count of the residual mean (its prior sd is `sigma / sqrt(KAPPA0)`).
const NIG_KAPPA0: f64 = 1.0;
/// Shape of the inverse-gamma prior of the residual variance (prior mean `beta0`).
const NIG_ALPHA0: f64 = 2.0;

/// `log` marginal likelihood of `n` (effective) Gaussian observations with mean `mean` and
/// centred sum of squares `ss` under the conjugate prior `mu | s2 ~ N(0, s2 / kappa0)`,
/// `s2 ~ InvGamma(alpha0, beta0)`.
pub(super) fn nig_log_marginal(n: f64, mean: f64, ss: f64, beta0: f64) -> f64 {
    let kappa_n = NIG_KAPPA0 + n;
    let alpha_n = NIG_ALPHA0 + 0.5 * n;
    let beta_n = beta0 + 0.5 * ss + 0.5 * NIG_KAPPA0 * n * mean * mean / kappa_n;
    antecedent_stats::ln_gamma(alpha_n) - antecedent_stats::ln_gamma(NIG_ALPHA0)
        + NIG_ALPHA0 * beta0.ln()
        - alpha_n * beta_n.ln()
        + 0.5 * (NIG_KAPPA0 / kappa_n).ln()
        - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
}

impl MechanismChangePosterior {
    /// The posterior as a detection record: `changed` when the probability exceeds one half, the
    /// adjusted statistic, no p-value (`NaN`), and the probability itself.
    #[must_use]
    pub fn into_detection(self) -> MechanismChangeDetection {
        MechanismChangeDetection {
            variable: self.variable,
            changed: self.probability > 0.5,
            statistic: self.statistic / self.inflation,
            p_value: f64::NAN,
            adjusted_p_value: None,
            method: Arc::from("posterior_probability.bic_block"),
            n_permutations: None,
            p_value_floor: None,
            posterior_probability: Some(self.probability),
        }
    }
}

/// Arithmetic mean through the shared compensated kernel; `0` for an empty sample.
fn sample_mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { antecedent_kernels::mean(xs) }
}

/// Bartlett long-run variance inflation `1 + 2 Σ_{k<block} (1 − k/block) ρ_k` of the residuals
/// of two populations, each centred on its own mean, floored at 1.
pub(super) fn serial_inflation(rb: &[f64], rc: &[f64], block: usize) -> f64 {
    match serial_inflation_with_runs(rb, rc, block, None, None) {
        Ok(value) => value,
        Err(_) => unreachable!("no cancellation context on the static path"),
    }
}

pub(super) fn serial_inflation_with_runs(
    rb: &[f64],
    rc: &[f64],
    block: usize,
    runs: Option<PopulationRuns<'_>>,
    ctx: Option<&ExecutionContext>,
) -> Result<f64, AttributionError> {
    let centred = |xs: &[f64]| {
        let mean = sample_mean(xs);
        xs.iter().map(|x| x - mean).collect::<Vec<f64>>()
    };
    let (a, b) = (centred(rb), centred(rc));
    let n = (a.len() + b.len()) as f64;
    let variance = (a.iter().chain(&b).map(|x| x * x).sum::<f64>() / n).max(f64::MIN_POSITIVE);
    let lag_product = |series: &[f64], k: usize| -> Result<f64, AttributionError> {
        let mut sum = 0.0;
        for (i, (x, y)) in series.iter().zip(series.iter().skip(k)).enumerate() {
            if let Some(ctx) = ctx {
                poll_run_cancellation(ctx, i)?;
            }
            sum += x * y;
        }
        Ok(sum)
    };
    let block = block.max(1);
    let mut inflation = 1.0;
    for k in 1..block.min(a.len().min(b.len())) {
        if let Some(ctx) = ctx {
            poll_run_cancellation(ctx, k)?;
        }
        let product = if let Some((ar, br)) = runs {
            let mut left = 0.0;
            let mut right = 0.0;
            for (i, r) in ar.iter().enumerate() {
                if let Some(ctx) = ctx {
                    poll_run_cancellation(ctx, i)?;
                }
                left += lag_product(&a[r.clone()], k)?;
            }
            for (i, r) in br.iter().enumerate() {
                if let Some(ctx) = ctx {
                    poll_run_cancellation(ctx, i)?;
                }
                right += lag_product(&b[r.clone()], k)?;
            }
            left + right
        } else {
            lag_product(&a, k)? + lag_product(&b, k)?
        };
        let rho = product / (n * variance);
        inflation += 2.0 * (1.0 - k as f64 / block as f64) * rho;
    }
    Ok(inflation.max(1.0))
}

/// The posterior probability of a mechanism change for each target, under a declared prior over
/// the change indicator.
///
/// Per target, the residuals of the two populations (as in [`detect_mechanism_changes_block`])
/// are modelled as Gaussian. With no change both populations share one mean and variance; with a
/// change each has its own. Each (mean, variance) carries the conjugate
/// normal-inverse-gamma prior `mu | s2 ~ N(0, s2)`, `s2 ~ InvGamma(2, v)` with `v` the pooled
/// residual variance (an empirical-Bayes scale, the same under both models), so the marginal
/// likelihoods are closed form and the Bayes factor is exact for that model. The likelihood is
/// tempered by the reciprocal of the Bartlett variance inflation of the residuals' serial
/// dependence at `block` lags (effective-sample deflation). With the declared prior `π` of a
/// change, `P(change | data) = π BF / (π BF + 1 − π)`. The prior is the caller's declaration and
/// is never assumed; the posterior is for the Gaussian model, not model-free.
///
/// # Errors
///
/// As [`detect_mechanism_changes`], and a `prior` outside (0, 1).
pub fn mechanism_change_posterior(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    prior: f64,
    block: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangePosterior>, AttributionError> {
    mechanism_change_posterior_impl(graph_model, data, query, prior, block, None, None, ctx)
}

/// Posterior detection using separate contiguous compact runs for each population.
/// Each run list must partition its compact population exactly.
#[allow(clippy::too_many_arguments, reason = "the inputs describe one attribution calculation")]
pub fn mechanism_change_posterior_with_runs(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    prior: f64,
    block: usize,
    baseline_runs: &[std::ops::Range<usize>],
    comparison_runs: &[std::ops::Range<usize>],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangePosterior>, AttributionError> {
    mechanism_change_posterior_impl(
        graph_model,
        data,
        query,
        prior,
        block,
        Some((baseline_runs, comparison_runs)),
        Some(budget),
        ctx,
    )
}

#[allow(clippy::too_many_arguments, reason = "the inputs describe one attribution calculation")]
fn mechanism_change_posterior_impl(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    prior: f64,
    block: usize,
    runs: Option<PopulationRuns<'_>>,
    mut budget: Option<&mut antecedent_core::TemporalBudget>,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangePosterior>, AttributionError> {
    if let (Some(runs), Some(budget)) = (runs, budget.as_deref_mut()) {
        charge_run_validation(runs, budget, ctx)?;
    }
    if !(prior > 0.0 && prior < 1.0) {
        return Err(AttributionError::invalid_input(
            "the prior probability of a change lies strictly between 0 and 1",
        ));
    }
    let pairs = residual_pairs(graph_model, data, query, ctx)?;
    if let Some((a, b)) = runs {
        if let Some((_, rb, rc)) = pairs.first() {
            validate_runs(a, rb.len(), ctx)?;
            validate_runs(b, rc.len(), ctx)?;
        }
    }
    if let Some(budget) = budget {
        let mut work = 0_u64;
        for (_, rb, rc) in &pairs {
            let n = rb
                .len()
                .checked_add(rc.len())
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
            let cost = n
                .checked_mul(block.min(n).max(1))
                .and_then(|v| u64::try_from(v).ok())
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
            work = work
                .checked_add(cost)
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
        }
        budget
            .charge_work(work)
            .map_err(|e| AttributionError::Budget { message: e.to_string() })?;
    }
    pairs
        .into_iter()
        .map(|(variable, rb, rc)| {
            if runs.is_some() && ctx.cancellation.is_cancelled() {
                return Err(AttributionError::Cancelled);
            }
            let statistic = gaussian_lr_statistic(&rb, &rc);
            let inflation = if runs.is_some() {
                serial_inflation_with_runs(&rb, &rc, block, runs, Some(ctx))?
            } else {
                serial_inflation(&rb, &rc, block)
            };
            // Tempering by the reciprocal inflation: the sufficient statistics count as
            // n / inflation effective observations.
            let weight = 1.0 / inflation;
            let part = |xs: &[f64]| {
                let n = xs.len() as f64;
                let mean = sample_mean(xs);
                let ss = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
                (weight * n, mean, weight * ss)
            };
            let pooled: Vec<f64> = rb.iter().chain(rc.iter()).copied().collect();
            let (n_all, mean_all, ss_all) = part(&pooled);
            let (n_a, mean_a, ss_a) = part(&rb);
            let (n_b, mean_b, ss_b) = part(&rc);
            // The variance prior is centred on the pooled residual variance (an
            // empirical-Bayes scale, the same under both models).
            let beta0 = (ss_all / n_all.max(1.0)).max(f64::MIN_POSITIVE) * (NIG_ALPHA0 - 1.0);
            let log_bayes_factor = nig_log_marginal(n_a, mean_a, ss_a, beta0)
                + nig_log_marginal(n_b, mean_b, ss_b, beta0)
                - nig_log_marginal(n_all, mean_all, ss_all, beta0);
            let log_odds = (prior / (1.0 - prior)).ln() + log_bayes_factor;
            let probability = 1.0 / (1.0 + (-log_odds).exp());
            Ok(MechanismChangePosterior {
                variable,
                statistic,
                inflation,
                log_bayes_factor,
                prior,
                probability,
            })
        })
        .collect()
}

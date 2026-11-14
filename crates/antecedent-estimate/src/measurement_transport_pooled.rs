//! Differential measurement error pooled across populations and carried to a target.
//!
//! [`crate::measurement_transport`] carries a misclassification model to a target only under a
//! declared invariance premise (the error rates are the same in source and target), and a
//! differential declaration carries nothing. Between the two lies the case this module
//! answers: the sensitivity and specificity *differ* by population, and several validated
//! populations inform how much. Their logits are modelled as draws from a population
//! distribution, `logit(se_k) ~ N(mu_se, tau_se^2)` and likewise for specificity, with each
//! population's validation sample giving `logit(se_k)` up to its sampling variance. An
//! invariant error is the special case `tau^2 = 0`, and a heterogeneous one widens the target's
//! predictive error rates.
//!
//! A handful of populations estimates `tau^2` poorly, so its uncertainty is propagated rather
//! than a point estimate plugged in. The generalized statistic
//! `Q(tau^2) = sum_k (y_k - mu(tau^2))^2 / (v_k + tau^2)` is chi-squared on `K - 1` degrees of
//! freedom at the true `tau^2` and decreases in it, so each propagation draw takes a chi-squared
//! variate `q` and the `tau^2` solving `Q(tau^2) = q` (zero when `Q(0) <= q`): a draw from the
//! Q-profile confidence distribution of `tau^2`. The pooled mean is then drawn from
//! `N(mu(tau^2), 1 / sum_k 1 / (v_k + tau^2))` and the target's logit rate from `N(mu, tau^2)`.
//! When the target has its own validation sample it is pooled as one more population and its
//! rate is drawn from the normal law of the rate given the drawn `mu`, `tau^2` and its own
//! estimate. The reported `tau^2` is the Paule-Mandel estimate, the solution at `q = K - 1`.
//! A plug-in `tau^2` with a normal predictive law under-covers with few populations
//! (measured 0.90 at six), which is why the draw is used.
//!
//! The corrected target prevalence is the Rogan-Gladen functional
//! `(p* + sp - 1) / (se + sp - 1)` of the target's recorded prevalence `p*`, whose
//! uncertainty is its Beta posterior under a uniform prior. Every draw takes one recorded
//! prevalence and one pair of rates; a draw whose Youden index is not positive is discarded and
//! counted, and a prevalence outside `[0, 1]` is truncated and counted. The result is a
//! simulation interval. At `tau^2 = 0` every drawn `tau^2` is at least the truth, so the
//! interval is conservative there.
//!
//! References: Paule and Mandel (1982), doi:10.6028/jres.087.022; Viechtbauer (2007),
//! doi:10.1002/sim.2514 (Q-profile); Higgins, Thompson and Spiegelhalter (2009),
//! doi:10.1111/j.1467-985X.2008.00552.x (prediction for a new population); Rogan and Gladen
//! (1978), doi:10.1093/oxfordjournals.aje.a112510.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, ExecutionContext, StreamDomain};

use crate::error::EstimationError;
use crate::util::refuse;

/// Stream key of the propagation draws.
const POOLED_STREAM: u64 = 0x9001_D1FF_0000_0000;
/// Largest share of draws allowed a non-positive Youden index.
const MAX_DISCARDED_SHARE: f64 = 0.05;

/// One population's validation counts of a misclassified binary outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PopulationValidation {
    /// Population label.
    pub population: String,
    /// Recorded positive among true positives.
    pub true_positive: u64,
    /// Recorded negative among true positives.
    pub false_negative: u64,
    /// Recorded negative among true negatives.
    pub true_negative: u64,
    /// Recorded positive among true negatives.
    pub false_positive: u64,
}

/// A pooled-differential transport request.
#[derive(Clone, Copy, Debug)]
pub struct PooledDifferentialRequest<'a> {
    /// Validated source populations; at least two, distinct from the target.
    pub sources: &'a [PopulationValidation],
    /// The target's own validation sample, when it has one.
    pub target_validation: Option<&'a PopulationValidation>,
    /// Target population label.
    pub target_population: &'a str,
    /// Recorded positives in the target sample.
    pub observed_positive: u64,
    /// Target sample size.
    pub observed_total: u64,
    /// Propagation draws; at least 1000.
    pub draws: usize,
}

/// A random-effects summary of one error rate across populations, on the logit scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PooledRate {
    /// Pooled mean logit.
    pub mean_logit: f64,
    /// Variance of the pooled mean.
    pub mean_variance: f64,
    /// Estimated between-population variance `tau^2` (Paule-Mandel).
    pub between_population_variance: f64,
    /// Mean of the target's rate under its predictive (or posterior) law, on the rate scale.
    pub target_rate: f64,
    /// Standard deviation of the target's rate under that law.
    pub target_rate_sd: f64,
}

/// The corrected target prevalence with the pooled error propagated.
#[derive(Clone, Debug, PartialEq)]
pub struct PooledDifferentialCorrection {
    /// Target population label.
    pub target_population: String,
    /// Mean of the corrected prevalence draws.
    pub prevalence: f64,
    /// Equal-tailed 95% simulation interval of the corrected prevalence.
    pub interval: (f64, f64),
    /// Sensitivity across populations and at the target.
    pub sensitivity: PooledRate,
    /// Specificity across populations and at the target.
    pub specificity: PooledRate,
    /// Draws discarded for a non-positive Youden index.
    pub discarded_draws: usize,
    /// Draws truncated into `[0, 1]`.
    pub truncated_draws: usize,
    /// Retained draws of the corrected prevalence.
    pub draws: Vec<f64>,
    /// Calibration status: always `not_measured`.
    pub calibration: &'static str,
}

fn invalid(message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("invalid_argument"), "measurement_pooled.invalid", message)
}

/// Logit of a proportion with the Haldane continuity correction and its variance.
fn logit_with_variance(successes: u64, failures: u64) -> (f64, f64) {
    let (a, b) = (successes as f64 + 0.5, failures as f64 + 0.5);
    ((a / b).ln(), 1.0 / a + 1.0 / b)
}

/// The generalized Q statistic of `(estimate, variance)` pairs at `tau2`, with the weighted
/// mean and its variance under the weights `1 / (variance + tau2)`.
fn generalized_q(estimates: &[(f64, f64)], tau2: f64) -> (f64, f64, f64) {
    let total: f64 = estimates.iter().map(|(_, v)| 1.0 / (v + tau2)).sum();
    let mean = estimates.iter().map(|(y, v)| y / (v + tau2)).sum::<f64>() / total;
    let q = estimates.iter().map(|(y, v)| (y - mean).powi(2) / (v + tau2)).sum();
    (q, mean, 1.0 / total)
}

/// The between-population variance at which the generalized Q statistic equals `level`; zero
/// when the statistic at zero is already at or below it. The statistic decreases in the
/// variance, so the root is bracketed by doubling and found by bisection.
fn between_variance_at(estimates: &[(f64, f64)], level: f64) -> f64 {
    if generalized_q(estimates, 0.0).0 <= level {
        return 0.0;
    }
    let mut lower = 0.0;
    let mut upper =
        estimates.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max).max(f64::MIN_POSITIVE);
    for _ in 0..200 {
        if generalized_q(estimates, upper).0 <= level {
            break;
        }
        lower = upper;
        upper *= 2.0;
    }
    for _ in 0..80 {
        let middle = 0.5 * (lower + upper);
        if generalized_q(estimates, middle).0 > level {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    0.5 * (lower + upper)
}

/// Paule-Mandel pooling of `(estimate, variance)` pairs: the pooled mean, its variance and the
/// between-population variance at which the generalized Q statistic equals its expectation.
fn pool(estimates: &[(f64, f64)]) -> (f64, f64, f64) {
    let tau2 = between_variance_at(estimates, estimates.len() as f64 - 1.0);
    let (_, mean, mean_variance) = generalized_q(estimates, tau2);
    (mean, mean_variance, tau2)
}

/// One draw of the target's logit rate. `populations` are the validated populations pooled (the
/// target's own sample included when it has one, also passed as `own`): the between-population
/// variance is drawn from its Q-profile confidence distribution, the pooled mean from its
/// normal law at that variance, and the target's rate from `N(mean, variance)`, conditioned on
/// its own estimate when it has one.
fn draw_target_logit(
    populations: &[(f64, f64)],
    own: Option<(f64, f64)>,
    rng: &mut CausalRng,
) -> f64 {
    let degrees = populations.len() as f64 - 1.0;
    let level = antecedent_kernels::sample_gamma(0.5 * degrees, 0.5, rng);
    let tau2 = between_variance_at(populations, level);
    let (_, mean, mean_variance) = generalized_q(populations, tau2);
    let mu = mean + mean_variance.sqrt() * antecedent_kernels::standard_normal(rng);
    match own {
        None => mu + tau2.sqrt() * antecedent_kernels::standard_normal(rng),
        Some(_) if tau2 <= 0.0 => mu,
        Some((y, v)) => {
            let precision = 1.0 / tau2 + 1.0 / v;
            (mu / tau2 + y / v) / precision
                + antecedent_kernels::standard_normal(rng) / precision.sqrt()
        }
    }
}

/// Correct the target's recorded prevalence with sensitivity and specificity pooled across
/// differentially validated populations.
///
/// # Errors
/// Fewer than two sources, a source labelled as the target, a repeated source, an empty
/// validation margin, a target sample with no rows or more positives than rows, fewer than
/// 1000 draws, more than five percent of draws with a non-positive Youden index (the pooled
/// rates leave the target unidentified), admission or cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "one linear procedure: validate the request, pool the rates, draw the corrected prevalence and summarize the draws"
)]
pub fn transport_pooled_differential_prevalence(
    request: PooledDifferentialRequest<'_>,
    ctx: &ExecutionContext,
) -> Result<PooledDifferentialCorrection, EstimationError> {
    let mut labels = std::collections::BTreeSet::new();
    if request.sources.len() < 2 {
        return Err(refuse(
            antecedent_core::reason_code!("differential_error_not_pooled"),
            "measurement_pooled.too_few_sources",
            "pooling differential error needs at least two validated source populations. Next \
             attempt: validate the measurement in another population, or validate it in the \
             target",
        ));
    }
    for source in request.sources {
        if source.population == request.target_population || !labels.insert(&source.population) {
            return Err(invalid("source populations must be distinct and differ from the target"));
        }
    }
    let margins = |v: &PopulationValidation| {
        v.true_positive + v.false_negative > 0 && v.true_negative + v.false_positive > 0
    };
    if !request.sources.iter().all(margins) || !request.target_validation.is_none_or(margins) {
        return Err(invalid("every validation sample needs true positives and true negatives"));
    }
    if request.observed_total == 0 || request.observed_positive > request.observed_total {
        return Err(invalid("the target sample needs rows and no more positives than rows"));
    }
    if request.draws < 1000 {
        return Err(invalid("the propagation needs at least 1000 draws"));
    }
    let mut sensitivity_logits: Vec<(f64, f64)> = request
        .sources
        .iter()
        .map(|v| logit_with_variance(v.true_positive, v.false_negative))
        .collect();
    let mut specificity_logits: Vec<(f64, f64)> = request
        .sources
        .iter()
        .map(|v| logit_with_variance(v.true_negative, v.false_positive))
        .collect();
    let sensitivity_pool = pool(&sensitivity_logits);
    let specificity_pool = pool(&specificity_logits);
    let own_sensitivity =
        request.target_validation.map(|v| logit_with_variance(v.true_positive, v.false_negative));
    let own_specificity =
        request.target_validation.map(|v| logit_with_variance(v.true_negative, v.false_positive));
    // A validated target is one more population of the pool its rate is drawn from.
    sensitivity_logits.extend(own_sensitivity);
    specificity_logits.extend(own_specificity);
    let _lease = crate::util::admit_array::<f64>(ctx, request.draws, 3)?;
    let (x, n) = (request.observed_positive as f64, request.observed_total as f64);
    let mut draws = Vec::with_capacity(request.draws);
    let mut se_draws = Vec::with_capacity(request.draws);
    let mut sp_draws = Vec::with_capacity(request.draws);
    let (mut discarded, mut truncated) = (0_usize, 0_usize);
    for draw in 0..request.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Transport, POOLED_STREAM | draw as u64);
        let se = antecedent_kernels::sigmoid(draw_target_logit(
            &sensitivity_logits,
            own_sensitivity,
            &mut rng,
        ));
        let sp = antecedent_kernels::sigmoid(draw_target_logit(
            &specificity_logits,
            own_specificity,
            &mut rng,
        ));
        let a = antecedent_kernels::sample_gamma(x + 1.0, 1.0, &mut rng);
        let b = antecedent_kernels::sample_gamma(n - x + 1.0, 1.0, &mut rng);
        let recorded = a / (a + b);
        se_draws.push(se);
        sp_draws.push(sp);
        let youden = se + sp - 1.0;
        if youden <= 0.0 {
            discarded += 1;
            continue;
        }
        let corrected = (recorded + sp - 1.0) / youden;
        if !(0.0..=1.0).contains(&corrected) {
            truncated += 1;
        }
        draws.push(corrected.clamp(0.0, 1.0));
    }
    if discarded as f64 > MAX_DISCARDED_SHARE * request.draws as f64 {
        return Err(refuse(
            antecedent_core::reason_code!("differential_error_not_pooled"),
            "measurement_pooled.unidentified",
            &format!(
                "{discarded} of {} draws of the pooled error rates leave the Youden index \
                 non-positive, so the target prevalence is not identified. Next attempt: \
                 validate the measurement in the target population",
                request.draws
            ),
        ));
    }
    let mut sorted = draws.clone();
    sorted.sort_by(f64::total_cmp);
    let interval = antecedent_stats::equal_tail_interval_sorted(
        &sorted,
        0.95,
        antecedent_stats::QuantileRule::ExchangeableRank,
    );
    let rate = |pooled: (f64, f64, f64), samples: &[f64]| PooledRate {
        mean_logit: pooled.0,
        mean_variance: pooled.1,
        between_population_variance: pooled.2,
        target_rate: crate::util::mean(samples),
        target_rate_sd: crate::util::sample_std(samples),
    };
    Ok(PooledDifferentialCorrection {
        target_population: request.target_population.to_owned(),
        prevalence: crate::util::mean(&draws),
        interval,
        sensitivity: rate(sensitivity_pool, &se_draws),
        specificity: rate(specificity_pool, &sp_draws),
        discarded_draws: discarded,
        truncated_draws: truncated,
        draws,
        calibration: "not_measured",
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "validation counts are small non-negative rounded products"
)]
mod tests {
    use super::*;

    fn validation(population: &str, se: f64, sp: f64, size: u64) -> PopulationValidation {
        let positives = size / 2;
        let negatives = size - positives;
        let true_positive = (se * positives as f64).round() as u64;
        let true_negative = (sp * negatives as f64).round() as u64;
        PopulationValidation {
            population: population.to_owned(),
            true_positive,
            false_negative: positives - true_positive,
            true_negative,
            false_positive: negatives - true_negative,
        }
    }

    fn request<'a>(
        sources: &'a [PopulationValidation],
        target: Option<&'a PopulationValidation>,
    ) -> PooledDifferentialRequest<'a> {
        // Target truth: prevalence 0.3 recorded through se = 0.85, sp = 0.9.
        let recorded = 0.3 * 0.85 + 0.7 * 0.1;
        PooledDifferentialRequest {
            sources,
            target_validation: target,
            target_population: "target",
            observed_positive: (recorded * 4000.0_f64).round() as u64,
            observed_total: 4000,
            draws: 4000,
        }
    }

    #[test]
    fn heterogeneous_sources_widen_the_interval_and_a_target_sample_narrows_it() {
        let ctx = ExecutionContext::for_tests(7);
        let invariant = [
            validation("a", 0.85, 0.9, 2000),
            validation("b", 0.85, 0.9, 2000),
            validation("c", 0.85, 0.9, 2000),
        ];
        let heterogeneous = [
            validation("a", 0.75, 0.94, 2000),
            validation("b", 0.85, 0.9, 2000),
            validation("c", 0.93, 0.84, 2000),
        ];
        let tight =
            transport_pooled_differential_prevalence(request(&invariant, None), &ctx).unwrap();
        let wide =
            transport_pooled_differential_prevalence(request(&heterogeneous, None), &ctx).unwrap();
        assert!((tight.prevalence - 0.3).abs() < 0.03, "{}", tight.prevalence);
        assert!(tight.sensitivity.between_population_variance < 0.05);
        assert!(wide.sensitivity.between_population_variance > 0.05);
        let width = |fit: &PooledDifferentialCorrection| fit.interval.1 - fit.interval.0;
        assert!(width(&wide) > width(&tight));
        let own = validation("target", 0.85, 0.9, 4000);
        let informed =
            transport_pooled_differential_prevalence(request(&heterogeneous, Some(&own)), &ctx)
                .unwrap();
        assert!(width(&informed) < width(&wide));
        assert!(informed.interval.0 < 0.3 && 0.3 < informed.interval.1);
        assert!(
            transport_pooled_differential_prevalence(request(&invariant[..1], None), &ctx).is_err()
        );
    }
}

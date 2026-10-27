//! Bayesian piecewise-exponential survival for a randomized two-arm trial.
//!
//! The likelihood is stated, not implied. Within each arm the hazard is constant on every
//! declared interval `(c_{k-1}, c_k]`, subjects are independent, censoring and delayed
//! entry are non-informative about the event time given the arm, and the arms are
//! randomized. A subject then contributes `h_k^{d} exp(-h_k e)` to interval `k`, where `d`
//! is its event indicator in that interval and `e` is its exact time at risk inside it
//! (entry truncates the exposure from the left, an early exit truncates it from the right).
//! Every interval hazard has an independent `Gamma(shape, rate)` prior, so the posterior is
//! `Gamma(shape + events_k, rate + exposure_k)` in closed form. The prior is part of the
//! answer: the interval cut points and the Gamma pair are declared, never selected here.
//!
//! The target is the same one the randomized product-limit summary reports: the
//! treated-minus-control survival at the horizon (the last cut) and the difference in
//! restricted mean survival time over `[0, horizon]`. Under the piecewise-constant hazard
//! model these are exact functions of the hazards, so the posterior is a posterior for that
//! target and not a relabelled interval. The model, unlike the product limit, restricts the
//! hazard to be flat inside each interval: a coarse partition changes the target's model.
//!
//! The posterior means have closed forms (`E[exp(-h l)] = (rate / (rate + l))^shape`); the
//! credible intervals and the probability that the treated arm survives longer come from
//! posterior draws of the hazards. Only the all-cause survival endpoint is supported:
//! cumulative incidence needs cause-specific hazards with a competing-risk likelihood.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;
use antecedent_kernels::sample_gamma;
use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::error::EstimationError;
use crate::survival::SurvivalRows;

/// Fewest posterior draws accepted for the credible intervals.
pub const MIN_HAZARD_POSTERIOR_DRAWS: usize = 200;
/// Most posterior draws one request may ask for.
pub const MAX_HAZARD_POSTERIOR_DRAWS: usize = 200_000;
/// Reference hazard interval count; not enforced.
pub const MAX_HAZARD_INTERVALS: usize = 64;

/// Independent gamma prior on every interval hazard, in events per unit time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HazardGammaPrior {
    /// Shape: the prior event count. Finite and positive.
    pub shape: f64,
    /// Rate: the prior time at risk, in the unit of the durations. Finite and positive.
    pub rate: f64,
}

/// Gamma posterior of one arm's hazard on one interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalHazardPosterior {
    /// Interval length.
    pub length: f64,
    /// Observed events in the interval.
    pub events: usize,
    /// Exact time at risk in the interval.
    pub exposure: f64,
    /// Posterior gamma shape.
    pub shape: f64,
    /// Posterior gamma rate.
    pub rate: f64,
}

/// Posterior of the randomized treated-minus-control survival contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct PiecewiseSurvivalPosterior {
    /// Interval right endpoints; the last one is the horizon.
    pub cuts: Vec<f64>,
    /// Control-arm interval posteriors.
    pub control: Vec<IntervalHazardPosterior>,
    /// Treated-arm interval posteriors.
    pub treated: Vec<IntervalHazardPosterior>,
    /// Closed-form posterior mean of control survival at the horizon.
    pub control_survival_mean: f64,
    /// Closed-form posterior mean of treated survival at the horizon.
    pub treated_survival_mean: f64,
    /// Treated minus control posterior mean survival at the horizon.
    pub survival_difference_mean: f64,
    /// Closed-form posterior mean restricted mean survival time, control.
    pub control_rmst_mean: f64,
    /// Closed-form posterior mean restricted mean survival time, treated.
    pub treated_rmst_mean: f64,
    /// Treated minus control posterior mean restricted mean survival time.
    pub rmst_difference_mean: f64,
    /// Equal-tailed 95% credible interval of the survival difference.
    pub survival_difference_interval_95: [f64; 2],
    /// Equal-tailed 95% credible interval of the restricted-mean difference.
    pub rmst_difference_interval_95: [f64; 2],
    /// Posterior probability that treated survival at the horizon exceeds control.
    pub probability_treated_survives_longer: f64,
    /// Posterior draws behind the intervals and the probability.
    pub draws: usize,
    /// Draws of the treated-minus-control survival at the horizon, in draw order; draw `k`
    /// of this column and of [`Self::rmst_difference_draws`] come from the same hazards.
    pub survival_difference_draws: Vec<f64>,
    /// Draws of the treated-minus-control restricted mean survival time, in draw order.
    pub rmst_difference_draws: Vec<f64>,
}

/// Posterior of the treated-minus-control survival contrast under a piecewise-exponential
/// likelihood with independent gamma hazard priors.
///
/// `cuts` are strictly increasing positive interval right endpoints; the first interval
/// starts at zero and the last cut is the horizon. Events are coded zero or one.
///
/// # Errors
///
/// Misaligned or non-finite rows, event codes other than zero and one (competing causes are
/// not supported by this likelihood), an invalid partition, prior or draw count, or an arm
/// with no time at risk in some interval, where the posterior would be the prior alone.
pub fn randomized_piecewise_survival_posterior(
    rows: &SurvivalRows<'_>,
    cuts: &[f64],
    prior: HazardGammaPrior,
    draws: usize,
    seed: u64,
) -> Result<PiecewiseSurvivalPosterior, EstimationError> {
    validate(rows, cuts, prior, draws)?;
    let control = arm_posteriors(rows, cuts, prior, false)?;
    let treated = arm_posteriors(rows, cuts, prior, true)?;
    let control_survival_mean = survival_mean(&control);
    let treated_survival_mean = survival_mean(&treated);
    let control_rmst_mean = rmst_mean(&control);
    let treated_rmst_mean = rmst_mean(&treated);

    let mut rng = CausalRng::from_seed(seed);
    let _admitted = crate::util::admit_bound_array::<f64>(draws, 2)?;
    let mut survival_difference = Vec::with_capacity(draws);
    let mut rmst_difference = Vec::with_capacity(draws);
    for _ in 0..draws {
        let (control_survival, control_rmst) = draw_curve(&control, &mut rng);
        let (treated_survival, treated_rmst) = draw_curve(&treated, &mut rng);
        survival_difference.push(treated_survival - control_survival);
        rmst_difference.push(treated_rmst - control_rmst);
    }
    if survival_difference.iter().chain(&rmst_difference).any(|value| !value.is_finite()) {
        return Err(EstimationError::data_msg("a posterior hazard draw was not finite"));
    }
    let higher = survival_difference.iter().filter(|value| **value > 0.0).count();
    let probability_treated_survives_longer = higher as f64 / draws as f64;
    let survival_difference_draws = survival_difference.clone();
    let rmst_difference_draws = rmst_difference.clone();
    Ok(PiecewiseSurvivalPosterior {
        cuts: cuts.to_vec(),
        control,
        treated,
        control_survival_mean,
        treated_survival_mean,
        survival_difference_mean: treated_survival_mean - control_survival_mean,
        control_rmst_mean,
        treated_rmst_mean,
        rmst_difference_mean: treated_rmst_mean - control_rmst_mean,
        survival_difference_interval_95: interval_95(&mut survival_difference),
        rmst_difference_interval_95: interval_95(&mut rmst_difference),
        probability_treated_survives_longer,
        draws,
        survival_difference_draws,
        rmst_difference_draws,
    })
}

fn interval_95(values: &mut [f64]) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(values, 0.95, QuantileRule::ExchangeableRank);
    [lower, upper]
}

fn validate(
    rows: &SurvivalRows<'_>,
    cuts: &[f64],
    prior: HazardGammaPrior,
    draws: usize,
) -> Result<(), EstimationError> {
    let SurvivalRows { duration, event_code, treated, delayed_entry } = *rows;
    let n = duration.len();
    if n == 0
        || event_code.len() != n
        || treated.len() != n
        || delayed_entry.is_some_and(|entry| entry.len() != n)
    {
        return Err(EstimationError::data_msg(
            "duration, event, treatment, and entry must have equal nonzero length",
        ));
    }
    if duration.iter().any(|time| !time.is_finite() || *time <= 0.0) {
        return Err(EstimationError::data_msg("durations must be finite and positive"));
    }
    if event_code.iter().any(|code| *code != 0 && *code != 1) {
        return Err(EstimationError::unsupported(
            "a piecewise-exponential survival posterior needs events coded zero or one; \
             competing causes need cause-specific hazards",
        ));
    }
    if let Some(entry) = delayed_entry {
        if entry
            .iter()
            .zip(duration)
            .any(|(left, right)| !left.is_finite() || *left < 0.0 || left >= right)
        {
            return Err(EstimationError::data_msg(
                "entry times must be finite, nonnegative, and earlier than exit",
            ));
        }
    }
    if cuts.is_empty()
        || cuts.iter().any(|cut| !cut.is_finite() || *cut <= 0.0)
        || cuts.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(EstimationError::data_msg(
            "hazard cuts must be at least one strictly increasing positive finite times",
        ));
    }
    if !prior.shape.is_finite()
        || prior.shape <= 0.0
        || !prior.rate.is_finite()
        || prior.rate <= 0.0
    {
        return Err(EstimationError::data_msg(
            "hazard prior shape and rate must be finite and positive",
        ));
    }
    if !(MIN_HAZARD_POSTERIOR_DRAWS..=MAX_HAZARD_POSTERIOR_DRAWS).contains(&draws) {
        return Err(EstimationError::data_msg(
            "hazard posterior draws must be between 200 and 200000",
        ));
    }
    Ok(())
}

/// Events and exact exposure of one arm on every interval, then the gamma update.
fn arm_posteriors(
    rows: &SurvivalRows<'_>,
    cuts: &[f64],
    prior: HazardGammaPrior,
    arm: bool,
) -> Result<Vec<IntervalHazardPosterior>, EstimationError> {
    let SurvivalRows { duration, event_code, treated, delayed_entry } = *rows;
    let members: Vec<usize> =
        treated.iter().enumerate().filter_map(|(i, value)| (*value == arm).then_some(i)).collect();
    if members.is_empty() {
        return Err(EstimationError::data_msg("both randomized arms require observed units"));
    }
    let mut lower = 0.0;
    let mut out = Vec::with_capacity(cuts.len());
    for &upper in cuts {
        let mut events = 0_usize;
        let mut exposure = 0.0;
        for &i in &members {
            let entry = delayed_entry.and_then(|values| values.get(i).copied()).unwrap_or(0.0);
            exposure += (duration[i].min(upper) - entry.max(lower)).max(0.0);
            if event_code[i] == 1 && duration[i] > lower && duration[i] <= upper {
                events += 1;
            }
        }
        if exposure <= 0.0 {
            return Err(EstimationError::data_msg(
                "an arm has no time at risk in a hazard interval; shorten the horizon or merge intervals",
            ));
        }
        let shape = prior.shape + events as f64;
        out.push(IntervalHazardPosterior {
            length: upper - lower,
            events,
            exposure,
            shape,
            rate: prior.rate + exposure,
        });
        lower = upper;
    }
    Ok(out)
}

/// `E[exp(-h l)] = (rate / (rate + l))^shape` for `h ~ Gamma(shape, rate)`.
fn stay_factor(posterior: &IntervalHazardPosterior) -> f64 {
    (posterior.shape * (posterior.rate / (posterior.rate + posterior.length)).ln()).exp()
}

fn survival_mean(arm: &[IntervalHazardPosterior]) -> f64 {
    arm.iter().map(stay_factor).product()
}

/// `E[(1 - exp(-h l)) / h] = ∫_0^l (rate / (rate + s))^shape ds`.
#[allow(
    clippy::float_cmp,
    reason = "exactly shape 1 is the exponential-mixture branch of the integral"
)]
fn interval_area(posterior: &IntervalHazardPosterior) -> f64 {
    let IntervalHazardPosterior { length, shape, rate, .. } = *posterior;
    // `ln(rate / (rate + length))`, exact for a small `length / rate`.
    let ratio = -(length / rate).ln_1p();
    let excess = shape - 1.0;
    if excess == 0.0 {
        -rate * ratio
    } else {
        // `−expm1` keeps the cancellation-free value arbitrarily close to `shape = 1`.
        -rate * (excess * ratio).exp_m1() / excess
    }
}

/// Posterior mean restricted mean survival time: intervals are independent, so the mean of
/// the prior-interval survival times the in-interval area factorizes exactly.
fn rmst_mean(arm: &[IntervalHazardPosterior]) -> f64 {
    let mut carried = 1.0;
    let mut total = 0.0;
    for posterior in arm {
        total += carried * interval_area(posterior);
        carried *= stay_factor(posterior);
    }
    total
}

/// One posterior draw of the survival at the horizon and the restricted mean.
fn draw_curve(arm: &[IntervalHazardPosterior], rng: &mut CausalRng) -> (f64, f64) {
    let mut survival = 1.0;
    let mut rmst = 0.0;
    for posterior in arm {
        let hazard = sample_gamma(posterior.shape, posterior.rate, rng);
        let exponent = hazard * posterior.length;
        let area = if hazard > 1e-300 { -(-exponent).exp_m1() / hazard } else { posterior.length };
        rmst += survival * area;
        survival *= (-exponent).exp();
    }
    (survival, rmst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survival::{SurvivalEndpoint, randomized_survival_summary};

    const CUTS: [f64; 3] = [1.0, 2.0, 3.0];
    const UNIFORM: HazardGammaPrior = HazardGammaPrior { shape: 1.0, rate: 1.0 };

    #[test]
    fn interval_area_is_continuous_through_shape_one() {
        let area = |shape: f64| {
            interval_area(&IntervalHazardPosterior {
                length: 2.0,
                events: 0,
                exposure: 0.0,
                shape,
                rate: 3.0,
            })
        };
        let exact = 3.0 * (2.0_f64 / 3.0).ln_1p();
        assert!((area(1.0) - exact).abs() < 1e-15);
        for offset in [1e-12, -1e-12, 1e-8, -1e-8] {
            assert!((area(1.0 + offset) - exact).abs() < offset.abs(), "{offset}");
        }
        // Closed form away from shape 1: rate * (1 - (rate/(rate+len))^(s-1)) / (s-1).
        let direct = 3.0 * (1.0 - (3.0_f64 / 5.0).powf(1.5)) / 1.5;
        assert!((area(2.5) - direct).abs() < 1e-14);
    }

    fn small() -> (Vec<f64>, Vec<i64>, Vec<bool>) {
        let mut duration = vec![0.5, 1.2, 2.5, 3.5, 1.8, 0.9];
        let mut event = vec![1, 0, 1, 0, 1, 1];
        let mut treated = vec![false; 6];
        duration.extend([0.7, 1.5, 2.2, 3.2, 3.8, 1.1]);
        event.extend([0, 1, 0, 1, 0, 0]);
        treated.extend([true; 6]);
        (duration, event, treated)
    }

    fn rows<'a>(
        duration: &'a [f64],
        event: &'a [i64],
        treated: &'a [bool],
        entry: Option<&'a [f64]>,
    ) -> SurvivalRows<'a> {
        SurvivalRows { duration, event_code: event, treated, delayed_entry: entry }
    }

    fn near(left: f64, right: f64, tolerance: f64) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn conjugate_updates_and_closed_form_means_match_the_reference() {
        // Reference values: exposure and event counts by hand, the posterior means from the
        // gamma moment generating function, the restricted mean by numerical quadrature.
        let (duration, event, treated) = small();
        let posterior = randomized_piecewise_survival_posterior(
            &rows(&duration, &event, &treated, None),
            &CUTS,
            UNIFORM,
            20_000,
            7,
        )
        .unwrap();
        let control = &posterior.control;
        assert_eq!(control.iter().map(|p| p.events).collect::<Vec<_>>(), vec![2, 1, 1],);
        near(control[0].exposure, 5.4, 1e-12);
        near(control[1].exposure, 3.0, 1e-12);
        near(control[2].exposure, 1.5, 1e-12);
        near(posterior.treated[0].exposure, 5.7, 1e-12);
        assert_eq!(posterior.treated.iter().map(|p| p.events).collect::<Vec<_>>(), vec![0, 1, 0]);
        near(posterior.control_survival_mean, 0.211_236_355_241_364_2, 1e-12);
        near(posterior.treated_survival_mean, 0.447_326_240_329_155_6, 1e-12);
        near(posterior.survival_difference_mean, 0.236_089_885_087_791_4, 1e-12);
        near(posterior.control_rmst_mean, 1.619_688_017_350_5, 1e-9);
        near(posterior.treated_rmst_mean, 2.157_706_276_225_9, 1e-9);
        near(posterior.rmst_difference_mean, 0.538_018_258_905_1, 1e-9);
    }

    #[test]
    fn draw_based_summaries_agree_with_a_large_independent_simulation() {
        // The reference interval comes from two million draws in numpy, not from this code.
        let (duration, event, treated) = small();
        let posterior = randomized_piecewise_survival_posterior(
            &rows(&duration, &event, &treated, None),
            &CUTS,
            UNIFORM,
            40_000,
            11,
        )
        .unwrap();
        near(posterior.survival_difference_interval_95[0], -0.191_49, 0.02);
        near(posterior.survival_difference_interval_95[1], 0.643_14, 0.02);
        near(posterior.rmst_difference_interval_95[0], -0.401_23, 0.04);
        near(posterior.rmst_difference_interval_95[1], 1.419_53, 0.04);
        near(posterior.probability_treated_survives_longer, 0.862_216, 0.01);
        // The mean of the draws agrees with the exact closed-form mean.
        assert!(posterior.survival_difference_interval_95[0] < posterior.survival_difference_mean);
        assert!(posterior.survival_difference_mean < posterior.survival_difference_interval_95[1]);
    }

    #[test]
    fn delayed_entry_truncates_the_exposure_exactly() {
        // Entry times 0, .3, 1.5, 0, .2, .1 remove the unobserved left part of each
        // control exposure. The reference means are the same closed form on those counts.
        let (duration, event, treated) = small();
        let mut entry = vec![0.0; 12];
        entry[..6].copy_from_slice(&[0.0, 0.3, 1.5, 0.0, 0.2, 0.1]);
        let posterior = randomized_piecewise_survival_posterior(
            &rows(&duration, &event, &treated, Some(&entry)),
            &CUTS,
            UNIFORM,
            200,
            1,
        )
        .unwrap();
        near(posterior.control[0].exposure, 3.8, 1e-12);
        near(posterior.control[1].exposure, 2.5, 1e-12);
        near(posterior.control[2].exposure, 1.5, 1e-12);
        near(posterior.control_survival_mean, 0.174_942_255_384_3, 1e-9);
        near(posterior.control_rmst_mean, 1.442_016_209_493_5, 1e-9);
    }

    #[test]
    fn flat_prior_posterior_tracks_the_product_limit_on_exponential_data() {
        // Deterministic exponential quantiles with hazard 0.4 and administrative censoring
        // at 3, so the piecewise-constant model is correct. A weak prior must land near the
        // existing Kaplan-Meier summary, which is the independent reference here.
        let n = 400;
        let mut duration = Vec::new();
        let mut event = Vec::new();
        let mut treated = Vec::new();
        for arm in [false, true] {
            let hazard = if arm { 0.25 } else { 0.4 };
            for i in 0..n {
                let u = (f64::from(i) + 0.5) / f64::from(n);
                let time = -(1.0 - u).ln() / hazard;
                duration.push(time.min(3.0));
                event.push(i64::from(time <= 3.0));
                treated.push(arm);
            }
        }
        let weak = HazardGammaPrior { shape: 1e-3, rate: 1e-3 };
        let cuts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let posterior = randomized_piecewise_survival_posterior(
            &rows(&duration, &event, &treated, None),
            &cuts,
            weak,
            400,
            3,
        )
        .unwrap();
        let km = randomized_survival_summary(
            &rows(&duration, &event, &treated, None),
            3.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        let (control_km, treated_km) =
            (km.control.last().copied().unwrap(), km.treated.last().copied().unwrap());
        near(posterior.control_survival_mean, control_km, 0.01);
        near(posterior.treated_survival_mean, treated_km, 0.01);
        near(posterior.control_rmst_mean, km.rmst_control.unwrap(), 0.02);
        near(posterior.treated_rmst_mean, km.rmst_treated.unwrap(), 0.02);
        assert!(posterior.survival_difference_interval_95[0] > 0.0);
        assert!(posterior.probability_treated_survives_longer > 0.99);
    }

    #[test]
    fn unsupported_likelihood_requests_are_refused() {
        let (duration, mut event, treated) = small();
        let data = rows(&duration, &event, &treated, None);
        assert!(randomized_piecewise_survival_posterior(&data, &[], UNIFORM, 200, 1).is_err());
        assert!(
            randomized_piecewise_survival_posterior(&data, &[2.0, 1.0], UNIFORM, 200, 1).is_err()
        );
        assert!(randomized_piecewise_survival_posterior(&data, &CUTS, UNIFORM, 10, 1).is_err());
        let improper = HazardGammaPrior { shape: 0.0, rate: 1.0 };
        assert!(randomized_piecewise_survival_posterior(&data, &CUTS, improper, 200, 1).is_err());
        // A horizon past the longest follow-up leaves an interval with no time at risk.
        let long = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(randomized_piecewise_survival_posterior(&data, &long, UNIFORM, 200, 1).is_err());
        event[0] = 2;
        let competing = rows(&duration, &event, &treated, None);
        let refused = randomized_piecewise_survival_posterior(&competing, &CUTS, UNIFORM, 200, 1)
            .unwrap_err();
        assert!(refused.to_string().contains("cause-specific"));
    }
}

//! Observational excursion effects: the micro-randomized-trial target with an assignment law
//! that is estimated, not recorded.
//!
//! # Target and identification
//!
//! The target is the one of the sequentially randomized designs: at each decision period `t`
//! the contrast of assigning treatment against control *at `t`* on the unit's outcome `lag`
//! periods later, every earlier and later assignment following the observed process. What
//! changes is the identification. No assignment probability is recorded, so the contrast is
//! identified by sequential exchangeability given the measured history (the decision-time
//! covariates the analyst names), positivity of both arms given that history, consistency, and
//! no interference between units. The history is declared, not checked, and the treatment
//! model is a pooled logistic regression of the decision on the history columns.
//!
//! # Estimator
//!
//! The propensity `p(h_t) = P(A_t = 1 | h_t)` is the pooled logistic fit over every decision
//! row, and the contrast is the Horvitz-Thompson ratio of the recorded-law designs with the
//! fitted probability in place of the recorded one: `theta = sum_g S_g / sum_g n_g` over
//! independent units `g`, with `S_g = sum_p A Y / p-hat - (1 - A) Y / (1 - p-hat)`. A fitted
//! probability under the positivity floor refuses the fit.
//!
//! # Uncertainty
//!
//! The nuisance fit is part of the procedure, so both intervals refit it. The frequentist
//! interval is the percentile interval of a unit bootstrap that refits the propensity model in
//! every replicate (at least 99 replicates; more than one percent failed replicates refuse).
//! The Bayesian posterior is the weighted-likelihood bootstrap: each draw takes Dirichlet
//! weights over units, refits the propensity model under those weights (a Gaussian coefficient
//! prior of the declared scale keeps it proper) and recomputes the weighted ratio, so
//! uncertainty about the assignment law flows into the posterior. Neither is calibrated: no
//! coverage of these intervals has been measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use super::linalg::fit_logistic;
use super::{LaggedPairs, MIN_CLUSTERS, PosteriorSummary, dense_labels, invalid};
use crate::error::EstimationError;
use crate::util::refuse;

/// Stream key of the unit bootstrap.
const BOOTSTRAP_STREAM: u64 = 0x7E5D_0B5E_0000_0000;
/// Fewest bootstrap replicates of the whole-procedure interval.
pub const MIN_REPLICATES: usize = 99;
/// Largest share of bootstrap replicates allowed to fail before the interval is refused.
const MAX_FAILED_SHARE: f64 = 0.01;

/// Coordinate names of an observational excursion.
pub const OBSERVATIONAL_COORDINATES: [&str; 3] =
    ["excursion_effect", "treated_mean", "control_mean"];

/// Inputs of an observational excursion; every array is row-aligned with the table.
#[derive(Clone, Copy, Debug)]
pub struct ObservationalRows<'a> {
    /// Decision rows paired with their outcome rows.
    pub pairs: &'a LaggedPairs,
    /// Outcome of every row.
    pub outcome: &'a [f64],
    /// Realized binary treatment of every row.
    pub treated: &'a [bool],
    /// History columns the treatment model reads, each one value per row.
    pub history: &'a [Vec<f64>],
    /// Independent unit of every row.
    pub unit: &'a [usize],
}

/// A fitted observational excursion.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalFit {
    /// `[excursion effect, treated mean, control mean]` at the full-data propensity fit.
    pub estimate: [f64; 3],
    /// Bootstrap standard deviation of each coordinate, under frequentist inference.
    pub standard_error: Option<[f64; 3]>,
    /// Percentile interval of each coordinate, under frequentist inference.
    pub interval: Option<[(f64, f64); 3]>,
    /// Replicates that succeeded, and those that failed.
    pub replicates: (usize, usize),
    /// Posterior, under Bayesian inference.
    pub posterior: Option<PosteriorSummary>,
    /// Smallest fitted probability of an arm among the paired decisions.
    pub minimum_probability: f64,
    /// Propensity coefficients at the full-data fit: intercept, then one per history column.
    pub propensity: Vec<f64>,
    /// Paired decisions.
    pub pairs: usize,
    /// Independent units.
    pub clusters: usize,
}

struct Prepared {
    /// Row-major propensity design over every row: intercept, then the history columns.
    design: Vec<f64>,
    width: usize,
    treated_indicator: Vec<f64>,
    /// Dense unit of every row.
    unit_of_row: Vec<usize>,
    clusters: usize,
    decision: Vec<usize>,
    pair_outcome: Vec<f64>,
    pair_unit: Vec<usize>,
}

fn prepare(rows: ObservationalRows<'_>) -> Result<Prepared, EstimationError> {
    let n = rows.outcome.len();
    if rows.treated.len() != n
        || rows.unit.len() != n
        || rows.history.iter().any(|column| column.len() != n)
    {
        return Err(invalid("design arrays must be aligned with the table rows"));
    }
    if rows.history.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid("every history value must be finite"));
    }
    let width = rows.history.len() + 1;
    let mut design = Vec::with_capacity(n * width);
    for row in 0..n {
        design.push(1.0);
        for column in rows.history {
            design.push(column[row]);
        }
    }
    let (unit_of_row, clusters) = dense_labels(rows.unit);
    if clusters < MIN_CLUSTERS {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "temporal_design.too_few_clusters",
            "the design needs at least two independent units or clusters",
        ));
    }
    let mut pair_outcome = Vec::with_capacity(rows.pairs.decision.len());
    for &o in &rows.pairs.outcome {
        let y = rows.outcome[o];
        if !y.is_finite() {
            return Err(invalid("every paired outcome must be finite"));
        }
        pair_outcome.push(y);
    }
    let pair_unit = rows.pairs.decision.iter().map(|&t| unit_of_row[t]).collect();
    Ok(Prepared {
        design,
        width,
        treated_indicator: rows.treated.iter().map(|a| f64::from(u8::from(*a))).collect(),
        unit_of_row,
        clusters,
        decision: rows.pairs.decision.clone(),
        pair_outcome,
        pair_unit,
    })
}

struct Evaluation {
    values: [f64; 3],
    minimum_probability: f64,
    coefficients: Vec<f64>,
}

/// Fit the propensity under unit weights and evaluate the weighted Horvitz-Thompson ratio.
///
/// `ridge` is the coefficient prior precision (zero for the maximum-likelihood fit).
fn evaluate(
    prepared: &Prepared,
    unit_weights: Option<&[f64]>,
    ridge: f64,
    floor: f64,
    treated: &[bool],
) -> Result<Evaluation, EstimationError> {
    let row_weights: Option<Vec<f64>> = unit_weights
        .map(|weights| prepared.unit_of_row.iter().map(|&unit| weights[unit]).collect());
    let fit = fit_logistic(
        &prepared.design,
        prepared.width,
        &prepared.treated_indicator,
        row_weights.as_deref(),
        ridge,
    )?;
    let (mut treated_sum, mut control_sum, mut denominator) = (0.0, 0.0, 0.0);
    let (mut treated_pairs, mut control_pairs) = (0_usize, 0_usize);
    let mut minimum = 1.0_f64;
    for ((&t, &y), &unit) in
        prepared.decision.iter().zip(&prepared.pair_outcome).zip(&prepared.pair_unit)
    {
        let w = unit_weights.map_or(1.0, |weights| weights[unit]);
        // A unit absent from this resample or carrying no posterior weight takes no part in
        // the fit or the ratio, so its fitted probability is not held to the floor.
        if w.to_bits() == 0.0_f64.to_bits() {
            continue;
        }
        let features = &prepared.design[t * prepared.width..(t + 1) * prepared.width];
        let eta: f64 = features.iter().zip(&fit.coefficients).map(|(a, b)| a * b).sum();
        let p = antecedent_kernels::sigmoid(eta);
        let arm = p.min(1.0 - p);
        if arm.is_nan() || arm < floor {
            return Err(refuse(
                antecedent_core::reason_code!("positivity_violated"),
                "temporal_design.positivity",
                &format!(
                    "a fitted assignment probability {arm} is below the declared floor {floor}. \
                     Next attempt: history columns under which both arms keep at least the \
                     floor, a lower declared floor, or a design with recorded probabilities"
                ),
            ));
        }
        minimum = minimum.min(if treated[t] { p } else { 1.0 - p });
        denominator += w;
        if treated[t] {
            treated_sum += w * y / p;
            treated_pairs += 1;
        } else {
            control_sum += w * y / (1.0 - p);
            control_pairs += 1;
        }
    }
    if treated_pairs == 0 || control_pairs == 0 || denominator.is_nan() || denominator <= 0.0 {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "temporal_design.arm_not_populated",
            "an arm has no paired decision, so its mean is not estimable",
        ));
    }
    let treated_mean = treated_sum / denominator;
    let control_mean = control_sum / denominator;
    let values = [treated_mean - control_mean, treated_mean, control_mean];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg("the observational excursion is not finite"));
    }
    Ok(Evaluation { values, minimum_probability: minimum, coefficients: fit.coefficients })
}

/// Fit the observational excursion.
///
/// `posterior_draws` selects the weighted-likelihood bootstrap with a Gaussian coefficient
/// prior of scale `prior_scale`; otherwise `replicates` whole-procedure unit-bootstrap
/// replicates give the percentile interval and the propensity is the maximum-likelihood fit.
///
/// # Errors
/// Misaligned or non-finite inputs, fewer than two units, a fitted probability below the
/// floor, an arm with no paired decision, a propensity fit that separates, fewer than
/// [`MIN_REPLICATES`] replicates, more than one percent failed replicates, admission or
/// cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "one fit: the full-data estimate, then the bootstrap or posterior loop"
)]
pub fn fit_observational_excursion(
    rows: ObservationalRows<'_>,
    floor: f64,
    prior_scale: f64,
    replicates: usize,
    posterior_draws: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<ObservationalFit, EstimationError> {
    if !(floor.is_finite() && floor > 0.0 && floor <= 0.5) {
        return Err(invalid("the positivity floor must be in (0, 0.5]"));
    }
    let prepared = prepare(rows)?;
    let treated = rows.treated;
    let pairs = prepared.decision.len();
    if let Some(draws) = posterior_draws {
        if !(prior_scale.is_finite() && prior_scale > 0.0) {
            return Err(invalid("the prior scale must be positive"));
        }
        let ridge = 1.0 / (prior_scale * prior_scale);
        let point = evaluate(&prepared, None, ridge, floor, treated)?;
        let posterior =
            super::posterior_by_cluster_weights(prepared.clusters, 3, draws, ctx, |weights| {
                evaluate(&prepared, Some(weights), ridge, floor, treated)
                    .map(|evaluation| evaluation.values.to_vec())
            })
            .map(|mut summary| {
                summary.method = "weighted_likelihood_bootstrap_refitting_assignment_model";
                summary
            })?;
        return Ok(ObservationalFit {
            estimate: point.values,
            standard_error: None,
            interval: None,
            replicates: (0, 0),
            posterior: Some(posterior),
            minimum_probability: point.minimum_probability,
            propensity: point.coefficients,
            pairs,
            clusters: prepared.clusters,
        });
    }
    if replicates < MIN_REPLICATES {
        return Err(refuse(
            antecedent_core::reason_code!("required_option_missing"),
            "temporal_design.bootstrap_replicates",
            "the observational excursion interval refits the assignment model in every \
             unit-bootstrap replicate and needs at least 99 replicates. Next attempt: set \
             bootstrap_replicates to 99 or more",
        ));
    }
    let point = evaluate(&prepared, None, 0.0, floor, treated)?;
    let g_count = prepared.clusters;
    let _lease = crate::util::admit_array::<f64>(ctx, replicates, 3 + g_count)?;
    let mut columns: [Vec<f64>; 3] = [
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
    ];
    let mut counts = vec![0.0; g_count];
    let mut failed = 0_usize;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, BOOTSTRAP_STREAM | replicate as u64);
        counts.iter_mut().for_each(|value| *value = 0.0);
        for _ in 0..g_count {
            counts[antecedent_kernels::unbiased_index(&mut rng, g_count)] += 1.0;
        }
        match evaluate(&prepared, Some(&counts), 0.0, floor, treated) {
            Ok(evaluation) => {
                for (column, value) in columns.iter_mut().zip(evaluation.values) {
                    column.push(value);
                }
            }
            Err(_) => failed += 1,
        }
    }
    if failed as f64 > MAX_FAILED_SHARE * replicates as f64 {
        return Err(refuse(
            antecedent_core::reason_code!("positivity_violated"),
            "temporal_design.bootstrap_failures",
            &format!(
                "{failed} of {replicates} unit-bootstrap replicates could not refit the \
                 assignment model (positivity, separation or an unpopulated arm). Next \
                 attempt: more units, fewer or better-scaled history columns, or a lower floor"
            ),
        ));
    }
    let mut standard_error = [0.0; 3];
    let mut interval = [(0.0, 0.0); 3];
    for (j, column) in columns.iter_mut().enumerate() {
        standard_error[j] = crate::util::sample_std(column);
        column.sort_by(f64::total_cmp);
        interval[j] = antecedent_stats::equal_tail_interval_sorted(
            column,
            super::NOMINAL_LEVEL,
            antecedent_stats::QuantileRule::ExchangeableRank,
        );
    }
    Ok(ObservationalFit {
        estimate: point.values,
        standard_error: Some(standard_error),
        interval: Some(interval),
        replicates: (replicates - failed, failed),
        posterior: None,
        minimum_probability: point.minimum_probability,
        propensity: point.coefficients,
        pairs,
        clusters: g_count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{PanelRows, lagged_pairs};
    use super::*;

    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x6A09_E667_F3BC_C908;
        state ^= state >> 31;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    struct Panel {
        unit: Vec<usize>,
        period: Vec<i64>,
        treated: Vec<bool>,
        history: Vec<f64>,
        outcome: Vec<f64>,
    }

    /// A decision-time covariate drives both treatment and outcome; the true excursion effect
    /// on the same-period outcome is `effect`.
    fn confounded_panel(units: u64, effect: f64) -> Panel {
        let mut panel = Panel {
            unit: Vec::new(),
            period: Vec::new(),
            treated: Vec::new(),
            history: Vec::new(),
            outcome: Vec::new(),
        };
        for i in 0..units {
            for t in 0..4_u64 {
                let key = i * 17 + t;
                let x = 2.0 * uniform(key) - 1.0;
                let p = antecedent_kernels::sigmoid(1.2 * x);
                let a = uniform(key + 500_000) < p;
                let noise = uniform(key + 900_000) - 0.5;
                panel.unit.push(usize::try_from(i).unwrap());
                panel.period.push(i64::try_from(t).unwrap());
                panel.treated.push(a);
                panel.history.push(x);
                panel.outcome.push(effect * f64::from(u8::from(a)) + 3.0 * x + noise);
            }
        }
        panel
    }

    fn naive_difference(panel: &Panel) -> f64 {
        let (mut t_sum, mut t_n, mut c_sum, mut c_n) = (0.0, 0.0, 0.0, 0.0);
        for (a, y) in panel.treated.iter().zip(&panel.outcome) {
            if *a {
                t_sum += y;
                t_n += 1.0;
            } else {
                c_sum += y;
                c_n += 1.0;
            }
        }
        t_sum / t_n - c_sum / c_n
    }

    #[test]
    fn the_fitted_assignment_law_removes_decision_time_confounding() {
        let panel = confounded_panel(400, 1.0);
        let pairs =
            lagged_pairs(PanelRows { unit: &panel.unit, period: &panel.period }, 0).unwrap();
        let history = vec![panel.history.clone()];
        let rows = ObservationalRows {
            pairs: &pairs,
            outcome: &panel.outcome,
            treated: &panel.treated,
            history: &history,
            unit: &panel.unit,
        };
        let ctx = ExecutionContext::for_tests(3);
        let fit = fit_observational_excursion(rows, 0.05, 1.0, 99, None, &ctx).unwrap();
        // Treatment is confounded upward by the covariate, so the raw contrast is far above one.
        assert!(naive_difference(&panel) > 1.6, "{}", naive_difference(&panel));
        assert!((fit.estimate[0] - 1.0).abs() < 0.4, "{:?}", fit.estimate);
        assert!((fit.propensity[1] - 1.2).abs() < 0.4, "{:?}", fit.propensity);
        let (low, high) = fit.interval.unwrap()[0];
        assert!(low < fit.estimate[0] && fit.estimate[0] < high);
        assert!(fit.standard_error.unwrap()[0] > 0.0);
        assert_eq!(fit.replicates.0 + fit.replicates.1, 99);
    }

    #[test]
    fn the_posterior_refits_the_assignment_model_and_repeats_under_the_seed() {
        let panel = confounded_panel(300, 1.0);
        let pairs =
            lagged_pairs(PanelRows { unit: &panel.unit, period: &panel.period }, 0).unwrap();
        let history = vec![panel.history.clone()];
        let rows = ObservationalRows {
            pairs: &pairs,
            outcome: &panel.outcome,
            treated: &panel.treated,
            history: &history,
            unit: &panel.unit,
        };
        let ctx = ExecutionContext::for_tests(9);
        let first = fit_observational_excursion(rows, 0.05, 5.0, 0, Some(100), &ctx).unwrap();
        let second = fit_observational_excursion(rows, 0.05, 5.0, 0, Some(100), &ctx).unwrap();
        assert_eq!(first.posterior, second.posterior);
        let posterior = first.posterior.unwrap();
        assert!((posterior.mean[0] - 1.0).abs() < 0.45, "{:?}", posterior.mean);
        assert!(posterior.standard_deviation[0] > 0.0);
        assert!(posterior.lower[0] < posterior.mean[0] && posterior.mean[0] < posterior.upper[0]);
    }

    #[test]
    fn a_history_that_separates_the_arms_is_refused_and_replicates_are_required() {
        let mut panel = confounded_panel(120, 1.0);
        // Treatment is a deterministic function of the covariate: positivity fails.
        for (a, x) in panel.treated.iter_mut().zip(&panel.history) {
            *a = *x > 0.0;
        }
        let pairs =
            lagged_pairs(PanelRows { unit: &panel.unit, period: &panel.period }, 0).unwrap();
        let history = vec![panel.history.clone()];
        let rows = ObservationalRows {
            pairs: &pairs,
            outcome: &panel.outcome,
            treated: &panel.treated,
            history: &history,
            unit: &panel.unit,
        };
        let ctx = ExecutionContext::for_tests(1);
        assert!(fit_observational_excursion(rows, 0.05, 1.0, 99, None, &ctx).is_err());
        let ordinary = confounded_panel(120, 1.0);
        let pairs =
            lagged_pairs(PanelRows { unit: &ordinary.unit, period: &ordinary.period }, 0).unwrap();
        let history = vec![ordinary.history.clone()];
        let rows = ObservationalRows {
            pairs: &pairs,
            outcome: &ordinary.outcome,
            treated: &ordinary.treated,
            history: &history,
            unit: &ordinary.unit,
        };
        assert!(fit_observational_excursion(rows, 0.05, 1.0, 10, None, &ctx).is_err());
    }
}

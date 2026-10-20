//! Conditional censoring survival with a Cox model and Breslow ties.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::diagnostic::{RouteCheck, RouteMass, RouteNativeDiagnostic};

use crate::{StatsError, chol_solve, cholesky_spd};

/// Executed censoring-model fit. No outcome-regression uncertainty is implied.
#[derive(Clone, Debug, PartialEq)]
pub struct CoxIpcwFit {
    /// Log hazard coefficients in the caller's covariate units (no intercept).
    pub coefficients: Vec<f64>,
    /// Conditional censoring survival immediately before each recorded time.
    pub survival_before: Vec<f64>,
    /// Zero on censored rows; inverse survival on observed rows.
    pub weights: Vec<f64>,
}

fn invalid(message: &'static str) -> StatsError {
    StatsError::Unsupported { message }
}

/// Cox proportional-hazards fit with a Breslow cumulative baseline hazard.
///
/// The hazard of a subject with covariates `z` is `exp(eta(z)) dLambda0(t)` with
/// `eta(z) = sum_j coefficients[j] z[j] - center`, so the cumulative hazard through `t` is
/// `Lambda0(t) exp(eta(z))`. Tied failures use Breslow.
#[derive(Clone, Debug, PartialEq)]
pub struct CoxBreslowFit {
    /// Log hazard coefficients in the caller's covariate units (no intercept).
    pub coefficients: Vec<f64>,
    /// Offset removed from the linear predictor so that `Lambda0` is the baseline at the
    /// covariate means.
    pub center: f64,
    /// Ascending distinct failure times.
    pub jump_times: Vec<f64>,
    /// Cumulative baseline hazard immediately after each jump time.
    pub cumulative_baseline: Vec<f64>,
}

impl CoxBreslowFit {
    /// Linear predictor `eta` of one covariate row.
    #[must_use]
    pub fn linear_predictor(&self, row: &[f64]) -> f64 {
        self.coefficients.iter().zip(row).map(|(b, z)| b * z).sum::<f64>() - self.center
    }

    /// Cumulative baseline hazard over failure times strictly before `t`.
    #[must_use]
    pub fn baseline_before(&self, t: f64) -> f64 {
        let before = self.jump_times.partition_point(|&jump| jump < t);
        if before == 0 { 0.0 } else { self.cumulative_baseline[before - 1] }
    }

    /// Cumulative baseline hazard over failure times at or before `t`.
    #[must_use]
    pub fn baseline_at(&self, t: f64) -> f64 {
        let through = self.jump_times.partition_point(|&jump| jump <= t);
        if through == 0 { 0.0 } else { self.cumulative_baseline[through - 1] }
    }
}

/// Standardized-covariate Cox solution shared by [`cox_breslow`] and [`cox_ipcw`].
struct StandardizedCox {
    x: Vec<f64>,
    beta: Vec<f64>,
    scales: Vec<f64>,
    means: Vec<f64>,
    jump_times: Vec<f64>,
    cumulative_baseline: Vec<f64>,
}

/// When the Newton iteration stops.
#[derive(Clone, Copy)]
enum Stopping {
    /// An absolute score tolerance together with a small step: the censoring-weight route's
    /// original criterion, kept so its iterates and results are unchanged.
    AbsoluteScore,
    /// A small Newton step in standardized units, which does not grow with the row count
    /// (the score is a sum over the failures, so an absolute bound on it does).
    ScaleFree,
}

/// Maximize the Cox partial likelihood on standardized covariates by damped Newton steps.
/// `failure[i]` marks the rows whose time is a failure; covariates are column-major with no
/// intercept, and `p = 0` yields the Nelson-Aalen baseline.
fn fit_standardized(
    time: &[f64],
    failure: &[bool],
    covariates: &[f64],
    p: usize,
    stopping: Stopping,
) -> Result<StandardizedCox, StatsError> {
    let n = time.len();
    let mut x = covariates.to_vec();
    let mut scales = vec![0.0; p];
    let mut means = vec![0.0; p];
    for j in 0..p {
        let col = &mut x[j * n..(j + 1) * n];
        let mean = col.iter().sum::<f64>() / n as f64;
        let scale = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        if scale <= f64::EPSILON || !scale.is_finite() {
            return Err(invalid("singular Cox IPCW covariate"));
        }
        scales[j] = scale;
        means[j] = mean;
        for v in col {
            *v = (*v - mean) / scale;
        }
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| time[b].total_cmp(&time[a]));
    let mut beta = vec![0.0; p];
    let mut scratch = CoxScratch::new(n, p);
    // Without covariates there is nothing to maximize: the baseline is Nelson-Aalen.
    let mut converged = p == 0;
    let iterations = if p == 0 { 0 } else { 100 };
    for _ in 0..iterations {
        let ll = evaluate(time, failure, &x, &order, &beta, &mut scratch)?;
        let chol = cholesky_spd(&scratch.info, p)
            .ok_or_else(|| invalid("singular Cox IPCW information"))?;
        let step =
            chol_solve(&chol, p, &scratch.score).ok_or_else(|| invalid("Cox IPCW solve failed"))?;
        // Separation can drive the score to zero while Newton steps remain large.
        let step_size = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let score_small = match stopping {
            Stopping::AbsoluteScore => {
                scratch.score.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-9
            }
            Stopping::ScaleFree => true,
        };
        if score_small && step_size < 1e-8 {
            converged = true;
            break;
        }
        // Rounding noise of the log partial likelihood grows with its magnitude, so the
        // scale-free mode compares relative to it; the legacy mode keeps its absolute bound.
        let tolerance = match stopping {
            Stopping::AbsoluteScore => 1e-12,
            Stopping::ScaleFree => 1e-12 * (1.0 + ll.abs()),
        };
        let mut fraction = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let candidate: Vec<_> = beta.iter().zip(&step).map(|(b, d)| b + fraction * d).collect();
            if let Ok(next) = evaluate(time, failure, &x, &order, &candidate, &mut scratch) {
                if next >= ll - tolerance {
                    beta = candidate;
                    accepted = true;
                    break;
                }
            }
            fraction *= 0.5;
        }
        if !accepted {
            // At the optimum no step improves the likelihood beyond rounding noise.
            if matches!(stopping, Stopping::ScaleFree) && step_size < 1e-6 {
                converged = true;
                break;
            }
            return Err(invalid("Cox IPCW line search failed"));
        }
    }
    if !converged {
        return Err(invalid("Cox IPCW did not converge"));
    }
    evaluate(time, failure, &x, &order, &beta, &mut scratch)?;
    if p > 0 {
        cholesky_spd(&scratch.info, p).ok_or_else(|| invalid("singular Cox IPCW information"))?;
    }
    let mut jumps = std::mem::take(&mut scratch.jumps);
    jumps.reverse();
    let mut cumulative = 0.0;
    let mut jump_times = Vec::with_capacity(jumps.len());
    let mut cumulative_baseline = Vec::with_capacity(jumps.len());
    for (t, h) in jumps {
        cumulative += h;
        jump_times.push(t);
        cumulative_baseline.push(cumulative);
    }
    Ok(StandardizedCox { x, beta, scales, means, jump_times, cumulative_baseline })
}

/// Fit a Cox model with a Breslow baseline to rows with a failure indicator.
/// Covariates are column-major, without an intercept; `p = 0` is the Nelson-Aalen estimate.
///
/// # Errors
/// Invalid inputs, a constant covariate, singular information, nonconvergence, or no failure.
pub fn cox_breslow(
    time: &[f64],
    failure: &[bool],
    covariates: &[f64],
    p: usize,
) -> Result<CoxBreslowFit, StatsError> {
    let n = time.len();
    if n == 0 || p >= n || failure.len() != n || covariates.len() != n * p {
        return Err(invalid("Cox fit requires aligned nonempty data and covariates"));
    }
    if time.iter().chain(covariates).any(|x| !x.is_finite()) || !failure.contains(&true) {
        return Err(invalid("invalid Cox times or covariates, or no failure"));
    }
    let fit = fit_standardized(time, failure, covariates, p, Stopping::ScaleFree)?;
    let coefficients: Vec<f64> = fit.beta.iter().zip(&fit.scales).map(|(b, s)| b / s).collect();
    let center = coefficients.iter().zip(&fit.means).map(|(c, m)| c * m).sum();
    Ok(CoxBreslowFit {
        coefficients,
        center,
        jump_times: fit.jump_times,
        cumulative_baseline: fit.cumulative_baseline,
    })
}

/// Fit the censoring hazard to `1-event` and evaluate `event/G(time-|Z)`.
/// Covariates are column-major, without an intercept. Times may be negative
/// (left censoring is represented by sign reversal). Tied failures use Breslow.
///
/// # Errors
/// Invalid inputs, singular information, nonconvergence, or observed-row survival
/// below `survival_floor`. No marginal-KM fallback or coefficient regularization.
#[allow(
    clippy::float_cmp,
    reason = "event indicators are validated as exactly 0.0 or 1.0 codes, so equality tests the coded value"
)]
pub fn cox_ipcw(
    time: &[f64],
    event: &[f64],
    covariates: &[f64],
    p: usize,
    survival_floor: f64,
) -> Result<CoxIpcwFit, StatsError> {
    let n = time.len();
    if n == 0 || p == 0 || p >= n || event.len() != n || covariates.len() != n * p {
        return Err(invalid("Cox IPCW requires aligned nonempty data and covariates"));
    }
    if time.iter().chain(covariates).any(|x| !x.is_finite())
        || event.iter().any(|&x| x != 0.0 && x != 1.0)
        || !event.contains(&1.0)
        || !survival_floor.is_finite()
        || survival_floor <= 0.0
        || survival_floor >= 1.0
    {
        return Err(invalid("invalid Cox IPCW times, event, covariates or positivity floor"));
    }
    let censored: Vec<bool> = event.iter().map(|&e| e == 0.0).collect();
    let fit = fit_standardized(time, &censored, covariates, p, Stopping::AbsoluteScore)?;
    let StandardizedCox { x, beta, scales, jump_times, cumulative_baseline, .. } = fit;
    let mut survival_before = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let eta = (0..p).map(|j| x[j * n + i] * beta[j]).sum::<f64>();
        let before = jump_times.partition_point(|t| *t < time[i]);
        let hazard = if before == 0 { 0.0 } else { cumulative_baseline[before - 1] };
        let survival = (-hazard * eta.exp()).exp();
        if !survival.is_finite() || (event[i] == 1.0 && survival < survival_floor) {
            return Err(StatsError::Route {
                message: "conditional censoring survival is below the configured positivity floor",
                diagnostic: RouteNativeDiagnostic::censoring(RouteCheck::new(
                    true,
                    RouteMass::zero(),
                )),
            });
        }
        survival_before.push(survival);
        weights.push(if event[i] == 1.0 { 1.0 / survival } else { 0.0 });
    }
    Ok(CoxIpcwFit {
        coefficients: beta.iter().zip(scales).map(|(b, s)| b / s).collect(),
        survival_before,
        weights,
    })
}

struct CoxScratch {
    eta: Vec<f64>,
    first: Vec<f64>,
    second: Vec<f64>,
    score: Vec<f64>,
    info: Vec<f64>,
    jumps: Vec<(f64, f64)>,
}

impl CoxScratch {
    fn new(n: usize, p: usize) -> Self {
        Self {
            eta: vec![0.0; n],
            first: vec![0.0; p],
            second: vec![0.0; p * p],
            score: vec![0.0; p],
            info: vec![0.0; p * p],
            jumps: Vec::with_capacity(n),
        }
    }

    fn reset_accumulators(&mut self) {
        self.first.fill(0.0);
        self.second.fill(0.0);
        self.score.fill(0.0);
        self.info.fill(0.0);
        self.jumps.clear();
    }
}

// Descending risk-set accumulation is O(n p²), including tied groups.
#[allow(
    clippy::float_cmp,
    reason = "tie groups are runs of exactly equal observed values, so bitwise equality is the definition of a tie"
)]
fn evaluate(
    time: &[f64],
    failure: &[bool],
    x: &[f64],
    order: &[usize],
    beta: &[f64],
    scratch: &mut CoxScratch,
) -> Result<f64, StatsError> {
    let n = time.len();
    let p = beta.len();
    scratch.eta.resize(n, 0.0);
    for (i, eta) in scratch.eta.iter_mut().enumerate() {
        *eta = (0..p).map(|j| beta[j] * x[j * n + i]).sum::<f64>();
    }
    if scratch.eta.iter().any(|v| !v.is_finite() || v.abs() > 500.0) {
        return Err(invalid("diverging Cox IPCW coefficients"));
    }
    scratch.reset_accumulators();
    let mut risk = 0.0;
    let mut ll = 0.0;
    let mut start = 0;
    while start < n {
        let t = time[order[start]];
        let mut end = start;
        while end < n && time[order[end]] == t {
            let i = order[end];
            let w = scratch.eta[i].exp();
            risk += w;
            for j in 0..p {
                scratch.first[j] += w * x[j * n + i];
                for k in 0..p {
                    scratch.second[j * p + k] += w * x[j * n + i] * x[k * n + i];
                }
            }
            end += 1;
        }
        let mut deaths = 0.0;
        for &i in &order[start..end] {
            if failure[i] {
                deaths += 1.0;
                ll += scratch.eta[i];
                for j in 0..p {
                    scratch.score[j] += x[j * n + i];
                }
            }
        }
        if deaths > 0.0 {
            ll -= deaths * risk.ln();
            for j in 0..p {
                scratch.score[j] -= deaths * scratch.first[j] / risk;
                for k in 0..p {
                    scratch.info[j * p + k] += deaths
                        * (scratch.second[j * p + k] / risk
                            - scratch.first[j] * scratch.first[k] / risk.powi(2));
                }
            }
            scratch.jumps.push((t, deaths / risk));
        }
        start = end;
    }
    Ok(ll)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn executing_survival_oracle() {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/conditional_ipcw/expected.json"
        ))
        .unwrap();
        let nums = |k: &str| -> Vec<f64> {
            pin["data"][k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
        };
        let mut x = nums("a");
        x.extend(nums("z"));
        let fit = cox_ipcw(&nums("time"), &nums("event"), &x, 2, 1e-6).unwrap();
        let beta: Vec<_> =
            pin["coefficients"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let atol = pin["atol"].as_f64().unwrap();
        // `zip` truncates to the shorter side: pin the lengths so a short fit cannot pass.
        assert_eq!(fit.coefficients.len(), beta.len());
        assert_eq!(fit.survival_before.len(), nums("survival").len());
        assert_eq!(fit.weights.len(), nums("weight").len());
        for (got, want) in fit
            .coefficients
            .iter()
            .zip(beta)
            .chain(fit.survival_before.iter().zip(nums("survival")))
            .chain(fit.weights.iter().zip(nums("weight")))
        {
            assert!((got - want).abs() < atol, "{got} != {want}");
        }
    }
    #[test]
    fn covariate_free_breslow_baseline_is_nelson_aalen() {
        // Failures at 1, 2 (twice, tied), 4 with censoring at 3 and 5.
        let time = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let failure = [true, true, true, false, true, false];
        let fit = cox_breslow(&time, &failure, &[], 0).unwrap();
        assert_eq!(fit.jump_times, vec![1.0, 2.0, 4.0]);
        // At risk 6, 5 (two failures), 2: increments 1/6, 2/5, 1/2.
        let want = [1.0 / 6.0, 1.0 / 6.0 + 2.0 / 5.0, 1.0 / 6.0 + 2.0 / 5.0 + 0.5];
        for (got, want) in fit.cumulative_baseline.iter().zip(want) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
        assert!((fit.baseline_before(2.0) - want[0]).abs() <= f64::EPSILON);
        assert!((fit.baseline_at(2.0) - want[1]).abs() <= f64::EPSILON);
        assert!(fit.baseline_at(0.5).abs() <= f64::EPSILON);
        assert!((fit.baseline_at(9.0) - want[2]).abs() <= f64::EPSILON);
        assert!(fit.coefficients.is_empty() && fit.center.abs() <= f64::EPSILON);
    }

    #[test]
    fn breslow_fit_agrees_with_the_censoring_fit_it_generalizes() {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/conditional_ipcw/expected.json"
        ))
        .unwrap();
        let nums = |k: &str| -> Vec<f64> {
            pin["data"][k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
        };
        let mut x = nums("a");
        x.extend(nums("z"));
        let (time, event) = (nums("time"), nums("event"));
        let censored: Vec<bool> = event.iter().map(|&e| e == 0.0).collect();
        let breslow = cox_breslow(&time, &censored, &x, 2).unwrap();
        let ipcw = cox_ipcw(&time, &event, &x, 2, 1e-6).unwrap();
        let n = time.len();
        for (got, want) in breslow.coefficients.iter().zip(&ipcw.coefficients) {
            assert!((got - want).abs() < 1e-9, "{got} != {want}");
        }
        // Survival immediately before each recorded time, from the shared baseline.
        for i in 0..n {
            let eta = breslow.linear_predictor(&[x[i], x[n + i]]);
            let survival = (-breslow.baseline_before(time[i]) * eta.exp()).exp();
            assert!((survival - ipcw.survival_before[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn breslow_converges_on_a_large_exponential_sample_with_discrete_covariates() {
        // Exponential failure times with log hazard 0.7 z1 + 0.5 z2 on 4000 rows: the score is
        // a sum over thousands of failures, so convergence must not depend on its absolute size.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let n = 4000;
        let (mut time, mut failure) = (Vec::new(), Vec::new());
        let (mut z1, mut z2) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let a = f64::from(u8::from(uniform() < 0.5));
            let b = (uniform() * 3.0).floor() - 1.0;
            let failure_time = -uniform().ln() / (0.3 * (0.7 * a + 0.5 * b).exp());
            let censor_time = -uniform().ln() / 0.15;
            time.push(failure_time.min(censor_time));
            failure.push(failure_time <= censor_time);
            z1.push(a);
            z2.push(b);
        }
        let x: Vec<f64> = z1.into_iter().chain(z2).collect();
        let fit = cox_breslow(&time, &failure, &x, 2).unwrap();
        assert!((fit.coefficients[0] - 0.7).abs() < 0.15, "{:?}", fit.coefficients);
        assert!((fit.coefficients[1] - 0.5).abs() < 0.1, "{:?}", fit.coefficients);
    }

    #[test]
    fn breslow_requires_a_failure_and_aligned_data() {
        assert!(cox_breslow(&[1.0, 2.0], &[false, false], &[], 0).is_err());
        assert!(cox_breslow(&[1.0, 2.0], &[true], &[], 0).is_err());
        assert!(cox_breslow(&[1.0, 2.0, 3.0], &[true, true, false], &[1.0, 1.0, 1.0], 1).is_err());
    }

    #[test]
    fn singular_and_invalid_cox_inputs_fail_closed() {
        assert!(cox_ipcw(&[1., 2., 3.], &[0., 0., 0.], &[1., 2., 3.], 1, 0.01).is_err());
        assert!(cox_ipcw(&[1., 2., 3.], &[1., 0., 1.], &[1., 1., 1.], 1, 0.01).is_err());
        assert!(cox_ipcw(&[1., 2., 3.], &[1., 1., 1.], &[1., 2., 3.], 1, 0.01).is_err());
        assert!(cox_ipcw(&[1., 2., 3.], &[1., 2., 1.], &[1., 2., 3.], 1, 0.01).is_err());
    }

    #[test]
    fn conditional_censoring_survival_below_the_floor_is_a_censoring_check() {
        let err = cox_ipcw(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 0.0, 1.0, 1.0],
            &[0.0, 1.0, 2.0, 3.0],
            1,
            0.99,
        )
        .unwrap_err();
        let record = err.route_diagnostic().expect("censoring record");
        assert!(record.failed());
        assert_eq!(record.route_name(), "censoring");
        assert!(!record.is_unsupported_inference());
        assert!(!record.is_model_restriction());
        assert!(!matches!(err, StatsError::Unsupported { .. }));
    }
}

//! Conditional censoring survival with a Cox model and Breslow ties.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]
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

/// Fit the censoring hazard to `1-event` and evaluate `event/G(time-|Z)`.
/// Covariates are column-major, without an intercept. Times may be negative
/// (left censoring is represented by sign reversal). Tied failures use Breslow.
///
/// # Errors
/// Invalid inputs, singular information, nonconvergence, or observed-row survival
/// below `survival_floor`. No marginal-KM fallback or coefficient regularization.
#[allow(clippy::too_many_lines)]
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
    let mut x = covariates.to_vec();
    let mut scales = vec![0.0; p];
    for j in 0..p {
        let col = &mut x[j * n..(j + 1) * n];
        let mean = col.iter().sum::<f64>() / n as f64;
        let scale = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        if scale <= f64::EPSILON || !scale.is_finite() {
            return Err(invalid("singular Cox IPCW covariate"));
        }
        scales[j] = scale;
        for v in col {
            *v = (*v - mean) / scale;
        }
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| time[b].total_cmp(&time[a]));
    let mut beta = vec![0.0; p];
    let mut scratch = CoxScratch::new(n, p);
    let mut converged = false;
    for _ in 0..100 {
        let ll = evaluate(time, event, &x, &order, &beta, &mut scratch)?;
        let chol = cholesky_spd(&scratch.info, p)
            .ok_or_else(|| invalid("singular Cox IPCW information"))?;
        let step =
            chol_solve(&chol, p, &scratch.score).ok_or_else(|| invalid("Cox IPCW solve failed"))?;
        // Separation can drive the score to zero while Newton steps remain large.
        if scratch.score.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-9
            && step.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-8
        {
            converged = true;
            break;
        }
        let mut fraction = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let candidate: Vec<_> = beta.iter().zip(&step).map(|(b, d)| b + fraction * d).collect();
            if let Ok(next) = evaluate(time, event, &x, &order, &candidate, &mut scratch) {
                if next >= ll - 1e-12 {
                    beta = candidate;
                    accepted = true;
                    break;
                }
            }
            fraction *= 0.5;
        }
        if !accepted {
            return Err(invalid("Cox IPCW line search failed"));
        }
    }
    if !converged {
        return Err(invalid("Cox IPCW did not converge"));
    }
    evaluate(time, event, &x, &order, &beta, &mut scratch)?;
    cholesky_spd(&scratch.info, p).ok_or_else(|| invalid("singular Cox IPCW information"))?;
    let mut jumps = std::mem::take(&mut scratch.jumps);
    jumps.reverse();
    let mut cumulative = 0.0;
    for (_, h) in &mut jumps {
        cumulative += *h;
        *h = cumulative;
    }
    let mut survival_before = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let eta = (0..p).map(|j| x[j * n + i] * beta[j]).sum::<f64>();
        let before = jumps.partition_point(|(t, _)| *t < time[i]);
        let hazard = if before == 0 { 0.0 } else { jumps[before - 1].1 };
        let survival = (-hazard * eta.exp()).exp();
        if !survival.is_finite() || (event[i] == 1.0 && survival < survival_floor) {
            return Err(invalid(
                "conditional censoring survival is below the configured positivity floor",
            ));
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
fn evaluate(
    time: &[f64],
    event: &[f64],
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
            if event[i] == 0.0 {
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
    fn singular_and_invalid_cox_inputs_fail_closed() {
        assert!(cox_ipcw(&[1., 2., 3.], &[0., 0., 0.], &[1., 2., 3.], 1, 0.01).is_err());
        assert!(cox_ipcw(&[1., 2., 3.], &[1., 0., 1.], &[1., 1., 1.], 1, 0.01).is_err());
        assert!(cox_ipcw(&[1., 2., 3.], &[1., 1., 1.], &[1., 2., 3.], 1, 0.01).is_err());
        assert!(cox_ipcw(&[1., 2., 3.], &[1., 2., 1.], &[1., 2., 3.], 1, 0.01).is_err());
    }
}

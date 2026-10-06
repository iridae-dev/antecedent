//! L1-penalized (lasso) logistic regression: the probability-task form of a lasso learner.
//!
//! The objective is `-loglik(beta) + lambda * sum_{j >= 1} |beta_j|` on a *sum* (not mean)
//! log likelihood, the same scale as [`crate::RidgeLogisticLearner`]. A constant first design
//! column is the intercept and is left unpenalized; without one every coefficient is
//! penalized and the fit has no intercept. The learner does not standardize: a caller that
//! wants scale-free selection standardizes the columns first.
//!
//! Solver: proximal Newton. Each outer step builds the weighted least-squares quadratic model
//! of the logistic loss at the current iterate and minimizes it plus the L1 term exactly by
//! cyclic coordinate descent (soft-thresholding), then backtracks on the true objective, so
//! every accepted step decreases it. The solution is accepted only when the Karush-Kuhn-Tucker
//! conditions of the *exact* objective hold: `|g_j| <= lambda` where the coefficient is zero,
//! `g_j = lambda * sign(beta_j)` where it is not, and `g_0 = 0` for the intercept, with
//! `g = X' (y - p)`. A fit that does not meet them is refused rather than used as a nuisance.
//!
//! Unlike the unpenalized logistic MLE, the penalized objective has a finite minimizer under
//! (quasi-)complete separation, so the learner fits where the plain GLM refuses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use antecedent_core::ExecutionContext;
use antecedent_stats::first_col_is_exact_ones;

use crate::dense::{
    gather_physical, materialize_dense_colmajor, predict_linear, require_binary_labels,
};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};

/// Proximal-Newton outer iterations.
const MAX_NEWTON: usize = 200;

/// Coordinate-descent sweeps per outer iteration.
const MAX_SWEEPS: usize = 2_000;

/// Step halvings of the backtracking line search.
const MAX_HALVINGS: usize = 40;

/// Largest KKT violation accepted, relative to `1 + lambda` (sum-scale score units).
const KKT_TOLERANCE: f64 = 1e-7;

/// A sweep ends when no coordinate's score moved by more than this share of the KKT tolerance,
/// so the coordinate-descent residual cannot keep the outer iteration from certifying its
/// optimality conditions.
const SWEEP_SHARE: f64 = 0.1;

/// Floor of the working weight `p (1 - p)`, so a saturated row cannot zero a curvature.
const MIN_WEIGHT: f64 = 1e-9;

/// Lasso-penalized logistic regression via proximal Newton with coordinate descent.
#[derive(Clone, Copy, Debug)]
pub struct LassoLogisticLearner {
    lambda: f64,
}

impl LassoLogisticLearner {
    /// Penalized logistic factory. `lambda` must be finite and positive.
    ///
    /// # Errors
    ///
    /// [`LearnError::Shape`] for a non-finite or non-positive penalty.
    pub fn new(lambda: f64) -> Result<Self, LearnError> {
        if lambda.is_finite() && lambda > 0.0 {
            Ok(Self { lambda })
        } else {
            Err(LearnError::Shape { message: "logistic lasso penalty must be finite and > 0" })
        }
    }
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// `log(1 + exp(v))` without overflow.
fn softplus(value: f64) -> f64 {
    if value > 0.0 { value + (-value).exp().ln_1p() } else { value.exp().ln_1p() }
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    value.signum() * (value.abs() - threshold).max(0.0)
}

/// `-loglik + lambda * sum |beta_j|` over the penalized coordinates.
fn objective(eta: &[f64], y: &[f64], beta: &[f64], lambda: f64, pen_start: usize) -> f64 {
    let loss: f64 = eta.iter().zip(y).map(|(&e, &t)| softplus(e) - t * e).sum();
    loss + lambda * beta[pen_start..].iter().map(|b| b.abs()).sum::<f64>()
}

/// `g = X' (y - p)`: the negative gradient of the negative log likelihood.
fn score(x: &[f64], n: usize, p: usize, y: &[f64], prob: &[f64]) -> Vec<f64> {
    (0..p)
        .map(|j| x[j * n..(j + 1) * n].iter().zip(y).zip(prob).map(|((v, t), q)| v * (t - q)).sum())
        .collect()
}

/// Largest violation of the exact objective's optimality conditions.
#[allow(
    clippy::float_cmp,
    reason = "a coefficient is exactly zero when the soft threshold clamped it"
)]
fn kkt_violation(score: &[f64], beta: &[f64], lambda: f64, pen_start: usize) -> f64 {
    let mut worst = 0.0_f64;
    for (j, (&g, &b)) in score.iter().zip(beta).enumerate() {
        let violation = if j < pen_start {
            g.abs()
        } else if b == 0.0 {
            (g.abs() - lambda).max(0.0)
        } else {
            (g - lambda * b.signum()).abs()
        };
        worst = worst.max(violation);
    }
    worst
}

/// Minimize `0.5 sum w_i (r_i - x_i . delta)^2 + lambda sum |beta_j + delta_j|` over `delta`
/// by cyclic coordinate descent; `resid` enters as `r` and is consumed as the running residual.
#[allow(clippy::too_many_arguments)]
fn coordinate_descent(
    x: &[f64],
    n: usize,
    weights: &[f64],
    resid: &mut [f64],
    curvature: &[f64],
    beta: &[f64],
    delta: &mut [f64],
    lambda: f64,
    pen_start: usize,
    sweep_tolerance: f64,
) {
    for _ in 0..MAX_SWEEPS {
        let mut largest = 0.0_f64;
        for j in 0..beta.len() {
            let a = curvature[j];
            if a <= 1e-300 {
                continue;
            }
            let col = &x[j * n..(j + 1) * n];
            let mut u = a * delta[j];
            for i in 0..n {
                u += weights[i] * col[i] * resid[i];
            }
            let target = a * beta[j] + u;
            let updated =
                if j >= pen_start { soft_threshold(target, lambda) / a } else { target / a };
            let new_delta = updated - beta[j];
            let diff = new_delta - delta[j];
            for i in 0..n {
                resid[i] -= col[i] * diff;
            }
            delta[j] = new_delta;
            largest = largest.max(a * diff.abs());
        }
        if largest <= sweep_tolerance {
            break;
        }
    }
}

/// Solve the penalized problem; the coefficients are returned only if they satisfy the KKT
/// conditions of the exact objective.
fn solve(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    lambda: f64,
    intercept: bool,
) -> Result<Vec<f64>, LearnError> {
    let pen_start = usize::from(intercept);
    let mut beta = vec![0.0; p];
    let mut eta = vec![0.0; n];
    if intercept {
        let ybar = y.iter().sum::<f64>() / n as f64;
        if !(ybar > 0.0 && ybar < 1.0) {
            return Err(LearnError::Shape {
                message: "logistic lasso needs both classes in the training target",
            });
        }
        beta[0] = (ybar / (1.0 - ybar)).ln();
        eta.fill(beta[0]);
    }
    let tolerance = KKT_TOLERANCE * (1.0 + lambda);
    let mut weights = vec![0.0; n];
    let mut resid = vec![0.0; n];
    let mut curvature = vec![0.0; p];
    let mut delta = vec![0.0; p];
    let mut x_delta = vec![0.0; n];
    let mut converged = false;
    for _ in 0..MAX_NEWTON {
        let prob: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
        if kkt_violation(&score(x, n, p, y, &prob), &beta, lambda, pen_start) <= tolerance {
            converged = true;
            break;
        }
        for i in 0..n {
            weights[i] = (prob[i] * (1.0 - prob[i])).max(MIN_WEIGHT);
            resid[i] = (y[i] - prob[i]) / weights[i];
        }
        for j in 0..p {
            curvature[j] = x[j * n..(j + 1) * n].iter().zip(&weights).map(|(v, w)| w * v * v).sum();
        }
        delta.fill(0.0);
        coordinate_descent(
            x,
            n,
            &weights,
            &mut resid,
            &curvature,
            &beta,
            &mut delta,
            lambda,
            pen_start,
            SWEEP_SHARE * tolerance,
        );
        x_delta.fill(0.0);
        for j in 0..p {
            for (slot, v) in x_delta.iter_mut().zip(&x[j * n..(j + 1) * n]) {
                *slot += v * delta[j];
            }
        }
        let current = objective(&eta, y, &beta, lambda, pen_start);
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..MAX_HALVINGS {
            let candidate_beta: Vec<f64> =
                beta.iter().zip(&delta).map(|(b, d)| b + step * d).collect();
            let candidate_eta: Vec<f64> =
                eta.iter().zip(&x_delta).map(|(e, d)| e + step * d).collect();
            let value = objective(&candidate_eta, y, &candidate_beta, lambda, pen_start);
            if value <= current + 1e-12 * current.abs() {
                beta = candidate_beta;
                eta = candidate_eta;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    if !converged {
        let prob: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
        converged = kkt_violation(&score(x, n, p, y, &prob), &beta, lambda, pen_start) <= tolerance;
    }
    if !converged {
        return Err(LearnError::Backend(format!(
            "logistic lasso did not reach its optimality conditions in {MAX_NEWTON} proximal \
             Newton iterations"
        )));
    }
    Ok(beta)
}

impl LearnerFactory for LassoLogisticLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::BinaryProbability
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            binary_probability: true,
            deterministic_seed: true,
            ..LearnerCapabilities::none()
        }
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if weights.is_some() {
            return Err(LearnError::Unsupported {
                message: "logistic lasso learner does not accept sample weights",
            });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        require_binary_labels(&gathered_y)?;
        if nrows == 0 || ncols == 0 {
            return Err(LearnError::Shape { message: "logistic lasso needs a non-empty design" });
        }
        let intercept = first_col_is_exact_ones(&design, nrows);
        let coefficients = solve(&design, nrows, ncols, &gathered_y, self.lambda, intercept)?;
        Ok(Box::new(LassoLogisticPredictor { coefficients }))
    }
}

struct LassoLogisticPredictor {
    coefficients: Vec<f64>,
}

impl FittedPredictor for LassoLogisticPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        predict_linear(&self.coefficients, x, out)?;
        for slot in out.iter_mut() {
            *slot = sigmoid(*slot).clamp(1e-9, 1.0 - 1e-9);
        }
        Ok(())
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Ok(crate::PortablePredictor {
            version: 1,
            columns: self.coefficients.len(),
            provenance: self.provenance(),
            model: crate::PredictionMap::Linear {
                coefficients: self.coefficients.clone(),
                logistic: true,
            },
        })
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "lasso_logistic".into(),
            implementation: "proximal_newton_coordinate_descent".into(),
            version: "1".into(),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "the tests build small synthetic designs and compare exact zeros of a thresholded fit"
)]
mod tests {
    use super::*;
    use crate::PredictionMap;

    fn coefficients(
        x: &[f64],
        n: usize,
        p: usize,
        y: &[f64],
        lambda: f64,
    ) -> Result<Vec<f64>, LearnError> {
        let view = DesignView::from_column_major(x, n, p)?;
        let fitted = LassoLogisticLearner::new(lambda)?.fit(
            view,
            TargetView::new(y),
            None,
            &ExecutionContext::for_tests(1),
        )?;
        match fitted.portable()?.model {
            PredictionMap::Linear { coefficients, logistic: true } => Ok(coefficients),
            other => panic!("unexpected portable map {other:?}"),
        }
    }

    fn uniform(state: &mut u64) -> f64 {
        *state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// `[1 | z_1 .. z_q]` with independent uniform-centered columns and a logistic response
    /// that depends on the first `active.len()` columns with the given coefficients.
    fn design(n: usize, q: usize, active: &[f64], seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut x = vec![1.0; n * (q + 1)];
        for j in 1..=q {
            for i in 0..n {
                x[j * n + i] = 2.0 * uniform(&mut state) - 1.0;
            }
        }
        let y = (0..n)
            .map(|i| {
                let eta: f64 = active.iter().enumerate().map(|(j, b)| b * x[(j + 1) * n + i]).sum();
                f64::from(u8::from(uniform(&mut state) < sigmoid(eta)))
            })
            .collect();
        (x, y)
    }

    /// Exact closed form of the two-group design `[1 | 1{group 1}]`: the intercept is
    /// unpenalized, so its score equation fixes `p0 = (k0 + lambda s) / n0` and the slope's
    /// stationarity fixes `p1 = (k1 - lambda s) / n1` (`s` the slope's sign) until
    /// `lambda` reaches `|k1 - n1 (k0 + k1) / n|`, where the slope is exactly zero.
    #[test]
    fn the_two_group_fit_matches_its_closed_form_and_threshold() {
        let (n0, n1, k0, k1) = (20usize, 20usize, 4.0, 14.0);
        let n = n0 + n1;
        let mut x = vec![1.0; 2 * n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[n + i] = f64::from(u8::from(i >= n0));
            y[i] = if i < 4 || (n0..n0 + 14).contains(&i) { 1.0 } else { 0.0 };
        }
        let logit = |q: f64| (q / (1.0 - q)).ln();
        let threshold = (k1 - n1 as f64 * (k0 + k1) / n as f64).abs();
        assert!((threshold - 5.0).abs() < 1e-12);
        for lambda in [0.5, 2.0, 4.9] {
            let beta = coefficients(&x, n, 2, &y, lambda).unwrap();
            let p0 = (k0 + lambda) / n0 as f64;
            let p1 = (k1 - lambda) / n1 as f64;
            assert!((beta[0] - logit(p0)).abs() < 1e-6, "{lambda}: {beta:?}");
            assert!((beta[1] - (logit(p1) - logit(p0))).abs() < 1e-6, "{lambda}: {beta:?}");
        }
        for lambda in [5.1, 7.0, 50.0] {
            let beta = coefficients(&x, n, 2, &y, lambda).unwrap();
            assert_eq!(beta[1], 0.0, "{lambda}: {beta:?}");
            assert!((beta[0] - logit((k0 + k1) / n as f64)).abs() < 1e-6, "{lambda}: {beta:?}");
        }
    }

    /// At the null model `p = ybar` every slope's score is `sum_i z_ij (y_i - ybar)`: the
    /// slopes are all zero iff `lambda` is at least the largest absolute score, and the
    /// first coordinate to enter below that is the argmax.
    #[test]
    fn the_null_threshold_is_the_largest_absolute_score_and_the_argmax_enters_first() {
        let (n, q) = (150usize, 6usize);
        let (x, y) = design(n, q, &[1.5, -0.5], 11);
        let ybar = y.iter().sum::<f64>() / n as f64;
        let scores: Vec<f64> =
            (1..=q).map(|j| (0..n).map(|i| x[j * n + i] * (y[i] - ybar)).sum::<f64>()).collect();
        let (argmax, lambda_max) = scores
            .iter()
            .map(|s| s.abs())
            .enumerate()
            .fold((0, 0.0), |best, (j, s)| if s > best.1 { (j + 1, s) } else { best });
        let above = coefficients(&x, n, q + 1, &y, lambda_max * 1.001).unwrap();
        assert!(above[1..].iter().all(|b| *b == 0.0), "{above:?}");
        let below = coefficients(&x, n, q + 1, &y, lambda_max * 0.999).unwrap();
        let entered: Vec<usize> = (1..=q).filter(|&j| below[j] != 0.0).collect();
        assert!(entered.contains(&argmax), "{below:?}");
        assert!(entered.iter().all(|&j| scores[j - 1].abs() > 0.9 * lambda_max), "{below:?}");
    }

    /// Independent optimality check: the score `X'(y - p)` at the returned coefficients,
    /// computed here from the raw design, satisfies the KKT conditions.
    #[test]
    fn the_solution_satisfies_the_kkt_conditions_of_the_exact_objective() {
        let (n, q) = (200usize, 12usize);
        let (x, y) = design(n, q, &[1.2, -1.0, 0.6], 5);
        for lambda in [0.5, 3.0, 10.0, 25.0] {
            let beta = coefficients(&x, n, q + 1, &y, lambda).unwrap();
            let mut max_violation = 0.0_f64;
            for j in 0..=q {
                let g: f64 = (0..n)
                    .map(|i| {
                        let eta: f64 = (0..=q).map(|c| x[c * n + i] * beta[c]).sum();
                        x[j * n + i] * (y[i] - 1.0 / (1.0 + (-eta).exp()))
                    })
                    .sum();
                let violation = if j == 0 {
                    g.abs()
                } else if beta[j] == 0.0 {
                    (g.abs() - lambda).max(0.0)
                } else {
                    (g - lambda * beta[j].signum()).abs()
                };
                max_violation = max_violation.max(violation);
            }
            assert!(max_violation < 1e-4 * (1.0 + lambda), "lambda {lambda}: {max_violation}");
        }
    }

    #[test]
    fn a_sparse_truth_is_recovered_as_the_support() {
        let (n, q) = (600usize, 10usize);
        let (x, y) = design(n, q, &[3.0, -3.0], 21);
        let beta = coefficients(&x, n, q + 1, &y, 12.0).unwrap();
        assert!(beta[1] > 0.5 && beta[2] < -0.5, "{beta:?}");
        let noise_selected = (3..=q).filter(|&j| beta[j] != 0.0).count();
        assert!(noise_selected <= 2, "{beta:?}");
    }

    #[test]
    fn a_separated_design_has_a_finite_penalized_solution() {
        let (n, q) = (80usize, 3usize);
        let (mut x, _) = design(n, q, &[], 9);
        let y: Vec<f64> = (0..n).map(|i| f64::from(u8::from(x[n + i] > 0.0))).collect();
        // The plain logistic has no finite maximizer here; the penalized one does.
        let beta = coefficients(&x, n, q + 1, &y, 1.0).unwrap();
        assert!(beta.iter().all(|b| b.is_finite()), "{beta:?}");
        assert!(beta[1] > 1.0, "{beta:?}");
        x[0] = 2.0;
        // A first column that is not exactly one is not an intercept: every coefficient is
        // penalized and the fit still converges.
        assert!(coefficients(&x, n, q + 1, &y, 1.0).unwrap().iter().all(|b| b.is_finite()));
    }

    #[test]
    fn invalid_penalties_targets_and_weights_are_refused() {
        for lambda in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(LassoLogisticLearner::new(lambda).is_err());
        }
        let x = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let view = DesignView::from_column_major(&x, 3, 2).unwrap();
        let learner = LassoLogisticLearner::new(1.0).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        assert!(learner.fit(view, TargetView::new(&[0.0, 0.5, 1.0]), None, &ctx).is_err());
        assert!(learner.fit(view, TargetView::new(&[1.0, 1.0, 1.0]), None, &ctx).is_err());
        assert!(
            learner.fit(view, TargetView::new(&[0.0, 1.0, 1.0]), Some(&[1.0; 3]), &ctx).is_err()
        );
        assert_eq!(learner.task(), PredictionTask::BinaryProbability);
        assert!(learner.capabilities().binary_probability);
    }

    #[test]
    fn predictions_are_probabilities_and_the_fit_replays_bit_for_bit() {
        let (n, q) = (120usize, 4usize);
        let (x, y) = design(n, q, &[1.0, -1.0], 3);
        let a = coefficients(&x, n, q + 1, &y, 4.0).unwrap();
        let b = coefficients(&x, n, q + 1, &y, 4.0).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let view = DesignView::from_column_major(&x, n, q + 1).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let fitted = LassoLogisticLearner::new(4.0)
            .unwrap()
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));
        assert_eq!(fitted.provenance().spec, "lasso_logistic");
    }
}

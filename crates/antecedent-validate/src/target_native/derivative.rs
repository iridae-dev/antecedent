//! Derivative-native validation.
//!
//! A reported derivative (an analytic or automatic derivative of a fitted response, a local
//! slope, or a Jacobian row) is checked against the fitted response itself:
//!
//! * [`finite_difference_consistency`] compares it with a central finite difference of the
//!   response, Richardson-extrapolated from steps `h` and `h/2`. The extrapolation's own
//!   truncation error is estimated by `|D(h/2) − D(h)| / 3` and the rounding error by
//!   `ε·max|f| / h`; the discrepancy is standardized by
//!   `tolerance·(1 + |D|) + 4·truncation + rounding`, so a smooth response with a correct
//!   derivative passes and an inconsistent derivative fails at any step size where the
//!   difference quotient is accurate.
//! * [`local_linearity`] measures how far the response bends over the step at which the
//!   derivative is read: the second difference `κ = |f(x+h) − 2f(x) + f(x−h)| / h²` against
//!   the slope, as the relative error `κ·h / (2·|f'|)` of the linear prediction. A large
//!   value says the reported derivative describes the response only on a smaller scale than
//!   the one the claim is read at.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::joint::{JointStatistic, JointStatisticKind};
use super::{RefutationTarget, TargetRefutationReport, TargetRefuterKind};
use crate::ValidationError;

/// One evaluation point and the derivative reported there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivativeCheckPoint {
    /// Evaluation point.
    pub x: f64,
    /// Reported derivative at `x`.
    pub reported: f64,
}

/// Result of [`finite_difference_consistency`].
#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeConsistency {
    /// Evaluation points.
    pub points: Vec<f64>,
    /// Reported derivatives.
    pub reported: Vec<f64>,
    /// Richardson-extrapolated central differences.
    pub finite_difference: Vec<f64>,
    /// Estimated truncation error of each finite difference.
    pub truncation_error: Vec<f64>,
    /// Standardized discrepancy at each point (pass when every value is at most one).
    pub standardized: Vec<f64>,
    /// Step `h`.
    pub step: f64,
    /// Relative tolerance.
    pub tolerance: f64,
}

impl DerivativeConsistency {
    /// Largest standardized discrepancy.
    #[must_use]
    pub fn max_standardized(&self) -> f64 {
        self.standardized.iter().copied().fold(0.0, f64::max)
    }

    /// Whether every point is consistent.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.max_standardized() <= 1.0
    }

    /// As a target-native report on `target` (whose coordinates are the evaluation points).
    ///
    /// # Errors
    ///
    /// `target` does not have one coordinate per evaluation point.
    pub fn to_report(
        &self,
        target: &RefutationTarget,
    ) -> Result<TargetRefutationReport, ValidationError> {
        if target.dimension() != self.points.len() {
            return Err(ValidationError::NotApplicable {
                message: "derivative check: the target needs one coordinate per evaluation point",
            });
        }
        let value = self.max_standardized();
        let passed = self.passed();
        Ok(TargetRefutationReport {
            target: target.clone(),
            kind: TargetRefuterKind::FiniteDifferenceConsistency,
            original: self.reported.clone().into(),
            refuted: self.finite_difference.clone().into(),
            statistic: JointStatistic {
                kind: JointStatisticKind::MaxStandardizedDiscrepancy,
                value,
                df: self.points.len(),
                p_value: None,
            },
            passed,
            informative: true,
            replicates: 0,
            failure: (!passed).then(|| {
                Arc::from(format!(
                    "reported derivative differs from the finite difference of the fitted \
                     response by {value:.3} tolerance units (step {}, relative tolerance {})",
                    self.step, self.tolerance
                ))
            }),
        })
    }
}

/// Result of [`local_linearity`].
#[derive(Clone, Debug, PartialEq)]
pub struct LocalLinearity {
    /// Evaluation points.
    pub points: Vec<f64>,
    /// Central-difference slopes.
    pub slope: Vec<f64>,
    /// Second-difference curvatures.
    pub curvature: Vec<f64>,
    /// Relative error of the linear prediction over the step, per point.
    pub nonlinearity: Vec<f64>,
    /// Step `h`.
    pub step: f64,
    /// Largest acceptable relative nonlinearity.
    pub tolerance: f64,
}

impl LocalLinearity {
    /// Largest relative nonlinearity.
    #[must_use]
    pub fn max_nonlinearity(&self) -> f64 {
        self.nonlinearity.iter().copied().fold(0.0, f64::max)
    }

    /// Whether the response is locally linear at every point at this step.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.max_nonlinearity() <= self.tolerance
    }

    /// As a target-native report on `target`.
    ///
    /// # Errors
    ///
    /// `target` does not have one coordinate per evaluation point.
    pub fn to_report(
        &self,
        target: &RefutationTarget,
    ) -> Result<TargetRefutationReport, ValidationError> {
        if target.dimension() != self.points.len() {
            return Err(ValidationError::NotApplicable {
                message: "local linearity: the target needs one coordinate per evaluation point",
            });
        }
        let value = self.max_nonlinearity();
        let passed = self.passed();
        Ok(TargetRefutationReport {
            target: target.clone(),
            kind: TargetRefuterKind::LocalLinearity,
            original: self.slope.clone().into(),
            refuted: self.curvature.clone().into(),
            statistic: JointStatistic {
                kind: JointStatisticKind::MaxStandardizedDiscrepancy,
                value,
                df: self.points.len(),
                p_value: None,
            },
            passed,
            informative: true,
            replicates: 0,
            failure: (!passed).then(|| {
                Arc::from(format!(
                    "the response bends by {value:.3} of its slope over the step {} (tolerance \
                     {}); read the derivative on a smaller scale or report the curve",
                    self.step, self.tolerance
                ))
            }),
        })
    }
}

fn check_step(step: f64, tolerance: f64, points: usize) -> Result<(), ValidationError> {
    if !(step.is_finite() && step > 0.0 && tolerance.is_finite() && tolerance > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "derivative checks need a positive finite step and tolerance",
        });
    }
    if points == 0 {
        return Err(ValidationError::NotApplicable {
            message: "derivative checks need at least one evaluation point",
        });
    }
    Ok(())
}

fn evaluate(
    response: &mut impl FnMut(f64) -> Result<f64, ValidationError>,
    x: f64,
) -> Result<f64, ValidationError> {
    let value = response(x)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ValidationError::estimation_msg("the fitted response is not finite at a check point"))
    }
}

/// Compare reported derivatives with Richardson-extrapolated central differences of
/// `response`.
///
/// # Errors
///
/// A non-positive step or tolerance, no points, or a non-finite response value.
pub fn finite_difference_consistency(
    mut response: impl FnMut(f64) -> Result<f64, ValidationError>,
    points: &[DerivativeCheckPoint],
    step: f64,
    tolerance: f64,
) -> Result<DerivativeConsistency, ValidationError> {
    check_step(step, tolerance, points.len())?;
    let n = points.len();
    let mut out = DerivativeConsistency {
        points: Vec::with_capacity(n),
        reported: Vec::with_capacity(n),
        finite_difference: Vec::with_capacity(n),
        truncation_error: Vec::with_capacity(n),
        standardized: Vec::with_capacity(n),
        step,
        tolerance,
    };
    let half = 0.5 * step;
    for point in points {
        let x = point.x;
        let f_ph = evaluate(&mut response, x + step)?;
        let f_mh = evaluate(&mut response, x - step)?;
        let f_ph2 = evaluate(&mut response, x + half)?;
        let f_mh2 = evaluate(&mut response, x - half)?;
        let d_h = (f_ph - f_mh) / (2.0 * step);
        let d_h2 = (f_ph2 - f_mh2) / step;
        let richardson = (4.0 * d_h2 - d_h) / 3.0;
        let truncation = (d_h2 - d_h).abs() / 3.0;
        let magnitude = f_ph.abs().max(f_mh.abs()).max(f_ph2.abs()).max(f_mh2.abs());
        let rounding = 8.0 * f64::EPSILON * magnitude / step;
        let scale = tolerance * (1.0 + richardson.abs()) + 4.0 * truncation + rounding;
        out.points.push(x);
        out.reported.push(point.reported);
        out.finite_difference.push(richardson);
        out.truncation_error.push(truncation);
        out.standardized.push((point.reported - richardson).abs() / scale);
    }
    Ok(out)
}

/// Second-difference curvature of `response` relative to its slope over `step`.
///
/// The slope floor `slope_floor` keeps a flat response (zero slope) from dividing by zero:
/// the relative error is `κ·h / (2·max(|f'|, slope_floor))`.
///
/// # Errors
///
/// A non-positive step, tolerance or slope floor, no points, or a non-finite response value.
pub fn local_linearity(
    mut response: impl FnMut(f64) -> Result<f64, ValidationError>,
    points: &[f64],
    step: f64,
    tolerance: f64,
    slope_floor: f64,
) -> Result<LocalLinearity, ValidationError> {
    check_step(step, tolerance, points.len())?;
    if !(slope_floor.is_finite() && slope_floor > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "local linearity needs a positive slope floor",
        });
    }
    let n = points.len();
    let mut out = LocalLinearity {
        points: Vec::with_capacity(n),
        slope: Vec::with_capacity(n),
        curvature: Vec::with_capacity(n),
        nonlinearity: Vec::with_capacity(n),
        step,
        tolerance,
    };
    for &x in points {
        let f0 = evaluate(&mut response, x)?;
        let fp = evaluate(&mut response, x + step)?;
        let fm = evaluate(&mut response, x - step)?;
        let slope = (fp - fm) / (2.0 * step);
        let curvature = (fp - 2.0 * f0 + fm) / (step * step);
        out.points.push(x);
        out.slope.push(slope);
        out.curvature.push(curvature);
        out.nonlinearity.push(curvature.abs() * step / (2.0 * slope.abs().max(slope_floor)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(xs: &[f64], derivative: impl Fn(f64) -> f64) -> Vec<DerivativeCheckPoint> {
        xs.iter().map(|&x| DerivativeCheckPoint { x, reported: derivative(x) }).collect()
    }

    #[test]
    fn a_correct_analytic_derivative_is_consistent() {
        let xs = [-1.0, -0.3, 0.0, 0.7, 1.4];
        let check =
            finite_difference_consistency(|x| Ok(x.sin()), &points(&xs, f64::cos), 1e-2, 1e-6)
                .unwrap();
        assert!(check.passed(), "{:?}", check.standardized);
        let cubic = finite_difference_consistency(
            |x| Ok(x * x * x),
            &points(&xs, |x| 3.0 * x * x),
            1e-1,
            1e-6,
        )
        .unwrap();
        assert!(cubic.passed(), "{:?}", cubic.standardized);
    }

    #[test]
    fn an_inconsistent_derivative_fails() {
        let xs = [-1.0, 0.0, 1.0];
        let check = finite_difference_consistency(
            |x| Ok(x.sin()),
            &points(&xs, |x| x.cos() + 0.05),
            1e-2,
            1e-3,
        )
        .unwrap();
        assert!(!check.passed());
    }

    #[test]
    fn linear_responses_are_locally_linear_and_steep_curves_are_not() {
        let xs = [0.0, 0.5, 1.0];
        let linear = local_linearity(|x| Ok(2.0 * x + 1.0), &xs, 0.5, 0.1, 1e-12).unwrap();
        assert!(linear.max_nonlinearity() < 1e-12);
        assert!(linear.passed());
        let curved = local_linearity(|x| Ok((5.0 * x).exp()), &xs, 0.5, 0.1, 1e-12).unwrap();
        // For exp(c x) the ratio (f(x+h) - 2 f(x) + f(x-h)) / (f(x+h) - f(x-h)) is
        // (cosh(ch) - 1) / sinh(ch) = tanh(ch / 2) at every x: tanh(1.25) = 0.848 here (the
        // continuous approximation c h / 2 = 1.25 overstates it), far above the 0.1 bar.
        let expected = (1.25_f64).tanh();
        for value in &curved.nonlinearity {
            assert!((value - expected).abs() < 1e-12, "{value}");
        }
        assert!(!curved.passed());
    }
}

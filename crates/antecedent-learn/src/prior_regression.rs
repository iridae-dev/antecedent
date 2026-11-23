//! A linear regression nuisance regularized by a Gaussian coefficient prior.
//!
//! The fit is the posterior mode of a Gaussian linear model under a conjugate coefficient
//! prior `β ~ N(m, σ² V₀)`: it minimizes `Σ wᵢ (yᵢ − xᵢβ)² + (β − m)ᵀ V₀⁻¹ (β − m)`, so
//!
//! ```text
//! β̂ = (XᵀWX + V₀⁻¹)⁻¹ (XᵀWy + V₀⁻¹ m).
//! ```
//!
//! The residual variance cancels when the prior is on the conjugate scale. A prior whose
//! variances are on the absolute scale is converted with the fit's own residual-variance
//! estimate (the unpenalized least-squares residual variance of the same training rows), the
//! plug-in the Bayesian g-computation applies to such a prior. The prior shapes the fit and
//! never the identification of an effect: the learner only predicts a nuisance function.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

use antecedent_core::ExecutionContext;
use antecedent_prob::{CoefficientPrecision, PriorSet};
use antecedent_stats::solve_spd;

use crate::dense::{gather_physical, materialize_dense_colmajor};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{FittedPredictor, LearnerCapabilities, LearnerFactory, PredictionTask};

/// Smallest residual variance a plug-in conversion of an absolute-scale prior will use.
const SIGMA2_FLOOR: f64 = 1e-12;

/// Regression factory whose fit is the posterior mode under a Gaussian coefficient prior.
#[derive(Clone, Debug)]
pub struct GaussianPriorRegression {
    prior: PriorSet,
}

impl GaussianPriorRegression {
    /// A factory for `prior`, which must carry a valid Gaussian coefficient block.
    ///
    /// # Errors
    ///
    /// A prior without a Gaussian coefficient block, or with an invalid one.
    pub fn new(prior: PriorSet) -> Result<Self, LearnError> {
        let coef = prior.gaussian_coefficients().ok_or(LearnError::Unsupported {
            message: "a prior-regularized regression needs a Gaussian coefficient prior",
        })?;
        coef.validate()?;
        Ok(Self { prior })
    }

    /// Number of coefficients the prior covers (the design width the learner accepts).
    #[must_use]
    pub fn n_coefficients(&self) -> usize {
        self.prior.gaussian_coefficients().map_or(0, antecedent_prob::GaussianCoefficientPrior::len)
    }
}

/// `Σ wᵣ aᵣ bᵣ` over the rows (`w ≡ 1` without weights).
fn weighted_dot(a: &[f64], b: &[f64], weights: Option<&[f64]>) -> f64 {
    match weights {
        Some(w) => a.iter().zip(b).zip(w).map(|((x, y), w)| w * x * y).sum(),
        None => a.iter().zip(b).map(|(x, y)| x * y).sum(),
    }
}

/// `XᵀWX` (full symmetric, row-major) and `XᵀWy` of a column-major design.
fn weighted_normal_equations(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0; ncols * ncols];
    let mut xty = vec![0.0; ncols];
    for i in 0..ncols {
        let ci = &design[i * nrows..(i + 1) * nrows];
        for j in i..ncols {
            let cj = &design[j * nrows..(j + 1) * nrows];
            let acc = weighted_dot(ci, cj, weights);
            gram[i * ncols + j] = acc;
            gram[j * ncols + i] = acc;
        }
        xty[i] = weighted_dot(ci, y, weights);
    }
    (gram, xty)
}

/// Unpenalized weighted residual variance `RSS / (n − p)` of the same rows.
#[allow(
    clippy::cast_precision_loss,
    reason = "the degrees of freedom are a row count far below 2^53"
)]
fn plug_in_sigma2(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    gram: &[f64],
    xty: &[f64],
) -> Result<f64, LearnError> {
    if nrows <= ncols {
        return Err(LearnError::Backend(
            "a prior on the absolute coefficient scale needs more training rows than \
             coefficients to estimate the residual variance it is converted with"
                .to_string(),
        ));
    }
    let beta = solve_spd(gram, ncols, xty).ok_or_else(|| {
        LearnError::Backend(
            "the training design is rank deficient, so the residual variance a prior on the \
             absolute coefficient scale is converted with cannot be estimated"
                .to_string(),
        )
    })?;
    let mut rss = 0.0;
    for r in 0..nrows {
        let mut fit = 0.0;
        for (c, b) in beta.iter().enumerate() {
            fit += design[c * nrows + r] * b;
        }
        let w = weights.map_or(1.0, |w| w[r]);
        rss += w * (y[r] - fit) * (y[r] - fit);
    }
    Ok((rss / (nrows - ncols) as f64).max(SIGMA2_FLOOR))
}

impl LearnerFactory for GaussianPriorRegression {
    fn task(&self) -> PredictionTask {
        PredictionTask::Regression
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: true,
            sample_weights: true,
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
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let gathered_w = match weights {
            Some(w) => {
                if w.len() != x.physical_nrows() {
                    return Err(LearnError::Shape { message: "weights length != physical rows" });
                }
                let gathered = gather_physical(w, x, nrows)?;
                if gathered.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(LearnError::Shape {
                        message: "prior weights must be finite and non-negative",
                    });
                }
                Some(gathered)
            }
            None => None,
        };
        if gathered_y.iter().chain(design.iter()).any(|v| !v.is_finite()) {
            return Err(LearnError::NonFinite {
                message: "a prior-regularized regression needs a finite design and target",
            });
        }
        let weights = gathered_w.as_deref();
        let (mut gram, mut xty) =
            weighted_normal_equations(&design, nrows, ncols, &gathered_y, weights);
        let prior: Cow<'_, PriorSet> = if self.prior.absolute_coefficient_scale().is_some() {
            let sigma2 = plug_in_sigma2(&design, nrows, ncols, &gathered_y, weights, &gram, &xty)?;
            match self.prior.resolve_absolute_coefficient_scale(sigma2)? {
                Some(resolved) => Cow::Owned(resolved),
                None => Cow::Borrowed(&self.prior),
            }
        } else {
            Cow::Borrowed(&self.prior)
        };
        let coef = prior.gaussian_coefficients().ok_or(LearnError::Unsupported {
            message: "a prior-regularized regression needs a Gaussian coefficient prior",
        })?;
        if coef.len() != ncols {
            return Err(LearnError::Shape {
                message: "the coefficient prior does not cover the design columns",
            });
        }
        let precision: CoefficientPrecision = prior.coefficient_precision(coef)?;
        if precision.len() != ncols {
            return Err(LearnError::Shape {
                message: "the coefficient precision does not cover the design columns",
            });
        }
        let mut prior_pull = vec![0.0; ncols];
        precision.mul_into(&coef.mean, &mut prior_pull);
        for i in 0..ncols {
            for j in 0..ncols {
                gram[i * ncols + j] += precision.get(i, j);
            }
            xty[i] += prior_pull[i];
        }
        let beta = solve_spd(&gram, ncols, &xty).ok_or_else(|| {
            LearnError::Backend(
                "the prior-regularized normal equations are not positive definite".to_string(),
            )
        })?;
        if beta.iter().any(|v| !v.is_finite()) {
            return Err(LearnError::NonFinite {
                message: "the prior-regularized coefficients must be finite",
            });
        }
        Ok(Box::new(crate::coefficient::CoefficientPredictor::new(
            beta,
            false,
            "gaussian_prior_linear",
            "antecedent-learn",
            env!("CARGO_PKG_VERSION"),
        )))
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "the tests build small designs from small indices and compare against closed forms"
)]
mod tests {
    use std::sync::Arc;

    use antecedent_prob::{GaussianCoefficientPrior, PriorSpec};

    use super::*;

    fn prior_set(mean: &[f64], variance: &[f64]) -> PriorSet {
        PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                mean: Arc::from(mean),
                variance: Arc::from(variance),
            })],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        }
    }

    /// Coefficients of a fitted predictor, read by predicting the standard basis rows.
    fn coefficients(fit: &dyn FittedPredictor, ncols: usize) -> Vec<f64> {
        let mut basis = vec![0.0; ncols * ncols];
        for c in 0..ncols {
            basis[c * ncols + c] = 1.0;
        }
        let view = DesignView::from_column_major(&basis, ncols, ncols).unwrap();
        let mut out = vec![0.0; ncols];
        fit.predict(view, &mut out, &ExecutionContext::for_tests(1)).unwrap();
        out
    }

    /// `[1 | x]` design with `x = 0, 1, 2, 3` and a fixed outcome.
    fn small_problem() -> (Vec<f64>, Vec<f64>) {
        let x = [0.0, 1.0, 2.0, 3.0];
        let mut design = vec![1.0; 4];
        design.extend_from_slice(&x);
        (design, vec![1.0, 2.5, 3.0, 5.5])
    }

    #[test]
    fn the_fit_is_the_closed_form_map_of_the_normal_equations() {
        let (design, y) = small_problem();
        let (mean, variance) = ([0.5, 2.0], [4.0, 0.25]);
        let learner = GaussianPriorRegression::new(prior_set(&mean, &variance)).unwrap();
        let view = DesignView::from_column_major(&design, 4, 2).unwrap();
        let fit =
            learner.fit(view, TargetView::new(&y), None, &ExecutionContext::for_tests(1)).unwrap();
        let beta = coefficients(fit.as_ref(), 2);
        // Independent 2 x 2 solve of (X'X + V0^-1) b = X'y + V0^-1 m.
        let (a00, a01, a11) = (4.0 + 1.0 / variance[0], 6.0, 14.0 + 1.0 / variance[1]);
        let (b0, b1) = (
            y.iter().sum::<f64>() + mean[0] / variance[0],
            (1.0 * 2.5 + 2.0 * 3.0 + 3.0 * 5.5) + mean[1] / variance[1],
        );
        let det = a00 * a11 - a01 * a01;
        let expected = [(a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det];
        for (got, want) in beta.iter().zip(expected) {
            assert!((got - want).abs() <= 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn a_very_wide_prior_recovers_least_squares_and_a_very_tight_one_its_mean() {
        let (design, y) = small_problem();
        let view = DesignView::from_column_major(&design, 4, 2).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let wide = GaussianPriorRegression::new(prior_set(&[7.0, -3.0], &[1e12, 1e12])).unwrap();
        let beta =
            coefficients(wide.fit(view, TargetView::new(&y), None, &ctx).unwrap().as_ref(), 2);
        // OLS of y on [1, x]: slope = Sxy / Sxx, intercept = ybar - slope xbar.
        let (xbar, ybar) = (1.5, y.iter().sum::<f64>() / 4.0);
        let sxy: f64 =
            [0.0, 1.0, 2.0, 3.0].iter().zip(&y).map(|(x, y)| (x - xbar) * (y - ybar)).sum();
        let slope = sxy / 5.0;
        assert!((beta[1] - slope).abs() <= 1e-6, "{beta:?}");
        assert!((beta[0] - (ybar - slope * xbar)).abs() <= 1e-6, "{beta:?}");
        let tight = GaussianPriorRegression::new(prior_set(&[7.0, -3.0], &[1e-12, 1e-12])).unwrap();
        let beta =
            coefficients(tight.fit(view, TargetView::new(&y), None, &ctx).unwrap().as_ref(), 2);
        assert!((beta[0] - 7.0).abs() <= 1e-6 && (beta[1] + 3.0).abs() <= 1e-6, "{beta:?}");
    }

    #[test]
    fn row_weights_enter_the_normal_equations_like_repeated_rows() {
        let (design, y) = small_problem();
        let view = DesignView::from_column_major(&design, 4, 2).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let learner = GaussianPriorRegression::new(prior_set(&[0.0, 1.0], &[2.0, 2.0])).unwrap();
        let weights = [1.0, 2.0, 1.0, 1.0];
        let weighted = coefficients(
            learner.fit(view, TargetView::new(&y), Some(&weights), &ctx).unwrap().as_ref(),
            2,
        );
        // Row 1 repeated once: the same normal equations.
        let repeated_design = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0];
        let repeated_y = [1.0, 2.5, 2.5, 3.0, 5.5];
        let repeated_view = DesignView::from_column_major(&repeated_design, 5, 2).unwrap();
        let expected = coefficients(
            learner.fit(repeated_view, TargetView::new(&repeated_y), None, &ctx).unwrap().as_ref(),
            2,
        );
        for (got, want) in weighted.iter().zip(&expected) {
            assert!((got - want).abs() <= 1e-12, "{weighted:?} vs {expected:?}");
        }
    }

    #[test]
    fn a_prior_that_does_not_cover_the_design_columns_is_refused() {
        let (design, y) = small_problem();
        let view = DesignView::from_column_major(&design, 4, 2).unwrap();
        let learner = GaussianPriorRegression::new(prior_set(&[0.0], &[1.0])).unwrap();
        let refused = learner.fit(view, TargetView::new(&y), None, &ExecutionContext::for_tests(1));
        assert!(matches!(refused, Err(LearnError::Shape { .. })));
        assert!(GaussianPriorRegression::new(PriorSet::new()).is_err());
    }

    #[test]
    fn an_absolute_scale_prior_is_converted_with_the_training_residual_variance() {
        let (design, y) = small_problem();
        let view = DesignView::from_column_major(&design, 4, 2).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut absolute = prior_set(&[0.5, 2.0], &[4.0, 0.25]);
        absolute.mark_absolute_coefficient_scale(&[0, 1]);
        let fitted = GaussianPriorRegression::new(absolute)
            .unwrap()
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        // The training residual variance of the unpenalized fit: RSS / (n - p).
        let (xbar, ybar) = (1.5, y.iter().sum::<f64>() / 4.0);
        let sxy: f64 =
            [0.0, 1.0, 2.0, 3.0].iter().zip(&y).map(|(x, y)| (x - xbar) * (y - ybar)).sum();
        let slope = sxy / 5.0;
        let intercept = ybar - slope * xbar;
        let rss: f64 = [0.0, 1.0, 2.0, 3.0]
            .iter()
            .zip(&y)
            .map(|(x, y)| (y - intercept - slope * x).powi(2))
            .sum();
        let sigma2 = rss / 2.0;
        let conjugate =
            GaussianPriorRegression::new(prior_set(&[0.5, 2.0], &[4.0 / sigma2, 0.25 / sigma2]))
                .unwrap()
                .fit(view, TargetView::new(&y), None, &ctx)
                .unwrap();
        let (got, want) = (coefficients(fitted.as_ref(), 2), coefficients(conjugate.as_ref(), 2));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-9, "{got:?} vs {want:?}");
        }
    }
}

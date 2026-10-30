//! Split out of the mechanism registry. Behavior is unchanged.
use super::{
    data_rss, demean_by_group, fit_linear_gaussian, group_structure_usable, residual_sigma,
};

use std::sync::Arc;

use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm,
    fit_glm_ridge, invert_square,
};

use super::super::cv::FitFrame;
use crate::compile::MechanismSlot;
use crate::error::ModelError;

// The shared raw-coefficient prior is N(0,tau²). Estimate its second
// moment after subtracting each slope's actual sampling variance, accounting
// for an unpenalized intercept and correlations between the parent columns.
fn linear_shrinkage(
    frame: &FitFrame<'_>,
    coeffs: &[f64],
    sigma2: f64,
    n: usize,
) -> Result<f64, ModelError> {
    let p = coeffs.len();
    if p == 0 {
        return Ok(0.0);
    }
    let means: Vec<f64> =
        frame.parents.iter().map(|col| col[..n].iter().sum::<f64>() / n as f64).collect();
    let mut gram = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            gram[i * p + j] = (0..n)
                .map(|r| (frame.parents[i][r] - means[i]) * (frame.parents[j][r] - means[j]))
                .sum();
        }
    }
    let inverse = invert_square(&gram, p).ok_or_else(|| ModelError::Numerical {
        message: "hierarchical EB slope information is singular".into(),
    })?;
    let sampling = (0..p).map(|j| inverse[j * p + j]).sum::<f64>() * sigma2 / p as f64;
    let mean_b2 = coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64;
    if !sampling.is_finite()
        || !mean_b2.is_finite()
        || sampling < 0.0
        || !sigma2.is_finite()
        || sigma2 < 0.0
    {
        return Err(ModelError::Numerical {
            message: "hierarchical EB moments are nonfinite".into(),
        });
    }
    let tau2 = (mean_b2 - sampling).max(1e-8);
    Ok((sigma2 / tau2).clamp(1e-6, 1e6))
}

/// Empirical-Bayes hierarchical linear: estimate τ² / λ from OLS; with a `UnitId`
/// column, the slopes come from the within-unit (fixed-effects) regression.
pub(super) fn fit_hierarchical_linear(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let ols = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let MechanismSlot::LinearGaussian { coeffs: ols_coeffs, sigma: ols_sigma, .. } = ols else {
        return Err(ModelError::Unsupported { message: "hierarchical base fit failed".into() });
    };
    let p = ols_coeffs.len();
    let mut lambda = linear_shrinkage(frame, &ols_coeffs, ols_sigma * ols_sigma, n)?;

    // Optional UnitId random intercept. The within estimator demeans the outcome *and*
    // every parent by unit: demeaning only `y` regresses it on a parent that still
    // carries its between-unit variance, attenuating the slope by
    // `Var_within(x)/Var(x)`. The between-unit spread of the outcome is not
    // discarded: the residual σ below is measured on the population-intercept mean
    // function, so it carries that spread, exactly as abduction (`y − mean`) and the
    // log-density will see it.
    let grouped = frame.groups.as_deref().filter(|g| group_structure_usable(g));
    let slot = if let Some(groups) = grouped {
        let y_work = demean_by_group(y, groups);
        let within = frame
            .with_parents(frame.parents.iter().map(|c| demean_by_group(&c[..n], groups)).collect());
        // The within fit estimates group intercepts as nuisance parameters.
        // Its EB noise variance uses n - groups - p degrees of freedom; the
        // returned population-predictive residual scale is computed separately.
        let group_count = groups.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
        let df = n
            .checked_sub(group_count)
            .and_then(|v| v.checked_sub(p))
            .filter(|&v| v > 0)
            .ok_or_else(|| ModelError::Numerical {
                message: "hierarchical within fit has no residual degrees of freedom".into(),
            })?;
        let ols2 = fit_linear_gaussian(&within, &y_work, backend, ls_ws, 0.0)?;
        let MechanismSlot::LinearGaussian { intercept, coeffs, .. } = ols2 else {
            return Err(ModelError::Unsupported {
                message: "hierarchical within base fit failed".into(),
            });
        };
        let within_sigma2 = data_rss(&within, &y_work, intercept, &coeffs, None) / df as f64;
        lambda = linear_shrinkage(&within, &coeffs, within_sigma2, n)?;
        fit_linear_gaussian(&within, &y_work, backend, ls_ws, lambda)?
    } else {
        fit_linear_gaussian(frame, y, backend, ls_ws, lambda)?
    };
    let MechanismSlot::LinearGaussian { intercept, coeffs, sigma } = slot else {
        return Ok(slot);
    };
    let (intercept, sigma) = if grouped.is_some() {
        // Population intercept, then σ on the same mean function every downstream
        // consumer evaluates.
        let mean_y = y.iter().sum::<f64>() / n.max(1) as f64;
        let mut intercept = mean_y;
        for (c, col) in frame.parents.iter().enumerate() {
            intercept -= coeffs[c] * col[..n].iter().sum::<f64>() / n.max(1) as f64;
        }
        let rss = data_rss(frame, y, intercept, &coeffs, None);
        (intercept, residual_sigma(rss, n, 1 + p, y))
    } else {
        (intercept, sigma)
    };
    Ok(MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage: lambda })
}

/// Hierarchical Bernoulli logit with always-on empirical-Bayes ridge.
///
/// EB λ is estimated from unpenalized logit coefficients and their inverse
/// Fisher information, then applied on the summed likelihood. The intercept
/// is left unpenalized; a separated or singular pilot cannot identify EB.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
/// Hierarchical Bernoulli-logit fit.
pub(super) fn fit_hierarchical_glm(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    #[allow(
        clippy::float_cmp,
        reason = "a binary outcome is coded exactly 0/1, so exact equality is the membership test"
    )]
    let binary = y.iter().all(|&yi| yi == 0.0 || yi == 1.0);
    if !binary {
        return Err(ModelError::Unsupported {
            message: "HierarchicalGlm requires binary {0,1} outcomes".into(),
        });
    }
    // A declared `UnitId` grouping is a statement about the rows' structure. This family
    // shrinks slopes; it has no unit-level term (a within-unit demeaning is not a logit
    // estimator), so under a usable grouping it refuses instead of fitting as if the units
    // had not been declared.
    if frame.groups.as_deref().is_some_and(group_structure_usable) {
        return Err(ModelError::Unsupported {
            message: "HierarchicalGlm has no unit-level effect, so it cannot honour the declared \
                      UnitId grouping; use the Discrete family, or HierarchicalLinear for a \
                      continuous outcome"
                .into(),
        });
    }
    let p = frame.n_parents();
    let ncols = 1 + p;
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, col) in frame.parents.iter().enumerate() {
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    let opts = GlmOptions::default();
    let lambda = if p == 0 {
        1.0
    } else {
        let pilot = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols, y },
            &backend,
            ls_ws,
            &opts.without_separation_ridge(),
        )
        .map_err(|e| ModelError::Numerical { message: e.to_string() })?;
        if !pilot.converged
            || pilot.separated
            || pilot.penalized
            || pilot.coefficients.iter().any(|v| !v.is_finite())
        {
            return Err(ModelError::Numerical {
                message: "hierarchical GLM EB requires a finite unseparated logit fit".into(),
            });
        }
        let mut fisher = vec![0.0; ncols * ncols];
        for r in 0..n {
            let eta: f64 = (0..ncols).map(|j| x[j * n + r] * pilot.coefficients[j]).sum();
            let probability = GlmFamily::BinomialLogit.mean_from_eta(eta);
            let weight = probability * (1.0 - probability);
            for i in 0..ncols {
                for j in 0..ncols {
                    fisher[i * ncols + j] += weight * x[i * n + r] * x[j * n + r];
                }
            }
        }
        let inverse = invert_square(&fisher, ncols).ok_or_else(|| ModelError::Numerical {
            message: "hierarchical GLM EB information is singular".into(),
        })?;
        let variance = (1..ncols).map(|j| inverse[j * ncols + j]).sum::<f64>() / p as f64;
        let mean_b2 = pilot.coefficients[1..].iter().map(|b| b * b).sum::<f64>() / p as f64;
        if !variance.is_finite() || variance < 0.0 || !mean_b2.is_finite() {
            return Err(ModelError::Numerical {
                message: "hierarchical GLM EB moments are nonfinite".into(),
            });
        }
        let tau2 = (mean_b2 - variance).max(1e-8);
        (1.0 / tau2).clamp(1e-4, 1e3)
    };
    let fit = fit_glm_ridge(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols, y },
        &backend,
        ls_ws,
        &opts,
        lambda,
    )
    .map_err(|e| ModelError::Numerical { message: e.to_string() })?;
    if !fit.converged {
        return Err(ModelError::Numerical {
            message: "hierarchical GLM logit did not converge".into(),
        });
    }
    // Encode as 2-category Discrete with baseline-category logits (cat0 = 0, cat1 = β).
    let mut logit_coeffs = vec![0.0; 2 * ncols];
    logit_coeffs[ncols..].copy_from_slice(&fit.coefficients[..ncols]);
    #[allow(
        clippy::float_cmp,
        reason = "a binary outcome is coded exactly 0/1, so exact equality is the membership test"
    )]
    let n1 = y.iter().filter(|&&yi| yi == 1.0).count() as f64;
    let p1 = n1 / n.max(1) as f64;
    Ok(MechanismSlot::Discrete {
        support: Arc::from([0.0, 1.0]),
        probs: Arc::from([1.0 - p1, p1]),
        logit_coeffs: Some(Arc::from(logit_coeffs)),
    })
}

#[cfg(test)]
mod eb_tests {
    use super::*;
    use crate::registry::ValidationSplits;
    use std::borrow::Cow;

    fn frame(x: &[f64], groups: Option<Vec<u32>>) -> FitFrame<'_> {
        FitFrame {
            parents: vec![Cow::Borrowed(x)],
            groups,
            n: x.len(),
            splits: ValidationSplits::Shuffled,
        }
    }

    #[test]
    fn linear_eb_uses_raw_slope_sampling_variance_and_within_df() {
        let x = [-2.0, -1.0, 1.0, 2.0];
        let f = frame(&x, None);
        let lambda = linear_shrinkage(&f, &[2.0], 2.0, x.len()).unwrap();
        assert!((lambda - 2.0 / (4.0 - 2.0 / 10.0)).abs() < 1e-12);
        let zero_df = frame(&x, Some(vec![0, 0, 1, 2]));
        assert!(
            fit_hierarchical_linear(
                &zero_df,
                &[0.0, 1.0, 2.0, 3.0],
                FaerBackend,
                &mut LeastSquaresWorkspace::default()
            )
            .is_err()
        );
        let constants = [1.0; 4];
        let degenerate = frame(&constants, None);
        assert!(linear_shrinkage(&degenerate, &[2.0], 1.0, 4).is_err());
    }

    #[test]
    fn logit_eb_uses_logit_fisher_and_refuses_separation() {
        let x = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        let y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let fit = fit_hierarchical_glm(
            &frame(&x, None),
            &y,
            FaerBackend,
            &mut LeastSquaresWorkspace::default(),
        )
        .unwrap();
        let MechanismSlot::Discrete { logit_coeffs: Some(beta), .. } = fit else {
            panic!("expected logit");
        };
        let slope = beta[3];
        let mle = 3.0_f64.ln();
        let lambda = 1.0 / (mle * mle - 2.0 / 3.0);
        let p = GlmFamily::BinomialLogit.mean_from_eta(slope);
        assert!((6.0 - 8.0 * p - lambda * slope).abs() < 1e-6);
        let separated = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        assert!(
            fit_hierarchical_glm(
                &frame(&x, None),
                &separated,
                FaerBackend,
                &mut LeastSquaresWorkspace::default()
            )
            .is_err()
        );
    }
}

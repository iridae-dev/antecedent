// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sensitivity-grid evaluation over kernel residuals.

use super::{
    ExecutionContext, PerturbMoments, RefutationProblem, RefutationReport, ValidationError,
    estimate_vanished, fill_gaussian, grid_failure_text, grid_report, grid_robustness_value,
    orthogonalize_confounder, probed_partial_r2, residual_ols_ate, robustness_passes, sample_sd,
};

/// Grid, threshold, report id and confounder stream of one residual-grid sensitivity run.
pub(super) struct ResidualGrid<'a> {
    pub(super) partial_r2_grid: &'a [f64],
    pub(super) pass_threshold: f64,
    pub(super) refuter: &'static str,
    pub(super) stream: u64,
}

/// The partial-R² grid on the residualized treatment and outcome (`n` complete cases): the
/// part of the nonparametric refuters that does not depend on the smoother.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the refutation reports its grid size as u32, and a partial-R-squared grid is far below 2^32 points"
)]
pub(super) fn residual_grid_report(
    problem: &RefutationProblem<'_>,
    ctx: &ExecutionContext,
    residuals: (Vec<f64>, Vec<f64>, usize),
    spec: &ResidualGrid<'_>,
) -> Result<RefutationReport, ValidationError> {
    let (t_res, y_res, n) = residuals;
    let (_, _, treatment_delta) = antecedent_estimate::prepare::treatment_contrast(
        &problem.query.active,
        &problem.query.control,
    )?;
    let residual_ate = residual_ols_ate(&t_res, &y_res);
    // The check perturbs the *residual* slope, which need not share the published
    // estimate's sign; a tipping value for a different effect says nothing about it.
    // (`residual_ate` is per unit of treatment; the published estimate is per contrast.)
    let residual_effect = residual_ate * treatment_delta;
    if !residual_effect.is_finite()
        || (residual_effect != 0.0
            && problem.original.ate != 0.0
            && residual_effect.signum() != problem.original.ate.signum())
    {
        return Err(ValidationError::NotApplicable {
            message: "the kernel-residualized effect is undefined or has the opposite sign \
                          of the published estimate, so its tipping value does not describe it",
        });
    }
    let sd_t = sample_sd(&t_res);
    let sd_y = sample_sd(&y_res);
    if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "nonparametric sensitivity requires positive residual variation in \
                          treatment and outcome after kernel residualization",
        });
    }
    let mut u = vec![0.0; n];
    fill_gaussian(&mut u, ctx, spec.stream);
    orthogonalize_confounder(&mut u, &vec![true; n], &[t_res.clone(), y_res.clone()]);

    let mut sorted_grid = spec.partial_r2_grid.to_vec();
    sorted_grid.sort_by(f64::total_cmp);
    let original_sign = residual_ate.signum();
    // Worst-case orientation, as in `run_grid`: load U on Y against the observed sign.
    let dir = if residual_ate >= 0.0 { -1.0 } else { 1.0 };
    let moments = PerturbMoments::new(&t_res, &y_res, &u);
    let mut last_ate = residual_ate;
    let mut explained_away_at = None;
    for &r in &sorted_grid {
        let scale = (r / (1.0 - r)).sqrt();
        // `t' = t + a u`, `y' = y + b u`. The OLS slope is a function of the
        // five raw second moments, so the grid does not rebuild the series.
        last_ate = moments.slope(scale * sd_t, dir * scale * sd_y);
        if !last_ate.is_finite() {
            return Err(ValidationError::NotApplicable {
                message: "nonparametric sensitivity has an undefined perturbed residual slope",
            });
        }
        #[allow(
            clippy::float_cmp,
            reason = "signum returns exactly +/-1 (or NaN), so comparing two signs for equality is exact"
        )]
        let sign_flipped = last_ate.signum() != original_sign;
        if estimate_vanished(last_ate, residual_ate) || sign_flipped {
            explained_away_at = Some(r);
            break;
        }
    }
    let robustness_value = grid_robustness_value(explained_away_at);
    let passed = robustness_passes(
        probed_partial_r2(spec.partial_r2_grid),
        robustness_value,
        spec.pass_threshold,
    );
    Ok(grid_report(
        spec.refuter,
        problem.original.ate,
        last_ate * treatment_delta,
        robustness_value,
        passed,
        || {
            grid_failure_text(
                "nonparametric residual effect",
                "partial R²",
                robustness_value,
                spec.pass_threshold,
            )
        },
        spec.partial_r2_grid.len() as u32,
    ))
}

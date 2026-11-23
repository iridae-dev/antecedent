//! A fitted GLM's immutable coefficient law and design for same-row standardization.
//! Target weights change averaging and its gradient, never the fitted score covariance.

use super::{GlmAdjustmentAte, PreparedGlmProblem};
use crate::EstimationError;
use antecedent_stats::GlmFamily;
use std::sync::Arc;

/// Retained unpenalized GLM coefficients and their estimator-specific covariance.
/// This is a fitted mean model, not a doubly robust score table or an outcome law.
#[derive(Clone, Debug)]
pub struct RetainedGlmPoint {
    matrix: Arc<[f64]>,
    nrows: usize,
    ncols: usize,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    family: GlmFamily,
    treatment_column: usize,
    active: f64,
    control: f64,
}

impl RetainedGlmPoint {
    /// Complete-case rows the target weights must cover, in frozen source order.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.nrows
    }

    /// Standardized contrast and its delta-method standard error, conditional on
    /// the supplied target law. No fit, residual resampling, or propensity model runs.
    ///
    /// # Errors
    /// Weight shape, invalid weights, an empty target, or non-finite arithmetic.
    pub fn retarget(&self, weights: &[f64]) -> Result<(f64, f64), EstimationError> {
        if weights.len() != self.nrows || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(EstimationError::data_msg(
                "adjusted retarget weights must be finite, non-negative and aligned with the frozen complete-case rows",
            ));
        }
        let scale = weights.iter().copied().fold(0.0_f64, f64::max);
        if scale <= 0.0 {
            return Err(EstimationError::data_msg("adjusted retarget needs positive target mass"));
        }
        let weights: Vec<f64> = weights.iter().map(|w| w / scale).collect();
        let contrast = crate::gcomp::GcompContrast {
            family: self.family,
            x_colmajor: &self.matrix,
            nrows: self.nrows,
            ncols: self.ncols,
            t_col: self.treatment_column,
            active: self.active,
            control: self.control,
        };
        let (value, gradient) = super::variance::gcomp_mean_and_gradient(
            &contrast,
            &self.coefficients,
            &super::variance::GcompTarget {
                treatment: &[],
                population: &antecedent_core::TargetPopulation::AllObserved,
                weights: Some(&weights),
            },
        );
        let mut variance = 0.0;
        let mut magnitude = 0.0;
        for (i, gi) in gradient.iter().enumerate() {
            for (j, gj) in gradient.iter().enumerate() {
                let term = gi * self.covariance[i * self.ncols + j] * gj;
                variance += term;
                magnitude += term.abs();
            }
        }
        if !value.is_finite() || !variance.is_finite() || variance < -1e-10 * magnitude {
            return Err(EstimationError::stats_msg(
                "adjusted retarget has a non-finite value or invalid delta covariance",
            ));
        }
        Ok((value, variance.max(0.0).sqrt()))
    }
}

pub(super) fn retain_point(
    fitter: &GlmAdjustmentAte,
    problem: &PreparedGlmProblem,
    coefficients: Vec<f64>,
    deviance: f64,
    nb_alpha: f64,
) -> Result<Option<RetainedGlmPoint>, EstimationError> {
    use super::variance::{adjustment_glm_covariance, glm_score_components, sandwich_cov_matrix};
    use crate::se::{AnalyticSeKind, SeSpec};
    let n = problem.design.nrows;
    let p = problem.design.ncols;
    let covariance = if fitter.inference.se_kind == AnalyticSeKind::Homoskedastic {
        adjustment_glm_covariance(
            problem.family,
            &problem.design.matrix,
            n,
            p,
            &coefficients,
            deviance,
            nb_alpha,
        )
    } else {
        let (mut scores, mut fisher) = glm_score_components(
            problem.family,
            &problem.design.matrix,
            n,
            p,
            &coefficients,
            &problem.design.outcome,
            nb_alpha,
        );
        if let Some(sampling) = &problem.sampling {
            for ((score, fisher), weight) in
                scores.iter_mut().zip(fisher.iter_mut()).zip(sampling.weights.iter())
            {
                *score *= weight;
                *fisher *= weight;
            }
        }
        let source_clusters = fitter
            .inference
            .source_cluster_ids
            .as_deref()
            .map(|ids| crate::se::retained_row_labels(ids, &problem.design.row_selection))
            .transpose()?;
        let clusters = source_clusters.as_deref().or(fitter.inference.cluster_ids.as_deref());
        sandwich_cov_matrix(
            SeSpec {
                kind: fitter.inference.se_kind,
                cluster_ids: clusters,
                multiway_ids: fitter.inference.multiway_ids.as_deref(),
                panel_times: fitter.inference.panel_times.as_deref(),
            },
            &problem.design.matrix,
            n,
            p,
            &scores,
            &fisher,
        )?
    };
    let Some(covariance) = covariance
        .filter(|cov| p.checked_mul(p) == Some(cov.len()) && cov.iter().all(|v| v.is_finite()))
    else {
        return Ok(None);
    };
    let treatment_column = problem
        .design
        .treatment_column()
        .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
    Ok(Some(RetainedGlmPoint {
        matrix: Arc::clone(&problem.design.matrix),
        nrows: n,
        ncols: p,
        coefficients,
        covariance,
        family: problem.family,
        treatment_column,
        active: problem.active,
        control: problem.control,
    }))
}

//! Propensity diagnostics shared by refuters.

use antecedent_data::DataError;
use antecedent_estimate::{IpwTarget, OverlapPolicy, OverlapReport};
use antecedent_stats::{
    FaerBackend, GlmOptions, PropensityFit, PropensityWorkspace, fit_propensity_diagnostic,
};

use super::{RefutationProblem, ValidationError};

/// Scores / treatment / optional outcome from a diagnostic propensity fit.
pub(crate) struct DiagnosticPropensityColumns {
    /// Fitted propensity scores.
    pub scores: Vec<f64>,
    /// Treatment column (complete cases).
    pub treatment: Vec<f64>,
    /// Outcome column when requested.
    pub outcome: Option<Vec<f64>>,
    /// Why the diagnostic logistic fit cannot be trusted as a propensity model, if it cannot.
    pub defect: Option<&'static str>,
}

/// A diagnostic logistic fit that separated the arms or failed to converge is not a usable
/// propensity model, and the ridge-regularized scores it returns shrink exactly the extreme
/// values that are the evidence of a positivity violation. It is reported as the violation.
pub(crate) fn propensity_fit_defect(glm: &antecedent_stats::GlmFit) -> Option<&'static str> {
    if glm.separated {
        Some(
            "the diagnostic propensity model (quasi-)completely separates the treatment arms: \
             the covariates almost perfectly predict treatment, a positivity violation",
        )
    } else if !glm.converged {
        Some(
            "the diagnostic propensity model did not converge, which happens under near-perfect \
             prediction of treatment (a positivity violation) or a degenerate design",
        )
    } else {
        None
    }
}

/// Whether the complete treatment sample satisfies the binary 0/1 contract.
///
/// Classification uses the same complete-case mask as estimation (treatment,
/// outcome, adjustment, modifiers). An empty remainder is not binary: `all()`
/// on no rows would otherwise vacuously accept the logistic path.
#[allow(
    clippy::float_cmp,
    reason = "a binary treatment is coded exactly 0.0 or 1.0; values merely near 0 or 1 are not binary"
)]
pub(crate) fn binary_treatment(problem: &RefutationProblem<'_>) -> Result<bool, ValidationError> {
    let ids = problem.estimation_ids(true);
    let mask = match problem.data.complete_case_mask(&ids) {
        Ok(mask) => mask,
        Err(DataError::EmptySelection { .. }) => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    let treatment = problem.data.float64_masked(problem.treatment(), &mask)?;
    if treatment.is_empty() {
        return Ok(false);
    }
    Ok(treatment.iter().all(|&value| value == 0.0 || value == 1.0))
}

/// Diagnostic-only logistic propensity on treatment + adjustment covariates.
///
/// Used by overlap / Riesz validators when the original estimate has no propensity report.
/// The complete-case mask always matches estimation (treatment, outcome, adjustment,
/// modifiers) so a non-0/1 treatment value on a row dropped for missing Y cannot
/// leak into the binary propensity path.
pub(crate) fn fit_diagnostic_propensity(
    problem: &RefutationProblem<'_>,
    glm_options: &GlmOptions,
    include_outcome_in_mask: bool,
    propensity: &mut PropensityWorkspace,
) -> Result<DiagnosticPropensityColumns, ValidationError> {
    let ids = problem.estimation_ids(true);
    let row_mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    let treatment = problem
        .data
        .float64_masked(problem.treatment(), &row_mask)
        .map_err(ValidationError::from)?;
    let outcome = if include_outcome_in_mask {
        Some(
            problem
                .data
                .float64_masked(problem.outcome(), &row_mask)
                .map_err(ValidationError::from)?,
        )
    } else {
        None
    };
    let nrows = treatment.len();
    let ncols = 1 + problem.estimand.adjustment_set.len();
    let mut design = vec![0.0; nrows * ncols];
    for r in design.iter_mut().take(nrows) {
        *r = 1.0;
    }
    for (i, &z) in problem.estimand.adjustment_set.iter().enumerate() {
        let col = problem.data.float64_masked(z, &row_mask).map_err(ValidationError::from)?;
        let base = (1 + i) * nrows;
        design[base..base + nrows].copy_from_slice(&col);
    }
    let backend = FaerBackend;
    let fit: PropensityFit = fit_propensity_diagnostic(
        &design,
        nrows,
        ncols,
        &treatment,
        &backend,
        propensity,
        glm_options,
    )
    .map_err(ValidationError::from)?;
    let defect = propensity_fit_defect(&fit.glm);
    Ok(DiagnosticPropensityColumns { scores: fit.scores, treatment, outcome, defect })
}

/// An [`OverlapReport`] from a diagnostic propensity fit, with the fit's defect, if any.
pub(crate) struct DiagnosticOverlap {
    /// Report on the (possibly regularized) scores.
    pub report: OverlapReport,
    /// Separation / non-convergence of the diagnostic fit (see [`propensity_fit_defect`]).
    pub defect: Option<&'static str>,
}

/// Build an [`OverlapReport`] from a diagnostic propensity fit, reusing a warmed
/// propensity workspace (callers without one pass a local `PropensityWorkspace::default()`).
pub(crate) fn diagnostic_overlap_report_with(
    problem: &RefutationProblem<'_>,
    glm_options: &GlmOptions,
    policy: OverlapPolicy,
    propensity: &mut PropensityWorkspace,
) -> Result<DiagnosticOverlap, ValidationError> {
    let cols = fit_diagnostic_propensity(problem, glm_options, false, propensity)?;
    // ATE IPW weights so ESS / extreme-weight fields are defined for the overlap refuter.
    let weights: Vec<f64> = cols
        .treatment
        .iter()
        .zip(cols.scores.iter())
        .map(|(&t, &p)| {
            let p = p.clamp(1e-9, 1.0 - 1e-9);
            if t > 0.5 { 1.0 / p } else { 1.0 / (1.0 - p) }
        })
        .collect();
    Ok(DiagnosticOverlap {
        report: OverlapReport::from_propensities(
            &cols.scores,
            Some(&weights),
            policy,
            Some(&cols.treatment),
            Some(IpwTarget::Ate),
            None,
        ),
        defect: cols.defect,
    })
}

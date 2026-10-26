//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Fit separate OLS outcome models on the control (`T=0`) and treated (`T=1`) arms of
/// `design_matrix` (column-major `[1 | Z…]`), returning `(beta0, beta1)`.
///
/// # Errors
///
/// Empty treated/control arm, or an OLS backend failure (e.g. rank deficiency within an arm).
#[allow(
    dead_code,
    reason = "the unweighted entry point of fit_outcome_models_weighted, kept for callers without weights"
)]
pub(crate) fn fit_outcome_models(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    outcome: &[f64],
    backend: FaerBackend,
    workspace: &mut AipwWorkspace,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    fit_outcome_models_weighted(
        design_matrix,
        nrows,
        ncols,
        treatment,
        outcome,
        None,
        backend,
        workspace,
    )
}

/// [`fit_outcome_models`] under optional prior (sampling) weights: each arm's model is the
/// weighted least-squares fit, computed as OLS on rows scaled by the square root of their
/// weights.
///
/// # Errors
///
/// As [`fit_outcome_models`].
#[allow(
    clippy::too_many_arguments,
    reason = "the unweighted entry point's arguments plus the weights"
)]
pub(crate) fn fit_outcome_models_weighted(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    outcome: &[f64],
    weights: Option<&[f64]>,
    backend: FaerBackend,
    workspace: &mut AipwWorkspace,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::data_msg(
            "AIPW outcome regression requires both treated and control rows",
        ));
    }

    select_rows_colmajor(design_matrix, nrows, ncols, &control_idx, &mut workspace.control_design);
    select_values(outcome, &control_idx, &mut workspace.control_outcome);
    if let Some(w) = weights {
        root_weight_rows(
            &mut workspace.control_design,
            &mut workspace.control_outcome,
            ncols,
            &control_idx,
            w,
        );
    }
    let fit0 = backend
        .least_squares(
            &workspace.control_design,
            control_idx.len(),
            ncols,
            &workspace.control_outcome,
            &mut workspace.outcome,
        )
        .map_err(crate::error::EstimationError::from)?;
    let beta0 = fit0.coefficients;

    select_rows_colmajor(design_matrix, nrows, ncols, &treated_idx, &mut workspace.treated_design);
    select_values(outcome, &treated_idx, &mut workspace.treated_outcome);
    if let Some(w) = weights {
        root_weight_rows(
            &mut workspace.treated_design,
            &mut workspace.treated_outcome,
            ncols,
            &treated_idx,
            w,
        );
    }
    let fit1 = backend
        .least_squares(
            &workspace.treated_design,
            treated_idx.len(),
            ncols,
            &workspace.treated_outcome,
            &mut workspace.outcome,
        )
        .map_err(crate::error::EstimationError::from)?;
    let beta1 = fit1.coefficients;

    Ok((beta0, beta1))
}

/// Multiply the rows of an arm's column-major design and outcome by the square root of the
/// weight of the source row each came from (`rows[r]` indexes `weights`).
pub(super) fn root_weight_rows(
    design: &mut [f64],
    outcome: &mut [f64],
    ncols: usize,
    rows: &[usize],
    weights: &[f64],
) {
    let m = rows.len();
    for (r, &source) in rows.iter().enumerate() {
        let root = weights[source].sqrt();
        outcome[r] *= root;
        for c in 0..ncols {
            design[c * m + r] *= root;
        }
    }
}

/// Predict `design · coef` for every row of a column-major `nrows × ncols` design.
pub(crate) fn predict_colmajor(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    coef: &[f64],
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(nrows, 0.0);
    for (r, pred) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for c in 0..ncols {
            s += design_matrix[c * nrows + r] * coef[c];
        }
        *pred = s;
    }
}

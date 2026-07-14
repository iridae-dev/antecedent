//! Shared g-computation contrast kernel (GLM + Bayesian).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments, clippy::cast_precision_loss)]

use causal_stats::GlmFamily;

/// Per-row mean-scale contrast `μ(T=active, Z) − μ(T=control, Z)`.
#[must_use]
pub fn gcomp_diffs(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    active: f64,
    control: f64,
) -> Vec<f64> {
    let mut diffs = Vec::with_capacity(nrows);
    for r in 0..nrows {
        diffs.push(gcomp_row_contrast(
            family,
            x_colmajor,
            nrows,
            ncols,
            t_col,
            coefficients,
            active,
            control,
            r,
        ));
    }
    diffs
}

/// Single-row mean-scale contrast.
#[must_use]
pub fn gcomp_row_contrast(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    active: f64,
    control: f64,
    row: usize,
) -> f64 {
    let mut eta_active = 0.0;
    let mut eta_control = 0.0;
    for c in 0..ncols {
        let coef = coefficients[c];
        if c == t_col {
            eta_active += active * coef;
            eta_control += control * coef;
        } else {
            let val = x_colmajor[c * nrows + row];
            eta_active += val * coef;
            eta_control += val * coef;
        }
    }
    family.mean_from_eta(eta_active) - family.mean_from_eta(eta_control)
}

/// Mean ATE across rows for one coefficient vector.
#[must_use]
pub fn gcomp_mean_ate(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    active: f64,
    control: f64,
) -> f64 {
    if nrows == 0 {
        return f64::NAN;
    }
    let sum: f64 = (0..nrows)
        .map(|r| {
            gcomp_row_contrast(
                family,
                x_colmajor,
                nrows,
                ncols,
                t_col,
                coefficients,
                active,
                control,
                r,
            )
        })
        .sum();
    sum / nrows as f64
}

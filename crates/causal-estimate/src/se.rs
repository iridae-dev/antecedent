//! Shared analytic SE policy for ATE estimators (DESIGN.md §11.3 wiring).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_stats::{SandwichKind, coefficient_covariance};

use crate::error::EstimationError;

/// Analytic standard-error kind shared across linear, IV, AIPW, matching, and GLM estimators.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum AnalyticSeKind {
    /// Classical / estimator-default homoskedastic (or IID influence) formula.
    #[default]
    Homoskedastic,
    /// HC0 sandwich (no finite-sample correction).
    Hc0,
    /// HC1 sandwich where OLS/2SLS applies; robust IF / heteroskedastic AI elsewhere.
    Hc1,
    /// HC2 leverage-corrected sandwich.
    Hc2,
    /// HC3 leverage-corrected sandwich.
    Hc3,
    /// Cluster-robust; requires `cluster_ids` on the estimator (`length = nrows`).
    Cluster,
    /// Multiway cluster-robust; requires `multiway_ids` (one `Vec<u32>` per dimension).
    Multiway,
    /// Newey–West HAC with the given lag.
    NeweyWest {
        /// Maximum autocorrelation lag.
        lag: usize,
    },
    /// Panel cluster + temporal HAC; requires `cluster_ids` and lag.
    PanelClusterHac {
        /// Temporal HAC lag within clusters.
        lag: usize,
    },
}

/// Alias retained for existing linear-adjustment call sites.
pub type LinearSeKind = AnalyticSeKind;

/// Default ridge λ applied by propensity / GLM estimators on separation.
///
/// Re-export of [`causal_stats::DEFAULT_RIDGE_ON_SEPARATION`] (single source of truth).
pub use causal_stats::DEFAULT_RIDGE_ON_SEPARATION;

/// Require cluster labels matching prepared row count.
///
/// # Errors
///
/// Missing ids or length mismatch.
pub(crate) fn require_clusters(
    ids: &Option<Vec<u32>>,
    n: usize,
) -> Result<&[u32], EstimationError> {
    let Some(ids) = ids.as_ref() else {
        return Err(EstimationError::unsupported("AnalyticSeKind::Cluster/PanelClusterHac requires estimator.cluster_ids"));
    };
    if ids.len() != n {
        return Err(EstimationError::data_msg(format!(
            "cluster_ids length {} != nrows {n}",
            ids.len()
        )));
    }
    Ok(ids.as_slice())
}

/// Require multiway cluster label dimensions matching prepared row count.
///
/// # Errors
///
/// Missing ids, empty dimensions, or length mismatch.
pub(crate) fn require_multiway(
    ids: &Option<Vec<Vec<u32>>>,
    n: usize,
) -> Result<&[Vec<u32>], EstimationError> {
    let Some(ids) = ids.as_ref() else {
        return Err(EstimationError::unsupported("AnalyticSeKind::Multiway requires estimator.multiway_ids"));
    };
    if ids.is_empty() {
        return Err(EstimationError::unsupported("AnalyticSeKind::Multiway requires at least one clustering dimension"));
    }
    for (i, dim) in ids.iter().enumerate() {
        if dim.len() != n {
            return Err(EstimationError::data_msg(format!(
                "multiway_ids[{i}] length {} != nrows {n}",
                dim.len()
            )));
        }
    }
    Ok(ids.as_slice())
}

/// Coefficient SE from residual sandwich, or `None` when [`AnalyticSeKind::Homoskedastic`].
///
/// # Errors
///
/// Missing cluster / multiway labels when required.
pub(crate) fn residual_sandwich_coef_se(
    kind: AnalyticSeKind,
    x: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    t_col: usize,
    cluster_ids: &Option<Vec<u32>>,
    multiway_ids: &Option<Vec<Vec<u32>>>,
) -> Result<Option<f64>, EstimationError> {
    if matches!(kind, AnalyticSeKind::Homoskedastic) {
        return Ok(None);
    }
    let se = match kind {
        AnalyticSeKind::Homoskedastic => unreachable!(),
        AnalyticSeKind::Hc0 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc0, t_col),
        AnalyticSeKind::Hc1 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc1, t_col),
        AnalyticSeKind::Hc2 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc2, t_col),
        AnalyticSeKind::Hc3 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc3, t_col),
        AnalyticSeKind::Cluster => {
            let groups = require_clusters(cluster_ids, nrows)?;
            sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Cluster { groups }, t_col)
        }
        AnalyticSeKind::Multiway => {
            let dims = require_multiway(multiway_ids, nrows)?;
            let refs: Vec<&[u32]> = dims.iter().map(Vec::as_slice).collect();
            sandwich_diag(
                x,
                nrows,
                ncols,
                residuals,
                SandwichKind::Multiway { dimensions: &refs },
                t_col,
            )
        }
        AnalyticSeKind::NeweyWest { lag } => {
            sandwich_diag(x, nrows, ncols, residuals, SandwichKind::NeweyWest { lag }, t_col)
        }
        AnalyticSeKind::PanelClusterHac { lag } => {
            let groups = require_clusters(cluster_ids, nrows)?;
            sandwich_diag(
                x,
                nrows,
                ncols,
                residuals,
                SandwichKind::PanelClusterHac { groups, lag },
                t_col,
            )
        }
    };
    Ok(Some(se))
}

fn sandwich_diag(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    kind: SandwichKind<'_>,
    t_col: usize,
) -> f64 {
    match coefficient_covariance(x, nrows, ncols, residuals, kind) {
        Ok(cov) => cov[t_col * ncols + t_col].max(0.0).sqrt(),
        Err(_) => f64::NAN,
    }
}

/// Cluster-robust SE for a scalar influence/score sequence (Arellano DF).
///
/// `Var = (G/(G−1)) · (1/n²) · Σ_g s_g²` with `s_g = Σ_{i∈g}(ψ_i − ψ̄)`.
#[must_use]
pub(crate) fn cluster_influence_se(psi: &[f64], groups: &[u32]) -> f64 {
    let n = psi.len();
    if n < 2 || groups.len() != n {
        return f64::NAN;
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| groups[i]);
    let mut sum_s2 = 0.0;
    let mut g_count = 0usize;
    let mut idx = 0usize;
    while idx < n {
        let g = groups[order[idx]];
        let mut s = 0.0;
        while idx < n && groups[order[idx]] == g {
            let i = order[idx];
            s += psi[i] - mean;
            idx += 1;
        }
        sum_s2 += s * s;
        g_count += 1;
    }
    if g_count <= 1 {
        return f64::NAN;
    }
    let scale = (g_count as f64 / (g_count as f64 - 1.0)) / (n as f64).powi(2);
    (scale * sum_s2).max(0.0).sqrt()
}

/// Heteroskedastic (HC1-style) SE for a scalar influence sequence: `√(Σ ψ_i² / (n(n−1)))`.
#[must_use]
pub(crate) fn hetero_influence_se(psi: &[f64]) -> f64 {
    let n = psi.len();
    if n < 2 {
        return f64::NAN;
    }
    let sum_sq: f64 = psi.iter().map(|v| v * v).sum();
    (sum_sq / ((n * (n - 1)) as f64)).max(0.0).sqrt()
}

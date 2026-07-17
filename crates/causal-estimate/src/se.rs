//! Shared analytic SE policy for ATE estimators (DESIGN.md §11.3 wiring).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;

/// Analytic standard-error kind shared across linear, IV, AIPW, and matching estimators.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum AnalyticSeKind {
    /// Classical / estimator-default homoskedastic (or IID influence) formula.
    #[default]
    Homoskedastic,
    /// HC1 sandwich where OLS/2SLS applies; robust IF / heteroskedastic AI elsewhere.
    Hc1,
    /// Cluster-robust; requires `cluster_ids` on the estimator (`length = nrows`).
    Cluster,
}

/// Alias retained for existing linear-adjustment call sites.
pub type LinearSeKind = AnalyticSeKind;

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
        return Err(EstimationError::UnsupportedQuery(
            "AnalyticSeKind::Cluster requires estimator.cluster_ids".into(),
        ));
    };
    if ids.len() != n {
        return Err(EstimationError::data_msg(format!(
            "cluster_ids length {} != nrows {n}",
            ids.len()
        )));
    }
    Ok(ids.as_slice())
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

//! Shared analytic SE policy for ATE estimators.
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

/// Heteroskedastic (HC1-style) SE for a scalar influence sequence:
/// `√(Σ (ψ_i − ψ̄)² / (n(n−1)))`.
///
/// Demeaning is required: without it the estimator targets `(Var(ψ) + ATE²)/n`
/// whenever `E[ψ] = ATE ≠ 0`.
#[must_use]
pub(crate) fn hetero_influence_se(psi: &[f64]) -> f64 {
    let n = psi.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = psi.iter().map(|v| {
        let d = v - mean;
        d * d
    }).sum();
    (sum_sq / ((n * (n - 1)) as f64)).max(0.0).sqrt()
}

/// Multiway cluster-robust SE for a scalar IF (Cameron–Gelbach–Miller style for ≤2 ways;
/// for >2 ways averages one-way cluster variances).
#[must_use]
pub(crate) fn multiway_influence_se(psi: &[f64], dimensions: &[Vec<u32>]) -> f64 {
    if dimensions.is_empty() || psi.len() < 2 {
        return f64::NAN;
    }
    if dimensions.len() == 1 {
        return cluster_influence_se(psi, &dimensions[0]);
    }
    if dimensions.len() == 2 {
        let se_a = cluster_influence_se(psi, &dimensions[0]);
        let se_b = cluster_influence_se(psi, &dimensions[1]);
        let intersect: Vec<u32> = dimensions[0]
            .iter()
            .zip(dimensions[1].iter())
            .map(|(&a, &b)| a.wrapping_mul(1_000_003).wrapping_add(b))
            .collect();
        let se_ab = cluster_influence_se(psi, &intersect);
        let var = se_a.powi(2) + se_b.powi(2) - se_ab.powi(2);
        return var.max(0.0).sqrt();
    }
    let mut var = 0.0;
    for dim in dimensions {
        let se = cluster_influence_se(psi, dim);
        var += se.powi(2);
    }
    (var / dimensions.len() as f64).max(0.0).sqrt()
}

/// Newey–West HAC SE for a scalar IF sequence (Bartlett kernel).
#[must_use]
pub(crate) fn newey_west_influence_se(psi: &[f64], lag: usize) -> f64 {
    let n = psi.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    let d: Vec<f64> = psi.iter().map(|v| v - mean).collect();
    let mut gamma0 = 0.0;
    for &x in &d {
        gamma0 += x * x;
    }
    gamma0 /= n as f64;
    let mut hac = gamma0;
    let l = lag.min(n.saturating_sub(1));
    for k in 1..=l {
        let mut g = 0.0;
        for i in k..n {
            g += d[i] * d[i - k];
        }
        g /= n as f64;
        let w = 1.0 - (k as f64) / ((l + 1) as f64);
        hac += 2.0 * w * g;
    }
    (hac.max(0.0) / n as f64).sqrt()
}

/// Panel cluster + Newey–West: cluster-robust SE with HAC-smoothed within-cluster scores.
///
/// Approximates by demeaning ψ, summing within clusters, then applying NW to the cluster
/// score series ordered by first occurrence (falls back to cluster SE when lag = 0).
#[must_use]
pub(crate) fn panel_cluster_hac_influence_se(psi: &[f64], groups: &[u32], lag: usize) -> f64 {
    if lag == 0 {
        return cluster_influence_se(psi, groups);
    }
    let n = psi.len();
    if n < 2 || groups.len() != n {
        return f64::NAN;
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| groups[i]);
    let mut scores = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        let g = groups[order[idx]];
        let mut s = 0.0;
        while idx < n && groups[order[idx]] == g {
            s += psi[order[idx]] - mean;
            idx += 1;
        }
        scores.push(s);
    }
    if scores.len() < 2 {
        return f64::NAN;
    }
    // Treat cluster scores as a short series; NW then rescale by 1/n.
    let se_scores = newey_west_influence_se(&scores, lag.min(scores.len().saturating_sub(1)));
    // newey_west divides by √G; we need /n relative to unit IF mean.
    se_scores * (scores.len() as f64).sqrt() / n as f64
}

/// Dispatch IF-based analytic SE kinds shared by AIPW / Wald / matching.
pub(crate) fn influence_se_kind(
    kind: AnalyticSeKind,
    psi: &[f64],
    nrows: usize,
    cluster_ids: &Option<Vec<u32>>,
    multiway_ids: &Option<Vec<Vec<u32>>>,
    row_map: Option<&[usize]>,
) -> Result<f64, EstimationError> {
    let gather_ids = |ids: &[u32]| -> Vec<u32> {
        match row_map {
            Some(map) => map.iter().map(|&i| ids[i]).collect(),
            None => ids.to_vec(),
        }
    };
    Ok(match kind {
        AnalyticSeKind::Homoskedastic => {
            let n = psi.len() as f64;
            crate::util::sample_std(psi) / n.sqrt()
        }
        AnalyticSeKind::Hc0
        | AnalyticSeKind::Hc1
        | AnalyticSeKind::Hc2
        | AnalyticSeKind::Hc3 => hetero_influence_se(psi),
        AnalyticSeKind::Cluster => {
            let groups_full = require_clusters(cluster_ids, nrows)?;
            let g = gather_ids(groups_full);
            cluster_influence_se(psi, &g)
        }
        AnalyticSeKind::Multiway => {
            let dims = require_multiway(multiway_ids, nrows)?;
            let gathered: Vec<Vec<u32>> = dims.iter().map(|d| gather_ids(d)).collect();
            multiway_influence_se(psi, &gathered)
        }
        AnalyticSeKind::NeweyWest { lag } => newey_west_influence_se(psi, lag),
        AnalyticSeKind::PanelClusterHac { lag } => {
            let groups_full = require_clusters(cluster_ids, nrows)?;
            let g = gather_ids(groups_full);
            panel_cluster_hac_influence_se(psi, &g, lag)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::hetero_influence_se;

    #[test]
    fn hetero_influence_se_demeans() {
        // Constant nonzero ψ: Var = 0 after demeaning → SE = 0 (not |ATE|/√(n−1)).
        let psi = vec![2.0_f64; 10];
        let se = hetero_influence_se(&psi);
        assert!(se.is_finite());
        assert!(se < 1e-12, "expected near-zero SE after demeaning, got {se}");
    }

    #[test]
    fn hetero_influence_se_matches_sample_sd_over_sqrt_n() {
        let psi = [1.0, 2.0, 3.0, 4.0, 5.0];
        let se = hetero_influence_se(&psi);
        let mean = 3.0;
        let var: f64 = psi.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        let expected = (var / 5.0).sqrt();
        assert!((se - expected).abs() < 1e-12);
    }
}

//! Shared analytic SE policy for ATE estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::manual_map,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use antecedent_stats::{
    MAX_CLUSTER_DIMENSIONS, SandwichKind, coefficient_covariance, combine_inclusion_exclusion,
    intern_cluster_tuples, multiway_subset_masks, panel_hac_meat_scalar,
};

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
    /// Panel cluster + temporal HAC; requires `cluster_ids`, `panel_times`, and lag.
    PanelClusterHac {
        /// Temporal HAC lag within clusters.
        lag: usize,
    },
}

/// Alias retained for existing linear-adjustment call sites.
pub type LinearSeKind = AnalyticSeKind;

/// Default ridge λ applied by propensity / GLM estimators on separation.
///
/// Re-export of [`antecedent_stats::DEFAULT_RIDGE_ON_SEPARATION`] (single source of truth).
pub use antecedent_stats::DEFAULT_RIDGE_ON_SEPARATION;

/// Require cluster labels matching prepared row count.
///
/// # Errors
///
/// Missing ids or length mismatch.
pub(crate) fn require_clusters(ids: Option<&[u32]>, n: usize) -> Result<&[u32], EstimationError> {
    let Some(ids) = ids else {
        return Err(EstimationError::unsupported(
            "AnalyticSeKind::Cluster/PanelClusterHac requires estimator.cluster_ids",
        ));
    };
    if ids.len() != n {
        return Err(EstimationError::data_msg(format!(
            "cluster_ids length {} != nrows {n}",
            ids.len()
        )));
    }
    Ok(ids)
}

/// Require panel time labels matching prepared row count.
///
/// # Errors
///
/// Missing times or length mismatch.
pub(crate) fn require_panel_times(
    times: Option<&[i64]>,
    n: usize,
) -> Result<&[i64], EstimationError> {
    let Some(times) = times else {
        return Err(EstimationError::unsupported(
            "AnalyticSeKind::PanelClusterHac requires estimator.panel_times",
        ));
    };
    if times.len() != n {
        return Err(EstimationError::data_msg(format!(
            "panel_times length {} != nrows {n}",
            times.len()
        )));
    }
    Ok(times)
}

/// Require multiway cluster label dimensions matching prepared row count.
///
/// # Errors
///
/// Missing ids, empty dimensions, or length mismatch.
pub(crate) fn require_multiway(
    ids: Option<&[Vec<u32>]>,
    n: usize,
) -> Result<&[Vec<u32>], EstimationError> {
    let Some(ids) = ids else {
        return Err(EstimationError::unsupported(
            "AnalyticSeKind::Multiway requires estimator.multiway_ids",
        ));
    };
    if ids.is_empty() {
        return Err(EstimationError::unsupported(
            "AnalyticSeKind::Multiway requires at least one clustering dimension",
        ));
    }
    if ids.len() > MAX_CLUSTER_DIMENSIONS {
        return Err(EstimationError::unsupported(
            "AnalyticSeKind::Multiway supports at most 4 clustering dimensions",
        ));
    }
    for (i, dim) in ids.iter().enumerate() {
        if dim.len() != n {
            return Err(EstimationError::data_msg(format!(
                "multiway_ids[{i}] length {} != nrows {n}",
                dim.len()
            )));
        }
    }
    Ok(ids)
}

/// Coefficient SE from residual sandwich, or `None` when [`AnalyticSeKind::Homoskedastic`].
///
/// # Errors
///
/// Missing cluster / multiway / panel labels when required.
pub(crate) fn residual_sandwich_coef_se(
    kind: AnalyticSeKind,
    x: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    t_col: usize,
    cluster_ids: Option<&[u32]>,
    multiway_ids: Option<&[Vec<u32>]>,
    panel_times: Option<&[i64]>,
) -> Result<Option<f64>, EstimationError> {
    if matches!(kind, AnalyticSeKind::Homoskedastic) {
        return Ok(None);
    }
    let se = match kind {
        AnalyticSeKind::Homoskedastic => unreachable!(),
        AnalyticSeKind::Hc0 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc0, t_col)?,
        AnalyticSeKind::Hc1 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc1, t_col)?,
        AnalyticSeKind::Hc2 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc2, t_col)?,
        AnalyticSeKind::Hc3 => sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Hc3, t_col)?,
        AnalyticSeKind::Cluster => {
            let groups = require_clusters(cluster_ids, nrows)?;
            sandwich_diag(x, nrows, ncols, residuals, SandwichKind::Cluster { groups }, t_col)?
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
            )?
        }
        AnalyticSeKind::NeweyWest { lag } => {
            sandwich_diag(x, nrows, ncols, residuals, SandwichKind::NeweyWest { lag }, t_col)?
        }
        AnalyticSeKind::PanelClusterHac { lag } => {
            let groups = require_clusters(cluster_ids, nrows)?;
            let time = require_panel_times(panel_times, nrows)?;
            sandwich_diag(
                x,
                nrows,
                ncols,
                residuals,
                SandwichKind::PanelClusterHac { groups, time, lag },
                t_col,
            )?
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
) -> Result<f64, EstimationError> {
    let cov = coefficient_covariance(x, nrows, ncols, residuals, kind)?;
    Ok(cov[t_col * ncols + t_col].max(0.0).sqrt())
}

/// Cluster-robust SE for a scalar influence/score sequence (Arellano DF).
///
/// `Var = (G/(G−1)) · (1/n²) · Σ_g s_g²` with `s_g = Σ_{i∈g}(ψ_i − ψ̄)`.
///
/// # Errors
///
/// Fewer than two clusters, or length mismatch.
pub(crate) fn cluster_influence_se(psi: &[f64], groups: &[u32]) -> Result<f64, EstimationError> {
    let n = psi.len();
    if n < 2 || groups.len() != n {
        return Err(EstimationError::data_msg(
            "cluster influence SE requires n >= 2 and matching group labels",
        ));
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    match cluster_meat_scalar(psi, groups, mean) {
        Some((sum_s2, g_count)) if g_count > 1 => {
            let scale = (g_count as f64 / (g_count as f64 - 1.0)) / (n as f64).powi(2);
            Ok((scale * sum_s2).max(0.0).sqrt())
        }
        Some((_, g_count)) if g_count < 2 => {
            Err(EstimationError::stats_msg("cluster-robust variance requires at least 2 clusters"))
        }
        _ => Err(EstimationError::data_msg("cluster influence SE failed to form meat")),
    }
}

/// One-way cluster meat `M = Σ_g s_g²` and `G`, with demeaning at `mean`.
fn cluster_meat_scalar(psi: &[f64], groups: &[u32], mean: f64) -> Option<(f64, usize)> {
    let n = psi.len();
    if groups.len() != n {
        return None;
    }
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
    Some((sum_s2, g_count))
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
    let sum_sq: f64 = psi
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum();
    (sum_sq / ((n * (n - 1)) as f64)).max(0.0).sqrt()
}

/// Multiway cluster-robust SE for a scalar IF (full Cameron–Gelbach–Miller IE).
///
/// # Errors
///
/// Empty / mismatched dimensions, any CGM subset with `G < 2`, interning /
/// dimension limits, or materially negative IE residual.
pub(crate) fn multiway_influence_se(
    psi: &[f64],
    dimensions: &[Vec<u32>],
) -> Result<f64, EstimationError> {
    if dimensions.is_empty() {
        return Err(EstimationError::data_msg(
            "multiway influence SE requires at least one clustering dimension",
        ));
    }
    if psi.len() < 2 {
        return Err(EstimationError::data_msg(
            "multiway influence SE requires n >= 2",
        ));
    }
    let d = dimensions.len();
    if d > MAX_CLUSTER_DIMENSIONS {
        return Err(EstimationError::unsupported(
            "multiway influence SE supports at most 4 clustering dimensions",
        ));
    }
    let n = psi.len();
    for dim in dimensions {
        if dim.len() != n {
            return Err(EstimationError::data_msg(format!(
                "multiway dimension length {} != n {n}",
                dim.len()
            )));
        }
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    let refs: Vec<&[u32]> = dimensions.iter().map(Vec::as_slice).collect();
    let mut combined = vec![0u32; n];
    let mut terms = Vec::with_capacity((1 << d) - 1);
    for (mask, sign) in multiway_subset_masks(d) {
        let _g = intern_cluster_tuples(&refs, mask, &mut combined)
            .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
        let Some((m_s, g_s)) = cluster_meat_scalar(psi, &combined, mean) else {
            return Err(EstimationError::data_msg(
                "multiway influence SE failed to form meat",
            ));
        };
        if g_s < 2 {
            return Err(EstimationError::stats_msg(
                "cluster-robust variance requires at least 2 clusters",
            ));
        }
        let c_s = g_s as f64 / (g_s as f64 - 1.0);
        // Accumulate signed `c_S M_S`; divide by `n²` after IE.
        terms.push(sign * c_s * m_s);
    }
    let meat = combine_inclusion_exclusion(&terms)
        .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
    Ok((meat.max(0.0) / (n as f64).powi(2)).sqrt())
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

/// Panel cluster + within-unit Newey–West SE for a scalar IF.
///
/// # Errors
///
/// Missing / invalid `(cluster, time)` labels, fewer than two clusters, or
/// non-finite ψ.
pub(crate) fn panel_cluster_hac_influence_se(
    psi: &[f64],
    groups: &[u32],
    time: &[i64],
    lag: usize,
) -> Result<f64, EstimationError> {
    // lag = 0 is Arellano/cluster meat, matching SandwichKind::PanelClusterHac.
    if lag == 0 {
        return cluster_influence_se(psi, groups);
    }
    let n = psi.len();
    if n < 2 {
        return Err(EstimationError::data_msg(
            "panel HAC influence SE requires n >= 2",
        ));
    }
    if groups.len() != n || time.len() != n {
        return Err(EstimationError::data_msg(
            "panel HAC groups/time length must match n",
        ));
    }
    let mean = psi.iter().sum::<f64>() / n as f64;
    let u: Vec<f64> = psi.iter().map(|v| v - mean).collect();
    let (meat, g) = panel_hac_meat_scalar(&u, groups, time, lag)
        .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
    if g < 2 {
        return Err(EstimationError::stats_msg(
            "cluster-robust variance requires at least 2 clusters",
        ));
    }
    let c_g = g as f64 / (g as f64 - 1.0);
    Ok((c_g * meat).max(0.0).sqrt() / n as f64)
}

/// Dispatch IF-based analytic SE kinds shared by AIPW / Wald / matching.
pub(crate) fn influence_se_kind(
    kind: AnalyticSeKind,
    psi: &[f64],
    nrows: usize,
    cluster_ids: Option<&[u32]>,
    multiway_ids: Option<&[Vec<u32>]>,
    panel_times: Option<&[i64]>,
    row_map: Option<&[usize]>,
) -> Result<f64, EstimationError> {
    let gather_ids = |ids: &[u32]| -> Vec<u32> {
        match row_map {
            Some(map) => map.iter().map(|&i| ids[i]).collect(),
            None => ids.to_vec(),
        }
    };
    let gather_times = |times: &[i64]| -> Vec<i64> {
        match row_map {
            Some(map) => map.iter().map(|&i| times[i]).collect(),
            None => times.to_vec(),
        }
    };
    Ok(match kind {
        AnalyticSeKind::Homoskedastic => {
            let n = psi.len() as f64;
            crate::util::sample_std(psi) / n.sqrt()
        }
        AnalyticSeKind::Hc0 | AnalyticSeKind::Hc1 | AnalyticSeKind::Hc2 | AnalyticSeKind::Hc3 => {
            hetero_influence_se(psi)
        }
        AnalyticSeKind::Cluster => {
            let groups_full = require_clusters(cluster_ids, nrows)?;
            let g = gather_ids(groups_full);
            cluster_influence_se(psi, &g)?
        }
        AnalyticSeKind::Multiway => {
            let dims = require_multiway(multiway_ids, nrows)?;
            let gathered: Vec<Vec<u32>> = dims.iter().map(|d| gather_ids(d)).collect();
            multiway_influence_se(psi, &gathered)?
        }
        AnalyticSeKind::NeweyWest { lag } => newey_west_influence_se(psi, lag),
        AnalyticSeKind::PanelClusterHac { lag } => {
            let groups_full = require_clusters(cluster_ids, nrows)?;
            let times_full = require_panel_times(panel_times, nrows)?;
            let g = gather_ids(groups_full);
            let t = gather_times(times_full);
            panel_cluster_hac_influence_se(psi, &g, &t, lag)?
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn multiway_collision_labels_differ() {
        // Packing collision (1,0) vs (0, 1_000_003) must stay distinct.
        let psi = [1.0, -1.0, 0.5, -0.5];
        let dim_a = vec![1u32, 0, 1, 0];
        let dim_b = vec![0u32, 1_000_003, 0, 1_000_003];
        let se = multiway_influence_se(&psi, &[dim_a, dim_b]).unwrap();
        assert!(se.is_finite() && se > 0.0);
        // Collapsed packing would merge all four rows into fewer intersection groups.
        let packed_collide_a = vec![1u32, 0];
        let packed_collide_b = vec![0u32, 1_000_003];
        let mut out = [0u32; 2];
        let g =
            intern_cluster_tuples(&[&packed_collide_a, &packed_collide_b], 0b11, &mut out).unwrap();
        assert_eq!(g, 2);
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn multiway_three_way_is_cgm_not_average() {
        let psi = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
        let dim_a = vec![0u32, 0, 0, 0, 1, 1, 1, 1];
        let dim_b = vec![0u32, 0, 1, 1, 0, 0, 1, 1];
        let dim_c = vec![0u32, 1, 0, 1, 0, 1, 0, 1];
        let se =
            multiway_influence_se(&psi, &[dim_a.clone(), dim_b.clone(), dim_c.clone()]).unwrap();
        let se_a = cluster_influence_se(&psi, &dim_a).unwrap();
        let se_b = cluster_influence_se(&psi, &dim_b).unwrap();
        let se_c = cluster_influence_se(&psi, &dim_c).unwrap();
        let avg = ((se_a.powi(2) + se_b.powi(2) + se_c.powi(2)) / 3.0).sqrt();
        // Full CGM must differ from the old average-of-one-ways heuristic.
        assert!((se - avg).abs() > 1e-6, "se={se} avg={avg}");
        // Dimension permutation invariance.
        let se_perm = multiway_influence_se(&psi, &[dim_c, dim_a, dim_b]).unwrap();
        assert!((se - se_perm).abs() < 1e-12);
    }

    #[test]
    fn one_cluster_influence_and_sandwich_both_error() {
        let psi = [1.0, -0.5, 0.25, -0.25];
        let groups = [0u32, 0, 0, 0];
        let err_if = cluster_influence_se(&psi, &groups).unwrap_err();
        assert!(err_if.to_string().contains("at least 2 clusters"), "err={err_if}");
        let n = psi.len();
        let mean = psi.iter().sum::<f64>() / n as f64;
        let e: Vec<f64> = psi.iter().map(|v| v - mean).collect();
        let x = vec![1.0; n];
        let err_sw =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::Cluster { groups: &groups })
                .unwrap_err();
        assert!(err_sw.to_string().contains("at least 2 clusters"), "err={err_sw}");
    }

    #[test]
    fn multiway_intercept_sandwich_parity_one_to_four_ways() {
        let psi = [
            1.0, -0.5, 0.25, -0.75, 0.5, -0.25, 0.1, -0.1, 0.3, -0.3, 0.4, -0.4, 0.2, -0.2, 0.15,
            -0.15,
        ];
        let n = psi.len();
        let mean = psi.iter().sum::<f64>() / n as f64;
        let e: Vec<f64> = psi.iter().map(|v| v - mean).collect();
        let x = vec![1.0; n];
        let dims = [
            (0..n).map(|i| (i % 4) as u32).collect::<Vec<_>>(),
            (0..n).map(|i| ((i / 2) % 3) as u32).collect::<Vec<_>>(),
            (0..n).map(|i| (i % 2) as u32).collect::<Vec<_>>(),
            (0..n).map(|i| ((i / 4) % 2) as u32).collect::<Vec<_>>(),
        ];
        for d in 1..=4 {
            let selected: Vec<Vec<u32>> = dims[..d].to_vec();
            let se_if = multiway_influence_se(&psi, &selected).unwrap();
            let refs: Vec<&[u32]> = selected.iter().map(Vec::as_slice).collect();
            let cov =
                coefficient_covariance(&x, n, 1, &e, SandwichKind::Multiway { dimensions: &refs })
                    .unwrap();
            let se_sw = cov[0].sqrt();
            assert!((se_if - se_sw).abs() < 1e-10, "d={d}: if={se_if} sandwich={se_sw}");
        }
    }

    #[test]
    fn multiway_relabel_invariant() {
        let psi = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
        let dim_a = vec![0u32, 0, 1, 1, 2, 2];
        let dim_b = vec![0u32, 1, 0, 1, 0, 1];
        let se = multiway_influence_se(&psi, &[dim_a, dim_b]).unwrap();
        // One-to-one relabel within each dimension.
        let dim_a2 = vec![10u32, 10, 20, 20, 30, 30];
        let dim_b2 = vec![7u32, 9, 7, 9, 7, 9];
        let se2 = multiway_influence_se(&psi, &[dim_a2, dim_b2]).unwrap();
        assert!((se - se2).abs() < 1e-12);
    }

    #[test]
    fn panel_hac_intercept_sandwich_parity() {
        let psi = [1.0, 0.5, 0.25, -1.0, -0.5, -0.25, 0.75, 0.4];
        let n = psi.len();
        let groups = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let time = [0i64, 1, 2, 3, 0, 1, 2, 3];
        let lag = 2usize;
        let se_if = panel_cluster_hac_influence_se(&psi, &groups, &time, lag).unwrap();
        let mean = psi.iter().sum::<f64>() / n as f64;
        let e: Vec<f64> = psi.iter().map(|v| v - mean).collect();
        let x = vec![1.0; n];
        let cov = coefficient_covariance(
            &x,
            n,
            1,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag },
        )
        .unwrap();
        let se_sw = cov[0].sqrt();
        assert!((se_if - se_sw).abs() < 1e-10, "if={se_if} sandwich={se_sw}");
    }

    #[test]
    fn multiway_singleton_dimension_errors() {
        let psi = [1.0, -0.5, 0.25, -0.25];
        let dim_ok = vec![0u32, 0, 1, 1];
        let dim_singleton = vec![0u32, 0, 0, 0];
        let err = multiway_influence_se(&psi, &[dim_ok, dim_singleton]).unwrap_err();
        assert!(err.to_string().contains("at least 2 clusters"), "err={err}");
        assert!(multiway_influence_se(&psi, &[]).is_err());
        assert!(multiway_influence_se(&psi, &[vec![0u32, 1]]).is_err());
    }

    #[test]
    fn panel_hac_lag_zero_matches_cluster() {
        let psi = [1.0, 0.5, 0.25, -1.0, -0.5, -0.25, 0.75, 0.4];
        let n = psi.len();
        let groups = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let time = [0i64, 1, 2, 3, 0, 1, 2, 3];
        let se_panel = panel_cluster_hac_influence_se(&psi, &groups, &time, 0).unwrap();
        let se_cluster = cluster_influence_se(&psi, &groups).unwrap();
        assert!(
            (se_panel - se_cluster).abs() < 1e-12,
            "panel lag0={se_panel} cluster={se_cluster}"
        );
        let mean = psi.iter().sum::<f64>() / n as f64;
        let e: Vec<f64> = psi.iter().map(|v| v - mean).collect();
        let x = vec![1.0; n];
        let cov_panel = coefficient_covariance(
            &x,
            n,
            1,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag: 0 },
        )
        .unwrap();
        let cov_cluster =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::Cluster { groups: &groups })
                .unwrap();
        assert!((cov_panel[0] - cov_cluster[0]).abs() < 1e-12);
    }
}

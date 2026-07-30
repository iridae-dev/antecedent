//! Shared cluster / multiway / panel-HAC primitives.
//!
//! Exact tuple interning (no lossy `u32` packing), Cameron–Gelbach–Miller (2011) subset
//! signs, Bartlett/`L_eff` helpers, and within-unit panel HAC meat.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use std::collections::{HashMap, HashSet};

use crate::error::StatsError;

/// Maximum clustering dimensions for multiway / tuple interning.
pub const MAX_CLUSTER_DIMENSIONS: usize = 4;

/// Exact cluster-label key for up to [`MAX_CLUSTER_DIMENSIONS`] selected dims.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ClusterTuple {
    labels: [u32; MAX_CLUSTER_DIMENSIONS],
    len: u8,
}

/// Intern selected dimension labels into dense ids `0..G-1`.
///
/// For each row, builds the tuple of labels where `subset_mask` bits are set and
/// assigns a unique dense id via [`HashMap`] with exact key equality (never
/// arithmetic `u32` packing).
///
/// # Errors
///
/// Empty / too many dimensions, bad mask, or length mismatch.
pub fn intern_cluster_tuples(
    dimensions: &[&[u32]],
    subset_mask: usize,
    out_ids: &mut [u32],
) -> Result<usize, StatsError> {
    let d = dimensions.len();
    if d == 0 || d > MAX_CLUSTER_DIMENSIONS {
        return Err(StatsError::Shape {
            message: "cluster tuple interning supports 1..=4 dimensions",
        });
    }
    if subset_mask == 0 || subset_mask >= (1usize << d) {
        return Err(StatsError::Shape {
            message: "subset_mask must be a nonempty subset of the dimensions",
        });
    }
    let n = out_ids.len();
    for dim in dimensions {
        if dim.len() != n {
            return Err(StatsError::Shape {
                message: "cluster dimension length != out_ids length",
            });
        }
    }

    let mut selected = [0usize; MAX_CLUSTER_DIMENSIONS];
    let mut k = 0u8;
    for i in 0..d {
        if (subset_mask & (1 << i)) != 0 {
            selected[k as usize] = i;
            k += 1;
        }
    }

    let mut map: HashMap<ClusterTuple, u32> = HashMap::new();
    let mut next = 0u32;
    for r in 0..n {
        let mut labels = [0u32; MAX_CLUSTER_DIMENSIONS];
        for j in 0..k {
            labels[j as usize] = dimensions[selected[j as usize]][r];
        }
        let key = ClusterTuple { labels, len: k };
        let id = *map.entry(key).or_insert_with(|| {
            let id = next;
            next = next.saturating_add(1);
            id
        });
        out_ids[r] = id;
    }
    Ok(usize::try_from(next).unwrap_or(usize::MAX))
}

/// Sign for Cameron–Gelbach–Miller term: `(-1)^{|S|+1}` → `+1` odd, `-1` even.
#[must_use]
pub fn multiway_subset_sign(subset_mask: usize) -> f64 {
    if subset_mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 }
}

/// Iterate nonempty subset masks `1..(1<<d)` with CGM signs.
pub fn multiway_subset_masks(d: usize) -> impl Iterator<Item = (usize, f64)> {
    let n_sub = 1usize << d;
    (1..n_sub).map(|mask| (mask, multiway_subset_sign(mask)))
}

/// Effective Newey–West / Bartlett max lag: `min(requested, max_span)`.
///
/// For a single series use `max_span = T - 1`. For panel HAC use
/// `max_g (max t_g − min t_g)`.
#[must_use]
pub fn effective_nw_lag(requested: usize, max_span: usize) -> usize {
    requested.min(max_span)
}

/// Bartlett kernel weight `1 − ℓ/(L_eff+1)` for lag `ℓ ≥ 1`.
#[must_use]
pub fn bartlett_weight(lag: usize, l_eff: usize) -> f64 {
    if lag == 0 || l_eff == 0 {
        return 0.0;
    }
    1.0 - (lag as f64) / ((l_eff as f64) + 1.0)
}

/// Sum signed inclusion–exclusion terms with FP-tolerant non-positive clamp.
///
/// Clamps a tiny negative result to 0 when `|neg| ≤ 64 ε Σ|term|`; otherwise
/// returns [`StatsError::NonPositiveVariance`].
///
/// # Errors
///
/// Materially negative IE residual.
pub fn combine_inclusion_exclusion(terms: &[f64]) -> Result<f64, StatsError> {
    let sum: f64 = terms.iter().sum();
    if sum >= 0.0 {
        return Ok(sum);
    }
    let abs_sum: f64 = terms.iter().map(|t| t.abs()).sum();
    let tol = 64.0 * f64::EPSILON * abs_sum;
    if (-sum) <= tol {
        Ok(0.0)
    } else {
        Err(StatsError::NonPositiveVariance {
            message: "multiway inclusion-exclusion meat is materially negative",
        })
    }
}

/// Validate panel HAC inputs: equal lengths, finite `u`, unique `(cluster, time)`.
fn validate_panel_hac(u: &[f64], clusters: &[u32], time: &[i64]) -> Result<usize, StatsError> {
    let n = u.len();
    if clusters.len() != n || time.len() != n {
        return Err(StatsError::Shape { message: "panel HAC u/clusters/time length mismatch" });
    }
    if n == 0 {
        return Err(StatsError::Shape { message: "panel HAC needs at least one row" });
    }
    if u.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape {
            message: "panel HAC requires finite influence/score values",
        });
    }
    let mut seen: HashSet<(u32, i64)> = HashSet::with_capacity(n);
    for i in 0..n {
        if !seen.insert((clusters[i], time[i])) {
            return Err(StatsError::Shape {
                message: "panel HAC requires unique (cluster, time) pairs",
            });
        }
    }
    Ok(n)
}

/// Global panel bandwidth span: `max_g (max t_g − min t_g)`.
fn panel_max_time_span(clusters: &[u32], time: &[i64]) -> usize {
    let mut bounds: HashMap<u32, (i64, i64)> = HashMap::new();
    for (&c, &t) in clusters.iter().zip(time.iter()) {
        bounds
            .entry(c)
            .and_modify(|(lo, hi)| {
                *lo = (*lo).min(t);
                *hi = (*hi).max(t);
            })
            .or_insert((t, t));
    }
    bounds
        .values()
        .map(|&(lo, hi)| usize::try_from(hi.saturating_sub(lo)).unwrap_or(0))
        .max()
        .unwrap_or(0)
}

/// Scalar within-unit panel HAC meat `M` and unit count `G`.
///
/// `demeaned` must already be demeaned (`ψ − ψ̄`). Lag products use explicit
/// integer time labels within each cluster; missing times do not invent
/// adjacencies.
///
/// # Errors
///
/// Shape / uniqueness / finiteness validation failures.
pub fn panel_hac_meat_scalar(
    demeaned: &[f64],
    clusters: &[u32],
    time: &[i64],
    max_lag: usize,
) -> Result<(f64, usize), StatsError> {
    let nrows = validate_panel_hac(demeaned, clusters, time)?;
    let max_span = panel_max_time_span(clusters, time);
    let l_eff = effective_nw_lag(max_lag, max_span);

    let mut order: Vec<usize> = (0..nrows).collect();
    order.sort_by_key(|&row| (clusters[row], time[row]));

    let mut meat = 0.0;
    let mut unit_count = 0usize;
    let mut idx = 0usize;
    while idx < nrows {
        let unit_id = clusters[order[idx]];
        let start = idx;
        while idx < nrows && clusters[order[idx]] == unit_id {
            idx += 1;
        }
        unit_count += 1;
        let rows = &order[start..idx];
        // time → demeaned value (unique by validation).
        let mut by_time: HashMap<i64, f64> = HashMap::with_capacity(rows.len());
        for &row in rows {
            by_time.insert(time[row], demeaned[row]);
            meat += demeaned[row] * demeaned[row];
        }
        for ell in 1..=l_eff {
            let weight = bartlett_weight(ell, l_eff);
            let ell_i = i64::try_from(ell).unwrap_or(i64::MAX);
            for &row in rows {
                let obs_time = time[row];
                if let Some(&lagged) = by_time.get(&(obs_time - ell_i)) {
                    meat += 2.0 * weight * demeaned[row] * lagged;
                }
            }
        }
    }
    Ok((meat, unit_count))
}

/// Matrix within-unit panel HAC meat (`ncols×ncols` row-major) and unit count `G`.
///
/// Score row `i` is `residuals[i] * x_i`. Same time-lag rules as
/// [`panel_hac_meat_scalar`].
///
/// # Errors
///
/// Shape / uniqueness validation failures.
pub fn panel_hac_meat_matrix(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    clusters: &[u32],
    time: &[i64],
    max_lag: usize,
) -> Result<(Vec<f64>, usize), StatsError> {
    if residuals.len() != nrows {
        return Err(StatsError::Shape { message: "panel HAC residual length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "panel HAC X buffer too short" });
    }
    // Reuse scalar validation on residual finiteness + (c,t) uniqueness.
    let _ = validate_panel_hac(residuals, clusters, time)?;
    let max_span = panel_max_time_span(clusters, time);
    let l_eff = effective_nw_lag(max_lag, max_span);

    let mut scores = vec![0.0; nrows * ncols];
    for row in 0..nrows {
        let residual = residuals[row];
        for col in 0..ncols {
            scores[row * ncols + col] = residual * x_colmajor[col * nrows + row];
        }
    }

    let mut order: Vec<usize> = (0..nrows).collect();
    order.sort_by_key(|&row| (clusters[row], time[row]));

    let mut meat = vec![0.0; ncols * ncols];
    let mut unit_count = 0usize;
    let mut idx = 0usize;
    while idx < nrows {
        let unit_id = clusters[order[idx]];
        let start = idx;
        while idx < nrows && clusters[order[idx]] == unit_id {
            idx += 1;
        }
        unit_count += 1;
        let rows = &order[start..idx];
        let mut by_time: HashMap<i64, usize> = HashMap::with_capacity(rows.len());
        for &row in rows {
            by_time.insert(time[row], row);
            for col_a in 0..ncols {
                for col_b in 0..ncols {
                    meat[col_a * ncols + col_b] +=
                        scores[row * ncols + col_a] * scores[row * ncols + col_b];
                }
            }
        }
        for ell in 1..=l_eff {
            let weight = bartlett_weight(ell, l_eff);
            let ell_i = i64::try_from(ell).unwrap_or(i64::MAX);
            for &row in rows {
                let obs_time = time[row];
                let Some(&lag_row) = by_time.get(&(obs_time - ell_i)) else {
                    continue;
                };
                for col_a in 0..ncols {
                    for col_b in 0..ncols {
                        let gamma_ab =
                            scores[row * ncols + col_a] * scores[lag_row * ncols + col_b];
                        let gamma_ba =
                            scores[row * ncols + col_b] * scores[lag_row * ncols + col_a];
                        meat[col_a * ncols + col_b] += weight * (gamma_ab + gamma_ba);
                    }
                }
            }
        }
    }
    Ok((meat, unit_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_nw_lag_and_bartlett_golden_weights() {
        // L_eff = min(requested, max_span); weights use capped denominator.
        assert_eq!(effective_nw_lag(2, 4), 2);
        assert_eq!(effective_nw_lag(4, 4), 4);
        assert_eq!(effective_nw_lag(10, 4), 4);
        assert_eq!(effective_nw_lag(0, 4), 0);
        assert_eq!(effective_nw_lag(3, 0), 0);

        let l_eff = 4usize;
        assert!((bartlett_weight(1, l_eff) - 0.8).abs() < 1e-15);
        assert!((bartlett_weight(2, l_eff) - 0.6).abs() < 1e-15);
        assert!((bartlett_weight(3, l_eff) - 0.4).abs() < 1e-15);
        assert!((bartlett_weight(4, l_eff) - 0.2).abs() < 1e-15);
        // Oversized requested lag must not change weights once L_eff is capped.
        assert!((bartlett_weight(1, effective_nw_lag(10, 4)) - 0.8).abs() < 1e-15);
        assert!((bartlett_weight(0, l_eff) - 0.0).abs() < 1e-15);
        assert!((bartlett_weight(1, 0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn intern_distinguishes_packing_collision() {
        // Lossy pack: (1,0) and (0, 1_000_003) both → 1_000_003.
        let dim_a = [1u32, 0];
        let dim_b = [0u32, 1_000_003];
        let dims: [&[u32]; 2] = [&dim_a, &dim_b];
        let mut out = [0u32; 2];
        let groups = intern_cluster_tuples(&dims, 0b11, &mut out).unwrap();
        assert_eq!(groups, 2);
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn subset_signs_match_cgm() {
        // 3-way: A,B,C +, AB,AC,BC −, ABC +
        assert!((multiway_subset_sign(0b001) - 1.0).abs() < 1e-15);
        assert!((multiway_subset_sign(0b010) - 1.0).abs() < 1e-15);
        assert!((multiway_subset_sign(0b100) - 1.0).abs() < 1e-15);
        assert!((multiway_subset_sign(0b011) + 1.0).abs() < 1e-15);
        assert!((multiway_subset_sign(0b101) + 1.0).abs() < 1e-15);
        assert!((multiway_subset_sign(0b110) + 1.0).abs() < 1e-15);
        assert!((multiway_subset_sign(0b111) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ie_clamp_tiny_negative() {
        let terms = [1.0, -1.0 + 1e-18];
        let combined = combine_inclusion_exclusion(&terms).unwrap();
        assert!(combined.abs() < 1e-15);
    }

    #[test]
    fn ie_errors_on_material_negative() {
        let terms = [1.0, -2.0];
        assert!(matches!(
            combine_inclusion_exclusion(&terms),
            Err(StatsError::NonPositiveVariance { .. })
        ));
    }

    #[test]
    fn panel_missing_time_skips_lag() {
        // One unit, times 0 and 2 (gap at 1): lag-1 product must be zero.
        let demeaned = [1.0, -1.0];
        let clusters = [0u32, 0];
        let time = [0i64, 2];
        let (meat0, units) = panel_hac_meat_scalar(&demeaned, &clusters, &time, 0).unwrap();
        assert_eq!(units, 1);
        assert!((meat0 - 2.0).abs() < 1e-12);
        let (meat1, _) = panel_hac_meat_scalar(&demeaned, &clusters, &time, 1).unwrap();
        // L_eff = min(1, 2-0) = 1, but (t=2,t-1=1) missing → no lag term.
        assert!((meat1 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn panel_rejects_duplicate_cluster_time() {
        let demeaned = [1.0, 2.0];
        let clusters = [0u32, 0];
        let time = [5i64, 5];
        assert!(panel_hac_meat_scalar(&demeaned, &clusters, &time, 1).is_err());
    }

    #[test]
    fn panel_relabel_and_permute_units_invariant() {
        let demeaned = [1.0, -0.5, 0.25, 0.5, -0.5, 0.5];
        let clusters = [10u32, 10, 10, 20, 20, 20];
        let time = [0i64, 1, 2, 0, 1, 2];
        let (meat0, units) = panel_hac_meat_scalar(&demeaned, &clusters, &time, 2).unwrap();
        assert_eq!(units, 2);

        // Relabel clusters.
        let clusters_relabel = [3u32, 3, 3, 7, 7, 7];
        let (meat_relabel, _) =
            panel_hac_meat_scalar(&demeaned, &clusters_relabel, &time, 2).unwrap();
        assert!((meat0 - meat_relabel).abs() < 1e-12);

        // Swap complete units in row order.
        let demeaned_swap = [0.5, -0.5, 0.5, 1.0, -0.5, 0.25];
        let clusters_swap = [20u32, 20, 20, 10, 10, 10];
        let (meat_swap, _) =
            panel_hac_meat_scalar(&demeaned_swap, &clusters_swap, &time, 2).unwrap();
        assert!((meat0 - meat_swap).abs() < 1e-12);

        // Global row shuffle with explicit (c,t).
        let perm = [5usize, 2, 0, 4, 1, 3];
        let demeaned_perm: Vec<f64> = perm.iter().map(|&i| demeaned[i]).collect();
        let clusters_perm: Vec<u32> = perm.iter().map(|&i| clusters[i]).collect();
        let time_perm: Vec<i64> = perm.iter().map(|&i| time[i]).collect();
        let (meat_perm, _) =
            panel_hac_meat_scalar(&demeaned_perm, &clusters_perm, &time_perm, 2).unwrap();
        assert!((meat0 - meat_perm).abs() < 1e-12);
    }

    #[test]
    fn panel_ar1_matches_direct_reference() {
        // One unit, consecutive times, AR structure in demeaned values.
        let demeaned = [1.0, 0.5, 0.25, 0.125];
        let clusters = [0u32; 4];
        let time = [0i64, 1, 2, 3];
        let lag = 2usize;
        let (meat, units) = panel_hac_meat_scalar(&demeaned, &clusters, &time, lag).unwrap();
        assert_eq!(units, 1);
        // Direct: γ0 + 2 k1 γ1 + 2 k2 γ2 with L_eff=2.
        let l_eff = 2usize;
        let mut reference = 0.0;
        for &value in &demeaned {
            reference += value * value;
        }
        for ell in 1..=l_eff {
            let weight = bartlett_weight(ell, l_eff);
            let mut gamma = 0.0;
            for t_idx in ell..demeaned.len() {
                gamma += demeaned[t_idx] * demeaned[t_idx - ell];
            }
            reference += 2.0 * weight * gamma;
        }
        assert!((meat - reference).abs() < 1e-12);
    }

    #[test]
    fn three_way_cgm_seven_signed_terms() {
        // Tiny 3-way fixture; verify IE equals hand-built signed one-way meats.
        let psi = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
        let dim_a = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let dim_b = [0u32, 0, 1, 1, 0, 0, 1, 1];
        let dim_c = [0u32, 1, 0, 1, 0, 1, 0, 1];
        let dims: [&[u32]; 3] = [&dim_a, &dim_b, &dim_c];
        let psi_mean: f64 = psi.iter().sum::<f64>() / psi.len() as f64;
        let mut combined = [0u32; 8];
        let mut terms = Vec::new();
        for (mask, sign) in multiway_subset_masks(3) {
            let groups = intern_cluster_tuples(&dims, mask, &mut combined).unwrap();
            assert!(groups >= 2);
            // Cluster meat with Arellano c_G.
            let mut order: Vec<usize> = (0..8).collect();
            order.sort_by_key(|&row| combined[row]);
            let mut sum_sq = 0.0;
            let mut group_count = 0usize;
            let mut idx = 0usize;
            while idx < 8 {
                let group_id = combined[order[idx]];
                let mut score = 0.0;
                while idx < 8 && combined[order[idx]] == group_id {
                    score += psi[order[idx]] - psi_mean;
                    idx += 1;
                }
                sum_sq += score * score;
                group_count += 1;
            }
            let correction = group_count as f64 / (group_count as f64 - 1.0);
            terms.push(sign * correction * sum_sq);
        }
        assert_eq!(terms.len(), 7);
        let ie_meat = combine_inclusion_exclusion(&terms).unwrap();
        assert!(ie_meat >= 0.0);
        // Permuting dimensions must leave the IE meat unchanged.
        let dims_perm: [&[u32]; 3] = [&dim_c, &dim_a, &dim_b];
        let mut terms_perm = Vec::new();
        for (mask, sign) in multiway_subset_masks(3) {
            let _ = intern_cluster_tuples(&dims_perm, mask, &mut combined).unwrap();
            let mut order: Vec<usize> = (0..8).collect();
            order.sort_by_key(|&row| combined[row]);
            let mut sum_sq = 0.0;
            let mut group_count = 0usize;
            let mut idx = 0usize;
            while idx < 8 {
                let group_id = combined[order[idx]];
                let mut score = 0.0;
                while idx < 8 && combined[order[idx]] == group_id {
                    score += psi[order[idx]] - psi_mean;
                    idx += 1;
                }
                sum_sq += score * score;
                group_count += 1;
            }
            let correction = group_count as f64 / (group_count as f64 - 1.0);
            terms_perm.push(sign * correction * sum_sq);
        }
        let ie_meat_perm = combine_inclusion_exclusion(&terms_perm).unwrap();
        assert!((ie_meat - ie_meat_perm).abs() < 1e-10);
    }
}

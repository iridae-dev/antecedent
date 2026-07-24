//! Shared cluster / multiway / panel-HAC primitives.
//!
//! Exact tuple interning (no lossy `u32` packing), Cameron–Gelbach–Miller subset
//! signs, Bartlett/`L_eff` helpers, and within-unit panel HAC meat.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use std::collections::HashMap;

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
    if subset_mask.count_ones() % 2 == 1 {
        1.0
    } else {
        -1.0
    }
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
        return Err(StatsError::Shape {
            message: "panel HAC u/clusters/time length mismatch",
        });
    }
    if n == 0 {
        return Err(StatsError::Shape {
            message: "panel HAC needs at least one row",
        });
    }
    if u.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape {
            message: "panel HAC requires finite influence/score values",
        });
    }
    let mut seen: HashMap<(u32, i64), ()> = HashMap::with_capacity(n);
    for i in 0..n {
        if seen.insert((clusters[i], time[i]), ()).is_some() {
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
/// `u` must already be demeaned (`ψ − ψ̄`). Lag products use explicit integer
/// time labels within each cluster; missing times do not invent adjacencies.
///
/// # Errors
///
/// Shape / uniqueness / finiteness validation failures.
pub fn panel_hac_meat_scalar(
    u: &[f64],
    clusters: &[u32],
    time: &[i64],
    max_lag: usize,
) -> Result<(f64, usize), StatsError> {
    let n = validate_panel_hac(u, clusters, time)?;
    let max_span = panel_max_time_span(clusters, time);
    let l_eff = effective_nw_lag(max_lag, max_span);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (clusters[i], time[i]));

    let mut meat = 0.0;
    let mut g_count = 0usize;
    let mut idx = 0usize;
    while idx < n {
        let g = clusters[order[idx]];
        let start = idx;
        while idx < n && clusters[order[idx]] == g {
            idx += 1;
        }
        g_count += 1;
        let rows = &order[start..idx];
        // time → demeaned u (unique by validation).
        let mut by_time: HashMap<i64, f64> = HashMap::with_capacity(rows.len());
        for &i in rows {
            by_time.insert(time[i], u[i]);
            meat += u[i] * u[i];
        }
        for ell in 1..=l_eff {
            let w = bartlett_weight(ell, l_eff);
            let ell_i = i64::try_from(ell).unwrap_or(i64::MAX);
            for &i in rows {
                let t = time[i];
                if let Some(&u_lag) = by_time.get(&(t - ell_i)) {
                    meat += 2.0 * w * u[i] * u_lag;
                }
            }
        }
    }
    Ok((meat, g_count))
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
        return Err(StatsError::Shape {
            message: "panel HAC residual length != nrows",
        });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape {
            message: "panel HAC X buffer too short",
        });
    }
    // Reuse scalar validation on residual finiteness + (c,t) uniqueness.
    let _ = validate_panel_hac(residuals, clusters, time)?;
    let max_span = panel_max_time_span(clusters, time);
    let l_eff = effective_nw_lag(max_lag, max_span);

    let mut scores = vec![0.0; nrows * ncols];
    for i in 0..nrows {
        let e = residuals[i];
        for c in 0..ncols {
            scores[i * ncols + c] = e * x_colmajor[c * nrows + i];
        }
    }

    let mut order: Vec<usize> = (0..nrows).collect();
    order.sort_by_key(|&i| (clusters[i], time[i]));

    let mut meat = vec![0.0; ncols * ncols];
    let mut g_count = 0usize;
    let mut idx = 0usize;
    while idx < nrows {
        let g = clusters[order[idx]];
        let start = idx;
        while idx < nrows && clusters[order[idx]] == g {
            idx += 1;
        }
        g_count += 1;
        let rows = &order[start..idx];
        let mut by_time: HashMap<i64, usize> = HashMap::with_capacity(rows.len());
        for &i in rows {
            by_time.insert(time[i], i);
            for a in 0..ncols {
                for b in 0..ncols {
                    meat[a * ncols + b] += scores[i * ncols + a] * scores[i * ncols + b];
                }
            }
        }
        for ell in 1..=l_eff {
            let w = bartlett_weight(ell, l_eff);
            let ell_i = i64::try_from(ell).unwrap_or(i64::MAX);
            for &i in rows {
                let t = time[i];
                let Some(&j) = by_time.get(&(t - ell_i)) else {
                    continue;
                };
                for a in 0..ncols {
                    for b in 0..ncols {
                        let g_ab = scores[i * ncols + a] * scores[j * ncols + b];
                        let g_ba = scores[i * ncols + b] * scores[j * ncols + a];
                        meat[a * ncols + b] += w * (g_ab + g_ba);
                    }
                }
            }
        }
    }
    Ok((meat, g_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_distinguishes_packing_collision() {
        // Lossy pack: (1,0) and (0, 1_000_003) both → 1_000_003.
        let a = [1u32, 0];
        let b = [0u32, 1_000_003];
        let dims: [&[u32]; 2] = [&a, &b];
        let mut out = [0u32; 2];
        let g = intern_cluster_tuples(&dims, 0b11, &mut out).unwrap();
        assert_eq!(g, 2);
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
        let v = combine_inclusion_exclusion(&terms).unwrap();
        assert_eq!(v, 0.0);
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
        let u = [1.0, -1.0];
        let clusters = [0u32, 0];
        let time = [0i64, 2];
        let (m0, g) = panel_hac_meat_scalar(&u, &clusters, &time, 0).unwrap();
        assert_eq!(g, 1);
        assert!((m0 - 2.0).abs() < 1e-12);
        let (m1, _) = panel_hac_meat_scalar(&u, &clusters, &time, 1).unwrap();
        // L_eff = min(1, 2-0) = 1, but (t=2,t-1=1) missing → no lag term.
        assert!((m1 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn panel_rejects_duplicate_cluster_time() {
        let u = [1.0, 2.0];
        let clusters = [0u32, 0];
        let time = [5i64, 5];
        assert!(panel_hac_meat_scalar(&u, &clusters, &time, 1).is_err());
    }

    #[test]
    fn panel_relabel_and_permute_units_invariant() {
        let u = [1.0, -0.5, 0.25, 0.5, -0.5, 0.5];
        let clusters = [10u32, 10, 10, 20, 20, 20];
        let time = [0i64, 1, 2, 0, 1, 2];
        let (m0, g) = panel_hac_meat_scalar(&u, &clusters, &time, 2).unwrap();
        assert_eq!(g, 2);

        // Relabel clusters.
        let clusters2 = [3u32, 3, 3, 7, 7, 7];
        let (m2, _) = panel_hac_meat_scalar(&u, &clusters2, &time, 2).unwrap();
        assert!((m0 - m2).abs() < 1e-12);

        // Swap complete units in row order.
        let u_swap = [0.5, -0.5, 0.5, 1.0, -0.5, 0.25];
        let c_swap = [20u32, 20, 20, 10, 10, 10];
        let (m_swap, _) = panel_hac_meat_scalar(&u_swap, &c_swap, &time, 2).unwrap();
        assert!((m0 - m_swap).abs() < 1e-12);

        // Global row shuffle with explicit (c,t).
        let perm = [5usize, 2, 0, 4, 1, 3];
        let u_p: Vec<f64> = perm.iter().map(|&i| u[i]).collect();
        let c_p: Vec<u32> = perm.iter().map(|&i| clusters[i]).collect();
        let t_p: Vec<i64> = perm.iter().map(|&i| time[i]).collect();
        let (m_p, _) = panel_hac_meat_scalar(&u_p, &c_p, &t_p, 2).unwrap();
        assert!((m0 - m_p).abs() < 1e-12);
    }

    #[test]
    fn panel_ar1_matches_direct_reference() {
        // One unit, consecutive times, AR structure in demeaned u.
        let u = [1.0, 0.5, 0.25, 0.125];
        let clusters = [0u32; 4];
        let time = [0i64, 1, 2, 3];
        let lag = 2usize;
        let (m, g) = panel_hac_meat_scalar(&u, &clusters, &time, lag).unwrap();
        assert_eq!(g, 1);
        // Direct: γ0 + 2 k1 γ1 + 2 k2 γ2 with L_eff=2.
        let l_eff = 2usize;
        let mut ref_m = 0.0;
        for &ui in &u {
            ref_m += ui * ui;
        }
        for ell in 1..=l_eff {
            let w = bartlett_weight(ell, l_eff);
            let mut g_ell = 0.0;
            for t in ell..u.len() {
                g_ell += u[t] * u[t - ell];
            }
            ref_m += 2.0 * w * g_ell;
        }
        assert!((m - ref_m).abs() < 1e-12);
    }

    #[test]
    fn three_way_cgm_seven_signed_terms() {
        // Tiny 3-way fixture; verify IE equals hand-built signed one-way meats.
        let psi = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
        let a = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let b = [0u32, 0, 1, 1, 0, 0, 1, 1];
        let c = [0u32, 1, 0, 1, 0, 1, 0, 1];
        let dims: [&[u32]; 3] = [&a, &b, &c];
        let mean: f64 = psi.iter().sum::<f64>() / psi.len() as f64;
        let mut combined = [0u32; 8];
        let mut terms = Vec::new();
        for (mask, sign) in multiway_subset_masks(3) {
            let g = intern_cluster_tuples(&dims, mask, &mut combined).unwrap();
            assert!(g >= 2);
            // Cluster meat with Arellano c_G.
            let mut order: Vec<usize> = (0..8).collect();
            order.sort_by_key(|&i| combined[i]);
            let mut m = 0.0;
            let mut g_count = 0usize;
            let mut idx = 0usize;
            while idx < 8 {
                let gid = combined[order[idx]];
                let mut s = 0.0;
                while idx < 8 && combined[order[idx]] == gid {
                    s += psi[order[idx]] - mean;
                    idx += 1;
                }
                m += s * s;
                g_count += 1;
            }
            let c_g = g_count as f64 / (g_count as f64 - 1.0);
            terms.push(sign * c_g * m);
        }
        assert_eq!(terms.len(), 7);
        let meat = combine_inclusion_exclusion(&terms).unwrap();
        assert!(meat >= 0.0);
        // Permuting dimensions must leave the IE meat unchanged.
        let dims2: [&[u32]; 3] = [&c, &a, &b];
        let mut terms2 = Vec::new();
        for (mask, sign) in multiway_subset_masks(3) {
            let _ = intern_cluster_tuples(&dims2, mask, &mut combined).unwrap();
            let mut order: Vec<usize> = (0..8).collect();
            order.sort_by_key(|&i| combined[i]);
            let mut m = 0.0;
            let mut g_count = 0usize;
            let mut idx = 0usize;
            while idx < 8 {
                let gid = combined[order[idx]];
                let mut s = 0.0;
                while idx < 8 && combined[order[idx]] == gid {
                    s += psi[order[idx]] - mean;
                    idx += 1;
                }
                m += s * s;
                g_count += 1;
            }
            let c_g = g_count as f64 / (g_count as f64 - 1.0);
            terms2.push(sign * c_g * m);
        }
        let meat2 = combine_inclusion_exclusion(&terms2).unwrap();
        assert!((meat - meat2).abs() < 1e-10);
    }
}

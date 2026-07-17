//! Sandwich / HAC coefficient covariance estimators (DESIGN.md §11.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use crate::error::StatsError;
use crate::gram::{form_xtx, invert_square};

/// Sandwich / HAC covariance kind (DESIGN.md §11.3).
#[derive(Clone, Copy, Debug)]
pub enum SandwichKind<'a> {
    /// Classical `σ² (XᵀX)⁻¹`.
    Homoskedastic,
    /// HC0 (White).
    Hc0,
    /// HC1: HC0 · n/(n−p).
    Hc1,
    /// HC2: leverage-adjusted.
    Hc2,
    /// HC3: jackknife-style leverage adjustment.
    Hc3,
    /// One-way cluster-robust.
    Cluster {
        /// Cluster id per row (length `nrows`).
        groups: &'a [u32],
    },
    /// Multiway cluster-robust (Cameron–Gelbach–Miller inclusion–exclusion).
    Multiway {
        /// One group-id slice per clustering dimension (each length `nrows`).
        dimensions: &'a [&'a [u32]],
    },
    /// Newey–West HAC with Bartlett kernel and given max lag.
    NeweyWest {
        /// Maximum lag (inclusive).
        lag: usize,
    },
    /// Panel cluster + within-unit temporal HAC (Arellano-style).
    ///
    /// Rows within each unit must be time-ordered. Cross-unit lag products are
    /// never formed. Finite-sample cluster DF correction uses `G = #units`.
    PanelClusterHac {
        /// Unit id per row (length `nrows`).
        groups: &'a [u32],
        /// Bartlett max lag within each unit.
        lag: usize,
    },
}

/// Coefficient covariance `p×p` (row-major) from design + residuals.
///
/// Consumes retained residuals and `X` only — does not refit.
///
/// # Errors
///
/// Shape mismatch, empty design, or singular bread matrix.
pub fn coefficient_covariance(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    kind: SandwichKind<'_>,
) -> Result<Vec<f64>, StatsError> {
    if residuals.len() != nrows {
        return Err(StatsError::Shape { message: "residuals length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if nrows == 0 || ncols == 0 {
        return Err(StatsError::Shape { message: "covariance needs positive dimensions" });
    }

    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let Some(bread) = invert_square(&xtx, ncols) else {
        return Err(StatsError::Backend("singular X'X in sandwich bread".into()));
    };

    match kind {
        SandwichKind::Homoskedastic => {
            let rss: f64 = residuals.iter().map(|e| e * e).sum();
            let sigma2 = rss / (nrows as f64 - ncols as f64).max(1.0);
            Ok(bread.iter().map(|v| v * sigma2).collect())
        }
        SandwichKind::Hc0 | SandwichKind::Hc1 | SandwichKind::Hc2 | SandwichKind::Hc3 => {
            let meat = hc_meat(x_colmajor, nrows, ncols, residuals, &xtx, kind)?;
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::Cluster { groups } => {
            if groups.len() != nrows {
                return Err(StatsError::Shape { message: "cluster groups length != nrows" });
            }
            let meat = cluster_meat(x_colmajor, nrows, ncols, residuals, groups)?;
            let g = distinct_count(groups);
            let scale = cluster_finite_sample(nrows, ncols, g);
            let meat: Vec<f64> = meat.iter().map(|v| v * scale).collect();
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::Multiway { dimensions } => {
            if dimensions.is_empty() {
                return Err(StatsError::Shape { message: "multiway needs ≥1 dimension" });
            }
            for d in dimensions {
                if d.len() != nrows {
                    return Err(StatsError::Shape {
                        message: "multiway dimension length != nrows",
                    });
                }
            }
            let meat = multiway_meat(x_colmajor, nrows, ncols, residuals, dimensions)?;
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::NeweyWest { lag } => {
            let meat = newey_west_meat(x_colmajor, nrows, ncols, residuals, lag)?;
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::PanelClusterHac { groups, lag } => {
            if groups.len() != nrows {
                return Err(StatsError::Shape {
                    message: "panel HAC groups length != nrows",
                });
            }
            let meat = panel_cluster_hac_meat(x_colmajor, nrows, ncols, residuals, groups, lag)?;
            let g = distinct_count(groups);
            let scale = cluster_finite_sample(nrows, ncols, g);
            let meat: Vec<f64> = meat.iter().map(|v| v * scale).collect();
            Ok(sandwich_product(&bread, &meat, ncols))
        }
    }
}

fn hc_meat(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    xtx: &[f64],
    kind: SandwichKind<'_>,
) -> Result<Vec<f64>, StatsError> {
    let hat = match kind {
        SandwichKind::Hc2 | SandwichKind::Hc3 => Some(leverages(x_colmajor, nrows, ncols, xtx)?),
        _ => None,
    };
    let mut meat = vec![0.0; ncols * ncols];
    for i in 0..nrows {
        let e = residuals[i];
        let adj = match kind {
            SandwichKind::Hc0 | SandwichKind::Hc1 => e * e,
            SandwichKind::Hc2 => {
                let h = hat.as_ref().unwrap()[i].clamp(0.0, 1.0 - 1e-12);
                (e * e) / (1.0 - h)
            }
            SandwichKind::Hc3 => {
                let h = hat.as_ref().unwrap()[i].clamp(0.0, 1.0 - 1e-12);
                let d = 1.0 - h;
                (e * e) / (d * d)
            }
            _ => unreachable!(),
        };
        accumulate_xx(&mut meat, x_colmajor, nrows, ncols, i, adj);
    }
    if matches!(kind, SandwichKind::Hc1) {
        let scale = nrows as f64 / (nrows as f64 - ncols as f64).max(1.0);
        for v in &mut meat {
            *v *= scale;
        }
    }
    Ok(meat)
}

fn leverages(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    xtx: &[f64],
) -> Result<Vec<f64>, StatsError> {
    let Some(inv) = invert_square(xtx, ncols) else {
        return Err(StatsError::Backend("singular X'X for leverages".into()));
    };
    let mut h = vec![0.0; nrows];
    for i in 0..nrows {
        // h_ii = x_i' (X'X)⁻¹ x_i
        let mut tmp = vec![0.0; ncols];
        for a in 0..ncols {
            let mut s = 0.0;
            for b in 0..ncols {
                s += inv[a * ncols + b] * x_colmajor[b * nrows + i];
            }
            tmp[a] = s;
        }
        let mut hi = 0.0;
        for a in 0..ncols {
            hi += x_colmajor[a * nrows + i] * tmp[a];
        }
        h[i] = hi;
    }
    Ok(h)
}

fn cluster_meat(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    groups: &[u32],
) -> Result<Vec<f64>, StatsError> {
    // Map cluster id → score sum vector.
    let mut order: Vec<usize> = (0..nrows).collect();
    order.sort_by_key(|&i| groups[i]);
    let mut meat = vec![0.0; ncols * ncols];
    let mut score = vec![0.0; ncols];
    let mut idx = 0usize;
    while idx < nrows {
        let g = groups[order[idx]];
        score.fill(0.0);
        while idx < nrows && groups[order[idx]] == g {
            let i = order[idx];
            let e = residuals[i];
            for c in 0..ncols {
                score[c] += e * x_colmajor[c * nrows + i];
            }
            idx += 1;
        }
        for a in 0..ncols {
            for b in 0..ncols {
                meat[a * ncols + b] += score[a] * score[b];
            }
        }
    }
    Ok(meat)
}

fn cluster_finite_sample(n: usize, p: usize, g: usize) -> f64 {
    // Standard cluster DF correction: (G/(G−1)) · ((n−1)/(n−p)).
    if g <= 1 {
        return 1.0;
    }
    (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / (n as f64 - p as f64).max(1.0))
}

fn multiway_meat(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    dimensions: &[&[u32]],
) -> Result<Vec<f64>, StatsError> {
    // Two-way: V1 + V2 − V12. General: inclusion–exclusion over non-empty subsets.
    let d = dimensions.len();
    if d > 4 {
        return Err(StatsError::Shape {
            message: "multiway supports at most 4 dimensions",
        });
    }
    let mut meat = vec![0.0; ncols * ncols];
    let n_sub = 1usize << d;
    for mask in 1..n_sub {
        let mut combined = vec![0u32; nrows];
        let mut bit = 0u32;
        for (dim_i, groups) in dimensions.iter().enumerate() {
            if (mask & (1 << dim_i)) != 0 {
                // Pack dimension labels into a composite key.
                for r in 0..nrows {
                    combined[r] = combined[r].wrapping_mul(1_000_003).wrapping_add(groups[r] + 1);
                }
                bit += 1;
            }
        }
        let part = cluster_meat(x_colmajor, nrows, ncols, residuals, &combined)?;
        let g = distinct_count(&combined);
        let scale = cluster_finite_sample(nrows, ncols, g);
        let sign = if bit % 2 == 1 { 1.0 } else { -1.0 };
        for k in 0..meat.len() {
            meat[k] += sign * scale * part[k];
        }
    }
    Ok(meat)
}

fn newey_west_meat(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    lag: usize,
) -> Result<Vec<f64>, StatsError> {
    let rows: Vec<usize> = (0..nrows).collect();
    newey_west_meat_on_rows(x_colmajor, nrows, ncols, residuals, &rows, lag)
}

/// Newey–West meat restricted to an ordered row subset (panel unit time path).
fn newey_west_meat_on_rows(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    rows: &[usize],
    lag: usize,
) -> Result<Vec<f64>, StatsError> {
    let t_len = rows.len();
    let mut scores = vec![0.0; t_len * ncols];
    for (t, &i) in rows.iter().enumerate() {
        let e = residuals[i];
        for c in 0..ncols {
            scores[t * ncols + c] = e * x_colmajor[c * nrows + i];
        }
    }
    let mut meat = vec![0.0; ncols * ncols];
    for t in 0..t_len {
        for a in 0..ncols {
            for b in 0..ncols {
                meat[a * ncols + b] += scores[t * ncols + a] * scores[t * ncols + b];
            }
        }
    }
    let l_max = lag.min(t_len.saturating_sub(1));
    for ell in 1..=l_max {
        let w = 1.0 - (ell as f64) / ((lag as f64) + 1.0);
        let mut gamma = vec![0.0; ncols * ncols];
        for t in ell..t_len {
            for a in 0..ncols {
                for b in 0..ncols {
                    gamma[a * ncols + b] +=
                        scores[t * ncols + a] * scores[(t - ell) * ncols + b];
                }
            }
        }
        for a in 0..ncols {
            for b in 0..ncols {
                let g_ab = gamma[a * ncols + b];
                let g_ba = gamma[b * ncols + a];
                meat[a * ncols + b] += w * (g_ab + g_ba);
            }
        }
    }
    Ok(meat)
}

fn panel_cluster_hac_meat(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    groups: &[u32],
    lag: usize,
) -> Result<Vec<f64>, StatsError> {
    // Preserve relative order within each unit (caller must time-order rows).
    let mut order: Vec<usize> = (0..nrows).collect();
    order.sort_by_key(|&i| groups[i]);
    let mut meat = vec![0.0; ncols * ncols];
    let mut idx = 0usize;
    while idx < nrows {
        let g = groups[order[idx]];
        let start = idx;
        while idx < nrows && groups[order[idx]] == g {
            idx += 1;
        }
        let rows = &order[start..idx];
        let part = newey_west_meat_on_rows(x_colmajor, nrows, ncols, residuals, rows, lag)?;
        for k in 0..meat.len() {
            meat[k] += part[k];
        }
    }
    Ok(meat)
}

fn accumulate_xx(
    meat: &mut [f64],
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    row: usize,
    weight: f64,
) {
    for a in 0..ncols {
        let xa = x_colmajor[a * nrows + row];
        for b in 0..ncols {
            meat[a * ncols + b] += weight * xa * x_colmajor[b * nrows + row];
        }
    }
}

fn sandwich_product(bread: &[f64], meat: &[f64], ncols: usize) -> Vec<f64> {
    // bread * meat * bread
    let mut tmp = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            let mut s = 0.0;
            for k in 0..ncols {
                s += bread[i * ncols + k] * meat[k * ncols + j];
            }
            tmp[i * ncols + j] = s;
        }
    }
    let mut out = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            let mut s = 0.0;
            for k in 0..ncols {
                s += tmp[i * ncols + k] * bread[k * ncols + j];
            }
            out[i * ncols + j] = s;
        }
    }
    out
}

fn distinct_count(groups: &[u32]) -> usize {
    let mut v = groups.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc0_matches_manual_two_row() {
        // X = [1,0; 1,1], y residuals e = [1, -1]
        let x = vec![1.0, 1.0, 0.0, 1.0]; // col-major 2×2
        let e = vec![1.0, -1.0];
        let cov = coefficient_covariance(&x, 2, 2, &e, SandwichKind::Hc0).unwrap();
        assert!(cov[0].is_finite() && cov[3].is_finite());
        // Diagonal entries positive.
        assert!(cov[0] > 0.0);
        assert!(cov[3] > 0.0);
    }

    #[test]
    fn cluster_se_exceeds_homoskedastic_under_correlation() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut e = vec![0.0; n];
        let mut groups = vec![0u32; n];
        for i in 0..n {
            let g = (i / 8) as u32;
            groups[i] = g;
            let t = (i % 8) as f64 / 7.0;
            x[i] = 1.0;
            x[n + i] = t;
            // Strong within-cluster residual shock + small idiosyncratic noise.
            e[i] = (g as f64) * 1.5 + if i % 2 == 0 { 0.05 } else { -0.05 };
        }
        let homo = coefficient_covariance(&x, n, 2, &e, SandwichKind::Homoskedastic).unwrap();
        let cl =
            coefficient_covariance(&x, n, 2, &e, SandwichKind::Cluster { groups: &groups })
                .unwrap();
        let se_homo = homo[0].sqrt();
        let se_cl = cl[0].sqrt();
        assert!(
            se_cl > se_homo,
            "cluster intercept SE {se_cl} should exceed homo {se_homo}"
        );
    }

    #[test]
    fn newey_west_finite() {
        let n = 30usize;
        let mut x = vec![0.0; n * 2];
        let mut e = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            e[i] = ((i % 3) as f64) - 1.0;
        }
        let cov =
            coefficient_covariance(&x, n, 2, &e, SandwichKind::NeweyWest { lag: 2 }).unwrap();
        assert!(cov.iter().all(|v| v.is_finite()));
        assert!(cov[0] > 0.0);
    }

    #[test]
    fn sandwich_kinds_match_closed_form_four_row() {
        // X rows (1,0),(1,1),(1,2),(1,3); e = [1,-0.5,0.25,-0.75].
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let e = vec![1.0, -0.5, 0.25, -0.75];
        let check = |kind: SandwichKind<'_>, expected: &[f64]| {
            let cov = coefficient_covariance(&x, 4, 2, &e, kind).unwrap();
            for (a, b) in cov.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-9, "got {cov:?} expected {expected:?}");
            }
        };
        check(SandwichKind::Hc0, &[0.553125, -0.253125, -0.253125, 0.14375]);
        check(SandwichKind::Hc1, &[1.10625, -0.50625, -0.50625, 0.2875]);
        check(
            SandwichKind::Hc2,
            &[1.766369047619049, -0.8258928571428579, -0.8258928571428581, 0.47321428571428625],
        );
        check(
            SandwichKind::Hc3,
            &[5.777352607709759, -2.727465986394562, -2.7274659863945616, 1.5688775510204103],
        );
        check(SandwichKind::Homoskedastic, &[0.65625, -0.28125, -0.28125, 0.1875]);
        check(
            SandwichKind::NeweyWest { lag: 1 },
            &[0.411875, -0.2084375, -0.2084375, 0.124375],
        );
    }

    #[test]
    fn panel_cluster_hac_exceeds_stacked_newey_west_bridge() {
        // Two units, each AR(1) residuals; stacked NW incorrectly bridges the seam.
        let t = 40usize;
        let n = 2 * t;
        let mut x = vec![0.0; n * 2];
        let mut e = vec![0.0; n];
        let mut groups = vec![0u32; n];
        for u in 0..2u32 {
            let mut prev = 1.0;
            for i in 0..t {
                let r = (u as usize) * t + i;
                groups[r] = u;
                x[r] = 1.0;
                x[n + r] = i as f64 / t as f64;
                // Strong AR(1) within unit; unit 1 starts with opposite shock.
                let innov = if i == 0 {
                    if u == 0 { 1.0 } else { -1.0 }
                } else {
                    0.05 * if i % 2 == 0 { 1.0 } else { -1.0 }
                };
                prev = 0.9 * prev + innov;
                e[r] = prev;
            }
        }
        let homo = coefficient_covariance(&x, n, 2, &e, SandwichKind::Homoskedastic).unwrap();
        let nw = coefficient_covariance(&x, n, 2, &e, SandwichKind::NeweyWest { lag: 4 }).unwrap();
        let panel = coefficient_covariance(
            &x,
            n,
            2,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, lag: 4 },
        )
        .unwrap();
        let se_h = homo[0].sqrt();
        let se_nw = nw[0].sqrt();
        let se_p = panel[0].sqrt();
        assert!(se_p > se_h, "panel {se_p} vs homo {se_h}");
        // Panel HAC should not equal stacked NW (seam bridging differs).
        assert!((se_p - se_nw).abs() > 1e-6, "panel={se_p} stacked_nw={se_nw}");
    }
}

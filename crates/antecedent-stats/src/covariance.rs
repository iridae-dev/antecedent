//! Sandwich / HAC coefficient covariance estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use crate::cluster::{
    MAX_CLUSTER_DIMENSIONS, MAX_MULTIWAY_WORK, bartlett_weight, effective_nw_lag,
    intern_cluster_tuples, multiway_work,
    multiway_subset_masks, panel_hac_meat_matrix, validate_panel_hac,
};
use crate::error::StatsError;
use crate::gram::{form_xtx, invert_square};

/// Sandwich / HAC covariance kind.
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
    /// Multiway cluster-robust (Cameron–Gelbach–Miller 2011 inclusion–exclusion).
    Multiway {
        /// One group-id slice per clustering dimension (each length `nrows`).
        dimensions: &'a [&'a [u32]],
    },
    /// Newey–West (1987) HAC with Bartlett kernel and given max lag.
    NeweyWest {
        /// Maximum lag (inclusive).
        lag: usize,
    },
    /// Panel cluster + within-unit temporal HAC (Arellano 1987-style).
    ///
    /// Lag products use explicit integer `time` labels within each unit.
    /// Cross-unit lag products are never formed. Finite-sample cluster DF
    /// correction uses `G = #units`.
    ///
    /// `lag = 0` is **not** "no serial correlation": it is the full Arellano cluster meat
    /// `Σ_g s_g s_g'`, which keeps every within-unit cross product at weight 1 and so allows
    /// arbitrary within-unit dependence. `lag = L ≥ 1` is a Bartlett-truncated within-unit HAC
    /// (weights `1 − ℓ/(L_eff+1) < 1`, lags beyond `L_eff` dropped) and therefore accounts
    /// for *less* dependence than `lag = 0`. A requested lag above the panel's widest time
    /// span is capped; see [`crate::panel_effective_lag`]. `(unit, time)` must be unique at
    /// every lag.
    PanelClusterHac {
        /// Unit id per row (length `nrows`).
        groups: &'a [u32],
        /// Calendar / index time per row (length `nrows`); unique with `groups`.
        time: &'a [i64],
        /// Bartlett max lag within each unit (`0` = full cluster meat, see above).
        lag: usize,
    },
}

/// Coefficient covariance `p×p` (row-major) from design + residuals.
///
/// Consumes retained residuals and `X` only — does not refit.
/// Bread is `(XᵀX)⁻¹` (OLS / working-residual sandwich).
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
    sandwich_from_multipliers(x_colmajor, nrows, ncols, residuals, None, kind)
}

/// HC2 variance of one OLS coefficient with its Bell–McCaffrey degrees of freedom.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Hc2Coefficient {
    /// HC2 sandwich variance of the coefficient.
    pub variance: f64,
    /// Satterthwaite degrees of freedom of the variance estimate under homoskedastic errors.
    pub degrees_of_freedom: f64,
}

/// HC2 variance of coefficient `coefficient` and its Bell–McCaffrey (2002) degrees of freedom,
/// the small-sample reference of Imbens and Kolesár (2016).
///
/// With `w = X (XᵀX)⁻¹ e_c`, leverages `h_i` and `M = I − H`, the HC2 variance is the quadratic
/// form `εᵀ A ε` in the errors with `A = M D M`, `D = diag(w_i² / (1 − h_i))`. Its mean under
/// homoskedastic errors is `σ² Σ w_i²`, so the Satterthwaite degrees of freedom are
/// `(tr A)² / tr(A²)`, with `tr A = Σ w_i²` and
/// `tr(A²) = Σ_i d_i² (1 − 2 h_i) + tr((BS)²)` for `B = (XᵀX)⁻¹`, `S = Σ d_i x_i x_iᵀ`, which
/// costs `O(n p²)` rather than forming `M`. For an intercept and a binary regressor the variance
/// is `s₁²/n₁ + s₀²/n₀` and the degrees of freedom reduce to Welch's with equal arm variances.
///
/// # Errors
///
/// Shape mismatch, a singular design, a row of leverage 1 (HC2 is undefined there), or degrees
/// of freedom that are not finite and positive.
pub fn hc2_coefficient_bell_mccaffrey(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    residuals: &[f64],
    coefficient: usize,
) -> Result<Hc2Coefficient, StatsError> {
    if coefficient >= ncols {
        return Err(StatsError::Shape { message: "coefficient index outside the design" });
    }
    let covariance =
        coefficient_covariance(x_colmajor, nrows, ncols, residuals, SandwichKind::Hc2)?;
    let variance = covariance[coefficient * ncols + coefficient];
    let mut gram = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut gram);
    let bread = invert_square(&gram, ncols)
        .ok_or_else(|| StatsError::Backend("singular sandwich bread".into()))?;
    let hat = leverages(x_colmajor, nrows, ncols, &bread, None);
    let selector = &bread[coefficient * ncols..(coefficient + 1) * ncols];
    let (mut trace, mut diagonal) = (0.0, 0.0);
    let mut weighted_gram = vec![0.0; ncols * ncols];
    for (i, h) in hat.iter().enumerate() {
        let w: f64 = (0..ncols).map(|a| selector[a] * x_colmajor[a * nrows + i]).sum();
        let d = w * w / (1.0 - h);
        trace += w * w;
        diagonal += d * d * (1.0 - 2.0 * h);
        accumulate_xx(&mut weighted_gram, x_colmajor, nrows, ncols, i, d);
    }
    let mut product_trace = 0.0;
    for a in 0..ncols {
        for b in 0..ncols {
            let bs_ab: f64 =
                (0..ncols).map(|k| bread[a * ncols + k] * weighted_gram[k * ncols + b]).sum();
            let bs_ba: f64 =
                (0..ncols).map(|k| bread[b * ncols + k] * weighted_gram[k * ncols + a]).sum();
            product_trace += bs_ab * bs_ba;
        }
    }
    let degrees_of_freedom = trace * trace / (diagonal + product_trace);
    if !(variance.is_finite() && degrees_of_freedom.is_finite() && degrees_of_freedom > 0.0) {
        return Err(StatsError::Backend(
            "Bell–McCaffrey degrees of freedom are not finite and positive".into(),
        ));
    }
    Ok(Hc2Coefficient { variance, degrees_of_freedom })
}

/// Score / GLM sandwich: meat from score multipliers `u_i` (`s_i = u_i x_i`),
/// bread `(XᵀWX)⁻¹` with diagonal Fisher weights `w`.
///
/// For identity-Gaussian with `u = y−μ` and `w = 1` and a robust `kind` this matches
/// [`coefficient_covariance`]. Does not refit.
///
/// [`SandwichKind::Homoskedastic`] is refused on this path: the model-based GLM covariance
/// is `φ (XᵀWX)⁻¹` and this function has no dispersion `φ`, so returning the bare Fisher
/// bread would silently assume `φ = 1`. Callers with a known dispersion scale the bread
/// themselves; every other kind is dispersion-free.
///
/// # Errors
///
/// Shape mismatch, empty design, singular bread matrix, or `Homoskedastic`.
pub fn score_coefficient_covariance(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    score_multipliers: &[f64],
    fisher_weights: &[f64],
    kind: SandwichKind<'_>,
) -> Result<Vec<f64>, StatsError> {
    if fisher_weights.len() != nrows {
        return Err(StatsError::Shape { message: "fisher_weights length != nrows" });
    }
    if matches!(kind, SandwichKind::Homoskedastic) {
        return Err(StatsError::Shape {
            message: "score covariance has no Homoskedastic form: it needs a dispersion, \
                      use a robust kind or scale (X'WX)^-1 by the model dispersion",
        });
    }
    sandwich_from_multipliers(
        x_colmajor,
        nrows,
        ncols,
        score_multipliers,
        Some(fisher_weights),
        kind,
    )
}

fn sandwich_from_multipliers(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    multipliers: &[f64],
    fisher_weights: Option<&[f64]>,
    kind: SandwichKind<'_>,
) -> Result<Vec<f64>, StatsError> {
    if multipliers.len() != nrows {
        return Err(StatsError::Shape { message: "score/residual length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if nrows == 0 || ncols == 0 {
        return Err(StatsError::Shape { message: "covariance needs positive dimensions" });
    }

    if x_colmajor[..nrows * ncols].iter().chain(multipliers).any(|v| !v.is_finite()) {
        return Err(StatsError::Backend("covariance inputs must be finite".into()));
    }
    if fisher_weights.is_some_and(|w| w.iter().any(|v| !v.is_finite() || *v < 0.0)) {
        return Err(StatsError::Backend("Fisher weights must be finite and nonnegative".into()));
    }

    let mut gram = vec![0.0; ncols * ncols];
    match fisher_weights {
        None => form_xtx(x_colmajor, nrows, ncols, &mut gram),
        Some(w) => form_xtwx(x_colmajor, nrows, ncols, w, &mut gram),
    }
    let Some(bread) = invert_square(&gram, ncols) else {
        return Err(StatsError::Backend("singular sandwich bread".into()));
    };

    match kind {
        SandwichKind::Homoskedastic => {
            if fisher_weights.is_some() {
                // Unreachable through the public entry points (the score path refuses
                // `Homoskedastic`); a bare Fisher bread would assume dispersion 1.
                return Err(StatsError::Shape {
                    message: "homoskedastic covariance needs a dispersion on the score path",
                });
            }
            if nrows <= ncols {
                return Err(StatsError::Shape { message: "non-positive residual df" });
            }
            let rss: f64 = multipliers.iter().map(|e| e * e).sum();
            let sigma2 = rss / (nrows as f64 - ncols as f64);
            Ok(bread.iter().map(|v| v * sigma2).collect())
        }
        SandwichKind::Hc0 | SandwichKind::Hc1 | SandwichKind::Hc2 | SandwichKind::Hc3 => {
            let meat =
                hc_meat(x_colmajor, nrows, ncols, multipliers, &bread, fisher_weights, kind)?;
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::Cluster { groups } => {
            if groups.len() != nrows {
                return Err(StatsError::Shape { message: "cluster groups length != nrows" });
            }
            let meat = cluster_meat(x_colmajor, nrows, ncols, multipliers, groups)?;
            let g = distinct_count(groups);
            let scale = cluster_finite_sample(nrows, ncols, g)?;
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
            let meat = multiway_meat(x_colmajor, nrows, ncols, multipliers, dimensions)?;
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::NeweyWest { lag } => {
            let meat = newey_west_meat(x_colmajor, nrows, ncols, multipliers, lag)?;
            Ok(sandwich_product(&bread, &meat, ncols))
        }
        SandwichKind::PanelClusterHac { groups, time, lag } => {
            if groups.len() != nrows {
                return Err(StatsError::Shape { message: "panel HAC groups length != nrows" });
            }
            if time.len() != nrows {
                return Err(StatsError::Shape { message: "panel HAC time length != nrows" });
            }
            // lag = 0 is Arellano/cluster meat (Σ_g s_g s_g'), not White Σ u_it². The same
            // (unit, time) uniqueness contract applies as for lag ≥ 1.
            if lag == 0 {
                validate_panel_hac(multipliers, groups, time)?;
                let meat = cluster_meat(x_colmajor, nrows, ncols, multipliers, groups)?;
                let g = distinct_count(groups);
                let scale = cluster_finite_sample(nrows, ncols, g)?;
                let meat: Vec<f64> = meat.iter().map(|v| v * scale).collect();
                return Ok(sandwich_product(&bread, &meat, ncols));
            }
            let (meat, g) =
                panel_hac_meat_matrix(x_colmajor, nrows, ncols, multipliers, groups, time, lag)?;
            let scale = cluster_finite_sample(nrows, ncols, g)?;
            let meat: Vec<f64> = meat.iter().map(|v| v * scale).collect();
            Ok(sandwich_product(&bread, &meat, ncols))
        }
    }
}

/// Fill symmetric `XᵀWX` (row-major) with diagonal weights `w`.
fn form_xtwx(x_colmajor: &[f64], nrows: usize, ncols: usize, w: &[f64], xtwx: &mut [f64]) {
    xtwx[..ncols * ncols].fill(0.0);
    for c1 in 0..ncols {
        for c2 in c1..ncols {
            let mut acc = 0.0;
            let col1 = &x_colmajor[c1 * nrows..(c1 + 1) * nrows];
            let col2 = &x_colmajor[c2 * nrows..(c2 + 1) * nrows];
            for r in 0..nrows {
                acc += w[r] * col1[r] * col2[r];
            }
            xtwx[c1 * ncols + c2] = acc;
            if c1 != c2 {
                xtwx[c2 * ncols + c1] = acc;
            }
        }
    }
}

/// A leverage within this distance of 1 is treated as exactly 1 (HC2/HC3 undefined).
const LEVERAGE_SATURATION: f64 = 1e-8;

#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn hc_meat(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    multipliers: &[f64],
    bread: &[f64],
    fisher_weights: Option<&[f64]>,
    kind: SandwichKind<'_>,
) -> Result<Vec<f64>, StatsError> {
    let hat = match kind {
        SandwichKind::Hc2 | SandwichKind::Hc3 => {
            let hat = leverages(x_colmajor, nrows, ncols, bread, fisher_weights);
            // `h_ii = 1` (a row that alone identifies a coefficient, e.g. its own dummy)
            // makes the residual exactly 0 and the `e²/(1−h)` adjustment 0/0: the estimator
            // is undefined there. Clamping `h` would inflate rounding-noise residuals by
            // `1/(1−h)²`, so refuse instead and name the rows.
            let saturated: Vec<usize> = (0..nrows)
                .filter(|&i| {
                    hat[i]
                        .partial_cmp(&(1.0 - LEVERAGE_SATURATION))
                        .is_none_or(std::cmp::Ordering::is_gt)
                })
                .collect();
            if !saturated.is_empty() {
                let shown: Vec<String> =
                    saturated.iter().take(5).map(ToString::to_string).collect();
                return Err(StatsError::Backend(format!(
                    "HC2/HC3 are undefined for {} row(s) with leverage 1 (rows {}{}); \
                     use HC0/HC1 or drop the saturated rows",
                    saturated.len(),
                    shown.join(", "),
                    if saturated.len() > shown.len() { ", …" } else { "" },
                )));
            }
            Some(hat)
        }
        _ => None,
    };
    let mut meat = vec![0.0; ncols * ncols];
    for i in 0..nrows {
        let e = multipliers[i];
        let adj = match kind {
            SandwichKind::Hc0 | SandwichKind::Hc1 => e * e,
            SandwichKind::Hc2 => {
                let h = hc_leverage(hat.as_deref(), i)?;
                (e * e) / (1.0 - h)
            }
            SandwichKind::Hc3 => {
                let h = hc_leverage(hat.as_deref(), i)?;
                let d = 1.0 - h;
                (e * e) / (d * d)
            }
            SandwichKind::Homoskedastic
            | SandwichKind::Cluster { .. }
            | SandwichKind::Multiway { .. }
            | SandwichKind::NeweyWest { .. }
            | SandwichKind::PanelClusterHac { .. } => {
                return Err(StatsError::Shape { message: "HC meat is only defined for HC0–HC3" });
            }
        };
        accumulate_xx(&mut meat, x_colmajor, nrows, ncols, i, adj);
    }
    if matches!(kind, SandwichKind::Hc1) {
        if nrows <= ncols {
            return Err(StatsError::Shape { message: "non-positive residual df" });
        }
        let scale = nrows as f64 / (nrows as f64 - ncols as f64);
        for v in &mut meat {
            *v *= scale;
        }
    }
    Ok(meat)
}

fn hc_leverage(hat: Option<&[f64]>, row: usize) -> Result<f64, StatsError> {
    let h = hat.ok_or(StatsError::Shape { message: "HC2/HC3 leverage vector missing" })?;
    h.get(row)
        .copied()
        .ok_or(StatsError::Shape { message: "HC2/HC3 leverage vector is shorter than the design" })
        .map(|v| v.max(0.0))
}

fn leverages(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    bread: &[f64],
    fisher_weights: Option<&[f64]>,
) -> Vec<f64> {
    let mut h = vec![0.0; nrows];
    let mut tmp = vec![0.0; ncols];
    for i in 0..nrows {
        // Unweighted: h_ii = x_i' (X'X)⁻¹ x_i
        // Weighted:   h_ii = w_i x_i' (X'WX)⁻¹ x_i
        for a in 0..ncols {
            let mut s = 0.0;
            for b in 0..ncols {
                s += bread[a * ncols + b] * x_colmajor[b * nrows + i];
            }
            tmp[a] = s;
        }
        let mut hi = 0.0;
        for a in 0..ncols {
            hi += x_colmajor[a * nrows + i] * tmp[a];
        }
        if let Some(w) = fisher_weights {
            hi *= w[i];
        }
        h[i] = hi;
    }
    h
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result to match the fallible sibling paths it is dispatched alongside"
)]
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

fn cluster_finite_sample(n: usize, p: usize, g: usize) -> Result<f64, StatsError> {
    // Standard cluster DF correction: (G/(G−1)) · ((n−1)/(n−p)).
    if g < 2 {
        return Err(StatsError::FewClusters {
            clusters: u64::try_from(g).unwrap_or(u64::MAX),
            minimum: 2,
            rendered: "shape error: cluster-robust variance requires at least 2 clusters",
        });
    }
    if n <= p {
        return Err(StatsError::Shape { message: "non-positive residual df" });
    }
    Ok((g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / (n as f64 - p as f64)))
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
    if d == 0 || d > MAX_CLUSTER_DIMENSIONS {
        return Err(StatsError::Shape {
            message: "multiway supports 1..=MAX_CLUSTER_DIMENSIONS dimensions",
        });
    }
    if multiway_work(d, nrows).is_none_or(|work| work > MAX_MULTIWAY_WORK) {
        return Err(StatsError::Shape {
            message: "multiway inclusion-exclusion work (2^d - 1) * n exceeds MAX_MULTIWAY_WORK",
        });
    }
    let mut meat = vec![0.0; ncols * ncols];
    let mut abs_diag = vec![0.0; ncols];
    let mut combined = vec![0u32; nrows];
    for (mask, sign) in multiway_subset_masks(d) {
        let g = intern_cluster_tuples(dimensions, mask, &mut combined)?;
        let part = cluster_meat(x_colmajor, nrows, ncols, residuals, &combined)?;
        let scale = cluster_finite_sample(nrows, ncols, g)?;
        for k in 0..meat.len() {
            meat[k] += sign * scale * part[k];
        }
        for j in 0..ncols {
            abs_diag[j] += (sign * scale * part[j * ncols + j]).abs();
        }
    }
    // CGM meat can be indefinite with nonnegative diagonals; refuse rather than
    // project onto the PSD cone (downstream SE clamps would otherwise publish 0).
    ensure_multiway_meat_psd(&meat, ncols, &abs_diag)?;
    Ok(meat)
}

/// Refuse a multiway meat with any eigenvalue materially below zero.
///
/// Tolerance is `64 ε` times the scale of the diagonal (max absolute IE
/// contribution to any diagonal entry), matching scalar inclusion–exclusion.
/// Matrices that are PSD within that tolerance are left unchanged.
fn ensure_multiway_meat_psd(
    meat: &[f64],
    ncols: usize,
    abs_diag: &[f64],
) -> Result<(), StatsError> {
    if ncols == 0 {
        return Ok(());
    }
    let mat = faer::Mat::<f64>::from_fn(ncols, ncols, |r, c| meat[r * ncols + c]);
    let (values, _) = crate::faer_backend::symmetric_eigen_in_scope(
        mat.as_ref(),
        crate::faer_backend::serial_scope(),
    )
    .map_err(|_| StatsError::Backend("multiway meat eigendecomposition failed".into()))?;
    let eigs: Vec<f64> = (0..ncols).map(|k| values.as_ref().column_vector()[k]).collect();
    let scale = abs_diag.iter().copied().fold(0.0_f64, f64::max);
    let tol = 64.0 * f64::EPSILON * scale;
    for &lam in &eigs {
        if lam < -tol {
            return Err(StatsError::NonPositiveVariance {
                message: "multiway inclusion-exclusion meat is not positive semidefinite",
            });
        }
    }
    Ok(())
}

/// Eigendecomposition of a symmetric row-major `d x d` matrix (lower triangle read).
///
/// Returns the eigenvalues in nondecreasing order and the eigenvectors as the columns of a
/// row-major matrix (`vectors[row * d + k]` is component `row` of eigenvector `k`).
///
/// # Errors
///
/// A shape mismatch, a non-finite entry, or a backend failure to converge.
pub fn symmetric_eigen(matrix: &[f64], d: usize) -> Result<(Vec<f64>, Vec<f64>), StatsError> {
    if matrix.len() != d * d || matrix.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape {
            message: "symmetric eigendecomposition needs a finite d x d matrix",
        });
    }
    let mat = faer::Mat::<f64>::from_fn(d, d, |r, c| matrix[r * d + c]);
    let (eig_values, eig_vectors) = crate::faer_backend::symmetric_eigen_in_scope(
        mat.as_ref(),
        crate::faer_backend::serial_scope(),
    )?;
    let values = (0..d).map(|k| eig_values.as_ref().column_vector()[k]).collect();
    let mut vectors = vec![0.0; d * d];
    for r in 0..d {
        for k in 0..d {
            vectors[r * d + k] = eig_vectors[(r, k)];
        }
    }
    Ok((values, vectors))
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
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result to match the fallible sibling paths it is dispatched alongside"
)]
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
    let l_max = effective_nw_lag(lag, t_len.saturating_sub(1));
    for ell in 1..=l_max {
        let w = bartlett_weight(ell, l_max);
        let mut gamma = vec![0.0; ncols * ncols];
        for t in ell..t_len {
            for a in 0..ncols {
                for b in 0..ncols {
                    gamma[a * ncols + b] += scores[t * ncols + a] * scores[(t - ell) * ncols + b];
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
    fn covariance_rejects_nonfinite_inputs_and_negative_information() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                coefficient_covariance(&[1.0, bad], 2, 1, &[1.0, -1.0], SandwichKind::Hc0).is_err()
            );
            assert!(
                coefficient_covariance(&[1.0, 1.0], 2, 1, &[1.0, bad], SandwichKind::Hc0).is_err()
            );
        }
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                score_coefficient_covariance(
                    &[1.0, 1.0],
                    2,
                    1,
                    &[1.0, -1.0],
                    &[bad, 2.0],
                    SandwichKind::Hc0
                )
                .is_err()
            );
        }
    }

    /// Intercept plus a binary regressor (3 treated, 5 control) with its group-mean residuals.
    fn two_group_design() -> (Vec<f64>, Vec<f64>) {
        let treated = [1.0, 2.0, 6.0];
        let control = [0.0, 1.0, 1.0, 2.0, 6.0];
        let mut x = vec![1.0; 8];
        x.extend([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let residuals = treated
            .iter()
            .map(|y| y - mean(&treated))
            .chain(control.iter().map(|y| y - mean(&control)))
            .collect();
        (x, residuals)
    }

    #[test]
    fn hc2_two_group_variance_is_the_per_arm_variance_sum() {
        // s₁² = 14/2 = 7 on 3 rows, s₀² = 22/4 = 5.5 on 5 rows.
        let (x, residuals) = two_group_design();
        let fit = hc2_coefficient_bell_mccaffrey(&x, 8, 2, &residuals, 1).unwrap();
        assert!((fit.variance - (7.0 / 3.0 + 5.5 / 5.0)).abs() < 1e-12, "{}", fit.variance);
    }

    #[test]
    fn bell_mccaffrey_two_group_df_matches_the_hand_computation() {
        // (1/n₁ + 1/n₀)² / (1/(n₁²(n₁−1)) + 1/(n₀²(n₀−1))) = (8/15)² / (1/18 + 1/100) = 512/118.
        let (x, residuals) = two_group_design();
        let fit = hc2_coefficient_bell_mccaffrey(&x, 8, 2, &residuals, 1).unwrap();
        assert!(
            (fit.degrees_of_freedom - 512.0 / 118.0).abs() < 1e-10,
            "{}",
            fit.degrees_of_freedom
        );
        // Two rows per arm: h = 1/2, w = ±1/2, d = 1/2, tr(A) = 1, tr(A²) = 1/2, so df = 2.
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let residuals = [0.5, -0.5, 1.0, -1.0];
        let fit = hc2_coefficient_bell_mccaffrey(&x, 4, 2, &residuals, 1).unwrap();
        assert!((fit.degrees_of_freedom - 2.0).abs() < 1e-12, "{}", fit.degrees_of_freedom);
    }

    #[test]
    fn bell_mccaffrey_refuses_a_saturated_row_and_a_bad_coefficient() {
        // One treated row alone identifies the treatment coefficient: leverage 1.
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        assert!(hc2_coefficient_bell_mccaffrey(&x, 4, 2, &[0.0, 0.5, -0.5, 0.0], 1).is_err());
        assert!(hc2_coefficient_bell_mccaffrey(&x, 4, 2, &[0.0, 0.5, -0.5, 0.0], 2).is_err());
    }

    #[test]
    fn hc0_intercept_only_is_sum_squares_over_n_squared() {
        // X = 1, bread = 1/n, meat = Σe²: Var = Σe² / n² = 10.5 / 36.
        let e = [1.0, 2.0, -1.0, -2.0, 0.5, 0.5];
        let x = vec![1.0; 6];
        let cov = coefficient_covariance(&x, 6, 1, &e, SandwichKind::Hc0).unwrap();
        assert!((cov[0] - 10.5 / 36.0).abs() < 1e-14, "{}", cov[0]);
    }

    #[test]
    fn cluster_intercept_only_matches_closed_form() {
        // Cluster sums S = [3, −3, 1]; G = 3, p = 1 ⇒ factor G/(G−1) · (n−1)/(n−p) = 1.5.
        // Var = 1.5 · ΣS² / n² = 1.5 · 19 / 36 = 19/24.
        let e = [1.0, 2.0, -1.0, -2.0, 0.5, 0.5];
        let groups = [0u32, 0, 1, 1, 2, 2];
        let x = vec![1.0; 6];
        let cov = coefficient_covariance(&x, 6, 1, &e, SandwichKind::Cluster { groups: &groups })
            .unwrap();
        assert!((cov[0] - 19.0 / 24.0).abs() < 1e-14, "{}", cov[0]);
    }

    #[test]
    fn cluster_two_column_matches_hand_sandwich() {
        // X rows (1,0),(1,1),(1,2),(1,3); e = [1,−.5,.25,−.75]; clusters {0,1},{2,3}.
        // Cluster score sums (.5,−.5) and (−.5,−1.75); meat = [[.5,.625],[.625,3.3125]];
        // factor G/(G−1)·(n−1)/(n−p) = 2 · 3/2 = 3; bread = [[.7,−.3],[−.3,.2]].
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let e = vec![1.0, -0.5, 0.25, -0.75];
        let groups = [0u32, 0, 1, 1];
        let cov = coefficient_covariance(&x, 4, 2, &e, SandwichKind::Cluster { groups: &groups })
            .unwrap();
        let expected = [0.841_875, -0.48, -0.48, 0.3075];
        for (g, w) in cov.iter().zip(expected) {
            assert!((g - w).abs() < 1e-12, "got {cov:?} expected {expected:?}");
        }
    }

    #[test]
    fn two_way_intercept_only_matches_inclusion_exclusion() {
        // e = 1..4, A = {01|23}, B = {02|13}. Each subset uses its own G and the factor
        // G/(G−1) (p = 1): V_A = 2·(3²+7²) = 116, V_B = 2·(4²+6²) = 104,
        // V_AB (G = 4, singleton cells) = 4/3 · 30 = 40 ⇒ meat = 180, Var = 180/16.
        let x = vec![1.0; 4];
        let e = [1.0, 2.0, 3.0, 4.0];
        let dim_a = [0u32, 0, 1, 1];
        let dim_b = [0u32, 1, 0, 1];
        let dims: [&[u32]; 2] = [&dim_a, &dim_b];
        let cov =
            coefficient_covariance(&x, 4, 1, &e, SandwichKind::Multiway { dimensions: &dims })
                .unwrap();
        assert!((cov[0] - 11.25).abs() < 1e-13, "{}", cov[0]);
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
            e[i] = f64::from(g) * 1.5 + if i % 2 == 0 { 0.05 } else { -0.05 };
        }
        let homo = coefficient_covariance(&x, n, 2, &e, SandwichKind::Homoskedastic).unwrap();
        let cl = coefficient_covariance(&x, n, 2, &e, SandwichKind::Cluster { groups: &groups })
            .unwrap();
        let se_homo = homo[0].sqrt();
        let se_cl = cl[0].sqrt();
        assert!(se_cl > se_homo, "cluster intercept SE {se_cl} should exceed homo {se_homo}");
    }

    #[test]
    fn newey_west_intercept_only_matches_closed_form() {
        // e = [1,2,3], lag 1: Γ₀ = 14, Γ₁ = 2 + 6 = 8, Bartlett weight 1 − 1/2.
        // meat = 14 + 2 · 0.5 · 8 = 22; Var = 22 / 3².
        let x = vec![1.0; 3];
        let e = [1.0, 2.0, 3.0];
        let cov = coefficient_covariance(&x, 3, 1, &e, SandwichKind::NeweyWest { lag: 1 }).unwrap();
        assert!((cov[0] - 22.0 / 9.0).abs() < 1e-14, "{}", cov[0]);
    }

    #[test]
    fn panel_hac_intercept_only_matches_closed_form() {
        // Units e = [1,2,3] and [1,−1,2] at t = 0,1,2; Σe² = 20. Lag 1 (weight ½):
        // products 2+6 and −1−2 sum to 5 ⇒ meat 20 + 2·½·5 = 25; G = 2 ⇒ factor 2;
        // Var = 50/36. Lag 0 is the full cluster meat (6² + 2²)·2 = 80 ⇒ 80/36, which is
        // *larger* than lag 1: lag 0 keeps all cross products at weight 1.
        let x = vec![1.0; 6];
        let e = [1.0, 2.0, 3.0, 1.0, -1.0, 2.0];
        let groups = [0u32, 0, 0, 1, 1, 1];
        let time = [0i64, 1, 2, 0, 1, 2];
        let at = |lag| {
            coefficient_covariance(
                &x,
                6,
                1,
                &e,
                SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag },
            )
            .unwrap()[0]
        };
        assert!((at(1) - 50.0 / 36.0).abs() < 1e-14, "{}", at(1));
        assert!((at(0) - 80.0 / 36.0).abs() < 1e-14, "{}", at(0));
    }

    #[test]
    fn panel_hac_lag_zero_rejects_duplicate_unit_time() {
        let x = vec![1.0; 4];
        let e = [1.0, 2.0, 3.0, 4.0];
        let groups = [0u32, 0, 1, 1];
        let time = [0i64, 0, 0, 1]; // (unit 0, time 0) twice
        for lag in [0usize, 1] {
            let err = coefficient_covariance(
                &x,
                4,
                1,
                &e,
                SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag },
            )
            .unwrap_err();
            assert!(err.to_string().contains("unique (cluster, time)"), "lag={lag} err={err}");
        }
    }

    #[test]
    fn panel_hac_time_labels_at_i64_min_do_not_overflow() {
        // `time − ℓ` would overflow at i64::MIN. Units e = [1,2] and [3,4] at (MIN, MIN+1):
        // lag-1 products 2 and 12, weight ½ ⇒ meat 30 + 14 = 44; G = 2 ⇒ 88; Var = 88/16.
        let x = vec![1.0; 4];
        let e = [1.0, 2.0, 3.0, 4.0];
        let groups = [0u32, 0, 1, 1];
        let time = [i64::MIN, i64::MIN + 1, i64::MIN, i64::MIN + 1];
        let cov = coefficient_covariance(
            &x,
            4,
            1,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag: 1 },
        )
        .unwrap();
        assert!((cov[0] - 5.5).abs() < 1e-13, "{}", cov[0]);
    }

    #[test]
    fn panel_effective_lag_reports_the_cap() {
        let groups = [0u32, 0, 1, 1];
        let time = [0i64, 3, 0, 1];
        assert_eq!(crate::panel_effective_lag(&groups, &time, 10), 3);
        assert_eq!(crate::panel_effective_lag(&groups, &time, 2), 2);
    }

    #[test]
    fn hc2_hc3_refuse_saturated_leverage() {
        // Row 0 is the only one with d = 1, so it carries its own dummy: h₀₀ = 1 and its
        // residual is 0. HC2/HC3 are 0/0 there; HC0/HC1 remain defined.
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let e = [0.0, 0.3, -0.1, -0.2];
        for kind in [SandwichKind::Hc2, SandwichKind::Hc3] {
            let err = coefficient_covariance(&x, 4, 2, &e, kind).unwrap_err();
            assert!(err.to_string().contains("leverage 1 (rows 0"), "kind={kind:?} err={err}");
        }
        assert!(coefficient_covariance(&x, 4, 2, &e, SandwichKind::Hc0).is_ok());
    }

    #[test]
    fn score_path_refuses_homoskedastic_without_dispersion() {
        let x = vec![1.0; 4];
        let err = score_coefficient_covariance(
            &x,
            4,
            1,
            &[0.1, -0.2, 0.3, -0.1],
            &[1.0; 4],
            SandwichKind::Homoskedastic,
        )
        .unwrap_err();
        assert!(err.to_string().contains("dispersion"), "err={err}");
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
        check(SandwichKind::Hc0, &[0.553_125, -0.253_125, -0.253_125, 0.14375]);
        check(SandwichKind::Hc1, &[1.10625, -0.50625, -0.50625, 0.2875]);
        check(
            SandwichKind::Hc2,
            &[
                1.766_369_047_619_049,
                -0.825_892_857_142_857_9,
                -0.825_892_857_142_858_1,
                0.473_214_285_714_286_25,
            ],
        );
        check(
            SandwichKind::Hc3,
            &[
                5.777_352_607_709_759,
                -2.727_465_986_394_562,
                -2.727_465_986_394_561_6,
                1.568_877_551_020_410_3,
            ],
        );
        check(SandwichKind::Homoskedastic, &[0.65625, -0.28125, -0.28125, 0.1875]);
        check(
            SandwichKind::NeweyWest { lag: 1 },
            &[0.411_875, -0.208_437_5, -0.208_437_5, 0.124_375],
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
        let mut times = vec![0i64; n];
        for u in 0..2usize {
            for i in 0..t {
                times[u * t + i] = i64::try_from(i).expect("panel time index fits i64");
            }
        }
        let panel = coefficient_covariance(
            &x,
            n,
            2,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, time: &times, lag: 4 },
        )
        .unwrap();
        let se_h = homo[0].sqrt();
        let se_nw = nw[0].sqrt();
        let se_p = panel[0].sqrt();
        assert!(se_p > se_h, "panel {se_p} vs homo {se_h}");
        // Panel HAC should not equal stacked NW (seam bridging differs).
        assert!((se_p - se_nw).abs() > 1e-6, "panel={se_p} stacked_nw={se_nw}");
    }

    #[test]
    fn score_sandwich_unit_weights_matches_residual() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let e = vec![1.0, -0.5, 0.25, -0.75];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let a = coefficient_covariance(&x, 4, 2, &e, SandwichKind::Hc0).unwrap();
        let b = score_coefficient_covariance(&x, 4, 2, &e, &w, SandwichKind::Hc0).unwrap();
        for (u, v) in a.iter().zip(b.iter()) {
            assert!((u - v).abs() < 1e-12, "a={a:?} b={b:?}");
        }
    }

    #[test]
    fn multiway_packing_collision_changes_meat() {
        // Two rows with labels that collide under ×1_000_003 packing.
        let n = 4usize;
        let x = vec![1.0; n];
        let e = [1.0, -1.0, 2.0, -2.0];
        let dim_a = [1u32, 0, 2, 3];
        let dim_b = [0u32, 1_000_003, 4, 5];
        let dims: [&[u32]; 2] = [&dim_a, &dim_b];
        let cov =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::Multiway { dimensions: &dims })
                .unwrap();
        assert!(cov[0].is_finite() && cov[0] > 0.0);

        // Distinct interned intersection groups: G=4 for the two-way intersection.
        let mut out = [0u32; 4];
        let g = crate::cluster::intern_cluster_tuples(&dims, 0b11, &mut out).unwrap();
        assert_eq!(g, 4);
        // Lossy pack would map first two rows to the same key.
        let packed: Vec<u32> = dim_a
            .iter()
            .zip(dim_b.iter())
            .map(|(&a, &b)| a.wrapping_mul(1_000_003).wrapping_add(b))
            .collect();
        assert_eq!(packed[0], packed[1]);
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn one_cluster_returns_error() {
        let x = vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let e = vec![0.5, -0.2, 0.1];
        let groups = [0u32, 0, 0];
        let err = coefficient_covariance(&x, 3, 2, &e, SandwichKind::Cluster { groups: &groups })
            .unwrap_err();
        assert!(err.to_string().contains("at least 2 clusters"), "err={err}");
    }

    #[test]
    fn nonpositive_residual_df_errors_for_homo_and_hc1() {
        // n = p = 2 → residual df = 0
        let x = vec![1.0, 1.0, 0.0, 1.0];
        let e = vec![1.0, -1.0];
        for kind in [SandwichKind::Homoskedastic, SandwichKind::Hc1] {
            let err = coefficient_covariance(&x, 2, 2, &e, kind).unwrap_err();
            assert!(
                err.to_string().contains("non-positive residual df"),
                "kind={kind:?} err={err}"
            );
        }
        // HC0 remains defined without residual-DF correction.
        assert!(coefficient_covariance(&x, 2, 2, &e, SandwichKind::Hc0).is_ok());
    }

    #[test]
    fn panel_hac_one_cluster_errors() {
        let n = 6usize;
        let x = vec![1.0; n];
        let e = vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25];
        let groups = [0u32; 6];
        let time = [0i64, 1, 2, 3, 4, 5];
        let err = coefficient_covariance(
            &x,
            n,
            1,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag: 1 },
        )
        .unwrap_err();
        assert!(err.to_string().contains("at least 2 clusters"), "err={err}");
    }

    #[test]
    fn newey_west_oversized_lag_matches_capped_leff() {
        // Weights must use L_eff = min(lag, T−1), not the requested lag.
        let n = 5usize;
        let x = vec![1.0; n];
        let e = vec![1.0, -0.5, 0.25, -0.75, 0.1];
        let capped =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::NeweyWest { lag: n - 1 }).unwrap();
        let oversized =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::NeweyWest { lag: 10 }).unwrap();
        assert!(
            (capped[0] - oversized[0]).abs() < 1e-12,
            "capped={} oversized={}",
            capped[0],
            oversized[0]
        );
    }

    #[test]
    fn multiway_singleton_dimension_errors() {
        let n = 4usize;
        let x = vec![1.0; n];
        let e = vec![1.0, -1.0, 0.5, -0.5];
        let dim_a = [0u32, 0, 1, 1];
        let dim_b = [0u32, 0, 0, 0];
        let dims: [&[u32]; 2] = [&dim_a, &dim_b];
        let err =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::Multiway { dimensions: &dims })
                .unwrap_err();
        assert!(err.to_string().contains("at least 2 clusters"), "err={err}");
    }

    #[test]
    fn multiway_refuses_indefinite_two_way_meat() {
        // Two-way IE meat with nonnegative diagonals but a negative eigenvalue
        // (indefinite). Diagonals alone would pass; full PSD check must refuse.
        let n = 8usize;
        let x = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // intercept
            0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375,
        ];
        let e = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25];
        let dim_a = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let dim_b = [0u32, 1, 0, 1, 0, 1, 0, 1];
        let dims: [&[u32]; 2] = [&dim_a, &dim_b];
        let err =
            coefficient_covariance(&x, n, 2, &e, SandwichKind::Multiway { dimensions: &dims })
                .unwrap_err();
        assert!(
            matches!(
                err,
                StatsError::NonPositiveVariance {
                    message: "multiway inclusion-exclusion meat is not positive semidefinite"
                }
            ),
            "err={err}"
        );
    }

    #[test]
    fn multiway_one_way_cluster_remains_psd() {
        let n = 8usize;
        let x = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // intercept
            0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375,
        ];
        let e = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25];
        let dim_a = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let dims: [&[u32]; 1] = [&dim_a];
        let cov =
            coefficient_covariance(&x, n, 2, &e, SandwichKind::Multiway { dimensions: &dims })
                .unwrap();
        assert!(cov.iter().all(|v| v.is_finite()));
        assert!(cov[0] > 0.0 && cov[3] > 0.0);
    }

    #[test]
    fn panel_hac_lag_zero_matches_cluster() {
        let n = 8usize;
        let x = vec![1.0; n];
        let e = vec![1.0, 0.5, 0.25, -1.0, -0.5, -0.25, 0.75, 0.4];
        let groups = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let time = [0i64, 1, 2, 3, 0, 1, 2, 3];
        let panel = coefficient_covariance(
            &x,
            n,
            1,
            &e,
            SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag: 0 },
        )
        .unwrap();
        let cluster =
            coefficient_covariance(&x, n, 1, &e, SandwichKind::Cluster { groups: &groups })
                .unwrap();
        assert!((panel[0] - cluster[0]).abs() < 1e-12);
    }
}

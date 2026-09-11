//! Quantile treatment effects by inverting an estimated interventional CDF.
//!
//! `Q_a(τ) = F_a^{-1}(τ)` with `F_a(c) = P(Y(a) ≤ c) = 1 − P(Y(a) > c)`.
//! The influence function is `φ_Q = −φ_F(q) / f(q)`. Density is a finite
//! difference of `F` on the estimation grid. The path refuses when `τ` is
//! outside the estimated CDF range or the density is too small to invert.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use crate::error::EstimationError;

/// Minimum finite-difference density licensed for inversion.
pub const MIN_QUANTILE_DENSITY: f64 = 1e-3;

/// Invert a monotone CDF on a strictly increasing threshold grid.
///
/// `f_le[j] = P(Y ≤ thresholds[j])`. `phi_f[j]` is the IF of that CDF
/// coordinate (length `n`).
///
/// # Errors
///
/// Empty grid, length mismatch, `τ` outside the estimated range, or density
/// below [`MIN_QUANTILE_DENSITY`].
pub fn invert_cdf_quantile(
    thresholds: &[f64],
    f_le: &[f64],
    phi_f: &[Vec<f64>],
    tau: f64,
) -> Result<(f64, Vec<f64>, f64), EstimationError> {
    if thresholds.len() < 2 || thresholds.len() != f_le.len() || phi_f.len() != thresholds.len() {
        return Err(EstimationError::data_msg(
            "quantile inversion requires a CDF grid of at least two aligned thresholds",
        ));
    }
    if thresholds.iter().any(|v| !v.is_finite())
        || thresholds.windows(2).any(|p| p[0] >= p[1])
        || f_le.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || f_le.windows(2).any(|p| p[0] > p[1])
        || phi_f.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(EstimationError::data_msg(
            "quantile inversion requires a finite increasing grid, monotone bounded CDF, and finite influences",
        ));
    }
    let n = phi_f[0].len();
    if n == 0 || phi_f.iter().any(|p| p.len() != n) {
        return Err(EstimationError::data_msg(
            "quantile inversion requires aligned influence columns",
        ));
    }
    if !(tau > 0.0 && tau < 1.0) {
        return Err(EstimationError::unsupported("quantile level must lie in (0, 1)"));
    }
    let Some(hi) = f_le.iter().position(|&f| f >= tau) else {
        return Err(EstimationError::unsupported(
            "quantile inversion refused: τ is above the estimated CDF range",
        ));
    };
    if hi == 0 {
        return Err(EstimationError::unsupported(
            "quantile inversion refused: τ is below the estimated CDF range",
        ));
    }
    let lo = hi - 1;
    let width = thresholds[hi] - thresholds[lo];
    let rise = f_le[hi] - f_le[lo];
    if !width.is_finite() || width <= 0.0 {
        return Err(EstimationError::data_msg("quantile grid must be strictly increasing"));
    }
    let density = rise / width;
    if !density.is_finite() || density < MIN_QUANTILE_DENSITY {
        return Err(EstimationError::unsupported(
            "quantile inversion refused: estimated density at the quantile is too small",
        ));
    }
    let weight = ((tau - f_le[lo]) / rise).clamp(0.0, 1.0);
    let quantile = thresholds[lo] + weight * width;
    let mut influence = vec![0.0; n];
    for (i, value) in influence.iter_mut().enumerate() {
        let phi = (1.0 - weight) * phi_f[lo][i] + weight * phi_f[hi][i];
        *value = -phi / density;
    }
    Ok((quantile, influence, density))
}

/// QTE and its row influence for the piecewise-linear CDF on a frozen grid.
#[derive(Clone, Debug)]
pub struct QuantileContrast {
    /// Active minus control quantile.
    pub value: f64,
    /// Centered row influence, including target-weight normalization.
    pub influence: Vec<f64>,
    /// Control and active quantiles.
    pub quantiles: [f64; 2],
    /// Local finite-difference densities.
    pub densities: [f64; 2],
}

/// Invert two score-table CDFs with uncertainty for their raw local brackets.
///
/// Refuses a bracket affected by isotonic projection: raw score influence is
/// not the derivative of that projection. Inference conditions on the grid;
/// finite-grid interpolation bias and grid-selection uncertainty are excluded.
///
/// # Errors
/// Missing arm grids, unsupported tails, projection at the crossing, or invalid scores.
pub fn quantile_contrast(
    table: &crate::scores::ScoreTable,
    weights: Option<&[f64]>,
    tau: f64,
) -> Result<QuantileContrast, EstimationError> {
    let control = quantile_arm(table, weights, tau, 0)?;
    let active = quantile_arm(table, weights, tau, 1)?;
    Ok(QuantileContrast {
        value: active.value - control.value,
        influence: active.influence.iter().zip(&control.influence).map(|(a, b)| a - b).collect(),
        quantiles: [control.value, active.value],
        densities: [control.density, active.density],
    })
}

/// A requested arm's quantile and centered influence on the frozen CDF grid.
#[derive(Clone, Debug)]
pub struct QuantileArm {
    /// Estimated quantile, in outcome units.
    pub value: f64,
    /// Centered row influence including target-weight normalization.
    pub influence: Vec<f64>,
    /// Local finite-difference density.
    pub density: f64,
}

/// Invert a raw CDF while refusing unsupported or projection-altered crossings.
///
/// # Errors
/// Invalid CDF inputs, an unsupported crossing, or projection at the crossing.
pub fn invert_supported_cdf(
    thresholds: &[f64],
    raw_cdf: &[f64],
    influences: &[Vec<f64>],
    supported: &[bool],
    tau: f64,
) -> Result<QuantileArm, EstimationError> {
    if raw_cdf.len() != thresholds.len()
        || supported.len() != thresholds.len()
        || raw_cdf.iter().any(|v| !v.is_finite())
    {
        return Err(EstimationError::data_msg("quantile CDF and support must match the grid"));
    }
    let cdf: Vec<_> =
        crate::monotone_increasing(raw_cdf).into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
    if let Some(hi) = cdf.iter().position(|&v| v >= tau).filter(|&j| j > 0) {
        for j in hi - 1..=hi {
            if (raw_cdf[j] - cdf[j]).abs() > 1e-12 {
                return Err(EstimationError::unsupported(
                    "quantile uncertainty refused: isotonic projection changes the inversion bracket",
                ));
            }
            if !supported[j] {
                return Err(EstimationError::unsupported(
                    "quantile inversion refused: insufficient tail or overlap support at the bracket",
                ));
            }
        }
    }
    let (value, influence, density) = invert_cdf_quantile(thresholds, &cdf, influences, tau)?;
    Ok(QuantileArm { value, influence, density })
}

/// Invert one binary or joint-cell score-table arm, preserving its actual cell id.
///
/// Inference conditions on the grid and excludes interpolation bias and grid selection.
///
/// # Errors
/// Missing cell, unsupported overlap/tails, projection at crossing, or invalid scores.
pub fn quantile_arm(
    table: &crate::scores::ScoreTable,
    weights: Option<&[f64]>,
    tau: f64,
    arm: u32,
) -> Result<QuantileArm, EstimationError> {
    let raw = table.summarize(weights)?;
    let inference = table.inference(weights)?;
    if !inference.support.overlap_ok {
        return Err(EstimationError::unsupported(
            "quantile inversion requires supported target overlap",
        ));
    }
    let n = table.n_rows;
    let mass = weights.map_or(n as f64, |w| w.iter().sum());
    let mut indexes: Vec<_> = table
        .columns
        .iter()
        .enumerate()
        .filter_map(|(j, c)| (c.arm == arm).then_some((j, c.threshold?)))
        .collect();
    indexes.sort_by(|a, b| a.1.total_cmp(&b.1));
    let thresholds: Vec<_> = indexes.iter().map(|(_, c)| *c).collect();
    let cdf: Vec<_> = indexes.iter().map(|(j, _)| 1.0 - raw.means[*j]).collect();
    let supported: Vec<_> =
        indexes.iter().map(|(j, _)| inference.threshold_supported[*j]).collect();
    let phi: Vec<Vec<f64>> = indexes
        .iter()
        .map(|(j, _)| {
            Ok(table
                .column(*j)?
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let scale = weights.map_or(1.0, |w| n as f64 * w[i] / mass);
                    -scale * (v - raw.means[*j])
                })
                .collect())
        })
        .collect::<Result<_, EstimationError>>()?;
    invert_supported_cdf(&thresholds, &cdf, &phi, &supported, tau)
}

/// Empirical strictly increasing threshold grid from observed `Y`.
///
/// # Errors
///
/// Too few finite values or a collapsed range.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Rounded index is nonnegative and bounded by len-1.
pub fn empirical_threshold_grid(y: &[f64], points: usize) -> Result<Vec<f64>, EstimationError> {
    let mut v: Vec<f64> = y.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 16 || points < 4 {
        return Err(EstimationError::data_msg(
            "quantile estimation requires at least 16 finite outcomes and 4 grid points",
        ));
    }
    v.sort_by(f64::total_cmp);
    let mut out = Vec::with_capacity(points);
    for i in 1..=points {
        let p = i as f64 / (points + 1) as f64;
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        out.push(v[idx.min(v.len() - 1)]);
    }
    out.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
    if out.len() < 4 {
        return Err(EstimationError::data_msg(
            "quantile estimation grid collapsed; outcome has too little spread",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_linear_cdf() {
        let thresholds = [0.0, 1.0, 2.0, 3.0];
        let f_le = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        let n = 8usize;
        let phi: Vec<Vec<f64>> =
            (0..4).map(|_| (0..n).map(|i| (i as f64) * 0.01).collect()).collect();
        let (q, infl, density) = invert_cdf_quantile(&thresholds, &f_le, &phi, 0.5).unwrap();
        assert!((q - 1.5).abs() < 1e-12);
        assert!((density - (1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(infl.len(), n);
    }

    #[test]
    fn quantile_influence_matches_perturbed_cdf_derivative() {
        let thresholds = [0.0, 1.0, 3.0];
        let cdf = [0.1, 0.4, 0.9];
        let phi = vec![vec![0.02, -0.02], vec![-0.03, 0.03], vec![0.08, -0.08]];
        let (_, influence, _) = invert_cdf_quantile(&thresholds, &cdf, &phi, 0.6).unwrap();
        let epsilon = 1e-6;
        for (row, expected) in influence.iter().enumerate() {
            let shifted = |sign: f64| {
                let values: Vec<_> = cdf
                    .iter()
                    .zip(&phi)
                    .map(|(f, column)| f + sign * epsilon * column[row])
                    .collect();
                invert_cdf_quantile(&thresholds, &values, &phi, 0.6).unwrap().0
            };
            let derivative = (shifted(1.0) - shifted(-1.0)) / (2.0 * epsilon);
            assert!((derivative - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn refuses_tail() {
        let thresholds = [0.0, 1.0];
        let f_le = [0.2, 0.4];
        let phi = vec![vec![0.0; 4], vec![0.0; 4]];
        let err = invert_cdf_quantile(&thresholds, &f_le, &phi, 0.9).unwrap_err();
        assert!(err.to_string().contains("above the estimated CDF"));
    }
}

#[cfg(test)]
mod review_tests {
    use super::*;
    #[test]
    fn rejects_malformed_cdf_inputs() {
        let phi = vec![vec![0.1; 8]; 3];
        for (x, f) in [
            ([0.0, 1.0, f64::NAN], [0.1, 0.5, 0.9]),
            ([0.0, 1.0, 0.5], [0.1, 0.5, 0.9]),
            ([0.0, 1.0, 2.0], [0.1, 0.7, 0.4]),
            ([0.0, 1.0, 2.0], [0.1, 0.5, 1.1]),
        ] {
            assert!(invert_cdf_quantile(&x, &f, &phi, 0.3).is_err());
        }
        let mut bad_phi = phi;
        bad_phi[2][0] = f64::NAN;
        assert!(invert_cdf_quantile(&[0.0, 1.0, 2.0], &[0.1, 0.5, 0.9], &bad_phi, 0.3).is_err());
    }

    #[test]
    fn refuses_projection_at_quantile_crossing() {
        let n = 200;
        let mut columns = Vec::new();
        let mut scores = Vec::new();
        for (threshold, cdf) in [(0.0, 0.2), (1.0, 0.7), (2.0, 0.5), (3.0, 0.9)] {
            for arm in 0..2 {
                columns.push(crate::scores::ScoreColumn { arm, threshold: Some(threshold) });
                scores.extend(vec![1.0 - cdf; n]);
            }
        }
        let table = crate::scores::ScoreTable {
            observed_arm: (0..n).map(|i| u32::try_from(i % 2).unwrap()).collect(),
            propensities: vec![0.5; scores.len()].into(),
            observed_outcome: (0..n).map(|i| if i % 4 < 2 { -2.0 } else { 5.0 }).collect(),
            n_rows: n,
            row_index: (0..n).map(|i| u32::try_from(i).unwrap()).collect(),
            fold_ids: (0..n).map(|i| u32::try_from(i % 5).unwrap()).collect(),
            n_folds: 5,
            scores: scores.into(),
            columns: columns.into(),
            adjustment_set: [].into(),
            nuisance_provenance: "test".into(),
            treatment: antecedent_core::VariableId::from_raw(0),
            intervened: [].into(),
        };
        let error = quantile_contrast(&table, None, 0.55).unwrap_err();
        assert!(error.to_string().contains("projection changes"), "{error}");
    }
}

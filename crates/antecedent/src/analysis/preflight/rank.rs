//! Shared rank and column-comparison primitives for preflight and rank-drop plans.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{
    COEFFICIENT_FLOOR, CausalError, ColumnWeight, DependentColumn, ExecutionContext,
    STAGE_PREFLIGHT,
};

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn l2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// A unit column and its norm as scale times factor. The ordinary path is
/// unchanged; the scaled path avoids squaring very small or large entries.
pub(super) fn normalized_column(col: &[f64]) -> Option<(Vec<f64>, (f64, f64))> {
    let squared = dot(col, col);
    if squared.is_normal() {
        let norm = squared.sqrt();
        return Some((col.iter().map(|v| v / norm).collect(), (norm, 1.0)));
    }
    let scale = col.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let mut unit: Vec<f64> = col.iter().map(|v| v / scale).collect();
    let factor = l2(&unit);
    for value in &mut unit {
        *value /= factor;
    }
    Some((unit, (scale, factor)))
}

/// Result of scanning columns in priority order with twice-applied Gram-Schmidt.
pub(super) struct RankScan {
    /// Orthonormal basis of the kept columns.
    pub(super) basis: Vec<Vec<f64>>,
    /// Per scanned column.
    pub(super) columns: Vec<ScannedColumn>,
}

pub(super) struct ScannedColumn {
    pub(super) kept: bool,
    /// Residual norm over own norm.
    pub(super) ratio: f64,
    /// `(scanned column index, coefficient in original units)` of the exact relation, for a
    /// dependent column.
    pub(super) explained_by: Vec<(usize, f64)>,
}

/// Scan `cols` in order: a column is kept when its residual after projecting out the earlier
/// kept columns exceeds `tol` times its own norm. Dependent columns record the relation that
/// explains them, obtained by back-substitution in the triangular factor of the kept columns.
#[allow(
    clippy::float_cmp,
    reason = "exact equality preserves the declared boundary and identity semantics"
)]
pub(super) fn scan_rank(
    cols: &[&[f64]],
    tol: f64,
    ctx: &ExecutionContext,
) -> Result<RankScan, CausalError> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    // Upper-triangular factor, one column per kept column: `r_kept[k][l] = <q_l, a_k>`.
    let mut r_kept: Vec<Vec<f64>> = Vec::new();
    let mut kept_index: Vec<usize> = Vec::new();
    let mut norms = Vec::with_capacity(cols.len());
    let mut scanned = Vec::with_capacity(cols.len());
    for (j, col) in cols.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_PREFLIGHT });
        }
        let Some((mut w, norm)) = normalized_column(col) else {
            norms.push((0.0, 1.0));
            scanned.push(ScannedColumn { kept: false, ratio: 0.0, explained_by: Vec::new() });
            continue;
        };
        norms.push(norm);
        let mut coef = vec![0.0; basis.len()];
        for _ in 0..2 {
            for (k, q) in basis.iter().enumerate() {
                let c = dot(q, &w);
                for (wi, qi) in w.iter_mut().zip(q) {
                    *wi -= c * qi;
                }
                coef[k] += c;
            }
        }
        let ratio = l2(&w);
        if ratio > tol {
            basis.push(w.iter().map(|v| v / ratio).collect());
            coef.push(ratio);
            r_kept.push(coef);
            kept_index.push(j);
            scanned.push(ScannedColumn { kept: true, ratio, explained_by: Vec::new() });
        } else {
            // Solve R c = coef for the coefficients on the kept (unit-norm) columns.
            let m = kept_index.len();
            let mut c = vec![0.0; m];
            for k in (0..m).rev() {
                let tail: f64 = ((k + 1)..m).map(|i| r_kept[i][k] * c[i]).sum();
                c[k] = (coef[k] - tail) / r_kept[k][k];
            }
            let explained_by = c
                .iter()
                .zip(&kept_index)
                .filter_map(|(&ck, &i)| {
                    let (scale, factor) = norm;
                    let (prior_scale, prior_factor) = norms[i];
                    // Preserve ordinary arithmetic; scaled cases retain the original
                    // units without first forming an unrepresentable column norm.
                    let original = if factor.to_bits() == 1.0_f64.to_bits()
                        && prior_factor.to_bits() == 1.0_f64.to_bits()
                    {
                        ck * scale / prior_scale
                    } else {
                        ((ck * factor / prior_factor) * scale) / prior_scale
                    };
                    (original.abs() > COEFFICIENT_FLOOR).then_some((i, original))
                })
                .collect();
            scanned.push(ScannedColumn { kept: false, ratio, explained_by });
        }
    }
    Ok(RankScan { basis, columns: scanned })
}

/// `1 - R²` of `values` regressed on the scanned basis (which spans the intercept first),
/// as the square root: the unexplained share of the centered norm. `None` for a constant.
pub(super) fn unexplained_ratio(basis: &[Vec<f64>], values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = antecedent_kernels::mean(values);
    let centered: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>().sqrt();
    if !(centered.is_finite() && centered > 0.0) {
        return None;
    }
    let mut w = values.to_vec();
    for _ in 0..2 {
        for q in basis {
            let c = dot(q, &w);
            for (wi, qi) in w.iter_mut().zip(q) {
                *wi -= c * qi;
            }
        }
    }
    Some(l2(&w) / centered)
}

/// Squared Pearson correlation; `None` when either side is constant.
pub(super) fn r_squared(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let ma = antecedent_kernels::mean(a);
    let mb = antecedent_kernels::mean(b);
    let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
        sab += (x - ma) * (y - mb);
    }
    if saa > 0.0 && sbb > 0.0 { Some(sab * sab / (saa * sbb)) } else { None }
}

/// Normalized bit pattern of a value (`-0.0` equals `0.0`).
fn bits(value: f64) -> u64 {
    super::super::recalc_common::canonical_bits(value)
}

pub(super) fn duplicate_groups(columns: &[Vec<f64>]) -> Vec<Vec<usize>> {
    // Bucket columns by an in-process hash, then verify bit equality only against the
    // representative of each group in the bucket: linear in the columns for distinct data.
    let mut buckets: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (j, col) in columns.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        for &v in col {
            bits(v).hash(&mut hasher);
        }
        let bucket = buckets.entry(hasher.finish()).or_default();
        let found = bucket.iter().copied().find(|&group| {
            let first = groups[group][0];
            columns[first].len() == col.len()
                && columns[first].iter().zip(col).all(|(a, b)| bits(*a) == bits(*b))
        });
        if let Some(group) = found {
            groups[group].push(j);
        } else {
            bucket.push(groups.len());
            groups.push(vec![j]);
        }
    }
    groups.retain(|group| group.len() > 1);
    groups
}

pub(super) fn tolerance(rows: usize, design_columns: usize) -> f64 {
    rows.max(design_columns) as f64 * f64::EPSILON
}

/// Names of the design columns for a scan: the intercept, then `adjustment` in scan order.
pub(super) fn design_names(adjustment: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(adjustment.len() + 1);
    names.push("(intercept)".to_string());
    names.extend(adjustment.iter().cloned());
    names
}

pub(super) fn dependent_columns(
    scan: &RankScan,
    names: &[String],
    skip: usize,
) -> Vec<DependentColumn> {
    scan.columns
        .iter()
        .enumerate()
        .skip(skip)
        .filter(|(_, c)| !c.kept)
        .map(|(j, c)| DependentColumn {
            column: names[j].clone(),
            residual_ratio: c.ratio,
            explained_by: c
                .explained_by
                .iter()
                .map(|&(i, coefficient)| ColumnWeight { column: names[i].clone(), coefficient })
                .collect(),
        })
        .collect()
}

//! Deterministic basis expansions of a parent row.
//!
//! A [`ParentBasis`] turns one row of parent values into a fixed vector of
//! model columns. It is the shared design behind the heterogeneity-capable
//! mechanism families: [`MechanismSlot::LinearBasis`](crate::MechanismSlot::LinearBasis)
//! is linear-Gaussian in these columns and
//! [`MechanismSlot::DiscreteBasis`](crate::MechanismSlot::DiscreteBasis) puts
//! multinomial logits on them.
//!
//! Three properties are load-bearing and are why the expansion is a value, not
//! a procedure:
//!
//! * **Deterministic.** Every column is a closed-form function of the row's own
//!   parent values and the constants stored in the basis (centers, scales,
//!   knots). No RNG, no seed, no data order: refitting on the same table gives
//!   the same columns, and evaluating at an unseen covariate cell needs nothing
//!   but the serialized basis.
//! * **Additive in the disturbance.** The expansion touches only the
//!   conditional mean. `y = f(φ(pa)) + ε` stays exactly invertible, so
//!   counterfactual abduction recovers `ε = y − f(φ(pa))` and
//!   `noise_inference = Invertible` is preserved.
//! * **Scale-free.** Parents are standardized by the stored centers and scales
//!   before any column is formed, so the fitted mechanism is unchanged (up to
//!   the affine change of variables) when a parent column is rescaled. A raw
//!   year column and its z-score give the same fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Parent indices are dense graph ids (`u32` by construction) and a term index is
// bounded by rows / `BASIS_MIN_ROWS_PER_COLUMN`, so `usize → u32` cannot truncate.
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use crate::error::ModelError;

/// One column of a [`ParentBasis`] expansion.
///
/// Terms are evaluated in order into a scratch buffer, so [`Self::Product`]
/// refers to two columns by index and both must appear earlier in the term
/// list. That keeps arbitrary products expressible without a nested term tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BasisTerm {
    /// `z_j^degree`, where `z_j` is the standardized parent `j`. Degree 1 is
    /// the plain main effect.
    Power {
        /// Parent index in compiled gather order.
        parent: u32,
        /// Exponent (≥ 1).
        degree: u8,
    },
    /// Truncated cubic power `((z_j − t)_+)^3` at knot `t = knots(parent)[knot]`.
    Truncated {
        /// Parent index in compiled gather order.
        parent: u32,
        /// Index into this parent's knot vector.
        knot: u32,
    },
    /// Product of two earlier columns, by index into the term list.
    Product {
        /// Index of the left factor (`< ` this term's own index).
        left: u32,
        /// Index of the right factor (`< ` this term's own index).
        right: u32,
    },
}

/// Interior knots used by the spline family, per parent.
///
/// Three interior knots at the 0.25 / 0.50 / 0.75 quantiles of the
/// standardized parent column. Fixed quantiles rather than a searched knot
/// count: a searched count would make the fit depend on a tolerance and would
/// need its own selection layer inside a family that is already competing on a
/// validation score.
pub const SPLINE_KNOT_QUANTILES: [f64; 3] = [0.25, 0.50, 0.75];

/// Distinct values a parent needs before it earns spline columns.
///
/// Below this a smooth term is not identified apart from the main effect (a
/// binary parent's "curve" is two points), and the extra columns are pure
/// variance.
pub const SPLINE_MIN_DISTINCT: usize = 5;

/// A deterministic expansion of a parent row into model columns.
#[derive(Clone, Debug)]
pub struct ParentBasis {
    n_parents: usize,
    centers: Arc<[f64]>,
    scales: Arc<[f64]>,
    knots: Arc<[Arc<[f64]>]>,
    terms: Arc<[BasisTerm]>,
}

impl ParentBasis {
    /// Build from parts, checking the internal invariants a fitted or
    /// deserialized basis must satisfy.
    ///
    /// # Errors
    ///
    /// Length mismatches, a non-finite or non-positive scale, a `Product` whose
    /// factors are not strictly earlier terms, a knot index out of range, or a
    /// zero exponent.
    pub fn new(
        n_parents: usize,
        centers: Arc<[f64]>,
        scales: Arc<[f64]>,
        knots: Arc<[Arc<[f64]>]>,
        terms: Arc<[BasisTerm]>,
    ) -> Result<Self, ModelError> {
        if centers.len() != n_parents || scales.len() != n_parents || knots.len() != n_parents {
            return Err(ModelError::Shape {
                message: "parent basis centers/scales/knots must have one entry per parent".into(),
            });
        }
        if centers.iter().any(|c| !c.is_finite()) {
            return Err(ModelError::Shape {
                message: "parent basis centers must be finite".into(),
            });
        }
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(ModelError::Shape {
                message: "parent basis scales must be finite and positive".into(),
            });
        }
        for (index, term) in terms.iter().enumerate() {
            match *term {
                BasisTerm::Power { parent, degree } => {
                    if parent as usize >= n_parents || degree == 0 {
                        return Err(ModelError::Shape {
                            message: "parent basis power term out of range".into(),
                        });
                    }
                }
                BasisTerm::Truncated { parent, knot } => {
                    let p = parent as usize;
                    if p >= n_parents || knot as usize >= knots[p].len() {
                        return Err(ModelError::Shape {
                            message: "parent basis truncated term out of range".into(),
                        });
                    }
                }
                BasisTerm::Product { left, right } => {
                    if left as usize >= index || right as usize >= index {
                        return Err(ModelError::Shape {
                            message: "parent basis product must reference earlier terms".into(),
                        });
                    }
                }
            }
        }
        Ok(Self { n_parents, centers, scales, knots, terms })
    }

    /// Parent arity this basis was built for.
    #[must_use]
    pub fn n_parents(&self) -> usize {
        self.n_parents
    }

    /// Number of expanded columns (excluding the intercept).
    #[must_use]
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Whether the expansion is empty (only an intercept remains).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Expanded columns in order.
    #[must_use]
    pub fn terms(&self) -> &[BasisTerm] {
        &self.terms
    }

    /// Per-parent centers subtracted before expansion.
    #[must_use]
    pub fn centers(&self) -> &[f64] {
        &self.centers
    }

    /// Per-parent scales divided out before expansion.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Per-parent interior knot vectors, on the standardized scale.
    #[must_use]
    pub fn knots(&self) -> &[Arc<[f64]>] {
        &self.knots
    }

    /// Whether any column multiplies two different parents.
    ///
    /// This is the property that decides whether the family can represent
    /// effect modification at all: without a cross-parent product, the
    /// conditional mean is additively separable in the parents and a hard
    /// intervention on one parent shifts the node by an amount that does not
    /// depend on the unit's other parents.
    #[must_use]
    pub fn has_cross_parent_product(&self) -> bool {
        self.terms.iter().enumerate().any(|(index, term)| {
            matches!(*term, BasisTerm::Product { .. }) && self.parents_of(index).len() > 1
        })
    }

    /// The set of parents a column depends on (sorted, deduplicated).
    fn parents_of(&self, index: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = vec![index];
        while let Some(i) = stack.pop() {
            match self.terms[i] {
                BasisTerm::Power { parent, .. } | BasisTerm::Truncated { parent, .. } => {
                    out.push(parent);
                }
                BasisTerm::Product { left, right } => {
                    stack.push(left as usize);
                    stack.push(right as usize);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Standardize one parent value with the stored constants.
    #[must_use]
    pub fn standardize(&self, parent: usize, value: f64) -> f64 {
        (value - self.centers[parent]) / self.scales[parent]
    }

    /// Expand one already-standardized parent row into `out` (length `n_terms`).
    ///
    /// # Errors
    ///
    /// `out` shorter than the term list, or `standardized` shorter than the
    /// parent arity.
    pub fn expand_standardized(
        &self,
        standardized: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if standardized.len() < self.n_parents || out.len() < self.terms.len() {
            return Err(ModelError::Shape {
                message: "parent basis expansion buffers too short".into(),
            });
        }
        for (index, term) in self.terms.iter().enumerate() {
            out[index] = match *term {
                BasisTerm::Power { parent, degree } => {
                    standardized[parent as usize].powi(i32::from(degree))
                }
                BasisTerm::Truncated { parent, knot } => {
                    let d =
                        standardized[parent as usize] - self.knots[parent as usize][knot as usize];
                    if d > 0.0 { d * d * d } else { 0.0 }
                }
                BasisTerm::Product { left, right } => out[left as usize] * out[right as usize],
            };
        }
        Ok(())
    }

    /// Expand one raw parent row (standardizing first) into `out`.
    ///
    /// # Errors
    ///
    /// Buffer lengths, as [`Self::expand_standardized`].
    pub fn expand_row(&self, raw: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        if raw.len() < self.n_parents {
            return Err(ModelError::Shape {
                message: "parent basis expansion buffers too short".into(),
            });
        }
        let mut standardized = [0.0_f64; 16];
        if self.n_parents <= standardized.len() {
            for p in 0..self.n_parents {
                standardized[p] = self.standardize(p, raw[p]);
            }
            return self.expand_standardized(&standardized[..self.n_parents], out);
        }
        let owned: Vec<f64> = (0..self.n_parents).map(|p| self.standardize(p, raw[p])).collect();
        self.expand_standardized(&owned, out)
    }

    /// Two-way interaction basis: every main effect plus every cross-parent
    /// product `z_j · z_k` (`j < k`).
    ///
    /// This is the minimum expansion whose unit effects vary: with the
    /// treatment among the parents, the `treatment × parent` columns are
    /// exactly the terms that let the contrast depend on the unit's covariates.
    ///
    /// # Errors
    ///
    /// Invariant failures from [`Self::new`].
    #[allow(
        clippy::cast_possible_truncation,
        reason = "basis term indices are u32 in the wire format, and the parent count is far below 2^32"
    )]
    pub fn interactions(
        n_parents: usize,
        centers: Arc<[f64]>,
        scales: Arc<[f64]>,
    ) -> Result<Self, ModelError> {
        let mut terms = Vec::new();
        for p in 0..n_parents {
            terms.push(BasisTerm::Power { parent: p as u32, degree: 1 });
        }
        for j in 0..n_parents {
            for k in (j + 1)..n_parents {
                terms.push(BasisTerm::Product { left: j as u32, right: k as u32 });
            }
        }
        Self::new(
            n_parents,
            centers,
            scales,
            Arc::from(vec![Arc::from([]) as Arc<[f64]>; n_parents]),
            Arc::from(terms),
        )
    }

    /// Additive cubic-spline basis with cross-parent products of every smooth
    /// column.
    ///
    /// Per parent with knots: `z, z², z³` and one truncated cubic per knot —
    /// the truncated power basis of a cubic regression spline. For each ordered
    /// pair of distinct parents `(j, l)` the main effect `z_j` multiplies every
    /// column of `l`, which is what carries a treatment effect that bends with
    /// a covariate. Parents below [`SPLINE_MIN_DISTINCT`] distinct values
    /// contribute their main effect only.
    ///
    /// The truncated power basis is chosen over B-splines because each column
    /// is a closed form of the parent value and the stored knot: a
    /// counterfactual evaluated at an unseen covariate cell needs only the
    /// serialized basis, with no recursion and no retained training rows. Its
    /// known weakness — conditioning — is handled by standardizing the parents
    /// first and keeping the knot count at three.
    ///
    /// # Errors
    ///
    /// Invariant failures from [`Self::new`].
    #[allow(
        clippy::cast_possible_truncation,
        reason = "basis term, parent and knot indices are u32 in the wire format, and their counts are far below 2^32"
    )]
    pub fn spline_interactions(
        n_parents: usize,
        centers: Arc<[f64]>,
        scales: Arc<[f64]>,
        knots: Arc<[Arc<[f64]>]>,
    ) -> Result<Self, ModelError> {
        if knots.len() != n_parents {
            return Err(ModelError::Shape {
                message: "parent basis centers/scales/knots must have one entry per parent".into(),
            });
        }
        let mut terms = Vec::new();
        // Columns of each parent, by term index, so the products below can name them.
        let mut columns: Vec<Vec<u32>> = vec![Vec::new(); n_parents];
        for p in 0..n_parents {
            columns[p].push(terms.len() as u32);
            terms.push(BasisTerm::Power { parent: p as u32, degree: 1 });
            if knots[p].is_empty() {
                continue;
            }
            for degree in [2_u8, 3_u8] {
                columns[p].push(terms.len() as u32);
                terms.push(BasisTerm::Power { parent: p as u32, degree });
            }
            for k in 0..knots[p].len() {
                columns[p].push(terms.len() as u32);
                terms.push(BasisTerm::Truncated { parent: p as u32, knot: k as u32 });
            }
        }
        for j in 0..n_parents {
            let left = columns[j][0];
            for (l, smooth) in columns.iter().enumerate() {
                if l == j {
                    continue;
                }
                for &right in smooth {
                    // `z_j · z_l` is symmetric; keep one copy.
                    if matches!(terms[right as usize], BasisTerm::Power { degree: 1, .. }) && l < j
                    {
                        continue;
                    }
                    terms.push(BasisTerm::Product { left, right });
                }
            }
        }
        Self::new(n_parents, centers, scales, knots, Arc::from(terms))
    }
}

/// Column centers and scales for standardizing a parent design.
///
/// The scale is the population standard deviation, floored away from zero so a
/// constant column standardizes to zeros rather than to `NaN`.
#[must_use]
pub fn column_moments(columns: &[&[f64]], n_rows: usize) -> (Vec<f64>, Vec<f64>) {
    let mut centers = Vec::with_capacity(columns.len());
    let mut scales = Vec::with_capacity(columns.len());
    #[allow(clippy::cast_precision_loss)]
    let n = n_rows.max(1) as f64;
    for col in columns {
        let mean = col[..n_rows].iter().sum::<f64>() / n;
        let var = col[..n_rows].iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let sd = var.sqrt();
        centers.push(if mean.is_finite() { mean } else { 0.0 });
        scales.push(if sd.is_finite() && sd > 1e-12 { sd } else { 1.0 });
    }
    (centers, scales)
}

/// Interior knots for each parent, on the standardized scale.
///
/// Empty for a parent with fewer than [`SPLINE_MIN_DISTINCT`] distinct values.
#[must_use]
pub fn spline_knots(
    columns: &[&[f64]],
    n_rows: usize,
    centers: &[f64],
    scales: &[f64],
) -> Vec<Arc<[f64]>> {
    let mut out = Vec::with_capacity(columns.len());
    for (p, col) in columns.iter().enumerate() {
        let mut values: Vec<f64> = col[..n_rows]
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| (v - centers[p]) / scales[p])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut distinct = values.clone();
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        if distinct.len() < SPLINE_MIN_DISTINCT {
            out.push(Arc::from([]) as Arc<[f64]>);
            continue;
        }
        let mut knots: Vec<f64> = SPLINE_KNOT_QUANTILES
            .iter()
            .map(|&q| {
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation,
                    reason = "q is a quantile in [0, 1], so the rounded position is non-negative and at most values.len() - 1"
                )]
                let idx = ((values.len() - 1) as f64 * q).round() as usize;
                values[idx]
            })
            .collect();
        knots.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        out.push(Arc::from(knots));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis_two_parents() -> ParentBasis {
        ParentBasis::interactions(2, Arc::from([0.0, 0.0]), Arc::from([1.0, 1.0])).unwrap()
    }

    #[test]
    fn interaction_basis_expands_main_effects_then_products() {
        let basis = basis_two_parents();
        assert_eq!(basis.n_terms(), 3);
        assert!(basis.has_cross_parent_product());
        let mut out = vec![0.0; 3];
        basis.expand_row(&[2.0, 3.0], &mut out).unwrap();
        assert_eq!(out, vec![2.0, 3.0, 6.0]);
    }

    #[test]
    fn single_parent_interaction_basis_has_no_cross_product() {
        let basis = ParentBasis::interactions(1, Arc::from([0.0]), Arc::from([1.0])).unwrap();
        assert_eq!(basis.n_terms(), 1);
        assert!(!basis.has_cross_parent_product());
    }

    #[test]
    fn standardization_makes_the_expansion_scale_free() {
        // The same underlying row on a rescaled parent expands identically.
        let raw =
            ParentBasis::interactions(2, Arc::from([2020.0, 0.0]), Arc::from([3.0, 1.0])).unwrap();
        let std = basis_two_parents();
        let mut a = vec![0.0; 3];
        let mut b = vec![0.0; 3];
        raw.expand_row(&[2023.0, 1.5], &mut a).unwrap();
        std.expand_row(&[1.0, 1.5], &mut b).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn spline_knots_skip_low_cardinality_parents() {
        let binary: Vec<f64> = (0..20).map(|i| f64::from(i % 2)).collect();
        let smooth: Vec<f64> = (0..20).map(f64::from).collect();
        let cols: Vec<&[f64]> = vec![&binary, &smooth];
        let (centers, scales) = column_moments(&cols, 20);
        let knots = spline_knots(&cols, 20, &centers, &scales);
        assert!(knots[0].is_empty());
        assert_eq!(knots[1].len(), 3);
    }

    #[test]
    fn spline_basis_is_a_closed_form_of_the_row_and_the_stored_knots() {
        let knots: Arc<[Arc<[f64]>]> =
            Arc::from(vec![Arc::from([]) as Arc<[f64]>, Arc::from([-0.5, 0.0, 0.5])]);
        let basis = ParentBasis::spline_interactions(
            2,
            Arc::from([0.0, 0.0]),
            Arc::from([1.0, 1.0]),
            knots,
        )
        .unwrap();
        assert!(basis.has_cross_parent_product());
        let mut out = vec![0.0; basis.n_terms()];
        basis.expand_row(&[1.0, 0.25], &mut out).unwrap();
        // z0 main, then z1 main / z1² / z1³ / three truncated cubics.
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 0.25).abs() < 1e-12);
        assert!((out[2] - 0.0625).abs() < 1e-12);
        let expected = 0.75_f64.powi(3);
        assert!(out.iter().any(|v| (v - expected).abs() < 1e-12), "{out:?}");
    }
}

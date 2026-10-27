//! Incremental sufficient statistics.
//!
//! Linear OLS summaries merge in either order. A row or a window summary can be
//! appended. A weight or mask change clears the summary instead of patching it.
//! Inserting a column, changing a row weight, or evicting a summarized row is
//! refused; the next attempt is to recompute from the retained rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names, reason = "xtx / xty")]

use std::collections::BTreeSet;

use antecedent_core::CanonicalWriter;
use antecedent_kernels::numeric::CompensatedSum;
use antecedent_kernels::{chol_solve_into, cholesky_spd_into};
use antecedent_stats::accumulate_xtx_xty_row;

use crate::error::StateError;
use crate::retention::RetentionPolicy;

/// Absolute term of the merge-versus-recompute tolerance.
///
/// [`merge_within_tolerance`] accepts
/// `|a - b| <= MERGE_ABS_TOL + MERGE_REL_TOL * max(|a|, |b|)`.
pub const MERGE_ABS_TOL: f64 = 1e-8;

/// Relative term of the merge-versus-recompute tolerance. See [`MERGE_ABS_TOL`].
pub const MERGE_REL_TOL: f64 = 1e-9;

/// Next attempt named by every refused summary update.
const RECOMPUTE_FROM_RETAINED_ROWS: &str = "recompute from the retained rows";

/// Whether `left` and `right` agree closely enough to treat a merge as a recompute.
///
/// The bound is [`MERGE_ABS_TOL`] + [`MERGE_REL_TOL`] times the larger magnitude.
/// Non-finite values do not agree.
#[must_use = "the merge contract is this comparison"]
pub const fn merge_within_tolerance(left: f64, right: f64) -> bool {
    let difference = (left - right).abs();
    if !difference.is_finite() {
        return false;
    }
    let left_abs = left.abs();
    let right_abs = right.abs();
    let scale = if left_abs > right_abs { left_abs } else { right_abs };
    difference <= MERGE_ABS_TOL + MERGE_REL_TOL * scale
}

/// One attempted change to a [`LinearOlsSuffStats`] summary.
#[derive(Clone, Copy, Debug)]
pub enum LinearOlsDelta<'a> {
    /// Append one finite unit-weight design row. This is applied.
    AppendRow {
        /// Design row, length `ncols`.
        row: &'a [f64],
        /// Response.
        y: f64,
    },
    /// Merge a summary of a contiguous window of the same columns. This is applied.
    AppendWindow {
        /// Window summary. Column count must match.
        window: &'a LinearOlsSuffStats,
    },
    /// Drop one summarized row.
    ///
    /// Refused. There is no exact downdate here: subtracting the row from the Gram
    /// and reversing the centered scatter are not claimed to match recomputation of
    /// the retained rows. The summary is left unchanged.
    EvictRow {
        /// Design row that would have been removed.
        row: &'a [f64],
        /// Response that would have been removed.
        y: f64,
    },
    /// A design column was inserted. Refused; the summary is left unchanged.
    ColumnInserted,
    /// The weight of a summarized row changed. The summary is invalidated.
    RowWeightChanged,
    /// Observation weights changed. The summary is invalidated.
    WeightsChanged,
    /// The row or column mask changed. The summary is invalidated.
    MaskChanged,
}

/// Linear OLS sufficient statistics (`XᵀX`, `Xᵀy`, `n`).
///
/// Summaries of the same columns merge in either order ([`Self::merge`]). Append a
/// row or a window with [`Self::apply_delta`]. A weight or mask change invalidates
/// the summary. An inserted column or an evicted row is refused.
///
/// The joint scatter of `[x | y]` is the source of the published moments. Those
/// moments are not writable from outside this type. [`Self::solve_beta`],
/// [`Self::merge`], [`Self::append_row`], and [`Self::residual_variance`] refuse a
/// summary whose published moments no longer agree with that scatter.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearOlsSuffStats {
    /// Number of columns `p`.
    ncols: usize,
    /// Row-major `p×p` Gram.
    xtx: Vec<f64>,
    /// `Xᵀy` length `p`.
    xty: Vec<f64>,
    /// Sample count.
    n: u64,
    /// Sum of squared responses (for residual variance).
    yty: f64,
    /// Welford scatter of `[x | y]` for a translation-stable residual scale.
    joint: StreamingCovariance,
    /// Retention declaration.
    pub(crate) retention: RetentionPolicy,
    /// Set when a weight or mask change has cleared this summary.
    invalidated: bool,
}

impl LinearOlsSuffStats {
    /// Empty stats for `ncols` predictors.
    ///
    /// # Errors
    ///
    /// `ncols²` or `(ncols + 1)²` does not fit in `usize`.
    pub fn new(ncols: usize) -> Result<Self, StateError> {
        let gram = square_len(ncols, "ols gram dimension overflow")?;
        let joint_dim = ncols
            .checked_add(1)
            .ok_or_else(|| StateError::Shape("ols joint dimension overflow".into()))?;
        Ok(Self {
            ncols,
            xtx: vec![0.0; gram],
            xty: vec![0.0; ncols],
            n: 0,
            yty: 0.0,
            joint: StreamingCovariance::new(joint_dim)?,
            retention: RetentionPolicy::SufficientStatisticsOnly,
            invalidated: false,
        })
    }

    /// Number of columns `p`.
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.ncols
    }

    /// Sample count.
    #[must_use]
    pub const fn n(&self) -> u64 {
        self.n
    }

    /// Row-major `p×p` Gram published from the joint scatter.
    #[must_use]
    pub fn xtx(&self) -> &[f64] {
        &self.xtx
    }

    /// `Xᵀy`, length `p`, published from the joint scatter.
    #[must_use]
    pub fn xty(&self) -> &[f64] {
        &self.xty
    }

    /// Sum of squared responses, published from the joint scatter.
    #[must_use]
    pub const fn yty(&self) -> f64 {
        self.yty
    }

    /// Whether a weight or mask change has cleared this summary.
    #[must_use = "an invalidated summary must be recomputed from the retained rows"]
    pub const fn is_invalidated(&self) -> bool {
        self.invalidated
    }

    /// Append one design row and response.
    ///
    /// The row is validated before anything is touched: a rejected row leaves the statistics
    /// exactly as they were, so a single bad observation cannot poison the accumulated
    /// Gram, the Welford scatter and the count out of step with each other.
    ///
    /// # Errors
    ///
    /// The summary was invalidated (recompute from the retained rows), the row length
    /// does not match, or an entry in the row or the response is non-finite.
    pub fn append_row(&mut self, row: &[f64], y: f64) -> Result<(), StateError> {
        if self.invalidated {
            return Err(unsupported("summary invalidated"));
        }
        self.validate_row(row, y)?;
        if !self.published_matches_scatter() {
            return Err(unsupported("summary is out of step with its scatter"));
        }
        let mut joint_row = Vec::with_capacity(self.ncols.saturating_add(1));
        self.append_validated(row, y, &mut joint_row)
    }

    /// Shape and finiteness of one row; touches nothing.
    fn validate_row(&self, row: &[f64], y: f64) -> Result<(), StateError> {
        if row.len() != self.ncols {
            return Err(StateError::Shape(format!(
                "row len {} != ncols {}",
                row.len(),
                self.ncols
            )));
        }
        if !y.is_finite() || row.iter().any(|v| !v.is_finite()) {
            return Err(StateError::Numerical("design row and response must be finite".into()));
        }
        Ok(())
    }

    /// Append a row already validated by [`Self::validate_row`] into a summary already
    /// verified against its scatter. `joint_row` is reusable scratch.
    fn append_validated(
        &mut self,
        row: &[f64],
        y: f64,
        joint_row: &mut Vec<f64>,
    ) -> Result<(), StateError> {
        joint_row.clear();
        joint_row.extend_from_slice(row);
        joint_row.push(y);
        // Fallible step first (count overflow); everything after it cannot fail.
        self.joint.append(joint_row)?;
        accumulate_xtx_xty_row(row, y, &mut self.xtx, &mut self.xty);
        self.yty += y * y;
        self.n = self.n.saturating_add(1);
        Ok(())
    }

    /// Append a batch of rows (row-major `n×p`) and responses.
    ///
    /// The batch is one unit: every row is validated and the summary is verified against its
    /// scatter once, before anything is appended, so a bad row leaves the summary untouched.
    /// Nothing inside the batch can put the published moments out of step with the scatter,
    /// so that one verification covers every row.
    ///
    /// # Errors
    ///
    /// Shape mismatch, a non-finite value, an invalidated or out-of-step summary, or a count
    /// that would overflow.
    pub fn append_batch(&mut self, rows_rowmajor: &[f64], y: &[f64]) -> Result<(), StateError> {
        if self.ncols == 0 {
            return Err(StateError::Shape("ncols is 0".into()));
        }
        if rows_rowmajor.len() % self.ncols != 0 {
            return Err(StateError::Shape("rows not multiple of ncols".into()));
        }
        let n = rows_rowmajor.len() / self.ncols;
        if y.len() != n {
            return Err(StateError::Shape("y length mismatch".into()));
        }
        if n == 0 {
            return Ok(());
        }
        if self.invalidated {
            return Err(unsupported("summary invalidated"));
        }
        for (row, response) in rows_rowmajor.chunks_exact(self.ncols).zip(y) {
            self.validate_row(row, *response)?;
        }
        if !self.published_matches_scatter() {
            return Err(unsupported("summary is out of step with its scatter"));
        }
        u64::try_from(n)
            .ok()
            .and_then(|count| self.n.checked_add(count))
            .ok_or_else(|| StateError::Numerical("covariance count overflow".into()))?;
        let mut joint_row = Vec::with_capacity(self.ncols.saturating_add(1));
        for (row, response) in rows_rowmajor.chunks_exact(self.ncols).zip(y) {
            self.append_validated(row, *response, &mut joint_row)?;
        }
        Ok(())
    }

    /// Solve OLS coefficients from `XᵀX β = Xᵀy` by Cholesky.
    ///
    /// The factorisation solves the Gram system directly. It does not form
    /// `(XᵀX)⁻¹`, so the solve does not square the conditioning of the Gram.
    ///
    /// # Errors
    ///
    /// The summary was invalidated, its published moments disagree with the joint
    /// scatter, there are no observations, or the Gram is singular.
    pub fn solve_beta(&self) -> Result<Vec<f64>, StateError> {
        if self.invalidated {
            return Err(unsupported("summary invalidated"));
        }
        if self.n == 0 {
            return Err(StateError::Numerical("no observations".into()));
        }
        if !self.published_matches_scatter() {
            return Err(unsupported("summary is out of step with its scatter"));
        }
        solve_gram(&self.xtx, &self.xty, self.ncols)
    }

    /// Residual variance estimate `σ² = SSE / (n − p)` when `n > p`.
    ///
    /// `SSE = Σ (y − xᵀβ)²` for *any* `β` and any design (with or without a constant
    /// column). It is assembled from the mean-centred Welford scatter, which is stable under
    /// a large common offset but only equals `SSE` when the residuals average to zero:
    /// `Σ r² = Σ (r − r̄)² + n r̄²` with `r̄ = ȳ − x̄ᵀβ`. For OLS on a design containing an
    /// intercept `r̄ = 0` and the second term vanishes; through-origin designs and
    /// externally supplied coefficients need it. The `n r̄²` term is kept whenever it
    /// is finite, including when `|r̄|` is only a few units in the last place of the
    /// means. A non-finite correction means the residual scale is not representable
    /// and the variance is `None`.
    ///
    /// Returns `None` when a weight or mask change has invalidated this summary, or
    /// when the published moments no longer agree with the joint scatter.
    #[must_use]
    pub fn residual_variance(&self, beta: &[f64]) -> Option<f64> {
        if self.invalidated {
            return None;
        }
        if beta.len() != self.ncols || usize::try_from(self.n).is_ok_and(|n| n <= self.ncols) {
            return None;
        }
        if !self.published_matches_scatter() {
            return None;
        }
        let sse = centered_residual_sse(
            &self.joint.m2,
            &self.joint.mean,
            self.ncols,
            beta,
            self.n as f64,
        )?;
        Some(sse / (self.n as f64 - self.ncols as f64))
    }

    /// Merge `other` into this summary.
    ///
    /// Gram terms add. The centered scatter combines with Chan's formula, taking the
    /// larger count as the base so a shared offset is not cancelled. Merging `A` then
    /// `B` matches merging `B` then `A`. Both match a summary rebuilt by appending the
    /// concatenated rows, within [`merge_within_tolerance`].
    ///
    /// An invalidated summary is refused. A summary whose scatter count was moved
    /// without its Gram is refused. Neither case is patched in place.
    ///
    /// # Errors
    ///
    /// Column-count mismatch, a non-finite summary, an invalidated summary, published
    /// moments that disagree with the joint scatter, or a count that would overflow.
    #[must_use = "a refused merge must be recomputed from the retained rows"]
    pub fn merge(&mut self, other: &Self) -> Result<(), StateError> {
        if self.invalidated || other.invalidated {
            return Err(unsupported("summary invalidated"));
        }
        if self.ncols != other.ncols {
            return Err(StateError::Shape(format!("ncols {} != {}", self.ncols, other.ncols)));
        }
        if self.xtx.len() != other.xtx.len() || self.xty.len() != other.xty.len() {
            return Err(StateError::Shape("gram length mismatch".into()));
        }
        if !self.scatter_aligned() || !other.scatter_aligned() {
            return Err(unsupported("summary is out of step with its scatter"));
        }
        if !self.entries_finite() || !other.entries_finite() {
            return Err(unsupported("summary is non-finite"));
        }
        if !self.published_matches_scatter() || !other.published_matches_scatter() {
            return Err(unsupported("summary is out of step with its scatter"));
        }
        if other.n == 0 {
            return Ok(());
        }
        if self.n == 0 {
            let retention = self.retention;
            *self = other.clone();
            self.retention = retention;
            return Ok(());
        }
        let n = self
            .n
            .checked_add(other.n)
            .ok_or_else(|| StateError::Numerical("observation count overflow".into()))?;
        let joint = self.joint.merged(&other.joint)?;
        let mut xtx = vec![0.0; self.xtx.len()];
        for (slot, (left, right)) in xtx.iter_mut().zip(self.xtx.iter().zip(&other.xtx)) {
            *slot = add_f64(*left, *right);
        }
        let mut xty = vec![0.0; self.xty.len()];
        for (slot, (left, right)) in xty.iter_mut().zip(self.xty.iter().zip(&other.xty)) {
            *slot = add_f64(*left, *right);
        }
        let yty = add_f64(self.yty, other.yty);
        self.xtx = xtx;
        self.xty = xty;
        self.yty = yty;
        self.n = n;
        self.joint = joint;
        Ok(())
    }

    /// Apply one supported append, or refuse an unsupported window change.
    ///
    /// [`LinearOlsDelta::AppendRow`] and [`LinearOlsDelta::AppendWindow`] update the
    /// summary. Evicting a summarized row or inserting a column is refused and leaves
    /// the summary unchanged. A weight or mask change, including a change to one row's
    /// weight, clears the summary instead of patching a term. Every refusal names the
    /// next attempt: recompute from the retained rows.
    ///
    /// # Errors
    ///
    /// The delta is unsupported, the summary is already invalidated, or the appended
    /// row or window is rejected by [`Self::append_row`] or [`Self::merge`].
    #[must_use = "an unsupported delta must be recomputed from the retained rows"]
    pub fn apply_delta(&mut self, delta: LinearOlsDelta<'_>) -> Result<(), StateError> {
        match delta {
            LinearOlsDelta::AppendRow { row, y } => self.append_row(row, y),
            LinearOlsDelta::AppendWindow { window } => self.merge(window),
            LinearOlsDelta::EvictRow { .. } => Err(unsupported("cannot evict a summarized row")),
            LinearOlsDelta::ColumnInserted => {
                Err(unsupported("cannot insert a column into a summary"))
            }
            LinearOlsDelta::RowWeightChanged => {
                Err(self.invalidate("a row weight change is not applied in place"))
            }
            LinearOlsDelta::WeightsChanged => {
                Err(self.invalidate("a weight change invalidates the summary"))
            }
            LinearOlsDelta::MaskChanged => {
                Err(self.invalidate("a mask change invalidates the summary"))
            }
        }
    }

    fn scatter_aligned(&self) -> bool {
        self.joint.n == self.n && self.joint.dim == self.ncols.saturating_add(1)
    }

    fn entries_finite(&self) -> bool {
        self.yty.is_finite()
            && self.xtx.iter().all(|value| value.is_finite())
            && self.xty.iter().all(|value| value.is_finite())
            && self.joint.mean.iter().all(|value| value.is_finite())
            && self.joint.m2.iter().all(|value| value.is_finite())
    }

    /// Published raw moments against `M2 + n μ μᵀ` from the joint scatter.
    ///
    /// Agreement is relative (`1e-6` of the larger magnitude, and at least `1e-6`).
    /// That admits the rounding between the raw products and the centered scatter,
    /// and still refuses a write that moves an O(1) moment by a unit.
    fn published_matches_scatter(&self) -> bool {
        let Some(joint_dim) = self.ncols.checked_add(1) else {
            return false;
        };
        if self.joint.dim != joint_dim || self.joint.n != self.n {
            return false;
        }
        let Some(gram) = self.ncols.checked_mul(self.ncols) else {
            return false;
        };
        if self.xtx.len() != gram || self.xty.len() != self.ncols {
            return false;
        }
        let p = self.ncols;
        for i in 0..p {
            if !moments_close(self.xty[i], rebuilt_moment(&self.joint, i, p)) {
                return false;
            }
            for j in 0..p {
                if !moments_close(self.xtx[i * p + j], rebuilt_moment(&self.joint, i, j)) {
                    return false;
                }
            }
        }
        moments_close(self.yty, rebuilt_moment(&self.joint, p, p))
    }

    fn invalidate(&mut self, reason: &str) -> StateError {
        let ncols = self.ncols;
        let retention = self.retention;
        *self = Self::new(ncols).expect("ncols already fitted in the summary");
        self.retention = retention;
        self.invalidated = true;
        unsupported(reason)
    }
}

/// `XᵀX β = Xᵀy` from a Cholesky factor. `None` of an inverse is formed.
pub(crate) fn solve_gram(xtx: &[f64], xty: &[f64], ncols: usize) -> Result<Vec<f64>, StateError> {
    let singular = StateError::Numerical("singular XtX".into());
    let Some(nn) = ncols.checked_mul(ncols) else {
        return Err(StateError::Shape("ols gram dimension overflow".into()));
    };
    if ncols == 0 || xtx.len() < nn || xty.len() < ncols {
        return Err(singular);
    }
    let mut factor = vec![0.0; nn];
    if cholesky_spd_into(xtx, ncols, &mut factor).is_err() {
        return Err(singular);
    }
    let mut scratch = vec![0.0; ncols];
    let mut beta = vec![0.0; ncols];
    chol_solve_into(&factor, ncols, xty, &mut scratch, &mut beta);
    if beta.iter().any(|value| !value.is_finite()) {
        return Err(singular);
    }
    // Refine against an error-free residual: a coefficient near a large common offset
    // otherwise carries a unit-in-the-last-place error that the residual variance squares.
    let mut residual = vec![0.0; ncols];
    let mut step = vec![0.0; ncols];
    for _ in 0..3 {
        for (i, slot) in residual.iter_mut().enumerate() {
            let mut sum = CompensatedSum::default();
            sum.add(xty[i]);
            for (j, b) in beta.iter().enumerate() {
                let a = xtx[i * ncols + j];
                let product = a * b;
                sum.add(-product);
                sum.add(-a.mul_add(*b, -product));
            }
            *slot = sum.total();
        }
        chol_solve_into(&factor, ncols, &residual, &mut scratch, &mut step);
        if step.iter().any(|value| !value.is_finite()) {
            break;
        }
        let settled = step
            .iter()
            .zip(&beta)
            .all(|(d, b)| d.abs() <= f64::EPSILON * b.abs().max(f64::MIN_POSITIVE));
        for (b, d) in beta.iter_mut().zip(&step) {
            *b += d;
        }
        if settled {
            break;
        }
    }
    Ok(beta)
}

/// `Σ w (y − xᵀβ)²` from a mean-centred scatter of `[x | y]` (`m2`, row-major
/// `(p + 1)²`) with total weight `mass`; `None` when the value is not representable or is
/// negative beyond rounding.
///
/// `SSE = Σ w (r − r̄)² + mass · r̄²` with `r̄ = ȳ − x̄ᵀβ`: the scatter is stable under a
/// large common offset, and the second term (zero for OLS with an intercept) makes the result
/// exact for through-origin designs and externally supplied coefficients. Shared by the
/// unit-weight and the real-weight summaries.
pub(crate) fn centered_residual_sse(
    m2: &[f64],
    mean: &[f64],
    p: usize,
    beta: &[f64],
    mass: f64,
) -> Option<f64> {
    let dim = p + 1;
    if beta.len() != p || m2.len() < dim * dim || mean.len() < dim {
        return None;
    }
    let mut sse = m2[p * dim + p];
    let mut bsy = 0.0;
    let mut bsb = 0.0;
    for i in 0..p {
        bsy += beta[i] * m2[i * dim + p];
        for j in 0..p {
            bsb += beta[i] * m2[i * dim + j] * beta[j];
        }
    }
    sse = sse - 2.0 * bsy + bsb;
    if !sse.is_finite() {
        return None;
    }
    let mut r_bar = mean[p];
    for i in 0..p {
        r_bar -= beta[i] * mean[i];
    }
    let correction = mass * r_bar * r_bar;
    if !correction.is_finite() {
        return None;
    }
    sse += correction;
    if sse < 0.0 {
        let scale = m2[p * dim + p].abs().max(1.0);
        if sse.abs() <= 1e-12 * scale {
            sse = 0.0;
        } else {
            return None;
        }
    }
    Some(sse)
}

/// Chan's merge of two mean-centred scatters, `base` taken as the reference so a shared
/// offset is not cancelled twice. Masses are counts or total weights; `total_mass` is their
/// sum. Returns the merged `(mean, m2)`.
pub(crate) fn chan_merge_scatter(
    dim: usize,
    base: (f64, &[f64], &[f64]),
    add: (f64, &[f64], &[f64]),
    total_mass: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (base_mass, base_mean, base_m2) = base;
    let (add_mass, add_mean, add_m2) = add;
    let weight = add_mass / total_mass;
    let scale = base_mass * add_mass / total_mass;
    let mut mean = vec![0.0; dim];
    let mut delta = vec![0.0; dim];
    let means = mean.iter_mut().zip(&mut delta);
    let sources = base_mean.iter().zip(add_mean);
    for ((slot, gap), (base_value, add_value)) in means.zip(sources) {
        let step = *add_value - *base_value;
        *gap = step;
        *slot = *base_value + step * weight;
    }
    let mut m2 = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let value =
                add_f64(base_m2[i * dim + j], add_m2[i * dim + j]) + delta[i] * delta[j] * scale;
            m2[i * dim + j] = value;
            m2[j * dim + i] = value;
        }
    }
    (mean, m2)
}

/// One weighted Welford step of the scatter of `z` (`dim` entries) with weight `weight`,
/// given the total weight `mass_before` already absorbed (zero for the first row).
///
/// `delta` is caller-owned scratch. Only the upper triangle is computed and mirrored, so the
/// scatter stays exactly symmetric.
pub(crate) fn update_weighted_scatter(
    mean: &mut [f64],
    m2: &mut [f64],
    mass_before: f64,
    weight: f64,
    z: &[f64],
    delta: &mut Vec<f64>,
) {
    let dim = z.len();
    let mass_next = mass_before + weight;
    delta.clear();
    delta.extend(z.iter().zip(mean.iter()).map(|(value, m)| value - m));
    for (m, step) in mean.iter_mut().zip(delta.iter()) {
        *m += step * (weight / mass_next);
    }
    for i in 0..dim {
        for j in i..dim {
            let value = m2[i * dim + j] + weight * delta[i] * (z[j] - mean[j]);
            m2[i * dim + j] = value;
            m2[j * dim + i] = value;
        }
    }
}

/// Start of a retained-row source digest: the tag, the source and the row width.
pub(crate) fn source_digest_start(tag: &str, source: &str, width: usize) -> CanonicalWriter {
    let mut writer = CanonicalWriter::new(tag);
    writer.text(source).word(width as u64);
    writer
}

/// Refusal of a row id that is already in a summary.
pub(crate) fn repeated_row(id: u64) -> StateError {
    StateError::InvalidEvent(format!(
        "row {id} is already in the summary; each observation enters once, so drop the \
         duplicate or evict the earlier row first"
    ))
}

/// Validate `rows` with `validate` (which returns each row's id) and collect the ids,
/// refusing an id that is in `taken` or repeats within `rows`.
pub(crate) fn checked_row_ids<R>(
    rows: &[R],
    taken: &BTreeSet<u64>,
    validate: impl Fn(&R) -> Result<u64, StateError>,
) -> Result<BTreeSet<u64>, StateError> {
    let mut ids = BTreeSet::new();
    for row in rows {
        let id = validate(row)?;
        if taken.contains(&id) || !ids.insert(id) {
            return Err(repeated_row(id));
        }
    }
    Ok(ids)
}

/// Whether two slices agree element-wise within [`merge_within_tolerance`].
pub(crate) fn all_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| merge_within_tolerance(*a, *b))
}

/// [`all_close`] for optional slices; two absent values agree.
pub(crate) fn optional_close(left: Option<&[f64]>, right: Option<&[f64]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => all_close(a, b),
        _ => false,
    }
}

/// [`merge_within_tolerance`] for optional scalars; two absent values agree.
pub(crate) fn optional_scalar_close(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => merge_within_tolerance(a, b),
        _ => false,
    }
}

fn rebuilt_moment(joint: &StreamingCovariance, row: usize, col: usize) -> f64 {
    let dim = joint.dim;
    joint.m2[row * dim + col] + (joint.n as f64) * joint.mean[row] * joint.mean[col]
}

fn moments_close(stored: f64, rebuilt: f64) -> bool {
    if !stored.is_finite() || !rebuilt.is_finite() {
        return false;
    }
    let scale = stored.abs().max(rebuilt.abs()).max(1.0);
    (stored - rebuilt).abs() <= 1e-6 * scale
}

fn square_len(dim: usize, message: &str) -> Result<usize, StateError> {
    dim.checked_mul(dim).ok_or_else(|| StateError::Shape(message.into()))
}

fn unsupported(reason: &str) -> StateError {
    StateError::Numerical(format!("{reason}; {RECOMPUTE_FROM_RETAINED_ROWS}"))
}

/// Add two finite values so `add_f64(a, b)` and `add_f64(b, a)` are the same bits.
#[allow(
    clippy::if_same_then_else,
    reason = "operand order is the recorded bits; a + b and b + a are not the same f64"
)]
pub(crate) const fn add_f64(left: f64, right: f64) -> f64 {
    let left_abs = left.abs().to_bits();
    let right_abs = right.abs().to_bits();
    let left_first =
        left_abs < right_abs || (left_abs == right_abs && left.to_bits() <= right.to_bits());
    if left_first { left + right } else { right + left }
}

/// Larger count is the Chan base. Equal counts break the tie on mean bits.
fn chan_base_is_left(left_n: u64, left_mean: &[f64], right_n: u64, right_mean: &[f64]) -> bool {
    if left_n != right_n {
        return left_n > right_n;
    }
    for (left, right) in left_mean.iter().zip(right_mean) {
        let left_bits = left.to_bits();
        let right_bits = right.to_bits();
        if left_bits != right_bits {
            return left_bits < right_bits;
        }
    }
    true
}

/// Streaming mean / covariance (Welford updates, merged by Chan's formula).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingCovariance {
    /// Dimension.
    dim: usize,
    /// Observation count.
    n: u64,
    /// Running mean.
    mean: Vec<f64>,
    /// Upper-triangular packed? — store full `dim×dim` unnormalized scatter `M2`.
    m2: Vec<f64>,
    /// Retention.
    retention: RetentionPolicy,
}

impl StreamingCovariance {
    /// Empty streaming covariance for `dim`.
    ///
    /// # Errors
    ///
    /// `dim²` does not fit in `usize`.
    pub fn new(dim: usize) -> Result<Self, StateError> {
        let cells = square_len(dim, "covariance dimension overflow")?;
        Ok(Self {
            dim,
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; cells],
            retention: RetentionPolicy::SufficientStatisticsOnly,
        })
    }

    /// Observation count.
    #[must_use]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Running mean.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Unnormalized `dim×dim` scatter `M2`.
    #[must_use]
    pub fn m2(&self) -> &[f64] {
        &self.m2
    }

    /// Append one observation.
    ///
    /// # Errors
    ///
    /// Dimension mismatch.
    pub fn append(&mut self, x: &[f64]) -> Result<(), StateError> {
        if x.len() != self.dim {
            return Err(StateError::Shape(format!("cov dim {} != {}", x.len(), self.dim)));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(StateError::Numerical("covariance observations must be finite".into()));
        }
        let next_n = self
            .n
            .checked_add(1)
            .ok_or_else(|| StateError::Numerical("covariance count overflow".into()))?;
        if self.n == 0 {
            self.mean.copy_from_slice(x);
            self.n = next_n;
            return Ok(());
        }
        // Same rank-one update this type has always applied, now shared with the
        // kernel crate. It is not `Welford::push`: the two evaluation orders can
        // differ in the last bit of a diagonal entry. Accessors expose that state.
        antecedent_kernels::update_scatter(&mut self.mean, &mut self.m2, self.n, x);
        self.n = next_n;
        Ok(())
    }

    /// Merge `other` into this scatter with Chan's formula.
    ///
    /// Merging `A` then `B` matches merging `B` then `A`. Gram entries and means match a
    /// scatter rebuilt by appending the concatenated rows, within
    /// [`merge_within_tolerance`]. A centered second moment of a shared large offset
    /// can also differ by the rounding of that offset: both the one-pass update and
    /// this merge subtract numbers whose magnitude is the offset. The larger count is
    /// the base so that offset is not cancelled twice.
    ///
    /// # Errors
    ///
    /// Dimension mismatch, a non-finite summary, or a count that would overflow.
    #[must_use = "a refused covariance merge must be recomputed from the retained rows"]
    pub fn merge(&mut self, other: &Self) -> Result<(), StateError> {
        let merged = self.merged(other)?;
        *self = merged;
        Ok(())
    }

    fn merged(&self, other: &Self) -> Result<Self, StateError> {
        if self.dim != other.dim {
            return Err(StateError::Shape(format!("cov dim {} != {}", self.dim, other.dim)));
        }
        if !self.entries_finite() || !other.entries_finite() {
            return Err(unsupported("covariance summary is non-finite"));
        }
        if other.n == 0 {
            return Ok(self.clone());
        }
        if self.n == 0 {
            let mut out = other.clone();
            out.retention = self.retention;
            return Ok(out);
        }
        let n = self
            .n
            .checked_add(other.n)
            .ok_or_else(|| StateError::Numerical("covariance count overflow".into()))?;
        let (base, add) = if chan_base_is_left(self.n, &self.mean, other.n, &other.mean) {
            (self, other)
        } else {
            (other, self)
        };
        let (mean, m2) = chan_merge_scatter(
            self.dim,
            (base.n as f64, &base.mean, &base.m2),
            (add.n as f64, &add.mean, &add.m2),
            n as f64,
        );
        let dim = self.dim;
        Ok(Self { dim, n, mean, m2, retention: self.retention })
    }

    fn entries_finite(&self) -> bool {
        self.mean.iter().all(|value| value.is_finite())
            && self.m2.iter().all(|value| value.is_finite())
    }

    /// Sample covariance matrix (`n − 1` denominator); `None` if `n < 2`.
    #[must_use]
    pub fn sample_covariance(&self) -> Option<Vec<f64>> {
        if self.n < 2 {
            return None;
        }
        let denom = (self.n - 1) as f64;
        Some(self.m2.iter().map(|v| v / denom).collect())
    }
}

/// Cached lagged sample-index key (semantic; no borrowed buffers).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LagIndexCacheKey {
    /// Data version.
    data_version: u64,
    /// Max lag.
    max_lag: u32,
    /// Variable set fingerprint.
    var_fingerprint: u64,
}

/// Lag-index cache entry metadata (values reconstructed by callers).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagIndexCacheEntry {
    /// Key.
    key: LagIndexCacheKey,
    /// Approximate retained bytes.
    bytes: u64,
    /// Retention.
    retention: RetentionPolicy,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ols(ncols: usize) -> LinearOlsSuffStats {
        LinearOlsSuffStats::new(ncols).expect("dimension")
    }

    fn cov(dim: usize) -> StreamingCovariance {
        StreamingCovariance::new(dim).expect("dimension")
    }

    #[test]
    fn streaming_covariance_rejects_nonfinite_rows_without_poisoning_state() {
        let mut stats = cov(2);
        stats.append(&[1.0, 2.0]).unwrap();
        let before = stats.clone();
        assert!(stats.append(&[3.0, f64::NAN]).is_err());
        assert_eq!(stats, before);
        stats.append(&[3.0, 4.0]).unwrap();
        let covariance = stats.sample_covariance().unwrap();
        assert!(covariance.iter().all(|value| (*value - 2.0).abs() < 1e-12));
    }

    /// Brute-force `Σ (y − xᵀβ)² / (n − p)`, independent of the accumulated scatter.
    fn brute_force_sigma2(rows: &[f64], y: &[f64], p: usize, beta: &[f64]) -> f64 {
        let n = y.len();
        let sse: f64 = (0..n)
            .map(|i| {
                let pred: f64 = (0..p).map(|j| rows[i * p + j] * beta[j]).sum();
                (y[i] - pred) * (y[i] - pred)
            })
            .sum();
        sse / (n - p) as f64
    }

    /// Through-origin fit (no constant column): the residuals do not average to zero, so the
    /// centred scatter alone understates σ². x = [1,2,3], y = [11,12,13]: β = 74/14,
    /// Σr² = 300/7, σ² = 150/7 (the centred scatter gives 36.7/2 = 18.4).
    #[test]
    fn residual_variance_is_exact_without_an_intercept_column() {
        let rows = [1.0, 2.0, 3.0];
        let y = [11.0, 12.0, 13.0];
        let mut stats = ols(1);
        stats.append_batch(&rows, &y).unwrap();
        let beta = stats.solve_beta().unwrap();
        assert!((beta[0] - 74.0 / 14.0).abs() < 1e-12);
        let var = stats.residual_variance(&beta).expect("variance");
        assert!((var - 150.0 / 7.0).abs() < 1e-10, "got {var}");
        assert!((var - brute_force_sigma2(&rows, &y, 1, &beta)).abs() < 1e-10);

        // x = [1,2,3], y = [2,3,5]: β = 23/14, Σr² = 38 − 23²/14 = 3/14, σ² = 3/28.
        let y2 = [2.0, 3.0, 5.0];
        let mut s2 = ols(1);
        s2.append_batch(&rows, &y2).unwrap();
        let b2 = s2.solve_beta().unwrap();
        let v2 = s2.residual_variance(&b2).expect("variance");
        assert!((v2 - 3.0 / 28.0).abs() < 1e-12, "got {v2}");
    }

    /// An externally supplied β that is not the OLS solution: y = 1 + 2x fitted exactly by
    /// (1, 2); the wrong intercept (0, 2) leaves a constant residual 1, so Σr² = 4 and
    /// σ² = 4 / (4 − 2) = 2. A centred scatter sees a constant and reports 0.
    #[test]
    fn residual_variance_is_exact_for_a_non_ols_beta() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let mut stats = ols(2);
        stats.append_batch(&rows, &y).unwrap();
        let wrong = [0.0, 2.0];
        let var = stats.residual_variance(&wrong).expect("variance");
        assert!((var - 2.0).abs() < 1e-12, "got {var}");
        assert!((var - brute_force_sigma2(&rows, &y, 2, &wrong)).abs() < 1e-12);
        // The true OLS β still gives (numerically) zero residual variance.
        let ols = stats.solve_beta().unwrap();
        assert!(stats.residual_variance(&ols).expect("variance").abs() < 1e-12);
    }

    /// A non-finite row is refused before any state changes; later valid rows still
    /// accumulate exactly as if the bad row had never been offered.
    #[test]
    fn nonfinite_row_is_rejected_atomically() {
        let mut stats = ols(2);
        stats.append_row(&[1.0, 0.0], 1.0).unwrap();
        stats.append_row(&[1.0, 1.0], 3.0).unwrap();
        let before = stats.clone();
        assert!(stats.append_row(&[1.0, f64::NAN], 2.0).is_err());
        assert!(stats.append_row(&[1.0, 2.0], f64::INFINITY).is_err());
        assert_eq!(stats, before);

        let mut clean = ols(2);
        clean.append_row(&[1.0, 0.0], 1.0).unwrap();
        clean.append_row(&[1.0, 1.0], 3.0).unwrap();
        stats.append_row(&[1.0, 2.0], 5.0).unwrap();
        clean.append_row(&[1.0, 2.0], 5.0).unwrap();
        assert_eq!(stats, clean);
        let beta = stats.solve_beta().unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-10 && (beta[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn incremental_ols_matches_full_batch() {
        let rows = [
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 2.0, //
            1.0, 3.0,
        ];
        let y = [1.0, 3.0, 5.0, 7.0];
        let mut full = ols(2);
        full.append_batch(&rows, &y).expect("batch");
        let mut inc = ols(2);
        for i in 0..4 {
            inc.append_row(&rows[i * 2..(i + 1) * 2], y[i]).expect("row");
        }
        assert_eq!(full.n, inc.n);
        for i in 0..4 {
            assert!((full.xtx[i] - inc.xtx[i]).abs() < 1e-12);
        }
        let b_full = full.solve_beta().expect("beta");
        let b_inc = inc.solve_beta().expect("beta");
        assert!((b_full[0] - b_inc[0]).abs() < 1e-10);
        assert!((b_full[1] - b_inc[1]).abs() < 1e-10);
        // y ≈ 1 + 2x
        assert!((b_inc[0] - 1.0).abs() < 1e-8);
        assert!((b_inc[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn incremental_ols_residual_variance_survives_large_offset() {
        let offset = 1e12;
        let eps = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = eps.iter().map(|e| offset + e).collect();
        let rows = vec![1.0; 5];
        let mut stats = ols(1);
        stats.append_batch(&rows, &y).unwrap();
        let beta = stats.solve_beta().unwrap();
        let var = stats.residual_variance(&beta).expect("residual variance");
        assert!((var - 2.5).abs() < 1e-9, "got {var}");

        let shifted: Vec<f64> = y.iter().map(|yi| yi + 1e9).collect();
        let mut shifted_stats = ols(1);
        shifted_stats.append_batch(&rows, &shifted).unwrap();
        let shifted_var = shifted_stats.residual_variance(&shifted_stats.solve_beta().unwrap());
        assert!((shifted_var.unwrap() - var).abs() < 1e-9);
    }

    #[test]
    fn streaming_cov_matches_batch() {
        // Distinct variances, partial correlation: x2 = 0.5*x1 + noise
        let data: Vec<[f64; 2]> = (0..20)
            .map(|i| {
                let x1 = f64::from(i) - 9.5;
                let x2 = 0.5 * x1 + f64::from(i % 3) - 1.0;
                [x1, x2 * 2.0]
            })
            .collect();
        let mut s = cov(2);
        for row in &data {
            s.append(row).expect("append");
        }
        let cov = s.sample_covariance().expect("cov");

        let n = data.len() as f64;
        let mut mean = [0.0, 0.0];
        for row in &data {
            mean[0] += row[0];
            mean[1] += row[1];
        }
        mean[0] /= n;
        mean[1] /= n;
        let mut batch = [0.0; 4];
        for row in &data {
            let d0 = row[0] - mean[0];
            let d1 = row[1] - mean[1];
            batch[0] += d0 * d0;
            batch[1] += d0 * d1;
            batch[2] += d1 * d0;
            batch[3] += d1 * d1;
        }
        let denom = n - 1.0;
        for v in &mut batch {
            *v /= denom;
        }

        for i in 0..4 {
            assert!((cov[i] - batch[i]).abs() < 1e-10, "cov[{i}]={} batch={}", cov[i], batch[i]);
        }
        assert!((cov[1] - cov[2]).abs() < 1e-12, "covariance must be symmetric");
        // Off-diagonal must be non-trivial (catches the old delta[i]*di bug)
        assert!(cov[1].abs() > 1.0);
    }

    fn batch(rows: &[f64], y: &[f64], ncols: usize) -> LinearOlsSuffStats {
        let mut stats = ols(ncols);
        stats.append_batch(rows, y).expect("batch");
        stats
    }

    fn assert_stats_within_tolerance(left: &LinearOlsSuffStats, right: &LinearOlsSuffStats) {
        assert_eq!(left.ncols, right.ncols);
        assert_eq!(left.n, right.n);
        assert_eq!(left.invalidated, right.invalidated);
        assert_eq!(left.joint.n, right.joint.n);
        assert_eq!(left.xtx.len(), right.xtx.len());
        assert_eq!(left.xty.len(), right.xty.len());
        assert!(merge_within_tolerance(left.yty, right.yty), "yty {} vs {}", left.yty, right.yty);
        for (i, (a, b)) in left.xtx.iter().zip(&right.xtx).enumerate() {
            assert!(merge_within_tolerance(*a, *b), "xtx[{i}] {a} vs {b}");
        }
        for (i, (a, b)) in left.xty.iter().zip(&right.xty).enumerate() {
            assert!(merge_within_tolerance(*a, *b), "xty[{i}] {a} vs {b}");
        }
        for (i, (a, b)) in left.joint.mean.iter().zip(&right.joint.mean).enumerate() {
            assert!(merge_within_tolerance(*a, *b), "mean[{i}] {a} vs {b}");
        }
        let dim = left.joint.mean.len();
        for i in 0..dim {
            for j in 0..dim {
                let a = left.joint.m2[i * dim + j];
                let b = right.joint.m2[i * dim + j];
                let magnitude = left.joint.mean[i]
                    .abs()
                    .max(left.joint.mean[j].abs())
                    .max(right.joint.mean[i].abs())
                    .max(right.joint.mean[j].abs());
                assert!(moment_within_tolerance(a, b, magnitude), "m2[{}] {a} vs {b}", i * dim + j);
            }
        }
    }

    /// [`merge_within_tolerance`], plus the rounding of a shared offset in a centered moment.
    fn moment_within_tolerance(left: f64, right: f64, magnitude: f64) -> bool {
        if merge_within_tolerance(left, right) {
            return true;
        }
        let spacing = magnitude.abs() * f64::EPSILON;
        (left - right).abs() <= spacing * 64.0
    }

    fn assert_merge_orders(rows: &[f64], y: &[f64], ncols: usize, split: usize) {
        let left = batch(&rows[..split * ncols], &y[..split], ncols);
        let right = batch(&rows[split * ncols..], &y[split..], ncols);
        let mut ab = left.clone();
        ab.merge(&right).expect("merge ab");
        let mut ba = right.clone();
        ba.merge(&left).expect("merge ba");
        assert_eq!(ab, ba, "merge order at split {split}");
        let mut via_delta = left.clone();
        via_delta
            .apply_delta(LinearOlsDelta::AppendWindow { window: &right })
            .expect("append window");
        assert_eq!(via_delta, ab);
        assert_stats_within_tolerance(&ab, &batch(rows, y, ncols));
    }

    /// `y = 1e12 + 1 + 2x + eps` on an intercept column. The offset sits in the
    /// response, so a centred merge has to keep the residual scale.
    fn offset_design() -> (Vec<f64>, Vec<f64>) {
        let offset = 1e12;
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let eps = [-1.5, 0.5, -0.25, 1.25, -0.75, 0.75];
        let mut rows = Vec::with_capacity(xs.len() * 2);
        let mut y = Vec::with_capacity(xs.len());
        for i in 0..xs.len() {
            rows.push(1.0);
            rows.push(xs[i]);
            y.push(offset + 1.0 + 2.0 * xs[i] + eps[i]);
        }
        (rows, y)
    }

    fn assert_recompute_refusal(err: &StateError) {
        assert!(err.to_string().contains("recompute from the retained rows"), "{err}");
    }

    #[test]
    fn merge_orders_match_each_other_and_full_recompute() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        assert_merge_orders(&rows, &y, 2, 1);
        assert_merge_orders(&rows, &y, 2, 2);
        let full = batch(&rows, &y, 2);
        let mut empty = ols(2);
        empty.merge(&full).expect("merge into empty");
        assert_eq!(empty, full);
        let mut restored = full.clone();
        restored.merge(&ols(2)).expect("merge empty");
        assert_eq!(restored, full);

        let mut widened = full.clone();
        let snapshot = widened.clone();
        let err = widened.merge(&ols(1)).expect_err("ncols");
        assert!(matches!(err, StateError::Shape(_)));
        assert_eq!(widened, snapshot);

        let mut dirty = full.clone();
        dirty.xtx[0] = f64::NAN;
        let err = dirty.merge(&full).expect_err("non-finite");
        assert_recompute_refusal(&err);
        assert_eq!(dirty.n, full.n);
        assert_eq!(dirty.xtx[0].to_bits(), f64::NAN.to_bits());
        assert_eq!(dirty.xty, full.xty);
    }

    #[test]
    fn merged_summary_matches_recompute_without_offset_drift() {
        let (rows, y) = offset_design();
        let ncols = 2;
        assert_merge_orders(&rows, &y, ncols, 2);
        assert_merge_orders(&rows, &y, ncols, 3);

        let mut chained = ols(ncols);
        for i in 0..y.len() {
            let mut one = ols(ncols);
            one.append_row(&rows[i * ncols..(i + 1) * ncols], y[i]).expect("row");
            chained.apply_delta(LinearOlsDelta::AppendWindow { window: &one }).expect("window");
        }
        let full = batch(&rows, &y, ncols);
        assert_stats_within_tolerance(&chained, &full);
        let beta = chained.solve_beta().expect("beta");
        let beta_full = full.solve_beta().expect("beta");
        for (got, expected) in beta.iter().zip(&beta_full) {
            assert!(merge_within_tolerance(*got, *expected), "{got} vs {expected}");
        }
        let merged_var = chained.residual_variance(&beta).expect("variance");
        let full_var = full.residual_variance(&beta_full).expect("variance");
        let brute = brute_force_sigma2(&rows, &y, ncols, &beta_full);
        assert!((merged_var - full_var).abs() < 1e-6, "merge {merged_var} full {full_var}");
        // A shared 1e12 offset rounds the centered moment. The residual scale still
        // has to match the rows; 1e-6 is the floor when that rounding is negligible.
        let offset_bound = (1e12 * f64::EPSILON * 64.0) / (y.len() as f64 - ncols as f64);
        let variance_bound = offset_bound.max(1e-6);
        assert!((merged_var - brute).abs() <= variance_bound, "merge {merged_var} brute {brute}");

        let points = [
            [1e12 - 2.0, 1.0],
            [1e12 - 1.0, 2.0],
            [1e12, 0.0],
            [1e12 + 1.0, 3.0],
            [1e12 + 2.0, 1.5],
        ];
        let mut left = cov(2);
        let mut right = cov(2);
        let mut full_cov = cov(2);
        for (i, row) in points.iter().enumerate() {
            full_cov.append(row).expect("cov row");
            if i < 2 {
                left.append(row).expect("left");
            } else {
                right.append(row).expect("right");
            }
        }
        let mut ab = left.clone();
        ab.merge(&right).expect("cov merge");
        let mut ba = right.clone();
        ba.merge(&left).expect("cov merge");
        assert_eq!(ab, ba);
        for (i, (got, expected)) in ab.m2().iter().zip(full_cov.m2()).enumerate() {
            let dim = ab.mean().len();
            let row = i / dim;
            let col = i % dim;
            let magnitude = ab.mean()[row]
                .abs()
                .max(ab.mean()[col].abs())
                .max(full_cov.mean()[row].abs())
                .max(full_cov.mean()[col].abs());
            assert!(
                moment_within_tolerance(*got, *expected, magnitude),
                "m2[{i}] {got} vs {expected}"
            );
        }
        for (i, (got, expected)) in ab.mean().iter().zip(full_cov.mean()).enumerate() {
            assert!(merge_within_tolerance(*got, *expected), "mean[{i}] {got} vs {expected}");
        }
        let cov = ab.sample_covariance().expect("sample cov");
        assert!((cov[0] - 2.5).abs() < 1e-6, "drifted variance {}", cov[0]);
    }

    #[test]
    fn weight_and_mask_changes_invalidate_instead_of_patching() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let live = batch(&rows, &y, 2);
        for (delta, reason) in [
            (LinearOlsDelta::WeightsChanged, "a weight change invalidates the summary"),
            (LinearOlsDelta::MaskChanged, "a mask change invalidates the summary"),
            (LinearOlsDelta::RowWeightChanged, "a row weight change is not applied in place"),
        ] {
            let mut stats = live.clone();
            let err = stats.apply_delta(delta).expect_err("invalidate");
            assert_eq!(
                err,
                StateError::Numerical(format!("{reason}; recompute from the retained rows"))
            );
            assert!(stats.is_invalidated());
            assert_eq!(stats.n, 0);
            assert!(stats.xtx.iter().all(|value| value.to_bits() == 0));
            assert!(stats.xty.iter().all(|value| value.to_bits() == 0));
            assert_eq!(stats.yty.to_bits(), 0);
            assert!(stats.residual_variance(&[0.0, 1.0]).is_none());
            let solve_err = stats.solve_beta().expect_err("invalidated solve");
            assert_eq!(
                solve_err,
                StateError::Numerical(
                    "summary invalidated; recompute from the retained rows".into()
                )
            );
            let append_err = stats.append_row(&[1.0, 9.0], 4.0).expect_err("append");
            assert_eq!(append_err, solve_err);
            assert_eq!(stats.n, 0);
            let mut other = live.clone();
            let merge_err = other.merge(&stats).expect_err("merge invalidated");
            assert_eq!(merge_err, solve_err);
            assert_eq!(other, live);
        }
    }

    #[test]
    fn column_insert_and_eviction_are_refused_without_a_downdate() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let full = batch(&rows, &y, 2);
        let mut stats = full.clone();
        let err = stats.apply_delta(LinearOlsDelta::ColumnInserted).expect_err("column");
        assert_eq!(
            err,
            StateError::Numerical(
                "cannot insert a column into a summary; recompute from the retained rows".into()
            )
        );
        assert_eq!(stats, full);
        assert!(!stats.is_invalidated());

        let err = stats
            .apply_delta(LinearOlsDelta::EvictRow { row: &rows[..2], y: y[0] })
            .expect_err("evict");
        assert_eq!(
            err,
            StateError::Numerical(
                "cannot evict a summarized row; recompute from the retained rows".into()
            )
        );
        assert_eq!(stats, full);
        let retained = batch(&rows[2..], &y[1..], 2);
        assert_ne!(stats.n, retained.n);
        assert!(!merge_within_tolerance(stats.xtx[0], retained.xtx[0]));

        let mut downdated = full.clone();
        downdated.n = downdated.n.saturating_sub(1);
        let downdated_before = downdated.clone();
        let err = downdated.merge(&ols(2)).expect_err("partial downdate");
        assert_recompute_refusal(&err);
        assert_eq!(downdated, downdated_before);
    }

    #[test]
    fn appended_rows_through_delta_match_recompute() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let mut via_delta = ols(2);
        for i in 0..y.len() {
            let row = &rows[i * 2..(i + 1) * 2];
            via_delta.apply_delta(LinearOlsDelta::AppendRow { row, y: y[i] }).expect("append");
        }
        assert_eq!(via_delta, batch(&rows, &y, 2));
    }

    #[test]
    fn dimension_product_overflow_is_an_error() {
        let wide = 1usize << (usize::BITS / 2);
        let err = StreamingCovariance::new(wide).expect_err("covariance");
        assert!(matches!(err, StateError::Shape(_)), "{err}");
        let err = LinearOlsSuffStats::new(usize::MAX).expect_err("ols");
        assert!(matches!(err, StateError::Shape(_)), "{err}");
    }

    /// `|r̄| = 32` sits under the old gate `8 ε · scale` when the means are `2^54`,
    /// but `n r̄²` is the whole residual sum of squares. Dropping it reports zero.
    #[test]
    fn residual_variance_keeps_a_finite_mean_residual_under_the_old_gate() {
        let level = 2.0_f64.powi(54);
        let shift = 32.0;
        let rows = [level, level, level, level];
        let y = [level + shift, level + shift, level + shift, level + shift];
        let mut stats = ols(1);
        stats.append_batch(&rows, &y).unwrap();
        let var = stats.residual_variance(&[1.0]).expect("finite correction");
        let expect = (4.0 * shift * shift) / 3.0;
        assert!((var - expect).abs() <= 1e-6, "got {var} expect {expect}");
    }

    /// Gram `[[1, 1], [1, 1 + 2^-42]]`. The explicit inverse's pivot floor is
    /// `1e-12`; Cholesky's relative floor still accepts the Schur complement.
    #[test]
    fn solve_beta_factors_a_gram_the_explicit_inverse_refuses() {
        let h = 2.0_f64.powi(-21);
        let rows = [1.0, 1.0, 0.0, h];
        let y = [2.0, h];
        let mut stats = ols(2);
        stats.append_batch(&rows, &y).unwrap();
        assert!(antecedent_kernels::invert_square(stats.xtx(), 2).is_none());
        let beta = stats.solve_beta().expect("cholesky solve");
        assert!((beta[0] - 1.0).abs() < 1e-6 && (beta[1] - 1.0).abs() < 1e-6, "{beta:?}");
    }

    #[test]
    fn a_published_moment_that_leaves_the_scatter_is_refused() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let full = batch(&rows, &y, 2);
        let mut poked = full.clone();
        poked.xtx[0] += 1.0;
        let before = poked.clone();
        let err = poked.solve_beta().expect_err("drifted gram");
        assert_recompute_refusal(&err);
        assert!(poked.residual_variance(&[1.0, 2.0]).is_none());
        let merge_err = poked.merge(&ols(2)).expect_err("drifted merge");
        assert_recompute_refusal(&merge_err);
        let append_err = poked.append_row(&[1.0, 4.0], 9.0).expect_err("drifted append");
        assert_recompute_refusal(&append_err);
        assert_eq!(poked, before);

        let mut counted = full.clone();
        counted.n = counted.n.saturating_sub(1);
        assert!(counted.solve_beta().is_err());
    }
}

//! Partial-correlation kernels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

/// Scratch for residualization and Pearson correlation.
///
/// `design` stays public so a caller can check that a batch reused the same
/// allocation. The Gram, coefficients, and residuals are crate-private.
#[derive(Clone, Debug, Default)]
pub struct ParCorrWorkspace {
    /// Design matrix column-major `[Z…]` (`n * p`). Every column is centered and scaled
    /// to unit sum of squares, which is the implicit intercept: X and Y are residualized
    /// on `[1, Z…]` in a scale-free basis. A constant or collinear column is a refusal,
    /// not a ridge.
    pub design: Vec<f64>,
    /// `XtX` / Gram (`p^2`).
    gram: Vec<f64>,
    /// Coefficients of X then Y (`2p`).
    beta: Vec<f64>,
    /// Residual of X.
    rx: Vec<f64>,
    /// Residual of Y.
    ry: Vec<f64>,
    capacity_n: usize,
    capacity_p: usize,
}

impl ParCorrWorkspace {
    /// Ensure capacity for `n` rows and `p` covariates (no intercept column).
    pub fn prepare(&mut self, n: usize, p: usize) {
        let ncols = p.max(1);
        let need_design = n.saturating_mul(ncols);
        if self.design.len() < need_design {
            self.design.resize(need_design, 0.0);
        }
        self.prepare_solve(n, ncols);
        self.capacity_n = self.capacity_n.max(n);
        self.capacity_p = self.capacity_p.max(p);
    }

    /// Everything but the design matrix: Gram, the two coefficient vectors, residuals.
    fn prepare_solve(&mut self, n: usize, ncols: usize) {
        let need_gram = ncols.saturating_mul(ncols);
        if self.gram.len() < need_gram {
            self.gram.resize(need_gram, 0.0);
        }
        if self.beta.len() < 2 * ncols {
            self.beta.resize(2 * ncols, 0.0);
        }
        if self.rx.len() < n {
            self.rx.resize(n, 0.0);
        }
        if self.ry.len() < n {
            self.ry.resize(n, 0.0);
        }
    }

    /// Retained row capacity.
    #[must_use]
    pub const fn capacity_n(&self) -> usize {
        self.capacity_n
    }

    /// Retained covariate capacity.
    #[must_use]
    pub const fn capacity_p(&self) -> usize {
        self.capacity_p
    }
}

/// One batch query: column indexes into a shared column list.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParCorrQuery {
    /// Index of X column.
    pub x: usize,
    /// Index of Y column.
    pub y: usize,
    /// Start index into a shared flat conditioning-index buffer.
    pub z_start: usize,
    /// Number of conditioning columns.
    pub z_len: usize,
}

/// Pearson correlation of two equal-length slices (population formula).
///
/// `None` for fewer than two points, a non-finite column, or an effectively
/// constant column (see `constant_column`).
#[must_use]
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    pearson_floored(x, y, None)
}

/// [`pearson`] with optional explicit constant-column floors on the centered sums of
/// squares of `x` and `y`. `None` judges each column against its own magnitude.
fn pearson_floored(x: &[f64], y: &[f64], floors: Option<(f64, f64)>) -> Option<f64> {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..n {
        mx += x[i];
        my += y[i];
    }
    mx /= nf;
    my /= nf;
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    let (floor_x, floor_y) = floors.unwrap_or((raw_floor(mx, nf), raw_floor(my, nf)));
    if constant_column(cxx, floor_x) || constant_column(cyy, floor_y) {
        return None;
    }
    Some(cxy / (cxx * cyy).sqrt())
}

/// Centered sum of squares below which a residual is rounding noise, as a fraction of
/// the raw column's own centered sum of squares (a relative sd of `1e-10`).
const RESIDUAL_REL_TOL: f64 = 1e-10;

/// Floor for a raw column: its centered sum of squares must exceed what representing
/// the mean alone can leave behind (`n * (eps * mean)^2`). Purely relative, so the
/// verdict does not depend on the data's units.
fn raw_floor(mean: f64, nf: f64) -> f64 {
    nf * (f64::EPSILON * mean).powi(2)
}

/// Floor for the residual of `raw` after regression on Z: relative to the *raw*
/// column's centered sum of squares, because the residual's own mean is ~0 and says
/// nothing about the scale the rounding noise lives at. A raw column that is itself
/// constant or non-finite has no variation to explain: any residual is noise.
fn residual_floor(raw: &[f64]) -> f64 {
    let nf = raw.len() as f64;
    let mean = raw.iter().sum::<f64>() / nf;
    let css = raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
    if constant_column(css, raw_floor(mean, nf)) {
        f64::INFINITY
    } else {
        RESIDUAL_REL_TOL * RESIDUAL_REL_TOL * css
    }
}

/// Effectively-constant test on a centered sum of squares. Also true for a non-finite
/// sum, so callers that must tell noise from NaN check finiteness separately.
fn constant_column(css: f64, floor: f64) -> bool {
    !(css.is_finite() && css > floor)
}

/// Center each conditioning column and scale it to unit sum of squares.
///
/// Singularity is then a property of the correlation Gram, not of the column units.
/// A constant or non-finite column returns `false` so the caller refuses the test
/// instead of ridge-shrinking that column toward zero.
fn build_design(z_cols: &[&[f64]], n: usize, design: &mut [f64]) -> bool {
    for (j, z) in z_cols.iter().enumerate() {
        if !scale_column(z, n, &mut design[j * n..(j + 1) * n]) {
            return false;
        }
    }
    true
}

/// One column of [`build_design`]: centered and scaled to unit sum of squares into `out`.
fn scale_column(z: &[f64], n: usize, out: &mut [f64]) -> bool {
    let nf = n as f64;
    let mut mean = 0.0;
    for &v in z.iter().take(n) {
        if !v.is_finite() {
            return false;
        }
        mean += v;
    }
    mean /= nf;
    let mut css = 0.0;
    for r in 0..n {
        let d = z[r] - mean;
        out[r] = d;
        css += d * d;
    }
    if constant_column(css, raw_floor(mean, nf)) {
        return false;
    }
    let scale = css.sqrt();
    for v in out.iter_mut().take(n) {
        *v /= scale;
    }
    true
}

fn form_gram(design: &[f64], n: usize, ncols: usize, gram: &mut [f64]) {
    gram.fill(0.0);
    for c1 in 0..ncols {
        for c2 in c1..ncols {
            let mut acc = 0.0;
            let a = &design[c1 * n..(c1 + 1) * n];
            let b = &design[c2 * n..(c2 + 1) * n];
            for r in 0..n {
                acc += a[r] * b[r];
            }
            gram[c1 * ncols + c2] = acc;
            gram[c2 * ncols + c1] = acc;
        }
    }
}

fn form_xty(design: &[f64], y: &[f64], n: usize, ncols: usize, out: &mut [f64]) {
    let y_mean = y.iter().sum::<f64>() / n as f64;
    for c in 0..ncols {
        let col = &design[c * n..(c + 1) * n];
        out[c] = centered_dot(col, y, y_mean);
    }
}

fn centered_dot(col: &[f64], y: &[f64], y_mean: f64) -> f64 {
    let mut acc = 0.0;
    for r in 0..col.len() {
        acc += col[r] * (y[r] - y_mean);
    }
    acc
}

/// Gauss–Jordan with partial pivoting on `nrhs` right-hand sides stored back to back
/// (`ncols` each). Columns are already unit-scaled, so the pivot floor is relative to a
/// correlation Gram (diagonals are 1) and a mixed-scale conditioning column is not
/// declared singular.
///
/// Pivoting and the row operations depend on the Gram alone, so each right-hand side
/// sees exactly the operations a solve of that vector by itself would apply, and one
/// elimination serving X and Y returns bit-identical coefficients to two. A Cholesky
/// factorization would not: it rounds differently and its pivot floor (`64 eps`
/// Gauss-Jordan on a unit-scaled Gram.
///
/// The diagonal is already 1, so [`crate::gram::symmetric_scales`] would be the
/// identity. The pivot floor is [`crate::gram::EQUILIBRATED_PIVOT_FLOOR`] times
/// that diagonal: the same predicate [`crate::gram::invert_square`] applies
/// after equilibration.
fn solve_inplace(gram: &mut [f64], rhs: &mut [f64], ncols: usize, nrhs: usize) -> bool {
    crate::gram::gauss_jordan_solve(gram, rhs, ncols, nrhs)
}

fn residual_from_cols(y: &[f64], cols: &[&[f64]], beta: &[f64], out: &mut [f64]) {
    let n = y.len();
    let y_mean = y.iter().sum::<f64>() / n as f64;
    for r in 0..n {
        let mut pred = 0.0;
        for (col, b) in cols.iter().zip(beta) {
            pred += col[r] * b;
        }
        out[r] = (y[r] - y_mean) - pred;
    }
}

/// Partial correlation of X and Y given the Z columns: the design and its Gram are formed
/// once and a single elimination yields both coefficient vectors.
///
/// `empty_z` is the plain Pearson kernel used when there is nothing to condition on, and
/// `corr` the floored one applied to the residuals.
fn partial_correlation_impl(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
    empty_z: fn(&[f64], &[f64]) -> Option<f64>,
    corr: FlooredPearson,
) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let n = x.len();
    if z_cols.is_empty() {
        return empty_z(x, y);
    }
    for col in z_cols {
        if col.len() != n {
            return None;
        }
    }
    workspace.prepare(n, z_cols.len());
    let ncols = z_cols.len();
    let design = &mut workspace.design[..n * ncols];
    if !build_design(z_cols, n, design) {
        return None;
    }
    let gram = &mut workspace.gram[..ncols * ncols];
    let beta = &mut workspace.beta[..2 * ncols];
    form_gram(design, n, ncols, gram);
    form_xty(design, x, n, ncols, &mut beta[..ncols]);
    form_xty(design, y, n, ncols, &mut beta[ncols..]);
    if !solve_inplace(gram, beta, ncols, 2) {
        return None;
    }
    let mut cols: Vec<&[f64]> = Vec::with_capacity(ncols);
    cols.extend(design.chunks_exact(n));
    residual_from_cols(x, &cols, &beta[..ncols], &mut workspace.rx[..n]);
    residual_from_cols(y, &cols, &beta[ncols..], &mut workspace.ry[..n]);
    // Finite zero-variance residuals ⇒ trivial conditional independence (r = 0).
    // Non-finite residuals must stay None: `constant_column` also fires on NaN css,
    // so pearson alone cannot distinguish them from a real constant residual.
    pearson_after_residualize(&workspace.rx[..n], &workspace.ry[..n], x, y, corr)
}

/// A Pearson kernel taking optional per-column constant-column floors.
type FlooredPearson = fn(&[f64], &[f64], Option<(f64, f64)>) -> Option<f64>;

/// Pearson on residuals, restoring `Some(0.0)` only for finite constant residuals.
///
/// `constant_column` returns true for both non-finite css and a finite near-zero css,
/// so a bare `pearson(..).or(Some(0.0))` would turn NaN input into false independence.
/// Residuals are judged against the raw columns' scale (see [`residual_floor`]).
fn pearson_after_residualize(
    rx: &[f64],
    ry: &[f64],
    x: &[f64],
    y: &[f64],
    corr: FlooredPearson,
) -> Option<f64> {
    match corr(rx, ry, Some((residual_floor(x), residual_floor(y)))) {
        Some(r) => Some(r),
        None if rx.iter().chain(ry.iter()).all(|v| v.is_finite()) => Some(0.0),
        None => None,
    }
}

/// Same reduction as [`pearson`].
///
/// A chunked accumulator was kept here so the portable path could vectorise.
/// The partial sums are a different addition order from [`pearson`], so the two
/// paths disagreed in the last bits and neither was benchmarked as a replacement
/// for the other. One reduction is canonical.
fn pearson_fused(x: &[f64], y: &[f64]) -> Option<f64> {
    pearson(x, y)
}

fn pearson_fused_floored(x: &[f64], y: &[f64], floors: Option<(f64, f64)>) -> Option<f64> {
    pearson_floored(x, y, floors)
}

/// Cap on the retained scaled conditioning columns of one batch (`f64` count, 128 MiB).
const SHARED_POOL_LIMIT: usize = 1 << 24;

/// Where a conditioning column's scaled copy lives in the pool.
#[derive(Clone, Copy)]
enum ColumnSlot {
    Unseen,
    /// Constant, non-finite or otherwise refused by [`scale_column`].
    Refused,
    At(usize),
}

/// Outcome of answering a query from [`SharedDesign`].
enum Served {
    Answer(Option<f64>),
    /// The pool is full or the shapes are unusual; answer with the lone kernel.
    Lone,
}

/// What the queries of one batch share: scaled conditioning columns, their pairwise
/// Gram entries and their products with the X/Y targets.
///
/// Each stored value is produced by the loop a lone query would run on the same column
/// data, so a query answered from here is bit-identical to one answered alone.
struct SharedDesign {
    n: usize,
    pool: Vec<f64>,
    slot: Vec<ColumnSlot>,
    gram_pair: HashMap<(usize, usize), f64>,
    target_dot: HashMap<(usize, usize), f64>,
    target_mean: Vec<Option<f64>>,
}

impl SharedDesign {
    fn new(n: usize, columns: usize) -> Self {
        Self {
            n,
            pool: Vec::new(),
            slot: vec![ColumnSlot::Unseen; columns],
            gram_pair: HashMap::new(),
            target_dot: HashMap::new(),
            target_mean: vec![None; columns],
        }
    }

    /// Scaled copy of column `idx`; `Err(true)` when refused, `Err(false)` when the pool
    /// is full and the caller must answer the query without the cache.
    fn column(&mut self, columns: &[&[f64]], idx: usize) -> Result<usize, bool> {
        let slot = self.slot[idx];
        match slot {
            ColumnSlot::At(off) => Ok(off),
            ColumnSlot::Refused => Err(true),
            ColumnSlot::Unseen => {
                let off = self.pool.len();
                if off.saturating_add(self.n) > SHARED_POOL_LIMIT {
                    return Err(false);
                }
                self.pool.resize(off + self.n, 0.0);
                if scale_column(columns[idx], self.n, &mut self.pool[off..off + self.n]) {
                    self.slot[idx] = ColumnSlot::At(off);
                    Ok(off)
                } else {
                    self.pool.truncate(off);
                    self.slot[idx] = ColumnSlot::Refused;
                    Err(true)
                }
            }
        }
    }

    fn scaled(&self, off: usize) -> &[f64] {
        &self.pool[off..off + self.n]
    }

    fn gram_entry(&mut self, key: (usize, usize), offs: (usize, usize)) -> f64 {
        if let Some(&v) = self.gram_pair.get(&key) {
            return v;
        }
        let (a, b) = (self.scaled(offs.0), self.scaled(offs.1));
        let mut acc = 0.0;
        for r in 0..self.n {
            acc += a[r] * b[r];
        }
        self.gram_pair.insert(key, acc);
        acc
    }

    fn target_entry(&mut self, columns: &[&[f64]], idx: usize, target: usize, off: usize) -> f64 {
        if let Some(&v) = self.target_dot.get(&(idx, target)) {
            return v;
        }
        let y = columns[target];
        let mean =
            *self.target_mean[target].get_or_insert_with(|| y.iter().sum::<f64>() / self.n as f64);
        let acc = centered_dot(self.scaled(off), y, mean);
        self.target_dot.insert((idx, target), acc);
        acc
    }

    /// The partial correlation of one query from shared pieces, or [`Served::Lone`] when
    /// the query must take the lone path.
    fn query(
        &mut self,
        columns: &[&[f64]],
        q: &ParCorrQuery,
        z_flat: &[usize],
        workspace: &mut ParCorrWorkspace,
        corr: FlooredPearson,
    ) -> Served {
        let n = self.n;
        let (x, y) = (columns[q.x], columns[q.y]);
        let z_idx = &z_flat[q.z_start..q.z_start + q.z_len];
        let ncols = z_idx.len();
        if x.len() != n || y.len() != n || z_idx.iter().any(|&zi| columns[zi].len() != n) {
            return Served::Lone;
        }
        let mut offs = Vec::with_capacity(ncols);
        for &zi in z_idx {
            match self.column(columns, zi) {
                Ok(off) => offs.push(off),
                Err(true) => return Served::Answer(None),
                Err(false) => return Served::Lone,
            }
        }
        workspace.prepare_solve(n, ncols);
        for c1 in 0..ncols {
            for c2 in c1..ncols {
                let key = (z_idx[c1].min(z_idx[c2]), z_idx[c1].max(z_idx[c2]));
                let v = self.gram_entry(key, (offs[c1], offs[c2]));
                workspace.gram[c1 * ncols + c2] = v;
                workspace.gram[c2 * ncols + c1] = v;
            }
        }
        for c in 0..ncols {
            workspace.beta[c] = self.target_entry(columns, z_idx[c], q.x, offs[c]);
            workspace.beta[ncols + c] = self.target_entry(columns, z_idx[c], q.y, offs[c]);
        }
        let gram = &mut workspace.gram[..ncols * ncols];
        let beta = &mut workspace.beta[..2 * ncols];
        if !solve_inplace(gram, beta, ncols, 2) {
            return Served::Answer(None);
        }
        let cols: Vec<&[f64]> = offs.iter().map(|&o| self.scaled(o)).collect();
        residual_from_cols(x, &cols, &beta[..ncols], &mut workspace.rx[..n]);
        residual_from_cols(y, &cols, &beta[ncols..], &mut workspace.ry[..n]);
        Served::Answer(pearson_after_residualize(
            &workspace.rx[..n],
            &workspace.ry[..n],
            x,
            y,
            corr,
        ))
    }
}

/// Scalar reference partial correlation.
#[must_use]
pub(crate) fn partial_correlation_scalar(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
) -> Option<f64> {
    partial_correlation_impl(x, y, z_cols, workspace, pearson, pearson_floored)
}

/// Portable optimized partial correlation (shared Gram, fused Pearson).
#[must_use]
pub(crate) fn partial_correlation_portable(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
) -> Option<f64> {
    partial_correlation_impl(x, y, z_cols, workspace, pearson_fused, pearson_fused_floored)
}

/// Which reduction a batch selects.
///
/// [`Self::Native`] and [`Self::Portable`] both call the scalar Pearson
/// reduction. The names stay so a kernel policy can still request a mode; they
/// no longer select different arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParCorrMode {
    /// Canonical reduction.
    Native,
    /// Same reduction as [`Self::Native`].
    Portable,
}

impl ParCorrMode {
    /// Whether this selects the portable kernel.
    #[must_use]
    pub const fn is_portable(self) -> bool {
        matches!(self, Self::Portable)
    }
}

impl From<bool> for ParCorrMode {
    fn from(portable: bool) -> Self {
        if portable { Self::Portable } else { Self::Native }
    }
}

/// Run a batch of [`ParCorrQuery`] items against shared columns, in order.
///
/// `z_flat` holds the concatenated conditioning indexes. Repeated conditioning
/// columns in one batch are scaled once. [`ParCorrMode::Native`] and
/// [`ParCorrMode::Portable`] perform the same arithmetic.
///
/// # Errors
///
/// [`crate::view::ViewError::InvalidShape`] when `out.len() != queries.len()`.
pub fn partial_correlation_batch(
    columns: &[&[f64]],
    queries: &[ParCorrQuery],
    z_flat: &[usize],
    out: &mut [Option<f64>],
    workspace: &mut ParCorrWorkspace,
    mode: ParCorrMode,
) -> Result<(), crate::view::ViewError> {
    if out.len() != queries.len() {
        return Err(crate::view::ViewError::InvalidShape {
            message: "partial correlation out.len() must equal queries.len()",
        });
    }
    // Queries of one batch usually repeat conditioning columns; a lone query gains
    // nothing from the bookkeeping.
    let mut shared = (queries.len() > 1)
        .then(|| columns.first().map(|c| SharedDesign::new(c.len(), columns.len())))
        .flatten();
    let corr: FlooredPearson =
        if mode.is_portable() { pearson_fused_floored } else { pearson_floored };
    let mut z_bufs: Vec<&[f64]> = Vec::new();
    for (qi, q) in queries.iter().enumerate() {
        // Queries with nothing to condition on, or too few rows, never touch the pool.
        if let Some(sd) = shared.as_mut().filter(|sd| q.z_len > 0 && sd.n >= 3) {
            if let Served::Answer(r) = sd.query(columns, q, z_flat, workspace, corr) {
                out[qi] = r;
                continue;
            }
        }
        z_bufs.clear();
        let end = q.z_start + q.z_len;
        for &zi in &z_flat[q.z_start..end] {
            z_bufs.push(columns[zi]);
        }
        let r = if mode.is_portable() {
            partial_correlation_portable(columns[q.x], columns[q.y], &z_bufs, workspace)
        } else {
            partial_correlation_scalar(columns[q.x], columns[q.y], &z_bufs, workspace)
        };
        out[qi] = r;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_core::{KernelPolicy, ToleranceClass};

    use super::*;
    use crate::dispatch::{KernelImpl, select_impl};

    /// The solve this module used before the two right-hand sides shared one
    /// elimination: Gram formed and eliminated once for X and again for Y.
    fn two_solve_reference(
        x: &[f64],
        y: &[f64],
        z_cols: &[&[f64]],
        corr: FlooredPearson,
    ) -> Option<f64> {
        let n = x.len();
        let ncols = z_cols.len();
        let mut design = vec![0.0; n * ncols];
        if !build_design(z_cols, n, &mut design) {
            return None;
        }
        let mut residuals = Vec::new();
        for target in [x, y] {
            let mut gram = vec![0.0; ncols * ncols];
            let mut beta = vec![0.0; ncols];
            form_gram(&design, n, ncols, &mut gram);
            form_xty(&design, target, n, ncols, &mut beta);
            if !solve_inplace(&mut gram, &mut beta, ncols, 1) {
                return None;
            }
            let cols: Vec<&[f64]> = design.chunks_exact(n).collect();
            let mut out = vec![0.0; n];
            residual_from_cols(target, &cols, &beta, &mut out);
            residuals.push(out);
        }
        pearson_after_residualize(&residuals[0], &residuals[1], x, y, corr)
    }

    #[test]
    fn shared_elimination_and_batch_cache_are_bit_identical_to_two_solves() {
        let mut state = 0xA076_1D64_78BD_642F_u64;
        let mut unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let n = 60;
        let p = 7;
        let mut cols: Vec<Vec<f64>> = (0..p)
            .map(|j| {
                let scale = 10f64.powi(i32::try_from(j).unwrap() - 3);
                (0..n).map(|_| scale * unit() + 1e3 * scale).collect()
            })
            .collect();
        // Column 4 near-collinear with 3, column 5 non-finite, column 6 constant.
        for r in 0..n {
            cols[4][r] = cols[3][r] * 2.0 + 1e-9 * unit();
        }
        cols[5][7] = f64::NAN;
        cols[6] = vec![3.0; n];
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let mut queries = Vec::new();
        let mut z_flat = Vec::new();
        for x in 0..p {
            for y in 0..p {
                for z in [vec![], vec![2], vec![2, 3], vec![3, 4], vec![2, 5], vec![1, 6, 3]] {
                    if x == y || z.contains(&x) || z.contains(&y) {
                        continue;
                    }
                    let z_start = z_flat.len();
                    z_flat.extend(&z);
                    queries.push(ParCorrQuery { x, y, z_start, z_len: z.len() });
                }
            }
        }
        let mut ws = ParCorrWorkspace::default();
        for (mode, corr) in [
            (ParCorrMode::Portable, pearson_fused_floored as FlooredPearson),
            (ParCorrMode::Native, pearson_floored as FlooredPearson),
        ] {
            let mut out = vec![None; queries.len()];
            partial_correlation_batch(&refs, &queries, &z_flat, &mut out, &mut ws, mode).unwrap();
            let mut some = 0;
            for (q, got) in queries.iter().zip(&out) {
                let z: Vec<&[f64]> =
                    z_flat[q.z_start..q.z_start + q.z_len].iter().map(|&i| refs[i]).collect();
                let single = if mode.is_portable() {
                    partial_correlation_portable(refs[q.x], refs[q.y], &z, &mut ws)
                } else {
                    partial_correlation_scalar(refs[q.x], refs[q.y], &z, &mut ws)
                };
                let reference = if z.is_empty() {
                    if mode.is_portable() {
                        pearson_fused(refs[q.x], refs[q.y])
                    } else {
                        pearson(refs[q.x], refs[q.y])
                    }
                } else {
                    two_solve_reference(refs[q.x], refs[q.y], &z, corr)
                };
                assert_eq!(got.map(f64::to_bits), reference.map(f64::to_bits), "batch {q:?}");
                assert_eq!(single.map(f64::to_bits), reference.map(f64::to_bits), "single {q:?}");
                some += usize::from(reference.is_some());
            }
            assert!(some > 20 && some < queries.len(), "mix of answers and refusals: {some}");
        }
    }

    #[test]
    fn pearson_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        assert!(ToleranceClass::StableFloat.close(pearson(&x, &y).unwrap(), 1.0));
    }

    #[test]
    fn parcorr_removes_confounder() {
        let n = 200usize;
        let z: Vec<f64> = (0..n).map(|i| ((i as f64) - 99.5) / 50.0).collect();
        let x: Vec<f64> = (0..n).map(|i| z[i] + ((i % 3) as f64 - 1.0) * 0.1).collect();
        let y: Vec<f64> = (0..n).map(|i| z[i] + ((i % 5) as f64 - 2.0) * 0.1).collect();
        let mut ws = ParCorrWorkspace::default();
        let raw = pearson(&x, &y).unwrap();
        let partial = partial_correlation_scalar(&x, &y, &[&z], &mut ws).unwrap();
        assert!(raw > 0.9);
        assert!(partial.abs() < 0.2, "partial={partial}");
    }

    fn one_z_intercept_oracle(x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let mx = mean(x);
        let my = mean(y);
        let mz = mean(z);
        let z_ss = z.iter().map(|v| (v - mz).powi(2)).sum::<f64>();
        let bx = x.iter().zip(z).map(|(a, b)| (a - mx) * (b - mz)).sum::<f64>() / z_ss;
        let by = y.iter().zip(z).map(|(a, b)| (a - my) * (b - mz)).sum::<f64>() / z_ss;
        let rx: Vec<_> = x.iter().zip(z).map(|(a, b)| (a - mx) - bx * (b - mz)).collect();
        let ry: Vec<_> = y.iter().zip(z).map(|(a, b)| (a - my) - by * (b - mz)).collect();
        pearson(&rx, &ry).unwrap()
    }

    #[test]
    fn parcorr_is_translation_invariant_and_matches_intercept_oracle() {
        let x = [
            -0.427_007, -1.147_988, 1.561_491, -1.892_812, 1.154_907, 0.455_615, -1.952_965,
            0.823_206,
        ];
        let y = [
            -2.391_322, -2.619_455, -1.515_701, -3.279_834, -1.518_526, -2.738_805, -2.950_828,
            -2.147_493,
        ];
        let z = [
            -0.704_669, -1.396_603, 0.603_738, -1.710_255, 0.143_528, -0.537_244, -1.768_004,
            0.029_743,
        ];
        let z_shifted: Vec<_> = z.iter().map(|v| v + 100.0).collect();
        let oracle = one_z_intercept_oracle(&x, &y, &z);
        let mut ws_scalar = ParCorrWorkspace::default();
        let mut ws_portable = ParCorrWorkspace::default();
        let scalar = partial_correlation_scalar(&x, &y, &[&z], &mut ws_scalar).unwrap();
        let translated = partial_correlation_scalar(&x, &y, &[&z_shifted], &mut ws_scalar).unwrap();
        let portable =
            partial_correlation_portable(&x, &y, &[&z_shifted], &mut ws_portable).unwrap();
        assert!((scalar - translated).abs() <= 1e-12, "{scalar} vs {translated}");
        assert!((scalar - oracle).abs() <= 1e-12, "{scalar} vs {oracle}");
        assert!((portable - oracle).abs() <= 1e-12, "{portable} vs {oracle}");
    }

    #[test]
    fn parcorr_batch_is_translation_invariant() {
        let n = 100usize;
        let z: Vec<_> = (0..n).map(|i| (i as f64 * 0.17).sin() + 3.0).collect();
        let z_shifted: Vec<_> = z.iter().map(|v| v - 10_000.0).collect();
        let x: Vec<_> = (0..n).map(|i| 1.5 * z[i] + (i as f64 * 0.31).cos()).collect();
        let y: Vec<_> = (0..n).map(|i| -0.7 * z[i] + (i as f64 * 0.23).sin()).collect();
        let queries = [ParCorrQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let mut base = [None];
        let mut shifted = [None];
        let mut ws = ParCorrWorkspace::default();
        partial_correlation_batch(
            &[&x, &y, &z],
            &queries,
            &z_flat,
            &mut base,
            &mut ws,
            ParCorrMode::Native,
        )
        .unwrap();
        partial_correlation_batch(
            &[&x, &y, &z_shifted],
            &queries,
            &z_flat,
            &mut shifted,
            &mut ws,
            ParCorrMode::Portable,
        )
        .unwrap();
        assert!((base[0].unwrap() - shifted[0].unwrap()).abs() <= 1e-10);
    }

    #[test]
    fn parcorr_is_invariant_across_seeded_random_offsets() {
        let n = 96usize;
        let z: Vec<_> = (0..n).map(|i| (i as f64 * 0.19).sin() + 0.01 * i as f64).collect();
        let x: Vec<_> = (0..n).map(|i| 0.8 * z[i] + (i as f64 * 0.37).cos() - 2.0).collect();
        let y: Vec<_> = (0..n).map(|i| -1.2 * z[i] + (i as f64 * 0.29).sin() + 4.0).collect();
        let mut ws = ParCorrWorkspace::default();
        let reference = partial_correlation_scalar(&x, &y, &[&z], &mut ws).unwrap();
        let mut rng = antecedent_core::CausalRng::from_seed(0x5eed_cafe_f00d_beef);
        for _ in 0..100 {
            let unit = rng.next_f64();
            let offset = -10_000.0 + 20_000.0 * unit;
            let shifted: Vec<_> = z.iter().map(|value| value + offset).collect();
            let scalar = partial_correlation_scalar(&x, &y, &[&shifted], &mut ws).unwrap();
            let portable = partial_correlation_portable(&x, &y, &[&shifted], &mut ws).unwrap();
            assert!((scalar - reference).abs() <= 1e-10, "offset={offset}");
            assert!((portable - reference).abs() <= 1e-10, "offset={offset}");
        }
    }

    #[test]
    fn scalar_portable_differential() {
        let n = 128usize;
        let z: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| z[i] + 0.1 * (i as f64)).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * z[i] + 0.05 * (i as f64)).collect();
        let mut ws_s = ParCorrWorkspace::default();
        let mut ws_p = ParCorrWorkspace::default();
        let s = partial_correlation_scalar(&x, &y, &[&z], &mut ws_s).unwrap();
        let p = partial_correlation_portable(&x, &y, &[&z], &mut ws_p).unwrap();
        assert_eq!(s.to_bits(), p.to_bits());
    }

    #[test]
    fn batch_reuses_workspace() {
        let n = 64usize;
        let c0: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let c1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
        let c2: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let columns: [&[f64]; 3] = [&c0, &c1, &c2];
        let queries = [
            ParCorrQuery { x: 0, y: 1, z_start: 0, z_len: 1 },
            ParCorrQuery { x: 0, y: 2, z_start: 1, z_len: 0 },
        ];
        let z_flat = [2usize];
        let mut out = [None; 2];
        let mut ws = ParCorrWorkspace::default();
        partial_correlation_batch(
            &columns,
            &queries,
            &z_flat,
            &mut out,
            &mut ws,
            ParCorrMode::Native,
        )
        .unwrap();
        let cap_n = ws.capacity_n();
        let cap_p = ws.capacity_p();
        for _ in 0..20 {
            partial_correlation_batch(
                &columns,
                &queries,
                &z_flat,
                &mut out,
                &mut ws,
                ParCorrMode::Portable,
            )
            .unwrap();
            assert_eq!(ws.capacity_n(), cap_n);
            assert_eq!(ws.capacity_p(), cap_p);
        }
        assert!(out[0].is_some());
    }

    #[test]
    fn dispatch_force_scalar() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut ws = ParCorrWorkspace::default();
        let policy = KernelPolicy::scalar_only();
        assert_eq!(select_impl(&policy), KernelImpl::Scalar);
        let r = partial_correlation_scalar(&x, &y, &[], &mut ws).unwrap();
        assert!(ToleranceClass::StableFloat.close(r, 1.0));
    }

    /// Non-empty Z + a NaN in X/Y must not collapse to r = 0 (false independence);
    /// a finite exact residual (Z explains the variable) must still yield Some(0.0).
    #[test]
    fn parcorr_nan_with_nonempty_z_is_not_independence() {
        let z = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2];
        let mut x = [0.2, -0.1, 0.4, -0.3, 0.1, -0.4, 0.2, -0.1];
        x[3] = f64::NAN;
        let mut ws = ParCorrWorkspace::default();
        let scalar = partial_correlation_scalar(&x, &y, &[&z], &mut ws);
        let portable = partial_correlation_portable(&x, &y, &[&z], &mut ws);
        assert_ne!(scalar, Some(0.0), "scalar NaN-in-X must not report r=0");
        assert_ne!(portable, Some(0.0), "portable NaN-in-X must not report r=0");
        assert!(scalar.is_none() || !scalar.unwrap().is_finite());
        assert!(portable.is_none() || !portable.unwrap().is_finite());

        let mut y_nan = y;
        y_nan[1] = f64::NAN;
        let x_clean = [0.2, -0.1, 0.4, -0.3, 0.1, -0.4, 0.2, -0.1];
        let scalar_y = partial_correlation_scalar(&x_clean, &y_nan, &[&z], &mut ws);
        let portable_y = partial_correlation_portable(&x_clean, &y_nan, &[&z], &mut ws);
        assert_ne!(scalar_y, Some(0.0), "scalar NaN-in-Y must not report r=0");
        assert_ne!(portable_y, Some(0.0), "portable NaN-in-Y must not report r=0");

        // Y is an affine function of Z → zero Y residual, finite X residual leftover.
        let y_exact: Vec<f64> = z.iter().map(|&zi| 2.0 * zi + 1.0).collect();
        let x_var: Vec<f64> =
            z.iter().enumerate().map(|(i, &zi)| zi + ((i % 3) as f64 - 1.0) * 0.5).collect();
        assert_eq!(
            partial_correlation_scalar(&x_var, &y_exact, &[&z], &mut ws),
            Some(0.0),
            "scalar: finite exact Y residual ⇒ r=0"
        );
        assert_eq!(
            partial_correlation_portable(&x_var, &y_exact, &[&z], &mut ws),
            Some(0.0),
            "portable: finite exact Y residual ⇒ r=0"
        );

        // Both residuals constant and finite (X and Y both affine in Z).
        let x_exact: Vec<f64> = z.iter().map(|&zi| -0.5 * zi + 3.0).collect();
        assert_eq!(
            partial_correlation_scalar(&x_exact, &y_exact, &[&z], &mut ws),
            Some(0.0),
            "scalar: both residuals finite-constant ⇒ r=0"
        );
        assert_eq!(
            partial_correlation_portable(&x_exact, &y_exact, &[&z], &mut ws),
            Some(0.0),
            "portable: both residuals finite-constant ⇒ r=0"
        );

        // Clean finite independent sample still yields a finite correlation.
        let n = 64usize;
        let z_ok: Vec<f64> = (0..n).map(|i| (i as f64) * 0.17).collect();
        let x_ok: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64 - 3.0).collect();
        let y_ok: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 - 5.0).collect();
        let r = partial_correlation_scalar(&x_ok, &y_ok, &[&z_ok], &mut ws);
        assert!(r.is_some_and(f64::is_finite), "finite independent sample → finite r, got {r:?}");
    }

    type Columns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    fn confounded_columns(n: usize, scale: f64, noise: f64) -> Columns {
        let z1: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).sin()).collect();
        let z2: Vec<f64> = (0..n).map(|i| (0.81 * i as f64).cos()).collect();
        let e: Vec<f64> = (0..n).map(|i| (1.7 * i as f64 + 0.3).cos()).collect();
        let x = (0..n).map(|i| scale * (0.3 * z1[i] + 0.7 * z2[i] + noise * e[i])).collect();
        let y = (0..n).map(|i| scale * (z1[i] - z2[i] + noise * e[i])).collect();
        (z1, z2, x, y)
    }

    #[test]
    fn exactly_explained_column_is_independent_at_every_unit_scale() {
        // X is an exact linear function of (Z1, Z2): its residual is rounding noise whose
        // size scales with X. The verdict must not depend on the data's units.
        let mut ws = ParCorrWorkspace::default();
        let n = 200;
        for scale in [1.0, 1e3, 1e6] {
            let (z1, z2, x, _) = confounded_columns(n, scale, 0.0);
            let y: Vec<f64> = (0..n).map(|i| (2.3 * i as f64).sin() + 0.5 * z1[i]).collect();
            let scalar = partial_correlation_scalar(&x, &y, &[&z1, &z2], &mut ws);
            let portable = partial_correlation_portable(&x, &y, &[&z1, &z2], &mut ws);
            assert_eq!(scalar, Some(0.0), "scalar at scale {scale}");
            assert_eq!(portable, Some(0.0), "portable at scale {scale}");
        }
    }

    #[test]
    fn small_unit_residual_variation_is_not_mistaken_for_constant() {
        // Both residuals equal 1e-3 * scale * (e projected off Z), so the partial
        // correlation is exactly 1 in exact arithmetic. At scale 1e-14 that residual sd
        // (~1e-17) is far below any absolute epsilon, yet it is 1e-3 of the column's own
        // spread, so it is real variation and must be correlated, not reported as r = 0.
        let mut ws = ParCorrWorkspace::default();
        for scale in [1.0, 1e-14] {
            let (z1, z2, x, y) = confounded_columns(200, scale, 1e-3);
            let scalar = partial_correlation_scalar(&x, &y, &[&z1, &z2], &mut ws).unwrap();
            let portable = partial_correlation_portable(&x, &y, &[&z1, &z2], &mut ws).unwrap();
            assert!((scalar - 1.0).abs() < 1e-8, "scalar at scale {scale}: {scalar}");
            assert!((portable - 1.0).abs() < 1e-8, "portable at scale {scale}: {portable}");
        }
    }

    #[test]
    fn raw_columns_in_tiny_units_are_not_constant() {
        let x: Vec<f64> = (0..20).map(|i| 1e-20 * f64::from(i)).collect();
        let y: Vec<f64> = (0..20).map(|i| 1e-20 * f64::from(i * i)).collect();
        let r = pearson(&x, &y).expect("tiny-unit columns vary");
        let xs: Vec<f64> = (0..20).map(f64::from).collect();
        let ys: Vec<f64> = (0..20).map(|i| f64::from(i * i)).collect();
        let oracle = pearson(&xs, &ys).unwrap();
        assert!((r - oracle).abs() < 1e-12, "{r} vs {oracle}");
        assert!(pearson(&[3.0; 20], &y).is_none(), "a genuinely constant column stays None");
    }

    #[test]
    fn mixed_scale_conditioning_column_is_actually_partialled_out() {
        // A 0/1 flag next to a column whose sd is ~1e8 used to fail the Gram pivot
        // (tol = 1e-12 * max diagonal) and get ridge-shrunk to ~0, so the partial
        // correlation collapsed to the marginal correlation through the flag.
        let n = 160;
        let z_big: Vec<f64> = (0..n).map(|i| 1e8 * (0.17 * i as f64).sin()).collect();
        let z_flag: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let ex: Vec<f64> = (0..n).map(|i| (0.9 * i as f64).sin()).collect();
        let ey: Vec<f64> = (0..n).map(|i| (1.3 * i as f64).cos()).collect();
        let x: Vec<f64> = (0..n).map(|i| 4.0 * z_flag[i] + ex[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| 4.0 * z_flag[i] + ey[i]).collect();
        let mut ws = ParCorrWorkspace::default();
        let partial = partial_correlation_scalar(&x, &y, &[&z_big, &z_flag], &mut ws)
            .expect("mixed-scale Z is a valid design");
        let flag_only =
            partial_correlation_scalar(&x, &y, &[&z_flag], &mut ws).expect("flag-only design");
        let marginal = pearson(&x, &y).expect("marginal");
        assert!(
            (partial - flag_only).abs() < 1e-4,
            "partial {partial} should match conditioning on the flag alone ({flag_only}), not the marginal {marginal}"
        );
        assert!(
            marginal.abs() - partial.abs() > 0.2,
            "conditioning on the flag must move the correlation: marginal {marginal}, partial {partial}"
        );
        let portable = partial_correlation_portable(&x, &y, &[&z_big, &z_flag], &mut ws)
            .expect("portable mixed-scale");
        assert!((portable - partial).abs() < 1e-8, "{portable} vs {partial}");
    }

    #[test]
    fn collinear_conditioning_columns_are_refused() {
        let n = 40;
        let z1: Vec<f64> = (0..n).map(f64::from).collect();
        let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v + 3.0).collect();
        let x: Vec<f64> = (0..n).map(|i| f64::from(i).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| f64::from(i).cos()).collect();
        let mut ws = ParCorrWorkspace::default();
        assert!(partial_correlation_scalar(&x, &y, &[&z1, &z2], &mut ws).is_none());
        assert!(partial_correlation_portable(&x, &y, &[&z1, &z2], &mut ws).is_none());
    }
}

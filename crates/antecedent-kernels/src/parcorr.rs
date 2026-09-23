//! Partial-correlation kernels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

/// Scratch for residualization and Pearson correlation.
#[derive(Clone, Debug, Default)]
pub struct ParCorrWorkspace {
    /// Design matrix column-major `[Z…]` (`n * p`). Every column is centered, which is the
    /// implicit intercept: X and Y are residualized on `[1, Z…]`.
    pub design: Vec<f64>,
    /// `XtX` / Gram (`p^2`).
    pub gram: Vec<f64>,
    /// RHS / coefficients (`p`).
    pub beta: Vec<f64>,
    /// Residual of X.
    pub rx: Vec<f64>,
    /// Residual of Y.
    pub ry: Vec<f64>,
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
        let need_gram = ncols.saturating_mul(ncols);
        if self.gram.len() < need_gram {
            self.gram.resize(need_gram, 0.0);
        }
        if self.beta.len() < ncols {
            self.beta.resize(ncols, 0.0);
        }
        if self.rx.len() < n {
            self.rx.resize(n, 0.0);
        }
        if self.ry.len() < n {
            self.ry.resize(n, 0.0);
        }
        self.capacity_n = self.capacity_n.max(n);
        self.capacity_p = self.capacity_p.max(p);
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
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
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

fn build_design(z_cols: &[&[f64]], n: usize, design: &mut [f64]) {
    for (j, z) in z_cols.iter().enumerate() {
        let base = j * n;
        let mean = z.iter().sum::<f64>() / n as f64;
        for r in 0..n {
            design[base + r] = z[r] - mean;
        }
    }
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
        let mut acc = 0.0;
        let col = &design[c * n..(c + 1) * n];
        for r in 0..n {
            acc += col[r] * (y[r] - y_mean);
        }
        out[c] = acc;
    }
}

/// Gauss–Jordan with partial pivoting; singularity is judged relative to the Gram's
/// largest diagonal so the verdict does not depend on the data's units.
fn solve_inplace(gram: &mut [f64], rhs: &mut [f64], ncols: usize) -> bool {
    let mut scale = 0.0_f64;
    for d in 0..ncols {
        scale = scale.max(gram[d * ncols + d].abs());
    }
    if !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    let tol = 1e-12 * scale;
    for col in 0..ncols {
        let mut best_row = col;
        let mut best = gram[col * ncols + col].abs();
        for row in (col + 1)..ncols {
            let v = gram[row * ncols + col].abs();
            if v > best {
                best = v;
                best_row = row;
            }
        }
        if best <= tol {
            return false;
        }
        if best_row != col {
            for j in 0..ncols {
                gram.swap(col * ncols + j, best_row * ncols + j);
            }
            rhs.swap(col, best_row);
        }
        let pivot = gram[col * ncols + col];
        for j in 0..ncols {
            gram[col * ncols + j] /= pivot;
        }
        rhs[col] /= pivot;
        for row in 0..ncols {
            if row == col {
                continue;
            }
            let factor = gram[row * ncols + col];
            for j in 0..ncols {
                gram[row * ncols + j] -= factor * gram[col * ncols + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    true
}

/// Relative ridge added to the Gram diagonal when the plain solve reports a singular
/// system (exactly collinear conditioning columns). The regularized projection keeps
/// residualization well defined — matching least-squares-based reference stacks — with
/// an O(1e-8) relative perturbation.
const SINGULAR_RIDGE: f64 = 1e-8;

/// Solve the normal equations for `y` on `design`, retrying once with a scaled ridge
/// when the Gram is singular. Reforms `gram`/`rhs` internally.
fn solve_normal_equations(
    design: &[f64],
    y: &[f64],
    n: usize,
    ncols: usize,
    gram: &mut [f64],
    beta: &mut [f64],
) -> bool {
    form_gram(design, n, ncols, gram);
    form_xty(design, y, n, ncols, beta);
    if solve_inplace(gram, beta, ncols) {
        return true;
    }
    form_gram(design, n, ncols, gram);
    form_xty(design, y, n, ncols, beta);
    let mut scale = 0.0_f64;
    for d in 0..ncols {
        scale = scale.max(gram[d * ncols + d].abs());
    }
    if !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    for d in 0..ncols {
        gram[d * ncols + d] += SINGULAR_RIDGE * scale;
    }
    solve_inplace(gram, beta, ncols)
}

fn residualize_into_scalar(
    y: &[f64],
    z_cols: &[&[f64]],
    design: &mut [f64],
    gram: &mut [f64],
    beta: &mut [f64],
    out: &mut [f64],
) -> bool {
    let n = y.len();
    let p = z_cols.len();
    for col in z_cols {
        if col.len() != n {
            return false;
        }
    }
    let ncols = p;
    build_design(z_cols, n, design);
    if !solve_normal_equations(design, y, n, ncols, gram, beta) {
        return false;
    }
    let y_mean = y.iter().sum::<f64>() / n as f64;
    for r in 0..n {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += design[c * n + r] * beta[c];
        }
        out[r] = (y[r] - y_mean) - pred;
    }
    true
}

/// Scalar reference: residualize X and Y independently (correctness path).
fn partial_correlation_scalar_impl(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let n = x.len();
    if z_cols.is_empty() {
        return pearson(x, y);
    }
    workspace.prepare(n, z_cols.len());
    let ncols = z_cols.len();
    let design = &mut workspace.design[..n * ncols];
    let gram = &mut workspace.gram[..ncols * ncols];
    let beta = &mut workspace.beta[..ncols];
    let rx = &mut workspace.rx[..n];
    if !residualize_into_scalar(x, z_cols, design, gram, beta, rx) {
        return None;
    }
    let design = &mut workspace.design[..n * ncols];
    let gram = &mut workspace.gram[..ncols * ncols];
    let beta = &mut workspace.beta[..ncols];
    let ry = &mut workspace.ry[..n];
    if !residualize_into_scalar(y, z_cols, design, gram, beta, ry) {
        return None;
    }
    // Finite zero-variance residuals ⇒ trivial conditional independence (r = 0).
    // Non-finite residuals must stay None: `constant_column` also fires on NaN css,
    // so pearson alone cannot distinguish them from a real constant residual.
    pearson_after_residualize(&workspace.rx[..n], &workspace.ry[..n], x, y, pearson_floored)
}

/// Portable optimized path: design built once, Gram reformed once between X/Y
/// solves, fused Pearson on residuals (chunked contiguous loops).
fn partial_correlation_portable_impl(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let n = x.len();
    if z_cols.is_empty() {
        return pearson_fused(x, y);
    }
    for col in z_cols {
        if col.len() != n {
            return None;
        }
    }
    workspace.prepare(n, z_cols.len());
    let ncols = z_cols.len();
    {
        let design = &mut workspace.design[..n * ncols];
        build_design(z_cols, n, design);
        let gram = &mut workspace.gram[..ncols * ncols];
        let beta = &mut workspace.beta[..ncols];
        if !solve_normal_equations(design, x, n, ncols, gram, beta) {
            return None;
        }
        let rx = &mut workspace.rx[..n];
        residual_from_beta(x, design, beta, n, ncols, rx);
    }
    {
        let design = &mut workspace.design[..n * ncols];
        let gram = &mut workspace.gram[..ncols * ncols];
        let beta = &mut workspace.beta[..ncols];
        if !solve_normal_equations(design, y, n, ncols, gram, beta) {
            return None;
        }
        let ry = &mut workspace.ry[..n];
        residual_from_beta(y, design, beta, n, ncols, ry);
    }
    // See the scalar path: finite constant residual ⇒ Some(0.0); non-finite ⇒ None.
    pearson_after_residualize(&workspace.rx[..n], &workspace.ry[..n], x, y, pearson_fused_floored)
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

fn residual_from_beta(
    y: &[f64],
    design: &[f64],
    beta: &[f64],
    n: usize,
    ncols: usize,
    out: &mut [f64],
) {
    let y_mean = y.iter().sum::<f64>() / n as f64;
    for r in 0..n {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += design[c * n + r] * beta[c];
        }
        out[r] = (y[r] - y_mean) - pred;
    }
}

/// Fused two-pass Pearson favoring contiguous auto-vectorization.
fn pearson_fused(x: &[f64], y: &[f64]) -> Option<f64> {
    pearson_fused_floored(x, y, None)
}

fn pearson_fused_floored(x: &[f64], y: &[f64], floors: Option<(f64, f64)>) -> Option<f64> {
    const CHUNK: usize = 8;
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    let mut i = 0;
    while i + CHUNK <= n {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for k in 0..CHUNK {
            sx += x[i + k];
            sy += y[i + k];
        }
        mx += sx;
        my += sy;
        i += CHUNK;
    }
    while i < n {
        mx += x[i];
        my += y[i];
        i += 1;
    }
    mx /= nf;
    my /= nf;
    let (mut cxx, mut cyy, mut cxy) = (0.0, 0.0, 0.0);
    i = 0;
    while i + CHUNK <= n {
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        for k in 0..CHUNK {
            let dx = x[i + k] - mx;
            let dy = y[i + k] - my;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        cxx += sxx;
        cyy += syy;
        cxy += sxy;
        i += CHUNK;
    }
    while i < n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
        i += 1;
    }
    let (floor_x, floor_y) = floors.unwrap_or((raw_floor(mx, nf), raw_floor(my, nf)));
    if constant_column(cxx, floor_x) || constant_column(cyy, floor_y) {
        return None;
    }
    Some(cxy / (cxx * cyy).sqrt())
}

/// Scalar reference partial correlation.
#[must_use]
pub fn partial_correlation_scalar(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
) -> Option<f64> {
    partial_correlation_scalar_impl(x, y, z_cols, workspace)
}

/// Portable optimized partial correlation (shared Gram, fused Pearson).
#[must_use]
pub fn partial_correlation_portable(
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut ParCorrWorkspace,
) -> Option<f64> {
    partial_correlation_portable_impl(x, y, z_cols, workspace)
}

/// Run a batch of [`ParCorrQuery`] items against shared columns (deterministic order).
///
/// Kernel path for batch partial correlation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParCorrMode {
    /// Scalar reference implementation.
    Native,
    /// Portable-optimized path.
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

/// `z_flat` holds concatenated conditioning indexes for all queries.
///
/// # Panics
///
/// Panics if `out.len() != queries.len()`.
pub fn partial_correlation_batch(
    columns: &[&[f64]],
    queries: &[ParCorrQuery],
    z_flat: &[usize],
    out: &mut [Option<f64>],
    workspace: &mut ParCorrWorkspace,
    mode: ParCorrMode,
) {
    assert_eq!(out.len(), queries.len());
    let mut z_bufs: Vec<&[f64]> = Vec::new();
    for (qi, q) in queries.iter().enumerate() {
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
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use antecedent_core::{KernelPolicy, ToleranceClass};

    use super::*;
    use crate::dispatch::{KernelImpl, partial_correlation, select_impl};

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
        );
        partial_correlation_batch(
            &[&x, &y, &z_shifted],
            &queries,
            &z_flat,
            &mut shifted,
            &mut ws,
            ParCorrMode::Portable,
        );
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
        let mut state = 0x5eed_cafe_f00d_beefu64;
        for _ in 0..100 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
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
        assert!(ToleranceClass::StableFloat.close(s, p));
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
        );
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
            );
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
        let r = partial_correlation(&policy, &x, &y, &[], &mut ws).unwrap();
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
}

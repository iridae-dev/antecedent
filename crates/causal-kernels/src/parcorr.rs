//! Partial-correlation kernels (DESIGN.md §12).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::similar_names)]

/// Scratch for residualization and Pearson correlation.
#[derive(Clone, Debug, Default)]
pub struct ParCorrWorkspace {
    /// Design matrix column-major `[1 | Z…]` (`n * (1+p)`).
    pub design: Vec<f64>,
    /// `XtX` / Gram (`(1+p)^2`).
    pub gram: Vec<f64>,
    /// RHS / coefficients (`1+p`).
    pub beta: Vec<f64>,
    /// Residual of X.
    pub rx: Vec<f64>,
    /// Residual of Y.
    pub ry: Vec<f64>,
    capacity_n: usize,
    capacity_p: usize,
}

impl ParCorrWorkspace {
    /// Ensure capacity for `n` rows and `p` covariates (excluding intercept).
    pub fn prepare(&mut self, n: usize, p: usize) {
        let ncols = 1 + p;
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
#[must_use]
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
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
    if constant_column(cxx, mx, nf) || constant_column(cyy, my, nf) {
        return None;
    }
    Some(cxy / (cxx * cyy).sqrt())
}

/// Effectively-constant test on a centered sum of squares, relative to the column's
/// magnitude so the verdict does not depend on the data's units.
fn constant_column(css: f64, mean: f64, nf: f64) -> bool {
    let tol = nf * (f64::EPSILON * (1.0 + mean.abs())).powi(2);
    !(css.is_finite() && css > tol)
}

fn build_design(z_cols: &[&[f64]], n: usize, design: &mut [f64]) {
    for r in 0..n {
        design[r] = 1.0;
    }
    for (j, z) in z_cols.iter().enumerate() {
        let base = (j + 1) * n;
        design[base..base + n].copy_from_slice(z);
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
    for c in 0..ncols {
        let mut acc = 0.0;
        let col = &design[c * n..(c + 1) * n];
        for r in 0..n {
            acc += col[r] * y[r];
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
    let ncols = 1 + p;
    build_design(z_cols, n, design);
    if !solve_normal_equations(design, y, n, ncols, gram, beta) {
        return false;
    }
    for r in 0..n {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += design[c * n + r] * beta[c];
        }
        out[r] = y[r] - pred;
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
    let ncols = 1 + z_cols.len();
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
    // A zero-variance residual means Z explains that variable exactly: nothing is left
    // to correlate, so conditional independence holds trivially (r = 0) rather than the
    // statistic being an error.
    pearson(&workspace.rx[..n], &workspace.ry[..n]).or(Some(0.0))
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
    let ncols = 1 + z_cols.len();
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
    // See the scalar path: zero-variance residual ⇒ trivial conditional independence.
    pearson_fused(&workspace.rx[..n], &workspace.ry[..n]).or(Some(0.0))
}

fn residual_from_beta(
    y: &[f64],
    design: &[f64],
    beta: &[f64],
    n: usize,
    ncols: usize,
    out: &mut [f64],
) {
    for r in 0..n {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += design[c * n + r] * beta[c];
        }
        out[r] = y[r] - pred;
    }
}

/// Fused two-pass Pearson favoring contiguous auto-vectorization.
fn pearson_fused(x: &[f64], y: &[f64]) -> Option<f64> {
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
    if constant_column(cxx, mx, nf) || constant_column(cyy, my, nf) {
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
    portable: bool,
) {
    assert_eq!(out.len(), queries.len());
    let mut z_bufs: Vec<&[f64]> = Vec::new();
    for (qi, q) in queries.iter().enumerate() {
        z_bufs.clear();
        let end = q.z_start + q.z_len;
        for &zi in &z_flat[q.z_start..end] {
            z_bufs.push(columns[zi]);
        }
        let r = if portable {
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
    use causal_core::{KernelPolicy, ToleranceClass};

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
        // x = z + e1, y = z + e2 → raw corr high, partial ~0
        let n = 200usize;
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x: Vec<f64> = (0..n).map(|i| z[i] + (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| z[i] + (i % 5) as f64).collect();
        let mut ws = ParCorrWorkspace::default();
        let raw = pearson(&x, &y).unwrap();
        let partial = partial_correlation_scalar(&x, &y, &[&z], &mut ws).unwrap();
        assert!(raw > 0.9);
        assert!(partial.abs() < 0.2, "partial={partial}");
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
        partial_correlation_batch(&columns, &queries, &z_flat, &mut out, &mut ws, false);
        let cap_n = ws.capacity_n();
        let cap_p = ws.capacity_p();
        for _ in 0..20 {
            partial_correlation_batch(&columns, &queries, &z_flat, &mut out, &mut ws, true);
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
}

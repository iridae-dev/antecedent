//! NOTEARS loss, augmented Lagrangian, and native L-BFGS.
//!
//! Documented native equivalent of the NOTEARS L-BFGS-B + AL pipeline
//! (Zheng et al. 2018): free parameters are the off-diagonal, non-forbidden
//! entries (diagonal / forbidden fixed at 0 by packing, not box bounds).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use super::acyclicity::{AcyclicityWorkspace, grad_h, h_of_w};

/// Grow-only scratch for the NOTEARS continuous solver.
#[derive(Clone, Debug, Default)]
pub(crate) struct NotearsWorkspace {
    pub w: Vec<f64>,
    pub grad: Vec<f64>,
    pub xw: Vec<f64>,
    pub xtx: Vec<f64>,
    pub xtx_w: Vec<f64>,
    pub free: Vec<f64>,
    pub free_grad: Vec<f64>,
    pub free_idx: Vec<usize>,
    pub acyclicity: AcyclicityWorkspace,
    /// L-BFGS history: `s` and `y` rings (each entry length = `n_free`).
    pub s_hist: Vec<Vec<f64>>,
    pub y_hist: Vec<Vec<f64>>,
    pub rho_hist: Vec<f64>,
    pub q: Vec<f64>,
    pub r: Vec<f64>,
    pub free_prev: Vec<f64>,
    pub grad_prev: Vec<f64>,
    pub dir: Vec<f64>,
    /// Armijo / two-loop α coefficients (length `lbfgs_m`).
    pub alpha_buf: Vec<f64>,
}

impl NotearsWorkspace {
    pub(crate) fn prepare(&mut self, n: usize, d: usize, n_free: usize, m_hist: usize) {
        let d2 = d.saturating_mul(d);
        let nd = n.saturating_mul(d);
        grow(&mut self.w, d2);
        grow(&mut self.grad, d2);
        grow(&mut self.xw, nd);
        grow(&mut self.xtx, d2);
        grow(&mut self.xtx_w, d2);
        grow(&mut self.free, n_free);
        grow(&mut self.free_grad, n_free);
        grow(&mut self.q, n_free);
        grow(&mut self.r, n_free);
        grow(&mut self.free_prev, n_free);
        grow(&mut self.grad_prev, n_free);
        grow(&mut self.dir, n_free);
        grow(&mut self.alpha_buf, m_hist);
        self.acyclicity.prepare(d);
        if self.s_hist.len() < m_hist {
            self.s_hist.resize_with(m_hist, Vec::new);
            self.y_hist.resize_with(m_hist, Vec::new);
            self.rho_hist.resize(m_hist, 0.0);
        }
        for k in 0..m_hist {
            grow(&mut self.s_hist[k], n_free);
            grow(&mut self.y_hist[k], n_free);
        }
    }
}

fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Solver knobs (copied from [`super::Notears`] at run time).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SolverConfig {
    pub lambda: f64,
    pub max_iter: u32,
    pub h_tol: f64,
    pub rho_max: f64,
    pub lbfgs_max_iter: u32,
    pub lbfgs_m: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iter: 100,
            h_tol: 1e-8,
            rho_max: 1e16,
            lbfgs_max_iter: 200,
            lbfgs_m: 10,
        }
    }
}

/// Solve soft weighted adjacency `W` (row-major, \(W_{ij}\) = weight of edge \(i\to j\)).
///
/// `x` is column-major \(n\times d\) (column `j` occupies `x[j*n..(j+1)*n]`).
/// `frozen[i*d+j] == true` forces \(W_{ij}=0\) (diagonal and forbidden).
///
/// # Errors
///
/// Non-finite numerics or failure to drive \(|h(W)|\) below `h_tol`.
pub(crate) fn solve_notears(
    x: &[f64],
    n: usize,
    d: usize,
    frozen: &[bool],
    cfg: &SolverConfig,
    ws: &mut NotearsWorkspace,
) -> Result<Vec<f64>, &'static str> {
    if n == 0 || d == 0 {
        return Err("NOTEARS requires non-empty data");
    }
    if x.len() != n * d || frozen.len() != d * d {
        return Err("NOTEARS shape mismatch");
    }
    for v in x {
        if !v.is_finite() {
            return Err("NOTEARS data contains non-finite values");
        }
    }

    // Free indices: not frozen.
    ws.free_idx.clear();
    for i in 0..d {
        for j in 0..d {
            let idx = i * d + j;
            if !frozen[idx] {
                ws.free_idx.push(idx);
            }
        }
    }
    let n_free = ws.free_idx.len();
    ws.prepare(n, d, n_free, cfg.lbfgs_m);

    // Precompute G = X^T X / n  (d×d).
    form_xtx_over_n(x, n, d, &mut ws.xtx);

    // Init W = 0.
    for i in 0..d * d {
        ws.w[i] = 0.0;
    }
    unpack_free_from_w(ws);

    let mut alpha = 0.0;
    let mut rho = 1.0;
    // Match upstream NOTEARS: seed h at +∞ so the first ρ-adaptation
    // comparison `h_new > 0.25 * h` does not immediately escalate ρ.
    let mut h_val = f64::INFINITY;

    // Augmented Lagrangian (NOTEARS / Zheng et al. 2018): dual ascent on h(W)=0 with
    // ρ adaptation — re-solve while |h_new| > 0.25 |h_old|, else accept and update α.
    for _outer in 0..cfg.max_iter {
        let h_old = h_val;
        loop {
            lbfgs_minimize_al(x, n, d, cfg, alpha, rho, ws)?;
            h_val = h_of_w(&ws.w[..d * d], d, &mut ws.acyclicity)?;
            if !h_val.is_finite() {
                return Err("NOTEARS acyclicity h(W) non-finite");
            }
            if h_val.abs() <= 0.25 * h_old.abs().max(1e-30) || rho >= cfg.rho_max {
                break;
            }
            rho = (rho * 10.0).min(cfg.rho_max);
        }

        alpha += rho * h_val;

        if h_val.abs() <= cfg.h_tol {
            return Ok(ws.w[..d * d].to_vec());
        }
        if rho >= cfg.rho_max {
            return Err("NOTEARS failed to converge: |h(W)| above tolerance at rho_max");
        }
    }

    Err("NOTEARS failed to converge: max augmented-Lagrangian iterations")
}

fn unpack_free_from_w(ws: &mut NotearsWorkspace) {
    for (k, &idx) in ws.free_idx.iter().enumerate() {
        ws.free[k] = ws.w[idx];
    }
}

fn pack_free_into_w(ws: &mut NotearsWorkspace) {
    for (k, &idx) in ws.free_idx.iter().enumerate() {
        ws.w[idx] = ws.free[k];
    }
}

fn form_xtx_over_n(x: &[f64], n: usize, d: usize, xtx: &mut [f64]) {
    let inv_n = 1.0 / n as f64;
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            let ci = &x[i * n..(i + 1) * n];
            let cj = &x[j * n..(j + 1) * n];
            for r in 0..n {
                s += ci[r] * cj[r];
            }
            xtx[i * d + j] = s * inv_n;
        }
    }
}

/// AL objective and dense gradient on free params.
fn al_value_and_grad(
    x: &[f64],
    n: usize,
    d: usize,
    lambda: f64,
    alpha: f64,
    rho: f64,
    ws: &mut NotearsWorkspace,
) -> Result<f64, &'static str> {
    pack_free_into_w(ws);
    let d2 = d * d;

    // XW (column-major): (XW)[:,j] = sum_i X[:,i] * W[i,j]
    for j in 0..d {
        for r in 0..n {
            ws.xw[j * n + r] = 0.0;
        }
        for i in 0..d {
            let wij = ws.w[i * d + j];
            if wij == 0.0 {
                continue;
            }
            let col = &x[i * n..(i + 1) * n];
            for r in 0..n {
                ws.xw[j * n + r] += col[r] * wij;
            }
        }
    }

    // LS = (1/(2n)) ||X - XW||_F^2
    let mut ls = 0.0;
    for j in 0..d {
        let xj = &x[j * n..(j + 1) * n];
        let xwj = &ws.xw[j * n..(j + 1) * n];
        for r in 0..n {
            let e = xj[r] - xwj[r];
            ls += e * e;
        }
    }
    ls *= 0.5 / n as f64;

    // L1
    let mut l1 = 0.0;
    for i in 0..d2 {
        l1 += ws.w[i].abs();
    }
    l1 *= lambda;

    let h = h_of_w(&ws.w[..d2], d, &mut ws.acyclicity)?;
    let al = ls + l1 + 0.5 * rho * h * h + alpha * h;
    if !al.is_finite() {
        return Err("NOTEARS augmented-Lagrangian objective non-finite");
    }

    // ∇ LS = (1/n) X^T (XW - X) = G W - G  where G = X^T X / n
    // xtx_w = G W
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..d {
                s += ws.xtx[i * d + k] * ws.w[k * d + j];
            }
            ws.xtx_w[i * d + j] = s;
        }
    }
    for i in 0..d2 {
        ws.grad[i] = ws.xtx_w[i] - ws.xtx[i];
    }

    // AL: (ρ h + α) ∇h  — added before the L1 term so the subgradient below sees the full
    // smooth gradient at each coordinate.
    grad_h(&ws.w[..d2], d, &mut ws.acyclicity);
    let coeff = rho * h + alpha;
    for i in 0..d2 {
        ws.grad[i] += coeff * ws.acyclicity.grad[i];
    }

    // L1 subgradient, with the |w| kink at 0 handled properly.
    //
    // `w.signum()` is +1 at `+0.0` (IEEE), so the previous `grad += lambda * w.signum()`
    // pushed every zero coordinate in the −λ direction regardless of what the smooth part
    // wanted. `W` starts at all zeros, so on the first inner solve any coordinate with
    // `0 < g < λ` — exactly those whose optimal weight *is* 0 — was driven off zero, which
    // undermines the sparsity the L1 penalty exists to produce.
    //
    // The subdifferential of `λ|w|` at 0 is the interval `[−λ, λ]`, so the whole objective's
    // subdifferential there is `[g − λ, g + λ]`. Take its minimum-norm element: that is 0
    // whenever `|g| ≤ λ` (the coordinate is already stationary and must stay at zero), and
    // the soft-thresholded value otherwise.
    for i in 0..d2 {
        let w = ws.w[i];
        if w > 0.0 {
            ws.grad[i] += lambda;
        } else if w < 0.0 {
            ws.grad[i] -= lambda;
        } else {
            let g = ws.grad[i];
            ws.grad[i] = if g > lambda {
                g - lambda
            } else if g < -lambda {
                g + lambda
            } else {
                0.0
            };
        }
    }

    for i in 0..d2 {
        if !ws.grad[i].is_finite() {
            return Err("NOTEARS gradient non-finite");
        }
    }

    for (k, &idx) in ws.free_idx.iter().enumerate() {
        ws.free_grad[k] = ws.grad[idx];
    }
    Ok(al)
}

fn lbfgs_minimize_al(
    x: &[f64],
    n: usize,
    d: usize,
    cfg: &SolverConfig,
    alpha: f64,
    rho: f64,
    ws: &mut NotearsWorkspace,
) -> Result<(), &'static str> {
    let n_free = ws.free_idx.len();
    if n_free == 0 {
        return Ok(());
    }
    unpack_free_from_w(ws);

    let mut f = al_value_and_grad(x, n, d, cfg.lambda, alpha, rho, ws)?;
    let m = cfg.lbfgs_m;
    let mut hist_len = 0usize;
    let mut hist_pos = 0usize;

    for _iter in 0..cfg.lbfgs_max_iter {
        let gnorm = inf_norm(&ws.free_grad[..n_free]);
        if gnorm < 1e-8 {
            pack_free_into_w(ws);
            return Ok(());
        }

        // Two-loop L-BFGS → direction in ws.dir (newest history at hist_pos-1).
        ws.q[..n_free].copy_from_slice(&ws.free_grad[..n_free]);
        for i in 0..hist_len {
            let j = (hist_pos + m - 1 - i) % m;
            let a = ws.rho_hist[j] * dot(&ws.s_hist[j][..n_free], &ws.q[..n_free]);
            ws.alpha_buf[i] = a;
            for t in 0..n_free {
                ws.q[t] -= a * ws.y_hist[j][t];
            }
        }
        let mut scale = 1.0;
        if hist_len > 0 {
            let last = (hist_pos + m - 1) % m;
            let ys = dot(&ws.y_hist[last][..n_free], &ws.s_hist[last][..n_free]);
            let yy = dot(&ws.y_hist[last][..n_free], &ws.y_hist[last][..n_free]);
            if yy > 1e-16 {
                scale = ys / yy;
            }
        }
        for t in 0..n_free {
            ws.r[t] = scale * ws.q[t];
        }
        for i in (0..hist_len).rev() {
            let j = (hist_pos + m - 1 - i) % m;
            let a = ws.alpha_buf[i];
            let b = ws.rho_hist[j] * dot(&ws.y_hist[j][..n_free], &ws.r[..n_free]);
            for t in 0..n_free {
                ws.r[t] += ws.s_hist[j][t] * (a - b);
            }
        }
        for t in 0..n_free {
            ws.dir[t] = -ws.r[t];
        }

        // Armijo backtracking
        ws.free_prev[..n_free].copy_from_slice(&ws.free[..n_free]);
        ws.grad_prev[..n_free].copy_from_slice(&ws.free_grad[..n_free]);
        let gtd = dot(&ws.free_grad[..n_free], &ws.dir[..n_free]);
        if gtd >= 0.0 {
            // Not a descent direction — fall back to steepest descent.
            for t in 0..n_free {
                ws.dir[t] = -ws.free_grad[t];
            }
        }
        let gtd = dot(&ws.free_grad[..n_free], &ws.dir[..n_free]);
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            for t in 0..n_free {
                ws.free[t] = ws.free_prev[t] + step * ws.dir[t];
            }
            match al_value_and_grad(x, n, d, cfg.lambda, alpha, rho, ws) {
                Ok(f_new) => {
                    if f_new <= f + 1e-4 * step * gtd {
                        // Update L-BFGS history
                        for t in 0..n_free {
                            ws.s_hist[hist_pos][t] = ws.free[t] - ws.free_prev[t];
                            ws.y_hist[hist_pos][t] = ws.free_grad[t] - ws.grad_prev[t];
                        }
                        let ys =
                            dot(&ws.y_hist[hist_pos][..n_free], &ws.s_hist[hist_pos][..n_free]);
                        if ys > 1e-16 {
                            ws.rho_hist[hist_pos] = 1.0 / ys;
                            hist_pos = (hist_pos + 1) % m;
                            if hist_len < m {
                                hist_len += 1;
                            }
                        }
                        f = f_new;
                        accepted = true;
                        break;
                    }
                }
                Err(e) => return Err(e),
            }
            step *= 0.5;
        }
        if !accepted {
            // Restore and stop inner solve.
            ws.free[..n_free].copy_from_slice(&ws.free_prev[..n_free]);
            pack_free_into_w(ws);
            let _ = al_value_and_grad(x, n, d, cfg.lambda, alpha, rho, ws)?;
            return Ok(());
        }
    }
    pack_free_into_w(ws);
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At `W = 0` the L1 term must not push coordinates the penalty should hold at zero.
    ///
    /// The subdifferential of `λ|w|` at 0 is `[−λ, λ]`, so the objective's subdifferential
    /// there is `[g − λ, g + λ]` for smooth gradient `g`. Its minimum-norm element is 0
    /// whenever `|g| ≤ λ` — the coordinate is stationary and belongs at zero. The previous
    /// code used `w.signum()`, which IEEE defines as `+1` at `+0.0`, so it reported `g + λ`
    /// for *every* zero coordinate regardless of `g`: a nonzero gradient claiming a
    /// stationary point is not.
    ///
    /// Asserted on the gradient directly rather than through `solve_notears`, because
    /// L-BFGS backtracking rejects steps that raise the objective and can mask the wrong
    /// gradient in the fitted weights.
    #[test]
    fn l1_subgradient_is_zero_for_stationary_zero_coordinates() {
        // Two columns with a small but nonzero cross-product, so 0 < |G_01| < λ.
        let n = 4;
        let d = 2;
        // Column-major: col0 = x, col1 = y.
        let x = vec![1.0, -1.0, 1.0, -1.0, 0.2, -0.2, 0.2, -0.2];
        let frozen = [true, false, false, true];
        let mut ws = NotearsWorkspace::default();
        ws.free_idx.clear();
        for (idx, &f) in frozen.iter().enumerate() {
            if !f {
                ws.free_idx.push(idx);
            }
        }
        ws.prepare(n, d, ws.free_idx.len(), 8);
        form_xtx_over_n(&x, n, d, &mut ws.xtx);
        let g01 = ws.xtx[1].abs();
        assert!(g01 > 0.0, "fixture must have a nonzero cross-product");

        // λ strictly above |G_01| ⇒ both off-diagonals are stationary at zero.
        let lambda = g01 + 0.5;
        for v in &mut ws.w {
            *v = 0.0;
        }
        for v in &mut ws.free {
            *v = 0.0;
        }
        al_value_and_grad(&x, n, d, lambda, 0.0, 1.0, &mut ws).unwrap();
        for &idx in &[1usize, 2] {
            assert_eq!(
                ws.grad[idx], 0.0,
                "grad[{idx}] = {} at W=0 with |G|={g01} < λ={lambda}; a stationary zero \
                 coordinate must report zero gradient",
                ws.grad[idx]
            );
        }

        // λ strictly below |G_01| ⇒ the coordinate should move, soft-thresholded.
        let lambda = g01 / 2.0;
        for v in &mut ws.w {
            *v = 0.0;
        }
        for v in &mut ws.free {
            *v = 0.0;
        }
        al_value_and_grad(&x, n, d, lambda, 0.0, 1.0, &mut ws).unwrap();
        assert!(
            ws.grad[1].abs() > 0.0,
            "a coordinate whose smooth gradient exceeds λ must not be pinned at zero"
        );
        assert!(
            (ws.grad[1].abs() - (g01 - lambda)).abs() < 1e-12,
            "expected soft-thresholded magnitude {}, got {}",
            g01 - lambda,
            ws.grad[1].abs()
        );
    }

    #[test]
    fn solve_zero_dim_rejected() {
        let mut ws = NotearsWorkspace::default();
        let err = solve_notears(&[], 0, 0, &[], &SolverConfig::default(), &mut ws);
        assert!(err.is_err());
    }

    #[test]
    fn nan_data_fail_closed() {
        let n = 10;
        let d = 2;
        let mut x = vec![0.0; n * d];
        x[0] = f64::NAN;
        let frozen = [true, false, false, true];
        let mut ws = NotearsWorkspace::default();
        let err = solve_notears(&x, n, d, &frozen, &SolverConfig::default(), &mut ws);
        assert!(err.is_err());
    }
}

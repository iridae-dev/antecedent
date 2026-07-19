//! Smooth exact acyclicity constraint \(h(W)=\operatorname{tr}(e^{W\circ W})-d\).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]

/// Scratch buffers for \(h(W)\) / \(\nabla h\) (grow-only).
#[derive(Clone, Debug, Default)]
pub(crate) struct AcyclicityWorkspace {
    pub hadamard: Vec<f64>,
    pub expm: Vec<f64>,
    pub grad: Vec<f64>,
    /// Temps for scaling-and-squaring Taylor expm.
    pub a: Vec<f64>,
    pub t0: Vec<f64>,
    pub t1: Vec<f64>,
}

impl AcyclicityWorkspace {
    pub(crate) fn prepare(&mut self, d: usize) {
        let n2 = d.saturating_mul(d);
        grow(&mut self.hadamard, n2);
        grow(&mut self.expm, n2);
        grow(&mut self.grad, n2);
        grow(&mut self.a, n2);
        grow(&mut self.t0, n2);
        grow(&mut self.t1, n2);
    }
}

fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// \(h(W)=\operatorname{tr}(\exp(W\circ W))-d\).
///
/// Writes `expm(W∘W)` into `ws.expm` for reuse by [`grad_h`].
pub(crate) fn h_of_w(w: &[f64], d: usize, ws: &mut AcyclicityWorkspace) -> Result<f64, &'static str> {
    ws.prepare(d);
    let n2 = d * d;
    for i in 0..n2 {
        let v = w[i] * w[i];
        if !v.is_finite() {
            return Err("non-finite W∘W entry in acyclicity constraint");
        }
        ws.hadamard[i] = v;
    }
    // Copy into `a` first so `matrix_exp` can borrow `ws` mutably.
    ws.a[..n2].copy_from_slice(&ws.hadamard[..n2]);
    matrix_exp_inplace(d, ws)?;
    let mut tr = 0.0;
    for i in 0..d {
        tr += ws.expm[i * d + i];
    }
    if !tr.is_finite() {
        return Err("non-finite trace(expm(W∘W))");
    }
    Ok(tr - d as f64)
}

/// \(\nabla h = \exp(W\circ W)^\top \circ 2W\).
///
/// Requires [`h_of_w`] (or equivalent) to have filled `ws.expm` for the same `W`.
pub(crate) fn grad_h(w: &[f64], d: usize, ws: &mut AcyclicityWorkspace) {
    // expm^T ∘ 2W
    for i in 0..d {
        for j in 0..d {
            let e_ji = ws.expm[j * d + i];
            ws.grad[i * d + j] = e_ji * 2.0 * w[i * d + j];
        }
    }
}

/// Scaling-and-squaring with truncated Taylor series into `ws.expm`.
/// Input matrix is already in `ws.a`.
fn matrix_exp_inplace(d: usize, ws: &mut AcyclicityWorkspace) -> Result<(), &'static str> {
    let n2 = d * d;

    // 1-norm for scaling.
    let mut norm1 = 0.0;
    for j in 0..d {
        let mut col = 0.0;
        for i in 0..d {
            col += ws.a[i * d + j].abs();
        }
        if col > norm1 {
            norm1 = col;
        }
    }
    if !norm1.is_finite() {
        return Err("non-finite matrix 1-norm in expm");
    }

    // Scale so ||A/2^s||_1 <= 1.
    let mut s = 0u32;
    let mut scaled_norm = norm1;
    while scaled_norm > 1.0 && s < 40 {
        scaled_norm *= 0.5;
        s += 1;
    }
    let scale = 2.0_f64.powi(-i32::try_from(s).unwrap_or(40));
    for i in 0..n2 {
        ws.a[i] *= scale;
    }

    // Taylor: exp(A) = Σ_{k=0}^{K} A^k / k!
    // t0 = current power A^k / k!, accumulate into expm.
    const TAYLOR_TERMS: u32 = 20;
    for i in 0..n2 {
        ws.expm[i] = 0.0;
        ws.t0[i] = 0.0;
    }
    for i in 0..d {
        ws.expm[i * d + i] = 1.0;
        ws.t0[i * d + i] = 1.0;
    }
    for k in 1..=TAYLOR_TERMS {
        // t1 = t0 * A
        matmul(&ws.t0[..n2], &ws.a[..n2], &mut ws.t1[..n2], d);
        let inv_k = 1.0 / f64::from(k);
        for i in 0..n2 {
            ws.t0[i] = ws.t1[i] * inv_k;
            ws.expm[i] += ws.t0[i];
        }
    }

    // Square s times.
    for _ in 0..s {
        matmul(&ws.expm[..n2], &ws.expm[..n2], &mut ws.t0[..n2], d);
        ws.expm[..n2].copy_from_slice(&ws.t0[..n2]);
    }

    for i in 0..n2 {
        if !ws.expm[i].is_finite() {
            return Err("non-finite expm entry");
        }
    }
    Ok(())
}

fn matmul(a: &[f64], b: &[f64], out: &mut [f64], d: usize) {
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..d {
                s += a[i * d + k] * b[k * d + j];
            }
            out[i * d + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_zero_at_zero() {
        let d = 3;
        let w = vec![0.0; d * d];
        let mut ws = AcyclicityWorkspace::default();
        let h = h_of_w(&w, d, &mut ws).unwrap();
        assert!(h.abs() < 1e-12, "h(0)={h}");
    }

    #[test]
    fn h_positive_on_two_cycle() {
        let d = 2;
        // W = [[0, a], [a, 0]] → 2-cycle when a≠0
        let a = 0.5;
        let w = [0.0, a, a, 0.0];
        let mut ws = AcyclicityWorkspace::default();
        let h = h_of_w(&w, d, &mut ws).unwrap();
        assert!(h > 0.0, "h(2-cycle)={h}");
    }

    #[test]
    fn grad_h_matches_finite_difference() {
        let d = 3;
        let mut w = vec![0.0; d * d];
        w[0 * d + 1] = 0.2;
        w[1 * d + 2] = -0.15;
        w[2 * d + 0] = 0.1;
        // zero diagonal already
        let mut ws = AcyclicityWorkspace::default();
        let h0 = h_of_w(&w, d, &mut ws).unwrap();
        grad_h(&w, d, &mut ws);
        let analytic = ws.grad[..d * d].to_vec();

        let eps = 1e-6;
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    continue;
                }
                let idx = i * d + j;
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp[idx] += eps;
                wm[idx] -= eps;
                let hp = h_of_w(&wp, d, &mut ws).unwrap();
                let hm = h_of_w(&wm, d, &mut ws).unwrap();
                let fd = (hp - hm) / (2.0 * eps);
                let err = (analytic[idx] - fd).abs();
                assert!(
                    err < 1e-4,
                    "grad mismatch at ({i},{j}): analytic={} fd={fd} h0={h0} err={err}",
                    analytic[idx]
                );
            }
        }
    }

    #[test]
    fn expm_identity() {
        let d = 2;
        let a = [0.0; 4];
        let mut ws = AcyclicityWorkspace::default();
        ws.prepare(d);
        ws.a[..4].copy_from_slice(&a);
        matrix_exp_inplace(d, &mut ws).unwrap();
        assert!((ws.expm[0] - 1.0).abs() < 1e-10);
        assert!(ws.expm[1].abs() < 1e-10);
        assert!(ws.expm[2].abs() < 1e-10);
        assert!((ws.expm[3] - 1.0).abs() < 1e-10);
    }
}

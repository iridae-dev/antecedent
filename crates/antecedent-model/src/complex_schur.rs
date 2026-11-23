// SPDX-License-Identifier: MIT OR Apache-2.0
//! Complex Schur decomposition (Givens Hessenberg reduction, shifted QR) and the
//! Bartels–Stewart solve of `A X + X A^T = -Q`. Memory is `O(n^2)` and time `O(n^3)`, so
//! the size of a continuous-time system is bounded by the caller's resource admission, not
//! by a dense `n^2 x n^2` Kronecker operator.

#[derive(Clone, Copy, Debug)]
pub(crate) struct Cx {
    pub re: f64,
    pub im: f64,
}

impl Cx {
    pub(crate) const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub(crate) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn plus(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn minus(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    fn times(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn sqrt(self) -> Self {
        let r = self.abs();
        let re = ((r + self.re) * 0.5).max(0.0).sqrt();
        let im = ((r - self.re) * 0.5).max(0.0).sqrt();
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
}

/// `A = U T U^H` with `T` upper triangular, both row-major `n x n`.
pub(crate) struct Schur {
    pub t: Vec<Cx>,
    pub u: Vec<Cx>,
}

/// Apply the similarity by the Givens rotation `G = [[conj c, conj s], [-s, c]]` on
/// coordinates `(j, i)`: `T <- G T G^H`, `U <- U G^H`.
fn rotate(t: &mut [Cx], u: &mut [Cx], n: usize, j: usize, i: usize, c: Cx, s: Cx) {
    for k in 0..n {
        let (tj, ti) = (t[j * n + k], t[i * n + k]);
        t[j * n + k] = c.conj().times(tj).plus(s.conj().times(ti));
        t[i * n + k] = c.times(ti).minus(s.times(tj));
    }
    for m in [&mut *t, &mut *u] {
        for k in 0..n {
            let (rj, ri) = (m[k * n + j], m[k * n + i]);
            m[k * n + j] = rj.times(c).plus(ri.times(s));
            m[k * n + i] = ri.times(c.conj()).minus(rj.times(s.conj()));
        }
    }
}

fn givens(x: Cx, y: Cx) -> Option<(Cx, Cx)> {
    let r = (x.re * x.re + x.im * x.im + y.re * y.re + y.im * y.im).sqrt();
    (r > 0.0).then(|| (x.scale(1.0 / r), y.scale(1.0 / r)))
}

/// Complex Schur form of a real matrix; `None` when the bounded iteration does not converge.
pub(crate) fn schur(a: &[f64], n: usize) -> Option<Schur> {
    if n == 0 || a.len() != n * n || a.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut t: Vec<Cx> = a.iter().map(|v| Cx::new(*v, 0.0)).collect();
    let mut u = vec![Cx::ZERO; n * n];
    for i in 0..n {
        u[i * n + i] = Cx::new(1.0, 0.0);
    }
    let fro = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if fro == 0.0 {
        return Some(Schur { t, u });
    }
    // Hessenberg reduction by Givens similarity.
    for j in 0..n.saturating_sub(2) {
        for i in (j + 2..n).rev() {
            if let Some((c, s)) = givens(t[(i - 1) * n + j], t[i * n + j]) {
                rotate(&mut t, &mut u, n, i - 1, i, c, s);
                t[i * n + j] = Cx::ZERO;
            }
        }
    }
    let mut active = n;
    let mut iterations = 0_usize;
    while active > 1 {
        let sub = t[(active - 1) * n + active - 2].abs();
        let local = t[(active - 1) * n + active - 1].abs() + t[(active - 2) * n + active - 2].abs();
        if sub <= 1e-15 * local.max(fro * 1e-3) {
            t[(active - 1) * n + active - 2] = Cx::ZERO;
            active -= 1;
            iterations = 0;
            continue;
        }
        iterations += 1;
        if iterations > 500 {
            return None;
        }
        let a11 = t[(active - 2) * n + active - 2];
        let a12 = t[(active - 2) * n + active - 1];
        let a21 = t[(active - 1) * n + active - 2];
        let a22 = t[(active - 1) * n + active - 1];
        let half = a11.minus(a22).scale(0.5);
        let root = half.times(half).plus(a12.times(a21)).sqrt();
        let e1 = a22.plus(half).plus(root);
        let e2 = a22.plus(half).minus(root);
        let mut shift = if e1.minus(a22).abs() <= e2.minus(a22).abs() { e1 } else { e2 };
        if iterations % 11 == 10 {
            shift = a22.plus(Cx::new(sub, 0.5 * sub));
        }
        // The rotations of one shifted QR step come from the Hessenberg block B = T - shift I.
        let mut b: Vec<Cx> = (0..active * active)
            .map(|k| {
                let (i, j) = (k / active, k % active);
                let v = t[i * n + j];
                if i == j { v.minus(shift) } else { v }
            })
            .collect();
        let mut rotations = Vec::with_capacity(active);
        for j in 0..active - 1 {
            let Some((c, s)) = givens(b[j * active + j], b[(j + 1) * active + j]) else {
                continue;
            };
            for k in j..active {
                let (bj, bi) = (b[j * active + k], b[(j + 1) * active + k]);
                b[j * active + k] = c.conj().times(bj).plus(s.conj().times(bi));
                b[(j + 1) * active + k] = c.times(bi).minus(s.times(bj));
            }
            rotations.push((j, c, s));
        }
        for (j, c, s) in rotations {
            rotate(&mut t, &mut u, n, j, j + 1, c, s);
        }
    }
    for i in 1..n {
        for j in 0..i {
            t[i * n + j] = Cx::ZERO;
        }
    }
    Some(Schur { t, u })
}

fn complex_product(a: &[Cx], b: &[Cx], n: usize, conj_b: bool, conj_a_transpose: bool) -> Vec<Cx> {
    // Returns A' B' with A' = A or A^H and B' = B or conj-transposed B when requested.
    let mut out = vec![Cx::ZERO; n * n];
    for i in 0..n {
        for k in 0..n {
            let x = if conj_a_transpose { a[k * n + i].conj() } else { a[i * n + k] };
            if x.re == 0.0 && x.im == 0.0 {
                continue;
            }
            for j in 0..n {
                let y = if conj_b { b[j * n + k].conj() } else { b[k * n + j] };
                out[i * n + j] = out[i * n + j].plus(x.times(y));
            }
        }
    }
    out
}

/// Solve `A X + X A^T = -Q` for real `A` and symmetric `Q` (row-major, `n x n`).
///
/// `None` when the Schur iteration does not converge or when two eigenvalues satisfy
/// `lambda_i + lambda_j = 0` to working precision (no unique solution).
pub(crate) fn solve_lyapunov(a: &[f64], q: &[f64], n: usize) -> Option<Vec<f64>> {
    let Schur { t, u } = schur(a, n)?;
    let q: Vec<Cx> = q.iter().map(|v| Cx::new(*v, 0.0)).collect();
    // Qt = U^H Q U.
    let uq = complex_product(&u, &q, n, false, true);
    let qt = complex_product(&uq, &u, n, false, false);
    let scale = t.iter().map(|z| z.abs()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    let mut y = vec![Cx::ZERO; n * n];
    // T Y + Y T^H = -Qt, solved from the last row and column upwards.
    for i in (0..n).rev() {
        for j in (0..n).rev() {
            let mut rhs = Cx::new(-qt[i * n + j].re, -qt[i * n + j].im);
            for k in i + 1..n {
                rhs = rhs.minus(t[i * n + k].times(y[k * n + j]));
            }
            for k in j + 1..n {
                rhs = rhs.minus(y[i * n + k].times(t[j * n + k].conj()));
            }
            let pivot = t[i * n + i].plus(t[j * n + j].conj());
            if pivot.abs() <= 1e-14 * scale {
                return None;
            }
            y[i * n + j] = rhs.div(pivot);
        }
    }
    // X = U Y U^H, whose real part is the solution.
    let uy = complex_product(&u, &y, n, false, false);
    let x = complex_product(&uy, &u, n, true, false);
    let mut out: Vec<f64> = x.iter().map(|z| z.re).collect();
    for i in 0..n {
        for j in 0..i {
            let mean = out[i * n + j].midpoint(out[j * n + i]);
            out[i * n + j] = mean;
            out[j * n + i] = mean;
        }
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyapunov_symmetry_preserves_large_and_subnormal_finite_entries() {
        for (diagonal, off_diagonal) in
            [(0.9 * f64::MAX, 0.6 * f64::MAX), (f64::from_bits(4), f64::from_bits(2))]
        {
            let q = [diagonal, off_diagonal, off_diagonal, diagonal];
            let x = solve_lyapunov(&[-0.5, 0.0, 0.0, -0.5], &q, 2).unwrap();
            assert_eq!(x, q);
        }
    }

    #[test]
    fn schur_and_bartels_stewart_solve_known_lyapunov_equations() {
        // Scalar: 2 a x = -q.
        let x = solve_lyapunov(&[-2.0], &[4.0], 1).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14);
        // Oscillator with decay: A = [[-1, -3], [3, -1]] gives X = (q / 2) I for Q = q I.
        let x = solve_lyapunov(&[-1.0, -3.0, 3.0, -1.0], &[2.0, 0.0, 0.0, 2.0], 2).unwrap();
        for (found, want) in x.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((found - want).abs() < 1e-12);
        }
        // A larger system: residual of the dense equation.
        let n = 12;
        let a: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j { -2.0 } else { 0.3 * ((i * 7 + j * 3) % 5) as f64 / 5.0 - 0.1 }
            })
            .collect();
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0 + i as f64 * 0.1;
        }
        let x = solve_lyapunov(&a, &q, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let r: f64 = q[i * n + j]
                    + (0..n)
                        .map(|k| a[i * n + k] * x[k * n + j] + x[i * n + k] * a[j * n + k])
                        .sum::<f64>();
                assert!(r.abs() < 1e-10, "residual {r} at ({i},{j})");
            }
        }
        // lambda_i + lambda_j = 0 has no unique solution.
        assert!(solve_lyapunov(&[1.0, 0.0, 0.0, -1.0], &[1.0, 0.0, 0.0, 1.0], 2).is_none());
    }
}

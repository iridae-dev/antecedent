//! Univariate distance correlation in `O(n log n)` time and `O(n)` memory.
//!
//! Székely, Rizzo & Bakirov (2007) distance correlation with the V-statistic
//! normalisation, evaluated with the sorting / Fenwick-tree identities of Huo & Székely
//! (2016) instead of materialising the `n × n` distance matrices. With `a_ij = |x_i − x_j|`
//! and `b_ij = |y_i − y_j|`,
//!
//! `dCov² = S₁ + S₂ − 2 S₃`, where
//! `S₁ = n⁻² Σ_ij a_ij b_ij`, `S₂ = (n⁻² Σ a)(n⁻² Σ b)`, `S₃ = n⁻³ Σ_i (Σ_j a_ij)(Σ_j b_ij)`.
//!
//! Row sums come from one sort. `S₁` uses `|dx||dy| = dx·dy·sgn(dx)·sgn(dy)` and, scanning
//! points in increasing `x`, four Fenwick trees over the ranks of `y` that accumulate the
//! count, `x`, `y` and `x·y` of earlier points, so each point's concordance-signed sums
//! over all earlier points are two prefix queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Row sums `Σ_j |x_i − x_j|` for every `i`.
fn abs_diff_row_sums(x: &[f64], order: &[usize]) -> Vec<f64> {
    let n = x.len();
    let total: f64 = x.iter().sum();
    let mut out = vec![0.0; n];
    let mut prefix = 0.0;
    for (pos, &i) in order.iter().enumerate() {
        // `pos` points precede `i` in sorted order; ties contribute zero on either side.
        out[i] = x[i] * (2.0 * pos as f64 - n as f64) + total - 2.0 * prefix;
        prefix += x[i];
    }
    out
}

/// Fenwick tree over `[count, Σx, Σy, Σxy]`.
struct Fenwick {
    t: Vec<[f64; 4]>,
}

impl Fenwick {
    fn new(m: usize) -> Self {
        Self { t: vec![[0.0; 4]; m + 1] }
    }

    fn add(&mut self, mut k: usize, x: f64, y: f64) {
        let m = self.t.len() - 1;
        while k <= m {
            let node = &mut self.t[k];
            node[0] += 1.0;
            node[1] += x;
            node[2] += y;
            node[3] += x * y;
            k += k & k.wrapping_neg();
        }
    }

    fn prefix(&self, mut k: usize) -> [f64; 4] {
        let mut acc = [0.0; 4];
        while k > 0 {
            for (a, v) in acc.iter_mut().zip(&self.t[k]) {
                *a += v;
            }
            k -= k & k.wrapping_neg();
        }
        acc
    }
}

/// `Σ_{i,j} |x_i − x_j| |y_i − y_j|` over ordered pairs.
#[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
fn cross_abs_diff_sum(x: &[f64], y: &[f64], order_x: &[usize]) -> f64 {
    let n = x.len();
    // Dense ranks of `y` (1-based); equal values share a rank so ties count as neither
    // concordant nor discordant.
    let mut ys: Vec<f64> = y.to_vec();
    ys.sort_unstable_by(f64::total_cmp);
    ys.dedup();
    let rank_of = |v: f64| ys.partition_point(|&u| u < v) + 1;

    let mut tree = Fenwick::new(ys.len());
    let mut total = [0.0f64; 4];
    let mut acc = 0.0;
    let mut g = 0;
    while g < n {
        // Points with equal `x` are not ordered against each other: query the whole group
        // before inserting any of it.
        let mut h = g;
        while h < n && x[order_x[h]] == x[order_x[g]] {
            h += 1;
        }
        for &i in &order_x[g..h] {
            let r = rank_of(y[i]);
            let below = tree.prefix(r - 1);
            let upto = tree.prefix(r);
            let mut signed = [0.0; 4];
            for k in 0..4 {
                // Earlier points with smaller `y` are concordant (+), larger `y` discordant (−).
                signed[k] = below[k] - (total[k] - upto[k]);
            }
            let [c1, cx, cy, cxy] = signed;
            acc += x[i] * y[i] * c1 - x[i] * cy - y[i] * cx + cxy;
        }
        for &i in &order_x[g..h] {
            tree.add(rank_of(y[i]), x[i], y[i]);
            total[0] += 1.0;
            total[1] += x[i];
            total[2] += y[i];
            total[3] += x[i] * y[i];
        }
        g = h;
    }
    2.0 * acc
}

fn sorted_order(x: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));
    order
}

/// Squared distance covariance (V-statistic) of two equal-length series.
fn distance_covariance_sq(x: &[f64], y: &[f64], order_x: &[usize], order_y: &[usize]) -> f64 {
    let nf = x.len() as f64;
    let ax = abs_diff_row_sums(x, order_x);
    let ay = abs_diff_row_sums(y, order_y);
    let s1 = cross_abs_diff_sum(x, y, order_x) / (nf * nf);
    let s2 = (ax.iter().sum::<f64>() / (nf * nf)) * (ay.iter().sum::<f64>() / (nf * nf));
    let s3 = ax.iter().zip(&ay).map(|(a, b)| a * b).sum::<f64>() / (nf * nf * nf);
    s1 + s2 - 2.0 * s3
}

/// Székely distance correlation `sqrt(dCov²(x,y) / sqrt(dVar²(x) dVar²(y)))` in
/// `O(n log n)` time and `O(n)` memory.
///
/// Returns `0` for fewer than two observations, a length mismatch, or a constant series
/// (zero distance variance), and `NaN` if either series holds a non-finite value.
#[must_use]
pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n {
        return 0.0;
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let order_x = sorted_order(x);
    let order_y = sorted_order(y);
    let dxx = distance_covariance_sq(x, x, &order_x, &order_x);
    let dyy = distance_covariance_sq(y, y, &order_y, &order_y);
    if dxx <= 0.0 || dyy <= 0.0 {
        return 0.0;
    }
    let dxy = distance_covariance_sq(x, y, &order_x, &order_y);
    (dxy.max(0.0) / (dxx * dyy).sqrt()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Definition: double-centre both `n × n` L1 distance matrices.
    fn brute(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let centered = |s: &[f64]| {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = (s[i] - s[j]).abs();
                }
            }
            let row: Vec<f64> =
                (0..n).map(|i| (0..n).map(|j| a[i * n + j]).sum::<f64>() / n as f64).collect();
            let mean = row.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = a[i * n + j] - row[i] - row[j] + mean;
                }
            }
            a
        };
        let (a, b) = (centered(x), centered(y));
        let nn = (n * n) as f64;
        let dxy: f64 = a.iter().zip(&b).map(|(p, q)| p * q).sum::<f64>() / nn;
        let dxx: f64 = a.iter().map(|p| p * p).sum::<f64>() / nn;
        let dyy: f64 = b.iter().map(|q| q * q).sum::<f64>() / nn;
        (dxy.max(0.0) / (dxx * dyy).sqrt()).sqrt()
    }

    fn series(n: usize, seed: u64, ties: bool) -> (Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let x: Vec<f64> = (0..n)
            .map(|_| {
                let v = next() * 4.0;
                if ties { (v * 3.0).round() / 3.0 } else { v }
            })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|v| {
                let w = v * v + 0.3 * next();
                if ties { (w * 2.0).round() / 2.0 } else { w }
            })
            .collect();
        (x, y)
    }

    #[test]
    fn matches_the_double_centred_definition() {
        for (n, seed, ties) in
            [(5usize, 1u64, false), (17, 2, false), (40, 3, false), (17, 4, true), (40, 5, true)]
        {
            let (x, y) = series(n, seed, ties);
            let fast = distance_correlation(&x, &y);
            let slow = brute(&x, &y);
            assert!((fast - slow).abs() < 1e-12, "n={n} ties={ties}: fast={fast} brute={slow}");
        }
    }

    #[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
    #[test]
    fn self_correlation_is_one_and_constant_is_zero() {
        let (x, _) = series(30, 9, false);
        assert!((distance_correlation(&x, &x) - 1.0).abs() < 1e-12);
        assert_eq!(distance_correlation(&x, &vec![2.0; 30]), 0.0);
        assert!(distance_correlation(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0]).is_nan());
    }
}

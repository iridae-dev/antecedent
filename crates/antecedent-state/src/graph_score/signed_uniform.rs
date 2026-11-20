//! Marginalize a unit-variance, zero-intercept Gaussian likelihood over the disjoint
//! coefficient boxes [-1,-0.5] ∪ [0.5,1]. Their combined length is one in each dimension.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

use super::{GraphScoreData, StateError, compensated_sum};

const DIM: usize = 3;
type Point = [f64; DIM];
type Matrix = [Point; DIM];
// Absolute quadrature error divided by a lower bound on the whole integral.
const REL_TOL: f64 = 1e-8;

pub(super) fn raw_gram(data: &GraphScoreData) -> Vec<f64> {
    let n = data.n_vars;
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let dot = compensated_sum(data.col(i).iter().zip(data.col(j)).map(|(x, y)| x * y));
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
    }
    gram
}

struct Quadratic {
    dim: usize,
    h: Point,
    g: Matrix,
}

impl Quadratic {
    fn gradient(&self, x: Point) -> Point {
        std::array::from_fn(|i| self.h[i] - dot(self.g[i], x))
    }

    fn at(&self, x: Point) -> f64 {
        0.5 * (dot(self.h, x) + dot(self.gradient(x), x))
    }
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone)]
struct Region {
    lo: Point,
    hi: Point,
    anchor: Point,
    gradient: Point,
    peak: f64,
    // A tangent to the concave log integrand bounds it everywhere in the box.
    upper: f64,
    // Jensen's inequality on a neighborhood of the anchor bounds the mass from below.
    log_lower_mass: f64,
}

impl Region {
    fn new(q: &Quadratic, signs: usize) -> Self {
        let mut lo = [0.0; DIM];
        let mut hi = [0.0; DIM];
        let mut anchor = [0.0; DIM];
        for i in 0..q.dim {
            lo[i] = if signs & (1 << i) == 0 { -1.0 } else { 0.5 };
            hi[i] = lo[i] + 0.5;
            anchor[i] = lo[i] + 0.25;
        }
        // Projected coordinate ascent only sharpens the tangent bound. The bound
        // is valid at every iterate, including when the Gram is singular.
        for _ in 0..16 {
            let mut movement: f64 = 0.0;
            for i in 0..q.dim {
                if q.g[i][i] > 0.0 {
                    let next = (anchor[i] + (q.h[i] - dot(q.g[i], anchor)) / q.g[i][i])
                        .clamp(lo[i], hi[i]);
                    movement = movement.max((next - anchor[i]).abs());
                    anchor[i] = next;
                }
            }
            if movement < 1e-12 {
                break;
            }
        }
        let gradient = q.gradient(anchor);
        let peak = q.at(anchor);
        let mut upper = peak;
        let mut middle = anchor;
        let mut log_lower = 0.0;
        for i in 0..q.dim {
            let end = if gradient[i] > 0.0 { hi[i] } else { lo[i] };
            upper += gradient[i] * (end - anchor[i]);
            let radius = 1.0 / (1.0 + gradient[i].abs() + q.g[i][i].sqrt());
            let a = lo[i].max(anchor[i] - radius);
            let b = hi[i].min(anchor[i] + radius);
            middle[i] = (a + b) * 0.5;
            log_lower += (b - a).ln() - q.g[i][i] * (b - a).powi(2) / 24.0;
        }
        Self { lo, hi, anchor, gradient, peak, upper, log_lower_mass: log_lower + q.at(middle) }
    }

    fn log_volume(&self, dim: usize) -> f64 {
        (0..dim).map(|i| (self.hi[i] - self.lo[i]).ln()).sum()
    }
}

// Positive Gauss–Legendre weights normalized to sum to one on [0,1].
const ORDERS: [usize; 10] = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96];
fn rules() -> &'static [Vec<(f64, f64)>] {
    static RULES: OnceLock<Vec<Vec<(f64, f64)>>> = OnceLock::new();
    RULES.get_or_init(|| {
        ORDERS
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|j| {
                        let mut z = (PI * (j as f64 + 0.75) / (n as f64 + 0.5)).cos();
                        let mut derivative = 0.0;
                        for _ in 0..16 {
                            let (mut previous, mut current) = (1.0, z);
                            for k in 2..=n {
                                let k = k as f64;
                                (previous, current) = (
                                    current,
                                    ((2.0 * k - 1.0) * z * current - (k - 1.0) * previous) / k,
                                );
                            }
                            derivative = n as f64 * (previous - z * current) / (1.0 - z * z);
                            let step = current / derivative;
                            z -= step;
                            if step.abs() <= f64::EPSILON {
                                break;
                            }
                        }
                        ((1.0 + z) * 0.5, 1.0 / ((1.0 - z * z) * derivative * derivative))
                    })
                    .collect()
            })
            .collect()
    })
}

/// For each coordinate approximate the integrand by its Chebyshev series through
/// degree 2m-1. On the Bernstein ellipse ρ=exp(t), its tail in supremum norm is
/// ≤ 2M ρ^(1-2m)/(ρ-1). Both the exact uniform integral and the positive normalized
/// Gaussian rule have norm one, so the error on that axis is at most twice the tail.
/// Tensorizing telescopes these axis errors, without multiplying their constants.
fn rule_for_axis(
    q: &Quadratic,
    region: &Region,
    axis: usize,
    log_budget: f64,
) -> Result<usize, StateError> {
    let half = (region.hi[axis] - region.lo[axis]) * 0.5;
    let log_mass_bound = region.upper + region.log_volume(q.dim);
    let mut required = f64::INFINITY;
    for t in [0.125_f64, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] {
        // Moving the real argument beyond the box extends its tangent bound by
        // |gradient|·extension. The imaginary displacement adds g_ii·Im(z)^2/2
        // to the real part of the log likelihood.
        let log_m = log_mass_bound
            + region.gradient[axis].abs() * half * (t.cosh() - 1.0)
            + 0.5 * q.g[axis][axis] * (half * t.sinh()).powi(2);
        let order = 0.5 * (1.0 + (log_m - log_budget + (4.0 / t.exp_m1()).ln()) / t);
        required = required.min(order);
    }
    ORDERS.iter().position(|&n| n as f64 >= required).ok_or_else(|| {
        StateError::Numerical(
            "signed-uniform Gaussian quadrature exceeds its supported precision budget".into(),
        )
    })
}

fn log_integral(q: &Quadratic) -> Result<f64, StateError> {
    let count = 1 << q.dim;
    let mut regions: Vec<_> = (0..count).map(|signs| Region::new(q, signs)).collect();
    let lower = regions.iter().map(|b| b.log_lower_mass).fold(f64::NEG_INFINITY, f64::max);
    // Whole-box omission, each of d trimmed slabs, and d integrated axes share
    // the budget. All discarded mass and quadrature errors together are ≤ REL_TOL·Z.
    let log_budget = lower + (REL_TOL / (count * (1 + 2 * q.dim)) as f64).ln();
    let mut values = Vec::with_capacity(count);
    for r in &mut regions {
        let upper_mass = r.upper + r.log_volume(q.dim);
        if upper_mass <= log_budget {
            continue;
        }
        for i in 0..q.dim {
            // Relative to the maximum of the tangent, a cut distance L from
            // its favored endpoint loses at least |gradient_i|·L in log mass.
            let width = (upper_mass - log_budget) / r.gradient[i].abs();
            if r.gradient[i] > 0.0 {
                r.lo[i] = r.lo[i].max(r.hi[i] - width);
            }
            if r.gradient[i] < 0.0 {
                r.hi[i] = r.hi[i].min(r.lo[i] + width);
            }
        }
        let unit = [(0.0, 1.0)];
        let mut grids: [&[(f64, f64)]; DIM] = [&unit; DIM];
        for i in 0..q.dim {
            grids[i] = &rules()[rule_for_axis(q, r, i, log_budget)?];
        }
        let coord = |i: usize, z: f64| (r.lo[i] - r.anchor[i]) + (r.hi[i] - r.lo[i]) * z;
        let mut integral = 0.0;
        for &(u, wu) in grids[0] {
            let a = coord(0, u);
            let first = a * (r.gradient[0] - 0.5 * q.g[0][0] * a);
            for &(v, wv) in grids[1] {
                let b = coord(1, v);
                let second = first + b * (r.gradient[1] - q.g[1][0] * a - 0.5 * q.g[1][1] * b);
                let linear = r.gradient[2] - q.g[2][0] * a - q.g[2][1] * b;
                let mut inner = 0.0;
                for &(w, ww) in grids[2] {
                    let c = coord(2, w);
                    inner += ww * (second + c * (linear - 0.5 * q.g[2][2] * c)).exp();
                }
                integral += wu * wv * inner;
            }
        }
        values.push(r.peak + r.log_volume(q.dim) + integral.ln());
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln())
}

pub(super) fn score(
    data: &GraphScoreData,
    gram: &[f64],
    node: u32,
    parents: &[u32],
) -> Result<f64, StateError> {
    let node = node as usize;
    let n = data.n_vars;
    if node >= n
        || parents.len() > DIM
        || parents
            .iter()
            .enumerate()
            .any(|(i, &p)| p as usize >= n || p as usize == node || parents[..i].contains(&p))
    {
        return Err(StateError::Shape(
            "signed-uniform Gaussian score requires up to three distinct valid parents".into(),
        ));
    }
    let mut q = Quadratic { dim: parents.len(), h: [0.0; DIM], g: [[0.0; DIM]; DIM] };
    let yy = gram[node * n + node];
    let mut norm_sum = yy.sqrt();
    for (i, &a) in parents.iter().enumerate() {
        q.h[i] = gram[a as usize * n + node];
        norm_sum += gram[a as usize * n + a as usize].sqrt();
        for (j, &b) in parents.iter().enumerate() {
            q.g[i][j] = gram[a as usize * n + b as usize];
        }
    }
    // Compensated Gram sums, at most three differences/products in each quadratic
    // term. Fail closed at scales where f64 roundoff can rival the integration budget.
    let magnitude = norm_sum * norm_sum + data.n_rows as f64;
    if !magnitude.is_finite() || 128.0 * f64::EPSILON * magnitude > REL_TOL {
        return Err(StateError::Numerical(
            "signed-uniform Gaussian score exceeds its floating-point scale budget".into(),
        ));
    }
    let base = -0.5 * (data.n_rows as f64 * TAU.ln() + yy);
    let value = if q.dim == 0 { base } else { base + log_integral(&q)? };
    if !value.is_finite() {
        return Err(StateError::Numerical(
            "non-finite signed-uniform Gaussian marginal likelihood".into(),
        ));
    }
    Ok(value)
}

//! Shared special functions for stats (PPF, gamma, incomplete beta).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Gauss–Hermite rule for the standard normal: `(nodes, weights)` with
/// `E[g(Z)] ≈ Σ w_i g(z_i)`, exact for polynomials of degree up to `2n − 1`.
///
/// Nodes are the roots of the physicists' Hermite polynomial `H_n`, found by Newton
/// iteration on the orthonormal three-term recurrence (asymptotic starting values),
/// rescaled by `√2` and the weights by `1/√π`. Nodes are returned in descending order.
/// `n = 0` returns empty vectors.
#[must_use]
pub fn gauss_hermite_standard_normal(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let nf = n as f64;
    let mut z = 0.0_f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.855_75 * (2.0 * nf + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * nodes[0] / std::f64::consts::SQRT_2,
            3 => 1.91 * z - 0.91 * nodes[1] / std::f64::consts::SQRT_2,
            _ => 2.0 * z - nodes[i - 2] / std::f64::consts::SQRT_2,
        };
        let mut derivative = 1.0;
        for _ in 0..100 {
            let mut p1 = std::f64::consts::PI.powf(-0.25);
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / (jf + 1.0)).sqrt() * p2 - (jf / (jf + 1.0)).sqrt() * p3;
            }
            derivative = (2.0 * nf).sqrt() * p2;
            let step = p1 / derivative;
            z -= step;
            if step.abs() < 1e-15 * z.abs().max(1.0) {
                break;
            }
        }
        nodes[i] = z * std::f64::consts::SQRT_2;
        weights[i] = 2.0 / (derivative * derivative * std::f64::consts::PI.sqrt());
        if n - 1 - i != i {
            nodes[n - 1 - i] = -nodes[i];
            weights[n - 1 - i] = weights[i];
        }
    }
    (nodes, weights)
}

/// Gauss–Legendre rule on `[-1, 1]`: `(nodes, weights)` with
/// `∫_{-1}^{1} g(u) du ≈ Σ w_i g(u_i)`, exact for polynomials of degree up to `2n − 1`.
///
/// Nodes are the roots of the Legendre polynomial `P_n`, found by Newton iteration on
/// the three-term recurrence from the asymptotic starting values
/// `cos(π (i + 3/4) / (n + 1/2))`; the weights are `2 / ((1 − u²) P_n'(u)²)`. Nodes are
/// returned in ascending order, symmetric about zero. `n = 0` returns empty vectors.
#[must_use]
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        let mut u = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut derivative = 1.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf + 1.0) * u * p2 - jf * p3) / (jf + 1.0);
            }
            derivative = nf * (u * p1 - p2) / (u * u - 1.0);
            let step = p1 / derivative;
            u -= step;
            if step.abs() < 1e-16 {
                break;
            }
        }
        // `i` counts down from the largest root; store ascending.
        nodes[n - 1 - i] = u;
        nodes[i] = -u;
        let weight = 2.0 / ((1.0 - u * u) * derivative * derivative);
        weights[n - 1 - i] = weight;
        weights[i] = weight;
    }
    (nodes, weights)
}

/// The 97.5% standard-normal quantile, the two-sided 95% critical value `z_{0.975}`.
///
/// Re-export of the kernels constant, the single workspace owner.
pub use antecedent_kernels::NORMAL_Q975;

/// Standard-normal PPF: an alias of [`antecedent_kernels::norm_inv`], the one high-accuracy
/// owner (Acklam seed plus one Halley step, a few ulps over the normal range). Returns
/// infinities at 0 and 1, and NaN outside [0, 1].
#[inline]
#[must_use]
pub fn normal_ppf(p: f64) -> f64 {
    antecedent_kernels::norm_inv(p)
}

/// Digamma `ψ(z) = d/dz ln Γ(z)` (reflection + asymptotic series).
#[must_use]
pub(crate) fn digamma(mut z: f64) -> f64 {
    if !(z.is_finite() && z > 0.0) {
        return f64::NAN;
    }
    let mut result = 0.0;
    // Reflection for z < 0.5: ψ(1−z) − ψ(z) = π cot(πz).
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        result -= pi / (pi * z).tan();
        z = 1.0 - z;
    }
    // Recurrence to z ≥ 8.
    while z < 8.0 {
        result -= 1.0 / z;
        z += 1.0;
    }
    // Asymptotic: ψ(z) ≈ ln z − 1/(2z) − Σ B_{2k}/(2k z^{2k})
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    result += z.ln() - 0.5 * iz;
    // Bernoulli terms: 1/12, −1/120, 1/252, −1/240, 1/132, −691/32760, …
    result -= iz2
        * (1.0 / 12.0
            - iz2
                * (1.0 / 120.0
                    - iz2
                        * (1.0 / 252.0
                            - iz2
                                * (1.0 / 240.0 - iz2 * (1.0 / 132.0 - iz2 * (691.0 / 32760.0))))));
    result
}

/// Trigamma `ψ₁(z) = d²/dz² ln Γ(z)` (reflection + asymptotic series).
#[must_use]
pub(crate) fn trigamma(mut z: f64) -> f64 {
    if !(z.is_finite() && z > 0.0) {
        return f64::NAN;
    }
    // Reflection: ψ₁(z) + ψ₁(1−z) = π² / sin²(πz) ⇒ ψ₁(z) = π²csc²(πz) − ψ₁(1−z).
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let s = (pi * z).sin();
        return (pi * pi) / (s * s) - trigamma(1.0 - z);
    }
    let mut result = 0.0;
    while z < 8.0 {
        result += 1.0 / (z * z);
        z += 1.0;
    }
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    // ψ₁(z) ≈ 1/z + 1/(2z²) + Σ B_{2k}/z^{2k+1}
    result += iz + 0.5 * iz2;
    result += iz2
        * iz
        * (1.0 / 6.0
            - iz2 * (1.0 / 30.0 - iz2 * (1.0 / 42.0 - iz2 * (1.0 / 30.0 - iz2 * (5.0 / 66.0)))));
    result
}

/// Lanczos approximation of `ln Γ(z)`.
#[must_use]
pub fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_654_078_675e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        // Gamma has simple poles at z = 0, -1, -2, ... In floating point, sin(pi * z) at
        // these poles is a tiny nonzero residual (e.g. sin(-pi) ~= -1.2246e-16 rather than
        // exactly 0), and it can land on either side of zero depending on z's exact bit
        // pattern. When it lands negative, `.ln()` of it is NaN, which then propagates
        // through every caller. `scipy.special.gammaln` returns +Inf at these poles by
        // convention, so detect them directly rather than trusting the reflection formula's
        // floating-point residual.
        #[allow(clippy::float_cmp, reason = "exact-integer test, not a computed-value comparison")]
        if z <= 0.0 && z == z.trunc() {
            return f64::INFINITY;
        }
        // Between the poles `sin(pi * z)` is genuinely negative on half the intervals
        // (z = -0.5, -2.5, …), so `.ln()` of it is NaN there too — not just at the poles.
        // Γ itself is negative on those intervals, so the real-valued quantity this function
        // can return is `ln|Γ(z)|`, which is what `scipy.special.gammaln` returns
        // (`gammaln(-0.5) = 1.2655…`). Take the magnitude.
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().abs().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
///
/// Below the threshold this is the direct sum; above it, `ln Γ(a)` and `ln Γ(a+b)` are each
/// `O(a ln a)` while their difference is `O(ln a)`, so subtracting them directly cancels to
/// rounding noise (at `a = 5·10⁶` that noise is `~10⁻⁸`, moving `ln B` — and anything computed
/// from it, such as the Student-t quantile at large df — by more than double precision should
/// allow). Above the threshold, expand `ln Γ(big + small) − ln Γ(big)` with Stirling's series,
/// combining the leading `(z − ½) ln z − z` terms algebraically (via `ln1p(small / big)`)
/// instead of forming their numerically cancelling difference; the Stirling error series
/// (to `z⁻⁹`, below `3e-16` for `z ≥ 15`) supplies the remaining terms.
fn ln_beta(a: f64, b: f64) -> f64 {
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    if big >= 15.0 {
        let sum = big + small;
        // `ln G(w) = (w - 1/2) ln w - w + (1/2) ln 2 pi + S(w)` with the Stirling error `S`
        // (series to the `w^-9` term, below 3e-16 for `w >= 15`); the polynomial parts of the
        // difference combine through `ln1p`, and `S(sum) - S(big)` is a small difference of small
        // terms. The previous threshold of 1e3 with a single `1/(12 w)` correction left the
        // direct `ln G(a) + ln G(b) - ln G(a + b)` for `15 <= a < 1e3`, whose Lanczos error scales
        // with `ln G(a)` (2.6e-12 at a = 500, a relative 1.2e-12 in the Student-t tail).
        let diff = (big - 0.5) * (small / big).ln_1p() + small * sum.ln() - small
            + stirling_error(sum)
            - stirling_error(big);
        ln_gamma(small) - diff
    } else {
        ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    }
}

/// Regularized incomplete beta `I_x(a, b)`.
#[must_use]
pub(crate) fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    regularized_incomplete_beta_with_complement(x, 1.0 - x, a, b)
}

/// `I_x(a, b)` given both `x` and its complement `y = 1 − x`.
///
/// Callers that can form `y` directly (the Student-t survival function has
/// `y = t²/(df + t²)`) pass it so that `x` near 1 does not lose `y`'s leading digits to the
/// subtraction `1 − x`, which otherwise limits accuracy near the median.
fn regularized_incomplete_beta_with_complement(x: f64, y: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Test the complement, not `x >= 1.0`: a caller-supplied `y` such as the Student-t
    // `t^2 / (df + t^2)` can be a positive number far below `x`'s ulp, so `x` rounds to exactly
    // 1 while `I_x(a, b)` still differs from 1 by order `sqrt(y)`. Returning 1 there flattened
    // the Student-t survival function to exactly 1/2 for `|t| < 1e-8 sqrt(df)`.
    if y <= 0.0 {
        return 1.0;
    }
    // Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) where the continued fraction
    // converges fastest (Numerical Recipes criterion).
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta_with_complement(y, x, b, a);
    }
    let lnb = ln_beta(a, b);
    // ln(x): when x is within a few ulp of 1 (y small), x's own rounding has already discarded
    // y's leading digits, and a*ln(x) amplifies that lost precision by a — enough to move the
    // Student-t quantile by ~1e-9 at df ~ 1e7. ln1p(-y) keeps the precision of the accurately
    // computed small y instead. When y is not small (x itself is the accurately-resolved small
    // operand, as after the symmetry swap above), ln1p(-y) would instead force the cancellation
    // 1 - y and must be avoided; x.ln() is exact there.
    let lnx = if y < 0.5 { (-y).ln_1p() } else { x.ln() };
    // The same argument applies to `ln(y)`: when x is the small operand, `ln1p(-x)` keeps what
    // the stored `y = 1 - x` has already rounded away (an absolute 1e-16, which `b` amplifies).
    let lny = if x < 0.5 { (-x).ln_1p() } else { y.ln() };
    let front = (lnx * a + lny * b - lnb).exp() / a;
    // Modified Lentz evaluation of the continued fraction. Both the starting `d` and every
    // odd-step `1 + num * d` have the form `1 - x (1 - small)`, which cancels catastrophically
    // when `x` is within ~1/a of 1 (relative error 1e-16 / ((a + b) y / a): 2.5e-5 at
    // df = 1e12 for the Student-t tail). With the exact complement `y` they are computed as
    // `((1 - b) + (a + b) y) / (a + 1)` and `((D - N) + N y) / D`, where
    // `N = (a + m)(a + b + m)`, `D = (a + 2m)(a + 2m + 1)` and
    // `D - N = a (2m + 1 - b) + 3 m^2 + 2 m - m b` is expanded so that it is not itself a
    // difference of two numbers near `a^2`. `c` and `d` return to `1 + e_c` and `1 + e_d` after
    // each even step; the deviations are carried explicitly so the odd step never forms
    // `1 + (-1 + small)` from rounded operands.
    let stable = y < 0.5;
    let mut c = 1.0;
    let mut d =
        if stable { ((1.0 - b) + (a + b) * y) / (a + 1.0) } else { 1.0 - (a + b) * x / (a + 1.0) };
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    // A fixed cap silently returns a partially-converged value once `a`/`b` grow: at
    // x=0.5, a=b=1e6 the continued fraction needs ~420 terms, but a 200-term cap stops
    // early and returns 0.4996679895 against a true value of 0.5 (rel. err 6.6e-4). Not
    // reachable from `student_t_sf` today (it pins b=0.5, which the symmetry switch above
    // routes to the fast side), but this is public API with no documented restriction on
    // a/b. Scale the cap with sqrt(max(a,b)) the same way `gamma_p_series` does for an
    // identical convergence-rate issue, so the tolerance break — not the cap — ends the
    // loop across the practically reachable range.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the sqrt is of a value >= 1 and capped at 1e9, so it is positive and fits usize"
    )]
    let max_iter = 500 + 10 * (a.max(b).max(1.0).sqrt().min(1.0e9) as usize);
    for m in 1..max_iter {
        let m_f = m as f64;
        // Even step.
        let num = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        let td = num * d;
        let mut d_raw = 1.0 + td;
        if d_raw.abs() < 1e-30 {
            d_raw = 1e-30;
        }
        let e_d = -td / d_raw;
        d = 1.0 / d_raw;
        let tc = num / c;
        c = 1.0 + tc;
        let mut e_c = tc;
        if c.abs() < 1e-30 {
            c = 1e-30;
            e_c = c - 1.0;
        }
        f *= d * c;
        // Odd step.
        let num = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        let (mut d_raw, mut c_raw) = if stable {
            let nn = (a + m_f) * (a + b + m_f);
            let dd = (a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0);
            let dn = a * (2.0 * m_f + 1.0 - b) + 3.0 * m_f * m_f + 2.0 * m_f - m_f * b;
            let one_plus = (dn + nn * y) / dd;
            (one_plus + num * e_d, one_plus - num * e_c / c)
        } else {
            (1.0 + num * d, 1.0 + num / c)
        };
        if d_raw.abs() < 1e-30 {
            d_raw = 1e-30;
        }
        if c_raw.abs() < 1e-30 {
            c_raw = 1e-30;
        }
        d = 1.0 / d_raw;
        c = c_raw;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (front * f).clamp(0.0, 1.0)
}

/// `x = df / (df + t^2)` and `y = t^2 / (df + t^2)`, each accurate to relative 1e-16 on its own
/// (not as `1 - the other`), formed without squaring a huge `t`, which would overflow at
/// `|t| > 1e154` and turn a valid tail into NaN.
fn student_t_beta_arguments(t: f64, df: f64) -> (f64, f64) {
    if t.abs() > 1.0 {
        let u = df / t.abs() / t.abs();
        (u / (1.0 + u), 1.0 / (1.0 + u))
    } else {
        let denom = df + t * t;
        (df / denom, t * t / denom)
    }
}

/// `P(0 < T <= t) = 1/2 - P(T > t)` for `t >= 0`, as `I_y(1/2, df/2) / 2`.
///
/// Near the median this is the quantity that carries information: `P(T > t)` is within
/// `t * pdf(0)` of 1/2, so a quantile solved on the survival function inherits an absolute
/// error of 1e-16 in that probability (a relative 2.8e-10 in `t` at `p - 1/2 = 1e-7`).
fn student_t_central_mass(t: f64, df: f64) -> f64 {
    let (x, y) = student_t_beta_arguments(t, df);
    0.5 * regularized_incomplete_beta_with_complement(y, x, 0.5, 0.5 * df)
}

/// Survival function P(T > t) for Student-t with `df` degrees of freedom.
#[must_use]
pub fn student_t_sf(t: f64, df: f64) -> f64 {
    if !(df.is_finite() && df > 0.0) || t.is_nan() {
        return f64::NAN;
    }
    if t == f64::INFINITY {
        return 0.0;
    }
    if t == f64::NEG_INFINITY {
        return 1.0;
    }
    let (x, y) = student_t_beta_arguments(t, df);
    let half_tail = 0.5 * regularized_incomplete_beta_with_complement(x, y, 0.5 * df, 0.5);
    if t >= 0.0 { half_tail } else { 1.0 - half_tail }
}

/// Degrees of freedom above which [`student_t_ppf`] uses the Cornish–Fisher expansion about
/// the normal quantile.
const STUDENT_T_NORMAL_SERIES_FROM_DF: f64 = 1.0e7;

/// Inverse Student-t CDF (quantile function) via bisection on [`student_t_sf`]
/// (Cornish–Fisher expansion about the normal quantile above `1e7` degrees of freedom).
///
/// Returns `t` such that `P(T <= t) = p` for `T ~ Student-t(df)`. `p` must lie strictly
/// inside `(0, 1)`; returns `NaN` otherwise.
///
/// `df <= 0` (or non-finite `df`) returns `f64::INFINITY`: a zero- or negative-degrees-of-
/// freedom Student-t has no meaningful critical value, and callers that combine this with
/// an already-infinite standard error (e.g. a single Monte Carlo draw, which has no sample
/// variance) want an unbounded — not NaN — critical multiplier out of `INFINITY * INFINITY`.
#[must_use]
pub fn student_t_ppf(p: f64, df: f64) -> f64 {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    if !(df.is_finite() && df > 0.0) {
        return f64::INFINITY;
    }
    #[allow(clippy::float_cmp, reason = "the median is exactly representable")]
    if p == 0.5 {
        return 0.0;
    }
    // For very large df the quantile is the normal one plus a Cornish–Fisher series in 1/df
    // whose truncation error (`O(df⁻⁴)`, below 1e-28 here) is far under double precision;
    // bisecting the survival function instead loses digits to the incomplete-beta
    // cancellation as `x = df/(df+t²) → 1`.
    if df > STUDENT_T_NORMAL_SERIES_FROM_DF {
        let z = normal_ppf(p);
        let z2 = z * z;
        let g1 = z * (z2 + 1.0) / 4.0;
        let g2 = z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / 96.0;
        let g3 = z * (3.0 * z2 * z2 * z2 + 19.0 * z2 * z2 + 17.0 * z2 - 15.0) / 384.0;
        return z + g1 / df + g2 / (df * df) + g3 / (df * df * df);
    }
    // student_t_sf(t, df) is strictly decreasing in t, from 1 (t -> -inf) to 0 (t -> +inf).
    // Solve for the non-negative root and mirror by symmetry for p < 0.5.
    let target_sf = if p > 0.5 { 1.0 - p } else { p };
    // `p - 1/2` is exact. Within 0.25 of the median the root is found on the central mass
    // `1/2 - target_sf`, which keeps full relative accuracy in `t` as `p -> 1/2`.
    let central = (p - 0.5).abs();
    let use_central = central < 0.25;
    let root_is_above = |t: f64| {
        if use_central {
            student_t_central_mass(t, df) < central
        } else {
            student_t_sf(t, df) > target_sf
        }
    };
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    // Heavy tails (df < 1) put a small-probability quantile at astronomically large t
    // (df = 0.3, p = 1e-10 gives 6.5e31), so the bracket must grow until the tail is below the
    // target rather than stop at a fixed cap and return the cap as if it were the root.
    while root_is_above(hi) {
        if hi >= 1e300 {
            return if p > 0.5 { f64::INFINITY } else { f64::NEG_INFINITY };
        }
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if root_is_above(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    if p > 0.5 { t } else { -t }
}

/// Chi-square survival function `P(χ²_df > statistic)` through the regularized upper
/// incomplete gamma, `Q(df / 2, statistic / 2)`.
///
/// A non-positive `statistic` gives `1.0`. `df` is converted through `u32` (saturating),
/// so a degrees-of-freedom count beyond `u32::MAX` is treated as `u32::MAX`.
#[must_use]
pub fn chi_square_sf(statistic: f64, df: usize) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    gamma_q(f64::from(u32::try_from(df).unwrap_or(u32::MAX)) * 0.5, statistic * 0.5)
}

/// Regularized upper incomplete gamma `Q(a, x)`.
#[must_use]
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        (1.0 - gamma_p_series(a, x)).clamp(0.0, 1.0)
    } else {
        gamma_q_cf(a, x).clamp(0.0, 1.0)
    }
}

/// Lower regularized incomplete gamma `P(a, x)` by series expansion.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    // Terms decay like exp(-n^2/2a), so convergence needs O(sqrt(a)) of them -- a fixed cap
    // silently returns a partial sum once `a` grows. Scale the cap with sqrt(a) so the
    // tolerance break, not the cap, ends the loop (mirrored in
    // `regularized_incomplete_beta`'s continued fraction below, an identical issue).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the sqrt is of a value >= 1 and capped at 1e9, so it is positive and fits usize"
    )]
    let max_iter = 500 + 10 * (a.max(1.0).sqrt().min(1.0e9) as usize);
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * gamma_density_factor(a, x)
}

/// Upper regularized incomplete gamma `Q(a, x)` by Lentz continued fraction.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    // As for the lower series, the transition region x ≈ a needs a
    // shape-dependent budget; a fixed 499 terms can stop before convergence.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the positive sqrt is capped at 1e9, as in the lower series"
    )]
    let max_iter = 500 + 10 * (a.max(1.0).sqrt().min(1.0e9) as usize);
    for i in 1..max_iter {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    gamma_density_factor(a, x) * h
}

/// `x^a e^{-x} / Gamma(a)`, the common prefactor of the incomplete-gamma series and fraction.
///
/// Evaluated as `-x + a ln x - ln Gamma(a)` the three terms are each `O(a ln a)` while their
/// sum is `O(ln a)`, so the rounding of the sum is `1e-16 * a ln a` absolute in the exponent
/// (2e-9 relative at `a = 1e6`). For `a >= 1` this uses Loader's saddle-point form
/// `sqrt(a / 2 pi) exp(-stirlerr(a) - bd0(a, x))`, where `bd0` is the deviance
/// `a ln(a / x) + x - a` evaluated without cancelling the three large terms.
fn gamma_density_factor(a: f64, x: f64) -> f64 {
    if a < 1.0 {
        return (-x + a * x.ln() - ln_gamma(a)).exp();
    }
    ((a / (2.0 * std::f64::consts::PI)).sqrt()) * (-stirling_error(a) - deviance_bd0(a, x)).exp()
}

/// `ln Gamma(n + 1) - [(n + 1/2) ln n - n + (1/2) ln 2 pi]`, the error of Stirling's formula.
fn stirling_error(n: f64) -> f64 {
    if n >= 15.0 {
        // Stirling series 1/(12n) - 1/(360 n^3) + 1/(1260 n^5) - 1/(1680 n^7) + 1/(1188 n^9);
        // the next term is below 3e-16 at n = 15.
        let n2 = n * n;
        return (1.0 / 12.0
            - (1.0 / 360.0 - (1.0 / 1260.0 - (1.0 / 1680.0 - 1.0 / (1188.0 * n2)) / n2) / n2)
                / n2)
            / n;
    }
    ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Binomial deviance `bd0(a, x) = a ln(a / x) + x - a` (Loader 2000), by a series in
/// `v = (a - x) / (a + x)` when the arguments are close so the three terms do not cancel.
#[allow(clippy::float_cmp, reason = "the series ends when a term no longer changes the sum")]
fn deviance_bd0(a: f64, x: f64) -> f64 {
    if (a - x).abs() < 0.1 * (a + x) {
        let v = (a - x) / (a + x);
        let mut s = (a - x) * v;
        let mut ej = 2.0 * a * v;
        let v2 = v * v;
        for j in 1..1000_u32 {
            ej *= v2;
            let next = s + ej / f64::from(2 * j + 1);
            if next == s {
                return next;
            }
            s = next;
        }
        return s;
    }
    a * (a / x).ln() + x - a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_square_sf_matches_closed_forms() {
        // df = 2: survival is exp(-x / 2).
        for x in [0.5, 2.0, 7.3] {
            assert!((chi_square_sf(x, 2) - (-x / 2.0).exp()).abs() < 1e-12);
        }
        assert!((chi_square_sf(0.0, 3) - 1.0).abs() < f64::EPSILON);
        assert!((chi_square_sf(-1.0, 3) - 1.0).abs() < f64::EPSILON);
        // 95th percentile of chi-square(1) is 3.841458820694124.
        assert!((chi_square_sf(3.841_458_820_694_124, 1) - 0.05).abs() < 1e-9);
    }

    #[test]
    fn gauss_hermite_reproduces_normal_moments_exactly() {
        // E[Z^{2k}] = (2k − 1)!!, odd moments vanish; an n-point rule is exact to degree 2n − 1.
        for n in [1_usize, 2, 5, 8, 31, 32] {
            let (nodes, weights) = gauss_hermite_standard_normal(n);
            assert_eq!(nodes.len(), n);
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12, "n={n}");
            let mut double_factorial = 1.0;
            for k in 0..=6_usize {
                if k > 0 {
                    double_factorial *= (2 * k - 1) as f64;
                }
                if 2 * k > 2 * n - 1 {
                    break;
                }
                let even: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(z, w)| w * z.powi(i32::try_from(2 * k).unwrap()))
                    .sum();
                assert!(
                    (even - double_factorial).abs() <= 1e-10 * double_factorial,
                    "n={n} k={k}: {even} vs {double_factorial}"
                );
            }
            let odd: f64 = nodes.iter().zip(&weights).map(|(z, w)| w * z.powi(3)).sum();
            assert!(odd.abs() < 1e-10, "n={n}");
        }
        assert!(gauss_hermite_standard_normal(0).0.is_empty());
    }

    #[test]
    fn gauss_hermite_two_point_rule_is_plus_minus_one() {
        let (nodes, weights) = gauss_hermite_standard_normal(2);
        assert!((nodes[0] - 1.0).abs() < 1e-13 && (nodes[1] + 1.0).abs() < 1e-13);
        assert!((weights[0] - 0.5).abs() < 1e-13 && (weights[1] - 0.5).abs() < 1e-13);
    }

    #[test]
    fn gauss_legendre_integrates_monomials_up_to_degree_2n_minus_1_exactly() {
        for n in [1usize, 2, 3, 5, 8, 16, 32, 64] {
            let (nodes, weights) = gauss_legendre(n);
            assert_eq!((nodes.len(), weights.len()), (n, n));
            assert!(nodes.windows(2).all(|w| w[0] < w[1]), "n = {n}: ascending");
            assert!(nodes.iter().zip(nodes.iter().rev()).all(|(a, b)| (a + b).abs() < 1e-15));
            for degree in 0..2 * n {
                let rule: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(u, w)| w * u.powi(i32::try_from(degree).unwrap()))
                    .sum();
                let exact = if degree % 2 == 1 { 0.0 } else { 2.0 / (degree as f64 + 1.0) };
                assert!(
                    (rule - exact).abs() < 1e-13,
                    "n = {n}, degree {degree}: {rule} vs {exact}"
                );
            }
            // Degree 2n is not integrated exactly: the rule is exactly as strong as stated
            // (checked where the error of the degree-2n monomial is visible in f64).
            if n <= 8 {
                let rule: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(u, w)| w * u.powi(i32::try_from(2 * n).unwrap()))
                    .sum();
                assert!((rule - 2.0 / (2.0 * n as f64 + 1.0)).abs() > 1e-6, "n = {n}");
            }
        }
        let (nodes, weights) = gauss_legendre(1);
        assert!(nodes[0].abs() < 1e-15 && (weights[0] - 2.0).abs() < 1e-15);
        assert!(gauss_legendre(0).0.is_empty());
    }

    #[test]
    fn student_t_ppf_is_accurate_next_to_the_median() {
        // df = 1 is Cauchy: t = tan(π (p − ½)). df = 2: t = (2p − 1) / √(2p(1 − p)).
        // The continued fraction stopped at relative 1e-10, which left 1.4e-5 (df = 1) and
        // 9.9e-4 (df = 2) relative error in the quantile at p = ½ + 1e-7.
        let p = 0.5 + 1e-7;
        let cauchy = (std::f64::consts::PI * (p - 0.5)).tan();
        assert!((student_t_ppf(p, 1.0) / cauchy - 1.0).abs() < 1e-6, "{}", student_t_ppf(p, 1.0));
        let df2 = (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
        assert!((student_t_ppf(p, 2.0) / df2 - 1.0).abs() < 1e-6, "{}", student_t_ppf(p, 2.0));
        // Mirror image.
        assert!((student_t_ppf(1.0 - p, 2.0) + df2).abs() < 1e-6 * df2);
    }

    #[test]
    fn student_t_ppf_is_smooth_across_the_large_df_series_switch() {
        // Two independent methods (bisection on the survival function below 1e7 df,
        // Cornish–Fisher above) must agree on either side of the switch, and at huge df
        // the quantile is the normal one plus g₁/df with g₁ = z(z² + 1)/4.
        let below = student_t_ppf(0.975, 1.0e7 * (1.0 - 1e-9));
        let above = student_t_ppf(0.975, 1.0e7 * (1.0 + 1e-9));
        assert!((below - above).abs() < 1e-9, "below={below} above={above}");
        let z = normal_ppf(0.975);
        let huge = student_t_ppf(0.975, 1.0e10);
        assert!((huge - (z + z * (z * z + 1.0) / 4.0 / 1.0e10)).abs() < 1e-14, "{huge}");
        assert!((student_t_ppf(0.025, 1.0e10) + huge).abs() < 1e-14);
    }

    #[test]
    fn normal_ppf_pins_common_quantiles() {
        assert!((normal_ppf(0.975) - 1.959_964).abs() < 1e-4);
        assert!((normal_ppf(0.995) - 2.575_829).abs() < 1e-4);
        assert!((normal_ppf(0.95) - 1.644_854).abs() < 1e-4);
        assert!((normal_ppf(0.99) - 2.326_348).abs() < 1e-4);
        assert!(normal_ppf(0.5).abs() < 1e-12);
    }

    #[test]
    fn normal_ppf_monotone_over_grid() {
        let mut prev = f64::NEG_INFINITY;
        for i in 1..1000 {
            let p = f64::from(i) / 1000.0;
            let x = normal_ppf(p);
            assert!(x > prev, "not monotone at p={p}: {x} <= {prev}");
            prev = x;
        }
    }

    #[test]
    fn normal_ppf_symmetric() {
        for &p in &[0.001, 0.01, 0.05, 0.1, 0.25, 0.4, 0.49] {
            let lo = normal_ppf(p);
            let hi = normal_ppf(1.0 - p);
            assert!((lo + hi).abs() < 1e-9, "asymmetry at p={p}: {lo} vs {hi}");
        }
    }

    /// `normal_ppf` must reach near-machine-epsilon accuracy, matching the ~1e-13
    /// precision of the Cody-rational `erf`/`erfc` it is now refined against —
    /// not just the ~1e-9 Acklam approximation on its own.
    ///
    /// References: `scipy.stats.norm.ppf(p)`, scipy 1.14.
    #[test]
    fn normal_ppf_matches_scipy_to_near_machine_epsilon() {
        let cases = [
            (0.975, 1.959_963_984_540_054),
            (0.995, 2.575_829_303_548_900_4),
            (0.9, 1.281_551_565_544_600_4),
            (0.99, 2.326_347_874_040_840_8),
            (1e-6, -4.753_424_308_822_899),
            (1e-10, -6.361_340_902_404_056),
            (1e-12, -7.034_483_825_301_131),
            (1e-15, -7.941_345_326_170_998),
            (1.0 - 1e-6, 4.753_424_308_817_087),
            (1.0 - 1e-12, 7.034_486_910_047_835_6),
            (1.0 - 1e-15, 7.941_444_487_415_979),
        ];
        for (p, expected) in cases {
            let got = normal_ppf(p);
            let rel = (got - expected).abs() / expected.abs().max(1e-8);
            assert!(rel < 5e-10, "normal_ppf({p}): got={got} expected={expected} rel={rel:e}");
        }
    }

    /// `ln_gamma` at non-positive integers (Gamma's poles) must return +Inf, matching
    /// `scipy.special.gammaln`'s convention -- not NaN from a negative-going floating-point
    /// residual in `sin(pi * z)` at the pole.
    #[test]
    fn ln_gamma_returns_infinity_at_non_positive_integer_poles() {
        for z in [0.0, -1.0, -2.0, -3.0, -10.0, -100.0] {
            let got = ln_gamma(z);
            assert!(got.is_infinite() && got > 0.0, "ln_gamma({z}) = {got}, expected +inf");
        }
    }

    #[test]
    fn digamma_trigamma_pin_known_values() {
        const EULER: f64 = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + EULER).abs() < 1e-10);
        assert!((digamma(0.5) + EULER + 2.0 * std::f64::consts::LN_2).abs() < 1e-10);
        assert!((trigamma(1.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-10);
        // Recurrence: ψ(z+1) = ψ(z) + 1/z
        let z = 2.3;
        assert!((digamma(z + 1.0) - digamma(z) - 1.0 / z).abs() < 1e-12);
        assert!((trigamma(z + 1.0) - trigamma(z) + 1.0 / (z * z)).abs() < 1e-12);
    }

    #[test]
    fn trigamma_golden_values() {
        // Reference values from scipy.special.polygamma(1, z).
        let cases = [
            (0.01, 10_001.621_213_528_313),
            (0.1, 101.433_299_150_792_76),
            (0.25, 17.197_329_154_507_113),
            (0.49, 5.108_092_483_881_403),
            (0.5, 4.934_802_200_544_68),
            (1.0, std::f64::consts::PI.powi(2) / 6.0),
            (5.0, 0.221_322_955_737_115_3),
        ];
        for &(z, expected) in &cases {
            let got = trigamma(z);
            let rel = (got - expected).abs() / expected.abs().max(1.0);
            assert!(rel < 1e-10, "trigamma({z}): got={got} expected={expected} rel={rel}");
        }
    }

    #[test]
    fn trigamma_reflection_identity_grid() {
        let pi = std::f64::consts::PI;
        let mut z = 0.01;
        while z < 0.5 {
            let lhs = trigamma(z) + trigamma(1.0 - z);
            let s = (pi * z).sin();
            let rhs = (pi * pi) / (s * s);
            let rel = (lhs - rhs).abs() / rhs.max(1.0);
            assert!(rel < 1e-10, "identity fail at z={z}: lhs={lhs} rhs={rhs}");
            z += 0.01;
        }
    }

    #[test]
    fn student_t_sf_symmetry_and_guards() {
        for &df in &[1.0, 2.0, 5.0, 30.0] {
            assert!((student_t_sf(0.0, df) - 0.5).abs() < 1e-12, "sf(0) df={df}");
            for &t in &[0.5, 1.0, 2.0, 3.5] {
                let pos = student_t_sf(t, df);
                let neg = student_t_sf(-t, df);
                assert!((neg - (1.0 - pos)).abs() < 1e-12, "symmetry t={t} df={df}");
            }
            let mut prev = 1.0;
            for i in -40..=40 {
                let t = f64::from(i) * 0.25;
                let s = student_t_sf(t, df);
                assert!(s <= prev + 1e-12, "not monotone at t={t} df={df}: {s} > {prev}");
                prev = s;
            }
        }
        assert!((student_t_sf(f64::INFINITY, 5.0) - 0.0).abs() < 1e-15);
        assert!((student_t_sf(f64::NEG_INFINITY, 5.0) - 1.0).abs() < 1e-15);
        assert!(student_t_sf(1.0, f64::NAN).is_nan());
        assert!(student_t_sf(1.0, 0.0).is_nan());
        assert!(student_t_sf(1.0, -1.0).is_nan());
        assert!(student_t_sf(f64::NAN, 5.0).is_nan());
    }

    /// `gamma_q` must stay accurate at the large `a` reached by G² on many strata.
    ///
    /// `gamma_q` routes to the series branch whenever `x < a + 1`, which covers the bulk of
    /// the null p-value distribution (a true non-edge puts the statistic near its df). The
    /// series needs O(√a) terms, so a fixed iteration cap silently returns a partial sum as
    /// `a` grows -- inflating p-values, which then propagates into any p-value-ordered step
    /// (BH/BY, edge ranking).
    ///
    /// References are `scipy.special.gammaincc(df/2, df/2)`, scipy 1.14.
    #[test]
    fn gamma_q_series_branch_stays_accurate_at_large_a() {
        // (df, gammaincc(df/2, df/2)) -- the median case, deepest in the series branch.
        let cases = [
            (1.0e3_f64, 0.494_052_853_829_239_64_f64),
            (1.0e4, 0.498_119_365_966_182_67),
            (1.0e5, 0.499_405_291_895_206_7),
            (1.0e6, 0.499_811_936_803_394_5),
        ];
        for (df, expected) in cases {
            let got = gamma_q(df * 0.5, df * 0.5);
            assert!(
                (got - expected).abs() < 1e-6,
                "gamma_q at df={df}: got {got}, scipy says {expected}"
            );
        }
    }

    #[test]
    fn gamma_q_fraction_matches_the_gamma_recurrence_at_large_shape() {
        // Q(a+1,x) = Q(a,x) + x^a exp(-x) / Gamma(a+1). At x=a+1,
        // Q(a+1,x) uses the independently evaluated lower-series branch.
        for a in [50_000.0, 500_000.0, 5_000_000.0] {
            let x = a + 1.0;
            let expected = gamma_q(a + 1.0, x) - gamma_density_factor(a, x) / a;
            let got = gamma_q(a, x);
            assert!((got - expected).abs() < 1e-10, "a={a}: {got} vs {expected}");
        }
    }

    /// `regularized_incomplete_beta`'s continued fraction must stay accurate for large,
    /// comparable shape parameters. At x=0.5, a=b=1e6 the true value is exactly 0.5 (up to
    /// float rounding), but a fixed 200-term cap converges only partially and returns
    /// 0.4996679895 -- a 6.6e-4 relative error. Convergence there needs ~420 terms; a
    /// 200-term cap silently returns a partial sum instead. Not currently reachable via
    /// `student_t_sf` (which pins b=0.5, routed to the fast symmetry side), but this is
    /// public API with no documented restriction on a/b.
    ///
    /// References are `scipy.special.betainc(a, b, x)`, scipy 1.14.
    #[test]
    fn regularized_incomplete_beta_stays_accurate_for_large_comparable_shapes() {
        let got = regularized_incomplete_beta(0.5, 1.0e6, 1.0e6);
        assert!((got - 0.5).abs() < 1e-6, "I_0.5(1e6, 1e6): got={got}, scipy says ~0.5");

        // Smaller-shape sanity checks against exact scipy references stay unaffected.
        let got2 = regularized_incomplete_beta(0.4, 2.0, 3.0);
        assert!((got2 - 0.5248).abs() < 1e-6, "I_0.4(2,3): got={got2}");
        let got3 = regularized_incomplete_beta(0.3, 0.5, 0.5);
        assert!((got3 - 0.369_010_119_565_545_36).abs() < 1e-6, "I_0.3(0.5,0.5): got={got3}");
    }

    #[test]
    fn student_t_sf_golden_values() {
        // scipy.stats.t.sf(t, df)
        let cases = [
            (1.0, 1.0, 0.25),
            (2.0, 5.0, 0.050_969_739_414_929_14),
            (0.0, 10.0, 0.5),
            (-1.0, 1.0, 0.75),
        ];
        for &(t, df, expected) in &cases {
            let got = student_t_sf(t, df);
            assert!((got - expected).abs() < 1e-8, "t={t} df={df}: got={got} expected={expected}");
        }
    }

    #[test]
    fn student_t_ppf_golden_values() {
        // Well-known two-sided-95%/90% critical-value table entries (t_{alpha,df}), to the
        // precision commonly tabulated in textbooks. A generous 1e-3 tolerance guards
        // against transcription imprecision in the reference digits while still catching a
        // grossly wrong implementation.
        let cases =
            [(0.975, 1.0, 12.706), (0.975, 5.0, 2.571), (0.975, 30.0, 2.042), (0.95, 10.0, 1.812)];
        for &(p, df, expected) in &cases {
            let got = student_t_ppf(p, df);
            assert!((got - expected).abs() < 1e-3, "ppf({p}, {df}): got={got} expected={expected}");
        }
        // Exact closed form at df == 1: Student-t(1) is the standard Cauchy distribution,
        // whose CDF inverts to t = tan(pi * (p - 0.5)).
        let exact_df1 = (std::f64::consts::PI * (0.975 - 0.5)).tan();
        assert!(
            (student_t_ppf(0.975, 1.0) - exact_df1).abs() < 1e-9,
            "df=1 closed form: got={} exact={exact_df1}",
            student_t_ppf(0.975, 1.0)
        );
    }

    #[test]
    fn student_t_ppf_symmetric_and_matches_normal_at_large_df() {
        for &df in &[1.0, 5.0, 30.0] {
            for &p in &[0.6, 0.75, 0.9, 0.99] {
                let hi = student_t_ppf(p, df);
                let lo = student_t_ppf(1.0 - p, df);
                assert!((hi + lo).abs() < 1e-6, "asymmetry p={p} df={df}: hi={hi} lo={lo}");
            }
        }
        // Student-t converges to the standard normal as df -> infinity.
        let t_large_df = student_t_ppf(0.975, 1e7);
        let z = normal_ppf(0.975);
        assert!((t_large_df - z).abs() < 1e-3, "t={t_large_df} z={z}");
    }

    #[test]
    fn student_t_ppf_degenerate_df_is_infinite_not_nan() {
        // df <= 0 (e.g. a single-sample Monte Carlo estimate, n_samples - 1 == 0) must
        // return +inf, not NaN, so combining with an already-infinite stderr in a
        // downstream CI computation stays inf * inf = inf rather than inf * NaN = NaN.
        assert!(student_t_ppf(0.975, 0.0).is_infinite() && student_t_ppf(0.975, 0.0) > 0.0);
        assert!(student_t_ppf(0.975, -1.0).is_infinite() && student_t_ppf(0.975, -1.0) > 0.0);
        assert!(student_t_ppf(1.5, 5.0).is_nan());
        assert!(student_t_ppf(0.0, 5.0).is_nan());
    }
}

#[cfg(test)]
mod ppf_domain_tests {
    #[test]
    #[allow(clippy::float_cmp, reason = "exact conservative adjustment / infinite endpoints")]
    fn ppf_preserves_mathematical_domain_and_tiny_probabilities() {
        use super::normal_ppf;
        assert_eq!(normal_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_ppf(1.0), f64::INFINITY);
        for p in [-1.0, 2.0, f64::NAN] {
            assert!(normal_ppf(p).is_nan());
        }
        assert!(normal_ppf(1e-310) < normal_ppf(1e-300));
    }
}

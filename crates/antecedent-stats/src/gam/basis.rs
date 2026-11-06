//! Cubic spline basis and roughness penalties.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CUBIC_DEGREE, CUBIC_ORDER, RoughnessPenalty, StatsError};

pub(super) fn validate_knots(knots: &[f64], n_basis: usize) -> Result<(), StatsError> {
    let need = n_basis + CUBIC_ORDER;
    if knots.len() != need {
        return Err(StatsError::Shape {
            message: "knot vector length must equal n_basis + 4 for cubic B-splines",
        });
    }
    for w in knots.windows(2) {
        if !(w[0].is_finite() && w[1].is_finite()) || w[1] < w[0] {
            return Err(StatsError::Shape { message: "knots must be finite and non-decreasing" });
        }
    }
    Ok(())
}

#[allow(clippy::float_cmp, reason = "Only exactly constant data need an artificial knot domain")]
pub(super) fn quantile_knots(x: &[f64], n_basis: usize) -> Result<Vec<f64>, StatsError> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let xmin = sorted[0];
    let xmax = sorted[sorted.len() - 1];
    if !(xmax - xmin).is_finite() {
        return Err(StatsError::Shape { message: "non-finite predictor range" });
    }
    // Degenerate constant column: spread slightly so basis is defined.
    let (xmin, xmax) = if xmax == xmin { (xmin - 1.0, xmax + 1.0) } else { (xmin, xmax) };
    let n_interior = n_basis.saturating_sub(CUBIC_ORDER);
    let mut knots = Vec::with_capacity(n_basis + CUBIC_ORDER);
    for _ in 0..CUBIC_ORDER {
        knots.push(xmin);
    }
    if n_interior > 0 {
        for i in 1..=n_interior {
            let q = i as f64 / (n_interior + 1) as f64;
            knots.push(antecedent_kernels::quantile_type7_sorted(&sorted, q));
        }
    }
    for _ in 0..CUBIC_ORDER {
        knots.push(xmax);
    }
    Ok(knots)
}

/// Span-local Cox–de Boor evaluation: the knot span and the `CUBIC_ORDER`
/// nonzero cubic basis values at `x` (all other basis functions are exactly 0).
pub(super) fn cubic_bspline_nonzeros(x: f64, knots: &[f64]) -> (usize, [f64; CUBIC_ORDER]) {
    // Evaluate the boundary on the last span, including caller-supplied knot
    // vectors that are not clamped. An absolute epsilon moves small-unit
    // inputs outside their knot domain.
    let left = knots[CUBIC_DEGREE];
    let right = knots[knots.len() - CUBIC_ORDER];
    let xx = x.max(left).min(right);

    // Find knot span.
    let mut span = CUBIC_DEGREE;
    for i in CUBIC_DEGREE..(knots.len() - CUBIC_ORDER) {
        if xx >= knots[i] && xx < knots[i + 1] {
            span = i;
            break;
        }
        if i == knots.len() - CUBIC_ORDER - 1 {
            span = i;
        }
    }
    if xx >= right {
        // Quantile knots can repeat the boundary more than degree + 1 times
        // for discrete covariates. Use the last nonempty interval; evaluating
        // a zero-width trailing span would divide by zero.
        span = (CUBIC_DEGREE..(knots.len() - CUBIC_ORDER))
            .rev()
            .find(|&i| knots[i] < right)
            .unwrap_or(CUBIC_DEGREE);
    }

    // Basis of degree 0..3 on the local span (Piegl/Tiller style).
    let mut ndu = [[0.0_f64; CUBIC_ORDER]; CUBIC_ORDER];
    ndu[0][0] = 1.0;
    let mut left = [0.0_f64; CUBIC_ORDER];
    let mut right = [0.0_f64; CUBIC_ORDER];
    for j in 1..CUBIC_ORDER {
        left[j] = xx - knots[span + 1 - j];
        right[j] = knots[span + j] - xx;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = ndu[r][j - 1] / (right[r + 1] + left[j - r]);
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    let mut values = [0.0_f64; CUBIC_ORDER];
    for (i, v) in values.iter_mut().enumerate() {
        *v = ndu[i][CUBIC_DEGREE];
    }
    (span, values)
}

/// First derivatives of the span-local cubic bases.
///
/// Outside the knot domain the clamped evaluation is constant, so this
/// returns zeros. Endpoint derivatives are one-sided from inside the domain.
/// Derivatives use the standard degree-drop recurrence
/// on the same span as [`cubic_bspline_nonzeros`]; coincident end knots contribute
/// a zero term rather than a division by zero.
pub(super) fn cubic_bspline_deriv_nonzeros(x: f64, knots: &[f64]) -> (usize, [f64; CUBIC_ORDER]) {
    let left = knots[CUBIC_DEGREE];
    let right = knots[knots.len() - CUBIC_ORDER];
    let (span, _) = cubic_bspline_nonzeros(x, knots);
    if x < left || x > right {
        return (span, [0.0; CUBIC_ORDER]);
    }
    let quadratic = quadratic_nonzeros(x, knots, span);
    let mut ders = [0.0_f64; CUBIC_ORDER];
    for k in 0..CUBIC_ORDER {
        let i = span - CUBIC_DEGREE + k;
        let dt0 = knots[i + CUBIC_DEGREE] - knots[i];
        let dt1 = knots[i + CUBIC_ORDER] - knots[i + 1];
        let n2_i = if k == 0 { 0.0 } else { quadratic[k - 1] };
        let n2_ip1 = if k == CUBIC_DEGREE { 0.0 } else { quadratic[k] };
        let t0 = if dt0 == 0.0 { 0.0 } else { n2_i / dt0 };
        let t1 = if dt1 == 0.0 { 0.0 } else { n2_ip1 / dt1 };
        ders[k] = (CUBIC_DEGREE as f64) * (t0 - t1);
    }
    (span, ders)
}

/// Degree-2 Cox–de Boor values on the same span as a cubic evaluation.
///
/// `values[k] = N_{span-2+k,2}(x)` for `k = 0,1,2`; `values[3]` is unused.
fn quadratic_nonzeros(x: f64, knots: &[f64], span: usize) -> [f64; CUBIC_ORDER] {
    const QUAD_DEGREE: usize = 2;
    const QUAD_ORDER: usize = 3;
    let mut ndu = [[0.0_f64; QUAD_ORDER]; QUAD_ORDER];
    ndu[0][0] = 1.0;
    let mut left = [0.0_f64; QUAD_ORDER];
    let mut right = [0.0_f64; QUAD_ORDER];
    for j in 1..QUAD_ORDER {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom == 0.0 { 0.0 } else { ndu[r][j - 1] / denom };
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let mut values = [0.0_f64; CUBIC_ORDER];
    for i in 0..QUAD_ORDER {
        values[i] = ndu[i][QUAD_DEGREE];
    }
    values
}

/// Cox–de Boor evaluation of all cubic basis functions at `x` into column-major `out`.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
pub(super) fn eval_cubic_bspline(
    x: f64,
    knots: &[f64],
    n_basis: usize,
    out: &mut [f64],
    row: usize,
    nrows: usize,
) {
    let (span, values) = cubic_bspline_nonzeros(x, knots);

    // Zero all bases for this row then write the order nonzeros.
    for b in 0..n_basis {
        out[b * nrows + row] = 0.0;
    }
    let first = span.saturating_sub(CUBIC_DEGREE);
    for i in 0..CUBIC_ORDER {
        let b = first + i;
        if b < n_basis {
            out[b * nrows + row] = values[i];
        }
    }
}

/// Second-difference matrix `D₂` of size `(K-2)×K` with rows `[1, -2, 1]`.
pub(super) fn second_difference_matrix(n_basis: usize) -> Result<Vec<f64>, StatsError> {
    if n_basis < 3 {
        return Err(StatsError::Shape { message: "second-difference penalty needs n_basis ≥ 3" });
    }
    let rows = n_basis - 2;
    let mut d2 = vec![0.0; rows * n_basis];
    for i in 0..rows {
        d2[i * n_basis + i] = 1.0;
        d2[i * n_basis + i + 1] = -2.0;
        d2[i * n_basis + i + 2] = 1.0;
    }
    Ok(d2)
}

/// Roughness penalty `P = D₂'D₂` (`K×K` row-major).
pub(super) fn second_difference_penalty(n_basis: usize) -> Result<Vec<f64>, StatsError> {
    let d2 = second_difference_matrix(n_basis)?;
    let rows = n_basis - 2;
    let mut p = vec![0.0; n_basis * n_basis];
    for i in 0..n_basis {
        for j in i..n_basis {
            let mut acc = 0.0;
            for r in 0..rows {
                acc += d2[r * n_basis + i] * d2[r * n_basis + j];
            }
            p[i * n_basis + j] = acc;
            if i != j {
                p[j * n_basis + i] = acc;
            }
        }
    }
    Ok(p)
}

/// The penalty matrix of one smooth (`K×K` row-major) for its knot vector.
pub(super) fn roughness_penalty(
    kind: RoughnessPenalty,
    knots: Option<&[f64]>,
    n_basis: usize,
) -> Result<Vec<f64>, StatsError> {
    match kind {
        RoughnessPenalty::CoefficientSecondDifference => second_difference_penalty(n_basis),
        RoughnessPenalty::IntegratedSecondDerivative => {
            let knots = knots.ok_or(StatsError::Shape {
                message: "the integrated second-derivative penalty needs resolved knots",
            })?;
            integrated_second_derivative_penalty(knots, n_basis)
        }
    }
}

/// `h̄³ ∫ B_i''(x) B_j''(x) dx` for the cubic B-splines on `knots`, `K×K` row-major, with `h̄` the
/// mean interior knot spacing.
///
/// The second derivative of `Σ cᵢBᵢ` is the linear spline `Σ dⱼNⱼ` on the knots `t₂ … t_{K+1}`
/// with `dⱼ` the repeated divided differences of `c` (de Boor's derivative formula; a repeated
/// knot contributes a zero coefficient). The integral is then `d'Gd` with `G` the exact Gram
/// matrix of the linear B-splines, so the penalty is a straight-line null space for any knots.
pub(super) fn integrated_second_derivative_penalty(
    knots: &[f64],
    n_basis: usize,
) -> Result<Vec<f64>, StatsError> {
    let k = n_basis;
    if k < CUBIC_ORDER || knots.len() != k + CUBIC_ORDER {
        return Err(StatsError::Shape { message: "integrated penalty needs n_basis + 4 knots" });
    }
    let t = knots;
    let ratio = |numerator: f64, gap: f64| if gap > 0.0 { numerator / gap } else { 0.0 };
    let rows = k - 2;
    // D maps the cubic coefficients to the coefficients of the second derivative.
    let mut d = vec![0.0; rows * k];
    for j in 0..rows {
        let outer = ratio(2.0, t[j + 4] - t[j + 2]);
        let upper = ratio(3.0, t[j + 5] - t[j + 2]);
        let lower = ratio(3.0, t[j + 4] - t[j + 1]);
        d[j * k + j] = outer * lower;
        d[j * k + j + 1] = -outer * (upper + lower);
        d[j * k + j + 2] = outer * upper;
    }
    // G: Gram matrix of the linear B-splines on t₂ … t_{K+1}.
    let mut gram = vec![0.0; rows * rows];
    for j in 0..rows {
        gram[j * rows + j] = (t[j + 4] - t[j + 2]) / 3.0;
        if j + 1 < rows {
            let off = (t[j + 4] - t[j + 3]) / 6.0;
            gram[j * rows + j + 1] = off;
            gram[(j + 1) * rows + j] = off;
        }
    }
    let spacing = (t[k] - t[CUBIC_DEGREE]) / (k - CUBIC_DEGREE) as f64;
    let scale = spacing.powi(3);
    // P = scale · D'GD.
    let mut gd = vec![0.0; rows * k];
    for r in 0..rows {
        for c in 0..k {
            gd[r * k + c] = (0..rows).map(|m| gram[r * rows + m] * d[m * k + c]).sum();
        }
    }
    let mut p = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let value = scale * (0..rows).map(|r| d[r * k + i] * gd[r * k + j]).sum::<f64>();
            p[i * k + j] = value;
            p[j * k + i] = value;
        }
    }
    Ok(p)
}

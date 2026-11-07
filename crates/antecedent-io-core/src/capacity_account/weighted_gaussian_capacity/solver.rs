use super::{
    CenteredSystem, DesignShape, FaerBackend, FitDiagnostics, Folded, GaussianCrossProducts,
    LeastSquaresWorkspace, Ordering, Solved, WeightedCapacityError, cholesky_spd, fit_wls,
    invert_square, zeros,
};

/// Solve the weighted normal equations of the folded rows.
///
/// A design with one constant column is solved in centered coordinates
/// ([`solve_centered`]); any other design on the raw Gram ([`solve_gram`]).
pub(super) fn solve_weighted(folded: &Folded, p: usize) -> Result<Solved, WeightedCapacityError> {
    match &folded.centered {
        Some(centered) => solve_centered(centered, p),
        None => solve_gram(&folded.cross_products, p),
    }
}

/// `L` is the row-major Cholesky factor of `X'WX`. That buffer is also the
/// column-major layout of `L'`. [`fit_wls`] on `L'` with right-hand side
/// `L⁻¹ X'Wy` is the weighted normal equation.
fn solve_gram(
    cross_products: &GaussianCrossProducts,
    p: usize,
) -> Result<Solved, WeightedCapacityError> {
    let factor = cholesky_spd(&cross_products.xtwx, p).ok_or(WeightedCapacityError::Singular)?;
    let reduced_y = forward_substitute(&factor, p, &cross_products.xtwy)?;
    let mut ones = zeros(p, "unit weights")?;
    ones.fill(1.0);
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_wls(&factor, p, p, &reduced_y, &ones, &FaerBackend, &mut workspace)
        .map_err(WeightedCapacityError::Solve)?;
    if fit.coefficients.len() != p {
        return Err(WeightedCapacityError::Shape(DesignShape::SolveWidth));
    }
    let bread = invert_square(&cross_products.xtwx, p).ok_or(WeightedCapacityError::Singular)?;
    Ok(Solved { coefficients: fit.coefficients, diagnostics: fit.diagnostics, bread })
}

/// Solve in coordinates centered on the weighted column means.
///
/// The non-constant coefficients solve `C β = c` on the centered co-moments, and the
/// constant column's coefficient is `(ȳ − Σ m_j β_j) / level`. The inverse Gram (the
/// bread of the sandwich and of the leverage) is `B' · blockdiag(C⁻¹, 1/Σw) · B` with
/// `B = A⁻¹`, so no step forms the raw Gram or inverts a matrix whose conditioning is
/// the square of the centered one's.
fn solve_centered(centered: &CenteredSystem, p: usize) -> Result<Solved, WeightedCapacityError> {
    let q = p - 1;
    let k = centered.constant;
    let factor = cholesky_spd(&centered.comoment, q).ok_or(WeightedCapacityError::Singular)?;
    let reduced_y = forward_substitute(&factor, q, &centered.cross)?;
    let mut ones = zeros(q, "unit weights")?;
    ones.fill(1.0);
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_wls(&factor, q, q, &reduced_y, &ones, &FaerBackend, &mut workspace)
        .map_err(WeightedCapacityError::Solve)?;
    if fit.coefficients.len() != q {
        return Err(WeightedCapacityError::Shape(DesignShape::SolveWidth));
    }
    let inverse = invert_square(&centered.comoment, q).ok_or(WeightedCapacityError::Singular)?;
    // Position of column `j` among the non-constant columns.
    let slot = |j: usize| if j < k { j } else { j - 1 };

    let mut coefficients = zeros(p, "centered coefficients")?;
    let mut shift = 0.0;
    for j in (0..p).filter(|&j| j != k) {
        coefficients[j] = fit.coefficients[slot(j)];
        shift += centered.mean[j] * coefficients[j];
    }
    coefficients[k] = (centered.mean_y - shift) / centered.level;

    // `B = A⁻¹`: x̃_j = x_j − (m_j / level)·x_k for j ≠ k, and x̃_k = x_k / level.
    let mut b = zeros(p * p, "centered basis change")?;
    for j in (0..p).filter(|&j| j != k) {
        b[j * p + j] = 1.0;
        b[j * p + k] = -centered.mean[j] / centered.level;
    }
    b[k * p + k] = 1.0 / centered.level;
    // `blockdiag(C⁻¹, 1/Σw)` in the full coordinate order.
    let mut middle = zeros(p * p, "centered inverse")?;
    for i in (0..p).filter(|&i| i != k) {
        for j in (0..p).filter(|&j| j != k) {
            middle[i * p + j] = inverse[slot(i) * q + slot(j)];
        }
    }
    middle[k * p + k] = 1.0 / centered.n_eff;
    // `bread = B' · middle · B`.
    let mut transposed = zeros(p * p, "centered basis change transpose")?;
    for i in 0..p {
        for j in 0..p {
            transposed[i * p + j] = b[j * p + i];
        }
    }
    let mut product = zeros(p * p, "centered bread intermediate")?;
    multiply_square(&transposed, &middle, p, &mut product);
    drop(middle);
    drop(transposed);
    let mut bread = zeros(p * p, "centered bread")?;
    multiply_square(&product, &b, p, &mut bread);

    let reported = &fit.diagnostics;
    let diagnostics = FitDiagnostics::new(
        reported.rank + 1,
        reported.rcond,
        reported.backend,
        reported.grow_count,
    );
    Ok(Solved { coefficients, diagnostics, bread })
}

fn forward_substitute(
    factor: &[f64],
    p: usize,
    rhs: &[f64],
) -> Result<Vec<f64>, WeightedCapacityError> {
    let mut solved = zeros(p, "forward substitution")?;
    for (i, (factor_row, &b)) in factor.chunks_exact(p).zip(rhs).enumerate() {
        let diag = factor_row[i];
        if diag.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(WeightedCapacityError::Singular);
        }
        let acc = factor_row[..i].iter().zip(&solved[..i]).fold(b, |acc, (&l, &s)| acc - l * s);
        solved[i] = acc / diag;
    }
    Ok(solved)
}

/// `bread · meat · bread` for square row-major matrices.
pub(super) fn sandwich_product(
    bread: &[f64],
    meat: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, WeightedCapacityError> {
    let cells = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    let mut tmp = zeros(cells, "sandwich intermediate")?;
    multiply_square(bread, meat, ncols, &mut tmp);
    let mut out = zeros(cells, "robust covariance")?;
    multiply_square(&tmp, bread, ncols, &mut out);
    Ok(out)
}

fn multiply_square(left: &[f64], right: &[f64], n: usize, out: &mut [f64]) {
    for (out_row, left_row) in out.chunks_exact_mut(n).zip(left.chunks_exact(n)) {
        for (j, cell) in out_row.iter_mut().enumerate() {
            *cell = left_row.iter().zip(right.iter().skip(j).step_by(n)).map(|(a, b)| a * b).sum();
        }
    }
}

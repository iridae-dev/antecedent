//! Admission and deterministic construction of correlated family tables.

use super::{CANCEL_STRIDE, ChainError, MAX_CHAIN_CELLS};

// ------------------------------------------------------------- family tables

/// Most variables one joint normal table may carry: the covariance (`k²` cells) must fit within
/// [`MAX_CHAIN_CELLS`].
pub const MAX_TABLE_VARIABLES: usize = 1 << 14;

/// `SplitMix64` stream, so a seed reproduces the table on every platform.
pub(super) struct SeedStream(pub(super) u64);

impl SeedStream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / 9_007_199_254_740_992.0
    }

    pub(super) fn normal_pair(&mut self) -> (f64, f64) {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        (radius * angle.cos(), radius * angle.sin())
    }
}

/// Lower-triangular `L` with `L Lᵀ = covariance`, allowing zero-variance
/// directions; `None` when the matrix is not positive semi-definite.
fn semidefinite_factor(covariance: &[f64], n: usize) -> Option<Vec<f64>> {
    // Factor in correlation coordinates: a variable's units cannot decide
    // whether its variance is zero or an impossible covariance is admissible.
    let scales: Vec<f64> = (0..n).map(|i| covariance[i * n + i].sqrt()).collect();
    if scales.iter().any(|scale| !scale.is_finite()) {
        return None;
    }
    let degenerate = 1e-13;
    let slack = 1e-9;
    let mut factor = vec![0.0; n * n];
    for column in 0..n {
        if antecedent_core::numerical_admission::is_cancelled() {
            return None;
        }
        let mut pivot: f64 = if scales[column] > 0.0 { 1.0 } else { 0.0 };
        for k in 0..column {
            pivot -= factor[column * n + k] * factor[column * n + k];
        }
        if pivot < -slack {
            return None;
        }
        let diagonal = if pivot <= degenerate { 0.0 } else { pivot.sqrt() };
        factor[column * n + column] = diagonal;
        for row in column + 1..n {
            if row % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                return None;
            }
            let covariance = covariance[row * n + column];
            let mut residual = if scales[row] == 0.0 || scales[column] == 0.0 {
                // A zero-variance variable has exactly zero covariance with
                // every variable; no absolute tolerance can establish that.
                if covariance != 0.0 {
                    return None;
                }
                0.0
            } else {
                covariance / scales[row] / scales[column]
            };
            if !residual.is_finite() {
                return None;
            }
            for k in 0..column {
                if k % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                    return None;
                }
                residual -= factor[row * n + k] * factor[column * n + k];
            }
            if diagonal == 0.0 {
                if residual.abs() > slack {
                    return None;
                }
            } else {
                factor[row * n + column] = residual / diagonal;
            }
        }
    }
    for row in 0..n {
        for column in 0..=row {
            factor[row * n + column] *= scales[row];
        }
    }
    Some(factor)
}

/// Correlation of two equal-weight mean-type estimators computed from
/// `rows_a` and `rows_b` rows of which `shared` are the same rows: the
/// covariance of the two means is `shared / (rows_a rows_b)` times the common
/// per-row variance, so the correlation is `shared / sqrt(rows_a rows_b)`.
///
/// # Errors
/// An empty row set, or more shared rows than either estimator uses.
pub fn shared_row_correlation(
    rows_a: usize,
    rows_b: usize,
    shared: usize,
) -> Result<f64, ChainError> {
    if rows_a == 0 || rows_b == 0 {
        return Err(ChainError::Shape("an estimator uses at least one row"));
    }
    if shared > rows_a.min(rows_b) {
        return Err(ChainError::Shape("shared rows cannot exceed the rows either estimator uses"));
    }
    Ok(shared as f64 / ((rows_a as f64) * (rows_b as f64)).sqrt())
}

/// Default ceiling for joint-table logical operations.
pub const MAX_MOMENT_MATCHED_TABLE_WORK: u64 = 4_000_000_000;

/// Refuse an oversized joint-table schedule before upstream table allocation.
/// Uses the same operation quote as [`moment_matched_normal_table_with_work_limit`].
pub fn admit_moment_matched_table_work(
    k: usize,
    rows: usize,
    limit: u64,
) -> Result<(), ChainError> {
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(ChainError::Cancelled);
    }
    if k == 0 || k > MAX_TABLE_VARIABLES {
        return Err(ChainError::Shape("invalid joint table dimension"));
    }
    if rows.checked_mul(k).is_none_or(|cells| cells > MAX_CHAIN_CELLS) {
        return Err(ChainError::Budget("table rows"));
    }
    if rows < k + 2 {
        return Err(ChainError::Shape("a moment-matched table needs more rows than variables"));
    }
    let work = (|| {
        let k = u64::try_from(k).ok()?;
        let rows = u64::try_from(rows).ok()?;
        let square = k.checked_mul(k)?;
        square
            .checked_mul(k)?
            .checked_mul(2)?
            .checked_add(square.checked_mul(rows)?.checked_mul(3)?)?
            .checked_add(square)
    })();
    if work.is_none_or(|work| work > limit) {
        return Err(ChainError::Budget("moment-matched table work"));
    }
    Ok(())
}

/// Joint normal draws whose sample mean and sample covariance (divisor
/// `rows - 1`) equal `means` and `covariance` up to rounding.
///
/// The base draws are independent standard normals, centered and whitened by
/// their own sample covariance before the declared covariance is applied, so a
/// downstream stage sees the declared dependence without Monte Carlo error in
/// the first two moments. Returns one column per variable. The default work ceiling
/// is [`MAX_MOMENT_MATCHED_TABLE_WORK`]; use
/// [`moment_matched_normal_table_with_work_limit`] to authorize a different ceiling.
/// A bound workflow context supplies cancellation through numerical admission.
///
/// # Errors
/// No variables or too many; a covariance of the wrong size, not finite, not
/// symmetric or not positive semidefinite; fewer than `variables + 2` rows or
/// whose `rows × variables` cells exceed [`MAX_CHAIN_CELLS`].
pub fn moment_matched_normal_table(
    means: &[f64],
    covariance: &[f64],
    rows: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, ChainError> {
    moment_matched_normal_table_with_work_limit(
        means,
        covariance,
        rows,
        seed,
        MAX_MOMENT_MATCHED_TABLE_WORK,
    )
}

/// Construct the same joint table with an explicit work ceiling and cancellation context.
/// Bind an `ExecutionContext` using `numerical_admission::with_context` to cancel.
#[allow(
    clippy::too_many_lines,
    reason = "one routine owns the bounded table construction and accounting"
)]
pub fn moment_matched_normal_table_with_work_limit(
    means: &[f64],
    covariance: &[f64],
    rows: usize,
    seed: u64,
    work_limit: u64,
) -> Result<Vec<Vec<f64>>, ChainError> {
    let k = means.len();
    if k == 0 || k > MAX_TABLE_VARIABLES || covariance.len() != k * k {
        return Err(ChainError::Shape(
            "a joint table needs one mean per variable and a k by k covariance",
        ));
    }
    if rows.checked_mul(k).is_none_or(|cells| cells > MAX_CHAIN_CELLS) {
        return Err(ChainError::Budget("table rows"));
    }
    if rows < k + 2 {
        return Err(ChainError::Shape("a moment-matched table needs more rows than variables"));
    }
    admit_moment_matched_table_work(k, rows, work_limit)?;
    for (i, value) in means.iter().chain(covariance).enumerate() {
        if i % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
            return Err(ChainError::Cancelled);
        }
        if !value.is_finite() {
            return Err(ChainError::NonFinite("joint table moment"));
        }
    }
    for row in 0..k {
        if antecedent_core::numerical_admission::is_cancelled() {
            return Err(ChainError::Cancelled);
        }
        if covariance[row * k + row] < 0.0 {
            return Err(ChainError::Operator("a variance is negative"));
        }
        for column in 0..row {
            if column % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                return Err(ChainError::Cancelled);
            }
            let left = covariance[row * k + column];
            let right = covariance[column * k + row];
            let row_sd = covariance[row * k + row].sqrt();
            let column_sd = covariance[column * k + column].sqrt();
            let symmetric = if row_sd == 0.0 || column_sd == 0.0 {
                left == 0.0 && right == 0.0
            } else {
                (left / row_sd / column_sd - right / row_sd / column_sd).abs() <= 1e-9
            };
            if !symmetric {
                return Err(ChainError::Operator("the covariance is not symmetric"));
            }
        }
    }
    let target = semidefinite_factor(covariance, k);
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(ChainError::Cancelled);
    }
    let target =
        target.ok_or(ChainError::Operator("the covariance is not positive semidefinite"))?;
    let mut stream = SeedStream(seed);
    let mut base: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut column = Vec::with_capacity(rows + 1);
        while column.len() < rows {
            if column.len() % CANCEL_STRIDE == 0
                && antecedent_core::numerical_admission::is_cancelled()
            {
                return Err(ChainError::Cancelled);
            }
            let (first, second) = stream.normal_pair();
            column.push(first);
            column.push(second);
        }
        column.truncate(rows);
        let mut sum = -0.0_f64;
        for (i, value) in column.iter().enumerate() {
            if i % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                return Err(ChainError::Cancelled);
            }
            sum += value;
        }
        let mean = sum / rows as f64;
        for (i, value) in column.iter_mut().enumerate() {
            if i % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                return Err(ChainError::Cancelled);
            }
            *value -= mean;
        }
        base.push(column);
    }
    let divisor = (rows - 1) as f64;
    let mut sample = vec![0.0; k * k];
    for i in 0..k {
        if antecedent_core::numerical_admission::is_cancelled() {
            return Err(ChainError::Cancelled);
        }
        for j in 0..=i {
            let mut cross = -0.0_f64;
            for (row, (a, b)) in base[i].iter().zip(&base[j]).enumerate() {
                if row % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled()
                {
                    return Err(ChainError::Cancelled);
                }
                cross += a * b;
            }
            let cross = cross / divisor;
            sample[i * k + j] = cross;
            sample[j * k + i] = cross;
        }
    }
    // The base draws are PD with probability one; a zero pivot means the seed
    // produced collinear columns and the table cannot be whitened.
    let whitener = semidefinite_factor(&sample, k);
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(ChainError::Cancelled);
    }
    let whitener = whitener
        .filter(|factor| (0..k).all(|i| factor[i * k + i] > 0.0))
        .ok_or(ChainError::Operator("the base draws are degenerate; change the seed"))?;
    let mut table = vec![Vec::with_capacity(rows); k];
    let mut whitened = vec![0.0; k];
    for row in 0..rows {
        if row % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
            return Err(ChainError::Cancelled);
        }
        for i in 0..k {
            let mut value = base[i][row];
            for j in 0..i {
                if j % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                    return Err(ChainError::Cancelled);
                }
                value -= whitener[i * k + j] * whitened[j];
            }
            whitened[i] = value / whitener[i * k + i];
        }
        for i in 0..k {
            let mut value = means[i];
            for j in 0..=i {
                if j % CANCEL_STRIDE == 0 && antecedent_core::numerical_admission::is_cancelled() {
                    return Err(ChainError::Cancelled);
                }
                value += target[i * k + j] * whitened[j];
            }
            table[i].push(value);
        }
    }
    Ok(table)
}

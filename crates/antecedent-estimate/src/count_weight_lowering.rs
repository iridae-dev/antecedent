//! Count-weight lowering of an IID repeated-row bootstrap.
//!
//! A multiplicity is the same multiset as drawing that row again. The lowered fit is
//! unpenalized Gaussian least squares: frequency weights `c_i` solve the same normal equations
//! as the expanded resample. The tests check that fit, its diagnostics, its penalty flag, and
//! the repeated-row sample size against the indexed draw.
//!
//! Ridge, lasso, Huber, GLM families, a prior weight, cross-fitting, and matching stay on
//! indexed resampling. A penalty or a prior is not a repeated-row proof.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm_weighted,
};

use crate::error::EstimationError;

/// Largest integer every `f64` can store exactly.
///
/// A larger multiplicity would round to a different frequency weight, so it is not the
/// repeated-row count.
const MAX_EXACT_COUNT: u64 = 1 << 53;

/// Estimator whose IID bootstrap may replace repeated rows with multiplicity counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepeatedRowEstimator {
    /// Unpenalized identity-link Gaussian least squares.
    ///
    /// Frequency weights match the expanded resample on this module's fixture.
    GaussianLeastSquares,
    /// Fixed-λ ridge. The penalty is not a repeated-row proof.
    Ridge,
    /// Lasso. The penalty is not a repeated-row proof.
    Lasso,
    /// Huber M-estimate. The repeated-row scale uses every copy.
    Huber,
    /// Unpenalized binomial logit. Its deviance diagnostic counts each distinct row once.
    BinomialLogit,
    /// Unpenalized binomial probit. Its deviance diagnostic counts each distinct row once.
    BinomialProbit,
    /// Poisson log-link GLM. Its deviance diagnostic counts each distinct row once.
    Poisson,
    /// Negative binomial. The dispersion update is not a repeated-row proof.
    NegativeBinomial,
    /// Ridge-penalized binomial. The penalty is not a repeated-row proof.
    PenalizedBinomial,
    /// A prior or fractional weight. It is not an integer row multiplicity.
    PriorWeighted,
    /// Cross-fitted AIPW. A repeated row can land in two folds.
    CrossFitAipw,
    /// Cross-fitted DML. A repeated row can land in two folds.
    CrossFitDml,
    /// DR-learner. Its cross-fit stage can put a repeated row in two folds.
    DrLearner,
    /// Matching. A repeated control is another donor slot.
    Matching,
}

/// How one IID bootstrap replicate is presented to the estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResamplePath {
    /// Row indexes drawn with replacement.
    Indexed,
    /// Integer multiplicities of the original rows, applied as frequency weights.
    CountWeights,
}

/// Path used for `estimator`.
///
/// [`RepeatedRowEstimator::GaussianLeastSquares`] uses count weights. Every other estimator
/// uses indexed resampling.
#[must_use]
pub const fn resample_path(estimator: RepeatedRowEstimator) -> ResamplePath {
    match estimator {
        RepeatedRowEstimator::GaussianLeastSquares => ResamplePath::CountWeights,
        RepeatedRowEstimator::Ridge
        | RepeatedRowEstimator::Lasso
        | RepeatedRowEstimator::Huber
        | RepeatedRowEstimator::BinomialLogit
        | RepeatedRowEstimator::BinomialProbit
        | RepeatedRowEstimator::Poisson
        | RepeatedRowEstimator::NegativeBinomial
        | RepeatedRowEstimator::PenalizedBinomial
        | RepeatedRowEstimator::PriorWeighted
        | RepeatedRowEstimator::CrossFitAipw
        | RepeatedRowEstimator::CrossFitDml
        | RepeatedRowEstimator::DrLearner
        | RepeatedRowEstimator::Matching => ResamplePath::Indexed,
    }
}

/// Integer row multiplicity encoded by `weight`.
///
/// Returns [`None`] for a non-finite, negative, or fractional weight, and for an integer that
/// `f64` cannot store exactly. A prior weight and a sampling weight are not row counts.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a repeated-row count is a nonnegative integer that must round-trip through f64"
)]
pub fn integer_multiplicity(weight: f64) -> Option<u64> {
    if !weight.is_finite() || weight < 0.0 {
        return None;
    }
    let rounded = weight.round();
    if (weight - rounded).abs() > 0.0 {
        return None;
    }
    let count = rounded as u64;
    if count > MAX_EXACT_COUNT || (count as f64 - rounded).abs() > 0.0 {
        return None;
    }
    Some(count)
}

/// Multiplicity of each original row in an indexed with-replacement draw.
///
/// `indexes` are positions in `0..n_rows`. The returned vector has length `n_rows` and sums to
/// `indexes.len()` when every index is in range.
///
/// # Errors
///
/// `n_rows` is zero, an index is outside the original rows, or a multiplicity exceeds `u64`.
pub fn multiplicity_counts(n_rows: usize, indexes: &[usize]) -> Result<Vec<u64>, EstimationError> {
    if n_rows == 0 {
        return Err(EstimationError::data_msg("count bootstrap needs a positive row count"));
    }
    let mut counts = vec![0_u64; n_rows];
    for &index in indexes {
        if index >= n_rows {
            return Err(EstimationError::data_msg("resample index is outside the original rows"));
        }
        counts[index] = counts[index].checked_add(1).ok_or_else(|| {
            EstimationError::data_msg("resample multiplicity exceeds u64")
        })?;
    }
    Ok(counts)
}

/// Unpenalized Gaussian least squares on multiplicity counts.
#[derive(Clone, Debug)]
pub struct GaussianCountFit {
    /// Coefficients, in design-column order.
    pub coefficients: Vec<f64>,
    /// Residual sum of squares of the repeated rows, `Σ c_i e_i²`.
    pub rss: f64,
    /// Numerical rank of the frequency-weighted design.
    pub rank: usize,
    /// Reciprocal condition estimate of that design.
    pub rcond: Option<f64>,
    /// Least-squares backend name.
    pub backend: &'static str,
    /// Workspace growths of this solve.
    pub grow_count: u32,
    /// Whether the solve minimized a penalized objective.
    pub penalized: bool,
    /// Number of rows in the indexed resample, `Σ c_i`.
    ///
    /// This is the repeated-row sample size. It is not the Kish effective size of the
    /// multiplicities.
    pub repeated_row_sample_size: usize,
}

/// Fit unpenalized Gaussian least squares with frequency weights `counts`.
///
/// A zero count drops that row, which is the same multiset as an indexed draw that never
/// selected it. Rows with a positive count stay in their original order.
///
/// # Errors
///
/// The design shape does not match `counts`, every count is zero, a count is above the exact
/// `f64` integer range, the count total does not fit in `usize`, or the least-squares solve
/// fails.
pub fn fit_gaussian_count_weights(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    counts: &[u64],
) -> Result<GaussianCountFit, EstimationError> {
    if nrows == 0 || ncols == 0 {
        return Err(EstimationError::data_msg(
            "count-weight least squares needs a positive row count and column count",
        ));
    }
    if counts.len() != nrows {
        return Err(EstimationError::data_msg(
            "count-weight least squares needs one multiplicity per row",
        ));
    }
    if y.len() != nrows || x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(EstimationError::data_msg(
            "count-weight design does not match the row and column counts",
        ));
    }
    let repeated_row_sample_size = repeated_row_sample_size(counts)?;
    if repeated_row_sample_size == 0 {
        return Err(EstimationError::data_msg(
            "a count bootstrap with no selected rows has no repeated-row sample",
        ));
    }
    let design = frequency_design(x_colmajor, nrows, ncols, y, counts)?;
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_glm_weighted(
        GlmFamily::GaussianIdentity,
        GlmDesignRef {
            x_colmajor: &design.x,
            nrows: design.nrows,
            ncols,
            y: &design.y,
        },
        &design.weights,
        &FaerBackend,
        &mut workspace,
        &GlmOptions::default(),
    )?;
    Ok(GaussianCountFit {
        coefficients: fit.coefficients,
        rss: fit.deviance,
        rank: fit.diagnostics.rank,
        rcond: fit.diagnostics.rcond,
        backend: fit.diagnostics.backend,
        grow_count: fit.diagnostics.grow_count,
        penalized: fit.penalized,
        repeated_row_sample_size,
    })
}

fn repeated_row_sample_size(counts: &[u64]) -> Result<usize, EstimationError> {
    counts.iter().try_fold(0_usize, |total, &count| {
        let count = usize::try_from(count).map_err(|_| {
            EstimationError::data_msg("repeated-row count does not fit in a row index")
        })?;
        total.checked_add(count).ok_or_else(|| {
            EstimationError::data_msg("repeated-row count total exceeds addressable length")
        })
    })
}

fn frequency_weight(count: u64) -> Result<f64, EstimationError> {
    if count > MAX_EXACT_COUNT {
        return Err(EstimationError::data_msg(
            "repeated-row count exceeds the exact f64 integer range (2^53)",
        ));
    }
    Ok(count as f64)
}

struct FrequencyDesign {
    x: Vec<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    nrows: usize,
}

fn frequency_design(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    counts: &[u64],
) -> Result<FrequencyDesign, EstimationError> {
    let mut kept = Vec::new();
    for (row, &count) in counts.iter().enumerate() {
        if count > 0 {
            kept.push((row, frequency_weight(count)?));
        }
    }
    let m = kept.len();
    let mut x = vec![0.0; m.saturating_mul(ncols)];
    let mut outcome = vec![0.0; m];
    let mut weights = Vec::with_capacity(m);
    for (new_row, &(old_row, weight)) in kept.iter().enumerate() {
        outcome[new_row] = y[old_row];
        weights.push(weight);
        for col in 0..ncols {
            x[col * m + new_row] = x_colmajor[col * nrows + old_row];
        }
    }
    Ok(FrequencyDesign { x, y: outcome, weights, nrows: m })
}

#[cfg(test)]
mod tests {
    use antecedent_kernels::kish_ess;
    use antecedent_stats::{
        DenseLinearAlgebra, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LassoOptions,
        LeastSquaresFit, LeastSquaresWorkspace, MEstimateOptions, fit_glm, fit_glm_ridge,
        fit_glm_ridge_weighted, fit_glm_weighted, fit_huber_m, fit_lasso_with_ones_column,
        fit_ridge,
    };

    use super::{
        RepeatedRowEstimator, ResamplePath, fit_gaussian_count_weights, integer_multiplicity,
        multiplicity_counts, resample_path,
    };

    fn near(label: &str, left: f64, right: f64, tol: f64) {
        assert!(
            (left - right).abs() <= tol,
            "{label}: {left} vs {right} (tolerance {tol})"
        );
    }

    fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    /// Intercept plus `x = 0, 1, 2`, outcomes `0, 0, 3`, multiplicities `1, 1, 4`.
    ///
    /// The repeated-row normal equations give intercept `-4/7`, slope `12/7`, and residual
    /// sum of squares `12/7`.
    fn line_fixture() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let x = vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let y = vec![0.0, 0.0, 3.0];
        let counts = vec![1, 1, 4];
        (x, y, counts)
    }

    fn outlier_fixture() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let x = vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let y = vec![30.0, 0.0, 1.0];
        let counts = vec![1, 1, 4];
        (x, y, counts)
    }

    /// Intercept plus a binary treatment. Each arm contains both outcomes.
    fn binomial_fixture() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let y = vec![0.0, 1.0, 1.0, 0.0];
        let counts = vec![2, 1, 2, 1];
        (x, y, counts)
    }

    /// Intercept-only Poisson outcomes `0, 1, 8` with multiplicities `1, 5, 1`.
    fn poisson_fixture() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        (vec![1.0, 1.0, 1.0], vec![0.0, 1.0, 8.0], vec![1, 5, 1])
    }

    fn repeat_indexes(counts: &[u64]) -> Vec<usize> {
        let mut indexes = Vec::new();
        for (row, &count) in counts.iter().enumerate() {
            for _ in 0..usize::try_from(count).expect("fixture count fits usize") {
                indexes.push(row);
            }
        }
        indexes
    }

    fn gather(
        x: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
        indexes: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = indexes.len();
        let mut x_boot = vec![0.0; n * ncols];
        let mut y_boot = vec![0.0; n];
        for (row, &src) in indexes.iter().enumerate() {
            y_boot[row] = y[src];
            for col in 0..ncols {
                x_boot[col * n + row] = x[col * nrows + src];
            }
        }
        (x_boot, y_boot)
    }

    fn indexed_least_squares(
        x: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
        indexes: &[usize],
    ) -> LeastSquaresFit {
        let (x_boot, y_boot) = gather(x, nrows, ncols, y, indexes);
        let mut workspace = LeastSquaresWorkspace::default();
        FaerBackend
            .least_squares(&x_boot, indexes.len(), ncols, &y_boot, &mut workspace)
            .expect("indexed least squares")
    }

    #[test]
    fn gaussian_count_weights_match_the_indexed_repeated_row_fit() {
        let (x, y, counts) = line_fixture();
        let indexes = repeat_indexes(&counts);
        assert_eq!(multiplicity_counts(3, &indexes).expect("counts"), counts);
        let count_fit =
            fit_gaussian_count_weights(&x, 3, 2, &y, &counts).expect("count-weight fit");
        let indexed = indexed_least_squares(&x, 3, 2, &y, &indexes);

        let intercept = -4.0 / 7.0;
        let slope = 12.0 / 7.0;
        let rss = 12.0 / 7.0;
        near("count intercept", count_fit.coefficients[0], intercept, 1e-9);
        near("count slope", count_fit.coefficients[1], slope, 1e-9);
        near("count rss", count_fit.rss, rss, 1e-9);
        near("indexed intercept", indexed.coefficients[0], intercept, 1e-9);
        near("indexed slope", indexed.coefficients[1], slope, 1e-9);
        near("indexed rss", indexed.rss, rss, 1e-9);
        assert_eq!(count_fit.rank, indexed.rank);
        assert_eq!(count_fit.rank, 2);
        let count_rcond = count_fit.rcond.expect("count-weight rcond");
        let indexed_rcond = indexed.diagnostics.rcond.expect("indexed rcond");
        near("rcond", count_rcond, indexed_rcond, 1e-8);
        assert_eq!(count_fit.backend, indexed.diagnostics.backend);
        assert_eq!(count_fit.backend, "faer");
        assert_eq!(count_fit.grow_count, indexed.diagnostics.grow_count);
        assert!(!count_fit.penalized);
        assert_eq!(count_fit.repeated_row_sample_size, indexes.len());
        assert_eq!(count_fit.repeated_row_sample_size, 6);
        let kish = kish_ess(&[1.0, 1.0, 4.0]);
        near("kish of the multiplicities", kish, 2.0, 1e-12);
        assert!(
            count_fit.repeated_row_sample_size as f64 > kish + 1.0,
            "the repeated-row sample size is the indexed length, not the Kish size"
        );

        let mut workspace = LeastSquaresWorkspace::default();
        let distinct =
            FaerBackend.least_squares(&x, 3, 2, &y, &mut workspace).expect("distinct-row fit");
        let shift = (distinct.coefficients[1] - slope).abs();
        assert!(shift > 0.1, "unequal counts must move the slope, shift {shift}");
        assert_eq!(
            resample_path(RepeatedRowEstimator::GaussianLeastSquares),
            ResamplePath::CountWeights
        );
    }

    #[test]
    fn count_weights_refuse_an_empty_draw_and_an_inexact_multiplicity() {
        let (x, y, _) = line_fixture();
        let empty = fit_gaussian_count_weights(&x, 3, 2, &y, &[0, 0, 0]);
        assert!(empty.is_err(), "an empty multiplicity draw has no repeated-row sample");
        let inexact = fit_gaussian_count_weights(&x, 3, 2, &y, &[1, 1, (1_u64 << 53) + 1]);
        let message = inexact.expect_err("2^53 + 1 is not an exact f64 weight").to_string();
        assert!(message.contains("exact f64"), "{message}");
    }

    #[test]
    fn multiplicity_counts_refuse_an_index_outside_the_sample() {
        let message = multiplicity_counts(3, &[0, 3]).expect_err("index 3").to_string();
        assert!(message.contains("outside the original rows"), "{message}");
    }

    #[test]
    fn ridge_penalty_stays_on_indexed_resampling() {
        let (x, y, counts) = line_fixture();
        let indexes = repeat_indexes(&counts);
        let (x_boot, y_boot) = gather(&x, 3, 2, &y, &indexes);
        let mut workspace = LeastSquaresWorkspace::default();
        let ridge = fit_ridge(
            &x_boot,
            indexes.len(),
            2,
            &y_boot,
            4.0,
            &FaerBackend,
            &mut workspace,
        )
        .expect("indexed ridge");
        let count_fit =
            fit_gaussian_count_weights(&x, 3, 2, &y, &counts).expect("count-weight least squares");
        let gap = max_abs_diff(&ridge.coefficients, &count_fit.coefficients);
        assert!(gap > 0.5, "ridge lambda 4 moves the fit by {gap}");
        assert_eq!(resample_path(RepeatedRowEstimator::Ridge), ResamplePath::Indexed);
    }

    #[test]
    fn lasso_penalty_stays_on_indexed_resampling() {
        let (x, y, counts) = line_fixture();
        let indexes = repeat_indexes(&counts);
        let (x_boot, y_boot) = gather(&x, 3, 2, &y, &indexes);
        let options = LassoOptions { lambda: 5.0, max_iter: 5_000, ..LassoOptions::default() };
        let lasso = fit_lasso_with_ones_column(&x_boot, indexes.len(), 2, &y_boot, &options)
            .expect("indexed lasso");
        let iterations = lasso.iterations;
        assert!(lasso.converged, "lasso iterations {iterations}");
        let mut coefficients = vec![lasso.intercept];
        coefficients.extend_from_slice(&lasso.coefficients);
        let count_fit =
            fit_gaussian_count_weights(&x, 3, 2, &y, &counts).expect("count-weight least squares");
        let gap = max_abs_diff(&coefficients, &count_fit.coefficients);
        assert!(gap > 0.5, "lasso lambda 5 moves the fit by {gap}");
        assert_eq!(resample_path(RepeatedRowEstimator::Lasso), ResamplePath::Indexed);
    }

    #[test]
    fn huber_stays_on_indexed_resampling() {
        let (x, y, counts) = outlier_fixture();
        let indexes = repeat_indexes(&counts);
        let (x_boot, y_boot) = gather(&x, 3, 2, &y, &indexes);
        let options = MEstimateOptions { max_iter: 200, ..MEstimateOptions::default() };
        let mut expanded_workspace = LeastSquaresWorkspace::default();
        let expanded = fit_huber_m(
            &x_boot,
            indexes.len(),
            2,
            &y_boot,
            &options,
            &FaerBackend,
            &mut expanded_workspace,
        )
        .expect("expanded Huber");
        let mut distinct_workspace = LeastSquaresWorkspace::default();
        let distinct = fit_huber_m(&x, 3, 2, &y, &options, &FaerBackend, &mut distinct_workspace)
            .expect("distinct-row Huber");
        assert!(expanded.converged && distinct.converged);
        let repetition = max_abs_diff(&expanded.coefficients, &distinct.coefficients);
        assert!(
            repetition > 0.5,
            "repeating rows changes Huber by {repetition}; a distinct-row fit is not the resample"
        );
        let mut ols_workspace = LeastSquaresWorkspace::default();
        let ols = FaerBackend
            .least_squares(&x_boot, indexes.len(), 2, &y_boot, &mut ols_workspace)
            .expect("expanded least squares");
        let robust = max_abs_diff(&expanded.coefficients, &ols.coefficients);
        assert!(robust > 0.5, "Huber leaves least squares by {robust}");
        assert_eq!(resample_path(RepeatedRowEstimator::Huber), ResamplePath::Indexed);
    }

    #[test]
    fn binomial_deviance_stays_on_indexed_resampling() {
        assert_binomial_family(GlmFamily::BinomialLogit, RepeatedRowEstimator::BinomialLogit);
        assert_binomial_family(GlmFamily::BinomialProbit, RepeatedRowEstimator::BinomialProbit);
    }

    fn assert_binomial_family(family: GlmFamily, estimator: RepeatedRowEstimator) {
        let (x, y, counts) = binomial_fixture();
        let indexes = repeat_indexes(&counts);
        let (x_boot, y_boot) = gather(&x, 4, 2, &y, &indexes);
        let weights: Vec<f64> = counts.iter().map(|&count| count as f64).collect();
        let options = GlmOptions::default().without_separation_ridge();
        let mut weighted_workspace = LeastSquaresWorkspace::default();
        let weighted = fit_glm_weighted(
            family,
            GlmDesignRef { x_colmajor: &x, nrows: 4, ncols: 2, y: &y },
            &weights,
            &FaerBackend,
            &mut weighted_workspace,
            &options,
        )
        .expect("weighted binomial");
        let mut indexed_workspace = LeastSquaresWorkspace::default();
        let indexed = fit_glm(
            family,
            GlmDesignRef { x_colmajor: &x_boot, nrows: indexes.len(), ncols: 2, y: &y_boot },
            &FaerBackend,
            &mut indexed_workspace,
            &options,
        )
        .expect("indexed binomial");
        assert!(weighted.converged && indexed.converged, "{family:?}");
        assert!(!weighted.penalized && !indexed.penalized, "{family:?}");
        let weighted_deviance = weighted.deviance;
        let indexed_deviance = indexed.deviance;
        let gap = (weighted_deviance - indexed_deviance).abs();
        assert!(
            gap > 1.0,
            "{family:?} deviance gap {gap}: distinct rows {weighted_deviance}, repeated rows \
             {indexed_deviance}"
        );
        assert_eq!(resample_path(estimator), ResamplePath::Indexed);
    }

    #[test]
    fn poisson_deviance_stays_on_indexed_resampling() {
        let (x, y, counts) = poisson_fixture();
        let indexes = repeat_indexes(&counts);
        let (x_boot, y_boot) = gather(&x, 3, 1, &y, &indexes);
        let weights: Vec<f64> = counts.iter().map(|&count| count as f64).collect();
        let options = GlmOptions::default().without_separation_ridge();
        let mut weighted_workspace = LeastSquaresWorkspace::default();
        let weighted = fit_glm_weighted(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: 3, ncols: 1, y: &y },
            &weights,
            &FaerBackend,
            &mut weighted_workspace,
            &options,
        )
        .expect("weighted Poisson");
        let mut indexed_workspace = LeastSquaresWorkspace::default();
        let indexed = fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x_boot, nrows: indexes.len(), ncols: 1, y: &y_boot },
            &FaerBackend,
            &mut indexed_workspace,
            &options,
        )
        .expect("indexed Poisson");
        assert!(weighted.converged && indexed.converged);
        let weighted_deviance = weighted.deviance;
        let indexed_deviance = indexed.deviance;
        let gap = (weighted_deviance - indexed_deviance).abs();
        assert!(
            gap > 1.0,
            "Poisson deviance gap {gap}: distinct rows {weighted_deviance}, repeated rows \
             {indexed_deviance}"
        );
        assert_eq!(resample_path(RepeatedRowEstimator::Poisson), ResamplePath::Indexed);
    }

    #[test]
    fn penalized_binomial_stays_on_indexed_resampling() {
        let (x, y, counts) = binomial_fixture();
        let indexes = repeat_indexes(&counts);
        let (x_boot, y_boot) = gather(&x, 4, 2, &y, &indexes);
        let weights: Vec<f64> = counts.iter().map(|&count| count as f64).collect();
        let options = GlmOptions::default();
        let lambda = 0.7;
        let mut weighted_workspace = LeastSquaresWorkspace::default();
        let weighted = fit_glm_ridge_weighted(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: 4, ncols: 2, y: &y },
            &weights,
            &FaerBackend,
            &mut weighted_workspace,
            &options,
            lambda,
        )
        .expect("weighted ridge logit");
        let mut indexed_workspace = LeastSquaresWorkspace::default();
        let indexed = fit_glm_ridge(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x_boot, nrows: indexes.len(), ncols: 2, y: &y_boot },
            &FaerBackend,
            &mut indexed_workspace,
            &options,
            lambda,
        )
        .expect("indexed ridge logit");
        assert!(weighted.converged && indexed.converged);
        assert!(weighted.penalized && indexed.penalized);
        let gap = (weighted.deviance - indexed.deviance).abs();
        assert!(
            gap > 0.5,
            "penalized deviance still counts distinct rows, gap {gap}"
        );
        assert_eq!(
            resample_path(RepeatedRowEstimator::PenalizedBinomial),
            ResamplePath::Indexed
        );
    }

    #[test]
    fn cross_fit_copies_stay_on_indexed_resampling() {
        let indexes = [0_usize, 0];
        let folds = [0_u8, 1];
        assert_eq!(indexes[0], indexes[1]);
        assert_ne!(folds[0], folds[1]);
        for estimator in [
            RepeatedRowEstimator::CrossFitAipw,
            RepeatedRowEstimator::CrossFitDml,
            RepeatedRowEstimator::DrLearner,
        ] {
            assert_eq!(resample_path(estimator), ResamplePath::Indexed);
        }
    }

    #[test]
    fn a_fractional_prior_weight_is_not_a_repeated_row_count() {
        assert_eq!(integer_multiplicity(2.0), Some(2));
        assert_eq!(integer_multiplicity(0.0), Some(0));
        assert_eq!(integer_multiplicity(0.5), None);
        assert_eq!(integer_multiplicity(1.5), None);
        assert_eq!(integer_multiplicity(-1.0), None);
        assert_eq!(integer_multiplicity(f64::NAN), None);
        assert_eq!(integer_multiplicity(f64::INFINITY), None);
        assert_eq!(resample_path(RepeatedRowEstimator::PriorWeighted), ResamplePath::Indexed);
    }

    #[test]
    fn matching_repeated_row_stays_on_indexed_resampling() {
        let repeated_controls = [0.0_f64, 0.0];
        let one_counted_control = [0.0_f64];
        assert_ne!(repeated_controls.len(), one_counted_control.len());
        assert_eq!(resample_path(RepeatedRowEstimator::Matching), ResamplePath::Indexed);
    }

    #[test]
    fn only_gaussian_least_squares_lowers_to_count_weights() {
        use RepeatedRowEstimator::{
            BinomialLogit, BinomialProbit, CrossFitAipw, CrossFitDml, DrLearner,
            GaussianLeastSquares, Huber, Lasso, Matching, NegativeBinomial, PenalizedBinomial,
            Poisson, PriorWeighted, Ridge,
        };
        assert_eq!(resample_path(GaussianLeastSquares), ResamplePath::CountWeights);
        for estimator in [
            Ridge,
            Lasso,
            Huber,
            BinomialLogit,
            BinomialProbit,
            Poisson,
            NegativeBinomial,
            PenalizedBinomial,
            PriorWeighted,
            CrossFitAipw,
            CrossFitDml,
            DrLearner,
            Matching,
        ] {
            assert_eq!(resample_path(estimator), ResamplePath::Indexed);
        }
    }
}

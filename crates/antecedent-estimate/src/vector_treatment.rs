//! B4 joint vector-treatment coefficients: several treatments estimated *together* by one
//! ordinary-least-squares regression that shares ONE adjustment set and ONE row snapshot.
//!
//! # Contract
//!
//! * **Model:** `y = b0 + A g + T b + e`, where `A` is the shared adjustment block and `T` the
//!   `k >= 2` treatment columns. The reported coefficient vector is `b`, in the order the
//!   treatments are supplied, each coefficient carrying its declared name.
//! * **Covariance:** the full `k x k` covariance of `b`, off-diagonals included, taken from the
//!   `(X'X)^-1` block of the one joint fit. [`VectorCovariance::ModelBased`] is
//!   `sigma^2 (X'X)^-1` with `sigma^2 = RSS / (n - p)`; the heteroskedasticity-robust sandwiches
//!   `(X'X)^-1 [sum_i w_i e_i^2 x_i x_i'] (X'X)^-1` use `w_i = 1` (HC0), `n / (n - p)` (HC1),
//!   `1 / (1 - h_i)` (HC2) or `1 / (1 - h_i)^2` (HC3). A cluster-robust option is deliberately
//!   not offered here.
//! * **Contrasts:** a declared [`Contrast`] is a weighted combination of the named coefficients.
//!   Its variance is `w' V w` and therefore uses the off-diagonals of `V`; the result also
//!   reports the *naive* standard error `sqrt(sum w_i^2 V_ii)` that wrongly treats the
//!   coefficients as independent, so the size of the correction is visible. Raw p-values of the
//!   declared contrasts get a Holm step-down adjustment across the declared family.
//! * **Joint test:** the Wald chi-square of `H0: all treatment coefficients are zero`
//!   ([`VECTOR_TREATMENT_NULL`]); [`VectorTreatmentFit::wald_test`] runs the same statistic for
//!   any declared set of contrasts.
//! * **Refusals (namespace `vector_treatment`):** a treatment that declares a different
//!   adjustment set or row snapshot than the shared one, a different row count, a treatment
//!   without variation, a rank-deficient or collinear design, and a degenerate covariance all
//!   refuse with a typed detail; nothing is dropped silently.
//!
//! # What a result does not say
//!
//! All p-values are asymptotic (standard normal / chi-square). Their calibration (coverage,
//! Type I error, power) is *unmeasured* ([`CalibrationStatus::Unmeasured`]); no coverage claim is
//! made and no interval is produced. The fit is a regression-coefficient estimate, not an
//! identification statement: the caller owns that the shared adjustment set is valid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::reason_code;
use antecedent_kernels::erfc;
use antecedent_stats::gamma_q;

use crate::effect_constancy::{CalibrationStatus, holm_adjust};
use crate::error::EstimationError;

/// The frozen null of the joint Wald test.
pub const VECTOR_TREATMENT_NULL: &str = "all treatment coefficients are zero";

/// Inference claim of the route: asymptotic Wald p-values with unmeasured calibration.
pub const VECTOR_TREATMENT_INFERENCE_CLAIM: &str = "asymptotic_wald_calibration_unmeasured";

/// Most treatments one joint fit accepts.
pub const VECTOR_TREATMENT_MAX_TREATMENTS: usize = 64;

/// Name reserved for the intercept; no column may use it.
pub const VECTOR_TREATMENT_INTERCEPT: &str = "(intercept)";

/// Relative tolerance of the Gram--Schmidt rank test.
const RANK_TOLERANCE: f64 = 1e-10;

/// Smallest `1 - h_i` the HC2/HC3 corrections accept.
const LEVERAGE_TOLERANCE: f64 = 1e-10;

/// Covariance estimator of the coefficient vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum VectorCovariance {
    /// `sigma^2 (X'X)^-1`, homoskedastic.
    #[default]
    ModelBased,
    /// HC0 sandwich (no finite-sample correction).
    Hc0,
    /// HC1 sandwich (`n / (n - p)` scaling).
    Hc1,
    /// HC2 leverage-corrected sandwich.
    Hc2,
    /// HC3 leverage-corrected sandwich.
    Hc3,
}

impl VectorCovariance {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelBased => "model_based",
            Self::Hc0 => "hc0",
            Self::Hc1 => "hc1",
            Self::Hc2 => "hc2",
            Self::Hc3 => "hc3",
        }
    }
}

/// A named numeric column.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedColumn {
    /// Unique column name.
    pub name: String,
    /// One value per row of the snapshot.
    pub values: Vec<f64>,
}

/// One treatment of the vector, with the adjustment set and snapshot it was prepared under.
#[derive(Clone, Debug, PartialEq)]
pub struct TreatmentColumn {
    /// Unique treatment name (the coefficient name).
    pub name: String,
    /// One value per row of the snapshot.
    pub values: Vec<f64>,
    /// Names of the adjustment columns this treatment was declared against.
    pub adjustment_set: Vec<String>,
    /// Identity of the row snapshot this treatment column was taken from.
    pub row_snapshot: String,
}

/// Joint fit input: one outcome, one shared adjustment block, one snapshot, `k >= 2` treatments.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTreatmentInput {
    /// Outcome, one value per row of the snapshot.
    pub outcome: Vec<f64>,
    /// Identity of the shared row snapshot.
    pub row_snapshot: String,
    /// The shared adjustment columns (may be empty).
    pub adjustment: Vec<NamedColumn>,
    /// The treatments, in coefficient order.
    pub treatments: Vec<TreatmentColumn>,
}

/// A declared linear contrast of the named treatment coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct Contrast {
    /// Unique contrast name.
    pub name: String,
    /// `(coefficient name, weight)` pairs; unlisted coefficients have weight zero.
    pub weights: Vec<(String, f64)>,
}

/// Options of the joint fit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorTreatmentOptions {
    /// Covariance estimator.
    pub covariance: VectorCovariance,
    /// Declared contrasts, reported in this order.
    pub contrasts: Vec<Contrast>,
}

/// One treatment coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct CoefficientEstimate {
    /// Coefficient (treatment) name.
    pub name: String,
    /// OLS point estimate.
    pub estimate: f64,
    /// Square root of the covariance diagonal entry.
    pub standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
}

/// One declared contrast of the coefficient vector.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastEstimate {
    /// Contrast name.
    pub name: String,
    /// `w' b`.
    pub estimate: f64,
    /// `sqrt(w' V w)`, using the off-diagonals of the covariance.
    pub standard_error: f64,
    /// `sqrt(sum w_i^2 V_ii)`: the standard error that wrongly treats the coefficients as
    /// independent, reported so the correction is visible.
    pub naive_independent_standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value across the declared contrast family.
    pub p_holm: f64,
}

/// A Wald chi-square test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointWald {
    /// `(R b)' (R V R')^-1 (R b)`.
    pub statistic: f64,
    /// Number of tested linear restrictions.
    pub degrees_of_freedom: usize,
    /// Asymptotic chi-square upper-tail p-value.
    pub p_value: f64,
}

/// The joint fit.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTreatmentFit {
    /// Treatment coefficient names, in input order.
    pub names: Vec<String>,
    /// Named coefficients, in input order.
    pub coefficients: Vec<CoefficientEstimate>,
    /// Row-major `k x k` covariance of the coefficient vector.
    pub covariance: Vec<f64>,
    /// Declared contrasts, in declaration order.
    pub contrasts: Vec<ContrastEstimate>,
    /// Wald test of the frozen null [`VECTOR_TREATMENT_NULL`].
    pub joint_wald: JointWald,
    /// Covariance estimator used.
    pub covariance_kind: VectorCovariance,
    /// Rows used.
    pub n_rows: usize,
    /// Residual degrees of freedom `n - p`.
    pub residual_df: usize,
    /// Residual variance `RSS / (n - p)` (reported for every covariance kind).
    pub residual_variance: f64,
    /// The frozen null.
    pub null: &'static str,
    /// Calibration of the joint inference.
    pub calibration: CalibrationStatus,
}

impl VectorTreatmentFit {
    /// Number of treatments `k`.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.names.len()
    }

    /// Covariance entry `(i, j)`; `None` outside the matrix.
    #[must_use]
    pub fn covariance_entry(&self, i: usize, j: usize) -> Option<f64> {
        let k = self.dimension();
        if i < k && j < k { self.covariance.get(i * k + j).copied() } else { None }
    }

    /// Wald test that every declared contrast is zero, using the full covariance.
    ///
    /// # Errors
    ///
    /// `invalid_argument` with `vector_treatment.empty_contrast`,
    /// `vector_treatment.unknown_contrast_coefficient`, `vector_treatment.duplicate_contrast_coefficient`,
    /// `vector_treatment.non_finite_value` or `vector_treatment.degenerate_covariance` (the
    /// contrasts are linearly dependent or the covariance is singular).
    pub fn wald_test(&self, contrasts: &[Contrast]) -> Result<JointWald, EstimationError> {
        if contrasts.is_empty() {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "vector_treatment.empty_contrast",
                "a Wald test needs at least one contrast",
            ));
        }
        let rows = contrasts
            .iter()
            .map(|contrast| weight_vector(&self.names, contrast))
            .collect::<Result<Vec<_>, _>>()?;
        let beta: Vec<f64> = self.coefficients.iter().map(|c| c.estimate).collect();
        wald_rows(&rows, &beta, &self.covariance)
    }
}

pub(crate) fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

pub(crate) fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// Two-sided normal p-value.
pub(crate) fn normal_two_sided(z: f64) -> f64 {
    erfc(z.abs() / std::f64::consts::SQRT_2)
}

fn chi_square_sf(statistic: f64, df: usize) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    gamma_q(count(df) * 0.5, statistic * 0.5)
}

/// The design: intercept, shared adjustment columns, then the treatments (column-major).
struct Design {
    names: Vec<String>,
    cols: Vec<Vec<f64>>,
    first_treatment: usize,
}

fn build_design(input: &VectorTreatmentInput) -> Design {
    let n = input.outcome.len();
    let mut names = vec![VECTOR_TREATMENT_INTERCEPT.to_string()];
    let mut cols = vec![vec![1.0; n]];
    for column in &input.adjustment {
        names.push(column.name.clone());
        cols.push(column.values.clone());
    }
    let first_treatment = cols.len();
    for treatment in &input.treatments {
        names.push(treatment.name.clone());
        cols.push(treatment.values.clone());
    }
    Design { names, cols, first_treatment }
}

pub(crate) fn validate_input(
    input: &VectorTreatmentInput,
    min_treatments: usize,
) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let n = input.outcome.len();
    let k = input.treatments.len();
    if n == 0 {
        return Err(refuse(invalid, "vector_treatment.too_few_rows", "the outcome has no rows"));
    }
    if k < min_treatments {
        return Err(refuse(
            invalid,
            "vector_treatment.too_few_treatments",
            "a vector treatment needs at least two treatments",
        ));
    }
    if k > VECTOR_TREATMENT_MAX_TREATMENTS {
        return Err(refuse(
            invalid,
            "vector_treatment.too_many_treatments",
            "more treatments than the joint fit accepts",
        ));
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let all_names = input
        .adjustment
        .iter()
        .map(|c| c.name.as_str())
        .chain(input.treatments.iter().map(|t| t.name.as_str()));
    for name in all_names {
        if name.is_empty() || name == VECTOR_TREATMENT_INTERCEPT || !seen.insert(name) {
            return Err(refuse(
                invalid,
                "vector_treatment.duplicate_name",
                "column names must be non-empty, unique and not the reserved intercept name",
            ));
        }
    }
    validate_shared(input)
}

/// Shared adjustment set, row snapshot, row count, finiteness and degrees of freedom.
fn validate_shared(input: &VectorTreatmentInput) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let incompatible = reason_code!("route_not_supported");
    let n = input.outcome.len();
    let k = input.treatments.len();
    let declared: BTreeSet<&str> = input.adjustment.iter().map(|c| c.name.as_str()).collect();
    for treatment in &input.treatments {
        let theirs: BTreeSet<&str> = treatment.adjustment_set.iter().map(String::as_str).collect();
        if theirs != declared {
            return Err(refuse(
                incompatible,
                "vector_treatment.adjustment_set_mismatch",
                &format!(
                    "treatment `{}` was declared against a different adjustment set than the \
                     shared one",
                    treatment.name
                ),
            ));
        }
        if treatment.row_snapshot != input.row_snapshot {
            return Err(refuse(
                incompatible,
                "vector_treatment.row_snapshot_mismatch",
                &format!("treatment `{}` comes from a different row snapshot", treatment.name),
            ));
        }
        if treatment.values.len() != n {
            return Err(refuse(
                incompatible,
                "vector_treatment.row_count_mismatch",
                &format!("treatment `{}` does not have one value per outcome row", treatment.name),
            ));
        }
    }
    for column in &input.adjustment {
        if column.values.len() != n {
            return Err(refuse(
                incompatible,
                "vector_treatment.row_count_mismatch",
                &format!("adjustment column `{}` does not have one value per row", column.name),
            ));
        }
    }
    let finite = |values: &[f64]| values.iter().all(|v| v.is_finite());
    let all_finite = finite(&input.outcome)
        && input.adjustment.iter().all(|c| finite(&c.values))
        && input.treatments.iter().all(|t| finite(&t.values));
    if !all_finite {
        return Err(refuse(
            invalid,
            "vector_treatment.non_finite_value",
            "outcome, adjustment and treatment values must be finite",
        ));
    }
    let p = 1 + input.adjustment.len() + k;
    if n <= p {
        return Err(refuse(
            invalid,
            "vector_treatment.too_few_rows",
            "the joint fit needs more rows than coefficients",
        ));
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Index of the first column that is a linear combination of the earlier ones.
fn first_dependent_column(cols: &[Vec<f64>]) -> Option<usize> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
    for (j, col) in cols.iter().enumerate() {
        let original = dot(col, col).sqrt();
        let mut v = col.clone();
        for _ in 0..2 {
            for q in &basis {
                let d = dot(&v, q);
                for (x, qx) in v.iter_mut().zip(q) {
                    *x -= d * qx;
                }
            }
        }
        let remaining = dot(&v, &v).sqrt();
        if !remaining.is_finite() || remaining <= RANK_TOLERANCE * original {
            return Some(j);
        }
        for x in &mut v {
            *x /= remaining;
        }
        basis.push(v);
    }
    None
}

fn check_design(design: &Design, input: &VectorTreatmentInput) -> Result<(), EstimationError> {
    let rank = reason_code!("design_rank_deficient");
    for treatment in &input.treatments {
        let lo = treatment.values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = treatment.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if hi - lo <= 0.0 {
            return Err(refuse(
                rank,
                "vector_treatment.treatment_without_variation",
                &format!("treatment `{}` is constant over the snapshot", treatment.name),
            ));
        }
    }
    if let Some(j) = first_dependent_column(&design.cols) {
        let name = &design.names[j];
        if j >= design.first_treatment {
            return Err(refuse(
                rank,
                "vector_treatment.collinear_treatments",
                &format!(
                    "treatment `{name}` is a linear combination of the intercept, the adjustment \
                     columns and the earlier treatments"
                ),
            ));
        }
        return Err(refuse(
            rank,
            "vector_treatment.rank_deficient_adjustment",
            &format!("column `{name}` is linearly dependent on the earlier design columns"),
        ));
    }
    Ok(())
}

/// Inverse of a row-major `p x p` matrix by Gauss--Jordan elimination with partial pivoting.
fn invert(matrix: &[f64], p: usize) -> Option<Vec<f64>> {
    let w = 2 * p;
    let mut aug = vec![0.0; p * w];
    for i in 0..p {
        for j in 0..p {
            aug[i * w + j] = matrix[i * p + j];
        }
        aug[i * w + p + i] = 1.0;
    }
    for col in 0..p {
        let mut pivot = col;
        for r in (col + 1)..p {
            if aug[r * w + col].abs() > aug[pivot * w + col].abs() {
                pivot = r;
            }
        }
        let value = aug[pivot * w + col];
        if !value.is_finite() || value.abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for j in 0..w {
                aug.swap(col * w + j, pivot * w + j);
            }
        }
        let scale = 1.0 / aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] *= scale;
        }
        for r in 0..p {
            if r != col {
                let factor = aug[r * w + col];
                for j in 0..w {
                    let delta = factor * aug[col * w + j];
                    aug[r * w + j] -= delta;
                }
            }
        }
    }
    let mut inverse = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            inverse[i * p + j] = aug[i * w + p + j];
        }
    }
    Some(inverse)
}

fn matmul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        for l in 0..p {
            let left = a[i * p + l];
            for j in 0..p {
                out[i * p + j] += left * b[l * p + j];
            }
        }
    }
    out
}

struct Ols {
    beta: Vec<f64>,
    ginv: Vec<f64>,
    residuals: Vec<f64>,
}

fn row_of(design: &Design, r: usize) -> Vec<f64> {
    design.cols.iter().map(|col| col[r]).collect()
}

fn fit_ols(design: &Design, y: &[f64]) -> Result<Ols, EstimationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::LeastSquaresSolve,
        || {
            let p = design.cols.len();
            let mut gram = vec![0.0; p * p];
            for i in 0..p {
                for j in 0..=i {
                    let value = dot(&design.cols[i], &design.cols[j]);
                    gram[i * p + j] = value;
                    gram[j * p + i] = value;
                }
            }
            let Some(ginv) = invert(&gram, p) else {
                return Err(refuse(
                    reason_code!("design_rank_deficient"),
                    "vector_treatment.rank_deficient_adjustment",
                    "the normal-equation matrix is singular",
                ));
            };
            let xty: Vec<f64> = design.cols.iter().map(|col| dot(col, y)).collect();
            let beta: Vec<f64> = (0..p).map(|i| dot(&ginv[i * p..(i + 1) * p], &xty)).collect();
            let residuals: Vec<f64> =
                (0..y.len()).map(|r| y[r] - dot(&row_of(design, r), &beta)).collect();
            if beta.iter().chain(&residuals).any(|v| !v.is_finite()) {
                return Err(refuse(
                    reason_code!("invalid_argument"),
                    "vector_treatment.non_finite_value",
                    "the fit overflowed finite precision",
                ));
            }
            Ok(Ols { beta, ginv, residuals })
        },
    )
}

fn full_covariance(
    design: &Design,
    ols: &Ols,
    kind: VectorCovariance,
) -> Result<Vec<f64>, EstimationError> {
    let p = design.cols.len();
    let n = ols.residuals.len();
    let df = count(n - p);
    if kind == VectorCovariance::ModelBased {
        let sigma2 = dot(&ols.residuals, &ols.residuals) / df;
        return Ok(ols.ginv.iter().map(|v| sigma2 * v).collect());
    }
    let mut meat = vec![0.0; p * p];
    for (r, e) in ols.residuals.iter().enumerate() {
        let x = row_of(design, r);
        let mut leverage = 0.0;
        for i in 0..p {
            leverage += x[i] * dot(&ols.ginv[i * p..(i + 1) * p], &x);
        }
        let weight = match kind {
            VectorCovariance::Hc0 | VectorCovariance::ModelBased => 1.0,
            VectorCovariance::Hc1 => count(n) / df,
            VectorCovariance::Hc2 | VectorCovariance::Hc3 => {
                let slack = 1.0 - leverage;
                if slack <= LEVERAGE_TOLERANCE {
                    return Err(refuse(
                        reason_code!("invalid_argument"),
                        "vector_treatment.leverage_one",
                        "an observation has leverage one, so HC2/HC3 are undefined",
                    ));
                }
                if kind == VectorCovariance::Hc2 { 1.0 / slack } else { 1.0 / (slack * slack) }
            }
        };
        let scale = weight * e * e;
        for i in 0..p {
            for j in 0..p {
                meat[i * p + j] += scale * x[i] * x[j];
            }
        }
    }
    Ok(matmul(&matmul(&ols.ginv, &meat, p), &ols.ginv, p))
}

/// Lower Cholesky factor of a row-major symmetric matrix; `None` unless positive definite.
fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                lower[i * n + i] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Some(lower)
}

/// `x' V^-1 x` from the Cholesky factor of `V`.
fn quadratic_form(lower: &[f64], n: usize, x: &[f64]) -> f64 {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= lower[i * n + k] * y[k];
        }
        y[i] = sum / lower[i * n + i];
    }
    y.iter().map(|v| v * v).sum()
}

fn degenerate(message: &str) -> EstimationError {
    refuse(reason_code!("invalid_argument"), "vector_treatment.degenerate_covariance", message)
}

/// Wald statistic of the restrictions `rows * beta = 0` under `covariance` (`k x k`).
fn wald_rows(
    rows: &[Vec<f64>],
    beta: &[f64],
    covariance: &[f64],
) -> Result<JointWald, EstimationError> {
    let k = beta.len();
    let m = rows.len();
    let rb: Vec<f64> = rows.iter().map(|row| dot(row, beta)).collect();
    let mut rvr = vec![0.0; m * m];
    for a in 0..m {
        for b in 0..m {
            let mut sum = 0.0;
            for i in 0..k {
                for j in 0..k {
                    sum += rows[a][i] * covariance[i * k + j] * rows[b][j];
                }
            }
            rvr[a * m + b] = sum;
        }
    }
    let Some(lower) = cholesky(&rvr, m) else {
        return Err(degenerate("the restriction covariance is singular or not positive definite"));
    };
    let statistic = quadratic_form(&lower, m, &rb);
    if !statistic.is_finite() {
        return Err(degenerate("the Wald statistic is not finite"));
    }
    Ok(JointWald { statistic, degrees_of_freedom: m, p_value: chi_square_sf(statistic, m) })
}

/// Weight vector of a contrast over the named coefficients.
pub(crate) fn weight_vector(
    names: &[String],
    contrast: &Contrast,
) -> Result<Vec<f64>, EstimationError> {
    let invalid = reason_code!("invalid_argument");
    if contrast.weights.is_empty() {
        return Err(refuse(
            invalid,
            "vector_treatment.empty_contrast",
            &format!("contrast `{}` lists no coefficients", contrast.name),
        ));
    }
    let mut weights = vec![0.0; names.len()];
    let mut used = vec![false; names.len()];
    for (name, weight) in &contrast.weights {
        let Some(index) = names.iter().position(|candidate| candidate == name) else {
            return Err(refuse(
                invalid,
                "vector_treatment.unknown_contrast_coefficient",
                &format!("contrast `{}` names the unknown coefficient `{name}`", contrast.name),
            ));
        };
        if used[index] {
            return Err(refuse(
                invalid,
                "vector_treatment.duplicate_contrast_coefficient",
                &format!("contrast `{}` lists coefficient `{name}` twice", contrast.name),
            ));
        }
        if !weight.is_finite() {
            return Err(refuse(
                invalid,
                "vector_treatment.non_finite_value",
                &format!("contrast `{}` has a non-finite weight", contrast.name),
            ));
        }
        used[index] = true;
        weights[index] = *weight;
    }
    Ok(weights)
}

fn evaluate_contrasts(
    names: &[String],
    beta: &[f64],
    covariance: &[f64],
    contrasts: &[Contrast],
) -> Result<Vec<ContrastEstimate>, EstimationError> {
    let k = names.len();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::with_capacity(contrasts.len());
    for contrast in contrasts {
        if contrast.name.is_empty() || !seen.insert(contrast.name.as_str()) {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "vector_treatment.duplicate_name",
                "contrast names must be non-empty and unique",
            ));
        }
        let w = weight_vector(names, contrast)?;
        let estimate = dot(&w, beta);
        let mut variance = 0.0;
        let mut naive = 0.0;
        for i in 0..k {
            naive += w[i] * w[i] * covariance[i * k + i];
            for j in 0..k {
                variance += w[i] * covariance[i * k + j] * w[j];
            }
        }
        if !(variance.is_finite() && variance > 0.0 && naive.is_finite() && naive > 0.0) {
            return Err(degenerate(&format!(
                "contrast `{}` has a non-positive or non-finite variance",
                contrast.name
            )));
        }
        let standard_error = variance.sqrt();
        let z = estimate / standard_error;
        out.push(ContrastEstimate {
            name: contrast.name.clone(),
            estimate,
            standard_error,
            naive_independent_standard_error: naive.sqrt(),
            z,
            p_value: normal_two_sided(z),
            p_holm: 0.0,
        });
    }
    let raw: Vec<f64> = out.iter().map(|c| c.p_value).collect();
    for (contrast, adjusted) in out.iter_mut().zip(holm_adjust(&raw)) {
        contrast.p_holm = adjusted;
    }
    Ok(out)
}

/// Fit the vector of treatment coefficients jointly.
///
/// # Errors
///
/// `route_not_supported` with `vector_treatment.adjustment_set_mismatch`,
/// `vector_treatment.row_snapshot_mismatch` or `vector_treatment.row_count_mismatch` when the
/// treatments do not share one adjustment set, snapshot and row count;
/// `design_rank_deficient` with `vector_treatment.treatment_without_variation`,
/// `vector_treatment.collinear_treatments` or `vector_treatment.rank_deficient_adjustment`;
/// `invalid_argument` with `vector_treatment.too_few_treatments`,
/// `vector_treatment.too_many_treatments`, `vector_treatment.too_few_rows`,
/// `vector_treatment.duplicate_name`, `vector_treatment.non_finite_value`,
/// `vector_treatment.leverage_one`, `vector_treatment.degenerate_covariance`, or a contrast
/// detail (`vector_treatment.empty_contrast`, `vector_treatment.unknown_contrast_coefficient`,
/// `vector_treatment.duplicate_contrast_coefficient`).
pub fn fit_vector_treatment(
    input: &VectorTreatmentInput,
    options: &VectorTreatmentOptions,
) -> Result<VectorTreatmentFit, EstimationError> {
    fit_joint(input, options, 2)
}

/// Joint OLS fit shared with the categorical (dummy-coded) route, which accepts one treatment
/// column (a two-level factor); the public vector entry point requires at least two.
pub(crate) fn fit_joint(
    input: &VectorTreatmentInput,
    options: &VectorTreatmentOptions,
    min_treatments: usize,
) -> Result<VectorTreatmentFit, EstimationError> {
    fit_joint_retained(input, options, min_treatments).map(|(fit, _, _)| fit)
}

/// Retain the full joint regression for compatible predictions.
pub(crate) fn fit_joint_retained(
    input: &VectorTreatmentInput,
    options: &VectorTreatmentOptions,
    min_treatments: usize,
) -> Result<(VectorTreatmentFit, Vec<f64>, Vec<f64>), EstimationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::AdjustedFit,
        || {
            validate_input(input, min_treatments)?;
            let design = build_design(input);
            check_design(&design, input)?;
            let ols = fit_ols(&design, &input.outcome)?;
            crate::adjustment_resume::record_model_fit();
            let full = full_covariance(&design, &ols, options.covariance)?;
            let n = input.outcome.len();
            let p = design.cols.len();
            let solved = Solved {
                names: input.treatments.iter().map(|t| t.name.clone()).collect(),
                beta: &ols.beta,
                full: &full,
                first_treatment: design.first_treatment,
                n,
                p,
                residual_variance: dot(&ols.residuals, &ols.residuals) / count(n - p),
            };
            let fit = assemble(solved, options)?;
            Ok((fit, ols.beta, full))
        },
    )
}

/// A solved joint fit awaiting its coefficient table, Wald test and contrasts.
struct Solved<'a> {
    names: Vec<String>,
    /// Full coefficient vector (intercept, adjustment, treatments).
    beta: &'a [f64],
    /// Row-major `p x p` covariance of the full coefficient vector.
    full: &'a [f64],
    first_treatment: usize,
    n: usize,
    p: usize,
    residual_variance: f64,
}

fn assemble(
    solved: Solved<'_>,
    options: &VectorTreatmentOptions,
) -> Result<VectorTreatmentFit, EstimationError> {
    let Solved { names, beta, full, first_treatment: t0, n, p, residual_variance } = solved;
    let k = names.len();
    let beta: Vec<f64> = beta[t0..].to_vec();
    let mut covariance = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            covariance[i * k + j] =
                0.5 * (full[(t0 + i) * p + t0 + j] + full[(t0 + j) * p + t0 + i]);
        }
    }
    let mut coefficients = Vec::with_capacity(k);
    for i in 0..k {
        let variance = covariance[i * k + i];
        if !(variance.is_finite() && variance > 0.0) {
            return Err(degenerate("a coefficient variance is not positive and finite"));
        }
        let standard_error = variance.sqrt();
        let z = beta[i] / standard_error;
        coefficients.push(CoefficientEstimate {
            name: names[i].clone(),
            estimate: beta[i],
            standard_error,
            z,
            p_value: normal_two_sided(z),
        });
    }
    let identity: Vec<Vec<f64>> =
        (0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let joint_wald = wald_rows(&identity, &beta, &covariance)?;
    let contrasts = evaluate_contrasts(&names, &beta, &covariance, &options.contrasts)?;
    Ok(VectorTreatmentFit {
        names,
        coefficients,
        covariance,
        contrasts,
        joint_wald,
        covariance_kind: options.covariance,
        n_rows: n,
        residual_df: n - p,
        residual_variance,
        null: VECTOR_TREATMENT_NULL,
        calibration: CalibrationStatus::Unmeasured,
    })
}

/// Sufficient statistics of a model-based joint fit: enough to recompute the coefficient vector
/// and its full covariance without the rows.
///
/// The design is `[intercept, adjustment columns..., treatment columns...]`; `gram` is the
/// row-major `p x p` matrix `X'X`, `xty` is `X'y`, and `rss` the residual sum of squares of the
/// producer's fit. A heteroskedasticity-robust covariance needs the rows themselves and has no
/// summary replay.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignSummary {
    /// Rows `n` of the snapshot.
    pub n_rows: usize,
    /// Index of the first treatment column (`1 + number of adjustment columns`).
    pub first_treatment: usize,
    /// Row-major `p x p` Gram matrix `X'X`.
    pub gram: Vec<f64>,
    /// `X'y`, length `p`.
    pub xty: Vec<f64>,
    /// Residual sum of squares `e'e` of the OLS fit.
    pub rss: f64,
}

fn gram_of(design: &Design) -> Vec<f64> {
    let p = design.cols.len();
    let mut gram = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let value = dot(&design.cols[i], &design.cols[j]);
            gram[i * p + j] = value;
            gram[j * p + i] = value;
        }
    }
    gram
}

/// Validate and fit once, then keep the sufficient statistics (shared with the categorical
/// route, which accepts one dummy column).
pub(crate) fn summarize_joint(
    input: &VectorTreatmentInput,
    min_treatments: usize,
) -> Result<DesignSummary, EstimationError> {
    validate_input(input, min_treatments)?;
    let design = build_design(input);
    check_design(&design, input)?;
    let ols = fit_ols(&design, &input.outcome)?;
    Ok(DesignSummary {
        n_rows: input.outcome.len(),
        first_treatment: design.first_treatment,
        gram: gram_of(&design),
        xty: design.cols.iter().map(|col| dot(col, &input.outcome)).collect(),
        rss: dot(&ols.residuals, &ols.residuals),
    })
}

/// Sufficient statistics of the joint fit of `input` (a model-based replay summary).
///
/// # Errors
///
/// The same refusals as [`fit_vector_treatment`] for the input and its design.
pub fn summarize_vector_design(
    input: &VectorTreatmentInput,
) -> Result<DesignSummary, EstimationError> {
    summarize_joint(input, 2)
}

/// Joint fit from sufficient statistics, bit-identical to the model-based fit of the rows the
/// summary came from.
///
/// # Errors
///
/// `route_not_supported` with `vector_treatment.summary_model_based_only` for a robust
/// covariance; `invalid_argument` with `vector_treatment.summary_inconsistent` for a malformed
/// summary, plus the refusals of [`fit_vector_treatment`] that do not need the rows.
pub fn fit_vector_treatment_from_summary(
    summary: &DesignSummary,
    names: &[String],
    options: &VectorTreatmentOptions,
) -> Result<VectorTreatmentFit, EstimationError> {
    fit_joint_from_summary(summary, names, options, 2)
}

pub(crate) fn fit_joint_from_summary(
    summary: &DesignSummary,
    names: &[String],
    options: &VectorTreatmentOptions,
    min_treatments: usize,
) -> Result<VectorTreatmentFit, EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let inconsistent =
        |message: &str| refuse(invalid, "vector_treatment.summary_inconsistent", message);
    if options.covariance != VectorCovariance::ModelBased {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "vector_treatment.summary_model_based_only",
            "a robust covariance needs the rows; a summary replays the model-based covariance only",
        ));
    }
    let k = names.len();
    if k < min_treatments {
        return Err(refuse(
            invalid,
            "vector_treatment.too_few_treatments",
            "a vector treatment needs at least two treatments",
        ));
    }
    if k > VECTOR_TREATMENT_MAX_TREATMENTS {
        return Err(refuse(
            invalid,
            "vector_treatment.too_many_treatments",
            "more treatments than the joint fit accepts",
        ));
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    if names.iter().any(|n| n.is_empty() || n == VECTOR_TREATMENT_INTERCEPT || !seen.insert(n)) {
        return Err(refuse(
            invalid,
            "vector_treatment.duplicate_name",
            "column names must be non-empty, unique and not the reserved intercept name",
        ));
    }
    let t0 = summary.first_treatment;
    let p = t0 + k;
    let n = summary.n_rows;
    if t0 == 0 || summary.gram.len() != p * p || summary.xty.len() != p {
        return Err(inconsistent("the Gram matrix or X'y does not match the declared columns"));
    }
    if n <= p {
        return Err(refuse(
            invalid,
            "vector_treatment.too_few_rows",
            "the joint fit needs more rows than coefficients",
        ));
    }
    let finite = summary.gram.iter().chain(&summary.xty).all(|v| v.is_finite())
        && summary.rss.is_finite()
        && summary.rss >= 0.0;
    if !finite {
        return Err(refuse(
            invalid,
            "vector_treatment.non_finite_value",
            "the summary must be finite with a non-negative residual sum of squares",
        ));
    }
    let Some(ginv) = invert(&summary.gram, p) else {
        return Err(refuse(
            reason_code!("design_rank_deficient"),
            "vector_treatment.rank_deficient_adjustment",
            "the normal-equation matrix is singular",
        ));
    };
    let beta: Vec<f64> = (0..p).map(|i| dot(&ginv[i * p..(i + 1) * p], &summary.xty)).collect();
    let sigma2 = summary.rss / count(n - p);
    let full: Vec<f64> = ginv.iter().map(|v| sigma2 * v).collect();
    if beta.iter().chain(&full).any(|v| !v.is_finite()) {
        return Err(refuse(
            invalid,
            "vector_treatment.non_finite_value",
            "the fit overflowed finite precision",
        ));
    }
    let solved = Solved {
        names: names.to_vec(),
        beta: &beta,
        full: &full,
        first_treatment: t0,
        n,
        p,
        residual_variance: sigma2,
    };
    assemble(solved, options)
}

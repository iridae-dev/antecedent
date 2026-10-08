//! B3 source-target mechanism discrepancy diagnostic: a Wald test of the null that the
//! conditional mechanism of one node `V` given its parents is the same in a source and a target
//! population, from comparable measurements.
//!
//! # Contract
//!
//! * **Null ([`MECHANISM_DISCREPANCY_NULL`]):** `E[V | pa]` is a linear regression with the *same*
//!   coefficient vector in the source and the target. The intercept is always fitted; whether it
//!   enters the comparison is declared ([`DiscrepancyOptions::compare_intercept`]).
//! * **Comparable measurements:** both populations must supply `V` and every parent on the same
//!   units and coordinates under the same declared measurement protocol id. A different node,
//!   variable set, unit, or protocol id refuses with `mechanism_discrepancy.incomparable_measurements`
//!   (`route_not_supported`); blank units or ids refuse the same way, because comparability is
//!   never assumed from silence. Parents are put in name order, so input order is irrelevant.
//! * **Independent samples:** the Wald statistic uses `V_source + V_target`, the sum of the two
//!   OLS covariance matrices `sigma^2 (X'X)^-1`. That is valid only for independent source and
//!   target samples. Declared shared units, row-identity overlap, or an unknown dependence refuse
//!   with `mechanism_discrepancy.dependence_unknown`.
//! * **Statistic:** `W = d' (V_s + V_t)^-1 d` with `d = b_target - b_source` over the compared
//!   coefficients, chi-square with `df` = the number of compared coefficients, upper tail through
//!   `gamma_q`. A per-coefficient breakdown reports `d_j`, its standard error
//!   `sqrt(V_s,jj + V_t,jj)`, the two-sided normal p-value and its Holm adjustment over the
//!   compared coefficients.
//! * **Power limit:** for every compared coefficient the result carries the minimal detectable
//!   difference `(z_{1-alpha/2} + z_{power}) * se_j` of a two-sided single-coefficient z test at the
//!   declared `alpha` and `power` (no multiplicity adjustment). The normal quantiles come from
//!   bisection on `erfc`. A non-rejection therefore reads "coefficient differences smaller than
//!   these were not detectable" (the `power_statement` field of [`MechanismDiscrepancyResult`]).
//!
//! # What a result does not say
//!
//! Non-rejection **never** certifies invariance: the `non_rejection_certifies_invariance` field of
//! [`MechanismDiscrepancyResult`] is always `false` and the caveat text is attached. The test
//! informs only a selection node on `V` itself (the `informs_selection_on` field): a rejection says the mechanism of `V` is not invariant, so such a node
//! cannot be excluded; non-rejection leaves it open. It is a linear-Gaussian-mean diagnostic: a
//! shift in a nonlinear or higher-moment feature of the mechanism may go undetected. Type I error
//! and power are *unmeasured* ([`CalibrationStatus::Unmeasured`]); no calibration claim is made.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::reason_code;
use antecedent_kernels::erfc;
use antecedent_stats::gamma_q;

use crate::effect_constancy::{CalibrationStatus, holm_adjust};
use crate::error::EstimationError;

/// The frozen null of the diagnostic.
pub const MECHANISM_DISCREPANCY_NULL: &str = "the conditional mechanism of the node given its \
                                              parents (one linear regression coefficient vector) \
                                              is the same in the source and the target population";

/// Inference claim of the route.
pub const MECHANISM_DISCREPANCY_INFERENCE_CLAIM: &str = "asymptotic_wald_calibration_unmeasured";

/// Population and selection alignment the diagnostic assumes and informs.
pub const MECHANISM_DISCREPANCY_ALIGNMENT: &str = "assumes two independent samples measured on \
    the same units and protocol; a rejection says the mechanism of the node is not invariant, so \
    a selection node on that node cannot be excluded; non-rejection leaves a selection node on \
    that node open and says nothing about selection nodes on other variables";

/// Statement attached to every result: non-rejection never certifies invariance.
pub const MECHANISM_DISCREPANCY_CAVEAT: &str = "failing to reject the null does not certify that \
    the mechanism is invariant between source and target; only differences larger than the \
    minimal detectable differences could have been detected";

/// Statement attached to every result about the unmeasured error rates.
pub const MECHANISM_DISCREPANCY_POWER_CAVEAT: &str = "Type I error and power of this test are \
    unmeasured; the minimal detectable differences are closed-form normal-theory values at the \
    observed standard errors, not a measured calibration";

/// Dependence assumption of the statistic.
pub const MECHANISM_DISCREPANCY_DEPENDENCE: &str = "source and target samples are independent \
    (no shared units); the covariance of the difference is the sum of the two OLS covariances";

/// Name of the intercept coefficient; no parent may use it.
pub const MECHANISM_DISCREPANCY_INTERCEPT: &str = "(intercept)";

/// Most parents one diagnostic accepts.
pub const MECHANISM_DISCREPANCY_MAX_PARENTS: usize = 64;

/// Smallest residual degrees of freedom `n - p` of either population.
pub const MECHANISM_DISCREPANCY_MIN_RESIDUAL_DF: usize = 2;

/// Normal quantile `z_{0.975}` to six decimals (reference constant; the code uses bisection).
pub const Z_ALPHA_TWO_SIDED_05: f64 = 1.959_964;

/// Normal quantile `z_{0.80}` to six decimals (reference constant; the code uses bisection).
pub const Z_POWER_80: f64 = 0.841_621;

/// Smallest accepted `alpha`, and `1 - ` the largest accepted `power`.
const LEVEL_FLOOR: f64 = 1e-12;

/// Relative pivot tolerance of the Cholesky rank test.
const RANK_TOLERANCE: f64 = 1e-10;

/// Residual sum of squares at or below this fraction of `y'y` counts as an exact fit.
const EXACT_FIT_TOLERANCE: f64 = 1e-12;

/// Relative tolerance of summary symmetry and of `X'X[0,0] = n`.
const SUMMARY_TOLERANCE: f64 = 1e-9;

/// One parent of the node: name and measurement unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParentSpec {
    /// Parent variable name.
    pub name: String,
    /// Unit / coordinate of the parent.
    pub unit: String,
}

/// What was measured and how: the comparability contract of one population.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismMeasurement {
    /// The node `V` whose mechanism is compared.
    pub node: String,
    /// Unit / coordinate of `V`.
    pub node_unit: String,
    /// The parents, in the order the population's columns are supplied.
    pub parents: Vec<ParentSpec>,
    /// Declared measurement protocol id.
    pub protocol_id: String,
}

/// One population's raw data.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationSample {
    /// Population label (for example `source`).
    pub label: String,
    /// The measurement contract.
    pub measurement: MechanismMeasurement,
    /// Values of `V`, one per row.
    pub outcome: Vec<f64>,
    /// One column per parent, in `measurement.parents` order.
    pub parent_values: Vec<Vec<f64>>,
    /// Optional row identities, used only to detect shared units; empty when not supplied.
    pub unit_ids: Vec<String>,
}

/// One population's sufficient statistics: all the test needs.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismSummary {
    /// Population label.
    pub label: String,
    /// The measurement contract.
    pub measurement: MechanismMeasurement,
    /// Row count.
    pub n: usize,
    /// Row-major `p x p` `X'X`, `p = 1 + parents`, design order: intercept, then the parents in
    /// `measurement.parents` order.
    pub xtx: Vec<f64>,
    /// `X'y`, same design order.
    pub xty: Vec<f64>,
    /// `y'y`.
    pub yty: f64,
}

/// Declared dependence between the two samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleDependence {
    /// Independent samples with no shared units: the only dependence the test runs under.
    Independent,
    /// The samples share units; the covariance of the difference is not the sum. Refused.
    SharedUnits,
    /// Dependence between the samples is not known. Refused.
    Unknown,
}

/// Options of the diagnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscrepancyOptions {
    /// Whether the intercept enters the compared coefficient vector.
    pub compare_intercept: bool,
    /// Level of the decisions, in `[1e-12, 1)`.
    pub alpha: f64,
    /// Target power of the minimal detectable difference, in `[0.5, 1 - 1e-12]`.
    pub power: f64,
    /// Declared dependence between the samples.
    pub dependence: SampleDependence,
}

/// Decision at the declared level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscrepancyConclusion {
    /// The null was not rejected. This does **not** establish invariance.
    NotRejected,
    /// The null was rejected: the mechanism of the node differs between source and target.
    Rejected,
}

/// The chi-square test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscrepancyTest {
    /// Wald statistic `d' (V_s + V_t)^-1 d`.
    pub statistic: f64,
    /// Number of compared coefficients.
    pub degrees_of_freedom: usize,
    /// Chi-square upper-tail probability.
    pub p_value: f64,
}

/// One population's fit, in canonical coefficient order (intercept, then parents by name).
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationFit {
    /// Population label.
    pub label: String,
    /// Row count.
    pub n: usize,
    /// Residual degrees of freedom `n - p`.
    pub residual_df: usize,
    /// Residual variance `RSS / (n - p)`.
    pub residual_variance: f64,
    /// OLS coefficients.
    pub coefficients: Vec<f64>,
    /// Their standard errors.
    pub standard_errors: Vec<f64>,
}

/// One compared coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct CoefficientDiscrepancy {
    /// Coefficient name (`(intercept)` or the parent name).
    pub name: String,
    /// Source coefficient.
    pub source: f64,
    /// Target coefficient.
    pub target: f64,
    /// `target - source`.
    pub difference: f64,
    /// `sqrt(V_s,jj + V_t,jj)`.
    pub standard_error: f64,
    /// `difference / standard_error`.
    pub z: f64,
    /// Two-sided normal p-value, unadjusted.
    pub p_value: f64,
    /// Holm-adjusted p-value over the compared coefficients.
    pub p_holm: f64,
    /// Whether the Holm-adjusted p-value is below `alpha`.
    pub rejected: bool,
    /// `(z_{1-alpha/2} + z_{power}) * standard_error`.
    pub minimal_detectable_difference: f64,
}

/// Result of the mechanism discrepancy diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismDiscrepancyResult {
    /// The frozen null.
    pub null: &'static str,
    /// The comparable measurement contract both populations share (parents in name order).
    pub measurement: MechanismMeasurement,
    /// Source fit.
    pub source: PopulationFit,
    /// Target fit.
    pub target: PopulationFit,
    /// Whether the intercept entered the comparison.
    pub compare_intercept: bool,
    /// Canonical coefficient names of the fits (intercept first, then parents by name).
    pub coefficient_names: Vec<String>,
    /// Chi-square test.
    pub test: DiscrepancyTest,
    /// Per-coefficient breakdown of the compared coefficients, in canonical order.
    pub coefficients: Vec<CoefficientDiscrepancy>,
    /// Level of the decisions.
    pub alpha: f64,
    /// Power of the minimal detectable differences.
    pub power: f64,
    /// `z_{1-alpha/2} + z_{power}`.
    pub detectability_factor: f64,
    /// Decision; non-rejection never certifies invariance.
    pub conclusion: DiscrepancyConclusion,
    /// Always `false`.
    pub non_rejection_certifies_invariance: bool,
    /// [`MECHANISM_DISCREPANCY_CAVEAT`].
    pub caveat: &'static str,
    /// What differences were detectable (the power limit), as text.
    pub power_statement: String,
    /// [`MECHANISM_DISCREPANCY_POWER_CAVEAT`].
    pub power_caveat: &'static str,
    /// [`MECHANISM_DISCREPANCY_ALIGNMENT`].
    pub alignment: &'static str,
    /// [`MECHANISM_DISCREPANCY_DEPENDENCE`].
    pub dependence_assumption: &'static str,
    /// The variables whose selection node this diagnostic informs: the node itself.
    pub informs_selection_on: Vec<String>,
    /// Calibration state: unmeasured.
    pub calibration: CalibrationStatus,
    /// [`MECHANISM_DISCREPANCY_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn chi_square_sf(statistic: f64, df: usize) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    gamma_q(count(df) * 0.5, statistic * 0.5)
}

/// Upper-tail probability of the standard normal at `z >= 0`.
fn normal_upper_tail(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// `z` with `P(Z > z) = tail`, for `tail` in `(0, 0.5]`, by bisection on `erfc` over `[0, 40]`
/// (200 halvings, far below the `f64` resolution).
#[must_use]
pub fn normal_upper_quantile(tail: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 40.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if normal_upper_tail(mid) > tail {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn validate_options(options: &DiscrepancyOptions) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    if !(options.alpha >= LEVEL_FLOOR && options.alpha < 1.0) {
        return Err(refuse(
            invalid,
            "mechanism_discrepancy.invalid_alpha",
            "alpha must lie in [1e-12, 1)",
        ));
    }
    if !(options.power >= 0.5 && options.power <= 1.0 - LEVEL_FLOOR) {
        return Err(refuse(
            invalid,
            "mechanism_discrepancy.invalid_power",
            "power must lie in [0.5, 1 - 1e-12]",
        ));
    }
    if options.dependence != SampleDependence::Independent {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "mechanism_discrepancy.dependence_unknown",
            "the covariance of the coefficient difference is the sum of the two OLS covariances \
             only for independent samples; shared units or unknown dependence are refused",
        ));
    }
    Ok(())
}

/// Sufficient statistics of one population.
///
/// # Errors
/// `invalid_argument` with `mechanism_discrepancy.too_many_parents`,
/// `mechanism_discrepancy.sample_too_small` (no rows), `mechanism_discrepancy.row_count_mismatch`
/// or `mechanism_discrepancy.non_finite_value`.
pub fn summarize_population(
    sample: &PopulationSample,
) -> Result<MechanismSummary, EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let n = sample.outcome.len();
    let k = sample.measurement.parents.len();
    if k > MECHANISM_DISCREPANCY_MAX_PARENTS {
        return Err(refuse(
            invalid,
            "mechanism_discrepancy.too_many_parents",
            "more parents than the diagnostic accepts",
        ));
    }
    if n == 0 {
        return Err(refuse(invalid, "mechanism_discrepancy.sample_too_small", "no rows"));
    }
    if sample.parent_values.len() != k
        || sample.parent_values.iter().any(|c| c.len() != n)
        || (!sample.unit_ids.is_empty() && sample.unit_ids.len() != n)
    {
        return Err(refuse(
            invalid,
            "mechanism_discrepancy.row_count_mismatch",
            "every parent column and the unit ids must have one value per outcome row",
        ));
    }
    let finite = |values: &[f64]| values.iter().all(|v| v.is_finite());
    if !(finite(&sample.outcome) && sample.parent_values.iter().all(|c| finite(c))) {
        return Err(refuse(
            invalid,
            "mechanism_discrepancy.non_finite_value",
            "outcome and parent values must be finite",
        ));
    }
    let ones = vec![1.0; n];
    let mut columns: Vec<&[f64]> = vec![ones.as_slice()];
    columns.extend(sample.parent_values.iter().map(Vec::as_slice));
    let p = k + 1;
    let mut xtx = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            xtx[i * p + j] = dot(columns[i], columns[j]);
        }
    }
    let xty: Vec<f64> = columns.iter().map(|c| dot(c, &sample.outcome)).collect();
    let yty = dot(&sample.outcome, &sample.outcome);
    if !(xtx.iter().chain(&xty).all(|v| v.is_finite()) && yty.is_finite()) {
        return Err(refuse(
            invalid,
            "mechanism_discrepancy.non_finite_value",
            "the summary statistics overflow at the supplied scale",
        ));
    }
    Ok(MechanismSummary {
        label: sample.label.clone(),
        measurement: sample.measurement.clone(),
        n,
        xtx,
        xty,
        yty,
    })
}

/// A validated summary with parents in name order.
struct Canonical {
    names: Vec<String>,
    measurement: MechanismMeasurement,
    n: usize,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
}

fn validate_summary_shape(s: &MechanismSummary) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let bad = |detail: &str, message: &str| refuse(invalid, detail, message);
    let k = s.measurement.parents.len();
    let p = k + 1;
    if k > MECHANISM_DISCREPANCY_MAX_PARENTS {
        return Err(bad("mechanism_discrepancy.too_many_parents", "too many parents"));
    }
    if s.xtx.len() != p * p || s.xty.len() != p {
        return Err(bad(
            "mechanism_discrepancy.inconsistent_summary",
            "the summary statistics do not match the declared parents",
        ));
    }
    if !(s.xtx.iter().chain(&s.xty).all(|v| v.is_finite()) && s.yty.is_finite() && s.yty >= 0.0) {
        return Err(bad(
            "mechanism_discrepancy.non_finite_value",
            "a summary statistic is not finite, or y'y is negative",
        ));
    }
    for i in 0..p {
        for j in 0..i {
            let (a, b) = (s.xtx[i * p + j], s.xtx[j * p + i]);
            if (a - b).abs() > SUMMARY_TOLERANCE * a.abs().max(b.abs()) {
                return Err(bad(
                    "mechanism_discrepancy.inconsistent_summary",
                    "X'X is not symmetric",
                ));
            }
        }
    }
    let n = count(s.n);
    if s.n < p + MECHANISM_DISCREPANCY_MIN_RESIDUAL_DF {
        return Err(bad(
            "mechanism_discrepancy.sample_too_small",
            "the sample leaves fewer than two residual degrees of freedom",
        ));
    }
    if (s.xtx[0] - n).abs() > SUMMARY_TOLERANCE * n {
        return Err(bad(
            "mechanism_discrepancy.inconsistent_summary",
            "the intercept sum of squares does not equal the row count",
        ));
    }
    let mut names = BTreeSet::new();
    for parent in &s.measurement.parents {
        if parent.name.trim().is_empty()
            || parent.name == MECHANISM_DISCREPANCY_INTERCEPT
            || !names.insert(parent.name.as_str())
        {
            return Err(bad(
                "mechanism_discrepancy.duplicate_name",
                "parent names must be non-blank, unique and not the reserved intercept name",
            ));
        }
    }
    Ok(())
}

fn canonicalize(s: &MechanismSummary) -> Result<Canonical, EstimationError> {
    validate_summary_shape(s)?;
    let k = s.measurement.parents.len();
    let p = k + 1;
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| s.measurement.parents[a].name.cmp(&s.measurement.parents[b].name));
    let mut perm = vec![0_usize];
    perm.extend(order.iter().map(|i| i + 1));
    let mut xtx = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            xtx[a * p + b] = s.xtx[perm[a] * p + perm[b]];
        }
    }
    let xty: Vec<f64> = perm.iter().map(|&i| s.xty[i]).collect();
    let parents: Vec<ParentSpec> =
        order.iter().map(|&i| s.measurement.parents[i].clone()).collect();
    let mut names = vec![MECHANISM_DISCREPANCY_INTERCEPT.to_owned()];
    names.extend(parents.iter().map(|q| q.name.clone()));
    Ok(Canonical {
        names,
        measurement: MechanismMeasurement { parents, ..s.measurement.clone() },
        n: s.n,
        xtx,
        xty,
        yty: s.yty,
    })
}

/// The measurement with parents in name order.
fn sorted_measurement(m: &MechanismMeasurement) -> MechanismMeasurement {
    let mut sorted = m.clone();
    sorted.parents.sort_by(|a, b| a.name.cmp(&b.name));
    sorted
}

fn check_comparable(
    ma: &MechanismMeasurement,
    mb: &MechanismMeasurement,
) -> Result<(), EstimationError> {
    let incomparable = reason_code!("route_not_supported");
    let detail = "mechanism_discrepancy.incomparable_measurements";
    let blank = |m: &MechanismMeasurement| {
        m.node.trim().is_empty()
            || m.node_unit.trim().is_empty()
            || m.protocol_id.trim().is_empty()
            || m.parents.iter().any(|q| q.unit.trim().is_empty())
    };
    if blank(ma) || blank(mb) {
        return Err(refuse(
            incomparable,
            detail,
            "the node, its units and the measurement protocol id must be declared",
        ));
    }
    let names = |m: &MechanismMeasurement| -> Vec<String> {
        m.parents.iter().map(|q| q.name.clone()).collect()
    };
    if ma.node != mb.node || names(ma) != names(mb) {
        return Err(refuse(
            incomparable,
            detail,
            "the populations do not supply the same node and parent variable set",
        ));
    }
    if ma.node_unit != mb.node_unit || ma.parents != mb.parents {
        return Err(refuse(incomparable, detail, "the populations measure on different units"));
    }
    if ma.protocol_id != mb.protocol_id {
        return Err(refuse(
            incomparable,
            detail,
            "the populations declare different measurement protocol ids",
        ));
    }
    Ok(())
}

/// Lower Cholesky factor of a row-major symmetric matrix; `None` unless every pivot exceeds
/// [`RANK_TOLERANCE`] times its diagonal entry (so rank deficiency is detected).
fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if sum.is_nan() || !sum.is_finite() || sum <= RANK_TOLERANCE * matrix[i * n + i] {
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

/// Solve `L y = b` (forward substitution).
fn forward_solve(lower: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= lower[i * n + k] * y[k];
        }
        y[i] = sum / lower[i * n + i];
    }
    y
}

/// Solve `L L' x = b`.
fn chol_solve(lower: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let y = forward_solve(lower, n, b);
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= lower[k * n + i] * x[k];
        }
        x[i] = sum / lower[i * n + i];
    }
    x
}

struct Ols {
    beta: Vec<f64>,
    covariance: Vec<f64>,
    residual_df: usize,
    sigma2: f64,
}

fn ols(c: &Canonical) -> Result<Ols, EstimationError> {
    let p = c.names.len();
    let lower = cholesky(&c.xtx, p).ok_or_else(|| {
        refuse(
            reason_code!("invalid_argument"),
            "mechanism_discrepancy.rank_deficient_design",
            "the design (intercept and parents) is rank deficient in a population",
        )
    })?;
    let beta = chol_solve(&lower, p, &c.xty);
    let mut inverse = vec![0.0; p * p];
    for j in 0..p {
        let mut unit = vec![0.0; p];
        unit[j] = 1.0;
        let column = chol_solve(&lower, p, &unit);
        for (i, v) in column.iter().enumerate() {
            inverse[i * p + j] = *v;
        }
    }
    let residual_df = c.n - p;
    let rss = (c.yty - dot(&beta, &c.xty)).max(0.0);
    if rss <= EXACT_FIT_TOLERANCE * c.yty {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mechanism_discrepancy.degenerate_covariance",
            "a population fits its mechanism exactly, so its sampling variance is not estimable",
        ));
    }
    let sigma2 = rss / count(residual_df);
    let covariance: Vec<f64> = inverse.iter().map(|v| sigma2 * v).collect();
    if !(beta.iter().chain(&covariance).all(|v| v.is_finite()) && sigma2.is_finite()) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mechanism_discrepancy.non_finite_value",
            "the fit overflows at the supplied scale",
        ));
    }
    Ok(Ols { beta, covariance, residual_df, sigma2 })
}

/// Wald statistic of `d` against `V_s + V_t` restricted to the compared coefficients.
fn joint_wald(compared: &[usize], p: usize, s: &Ols, t: &Ols) -> Result<f64, EstimationError> {
    let m = compared.len();
    let d: Vec<f64> = compared.iter().map(|&i| t.beta[i] - s.beta[i]).collect();
    let mut v = vec![0.0; m * m];
    for (a, &i) in compared.iter().enumerate() {
        for (b, &j) in compared.iter().enumerate() {
            v[a * m + b] = s.covariance[i * p + j] + t.covariance[i * p + j];
        }
    }
    let lower = cholesky(&v, m).ok_or_else(|| {
        refuse(
            reason_code!("invalid_argument"),
            "mechanism_discrepancy.degenerate_covariance",
            "the summed covariance of the coefficient difference is singular (for example, \
             both populations fit exactly)",
        )
    })?;
    let statistic: f64 = forward_solve(&lower, m, &d).iter().map(|y| y * y).sum();
    if !statistic.is_finite() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mechanism_discrepancy.non_finite_value",
            "the statistic overflows at the supplied scale",
        ));
    }
    Ok(statistic)
}

fn coefficient_rows(
    names: &[String],
    compared: &[usize],
    s: &Ols,
    t: &Ols,
    options: &DiscrepancyOptions,
    factor: f64,
) -> Vec<CoefficientDiscrepancy> {
    let p = names.len();
    let raw: Vec<(usize, f64, f64, f64, f64)> = compared
        .iter()
        .map(|&i| {
            let difference = t.beta[i] - s.beta[i];
            let se = (s.covariance[i * p + i] + t.covariance[i * p + i]).sqrt();
            let z = difference / se;
            (i, difference, se, z, erfc(z.abs() / std::f64::consts::SQRT_2))
        })
        .collect();
    let adjusted = holm_adjust(&raw.iter().map(|r| r.4).collect::<Vec<_>>());
    raw.into_iter()
        .zip(adjusted)
        .map(|((i, difference, se, z, p_value), p_holm)| CoefficientDiscrepancy {
            name: names[i].clone(),
            source: s.beta[i],
            target: t.beta[i],
            difference,
            standard_error: se,
            z,
            p_value,
            p_holm,
            rejected: p_holm < options.alpha,
            minimal_detectable_difference: factor * se,
        })
        .collect()
}

fn power_statement(rows: &[CoefficientDiscrepancy], options: &DiscrepancyOptions) -> String {
    let list: Vec<String> =
        rows.iter().map(|r| format!("{}: {}", r.name, r.minimal_detectable_difference)).collect();
    format!(
        "at alpha {} and power {} (two-sided single-coefficient z test, no multiplicity \
         adjustment), coefficient differences smaller than these were not detectable: {}",
        options.alpha,
        options.power,
        list.join(", ")
    )
}

fn population_fit(label: &str, c: &Canonical, fit: &Ols) -> PopulationFit {
    let p = c.names.len();
    PopulationFit {
        label: label.to_owned(),
        n: c.n,
        residual_df: fit.residual_df,
        residual_variance: fit.sigma2,
        coefficients: fit.beta.clone(),
        standard_errors: (0..p).map(|i| fit.covariance[i * p + i].sqrt()).collect(),
    }
}

/// Run the diagnostic on two populations' raw data.
///
/// Shared row identities (a non-empty `unit_ids` on both sides with a common id) refuse as
/// `mechanism_discrepancy.dependence_unknown` before anything is computed.
///
/// # Errors
/// See [`test_mechanism_discrepancy_from_summaries`] and [`summarize_population`].
pub fn test_mechanism_discrepancy(
    source: &PopulationSample,
    target: &PopulationSample,
    options: &DiscrepancyOptions,
) -> Result<MechanismDiscrepancyResult, EstimationError> {
    validate_options(options)?;
    if !source.unit_ids.is_empty() && !target.unit_ids.is_empty() {
        let ids: BTreeSet<&str> = source.unit_ids.iter().map(String::as_str).collect();
        if target.unit_ids.iter().any(|id| ids.contains(id.as_str())) {
            return Err(refuse(
                reason_code!("route_not_supported"),
                "mechanism_discrepancy.dependence_unknown",
                "source and target share units, so their estimates are not independent",
            ));
        }
    }
    let s = summarize_population(source)?;
    let t = summarize_population(target)?;
    test_mechanism_discrepancy_from_summaries(&s, &t, options)
}

/// Run the diagnostic on two populations' sufficient statistics (what a portable artifact
/// carries).
///
/// # Errors
/// `route_not_supported` with `mechanism_discrepancy.incomparable_measurements` (different node,
/// parent set, units or protocol id, or blank declarations) or
/// `mechanism_discrepancy.dependence_unknown`; `invalid_argument` with
/// `mechanism_discrepancy.invalid_alpha`, `mechanism_discrepancy.invalid_power`,
/// `mechanism_discrepancy.sample_too_small`, `mechanism_discrepancy.rank_deficient_design`,
/// `mechanism_discrepancy.degenerate_covariance`, `mechanism_discrepancy.non_finite_value`,
/// `mechanism_discrepancy.inconsistent_summary`, `mechanism_discrepancy.duplicate_name`,
/// `mechanism_discrepancy.too_many_parents` or `mechanism_discrepancy.no_compared_coefficients`.
pub fn test_mechanism_discrepancy_from_summaries(
    source: &MechanismSummary,
    target: &MechanismSummary,
    options: &DiscrepancyOptions,
) -> Result<MechanismDiscrepancyResult, EstimationError> {
    validate_options(options)?;
    check_comparable(
        &sorted_measurement(&source.measurement),
        &sorted_measurement(&target.measurement),
    )?;
    let s = canonicalize(source)?;
    let t = canonicalize(target)?;
    let fit_s = ols(&s)?;
    let fit_t = ols(&t)?;
    let p = s.names.len();
    let compared: Vec<usize> = (usize::from(!options.compare_intercept)..p).collect();
    if compared.is_empty() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mechanism_discrepancy.no_compared_coefficients",
            "the node has no parents and the intercept is not compared",
        ));
    }
    let statistic = joint_wald(&compared, p, &fit_s, &fit_t)?;
    let df = compared.len();
    let p_value = chi_square_sf(statistic, df);
    let factor = normal_upper_quantile(options.alpha * 0.5)
        + normal_upper_quantile((1.0 - options.power).min(0.5));
    let rows = coefficient_rows(&s.names, &compared, &fit_s, &fit_t, options, factor);
    Ok(MechanismDiscrepancyResult {
        null: MECHANISM_DISCREPANCY_NULL,
        informs_selection_on: vec![s.measurement.node.clone()],
        source: population_fit(&source.label, &s, &fit_s),
        target: population_fit(&target.label, &t, &fit_t),
        compare_intercept: options.compare_intercept,
        coefficient_names: s.names.clone(),
        test: DiscrepancyTest { statistic, degrees_of_freedom: df, p_value },
        alpha: options.alpha,
        power: options.power,
        detectability_factor: factor,
        conclusion: if p_value < options.alpha {
            DiscrepancyConclusion::Rejected
        } else {
            DiscrepancyConclusion::NotRejected
        },
        non_rejection_certifies_invariance: false,
        caveat: MECHANISM_DISCREPANCY_CAVEAT,
        power_statement: power_statement(&rows, options),
        power_caveat: MECHANISM_DISCREPANCY_POWER_CAVEAT,
        alignment: MECHANISM_DISCREPANCY_ALIGNMENT,
        dependence_assumption: MECHANISM_DISCREPANCY_DEPENDENCE,
        calibration: CalibrationStatus::Unmeasured,
        inference_claim: MECHANISM_DISCREPANCY_INFERENCE_CLAIM,
        coefficients: rows,
        measurement: s.measurement,
    })
}

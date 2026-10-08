//! B4 categorical treatments: ordered and unordered regimes, a declared reference level,
//! per-level and requested pairwise contrasts with the covariance of the dummy-coded
//! regression, Holm multiplicity over the declared family, and a declared monotonicity test for
//! ordered levels.
//!
//! # Contract
//!
//! * **Coding:** the regression is `y = b0 + A g + sum_{L != ref} b_L 1[level = L] + e` with the
//!   declared reference level omitted, fitted jointly through
//!   [`crate::vector_treatment`] (same `(X'X)^-1` covariance, off-diagonals included). The
//!   effect of level `L` is `b_L`, the reference has effect zero, and the effect of `to`
//!   relative to `from` is `b_to - b_from`; every contrast variance is `w' V w`.
//! * **Level order:** for [`LevelScale::Unordered`] the levels are put in canonical (sorted)
//!   order before any arithmetic, so permuting the declared level list gives bit-identical
//!   output. For [`LevelScale::Ordered`] the declared order is the scale and is part of the
//!   estimand.
//! * **Sparse or absent levels:** every declared level must have at least `min_level_rows`
//!   rows. A level with none (`categorical_treatment.absent_level`) or too few
//!   (`categorical_treatment.sparse_level`) refuses with the level named; a level is never
//!   silently dropped or merged, and a row with an undeclared label refuses too.
//! * **Multiplicity:** the declared family is the per-level contrasts against the reference
//!   plus the requested pairwise contrasts; their raw two-sided asymptotic normal p-values get
//!   a Holm step-down adjustment ([`CategoricalTreatmentFit::family_size`] contrasts).
//! * **Omnibus:** a Wald chi-square of `H0: every level has the reference's effect`.
//!
//! # Monotonicity test (ordered levels only)
//!
//! Along the declared order `l_0 < ... < l_{J-1}` let `d_j = effect(l_{j+1}) - effect(l_j)` be
//! the `m = J - 1` adjacent steps with standard errors from the full covariance.
//!
//! * Null, [`MonotonicityDirection::NonDecreasing`]: **every** `d_j >= 0` (the adjusted level
//!   effects are non-decreasing); alternative: some `d_j < 0`.
//! * Null, [`MonotonicityDirection::NonIncreasing`]: **every** `d_j <= 0`; alternative: some
//!   `d_j > 0`.
//! * Statistic: the oriented minimum step z-score `T = min_j s z_j`, `s = +1` (non-decreasing)
//!   or `-1` (non-increasing), `z_j = d_j / se_j`.
//! * p-value: the conservative one-sided union-intersection (Bonferroni) bound
//!   `p = min(1, m Phi(T))`. The least favourable null is every `d_j = 0`; Bonferroni is valid
//!   under any dependence among the `z_j` and is exact-in-form for `m = 1`, but conservative for
//!   `m > 1`. Rejection is evidence that the effects are *not* monotone in that direction;
//!   failing to reject does not prove monotonicity.
//!
//! The Type I error and power of the test are *unmeasured*
//! ([`CalibrationStatus::Unmeasured`]); every p-value is asymptotic and no coverage claim is
//! made.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::reason_code;
use antecedent_kernels::erfc;

use crate::effect_constancy::{CalibrationStatus, holm_adjust};
use crate::error::EstimationError;
use crate::vector_treatment::{
    CoefficientEstimate, Contrast, ContrastEstimate, DesignSummary, JointWald, NamedColumn,
    TreatmentColumn, VectorCovariance, VectorTreatmentFit, VectorTreatmentInput,
    VectorTreatmentOptions, count, fit_joint, fit_joint_from_summary, refuse, summarize_joint,
};

/// Null of the non-decreasing monotonicity test.
pub const MONOTONICITY_NON_DECREASING_NULL: &str =
    "every adjacent step of the adjusted level effects is non-negative (non-decreasing)";

/// Null of the non-increasing monotonicity test.
pub const MONOTONICITY_NON_INCREASING_NULL: &str =
    "every adjacent step of the adjusted level effects is non-positive (non-increasing)";

/// Null of the omnibus test.
pub const CATEGORICAL_OMNIBUS_NULL: &str = "every level has the reference level's effect";

/// Inference claim of the route.
pub const CATEGORICAL_INFERENCE_CLAIM: &str = "asymptotic_wald_calibration_unmeasured";

/// Whether the declared level order is a scale.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LevelScale {
    /// Nominal levels: order carries no meaning and is canonicalised before use.
    Unordered,
    /// Ordinal levels: the declared order is the scale.
    Ordered,
}

/// Direction of the declared monotonicity null.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MonotonicityDirection {
    /// Null: effects non-decreasing along the declared order.
    NonDecreasing,
    /// Null: effects non-increasing along the declared order.
    NonIncreasing,
}

/// Rows with their level label.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentInput {
    /// Outcome, one value per row.
    pub outcome: Vec<f64>,
    /// Identity of the row snapshot.
    pub row_snapshot: String,
    /// Shared adjustment columns (may be empty).
    pub adjustment: Vec<NamedColumn>,
    /// Level label of each row.
    pub levels: Vec<String>,
}

/// Declared design of the categorical regime.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentSpec {
    /// Declared level set (the scale, when ordered); at least two distinct levels.
    pub declared_levels: Vec<String>,
    /// Ordered or unordered.
    pub scale: LevelScale,
    /// Reference level (must be declared).
    pub reference: String,
    /// Fewest rows a declared level may have (at least one).
    pub min_level_rows: usize,
    /// Requested pairwise contrasts `(from, to)`, each estimating `effect(to) - effect(from)`.
    pub pairwise: Vec<(String, String)>,
    /// Declared monotonicity test; requires [`LevelScale::Ordered`].
    pub monotonicity: Option<MonotonicityDirection>,
    /// Covariance estimator.
    pub covariance: VectorCovariance,
}

/// Rows observed at a level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LevelCount {
    /// Level name.
    pub level: String,
    /// Rows at the level.
    pub rows: usize,
}

/// Effect of one level against the reference.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelContrast {
    /// The level.
    pub level: String,
    /// The reference level.
    pub reference: String,
    /// `effect(level) - effect(reference)`.
    pub estimate: f64,
    /// Standard error from the dummy-regression covariance.
    pub standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value across the declared family.
    pub p_holm: f64,
}

/// A requested pairwise contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct PairContrast {
    /// Baseline level.
    pub from: String,
    /// Compared level.
    pub to: String,
    /// `effect(to) - effect(from)`.
    pub estimate: f64,
    /// Standard error from the dummy-regression covariance (off-diagonals included).
    pub standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value across the declared family.
    pub p_holm: f64,
}

/// One adjacent step of the ordered effects.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjacentStep {
    /// Lower level.
    pub from: String,
    /// Next level up.
    pub to: String,
    /// `effect(to) - effect(from)`.
    pub difference: f64,
    /// Its standard error.
    pub standard_error: f64,
    /// `difference / standard_error`.
    pub z: f64,
}

/// Result of the declared monotonicity test.
#[derive(Clone, Debug, PartialEq)]
pub struct MonotonicityResult {
    /// Declared direction.
    pub direction: MonotonicityDirection,
    /// The frozen null of that direction.
    pub null: &'static str,
    /// The adjacent steps.
    pub steps: Vec<AdjacentStep>,
    /// Oriented minimum step z-score `T`.
    pub statistic: f64,
    /// Conservative union-intersection p-value `min(1, m Phi(T))`.
    pub p_value: f64,
    /// Always `true`: the Bonferroni bound is conservative for more than one step.
    pub conservative: bool,
    /// Calibration of the test.
    pub calibration: CalibrationStatus,
}

/// The categorical fit.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentFit {
    /// Reference level.
    pub reference: String,
    /// Scale.
    pub scale: LevelScale,
    /// Canonical level order used (declared order when ordered, sorted when unordered).
    pub level_order: Vec<String>,
    /// Rows per level, in canonical order.
    pub counts: Vec<LevelCount>,
    /// Dummy coefficients (`level:<name>`), non-reference levels in canonical order.
    pub coefficients: Vec<CoefficientEstimate>,
    /// Row-major covariance of the dummy coefficients.
    pub covariance: Vec<f64>,
    /// Per-level contrasts against the reference, in canonical order.
    pub level_contrasts: Vec<LevelContrast>,
    /// Requested pairwise contrasts, in request order.
    pub pairwise: Vec<PairContrast>,
    /// Size of the Holm family (levels vs reference plus requested pairs).
    pub family_size: usize,
    /// Wald test of [`CATEGORICAL_OMNIBUS_NULL`].
    pub omnibus: JointWald,
    /// Monotonicity test, when declared.
    pub monotonicity: Option<MonotonicityResult>,
    /// Covariance estimator.
    pub covariance_kind: VectorCovariance,
    /// Calibration of the inference.
    pub calibration: CalibrationStatus,
}

fn dummy_name(level: &str) -> String {
    format!("level:{level}")
}

fn pair_contrast(name: String, from: &str, to: &str, reference: &str) -> Contrast {
    let mut weights = Vec::with_capacity(2);
    if to != reference {
        weights.push((dummy_name(to), 1.0));
    }
    if from != reference {
        weights.push((dummy_name(from), -1.0));
    }
    Contrast { name, weights }
}

/// Validate the declared design and return the canonical level order.
pub(crate) fn validate_spec(
    spec: &CategoricalTreatmentSpec,
) -> Result<Vec<String>, EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let mut order = spec.declared_levels.clone();
    let distinct: BTreeSet<&str> = order.iter().map(String::as_str).collect();
    if order.iter().any(String::is_empty) || distinct.len() != order.len() {
        return Err(refuse(
            invalid,
            "categorical_treatment.invalid_level",
            "declared levels must be non-empty and distinct",
        ));
    }
    if order.len() < 2 {
        return Err(refuse(
            invalid,
            "categorical_treatment.too_few_levels",
            "a categorical treatment needs at least two declared levels",
        ));
    }
    if !distinct.contains(spec.reference.as_str()) {
        return Err(refuse(
            invalid,
            "categorical_treatment.unknown_reference",
            &format!("the reference level `{}` is not a declared level", spec.reference),
        ));
    }
    if spec.min_level_rows == 0 {
        return Err(refuse(
            invalid,
            "categorical_treatment.invalid_min_level_rows",
            "min_level_rows must be at least one",
        ));
    }
    let mut pairs: BTreeSet<(&str, &str)> = BTreeSet::new();
    for (from, to) in &spec.pairwise {
        if !distinct.contains(from.as_str()) || !distinct.contains(to.as_str()) {
            return Err(refuse(
                invalid,
                "categorical_treatment.unknown_level",
                &format!("the pairwise contrast `{from}` -> `{to}` names an undeclared level"),
            ));
        }
        if from == to || !pairs.insert((from.as_str(), to.as_str())) {
            return Err(refuse(
                invalid,
                "categorical_treatment.invalid_pair",
                &format!("the pairwise contrast `{from}` -> `{to}` is trivial or repeated"),
            ));
        }
    }
    if spec.monotonicity.is_some() && spec.scale == LevelScale::Unordered {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "categorical_treatment.monotonicity_requires_ordered",
            "a monotonicity test needs ordered levels; unordered levels have no scale",
        ));
    }
    if spec.scale == LevelScale::Unordered {
        order.sort();
    }
    Ok(order)
}

/// Count rows per level; refuse undeclared, absent and sparse levels (the level is named).
pub(crate) fn count_levels(
    input: &CategoricalTreatmentInput,
    spec: &CategoricalTreatmentSpec,
    order: &[String],
) -> Result<Vec<LevelCount>, EstimationError> {
    if input.levels.len() != input.outcome.len() {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "categorical_treatment.row_count_mismatch",
            "the level labels do not have one entry per outcome row",
        ));
    }
    let mut counts: BTreeMap<&str, usize> = order.iter().map(|l| (l.as_str(), 0)).collect();
    for label in &input.levels {
        let Some(slot) = counts.get_mut(label.as_str()) else {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "categorical_treatment.undeclared_level",
                &format!("a row carries the undeclared level `{label}`"),
            ));
        };
        *slot += 1;
    }
    let mut out = Vec::with_capacity(order.len());
    for level in order {
        let rows = counts.get(level.as_str()).copied().unwrap_or(0);
        if rows == 0 {
            return Err(refuse(
                reason_code!("arm_not_populated"),
                "categorical_treatment.absent_level",
                &format!("declared level `{level}` has no rows"),
            ));
        }
        if rows < spec.min_level_rows {
            return Err(refuse(
                reason_code!("arm_not_populated"),
                "categorical_treatment.sparse_level",
                &format!(
                    "declared level `{level}` has {rows} rows, fewer than the declared minimum {}",
                    spec.min_level_rows
                ),
            ));
        }
        out.push(LevelCount { level: level.clone(), rows });
    }
    Ok(out)
}

pub(crate) fn dummy_columns(
    input: &CategoricalTreatmentInput,
    order: &[String],
    reference: &str,
) -> Vec<TreatmentColumn> {
    let adjustment_set: Vec<String> = input.adjustment.iter().map(|c| c.name.clone()).collect();
    order
        .iter()
        .filter(|level| level.as_str() != reference)
        .map(|level| TreatmentColumn {
            name: dummy_name(level),
            values: input
                .levels
                .iter()
                .map(|label| if label == level { 1.0 } else { 0.0 })
                .collect(),
            adjustment_set: adjustment_set.clone(),
            row_snapshot: input.row_snapshot.clone(),
        })
        .collect()
}

/// The declared contrasts: levels vs reference, requested pairs, then (ordered) adjacent steps.
fn declared_contrasts(
    spec: &CategoricalTreatmentSpec,
    order: &[String],
) -> (Vec<Contrast>, usize, usize) {
    let reference = spec.reference.as_str();
    let mut contrasts = Vec::new();
    for level in order.iter().filter(|l| l.as_str() != reference) {
        contrasts.push(pair_contrast(format!("level|{level}"), reference, level, reference));
    }
    let n_levels = contrasts.len();
    for (from, to) in &spec.pairwise {
        contrasts.push(pair_contrast(format!("pair|{from}|{to}"), from, to, reference));
    }
    let n_pairs = spec.pairwise.len();
    if spec.monotonicity.is_some() {
        for window in order.windows(2) {
            let (from, to) = (&window[0], &window[1]);
            contrasts.push(pair_contrast(format!("step|{from}|{to}"), from, to, reference));
        }
    }
    (contrasts, n_levels, n_pairs)
}

fn normal_cdf(t: f64) -> f64 {
    0.5 * erfc(-t / std::f64::consts::SQRT_2)
}

fn monotonicity_result(
    direction: MonotonicityDirection,
    order: &[String],
    estimates: &[ContrastEstimate],
) -> MonotonicityResult {
    let sign = match direction {
        MonotonicityDirection::NonDecreasing => 1.0,
        MonotonicityDirection::NonIncreasing => -1.0,
    };
    let null = match direction {
        MonotonicityDirection::NonDecreasing => MONOTONICITY_NON_DECREASING_NULL,
        MonotonicityDirection::NonIncreasing => MONOTONICITY_NON_INCREASING_NULL,
    };
    let steps: Vec<AdjacentStep> = order
        .windows(2)
        .zip(estimates)
        .map(|(window, estimate)| AdjacentStep {
            from: window[0].clone(),
            to: window[1].clone(),
            difference: estimate.estimate,
            standard_error: estimate.standard_error,
            z: estimate.z,
        })
        .collect();
    let statistic = steps.iter().map(|s| sign * s.z).fold(f64::INFINITY, f64::min);
    let p_value = (count(steps.len()) * normal_cdf(statistic)).min(1.0);
    MonotonicityResult {
        direction,
        null,
        steps,
        statistic,
        p_value,
        conservative: true,
        calibration: CalibrationStatus::Unmeasured,
    }
}

/// Fit the categorical regime and report its contrasts, multiplicity and monotonicity test.
///
/// # Errors
///
/// `invalid_argument` with `categorical_treatment.invalid_level`,
/// `categorical_treatment.too_few_levels`, `categorical_treatment.unknown_reference`,
/// `categorical_treatment.invalid_min_level_rows`, `categorical_treatment.unknown_level`,
/// `categorical_treatment.invalid_pair` or `categorical_treatment.undeclared_level`;
/// `route_not_supported` with `categorical_treatment.monotonicity_requires_ordered` or
/// `categorical_treatment.row_count_mismatch`; `arm_not_populated` with
/// `categorical_treatment.absent_level` or `categorical_treatment.sparse_level` (the level is
/// named in the message); and any refusal of the underlying joint fit, which carries a
/// `vector_treatment.*` detail (for example a dummy column collinear with the adjustment
/// block).
pub fn fit_categorical_treatment(
    input: &CategoricalTreatmentInput,
    spec: &CategoricalTreatmentSpec,
) -> Result<CategoricalTreatmentFit, EstimationError> {
    let order = validate_spec(spec)?;
    let counts = count_levels(input, spec, &order)?;
    let (contrasts, n_levels, n_pairs) = declared_contrasts(spec, &order);
    let vector_input = VectorTreatmentInput {
        outcome: input.outcome.clone(),
        row_snapshot: input.row_snapshot.clone(),
        adjustment: input.adjustment.clone(),
        treatments: dummy_columns(input, &order, &spec.reference),
    };
    let options = VectorTreatmentOptions { covariance: spec.covariance, contrasts };
    let fit = fit_joint(&vector_input, &options, 1)?;
    Ok(assemble_categorical(fit, spec, order, counts, (n_levels, n_pairs)))
}

/// Level counts and the sufficient statistics of the dummy-coded design: the model-based replay
/// summary of a categorical fit (no rows).
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalSummary {
    /// Rows per level, in canonical order.
    pub counts: Vec<LevelCount>,
    /// Gram matrix, `X'y` and residual sum of squares of the design
    /// `[intercept, adjustment..., level:<non-reference levels>...]`.
    pub design: DesignSummary,
}

/// Validate, count and fit the declared design once and keep its sufficient statistics.
///
/// # Errors
///
/// The refusals of [`fit_categorical_treatment`] for the declared design, the level counts and
/// the dummy-coded regression.
pub fn summarize_categorical_design(
    input: &CategoricalTreatmentInput,
    spec: &CategoricalTreatmentSpec,
) -> Result<CategoricalSummary, EstimationError> {
    let order = validate_spec(spec)?;
    let counts = count_levels(input, spec, &order)?;
    let vector_input = VectorTreatmentInput {
        outcome: input.outcome.clone(),
        row_snapshot: input.row_snapshot.clone(),
        adjustment: input.adjustment.clone(),
        treatments: dummy_columns(input, &order, &spec.reference),
    };
    let design = summarize_joint(&vector_input, 1)?;
    Ok(CategoricalSummary { counts, design })
}

/// Categorical fit from sufficient statistics, bit-identical to the model-based
/// [`fit_categorical_treatment`] of the rows the summary came from.
///
/// # Errors
///
/// `route_not_supported` with `vector_treatment.summary_model_based_only` for a robust
/// covariance; `invalid_argument` with `categorical_treatment.summary_inconsistent` when the
/// counts disagree with the design or the declared levels; the declared-design refusals of
/// [`fit_categorical_treatment`] (`categorical_treatment.absent_level`,
/// `categorical_treatment.sparse_level`, ...).
pub fn fit_categorical_from_summary(
    spec: &CategoricalTreatmentSpec,
    summary: &CategoricalSummary,
) -> Result<CategoricalTreatmentFit, EstimationError> {
    let order = validate_spec(spec)?;
    let inconsistent = |message: &str| {
        refuse(
            reason_code!("invalid_argument"),
            "categorical_treatment.summary_inconsistent",
            message,
        )
    };
    let matches_order = summary.counts.len() == order.len()
        && summary.counts.iter().zip(&order).all(|(c, level)| &c.level == level);
    if !matches_order {
        return Err(inconsistent("the stored level counts do not follow the declared levels"));
    }
    for entry in &summary.counts {
        if entry.rows == 0 {
            return Err(refuse(
                reason_code!("arm_not_populated"),
                "categorical_treatment.absent_level",
                &format!("declared level `{}` has no rows", entry.level),
            ));
        }
        if entry.rows < spec.min_level_rows {
            return Err(refuse(
                reason_code!("arm_not_populated"),
                "categorical_treatment.sparse_level",
                &format!(
                    "declared level `{}` has {} rows, fewer than the declared minimum {}",
                    entry.level, entry.rows, spec.min_level_rows
                ),
            ));
        }
    }
    let non_reference: Vec<&String> = order.iter().filter(|l| **l != spec.reference).collect();
    let design = &summary.design;
    let t0 = design.first_treatment;
    let p = t0 + non_reference.len();
    let total: usize = summary.counts.iter().map(|c| c.rows).sum();
    let stored_n = |i: usize| design.gram.get(i * p + i).copied();
    let counts_agree = t0 >= 1
        && design.gram.len() == p * p
        && design.n_rows == total
        && stored_n(0) == Some(count(total))
        && non_reference.iter().enumerate().all(|(j, level)| {
            let rows = summary.counts.iter().find(|c| &&c.level == level).map(|c| c.rows);
            rows.map(count) == stored_n(t0 + j)
        });
    if !counts_agree {
        return Err(inconsistent("the level counts disagree with the stored Gram matrix"));
    }
    let (contrasts, n_levels, n_pairs) = declared_contrasts(spec, &order);
    let names: Vec<String> = non_reference.iter().map(|l| dummy_name(l)).collect();
    let options = VectorTreatmentOptions { covariance: spec.covariance, contrasts };
    let fit = fit_joint_from_summary(design, &names, &options, 1)?;
    Ok(assemble_categorical(fit, spec, order, summary.counts.clone(), (n_levels, n_pairs)))
}

fn assemble_categorical(
    fit: VectorTreatmentFit,
    spec: &CategoricalTreatmentSpec,
    order: Vec<String>,
    counts: Vec<LevelCount>,
    (n_levels, n_pairs): (usize, usize),
) -> CategoricalTreatmentFit {
    let family = &fit.contrasts[..n_levels + n_pairs];
    let raw: Vec<f64> = family.iter().map(|c| c.p_value).collect();
    let holm = holm_adjust(&raw);
    let non_reference: Vec<&String> = order.iter().filter(|l| **l != spec.reference).collect();
    let level_contrasts = family[..n_levels]
        .iter()
        .zip(&non_reference)
        .zip(&holm)
        .map(|((c, level), p_holm)| LevelContrast {
            level: (*level).clone(),
            reference: spec.reference.clone(),
            estimate: c.estimate,
            standard_error: c.standard_error,
            z: c.z,
            p_value: c.p_value,
            p_holm: *p_holm,
        })
        .collect();
    let pairwise = family[n_levels..]
        .iter()
        .zip(&spec.pairwise)
        .zip(&holm[n_levels..])
        .map(|((c, (from, to)), p_holm)| PairContrast {
            from: from.clone(),
            to: to.clone(),
            estimate: c.estimate,
            standard_error: c.standard_error,
            z: c.z,
            p_value: c.p_value,
            p_holm: *p_holm,
        })
        .collect();
    let monotonicity = spec.monotonicity.map(|direction| {
        monotonicity_result(direction, &order, &fit.contrasts[n_levels + n_pairs..])
    });
    CategoricalTreatmentFit {
        reference: spec.reference.clone(),
        scale: spec.scale,
        level_order: order,
        counts,
        coefficients: fit.coefficients,
        covariance: fit.covariance,
        level_contrasts,
        pairwise,
        family_size: n_levels + n_pairs,
        omnibus: fit.joint_wald,
        monotonicity,
        covariance_kind: spec.covariance,
        calibration: CalibrationStatus::Unmeasured,
    }
}

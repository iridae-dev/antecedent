//! One summary pass over a table: the profile a baseline stores and new data is compared by.
//!
//! The same function profiles the plan's data (the baseline) and new data. New data is binned
//! on the baseline's quantile edges, so the two bin-count vectors are directly comparable; its
//! subsample is drawn from its own keyed stream. A column is read as numbers when its storage
//! is `float64`, `int64` or `boolean` (a missing cell is `NaN`), as levels when it is
//! categorical, and only for missingness otherwise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::collections::BTreeMap;

use antecedent_core::{SemanticOperation, ValueType, VariableId};
use antecedent_data::{CategoricalColumn, ColumnView, TableView, TabularData};
use antecedent_io_core::staleness_artifact::{
    JointProfileWire, LevelCountWire, MissingPatternWire, NARROW_PATTERN_COLUMNS, NumericProfileWire, OutcomeModelWire,
    OverlapSummaryWire, RecordedCheckWire, StalenessBaselineWire, StalenessColumnWire,
};
use antecedent_kernels::{Welford, update_scatter};
use antecedent_stats::cholesky_spd;

use super::policy::StalenessPolicy;
use super::stats::{logistic_ridge, propensities, stream, variance};
use super::{StalenessError, StalenessPlan};

/// Role labels.
pub(crate) const ROLE_TREATMENT: &str = "treatment";
pub(crate) const ROLE_OUTCOME: &str = "outcome";
pub(crate) const ROLE_ADJUSTMENT: &str = "adjustment";
pub(crate) const ROLE_PERIOD: &str = "period";
pub(crate) const ROLE_OTHER: &str = "other";

/// Recorded-check names and levels.
pub(crate) const CHECK_BALANCE: &str = "covariate_balance";
pub(crate) const CHECK_CONDITIONING: &str = "adjustment_conditioning";
pub(crate) const CHECK_ARMS: &str = "treatment_arms";
pub(crate) const CHECK_OUTCOME_VARIATION: &str = "outcome_variation";
pub(crate) const CHECK_COMPLETE_CASES: &str = "complete_cases";
pub(crate) const LEVEL_IDENTIFICATION: &str = "identification";
pub(crate) const LEVEL_DESIGN: &str = "design";
pub(crate) const LEVEL_TARGET: &str = "target";
pub(crate) const AT_MOST: &str = "at_most";
pub(crate) const AT_LEAST: &str = "at_least";

/// Ridge of the diagnostic propensity fit.
const PROPENSITY_RIDGE: f64 = 1e-4;

/// One column, read for profiling.
pub(crate) enum ColumnRead<'a> {
    /// Numbers; a missing cell is `NaN`.
    Numeric(Cow<'a, [f64]>),
    /// Categorical codes and their domain.
    Categorical(&'a CategoricalColumn),
    /// Another storage kind: only its validity is read.
    Other { missing: u64 },
}

impl ColumnRead<'_> {
    /// Whether row `i` is observed.
    pub(crate) fn observed(&self, i: usize) -> bool {
        match self {
            Self::Numeric(values) => values.get(i).is_some_and(|v| !v.is_nan()),
            Self::Categorical(column) => column.validity.is_valid(i),
            Self::Other { .. } => false,
        }
    }

    /// Numeric value of row `i` (`NaN` when missing or not numeric).
    pub(crate) fn value(&self, i: usize) -> f64 {
        match self {
            Self::Numeric(values) => values.get(i).copied().unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }

    /// Level label of row `i`, for an observed categorical cell.
    pub(crate) fn label(&self, i: usize) -> Option<&str> {
        match self {
            Self::Categorical(column) if column.validity.is_valid(i) => column
                .codes
                .get(i)
                .and_then(|code| column.domain.levels.get(code.as_usize()))
                .map(|level| level.label.as_ref()),
            _ => None,
        }
    }

    /// Whether this column is read as numbers.
    pub(crate) const fn is_numeric(&self) -> bool {
        matches!(self, Self::Numeric(_))
    }
}

/// Stable label of a declared value type.
pub(crate) fn value_type_label(value_type: &ValueType) -> String {
    match value_type {
        ValueType::Continuous => "continuous".into(),
        ValueType::Count => "count".into(),
        ValueType::Binary => "binary".into(),
        ValueType::Categorical => "categorical".into(),
        ValueType::Ordinal => "ordinal".into(),
        ValueType::Unspecified => "unspecified".into(),
        ValueType::Vector { width, element } => format!("vector[{width};{element:?}]"),
    }
}

const fn storage_label(view: ColumnView<'_>) -> &'static str {
    match view {
        ColumnView::Float64(_) => "float64",
        ColumnView::Int64(_) => "int64",
        ColumnView::Boolean(_) => "boolean",
        ColumnView::Categorical(_) => "categorical",
        ColumnView::Timestamp(_) => "timestamp",
        ColumnView::FixedVector(_) => "fixed_vector",
    }
}

/// Read column `id` of `data`, with its storage label.
pub(crate) fn read_column(
    data: &TabularData,
    id: VariableId,
) -> Result<(ColumnRead<'_>, &'static str), StalenessError> {
    let view = data.column(id)?;
    let storage = storage_label(view);
    let read = match view {
        ColumnView::Float64(_) | ColumnView::Int64(_) | ColumnView::Boolean(_) => {
            ColumnRead::Numeric(data.float64_cow(id)?)
        }
        ColumnView::Categorical(column) => ColumnRead::Categorical(column),
        other => {
            let validity = other.validity();
            let missing = (0..other.len()).filter(|&i| !validity.is_valid(i)).count();
            ColumnRead::Other { missing: missing as u64 }
        }
    };
    Ok((read, storage))
}

/// The role of `name` under `plan` and `period`.
pub(crate) fn role_of(name: &str, plan: &StalenessPlan, period: Option<&str>) -> &'static str {
    if plan.treatment.as_deref() == Some(name) {
        ROLE_TREATMENT
    } else if plan.outcome.as_deref() == Some(name) {
        ROLE_OUTCOME
    } else if plan.adjustment.iter().any(|a| a == name) {
        ROLE_ADJUSTMENT
    } else if period == Some(name) {
        ROLE_PERIOD
    } else {
        ROLE_OTHER
    }
}

/// The plan's columns in order (treatment, outcome, adjustment set), without repeats.
pub(crate) fn plan_columns(plan: &StalenessPlan) -> Vec<String> {
    let mut columns: Vec<String> = Vec::new();
    for name in plan.treatment.iter().chain(plan.outcome.iter()).chain(plan.adjustment.iter()) {
        if !columns.contains(name) {
            columns.push(name.clone());
        }
    }
    columns
}

/// Quantile edges of the sorted finite values: the inverse empirical distribution at `k/bins`,
/// strictly increasing.
fn quantile_edges(sorted: &[f64], bins: usize) -> Vec<f64> {
    let n = sorted.len();
    let mut edges: Vec<f64> = Vec::with_capacity(bins.saturating_sub(1));
    if n == 0 {
        return edges;
    }
    for k in 1..bins {
        let index = (k * n).div_ceil(bins).saturating_sub(1).min(n - 1);
        let edge = sorted[index];
        if edges.last().is_none_or(|last| edge > *last) {
            edges.push(edge);
        }
    }
    edges
}

/// Bin of `x`: the number of edges at or below it.
pub(crate) fn bin_of(edges: &[f64], x: f64) -> usize {
    edges.partition_point(|edge| *edge <= x)
}

fn numeric_profile(
    values: &[f64],
    reference_edges: Option<&[f64]>,
    bins: usize,
) -> NumericProfileWire {
    let mut welford = Welford::new();
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &value in values.iter().filter(|v| v.is_finite()) {
        welford.push(value);
        min = min.min(value);
        max = max.max(value);
    }
    let edges = reference_edges.map_or_else(
        || {
            let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
            sorted.sort_by(f64::total_cmp);
            quantile_edges(&sorted, bins)
        },
        <[f64]>::to_vec,
    );
    let mut bin_counts = vec![0u64; edges.len() + 1];
    for &value in values.iter().filter(|v| v.is_finite()) {
        bin_counts[bin_of(&edges, value)] += 1;
    }
    if welford.count() == 0 {
        (min, max) = (f64::NAN, f64::NAN);
    }
    NumericProfileWire { mean: welford.mean(), m2: welford.m2(), min, max, edges, bin_counts }
}

fn level_counts(column: &CategoricalColumn) -> Vec<LevelCountWire> {
    let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
    for (i, code) in column.codes.iter().enumerate() {
        if !column.validity.is_valid(i) {
            continue;
        }
        if let Some(level) = column.domain.levels.get(code.as_usize()) {
            *counts.entry(level.label.as_ref()).or_insert(0) += 1;
        }
    }
    counts.into_iter().map(|(label, count)| LevelCountWire { label: label.into(), count }).collect()
}

/// A table's columns resolved by name.
pub(crate) struct ReadTable<'a> {
    /// `(name, read, storage, value type label)` in schema order.
    pub columns: Vec<(String, ColumnRead<'a>, &'static str, String)>,
    /// Rows.
    pub rows: usize,
}

impl<'a> ReadTable<'a> {
    /// Read every column of `data`.
    pub(crate) fn new(data: &'a TabularData) -> Result<Self, StalenessError> {
        let mut columns = Vec::with_capacity(data.schema().len());
        for variable in data.schema().variables() {
            let (read, storage) = read_column(data, variable.id)?;
            columns.push((
                variable.name.to_string(),
                read,
                storage,
                value_type_label(&variable.value_type),
            ));
        }
        Ok(Self { columns, rows: data.row_count() })
    }

    /// The read of column `name`.
    pub(crate) fn get(&self, name: &str) -> Option<&ColumnRead<'a>> {
        self.columns.iter().find(|(n, ..)| n == name).map(|(_, read, ..)| read)
    }

    fn numeric(&self, name: &str) -> Option<&ColumnRead<'a>> {
        self.get(name).filter(|read| read.is_numeric())
    }
}

/// The profile of one table.
pub(crate) struct ProfileInputs<'p> {
    /// The plan's roles.
    pub plan: &'p StalenessPlan,
    /// The policy (bins, subsample, margins, seed, period column).
    pub policy: &'p StalenessPolicy,
    /// The baseline whose edges, covariates and design new data is profiled on.
    pub reference: Option<&'p StalenessBaselineWire>,
    /// Operation tag of the subsample stream.
    pub subsample_op: SemanticOperation,
}

/// Profile `table`: an unsealed baseline wire (seal, version, kind empty).
pub(crate) fn profile(
    table: &ReadTable<'_>,
    snapshot: String,
    inputs: &ProfileInputs<'_>,
) -> StalenessBaselineWire {
    let ProfileInputs { plan, policy, reference, subsample_op } = *inputs;
    let period = policy.period_column.as_deref();
    let reference_column =
        |name: &str| reference.and_then(|r| r.columns.iter().find(|c| c.name == name));
    let mut columns = Vec::with_capacity(table.columns.len());
    for (name, read, storage, value_type) in &table.columns {
        let missing = match read {
            ColumnRead::Numeric(values) => values.iter().filter(|v| v.is_nan()).count() as u64,
            ColumnRead::Categorical(column) => {
                (0..column.len()).filter(|&i| !column.validity.is_valid(i)).count() as u64
            }
            ColumnRead::Other { missing } => *missing,
        };
        let (numeric, levels) = match read {
            ColumnRead::Numeric(values) => {
                let edges = reference_column(name)
                    .and_then(|c| c.numeric.as_ref())
                    .map(|n| n.edges.as_slice());
                (Some(numeric_profile(values, edges, policy.bins)), None)
            }
            ColumnRead::Categorical(column) => (None, Some(level_counts(column))),
            ColumnRead::Other { .. } => (None, None),
        };
        columns.push(StalenessColumnWire {
            name: name.clone(),
            value_type: value_type.clone(),
            storage: (*storage).into(),
            role: role_of(name, plan, period).into(),
            n_valid: table.rows as u64 - missing,
            n_missing: missing,
            numeric,
            levels,
        });
    }
    let plan_cols = plan_columns(plan);
    let complete = complete_rows(table, &plan_cols);
    let covariates = reference
        .and_then(|r| r.joint.as_ref())
        .map_or_else(|| covariate_names(table, plan, period), |joint| joint.columns.clone());
    let joint = joint_profile(table, &covariates, policy, subsample_op);
    let overlap = overlap_summary(table, plan, policy);
    let arms = treatment_arms(table, plan, &complete);
    let checks =
        recorded_checks(table, plan, policy, &complete, (joint.as_ref(), overlap.as_ref(), &arms));
    let outcome_model = if plan.bayesian { outcome_model(table, plan) } else { None };
    StalenessBaselineWire {
        version: 0,
        required_features: Vec::new(),
        kind: String::new(),
        plan_identity: plan.plan_identity.clone(),
        snapshot,
        policy: policy.identity(),
        treatment: plan.treatment.clone(),
        outcome: plan.outcome.clone(),
        adjustment: plan.adjustment.clone(),
        bayesian: plan.bayesian,
        period_column: policy.period_column.clone(),
        rows: table.rows as u64,
        complete_rows: complete.iter().filter(|c| **c).count() as u64,
        arms,
        columns,
        joint,
        missing: missing_patterns(table, &plan_cols),
        overlap,
        checks,
        outcome_model,
        seal: String::new(),
    }
}

/// Rows observed on every one of `names` that the table has (a name it lacks is skipped).
pub(crate) fn complete_rows(table: &ReadTable<'_>, names: &[String]) -> Vec<bool> {
    let reads: Vec<&ColumnRead<'_>> = names.iter().filter_map(|n| table.get(n)).collect();
    (0..table.rows).map(|i| reads.iter().all(|r| r.observed(i))).collect()
}

/// Numeric covariates the joint diagnostics use: the numeric adjustment set, or, without one,
/// every numeric column that is not the treatment, the outcome or the period.
fn covariate_names(
    table: &ReadTable<'_>,
    plan: &StalenessPlan,
    period: Option<&str>,
) -> Vec<String> {
    let numeric_adjustment: Vec<String> =
        plan.adjustment.iter().filter(|a| table.numeric(a).is_some()).cloned().collect();
    if !numeric_adjustment.is_empty() {
        return numeric_adjustment;
    }
    table
        .columns
        .iter()
        .filter(|(name, read, ..)| read.is_numeric() && role_of(name, plan, period) == ROLE_OTHER)
        .map(|(name, ..)| name.clone())
        .collect()
}

fn joint_profile(
    table: &ReadTable<'_>,
    covariates: &[String],
    policy: &StalenessPolicy,
    subsample_op: SemanticOperation,
) -> Option<JointProfileWire> {
    let reads: Vec<&ColumnRead<'_>> =
        covariates.iter().map(|name| table.numeric(name)).collect::<Option<_>>()?;
    let dim = reads.len();
    if dim == 0 {
        return None;
    }
    let complete: Vec<usize> =
        (0..table.rows).filter(|&i| reads.iter().all(|r| r.value(i).is_finite())).collect();
    let total = complete.len();
    let mut mean = vec![0.0; dim];
    let mut m2 = vec![0.0; dim * dim];
    let mut row = vec![0.0; dim];
    let mut subsample = Vec::with_capacity(policy.subsample.min(total) * dim);
    let mut rng = stream(subsample_op, 0, policy.seed);
    let mut selected = 0usize;
    for (seen, &i) in complete.iter().enumerate() {
        for (slot, read) in row.iter_mut().zip(&reads) {
            *slot = read.value(i);
        }
        if seen == 0 {
            mean.copy_from_slice(&row);
        } else {
            update_scatter(&mut mean, &mut m2, seen as u64, &row);
        }
        // Selection sampling: a uniform subsample without replacement in one pass.
        let wanted = policy.subsample.min(total) - selected;
        let remaining = total - seen;
        if wanted > 0 && (rng.next_f64() * remaining as f64) < wanted as f64 {
            subsample.extend_from_slice(&row);
            selected += 1;
        }
    }
    Some(JointProfileWire {
        columns: covariates.to_vec(),
        n: total as u64,
        mean,
        m2,
        subsample,
        subsample_rows: selected as u64,
    })
}

fn missing_patterns(table: &ReadTable<'_>, names: &[String]) -> Option<MissingPatternWire> {
    let reads: Vec<(&String, &ColumnRead<'_>)> =
        names.iter().filter_map(|n| table.get(n).map(|r| (n, r))).collect();
    if reads.is_empty() {
        return None;
    }
    let columns = reads.iter().map(|(n, _)| (*n).clone()).collect();
    if reads.len() <= NARROW_PATTERN_COLUMNS {
        let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
        for i in 0..table.rows {
            let mut pattern = 0u64;
            for (bit, (_, read)) in reads.iter().enumerate() {
                if !read.observed(i) {
                    pattern |= 1 << bit;
                }
            }
            *counts.entry(pattern).or_insert(0) += 1;
        }
        return Some(MissingPatternWire {
            columns,
            patterns: counts.into_iter().collect(),
            wide_patterns: Vec::new(),
        });
    }
    // More columns than one word holds: the pattern is a vector of words, trailing zero words
    // omitted so equal patterns are equal keys.
    let mut counts: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
    for i in 0..table.rows {
        let mut words = vec![0u64; reads.len().div_ceil(64)];
        for (bit, (_, read)) in reads.iter().enumerate() {
            if !read.observed(i) {
                words[bit / 64] |= 1 << (bit % 64);
            }
        }
        while words.last() == Some(&0) {
            words.pop();
        }
        *counts.entry(words).or_insert(0) += 1;
    }
    Some(MissingPatternWire {
        columns,
        patterns: Vec::new(),
        wide_patterns: counts.into_iter().collect(),
    })
}

/// Binary 0/1 treatment over the rows where it is observed, both arms present.
#[allow(clippy::float_cmp, reason = "a binary treatment is coded exactly 0 or 1")]
pub(crate) fn binary_treatment<'a>(
    table: &'a ReadTable<'a>,
    plan: &StalenessPlan,
) -> Option<&'a ColumnRead<'a>> {
    let read = table.numeric(plan.treatment.as_deref()?)?;
    let mut arms = [false, false];
    for i in 0..table.rows {
        let v = read.value(i);
        if v.is_nan() {
            continue;
        }
        if v == 0.0 {
            arms[0] = true;
        } else if v == 1.0 {
            arms[1] = true;
        } else {
            return None;
        }
    }
    (arms[0] && arms[1]).then_some(read)
}

fn overlap_summary(
    table: &ReadTable<'_>,
    plan: &StalenessPlan,
    policy: &StalenessPolicy,
) -> Option<OverlapSummaryWire> {
    let treatment = binary_treatment(table, plan)?;
    let reads: Vec<&ColumnRead<'_>> =
        plan.adjustment.iter().map(|name| table.numeric(name)).collect::<Option<_>>()?;
    if reads.is_empty() {
        return None;
    }
    let rows: Vec<usize> = (0..table.rows)
        .filter(|&i| treatment.value(i).is_finite() && reads.iter().all(|r| r.value(i).is_finite()))
        .collect();
    let n = rows.len();
    if n < 2 {
        return None;
    }
    let t: Vec<f64> = rows.iter().map(|&i| treatment.value(i)).collect();
    let mut raw = Vec::with_capacity(reads.len());
    let mut standardized = Vec::with_capacity(reads.len());
    for read in &reads {
        let values: Vec<f64> = rows.iter().map(|&i| read.value(i)).collect();
        let mut welford = Welford::new();
        values.iter().for_each(|v| welford.push(*v));
        let sd = variance(welford.count(), welford.m2()).sqrt();
        let scale = if sd > 0.0 { sd } else { 1.0 };
        standardized.push(values.iter().map(|v| (v - welford.mean()) / scale).collect::<Vec<_>>());
        raw.push(values);
    }
    let beta = logistic_ridge(&standardized, &t, PROPENSITY_RIDGE)?;
    let e = propensities(&standardized, &beta, n);
    let eps = policy.positivity_epsilon;
    let extreme = e.iter().filter(|p| **p < eps || **p > 1.0 - eps).count();
    let weights: Vec<f64> = e
        .iter()
        .zip(&t)
        .map(|(p, ti)| {
            let p = p.clamp(1e-12, 1.0 - 1e-12);
            if *ti > 0.5 { 1.0 / p } else { 1.0 / (1.0 - p) }
        })
        .collect();
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    let ess = if sum_sq > 0.0 { sum * sum / sum_sq } else { 0.0 };
    let max_weighted_smd =
        raw.iter().map(|values| weighted_smd(values, &t, &weights).abs()).fold(0.0_f64, f64::max);
    Some(OverlapSummaryWire {
        n: n as u64,
        extreme_fraction: extreme as f64 / n as f64,
        ess_ratio: ess / n as f64,
        min_propensity: e.iter().copied().fold(f64::INFINITY, f64::min),
        max_propensity: e.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        max_weight: weights.iter().copied().fold(0.0_f64, f64::max),
        max_weighted_smd,
        epsilon: eps,
    })
}

/// Inverse-propensity-weighted standardized mean difference of one covariate between arms,
/// scaled by the unweighted arm variances.
fn weighted_smd(values: &[f64], t: &[f64], weights: &[f64]) -> f64 {
    let mut weighted = [Welford::new(), Welford::new()];
    let mut plain = [Welford::new(), Welford::new()];
    for ((v, ti), w) in values.iter().zip(t).zip(weights) {
        let arm = usize::from(*ti > 0.5);
        weighted[arm].push_weighted(*w, *v);
        plain[arm].push(*v);
    }
    let var = |w: &Welford| variance(w.count(), w.m2());
    super::stats::smd(weighted[0].mean(), var(&plain[0]), weighted[1].mean(), var(&plain[1]))
}

fn check(
    name: &str,
    level: &str,
    statistic: f64,
    threshold: f64,
    direction: &str,
) -> RecordedCheckWire {
    RecordedCheckWire {
        name: name.into(),
        level: level.into(),
        statistic,
        threshold,
        direction: direction.into(),
        passed: passes(statistic, threshold, direction),
    }
}

/// Whether `statistic` passes `threshold` in `direction` (a `NaN` statistic fails).
pub(crate) fn passes(statistic: f64, threshold: f64, direction: &str) -> bool {
    if statistic.is_nan() {
        return false;
    }
    if direction == AT_MOST { statistic <= threshold } else { statistic >= threshold }
}

/// The catalogue of checks a baseline records and an assessment re-runs.
fn recorded_checks(
    table: &ReadTable<'_>,
    plan: &StalenessPlan,
    policy: &StalenessPolicy,
    complete: &[bool],
    summaries: (Option<&JointProfileWire>, Option<&OverlapSummaryWire>, &[LevelCountWire]),
) -> Vec<RecordedCheckWire> {
    let (joint, overlap, arms) = summaries;
    let mut checks = Vec::new();
    if let Some(overlap) = overlap {
        checks.push(check(
            CHECK_BALANCE,
            LEVEL_IDENTIFICATION,
            overlap.max_weighted_smd,
            policy.balance_threshold,
            AT_MOST,
        ));
    }
    let adjustment_joint =
        joint.filter(|j| !plan.adjustment.is_empty() && j.columns == plan.adjustment);
    if let Some(joint) = adjustment_joint {
        checks.push(check(
            CHECK_CONDITIONING,
            LEVEL_IDENTIFICATION,
            correlation_condition(joint),
            policy.max_condition,
            AT_MOST,
        ));
    }
    if !arms.is_empty() {
        let smallest = arms.iter().map(|arm| arm.count).min().unwrap_or(0);
        checks.push(check(
            CHECK_ARMS,
            LEVEL_TARGET,
            smallest as f64,
            policy.min_rows_per_arm as f64,
            AT_LEAST,
        ));
    }
    if let Some(outcome) = plan.outcome.as_deref().and_then(|o| table.numeric(o)) {
        let mut welford = Welford::new();
        for i in (0..table.rows).filter(|&i| complete[i]) {
            welford.push(outcome.value(i));
        }
        let sd = variance(welford.count(), welford.m2()).sqrt();
        checks.push(check(CHECK_OUTCOME_VARIATION, LEVEL_TARGET, sd, f64::MIN_POSITIVE, AT_LEAST));
    }
    if table.rows > 0 && !plan_columns(plan).is_empty() {
        let fraction = complete.iter().filter(|c| **c).count() as f64 / table.rows as f64;
        checks.push(check(
            CHECK_COMPLETE_CASES,
            LEVEL_DESIGN,
            fraction,
            policy.min_complete_fraction,
            AT_LEAST,
        ));
    }
    checks
}

/// Lower bound of the condition number of the covariates' correlation matrix
/// (`+∞` for a constant covariate or a singular matrix).
fn correlation_condition(joint: &JointProfileWire) -> f64 {
    let dim = joint.columns.len();
    let diag: Vec<f64> = (0..dim).map(|i| joint.m2[i * dim + i]).collect();
    if diag.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
        return f64::INFINITY;
    }
    let correlation: Vec<f64> =
        (0..dim * dim).map(|k| joint.m2[k] / (diag[k / dim] * diag[k % dim]).sqrt()).collect();
    cholesky_spd(&correlation, dim).map_or(f64::INFINITY, |chol| {
        antecedent_kernels::cholesky_condition_lower_bound(&chol, dim)
    })
}

/// Distinct treatment values above which the treatment is read as continuous rather than as arms.
///
/// This is the classifier between a discrete and a continuous treatment, not a limit on how many
/// arms a discrete treatment may have: the counts are a map, so it only decides when to stop
/// collecting values that are evidently a continuum.
const CONTINUOUS_TREATMENT_DISTINCT_VALUES: usize = 16;

/// Treatment arms (distinct values or levels) and their counts over the complete rows; empty
/// for a continuous treatment (more than [`CONTINUOUS_TREATMENT_DISTINCT_VALUES`] distinct
/// numeric values) or no treatment. A categorical treatment is always read as arms, however many
/// levels it has.
pub(crate) fn treatment_arms(
    table: &ReadTable<'_>,
    plan: &StalenessPlan,
    complete: &[bool],
) -> Vec<LevelCountWire> {
    let Some(treatment) = plan.treatment.as_deref().and_then(|t| table.get(t)) else {
        return Vec::new();
    };
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let numeric = matches!(treatment, ColumnRead::Numeric(_));
    for i in (0..table.rows).filter(|&i| complete[i]) {
        let key = match treatment {
            ColumnRead::Numeric(_) => format!("{}", treatment.value(i)),
            ColumnRead::Categorical(_) => match treatment.label(i) {
                Some(label) => label.to_owned(),
                None => continue,
            },
            ColumnRead::Other { .. } => return Vec::new(),
        };
        *counts.entry(key).or_insert(0) += 1;
        if numeric && counts.len() > CONTINUOUS_TREATMENT_DISTINCT_VALUES {
            return Vec::new();
        }
    }
    counts.into_iter().map(|(label, count)| LevelCountWire { label, count }).collect()
}

fn outcome_model(table: &ReadTable<'_>, plan: &StalenessPlan) -> Option<OutcomeModelWire> {
    let outcome = table.numeric(plan.outcome.as_deref()?)?;
    let mut design: Vec<String> = plan.treatment.iter().cloned().collect();
    design.extend(plan.adjustment.iter().cloned());
    let reads: Vec<&ColumnRead<'_>> =
        design.iter().map(|name| table.numeric(name)).collect::<Option<_>>()?;
    let p = reads.len() + 1;
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    let mut yty = 0.0;
    let mut n = 0u64;
    let mut x = vec![0.0; p];
    for i in 0..table.rows {
        let y = outcome.value(i);
        if !y.is_finite() || reads.iter().any(|r| !r.value(i).is_finite()) {
            continue;
        }
        x[0] = 1.0;
        for (slot, read) in x[1..].iter_mut().zip(&reads) {
            *slot = read.value(i);
        }
        for (a, &xa) in x.iter().enumerate() {
            xty[a] += xa * y;
            for (b, &xb) in x.iter().enumerate() {
                xtx[a * p + b] += xa * xb;
            }
        }
        yty += y * y;
        n += 1;
    }
    Some(OutcomeModelWire { design, outcome: plan.outcome.clone()?, n, xtx, xty, yty })
}

/// Per-period moments of one monitored column, in period order.
pub(crate) struct PeriodSeries {
    /// Monitored column.
    pub column: String,
    /// `(period label, count, mean, centered sum of squares)`.
    pub periods: Vec<(String, u64, f64, f64)>,
}

/// Per-period moments of every numeric column other than the period itself, over the periods
/// the period column defines. `None` when the column is absent or neither numeric nor
/// categorical.
pub(crate) fn period_series(table: &ReadTable<'_>, period: &str) -> Option<Vec<PeriodSeries>> {
    let key_read = table.get(period)?;
    // Period key: an order key and a label.
    let key = |i: usize| -> Option<(u64, String)> {
        match key_read {
            ColumnRead::Numeric(_) => {
                let v = key_read.value(i);
                v.is_finite().then(|| (order_bits(v), format!("{v}")))
            }
            ColumnRead::Categorical(column) => {
                let code = column.codes.get(i)?;
                key_read.label(i).map(|label| (u64::from(code.raw()), label.to_owned()))
            }
            ColumnRead::Other { .. } => None,
        }
    };
    if matches!(key_read, ColumnRead::Other { .. }) {
        return None;
    }
    let mut series = Vec::new();
    for (name, read, ..) in &table.columns {
        if name == period || !read.is_numeric() {
            continue;
        }
        let mut groups: BTreeMap<u64, (String, Welford)> = BTreeMap::new();
        for i in 0..table.rows {
            let value = read.value(i);
            if !value.is_finite() {
                continue;
            }
            if let Some((order, label)) = key(i) {
                groups.entry(order).or_insert_with(|| (label, Welford::new())).1.push(value);
            }
        }
        series.push(PeriodSeries {
            column: name.clone(),
            periods: groups
                .into_values()
                .map(|(label, w)| (label, w.count(), w.mean(), w.m2()))
                .collect(),
        });
    }
    Some(series)
}

/// Bits of `v` whose unsigned order is the numeric order of finite values.
fn order_bits(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 1 { !bits } else { bits | (1 << 63) }
}

/// Bytes one assessment of a table of `rows × columns` holds at its peak: copies of the
/// numeric columns, the per-row masks, the sorted copy a baseline bins from, the propensity
/// design, and the energy distance matrix.
pub(crate) fn peak_bytes(rows: usize, columns: usize, policy: &StalenessPolicy) -> u64 {
    let rows = rows as u64;
    let columns = columns as u64;
    let per_row = columns.saturating_add(8).saturating_mul(8);
    let pooled = (2 * policy.subsample) as u64;
    rows.saturating_mul(per_row).saturating_add(pooled.saturating_mul(pooled).saturating_mul(8))
}

//! Bounded, fit-free descriptions of suspicious data characteristics.
//!
//! All thresholds are review policies, never tests of causal assumptions.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{CanonicalWriter, ExecutionContext, ValueType};
use antecedent_data::{ColumnView, TableView, TabularData};
use serde::{Deserialize, Serialize};

/// Version of the measurements, registry and report wire format.
pub const DATA_DIAGNOSTICS_VERSION: u32 = 1;

/// One independently selectable family member in the built-in diagnostic registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DataDiagnosticSpec {
    /// Stable check/finding code.
    pub code: &'static str,
    /// Policy switch controlling this check.
    pub family: &'static str,
    /// Required storage, scientific declaration or prepared context.
    pub applicability: &'static str,
}

/// Closed built-in registry. New checks can be added without changing compiler semantics.
pub const DATA_DIAGNOSTIC_CHECKS: &[DataDiagnosticSpec] = &[
    DataDiagnosticSpec {
        code: "data.distribution",
        family: "distribution",
        applicability: "numeric scalar column summaries; selected rows",
    },
    DataDiagnosticSpec {
        code: "data.treatment_imbalance",
        family: "treatment_support",
        applicability: "at least one observed declared contrast arm; full-table counts",
    },
    DataDiagnosticSpec {
        code: "data.missingness",
        family: "missingness",
        applicability: "all typed columns; exact full-table counts",
    },
    DataDiagnosticSpec {
        code: "data.missing_patterns",
        family: "missingness",
        applicability: "all typed columns; selected rows",
    },
    DataDiagnosticSpec {
        code: "data.constant",
        family: "distribution",
        applicability: "numeric measurement; at least one observed value",
    },
    DataDiagnosticSpec {
        code: "data.near_constant",
        family: "distribution",
        applicability: "numeric measurement; min_observations",
    },
    DataDiagnosticSpec {
        code: "data.repeated_values",
        family: "distribution",
        applicability: "numeric measurement; min_observations",
    },
    DataDiagnosticSpec {
        code: "data.unexpected_discretization",
        family: "distribution",
        applicability: "declared continuous measurement; resolution_min_observations",
    },
    DataDiagnosticSpec {
        code: "data.low_resolution",
        family: "distribution",
        applicability: "unspecified numeric measurement; resolution_min_observations",
    },
    DataDiagnosticSpec {
        code: "data.skewness",
        family: "distribution",
        applicability: "nonconstant numeric measurement; min_observations",
    },
    DataDiagnosticSpec {
        code: "data.heavy_tails",
        family: "distribution",
        applicability: "nonconstant numeric measurement; min_observations",
    },
    DataDiagnosticSpec {
        code: "data.duplicate_rows",
        family: "duplicate_rows",
        applicability: "all inspected typed columns; selected rows",
    },
    DataDiagnosticSpec {
        code: "data.treatment_support",
        family: "treatment_support",
        applicability: "declared numeric or boolean contrast",
    },
    DataDiagnosticSpec {
        code: "data.relationships",
        family: "relationships",
        applicability: "existing supported prepared preflight designs within declared bounds",
    },
];

/// Built-in check selection and review policy. No check fits a model.
#[allow(
    clippy::struct_excessive_bools,
    reason = "Independent built-in diagnostic families are selected explicitly."
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DataDiagnostics {
    /// Missing cells, complete cases and missingness patterns.
    pub missingness: bool,
    /// Concentration, resolution and moments of numeric columns.
    pub distribution: bool,
    /// Duplicate observations over all inspected columns.
    pub duplicate_rows: bool,
    /// Counts for declared discrete treatment contrasts.
    pub treatment_support: bool,
    /// Existing facade preflight checks when applicable and within bounds.
    pub relationships: bool,
    /// Maximum deterministic sample size; `None` requests all rows.
    pub max_sample_rows: Option<usize>,
    /// Maximum adjustment-design width for bounded preflight.
    pub max_relationship_columns: usize,
    /// Modal share at which a column is near constant.
    pub near_constant_share: f64,
    /// Modal share at which repeated measurements warrant review.
    pub repeated_share: f64,
    /// Absolute population skewness review threshold.
    pub skewness: f64,
    /// Population excess kurtosis review threshold.
    pub excess_kurtosis: f64,
    /// Missing share review threshold.
    pub missing_share: f64,
    /// Minimum share of a declared contrast arm.
    pub min_arm_share: f64,
    /// Minimum observed sample size for concentration and shape checks.
    pub min_observations: usize,
    /// Minimum observed sample size for resolution checks.
    pub resolution_min_observations: usize,
    /// Maximum distinct values for low-resolution findings.
    pub resolution_max_distinct: usize,
    /// Maximum distinct fraction for low-resolution findings.
    pub resolution_max_fraction: f64,
}

impl Default for DataDiagnostics {
    fn default() -> Self {
        Self {
            missingness: true,
            distribution: true,
            duplicate_rows: true,
            treatment_support: true,
            relationships: true,
            max_sample_rows: Some(4096),
            max_relationship_columns: 64,
            near_constant_share: 0.99,
            repeated_share: 0.5,
            skewness: 2.0,
            excess_kurtosis: 10.0,
            missing_share: 0.05,
            min_arm_share: 0.05,
            min_observations: 100,
            resolution_min_observations: 1000,
            resolution_max_distinct: 20,
            resolution_max_fraction: 0.01,
        }
    }
}

impl DataDiagnostics {
    /// Validate thresholds and bounds before reading any observations.
    pub fn validate(&self) -> Result<(), DataDiagnosticsError> {
        let shares = [
            self.near_constant_share,
            self.repeated_share,
            self.missing_share,
            self.min_arm_share,
            self.resolution_max_fraction,
        ];
        if shares.iter().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || !self.skewness.is_finite()
            || self.skewness < 0.0
            || !self.excess_kurtosis.is_finite()
            || self.excess_kurtosis < 0.0
            || self.max_sample_rows == Some(0)
            || self.max_relationship_columns == 0
            || self.min_observations < 2
            || self.resolution_min_observations < 2
            || self.resolution_max_distinct == 0
        {
            return Err(DataDiagnosticsError::Policy(
                "invalid diagnostic thresholds or bounds".into(),
            ));
        }
        Ok(())
    }
}

/// Diagnostic-only roles and declared treatment contrasts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DataDiagnosticContext {
    /// Scientific context identity (does not enter scientific program identities).
    pub plan_identity: String,
    /// Column name to role, such as treatment, outcome or adjustment.
    pub roles: BTreeMap<String, String>,
    /// Declared numeric levels per treatment column.
    pub contrasts: BTreeMap<String, Vec<f64>>,
}

/// Measured evidence and its explicit interpretation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataDiagnosticFinding {
    /// Stable registry code.
    pub code: String,
    /// `informational`, `review`, or an adapted preflight `blocking` finding.
    pub severity: String,
    /// Implicated columns.
    pub columns: Vec<String>,
    /// Roles parallel to columns.
    pub roles: Vec<String>,
    /// Finite measured quantities only.
    pub observation: BTreeMap<String, f64>,
    /// Explicit comparison rule.
    pub criterion: String,
    /// Meaning and limitations of the observation.
    pub interpretation: String,
    /// Suggested human follow-up; never automatically executed.
    pub next_action: String,
    /// Whether the observation used the bounded sample.
    pub sampled: bool,
}

/// Coverage of one built-in check on one column, or the whole table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataDiagnosticCheck {
    /// Stable check identifier.
    pub code: String,
    /// Column name; absent for table-wide checks.
    pub column: Option<String>,
    /// `evaluated`, `not_evaluated`, or `not_applicable`.
    pub status: String,
    /// Explicit reason, including disabled checks and insufficient evidence.
    pub reason: Option<String>,
    /// Number of rows inspected by this check.
    pub rows: usize,
    /// Whether this is sample rather than full-table coverage.
    pub sampled: bool,
}

/// Bounded summary of one typed column.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataDiagnosticColumn {
    /// Column name.
    pub column: String,
    /// Role supplied by the prepared context, or `other`.
    pub role: String,
    /// Declared scientific value type.
    pub value_type: String,
    /// Full-table missing or non-finite count.
    pub non_finite: usize,
    /// Observed values in the selected sample.
    pub sample_observed: usize,
    /// Distinct observed sample values, preserving typed equality.
    pub sample_distinct: usize,
    /// Largest sample value frequency divided by observed sample count.
    pub modal_share: Option<f64>,
    /// Population skewness of the sample; absent when undefined.
    pub skewness: Option<f64>,
    /// Population excess kurtosis of the sample; absent when undefined.
    pub excess_kurtosis: Option<f64>,
}

/// Immutable-by-convention Rust diagnostic value; Python exposes frozen records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataDiagnosticsReport {
    /// Report format and measurement version.
    pub version: u32,
    /// Canonical identity covering evidence, context, policy and report contents.
    pub identity: String,
    /// Content identity of the inspected table, absent for unsupported modalities.
    pub snapshot_identity: Option<String>,
    /// Diagnostic context used to interpret the table.
    pub context: DataDiagnosticContext,
    /// Exact policy used.
    pub policy: DataDiagnostics,
    /// Full-table row count.
    pub rows_total: usize,
    /// Complete rows across every inspected column, when measured.
    pub rows_complete: Option<usize>,
    /// Deterministically selected rows for bounded checks.
    pub sample_rows: usize,
    /// Columns in schema order.
    pub columns: Vec<DataDiagnosticColumn>,
    /// Sample missingness patterns, keyed by the missing column names.
    pub missing_patterns: Vec<(Vec<String>, usize)>,
    /// Sample duplicate count, excluding the first copy.
    pub duplicate_rows: Option<usize>,
    /// Full-table counts for declared treatment levels.
    pub treatment_arms: BTreeMap<String, Vec<(f64, usize)>>,
    /// Evaluation status, including checks that could not run.
    pub checks: Vec<DataDiagnosticCheck>,
    /// Advisory findings and explicitly adapted preflight findings.
    pub findings: Vec<DataDiagnosticFinding>,
}

impl DataDiagnosticsReport {
    /// Construct an explicit unavailable report without inventing observations.
    #[must_use]
    pub fn unavailable(
        policy: DataDiagnostics,
        context: DataDiagnosticContext,
        reason: &str,
    ) -> Self {
        let mut report = Self {
            version: DATA_DIAGNOSTICS_VERSION,
            identity: String::new(),
            snapshot_identity: None,
            context,
            policy,
            rows_total: 0,
            rows_complete: None,
            sample_rows: 0,
            columns: vec![],
            missing_patterns: vec![],
            duplicate_rows: None,
            treatment_arms: BTreeMap::new(),
            checks: vec![],
            findings: vec![],
        };
        for code in [
            "data.missingness",
            "data.distribution",
            "data.duplicate_rows",
            "data.treatment_support",
            "data.relationships",
        ] {
            report.check(code, None, "not_evaluated", Some(reason), 0, false);
        }
        report.seal();
        report
    }

    /// Recompute identity after facade adapters append their observations.
    pub fn seal(&mut self) {
        self.identity.clear();
        let mut writer = CanonicalWriter::new("antecedent.data_diagnostics.v1");
        // All floating values have been checked finite; BTreeMaps canonicalize key order.
        writer.text(&serde_json::to_string(self).expect("finite diagnostic report"));
        self.identity = writer.finish_hex();
    }

    /// Append coverage for a built-in check.
    pub fn check(
        &mut self,
        code: &str,
        column: Option<String>,
        status: &str,
        reason: Option<&str>,
        rows: usize,
        sampled: bool,
    ) {
        self.checks.push(DataDiagnosticCheck {
            code: code.into(),
            column,
            status: status.into(),
            reason: reason.map(str::to_owned),
            rows,
            sampled,
        });
    }

    /// Append a measured finding using the context's roles.
    #[allow(clippy::too_many_arguments)]
    pub fn finding(
        &mut self,
        code: &str,
        severity: &str,
        columns: Vec<String>,
        observation: BTreeMap<String, f64>,
        criterion: String,
        interpretation: &str,
        next_action: &str,
        sampled: bool,
    ) {
        let roles = columns
            .iter()
            .map(|name| self.context.roles.get(name).cloned().unwrap_or_else(|| "other".into()))
            .collect();
        self.findings.push(DataDiagnosticFinding {
            code: code.into(),
            severity: severity.into(),
            columns,
            roles,
            observation,
            criterion,
            interpretation: interpretation.into(),
            next_action: next_action.into(),
            sampled,
        });
    }
}

/// Operational failures of an explicit diagnostic evaluation.
#[derive(Debug, thiserror::Error)]
pub enum DataDiagnosticsError {
    /// Invalid caller policy or context.
    #[error("data diagnostics: {0}")]
    Policy(String),
    /// Storage access failed.
    #[error(transparent)]
    Data(#[from] antecedent_data::DataError),
    /// The caller cancelled the operation.
    #[error("data diagnostics cancelled")]
    Cancelled,
    /// Admission refused temporary diagnostic buffers.
    #[error("data diagnostics resource admission: {0}")]
    Resource(String),
}

fn float_key(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

// Never cast integers to f64 for equality. Missing and non-finite values share one token.
fn cell(column: ColumnView<'_>, row: usize) -> Option<Vec<u64>> {
    if !column.validity().is_valid(row) {
        return None;
    }
    match column {
        ColumnView::Float64(c) => c.values[row].is_finite().then(|| vec![float_key(c.values[row])]),
        ColumnView::Int64(c) => Some(vec![u64::from_ne_bytes(c.values[row].to_ne_bytes())]),
        ColumnView::Boolean(c) => Some(vec![u64::from(c.values[row])]),
        ColumnView::Timestamp(c) => Some(vec![u64::from_ne_bytes(c.values_ns[row].to_ne_bytes())]),
        ColumnView::Categorical(c) => Some(vec![c.codes[row].as_usize() as u64]),
        ColumnView::FixedVector(c) => {
            let values = &c.values[row * c.dim..(row + 1) * c.dim];
            values
                .iter()
                .all(|v| v.is_finite())
                .then(|| values.iter().map(|v| float_key(*v)).collect())
        }
    }
}

fn observed(column: ColumnView<'_>, row: usize) -> bool {
    if !column.validity().is_valid(row) {
        return false;
    }
    match column {
        ColumnView::Float64(c) => c.values[row].is_finite(),
        ColumnView::FixedVector(c) => {
            c.values[row * c.dim..(row + 1) * c.dim].iter().all(|v| v.is_finite())
        }
        _ => true,
    }
}

fn numeric(column: ColumnView<'_>, row: usize) -> Option<f64> {
    match column {
        ColumnView::Float64(c) => Some(c.values[row]),
        ColumnView::Int64(c) => Some(c.values[row] as f64),
        _ => None,
    }
    .filter(|v| v.is_finite())
}

fn moments(values: &[f64]) -> (Option<f64>, Option<f64>) {
    let scale = values.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    if values.len() < 2 || scale == 0.0 {
        return (None, None);
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|v| v / scale).sum::<f64>() / n;
    let mut m = [0.0; 3];
    for value in values {
        let d = value / scale - mean;
        m[0] += d * d / n;
        m[1] += d.powi(3) / n;
        m[2] += d.powi(4) / n;
    }
    if m[0] <= 0.0 {
        return (None, None);
    }
    let skew = m[1] / m[0].powf(1.5);
    let kurtosis = m[2] / m[0].powi(2) - 3.0;
    (skew.is_finite().then_some(skew), kurtosis.is_finite().then_some(kurtosis))
}

fn observation(key: &str, value: f64) -> BTreeMap<String, f64> {
    BTreeMap::from([(key.into(), value)])
}

/// Profile typed data without fitting or modifying scientific inputs.
///
/// Sampling selects evenly spaced row indices, independent of execution seed and threads.
/// Counts use all rows; sample statistics make no random-sampling confidence claim.
#[allow(clippy::too_many_lines)]
pub fn diagnose_data(
    data: &TabularData,
    policy: &DataDiagnostics,
    context: &DataDiagnosticContext,
    ctx: &ExecutionContext,
) -> Result<DataDiagnosticsReport, DataDiagnosticsError> {
    policy.validate()?;
    for (name, levels) in &context.contrasts {
        data.schema().id_of(name).map_err(antecedent_data::DataError::from)?;
        if levels.is_empty() || levels.iter().any(|v| !v.is_finite()) {
            return Err(DataDiagnosticsError::Policy(
                "contrasts require finite declared levels".into(),
            ));
        }
    }
    for name in context.roles.keys() {
        data.schema().id_of(name).map_err(antecedent_data::DataError::from)?;
    }
    if ctx.cancellation.is_cancelled() {
        return Err(DataDiagnosticsError::Cancelled);
    }
    let n = data.row_count();
    let k = policy.max_sample_rows.unwrap_or(n).min(n);
    let width = data.schema().len();
    let vector_width: usize = data
        .schema()
        .variables()
        .iter()
        .map(|v| match data.column(v.id) {
            Ok(ColumnView::FixedVector(c)) => c.dim,
            _ => 1,
        })
        .sum();
    // Missing-pattern keys and JSON export repeat names; column count alone cannot
    // bound those buffers when a valid schema uses long measurement labels.
    let name_bytes = data
        .schema()
        .variables()
        .iter()
        .fold(0_u64, |bytes, variable| bytes.saturating_add(variable.name.len() as u64));
    let context_bytes = context
        .roles
        .iter()
        .fold(context.plan_identity.len() as u64, |bytes, (name, role)| {
            bytes.saturating_add(name.len() as u64).saturating_add(role.len() as u64)
        })
        .saturating_add(context.contrasts.iter().fold(0_u64, |bytes, (name, levels)| {
            bytes
                .saturating_add(name.len() as u64)
                .saturating_add((levels.len() as u64).saturating_mul(16))
        }));
    let bytes = (k as u64)
        .saturating_mul(vector_width as u64)
        .saturating_mul(256)
        .saturating_add((k as u64).saturating_mul(name_bytes).saturating_mul(2))
        .saturating_add((width as u64).saturating_mul(8192))
        .saturating_add(name_bytes.saturating_mul(64))
        .saturating_add(context_bytes.saturating_mul(16));
    let _lease = ctx.acquire_admission(0, bytes).map_err(|e| {
        if ctx.cancellation.is_cancelled() {
            DataDiagnosticsError::Cancelled
        } else {
            DataDiagnosticsError::Resource(e.to_string())
        }
    })?;
    let sample: Vec<usize> =
        (0..k).map(|i| ((i as u128 * n as u128) / k as u128) as usize).collect();
    let sampled = k < n;
    let mut report = DataDiagnosticsReport::unavailable(policy.clone(), context.clone(), "pending");
    report.checks.clear();
    report.snapshot_identity = Some(crate::staleness::snapshot_identity(data));
    report.rows_total = n;
    report.sample_rows = k;
    let views = data
        .schema()
        .variables()
        .iter()
        .map(|v| data.column(v.id))
        .collect::<Result<Vec<_>, _>>()?;
    let mut missing = vec![0_usize; width];
    let mut complete = 0;
    for row in 0..n {
        if row % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(DataDiagnosticsError::Cancelled);
        }
        let mut all = true;
        for (index, view) in views.iter().enumerate() {
            if !observed(*view, row) {
                missing[index] += 1;
                all = false;
            }
        }
        complete += usize::from(all);
    }
    report.rows_complete = Some(complete);
    report.check(
        "data.missingness",
        None,
        if policy.missingness { "evaluated" } else { "not_evaluated" },
        (!policy.missingness).then_some("disabled by suite"),
        n,
        false,
    );
    let mut row_keys: BTreeMap<Vec<Option<Vec<u64>>>, usize> = BTreeMap::new();
    let mut patterns: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    if policy.duplicate_rows || policy.missingness {
        for (sample_index, &row) in sample.iter().enumerate() {
            if sample_index % 1024 == 0 && ctx.cancellation.is_cancelled() {
                return Err(DataDiagnosticsError::Cancelled);
            }
            let key = views.iter().map(|v| cell(*v, row)).collect::<Vec<_>>();
            if policy.missingness {
                let pattern = key
                    .iter()
                    .zip(data.schema().variables())
                    .filter(|(cell, _)| cell.is_none())
                    .map(|(_, v)| v.name.to_string())
                    .collect::<Vec<_>>();
                *patterns.entry(pattern).or_default() += 1;
            }
            if policy.duplicate_rows {
                *row_keys.entry(key).or_default() += 1;
            }
        }
    }
    report.missing_patterns = patterns.into_iter().collect();
    report.check(
        "data.missing_patterns",
        None,
        if policy.missingness { "evaluated" } else { "not_evaluated" },
        (!policy.missingness).then_some("disabled by suite"),
        if policy.missingness { k } else { 0 },
        sampled,
    );
    if policy.duplicate_rows {
        let duplicates = k - row_keys.len();
        report.duplicate_rows = Some(duplicates);
        if duplicates > 0 {
            report.finding("data.duplicate_rows", "review", data.schema().variables().iter().map(|v| v.name.to_string()).collect(),
                observation("duplicate_rows", duplicates as f64), "duplicate_rows > 0".into(),
                "Repeated observations over the inspected columns; repeated measurements may be intentional.",
                "Inspect row provenance and repeated-measurement design.", sampled);
        }
    }
    report.check(
        "data.duplicate_rows",
        None,
        if policy.duplicate_rows { "evaluated" } else { "not_evaluated" },
        (!policy.duplicate_rows).then_some("disabled by suite"),
        if policy.duplicate_rows { k } else { 0 },
        sampled,
    );
    for (index, (variable, view)) in data.schema().variables().iter().zip(&views).enumerate() {
        let name = variable.name.to_string();
        let mut frequencies = BTreeMap::new();
        let mut values = vec![];
        for (sample_index, &row) in sample.iter().enumerate() {
            if sample_index % 1024 == 0 && ctx.cancellation.is_cancelled() {
                return Err(DataDiagnosticsError::Cancelled);
            }
            if let Some(key) = cell(*view, row) {
                *frequencies.entry(key).or_insert(0_usize) += 1;
                if let Some(value) = numeric(*view, row) {
                    values.push(value);
                }
            }
        }
        let observed: usize = frequencies.values().sum();
        let distinct = frequencies.len();
        let modal = (observed > 0)
            .then(|| *frequencies.values().max().expect("observed") as f64 / observed as f64);
        let (skew, kurtosis) = moments(&values);
        report.columns.push(DataDiagnosticColumn {
            column: name.clone(),
            role: context.roles.get(&name).cloned().unwrap_or_else(|| "other".into()),
            value_type: format!("{:?}", variable.value_type).to_lowercase(),
            non_finite: missing[index],
            sample_observed: observed,
            sample_distinct: distinct,
            modal_share: modal,
            skewness: skew,
            excess_kurtosis: kurtosis,
        });
        if policy.missingness
            && n > 0
            && missing[index] > 0
            && missing[index] as f64 / n as f64 >= policy.missing_share
        {
            report.finding(
                "data.missingness",
                "review",
                vec![name.clone()],
                observation("missing_share", missing[index] as f64 / n as f64),
                format!("missing_share >= {}", policy.missing_share),
                "Missing or non-finite observations reduce available information.",
                "Inspect collection failures and the declared missing-data policy.",
                false,
            );
        }
        let supported = matches!(view, ColumnView::Float64(_) | ColumnView::Int64(_));
        let measurement = supported
            && matches!(
                variable.value_type,
                ValueType::Continuous | ValueType::Unspecified | ValueType::Count
            );
        for (code, minimum, applicable) in [
            ("data.constant", 1, measurement),
            ("data.near_constant", policy.min_observations, measurement),
            ("data.repeated_values", policy.min_observations, measurement),
            (
                "data.unexpected_discretization",
                policy.resolution_min_observations,
                supported && variable.value_type == ValueType::Continuous,
            ),
            (
                "data.low_resolution",
                policy.resolution_min_observations,
                supported && variable.value_type == ValueType::Unspecified,
            ),
        ] {
            let (status, reason) = if !policy.distribution {
                ("not_evaluated", Some("disabled by suite"))
            } else if !applicable {
                ("not_applicable", Some("requires the declared numeric measurement type"))
            } else if observed < minimum {
                ("not_evaluated", Some("insufficient observed sample values"))
            } else {
                ("evaluated", None)
            };
            report.check(code, Some(name.clone()), status, reason, k, sampled);
        }

        if !policy.distribution || !supported || observed == 0 {
            report.check(
                "data.distribution",
                Some(name),
                if !policy.distribution || observed == 0 {
                    "not_evaluated"
                } else {
                    "not_applicable"
                },
                Some(if !policy.distribution {
                    "disabled by suite"
                } else if observed == 0 {
                    "no observed values"
                } else {
                    "numeric scalar storage required"
                }),
                k,
                sampled,
            );
            continue;
        }
        report.check("data.distribution", Some(name.clone()), "evaluated", None, k, sampled);
        let numeric_scale = matches!(
            variable.value_type,
            ValueType::Continuous | ValueType::Unspecified | ValueType::Count
        );
        if numeric_scale && distinct == 1 {
            report.finding(
                "data.constant",
                "review",
                vec![name.clone()],
                observation("distinct", 1.0),
                "observed distinct values == 1".into(),
                "The inspected observations have no variation.",
                "Check measurement, coding and collection.",
                sampled,
            );
        } else if numeric_scale && observed >= policy.min_observations {
            for (code, threshold) in [
                ("data.near_constant", policy.near_constant_share),
                ("data.repeated_values", policy.repeated_share),
            ] {
                if modal.is_some_and(|v| v >= threshold) {
                    report.finding(code, "review", vec![name.clone()], observation("modal_share", modal.unwrap()), format!("modal_share >= {threshold}"),
                        "Many inspected measurements share one value; this can be intentional discretization or saturation.",
                        "Check the measurement scale, rounding and collection process.", sampled);
                    break;
                }
            }
        }
        if matches!(variable.value_type, ValueType::Continuous | ValueType::Unspecified)
            && observed >= policy.resolution_min_observations
            && distinct <= policy.resolution_max_distinct
            && distinct as f64 / observed as f64 <= policy.resolution_max_fraction
        {
            let declared = variable.value_type == ValueType::Continuous;
            report.finding(if declared { "data.unexpected_discretization" } else { "data.low_resolution" },
                if declared { "review" } else { "informational" }, vec![name.clone()],
                BTreeMap::from([("distinct".into(), distinct as f64), ("observed".into(), observed as f64)]),
                format!("observed >= {}; distinct <= {}; distinct/observed <= {}", policy.resolution_min_observations, policy.resolution_max_distinct, policy.resolution_max_fraction),
                "Observed resolution is low; this describes measurements, not their scientific validity.",
                "Check declared value type, measurement scale and rounding.", sampled);
        }
        for (code, value, threshold) in [
            ("data.skewness", skew.map(f64::abs), policy.skewness),
            ("data.heavy_tails", kurtosis, policy.excess_kurtosis),
        ] {
            if numeric_scale
                && values.len() >= policy.min_observations
                && value.is_some_and(|v| v > threshold)
            {
                report.finding(
                    code,
                    "review",
                    vec![name.clone()],
                    observation("statistic", value.unwrap()),
                    format!("statistic > {threshold}; observed >= {}", policy.min_observations),
                    "An unusual distribution shape is descriptive evidence, not a model rejection.",
                    "Inspect the distribution and the appropriateness of declared outcome models.",
                    sampled,
                );
            }
            report.check(
                code,
                Some(name.clone()),
                if values.len() >= policy.min_observations && value.is_some() {
                    "evaluated"
                } else {
                    "not_evaluated"
                },
                (values.len() < policy.min_observations || value.is_none())
                    .then_some("insufficient observations or undefined moment"),
                k,
                sampled,
            );
        }
    }
    bind_context(data, &mut report, context, ctx)?;
    report.check(
        "data.relationships",
        None,
        "not_evaluated",
        Some("requires prepared preflight adapter"),
        0,
        false,
    );
    report.seal();
    Ok(report)
}

/// Bind an existing profile to new scientific roles and contrasts without repeating profiling.
/// The caller must have matched table snapshot and diagnostic policy before reuse.
#[allow(
    clippy::too_many_lines,
    reason = "Rebinding applies one contrast counting and applicability policy without re-profiling."
)]
pub fn bind_context(
    data: &TabularData,
    report: &mut DataDiagnosticsReport,
    context: &DataDiagnosticContext,
    ctx: &ExecutionContext,
) -> Result<(), DataDiagnosticsError> {
    if ctx.cancellation.is_cancelled() {
        return Err(DataDiagnosticsError::Cancelled);
    }
    for name in context.roles.keys() {
        data.schema().id_of(name).map_err(antecedent_data::DataError::from)?;
    }
    report.context = context.clone();
    for column in &mut report.columns {
        column.role = context.roles.get(&column.column).cloned().unwrap_or_else(|| "other".into());
    }
    report.findings.retain(|f| f.code != "data.treatment_imbalance");
    for finding in &mut report.findings {
        finding.roles = finding
            .columns
            .iter()
            .map(|name| context.roles.get(name).cloned().unwrap_or_else(|| "other".into()))
            .collect();
    }
    report.treatment_arms.clear();
    report.checks.retain(|c| c.code != "data.treatment_support");
    let policy = report.policy.clone();
    if !policy.treatment_support {
        report.check(
            "data.treatment_support",
            None,
            "not_evaluated",
            Some("disabled by suite"),
            0,
            false,
        );
    } else if context.contrasts.is_empty() {
        report.check(
            "data.treatment_support",
            None,
            "not_applicable",
            Some("no declared discrete contrast"),
            0,
            false,
        );
    } else {
        for (name, levels) in &context.contrasts {
            if levels.is_empty()
                || levels.iter().any(|v| !v.is_finite())
                || levels.iter().enumerate().any(|(i, v)| levels[..i].contains(v))
            {
                return Err(DataDiagnosticsError::Policy(
                    "contrasts require distinct finite declared levels".into(),
                ));
            }
            let id = data.schema().id_of(name).map_err(antecedent_data::DataError::from)?;
            let view = data.column(id)?;
            if !matches!(
                view,
                ColumnView::Float64(_) | ColumnView::Int64(_) | ColumnView::Boolean(_)
            ) {
                report.check(
                    "data.treatment_support",
                    Some(name.clone()),
                    "not_applicable",
                    Some("numeric or boolean contrast storage required"),
                    0,
                    false,
                );
                continue;
            }
            let mut counts: Vec<(f64, usize)> = levels.iter().map(|v| (*v, 0)).collect();
            for row in 0..data.row_count() {
                if row % 1024 == 0 && ctx.cancellation.is_cancelled() {
                    return Err(DataDiagnosticsError::Cancelled);
                }
                if !observed(view, row) {
                    continue;
                }
                for (level, count) in &mut counts {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::float_cmp,
                        reason = "Declared levels use exact equality; integer conversion follows range and integrality checks."
                    )]
                    let matches = match view {
                        ColumnView::Int64(c) => {
                            // A declared f64 level must round-trip to the exact stored integer.
                            let value = c.values[row];
                            *level >= i64::MIN as f64
                                && *level < -(i64::MIN as f64)
                                && level.fract() == 0.0
                                && (*level as i64) == value
                        }
                        ColumnView::Boolean(c) => f64::from(c.values[row]) == *level,
                        _ => numeric(view, row).is_some_and(|value| value == *level),
                    };
                    *count += usize::from(matches);
                }
            }
            let total: usize = counts.iter().map(|(_, count)| count).sum();
            let least = counts.iter().map(|(_, count)| *count).min().unwrap_or(0);
            if total > 0 && (least as f64 / total as f64) < policy.min_arm_share {
                report.finding("data.treatment_imbalance", "review", vec![name.clone()], observation("smallest_arm_share", least as f64 / total as f64),
                    format!("smallest_arm_share < {}", policy.min_arm_share), "Declared contrast arms are unevenly represented; this is not an identification verdict.",
                    "Inspect treatment support and consider explicit propensity diagnostics.", false);
            }
            let fully_discrete =
                report.columns.iter().find(|column| column.column == *name).is_some_and(|column| {
                    total == data.row_count().saturating_sub(column.non_finite)
                });
            if fully_discrete {
                let codes = [
                    "data.constant",
                    "data.near_constant",
                    "data.repeated_values",
                    "data.unexpected_discretization",
                    "data.low_resolution",
                    "data.skewness",
                    "data.heavy_tails",
                ];
                report.findings.retain(|finding| {
                    !(codes.contains(&finding.code.as_str()) && finding.columns == [name.clone()])
                });
                for check in &mut report.checks {
                    if check.column.as_deref() == Some(name)
                        && (check.code == "data.distribution"
                            || codes.contains(&check.code.as_str()))
                    {
                        check.status = "not_applicable".into();
                        check.reason = Some("declared discrete treatment contrast".into());
                    }
                }
            }
            report.treatment_arms.insert(name.clone(), counts);
            report.check(
                "data.treatment_support",
                Some(name.clone()),
                if total == 0 { "not_evaluated" } else { "evaluated" },
                (total == 0).then_some("no observations at declared contrast levels"),
                data.row_count(),
                false,
            );
        }
    }
    report.seal();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;
    use antecedent_data::{Int64Column, ValidityBitmap};
    use std::sync::Arc;

    fn profile(values: &[f64], policy: DataDiagnostics) -> DataDiagnosticsReport {
        let data = TabularData::from_f64_columns([("x", values)]).unwrap();
        diagnose_data(
            &data,
            &policy,
            &DataDiagnosticContext::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap()
    }

    #[test]
    fn population_moments_match_hand_arithmetic() {
        // Uniform {-2,-1,0,1,2}: E[X²]=2, E[X⁴]=34/5, excess=1.7-3.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/data_diagnostics/basic/fixture.json"
        ))
        .unwrap();
        let values: Vec<f64> = fixture["columns"]["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let report = profile(&values, DataDiagnostics::default());
        assert!(report.columns[0].skewness.unwrap().abs() < 1e-14);
        assert!((report.columns[0].excess_kurtosis.unwrap() + 1.3).abs() < 1e-14);
    }

    #[test]
    fn sampling_is_bounded_and_identities_bind_policy_and_evidence() {
        let values: Vec<f64> = (0..10_000).map(f64::from).collect();
        let a = profile(&values, DataDiagnostics::default());
        let b = profile(&values, DataDiagnostics::default());
        assert_eq!(a.identity, b.identity);
        assert_eq!(a.sample_rows, 4096);
        assert_eq!(a.columns[0].sample_distinct, 4096);
        assert!(a.checks.iter().any(|c| c.sampled));
        let c = profile(
            &values,
            DataDiagnostics { max_sample_rows: None, ..DataDiagnostics::default() },
        );
        assert_ne!(a.identity, c.identity);
        assert_eq!(c.columns[0].sample_distinct, 10_000);
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn nonfinite_counts_are_exact_but_repetitions_are_sampled() {
        let report =
            profile(&[0., -0., f64::NAN, f64::INFINITY, 2., 2.], DataDiagnostics::default());
        assert_eq!(report.rows_complete, Some(4));
        assert_eq!(report.columns[0].non_finite, 2);
        assert_eq!(report.columns[0].sample_distinct, 2);
        assert_eq!(report.duplicate_rows, Some(3)); // zero, missing and two each repeat.
        assert_eq!(report.missing_patterns.iter().map(|(_, n)| n).sum::<usize>(), 6);
    }

    #[test]
    fn typed_integer_equality_does_not_round_above_two_to_the_53() {
        let column = Int64Column::new(
            VariableId::from_raw(0),
            Arc::from([9_007_199_254_740_992_i64, 9_007_199_254_740_993]),
            ValidityBitmap::all_valid(2),
        )
        .unwrap();
        assert_ne!(cell(ColumnView::Int64(&column), 0), cell(ColumnView::Int64(&column), 1));
    }

    #[test]
    fn concentration_and_resolution_thresholds_are_inclusive() {
        let mut values = vec![0.; 1000];
        values[990..].fill(1.);
        let report = profile(&values, DataDiagnostics::default());
        assert!(report.findings.iter().any(|f| f.code == "data.near_constant"));
        let resolution =
            report.findings.iter().find(|f| f.code == "data.unexpected_discretization").unwrap();
        assert_eq!(resolution.severity, "review");
        assert_eq!(resolution.observation["observed"], 1000.);
        let small = profile(&values[..99], DataDiagnostics::default());
        assert!(!small.findings.iter().any(|f| f.code == "data.near_constant"));
    }

    #[test]
    fn constant_all_missing_and_extreme_values_never_fabricate_moments() {
        for values in [vec![1.; 100], vec![f64::NAN; 100]] {
            let report = profile(&values, DataDiagnostics::default());
            assert_eq!(report.columns[0].skewness, None);
            assert_eq!(report.columns[0].excess_kurtosis, None);
            assert!(report.checks.iter().any(|c| c.status == "not_evaluated"));
            serde_json::to_string(&report).unwrap();
        }
        let extreme = profile(&[-f64::MAX, 0., f64::MAX], DataDiagnostics::default());
        assert_eq!(extreme.columns[0].skewness, Some(0.));
        assert!((extreme.columns[0].excess_kurtosis.unwrap() + 1.5).abs() < 1e-14);
    }

    #[test]
    fn explicit_resource_and_cancellation_failures_are_operational() {
        let data = TabularData::from_f64_columns([("x", &[0., 1.][..])]).unwrap();
        let context = DataDiagnosticContext::default();
        let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(1));
        assert!(matches!(
            diagnose_data(&data, &DataDiagnostics::default(), &context, &ctx),
            Err(DataDiagnosticsError::Resource(_))
        ));
        let long_name = "x".repeat(8192);
        let long_labels =
            TabularData::from_f64_columns([(long_name.as_str(), &[0., 1.][..])]).unwrap();
        let limited = ExecutionContext::for_tests(1).with_memory_limit(Some(100_000));
        assert!(matches!(
            diagnose_data(&long_labels, &DataDiagnostics::default(), &context, &limited),
            Err(DataDiagnosticsError::Resource(_))
        ));
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        assert!(matches!(
            diagnose_data(&data, &DataDiagnostics::default(), &context, &ctx),
            Err(DataDiagnosticsError::Cancelled)
        ));
    }

    #[test]
    fn policy_and_context_are_validated_before_reading_data() {
        let data = TabularData::from_f64_columns([("x", &[0., 1.][..])]).unwrap();
        let policy = DataDiagnostics { skewness: f64::NAN, ..DataDiagnostics::default() };
        assert!(matches!(
            diagnose_data(
                &data,
                &policy,
                &DataDiagnosticContext::default(),
                &ExecutionContext::for_tests(1)
            ),
            Err(DataDiagnosticsError::Policy(_))
        ));
        let context = DataDiagnosticContext {
            contrasts: BTreeMap::from([("absent".into(), vec![0., 1.])]),
            ..DataDiagnosticContext::default()
        };
        assert!(
            diagnose_data(
                &data,
                &DataDiagnostics::default(),
                &context,
                &ExecutionContext::for_tests(1)
            )
            .is_err()
        );
    }
}

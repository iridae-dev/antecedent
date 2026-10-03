//! Preflight diagnostics: what a table can and cannot support before any nuisance is fit.
//!
//! Two kinds of check, kept in separate types so one is never read as the other:
//!
//! - [`PreflightReport`] (`diagnose`) is **fit-free**. It reads the retained table and the
//!   certified adjustment set only: complete-case and arm counts, missingness, exact
//!   duplicate columns, the numerical rank of the `[1 | Z]` design with the dependent
//!   columns named, and bounded near-determinism indicators raised for human review.
//! - [`NuisanceFitDiagnostics`] (`diagnose_fit`) **fits a propensity model** and reports the
//!   fitted-score range and per-arm effective sample size. When that fit fails before any
//!   score exists, every fitted quantity is explicitly absent, never reconstructed.
//!
//! Review flags (a column that tracks the treatment or the outcome, a near-collinear pair,
//! a column that separates the arms) are predictive facts about this table. They are not
//! causal statements and never change the adjustment set or the identification verdict. Only
//! a numerically rank-deficient design or an unpopulated arm is *blocking*, because the
//! regression-based nuisance fits refuse those inputs.
//!
//! The opt-in rank-deficiency drop ([`plan_rank_drop`]) declares a deterministic column
//! priority first and returns a plan: the dropped columns, the exact linear relation that
//! makes each redundant, and the resulting design identity. It refuses (rather than drops)
//! when a treatment, outcome or effect modifier is involved. The plan is a record; no
//! estimator is re-run on the reduced design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, VariableId, reason_code,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{EstimationError, ExactF64, RefusalFields};
use antecedent_stats::{
    FaerBackend, GlmOptions, PropensityWorkspace, StatsError, fit_propensity_diagnostic,
};
use serde::Serialize;

use super::batch::PreparedBatch;
use super::builder::DataInput;
use super::execute::Study;
use super::prepared::PreparedStudy;
use crate::error::CausalError;

const STAGE_PREFLIGHT: &str = "preflight";
const STAGE_RANK_DROP: &str = "rank_drop";

/// A kept column whose residual, after projecting out the higher-priority columns, falls
/// below this fraction of its own norm is flagged as near collinear. A review threshold, not
/// a verdict.
pub const NEAR_COLLINEAR_RATIO: f64 = 1e-3;

/// A linear relation with `R² >=` this value is flagged as near deterministic. A review
/// threshold, not a verdict.
pub const NEAR_DETERMINISTIC_R2: f64 = 0.999;

/// Largest table the fit-free checks will copy (rows times columns read). Larger inputs are
/// refused as a resource limit rather than materialized.
pub const MAX_PREFLIGHT_CELLS: usize = 100_000_000;

/// Linear-relation coefficients smaller than this (in original column units) are not listed.
const COEFFICIENT_FLOOR: f64 = 1e-8;

/// Probabilities at which fitted propensities are reported (nearest-rank quantiles).
const PROPENSITY_QUANTILES: [f64; 7] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];

/// How loudly a finding speaks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    /// Raised for human review; changes nothing.
    Review,
    /// The regression-based nuisance fits would refuse this input.
    Blocking,
}

/// One preflight observation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreflightFinding {
    /// Stable finding code.
    pub code: &'static str,
    /// Review flag or blocking finding.
    pub severity: FindingSeverity,
    /// Columns the finding implicates, by name.
    pub columns: Vec<String>,
    /// The measured quantity behind the finding, when it has one.
    pub measure: Option<f64>,
    /// The review threshold the measure was compared with, when it has one.
    pub threshold: Option<f64>,
    /// What was observed, and what it does and does not mean.
    pub detail: String,
}

/// Complete-case rows in one treatment arm or joint cell (unweighted).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArmCount {
    /// Arm role (`control`, `active`, `cell`) and its coded levels.
    pub label: String,
    /// Complete-case rows in the arm. For an unweighted arm this is also its effective
    /// sample size.
    pub rows: usize,
}

/// Non-finite cells (missing values read as `NaN`) in one column.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ColumnMissingness {
    /// Column name.
    pub column: String,
    /// Rows whose value is missing or non-finite.
    pub non_finite: usize,
}

/// Columns that are bit-for-bit equal on the complete-case rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DuplicateGroup {
    /// Column names in schema role order (treatments, outcome, then the adjustment set).
    pub columns: Vec<String>,
}

/// One term of an exact linear relation, in original column units.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ColumnWeight {
    /// Column name (`(intercept)` for the constant column).
    pub column: String,
    /// Coefficient on that column.
    pub coefficient: f64,
}

/// A design column that is numerically a linear function of higher-priority columns.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DependentColumn {
    /// Column name.
    pub column: String,
    /// Norm of the residual after projecting out the higher-priority columns, as a fraction
    /// of the column's own norm.
    pub residual_ratio: f64,
    /// The relation `column = sum(coefficient * explained_by)` it satisfies up to the
    /// tolerance. Empty for an all-zero column.
    pub explained_by: Vec<ColumnWeight>,
}

/// Numerical rank of the `[1 | Z]` design on the complete-case rows.
///
/// Columns are scanned in priority order (the adjustment-set order here); a column is
/// dependent when earlier kept columns explain it. The rank does not depend on that order;
/// which columns are *named* dependent does.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankReport {
    /// Design columns, the intercept included.
    pub design_columns: usize,
    /// Numerical rank.
    pub numerical_rank: usize,
    /// Residual-ratio tolerance below which a column counts as dependent
    /// (`max(rows, columns) * f64::EPSILON`).
    pub tolerance: f64,
    /// Dependent columns, in scan order.
    pub dependent: Vec<DependentColumn>,
}

/// Fit-free preflight report for one prepared plan.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreflightReport {
    /// Plan label (treatment or cell, and outcome).
    pub subject: String,
    /// Treatment columns.
    pub treatment: Vec<String>,
    /// Outcome column.
    pub outcome: String,
    /// The adjustment set this report read, unchanged.
    pub adjustment_set: Vec<String>,
    /// Rows in the table.
    pub rows_total: usize,
    /// Rows where every column above is finite.
    pub rows_complete: usize,
    /// Non-finite cells per column.
    pub missingness: Vec<ColumnMissingness>,
    /// Complete-case rows per arm or cell.
    pub arms: Vec<ArmCount>,
    /// Complete-case rows in none of the listed arms.
    pub rows_other_levels: usize,
    /// Exact duplicate columns.
    pub duplicates: Vec<DuplicateGroup>,
    /// Numerical rank; absent when there are no complete-case rows to evaluate it on.
    pub rank: Option<RankReport>,
    /// Review flags and blocking findings.
    pub findings: Vec<PreflightFinding>,
}

impl PreflightReport {
    /// Whether any finding is blocking.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.findings.iter().any(|f| f.severity == FindingSeverity::Blocking)
    }

    /// The typed refusal for the first blocking finding, with structured fields, or `None`
    /// when nothing blocks. Review flags never produce a refusal.
    #[must_use]
    pub fn refusal(&self) -> Option<CausalError> {
        let finding = self.findings.iter().find(|f| f.severity == FindingSeverity::Blocking)?;
        if finding.code == "arm_not_populated" {
            return Some(arm_refusal(&self.subject, &self.arms, &finding.columns));
        }
        let rank = self.rank.as_ref();
        let fields = RefusalFields {
            stage: Some(STAGE_PREFLIGHT.to_string()),
            subject: Some(self.subject.clone()),
            reason: Some(finding.detail.clone()),
            numerical_rank: rank.map(|r| as_u64(r.numerical_rank)),
            design_columns: rank.map(|r| as_u64(r.design_columns)),
            implicated_columns: finding.columns.clone(),
            remedy: Some(
                "remove or merge the named dependent columns, or opt in to a declared-priority \
                 rank drop with plan_rank_drop"
                    .to_string(),
            ),
            ..RefusalFields::default()
        };
        Some(
            EstimationError::refused_with_fields(
                reason_code!("design_rank_deficient"),
                finding.detail.clone(),
                fields,
            )
            .into(),
        )
    }
}

/// Fit-free preflight for every plan of a prepared batch.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchPreflightReport {
    /// One report per plan, in query order.
    pub reports: Vec<PreflightReport>,
    /// Whether the batch froze one shared covariate design.
    pub shares_covariates: bool,
}

impl BatchPreflightReport {
    /// Indexes of plans with a blocking finding.
    #[must_use]
    pub fn blocked_plans(&self) -> Vec<usize> {
        self.reports.iter().enumerate().filter(|(_, r)| r.is_blocked()).map(|(i, _)| i).collect()
    }
}

/// One treatment arm or joint cell to count rows in.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmSpec {
    /// Role label (`control`, `active`, `cell`).
    pub role: &'static str,
    /// Treatment columns and the coded level each takes in the arm.
    pub levels: Vec<(VariableId, f64)>,
}

/// What a preflight reads: the table and the declared design.
#[derive(Clone, Debug)]
pub struct PreflightInput<'a> {
    /// The table.
    pub data: &'a TabularData,
    /// Treatment columns (several for a joint cell).
    pub treatments: Vec<VariableId>,
    /// Outcome column.
    pub outcome: VariableId,
    /// The certified adjustment set, in the order that fixes the default column priority.
    pub adjustment: Vec<VariableId>,
    /// Extra columns the query needs and a rank drop must never remove (effect modifiers).
    pub protected: Vec<VariableId>,
    /// Arms or cells to count. One arm is a joint cell; two are `[control, active]`.
    pub arms: Vec<ArmSpec>,
}

impl<'a> PreflightInput<'a> {
    /// A binary contrast between two coded levels of one treatment.
    #[must_use]
    pub fn binary_effect(
        data: &'a TabularData,
        treatment: VariableId,
        outcome: VariableId,
        adjustment: &[VariableId],
        control: f64,
        active: f64,
    ) -> Self {
        Self {
            data,
            treatments: vec![treatment],
            outcome,
            adjustment: adjustment.to_vec(),
            protected: Vec::new(),
            arms: vec![
                ArmSpec { role: "control", levels: vec![(treatment, control)] },
                ArmSpec { role: "active", levels: vec![(treatment, active)] },
            ],
        }
    }

    /// One discrete joint cell: every listed treatment at its coded level.
    #[must_use]
    pub fn joint_cell(
        data: &'a TabularData,
        outcome: VariableId,
        adjustment: &[VariableId],
        levels: Vec<(VariableId, f64)>,
    ) -> Self {
        let mut treatments: Vec<VariableId> = Vec::new();
        for (variable, _) in &levels {
            if !treatments.contains(variable) {
                treatments.push(*variable);
            }
        }
        Self {
            data,
            treatments,
            outcome,
            adjustment: adjustment.to_vec(),
            protected: Vec::new(),
            arms: vec![ArmSpec { role: "cell", levels }],
        }
    }
}

/// Complete-case table the checks run on.
struct Frame<'a> {
    input: &'a PreflightInput<'a>,
    subject: String,
    /// Names in load order: treatments, outcome, adjustment set.
    names: Vec<String>,
    /// Complete-case columns in load order.
    columns: Vec<Vec<f64>>,
    rows_total: usize,
    rows_complete: usize,
    missingness: Vec<ColumnMissingness>,
    arms: Vec<ArmCount>,
    rows_other: usize,
    /// Per complete row: `Some(true)` active or in-cell, `Some(false)` control or outside the
    /// cell, `None` in neither arm of a binary contrast.
    indicator: Vec<Option<bool>>,
}

impl<'a> Frame<'a> {
    fn new(input: &'a PreflightInput<'a>) -> Result<Self, CausalError> {
        if input.treatments.is_empty() || input.arms.is_empty() {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "preflight needs a treatment column and at least one arm"
            ));
        }
        let data: &'a TabularData = input.data;
        let rows_total = data.row_count();
        let nt = input.treatments.len();
        let width = nt + 1 + input.adjustment.len();
        if rows_total.saturating_mul(width) > MAX_PREFLIGHT_CELLS {
            return Err(CausalError::Resource {
                message: format!(
                    "preflight would read {rows_total} rows x {width} columns, over the \
                     {MAX_PREFLIGHT_CELLS}-cell limit; run it on a subset or fewer columns"
                ),
            });
        }
        let schema = data.schema();
        let mut ids = input.treatments.clone();
        ids.push(input.outcome);
        ids.extend_from_slice(&input.adjustment);
        let mut names = Vec::with_capacity(ids.len());
        let mut raw: Vec<Cow<'a, [f64]>> = Vec::with_capacity(ids.len());
        for &id in &ids {
            names.push(schema.get(id)?.name.to_string());
            raw.push(data.float64_cow(id)?);
        }
        let complete: Vec<usize> =
            (0..rows_total).filter(|&i| raw.iter().all(|c| c[i].is_finite())).collect();
        let missingness = names
            .iter()
            .zip(&raw)
            .map(|(name, col)| ColumnMissingness {
                column: name.clone(),
                non_finite: col.iter().filter(|v| !v.is_finite()).count(),
            })
            .collect();
        let columns: Vec<Vec<f64>> =
            raw.iter().map(|col| complete.iter().map(|&i| col[i]).collect()).collect();

        let mut resolved = Vec::with_capacity(input.arms.len());
        for arm in &input.arms {
            let mut levels = Vec::with_capacity(arm.levels.len());
            for &(variable, level) in &arm.levels {
                let col =
                    input.treatments.iter().position(|t| *t == variable).ok_or_else(|| {
                        crate::compile_reason!(
                            "invalid_argument",
                            "an arm sets a variable that is not one of the treatment columns"
                        )
                    })?;
                levels.push((col, level));
            }
            resolved.push(levels);
        }
        let in_arm = |levels: &[(usize, f64)], row: usize| {
            levels.iter().all(|&(col, level)| same_level(columns[col][row], level))
        };
        let n_complete = complete.len();
        let mut counts = vec![0usize; resolved.len()];
        let mut indicator = Vec::with_capacity(n_complete);
        for row in 0..n_complete {
            let hit = resolved.iter().position(|levels| in_arm(levels.as_slice(), row));
            if let Some(k) = hit {
                counts[k] += 1;
            }
            indicator.push(match (resolved.len(), hit) {
                (2, Some(k)) => Some(k == 1),
                (2, None) => None,
                (_, hit) => Some(hit.is_some()),
            });
        }
        let in_any: usize = counts.iter().sum();
        let subject = subject_label(&names, nt, input);
        let arms = input
            .arms
            .iter()
            .zip(&counts)
            .map(|(arm, &rows)| ArmCount { label: arm_label(arm, &ids, &names), rows })
            .collect();
        Ok(Self {
            input,
            subject,
            names,
            columns,
            rows_total,
            rows_complete: n_complete,
            missingness,
            arms,
            rows_other: n_complete - in_any,
            indicator,
        })
    }

    fn nt(&self) -> usize {
        self.input.treatments.len()
    }

    fn adjustment_columns(&self) -> &[Vec<f64>] {
        &self.columns[self.nt() + 1..]
    }

    fn adjustment_names(&self) -> &[String] {
        &self.names[self.nt() + 1..]
    }

    fn outcome_column(&self) -> &[f64] {
        &self.columns[self.nt()]
    }
}

/// Treatment levels are coded constants (`0`, `1`, a category code), compared exactly.
#[allow(clippy::float_cmp, reason = "coded treatment levels are compared exactly")]
fn same_level(value: f64, level: f64) -> bool {
    value == level
}

fn subject_label(names: &[String], nt: usize, input: &PreflightInput<'_>) -> String {
    let treatments = names[..nt].join(", ");
    let kind = if input.arms.len() == 1 { "cell" } else { "effect" };
    format!("{kind}({treatments} -> {})", names[nt])
}

fn arm_label(arm: &ArmSpec, ids: &[VariableId], names: &[String]) -> String {
    let levels: Vec<String> = arm
        .levels
        .iter()
        .map(|&(variable, level)| {
            let name = ids.iter().position(|id| *id == variable).map_or("?", |i| names[i].as_str());
            format!("{name}={level}")
        })
        .collect();
    format!("{}: {}", arm.role, levels.join(", "))
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn arm_refusal(subject: &str, arms: &[ArmCount], empty: &[String]) -> CausalError {
    let fields = RefusalFields {
        stage: Some(STAGE_PREFLIGHT.to_string()),
        subject: Some(subject.to_string()),
        reason: Some("an arm or cell has no complete-case rows".to_string()),
        arm_ess: arms.iter().map(|a| (a.label.clone(), ExactF64(a.rows as f64))).collect(),
        implicated_columns: empty.to_vec(),
        remedy: Some(
            "supply rows at every contrast level, or choose levels the data populate".to_string(),
        ),
        ..RefusalFields::default()
    };
    EstimationError::refused_with_fields(
        reason_code!("arm_not_populated"),
        format!(
            "{subject}: an arm or cell has no complete-case rows ({})",
            arms.iter()
                .map(|a| format!("{} has {}", a.label, a.rows))
                .collect::<Vec<_>>()
                .join("; ")
        ),
        fields,
    )
    .into()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Result of scanning columns in priority order with twice-applied Gram-Schmidt.
struct RankScan {
    /// Orthonormal basis of the kept columns.
    basis: Vec<Vec<f64>>,
    /// Per scanned column.
    columns: Vec<ScannedColumn>,
}

struct ScannedColumn {
    kept: bool,
    /// Residual norm over own norm.
    ratio: f64,
    /// `(scanned column index, coefficient in original units)` of the exact relation, for a
    /// dependent column.
    explained_by: Vec<(usize, f64)>,
}

/// Scan `cols` in order: a column is kept when its residual after projecting out the earlier
/// kept columns exceeds `tol` times its own norm. Dependent columns record the relation that
/// explains them, obtained by back-substitution in the triangular factor of the kept columns.
fn scan_rank(cols: &[&[f64]], tol: f64, ctx: &ExecutionContext) -> Result<RankScan, CausalError> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    // Upper-triangular factor, one column per kept column: `r_kept[k][l] = <q_l, a_k>`.
    let mut r_kept: Vec<Vec<f64>> = Vec::new();
    let mut kept_index: Vec<usize> = Vec::new();
    let mut norms = Vec::with_capacity(cols.len());
    let mut scanned = Vec::with_capacity(cols.len());
    for (j, col) in cols.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_PREFLIGHT });
        }
        let norm = l2(col);
        norms.push(norm);
        if !(norm.is_finite() && norm > 0.0) {
            scanned.push(ScannedColumn { kept: false, ratio: 0.0, explained_by: Vec::new() });
            continue;
        }
        let mut w: Vec<f64> = col.iter().map(|v| v / norm).collect();
        let mut coef = vec![0.0; basis.len()];
        for _ in 0..2 {
            for (k, q) in basis.iter().enumerate() {
                let c = dot(q, &w);
                for (wi, qi) in w.iter_mut().zip(q) {
                    *wi -= c * qi;
                }
                coef[k] += c;
            }
        }
        let ratio = l2(&w);
        if ratio > tol {
            basis.push(w.iter().map(|v| v / ratio).collect());
            coef.push(ratio);
            r_kept.push(coef);
            kept_index.push(j);
            scanned.push(ScannedColumn { kept: true, ratio, explained_by: Vec::new() });
        } else {
            // Solve R c = coef for the coefficients on the kept (unit-norm) columns.
            let m = kept_index.len();
            let mut c = vec![0.0; m];
            for k in (0..m).rev() {
                let tail: f64 = ((k + 1)..m).map(|i| r_kept[i][k] * c[i]).sum();
                c[k] = (coef[k] - tail) / r_kept[k][k];
            }
            let explained_by = c
                .iter()
                .zip(&kept_index)
                .filter_map(|(&ck, &i)| {
                    let original = ck * norm / norms[i];
                    (original.abs() > COEFFICIENT_FLOOR).then_some((i, original))
                })
                .collect();
            scanned.push(ScannedColumn { kept: false, ratio, explained_by });
        }
    }
    Ok(RankScan { basis, columns: scanned })
}

/// `1 - R²` of `values` regressed on the scanned basis (which spans the intercept first),
/// as the square root: the unexplained share of the centered norm. `None` for a constant.
fn unexplained_ratio(basis: &[Vec<f64>], values: &[f64]) -> Option<f64> {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let centered: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>().sqrt();
    if !(centered.is_finite() && centered > 0.0) {
        return None;
    }
    let mut w = values.to_vec();
    for _ in 0..2 {
        for q in basis {
            let c = dot(q, &w);
            for (wi, qi) in w.iter_mut().zip(q) {
                *wi -= c * qi;
            }
        }
    }
    Some(l2(&w) / centered)
}

/// Squared Pearson correlation; `None` when either side is constant.
fn r_squared(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
        sab += (x - ma) * (y - mb);
    }
    if saa > 0.0 && sbb > 0.0 { Some(sab * sab / (saa * sbb)) } else { None }
}

/// Normalized bit pattern of a value (`-0.0` equals `0.0`).
fn bits(value: f64) -> u64 {
    (value + 0.0).to_bits()
}

#[allow(clippy::needless_range_loop, reason = "index pairs walk several parallel arrays")]
fn duplicate_groups(columns: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let hashes: Vec<u64> = columns
        .iter()
        .map(|col| {
            let mut hasher = DefaultHasher::new();
            for &v in col {
                bits(v).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();
    let mut seen = vec![false; columns.len()];
    let mut groups = Vec::new();
    for j in 0..columns.len() {
        if seen[j] {
            continue;
        }
        let mut group = vec![j];
        for k in (j + 1)..columns.len() {
            if !seen[k]
                && hashes[k] == hashes[j]
                && columns[k].iter().zip(&columns[j]).all(|(a, b)| bits(*a) == bits(*b))
            {
                seen[k] = true;
                group.push(k);
            }
        }
        if group.len() > 1 {
            groups.push(group);
        }
    }
    groups
}

fn tolerance(rows: usize, design_columns: usize) -> f64 {
    rows.max(design_columns) as f64 * f64::EPSILON
}

/// Names of the design columns for a scan: the intercept, then `adjustment` in scan order.
fn design_names(adjustment: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(adjustment.len() + 1);
    names.push("(intercept)".to_string());
    names.extend(adjustment.iter().cloned());
    names
}

fn dependent_columns(scan: &RankScan, names: &[String], skip: usize) -> Vec<DependentColumn> {
    scan.columns
        .iter()
        .enumerate()
        .skip(skip)
        .filter(|(_, c)| !c.kept)
        .map(|(j, c)| DependentColumn {
            column: names[j].clone(),
            residual_ratio: c.ratio,
            explained_by: c
                .explained_by
                .iter()
                .map(|&(i, coefficient)| ColumnWeight { column: names[i].clone(), coefficient })
                .collect(),
        })
        .collect()
}

/// Run every fit-free check on the declared design.
///
/// # Errors
///
/// An unknown or non-numeric column, a table over [`MAX_PREFLIGHT_CELLS`], or cancellation.
pub fn preflight_design(
    input: &PreflightInput<'_>,
    ctx: &ExecutionContext,
) -> Result<PreflightReport, CausalError> {
    let frame = Frame::new(input)?;
    let mut findings = Vec::new();
    let nt = frame.nt();

    for arm in frame.arms.iter().filter(|a| a.rows == 0) {
        findings.push(PreflightFinding {
            code: "arm_not_populated",
            severity: FindingSeverity::Blocking,
            columns: Vec::new(),
            measure: Some(0.0),
            threshold: None,
            detail: format!("{} has no complete-case rows", arm.label),
        });
    }

    let duplicates: Vec<Vec<usize>> = duplicate_groups(&frame.columns);
    for group in &duplicates {
        let columns: Vec<String> = group.iter().map(|&j| frame.names[j].clone()).collect();
        let touches_query = group.iter().any(|&j| j <= nt);
        let touches_adjustment = group.iter().any(|&j| j > nt);
        let (code, detail) = if touches_query && touches_adjustment {
            (
                "adjustment_duplicates_query_variable",
                "an adjustment column is an exact copy of a treatment or the outcome; review \
                 whether it is a post-treatment or label-leaking column (the adjustment set is \
                 not changed)",
            )
        } else if touches_adjustment {
            (
                "duplicate_adjustment_columns",
                "adjustment columns are exact copies of one another; the design is rank \
                 deficient by exactly this duplication",
            )
        } else {
            ("duplicate_query_columns", "the treatment and outcome columns are exact copies")
        };
        findings.push(PreflightFinding {
            code,
            severity: FindingSeverity::Review,
            columns,
            measure: None,
            threshold: None,
            detail: detail.to_string(),
        });
    }

    let mut rank = None;
    let adjustment_names = frame.adjustment_names();
    if frame.rows_complete > 0 {
        let ones = vec![1.0; frame.rows_complete];
        let mut design: Vec<&[f64]> = Vec::with_capacity(adjustment_names.len() + 1);
        design.push(&ones);
        design.extend(frame.adjustment_columns().iter().map(Vec::as_slice));
        let tol = tolerance(frame.rows_complete, design.len());
        let scan = scan_rank(&design, tol, ctx)?;
        let names = design_names(adjustment_names);
        let dependent = dependent_columns(&scan, &names, 0);
        let numerical_rank = scan.basis.len();
        if !dependent.is_empty() {
            let rank_text = format!(
                "the {} design columns have numerical rank {numerical_rank}; dependent: {}",
                design.len(),
                dependent.iter().map(|d| d.column.as_str()).collect::<Vec<_>>().join(", ")
            );
            findings.push(PreflightFinding {
                code: "design_rank_deficient",
                severity: FindingSeverity::Blocking,
                columns: dependent.iter().map(|d| d.column.clone()).collect(),
                measure: Some(numerical_rank as f64),
                threshold: Some(design.len() as f64),
                detail: format!(
                    "{rank_text}; regression-based nuisance fits refuse a rank-deficient design"
                ),
            });
        }
        for (j, scanned) in scan.columns.iter().enumerate().skip(1) {
            if scanned.kept && scanned.ratio < NEAR_COLLINEAR_RATIO {
                findings.push(PreflightFinding {
                    code: "near_collinear_column",
                    severity: FindingSeverity::Review,
                    columns: vec![names[j].clone()],
                    measure: Some(scanned.ratio),
                    threshold: Some(NEAR_COLLINEAR_RATIO),
                    detail: "the column is not an exact linear function of higher-priority \
                             columns but leaves almost no residual after projecting them out \
                             (uncentered); fits are ill conditioned, nothing is dropped"
                        .to_string(),
                });
            }
        }
        for (what, code, values) in query_targets(&frame) {
            if let Some(ratio) = unexplained_ratio(&scan.basis, values) {
                let r2 = 1.0 - ratio * ratio;
                if r2 >= NEAR_DETERMINISTIC_R2 {
                    findings.push(review_flag(code, what, "the adjustment columns", r2));
                }
            }
        }
        rank = Some(RankReport {
            design_columns: design.len(),
            numerical_rank,
            tolerance: tol,
            dependent,
        });
    }

    for (j, col) in frame.adjustment_columns().iter().enumerate() {
        for (what, code, values) in query_columns(&frame) {
            if let Some(r2) = r_squared(col, values) {
                if r2 >= NEAR_DETERMINISTIC_R2 {
                    let mut flag = review_flag(code, what, &adjustment_names[j], r2);
                    flag.columns = vec![adjustment_names[j].clone(), what.to_string()];
                    findings.push(flag);
                }
            }
        }
        if let Some(kind) = separates_arms(col, &frame.indicator) {
            findings.push(PreflightFinding {
                code: "column_separates_arms",
                severity: FindingSeverity::Review,
                columns: vec![adjustment_names[j].clone()],
                measure: None,
                threshold: None,
                detail: format!(
                    "a single adjustment column {kind} separates the arms: no overlap on that \
                     column, so a propensity model will saturate (a positivity warning for \
                     human review, not an identification verdict)"
                ),
            });
        }
    }

    Ok(PreflightReport {
        subject: frame.subject.clone(),
        treatment: frame.names[..nt].to_vec(),
        outcome: frame.names[nt].clone(),
        adjustment_set: adjustment_names.to_vec(),
        rows_total: frame.rows_total,
        rows_complete: frame.rows_complete,
        missingness: frame.missingness.clone(),
        arms: frame.arms.clone(),
        rows_other_levels: frame.rows_other,
        duplicates: duplicates
            .iter()
            .map(|g| DuplicateGroup {
                columns: g.iter().map(|&j| frame.names[j].clone()).collect(),
            })
            .collect(),
        rank,
        findings,
    })
}

/// The treatment and outcome vectors a design-span near-determinism check looks at.
fn query_targets<'f>(frame: &'f Frame<'_>) -> Vec<(&'f str, &'static str, &'f [f64])> {
    let mut out: Vec<(&str, &'static str, &[f64])> = Vec::new();
    for (j, name) in frame.names[..frame.nt()].iter().enumerate() {
        out.push((
            name.as_str(),
            "treatment_near_determined_by_adjustment",
            frame.columns[j].as_slice(),
        ));
    }
    out.push((
        frame.names[frame.nt()].as_str(),
        "outcome_near_determined_by_adjustment",
        frame.outcome_column(),
    ));
    out
}

/// The treatment and outcome vectors a single-column near-determinism check looks at.
fn query_columns<'f>(frame: &'f Frame<'_>) -> Vec<(&'f str, &'static str, &'f [f64])> {
    let mut out: Vec<(&str, &'static str, &[f64])> = Vec::new();
    for (j, name) in frame.names[..frame.nt()].iter().enumerate() {
        out.push((
            name.as_str(),
            "adjustment_column_tracks_treatment",
            frame.columns[j].as_slice(),
        ));
    }
    out.push((
        frame.names[frame.nt()].as_str(),
        "adjustment_column_tracks_outcome",
        frame.outcome_column(),
    ));
    out
}

fn review_flag(code: &'static str, target: &str, source: &str, r2: f64) -> PreflightFinding {
    PreflightFinding {
        code,
        severity: FindingSeverity::Review,
        columns: vec![target.to_string()],
        measure: Some(r2),
        threshold: Some(NEAR_DETERMINISTIC_R2),
        detail: format!(
            "{target} is near-deterministically predicted by {source} (linear R^2 {r2:.6}); a \
             predictive relation on this table, not a causal statement. Review for a \
             post-treatment variable or label leakage; the adjustment set is not changed"
        ),
    }
}

/// `"completely"` or `"quasi-completely"` when `col` splits the two groups of `indicator`
/// by value alone, `None` when the groups overlap or either is empty.
fn separates_arms(col: &[f64], indicator: &[Option<bool>]) -> Option<&'static str> {
    let (mut lo_t, mut hi_t) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut lo_c, mut hi_c) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut n_t, mut n_c) = (0usize, 0usize);
    for (&v, ind) in col.iter().zip(indicator) {
        match ind {
            Some(true) => {
                n_t += 1;
                lo_t = lo_t.min(v);
                hi_t = hi_t.max(v);
            }
            Some(false) => {
                n_c += 1;
                lo_c = lo_c.min(v);
                hi_c = hi_c.max(v);
            }
            None => {}
        }
    }
    if n_t == 0 || n_c == 0 {
        return None;
    }
    if hi_c < lo_t || hi_t < lo_c {
        Some("completely")
    } else if hi_c <= lo_t || hi_t <= lo_c {
        Some("quasi-completely")
    } else {
        None
    }
}

/// One fitted-score quantile.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScoreQuantile {
    /// Probability level.
    pub probability: f64,
    /// Nearest-rank quantile of the fitted propensities.
    pub value: f64,
}

/// Effective sample size of the inverse-propensity weights in one arm.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ArmWeightEss {
    /// Arm label.
    pub label: String,
    /// Complete-case rows in the arm.
    pub rows: usize,
    /// Kish effective sample size `(sum w)^2 / sum w^2` of the arm's inverse-propensity
    /// weights.
    pub ess: f64,
}

/// A propensity fit that produced scores (possibly flagged as saturated or unconverged).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FittedPropensity {
    /// Whether the IRLS loop converged.
    pub converged: bool,
    /// Whether (quasi-)complete separation was detected.
    pub separated: bool,
    /// Whether some fitted probability lies within `1e-8` of 0 or 1.
    pub boundary_saturated: bool,
    /// IRLS iterations used.
    pub iterations: u32,
    /// Smallest fitted propensity.
    pub min: f64,
    /// Largest fitted propensity.
    pub max: f64,
    /// Nearest-rank quantiles.
    pub quantiles: Vec<ScoreQuantile>,
    /// Per-arm weight ESS.
    pub arm_ess: Vec<ArmWeightEss>,
}

/// Outcome of the diagnostic propensity fit.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PropensityOutcome {
    /// Scores exist.
    Fitted(FittedPropensity),
    /// The fit failed before any score existed; no fitted quantity is reported.
    Absent {
        /// Why no fit exists.
        reason: String,
        /// Numerical rank when the failure was a rank deficiency.
        numerical_rank: Option<usize>,
        /// Columns of the fitted design.
        design_columns: usize,
    },
}

/// Diagnostics that need a nuisance fit. Never preflight: the fit is part of the evidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NuisanceFitDiagnostics {
    /// Plan label.
    pub subject: String,
    /// The propensity fit, or why there is none.
    pub propensity: PropensityOutcome,
}

impl NuisanceFitDiagnostics {
    /// Structured refusal fields from this fit for a caller that refuses on it, with every
    /// quantity the fit did not produce left absent.
    #[must_use]
    pub fn refusal_fields(&self, stage: &str, reason: &str) -> RefusalFields {
        let mut fields = RefusalFields {
            stage: Some(stage.to_string()),
            subject: Some(self.subject.clone()),
            reason: Some(reason.to_string()),
            ..RefusalFields::default()
        };
        match &self.propensity {
            PropensityOutcome::Fitted(fit) => {
                fields.arm_ess =
                    fit.arm_ess.iter().map(|a| (a.label.clone(), ExactF64(a.ess))).collect();
                fields.propensity_min = Some(ExactF64(fit.min));
                fields.propensity_max = Some(ExactF64(fit.max));
                fields.propensity_quantiles = fit
                    .quantiles
                    .iter()
                    .map(|q| (ExactF64(q.probability), ExactF64(q.value)))
                    .collect();
            }
            PropensityOutcome::Absent { numerical_rank, design_columns, .. } => {
                fields.numerical_rank = numerical_rank.map(as_u64);
                fields.design_columns = Some(as_u64(*design_columns));
            }
        }
        fields
    }
}

fn kish_ess(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    if sum_sq > 0.0 { sum * sum / sum_sq } else { 0.0 }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a probability times a row count is a small non-negative rank"
)]
fn nearest_rank(sorted: &[f64], probability: f64) -> f64 {
    let n = sorted.len();
    let rank = ((probability * n as f64).ceil() as usize).clamp(1, n);
    sorted[rank - 1]
}

/// Fit a diagnostic propensity model (arm membership on `[1 | Z]`, no separation ridge) and
/// report the score range and per-arm weight ESS.
///
/// A failure before any score exists is returned as [`PropensityOutcome::Absent`] with the
/// reason; it is not an `Err`, because "the fit fails here" is the diagnostic.
///
/// # Errors
///
/// An unknown or non-numeric column, a table over [`MAX_PREFLIGHT_CELLS`], or cancellation.
pub fn fit_diagnostics_design(
    input: &PreflightInput<'_>,
    ctx: &ExecutionContext,
) -> Result<NuisanceFitDiagnostics, CausalError> {
    let frame = Frame::new(input)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: STAGE_PREFLIGHT });
    }
    let ncols = frame.adjustment_columns().len() + 1;
    let absent = |reason: String, numerical_rank: Option<usize>| NuisanceFitDiagnostics {
        subject: frame.subject.clone(),
        propensity: PropensityOutcome::Absent { reason, numerical_rank, design_columns: ncols },
    };
    let rows: Vec<usize> = frame
        .indicator
        .iter()
        .enumerate()
        .filter(|(_, ind)| ind.is_some())
        .map(|(i, _)| i)
        .collect();
    let treated: Vec<f64> = rows
        .iter()
        .map(|&i| if matches!(frame.indicator[i], Some(true)) { 1.0 } else { 0.0 })
        .collect();
    let n_treated = treated.iter().filter(|t| **t > 0.5).count();
    if n_treated == 0 || n_treated == treated.len() {
        return Ok(absent(
            "an arm has no complete-case rows, so no propensity model can be fit".to_string(),
            None,
        ));
    }
    let nr = rows.len();
    let mut x = vec![1.0; nr * ncols];
    for (j, col) in frame.adjustment_columns().iter().enumerate() {
        for (r, &i) in rows.iter().enumerate() {
            x[(j + 1) * nr + r] = col[i];
        }
    }
    let options = GlmOptions::default().without_separation_ridge();
    let mut workspace = PropensityWorkspace::default();
    let fit = match fit_propensity_diagnostic(
        &x,
        nr,
        ncols,
        &treated,
        &FaerBackend,
        &mut workspace,
        &options,
    ) {
        Ok(fit) => fit,
        Err(StatsError::RankDeficient { rank, .. }) => {
            return Ok(absent(
                format!(
                    "the propensity design is rank deficient (numerical rank {rank} of {ncols} \
                     columns); the fit fails before any score exists"
                ),
                Some(rank),
            ));
        }
        Err(error) => return Ok(absent(error.to_string(), None)),
    };
    let weights: Vec<f64> = fit
        .scores
        .iter()
        .zip(&treated)
        .map(|(&e, &t)| if t > 0.5 { 1.0 / e } else { 1.0 / (1.0 - e) })
        .collect();
    let arm_ess = group_labels(&frame)
        .into_iter()
        .map(|(label, is_treated)| {
            let group: Vec<f64> = weights
                .iter()
                .zip(&treated)
                .filter(|(_, t)| (**t > 0.5) == is_treated)
                .map(|(w, _)| *w)
                .collect();
            ArmWeightEss { label, rows: group.len(), ess: kish_ess(&group) }
        })
        .collect();
    let mut sorted = fit.scores.clone();
    sorted.sort_by(f64::total_cmp);
    let quantiles = PROPENSITY_QUANTILES
        .iter()
        .map(|&p| ScoreQuantile { probability: p, value: nearest_rank(&sorted, p) })
        .collect();
    Ok(NuisanceFitDiagnostics {
        subject: frame.subject.clone(),
        propensity: PropensityOutcome::Fitted(FittedPropensity {
            converged: fit.glm.converged,
            separated: fit.glm.separated,
            boundary_saturated: fit.glm.boundary_saturated,
            iterations: fit.glm.iterations,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            quantiles,
            arm_ess,
        }),
    })
}

/// `(label, is_treated)` of the two groups the propensity model separates.
fn group_labels(frame: &Frame<'_>) -> Vec<(String, bool)> {
    if frame.arms.len() == 2 {
        vec![(frame.arms[0].label.clone(), false), (frame.arms[1].label.clone(), true)]
    } else {
        vec![(frame.arms[0].label.clone(), true), ("rest of the table".to_string(), false)]
    }
}

/// How the columns of the design are ranked for an opt-in drop.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnPriority {
    /// The adjustment set's own order, first column highest priority.
    AdjustmentOrder,
    /// An explicit order over **every** adjustment column exactly once, highest priority
    /// first. A list that does not cover the adjustment set is refused.
    Declared(Vec<VariableId>),
}

/// Opt-in policy for handling a numerically rank-deficient design.
#[derive(Clone, Debug, PartialEq)]
pub struct RankDropPolicy {
    /// Deterministic column priority, declared before anything is dropped.
    pub priority: ColumnPriority,
}

/// One column a rank drop would remove, and why.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DroppedColumn {
    /// Column name.
    pub column: String,
    /// Residual norm over own norm after projecting out the kept higher-priority columns.
    pub residual_ratio: f64,
    /// The exact linear relation that makes the column redundant.
    pub explained_by: Vec<ColumnWeight>,
}

/// Record of an opt-in rank drop: what goes, why, and the resulting design identity.
///
/// A plan only: it removes columns from the numerical design, not from the study. A dropped
/// column is a numerically exact linear function of the kept ones on this table, so it
/// carries no information they do not, and the treatment, outcome and effect modifiers are
/// untouched by construction (the drop is refused when they are involved).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankDropPlan {
    /// Plan label.
    pub subject: String,
    /// The priority order used, highest first.
    pub priority: Vec<String>,
    /// The adjustment set before the drop.
    pub original_adjustment: Vec<String>,
    /// Columns that would be dropped, in priority-scan order.
    pub dropped: Vec<DroppedColumn>,
    /// The adjustment set after the drop, in its original order.
    pub kept_adjustment: Vec<String>,
    /// Numerical rank of the design (unchanged by the drop).
    pub numerical_rank: usize,
    /// Canonical text of the resulting design (adjustment kept, columns dropped, priority).
    pub design_identity: String,
}

fn rank_drop_refusal(
    subject: &str,
    reason: &str,
    implicated: Vec<String>,
    remedy: &str,
) -> CausalError {
    let fields = RefusalFields {
        stage: Some(STAGE_RANK_DROP.to_string()),
        subject: Some(subject.to_string()),
        reason: Some(reason.to_string()),
        implicated_columns: implicated,
        remedy: Some(remedy.to_string()),
        ..RefusalFields::default()
    };
    EstimationError::refused_with_fields(
        reason_code!("rank_drop_not_licensed"),
        format!("{subject}: {reason}; nothing was dropped"),
        fields,
    )
    .into()
}

/// Plan an opt-in rank-deficiency drop under a declared column priority.
///
/// # Errors
///
/// `rank_drop_not_licensed` when the priority does not cover the adjustment set, when a
/// dependent column is a treatment, outcome or effect modifier, or when an adjustment column
/// is an exact copy of a treatment or the outcome; `arm_not_populated` when an arm is empty;
/// or the preflight errors (unknown column, size limit, cancellation).
pub fn plan_rank_drop(
    input: &PreflightInput<'_>,
    policy: &RankDropPolicy,
    ctx: &ExecutionContext,
) -> Result<RankDropPlan, CausalError> {
    let frame = Frame::new(input)?;
    let nt = frame.nt();
    let p = input.adjustment.len();
    let order: Vec<usize> = match &policy.priority {
        ColumnPriority::AdjustmentOrder => (0..p).collect(),
        ColumnPriority::Declared(ids) => {
            let mut used = vec![false; p];
            let mut order = Vec::with_capacity(p);
            for id in ids {
                let slot = (0..p).find(|&k| !used[k] && input.adjustment[k] == *id);
                match slot {
                    Some(k) => {
                        used[k] = true;
                        order.push(k);
                    }
                    None => {
                        return Err(rank_drop_refusal(
                            &frame.subject,
                            "the declared column priority names a column that is not in the \
                             adjustment set (or names it twice)",
                            Vec::new(),
                            "declare every adjustment column exactly once, highest priority first",
                        ));
                    }
                }
            }
            if order.len() != p {
                let missing: Vec<String> = (0..p)
                    .filter(|&k| !used[k])
                    .map(|k| frame.adjustment_names()[k].clone())
                    .collect();
                return Err(rank_drop_refusal(
                    &frame.subject,
                    "the declared column priority does not cover the adjustment set",
                    missing,
                    "declare every adjustment column exactly once, highest priority first",
                ));
            }
            order
        }
    };
    if frame.arms.iter().any(|a| a.rows == 0) {
        return Err(arm_refusal(&frame.subject, &frame.arms, &[]));
    }

    let protected: Vec<VariableId> = input
        .treatments
        .iter()
        .copied()
        .chain(std::iter::once(input.outcome))
        .chain(input.protected.iter().copied())
        .collect();
    for group in duplicate_groups(&frame.columns) {
        if group.iter().any(|&j| j <= nt) && group.iter().any(|&j| j > nt) {
            return Err(rank_drop_refusal(
                &frame.subject,
                "an adjustment column is an exact copy of a treatment or the outcome, which a \
                 rank drop would not resolve (a treatment-definition or leakage question)",
                group.iter().map(|&j| frame.names[j].clone()).collect(),
                "review the aliasing column and remove it from the adjustment set deliberately",
            ));
        }
    }

    let ones = vec![1.0; frame.rows_complete];
    let columns = frame.adjustment_columns();
    let mut design: Vec<&[f64]> = Vec::with_capacity(p + 1);
    design.push(&ones);
    design.extend(order.iter().map(|&k| columns[k].as_slice()));
    let scan_names = design_names(
        &order.iter().map(|&k| frame.adjustment_names()[k].clone()).collect::<Vec<_>>(),
    );
    let scan = scan_rank(&design, tolerance(frame.rows_complete, design.len()), ctx)?;
    let dependent = dependent_columns(&scan, &scan_names, 1);
    let mut dropped_adjustment = vec![false; p];
    let mut dropped = Vec::with_capacity(dependent.len());
    for (scan_pos, column) in scan.columns.iter().enumerate().skip(1) {
        if column.kept {
            continue;
        }
        let k = order[scan_pos - 1];
        if protected.contains(&input.adjustment[k]) {
            return Err(rank_drop_refusal(
                &frame.subject,
                "the dependent column is a treatment, outcome or effect modifier the query \
                 needs",
                vec![frame.adjustment_names()[k].clone()],
                "give the query variable the highest priority or remove the aliasing column",
            ));
        }
        dropped_adjustment[k] = true;
    }
    for column in dependent {
        dropped.push(DroppedColumn {
            column: column.column,
            residual_ratio: column.residual_ratio,
            explained_by: column.explained_by,
        });
    }
    let original: Vec<String> = frame.adjustment_names().to_vec();
    let kept: Vec<String> = original
        .iter()
        .zip(&dropped_adjustment)
        .filter(|(_, drop)| !**drop)
        .map(|(name, _)| name.clone())
        .collect();
    let priority: Vec<String> = order.iter().map(|&k| original[k].clone()).collect();
    let design_identity = format!(
        "adjustment=[{}];dropped=[{}];priority=[{}]",
        kept.join(","),
        dropped.iter().map(|d| d.column.as_str()).collect::<Vec<_>>().join(","),
        priority.join(",")
    );
    Ok(RankDropPlan {
        subject: frame.subject.clone(),
        priority,
        original_adjustment: original,
        dropped,
        kept_adjustment: kept,
        numerical_rank: scan.basis.len(),
        design_identity,
    })
}

/// A dropped column's residual after projecting onto the retained span must stay within
/// this multiple of the scan tolerance (rounding slack for the second projection).
const SPAN_SLACK: f64 = 10.0;

/// Numerical evidence that a rank drop kept the column space of the `[1 | Z]` design.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpanCheck {
    /// Numerical rank of the original `[1 | Z]` design.
    pub original_rank: usize,
    /// Numerical rank of the retained `[1 | Z_kept]` design (equal to its column count).
    pub retained_rank: usize,
    /// Largest residual-over-norm of a dropped column after projecting onto the retained
    /// span (0 when nothing was dropped).
    pub max_dropped_residual_ratio: f64,
    /// Scan tolerance the ratios were compared with, after the rounding slack.
    pub tolerance: f64,
}

/// The adjustment columns a plan keeps, in the input's order.
pub(super) fn kept_adjustment_ids(
    input: &PreflightInput<'_>,
    plan: &RankDropPlan,
) -> Result<Vec<VariableId>, CausalError> {
    let schema = input.data.schema();
    let mut kept = Vec::with_capacity(plan.kept_adjustment.len());
    for &id in &input.adjustment {
        let meta = schema.get(id)?;
        if plan.kept_adjustment.iter().any(|name| name.as_str() == &*meta.name) {
            kept.push(id);
        }
    }
    Ok(kept)
}

/// Residual norm over own norm of `values` after projecting out an orthonormal `basis`
/// (twice, for orthogonality). A zero column lies in every span.
fn residual_over_norm(basis: &[Vec<f64>], values: &[f64]) -> f64 {
    let norm = l2(values);
    if !(norm.is_finite() && norm > 0.0) {
        return 0.0;
    }
    let mut w = values.to_vec();
    for _ in 0..2 {
        for q in basis {
            let c = dot(q, &w);
            for (wi, qi) in w.iter_mut().zip(q) {
                *wi -= c * qi;
            }
        }
    }
    l2(&w) / norm
}

/// Verify, independently of the plan's scan, that dropping `plan.dropped` left the column
/// space unchanged: the retained design has full numerical rank equal to the original's, and
/// every dropped column lies in the retained span within the scan tolerance.
///
/// # Errors
///
/// `rank_drop_not_licensed` when the retained design is still rank deficient, its rank
/// differs from the original, or a dropped column is not in the retained span; the
/// preflight errors (unknown column, size limit, cancellation).
pub(super) fn verify_span_preserved(
    input: &PreflightInput<'_>,
    plan: &RankDropPlan,
    kept: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<SpanCheck, CausalError> {
    let frame = Frame::new(input)?;
    let columns = frame.adjustment_columns();
    let ones = vec![1.0; frame.rows_complete];
    let keep_mask: Vec<bool> = input.adjustment.iter().map(|id| kept.contains(id)).collect();
    let mut design: Vec<&[f64]> = vec![&ones];
    design.extend(
        columns.iter().zip(&keep_mask).filter(|(_, keep)| **keep).map(|(c, _)| c.as_slice()),
    );
    let tol = tolerance(frame.rows_complete, input.adjustment.len() + 1);
    let scan = scan_rank(&design, tol, ctx)?;
    let retained_rank = scan.basis.len();
    if retained_rank != design.len() || retained_rank != plan.numerical_rank {
        return Err(rank_drop_refusal(
            &frame.subject,
            "rank_drop_estimate.span_not_preserved: the retained design does not span the \
             original column space (its numerical rank differs from the original's or it is \
             still rank deficient)",
            Vec::new(),
            "the drop would change the fitted projection; keep the original adjustment set",
        ));
    }
    let mut worst = 0.0_f64;
    let mut worst_column = String::new();
    for ((column, keep), name) in columns.iter().zip(&keep_mask).zip(frame.adjustment_names()) {
        if *keep {
            continue;
        }
        let ratio = residual_over_norm(&scan.basis, column);
        if ratio > worst {
            worst = ratio;
            worst_column.clone_from(name);
        }
    }
    if worst > SPAN_SLACK * tol {
        return Err(rank_drop_refusal(
            &frame.subject,
            "rank_drop_estimate.span_not_preserved: a dropped column is not an exact linear \
             combination of the retained columns",
            vec![worst_column],
            "the drop would change the fitted projection; keep the original adjustment set",
        ));
    }
    Ok(SpanCheck {
        original_rank: plan.numerical_rank,
        retained_rank,
        max_dropped_residual_ratio: worst,
        tolerance: SPAN_SLACK * tol,
    })
}

fn outside_cell() -> CausalError {
    crate::unsupported_reason!(
        "route_not_supported",
        "preflight diagnostics cover prepared static AverageEffect plans and discrete joint \
         InterventionResponse cells on tabular data with a certified adjustment set"
    )
}

/// The certified adjustment set a prepared plan retains, when it retains one.
pub(super) fn adjustment_of(study: &Study) -> Option<Vec<VariableId>> {
    if let Some(cache) = &study.identification_cache {
        return Some(cache.estimand.adjustment_set.to_vec());
    }
    study.shared_batch_design.as_ref()?.covariate.as_ref().map(|c| c.adjustment_set.to_vec())
}

/// The preflight input a prepared plan implies.
fn input_of(study: &Study) -> Result<PreflightInput<'_>, CausalError> {
    let DataInput::Tabular(data) = &study.data else {
        return Err(outside_cell());
    };
    let adjustment = adjustment_of(study).ok_or_else(outside_cell)?;
    match &study.query {
        CausalQuery::AverageEffect(q) => {
            let (Intervention::Set { value: control, .. }, Intervention::Set { value: active, .. }) =
                (&q.control, &q.active)
            else {
                return Err(outside_cell());
            };
            let (Some(control), Some(active)) = (control.as_f64(), active.as_f64()) else {
                return Err(outside_cell());
            };
            let mut input = PreflightInput::binary_effect(
                data,
                q.treatment,
                q.outcome,
                &adjustment,
                control,
                active,
            );
            input.protected = q.effect_modifiers.to_vec();
            Ok(input)
        }
        CausalQuery::Response(q) if q.temporal.is_none() => {
            let ResponseFunctional::InterventionResponse { outcome, interventions } = &q.functional
            else {
                return Err(outside_cell());
            };
            let mut levels = Vec::with_capacity(interventions.len());
            for intervention in interventions.iter() {
                let Intervention::Set { variable, value } = intervention else {
                    return Err(outside_cell());
                };
                let Some(level) = value.as_f64() else {
                    return Err(outside_cell());
                };
                levels.push((*variable, level));
            }
            Ok(PreflightInput::joint_cell(data, *outcome, &adjustment, levels))
        }
        _ => Err(outside_cell()),
    }
}

impl PreparedStudy {
    /// Fit-free preflight of this plan's retained table and certified adjustment set.
    ///
    /// Reports complete-case and arm counts, missingness, exact duplicate columns, the
    /// numerical rank of the `[1 | Z]` design with dependent columns named, and bounded
    /// near-determinism flags for human review. Nothing is fit; see
    /// [`Self::diagnose_fit`] for checks that need a nuisance fit. Adjustment sets are never
    /// changed by a finding.
    ///
    /// # Errors
    ///
    /// `route_not_supported` outside static AverageEffect and discrete joint cells on tabular
    /// data; the resource limit; cancellation.
    pub fn diagnose(&self, ctx: &ExecutionContext) -> Result<PreflightReport, CausalError> {
        preflight_design(&input_of(self.study())?, ctx)
    }

    /// Diagnostics that **fit** a propensity model on the retained table. Not preflight: the
    /// fit is part of the evidence, and a failure before any score exists is reported as
    /// absent, never reconstructed.
    ///
    /// # Errors
    ///
    /// As [`Self::diagnose`].
    pub fn diagnose_fit(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<NuisanceFitDiagnostics, CausalError> {
        fit_diagnostics_design(&input_of(self.study())?, ctx)
    }

    /// Opt-in rank-deficiency drop plan under a declared column priority. A record, not an
    /// execution: no estimator is re-run on the reduced design.
    ///
    /// # Errors
    ///
    /// As [`plan_rank_drop`].
    pub fn plan_rank_drop(
        &self,
        policy: &RankDropPolicy,
        ctx: &ExecutionContext,
    ) -> Result<RankDropPlan, CausalError> {
        plan_rank_drop(&input_of(self.study())?, policy, ctx)
    }
}

impl PreparedBatch {
    /// Fit-free preflight of every plan in the batch (see [`PreparedStudy::diagnose`]).
    ///
    /// # Errors
    ///
    /// The first plan's error, as [`PreparedStudy::diagnose`].
    pub fn diagnose(&self, ctx: &ExecutionContext) -> Result<BatchPreflightReport, CausalError> {
        let reports =
            self.plans().iter().map(|plan| plan.diagnose(ctx)).collect::<Result<Vec<_>, _>>()?;
        Ok(BatchPreflightReport {
            reports,
            shares_covariates: self.shared_design().is_some_and(|d| d.covariate.is_some()),
        })
    }

    /// Propensity-fit diagnostics for every plan (see [`PreparedStudy::diagnose_fit`]).
    ///
    /// # Errors
    ///
    /// The first plan's error.
    pub fn diagnose_fit(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<NuisanceFitDiagnostics>, CausalError> {
        self.plans().iter().map(|plan| plan.diagnose_fit(ctx)).collect()
    }

    /// Rank-drop plans for every plan (see [`PreparedStudy::plan_rank_drop`]).
    ///
    /// # Errors
    ///
    /// The first plan's refusal.
    pub fn plan_rank_drop(
        &self,
        policy: &RankDropPolicy,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RankDropPlan>, CausalError> {
        self.plans().iter().map(|plan| plan.plan_rank_drop(policy, ctx)).collect()
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::ExecutionContext;

    use antecedent_data::{TableView, TabularData};

    use super::{
        PreflightInput, RankDropPlan, scan_rank, separates_arms, tolerance, verify_span_preserved,
    };

    fn column(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic, well-spread values from a small linear congruential stream.
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    /// A column that is an exact combination of earlier ones is named with its relation in
    /// original units; independent columns are kept; a constant is explained by the
    /// intercept.
    #[test]
    fn scan_names_the_exact_linear_relation_in_original_units() {
        let n = 200;
        let ones = vec![1.0; n];
        let a = column(n, 1);
        let b = column(n, 2);
        let combo: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 3.0 * x - 2.0 * y + 5.0).collect();
        let constant = vec![7.0; n];
        let cols: Vec<&[f64]> = vec![&ones, &a, &b, &combo, &constant];
        let ctx = ExecutionContext::for_tests(1);
        let scan = scan_rank(&cols, tolerance(n, cols.len()), &ctx).unwrap();
        assert_eq!(scan.basis.len(), 3);
        assert!(scan.columns[..3].iter().all(|c| c.kept));
        assert!(!scan.columns[3].kept && !scan.columns[4].kept);
        let relation = &scan.columns[3].explained_by;
        let coefficient =
            |index: usize| relation.iter().find(|(i, _)| *i == index).map_or(f64::NAN, |(_, c)| *c);
        assert!((coefficient(0) - 5.0).abs() < 1e-8, "{relation:?}");
        assert!((coefficient(1) - 3.0).abs() < 1e-8, "{relation:?}");
        assert!((coefficient(2) + 2.0).abs() < 1e-8, "{relation:?}");
        assert_eq!(scan.columns[4].explained_by.len(), 1);
        assert!((scan.columns[4].explained_by[0].1 - 7.0).abs() < 1e-8);
    }

    /// A cancelled context stops the scan with a typed cancellation, not a partial result.
    #[test]
    fn scan_observes_cancellation() {
        let n = 50;
        let ones = vec![1.0; n];
        let cols: Vec<&[f64]> = vec![&ones];
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let Err(error) = scan_rank(&cols, tolerance(n, 1), &ctx) else {
            panic!("a cancelled scan must not return a result");
        };
        assert!(matches!(error, crate::CausalError::Cancelled { .. }), "{error}");
    }

    /// Separation is complete when the groups' ranges are disjoint and quasi-complete when
    /// they only touch; overlap or an empty group is not separation.
    #[test]
    fn separation_is_complete_quasi_or_absent() {
        let col = [0.0, 1.0, 2.0, 3.0];
        let apart = [Some(false), Some(false), Some(true), Some(true)];
        assert_eq!(separates_arms(&col, &apart), Some("completely"));
        assert_eq!(separates_arms(&[0.0, 1.0, 1.0, 3.0], &apart), Some("quasi-completely"));
        let mixed = [Some(false), Some(true), Some(false), Some(true)];
        assert_eq!(separates_arms(&col, &mixed), None);
        assert_eq!(separates_arms(&col, &[Some(true); 4]), None);
        assert_eq!(
            separates_arms(&col, &[None, Some(false), None, Some(true)]),
            Some("completely")
        );
    }

    /// The independent span check refuses a plan whose retained columns lose the original
    /// column space: once because the rank differs, once because a dropped column is not in
    /// the retained span although the rank is (falsely) recorded as unchanged.
    #[test]
    fn span_verification_refuses_a_drop_that_loses_an_independent_column() {
        let n = 120;
        let (a, b) = (column(n, 11), column(n, 12));
        let (t, y) = (column(n, 13), column(n, 14));
        let data = TabularData::from_f64_columns(vec![
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("a", a.as_slice()),
            ("b", b.as_slice()),
        ])
        .unwrap();
        let schema = data.schema();
        let (ia, ib) = (schema.id_of("a").unwrap(), schema.id_of("b").unwrap());
        let input = PreflightInput::binary_effect(
            &data,
            schema.id_of("t").unwrap(),
            schema.id_of("y").unwrap(),
            &[ia, ib],
            0.0,
            1.0,
        );
        let plan = |rank: usize| RankDropPlan {
            subject: "t -> y".to_string(),
            priority: vec!["a".to_string(), "b".to_string()],
            original_adjustment: vec!["a".to_string(), "b".to_string()],
            dropped: Vec::new(),
            kept_adjustment: vec!["a".to_string()],
            numerical_rank: rank,
            design_identity: String::new(),
        };
        let ctx = ExecutionContext::for_tests(1);
        for recorded_rank in [3, 2] {
            let error =
                verify_span_preserved(&input, &plan(recorded_rank), &[ia], &ctx).unwrap_err();
            assert_eq!(error.reason_code(), Some("rank_drop_not_licensed"), "{error}");
            assert!(error.to_string().contains("rank_drop_estimate.span_not_preserved"), "{error}");
        }
        // Keeping both columns spans the original space exactly.
        let kept_all = verify_span_preserved(&input, &plan(3), &[ia, ib], &ctx).unwrap();
        assert_eq!((kept_all.original_rank, kept_all.retained_rank), (3, 3));
        assert!(kept_all.max_dropped_residual_ratio <= 0.0);
    }
}

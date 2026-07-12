//! Shared refuter types and data transforms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use causal_estimate::{EffectEstimate, EstimationWorkspace, LinearAdjustmentAte};
use causal_identify::IdentifiedEstimand;

use crate::error::ValidationError;

/// Comparison of original vs refuted estimates.
#[derive(Clone, Debug)]
pub struct RefutationReport {
    /// Refuter id.
    pub refuter: Arc<str>,
    /// Original ATE.
    pub original_ate: f64,
    /// Refuted / transformed ATE (mean across replicates when applicable).
    pub refuted_ate: f64,
    /// Absolute difference `|refuted - original|` (or `|refuted|` for placebo).
    pub comparison: f64,
    /// Whether the check is informative for the estimator used.
    pub informative: bool,
    /// Whether the check passed the configured threshold.
    pub passed: bool,
    /// Failure condition description when `passed` is false.
    pub failure_condition: Option<Arc<str>>,
    /// Number of replicate estimates.
    pub replicates: u32,
}

/// Inputs shared by Phase 1 effect refuters.
#[derive(Clone, Debug)]
pub struct RefutationProblem<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Identified estimand (backdoor adjustment).
    pub estimand: &'a IdentifiedEstimand,
    /// Treatment.
    pub treatment: VariableId,
    /// Outcome.
    pub outcome: VariableId,
    /// Original point estimate.
    pub original: &'a EffectEstimate,
}

/// Extract a float64 column as owned values.
pub(crate) fn float_col(data: &TabularData, id: VariableId) -> Result<Vec<f64>, ValidationError> {
    data.float64_values(id).map_err(|e| ValidationError::Data(e.to_string()))
}

/// Rebuild tabular data replacing one float column.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn with_replaced_float(
    data: &TabularData,
    id: VariableId,
    values: Arc<[f64]>,
) -> Result<TabularData, ValidationError> {
    let schema = data.schema().clone();
    let n = data.row_count();
    if values.len() != n {
        return Err(ValidationError::Data("replacement length mismatch".into()));
    }
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        if v.id == id {
            cols.push(OwnedColumn::Float64(
                Float64Column::new(v.id, Arc::clone(&values), ValidityBitmap::all_valid(n))
                    .map_err(|e| ValidationError::Data(e.to_string()))?,
            ));
        } else {
            let existing = float_col(data, v.id)?;
            cols.push(OwnedColumn::Float64(
                Float64Column::new(
                    v.id,
                    Arc::<[f64]>::from(existing),
                    ValidityBitmap::all_valid(n),
                )
                .map_err(|e| ValidationError::Data(e.to_string()))?,
            ));
        }
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)
        .map_err(|e| ValidationError::Data(e.to_string()))?;
    Ok(TabularData::new(storage))
}

/// Append an independent continuous covariate; returns new data and its id.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn with_extra_float(
    data: &TabularData,
    name: &str,
    values: Arc<[f64]>,
) -> Result<(TabularData, VariableId), ValidationError> {
    let n = data.row_count();
    if values.len() != n {
        return Err(ValidationError::Data("extra column length mismatch".into()));
    }
    let mut builder = CausalSchemaBuilder::new();
    for v in data.schema().variables() {
        builder
            .add_variable(
                Arc::clone(&v.name),
                v.value_type.clone(),
                v.role_hints,
                v.unit.clone(),
                v.category_domain,
                v.measurement.clone(),
            )
            .map_err(|e| ValidationError::Data(e.to_string()))?;
    }
    builder
        .add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .map_err(|e| ValidationError::Data(e.to_string()))?;
    let schema = builder.build().map_err(|e| ValidationError::Data(e.to_string()))?;
    let new_id = VariableId::from_raw(u32::try_from(schema.len() - 1).expect("schema sized"));
    let mut cols = Vec::with_capacity(schema.len());
    for v in data.schema().variables() {
        let existing = float_col(data, v.id)?;
        cols.push(OwnedColumn::Float64(
            Float64Column::new(v.id, Arc::<[f64]>::from(existing), ValidityBitmap::all_valid(n))
                .map_err(|e| ValidationError::Data(e.to_string()))?,
        ));
    }
    cols.push(OwnedColumn::Float64(
        Float64Column::new(new_id, values, ValidityBitmap::all_valid(n))
            .map_err(|e| ValidationError::Data(e.to_string()))?,
    ));
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)
        .map_err(|e| ValidationError::Data(e.to_string()))?;
    Ok((TabularData::new(storage), new_id))
}

/// Fit linear adjustment once (no nested bootstrap pools).
pub(crate) fn fit_once(
    estimator: &LinearAdjustmentAte,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    treatment: VariableId,
    outcome: VariableId,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, ValidationError> {
    let prep = estimator
        .prepare(data, estimand, treatment, outcome)
        .map_err(|e| ValidationError::Estimation(e.to_string()))?;
    estimator
        .fit(&prep, workspace, ctx, causal_core::AssumptionSet::new())
        .map_err(|e| ValidationError::Estimation(e.to_string()))
}

/// Standard-normal-ish draws via Box–Muller from [`ExecutionContext`] RNG.
pub(crate) fn fill_gaussian(out: &mut [f64], ctx: &ExecutionContext, stream_id: u64) {
    let mut rng = ctx.rng.stream(stream_id);
    let mut i = 0;
    while i < out.len() {
        let u1 = rng.next_f64().clamp(1e-12, 1.0);
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = core::f64::consts::TAU * u2;
        out[i] = r * theta.cos();
        i += 1;
        if i < out.len() {
            out[i] = r * theta.sin();
            i += 1;
        }
    }
}

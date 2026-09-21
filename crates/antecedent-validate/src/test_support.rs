//! Shared fixtures for the crate's unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EffectEstimate, EstimationWorkspace, LinearAdjustmentAte};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;

/// Table whose column 0 is the treatment `t`, column 1 the outcome `y`, and any further
/// columns adjustment covariates `z0, z1, ...`. All rows valid.
pub(crate) fn tabular(columns: &[Vec<f64>]) -> TabularData {
    let n = columns[0].len();
    let mut builder = CausalSchemaBuilder::new();
    for j in 0..columns.len() {
        let (name, role) = match j {
            0 => ("t".to_owned(), RoleHint::TreatmentCandidate),
            1 => ("y".to_owned(), RoleHint::OutcomeCandidate),
            _ => (format!("z{}", j - 2), RoleHint::Context),
        };
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let cols = columns
        .iter()
        .enumerate()
        .map(|(j, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(j).unwrap()),
                    Arc::from(values.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap(),
    )
}

/// Backdoor estimand adjusting for every covariate column of [`tabular`].
pub(crate) fn backdoor(n_covariates: usize) -> IdentifiedEstimand {
    let adjustment: Vec<VariableId> =
        (0..n_covariates).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from(adjustment), ExprId::from_raw(0))
}

/// `binary_ate` query on columns 0 (treatment) and 1 (outcome).
pub(crate) fn ate_query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

/// Least-squares point estimate (no bootstrap) of the problem, with its workspace and context.
pub(crate) fn linear_original(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> (EffectEstimate, EstimationWorkspace, ExecutionContext) {
    let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
    let prep = est.prepare(data, estimand, query).unwrap();
    let mut workspace = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(5);
    let original = est.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
    (original, workspace, ctx)
}

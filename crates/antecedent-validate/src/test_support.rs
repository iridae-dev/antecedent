//! Shared fixtures for the crate's unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView, TabularData,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{EffectEstimate, EstimationWorkspace, LinearAdjustmentAte};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;

/// Two noise-driven series with one true lag-1 link `x -> y`: `x` is a stationary AR(1)
/// and `y_t = 0.9 x_{t-1} + noise`. Both carry independent innovations, so no lagged
/// column is an exact linear combination of others (a noise-free sinusoid is: every
/// three of its lags are collinear, which a partial correlation refuses).
pub(crate) fn driven_xy(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut noise = move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.7 * x[t - 1] + noise();
        y[t] = 0.9 * x[t - 1] + 0.3 * noise();
    }
    (x, y)
}

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

/// Regular time series over the columns of [`tabular`], with each `(column, row)` of `missing`
/// marked invalid.
pub(crate) fn series_with_missing(
    columns: &[Vec<f64>],
    missing: &[(usize, usize)],
) -> TimeSeriesData {
    let n = columns[0].len();
    let schema = tabular(columns).schema().clone();
    let cols = columns
        .iter()
        .enumerate()
        .map(|(j, values)| {
            let flags: Vec<bool> = (0..n).map(|i| !missing.contains(&(j, i))).collect();
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(j).unwrap()),
                    Arc::from(values.clone()),
                    crate::common::validity_from_flags(&flags).unwrap(),
                )
                .unwrap(),
            )
        })
        .collect();
    TimeSeriesData::try_new(
        OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
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
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(data, estimand, query).unwrap();
    let mut workspace = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(5);
    let original = est.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
    (original, workspace, ctx)
}

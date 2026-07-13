//! Effect refuters and validation diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod common;
pub mod error;
pub mod placebo;
pub mod rcc;
pub mod stability;

pub use common::{RefutationProblem, RefutationReport};
pub use error::ValidationError;
pub use placebo::PlaceboTreatment;
pub use rcc::RandomCommonCause;
pub use stability::{BlockBootstrapStability, DiscoveryStabilityReport, LinkStability};

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
    use causal_expr::ExprId;
    use causal_identify::IdentifiedEstimand;

    use super::*;

    fn toy_confounded() -> (TabularData, IdentifiedEstimand, f64) {
        // True ATE = 2; Z confounds T and Y.
        let n = 400usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| if z[i] > 0.5 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 3.0 * z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand, 2.0)
    }

    #[test]
    fn placebo_near_zero_on_null() {
        let (data, estimand, _) = toy_confounded();
        let mut est = LinearAdjustmentAte::new();
        est.bootstrap_replicates = 0;
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(7);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((original.ate - 2.0).abs() < 1e-6);

        let problem = RefutationProblem {
            data: &data,
            estimand: &estimand,
            query: &query,
            original: &original,
        };
        let report = PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap();
        assert!(report.passed, "{:?}", report.failure_condition);
        assert!(report.comparison < 0.25);
    }

    #[test]
    fn rcc_preserves_ate() {
        let (data, estimand, _) = toy_confounded();
        let mut est = LinearAdjustmentAte::new();
        est.bootstrap_replicates = 0;
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(11);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

        let problem = RefutationProblem {
            data: &data,
            estimand: &estimand,
            query: &query,
            original: &original,
        };
        let report = RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap();
        assert!(report.passed, "{:?}", report.failure_condition);
        assert!((report.refuted_ate - original.ate).abs() < 0.15);
    }
}

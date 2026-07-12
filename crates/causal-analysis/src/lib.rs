//! Static ATE identify-estimate-refute facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod analysis;
pub mod error;
pub mod result;

pub use analysis::{CausalAnalysis, CausalAnalysisBuilder, RefuteSuite};
pub use error::AnalysisError;
pub use result::CausalAnalysisResult;

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_graph::{Dag, DenseNodeId};

    use super::*;

    fn scm() -> (TabularData, Dag, AverageEffectQuery) {
        let n = 200usize;
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
        let mut dag = Dag::with_variables(3);
        // z -> t, z -> y, t -> y
        let z_id = DenseNodeId::from_raw(2);
        let t_id = DenseNodeId::from_raw(0);
        let y_id = DenseNodeId::from_raw(1);
        dag.insert_directed(z_id, t_id).unwrap();
        dag.insert_directed(z_id, y_id).unwrap();
        dag.insert_directed(t_id, y_id).unwrap();
        let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_ate() {
        let (data, graph, query) = scm();
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .refute(RefuteSuite::PlaceboAndRcc)
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(3);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 1e-6);
        assert_eq!(result.refutations.len(), 2);
        assert!(!result.provenance.is_empty());
        assert!(!result.identification.derivation.steps.is_empty());
    }
}

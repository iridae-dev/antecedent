//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_estimate::{AnalyticSeKind, TwoStageLeastSquares, WaldIv};
use antecedent_graph::Dag;

use crate::estimator_spec::EstimatorSpec;
use crate::{Study, analysis::builder::RefuteSuite, strategy_table::IdentifierId};

use crate::analysis::prepared::{CheckedIvOperation, PreparedExecution};

fn data(shift: f64) -> TabularData {
    data_rows(shift, 1_600)
}

fn data_rows(shift: f64, n: usize) -> TabularData {
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let ids = [schema.id_of("t").unwrap(), schema.id_of("y").unwrap(), schema.id_of("z").unwrap()];
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let zi = (i % 2) as f64;
        let u = ((i * 37 % 101) as f64 - 50.0) / 30.0;
        let ti = 0.6 * zi + u;
        z.push(zi);
        t.push(ti);
        y.push(shift + 2.0 * ti + u);
    }
    let columns = [t, y, z]
        .into_iter()
        .zip(ids)
        .map(|(values, id)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

fn study(data: TabularData, estimator: impl Into<crate::estimator_spec::EstimatorSpec>) -> Study {
    let graph = Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap();
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        ))
        .identifier(IdentifierId::Auto)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn checked_iv_refresh_publishes_the_refreshed_row_count() {
    let context = ExecutionContext::for_tests(74);
    for rows in [800_usize, 3_200] {
        let mut prepared =
            study(data(0.0), WaldIv::new().with_se_kind(AnalyticSeKind::Homoskedastic))
                .prepare(&context)
                .unwrap();
        let refreshed = prepared.refresh(data_rows(0.4, rows), &context).unwrap();
        let artifact = prepared.encode_contracted_result(&refreshed, "iv", &context).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(
            consumed.acceptance.accepts_as_verified_program(),
            "unresolved={:?}",
            consumed.acceptance.unresolved
        );
        let published = consumed
            .contract
            .as_ref()
            .and_then(|contract| contract.program.as_ref())
            .and_then(|program| program.checked_iv_lowering.as_ref())
            .map(|lowering| lowering.complete_case_rows);
        assert_eq!(published, Some(rows as u64));
    }
}

#[test]
fn checked_iv_survives_builder_drop_and_refresh_with_selected_fitter() {
    let context = ExecutionContext::for_tests(74);
    for (fitter, is_wald) in [
        (study(data(0.0), WaldIv::new().with_se_kind(AnalyticSeKind::Hc1)), true),
        (study(data(0.0), TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1)), false),
    ] {
        let mut prepared = fitter.prepare(&context).unwrap();
        match &prepared.execution {
            PreparedExecution::Iv(CheckedIvOperation::Wald { fitter, preparation }) if is_wald => {
                assert_eq!(fitter.se_kind, AnalyticSeKind::Hc1);
                assert_eq!(
                    preparation.lowering().procedure,
                    antecedent_estimate::CheckedIvProcedure::Wald
                );
            }
            PreparedExecution::Iv(CheckedIvOperation::TwoSls { fitter, preparation })
                if !is_wald =>
            {
                assert_eq!(fitter.se_kind, AnalyticSeKind::Hc1);
                assert_eq!(
                    preparation.lowering().procedure,
                    antecedent_estimate::CheckedIvProcedure::TwoStageLeastSquares
                );
            }
            other => panic!("expected retained checked IV operation, got {other:?}"),
        }
        prepared.analysis.estimator_spec = Some(Arc::new(EstimatorSpec::LinearAdjustmentAte(
            Box::new(antecedent_estimate::LinearAdjustmentAte::new()),
        )));
        let result = prepared.estimate(&data(0.0), &context).unwrap();
        assert!(
            (result.estimate.as_effect().unwrap().ate - 2.0).abs() < 0.15,
            "estimate={}",
            result.estimate.as_effect().unwrap().ate
        );
        assert_eq!(
            result.estimate.as_effect().unwrap().uncertainty.analytic.kind,
            Some(AnalyticSeKind::Hc1)
        );
        prepared.study_mut().estimator_spec = Some(Arc::new(EstimatorSpec::LinearAdjustmentAte(
            Box::new(antecedent_estimate::LinearAdjustmentAte::new()),
        )));
        let refreshed = prepared.refresh(data(0.4), &context).unwrap();
        assert!(
            (refreshed.estimate.as_effect().unwrap().ate - 2.0).abs() < 0.15,
            "estimate={}",
            refreshed.estimate.as_effect().unwrap().ate
        );
        assert_eq!(
            refreshed.estimate.as_effect().unwrap().uncertainty.analytic.kind,
            Some(AnalyticSeKind::Hc1)
        );
        assert!(matches!(prepared.execution, PreparedExecution::Iv(_)));
        prepared.study_mut().query =
            crate::CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                0.0,
                2.0,
            ));
        assert!(
            prepared.estimate(&data(0.0), &context).is_err(),
            "changed contrast must be refused"
        );
    }
}

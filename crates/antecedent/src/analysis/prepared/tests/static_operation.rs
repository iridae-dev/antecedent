//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_estimate::AipwAte;
use antecedent_estimate::GlmAdjustmentAte;
use antecedent_graph::Dag;
use antecedent_stats::GlmFamily;

use crate::Study;
use crate::analysis::builder::RefuteSuite;
use crate::analysis::prepared::PreparedExecution;
use crate::estimator_spec::EstimatorSpec;
use crate::strategy_table::EstimatorId;

fn data() -> TabularData {
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
    let n = 512;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let ti = (i % 2) as f64;
        let zi = ((i * 37 % 101) as f64 - 50.0) / 50.0;
        t.push(ti);
        z.push(zi);
        y.push(2.0 * ti + zi + ((i * 13 % 47) as f64 - 23.0) / 100.0);
    }
    let columns = [("t", t), ("y", y), ("z", z)]
        .into_iter()
        .map(|(name, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    schema.id_of(name).unwrap(),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

#[test]
fn checked_aipw_click_uses_retained_fitter_after_study_config_changes() {
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::with_levels(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
        0.0,
        1.0,
    );
    let context = ExecutionContext::for_tests(23);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(study);
    let PreparedExecution::CheckedAipw(operation) = &prepared.execution else {
        panic!("expected retained checked AIPW operation")
    };
    assert_eq!(operation.fitter.bootstrap_replicates, 0);
    assert_eq!(operation.preparation.lowering().bootstrap_replicates, 0);
    let first = prepared.estimate(&data, &context).unwrap();
    assert!((first.effect() - 2.0).abs() < 0.2);

    // A copied Study is execution metadata, not the source of the fitter.
    // If a click reconstructed AIPW from it, the changed replicate count
    // would conflict with the checked receipt and reject the click.
    let mut changed_fitter = AipwAte::new();
    changed_fitter.bootstrap_replicates = 7;
    prepared.analysis.estimator_spec =
        Some(Arc::new(EstimatorSpec::Aipw(Box::new(changed_fitter))));
    let second = prepared.estimate(&data, &context).unwrap();
    assert!((second.effect() - first.effect()).abs() < 1e-10);
    assert!(second.estimate.as_effect().unwrap().se_bootstrap().is_none());

    prepared.analysis.estimator_spec = Some(Arc::new(EstimatorSpec::LinearAdjustmentAte(
        Box::new(antecedent_estimate::LinearAdjustmentAte::new()),
    )));
    let third = prepared.estimate(&data, &context).unwrap();
    assert!((third.effect() - first.effect()).abs() < 1e-10);
}

#[test]
fn checked_linear_click_uses_retained_default_uncertainty_after_study_changes() {
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::with_levels(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
        0.0,
        1.0,
    );
    let context = ExecutionContext::for_tests(24);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(study);
    let PreparedExecution::CheckedLinear(operation) = &prepared.execution else {
        panic!("expected retained checked linear operation")
    };
    assert!(operation.default_id);
    assert_eq!(operation.fitter.bootstrap_replicates, 0);
    let first = prepared.estimate(&data, &context).unwrap();
    assert!((first.effect() - 2.0).abs() < 0.2);
    assert!(first.estimate.as_effect().unwrap().se_bootstrap().is_none());

    prepared.analysis.bootstrap_replicates = 9;
    prepared.analysis.estimator_spec =
        Some(Arc::new(EstimatorSpec::Aipw(Box::new(AipwAte::new()))));
    let second = prepared.estimate(&data, &context).unwrap();
    assert!((second.effect() - first.effect()).abs() < 1e-10);
    assert!(second.estimate.as_effect().unwrap().se_bootstrap().is_none());

    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let with_bootstrap = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::with_levels(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
            0.0,
            1.0,
        ))
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(8)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let with_bootstrap_result = with_bootstrap.estimate(&data, &context).unwrap();
    assert!(with_bootstrap_result.estimate.as_effect().unwrap().se_bootstrap().is_some());
}

#[test]
fn checked_glm_adjustment_survives_builder_drop_and_refresh() {
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::with_levels(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
        0.0,
        1.0,
    );
    let mut fitter = GlmAdjustmentAte::new().with_family(GlmFamily::GaussianIdentity);
    fitter.bootstrap_replicates = 0;
    let context = ExecutionContext::for_tests(28);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .estimator(fitter)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(study);
    let PreparedExecution::CheckedGlmAdjustment(operation) = &prepared.execution else {
        panic!("expected checked GLM operation")
    };
    assert_eq!(operation.estimator, EstimatorId::GlmAdjustment);
    let first = prepared.estimate(&data, &context).unwrap();
    assert!((first.effect() - 2.0).abs() < 0.2);
    let refreshed = prepared.refresh(data.clone(), &context).unwrap();
    assert!((refreshed.effect() - first.effect()).abs() < 1e-10);
}

#[test]
fn checked_rd_retains_cutoff_target_and_bandwidth_on_refresh() {
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("r", RoleHint::Context),
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
    let n = 200usize;
    let r: Vec<f64> = (0..n).map(|i| (i as f64 - 99.5) / 100.0).collect();
    let t: Vec<f64> = r.iter().map(|&v| f64::from(v >= 0.0)).collect();
    let y: Vec<f64> = r.iter().zip(&t).map(|(&v, &ti)| 1.0 + 0.4 * v + 2.0 * ti).collect();
    let columns = [("t", t), ("y", y), ("r", r)]
        .into_iter()
        .map(|(name, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    schema.id_of(name).unwrap(),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let data =
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
    let graph =
        Dag::from_named_edges(data.schema(), &[("r", "t"), ("r", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
    );
    let context = ExecutionContext::for_tests(29);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .identifier(crate::strategy_table::IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(data.schema().id_of("r").unwrap(), 0.0, 0.8)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(study);
    let PreparedExecution::CheckedRd(operation) = &prepared.execution else {
        panic!("expected checked RD operation")
    };
    assert!((operation.preparation.lowering().bandwidth - 0.8).abs() < 1e-12);
    let first = prepared.estimate(&data, &context).unwrap();
    assert!((first.effect() - 2.0).abs() < 1e-10);
    assert_eq!(
        first.identification.average_effect().unwrap().target_population,
        antecedent_core::TargetPopulation::local_at_cutoff(data.schema().id_of("r").unwrap(), 0.0)
    );
    let refreshed = prepared.refresh(data, &context).unwrap();
    assert!((refreshed.effect() - first.effect()).abs() < 1e-10);
}

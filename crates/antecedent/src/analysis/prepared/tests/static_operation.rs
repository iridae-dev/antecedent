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

struct NoRefreshIdentification;
impl antecedent_core::ProgressSink for NoRefreshIdentification {
    fn report(&self, _fraction: f64, stage: &str) {
        assert_ne!(stage, "identify.compute", "numerical refresh must retain its causal proof");
    }
}

fn data() -> TabularData {
    data_rows(512)
}

fn data_rows(n: usize) -> TabularData {
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
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let ti = f64::from(u32::try_from(i % 2).unwrap());
        let zi = (f64::from(u32::try_from(i * 37 % 101).unwrap()) - 50.0) / 50.0;
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
    let rebound = operation.rebind(&data).unwrap();
    assert!(Arc::ptr_eq(
        &operation.preparation.problem().design.matrix,
        &rebound.preparation.problem().design.matrix
    ));
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

#[test]
fn gaussian_design_reuse_rebinds_rows_and_recomputes_draws() {
    use crate::{BayesianConfig, InferenceMode};
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::with_levels(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
        0.0,
        1.0,
    );
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::production(24, 1);
    let prepared = study.prepare(&ctx).unwrap();
    let cached = |r: &crate::StudyResult| {
        r.diagnostics.iter().any(|d| d.code.as_ref() == "exec.design.cached")
    };
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert!(!cached(&first));
    let second = prepared.estimate(&data, &ctx).unwrap();
    assert!(cached(&second));
    assert_eq!(
        first.posterior.as_ref().unwrap().draws.values,
        second.posterior.as_ref().unwrap().draws.values
    );
    assert_eq!(second.performance.n_draws, Some(64));
    // The Python public default is a bounded 4 MB cache, not an unlimited one.
    let mut python_budget = ctx.clone();
    python_budget.cache_policy = antecedent_core::CachePolicy::enabled(Some(4_000_000));
    assert!(cached(&prepared.estimate(&data, &python_budget).unwrap()));
    let mut tiny_budget = ctx.clone();
    tiny_budget.cache_policy = antecedent_core::CachePolicy::enabled(Some(1));
    assert!(!cached(&prepared.estimate(&data, &tiny_budget).unwrap()));
    assert!(!cached(&prepared.estimate(&data, &python_budget).unwrap()));
    assert!(cached(&prepared.estimate(&data, &python_budget).unwrap()));

    let different_seed = ExecutionContext::production(25, 1);
    let third = prepared.estimate(&data, &different_seed).unwrap();
    assert!(cached(&third));
    assert_ne!(
        second.posterior.as_ref().unwrap().draws.values,
        third.posterior.as_ref().unwrap().draws.values
    );
    let mut no_cache = ctx.clone();
    no_cache.cache_policy = antecedent_core::CachePolicy::disabled();
    let cold = prepared.estimate(&data, &no_cache).unwrap();
    assert!(!cached(&cold));
    assert_eq!(
        first.posterior.as_ref().unwrap().draws.values,
        cold.posterior.as_ref().unwrap().draws.values
    );
    assert!(!cached(&prepared.estimate(&data, &ctx).unwrap()));

    // A new outcome, under the same schema and shape, must replace the binding.
    let t: Vec<_> = (0..512).map(|i| f64::from(u32::try_from(i % 2).unwrap())).collect();
    let z: Vec<_> =
        (0..512).map(|i| (f64::from(u32::try_from(i * 37 % 101).unwrap()) - 50.0) / 50.0).collect();
    let y: Vec<_> =
        (0..512).map(|i| 4.0 * t[i] + z[i] + ((i * 13 % 47) as f64 - 23.0) / 100.0).collect();
    let changed = TabularData::try_from_schema_f64(
        data.schema().clone(),
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
    )
    .unwrap();
    let changed_result = prepared.estimate(&changed, &ctx).unwrap();
    assert!(!cached(&changed_result));
    assert!((changed_result.effect() - 4.0).abs() < 0.2);
    assert!(cached(&prepared.estimate(&changed, &ctx).unwrap()));
    let mut bounded = ctx.clone();
    bounded.cache_policy = antecedent_core::CachePolicy::enabled(Some(0));
    assert!(!cached(&prepared.estimate(&changed, &bounded).unwrap()));
    assert!(!cached(&prepared.estimate(&changed, &ctx).unwrap()));
    let cancelled = ctx.clone();
    cancelled.cancellation.cancel();
    assert!(prepared.estimate(&changed, &cancelled).is_err());
}

#[test]
fn failed_dr_configuration_refresh_preserves_estimator_identity_and_cached_state() {
    use antecedent_estimate::{DmlAte, DrLearner, OverlapPolicy};
    use antecedent_learn::{LearnerSpec, LinearSpec};

    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
    );
    let learner = LearnerSpec::Linear(LinearSpec::default());
    let overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: None };
    let configurations: [(EstimatorSpec, EstimatorSpec); 2] = [
        (
            DmlAte::new().with_outcome(learner.clone()).with_folds(4).into(),
            DmlAte::new().with_outcome(learner.clone()).with_folds(3).with_overlap(overlap).into(),
        ),
        (
            DrLearner::new().with_outcome(learner.clone()).with_folds(4).into(),
            DrLearner::new().with_outcome(learner).with_folds(3).with_overlap(overlap).into(),
        ),
    ];
    for (original, changed) in configurations {
        let ctx = ExecutionContext::for_tests(47);
        let mut prepared = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(query.clone())
            .estimator(original)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let before = prepared.contract().unwrap().identities;
        let config = Arc::clone(&prepared.analysis.config);
        let estimator = Arc::clone(prepared.analysis.estimator_spec.as_ref().unwrap());
        let program = Arc::clone(prepared.program_cache.get().unwrap());
        let physical = Arc::clone(&prepared.plan);
        let prior_plan = prepared.plan_view().unwrap();
        assert_eq!(prior_plan.statistical().folds().unwrap().folds, 4);
        let old_overlap = prepared.analysis.overlap_policy;
        let baseline = prepared.estimate(&data, &ctx).unwrap().effect();

        // Configuration is installed on the candidate before refresh checks
        // this incompatible schema, so that refusal tests rollback after install.
        // A precancelled attempt separately refuses before cloning or fitting.
        let incompatible = TabularData::from_f64_columns(vec![("other", &[0.0, 1.0][..])]).unwrap();
        assert!(prepared.refresh_dr_configuration(incompatible, changed.clone(), &ctx).is_err());
        let cancelled = ExecutionContext::for_tests(47);
        cancelled.cancellation.cancel();
        assert!(
            prepared.refresh_dr_configuration(data.clone(), changed.clone(), &cancelled).is_err()
        );
        let bounded = ExecutionContext::for_tests(47).with_memory_limit(Some(1));
        assert!(matches!(
            prepared.refresh_dr_configuration(data.clone(), changed.clone(), &bounded),
            Err(crate::CausalError::Resource { .. })
        ));
        assert!(Arc::ptr_eq(&physical, &prepared.plan));
        assert!(Arc::ptr_eq(&config, &prepared.analysis.config));
        assert!(Arc::ptr_eq(&estimator, prepared.analysis.estimator_spec.as_ref().unwrap()));
        assert!(Arc::ptr_eq(&program, prepared.program_cache.get().unwrap()));
        assert_eq!(old_overlap, prepared.analysis.overlap_policy);
        assert_eq!(before, prepared.contract().unwrap().identities);
        assert_eq!(baseline.to_bits(), prepared.estimate(&data, &ctx).unwrap().effect().to_bits());

        let expected_spec = crate::analysis::contract_identity::estimator_spec_identity(&changed);
        let replacement = data_rows(384);
        let fresh = Study::tabular(replacement.clone())
            .graph(graph.clone())
            .query(query.clone())
            .estimator(changed.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let expected_result = fresh.estimate(&replacement, &ctx).unwrap();
        let mut refresh_ctx = ctx.clone();
        refresh_ctx.progress = Some(Arc::new(NoRefreshIdentification));
        let refreshed =
            prepared.refresh_dr_configuration(replacement, changed, &refresh_ctx).unwrap();
        assert!((refreshed.effect() - 2.0).abs() < 0.2);
        assert_eq!(refreshed.effect().to_bits(), expected_result.effect().to_bits());
        assert_eq!(prepared.analysis.estimator_spec_identity.as_ref(), Some(&expected_spec));
        let after = prepared.contract().unwrap().identities;
        assert_eq!(before.target, after.target);
        assert_eq!(before.identification, after.identification);
        assert_eq!(before.program, after.program, "numerical controls retain the causal program");
        assert_ne!(before.inference_binding, after.inference_binding);
        assert_eq!(after, fresh.contract().unwrap().identities);
        let view = prepared.plan_view().unwrap();
        let expected = fresh.plan_view().unwrap();
        assert_eq!(view.statistical().folds().unwrap().folds, 3);
        assert_ne!(prior_plan.identity(), view.identity());
        assert_eq!(view.identity(), expected.identity());
        assert_eq!(view.physical().design().rows(), Some(384));
        assert_eq!(view.statistical().models(), expected.statistical().models());
        assert_eq!(view.statistical().resampling(), expected.statistical().resampling());
        assert_eq!(view.to_json()["physical"], expected.to_json()["physical"]);
        assert_eq!(prepared.analysis.overlap_policy, None);
        assert_eq!(
            prepared.analysis.estimator_spec.as_ref().unwrap().propensity_overlap(),
            Some(overlap)
        );
    }
}

#[test]
fn aipw_configuration_refresh_relowers_resampling_and_reseals_its_physical_plan() {
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
    );
    let ctx = ExecutionContext::for_tests(49);
    let build = |estimator: AipwAte| {
        Study::tabular(data.clone())
            .graph(graph.clone())
            .query(query.clone())
            .estimator(estimator)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
    };
    let mut prepared = build(AipwAte::new().with_bootstrap_replicates(0));
    let before = prepared.contract().unwrap().identities;
    let prior_plan = Arc::clone(&prepared.plan);
    let prior_config = Arc::clone(&prepared.analysis.config);
    let baseline = prepared.estimate(&data, &ctx).unwrap();
    let changed = AipwAte::new().with_bootstrap_replicates(8).with_overlap(
        antecedent_estimate::OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: None },
    );
    let bounded = ExecutionContext::for_tests(49).with_memory_limit(Some(1));
    assert!(matches!(
        prepared.refresh_aipw_configuration(data.clone(), changed.clone(), &bounded),
        Err(crate::CausalError::Resource { .. })
    ));
    assert!(Arc::ptr_eq(&prior_plan, &prepared.plan));
    assert!(Arc::ptr_eq(&prior_config, &prepared.analysis.config));
    assert_eq!(before, prepared.contract().unwrap().identities);
    assert_eq!(
        baseline.effect().to_bits(),
        prepared.estimate(&data, &ctx).unwrap().effect().to_bits()
    );
    let fresh = build(changed.clone());
    let expected_result = fresh.estimate(&data, &ctx).unwrap();
    let mut refresh_ctx = ctx.clone();
    refresh_ctx.progress = Some(Arc::new(NoRefreshIdentification));
    let result = prepared.refresh_aipw_configuration(data.clone(), changed, &refresh_ctx).unwrap();
    assert_eq!(result.effect().to_bits(), expected_result.effect().to_bits());
    assert_eq!(prepared.contract().unwrap().identities, fresh.contract().unwrap().identities);
    let view = prepared.plan_view().unwrap();
    let expected = fresh.plan_view().unwrap();
    assert_eq!(view.identity(), expected.identity());
    assert_eq!(view.statistical().models(), expected.statistical().models());
    assert_eq!(view.statistical().folds(), expected.statistical().folds());
    assert_eq!(view.statistical().resampling().unwrap().replicates(), 8);
    assert_eq!(prepared.analysis.bootstrap_replicates, 8);
    assert_eq!(view.statistical().resampling(), expected.statistical().resampling());
    assert_eq!(view.to_json()["physical"], expected.to_json()["physical"]);
    let PreparedExecution::CheckedAipw(operation) = &prepared.execution else {
        panic!("configured refresh must retain the checked AIPW family");
    };
    assert!(Arc::ptr_eq(&operation.common.physical, &prepared.plan));
}

#[test]
fn admitted_population_retarget_after_refresh_matches_fresh_preparation() {
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let t = data.schema().id_of("t").unwrap();
    let y = data.schema().id_of("y").unwrap();
    let z = data.schema().id_of("z").unwrap();
    let ctx = ExecutionContext::for_tests(48);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(t, y))
        .estimator(AipwAte::new().with_bootstrap_replicates(0))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut retained = study.prepare(&ctx).unwrap();
    let before = retained.contract().unwrap().identities;
    let frozen = retained.clone();
    retained.refresh(data.clone(), &ctx).unwrap();
    let fresh = study.prepare(&ctx).unwrap();
    let table = retained.score_table().unwrap();
    let z_values = data.float64_values(z).unwrap();
    let weights: Vec<_> = table
        .row_index
        .iter()
        .map(|row| 1.0 + 0.2 * z_values[usize::try_from(row.get()).unwrap()])
        .collect();
    let actual = retained.retarget(&weights, &[z], &ctx).unwrap();
    let expected = fresh.retarget(&weights, &[z], &ctx).unwrap();
    assert_eq!(actual.effect().to_bits(), expected.effect().to_bits());
    assert_eq!(actual.executed_contract, expected.executed_contract);
    assert_eq!(before, retained.contract().unwrap().identities);
    assert_eq!(before, frozen.contract().unwrap().identities);
    let body = retained.encode_contracted_result(&actual, "retarget-after-refresh", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&body).unwrap();
    assert_eq!(consumed.body.estimate, Some(actual.effect()));
}

#[test]
fn signed_zero_query_drift_is_refused_before_posterior_contract_work() {
    use antecedent_core::{
        CausalQuery, Intervention, InterventionalDistributionQuery, Value, VariableId,
    };
    use antecedent_discovery::{GraphPosterior, SamplingSummary, set_edge};
    use antecedent_prob::InferenceDiagnostics;

    let treatment = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let outcome = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let context = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let rows = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", context.as_slice()),
    ])
    .unwrap();
    let posterior = GraphPosterior::new(
        3,
        vec![1.0],
        vec![set_edge(0, 3, 0, 1, true)],
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("signed_zero_retained_query"),
            rejected_invalid: 0,
        },
    )
    .unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let original = CausalQuery::Distribution(InterventionalDistributionQuery::new(
        y,
        [Intervention::set(t, Value::f64(0.0))],
    ));
    let changed = CausalQuery::Distribution(InterventionalDistributionQuery::new(
        y,
        [Intervention::set(t, Value::f64(-0.0))],
    ));
    let mut prepared = Study::tabular(rows)
        .graph_posterior(posterior)
        .query(original.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(49))
        .unwrap();
    assert!(prepared.posterior_programs(&original).unwrap().is_some());
    let (refused, work) = antecedent_expr::execution_counts::count_static_work(|| {
        prepared.posterior_programs(&changed)
    });
    assert!(refused.is_err(), "bit-distinct signed-zero target must require preparation");
    assert_eq!(work, antecedent_expr::execution_counts::StaticWorkCounts::default());
    prepared.analysis.query = changed;
    let (refused, work) =
        antecedent_expr::execution_counts::count_static_work(|| prepared.contract());
    assert!(refused.is_err(), "contract must refuse internally altered frozen target");
    assert_eq!(work, antecedent_expr::execution_counts::StaticWorkCounts::default());
}

#[test]
fn late_stage_cancellation_never_commits_a_refresh_candidate() {
    use crate::analysis::{StageEvent, StageResultSink};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct CancelAfterValidation {
        token: antecedent_core::CancellationToken,
        completed: AtomicBool,
    }
    impl StageResultSink for CancelAfterValidation {
        fn on_stage(&self, event: &StageEvent) {
            if matches!(event, StageEvent::Validate { .. }) {
                self.completed.store(true, Ordering::SeqCst);
                self.token.cancel();
            }
        }
    }

    let rows = data();
    let graph =
        Dag::from_named_edges(rows.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(
        rows.schema().id_of("t").unwrap(),
        rows.schema().id_of("y").unwrap(),
    );
    let ctx = ExecutionContext::for_tests(50);
    let study = Study::tabular(rows.clone())
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let before = prepared.contract().unwrap().identities;
    let baseline = prepared.estimate(&rows, &ctx).unwrap().effect();
    let sink = Arc::new(CancelAfterValidation {
        token: ctx.cancellation.clone(),
        completed: AtomicBool::new(false),
    });
    prepared.set_stage_sink(Some(sink.clone()));
    let frozen = prepared.clone();
    let outcome = rows.schema().id_of("y").unwrap();
    let mut columns = rows.storage().columns().to_vec();
    columns[usize::try_from(outcome.raw()).unwrap()] = OwnedColumn::Float64(
        Float64Column::new(
            outcome,
            Arc::from(
                rows.float64_values(outcome)
                    .unwrap()
                    .into_iter()
                    .map(|value| 2.0 * value)
                    .collect::<Vec<_>>(),
            ),
            ValidityBitmap::all_valid(rows.row_count()),
        )
        .unwrap(),
    );
    let changed = TabularData::new(
        OwnedColumnarStorage::try_new(rows.schema().clone(), columns, None, None).unwrap(),
    );
    let refused = prepared.refresh(changed.clone(), &ctx).unwrap_err();
    assert!(
        sink.completed.load(Ordering::SeqCst),
        "cancel must occur after point/uncertainty/validation completed"
    );
    assert!(matches!(refused, crate::CausalError::Cancelled { stage: "refresh" }), "{refused}");
    assert_eq!(before, prepared.contract().unwrap().identities);
    assert_eq!(before, frozen.contract().unwrap().identities);
    assert!(prepared.temporal_invalidations().entries.is_empty());
    prepared.set_stage_sink(None);
    let recovered = ExecutionContext::for_tests(50);
    assert_eq!(
        baseline.to_bits(),
        prepared.estimate(&rows, &recovered).unwrap().effect().to_bits()
    );
    let refreshed = prepared.refresh(changed, &recovered).unwrap();
    assert!((refreshed.effect() - 2.0 * baseline).abs() < 1e-10);
    assert_ne!(before.data_snapshot, prepared.contract().unwrap().identities.data_snapshot);
}

#[test]
fn unsealed_bayesian_basis_proof_replay_uses_the_construction_context() {
    let data = data();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
    );
    let inference = crate::InferenceMode::Bayesian(crate::BayesianConfig::conjugate().n_draws(16));
    let study = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query.clone())
        .estimator(EstimatorId::BayesianBasisGcomp)
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(87);
    let prepared = study.prepare(&ctx).unwrap();
    let cache = prepared.study().identification_cache.as_ref().unwrap();
    let mut physical = (*prepared.plan).clone();
    physical.lowered = None;
    let physical = Arc::new(physical);
    let check = |ctx: &ExecutionContext| {
        crate::analysis::execute::CheckedBayesianBasisAteExecution::checked(
            &data,
            &graph,
            query.clone(),
            cache.clone(),
            inference.clone(),
            crate::analysis::execute::IdentifiedResultContext::from_study(prepared.study()),
            crate::analysis::proof_support::CheckedCompileContext {
                physical: physical.clone(),
                ctx,
            },
        )
    };
    assert!(check(&ctx).is_ok());
    let cancelled = ExecutionContext::for_tests(87);
    cancelled.cancellation.cancel();
    assert!(matches!(check(&cancelled), Err(crate::CausalError::Cancelled { .. })));
    let limited = ExecutionContext::for_tests(87).with_memory_limit(Some(1));
    assert!(matches!(check(&limited), Err(crate::CausalError::Resource { .. })));
}

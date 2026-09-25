//! Builder-independent evidence for the licensed frequentist static mediation procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, MediationContrast, MediationQuery, Value,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn graph() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (a, b) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph
}

fn bayesian_data() -> TabularData {
    let (mut treatment, mut mediator, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..320 {
        let a = ((i as f64) * 0.71).sin();
        let m = 2.0 * a + ((i as f64) * 1.13).cos();
        let y = 3.0 * a + 4.0 * m + 0.1 * ((i as f64) * 0.31).sin();
        treatment.push(a);
        mediator.push(m);
        outcome.push(y);
    }
    TabularData::from_f64_columns([
        ("a", treatment.as_slice()),
        ("m", mediator.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap()
}

#[test]
fn static_mediation_preserves_declared_nonbinary_levels_and_default_validation() {
    let base = data(0.0);
    let context = ExecutionContext::for_tests(204);
    for (contrast, unit_truth) in [
        (MediationContrast::Total, 0.7),
        (MediationContrast::Direct, 0.4),
        (MediationContrast::Mediated, 0.3),
        (MediationContrast::NaturalDirect, 0.4),
        (MediationContrast::NaturalIndirect, 0.3),
    ] {
        let mut query = MediationQuery::binary(v(0), v(2), [v(1)], contrast);
        query.control = Intervention::set(v(0), Value::f64(0.2));
        query.active = Intervention::set(v(0), Value::f64(0.8));
        let builder = Study::tabular(base.clone())
            .graph(graph())
            .query(CausalQuery::Mediation(query.clone()))
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let mut prepared = builder.prepare(&context).unwrap();
        drop(builder);
        let plan = prepared.checked_static_mediation_info().unwrap();
        assert_eq!(plan.query, query);
        assert_eq!(plan.validation, RefuteSuite::PlaceboAndRcc);
        let result = prepared.estimate(&base, &context).unwrap();
        assert!((result.estimate.ate - 0.6 * unit_truth).abs() < 1e-10);
        assert!(!result.refutations.is_empty());
        let refreshed = prepared.refresh(data(0.15), &context).unwrap();
        let direct_delta = if matches!(
            contrast,
            MediationContrast::Total | MediationContrast::Direct | MediationContrast::NaturalDirect
        ) {
            0.15
        } else {
            0.0
        };
        assert!((refreshed.estimate.ate - 0.6 * (unit_truth + direct_delta)).abs() < 1e-10);
        assert_eq!(prepared.checked_static_mediation_info().unwrap().query, query);
    }
}

fn data(direct_delta: f64) -> TabularData {
    let (mut t, mut m, mut y, mut x) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // Balanced four-factor design gives independent errors and exact known
    // structural coefficients in each equation, without fitting as an oracle.
    for _ in 0..20 {
        for mask in 0..16 {
            let sign = |bit| if mask & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let z = sign(0);
            let treatment = 0.5 * z + sign(1);
            let mediator = 0.6 * treatment + 0.2 * z + 0.3 * sign(2);
            let outcome =
                (0.4 + direct_delta) * treatment + 0.5 * mediator + 0.3 * z + 0.2 * sign(3);
            x.push(z);
            t.push(treatment);
            m.push(mediator);
            y.push(outcome);
        }
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

#[test]
fn static_mediation_plan_executes_every_contrast_and_licensed_validation_suite() {
    run_on_large_stack(
        static_mediation_plan_executes_every_contrast_and_licensed_validation_suite_inner,
    );
}

fn static_mediation_plan_executes_every_contrast_and_licensed_validation_suite_inner() {
    let base = data(0.0);
    let context = ExecutionContext::for_tests(203);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            for (contrast, truth) in [
                (MediationContrast::Total, 0.7),
                (MediationContrast::Direct, 0.4),
                (MediationContrast::Mediated, 0.3),
                (MediationContrast::NaturalDirect, 0.4),
                (MediationContrast::NaturalIndirect, 0.3),
            ] {
                let query = MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), contrast);
                let builder = Study::tabular(base.clone());
                let builder = if accepted {
                    builder.graph(AcceptedGraph::dag(graph()))
                } else {
                    builder.graph(graph())
                };
                let builder = builder
                    .query(CausalQuery::Mediation(query.clone()))
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&context).unwrap();
                assert!((one_shot.estimate.ate - truth).abs() < 1e-10);
                assert!(
                    one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );
                let mut prepared = builder.prepare(&context).unwrap();
                drop(builder);
                let plan = prepared.checked_static_mediation_info().expect("sealed mediation plan");
                assert_eq!(plan.query, query);
                assert_eq!(plan.identifier.as_str(), "path_specific.natural");
                assert_eq!(plan.estimator.as_str(), "mediation.linear");
                assert_eq!(plan.validation, suite);
                assert_eq!(plan.bootstrap_replicates, 0);
                assert_eq!(plan.graph_edges.len(), 6);
                assert_eq!(plan.functional_roots.0, plan.functional_roots.1);
                let result = prepared.estimate(&base, &context).unwrap();
                assert!(
                    (result.estimate.ate - truth).abs() < 1e-10,
                    "{accepted} {suite:?} {contrast:?}: {} vs {truth}",
                    result.estimate.ate
                );
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );
                if suite == RefuteSuite::None {
                    assert!(result.refutations.is_empty());
                } else {
                    assert!(
                        !result.refutations.is_empty(),
                        "{suite:?} must execute its retained refuter"
                    );
                }
                let refreshed = prepared.refresh(data(0.15), &context).unwrap();
                let refreshed_truth = if matches!(
                    contrast,
                    MediationContrast::Total
                        | MediationContrast::Direct
                        | MediationContrast::NaturalDirect
                ) {
                    truth + 0.15
                } else {
                    truth
                };
                assert!((refreshed.estimate.ate - refreshed_truth).abs() < 1e-10);
                assert_eq!(prepared.checked_static_mediation_info().unwrap().query, query);
                let artifact = prepared
                    .encode_contracted_result(&refreshed, "checked-mediation", &context)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_mediation_operation"
                }));
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

#[test]
fn bayesian_static_mediation_plan_survives_builder_refresh_and_artifact_consume() {
    let base = bayesian_data();
    let context = ExecutionContext::for_tests(205);
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            for (contrast, expected) in
                [(MediationContrast::NaturalDirect, 1.8), (MediationContrast::NaturalIndirect, 4.8)]
            {
                let mut query = MediationQuery::binary(v(0), v(2), [v(1)], contrast);
                query.control = Intervention::set(v(0), Value::f64(0.2));
                query.active = Intervention::set(v(0), Value::f64(0.8));
                let builder = Study::tabular(base.clone());
                let builder = if accepted {
                    builder.graph(AcceptedGraph::dag(graph.clone()))
                } else {
                    builder.graph(graph.clone())
                }
                .query(CausalQuery::Mediation(query.clone()))
                .inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(32).prior_scale(1_000.0),
                ))
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
                let mut prepared = builder.prepare(&context).unwrap();
                let plan = prepared
                    .checked_bayesian_static_mediation_info()
                    .expect("sealed Bayesian mediation plan");
                assert_eq!(plan.query, query);
                assert_eq!(plan.validation, suite);
                assert_eq!(plan.identifier.as_str(), "path_specific.natural");
                assert_eq!(plan.estimator.as_str(), "mediation.linear");
                assert_eq!(plan.functional_roots.0, plan.functional_roots.1);
                drop(builder);

                let result = prepared.estimate(&base, &context).unwrap();
                assert!(result.posterior.is_some());
                assert!(
                    (result.estimate.ate - expected).abs() < 0.5,
                    "{accepted} {suite:?}: {}",
                    result.estimate.ate
                );
                if contrast == MediationContrast::NaturalIndirect {
                    assert!(
                        result.posterior.as_ref().unwrap().assumptions.entries.iter().any(
                            |record| matches!(
                                &record.assumption,
                                antecedent_core::Assumption::ParametricRestriction(restriction)
                                    if restriction.id.as_ref() == "mediation.no_interaction"
                            )
                        ),
                        "NaturalIndirect must retain the no-interaction alias premise"
                    );
                }
                assert_eq!(result.refutations.is_empty(), suite == RefuteSuite::None);
                let refreshed = prepared.refresh(base.clone(), &context).unwrap();
                assert!((refreshed.estimate.ate - result.estimate.ate).abs() < 1e-10);
                assert_eq!(prepared.checked_bayesian_static_mediation_info().unwrap().query, query);
                let artifact = prepared
                    .encode_contracted_result(&refreshed, "checked-bayesian-mediation", &context)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_bayesian_mediation_operation"
                }));
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

fn equal_width_mediation_data() -> (TabularData, Dag) {
    let (mut z_values, mut t_values, mut m_values, mut y_values) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for repeat in 0..80 {
        for mask in 0..8 {
            let sign = |bit| if mask & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let z = sign(0);
            let treatment = sign(1);
            let mediator = 0.6 * treatment + 0.2 * z + 0.1 * sign(2);
            let outcome = 0.4 * treatment
                + 0.5 * mediator
                + 0.3 * z
                + 0.01 * (f64::from(repeat) * 0.13).sin();
            z_values.push(z);
            t_values.push(treatment);
            m_values.push(mediator);
            y_values.push(outcome);
        }
    }
    let data = TabularData::from_f64_columns([
        ("z", z_values.as_slice()),
        ("t", t_values.as_slice()),
        ("m", m_values.as_slice()),
        ("y", y_values.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (parent, child) in [(0, 2), (1, 2), (1, 3), (2, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(parent), DenseNodeId::from_raw(child)).unwrap();
    }
    (data, graph)
}

#[test]
fn bayesian_static_mediation_prior_cannot_alias_mechanisms_with_equal_design_width() {
    run_on_large_stack(
        bayesian_static_mediation_prior_cannot_alias_mechanisms_with_equal_design_width_inner,
    );
}

fn bayesian_static_mediation_prior_cannot_alias_mechanisms_with_equal_design_width_inner() {
    let (target_data, target_graph) = equal_width_mediation_data();
    let source_t = target_data.float64_values(v(1)).unwrap();
    let source_y = source_t
        .iter()
        .enumerate()
        .map(|(row, treatment)| 1.5 + 2.0 * treatment + 0.02 * (row as f64 * 0.3).sin())
        .collect::<Vec<_>>();
    let source_data = TabularData::from_f64_columns([
        ("t", source_t.as_slice()),
        ("y_source", source_y.as_slice()),
    ])
    .unwrap();
    let mut source_graph = Dag::with_variables(2);
    source_graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let context = ExecutionContext::for_tests(206);
    let source = Study::tabular(source_data)
        .graph(source_graph)
        .query(antecedent_core::AverageEffectQuery::with_levels(v(0), v(1), -1.0, 1.0))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&context)
        .unwrap();
    let bytes = antecedent::io::encode_causal_posterior_bytes(
        source.posterior.as_ref().unwrap(),
        "equal-width-coefficient-prior",
    )
    .unwrap();
    let mut query = MediationQuery::binary(v(1), v(3), [v(2)], MediationContrast::NaturalDirect);
    query.control = Intervention::set(v(1), Value::f64(-1.0));
    query.active = Intervention::set(v(1), Value::f64(1.0));

    let identical = BayesianConfig::conjugate().n_draws(64).prior_from_artifact(
        bytes.to_vec(),
        Some(antecedent_io::PriorMapping::IdenticalCoefficientSubspace),
    );
    let builder = Study::tabular(target_data.clone())
        .graph(target_graph.clone())
        .query(CausalQuery::Mediation(query.clone()))
        .inference(InferenceMode::Bayesian(identical))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let error = builder.prepare(&context).unwrap_err();
    assert!(error.to_string().contains("identical coefficient-subspace mapping"), "{error}");

    let named = BayesianConfig::conjugate().n_draws(64).prior_from_artifact(
        bytes.to_vec(),
        Some(antecedent_io::PriorMapping::NamedParameters {
            pairs: vec![("coef_t".into(), "coef_t".into())],
        }),
    );
    let result = Study::tabular(target_data)
        .graph(target_graph)
        .query(CausalQuery::Mediation(query))
        .inference(InferenceMode::Bayesian(named))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&context)
        .unwrap();
    let scopes = result
        .posterior
        .as_ref()
        .unwrap()
        .assumptions
        .entries
        .iter()
        .filter_map(|record| {
            (matches!(
                &record.assumption,
                antecedent_core::Assumption::PriorRestriction(prior)
                    if prior.id.as_ref() == "external_named_prior"
            ))
            .then_some(&record.scope)
        })
        .collect::<Vec<_>>();
    assert_eq!(scopes.len(), 1, "the named prior may bind one mechanism only: {scopes:?}");
    assert!(matches!(
        scopes[0],
        antecedent_core::AssumptionScope::Variables { variables }
            if variables.as_ref() == [v(3)]
    ));
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-mediation-evidence".into())
        .stack_size(4 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

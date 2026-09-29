//! Builder-independent evidence for Bayesian DAG conditional effects.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, StructureSource, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, SlotAvailability,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

fn conditional_scm() -> (TabularData, Dag, ConditionalEffectQuery) {
    conditional_scm_with_noise(false)
}

/// The conditional SCM, optionally with a trailing column no estimand reads.
fn conditional_scm_with_noise(noise: bool) -> (TabularData, Dag, ConditionalEffectQuery) {
    let n = 1_200usize;
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut modifier = Vec::with_capacity(n);
    let mut second_modifier = Vec::with_capacity(n);
    let mut confounder = Vec::with_capacity(n);
    for row in 0..n {
        let z = (row % 8) as f64 - 3.5;
        let w = (row % 6) as f64 - 2.5;
        let u = ((row * 17 % 31) as f64 - 15.0) / 15.0;
        let t = f64::from((row * 13 + row / 8) % 11 < 5);
        modifier.push(z);
        second_modifier.push(w);
        confounder.push(u);
        treatment.push(t);
        outcome.push(1.0 + 0.7 * u + 2.0 * t + 0.5 * t * z + 0.25 * t * w);
    }
    let unused: Vec<f64> = (0..n).map(|row| ((row * 7 % 13) as f64) / 13.0).collect();
    let mut columns = vec![
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", modifier.as_slice()),
        ("w", second_modifier.as_slice()),
        ("u", confounder.as_slice()),
    ];
    if noise {
        columns.push(("noise", unused.as_slice()));
    }
    let data = TabularData::from_f64_columns(columns).unwrap();
    let mut graph = Dag::with_variables(if noise { 6 } else { 5 });
    for (from, to) in [(4, 0), (4, 1), (0, 1), (2, 1), (3, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2), VariableId::from_raw(3)]),
    )
    .unwrap();
    (data, graph, query)
}

#[test]
fn sealed_and_one_shot_bayesian_conditional_agree_on_a_wide_table() {
    let ctx = ExecutionContext::for_tests(813);
    let (data, graph, query) = conditional_scm_with_noise(true);
    let config = BayesianConfig::conjugate().n_draws(64).prior_scale(30.0);
    let study = |suite| {
        Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::ConditionalEffect(query.clone()))
            .inference(InferenceMode::Bayesian(config.clone()))
            .refute(suite)
            .build()
            .unwrap()
    };
    let prepared = study(RefuteSuite::None).prepare(&ctx).unwrap();
    assert!(prepared.checked_bayesian_conditional_operation().is_some());
    let sealed = prepared.estimate(&data, &ctx).unwrap();
    // Placebo-and-RCC is not a sealed suite, so this runs the one-shot path.
    let one_shot = study(RefuteSuite::PlaceboAndRcc).run(&ctx).unwrap();
    let projected = |result: &antecedent::StudyResult| {
        result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.project.columns")
    };
    assert!(projected(&one_shot), "one-shot projects the unused column away");
    assert!(projected(&sealed), "sealed execution projects the unused column away");
    assert_eq!(sealed.estimate.ate.to_bits(), one_shot.estimate.ate.to_bits());
}

#[test]
fn bayesian_conditional_default_estimator_ignores_builder_call_order() {
    let ctx = ExecutionContext::for_tests(812);
    let (data, graph, query) = conditional_scm();
    let config = BayesianConfig::conjugate().n_draws(64).prior_scale(30.0);
    let inference_first = Study::tabular(data.clone())
        .graph(graph.clone())
        .inference(InferenceMode::Bayesian(config.clone()))
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .refute(RefuteSuite::None);
    let query_first = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::ConditionalEffect(query))
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None);
    let mut estimators = Vec::new();
    for (label, builder) in [("inference-first", inference_first), ("query-first", query_first)] {
        let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
        assert!(
            prepared.checked_bayesian_conditional_operation().is_some(),
            "{label}: expected the sealed Bayesian conditional operation"
        );
        assert_eq!(
            prepared.plan().logical.record.estimator.as_deref(),
            Some("conditional.bayesian"),
            "{label}: resolved estimator"
        );
        let contract = prepared.contract().unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        prepared
            .encode_contracted_result(&result, &format!("bayesian-conditional-{label}"), &ctx)
            .unwrap();
        estimators.push((contract.estimator.clone(), contract.identities.program));
    }
    assert_eq!(estimators[0], estimators[1], "call order changed the contract");
}

#[test]
fn bayesian_conditional_effect_keeps_target_prior_and_procedure_after_builder_drop() {
    let ctx = ExecutionContext::for_tests(811);
    let (data, graph, query) = conditional_scm();
    for accepted in [false, true] {
        for (label, suite) in [
            ("none", RefuteSuite::None),
            ("cheap", RefuteSuite::Cheap),
            ("full", RefuteSuite::Full),
        ] {
            let config = BayesianConfig::conjugate().n_draws(96).prior_scale(30.0);
            let base = Study::tabular(data.clone())
                .query(CausalQuery::ConditionalEffect(query.clone()))
                .estimator(EstimatorId::BayesianConditional)
                .inference(InferenceMode::Bayesian(config.clone()))
                .refute(suite);
            let builder = if accepted {
                base.graph(AcceptedGraph::from(graph.clone()))
            } else {
                base.graph(graph.clone())
            };
            let study = builder.clone().build().unwrap();
            let mut prepared = study.prepare(&ctx).unwrap();
            drop(builder);
            drop(study);

            let coordinate = format!(
                "ConditionalEffect:Dag:{}:Bayesian:{label}",
                if accepted { "accepted" } else { "explicit" }
            );
            assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed));
            assert_eq!(
                prepared.structure_source(),
                if accepted { StructureSource::Accepted } else { StructureSource::Explicit }
            );
            match &prepared.contract().unwrap().reasoning.support {
                SlotAvailability::Available(slot) => {
                    assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate.as_str()));
                }
                other => panic!("{coordinate}: support was not available: {other:?}"),
            }
            let plan = prepared
                .checked_bayesian_conditional_operation()
                .expect("retained Bayesian conditional operation");
            assert_eq!(plan.query(), &query);
            assert_eq!(plan.inference(), &InferenceMode::Bayesian(config));
            assert_eq!(plan.validation(), suite);
            assert_eq!(
                plan.identification_status(),
                antecedent_identify::IdentificationStatus::NonparametricallyIdentified
            );
            assert!(plan.estimand().adjustment_set.contains(&VariableId::from_raw(4)));
            assert_eq!(plan.modifier_roles(), &[VariableId::from_raw(2), VariableId::from_raw(3)]);
            assert_eq!(
                prepared.plan().logical.record.estimator.as_deref(),
                Some("conditional.bayesian")
            );

            let result = prepared.estimate(&data, &ctx).unwrap();
            // The interaction terms are centered, so the treatment coefficient
            // is the empirical-population mean of the conditional contrast.
            // Both modifiers have mean zero and the independent SCM mean is 2.
            assert!(
                (result.estimate.ate - 2.0).abs() < 0.2,
                "{coordinate}: {}",
                result.estimate.ate
            );
            assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 96);
            assert!(result.posterior.as_ref().unwrap().draws.schema.quantities.len() >= 7);
            if suite != RefuteSuite::None {
                assert!(!result.predictive_checks.is_empty(), "{coordinate}: missing PPC");
            }
            if suite == RefuteSuite::Full {
                assert!(
                    result.posterior.as_ref().unwrap().prior_sensitivity.is_some(),
                    "{coordinate}: missing prior sensitivity"
                );
            }
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            assert_eq!(refreshed.estimate.ate.to_bits(), result.estimate.ate.to_bits());

            let artifact = prepared
                .encode_contracted_result(
                    &refreshed,
                    &format!("checked-bayesian-conditional-{label}"),
                    &ctx,
                )
                .unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_bayesian_conditional_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

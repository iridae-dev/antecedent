//! Builder-independent evidence for the Bayesian DAG mean ATE operation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, StructureSource, Study,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SlotAvailability, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;
use antecedent_prob::BayesLikelihood;

mod common;

use common::fixtures::confounded_scm;

#[test]
fn bayesian_dag_ate_retains_proof_prior_validation_and_draws_across_refresh() {
    let ctx = ExecutionContext::for_tests(271);
    let (data, graph, query) = confounded_scm(512, 271);
    for accepted in [false, true] {
        for (label, suite) in [
            ("none", RefuteSuite::None),
            ("cheap", RefuteSuite::Cheap),
            ("full", RefuteSuite::Full),
        ] {
            let config = BayesianConfig::conjugate().n_draws(96).prior_scale(30.0);
            let base = Study::tabular(data.clone())
                .query(query.clone())
                .estimator(EstimatorId::BayesianGcomp)
                .inference(InferenceMode::Bayesian(config.clone()))
                .refute(suite);
            let builder = if accepted {
                base.graph(AcceptedGraph::from(graph.clone()))
            } else {
                base.graph(graph.clone())
            };
            let study = builder.clone().build().unwrap();
            let one_shot = if !accepted && suite == RefuteSuite::None {
                Some(study.run(&ctx).unwrap())
            } else {
                None
            };
            let mut prepared = study.prepare(&ctx).unwrap();
            drop(builder);
            drop(study);

            let coordinate = format!(
                "AverageEffect:Dag:{}:Bayesian:{label}",
                if accepted { "accepted" } else { "explicit" }
            );
            assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed));
            assert_eq!(
                prepared.structure_source(),
                if accepted { StructureSource::Accepted } else { StructureSource::Explicit }
            );
            let contract = prepared.contract().unwrap();
            match &contract.reasoning.support {
                SlotAvailability::Available(slot) => {
                    assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate.as_str()));
                }
                other => panic!("{coordinate}: support was not available: {other:?}"),
            }
            let plan = prepared
                .checked_bayesian_gcomp_operation()
                .expect("retained Bayesian DAG operation");
            assert_eq!(prepared.checked_bayesian_gcomp_validation(), Some(suite));
            assert_eq!(plan.query(), &query);
            assert_eq!(plan.identification().average_effect(), Some(&query));
            assert!(
                plan.identification()
                    .estimands
                    .iter()
                    .any(|candidate| { candidate.functional == plan.estimand().functional })
            );
            assert_eq!(plan.inference(), &InferenceMode::Bayesian(config));
            assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("bayesian.gcomp"));

            let result = prepared.estimate(&data, &ctx).unwrap();
            if let Some(one_shot) = one_shot {
                assert!(
                    one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                    "one-shot DAG Bayesian path must execute the retained proof"
                );
                assert_eq!(one_shot.estimate.ate.to_bits(), result.estimate.ate.to_bits());
            }
            // The independent fixture is y = 2*t + z + noise under a complete
            // back-door graph. The posterior must estimate the known effect.
            assert!(
                (result.estimate.ate - 2.0).abs() < 0.2,
                "{coordinate}: {}",
                result.estimate.ate
            );
            assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 96);
            assert_eq!(
                result.posterior.as_ref().unwrap().diagnostics.backend_id.as_ref(),
                "conjugate_gaussian"
            );
            assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
            if suite != RefuteSuite::None {
                assert!(!result.predictive_checks.is_empty(), "{coordinate}: missing PPC");
            }
            if suite == RefuteSuite::Full {
                assert!(
                    result.posterior.as_ref().unwrap().prior_sensitivity.is_some(),
                    "{coordinate}: missing prior-sensitivity grid"
                );
            }
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            assert_eq!(refreshed.estimate.ate.to_bits(), result.estimate.ate.to_bits());
            assert_eq!(prepared.checked_bayesian_gcomp_operation().unwrap().query(), &query);

            let artifact = prepared
                .encode_contracted_result(
                    &refreshed,
                    &format!("checked-bayesian-dag-{label}"),
                    &ctx,
                )
                .unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_bayesian_dag_ate_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

fn non_gaussian_data() -> (TabularData, Dag, AverageEffectQuery) {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    for i in 0..600usize {
        let z = (i % 10) as f64 / 10.0;
        let t = f64::from((i * 13 % 17) < 9);
        treatment.push(t);
        confounder.push(z);
        outcome.push(f64::from((i * 19 % 29) as f64 / 29.0 < 1.0 / (1.0 + (0.3 - t - z).exp())));
    }
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, graph, query)
}

#[test]
fn bayesian_dag_ate_keeps_non_gaussian_likelihood_on_outcome_scale() {
    let (data, graph, query) = non_gaussian_data();
    let ctx = ExecutionContext::for_tests(91);
    for likelihood in [BayesLikelihood::BernoulliLogit, BayesLikelihood::BernoulliProbit] {
        let config = BayesianConfig::laplace().likelihood(likelihood).n_draws(64);
        let builder = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(query.clone())
            .inference(InferenceMode::Bayesian(config.clone()))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let prepared = builder.prepare(&ctx).unwrap();
        drop(builder);
        let plan = prepared.checked_bayesian_gcomp_operation().unwrap();
        assert_eq!(plan.inference(), &InferenceMode::Bayesian(config));
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.estimate.ate.is_finite());
        assert!(result.estimate.ate > 0.1 && result.estimate.ate < 0.5);
        assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 64);
        assert_eq!(result.posterior.as_ref().unwrap().diagnostics.backend_id.as_ref(), "laplace");
    }
}

#[test]
fn bayesian_dag_ate_poisson_mean_contrast_uses_the_count_scale() {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    for row in 0..800usize {
        let t = f64::from(row % 4 >= 2);
        treatment.push(t);
        // Exact arm means are 1 and 2. The Poisson log-link g-computation
        // target is E[Y(1)] - E[Y(0)] = 1, not the log-rate contrast.
        outcome.push(if t == 0.0 { (row % 2) as f64 * 2.0 } else { 1.0 + (row % 2) as f64 * 2.0 });
    }
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let config = BayesianConfig::laplace().likelihood(BayesLikelihood::PoissonLog).n_draws(96);
    let builder = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Bayesian(config.clone()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ExecutionContext::for_tests(309)).unwrap();
    drop(builder);
    let plan = prepared.checked_bayesian_gcomp_operation().unwrap();
    assert_eq!(plan.inference(), &InferenceMode::Bayesian(config));
    let result = prepared.estimate(&data, &ExecutionContext::for_tests(309)).unwrap();
    assert!((result.estimate.ate - 1.0).abs() < 0.1, "{}", result.estimate.ate);
    assert_eq!(result.posterior.as_ref().unwrap().diagnostics.backend_id.as_ref(), "laplace");
}

#[test]
fn bayesian_dag_ate_hmc_draw_floor_and_population_refusal_keep_scope_visible() {
    let (data, graph, query) = confounded_scm(256, 11);
    let ctx = ExecutionContext::for_tests(3);
    let config = BayesianConfig::hmc().n_draws(50);
    let builder = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(config.clone()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    assert_eq!(
        prepared.checked_bayesian_gcomp_operation().unwrap().inference(),
        &InferenceMode::Bayesian(config)
    );
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.posterior.as_ref().unwrap().diagnostics.backend_id.as_ref(), "hmc");
    assert!(result.posterior.as_ref().unwrap().draws.n_draws >= 3_000);
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.bayesian.hmc_draw_floor")
    );

    let mut other_population = query;
    other_population.target_population = antecedent_core::TargetPopulation::Treated;
    let maybe_study = Study::tabular(data.clone())
        .graph(graph)
        .query(other_population)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build();
    let error = match maybe_study {
        Ok(study) => {
            study.prepare(&ctx).and_then(|prepared| prepared.estimate(&data, &ctx)).unwrap_err()
        }
        Err(error) => error,
    };
    assert!(error.to_string().contains("population_not_estimable"), "{error}");
}

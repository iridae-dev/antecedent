//! Route-native diagnostics reach `CausalError::route_diagnostics` from the code paths
//! that detect them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    BayesianConfig, CausalError, EstimatorId, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::diagnostic::{RouteKind, RouteNativeDiagnostic};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalRng, ConditionalEffectQuery, DeclaredSamplingDesign,
    ExecutionContext, SamplingDesignDeclaration, VariableId,
};
use antecedent_data::{ResamplingPlan, TabularData, fill_declared_plan};
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn a_dependent_series_resampled_by_an_iid_plan_is_a_dependent_resampling_check() {
    let declaration =
        SamplingDesignDeclaration::new(DeclaredSamplingDesign::DependentTimeSeries, true);
    let mut out = Vec::new();
    let error = fill_declared_plan(
        declaration,
        ResamplingPlan::IidBootstrap,
        8,
        None,
        &mut CausalRng::from_seed(5),
        &mut out,
    )
    .expect_err("an iid plan drops the series dependence");
    let records = CausalError::from(error).route_diagnostics();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0].kind(), Some(RouteKind::DependentResampling));
    assert!(records[0].failed());
    assert!(records[0].unevaluable_mass().is_unreported());
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// `t, y, z` with an effect that depends on the modifier `z`; `z -> t`, `z -> y`, `t -> y`.
fn modified_effect_data() -> (TabularData, Dag) {
    let n = 400;
    let (mut treatment, mut outcome, mut modifier) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let z = if i % 2 == 0 { -1.0 } else { 1.0 };
        let t = f64::from((i / 2) % 2 == 1);
        let noise = ((i * 37 % 101) as f64 - 50.0) / 500.0;
        modifier.push(z);
        treatment.push(t);
        outcome.push(2.0 + 0.5 * t + 0.4 * z + 0.75 * t * z + noise);
    }
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", modifier.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn bayesian_result(estimator: EstimatorId, query: impl Into<CausalQuery>) -> StudyResult {
    let (data, graph) = modified_effect_data();
    let ctx = ExecutionContext::for_tests(31);
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .estimator(estimator)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(128).prior_scale(20.0),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
}

/// The result states the checks its posterior passed, each carrying the mass the posterior
/// left unevaluated, and none of them is a failure.
fn assert_posterior_checks_passed(result: &StudyResult) {
    let posterior = result.posterior.as_ref().expect("a Bayesian result carries its posterior");
    let records = result.route_diagnostics();
    let convergence: Vec<&RouteNativeDiagnostic> =
        records.iter().filter(|record| record.kind() == Some(RouteKind::Convergence)).collect();
    assert_eq!(convergence.len(), 1, "{records:?}");
    assert!(!convergence[0].failed(), "{records:?}");
    assert_eq!(
        convergence[0].unevaluable_mass().get(),
        Some(posterior.unevaluable_mass),
        "the passed check carries the measured unevaluable mass"
    );
    assert!(records.iter().all(|record| !record.failed()), "{records:?}");
}

#[test]
fn a_bayesian_basis_result_states_its_passed_convergence_check_with_its_mass() {
    let result = bayesian_result(
        EstimatorId::BayesianBasisGcomp,
        AverageEffectQuery::binary_ate(v(0), v(1)),
    );
    assert_posterior_checks_passed(&result);
}

#[test]
fn a_bayesian_conditional_result_states_its_passed_convergence_check_with_its_mass() {
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(v(0), v(1)).with_effect_modifiers([v(2)]),
    )
    .unwrap();
    let result =
        bayesian_result(EstimatorId::BayesianConditional, CausalQuery::ConditionalEffect(query));
    assert_posterior_checks_passed(&result);
}

#[test]
fn a_bayesian_robust_result_states_its_passed_convergence_check_with_its_mass() {
    let result =
        bayesian_result(EstimatorId::BayesianRobustAte, AverageEffectQuery::binary_ate(v(0), v(1)));
    assert_posterior_checks_passed(&result);
}

#[test]
fn a_frequentist_result_carries_no_posterior_check_record() {
    let (data, graph) = modified_effect_data();
    let result = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(31))
        .unwrap();
    assert!(result.route_diagnostics().is_empty(), "{:?}", result.route_diagnostics());
}

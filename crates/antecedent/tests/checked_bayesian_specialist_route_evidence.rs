//! Builder independent execution evidence for the Bayesian IV and sharp-RD specialists.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

struct Fixture {
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    truth: f64,
    tolerance: f64,
    identifier: IdentifierId,
    estimator: EstimatorId,
    bandwidth: Option<f64>,
    dependency: &'static str,
}

fn columns(t: &[f64], y: &[f64], z: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("t", t), ("y", y), ("z", z)]).unwrap()
}

/// Joint linear IV design with a binary instrument and a shared disturbance.
fn iv_fixture() -> Fixture {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_iv/expected.json"
    ))
    .unwrap();
    let n = 1200_u16;
    let z: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
    let u: Vec<f64> = (0..n).map(|i| (f64::from(i % 17) - 8.0) / 5.0).collect();
    let t: Vec<f64> = z.iter().zip(&u).map(|(z, u)| 0.6 * z + u).collect();
    let y: Vec<f64> = t.iter().zip(&u).map(|(t, u)| 2.0 * t + u).collect();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Fixture {
        data: columns(&t, &y, &z),
        graph,
        query: AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        ),
        truth: expected["true_structural_effect"].as_f64().unwrap(),
        tolerance: 0.2,
        identifier: IdentifierId::Iv,
        estimator: EstimatorId::BayesianIvJointLinear,
        bandwidth: None,
        dependency: "dependencies.checked_bayesian_iv_operation",
    }
}

/// Sharp local-linear design with a known jump at the cutoff.
fn rd_fixture() -> Fixture {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_rd/expected.json"
    ))
    .unwrap();
    let n = 400_u16;
    let r: Vec<f64> = (0..n).map(|i| (f64::from(i) - 200.0) / 100.0).collect();
    let t: Vec<f64> = r.iter().map(|r| f64::from(*r >= 0.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let i = usize::from(i);
            1.0 + 2.5 * t[i]
                + 0.8 * r[i]
                + 1.2 * t[i] * r[i]
                + (f64::from(u8::try_from(i % 5).unwrap()) - 2.0) * 0.01
        })
        .collect();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    Fixture {
        data: columns(&t, &y, &r),
        graph,
        query: AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        truth: expected["true_jump"].as_f64().unwrap(),
        tolerance: 0.08,
        identifier: IdentifierId::RdSharp,
        estimator: EstimatorId::BayesianRdLocalLinear,
        bandwidth: Some(expected["fixed_bandwidth"].as_f64().unwrap()),
        dependency: "dependencies.checked_bayesian_rd_operation",
    }
}

fn widened(data: &TabularData) -> TabularData {
    let t = data.float64_values(VariableId::from_raw(0)).unwrap();
    let y = data.float64_values(VariableId::from_raw(1)).unwrap();
    let z = data.float64_values(VariableId::from_raw(2)).unwrap();
    TabularData::from_f64_columns([
        ("t", t.as_ref()),
        ("y", y.as_ref()),
        ("z", z.as_ref()),
        ("extra", z.as_ref()),
    ])
    .unwrap()
}

#[test]
fn bayesian_specialists_execute_from_retained_plan_after_builder_drop() {
    run_on_large_stack(bayesian_specialists_execute_from_retained_plan_after_builder_drop_body);
}

fn bayesian_specialists_execute_from_retained_plan_after_builder_drop_body() {
    for fixture in [iv_fixture(), rd_fixture()] {
        let ctx = ExecutionContext::for_tests(81);
        let mut builder = Study::tabular(fixture.data.clone())
            .graph(fixture.graph.clone())
            .query(fixture.query.clone())
            .identifier(fixture.identifier)
            .estimator(fixture.estimator)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)))
            .refute(RefuteSuite::None);
        if let Some(bandwidth) = fixture.bandwidth {
            builder = builder.rd_config(VariableId::from_raw(2), 0.0, bandwidth);
        }
        let builder = builder.build().unwrap();
        let one_shot = builder.run(&ctx).unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);

        let plan =
            prepared.checked_bayesian_specialist_info().expect("retained Bayesian specialist plan");
        // The sharp design binds its target to the cutoff population at build.
        assert_eq!(CausalQuery::AverageEffect(plan.query.clone()), *prepared.query());
        assert_eq!(plan.query.treatment, fixture.query.treatment);
        assert_eq!(plan.query.outcome, fixture.query.outcome);
        assert_eq!(plan.identifier, fixture.identifier);
        assert_eq!(plan.estimator, fixture.estimator);
        assert_eq!(plan.n_draws, 512);
        assert_eq!(plan.rd.map(|rd| rd.bandwidth), fixture.bandwidth);
        assert_eq!(prepared.plan().record.plan_id, plan.plan_id);

        let result = prepared.estimate(&fixture.data, &ctx).unwrap();
        assert!(
            (result.estimate.ate - fixture.truth).abs() < fixture.tolerance,
            "{:?} effect {}",
            fixture.estimator,
            result.estimate.ate
        );
        assert!(result.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached"));
        assert_eq!(result.logical_plan.estimator.as_deref(), Some(fixture.estimator.as_str()));
        let posterior = result.posterior.as_ref().expect("specialist posterior");
        assert_eq!(posterior.draws.n_draws, 512);
        assert!(
            posterior.summaries.q025[0] < fixture.truth
                && fixture.truth < posterior.summaries.q975[0]
        );
        // The one-shot facade executes the same retained model and seed.
        assert!((one_shot.estimate.ate - result.estimate.ate).abs() < 1e-12);
        assert_eq!(
            one_shot.posterior.as_ref().unwrap().draws.values.as_ref(),
            posterior.draws.values.as_ref()
        );

        let refreshed = prepared.refresh(fixture.data.clone(), &ctx).unwrap();
        assert!((refreshed.estimate.ate - result.estimate.ate).abs() < 1e-12);
        assert!(prepared.has_checked_bayesian_specialist_operation());

        let bytes = prepared
            .encode_contracted_result(&refreshed, "checked-bayesian-specialist", &ctx)
            .unwrap();
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|dependency| dependency.as_ref() == fixture.dependency),
            "{:?}",
            consumed.acceptance.unresolved
        );
        assert!(!consumed.acceptance.accepts_as_verified_program());

        assert!(prepared.refresh(widened(&fixture.data), &ctx).is_err());
    }
}

/// Debug builds of the shared prepared dispatcher carry frames close to the
/// default test-thread stack, so the evidence body runs on its own thread.
fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-route-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

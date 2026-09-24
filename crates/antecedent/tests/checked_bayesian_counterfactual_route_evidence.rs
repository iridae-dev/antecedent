//! Builder-independent checked Bayesian GCM counterfactual lifecycle.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, IntervalMethod, Intervention, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn fixture(offset: f64) -> (TabularData, Dag) {
    let x = (0..120).map(|i| f64::from(i) / 30.0 - 2.0).collect::<Vec<_>>();
    // Independent structural truth: Y = 1 + 2X + ε, with ε fixed per row.
    // Therefore every row's cross-world ITE is exactly 2.
    let y = x
        .iter()
        .enumerate()
        .map(|(i, x)| 1.0 + 2.0 * x + 0.025 * (i as f64 * 0.71).sin() + offset)
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, graph)
}

fn query() -> CausalQuery {
    CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        )
        .with_control_level(0.0),
    )
}

#[test]
fn bayesian_counterfactual_accepted_and_explicit_lifecycle_exposes_dependency_boundary() {
    let ctx = ExecutionContext::for_tests(611);
    for accepted in [false, true] {
        let (data, graph) = fixture(0.0);
        let builder = if accepted {
            Study::tabular(data.clone()).graph(AcceptedGraph::from(graph.clone()))
        } else {
            Study::tabular(data.clone()).graph(graph.clone())
        }
        .query(query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);
        let operation = prepared
            .checked_counterfactual_operation()
            .expect("prepared Bayesian route retains its typed operation");
        assert_eq!(operation.query().outcomes.as_ref(), &[VariableId::from_raw(1)]);
        assert_eq!(operation.query().interventions.len(), 1);
        assert_eq!(operation.graph().edges().count(), 1);

        let result = prepared.estimate(&data, &ctx).unwrap();
        let counterfactual = result.counterfactual.as_ref().unwrap();
        assert!(
            (counterfactual.mean_ite - 2.0).abs() < 0.12,
            "accepted={accepted}: {counterfactual:?}"
        );
        assert_eq!(counterfactual.unit_effects.len(), 120);
        let unit_intervals = counterfactual.unit_effect_intervals.as_ref().unwrap();
        assert_eq!(unit_intervals.method, "unit_posterior_quantile");
        assert_eq!(unit_intervals.lower.len(), 120);
        assert_eq!(unit_intervals.upper.len(), 120);
        let interval = result.primary_interval_binding(true);
        assert_eq!(interval.method, IntervalMethod::PosteriorQuantile);
        assert_eq!(interval.posterior_draws, Some(48));
        assert!(result.diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_ref() == "gcm.counterfactual.bayesian"
                && diagnostic.message.contains("Dirichlet row-weight posterior")
                && diagnostic.message.contains("mean_ite interval")
        }));

        let (refreshed_data, _) = fixture(0.4);
        let refreshed = prepared.refresh(refreshed_data, &ctx).unwrap();
        assert!((refreshed.counterfactual.as_ref().unwrap().mean_ite - 2.0).abs() < 0.12);
        assert!(prepared.checked_counterfactual_operation().is_some());

        let artifact = prepared
            .encode_contracted_result(&refreshed, "checked-bayesian-counterfactual", &ctx)
            .unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.fitted_counterfactual_mechanisms"
            })
        );
    }
}

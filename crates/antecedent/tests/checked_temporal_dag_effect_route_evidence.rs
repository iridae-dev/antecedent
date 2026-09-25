//! Builder-independent evidence for fixed-graph frequentist temporal contrasts.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series(scale: f64) -> TimeSeriesData {
    let n = 320usize;
    let t = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            scale
                * (1.0
                    + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                    + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j]))
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let t1 = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, t, Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

fn effect(multi_step: bool) -> TemporalEffectQuery {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    if multi_step {
        TemporalEffectQuery::sustained(t, y, 0, 1.0).with_policy(TemporalPolicy::sustained(-2, -1))
    } else {
        TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::pulse(-1))
    }
}

#[test]
fn fixed_temporal_contrasts_execute_from_retained_proof_and_procedure() {
    let data = series(1.0);
    let graph = graph();
    let ctx = ExecutionContext::for_tests(415);
    for multi_step in [false, true] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let query = effect(multi_step);
                let structure = if accepted {
                    AcceptedGraph::temporal_dag(graph.clone())
                } else {
                    graph.clone().into()
                };
                let builder = Study::series(data.clone())
                    .graph(structure)
                    .query(CausalQuery::TemporalEffect(query.clone()))
                    .inference(InferenceMode::Frequentist)
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&ctx).unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);

                let plan = prepared.checked_temporal_dag_effect_info().expect("checked plan");
                assert_eq!(plan.query, query);
                assert_eq!(plan.identifier.as_str(), "temporal.backdoor.unfolded");
                assert_eq!(
                    plan.estimator.as_str(),
                    if multi_step {
                        "temporal.sequential.gcomp"
                    } else {
                        "temporal.linear.adjustment"
                    },
                );
                assert_eq!(plan.validation, suite);
                assert_eq!(plan.bootstrap_replicates, 0);

                let result = prepared.estimate_series(&data, &ctx).unwrap();
                let expected = if multi_step { 5.0 } else { 2.0 };
                assert!(
                    (result.estimate.ate - expected).abs() < 0.08,
                    "multi_step={multi_step}, accepted={accepted}, suite={suite:?}: {}",
                    result.estimate.ate
                );
                assert!((one_shot.estimate.ate - expected).abs() < 0.08);
                assert!(
                    one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );

                let scaled = prepared.refresh_series(series(1.2), &ctx).unwrap();
                assert!((scaled.estimate.ate - 1.2 * expected).abs() < 0.1);
                assert_eq!(prepared.checked_temporal_dag_effect_info().unwrap().query, plan.query);

                let artifact = prepared
                    .encode_contracted_result(&scaled, "checked-temporal-effect", &ctx)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(
                    consumed.acceptance.unresolved.iter().any(|reason| {
                        reason.as_ref() == "dependencies.checked_temporal_dag_effect_operation"
                    }),
                    "unresolved={:?} contract={:?}",
                    consumed.acceptance.unresolved,
                    consumed.contract.as_ref().map(|c| (&c.target.query, &c.estimator))
                );
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

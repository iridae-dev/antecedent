//! Public lifecycle evidence for three-modifier Bayesian DAG CATE execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, StructureSource, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, SlotAvailability,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

fn fixture() -> (TabularData, Dag, ConditionalEffectQuery) {
    let n = 1365usize;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    for row in 0..n {
        let zi = (row % 7) as f64 - 3.0;
        let wi = ((row / 7) % 5) as f64 - 2.0;
        let vi = ((row / 35) % 3) as f64 - 1.0;
        let ui = ((row * 11 % 17) as f64 - 8.0) / 8.0;
        let ti = f64::from((row * 17 + row / 7 * 3) % 13 < 6);
        z.push(zi);
        w.push(wi);
        v.push(vi);
        u.push(ui);
        t.push(ti);
        y.push(
            1.0 + 0.5 * ui
                + 2.0 * ti
                + 0.2 * ti * zi
                + 0.4 * ti * wi
                + 0.6 * ti * vi
                + 0.01 * (row as f64 * 0.37).sin(),
        );
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("w", w.as_slice()),
        ("v", v.as_slice()),
        ("u", u.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(6);
    for (from, to) in [(5, 0), (5, 1), (0, 1), (2, 1), (3, 1), (4, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([
                VariableId::from_raw(2),
                VariableId::from_raw(3),
                VariableId::from_raw(4),
            ]),
    )
    .unwrap();
    (data, graph, query)
}

#[test]
fn three_modifier_bayesian_cate_is_sealed_refreshable_and_refuses_unavailable_artifact_replay() {
    let (data, graph, query) = fixture();
    let context = ExecutionContext::for_tests(4221);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let config = BayesianConfig::conjugate().n_draws(96).prior_scale(25.0);
            let base = Study::tabular(data.clone())
                .query(CausalQuery::ConditionalEffect(query.clone()))
                .estimator(EstimatorId::BayesianConditional)
                .inference(InferenceMode::Bayesian(config.clone()))
                .refute(suite);
            let builder = if accepted {
                base.graph(AcceptedGraph::from(graph.clone())).build().unwrap()
            } else {
                base.graph(graph.clone()).build().unwrap()
            };
            let mut prepared = builder.prepare(&context).unwrap();
            drop(builder);

            let coordinate = format!(
                "ConditionalEffect:Dag:{}:Bayesian:{}:three_modifiers",
                if accepted { "accepted" } else { "explicit" },
                match suite {
                    RefuteSuite::None => "none",
                    RefuteSuite::Cheap => "cheap",
                    RefuteSuite::Full => "full",
                    RefuteSuite::PlaceboAndRcc => unreachable!("unsupported CATE suite"),
                }
            );
            assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed));
            assert_eq!(
                prepared.structure_source(),
                if accepted { StructureSource::Accepted } else { StructureSource::Explicit }
            );
            match &prepared.contract().unwrap().reasoning.support {
                SlotAvailability::Available(slot) => {
                    let expected = format!(
                        "ConditionalEffect:Dag:{}:Bayesian:{}",
                        if accepted { "accepted" } else { "explicit" },
                        match suite {
                            RefuteSuite::None => "none",
                            RefuteSuite::Cheap => "cheap",
                            RefuteSuite::Full => "full",
                            RefuteSuite::PlaceboAndRcc => unreachable!("unsupported CATE suite"),
                        }
                    );
                    assert_eq!(slot.matrix_coordinate.as_deref(), Some(expected.as_str()));
                }
                other => panic!("{coordinate}: support not available: {other:?}"),
            }
            let plan = prepared
                .checked_bayesian_conditional_operation()
                .expect("prepared route retains its full conditional target and prior");
            assert_eq!(plan.query(), &query);
            assert_eq!(plan.inference(), &InferenceMode::Bayesian(config));
            assert_eq!(plan.validation(), suite);
            assert_eq!(
                plan.modifier_roles(),
                &[VariableId::from_raw(2), VariableId::from_raw(3), VariableId::from_raw(4),]
            );
            assert!(plan.estimand().adjustment_set.contains(&VariableId::from_raw(5)));

            let result = prepared.estimate(&data, &context).unwrap();
            // Independent SCM truth is 2.0 because all three modifiers have
            // zero population mean and their treatment interactions are centered.
            assert!(
                (result.estimate.ate - 2.0).abs() < 0.3,
                "{coordinate}: {}",
                result.estimate.ate
            );
            assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 96);
            if suite != RefuteSuite::None {
                assert!(!result.predictive_checks.is_empty());
            }
            if suite == RefuteSuite::Full {
                assert!(result.posterior.as_ref().unwrap().prior_sensitivity.is_some());
            }

            let columns = ["t", "y", "z", "w", "v", "u"]
                .map(|name| data.schema().id_of(name).unwrap())
                .map(|id| data.float64_values(id).unwrap());
            let shifted_y = columns[1].iter().map(|value| value + 0.25).collect::<Vec<_>>();
            let shifted = TabularData::from_f64_columns([
                ("t", columns[0].as_slice()),
                ("y", shifted_y.as_slice()),
                ("z", columns[2].as_slice()),
                ("w", columns[3].as_slice()),
                ("v", columns[4].as_slice()),
                ("u", columns[5].as_slice()),
            ])
            .unwrap();
            let refreshed = prepared.refresh(shifted, &context).unwrap();
            assert_eq!(prepared.checked_bayesian_conditional_operation().unwrap().query(), &query);
            assert!((refreshed.estimate.ate - result.estimate.ate).abs() < 0.3);

            let artifact =
                prepared.encode_contracted_result(&refreshed, &coordinate, &context).unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref() == "dependencies.checked_bayesian_conditional_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

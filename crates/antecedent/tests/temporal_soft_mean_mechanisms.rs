//! Known-truth mean-dynamics pins: clipping applies to propagated means, not draws.
#![allow(clippy::cast_precision_loss)]
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionSequence, Lag, MechanismOverride,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue,
    SequencedIntervention, TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::{SamplingRegularity, TabularData, TimeIndex, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};
use std::sync::Arc;

fn frozen() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_soft_mean_mechanisms/expected.json"
    ))
    .unwrap()
}

fn fixture() -> (TimeSeriesData, TemporalDag) {
    let n = usize::try_from(frozen()["generation"]["n"].as_u64().unwrap()).unwrap();
    let t = (0..n).map(|i| 1.0 + (i as f64 * 1.719).sin()).collect::<Vec<_>>();
    let z = (0..n).map(|i| 2.0 + (i as f64 * 0.813).cos()).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * t[i.saturating_sub(1)]
                + 3.0 * t[i.saturating_sub(2)]
                + 4.0 * z[i.saturating_sub(1)]
                + 0.05 * (i as f64 * 0.419).sin()
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("z", z.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let series = TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { length: n, regularity: SamplingRegularity::Regular { interval_ns: 1 } },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    for (variable, lag) in [(0, 1), (0, 2), (1, 1)] {
        let node =
            ensure_lagged(&mut graph, VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap();
        graph.insert_directed(node, y0).unwrap();
    }
    (series, graph)
}

fn soft(variable: u32, mechanism: MechanismOverride) -> Intervention {
    Intervention::soft(VariableId::from_raw(variable), mechanism)
}

fn sequence(steps: Vec<Intervention>) -> Intervention {
    Intervention::Sequence(InterventionSequence::new(
        steps
            .into_iter()
            .map(|intervention| SequencedIntervention {
                intervention,
                temporal: TemporalPolicy::pulse(0),
            })
            .collect::<Vec<_>>(),
    ))
}

fn query(intervention: Intervention) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([intervention]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap())
}

#[test]
fn extra_soft_mean_families_pin_single_joint_and_multistep_in_both_inferences() {
    let (data, graph) = fixture();
    let cases = [
        soft(0, MechanismOverride::multiplicative(2.0)),
        soft(0, MechanismOverride::truncated_shift(3.0, 0.0, 2.0)),
        sequence(vec![
            soft(0, MechanismOverride::multiplicative(2.0)),
            soft(1, MechanismOverride::truncated_shift(2.0, 0.0, 3.0)),
        ]),
        sequence(vec![
            soft(0, MechanismOverride::multiplicative(2.0)),
            soft(0, MechanismOverride::truncated_shift(3.0, 0.0, 2.0)),
        ]),
    ];
    for (index, intervention) in cases.into_iter().enumerate() {
        let frozen = frozen();
        let expected = frozen["cases"][index]["mean"].as_f64().unwrap();
        let tolerance = frozen["atol"].as_f64().unwrap();
        let query = query(intervention);
        let original = CausalQuery::Response(query.clone());
        let wire = antecedent_io::causal_query_to_wire(&original).unwrap();
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded =
            antecedent_io::causal_query_from_wire(&serde_json::from_slice(&bytes).unwrap())
                .unwrap();
        assert_eq!(decoded, original, "Soft family and packed parameters must roundtrip");
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128).prior_scale(1000.0)),
        ] {
            let study = Study::series(data.clone())
                .graph(graph.clone())
                .query(original.clone())
                .inference(inference)
                .refute(RefuteSuite::None)
                .bootstrap_replicates(4)
                .build()
                .unwrap();
            let context = ExecutionContext::for_tests(17);
            for result in [
                study.run(&context).unwrap(),
                study.prepare(&context).unwrap().estimate_series(&data, &context).unwrap(),
            ] {
                let response = result.response.unwrap();
                let ResponseIdentification::PointIdentified(ResponseValue::Surface {
                    mean, ..
                }) = response.estimate
                else {
                    panic!("expected point surface")
                };
                assert!(
                    (mean[0] - expected).abs() < tolerance,
                    "got {} expected {expected}",
                    mean[0]
                );
            }
        }
    }
}

#[test]
fn extra_soft_rejects_bad_parameter_shapes_values_and_bounds() {
    for (family, parameters) in [
        ("multiplicative", vec![]),
        ("multiplicative", vec![f64::NAN]),
        ("multiplicative", vec![1.0, 2.0]),
        ("truncated_shift", vec![1.0]),
        ("truncated_shift", vec![1.0, 3.0, 2.0]),
        ("truncated_shift", vec![1.0, 0.0, f64::INFINITY]),
    ] {
        assert!(
            antecedent_estimate::plan_from_response_query(&query(soft(
                0,
                MechanismOverride::named(family, parameters)
            )))
            .is_err()
        );
    }
}

#[test]
fn bounded_mean_policy_preserves_normal_innovations_instead_of_clipping_samples() {
    let pin = frozen();
    let pin = &pin["nondegenerate_counterexample"];
    let mut rng = antecedent_core::CausalRng::from_seed(192);
    // Antithetic normal draws give an exactly centered, nondegenerate population.
    let mut x = vec![0.0];
    for _ in 0..1000 {
        let draw = antecedent_kernels::standard_normal(&mut rng);
        x.extend([draw, -draw]);
    }
    x.push(0.0);
    let y = std::iter::once(0.0).chain(x.iter().copied().take(x.len() - 1)).collect::<Vec<_>>();
    let clipped_mean = x.iter().map(|v| v.clamp(0.0, 1.0)).sum::<f64>() / x.len() as f64;
    let stochastic_truth = pin["stochastic_clipping_mean"].as_f64().unwrap();
    assert!((clipped_mean - stochastic_truth).abs() < 0.025);
    let data =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let mut graph = TemporalDag::empty();
    let x_lag = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y_now = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x_lag, y_now).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([soft(0, MechanismOverride::truncated_shift(0.0, 0.0, 1.0))]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let result = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(91))
        .unwrap();
    let response = result.response.unwrap();
    assert!(response.assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::ParametricRestriction(p)
            if p.id.as_ref() == "temporal.soft.population_mean_target"
            && p.description.contains("f_policy = f + clip(mu + delta, lower, upper) - mu")
            && p.description.contains("realized outcomes need not satisfy the bounds"))
    }));
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        response.estimate
    else {
        panic!("expected surface")
    };
    assert!((mean[0] - pin["policy_mean"].as_f64().unwrap()).abs() < 1e-10);
    assert!((mean[0] - stochastic_truth).abs() > 0.3);
}

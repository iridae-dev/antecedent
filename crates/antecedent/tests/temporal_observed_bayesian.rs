//! Consuming observed-data posterior pins; no pseudo-outcome Gaussian likelihood.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::float_cmp
)]
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, InterventionSequence,
    Lag, ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SequencedIntervention, TemporalPolicy,
    TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use std::sync::Arc;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/bayesian/temporal_observed_response/expected.json"
    ))
    .unwrap()
}
fn data(case: usize) -> (TimeSeriesData, TemporalDag, ObservationSpec, ObservationAssumption, f64) {
    let fixture = pin();
    let n = usize::try_from(fixture["rows"].as_u64().unwrap()).unwrap();
    let id = VariableId::from_raw;
    let mut rng = antecedent_core::CausalRng::from_seed(fixture["seed"].as_u64().unwrap());
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut event = vec![1.0; n];
    let sign = if case == 2 || case == 4 { -1.0 } else { 1.0 };
    for i in 0..n {
        x[i] = antecedent_kernels::standard_normal(&mut rng);
        let x1 = if i > 0 { x[i - 1] } else { 0.0 };
        let x2 = if i > 1 { x[i - 2] } else { 0.0 };
        let latent =
            2.0 + 1.5 * x1 + 0.75 * x2 + 0.5 * antecedent_kernels::standard_normal(&mut rng);
        if case == 0 {
            let p = 1.0 / (1.0 + (-0.4 * x1).exp());
            event[i] = if rng.next_f64() < p { 1.0 } else { 0.0 };
            // Deliberate arbitrary missing placeholder: never treat it as data.
            y[i] = if event[i] == 1.0 { latent } else { 12345.0 };
        } else {
            c[i] = 2.5
                + if case >= 3 { 0.6 * x1 } else { 0.0 }
                + antecedent_kernels::standard_normal(&mut rng);
            event[i] = if latent <= c[i] { 1.0 } else { 0.0 };
            y[i] = sign * latent.min(c[i]);
            c[i] *= sign;
        }
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let series = TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("c", c.as_slice()), ("r", event.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let a = ensure_lagged(&mut graph, id(0), Lag::from_raw(1)).unwrap();
    let b = ensure_lagged(&mut graph, id(0), Lag::from_raw(2)).unwrap();
    let out = ensure_lagged(&mut graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(a, out).unwrap();
    graph.insert_directed(b, out).unwrap();
    let spec = if case == 0 {
        ObservationSpec::Selected { latent: id(1), observed: id(1), indicator: id(3) }
    } else if sign > 0.0 {
        ObservationSpec::RightCensored {
            latent: id(1),
            observed: id(1),
            censoring: id(2),
            event: id(3),
        }
    } else {
        ObservationSpec::LeftCensored {
            latent: id(1),
            observed: id(1),
            censoring: id(2),
            event: id(3),
        }
    };
    let assumption = if case == 0 {
        ObservationAssumption::OutcomeIndependentGiven(Arc::from([id(0)]))
    } else {
        ObservationAssumption::IndependentGiven(if case >= 3 {
            Arc::from([id(0)])
        } else {
            Arc::from([])
        })
    };
    (series, graph, spec, assumption, mean)
}

#[test]
fn observed_temporal_bayesian_pairs_consume_real_censoring_likelihood() {
    let id = VariableId::from_raw;
    for case in 0..5 {
        let (series, graph, spec, assumption, mean) = data(case);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: id(1),
            treatment: ContinuousDomain::new(id(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        })
        .with_temporal(TemporalResponseSpec::new([1, 2], TemporalPolicy::pulse(-1), None).unwrap())
        .with_observation(spec, [assumption]);
        let study = Study::series(series.clone())
            .graph(graph)
            .query(CausalQuery::Response(query))
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4096)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(330);
        let prepared = study.prepare(&ctx).unwrap();
        let result = prepared.estimate_series(&series, &ctx).unwrap();
        let response = result.response.as_ref().unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Surface {
            mean: values, ..
        }) = &response.estimate
        else {
            panic!("surface")
        };
        let sign = if case == 2 || case == 4 { -1.0 } else { 1.0 };
        let truth = [2.0 + 0.75 * mean, 2.0 + 1.5 * mean, 3.5 + 0.75 * mean, 2.75 + 1.5 * mean];
        for (&value, expected) in values.iter().zip(truth) {
            assert!(
                (value - sign * expected).abs() < pin()["mean_atol"].as_f64().unwrap(),
                "case {case}: {value} versus {}",
                sign * expected
            );
        }
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
            panic!("posterior band")
        };
        assert!(lower.iter().zip(upper.iter()).all(|(l, u)| l.is_finite() && u > l));
        let posterior = result.posterior.as_ref().unwrap();
        assert!(posterior.diagnostics.allows_posterior());
        assert_eq!(posterior.diagnostics.backend_id.as_ref(), "temporal.observed_gaussian.gibbs");
        assert_eq!(posterior.draws.n_quantities(), 4);
        assert!(response.assumptions.entries.iter().any(|a| matches!(&a.assumption,antecedent_core::Assumption::ParametricRestriction(p) if p.id.as_ref()=="temporal.observed_gaussian_sem")));
        assert_eq!(response.horizon_identification.as_ref().unwrap().len(), 2);
    }
}

#[test]
fn observed_bayesian_sequence_changes_every_assigned_time() {
    let (series, graph, spec, assumption, _) = data(1);
    let id = VariableId::from_raw;
    let sequence = Intervention::Sequence(InterventionSequence {
        steps: Arc::from([
            SequencedIntervention {
                temporal: TemporalPolicy::pulse(-2),
                intervention: Intervention::set(id(0), antecedent_core::Value::f64(1.0)),
            },
            SequencedIntervention {
                temporal: TemporalPolicy::pulse(-1),
                intervention: Intervention::set(id(0), antecedent_core::Value::f64(2.0)),
            },
        ]),
    });
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: id(1),
        interventions: Arc::from([sequence]),
    })
    .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-2), None).unwrap())
    .with_observation(spec, [assumption]);
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4096)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(330))
        .unwrap();
    let response = result.response.unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) = response.estimate
    else {
        panic!("scalar")
    };
    assert!(
        (value - pin()["sequence_mean"].as_f64().unwrap()).abs()
            < pin()["sequence_atol"].as_f64().unwrap(),
        "{value}"
    );
    assert!(result.posterior.unwrap().diagnostics.allows_posterior());
}

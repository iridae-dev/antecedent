//! Conditional censoring response truth, including refusal of marginal substitution.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ObservationAssumption,
    ObservationSpec, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use std::sync::Arc;

#[test]
fn conditional_response_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/conditional_ipcw/response_truth.json"
    ))
    .unwrap();
    let ctx = ExecutionContext::for_tests(1300);
    let mut rng = ctx.rng.stream(17);
    let mut a = Vec::new();
    let mut z = Vec::new();
    let mut y = Vec::new();
    let mut c = Vec::new();
    let mut event = Vec::new();
    for _ in 0..6000 {
        let av = 2.0 * rng.next_f64() - 1.0;
        let zv = 2.0 * rng.next_f64() - 1.0;
        let latent = 5.0 + 2.0 * av + zv + 0.5 * (2.0 * rng.next_f64() - 1.0);
        let cv = -rng.next_f64().max(1e-12).ln() / (0.07 * (0.6 * av + 0.5 * zv).exp());
        a.push(av);
        z.push(zv);
        y.push(latent.min(cv));
        c.push(cv);
        event.push(if latent <= cv { 1.0 } else { 0.0 });
    }
    for reverse in [false, true] {
        let sign = if reverse { -1.0 } else { 1.0 };
        let observed: Vec<_> = y.iter().map(|v| sign * v).collect();
        let censoring: Vec<_> = c.iter().map(|v| sign * v).collect();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("z", z.as_slice()),
            ("y", observed.as_slice()),
            ("c", censoring.as_slice()),
            ("event", event.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(5);
        for (s, t) in [(0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
        }
        let id = VariableId::from_raw;
        let mechanism = if reverse {
            ObservationSpec::LeftCensored {
                latent: id(2),
                observed: id(2),
                censoring: id(3),
                event: id(4),
            }
        } else {
            ObservationSpec::RightCensored {
                latent: id(2),
                observed: id(2),
                censoring: id(3),
                event: id(4),
            }
        };
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: id(2),
            treatment: ContinuousDomain::new(id(0), GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
        })
        .with_observation(
            mechanism,
            [ObservationAssumption::IndependentGiven(Arc::from([id(0), id(1)]))],
        );
        let run = |query| {
            let study = Study::tabular(data.clone())
                .graph(graph.clone())
                .query(CausalQuery::Response(query))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap().response.unwrap()
        };
        let response = run(query.clone());
        assert!(matches!(response.uncertainty, ResponseUncertainty::None));
        let ResponseIdentification::PointIdentified(ResponseValue::Surface {
            mean: values, ..
        }) = &response.estimate
        else {
            panic!("curve")
        };
        let truth: Vec<_> =
            pin["values"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() * sign).collect();
        let error = values.iter().zip(&truth).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(
            error < pin["atol"].as_f64().unwrap(),
            "conditional {values:?} truth {truth:?}, max error={error}"
        );
        let mut marginal = query;
        marginal.observation_assumptions =
            Arc::from([ObservationAssumption::IndependentGiven(Arc::from([]))]);
        let marginal = run(marginal);
        let ResponseIdentification::PointIdentified(ResponseValue::Surface {
            mean: values, ..
        }) = &marginal.estimate
        else {
            panic!("curve")
        };
        let marginal_error =
            values.iter().zip(&truth).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(
            marginal_error > error + 0.1,
            "fixture must expose marginal substitution: conditional={error}, marginal={marginal_error}"
        );
    }
}

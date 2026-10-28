//! Absolute known-truth check for the temporal graph-posterior response routes.
//!
//! The existing staged tests compare the frozen-weight mixture with the explicit identified-atom
//! run, which is a comparison of two product paths. Here the expected numbers come from the
//! data-generating equation `defect_t = 0.9 * pressure_{t-1}` (no intercept, no noise): a pulse of
//! dose `d` one step back moves the outcome by `0.9 d`, so the mean curve over doses `[0, 1]` is
//! `[0, 0.9]` and the intervention response at dose 1 is `0.9`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::discovery::{GraphPosterior, SamplingSummary};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
    Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_prob::InferenceDiagnostics;

const SLOPE: f64 = 0.9;

fn series(n: usize) -> TimeSeriesData {
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    let mut state = 42_u64;
    for t in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        pressure[t] = ((state >> 33) as f64 / (1_u64 << 31) as f64) * 2.0 - 1.0;
        if t > 0 {
            defect[t] = SLOPE * pressure[t - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("pressure", pressure.as_slice()), ("defect", defect.as_slice())],
        1,
    )
    .unwrap()
}

/// Atom 0 (weight 0.7): `pressure_{t-1} -> defect_t`. Atom 1 (weight 0.3) adds the autoregressive
/// pressure edge, which the temporal back-door identifier cannot certify, so it is unidentified mass.
fn posterior() -> GraphPosterior {
    let weights = vec![0.7, 0.3];
    GraphPosterior::from_atoms(
        2,
        weights.clone(),
        vec![0, 0],
        vec![0.0; 4],
        vec![0.0; 4],
        SamplingSummary {
            ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("claims_03_dbn_posterior_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.3, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![2, 3])
    .unwrap()
    .with_algorithm("claims_03_dbn_posterior_truth")
}

fn points(result: &antecedent::StudyResult) -> Vec<f64> {
    let mixture = result.structural_response.as_ref().expect("structural mixture");
    assert!((mixture.unidentified_mass - 0.3).abs() < 1e-12);
    let value = mixture.conditional_on_identified.as_ref().expect("conditional summary");
    match value {
        ResponseValue::Surface { mean, .. } => mean.to_vec(),
        ResponseValue::Scalar(v) => vec![*v],
        other => panic!("unexpected {other:?}"),
    }
}

fn run(query: ResponseQuery, inference: InferenceMode) -> antecedent::StudyResult {
    let result = Study::series(series(400))
        .graph_posterior(posterior())
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();
    assert!(!matches!(
        result.response().unwrap().estimate,
        ResponseIdentification::Unidentified { .. }
    ));
    result
}

fn spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap()
}

#[test]
fn claims_03_dbn_posterior_response_equals_the_generating_equation() {
    let pressure = VariableId::from_raw(0);
    let defect = VariableId::from_raw(1);
    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: defect,
        treatment: ContinuousDomain::new(pressure, GridSpec::Values(Arc::from([0.0, 1.0]))),
    })
    .with_temporal(spec());
    let intervention = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: defect,
        interventions: Arc::from([Intervention::set(pressure, Value::f64(1.0))]),
    })
    .with_temporal(spec());
    for (inference, atol) in [
        (InferenceMode::Frequentist, 1e-6),
        (
            InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(64).prior_scale(1_000_000.0),
            ),
            0.02,
        ),
    ] {
        let got = points(&run(curve.clone(), inference.clone()));
        assert_eq!(got.len(), 2, "{inference:?}");
        for (value, want) in got.iter().zip([0.0, SLOPE]) {
            assert!((value - want).abs() < atol, "{inference:?} curve {value} vs {want}");
        }
        let got = points(&run(intervention.clone(), inference.clone()));
        assert!((got[0] - SLOPE).abs() < atol, "{inference:?} do(pressure=1) {}", got[0]);
    }
}

//! Builder-independent evidence for checked TemporalCpdag/TemporalPag responses
//! (mean curve and single-step intervention response, both inferences).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, GraphClass, IdentifierId, InferenceMode,
    RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseQuery, ResponseUncertainty, ResponseValue, TemporalPolicy, TemporalResponseSpec, Value,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{set_edge, temporal_cpdag_from_dbn_masks, temporal_pag_from_dbn_masks};

/// The law of `conformance/estimate/temporal_class_posterior_response_truth`
/// (`y_i = 1.4 t_{i-1} + 0.6 z_{i-1} + w_i`) plus a constant outcome shift.
fn series(shift: f64) -> TimeSeriesData {
    let n = 48usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        let phase = f64::from(u32::try_from(i).unwrap()) * 0.37;
        z[i] = phase.sin();
        t[i] = f64::from(u8::from(phase.cos() > 0.0));
        if i > 0 {
            y[i] = shift + 1.4 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (phase * 1.7).sin();
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

fn variables() -> [VariableId; 3] {
    [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)]
}

/// `t@1 -> y`, `z@1 -> y` with no contemporaneous structure.
fn cpdag() -> AcceptedGraph {
    let lag = lag_bit(3, 0, 1) | lag_bit(3, 2, 1);
    AcceptedGraph::from(temporal_cpdag_from_dbn_masks(0, lag, 3, 1, &variables()).unwrap())
}

/// `z -> t`, `z -> y` contemporaneously and `t@1 -> y`.
fn pag() -> AcceptedGraph {
    let contemporaneous = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true);
    AcceptedGraph::from(
        temporal_pag_from_dbn_masks(contemporaneous, lag_bit(3, 0, 1), 0, 3, 1, &variables())
            .unwrap(),
    )
}

fn spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap()
}

fn mean_curve() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(spec())
}

fn intervention_response() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
    .with_temporal(spec())
}

fn closed_form_level(kind: &str) -> Vec<f64> {
    let truth: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_posterior_response_truth/expected.json"
    ))
    .unwrap();
    truth["horizon_1"]["level"][kind]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect()
}

/// Every evaluated completion atom lies within tolerance of the closed-form
/// structural level (an identified-set envelope must bracket it at both ends).
fn assert_atoms_match_level(
    result: &antecedent::result::StudyResult,
    level: &[f64],
    draws: Option<u32>,
    label: &str,
) {
    let base = 0.1;
    let mixture = result.structural_response.as_ref().expect("class response mixture");
    let mut checked = 0;
    for atom in mixture.atoms.iter().filter(|atom| atom.value.is_some()) {
        let bounds: Vec<(f64, f64)> = match atom.value.as_ref().unwrap() {
            ResponseValue::Surface { mean, .. } => mean.iter().map(|&m| (m, m)).collect(),
            ResponseValue::Scalar(value) => vec![(*value, *value)],
            ResponseValue::Envelope(envelope) => {
                envelope.lower.iter().zip(envelope.upper.iter()).map(|(&l, &u)| (l, u)).collect()
            }
            other => panic!("{label}: unexpected atom value {other:?}"),
        };
        assert_eq!(bounds.len(), level.len(), "{label}: cells");
        for (cell, &(lower, upper)) in bounds.iter().enumerate() {
            let monte_carlo = match (draws, atom.response.as_ref().map(|r| &r.uncertainty)) {
                (
                    Some(n_draws),
                    Some(ResponseUncertainty::PointwiseBand { lower: lo, upper: hi, .. }),
                ) => 4.0 * ((hi[cell] - lo[cell]) / (2.0 * 1.96)) / f64::from(n_draws).sqrt(),
                _ => 0.0,
            };
            let tolerance = base + monte_carlo;
            assert!(
                (lower - level[cell]).abs() < tolerance && (upper - level[cell]).abs() < tolerance,
                "{label} cell {cell}: atom [{lower}, {upper}] vs closed-form level {} (tolerance {tolerance})",
                level[cell]
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "{label}: no evaluated atom to check");
}

#[test]
fn all_temporal_class_response_coordinates_execute_from_retained_source_proofs() {
    let ctx = ExecutionContext::for_tests(977);
    let initial = series(0.0);
    for (class, graph) in [(GraphClass::TemporalCpdag, cpdag()), (GraphClass::TemporalPag, pag())] {
        for (kind, query) in [("curve", mean_curve()), ("set", intervention_response())] {
            let level = closed_form_level(kind);
            for source in ["explicit", "accepted"] {
                for (inference, draws) in [
                    (InferenceMode::Frequentist, None),
                    (
                        InferenceMode::Bayesian(
                            BayesianConfig::conjugate().n_draws(400).prior_scale(1000.0),
                        ),
                        Some(400),
                    ),
                ] {
                    let label = format!("{class:?}/{kind}/{source}/{draws:?}");
                    let builder = match (source, class) {
                        ("accepted", _) => Study::series(initial.clone()).graph(graph.clone()),
                        (_, GraphClass::TemporalCpdag) => Study::series(initial.clone())
                            .graph(graph.as_temporal_cpdag().unwrap().clone()),
                        _ => Study::series(initial.clone())
                            .graph(graph.as_temporal_pag().unwrap().clone()),
                    }
                    .query(CausalQuery::Response(query.clone()))
                    .inference(inference.clone())
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                    assert_eq!(builder.structure_source().as_str(), source);
                    let one_shot = builder.run(&ctx).unwrap();
                    let mut prepared = builder.prepare(&ctx).unwrap();
                    drop(builder);
                    let plan = prepared
                        .checked_temporal_class_response_info()
                        .expect("class response must retain its complete execution plan");
                    assert_eq!(plan.query, query);
                    assert_eq!(plan.graph_class, class);
                    assert_eq!(plan.identifier, IdentifierId::GeneralizedAdjustment);
                    assert_eq!(
                        plan.estimator,
                        if draws.is_some() {
                            EstimatorId::TemporalResponseBayesian
                        } else {
                            EstimatorId::TemporalResponseGcomp
                        }
                    );
                    assert_eq!(plan.validation, RefuteSuite::None);
                    assert_eq!(plan.horizons.as_ref(), &[1]);
                    assert!(plan.completion_count > 0);
                    assert!(plan.identified_mass > 0.0);

                    let result = prepared.estimate_series(&initial, &ctx).unwrap();
                    assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "exec.identify.cached"),
                        "{label}: the click must reuse the retained completion proof"
                    );
                    assert_atoms_match_level(&result, &level, draws, &label);
                    assert_atoms_match_level(
                        &one_shot,
                        &level,
                        draws,
                        &format!("{label}/one-shot"),
                    );
                    let mixture = result.structural_response.as_ref().unwrap();
                    assert!(
                        (mixture.identified_mass
                            + mixture.unidentified_mass
                            + mixture.unevaluable_mass
                            - 1.0)
                            .abs()
                            < 1e-9,
                        "{label}: completion mass is retained"
                    );

                    let shifted: Vec<f64> = level.iter().map(|value| value + 0.3).collect();
                    let refreshed = prepared.refresh_series(series(0.3), &ctx).unwrap();
                    assert_atoms_match_level(
                        &refreshed,
                        &shifted,
                        draws,
                        &format!("{label}/refresh"),
                    );
                    let widened = TimeSeriesData::from_f64_columns(
                        [
                            ("t", &[0.0; 16][..]),
                            ("y", &[0.0; 16][..]),
                            ("z", &[0.0; 16][..]),
                            ("extra", &[0.0; 16][..]),
                        ],
                        1,
                    )
                    .unwrap();
                    assert!(
                        prepared.refresh_series(widened, &ctx).is_err(),
                        "{label}: schema-changing refresh must be refused"
                    );
                    assert_eq!(
                        prepared.checked_temporal_class_response_info().unwrap().query,
                        query
                    );
                    let artifact = prepared
                        .encode_contracted_result(
                            &refreshed,
                            "checked-temporal-class-response",
                            &ctx,
                        )
                        .unwrap();
                    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                    assert!(
                        consumed.acceptance.unresolved.iter().any(|dependency| {
                            dependency.as_ref()
                                == "dependencies.checked_temporal_class_response_operation"
                        }),
                        "{label}: {:?}",
                        consumed.acceptance.unresolved
                    );
                    assert!(!consumed.acceptance.accepts_as_verified_program());
                }
            }
        }
    }
}

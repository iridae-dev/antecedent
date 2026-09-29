//! Builder-independent evidence for checked temporal graph-posterior responses
//! over `TemporalDag`, `TemporalCpdag`, and `TemporalPag` atoms (mean curve and
//! single-step intervention response, both inferences, all licensed suites).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one loop covers every licensed coordinate")]

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseQuery, ResponseUncertainty, ResponseValue, TemporalPolicy, TemporalResponseSpec, Value,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, set_edge};
use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics};

/// `defect_t = 0.9 pressure_{t-1}` on uniform white-noise pressure: the law of
/// `conformance/bayesian/known_truth_mixtures` `temporal_effect`.
fn dbn_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    let mut state = seed;
    for t in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        #[allow(clippy::cast_precision_loss)]
        let u = (state >> 33) as f64 / (1u64 << 31) as f64;
        pressure[t] = u * 2.0 - 1.0;
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("pressure", pressure.as_slice()), ("defect", defect.as_slice())],
        1,
    )
    .unwrap()
}

/// Two `TemporalDag` atoms from the known-truth pin: the lag-one effect graph and
/// an autoregressive atom whose finite-history certification fails, so its
/// posterior weight stays unidentified.
fn dbn_posterior(pin: &serde_json::Value) -> GraphPosterior {
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let identified_c = pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let unidentified_c = pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let identified_l = pin["identified_atom"]["lag_mask"].as_u64().unwrap();
    let unidentified_l = pin["unidentified_atom"]["lag_mask"].as_u64().unwrap();
    let lagged: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    GraphPosterior::new(
        2,
        weights.clone(),
        vec![identified_c, unidentified_c],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, lagged)
    .unwrap()
    .with_lag_masks(vec![identified_l, unidentified_l])
    .unwrap()
}

/// The law of `conformance/estimate/temporal_class_posterior_response_truth`
/// (`y_i = 1.4 t_{i-1} + 0.6 z_{i-1} + w_i`) plus a constant outcome shift.
fn class_series(shift: f64) -> TimeSeriesData {
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

/// One-atom `TemporalCpdag` (`t@1 -> y`, `z@1 -> y`) or `TemporalPag`
/// (`z -> t`, `z -> y`, `t@1 -> y`) posterior with all mass on that atom.
fn class_posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let n = 3;
    let (contemporaneous, lag) = match kind {
        GraphPosteriorAtomKind::Pag => {
            (set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true), lag_bit(3, 0, 1))
        }
        _ => (0, lag_bit(3, 0, 1) | lag_bit(3, 2, 1)),
    };
    GraphPosterior::new(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("temporal_class_posterior"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag])
    .unwrap()
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

fn class_level(kind: &str) -> Vec<f64> {
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

/// Every evaluated atom lies within tolerance of the known level; an
/// identified-set envelope must bracket it at both ends.
fn assert_atoms_match_level(
    result: &antecedent::result::StudyResult,
    level: &[f64],
    tolerance: f64,
    draws: Option<u32>,
    label: &str,
) {
    let mixture = result.structural_response.as_ref().expect("posterior response mixture");
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
            let tolerance = tolerance + monte_carlo;
            assert!(
                (lower - level[cell]).abs() < tolerance && (upper - level[cell]).abs() < tolerance,
                "{label} cell {cell}: atom [{lower}, {upper}] vs level {} (tolerance {tolerance})",
                level[cell]
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "{label}: no evaluated atom to check");
}

struct Fixture {
    kind: GraphPosteriorAtomKind,
    posterior: GraphPosterior,
    initial: TimeSeriesData,
    refreshed: TimeSeriesData,
    widened: TimeSeriesData,
    unidentified_mass: f64,
    level: fn(&str, f64) -> Vec<f64>,
    refresh_shift: f64,
    tolerance: f64,
}

fn dbn_level(kind: &str, _shift: f64) -> Vec<f64> {
    if kind == "curve" { vec![0.0, 0.9] } else { vec![0.9] }
}

fn class_shifted_level(kind: &str, shift: f64) -> Vec<f64> {
    class_level(kind).into_iter().map(|value| value + shift).collect()
}

fn fixtures() -> Vec<Fixture> {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let temporal = &pin["temporal_effect"];
    let n = usize::try_from(temporal["n"].as_u64().unwrap()).unwrap();
    let seed = temporal["seed"].as_u64().unwrap();
    let mut fixtures = vec![Fixture {
        kind: GraphPosteriorAtomKind::Dag,
        posterior: dbn_posterior(temporal),
        initial: dbn_series(n, seed),
        refreshed: dbn_series(n, seed + 1),
        widened: TimeSeriesData::from_f64_columns(
            [("pressure", &[0.0; 16][..]), ("defect", &[0.0; 16][..]), ("extra", &[0.0; 16][..])],
            1,
        )
        .unwrap(),
        unidentified_mass: temporal["expected_unidentified_mass"].as_f64().unwrap(),
        level: dbn_level,
        refresh_shift: 0.0,
        tolerance: temporal["effect_abs_tolerance"].as_f64().unwrap(),
    }];
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        fixtures.push(Fixture {
            kind,
            posterior: class_posterior(kind),
            initial: class_series(0.0),
            refreshed: class_series(0.3),
            widened: TimeSeriesData::from_f64_columns(
                [
                    ("t", &[0.0; 16][..]),
                    ("y", &[0.0; 16][..]),
                    ("z", &[0.0; 16][..]),
                    ("extra", &[0.0; 16][..]),
                ],
                1,
            )
            .unwrap(),
            unidentified_mass: 0.0,
            level: class_shifted_level,
            refresh_shift: 0.3,
            tolerance: 0.1,
        });
    }
    fixtures
}

#[test]
fn all_temporal_graph_posterior_response_coordinates_execute_from_retained_atoms() {
    let ctx = ExecutionContext::for_tests(211);
    for fixture in fixtures() {
        for (query_kind, query, suites) in [
            ("curve", mean_curve(), &[RefuteSuite::None][..]),
            (
                "set",
                intervention_response(),
                &[RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full][..],
            ),
        ] {
            for (inference, draws) in [
                (InferenceMode::Frequentist, None),
                (
                    InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(64).prior_scale(1_000_000.0),
                    ),
                    Some(64),
                ),
            ] {
                for &suite in suites {
                    let label = format!("{:?}/{query_kind}/{draws:?}/{suite:?}", fixture.kind);
                    let builder = Study::series(fixture.initial.clone())
                        .graph_posterior(fixture.posterior.clone())
                        .query(CausalQuery::Response(query.clone()))
                        .inference(inference.clone())
                        .refute(suite)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap();
                    assert_eq!(builder.structure_source().as_str(), "graph_posterior");
                    let one_shot = builder.run(&ctx).unwrap();
                    let mut prepared = builder.prepare(&ctx).unwrap();
                    drop(builder);
                    let plan = prepared
                        .checked_temporal_graph_posterior_response_info()
                        .expect("graph-posterior response must retain its complete execution plan");
                    assert_eq!(plan.query, query);
                    assert_eq!(plan.atom_kind, fixture.kind);
                    assert_eq!(plan.weights.as_ref(), fixture.posterior.weights.as_ref());
                    assert_eq!(plan.graph_keys.len(), fixture.posterior.n_graphs);
                    assert_eq!(plan.validation, suite);
                    assert_eq!(
                        plan.identifier,
                        if fixture.kind == GraphPosteriorAtomKind::Dag {
                            IdentifierId::TemporalBackdoorUnfolded
                        } else {
                            IdentifierId::GeneralizedAdjustment
                        }
                    );
                    assert_eq!(
                        plan.estimator,
                        if draws.is_some() {
                            EstimatorId::TemporalResponseBayesian
                        } else {
                            EstimatorId::TemporalResponseGcomp
                        }
                    );
                    let unidentified_weight: f64 = plan
                        .weights
                        .iter()
                        .zip(plan.identified.iter())
                        .filter(|(_, flag)| **flag == GraphIdentFlag::Unidentified)
                        .map(|(weight, _)| *weight)
                        .sum();
                    assert!(
                        (unidentified_weight - fixture.unidentified_mass).abs() < 1e-12,
                        "{label}: retained unidentified atom mass {unidentified_weight}"
                    );

                    let result = prepared.estimate_series(&fixture.initial, &ctx).unwrap();
                    assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "exec.identify.cached"),
                        "{label}: the click must reuse the retained atom proofs"
                    );
                    let level = (fixture.level)(query_kind, 0.0);
                    assert_atoms_match_level(&result, &level, fixture.tolerance, draws, &label);
                    assert_atoms_match_level(
                        &one_shot,
                        &level,
                        fixture.tolerance,
                        draws,
                        &format!("{label}/one-shot"),
                    );
                    let mixture = result.structural_response.as_ref().unwrap();
                    assert!(
                        (mixture.unidentified_mass - fixture.unidentified_mass).abs() < 1e-12,
                        "{label}: unidentified mass {}",
                        mixture.unidentified_mass
                    );
                    if suite != RefuteSuite::None {
                        assert!(
                            !result.refutations.is_empty()
                                || result.diagnostics.iter().any(|d| {
                                    d.code.as_ref()
                                        == "refute.dbn_posterior.intervention_pulse_suite"
                                        || d.code.as_ref()
                                            == "refute.envelope.temporal_class_graph_posterior"
                                }),
                            "{label}: the Pulse-native suite must run on the retained atoms"
                        );
                    }

                    let refreshed =
                        prepared.refresh_series(fixture.refreshed.clone(), &ctx).unwrap();
                    let shifted = (fixture.level)(query_kind, fixture.refresh_shift);
                    assert_atoms_match_level(
                        &refreshed,
                        &shifted,
                        fixture.tolerance,
                        draws,
                        &format!("{label}/refresh"),
                    );
                    assert!(
                        prepared.refresh_series(fixture.widened.clone(), &ctx).is_err(),
                        "{label}: schema-changing refresh must be refused"
                    );
                    assert_eq!(
                        prepared.checked_temporal_graph_posterior_response_info().unwrap().query,
                        query
                    );
                    let artifact = prepared
                        .encode_contracted_result(
                            &refreshed,
                            "checked-temporal-graph-posterior-response",
                            &ctx,
                        )
                        .unwrap();
                    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                    assert!(
                        consumed.acceptance.unresolved.iter().any(|dependency| {
                            dependency.as_ref()
                                == "dependencies.checked_temporal_graph_posterior_response_operation"
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

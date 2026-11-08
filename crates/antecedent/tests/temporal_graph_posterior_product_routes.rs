//! Temporal graph-posterior routes on event data against the atom-weighted truth.
//!
//! Event observations are aligned onto the regular grid before any estimator runs, so a
//! graph-posterior study built from dense events must answer exactly what the same study
//! answers on the aligned series, and that answer is the posterior-weighted truth of the
//! atoms the posterior identifies. Each test checks both: the event route equals the series
//! route, and the event route lies within tolerance of the independently derived truth.
//! The pulse, sustained, response-curve, intervention-response and temporal-mediation
//! queries are covered; the series-data side of each lives in the checked temporal
//! graph-posterior route evidence files.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "each test states its fixture, route and truth inline")]

#[path = "fixtures_common/mod.rs"]
mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_contract::DataClassification;
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    MeasurementSpec, MediationContrast, MediationQuery, ResponseFunctional, ResponseQuery,
    ResponseValue, RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, ValueType, VariableId,
};
use antecedent_data::{
    EventData, Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{GraphPosterior, SamplingSummary};
use antecedent_prob::InferenceDiagnostics;
use common::fixtures::{self, confounded_series, heterogeneous_dbn, B2, DBN_UNIDENTIFIED_MASS};

/// Dense events, one per nanosecond, so alignment to a one-nanosecond grid is the identity.
fn events_of(series: &TimeSeriesData, rows: usize) -> EventData {
    let times: Vec<i64> =
        (0..rows).map(|i| i64::try_from(i).expect("row index fits i64")).collect();
    EventData::try_new(series.storage().clone(), Arc::from(times)).unwrap()
}

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

// --- effect routes (pulse, sustained) ---------------------------------------------------

const EFFECT_ROWS: usize = 1_200;
const EFFECT_TOLERANCE: f64 = 0.12;

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(var(0), var(1), 1.0).with_policy(TemporalPolicy::pulse(-1))
}

fn multi_step_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(var(0), var(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
}

fn effect_point(result: &StudyResult, inference: &InferenceMode) -> f64 {
    match inference {
        InferenceMode::Frequentist => result.estimate().as_effect().unwrap().ate,
        InferenceMode::Bayesian(_) => {
            let posterior = result.posterior.as_ref().expect("a Bayesian effect posterior");
            posterior.summaries.mean[posterior.effect_column().expect("the effect column")]
        }
    }
}

fn effect_inferences() -> [InferenceMode; 2] {
    [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96).prior_scale(100.0)),
    ]
}

fn assert_effect_event_route(query: &TemporalEffectQuery, truth: f64) {
    let series = confounded_series(EFFECT_ROWS, B2, 0.0, 4_101);
    let event = events_of(&series, EFFECT_ROWS);
    let ctx = ExecutionContext::for_tests(4_100);
    for inference in effect_inferences() {
        let build = |builder: antecedent::StudyBuilder| {
            builder
                .graph_posterior(heterogeneous_dbn())
                .query(CausalQuery::TemporalEffect(query.clone()))
                .inference(inference.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
        };
        let from_events = build(Study::events(&event, 1).unwrap()).run(&ctx).unwrap();
        let from_series = build(Study::series(series.clone())).run(&ctx).unwrap();
        assert_eq!(from_events.logical_plan.data_classification, DataClassification::Event);

        let estimate = effect_point(&from_events, &inference);
        assert!(
            (estimate - truth).abs() < EFFECT_TOLERANCE,
            "event route {estimate} vs atom-weighted truth {truth}"
        );
        let reference = effect_point(&from_series, &inference);
        assert!(
            (estimate - reference).abs() < 1e-9,
            "event route {estimate} differs from the aligned series route {reference}"
        );
        let unidentified = from_events
            .structural_response
            .as_ref()
            .map(|mixture| mixture.unidentified_mass)
            .or_else(|| from_events.posterior.as_ref().map(|posterior| posterior.unidentified_mass))
            .expect("the unidentified posterior mass is retained");
        assert!(
            (unidentified - DBN_UNIDENTIFIED_MASS).abs() < 1e-9,
            "unidentified posterior mass {unidentified}"
        );
    }
}

/// A pulse through a two-identified-atom posterior: the answer is the frozen-weight mean of
/// the atoms' own plim truths over the identified mass; the unidentified atom is excluded.
#[test]
fn pulse_on_event_data_recovers_the_atom_weighted_truth_and_the_series_answer() {
    let truth = fixtures::dbn_mixture_truth(fixtures::dbn_pulse_atom_truths(0.0));
    assert_effect_event_route(&pulse(), truth);
}

/// Two-step sustained exposure runs sequential g-computation per atom; same weighting.
#[test]
fn sustained_on_event_data_recovers_the_atom_weighted_truth_and_the_series_answer() {
    let truth = fixtures::dbn_mixture_truth(fixtures::dbn_multistep_atom_truths(0.0));
    assert_effect_event_route(&multi_step_sustained(), truth);
}

// --- response routes (curve, intervention response) -------------------------------------

/// `defect_t = 0.9 pressure_{t-1}` on uniform white-noise pressure: the law of the
/// known-truth mixture pin's `temporal_effect`.
fn response_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    let mut state = seed;
    for t in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
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

/// The pin's two `TemporalDag` atoms: the lag-one effect graph (identified) and an
/// autoregressive atom whose finite-history certification fails (unidentified).
fn response_posterior(pin: &serde_json::Value) -> GraphPosterior {
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let lagged: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    GraphPosterior::from_atoms(
        2,
        weights.clone(),
        vec![
            pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
            pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        ],
        vec![0.0; 4],
        vec![0.0; 4],
        SamplingSummary {
            ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("known_truth_mixtures"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, lagged)
    .unwrap()
    .with_lag_masks(vec![
        pin["identified_atom"]["lag_mask"].as_u64().unwrap(),
        pin["unidentified_atom"]["lag_mask"].as_u64().unwrap(),
    ])
    .unwrap()
}

fn response_spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap()
}

fn mean_curve() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: var(1),
        treatment: ContinuousDomain::new(var(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    })
    .with_temporal(response_spec())
}

fn intervention_response() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: var(1),
        interventions: Arc::from([Intervention::set(var(0), Value::f64(1.0))]),
    })
    .with_temporal(response_spec())
}

/// The evaluated atoms' cell values (an atom the posterior cannot identify has none).
fn evaluated_atoms(result: &StudyResult) -> Vec<Vec<f64>> {
    let mixture = result.structural_response.as_ref().expect("posterior response mixture");
    mixture
        .atoms
        .iter()
        .filter_map(|atom| atom.value.as_ref())
        .map(|value| match value {
            ResponseValue::Surface { mean, .. } => mean.to_vec(),
            ResponseValue::Scalar(value) => vec![*value],
            other => panic!("unexpected atom value {other:?}"),
        })
        .collect()
}

fn assert_response_event_route(query: &ResponseQuery, level: &[f64]) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &pin["temporal_effect"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let series = response_series(n, pin["seed"].as_u64().unwrap());
    let event = events_of(&series, n);
    let ctx = ExecutionContext::for_tests(211);
    let build = |builder: antecedent::StudyBuilder| {
        builder
            .graph_posterior(response_posterior(pin))
            .query(CausalQuery::Response(query.clone()))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    };
    let from_events = build(Study::events(&event, 1).unwrap()).run(&ctx).unwrap();
    let from_series = build(Study::series(series.clone())).run(&ctx).unwrap();
    assert_eq!(from_events.logical_plan.data_classification, DataClassification::Event);
    assert_eq!(from_events.support_status().unwrap().as_str(), "supported");

    let atoms = evaluated_atoms(&from_events);
    assert!(!atoms.is_empty(), "no evaluated atom to check");
    for cells in &atoms {
        assert_eq!(cells.len(), level.len(), "cells");
        for (cell, (value, truth)) in cells.iter().zip(level).enumerate() {
            assert!(
                (value - truth).abs() < tolerance,
                "cell {cell}: event route {value} vs known level {truth} (tolerance {tolerance})"
            );
        }
    }
    assert_eq!(atoms.len(), evaluated_atoms(&from_series).len());
    for (event_cells, series_cells) in atoms.iter().zip(evaluated_atoms(&from_series)) {
        for (event_value, series_value) in event_cells.iter().zip(series_cells) {
            assert!(
                (event_value - series_value).abs() < 1e-9,
                "event route {event_value} differs from the aligned series route {series_value}"
            );
        }
    }
    let mixture = from_events.structural_response.as_ref().unwrap();
    assert!(
        (mixture.unidentified_mass - unidentified_truth).abs() < 1e-12,
        "unidentified mass {}",
        mixture.unidentified_mass
    );
}

/// `E[defect | do(pressure = x)]` at lag one is `0.9 x`: the curve on the grid {0, 1}.
#[test]
fn response_curve_on_event_data_matches_the_known_level_and_the_series_answer() {
    assert_response_event_route(&mean_curve(), &[0.0, 0.9]);
}

/// The same law under `do(pressure = 1)` gives the single level 0.9.
#[test]
fn intervention_response_on_event_data_matches_the_known_level_and_the_series_answer() {
    assert_response_event_route(&intervention_response(), &[0.9]);
}

// --- temporal mediation -----------------------------------------------------------------

/// `m_t = 0.8 t_{t-1}`, `y_t = 0.25 t_{t-1} + 0.55 m_t` on a deterministic sinusoid treatment
/// (the mediation grid law): the mediated path is `0.8 * 0.55 = 0.44`.
fn mediation_series(n: usize) -> TimeSeriesData {
    let mut builder = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
    }
    for i in 1..n {
        m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
    }
    let columns = [t, m, y]
        .into_iter()
        .enumerate()
        .map(|(raw, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    var(u32::try_from(raw).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn mediation_posterior(pin: &serde_json::Value) -> GraphPosterior {
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let lagged: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let marginals = vec![0.0; 9];
    GraphPosterior::from_atoms(
        3,
        weights.clone(),
        vec![
            pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
            pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        ],
        marginals.clone(),
        marginals,
        SamplingSummary {
            ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("known_truth_mixtures"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, lagged)
    .unwrap()
    .with_lag_masks(vec![
        pin["identified_atom"]["lag_mask"].as_u64().unwrap(),
        pin["unidentified_atom"]["lag_mask"].as_u64().unwrap(),
    ])
    .unwrap()
}

/// The mediated contrast of the one identified atom is `0.8 * 0.55 = 0.44`; the
/// autoregressive atom (weight 0.3) is unidentified and drops out of the mixture.
#[test]
fn temporal_mediation_on_event_data_recovers_the_mediated_truth_and_the_series_answer() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &pin["temporal_mediation"];
    assert_eq!(pin["contrast"], "mediated");
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    // The series-data evidence for this route pins the frequentist estimate within 0.015.
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap().min(0.015);
    let series = mediation_series(n);
    let event = events_of(&series, n);
    let ctx = ExecutionContext::for_tests(pin["seed"].as_u64().unwrap());
    let query = MediationQuery::binary(var(0), var(2), [var(1)], MediationContrast::Mediated);
    let build = |builder: antecedent::StudyBuilder| {
        builder
            .graph_posterior(mediation_posterior(pin))
            .query(CausalQuery::Mediation(query.clone()))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    };
    let from_events = build(Study::events(&event, 1).unwrap()).run(&ctx).unwrap();
    let from_series = build(Study::series(series)).run(&ctx).unwrap();
    assert_eq!(from_events.logical_plan.data_classification, DataClassification::Event);
    assert_eq!(from_events.support_status().unwrap().as_str(), "supported");

    let estimate = from_events.estimate().as_effect().unwrap().ate;
    assert!(
        (estimate - truth).abs() < tolerance,
        "event route {estimate} vs mediated truth {truth} (tolerance {tolerance})"
    );
    let reference = from_series.estimate().as_effect().unwrap().ate;
    assert!(
        (estimate - reference).abs() < 1e-9,
        "event route {estimate} differs from the aligned series route {reference}"
    );
}

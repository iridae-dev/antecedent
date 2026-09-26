//! Builder-independent evidence for checked `TemporalCpdag` class-envelope and
//! temporal graph-posterior mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, MediationContrast,
    MediationQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_graph::TemporalCpdag;
use antecedent_prob::InferenceDiagnostics;

const N: usize = 241;
/// Structural horizon-1 mediated effect of the confounded law: the path
/// product `a * b` (`conformance/estimate/temporal_class_mediation_truth`).
const MEDIATED_TRUTH: f64 = 0.8 * 0.55;
const CHECKED_DEPENDENCY: &str = "dependencies.checked_temporal_mediation_operation";

/// The confounded mediation law `T[t-1] -> M[t] -> Y[t]` with `Z` confounding
/// treatment, mediator, and outcome. An intercept shift leaves every mediation
/// contrast unchanged; an extra column changes the semantic schema.
fn confounded_series(outcome_shift: f64, extra_column: bool) -> TimeSeriesData {
    let mut schema = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        schema
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    if extra_column {
        schema
            .add_variable(
                "extra",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = schema.build().unwrap();
    let z: Vec<f64> = (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..N).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut m = vec![0.0; N];
    let mut y = vec![1.0 + outcome_shift; N];
    for i in 1..N {
        let noise = (i % 5) as f64 - 2.0;
        m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
        y[i] = 1.0 + outcome_shift + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
    }
    let mut columns = vec![t, m, y, z];
    if extra_column {
        columns.push((0..N).map(|i| (i as f64).sin()).collect());
    }
    let columns = columns
        .into_iter()
        .enumerate()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(id).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(N),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap()
}

/// The confounded law as a `TemporalCpdag` whose every edge is oriented: one
/// completion, so the horizon-1 identified set is that completion's contrast.
fn oriented_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let t0 = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = graph.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    for (from, to) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

fn changed_cpdag() -> TemporalCpdag {
    let mut graph = oriented_cpdag();
    // An unused older node changes the prepared structure while leaving the
    // causal target and observed schema intact.
    graph.add_lagged(VariableId::from_raw(3), Lag::from_raw(2)).unwrap();
    graph
}

fn class_graph(accepted: bool) -> AcceptedGraph {
    if accepted {
        AcceptedGraph::temporal_cpdag(oriented_cpdag()).unwrap()
    } else {
        oriented_cpdag().into()
    }
}

fn mediation_query(horizons: &[u32]) -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(horizons.to_vec())
    .unwrap()
}

fn inferences() -> [(InferenceMode, EstimatorId, f64); 2] {
    [
        (InferenceMode::Frequentist, EstimatorId::TemporalMediation, 1e-6),
        (
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)),
            EstimatorId::BayesianTemporalMediation,
            0.08,
        ),
    ]
}

fn has_cached_identification(result: &antecedent::StudyResult) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
}

#[test]
fn checked_temporal_class_mediation_lifecycle_covers_all_class_envelope_coordinates() {
    let initial = confounded_series(0.0, false);
    let query = mediation_query(&[1]);
    let context = ExecutionContext::for_tests(821);

    for accepted in [false, true] {
        for (inference, expected_estimator, tolerance) in inferences() {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let case = format!("accepted={accepted} {expected_estimator:?} {suite:?}");
                let builder = Study::series(initial.clone())
                    .graph(class_graph(accepted))
                    .query(CausalQuery::Mediation(query.clone()))
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(8)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&context).unwrap();
                let mut prepared = builder.prepare(&context).unwrap();
                drop(builder);

                let plan = prepared
                    .checked_temporal_class_mediation_info()
                    .expect("TemporalCpdag mediation must retain its checked operation");
                assert_eq!(plan.query, query, "{case}");
                assert_eq!(plan.graph_class, antecedent::GraphClass::TemporalCpdag, "{case}");
                assert_eq!(plan.identifier.as_ref(), "generalized.adjustment", "{case}");
                assert_eq!(plan.estimator, expected_estimator, "{case}");
                assert_eq!(plan.validation, suite, "{case}");
                assert_eq!(plan.bootstrap_replicates, 8, "{case}");
                assert_eq!(plan.horizons.as_ref(), &[1], "{case}");
                assert_eq!(plan.atom_count, 1, "{case}: one oriented completion");
                assert!((plan.identified_mass - 1.0).abs() < 1e-12, "{case}");
                assert!(plan.unidentified_mass.abs() < 1e-12, "{case}");

                let first = prepared.estimate_series(&initial, &context).unwrap();
                assert!(
                    (first.estimate.ate - MEDIATED_TRUTH).abs() < tolerance,
                    "{case}: mediated {} vs structural {MEDIATED_TRUTH}",
                    first.estimate.ate
                );
                assert_eq!(
                    first.logical_plan.estimator.as_deref(),
                    Some(expected_estimator.as_str()),
                    "{case}"
                );
                assert_eq!(first.refutations.is_empty(), suite == RefuteSuite::None, "{case}");
                let grid = first.mediation_grid.as_ref().expect("class mediation grid");
                let set = grid.slices[0].identified_set.expect("horizon-1 identified set");
                assert!(
                    (set.lower - MEDIATED_TRUTH).abs() < tolerance
                        && (set.upper - MEDIATED_TRUTH).abs() < tolerance,
                    "{case}: identified set [{}, {}]",
                    set.lower,
                    set.upper
                );
                assert!(!grid.joint_posterior, "{case}");
                assert!(has_cached_identification(&first), "{case}: prepared proof reused");
                assert!(
                    has_cached_identification(&one_shot),
                    "{case}: one-shot run must execute its retained prepared plan"
                );
                assert!(
                    (one_shot.estimate.ate - first.estimate.ate).abs() < 1e-12,
                    "{case}: one-shot and prepared execution share one program"
                );

                let refreshed =
                    prepared.refresh_series(confounded_series(0.4, false), &context).unwrap();
                assert!(
                    (refreshed.estimate.ate - MEDIATED_TRUTH).abs() < tolerance,
                    "{case}: intercept shift must not change the mediated contrast"
                );
                assert_eq!(
                    prepared.checked_temporal_class_mediation_info().unwrap().proof_signature,
                    plan.proof_signature,
                    "{case}"
                );

                let artifact = prepared
                    .encode_contracted_result(
                        &refreshed,
                        "checked-temporal-class-mediation",
                        &context,
                    )
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(
                    consumed
                        .acceptance
                        .unresolved
                        .iter()
                        .any(|dependency| dependency.as_ref() == CHECKED_DEPENDENCY),
                    "{case}: independent consumption must report the missing checked operation"
                );
                assert!(!consumed.acceptance.accepts_as_verified_program(), "{case}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Graph-posterior coordinates
// ---------------------------------------------------------------------------

const POSTERIOR_N: usize = 320;

/// `M_t = 0.8 T_{t-1}; Y_t = 0.25 T_{t-1} + 0.55 M_t` with a deterministic
/// sinusoid treatment (`conformance/bayesian/known_truth_mixtures`,
/// `temporal_mediation`).
fn posterior_series(outcome_shift: f64, extra_column: bool) -> TimeSeriesData {
    let n = POSTERIOR_N;
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![outcome_shift; n];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
    }
    for i in 1..n {
        m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
        y[i] = outcome_shift + 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
    }
    let extra: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut columns = vec![("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())];
    if extra_column {
        columns.push(("extra", extra.as_slice()));
    }
    TimeSeriesData::from_f64_columns(columns, 1).unwrap()
}

struct MediationPin {
    weights: Vec<f64>,
    identified_contemporaneous: u64,
    identified_lag: u64,
    unidentified_contemporaneous: u64,
    unidentified_lag: u64,
    lagged_edge_marginals: Vec<f64>,
    effect: f64,
    unidentified_mass: f64,
    tolerance: f64,
}

fn mediation_pin() -> MediationPin {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_mediation"];
    assert_eq!(pin["n"].as_u64().unwrap(), POSTERIOR_N as u64);
    let floats = |key: &str| -> Vec<f64> {
        pin[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    };
    MediationPin {
        weights: floats("posterior_weights"),
        identified_contemporaneous: pin["identified_atom"]["contemporaneous_mask"]
            .as_u64()
            .unwrap(),
        identified_lag: pin["identified_atom"]["lag_mask"].as_u64().unwrap(),
        unidentified_contemporaneous: pin["unidentified_atom"]["contemporaneous_mask"]
            .as_u64()
            .unwrap(),
        unidentified_lag: pin["unidentified_atom"]["lag_mask"].as_u64().unwrap(),
        lagged_edge_marginals: floats("lagged_edge_marginals"),
        effect: pin["expected_effect_given_identified"].as_f64().unwrap(),
        unidentified_mass: pin["expected_unidentified_mass"].as_f64().unwrap(),
        tolerance: pin["effect_abs_tolerance"].as_f64().unwrap(),
    }
}

/// The known-truth two-atom temporal posterior as DBN (`TemporalDag`) or
/// class (`TemporalCpdag`) atoms over `(t, m, y)`.
fn known_truth_posterior(pin: &MediationPin, kind: GraphPosteriorAtomKind) -> GraphPosterior {
    posterior_with_weights(pin, kind, &pin.weights)
}

/// The same atoms under other frozen weights: a different structure proof.
fn changed_posterior(pin: &MediationPin, kind: GraphPosteriorAtomKind) -> GraphPosterior {
    posterior_with_weights(pin, kind, &[0.6, 0.4])
}

fn posterior_with_weights(
    pin: &MediationPin,
    kind: GraphPosteriorAtomKind,
    weights: &[f64],
) -> GraphPosterior {
    let contemporaneous_marginals = vec![0.0; 9];
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![pin.identified_contemporaneous, pin.unidentified_contemporaneous],
        contemporaneous_marginals.clone(),
        contemporaneous_marginals,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, pin.lagged_edge_marginals.clone())
    .unwrap()
    .with_lag_masks(vec![pin.identified_lag, pin.unidentified_lag])
    .unwrap()
}

#[test]
#[allow(clippy::too_many_lines)]
fn checked_temporal_class_mediation_lifecycle_covers_all_graph_posterior_coordinates() {
    let pin = mediation_pin();
    let initial = posterior_series(0.0, false);
    let query = mediation_query(&[1]);
    let context = ExecutionContext::for_tests(823);

    for (kind, graph_class) in [
        (GraphPosteriorAtomKind::Dag, antecedent::GraphClass::TemporalDag),
        (GraphPosteriorAtomKind::Cpdag, antecedent::GraphClass::TemporalCpdag),
    ] {
        let posterior = known_truth_posterior(&pin, kind);
        for (inference, expected_estimator, _) in inferences() {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let case = format!("{graph_class:?} {expected_estimator:?} {suite:?}");
                let builder = Study::series(initial.clone())
                    .graph_posterior(posterior.clone())
                    .query(CausalQuery::Mediation(query.clone()))
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(8)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&context).unwrap();
                let mut prepared = builder.prepare(&context).unwrap();
                drop(builder);

                let plan = prepared
                    .checked_temporal_class_mediation_info()
                    .expect("graph-posterior mediation must retain its checked operation");
                assert_eq!(plan.query, query, "{case}");
                assert_eq!(plan.graph_class, graph_class, "{case}");
                assert_eq!(plan.identifier.as_ref(), "temporal.mediation", "{case}");
                assert_eq!(plan.estimator, expected_estimator, "{case}");
                assert_eq!(plan.validation, suite, "{case}");
                assert_eq!(plan.bootstrap_replicates, 8, "{case}");
                assert_eq!(plan.atom_count, 2, "{case}: both posterior atoms retained");
                assert!(
                    (plan.unidentified_mass - pin.unidentified_mass).abs() < 1e-12,
                    "{case}: unidentified mass {} vs pin {}",
                    plan.unidentified_mass,
                    pin.unidentified_mass
                );
                assert!(
                    (plan.identified_mass + plan.unidentified_mass - 1.0).abs() < 1e-12,
                    "{case}"
                );

                let first = prepared.estimate_series(&initial, &context).unwrap();
                assert!(
                    (first.estimate.ate - pin.effect).abs() < pin.tolerance,
                    "{case}: mediated {} vs truth {}",
                    first.estimate.ate,
                    pin.effect
                );
                assert_eq!(
                    first.logical_plan.estimator.as_deref(),
                    Some(expected_estimator.as_str()),
                    "{case}"
                );
                // The frequentist and class executors publish the frozen mass split
                // as a structural mixture; the Bayesian DBN executor keeps it on
                // the mixed posterior. Either way the unidentified atom keeps its
                // posterior weight through execution.
                let unidentified_mass = first
                    .structural_response
                    .as_ref()
                    .map(|mixture| mixture.unidentified_mass)
                    .or_else(|| first.posterior.as_ref().map(|p| p.unidentified_mass))
                    .expect("posterior mass split retained");
                assert!(
                    (unidentified_mass - pin.unidentified_mass).abs() < 1e-12,
                    "{case}: unidentified mass {unidentified_mass} retained through execution"
                );
                assert_eq!(first.refutations.is_empty(), suite == RefuteSuite::None, "{case}");
                assert!(has_cached_identification(&first), "{case}: prepared proof reused");
                assert!(
                    has_cached_identification(&one_shot),
                    "{case}: one-shot run must execute its retained prepared plan"
                );
                assert!(
                    (one_shot.estimate.ate - first.estimate.ate).abs() < 1e-12,
                    "{case}: one-shot and prepared execution share one program"
                );

                let refreshed =
                    prepared.refresh_series(posterior_series(0.7, false), &context).unwrap();
                assert!(
                    (refreshed.estimate.ate - pin.effect).abs() < pin.tolerance,
                    "{case}: intercept shift must not change the mediated contrast"
                );
                assert_eq!(
                    prepared.checked_temporal_class_mediation_info().unwrap().proof_signature,
                    plan.proof_signature,
                    "{case}"
                );

                let artifact = prepared
                    .encode_contracted_result(
                        &refreshed,
                        "checked-temporal-posterior-mediation",
                        &context,
                    )
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(
                    consumed
                        .acceptance
                        .unresolved
                        .iter()
                        .any(|dependency| dependency.as_ref() == CHECKED_DEPENDENCY),
                    "{case}: independent consumption must report the missing checked operation"
                );
                assert!(!consumed.acceptance.accepts_as_verified_program(), "{case}");
            }
        }
    }
}

#[test]
fn checked_temporal_class_mediation_refuses_schema_changes_and_binds_its_structure() {
    let context = ExecutionContext::for_tests(829);
    let query = mediation_query(&[1]);

    // Fixed TemporalCpdag: schema change refused, structure bound to the proof.
    let initial = confounded_series(0.0, false);
    let mut prepared = Study::series(initial.clone())
        .graph(oriented_cpdag())
        .query(CausalQuery::Mediation(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let plan = prepared.checked_temporal_class_mediation_info().unwrap();
    assert!(prepared.refresh_series(confounded_series(0.0, true), &context).is_err());
    let other = Study::series(initial.clone())
        .graph(changed_cpdag())
        .query(CausalQuery::Mediation(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    assert_ne!(
        plan.proof_signature,
        other.checked_temporal_class_mediation_info().unwrap().proof_signature,
        "the retained proof identity must include disconnected lagged nodes"
    );
    let other_result = other.estimate_series(&initial, &context).unwrap();
    assert!(
        prepared.encode_contracted_result(&other_result, "wrong-class-graph", &context).is_err(),
        "a result prepared under another graph must not bind to this handle"
    );

    // Graph posterior: schema change refused, posterior identity retained.
    let pin = mediation_pin();
    let series = posterior_series(0.0, false);
    for kind in [GraphPosteriorAtomKind::Dag, GraphPosteriorAtomKind::Cpdag] {
        let mut prepared = Study::series(series.clone())
            .graph_posterior(known_truth_posterior(&pin, kind))
            .query(CausalQuery::Mediation(query.clone()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        let plan = prepared.checked_temporal_class_mediation_info().unwrap();
        assert!(prepared.refresh_series(posterior_series(0.0, true), &context).is_err());
        let other = Study::series(series.clone())
            .graph_posterior(changed_posterior(&pin, kind))
            .query(CausalQuery::Mediation(query.clone()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        assert_ne!(
            plan.proof_signature,
            other.checked_temporal_class_mediation_info().unwrap().proof_signature,
            "{kind:?}: the retained proof identity must include the frozen posterior weights"
        );
    }

    // A multi-horizon frequentist posterior grid has no joint uncertainty
    // contract, so the one-shot facade keeps the legacy refusal rather than
    // retaining a checked operation for it.
    let multi = Study::series(series)
        .graph_posterior(known_truth_posterior(&pin, GraphPosteriorAtomKind::Dag))
        .query(CausalQuery::Mediation(mediation_query(&[1, 2])))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!(multi.run(&context).is_err());
    assert!(
        multi
            .prepare(&context)
            .map_or(true, |prepared| prepared.checked_temporal_class_mediation_info().is_none())
    );
}

//! Prepared lifecycle, refresh and independent artifact consumption of one
//! finite two-step temporal transport sequence (2.2A, X5).
//!
//! The fixture is the known temporal SCM shared with the estimate-crate tests:
//! source and target differ at the initial state and at the step-2 covariate
//! mechanism, `l2` is a time-varying confounder, and two latent bits confound
//! each action with the outcome. Every expected value is enumerated from the
//! structural model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;

use antecedent::{PreparedTemporalTransport, StudyBuilder, consume_temporal_transport_artifact};
use antecedent_core::{ExecutionContext, SearchLimits, Value};
use antecedent_estimate::temporal_transport::TemporalSequenceReport;
use antecedent_expr::{
    DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData, LawTolerance,
};
use antecedent_io::IoError;
use antecedent_io::temporal_transport_artifact::{
    TemporalSequenceArtifactWire, TemporalTransportArtifactError, TemporalTransportConsumeLimits,
    temporal_sequence_identity,
};
use common::temporal_fixture::{catalog, laws, sequence, source_scm, spec, target_scm, truth, v};

/// One edit of a decoded artifact.
type Edit = Box<dyn Fn(&mut TemporalSequenceArtifactWire)>;

const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn prepare_with(seq: &[Value], data: ExactTransportData) -> PreparedTemporalTransport {
    StudyBuilder::temporal_transport_sequence(
        &spec(),
        seq,
        "source",
        "target",
        catalog(),
        BUDGET,
        data,
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn prepare(seq: &[Value]) -> PreparedTemporalTransport {
    prepare_with(seq, laws(&source_scm(), &target_scm(), &[]))
}

fn exported(a1: f64, a2: f64) -> (PreparedTemporalTransport, TemporalSequenceReport, Vec<u8>) {
    let prepared = prepare(&sequence(a1, a2));
    let report = prepared.estimate(&ctx()).unwrap();
    let bytes = prepared.export(&report).unwrap();
    (prepared, report, bytes)
}

fn consume(bytes: &[u8]) -> Result<TemporalSequenceReport, IoError> {
    consume_temporal_transport_artifact(bytes, TemporalTransportConsumeLimits::default(), &ctx())
}

/// Mutate the decoded artifact and encode it again, keeping or recomputing its digest.
fn mutated(
    bytes: &[u8],
    edit: impl FnOnce(&mut TemporalSequenceArtifactWire),
    redigest: bool,
) -> Vec<u8> {
    let mut wire = TemporalSequenceArtifactWire::decode(bytes).unwrap();
    edit(&mut wire);
    if redigest {
        wire.premises_digest = temporal_sequence_identity(&wire).unwrap();
    }
    wire.export().unwrap()
}

fn typed(result: Result<TemporalSequenceReport, IoError>) -> TemporalTransportArtifactError {
    match result {
        Err(IoError::TemporalTransport(error)) => error,
        other => panic!("expected a typed temporal artifact error, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn exported_point_is_recomputed_by_an_independent_consumer() {
    let (_, report, bytes) = exported(1.0, 0.0);
    let consumed = consume(&bytes).unwrap();
    assert_eq!(consumed.mean.to_bits(), report.mean.to_bits());
    assert_eq!(consumed.sequence.len(), 2);
    assert!((consumed.mean - truth(&target_scm(), [1, 0])).abs() < 1e-12);
    // Every probability of the recomputed point agrees bit for bit.
    for (a, b) in
        consumed.distribution.probabilities.iter().zip(report.distribution.probabilities.iter())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(consumed.support, report.support);
    assert_eq!(consumed.invariances, report.invariances);
    assert_eq!(consumed.time_varying_confounders, vec![v(common::temporal_fixture::L2)]);
}

#[test]
fn identity_includes_the_sequence_and_the_horizon() {
    let (_, _, ab) = exported(0.0, 1.0);
    let (_, _, ba) = exported(1.0, 0.0);
    let (wa, wb) = (
        TemporalSequenceArtifactWire::decode(&ab).unwrap(),
        TemporalSequenceArtifactWire::decode(&ba).unwrap(),
    );
    assert_ne!(wa.premises_digest, wb.premises_digest);
    assert_eq!(wa.horizon, 2);
    // The horizon is part of the identity: another horizon is another digest.
    let mut longer = wa.clone();
    longer.horizon = 3;
    assert_ne!(temporal_sequence_identity(&longer).unwrap(), wa.premises_digest);
    // The identity is stable across exports of the same premises.
    let (_, _, again) = exported(0.0, 1.0);
    assert_eq!(
        TemporalSequenceArtifactWire::decode(&again).unwrap().premises_digest,
        wa.premises_digest
    );
}

#[test]
fn a_mutated_artifact_fails_independent_consumption_with_typed_errors() {
    let (_, _, bytes) = exported(1.0, 0.0);
    let premises = TemporalTransportArtifactError::PremisesMismatch;
    // A premise changed without its digest: the sequence, horizon, names, graph,
    // selections, slots, question and every limit are bound.
    let cases: Vec<(&str, Edit)> = vec![
        ("sequence", Box::new(|w| w.sequence.swap(0, 1))),
        ("horizon", Box::new(|w| w.horizon = 3)),
        ("variable name", Box::new(|w| w.coordinates[3].name = "renamed".into())),
        ("graph edge", Box::new(|w| w.graph.directed.push((0, 5)))),
        ("selection", Box::new(|w| w.selections.push(1))),
        ("slots", Box::new(|w| w.slots.actions.swap(0, 1))),
        ("source", Box::new(|w| w.source = "target".into())),
        ("search operations", Box::new(|w| w.search_operations -= 1)),
        ("evaluation limit", Box::new(|w| w.operation_limit -= 1)),
    ];
    for (label, edit) in &cases {
        let error = typed(consume(&mutated(&bytes, |w| edit(w), false)));
        assert_eq!(error, premises, "{label}");
    }
    // With the digest recomputed, the replay itself rejects a semantic change.
    let stale = |edit: &dyn Fn(&mut TemporalSequenceArtifactWire)| {
        consume(&mutated(&bytes, |w| edit(w), true))
    };
    assert_eq!(
        typed(stale(&|w| w.report.mean += 1e-9)),
        TemporalTransportArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(stale(&|w| w.report.sequence.swap(0, 1))),
        TemporalTransportArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(stale(&|w| w.report.rows[0].status = "supported_elsewhere".into())),
        TemporalTransportArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(stale(&|w| w.report.invariances[0].borrowed_from_source ^= true)),
        TemporalTransportArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(stale(&|w| w.report.time_varying_confounders.clear())),
        TemporalTransportArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(stale(&|w| w.report.point.probabilities[0] += 1e-9)),
        TemporalTransportArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(stale(&|w| w.provider = "transport.other_provider".into())),
        TemporalTransportArtifactError::ProviderMismatch("unknown provider")
    );
    // A changed law replays to another point.
    assert_eq!(
        typed(stale(&|w| w
            .laws
            .iter_mut()
            .find(|l| l.population == "target")
            .unwrap()
            .probabilities
            .reverse())),
        TemporalTransportArtifactError::ReportMismatch,
    );
    // The consumer's own limits bound every recorded one.
    let small = TemporalTransportConsumeLimits {
        budget: SearchLimits { operations: 5, depth: 256 },
        ..TemporalTransportConsumeLimits::default()
    };
    let refused = consume_temporal_transport_artifact(&bytes, small, &ctx());
    assert_eq!(typed(refused), TemporalTransportArtifactError::LimitsExceeded("search budget"));
    // A foreign version or feature marker never decodes.
    assert!(matches!(
        consume(&mutated(&bytes, |w| w.version = 2, false)),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert_eq!(
        typed(consume(&mutated(&bytes, |w| w.required_features = vec!["other".into()], false))),
        TemporalTransportArtifactError::UnsupportedSemantics("required features"),
    );
}

#[test]
fn a_mutated_horizon_or_history_refuses_on_replay_not_only_by_digest() {
    let (_, _, bytes) = exported(1.0, 0.0);
    // Even with a recomputed digest, horizon three is refused by the replay.
    let three = mutated(&bytes, |w| w.horizon = 3, true);
    match consume(&three) {
        Err(IoError::Refused { code, message }) => {
            assert_eq!(code, "route_not_supported");
            assert!(message.starts_with("temporal_transport.horizon"), "{message}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

#[test]
fn same_window_refresh_re_estimates_under_the_same_proof() {
    let (prepared, before, bytes_before) = exported(1.0, 0.0);
    // New laws for the same window: the source's outcome mechanism weights move.
    let mut source = source_scm();
    source.exo_p[7] = 0.5;
    let refreshed = prepared.refresh(laws(&source, &target_scm(), &[]), &ctx()).unwrap();
    let after = refreshed.estimate(&ctx()).unwrap();
    assert!((after.mean - before.mean).abs() > 1e-4, "the point must follow the new evidence");
    // The outcome mechanism is invariant, so the truth in the target moves with it.
    let mut target = target_scm();
    target.exo_p[7] = 0.5;
    assert!((after.mean - truth(&target, [1, 0])).abs() < 1e-12);
    // The proof is untouched and the artifact identity is unchanged.
    let bytes_after = refreshed.export(&after).unwrap();
    let (a, b) = (
        TemporalSequenceArtifactWire::decode(&bytes_before).unwrap(),
        TemporalSequenceArtifactWire::decode(&bytes_after).unwrap(),
    );
    assert_eq!(a.premises_digest, b.premises_digest);
    assert_eq!(crate::proof(&a), crate::proof(&b));
    assert_eq!(consume(&bytes_after).unwrap().mean.to_bits(), after.mean.to_bits());
}

fn proof(wire: &TemporalSequenceArtifactWire) -> Vec<u8> {
    antecedent_io::to_cbor(&wire.report.proof).unwrap()
}

#[test]
fn a_new_period_or_a_changed_window_needs_a_new_preparation() {
    let prepared = prepare(&sequence(1.0, 0.0));
    let refused = |result: Result<PreparedTemporalTransport, IoError>| match result {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    // A law over a third-period coordinate is outside the two-step window.
    let full = laws(&source_scm(), &target_scm(), &[]);
    let target = full.laws().iter().find(|l| l.population() == "target").unwrap();
    let mut axes = target.axes().to_vec();
    axes.push(DiscreteAxis {
        variable: v(6),
        values: std::sync::Arc::from([Value::f64(0.0), Value::f64(1.0)]),
    });
    let widened = ExactDiscreteLaw::try_new(
        "target",
        target.regime(),
        [],
        axes,
        target.probabilities().iter().flat_map(|p| [p / 2.0, p / 2.0]).collect::<Vec<_>>(),
        target.snapshot_identity(),
        LawTolerance::default(),
    )
    .unwrap();
    let mut all = vec![widened];
    all.extend(full.laws().iter().filter(|l| l.population() != "target").cloned());
    let (code, message) =
        refused(prepared.refresh(ExactTransportData::try_new(all, 4096).unwrap(), &ctx()));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("temporal_transport.horizon"), "{message}");
    // Dropping a source history experiment is a history outside support, not a new window.
    let (code, message) =
        refused(prepared.refresh(laws(&source_scm(), &target_scm(), &[(1, 1, 1)]), &ctx()));
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("temporal_transport.history_outside_support"), "{message}");
    // A different measurement window (no source experiments at all) needs a new preparation.
    let target_only = ExactTransportData::try_new(
        full.laws().iter().filter(|l| l.population() == "target").cloned().collect::<Vec<_>>(),
        4096,
    )
    .unwrap();
    let (code, message) = refused(prepared.refresh(target_only, &ctx()));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("temporal_transport.horizon"), "{message}");
    // A longer horizon never prepares at all.
    let mut three = sequence(1.0, 0.0);
    three.push(Value::f64(1.0));
    let error = StudyBuilder::temporal_transport_sequence(
        &spec(),
        &three,
        "source",
        "target",
        catalog(),
        BUDGET,
        laws(&source_scm(), &target_scm(), &[]),
        ExactEvaluationLimits::default(),
        &ctx(),
    );
    let (code, message) = refused(error);
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("temporal_transport.horizon"), "{message}");
}

#[test]
fn an_interval_request_refuses_with_estimator_inference_mismatch() {
    let prepared = prepare(&sequence(0.0, 1.0));
    match prepared.interval() {
        Err(IoError::Refused { code, message }) => {
            assert_eq!(code, "estimator_inference_mismatch");
            assert!(message.starts_with("temporal_transport.interval_requested"), "{message}");
        }
        other => panic!("expected estimator_inference_mismatch, got {other:?}"),
    }
    // The estimate itself is exact and point-only.
    let report = prepared.estimate(&ctx()).unwrap();
    assert_eq!(report.inference_claim, "point_only");
}

#[test]
fn the_prepared_sequence_estimates_the_point_and_reports_its_premises() {
    let prepared = prepare(&sequence(0.0, 1.0));
    let report = prepared.estimate(&ctx()).unwrap();
    assert!((report.mean - truth(&target_scm(), [0, 1])).abs() < 1e-12);
    assert_eq!((report.horizon, report.inference_claim), (2, "point_only"));
    assert_eq!(report.time_varying_confounders, vec![v(common::temporal_fixture::L2)]);
    assert!(report.support.rows.iter().all(|row| row.status == "supported"));
    // Estimation reuses the retained plan: a second call is bit for bit the first.
    let again = prepared.estimate(&ctx()).unwrap();
    assert_eq!(again.mean.to_bits(), report.mean.to_bits());
}

#[test]
fn a_swapped_catalog_or_law_snapshot_fails_the_data_identity() {
    let (_, _, bytes) = exported(1.0, 0.0);
    let identity = TemporalTransportArtifactError::DataIdentityMismatch;
    // The catalog and the laws are data identity outside the premises digest: the
    // premises still verify, but a swapped snapshot is caught by the data digest
    // even though it leaves the replayed point unchanged.
    let cases: Vec<(&str, Edit)> = vec![
        (
            "catalog snapshot",
            Box::new(|w| w.catalog.bindings[1].snapshot_identity = "swapped".into()),
        ),
        (
            "catalog dataset",
            Box::new(|w| w.catalog.bindings[0].dataset_identity = Some("other".into())),
        ),
        ("law snapshot", Box::new(|w| w.laws[0].snapshot = "swapped".into())),
        ("digest", Box::new(|w| w.data_digest.push('0'))),
    ];
    for (label, edit) in &cases {
        let error = typed(consume(&mutated(&bytes, |w| edit(w), false)));
        assert_eq!(error, identity, "{label}");
    }
    // A catalog regime removed is a swapped catalog.
    let removed = typed(consume(&mutated(
        &bytes,
        |w| {
            w.catalog.regimes.pop();
            w.catalog.bindings.pop();
        },
        false,
    )));
    assert_eq!(removed, identity);
    // A refresh keeps the catalog's snapshot bindings (the evaluator refuses laws
    // whose snapshot differs from the binding), so both digests stay while the
    // replay binds the new numbers.
    let (prepared, before, _) = exported(1.0, 0.0);
    let mut source = source_scm();
    source.exo_p[7] = 0.5;
    let refreshed = prepared.refresh(laws(&source, &target_scm(), &[]), &ctx()).unwrap();
    let after = refreshed.estimate(&ctx()).unwrap();
    let (a, b) = (
        TemporalSequenceArtifactWire::decode(&prepared.export(&before).unwrap()).unwrap(),
        TemporalSequenceArtifactWire::decode(&refreshed.export(&after).unwrap()).unwrap(),
    );
    assert_eq!((a.premises_digest, a.data_digest), (b.premises_digest, b.data_digest));
    assert_eq!(
        consume(&refreshed.export(&after).unwrap()).unwrap().mean.to_bits(),
        after.mean.to_bits()
    );
}

#[test]
fn a_report_of_another_preparation_is_not_exported() {
    let (prepared, report, _) = exported(1.0, 0.0);
    let other = prepare(&sequence(0.0, 1.0));
    let foreign = other.estimate(&ctx()).unwrap();
    match prepared.export(&foreign) {
        Err(IoError::TemporalTransport(TemporalTransportArtifactError::ForeignReport)) => {}
        result => panic!("expected a foreign report refusal, got {:?}", result.map(|_| ())),
    }
    // Even a report of the same premises from a refreshed preparation is foreign.
    let refreshed = prepared.refresh(laws(&source_scm(), &target_scm(), &[]), &ctx()).unwrap();
    assert!(matches!(
        refreshed.export(&report),
        Err(IoError::TemporalTransport(TemporalTransportArtifactError::ForeignReport))
    ));
    assert!(prepared.export(&report).is_ok());
}

#[test]
fn a_refresh_that_moves_the_fixed_initial_state_needs_a_new_preparation() {
    let prepared = prepare(&sequence(1.0, 0.0));
    let refused = |result: Result<PreparedTemporalTransport, IoError>| match result {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    // The target's baseline law moves: initial-state uncertainty, not fresh evidence.
    let mut moved = target_scm();
    moved.exo_p[0] = 0.5;
    let (code, message) = refused(prepared.refresh(laws(&source_scm(), &moved, &[]), &ctx()));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("temporal_transport.horizon"), "{message}");
    assert!(message.contains("initial-state"), "{message}");
    // The same initial state with other target mechanisms is a same-window refresh.
    let mut later = target_scm();
    later.exo_p[7] = 0.4;
    later.exo_p[4] = 0.8;
    assert!(prepared.refresh(laws(&source_scm(), &later, &[]), &ctx()).is_ok());
}

#[test]
fn a_template_route_and_a_three_level_alphabet_round_trip_through_the_artifact() {
    use common::temporal_fixture::{categorical, template};
    // The specification unrolled from a lagged TemporalDag template (ADR 0021).
    let (source, target) = (template::source(), template::target());
    let prepared = StudyBuilder::temporal_transport_sequence(
        &template::spec(),
        &sequence(1.0, 0.0),
        "source",
        "target",
        template::catalog(),
        BUDGET,
        template::laws(&source, &target),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    let report = prepared.estimate(&ctx()).unwrap();
    assert!((report.mean - target.truth([1, 0])).abs() < 1e-12);
    let consumed = consume(&prepared.export(&report).unwrap()).unwrap();
    assert_eq!(consumed.mean.to_bits(), report.mean.to_bits());
    assert_eq!(consumed.support, report.support);
    // Three action levels round-trip too.
    let (source, target) = (categorical::source(), categorical::target());
    let prepared = StudyBuilder::temporal_transport_sequence(
        &categorical::spec(),
        &sequence(2.0, 1.0),
        "source",
        "target",
        categorical::evidence(),
        BUDGET,
        categorical::laws(&source, &target),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    let report = prepared.estimate(&ctx()).unwrap();
    assert!((report.mean - target.truth([2, 1])).abs() < 1e-12);
    let consumed = consume(&prepared.export(&report).unwrap()).unwrap();
    assert_eq!(consumed.mean.to_bits(), report.mean.to_bits());
}

#[test]
fn the_io_wire_replays_bytes_under_its_own_limits_and_the_stored_report() {
    let (_, report, bytes) = exported(0.0, 1.0);
    let defaults = TemporalTransportConsumeLimits::default();
    // Bytes to a verified (wire, report) pair through the io type, no facade.
    let (wire, replayed) =
        TemporalSequenceArtifactWire::consume_with_limits(&bytes, defaults, &ctx()).unwrap();
    assert_eq!(replayed.mean.to_bits(), report.mean.to_bits());
    assert_eq!(wire.report.mean.to_bits(), report.mean.to_bits());
    assert_eq!(wire.premises_digest, temporal_sequence_identity(&wire).unwrap());
    let typed_err = |limits: TemporalTransportConsumeLimits| {
        match TemporalSequenceArtifactWire::consume_with_limits(&bytes, limits, &ctx()) {
            Err(IoError::TemporalTransport(error)) => error,
            other => panic!("expected a typed refusal, got {:?}", other.map(|_| ())),
        }
    };
    let exceeded = TemporalTransportArtifactError::LimitsExceeded;
    // Each consumer bound refuses on its own, whatever the artifact records.
    assert_eq!(
        typed_err(TemporalTransportConsumeLimits {
            budget: SearchLimits { depth: 0, ..defaults.budget },
            ..defaults
        }),
        exceeded("search budget")
    );
    assert_eq!(
        typed_err(TemporalTransportConsumeLimits {
            evaluation: ExactEvaluationLimits { operations: 1, ..defaults.evaluation },
            ..defaults
        }),
        exceeded("evaluation limits")
    );
    assert_eq!(
        typed_err(TemporalTransportConsumeLimits { max_support_rows: 1, ..defaults }),
        exceeded("support rows")
    );
    assert_eq!(
        typed_err(TemporalTransportConsumeLimits { max_laws: 0, ..defaults }),
        exceeded("law count")
    );
    // The data digest binds snapshot identities, not numbers: an edited law with the
    // premises untouched is caught by the replay against the stored report.
    let edited = mutated(
        &bytes,
        |w| w.laws.iter_mut().find(|l| l.population == "target").unwrap().probabilities.reverse(),
        false,
    );
    match TemporalSequenceArtifactWire::consume_with_limits(&edited, defaults, &ctx()) {
        Err(IoError::TemporalTransport(error)) => {
            assert_eq!(error, TemporalTransportArtifactError::ReportMismatch);
        }
        other => panic!("expected ReportMismatch, got {:?}", other.map(|_| ())),
    }
    // Undecodable bytes fail at decode, not as a typed artifact refusal.
    assert!(!matches!(
        TemporalSequenceArtifactWire::consume_with_limits(
            &bytes[..bytes.len() / 2],
            defaults,
            &ctx()
        ),
        Err(IoError::TemporalTransport(_))
    ));
}

/// Re-sealed data-identity mutations: with both digests valid again, a law
/// whose cells change, or a law moved to another snapshot than its catalog
/// binding, still fails verification.
#[test]
fn resealed_data_identity_mutations_fail_replay() {
    let (_, _, bytes) = exported(1.0, 0.0);
    let original = TemporalSequenceArtifactWire::decode(&bytes).unwrap();
    assert_eq!(original.data_digest, original.expected_data_digest().unwrap());
    let resealed = |edit: &dyn Fn(&mut TemporalSequenceArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        wire.data_digest = wire.expected_data_digest().unwrap();
        wire.premises_digest = temporal_sequence_identity(&wire).unwrap();
        wire.export().unwrap()
    };
    // The law's cells: the replayed report differs.
    let cells = consume(&resealed(&|w| {
        w.laws.iter_mut().find(|l| l.population == "target").unwrap().probabilities.reverse();
    }));
    assert_eq!(typed(cells), TemporalTransportArtifactError::ReportMismatch);
    // A law moved to a snapshot its catalog binding does not name.
    let moved = consume(&resealed(&|w| {
        w.laws.iter_mut().find(|l| l.population == "target").unwrap().snapshot = "elsewhere".into();
    }));
    assert!(moved.is_err(), "a law outside its catalog binding must not replay");
}

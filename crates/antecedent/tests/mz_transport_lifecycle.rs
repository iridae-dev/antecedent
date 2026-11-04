//! Prepared lifecycle, refresh and independent artifact consumption of a
//! multi-source limited-experiment (`TR^mz`) formula.
//!
//! The fixture is R-443 Figure 1(c,d), shared with the estimate-crate tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;

use antecedent::{PreparedMzTransport, StudyBuilder, consume_mz_transport_artifact};
use antecedent_core::{EvidenceCatalog, ExecutionContext, Value};
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, bind_mz_transport_catalog,
    decide_mz_transport,
};
use antecedent_io::IoError;
use antecedent_io::contract_section::CalibrationStatus;
use antecedent_io::mz_transport_artifact::{
    MZ_NOMINAL_INTERVAL, MZ_POINT_ONLY, MZ_WITHHELD, MzTransportArtifactError,
    MzTransportArtifactWire, MzTransportConsumeLimits, MzUncertaintyWire,
};
use common::mz_fixture::{
    SharedTable, X, Y, empirical, evidence, graph, query, sources, target_scm,
    with_conflicting_a_trial, with_shared_b_trial, with_studies,
};
use common::z_scm::{risk_of, vid};

fn request(x: bool) -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(x))])
}

/// Decide and bind, then prepare `requests`; every builder input is moved into
/// the prepared state and dropped here.
fn prepare_requests(
    catalog: &EvidenceCatalog,
    data: ExactTransportData,
    counted: bool,
    requests: Vec<Assignment>,
) -> PreparedMzTransport {
    let ctx = ExecutionContext::for_tests(7);
    let MzTransportDecision::Identified { derivation, .. } = decide_mz_transport(
        &graph(),
        &query(sources()),
        catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("the complementary catalog identifies");
    };
    let functional = bind_mz_transport_catalog(&graph(), &derivation, catalog).unwrap();
    let build =
        if counted { StudyBuilder::mz_transport_empirical } else { StudyBuilder::mz_transport };
    build(
        graph(),
        functional,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        data,
        requests,
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap()
}

fn prepare(
    catalog: &EvidenceCatalog,
    data: ExactTransportData,
    counted: bool,
) -> PreparedMzTransport {
    prepare_requests(catalog, data, counted, vec![request(true)])
}

#[test]
fn estimate_runs_the_retained_plan_after_builder_disposal() {
    let prepared = {
        let (catalog, data) = evidence();
        prepare(&catalog, data, false)
    };
    assert_eq!(prepared.plans().len(), 1);
    let result = prepared.estimate(&ExecutionContext::for_tests(1)).unwrap();
    let truth = target_scm().risk(&[(X, 1)], Y);
    assert!((risk_of(result.distribution()) - truth).abs() < 1e-12);
    assert_eq!(result.uncertainty(), &MzUncertaintyWire::point_only());
    assert!(result.contrasts().is_empty());
    assert_eq!(
        prepared.theorem_scope().family,
        antecedent_core::TheoremScope::mz_transportability().family
    );
}

#[test]
fn contrasts_run_through_prepare_estimate_refresh_export_and_consume() {
    let (catalog, exact) = evidence();
    let prepared =
        prepare_requests(&catalog, exact.clone(), false, vec![request(false), request(true)]);
    assert_eq!(prepared.plans().len(), 2);
    let ctx = ExecutionContext::for_tests(8);
    let result = prepared.estimate(&ctx).unwrap();
    let (p0, p1) = (risk_of(&result.distributions()[0]), risk_of(&result.distributions()[1]));
    let truth = |x: u8| target_scm().risk(&[(X, x)], Y);
    assert!((p0 - truth(0)).abs() < 1e-12 && (p1 - truth(1)).abs() < 1e-12);
    let contrast = result.contrasts()[0];
    assert_eq!((contrast.request, contrast.outcome), (1, vid(Y).raw()));
    assert!((contrast.estimate - (truth(1) - truth(0))).abs() < 1e-12);
    // Refresh keeps every request; the contrast moves with the laws.
    let refreshed = prepared.refresh(empirical(&exact, 5_000.0), &ctx).unwrap();
    assert_eq!(refreshed.requests().len(), prepared.requests().len());
    assert_eq!(refreshed.requests()[1].entries(), prepared.requests()[1].entries());
    let moved = refreshed.estimate(&ctx).unwrap();
    assert_ne!(moved.contrasts()[0].estimate.to_bits(), contrast.estimate.to_bits());
    // The consumer recomputes every point and contrast bit for bit.
    let bytes = result.export(&prepared, &ctx).unwrap();
    let consumed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(consumed.distributions().len(), 2);
    assert_eq!(consumed.contrasts()[0].estimate.to_bits(), contrast.estimate.to_bits());
    let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
    wire.contrasts[0].estimate += 1e-12;
    assert!(matches!(refused(&wire.export().unwrap()), MzTransportArtifactError::ContrastMismatch));
    let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
    wire.results[1].probabilities[0] += 1e-9;
    assert!(matches!(refused(&wire.export().unwrap()), MzTransportArtifactError::PointMismatch));
}

#[test]
fn exported_point_is_recomputed_by_an_independent_consumer() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, false);
    let ctx = ExecutionContext::for_tests(2);
    let result = prepared.estimate(&ctx).unwrap();
    let bytes = result.export(&prepared, &ctx).unwrap();
    drop(prepared);
    let consumed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(consumed.distribution().probabilities, result.distribution().probabilities);
    // The artifact names the source of every cited regime and the search receipt.
    let wire = MzTransportArtifactWire::decode(&bytes).unwrap();
    let populations = wire.bindings.iter().map(|(_, p)| p.as_str()).collect::<Vec<_>>();
    assert!(populations.contains(&"a") && populations.contains(&"b"));
    assert_eq!(wire.proof.route, "combined");
    assert!(
        wire.proof
            .stages
            .iter()
            .any(|(stage, outcome)| stage == "source:a" && outcome == "obstruction")
    );
}

fn refused(bytes: &[u8]) -> MzTransportArtifactError {
    match consume_mz_transport_artifact(
        bytes,
        MzTransportConsumeLimits::default(),
        &ExecutionContext::for_tests(3),
    ) {
        Err(IoError::MzTransport(error)) => error,
        other => {
            panic!("expected a typed mz refusal, got {:?}", other.map(|r| r.distribution().clone()))
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for a mutated artifact fails independent consumption"
)]
fn a_mutated_artifact_fails_independent_consumption() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, false);
    let ctx = ExecutionContext::for_tests(4);
    let names = ["z1", "x", "z2", "y"].map(String::from);
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &names, &ctx).unwrap();
    let original = MzTransportArtifactWire::decode(&bytes).unwrap();
    let mutate = |edit: &dyn Fn(&mut MzTransportArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        wire.export().unwrap()
    };
    // Result body.
    let point = mutate(&|w| w.results[0].probabilities[0] += 1e-9);
    assert!(matches!(refused(&point), MzTransportArtifactError::PointMismatch));
    // Theorem premises and source identity are covered by the premises digest.
    let rules = mutate(&|w| w.proof.rules.pop().map(|_| ()).unwrap());
    assert!(matches!(refused(&rules), MzTransportArtifactError::PremisesMismatch));
    let source = mutate(&|w| w.query.sources[1].population = "c".into());
    assert!(matches!(refused(&source), MzTransportArtifactError::PremisesMismatch));
    // So are the search and evaluation limits the proof was decided and run under.
    let search = mutate(&|w| w.search_operations -= 1);
    assert!(matches!(refused(&search), MzTransportArtifactError::PremisesMismatch));
    let depth = mutate(&|w| w.search_depth -= 1);
    assert!(matches!(refused(&depth), MzTransportArtifactError::PremisesMismatch));
    let evaluation = mutate(&|w| w.operation_limit -= 1);
    assert!(matches!(refused(&evaluation), MzTransportArtifactError::PremisesMismatch));
    // And the variable-name mapping: relabelling coordinates is not silent.
    let swapped = mutate(&|w| w.variable_names.swap(0, 2));
    assert!(matches!(refused(&swapped), MzTransportArtifactError::PremisesMismatch));
    assert_eq!(original.check_variable_names(&names), Ok(()));
    let mut relabelled = names.clone();
    relabelled.swap(0, 2);
    assert_eq!(
        original.check_variable_names(&relabelled),
        Err(MzTransportArtifactError::NamesMismatch)
    );
    // A consistent rewrite of the premises must still reproduce the decision.
    let forged = mutate(&|w| {
        w.proof.stages.clear();
        w.premises_digest = w.expected_premises_digest().unwrap();
    });
    assert!(matches!(refused(&forged), MzTransportArtifactError::ProofMismatch(_)));
    // Evidence the premises and data digests do not bind is sealed by its own digest: a
    // resealed premise edit that still replays leaves the evidence digest stale.
    let resealed = mutate(&|w| {
        w.operation_limit -= 1;
        w.premises_digest = w.expected_premises_digest().unwrap();
    });
    assert!(matches!(refused(&resealed), MzTransportArtifactError::EvidenceMismatch));
    // Source-specific bindings.
    let bindings = mutate(&|w| w.bindings.reverse());
    assert!(matches!(refused(&bindings), MzTransportArtifactError::BindingMismatch));
    // Data identity: snapshot ids are bound by the data-identity digest ...
    let law_snapshot = mutate(&|w| w.laws[0].snapshot = "other".into());
    assert!(matches!(refused(&law_snapshot), MzTransportArtifactError::DataIdentityMismatch));
    let catalog_snapshot = mutate(&|w| {
        let mut catalog = w.catalog.to_catalog().unwrap();
        let mut bindings = catalog.bindings.to_vec();
        bindings[0].snapshot_identity = "other".into();
        catalog.bindings = bindings.into();
        w.catalog =
            antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&catalog);
    });
    assert!(matches!(refused(&catalog_snapshot), MzTransportArtifactError::DataIdentityMismatch));
    // ... and a consistent rewrite of the digest still has to bind: the law no
    // longer matches its catalog binding.
    let rebound = mutate(&|w| {
        w.laws[0].snapshot = "other".into();
        w.data_digest = w.expected_data_digest().unwrap();
    });
    assert!(matches!(
        consume_mz_transport_artifact(&rebound, MzTransportConsumeLimits::default(), &ctx),
        Err(IoError::Refused { .. })
    ));
    // Catalog contents beyond the snapshot ids are bound by the data digest too:
    // the sampling design, the dependence declaration and the target sampling.
    let sampling = mutate(&|w| {
        let mut catalog = w.catalog.to_catalog().unwrap();
        let mut bindings = catalog.bindings.to_vec();
        bindings[0].sampling = antecedent_core::SamplingDesign::Clustered;
        bindings[0].dependence = antecedent_core::DependenceGroup::UnknownDependence;
        catalog.bindings = bindings.into();
        w.catalog =
            antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&catalog);
    });
    assert_ne!(MzTransportArtifactWire::decode(&sampling).unwrap().catalog, original.catalog);
    assert!(matches!(refused(&sampling), MzTransportArtifactError::DataIdentityMismatch));
    let target = mutate(&|w| {
        w.catalog.target_sampling = Some("convenience_sample".into());
    });
    assert!(matches!(refused(&target), MzTransportArtifactError::DataIdentityMismatch));
    // The search receipt is part of the checked proof: the consumer replays under
    // the stored limits and compares every field it can reproduce exactly.
    let consistent = |edit: &dyn Fn(&mut MzTransportArtifactWire)| {
        mutate(&|w| {
            edit(w);
            w.premises_digest = w.expected_premises_digest().unwrap();
        })
    };
    assert!(original.proof.search.operations_consumed > 0);
    for edit in [
        (&|w: &mut MzTransportArtifactWire| w.proof.search.operations_consumed += 1)
            as &dyn Fn(&mut MzTransportArtifactWire),
        &|w| w.proof.search.depth_reached += 1,
        &|w| w.proof.search.explored.pop().map(|_| ()).unwrap(),
        &|w| w.proof.search.unevaluated.push("stage:extra".into()),
        // Limits the decision cannot fit inside never replay to the stored receipt.
        &|w| {
            w.proof.search.operations_limit = w.proof.search.operations_consumed - 1;
            w.search_operations = w.proof.search.operations_limit;
        },
        &|w| w.proof.search.memory_limit_bytes = 1,
    ] {
        let edited = consistent(edit);
        assert!(matches!(refused(&edited), MzTransportArtifactError::ProofMismatch(_)));
    }
    // The stored limits and the proof's receipt must agree.
    let split = consistent(&|w| w.search_operations -= 1);
    assert!(matches!(refused(&split), MzTransportArtifactError::UnsupportedSemantics(_)));
    // A stored memory cap above the consumer's maximum refuses.
    let wide = consistent(&|w| w.proof.search.memory_limit_bytes += 1);
    assert!(matches!(refused(&wide), MzTransportArtifactError::LimitsExceeded(_)));
    // Every record pairing is `(code, detail)`: a proof that does not replay is
    // `transport_not_certified` / `mz_transport.invalid_derivation`.
    let forged = consistent(&|w| w.proof.search.operations_consumed += 1);
    let MzTransportArtifactError::ProofMismatch(inner) = refused(&forged) else { unreachable!() };
    assert_eq!(
        antecedent_identify::mz_transport_refusal(&inner),
        Some((
            "transport_not_certified",
            antecedent_core::reason_details::mz_transport::INVALID_DERIVATION
        ))
    );
    assert_eq!(
        MzTransportArtifactError::ProofMismatch(inner).refusal(),
        Some((
            "transport_not_certified",
            antecedent_core::reason_details::mz_transport::INVALID_DERIVATION
        ))
    );
    // Uncertainty bookkeeping cannot claim an interval exact laws do not have.
    let claimed = mutate(&|w| w.uncertainty.status = MZ_NOMINAL_INTERVAL.into());
    assert!(matches!(refused(&claimed), MzTransportArtifactError::UncertaintyMismatch(_)));
    let relabelled_reason =
        mutate(&|w| w.uncertainty.reason = "estimator_grid_not_measured".into());
    assert!(matches!(
        refused(&relabelled_reason),
        MzTransportArtifactError::UncertaintyMismatch(_)
    ));
    // A consumer with smaller search limits than the producer refuses.
    let small = MzTransportConsumeLimits {
        search: antecedent_core::SearchLimits { operations: 10, depth: 24 },
        ..MzTransportConsumeLimits::default()
    };
    assert!(matches!(
        consume_mz_transport_artifact(&bytes, small, &ExecutionContext::for_tests(3)),
        Err(IoError::MzTransport(MzTransportArtifactError::LimitsExceeded(_)))
    ));
}

#[test]
fn a_consumer_that_cannot_afford_the_stored_memory_cap_refuses_with_a_limits_error() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, false);
    let ctx = ExecutionContext::for_tests(4);
    let bytes = prepared.estimate(&ctx).unwrap().export(&prepared, &ctx).unwrap();
    let stored = MzTransportArtifactWire::decode(&bytes).unwrap().proof.search.memory_limit_bytes;
    let mut tight = ExecutionContext::for_tests(4);
    tight.set_memory_limit(Some(stored - 1));
    // The replay runs under the producer's cap; the consumer's own hard limit is
    // below it, so the artifact is refused up front as a limits problem the caller
    // can retry with a larger budget, never as an invalid proof.
    let refusal =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &tight)
            .map(|_| ())
            .unwrap_err();
    assert!(
        matches!(
            refusal,
            IoError::MzTransport(MzTransportArtifactError::LimitsExceeded("search memory limit"))
        ),
        "{refusal:?}"
    );
    // With exactly the stored cap the same artifact consumes and replays.
    tight.set_memory_limit(Some(stored));
    let replayed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &tight)
            .expect("a consumer that can afford the cap verifies the artifact");
    assert_eq!(
        risk_of(replayed.distribution()).to_bits(),
        risk_of(prepared.estimate(&ctx).unwrap().distribution()).to_bits()
    );
}

#[test]
fn counted_laws_report_the_interval_with_its_calibration_annotation() {
    let (catalog, exact) = evidence();
    let data = empirical(&exact, 40_000.0);
    let ctx = ExecutionContext::for_tests(5);
    let prepared = prepare(&with_studies(&catalog), data, true);
    let result = prepared.estimate(&ctx).unwrap();
    // The point is returned ...
    assert!((risk_of(result.distribution()) - target_scm().risk(&[(X, 1)], Y)).abs() < 1e-2);
    // ... with the nominal pointwise joint-bootstrap interval, seeded by the execution.
    let uncertainty = result.uncertainty();
    assert_eq!(uncertainty.status, MZ_NOMINAL_INTERVAL);
    assert!(uncertainty.available());
    assert_eq!(uncertainty.seed, Some(5));
    assert_eq!(uncertainty.method.as_deref(), Some(antecedent_estimate::PERCENTILE_BOOTSTRAP));
    assert_eq!(uncertainty.coverage_target, Some(0.95));
    let (outcome, lower, upper) = uncertainty.mean_intervals[0][0];
    assert!(lower <= upper, "outcome {outcome}");
    // Calibration annotates the interval; it never withholds it.
    assert!(
        matches!(
            result.calibration().status,
            CalibrationStatus::Calibrated
                | CalibrationStatus::OutsideMeasuredScope
                | CalibrationStatus::MeasuredBelowNominal
                | CalibrationStatus::NotMeasured
        ),
        "{:?}",
        result.calibration()
    );
    // The artifact carries the interval and a consumer recomputes it bit for bit.
    let bytes = result.export(&prepared, &ctx).unwrap();
    let consumed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(consumed.uncertainty(), uncertainty);
    assert_eq!(consumed.calibration(), result.calibration());
    // Moved bounds, another seed or another replicate count do not recompute.
    let original = MzTransportArtifactWire::decode(&bytes).unwrap();
    let rejects = |edit: &dyn Fn(&mut MzUncertaintyWire)| {
        let mut wire = original.clone();
        edit(&mut wire.uncertainty);
        assert!(matches!(
            refused(&wire.export().unwrap()),
            MzTransportArtifactError::UncertaintyMismatch(_)
        ));
    };
    rejects(&|u| u.mean_intervals[0][0].1 -= 1e-12);
    rejects(&|u| u.mean_intervals[0][0].2 += 1e-12);
    rejects(&|u| u.seed = Some(6));
    rejects(&|u| u.replicates_requested += 1);
    // Exact laws carry no interval.
    let exact_prepared = prepare(&with_studies(&catalog), exact.clone(), false);
    let exact_result = exact_prepared.estimate(&ctx).unwrap();
    assert_eq!(exact_result.uncertainty().status, MZ_POINT_ONLY);
    assert_eq!(exact_result.calibration().status, CalibrationStatus::NotMeasured);
    // Undeclared dependence withholds the interval for a stated data reason.
    let undeclared = prepare(&catalog, empirical(&exact, 40_000.0), true).estimate(&ctx).unwrap();
    assert_eq!(undeclared.uncertainty().status, MZ_WITHHELD);
    assert_eq!(undeclared.uncertainty().reason, "sampling_dependence_unknown");
    assert_eq!(undeclared.calibration().status, CalibrationStatus::NotMeasured);
}

#[test]
fn shared_dataset_regimes_always_return_the_point() {
    let (catalog, exact) = evidence();
    let counted = empirical(&exact, 40_000.0);
    let ctx = ExecutionContext::for_tests(12);
    let truth = target_scm().risk(&[(X, 1)], Y);
    let cases = [
        (with_shared_b_trial(&catalog, &counted, SharedTable::Identical), None),
        (with_shared_b_trial(&catalog, &counted, SharedTable::Margin), None),
        (
            (with_conflicting_a_trial(&catalog), counted.clone()),
            Some("transport.unsupported_dependence"),
        ),
    ];
    for ((catalog, data), dependence) in cases {
        let prepared = prepare(&catalog, data, true);
        let result = prepared.estimate(&ctx).unwrap();
        assert!((risk_of(result.distribution()) - truth).abs() < 1e-2);
        match dependence {
            None => assert!(result.uncertainty().available()),
            Some(reason) => {
                assert_eq!(result.uncertainty().status, MZ_WITHHELD);
                assert_eq!(result.uncertainty().reason, reason);
            }
        }
        // The consumer re-derives the same dependence reason.
        let bytes = result.export(&prepared, &ctx).unwrap();
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
        let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
        wire.uncertainty.reason = "estimator_grid_not_measured_tampered".into();
        assert!(matches!(
            refused(&wire.export().unwrap()),
            MzTransportArtifactError::UncertaintyMismatch(_)
        ));
    }
}

#[test]
fn refresh_keeps_the_proof_and_refuses_changed_evidence_identity() {
    let (catalog, exact) = evidence();
    let studies = with_studies(&catalog);
    let prepared = prepare(&studies, empirical(&exact, 40_000.0), true);
    let ctx = ExecutionContext::for_tests(6);
    let before = prepared.estimate(&ctx).unwrap();
    // Same snapshots, new tables: the proof is untouched and the point moves.
    let refreshed = prepared.refresh(empirical(&exact, 5_000.0), &ctx).unwrap();
    assert_eq!(
        refreshed.functional().derivation().to_record(),
        prepared.functional().derivation().to_record()
    );
    assert_eq!(refreshed.functional().cited_regimes(), prepared.functional().cited_regimes());
    let after = refreshed.estimate(&ctx).unwrap();
    assert_ne!(after.distribution().probabilities, before.distribution().probabilities);
    // A law under another snapshot is evidence the frozen catalog never bound.
    let relabelled = exact
        .laws()
        .iter()
        .map(|law| {
            ExactDiscreteLaw::try_new(
                law.population(),
                law.regime(),
                law.interventions().to_vec(),
                law.axes().to_vec(),
                law.probabilities().to_vec(),
                format!("{}-new", law.snapshot_identity()),
                law.tolerance(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let relabelled = ExactTransportData::try_new(relabelled, exact.max_support_rows()).unwrap();
    // A law for another source population and regime is not evidence the frozen
    // catalog bound either: changed source or regime identity needs re-preparation.
    let mut foreign = exact.laws().to_vec();
    let law = &foreign[1];
    foreign[1] = ExactDiscreteLaw::try_new(
        "c",
        antecedent_core::RegimeId::from_raw(99),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap();
    let foreign = ExactTransportData::try_new(foreign, exact.max_support_rows()).unwrap();
    let exact_prepared = prepare(&studies, exact, false);
    let relabelled_error = exact_prepared.refresh(relabelled, &ctx).map(|_| ()).unwrap_err();
    let foreign_error = exact_prepared.refresh(foreign, &ctx).map(|_| ()).unwrap_err();
    // An empirical preparation refuses tables without counts.
    let counts_error =
        prepared.refresh(exact_prepared.data().clone(), &ctx).map(|_| ()).unwrap_err();
    let message = |error: IoError| match error {
        IoError::Refused { code, message } => (code, message),
        other => panic!("expected a coded refusal, got {other:?}"),
    };
    assert_eq!(
        message(relabelled_error),
        (
            "transport_missing_provider",
            "exact provider snapshot does not match catalog binding".to_owned()
        )
    );
    assert_eq!(
        message(foreign_error),
        (
            "transport_missing_provider",
            "exact provider names an unknown evidence regime".to_owned()
        )
    );
    let (code, text) = message(counts_error);
    assert_eq!(code, "transport_missing_provider");
    assert!(
        text.starts_with(antecedent_core::reason_details::mz_transport::EMPIRICAL_COUNTS_REQUIRED),
        "{text}"
    );
}

#[test]
fn preparation_refusals_carry_their_recorded_code_and_detail() {
    let (catalog, exact) = evidence();
    let ctx = ExecutionContext::for_tests(11);
    let MzTransportDecision::Identified { derivation, .. } = decide_mz_transport(
        &graph(),
        &query(sources()),
        &catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("the complementary catalog identifies");
    };
    let functional = bind_mz_transport_catalog(&graph(), &derivation, &catalog).unwrap();
    let pair = |error: IoError| match error {
        IoError::Refused { code, message } => {
            let detail = message.split(':').next().unwrap().to_owned();
            assert_eq!(IoError::Refused { code, message }.reason_code(), Some(code));
            (code, detail)
        }
        other => panic!("expected a coded refusal, got {other:?}"),
    };
    // A functional decided on another graph.
    let mut other_graph = graph();
    other_graph
        .insert_directed(
            antecedent_graph::DenseNodeId::from_raw(0),
            antecedent_graph::DenseNodeId::from_raw(3),
        )
        .unwrap();
    let mismatch = StudyBuilder::mz_transport(
        other_graph,
        functional.clone(),
        MZ_TRANSPORT_DEFAULT_LIMITS,
        exact.clone(),
        vec![request(true)],
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        pair(mismatch),
        (
            "invalid_argument",
            antecedent_core::reason_details::mz_transport::FUNCTIONAL_GRAPH_MISMATCH.to_owned()
        )
    );
    // An empirical preparation over laws without counts.
    let counts = StudyBuilder::mz_transport_empirical(
        graph(),
        functional,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        exact,
        vec![request(true)],
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        pair(counts),
        (
            "transport_missing_provider",
            antecedent_core::reason_details::mz_transport::EMPIRICAL_COUNTS_REQUIRED.to_owned()
        )
    );
}

#[test]
fn the_facade_error_reports_the_recorded_top_level_code() {
    use antecedent::CausalError;
    use antecedent_identify::IdentificationError;
    let proof = CausalError::Serialization(IoError::MzTransport(
        MzTransportArtifactError::ProofMismatch(IdentificationError::invalid_derivation(
            antecedent_core::reason_details::mz_transport::INVALID_DERIVATION,
        )),
    ));
    assert_eq!(proof.reason_code(), Some("transport_not_certified"));
    let bound = CausalError::Identify(IdentificationError::UnsupportedInput {
        code: antecedent_core::reason_details::mz_transport::BOUNDS_EXCEEDED,
    });
    assert_eq!(bound.reason_code(), Some("route_not_supported"));
    let invalid = CausalError::Identify(IdentificationError::invalid_input(
        "mz_transport.invalid_query: source populations must be distinct",
    ));
    assert_eq!(invalid.reason_code(), Some("invalid_argument"));
}

/// Compatibility: a reader refuses any other format version before reading the
/// payload (so an older reader never half-reads a newer artifact, and an older
/// one never reads this), and refuses a required feature it does not know.
#[test]
fn another_format_version_or_an_unknown_required_feature_is_refused() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, false);
    let ctx = ExecutionContext::for_tests(4);
    let names = ["z1", "x", "z2", "y"].map(String::from);
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &names, &ctx).unwrap();
    let original = MzTransportArtifactWire::decode(&bytes).unwrap();
    for version in [0, antecedent_io::mz_transport_artifact::MZ_TRANSPORT_ARTIFACT_VERSION + 1] {
        let mut wire = original.clone();
        wire.version = version;
        assert!(matches!(
            consume_mz_transport_artifact(
                &wire.export().unwrap(),
                MzTransportConsumeLimits::default(),
                &ctx
            ),
            Err(IoError::UnsupportedVersion { version: v }) if v == version
        ));
    }
    let mut wire = original.clone();
    wire.required_features.push("future_semantics_v9".into());
    assert_eq!(
        refused(&wire.export().unwrap()),
        MzTransportArtifactError::UnsupportedSemantics("required features")
    );
    let mut wire = original;
    wire.required_features.clear();
    assert_eq!(
        refused(&wire.export().unwrap()),
        MzTransportArtifactError::UnsupportedSemantics("required features")
    );
    assert!(
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).is_ok()
    );
    // Cross-format: the mz artifact (version 2) is refused by the
    // z-transport reader (version 4) by version, before its payload is read.
    assert!(matches!(
        antecedent_io::z_transport_artifact::ZTransportArtifactWire::decode(&bytes),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert!(antecedent::consume_z_transport_artifact(&bytes, &ctx).is_err());
}

/// Re-sealed mutations of source identity and of the result body: every digest
/// is valid again, and the replay still refuses.
#[test]
fn resealed_source_identity_and_result_mutations_fail_replay() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, false);
    let ctx = ExecutionContext::for_tests(4);
    let names = ["z1", "x", "z2", "y"].map(String::from);
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &names, &ctx).unwrap();
    let original = MzTransportArtifactWire::decode(&bytes).unwrap();
    let resealed = |edit: &dyn Fn(&mut MzTransportArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        wire.premises_digest = wire.expected_premises_digest().unwrap();
        wire.data_digest = wire.expected_data_digest().unwrap();
        wire.export().unwrap()
    };
    let source = refused(&resealed(&|w| w.query.sources[1].population = "c".into()));
    assert_ne!(source, MzTransportArtifactError::PremisesMismatch, "{source:?}");
    let point = refused(&resealed(&|w| w.results[0].probabilities[0] += 1e-9));
    assert_eq!(point, MzTransportArtifactError::PointMismatch);
}

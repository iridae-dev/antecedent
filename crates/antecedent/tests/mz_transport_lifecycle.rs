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
use antecedent_io::mz_transport_artifact::{
    MZ_INTERVAL_NOT_LICENSED, MZ_NOMINAL_INTERVAL, MZ_WITHHELD, MzTransportArtifactError,
    MzTransportArtifactInput, MzTransportArtifactWire, MzTransportConsumeLimits, MzUncertaintyWire,
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
    // Uncertainty bookkeeping cannot claim an interval exact laws do not have.
    let claimed = mutate(&|w| w.uncertainty.status = MZ_NOMINAL_INTERVAL.into());
    assert!(matches!(refused(&claimed), MzTransportArtifactError::UncertaintyMismatch(_)));
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
fn the_unmeasured_interval_route_refuses_with_cell_not_licensed() {
    let (catalog, exact) = evidence();
    let data = empirical(&exact, 40_000.0);
    let ctx = ExecutionContext::for_tests(5);
    let prepared = prepare(&with_studies(&catalog), data, true);
    let result = prepared.estimate(&ctx).unwrap();
    // The point is returned ...
    assert!((risk_of(result.distribution()) - target_scm().risk(&[(X, 1)], Y)).abs() < 1e-2);
    // ... and no interval: the route is closed until its coverage records exist.
    let uncertainty = result.uncertainty();
    assert_eq!(uncertainty.status, MZ_WITHHELD);
    assert_eq!(uncertainty.reason, MZ_INTERVAL_NOT_LICENSED);
    assert_eq!(uncertainty.reason, "cell_not_licensed");
    assert!(!uncertainty.available());
    assert!(uncertainty.mean_intervals.is_empty() && uncertainty.seed.is_none());
    assert_eq!(uncertainty.dependence_reason, None);
    // The artifact carries the same refusal and a consumer rechecks it.
    let bytes = result.export(&prepared, &ctx).unwrap();
    let consumed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(consumed.uncertainty(), uncertainty);
    let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
    wire.uncertainty.reason = "estimator_grid_not_measured".into();
    assert!(matches!(
        refused(&wire.export().unwrap()),
        MzTransportArtifactError::UncertaintyMismatch(_)
    ));
    // Undeclared dependence is reported beside the closed-route reason.
    let undeclared = prepare(&catalog, empirical(&exact, 40_000.0), true).estimate(&ctx).unwrap();
    assert_eq!(undeclared.uncertainty().reason, MZ_INTERVAL_NOT_LICENSED);
    assert_eq!(
        undeclared.uncertainty().dependence_reason.as_deref(),
        Some("sampling_dependence_unknown")
    );
}

#[test]
fn an_internal_interval_is_recomputed_bit_for_bit_and_refused_by_public_consumers() {
    let (catalog, exact) = evidence();
    let catalog = with_studies(&catalog);
    let data = empirical(&exact, 40_000.0);
    let prepared =
        prepare_requests(&catalog, data.clone(), true, vec![request(false), request(true)]);
    let ctx = ExecutionContext::for_tests(9);
    let result = prepared.estimate(&ctx).unwrap();
    // The internal estimator the calibration harness measures, under seed 9.
    let run = antecedent_estimate::mz_transport_bootstrap_interval(
        prepared.functional(),
        prepared.data(),
        prepared.requests(),
        ExactEvaluationLimits::default(),
        39,
        0.95,
        &ctx,
    )
    .unwrap();
    assert!(run.is_ok());
    let uncertainty = MzUncertaintyWire::from_bootstrap(&run, 39, 0.95, 9);
    let bytes = MzTransportArtifactWire::checked(
        MzTransportArtifactInput {
            graph: &graph(),
            functional: prepared.functional(),
            search: MZ_TRANSPORT_DEFAULT_LIMITS,
            data: &data,
            requests: prepared.requests(),
            limits: ExactEvaluationLimits::default(),
            variable_names: &[],
            results: result.distributions(),
            uncertainty,
        },
        &ctx,
    )
    .unwrap()
    .export()
    .unwrap();
    // The io consumer recomputes the stored interval from its seed ...
    let consumed =
        MzTransportArtifactWire::consume(&bytes, &ExecutionContext::for_tests(0)).unwrap();
    assert_eq!(consumed.wire.uncertainty.status, MZ_NOMINAL_INTERVAL);
    // ... and refuses moved bounds, a moved contrast bound, or another seed.
    let original = MzTransportArtifactWire::decode(&bytes).unwrap();
    let refuse = |edit: &dyn Fn(&mut MzUncertaintyWire)| {
        let mut wire = original.clone();
        edit(&mut wire.uncertainty);
        match MzTransportArtifactWire::consume(&wire.export().unwrap(), &ctx) {
            Err(IoError::MzTransport(MzTransportArtifactError::UncertaintyMismatch(_))) => {}
            other => panic!("expected an uncertainty mismatch, got {:?}", other.map(|_| ())),
        }
    };
    refuse(&|u| u.mean_intervals[1][0].1 -= 1e-12);
    refuse(&|u| u.mean_intervals[0][0].2 += 1e-12);
    refuse(&|u| u.contrast_intervals[0].2 -= 1e-12);
    refuse(&|u| u.seed = Some(10));
    refuse(&|u| u.replicates_requested = 41);
    // A public consumer refuses any carried interval while the route is closed.
    match consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx) {
        Err(IoError::Refused { code, message }) => {
            assert_eq!(code, "cell_not_licensed");
            assert!(message.starts_with("mz_transport.interval_withheld"));
        }
        other => panic!("expected cell_not_licensed, got {:?}", other.map(|_| ())),
    }
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
        assert_eq!(result.uncertainty().reason, MZ_INTERVAL_NOT_LICENSED);
        assert_eq!(result.uncertainty().dependence_reason.as_deref(), dependence);
        // The consumer re-derives the same dependence reason.
        let bytes = result.export(&prepared, &ctx).unwrap();
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
        let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
        wire.uncertainty.dependence_reason = if dependence.is_some() {
            None
        } else {
            Some("transport.unsupported_dependence".into())
        };
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
    assert!(exact_prepared.refresh(relabelled, &ctx).is_err());
    assert!(exact_prepared.refresh(foreign, &ctx).is_err());
    // An empirical preparation refuses tables without counts.
    assert!(prepared.refresh(exact_prepared.data().clone(), &ctx).is_err());
}

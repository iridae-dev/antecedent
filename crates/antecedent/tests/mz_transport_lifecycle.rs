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
    MZ_NOMINAL_INTERVAL, MZ_WITHHELD, MzTransportArtifactError, MzTransportArtifactWire,
    MzTransportConsumeLimits, MzUncertaintyWire,
};
use common::mz_fixture::{
    X, Y, empirical, evidence, graph, query, sources, target_scm, with_studies,
};
use common::z_scm::{risk_of, vid};

fn request(x: bool) -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(x))])
}

/// Decide and bind, then prepare; every builder input is moved into the
/// prepared state and dropped here.
fn prepare(
    catalog: &EvidenceCatalog,
    data: ExactTransportData,
    counted: bool,
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
        request(true),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap()
}

#[test]
fn estimate_runs_the_retained_plan_after_builder_disposal() {
    let prepared = {
        let (catalog, data) = evidence();
        prepare(&catalog, data, false)
    };
    let result = prepared.estimate(&ExecutionContext::for_tests(1)).unwrap();
    let truth = target_scm().risk(&[(X, 1)], Y);
    assert!((risk_of(result.distribution()) - truth).abs() < 1e-12);
    assert_eq!(result.uncertainty(), &MzUncertaintyWire::point_only());
    assert_eq!(
        prepared.theorem_scope().family,
        antecedent_core::TheoremScope::mz_transportability().family
    );
}

#[test]
fn exported_point_is_recomputed_by_an_independent_consumer() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, false);
    let ctx = ExecutionContext::for_tests(2);
    let result = prepared.estimate(&ctx).unwrap();
    let bytes = result.export(&prepared).unwrap();
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
    let bytes =
        prepared.estimate(&ExecutionContext::for_tests(4)).unwrap().export(&prepared).unwrap();
    let original = MzTransportArtifactWire::decode(&bytes).unwrap();
    let mutate = |edit: &dyn Fn(&mut MzTransportArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        wire.export().unwrap()
    };
    // Result body.
    let point = mutate(&|w| w.result.probabilities[0] += 1e-9);
    assert!(matches!(refused(&point), MzTransportArtifactError::PointMismatch));
    // Theorem premises and source identity are covered by the premises digest.
    let rules = mutate(&|w| w.proof.rules.pop().map(|_| ()).unwrap());
    assert!(matches!(refused(&rules), MzTransportArtifactError::PremisesMismatch));
    let source = mutate(&|w| w.query.sources[1].population = "c".into());
    assert!(matches!(refused(&source), MzTransportArtifactError::PremisesMismatch));
    // A consistent rewrite of the premises must still reproduce the decision.
    let forged = mutate(&|w| {
        w.proof.stages.clear();
        w.premises_digest = w.expected_premises_digest().unwrap();
    });
    assert!(matches!(refused(&forged), MzTransportArtifactError::ProofMismatch(_)));
    // Source-specific bindings.
    let bindings = mutate(&|w| w.bindings.reverse());
    assert!(matches!(refused(&bindings), MzTransportArtifactError::BindingMismatch));
    // Data identity: a law that no longer matches its catalog binding.
    let snapshot = mutate(&|w| w.laws[0].snapshot = "other".into());
    assert!(
        consume_mz_transport_artifact(
            &snapshot,
            MzTransportConsumeLimits::default(),
            &ExecutionContext::for_tests(3)
        )
        .is_err()
    );
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
fn empirical_bookkeeping_is_rechecked_against_declared_sampling() {
    let (catalog, exact) = evidence();
    let data = empirical(&exact, 40_000.0);
    let ctx = ExecutionContext::for_tests(5);
    // Declared independent studies publish a nominal interval.
    let studies = with_studies(&catalog);
    let prepared = prepare(&studies, data.clone(), true);
    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(result.uncertainty().status, MZ_NOMINAL_INTERVAL);
    assert_eq!(result.uncertainty().reason, antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED);
    let bytes = result.export(&prepared).unwrap();
    consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
    wire.uncertainty.replicates_ok += 1;
    assert!(matches!(
        refused(&wire.export().unwrap()),
        MzTransportArtifactError::UncertaintyMismatch(_)
    ));
    // Without study identities dependence is unknown: withheld, and a consumer
    // refuses bookkeeping that claims otherwise.
    let prepared = prepare(&catalog, data, true);
    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(result.uncertainty().status, MZ_WITHHELD);
    assert_eq!(result.uncertainty().reason, "sampling_dependence_unknown");
    let bytes = result.export(&prepared).unwrap();
    let mut wire = MzTransportArtifactWire::decode(&bytes).unwrap();
    wire.uncertainty.reason = "transport.unsupported_dependence".into();
    assert!(matches!(
        refused(&wire.export().unwrap()),
        MzTransportArtifactError::UncertaintyMismatch(_)
    ));
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

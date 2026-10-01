//! Prepared lifecycle, refresh and independent artifact consumption of a
//! mixed-source proof-search formula.
//!
//! The fixture is the confounded front-door graph `X -> Z -> Y`, `X <-> Y`: an
//! observational study measures `{X, Z}` and a trial of `do(Z)` measures
//! `{X, Y}`. Every law is enumerated from one structural model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;
#[path = "../../antecedent-estimate/tests/common/mixed_fixture.rs"]
mod mixed_fixture;

use antecedent::{PreparedMixedSource, StudyBuilder, consume_mixed_source_artifact};
use antecedent_core::{EvidenceCatalog, ExecutionContext};
use antecedent_expr::{ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    MIXED_SOURCE_DEFAULT_LIMITS, MixedSourceDecision, bind_mixed_source_catalog,
    decide_mixed_source,
};
use antecedent_io::IoError;
use antecedent_io::mixed_source_artifact::{
    MixedSourceArtifactError, MixedSourceArtifactWire, MixedSourceConsumeLimits,
};
use common::z_scm::risk_of;
use mixed_fixture::{X, Y, Z, build, frontdoor_scm, graph, query, request, study};

fn graph_x() -> antecedent_graph::Admg {
    graph(3, &[(0, 1), (1, 2)], &[(0, 2)])
}

fn evidence() -> (EvidenceCatalog, ExactTransportData) {
    build(&frontdoor_scm(), &[study("observational", &[], &[X, Z]), study("trial", &[Z], &[X, Y])])
}

fn prepare(
    catalog: &EvidenceCatalog,
    data: ExactTransportData,
    requests: Vec<antecedent_expr::Assignment>,
) -> PreparedMixedSource {
    let ctx = ExecutionContext::for_tests(7);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(&graph_x(), &query(Y, X), catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
            .unwrap()
    else {
        panic!("the complementary catalog identifies");
    };
    let functional = bind_mixed_source_catalog(&graph_x(), &derivation, catalog).unwrap();
    StudyBuilder::mixed_source(
        graph_x(),
        functional,
        MIXED_SOURCE_DEFAULT_LIMITS,
        data,
        requests,
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap()
}

fn truth(x: u8) -> f64 {
    frontdoor_scm().risk(&[(X, x)], Y)
}

#[test]
fn estimate_runs_the_retained_plan_after_builder_disposal() {
    let prepared = {
        let (catalog, data) = evidence();
        prepare(&catalog, data, vec![request(X, false), request(X, true)])
    };
    assert_eq!(prepared.plans().len(), 2);
    let result = prepared.estimate(&ExecutionContext::for_tests(1)).unwrap();
    for x in [0u8, 1] {
        let point = risk_of(&result.distributions()[usize::from(x)]);
        assert!((point - truth(x)).abs() < 1e-12, "do(X={x}): {point} vs {}", truth(x));
    }
}

fn refused(bytes: &[u8]) -> MixedSourceArtifactError {
    match consume_mixed_source_artifact(
        bytes,
        MixedSourceConsumeLimits::default(),
        &ExecutionContext::for_tests(3),
    ) {
        Err(IoError::MixedSource(error)) => error,
        other => panic!(
            "expected a typed mixed-source refusal, got {:?}",
            other.map(|r| r.distribution().clone())
        ),
    }
}

#[test]
fn exported_point_is_recomputed_by_an_independent_consumer() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, vec![request(X, true)]);
    let ctx = ExecutionContext::for_tests(2);
    let result = prepared.estimate(&ctx).unwrap();
    let bytes = result.export(&prepared).unwrap();
    drop(prepared);
    let consumed =
        consume_mixed_source_artifact(&bytes, MixedSourceConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(consumed.distribution().probabilities, result.distribution().probabilities);
    // The artifact names the study and regime of every cited leaf, and the proof carries
    // the frozen rule-set version and each step's source distribution.
    let wire = MixedSourceArtifactWire::decode(&bytes).unwrap();
    assert_eq!(
        wire.bindings,
        [
            (1, "observational".to_owned(), "target".to_owned()),
            (2, "trial".to_owned(), "target".to_owned()),
        ]
    );
    assert_eq!(wire.proof.rule_set, "x9.rules.v1");
    let sourced = wire.proof.steps.iter().filter(|s| s.regime.is_some()).collect::<Vec<_>>();
    assert_eq!(sourced.len(), 2);
    assert!(sourced.iter().all(|s| s.source.is_some() && s.snapshot.is_some()));
    assert_eq!(wire.uncertainty, "point_only");
    // Every proof step of the stored derivation is recomputable: rechecking the record
    // under a larger-limit consumer accepts it, a smaller one refuses.
    assert!(wire.proof.steps.iter().any(|s| s.rule == "rule3_insert"));
}

#[test]
fn a_mutated_artifact_fails_independent_consumption() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, vec![request(X, true)]);
    let ctx = ExecutionContext::for_tests(4);
    let names = ["x", "z", "y"].map(String::from);
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &names).unwrap();
    let original = MixedSourceArtifactWire::decode(&bytes).unwrap();
    let mutate = |edit: &dyn Fn(&mut MixedSourceArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        wire.export().unwrap()
    };
    let refuse = |bytes: Vec<u8>, expected: &dyn Fn(&MixedSourceArtifactError) -> bool| {
        let error = refused(&bytes);
        assert!(expected(&error), "unexpected refusal {error:?}");
    };
    // Result body.
    refuse(mutate(&|w| w.results[0].probabilities[0] += 1e-9), &|e| {
        matches!(e, MixedSourceArtifactError::PointMismatch)
    });
    // Theorem premises, source identity, limits and names are covered by the premises digest.
    for edit in [
        (&|w: &mut MixedSourceArtifactWire| w.proof.steps.last_mut().unwrap().params.push(9))
            as &dyn Fn(&mut MixedSourceArtifactWire),
        &|w| w.proof.steps[0].study = Some("someone-else".into()),
        &|w| w.proof.rule_set = "x9.rules.v0".into(),
        &|w| w.query.target = "elsewhere".into(),
        &|w| w.search_operations -= 1,
        &|w| w.search_depth -= 1,
        &|w| w.operation_limit -= 1,
        &|w| w.variable_names.swap(0, 2),
    ] {
        refuse(mutate(edit), &|e| matches!(e, MixedSourceArtifactError::PremisesMismatch));
    }
    assert_eq!(original.check_variable_names(&names), Ok(()));
    let mut relabelled = names.clone();
    relabelled.swap(0, 2);
    assert_eq!(
        original.check_variable_names(&relabelled),
        Err(MixedSourceArtifactError::NamesMismatch)
    );
    // A consistent rewrite of the premises must still reproduce the decision.
    for edit in [
        (&|w: &mut MixedSourceArtifactWire| w.proof.stages.push(("rule_search".into(), "x".into())))
            as &dyn Fn(&mut MixedSourceArtifactWire),
        &|w| w.proof.steps[0].source = Some("catalog_distribution.v1|forged".into()),
        &|w| w.proof.operations_to_proof += 1,
        &|w| w.proof.steps.swap(0, 1),
    ] {
        refuse(
            mutate(&|w| {
                edit(w);
                w.premises_digest = w.expected_premises_digest().unwrap();
            }),
            &|e| matches!(e, MixedSourceArtifactError::ProofMismatch(_)),
        );
    }
    // Source-named bindings.
    refuse(mutate(&|w| w.bindings.reverse()), &|e| {
        matches!(e, MixedSourceArtifactError::BindingMismatch)
    });
    // Data identity: snapshot ids are bound by the data-identity digest ...
    refuse(mutate(&|w| w.laws[0].snapshot = "other".into()), &|e| {
        matches!(e, MixedSourceArtifactError::DataIdentityMismatch)
    });
    // ... and a consistent rewrite of the digest still has to bind: the law no longer
    // matches its catalog binding.
    let rebound = mutate(&|w| {
        w.laws[0].snapshot = "other".into();
        w.data_digest = w.expected_data_digest().unwrap();
    });
    assert!(matches!(
        consume_mixed_source_artifact(&rebound, MixedSourceConsumeLimits::default(), &ctx),
        Err(IoError::Refused { .. })
    ));
    // A route that publishes points only accepts no other claim.
    refuse(mutate(&|w| w.uncertainty = "nominal_interval".into()), &|e| {
        matches!(e, MixedSourceArtifactError::UnsupportedSemantics(_))
    });
    // A consumer with smaller limits than the producer refuses.
    let small = MixedSourceConsumeLimits {
        search: antecedent_core::SearchLimits { operations: 10, depth: 16 },
        ..MixedSourceConsumeLimits::default()
    };
    assert!(matches!(
        consume_mixed_source_artifact(&bytes, small, &ExecutionContext::for_tests(3)),
        Err(IoError::MixedSource(MixedSourceArtifactError::LimitsExceeded(_)))
    ));
    // Another version is refused before the payload is read.
    let mut versioned = original.clone();
    versioned.version = 2;
    assert!(matches!(
        consume_mixed_source_artifact(
            &versioned.export().unwrap(),
            MixedSourceConsumeLimits::default(),
            &ctx
        ),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
}

#[test]
fn refresh_keeps_the_proof_and_refuses_changed_evidence_identity() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data.clone(), vec![request(X, true)]);
    let ctx = ExecutionContext::for_tests(6);
    let before = prepared.estimate(&ctx).unwrap();
    // Same snapshots, new numbers: the proof is untouched and the point moves.
    let mut shifted = frontdoor_scm();
    shifted.exo_p[1] = 0.6;
    let (_, moved) =
        build(&shifted, &[study("observational", &[], &[X, Z]), study("trial", &[Z], &[X, Y])]);
    let refreshed = prepared.refresh(moved, &ctx).unwrap();
    assert_eq!(
        refreshed.functional().derivation().to_record(),
        prepared.functional().derivation().to_record()
    );
    let after = refreshed.estimate(&ctx).unwrap();
    assert_ne!(after.distribution().probabilities, before.distribution().probabilities);
    // A law under another snapshot is evidence the frozen catalog never bound.
    let relabelled = data
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
    let relabelled = ExactTransportData::try_new(relabelled, data.max_support_rows()).unwrap();
    assert!(prepared.refresh(relabelled, &ctx).is_err());
    // A law for another regime is not evidence the frozen catalog bound either.
    let mut foreign = data.laws().to_vec();
    let law = &foreign[0];
    foreign[0] = ExactDiscreteLaw::try_new(
        "target",
        antecedent_core::RegimeId::from_raw(99),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap();
    let foreign = ExactTransportData::try_new(foreign, data.max_support_rows()).unwrap();
    assert!(prepared.refresh(foreign, &ctx).is_err());
}

#[test]
fn a_functional_decided_on_another_graph_is_refused() {
    let (catalog, data) = evidence();
    let ctx = ExecutionContext::for_tests(7);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(&graph_x(), &query(Y, X), &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
            .unwrap()
    else {
        panic!("identified");
    };
    let functional = bind_mixed_source_catalog(&graph_x(), &derivation, &catalog).unwrap();
    let other = graph(3, &[(0, 1), (1, 2)], &[]);
    let error = StudyBuilder::mixed_source(
        other,
        functional,
        MIXED_SOURCE_DEFAULT_LIMITS,
        data,
        vec![request(X, true)],
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    match error {
        IoError::Refused { code, message } => {
            assert_eq!(code, "transport_not_certified");
            assert!(message.starts_with("mixed_search.invalid_derivation"), "{message}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

#[test]
fn prepare_requires_a_request() {
    let (catalog, data) = evidence();
    let ctx = ExecutionContext::for_tests(7);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(&graph_x(), &query(Y, X), &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
            .unwrap()
    else {
        panic!("identified");
    };
    let functional = bind_mixed_source_catalog(&graph_x(), &derivation, &catalog).unwrap();
    assert!(
        StudyBuilder::mixed_source(
            graph_x(),
            functional,
            MIXED_SOURCE_DEFAULT_LIMITS,
            data,
            Vec::new(),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .is_err()
    );
}

/// Compatibility: an unknown required feature (or none) is refused, as is any
/// other format version, so an older reader never accepts a newer artifact.
#[test]
fn an_unknown_required_feature_or_another_version_is_refused() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, vec![request(X, true)]);
    let ctx = ExecutionContext::for_tests(4);
    let names = ["x", "z", "y"].map(String::from);
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &names).unwrap();
    let original = MixedSourceArtifactWire::decode(&bytes).unwrap();
    let mut wire = original.clone();
    wire.required_features.push("future_semantics_v9".into());
    assert_eq!(
        refused(&wire.export().unwrap()),
        MixedSourceArtifactError::UnsupportedSemantics("required features")
    );
    let mut wire = original.clone();
    wire.required_features.clear();
    assert_eq!(
        refused(&wire.export().unwrap()),
        MixedSourceArtifactError::UnsupportedSemantics("required features")
    );
    let mut wire = original;
    wire.version = antecedent_io::mixed_source_artifact::MIXED_SOURCE_ARTIFACT_VERSION + 1;
    assert!(matches!(
        consume_mixed_source_artifact(
            &wire.export().unwrap(),
            MixedSourceConsumeLimits::default(),
            &ctx
        ),
        Err(IoError::UnsupportedVersion { .. })
    ));
}

/// A re-sealed edit of the result body: both digests are valid again and the
/// replayed point refuses.
#[test]
fn a_resealed_result_mutation_fails_replay() {
    let (catalog, data) = evidence();
    let prepared = prepare(&catalog, data, vec![request(X, true)]);
    let ctx = ExecutionContext::for_tests(4);
    let names = ["x", "z", "y"].map(String::from);
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &names).unwrap();
    let mut wire = MixedSourceArtifactWire::decode(&bytes).unwrap();
    wire.results[0].probabilities[0] += 1e-9;
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    assert_eq!(refused(&wire.export().unwrap()), MixedSourceArtifactError::PointMismatch);
}

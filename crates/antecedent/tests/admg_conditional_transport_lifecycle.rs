//! Prepared lifecycle, refresh and artifact consumption of an ADMG conditional
//! transport formula (2.2B B1, X2). The consumer is independent of the artifact
//! (it re-derives everything), not of the implementation (it re-runs the same
//! decision and evaluator); only its proof check is distinct code.
//!
//! The fixture is `X(0) -> Y(1) -> W(2)`, `X <-> Y`, with the target's `W`
//! mechanism selected: `W` cannot move, so the query `P*(y | do(x), w)` is the
//! reduced joint `P*(y, w | do(x))` normalized at `w`. Every law and the truth
//! are enumerated from one latent SCM.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/admg_conditional_scm.rs"]
mod admg_conditional_scm;

use admg_conditional_scm::{Scm, query, request};
use antecedent::{
    PreparedAdmgConditionalTransport, StudyBuilder, consume_admg_conditional_transport_artifact,
};
use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, BoundConditionalTransportFunctional,
    ConditionalTransportDecision, decide_admg_conditional_transport,
};
use antecedent_io::IoError;
use antecedent_io::admg_conditional_transport_artifact::{
    AdmgConditionalArtifactError, AdmgConditionalArtifactWire, AdmgConditionalConsumeLimits,
};

fn scm(params: usize) -> Scm {
    Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![(0, 1)],
        selected: vec![2],
        params,
        target_zero: Vec::new(),
    }
}

/// The four requests: every level of (X, W).
const LEVELS: [usize; 4] = [0b000, 0b001, 0b100, 0b101];

fn decided(scm: &Scm) -> BoundConditionalTransportFunctional {
    let (catalog, _) = scm.catalog_and_laws();
    let ctx = ExecutionContext::for_tests(5);
    match decide_admg_conditional_transport(
        &scm.diagram(),
        &query(&[1], &[0], &[2]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap()
    {
        ConditionalTransportDecision::Identified(bound) => *bound,
        other => panic!("identified: {other:?}"),
    }
}

fn prepare(scm: &Scm) -> PreparedAdmgConditionalTransport {
    let (_, data) = scm.catalog_and_laws();
    StudyBuilder::admg_conditional_transport(
        scm.diagram(),
        decided(scm),
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        data,
        LEVELS.iter().map(|level| request(&[0], &[2], *level)).collect(),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(5),
    )
    .unwrap()
}

fn assert_truth(scm: &Scm, distributions: &[antecedent_expr::ExactDistribution]) {
    assert_eq!(distributions.len(), LEVELS.len());
    for (distribution, level) in distributions.iter().zip(LEVELS) {
        let truth = scm.truth(&[1], &[0], &[2], level).unwrap();
        for (p, t) in distribution.probabilities.iter().zip(&truth) {
            assert!((p - t).abs() < 1e-12, "level {level}: {p} vs {t}");
        }
    }
}

#[test]
fn the_prepared_estimate_matches_the_enumerated_truth() {
    let prepared = prepare(&scm(0));
    assert_eq!(prepared.plans().len(), 4);
    let result = prepared.estimate(&ExecutionContext::for_tests(1)).unwrap();
    assert_truth(&scm(0), result.distributions());
    // The conditional differs across W levels: conditioning is not a no-op.
    let at = |i: usize| result.distributions()[i].probabilities[1];
    assert!((at(0) - at(2)).abs() > 1e-3, "{} vs {}", at(0), at(2));
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let error = prepared.estimate(&cancelled).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn preparation_rechecks_the_derivation_on_the_diagram() {
    let model = scm(0);
    let (_, data) = model.catalog_and_laws();
    let ctx = ExecutionContext::for_tests(5);
    let requests = vec![request(&[0], &[2], 0)];
    // A functional decided on this diagram prepares.
    StudyBuilder::admg_conditional_transport(
        model.diagram(),
        decided(&model),
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        data.clone(),
        requests.clone(),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    // The same functional on a diagram with another selection refuses.
    let other = SelectionDiagram::try_new(
        model.diagram().causal_graph().clone(),
        [admg_conditional_scm::v(1)],
    )
    .unwrap();
    let error = StudyBuilder::admg_conditional_transport(
        other,
        decided(&model),
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        data.clone(),
        requests.clone(),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    let IoError::Refused { code, message } = error else { panic!("coded refusal") };
    assert_eq!(code, "transport_not_certified");
    assert!(message.starts_with("admg_transport.invalid_derivation"), "{message}");
    // No request, and a request that does not bind W, refuse by name.
    for bad in [Vec::new(), vec![request(&[0], &[], 0)]] {
        let error = StudyBuilder::admg_conditional_transport(
            model.diagram(),
            decided(&model),
            ADMG_CONDITIONAL_DEFAULT_LIMITS,
            data.clone(),
            bad,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap_err();
        let IoError::Refused { code, message } = error else { panic!("coded refusal") };
        assert_eq!(code, "invalid_argument");
        assert!(message.starts_with("admg_transport.invalid_request"), "{message}");
    }
}

#[test]
fn refresh_keeps_the_proof_and_moves_the_point() {
    let prepared = prepare(&scm(0));
    let ctx = ExecutionContext::for_tests(2);
    let before = prepared.estimate(&ctx).unwrap();
    // Laws of another parameterization under the same regimes and snapshots.
    let (_, data) = scm(2).catalog_and_laws();
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(
        refreshed.functional().derivation().to_record().joint.steps.len(),
        prepared.functional().derivation().to_record().joint.steps.len()
    );
    let after = refreshed.estimate(&ctx).unwrap();
    assert_truth(&scm(2), after.distributions());
    assert_ne!(
        before.distributions()[0].probabilities[1].to_bits(),
        after.distributions()[0].probabilities[1].to_bits()
    );
    // Counted laws are refused: the route publishes exact points only.
    let (_, data) = scm(0).catalog_and_laws();
    let mut laws = data.laws().to_vec();
    let first = &laws[0];
    let cells = first.probabilities().len();
    laws[0] = antecedent_expr::ExactDiscreteLaw::try_new(
        first.population(),
        first.regime(),
        first.interventions().to_vec(),
        first.axes().to_vec(),
        vec![1.0 / cells as f64; cells],
        first.snapshot_identity(),
        antecedent_expr::LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(vec![1; cells])
    .unwrap();
    let counted = ExactTransportData::try_new(laws, 1_000_000).unwrap();
    let error = prepared.refresh(counted, &ctx).unwrap_err();
    let IoError::Refused { code, message } = error else { panic!("coded refusal") };
    assert_eq!(code, "cell_not_licensed");
    assert!(message.starts_with("admg_transport.interval_withheld"), "{message}");
}

fn export(prepared: &PreparedAdmgConditionalTransport) -> Vec<u8> {
    let result = prepared.estimate(&ExecutionContext::for_tests(3)).unwrap();
    result.export_named(prepared, &["X".into(), "Y".into(), "W".into()]).unwrap()
}

#[test]
fn exported_point_is_recomputed_by_the_artifact_consumer() {
    let bytes = {
        let prepared = prepare(&scm(1));
        export(&prepared)
    };
    let ctx = ExecutionContext::for_tests(9);
    let consumed = consume_admg_conditional_transport_artifact(
        &bytes,
        AdmgConditionalConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_truth(&scm(1), consumed.distributions());
    // Bit-for-bit: the stored points are the recomputed ones.
    let wire = AdmgConditionalArtifactWire::decode(&bytes).unwrap();
    for (stored, fresh) in wire.results.iter().zip(consumed.distributions()) {
        let fresh: Vec<u64> = fresh.probabilities.iter().map(|p| p.to_bits()).collect();
        let stored: Vec<u64> = stored.probabilities.iter().map(|p| p.to_bits()).collect();
        assert_eq!(stored, fresh);
    }
    wire.check_variable_names(&["X".into(), "Y".into(), "W".into()]).unwrap();
    assert_eq!(
        wire.check_variable_names(&["Y".into(), "X".into(), "W".into()]).unwrap_err().refusal(),
        ("invalid_argument", "admg_transport.invalid_artifact")
    );
}

fn refusal(bytes: &[u8]) -> AdmgConditionalArtifactError {
    match AdmgConditionalArtifactWire::consume_with_limits(
        bytes,
        AdmgConditionalConsumeLimits::default(),
        &ExecutionContext::for_tests(4),
    ) {
        Err(IoError::AdmgConditional(error)) => error,
        Err(other) => panic!("typed artifact refusal expected: {other:?}"),
        Ok(_) => panic!("mutated artifact must not be consumed"),
    }
}

fn reseal(wire: &mut AdmgConditionalArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

#[test]
#[allow(clippy::too_many_lines)] // One mutation per consumer check, each with its reason.
fn a_mutated_artifact_fails_consumption() {
    let bytes = export(&prepare(&scm(1)));
    let original = AdmgConditionalArtifactWire::decode(&bytes).unwrap();
    let pair = |error: AdmgConditionalArtifactError| error.refusal();

    // Unsealed edits: the digests catch premises, the replay catches points.
    let mut wire = original.clone();
    wire.query.conditioned_on = vec![0];
    assert_eq!(refusal(&wire.export().unwrap()), AdmgConditionalArtifactError::PremisesMismatch);
    let mut wire = original.clone();
    wire.laws[0].snapshot = "renamed".into();
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::DataIdentityMismatch
    );
    let mut wire = original.clone();
    let mut catalog = wire.catalog.to_catalog().unwrap();
    catalog =
        antecedent_core::EvidenceCatalog::try_new([], catalog.regimes[1..].to_vec(), [], None)
            .unwrap();
    wire.catalog =
        antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&catalog);
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::DataIdentityMismatch,
        "the whole catalog is bound by the data digest"
    );
    let mut wire = original.clone();
    wire.results[0].probabilities[0] += 1e-9;
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::ReplayMismatch("a point")
    );

    // Re-sealed semantic mutations: every digest is valid, the replay refuses
    // for the right reason.
    // (a) The query conditions on another variable: the proof no longer checks.
    let mut wire = original.clone();
    wire.query.conditioned_on = vec![0];
    wire.query.treatments = vec![2];
    let error = refusal(&reseal(&mut wire));
    assert!(matches!(error, AdmgConditionalArtifactError::ProofMismatch(_)), "{error:?}");
    assert_eq!(pair(error), ("transport_not_certified", "admg_transport.invalid_derivation"));
    // (b) The non-movable W claimed moved: its rule-2 premise fails.
    let mut wire = original.clone();
    wire.proof.moves = vec![2];
    wire.proof.remaining.clear();
    let error = refusal(&reseal(&mut wire));
    assert_eq!(pair(error), ("transport_not_certified", "admg_transport.invalid_derivation"));
    // (c) A law's cells permuted under the same snapshot (the data digest names
    // snapshots, so it is unchanged): the recomputed point differs.
    let mut wire = original.clone();
    let target = wire.laws.iter().position(|law| law.population == "target").unwrap();
    wire.laws[target].probabilities.reverse();
    assert_eq!(
        refusal(&reseal(&mut wire)),
        AdmgConditionalArtifactError::ReplayMismatch("a point")
    );
    // (d) The catalog loses the source experiments on X: the re-decision is no
    // longer the stored identified one.
    let mut wire = original.clone();
    let mut catalog = wire.catalog.to_catalog().unwrap();
    let kept: Vec<_> = catalog
        .regimes
        .iter()
        .filter(|r| {
            !(r.population.as_ref() == "source"
                && r.interventions.contains(&admg_conditional_scm::v(0)))
        })
        .cloned()
        .collect();
    catalog = antecedent_core::EvidenceCatalog::try_new([], kept, [], None).unwrap();
    wire.catalog =
        antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&catalog);
    let error = refusal(&reseal(&mut wire));
    assert_eq!(error, AdmgConditionalArtifactError::ReplayMismatch("the decision"));
    assert_eq!(pair(error), ("transport_not_certified", "admg_transport.replay_mismatch"));
    // (e) Leaf bindings relabelled.
    let mut wire = original.clone();
    wire.bindings[0].0 = "elsewhere".into();
    assert_eq!(
        refusal(&reseal(&mut wire)),
        AdmgConditionalArtifactError::ReplayMismatch("the leaf bindings")
    );
    // (f) Stored search limits forged below what the proof needs: the checker
    // replays under them, and a proof longer than the stored limit does not
    // verify, so the forged limit cannot certify anything.
    let mut wire = original.clone();
    wire.search_operations = 1;
    let error = refusal(&reseal(&mut wire));
    assert!(matches!(error, AdmgConditionalArtifactError::ProofMismatch(_)), "{error:?}");
    assert_eq!(pair(error), ("transport_not_certified", "admg_transport.invalid_derivation"));
    // (g) Stored limits above the consumer's maxima refuse before any work.
    let mut wire = original.clone();
    wire.search_operations = ADMG_CONDITIONAL_DEFAULT_LIMITS.operations + 1;
    let error = refusal(&reseal(&mut wire));
    assert_eq!(pair(error), ("route_not_supported", "admg_transport.consumer_limits"));
    let small = AdmgConditionalConsumeLimits {
        search: SearchLimits { operations: 10, depth: 24 },
        ..AdmgConditionalConsumeLimits::default()
    };
    let Err(IoError::AdmgConditional(error)) = AdmgConditionalArtifactWire::consume_with_limits(
        &bytes,
        small,
        &ExecutionContext::for_tests(1),
    ) else {
        panic!("a consumer with smaller limits refuses");
    };
    assert_eq!(error, AdmgConditionalArtifactError::LimitsExceeded("search operation limit"));
    // (h) An interval claim or counted laws: the route is point-only.
    let mut wire = original.clone();
    wire.uncertainty = "percentile_bootstrap".into();
    assert_eq!(refusal(&reseal(&mut wire)), AdmgConditionalArtifactError::IntervalWithheld);
    let mut wire = original.clone();
    wire.laws[0].empirical_counts = Some(vec![1; wire.laws[0].probabilities.len()]);
    let error = refusal(&reseal(&mut wire));
    assert_eq!(pair(error), ("cell_not_licensed", "admg_transport.interval_withheld"));
    // (i) An unknown field is refused by the wire (deny_unknown_fields).
    let mut value: serde_json::Value = antecedent_io::from_cbor(&bytes).unwrap();
    value["unexpected"] = serde_json::json!(1);
    let tampered = antecedent_io::to_cbor(&value).unwrap();
    assert!(AdmgConditionalArtifactWire::decode(&tampered).is_err());
    let mut value: serde_json::Value = antecedent_io::from_cbor(&bytes).unwrap();
    value["proof"]["extra"] = serde_json::json!(true);
    assert!(AdmgConditionalArtifactWire::decode(&antecedent_io::to_cbor(&value).unwrap()).is_err());
    // (j) Another version is refused before the payload is interpreted.
    let mut wire = original.clone();
    wire.version = 2;
    assert!(matches!(
        AdmgConditionalArtifactWire::decode(&wire.export().unwrap()),
        Err(IoError::AdmgConditional(AdmgConditionalArtifactError::UnsupportedVersion { .. }))
    ));
    // The unmutated artifact still replays; every refusal carries its code.
    assert!(
        AdmgConditionalArtifactWire::consume_with_limits(
            &bytes,
            AdmgConditionalConsumeLimits::default(),
            &ExecutionContext::for_tests(4)
        )
        .is_ok()
    );
    assert_eq!(
        IoError::AdmgConditional(AdmgConditionalArtifactError::PremisesMismatch).reason_code(),
        Some("transport_not_certified")
    );
}

/// An oversized graph or query is refused as a consumer limit before either
/// digest is hashed: an unsealed oversized artifact (whose premises digest is
/// also wrong) reports the size, never the digest.
#[test]
fn an_oversized_graph_or_query_refuses_before_any_digest() {
    let bytes = export(&prepare(&scm(0)));
    let original = AdmgConditionalArtifactWire::decode(&bytes).unwrap();
    let limits = ("route_not_supported", "admg_transport.consumer_limits");
    // Seven nodes (names dropped so the shape check passes).
    let mut wire = original.clone();
    wire.graph.node_count = 7;
    wire.variable_names.clear();
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::LimitsExceeded("graph size")
    );
    let error = refusal(&reseal(&mut wire));
    assert_eq!(error, AdmgConditionalArtifactError::LimitsExceeded("graph size"));
    assert_eq!(error.refusal(), limits);
    // An edge list longer than the node count allows.
    let mut wire = original.clone();
    wire.graph.directed = vec![(0, 1); 7];
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::LimitsExceeded("graph size")
    );
    // More outcomes than nodes, four treatments, four conditioned coordinates,
    // and four conditioned coordinates in the proof.
    let mut wire = original.clone();
    wire.query.outcomes = vec![1, 1, 1, 1];
    for result in &mut wire.results {
        result.outcomes.clone_from(&wire.query.outcomes);
    }
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::LimitsExceeded("query size")
    );
    let mut wire = original.clone();
    wire.query.conditioned_on = vec![2, 2, 2, 2];
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::LimitsExceeded("query size")
    );
    let mut wire = original.clone();
    wire.query.treatments = vec![0, 0, 0, 0];
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::LimitsExceeded("query size")
    );
    let mut wire = original.clone();
    wire.proof.remaining = vec![2, 2, 2, 2];
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::LimitsExceeded("query size")
    );
    // At the bounds the original still replays.
    assert!(
        AdmgConditionalArtifactWire::consume_with_limits(
            &bytes,
            AdmgConditionalConsumeLimits::default(),
            &ExecutionContext::for_tests(4)
        )
        .is_ok()
    );
}

#[test]
fn the_io_consumer_replays_under_the_stored_limits() {
    let prepared = prepare(&scm(0));
    let bytes = export(&prepared);
    let ctx = ExecutionContext::for_tests(6);
    let consumed = AdmgConditionalArtifactWire::consume_with_limits(
        &bytes,
        AdmgConditionalConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(consumed.wire.search_operations, ADMG_CONDITIONAL_DEFAULT_LIMITS.operations);
    assert_eq!(consumed.requests.len(), 4);
    assert_truth(&scm(0), &consumed.distributions);
    assert_eq!(consumed.functional.derivation().remaining(), [admg_conditional_scm::v(2)]);
    // A consumer whose evaluation limit is below the stored one refuses first.
    let tight = AdmgConditionalConsumeLimits {
        evaluation: ExactEvaluationLimits { operations: 10, depth: 256 },
        ..AdmgConditionalConsumeLimits::default()
    };
    let Err(IoError::AdmgConditional(error)) =
        AdmgConditionalArtifactWire::consume_with_limits(&bytes, tight, &ctx)
    else {
        panic!("limits refusal");
    };
    assert_eq!(error, AdmgConditionalArtifactError::LimitsExceeded("operation limit"));
    // Cancellation during replay is a budget refusal.
    let cancelled = ExecutionContext::for_tests(6);
    cancelled.cancellation.cancel();
    let error = AdmgConditionalArtifactWire::consume_with_limits(
        &bytes,
        AdmgConditionalConsumeLimits::default(),
        &cancelled,
    )
    .err()
    .expect("cancelled replay refuses");
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

/// `X(0) -> Y(1)`, `W1(2) -> Y`, `W3(3) -> Y`: both conditioned parents move, in
/// either order the checker accepts. The replayed decision fixes the order, so a
/// re-sealed artifact whose valid proof moves them in the other order is not the
/// producer's decision.
#[test]
fn a_reordered_but_valid_proof_is_not_the_replayed_decision() {
    let model = Scm {
        n: 4,
        directed: vec![(0, 1), (2, 1), (3, 1)],
        bidirected: vec![(0, 1)],
        selected: vec![2],
        params: 0,
        target_zero: Vec::new(),
    };
    let (catalog, data) = model.catalog_and_laws();
    let ctx = ExecutionContext::for_tests(8);
    let ConditionalTransportDecision::Identified(bound) = decide_admg_conditional_transport(
        &model.diagram(),
        &query(&[1], &[0], &[2, 3]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("identified");
    };
    assert_eq!(bound.derivation().moves().len(), 2);
    let prepared = StudyBuilder::admg_conditional_transport(
        model.diagram(),
        *bound,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        data,
        vec![request(&[0], &[2, 3], 0b1101)],
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    let bytes = prepared.estimate(&ctx).unwrap().export_named(&prepared, &[]).unwrap();
    let mut wire = AdmgConditionalArtifactWire::decode(&bytes).unwrap();
    wire.proof.moves.reverse();
    // The reversed order is itself a valid proof: the independent checker accepts it.
    let reordered = antecedent_identify::ConditionalTransportDerivation::from_record_checked(
        wire.proof.clone(),
        antecedent_io::expr_wire::expr_arena_from_wire(&wire.expression).unwrap(),
        &model.diagram(),
        &query(&[1], &[0], &[2, 3]),
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    );
    assert!(reordered.is_ok(), "{reordered:?}");
    assert_eq!(
        refusal(&reseal(&mut wire)),
        AdmgConditionalArtifactError::ReplayMismatch("the derivation")
    );
}

/// Compatibility: an unknown required feature (or none) is refused, and so is
/// any other format version, before the payload is interpreted.
#[test]
fn an_unknown_required_feature_or_another_version_is_refused() {
    let bytes = export(&prepare(&scm(0)));
    let original = AdmgConditionalArtifactWire::decode(&bytes).unwrap();
    let mut wire = original.clone();
    wire.required_features.push("future_semantics_v9".into());
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::UnsupportedSemantics("required features")
    );
    let mut wire = original.clone();
    wire.required_features.clear();
    assert_eq!(
        refusal(&wire.export().unwrap()),
        AdmgConditionalArtifactError::UnsupportedSemantics("required features")
    );
    for version in [
        0,
        antecedent_io::admg_conditional_transport_artifact::ADMG_CONDITIONAL_ARTIFACT_VERSION + 1,
    ] {
        let mut wire = original.clone();
        wire.version = version;
        assert!(matches!(
            AdmgConditionalArtifactWire::decode(&wire.export().unwrap()),
            Err(IoError::AdmgConditional(AdmgConditionalArtifactError::UnsupportedVersion { .. }))
        ));
    }
}

/// A proven obstruction (selection on the confounded outcome `Y`) exports with
/// its two-model witness and is re-verified by the consumer; an unsealed edit
/// is a premises mismatch, a re-sealed witness mutation fails the exact
/// verifier, and an oversized graph refuses before any digest.
#[test]
fn a_proven_obstruction_is_reverified_by_the_artifact_consumer() {
    use antecedent::{
        consume_admg_conditional_obstruction_artifact, export_admg_conditional_obstruction,
    };
    use antecedent_io::admg_conditional_transport_artifact::AdmgConditionalObstructionWire;
    let blocked = Scm { selected: vec![1], ..scm(0) };
    let (catalog, _) = blocked.catalog_and_laws();
    let ctx = ExecutionContext::for_tests(5);
    let diagram = blocked.diagram();
    let q = query(&[1], &[0], &[2]);
    let ConditionalTransportDecision::ProvenNonTransportable(proof) =
        decide_admg_conditional_transport(
            &diagram,
            &q,
            &catalog,
            ADMG_CONDITIONAL_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap()
    else {
        panic!("proven");
    };
    let names: Vec<String> = ["x", "y", "w"].map(String::from).to_vec();
    let bytes = export_admg_conditional_obstruction(&diagram, &proof, &names, &ctx).unwrap();
    let consumed = consume_admg_conditional_obstruction_artifact(&bytes, &ctx).unwrap();
    assert_eq!(consumed.to_record(), proof.to_record());
    assert_eq!(consumed.query(), &q);
    let wire = AdmgConditionalObstructionWire::decode(&bytes).unwrap();
    wire.check_variable_names(&names).unwrap();
    assert_eq!(
        wire.check_variable_names(&["a".into(), "b".into(), "c".into()]),
        Err(AdmgConditionalArtifactError::NamesMismatch)
    );
    // The same proof on another diagram does not export.
    let unselected = scm(0).diagram();
    assert!(export_admg_conditional_obstruction(&unselected, &proof, &names, &ctx).is_err());

    let refusal = |wire: &AdmgConditionalObstructionWire| -> (&'static str, &'static str) {
        match consume_admg_conditional_obstruction_artifact(&wire.export().unwrap(), &ctx) {
            Err(IoError::AdmgConditional(inner)) => inner.refusal(),
            other => panic!("typed refusal: {:?}", other.map(|p| p.to_record())),
        }
    };
    let reseal = |mut w: AdmgConditionalObstructionWire| {
        w.premises_digest = w.expected_premises_digest().unwrap();
        w
    };
    // Unsealed: a changed query value is a premises mismatch.
    let mut w = wire.clone();
    w.proof.witness.first_value = w.proof.witness.second_value.clone();
    assert_eq!(refusal(&w), ("transport_not_certified", "admg_transport.premises_mismatch"));
    // Re-sealed witness mutations fail the exact verifier.
    let invalid = ("transport_not_certified", "admg_transport.invalid_derivation");
    assert_eq!(refusal(&reseal(w)), invalid);
    let mut w = wire.clone();
    w.proof.witness.second.source[1].ones[0] = "1/7".into();
    assert_eq!(refusal(&reseal(w)), invalid);
    let mut w = wire.clone();
    w.proof.witness.second = w.proof.witness.first.clone();
    w.proof.witness.second_value = w.proof.witness.first_value.clone();
    assert_eq!(refusal(&reseal(w)), invalid);
    // A re-sealed premise change: the witness no longer fits the diagram.
    let mut w = wire.clone();
    w.selections.clear();
    assert_eq!(refusal(&reseal(w)), invalid);
    // A forged move fails the reduction check.
    let mut w = wire.clone();
    w.proof.moves = vec![2];
    w.proof.remaining.clear();
    assert_eq!(refusal(&reseal(w)), invalid);
    // Oversized graph: refused before any digest, sealed or not.
    let mut w = wire.clone();
    w.graph.node_count = 7;
    w.variable_names.clear();
    assert_eq!(refusal(&w), ("route_not_supported", "admg_transport.consumer_limits"));
    // Another version or feature: refused by version before the payload, and by
    // feature marker (the point format's marker included).
    for version in [0, 2] {
        let mut w = wire.clone();
        w.version = version;
        assert!(matches!(
            consume_admg_conditional_obstruction_artifact(&w.export().unwrap(), &ctx),
            Err(IoError::AdmgConditional(AdmgConditionalArtifactError::UnsupportedVersion {
                version: v
            })) if v == version
        ));
    }
    for features in [vec!["checked_admg_conditional_point_v1".to_owned()], Vec::new()] {
        let mut w = wire.clone();
        w.required_features = features;
        assert!(matches!(
            consume_admg_conditional_obstruction_artifact(&w.export().unwrap(), &ctx),
            Err(IoError::AdmgConditional(AdmgConditionalArtifactError::UnsupportedSemantics(
                "required features"
            )))
        ));
    }
    // Cross-format: a point artifact is never read as an obstruction, nor the
    // reverse.
    let point = export(&prepare(&scm(0)));
    assert!(consume_admg_conditional_obstruction_artifact(&point, &ctx).is_err());
    assert!(AdmgConditionalObstructionWire::decode(&point).is_err());
    assert!(AdmgConditionalArtifactWire::decode(&bytes).is_err());
    assert!(
        antecedent::consume_admg_conditional_transport_artifact(
            &bytes,
            AdmgConditionalConsumeLimits::default(),
            &ctx
        )
        .is_err()
    );
}

#[test]
fn the_obstruction_wire_consumer_rebuilds_the_diagram_and_reverifies_the_witness() {
    use antecedent::export_admg_conditional_obstruction;
    use antecedent_io::admg_conditional_transport_artifact::AdmgConditionalObstructionWire;
    let blocked = Scm { selected: vec![1], ..scm(0) };
    let (catalog, _) = blocked.catalog_and_laws();
    let ctx = ExecutionContext::for_tests(5);
    let diagram = blocked.diagram();
    let q = query(&[1], &[0], &[2]);
    let ConditionalTransportDecision::ProvenNonTransportable(proof) =
        decide_admg_conditional_transport(
            &diagram,
            &q,
            &catalog,
            ADMG_CONDITIONAL_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap()
    else {
        panic!("proven");
    };
    let names: Vec<String> = ["x", "y", "w"].map(String::from).to_vec();
    let bytes = export_admg_conditional_obstruction(&diagram, &proof, &names, &ctx).unwrap();
    // The io consumer alone, with no producer state: the diagram it rebuilds
    // is the producer's, and the proof it re-verifies is the exported one.
    let consumed = AdmgConditionalObstructionWire::consume(&bytes, &ctx).unwrap();
    assert_eq!(consumed.diagram.selection_targets(), diagram.selection_targets());
    assert_eq!(
        format!("{:?}", consumed.diagram.causal_graph()),
        format!("{:?}", diagram.causal_graph())
    );
    assert_eq!(consumed.proof.to_record(), proof.to_record());
    assert_eq!(consumed.wire, AdmgConditionalObstructionWire::decode(&bytes).unwrap());
    // A re-sealed witness whose second model copies the first no longer
    // separates the query, so the exact verifier refuses it.
    let mut wire = consumed.wire.clone();
    wire.proof.witness.second = wire.proof.witness.first.clone();
    wire.proof.witness.second_value = wire.proof.witness.first_value.clone();
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    match AdmgConditionalObstructionWire::consume(&wire.export().unwrap(), &ctx) {
        Err(IoError::AdmgConditional(inner)) => assert_eq!(
            inner.refusal(),
            ("transport_not_certified", "admg_transport.invalid_derivation")
        ),
        other => panic!("typed refusal: {:?}", other.map(|c| c.proof.to_record())),
    }
}

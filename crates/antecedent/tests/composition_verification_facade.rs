//! Facade integration of composition verification: the classification is built
//! on the stage planner, the invalidation log, claim freshness and the evidence
//! ledger, and a fresh consumer replays the exported verification.
//!
//! Hand oracles (planner dependency rules, retarget licensed): replacing the
//! data snapshot recomputes the score artifact, the law and the decision and
//! leaves identification reused; changing the target population recomputes the
//! law and the decision and leaves the score artifact reused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent::analysis::composition_verification::{
    CalibrationDeclaration, CalibrationState, ClaimStanding, CompositionGraph, CompositionNode,
    CompositionVerification, DependencyKind, NodeKind, NumericalCause, ObservedEffect,
    Revalidation, WithdrawalKind, consume_composition_verification_artifact, freshness_effects,
    invalidation_effects, ledger_effects, query_revalidation, stage_effects, stage_node,
    stage_node_with_standing,
};
use std::sync::Arc;

use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, ObservationId, PoolingDependence, SourceDeclaration,
    SourceKind, SynthesisOperation,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ReasoningView, SlotAvailability,
};
use antecedent_core::recalc::{
    CalibrationInput, RecalcCapabilities, RecalcPlan, RetargetSupport, Stage, StageIdentities,
    StageIdentity,
};
use antecedent_core::{
    CancellationToken, ContractIdentities, QueryId, SemanticDigest, StateVersion,
};
use antecedent_state::{InvalidationLog, InvalidationTarget};

const STAGES: [Stage; 15] = [
    Stage::Graph,
    Stage::Query,
    Stage::Regime,
    Stage::Evidence,
    Stage::SourcePopulation,
    Stage::TargetPopulation,
    Stage::DataSnapshot,
    Stage::RowDesign,
    Stage::TreatmentGrid,
    Stage::LearnerFoldsRng,
    Stage::Utility,
    Stage::Identification,
    Stage::ScoreArtifact,
    Stage::Law,
    Stage::Decision,
];

fn identities(versions: &BTreeMap<Stage, &str>) -> StageIdentities {
    let mut ids = StageIdentities::new();
    for stage in STAGES {
        let version = versions.get(&stage).copied().unwrap_or("v1");
        ids.set(stage, StageIdentity::of(&stage.label(), &[version.as_bytes()]));
    }
    ids
}

fn plan(previous: &StageIdentities, requested: &StageIdentities) -> RecalcPlan {
    RecalcPlan::plan(
        previous,
        requested,
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
    )
}

fn d(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn fixture_claim() -> ClaimEnvelope {
    ClaimEnvelope::new(
        d(1),
        ContractIdentities::new(d(10), d(11), Some(d(12)), Some(d(13)), d(14), d(15), d(16)),
        ClaimKind::Point,
        Some(0.5),
        Some(Arc::from("units")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        Some(d(77)),
        [Arc::<str>::from("prior:p0")],
    )
}

/// data -> `score_fit` (structure: identification) -> law (target support) -> decision.
fn composition(plan: &RecalcPlan, law_calibration: CalibrationDeclaration) -> CompositionGraph {
    let mut g = CompositionGraph::new();
    let node = |key, kind, stage| stage_node(key, kind, plan, stage).expect("stage in plan");
    let identification = node("identification", NodeKind::Claim, Stage::Identification);
    let structure = identification.identity;
    g.add_node(identification).expect("node");
    g.add_node(node("data", NodeKind::Data, Stage::DataSnapshot)).expect("node");
    g.add_node(node("target", NodeKind::TargetPopulation, Stage::TargetPopulation)).expect("node");
    let target_identity = g.node("target").expect("target").identity;
    g.add_node(
        node("score_fit", NodeKind::LedgerResult, Stage::ScoreArtifact).with_structure(structure),
    )
    .expect("node");
    g.add_node(
        node("law", NodeKind::TransportedResult, Stage::Law)
            .with_coordinate(target_identity)
            .with_calibration(law_calibration),
    )
    .expect("node");
    g.add_node(node("decision", NodeKind::Decision, Stage::Decision)).expect("node");
    g.add_edge("score_fit", "identification", DependencyKind::StructuralDerivation).expect("edge");
    g.add_edge("score_fit", "data", DependencyKind::Data).expect("edge");
    g.add_edge("law", "score_fit", DependencyKind::Data).expect("edge");
    g.add_edge("law", "target", DependencyKind::Support).expect("edge");
    g.add_edge("decision", "law", DependencyKind::Data).expect("edge");
    g
}

const BINDINGS: [(&str, Stage); 5] = [
    ("identification", Stage::Identification),
    ("data", Stage::DataSnapshot),
    ("score_fit", Stage::ScoreArtifact),
    ("law", Stage::Law),
    ("decision", Stage::Decision),
];

#[test]
fn replacing_the_data_snapshot_recomputes_the_fit_and_keeps_no_old_authority() {
    let base = BTreeMap::new();
    let mut changed = BTreeMap::new();
    changed.insert(Stage::DataSnapshot, "v2");
    let (previous_ids, requested_ids) = (identities(&base), identities(&changed));
    let requested = plan(&previous_ids, &requested_ids);
    let previous_plan = plan(&previous_ids, &previous_ids);
    // planner oracle
    assert_eq!(requested.status(Stage::Identification).expect("status").tag(), "reused");
    for stage in [Stage::ScoreArtifact, Stage::Law, Stage::Decision] {
        assert_eq!(requested.status(stage).expect("status").tag(), "recomputed", "{stage}");
    }

    let before = composition(&previous_plan, CalibrationDeclaration::PointOnly);
    let after = composition(&requested, CalibrationDeclaration::PointOnly);
    let effects = stage_effects(&requested, &BINDINGS);
    let verification =
        CompositionVerification::verify(Some(&before), &after, &effects, None).expect("verify");
    let r = verification.revalidation();
    assert_eq!(r.keys_of(Revalidation::Recompute), vec!["data", "decision", "law", "score_fit"]);
    assert_eq!(r.keys_of(Revalidation::StillValid), vec!["identification", "target"]);
    // the structural derivation was reused; the fit and its uncertainty were not retained
    let fit = verification.classify("score_fit").expect("claim");
    assert!(fit.numerical.iter().any(|n| n.cause == NumericalCause::StructureReusedFitNotRetained));
    let refusal = verification.require_valid("decision").expect_err("stale");
    assert_eq!(refusal.code, "claim_revalidation_required");
    assert_eq!(refusal.detail, "composition_verification.recompute_required");
    assert!(refusal.remedy.is_some());
    assert!(verification.require_valid("identification").is_ok());
}

#[test]
fn a_target_change_recomputes_the_law_and_never_inherits_its_calibration() {
    let base = BTreeMap::new();
    let mut changed = BTreeMap::new();
    changed.insert(Stage::TargetPopulation, "v2");
    let (previous_ids, requested_ids) = (identities(&base), identities(&changed));
    let previous_plan = plan(&previous_ids, &previous_ids);
    let requested = plan(&previous_ids, &requested_ids);
    assert_eq!(requested.status(Stage::ScoreArtifact).expect("status").tag(), "reused");
    assert_eq!(requested.status(Stage::Law).expect("status").tag(), "recomputed");

    // the law held a record established for the previous target population
    let previous_target =
        stage_node("target", NodeKind::TargetPopulation, &previous_plan, Stage::TargetPopulation)
            .expect("target")
            .identity;
    let held = CalibrationDeclaration::Held { record: d(90), coordinate: previous_target };
    let before = composition(&previous_plan, held.clone());
    let after = composition(&requested, held);
    let effects = stage_effects(&requested, &BINDINGS);
    let verification =
        CompositionVerification::verify(Some(&before), &after, &effects, None).expect("verify");
    let law = verification.classify("law").expect("claim");
    assert_eq!(law.class, Revalidation::Withdrawn);
    assert_eq!(law.calibration.state, CalibrationState::Lost);
    assert!(law.calibration.newly_required);
    assert!(law.withdrawn_for(WithdrawalKind::Calibration));
    let refusal = verification.require_valid("law").expect_err("refused");
    assert_eq!(refusal.code, "cell_not_licensed");
    // the score artifact and identification do not depend on the target
    assert_eq!(verification.classify("score_fit").expect("c").class, Revalidation::StillValid);
    // the decision rests on a law that must be recomputed; its point value is not withdrawn
    assert_eq!(verification.classify("decision").expect("c").class, Revalidation::Recompute);
}

#[test]
fn a_declared_record_that_names_no_attesting_registry_record_reads_as_required() {
    use antecedent::analysis::composition_verification::{
        registry_record_attests, registry_record_identity, registry_resolved_calibration,
        unresolved_registry_calibrations,
    };
    let base = BTreeMap::new();
    let ids = identities(&base);
    let current = plan(&ids, &ids);
    let target =
        stage_node("target", NodeKind::TargetPopulation, &current, Stage::TargetPopulation)
            .expect("target")
            .identity;
    // A producer's invented digest names no shipped record.
    let invented = CalibrationDeclaration::Held { record: d(90), coordinate: target };
    assert!(!registry_record_attests(&d(90)));
    assert!(!registry_record_attests(&registry_record_identity("cov.no_such_record")));
    assert_eq!(registry_resolved_calibration(&invented), CalibrationDeclaration::Required);
    assert_eq!(
        registry_resolved_calibration(&CalibrationDeclaration::PointOnly),
        CalibrationDeclaration::PointOnly
    );

    let graph = composition(&current, invented);
    assert_eq!(unresolved_registry_calibrations(&graph), vec!["law".to_owned()]);
    let plain =
        CompositionVerification::verify(None, &graph, &BTreeMap::new(), None).expect("declared");
    assert_eq!(plain.classify("law").expect("claim").calibration.state, CalibrationState::Held);
    let resolved =
        CompositionVerification::verify_against_registry(None, &graph, &BTreeMap::new(), None)
            .expect("resolved");
    assert_eq!(
        resolved.classify("law").expect("claim").calibration.state,
        CalibrationState::Required
    );
}

#[test]
fn a_fresh_consumer_replays_the_export_and_refuses_tampering() {
    let base = BTreeMap::new();
    let mut changed = BTreeMap::new();
    changed.insert(Stage::DataSnapshot, "v2");
    let (previous_ids, requested_ids) = (identities(&base), identities(&changed));
    let requested = plan(&previous_ids, &requested_ids);
    let previous_plan = plan(&previous_ids, &previous_ids);
    let before = composition(&previous_plan, CalibrationDeclaration::PointOnly);
    let after = composition(&requested, CalibrationDeclaration::PointOnly);
    let effects = stage_effects(&requested, &BINDINGS);
    let verification =
        CompositionVerification::verify(Some(&before), &after, &effects, None).expect("verify");
    let bytes = verification.export("composition-facade").expect("export");

    let consumed = consume_composition_verification_artifact(&bytes, Some(verification.identity()))
        .expect("consume");
    assert_eq!(consumed.identity(), verification.identity());
    assert_eq!(consumed.revalidation(), verification.revalidation());
    // the decision rests on the law, the score fit, the identification, the data and the target
    assert_eq!(consumed.view("decision").expect("view").upstream.len(), 5);
    assert!(consumed.require_valid("decision").is_err());

    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    assert!(consume_composition_verification_artifact(&flipped, None).is_err());
    let other_identity = "0".repeat(64);
    let wrong = consume_composition_verification_artifact(&bytes, Some(other_identity.as_str()))
        .expect_err("retained identity");
    assert_eq!(wrong.detail, "composition_verification.identity_mismatch");
    assert!(wrong.remedy.is_some());
}

#[test]
fn invalidation_log_and_claim_freshness_drive_the_classification() {
    let mut g = CompositionGraph::new();
    for (key, kind, id) in [
        ("query_result", NodeKind::LedgerResult, 1),
        ("model", NodeKind::MechanismComposition, 2),
        ("report", NodeKind::Claim, 3),
        ("other_report", NodeKind::Claim, 4),
    ] {
        g.add_node(CompositionNode::new(key, kind, d(id))).expect("node");
    }
    g.add_edge("report", "query_result", DependencyKind::Data).expect("edge");
    g.add_edge("other_report", "model", DependencyKind::ModelVersion).expect("edge");

    let mut log = InvalidationLog::new();
    log.push(
        StateVersion::from_raw(3),
        InvalidationTarget::QueryResult(QueryId::from_raw(7)),
        "new rows arrived",
    );
    let bindings = [
        ("query_result", InvalidationTarget::QueryResult(QueryId::from_raw(7))),
        ("model", InvalidationTarget::Model(antecedent_core::ModelId::from_raw(1))),
    ];
    // before the entry: nothing; after it: the bound query result recomputes, the model does not
    assert!(invalidation_effects(&log, StateVersion::from_raw(3), &bindings).is_empty());
    assert_eq!(
        query_revalidation(&log, QueryId::from_raw(7), StateVersion::from_raw(3)),
        Revalidation::StillValid
    );
    let effects = invalidation_effects(&log, StateVersion::from_raw(2), &bindings);
    assert_eq!(
        query_revalidation(&log, QueryId::from_raw(7), StateVersion::from_raw(2)),
        Revalidation::Recompute
    );
    assert_eq!(effects.keys().map(String::as_str).collect::<Vec<_>>(), vec!["query_result"]);
    let verification =
        CompositionVerification::verify(Some(&g), &g, &effects, None).expect("verify");
    assert_eq!(verification.classify("report").expect("c").class, Revalidation::Recompute);
    assert_eq!(verification.classify("other_report").expect("c").class, Revalidation::StillValid);

    // a withdrawn claim withdraws what is built on it; an offline receiver learns nothing
    let claim = fixture_claim();
    let withdrawn = claim.freshness(None, None, None, true, true);
    let offline = claim.freshness(None, None, None, false, true);
    // a snapshot that no longer matches asks for recomputation, not withdrawal
    let moved_on = claim.freshness(None, Some(d(200)), None, false, true);
    assert!(matches!(
        ObservedEffect::from_freshness(&moved_on),
        Some(ObservedEffect::Recomputed { .. })
    ));
    let effects = freshness_effects(&[("model", &withdrawn), ("query_result", &offline)]);
    assert_eq!(effects.len(), 1);
    assert!(matches!(effects.get("model"), Some(ObservedEffect::Withdrawn { .. })));
    let verification =
        CompositionVerification::verify(Some(&g), &g, &effects, None).expect("verify");
    assert_eq!(verification.classify("other_report").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(verification.classify("report").expect("c").class, Revalidation::StillValid);
    let refusal = verification.require_valid("other_report").expect_err("withdrawn");
    assert_eq!(refusal.code, "claim_license_withdrawn");
    assert_eq!(refusal.detail, "composition_verification.input_retired");
}

fn obs(range: std::ops::Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

#[test]
fn the_ledger_supplies_standings_overlap_and_a_state_the_consumer_checks() {
    let mut ledger = EvidenceLedger::new();
    let mut add = |label: &str, range| {
        ledger
            .add_source(
                SourceDeclaration::new(label, SourceKind::ObservationSet)
                    .with_observations(obs(range)),
            )
            .expect("source")
    };
    let (s1, s2, s3) = (add("s1", 0..10), add("s2", 5..15), add("s3", 20..30));
    let pooled = |sources: &[_]| CombinationRequest {
        label: "pooled".to_owned(),
        operation: SynthesisOperation::StatisticalPooling {
            sources: sources.to_vec(),
            dependence: PoolingDependence::Independent,
        },
        replaces: None,
    };
    let r_13 = ledger.combine(pooled(&[s1, s3])).expect("pool");

    let mut g = CompositionGraph::new();
    for (key, kind, id, sources) in [
        ("study_1", NodeKind::CohortSource, 1, vec![s1]),
        ("study_2", NodeKind::CohortSource, 2, vec![s2]),
        ("study_3", NodeKind::CohortSource, 3, vec![s3]),
    ] {
        g.add_node(CompositionNode::new(key, kind, d(id)).with_evidence(sources)).expect("node");
    }
    g.add_node(CompositionNode::new("pooled_13", NodeKind::LedgerResult, d(4))).expect("node");
    g.add_node(CompositionNode::new("decision", NodeKind::Decision, d(5))).expect("node");
    g.add_node(CompositionNode::new("overlap_check", NodeKind::Claim, d(6))).expect("node");
    g.add_edge("overlap_check", "study_1", DependencyKind::Evidence).expect("edge");
    g.add_edge("overlap_check", "study_2", DependencyKind::Evidence).expect("edge");
    g.add_edge("pooled_13", "study_1", DependencyKind::Evidence).expect("edge");
    g.add_edge("pooled_13", "study_3", DependencyKind::Evidence).expect("edge");
    g.add_edge("decision", "pooled_13", DependencyKind::Data).expect("edge");

    let cancel = CancellationToken::new();
    let effects = ledger_effects(&ledger, &[("pooled_13", r_13)]).expect("effects");
    assert!(effects.is_empty());
    let verification =
        CompositionVerification::verify(None, &g, &effects, Some((&ledger, &cancel))).expect("v");
    let view = verification.view("decision").expect("view");
    // study_2 is not in the decision's lineage, so the overlap with study_1 is not either
    assert!(view.shared_data.is_empty());
    assert_eq!(view.evidence.len(), 2);
    // rows 5..10 are shared by study_1 and study_2
    let overlap = verification.view("overlap_check").expect("view");
    assert_eq!(overlap.shared_data.len(), 1);
    assert_eq!(
        (overlap.shared_data[0].relation, overlap.shared_data[0].shared_observations),
        ("shared_data", 5)
    );

    let bytes = verification.export("composition-ledger").expect("export");
    let consumed = consume_composition_verification_artifact(&bytes, None).expect("consume");
    consumed.verify_ledger(&ledger, &cancel).expect("same ledger state");
    // after a withdrawal the consumer's ledger is at another state than the one declared
    ledger.withdraw_source(s1, "consent withdrawn").expect("withdraw");
    let refusal = consumed.verify_ledger(&ledger, &cancel).expect_err("moved");
    assert_eq!(refusal.detail, "composition_verification.ledger_mismatch");
    // classification of the new standing: the pooled result of two now has one member
    let effects = ledger_effects(&ledger, &[("pooled_13", r_13)]).expect("effects");
    let moved = CompositionVerification::verify(Some(&g), &g, &effects, Some((&ledger, &cancel)))
        .expect("verify");
    assert_eq!(moved.classify("pooled_13").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(moved.classify("decision").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(moved.classify("study_3").expect("c").class, Revalidation::StillValid);
}

/// The law, stated with the calibration record the plan's calibration input names.
fn calibrated_composition(plan: &RecalcPlan, record: u8) -> CompositionGraph {
    let g = composition(plan, CalibrationDeclaration::PointOnly);
    let target = g.node("target").expect("target").identity;
    let law = stage_node_with_standing(
        "law",
        NodeKind::TransportedResult,
        plan,
        Stage::Law,
        ClaimStanding::native_point().with_calibration(CalibrationDeclaration::Held {
            record: d(record),
            coordinate: target,
        }),
    )
    .expect("law in plan")
    .with_coordinate(target);
    // rebuild with the stated standing: nodes are immutable once added
    let mut out = CompositionGraph::new();
    for node in g.nodes().filter(|n| n.key != "law") {
        out.add_node(node.clone()).expect("node");
    }
    out.add_node(law).expect("node");
    for edge in g.edges() {
        out.add_edge(&edge.downstream, &edge.upstream, edge.kind).expect("edge");
    }
    out
}

fn calibrated(record: Option<(u8, u8)>) -> (StageIdentities, StageIdentities) {
    let mut previous = identities(&BTreeMap::new());
    previous.set_calibration(&CalibrationInput::Held {
        record: StageIdentity::of("record", &[&[90]]),
        coordinate: StageIdentity::of("coordinate", &[&[100]]),
    });
    let mut requested = identities(&BTreeMap::new());
    if let Some((record, coordinate)) = record {
        requested.set_calibration(&CalibrationInput::Held {
            record: StageIdentity::of("record", &[&[record]]),
            coordinate: StageIdentity::of("coordinate", &[&[coordinate]]),
        });
    }
    (previous, requested)
}

#[test]
fn a_re_measured_calibration_record_recomputes_the_law_and_leaves_the_fit_valid() {
    let (previous_ids, requested_ids) = calibrated(Some((91, 100)));
    let requested = plan(&previous_ids, &requested_ids);
    let previous_plan = plan(&previous_ids, &previous_ids);
    assert_eq!(requested.revalidation(Stage::Law), Some(Revalidation::Recompute));
    assert_eq!(requested.revalidation(Stage::ScoreArtifact), Some(Revalidation::StillValid));
    let before = calibrated_composition(&previous_plan, 90);
    let after = calibrated_composition(&requested, 91);
    let effects = stage_effects(&requested, &BINDINGS);
    let verification =
        CompositionVerification::verify(Some(&before), &after, &effects, None).expect("verify");
    assert_eq!(verification.classify("law").expect("c").class, Revalidation::Recompute);
    assert_eq!(verification.classify("decision").expect("c").class, Revalidation::Recompute);
    assert_eq!(verification.classify("score_fit").expect("c").class, Revalidation::StillValid);
    assert_eq!(verification.classify("data").expect("c").class, Revalidation::StillValid);
}

#[test]
fn a_calibration_input_that_is_gone_withdraws_the_law_and_the_decision() {
    let (previous_ids, requested_ids) = calibrated(None);
    let requested = plan(&previous_ids, &requested_ids);
    let previous_plan = plan(&previous_ids, &previous_ids);
    assert_eq!(requested.revalidation(Stage::Law), Some(Revalidation::Withdrawn));
    assert_eq!(requested.revalidation(Stage::Decision), Some(Revalidation::Withdrawn));
    // the plan's class is adapted into the claim graph as an observed effect
    let mut effects = BTreeMap::new();
    for (key, stage) in [("law", Stage::Law), ("decision", Stage::Decision)] {
        let class = requested.revalidation(stage).expect("stage");
        if let Some(effect) = ObservedEffect::from_revalidation(class, format!("plan:{stage}")) {
            effects.insert(key.to_owned(), effect);
        }
    }
    let before = calibrated_composition(&previous_plan, 90);
    let after = calibrated_composition(&previous_plan, 90);
    let verification =
        CompositionVerification::verify(Some(&before), &after, &effects, None).expect("verify");
    assert_eq!(verification.classify("law").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(verification.classify("decision").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(verification.classify("score_fit").expect("c").class, Revalidation::StillValid);
}

#[test]
fn a_removed_premise_withdraws_a_bound_result_and_data_only_recomputes_it() {
    use antecedent_core::recalc::{ChangeKind, StageIdentity as Id};
    use antecedent_state::ChangedInput;
    let mut log = InvalidationLog::new();
    let at = |n| StateVersion::from_raw(n);
    let bound = [("report", InvalidationTarget::QueryResult(QueryId::from_raw(7)))];
    log.push_input(
        at(2),
        InvalidationTarget::AllResults,
        "append_data",
        ChangedInput {
            stage: Stage::DataSnapshot,
            change: ChangeKind::Modified,
            identity: Id::of("data", &[b"1"]),
        },
    );
    let effects = invalidation_effects(&log, at(1), &bound);
    assert!(matches!(effects.get("report"), Some(ObservedEffect::Recomputed { .. })));
    assert_eq!(query_revalidation(&log, QueryId::from_raw(7), at(1)), Revalidation::Recompute);
    log.push_input(
        at(3),
        InvalidationTarget::AllResults,
        "remove_constraint",
        ChangedInput {
            stage: Stage::Graph,
            change: ChangeKind::Removed,
            identity: Id::of("constraint", &[b"c1"]),
        },
    );
    let effects = invalidation_effects(&log, at(1), &bound);
    assert!(matches!(effects.get("report"), Some(ObservedEffect::Withdrawn { .. })));
    assert_eq!(query_revalidation(&log, QueryId::from_raw(7), at(1)), Revalidation::Withdrawn);
    // only the entries after the observer's version count
    assert!(matches!(
        invalidation_effects(&log, at(2), &bound).get("report"),
        Some(ObservedEffect::Withdrawn { .. })
    ));
    assert_eq!(query_revalidation(&log, QueryId::from_raw(7), at(3)), Revalidation::StillValid);
}

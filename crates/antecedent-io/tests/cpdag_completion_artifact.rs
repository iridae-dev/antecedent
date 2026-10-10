//! 2.3A A1 (X2): independent replay of the DAG completions of a CPDAG.
//!
//! The fixture is the one the estimate-layer oracle uses: the chain `a - b - c` with
//! three completions, one enumerated joint law of the SCM `a -> b -> c`, and the
//! query `P(c | do(b = 1))`. A fresh consumer must reproduce the report from the
//! artifact alone and refuse every re-sealed premise or evidence change.
#![allow(clippy::float_cmp, reason = "exact replay comparisons")]

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, Value, VariableDomain, VariableId,
};
use antecedent_estimate::cpdag_scenarios::CpdagScenarioReport;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    LawTolerance,
};
use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::cpdag_completion::{
    CompletionEvidenceBinding, CpdagCompletionInput, CpdagEvidence,
};
use antecedent_identify::sid::scenarios::ScenarioCoordinate;
use antecedent_io::IoError;
use antecedent_io::cpdag_completion_artifact::{
    CpdagCompletionArtifactWire, CpdagConsumeLimits, CpdagScenarioPremises,
};

const SEARCH: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// `P(a = 1) = 0.4`, `P(b = 1 | a) = 0.2, 0.7`, `P(c = 1 | b) = 0.1, 0.8`.
fn joint(a: usize, b: usize, c: usize) -> f64 {
    let p = |one: bool, p_one: f64| if one { p_one } else { 1.0 - p_one };
    p(a == 1, 0.4) * p(b == 1, [0.2, 0.7][a]) * p(c == 1, [0.1, 0.8][b])
}

fn axis(variable: u32) -> DiscreteAxis {
    DiscreteAxis { variable: v(variable), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

fn laws() -> ExactTransportData {
    let mut table = [0.0; 8];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                table[a * 4 + b * 2 + c] = joint(a, b, c);
            }
        }
    }
    let law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(0), axis(1), axis(2)],
        table,
        "target",
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new(vec![law], 1000).unwrap()
}

fn catalog(measured: &[u32]) -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        Vec::<VariableId>::new(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new([], vec![regime], [], None).unwrap()
}

fn coordinates() -> Arc<[ScenarioCoordinate]> {
    (0..3)
        .map(|i| ScenarioCoordinate {
            variable: v(i),
            name: Arc::from(format!("v{i}")),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

fn query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn chain() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_undirected(d(0), d(1)).unwrap();
    graph.insert_undirected(d(1), d(2)).unwrap();
    graph
}

fn shared() -> CpdagEvidence {
    CpdagEvidence::Shared {
        evidence_identity: Arc::from("target-law"),
        catalog: catalog(&[0, 1, 2]),
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn premises(
    evidence: CpdagEvidence,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> CpdagScenarioPremises {
    CpdagScenarioPremises {
        input: CpdagCompletionInput::new(chain()),
        coordinates: coordinates(),
        query: query(),
        evidence,
        budget,
        memory_limit_bytes: ctx.memory.hard_limit_bytes,
        data: laws(),
        request: Assignment::from_pairs([(v(1), Value::f64(1.0))]),
        evaluation: ExactEvaluationLimits::default(),
    }
}

/// Run the producer and export, returning the artifact and the original report.
fn produce(
    premises: &CpdagScenarioPremises,
    ctx: &ExecutionContext,
) -> (Vec<u8>, CpdagScenarioReport) {
    let prepared = premises.prepare(ctx).unwrap();
    let report = prepared.evaluate(ctx).unwrap();
    let bytes = CpdagCompletionArtifactWire::checked(premises, &prepared, &report)
        .unwrap()
        .export()
        .unwrap();
    (bytes, report)
}

fn consume(bytes: &[u8]) -> Result<CpdagScenarioReport, IoError> {
    CpdagCompletionArtifactWire::consume_with_limits(bytes, CpdagConsumeLimits::default(), &ctx())
        .map(|(_, report)| report)
}

fn refused(result: Result<CpdagScenarioReport, IoError>) -> (&'static str, String) {
    match result {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a coded refusal, got {:?}", other.map(|r| r.completions.len())),
    }
}

fn assert_same(first: &CpdagScenarioReport, second: &CpdagScenarioReport) {
    assert_eq!(first.cpdag_identity, second.cpdag_identity);
    assert_eq!(first.completions, second.completions);
    assert_eq!(
        (first.not_enumerated, first.identified, first.unidentified, first.unevaluated),
        (second.not_enumerated, second.identified, second.unidentified, second.unevaluated)
    );
    assert_eq!(first.receipt, second.receipt);
    let envelope = |r: &CpdagScenarioReport| r.report.as_ref().and_then(|s| s.envelope.clone());
    assert_eq!(envelope(first), envelope(second));
    let masses = |r: &CpdagScenarioReport| r.masses().to_vec();
    assert_eq!(masses(first), masses(second));
}

fn edit(
    bytes: &[u8],
    change: impl FnOnce(&mut CpdagCompletionArtifactWire),
) -> CpdagCompletionArtifactWire {
    let mut wire = CpdagCompletionArtifactWire::decode(bytes).unwrap();
    change(&mut wire);
    wire
}

fn reseal(mut wire: CpdagCompletionArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

#[test]
fn x2_cpdag_artifact_replay_equals_original() {
    let ctx = ctx();
    let (bytes, original) = produce(&premises(shared(), SEARCH, &ctx), &ctx);
    assert_eq!(original.completions.len(), 3);
    assert!(original.is_exportable());
    let replayed = consume(&bytes).unwrap();
    assert_same(&original, &replayed);
    // The artifact binds every completion's identity, edges, status and evidence.
    let wire = CpdagCompletionArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.report.completions.len(), 3);
    assert!(wire.report.completions.iter().all(|c| {
        c.status == "identified"
            && c.evidence_identity.as_deref() == Some("target-law")
            && c.id.len() == 64
    }));
    assert_eq!(wire.premises_digest, wire.expected_premises_digest().unwrap());
    assert_eq!(wire.data_digest, wire.expected_data_digest().unwrap());
    // Unweighted: no mass anywhere in the stored scenario report.
    let stored = wire.report.scenario_report.as_ref().unwrap();
    assert!(stored.masses.iter().all(|(_, _, mass)| mass.is_none()));
    assert!(stored.weighted.is_none() && stored.residual_mass.is_none());
    // Byte-stable: exporting the decoded wire reproduces the artifact.
    assert_eq!(wire.export().unwrap(), bytes);
}

#[test]
fn x2_cpdag_artifact_resealed_changes_are_refused() {
    let ctx = ctx();
    let (bytes, _) = produce(&premises(shared(), SEARCH, &ctx), &ctx);

    // A changed completion: its identity or its edges.
    let (code, message) = refused(consume(&reseal(edit(&bytes, |w| {
        w.report.completions[0].id = "0".repeat(64);
    }))));
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
    let (_, message) = refused(consume(&reseal(edit(&bytes, |w| {
        w.report.completions[1].edges.reverse();
    }))));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");

    // A changed snapshot: the exact law's masses (a swapped pair of cells), digests re-sealed.
    let (_, message) = refused(consume(&reseal(edit(&bytes, |w| {
        w.laws[0].probabilities.swap(0, 1);
    }))));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
    // ... and, un-resealed, the data identity refuses first.
    let (_, message) =
        refused(consume(&edit(&bytes, |w| w.laws[0].snapshot = "forged".into()).export().unwrap()));
    assert!(message.starts_with("cpdag_scenarios.data_identity_mismatch"), "{message}");

    // A changed evidence binding.
    let (_, message) = refused(consume(&reseal(edit(&bytes, |w| {
        w.evidence.shared.as_mut().unwrap().identity = "forged".into();
    }))));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
    let (_, message) = refused(consume(
        &edit(&bytes, |w| w.evidence.shared.as_mut().unwrap().identity = "forged".into())
            .export()
            .unwrap(),
    ));
    assert!(message.starts_with("cpdag_scenarios.data_identity_mismatch"), "{message}");

    // A changed CPDAG: one edge fewer, re-sealed; un-resealed, the premises refuse.
    let drop_edge = |w: &mut CpdagCompletionArtifactWire| w.cpdag.undirected = vec![(0, 1)];
    let (_, message) = refused(consume(&reseal(edit(&bytes, drop_edge))));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
    let (_, message) = refused(consume(&edit(&bytes, drop_edge).export().unwrap()));
    assert!(message.starts_with("cpdag_scenarios.premises_mismatch"), "{message}");

    // A changed count: a completion the replay finds but the report omits.
    let (_, message) = refused(consume(&reseal(edit(&bytes, |w| {
        w.report.completions.pop();
    }))));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
}

#[test]
fn x2_cpdag_artifact_per_completion_evidence_replays_and_binds_identities() {
    let ctx = ctx();
    let ids = produce(&premises(shared(), SEARCH, &ctx), &ctx).1;
    let bind = |index: usize, name: &str, measured: &[u32]| CompletionEvidenceBinding {
        completion: Arc::clone(&ids.completions[index].id),
        certified_for: Arc::clone(&ids.completions[index].id),
        evidence_identity: Arc::from(name),
        catalog: catalog(measured),
    };
    // The third completion's catalog lacks the outcome; the second has no binding.
    let evidence =
        CpdagEvidence::PerCompletion(vec![bind(0, "ev-0", &[0, 1, 2]), bind(2, "ev-2", &[0, 1])]);
    let (bytes, original) = produce(&premises(evidence, SEARCH, &ctx), &ctx);
    assert_eq!((original.identified, original.unidentified, original.unevaluated), (1, 2, 0));
    assert_same(&original, &consume(&bytes).unwrap());
    let wire = CpdagCompletionArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.report.completions[1].evidence_identity, None);
    assert_eq!(wire.report.completions[2].evidence_identity.as_deref(), Some("ev-2"));

    // Evidence certified for another completion never satisfies this one.
    let forged = edit(&bytes, |w| {
        let other = w.evidence.bindings[1].completion.clone();
        w.evidence.bindings[0].certified_for = other;
    });
    let (code, message) = refused(consume(&reseal(forged)));
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("cpdag_scenarios.evidence_identity_mismatch"), "{message}");

    // A binding moved to another completion changes the replayed receipt.
    let moved = edit(&bytes, |w| {
        let second = ids.completions[1].id.to_string();
        w.evidence.bindings[0].completion = second.clone();
        w.evidence.bindings[0].certified_for = second;
    });
    let (_, message) = refused(consume(&reseal(moved)));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
}

#[test]
fn x2_cpdag_artifact_budget_stop_exports_and_replays_the_prefix() {
    let ctx = ctx();
    // Enumeration stops after one of three completions: two are never enumerated.
    let limits = SearchLimits { operations: 3, depth: 256 };
    let (bytes, original) = produce(&premises(shared(), limits, &ctx), &ctx);
    assert!(!original.is_exportable());
    assert_eq!((original.completions.len(), original.not_enumerated), (1, 3));
    assert_eq!(original.completions[0].status, "unevaluated");
    let replayed = consume(&bytes).unwrap();
    assert_same(&original, &replayed);
    let wire = CpdagCompletionArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.report.not_enumerated, 3);
    assert_eq!(wire.report.unevaluated, 4);
    let receipt = wire.report.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, "search.operations");
    assert_eq!(receipt.explored.len(), 1, "the one enumerated completion");
    assert_eq!(
        receipt.unevaluated,
        vec!["cpdag_completions_not_enumerated_upper_bound:3".to_string()]
    );

    // The recorded limit cannot be raised by the artifact: a stricter consumer refuses.
    let strict = CpdagConsumeLimits {
        search_budget: SearchLimits { operations: 2, depth: 256 },
        ..CpdagConsumeLimits::default()
    };
    let (code, message) = refused(
        CpdagCompletionArtifactWire::consume_with_limits(&bytes, strict, &ctx)
            .map(|(_, report)| report),
    );
    assert_eq!(code, "transport_budget_cancel");
    assert!(message.starts_with("cpdag_scenarios.consumer_limit_exceeded"), "{message}");
    // A changed recorded budget changes the replayed prefix.
    let widened = edit(&bytes, |w| w.search_operations = 6);
    let (_, message) = refused(consume(&reseal(widened)));
    assert!(message.starts_with("cpdag_scenarios.report_replay_mismatch"), "{message}");
}

#[test]
fn x2_cpdag_artifact_cancelled_report_never_exports() {
    let live = ctx();
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let premises = premises(shared(), SEARCH, &cancelled);
    let prepared = premises.prepare(&cancelled).unwrap();
    let report = prepared.evaluate(&live).unwrap();
    assert!(report.completions.is_empty() && report.not_enumerated == 4);
    assert!(!report.is_exportable());
    match CpdagCompletionArtifactWire::checked(&premises, &prepared, &report) {
        Err(IoError::Refused { code, message }) => {
            assert_eq!(code, "cancelled_no_claim");
            assert!(message.starts_with("cpdag_scenarios.cancelled_not_exportable"), "{message}");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn x2_cpdag_artifact_unknown_version_and_foreign_features_refuse() {
    let ctx = ctx();
    let (bytes, _) = produce(&premises(shared(), SEARCH, &ctx), &ctx);
    let future = edit(&bytes, |w| w.version = 2).export().unwrap();
    assert!(matches!(
        CpdagCompletionArtifactWire::decode(&future),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert!(matches!(consume(&future), Err(IoError::UnsupportedVersion { version: 2 })));
    let foreign =
        edit(&bytes, |w| w.required_features.push("other_feature_v9".into())).export().unwrap();
    let (code, message) = refused(consume(&foreign));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("cpdag_scenarios.unsupported_semantics"), "{message}");
    // Evidence mode and fields that disagree refuse as unsupported semantics.
    let confused = edit(&bytes, |w| w.evidence.mode = "per_completion".into());
    let (_, message) = refused(consume(&reseal(confused)));
    assert!(message.starts_with("cpdag_scenarios.unsupported_semantics"), "{message}");
}

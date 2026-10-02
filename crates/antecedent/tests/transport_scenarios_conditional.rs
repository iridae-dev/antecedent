//! Conditional ADMG transport questions inside the finite scenario envelope
//! (X2: the 2.2B B1 row promoted into the 2.2A A2 scenario system).
//!
//! Every scenario is a latent-confounded selection ADMG over `x(0)`, `y(1)`,
//! `w(2)` and the question is `P*(y | do(x), w)`:
//!
//! - `chain` (`x -> y -> w`, `x <-> y`, selection on `w`): `w` cannot move, so
//!   the reduced joint `P*(y, w | do(x))` is normalized at `w`;
//! - `parent` (`x -> y <- w`, `x <-> y`, selection on `w`): rule 2 moves `w`,
//!   so the scenario's answer is the unconditional `P*(y | do(x, w))` of the
//!   classical route, with nothing left to normalize;
//! - `shifted` (`chain` with selection on `y`): the reduced joint has an s-hedge.
//!   The scenario is `not_certified` with the inspection-only candidate, unless
//!   the route's witness stage finds an exactly verified two-model witness, which
//!   alone makes it `structurally_unidentified`; it is never identified;
//! - `unconfounded` (`chain` without `x <-> y`): identified from the target law.
//!
//! Every law and every truth is enumerated from latent SCMs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/admg_conditional_scm.rs"]
mod admg_conditional_scm;

use std::sync::Arc;

use admg_conditional_scm::{Scm, query, request};
use antecedent::{PreparedTransportScenarios, StudyBuilder, consume_transport_scenarios_artifact};
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, SearchLimits, SearchStop, TransportOutcomeKind,
    VariableDomain, VariableId,
};
use antecedent_estimate::transport_scenarios::ScenarioSetReport;
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    ClassicalTransportQuery,
    sid::scenarios::{
        ScenarioCoordinate, ScenarioOutcome, ScenarioQuestion, TransportScenario,
        TransportScenarioSet, decide_conditional_transport_scenarios,
    },
};
use antecedent_io::IoError;
use antecedent_io::transport_scenario_artifact::{
    TRANSPORT_SCENARIO_ARTIFACT_FEATURE, TRANSPORT_SCENARIO_ARTIFACT_VERSION,
    TRANSPORT_SCENARIO_CONDITIONAL_ARTIFACT_VERSION, TRANSPORT_SCENARIO_CONDITIONAL_FEATURE,
    TransportScenarioArtifactError, TransportScenarioArtifactWire, TransportScenarioConsumeLimits,
};

const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };
/// Request level bits over (x, y, w): x = 1, w = 1.
const LEVEL: usize = 0b101;

type Edges = Vec<(usize, usize)>;

/// `(name, directed, bidirected, selected)` of every scenario.
fn shapes() -> Vec<(&'static str, Edges, Edges, Vec<usize>)> {
    vec![
        ("chain", vec![(0, 1), (1, 2)], vec![(0, 1)], vec![2]),
        ("parent", vec![(0, 1), (2, 1)], vec![(0, 1)], vec![2]),
        ("shifted", vec![(0, 1), (1, 2)], vec![(0, 1)], vec![1]),
        ("unconfounded", vec![(0, 1), (1, 2)], vec![], vec![2]),
    ]
}

fn scm_of(name: &str, params: usize) -> Scm {
    let (_, directed, bidirected, selected) =
        shapes().into_iter().find(|(n, ..)| *n == name).unwrap();
    Scm { n: 3, directed, bidirected, selected, params, target_zero: Vec::new() }
}

fn coordinates() -> Arc<[ScenarioCoordinate]> {
    ["x", "y", "w"]
        .into_iter()
        .enumerate()
        .map(|(i, name)| ScenarioCoordinate {
            variable: VariableId::from_raw(u32::try_from(i).unwrap()),
            name: Arc::from(name),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

fn scenarios(names: &[&str], weights: Option<&[f64]>) -> Vec<TransportScenario> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| TransportScenario {
            name: Arc::from(*name),
            diagram: scm_of(name, 0).diagram(),
            weight: weights.map(|w| w[i]),
            coordinates: coordinates(),
        })
        .collect()
}

const ALL: [&str; 4] = ["chain", "parent", "shifted", "unconfounded"];

fn try_prepare(
    set: Vec<TransportScenario>,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    at: Assignment,
    budget: SearchLimits,
) -> Result<PreparedTransportScenarios, IoError> {
    StudyBuilder::conditional_transport_scenarios(
        &TransportScenarioSet::try_new(set).unwrap(),
        query(&[1], &[0], &[2]),
        catalog,
        budget,
        data,
        at,
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    )
}

/// The set over `names`, with every law enumerated from the `truth` scenario's SCM.
fn prepare(names: &[&str], truth: &str) -> PreparedTransportScenarios {
    let (catalog, data) = scm_of(truth, 0).catalog_and_laws();
    try_prepare(scenarios(names, None), catalog, data, request(&[0], &[2], LEVEL), BUDGET).unwrap()
}

fn row<'a>(
    report: &'a ScenarioSetReport,
    name: &str,
) -> &'a antecedent_estimate::transport_scenarios::ScenarioResult {
    report.scenarios.iter().find(|s| &*s.name == name).unwrap()
}

fn p_one(report: &ScenarioSetReport, name: &str) -> f64 {
    row(report, name).distribution.as_ref().unwrap().probabilities[1]
}

fn count(report: &ScenarioSetReport, status: &str) -> usize {
    report.masses.iter().find(|m| m.status == status).unwrap().count
}

#[test]
fn conditional_scenarios_enter_the_envelope_and_the_truth_lies_inside_it() {
    let prepared = prepare(&ALL, "chain");
    let ctx = ExecutionContext::for_tests(1);
    let report = prepared.estimate(&ctx).unwrap();
    let statuses = report.scenarios.iter().map(|s| s.status).collect::<Vec<_>>();
    let negative = shifted_status(&report);
    assert_eq!(statuses, ["identified", "identified", negative, "identified"]);
    // The true scenario's point is the enumerated target truth.
    let truth = scm_of("chain", 0).truth(&[1], &[0], &[2], LEVEL).unwrap();
    assert!((p_one(&report, "chain") - truth[1]).abs() < 1e-12);
    // The envelope ranges over the identified scenarios only, and holds the truth.
    let envelope = report.envelope.as_ref().unwrap();
    assert_eq!(
        envelope.scenarios.iter().map(|s| &**s).collect::<Vec<_>>(),
        ["chain", "parent", "unconfounded"]
    );
    let mean = &envelope.means[0];
    assert!(mean.lower <= truth[1] && truth[1] <= mean.upper, "{mean:?} vs {}", truth[1]);
    assert!(mean.upper - mean.lower > 1e-3, "the scenarios disagree: {mean:?}");
    // Status accounting is A2's: the s-hedge scenario is retained under its own
    // status, outside the envelope.
    assert_eq!(count(&report, "identified"), 3);
    assert_eq!(count(&report, negative), 1);
    let other =
        if negative == "not_certified" { "structurally_unidentified" } else { "not_certified" };
    assert_eq!(count(&report, other), 0);
    assert!(report.receipt.is_none());
    // Both reductions occur: `parent` moves w (no normalization), `chain` keeps it.
    let decisions = &prepared.prepared().decision().decisions;
    let moves =
        |name: &str| match &decisions.iter().find(|d| &*d.scenario.name == name).unwrap().outcome {
            ScenarioOutcome::ConditionalIdentified(bound) => {
                (bound.derivation().moves().len(), bound.derivation().remaining().len())
            }
            other => panic!("{name}: {}", other.status()),
        };
    assert_eq!(moves("parent"), (1, 0));
    assert_eq!(moves("chain"), (0, 1));
    // The shared vocabulary.
    let kinds = decisions.iter().map(|d| d.outcome.identification_status()).collect::<Vec<_>>();
    let negative_kind = if negative == "not_certified" {
        TransportOutcomeKind::NotCertified
    } else {
        TransportOutcomeKind::ProvenNonTransportable
    };
    assert_eq!(
        kinds,
        [
            TransportOutcomeKind::Identified,
            TransportOutcomeKind::Identified,
            negative_kind,
            TransportOutcomeKind::Identified
        ]
    );
    assert!(matches!(prepared.prepared().decision().question, ScenarioQuestion::Conditional(_)));
    // No cross-scenario inference with shared data (2.3A).
    let refusal = prepared.aggregate_interval().unwrap_err();
    assert!(refusal.to_string().contains("scenario_aggregate_not_licensed"), "{refusal}");
}

#[test]
fn each_identified_scenario_matches_its_own_enumerated_truth() {
    // Laws enumerated from a scenario's own SCM: that scenario's point is its
    // truth, by normalization (`chain`, `unconfounded`) or by a moved w (`parent`).
    for name in ["chain", "parent", "unconfounded"] {
        for params in [0, 3] {
            let model = scm_of(name, params);
            let (catalog, data) = model.catalog_and_laws();
            for level in [0b000, 0b001, 0b100, 0b101] {
                let report = try_prepare(
                    scenarios(&ALL, None),
                    catalog.clone(),
                    data.clone(),
                    request(&[0], &[2], level),
                    BUDGET,
                )
                .unwrap()
                .estimate(&ExecutionContext::for_tests(1))
                .unwrap();
                let truth = model.truth(&[1], &[0], &[2], level).unwrap();
                let point = &row(&report, name).distribution.as_ref().unwrap().probabilities;
                for (p, t) in point.iter().zip(&truth) {
                    assert!((p - t).abs() < 1e-12, "{name} params {params} level {level}");
                }
            }
        }
    }
}

/// The status of `shifted`: `not_certified`, or `structurally_unidentified` only
/// with a verified two-model witness.
fn shifted_status(report: &ScenarioSetReport) -> &'static str {
    let status = row(report, "shifted").status;
    assert!(matches!(status, "not_certified" | "structurally_unidentified"), "{status}");
    status
}

#[test]
fn an_s_hedge_scenario_carries_its_checked_record_and_only_a_witness_proves() {
    let prepared = prepare(&ALL, "chain");
    let decision = prepared.prepared().decision();
    let shifted = decision.decisions.iter().find(|d| &*d.scenario.name == "shifted").unwrap();
    let ScenarioQuestion::Conditional(question) = &decision.question else { panic!() };
    let ctx = ExecutionContext::for_tests(1);
    let report = prepared.estimate(&ctx).unwrap();
    let detail = row(&report, "shifted").detail.clone().unwrap();
    match &shifted.outcome {
        ScenarioOutcome::ConditionalNotCertified { obligations, candidate } => {
            assert_eq!(&*obligations[0], "admg_transport.not_certified");
            let candidate = candidate.as_ref().expect("the reduced joint's s-hedge is kept");
            candidate.recheck(&shifted.scenario.diagram, question, &ctx).unwrap();
            assert!(detail.starts_with("admg_transport.not_certified"), "{detail}");
            assert!(detail.contains("inspection-only candidate (not a proof)"), "{detail}");
        }
        ScenarioOutcome::ConditionalProvenNonTransportable(proof) => {
            proof.recheck(&shifted.scenario.diagram, question, &ctx).unwrap();
            assert!(detail.starts_with("admg_transport.proven_non_transportable"), "{detail}");
        }
        other => panic!("an s-hedge scenario is never {}", other.status()),
    }
    assert_eq!(shifted.outcome.reason_code(), None, "a decided outcome, not a budget stop");
    assert!(row(&report, "shifted").distribution.is_none());
}

/// Drop every source regime (and its laws) whose intervention mask is in `skip`.
fn without(model: &Scm, skip: &[u32]) -> (EvidenceCatalog, ExactTransportData) {
    let (catalog, data) = model.catalog_and_laws();
    let regimes =
        catalog.regimes.iter().filter(|r| !skip.contains(&r.id.raw())).cloned().collect::<Vec<_>>();
    let laws = data
        .laws()
        .iter()
        .filter(|law| law.population() != "source" || !skip.contains(&law.regime().raw()))
        .cloned()
        .collect::<Vec<_>>();
    (
        EvidenceCatalog::try_new([], regimes, [], None).unwrap(),
        ExactTransportData::try_new(laws, 1_000_000).unwrap(),
    )
}

#[test]
fn each_scenario_binds_its_own_evidence_and_missing_evidence_is_scenario_local() {
    // No source experiment intervenes on x: `chain` and `parent` need one,
    // `unconfounded` reads the target law alone.
    let (catalog, data) = without(&scm_of("unconfounded", 0), &[1, 3, 5, 7]);
    let report = try_prepare(
        scenarios(&["chain", "parent", "unconfounded"], Some(&[0.25, 0.25, 0.5])),
        catalog,
        data,
        request(&[0], &[2], LEVEL),
        BUDGET,
    )
    .unwrap()
    .estimate(&ExecutionContext::for_tests(1))
    .unwrap();
    assert_eq!(row(&report, "chain").status, "missing_evidence");
    assert_eq!(row(&report, "parent").status, "missing_evidence");
    assert_eq!(row(&report, "unconfounded").status, "identified");
    let detail = row(&report, "chain").detail.clone().unwrap();
    assert!(detail.contains("source"), "the unmet leaf is named: {detail}");
    let truth = scm_of("unconfounded", 0).truth(&[1], &[0], &[2], LEVEL).unwrap();
    assert!((p_one(&report, "unconfounded") - truth[1]).abs() < 1e-12);
    // Missing-evidence mass is unaccounted, never renormalized away.
    let weighted = report.weighted.as_ref().unwrap();
    assert!((weighted.identified_mass - 0.5).abs() < 1e-12);
    assert!((weighted.unaccounted_mass - 0.5).abs() < 1e-12);
    let mass = report.masses.iter().find(|m| m.status == "missing_evidence").unwrap();
    assert_eq!((mass.count, mass.mass), (2, Some(0.5)));
}

fn decided_under(operations: usize) -> antecedent_identify::sid::scenarios::ScenarioSetDecision {
    let (catalog, _) = scm_of("chain", 0).catalog_and_laws();
    decide_conditional_transport_scenarios(
        &TransportScenarioSet::try_new(scenarios(&ALL, None)).unwrap(),
        &query(&[1], &[0], &[2]),
        &catalog,
        SearchLimits { operations, depth: 256 },
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

#[test]
fn the_shared_budget_stops_conditional_scenarios_with_one_receipt() {
    let explored = |d: &antecedent_identify::sid::scenarios::ScenarioSetDecision| {
        d.receipt.as_ref().map(|r| (r.explored.clone(), r.unevaluated.clone()))
    };
    let all = (1..10_000).find(|ops| decided_under(*ops).receipt.is_none()).unwrap();
    // The first scenario in canonical order (`chain`) alone.
    let first = (1..all)
        .find(|ops| decided_under(*ops).decisions[0].outcome.status() != "unevaluated")
        .unwrap();
    let decision = decided_under(first);
    let statuses = decision.decisions.iter().map(|d| d.outcome.status()).collect::<Vec<_>>();
    assert_eq!(statuses, ["identified", "unevaluated", "unevaluated", "unevaluated"]);
    let receipt = decision.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(
        explored(&decision).unwrap(),
        (vec!["chain".to_owned()], vec!["parent".into(), "shifted".into(), "unconfounded".into()])
    );
    // A stop inside the conditional route is the scenario's budget stop, with
    // the stopping scenario among the unevaluated ones, never a verdict.
    let inside = decided_under(first - 1);
    assert!(matches!(
        inside.decisions[0].outcome,
        ScenarioOutcome::Unevaluated { stop: SearchStop::Operations }
    ));
    assert_eq!(inside.decisions[0].outcome.reason_code(), Some("transport_budget_cancel"));
    assert_eq!(
        inside.decisions[0].outcome.identification_status(),
        TransportOutcomeKind::BudgetCancel
    );
    assert!(explored(&inside).unwrap().0.is_empty());
    // The s-hedge scenario alone runs every stage of the route, the witness
    // search included: under every budget short of the one that decides it the
    // scenario is unevaluated with the set's receipt, never a verdict (neither
    // not_certified nor structurally_unidentified).
    let shifted_under = |operations: usize| {
        let (catalog, _) = scm_of("chain", 0).catalog_and_laws();
        decide_conditional_transport_scenarios(
            &TransportScenarioSet::try_new(scenarios(&["shifted"], None)).unwrap(),
            &query(&[1], &[0], &[2]),
            &catalog,
            SearchLimits { operations, depth: 256 },
            &ExecutionContext::for_tests(1),
        )
        .unwrap()
    };
    let decides = (1..10_000).find(|ops| shifted_under(*ops).receipt.is_none()).unwrap();
    assert!(matches!(
        shifted_under(decides).decisions[0].outcome,
        ScenarioOutcome::ConditionalNotCertified { .. }
            | ScenarioOutcome::ConditionalProvenNonTransportable(_)
    ));
    for operations in 1..decides {
        let stopped = shifted_under(operations);
        assert!(
            matches!(
                stopped.decisions[0].outcome,
                ScenarioOutcome::Unevaluated { stop: SearchStop::Operations }
            ),
            "{operations} of {decides} operations: {}",
            stopped.decisions[0].outcome.status()
        );
        let receipt = stopped.receipt.as_ref().unwrap();
        assert_eq!(
            (&receipt.explored[..], &receipt.unevaluated[..]),
            (&[][..], &["shifted".to_owned()][..])
        );
    }
    // A stopped report still exports and replays its identical prefix.
    let (catalog, data) = scm_of("chain", 0).catalog_and_laws();
    let prepared = try_prepare(
        scenarios(&ALL, None),
        catalog,
        data,
        request(&[0], &[2], LEVEL),
        SearchLimits { operations: first, depth: 256 },
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let report = prepared.estimate(&ctx).unwrap();
    assert_eq!(
        row(&report, "parent").detail.as_deref(),
        Some("scenarios.unevaluated_budget: search.operations")
    );
    let bytes = prepared.export(&report).unwrap();
    let replayed = consume_transport_scenarios_artifact(
        &bytes,
        TransportScenarioConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(replayed.scenarios.len(), 4);
    assert!(replayed.receipt.is_some());
}

#[test]
fn a_zero_mass_conditioning_event_is_a_scenario_local_support_failure() {
    // The target never shows w = 1: normalizing a target-based joint at w = 1 has
    // nothing to divide by, while the moved-w scenario reads the source do(x, w).
    let mut model = scm_of("chain", 0);
    model.target_zero = vec![2];
    let (catalog, data) = model.catalog_and_laws();
    let report = try_prepare(
        scenarios(&["chain", "parent"], None),
        catalog,
        data,
        request(&[0], &[2], LEVEL),
        BUDGET,
    )
    .unwrap()
    .estimate(&ExecutionContext::for_tests(1))
    .unwrap();
    assert_eq!(row(&report, "chain").status, "support_failure");
    assert!(
        row(&report, "chain").detail.clone().unwrap().contains("admg_transport.support_failure")
    );
    assert_eq!(row(&report, "parent").status, "identified");
    assert_eq!(report.envelope.as_ref().unwrap().scenarios.len(), 1);
}

fn refused(result: Result<PreparedTransportScenarios, IoError>) -> (&'static str, String) {
    match result {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a coded refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn route_bounds_requests_and_counted_laws_refuse_with_their_frozen_codes() {
    let (catalog, data) = scm_of("chain", 0).catalog_and_laws();
    // The request binds exactly the treatments and the conditioned variables.
    let (code, message) = refused(try_prepare(
        scenarios(&ALL, None),
        catalog.clone(),
        data.clone(),
        request(&[0], &[], LEVEL),
        BUDGET,
    ));
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("admg_transport.invalid_request"), "{message}");
    // A conditioned variable outside the shared schema.
    let mut outside = query(&[1], &[0], &[2]);
    outside.conditioned_on = Arc::from([VariableId::from_raw(9)]);
    let result = StudyBuilder::conditional_transport_scenarios(
        &TransportScenarioSet::try_new(scenarios(&ALL, None)).unwrap(),
        outside,
        catalog.clone(),
        BUDGET,
        data.clone(),
        request(&[0], &[2], LEVEL),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    );
    let (code, message) = refused(result);
    assert_eq!(code, "schema_mismatch");
    assert!(message.starts_with("scenarios.coordinate_mismatch"), "{message}");
    // Counted laws are not licensed for a conditional question.
    let mut laws = data.laws().to_vec();
    let first = &laws[0];
    let cells = first.probabilities().len();
    laws[0] = antecedent_expr::ExactDiscreteLaw::try_new(
        first.population(),
        first.regime(),
        first.interventions().to_vec(),
        first.axes().to_vec(),
        vec![1.0 / f64::from(u32::try_from(cells).unwrap()); cells],
        first.snapshot_identity(),
        antecedent_expr::LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(vec![1; cells])
    .unwrap();
    let counted = ExactTransportData::try_new(laws, 1_000_000).unwrap();
    let (code, message) = refused(try_prepare(
        scenarios(&ALL, None),
        catalog,
        counted,
        request(&[0], &[2], LEVEL),
        BUDGET,
    ));
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("admg_transport.interval_withheld"), "{message}");
}

#[test]
fn a_graph_beyond_the_conditional_class_refuses_the_whole_set() {
    // Seven observed variables: inside A2's 12-variable bound, outside B1's 6.
    let mut graph = antecedent_graph::Admg::with_variables(7);
    graph
        .insert_directed(
            antecedent_graph::DenseNodeId::from_raw(0),
            antecedent_graph::DenseNodeId::from_raw(1),
        )
        .unwrap();
    let coordinates = (0..7u32)
        .map(|i| ScenarioCoordinate {
            variable: VariableId::from_raw(i),
            name: Arc::from(format!("v{i}")),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Arc<[_]>>();
    let set = TransportScenarioSet::try_new(vec![TransportScenario {
        name: Arc::from("wide"),
        diagram: antecedent_graph::SelectionDiagram::try_new(graph, Vec::<VariableId>::new())
            .unwrap(),
        weight: None,
        coordinates,
    }])
    .unwrap();
    let (catalog, data) = scm_of("chain", 0).catalog_and_laws();
    let result = StudyBuilder::conditional_transport_scenarios(
        &set,
        query(&[1], &[0], &[2]),
        catalog,
        BUDGET,
        data,
        request(&[0], &[2], LEVEL),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    );
    let (code, message) = refused(result);
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("admg_transport.bounds_exceeded"), "{message}");
}

fn consume(bytes: &[u8]) -> Result<ScenarioSetReport, IoError> {
    consume_transport_scenarios_artifact(
        bytes,
        TransportScenarioConsumeLimits::default(),
        &ExecutionContext::for_tests(1),
    )
}

fn typed(result: Result<ScenarioSetReport, IoError>) -> TransportScenarioArtifactError {
    match result {
        Err(IoError::TransportScenario(error)) => error,
        other => panic!("expected a typed refusal, got {:?}", other.map(|r| r.scenarios.len())),
    }
}

#[test]
#[allow(clippy::too_many_lines)] // One round trip, then each mutation it must refuse.
fn a_conditional_artifact_is_version_two_replays_and_fails_on_mutation() {
    let prepared = prepare(&ALL, "chain");
    let ctx = ExecutionContext::for_tests(1);
    let report = prepared.estimate(&ctx).unwrap();
    let bytes = prepared.export(&report).unwrap();
    let replayed = consume(&bytes).unwrap();
    assert_eq!(
        replayed.scenarios.iter().map(|s| s.status).collect::<Vec<_>>(),
        report.scenarios.iter().map(|s| s.status).collect::<Vec<_>>()
    );
    assert!((p_one(&replayed, "chain") - p_one(&report, "chain")).abs() == 0.0);
    let original = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    assert_eq!(original.version, TRANSPORT_SCENARIO_CONDITIONAL_ARTIFACT_VERSION);
    assert_eq!(
        original.required_features,
        [TRANSPORT_SCENARIO_ARTIFACT_FEATURE, TRANSPORT_SCENARIO_CONDITIONAL_FEATURE]
    );
    assert_eq!(original.query.conditioned_on, Some(vec![2]));
    let result =
        |name: &str| original.report.scenarios.iter().position(|s| s.name == name).unwrap();
    let (chain, shifted) = (result("chain"), result("shifted"));
    assert!(original.report.scenarios[chain].conditional_proof.is_some());
    assert!(original.report.scenarios[chain].proof.is_none());
    let proven = original.report.scenarios[shifted].non_transportability.is_some();
    assert!(proven || original.report.scenarios[shifted].candidate.is_some());
    let reseal = |edit: &dyn Fn(&mut TransportScenarioArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        wire.export().unwrap()
    };
    // The proof, candidate and status are replayed, not trusted.
    assert_eq!(
        typed(consume(&reseal(&|w| w.report.scenarios[chain].conditional_proof = None))),
        TransportScenarioArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(consume(&reseal(&|w| {
            let proof = w.report.scenarios[chain].conditional_proof.as_mut().unwrap();
            proof.moves = vec![2];
            proof.remaining = vec![];
        }))),
        TransportScenarioArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(consume(&reseal(&|w| {
            w.report.scenarios[shifted].candidate = None;
            w.report.scenarios[shifted].non_transportability = None;
        }))),
        TransportScenarioArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(consume(&reseal(&|w| {
            w.report.scenarios[shifted].status =
                if proven { "not_certified" } else { "structurally_unidentified" }.into();
        }))),
        TransportScenarioArtifactError::ReportMismatch
    );
    assert_eq!(
        typed(consume(&reseal(&|w| w.report.envelope.as_mut().unwrap().means[0].lower -= 1e-9))),
        TransportScenarioArtifactError::ReportMismatch
    );
    // The conditioned set is a premise.
    assert_eq!(
        typed(consume(&reseal(&|w| w.query.conditioned_on = Some(vec![0])))),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        typed(consume(&reseal(&|w| w.request.retain(|(v, _)| *v != 2)))),
        TransportScenarioArtifactError::PremisesMismatch
    );
    // Version and feature markers must agree with the content.
    assert_eq!(
        typed(consume(&reseal(&|w| {
            w.version = TRANSPORT_SCENARIO_ARTIFACT_VERSION;
            w.required_features = vec![TRANSPORT_SCENARIO_ARTIFACT_FEATURE.into()];
        }))),
        TransportScenarioArtifactError::UnsupportedSemantics(
            "a version 1 artifact carries a conditional question or record"
        )
    );
    assert_eq!(
        typed(consume(&reseal(&|w| w.query.conditioned_on = None))),
        TransportScenarioArtifactError::UnsupportedSemantics(
            "a version 2 artifact needs a conditioned set"
        )
    );
    assert_eq!(
        typed(consume(&reseal(&|w| w.required_features.truncate(1)))),
        TransportScenarioArtifactError::UnsupportedSemantics("required features")
    );
    assert!(matches!(
        consume(&reseal(&|w| w.version = 3)),
        Err(IoError::UnsupportedVersion { version: 3 })
    ));
}

#[test]
fn a_classical_question_still_writes_and_reads_the_version_one_artifact() {
    // The same scenarios asked the classical question P*(y | do(x)): the
    // artifact is version 1 with only the version-1 marker and no conditional
    // field encoded, so a version-1-only reader keeps reading it; such a reader
    // refuses a conditional artifact by its version (2) before decoding it.
    let (catalog, data) = scm_of("chain", 0).catalog_and_laws();
    let classical = StudyBuilder::transport_scenarios(
        &TransportScenarioSet::try_new(scenarios(&ALL, None)).unwrap(),
        ClassicalTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(1)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        },
        catalog,
        BUDGET,
        data,
        Assignment::from_pairs([(VariableId::from_raw(0), antecedent_core::Value::Int64(1))]),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let report = classical.estimate(&ctx).unwrap();
    assert!(matches!(classical.prepared().decision().question, ScenarioQuestion::Classical(_)));
    let bytes = classical.export(&report).unwrap();
    let wire = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.version, TRANSPORT_SCENARIO_ARTIFACT_VERSION);
    assert_eq!(wire.required_features, [TRANSPORT_SCENARIO_ARTIFACT_FEATURE]);
    assert!(wire.query.conditioned_on.is_none());
    assert!(wire.report.scenarios.iter().all(|s| s.conditional_proof.is_none()
        && s.candidate.is_none()
        && s.non_transportability.is_none()));
    // No version-2 key is encoded at all: the question and every result encode
    // in the version-1 shape.
    let encoded = serde_json::to_value(&wire.query).unwrap();
    assert!(encoded.get("conditioned_on").is_none(), "{encoded}");
    for result in &wire.report.scenarios {
        let encoded = serde_json::to_value(result).unwrap();
        for key in ["conditional_proof", "candidate", "non_transportability"] {
            assert!(encoded.get(key).is_none(), "{key}");
        }
    }
    consume(&bytes).unwrap();
    // The classical route decides `shifted` as a verified s-hedge, the
    // conditional route only as not certified: the routes stay distinct.
    assert_eq!(row(&report, "shifted").status, "structurally_unidentified");
    // A conditional artifact's version field is what a version-1 reader peeks.
    let conditional = prepare(&ALL, "chain");
    let report = conditional.estimate(&ctx).unwrap();
    let bytes = conditional.export(&report).unwrap();
    assert_ne!(
        TransportScenarioArtifactWire::decode(&bytes).unwrap().version,
        TRANSPORT_SCENARIO_ARTIFACT_VERSION
    );
}

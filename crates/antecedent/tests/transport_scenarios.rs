//! Finite graph/selection scenario sets for one transport question (X2, A2).
//!
//! Every scenario shares `z -> x`, `z -> y`, `x -> y`, `x <-> y` and differs only
//! in which mechanisms may change between source and target:
//! `standardize` (selection on `z`) gives `Σ_z P_s(y | do(x), z) P*(z)`;
//! `direct` (no selection) gives `P_s(y | do(x))`; `outcome_shift` (selection on
//! `y`) is not transportable. Expected values are computed here from the laws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{PreparedTransportScenarios, StudyBuilder, consume_transport_scenarios_artifact};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, SearchStop, Value, VariableId,
};
use antecedent_estimate::transport_scenarios::{
    STRUCTURAL_ENVELOPE_INTERPRETATION, ScenarioSetReport,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery, SidLimits,
    sid::scenarios::{TransportScenario, TransportScenarioSet},
};
use antecedent_io::IoError;
use antecedent_io::transport_scenario_artifact::{
    TransportScenarioArtifactError, TransportScenarioArtifactWire, TransportScenarioConsumeLimits,
};

const Z: u32 = 0;
const X: u32 = 1;
const Y: u32 = 2;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn scenario(name: &str, selections: &[u32], weight: Option<f64>) -> TransportScenario {
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(Z, X), (Z, Y), (X, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(X), DenseNodeId::from_raw(Y)).unwrap();
    let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
    TransportScenario {
        name: Arc::from(name),
        diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap(),
        weight,
    }
}

fn three(weights: Option<[f64; 3]>) -> Vec<TransportScenario> {
    let w = |k: usize| weights.map(|w| w[k]);
    vec![
        scenario("standardize", &[Z], w(0)),
        scenario("direct", &[], w(1)),
        scenario("outcome_shift", &[Y], w(2)),
    ]
}

fn query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn regime(id: u32, population: &str, interventions: &[u32], measured: &[u32]) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        interventions.iter().copied().map(v).collect::<Vec<_>>(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn catalog(with_target: bool) -> EvidenceCatalog {
    let mut regimes = vec![regime(1, "source", &[X], &[Z, Y])];
    if with_target {
        regimes.push(regime(0, "target", &[], &[Z, X, Y]));
    }
    EvidenceCatalog::try_new([], regimes, [], None).unwrap()
}

fn axis(variable: u32) -> DiscreteAxis {
    DiscreteAxis { variable: v(variable), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

/// Source `do(x = 1)` over `(z, y)`: `P_s(z = 1) = 0.6`, `P_s(y = 1 | z) = 0.2, 0.8`.
const SOURCE: [f64; 4] = [0.32, 0.08, 0.12, 0.48];
/// Target observational `(z, x, y)` with `P*(z = 1) = 0.25`.
const TARGET: [f64; 8] = [0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1];
const STANDARDIZED: f64 = 0.75 * 0.2 + 0.25 * 0.8;
const DIRECT: f64 = 0.08 + 0.48;

fn laws(source: [f64; 4], with_target: bool) -> ExactTransportData {
    let mut laws = vec![
        ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(1),
            [InterventionAssignment::concrete(v(X), Value::f64(1.0))],
            [axis(Z), axis(Y)],
            source,
            "trial",
            LawTolerance::default(),
        )
        .unwrap(),
    ];
    if with_target {
        laws.push(
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                [axis(Z), axis(X), axis(Y)],
                TARGET,
                "target",
                LawTolerance::default(),
            )
            .unwrap(),
        );
    }
    ExactTransportData::try_new(laws, 1000).unwrap()
}

const BUDGET: SearchLimits = SearchLimits { operations: 64, depth: 1 };

fn prepare(
    scenarios: Vec<TransportScenario>,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    budget: SearchLimits,
) -> PreparedTransportScenarios {
    let set = TransportScenarioSet::try_new(scenarios).unwrap();
    StudyBuilder::transport_scenarios(
        &set,
        query(),
        catalog,
        SidLimits::default(),
        budget,
        data,
        Assignment::from_pairs([(v(X), Value::f64(1.0))]),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

fn status<'a>(report: &'a ScenarioSetReport, name: &str) -> (&'a str, Option<f64>) {
    let s = report.scenarios.iter().find(|s| &*s.name == name).unwrap();
    (s.status, s.distribution.as_ref().map(|d| d.mean(v(Y)).unwrap()))
}

fn close(a: Option<f64>, b: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() < 1e-12)
}

#[test]
fn identified_scenarios_disagree_and_the_envelope_spans_them() {
    let report = prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    let (s, standardized) = status(&report, "standardize");
    let (d, direct) = status(&report, "direct");
    assert_eq!((s, d), ("identified", "identified"));
    assert!(close(standardized, STANDARDIZED) && close(direct, DIRECT));
    let envelope = report.envelope.as_ref().unwrap();
    assert_eq!(envelope.interpretation, STRUCTURAL_ENVELOPE_INTERPRETATION);
    let mean = &envelope.means[0];
    assert!((mean.lower - STANDARDIZED).abs() < 1e-12 && (mean.upper - DIRECT).abs() < 1e-12);
    assert_eq!((&*mean.lower_scenario, &*mean.upper_scenario), ("standardize", "direct"));
}

#[test]
fn an_unidentified_scenario_is_retained_with_positive_mass() {
    let report = prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    let (shift, point) = status(&report, "outcome_shift");
    assert_eq!(shift, "structurally_unidentified");
    assert!(point.is_none());
    let count = |name: &str| report.masses.iter().find(|m| m.status == name).unwrap().count;
    assert_eq!((count("identified"), count("structurally_unidentified")), (2, 1));
    assert_eq!(report.scenarios.len(), 3);
    // The envelope ranges over the identified scenarios only and names them.
    assert_eq!(report.envelope.as_ref().unwrap().scenarios.len(), 2);
}

#[test]
fn declared_weights_are_never_renormalized_over_survivors() {
    let report = prepare(three(Some([0.3, 0.2, 0.4])), catalog(true), laws(SOURCE, true), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    assert!((report.residual_mass.unwrap() - 0.1).abs() < 1e-12);
    let mass = |name: &str| report.masses.iter().find(|m| m.status == name).unwrap().mass.unwrap();
    assert!((mass("structurally_unidentified") - 0.4).abs() < 1e-12);
    let weighted = report.weighted.as_ref().unwrap();
    assert!((weighted.identified_mass - 0.5).abs() < 1e-12);
    assert!((weighted.unaccounted_mass - 0.5).abs() < 1e-12);
    let sum = 0.3 * STANDARDIZED + 0.2 * DIRECT;
    assert!((weighted.identified_weighted_sums[0].1 - sum).abs() < 1e-12);
    // The unaccounted half may sit anywhere in {0, 1}; no survivor-renormalized mean.
    let (_, lo, hi) = weighted.ranges.as_ref().unwrap()[0];
    assert!((lo - sum).abs() < 1e-12 && (hi - (sum + 0.5)).abs() < 1e-12);
    // Weights must be declared for all scenarios or none, and sum to at most one.
    let mut partial = three(Some([0.3, 0.2, 0.4]));
    partial[0].weight = None;
    assert!(TransportScenarioSet::try_new(partial).is_err());
    assert!(TransportScenarioSet::try_new(three(Some([0.6, 0.3, 0.4]))).is_err());
}

#[test]
fn each_scenario_binds_its_own_evidence_obligations() {
    // Without the target law, standardization cannot bind, the direct formula
    // still can, and nothing certified for one scenario satisfies the other.
    let report = prepare(three(None), catalog(false), laws(SOURCE, false), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    assert_eq!(status(&report, "standardize").0, "missing_evidence");
    let (direct, point) = status(&report, "direct");
    assert_eq!(direct, "identified");
    assert!(close(point, DIRECT));
    // A catalog regime whose law the provider lacks is an unsupported provider.
    let report = prepare(three(None), catalog(true), laws(SOURCE, false), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    assert_eq!(status(&report, "standardize").0, "unsupported_provider");
    assert_eq!(status(&report, "direct").0, "identified");
}

#[test]
fn a_scenario_budget_leaves_the_rest_unevaluated_with_a_receipt() {
    let report = prepare(
        three(None),
        catalog(true),
        laws(SOURCE, true),
        SearchLimits { operations: 2, depth: 1 },
    )
    .estimate(&ExecutionContext::for_tests(1))
    .unwrap();
    // Canonical (name) order: direct, outcome_shift, standardize.
    assert_eq!(status(&report, "standardize").0, "unevaluated");
    let receipt = report.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.unevaluated, ["standardize"]);
    assert_eq!(receipt.explored, ["direct", "outcome_shift"]);
}

#[test]
fn scenario_order_does_not_change_the_report_or_identity() {
    let forward = prepare(three(Some([0.3, 0.2, 0.4])), catalog(true), laws(SOURCE, true), BUDGET);
    let mut reversed = three(Some([0.3, 0.2, 0.4]));
    reversed.reverse();
    let backward = prepare(reversed, catalog(true), laws(SOURCE, true), BUDGET);
    let ctx = ExecutionContext::for_tests(1);
    let (a, b) = (forward.estimate(&ctx).unwrap(), backward.estimate(&ctx).unwrap());
    assert_eq!(forward.export(&a).unwrap(), backward.export(&b).unwrap());
}

#[test]
fn estimate_after_refresh_keeps_decisions_and_moves_points() {
    let prepared = prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET);
    let ctx = ExecutionContext::for_tests(1);
    let refreshed = prepared.refresh(laws([0.1, 0.3, 0.3, 0.3], true), &ctx).unwrap();
    let (before, after) = (prepared.estimate(&ctx).unwrap(), refreshed.estimate(&ctx).unwrap());
    for (b, a) in before.scenarios.iter().zip(&after.scenarios) {
        assert_eq!(b.status, a.status);
    }
    assert!(close(status(&after, "direct").1, 0.6));
    assert!(!close(status(&after, "direct").1, DIRECT));
}

#[test]
fn inference_across_scenarios_is_refused() {
    let prepared = prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET);
    let refusal = prepared.aggregate_interval().unwrap_err();
    assert!(refusal.to_string().contains("scenario_aggregate_not_licensed"), "{refusal}");
}

#[test]
fn artifact_keeps_every_scenario_and_fails_on_mutation() {
    let prepared = prepare(three(Some([0.3, 0.2, 0.4])), catalog(true), laws(SOURCE, true), BUDGET);
    let ctx = ExecutionContext::for_tests(1);
    let report = prepared.estimate(&ctx).unwrap();
    let bytes = prepared.export(&report).unwrap();
    drop(prepared);
    let consumed = consume_transport_scenarios_artifact(
        &bytes,
        TransportScenarioConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(consumed.scenarios.len(), 3);
    assert_eq!(status(&consumed, "outcome_shift").0, "structurally_unidentified");
    let original = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    let refused = |edit: &dyn Fn(&mut TransportScenarioArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        match consume_transport_scenarios_artifact(
            &wire.export().unwrap(),
            TransportScenarioConsumeLimits::default(),
            &ctx,
        ) {
            Err(IoError::TransportScenario(error)) => error,
            other => panic!("expected a typed refusal, got {:?}", other.map(|r| r.scenarios.len())),
        }
    };
    assert_eq!(
        refused(&|w| w.scenarios[0].weight = Some(0.35)),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.scenarios[1].selections.clear()),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.report.scenarios[1].status = "identified".into()),
        TransportScenarioArtifactError::ReportMismatch
    );
    assert_eq!(
        refused(&|w| w.report.scenarios.retain(|s| s.status == "identified")),
        TransportScenarioArtifactError::ReportMismatch
    );
    assert_eq!(
        refused(&|w| w.report.envelope.as_mut().unwrap().means[0].upper += 1e-9),
        TransportScenarioArtifactError::ReportMismatch
    );
    let small = TransportScenarioConsumeLimits {
        scenario_budget: SearchLimits { operations: 2, depth: 1 },
        ..TransportScenarioConsumeLimits::default()
    };
    assert!(matches!(
        consume_transport_scenarios_artifact(&bytes, small, &ctx),
        Err(IoError::TransportScenario(TransportScenarioArtifactError::LimitsExceeded(_)))
    ));
}

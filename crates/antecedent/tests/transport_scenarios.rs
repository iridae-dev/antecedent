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
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, MemoryBudget, ProgressSink, RegimeId, RegimeKind, SearchLimits, SearchStop,
    Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::transport_scenarios::{
    STRUCTURAL_ENVELOPE_INTERPRETATION, ScenarioSetReport,
};
use antecedent_estimate::{RegimeSample, StatisticalTransportInput};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery,
    sid::scenarios::{
        ScenarioCoordinate, ScenarioDecision, ScenarioDecisionLimits, ScenarioOutcome,
        ScenarioSetDecision, TransportScenario, TransportScenarioSet,
    },
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

/// The shared schema: `z`, `x`, `y`, each binary unless `y_domain` says otherwise.
fn schema(y_domain: VariableDomain) -> Arc<[ScenarioCoordinate]> {
    [(Z, "z", VariableDomain::Binary), (X, "x", VariableDomain::Binary), (Y, "y", y_domain)]
        .into_iter()
        .map(|(id, name, domain)| ScenarioCoordinate {
            variable: v(id),
            name: Arc::from(name),
            domain,
            unit: None,
        })
        .collect()
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
        coordinates: schema(VariableDomain::Binary),
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

const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn request() -> Assignment {
    Assignment::from_pairs([(v(X), Value::f64(1.0))])
}

fn try_prepare(
    scenarios: Vec<TransportScenario>,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<PreparedTransportScenarios, IoError> {
    let set = TransportScenarioSet::try_new(scenarios).unwrap();
    StudyBuilder::transport_scenarios(
        &set,
        query(),
        catalog,
        budget,
        data,
        request(),
        ExactEvaluationLimits::default(),
        ctx,
    )
}

fn prepare(
    scenarios: Vec<TransportScenario>,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    budget: SearchLimits,
) -> PreparedTransportScenarios {
    try_prepare(scenarios, catalog, data, budget, &ExecutionContext::for_tests(1)).unwrap()
}

fn refused_code(result: Result<PreparedTransportScenarios, IoError>) -> (&'static str, String) {
    match result {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a coded refusal, got {:?}", other.map(|_| ())),
    }
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

/// Decide `scenarios` under `operations` shared operations.
fn decided_under(
    scenarios: Vec<TransportScenario>,
    operations: usize,
) -> antecedent_identify::sid::scenarios::ScenarioSetDecision {
    antecedent_identify::sid::scenarios::decide_transport_scenarios(
        &TransportScenarioSet::try_new(scenarios).unwrap(),
        &query(),
        &catalog(true),
        SearchLimits { operations, depth: 256 },
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

/// The fewest shared operations that decide every scenario of `scenarios`.
fn operations_to_decide(scenarios: impl Fn() -> Vec<TransportScenario>) -> usize {
    (1..10_000)
        .find(|ops| decided_under(scenarios(), *ops).receipt.is_none())
        .expect("a small graph decides within 10k operations")
}

#[test]
fn a_scenario_budget_leaves_the_rest_unevaluated_with_a_receipt() {
    // Enough operations to enter and decide `direct` (first in canonical
    // order), but not to enter `outcome_shift`: it and `standardize` are
    // unevaluated.
    let direct = operations_to_decide(|| vec![scenario("direct", &[], None)]);
    let report = prepare(
        three(None),
        catalog(true),
        laws(SOURCE, true),
        SearchLimits { operations: direct, depth: 256 },
    )
    .estimate(&ExecutionContext::for_tests(1))
    .unwrap();
    // Canonical (name) order: direct, outcome_shift, standardize.
    assert_eq!(status(&report, "direct").0, "identified");
    assert_eq!(status(&report, "standardize").0, "unevaluated");
    let receipt = report.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_consumed, Some(direct));
    assert_eq!(receipt.explored, ["direct"]);
    assert_eq!(receipt.unevaluated, ["outcome_shift", "standardize"]);
    let detail = |name: &str| {
        report.scenarios.iter().find(|s| &*s.name == name).unwrap().detail.clone().unwrap()
    };
    assert_eq!(detail("outcome_shift"), "scenarios.unevaluated_budget: search.operations");
}

#[test]
fn a_budget_that_suffices_per_scenario_does_not_suffice_in_total() {
    // Each scenario's search and verification replay charge the one shared
    // budget: every scenario alone decides within `each`, the set does not.
    let each = ["direct", "outcome_shift", "standardize"]
        .into_iter()
        .map(|name| {
            operations_to_decide(|| three(None).into_iter().filter(|s| &*s.name == name).collect())
        })
        .max()
        .unwrap();
    let total = operations_to_decide(|| three(None));
    assert!(total > each, "total {total} per scenario {each}");
    for name in ["direct", "outcome_shift", "standardize"] {
        let alone =
            decided_under(three(None).into_iter().filter(|s| &*s.name == name).collect(), each);
        assert!(alone.receipt.is_none(), "{name} alone");
    }
    let together = decided_under(three(None), each);
    let receipt = together.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_consumed, Some(each));
    assert!(together.decisions.iter().any(|d| d.outcome.status() == "unevaluated"));
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
    // Identity binds the variable names, the schema, the budgets and the request.
    let swap_names = |w: &mut TransportScenarioArtifactWire| {
        let first = w.coordinates[0].name.clone();
        w.coordinates[0].name = w.coordinates[2].name.clone();
        w.coordinates[2].name = first;
    };
    assert_eq!(refused(&swap_names), TransportScenarioArtifactError::PremisesMismatch);
    assert_eq!(
        refused(&|w| {
            w.coordinates[2].domain = "categorical".into();
            w.coordinates[2].cardinality = Some(3);
        }),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.coordinates[1].unit = Some("mg".into())),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.scenario_operations -= 1),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.scenario_depth -= 1),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.operation_limit -= 1),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.request[0].1 =
            antecedent_io::query_wire::ValueWire::from_value(&Value::f64(0.0))),
        TransportScenarioArtifactError::PremisesMismatch
    );
    assert_eq!(
        refused(&|w| w.provider = "transport.empirical_table_plugin".into()),
        TransportScenarioArtifactError::ProviderMismatch("an empirical provider needs a sample")
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

#[test]
fn scenarios_disagreeing_on_names_domains_or_units_refuse_with_schema_mismatch() {
    let with = |edit: &dyn Fn(&mut ScenarioCoordinate)| {
        let mut changed = scenario("direct", &[], None);
        let mut coordinates = changed.coordinates.to_vec();
        edit(&mut coordinates[2]);
        changed.coordinates = coordinates.into();
        TransportScenarioSet::try_new(vec![scenario("standardize", &[Z], None), changed])
            .unwrap_err()
    };
    for refusal in [
        with(&|c| c.name = Arc::from("outcome")),
        with(&|c| c.domain = VariableDomain::Categorical { cardinality: 3 }),
        with(&|c| c.domain = VariableDomain::Continuous),
        with(&|c| c.unit = Some(Arc::from("mmHg"))),
    ] {
        assert_eq!(
            (refusal.code, refusal.detail),
            ("schema_mismatch", "scenarios.coordinate_mismatch")
        );
    }
}

#[test]
fn laws_requests_and_catalogs_are_validated_against_the_shared_schema() {
    let ctx = ExecutionContext::for_tests(1);
    // A law whose outcome axis takes a value outside the declared binary domain.
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [InterventionAssignment::concrete(v(X), Value::f64(1.0))],
        [
            axis(Z),
            DiscreteAxis { variable: v(Y), values: Arc::from([Value::f64(0.0), Value::f64(2.0)]) },
        ],
        SOURCE,
        "trial",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new(vec![law], 1000).unwrap();
    let (code, message) =
        refused_code(try_prepare(three(None), catalog(false), data, BUDGET, &ctx));
    assert_eq!(code, "schema_mismatch");
    assert!(message.starts_with("scenarios.coordinate_mismatch"), "{message}");
    // A request outside the declared domain of the treatment.
    let set = TransportScenarioSet::try_new(three(None)).unwrap();
    let outside = StudyBuilder::transport_scenarios(
        &set,
        query(),
        catalog(true),
        BUDGET,
        laws(SOURCE, true),
        Assignment::from_pairs([(v(X), Value::f64(2.0))]),
        ExactEvaluationLimits::default(),
        &ctx,
    );
    assert_eq!(refused_code(outside).0, "schema_mismatch");
    // A question naming a variable outside the schema.
    let stray = StudyBuilder::transport_scenarios(
        &set,
        ClassicalTransportQuery { outcomes: Arc::from([v(7)]), ..query() },
        catalog(true),
        BUDGET,
        laws(SOURCE, true),
        request(),
        ExactEvaluationLimits::default(),
        &ctx,
    );
    assert_eq!(refused_code(stray).0, "schema_mismatch");
    // A catalog environment declaring a different domain for a shared variable.
    let declare = |population: &str, y: VariableDomain, others: VariableDomain| {
        let coordinates = [(Z, others.clone()), (X, others), (Y, y)]
            .into_iter()
            .map(|(id, domain)| VariableCoordinate { variable: v(id), domain, unit: None })
            .collect::<Vec<_>>();
        Environment::try_new(population, coordinates, Vec::<VariableId>::new()).unwrap()
    };
    let env =
        declare("target", VariableDomain::Categorical { cardinality: 3 }, VariableDomain::Binary);
    let source = declare("source", VariableDomain::Unspecified, VariableDomain::Unspecified);
    let regimes = catalog(true).regimes.to_vec();
    let mismatched =
        EvidenceCatalog::try_new(vec![source, env], regimes.clone(), [], None).unwrap();
    let (code, _) =
        refused_code(try_prepare(three(None), mismatched, laws(SOURCE, true), BUDGET, &ctx));
    assert_eq!(code, "schema_mismatch");
    // Selection targets belong to each scenario, never to the shared catalog.
    let selecting =
        Environment::try_new("source", Vec::<VariableCoordinate>::new(), vec![v(Z)]).unwrap();
    let pinned =
        EvidenceCatalog::try_new(vec![selecting, binary_environment("target")], regimes, [], None)
            .unwrap();
    let (code, message) =
        refused_code(try_prepare(three(None), pinned, laws(SOURCE, true), BUDGET, &ctx));
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("scenarios.catalog_selections"), "{message}");
}

#[test]
fn weighted_ranges_use_the_declared_outcome_domain_not_the_observed_atoms() {
    // `y` is declared with three levels; every law only realizes {0, 1}, so the
    // level 2 has probability zero and appears in no atom. Unaccounted mass may
    // still sit there.
    let declared = |mut s: TransportScenario| {
        s.coordinates = schema(VariableDomain::Categorical { cardinality: 3 });
        s
    };
    let scenarios = three(Some([0.3, 0.2, 0.4])).into_iter().map(declared).collect();
    let report = prepare(scenarios, catalog(true), laws(SOURCE, true), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    let weighted = report.weighted.as_ref().unwrap();
    let sum = 0.3 * STANDARDIZED + 0.2 * DIRECT;
    let (_, lo, hi) = weighted.ranges.as_ref().unwrap()[0];
    assert!((lo - sum).abs() < 1e-12);
    assert!((hi - (sum + 0.5 * 2.0)).abs() < 1e-12, "{hi}");
}

#[test]
fn a_support_failure_is_scenario_local() {
    // The source trial never shows z = 1 while the target does, so
    // standardization divides by P_s(z = 1) = 0; the direct formula is fine.
    let report = prepare(three(None), catalog(true), laws([0.4, 0.6, 0.0, 0.0], true), BUDGET)
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    assert_eq!(status(&report, "standardize").0, "support_failure");
    let (direct, point) = status(&report, "direct");
    assert_eq!(direct, "identified");
    assert!(close(point, 0.6));
    let count = |name: &str| report.masses.iter().find(|m| m.status == name).unwrap().count;
    assert_eq!(count("support_failure"), 1);
}

#[test]
fn a_not_certified_scenario_is_reported_and_carries_its_obligations() {
    // The classical route is complete on these graphs, so a not-certified
    // outcome is supplied directly to exercise the report path.
    let set = TransportScenarioSet::try_new(three(Some([0.3, 0.2, 0.4]))).unwrap();
    let outcome = |s: &TransportScenario| match &*s.name {
        "outcome_shift" => ScenarioOutcome::NotCertified {
            obligations: Arc::from([Arc::from("search certified nothing")]),
        },
        _ => ScenarioOutcome::Unevaluated { stop: SearchStop::Operations },
    };
    let decision = ScenarioSetDecision {
        decisions: set
            .scenarios()
            .iter()
            .map(|s| ScenarioDecision { scenario: s.clone(), outcome: outcome(s) })
            .collect(),
        set,
        receipt: None,
        limits: ScenarioDecisionLimits { budget: BUDGET, memory_limit_bytes: None },
    };
    let ctx = ExecutionContext::for_tests(1);
    let prepared = antecedent_estimate::transport_scenarios::prepare_transport_scenarios(
        decision,
        laws(SOURCE, true),
        request(),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    let report = prepared.evaluate(&ctx).unwrap();
    let shift = report.scenarios.iter().find(|s| &*s.name == "outcome_shift").unwrap();
    assert_eq!(shift.status, "not_certified");
    assert_eq!(shift.detail.as_deref(), Some("search certified nothing"));
    let mass = report.masses.iter().find(|m| m.status == "not_certified").unwrap();
    assert_eq!((mass.count, mass.mass), (1, Some(0.4)));
    let direct = report.scenarios.iter().find(|s| &*s.name == "direct").unwrap();
    assert_eq!(direct.detail.as_deref(), Some("scenarios.unevaluated_budget: search.operations"));
    assert!(report.envelope.is_none());
    assert!(report.weighted.as_ref().unwrap().ranges.is_none());
}

#[test]
fn the_shared_budget_observes_depth_and_memory() {
    // A zero depth limit stops the set before any scenario is entered.
    let ctx = ExecutionContext::for_tests(1);
    let report = try_prepare(
        three(None),
        catalog(true),
        laws(SOURCE, true),
        SearchLimits { operations: 64, depth: 0 },
        &ctx,
    )
    .unwrap()
    .estimate(&ctx)
    .unwrap();
    let receipt = report.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Depth);
    assert_eq!(receipt.operations_consumed, None);
    assert_eq!(receipt.unevaluated.len(), 3);
    assert!(report.scenarios.iter().all(|s| s.status == "unevaluated"));
    // Each entered scenario is charged the decision's live bytes (three nodes:
    // 9 * 64 + 512 = 1088 bytes per retained scenario): the first fits a
    // 1500-byte hard limit, the second (2176 bytes live) does not.
    let mut tight = ExecutionContext::for_tests(1);
    tight.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(1500) };
    let report = try_prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET, &tight)
        .unwrap()
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    let receipt = report.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(1500));
    assert_eq!(receipt.explored, ["direct"]);
    assert_eq!(receipt.unevaluated, ["direct", "outcome_shift", "standardize"]);
    assert!(report.scenarios.iter().all(|s| s.status == "unevaluated"));
}

/// Cancels the token after the first scenario is decided.
struct CancelAfterFirst(antecedent_core::CancellationToken);

impl ProgressSink for CancelAfterFirst {
    fn report(&self, _fraction: f64, stage: &str) {
        if stage == "transport scenarios" {
            self.0.cancel();
        }
    }
}

#[test]
fn cancellation_mid_set_leaves_the_remaining_scenarios_unevaluated() {
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.progress = Some(Arc::new(CancelAfterFirst(ctx.cancellation.clone())));
    let set = TransportScenarioSet::try_new(three(None)).unwrap();
    let decision = antecedent_identify::sid::scenarios::decide_transport_scenarios(
        &set,
        &query(),
        &catalog(true),
        BUDGET,
        &ctx,
    )
    .unwrap();
    let statuses = decision.decisions.iter().map(|d| d.outcome.status()).collect::<Vec<_>>();
    assert_eq!(statuses, ["identified", "unevaluated", "unevaluated"]);
    let receipt = decision.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Cancelled);
    assert_eq!(receipt.explored, ["direct"]);
    assert_eq!(receipt.unevaluated, ["outcome_shift", "standardize"]);
}

fn binary_environment(population: &str) -> Environment {
    let coordinates = [Z, X, Y]
        .into_iter()
        .map(|id| VariableCoordinate {
            variable: v(id),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    Environment::try_new(population, coordinates, Vec::<VariableId>::new()).unwrap()
}

fn sample(
    population: &str,
    regime: u32,
    snapshot: &str,
    interventions: &[(u32, f64)],
    columns: &[(u32, &[f64])],
) -> RegimeSample {
    RegimeSample {
        population: Arc::from(population),
        regime: RegimeId::from_raw(regime),
        snapshot_identity: Arc::from(snapshot),
        interventions: interventions
            .iter()
            .map(|(id, x)| InterventionAssignment::concrete(v(*id), Value::f64(*x)))
            .collect(),
        columns: columns
            .iter()
            .map(|(id, xs)| (v(*id), xs.iter().copied().map(Some).collect()))
            .collect(),
    }
}

#[test]
fn empirical_plug_in_points_compile_per_scenario_through_the_same_plan() {
    let regimes = catalog(true).regimes.to_vec();
    let catalog = EvidenceCatalog::try_new(
        vec![binary_environment("source"), binary_environment("target")],
        regimes,
        [],
        None,
    )
    .unwrap();
    // Source trial under do(x = 1): P_s(y = 1) = 3/5, P_s(y = 1 | z) = 1/2, 2/3.
    let trial = sample(
        "source",
        1,
        "trial",
        &[(X, 1.0)],
        &[(Z, &[0.0, 0.0, 1.0, 1.0, 1.0]), (Y, &[0.0, 1.0, 1.0, 1.0, 0.0])],
    );
    // Target observational rows with P*(z = 1) = 1/4.
    let target = sample(
        "target",
        0,
        "target",
        &[],
        &[(Z, &[0.0, 0.0, 0.0, 1.0]), (X, &[0.0, 1.0, 0.0, 1.0]), (Y, &[0.0, 1.0, 1.0, 0.0])],
    );
    let input = StatisticalTransportInput { supplied: Vec::new(), samples: vec![target, trial] };
    let set = TransportScenarioSet::try_new(three(None)).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let prepared = StudyBuilder::transport_scenarios_empirical(
        &set,
        query(),
        catalog,
        BUDGET,
        input,
        1000,
        request(),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(prepared.prepared().provider(), antecedent_estimate::EMPIRICAL_TABLE_PLUGIN);
    let report = prepared.estimate(&ctx).unwrap();
    assert!(close(status(&report, "direct").1, 0.6));
    assert!(close(status(&report, "standardize").1, 0.75 * 0.5 + 0.25 * (2.0 / 3.0)));
    assert_eq!(status(&report, "outcome_shift").0, "structurally_unidentified");
    // Points only: the envelope is a structural range, and the artifact replays
    // from the fitted tables with their sample summaries.
    let bytes = prepared.export(&report).unwrap();
    let consumed = consume_transport_scenarios_artifact(
        &bytes,
        TransportScenarioConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert!(close(status(&consumed, "standardize").1, 0.75 * 0.5 + 0.25 * (2.0 / 3.0)));
    let mut wire = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    wire.samples[0].n += 1;
    assert!(matches!(
        consume_transport_scenarios_artifact(
            &wire.export().unwrap(),
            TransportScenarioConsumeLimits::default(),
            &ctx
        ),
        Err(IoError::TransportScenario(TransportScenarioArtifactError::ProviderMismatch(_)))
    ));
}

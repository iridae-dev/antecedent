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
#[allow(clippy::too_many_lines)] // One artifact round trip, then each mutation it must refuse.
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
    // The provider label is bound by the data identity; a consistent rewrite of
    // that digest still has to agree with the laws and samples it names.
    assert_eq!(
        refused(&|w| w.provider = "transport.empirical_table_plugin".into()),
        TransportScenarioArtifactError::DataIdentityMismatch
    );
    assert_eq!(
        refused(&|w| {
            w.provider = "transport.empirical_table_plugin".into();
            w.data_digest = w.expected_data_digest().unwrap();
        }),
        TransportScenarioArtifactError::ProviderMismatch("an empirical provider needs a sample")
    );
    assert_eq!(
        refused(&|w| w.scenario_memory_bytes = Some(1 << 40)),
        TransportScenarioArtifactError::PremisesMismatch
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
        question: query().into(),
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
    // A scenario's engine holds 1088 bytes of live state per charged step (three
    // nodes: 9 * 64 + 512): the 1500-byte hard limit fits one step, so the first
    // scenario entered stops mid-search on its second charge. It is unevaluated,
    // not decided, so it is not `explored`; the entry charge on a later scenario
    // and the cumulative memory of the set are pinned in their own tests.
    let mut tight = ExecutionContext::for_tests(1);
    tight.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(1500) };
    let report = try_prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET, &tight)
        .unwrap()
        .estimate(&ExecutionContext::for_tests(1))
        .unwrap();
    let receipt = report.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(1500));
    assert!(receipt.explored.is_empty());
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

/// The empirical fixture: a catalog with binary environments, and one trial and
/// one target sample.
fn empirical_fixture() -> (EvidenceCatalog, StatisticalTransportInput) {
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
    (catalog, input)
}

#[test]
fn empirical_plug_in_points_compile_per_scenario_through_the_same_plan() {
    let (catalog, input) = empirical_fixture();
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
    let consume = |wire: &TransportScenarioArtifactWire| match consume_transport_scenarios_artifact(
        &wire.export().unwrap(),
        TransportScenarioConsumeLimits::default(),
        &ctx,
    ) {
        Err(IoError::TransportScenario(error)) => error,
        other => panic!("expected a typed refusal, got {:?}", other.map(|r| r.scenarios.len())),
    };
    // The sample summary is bound by the data identity ...
    assert_eq!(consume(&wire), TransportScenarioArtifactError::DataIdentityMismatch);
    // ... and a consistent rewrite of the digest still has to match the fitted table.
    wire.data_digest = wire.expected_data_digest().unwrap();
    assert!(matches!(consume(&wire), TransportScenarioArtifactError::ProviderMismatch(_)));
    // Bootstrap is fixed at zero: the plug-in tables are points, and they are
    // fitted once for the whole set (one table per sample), not per scenario.
    assert_eq!(prepared.prepared().data().laws().len(), 2);
    let plan = prepared.prepared().plan_summary();
    assert_eq!(plan.iter().filter(|(_, kind)| kind == "compiled").count(), 2);
    // A snapshot label the report never shows cannot be edited, consistently in
    // the fitted law and its sample summary, without breaking the data identity.
    let mut relabelled = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    let old = relabelled.samples[0].snapshot.clone();
    relabelled.samples[0].snapshot = "forged".into();
    for law in relabelled.laws.iter_mut().filter(|law| law.snapshot == old) {
        law.snapshot = "forged".into();
    }
    assert_eq!(consume(&relabelled), TransportScenarioArtifactError::DataIdentityMismatch);
    let mut altered = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    altered.samples[0].content_digest = "0".repeat(altered.samples[0].content_digest.len());
    assert_eq!(consume(&altered), TransportScenarioArtifactError::DataIdentityMismatch);
}

// ---------------------------------------------------------------------------
// Data identity (D1)
// ---------------------------------------------------------------------------

/// `catalog(true)` with snapshot bindings for its two regimes.
fn bound_catalog() -> EvidenceCatalog {
    let binding = |regime: u32, snapshot: &str| antecedent_core::RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(regime),
        snapshot_identity: Arc::from(snapshot),
        schema_names: Arc::from([Arc::from("z"), Arc::from("x"), Arc::from("y")]),
        sampling: antecedent_core::SamplingDesign::Independent,
        weights: None,
        dependence: antecedent_core::DependenceGroup::IndependentStudies,
    };
    EvidenceCatalog::try_new(
        [],
        catalog(true).regimes.to_vec(),
        [binding(1, "trial"), binding(0, "target")],
        None,
    )
    .unwrap()
}

fn consume_error(wire: &TransportScenarioArtifactWire) -> TransportScenarioArtifactError {
    match consume_transport_scenarios_artifact(
        &wire.export().unwrap(),
        TransportScenarioConsumeLimits::default(),
        &ExecutionContext::for_tests(1),
    ) {
        Err(IoError::TransportScenario(error)) => error,
        other => panic!("expected a typed refusal, got {:?}", other.map(|r| r.scenarios.len())),
    }
}

#[test]
fn the_data_identity_binds_the_catalog_laws_provider_and_samples() {
    let prepared = prepare(three(None), bound_catalog(), laws(SOURCE, true), BUDGET);
    let ctx = ExecutionContext::for_tests(1);
    let bytes = prepared.export(&prepared.estimate(&ctx).unwrap()).unwrap();
    let original = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    assert_eq!(original.data_digest, original.expected_data_digest().unwrap());
    // A catalog binding's snapshot is data identity the report never shows.
    let mut catalog = original.clone();
    let mut edited = catalog.catalog.to_catalog().unwrap();
    let mut bindings = edited.bindings.to_vec();
    bindings[0].snapshot_identity = Arc::from("other");
    edited.bindings = bindings.into();
    catalog.catalog =
        antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&edited);
    assert_eq!(consume_error(&catalog), TransportScenarioArtifactError::DataIdentityMismatch);
    // A law's snapshot label.
    let mut law = original.clone();
    law.laws[0].snapshot = "other".into();
    assert_eq!(consume_error(&law), TransportScenarioArtifactError::DataIdentityMismatch);
    // The provider.
    let mut provider = original.clone();
    provider.provider = antecedent_estimate::EMPIRICAL_TABLE_PLUGIN.into();
    assert_eq!(consume_error(&provider), TransportScenarioArtifactError::DataIdentityMismatch);
    // A sample summary an exact provider never had.
    let mut samples = original.clone();
    samples.samples = TransportScenarioArtifactWire::decode(&{
        let (catalog, input) = empirical_fixture();
        let set = TransportScenarioSet::try_new(three(None)).unwrap();
        let empirical = StudyBuilder::transport_scenarios_empirical(
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
        empirical.export(&empirical.estimate(&ctx).unwrap()).unwrap()
    })
    .unwrap()
    .samples;
    assert_eq!(consume_error(&samples), TransportScenarioArtifactError::DataIdentityMismatch);
    // The memory limit the set was decided under is a scientific premise.
    let mut memory = original.clone();
    memory.scenario_memory_bytes = Some(1 << 40);
    assert_eq!(consume_error(&memory), TransportScenarioArtifactError::PremisesMismatch);
    // Premises and data identity are separate digests: data can be replaced
    // (refresh) without touching the premises.
    let refreshed = prepared.refresh(laws([0.1, 0.3, 0.3, 0.3], true), &ctx).unwrap();
    let after = TransportScenarioArtifactWire::decode(
        &refreshed.export(&refreshed.estimate(&ctx).unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(after.premises_digest, original.premises_digest);
    assert_ne!(after.data_digest, original.data_digest);
}

// ---------------------------------------------------------------------------
// Coordinate schema on sample rows (D2)
// ---------------------------------------------------------------------------

#[test]
fn sample_rows_are_validated_against_the_shared_schema() {
    let ctx = ExecutionContext::for_tests(1);
    let set = TransportScenarioSet::try_new(three(None)).unwrap();
    let attempt = |edit: &dyn Fn(&mut StatisticalTransportInput)| {
        let (catalog, mut input) = empirical_fixture();
        edit(&mut input);
        StudyBuilder::transport_scenarios_empirical(
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
    };
    // A binary outcome that takes the value 2, or a non-finite value.
    for bad in [2.0, f64::NAN, -1.0] {
        let (code, message) = refused_code(attempt(&|input| {
            input.samples[1].columns.get_mut(&v(Y)).unwrap()[0] = Some(bad);
        }));
        assert_eq!(code, "schema_mismatch", "{bad}");
        assert!(message.starts_with("scenarios.coordinate_mismatch"), "{message}");
    }
    // A column for a variable the schema does not declare.
    let (code, _) = refused_code(attempt(&|input| {
        input.samples[0].columns.insert(v(9), vec![Some(0.0); 5]);
    }));
    assert_eq!(code, "schema_mismatch");
    // A concrete intervention outside the declared domain of its variable.
    let (code, _) = refused_code(attempt(&|input| {
        input.samples[1].interventions =
            Arc::from([InterventionAssignment::concrete(v(X), Value::f64(3.0))]);
    }));
    assert_eq!(code, "schema_mismatch");
    // A missing entry is not a value: it is refused or handled by missingness
    // elsewhere, never as a domain violation of the schema.
    assert!(attempt(&|_| {}).is_ok());
}

// ---------------------------------------------------------------------------
// Shared budget (D3)
// ---------------------------------------------------------------------------

/// Bytes an engine holds per charged step on the three-variable schema.
const PER: u64 = 3 * 3 * 64 + 512;

fn decided_with(
    scenarios: Vec<TransportScenario>,
    budget: SearchLimits,
    memory: Option<u64>,
) -> ScenarioSetDecision {
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: memory };
    antecedent_identify::sid::scenarios::decide_transport_scenarios(
        &TransportScenarioSet::try_new(scenarios).unwrap(),
        &query(),
        &catalog(true),
        budget,
        &ctx,
    )
    .unwrap()
}

/// The smallest hard memory limit (a multiple of one engine step) under which
/// `scenarios` decide completely.
fn memory_to_decide(scenarios: impl Fn() -> Vec<TransportScenario>) -> u64 {
    (1..400u64)
        .map(|k| k * PER)
        .find(|limit| decided_with(scenarios(), BUDGET, Some(*limit)).receipt.is_none())
        .expect("a small graph decides within 400 engine steps of memory")
}

fn only(name: &'static str) -> impl Fn() -> Vec<TransportScenario> {
    move || three(None).into_iter().filter(|s| &*s.name == name).collect()
}

#[test]
fn memory_is_cumulative_across_the_scenarios_of_a_set() {
    // `outcome_shift` alone decides within its own peak; after `direct` has been
    // decided, the memory `direct` keeps holding counts against it too.
    let alone = memory_to_decide(only("outcome_shift"));
    assert!(decided_with(only("outcome_shift")(), BUDGET, Some(alone)).receipt.is_none());
    let set = decided_with(three(None), BUDGET, Some(alone));
    let receipt = set.receipt.as_ref().expect("the set does not fit where one scenario does");
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(alone));
    assert_eq!(receipt.explored, ["direct"]);
    assert_eq!(receipt.unevaluated, ["outcome_shift", "standardize"]);
    // The whole set needs more than any one scenario, and decides at that limit.
    let total = memory_to_decide(|| three(None));
    assert!(total > alone, "{total} vs {alone}");
}

#[test]
fn the_entry_charge_stops_a_later_scenario_before_its_search_starts() {
    // `direct` is decided first (canonical order). Under exactly the memory it
    // needs, the second scenario's entry charge (its graph on top of what
    // `direct` holds) is the first charge to exceed the limit.
    let direct_ops = operations_to_decide(|| vec![scenario("direct", &[], None)]);
    let limit = memory_to_decide(only("direct"));
    let at_entry = decided_with(three(None), BUDGET, Some(limit));
    let receipt = at_entry.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(
        receipt.operations_consumed,
        Some(direct_ops),
        "no operation of the second scenario"
    );
    assert_eq!(receipt.explored, ["direct"]);
    assert_eq!(receipt.unevaluated, ["outcome_shift", "standardize"]);
    let statuses = at_entry.decisions.iter().map(|d| d.outcome.status()).collect::<Vec<_>>();
    assert_eq!(statuses, ["identified", "unevaluated", "unevaluated"]);
    // One more graph of room lets the second scenario enter and search; the
    // limit then stops it in the middle of its search, after further operations.
    let mid = decided_with(three(None), BUDGET, Some(limit + PER));
    let receipt = mid.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert!(receipt.operations_consumed.unwrap() > direct_ops);
    assert_eq!(receipt.explored, ["direct"]);
    assert_eq!(receipt.unevaluated, ["outcome_shift", "standardize"]);
}

#[test]
fn the_depth_limit_stops_a_scenario_in_the_middle_of_its_search() {
    let decides = |depth: usize| {
        decided_with(three(None), SearchLimits { operations: 100_000, depth }, None)
            .receipt
            .is_none()
    };
    let needed = (1..256).find(|d| decides(*d)).unwrap();
    assert!(needed > 1, "the entry charge is at depth one; searches go deeper");
    let stopped =
        decided_with(three(None), SearchLimits { operations: 100_000, depth: needed - 1 }, None);
    let receipt = stopped.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Depth);
    // The stop came after the scenario was entered, not at the door.
    assert!(receipt.operations_consumed.unwrap() > 1, "{receipt:?}");
    assert!(receipt.depth_reached.unwrap() >= needed);
    assert!(stopped.decisions.iter().any(|d| d.outcome.status() == "unevaluated"));
}

#[test]
fn explored_and_unevaluated_scenarios_are_disjoint_and_cover_the_set() {
    let total = operations_to_decide(|| three(None));
    for operations in 1..total {
        let decision = decided_under(three(None), operations);
        let receipt = decision.receipt.as_ref().unwrap();
        for name in &receipt.explored {
            assert!(
                !receipt.unevaluated.contains(name),
                "{name} is both explored and unevaluated after {operations} operations"
            );
        }
        let mut all = receipt.explored.iter().chain(&receipt.unevaluated).collect::<Vec<_>>();
        all.sort();
        assert_eq!(all, ["direct", "outcome_shift", "standardize"], "{operations}");
        // Explored is exactly the scenarios that were decided.
        let decided = decision
            .decisions
            .iter()
            .filter(|d| d.outcome.status() != "unevaluated")
            .map(|d| d.scenario.name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(receipt.explored, decided, "{operations}");
    }
}

// ---------------------------------------------------------------------------
// Truncated reports and artifacts (D3e)
// ---------------------------------------------------------------------------

#[test]
fn a_report_truncated_by_cancellation_is_not_exported() {
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.progress = Some(Arc::new(CancelAfterFirst(ctx.cancellation.clone())));
    let prepared =
        try_prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET, &ctx).unwrap();
    let report = prepared.estimate(&ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(report.receipt.as_ref().unwrap().stop, SearchStop::Cancelled);
    // No consumer can reproduce where the interruption fell, so the exporter
    // refuses instead of writing an artifact that can only fail replay.
    assert!(matches!(
        prepared.export(&report),
        Err(IoError::TransportScenario(TransportScenarioArtifactError::CancelledNotReplayable))
    ));
}

#[test]
fn a_report_truncated_by_a_recorded_bound_replays_to_the_identical_prefix() {
    let consume = |bytes: &[u8]| {
        consume_transport_scenarios_artifact(
            bytes,
            TransportScenarioConsumeLimits::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap()
    };
    let same = |a: &ScenarioSetReport, b: &ScenarioSetReport| {
        let statuses =
            |r: &ScenarioSetReport| r.scenarios.iter().map(|s| s.status).collect::<Vec<_>>();
        assert_eq!(statuses(a), statuses(b));
        let (ra, rb) = (a.receipt.as_ref().unwrap(), b.receipt.as_ref().unwrap());
        assert_eq!(
            (ra.stop, &ra.explored, &ra.unevaluated),
            (rb.stop, &rb.explored, &rb.unevaluated)
        );
        assert_eq!(ra.operations_consumed, rb.operations_consumed);
    };
    let ctx = ExecutionContext::for_tests(1);
    // Operation count.
    let direct = operations_to_decide(|| vec![scenario("direct", &[], None)]);
    for operations in [direct, direct + 5] {
        let prepared = prepare(
            three(None),
            catalog(true),
            laws(SOURCE, true),
            SearchLimits { operations, depth: 256 },
        );
        let report = prepared.estimate(&ctx).unwrap();
        same(&report, &consume(&prepared.export(&report).unwrap()));
    }
    // Memory: the producer's hard limit is recorded, and the consumer replays
    // under it even though its own context has no limit.
    for limit in [memory_to_decide(only("direct")), memory_to_decide(only("direct")) + PER] {
        let mut tight = ExecutionContext::for_tests(1);
        tight.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(limit) };
        let prepared =
            try_prepare(three(None), catalog(true), laws(SOURCE, true), BUDGET, &tight).unwrap();
        let report = prepared.estimate(&ctx).unwrap();
        assert_eq!(report.receipt.as_ref().unwrap().stop, SearchStop::Memory);
        same(&report, &consume(&prepared.export(&report).unwrap()));
    }
    // Depth.
    let prepared = prepare(
        three(None),
        catalog(true),
        laws(SOURCE, true),
        SearchLimits { operations: 100_000, depth: 2 },
    );
    let report = prepared.estimate(&ctx).unwrap();
    assert_eq!(report.receipt.as_ref().unwrap().stop, SearchStop::Depth);
    same(&report, &consume(&prepared.export(&report).unwrap()));
}

// ---------------------------------------------------------------------------
// Empirical path (D4)
// ---------------------------------------------------------------------------

#[test]
fn empirical_tables_are_fitted_once_and_coverage_is_per_scenario() {
    let (catalog, input) = empirical_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let set = TransportScenarioSet::try_new(three(None)).unwrap();
    let prepare = |input: StatisticalTransportInput| {
        StudyBuilder::transport_scenarios_empirical(
            &set,
            query(),
            catalog.clone(),
            BUDGET,
            input,
            1000,
            request(),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap()
    };
    let full = prepare(input.clone());
    // Two identified scenarios, two samples, two fitted tables: fitted once for
    // the set, never once per scenario.
    let data = full.prepared().data();
    assert_eq!(data.laws().len(), input.samples.len());
    assert!(
        data.laws()
            .iter()
            .all(|law| { law.origin() == antecedent_expr::LawOrigin::EmpiricalPlugin })
    );
    assert_eq!(full.prepared().plan_summary().iter().filter(|(_, k)| k == "compiled").count(), 2);
    // Without the target sample, `standardize` (whose derivation cites the
    // target law) has no provider while `direct` (source only) still compiles:
    // coverage is decided per scenario against the one set of fitted tables.
    let trial_only =
        StatisticalTransportInput { supplied: Vec::new(), samples: vec![input.samples[1].clone()] };
    let partial = prepare(trial_only);
    assert_eq!(partial.prepared().data().laws().len(), 1);
    let report = partial.estimate(&ctx).unwrap();
    assert_eq!(status(&report, "standardize").0, "unsupported_provider");
    assert_eq!(status(&report, "direct").0, "identified");
    assert!(close(status(&report, "direct").1, 0.6));
}

// ---------------------------------------------------------------------------
// Order-independent masses (D5)
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::float_cmp, reason = "weights summing to one leave exactly zero unaccounted mass")]
fn weights_that_sum_to_one_leave_no_unaccounted_mass_whatever_the_names() {
    // 0.7 + 0.2 + 0.1 is 0.9999999999999999 added left to right and 1.0 right to
    // left: the result must not depend on which order the names put them in.
    let selections: [&[u32]; 3] = [&[], &[Z], &[X]];
    let weights = [0.7, 0.2, 0.1];
    let mut sums = Vec::new();
    for names in [["a", "b", "c"], ["c", "b", "a"], ["b", "c", "a"]] {
        let scenarios =
            (0..3).map(|k| scenario(names[k], selections[k], Some(weights[k]))).collect();
        let report = prepare(scenarios, catalog(true), laws(SOURCE, true), BUDGET)
            .estimate(&ExecutionContext::for_tests(1))
            .unwrap();
        assert!(report.scenarios.iter().all(|s| s.status == "identified"), "{names:?}");
        let weighted = report.weighted.as_ref().unwrap();
        assert_eq!(weighted.unaccounted_mass, 0.0, "{names:?}");
        assert_eq!(report.residual_mass, Some(0.0), "{names:?}");
        let (_, lo, hi) = weighted.ranges.as_ref().unwrap()[0];
        assert_eq!(lo.to_bits(), hi.to_bits(), "a point once every unit of mass is accounted for");
        sums.push((
            weighted.identified_mass.to_bits(),
            weighted.identified_weighted_sums[0].1.to_bits(),
        ));
    }
    assert!(sums.iter().all(|s| *s == sums[0]), "{sums:?}");
}

// ---------------------------------------------------------------------------
// Classical route scope (D6, D7)
// ---------------------------------------------------------------------------

/// The classical row's own limits say only single-outcome four-node families are
/// enumerated against a latent-SCM oracle; the route itself is a sound checker
/// (a derivation or an s-hedge, each independently verified) for any selection
/// ADMG. This sweep looks for a real input on which the scenario path returns
/// `not_certified`: random ADMGs of five to nine variables, several outcomes and
/// treatments, random selections. None is found, so the route's `not_certified`
/// status stays exercised only by a supplied decision (see the test above).
#[test]
fn no_random_admg_makes_the_classical_route_return_not_certified() {
    let ctx = ExecutionContext::for_tests(1);
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut statuses = std::collections::BTreeMap::<&str, usize>::new();
    for _ in 0..1500 {
        let n = 5 + (next() % 5) as u32;
        let mut graph = Admg::with_variables(n);
        for a in 0..n {
            for b in (a + 1)..n {
                if next() % 3 == 0 {
                    graph
                        .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                        .unwrap();
                }
                if next() % 4 == 0 {
                    let _ =
                        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b));
                }
            }
        }
        let selections = (0..n).filter(|_| next() % 3 == 0).map(v).collect::<Vec<_>>();
        let Ok(diagram) = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections))
        else {
            continue;
        };
        let (mut outcomes, mut treatments) = (Vec::new(), Vec::new());
        for i in 0..n {
            match next() % 3 {
                0 => outcomes.push(v(i)),
                1 => treatments.push(v(i)),
                _ => {}
            }
        }
        if outcomes.is_empty() || treatments.is_empty() {
            continue;
        }
        let coordinates = (0..n)
            .map(|i| ScenarioCoordinate {
                variable: v(i),
                name: Arc::from(format!("v{i}")),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect();
        let Ok(set) = TransportScenarioSet::try_new(vec![TransportScenario {
            name: Arc::from("only"),
            diagram,
            weight: None,
            coordinates,
        }]) else {
            continue;
        };
        let everything = (0..n).map(v).collect::<Vec<_>>();
        let measured = everything.iter().copied().filter(|x| !treatments.contains(x));
        let regimes = vec![
            regime_over(1, "source", &treatments, measured.collect()),
            regime_over(0, "target", &[], everything),
        ];
        let catalog = EvidenceCatalog::try_new([], regimes, [], None).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: outcomes.into(),
            treatments: treatments.into(),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let decision = antecedent_identify::sid::scenarios::decide_transport_scenarios(
            &set,
            &query,
            &catalog,
            SearchLimits { operations: 200_000, depth: 256 },
            &ctx,
        )
        .unwrap();
        *statuses.entry(decision.decisions[0].outcome.status()).or_default() += 1;
    }
    assert!(statuses.get("identified").copied().unwrap_or(0) > 100, "{statuses:?}");
    assert!(statuses.get("structurally_unidentified").copied().unwrap_or(0) > 10, "{statuses:?}");
    assert_eq!(statuses.get("not_certified"), None, "{statuses:?}");
    assert_eq!(statuses.get("unevaluated"), None, "{statuses:?}");
}

fn regime_over(
    id: u32,
    population: &str,
    interventions: &[VariableId],
    measured: Vec<VariableId>,
) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        interventions.to_vec(),
        [],
        measured,
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

/// Compatibility of the version 1 format: any version other than 1 and 2 is
/// refused before the payload is read, a version 1 artifact relabelled 2 is
/// refused for its content, and an unknown or missing feature is refused.
#[test]
fn a_version_one_artifact_refuses_other_versions_and_unknown_features() {
    let prepared = prepare(three(None), bound_catalog(), laws(SOURCE, true), BUDGET);
    let ctx = ExecutionContext::for_tests(1);
    let bytes = prepared.export(&prepared.estimate(&ctx).unwrap()).unwrap();
    let original = TransportScenarioArtifactWire::decode(&bytes).unwrap();
    assert_eq!(
        original.version,
        antecedent_io::transport_scenario_artifact::TRANSPORT_SCENARIO_ARTIFACT_VERSION
    );
    for version in [0, 3] {
        let mut wire = original.clone();
        wire.version = version;
        assert!(matches!(
            consume_transport_scenarios_artifact(
                &wire.export().unwrap(),
                TransportScenarioConsumeLimits::default(),
                &ctx
            ),
            Err(IoError::UnsupportedVersion { version: v }) if v == version
        ));
    }
    let mut relabelled = original.clone();
    relabelled.version = 2;
    assert!(matches!(
        consume_error(&relabelled),
        TransportScenarioArtifactError::UnsupportedSemantics(_)
    ));
    for features in [vec!["future_semantics_v9".to_owned()], Vec::new()] {
        let mut wire = original.clone();
        wire.required_features = features;
        assert_eq!(
            consume_error(&wire),
            TransportScenarioArtifactError::UnsupportedSemantics("required features")
        );
    }
}

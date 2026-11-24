//! The direct transport and recovery identifiers are the same computation as their compiled
//! identification programs.
//!
//! `compile_transport_identification` lowers each request to a program bound to the support
//! unit of its capability, and `identify` calls the same kernel function with the same inputs
//! inside the program's dispatch. Each test calls the kernel identifier directly, then the
//! compiled program, on the fixtures the kernel's own known-truth and lifecycle tests use, and
//! compares the decisions: the variant, the derivation record, the rendered functional, the
//! cited evidence and the binding. Identification computes no estimate and carries no
//! interval, so there is no calibration key to compare; the numbers the bound functional later
//! evaluates come from the same derivation.
//!
//! The arena of a derivation interns through hash maps, so its `Debug` is not compared; the
//! canonical record and the rendered expression are.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    dead_code,
    clippy::too_many_lines,
    clippy::similar_names,
    reason = "each fixture states its graph, query and catalog in full; the shared fixture modules serve several binaries"
)]

#[path = "../../antecedent-estimate/tests/common/admg_conditional_scm.rs"]
mod admg_conditional_scm;
#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;
#[path = "../../antecedent-estimate/tests/common/mixed_fixture.rs"]
mod mixed_fixture;
#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::transport_identification::{
    TransportIdentificationOutcome as Outcome, TransportIdentificationRequest as Request,
    compile_transport_identification,
};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef, SelectionDiagram};
use antecedent_identify::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, BoundMixedSourceFunctional, BoundMzTransportFunctional,
    BoundTransportFunctional, BoundZTransportFunctional, COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
    CatalogTransportResult, ChainPartial, ChainRecoveryQuery, ClassicalTransportDerivation,
    ClassicalTransportQuery, ClassicalTransportResult, ConditionalTransportDecision,
    CounterfactualTransportDecision, MIXED_SOURCE_DEFAULT_LIMITS, MZ_TRANSPORT_DEFAULT_LIMITS,
    MetaSource, MetaTransportQuery, MixedSourceDecision, MzTransportDecision, RecoveredEffectQuery,
    RecoveryDecision, RecoveryLimits, SidLimits, TwoSourceZTransportDecision,
    TwoSourceZTransportQuery, ZTransportDecision, ZTransportDerivation, ZTransportOutcome,
    ZTransportQuery, ZTransportResult, ZTransportSourceSpec,
};

use crate::counterfactual_transport_adapter as adapter;
use crate::counterfactual_transport_cases::all_cases;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0)
}

/// Run `request` through its compiled program.
fn compiled(request: Request) -> Outcome {
    compile_transport_identification(request, &ctx()).unwrap().identify(&ctx()).unwrap()
}

fn json(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).unwrap()
}

// --- canonical texts of the kernel results ---------------------------------------------------

fn classical_derivation(d: &ClassicalTransportDerivation) -> String {
    format!("{} | {}", json(&d.to_record()), d.arena().pretty(d.root()))
}

fn classical(result: &ClassicalTransportResult) -> String {
    match result {
        ClassicalTransportResult::Identified(d) => {
            format!("identified {}", classical_derivation(d))
        }
        ClassicalTransportResult::NotCertified => "not_certified".to_owned(),
        ClassicalTransportResult::ProvenNonTransportable(witness) => {
            format!("obstruction {witness:?}")
        }
    }
}

fn bound_classical(b: &BoundTransportFunctional) -> String {
    format!(
        "{} | {} | {:?}",
        classical_derivation(b.derivation()),
        b.arena().pretty(b.root()),
        b.searched_alternatives()
    )
}

fn catalog_result(result: &CatalogTransportResult) -> String {
    match result {
        CatalogTransportResult::Identified(b) => format!("identified {}", bound_classical(b)),
        CatalogTransportResult::MissingEvidence { searched, obligations } => {
            format!("missing {searched:?} {obligations:?}")
        }
        CatalogTransportResult::NotCertified { searched, obligations } => {
            format!("not_certified {searched:?} {obligations:?}")
        }
    }
}

fn z_derivation(d: &ZTransportDerivation) -> String {
    format!("{} | {}", json(&d.to_record()), d.arena().pretty(d.root()))
}

fn z_result(result: &ZTransportResult) -> String {
    match result {
        ZTransportResult::Identified(d) => format!("identified {}", z_derivation(d)),
        ZTransportResult::NotCertified { reason, .. } => format!("not_certified {reason}"),
    }
}

fn bound_z(b: &BoundZTransportFunctional) -> String {
    format!(
        "{} | {} | {:?}",
        z_derivation(b.derivation()),
        b.arena().pretty(b.root()),
        b.cited_regimes()
    )
}

fn bound_mz(b: &BoundMzTransportFunctional) -> String {
    format!(
        "{} | {} | {:?}",
        json(&b.derivation().to_record()),
        b.arena().pretty(b.root()),
        b.cited_populations()
    )
}

fn bound_mixed(b: &BoundMixedSourceFunctional) -> String {
    format!("{:?} | {:?} | {:?}", b.cited_regimes(), b.cited_sources(), b.root())
}

// --- classical sID, the evidence catalog and meta-transport ----------------------------------

/// `X -> Y` with a selection node on `Y`: the target response is transported by the target's
/// own law (Pearl and Bareinboim 2011, the S-admissible direct case).
fn classical_fixture() -> (SelectionDiagram, ClassicalTransportQuery, EvidenceCatalog) {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let catalog = EvidenceCatalog::try_new(
        [],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(0), v(1)],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap()],
        [],
        None,
    )
    .unwrap();
    (diagram, query, catalog)
}

#[test]
fn classical_sid_equals_the_compiled_identification() {
    let (diagram, query, _) = classical_fixture();
    let direct = antecedent_identify::identify_classical_transport(
        &diagram,
        &query,
        SidLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!(matches!(direct, ClassicalTransportResult::Identified(_)), "the fixture identifies");
    let Outcome::Classical(twin) =
        compiled(Request::Classical { diagram, query, limits: SidLimits::default() })
    else {
        panic!("a classical outcome");
    };
    assert_eq!(classical(&direct), classical(&twin));
}

#[test]
fn catalog_transport_equals_the_compiled_identification() {
    let (diagram, query, catalog) = classical_fixture();
    let direct = antecedent_identify::identify_catalog_transport(
        &diagram,
        &query,
        &catalog,
        SidLimits::default(),
        &ctx(),
    )
    .unwrap();
    let Outcome::Catalog(twin) =
        compiled(Request::Catalog { diagram, query, catalog, limits: SidLimits::default() })
    else {
        panic!("a catalog outcome");
    };
    assert_eq!(catalog_result(&direct), catalog_result(&twin));
}

#[test]
fn catalog_search_and_binding_of_a_derivation_equal_the_compiled_identification() {
    let (diagram, query, catalog) = classical_fixture();
    let ClassicalTransportResult::Identified(derivation) =
        antecedent_identify::identify_classical_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &ctx(),
        )
        .unwrap()
    else {
        panic!("the fixture identifies");
    };

    let searched =
        derivation.search_catalog(&diagram, &catalog, SidLimits::default(), &ctx()).unwrap();
    let Outcome::CatalogSearch(twin) = compiled(Request::CatalogSearch {
        derivation: derivation.clone(),
        diagram,
        catalog: catalog.clone(),
        limits: SidLimits::default(),
    }) else {
        panic!("a catalog search outcome");
    };
    assert_eq!(catalog_result(&searched), catalog_result(&twin));

    let bound =
        derivation.bind_catalog_with_context(&catalog, SidLimits::default(), &ctx()).unwrap();
    let Outcome::CatalogBinding(twin) =
        compiled(Request::CatalogBinding { derivation, catalog, limits: SidLimits::default() })
    else {
        panic!("a catalog binding outcome");
    };
    assert_eq!(bound_classical(&bound), bound_classical(&twin));
}

#[test]
fn meta_transport_equals_the_compiled_identification() {
    // Two complementary sources, neither of which transports alone.
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(0, 1), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 1), (0, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = MetaTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        target: Arc::from("target"),
        sources: vec![
            MetaSource { population: "a".into(), selections: vec![2] },
            MetaSource { population: "b".into(), selections: vec![1] },
        ],
    };
    let direct =
        antecedent_identify::identify_meta_transport(&graph, &query, SidLimits::default(), &ctx())
            .unwrap();
    assert!(matches!(direct, ClassicalTransportResult::Identified(_)), "combined sources identify");
    let Outcome::Meta(twin) =
        compiled(Request::Meta { graph, query, limits: SidLimits::default() })
    else {
        panic!("a meta outcome");
    };
    assert_eq!(classical(&direct), classical(&twin));
}

// --- z-transport -------------------------------------------------------------------------------

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

/// The four-node surrogate graph `W -> Z -> X -> Y`, `W -> Y` with `W <-> Y`, `Z <-> Y`,
/// `Z <-> X`, and a source experiment on `Z`.
fn z_fixture() -> (SelectionDiagram, ZTransportQuery, EvidenceCatalog) {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [Environment::try_new(
            "source",
            [W, Z, X, Y]
                .into_iter()
                .map(|variable| VariableCoordinate {
                    variable,
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            [],
        )
        .unwrap()],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("snapshot-0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap();
    (diagram, query, catalog)
}

#[test]
fn z_transport_equals_the_compiled_identification() {
    let (diagram, query, _) = z_fixture();
    let direct =
        antecedent_identify::identify_z_transport(&diagram, &query, SidLimits::default(), &ctx())
            .unwrap();
    let Outcome::ZTransport(twin) =
        compiled(Request::ZTransport { diagram, query, limits: SidLimits::default() })
    else {
        panic!("a z-transport outcome");
    };
    assert_eq!(z_result(&direct), z_result(&twin));
}

#[test]
fn a_z_transport_binding_equals_the_compiled_binding() {
    let (diagram, query, catalog) = z_fixture();
    let direct =
        antecedent_identify::identify_z_transport(&diagram, &query, SidLimits::default(), &ctx())
            .unwrap();
    let ZTransportResult::Identified(derivation) = &direct else {
        panic!("the surrogate formula identifies");
    };
    let bound =
        antecedent_identify::bind_z_transport_catalog(&diagram, &query, derivation, &catalog)
            .unwrap();

    let Outcome::ZTransportBinding(twin_bound) = compiled(Request::ZTransportBinding {
        diagram,
        query,
        derivation: derivation.clone(),
        catalog,
    }) else {
        panic!("a z-transport binding outcome");
    };
    assert_eq!(bound_z(&bound), bound_z(&twin_bound));
}

#[test]
fn inspected_z_transport_equals_the_compiled_identification() {
    let (diagram, query, catalog) = z_fixture();
    let text = |outcome: &ZTransportOutcome| match outcome {
        ZTransportOutcome::Decided(ZTransportDecision::Identified(d)) => {
            format!("identified {}", z_derivation(d))
        }
        ZTransportOutcome::Decided(ZTransportDecision::MissingEvidence { missing }) => {
            format!("missing {missing:?}")
        }
        ZTransportOutcome::Decided(ZTransportDecision::NotCertified { reason, .. }) => {
            format!("not_certified {reason}")
        }
        ZTransportOutcome::Decided(ZTransportDecision::ProvenNonTransportable(o)) => {
            format!("obstruction {}", json(&o.to_record()))
        }
        ZTransportOutcome::Exhausted(receipt) => format!("exhausted {receipt:?}"),
    };
    let direct = antecedent_identify::decide_z_transport_inspecting(
        &diagram,
        &query,
        &catalog,
        SidLimits::default(),
        &ctx(),
    )
    .unwrap();
    let Outcome::ZTransportInspecting(twin) = compiled(Request::ZTransportInspecting {
        diagram,
        query,
        catalog,
        limits: SidLimits::default(),
    }) else {
        panic!("an inspected z-transport outcome");
    };
    assert_eq!(text(&direct), text(&twin));
}

/// The disconnected graph `W -> Z`, `X -> Y`: source `alpha` runs `do(W)` measuring `Z`, source
/// `beta` runs `do(X)` measuring `Y`; together they identify the joint target response.
fn two_source_catalog(
    population: &str,
    treatment: VariableId,
    outcome: VariableId,
) -> EvidenceCatalog {
    let coordinates = [0, 1, 2, 3].map(|variable| VariableCoordinate {
        variable: v(variable),
        domain: VariableDomain::Binary,
        unit: None,
    });
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [treatment],
        [InterventionAssignment { variable: treatment, value: Value::Bool(false) }],
        [outcome],
        population,
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new(
        [
            Environment::try_new("alpha", coordinates.to_vec(), []).unwrap(),
            Environment::try_new("beta", coordinates.to_vec(), []).unwrap(),
            Environment::try_new("target", coordinates.to_vec(), []).unwrap(),
        ],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from(format!("{population}-0")),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap()
}

#[test]
fn two_source_z_transport_equals_the_compiled_identification() {
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    let source = |population: &str, treatment: VariableId| ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: Arc::from([treatment]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: treatment,
            value: Value::Bool(false),
        }]),
        selection_targets: Arc::from([]),
    };
    let query = TwoSourceZTransportQuery {
        outcomes: Arc::from([Z, Y]),
        treatments: Arc::from([W, X]),
        target: Arc::from("target"),
        sources: [source("alpha", W), source("beta", X)],
    };
    let alpha = two_source_catalog("alpha", W, Z);
    let beta = two_source_catalog("beta", X, Y);
    let text = |decision: &TwoSourceZTransportDecision| match decision {
        TwoSourceZTransportDecision::CombinedIdentified { components, factorization } => format!(
            "combined {} / {} | {factorization:?}",
            z_derivation(&components[0].derivation),
            z_derivation(&components[1].derivation)
        ),
        TwoSourceZTransportDecision::Identified { source, derivation } => {
            format!("identified {source} {}", z_derivation(derivation))
        }
        TwoSourceZTransportDecision::NotCertified { reason } => format!("not_certified {reason}"),
        TwoSourceZTransportDecision::ProvenNonTransportable { obstructions } => format!(
            "obstruction {} / {}",
            json(&obstructions[0].to_record()),
            json(&obstructions[1].to_record())
        ),
    };
    let direct = antecedent_identify::decide_two_source_z_transport(
        &graph,
        &query,
        [&alpha, &beta],
        SidLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!(
        matches!(direct, TwoSourceZTransportDecision::CombinedIdentified { .. }),
        "complementary joint regimes identify"
    );
    let Outcome::TwoSourceZTransport(twin) = compiled(Request::TwoSourceZTransport {
        graph,
        query,
        catalogs: [alpha, beta],
        limits: SidLimits::default(),
    }) else {
        panic!("a two-source outcome");
    };
    assert_eq!(text(&direct), text(&twin));
}

// --- multi-source mz-transport and the mixed-source search -------------------------------------

#[test]
fn mz_transport_and_its_binding_equal_the_compiled_identification() {
    use common::mz_fixture::{evidence, graph, query, sources};
    let (catalog, _) = evidence();
    let direct = antecedent_identify::decide_mz_transport(
        &graph(),
        &query(sources()),
        &catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ctx(),
    )
    .unwrap();
    let MzTransportDecision::Identified { derivation, cited } = &direct else {
        panic!("the combined catalog identifies: {:?}", direct.reason_code());
    };
    let bound =
        antecedent_identify::bind_mz_transport_catalog(&graph(), derivation, &catalog).unwrap();
    let Outcome::MzTransport(decision) = compiled(Request::MzTransport {
        graph: graph(),
        query: query(sources()),
        catalog: catalog.clone(),
        limits: MZ_TRANSPORT_DEFAULT_LIMITS,
    }) else {
        panic!("an mz-transport outcome");
    };
    let MzTransportDecision::Identified { derivation: twin, cited: twin_cited } = &decision else {
        panic!("the compiled decision identifies");
    };
    assert_eq!(derivation.to_record(), twin.to_record());
    assert_eq!(derivation.arena().pretty(derivation.root()), twin.arena().pretty(twin.root()));
    assert_eq!(cited, twin_cited);

    let Outcome::MzTransportBinding(twin_bound) = compiled(Request::MzTransportBinding {
        graph: graph(),
        derivation: derivation.clone(),
        catalog,
    }) else {
        panic!("an mz-transport binding outcome");
    };
    assert_eq!(bound_mz(&bound), bound_mz(&twin_bound));
}

#[test]
fn mixed_source_search_and_its_binding_equal_the_compiled_identification() {
    use mixed_fixture::{X as MX, Y as MY, Z as MZ, build, frontdoor_scm, graph, query, study};
    let front_door = || graph(3, &[(0, 1), (1, 2)], &[(0, 2)]);
    let (catalog, _) = build(
        &frontdoor_scm(),
        &[study("observational", &[], &[MX, MZ]), study("trial", &[MZ], &[MX, MY])],
    );
    let direct = antecedent_identify::decide_mixed_source(
        &front_door(),
        &query(MY, MX),
        &catalog,
        MIXED_SOURCE_DEFAULT_LIMITS,
        &ctx(),
    )
    .unwrap();
    let MixedSourceDecision::Identified { derivation, cited, alternatives } = &direct else {
        panic!("the complementary catalog identifies");
    };
    let bound = antecedent_identify::bind_mixed_source_catalog(&front_door(), derivation, &catalog)
        .unwrap();
    let Outcome::MixedSource(decision) = compiled(Request::MixedSource {
        graph: front_door(),
        query: query(MY, MX),
        catalog: catalog.clone(),
        limits: MIXED_SOURCE_DEFAULT_LIMITS,
    }) else {
        panic!("a mixed-source outcome");
    };
    let MixedSourceDecision::Identified {
        derivation: twin,
        cited: twin_cited,
        alternatives: twin_alternatives,
    } = &decision
    else {
        panic!("the compiled decision identifies");
    };
    assert_eq!(derivation.to_record(), twin.to_record());
    assert_eq!(cited, twin_cited);
    assert_eq!(
        alternatives
            .iter()
            .map(antecedent_identify::MixedSourceDerivation::to_record)
            .collect::<Vec<_>>(),
        twin_alternatives
            .iter()
            .map(antecedent_identify::MixedSourceDerivation::to_record)
            .collect::<Vec<_>>()
    );

    let Outcome::MixedSourceBinding(twin_bound) = compiled(Request::MixedSourceBinding {
        graph: front_door(),
        derivation: derivation.clone(),
        catalog,
    }) else {
        panic!("a mixed-source binding outcome");
    };
    assert_eq!(bound_mixed(&bound), bound_mixed(&twin_bound));
}

// --- conditional and counterfactual transport --------------------------------------------------

#[test]
fn admg_conditional_transport_equals_the_compiled_identification() {
    use admg_conditional_scm::{Scm, query};
    // `X(0) -> Y(1) -> W(2)`, `X <-> Y`, the target's `W` mechanism selected.
    let model = Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![(0, 1)],
        selected: vec![2],
        params: 1,
        target_zero: Vec::new(),
    };
    let (catalog, _) = model.catalog_and_laws();
    let text = |decision: &ConditionalTransportDecision| match decision {
        ConditionalTransportDecision::Identified(b) => format!(
            "identified {} | {} | {:?}",
            json(&b.derivation().to_record()),
            bound_classical(b.joint()),
            b.cited_leaves()
        ),
        other => format!("{:?} {:?}", other.reason_code(), other.detail_code()),
    };
    let direct = antecedent_identify::decide_admg_conditional_transport(
        &model.diagram(),
        &query(&[1], &[0], &[2]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx(),
    )
    .unwrap();
    assert!(matches!(direct, ConditionalTransportDecision::Identified(_)));
    let Outcome::AdmgConditional(twin) = compiled(Request::AdmgConditional {
        diagram: model.diagram(),
        query: query(&[1], &[0], &[2]),
        catalog,
        limits: ADMG_CONDITIONAL_DEFAULT_LIMITS,
    }) else {
        panic!("a conditional transport outcome");
    };
    assert_eq!(text(&direct), text(&twin));
}

#[test]
fn counterfactual_transport_equals_the_compiled_identification_on_every_named_case() {
    let text = |decision: &CounterfactualTransportDecision| match decision.derivation() {
        Some(derivation) => format!("derivation {}", derivation.canonical_text()),
        None => format!(
            "{:?} {:?} {:?}",
            decision.identification_status(),
            decision.reason_code(),
            decision.detail_code()
        ),
    };
    let mut identified = 0;
    for case in all_cases() {
        let problem = adapter::problem(&case);
        let adapted = adapter::catalog(&case);
        for target in case.targets.iter().filter(|t| t.query.population == case.target_population) {
            let sources = adapter::sources(&case, &adapter::evidence_controllable(&case));
            let query = adapter::query(&case, &target.query, sources);
            let direct = antecedent_identify::identify_counterfactual_transport(
                &problem,
                &query,
                &adapted.catalog,
                COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            let request = Request::Counterfactual {
                problem: problem.clone(),
                query,
                catalog: adapted.catalog.clone(),
                limits: COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
            };
            let compiled =
                compile_transport_identification(request, &ExecutionContext::for_tests(1))
                    .unwrap()
                    .identify(&ExecutionContext::for_tests(1))
                    .unwrap();
            let Outcome::Counterfactual(twin) = compiled else {
                panic!("a counterfactual transport outcome");
            };
            identified += usize::from(direct.derivation().is_some());
            assert_eq!(text(&direct), text(&twin), "{} / {}", case.name, target.label);
        }
    }
    assert!(identified > 0, "at least one named case identifies");
}

// --- observation recovery ----------------------------------------------------------------------

#[test]
fn observation_recovery_equals_the_compiled_identification() {
    use recovery_scm::{MModel, v as rv};
    // Treatment X0, outcome X1, confounder O; R0 <- O, R1 <- X0.
    const EDGES: [(u32, u32); 3] = [(2, 0), (2, 1), (0, 1)];
    let model = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 17).unwrap();
    let mut effect_graph = Admg::empty();
    for n in 0..model.k + model.m {
        effect_graph.add_node(NodeRef::Static(rv(n))).unwrap();
    }
    for (a, b) in EDGES {
        effect_graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let effect = RecoveredEffectQuery {
        graph: effect_graph,
        outcomes: Arc::from([rv(1)]),
        treatments: Arc::from([rv(0)]),
    };
    let direct = antecedent_identify::decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        Some(&effect),
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap();
    let Outcome::ObservationRecovery(twin) = compiled(Request::ObservationRecovery {
        graph: model.graph.clone(),
        query: model.query(),
        catalog: model.catalog(),
        effect: Some(effect),
        limits: RecoveryLimits::default(),
    }) else {
        panic!("an observation recovery outcome");
    };
    match (&direct, &twin) {
        (RecoveryDecision::Recovered(a), RecoveryDecision::Recovered(b)) => {
            assert_eq!(a.record(), b.record());
            assert_eq!(a.identity(), b.identity());
            assert_eq!(a.arena().pretty(a.root()), b.arena().pretty(b.root()));
        }
        (RecoveryDecision::NonRecoverable(a), RecoveryDecision::NonRecoverable(b)) => {
            assert_eq!(a, b);
        }
        _ => panic!("the decisions differ"),
    }
    assert!(matches!(direct, RecoveryDecision::Recovered(_)), "the M-graph recovers");
}

/// Graph over the raw ids in `order` with directed `edges`.
fn chain_graph(order: &[u32], edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense: BTreeMap<u32, DenseNodeId> = BTreeMap::new();
    for raw in order {
        dense.insert(*raw, graph.add_node(NodeRef::Static(v(*raw))).unwrap());
    }
    for (a, b) in edges {
        graph.insert_directed(dense[a], dense[b]).unwrap();
    }
    graph
}

#[test]
fn chain_recovery_equals_the_compiled_identification() {
    // `X1 = 0, X2 = 1, R1 = 2, R2 = 3, X*1 = 4, X*2 = 5`; the chain `R1 -> R2` with `X1 -> R2`.
    let ids = [0, 1, 2, 3, 4, 5];
    let edges = [(0, 1), (0, 4), (2, 4), (1, 5), (3, 5), (2, 3), (0, 3)];
    let query = ChainRecoveryQuery {
        first: ChainPartial { variable: v(0), response: v(2), proxy: v(4) },
        second: ChainPartial { variable: v(1), response: v(3), proxy: v(5) },
    };
    let graph = chain_graph(&ids, &edges);
    let direct = antecedent_identify::decide_chain_recovery(&graph, &query, &ctx()).unwrap();
    let Outcome::ChainRecovery(twin) = compiled(Request::ChainRecovery { graph, query }) else {
        panic!("a chain recovery outcome");
    };
    match (&direct, &twin) {
        (
            antecedent_identify::ChainRecoveryDecision::Recovered(a),
            antecedent_identify::ChainRecoveryDecision::Recovered(b),
        ) => assert_eq!(a, b),
        (
            antecedent_identify::ChainRecoveryDecision::NonRecoverable(a),
            antecedent_identify::ChainRecoveryDecision::NonRecoverable(b),
        ) => assert_eq!(a, b),
        _ => panic!("the decisions differ"),
    }
    assert!(
        matches!(direct, antecedent_identify::ChainRecoveryDecision::Recovered(_)),
        "the response chain recovers"
    );
}

// --- the compiled program -----------------------------------------------------------------------

#[test]
fn each_family_lowers_to_its_own_route() {
    let (diagram, query, catalog) = classical_fixture();
    let classical = compile_transport_identification(
        Request::Classical {
            diagram: diagram.clone(),
            query: query.clone(),
            limits: SidLimits::default(),
        },
        &ctx(),
    )
    .unwrap();
    let catalog_program = compile_transport_identification(
        Request::Catalog { diagram, query, catalog, limits: SidLimits::default() },
        &ctx(),
    )
    .unwrap();
    let (z_diagram, z_query, _) = z_fixture();
    let z = compile_transport_identification(
        Request::ZTransport { diagram: z_diagram, query: z_query, limits: SidLimits::default() },
        &ctx(),
    )
    .unwrap();
    assert_eq!(classical.lowered().route(), "ClassicalTransportIdentification");
    assert_eq!(catalog_program.lowered().route(), "ClassicalTransportIdentification");
    assert_eq!(z.lowered().route(), "ZTransportIdentification");
    assert_ne!(classical.lowered().identity(), catalog_program.lowered().identity());
}

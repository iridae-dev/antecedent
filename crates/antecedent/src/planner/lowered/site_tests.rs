//! Identity-site laws of the lowered program (formal/identity_sites.toml): the scientific
//! problem, the identification layer, the non-graphical identification of a feedback program,
//! every program builder's `finish`, and the decision-free identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CanonicalWriter, CausalSchemaBuilder, Conditioning,
    ExecutionContext, GraphFamily, InferenceSelection, Intervention, InterventionWorld,
    ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget,
    ScientificWorldId, SemanticDigest, TargetForm, TargetPopulation, TimeFrame, Value, VariableId,
    WorldAction, WorldBinding,
};
use antecedent_data::TabularData;
use antecedent_io::cyclic_family_artifact::{
    ExperimentRegimeWire, RegimeClampWire, design_unresolved,
};

use super::{
    CarrierDemand, CategoricalPosteriorLoweredParts, ChainLoweredParts, ClaimDemand, Composition,
    CyclicLoweredParts, DecisionProgram, DesignShape, DiscoveryValidationLoweredParts,
    EventProcessLoweredParts, EventProcessRoute, EvidenceDecisionProgram, FunctionalKind,
    GraphFalsificationLoweredParts, GraphImputationLoweredParts, IdentificationLayer,
    JointPanelLoweredParts, JointPanelRoute, LatentSoftInterventionLoweredParts,
    LocalCyclicCallbackLoweredParts, LoweredParts, LoweredProgram, LoweringDepth,
    MnarFixedShiftLoweredParts, ModelOperationLoweredParts, NonGraphicalIdentification,
    PhysicalLayer, PlanningLoweredParts, ProblemTarget, ProgramCell, ProgramIdentityScope,
    RetainedIdentification, ScientificProblem, StandaloneCompositionLoweredParts,
    TransportedPathSpecificLoweredParts, UncertaintyReading,
};
use crate::analysis::lowering::RouteCell;
use crate::analysis::{CachedTemporalIdentification, IdentificationCache};
use crate::identity_laws::{ADVERSARIAL_TEXT, all_distinct, injective, ordered_laws, set_laws};
use crate::planner::statistical::{DeclaredProgram, StatisticalProgram};
use crate::{Dag, InferenceMode, RefuteSuite, Study};

fn digest(write: impl FnOnce(&mut CanonicalWriter)) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("site");
    write(&mut writer);
    writer.finish()
}

/// The world `do(v0 = level)` read on the mean of `v1`.
fn world_target(level: f64) -> ScientificTarget {
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("row").expect("unit"),
        OutcomeQuantity::mean(VariableId::from_raw(1), QuantityUnits::dimensionless())
            .expect("quantity"),
        Conditioning::unconditional(),
    )
    .expect("binding");
    let action =
        WorldAction::at_decision([Intervention::set(VariableId::from_raw(0), Value::f64(level))])
            .expect("action");
    let world = InterventionWorld::new(
        ScientificWorldId::from_raw(0),
        action,
        true,
        binding,
        TimeFrame::static_decision(),
    )
    .expect("world");
    ScientificTarget::new(
        TargetForm::World(world),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    )
}

fn problem(target: ProblemTarget, variables: &[u32]) -> ScientificProblem {
    ScientificProblem {
        target,
        identity_scope: ProgramIdentityScope::Semantic,
        runtime_query: None,
        plan_id: Arc::from("site"),
        data: DataClassification::Tabular,
        graph_class: "dag",
        query_variables: variables.iter().copied().map(VariableId::from_raw).collect(),
        source_digest: None,
        support_coordinate: None,
        supported: true,
        support_status: None,
    }
}

fn problem_digest(problem: &ScientificProblem) -> [u8; 32] {
    digest(|writer| problem.write(writer))
}

#[test]
fn site_lowered_scientific_problem_keeps_target_and_variable_order() {
    // A per-horizon family answers its targets in request order.
    ordered_laws(&[0.0, 1.0, 2.0], |levels: &[f64]| {
        let targets = levels.iter().map(|&level| world_target(level)).collect();
        Some(problem_digest(&problem(ProblemTarget::Targets(targets), &[0, 1])))
    });
    // The query variables are the route record's role-ordered list.
    ordered_laws(&[0_u32, 1, 2], |variables: &[u32]| {
        Some(problem_digest(&problem(ProblemTarget::Target(world_target(1.0)), variables)))
    });
    // One target and a one-target family, and a repeated horizon, are different questions.
    let one = world_target(1.0);
    all_distinct(&[
        problem_digest(&problem(ProblemTarget::Target(one.clone()), &[0, 1])),
        problem_digest(&problem(ProblemTarget::Targets(Arc::from(vec![one.clone()])), &[0, 1])),
        problem_digest(&problem(
            ProblemTarget::Targets(Arc::from(vec![one.clone(), one])),
            &[0, 1],
        )),
        problem_digest(&problem(ProblemTarget::Targets(Arc::from(Vec::new())), &[0, 1])),
    ]);
}

#[test]
fn site_lowered_scientific_problem_frames_its_texts_and_options() {
    injective(
        &ADVERSARIAL_TEXT,
        |parts| *parts,
        |(plan, coordinate)| {
            let mut problem = problem(ProblemTarget::Target(world_target(1.0)), &[0, 1]);
            problem.plan_id = Arc::from(*plan);
            problem.support_coordinate = Some(Arc::from(*coordinate));
            Some(problem_digest(&problem))
        },
    );
    let functional = |kind, outcome: Option<u32>, query: u8| {
        problem_digest(&problem(
            ProblemTarget::QueryFunctional {
                kind,
                outcome: outcome.map(VariableId::from_raw),
                query: [query; 32],
            },
            &[0, 1],
        ))
    };
    let mut with_source = problem(ProblemTarget::Target(world_target(1.0)), &[0, 1]);
    with_source.source_digest = Some([0; 32]);
    let mut unsupported = problem(ProblemTarget::Target(world_target(1.0)), &[0, 1]);
    unsupported.supported = false;
    all_distinct(&[
        functional(FunctionalKind::Association, None, 0),
        functional(FunctionalKind::Association, Some(0), 0),
        functional(FunctionalKind::Association, Some(1), 0),
        functional(FunctionalKind::Association, None, 1),
        functional(FunctionalKind::Simulation, None, 0),
        problem_digest(&problem(ProblemTarget::Target(world_target(1.0)), &[0, 1])),
        problem_digest(&with_source),
        problem_digest(&unsupported),
    ]);
}

/// The unresolved directions a clamp design leaves, one regime clamping each of `clamped`, on
/// units `0, 1, 2`, as the cyclic lowering carries them into layer two.
fn recovery(clamped: &[u32]) -> NonGraphicalIdentification {
    let regimes: Vec<ExperimentRegimeWire> = clamped
        .iter()
        .map(|&unit| ExperimentRegimeWire {
            clamps: vec![RegimeClampWire { unit, value: 1.0 }],
            units: vec![0, 1, 2],
            population: 0,
            dependent: false,
            source: u64::from(unit) + 1,
        })
        .collect();
    NonGraphicalIdentification::ExperimentalRecovery {
        design: [0; 32],
        unresolved: design_unresolved(&regimes, &[])
            .into_iter()
            .map(|direction| {
                (
                    VariableId::from_raw(direction.target),
                    direction
                        .missing_clamps
                        .into_iter()
                        .map(VariableId::from_raw)
                        .collect::<Arc<[VariableId]>>(),
                )
            })
            .collect(),
    }
}

fn identification_digest(identification: &NonGraphicalIdentification) -> [u8; 32] {
    digest(|writer| identification.write(writer))
}

#[test]
fn site_lowered_unresolved_directions_are_a_set() {
    // Which directions a design leaves unresolved does not depend on the order or repetition
    // of its regimes; dropping the only regime that clamps a source changes it.
    set_laws(&[0_u32, 1], |clamped: &[u32]| Some(identification_digest(&recovery(clamped))));
}

#[test]
fn site_lowered_non_graphical_identification_tags_and_counts() {
    let unresolved = |rows: &[(u32, Vec<u32>)]| {
        identification_digest(&NonGraphicalIdentification::ExperimentalRecovery {
            design: [0; 32],
            unresolved: rows
                .iter()
                .map(|(target, missing)| {
                    (
                        VariableId::from_raw(*target),
                        missing
                            .iter()
                            .copied()
                            .map(VariableId::from_raw)
                            .collect::<Arc<[VariableId]>>(),
                    )
                })
                .collect(),
        })
    };
    all_distinct(&[
        unresolved(&[]),
        unresolved(&[(0, vec![1, 2])]),
        unresolved(&[(0, vec![1]), (2, vec![])]),
        unresolved(&[(0, vec![]), (1, vec![2])]),
        unresolved(&[(0, vec![1]), (2, vec![1])]),
        identification_digest(&NonGraphicalIdentification::SpecifiedModelConditional {
            model: [0; 32],
        }),
        identification_digest(&NonGraphicalIdentification::CyclicIoScm { source: [0; 32] }),
        identification_digest(&NonGraphicalIdentification::SharedEvidenceContract {
            contract: [0; 32],
        }),
        identification_digest(&NonGraphicalIdentification::DeclaredPolicyEvaluation {
            request: [0; 32],
        }),
        identification_digest(&NonGraphicalIdentification::GraphCheckedPolicyEvaluation {
            request: [0; 32],
        }),
    ]);
}

/// The static identification compile retains for `t -> y`.
fn static_cache() -> IdentificationCache {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.1, 0.9, 2.9, 1.0, 3.0];
    let columns: [(&str, &[f64]); 2] = [("t", &treatment), ("y", &outcome)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let study = Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let plan = study.compile(&ExecutionContext::for_tests(1)).unwrap();
    let retained = plan
        .lowered()
        .and_then(LoweredProgram::retained_identification)
        .expect("compile retains the static identification");
    retained.caches()[0].clone()
}

fn layer_digest(layer: &IdentificationLayer) -> [u8; 32] {
    digest(|writer| layer.write(writer))
}

fn retained(caches: Vec<IdentificationCache>) -> IdentificationLayer {
    IdentificationLayer::Retained(Arc::new(RetainedIdentification {
        product: None,
        caches: Arc::from(caches),
        key: None,
        seals: Arc::from(Vec::new()),
        execution: None,
    }))
}

#[test]
fn site_lowered_identification_layer_orders_caches_and_tags_variants() {
    let pool = [
        static_cache(),
        IdentificationCache::Temporal(Arc::new(CachedTemporalIdentification {
            by_horizon: Arc::from(Vec::new()),
        })),
    ];
    assert_ne!(pool[0].kind(), pool[1].kind());
    // Preparation installs the retained products in order.
    ordered_laws(&[0_usize, 1], |picked: &[usize]| {
        Some(layer_digest(&retained(picked.iter().map(|&index| pool[index].clone()).collect())))
    });
    all_distinct(&[
        layer_digest(&retained(Vec::new())),
        layer_digest(&retained(vec![pool[0].clone()])),
        layer_digest(&retained(vec![pool[0].clone(), pool[0].clone()])),
        layer_digest(&IdentificationLayer::Deferred { reason: "site", failure: None }),
        layer_digest(&IdentificationLayer::PerExecution { reason: "site" }),
        layer_digest(&IdentificationLayer::NotClaimed { refusal: None }),
        layer_digest(&IdentificationLayer::NotClaimed { refusal: Some(Arc::from("site")) }),
        layer_digest(&IdentificationLayer::NonGraphical(Arc::new(
            NonGraphicalIdentification::SpecifiedModelConditional { model: [0; 32] },
        ))),
    ]);
}

/// Every layer but the cell, from a scientific problem with plan id `plan`.
struct Layers {
    scientific: ScientificProblem,
    identification: IdentificationLayer,
    statistical: StatisticalProgram,
    evidence: EvidenceDecisionProgram,
    physical: PhysicalLayer,
    claim: ClaimDemand,
}

fn layers(plan: &str, supported: bool) -> Layers {
    let mut scientific = problem(ProblemTarget::Target(world_target(1.0)), &[0, 1]);
    scientific.plan_id = Arc::from(plan);
    scientific.supported = supported;
    Layers {
        scientific,
        identification: IdentificationLayer::PerExecution { reason: "site" },
        statistical: StatisticalProgram::declared(DeclaredProgram {
            identifier: None,
            estimator: None,
            inference: &InferenceMode::Frequentist,
            bootstrap_replicates: 0,
            sampling_design: None,
            design: DesignShape::not_design_based("site"),
            operators: Vec::new(),
        })
        .expect("declared program"),
        evidence: EvidenceDecisionProgram {
            sources: Arc::from(Vec::new()),
            composition: Composition::Single,
            decision: None,
            planning: None,
            chain: None,
        },
        physical: PhysicalLayer {
            operators: Arc::from(Vec::new()),
            design: DesignShape::not_design_based("site"),
        },
        claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None },
    }
}

/// Seal `layers(plan, supported)` with a builder and its own fields.
macro_rules! sealed {
    ($plan:expr, $supported:expr, $parts:ident { $($field:ident: $value:expr),* }) => {{
        let Layers { scientific, identification, statistical, evidence, physical, claim } =
            layers($plan, $supported);
        $parts { $($field: $value,)* scientific, identification, statistical, evidence, physical, claim }
            .finish()
    }};
}

const EVENT_ROUTES: [EventProcessRoute; 4] = [
    EventProcessRoute::Forecast,
    EventProcessRoute::Marked,
    EventProcessRoute::Excitation,
    EventProcessRoute::ParameterBridge,
];

/// One program from every builder, each with its own cell, family, design or reading.
fn programs(plan: &str, supported: bool) -> Vec<LoweredProgram> {
    let mut programs = vec![
        sealed!(
            plan,
            supported,
            LoweredParts { cell: RouteCell::StaticAteDag, depth: LoweringDepth::Complete }
        ),
        sealed!(
            plan,
            supported,
            LoweredParts {
                cell: RouteCell::StaticAteDag,
                depth: LoweringDepth::AtomOperationsDeferred { operations: "atoms" }
            }
        ),
        sealed!(
            plan,
            supported,
            LoweredParts { cell: RouteCell::GcmTemporal, depth: LoweringDepth::Complete }
        ),
        sealed!(plan, supported, CyclicLoweredParts { family: "linear" }),
        sealed!(plan, supported, CyclicLoweredParts { family: "dynamic" }),
        sealed!(plan, supported, PlanningLoweredParts { design: "parallel" }),
        sealed!(plan, supported, PlanningLoweredParts { design: "crossover" }),
        sealed!(plan, supported, ChainLoweredParts { reading: "spread" }),
        sealed!(plan, supported, ChainLoweredParts { reading: "enclosure" }),
        sealed!(
            plan,
            supported,
            StandaloneCompositionLoweredParts { cell: ProgramCell::DoseGridBiasAware }
        ),
        sealed!(
            plan,
            supported,
            StandaloneCompositionLoweredParts { cell: ProgramCell::ObservationalPolicy }
        ),
        sealed!(plan, supported, GraphImputationLoweredParts {}),
        sealed!(plan, supported, CategoricalPosteriorLoweredParts {}),
        sealed!(plan, supported, GraphFalsificationLoweredParts {}),
        sealed!(plan, supported, TransportedPathSpecificLoweredParts {}),
        sealed!(plan, supported, ModelOperationLoweredParts { attribution: true }),
        sealed!(plan, supported, ModelOperationLoweredParts { attribution: false }),
        sealed!(plan, supported, LocalCyclicCallbackLoweredParts {}),
        sealed!(plan, supported, MnarFixedShiftLoweredParts {}),
        sealed!(plan, supported, DiscoveryValidationLoweredParts {}),
        sealed!(plan, supported, LatentSoftInterventionLoweredParts {}),
    ];
    for event in EVENT_ROUTES {
        programs.push(sealed!(plan, supported, EventProcessLoweredParts { route: event }));
    }
    for panel in JointPanelRoute::ALL {
        programs.push(sealed!(plan, supported, JointPanelLoweredParts { route: panel }));
    }
    programs
}

fn identities(programs: &[LoweredProgram]) -> Vec<SemanticDigest> {
    programs.iter().map(LoweredProgram::identity).collect()
}

#[test]
fn site_lowered_every_builder_binds_its_cell_and_layers() {
    // Every builder names its cell (and family, design, reading or depth), and every builder
    // binds its scientific layer: two plan ids and every cell give as many identities.
    let mut all = identities(&programs("a", true));
    all.extend(identities(&programs("b", true)));
    all_distinct(&all);
}

#[test]
fn site_lowered_annotating_builders_ignore_support_promotion() {
    // A latent soft-intervention, event-process or joint-panel program is the same scientific
    // program whether or not its capability is promoted; every other builder binds the support.
    let supported = programs("a", true);
    let unsupported = programs("a", false);
    for (left, right) in supported.iter().zip(&unsupported) {
        let annotates = matches!(left.cell(), ProgramCell::LatentSoftIntervention)
            || JointPanelRoute::ALL.iter().any(|route| route.cell() == left.cell())
            || EVENT_ROUTES.iter().any(|route| route.cell() == left.cell());
        assert_eq!(annotates, left.identity() == right.identity(), "{}", left.route());
    }
}

#[test]
fn site_lowered_decision_free_identity_names_the_cell_payload() {
    // A named cell's name leaves out its family, design or reading; the decision-free identity
    // writes each after a presence word, so named cells of one kind never compare as one.
    let named = [
        sealed!("a", true, CyclicLoweredParts { family: "linear" }),
        sealed!("a", true, CyclicLoweredParts { family: "dynamic" }),
        sealed!("a", true, PlanningLoweredParts { design: "linear" }),
        sealed!("a", true, ChainLoweredParts { reading: "linear" }),
        sealed!("a", true, GraphImputationLoweredParts {}),
    ];
    all_distinct(&named.iter().map(LoweredProgram::identity_without_decision).collect::<Vec<_>>());
    // Attaching a decision changes the identity by exactly the decision.
    let decision = DecisionProgram {
        carrier: CarrierDemand::Mean,
        utility: DecisionProgram::utility_form(None),
        actions: 2,
        constraints: 0,
        chance_threshold: 0.95,
        utility_unit: None,
        contract_identity: None,
        information_value: None,
        scenario_source: None,
    };
    let estimate = &named[4];
    let decided = estimate.with_decision(decision.clone()).expect("no decision yet");
    assert_ne!(decided.identity(), estimate.identity());
    assert!(decided.decides_over(estimate));
    assert!(decided.with_decision(decision.clone()).is_none());
    let other = named[3].with_decision(decision).expect("no decision yet");
    assert!(!other.decides_over(estimate));
}

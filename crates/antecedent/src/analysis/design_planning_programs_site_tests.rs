//! Category laws on the identity sites of the planning and ranking families
//! (`formal/identity_sites.toml`): each test calls the production compile entry or digest and
//! checks the class of the collections it writes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EnvironmentId, EvidenceCatalog,
    EvidenceCatalogDelta, EvidenceKind, EvidenceRegime, ExecutionContext, InterventionAssignment,
    QueryId, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableCoordinate,
    VariableDomain, VariableId,
};
use antecedent_design::choice_evsi::StudyCandidate as ChoiceStudyCandidate;
use antecedent_design::structure_identification::{GraphIdentFlag, GraphPosteriorSample};
use antecedent_design::{
    AffineUtility, BackdoorRepairFamily, BinomialSignal, CandidateDesign, DecisionPrior,
    DecisionProblem, DecisionRegistry, DesignCost, DesignEvaluationContext, DesignObjective,
    DesignRanker, DurableStudyCandidate, ExpectedEvidence, ExperimentPlan, MeasurementPlan,
    RepairLimits, RepairObjective, StudyCostDeclaration, StudyKind, TransportEvidenceCandidate,
    UnitRules, ZTransportPlanSpec, snapshot_z_transport_failure,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::{SidLimits, ZTransportQuery};
use antecedent_prob::WeightedGraphSamples;
use antecedent_stats::adaptive_inference::HierarchicalChoicePrior;

use super::*;
use crate::identity_laws::{all_distinct, bit_exact, multiset_laws, ordered_laws, set_laws};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

// --- identification repair -------------------------------------------------------------------

const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;

fn backdoor_family() -> BackdoorRepairFamily {
    let mut graph = Dag::with_variables(7);
    for (from, to) in [(Z1, T), (Z1, Y), (Z2, T), (Z2, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    BackdoorRepairFamily::try_new(graph, v(T), v(Y), "pop", [v(T), v(Y)], &[]).unwrap()
}

fn observation(label: &str, measured: &[u32], units: u64) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from(label),
        kind: StudyKind::Observation,
        population: Arc::from("pop"),
        interventions: vars(&[]),
        measured: vars(measured),
        joint_measurement: true,
        sample_size: 200,
        recruitment: Arc::from("consecutive recruitment"),
        timing: Arc::from("baseline"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units, unit_label: Arc::from("USD"), sample_budget: 200 },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: Arc::from([ExpectedEvidence {
            population: Arc::from("pop"),
            interventions: vars(&[]),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: vars(measured),
            distribution: DistributionAvailability::Joint,
        }]),
        external_provider: None,
    }
}

fn repair_identity(
    candidates: &[DurableStudyCandidate],
    objective: RepairObjective,
) -> Option<antecedent_core::SemanticDigest> {
    compile_identification_repair(
        backdoor_family(),
        candidates.to_vec(),
        objective,
        RepairLimits::default(),
        &ctx(),
    )
    .ok()
    .map(|program| program.identity())
}

#[test]
fn site_design_planning_programs_repair_candidates_are_a_multiset_by_semantic_id() {
    let candidates = [
        observation("a", &[T, Y, Z1], 5),
        observation("b", &[T, Y, Z2], 5),
        observation("c", &[T, Y, Z1, Z2], 20),
    ];
    // The search reads the candidates sorted by semantic id and refuses a repeat: reordering
    // keeps the identity, a repeat changes it.
    multiset_laws(&candidates, |items| repair_identity(items, RepairObjective::MinimizeCost));
    all_distinct(&[
        repair_identity(&candidates, RepairObjective::MinimizeCost),
        repair_identity(&candidates, RepairObjective::MinimizeSampleBudget),
        repair_identity(&candidates[..2], RepairObjective::MinimizeCost),
    ]);
}

// --- z-transport evidence planning -----------------------------------------------------------

fn z_query(target: &str) -> ZTransportQuery {
    ZTransportQuery {
        outcomes: vars(&[3]),
        treatments: vars(&[2]),
        controllable: vars(&[1]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: v(1),
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from(target),
    }
}

fn z_diagram() -> SelectionDiagram {
    let mut graph = Admg::with_variables(5);
    for (a, b) in [(1, 2), (2, 3), (0, 2), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap()
}

/// The source experiment on Z, over environments named `names` (each must be distinct).
fn z_catalog(names: &[&str]) -> Option<EvidenceCatalog> {
    let coordinates = (0..5)
        .map(|raw| VariableCoordinate {
            variable: v(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let environments = names
        .iter()
        .map(|name| {
            Environment::try_new(*name, coordinates.clone(), Arc::<[VariableId]>::from([])).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    let experiment = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(1)],
        [InterventionAssignment { variable: v(1), value: Value::Bool(false) }],
        [v(0), v(2), v(3)],
        "source",
        DistributionAvailability::Joint,
    )
    .ok()?;
    EvidenceCatalog::try_new(
        environments,
        [experiment],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(1),
            snapshot_identity: Arc::from("source-do-z0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .ok()
}

fn snapshot_identity(
    catalog: EvidenceCatalog,
    target: &str,
    limits: SidLimits,
) -> Option<antecedent_core::SemanticDigest> {
    compile_z_transport_failure_snapshot(z_diagram(), z_query(target), catalog, limits, &ctx())
        .ok()
        .map(|program| program.identity())
}

#[test]
fn site_design_planning_programs_snapshot_catalog_is_its_canonical_form() {
    // The catalog is written in its canonical form (environments sorted by identity), so the
    // declared order of the environments never reaches the identity.
    set_laws(&["source", "target", "other"], |names| {
        snapshot_identity(z_catalog(names)?, "target", SidLimits::default())
    });
    let base = || z_catalog(&["source", "target"]).unwrap();
    all_distinct(&[
        snapshot_identity(base(), "target", SidLimits::default()),
        snapshot_identity(base(), "source", SidLimits::default()),
        snapshot_identity(base(), "target", SidLimits { steps: 7, depth: 3 }),
        snapshot_identity(base(), "target", SidLimits { steps: 3, depth: 7 }),
    ]);
}

fn measure(base: &EvidenceCatalog, id: &str, population: &str) -> TransportEvidenceCandidate {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(2),
        RegimeKind::Observational,
        EvidenceKind::Proposed,
        [],
        [],
        (0..5).map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap();
    TransportEvidenceCandidate {
        id: Arc::from(id),
        delta: EvidenceCatalogDelta::try_new(base, [regime]).unwrap(),
        design: CandidateDesign::Measure(MeasurementPlan {
            variables: (0..5).map(v).collect::<Vec<_>>().into(),
            cost: DesignCost { amount: 2.0, sample_budget: 50 },
            tag: 1,
        }),
        recruitment_sampling: Arc::from(format!("{population} population")),
        feasibility_constraints: Arc::from([]),
        cost: DesignCost { amount: 2.0, sample_budget: 50 },
    }
}

#[test]
fn site_design_planning_programs_evidence_plan_candidates_are_ordered() {
    let base = z_catalog(&["source", "target"]).unwrap();
    let snapshot = Arc::new(
        snapshot_z_transport_failure(
            &z_diagram(),
            &z_query("target"),
            &base,
            SidLimits::default(),
            &ctx(),
        )
        .unwrap(),
    );
    let identity = |ids: &[&str], max_evaluated: usize| {
        let spec = ZTransportPlanSpec {
            candidates: ids.iter().map(|id| measure(&base, id, "target")).collect(),
            max_evaluated,
            identification_limits: SidLimits::default(),
        };
        compile_z_transport_evidence_plan(Arc::clone(&snapshot), spec, &ctx())
            .ok()
            .map(|program| program.identity())
    };
    // The planner evaluates the first `max_evaluated` candidates in declared order.
    ordered_laws(&["first", "second", "third"], |ids| identity(ids, 2));
    all_distinct(&[identity(&["first", "second"], 1), identity(&["first", "second"], 2)]);
}

// --- design ranking --------------------------------------------------------------------------

fn graphs(keys: &[u64]) -> Option<WeightedGraphSamples> {
    WeightedGraphSamples::new(
        vec![1.0; keys.len()],
        vec![GraphIdentFlag::Unidentified; keys.len()],
        keys.to_vec(),
    )
    .ok()
}

fn evaluation(
    graphs: &WeightedGraphSamples,
    unlock: Option<&[QueryUnlock]>,
    environments: Option<&[EnvironmentUnlock]>,
) -> [u8; 32] {
    let eval = DesignEvaluationContext::<(), ()> {
        graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: unlock,
        env_id_unlock: environments,
        identified_under_intervention: None,
        graph_features: None,
    };
    evaluation_digest(&eval)
}

#[test]
fn site_design_planning_programs_evaluation_unlocks_are_sets_and_entries_ordered() {
    let posterior = graphs(&[1, 2]).unwrap();
    // An unlock list is read by membership only: a set.
    set_laws(&[v(4), v(2), v(7)], |unlocked| {
        let table = [(QueryId::from_raw(0), Arc::<[VariableId]>::from(unlocked))];
        Some(evaluation(&posterior, Some(table.as_slice()), None))
    });
    let environment = EnvironmentId::from_raw;
    set_laws(&[environment(3), environment(1), environment(5)], |listed| {
        let table = [(QueryId::from_raw(0), Arc::<[EnvironmentId]>::from(listed))];
        Some(evaluation(&posterior, None, Some(table.as_slice())))
    });
    // The ranker reads the first entry for a query: the entries are ordered.
    ordered_laws(&[0_u32, 1, 2], |queries| {
        let table: Vec<QueryUnlock> = queries
            .iter()
            .map(|&q| (QueryId::from_raw(q), Arc::<[VariableId]>::from([v(9)])))
            .collect();
        Some(evaluation(&posterior, Some(table.as_slice()), None))
    });
    // Each entry is framed by its query: the same list under another query differs.
    let under = |q: u32| {
        let table = [(QueryId::from_raw(q), Arc::<[VariableId]>::from([v(9)]))];
        evaluation(&posterior, Some(table.as_slice()), None)
    };
    let environments_under = |q: u32| {
        let table = [(QueryId::from_raw(q), Arc::<[EnvironmentId]>::from([environment(9)]))];
        evaluation(&posterior, None, Some(table.as_slice()))
    };
    all_distinct(&[under(0), under(1), environments_under(0), environments_under(1)]);
    // The per-structure arrays are parallel, in sample order.
    ordered_laws(&[1_u64, 2, 3], |keys| Some(evaluation(&graphs(keys)?, None, None)));
}

fn utility(actions: usize) -> Arc<dyn antecedent_design::Utility<usize, f64>> {
    Arc::new(AffineUtility::new(vec![0.0; actions], vec![1.0; actions]).unwrap())
}

fn registry(problems: &[Option<usize>]) -> DecisionRegistry<usize, f64> {
    DecisionRegistry {
        problems: problems
            .iter()
            .map(|&actions| {
                actions.map(|n| DecisionProblem::new((0..n).collect(), utility(n), vec![]))
            })
            .collect(),
        prior: DecisionPrior::Draws(vec![0.2, 0.8]),
        signal: Arc::new(BinomialSignal),
    }
}

#[test]
fn site_design_planning_programs_registry_problems_are_keyed_by_id() {
    // Problems are looked up by raw id: the slot of each problem is its key.
    ordered_laws(&[1_usize, 2, 3], |counts| {
        let slots: Vec<Option<usize>> = counts.iter().copied().map(Some).collect();
        Some(registry_identity(&registry(&slots)))
    });
    all_distinct(&[
        registry_identity(&registry(&[Some(2)])),
        registry_identity(&registry(&[None, Some(2)])),
        registry_identity(&registry(&[Some(2), None])),
    ]);
}

#[test]
fn site_design_planning_programs_ranking_candidates_are_ordered() {
    let posterior = graphs(&[1, 2]).unwrap();
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &posterior,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let experiment = |tag: u64| {
        CandidateDesign::Intervene(ExperimentPlan {
            targets: Arc::from([v(T)]),
            cost: DesignCost::zero(),
            tag,
        })
    };
    let objective =
        || DesignObjective::IncreaseIdentificationProbability { query: QueryId::from_raw(0) };
    // The ranking reports candidates by declared index.
    ordered_laws(&[experiment(1), experiment(2), experiment(3)], |candidates| {
        compile_design_ranking(
            &DesignRanker::new(),
            objective(),
            candidates.to_vec(),
            &eval,
            &ctx(),
        )
        .ok()
        .map(|program| program.identity())
    });
}

// --- choice study ranking --------------------------------------------------------------------

fn choice_identity(
    posterior: ChoicePosterior,
    values: &[f64],
    candidates: &[ChoiceStudyCandidate],
) -> Option<antecedent_core::SemanticDigest> {
    compile_choice_study_ranking(posterior, values.to_vec(), candidates.to_vec(), &ctx())
        .ok()
        .map(|program| program.identity())
}

fn pooled() -> ChoicePosterior {
    ChoicePosterior::Pooled { alpha: vec![1.0, 2.0], seed: 17 }
}

#[test]
fn site_design_planning_programs_choice_ranking_lists_are_ordered_and_reals_bits() {
    let candidate = |label: &str, observations: u32, cost: f64| ChoiceStudyCandidate {
        label: label.into(),
        observations,
        cost,
    };
    let candidates =
        [candidate("one", 1, 0.0), candidate("two", 2, 0.1), candidate("three", 3, 0.2)];
    ordered_laws(&candidates, |items| choice_identity(pooled(), &[1.0, 2.0], items));
    // Values and shapes are per option.
    ordered_laws(&[1.0, 2.0, 3.0], |values| choice_identity(pooled(), values, &candidates));
    ordered_laws(&[1.0, 2.0, 3.0], |alpha| {
        choice_identity(
            ChoicePosterior::Pooled { alpha: alpha.to_vec(), seed: 17 },
            &[1.0],
            &candidates,
        )
    });
    // Respondents are read in declared order (the quadrature sums them in that order).
    let prior = HierarchicalChoicePrior {
        shapes: vec![1.0; 3],
        log_concentration_mean: 1.0,
        log_concentration_sd: 1.0,
    };
    ordered_laws(&[vec![2_u32, 1, 0], vec![0, 2, 1], vec![1, 1, 1]], |respondents| {
        choice_identity(
            ChoicePosterior::Hierarchical {
                prior: prior.clone(),
                respondents: respondents.to_vec(),
            },
            &[1.0, 1.0, 1.0],
            &candidates,
        )
    });
    bit_exact(|cost| choice_identity(pooled(), &[1.0, 2.0], &[candidate("one", 1, cost)]));
    // A label is framed: moving a character between the label and the count is a different
    // candidate.
    all_distinct(&[
        choice_identity(pooled(), &[1.0], &[candidate("a1", 1, 0.0)]),
        choice_identity(pooled(), &[1.0], &[candidate("a", 11, 0.0)]),
        choice_identity(pooled(), &[1.0], &[candidate("a", 1, 0.0)]),
    ]);
}

// --- structure identification ----------------------------------------------------------------

fn sample(key: u64) -> GraphPosteriorSample {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    if key.is_multiple_of(2) {
        graph.insert_bidirected(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    }
    GraphPosteriorSample { weight: 1.0, key, graph }
}

fn structure_identity(
    keys: &[u64],
    interventions: Option<Vec<VariableId>>,
) -> Option<antecedent_core::SemanticDigest> {
    let samples = keys.iter().copied().map(sample).collect();
    compile_structure_identification(samples, v(T), v(Y), interventions, &ctx())
        .ok()
        .map(|program| program.identity())
}

#[test]
fn site_design_planning_programs_structure_samples_and_targets_are_ordered() {
    // Flags are reported per sample in declared order.
    ordered_laws(&[1_u64, 2, 3], |keys| structure_identity(keys, None));
    // An unknown target is reported by the first one met.
    ordered_laws(&[v(T), v(Y)], |targets| structure_identity(&[1, 2], Some(targets.to_vec())));
    all_distinct(&[
        structure_identity(&[1, 2], None),
        structure_identity(&[1, 2], Some(vec![])),
        structure_identity(&[1, 2], Some(vec![v(T)])),
    ]);
    bit_exact(|weight| {
        let mut one = sample(1);
        one.weight = weight;
        compile_structure_identification(vec![one], v(T), v(Y), None, &ctx())
            .ok()
            .map(|program| program.identity())
    });
}

#[test]
fn site_design_planning_programs_graph_identification_frames_its_parts() {
    let identity = |graph: Admg, treatment: u32, outcome: u32| {
        compile_graph_identification(graph, v(treatment), v(outcome), &ctx())
            .ok()
            .map(|program| program.identity())
    };
    all_distinct(&[
        identity(sample(1).graph, T, Y),
        identity(sample(2).graph, T, Y),
        identity(sample(1).graph, Y, T),
        identity(Admg::with_variables(2), T, Y),
    ]);
}

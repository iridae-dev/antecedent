//! The explicit two-slice unrolling of one finite temporal transport sequence
//! (2.2A, X5): its frozen horizon, time order, alphabet and hard caps, and its
//! construction from a lagged template through the ADR 0021 unfolding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{Value, VariableDomain, VariableId};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram, TemporalDag};
use antecedent_identify::sid::{
    scenarios::ScenarioCoordinate,
    temporal_sequence::{
        TEMPORAL_BOUNDS_EXCEEDED, TEMPORAL_HORIZON_DETAIL, TEMPORAL_INVALID_SEQUENCE,
        TEMPORAL_INVALID_SPEC, TemplateRoles, TemporalRefusal, TemporalSequenceSpec, TemporalSlots,
        unroll_two_slice,
    },
};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn coordinates(n: u32) -> Vec<ScenarioCoordinate> {
    (0..n)
        .map(|i| ScenarioCoordinate {
            variable: v(i),
            name: Arc::from(format!("v{i}")),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

/// `b -> l1 -> a1 -> l2 -> a2 -> y`, ids 0..5 in time order.
fn chain() -> (TemporalSlots, Admg) {
    let mut graph = Admg::with_variables(6);
    for i in 0..5 {
        graph.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    let slots = TemporalSlots {
        baseline: vec![v(0)],
        covariates: [vec![v(1)], vec![v(3)]],
        actions: [v(2), v(4)],
        outcome: v(5),
    };
    (slots, graph)
}

fn spec(horizon: usize, selections: &[u32]) -> Result<TemporalSequenceSpec, TemporalRefusal> {
    let (slots, graph) = chain();
    let diagram = SelectionDiagram::try_new(
        graph,
        Arc::<[VariableId]>::from(selections.iter().copied().map(v).collect::<Vec<_>>()),
    )
    .unwrap();
    TemporalSequenceSpec::try_new(horizon, slots, diagram, coordinates(6))
}

fn detail(result: Result<TemporalSequenceSpec, TemporalRefusal>) -> &'static str {
    result.expect_err("expected a refusal").detail
}

#[test]
fn the_specification_freezes_horizon_two_and_its_time_order() {
    let ok = spec(2, &[3]).unwrap();
    assert_eq!(ok.selection_slices(), vec![(v(3), 2)]);
    assert_eq!(detail(spec(3, &[])), TEMPORAL_HORIZON_DETAIL);
    assert_eq!(detail(spec(1, &[])), TEMPORAL_HORIZON_DETAIL);
    // An action's mechanism is replaced by the sequence.
    assert_eq!(detail(spec(2, &[2])), TEMPORAL_INVALID_SPEC);
    // A backward edge is not an unrolling.
    let (slots, mut graph) = chain();
    graph.insert_directed(DenseNodeId::from_raw(5), DenseNodeId::from_raw(1)).unwrap_err();
    let mut backward = Admg::with_variables(6);
    backward.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(backward, Arc::<[VariableId]>::from([])).unwrap();
    let refused = TemporalSequenceSpec::try_new(2, slots, diagram, coordinates(6));
    assert_eq!(detail(refused), TEMPORAL_INVALID_SPEC);
}

#[test]
fn sequences_are_one_action_per_step_over_the_alphabet() {
    let s = spec(2, &[]).unwrap();
    let seq = |a: &[f64]| a.iter().map(|x| Value::f64(*x)).collect::<Vec<_>>();
    assert!(s.check_sequence(&seq(&[0.0, 1.0])).is_ok());
    assert_eq!(s.check_sequence(&seq(&[0.0])).unwrap_err().detail, TEMPORAL_INVALID_SEQUENCE);
    assert_eq!(s.check_sequence(&seq(&[0.0, 2.0])).unwrap_err().detail, TEMPORAL_INVALID_SEQUENCE);
    // A third period is a new period, not a longer sequence.
    assert_eq!(
        s.check_sequence(&seq(&[0.0, 1.0, 1.0])).unwrap_err().detail,
        TEMPORAL_HORIZON_DETAIL
    );
}

#[test]
fn caps_on_actions_coordinates_and_histories_refuse_as_bounds_exceeded() {
    let (slots, graph) = chain();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let mut wide = coordinates(6);
    for a in [2usize, 4] {
        wide[a].domain = VariableDomain::Categorical { cardinality: 9 };
    }
    let refused = TemporalSequenceSpec::try_new(2, slots.clone(), diagram.clone(), wide);
    assert_eq!(detail(refused), TEMPORAL_BOUNDS_EXCEEDED);
    let mut eight = coordinates(6);
    for a in [2usize, 4] {
        eight[a].domain = VariableDomain::Categorical { cardinality: 8 };
    }
    assert!(TemporalSequenceSpec::try_new(2, slots.clone(), diagram.clone(), eight).is_ok());
    // Three covariates of 20 levels give 8000 complete histories.
    let mut large = coordinates(6);
    for c in [0usize, 1, 3] {
        large[c].domain = VariableDomain::Categorical { cardinality: 20 };
    }
    assert_eq!(
        detail(TemporalSequenceSpec::try_new(2, slots, diagram, large)),
        TEMPORAL_BOUNDS_EXCEEDED
    );
    let many = Admg::with_variables(13);
    let slots = TemporalSlots {
        baseline: (0..8).map(v).collect(),
        covariates: [vec![v(8)], vec![v(9)]],
        actions: [v(10), v(11)],
        outcome: v(12),
    };
    let diagram = SelectionDiagram::try_new(many, Arc::<[VariableId]>::from([])).unwrap();
    assert_eq!(
        detail(TemporalSequenceSpec::try_new(2, slots, diagram, coordinates(13))),
        TEMPORAL_BOUNDS_EXCEEDED
    );
}

#[test]
fn a_template_unrolls_through_the_temporal_dag_unfolding() {
    use antecedent_core::Lag;
    // Template variables: L = 0, A = 1, Y = 2 with
    // L[t-1] -> L[t], A[t-1] -> L[t], L[t] -> A[t], L[t] -> Y[t], A[t] -> Y[t].
    let mut template = TemporalDag::empty();
    let l0 = template.add_lagged(v(0), Lag::CONTEMPORANEOUS).unwrap();
    let l1 = template.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let a0 = template.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    let a1 = template.add_lagged(v(1), Lag::from_raw(1)).unwrap();
    let y0 = template.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(l1, l0), (a1, l0), (l0, a0), (l0, y0), (a0, y0)] {
        template.insert_directed(from, to).unwrap();
    }
    let roles = TemplateRoles { covariates: vec![v(0)], action: v(1), outcome: v(2) };
    let unrolled = unroll_two_slice(&template, &roles, 1, &[(0, v(0)), (0, v(2))]).unwrap();
    let slots = &unrolled.slots;
    // L@1 = 0, A@1 = 1, Y@1 = 2, L@2 = 3, A@2 = 4, Y@2 = 5, baseline = 6.
    assert_eq!(slots.baseline, vec![v(6)]);
    assert_eq!(slots.covariates, [vec![v(0)], vec![v(3), v(2)]]);
    assert_eq!((slots.actions, slots.outcome), ([v(1), v(4)], v(5)));
    let graph = &unrolled.admg;
    let has = |a: u32, b: u32| {
        graph.children(DenseNodeId::from_raw(a)).contains(&DenseNodeId::from_raw(b))
    };
    // The first slice has no lagged parents; the second slice inherits them.
    assert!(
        has(0, 1) && has(1, 2) && has(0, 2) && has(0, 3) && has(1, 3) && has(3, 4) && has(4, 5)
    );
    assert!(!has(3, 0) && has(6, 0) && has(6, 3) && has(6, 2) && has(6, 5));
    assert_eq!(unrolled.coordinate(v(1), 2), v(4));
    let diagram =
        SelectionDiagram::try_new(unrolled.admg.clone(), Arc::<[VariableId]>::from([v(3)]))
            .unwrap();
    assert!(TemporalSequenceSpec::try_new(2, unrolled.slots, diagram, coordinates(7)).is_ok());
}

#[test]
fn the_caps_admit_exactly_the_bound_and_refuse_one_more() {
    // Coordinates: twelve admitted, thirteen refused.
    let twelve = TemporalSlots {
        baseline: (0..7).map(v).collect(),
        covariates: [vec![v(7)], vec![v(8)]],
        actions: [v(9), v(10)],
        outcome: v(11),
    };
    let build = |n: u32, slots: TemporalSlots| {
        let diagram = SelectionDiagram::try_new(
            Admg::with_variables(n),
            Arc::<[VariableId]>::from(Vec::<VariableId>::new()),
        )
        .unwrap();
        TemporalSequenceSpec::try_new(2, slots, diagram, coordinates(n))
    };
    assert!(build(12, twelve).is_ok());
    let thirteen = TemporalSlots {
        baseline: (0..8).map(v).collect(),
        covariates: [vec![v(8)], vec![v(9)]],
        actions: [v(10), v(11)],
        outcome: v(12),
    };
    assert_eq!(detail(build(13, thirteen)), TEMPORAL_BOUNDS_EXCEEDED);
    // Complete histories: 16 * 16 * 16 = 4096 is admitted; 17 * 241 = 4097 is one more.
    let (slots, graph) = chain();
    let with_levels = |baseline: u32, first: u32, second: u32| {
        let mut declared = coordinates(6);
        for (variable, cardinality) in [(0usize, baseline), (1, first), (3, second)] {
            declared[variable].domain = VariableDomain::Categorical { cardinality };
        }
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([])).unwrap();
        TemporalSequenceSpec::try_new(2, slots.clone(), diagram, declared)
    };
    assert!(with_levels(16, 16, 16).is_ok());
    assert_eq!(detail(with_levels(17, 241, 1)), TEMPORAL_BOUNDS_EXCEEDED);
}

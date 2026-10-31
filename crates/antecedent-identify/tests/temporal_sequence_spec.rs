//! The explicit two-slice unrolling of one finite temporal transport sequence
//! its frozen horizon, time order, alphabet and hard caps, and its
//! construction from a lagged template through the ADR 0021 unfolding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{Value, VariableDomain, VariableId};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram, TemporalDag};
use antecedent_identify::sid::{
    scenarios::ScenarioCoordinate,
    temporal_sequence::{
        TEMPORAL_HORIZON_DETAIL, TEMPORAL_INVALID_SEQUENCE, TEMPORAL_INVALID_SPEC, TemplateRoles,
        TemporalRefusal, TemporalSequenceSpec, TemporalSlots, unroll_two_slice,
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
        covariates: vec![vec![v(1)], vec![v(3)]],
        actions: vec![v(2), v(4)],
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
fn the_specification_binds_its_horizon_to_the_slots_and_its_time_order() {
    let ok = spec(2, &[3]).unwrap();
    assert_eq!(ok.selection_slices(), vec![(v(3), 2)]);
    // A horizon the slots do not carry is refused, whatever its size.
    assert_eq!(detail(spec(3, &[])), TEMPORAL_HORIZON_DETAIL);
    assert_eq!(detail(spec(1, &[])), TEMPORAL_HORIZON_DETAIL);
    assert_eq!(detail(spec(0, &[])), TEMPORAL_HORIZON_DETAIL);
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

/// `s0 -> a1 -> l2 -> a2 -> l3 -> a3 -> y`, ids 0..7 in time order.
fn three_step_chain() -> (TemporalSlots, Admg) {
    let mut graph = Admg::with_variables(7);
    for i in 0..6 {
        graph.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    let slots = TemporalSlots {
        baseline: vec![v(0)],
        covariates: vec![vec![], vec![v(2)], vec![v(4)]],
        actions: vec![v(1), v(3), v(5)],
        outcome: v(6),
    };
    (slots, graph)
}

#[test]
fn a_longer_horizon_is_a_specification_with_one_action_per_step() {
    let (slots, graph) = three_step_chain();
    assert_eq!(slots.horizon(), 3);
    assert_eq!(slots.position(v(0)), Some(0));
    assert_eq!(slots.position(v(1)), Some(2));
    assert_eq!(slots.position(v(2)), Some(3));
    assert_eq!(slots.position(v(5)), Some(6));
    assert_eq!(slots.position(v(6)), Some(7));
    assert_eq!(slots.slice(v(0)), Some(0));
    assert_eq!(slots.slice(v(1)), Some(1));
    assert_eq!(slots.slice(v(2)), Some(2));
    assert_eq!(slots.slice(v(6)), Some(3));
    assert_eq!(slots.variables(), (0..7).map(v).collect::<Vec<_>>());
    assert_eq!(slots.initial_state(), vec![v(0)]);
    assert_eq!(slots.history_coordinates(), vec![v(0), v(2), v(4)]);
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([v(0)])).unwrap();
    let spec = TemporalSequenceSpec::try_new(3, slots, diagram, coordinates(7)).unwrap();
    assert_eq!(spec.horizon(), 3);
    let seq = |a: &[f64]| a.iter().map(|x| Value::f64(*x)).collect::<Vec<_>>();
    assert!(spec.check_sequence(&seq(&[0.0, 1.0, 1.0])).is_ok());
    assert_eq!(
        spec.check_sequence(&seq(&[0.0, 1.0])).unwrap_err().detail,
        TEMPORAL_INVALID_SEQUENCE
    );
    assert_eq!(
        spec.check_sequence(&seq(&[0.0, 1.0, 1.0, 0.0])).unwrap_err().detail,
        TEMPORAL_HORIZON_DETAIL
    );
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
fn actions_and_histories_have_no_count_ceiling() {
    // Acceptance companion: these declarations are valid at any cardinality (the history budget,
    // not declaration, is the limit), so success is the whole proposition; refusals are pinned
    // by the neighbouring invalid-declaration tests.
    let (slots, graph) = chain();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let mut wide = coordinates(6);
    for a in [2usize, 4] {
        wide[a].domain = VariableDomain::Categorical { cardinality: 9 };
    }
    assert!(TemporalSequenceSpec::try_new(2, slots.clone(), diagram.clone(), wide).is_ok());
    let mut eight = coordinates(6);
    for a in [2usize, 4] {
        eight[a].domain = VariableDomain::Categorical { cardinality: 8 };
    }
    assert!(TemporalSequenceSpec::try_new(2, slots.clone(), diagram.clone(), eight).is_ok());
    // Three covariates of 20 levels give 8000 complete histories: each is charged to the
    // decision's budget when the lattice grows, not refused at declaration.
    let mut large = coordinates(6);
    for c in [0usize, 1, 3] {
        large[c].domain = VariableDomain::Categorical { cardinality: 20 };
    }
    assert!(TemporalSequenceSpec::try_new(2, slots, diagram, large).is_ok());
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
    assert_eq!(slots.actions, [v(1), v(4)]);
    assert_eq!(slots.outcome, v(5));
    let graph = &unrolled.admg;
    let has = |a: u32, b: u32| {
        graph.children(DenseNodeId::from_raw(a)).contains(&DenseNodeId::from_raw(b))
    };
    // The first slice has no lagged parents; the second slice inherits them.
    assert!(
        has(0, 1) && has(1, 2) && has(0, 2) && has(0, 3) && has(1, 3) && has(3, 4) && has(4, 5)
    );
    assert!(!has(3, 0) && has(6, 0) && has(6, 3) && has(6, 2) && has(6, 5));
    assert_eq!(unrolled.coordinate(v(1), 2), Some(v(4)));
    assert_eq!(unrolled.coordinate(v(1), 1), Some(v(1)));
    assert_eq!(unrolled.coordinate(v(0), 0), Some(v(6)));
    let diagram =
        SelectionDiagram::try_new(unrolled.admg.clone(), Arc::<[VariableId]>::from([v(3)]))
            .unwrap();
    assert!(TemporalSequenceSpec::try_new(2, unrolled.slots, diagram, coordinates(7)).is_ok());
}

#[test]
fn the_history_count_has_no_ceiling() {
    // Acceptance companion: establishes that neither coordinate count nor complete-history count
    // is capped at declaration; invalid declarations are refused in the tests around it.
    // The number of coordinates is not capped: the old limit of twelve admits thirteen.
    let twelve = TemporalSlots {
        baseline: (0..7).map(v).collect(),
        covariates: vec![vec![v(7)], vec![v(8)]],
        actions: vec![v(9), v(10)],
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
        covariates: vec![vec![v(8)], vec![v(9)]],
        actions: vec![v(10), v(11)],
        outcome: v(12),
    };
    assert!(build(13, thirteen).is_ok());
    // Complete histories: 4096 and 4097 are both admitted; each is charged to the budget.
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
    assert!(with_levels(17, 241, 1).is_ok());
}

#[test]
fn unrolled_coordinate_rejects_out_of_range_periods() {
    use antecedent_core::Lag;
    let mut template = TemporalDag::empty();
    let l0 = template.add_lagged(v(0), Lag::CONTEMPORANEOUS).unwrap();
    let a0 = template.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = template.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
    template.insert_directed(l0, a0).unwrap();
    template.insert_directed(a0, y0).unwrap();
    let roles = TemplateRoles { covariates: vec![v(0)], action: v(1), outcome: v(2) };
    let unrolled = unroll_two_slice(&template, &roles, 1, &[(0, v(0))]).unwrap();
    // Three template variables: period 3 would collide with the baseline coordinate.
    assert_eq!(unrolled.coordinate(v(1), 3), None);
    assert_eq!(unrolled.coordinate(v(1), usize::MAX), None);
    assert_eq!(unrolled.coordinate(v(1), 1usize << 32), None);
    assert_eq!(unrolled.coordinate(v(3), 1), None);
    assert_eq!(unrolled.coordinate(v(u32::MAX), 2), None);
    // Period 0 is the baseline coordinate, never a period-1 coordinate.
    assert_eq!(unrolled.coordinate(v(0), 0), Some(v(6)));
    assert_eq!(unrolled.coordinate(v(1), 0), None);
}

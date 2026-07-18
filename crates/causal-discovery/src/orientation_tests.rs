use causal_core::{Lag, VariableId};
use causal_graph::TemporalCpdag;

use super::*;

#[test]
fn meek_r1_cycle_records_conflict_and_continues() {
    // a → b — c with c → d → a so c reaches b; orienting b→c would cycle.
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, a).unwrap();
    let mut state = OrientationState::default();
    let rules: [&dyn OrientationRule; 1] = [&MeekR1];
    let delta = run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert!(state.conflicts >= 1, "conflicts={}", state.conflicts);
    assert!(delta.conflicts >= 1);
    assert!(
        g.edge_between(b, c).unwrap().is_conflict(),
        "cycle conflict should mark x-x"
    );
}

#[test]
fn collider_opposite_direction_records_conflict() {
    // Pre-orient c→a; sepset says collider a→c←b → conflict on a—c.
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(c, a).unwrap();
    g.insert_undirected(c, b).unwrap();
    let mut state = OrientationState::default();
    state.set_sepset(a, b, Arc::from([]));
    let mut queue = OrientationQueue::new();
    let d = OrientationRule::apply(&OrientCollider, &mut g, &mut state, &mut queue).unwrap();
    assert!(state.conflicts >= 1 || d.conflicts >= 1);
    assert!(
        g.edge_between(c, a).unwrap().is_conflict(),
        "opposite-direction conflict should mark x-x"
    );
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
}

#[test]
fn meek_r1_orients_chain() {
    // a → b — c ⇒ b → c
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    queue.push(b);
    let d = OrientationRule::apply(&MeekR1, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed >= 1);
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
    assert!(d.enqueued > 0);
    assert!(d.enqueued < 20); // local, not full-graph blowup
}

#[test]
fn collider_with_sepset() {
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_undirected(a, c).unwrap();
    g.insert_undirected(c, b).unwrap();
    let mut state = OrientationState::default();
    // Sep(a,b) empty ⇒ c not in sepset ⇒ collider
    state.set_sepset(a, b, Arc::from([]));
    let mut queue = OrientationQueue::new();
    let d = OrientationRule::apply(&OrientCollider, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed >= 2);
    assert_eq!(g.edge_between(a, c).unwrap().parent_child(), Some((a, c)));
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
}

#[test]
fn collider_fires_on_triple_with_directed_lagged_leg() {
    // X@1 → K (auto-oriented by time), K — J undirected, X@1 ⟂ J with sepset ∅
    // excluding K ⇒ collider at K: orient J → K. Meek R1 must not run first and
    // orient K → J.
    let mut g = TemporalCpdag::empty();
    let x1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let k = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let j = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, k).unwrap();
    g.insert_undirected(k, j).unwrap();
    let mut state = OrientationState::default();
    state.set_sepset(x1, j, Arc::from([]));
    let rules: [&dyn OrientationRule; 5] =
        [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
    run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert_eq!(g.edge_between(j, k).unwrap().parent_child(), Some((j, k)));
}

#[test]
fn meek_r3_orients_diagonal() {
    // a—b with a—c→b and a—d→b, c not adj d ⇒ a→b
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_undirected(a, b).unwrap();
    g.insert_undirected(a, c).unwrap();
    g.insert_undirected(a, d).unwrap();
    g.insert_directed(c, b).unwrap();
    g.insert_directed(d, b).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let delta = OrientationRule::apply(&MeekR3, &mut g, &mut state, &mut queue).unwrap();
    assert!(delta.edges_changed >= 1);
    assert_eq!(g.edge_between(a, b).unwrap().parent_child(), Some((a, b)));
}

#[test]
fn static_meek_r1_orients_chain() {
    let mut g = Cpdag::with_variables(3);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    let rules: [&dyn StaticOrientationRule; 1] = [&MeekR1];
    let d = run_static_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert!(d.edges_changed >= 1);
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
}

#[test]
fn contemp_meek_r1_skips_lagged_undirected() {
    // Lagged undirected should not be oriented by ContempMeekR1.
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(a, b).unwrap();
    // Force an undirected between b and lagged c (unusual but guards the gate).
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    queue.push(b);
    let d = ContempMeekR1.apply(&mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    assert!(g.edge_between(b, c).unwrap().is_undirected());
}

#[test]
fn contemp_meek_r1_orients_contemporaneous_chain() {
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    queue.push(b);
    let d = ContempMeekR1.apply(&mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed >= 1);
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
}

#[test]
fn fixed_point_runner() {
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    let rules: [&dyn OrientationRule; 4] = [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
    let d = run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert!(d.fixed_point);
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
}

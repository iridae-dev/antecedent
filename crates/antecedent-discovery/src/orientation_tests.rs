use antecedent_core::{Lag, VariableId};
use antecedent_graph::{DenseNodeId, TemporalCpdag};

use super::*;

#[test]
fn context_nodes_share_the_contemporaneous_slice() {
    let mut g = TemporalCpdag::empty();
    let lagged = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let context = g.add_context(VariableId::from_raw(2), None).unwrap();
    assert!(!is_contemporaneous_node(&g, lagged));
    assert!(is_contemporaneous_node(&g, now));
    assert!(is_contemporaneous_node(&g, context));
}

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
    let rules: [&dyn OrientationRule<antecedent_graph::TemporalCpdag>; 1] = [&MeekR1];
    let delta = run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert!(state.conflicts >= 1, "conflicts={}", state.conflicts);
    assert!(delta.conflicts >= 1);
    assert!(g.edge_between(b, c).unwrap().is_conflict(), "cycle conflict should mark x-x");
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
    let d = OrientationRule::<antecedent_graph::TemporalCpdag>::apply(
        &OrientCollider,
        &mut g,
        &mut state,
        &mut queue,
    )
    .unwrap();
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
    let d = OrientationRule::<antecedent_graph::TemporalCpdag>::apply(
        &MeekR1, &mut g, &mut state, &mut queue,
    )
    .unwrap();
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
    let d = OrientationRule::<antecedent_graph::TemporalCpdag>::apply(
        &OrientCollider,
        &mut g,
        &mut state,
        &mut queue,
    )
    .unwrap();
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
    let rules: [&dyn OrientationRule<antecedent_graph::TemporalCpdag>; 5] =
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
    let delta = OrientationRule::<antecedent_graph::TemporalCpdag>::apply(
        &MeekR3, &mut g, &mut state, &mut queue,
    )
    .unwrap();
    assert!(delta.edges_changed >= 1);
    assert_eq!(g.edge_between(a, b).unwrap().parent_child(), Some((a, b)));
}

/// R3's premise is that `c — a — d` is a definite non-collider. When that triple is marked
/// ambiguous (collider status unknown), R3 must not orient `a → b`.
#[test]
fn meek_r3_and_contemp_r3_skip_ambiguous_triple() {
    for contemp in [false, true] {
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
        state.mark_ambiguous_triple(c, a, d);
        let mut queue = OrientationQueue::new();
        let delta = if contemp {
            ContempMeekR3.apply(&mut g, &mut state, &mut queue).unwrap()
        } else {
            OrientationRule::<antecedent_graph::TemporalCpdag>::apply(
                &MeekR3, &mut g, &mut state, &mut queue,
            )
            .unwrap()
        };
        assert_eq!(delta.edges_changed, 0, "contemp={contemp}");
        assert!(g.edge_between(a, b).unwrap().is_undirected(), "contemp={contemp}");
    }
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
    let rules: [&dyn OrientationRule<antecedent_graph::Cpdag>; 1] = [&MeekR1];
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
fn contemp_meek_r1_matches_meek_r1_on_a_contemporaneous_chain() {
    fn chain() -> (TemporalCpdag, DenseNodeId, DenseNodeId) {
        let mut g = TemporalCpdag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_undirected(b, c).unwrap();
        (g, b, c)
    }
    let (mut general, b, c) = chain();
    let (mut contemp, _, _) = chain();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    queue.push(b);
    let general_delta =
        OrientationRule::<TemporalCpdag>::apply(&MeekR1, &mut general, &mut state, &mut queue)
            .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    queue.push(b);
    let contemp_delta = ContempMeekR1.apply(&mut contemp, &mut state, &mut queue).unwrap();
    assert_eq!(
        general.edge_between(b, c).unwrap().parent_child(),
        contemp.edge_between(b, c).unwrap().parent_child()
    );
    assert_eq!(general_delta.edges_changed, contemp_delta.edges_changed);
    assert!(general_delta.premises.iter().any(|p| p.starts_with("meek.r1:")));
    assert!(contemp_delta.premises.iter().any(|p| p.starts_with("meek.r1.contemp:")));
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
fn contemp_meek_r1_skips_ambiguous_triple() {
    // Same a → b — c pattern as the definite non-collider case, but the triple is
    // marked ambiguous: R1 must not orient b → c.
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    state.mark_ambiguous_triple(a, b, c);
    let mut queue = OrientationQueue::new();
    queue.push(b);
    let d = ContempMeekR1.apply(&mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    assert!(g.edge_between(b, c).unwrap().is_undirected());
}

#[test]
fn meek_r1_skips_ambiguous_triple() {
    let mut g = Cpdag::with_variables(3);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    g.insert_directed(a, b).unwrap();
    g.insert_undirected(b, c).unwrap();
    let mut state = OrientationState::default();
    state.mark_ambiguous_triple(a, b, c);
    let rules: [&dyn OrientationRule<antecedent_graph::Cpdag>; 1] = [&MeekR1];
    let d = run_static_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert_eq!(d.edges_changed, 0);
    assert!(g.edge_between(b, c).unwrap().is_undirected());
}

#[test]
fn orient_collider_marks_missing_sepset_ambiguous() {
    // Unshielded a — c — b with no Sep(a,b): collider status unknown.
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_undirected(a, c).unwrap();
    g.insert_undirected(c, b).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let _ = OrientationRule::<antecedent_graph::TemporalCpdag>::apply(
        &OrientCollider,
        &mut g,
        &mut state,
        &mut queue,
    )
    .unwrap();
    assert!(state.is_ambiguous_triple(a, c, b));
    assert!(g.edge_between(a, c).unwrap().is_undirected());
    assert!(g.edge_between(c, b).unwrap().is_undirected());
}

#[test]
fn contemp_meek_r4_orients_discriminating_path() {
    // a — b, a — c → d → b, adj(a,d), not adj(c,b) → a → b (all contemporaneous).
    let mut g = TemporalCpdag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_undirected(a, b).unwrap();
    g.insert_undirected(a, c).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, b).unwrap();
    g.insert_undirected(a, d).unwrap(); // adj(a,d); Meek R4 only requires adjacency
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    queue.push(a);
    let delta = ContempMeekR4.apply(&mut g, &mut state, &mut queue).unwrap();
    assert!(delta.edges_changed >= 1);
    assert_eq!(g.edge_between(a, b).unwrap().parent_child(), Some((a, b)));
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
    let rules: [&dyn OrientationRule<antecedent_graph::TemporalCpdag>; 4] =
        [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
    let d = run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert!(d.fixed_point);
    assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
}

/// Rule *k* only sees what rule *k−1* enqueued, so a pass that changes nothing has not
/// necessarily shown rule 1 the whole graph. Here rule B enables rule A at node 0 while the last
/// rule leaves only node 5 queued: the next pass gives A the focus `{5}`, finds nothing, and a
/// scheduler that stopped on the first quiet pass would end with A's application at node 0
/// still pending. It must confirm with a full-scan pass and apply it.
#[test]
fn quiet_pass_does_not_end_scheduling_before_a_full_scan() {
    use std::cell::Cell;

    let n = 8usize;
    let node = DenseNodeId::from_raw;
    let enabled = Cell::new(false);
    let applied = Cell::new(false);
    let b_fired = Cell::new(false);
    let c_fired = Cell::new(false);
    let d = drive_to_fixed_point(n, |queue| {
        let mut pass = RuleDelta::default();
        // Rule A: acts at node 0 once enabled, but only if 0 is in its focus.
        let focus = focus_nodes(n, queue);
        if enabled.get() && !applied.get() && focus.contains(&node(0)) {
            applied.set(true);
            pass.edges_changed += 1;
        }
        // Rule B: enables A; the node it enqueues is drained by rule C.
        let _ = focus_nodes(n, queue);
        if !b_fired.get() {
            b_fired.set(true);
            enabled.set(true);
            pass.edges_changed += 1;
            queue.push(node(1));
        }
        // Rule C (last): fires once and leaves node 5 queued for the next pass.
        let _ = focus_nodes(n, queue);
        if !c_fired.get() {
            c_fired.set(true);
            pass.edges_changed += 1;
            queue.push(node(5));
        }
        Ok(pass)
    })
    .unwrap();
    assert!(applied.get(), "rule A's pending application was never examined");
    assert!(d.fixed_point);
}

#[test]
fn context_and_lag_zero_are_contemporaneous() {
    use antecedent_graph::NodeRef;
    let mut g = TemporalCpdag::empty();
    let now = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let ctx = g
        .add_node(NodeRef::Context { variable: VariableId::from_raw(1), environment: None })
        .unwrap();
    assert!(is_contemporaneous_node(&g, now));
    assert!(is_contemporaneous_node(&g, ctx));
    assert!(!is_contemporaneous_node(&g, past));
    assert!(crate::pcmci_plus::is_contemp_node(&g, now));
    assert!(crate::pcmci_plus::is_contemp_node(&g, ctx));
    assert!(!crate::pcmci_plus::is_contemp_node(&g, past));
}

/// Independent enumeration reference for the Meek rules. For every labelled DAG on four
/// nodes, the Markov equivalence class is enumerated directly (every orientation of the
/// skeleton that is acyclic and has the same v-structures), and an edge of the essential
/// graph is directed exactly when all members agree on it. The rules under test start from
/// the skeleton with only the v-structure legs oriented, as a constraint-based search leaves
/// it.
mod essential_graph_enumeration {
    use std::collections::BTreeSet;

    use super::*;

    const N: usize = 4;

    fn acyclic(arcs: &[(usize, usize)]) -> bool {
        let mut indegree = [0usize; N];
        for &(_, to) in arcs {
            indegree[to] += 1;
        }
        let mut ready: Vec<usize> = (0..N).filter(|&v| indegree[v] == 0).collect();
        let mut seen = 0;
        while let Some(v) = ready.pop() {
            seen += 1;
            for &(from, to) in arcs {
                if from == v {
                    indegree[to] -= 1;
                    if indegree[to] == 0 {
                        ready.push(to);
                    }
                }
            }
        }
        seen == N
    }

    fn v_structures(arcs: &[(usize, usize)]) -> BTreeSet<(usize, usize, usize)> {
        let adjacent = |x: usize, y: usize| arcs.contains(&(x, y)) || arcs.contains(&(y, x));
        let mut out = BTreeSet::new();
        for c in 0..N {
            for a in 0..N {
                for b in (a + 1)..N {
                    if a != c
                        && b != c
                        && arcs.contains(&(a, c))
                        && arcs.contains(&(b, c))
                        && !adjacent(a, b)
                    {
                        out.insert((a, c, b));
                    }
                }
            }
        }
        out
    }

    /// Number of (DAG, edge) disagreements between the rules' output and the essential graph.
    fn disagreements(rules: &[&dyn OrientationRule<TemporalCpdag>]) -> (usize, usize) {
        let pairs: Vec<(usize, usize)> =
            (0..N).flat_map(|i| ((i + 1)..N).map(move |j| (i, j))).collect();
        let (mut dags, mut wrong) = (0, 0);
        for code in 0..3usize.pow(u32::try_from(pairs.len()).unwrap()) {
            let (mut arcs, mut rest) = (Vec::new(), code);
            for &(i, j) in &pairs {
                match rest % 3 {
                    1 => arcs.push((i, j)),
                    2 => arcs.push((j, i)),
                    _ => {}
                }
                rest /= 3;
            }
            if !acyclic(&arcs) {
                continue;
            }
            dags += 1;
            let colliders = v_structures(&arcs);
            let skeleton: Vec<(usize, usize)> =
                arcs.iter().map(|&(p, q)| (p.min(q), p.max(q))).collect();
            let m = skeleton.len();
            let mut first: Vec<Option<bool>> = vec![None; m];
            let mut agree = vec![true; m];
            for bits in 0..(1usize << m) {
                let member: Vec<(usize, usize)> = skeleton
                    .iter()
                    .enumerate()
                    .map(|(k, &(a, b))| if (bits >> k) & 1 == 1 { (a, b) } else { (b, a) })
                    .collect();
                if !acyclic(&member) || v_structures(&member) != colliders {
                    continue;
                }
                for k in 0..m {
                    let up = (bits >> k) & 1 == 1;
                    match first[k] {
                        None => first[k] = Some(up),
                        Some(f) if f != up => agree[k] = false,
                        Some(_) => {}
                    }
                }
            }
            let mut g = TemporalCpdag::empty();
            let ids: Vec<DenseNodeId> = (0..N)
                .map(|i| {
                    g.add_lagged(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Lag::CONTEMPORANEOUS,
                    )
                    .unwrap()
                })
                .collect();
            let legs: BTreeSet<(usize, usize)> =
                colliders.iter().flat_map(|&(a, c, b)| [(a, c), (b, c)]).collect();
            for &(a, b) in &skeleton {
                if legs.contains(&(a, b)) {
                    g.insert_directed(ids[a], ids[b]).unwrap();
                } else if legs.contains(&(b, a)) {
                    g.insert_directed(ids[b], ids[a]).unwrap();
                } else {
                    g.insert_undirected(ids[a], ids[b]).unwrap();
                }
            }
            run_orientation_to_fixed_point(&mut g, rules, &mut OrientationState::default())
                .unwrap();
            for (k, &(a, b)) in skeleton.iter().enumerate() {
                let edge = g.edge_between(ids[a], ids[b]).unwrap();
                let correct = if agree[k] {
                    let (p, q) = if first[k] == Some(true) { (a, b) } else { (b, a) };
                    edge.parent_child() == Some((ids[p], ids[q]))
                } else {
                    edge.is_undirected()
                };
                wrong += usize::from(!correct);
            }
        }
        (dags, wrong)
    }

    #[test]
    fn rules_one_to_three_complete_every_four_node_pattern_to_its_essential_graph() {
        let rules: [&dyn OrientationRule<TemporalCpdag>; 3] = [&MeekR1, &MeekR2, &MeekR3];
        let (dags, wrong) = disagreements(&rules);
        assert_eq!(dags, 543, "labelled DAGs on four nodes");
        assert_eq!(wrong, 0, "Meek R1-R3 must reproduce every essential graph");
    }

    #[test]
    fn dropping_rule_three_leaves_an_orientation_undone() {
        // The control: the enumeration is able to detect an incomplete rule set.
        let rules: [&dyn OrientationRule<TemporalCpdag>; 2] = [&MeekR1, &MeekR2];
        assert!(disagreements(&rules).1 > 0);
        let rules: [&dyn OrientationRule<TemporalCpdag>; 1] = [&MeekR1];
        assert!(disagreements(&rules).1 > 0);
    }
}

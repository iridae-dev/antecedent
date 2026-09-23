use super::*;
use antecedent_core::{Lag, VariableId};
use antecedent_graph::Pag;

#[test]
fn r2_orients_circle_into_arrow() {
    // a → b o→ c and a o-o c ⇒ a o→ c
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_circle_arrow(b, c).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b: c,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR2, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r2_fires_on_fully_directed_chain() {
    // a → b → c and a *–o c (circle at c) ⇒ orient arrow at c.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_directed(b, c).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b: c,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR2, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r2_does_not_overwrite_tail_at_c() {
    // a → b → c and a → c already (tail at c would be illegal for R2 premise;
    // use a *– Tail at c to ensure we refuse to overwrite).
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_directed(b, c).unwrap();
    // Circle at a, Tail at c on a–c — R2 must not turn the Tail into an Arrow.
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b: c,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Tail,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR2, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    let (_, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_c, Endpoint::Tail));
}

#[test]
fn r1_orients_from_circle_arrow_premise() {
    // a o→ b o–o c, a ≁ c ⇒ b → c (both marks).
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, b).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: b,
        b: c,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR1, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_b, at_c) = marks_between(&g, b, c).unwrap();
    assert!(matches!(at_b, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r8_orients_triangle() {
    // a → b → c and a o→ c ⇒ a → c
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_directed(b, c).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR8, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r8_does_not_orient_with_circle_arrow_middle() {
    // a → b o→ c and a o→ c must NOT orient a → c (Zhang R8 needs definite b → c).
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_circle_arrow(b, c).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR8, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r3_does_not_orient_with_circles_only_at_a_and_c() {
    // Circles at a and c (not at θ=d): must not orient d *→ b.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: d,
        b: a,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: d,
        b: c,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: d,
        b,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let delta = LpcmciOrientationRule::apply(&LpcmciR3, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(delta.edges_changed, 0);
}

#[test]
fn r3_orients_with_circles_at_theta() {
    // Collider a *→ b ←* c, a ≁ c; circles at θ=d on d—a, d—c; circle at b on d—b ⇒ d *→ b.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b: d,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: c,
        b: d,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: d,
        b,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let delta = LpcmciOrientationRule::apply(&LpcmciR3, &mut g, &mut state, &mut queue).unwrap();
    assert!(delta.edges_changed > 0);
    let (at_d, at_b) = marks_between(&g, d, b).unwrap();
    assert!(matches!(at_b, Endpoint::Arrow));
    assert!(matches!(at_d, Endpoint::Tail));
}

#[test]
fn rule_ids_cover_r1_r2_r3() {
    assert_eq!(LpcmciOrientationRule::id(&LpcmciR1), "lpcmci.r1");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciR2), "lpcmci.r2");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciR3), "lpcmci.r3");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciR8), "lpcmci.r8");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciR9), "lpcmci.r9");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciR10), "lpcmci.r10");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciApr), "lpcmci.apr");
    assert_eq!(LpcmciOrientationRule::id(&LpcmciMmr), "lpcmci.mmr");
    assert_eq!(FciOrientationRule::id(&LpcmciR1), "fci.r1");
    assert_eq!(FciOrientationRule::id(&LpcmciR2), "fci.r2");
    assert_eq!(FciOrientationRule::id(&LpcmciDiscriminatingPathRule), "fci.discriminating_path");
}

#[test]
fn scheduler_honors_delta_queue_without_reseed() {
    // a → b o→ c and a o-o c ⇒ R2 orients a o→ c; subsequent rounds must not
    // require a full-graph re-seed to finish.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, b).unwrap();
    g.insert_circle_arrow(b, c).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b: c,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let rules: [&dyn LpcmciOrientationRule; 1] = [&LpcmciR2];
    let d = run_lpcmci_orientation(&mut g, &rules, &mut state).unwrap();
    assert!(d.edges_changed > 0);
    assert!(d.fixed_point);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r10_does_not_orient_with_single_uncovered_pd_path() {
    // a o→ c; one uncovered PD path a → d1 → p1 → c into parent p1 — insufficient for R10′.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let d1 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let p1 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    g.insert_directed(a, d1).unwrap();
    g.insert_directed(d1, p1).unwrap();
    g.insert_directed(p1, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR10, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r10_orients_with_two_disjoint_uncovered_pd_paths() {
    // 7-node oracle MAG: a o→ c; paths a → d1 → m1 → p1 → c and a → d2 → p2 → c
    // into definite parents; first nodes d1, d2 distinct and non-adjacent ⇒ a → c.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let d1 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let m1 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d2 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let p1 = g.add_lagged(VariableId::from_raw(4), Lag::CONTEMPORANEOUS).unwrap();
    let p2 = g.add_lagged(VariableId::from_raw(5), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(6), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    g.insert_directed(a, d1).unwrap();
    g.insert_directed(d1, m1).unwrap();
    g.insert_directed(m1, p1).unwrap();
    g.insert_directed(p1, c).unwrap();
    g.insert_directed(a, d2).unwrap();
    g.insert_directed(d2, p2).unwrap();
    g.insert_directed(p2, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR10, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r10_does_not_orient_when_first_nodes_adjacent() {
    // Same skeleton as the positive 7-node MAG, but d1—d2 adjacent: Zhang R10 fails.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let d1 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let m1 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d2 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let p1 = g.add_lagged(VariableId::from_raw(4), Lag::CONTEMPORANEOUS).unwrap();
    let p2 = g.add_lagged(VariableId::from_raw(5), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(6), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    g.insert_directed(a, d1).unwrap();
    g.insert_directed(d1, m1).unwrap();
    g.insert_directed(m1, p1).unwrap();
    g.insert_directed(p1, c).unwrap();
    g.insert_directed(a, d2).unwrap();
    g.insert_directed(d2, p2).unwrap();
    g.insert_directed(p2, c).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: d1,
        b: d2,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR10, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn r10_does_not_orient_with_circle_arrow_parent() {
    // 7-node MAG where one "parent" is only o→ c: not a definite parent ⇒ no R10.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let d1 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let m1 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let d2 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let p1 = g.add_lagged(VariableId::from_raw(4), Lag::CONTEMPORANEOUS).unwrap();
    let p2 = g.add_lagged(VariableId::from_raw(5), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(6), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    g.insert_directed(a, d1).unwrap();
    g.insert_directed(d1, m1).unwrap();
    g.insert_directed(m1, p1).unwrap();
    g.insert_directed(p1, c).unwrap();
    g.insert_directed(a, d2).unwrap();
    g.insert_directed(d2, p2).unwrap();
    g.insert_circle_arrow(p2, c).unwrap(); // o→ parent — not definite
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = LpcmciOrientationRule::apply(&LpcmciR10, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.edges_changed, 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn discriminating_r4_orients_collider_when_c_not_in_sep_ab() {
    // Zhang path ⟨a, d, c, b⟩: a → d ← c, d → b, c o→ b; c ∉ Sep(a,b) ⇒ d *→ c ←* b.
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, d).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();

    let mut state = OrientationState::default();
    state.set_sepset(a, b, std::sync::Arc::from([])); // c ∉ Sep(a,b)
    let mut queue = OrientationQueue::new();
    let delta =
        LpcmciOrientationRule::apply(&LpcmciDiscriminatingPathRule, &mut g, &mut state, &mut queue)
            .unwrap();
    assert!(delta.edges_changed > 0);

    let (at_c_cb, at_b) = marks_between(&g, c, b).unwrap();
    assert!(matches!(at_c_cb, Endpoint::Arrow), "arrow into c from b");
    assert!(matches!(at_b, Endpoint::Arrow));

    let (at_c_cd, at_d) = marks_between(&g, c, d).unwrap();
    assert!(matches!(at_c_cd, Endpoint::Arrow), "arrow into c from d");
    assert!(matches!(at_d, Endpoint::Arrow), "keep arrow at d");
}

#[test]
fn discriminating_r4_orients_noncollider_when_c_in_sep_ab() {
    let mut g = TemporalPag::empty();
    let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let d = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let b = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(a, d).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();

    let mut state = OrientationState::default();
    state.set_sepset(a, b, std::sync::Arc::from([c])); // c ∈ Sep(a,b)
    let mut queue = OrientationQueue::new();
    let delta =
        LpcmciOrientationRule::apply(&LpcmciDiscriminatingPathRule, &mut g, &mut state, &mut queue)
            .unwrap();
    assert!(delta.edges_changed > 0);

    let (at_c, at_b) = marks_between(&g, c, b).unwrap();
    assert!(matches!(at_c, Endpoint::Tail));
    assert!(matches!(at_b, Endpoint::Arrow));
}

// --- Static Pag surface ---

#[test]
fn static_r1_orients_from_circle_arrow_premise() {
    let mut g = Pag::with_variables(3);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    g.insert_circle_arrow(a, b).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: b,
        b: c,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = FciOrientationRule::apply(&LpcmciR1, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_b, at_c) = marks_between(&g, b, c).unwrap();
    assert!(matches!(at_b, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn static_r2_orients_circle_into_arrow() {
    let mut g = Pag::with_variables(3);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    g.insert_directed(a, b).unwrap();
    g.insert_circle_arrow(b, c).unwrap();
    g.insert_circle_circle(a, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = FciOrientationRule::apply(&LpcmciR2, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn static_r8_orients_triangle() {
    let mut g = Pag::with_variables(3);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    g.insert_directed(a, b).unwrap();
    g.insert_directed(b, c).unwrap();
    g.insert_circle_arrow(a, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = FciOrientationRule::apply(&LpcmciR8, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn static_r3_orients_with_circles_at_theta() {
    let mut g = Pag::with_variables(4);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    let d = DenseNodeId::from_raw(3);
    g.insert_circle_arrow(a, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b: d,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: c,
        b: d,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_marked(antecedent_graph::MarkedEdge {
        a: d,
        b,
        at_a: Endpoint::Tail,
        at_b: Endpoint::Circle,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let delta = FciOrientationRule::apply(&LpcmciR3, &mut g, &mut state, &mut queue).unwrap();
    assert!(delta.edges_changed > 0);
    let (at_d, at_b) = marks_between(&g, d, b).unwrap();
    assert!(matches!(at_b, Endpoint::Arrow));
    assert!(matches!(at_d, Endpoint::Tail));
}

#[test]
fn static_discriminating_r4_orients_collider_when_c_not_in_sep_ab() {
    let mut g = Pag::with_variables(4);
    let a = DenseNodeId::from_raw(0);
    let d = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    let b = DenseNodeId::from_raw(3);
    g.insert_directed(a, d).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();

    let mut state = OrientationState::default();
    state.set_sepset(a, b, std::sync::Arc::from([]));
    let mut queue = OrientationQueue::new();
    let delta =
        FciOrientationRule::apply(&LpcmciDiscriminatingPathRule, &mut g, &mut state, &mut queue)
            .unwrap();
    assert!(delta.edges_changed > 0);

    let (at_c_cb, at_b) = marks_between(&g, c, b).unwrap();
    assert!(matches!(at_c_cb, Endpoint::Arrow));
    assert!(matches!(at_b, Endpoint::Arrow));

    let (at_c_cd, at_d) = marks_between(&g, c, d).unwrap();
    assert!(matches!(at_c_cd, Endpoint::Arrow));
    assert!(matches!(at_d, Endpoint::Arrow));
}

#[test]
fn static_r10_orients_with_two_disjoint_uncovered_pd_paths() {
    let mut g = Pag::with_variables(7);
    let a = DenseNodeId::from_raw(0);
    let d1 = DenseNodeId::from_raw(1);
    let m1 = DenseNodeId::from_raw(2);
    let d2 = DenseNodeId::from_raw(3);
    let p1 = DenseNodeId::from_raw(4);
    let p2 = DenseNodeId::from_raw(5);
    let c = DenseNodeId::from_raw(6);
    g.insert_circle_arrow(a, c).unwrap();
    g.insert_directed(a, d1).unwrap();
    g.insert_directed(d1, m1).unwrap();
    g.insert_directed(m1, p1).unwrap();
    g.insert_directed(p1, c).unwrap();
    g.insert_directed(a, d2).unwrap();
    g.insert_directed(d2, p2).unwrap();
    g.insert_directed(p2, c).unwrap();
    let mut state = OrientationState::default();
    let mut queue = OrientationQueue::new();
    let d = FciOrientationRule::apply(&LpcmciR10, &mut g, &mut state, &mut queue).unwrap();
    assert!(d.edges_changed > 0);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Tail));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn static_fci_scheduler_reaches_fixed_point() {
    let mut g = Pag::with_variables(3);
    let a = DenseNodeId::from_raw(0);
    let b = DenseNodeId::from_raw(1);
    let c = DenseNodeId::from_raw(2);
    g.insert_directed(a, b).unwrap();
    g.insert_circle_arrow(b, c).unwrap();
    g.insert_circle_circle(a, c).unwrap();
    let mut state = OrientationState::default();
    let rules: [&dyn FciOrientationRule; 1] = [&LpcmciR2];
    let d = run_fci_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
    assert!(d.edges_changed > 0);
    assert!(d.fixed_point);
    let (at_a, at_c) = marks_between(&g, a, c).unwrap();
    assert!(matches!(at_a, Endpoint::Circle));
    assert!(matches!(at_c, Endpoint::Arrow));
}

#[test]
fn collider_records_conflict_instead_of_overwriting_tail() {
    // a <- b (tail at b, e.g. from an earlier R1) and c o-o b with b not in Sep(a,c): the
    // collider a *-> b <-* c contradicts the tail at b, so that leg is a conflict while the
    // free leg still orients.
    let mut g = Pag::with_variables(3);
    let (a, b, c) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
    g.insert_marked(antecedent_graph::MarkedEdge {
        a,
        b,
        at_a: Endpoint::Arrow,
        at_b: Endpoint::Tail,
        middle: antecedent_graph::MiddleMark::Empty,
    })
    .unwrap();
    g.insert_circle_circle(c, b).unwrap();
    let mut state = OrientationState::default();
    state.set_sepset(a, c, std::sync::Arc::from([]));
    let mut queue = OrientationQueue::new();
    let d =
        FciOrientationRule::apply(&LpcmciOrientCollider, &mut g, &mut state, &mut queue).unwrap();
    assert_eq!(d.conflicts, 1);
    assert_eq!(state.conflicts, 1);
    let (m_a, m_b) = marks_between(&g, a, b).unwrap();
    assert!(matches!(m_a, Endpoint::Conflict) && matches!(m_b, Endpoint::Conflict));
    let (m_c, m_b2) = marks_between(&g, c, b).unwrap();
    assert!(matches!(m_c, Endpoint::Circle));
    assert!(matches!(m_b2, Endpoint::Arrow));
}

/// 70 independent discriminating paths, one per edge `c o→ b`: far past the old global cap of
/// 64 paths, but every edge's own search is tiny. The rule must orient all of them.
#[test]
fn discriminating_paths_are_budgeted_per_edge_not_globally() {
    let copies = 70u32;
    let mut g = Pag::with_variables(4 * copies);
    let mut state = OrientationState::default();
    let id = DenseNodeId::from_raw;
    for k in 0..copies {
        let (a, d, c, b) = (id(4 * k), id(4 * k + 1), id(4 * k + 2), id(4 * k + 3));
        g.insert_directed(a, d).unwrap();
        g.insert_directed(c, d).unwrap();
        g.insert_directed(d, b).unwrap();
        g.insert_circle_arrow(c, b).unwrap();
        state.set_sepset(a, b, std::sync::Arc::from([])); // c ∉ Sep(a,b)
    }
    let mut queue = OrientationQueue::new();
    let delta =
        FciOrientationRule::apply(&LpcmciDiscriminatingPathRule, &mut g, &mut state, &mut queue)
            .unwrap();
    assert_eq!(delta.edges_changed, copies);
    assert!(state.discriminating_skipped.is_empty());
    for k in 0..copies {
        let (c, b) = (id(4 * k + 2), id(4 * k + 3));
        let (at_c, _) = marks_between(&g, c, b).unwrap();
        assert!(matches!(at_c, Endpoint::Arrow), "copy {k} not oriented");
    }
}

/// An edge whose search budget is exhausted keeps its circle and is recorded: the run does not
/// fail, and no orientation is invented.
#[test]
fn exhausted_edge_budget_is_skipped_not_fatal() {
    let mut g = Pag::with_variables(4);
    let id = DenseNodeId::from_raw;
    let (a, d, c, b) = (id(0), id(1), id(2), id(3));
    g.insert_directed(a, d).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, b).unwrap();
    g.insert_circle_arrow(c, b).unwrap();
    let mut state = OrientationState::default();
    state.set_sepset(a, b, std::sync::Arc::from([]));
    // The path ⟨a, d, c, b⟩ has 4 nodes; a length cap of 3 cannot hold it.
    state.discriminating_budget =
        crate::discriminating_paths::DiscriminatingPathBudget { max_paths_per_edge: 8, max_len: 3 };
    let mut queue = OrientationQueue::new();
    let delta =
        FciOrientationRule::apply(&LpcmciDiscriminatingPathRule, &mut g, &mut state, &mut queue)
            .unwrap();
    assert_eq!(delta.edges_changed, 0);
    assert!(state.discriminating_skipped.contains(&(c.raw(), b.raw())));
    let (at_c, _) = marks_between(&g, c, b).unwrap();
    assert!(matches!(at_c, Endpoint::Circle));
}

/// Zhang R4 collider branch orients `d_k ↔ c ↔ b`: the arrowhead at `b` on `c *-* b` too, not
/// only the one at `c`. Here the edge starts `c o-o b`, so the head at `b` is new information.
#[test]
fn discriminating_collider_branch_orients_both_ends_of_cb() {
    let mut g = Pag::with_variables(4);
    let id = DenseNodeId::from_raw;
    let (a, d, c, b) = (id(0), id(1), id(2), id(3));
    g.insert_directed(a, d).unwrap();
    g.insert_directed(c, d).unwrap();
    g.insert_directed(d, b).unwrap();
    g.insert_circle_circle(c, b).unwrap();
    let mut state = OrientationState::default();
    state.set_sepset(a, b, std::sync::Arc::from([])); // c not in Sep(a,b)
    let mut queue = OrientationQueue::new();
    let delta =
        FciOrientationRule::apply(&LpcmciDiscriminatingPathRule, &mut g, &mut state, &mut queue)
            .unwrap();
    assert!(delta.edges_changed > 0);
    let (at_c, at_b) = marks_between(&g, c, b).unwrap();
    assert!(matches!(at_c, Endpoint::Arrow), "arrow at c");
    assert!(matches!(at_b, Endpoint::Arrow), "arrow at b");
}

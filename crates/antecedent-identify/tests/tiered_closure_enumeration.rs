//! The tier-closure identification restated from the back-door definition and swept over every
//! tier layout of up to five variables.
//!
//! A co-determined background asserts an arrow from every earlier-tier variable to every
//! later-tier variable and a bidirected clique inside each tier. The reference rebuilds that
//! graph itself, decides every m-connection by enumerating simple paths with the collider rule,
//! and applies the definition of a joint back-door set (no member of the set is a descendant of
//! a treatment or a treatment or the outcome, and each treatment is separated from the outcome
//! given the set and the other treatments once its outgoing arrows are removed). It shares no
//! code with the product's tier closure or its m-separation.
//!
//! Checked on every layout, treatment set and outcome in a later tier:
//!
//! * the single-treatment closure is a valid back-door set, which is what
//!   `identify_tiered` assumes without checking;
//! * the joint closure is accepted by `identify_tiered_joint` exactly when the reference accepts
//!   it, and every joint query it accepts is also identified by general ID.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, Intervention,
    InterventionalDistributionQuery, ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, TieredBackground, WithinTier};
use antecedent_identify::{
    IdIdentifier, IdentificationStatus, IdentificationWorkspace, identify_tiered,
    identify_tiered_joint,
};
use std::sync::Arc;

fn v(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

struct Layout {
    n: usize,
    tier: Vec<usize>,
}

impl Layout {
    fn child(&self, a: usize, b: usize) -> bool {
        self.tier[a] < self.tier[b]
    }
    fn bi(&self, a: usize, b: usize) -> bool {
        a != b && self.tier[a] == self.tier[b]
    }
    fn background(&self) -> TieredBackground {
        let tiers = (0..=*self.tier.iter().max().unwrap())
            .map(|k| (0..self.n).filter(|&i| self.tier[i] == k).map(v).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        TieredBackground::new(tiers, WithinTier::CoDetermined).unwrap()
    }
    fn admg(&self) -> Admg {
        let mut g = Admg::with_variables(u32::try_from(self.n).unwrap());
        for a in 0..self.n {
            for b in 0..self.n {
                if self.child(a, b) {
                    g.insert_directed(
                        DenseNodeId::from_raw(a as u32),
                        DenseNodeId::from_raw(b as u32),
                    )
                    .unwrap();
                }
                if a < b && self.bi(a, b) {
                    g.insert_bidirected(
                        DenseNodeId::from_raw(a as u32),
                        DenseNodeId::from_raw(b as u32),
                    )
                    .unwrap();
                }
            }
        }
        g
    }
    fn descendants(&self, targets: usize) -> usize {
        let mut d = targets;
        for a in 0..self.n {
            if targets >> a & 1 == 1 {
                for b in 0..self.n {
                    if self.child(a, b) {
                        d |= 1 << b;
                    }
                }
            }
        }
        d // the layout is saturated, so one step reaches every descendant
    }
    fn ancestors_of(&self, set: usize) -> usize {
        let mut anc = set;
        for b in 0..self.n {
            if set >> b & 1 == 1 {
                for a in 0..self.n {
                    if self.child(a, b) {
                        anc |= 1 << a;
                    }
                }
            }
        }
        anc
    }
    /// Some simple path `from .. to` is m-connecting given `cond`, with the arrows leaving
    /// `cut` removed.
    fn connected(&self, from: usize, to: usize, cond: usize, cut: usize) -> bool {
        let anc = self.ancestors_of(cond);
        let mut path = vec![from];
        let mut heads: Vec<bool> = Vec::new();
        self.walk(&mut path, &mut heads, to, cond, anc, cut)
    }
    fn edges(&self, a: usize, b: usize, cut: usize) -> Vec<(bool, bool)> {
        let mut out = Vec::new();
        if self.child(a, b) && cut >> a & 1 == 0 {
            out.push((false, true));
        }
        if self.child(b, a) && cut >> b & 1 == 0 {
            out.push((true, false));
        }
        if self.bi(a, b) {
            out.push((true, true));
        }
        out
    }
    fn walk(
        &self,
        path: &mut Vec<usize>,
        heads: &mut Vec<bool>,
        to: usize,
        cond: usize,
        anc: usize,
        cut: usize,
    ) -> bool {
        let last = *path.last().unwrap();
        if last == to {
            return true;
        }
        for next in 0..self.n {
            if path.contains(&next) {
                continue;
            }
            for (head_last, head_next) in self.edges(last, next, cut) {
                if path.len() >= 2 {
                    let collider = *heads.last().unwrap() && head_last;
                    let blocked =
                        if collider { anc >> last & 1 == 0 } else { cond >> last & 1 == 1 };
                    if blocked {
                        continue;
                    }
                }
                path.push(next);
                heads.push(head_next);
                let found = self.walk(path, heads, to, cond, anc, cut);
                path.pop();
                heads.pop();
                if found {
                    return true;
                }
            }
        }
        false
    }
    /// The definition of a joint back-door set for the treatments `t` and outcome `y`.
    fn joint_backdoor(&self, t: usize, y: usize, z: usize) -> bool {
        if z & (t | 1 << y) != 0 || z & self.descendants(t) != 0 {
            return false;
        }
        (0..self.n).filter(|&a| t >> a & 1 == 1).all(|a| {
            let others = t & !(1 << a);
            !self.connected(a, y, z | others, 1 << a)
        })
    }
    /// All non-treatment nodes in tiers up to the latest treatment's tier.
    fn closure(&self, t: usize) -> usize {
        let k = (0..self.n).filter(|&a| t >> a & 1 == 1).map(|a| self.tier[a]).max().unwrap();
        (0..self.n).filter(|&a| self.tier[a] <= k && t >> a & 1 == 0).fold(0, |m, a| m | 1 << a)
    }
}

fn layouts(n: usize) -> Vec<Layout> {
    let mut out = Vec::new();
    let mut tier = vec![0usize; n];
    loop {
        let top = *tier.iter().max().unwrap();
        if (0..=top).all(|k| tier.contains(&k)) {
            out.push(Layout { n, tier: tier.clone() });
        }
        let mut i = 0;
        loop {
            if i == n {
                return out;
            }
            tier[i] += 1;
            if tier[i] < n {
                break;
            }
            tier[i] = 0;
            i += 1;
        }
    }
}

fn set_of(mask: usize, n: usize) -> Vec<usize> {
    (0..n).filter(|&i| mask >> i & 1 == 1).collect()
}

fn id_identifies(
    id: &IdIdentifier,
    ws: &mut IdentificationWorkspace,
    layout: &Layout,
    t: usize,
    y: usize,
) -> bool {
    let prepared = id.prepare(&layout.admg()).unwrap();
    let interventions: Vec<_> =
        set_of(t, layout.n).iter().map(|&a| Intervention::set(v(a), Value::f64(1.0))).collect();
    let q = CausalQuery::Distribution(InterventionalDistributionQuery::new(v(y), interventions));
    id.identify(&prepared, &q, ws).unwrap().status
        == IdentificationStatus::NonparametricallyIdentified
}

#[test]
fn the_tier_closure_matches_the_back_door_definition_on_every_layout() {
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let (mut single, mut joint_ok, mut joint_refused, mut refused_but_id) = (0, 0, 0, 0);
    for n in 2..=5usize {
        let mut builder = CausalSchemaBuilder::new();
        for i in 0..n {
            builder = builder.continuous(format!("v{i}")).finish();
        }
        let schema = builder.build().unwrap();
        for layout in layouts(n) {
            let background = layout.background();
            for t in 1..1usize << n {
                let k = (0..n).filter(|&a| t >> a & 1 == 1).map(|a| layout.tier[a]).max().unwrap();
                for y in (0..n).filter(|&y| layout.tier[y] > k) {
                    let context = format!("tiers={:?} T={:?} Y={y}", layout.tier, set_of(t, n));
                    let z = layout.closure(t);
                    let accepted = layout.joint_backdoor(t, y, z);
                    if t.count_ones() == 1 {
                        // `identify_tiered` certifies the closure without a check.
                        assert!(
                            accepted,
                            "the single-treatment closure is not a back-door set: {context}"
                        );
                        let a = set_of(t, n)[0];
                        let result = identify_tiered(
                            &background,
                            &AverageEffectQuery::binary_ate(v(a), v(y)),
                        )
                        .unwrap();
                        assert_eq!(
                            result.status,
                            IdentificationStatus::NonparametricallyIdentified,
                            "{context}"
                        );
                        let mut sets: Vec<_> = result.estimands[0]
                            .adjustment_set
                            .iter()
                            .map(|x| x.raw() as usize)
                            .collect();
                        sets.sort_unstable();
                        assert_eq!(sets, set_of(z, n), "{context}");
                        single += 1;
                    } else {
                        let interventions: Vec<_> = set_of(t, n)
                            .iter()
                            .map(|&a| Intervention::set(v(a), Value::f64(1.0)))
                            .collect();
                        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                            outcome: v(y),
                            interventions: Arc::from(interventions),
                        });
                        let result = identify_tiered_joint(&background, &schema, &query).unwrap();
                        let product =
                            result.status == IdentificationStatus::NonparametricallyIdentified;
                        assert_eq!(product, accepted, "joint closure disagrees: {context}");
                        if product {
                            joint_ok += 1;
                            assert!(
                                id_identifies(&id, &mut ws, &layout, t, y),
                                "closure identifies but general ID does not: {context}"
                            );
                        } else {
                            joint_refused += 1;
                            if id_identifies(&id, &mut ws, &layout, t, y) {
                                refused_but_id += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "single={single} joint_ok={joint_ok} joint_refused={joint_refused} refused_but_id={refused_but_id}"
    );
    assert!(
        single > 1_000 && joint_ok > 100 && joint_refused > 100,
        "{single} {joint_ok} {joint_refused}"
    );
    // Documented limit: a refusal means "no joint back-door set on the closure", not
    // "not identified"; general ID identifies some refused queries.
    assert!(refused_but_id > 0, "{refused_but_id}");
}

#[test]
fn an_effect_modifier_after_the_treatment_is_refused_not_dropped() {
    let (mut refused, mut kept) = (0, 0);
    for n in 3..=5usize {
        for layout in layouts(n) {
            for unknown in [false, true] {
                let mut tiers = layout.background();
                if unknown {
                    tiers =
                        TieredBackground::new(tiers.tiers.to_vec(), WithinTier::Unknown).unwrap();
                }
                for t in 0..n {
                    for y in (0..n).filter(|&y| layout.tier[y] > layout.tier[t]) {
                        for w in (0..n).filter(|&w| w != t && w != y) {
                            let query = AverageEffectQuery::binary_ate(v(t), v(y))
                                .with_effect_modifiers([v(w)]);
                            let result = identify_tiered(&tiers, &query).unwrap();
                            let post = layout.tier[w] > layout.tier[t]
                                || (unknown && layout.tier[w] == layout.tier[t]);
                            let context = format!(
                                "tiers={:?} T={t} Y={y} W={w} unknown={unknown}",
                                layout.tier
                            );
                            if post {
                                refused += 1;
                                assert_eq!(
                                    result.status,
                                    IdentificationStatus::NotIdentified,
                                    "{context}"
                                );
                                assert!(result.estimands.is_empty(), "{context}");
                            } else {
                                kept += 1;
                                assert_ne!(
                                    result.status,
                                    IdentificationStatus::NotIdentified,
                                    "{context}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(refused > 500 && kept > 500, "{refused} {kept}");
    // A modifier equal to the treatment or outcome is not a query.
    let layout = Layout { n: 3, tier: vec![0, 1, 2] };
    for w in [0, 2] {
        let query = AverageEffectQuery::binary_ate(v(0), v(2)).with_effect_modifiers([v(w)]);
        let error = identify_tiered(&layout.background(), &query)
            .err()
            .expect("a modifier equal to the treatment or outcome is not a query")
            .to_string();
        assert!(error.contains("invalid average-effect query"), "w={w}: {error}");
    }
}

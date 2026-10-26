//! The orientation of one node's edges across the resolutions of PC's conflicting collider
//! evidence.
//!
//! PC orients every unshielded triple `a — c — b` whose centre is outside the separating set of
//! `(a, b)` as a collider and propagates the result with Meek's rules. When two colliders (or a
//! collider and background knowledge) disagree about an edge, the edge is marked as a conflict
//! and the orientation that happened to be applied first has already propagated, so the
//! returned graph depends on the order in which the evidence was visited. A conflict is
//! therefore not a statement about the edges it marks: it says that some colliders cannot all be
//! right, and any of the consistent choices among them is as supported as the one PC made.
//!
//! [`resolve_incident_orientation`] asks the question a consumer of one node's edges has: do
//! those edges come out the same under every consistent choice? A *resolution* keeps a subset of
//! the colliders and declares the others non-colliders, which is the same as adding their centre
//! to the separating set. The colliders implicated in a conflict are found as the closure of
//! those with a leg on a conflicted edge and those whose removal changes which edges conflict;
//! colliders outside the closure are kept in every resolution, exactly as in a conflict-free
//! run. Every subset of the implicated colliders is oriented again from the skeleton, and the
//! node's edges are compared across the subsets that leave no conflict anywhere. The number of
//! subsets is bounded by the caller: past the bound the resolutions cannot all be compared, and
//! the answer is [`ConflictResolution::Varies`] only when a sample of them already contains two
//! consistent resolutions that disagree (a witness needs no exhaustiveness), and
//! [`ConflictResolution::BudgetExceeded`] otherwise: invariance is never claimed from a partial
//! comparison.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_graph::{Cpdag, DenseNodeId, MarkedEdge};

use crate::constraints::DiscoveryConstraints;
use crate::error::DiscoveryError;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationRule, OrientationState,
    run_orientation_to_fixed_point,
};
use crate::pc::{load_sepsets_into_state, orient_fixed_directions};
use crate::pipeline::dense_of;
use crate::result::PcSepsets;

/// Largest implicated-collider bound a caller may request: the resolutions are `2^bound`, indexed
/// by a machine word. The cost of an exhaustive comparison is the caller's to choose through the
/// bound.
pub const MAX_IMPLICATED_COLLIDERS: usize = 32;

/// Resolutions sampled when the implicated colliders exceed the bound.
const SAMPLED_RESOLUTIONS: usize = 256;

/// How the edges at one node are oriented in one resolution, as sorted raw variable ids.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IncidentOrientation {
    /// Neighbours with an edge directed into the node.
    pub parents: Vec<u32>,
    /// Neighbours with an edge directed out of the node.
    pub children: Vec<u32>,
    /// Neighbours whose edge is undirected, or marked as a conflict.
    pub undirected: Vec<u32>,
}

/// What the resolutions of the conflicting colliders say about one node's edges.
#[derive(Clone, Debug, PartialEq)]
pub enum ConflictResolution {
    /// Every conflict-free resolution orients the node's edges in this way.
    Invariant(IncidentOrientation),
    /// Two conflict-free resolutions orient the node's edges differently.
    Varies,
    /// No choice among the implicated colliders removes every conflict.
    Unresolvable,
    /// More colliders are implicated than the bound allows resolutions for.
    BudgetExceeded {
        /// Colliders implicated when the bound was crossed.
        implicated: usize,
        /// The bound.
        budget: usize,
        /// Fraction of the `2^implicated` resolutions the sample compared (distinct
        /// resolutions drawn over all of them), in `[0, 1]`. Invariance is not claimed at any
        /// coverage below the exhaustive comparison the bound would have allowed.
        coverage: f64,
    },
}

/// One unshielded collider `a → centre ← b` of the separating sets, with `a < b`.
#[derive(Clone, Copy, Debug)]
struct Collider {
    a: DenseNodeId,
    centre: DenseNodeId,
    b: DenseNodeId,
}

/// One orientation of the skeleton: the edges in conflict and the node's incident edges.
#[derive(Clone, Debug)]
struct Trial {
    conflicts: BTreeSet<(u32, u32)>,
    incident: IncidentOrientation,
}

type Memo = HashMap<BTreeSet<usize>, Trial>;

fn edge_key(x: DenseNodeId, y: DenseNodeId) -> (u32, u32) {
    if x.raw() <= y.raw() { (x.raw(), y.raw()) } else { (y.raw(), x.raw()) }
}

struct Resolver<'a> {
    constraints: &'a DiscoveryConstraints,
    variables: &'a [VariableId],
    var_index: HashMap<VariableId, usize>,
    skeleton: Cpdag,
    state: OrientationState,
    colliders: Vec<Collider>,
    node: DenseNodeId,
    neighbours: Vec<DenseNodeId>,
}

impl<'a> Resolver<'a> {
    fn new(
        graph: &Cpdag,
        sepsets: &PcSepsets,
        constraints: &'a DiscoveryConstraints,
        variables: &'a [VariableId],
        node: VariableId,
    ) -> Result<Self, DiscoveryError> {
        if graph.node_count() != variables.len() {
            return Err(DiscoveryError::data_msg(
                "the graph and the variable list disagree on the node count",
            ));
        }
        let var_index: HashMap<VariableId, usize> =
            variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        let mut skeleton = Cpdag::empty();
        for node_ref in graph.nodes() {
            skeleton.add_node(*node_ref)?;
        }
        for edge in graph.edges() {
            skeleton.insert_undirected(edge.a, edge.b)?;
        }
        let mut state = OrientationState::default();
        load_sepsets_into_state(sepsets, &|v: VariableId| dense_of(&var_index, v), &mut state)?;

        let mut colliders = Vec::new();
        for a in DenseNodeId::all(skeleton.node_count()) {
            for b in DenseNodeId::all(skeleton.node_count()) {
                if b.raw() <= a.raw() || skeleton.has_edge(a, b) {
                    continue;
                }
                let Some(separator) = state.sepset(a, b) else {
                    continue;
                };
                for centre in skeleton.adjacent(a) {
                    if skeleton.has_edge(centre, b) && !separator.contains(&centre) {
                        colliders.push(Collider { a, centre, b });
                    }
                }
            }
        }
        let node = dense_of(&var_index, node)?;
        let neighbours = skeleton.adjacent(node);
        Ok(Self { constraints, variables, var_index, skeleton, state, colliders, node, neighbours })
    }

    /// Orient the skeleton from scratch with the colliders in `dropped` declared non-colliders.
    fn orient(&self, dropped: &BTreeSet<usize>) -> Result<Trial, DiscoveryError> {
        let mut graph = self.skeleton.clone();
        let mut state = self.state.clone();
        for &k in dropped {
            let Collider { a, centre, b } = self.colliders[k];
            let key = edge_key(a, b);
            let widened: Vec<DenseNodeId> = state
                .sepsets
                .get(&key)
                .map(|set| set.iter().copied().chain(std::iter::once(centre)).collect())
                .unwrap_or_default();
            state.sepsets.insert(key, Arc::from(widened));
        }
        let dense = |v: VariableId| dense_of(&self.var_index, v);
        orient_fixed_directions(&mut graph, self.constraints, self.variables, &dense, &mut state)?;
        let rules: [&dyn OrientationRule<Cpdag>; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        run_orientation_to_fixed_point(&mut graph, &rules, &mut state)?;

        let conflicts = graph
            .edges()
            .into_iter()
            .filter(|edge| edge.is_conflict())
            .map(|edge| edge_key(edge.a, edge.b))
            .collect();
        let mut incident = IncidentOrientation::default();
        for &neighbour in &self.neighbours {
            let raw = self.variables[neighbour.as_usize()].raw();
            match graph.edge_between(self.node, neighbour).and_then(MarkedEdge::parent_child) {
                Some((parent, _)) if parent == neighbour => incident.parents.push(raw),
                Some(_) => incident.children.push(raw),
                None => incident.undirected.push(raw),
            }
        }
        incident.parents.sort_unstable();
        incident.children.sort_unstable();
        incident.undirected.sort_unstable();
        Ok(Trial { conflicts, incident })
    }

    fn trial(&self, memo: &mut Memo, dropped: &BTreeSet<usize>) -> Result<Trial, DiscoveryError> {
        if let Some(known) = memo.get(dropped) {
            return Ok(known.clone());
        }
        let fresh = self.orient(dropped)?;
        memo.insert(dropped.clone(), fresh.clone());
        Ok(fresh)
    }

    /// Whether a sample of the resolutions of `implicated` already holds two conflict-free ones
    /// that orient the node's edges differently, and the fraction of all `2^|implicated|`
    /// resolutions the sample covered.
    ///
    /// The subsets are drawn from a stream keyed by the node and the implicated set, so the
    /// answer is reproducible. A `true` is a witness; a `false` proves nothing.
    fn sampled_resolutions_disagree(
        &self,
        memo: &mut Memo,
        implicated: &BTreeSet<usize>,
        node: VariableId,
    ) -> Result<(bool, f64), DiscoveryError> {
        let members: Vec<usize> = implicated.iter().copied().collect();
        let seed = members
            .iter()
            .fold(u64::from(node.raw()) ^ 0x9E37_79B9_7F4A_7C15, |acc, &k| {
                acc.rotate_left(7) ^ (k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
            });
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        let mut first: Option<IncidentOrientation> = None;
        let mut drawn: HashSet<BTreeSet<usize>> = HashSet::new();
        for _ in 0..SAMPLED_RESOLUTIONS {
            let dropped: BTreeSet<usize> =
                members.iter().copied().filter(|_| rng.next_f64() < 0.5).collect();
            let trial = self.trial(memo, &dropped)?;
            drawn.insert(dropped);
            if !trial.conflicts.is_empty() {
                continue;
            }
            match &first {
                None => first = Some(trial.incident),
                Some(known) if *known != trial.incident => return Ok((true, 0.0)),
                Some(_) => {}
            }
        }
        let total = 2f64.powi(i32::try_from(members.len()).unwrap_or(i32::MAX));
        Ok((false, (drawn.len() as f64 / total).min(1.0)))
    }

    /// The colliders kept in `dropped`'s resolution that conflict evidence rests on: those with
    /// a leg on a conflicted edge, and those whose removal changes which edges conflict (a
    /// conflict reached through Meek propagation has its colliders upstream of the conflicted
    /// edge).
    fn implicated(
        &self,
        memo: &mut Memo,
        dropped: &BTreeSet<usize>,
        trial: &Trial,
    ) -> Result<BTreeSet<usize>, DiscoveryError> {
        let mut found = BTreeSet::new();
        for (k, collider) in self.colliders.iter().enumerate() {
            if dropped.contains(&k) {
                continue;
            }
            let on_leg = trial.conflicts.contains(&edge_key(collider.a, collider.centre))
                || trial.conflicts.contains(&edge_key(collider.b, collider.centre));
            if on_leg {
                found.insert(k);
                continue;
            }
            let mut without = dropped.clone();
            without.insert(k);
            if self.trial(memo, &without)?.conflicts != trial.conflicts {
                found.insert(k);
            }
        }
        Ok(found)
    }
}

/// Whether the edges at `node` are oriented the same way in every conflict-free resolution of
/// the conflicting collider evidence behind `graph`.
///
/// `graph` supplies the skeleton (its orientations and conflict marks are ignored), `sepsets`
/// the separating sets PC recorded, and `constraints` the background knowledge PC was given.
/// With no conflict the single resolution is the graph PC returned. At most `max_implicated`
/// colliders are resolved (`2^max_implicated` orientations of the skeleton).
///
/// # Errors
///
/// A bound above [`MAX_IMPLICATED_COLLIDERS`], a graph that does not match `variables`, a
/// variable outside the graph, or an orientation failure.
pub fn resolve_incident_orientation(
    graph: &Cpdag,
    sepsets: &PcSepsets,
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    node: VariableId,
    max_implicated: usize,
) -> Result<ConflictResolution, DiscoveryError> {
    if max_implicated > MAX_IMPLICATED_COLLIDERS {
        return Err(DiscoveryError::unsupported(
            "the implicated-collider bound is above the supported maximum",
        ));
    }
    let resolver = Resolver::new(graph, sepsets, constraints, variables, node)?;
    let mut memo = Memo::new();
    let mut implicated: BTreeSet<usize> = BTreeSet::new();
    loop {
        if implicated.len() > max_implicated {
            let (disagree, coverage) =
                resolver.sampled_resolutions_disagree(&mut memo, &implicated, node)?;
            if disagree {
                return Ok(ConflictResolution::Varies);
            }
            return Ok(ConflictResolution::BudgetExceeded {
                implicated: implicated.len(),
                budget: max_implicated,
                coverage,
            });
        }
        let members: Vec<usize> = implicated.iter().copied().collect();
        let mut consistent: Vec<IncidentOrientation> = Vec::new();
        let mut grown = false;
        for mask in 0..(1_usize << members.len()) {
            let dropped: BTreeSet<usize> = members
                .iter()
                .enumerate()
                .filter(|&(bit, _)| (mask >> bit) & 1 == 1)
                .map(|(_, &k)| k)
                .collect();
            let trial = resolver.trial(&mut memo, &dropped)?;
            if trial.conflicts.is_empty() {
                consistent.push(trial.incident);
                continue;
            }
            for k in resolver.implicated(&mut memo, &dropped, &trial)? {
                grown |= implicated.insert(k);
            }
        }
        if grown {
            continue;
        }
        let Some(first) = consistent.first() else {
            return Ok(ConflictResolution::Unresolvable);
        };
        return Ok(if consistent.iter().all(|other| other == first) {
            ConflictResolution::Invariant(first.clone())
        } else {
            ConflictResolution::Varies
        });
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::Lag;

    use super::*;
    use crate::pc::Pc;
    use crate::result::LaggedParent;

    fn variables(n: u32) -> Vec<VariableId> {
        (0..n).map(VariableId::from_raw).collect()
    }

    fn skeleton(n: u32, edges: &[(u32, u32)]) -> Cpdag {
        let mut graph = Cpdag::with_variables(n);
        for &(a, b) in edges {
            graph.insert_undirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        graph
    }

    /// Separating sets for `(a, b, set)` with `a < b`.
    fn sepsets(entries: &[(u32, u32, &[u32])]) -> PcSepsets {
        entries
            .iter()
            .map(|&(a, b, set)| {
                let members: Vec<LaggedParent> =
                    set.iter().map(|&v| (VariableId::from_raw(v), Lag::CONTEMPORANEOUS)).collect();
                (
                    (
                        VariableId::from_raw(a),
                        Lag::CONTEMPORANEOUS,
                        VariableId::from_raw(b),
                        Lag::CONTEMPORANEOUS,
                    ),
                    Arc::from(members),
                )
            })
            .collect()
    }

    fn resolve(
        n: u32,
        edges: &[(u32, u32)],
        separators: &[(u32, u32, &[u32])],
        node: u32,
        budget: usize,
    ) -> ConflictResolution {
        resolve_incident_orientation(
            &skeleton(n, edges),
            &sepsets(separators),
            &Pc::new().constraints,
            &variables(n),
            VariableId::from_raw(node),
            budget,
        )
        .unwrap()
    }

    /// Nodes 0 → 1 ← 2 form a collider at the node 1. Nodes 3, 4, 5, 6 form the edge 3 — 4 that
    /// the collider at 4 (3 → 4 ← 5) orients one way and the collider at 3 (4 → 3 ← 6) the other.
    const EDGES: [(u32, u32); 5] = [(0, 1), (1, 2), (3, 4), (4, 5), (3, 6)];
    const SEPARATORS: [(u32, u32, &[u32]); 3] = [(0, 2, &[]), (3, 5, &[]), (4, 6, &[])];

    fn collider_at_one() -> IncidentOrientation {
        IncidentOrientation { parents: vec![0, 2], children: vec![], undirected: vec![] }
    }

    #[test]
    fn a_conflict_away_from_the_node_leaves_its_edges_as_in_the_conflict_free_graph() {
        let with_conflict = resolve(7, &EDGES, &SEPARATORS, 1, 8);
        let without = resolve(3, &EDGES[..2], &SEPARATORS[..1], 1, 8);
        assert_eq!(with_conflict, ConflictResolution::Invariant(collider_at_one()));
        assert_eq!(without, with_conflict);
    }

    #[test]
    fn a_conflict_on_the_nodes_own_edge_makes_its_orientation_depend_on_the_resolution() {
        // Keeping the collider at 3 gives 4 → 3 ← 6; keeping the one at 4 gives 3 → 4 and
        // leaves 3 — 6 open; keeping neither leaves both open.
        assert_eq!(resolve(7, &EDGES, &SEPARATORS, 3, 8), ConflictResolution::Varies);
    }

    #[test]
    fn more_implicated_colliders_than_the_bound_are_reported_not_truncated() {
        let ConflictResolution::BudgetExceeded { implicated, budget, coverage } =
            resolve(7, &EDGES, &SEPARATORS, 1, 1)
        else {
            panic!("two implicated colliders exceed a bound of one");
        };
        assert_eq!((implicated, budget), (2, 1));
        assert!(coverage > 0.0 && coverage <= 1.0, "coverage {coverage}");
    }

    #[test]
    fn a_bound_above_the_supported_maximum_is_refused() {
        let refused = resolve_incident_orientation(
            &skeleton(3, &EDGES[..2]),
            &sepsets(&SEPARATORS[..1]),
            &Pc::new().constraints,
            &variables(3),
            VariableId::from_raw(1),
            MAX_IMPLICATED_COLLIDERS + 1,
        );
        assert!(refused.is_err());
    }
}

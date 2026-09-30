//! Checking a cross-world query on a fixed Markovian DAG (stage one of three).
//!
//! The cross-world estimand is evaluated in three separable stages: this check
//! decides whether the graph and the query license it and returns a
//! machine-checkable witness; a coupled operation abducts, acts and predicts over
//! shared exogenous terms; and a result carries the point beside the witness. A
//! later counterfactual-identification route on latent-variable graphs replaces
//! this check and keeps the other two.
//!
//! # Contract
//!
//! Two worlds over one fixed directed acyclic graph on observed variables (so no
//! latent confounding): world 0 is the baseline `do(X = control)` with every
//! edge reading its own world; world 1 is `do(X = active)` where each edge reads
//! either world 1 (it sees the intervened value) or world 0 (it sees the
//! baseline value). The estimand is `E[Y_1] - E[Y_0]` under one exogenous term
//! per variable and unit shared by both worlds.
//!
//! # Assumptions the witness names
//!
//! - `consistency`: a unit's observed values are its values under the treatment
//!   it received, which is what lets its exogenous terms be abducted. It holds
//!   by construction in the abduction pipeline (each exogenous term is defined as
//!   the residual that regenerates the observed value) and is *not testable from
//!   the data*, so it is named, never checked. What the coupled operation does
//!   enforce is well-posed abduction: it refuses (`cross_world.invalid_query`)
//!   when abduction is not exact inversion or a mechanism cannot be inverted.
//! - `markovian_no_latent_confounding`: every variable is observed and the
//!   exogenous terms are mutually independent, so no confounder is missing.
//! - `mechanism_invariance_across_worlds`: each variable's structural mechanism
//!   is the same function in every world; only what its inputs read differs.
//! - `no_recanting_witness`: no non-treatment variable is needed in both worlds
//!   by one term while taking a different value in each; its cross-world joint
//!   law would be needed and no observational law pins it down (Avin, Shpitser &
//!   Pearl 2005; Shpitser 2013).
//!
//! # The counterfactual nodes
//!
//! The witness lists every `(variable, world)` pair each contrast term depends
//! on, with the world each parent is read from. That is the merged
//! counterfactual graph of the query restricted to what the estimand needs, and
//! the consumer recomputes it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{CrossWorldQuery, ExogenousCoupling, VariableId, WorldId, WorldObservation};
use antecedent_graph::Dag;
use serde::{Deserialize, Serialize};

/// Most variables a graph may have under this contract.
pub const CROSS_WORLD_MAX_NODES: usize = 8;
/// The contract the witness attests.
pub const CROSS_WORLD_CONTRACT: &str = "path_specific_two_world_markovian_dag";

const ASSUMPTIONS: [&str; 4] = [
    "consistency",
    "markovian_no_latent_confounding",
    "mechanism_invariance_across_worlds",
    "no_recanting_witness",
];

/// Why a cross-world query is refused on a graph.
///
/// `cross_world_not_identified` is reserved for genuine nonidentification (a
/// recanting witness). A query shape this cell does not implement is
/// `route_not_supported`, never a claim about identification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrossWorldRefusal {
    /// Registered reason code (`parity/reason_codes.toml`).
    pub code: &'static str,
    /// Namespaced stable detail.
    pub detail: &'static str,
    /// Explanation naming what failed.
    pub message: String,
}

impl CrossWorldRefusal {
    /// A well-formed query whose shape this cell does not implement. Not a
    /// nonidentification finding: nothing is claimed about whether the shape is
    /// identified, only that no route here evaluates it.
    fn outside_contract(message: impl Into<String>) -> Self {
        Self {
            code: "route_not_supported",
            detail: "cross_world.query_outside_contract",
            message: message.into(),
        }
    }

    fn recanting(variable: u32, message: impl Into<String>) -> Self {
        Self {
            code: "cross_world_not_identified",
            detail: "cross_world.recanting_witness",
            message: format!("variable {variable} {}", message.into()),
        }
    }

    fn graph(message: impl Into<String>) -> Self {
        Self {
            code: "cell_not_licensed",
            detail: "cross_world.graph_outside_contract",
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_argument",
            detail: "cross_world.invalid_query",
            message: message.into(),
        }
    }
}

/// One `(variable, world)` of the merged counterfactual graph and the world each
/// of its parents is read from.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualNode {
    /// Variable.
    pub variable: u32,
    /// World the variable is evaluated in.
    pub world: u8,
    /// `(parent, world read)` for every parent; empty for a hard-set variable.
    pub reads: Vec<(u32, u8)>,
}

impl CounterfactualNode {
    /// The variable this node names, for callers holding [`VariableId`]s.
    #[must_use]
    pub fn variable_id(&self) -> VariableId {
        VariableId::from_raw(self.variable)
    }
}

/// The machine-checkable derivation of a cross-world estimand.
///
/// The struct is `#[non_exhaustive]` and rejects unknown fields on the wire: a
/// later route (ADMG counterfactual identification, B5) adds fields or a new
/// witness kind under a new artifact version, and a reader of this version must
/// refuse it rather than misread it. Today it describes one fixed-DAG contract
/// whose reroutes all read world 0 (`rerouted_edges` carries the source world so
/// a wider contract can reuse the field, but [`check_cross_world_edges`] only
/// produces source 0).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CrossWorldWitness {
    /// Contract attested ([`CROSS_WORLD_CONTRACT`]).
    pub contract: String,
    /// Assumptions the estimand relies on, by name.
    pub assumptions: Vec<String>,
    /// The graph's directed edges, sorted.
    pub graph_edges: Vec<(u32, u32)>,
    /// Treatment variable.
    pub treatment: u32,
    /// Outcome variable.
    pub outcome: u32,
    /// Edges whose child sees the intervened world's parent value, sorted.
    pub intervened_edges: Vec<(u32, u32)>,
    /// Edges rerouted to another world as `(parent, child, source world)`, sorted.
    pub rerouted_edges: Vec<(u32, u32, u8)>,
    /// Worlds whose exogenous terms are shared.
    pub coupled_worlds: Vec<u8>,
    /// The merged counterfactual graph restricted to the estimand, sorted.
    pub counterfactual_nodes: Vec<CounterfactualNode>,
    /// Canonical text of the checked query.
    pub query_text: String,
}

impl CrossWorldWitness {
    /// Recompute the witness from `graph_edges` and `query` alone and require it
    /// to equal this one, so a consumer trusts nothing the producer stored.
    ///
    /// # Errors
    /// The refusal the recomputation raises, or a message when the recomputed
    /// witness differs from the stored one.
    pub fn verify(&self, node_count: usize, query: &CrossWorldQuery) -> Result<(), String> {
        let recomputed = check_cross_world_edges(node_count, &self.graph_edges, query)
            .map_err(|r| format!("{}: {}", r.detail, r.message))?;
        if &recomputed == self {
            Ok(())
        } else {
            Err("stored witness differs from the recomputed derivation".into())
        }
    }
}

/// Check `query` on `graph`.
///
/// # Errors
/// A [`CrossWorldRefusal`] naming the failed condition.
pub fn check_cross_world(
    graph: &Dag,
    query: &CrossWorldQuery,
) -> Result<CrossWorldWitness, CrossWorldRefusal> {
    let mut edges: Vec<(u32, u32)> = graph.edges().map(|e| (e.a.raw(), e.b.raw())).collect();
    edges.sort_unstable();
    check_cross_world_edges(graph.node_count(), &edges, query)
}

/// [`check_cross_world`] on an explicit node count and sorted edge list.
///
/// # Errors
/// A [`CrossWorldRefusal`] naming the failed condition.
#[allow(clippy::too_many_lines, reason = "one linear list of contract conditions")]
#[doc(hidden)]
pub fn check_cross_world_edges(
    node_count: usize,
    edges: &[(u32, u32)],
    query: &CrossWorldQuery,
) -> Result<CrossWorldWitness, CrossWorldRefusal> {
    if node_count == 0 || node_count > CROSS_WORLD_MAX_NODES {
        return Err(CrossWorldRefusal::graph(format!(
            "the graph has {node_count} variables; the contract covers 1 to {CROSS_WORLD_MAX_NODES}"
        )));
    }
    let n = u32::try_from(node_count).unwrap_or(u32::MAX);
    if edges.iter().any(|&(a, b)| a >= n || b >= n || a == b) || !is_acyclic(node_count, edges) {
        return Err(CrossWorldRefusal::graph("the edge list is not a DAG over the variables"));
    }
    let mut sorted = edges.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let edges = sorted.as_slice();

    // Shape of the worlds: baseline and one intervened world reading the baseline.
    let worlds = query.worlds();
    if worlds.len() != 2 || !matches!(query.coupling(), ExogenousCoupling::SharedAbducedExogenous) {
        return Err(CrossWorldRefusal::outside_contract(
            "the contract covers exactly two worlds sharing every exogenous term",
        ));
    }
    let base: Vec<_> = worlds[0].interventions().collect();
    let live: Vec<_> = worlds[1].interventions().collect();
    let (&[(treatment, _)], &[(live_treatment, _)]) = (base.as_slice(), live.as_slice()) else {
        return Err(CrossWorldRefusal::outside_contract(
            "each world sets exactly the treatment variable",
        ));
    };
    if treatment != live_treatment
        || !worlds[0].routes().is_empty()
        || worlds[1].routes().iter().any(|r| r.source != WorldId::new(0))
    {
        return Err(CrossWorldRefusal::outside_contract(
            "the baseline world reads only itself and the intervened world reads only the baseline",
        ));
    }
    let outcome = query.plus().variable;
    let expected_plus = WorldObservation { world: WorldId::new(1), variable: outcome };
    let expected_minus = WorldObservation { world: WorldId::new(0), variable: outcome };
    if query.plus() != expected_plus || query.minus() != expected_minus {
        return Err(CrossWorldRefusal::outside_contract(
            "the contrast must read the one outcome in the intervened world minus the baseline",
        ));
    }
    if treatment == outcome {
        return Err(CrossWorldRefusal::invalid("treatment and outcome must be distinct"));
    }
    for variable in [treatment, outcome] {
        if variable.raw() >= n {
            return Err(CrossWorldRefusal::invalid(format!(
                "variable {} is not in the graph",
                variable.raw()
            )));
        }
    }
    let rerouted: Vec<(u32, u32, u8)> =
        worlds[1].routes().iter().map(|r| (r.parent.raw(), r.child.raw(), 0u8)).collect();
    if let Some(&(p, c, _)) = rerouted.iter().find(|&&(p, c, _)| !edges.contains(&(p, c))) {
        return Err(CrossWorldRefusal::invalid(format!(
            "routed edge {p} -> {c} is not an edge of the graph"
        )));
    }
    let intervened: Vec<(u32, u32)> = edges
        .iter()
        .copied()
        .filter(|&(p, c)| !rerouted.iter().any(|&(rp, rc, _)| (rp, rc) == (p, c)))
        .collect();

    // The merged counterfactual graph each term depends on.
    let parents = |v: u32| -> Vec<u32> { edges.iter().filter(|e| e.1 == v).map(|e| e.0).collect() };
    let read_world = |world: u8, parent: u32, child: u32| -> u8 {
        if world == 1 && rerouted.iter().any(|&(p, c, _)| (p, c) == (parent, child)) {
            0
        } else {
            world
        }
    };
    let reads_of = |variable: u32, world: u8| -> Vec<(u32, u8)> {
        if variable == treatment.raw() {
            return Vec::new();
        }
        parents(variable).into_iter().map(|p| (p, read_world(world, p, variable))).collect()
    };
    let closure = |world: u8| -> BTreeSet<(u32, u8)> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![(outcome.raw(), world)];
        while let Some((v, w)) = stack.pop() {
            if seen.insert((v, w)) {
                stack.extend(reads_of(v, w));
            }
        }
        seen
    };
    // A variable takes a different value in world 1 than in world 0 only through
    // an edge reading world 1 from a variable that does.
    let mut differs: BTreeMap<u32, bool> = BTreeMap::new();
    let mut order: Vec<u32> = (0..n).collect();
    order.sort_by_key(|&v| depth(v, edges));
    for &v in &order {
        let d = v == treatment.raw()
            || (v != treatment.raw()
                && edges.iter().any(|&(p, c)| {
                    c == v && read_world(1, p, c) == 1 && *differs.get(&p).unwrap_or(&false)
                }));
        differs.insert(v, d);
    }
    let plus_terms = closure(1);
    let minus_terms = closure(0);
    for terms in [&plus_terms, &minus_terms] {
        let worlds_of = |v: u32| terms.iter().filter(|(x, _)| *x == v).count();
        if let Some(w) = (0..n).find(|&v| {
            v != treatment.raw() && v != outcome.raw() && worlds_of(v) > 1 && differs[&v]
        }) {
            return Err(CrossWorldRefusal::recanting(
                w,
                "is needed at both worlds by one term while taking a different value in each; \
                 its cross-world joint law is not identified from observational data",
            ));
        }
    }
    let nodes: BTreeSet<(u32, u8)> = plus_terms.union(&minus_terms).copied().collect();
    let counterfactual_nodes = nodes
        .into_iter()
        .map(|(variable, world)| CounterfactualNode {
            variable,
            world,
            reads: reads_of(variable, world),
        })
        .collect();
    Ok(CrossWorldWitness {
        contract: CROSS_WORLD_CONTRACT.into(),
        assumptions: ASSUMPTIONS.iter().map(|a| (*a).into()).collect(),
        graph_edges: edges.to_vec(),
        treatment: treatment.raw(),
        outcome: outcome.raw(),
        intervened_edges: intervened,
        rerouted_edges: rerouted,
        coupled_worlds: vec![0, 1],
        counterfactual_nodes,
        query_text: query.canonical_text(),
    })
}

fn is_acyclic(n: usize, edges: &[(u32, u32)]) -> bool {
    let mut indegree = vec![0usize; n];
    for &(_, b) in edges {
        indegree[b as usize] += 1;
    }
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut removed = 0;
    while let Some(v) = ready.pop() {
        removed += 1;
        for &(a, b) in edges {
            if a as usize == v {
                indegree[b as usize] -= 1;
                if indegree[b as usize] == 0 {
                    ready.push(b as usize);
                }
            }
        }
    }
    removed == n
}

/// Longest directed path ending at `v`; a topological sort key.
fn depth(v: u32, edges: &[(u32, u32)]) -> usize {
    edges.iter().filter(|e| e.1 == v).map(|e| 1 + depth(e.0, edges)).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    const MEDIATION: [(u32, u32); 3] = [(0, 1), (0, 2), (1, 2)];

    #[test]
    fn direct_and_indirect_are_checked_with_their_dependencies() {
        let direct = CrossWorldQuery::natural_direct(v(0), v(1), v(2), 0.0, 1.0).unwrap();
        let w = check_cross_world_edges(3, &MEDIATION, &direct).unwrap();
        assert_eq!(w.intervened_edges, vec![(0, 2)]);
        assert_eq!(w.rerouted_edges, vec![(0, 1, 0), (1, 2, 0)]);
        // Y in world 1 reads X from world 1 and M from world 0.
        let y1 = w.counterfactual_nodes.iter().find(|n| (n.variable, n.world) == (2, 1)).unwrap();
        assert_eq!(y1.reads, vec![(0, 1), (1, 0)]);
        w.verify(3, &direct).unwrap();

        let indirect = CrossWorldQuery::natural_indirect(v(0), v(1), v(2), 0.0, 1.0).unwrap();
        let w = check_cross_world_edges(3, &MEDIATION, &indirect).unwrap();
        assert_eq!(w.intervened_edges, vec![(0, 1), (1, 2)]);
        let y1 = w.counterfactual_nodes.iter().find(|n| (n.variable, n.world) == (2, 1)).unwrap();
        assert_eq!(y1.reads, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn a_recanting_witness_is_refused_by_name() {
        // X -> W -> Y and W -> M -> Y: the path X -> W -> Y sees the intervened
        // value while X -> W -> M -> Y sees the baseline, so W is needed in both.
        let edges = [(0, 1), (1, 2), (1, 3), (2, 3)];
        let query = CrossWorldQuery::path_specific(
            v(0),
            v(3),
            0.0,
            1.0,
            &edges.map(|(a, b)| (v(a), v(b))),
            &[(v(0), v(1)), (v(1), v(3))],
        )
        .unwrap();
        let refusal = check_cross_world_edges(4, &edges, &query).unwrap_err();
        assert_eq!(refusal.code, "cross_world_not_identified");
        assert_eq!(refusal.detail, "cross_world.recanting_witness");
    }

    #[test]
    fn worlds_outside_the_contract_are_not_a_nonidentification_claim() {
        use antecedent_core::WorldSpec;
        let query = CrossWorldQuery::new(
            vec![
                WorldSpec::new([(v(0), 0.0)], []).unwrap(),
                WorldSpec::new([(v(0), 1.0)], []).unwrap(),
                WorldSpec::new([(v(0), 2.0)], []).unwrap(),
            ],
            ExogenousCoupling::SharedAbducedExogenous,
            WorldObservation { world: WorldId::new(1), variable: v(2) },
            WorldObservation { world: WorldId::new(0), variable: v(2) },
        )
        .unwrap();
        let refusal = check_cross_world_edges(3, &MEDIATION, &query).unwrap_err();
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, "cross_world.query_outside_contract");
    }

    /// A contrast that reads different outcomes per world is another shape, not
    /// an unidentified one, and unknown witness fields are refused on the wire.
    #[test]
    fn mixed_outcomes_are_outside_the_contract_and_the_witness_wire_is_closed() {
        let mixed = CrossWorldQuery::new(
            vec![
                antecedent_core::WorldSpec::new([(v(0), 0.0)], []).unwrap(),
                antecedent_core::WorldSpec::new([(v(0), 1.0)], []).unwrap(),
            ],
            ExogenousCoupling::SharedAbducedExogenous,
            WorldObservation { world: WorldId::new(1), variable: v(2) },
            WorldObservation { world: WorldId::new(0), variable: v(1) },
        )
        .unwrap();
        let refusal = check_cross_world_edges(3, &MEDIATION, &mixed).unwrap_err();
        assert_eq!(
            (refusal.code, refusal.detail),
            ("route_not_supported", "cross_world.query_outside_contract")
        );

        let direct = CrossWorldQuery::natural_direct(v(0), v(1), v(2), 0.0, 1.0).unwrap();
        let witness = check_cross_world_edges(3, &MEDIATION, &direct).unwrap();
        let mut json = serde_json::to_value(&witness).unwrap();
        assert!(serde_json::from_value::<CrossWorldWitness>(json.clone()).is_ok());
        json["unexpected"] = serde_json::json!(1);
        assert!(serde_json::from_value::<CrossWorldWitness>(json).is_err());
        let mut node = serde_json::to_value(&witness.counterfactual_nodes[0]).unwrap();
        node["unexpected"] = serde_json::json!(1);
        assert!(serde_json::from_value::<CounterfactualNode>(node).is_err());
    }
}

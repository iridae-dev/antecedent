// SPDX-License-Identifier: MIT OR Apache-2.0
//! FCI interpretation for simple, sigma-faithful cyclic SCMs (Mooij and
//! Claassen, UAI 2020). FCI's PAG is an equivalence-class representation, not
//! a recovered original directed feedback graph. Exact sigma-CI supplies a
//! reference oracle; sampled-data runs retain their CI/search limitations.

use crate::{
    engine::DiscoveryWorkspace,
    error::DiscoveryError,
    fci::{Fci, StaticPagDiscoveryResult},
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{CyclicMixedGraph, DenseNodeId, SigmaSeparationWorkspace, sigma_separates};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};
use std::sync::Arc;

/// Required scientific assumptions; declarations are retained assumptions, not
/// a statistical test of simplicity or faithfulness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CyclicFciPremises {
    /// Unique solvability for every subset (simple SCM).
    pub simple_scm: bool,
    /// Conditional independences match sigma-separation.
    pub sigma_faithful: bool,
    /// Sampled rows are independent equilibrium replicates; no time-series substitution.
    pub independent_equilibrium_replicates: bool,
}

/// PAG evidence together with the explicit cyclic interpretation.
pub struct CyclicFciResult {
    /// Existing FCI result, preserving CI evidence and incomplete-search diagnostics.
    pub discovery: StaticPagDiscoveryResult,
    /// Assumptions under which the graph is interpreted.
    pub premises: CyclicFciPremises,
    /// Interpretation boundary carried to downstream consumers.
    pub interpretation: &'static str,
}

/// Sufficient PAG conclusions under the declared cyclic FCI premises.
/// Unknown relations are omitted; no adjacency is automatically a direct edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicPagConclusion {
    /// Ordered source node.
    pub source: u32,
    /// Ordered target node.
    pub target: u32,
    /// `ancestor`, `not_ancestor`, `direct_edge`, `no_direct_edge`, or `no_confounder`.
    pub relation: String,
    /// Published sufficient criterion, without a completeness claim.
    pub criterion: String,
}
/// Read sound path-based conclusions (Mooij–Claassen 2020, Propositions 4,5,6,7,8(i)).
/// The caller must supply a sound DPAG under the stated assumptions. This does
/// not test faithfulness or convert a finite-sample discovery into a certificate.
/// # Errors
/// Cancellation, resource refusal, conflicting marks or more than 128 nodes.
pub fn cyclic_pag_conclusions(
    pag: &antecedent_graph::Pag,
    ctx: &ExecutionContext,
) -> Result<Vec<CyclicPagConclusion>, DiscoveryError> {
    use antecedent_graph::Endpoint;
    let n = pag.node_count();
    if n > 128
        || (0..n).any(|i| {
            pag.neighbors(DenseNodeId::from_raw(u32::try_from(i).expect("bounded node"))).any(
                |(_, a, b)| {
                    a == Endpoint::Conflict
                        || b == Endpoint::Conflict
                        || (a == Endpoint::Tail && b == Endpoint::Tail)
                },
            )
        })
    {
        return Err(DiscoveryError::unsupported(
            "cyclic PAG interpretation requires bounded nonconflicting marks",
        ));
    }
    let _lease = ctx
        .acquire_admission(1, (n * n * 1024 + 4096) as u64)
        .map_err(|_| DiscoveryError::unsupported("cyclic PAG interpretation resources"))?;
    let id = |i: usize| DenseNodeId::from_raw(u32::try_from(i).expect("bounded node"));
    let reachable = |start: usize, end: usize, directed: bool, skip: Option<(usize, usize)>| {
        let mut seen = vec![false; n];
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(a) = stack.pop() {
            for (b, at_a, at_b) in pag.neighbors(id(a)) {
                let b = b.as_usize();
                if skip == Some((a, b)) || skip == Some((b, a)) {
                    continue;
                }
                let allowed = if directed {
                    at_a == Endpoint::Tail && at_b == Endpoint::Arrow
                } else {
                    at_a != Endpoint::Arrow && at_b != Endpoint::Tail
                };
                if allowed {
                    if b == end {
                        return true;
                    }
                    if !seen[b] {
                        seen[b] = true;
                        stack.push(b);
                    }
                }
            }
        }
        false
    };
    let mut answer = Vec::new();
    for a in 0..n {
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::unsupported("cyclic PAG interpretation cancelled"));
        }
        for b in 0..n {
            if a == b {
                continue;
            }
            let mut emit = |relation: &str, criterion: &str| {
                answer.push(CyclicPagConclusion {
                    source: u32::try_from(a).expect("bounded node"),
                    target: u32::try_from(b).expect("bounded node"),
                    relation: relation.into(),
                    criterion: criterion.into(),
                })
            };
            if reachable(a, b, true, None) {
                emit("ancestor", "Proposition 4: directed path");
            }
            if !reachable(a, b, false, None) {
                emit("not_ancestor", "Proposition 5: no possibly directed path");
            }
            let edge = pag.edge_between(id(a), id(b));
            if edge.is_none() || edge.is_some_and(|e| e.at_a == Endpoint::Arrow) {
                emit("no_direct_edge", "Proposition 7: nonadjacent or arrowhead at source");
            }
            if edge.is_none() {
                emit("no_confounder", "Proposition 6: nonadjacent");
            }
            if edge.is_some_and(|e| e.at_a == Endpoint::Tail && e.at_b == Endpoint::Arrow)
                && !reachable(a, b, false, Some((a, b)))
            {
                emit(
                    "direct_edge",
                    "Proposition 8(i): directed edge with no alternative possibly directed path",
                );
            }
        }
    }
    Ok(answer)
}

/// Run the existing FCI engine under its cyclic interpretation. The engine and
/// all of its conditioning/path budgets remain unchanged and explicit.
pub fn discover_cyclic_fci(
    engine: &Fci,
    premises: CyclicFciPremises,
    data: &TabularData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<CyclicFciResult, DiscoveryError> {
    if !premises.simple_scm
        || !premises.sigma_faithful
        || !premises.independent_equilibrium_replicates
    {
        return Err(DiscoveryError::unsupported(
            "cyclic FCI requires simple sigma-faithful SCMs and independent equilibrium replicates; these are scientific assumptions",
        ));
    }
    let discovery = engine.run(data, variables, workspace, ctx)?;
    Ok(CyclicFciResult {
        discovery,
        premises,
        interpretation: "sigma-Markov equivalence-class PAG; adjacency may denote an inducing path, not an original direct mechanism; no effect magnitude is identified",
    })
}

/// Exact sigma-separation CI oracle on a specified cyclic mixed graph.
/// Column-to-node correspondence is explicit and is independent of graph ids.
#[derive(Clone, Debug)]
pub struct SigmaCiOracle {
    graph: Arc<CyclicMixedGraph>,
    column_nodes: Vec<DenseNodeId>,
}
impl SigmaCiOracle {
    /// Bind each input column to a unique original graph node.
    pub fn new(
        graph: CyclicMixedGraph,
        column_nodes: Vec<DenseNodeId>,
    ) -> Result<Self, DiscoveryError> {
        let mut seen = std::collections::BTreeSet::new();
        if column_nodes.is_empty()
            || column_nodes
                .iter()
                .any(|id| id.as_usize() >= graph.node_count() || !seen.insert(*id))
        {
            return Err(DiscoveryError::unsupported(
                "sigma oracle needs unique in-range column node bindings",
            ));
        }
        Ok(Self { graph: Arc::new(graph), column_nodes })
    }
}
impl ConditionalIndependenceTest for SigmaCiOracle {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let mut results = Vec::with_capacity(request.queries.len());
        let mut ws = SigmaSeparationWorkspace::default();
        let node = |i: usize| {
            self.column_nodes
                .get(i)
                .copied()
                .ok_or(StatsError::Shape { message: "sigma oracle column index" })
        };
        for q in request.queries {
            if ctx.cancellation.is_cancelled() {
                return Err(StatsError::Shape { message: "sigma oracle cancelled" });
            }
            let end = q
                .z_start
                .checked_add(q.z_len)
                .ok_or(StatsError::Shape { message: "sigma oracle conditioning overflow" })?;
            let z = request
                .z_flat
                .get(q.z_start..end)
                .ok_or(StatsError::Shape { message: "sigma oracle conditioning range" })?;
            let conditioning = z.iter().map(|&i| node(i)).collect::<Result<Vec<_>, _>>()?;
            let answer =
                sigma_separates(&self.graph, node(q.x)?, node(q.y)?, &conditioning, &mut ws)
                    .map_err(|_| StatsError::Shape { message: "sigma oracle graph query" })?;
            let separated = answer.status.is_separated();
            results.push(CiResult {
                statistic: if separated { 0.0 } else { 1.0 },
                p_value: if separated { 1.0 } else { 0.0 },
                df: 0.0,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
    fn p_value_is_frequentist(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_sigma_fci_accepts_cycles_without_turning_pag_into_direct_mechanisms() {
        let data = crate::test_tables::tabular_n(4, 50);
        let id = DenseNodeId::from_raw;
        let mut graph = CyclicMixedGraph::with_variables(4);
        graph.insert_directed(id(0), id(1)).unwrap();
        graph.insert_directed(id(1), id(0)).unwrap();
        graph.insert_directed(id(1), id(2)).unwrap();
        let oracle = SigmaCiOracle::new(graph, (0..4).map(id).collect()).unwrap();
        let engine = Fci::new().with_ci(Arc::new(oracle));
        let premises = CyclicFciPremises {
            simple_scm: true,
            sigma_faithful: true,
            independent_equilibrium_replicates: true,
        };
        let vars: Vec<_> = (0..4).map(VariableId::from_raw).collect();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = discover_cyclic_fci(&engine, premises, &data, &vars, &mut ws, &ctx).unwrap();
        assert!(result.discovery.evidence.graph.has_edge(id(0), id(1)));
        assert!(!result.discovery.evidence.graph.has_edge(id(2), id(3)));
        assert!(result.interpretation.contains("inducing path"));
        assert!(
            discover_cyclic_fci(
                &engine,
                CyclicFciPremises { sigma_faithful: false, ..premises },
                &data,
                &vars,
                &mut ws,
                &ctx
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod exhaustive_tests {
    use super::*;
    use antecedent_graph::Endpoint;

    #[test]
    fn all_three_node_cyclic_mixed_oracles_have_sound_skeletons_and_ancestry_marks() {
        let data = crate::test_tables::tabular_n(3, 30);
        let id = DenseNodeId::from_raw;
        let variables: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        let premises = CyclicFciPremises {
            simple_scm: true,
            sigma_faithful: true,
            independent_equilibrium_replicates: true,
        };
        for mask in 0_u16..512 {
            let mut graph = CyclicMixedGraph::with_variables(3);
            let mut ancestor = [[false; 3]; 3];
            let mut bit = 0;
            for (a, row) in ancestor.iter_mut().enumerate() {
                for (b, reachable) in row.iter_mut().enumerate() {
                    if a != b {
                        if mask & (1 << bit) != 0 {
                            graph.insert_directed(id(a as u32), id(b as u32)).unwrap();
                            *reachable = true;
                        }
                        bit += 1;
                    }
                }
            }
            for a in 0..3 {
                for b in a + 1..3 {
                    if mask & (1 << bit) != 0 {
                        graph.insert_bidirected(id(a as u32), id(b as u32)).unwrap();
                    }
                    bit += 1;
                }
            }
            for k in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        ancestor[a][b] |= ancestor[a][k] && ancestor[k][b];
                    }
                }
            }
            let oracle = SigmaCiOracle::new(graph.clone(), (0..3).map(id).collect()).unwrap();
            let engine = Fci::new().with_ci(Arc::new(oracle));
            let result = discover_cyclic_fci(
                &engine,
                premises,
                &data,
                &variables,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            for conclusion in cyclic_pag_conclusions(
                &result.discovery.evidence.graph,
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
            {
                let a = conclusion.source as usize;
                let b = conclusion.target as usize;
                match conclusion.relation.as_str() {
                    "ancestor" => assert!(ancestor[a][b], "ancestor mask {mask}"),
                    "not_ancestor" => assert!(!ancestor[a][b], "nonancestor mask {mask}"),
                    "direct_edge" => assert!(
                        graph.directed_edge_pairs_sorted().contains(&(a as u32, b as u32)),
                        "direct mask {mask}"
                    ),
                    "no_direct_edge" => assert!(
                        !graph.directed_edge_pairs_sorted().contains(&(a as u32, b as u32)),
                        "no direct mask {mask}"
                    ),
                    "no_confounder" => assert!(
                        !graph
                            .bidirected_edge_pairs_sorted()
                            .contains(&(a.min(b) as u32, a.max(b) as u32)),
                        "confounding mask {mask}"
                    ),
                    _ => panic!("unknown conclusion"),
                }
            }
            let pag = &result.discovery.evidence.graph;
            let mut ws = SigmaSeparationWorkspace::default();
            for a in 0..3 {
                for b in a + 1..3 {
                    let remaining = 3 - a - b;
                    let separable = [vec![], vec![id(remaining as u32)]].iter().any(|z| {
                        sigma_separates(&graph, id(a as u32), id(b as u32), z, &mut ws)
                            .unwrap()
                            .status
                            .is_separated()
                    });
                    assert_eq!(
                        pag.has_edge(id(a as u32), id(b as u32)),
                        !separable,
                        "mask {mask} pair {a},{b}"
                    );
                    if let Some(edge) = pag.edge_between(id(a as u32), id(b as u32)) {
                        for (at, from, to) in [(edge.at_a, a, b), (edge.at_b, b, a)] {
                            match at {
                                Endpoint::Arrow => {
                                    assert!(!ancestor[from][to], "arrow mask {mask}, {from},{to}")
                                }
                                Endpoint::Tail => {
                                    assert!(ancestor[from][to], "tail mask {mask}, {from},{to}")
                                }
                                Endpoint::Circle => {}
                                Endpoint::Conflict => panic!("conflict on exact oracle {mask}"),
                            }
                        }
                    }
                }
            }
        }
    }
}

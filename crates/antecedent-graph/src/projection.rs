//! Latent projection from DAGs onto ADMGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use crate::admg::Admg;
use crate::dag::Dag;
use crate::dsep::DSeparationWorkspace;
use crate::error::GraphError;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

/// Project a DAG onto an observed subset, producing an ADMG.
///
/// Dense indices follow `observed` order; original variable identities are retained.
///
/// Directed edges: observed→observed paths whose internal nodes are all latent.
/// Bidirected edges: pairs of observed nodes that share a latent common ancestor
/// reachable via latent-only directed paths (including latent parents).
///
/// # Errors
///
/// Unknown or duplicate observed node ids.
pub fn latent_project(dag: &Dag, observed: &[DenseNodeId]) -> Result<Admg, GraphError> {
    for &o in observed {
        if o.as_usize() >= dag.node_count() {
            return Err(GraphError::UnknownNode { id: o.raw() });
        }
    }
    let mut observed_set = BitSet::with_len(dag.node_count());
    let mut admg = Admg::empty();
    for &o in observed {
        if observed_set.contains(o) {
            return Err(GraphError::InvalidEndpoints {
                message: "latent projection observed nodes must be unique",
            });
        }
        observed_set.insert(o);
        admg.add_node(dag.nodes()[o.as_usize()])?;
    }
    // Map original dense id → projected dense id.
    let mut map = vec![None; dag.node_count()];
    for (i, &o) in observed.iter().enumerate() {
        map[o.as_usize()] = Some(DenseNodeId::try_from_usize(i)?);
    }

    let k = observed.len();
    let mut ws = GraphWorkspace::default();
    let mut reached = BitSet::with_len(k);
    // Directed projection edges: one latent-only search per observed source.
    for (i, &u) in observed.iter().enumerate() {
        observed_reachable_via_latents(dag, u, &observed_set, &map, &mut ws, &mut reached)?;
        for j in 0..k {
            if i == j || !reached.contains(DenseNodeId::try_from_usize(j)?) {
                continue;
            }
            let from = DenseNodeId::try_from_usize(i)?;
            let to = DenseNodeId::try_from_usize(j)?;
            // Longer latent paths can propose edges that cycle with shorter ones already
            // inserted; skip only those conflicts.
            match admg.insert_directed(from, to) {
                Ok(()) | Err(GraphError::DuplicateEdge { .. }) => {}
                Err(GraphError::Cycle { .. }) => {
                    // Silent skip would drop a required projection edge and can
                    // break d/m-separation equivalence — fail closed instead.
                    return Err(GraphError::InvalidEndpoints {
                        message: "latent projection directed edge conflicts with an existing path (cycle); refuse incomplete projection",
                    });
                }
                Err(e) => return Err(e),
            }
        }
    }

    // Bidirected: observed u, v are joined iff some latent L reaches both by latent-only
    // directed paths (a divergent path with every interior node latent has a last common
    // node, which is such an L). One search per latent gives its reachable observed set S_L;
    // every pair inside S_L is joined, so OR S_L into the row of each of its members.
    let mut joined: Vec<BitSet> = (0..k).map(|_| BitSet::with_len(k)).collect();
    for l in 0..dag.node_count() {
        let l = DenseNodeId::try_from_usize(l)?;
        if observed_set.contains(l) {
            continue;
        }
        observed_reachable_via_latents(dag, l, &observed_set, &map, &mut ws, &mut reached)?;
        for member in reached.to_dense_ids() {
            joined[member.as_usize()].union_with(&reached);
        }
    }
    for (i, row) in joined.iter().enumerate() {
        for j in (i + 1)..k {
            if !row.contains(DenseNodeId::try_from_usize(j)?) {
                continue;
            }
            let a = DenseNodeId::try_from_usize(i)?;
            let b = DenseNodeId::try_from_usize(j)?;
            match admg.insert_bidirected(a, b) {
                Ok(()) | Err(GraphError::Cycle { .. } | GraphError::DuplicateEdge { .. }) => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(admg)
}

/// Observed nodes reachable from `start` by a directed path whose *internal* nodes are all
/// latent (an observed node ends the path). Fills `out` (over projected positions, via `map`).
fn observed_reachable_via_latents(
    dag: &Dag,
    start: DenseNodeId,
    observed: &BitSet,
    map: &[Option<DenseNodeId>],
    ws: &mut GraphWorkspace,
    out: &mut BitSet,
) -> Result<(), GraphError> {
    out.clear();
    ws.prepare(dag.node_count());
    ws.frontier.push(start);
    ws.visited.insert(start);
    while let Some(n) = ws.frontier.pop() {
        for &c in dag.children(n) {
            if observed.contains(c) {
                let pos = map[c.as_usize()].ok_or(GraphError::UnknownNode { id: c.raw() })?;
                out.insert(pos);
            } else if !ws.visited.contains(c) {
                ws.visited.insert(c);
                ws.frontier.push(c);
            }
        }
    }
    Ok(())
}

/// Check that m-separation on the projected ADMG agrees with d-separation on the
/// original DAG for queries restricted to observed nodes .
///
/// # Errors
///
/// Graph errors from separation APIs.
pub fn projection_preserves_msep_sample(
    dag: &Dag,
    observed: &[DenseNodeId],
    queries: &[(DenseNodeId, DenseNodeId, Vec<DenseNodeId>)],
) -> Result<bool, GraphError> {
    let admg = latent_project(dag, observed)?;
    let mut map = vec![None; dag.node_count()];
    for (i, &o) in observed.iter().enumerate() {
        map[o.as_usize()] = Some(DenseNodeId::try_from_usize(i)?);
    }
    let mut dws = DSeparationWorkspace::default();
    let mut mws = DSeparationWorkspace::default();
    for (x, y, z) in queries {
        let dx = dag.is_d_separated(*x, *y, z, &mut dws)?;
        let mx = map[x.as_usize()].ok_or(GraphError::UnknownNode { id: x.raw() })?;
        let my = map[y.as_usize()].ok_or(GraphError::UnknownNode { id: y.raw() })?;
        let mz: Result<Vec<_>, _> = z
            .iter()
            .map(|v| map[v.as_usize()].ok_or(GraphError::UnknownNode { id: v.raw() }))
            .collect();
        let mz = mz?;
        let mx_sep = admg.is_m_separated(mx, my, &mz, &mut mws)?;
        if dx != mx_sep {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]
mod tests {
    use super::*;
    use crate::dag::Dag;

    #[test]
    fn projects_latent_common_cause_to_bidirected() {
        // L → X, L → Y; observe X,Y
        let mut dag = Dag::with_variables(3);
        let l = DenseNodeId::from_raw(0);
        let x = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        dag.insert_directed(l, x).unwrap();
        dag.insert_directed(l, y).unwrap();
        let admg = latent_project(&dag, &[x, y]).unwrap();
        assert!(
            admg.bidirected_neighbors(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(1))
        );
        let mut ws = DSeparationWorkspace::default();
        assert!(
            !admg
                .is_m_separated(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), &[], &mut ws)
                .unwrap()
        );
        assert!(projection_preserves_msep_sample(&dag, &[x, y], &[(x, y, vec![])]).unwrap());
    }

    #[test]
    fn projects_latent_chain_to_directed() {
        // X → L → Y
        let mut dag = Dag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let l = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        dag.insert_directed(x, l).unwrap();
        dag.insert_directed(l, y).unwrap();
        let admg = latent_project(&dag, &[x, y]).unwrap();
        assert!(admg.children(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(1)));
    }

    #[test]
    fn latent_projection_cycle_conflict_errors() {
        // Latent skeleton X → L1 → Y → L2 → X: projecting onto {X, Y} proposes both
        // X → Y and Y → X. Built with unchecked edges (invalid as a DAG, but exercises
        // fail-closed refusal when projection would introduce a directed cycle).
        let mut dag = Dag::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let l1 = DenseNodeId::from_raw(2);
        let l2 = DenseNodeId::from_raw(3);
        dag.insert_directed_unchecked(x, l1);
        dag.insert_directed_unchecked(l1, y);
        dag.insert_directed_unchecked(y, l2);
        dag.insert_directed_unchecked(l2, x);
        let err = latent_project(&dag, &[x, y]).unwrap_err();
        assert!(matches!(err, GraphError::InvalidEndpoints { .. }));
    }
    #[test]
    fn projection_retains_variable_identity_in_observed_order() {
        use antecedent_core::{NodeRef, VariableId};
        let mut dag = Dag::empty();
        let x = dag.add_node(NodeRef::Static(VariableId::from_raw(42))).unwrap();
        let latent = dag.add_node(NodeRef::Static(VariableId::from_raw(7))).unwrap();
        let y = dag.add_node(NodeRef::Static(VariableId::from_raw(99))).unwrap();
        dag.insert_directed(x, latent).unwrap();
        dag.insert_directed(latent, y).unwrap();
        let projected = latent_project(&dag, &[y, x]).unwrap();
        assert_eq!(projected.nodes(), &[dag.nodes()[y.as_usize()], dag.nodes()[x.as_usize()]]);
        assert_eq!(projected.children(DenseNodeId::from_raw(1)), &[DenseNodeId::from_raw(0)]);
    }

    /// Independent oracle: is there a directed path `from ⇝ to` whose internal nodes are all
    /// latent (plain recursive path enumeration on adjacency-matrix form)?
    fn latent_path(
        adj: &[Vec<bool>],
        obs: &[bool],
        from: usize,
        to: usize,
        seen: &mut [bool],
    ) -> bool {
        for c in 0..adj.len() {
            if !adj[from][c] {
                continue;
            }
            if c == to {
                return true;
            }
            if !obs[c] && !seen[c] {
                seen[c] = true;
                if latent_path(adj, obs, c, to, seen) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn projection_matches_path_enumeration_on_random_dags() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        for _ in 0..200 {
            let n = 4 + (next() % 6) as usize;
            let mut dag = Dag::with_variables(n as u32);
            let mut adj = vec![vec![false; n]; n];
            for a in 0..n {
                for b in (a + 1)..n {
                    if next() % 100 < 30 {
                        dag.insert_directed(
                            DenseNodeId::from_raw(a as u32),
                            DenseNodeId::from_raw(b as u32),
                        )
                        .unwrap();
                        adj[a][b] = true;
                    }
                }
            }
            let obs: Vec<bool> = (0..n).map(|_| next() % 100 < 55).collect();
            let observed: Vec<DenseNodeId> =
                (0..n).filter(|&i| obs[i]).map(|i| DenseNodeId::from_raw(i as u32)).collect();
            if observed.is_empty() {
                continue;
            }
            let admg = latent_project(&dag, &observed).unwrap();
            for (pi, &u) in observed.iter().enumerate() {
                for (pj, &v) in observed.iter().enumerate() {
                    if pi == pj {
                        continue;
                    }
                    let (ui, vi) = (u.as_usize(), v.as_usize());
                    let want_dir = latent_path(&adj, &obs, ui, vi, &mut vec![false; n]);
                    let p_i = DenseNodeId::from_raw(pi as u32);
                    let p_j = DenseNodeId::from_raw(pj as u32);
                    assert_eq!(admg.children(p_i).contains(&p_j), want_dir, "{u:?}->{v:?}");
                    let want_bi = (0..n).filter(|&l| !obs[l]).any(|l| {
                        latent_path(&adj, &obs, l, ui, &mut vec![false; n])
                            && latent_path(&adj, &obs, l, vi, &mut vec![false; n])
                    });
                    assert_eq!(
                        admg.bidirected_neighbors(p_i).contains(&p_j),
                        want_bi,
                        "{u:?}<->{v:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn admg_m_separation_equals_d_separation_in_the_latent_dag() {
        // Oracle: d-separation in the DAG with explicit latents (itself checked against a
        // path-enumeration oracle in dsep_tests), for every pair and every conditioning set
        // over the observed nodes.
        let mut state = 0xA076_1D64_78BD_642F_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut connected = 0usize;
        for _ in 0..60 {
            let n = 5 + (next() % 3) as usize;
            let mut dag = Dag::with_variables(n as u32);
            for a in 0..n {
                for b in (a + 1)..n {
                    if next() % 100 < 35 {
                        dag.insert_directed(
                            DenseNodeId::from_raw(a as u32),
                            DenseNodeId::from_raw(b as u32),
                        )
                        .unwrap();
                    }
                }
            }
            let observed: Vec<DenseNodeId> = (0..n)
                .filter(|_| next() % 100 < 65)
                .map(|i| DenseNodeId::from_raw(i as u32))
                .collect();
            if observed.len() < 2 {
                continue;
            }
            let admg = latent_project(&dag, &observed).unwrap();
            let (mut dws, mut mws) =
                (DSeparationWorkspace::default(), DSeparationWorkspace::default());
            let k = observed.len();
            for i in 0..k {
                for j in (i + 1)..k {
                    let rest: Vec<usize> = (0..k).filter(|&m| m != i && m != j).collect();
                    for mask in 0..(1usize << rest.len()) {
                        let z_dag: Vec<DenseNodeId> = rest
                            .iter()
                            .enumerate()
                            .filter(|(bit, _)| (mask >> bit) & 1 == 1)
                            .map(|(_, &m)| observed[m])
                            .collect();
                        let z_admg: Vec<DenseNodeId> = rest
                            .iter()
                            .enumerate()
                            .filter(|(bit, _)| (mask >> bit) & 1 == 1)
                            .map(|(_, &m)| DenseNodeId::from_raw(m as u32))
                            .collect();
                        let want =
                            dag.is_d_separated(observed[i], observed[j], &z_dag, &mut dws).unwrap();
                        let got = admg
                            .is_m_separated(
                                DenseNodeId::from_raw(i as u32),
                                DenseNodeId::from_raw(j as u32),
                                &z_admg,
                                &mut mws,
                            )
                            .unwrap();
                        assert_eq!(got, want, "pair ({i},{j}) mask {mask:b}");
                        connected += usize::from(!want);
                    }
                }
            }
        }
        assert!(connected > 50, "the oracle must see connected queries too ({connected})");
    }

    #[test]
    fn projection_rejects_duplicate_observed_nodes() {
        let dag = Dag::with_variables(1);
        let x = DenseNodeId::from_raw(0);
        assert!(matches!(latent_project(&dag, &[x, x]), Err(GraphError::InvalidEndpoints { .. })));
    }
}

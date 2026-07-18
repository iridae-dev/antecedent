//! Latent projection from DAGs onto ADMGs (DESIGN.md §6.5 / §28.1).
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
/// Directed edges: observed→observed paths whose internal nodes are all latent.
/// Bidirected edges: pairs of observed nodes that share a latent common ancestor
/// reachable via latent-only directed paths (including latent parents).
///
/// # Errors
///
/// Unknown observed node ids.
pub fn latent_project(dag: &Dag, observed: &[DenseNodeId]) -> Result<Admg, GraphError> {
    for &o in observed {
        if o.as_usize() >= dag.node_count() {
            return Err(GraphError::UnknownNode { id: o.raw() });
        }
    }
    let mut observed_set = BitSet::with_len(dag.node_count());
    for &o in observed {
        observed_set.insert(o);
    }
    let k = u32::try_from(observed.len()).map_err(|_| GraphError::TooManyNodes)?;
    let mut admg = Admg::with_variables(k);
    // Map original dense id → projected dense id.
    let mut map = vec![None; dag.node_count()];
    for (i, &o) in observed.iter().enumerate() {
        map[o.as_usize()] = Some(DenseNodeId::try_from_usize(i)?);
    }

    let mut ws = GraphWorkspace::default();
    // Directed projection edges.
    for (i, &u) in observed.iter().enumerate() {
        for (j, &v) in observed.iter().enumerate() {
            if i == j {
                continue;
            }
            if directed_path_through_latents(dag, u, v, &observed_set, &mut ws) {
                let from = map[u.as_usize()].expect("mapped");
                let to = map[v.as_usize()].expect("mapped");
                // Longer latent paths can propose edges that cycle with shorter ones already
                // inserted; skip only those conflicts.
                match admg.insert_directed(from, to) {
                    Ok(()) => {}
                    Err(GraphError::Cycle { .. } | GraphError::DuplicateEdge { .. }) => {}
                    Err(e) => return Err(e),
                }
            }
        }
    }

    // Bidirected: shared latent ancestor with latent-only paths to both.
    for i in 0..observed.len() {
        for j in (i + 1)..observed.len() {
            let u = observed[i];
            let v = observed[j];
            if share_latent_common_ancestor(dag, u, v, &observed_set, &mut ws)? {
                let a = map[u.as_usize()].expect("mapped");
                let b = map[v.as_usize()].expect("mapped");
                match admg.insert_bidirected(a, b) {
                    Ok(()) => {}
                    Err(GraphError::Cycle { .. } | GraphError::DuplicateEdge { .. }) => {}
                    Err(e) => return Err(e),
                }
            }
        }
    }
    Ok(admg)
}

fn is_latent(id: DenseNodeId, observed: &BitSet) -> bool {
    !observed.contains(id)
}

/// Directed path u ⇝ v with all *internal* nodes latent (endpoints may be observed).
fn directed_path_through_latents(
    dag: &Dag,
    u: DenseNodeId,
    v: DenseNodeId,
    observed: &BitSet,
    ws: &mut GraphWorkspace,
) -> bool {
    if u == v {
        return false;
    }
    // Direct edge.
    if dag.children(u).contains(&v) {
        return true;
    }
    ws.prepare(dag.node_count());
    ws.frontier.push(u);
    ws.visited.insert(u);
    while let Some(n) = ws.frontier.pop() {
        for &c in dag.children(n) {
            if c == v {
                return true;
            }
            // May only traverse through latents (not other observed).
            if observed.contains(c) {
                continue;
            }
            if !ws.visited.contains(c) {
                ws.visited.insert(c);
                ws.frontier.push(c);
            }
        }
    }
    false
}

fn share_latent_common_ancestor(
    dag: &Dag,
    u: DenseNodeId,
    v: DenseNodeId,
    observed: &BitSet,
    ws: &mut GraphWorkspace,
) -> Result<bool, GraphError> {
    let n = dag.node_count();
    for i in 0..n {
        let l = DenseNodeId::try_from_usize(i)?;
        if !is_latent(l, observed) {
            continue;
        }
        // Latent L reaches both via paths that do not pass through other observed
        // (except the targets).
        if reaches_observed_via_latents(dag, l, u, observed, ws)
            && reaches_observed_via_latents(dag, l, v, observed, ws)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn reaches_observed_via_latents(
    dag: &Dag,
    from: DenseNodeId,
    target: DenseNodeId,
    observed: &BitSet,
    ws: &mut GraphWorkspace,
) -> bool {
    if from == target {
        return true;
    }
    ws.prepare(dag.node_count());
    ws.frontier.push(from);
    ws.visited.insert(from);
    while let Some(n) = ws.frontier.pop() {
        for &c in dag.children(n) {
            if c == target {
                return true;
            }
            if observed.contains(c) {
                continue;
            }
            if !ws.visited.contains(c) {
                ws.visited.insert(c);
                ws.frontier.push(c);
            }
        }
    }
    false
}

/// Check that m-separation on the projected ADMG agrees with d-separation on the
/// original DAG for queries restricted to observed nodes (DESIGN §28.1).
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
}

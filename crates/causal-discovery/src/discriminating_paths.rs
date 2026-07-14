//! Discriminating paths for LPCMCI PAG orientation (DESIGN.md §13.6).
//!
//! Explicit module — not embedded in a single orientation loop.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]

use causal_graph::{DenseNodeId, Endpoint, TemporalPag};

/// A discriminating path `a … c *-* b` used by LPCMCI orientation rules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscriminatingPath {
    /// Path nodes from `a` to `b` inclusive (`len >= 3`).
    pub nodes: Vec<DenseNodeId>,
}

/// Find discriminating paths ending at edge `{c,b}` with circle marks, bounded.
#[must_use]
pub fn find_discriminating_paths(
    pag: &TemporalPag,
    max_paths: usize,
    max_len: usize,
) -> Vec<DiscriminatingPath> {
    let mut out = Vec::new();
    if max_paths == 0 || max_len < 3 {
        return out;
    }
    let n = pag.node_count();
    for i in 0..n {
        let b = DenseNodeId::from_raw(i as u32);
        for (c, at_b, at_c) in pag.neighbors(b) {
            // Need circle at c on edge c–b (uncertain collider status).
            let mark_at_c = at_c;
            let _ = at_b;
            if !matches!(mark_at_c, Endpoint::Circle) {
                continue;
            }
            // Search paths a → … → c of definite directed edges, then c *-* b.
            let mut stack = vec![vec![c]];
            while let Some(path) = stack.pop() {
                if out.len() >= max_paths {
                    return out;
                }
                let last = *path.last().expect("nonempty");
                if path.len() >= 2 {
                    // path is a…c; append b
                    let mut full = path.clone();
                    full.push(b);
                    if full.len() >= 3 && full.len() <= max_len {
                        out.push(DiscriminatingPath { nodes: full });
                        if out.len() >= max_paths {
                            return out;
                        }
                    }
                }
                if path.len() >= max_len - 1 {
                    continue;
                }
                for (pred, at_self, at_nbr) in pag.neighbors(last) {
                    // Walk backwards along definite directed pred → last
                    if !matches!((at_nbr, at_self), (Endpoint::Tail, Endpoint::Arrow))
                        && !matches!((at_self, at_nbr), (Endpoint::Arrow, Endpoint::Tail))
                    {
                        // Accept definite directed into `last` from `pred`: mark at last is Arrow, at pred is Tail
                        let into_last = if pred == last {
                            continue;
                        } else {
                            // edge between pred and last: at_self is mark at last, at_nbr at pred when iterating from last
                            matches!(at_self, Endpoint::Arrow) && matches!(at_nbr, Endpoint::Tail)
                        };
                        if !into_last {
                            continue;
                        }
                    } else {
                        // neighbors() from last: at_self at last, at_neighbor at pred
                        let into_last =
                            matches!(at_self, Endpoint::Arrow) && matches!(at_nbr, Endpoint::Tail);
                        if !into_last {
                            continue;
                        }
                    }
                    if path.contains(&pred) {
                        continue;
                    }
                    let mut next = path.clone();
                    next.insert(0, pred);
                    stack.push(next);
                }
            }
        }
    }
    out
}

/// Whether the discriminating path implies a collider (or non-collider) orientation at `c`.
#[must_use]
pub fn discriminating_implies_collider(path: &DiscriminatingPath, sepset_contains_b: bool) -> bool {
    // Standard FCI/LPCMCI: if b ∈ sepset(a,c) then non-collider at c; else collider.
    let _ = path;
    !sepset_contains_b
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{Lag, VariableId};
    use causal_graph::TemporalPag;

    #[test]
    fn finds_path_on_chain() {
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let c = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(0)).unwrap();
        g.insert_directed(a, c).unwrap();
        g.insert_circle_arrow(c, b).unwrap();
        let paths = find_discriminating_paths(&g, 16, 8);
        assert!(!paths.is_empty());
    }
}

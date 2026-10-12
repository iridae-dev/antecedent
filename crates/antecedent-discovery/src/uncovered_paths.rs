//! Uncovered potentially directed paths for FCI R8–R10 / LPCMCI primes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{DenseNodeId, Endpoint};

use crate::orientation::PagOps;

/// Edge pattern: left endpoint, right endpoint (from `from`'s perspective toward `to`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct EndpointPattern {
    /// Mark at the path-forward source.
    pub at_from: Option<Endpoint>,
    /// Mark at the path-forward target (`None` = any).
    pub at_to: Option<Endpoint>,
}

impl EndpointPattern {
    /// Tail→Arrow (`→`).
    #[must_use]
    pub const fn directed() -> Self {
        Self { at_from: Some(Endpoint::Tail), at_to: Some(Endpoint::Arrow) }
    }

    /// Circle→Arrow (`o→`).
    #[must_use]
    pub const fn circle_arrow() -> Self {
        Self { at_from: Some(Endpoint::Circle), at_to: Some(Endpoint::Arrow) }
    }

    /// Circle–Circle (`o–o`).
    #[must_use]
    pub const fn circle_circle() -> Self {
        Self { at_from: Some(Endpoint::Circle), at_to: Some(Endpoint::Circle) }
    }
}

fn marks_from_to<G: PagOps>(
    graph: &G,
    from: DenseNodeId,
    to: DenseNodeId,
) -> Option<(Endpoint, Endpoint)> {
    let e = graph.edge_between(from, to)?;
    if e.a == from { Some((e.at_a, e.at_b)) } else { Some((e.at_b, e.at_a)) }
}

const AFTER_CIRCLE_CIRCLE: &[EndpointPattern] = &[
    EndpointPattern::circle_circle(),
    EndpointPattern::circle_arrow(),
    EndpointPattern::directed(),
];
const AFTER_DIRECTED: &[EndpointPattern] = &[EndpointPattern::directed()];

fn next_allowed_after(at_from: Endpoint, at_to: Endpoint) -> &'static [EndpointPattern] {
    if matches!((at_from, at_to), (Endpoint::Circle, Endpoint::Circle)) {
        AFTER_CIRCLE_CIRCLE
    } else {
        AFTER_DIRECTED
    }
}

fn matches_pattern(at_from: Endpoint, at_to: Endpoint, pat: EndpointPattern) -> bool {
    if let Some(p) = pat.at_from {
        if at_from != p {
            return false;
        }
    }
    if let Some(p) = pat.at_to {
        if at_to != p {
            return false;
        }
    }
    true
}

/// Find uncovered potentially directed paths from `start` to `end` (length ≥ 3 nodes).
///
/// Uncovered: no edge between non-consecutive nodes on the path (checked locally:
/// `path[i]` not adjacent to `path[i+2]`).
///
/// Returns `(paths, truncated)` where `truncated` is true if the search stopped because
/// `max_paths` was reached, `max_len` pruned a path that had not reached `end`, or
/// the budget was zero (`max_paths == 0`).
#[must_use]
pub(crate) fn uncovered_pd_paths_with_budget<G: PagOps>(
    graph: &G,
    start: DenseNodeId,
    end: DenseNodeId,
    initial: &[EndpointPattern],
    max_paths: usize,
    max_len: usize,
) -> (Vec<Vec<DenseNodeId>>, bool) {
    if max_paths == 0 {
        return (Vec::new(), true);
    }
    if start == end || max_len < 3 {
        return (Vec::new(), max_len < 3 && start != end);
    }

    /// Depth-first search state: the target, the budgets and the paths found so far.
    #[allow(clippy::items_after_statements, reason = "helper defined next to its single use")]
    struct Search {
        end: DenseNodeId,
        max_paths: usize,
        max_len: usize,
        out: Vec<Vec<DenseNodeId>>,
        truncated: bool,
    }

    #[allow(clippy::items_after_statements, reason = "helper defined next to its single use")]
    fn search<G: PagOps>(
        graph: &G,
        state: &mut Search,
        path: &mut Vec<DenseNodeId>,
        allowed: &[EndpointPattern],
    ) {
        let cur = *path.last().expect("non-empty");
        if cur == state.end {
            if path.len() >= 3 {
                state.out.push(path.clone());
            }
            return;
        }
        if path.len() >= state.max_len {
            for (next, _, _) in graph.neighbors(cur) {
                if !path.contains(&next) {
                    state.truncated = true;
                    break;
                }
            }
            return;
        }
        let nbrs: Vec<_> = graph.neighbors(cur).into_iter().map(|(n, _, _)| n).collect();
        for next in nbrs {
            if state.out.len() >= state.max_paths {
                state.truncated = true;
                return;
            }
            if path.contains(&next) {
                continue;
            }
            if path.len() >= 2 {
                let prev = path[path.len() - 2];
                if graph.has_edge(prev, next) {
                    continue; // not uncovered
                }
            }
            let Some((at_from, at_to)) = marks_from_to(graph, cur, next) else {
                continue;
            };
            if !allowed.iter().any(|p| matches_pattern(at_from, at_to, *p)) {
                continue;
            }
            path.push(next);
            search(graph, state, path, next_allowed_after(at_from, at_to));
            path.pop();
        }
    }

    let mut state = Search { end, max_paths, max_len, out: Vec::new(), truncated: false };
    let mut path = vec![start];
    let nbrs: Vec<_> = graph.neighbors(start).into_iter().map(|(n, _, _)| n).collect();
    for next in nbrs {
        if state.out.len() >= max_paths {
            state.truncated = true;
            break;
        }
        if next == end {
            continue; // need length ≥ 3
        }
        let Some((at_from, at_to)) = marks_from_to(graph, start, next) else {
            continue;
        };
        if !initial.iter().any(|p| matches_pattern(at_from, at_to, *p)) {
            continue;
        }
        path.push(next);
        search(graph, &mut state, &mut path, next_allowed_after(at_from, at_to));
        path.pop();
    }
    (state.out, state.truncated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::Pag;

    fn directed_chain4() -> Pag {
        let mut g = Pag::with_variables(4);
        let n = |i| DenseNodeId::from_raw(i);
        g.insert_directed(n(0), n(1)).unwrap();
        g.insert_directed(n(1), n(2)).unwrap();
        g.insert_directed(n(2), n(3)).unwrap();
        g
    }

    #[test]
    fn max_len_that_prunes_an_unfinished_path_is_truncated() {
        let g = directed_chain4();
        let a = DenseNodeId::from_raw(0);
        let d = DenseNodeId::from_raw(3);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        let initial = [EndpointPattern::directed()];
        let (paths, truncated) = uncovered_pd_paths_with_budget(&g, a, d, &initial, 8, 3);
        assert!(paths.is_empty());
        assert!(truncated, "max_len=3 must not claim a complete search of a 4-node path");
        let (paths, truncated) = uncovered_pd_paths_with_budget(&g, a, d, &initial, 8, 4);
        assert_eq!(paths, vec![vec![a, b, c, d]]);
        assert!(!truncated);
    }

    #[test]
    fn zero_path_budget_is_truncated() {
        let g = directed_chain4();
        let (_, truncated) = uncovered_pd_paths_with_budget(
            &g,
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(3),
            &[EndpointPattern::directed()],
            0,
            8,
        );
        assert!(truncated);
    }
}

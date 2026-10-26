//! Uncovered potentially directed paths for FCI R8–R10 / LPCMCI primes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::ControlFlow;

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

/// Effort bound on one uncovered-path search.
///
/// Deciding whether an uncovered potentially directed path exists is a search for a simple path
/// that avoids forbidden transitions (a triple whose end nodes are adjacent), which has no known
/// polynomial algorithm; the search is therefore exhaustive and bounded only by the number of
/// nodes it extends a path with. A search that runs out is reported, never silently complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PdPathBudget {
    /// Path extensions one search may make before it gives up.
    pub max_expansions: usize,
}

impl Default for PdPathBudget {
    fn default() -> Self {
        Self { max_expansions: 1 << 22 }
    }
}

/// Visit the uncovered potentially directed paths from `start` to `end` (length ≥ 3 nodes), in
/// depth-first order, until `visit` breaks.
///
/// Uncovered: no edge between non-consecutive nodes on the path (checked locally:
/// `path[i]` not adjacent to `path[i+2]`). Paths are simple and have any length.
///
/// Returns `true` when the search was cut short by `budget` before it finished and before
/// `visit` broke, so the paths visited are valid but possibly not all of them.
#[allow(clippy::items_after_statements, reason = "helpers defined next to their single use")]
pub(crate) fn for_each_uncovered_pd_path<G: PagOps>(
    graph: &G,
    start: DenseNodeId,
    end: DenseNodeId,
    initial: &[EndpointPattern],
    budget: PdPathBudget,
    mut visit: impl FnMut(&[DenseNodeId]) -> ControlFlow<()>,
) -> bool {
    if start == end {
        return false;
    }

    /// Depth-first search state: the target, what is left of the budget and the visitor.
    struct Search<'a> {
        end: DenseNodeId,
        remaining: usize,
        exhausted: bool,
        visit: &'a mut dyn FnMut(&[DenseNodeId]) -> ControlFlow<()>,
    }

    /// Extend `path` (ending at `cur`) by every allowed next node. `false` stops the search:
    /// the visitor broke or the budget ran out.
    fn extend<G: PagOps>(
        graph: &G,
        state: &mut Search<'_>,
        path: &mut Vec<DenseNodeId>,
        allowed: &[EndpointPattern],
    ) -> bool {
        let cur = *path.last().expect("non-empty");
        if cur == state.end {
            return !(path.len() >= 3 && (state.visit)(&path[..]).is_break());
        }
        let nbrs: Vec<_> = graph.neighbors(cur).into_iter().map(|(n, _, _)| n).collect();
        for next in nbrs {
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
            if state.remaining == 0 {
                state.exhausted = true;
                return false;
            }
            state.remaining -= 1;
            path.push(next);
            let go_on = extend(graph, state, path, next_allowed_after(at_from, at_to));
            path.pop();
            if !go_on {
                return false;
            }
        }
        true
    }

    let mut state = Search { end, remaining: budget.max_expansions, exhausted: false, visit: &mut visit };
    let mut path = vec![start];
    let nbrs: Vec<_> = graph.neighbors(start).into_iter().map(|(n, _, _)| n).collect();
    for next in nbrs {
        if next == end {
            continue; // need length ≥ 3
        }
        let Some((at_from, at_to)) = marks_from_to(graph, start, next) else {
            continue;
        };
        if !initial.iter().any(|p| matches_pattern(at_from, at_to, *p)) {
            continue;
        }
        if state.remaining == 0 {
            return true;
        }
        state.remaining -= 1;
        path.push(next);
        let go_on = extend(graph, &mut state, &mut path, next_allowed_after(at_from, at_to));
        path.pop();
        if !go_on {
            break;
        }
    }
    state.exhausted
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

    fn collect(
        g: &Pag,
        start: u32,
        end: u32,
        budget: PdPathBudget,
    ) -> (Vec<Vec<DenseNodeId>>, bool) {
        let mut paths = Vec::new();
        let exhausted = for_each_uncovered_pd_path(
            g,
            DenseNodeId::from_raw(start),
            DenseNodeId::from_raw(end),
            &[EndpointPattern::directed()],
            budget,
            |path| {
                paths.push(path.to_vec());
                ControlFlow::Continue(())
            },
        );
        (paths, exhausted)
    }

    #[test]
    fn a_path_of_any_length_is_found_under_the_default_budget() {
        let (paths, exhausted) = collect(&directed_chain4(), 0, 3, PdPathBudget::default());
        let n = DenseNodeId::from_raw;
        assert_eq!(paths, vec![vec![n(0), n(1), n(2), n(3)]]);
        assert!(!exhausted);
    }

    #[test]
    fn a_budget_that_stops_the_search_is_reported() {
        let (paths, exhausted) =
            collect(&directed_chain4(), 0, 3, PdPathBudget { max_expansions: 2 });
        assert!(paths.is_empty());
        assert!(exhausted, "two extensions cannot reach a four-node path");
        let (_, exhausted) = collect(&directed_chain4(), 0, 3, PdPathBudget { max_expansions: 0 });
        assert!(exhausted);
    }

    #[test]
    fn a_visitor_that_breaks_ends_the_search_without_exhaustion() {
        let g = directed_chain4();
        let exhausted = for_each_uncovered_pd_path(
            &g,
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(3),
            &[EndpointPattern::directed()],
            PdPathBudget::default(),
            |_| ControlFlow::Break(()),
        );
        assert!(!exhausted);
    }
}

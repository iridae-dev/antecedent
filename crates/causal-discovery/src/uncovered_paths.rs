//! Uncovered potentially directed paths for FCI R8–R10 / LPCMCI primes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use causal_graph::{DenseNodeId, Endpoint};

use crate::orientation::PagOps;

/// Edge pattern: left endpoint, right endpoint (from `from`'s perspective toward `to`).
#[derive(Clone, Copy, Debug)]
pub struct EndpointPattern {
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

    /// `*→` (any at from, arrow at to).
    #[must_use]
    pub const fn into_arrow() -> Self {
        Self { at_from: None, at_to: Some(Endpoint::Arrow) }
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

/// Whether the edge from `from` toward `to` is potentially directed (`o→`, `→`, or `o–o`).
#[must_use]
pub fn is_potentially_directed<G: PagOps>(graph: &G, from: DenseNodeId, to: DenseNodeId) -> bool {
    let Some((at_from, at_to)) = marks_from_to(graph, from, to) else {
        return false;
    };
    matches!(
        (at_from, at_to),
        (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Circle, Endpoint::Arrow | Endpoint::Circle)
    )
}

/// Find uncovered potentially directed paths from `start` to `end` (length ≥ 3 nodes).
///
/// Uncovered: no edge between non-consecutive nodes on the path (checked locally:
/// `path[i]` not adjacent to `path[i+2]`).
///
/// Returns `(paths, truncated)` where `truncated` is true if the search stopped because
/// `max_paths` was reached while further candidates remained.
#[must_use]
pub fn uncovered_pd_paths_with_budget<G: PagOps>(
    graph: &G,
    start: DenseNodeId,
    end: DenseNodeId,
    initial: &[EndpointPattern],
    max_paths: usize,
    max_len: usize,
) -> (Vec<Vec<DenseNodeId>>, bool) {
    let mut out = Vec::new();
    let mut truncated = false;
    if start == end || max_paths == 0 || max_len < 3 {
        return (out, false);
    }
    #[allow(clippy::too_many_arguments, clippy::items_after_statements)]
    fn search<G: PagOps>(
        graph: &G,
        end: DenseNodeId,
        path: &mut Vec<DenseNodeId>,
        allowed: &[EndpointPattern],
        max_paths: usize,
        max_len: usize,
        out: &mut Vec<Vec<DenseNodeId>>,
        truncated: &mut bool,
    ) {
        let cur = *path.last().expect("non-empty");
        if cur == end {
            if path.len() >= 3 {
                out.push(path.clone());
            }
            return;
        }
        if path.len() >= max_len {
            return;
        }
        let nbrs: Vec<_> = graph.neighbors(cur).into_iter().map(|(n, _, _)| n).collect();
        for next in nbrs {
            if out.len() >= max_paths {
                *truncated = true;
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
            let next_allowed: &[EndpointPattern] =
                if matches!((at_from, at_to), (Endpoint::Circle, Endpoint::Circle)) {
                    &[
                        EndpointPattern::circle_circle(),
                        EndpointPattern::circle_arrow(),
                        EndpointPattern::directed(),
                    ]
                } else {
                    &[EndpointPattern::directed()]
                };
            path.push(next);
            search(graph, end, path, next_allowed, max_paths, max_len, out, truncated);
            path.pop();
        }
    }

    let mut path = vec![start];
    let nbrs: Vec<_> = graph.neighbors(start).into_iter().map(|(n, _, _)| n).collect();
    for next in nbrs {
        if out.len() >= max_paths {
            truncated = true;
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
        let next_allowed: &[EndpointPattern] =
            if matches!((at_from, at_to), (Endpoint::Circle, Endpoint::Circle)) {
                &[
                    EndpointPattern::circle_circle(),
                    EndpointPattern::circle_arrow(),
                    EndpointPattern::directed(),
                ]
            } else {
                &[EndpointPattern::directed()]
            };
        path.push(next);
        search(graph, end, &mut path, next_allowed, max_paths, max_len, &mut out, &mut truncated);
        path.pop();
    }
    (out, truncated)
}

/// Find uncovered potentially directed paths (paths only; ignores truncation).
#[must_use]
pub fn uncovered_pd_paths<G: PagOps>(
    graph: &G,
    start: DenseNodeId,
    end: DenseNodeId,
    initial: &[EndpointPattern],
    max_paths: usize,
    max_len: usize,
) -> Vec<Vec<DenseNodeId>> {
    uncovered_pd_paths_with_budget(graph, start, end, initial, max_paths, max_len).0
}

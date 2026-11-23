//! Exactly uniform draws of the maximal ancestral graphs a PAG represents, within a budget.
//!
//! The members of the Markov equivalence class of a PAG are the maximal ancestral graphs
//! (directed and bidirected edges, no selection) obtained by replacing each circle endpoint
//! with a tail or an arrowhead: every member shares the PAG's skeleton and invariant marks,
//! and different assignments give different graphs. A draw assigns every circle endpoint a
//! tail or an arrowhead with probability one half each, independently, and is accepted
//! exactly when the completion enumerator would accept it: the graph is ancestral and
//! maximal, keeps the PAG's unshielded colliders and non-colliders, and has the m-separation
//! model of a reference member. Each member is the image of exactly one assignment, so an
//! accepted draw is exactly uniform over the class (uniform over MAGs, not over the
//! latent-variable DAGs each MAG represents).
//!
//! Every assignment drawn and every m-separation query of the equivalence audit is charged
//! to a [`SearchBudget`]. A class whose acceptance rate is too low for the requested number of
//! members within the budget is refused with the budget's stop, never answered with fewer
//! members; a PAG too large for one m-separation model within the budget is refused before
//! anything is drawn. A locally valid draw with a different m-separation model shows that the
//! marks admit several classes, which the enumerator refuses too.
//!
//! The class size is not counted: the distinct members seen are a lower bound on it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use antecedent_core::{
    CanonicalWriter, ExecutionContext, SearchBudget, SearchLimits, SearchStop, derive_stream,
    splitmix64,
};

use crate::completion::{is_maximal_ancestral_graph, m_separation_signature};
use crate::mec_uniform::first_word;
use crate::pag::Pag;
use crate::types::{DenseNodeId, Endpoint};

const MAG_MEMBER_TAG: &str = "antecedent.graph.mag_member";

/// Why members of a PAG's class could not be drawn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagUniformError {
    /// The reference graph is not a maximal ancestral completion of the PAG.
    ReferenceNotMember,
    /// One m-separation model of the PAG does not fit the budget's operations or memory, so
    /// no draw can be audited for class membership.
    AuditUnaffordable,
    /// A locally valid draw has an m-separation model other than the reference's: the marks
    /// admit several Markov equivalence classes.
    AmbiguousClass,
    /// The budget stopped before the requested number of members was accepted.
    BudgetExhausted {
        /// Members accepted before the stop.
        accepted: usize,
        /// Endpoint assignments drawn before the stop.
        attempts: u64,
        /// The bound that stopped the draws.
        stop: SearchStop,
    },
}

impl fmt::Display for PagUniformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReferenceNotMember => {
                formatter.write_str("the reference graph is not a completion of the PAG")
            }
            Self::AuditUnaffordable => formatter
                .write_str("one m-separation model of the PAG does not fit the search budget"),
            Self::AmbiguousClass => {
                formatter.write_str("the PAG's marks admit several Markov equivalence classes")
            }
            Self::BudgetExhausted { accepted, attempts, stop } => write!(
                formatter,
                "the search budget stopped ({}) after {accepted} member(s) were accepted from \
                 {attempts} endpoint assignment(s)",
                stop.code()
            ),
        }
    }
}

impl std::error::Error for PagUniformError {}

/// Members drawn uniformly from a PAG's class.
#[derive(Clone, Debug)]
pub struct PagUniformSample {
    /// The accepted members, in draw order, with repetitions.
    pub members: Vec<Pag>,
    /// Endpoint assignments drawn, accepted or not.
    pub attempts: u64,
}

/// The 64-bit content fingerprint of a MAG member: its node count and every edge with its
/// two marks, in ascending node order.
#[must_use]
pub fn mag_fingerprint(mag: &Pag) -> u64 {
    let mut writer = CanonicalWriter::new(MAG_MEMBER_TAG);
    writer.word(mag.node_count() as u64);
    for i in 0..mag.node_count() {
        let a = DenseNodeId::from_index(i);
        let mut neighbours: Vec<(DenseNodeId, Endpoint, Endpoint)> =
            mag.neighbors(a).filter(|(b, _, _)| b.raw() > a.raw()).collect();
        neighbours.sort_unstable_by_key(|(b, _, _)| b.raw());
        for (b, at_a, at_b) in neighbours {
            writer
                .word(u64::from(a.raw()))
                .word(u64::from(b.raw()))
                .word(endpoint_code(at_a))
                .word(endpoint_code(at_b));
        }
    }
    first_word(&writer.finish())
}

const fn endpoint_code(endpoint: Endpoint) -> u64 {
    match endpoint {
        Endpoint::Tail => 0,
        Endpoint::Arrow => 1,
        Endpoint::Circle => 2,
        Endpoint::Conflict => 3,
    }
}

/// One circle endpoint: edge `(a, b)` and whether the circle is at `a`.
type CircleSite = (DenseNodeId, DenseNodeId, bool);

fn circle_sites(pag: &Pag) -> Vec<CircleSite> {
    let mut sites = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_index(i);
        let mut neighbours: Vec<(DenseNodeId, Endpoint, Endpoint)> =
            pag.neighbors(a).filter(|(b, _, _)| b.raw() > a.raw()).collect();
        neighbours.sort_unstable_by_key(|(b, _, _)| b.raw());
        for (b, at_a, at_b) in neighbours {
            if at_a == Endpoint::Circle {
                sites.push((a, b, true));
            }
            if at_b == Endpoint::Circle {
                sites.push((a, b, false));
            }
        }
    }
    sites
}

fn set_endpoint(graph: &mut Pag, (a, b, at_a_circle): CircleSite, endpoint: Endpoint) -> bool {
    let Some(edge) = graph.edge_between(a, b) else {
        return false;
    };
    let marks = if at_a_circle { (endpoint, edge.at_b) } else { (edge.at_a, endpoint) };
    graph.set_marks(a, b, marks.0, marks.1).is_ok()
}

/// Whether `mag` has exactly the PAG's skeleton and agrees with it on every unshielded triple:
/// a collider in the PAG (two arrowheads at the middle) is a collider in `mag`, and nothing
/// else is.
fn keeps_skeleton_and_colliders(pag: &Pag, mag: &Pag) -> bool {
    let n = pag.node_count();
    if mag.node_count() != n {
        return false;
    }
    for i in 0..n {
        let middle = DenseNodeId::from_index(i);
        let pag_marks: Vec<(DenseNodeId, Endpoint)> =
            pag.neighbors(middle).map(|(node, mark, _)| (node, mark)).collect();
        let mut mag_marks: Vec<(DenseNodeId, Endpoint)> =
            mag.neighbors(middle).map(|(node, mark, _)| (node, mark)).collect();
        if mag_marks.len() != pag_marks.len() {
            return false;
        }
        mag_marks.sort_unstable_by_key(|(node, _)| node.raw());
        let mag_mark = |node: DenseNodeId| -> Option<Endpoint> {
            mag_marks
                .binary_search_by_key(&node.raw(), |(neighbour, _)| neighbour.raw())
                .ok()
                .map(|at| mag_marks[at].1)
        };
        if pag_marks.iter().any(|(node, _)| mag_mark(*node).is_none()) {
            return false;
        }
        for (k, &(left, pag_left)) in pag_marks.iter().enumerate() {
            for &(right, pag_right) in &pag_marks[k + 1..] {
                if pag.has_edge(left, right) {
                    continue;
                }
                let pag_collider = pag_left == Endpoint::Arrow && pag_right == Endpoint::Arrow;
                let mag_collider = mag_mark(left) == Some(Endpoint::Arrow)
                    && mag_mark(right) == Some(Endpoint::Arrow);
                if pag_collider != mag_collider {
                    return false;
                }
            }
        }
    }
    true
}

/// m-separation queries in one model: every node pair under every conditioning subset of the
/// other nodes. `None` when the count does not fit `u128`.
fn signature_queries(n: usize) -> Option<u128> {
    let n = u128::try_from(n).ok()?;
    let pairs = n.checked_mul(n.saturating_sub(1))? / 2;
    let subsets = 1_u128.checked_shl(u32::try_from(n.saturating_sub(2)).ok()?)?;
    pairs.checked_mul(subsets)
}

/// Draw `draws` members of the class of `pag` uniformly and independently (with replacement)
/// from the stream `(seed, stream)`. `reference` is one member, whose m-separation model every
/// accepted draw must share. The same arguments give the same members in the same order.
///
/// # Errors
///
/// A reference that is not a completion of `pag`; a PAG whose m-separation model does not
/// fit `limits`; marks that admit several classes; a budget stop before `draws` members were
/// accepted (or a cancelled context), with the members accepted so far counted.
pub fn sample_pag_members(
    pag: &Pag,
    reference: &Pag,
    draws: usize,
    (seed, stream): (u64, u64),
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<PagUniformSample, PagUniformError> {
    if !circle_sites(reference).is_empty()
        || !keeps_skeleton_and_colliders(pag, reference)
        || !is_maximal_ancestral_graph(reference)
    {
        return Err(PagUniformError::ReferenceNotMember);
    }
    let sites = circle_sites(pag);
    if sites.is_empty() {
        return Ok(PagUniformSample { members: vec![reference.clone(); draws], attempts: 0 });
    }
    let n = pag.node_count();
    let queries = signature_queries(n).ok_or(PagUniformError::AuditUnaffordable)?;
    let words = queries.div_ceil(64);
    let bytes = words
        .checked_mul(16)
        .and_then(|b| u64::try_from(b).ok())
        .ok_or(PagUniformError::AuditUnaffordable)?;
    let mut budget = SearchBudget::new(limits, ctx).map_err(|receipt| {
        PagUniformError::BudgetExhausted { accepted: 0, attempts: 0, stop: receipt.stop }
    })?;
    if queries > u128::try_from(limits.operations).unwrap_or(u128::MAX)
        || bytes > budget.memory_limit_bytes()
    {
        return Err(PagUniformError::AuditUnaffordable);
    }
    let exhausted = |accepted: usize, attempts: u64, stop: SearchStop| {
        PagUniformError::BudgetExhausted { accepted, attempts, stop }
    };
    let reference_model = m_separation_signature(reference, || budget.charge(1, bytes))
        .map_err(|stop| exhausted(0, 0, stop))?;
    let mut state = derive_stream(seed, stream);
    let mut members = Vec::with_capacity(draws);
    let mut attempts: u64 = 0;
    while members.len() < draws {
        budget.charge(1, bytes).map_err(|stop| exhausted(members.len(), attempts, stop))?;
        attempts += 1;
        let mut graph = pag.clone();
        let mut assigned = true;
        let mut word = 0_u64;
        for (i, site) in sites.iter().enumerate() {
            if i % 64 == 0 {
                word = splitmix64(&mut state);
            }
            let endpoint =
                if (word >> (i % 64)) & 1 == 1 { Endpoint::Arrow } else { Endpoint::Tail };
            if !set_endpoint(&mut graph, *site, endpoint) {
                assigned = false;
                break;
            }
        }
        if !assigned
            || !keeps_skeleton_and_colliders(pag, &graph)
            || !is_maximal_ancestral_graph(&graph)
        {
            continue;
        }
        let model = m_separation_signature(&graph, || budget.charge(1, bytes))
            .map_err(|stop| exhausted(members.len(), attempts, stop))?;
        if model != reference_model {
            return Err(PagUniformError::AmbiguousClass);
        }
        members.push(graph);
    }
    Ok(PagUniformSample { members, attempts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(n: u32) -> Pag {
        let mut pag = Pag::with_variables(n);
        for i in 0..n - 1 {
            pag.insert_circle_circle(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1))
                .unwrap();
        }
        pag
    }

    fn reference_of(pag: &Pag) -> Pag {
        let mut sampler = crate::CompletionSampler::new(pag.clone(), 1).unwrap();
        sampler.next().unwrap().graph
    }

    #[test]
    fn a_circle_chain_draws_only_members_and_reproduces_by_seed() {
        let pag = chain(4);
        let reference = reference_of(&pag);
        let limits = SearchLimits { operations: 10_000_000, depth: usize::MAX };
        let ctx = ExecutionContext::for_tests(0);
        let first = sample_pag_members(&pag, &reference, 50, (3, 9), limits, &ctx).unwrap();
        let again = sample_pag_members(&pag, &reference, 50, (3, 9), limits, &ctx).unwrap();
        assert_eq!(first.members.len(), 50);
        let prints: Vec<u64> = first.members.iter().map(mag_fingerprint).collect();
        let reprinted: Vec<u64> = again.members.iter().map(mag_fingerprint).collect();
        assert_eq!(prints, reprinted);
        for member in &first.members {
            assert!(is_maximal_ancestral_graph(member));
            assert!(keeps_skeleton_and_colliders(&pag, member));
        }
    }

    #[test]
    fn a_starved_budget_is_a_typed_refusal() {
        let pag = chain(4);
        let reference = reference_of(&pag);
        let limits = SearchLimits { operations: 60, depth: usize::MAX };
        let ctx = ExecutionContext::for_tests(0);
        let refused = sample_pag_members(&pag, &reference, 1_000, (1, 1), limits, &ctx);
        assert!(matches!(
            refused,
            Err(PagUniformError::BudgetExhausted { stop: SearchStop::Operations, .. })
        ));
    }
}

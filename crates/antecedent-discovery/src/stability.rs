//! Sampling distribution of a frequentist structure search.
//!
//! A constraint-based or score-based search returns one structure and states no sampling
//! error for it. Re-running the search on resamples drawn under the study's declared
//! sampling design and tabulating what it returned gives that error as selection
//! frequencies: how often each pair of nodes is adjacent, how often each pair carries each
//! pair of endpoint marks, and how often the search returns each whole structure. The
//! frequencies are a bootstrap distribution of the search, not a posterior: nothing here
//! multiplies them by a prior, and they are not graph weights.
//!
//! The tabulation is independent of the search that produced the structures. The caller
//! draws the resamples, runs the search and passes the structures it returned; a resample
//! on which the search failed is counted as failed mass, never redrawn.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use crate::error::DiscoveryError;
use crate::selective::structure::{
    DiscoveredStructure, Mark, StructureClass, StructureEdge, StructureNode,
};

/// Share of the requested resamples that must have returned a structure for the
/// frequencies to be published.
const MIN_SUCCESS_SHARE: f64 = 0.5;

/// How often one endpoint-mark pair appeared on a node pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarkSelection {
    /// Mark at the first (smaller) node.
    pub at_a: Mark,
    /// Mark at the second node.
    pub at_b: Mark,
    /// Share of the successful resamples whose structure carried this pair of marks on the
    /// node pair.
    pub frequency: f64,
}

/// How often a node pair was adjacent.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeSelection {
    /// First node (the smaller).
    pub a: StructureNode,
    /// Second node.
    pub b: StructureNode,
    /// Share of the successful resamples whose structure joined the pair.
    pub frequency: f64,
    /// Whether the full-sample structure joins the pair.
    pub in_reference: bool,
    /// Frequency of each mark pair seen on the node pair, most frequent first.
    pub marks: Vec<MarkSelection>,
}

/// How often the search returned one whole structure.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureSelection {
    /// The structure, with the marks the search returned.
    pub structure: DiscoveredStructure,
    /// Successful resamples that returned it.
    pub count: u32,
    /// `count` over the successful resamples.
    pub frequency: f64,
}

/// The tabulated sampling distribution of one search.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureStability {
    /// Class of every structure tabulated.
    pub class: StructureClass,
    /// Resamples requested.
    pub requested: u32,
    /// Resamples on which the search returned a structure.
    pub succeeded: u32,
    /// Resamples on which the search failed: `requested - succeeded`.
    pub failed: u32,
    /// The structure the search returned on the full sample.
    pub reference: DiscoveredStructure,
    /// Share of the successful resamples that returned exactly the reference.
    pub reference_frequency: f64,
    /// Every node pair some structure joined, in canonical order.
    pub edges: Vec<EdgeSelection>,
    /// Every distinct structure returned, most frequent first.
    pub structures: Vec<StructureSelection>,
}

type Signature = Vec<(StructureNode, StructureNode, Mark, Mark)>;
/// Adjacent resamples of one node pair, and the same count split by endpoint marks.
type PairTally = (u32, BTreeMap<(Mark, Mark), u32>);

fn signature(structure: &DiscoveredStructure) -> Signature {
    structure.edges().iter().map(|e: &StructureEdge| (e.a, e.b, e.at_a, e.at_b)).collect()
}

impl StructureStability {
    /// Tabulate `resamples`, the structures the search returned on the resamples it
    /// succeeded on, against `reference`, the structure it returned on the full sample.
    ///
    /// Two structures are the same when they join the same node pairs with the same
    /// endpoint marks; the search-state middle mark does not separate them.
    ///
    /// # Errors
    ///
    /// No resample was requested, more structures than resamples were supplied, a structure
    /// of another class than the reference, fewer than two successful resamples, or fewer
    /// than half of the requested resamples succeeded.
    pub fn tabulate(
        reference: DiscoveredStructure,
        resamples: &[DiscoveredStructure],
        requested: u32,
    ) -> Result<Self, DiscoveryError> {
        let succeeded = u32::try_from(resamples.len()).unwrap_or(u32::MAX);
        if requested == 0 || succeeded > requested {
            return Err(DiscoveryError::unsupported(
                "a stability tabulation needs at least one requested resample and no more \
                 structures than resamples",
            ));
        }
        if resamples.iter().any(|s| s.class() != reference.class()) {
            return Err(DiscoveryError::unsupported(
                "every resampled structure must be of the class the full-sample search returned",
            ));
        }
        if succeeded < 2 || f64::from(succeeded) < MIN_SUCCESS_SHARE * f64::from(requested) {
            return Err(DiscoveryError::unsupported(
                "fewer than half of the requested resamples returned a structure; the \
                 frequencies would describe the resamples that happened to succeed",
            ));
        }
        let total = f64::from(succeeded);

        let mut pairs: BTreeMap<(StructureNode, StructureNode), PairTally> = BTreeMap::new();
        let mut whole: BTreeMap<Signature, (usize, u32)> = BTreeMap::new();
        for (index, structure) in resamples.iter().enumerate() {
            for edge in structure.edges() {
                let (adjacent, marks) = pairs.entry((edge.a, edge.b)).or_default();
                *adjacent += 1;
                *marks.entry((edge.at_a, edge.at_b)).or_insert(0) += 1;
            }
            let slot = whole.entry(signature(structure)).or_insert((index, 0));
            slot.1 += 1;
        }
        let reference_signature = signature(&reference);
        let reference_count = whole.get(&reference_signature).map_or(0, |slot| slot.1);
        for edge in reference.edges() {
            pairs.entry((edge.a, edge.b)).or_default();
        }

        let reference_pairs: std::collections::BTreeSet<(StructureNode, StructureNode)> =
            reference.edges().iter().map(|e| (e.a, e.b)).collect();
        let edges = pairs
            .into_iter()
            .map(|((a, b), (adjacent, marks))| {
                let mut marks: Vec<MarkSelection> = marks
                    .into_iter()
                    .map(|((at_a, at_b), count)| MarkSelection {
                        at_a,
                        at_b,
                        frequency: f64::from(count) / total,
                    })
                    .collect();
                marks.sort_by(|x, y| {
                    y.frequency
                        .total_cmp(&x.frequency)
                        .then_with(|| (x.at_a, x.at_b).cmp(&(y.at_a, y.at_b)))
                });
                EdgeSelection {
                    a,
                    b,
                    frequency: f64::from(adjacent) / total,
                    in_reference: reference_pairs.contains(&(a, b)),
                    marks,
                }
            })
            .collect();

        let mut structures: Vec<StructureSelection> = whole
            .into_values()
            .map(|(first, count)| StructureSelection {
                structure: resamples[first].clone(),
                count,
                frequency: f64::from(count) / total,
            })
            .collect();
        structures.sort_by(|x, y| {
            y.count.cmp(&x.count).then_with(|| x.structure.identity().cmp(&y.structure.identity()))
        });

        Ok(Self {
            class: reference.class(),
            requested,
            succeeded,
            failed: requested - succeeded,
            reference_frequency: f64::from(reference_count) / total,
            reference,
            edges,
            structures,
        })
    }

    /// Frequency with which the node pair was adjacent, or `0` for a pair no structure
    /// joined.
    #[must_use]
    pub fn adjacency_frequency(&self, a: StructureNode, b: StructureNode) -> f64 {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.edges.iter().find(|e| (e.a, e.b) == key).map_or(0.0, |e| e.frequency)
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{NodeRef, VariableId};
    use antecedent_graph::Cpdag;

    use super::*;

    fn node(v: u32) -> NodeRef {
        NodeRef::Static(VariableId::from_raw(v))
    }

    /// A three-node CPDAG with the listed edges; `true` is directed `a -> b`, `false` is
    /// undirected.
    fn cpdag(edges: &[(u32, u32, bool)]) -> DiscoveredStructure {
        let mut graph = Cpdag::empty();
        let ids: Vec<_> = (0..3).map(|v| graph.add_node(node(v)).unwrap()).collect();
        for &(a, b, directed) in edges {
            if directed {
                graph.insert_directed(ids[a as usize], ids[b as usize]).unwrap();
            } else {
                graph.insert_undirected(ids[a as usize], ids[b as usize]).unwrap();
            }
        }
        DiscoveredStructure::from_cpdag(&graph)
    }

    fn static_node(v: u32) -> StructureNode {
        StructureNode::from_node_ref(node(v))
    }

    #[test]
    fn frequencies_count_pairs_marks_and_whole_structures() {
        let chain = cpdag(&[(0, 1, false), (1, 2, false)]);
        let collider = cpdag(&[(0, 1, true), (2, 1, true)]);
        let sparse = cpdag(&[(0, 1, false)]);
        let resamples =
            vec![chain.clone(), chain.clone(), chain.clone(), collider.clone(), sparse.clone()];
        let stability = StructureStability::tabulate(chain.clone(), &resamples, 6).unwrap();
        assert_eq!((stability.requested, stability.succeeded, stability.failed), (6, 5, 1));
        assert!((stability.reference_frequency - 0.6).abs() < 1e-12);
        assert!(
            (stability.adjacency_frequency(static_node(0), static_node(1)) - 1.0).abs() < 1e-12
        );
        assert!(
            (stability.adjacency_frequency(static_node(2), static_node(1)) - 0.8).abs() < 1e-12
        );
        assert_eq!(stability.adjacency_frequency(static_node(0), static_node(2)), 0.0);
        // The 1 - 2 pair is undirected in three resamples and directed in one.
        let pair = stability
            .edges
            .iter()
            .find(|e| (e.a, e.b) == (static_node(1), static_node(2)))
            .unwrap();
        assert!(pair.in_reference);
        assert_eq!(pair.marks.len(), 2);
        assert!((pair.marks[0].frequency - 0.6).abs() < 1e-12);
        assert!((pair.marks[1].frequency - 0.2).abs() < 1e-12);
        // Whole structures, most frequent first.
        assert_eq!(stability.structures.len(), 3);
        assert_eq!(stability.structures[0].count, 3);
        assert_eq!(stability.structures[0].structure, chain);
        let mass: f64 = stability.structures.iter().map(|s| s.frequency).sum();
        assert!((mass - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_reference_no_resample_returned_has_zero_frequency() {
        let chain = cpdag(&[(0, 1, false), (1, 2, false)]);
        let sparse = cpdag(&[(0, 1, false)]);
        let stability = StructureStability::tabulate(chain, &[sparse.clone(), sparse], 2).unwrap();
        assert_eq!(stability.reference_frequency, 0.0);
        let missing = stability
            .edges
            .iter()
            .find(|e| (e.a, e.b) == (static_node(1), static_node(2)))
            .unwrap();
        assert!(missing.in_reference);
        assert_eq!(missing.frequency, 0.0);
        assert!(missing.marks.is_empty());
    }

    #[test]
    fn a_tabulation_that_lost_most_resamples_is_refused() {
        let chain = cpdag(&[(0, 1, false)]);
        let resamples = vec![chain.clone(), chain.clone()];
        assert!(StructureStability::tabulate(chain.clone(), &resamples, 3).is_ok());
        assert!(StructureStability::tabulate(chain.clone(), &resamples, 5).is_err());
        assert!(StructureStability::tabulate(chain.clone(), &resamples[..1], 1).is_err());
        assert!(StructureStability::tabulate(chain.clone(), &resamples, 0).is_err());
        assert!(StructureStability::tabulate(chain, &resamples, 1).is_err());
    }
}

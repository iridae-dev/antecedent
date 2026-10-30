//! Lagged temporal-ADMG posterior atoms.
//!
//! A template atom of a stationary latent-confounded process is one contemporaneous ADMG over
//! the `p` variables (directed edges and latent-confounder edges between variables of the same
//! period) plus lagged edges `X_{t-ℓ} → Y_t` and lagged latent-confounder edges
//! `X_{t-ℓ} ↔ Y_t` for `ℓ ∈ 1..=max_lag`. The contemporaneous part packs as an
//! [`AdjacencyMask`] (a bidirected edge is both directed bits, as for a static ADMG atom); each
//! lagged part packs as a [`LagMask`] (a growable bit set of any width) whose bit `(ℓ - 1) * p² + from * p + to` is the edge from
//! `from` at `t - ℓ` to `to` at `t`, the layout the DBN lag masks use.
//!
//! The atom of position `i` of a [`GraphPosterior`] is
//! [`GraphPosterior::temporal_admg_atom`]; [`TemporalAdmgAtom::to_template`] unrolls nothing: it
//! is the stationary template a window identification unrolls itself.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, NodeRef, VariableId};
use antecedent_graph::{DenseNodeId, TemporalAdmg};

use crate::adjacency_mask::{AdjacencyMask, LagMask, admg_from_adjacency_mask};
use crate::error::DiscoveryError;

/// Checked bit index for a lagged edge.
///
/// # Errors
///
/// A zero lag, a variable outside the template, or an index outside the u32 bit space.
pub fn checked_lagged_edge_bit(
    n_vars: usize,
    lag: u32,
    from: usize,
    to: usize,
) -> Result<u32, DiscoveryError> {
    if lag == 0 || from >= n_vars || to >= n_vars {
        return Err(DiscoveryError::unsupported("lagged edge outside the template"));
    }
    let bit = n_vars
        .checked_mul(n_vars)
        .and_then(|block| (lag as usize - 1).checked_mul(block))
        .and_then(|base| from.checked_mul(n_vars).and_then(|offset| base.checked_add(offset)))
        .and_then(|base| base.checked_add(to))
        .and_then(|bit| u32::try_from(bit).ok())
        .ok_or_else(|| DiscoveryError::unsupported("lagged edge bit exceeds u32 index space"))?;
    Ok(bit)
}

/// Bit of the lagged edge `from(t - lag) → to(t)` (or `from(t - lag) ↔ to(t)`).
///
/// # Panics
///
/// Panics when the coordinates are invalid or their bit is not addressable by u32. Use
/// [`checked_lagged_edge_bit`] to handle such inputs without a panic.
#[must_use]
pub fn lagged_edge_bit(n_vars: usize, lag: u32, from: usize, to: usize) -> u32 {
    checked_lagged_edge_bit(n_vars, lag, from, to).expect("invalid lagged edge bit")
}

/// Whether a lag set holds the edge `from(t - lag) → to(t)`.
#[must_use]
pub fn has_lagged_edge(mask: &LagMask, n_vars: usize, lag: u32, from: usize, to: usize) -> bool {
    checked_lagged_edge_bit(n_vars, lag, from, to).is_ok_and(|bit| mask.bit(bit))
}

/// Set or clear the lagged edge `from(t - lag) → to(t)` in a lag set (a [`LagMask`] or the
/// single-word `u64` packing of one).
///
/// # Errors
///
/// A lag of zero or a variable outside the template.
pub fn with_lagged_edge(
    mask: impl Into<LagMask>,
    n_vars: usize,
    lag: u32,
    from: usize,
    to: usize,
    present: bool,
) -> Result<LagMask, DiscoveryError> {
    Ok(mask.into().with_bit(checked_lagged_edge_bit(n_vars, lag, from, to)?, present))
}

/// One atom of a lagged temporal-ADMG posterior.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct TemporalAdmgAtom {
    /// Contemporaneous ADMG: both directed bits of a pair are a latent-confounder edge.
    pub contemporaneous: AdjacencyMask,
    /// Lagged directed edges `X_{t-ℓ} → Y_t`.
    pub lagged_directed: LagMask,
    /// Lagged latent-confounder edges `X_{t-ℓ} ↔ Y_t`.
    pub lagged_bidirected: LagMask,
}

fn node_of(
    graph: &mut TemporalAdmg,
    variable: VariableId,
    lag: Lag,
) -> Result<DenseNodeId, DiscoveryError> {
    for (index, node) in graph.nodes().iter().enumerate() {
        if let NodeRef::Lagged { variable: existing, lag: existing_lag } = node {
            if *existing == variable && *existing_lag == lag {
                return Ok(DenseNodeId::try_from_usize(index)?);
            }
        }
    }
    Ok(graph.add_lagged(variable, lag)?)
}

impl TemporalAdmgAtom {
    /// The stationary [`TemporalAdmg`] template of this atom over `variables` (position `i` is
    /// variable `i` of the masks).
    ///
    /// # Errors
    ///
    /// A variable count that differs from `n_vars`, a contemporaneous mask that is not an
    /// acyclic ADMG, or an edge the template rejects.
    pub fn to_template(
        &self,
        n_vars: usize,
        max_lag: u32,
        variables: &[VariableId],
    ) -> Result<TemporalAdmg, DiscoveryError> {
        if variables.len() != n_vars {
            return Err(DiscoveryError::data_msg(
                "temporal ADMG atom: variables length differs from n_vars",
            ));
        }
        let contemporaneous = admg_from_adjacency_mask(&self.contemporaneous, n_vars)?;
        let mut graph = TemporalAdmg::empty();
        for (from, to) in contemporaneous.directed_edge_pairs_sorted() {
            let from = node_of(&mut graph, variables[from as usize], Lag::CONTEMPORANEOUS)?;
            let to = node_of(&mut graph, variables[to as usize], Lag::CONTEMPORANEOUS)?;
            graph.insert_directed(from, to)?;
        }
        for (a, b) in contemporaneous.bidirected_edge_pairs_sorted() {
            let a = node_of(&mut graph, variables[a as usize], Lag::CONTEMPORANEOUS)?;
            let b = node_of(&mut graph, variables[b as usize], Lag::CONTEMPORANEOUS)?;
            graph.insert_bidirected(a, b)?;
        }
        let block = n_vars
            .checked_mul(n_vars)
            .ok_or_else(|| DiscoveryError::unsupported("temporal ADMG lag block overflow"))?;
        // Both masks enumerate in increasing bit order; merge them without an additional
        // set-sized allocation, visiting a bit shared by both masks once.
        let mut directed = self.lagged_directed.set_bits().peekable();
        let mut bidirected = self.lagged_bidirected.set_bits().peekable();
        while let Some(bit) = match (directed.peek().copied(), bidirected.peek().copied()) {
            (Some(a), Some(b)) if a == b => {
                directed.next();
                bidirected.next();
                Some(a)
            }
            (Some(a), Some(b)) if a < b => directed.next(),
            (Some(_), Some(_)) => bidirected.next(),
            (Some(_), None) => directed.next(),
            (None, Some(_)) => bidirected.next(),
            (None, None) => None,
        } {
            if block == 0 {
                return Err(DiscoveryError::unsupported("lagged edge outside the template"));
            }
            let bit_index = bit as usize;
            let lag = bit_index / block + 1;
            if lag > max_lag as usize {
                return Err(DiscoveryError::unsupported("lagged edge outside the template"));
            }
            let within = bit_index % block;
            let from = within / n_vars;
            let to = within % n_vars;
            let past = node_of(&mut graph, variables[from], Lag::from_raw(lag as u32))?;
            let now = node_of(&mut graph, variables[to], Lag::CONTEMPORANEOUS)?;
            if self.lagged_directed.bit(bit) {
                graph.insert_directed(past, now)?;
            }
            if self.lagged_bidirected.bit(bit) {
                graph.insert_bidirected(past, now)?;
            }
        }
        Ok(graph)
    }
}

/// Build the stationary [`TemporalAdmg`] template of a contemporaneous ADMG mask and two lag
/// sets: [`TemporalAdmgAtom::to_template`] for loose parts.
///
/// # Errors
///
/// As [`TemporalAdmgAtom::to_template`].
pub fn temporal_admg_from_lagged_atoms(
    contemporaneous: impl Into<AdjacencyMask>,
    lagged_directed: impl Into<LagMask>,
    lagged_bidirected: impl Into<LagMask>,
    n_vars: usize,
    max_lag: u32,
    variables: &[VariableId],
) -> Result<TemporalAdmg, DiscoveryError> {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous.into(),
        lagged_directed: lagged_directed.into(),
        lagged_bidirected: lagged_bidirected.into(),
    }
    .to_template(n_vars, max_lag, variables)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency_mask::set_edge;

    fn variables(n: usize) -> Vec<VariableId> {
        (0..n as u32).map(VariableId::from_raw).collect()
    }

    #[test]
    fn sparse_template_checks_width_without_scanning_empty_lags() {
        assert!(checked_lagged_edge_bit(usize::MAX, 2, 0, 0).is_err());
        assert!(with_lagged_edge(0u64, 65536, 2, 0, 0, true).is_err());
        assert!(!has_lagged_edge(&LagMask::EMPTY, 65536, 2, 0, 0));
        let atom = TemporalAdmgAtom::default();
        assert!(atom.to_template(1, u32::MAX, &variables(1)).is_ok());
        let outside =
            TemporalAdmgAtom { lagged_directed: LagMask::EMPTY.with_bit(1, true), ..atom };
        assert!(outside.to_template(1, 1, &variables(1)).is_err());
    }

    #[test]
    fn the_lag_layout_matches_the_dbn_lag_masks() {
        // Lag 1 occupies the first p*p bits, lag 2 the next block; `from * p + to` inside.
        assert_eq!(lagged_edge_bit(3, 1, 0, 0), 0);
        assert_eq!(lagged_edge_bit(3, 1, 2, 1), 7);
        assert_eq!(lagged_edge_bit(3, 2, 0, 0), 9);
        assert_eq!(lagged_edge_bit(3, 2, 1, 2), 9 + 5);
        let word = with_lagged_edge(0u64, 3, 2, 1, 2, true).unwrap();
        assert!(has_lagged_edge(&word, 3, 2, 1, 2));
        assert!(!has_lagged_edge(&word, 3, 1, 1, 2));
        assert_eq!(with_lagged_edge(word, 3, 2, 1, 2, false).unwrap(), LagMask::EMPTY);
        assert!(with_lagged_edge(0u64, 3, 0, 1, 2, true).is_err(), "lag 0 is not a lagged edge");
        assert!(with_lagged_edge(0u64, 3, 1, 3, 0, true).is_err(), "variable outside the template");
    }

    #[test]
    fn a_wide_lag_space_works() {
        // Bit 127 is far past one word.
        let word = with_lagged_edge(0u64, 8, 2, 7, 7, true).unwrap();
        assert_eq!(lagged_edge_bit(8, 2, 7, 7), 127);
        assert!(has_lagged_edge(&word, 8, 2, 7, 7));
        assert!(!has_lagged_edge(&word, 8, 2, 7, 6));
        let directed = with_lagged_edge(word, 8, 12, 0, 1, true).unwrap();
        let template =
            temporal_admg_from_lagged_atoms(0u64, directed, 0u64, 8, 12, &variables(8)).unwrap();
        assert_eq!(template.node_count(), 4);
    }

    #[test]
    fn an_atom_unrolls_to_its_stationary_template() {
        let n = 3;
        // x0 -> x1 and x1 <-> x2 in a period; x0(t-1) -> x1(t); x2(t-1) <-> x0(t).
        let mut contemporaneous = AdjacencyMask::EMPTY;
        contemporaneous = set_edge(contemporaneous, n, 0, 1, true);
        contemporaneous = set_edge(contemporaneous, n, 1, 2, true);
        contemporaneous = set_edge(contemporaneous, n, 2, 1, true);
        let directed = with_lagged_edge(0u64, n, 1, 0, 1, true).unwrap();
        let bidirected = with_lagged_edge(0u64, n, 1, 2, 0, true).unwrap();
        let atom = TemporalAdmgAtom {
            contemporaneous,
            lagged_directed: directed,
            lagged_bidirected: bidirected,
        };
        let template = atom.to_template(n, 1, &variables(n)).unwrap();
        let find = |variable: u32, lag: u32| {
            template
                .nodes()
                .iter()
                .position(|node| {
                    *node
                        == NodeRef::Lagged {
                            variable: VariableId::from_raw(variable),
                            lag: Lag::from_raw(lag),
                        }
                })
                .map(DenseNodeId::from_index)
        };
        let (x0_now, x1_now, x2_now) =
            (find(0, 0).unwrap(), find(1, 0).unwrap(), find(2, 0).unwrap());
        let (x0_past, x2_past) = (find(0, 1).unwrap(), find(2, 1).unwrap());
        assert!(template.children(x0_now).contains(&x1_now));
        assert!(template.bidirected_neighbors(x1_now).contains(&x2_now));
        assert!(template.children(x0_past).contains(&x1_now));
        assert!(template.bidirected_neighbors(x2_past).contains(&x0_now));
        // Nothing else: three contemporaneous nodes and two lagged ones.
        assert_eq!(template.node_count(), 5);
        assert_eq!(template.edges().len(), 4);
    }

    #[test]
    fn both_directed_bits_are_a_latent_confounder_and_the_variable_count_must_match() {
        let n = 2;
        let mut both = AdjacencyMask::EMPTY;
        both = set_edge(both, n, 0, 1, true);
        both = set_edge(both, n, 1, 0, true);
        let template =
            temporal_admg_from_lagged_atoms(both.clone(), 0u64, 0u64, n, 1, &variables(n)).unwrap();
        assert_eq!(template.node_count(), 2);
        assert!(template.has_bidirected());
        assert!(temporal_admg_from_lagged_atoms(both, 0u64, 0u64, n, 1, &variables(3)).is_err());
    }
}

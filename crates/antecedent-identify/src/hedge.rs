//! Hedge certificates for non-identifiability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_graph::{BitSet, DenseNodeId};

/// Witness that `P(Y | do(X))` is not identifiable (Shpitser–Pearl hedge).
///
/// Recovered from ID line 5: the pair `(F, F')` of R-rooted C-forests where
/// `F` is the current subgraph `G` and `F'` is the C-component `S` of `G[V\X]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HedgeCertificate {
    /// Nodes of the larger C-forest `F` (current active subgraph).
    pub f: Arc<[VariableId]>,
    /// Nodes of the smaller C-forest `F' ⊆ F` (C-component of `G[V\X]`).
    pub f_prime: Arc<[VariableId]>,
    /// Dense ids of `F` (stable for diagnostics).
    pub f_dense: Arc<[DenseNodeId]>,
    /// Dense ids of `F'`.
    pub f_prime_dense: Arc<[DenseNodeId]>,
}

impl HedgeCertificate {
    /// Build a certificate from dense node sets and a variable map.
    #[must_use]
    pub fn from_sets(
        f: &BitSet,
        f_prime: &BitSet,
        dense_to_var: impl Fn(DenseNodeId) -> VariableId,
    ) -> Self {
        let f_dense = f.to_dense_ids();
        let f_prime_dense = f_prime.to_dense_ids();
        let f_vars: Vec<VariableId> = f_dense.iter().copied().map(&dense_to_var).collect();
        let fp_vars: Vec<VariableId> = f_prime_dense.iter().copied().map(&dense_to_var).collect();
        Self {
            f: Arc::from(f_vars),
            f_prime: Arc::from(fp_vars),
            f_dense: Arc::from(f_dense),
            f_prime_dense: Arc::from(f_prime_dense),
        }
    }
}

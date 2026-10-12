//! Consistency checks between a frozen graph posterior and the per-atom
//! identification cache sealed beside it.
//!
//! Every checked graph-posterior operation freezes the posterior samples and
//! the prepare-time proofs together; these helpers are the one place that
//! decides whether the two describe the same sample set. Atom lookups go
//! through hash sets, so a check is linear in samples plus atoms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_discovery::GraphPosterior;
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::checked_temporal_graph_posterior_effect::CheckedTemporalGraphPosteriorProof;
use super::prepared::CachedGraphPosteriorIdentification;
use crate::error::CausalError;

/// Which graph keys the cache's atoms are looked up in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheKeys {
    /// Static atoms: the cache keys are the posterior's own `graph_keys`.
    PosteriorKeys,
    /// DBN atoms carry collision-free keys local to the frozen cache, so the
    /// posterior's contemporaneous adjacency keys are not comparable.
    CacheLocal,
}

/// The frozen sample set and retained atoms of a prepare-time posterior proof.
pub(crate) trait PosteriorProof {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    fn graphs(&self) -> &WeightedGraphSamples;

    /// Keys of the identified atoms the proof retains.
    fn atom_keys(&self) -> Vec<u64>;

    /// Per-sample atom identities, posterior weights, and identification
    /// flags, including unidentified samples.
    fn sample_mass(&self) -> (&[u64], &[f64], &[GraphIdentFlag]) {
        let graphs = self.graphs();
        (&graphs.graph_keys, &graphs.weights, &graphs.identified)
    }
}

impl PosteriorProof for CachedGraphPosteriorIdentification {
    fn graphs(&self) -> &WeightedGraphSamples {
        &self.graphs
    }

    fn atom_keys(&self) -> Vec<u64> {
        self.atoms
            .iter()
            .map(|atom| atom.key)
            .chain(self.class_atoms.iter().map(|atom| atom.key))
            .collect()
    }
}

impl PosteriorProof for CheckedTemporalGraphPosteriorProof {
    fn graphs(&self) -> &WeightedGraphSamples {
        match self {
            Self::Dbn(cache) => &cache.graphs,
            Self::Class(cache) => &cache.graphs,
        }
    }

    fn atom_keys(&self) -> Vec<u64> {
        match self {
            Self::Dbn(cache) => cache.atoms.iter().map(|atom| atom.key).collect(),
            Self::Class(cache) => cache.class_atoms.iter().map(|atom| atom.key).collect(),
        }
    }
}

/// Whether `proof` was built for exactly `posterior`'s samples: same sample
/// count, weights, and identification-flag length, and every retained atom
/// names one of the frozen graph keys.
#[must_use]
pub(crate) fn posterior_cache_matches(
    posterior: &GraphPosterior,
    proof: &impl PosteriorProof,
    keys: CacheKeys,
) -> bool {
    let graphs = proof.graphs();
    if graphs.n_samples != posterior.n_graphs
        || graphs.identified.len() != posterior.n_graphs
        || graphs.weights.as_ref() != posterior.weights.as_ref()
    {
        return false;
    }
    if keys == CacheKeys::PosteriorKeys
        && graphs.graph_keys.as_ref() != posterior.graph_keys.as_ref()
    {
        return false;
    }
    let frozen: HashSet<u64> = graphs.graph_keys.iter().copied().collect();
    proof.atom_keys().iter().all(|key| frozen.contains(key))
}

/// [`posterior_cache_matches`] as a typed refusal carrying `message`.
pub(crate) fn validate_posterior_cache(
    posterior: &GraphPosterior,
    proof: &impl PosteriorProof,
    keys: CacheKeys,
    message: &'static str,
) -> Result<(), CausalError> {
    if posterior_cache_matches(posterior, proof, keys) {
        Ok(())
    } else {
        Err(CausalError::Unsupported { message })
    }
}

/// Keys of the samples flagged identified.
#[must_use]
pub(crate) fn identified_key_set(graphs: &WeightedGraphSamples) -> HashSet<u64> {
    graphs
        .graph_keys
        .iter()
        .zip(graphs.identified.iter())
        .filter(|(_, flag)| matches!(flag, GraphIdentFlag::Identified))
        .map(|(key, _)| *key)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_discovery::SamplingSummary;
    use antecedent_prob::InferenceDiagnostics;
    use std::sync::Arc;

    fn posterior() -> GraphPosterior {
        GraphPosterior::new(
            2,
            vec![0.25, 0.75],
            vec![0, 0],
            vec![0.0; 4],
            vec![0.0; 4],
            SamplingSummary {
                ess: 1.6,
                diagnostics: InferenceDiagnostics::analytic("posterior_binding_test"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
    }

    fn cache(weights: Vec<f64>, keys: Vec<u64>) -> CachedGraphPosteriorIdentification {
        let flags = vec![GraphIdentFlag::Unidentified; weights.len()];
        CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(weights, flags, keys).unwrap(),
            atoms: Arc::from([]),
            class_atoms: Arc::from([]),
        }
    }

    #[test]
    fn cache_matches_only_its_own_samples() {
        let graphs = posterior();
        let own = cache(graphs.weights.to_vec(), graphs.graph_keys.to_vec());
        assert!(posterior_cache_matches(&graphs, &own, CacheKeys::PosteriorKeys));
        let reweighted = cache(vec![0.5, 0.5], graphs.graph_keys.to_vec());
        assert!(!posterior_cache_matches(&graphs, &reweighted, CacheKeys::PosteriorKeys));
        let rekeyed = cache(graphs.weights.to_vec(), vec![1, 2]);
        assert!(!posterior_cache_matches(&graphs, &rekeyed, CacheKeys::PosteriorKeys));
        assert!(posterior_cache_matches(&graphs, &rekeyed, CacheKeys::CacheLocal));
    }

    #[test]
    fn sample_mass_exposes_keys_weights_and_flags() {
        let graphs = posterior();
        let own = cache(graphs.weights.to_vec(), graphs.graph_keys.to_vec());
        let (keys, weights, flags) = own.sample_mass();
        assert_eq!(keys, graphs.graph_keys.as_ref());
        assert_eq!(weights, &[0.25, 0.75]);
        assert_eq!(flags, &[GraphIdentFlag::Unidentified; 2]);
        assert!(identified_key_set(&own.graphs).is_empty());
    }
}

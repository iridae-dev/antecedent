//! Weighted graph samples for effect envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::CausalRng;

use crate::error::ProbError;

/// One graph sample's identification flag (does not invent identification).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphIdentFlag {
    /// Nonparametrically identified under this graph.
    Identified,
    /// Not identified under this graph.
    Unidentified,
}

/// Result of an Interactive stratified graph subsample.
#[derive(Clone, Debug)]
pub struct GraphEnvelopeSubsample {
    /// Ensemble used for the mixture (excluded identified mass flipped to Unidentified).
    pub graphs: WeightedGraphSamples,
    /// Absolute weight of identified graphs reclassified as Unidentified for UI.
    pub leftover_identified_mass: f64,
    /// True when the ensemble was truncated below the full identified set.
    pub approximate: bool,
}

/// Columnar ensemble of weighted graphs.
///
/// Weights are stored as a flat `f64` array (normalized or not). Edge / orientation
/// marginals are optional flat arrays — never one boxed graph object per sample
/// for hot-path aggregation.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedGraphSamples {
    /// Number of graph samples.
    pub n_samples: usize,
    /// Per-sample weight (length `n_samples`).
    pub weights: Arc<[f64]>,
    /// Per-sample identification flag.
    pub identified: Arc<[GraphIdentFlag]>,
    /// Optional opaque graph keys (e.g. hash / index into an external store).
    pub graph_keys: Arc<[u64]>,
    /// Optional edge-marginal probabilities (caller-defined packing).
    pub edge_marginals: Option<Arc<[f64]>>,
    /// Optional orientation-marginal probabilities.
    pub orientation_marginals: Option<Arc<[f64]>>,
}

impl WeightedGraphSamples {
    /// Construct from parallel arrays.
    ///
    /// # Errors
    ///
    /// Length mismatch or empty ensemble.
    pub fn new(
        weights: impl Into<Arc<[f64]>>,
        identified: impl Into<Arc<[GraphIdentFlag]>>,
        graph_keys: impl Into<Arc<[u64]>>,
    ) -> Result<Self, ProbError> {
        let weights = weights.into();
        let identified = identified.into();
        let graph_keys = graph_keys.into();
        let n = weights.len();
        if n == 0 {
            return Err(ProbError::Shape { message: "empty graph ensemble" });
        }
        if identified.len() != n || graph_keys.len() != n {
            return Err(ProbError::Shape { message: "weights/identified/keys length mismatch" });
        }
        Ok(Self {
            n_samples: n,
            weights,
            identified,
            graph_keys,
            edge_marginals: None,
            orientation_marginals: None,
        })
    }

    /// Total weight mass.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Weight mass on unidentified graphs.
    #[must_use]
    pub fn unidentified_mass(&self) -> f64 {
        self.weights
            .iter()
            .zip(self.identified.iter())
            .filter(|(_, f)| **f == GraphIdentFlag::Unidentified)
            .map(|(w, _)| *w)
            .sum()
    }

    /// Weight mass on identified graphs.
    #[must_use]
    pub fn identified_mass(&self) -> f64 {
        self.total_weight() - self.unidentified_mass()
    }

    /// Return a copy with weights normalized to sum to 1 (if total > 0).
    ///
    /// # Errors
    ///
    /// Non-positive total weight.
    pub fn normalized(&self) -> Result<Self, ProbError> {
        let total = self.total_weight();
        if !(total > 0.0) {
            return Err(ProbError::Shape { message: "non-positive total weight" });
        }
        let weights: Arc<[f64]> =
            Arc::from(self.weights.iter().map(|w| w / total).collect::<Vec<_>>());
        Ok(Self {
            n_samples: self.n_samples,
            weights,
            identified: Arc::clone(&self.identified),
            graph_keys: Arc::clone(&self.graph_keys),
            edge_marginals: self.edge_marginals.clone(),
            orientation_marginals: self.orientation_marginals.clone(),
        })
    }

    /// Interactive stratified subsample: keep at most `max_identified` Identified
    /// graphs (plus all Unidentified), flipping leftover Identified flags to
    /// Unidentified so their mass is never silently dropped.
    ///
    /// Total weight is unchanged. Mixture draws use E[τ | identified-in-subset].
    ///
    /// # Errors
    ///
    /// Empty ensemble.
    pub fn stratified_interactive_subsample(
        &self,
        max_identified: usize,
        rng: &mut CausalRng,
    ) -> Result<GraphEnvelopeSubsample, ProbError> {
        if self.n_samples == 0 {
            return Err(ProbError::Shape { message: "empty graph ensemble" });
        }
        let mut identified_idx: Vec<usize> = (0..self.n_samples)
            .filter(|&i| self.identified[i] == GraphIdentFlag::Identified)
            .collect();
        if identified_idx.len() <= max_identified {
            return Ok(GraphEnvelopeSubsample {
                graphs: self.clone(),
                leftover_identified_mass: 0.0,
                approximate: false,
            });
        }
        // Fisher–Yates partial shuffle: first `max_identified` slots are the keep set.
        for i in 0..max_identified {
            let j = i + (rng.next_u64() as usize % (identified_idx.len() - i));
            identified_idx.swap(i, j);
        }
        let mut flags = self.identified.to_vec();
        let mut leftover = 0.0;
        for &i in &identified_idx[max_identified..] {
            leftover += self.weights[i];
            flags[i] = GraphIdentFlag::Unidentified;
        }
        let graphs = Self {
            n_samples: self.n_samples,
            weights: Arc::clone(&self.weights),
            identified: Arc::from(flags),
            graph_keys: Arc::clone(&self.graph_keys),
            edge_marginals: self.edge_marginals.clone(),
            orientation_marginals: self.orientation_marginals.clone(),
        };
        Ok(GraphEnvelopeSubsample {
            graphs,
            leftover_identified_mass: leftover,
            approximate: leftover > 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unidentified_mass_preserved() {
        let g = WeightedGraphSamples::new(
            vec![0.5, 0.3, 0.2],
            vec![
                GraphIdentFlag::Identified,
                GraphIdentFlag::Unidentified,
                GraphIdentFlag::Identified,
            ],
            vec![1, 2, 3],
        )
        .unwrap();
        assert!((g.unidentified_mass() - 0.3).abs() < 1e-12);
        assert!((g.identified_mass() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn stratified_subsample_moves_leftover_to_unidentified() {
        let g = WeightedGraphSamples::new(
            vec![0.25, 0.25, 0.25, 0.25],
            vec![
                GraphIdentFlag::Identified,
                GraphIdentFlag::Identified,
                GraphIdentFlag::Identified,
                GraphIdentFlag::Unidentified,
            ],
            vec![10, 11, 12, 13],
        )
        .unwrap();
        let mut rng = CausalRng::from_seed(7);
        let sub = g.stratified_interactive_subsample(1, &mut rng).unwrap();
        assert!(sub.approximate);
        assert!(sub.leftover_identified_mass > 0.0);
        assert!((sub.graphs.total_weight() - g.total_weight()).abs() < 1e-12);
        let expected_uid = g.unidentified_mass() + sub.leftover_identified_mass;
        assert!((sub.graphs.unidentified_mass() - expected_uid).abs() < 1e-12);
        let n_id =
            sub.graphs.identified.iter().filter(|f| **f == GraphIdentFlag::Identified).count();
        assert_eq!(n_id, 1);
    }
}

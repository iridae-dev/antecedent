//! Weighted graph samples for effect envelopes (DESIGN.md §3, §14.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::error::ProbError;

/// One graph sample's identification flag (does not invent identification).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphIdentFlag {
    /// Nonparametrically identified under this graph.
    Identified,
    /// Not identified under this graph.
    Unidentified,
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
            return Err(ProbError::Shape {
                message: "weights/identified/keys length mismatch",
            });
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
        let weights: Arc<[f64]> = Arc::from(
            self.weights.iter().map(|w| w / total).collect::<Vec<_>>(),
        );
        Ok(Self {
            n_samples: self.n_samples,
            weights,
            identified: Arc::clone(&self.identified),
            graph_keys: Arc::clone(&self.graph_keys),
            edge_marginals: self.edge_marginals.clone(),
            orientation_marginals: self.orientation_marginals.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unidentified_mass_preserved() {
        let g = WeightedGraphSamples::new(
            vec![0.4, 0.3, 0.3],
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
}

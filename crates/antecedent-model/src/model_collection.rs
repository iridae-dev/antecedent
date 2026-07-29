//! Weighted collection of fitted per-graph causal models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::compile::CompiledCausalModel;
use crate::error::ModelError;

/// Collection of fitted models weighted by graph posterior mass.
#[derive(Clone, Debug)]
pub struct ModelCollection {
    /// Per-graph compiled models.
    pub models: Arc<[CompiledCausalModel]>,
    /// Graph keys aligned with `models`.
    pub graph_keys: Arc<[u64]>,
    /// Normalized weights (sum to 1 over identified graphs).
    pub weights: Arc<[f64]>,
}

impl ModelCollection {
    /// Build from parallel arrays.
    ///
    /// # Errors
    ///
    /// Length mismatch or non-positive weight sum.
    pub fn new(
        models: impl Into<Arc<[CompiledCausalModel]>>,
        graph_keys: impl Into<Arc<[u64]>>,
        weights: impl Into<Arc<[f64]>>,
    ) -> Result<Self, ModelError> {
        let models = models.into();
        let graph_keys = graph_keys.into();
        let weights = weights.into();
        if models.len() != graph_keys.len() || models.len() != weights.len() {
            return Err(ModelError::Shape { message: "ModelCollection length mismatch".into() });
        }
        let sum: f64 = weights.iter().sum();
        if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(ModelError::Shape {
                message: "ModelCollection weights non-positive".into(),
            });
        }
        let weights: Arc<[f64]> = Arc::from(weights.iter().map(|w| w / sum).collect::<Vec<_>>());
        Ok(Self { models, graph_keys, weights })
    }

    /// Number of graphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Empty check.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

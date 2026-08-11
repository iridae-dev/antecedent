//! Structural transportability queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::VariableId;

use super::{QueryError, ResponseQuery};

/// Transport a response from one explicitly named population to another.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportQuery {
    /// Response functional requested in the target population.
    pub response: ResponseQuery,
    /// Source population key.
    pub source_population: Arc<str>,
    /// Target population key.
    pub target_population: Arc<str>,
    /// Variables for which source experiments are available.
    pub source_experiments: Arc<[VariableId]>,
}

impl TransportQuery {
    /// Construct a single-source transport query.
    #[must_use]
    pub fn new(
        response: ResponseQuery,
        source_population: impl Into<Arc<str>>,
        target_population: impl Into<Arc<str>>,
        source_experiments: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        Self {
            response,
            source_population: source_population.into(),
            target_population: target_population.into(),
            source_experiments: source_experiments.into(),
        }
    }

    /// Validate population keys, response semantics, and experiment uniqueness.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] or the nested response validation error.
    pub fn validate(&self) -> Result<(), QueryError> {
        self.response.validate()?;
        if self.source_population.trim().is_empty()
            || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
        {
            return Err(QueryError::InvalidTransport(
                "source and target population keys must be non-empty and distinct".into(),
            ));
        }
        let mut experiments = self.source_experiments.to_vec();
        experiments.sort_unstable_by_key(|id| id.raw());
        if experiments.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(QueryError::InvalidTransport(
                "source experiment variables must be unique".into(),
            ));
        }
        Ok(())
    }
}

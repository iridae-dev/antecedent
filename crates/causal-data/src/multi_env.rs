//! Multi-environment / multi-dataset container (DESIGN.md §5.1).
//!
//! Typed list of series sharing one schema. Sample planning without per-env
//! full copies lives in [`crate::multi_env_plan`]. J-PCMCI+ discovery
//! constraints are wired in Phase 9 discovery.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::CausalSchema;

use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::table::TableView;

/// Collection of time-series environments with a shared schema.
#[derive(Clone, Debug)]
pub struct MultiEnvironmentData {
    schema: Arc<CausalSchema>,
    environments: Arc<[TimeSeriesData]>,
}

impl MultiEnvironmentData {
    /// Construct from one or more environments; all must share an identical schema.
    ///
    /// # Errors
    ///
    /// Empty list or schema mismatch across environments.
    pub fn try_new(environments: impl Into<Arc<[TimeSeriesData]>>) -> Result<Self, DataError> {
        let environments = environments.into();
        if environments.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "multi-environment data needs ≥1 environment".into(),
            });
        }
        let schema = Arc::new(environments[0].schema().clone());
        for env in environments.iter().skip(1) {
            if env.schema() != schema.as_ref() {
                return Err(DataError::InvalidArgument {
                    message: "environment schemas must match".into(),
                });
            }
        }
        Ok(Self { schema, environments })
    }

    /// Shared schema.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Number of environments.
    #[must_use]
    pub fn env_count(&self) -> usize {
        self.environments.len()
    }

    /// Borrow environment `i`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn environment(&self, i: usize) -> Result<&TimeSeriesData, DataError> {
        self.environments
            .get(i)
            .ok_or(DataError::InvalidArgument {
                message: "environment index out of range".into(),
            })
    }

    /// All environments.
    #[must_use]
    pub fn environments(&self) -> &[TimeSeriesData] {
        &self.environments
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::testing::float_series;

    #[test]
    fn multi_env_requires_matching_schema() {
        let a = float_series(10, 1);
        let b = float_series(20, 1);
        let m = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        assert_eq!(m.env_count(), 2);
        assert_eq!(m.environment(1).unwrap().row_count(), 20);
    }
}

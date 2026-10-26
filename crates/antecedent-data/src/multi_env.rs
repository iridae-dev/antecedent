//! Multi-environment / multi-dataset container.
//!
//! Typed list of series sharing one schema. J-PCMCI+ discovery constraints
//! are wired in discovery.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalSchema;

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
            return Err(DataError::empty("multi-environment data environments"));
        }
        let schema = Arc::new(environments[0].schema().clone());
        for env in environments.iter().skip(1) {
            if env.schema() != schema.as_ref() {
                return Err(DataError::mismatch("environment schemas"));
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
        self.environments.get(i).ok_or(DataError::out_of_range(
            "environment",
            i,
            self.environments.len(),
        ))
    }

    /// All environments.
    #[must_use]
    pub fn environments(&self) -> &[TimeSeriesData] {
        &self.environments
    }

    /// These environments as a unified segmented source: one part per environment, in order.
    #[must_use]
    pub fn source(&self) -> crate::source::SegmentedSource {
        crate::source::SegmentedSource::from_multi_environment(self)
    }

    /// Replace environment `index`. The replacement must share the schema.
    ///
    /// # Errors
    ///
    /// `index` is past the environments, or the series has another schema.
    pub fn with_environment(
        &self,
        index: usize,
        series: TimeSeriesData,
    ) -> Result<Self, DataError> {
        let mut environments: Vec<TimeSeriesData> = self.environments.to_vec();
        let Some(slot) = environments.get_mut(index) else {
            return Err(DataError::out_of_range("environment", index, self.environments.len()));
        };
        *slot = series;
        Self::try_new(environments)
    }
}

#[cfg(test)]
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

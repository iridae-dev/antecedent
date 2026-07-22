//! Panel data: unit partitions with per-unit time indexes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{CausalSchema, VariableId};

use crate::column::ColumnView;
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::table::TableView;

/// One panel unit: a time series for a single cross-sectional unit.
#[derive(Clone, Debug)]
pub struct PanelUnit {
    /// Unit label / dense index.
    pub unit_id: u32,
    /// Per-unit time series (borrowed schema must match the panel).
    pub series: TimeSeriesData,
}

/// Panel dataset: multiple units sharing one schema, each with its own time index.
///
/// Units are stored by `Arc` so planners can reference them without cloning series.
#[derive(Clone, Debug)]
pub struct PanelData {
    schema: Arc<CausalSchema>,
    units: Arc<[PanelUnit]>,
}

impl PanelData {
    /// Construct from units; all must share an identical schema.
    ///
    /// # Errors
    ///
    /// Empty list or schema mismatch.
    pub fn try_new(units: impl Into<Arc<[PanelUnit]>>) -> Result<Self, DataError> {
        let units = units.into();
        if units.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "panel data needs ≥1 unit".into()
            });
        }
        let schema = Arc::new(units[0].series.schema().clone());
        for u in units.iter().skip(1) {
            if u.series.schema() != schema.as_ref() {
                return Err(DataError::InvalidArgument {
                    message: "panel unit schemas must match".into(),
                });
            }
        }
        Ok(Self { schema, units })
    }

    /// Shared schema.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Number of units.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Borrow unit `i`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn unit(&self, i: usize) -> Result<&PanelUnit, DataError> {
        self.units
            .get(i)
            .ok_or(DataError::InvalidArgument { message: "panel unit index out of range".into() })
    }

    /// All units.
    #[must_use]
    pub fn units(&self) -> &[PanelUnit] {
        &self.units
    }

    /// Total rows across all units (sum of lengths; not a contiguous table).
    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.units.iter().map(|u| u.series.row_count()).sum()
    }

    /// View panel units as multi-environment series (shared schema, one env per unit).
    ///
    /// Used for pooled discovery (e.g. J-PCMCI+) without cloning payloads.
    ///
    /// # Errors
    ///
    /// Propagates [`MultiEnvironmentData::try_new`] failures (should not occur for a
    /// well-formed panel).
    pub fn as_multi_env(&self) -> Result<crate::multi_env::MultiEnvironmentData, DataError> {
        let series: Vec<TimeSeriesData> = self.units.iter().map(|u| u.series.clone()).collect();
        crate::multi_env::MultiEnvironmentData::try_new(Arc::from(series))
    }
}

/// Borrowed view of a single panel unit as a table.
pub struct PanelUnitView<'a> {
    unit: &'a PanelUnit,
}

impl<'a> PanelUnitView<'a> {
    /// Wrap a unit.
    #[must_use]
    pub fn new(unit: &'a PanelUnit) -> Self {
        Self { unit }
    }
}

impl TableView for PanelUnitView<'_> {
    fn schema(&self) -> &CausalSchema {
        self.unit.series.schema()
    }

    fn row_count(&self) -> usize {
        self.unit.series.row_count()
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.unit.series.column(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::float_series;

    #[test]
    fn rejects_empty() {
        assert!(PanelData::try_new(Arc::from([])).is_err());
    }

    #[test]
    fn builds_two_unit_panel() {
        let panel = PanelData::try_new(Arc::from([
            PanelUnit { unit_id: 0, series: float_series(10, 2) },
            PanelUnit { unit_id: 1, series: float_series(12, 2) },
        ]))
        .unwrap();
        assert_eq!(panel.unit_count(), 2);
        assert_eq!(panel.total_rows(), 22);
    }
}

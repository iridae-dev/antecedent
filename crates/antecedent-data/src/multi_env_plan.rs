//! Multi-environment / panel sample planning without per-env full series clones.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::multi_env::MultiEnvironmentData;
use crate::panel::PanelData;
use crate::reference::ReferencePointPolicy;
use crate::sample::{LagMap, LaggedColumn, LaggedSamplePlan};
use crate::table::TableView;

/// Build one [`LaggedSamplePlan`] per series length, sharing [`LagMap`] Arcs for equal lengths.
///
/// # Errors
///
/// Empty column list, or invalid lag maps / column lags.
pub fn plans_for_series_lengths(
    lengths: impl IntoIterator<Item = usize>,
    max_lag: u32,
    columns: &Arc<[LaggedColumn]>,
) -> Result<Vec<LaggedSamplePlan>, DataError> {
    if columns.is_empty() {
        return Err(DataError::InvalidArgument { message: "sample plan needs ≥1 column".into() });
    }
    let mut by_len: HashMap<usize, Arc<LagMap>> = HashMap::new();
    let mut plans = Vec::new();
    for series_len in lengths {
        let lag_map = if let Some(m) = by_len.get(&series_len) {
            Arc::clone(m)
        } else {
            let m = Arc::new(LagMap::with_reference(
                series_len,
                max_lag,
                ReferencePointPolicy::SeriesOrigin,
            )?);
            by_len.insert(series_len, Arc::clone(&m));
            m
        };
        plans.push(LaggedSamplePlan::with_shared(lag_map, Arc::clone(columns))?);
    }
    Ok(plans)
}

/// Shared lag maps + column list for multi-environment sample planning.
///
/// Environments are only borrowed from the parent [`MultiEnvironmentData`].
/// Plans share one [`LaggedColumn`] Arc and reuse [`LagMap`] Arcs for equal lengths.
#[derive(Clone, Debug)]
pub struct MultiEnvSamplePlan {
    /// Shared lagged-column specification.
    pub columns: Arc<[LaggedColumn]>,
    /// One plan per environment (index-aligned with the parent container).
    pub plans: Arc<[LaggedSamplePlan]>,
}

impl MultiEnvSamplePlan {
    /// Build plans for every environment without cloning series payloads.
    ///
    /// # Errors
    ///
    /// Empty multi-env, empty columns, or invalid lag maps.
    pub fn try_from_multi_env(
        data: &MultiEnvironmentData,
        max_lag: u32,
        columns: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<Self, DataError> {
        let columns = columns.into();
        let n_env = data.env_count();
        if n_env == 0 {
            return Err(DataError::InvalidArgument {
                message: "multi-env sample plan needs ≥1 environment".into(),
            });
        }
        let lengths = (0..n_env).map(|i| data.environment(i).map(TableView::row_count));
        let lengths: Result<Vec<_>, _> = lengths.collect();
        let plans = plans_for_series_lengths(lengths?, max_lag, &columns)?;
        Ok(Self { columns, plans: Arc::from(plans) })
    }

    /// Number of environment plans.
    #[must_use]
    pub fn env_count(&self) -> usize {
        self.plans.len()
    }

    /// Borrow plan for environment `i`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn plan(&self, i: usize) -> Result<&LaggedSamplePlan, DataError> {
        self.plans.get(i).ok_or(DataError::InvalidArgument {
            message: "multi-env plan index out of range".into(),
        })
    }
}

/// Shared column list + per-unit plans for panel data.
#[derive(Clone, Debug)]
pub struct PanelSamplePlan {
    /// Shared lagged-column specification.
    pub columns: Arc<[LaggedColumn]>,
    /// One plan per panel unit.
    pub plans: Arc<[LaggedSamplePlan]>,
}

impl PanelSamplePlan {
    /// Build plans for every unit without cloning sibling series.
    ///
    /// # Errors
    ///
    /// Empty panel, empty columns, or invalid lag maps.
    pub fn try_from_panel(
        panel: &PanelData,
        max_lag: u32,
        columns: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<Self, DataError> {
        let columns = columns.into();
        if panel.unit_count() == 0 {
            return Err(DataError::InvalidArgument {
                message: "panel sample plan needs ≥1 unit".into(),
            });
        }
        let lengths = (0..panel.unit_count()).map(|i| panel.unit(i).map(|u| u.series.row_count()));
        let lengths: Result<Vec<_>, _> = lengths.collect();
        let plans = plans_for_series_lengths(lengths?, max_lag, &columns)?;
        Ok(Self { columns, plans: Arc::from(plans) })
    }

    /// Number of unit plans.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.plans.len()
    }
}

/// Pointer identity helper for copy-avoidance tests.
#[cfg(test)]
#[must_use]
pub(crate) fn series_columnar_ptr(series: &TimeSeriesData) -> *const [crate::column::OwnedColumn] {
    series.columnar_ptr()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{Lag, VariableId};

    use super::*;
    use crate::panel::PanelUnit;
    use crate::testing::float_series;

    fn one_col() -> Arc<[LaggedColumn]> {
        Arc::from([LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::CONTEMPORANEOUS }])
    }

    #[test]
    fn multi_env_plan_shares_geometry_without_cloning_series() {
        let a = float_series(40, 2);
        let b = float_series(50, 2);
        let c = float_series(40, 2); // same length as `a` → shared LagMap
        let ptr_a = series_columnar_ptr(&a);
        let ptr_b = series_columnar_ptr(&b);
        let ptr_c = series_columnar_ptr(&c);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b, c])).unwrap();

        let plan = MultiEnvSamplePlan::try_from_multi_env(&multi, 2, one_col()).unwrap();
        assert_eq!(plan.env_count(), 3);
        assert!(Arc::ptr_eq(plan.plans[0].columns_arc(), plan.plans[1].columns_arc()));
        assert!(Arc::ptr_eq(plan.plans[0].lag_map_arc(), plan.plans[2].lag_map_arc()));
        assert!(!Arc::ptr_eq(plan.plans[0].lag_map_arc(), plan.plans[1].lag_map_arc()));

        // Sibling environments unchanged after planning (no full series clone).
        assert_eq!(series_columnar_ptr(multi.environment(0).unwrap()), ptr_a);
        assert_eq!(series_columnar_ptr(multi.environment(1).unwrap()), ptr_b);
        assert_eq!(series_columnar_ptr(multi.environment(2).unwrap()), ptr_c);
    }

    #[test]
    fn panel_plan_shares_columns() {
        let panel = PanelData::try_new(Arc::from([
            PanelUnit { unit_id: 0, series: float_series(30, 2) },
            PanelUnit { unit_id: 1, series: float_series(30, 2) },
        ]))
        .unwrap();
        let plan = PanelSamplePlan::try_from_panel(&panel, 1, one_col()).unwrap();
        assert_eq!(plan.unit_count(), 2);
        assert!(Arc::ptr_eq(plan.plans[0].columns_arc(), plan.plans[1].columns_arc()));
        assert!(Arc::ptr_eq(plan.plans[0].lag_map_arc(), plan.plans[1].lag_map_arc()));
    }
}

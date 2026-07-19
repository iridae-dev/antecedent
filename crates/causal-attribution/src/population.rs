//! Resolve [`PopulationSelector`] into row indices.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::PopulationSelector;
use causal_data::{MultiEnvironmentData, TableView, TabularData, TimeSeriesData};

use crate::error::AttributionError;

/// Resolve a population selector against a single tabular table.
///
/// # Errors
///
/// Out-of-range rows / time ranges, or environment selectors on non-multi-env data.
pub fn resolve_rows(
    data: &TabularData,
    selector: &PopulationSelector,
) -> Result<Vec<usize>, AttributionError> {
    resolve_rows_n(data.row_count(), selector)
}

fn resolve_rows_n(n: usize, selector: &PopulationSelector) -> Result<Vec<usize>, AttributionError> {
    match selector {
        PopulationSelector::All => Ok((0..n).collect()),
        PopulationSelector::Rows(rows) => {
            for &r in rows.iter() {
                if r >= n {
                    return Err(AttributionError::PopulationOutOfRange {
                        kind: "row",
                        index: r,
                        limit: n,
                    });
                }
            }
            Ok(rows.to_vec())
        }
        PopulationSelector::TimeRange { start, end } => {
            if *end > n {
                return Err(AttributionError::PopulationOutOfRange {
                    kind: "time_range_end",
                    index: *end,
                    limit: n,
                });
            }
            Ok((*start..*end).collect())
        }
        PopulationSelector::Environment { .. } => Err(AttributionError::unsupported(
            "environment selector requires MultiEnvironmentData",
        )),
        _ => Err(AttributionError::unsupported("unsupported population selector")),
    }
}

/// Resolve a population selector against multi-environment data.
///
/// Returns `(env_index, row_indices_within_env)`.
///
/// # Errors
///
/// Unknown environment or invalid rows.
pub fn resolve_multi_env_rows(
    data: &MultiEnvironmentData,
    selector: &PopulationSelector,
) -> Result<(usize, Vec<usize>), AttributionError> {
    match selector {
        PopulationSelector::Environment { env_index } => {
            let env = data.environment(*env_index)?;
            Ok((*env_index, (0..env.row_count()).collect()))
        }
        PopulationSelector::All if data.env_count() == 1 => {
            let env = data.environment(0)?;
            Ok((0, (0..env.row_count()).collect()))
        }
        PopulationSelector::Rows(_) | PopulationSelector::TimeRange { .. }
            if data.env_count() == 1 =>
        {
            let env = data.environment(0)?;
            Ok((0, resolve_rows_n(env.row_count(), selector)?))
        }
        _ => Err(AttributionError::unsupported(
            "cannot resolve selector against multi-env data",
        )),
    }
}

/// Borrow a multi-env series by resolved index.
///
/// # Errors
///
/// Out of range.
pub fn multi_env_series(
    data: &MultiEnvironmentData,
    env_index: usize,
) -> Result<&TimeSeriesData, AttributionError> {
    Ok(data.environment(env_index)?)
}

/// Slice float columns for selected rows into a new owned [`TabularData`].
///
/// # Errors
///
/// Data access failures.
pub fn subset_table(data: &TabularData, rows: &[usize]) -> Result<TabularData, AttributionError> {
    use std::sync::Arc;

    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};

    let schema = data.schema().clone();
    let n = rows.len();
    let mut cols = Vec::with_capacity(schema.len());
    for var in schema.variables() {
        let id = var.id;
        let src = data.float64_values(id)?;
        let mut values = Vec::with_capacity(n);
        for &r in rows {
            values.push(src[r]);
        }
        let validity = ValidityBitmap::all_valid(n);
        cols.push(OwnedColumn::Float64(
            Float64Column::new(id, Arc::from(values), validity)?,
        ));
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)?;
    Ok(TabularData::new(storage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};

    fn tiny_table(n: usize) -> TabularData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let xv: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let cols = vec![OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(xv),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        )];
        TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
    }

    #[test]
    fn resolve_time_range_and_subset() {
        let data = tiny_table(10);
        let rows =
            resolve_rows(&data, &PopulationSelector::TimeRange { start: 2, end: 5 }).unwrap();
        assert_eq!(rows, vec![2, 3, 4]);
        let sub = subset_table(&data, &rows).unwrap();
        assert_eq!(sub.row_count(), 3);
    }
}

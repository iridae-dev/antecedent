//! Resolve [`PopulationSelector`] into row indices.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::PopulationSelector;
use antecedent_data::{TableView, TabularData};

use crate::error::AttributionError;

/// Row indices an anomaly or unit-change query scores.
///
/// `None` is every row. An index at or past `n_rows`, or more rows than
/// `max_units`, is the same typed error either caller used to raise.
///
/// # Errors
///
/// [`AttributionError::PopulationOutOfRange`] or [`AttributionError::SizeLimit`].
pub(crate) fn resolve_unit_rows(
    unit_rows: Option<&[usize]>,
    n_rows: usize,
    max_units: usize,
) -> Result<Vec<usize>, AttributionError> {
    let rows = match unit_rows {
        Some(rows) => rows.to_vec(),
        None => (0..n_rows).collect(),
    };
    for &row in &rows {
        if row >= n_rows {
            return Err(AttributionError::PopulationOutOfRange {
                kind: "row",
                index: row,
                limit: n_rows,
            });
        }
    }
    if rows.len() > max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: rows.len(),
            max: max_units,
        });
    }
    Ok(rows)
}

/// Resolve a population selector against a single tabular table.
///
/// # Errors
///
/// Out-of-range rows / time ranges, or environment selectors on non-multi-env data.
pub fn resolve_rows(
    data: &TabularData,
    selector: &PopulationSelector,
) -> Result<Vec<usize>, AttributionError> {
    let mut rows = resolve_rows_n(data.row_count(), selector)?;
    if let Some(mask) = data.storage().analysis_mask() {
        rows.retain(|&row| mask.is_valid(row));
    }
    Ok(rows)
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
        PopulationSelector::Environment { .. } => {
            Err(AttributionError::unsupported("environment selector requires MultiEnvironmentData"))
        }
        _ => Err(AttributionError::unsupported("unsupported population selector")),
    }
}

/// Reject stored observation weights until the consuming attribution fit uses them.
pub(crate) fn reject_storage_weights(data: &TabularData) -> Result<(), AttributionError> {
    if data.storage().weights().is_some() {
        return Err(AttributionError::unsupported(
            "attribution does not support table observation weights",
        ));
    }
    Ok(())
}

/// Compaction changes temporal adjacency, so ordered operations must refuse a hidden row.
pub(crate) fn reject_mask_for_ordered(data: &TabularData) -> Result<(), AttributionError> {
    if data.storage().analysis_mask().is_some_and(|mask| !mask.is_all_valid()) {
        return Err(AttributionError::unsupported(
            "ordered attribution does not support a nontrivial analysis mask",
        ));
    }
    Ok(())
}

/// Slice float columns for selected rows into a new owned [`TabularData`].
///
/// # Errors
///
/// Data access failures.
pub(crate) fn subset_table(
    data: &TabularData,
    rows: &[usize],
) -> Result<TabularData, AttributionError> {
    use std::sync::Arc;

    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};

    reject_storage_weights(data)?;
    let rows: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&r| data.storage().analysis_mask().is_none_or(|mask| mask.is_valid(r)))
        .collect();
    let schema = data.schema().clone();
    let n = rows.len();
    let mut cols = Vec::with_capacity(schema.len());
    for var in schema.variables() {
        let id = var.id;
        let src = data.float64_values(id)?;
        let mut values = Vec::with_capacity(n);
        for &r in &rows {
            values.push(src[r]);
        }
        let validity = ValidityBitmap::all_valid(n);
        cols.push(OwnedColumn::Float64(Float64Column::new(id, Arc::from(values), validity)?));
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)?;
    Ok(TabularData::new(storage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};

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
    #[test]
    fn selected_rows_respect_analysis_mask_and_reject_stored_weights() {
        let data = tiny_table(4)
            .with_analysis_mask(
                ValidityBitmap::from_bytes(Arc::<[u8]>::from([0b0000_1101]), 4).unwrap(),
            )
            .unwrap();
        let selector = PopulationSelector::Rows(Arc::from([3usize, 1, 2, 2]));
        let rows = resolve_rows(&data, &selector).unwrap();
        assert_eq!(rows, vec![3, 2, 2]);
        let subset = subset_table(&data, &[3, 1, 2, 2]).unwrap();
        assert_eq!(subset.float64_values(VariableId::from_raw(0)).unwrap(), vec![3.0, 2.0, 2.0]);
        let weighted = data.with_weights(Arc::from([1.0, 1.0, 1.0, 1.0])).unwrap();
        assert!(matches!(
            subset_table(&weighted, &rows),
            Err(AttributionError::Unsupported { .. })
        ));
    }
}

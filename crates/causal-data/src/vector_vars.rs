//! pinned baseline-style vector variable groups.
//!
//! Components are separate Float64 columns. Logical discovery nodes are the
//! first component of each group; [`column_blocks_for_frame`] builds per-lag
//! frame-index blocks for pairwise multivariate CI.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use causal_core::{CausalSchemaBuilder, Lag, MeasurementSpec, ValueType, VariableId};

use crate::column::{Float64Column, OwnedColumn};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::lagged_frame::LaggedFrame;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// Ordered groups of component variable ids (pinned baseline `vector_vars` style).
///
/// The first id in each group is the **logical** discovery node; remaining ids
/// are CI-only components excluded from the search variable list.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VectorVariableGroups {
    groups: Arc<[Arc<[VariableId]>]>,
}

impl VectorVariableGroups {
    /// Empty groups (scalar CI).
    #[must_use]
    pub fn empty() -> Self {
        Self { groups: Arc::from([]) }
    }

    /// Construct from component groups.
    ///
    /// # Errors
    ///
    /// Empty group or duplicate component across groups.
    pub fn try_new(groups: impl Into<Arc<[Arc<[VariableId]>]>>) -> Result<Self, DataError> {
        let groups = groups.into();
        let mut seen = HashSet::new();
        for g in groups.iter() {
            if g.is_empty() {
                return Err(DataError::InvalidArgument {
                    message: "vector variable group must be non-empty".into(),
                });
            }
            for &id in g.iter() {
                if !seen.insert(id) {
                    return Err(DataError::InvalidArgument {
                        message: format!(
                            "vector variable component {id} appears in multiple groups"
                        ),
                    });
                }
            }
        }
        Ok(Self { groups })
    }

    /// Whether any group is present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Borrow groups.
    #[must_use]
    pub fn groups(&self) -> &[Arc<[VariableId]>] {
        &self.groups
    }

    /// Logical discovery nodes (first component of each group).
    #[must_use]
    pub fn logical_ids(&self) -> Vec<VariableId> {
        self.groups.iter().filter_map(|g| g.first().copied()).collect()
    }

    /// Component ids that are not logical heads (excluded from search vars).
    #[must_use]
    pub fn secondary_component_ids(&self) -> Vec<VariableId> {
        let mut out = Vec::new();
        for g in self.groups.iter() {
            for &id in g.iter().skip(1) {
                out.push(id);
            }
        }
        out
    }

    /// Filter `variables` to logical search nodes (drop secondary components).
    #[must_use]
    pub fn filter_search_variables(&self, variables: &[VariableId]) -> Vec<VariableId> {
        let secondary: HashSet<VariableId> = self.secondary_component_ids().into_iter().collect();
        variables.iter().copied().filter(|v| !secondary.contains(v)).collect()
    }
}

/// Build per-lag column blocks for a lagged frame from vector-variable groups.
///
/// For each multi-component group and each lag `0..=max_lag`, emits one block of
/// frame column indexes `[idx(c0,τ), idx(c1,τ), …]`.
///
/// # Errors
///
/// Missing lagged columns for a component.
pub fn column_blocks_for_frame(
    groups: &VectorVariableGroups,
    frame: &LaggedFrame,
) -> Result<Arc<[Arc<[usize]>]>, DataError> {
    if groups.is_empty() {
        return Ok(Arc::from([]));
    }
    let max_lag = frame.max_lag();
    let mut blocks: Vec<Arc<[usize]>> = Vec::new();
    for g in groups.groups() {
        if g.len() < 2 {
            continue;
        }
        for lag in 0..=max_lag {
            let lag = Lag::from_raw(lag);
            let mut block = Vec::with_capacity(g.len());
            for &id in g.iter() {
                let Some(idx) = frame.column_index(id, lag) else {
                    return Err(DataError::InvalidArgument {
                        message: format!(
                            "vector component {id} lag {lag:?} missing from lagged frame"
                        ),
                    });
                };
                block.push(idx);
            }
            blocks.push(Arc::from(block));
        }
    }
    Ok(Arc::from(blocks))
}

/// Expand every [`FixedVectorColumn`] in `data` into `dim` Float64 columns and
/// register them as vector-variable groups.
///
/// The original vector variable id becomes the first (logical) component; additional
/// component ids are appended to the schema as `{name}__c{k}`.
///
/// # Errors
///
/// Schema construction failure, width mismatch, or unsupported non-float companion columns.
pub fn expand_fixed_vector_columns(
    data: &TimeSeriesData,
) -> Result<(TimeSeriesData, VectorVariableGroups), DataError> {
    let storage = data.storage();
    let n = storage.row_count();
    let schema = storage.schema();
    let mut builder = CausalSchemaBuilder::new();
    let mut new_cols: Vec<OwnedColumn> = Vec::new();
    let mut groups: Vec<Arc<[VariableId]>> = Vec::new();
    let mut next_id = 0u32;

    for old in storage.columns() {
        match old {
            OwnedColumn::FixedVector(fv) => {
                let width = NonZeroU32::new(u32::try_from(fv.dim).map_err(|_| {
                    DataError::InvalidArgument { message: "fixed vector dim overflow".into() }
                })?)
                .ok_or_else(|| DataError::InvalidArgument {
                    message: "fixed vector dim must be > 0".into(),
                })?;
                let old_schema = schema.get(fv.id).map_err(|e| DataError::Schema(e.to_string()))?;
                if let ValueType::Vector { width: declared, .. } = &old_schema.value_type {
                    if declared != &width {
                        return Err(DataError::InvalidArgument {
                            message: format!(
                                "FixedVector dim {} != schema Vector width {}",
                                fv.dim,
                                declared.get()
                            ),
                        });
                    }
                }
                let mut comp_ids = Vec::with_capacity(fv.dim);
                for k in 0..fv.dim {
                    let name = if k == 0 {
                        Arc::clone(&old_schema.name)
                    } else {
                        Arc::<str>::from(format!("{}__c{k}", old_schema.name))
                    };
                    builder
                        .add_variable(
                            name,
                            ValueType::Continuous,
                            old_schema.role_hints,
                            old_schema.unit.clone(),
                            None,
                            MeasurementSpec::default(),
                        )
                        .map_err(|e| DataError::Schema(e.to_string()))?;
                    let id = VariableId::from_raw(next_id);
                    next_id += 1;
                    comp_ids.push(id);
                    let mut values = vec![0.0; n];
                    for (row, slot) in values.iter_mut().enumerate() {
                        *slot = fv.values[row * fv.dim + k];
                    }
                    new_cols.push(OwnedColumn::Float64(Float64Column::new(
                        id,
                        Arc::from(values),
                        fv.validity.clone(),
                    )?));
                }
                groups.push(Arc::from(comp_ids));
            }
            OwnedColumn::Float64(c) => {
                let old_schema = schema.get(c.id).map_err(|e| DataError::Schema(e.to_string()))?;
                builder
                    .add_variable(
                        Arc::clone(&old_schema.name),
                        old_schema.value_type.clone(),
                        old_schema.role_hints,
                        old_schema.unit.clone(),
                        old_schema.category_domain,
                        old_schema.measurement.clone(),
                    )
                    .map_err(|e| DataError::Schema(e.to_string()))?;
                let new_id = VariableId::from_raw(next_id);
                next_id += 1;
                new_cols.push(OwnedColumn::Float64(Float64Column::new(
                    new_id,
                    c.values.clone(),
                    c.validity.clone(),
                )?));
            }
            other => {
                return Err(DataError::InvalidArgument {
                    message: format!(
                        "expand_fixed_vector_columns: unsupported column kind for id {}",
                        other.id()
                    ),
                });
            }
        }
    }

    let new_schema = builder.build().map_err(|e| DataError::Schema(e.to_string()))?;
    let new_storage = OwnedColumnarStorage::try_new(
        new_schema,
        new_cols,
        storage.analysis_mask().cloned(),
        storage.weights().map(Arc::from),
    )?;
    let series = TimeSeriesData::try_new(new_storage, data.time_index().clone())?;
    let groups = VectorVariableGroups::try_new(Arc::from(groups))?;
    Ok((series, groups))
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::column::{FixedVectorColumn, ValidityBitmap};
    use crate::testing::float_series;

    #[test]
    fn column_blocks_per_lag() {
        let data = float_series(20, 3);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let frame =
            LaggedFrame::from_series(&data, &vars, 1, &causal_core::KernelPolicy::default_policy())
                .unwrap();
        let groups = VectorVariableGroups::try_new(Arc::from([Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ])]))
        .unwrap();
        let blocks = column_blocks_for_frame(&groups, &frame).unwrap();
        // 2 lags (0,1) × 1 group
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 2);
        assert_eq!(
            blocks[0].as_ref(),
            &[
                frame.column_index(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap(),
                frame.column_index(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap(),
            ]
        );
    }

    #[test]
    fn filter_drops_secondary_components() {
        let groups = VectorVariableGroups::try_new(Arc::from([Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ])]))
        .unwrap();
        let search = groups.filter_search_variables(&[
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ]);
        assert_eq!(search, vec![VariableId::from_raw(0), VariableId::from_raw(2)]);
    }

    #[test]
    fn expand_fixed_vector_round_trip() {
        use crate::temporal::{SamplingRegularity, TimeIndex};
        use causal_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };

        let n = 10usize;
        let dim = 2usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "v",
            ValueType::Vector {
                width: NonZeroU32::new(2).unwrap(),
                element: causal_core::ScalarType::Float64,
            },
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut fv_vals = vec![0.0; n * dim];
        let mut y = vec![0.0; n];
        for t in 0..n {
            fv_vals[t * dim] = t as f64;
            fv_vals[t * dim + 1] = (t as f64) + 100.0;
            y[t] = t as f64;
        }
        let cols = vec![
            OwnedColumn::FixedVector(
                FixedVectorColumn::new(
                    VariableId::from_raw(0),
                    dim,
                    Arc::from(fv_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let (expanded, groups) = expand_fixed_vector_columns(&data).unwrap();
        assert_eq!(expanded.storage().columns().len(), 3);
        assert_eq!(groups.groups().len(), 1);
        assert_eq!(groups.groups()[0].len(), 2);
        assert_eq!(groups.logical_ids(), vec![VariableId::from_raw(0)]);
    }
}

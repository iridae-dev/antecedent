//! Vector variable groups for multivariate conditional-independence testing.
//!
//! Components are separate Float64 columns. Logical discovery nodes are the
//! first component of each group; [`column_blocks_for_frame`] builds per-lag
//! frame-index blocks for pairwise multivariate CI.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{Lag, VariableId};

use crate::error::DataError;
use crate::lagged_frame::LaggedFrame;

/// Ordered groups of component variable ids (a vector-variable grouping for multivariate CI).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::float_series;

    #[test]
    fn column_blocks_per_lag() {
        let data = float_series(20, 3);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let frame = LaggedFrame::from_series(
            &data,
            &vars,
            1,
            &antecedent_core::KernelPolicy::default_policy(),
        )
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
}

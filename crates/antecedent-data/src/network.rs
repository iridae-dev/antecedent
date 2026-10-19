//! Unit-level tabular data with a fixed interference network.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{DataError, GlobalId, TableView, TabularData};

/// Directed weighted edge from a source unit to an exposed target unit.
///
/// `from` and `to` are [`GlobalId`]s. They are not narrowed to `u32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkEdge {
    /// Source unit row.
    pub from: GlobalId,
    /// Target unit row.
    pub to: GlobalId,
    /// Non-negative exposure weight.
    pub weight: f64,
}

impl NetworkEdge {
    /// Edge between two row identities.
    #[must_use]
    pub const fn new(from: u64, to: u64, weight: f64) -> Self {
        Self { from: GlobalId::new(from), to: GlobalId::new(to), weight }
    }
}

/// Tabular unit data paired with a fixed, row-indexed network.
///
/// Incoming adjacency is stored CSR-style: one edge array sorted by
/// `(to, from)` plus an offsets table, so construction is two allocations
/// (the array-of-`Arc`s layout it replaces made `n + 1`, most of them empty)
/// and `incoming(unit)` is a contiguous slice.
#[derive(Clone, Debug)]
pub struct NetworkData {
    units: TabularData,
    edges: Arc<[NetworkEdge]>,
    /// All edges re-sorted by `(to, from)`; unit `u`'s incoming edges are
    /// `incoming_edges[incoming_offsets[u]..incoming_offsets[u + 1]]`.
    incoming_edges: Arc<[NetworkEdge]>,
    incoming_offsets: Arc<[usize]>,
}

impl NetworkData {
    /// Build a network, validating row indexes, weights, self-edges, and duplicates.
    ///
    /// # Errors
    ///
    /// [`DataError::InvalidArgument`] when an edge is invalid for the unit table.
    pub fn try_new(
        units: TabularData,
        edges: impl Into<Arc<[NetworkEdge]>>,
    ) -> Result<Self, DataError> {
        let edges = edges.into();
        let n = units.row_count();
        let mut sorted = edges.to_vec();
        sorted.sort_by_key(|edge| (edge.from, edge.to));
        for (i, edge) in sorted.iter().enumerate() {
            let from = table_row(edge.from, n)?;
            let to = table_row(edge.to, n)?;
            if from == to {
                return Err(DataError::InvalidArgument {
                    message: "network self-edges are not allowed".into(),
                });
            }
            if !edge.weight.is_finite() || edge.weight < 0.0 {
                return Err(DataError::InvalidArgument {
                    message: "network weights must be finite and non-negative".into(),
                });
            }
            if i > 0 && (sorted[i - 1].from, sorted[i - 1].to) == (edge.from, edge.to) {
                return Err(DataError::InvalidArgument {
                    message: "duplicate network edge".into(),
                });
            }
        }
        let edges: Arc<[NetworkEdge]> = sorted.into();
        // (to, from) order keeps each unit's incoming edges in ascending-from
        // order, matching the per-unit push order of the previous layout.
        let mut by_to = edges.to_vec();
        by_to.sort_by_key(|edge| (edge.to, edge.from));
        let mut incoming_offsets = vec![0usize; n + 1];
        for edge in &by_to {
            let to = table_row(edge.to, n)?;
            incoming_offsets[to + 1] += 1;
        }
        for u in 0..n {
            incoming_offsets[u + 1] += incoming_offsets[u];
        }
        Ok(Self {
            units,
            edges,
            incoming_edges: by_to.into(),
            incoming_offsets: incoming_offsets.into(),
        })
    }

    /// Borrow unit-level columns.
    #[must_use]
    pub const fn units(&self) -> &TabularData {
        &self.units
    }

    /// Borrow all directed network edges.
    #[must_use]
    pub fn edges(&self) -> &[NetworkEdge] {
        &self.edges
    }

    /// Incoming neighbors whose assignments define exposure for `unit`.
    ///
    /// # Errors
    ///
    /// [`DataError::InvalidArgument`] when `unit` is outside the table.
    pub fn incoming(&self, unit: usize) -> Result<&[NetworkEdge], DataError> {
        if unit + 1 >= self.incoming_offsets.len() {
            return Err(DataError::InvalidArgument {
                message: "network unit index is out of range".into(),
            });
        }
        Ok(&self.incoming_edges[self.incoming_offsets[unit]..self.incoming_offsets[unit + 1]])
    }
}

fn table_row(id: GlobalId, n: usize) -> Result<usize, DataError> {
    let index = usize::try_from(id.get()).map_err(|_| DataError::InvalidArgument {
        message: "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. \
                  A u32 backend feature flag is not that attempt"
            .into(),
    })?;
    if index >= n {
        return Err(DataError::InvalidArgument {
            message: "network edge row index is outside the unit table".into(),
        });
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_network_is_valid_and_has_no_incoming_edges() {
        let values = [1.0, 2.0];
        let table = TabularData::from_f64_columns([("y", values.as_slice())]).unwrap();
        let network = NetworkData::try_new(table, []).unwrap();
        assert!(network.incoming(0).unwrap().is_empty());
    }
}

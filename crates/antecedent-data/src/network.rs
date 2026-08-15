//! Unit-level tabular data with a fixed interference network.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{DataError, TableView, TabularData};

/// Directed weighted edge from a source unit to an exposed target unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkEdge {
    /// Source unit row.
    pub from: u32,
    /// Target unit row.
    pub to: u32,
    /// Non-negative exposure weight.
    pub weight: f64,
}

/// Tabular unit data paired with a fixed, row-indexed network.
#[derive(Clone, Debug)]
pub struct NetworkData {
    units: TabularData,
    edges: Arc<[NetworkEdge]>,
    incoming: Arc<[Arc<[NetworkEdge]>]>,
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
            if edge.from as usize >= n || edge.to as usize >= n {
                return Err(DataError::InvalidArgument {
                    message: "network edge row index is outside the unit table".into(),
                });
            }
            if edge.from == edge.to {
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
        let mut incoming = vec![Vec::new(); n];
        for edge in edges.iter().copied() {
            incoming[edge.to as usize].push(edge);
        }
        let incoming =
            incoming.into_iter().map(Arc::<[NetworkEdge]>::from).collect::<Vec<_>>().into();
        Ok(Self { units, edges, incoming })
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
        self.incoming.get(unit).map(AsRef::as_ref).ok_or(DataError::InvalidArgument {
            message: "network unit index is out of range".into(),
        })
    }
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

//! Name-resolved graph construction helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::CausalSchema;

use crate::error::GraphError;
use crate::types::DenseNodeId;

/// Resolve two schema variable names to dense node ids (`VariableId::raw` == dense id).
pub(crate) fn resolve_named_edge(
    schema: &CausalSchema,
    from_name: &str,
    to_name: &str,
) -> Result<(DenseNodeId, DenseNodeId), GraphError> {
    let from_id = schema
        .id_of(from_name)
        .map_err(|_| GraphError::UnknownVariableName {
            name: from_name.to_owned(),
        })?;
    let to_id = schema
        .id_of(to_name)
        .map_err(|_| GraphError::UnknownVariableName {
            name: to_name.to_owned(),
        })?;
    Ok((DenseNodeId::from_raw(from_id.raw()), DenseNodeId::from_raw(to_id.raw())))
}

/// Schema length as `u32` node count.
pub(crate) fn schema_node_count(schema: &CausalSchema) -> Result<u32, GraphError> {
    u32::try_from(schema.len()).map_err(|_| GraphError::TooManyNodes)
}

//! Lowering of the static mediation and structural-model cells: mediation and nested
//! counterfactuals on a DAG, and the counterfactual, attribution and change queries.
//!
//! Every cell takes the generic lowering until this family lowers its own layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::LoweringContext;
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    super::generic(cx)
}

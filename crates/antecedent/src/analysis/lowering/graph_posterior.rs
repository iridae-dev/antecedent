//! Lowering of studies that carry a graph posterior instead of one graph.
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

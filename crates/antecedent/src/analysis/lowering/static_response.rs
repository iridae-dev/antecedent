//! Lowering of the static response cells: DAG, PAG and CPDAG response curves, the
//! co-determined joint response, and the mixed-graph response.
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

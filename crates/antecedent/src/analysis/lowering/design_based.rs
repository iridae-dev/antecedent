//! Lowering of the design-based and specialist cells: transport, interference, randomized
//! effects, policy value, dose response, difference-in-differences, synthetic control,
//! local-polynomial ratio, survival, longitudinal regimes, sequential contrasts and
//! history matching.
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

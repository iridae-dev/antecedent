//! Lowering of the temporal cells: series effect, response and mediation on a
//! temporal DAG or class, temporal cross-world queries, multi-environment effects,
//! and panel effects and responses.
//!
//! Every cell other than the refused one takes the generic lowering until this
//! family lowers its own layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{LoweringContext, RouteCell};
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    if cx.cell == RouteCell::TemporalMediationPag {
        // Logical compile refuses this cell before lowering; the lowering refuses too.
        return Err(CausalError::Unsupported {
            message: "Latent-confounded TemporalPag natural mediation is outside the existing \
                      cross-world identification theory; the linear DAG mechanism product on \
                      completions cannot license it",
        });
    }
    super::generic(cx)
}

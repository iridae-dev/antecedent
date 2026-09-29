//! LaTeX rendering for diagnostics (not equality keys).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CausalExprArena, ExprId};

pub(crate) fn latex_expr(arena: &CausalExprArena, id: ExprId) -> String {
    crate::render::render_expr(arena, id, &crate::render::LATEX)
}

//! One child order for every expression walk.
//!
//! Kernel, product, sum, integral, ratio, expectation, and contrast list their
//! child ids here. A fold memoises the value of each id. Callers that used to
//! recurse with their own match keep that match for the value, and take the
//! children from this order.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use crate::{CausalExprArena, ExprId, ExprNode};

/// Child expression ids of `id`, in the order the direct recursions visit them.
pub(crate) fn child_ids(arena: &CausalExprArena, id: ExprId) -> Vec<ExprId> {
    match arena.node(id) {
        ExprNode::Distribution { .. } => Vec::new(),
        ExprNode::Kernel { body, .. } => vec![*body],
        ExprNode::Product(list) => arena.list(*list).to_vec(),
        ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => vec![*expr],
        ExprNode::Ratio { numerator, denominator } => vec![*numerator, *denominator],
        ExprNode::Expectation { distribution, .. } => vec![*distribution],
        ExprNode::Contrast { left, right, .. } => vec![*left, *right],
    }
}

/// Post-order fold. `fold` sees `id` after each child's value, in [`child_ids`] order.
pub(crate) fn fold_expr<T: Clone>(
    arena: &CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, T>,
    fold: &mut impl FnMut(&CausalExprArena, ExprId, &[T]) -> T,
) -> T {
    if let Some(cached) = memo.get(&id) {
        return cached.clone();
    }
    let children = child_ids(arena, id);
    let mut child_values = Vec::with_capacity(children.len());
    for child in children {
        child_values.push(fold_expr(arena, child, memo, fold));
    }
    let value = fold(arena, id, &child_values);
    memo.insert(id, value.clone());
    value
}

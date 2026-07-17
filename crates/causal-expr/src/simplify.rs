//! Algebraic simplification via worklist + memoization (DESIGN.md §9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::VariableId;

use crate::{CausalExprArena, DerivationMeta, ExprId, ExprNode, VarSetId};

/// Simplify `root` bottom-up with memoization; returns a (possibly new) `ExprId`.
pub(crate) fn simplify(arena: &mut CausalExprArena, root: ExprId) -> ExprId {
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let mut free_memo: HashMap<ExprId, VarSetId> = HashMap::new();
    simplify_rec(arena, root, &mut memo, &mut free_memo)
}

fn simplify_rec(
    arena: &mut CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, ExprId>,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> ExprId {
    if let Some(&cached) = memo.get(&id) {
        return cached;
    }
    let rebuilt = rebuild_children(arena, id, memo, free_memo);
    let simplified = apply_rules_fixpoint(arena, rebuilt, free_memo);
    memo.insert(id, simplified);
    simplified
}

fn rebuild_children(
    arena: &mut CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, ExprId>,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> ExprId {
    let node = arena.node(id).clone();
    match node {
        ExprNode::Distribution { .. } => id,
        ExprNode::Product(list) => {
            let children_ids: Vec<ExprId> = arena.list(list).to_vec();
            let children: Vec<ExprId> =
                children_ids.into_iter().map(|c| simplify_rec(arena, c, memo, free_memo)).collect();
            let list_id = arena.intern_list(children);
            arena.intern(ExprNode::Product(list_id))
        }
        ExprNode::SumOut { variables, expr } => {
            let body = simplify_rec(arena, expr, memo, free_memo);
            arena.intern(ExprNode::SumOut { variables, expr: body })
        }
        ExprNode::IntegralOut { variables, expr } => {
            let body = simplify_rec(arena, expr, memo, free_memo);
            arena.intern(ExprNode::IntegralOut { variables, expr: body })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let num = simplify_rec(arena, numerator, memo, free_memo);
            let den = simplify_rec(arena, denominator, memo, free_memo);
            arena.intern(ExprNode::Ratio { numerator: num, denominator: den })
        }
        ExprNode::Expectation { function, distribution } => {
            let dist = simplify_rec(arena, distribution, memo, free_memo);
            arena.intern(ExprNode::Expectation { function, distribution: dist })
        }
        ExprNode::Contrast { left, right, op } => {
            let l = simplify_rec(arena, left, memo, free_memo);
            let r = simplify_rec(arena, right, memo, free_memo);
            arena.intern(ExprNode::Contrast { left: l, right: r, op })
        }
    }
}

fn apply_rules_fixpoint(
    arena: &mut CausalExprArena,
    mut id: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> ExprId {
    // Local rules only; children are already simplified.
    loop {
        let next = apply_local_rules(arena, id, free_memo);
        if next == id {
            return id;
        }
        id = next;
    }
}

fn apply_local_rules(
    arena: &mut CausalExprArena,
    id: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> ExprId {
    match arena.node(id).clone() {
        ExprNode::SumOut { variables, expr } => {
            rewrite_sum_out(arena, id, variables, expr, free_memo)
        }
        ExprNode::IntegralOut { variables, expr } => {
            rewrite_integral_out(arena, id, variables, expr, free_memo)
        }
        ExprNode::Product(list) => rewrite_product(arena, id, list),
        ExprNode::Ratio { numerator, denominator } => {
            rewrite_ratio(arena, id, numerator, denominator)
        }
        _ => id,
    }
}

fn rewrite_sum_out(
    arena: &mut CausalExprArena,
    id: ExprId,
    variables: VarSetId,
    expr: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> ExprId {
    if arena.var_set(variables).is_empty() {
        return tag_if_new(arena, expr, id, "simplify.empty_sum_out");
    }
    if let ExprNode::SumOut { variables: inner_v, expr: inner_e } = arena.node(expr).clone() {
        let merged: Vec<VariableId> = arena
            .var_set(variables)
            .iter()
            .copied()
            .chain(arena.var_set(inner_v).iter().copied())
            .collect();
        let union = arena.intern_var_set(merged);
        let node = ExprNode::SumOut { variables: union, expr: inner_e };
        return intern_derived(arena, node, "simplify.merge_sum_out");
    }
    let free = free_vars(arena, expr, free_memo);
    if !intersects(arena, variables, free) {
        return tag_if_new(arena, expr, id, "simplify.dead_sum_out");
    }
    id
}

fn rewrite_integral_out(
    arena: &mut CausalExprArena,
    id: ExprId,
    variables: VarSetId,
    expr: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> ExprId {
    if arena.var_set(variables).is_empty() {
        return tag_if_new(arena, expr, id, "simplify.empty_integral_out");
    }
    if let ExprNode::IntegralOut { variables: inner_v, expr: inner_e } = arena.node(expr).clone() {
        let merged: Vec<VariableId> = arena
            .var_set(variables)
            .iter()
            .copied()
            .chain(arena.var_set(inner_v).iter().copied())
            .collect();
        let union = arena.intern_var_set(merged);
        let node = ExprNode::IntegralOut { variables: union, expr: inner_e };
        return intern_derived(arena, node, "simplify.merge_integral_out");
    }
    let free = free_vars(arena, expr, free_memo);
    if !intersects(arena, variables, free) {
        return tag_if_new(arena, expr, id, "simplify.dead_integral_out");
    }
    id
}

fn rewrite_product(arena: &mut CausalExprArena, id: ExprId, list: crate::ExprListId) -> ExprId {
    let children = arena.list(list).to_vec();
    if children.len() == 1 {
        return tag_if_new(arena, children[0], id, "simplify.singleton_product");
    }
    let mut flat: Vec<ExprId> = Vec::with_capacity(children.len());
    let mut flattened = false;
    for c in &children {
        if let ExprNode::Product(inner) = arena.node(*c) {
            flat.extend_from_slice(arena.list(*inner));
            flattened = true;
        } else {
            flat.push(*c);
        }
    }
    flat.sort_unstable();
    let sorted_changed = flat.as_slice() != children.as_slice();
    if flattened || sorted_changed {
        if flat.len() == 1 {
            return tag_if_new(arena, flat[0], id, "simplify.singleton_product");
        }
        let list_id = arena.intern_list(flat);
        let rule =
            if flattened { "simplify.flatten_product" } else { "simplify.canonical_product" };
        return intern_derived(arena, ExprNode::Product(list_id), rule);
    }
    id
}

fn rewrite_ratio(
    arena: &mut CausalExprArena,
    id: ExprId,
    numerator: ExprId,
    denominator: ExprId,
) -> ExprId {
    // (a/b)/c → a/(b*c)
    if let ExprNode::Ratio { numerator: a, denominator: b } = arena.node(numerator).clone() {
        let bc = {
            let mut kids = vec![b, denominator];
            kids.sort_unstable();
            let list = arena.intern_list(kids);
            arena.intern(ExprNode::Product(list))
        };
        return intern_derived(
            arena,
            ExprNode::Ratio { numerator: a, denominator: bc },
            "simplify.ratio_assoc_left",
        );
    }
    // a/(b/c) → (a*c)/b
    if let ExprNode::Ratio { numerator: b, denominator: c } = arena.node(denominator).clone() {
        let ac = {
            let mut kids = vec![numerator, c];
            kids.sort_unstable();
            let list = arena.intern_list(kids);
            arena.intern(ExprNode::Product(list))
        };
        return intern_derived(
            arena,
            ExprNode::Ratio { numerator: ac, denominator: b },
            "simplify.ratio_assoc_right",
        );
    }
    id
}

fn tag_if_new(arena: &mut CausalExprArena, result: ExprId, _from: ExprId, _rule: &str) -> ExprId {
    // Identity rewrite to an existing child — no new node; leave child's derivation alone.
    let _ = arena;
    result
}

fn intern_derived(arena: &mut CausalExprArena, node: ExprNode, rule: &str) -> ExprId {
    let before = arena.len();
    let id = arena.intern(node);
    if arena.len() > before {
        arena.set_derivation_if_absent(id, DerivationMeta { rule: Arc::from(rule), note: None });
    }
    id
}

fn intersects(arena: &CausalExprArena, a: VarSetId, b: VarSetId) -> bool {
    let av = arena.var_set(a);
    let bv = arena.var_set(b);
    let mut i = 0;
    let mut j = 0;
    while i < av.len() && j < bv.len() {
        match av[i].raw().cmp(&bv[j].raw()) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

fn free_vars(
    arena: &mut CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, VarSetId>,
) -> VarSetId {
    if let Some(&cached) = memo.get(&id) {
        return cached;
    }
    let result = match arena.node(id).clone() {
        ExprNode::Distribution { variables, conditioned_on, intervention, .. } => {
            let mut vars: Vec<VariableId> = arena.var_set(variables).to_vec();
            vars.extend_from_slice(arena.var_set(conditioned_on));
            // Intervention targets are bound by do(·), not free.
            let _ = intervention;
            arena.intern_var_set(vars)
        }
        ExprNode::Product(list) => {
            let children: Vec<ExprId> = arena.list(list).to_vec();
            let mut vars = Vec::new();
            for c in children {
                let fv = free_vars(arena, c, memo);
                vars.extend_from_slice(arena.var_set(fv));
            }
            arena.intern_var_set(vars)
        }
        ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
            let body = free_vars(arena, expr, memo);
            let bound = arena.var_set(variables);
            let remaining: Vec<VariableId> = arena
                .var_set(body)
                .iter()
                .copied()
                .filter(|v| !bound.iter().any(|b| b == v))
                .collect();
            arena.intern_var_set(remaining)
        }
        ExprNode::Ratio { numerator, denominator } => {
            let n = free_vars(arena, numerator, memo);
            let d = free_vars(arena, denominator, memo);
            let mut vars = arena.var_set(n).to_vec();
            vars.extend_from_slice(arena.var_set(d));
            arena.intern_var_set(vars)
        }
        ExprNode::Expectation { function, distribution } => {
            let dist = free_vars(arena, distribution, memo);
            let mut vars = arena.var_set(dist).to_vec();
            vars.push(function.variable());
            arena.intern_var_set(vars)
        }
        ExprNode::Contrast { left, right, .. } => {
            let l = free_vars(arena, left, memo);
            let r = free_vars(arena, right, memo);
            let mut vars = arena.var_set(l).to_vec();
            vars.extend_from_slice(arena.var_set(r));
            arena.intern_var_set(vars)
        }
    };
    memo.insert(id, result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContrastOp, DomainRef, OutcomeExprId};
    use causal_core::Value;

    #[test]
    fn empty_sum_out_eliminates() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let dist = a.intern(ExprNode::Distribution {
            variables: empty,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let summed = a.intern(ExprNode::SumOut { variables: empty, expr: dist });
        assert_eq!(simplify(&mut a, summed), dist);
    }

    #[test]
    fn merge_nested_sum_out() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let v1 = a.intern_var_set([VariableId::from_raw(1)]);
        let v2 = a.intern_var_set([VariableId::from_raw(2)]);
        let vars12 = a.intern_var_set([VariableId::from_raw(1), VariableId::from_raw(2)]);
        let dist = a.intern(ExprNode::Distribution {
            variables: vars12,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let inner = a.intern(ExprNode::SumOut { variables: v2, expr: dist });
        let outer = a.intern(ExprNode::SumOut { variables: v1, expr: inner });
        let s = simplify(&mut a, outer);
        match a.node(s) {
            ExprNode::SumOut { variables, expr } => {
                assert_eq!(
                    a.var_set(*variables),
                    &[VariableId::from_raw(1), VariableId::from_raw(2)]
                );
                assert_eq!(*expr, dist);
            }
            other => panic!("expected merged SumOut, got {other:?}"),
        }
    }

    #[test]
    fn dead_sum_out_eliminates() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let y = a.intern_var_set([VariableId::from_raw(0)]);
        let z = a.intern_var_set([VariableId::from_raw(1)]);
        let dist = a.intern(ExprNode::Distribution {
            variables: y,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let summed = a.intern(ExprNode::SumOut { variables: z, expr: dist });
        assert_eq!(simplify(&mut a, summed), dist);
    }

    #[test]
    fn singleton_and_flatten_product() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let v0 = a.intern_var_set([VariableId::from_raw(0)]);
        let v1 = a.intern_var_set([VariableId::from_raw(1)]);
        let d1 = a.intern(ExprNode::Distribution {
            variables: v0,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let d2 = a.intern(ExprNode::Distribution {
            variables: v1,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let inner = {
            let list = a.intern_list([d1]);
            a.intern(ExprNode::Product(list))
        };
        assert_eq!(simplify(&mut a, inner), d1);

        let nest = {
            let list_inner = a.intern_list([d1, d2]);
            let p_inner = a.intern(ExprNode::Product(list_inner));
            let list_outer = a.intern_list([p_inner, d1]);
            a.intern(ExprNode::Product(list_outer))
        };
        let s = simplify(&mut a, nest);
        match a.node(s) {
            ExprNode::Product(list) => {
                let kids = a.list(*list);
                assert_eq!(kids.len(), 3);
                let mut sorted = kids.to_vec();
                sorted.sort_unstable();
                assert_eq!(kids, sorted.as_slice());
            }
            other => panic!("expected product, got {other:?}"),
        }
    }

    #[test]
    fn product_order_independent() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let v0 = a.intern_var_set([VariableId::from_raw(0)]);
        let v1 = a.intern_var_set([VariableId::from_raw(1)]);
        let d1 = a.intern(ExprNode::Distribution {
            variables: v0,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let d2 = a.intern(ExprNode::Distribution {
            variables: v1,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let p1 = {
            let list = a.intern_list([d1, d2]);
            a.intern(ExprNode::Product(list))
        };
        let p2 = {
            let list = a.intern_list([d2, d1]);
            a.intern(ExprNode::Product(list))
        };
        assert_eq!(simplify(&mut a, p1), simplify(&mut a, p2));
    }

    #[test]
    fn simplify_idempotent() {
        let mut a = CausalExprArena::new();
        let id = a.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        let s1 = simplify(&mut a, id);
        let s2 = simplify(&mut a, s1);
        assert_eq!(s1, s2);
    }

    #[test]
    fn ratio_assoc_left() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let v0 = a.intern_var_set([VariableId::from_raw(0)]);
        let v1 = a.intern_var_set([VariableId::from_raw(1)]);
        let v2 = a.intern_var_set([VariableId::from_raw(2)]);
        let da = a.intern(ExprNode::Distribution {
            variables: v0,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let db = a.intern(ExprNode::Distribution {
            variables: v1,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let dc = a.intern(ExprNode::Distribution {
            variables: v2,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let ab = a.intern(ExprNode::Ratio { numerator: da, denominator: db });
        let nested = a.intern(ExprNode::Ratio { numerator: ab, denominator: dc });
        let s = simplify(&mut a, nested);
        match a.node(s) {
            ExprNode::Ratio { numerator, denominator } => {
                assert_eq!(*numerator, da);
                match a.node(*denominator) {
                    ExprNode::Product(list) => {
                        let kids = a.list(*list);
                        assert_eq!(kids.len(), 2);
                        assert!(kids.contains(&db) && kids.contains(&dc));
                    }
                    other => panic!("expected product denom, got {other:?}"),
                }
            }
            other => panic!("expected ratio, got {other:?}"),
        }
    }

    #[test]
    fn contrast_rebuilds_children() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let dist = a.intern(ExprNode::Distribution {
            variables: empty,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let summed = a.intern(ExprNode::SumOut { variables: empty, expr: dist });
        let exp = a.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(VariableId::from_raw(0)),
            distribution: summed,
        });
        let contrast =
            a.intern(ExprNode::Contrast { left: exp, right: exp, op: ContrastOp::Difference });
        let s = simplify(&mut a, contrast);
        match a.node(s) {
            ExprNode::Contrast { left, right, .. } => {
                match a.node(*left) {
                    ExprNode::Expectation { distribution, .. } => assert_eq!(*distribution, dist),
                    other => panic!("expected expectation, got {other:?}"),
                }
                assert_eq!(left, right);
            }
            other => panic!("expected contrast, got {other:?}"),
        }
    }
}

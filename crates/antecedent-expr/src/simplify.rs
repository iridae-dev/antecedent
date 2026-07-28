//! Algebraic simplification via worklist + memoization.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use antecedent_core::VariableId;

use crate::{CausalExprArena, DerivationMeta, ExprId, ExprNode, VarSetId};

/// Errors surfaced by [`simplify`] when it detects an ill-formed estimand instead of
/// silently rewriting it.
///
/// `eval_sum_out` / `eval_integral_out` (`crate::eval`) evaluate `SumOut` /
/// `IntegralOut` as a **literal, unnormalized** sum/integral over
/// `support(variables)`. A well-formed estimand always folds a `P(v|·)` factor into
/// the body for each bound variable `v`, so the body's free variables always
/// intersect `variables`. If they don't, the node is malformed: collapsing it to
/// its body (the old behavior) would silently divide the true value by
/// `|support(v)|` (`SumOut`) or drop the integration measure entirely
/// (`IntegralOut`). Rather than guess, `simplify` fails closed and reports it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimplifyError {
    /// A `SumOut` binds variable(s) that are absent from the free variables of its
    /// body.
    DeadSumOut {
        /// The bound variables, none of which occur free in the summed body.
        variables: Vec<VariableId>,
    },
    /// An `IntegralOut` binds variable(s) that are absent from the free variables of
    /// its body.
    DeadIntegralOut {
        /// The bound variables, none of which occur free in the integrated body.
        variables: Vec<VariableId>,
    },
}

impl fmt::Display for SimplifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeadSumOut { variables } => {
                write!(f, "SumOut binds variable(s) not free in its body: ")?;
                write_var_list(f, variables)
            }
            Self::DeadIntegralOut { variables } => {
                write!(f, "IntegralOut binds variable(s) not free in its body: ")?;
                write_var_list(f, variables)
            }
        }
    }
}

impl std::error::Error for SimplifyError {}

fn write_var_list(f: &mut fmt::Formatter<'_>, variables: &[VariableId]) -> fmt::Result {
    for (i, v) in variables.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "V{}", v.raw())?;
    }
    Ok(())
}

/// Simplify `root` bottom-up with memoization; returns a (possibly new) `ExprId`.
///
/// # Errors
///
/// [`SimplifyError`] if a `SumOut`/`IntegralOut` binds a variable absent from its
/// body's free variables (an ill-formed estimand; see [`SimplifyError`] docs).
pub(crate) fn simplify(arena: &mut CausalExprArena, root: ExprId) -> Result<ExprId, SimplifyError> {
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let mut free_memo: HashMap<ExprId, VarSetId> = HashMap::new();
    simplify_rec(arena, root, &mut memo, &mut free_memo)
}

fn simplify_rec(
    arena: &mut CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, ExprId>,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> Result<ExprId, SimplifyError> {
    if let Some(&cached) = memo.get(&id) {
        return Ok(cached);
    }
    let rebuilt = rebuild_children(arena, id, memo, free_memo)?;
    let simplified = apply_rules_fixpoint(arena, rebuilt, free_memo)?;
    memo.insert(id, simplified);
    Ok(simplified)
}

fn rebuild_children(
    arena: &mut CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, ExprId>,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> Result<ExprId, SimplifyError> {
    let node = arena.node(id).clone();
    let rebuilt = match node {
        ExprNode::Distribution { .. } => id,
        ExprNode::Product(list) => {
            let children_ids: Vec<ExprId> = arena.list(list).to_vec();
            let mut children: Vec<ExprId> = Vec::with_capacity(children_ids.len());
            for c in children_ids {
                children.push(simplify_rec(arena, c, memo, free_memo)?);
            }
            let list_id = arena.intern_list(children);
            arena.intern(ExprNode::Product(list_id))
        }
        ExprNode::SumOut { variables, expr } => {
            let body = simplify_rec(arena, expr, memo, free_memo)?;
            arena.intern(ExprNode::SumOut { variables, expr: body })
        }
        ExprNode::IntegralOut { variables, expr } => {
            let body = simplify_rec(arena, expr, memo, free_memo)?;
            arena.intern(ExprNode::IntegralOut { variables, expr: body })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let num = simplify_rec(arena, numerator, memo, free_memo)?;
            let den = simplify_rec(arena, denominator, memo, free_memo)?;
            arena.intern(ExprNode::Ratio { numerator: num, denominator: den })
        }
        ExprNode::Expectation { function, distribution } => {
            let dist = simplify_rec(arena, distribution, memo, free_memo)?;
            arena.intern(ExprNode::Expectation { function, distribution: dist })
        }
        ExprNode::Contrast { left, right, op } => {
            let l = simplify_rec(arena, left, memo, free_memo)?;
            let r = simplify_rec(arena, right, memo, free_memo)?;
            arena.intern(ExprNode::Contrast { left: l, right: r, op })
        }
    };
    Ok(rebuilt)
}

fn apply_rules_fixpoint(
    arena: &mut CausalExprArena,
    mut id: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> Result<ExprId, SimplifyError> {
    // Local rules only; children are already simplified.
    loop {
        let next = apply_local_rules(arena, id, free_memo)?;
        if next == id {
            return Ok(id);
        }
        id = next;
    }
}

fn apply_local_rules(
    arena: &mut CausalExprArena,
    id: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> Result<ExprId, SimplifyError> {
    match arena.node(id).clone() {
        ExprNode::SumOut { variables, expr } => {
            rewrite_sum_out(arena, id, variables, expr, free_memo)
        }
        ExprNode::IntegralOut { variables, expr } => {
            rewrite_integral_out(arena, id, variables, expr, free_memo)
        }
        ExprNode::Product(list) => Ok(rewrite_product(arena, id, list)),
        ExprNode::Ratio { numerator, denominator } => {
            Ok(rewrite_ratio(arena, id, numerator, denominator))
        }
        _ => Ok(id),
    }
}

fn rewrite_sum_out(
    arena: &mut CausalExprArena,
    id: ExprId,
    variables: VarSetId,
    expr: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> Result<ExprId, SimplifyError> {
    if arena.var_set(variables).is_empty() {
        return Ok(tag_if_new(arena, expr, id, "simplify.empty_sum_out"));
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
        return Ok(intern_derived(arena, node, "simplify.merge_sum_out"));
    }
    let free = free_vars(arena, expr, free_memo);
    if !intersects(arena, variables, free) {
        // Ill-formed estimand (see `SimplifyError` docs) — fail closed rather than
        // silently eliminating the sum (which would drop the `|support(v)|` factor).
        return Err(SimplifyError::DeadSumOut { variables: arena.var_set(variables).to_vec() });
    }
    Ok(id)
}

fn rewrite_integral_out(
    arena: &mut CausalExprArena,
    id: ExprId,
    variables: VarSetId,
    expr: ExprId,
    free_memo: &mut HashMap<ExprId, VarSetId>,
) -> Result<ExprId, SimplifyError> {
    if arena.var_set(variables).is_empty() {
        return Ok(tag_if_new(arena, expr, id, "simplify.empty_integral_out"));
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
        return Ok(intern_derived(arena, node, "simplify.merge_integral_out"));
    }
    let free = free_vars(arena, expr, free_memo);
    if !intersects(arena, variables, free) {
        // Ill-formed estimand (see `SimplifyError` docs) — fail closed rather than
        // silently collapsing the integral (which would drop the integration measure).
        return Err(SimplifyError::DeadIntegralOut {
            variables: arena.var_set(variables).to_vec(),
        });
    }
    Ok(id)
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
            // `conditioned_on` variables bound by the accompanying `intervention` set
            // are do(·)-fixed, not free — mirrors `eval::free_vars_rec`'s
            // `Distribution` arm (`eval.rs`), which this function must agree with:
            // both feed the "does the body depend on the summed variable" check in
            // `rewrite_sum_out`/`rewrite_integral_out` above, and a discrepancy there
            // was previously masking ill-formed estimands (B3). This can only shrink
            // the free-variable set relative to the old (unconditionally-inclusive)
            // version, which can only make `intersects()` return true *less* often —
            // i.e. it can only turn a previously-missed dead-sum/integral into a
            // now-detected `SimplifyError`, never turn a legitimate dependency into a
            // spurious elimination. It cannot newly enable an unsound rewrite.
            let bound: Vec<VariableId> =
                arena.intervention_assignments(intervention).iter().map(|a| a.variable).collect();
            for &v in arena.var_set(conditioned_on) {
                if !bound.iter().any(|b| *b == v) {
                    vars.push(v);
                }
            }
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
    use antecedent_core::Value;

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
        assert_eq!(simplify(&mut a, summed).unwrap(), dist);
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
        let s = simplify(&mut a, outer).unwrap();
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
    fn dead_sum_out_rejected() {
        // SumOut{z} over a body whose free variables are disjoint from {z} is an
        // ill-formed estimand (see `SimplifyError` docs): `eval_sum_out` evaluates it
        // as a literal `Σ_{z ∈ support(z)} dist`, so silently eliminating the SumOut
        // (the old, buggy behavior) would drop the `|support(z)|` multiplier and
        // divide the true value by it. `simplify` must reject it instead.
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
        let err = simplify(&mut a, summed).unwrap_err();
        assert_eq!(err, SimplifyError::DeadSumOut { variables: vec![VariableId::from_raw(1)] });
    }

    #[test]
    fn dead_integral_out_rejected() {
        // IntegralOut analogue of `dead_sum_out_rejected`: collapsing IntegralOut{z}
        // to its z-independent body would drop the integration measure over z
        // entirely, which is worse than the SumOut case's scaling error. Must reject.
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
        let integrated = a.intern(ExprNode::IntegralOut { variables: z, expr: dist });
        let err = simplify(&mut a, integrated).unwrap_err();
        assert_eq!(
            err,
            SimplifyError::DeadIntegralOut { variables: vec![VariableId::from_raw(1)] }
        );
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
        assert_eq!(simplify(&mut a, inner).unwrap(), d1);

        let nest = {
            let list_inner = a.intern_list([d1, d2]);
            let p_inner = a.intern(ExprNode::Product(list_inner));
            let list_outer = a.intern_list([p_inner, d1]);
            a.intern(ExprNode::Product(list_outer))
        };
        let s = simplify(&mut a, nest).unwrap();
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
        assert_eq!(simplify(&mut a, p1).unwrap(), simplify(&mut a, p2).unwrap());
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
        let s1 = simplify(&mut a, id).unwrap();
        let s2 = simplify(&mut a, s1).unwrap();
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
        let s = simplify(&mut a, nested).unwrap();
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
        let s = simplify(&mut a, contrast).unwrap();
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

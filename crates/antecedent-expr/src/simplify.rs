//! Algebraic simplification via worklist + memoization.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::{CausalExprArena, DerivationMeta, ExprId, ExprNode, VarSetId};

/// Errors surfaced by the crate's simplify entry points when they detect an ill-formed estimand instead of
/// silently rewriting it.
///
/// `eval_sum_out` / `eval_integral_out` (`crate::eval`) evaluate `SumOut` /
/// `IntegralOut` as a **literal, unnormalized** sum/integral over
/// `support(variables)`. A well-formed estimand always folds a `P(v|·)` factor into
/// the body for each bound variable `v`, so the body's free variables always
/// contain every variable in `variables`. If any bound variable is absent, the node
/// is malformed: that variable contributes a bare `|support(v)|` factor (`SumOut`)
/// or an unweighted integration measure (`IntegralOut`). Rather than guess,
/// `simplify` fails closed and reports the offending variables. The check runs
/// before nested binders are merged, so merging can never launder a dead binder.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SimplifyError {
    /// A `SumOut` binds variable(s) that are absent from the free variables of its
    /// body.
    DeadSumOut {
        /// The bound variables that do not occur free in the summed body.
        variables: Vec<VariableId>,
    },
    /// An `IntegralOut` binds variable(s) that are absent from the free variables of
    /// its body.
    DeadIntegralOut {
        /// The bound variables that do not occur free in the integrated body.
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

/// Free-variable lists (sorted, deduplicated) memoised per expression.
pub(crate) type FreeMemo = HashMap<ExprId, Arc<[VariableId]>>;

/// Simplify `root` bottom-up with memoization; returns a (possibly new) `ExprId`.
///
/// # Errors
///
/// [`SimplifyError`] if a `SumOut`/`IntegralOut` binds any variable absent from its
/// body's free variables (an ill-formed estimand; see [`SimplifyError`] docs).
pub(crate) fn simplify(arena: &mut CausalExprArena, root: ExprId) -> Result<ExprId, SimplifyError> {
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let mut free_memo: FreeMemo = HashMap::new();
    simplify_rec(arena, root, &mut memo, &mut free_memo)
}

fn simplify_rec(
    arena: &mut CausalExprArena,
    id: ExprId,
    memo: &mut HashMap<ExprId, ExprId>,
    free_memo: &mut FreeMemo,
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
    free_memo: &mut FreeMemo,
) -> Result<ExprId, SimplifyError> {
    let node = arena.node(id).clone();
    let rebuilt = match node {
        ExprNode::Distribution { .. } => id,
        ExprNode::Kernel { body, bound, population, regime } => {
            let body = simplify_rec(arena, body, memo, free_memo)?;
            arena.intern(ExprNode::Kernel { body, bound, population, regime })
        }
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
    free_memo: &mut FreeMemo,
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
    free_memo: &mut FreeMemo,
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
    free_memo: &mut FreeMemo,
) -> Result<ExprId, SimplifyError> {
    if arena.var_set(variables).is_empty() {
        return Ok(tag_if_new(arena, expr, id, "simplify.empty_sum_out"));
    }
    let free = free_vars(arena, expr, free_memo);
    let dead = difference(arena.var_set(variables), &free);
    if !dead.is_empty() {
        // Ill-formed estimand (see `SimplifyError` docs) — fail closed rather than
        // silently eliminating the sum (which would leave a bare `|support(v)|` factor).
        // Checked against the body's *free* variables, so a binder that the inner
        // `SumOut` already consumed is dead here too and cannot be merged away.
        return Err(SimplifyError::DeadSumOut { variables: dead });
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
    Ok(id)
}

fn rewrite_integral_out(
    arena: &mut CausalExprArena,
    id: ExprId,
    variables: VarSetId,
    expr: ExprId,
    free_memo: &mut FreeMemo,
) -> Result<ExprId, SimplifyError> {
    if arena.var_set(variables).is_empty() {
        return Ok(tag_if_new(arena, expr, id, "simplify.empty_integral_out"));
    }
    let free = free_vars(arena, expr, free_memo);
    let dead = difference(arena.var_set(variables), &free);
    if !dead.is_empty() {
        // Ill-formed estimand (see `SimplifyError` docs) — fail closed rather than
        // silently collapsing the integral (which would drop the integration measure).
        // Checked against the body's *free* variables, so a binder that the inner
        // `IntegralOut` already consumed is dead here too and cannot be merged away.
        return Err(SimplifyError::DeadIntegralOut { variables: dead });
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
    // Keep a/(b/c) nested: (a*c)/b would be defined at c=0 even
    // though the original expression has a zero denominator. Positivity is
    // not guaranteed by the expression arena or every provider.
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
        arena.set_derivation_if_absent(id, DerivationMeta::rule(rule, None));
    }
    id
}

/// Variables of `a` that are not in `b` (both sorted), in order.
fn difference(a: &[VariableId], b: &[VariableId]) -> Vec<VariableId> {
    a.iter().copied().filter(|v| b.binary_search(v).is_err()).collect()
}

/// Free variables of `id`, sorted and deduplicated. Reads the arena without interning, so it
/// needs only a shared borrow.
pub(crate) fn free_vars(
    arena: &CausalExprArena,
    id: ExprId,
    memo: &mut FreeMemo,
) -> Arc<[VariableId]> {
    if let Some(cached) = memo.get(&id) {
        return Arc::clone(cached);
    }
    let mut vars: Vec<VariableId> = match arena.node(id) {
        ExprNode::Distribution { variables, conditioned_on, intervention, .. } => {
            let mut vars: Vec<VariableId> = arena.var_set(*variables).to_vec();
            // `conditioned_on` variables bound by the accompanying `intervention` set
            // are do(·)-fixed, not free — mirrors `eval::compute_free_vars`'s
            // `Distribution` arm (`eval.rs`), which this function must agree with:
            // both feed the "does the body depend on the summed variable" check in
            // `rewrite_sum_out`/`rewrite_integral_out` above, and a discrepancy there
            // was previously masking ill-formed estimands. A symbolic coordinate is free.
            let assignments = arena.intervention_assignments(*intervention);
            for &v in arena.var_set(*conditioned_on) {
                if !assignments.iter().any(|a| a.variable == v && !a.is_symbolic()) {
                    vars.push(v);
                }
            }
            vars.extend(assignments.iter().filter(|a| a.is_symbolic()).map(|a| a.variable));
            vars
        }
        ExprNode::Kernel { body, .. } => free_vars(arena, *body, memo).to_vec(),
        ExprNode::Product(list) => {
            let mut vars = Vec::new();
            for &c in arena.list(*list) {
                vars.extend_from_slice(&free_vars(arena, c, memo));
            }
            vars
        }
        ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
            let bound = arena.var_set(*variables);
            free_vars(arena, *expr, memo)
                .iter()
                .copied()
                .filter(|v| bound.binary_search(v).is_err())
                .collect()
        }
        ExprNode::Ratio { numerator, denominator } => {
            let mut vars = free_vars(arena, *numerator, memo).to_vec();
            vars.extend_from_slice(&free_vars(arena, *denominator, memo));
            vars
        }
        ExprNode::Expectation { function, distribution } => free_vars(arena, *distribution, memo)
            .iter()
            .copied()
            .filter(|v| *v != function.variable())
            .collect(),
        ExprNode::Contrast { left, right, .. } => {
            let mut vars = free_vars(arena, *left, memo).to_vec();
            vars.extend_from_slice(&free_vars(arena, *right, memo));
            vars
        }
    };
    vars.sort_unstable();
    vars.dedup();
    let result: Arc<[VariableId]> = Arc::from(vars);
    memo.insert(id, Arc::clone(&result));
    result
}

/// A ratio `Σ_A k / Σ_B k` of two marginals of one joint `k` with `A ⊊ B`: the conditional
/// distribution of `B∖A` given the remaining variables of `k`.
///
/// This is the one structural shape an exact evaluation may extend across a null conditioning
/// event (`Σ_B k = 0` forces every summand, hence `Σ_A k`, to zero). It is owned here beside the
/// rewrites that produce and reshape `SumOut` nests, so a rewrite cannot silently move a
/// conditional out of the shape the evaluator recognises.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MarginalConditional {
    /// The joint both marginals are taken of.
    pub joint: ExprId,
    /// Variables summed out of the numerator (sorted; empty when the numerator is the joint).
    pub numerator_summed: Vec<VariableId>,
    /// Variables summed out of the denominator (sorted; a strict superset of the numerator's).
    pub denominator_summed: Vec<VariableId>,
}

/// Peel `Kernel` wrappers and nested `SumOut`/`IntegralOut` layers off `id`, returning the
/// innermost body and every variable summed on the way (sorted, deduplicated).
fn peel_marginalisation(arena: &CausalExprArena, mut id: ExprId) -> (ExprId, Vec<VariableId>) {
    let mut summed: Vec<VariableId> = Vec::new();
    loop {
        match arena.node(id) {
            ExprNode::Kernel { body, .. } => id = *body,
            ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
                summed.extend_from_slice(arena.var_set(*variables));
                id = *expr;
            }
            _ => break,
        }
    }
    summed.sort_unstable();
    summed.dedup();
    (id, summed)
}

/// Recognise `numerator / denominator` as [`MarginalConditional`].
pub(crate) fn marginal_conditional(
    arena: &CausalExprArena,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<MarginalConditional> {
    let (joint, numerator_summed) = peel_marginalisation(arena, numerator);
    let (denominator_joint, denominator_summed) = peel_marginalisation(arena, denominator);
    let strict_superset = denominator_summed.len() > numerator_summed.len()
        && numerator_summed.iter().all(|v| denominator_summed.binary_search(v).is_ok());
    (joint == denominator_joint && strict_superset).then_some(MarginalConditional {
        joint,
        numerator_summed,
        denominator_summed,
    })
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
        let dist = a.intern_distribution(empty, empty, empty_i, DomainRef::Observational);
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
        let dist = a.intern_distribution(vars12, empty, empty_i, DomainRef::Observational);
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
        let dist = a.intern_distribution(y, empty, empty_i, DomainRef::Observational);
        let summed = a.intern(ExprNode::SumOut { variables: z, expr: dist });
        let err = simplify(&mut a, summed).unwrap_err();
        assert_eq!(err, SimplifyError::DeadSumOut { variables: vec![VariableId::from_raw(1)] });
    }

    #[test]
    fn partially_dead_sum_out_reports_only_the_dead_binder() {
        // SumOut{w,z}(P(z)): z is live, w is dead and would multiply the result by |supp(w)|.
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let z = a.intern_var_set([VariableId::from_raw(1)]);
        let wz = a.intern_var_set([VariableId::from_raw(0), VariableId::from_raw(1)]);
        let dist = a.intern_distribution(z, empty, empty_i, DomainRef::Observational);
        let summed = a.intern(ExprNode::SumOut { variables: wz, expr: dist });
        let err = simplify(&mut a, summed).unwrap_err();
        assert_eq!(err, SimplifyError::DeadSumOut { variables: vec![VariableId::from_raw(0)] });
    }

    #[test]
    fn merge_cannot_launder_a_dead_outer_sum_out() {
        // SumOut{w}(SumOut{z}(P(z))): the outer sum is fully dead. Merging into
        // SumOut{w,z}(P(z)) would make it look partially live.
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let w = a.intern_var_set([VariableId::from_raw(0)]);
        let z = a.intern_var_set([VariableId::from_raw(1)]);
        let dist = a.intern_distribution(z, empty, empty_i, DomainRef::Observational);
        let inner = a.intern(ExprNode::SumOut { variables: z, expr: dist });
        let outer = a.intern(ExprNode::SumOut { variables: w, expr: inner });
        let err = simplify(&mut a, outer).unwrap_err();
        assert_eq!(err, SimplifyError::DeadSumOut { variables: vec![VariableId::from_raw(0)] });
    }

    #[test]
    fn partially_dead_and_laundered_integral_out_rejected() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let w = a.intern_var_set([VariableId::from_raw(0)]);
        let z = a.intern_var_set([VariableId::from_raw(1)]);
        let wz = a.intern_var_set([VariableId::from_raw(0), VariableId::from_raw(1)]);
        let dist = a.intern_distribution(z, empty, empty_i, DomainRef::Observational);
        let partial = a.intern(ExprNode::IntegralOut { variables: wz, expr: dist });
        assert_eq!(
            simplify(&mut a, partial).unwrap_err(),
            SimplifyError::DeadIntegralOut { variables: vec![VariableId::from_raw(0)] }
        );
        let inner = a.intern(ExprNode::IntegralOut { variables: z, expr: dist });
        let outer = a.intern(ExprNode::IntegralOut { variables: w, expr: inner });
        assert_eq!(
            simplify(&mut a, outer).unwrap_err(),
            SimplifyError::DeadIntegralOut { variables: vec![VariableId::from_raw(0)] }
        );
    }

    #[test]
    fn marginal_conditional_recognises_nested_and_merged_shapes() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let none = a.empty_intervention_set();
        let v = VariableId::from_raw;
        let all = a.intern_var_set([v(0), v(1), v(2)]);
        let joint = a.intern_distribution(all, empty, none, DomainRef::Observational);
        let s2 = a.intern_var_set([v(2)]);
        let s12 = a.intern_var_set([v(1), v(2)]);
        let num = a.intern(ExprNode::SumOut { variables: s2, expr: joint });
        let den = a.intern(ExprNode::SumOut { variables: s12, expr: joint });
        // Σ_{v2} k / Σ_{v1,v2} k = P(v1 | v0).
        let found = marginal_conditional(&a, num, den).unwrap();
        assert_eq!(found.joint, joint);
        assert_eq!(found.numerator_summed, vec![v(2)]);
        assert_eq!(found.denominator_summed, vec![v(1), v(2)]);
        // k / Σ_{v1,v2} k has an empty numerator binder.
        let found = marginal_conditional(&a, joint, den).unwrap();
        assert!(found.numerator_summed.is_empty());
        // Nested binders (as built before merging) and kernel wrappers are peeled.
        let inner = a.intern(ExprNode::SumOut { variables: s2, expr: joint });
        let s1 = a.intern_var_set([v(1)]);
        let nested = a.intern(ExprNode::SumOut { variables: s1, expr: inner });
        assert_eq!(
            marginal_conditional(&a, num, nested).unwrap().denominator_summed,
            vec![v(1), v(2)]
        );
        // Same binders on both sides, the reverse containment, or different joints do not match.
        assert!(marginal_conditional(&a, den, den).is_none());
        assert!(marginal_conditional(&a, den, num).is_none());
        let other = a.intern_distribution(all, empty, none, DomainRef::Interventional);
        let other_den = a.intern(ExprNode::SumOut { variables: s12, expr: other });
        assert!(marginal_conditional(&a, num, other_den).is_none());
        // The merged form the simplifier produces still matches.
        let simplified = simplify(&mut a, nested).unwrap();
        assert_eq!(marginal_conditional(&a, num, simplified).unwrap().joint, joint);
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
        let dist = a.intern_distribution(y, empty, empty_i, DomainRef::Observational);
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
        let d1 = a.intern_distribution(v0, empty, empty_i, DomainRef::Observational);
        let d2 = a.intern_distribution(v1, empty, empty_i, DomainRef::Observational);
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
        let d1 = a.intern_distribution(v0, empty, empty_i, DomainRef::Observational);
        let d2 = a.intern_distribution(v1, empty, empty_i, DomainRef::Observational);
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
        let da = a.intern_distribution(v0, empty, empty_i, DomainRef::Observational);
        let db = a.intern_distribution(v1, empty, empty_i, DomainRef::Observational);
        let dc = a.intern_distribution(v2, empty, empty_i, DomainRef::Observational);
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
        let dist = a.intern_distribution(empty, empty, empty_i, DomainRef::Observational);
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

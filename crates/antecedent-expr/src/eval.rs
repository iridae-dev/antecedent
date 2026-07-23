//! Compiled topological evaluators for causal expressions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::VariableId;

use crate::provider::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use crate::{
    CausalExprArena, ContrastOp, DomainRef, ExprId, ExprNode, InterventionSetId, OutcomeExprId,
    VarSetId,
};

/// One step in a compiled evaluation plan (child references are slot indices).
#[derive(Clone, Debug)]
enum EvalOp {
    Distribution {
        variables: VarSetId,
        conditioned_on: VarSetId,
        intervention: InterventionSetId,
        domain: DomainRef,
    },
    Product {
        children: Arc<[usize]>,
    },
    SumOut {
        variables: VarSetId,
        body: usize,
    },
    IntegralOut {
        variables: VarSetId,
        body: usize,
    },
    Ratio {
        numerator: usize,
        denominator: usize,
    },
    Expectation {
        function: OutcomeExprId,
        distribution: usize,
    },
    Contrast {
        left: usize,
        right: usize,
        op: ContrastOp,
    },
}

/// Topologically ordered compiled evaluator for repeated provider evaluation.
#[derive(Clone, Debug)]
pub struct CompiledEvaluator {
    ops: Vec<EvalOp>,
    root: usize,
}

impl CausalExprArena {
    /// Compile `root` into a topological evaluation plan.
    ///
    /// Continuous [`ExprNode::IntegralOut`] compiles successfully; evaluation uses
    /// [`DistributionProvider::quadrature`] or discrete [`DistributionProvider::support`].
    pub fn compile(&self, root: ExprId) -> Result<CompiledEvaluator, EvalError> {
        CompiledEvaluator::compile(self, root)
    }
}

impl CompiledEvaluator {
    /// Compile an expression DAG into slot-addressed ops (post-order).
    ///
    /// Continuous [`ExprNode::IntegralOut`] is supported (see [`CausalExprArena::compile`]).
    pub fn compile(arena: &CausalExprArena, root: ExprId) -> Result<Self, EvalError> {
        let mut ops = Vec::new();
        let mut expr_to_slot = HashMap::new();
        let root_slot = compile_rec(arena, root, &mut ops, &mut expr_to_slot)?;
        Ok(Self { ops, root: root_slot })
    }

    /// Evaluate once against a provider.
    ///
    /// # Errors
    ///
    /// Provider / numeric failures.
    pub fn evaluate(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.evaluate_with(arena, provider, ctx, &Assignment::new())
    }

    /// Evaluate with an initial variable binding (e.g. `do(X=x)` and outcome levels).
    ///
    /// # Errors
    ///
    /// Provider / numeric failures.
    pub fn evaluate_with(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &Assignment,
    ) -> Result<f64, EvalError> {
        self.eval_slot(arena, provider, ctx, env, self.root)
    }

    /// Evaluate over all posterior draws (`provider.n_draws()`), or a single
    /// empirical evaluation when `n_draws` is `None`.
    ///
    /// # Errors
    ///
    /// Provider / numeric failures.
    pub fn evaluate_batch(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
    ) -> Result<Vec<f64>, EvalError> {
        match provider.n_draws() {
            None => Ok(vec![self.evaluate(arena, provider, &EvalContext::default())?]),
            Some(n) => {
                let mut out = Vec::with_capacity(n);
                for draw in 0..n {
                    let ctx = EvalContext { draw: Some(draw) };
                    out.push(self.evaluate(arena, provider, &ctx)?);
                }
                Ok(out)
            }
        }
    }

    fn eval_slot(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &Assignment,
        slot: usize,
    ) -> Result<f64, EvalError> {
        // Density / scalar under `env`. Expectations and contrasts are scalars;
        // other ops are densities in the free variables bound by `env`.
        match &self.ops[slot] {
            EvalOp::Distribution { variables, conditioned_on, intervention, domain } => {
                let spec = FactorSpec {
                    variables: arena.var_set(*variables),
                    conditioned_on: arena.var_set(*conditioned_on),
                    intervention: arena.intervention_assignments(*intervention),
                    domain: *domain,
                };
                // Interventions bind targets; merge into lookup assignment.
                let mut lookup = env.clone();
                for a in spec.intervention {
                    lookup.set(a.variable, a.value.clone());
                }
                provider.probability(&spec, &lookup, ctx)
            }
            EvalOp::Product { children } => {
                let mut prod = 1.0;
                for &c in children.iter() {
                    prod *= self.eval_slot(arena, provider, ctx, env, c)?;
                }
                Ok(prod)
            }
            EvalOp::SumOut { variables, body } => {
                self.eval_sum_out(arena, provider, ctx, env, *variables, *body)
            }
            EvalOp::IntegralOut { variables, body } => {
                self.eval_integral_out(arena, provider, ctx, env, *variables, *body)
            }
            EvalOp::Ratio { numerator, denominator } => {
                let num = self.eval_slot(arena, provider, ctx, env, *numerator)?;
                let den = self.eval_slot(arena, provider, ctx, env, *denominator)?;
                if den == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(num / den)
            }
            EvalOp::Expectation { function, distribution } => {
                self.eval_expectation(arena, provider, ctx, env, function.variable(), *distribution)
            }
            EvalOp::Contrast { left, right, op } => {
                let l = self.eval_slot(arena, provider, ctx, env, *left)?;
                let r = self.eval_slot(arena, provider, ctx, env, *right)?;
                match op {
                    ContrastOp::Difference => Ok(l - r),
                }
            }
        }
    }

    fn eval_sum_out(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &Assignment,
        variables: VarSetId,
        body: usize,
    ) -> Result<f64, EvalError> {
        let vars = arena.var_set(variables);
        let rows = provider.support(vars, ctx)?;
        let mut sum = 0.0;
        for row in rows.iter() {
            if row.len() != vars.len() {
                return Err(EvalError::SupportShape { expected: vars.len(), actual: row.len() });
            }
            let mut extended = env.clone();
            for (i, &v) in vars.iter().enumerate() {
                extended.set(v, row[i].clone());
            }
            sum += self.eval_slot(arena, provider, ctx, &extended, body)?;
        }
        Ok(sum)
    }

    fn eval_integral_out(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &Assignment,
        variables: VarSetId,
        body: usize,
    ) -> Result<f64, EvalError> {
        let vars = arena.var_set(variables);
        if let Some(nodes) = provider.quadrature(vars, ctx)? {
            let mut acc = 0.0;
            for (row, weight) in nodes.iter() {
                if row.len() != vars.len() {
                    return Err(EvalError::SupportShape {
                        expected: vars.len(),
                        actual: row.len(),
                    });
                }
                let mut extended = env.clone();
                for (i, &v) in vars.iter().enumerate() {
                    extended.set(v, row[i].clone());
                }
                acc += *weight * self.eval_slot(arena, provider, ctx, &extended, body)?;
            }
            return Ok(acc);
        }
        // Discrete / counting-measure fallback (IntegralOut ≡ SumOut).
        let rows = provider.support(vars, ctx).map_err(|e| match e {
            EvalError::EmptySupport(_) => EvalError::UnsupportedIntegralOut,
            other => other,
        })?;
        let mut sum = 0.0;
        for row in rows.iter() {
            if row.len() != vars.len() {
                return Err(EvalError::SupportShape { expected: vars.len(), actual: row.len() });
            }
            let mut extended = env.clone();
            for (i, &v) in vars.iter().enumerate() {
                extended.set(v, row[i].clone());
            }
            sum += self.eval_slot(arena, provider, ctx, &extended, body)?;
        }
        Ok(sum)
    }

    fn eval_expectation(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &Assignment,
        outcome_var: VariableId,
        distribution: usize,
    ) -> Result<f64, EvalError> {
        // E[f | D] = Σ_{x ∈ support(free(D))} f(x) · dens(D, x)
        let free = free_vars_of_slot(self, arena, distribution);
        let unbound: Vec<VariableId> = free.into_iter().filter(|v| env.get(*v).is_none()).collect();
        let mut enum_vars = unbound;
        if !enum_vars.contains(&outcome_var) && env.get(outcome_var).is_none() {
            enum_vars.push(outcome_var);
        }
        enum_vars.sort_by_key(|v| v.raw());
        enum_vars.dedup();

        if enum_vars.is_empty() {
            let dens = self.eval_slot(arena, provider, ctx, env, distribution)?;
            let y = provider.outcome(outcome_var, env, ctx)?;
            return Ok(y * dens);
        }

        let rows = provider.support(&enum_vars, ctx)?;
        let mut acc = 0.0;
        for row in rows.iter() {
            if row.len() != enum_vars.len() {
                return Err(EvalError::SupportShape {
                    expected: enum_vars.len(),
                    actual: row.len(),
                });
            }
            let mut extended = env.clone();
            for (i, &v) in enum_vars.iter().enumerate() {
                extended.set(v, row[i].clone());
            }
            let dens = self.eval_slot(arena, provider, ctx, &extended, distribution)?;
            let y = provider.outcome(outcome_var, &extended, ctx)?;
            acc += y * dens;
        }
        Ok(acc)
    }
}

fn compile_rec(
    arena: &CausalExprArena,
    id: ExprId,
    ops: &mut Vec<EvalOp>,
    expr_to_slot: &mut HashMap<u32, usize>,
) -> Result<usize, EvalError> {
    if let Some(&slot) = expr_to_slot.get(&id.raw()) {
        return Ok(slot);
    }
    let op = match arena.node(id).clone() {
        ExprNode::Distribution { variables, conditioned_on, intervention, domain } => {
            EvalOp::Distribution { variables, conditioned_on, intervention, domain }
        }
        ExprNode::Product(list) => {
            let mut children = Vec::new();
            for &c in arena.list(list) {
                children.push(compile_rec(arena, c, ops, expr_to_slot)?);
            }
            EvalOp::Product { children: Arc::from(children) }
        }
        ExprNode::SumOut { variables, expr } => {
            let body = compile_rec(arena, expr, ops, expr_to_slot)?;
            EvalOp::SumOut { variables, body }
        }
        ExprNode::IntegralOut { variables, expr } => {
            let body = compile_rec(arena, expr, ops, expr_to_slot)?;
            EvalOp::IntegralOut { variables, body }
        }
        ExprNode::Ratio { numerator, denominator } => {
            let n = compile_rec(arena, numerator, ops, expr_to_slot)?;
            let d = compile_rec(arena, denominator, ops, expr_to_slot)?;
            EvalOp::Ratio { numerator: n, denominator: d }
        }
        ExprNode::Expectation { function, distribution } => {
            let dist = compile_rec(arena, distribution, ops, expr_to_slot)?;
            EvalOp::Expectation { function, distribution: dist }
        }
        ExprNode::Contrast { left, right, op } => {
            let l = compile_rec(arena, left, ops, expr_to_slot)?;
            let r = compile_rec(arena, right, ops, expr_to_slot)?;
            EvalOp::Contrast { left: l, right: r, op }
        }
    };
    let slot = ops.len();
    ops.push(op);
    expr_to_slot.insert(id.raw(), slot);
    Ok(slot)
}

fn free_vars_of_slot(
    compiled: &CompiledEvaluator,
    arena: &CausalExprArena,
    slot: usize,
) -> Vec<VariableId> {
    let mut out = Vec::new();
    free_vars_rec(compiled, arena, slot, &mut out);
    out.sort_by_key(|v| v.raw());
    out.dedup();
    out
}

fn free_vars_rec(
    compiled: &CompiledEvaluator,
    arena: &CausalExprArena,
    slot: usize,
    out: &mut Vec<VariableId>,
) {
    match &compiled.ops[slot] {
        EvalOp::Distribution { variables, conditioned_on, intervention, .. } => {
            out.extend_from_slice(arena.var_set(*variables));
            let bound: Vec<VariableId> =
                arena.intervention_assignments(*intervention).iter().map(|a| a.variable).collect();
            for &v in arena.var_set(*conditioned_on) {
                if !bound.iter().any(|b| *b == v) {
                    out.push(v);
                }
            }
        }
        EvalOp::Product { children } => {
            for &c in children.iter() {
                free_vars_rec(compiled, arena, c, out);
            }
        }
        EvalOp::SumOut { variables, body } | EvalOp::IntegralOut { variables, body } => {
            let mut inner = Vec::new();
            free_vars_rec(compiled, arena, *body, &mut inner);
            let bound = arena.var_set(*variables);
            for v in inner {
                if !bound.iter().any(|b| *b == v) {
                    out.push(v);
                }
            }
        }
        EvalOp::Ratio { numerator, denominator } => {
            free_vars_rec(compiled, arena, *numerator, out);
            free_vars_rec(compiled, arena, *denominator, out);
        }
        EvalOp::Expectation { function, distribution } => {
            free_vars_rec(compiled, arena, *distribution, out);
            out.push(function.variable());
        }
        EvalOp::Contrast { left, right, .. } => {
            free_vars_rec(compiled, arena, *left, out);
            free_vars_rec(compiled, arena, *right, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{EmpiricalTableProvider, PosteriorDrawProvider};
    use crate::{InterventionAssignment, OutcomeExprId};
    use antecedent_core::Value;

    fn v(id: u32) -> VariableId {
        VariableId::from_raw(id)
    }

    fn f(x: f64) -> Value {
        Value::f64(x)
    }

    /// Binary confounder Z, binary Y; backdoor ATE = 0.45.
    fn backdoor_provider(t: VariableId, y: VariableId, z: VariableId) -> EmpiricalTableProvider {
        let mut p = EmpiricalTableProvider::new();
        p.set_domain(z, [f(0.0), f(1.0)]);
        p.set_domain(y, [f(0.0), f(1.0)]);
        p.set_domain(t, [f(0.0), f(1.0)]);

        // P(Z)
        for (zval, prob) in [(0.0, 0.5), (1.0, 0.5)] {
            let spec = FactorSpec {
                variables: &[z],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
            };
            let assign = Assignment::from_pairs([(z, f(zval))]);
            p.insert_probability(&spec, &assign, prob).unwrap();
        }

        // P(Y | Z, do(T=t)) = P(Y | T=t, Z) under backdoor.
        // E[Y|T=1,Z=0]=0.8, E[Y|T=1,Z=1]=0.6, E[Y|T=0,Z=0]=0.3, E[Y|T=0,Z=1]=0.2
        let ey = |tlev: f64, zlev: f64| -> f64 {
            match (tlev.to_bits(), zlev.to_bits()) {
                (t, z) if t == 1.0f64.to_bits() && z == 0.0f64.to_bits() => 0.8,
                (t, z) if t == 1.0f64.to_bits() && z == 1.0f64.to_bits() => 0.6,
                (t, z) if t == 0.0f64.to_bits() && z == 0.0f64.to_bits() => 0.3,
                (t, z) if t == 0.0f64.to_bits() && z == 1.0f64.to_bits() => 0.2,
                _ => panic!("bad levels"),
            }
        };
        for tlev in [0.0, 1.0] {
            let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
            for zlev in [0.0, 1.0] {
                let p_y1 = ey(tlev, zlev);
                for (yval, prob) in [(1.0, p_y1), (0.0, 1.0 - p_y1)] {
                    let spec = FactorSpec {
                        variables: &[y],
                        conditioned_on: &[z],
                        intervention: &interv,
                        domain: DomainRef::Interventional,
                    };
                    let assign = Assignment::from_pairs([(y, f(yval)), (z, f(zlev))]);
                    p.insert_probability(&spec, &assign, prob).unwrap();
                }
            }
        }
        p
    }

    #[test]
    fn backdoor_ate_matches_closed_form() {
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let z = v(2);
        let expr = arena.backdoor_ate(t, y, &[z], f(1.0), f(0.0));
        let provider = backdoor_provider(t, y, z);
        let compiled = arena.compile(expr).unwrap();
        let ate = compiled.evaluate(&arena, &provider, &EvalContext::default()).unwrap();
        assert!((ate - 0.45).abs() < 1e-12, "ate={ate}");
    }

    #[test]
    fn simplify_preserves_backdoor_evaluation() {
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let z = v(2);
        let expr = arena.backdoor_ate(t, y, &[z], f(1.0), f(0.0));
        let provider = backdoor_provider(t, y, z);
        let before = arena
            .compile(expr)
            .unwrap()
            .evaluate(&arena, &provider, &EvalContext::default())
            .unwrap();
        let simplified = arena.simplify(expr);
        let after = arena
            .compile(simplified)
            .unwrap()
            .evaluate(&arena, &provider, &EvalContext::default())
            .unwrap();
        assert!((before - after).abs() < 1e-12, "before={before} after={after}");
        assert!((after - 0.45).abs() < 1e-12);
    }

    /// Empty adjustment (second Z set): simplify must preserve numeric eval.
    #[test]
    fn simplify_preserves_backdoor_empty_evaluation() {
        fn assert_simplify_preserves(
            arena: &mut CausalExprArena,
            expr: ExprId,
            provider: &EmpiricalTableProvider,
            expected: f64,
            label: &str,
        ) {
            let before = arena
                .compile(expr)
                .unwrap()
                .evaluate(arena, provider, &EvalContext::default())
                .unwrap();
            let simplified = arena.simplify(expr);
            let after = arena
                .compile(simplified)
                .unwrap()
                .evaluate(arena, provider, &EvalContext::default())
                .unwrap();
            assert!((before - after).abs() < 1e-12, "{label}: before={before} after={after}");
            assert!((after - expected).abs() < 1e-12, "{label}: after={after}");
        }

        // Backdoor with empty Z: E[Y|do(1)]=0.7, E[Y|do(0)]=0.2 → ATE = 0.5.
        // Exercises simplify.empty_sum_out / singleton product on the adjustment set.
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let expr = arena.backdoor_ate(t, y, &[], f(1.0), f(0.0));
        let mut p = EmpiricalTableProvider::new();
        p.set_domain(y, [f(0.0), f(1.0)]);
        p.set_domain(t, [f(0.0), f(1.0)]);
        // Vacuous P(∅) factor from empty adjustment marginal.
        let empty_spec = FactorSpec {
            variables: &[],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
        };
        p.insert_probability(&empty_spec, &Assignment::from_pairs([]), 1.0).unwrap();
        for tlev in [0.0, 1.0] {
            let ey = if (tlev - 1.0_f64).abs() < f64::EPSILON { 0.7 } else { 0.2 };
            let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
            for (yval, prob) in [(1.0, ey), (0.0, 1.0 - ey)] {
                let spec = FactorSpec {
                    variables: &[y],
                    conditioned_on: &[],
                    intervention: &interv,
                    domain: DomainRef::Interventional,
                };
                p.insert_probability(&spec, &Assignment::from_pairs([(y, f(yval))]), prob).unwrap();
            }
        }
        assert_simplify_preserves(&mut arena, expr, &p, 0.5, "backdoor_empty_z");
    }

    /// Frontdoor: simplify must preserve numeric eval.
    #[test]
    fn simplify_preserves_frontdoor_evaluation() {
        fn assert_simplify_preserves(
            arena: &mut CausalExprArena,
            expr: ExprId,
            provider: &EmpiricalTableProvider,
            expected: f64,
            label: &str,
        ) {
            let before = arena
                .compile(expr)
                .unwrap()
                .evaluate(arena, provider, &EvalContext::default())
                .unwrap();
            let simplified = arena.simplify(expr);
            let after = arena
                .compile(simplified)
                .unwrap()
                .evaluate(arena, provider, &EvalContext::default())
                .unwrap();
            assert!((before - after).abs() < 1e-12, "{label}: before={before} after={after}");
            assert!((after - expected).abs() < 1e-12, "{label}: after={after}");
        }

        // Frontdoor (same tables as shallow_frontdoor_evaluates): ATE = 0.32.
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let m = v(2);
        let expr = arena.frontdoor_ate(t, y, &[m], f(1.0), f(0.0));
        let mut p = EmpiricalTableProvider::new();
        p.set_domain(t, [f(0.0), f(1.0)]);
        p.set_domain(y, [f(0.0), f(1.0)]);
        p.set_domain(m, [f(0.0), f(1.0)]);
        for (tval, prob) in [(0.0, 0.5), (1.0, 0.5)] {
            let spec = FactorSpec {
                variables: &[t],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
            };
            p.insert_probability(&spec, &Assignment::from_pairs([(t, f(tval))]), prob).unwrap();
        }
        for tlev in [0.0, 1.0] {
            let pm1 = if (tlev - 1.0_f64).abs() < f64::EPSILON { 0.7 } else { 0.3 };
            let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
            for (mval, prob) in [(1.0, pm1), (0.0, 1.0 - pm1)] {
                let spec = FactorSpec {
                    variables: &[m],
                    conditioned_on: &[t],
                    intervention: &interv,
                    domain: DomainRef::Observational,
                };
                p.insert_probability(
                    &spec,
                    &Assignment::from_pairs([(m, f(mval)), (t, f(tlev))]),
                    prob,
                )
                .unwrap();
            }
        }
        for tlev in [0.0, 1.0] {
            for mlev in [0.0, 1.0] {
                let py1 = if (mlev - 1.0_f64).abs() < f64::EPSILON { 0.9 } else { 0.1 };
                for (yval, prob) in [(1.0, py1), (0.0, 1.0 - py1)] {
                    let spec = FactorSpec {
                        variables: &[y],
                        conditioned_on: &[t, m],
                        intervention: &[],
                        domain: DomainRef::Observational,
                    };
                    let assign = Assignment::from_pairs([(y, f(yval)), (m, f(mlev)), (t, f(tlev))]);
                    p.insert_probability(&spec, &assign, prob).unwrap();
                }
            }
        }
        assert_simplify_preserves(&mut arena, expr, &p, 0.32, "frontdoor");
    }

    #[test]
    fn shallow_frontdoor_evaluates() {
        // Minimal front-door: T→M→Y with no hidden confounding encoded in tables.
        // P(M|T=t); P(Y|M,T'); P(T').
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let m = v(2);
        let expr = arena.frontdoor_ate(t, y, &[m], f(1.0), f(0.0));

        let mut p = EmpiricalTableProvider::new();
        p.set_domain(t, [f(0.0), f(1.0)]);
        p.set_domain(y, [f(0.0), f(1.0)]);
        p.set_domain(m, [f(0.0), f(1.0)]);

        // P(T')
        for (tval, prob) in [(0.0, 0.5), (1.0, 0.5)] {
            let spec = FactorSpec {
                variables: &[t],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
            };
            p.insert_probability(&spec, &Assignment::from_pairs([(t, f(tval))]), prob).unwrap();
        }

        // P(M | T=t): P(M=1|T=1)=0.7, P(M=1|T=0)=0.3 (FD condition 2).
        for tlev in [0.0, 1.0] {
            let pm1 = if (tlev - 1.0_f64).abs() < f64::EPSILON { 0.7 } else { 0.3 };
            let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
            for (mval, prob) in [(1.0, pm1), (0.0, 1.0 - pm1)] {
                let spec = FactorSpec {
                    variables: &[m],
                    conditioned_on: &[t],
                    intervention: &interv,
                    domain: DomainRef::Observational,
                };
                p.insert_probability(
                    &spec,
                    &Assignment::from_pairs([(m, f(mval)), (t, f(tlev))]),
                    prob,
                )
                .unwrap();
            }
        }

        // P(Y | M, T'): E[Y|M=1,*]=0.9, E[Y|M=0,*]=0.1 (T' irrelevant)
        // Arena sorts m_and_t as [t, m] when t.raw() < m.raw().
        for tlev in [0.0, 1.0] {
            for mlev in [0.0, 1.0] {
                let py1 = if (mlev - 1.0_f64).abs() < f64::EPSILON { 0.9 } else { 0.1 };
                for (yval, prob) in [(1.0, py1), (0.0, 1.0 - py1)] {
                    let spec = FactorSpec {
                        variables: &[y],
                        conditioned_on: &[t, m],
                        intervention: &[],
                        domain: DomainRef::Observational,
                    };
                    let assign = Assignment::from_pairs([(y, f(yval)), (m, f(mlev)), (t, f(tlev))]);
                    p.insert_probability(&spec, &assign, prob).unwrap();
                }
            }
        }

        // Front-door: E[Y|do(T=t)] = Σ_m P(m|t) Σ_t' P(y|m,t') P(t')
        // With P(Y|M) independent of T': E[Y|do(T=1)] = 0.7*0.9 + 0.3*0.1 = 0.66
        // E[Y|do(T=0)] = 0.3*0.9 + 0.7*0.1 = 0.34
        // ATE = 0.32
        let compiled = arena.compile(expr).unwrap();
        let ate = compiled.evaluate(&arena, &p, &EvalContext::default()).unwrap();
        assert!((ate - 0.32).abs() < 1e-12, "ate={ate}");

        let simplified = arena.simplify(expr);
        let ate2 = arena
            .compile(simplified)
            .unwrap()
            .evaluate(&arena, &p, &EvalContext::default())
            .unwrap();
        assert!((ate - ate2).abs() < 1e-12);
    }

    #[test]
    fn discrete_integral_out_matches_sum_out() {
        let mut arena = CausalExprArena::new();
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let z = v(0);
        let zset = arena.intern_var_set([z]);
        let dist = arena.intern(ExprNode::Distribution {
            variables: zset,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let sum = arena.intern(ExprNode::SumOut { variables: zset, expr: dist });
        let integ = arena.intern(ExprNode::IntegralOut { variables: zset, expr: dist });

        let mut p = EmpiricalTableProvider::new();
        p.set_domain(z, [f(0.0), f(1.0)]);
        for (zval, prob) in [(0.0, 0.3), (1.0, 0.7)] {
            let spec = FactorSpec {
                variables: &[z],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
            };
            p.insert_probability(&spec, &Assignment::from_pairs([(z, f(zval))]), prob).unwrap();
        }
        let s = arena.compile(sum).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        let i =
            arena.compile(integ).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        assert!((s - 1.0).abs() < 1e-12);
        assert!((i - s).abs() < 1e-12);
    }

    #[test]
    fn continuous_gaussian_integral_out_normalizes() {
        use crate::provider::GaussianDensityProvider;
        let mut arena = CausalExprArena::new();
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let x = v(0);
        let xset = arena.intern_var_set([x]);
        let dist = arena.intern(ExprNode::Distribution {
            variables: xset,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let integ = arena.intern(ExprNode::IntegralOut { variables: xset, expr: dist });
        let mut p = GaussianDensityProvider::new();
        p.set_gaussian(x, 0.0, 1.0);
        let mass =
            arena.compile(integ).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        assert!((mass - 1.0).abs() < 1e-6, "∫ φ = {mass}");
    }

    #[test]
    fn nested_integral_out_product_gaussian() {
        use crate::provider::GaussianDensityProvider;
        let mut arena = CausalExprArena::new();
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let x = v(0);
        let y = v(1);
        let xset = arena.intern_var_set([x]);
        let yset = arena.intern_var_set([y]);
        let both = arena.intern_var_set([x, y]);
        let dist = arena.intern(ExprNode::Distribution {
            variables: both,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let inner = arena.intern(ExprNode::IntegralOut { variables: yset, expr: dist });
        let outer = arena.intern(ExprNode::IntegralOut { variables: xset, expr: inner });
        let mut p = GaussianDensityProvider::new();
        p.set_gaussian(x, 1.0, 0.25);
        p.set_gaussian(y, -0.5, 4.0);
        let mass =
            arena.compile(outer).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        assert!((mass - 1.0).abs() < 1e-5, "∬ φ = {mass}");
    }

    #[test]
    fn posterior_evaluate_batch() {
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let z = v(2);
        let expr = arena.backdoor_ate(t, y, &[z], f(1.0), f(0.0));

        let draw0 = backdoor_provider(t, y, z);
        // Perturb P(Z) in draw1 so ATE still 0.45 if conditionals unchanged...
        // Actually change E[Y|T=1,Z=*] so ATE differs.
        let mut draw1 = EmpiricalTableProvider::new();
        draw1.set_domain(z, [f(0.0), f(1.0)]);
        draw1.set_domain(y, [f(0.0), f(1.0)]);
        draw1.set_domain(t, [f(0.0), f(1.0)]);
        for (zval, prob) in [(0.0, 0.5), (1.0, 0.5)] {
            let spec = FactorSpec {
                variables: &[z],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
            };
            draw1.insert_probability(&spec, &Assignment::from_pairs([(z, f(zval))]), prob).unwrap();
        }
        // E[Y|T=1,*]=1.0, E[Y|T=0,*]=0.0 → ATE = 1.0
        for tlev in [0.0, 1.0] {
            let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
            let py1 = tlev;
            for zlev in [0.0, 1.0] {
                for (yval, prob) in [(1.0, py1), (0.0, 1.0 - py1)] {
                    let spec = FactorSpec {
                        variables: &[y],
                        conditioned_on: &[z],
                        intervention: &interv,
                        domain: DomainRef::Interventional,
                    };
                    draw1
                        .insert_probability(
                            &spec,
                            &Assignment::from_pairs([(y, f(yval)), (z, f(zlev))]),
                            prob,
                        )
                        .unwrap();
                }
            }
        }

        let posterior = PosteriorDrawProvider::from_draws(vec![draw0, draw1]);
        let compiled = arena.compile(expr).unwrap();
        let batch = compiled.evaluate_batch(&arena, &posterior).unwrap();
        assert_eq!(batch.len(), 2);
        assert!((batch[0] - 0.45).abs() < 1e-12, "draw0={}", batch[0]);
        assert!((batch[1] - 1.0).abs() < 1e-12, "draw1={}", batch[1]);

        let single0 =
            compiled.evaluate(&arena, &posterior, &EvalContext { draw: Some(0) }).unwrap();
        let single1 =
            compiled.evaluate(&arena, &posterior, &EvalContext { draw: Some(1) }).unwrap();
        assert!((single0 - batch[0]).abs() < 1e-15);
        assert!((single1 - batch[1]).abs() < 1e-15);
    }

    #[test]
    fn expectation_of_simple_marginal() {
        let mut arena = CausalExprArena::new();
        let y = v(0);
        let yset = arena.intern_var_set([y]);
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let dist = arena.intern(ExprNode::Distribution {
            variables: yset,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let exp = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(y),
            distribution: dist,
        });

        let mut p = EmpiricalTableProvider::new();
        p.set_domain(y, [f(0.0), f(2.0)]);
        let spec = FactorSpec {
            variables: &[y],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
        };
        p.insert_probability(&spec, &Assignment::from_pairs([(y, f(0.0))]), 0.25).unwrap();
        p.insert_probability(&spec, &Assignment::from_pairs([(y, f(2.0))]), 0.75).unwrap();

        let val =
            arena.compile(exp).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        // 0*0.25 + 2*0.75 = 1.5
        assert!((val - 1.5).abs() < 1e-12);
    }
}

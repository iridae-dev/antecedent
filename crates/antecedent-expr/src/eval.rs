//! Compiled topological evaluators for causal expressions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{Value, VariableId};

use crate::provider::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use crate::{
    CausalExprArena, ContrastOp, DomainRef, ExprId, ExprNode, InterventionSetId, OutcomeExprId,
    PopulationKeyId, VarSetId,
};
use antecedent_core::RegimeId;

/// One step in a compiled evaluation plan (child references are slot indices).
#[derive(Clone, Debug)]
pub(crate) enum EvalOp {
    Distribution {
        variables: VarSetId,
        conditioned_on: VarSetId,
        intervention: InterventionSetId,
        domain: DomainRef,
        population: PopulationKeyId,
        regime: Option<RegimeId>,
    },
    Kernel {
        body: usize,
        #[allow(dead_code)]
        population: PopulationKeyId,
        #[allow(dead_code)]
        regime: Option<RegimeId>,
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
    pub(crate) ops: Vec<EvalOp>,
    pub(crate) origins: Vec<ExprId>,
    /// Sorted, deduplicated free variables per slot. A static property of the
    /// plan, computed once at compile time; `Expectation` evaluation reads it
    /// on every call instead of re-deriving it per evaluation.
    pub(crate) free_vars: Vec<Arc<[VariableId]>>,
    /// Sorted, deduplicated variables each slot is a density in; an `Expectation`
    /// integrates over these only.
    pub(crate) density_vars: Vec<Arc<[VariableId]>>,
    pub(crate) root: usize,
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
        let free_vars = compute_free_vars(&ops, arena);
        let mut origins = vec![root; ops.len()];
        for (expression, slot) in expr_to_slot {
            origins[slot] = ExprId::from_raw(expression);
        }
        let density_vars = compute_density_vars(&ops, arena);
        Ok(Self { ops, origins, free_vars, density_vars, root: root_slot })
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
        // One clone per evaluation: `eval_slot` threads a single mutable
        // scratch assignment through the whole plan, with each scope binding
        // and restoring its own variables (see `with_scoped_bindings`) rather
        // than cloning the assignment per support row.
        let mut scratch = env.clone();
        self.eval_slot(arena, provider, ctx, &mut scratch, self.root)
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
        env: &mut Assignment,
        slot: usize,
    ) -> Result<f64, EvalError> {
        // Density / scalar under `env`. Expectations and contrasts are scalars;
        // other ops are densities in the free variables bound by `env`.
        match &self.ops[slot] {
            EvalOp::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => {
                let original = arena.intervention_assignments(*intervention);
                let mut assignments = std::borrow::Cow::Borrowed(original);
                if original.iter().any(|a| a.is_symbolic()) {
                    for assignment in assignments.to_mut() {
                        if assignment.is_symbolic() {
                            assignment.value = env
                                .get(assignment.variable)
                                .filter(|v| v.as_f64().is_none_or(f64::is_finite))
                                .cloned()
                                .ok_or(EvalError::MissingBinding(assignment.variable))?;
                        }
                    }
                }
                let spec = FactorSpec {
                    variables: arena.var_set(*variables),
                    conditioned_on: arena.var_set(*conditioned_on),
                    intervention: &assignments,
                    domain: *domain,
                    population: arena.population(*population),
                    regime: *regime,
                };
                // Interventions bind targets; bind them into the shared
                // scratch assignment for the lookup, restored on exit.
                with_scoped_bindings(env, spec.intervention.iter().map(|a| a.variable), |env| {
                    for a in spec.intervention {
                        env.set(a.variable, a.value.clone());
                    }
                    provider.probability(&spec, env, ctx)
                })
            }
            EvalOp::Kernel { body, .. } => self.eval_slot(arena, provider, ctx, env, *body),
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
                    return Err(provider.zero_denominator(arena, self.origins[slot], env));
                }
                Ok(num / den)
            }
            EvalOp::Expectation { function, distribution } => {
                with_scoped_bindings(env, [function.variable()], |env| {
                    env.remove(function.variable());
                    self.eval_expectation(
                        arena,
                        provider,
                        ctx,
                        env,
                        function.variable(),
                        *distribution,
                    )
                })
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
        env: &mut Assignment,
        variables: VarSetId,
        body: usize,
    ) -> Result<f64, EvalError> {
        let vars = arena.var_set(variables);
        let rows = provider.support(vars, ctx)?;
        with_scoped_bindings(env, vars.iter().copied(), |env| {
            let mut sum = 0.0;
            for row in rows.iter() {
                if row.len() != vars.len() {
                    return Err(EvalError::SupportShape {
                        expected: vars.len(),
                        actual: row.len(),
                    });
                }
                for (i, &v) in vars.iter().enumerate() {
                    env.set(v, row[i].clone());
                }
                sum += self.eval_slot(arena, provider, ctx, env, body)?;
            }
            Ok(sum)
        })
    }

    fn eval_integral_out(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &mut Assignment,
        variables: VarSetId,
        body: usize,
    ) -> Result<f64, EvalError> {
        let vars = arena.var_set(variables);
        if let Some(nodes) = provider.quadrature(vars, ctx)? {
            return with_scoped_bindings(env, vars.iter().copied(), |env| {
                let mut acc = 0.0;
                for (row, weight) in nodes.iter() {
                    if row.len() != vars.len() {
                        return Err(EvalError::SupportShape {
                            expected: vars.len(),
                            actual: row.len(),
                        });
                    }
                    for (i, &v) in vars.iter().enumerate() {
                        env.set(v, row[i].clone());
                    }
                    acc += *weight * self.eval_slot(arena, provider, ctx, env, body)?;
                }
                Ok(acc)
            });
        }
        // Discrete / counting-measure fallback (IntegralOut ≡ SumOut).
        let rows = provider.support(vars, ctx).map_err(|e| match e {
            EvalError::EmptySupport(_) => EvalError::UnsupportedIntegralOut,
            other => other,
        })?;
        with_scoped_bindings(env, vars.iter().copied(), |env| {
            let mut sum = 0.0;
            for row in rows.iter() {
                if row.len() != vars.len() {
                    return Err(EvalError::SupportShape {
                        expected: vars.len(),
                        actual: row.len(),
                    });
                }
                for (i, &v) in vars.iter().enumerate() {
                    env.set(v, row[i].clone());
                }
                sum += self.eval_slot(arena, provider, ctx, env, body)?;
            }
            Ok(sum)
        })
    }

    fn eval_expectation(
        &self,
        arena: &CausalExprArena,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &mut Assignment,
        outcome_var: VariableId,
        distribution: usize,
    ) -> Result<f64, EvalError> {
        // E[f | D] = Σ_{x ∈ support(free(D))} f(x) · dens(D, x)
        // Free variables per slot are precomputed at compile time; only the
        // env-dependent filtering happens per evaluation.
        // Only variables the density is a density *in* are integrated. A free variable
        // that occurs only behind a conditioning bar (or as a symbolic do-value) is a
        // parameter of the expectation: summing over it is not a mean, so it must be bound.
        let free = &self.free_vars[distribution];
        let random = &self.density_vars[distribution];
        if let Some(parameter) = free.iter().find(|v| env.get(**v).is_none() && !random.contains(v))
        {
            return Err(EvalError::MissingBinding(*parameter));
        }
        let mut enum_vars: Vec<VariableId> =
            free.iter().copied().filter(|v| env.get(*v).is_none()).collect();
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
        with_scoped_bindings(env, enum_vars.iter().copied(), |env| {
            let mut acc = 0.0;
            for row in rows.iter() {
                if row.len() != enum_vars.len() {
                    return Err(EvalError::SupportShape {
                        expected: enum_vars.len(),
                        actual: row.len(),
                    });
                }
                for (i, &v) in enum_vars.iter().enumerate() {
                    env.set(v, row[i].clone());
                }
                let dens = self.eval_slot(arena, provider, ctx, env, distribution)?;
                let y = provider.outcome(outcome_var, env, ctx)?;
                acc += y * dens;
            }
            Ok(acc)
        })
    }
}

/// Run `f` against the shared scratch assignment, then restore any prior
/// bindings of `vars` (removing bindings that did not exist before).
///
/// Evaluation bindings are strictly stack-scoped — sum/integral/expectation
/// rows and intervention targets shadow outer bindings only for the duration
/// of the nested evaluation — so saving and restoring just those variables is
/// observationally identical to the previous clone-per-row scheme, without the
/// per-row `Assignment` clone. Restoration also runs on the error path so a
/// failed inner evaluation leaves the scratch assignment as it found it.
pub(crate) fn with_scoped_bindings<T>(
    env: &mut Assignment,
    vars: impl IntoIterator<Item = VariableId>,
    f: impl FnOnce(&mut Assignment) -> Result<T, EvalError>,
) -> Result<T, EvalError> {
    let saved: Vec<(VariableId, Option<Value>)> =
        vars.into_iter().map(|v| (v, env.get(v).cloned())).collect();
    let result = f(env);
    for (v, prev) in saved {
        match prev {
            Some(value) => env.set(v, value),
            None => {
                env.remove(v);
            }
        }
    }
    result
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
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => EvalOp::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        },
        ExprNode::Kernel { body, population, regime, .. } => {
            let compiled = compile_rec(arena, body, ops, expr_to_slot)?;
            EvalOp::Kernel { body: compiled, population, regime }
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

/// Per-slot variables the slot is a density in (sorted, deduplicated): the `variables`
/// position of its factors, less what is summed or integrated out. A ratio is a density in
/// its numerator's variables. Expectations and contrasts are scalars.
fn compute_density_vars(ops: &[EvalOp], arena: &CausalExprArena) -> Vec<Arc<[VariableId]>> {
    let mut out: Vec<Arc<[VariableId]>> = Vec::with_capacity(ops.len());
    for op in ops {
        let mut vars: Vec<VariableId> = match op {
            EvalOp::Distribution { variables, .. } => arena.var_set(*variables).to_vec(),
            EvalOp::Kernel { body, .. } => out[*body].to_vec(),
            EvalOp::Product { children } => {
                children.iter().flat_map(|&c| out[c].iter().copied()).collect()
            }
            EvalOp::SumOut { variables, body } | EvalOp::IntegralOut { variables, body } => {
                let bound = arena.var_set(*variables);
                out[*body].iter().copied().filter(|v| !bound.contains(v)).collect()
            }
            EvalOp::Ratio { numerator, .. } => out[*numerator].to_vec(),
            EvalOp::Expectation { .. } | EvalOp::Contrast { .. } => Vec::new(),
        };
        vars.sort_by_key(|v| v.raw());
        vars.dedup();
        out.push(Arc::from(vars));
    }
    out
}

/// Per-slot free variables (sorted, deduplicated), computed once per compile.
///
/// Slots are emitted post-order by `compile_rec`, so every child index is
/// smaller than its parent's and a single forward pass suffices.
///
/// The `Distribution` arm must agree with `simplify::free_vars` (see the
/// comment there): `conditioned_on` variables bound by the accompanying
/// `intervention` set are do(·)-fixed, not free.
fn compute_free_vars(ops: &[EvalOp], arena: &CausalExprArena) -> Vec<Arc<[VariableId]>> {
    let mut out: Vec<Arc<[VariableId]>> = Vec::with_capacity(ops.len());
    for op in ops {
        let mut vars: Vec<VariableId> = match op {
            EvalOp::Distribution { variables, conditioned_on, intervention, .. } => {
                let mut vars = arena.var_set(*variables).to_vec();
                let bound = arena.intervention_assignments(*intervention);
                for &v in arena.var_set(*conditioned_on) {
                    if !bound.iter().any(|a| a.variable == v && !a.is_symbolic()) {
                        vars.push(v);
                    }
                }
                vars.extend(bound.iter().filter(|a| a.is_symbolic()).map(|a| a.variable));
                vars
            }
            EvalOp::Kernel { body, .. } => out[*body].to_vec(),
            EvalOp::Product { children } => {
                children.iter().flat_map(|&c| out[c].iter().copied()).collect()
            }
            EvalOp::SumOut { variables, body } | EvalOp::IntegralOut { variables, body } => {
                let bound = arena.var_set(*variables);
                out[*body].iter().copied().filter(|v| !bound.contains(v)).collect()
            }
            EvalOp::Ratio { numerator, denominator } => {
                out[*numerator].iter().chain(out[*denominator].iter()).copied().collect()
            }
            EvalOp::Expectation { function, distribution } => {
                out[*distribution].iter().copied().filter(|v| *v != function.variable()).collect()
            }
            EvalOp::Contrast { left, right, .. } => {
                out[*left].iter().chain(out[*right].iter()).copied().collect()
            }
        };
        vars.sort_by_key(|v| v.raw());
        vars.dedup();
        out.push(Arc::from(vars));
    }
    out
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
                population: "",
                regime: None,
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
                        population: "",
                        regime: None,
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

    /// `E[Y | do(T=1), z]` keeps `z` free. Evaluating it without a value for `z` must be
    /// refused: enumerating `z` inside the expectation would return
    /// `E[Y|do(1),z=0] + E[Y|do(1),z=1] = 1.4`, which is not a mean of anything.
    #[test]
    fn unbound_free_variable_is_refused_not_summed() {
        let mut arena = CausalExprArena::new();
        let (t, y, z) = (v(0), v(1), v(2));
        let ys = arena.intern_var_set([y]);
        let zs = arena.intern_var_set([z]);
        let do_t = arena.intern_intervention_assignments([InterventionAssignment {
            variable: t,
            value: f(1.0),
        }]);
        let conditional = arena.intern_distribution(ys, zs, do_t, DomainRef::Interventional);
        let expr = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(y),
            distribution: conditional,
        });
        let provider = backdoor_provider(t, y, z);
        let compiled = arena.compile(expr).unwrap();
        let ctx = EvalContext::default();
        assert_eq!(compiled.evaluate(&arena, &provider, &ctx), Err(EvalError::MissingBinding(z)));
        for (level, expected) in [(0.0, 0.8), (1.0, 0.6)] {
            let env = Assignment::from_pairs([(z, f(level))]);
            let value = compiled.evaluate_with(&arena, &provider, &ctx, &env).unwrap();
            assert!((value - expected).abs() < 1e-12, "z={level}: {value}");
        }
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
        let simplified = arena.simplify(expr).unwrap();
        let after = arena
            .compile(simplified)
            .unwrap()
            .evaluate(&arena, &provider, &EvalContext::default())
            .unwrap();
        assert!((before - after).abs() < 1e-12, "before={before} after={after}");
        assert!((after - 0.45).abs() < 1e-12);
    }

    #[test]
    fn variable_rename_preserves_dummy_eval() {
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let z = v(2);
        let expr = arena.backdoor_ate(t, y, &[z], f(1.0), f(0.0));
        let provider = backdoor_provider(t, y, z);
        let original = arena
            .compile(expr)
            .unwrap()
            .evaluate(&arena, &provider, &EvalContext::default())
            .unwrap();
        let y2 = v(7);
        assert!(arena.substitute(expr, &[(y, y2)]).is_err());
        let renamed = arena.backdoor_ate(t, y2, &[z], f(1.0), f(0.0));
        let renamed_provider = backdoor_provider(t, y2, z);
        let renamed_value = arena
            .compile(renamed)
            .unwrap()
            .evaluate(&arena, &renamed_provider, &EvalContext::default())
            .unwrap();
        assert!((original - renamed_value).abs() < 1e-12);
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
            let simplified = arena.simplify(expr).unwrap();
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
            population: "",
            regime: None,
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
                    population: "",
                    regime: None,
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
            let simplified = arena.simplify(expr).unwrap();
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
                population: "",
                regime: None,
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
                    population: "",
                    regime: None,
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
                        population: "",
                        regime: None,
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
                population: "",
                regime: None,
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
                    population: "",
                    regime: None,
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
                        population: "",
                        regime: None,
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

        let simplified = arena.simplify(expr).unwrap();
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
        let dist = arena.intern_distribution(zset, empty, empty_i, DomainRef::Observational);
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
                population: "",
                regime: None,
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
        let dist = arena.intern_distribution(xset, empty, empty_i, DomainRef::Observational);
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
        let dist = arena.intern_distribution(both, empty, empty_i, DomainRef::Observational);
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
                population: "",
                regime: None,
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
                        population: "",
                        regime: None,
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
        let dist = arena.intern_distribution(yset, empty, empty_i, DomainRef::Observational);
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
            population: "",
            regime: None,
        };
        p.insert_probability(&spec, &Assignment::from_pairs([(y, f(0.0))]), 0.25).unwrap();
        p.insert_probability(&spec, &Assignment::from_pairs([(y, f(2.0))]), 0.75).unwrap();

        let val =
            arena.compile(exp).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        // 0*0.25 + 2*0.75 = 1.5
        assert!((val - 1.5).abs() < 1e-12);
    }

    #[test]
    fn evaluation_is_stable_across_repeated_calls() {
        // Support memoization and the shared scratch assignment must leave
        // repeated evaluations bitwise identical (nested SumOut + Expectation
        // exercise both caches on the second call).
        let mut arena = CausalExprArena::new();
        let t = v(0);
        let y = v(1);
        let z = v(2);
        let expr = arena.backdoor_ate(t, y, &[z], f(1.0), f(0.0));
        let provider = backdoor_provider(t, y, z);
        let compiled = arena.compile(expr).unwrap();
        let first = compiled.evaluate(&arena, &provider, &EvalContext::default()).unwrap();
        let second = compiled.evaluate(&arena, &provider, &EvalContext::default()).unwrap();
        let third = compiled.evaluate(&arena, &provider, &EvalContext::default()).unwrap();
        assert_eq!(first.to_bits(), second.to_bits());
        assert_eq!(first.to_bits(), third.to_bits());
        assert!((first - 0.45).abs() < 1e-12, "ate={first}");
    }

    #[test]
    fn scoped_intervention_binding_restores_between_siblings() {
        // SumOut_z Product[ P(· | z, do(z:=1)), P(z) ]: the first factor binds
        // z:=1 for its own lookup only; the sibling P(z) must still see the
        // row's z. Correct scoping gives Σ_z 2.0 · P(z) = 2.0; a leaked
        // binding would give 2.0 · P(z=1) per row = 2.8.
        let mut arena = CausalExprArena::new();
        let z = v(0);
        let zset = arena.intern_var_set([z]);
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let do_z1 = arena.intern_intervention_assignments([InterventionAssignment {
            variable: z,
            value: f(1.0),
        }]);
        let shadowed = arena.intern_distribution(empty, zset, do_z1, DomainRef::Observational);
        let z_marginal = arena.intern_distribution(zset, empty, empty_i, DomainRef::Observational);
        let product = {
            let list = arena.intern_list([shadowed, z_marginal]);
            arena.intern(ExprNode::Product(list))
        };
        let sum = arena.intern(ExprNode::SumOut { variables: zset, expr: product });

        let mut p = EmpiricalTableProvider::new();
        p.set_domain(z, [f(0.0), f(1.0)]);
        let interv = [InterventionAssignment { variable: z, value: f(1.0) }];
        let shadow_spec = FactorSpec {
            variables: &[],
            conditioned_on: &[z],
            intervention: &interv,
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        p.insert_probability(&shadow_spec, &Assignment::from_pairs([(z, f(1.0))]), 2.0).unwrap();
        let marg_spec = FactorSpec {
            variables: &[z],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        p.insert_probability(&marg_spec, &Assignment::from_pairs([(z, f(0.0))]), 0.3).unwrap();
        p.insert_probability(&marg_spec, &Assignment::from_pairs([(z, f(1.0))]), 0.7).unwrap();

        let val =
            arena.compile(sum).unwrap().evaluate(&arena, &p, &EvalContext::default()).unwrap();
        assert!((val - 2.0).abs() < 1e-12, "val={val}");
    }

    #[test]
    fn expectation_respects_env_bound_conditioning() {
        // E[Y | z] with z pre-bound in the environment: the compile-time
        // free-variable set of the distribution slot is filtered against the
        // environment, so only Y is enumerated and the bound z selects the
        // right conditional column.
        let mut arena = CausalExprArena::new();
        let y = v(0);
        let z = v(1);
        let yset = arena.intern_var_set([y]);
        let zset = arena.intern_var_set([z]);
        let empty_i = arena.empty_intervention_set();
        let dist = arena.intern_distribution(yset, zset, empty_i, DomainRef::Observational);
        let exp = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(y),
            distribution: dist,
        });

        let mut p = EmpiricalTableProvider::new();
        p.set_domain(y, [f(0.0), f(2.0)]);
        p.set_domain(z, [f(0.0), f(1.0)]);
        let spec = FactorSpec {
            variables: &[y],
            conditioned_on: &[z],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        for (yv, zv, prob) in [(0.0, 0.0, 0.25), (2.0, 0.0, 0.75), (0.0, 1.0, 1.0), (2.0, 1.0, 0.0)]
        {
            p.insert_probability(&spec, &Assignment::from_pairs([(y, f(yv)), (z, f(zv))]), prob)
                .unwrap();
        }
        let compiled = arena.compile(exp).unwrap();
        let env0 = Assignment::from_pairs([(z, f(0.0))]);
        let e0 = compiled.evaluate_with(&arena, &p, &EvalContext::default(), &env0).unwrap();
        assert!((e0 - 1.5).abs() < 1e-12, "E[Y|z=0]={e0}");
        let env1 = Assignment::from_pairs([(z, f(1.0))]);
        let e1 = compiled.evaluate_with(&arena, &p, &EvalContext::default(), &env1).unwrap();
        assert!(e1.abs() < 1e-12, "E[Y|z=1]={e1}");
        // The caller's environment is never mutated by evaluation.
        assert_eq!(env0.entries(), &[(z, f(0.0))]);
    }

    #[test]
    fn ratio_zero_denominator_is_division_by_zero() {
        // `EvalOp::Ratio` must reject an exactly-zero denominator rather than
        // returning `f64::INFINITY`/NaN. `iv_wald_is_ratio_of_instrument_contrasts`
        // (lib.rs) only checks the compiled shape, not this evaluation-time guard.
        let mut arena = CausalExprArena::new();
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        // Two vacuous (no free variables) factors, distinguished by domain so they
        // hash-cons to distinct nodes with independently settable probabilities.
        let numerator = arena.intern_distribution(empty, empty, empty_i, DomainRef::Observational);
        let denominator =
            arena.intern_distribution(empty, empty, empty_i, DomainRef::Interventional);
        let ratio = arena.intern(ExprNode::Ratio { numerator, denominator });

        let mut p = EmpiricalTableProvider::new();
        let obs_spec = FactorSpec {
            variables: &[],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        let interv_spec = FactorSpec {
            variables: &[],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Interventional,
            population: "",
            regime: None,
        };
        p.insert_probability(&obs_spec, &Assignment::from_pairs([]), 3.0).unwrap();
        p.insert_probability(&interv_spec, &Assignment::from_pairs([]), 0.0).unwrap();

        let err = arena
            .compile(ratio)
            .unwrap()
            .evaluate(&arena, &p, &EvalContext::default())
            .unwrap_err();
        assert_eq!(err, EvalError::DivisionByZero);
    }

    /// A genuine `do(T=NaN)` must not alias the symbolic wildcard. Wrapped in
    /// `SumOut(T, ·)`, a symbolic coordinate becomes `Σ_t E[Y|do(T=t)]`; a
    /// concrete NaN must not evaluate to that sum.
    #[test]
    fn nan_intervention_does_not_sum_over_treatment_support() {
        let mut arena = CausalExprArena::new();
        let (t, y) = (v(0), v(1));
        let mut p = EmpiricalTableProvider::new();
        p.set_domain(t, [f(0.0), f(1.0)]);
        p.set_domain(y, [f(0.0), f(1.0)]);
        for (tlev, ey) in [(0.0, 0.2), (1.0, 0.8)] {
            let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
            for (yval, prob) in [(1.0, ey), (0.0, 1.0 - ey)] {
                let spec = FactorSpec {
                    variables: &[y],
                    conditioned_on: &[],
                    intervention: &interv,
                    domain: DomainRef::Interventional,
                    population: "",
                    regime: None,
                };
                let assign = Assignment::from_pairs([(y, f(yval))]);
                p.insert_probability(&spec, &assign, prob).unwrap();
            }
        }

        let ys = arena.intern_var_set([y]);
        let ts = arena.intern_var_set([t]);
        let empty = arena.empty_var_set();
        let ctx = EvalContext::default();

        // Σ_t E[Y|do(T=t)] via symbolic coordinate under SumOut.
        let do_sym = arena.intern_intervention_set([t]);
        let sym_dist = arena.intern_distribution(ys, empty, do_sym, DomainRef::Interventional);
        let sym_mean = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(y),
            distribution: sym_dist,
        });
        let sym_sum = arena.intern(ExprNode::SumOut { variables: ts, expr: sym_mean });
        let sum_over_t = arena.compile(sym_sum).unwrap().evaluate(&arena, &p, &ctx).unwrap();
        assert!((sum_over_t - 1.0).abs() < 1e-12, "symbolic Σ_t E[Y|do(t)] = {sum_over_t}");

        // Same SumOut shape with concrete NaN must not yield that sum.
        let do_nan = arena.intern_intervention_assignments([InterventionAssignment {
            variable: t,
            value: f(f64::NAN),
        }]);
        assert!(!arena.intervention_assignments(do_nan)[0].is_symbolic());
        let nan_dist = arena.intern_distribution(ys, empty, do_nan, DomainRef::Interventional);
        let nan_mean = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(y),
            distribution: nan_dist,
        });
        let nan_sum = arena.intern(ExprNode::SumOut { variables: ts, expr: nan_mean });
        let nan_result = arena.compile(nan_sum).unwrap().evaluate(&arena, &p, &ctx);
        assert_ne!(nan_result, Ok(sum_over_t), "NaN intervention aliased sum-over-T");
        assert_eq!(nan_result, Err(EvalError::MissingTableEntry));
    }

    #[test]
    fn simplification_preserves_overlapping_binding_multiplicity() {
        // The inner x shadows the outer x: summing over outer (x,y)
        // therefore counts each inner marginal twice. Unioning binders loses 2.
        for integral in [false, true] {
            let mut arena = CausalExprArena::new();
            let empty = arena.empty_var_set();
            let empty_i = arena.empty_intervention_set();
            let x = v(0);
            let y = v(1);
            let xset = arena.intern_var_set([x]);
            let xyset = arena.intern_var_set([x, y]);
            let dist = arena.intern_distribution(xyset, empty, empty_i, DomainRef::Observational);
            let inner = arena.intern(if integral {
                ExprNode::IntegralOut { variables: xset, expr: dist }
            } else {
                ExprNode::SumOut { variables: xset, expr: dist }
            });
            let root = arena.intern(if integral {
                ExprNode::IntegralOut { variables: xyset, expr: inner }
            } else {
                ExprNode::SumOut { variables: xyset, expr: inner }
            });
            let simplified = arena.simplify(root).unwrap();
            let mut provider = EmpiricalTableProvider::new();
            provider.set_domain(x, [f(0.0), f(1.0)]);
            provider.set_domain(y, [f(0.0), f(1.0)]);
            let spec = FactorSpec {
                variables: &[x, y],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
                population: "",
                regime: None,
            };
            for xv in [0.0, 1.0] {
                for yv in [0.0, 1.0] {
                    provider
                        .insert_probability(
                            &spec,
                            &Assignment::from_pairs([(x, f(xv)), (y, f(yv))]),
                            0.25,
                        )
                        .unwrap();
                }
            }
            for expr in [root, simplified] {
                let value = arena
                    .compile(expr)
                    .unwrap()
                    .evaluate(&arena, &provider, &EvalContext::default())
                    .unwrap();
                assert!((value - 2.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn simplification_preserves_nested_ratio_zero_denominator() {
        let mut arena = CausalExprArena::new();
        let empty = arena.empty_var_set();
        let intervention = arena.empty_intervention_set();
        let one = arena.intern_distribution(empty, empty, intervention, DomainRef::Observational);
        let zero = arena.intern_distribution(empty, empty, intervention, DomainRef::Interventional);
        let inner = arena.intern(ExprNode::Ratio { numerator: one, denominator: zero });
        let outer = arena.intern(ExprNode::Ratio { numerator: one, denominator: inner });
        let simplified = arena.simplify(outer).unwrap();
        let mut provider = EmpiricalTableProvider::new();
        for (domain, value) in [(DomainRef::Observational, 1.0), (DomainRef::Interventional, 0.0)] {
            provider
                .insert_probability(
                    &FactorSpec {
                        variables: &[],
                        conditioned_on: &[],
                        intervention: &[],
                        domain,
                        population: "",
                        regime: None,
                    },
                    &Assignment::new(),
                    value,
                )
                .unwrap();
        }
        for expr in [outer, simplified] {
            assert_eq!(
                arena.compile(expr).unwrap().evaluate(&arena, &provider, &EvalContext::default()),
                Err(EvalError::DivisionByZero)
            );
        }
    }
}

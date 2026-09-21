//! Provider-frozen memoized elimination with checked zero-mass extensions.
use crate::eval::{EvalOp, with_scoped_bindings};
use crate::{
    Assignment, CausalExprArena, CompiledEvaluator, DistributionProvider, EvalContext, EvalError,
    FactorSpec,
};
use antecedent_core::{Value, VariableId};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
enum Term {
    Defined(f64),
    Extendable(EvalError),
}
impl Term {
    fn required(self) -> Result<f64, EvalError> {
        match self {
            Self::Defined(value) => Ok(value),
            Self::Extendable(error) => Err(error),
        }
    }
}
type SupportRows = Arc<[Arc<[Value]>]>;

/// Cache lifetime is exactly one immutable provider-bound execution, including
/// all target atoms. Keys project assignments onto each compiled node's free axes.
pub(crate) struct ExactSession<'a> {
    arena: &'a CausalExprArena,
    plan: &'a CompiledEvaluator,
    provider: &'a dyn DistributionProvider,
    cache: HashMap<(usize, Vec<Value>), Term>,
    supports: HashMap<Vec<VariableId>, SupportRows>,
    pub support: Vec<crate::exact_plan::ExactSupportRecord>,
    ctx: &'a antecedent_core::ExecutionContext,
    remaining: usize,
}
impl<'a> ExactSession<'a> {
    pub fn new(
        arena: &'a CausalExprArena,
        plan: &'a CompiledEvaluator,
        provider: &'a dyn DistributionProvider,
        ctx: &'a antecedent_core::ExecutionContext,
        operations: usize,
    ) -> Self {
        Self {
            arena,
            plan,
            provider,
            cache: HashMap::new(),
            supports: HashMap::new(),
            support: Vec::new(),
            ctx,
            remaining: operations,
        }
    }
    pub fn evaluate(&mut self, assignment: &mut Assignment) -> Result<f64, EvalError> {
        self.slot(self.plan.root, assignment)?.required()
    }
    #[allow(clippy::too_many_lines)] // Keep exact operation and null-event semantics in one dispatch.
    fn slot(&mut self, slot: usize, env: &mut Assignment) -> Result<Term, EvalError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(EvalError::ProviderKind("exact evaluation cancelled"));
        }
        self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or(EvalError::ProviderKind("exact operation budget exceeded"))?;
        let key = (
            slot,
            self.plan.free_vars[slot]
                .iter()
                .map(|v| {
                    let value = env.get(*v).ok_or(EvalError::MissingBinding(*v))?;
                    Ok(match value {
                        Value::Float64(x) if *x == 0.0 => Value::Float64(0.0),
                        other => other.clone(),
                    })
                })
                .collect::<Result<Vec<_>, EvalError>>()?,
        );
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        let mut denominator_value = None;
        let result = match self.plan.ops[slot].clone() {
            EvalOp::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => {
                let mut assignments = self.arena.intervention_assignments(intervention).to_vec();
                for a in &mut assignments {
                    if a.is_symbolic() {
                        a.value = env
                            .get(a.variable)
                            .cloned()
                            .ok_or(EvalError::MissingBinding(a.variable))?;
                    }
                }
                let spec = FactorSpec {
                    variables: self.arena.var_set(variables),
                    conditioned_on: self.arena.var_set(conditioned_on),
                    intervention: &assignments,
                    domain,
                    population: self.arena.population(population),
                    regime,
                };
                match with_scoped_bindings(env, assignments.iter().map(|a| a.variable), |env| {
                    for a in &assignments {
                        env.set(a.variable, a.value.clone());
                    }
                    self.provider.probability(&spec, env, &EvalContext::default())
                }) {
                    Ok(value) => Term::Defined(value),
                    Err(error @ EvalError::ExactLaw(_)) if matches!(&error,EvalError::ExactLaw(e) if e.kind=="zero_conditioning_mass") => {
                        Term::Extendable(error)
                    }
                    Err(error) => return Err(error),
                }
            }
            EvalOp::Kernel { body, .. } => self.slot(body, env)?,
            EvalOp::Product { children } => {
                let mut value = 1.0;
                let mut explicit_zero = false;
                let mut undefined = None;
                for &child in children.iter() {
                    match self.slot(child, env)? {
                        Term::Defined(x) => {
                            explicit_zero |= x == 0.0;
                            let next = value * x;
                            if value != 0.0 && x != 0.0 && next == 0.0 {
                                return Err(EvalError::ProviderKind("exact product underflow"));
                            }
                            if !next.is_finite() {
                                return Err(EvalError::ProviderKind("exact product overflow"));
                            }
                            value = next;
                        }
                        Term::Extendable(error) => {
                            undefined.get_or_insert(error);
                        }
                    }
                }
                if let Some(error) = undefined {
                    if explicit_zero {
                        // The only deferred terms are conditionals on null events,
                        // or finite products thereof. Every stochastic extension is
                        // bounded. An explicit zero multiplier therefore makes the
                        // entire summand zero for every admissible extension.
                        self.support.push(crate::exact_plan::ExactSupportRecord {
                            expression: self.plan.origins[slot],
                            assignment: env.entries().into(),
                            status: "checked_zero_mass_extension",
                            denominator: None,
                        });
                        Term::Defined(0.0)
                    } else {
                        Term::Extendable(error)
                    }
                } else {
                    Term::Defined(value)
                }
            }
            EvalOp::SumOut { variables, body } | EvalOp::IntegralOut { variables, body } => {
                let variables = self.arena.var_set(variables).to_vec();
                let rows = if let Some(rows) = self.supports.get(&variables) {
                    rows.clone()
                } else {
                    let rows = self.provider.support(&variables, &EvalContext::default())?;
                    self.supports.insert(variables.clone(), rows.clone());
                    rows
                };
                let value = with_scoped_bindings(env, variables.iter().copied(), |env| {
                    let (mut total, mut correction) = (0.0, 0.0);
                    for row in rows.iter() {
                        for (v, x) in variables.iter().zip(row.iter()) {
                            env.set(*v, x.clone());
                        }
                        let value = self.slot(body, env)?.required()?;
                        let adjusted = value - correction;
                        let next = total + adjusted;
                        correction = (next - total) - adjusted;
                        total = next;
                    }
                    Ok(total)
                })?;
                Term::Defined(value)
            }
            EvalOp::Ratio { numerator, denominator } => {
                let num = self.slot(numerator, env)?.required()?;
                let den = self.slot(denominator, env)?.required()?;
                denominator_value = Some(den);
                if den == 0.0 {
                    let error =
                        self.provider.zero_denominator(self.arena, self.plan.origins[slot], env);
                    // A denominator that literally marginalizes this numerator
                    // certifies a bounded conditional extension, never a generic 0/0.
                    if num == 0.0
                        && matches!(self.plan.ops[denominator],EvalOp::SumOut { body,.. } if body==numerator)
                    {
                        Term::Extendable(error)
                    } else {
                        return Err(error);
                    }
                } else {
                    Term::Defined(num / den)
                }
            }
            _ => return Err(EvalError::ProviderKind("unsupported exact physical operation")),
        };
        if matches!(self.plan.ops[slot], EvalOp::Distribution { .. } | EvalOp::Ratio { .. }) {
            self.support.push(crate::exact_plan::ExactSupportRecord {
                expression: self.plan.origins[slot],
                assignment: env.entries().into(),
                status: if matches!(result, Term::Defined(_)) {
                    "checked"
                } else {
                    "null_condition_requires_zero_weight"
                },
                denominator: denominator_value,
            });
        }
        if let Term::Defined(value) = result {
            if !value.is_finite() || value < 0.0 {
                return Err(EvalError::ProviderKind("nonfinite or negative exact intermediate"));
            }
            self.cache.insert(key, Term::Defined(value));
            Ok(Term::Defined(value))
        } else {
            self.cache.insert(key, result.clone());
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct CountingLaw {
        probability: f64,
        calls: Cell<usize>,
    }
    impl DistributionProvider for CountingLaw {
        fn probability(
            &self,
            _: &FactorSpec<'_>,
            _: &Assignment,
            _: &EvalContext,
        ) -> Result<f64, EvalError> {
            self.calls.set(self.calls.get() + 1);
            Ok(self.probability)
        }
        fn support(&self, _: &[VariableId], _: &EvalContext) -> Result<SupportRows, EvalError> {
            Ok(Arc::from([Arc::from([Value::Int64(0)]), Arc::from([Value::Int64(1)])]))
        }
        fn outcome(
            &self,
            v: VariableId,
            a: &Assignment,
            _: &EvalContext,
        ) -> Result<f64, EvalError> {
            a.get(v).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(v))
        }
        fn n_draws(&self) -> Option<usize> {
            None
        }
    }

    #[test]
    fn shared_subexpressions_reuse_values_only_within_one_provider_execution() {
        let mut arena = CausalExprArena::new();
        let variable = VariableId::from_raw(0);
        let vars = arena.intern_var_set([variable]);
        let empty = arena.empty_var_set();
        let interventions = arena.intern_intervention_set([]);
        let leaf =
            arena.intern_distribution(vars, empty, interventions, crate::DomainRef::Observational);
        let list = arena.intern_list([leaf, leaf]);
        let root = arena.intern(crate::ExprNode::Product(list));
        let plan = arena.compile(root).unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let first = CountingLaw { probability: 0.5, calls: Cell::new(0) };
        let mut session = ExactSession::new(&arena, &plan, &first, &ctx, 100);
        let mut assignment = Assignment::from_pairs([(variable, Value::Int64(0))]);
        assert!((session.evaluate(&mut assignment).unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(first.calls.get(), 1);
        assignment.set(VariableId::from_raw(9), Value::Int64(42));
        assert!((session.evaluate(&mut assignment).unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(first.calls.get(), 1, "irrelevant bindings do not defeat CSE");
        assignment.set(variable, Value::Int64(1));
        session.evaluate(&mut assignment).unwrap();
        assert_eq!(first.calls.get(), 2, "different free assignments remain distinct");
        let replacement = CountingLaw { probability: 0.25, calls: Cell::new(0) };
        let mut session = ExactSession::new(&arena, &plan, &replacement, &ctx, 100);
        assert!((session.evaluate(&mut assignment).unwrap() - 0.0625).abs() < 1e-12);
        assert_eq!(replacement.calls.get(), 1, "provider replacement cannot reuse old values");
    }
}

//! Resource-bounded evaluation of immutable population-tagged exact functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::{
    Assignment, CausalExprArena, CompiledEvaluator, DistributionProvider, EvalContext, EvalError,
    ExactTransportData, ExprId, ExprNode, FactorSpec, LawTolerance,
};
use antecedent_core::{ExecutionContext, Value, VariableId};

/// Explicit exact execution limits. Memory and cancellation also use `ExecutionContext`.
#[derive(Clone, Copy, Debug)]
pub struct ExactEvaluationLimits {
    /// Maximum expanded expression operations and provider row inspections, one budget shared
    /// by both.
    pub operations: usize,
    /// Maximum nested expression depth.
    pub depth: usize,
}
impl Default for ExactEvaluationLimits {
    fn default() -> Self {
        Self { operations: 10_000_000, depth: 256 }
    }
}

/// Factor/assignment support premise consumed during exact execution.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactSupportRecord {
    /// Original functional node.
    pub expression: ExprId,
    /// Concrete original-coordinate assignment.
    pub assignment: Arc<[(VariableId, Value)]>,
    /// Checked, null conditional, or explicit zero-mass extension.
    pub status: &'static str,
    /// Evaluated denominator when applicable.
    pub denominator: Option<f64>,
}

/// Full target distribution. No sampling uncertainty is implied by this result.
#[derive(Clone, Debug)]
pub struct ExactDistribution {
    /// Outcome coordinates in atom order.
    pub outcomes: Arc<[VariableId]>,
    /// Complete assignment domain, including zero probability atoms.
    pub atoms: Arc<[Arc<[Value]>]>,
    /// Probabilities in the same order as atoms.
    pub probabilities: Arc<[f64]>,
    /// Factor-level support and any checked zero-mass extension rules.
    pub support: Arc<[ExactSupportRecord]>,
}
impl ExactDistribution {
    /// Mean of a numeric outcome coordinate under the evaluated law.
    ///
    /// # Errors
    /// Unknown or nonnumeric outcome.
    pub fn mean(&self, outcome: VariableId) -> Result<f64, EvalError> {
        let axis = self
            .outcomes
            .iter()
            .position(|v| *v == outcome)
            .ok_or(EvalError::MissingBinding(outcome))?;
        if self.atoms.len() != self.probabilities.len() {
            return Err(EvalError::ProviderKind("invalid exact distribution shape"));
        }
        let mut invalid = false;
        let total =
            crate::exact::sum(self.atoms.iter().zip(self.probabilities.iter()).map(|(atom, p)| {
                if let Some(value) = atom.get(axis).and_then(Value::as_f64) {
                    value * p
                } else {
                    invalid = true;
                    f64::NAN
                }
            }));
        if invalid || !total.is_finite() {
            return Err(EvalError::ProviderKind("exact mean requires finite numeric outcomes"));
        }
        Ok(total)
    }
    /// Difference of numeric means between two complete laws.
    ///
    /// # Errors
    /// Incompatible outcome coordinates or nonnumeric means.
    pub fn mean_difference(&self, reference: &Self, outcome: VariableId) -> Result<f64, EvalError> {
        if self.outcomes != reference.outcomes {
            return Err(EvalError::ProviderKind("contrast outcome coordinates differ"));
        }
        let value = self.mean(outcome)? - reference.mean(outcome)?;
        if !value.is_finite() {
            return Err(EvalError::ProviderKind("nonfinite exact contrast"));
        }
        Ok(value)
    }
}

/// A physical plan frozen to its expression, concrete request and provider snapshots.
/// This layer evaluates expressions; causal authorization belongs to the identifier.
#[derive(Clone, Debug)]
pub struct ExactEvaluationPlan {
    arena: CausalExprArena,
    root: ExprId,
    evaluator: CompiledEvaluator,
    data: ExactTransportData,
    outcomes: Arc<[VariableId]>,
    request: Assignment,
    limits: ExactEvaluationLimits,
    tolerance: LawTolerance,
}

impl ExactEvaluationPlan {
    /// Bind providers and preflight all expression leaves and expansion costs.
    ///
    /// # Errors
    /// Missing providers/assignments, unsupported scope, or resource exhaustion.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        arena: &CausalExprArena,
        root: ExprId,
        data: ExactTransportData,
        outcomes: impl Into<Arc<[VariableId]>>,
        request: Assignment,
        limits: ExactEvaluationLimits,
        tolerance: LawTolerance,
        ctx: &ExecutionContext,
    ) -> Result<Self, EvalError> {
        if root.raw() as usize >= arena.len()
            || limits.depth == 0
            || limits.operations == 0
            || !tolerance.absolute.is_finite()
            || !tolerance.relative.is_finite()
            || tolerance.absolute < 0.0
            || tolerance.relative < 0.0
            || tolerance.absolute + tolerance.relative >= 1.0
        {
            return Err(EvalError::ProviderKind("invalid exact evaluation request"));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(EvalError::ProviderKind("exact evaluation cancelled"));
        }
        if request
            .entries()
            .iter()
            .any(|(_, value)| matches!(value, Value::Float64(x) if !x.is_finite()))
        {
            return Err(EvalError::ProviderKind("exact request contains a nonfinite assignment"));
        }
        let outcomes = outcomes.into();
        if outcomes.is_empty()
            || outcomes.iter().collect::<BTreeSet<_>>().len() != outcomes.len()
            || outcomes.iter().any(|v| request.get(*v).is_some())
        {
            return Err(EvalError::ProviderKind("invalid exact outcome coordinates"));
        }
        let arena = arena.clone();
        let mut preflight = Preflight {
            arena: &arena,
            data: &data,
            request: &request,
            limits,
            ctx,
            bound: outcomes.iter().copied().collect(),
            intermediate_bytes: Cell::new(0),
            remaining_checks: Cell::new(limits.operations),
        };
        let atoms = preflight.cardinality(&outcomes)?;
        // The depth-bounded preflight walk runs before any unbounded recursion (free-variable
        // analysis, compilation), so a deep expression is refused rather than overflowing the stack.
        let cost = preflight.visit(root, 0)?;
        if cost.checked_mul(atoms).is_none_or(|n| n > limits.operations) {
            return Err(EvalError::ProviderKind("exact operation budget exceeded"));
        }
        for v in arena.free_variables(root) {
            if !outcomes.contains(&v) && request.get(v).is_none() {
                return Err(EvalError::MissingBinding(v));
            }
        }
        let evaluator = arena.compile(root)?;
        check_cache_memory(
            &evaluator,
            &data,
            &request,
            limits,
            preflight.intermediate_bytes.get(),
            ctx,
        )?;
        Ok(Self { arena, root, evaluator, data, outcomes, request, limits, tolerance })
    }
    /// Original, unoptimized functional identity within the retained arena.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
    /// Inspect the original functional without invoking providers.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Evaluate atomically; no partial result escapes on failure.
    ///
    /// # Errors
    /// Located support failure, cancellation, resource exhaustion, or invalid mass.
    pub fn evaluate(&self, ctx: &ExecutionContext) -> Result<ExactDistribution, EvalError> {
        // Refresh may impose a smaller execution budget than preparation did.
        let mut preflight = Preflight {
            arena: &self.arena,
            data: &self.data,
            request: &self.request,
            limits: self.limits,
            ctx,
            bound: self.outcomes.iter().copied().collect(),
            intermediate_bytes: Cell::new(0),
            remaining_checks: Cell::new(self.limits.operations),
        };
        preflight.cardinality(&self.outcomes)?;
        preflight.visit(self.root, 0)?;
        check_cache_memory(
            &self.evaluator,
            &self.data,
            &self.request,
            self.limits,
            preflight.intermediate_bytes.get(),
            ctx,
        )?;
        // One budget covers expression operations and provider row inspections alike.
        let budget = Cell::new(self.limits.operations);
        let provider = BoundedProvider { data: &self.data, ctx, remaining: &budget };
        let eval = EvalContext::default();
        let atoms = provider.support(&self.outcomes, &eval)?;
        let mut probabilities = Vec::with_capacity(atoms.len());
        let mut session = crate::exact_engine::ExactSession::new(
            &self.arena,
            &self.evaluator,
            &provider,
            ctx,
            &budget,
        );
        let mut assignment = self.request.clone();
        for atom in atoms.iter() {
            provider.charge(1)?;
            for (v, value) in self.outcomes.iter().zip(atom.iter()) {
                assignment.set(*v, value.clone());
            }
            let p = session.evaluate(&mut assignment)?;
            if !p.is_finite()
                || p < 0.0
                || p > 1.0 + self.tolerance.absolute + self.tolerance.relative
            {
                return Err(EvalError::ProviderKind("invalid exact target probability"));
            }
            probabilities.push(p);
        }
        let total = crate::exact::sum(probabilities.iter().copied());
        if !total.is_finite()
            || (total - 1.0).abs() > self.tolerance.absolute + self.tolerance.relative
        {
            return Err(EvalError::ProviderKind("unnormalized exact target distribution"));
        }
        Ok(ExactDistribution {
            outcomes: self.outcomes.clone(),
            atoms,
            probabilities: probabilities.into(),
            support: session.support.into(),
        })
    }
}

fn check_cache_memory(
    evaluator: &CompiledEvaluator,
    data: &ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    intermediates: usize,
    ctx: &ExecutionContext,
) -> Result<(), EvalError> {
    let bad = || EvalError::ProviderKind("exact cache memory/operation budget exceeded");
    let all_coordinates: BTreeSet<_> = evaluator.free_vars.iter().flat_map(|v| v.iter()).collect();
    let entry_bytes = all_coordinates
        .len()
        .checked_mul(std::mem::size_of::<(VariableId, Value)>() * 3)
        .and_then(|n| n.checked_add(512))
        .ok_or_else(bad)?;
    let mut entries = 0usize;
    for variables in &evaluator.free_vars {
        if ctx.cancellation.is_cancelled() {
            return Err(EvalError::ProviderKind("exact evaluation cancelled"));
        }
        let mut count = 1usize;
        for variable in variables.iter() {
            let cardinality = data
                .domain(*variable)
                .map_or_else(|| usize::from(request.get(*variable).is_some()), <[Value]>::len);
            count = count
                .checked_mul(cardinality)
                .filter(|n| *n <= limits.operations)
                .ok_or_else(bad)?;
        }
        entries = entries.checked_add(count).ok_or_else(bad)?;
    }
    let bytes = entries
        .checked_mul(entry_bytes)
        .and_then(|n| n.checked_add(intermediates))
        .and_then(|n| n.checked_add(data.factor_cache_bytes()?))
        .ok_or_else(bad)?;
    if ctx.memory.hard_limit_bytes.is_some_and(|n| u64::try_from(bytes).map_or(true, |b| b > n)) {
        return Err(bad());
    }
    Ok(())
}

fn factor_work(law: &crate::ExactDiscreteLaw, spec: &FactorSpec<'_>) -> Result<usize, EvalError> {
    spec.variables
        .len()
        .checked_add(spec.conditioned_on.len())
        .and_then(|n| n.checked_add(spec.intervention.len()))
        .and_then(|n| n.checked_add(2))
        .and_then(|n| n.checked_mul(law.probabilities().len()))
        .ok_or(EvalError::ProviderKind("exact factor cost overflow"))
}

struct Preflight<'a> {
    arena: &'a CausalExprArena,
    data: &'a ExactTransportData,
    request: &'a Assignment,
    limits: ExactEvaluationLimits,
    ctx: &'a ExecutionContext,
    bound: BTreeSet<VariableId>,
    intermediate_bytes: Cell<usize>,
    remaining_checks: Cell<usize>,
}
impl Preflight<'_> {
    fn charge_check(&self) -> Result<(), EvalError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(EvalError::ProviderKind("exact evaluation cancelled"));
        }
        let remaining = self
            .remaining_checks
            .get()
            .checked_sub(1)
            .ok_or(EvalError::ProviderKind("exact preflight budget exceeded"))?;
        self.remaining_checks.set(remaining);
        Ok(())
    }
    fn cardinality(&self, vars: &[VariableId]) -> Result<usize, EvalError> {
        let mut size = 1usize;
        for v in vars {
            let domain = self.data.domain(*v).ok_or(EvalError::EmptySupport(*v))?;
            size = size
                .checked_mul(domain.len())
                .filter(|n| *n <= self.limits.operations)
                .ok_or(EvalError::ProviderKind("exact domain budget exceeded"))?;
        }
        // Conservative bound for the materialized support, values, and output masses.
        let bytes = size
            .checked_mul(
                vars.len()
                    .checked_mul(std::mem::size_of::<Value>())
                    .and_then(|n| n.checked_add(64))
                    .ok_or(EvalError::ProviderKind("exact size overflow"))?,
            )
            .ok_or(EvalError::ProviderKind("exact size overflow"))?;
        // Sum all syntactic support allocations as a conservative live bound:
        // nested marginalizations retain their outer support while evaluating
        // inner factors. Checking only the largest support understates that peak.
        let bytes = self
            .intermediate_bytes
            .get()
            .checked_add(bytes)
            .ok_or(EvalError::ProviderKind("exact size overflow"))?;
        self.intermediate_bytes.set(bytes);
        if self
            .ctx
            .memory
            .hard_limit_bytes
            .is_some_and(|limit| u64::try_from(bytes).map_or(true, |b| b > limit))
        {
            return Err(EvalError::ProviderKind("exact memory budget exceeded"));
        }
        Ok(size)
    }
    fn leaf_cost(&self, leaf: &FactorSpec<'_>) -> Result<usize, EvalError> {
        let assignments = leaf.intervention;
        let mut domains = Vec::new();
        let mut world_count = 1usize;
        for a in assignments {
            let values = if a.is_symbolic() && !self.bound.contains(&a.variable) {
                vec![
                    self.request
                        .get(a.variable)
                        .ok_or(EvalError::MissingBinding(a.variable))?
                        .clone(),
                ]
            } else if a.is_symbolic() {
                // A symbolic coordinate may be bound by an enclosing sum or
                // by the target atom at execution; cover its complete domain.
                let mut values =
                    self.data.domain(a.variable).map(<[Value]>::to_vec).unwrap_or_default();
                if values.is_empty() {
                    values = self
                        .data
                        .laws()
                        .iter()
                        .flat_map(crate::ExactDiscreteLaw::interventions)
                        .filter(|assignment| assignment.variable == a.variable)
                        .map(|a| a.value.clone())
                        .collect();
                    values.dedup();
                }
                if let Some(value) = self.request.get(a.variable) {
                    if !values.iter().any(|level| {
                        crate::exact::value_key(level) == crate::exact::value_key(value)
                    }) {
                        values.push(value.clone());
                    }
                }
                if values.is_empty() {
                    return Err(EvalError::MissingBinding(a.variable));
                }
                values
            } else {
                vec![a.value.clone()]
            };
            world_count = world_count
                .checked_mul(values.len())
                .filter(|n| *n <= self.limits.operations)
                .ok_or(EvalError::ProviderKind("exact intervention coverage budget exceeded"))?;
            domains.push(values);
        }
        let mut largest = 0;
        let mut world = assignments.to_vec();
        for index in 0..world_count {
            self.charge_check()?;
            let mut remainder = index;
            for (assignment, domain) in world.iter_mut().zip(domains.iter()).rev() {
                assignment.value = domain[remainder % domain.len()].clone();
                remainder /= domain.len();
            }
            let spec = FactorSpec {
                variables: leaf.variables,
                conditioned_on: leaf.conditioned_on,
                intervention: &world,
                domain: leaf.domain,
                population: leaf.population,
                regime: leaf.regime,
            };
            let law =
                self.data.require_factor(&spec).map_err(|e| EvalError::ExactLaw(Box::new(e)))?;
            largest = largest.max(factor_work(law, &spec)?);
        }
        Ok(largest)
    }
    fn visit(&mut self, id: ExprId, depth: usize) -> Result<usize, EvalError> {
        self.charge_check()?;
        if depth >= self.limits.depth {
            return Err(EvalError::ProviderKind("exact recursion budget exceeded"));
        }
        let cost = match self.arena.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => Some(self.leaf_cost(&FactorSpec {
                variables: self.arena.var_set(*variables),
                conditioned_on: self.arena.var_set(*conditioned_on),
                intervention: self.arena.intervention_assignments(*intervention),
                domain: *domain,
                population: self.arena.population(*population),
                regime: *regime,
            })?),
            ExprNode::Kernel { body, .. } => Some(self.visit(*body, depth + 1)?),
            ExprNode::Product(list) => {
                let mut total = 0usize;
                for child in self.arena.list(*list) {
                    total = total
                        .checked_add(self.visit(*child, depth + 1)?)
                        .ok_or(EvalError::ProviderKind("exact cost overflow"))?;
                }
                Some(total)
            }
            ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
                let count = self.cardinality(self.arena.var_set(*variables))?;
                let saved = self.bound.clone();
                self.bound.extend(self.arena.var_set(*variables));
                let body = self.visit(*expr, depth + 1)?;
                self.bound = saved;
                count.checked_mul(body)
            }
            ExprNode::Ratio { numerator, denominator } => {
                self.visit(*numerator, depth + 1)?.checked_add(self.visit(*denominator, depth + 1)?)
            }
            ExprNode::Expectation { .. } | ExprNode::Contrast { .. } => {
                return Err(EvalError::ProviderKind(
                    "exact distribution root requires a law; derive functionals after evaluation",
                ));
            }
        };
        cost.and_then(|n| n.checked_add(1))
            .filter(|n| *n <= self.limits.operations)
            .ok_or(EvalError::ProviderKind("exact operation budget exceeded"))
    }
}
struct BoundedProvider<'a> {
    data: &'a ExactTransportData,
    ctx: &'a ExecutionContext,
    remaining: &'a Cell<usize>,
}
impl BoundedProvider<'_> {
    fn charge(&self, cost: usize) -> Result<(), EvalError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(EvalError::ProviderKind("exact evaluation cancelled"));
        }
        let remaining = self
            .remaining
            .get()
            .checked_sub(cost)
            .ok_or(EvalError::ProviderKind("exact operation budget exceeded"))?;
        self.remaining.set(remaining);
        Ok(())
    }
}
impl DistributionProvider for BoundedProvider<'_> {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let law = self.data.require_factor(spec).map_err(|e| EvalError::ExactLaw(Box::new(e)))?;
        self.charge(factor_work(law, spec)?)?;
        self.data.probability(spec, assignment, ctx)
    }
    fn support(
        &self,
        vars: &[VariableId],
        ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        self.charge(1)?;
        self.data.support(vars, ctx)
    }
    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.charge(1)?;
        self.data.outcome(var, assignment, ctx)
    }
    fn zero_denominator(
        &self,
        arena: &CausalExprArena,
        expression: ExprId,
        assignment: &Assignment,
    ) -> EvalError {
        EvalError::ExactRatioSupport {
            expression,
            assignment: Arc::from(assignment.entries()),
            bindings: arena.leaf_bindings(expression).into(),
        }
    }
    fn n_draws(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DiscreteAxis, DomainRef, ExactDiscreteLaw, InterventionAssignment};
    use antecedent_core::RegimeId;
    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }
    fn law() -> ExactTransportData {
        let axes = [0, 1].map(|i| DiscreteAxis {
            variable: v(i),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        });
        let law = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            axes,
            [0.0, 0.0, 0.3, 0.7],
            "s",
            LawTolerance::default(),
        )
        .unwrap();
        ExactTransportData::try_new([law], 100).unwrap()
    }
    fn conditional() -> (CausalExprArena, ExprId) {
        let mut arena = CausalExprArena::new();
        let variables = arena.intern_var_set([v(0), v(1)]);
        let conditioned_on = arena.empty_var_set();
        let intervention = arena.intern_intervention_set([]);
        let population = arena.intern_population("target");
        let joint = arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            population,
            domain: DomainRef::Observational,
            regime: Some(RegimeId::from_raw(0)),
        });
        let summed = arena.intern_var_set([v(1)]);
        let marginal = arena.intern(ExprNode::SumOut { variables: summed, expr: joint });
        let root = arena.intern(ExprNode::Ratio { numerator: joint, denominator: marginal });
        (arena, root)
    }
    #[test]
    fn ratio_failure_retains_original_expression_and_assignment() {
        let (arena, root) = conditional();
        let ctx = ExecutionContext::for_tests(0);
        let plan = ExactEvaluationPlan::compile(
            &arena,
            root,
            law(),
            [v(1)],
            Assignment::from_pairs([(v(0), Value::Int64(0))]),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap();
        let EvalError::ExactRatioSupport { expression, assignment, bindings } =
            plan.evaluate(&ctx).unwrap_err()
        else {
            panic!("located ratio failure");
        };
        assert_eq!(expression, root);
        assert!(assignment.contains(&(v(0), Value::Int64(0))));
        assert_eq!(bindings[0].population.as_ref(), "target");
        assert_eq!(bindings[0].regime, Some(RegimeId::from_raw(0)));
    }
    #[test]
    fn numeric_float_requests_resolve_integer_table_domains() {
        let (arena, root) = conditional();
        let ctx = ExecutionContext::for_tests(0);
        let plan = ExactEvaluationPlan::compile(
            &arena,
            root,
            law(),
            [v(1)],
            Assignment::from_pairs([(v(0), Value::Float64(1.0))]),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap();
        let result = plan.evaluate(&ctx).unwrap();
        assert!((result.probabilities[0] - 0.3).abs() < 1e-12);
        assert!((result.probabilities[1] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn signed_mean_uses_compensated_summation() {
        let distribution = ExactDistribution {
            outcomes: Arc::from([v(0)]),
            atoms: vec![
                Arc::from([Value::f64(-1e17)]),
                Arc::from([Value::f64(1.0)]),
                Arc::from([Value::f64(1e17)]),
            ]
            .into(),
            probabilities: Arc::from([0.25, 0.5, 0.25]),
            support: Arc::from([]),
        };
        assert!((distribution.mean(v(0)).unwrap() - 0.5).abs() < 1e-12);
        assert!(distribution.mean_difference(&distribution, v(0)).unwrap().abs() < 1e-12);
    }

    #[test]
    fn zero_weight_skips_only_checked_bounded_conditional_extension() {
        let (mut arena, ratio) = conditional();
        let ExprNode::Ratio { numerator, denominator } = arena.node(ratio).clone() else {
            unreachable!()
        };
        let list = arena.intern_list([ratio, denominator]);
        let product = arena.intern(ExprNode::Product(list));
        let variables = arena.intern_var_set([v(0)]);
        let root = arena.intern(ExprNode::SumOut { variables, expr: product });
        let ctx = ExecutionContext::for_tests(0);
        let result = ExactEvaluationPlan::compile(
            &arena,
            root,
            law(),
            [v(1)],
            Assignment::new(),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap()
        .evaluate(&ctx)
        .unwrap();
        assert_eq!(result.probabilities.as_ref(), &[0.3, 0.7]);
        assert!(result.support.iter().any(|record| record.status == "checked_zero_mass_extension"));
        let arbitrary = arena.intern(ExprNode::Ratio { numerator, denominator: numerator });
        let list = arena.intern_list([arbitrary, denominator]);
        let product = arena.intern(ExprNode::Product(list));
        let root = arena.intern(ExprNode::SumOut { variables, expr: product });
        assert!(
            ExactEvaluationPlan::compile(
                &arena,
                root,
                law(),
                [v(1)],
                Assignment::new(),
                ExactEvaluationLimits::default(),
                LawTolerance::default(),
                &ctx
            )
            .unwrap()
            .evaluate(&ctx)
            .is_err()
        );
    }

    #[test]
    fn preflight_checks_share_one_budget_across_repeated_visits() {
        let (arena, root) = conditional();
        let data = law();
        let request = Assignment::from_pairs([(v(0), Value::Int64(1))]);
        let ctx = ExecutionContext::for_tests(0);
        let mut preflight = Preflight {
            arena: &arena,
            data: &data,
            request: &request,
            limits: ExactEvaluationLimits::default(),
            ctx: &ctx,
            bound: BTreeSet::from([v(1)]),
            intermediate_bytes: Cell::new(0),
            remaining_checks: Cell::new(6),
        };
        preflight.visit(root, 0).unwrap();
        assert!(matches!(
            preflight.visit(root, 0),
            Err(EvalError::ProviderKind("exact preflight budget exceeded"))
        ));
    }

    #[test]
    fn cancellation_memory_and_depth_do_not_return_partial_values() {
        let (arena, root) = conditional();
        let mut ctx = ExecutionContext::for_tests(0);
        let request = Assignment::from_pairs([(v(0), Value::Int64(1))]);
        let compile = |ctx: &ExecutionContext, limits| {
            ExactEvaluationPlan::compile(
                &arena,
                root,
                law(),
                [v(1)],
                request.clone(),
                limits,
                LawTolerance::default(),
                ctx,
            )
        };
        assert!(
            compile(&ctx, ExactEvaluationLimits { depth: 1, ..ExactEvaluationLimits::default() })
                .is_err()
        );
        let plan = compile(&ctx, ExactEvaluationLimits::default()).unwrap();
        ctx.memory.hard_limit_bytes = Some(1);
        assert!(plan.evaluate(&ctx).is_err());
        ctx.memory.hard_limit_bytes = None;
        ctx.cancellation.cancel();
        assert!(plan.evaluate(&ctx).is_err());
        assert!(compile(&ctx, ExactEvaluationLimits::default()).is_err());
    }

    /// Under the old NaN sentinel, `leaf_cost` expanded `do(T=NaN)` over the
    /// domain of `T` whenever `T` was bound by an enclosing sum. Concrete NaN
    /// must stay a single (invalid) world and must not cover `do(T=0)`/`do(T=1)`.
    #[test]
    fn nan_intervention_does_not_expand_exact_treatment_domain() {
        let t = v(0);
        let y = v(1);
        let regime = RegimeId::from_raw(0);
        let axis_y =
            DiscreteAxis { variable: y, values: Arc::from([Value::Int64(0), Value::Int64(1)]) };
        // E[Y|do(0)] = 0.2, E[Y|do(1)] = 0.8 as P(Y=1|do(t)).
        let law0 = ExactDiscreteLaw::try_new(
            "target",
            regime,
            [InterventionAssignment { variable: t, value: Value::Int64(0) }],
            [axis_y.clone()],
            [0.8, 0.2],
            "do0",
            LawTolerance::default(),
        )
        .unwrap();
        let law1 = ExactDiscreteLaw::try_new(
            "target",
            regime,
            [InterventionAssignment { variable: t, value: Value::Int64(1) }],
            [axis_y],
            [0.2, 0.8],
            "do1",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law0, law1], 100).unwrap();

        let mut arena = CausalExprArena::new();
        let ys = arena.intern_var_set([y]);
        let empty = arena.empty_var_set();
        let population = arena.intern_population("target");
        let do_sym = arena.intern_intervention_set([t]);
        let do_nan = arena.intern_intervention_assignments([InterventionAssignment {
            variable: t,
            value: Value::f64(f64::NAN),
        }]);
        assert!(!arena.intervention_assignments(do_nan)[0].is_symbolic());
        let sym = arena.intern(ExprNode::Distribution {
            variables: ys,
            conditioned_on: empty,
            intervention: do_sym,
            population,
            domain: DomainRef::Interventional,
            regime: Some(regime),
        });
        let nan = arena.intern(ExprNode::Distribution {
            variables: ys,
            conditioned_on: empty,
            intervention: do_nan,
            population,
            domain: DomainRef::Interventional,
            regime: Some(regime),
        });

        let ctx = ExecutionContext::for_tests(0);
        let request = Assignment::new();
        let limits = ExactEvaluationLimits::default();
        // Enclosing SumOut binds T: symbolic covers the full domain of T.
        let mut sym_preflight = Preflight {
            arena: &arena,
            data: &data,
            request: &request,
            limits,
            ctx: &ctx,
            bound: BTreeSet::from([t]),
            intermediate_bytes: Cell::new(0),
            remaining_checks: Cell::new(limits.operations),
        };
        assert!(
            sym_preflight.visit(sym, 0).is_ok(),
            "symbolic do(T) must expand over the treatment domain"
        );

        let mut nan_preflight = Preflight {
            arena: &arena,
            data: &data,
            request: &request,
            limits,
            ctx: &ctx,
            bound: BTreeSet::from([t]),
            intermediate_bytes: Cell::new(0),
            remaining_checks: Cell::new(limits.operations),
        };
        let nan_err = nan_preflight.visit(nan, 0);
        assert!(
            nan_err.is_err(),
            "concrete NaN must not expand over do(T) support; got {nan_err:?}"
        );
    }

    /// Three binary axes `(v0, v1, v2)`, last fastest. `P(v0=0) = 0` is a structural zero.
    fn three_axis_data(origin_empirical: bool, counts: Option<Vec<u64>>) -> ExactTransportData {
        let axes = [0, 1, 2].map(|i| DiscreteAxis {
            variable: v(i),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        });
        let probabilities = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4];
        let mut law = if origin_empirical {
            ExactDiscreteLaw::try_empirical(
                "target",
                RegimeId::from_raw(0),
                [],
                axes,
                probabilities,
                "s",
                LawTolerance::default(),
            )
        } else {
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                axes,
                probabilities,
                "s",
                LawTolerance::default(),
            )
        }
        .unwrap();
        if let Some(counts) = counts {
            law = law.with_learned_support(counts).unwrap();
        }
        ExactTransportData::try_new([law], 100).unwrap()
    }

    /// `P(v0) · Σ_{v2} k / Σ_{v1,v2} k` = `P(v0) · P(v1 | v0)` over the joint leaf `k`, with the
    /// conditional written the way a district factor is: marginals of one joint whose
    /// numerator is itself a marginal (not the joint).
    fn marginal_ratio_product() -> (CausalExprArena, ExprId) {
        let mut arena = CausalExprArena::new();
        let all = arena.intern_var_set([v(0), v(1), v(2)]);
        let empty = arena.empty_var_set();
        let none = arena.intern_intervention_set([]);
        let population = arena.intern_population("target");
        let joint = arena.intern(ExprNode::Distribution {
            variables: all,
            conditioned_on: empty,
            intervention: none,
            population,
            domain: DomainRef::Observational,
            regime: Some(RegimeId::from_raw(0)),
        });
        let s2 = arena.intern_var_set([v(2)]);
        let s12 = arena.intern_var_set([v(1), v(2)]);
        let numerator = arena.intern(ExprNode::SumOut { variables: s2, expr: joint });
        let denominator = arena.intern(ExprNode::SumOut { variables: s12, expr: joint });
        let ratio = arena.intern(ExprNode::Ratio { numerator, denominator });
        let list = arena.intern_list([denominator, ratio]);
        let root = arena.intern(ExprNode::Product(list));
        (arena, root)
    }

    fn evaluate_three_axis(
        arena: &CausalExprArena,
        root: ExprId,
        data: ExactTransportData,
    ) -> Result<ExactDistribution, EvalError> {
        let ctx = ExecutionContext::for_tests(0);
        ExactEvaluationPlan::compile(
            arena,
            root,
            data,
            [v(0), v(1)],
            Assignment::new(),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )?
        .evaluate(&ctx)
    }

    #[test]
    fn zero_weight_extension_covers_conditionals_over_partial_marginals() {
        // At v0 = 0 the conditional is 0/0, multiplied by the explicit zero P(v0=0). The
        // conditional's numerator is Σ_{v2} k, not k, so only the semantic marginal-ratio
        // shape (not `denominator == SumOut(numerator)`) certifies the extension.
        let (arena, root) = marginal_ratio_product();
        let result = evaluate_three_axis(&arena, root, three_axis_data(false, None)).unwrap();
        // Atoms (v0, v1): P(0,·) = 0; P(1,0) = 0.1 + 0.2, P(1,1) = 0.3 + 0.4.
        let expected = [0.0, 0.0, 0.3, 0.7];
        for (got, want) in result.probabilities.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{:?}", result.probabilities);
        }
        assert!(result.support.iter().any(|record| record.status == "checked_zero_mass_extension"));
    }

    #[test]
    fn partial_marginal_conditionals_keep_the_empirical_support_guard() {
        // The same expression over an empirical law that never observed v0 = 0: the zero is a
        // sampling zero, so the conditional must refuse rather than be extended as structural.
        let (arena, root) = marginal_ratio_product();
        let data = three_axis_data(true, None);
        let EvalError::ExactLaw(error) = evaluate_three_axis(&arena, root, data).unwrap_err()
        else {
            panic!("an unobserved conditioning stratum must be located");
        };
        assert_eq!(error.kind, "sampling_zero");
        // ... and likewise when a learned law has positive fitted mass there.
        let learned = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            [0, 1, 2].map(|i| DiscreteAxis {
                variable: v(i),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            }),
            [0.125; 8],
            "s",
            LawTolerance::default(),
        )
        .unwrap()
        .with_learned_support(vec![0, 0, 0, 0, 1, 2, 3, 4])
        .unwrap();
        let data = ExactTransportData::try_new([learned], 100).unwrap();
        let EvalError::ExactLaw(error) = evaluate_three_axis(&arena, root, data).unwrap_err()
        else {
            panic!("fitted mass must not certify support");
        };
        assert_eq!(error.kind, "sampling_zero");
    }

    #[test]
    fn expression_operations_and_provider_rows_share_one_budget() {
        // Evaluate the same plan once against a provider that charges row inspections to the
        // session's budget and once against one that charges nothing. The difference is what the
        // provider drew, which is only visible in the session's cell if there is one budget.
        let (arena, root) = conditional();
        let plan = arena.compile(root).unwrap();
        let ctx = ExecutionContext::for_tests(0);
        let consumed = |charged: bool| {
            let data = law();
            let budget = Cell::new(1_000_000);
            let mut assignment =
                Assignment::from_pairs([(v(0), Value::Int64(1)), (v(1), Value::Int64(0))]);
            let bounded = BoundedProvider { data: &data, ctx: &ctx, remaining: &budget };
            let provider: &dyn DistributionProvider = if charged { &bounded } else { &data };
            let mut session =
                crate::exact_engine::ExactSession::new(&arena, &plan, provider, &ctx, &budget);
            let value = session.evaluate(&mut assignment).unwrap();
            assert!((value - 0.3).abs() < 1e-12, "P(v1=0 | v0=1) = 0.3/1.0, got {value}");
            1_000_000 - budget.get()
        };
        let expression_only = consumed(false);
        let with_rows = consumed(true);
        assert!(expression_only > 0);
        assert!(with_rows > expression_only, "{with_rows} vs {expression_only}");
    }

    #[test]
    fn deep_expression_is_refused_before_unbounded_recursion() {
        // A chain of kernels far deeper than the depth budget. The depth-limited preflight
        // must reject it before free-variable analysis or compilation recurse over it.
        let mut arena = CausalExprArena::new();
        let vars = arena.intern_var_set([v(1)]);
        let empty = arena.empty_var_set();
        let none = arena.intern_intervention_set([]);
        let population = arena.intern_population("target");
        let mut node = arena.intern(ExprNode::Distribution {
            variables: vars,
            conditioned_on: empty,
            intervention: none,
            population,
            domain: DomainRef::Observational,
            regime: Some(RegimeId::from_raw(0)),
        });
        for i in 0..20_000u32 {
            let bound = arena.intern_var_set([v(1_000 + i)]);
            node = arena.intern(ExprNode::Kernel {
                body: node,
                bound,
                population,
                regime: Some(RegimeId::from_raw(0)),
            });
        }
        let ctx = ExecutionContext::for_tests(0);
        let refused = ExactEvaluationPlan::compile(
            &arena,
            node,
            law(),
            [v(1)],
            Assignment::new(),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap_err();
        assert_eq!(refused, EvalError::ProviderKind("exact recursion budget exceeded"));
    }
}

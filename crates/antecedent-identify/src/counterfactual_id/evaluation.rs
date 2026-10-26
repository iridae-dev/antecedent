//! Evaluate an identified counterfactual functional on one joint law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Assignment, BTreeSet, CausalExprArena, CfTerm, CfValue, CounterfactualFunctional,
    CounterfactualIdDerivation, CounterfactualIdProblem, CounterfactualIdRefusal,
    CounterfactualIdShape, DomainRef, EmpiricalTableProvider, EvalContext, ExactDiscreteLaw,
    ExecutionContext, ExprId, ExprNode, FactorSpec, HashMap, SearchReceipt, SearchStop, Value,
    VariableId,
};

/// The exact point of an identified derivation on one joint law.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualIdPoint {
    /// `P(given)`, the conditioning probability (`P(X = x')`).
    pub conditioning_probability: f64,
    /// `(outcome level, P(Y_x = level, X = x'))` in declared level order; empty
    /// for an exact zero.
    pub numerators: Vec<(f64, f64)>,
    /// The requested probability `P(event | given)`.
    pub probability: f64,
    /// `E[Y_x | X = x']`; `None` for an exact zero.
    pub counterfactual_mean: Option<f64>,
    /// `E[Y | X = x']`; `None` for an exact zero.
    pub observed_mean: Option<f64>,
    /// `E[Y_x | X = x'] - E[Y | X = x']`; `None` for an exact zero.
    pub effect: Option<f64>,
}

/// A joint law laid out in the problem's variable and level order.
#[derive(Clone, Debug)]
pub(crate) struct DenseJoint {
    /// Levels per variable (values).
    pub levels: Vec<Vec<f64>>,
    /// Row-major probabilities, last variable fastest.
    pub probabilities: Vec<f64>,
}

impl DenseJoint {
    /// Marginal table over `vars` (row-major in the given order, last fastest).
    #[must_use]
    pub(crate) fn marginal(&self, vars: &[usize]) -> Vec<f64> {
        let cards: Vec<usize> = self.levels.iter().map(Vec::len).collect();
        let size: usize = vars.iter().map(|&v| cards[v]).product();
        let mut out = vec![0.0; size];
        let mut index = vec![0usize; cards.len()];
        for p in &self.probabilities {
            let mut flat = 0usize;
            for &v in vars {
                flat = flat * cards[v] + index[v];
            }
            out[flat] += p;
            for k in (0..cards.len()).rev() {
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
        out
    }

    /// Mass of the cells where `fixed` variables take the given level indices.
    #[must_use]
    pub(crate) fn mass(&self, fixed: &[(usize, usize)]) -> f64 {
        let cards: Vec<usize> = self.levels.iter().map(Vec::len).collect();
        let mut total = 0.0;
        let mut index = vec![0usize; cards.len()];
        for p in &self.probabilities {
            if fixed.iter().all(|&(v, l)| index[v] == l) {
                total += p;
            }
            for k in (0..cards.len()).rev() {
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
        total
    }
}

/// Lay `law` out in `problem`'s variable and level order.
///
/// # Errors
///
/// `counterfactual_id.invalid_query` when the law's axes are not exactly the
/// problem's variables with the problem's levels (in any order).
pub(crate) fn dense_joint(
    problem: &CounterfactualIdProblem,
    law: &ExactDiscreteLaw,
) -> Result<DenseJoint, CounterfactualIdRefusal> {
    let n = problem.node_count();
    let axes = law.axes();
    if axes.len() != n || !law.interventions().is_empty() {
        return Err(CounterfactualIdRefusal::invalid_query(format!(
            "the law has {} axes (and {} interventions) for an observational problem of {n} \
             variables",
            axes.len(),
            law.interventions().len()
        )));
    }
    // axis_of[v] = axis index; level_map[v][law level index] = problem level index.
    let mut axis_of = vec![usize::MAX; n];
    let mut level_map = vec![Vec::new(); n];
    for (a, axis) in axes.iter().enumerate() {
        let v = variable_slot(axis.variable.raw());
        if v >= n || axis_of[v] != usize::MAX {
            return Err(CounterfactualIdRefusal::invalid_query(
                "the law's axes are not the problem's variables",
            ));
        }
        axis_of[v] = a;
        let mut map = Vec::with_capacity(axis.values.len());
        for value in axis.values.iter() {
            let bits = value.as_f64().map(|x| (x + 0.0).to_bits());
            let Some(index) =
                bits.and_then(|b| problem.level_index(u32::try_from(v).unwrap_or(u32::MAX), b))
            else {
                return Err(CounterfactualIdRefusal::invalid_query(format!(
                    "the law's levels of variable {v} are not the prepared levels"
                )));
            };
            map.push(index);
        }
        if map.len() != problem.levels[v].len() {
            return Err(CounterfactualIdRefusal::invalid_query(format!(
                "the law's levels of variable {v} are not the prepared levels"
            )));
        }
        level_map[v] = map;
    }
    let cards: Vec<usize> = (0..n).map(|v| problem.levels[v].len()).collect();
    let mut probabilities = vec![0.0; cards.iter().product()];
    let law_cards: Vec<usize> = axes.iter().map(|a| a.values.len()).collect();
    let mut law_index = vec![0usize; axes.len()];
    for &p in law.probabilities() {
        let mut flat = 0usize;
        for v in 0..n {
            flat = flat * cards[v] + level_map[v][law_index[axis_of[v]]];
        }
        probabilities[flat] = p;
        for k in (0..law_cards.len()).rev() {
            law_index[k] += 1;
            if law_index[k] < law_cards[k] {
                break;
            }
            law_index[k] = 0;
        }
    }
    Ok(DenseJoint { levels: (0..n).map(|v| problem.levels(v)).collect(), probabilities })
}

/// Evaluate `derivation` on `law`: each ID term through the existing compiled
/// evaluator over factor tables built from the joint, composed by the
/// functional's sums and products.
///
/// # Errors
///
/// `counterfactual_id.invalid_query` for a law that does not match the problem
/// or a zero-probability conditioning event, `counterfactual_id.positivity_violation`
/// when a conditional the functional evaluates is on a zero-mass event.
#[doc(hidden)]
pub fn evaluate_counterfactual_functional(
    problem: &CounterfactualIdProblem,
    derivation: &CounterfactualIdDerivation,
    law: &ExactDiscreteLaw,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdPoint, CounterfactualIdRefusal> {
    let joint = dense_joint(problem, law)?;
    let given: Vec<(usize, usize)> = derivation
        .given
        .iter()
        .map(|&(v, bits)| {
            problem
                .level_index(v, bits)
                .map(|l| (variable_slot(v), l))
                .ok_or_else(|| CounterfactualIdRefusal::invalid_query("an unknown given level"))
        })
        .collect::<Result<_, _>>()?;
    let conditioning = joint.mass(&given);
    if conditioning <= 0.0 {
        return Err(CounterfactualIdRefusal::invalid_query(
            "the conditioning event has probability zero in the supplied law",
        ));
    }
    let CounterfactualIdShape::EffectOnTreated { outcome, outcome_level, .. } = derivation.shape
    else {
        return Ok(CounterfactualIdPoint {
            conditioning_probability: conditioning,
            numerators: Vec::new(),
            probability: 0.0,
            counterfactual_mean: None,
            observed_mean: None,
            effect: None,
        });
    };
    let mut evaluator = TermEvaluator::new(&joint);
    let mut numerators = Vec::with_capacity(derivation.numerators.len());
    for (bits, functional) in &derivation.numerators {
        let mut symbols = HashMap::new();
        let value = evaluator.functional(functional, &mut symbols, ctx)?;
        numerators.push((f64::from_bits(*bits), value));
    }
    // A cheap check of the derivation on this law: the numerators over every
    // level of Y are P(Y_x = y, X = x') for all y, so they sum to P(X = x').
    let total: f64 = numerators.iter().map(|(_, p)| p).sum();
    if !total.is_finite()
        || (total - conditioning).abs() > NUMERATOR_SUM_TOLERANCE * (1.0 + conditioning)
    {
        return Err(CounterfactualIdRefusal::invalid_query(format!(
            "the derivation failed its check on this law: the numerators sum to {total}, not \
             P(X = x') = {conditioning}"
        )));
    }
    let probability = numerators
        .iter()
        .find(|(level, _)| level.to_bits() == outcome_level.to_bits())
        .map_or(f64::NAN, |(_, p)| p / conditioning);
    let counterfactual_mean =
        numerators.iter().map(|(level, p)| level * p).sum::<f64>() / conditioning;
    let observed_mean = (0..joint.levels[variable_slot(outcome)].len())
        .map(|l| {
            let mut fixed = given.clone();
            fixed.push((variable_slot(outcome), l));
            joint.levels[variable_slot(outcome)][l] * joint.mass(&fixed)
        })
        .sum::<f64>()
        / conditioning;
    if !probability.is_finite() || !counterfactual_mean.is_finite() {
        return Err(CounterfactualIdRefusal::positivity(
            "the functional is not finite on this law",
        ));
    }
    Ok(CounterfactualIdPoint {
        conditioning_probability: conditioning,
        numerators,
        probability,
        counterfactual_mean: Some(counterfactual_mean),
        observed_mean: Some(observed_mean),
        effect: Some(counterfactual_mean - observed_mean),
    })
}

/// Evaluates district terms through the compiled evaluator, one provider per
/// term and null-event extension built from the joint.
pub(super) struct TermEvaluator<'j> {
    joint: &'j DenseJoint,
    /// Per term: the factor tables under every null-event extension (one when
    /// the law has no zero-mass conditioning cell for the term), and the plan.
    compiled: HashMap<usize, (Vec<EmpiricalTableProvider>, antecedent_expr::CompiledEvaluator)>,
    polls: usize,
}

/// How a conditional on a zero-mass conditioning event is extended. The
/// functional is accepted only where its value is the same under every
/// extension tried (the uniform one and, for each `k` below the largest
/// cardinality, all mass on level `k` of every conditioned variable); otherwise
/// the law violates positivity for this functional. Every level is tried, so a
/// value that depends on the extension of one undefined conditional of one
/// variable is caught whatever its cardinality (the value is affine in that
/// extension, and constant on every point mass only if constant). Agreement is
/// a check, not a proof: the same extension is applied to every zero-mass
/// cell, and a joint conditional of several variables gets only the diagonal
/// point masses, so a dependence that only a different point mass per cell or
/// an off-diagonal configuration reveals is not detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NullExtension {
    /// Every value of the conditioned variables equally likely.
    Uniform,
    /// All mass on level `k` of every conditioned variable (its last level
    /// when it has `k` or fewer).
    Level(usize),
}

impl NullExtension {
    /// The extensions tried on a law whose variables have at most
    /// `max_levels` levels.
    fn all(max_levels: usize) -> Vec<Self> {
        std::iter::once(Self::Uniform).chain((0..max_levels.max(1)).map(Self::Level)).collect()
    }
}

/// Relative tolerance of the agreement between the null-event extensions.
const EXTENSION_TOLERANCE: f64 = 1e-12;
/// Relative tolerance of the check that the numerators sum to `P(X = x')`.
const NUMERATOR_SUM_TOLERANCE: f64 = 1e-9;
/// Largest negative rounding a consistency complement may show before the law
/// is refused as incompatible with the graph.
const COMPLEMENT_TOLERANCE: f64 = 1e-12;

/// Position of a declared level within one variable's level list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LevelIx(usize);

/// Identifier of a summation symbol bound by an enclosing sum.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SymbolId(u32);

/// Level positions currently bound to the summation symbols in scope.
type Bindings = HashMap<SymbolId, LevelIx>;

/// Index of a variable in the joint law's variable and level tables.
const fn variable_slot(variable: u32) -> usize {
    variable as usize
}

impl<'j> TermEvaluator<'j> {
    pub(super) fn new(joint: &'j DenseJoint) -> Self {
        Self { joint, compiled: HashMap::new(), polls: 0 }
    }

    fn functional(
        &mut self,
        functional: &CounterfactualFunctional,
        symbols: &mut Bindings,
        ctx: &ExecutionContext,
    ) -> Result<f64, CounterfactualIdRefusal> {
        match functional {
            CounterfactualFunctional::One => Ok(1.0),
            CounterfactualFunctional::Zero => Ok(0.0),
            CounterfactualFunctional::Term(term) => self.term(term, symbols, ctx),
            CounterfactualFunctional::ConsistencyComplement { interventional, observed } => {
                let interventional = self.term(interventional, symbols, ctx)?;
                let mut fixed = Vec::with_capacity(observed.len());
                for &(variable, value) in observed {
                    fixed.push((
                        variable_slot(variable),
                        self.level_index(variable, value, symbols)?.0,
                    ));
                }
                let value = interventional - self.joint.mass(&fixed);
                if value < -COMPLEMENT_TOLERANCE * (1.0 + interventional.abs()) {
                    return Err(CounterfactualIdRefusal::invalid_query(format!(
                        "the law is not compatible with the graph: the consistency complement \
                         P_x(y) - P(y, x) is {value}"
                    )));
                }
                Ok(value.max(0.0))
            }
            CounterfactualFunctional::Sum { symbols: summed, factors } => {
                let cards: Vec<usize> =
                    summed.iter().map(|s| self.joint.levels[variable_slot(s.variable)].len()).collect();
                let mut index = vec![0usize; summed.len()];
                let mut total = antecedent_kernels::numeric::CompensatedSum::default();
                loop {
                    for (s, &l) in summed.iter().zip(&index) {
                        symbols.insert(SymbolId(s.id), LevelIx(l));
                    }
                    // A factor that is exactly zero makes the term a structural
                    // zero and the remaining factors are not evaluated. A
                    // product that merely underflows keeps multiplying, so a
                    // later non-finite factor is reported instead of skipped.
                    let mut product = 1.0;
                    for factor in factors {
                        let value = self.functional(factor, symbols, ctx)?;
                        if value == 0.0 {
                            product = 0.0;
                            break;
                        }
                        product *= value;
                    }
                    if !product.is_finite() {
                        return Err(CounterfactualIdRefusal::invalid_query(format!(
                            "a product of probability functionals is not finite ({product})"
                        )));
                    }
                    total.add(product);
                    let mut k = summed.len();
                    loop {
                        if k == 0 {
                            for s in summed {
                                symbols.remove(&SymbolId(s.id));
                            }
                            return Ok(total.total());
                        }
                        k -= 1;
                        index[k] += 1;
                        if index[k] < cards[k] {
                            break;
                        }
                        index[k] = 0;
                    }
                }
            }
        }
    }

    fn level_index(
        &self,
        variable: u32,
        value: CfValue,
        symbols: &Bindings,
    ) -> Result<LevelIx, CounterfactualIdRefusal> {
        let levels = &self.joint.levels[variable_slot(variable)];
        match value {
            CfValue::Level(bits) => {
                levels.iter().position(|l| l.to_bits() == bits).map(LevelIx)
            }
            CfValue::Symbol(id) => symbols.get(&SymbolId(id)).copied(),
        }
        .ok_or_else(|| {
            CounterfactualIdRefusal::invalid_query("a functional level is not a level of the law")
        })
    }

    fn level(
        &self,
        variable: u32,
        value: CfValue,
        symbols: &Bindings,
    ) -> Result<f64, CounterfactualIdRefusal> {
        let index = self.level_index(variable, value, symbols)?;
        Ok(self.joint.levels[variable_slot(variable)][index.0])
    }

    pub(super) fn term(
        &mut self,
        term: &CfTerm,
        symbols: &Bindings,
        ctx: &ExecutionContext,
    ) -> Result<f64, CounterfactualIdRefusal> {
        self.polls += 1;
        if self.polls % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(CounterfactualIdRefusal::budget(SearchReceipt {
                stop: SearchStop::Cancelled,
                operations_limit: 0,
                depth_limit: 0,
                memory_limit_bytes: None,
                operations_consumed: None,
                depth_reached: None,
                explored: vec!["evaluation".into()],
                unevaluated: Vec::new(),
            }));
        }
        let key = std::ptr::from_ref(term) as usize;
        let joint = self.joint;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.compiled.entry(key) {
            let (uniform, touched_null) =
                factor_provider(joint, &term.arena, term.expression, NullExtension::Uniform);
            let mut providers = vec![uniform];
            if touched_null {
                let most = joint.levels.iter().map(Vec::len).max().unwrap_or(1);
                providers.extend(
                    NullExtension::all(most)
                        .into_iter()
                        .skip(1)
                        .map(|fill| factor_provider(joint, &term.arena, term.expression, fill).0),
                );
            }
            let compiled = term.arena.compile(term.expression).map_err(|e| {
                CounterfactualIdRefusal::invalid_query(format!("the term does not compile: {e}"))
            })?;
            slot.insert((providers, compiled));
        }
        let mut env = Vec::new();
        for &(v, value) in term.intervention.iter().chain(&term.event) {
            env.push((VariableId::from_raw(v), Value::f64(self.level(v, value, symbols)?)));
        }
        let bound: BTreeSet<VariableId> = env.iter().map(|(v, _)| *v).collect();
        let extra: Vec<VariableId> = term
            .arena
            .free_variables(term.expression)
            .into_iter()
            .filter(|v| !bound.contains(v))
            .collect();
        let (providers, compiled) = &self.compiled[&key];
        // A variable the functional leaves free beyond the treatments and the
        // event (ID line 3 then 7) does not change its value wherever the
        // conditionals exist: the first level where the term is defined is used.
        let cards: Vec<usize> =
            extra.iter().map(|v| self.joint.levels[variable_slot(v.raw())].len()).collect();
        let mut index = vec![0usize; extra.len()];
        let mut last_error = String::new();
        loop {
            let mut full = env.clone();
            for (v, &l) in extra.iter().zip(&index) {
                full.push((*v, Value::f64(self.joint.levels[variable_slot(v.raw())][l])));
            }
            let assignment = Assignment::from_pairs(full);
            let value = |provider: &EmpiricalTableProvider| {
                compiled.evaluate_with(&term.arena, provider, &EvalContext::default(), &assignment)
            };
            let values: Result<Vec<f64>, _> = providers.iter().map(value).collect();
            match values {
                Ok(values) => {
                    let first = values[0];
                    if values.iter().all(|&b| {
                        b.is_finite()
                            && (first - b).abs()
                                <= EXTENSION_TOLERANCE * (1.0 + first.abs().max(b.abs()))
                    }) {
                        return Ok(first);
                    }
                    "the value depends on how a conditional on a zero-mass event is extended"
                        .clone_into(&mut last_error);
                }
                Err(e) => last_error = e.to_string(),
            }
            let mut k = extra.len();
            loop {
                if k == 0 {
                    return Err(CounterfactualIdRefusal::positivity(format!(
                        "a district term is undefined on this law: {last_error}"
                    )));
                }
                k -= 1;
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
    }
}

/// Factor tables of every distribution the term's functional reads: the
/// conditional of the joint on every conditioning assignment with positive
/// mass, and `fill` on a zero-mass one. The flag says whether any zero-mass
/// conditioning assignment was met (otherwise every fill gives these tables).
pub(super) fn factor_provider(
    joint: &DenseJoint,
    arena: &CausalExprArena,
    root: ExprId,
    fill: NullExtension,
) -> (EmpiricalTableProvider, bool) {
    let mut touched_null = false;
    let mut provider = EmpiricalTableProvider::new();
    for (v, levels) in joint.levels.iter().enumerate() {
        provider.set_domain(
            VariableId::from_raw(
                u32::try_from(v).expect("counterfactual variable index fits in u32"),
            ),
            levels.iter().map(|x| Value::f64(*x)),
        );
    }
    let mut seen = BTreeSet::new();
    let mut stack = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, .. } => {
                let vars = arena.var_set(*variables).to_vec();
                let cond = arena.var_set(*conditioned_on).to_vec();
                if seen.insert((vars.clone(), cond.clone())) {
                    touched_null |= insert_conditional(&mut provider, joint, &vars, &cond, fill);
                }
            }
            ExprNode::Product(list) => stack.extend(arena.list(*list).iter().copied()),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => stack.push(*expr),
            ExprNode::Ratio { numerator, denominator } => {
                stack.push(*numerator);
                stack.push(*denominator);
            }
            ExprNode::Kernel { body, .. } => stack.push(*body),
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    (provider, touched_null)
}

fn insert_conditional(
    provider: &mut EmpiricalTableProvider,
    joint: &DenseJoint,
    vars: &[VariableId],
    cond: &[VariableId],
    fill: NullExtension,
) -> bool {
    let mut touched_null = false;
    let all: Vec<usize> = vars.iter().chain(cond).map(|v| variable_slot(v.raw())).collect();
    let cond_dense: Vec<usize> = cond.iter().map(|v| variable_slot(v.raw())).collect();
    let joint_table = joint.marginal(&all);
    let cond_table = joint.marginal(&cond_dense);
    let cond_size = cond_table.len();
    let spec = FactorSpec {
        variables: vars,
        conditioned_on: cond,
        intervention: &[],
        domain: DomainRef::Observational,
        population: "",
        regime: None,
    };
    let cards: Vec<usize> = all.iter().map(|&v| joint.levels[v].len()).collect();
    let mut index = vec![0usize; all.len()];
    for (flat, &numerator) in joint_table.iter().enumerate() {
        // The conditioning variables are the trailing axes of `all`.
        let denominator = cond_table[flat % cond_size.max(1)];
        let value = if denominator > 0.0 {
            numerator / denominator
        } else {
            touched_null = true;
            match fill {
                // One over the number of configurations of the conditioned variables.
                NullExtension::Uniform => 1.0 / (joint_table.len() / cond_size.max(1)) as f64,
                // All mass on level `k` (clamped to the last level) of each.
                NullExtension::Level(k) => {
                    if index[..vars.len()]
                        .iter()
                        .zip(&cards)
                        .all(|(&l, &card)| l == k.min(card - 1))
                    {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        };
        {
            let assignment = Assignment::from_pairs(all.iter().zip(&index).map(|(&v, &l)| {
                (
                    VariableId::from_raw(
                        u32::try_from(v).expect("counterfactual variable index fits in u32"),
                    ),
                    Value::f64(joint.levels[v][l]),
                )
            }));
            // The key is built from the spec's own variables, which are all bound.
            let _ = provider.insert_probability(&spec, &assignment, value);
        }
        for k in (0..all.len()).rev() {
            index[k] += 1;
            if index[k] < cards[k] {
                break;
            }
            index[k] = 0;
        }
    }
    touched_null
}

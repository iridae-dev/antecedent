//! The wire half of the counterfactual identification cell: laws on the wire,
//! artifact export, and consumption by replay plus the direct-sum
//! recomputation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CausalError;
use antecedent_core::{ExecutionContext, RegimeId, SearchLimits, Value, VariableId};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, ExprId, ExprNode, LawOrigin, LawTolerance};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_CONTRACT, COUNTERFACTUAL_ID_MAX_DEPTH, COUNTERFACTUAL_ID_MAX_OPERATIONS,
    COUNTERFACTUAL_ID_MEMORY_BYTES, CfSymbol, CfTerm, CfValue, CounterfactualFunctional,
    CounterfactualIdDerivation, CounterfactualIdProblem, CounterfactualIdRefusal,
    CounterfactualIdShape, decide_counterfactual_id,
};
use antecedent_io::counterfactual_id_artifact::{
    COUNTERFACTUAL_ID_ARTIFACT_FEATURE, COUNTERFACTUAL_ID_ARTIFACT_VERSION,
    COUNTERFACTUAL_ID_ESTIMAND, CounterfactualIdArtifactError, CounterfactualIdArtifactWire,
    CounterfactualIdAxisWire, CounterfactualIdLawWire, CounterfactualIdPointWire,
    CounterfactualIdQueryWire,
};
use std::collections::BTreeMap;

use super::{CounterfactualIdEffect, DIRECT_SUM_TOLERANCE, PreparedCounterfactualId, check_names};

/// The wire form of a law, refusing an origin the cell does not accept.
pub(super) fn law_wire(
    law: &ExactDiscreteLaw,
) -> Result<CounterfactualIdLawWire, CounterfactualIdRefusal> {
    let counts = match (law.origin(), law.empirical_counts()) {
        (LawOrigin::SuppliedExact, None) => None,
        (LawOrigin::EmpiricalPlugin, Some(counts)) => Some(counts.to_vec()),
        _ => {
            return Err(CounterfactualIdRefusal::invalid_query(
                "the law must be a supplied exact law or an empirical count table",
            ));
        }
    };
    let mut axes = Vec::with_capacity(law.axes().len());
    for axis in law.axes() {
        let levels = axis
            .values
            .iter()
            .map(|v| v.as_f64().map(|x| (x + 0.0).to_bits()))
            .collect::<Option<Vec<u64>>>()
            .ok_or_else(|| CounterfactualIdRefusal::invalid_query("law levels must be numeric"))?;
        axes.push(CounterfactualIdAxisWire { variable: axis.variable.raw(), levels });
    }
    Ok(CounterfactualIdLawWire {
        axes,
        probabilities: law.probabilities().iter().map(|p| p.to_bits()).collect(),
        counts,
        origin: law.origin().as_str().into(),
        population: law.population().into(),
        snapshot: law.snapshot_identity().into(),
    })
}

/// Rebuild the law a wire stores.
fn law_from_wire(
    wire: &CounterfactualIdLawWire,
) -> Result<ExactDiscreteLaw, CounterfactualIdArtifactError> {
    let malformed =
        |e: antecedent_expr::ExactLawError| CounterfactualIdArtifactError::Malformed(e.to_string());
    let axes: Vec<DiscreteAxis> = wire
        .axes
        .iter()
        .map(|a| DiscreteAxis {
            variable: VariableId::from_raw(a.variable),
            values: a
                .levels
                .iter()
                .map(|b| Value::f64(f64::from_bits(*b)))
                .collect::<Vec<_>>()
                .into(),
        })
        .collect();
    let probabilities: Vec<f64> = wire.probabilities.iter().map(|b| f64::from_bits(*b)).collect();
    let tolerance = LawTolerance::default();
    match (wire.origin.as_str(), &wire.counts) {
        ("supplied_exact", None) => ExactDiscreteLaw::try_new(
            wire.population.as_str(),
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            wire.snapshot.as_str(),
            tolerance,
        )
        .map_err(malformed),
        ("empirical_plugin", Some(counts)) => ExactDiscreteLaw::try_empirical(
            wire.population.as_str(),
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            wire.snapshot.as_str(),
            tolerance,
        )
        .and_then(|law| law.with_empirical_counts(counts.clone()))
        .map_err(malformed),
        _ => Err(CounterfactualIdArtifactError::UnsupportedSemantics("law origin")),
    }
}

impl CounterfactualIdEffect {
    /// Export the replayable `counterfactual_id_admg_v1` artifact.
    ///
    /// # Errors
    ///
    /// [`CausalError::Resource`] when the artifact exceeds the format's bounds,
    /// or an encoding failure.
    pub fn export_artifact(&self) -> Result<Vec<u8>, CausalError> {
        self.artifact_wire().and_then(|wire| wire.export()).map_err(|error| match error {
            CounterfactualIdArtifactError::LimitsExceeded(what) => CausalError::Resource {
                message: format!(
                    "the counterfactual identification artifact does not carry {what}"
                ),
            },
            other => CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string())),
        })
    }

    fn artifact_wire(&self) -> Result<CounterfactualIdArtifactWire, CounterfactualIdArtifactError> {
        let problem = &self.prepared.problem;
        let bits = |x: Option<f64>| x.map(f64::to_bits);
        CounterfactualIdArtifactWire {
            version: COUNTERFACTUAL_ID_ARTIFACT_VERSION,
            required_features: vec![COUNTERFACTUAL_ID_ARTIFACT_FEATURE.into()],
            variable_names: self.prepared.names.clone(),
            levels: (0..problem.node_count())
                .map(|v| problem.levels(v).into_iter().map(f64::to_bits).collect())
                .collect(),
            directed: problem.directed().to_vec(),
            bidirected: problem.bidirected().to_vec(),
            query: CounterfactualIdQueryWire::from_query(&self.prepared.query),
            estimand: COUNTERFACTUAL_ID_ESTIMAND.into(),
            contract: COUNTERFACTUAL_ID_CONTRACT.into(),
            search: self.prepared.derivation.search,
            derivation: self.prepared.derivation.canonical_text(),
            counterfactual_graph: self.prepared.derivation.counterfactual_graph.clone(),
            law: self.law.clone(),
            point: CounterfactualIdPointWire {
                probability: self.probability.to_bits(),
                conditioning_probability: self.conditioning_probability.to_bits(),
                numerators: self
                    .numerators
                    .iter()
                    .map(|(l, p)| (l.to_bits(), p.to_bits()))
                    .collect(),
                counterfactual_mean: bits(self.counterfactual_mean),
                observed_mean: bits(self.observed_mean),
                effect: bits(self.effect),
            },
            data_digest: String::new(),
            premises_digest: String::new(),
        }
        .sealed()
    }
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct CounterfactualIdConsumeLimits {
    /// Largest search limits a stored derivation may have run under.
    pub search: SearchLimits,
    /// Largest search memory cap a stored derivation may have run under.
    pub search_memory_bytes: u64,
}

impl Default for CounterfactualIdConsumeLimits {
    fn default() -> Self {
        Self {
            search: SearchLimits {
                operations: COUNTERFACTUAL_ID_MAX_OPERATIONS,
                depth: COUNTERFACTUAL_ID_MAX_DEPTH,
            },
            search_memory_bytes: COUNTERFACTUAL_ID_MEMORY_BYTES,
        }
    }
}

/// Replay a `counterfactual_id_admg_v1` artifact and accept it only when its
/// digests, derivation, search accounting and point all replay and the
/// separate direct-sum evaluation agrees.
///
/// # Errors
///
/// A [`CounterfactualIdArtifactError`] naming the failed check; its
/// [`CounterfactualIdArtifactError::reason`] is the registered pair.
pub fn consume_counterfactual_id_artifact(
    bytes: &[u8],
    limits: CounterfactualIdConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    consume_inner(bytes, None, limits, ctx)
}

/// [`consume_counterfactual_id_artifact`] that also requires the artifact's law
/// to have the caller's data digest.
///
/// # Errors
///
/// As [`consume_counterfactual_id_artifact`], and `DataIdentityMismatch` for
/// an artifact computed on another law.
pub fn consume_counterfactual_id_artifact_for_data(
    bytes: &[u8],
    expected_data_digest: &str,
    limits: CounterfactualIdConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    consume_inner(bytes, Some(expected_data_digest), limits, ctx)
}

fn consume_inner(
    bytes: &[u8],
    expected_data_digest: Option<&str>,
    limits: CounterfactualIdConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    let wire = CounterfactualIdArtifactWire::decode(bytes)?;
    wire.check_bounds()?;
    // Stored limits above the consumer's refuse before any work: a limits
    // refusal the caller can retry, not a claim that the artifact is invalid.
    let exceeded = CounterfactualIdArtifactError::LimitsExceeded;
    if wire.search.operations_limit > limits.search.operations {
        return Err(exceeded("search operation limit"));
    }
    if wire.search.depth_limit > limits.search.depth {
        return Err(exceeded("search depth limit"));
    }
    let memory_cap = ctx
        .memory_limit_bytes()
        .map_or(limits.search_memory_bytes, |hard| hard.min(limits.search_memory_bytes));
    if wire.search.memory_limit_bytes > memory_cap {
        return Err(exceeded("search memory limit"));
    }
    wire.check_digests(expected_data_digest)?;
    if wire.estimand != COUNTERFACTUAL_ID_ESTIMAND {
        return Err(CounterfactualIdArtifactError::UnsupportedSemantics("estimand"));
    }
    if wire.contract != COUNTERFACTUAL_ID_CONTRACT {
        return Err(CounterfactualIdArtifactError::UnsupportedSemantics("contract"));
    }
    let levels: Vec<Vec<f64>> =
        wire.levels.iter().map(|l| l.iter().map(|b| f64::from_bits(*b)).collect()).collect();
    let problem = CounterfactualIdProblem::new(levels, &wire.directed, &wire.bidirected)
        .map_err(|r| CounterfactualIdArtifactError::Malformed(r.to_string()))?;
    if problem.directed() != wire.directed || problem.bidirected() != wire.bidirected {
        return Err(CounterfactualIdArtifactError::Malformed(
            "the stored graph is not canonical".into(),
        ));
    }
    check_names(&wire.variable_names, problem.node_count())
        .map_err(|r| CounterfactualIdArtifactError::Malformed(r.to_string()))?;
    let query = wire.query.to_query()?;
    let law = law_from_wire(&wire.law)?;
    // Re-derive under the producer's stored limits and memory cap.
    let derivation = decide_counterfactual_id(
        &problem,
        &query,
        SearchLimits { operations: wire.search.operations_limit, depth: wire.search.depth_limit },
        wire.search.memory_limit_bytes,
        ctx,
    )
    .map_err(|r| CounterfactualIdArtifactError::DerivationMismatch(r.to_string()))?;
    if derivation.canonical_text() != wire.derivation
        || derivation.search != wire.search
        || derivation.counterfactual_graph != wire.counterfactual_graph
    {
        return Err(CounterfactualIdArtifactError::DerivationMismatch(
            "the re-derived functional, search accounting or counterfactual graph differs".into(),
        ));
    }
    let prepared =
        PreparedCounterfactualId { problem, query, derivation, names: wire.variable_names.clone() };
    let mut effect = prepared.evaluate(&law, ctx).map_err(|e| {
        CounterfactualIdArtifactError::DerivationMismatch(format!("evaluation refused: {e}"))
    })?;
    let bits = |x: Option<f64>| x.map(f64::to_bits);
    let replayed = CounterfactualIdPointWire {
        probability: effect.probability.to_bits(),
        conditioning_probability: effect.conditioning_probability.to_bits(),
        numerators: effect.numerators.iter().map(|(l, p)| (l.to_bits(), p.to_bits())).collect(),
        counterfactual_mean: bits(effect.counterfactual_mean),
        observed_mean: bits(effect.observed_mean),
        effect: bits(effect.effect),
    };
    if replayed != wire.point {
        return Err(CounterfactualIdArtifactError::PointMismatch);
    }
    let direct = direct_sum_numerators(&prepared.derivation, &wire.law, &wire.levels)
        .ok_or(CounterfactualIdArtifactError::IndependentCheckMismatch)?;
    if direct.len() != effect.numerators.len()
        || direct.iter().zip(&effect.numerators).any(|(d, (_, p))| !agrees(*d, *p))
    {
        return Err(CounterfactualIdArtifactError::IndependentCheckMismatch);
    }
    effect.independently_verified = true;
    Ok(effect)
}

pub(super) fn agrees(a: f64, b: f64) -> bool {
    (a - b).abs() <= DIRECT_SUM_TOLERANCE * (1.0 + a.abs().max(b.abs()))
}

/// Independent evaluation of every level's numerator: walks the functional and
/// each term's expression with its own arithmetic, reading masses straight from
/// the stored law's cells. It shares no evaluation code with the compiled
/// evaluator or its factor tables, but it evaluates the same re-derived
/// functional. `None` when the value depends on how a conditional on a
/// zero-mass event is extended, or a consistency complement is negative.
pub(super) fn direct_sum_numerators(
    derivation: &CounterfactualIdDerivation,
    law: &CounterfactualIdLawWire,
    levels: &[Vec<u64>],
) -> Option<Vec<f64>> {
    if !matches!(derivation.shape, CounterfactualIdShape::EffectOnTreated { .. }) {
        return Some(Vec::new());
    }
    // A conditional on a zero-mass event is extended uniformly and by all mass on
    // each level `k` of its conditioned variables; the value must not depend on
    // the extension (it is multiplied by zero mass).
    let most = levels.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let fills: Vec<WireCells> = std::iter::once(Fill::Uniform)
        .chain((0..most).map(Fill::Level))
        .map(|fill| WireCells::new(law, levels, fill))
        .collect();
    let mut out = Vec::with_capacity(derivation.numerators.len());
    for (_, functional) in &derivation.numerators {
        let first = fills[0].functional(functional, &mut BTreeMap::new())?;
        for cells in &fills[1..] {
            if !agrees(first, cells.functional(functional, &mut BTreeMap::new())?) {
                return None;
            }
        }
        out.push(first);
    }
    Some(out)
}

/// How the direct-sum evaluator extends a conditional on a zero-mass event.
#[derive(Clone, Copy)]
enum Fill {
    /// Every configuration of the conditioned variables equally likely.
    Uniform,
    /// All mass on level `k` (clamped to the last level) of each conditioned variable.
    Level(usize),
}

/// The stored law's cells as `(variable -> level bits, probability)` rows.
struct WireCells {
    rows: Vec<(BTreeMap<u32, u64>, f64)>,
    levels: BTreeMap<u32, Vec<u64>>,
    /// How a conditional on a zero-mass event is extended.
    fill: Fill,
}

impl WireCells {
    /// `levels` is the declared level order per variable: sums and the choice
    /// of a free variable's level follow it, as the compiled evaluation does.
    fn new(law: &CounterfactualIdLawWire, levels: &[Vec<u64>], fill: Fill) -> Self {
        let mut rows = Vec::with_capacity(law.probabilities.len());
        let cards: Vec<usize> = law.axes.iter().map(|a| a.levels.len()).collect();
        let mut index = vec![0usize; cards.len()];
        for p in &law.probabilities {
            let row: BTreeMap<u32, u64> =
                law.axes.iter().zip(&index).map(|(a, &i)| (a.variable, a.levels[i])).collect();
            rows.push((row, f64::from_bits(*p)));
            for k in (0..cards.len()).rev() {
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
        let levels = levels
            .iter()
            .enumerate()
            .map(|(v, l)| (u32::try_from(v).unwrap_or(u32::MAX), l.clone()))
            .collect();
        Self { rows, levels, fill }
    }

    fn mass(&self, fixed: &BTreeMap<u32, u64>) -> f64 {
        self.rows
            .iter()
            .filter(|(row, _)| fixed.iter().all(|(v, bits)| row.get(v) == Some(bits)))
            .map(|(_, p)| p)
            .sum()
    }

    fn resolve(&self, value: CfValue, variable: u32, symbols: &BTreeMap<u32, u64>) -> Option<u64> {
        match value {
            CfValue::Level(bits) => Some(bits),
            CfValue::Symbol(id) => symbols.get(&id).copied(),
        }
        .filter(|bits| self.levels.get(&variable).is_some_and(|l| l.contains(bits)))
    }

    fn functional(
        &self,
        functional: &CounterfactualFunctional,
        symbols: &mut BTreeMap<u32, u64>,
    ) -> Option<f64> {
        match functional {
            CounterfactualFunctional::One => Some(1.0),
            CounterfactualFunctional::Zero => Some(0.0),
            CounterfactualFunctional::Term(term) => self.term(term, symbols),
            CounterfactualFunctional::ConsistencyComplement { interventional, observed } => {
                let interventional = self.term(interventional, symbols)?;
                let mut fixed = BTreeMap::new();
                for &(v, value) in observed {
                    fixed.insert(v, self.resolve(value, v, symbols)?);
                }
                let value = interventional - self.mass(&fixed);
                // A negative complement means a law incompatible with the graph.
                (value >= -1e-12 * (1.0 + interventional.abs())).then_some(value.max(0.0))
            }
            CounterfactualFunctional::Sum { symbols: summed, factors } => {
                self.sum(summed, factors, symbols)
            }
        }
    }

    /// `sum_{symbols} prod(factors)`: the symbols' levels are bound in declared
    /// order, innermost last.
    fn sum(
        &self,
        summed: &[CfSymbol],
        factors: &[CounterfactualFunctional],
        symbols: &mut BTreeMap<u32, u64>,
    ) -> Option<f64> {
        let Some((first, rest)) = summed.split_first() else {
            // A zero factor ends the product: a later factor on a null event is
            // not read (the compiled evaluation's convention).
            // The short-circuit is keyed on the factor itself being a null event, never on
            // the running product, which can also reach zero by underflow.
            let mut product = 1.0;
            for factor in factors {
                let value = self.functional(factor, symbols)?;
                if value == 0.0 {
                    return Some(0.0);
                }
                product *= value;
            }
            return Some(product);
        };
        let mut total = 0.0;
        for &bits in self.levels.get(&first.variable)? {
            symbols.insert(first.id, bits);
            total += self.sum(rest, factors, symbols)?;
        }
        symbols.remove(&first.id);
        Some(total)
    }

    fn term(&self, term: &CfTerm, symbols: &BTreeMap<u32, u64>) -> Option<f64> {
        let mut env = BTreeMap::new();
        for &(v, value) in term.intervention.iter().chain(&term.event) {
            env.insert(v, self.resolve(value, v, symbols)?);
        }
        // Variables the functional leaves free beyond the treatments and the event:
        // the first level (declared order of the stored law) where it is defined.
        let free: Vec<u32> = term
            .arena
            .free_variables(term.expression)
            .into_iter()
            .map(VariableId::raw)
            .filter(|v| !env.contains_key(v))
            .collect();
        self.first_defined(&term.arena, term.expression, &mut env, &free)
    }

    fn first_defined(
        &self,
        arena: &antecedent_expr::CausalExprArena,
        root: ExprId,
        env: &mut BTreeMap<u32, u64>,
        free: &[u32],
    ) -> Option<f64> {
        let Some((&first, rest)) = free.split_first() else {
            return self.expr(arena, root, env).filter(|x| x.is_finite());
        };
        for &bits in self.levels.get(&first)? {
            env.insert(first, bits);
            if let Some(value) = self.first_defined(arena, root, env, rest) {
                env.remove(&first);
                return Some(value);
            }
        }
        env.remove(&first);
        None
    }

    fn expr(
        &self,
        arena: &antecedent_expr::CausalExprArena,
        id: ExprId,
        env: &mut BTreeMap<u32, u64>,
    ) -> Option<f64> {
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, .. } => {
                let mut cond = BTreeMap::new();
                for v in arena.var_set(*conditioned_on) {
                    cond.insert(v.raw(), *env.get(&v.raw())?);
                }
                let mut joint = cond.clone();
                for v in arena.var_set(*variables) {
                    joint.insert(v.raw(), *env.get(&v.raw())?);
                }
                let denominator = self.mass(&cond);
                if denominator > 0.0 {
                    return Some(self.mass(&joint) / denominator);
                }
                let vars = arena.var_set(*variables);
                match self.fill {
                    Fill::Level(k) => {
                        let at_k = vars.iter().all(|v| {
                            self.levels.get(&v.raw()).and_then(|l| l.get(k.min(l.len() - 1)))
                                == env.get(&v.raw())
                        });
                        Some(if at_k { 1.0 } else { 0.0 })
                    }
                    Fill::Uniform => {
                        let configurations: usize = vars
                            .iter()
                            .map(|v| self.levels.get(&v.raw()).map_or(1, Vec::len))
                            .product();
                        Some(1.0 / configurations as f64)
                    }
                }
            }
            ExprNode::Product(list) => arena
                .list(*list)
                .iter()
                .try_fold(1.0, |acc, &c| Some(acc * self.expr(arena, c, env)?)),
            ExprNode::SumOut { variables, expr } => {
                let vars: Vec<u32> = arena.var_set(*variables).iter().map(|v| v.raw()).collect();
                let saved: Vec<Option<u64>> = vars.iter().map(|v| env.get(v).copied()).collect();
                let total = self.sum_over(arena, *expr, env, &vars);
                for (v, old) in vars.iter().zip(saved) {
                    match old {
                        Some(bits) => env.insert(*v, bits),
                        None => env.remove(v),
                    };
                }
                total
            }
            ExprNode::Ratio { numerator, denominator } => {
                let d = self.expr(arena, *denominator, env)?;
                (d != 0.0).then_some(())?;
                Some(self.expr(arena, *numerator, env)? / d)
            }
            _ => None,
        }
    }

    fn sum_over(
        &self,
        arena: &antecedent_expr::CausalExprArena,
        body: ExprId,
        env: &mut BTreeMap<u32, u64>,
        vars: &[u32],
    ) -> Option<f64> {
        let Some((&first, rest)) = vars.split_first() else {
            return self.expr(arena, body, env);
        };
        let mut total = 0.0;
        for &bits in self.levels.get(&first)? {
            env.insert(first, bits);
            total += self.sum_over(arena, body, env, rest)?;
        }
        Some(total)
    }
}

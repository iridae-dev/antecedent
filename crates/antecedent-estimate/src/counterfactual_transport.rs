//! Exact evaluation of a checked counterfactual transport derivation.
//!
//! The identifier derives, for a counterfactual query, a sum over free nodes of
//! a product of single-world factors `Q[S](s | parents)` of the target
//! population, each obtained from one supplied regime law by marginalization,
//! district extraction and (for a source) transfer. This module reads those laws
//! from supplied exact finite tables and computes the number.
//!
//! Evidence. Every cited regime is read from the exact law whose population and
//! regime id the derivation cites; the laws are first checked against the
//! derivation's own evidence catalog (the same provider/catalog contract the
//! other exact transport routes use), so a law of the wrong population or regime
//! is refused before any arithmetic. Counted or otherwise non-exact laws are
//! refused: sampled inference is a separate route.
//!
//! Support. A factor is the conditional of a regime law; where the regime law
//! gives the conditioning event zero mass, or no law was supplied for the
//! intervention level a factor needs, the factor is *undefined* and is never
//! replaced by a number. An undefined factor is skipped only inside a summand
//! that another, defined factor makes exactly zero; otherwise the evaluation
//! refuses and names the failure. A conditioning event of zero mass is a support
//! failure, never an extended value.
//!
//! Point only: exact laws carry no sampling uncertainty and no interval or
//! posterior is computed here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, RegimeId, VariableId};
use antecedent_expr::{
    CausalExprArena, DomainRef, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    ExprId, ExprNode, InterventionAssignment, LawOrigin, VarSetId,
};
use antecedent_identify::{
    CounterfactualFactor, CounterfactualFunctional, CounterfactualItem, CounterfactualItemSource,
    CounterfactualLevel, CounterfactualSum, CounterfactualTransportDerivation,
};

use crate::error::EstimationError;

/// A law the evaluation read, with the identities the derivation cites.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualTransportEvidence {
    /// Population of the law.
    pub population: Arc<str>,
    /// Catalog regime of the law.
    pub regime: RegimeId,
    /// `(intervened variable, level)` of the law's concrete intervention world.
    pub interventions: Vec<(VariableId, f64)>,
    /// Physical snapshot identity the provider supplied.
    pub snapshot_identity: String,
    /// Content identity (hex digest) of the supplied table.
    pub content_identity: String,
}

/// The exact value of a derivation on supplied finite laws.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualTransportPoint {
    /// `P*(event and conditioning)` of the (possibly reduced) query.
    pub numerator: f64,
    /// `P*(conditioning)` of the (possibly reduced) query; one when unconditional.
    pub denominator: f64,
    /// The requested probability `numerator / denominator`.
    pub probability: f64,
    /// The original conditioning event's mass when the derivation carries a
    /// positivity guard or its denominator is that mass.
    pub conditioning_mass: Option<f64>,
    /// Whether the original conditioning event was checked to have positive
    /// mass; `false` means the value holds conditional on that premise.
    pub conditioning_mass_checked: bool,
    /// Every law that was read, ascending by (population, regime, world).
    pub evidence: Vec<CounterfactualTransportEvidence>,
    /// Regime-law cells read.
    pub cells_read: u64,
}

const DETAIL_SUPPORT: &str = "counterfactual_transport.support_failure";
const DETAIL_ZERO_MASS: &str = "counterfactual_transport.zero_conditioning_mass";
const DETAIL_MISSING: &str = "counterfactual_transport.missing_law";
const DETAIL_INCOMPATIBLE: &str = "counterfactual_transport.evidence_incompatible";
const DETAIL_INVALID: &str = "counterfactual_transport.invalid_evidence";
const DETAIL_COUNTED: &str = "counterfactual_transport.sampled_inference_closed";
const DETAIL_BUDGET: &str = "counterfactual_transport.budget";

fn refused(code: &'static str, detail: &str, message: impl AsRef<str>) -> EstimationError {
    EstimationError::refused_detail(code, detail, message)
}

fn invalid(message: impl AsRef<str>) -> EstimationError {
    refused(antecedent_core::reason_code!("invalid_argument"), DETAIL_INVALID, message)
}

/// Why an evaluation stopped.
enum Failure {
    Budget(&'static str),
    Incompatible(String),
}

impl Failure {
    fn into_error(self) -> EstimationError {
        match self {
            Self::Budget(message) => refused(
                antecedent_core::reason_code!("transport_budget_cancel"),
                DETAIL_BUDGET,
                message,
            ),
            Self::Incompatible(message) => refused(
                antecedent_core::reason_code!("transport_numerical_failure"),
                DETAIL_INCOMPATIBLE,
                message,
            ),
        }
    }
}

/// A cell that is not a probability: the value is unknown, never a number.
const UNSET: f64 = -1.0;
const ZERO_MASS: f64 = -2.0;
const MISSING: f64 = -3.0;

/// Relative disagreement allowed between two evaluations of one factor at
/// different values of coordinates the factor does not depend on.
const AGREEMENT: f64 = 1e-9;

/// How a factor that reads coordinates its declared parents do not include is
/// evaluated.
///
/// On supplied exact laws such a factor takes one value at every setting of
/// those coordinates, and a disagreement means the laws contradict the graph
/// (`RequireAgreement`). On laws estimated from a sample the factor differs
/// across those coordinates by sampling noise alone, so the estimated route
/// reads the factor as the mass-weighted average, over the law's own mass on
/// those coordinates given the factor's parents, of the per-setting values
/// (`MassWeightedAverage`). The two coincide when the laws satisfy the graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FreeCoordinatePolicy {
    RequireAgreement,
    MassWeightedAverage,
}

/// Tolerance repairs an estimated-table evaluation applied, reported rather
/// than hidden.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EvaluationRepairs {
    /// Factors whose free coordinates were averaged (not required to agree).
    pub(crate) averaged_factors: u64,
    /// Consistency complements that were negative beyond tolerance and set to zero.
    pub(crate) complements_clamped: u64,
    /// Whether the final probability left `[0, 1]` beyond tolerance and was clamped.
    pub(crate) probability_clamped: bool,
    /// Engine operations used by this evaluation (private accounting only).
    pub(crate) operations: u64,
}

struct LawView<'a> {
    probabilities: &'a [f64],
    /// Variable (internal index) of each axis.
    axes: Vec<usize>,
    /// `[axis][declared level]` -> position along the axis.
    positions: Vec<Vec<usize>>,
    strides: Vec<usize>,
}

struct Engine<'a> {
    items: &'a [CounterfactualItem],
    cards: Vec<usize>,
    strides: Vec<usize>,
    rank: Vec<usize>,
    total: usize,
    tables: Vec<Option<Vec<f64>>>,
    laws: Vec<HashMap<Vec<usize>, LawView<'a>>>,
    factor_cache: HashMap<(usize, Vec<usize>), f64>,
    reads: u64,
    work: u64,
    limit: u64,
    ctx: &'a ExecutionContext,
    policy: FreeCoordinatePolicy,
    repairs: EvaluationRepairs,
}

impl Engine<'_> {
    fn tick(&mut self) -> Result<(), Failure> {
        self.work = self
            .work
            .checked_add(1)
            .ok_or(Failure::Budget("the exact evaluation operation budget is exhausted"))?;
        if self.work > self.limit {
            return Err(Failure::Budget("the exact evaluation operation budget is exhausted"));
        }
        if self.work % 4096 == 0 && self.ctx.cancellation.is_cancelled() {
            return Err(Failure::Budget("the exact evaluation was cancelled"));
        }
        Ok(())
    }

    fn level_at(&self, idx: usize, variable: usize) -> usize {
        (idx / self.strides[variable]) % self.cards[variable]
    }

    fn with_level(&self, idx: usize, variable: usize, level: usize) -> usize {
        let current = self.level_at(idx, variable);
        idx - current * self.strides[variable] + level * self.strides[variable]
    }

    /// Sum `value(item, .)` over every assignment of `dims` starting from `idx`.
    fn sum_over(&mut self, item: usize, idx: usize, dims: &[usize]) -> Result<f64, Failure> {
        let mut counters = vec![0usize; dims.len()];
        let mut total = 0.0;
        let mut sentinel: Option<f64> = None;
        loop {
            let mut cursor = idx;
            for (k, &d) in dims.iter().enumerate() {
                cursor = self.with_level(cursor, d, counters[k]);
            }
            let c = self.value(item, cursor)?;
            if c < 0.0 {
                sentinel = Some(sentinel.map_or(c, |s: f64| s.min(c)));
            } else {
                total += c;
            }
            let mut k = dims.len();
            loop {
                if k == 0 {
                    return Ok(sentinel.unwrap_or(total));
                }
                k -= 1;
                counters[k] += 1;
                if counters[k] < self.cards[dims[k]] {
                    break;
                }
                counters[k] = 0;
            }
        }
    }

    fn value(&mut self, item: usize, idx: usize) -> Result<f64, Failure> {
        if let Some(Some(table)) = self.tables.get(item) {
            let cell = table[idx];
            if cell != UNSET {
                return Ok(cell);
            }
        }
        self.tick()?;
        let computed = self.compute(item, idx)?;
        let total = self.total;
        let table = self.tables[item].get_or_insert_with(|| vec![UNSET; total]);
        table[idx] = computed;
        Ok(computed)
    }

    fn compute(&mut self, item: usize, idx: usize) -> Result<f64, Failure> {
        let items = self.items;
        let entry = items[item];
        match entry.source {
            CounterfactualItemSource::Transfer { from } => self.value(from, idx),
            CounterfactualItemSource::Marginal { from } => {
                let dropped = items[from].variables & !entry.variables;
                let dims: Vec<usize> =
                    (0..self.cards.len()).filter(|v| (dropped >> *v) & 1 == 1).collect();
                self.sum_over(from, idx, &dims)
            }
            CounterfactualItemSource::Extract { from } => {
                let all = items[from].variables;
                let mut product = 1.0;
                let mut sentinel: Option<f64> = None;
                let mut zero = false;
                for i in 0..self.cards.len() {
                    if (entry.variables >> i) & 1 == 0 {
                        continue;
                    }
                    let later: Vec<usize> = (0..self.cards.len())
                        .filter(|&k| (all >> k) & 1 == 1 && self.rank[k] > self.rank[i])
                        .collect();
                    let actual = self.level_at(idx, i);
                    let mut masses: Vec<f64> = Vec::with_capacity(self.cards[i]);
                    for level in 0..self.cards[i] {
                        let at = self.with_level(idx, i, level);
                        masses.push(self.sum_over(from, at, &later)?);
                    }
                    if let Some(&bad) = masses.iter().find(|m| **m < 0.0) {
                        let worst = masses.iter().copied().fold(bad, f64::min);
                        sentinel = Some(sentinel.map_or(worst, |s: f64| s.min(worst)));
                        continue;
                    }
                    let denominator: f64 = masses.iter().sum();
                    if denominator > 0.0 {
                        let factor = masses[actual] / denominator;
                        if factor == 0.0 {
                            zero = true;
                        }
                        product *= factor;
                    } else {
                        sentinel = Some(sentinel.map_or(ZERO_MASS, |s: f64| s.min(ZERO_MASS)));
                    }
                }
                if zero { Ok(0.0) } else { Ok(sentinel.unwrap_or(product)) }
            }
            CounterfactualItemSource::Regime { intervened, .. } => {
                let key: Vec<usize> = (0..self.cards.len())
                    .filter(|v| (intervened >> *v) & 1 == 1)
                    .map(|v| self.level_at(idx, v))
                    .collect();
                let levels: Vec<usize> =
                    (0..self.cards.len()).map(|v| self.level_at(idx, v)).collect();
                let Some(view) = self.laws[item].get(&key) else {
                    return Ok(MISSING);
                };
                let mut flat = 0usize;
                for (axis, &variable) in view.axes.iter().enumerate() {
                    flat += view.positions[axis][levels[variable]] * view.strides[axis];
                }
                self.reads += 1;
                Ok(view.probabilities[flat])
            }
        }
    }

    /// The district item and the law it extracts from, following transfers.
    fn extract_of(&self, mut item: usize) -> Option<(usize, usize)> {
        loop {
            match self.items[item].source {
                CounterfactualItemSource::Transfer { from } => item = from,
                CounterfactualItemSource::Extract { from } => return Some((item, from)),
                _ => return None,
            }
        }
    }

    /// The mass-weighted average of a district factor over the coordinates that
    /// precede its last member in the topological order and are neither district
    /// members nor bound parents. `None` when the item is not a district
    /// extraction (the agreement rule then applies).
    fn averaged_factor(
        &mut self,
        factor_item: usize,
        base: &[usize],
    ) -> Result<Option<f64>, Failure> {
        let Some((extract, from)) = self.extract_of(factor_item) else {
            return Ok(None);
        };
        let n = self.cards.len();
        let all = self.items[from].variables;
        let district = self.items[extract].variables;
        let Some(last_rank) =
            (0..n).filter(|v| (district >> *v) & 1 == 1).map(|v| self.rank[v]).max()
        else {
            return Ok(None);
        };
        let early: Vec<usize> = (0..n)
            .filter(|&v| {
                base[v] == usize::MAX
                    && (all >> v) & 1 == 1
                    && (district >> v) & 1 == 0
                    && self.rank[v] < last_rank
            })
            .collect();
        let mut idx0 = 0usize;
        for (v, &b) in base.iter().enumerate().take(n) {
            if b != usize::MAX {
                idx0 += b * self.strides[v];
            }
        }
        if early.is_empty() {
            let value = self.value(factor_item, idx0)?;
            return Ok(Some(value));
        }
        let summed: Vec<usize> = (0..n)
            .filter(|&v| {
                (all >> v) & 1 == 1
                    && ((district >> v) & 1 == 1
                        || (base[v] == usize::MAX && self.rank[v] > last_rank))
            })
            .collect();
        let mut counters = vec![0usize; early.len()];
        let mut weighted = 0.0;
        let mut mass = 0.0;
        let mut sentinel: Option<f64> = None;
        loop {
            let mut idx = idx0;
            for (k, &v) in early.iter().enumerate() {
                idx += counters[k] * self.strides[v];
            }
            let w = self.sum_over(from, idx, &summed)?;
            if w < 0.0 {
                sentinel = Some(sentinel.map_or(w, |s: f64| s.min(w)));
            } else if w > 0.0 {
                let c = self.value(factor_item, idx)?;
                if c < 0.0 {
                    sentinel = Some(sentinel.map_or(c, |s: f64| s.min(c)));
                } else {
                    weighted += w * c;
                    mass += w;
                }
            }
            let mut k = early.len();
            loop {
                if k == 0 {
                    self.repairs.averaged_factors += 1;
                    // A setting that is undefined (zero conditioning mass, no law) is
                    // left out of the average, as the agreement rule leaves it out;
                    // the factor is undefined only when no setting is defined.
                    return Ok(Some(if mass > 0.0 {
                        weighted / mass
                    } else {
                        sentinel.unwrap_or(ZERO_MASS)
                    }));
                }
                k -= 1;
                counters[k] += 1;
                if counters[k] < self.cards[early[k]] {
                    break;
                }
                counters[k] = 0;
            }
        }
    }

    fn factor_value(
        &mut self,
        factor: &CounterfactualFactor,
        symbols: &BTreeMap<usize, usize>,
    ) -> Result<f64, Failure> {
        let n = self.cards.len();
        let mut base = vec![usize::MAX; n];
        for &(variable, level) in factor.outputs.iter().chain(&factor.parents) {
            let level = match level {
                CounterfactualLevel::Fixed(l) => l.get(),
                CounterfactualLevel::Symbol(id) => match symbols.get(&id.get()) {
                    Some(&l) => l,
                    None => {
                        return Err(Failure::Incompatible(
                            "a factor reads a summation symbol that is not bound".to_owned(),
                        ));
                    }
                },
            };
            base[variable.get()] = level;
        }
        let key = (factor.item, base.clone());
        if let Some(&cached) = self.factor_cache.get(&key) {
            return Ok(cached);
        }
        if self.policy == FreeCoordinatePolicy::MassWeightedAverage {
            if let Some(value) = self.averaged_factor(factor.item, &base)? {
                self.factor_cache.insert(key, value);
                return Ok(value);
            }
        }
        let free: Vec<usize> = (0..n).filter(|&v| base[v] == usize::MAX).collect();
        let mut counters = vec![0usize; free.len()];
        let mut first: Option<f64> = None;
        let mut sentinel: Option<f64> = None;
        loop {
            let mut idx = 0usize;
            for (v, &b) in base.iter().enumerate().take(n) {
                let level = if b == usize::MAX {
                    free.iter().position(|&f| f == v).map_or(0, |p| counters[p])
                } else {
                    b
                };
                idx += level * self.strides[v];
            }
            let c = self.value(factor.item, idx)?;
            if c < 0.0 {
                sentinel = Some(sentinel.map_or(c, |s: f64| s.min(c)));
            } else if let Some(reference) = first {
                if (c - reference).abs() > AGREEMENT * (1.0 + reference.abs()) {
                    return Err(Failure::Incompatible(format!(
                        "a single-world factor takes values {reference} and {c} at two settings \
                         of coordinates it does not depend on: the supplied laws are not \
                         compatible with the declared graph"
                    )));
                }
            } else {
                first = Some(c);
            }
            let mut k = free.len();
            loop {
                if k == 0 {
                    let result = first.or(sentinel).unwrap_or(ZERO_MASS);
                    self.factor_cache.insert(key, result);
                    return Ok(result);
                }
                k -= 1;
                counters[k] += 1;
                if counters[k] < self.cards[free[k]] {
                    break;
                }
                counters[k] = 0;
            }
        }
    }

    fn sum_value(&mut self, sum: &CounterfactualSum) -> Result<f64, Failure> {
        let mut counters = vec![0usize; sum.symbols.len()];
        let mut total = 0.0;
        loop {
            let mut assignment: BTreeMap<usize, usize> = BTreeMap::new();
            for (k, symbol) in sum.symbols.iter().enumerate() {
                assignment.insert(symbol.id.get(), counters[k]);
            }
            let mut product = 1.0;
            let mut zero = false;
            let mut sentinel: Option<f64> = None;
            for factor in &sum.factors {
                let c = self.factor_value(factor, &assignment)?;
                if c < 0.0 {
                    sentinel = Some(sentinel.map_or(c, |s: f64| s.min(c)));
                } else if c == 0.0 {
                    zero = true;
                } else {
                    product *= c;
                }
            }
            if !zero {
                if let Some(s) = sentinel {
                    return Ok(s);
                }
                total += product;
            }
            let mut k = sum.symbols.len();
            loop {
                if k == 0 {
                    return Ok(total);
                }
                k -= 1;
                counters[k] += 1;
                if counters[k] < self.cards[sum.symbols[k].variable.get()] {
                    break;
                }
                counters[k] = 0;
            }
        }
    }

    fn functional(&mut self, functional: &CounterfactualFunctional) -> Result<f64, Failure> {
        match functional {
            CounterfactualFunctional::Zero => Ok(0.0),
            CounterfactualFunctional::Sum(sum) => self.sum_value(sum),
            CounterfactualFunctional::Complement { whole, flipped, .. } => {
                let a = self.functional(whole)?;
                let b = self.functional(flipped)?;
                if a < 0.0 {
                    return Ok(a);
                }
                if b < 0.0 {
                    return Ok(b);
                }
                let difference = a - b;
                if difference < -AGREEMENT * (1.0 + a.abs())
                    && self.policy == FreeCoordinatePolicy::MassWeightedAverage
                {
                    self.repairs.complements_clamped += 1;
                    return Ok(0.0);
                }
                if difference < -AGREEMENT * (1.0 + a.abs()) {
                    return Err(Failure::Incompatible(format!(
                        "the consistency complement is {difference}: the supplied laws are not \
                         compatible with the declared graph"
                    )));
                }
                Ok(difference.max(0.0))
            }
        }
    }
}

fn describe(code: f64, what: &str) -> EstimationError {
    if code <= MISSING {
        refused(
            antecedent_core::reason_code!("transport_missing_provider"),
            DETAIL_MISSING,
            format!(
                "{what} needs a regime law at an intervention level for which no law was \
                 supplied"
            ),
        )
    } else {
        refused(
            antecedent_core::reason_code!("transport_support_failure"),
            DETAIL_SUPPORT,
            format!(
                "{what} conditions on an event of zero mass in a supplied law, so a factor of \
                 the identified formula is undefined; it is not extended to a value"
            ),
        )
    }
}

/// Evaluate a checked derivation on supplied exact finite laws.
///
/// # Errors
///
/// `cell_not_licensed` (`counterfactual_transport.sampled_inference_closed`) for
/// a counted or non-exact law; `invalid_argument`
/// (`counterfactual_transport.invalid_evidence`) for a law whose population,
/// regime, axes or levels do not match the catalog and the declared levels;
/// `transport_missing_provider` (`counterfactual_transport.missing_law`) when a
/// cited regime has no law at a needed intervention level;
/// `transport_support_failure` (`counterfactual_transport.support_failure`,
/// `counterfactual_transport.zero_conditioning_mass`) for an undefined factor or a
/// conditioning event of zero mass; `transport_numerical_failure`
/// (`counterfactual_transport.evidence_incompatible`) when the laws contradict the
/// graph; `transport_budget_cancel` for a limit or cancellation.
pub fn evaluate_counterfactual_transport(
    derivation: &CounterfactualTransportDerivation,
    data: &ExactTransportData,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportPoint, EstimationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(Failure::Budget("the exact evaluation was cancelled").into_error());
    }
    if data.laws().iter().any(|law| {
        law.empirical_counts().is_some() || !matches!(law.origin(), LawOrigin::SuppliedExact)
    }) {
        return Err(refused(
            antecedent_core::reason_code!("cell_not_licensed"),
            DETAIL_COUNTED,
            "counted, empirical or posterior laws are not licensed here; the route evaluates \
             supplied exact laws only and publishes no sampling inference",
        ));
    }
    evaluate_with_policy(derivation, data, limits, ctx, FreeCoordinatePolicy::RequireAgreement)
        .map(|(point, _)| point)
}

/// Evaluate a stored derivation on finite laws estimated from counts: the same
/// arithmetic, with factor coordinates the graph does not read averaged under
/// the law's own mass instead of required to agree, and the repairs reported.
///
/// The caller owns the claim label: this value is a plug-in estimate of the
/// identified functional, not an exact finite-law evaluation.
pub(crate) fn evaluate_estimated_tables(
    derivation: &CounterfactualTransportDerivation,
    data: &ExactTransportData,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(CounterfactualTransportPoint, EvaluationRepairs), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(Failure::Budget("the exact evaluation was cancelled").into_error());
    }
    evaluate_with_policy(derivation, data, limits, ctx, FreeCoordinatePolicy::MassWeightedAverage)
}

#[allow(
    clippy::too_many_lines,
    reason = "one evaluation over the closed-form policy whose per-term accumulation shares one run state"
)]
fn evaluate_with_policy(
    derivation: &CounterfactualTransportDerivation,
    data: &ExactTransportData,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
    policy: FreeCoordinatePolicy,
) -> Result<(CounterfactualTransportPoint, EvaluationRepairs), EstimationError> {
    crate::transport::validate_exact_laws(&derivation.catalog, data)
        .map_err(|e| crate::transport::refuse_eval(&e))?;
    let problem = &derivation.problem;
    let n = problem.variable_count();
    let cards: Vec<usize> = (0..n).map(|v| problem.levels(v).len()).collect();
    let mut strides = vec![1usize; n];
    for v in (0..n.saturating_sub(1)).rev() {
        strides[v] = strides[v + 1] * cards[v + 1];
    }
    let total: usize = cards.iter().product();
    let mut rank = vec![0usize; n];
    for (position, &v) in problem.topological_order().iter().enumerate() {
        rank[v] = position;
    }
    // Per regime item, the laws by intervention levels.
    let mut laws: Vec<HashMap<Vec<usize>, LawView<'_>>> =
        Vec::with_capacity(derivation.items.len());
    for item in &derivation.items {
        let mut views: HashMap<Vec<usize>, LawView<'_>> = HashMap::new();
        if let CounterfactualItemSource::Regime { regime, intervened } = item.source {
            let population = derivation.populations[item.population].as_ref();
            for law in data.laws() {
                if law.population() != population || law.regime() != regime {
                    continue;
                }
                let (key, view) = view_of(derivation, law, item, intervened, &cards)?;
                if views.insert(key, view).is_some() {
                    return Err(invalid(format!(
                        "two laws of population {population} regime {} cover one intervention \
                         level",
                        regime.raw()
                    )));
                }
            }
        }
        laws.push(views);
    }
    let mut engine = Engine {
        items: &derivation.items,
        cards,
        strides,
        rank,
        total,
        tables: vec![None; derivation.items.len()],
        laws,
        factor_cache: HashMap::new(),
        reads: 0,
        work: 0,
        limit: u64::try_from(limits.operations).unwrap_or(u64::MAX),
        ctx,
        policy,
        repairs: EvaluationRepairs::default(),
    };
    let mut evaluate =
        |functional: &CounterfactualFunctional, what: &str| -> Result<f64, EstimationError> {
            let value = engine.functional(functional).map_err(Failure::into_error)?;
            if value < 0.0 {
                return Err(describe(value, what));
            }
            Ok(value)
        };
    let numerator = evaluate(&derivation.numerator, "the event probability")?;
    let denominator = evaluate(&derivation.denominator, "the conditioning probability")?;
    let guard = match &derivation.positivity_guard {
        Some(guard) => Some(evaluate(guard, "the original conditioning probability")?),
        None => None,
    };
    let conditional = !derivation.query.event.given().is_empty();
    let conditioning_mass = if conditional {
        guard.or_else(|| derivation.conditioning_mass_checked.then_some(denominator))
    } else {
        None
    };
    if let Some(mass) = guard {
        if mass <= 0.0 {
            return Err(refused(
                antecedent_core::reason_code!("transport_support_failure"),
                DETAIL_ZERO_MASS,
                "the original conditioning event has zero mass in the target population, so \
                 the conditional counterfactual probability is undefined",
            ));
        }
    }
    if denominator <= 0.0 {
        return Err(refused(
            antecedent_core::reason_code!("transport_support_failure"),
            DETAIL_ZERO_MASS,
            "the conditioning event has zero mass in the target population, so the conditional \
             counterfactual probability is undefined",
        ));
    }
    let probability = numerator / denominator;
    let out_of_range = !probability.is_finite()
        || probability < -AGREEMENT
        || probability > 1.0 + AGREEMENT * (1.0 + denominator.abs());
    if out_of_range
        && policy == FreeCoordinatePolicy::MassWeightedAverage
        && probability.is_finite()
    {
        engine.repairs.probability_clamped = true;
    } else if out_of_range {
        return Err(refused(
            antecedent_core::reason_code!("transport_numerical_failure"),
            DETAIL_INCOMPATIBLE,
            format!(
                "the probability {probability} is outside [0, 1]: the laws contradict the graph"
            ),
        ));
    }
    let probability = probability.clamp(0.0, 1.0);
    let mut evidence: Vec<CounterfactualTransportEvidence> = Vec::new();
    for cited in &derivation.cited {
        for law in data.laws() {
            if law.regime() == cited.regime && law.population() == cited.population.as_ref() {
                evidence.push(CounterfactualTransportEvidence {
                    population: Arc::clone(&cited.population),
                    regime: cited.regime,
                    interventions: law
                        .interventions()
                        .iter()
                        .map(|a| (a.variable, a.value.as_f64().unwrap_or(f64::NAN)))
                        .collect(),
                    snapshot_identity: law.snapshot_identity().to_owned(),
                    content_identity: law.content_identity().to_hex(),
                });
            }
        }
    }
    evidence.sort_by(|a, b| {
        (a.population.as_ref(), a.regime, a.content_identity.as_str()).cmp(&(
            b.population.as_ref(),
            b.regime,
            b.content_identity.as_str(),
        ))
    });
    engine.repairs.operations = engine.work;
    let repairs = engine.repairs;
    Ok((
        CounterfactualTransportPoint {
            numerator,
            denominator,
            probability,
            conditioning_mass,
            conditioning_mass_checked: derivation.conditioning_mass_checked,
            evidence,
            cells_read: engine.reads,
        },
        repairs,
    ))
}

/// The `(intervention levels, view)` of one supplied law of a regime item.
fn view_of<'a>(
    derivation: &CounterfactualTransportDerivation,
    law: &'a ExactDiscreteLaw,
    item: &CounterfactualItem,
    intervened: u32,
    cards: &[usize],
) -> Result<(Vec<usize>, LawView<'a>), EstimationError> {
    let problem = &derivation.problem;
    let n = problem.variable_count();
    // Intervention world.
    let mut levels: BTreeMap<usize, usize> = BTreeMap::new();
    for assignment in law.interventions() {
        let Some(variable) = problem.index_of(assignment.variable) else {
            return Err(invalid("a law intervenes on a variable outside the problem"));
        };
        let level = assignment
            .value
            .as_f64()
            .and_then(|x| problem.level_index(variable, x))
            .ok_or_else(|| invalid("a law's intervention level is not a declared level"))?;
        levels.insert(variable, level);
    }
    let intervened_set: Vec<usize> = (0..n).filter(|v| (intervened >> *v) & 1 == 1).collect();
    if levels.len() != intervened_set.len()
        || !intervened_set.iter().all(|v| levels.contains_key(v))
    {
        return Err(invalid("a law's intervention set is not its regime's"));
    }
    let key: Vec<usize> = intervened_set.iter().map(|v| levels[v]).collect();
    // Axes.
    let mut axes: Vec<usize> = Vec::with_capacity(law.axes().len());
    let mut positions: Vec<Vec<usize>> = Vec::with_capacity(law.axes().len());
    let mut mask = 0u32;
    for axis in law.axes() {
        let Some(variable) = problem.index_of(axis.variable) else {
            return Err(invalid("a law measures a variable outside the problem"));
        };
        if axis.values.len() != cards[variable] {
            return Err(invalid("a law's levels do not match the variable's declared levels"));
        }
        let mut position = vec![usize::MAX; cards[variable]];
        for (p, value) in axis.values.iter().enumerate() {
            let level = value
                .as_f64()
                .and_then(|x| problem.level_index(variable, x))
                .ok_or_else(|| invalid("a law's level is not a declared level"))?;
            position[level] = p;
        }
        if position.contains(&usize::MAX) {
            return Err(invalid("a law does not cover every declared level of a variable"));
        }
        mask |= 1 << variable;
        axes.push(variable);
        positions.push(position);
    }
    if mask != item.variables {
        return Err(invalid(
            "a law's measured variables are not the variable set of the regime the derivation \
             cites",
        ));
    }
    let mut law_strides = vec![1usize; axes.len()];
    for k in (0..axes.len().saturating_sub(1)).rev() {
        law_strides[k] = law_strides[k + 1] * law.axes()[k + 1].values.len();
    }
    Ok((key, LawView { probabilities: law.probabilities(), axes, positions, strides: law_strides }))
}

// ---------------------------------------------------------------------------
// Lowering to the expression arena (inspection and identity of the formula).
// ---------------------------------------------------------------------------

/// One derived law as a population- and regime-tagged expression: its leaves are
/// the supplied regime laws, composed by sums and ratios exactly as the
/// derivation states.
#[derive(Clone, Debug)]
pub struct LoweredCounterfactualItem {
    /// Variables of the law.
    pub variables: Vec<VariableId>,
    /// Arena holding the expression.
    pub arena: CausalExprArena,
    /// Root of the expression; the law's variables and its parents are free.
    pub root: ExprId,
}

/// Lower every item of a derivation to an expression whose leaves carry the
/// population and regime of the law they read.
#[must_use]
pub fn lower_counterfactual_transport_items(
    derivation: &CounterfactualTransportDerivation,
) -> Vec<LoweredCounterfactualItem> {
    let problem = &derivation.problem;
    let n = problem.variable_count();
    let rank: Vec<usize> = {
        let mut rank = vec![0usize; n];
        for (position, &v) in problem.topological_order().iter().enumerate() {
            rank[v] = position;
        }
        rank
    };
    let mut out: Vec<LoweredCounterfactualItem> = Vec::with_capacity(derivation.items.len());
    for (index, item) in derivation.items.iter().enumerate() {
        let mut arena = CausalExprArena::new();
        let root = lower_item(derivation, &rank, index, &mut arena);
        out.push(LoweredCounterfactualItem {
            variables: problem.variables_of_mask(item.variables),
            arena,
            root,
        });
    }
    out
}

fn sum_out(arena: &mut CausalExprArena, variables: Vec<VariableId>, expr: ExprId) -> ExprId {
    if variables.is_empty() {
        return expr;
    }
    let set: VarSetId = arena.intern_var_set(variables);
    arena.intern(ExprNode::SumOut { variables: set, expr })
}

fn lower_item(
    derivation: &CounterfactualTransportDerivation,
    rank: &[usize],
    index: usize,
    arena: &mut CausalExprArena,
) -> ExprId {
    let problem = &derivation.problem;
    let item = derivation.items[index];
    match item.source {
        CounterfactualItemSource::Regime { regime, intervened } => {
            let variables = arena.intern_var_set(problem.variables_of_mask(item.variables));
            let conditioned_on = arena.empty_var_set();
            let assignments: Vec<InterventionAssignment> = problem
                .variables_of_mask(intervened)
                .into_iter()
                .map(InterventionAssignment::symbolic)
                .collect();
            let intervention = arena.intern_intervention_assignments(assignments);
            let domain =
                if intervened == 0 { DomainRef::Observational } else { DomainRef::Interventional };
            let population =
                arena.intern_population(derivation.populations[item.population].as_ref());
            arena.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime: Some(regime),
            })
        }
        CounterfactualItemSource::Transfer { from } => lower_item(derivation, rank, from, arena),
        CounterfactualItemSource::Marginal { from } => {
            let body = lower_item(derivation, rank, from, arena);
            let dropped = derivation.items[from].variables & !item.variables;
            sum_out(arena, problem.variables_of_mask(dropped), body)
        }
        CounterfactualItemSource::Extract { from } => {
            let body = lower_item(derivation, rank, from, arena);
            let all = derivation.items[from].variables;
            let mut factors: Vec<ExprId> = Vec::new();
            for i in 0..problem.variable_count() {
                if (item.variables >> i) & 1 == 0 {
                    continue;
                }
                let mut later = 0u32;
                for k in 0..problem.variable_count() {
                    if (all >> k) & 1 == 1 && rank[k] > rank[i] {
                        later |= 1 << k;
                    }
                }
                let numerator = sum_out(arena, problem.variables_of_mask(later), body);
                let denominator = sum_out(arena, problem.variables_of_mask(later | (1 << i)), body);
                factors.push(arena.intern(ExprNode::Ratio { numerator, denominator }));
            }
            if factors.len() == 1 {
                factors[0]
            } else {
                let list = arena.intern_list(factors);
                arena.intern(ExprNode::Product(list))
            }
        }
    }
}

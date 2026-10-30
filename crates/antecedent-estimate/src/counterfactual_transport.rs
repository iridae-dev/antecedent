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

/// Immutable shared result values whose admission remains held by every clone.
///
/// It dereferences to a slice; cloning shares both data and its resource lease.
#[derive(Clone)]
pub struct RetainedSlice<T>(Arc<RetainedSliceInner<T>>);
struct RetainedSliceInner<T> {
    values: Vec<T>,
    // Declared after the values so their storage is dropped before its leases.
    leases: Vec<Arc<antecedent_core::execution::AdmissionLease>>,
}
impl<T> RetainedSlice<T> {
    pub(crate) fn new(values: Vec<T>, leases: Vec<Arc<antecedent_core::execution::AdmissionLease>>) -> Self {
        Self(Arc::new(RetainedSliceInner { values, leases }))
    }
    /// Bytes reserved with this shared collection, not allocator or RSS instrumentation.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.0.leases.iter().map(|lease| lease.remainder().bytes()).sum()
    }
}
impl<T> std::ops::Deref for RetainedSlice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] { &self.0.values }
}
impl<T> AsRef<[T]> for RetainedSlice<T> {
    fn as_ref(&self) -> &[T] { &self.0.values }
}
impl<'a, T> IntoIterator for &'a RetainedSlice<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.0.values.iter() }
}
impl<T: std::fmt::Debug> std::fmt::Debug for RetainedSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { std::fmt::Debug::fmt(&self.0.values, f) }
}
impl<T: PartialEq> PartialEq for RetainedSlice<T> {
    fn eq(&self, other: &Self) -> bool { self.0.values == other.0.values }
}

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
    pub evidence: RetainedSlice<CounterfactualTransportEvidence>,
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
    Invalid(&'static str),
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
            Self::Invalid(message) => invalid(message),
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

// Preparation shares the same operation allowance as the engine.
struct SetupBudget<'a> {
    work: u64,
    limit: u64,
    ctx: &'a ExecutionContext,
    leases: Vec<antecedent_core::execution::AdmissionLease>,
    reserved: usize,
    used: usize,
}
impl SetupBudget<'_> {
    fn charge(&mut self, amount: usize) -> Result<(), EstimationError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(Failure::Budget("the exact evaluation was cancelled").into_error());
        }
        if amount == usize::MAX {
            return Err(Failure::Budget("the exact evaluation operation budget is exhausted").into_error());
        }
        let amount = u64::try_from(amount).unwrap_or(u64::MAX);
        self.work = self.work.checked_add(amount)
            .ok_or_else(|| Failure::Budget("the exact evaluation operation budget is exhausted").into_error())?;
        if self.work > self.limit {
            return Err(Failure::Budget("the exact evaluation operation budget is exhausted").into_error());
        }
        Ok(())
    }
    fn reserve(&mut self, bytes: usize) -> Result<(), EstimationError> {
        self.charge(0)?;
        let used = self.used.checked_add(bytes)
            .ok_or_else(|| Failure::Budget("the exact evaluation setup size overflows").into_error())?;
        if used > self.reserved {
            let needed = used - self.reserved;
            // Amortize small allocations, but do not reject a request that fits
            // merely because a larger reservation block does not.
            let block = needed.max(4096);
            let acquire = |bytes: usize| {
                u64::try_from(bytes).ok().and_then(|n| self.ctx.acquire_admission(0, n).ok())
            };
            let (lease, acquired) = match acquire(block) {
                Some(lease) => (lease, block),
                None => (acquire(needed).ok_or_else(|| Failure::Budget("the exact evaluation setup exceeds memory admission").into_error())?, needed),
            };
            self.reserved = self.reserved.checked_add(acquired)
                .ok_or_else(|| Failure::Budget("the exact evaluation setup size overflows").into_error())?;
            self.leases.push(lease);
        }
        self.used = used;
        Ok(())
    }

}

// Upper bound the catalog scans that the shared validator performs. Each
// component is charged before the validator enters it; saturation refuses.
fn charge_catalog_validation(
    catalog: &antecedent_core::EvidenceCatalog,
    data: &ExactTransportData,
    work: &mut SetupBudget<'_>,
) -> Result<(), EstimationError> {
    let mut regime_fields = 0usize;
    for regime in catalog.regimes.iter() {
        work.charge(1)?;
        regime_fields = regime_fields.saturating_add(1)
            .saturating_add(regime.measured.len())
            .saturating_add(regime.interventions.len())
            .saturating_add(regime.intervention_values.len());
    }
    let mut coordinates = 0usize;
    for environment in catalog.environments.iter() {
        work.charge(1)?;
        coordinates = coordinates.saturating_add(environment.variables.len());
    }
    for law in data.laws() {
        work.charge(1)?;
        let mut intervention_size = law.interventions().len();
        for assignment in law.interventions() {
            work.charge(1)?;
            if let antecedent_core::Value::Label(text) = &assignment.value {
                intervention_size = intervention_size.saturating_add(text.len());
            }
        }
        let mut units = catalog.regimes.len().saturating_mul(law.population().len().saturating_add(1))
            .saturating_add(catalog.bindings.len().saturating_mul(law.snapshot_identity().len().saturating_add(1)))
            .saturating_add(intervention_size.saturating_mul(regime_fields.saturating_add(1)))
            .saturating_add(law.axes().len().saturating_mul(regime_fields.saturating_add(catalog.environments.len()).saturating_add(coordinates).saturating_add(1)));
        for axis in law.axes() {
            work.charge(1)?;
            units = units.saturating_add(coordinates.saturating_mul(
                axis.values.len().saturating_mul(axis.values.len()).saturating_add(1),
            ));
        }
        work.charge(units)?;
    }
    Ok(())
}

// Guard the public, caller-constructible execution shape. This is not a replay
// of identification or a scientific validation of the stored derivation.
fn validate_execution_shape(
    derivation: &CounterfactualTransportDerivation,
    cards: &[usize],
    depth_limit: usize,
    work: &mut SetupBudget<'_>,
) -> Result<(), EstimationError> {
    work.charge(derivation.items.len())?;
    let item_bytes = derivation.items.len().checked_mul(2 * std::mem::size_of::<usize>() + 2)
        .ok_or_else(|| Failure::Budget("the exact evaluation shape size overflows").into_error())?;
    work.reserve(item_bytes)?;
    let mut colors = vec![0u8; derivation.items.len()];
    for start in 0..derivation.items.len() {
        work.charge(1)?;
        if colors[start] != 0 { continue; }
        let mut current = start;
        let mut trail = Vec::new();
        loop {
            work.charge(1)?;
            match colors[current] {
                1 => return Err(invalid("a derivation item dependency is cyclic")),
                2 => break,
                _ => {}
            }
            colors[current] = 1;
            trail.push(current);
            let item = derivation.items[current];
            if item.population >= derivation.populations.len() {
                return Err(invalid("a derivation item references an invalid population"));
            }
            current = match item.source {
                CounterfactualItemSource::Regime { .. } => break,
                CounterfactualItemSource::Marginal { from }
                | CounterfactualItemSource::Extract { from }
                | CounterfactualItemSource::Transfer { from } => {
                    if from >= derivation.items.len() {
                        return Err(invalid("a derivation item references an invalid source"));
                    }
                    from
                }
            };
        }
        for index in trail { work.charge(1)?; colors[index] = 2; }
    }
    let mut pending = vec![(&derivation.numerator, 0usize), (&derivation.denominator, 0usize)];
    if let Some(guard) = &derivation.positivity_guard { pending.push((guard, 0)); }
    while let Some((functional, depth)) = pending.pop() {
        work.charge(1)?;
        // Cover both the validation and later execution frame/value stacks.
        work.reserve(128)?;
        if depth > depth_limit {
            return Err(Failure::Budget("the exact evaluation depth budget is exhausted").into_error());
        }
        match functional {
            CounterfactualFunctional::Zero => {}
            CounterfactualFunctional::Complement { whole, flipped, .. } => {
                let next = depth.checked_add(1)
                    .ok_or_else(|| Failure::Budget("the exact evaluation depth budget is exhausted").into_error())?;
                pending.push((whole, next));
                pending.push((flipped, next));
            }
            CounterfactualFunctional::Sum(sum) => {
                let bytes = sum.symbols.len().checked_mul(160)
                    .ok_or_else(|| Failure::Budget("the exact evaluation sum size overflows").into_error())?;
                work.reserve(bytes)?;
                for symbol in &sum.symbols {
                    work.charge(1)?;
                    if symbol.variable.get() >= cards.len() {
                        return Err(invalid("a summation symbol references an invalid variable"));
                    }
                }
                for factor in &sum.factors {
                    work.charge(1)?;
                    if factor.item >= derivation.items.len() {
                        return Err(invalid("a factor references an invalid item"));
                    }
                    for &(variable, level) in factor.outputs.iter().chain(&factor.parents) {
                        work.charge(1)?;
                        let Some(card) = cards.get(variable.get()) else {
                            return Err(invalid("a factor references an invalid variable"));
                        };
                        if let CounterfactualLevel::Fixed(level) = level {
                            if level.get() >= *card {
                                return Err(invalid("a factor references an invalid level"));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// State for one lazy item/sum evaluation. Frames retain their accumulated
// partial sums so no already-visited cell is recomputed.
struct ItemSum {
    item: usize,
    base: usize,
    dims: Vec<usize>,
    counters: Vec<usize>,
    total: f64,
    sentinel: Option<f64>,
    depth: usize,
}
struct ItemExtract {
    from: usize,
    idx: usize,
    variables: u32,
    all: u32,
    axis: usize,
    level: usize,
    actual: usize,
    later: Vec<usize>,
    // The lease follows its masses Vec through every continuation.
    masses: Vec<f64>,
    masses_lease: Option<antecedent_core::execution::AdmissionLease>,
    product: f64,
    sentinel: Option<f64>,
    zero: bool,
    depth: usize,
}
enum ItemFrame {
    Value { item: usize, idx: usize, depth: usize },
    Store { item: usize, idx: usize },
    Sum(ItemSum),
    SumAfter(ItemSum),
    Extract(ItemExtract),
    ExtractLevel(ItemExtract),
    ExtractAfter(ItemExtract),
}
struct ItemStack {
    frames: Vec<ItemFrame>,
    // Frames and their nested bounded dimension vectors drop before leases.
    leases: Vec<antecedent_core::execution::AdmissionLease>,
}
impl ItemStack {
    fn new() -> Self { Self { frames: Vec::new(), leases: Vec::new() } }
    fn push(&mut self, frame: ItemFrame, ctx: &ExecutionContext) -> Result<(), Failure> {
        if self.frames.len() == self.frames.capacity() {
            // At most 31 problem coordinates; cover each frame's two possible
            // coordinate vectors as well as the enum storage.
            const BLOCK: usize = 32;
            let per_frame = std::mem::size_of::<ItemFrame>().saturating_add(31 * 2 * std::mem::size_of::<usize>());
            let bytes = BLOCK.checked_mul(per_frame).and_then(|n| u64::try_from(n).ok())
                .ok_or(Failure::Budget("the exact evaluation stack size overflows"))?;
            let lease = ctx.acquire_admission(0, bytes)
                .map_err(|_| Failure::Budget("the exact evaluation stack exceeds memory admission"))?;
            self.frames.try_reserve_exact(BLOCK)
                .map_err(|_| Failure::Budget("the exact evaluation stack allocation failed"))?;
            self.leases.push(lease);
        }
        self.frames.push(frame);
        Ok(())
    }
}

struct Engine<'a> {
    items: &'a [CounterfactualItem],
    cards: Vec<usize>,
    strides: Vec<usize>,
    rank: Vec<usize>,
    total: usize,
    tables: Vec<Option<Vec<f64>>>,
    // Declared after tables so cache storage drops before its reservations.
    cache_leases: Vec<antecedent_core::execution::AdmissionLease>,
    laws: Vec<HashMap<Vec<usize>, LawView<'a>>>,
    factor_cache: HashMap<(usize, Vec<usize>), f64>,
    // The map and setup views drop before their corresponding leases.
    factor_cache_leases: Vec<antecedent_core::execution::AdmissionLease>,
    _setup_leases: Vec<antecedent_core::execution::AdmissionLease>,
    reads: u64,
    work: u64,
    limit: u64,
    depth_limit: usize,
    ctx: &'a ExecutionContext,
    policy: FreeCoordinatePolicy,
    repairs: EvaluationRepairs,
}

impl Engine<'_> {
    fn charge(&mut self, amount: usize) -> Result<(), Failure> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(Failure::Budget("the exact evaluation was cancelled"));
        }
        let amount = u64::try_from(amount).unwrap_or(u64::MAX);
        self.work = self.work.checked_add(amount)
            .ok_or(Failure::Budget("the exact evaluation operation budget is exhausted"))?;
        if self.work > self.limit {
            return Err(Failure::Budget("the exact evaluation operation budget is exhausted"));
        }
        Ok(())
    }
    fn tick(&mut self) -> Result<(), Failure> { self.charge(1) }

    fn retain_factor(&mut self, key: (usize, Vec<usize>), value: f64) -> Result<(), Failure> {
        self.tick()?;
        let bytes = key.1.len().checked_mul(std::mem::size_of::<usize>())
            .and_then(|n| n.checked_add(256))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(Failure::Budget("the exact factor cache size overflows"))?;
        let lease = self.ctx.acquire_admission(0, bytes)
            .map_err(|_| Failure::Budget("the exact factor cache exceeds memory admission"))?;
        self.factor_cache.try_reserve(1)
            .map_err(|_| Failure::Budget("the exact factor cache allocation failed"))?;
        self.factor_cache.insert(key, value);
        self.factor_cache_leases.push(lease);
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
            self.tick()?;
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

    fn ensure_item_cache(&mut self, item: usize) -> Result<(), Failure> {
        if self.tables[item].is_some() { return Ok(()); }
        self.charge(self.total)?;
        let bytes = self.total.checked_mul(std::mem::size_of::<f64>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(Failure::Budget("the exact evaluation cache size overflows"))?;
        let lease = self.ctx.acquire_admission(0, bytes)
            .map_err(|_| Failure::Budget("the exact evaluation cache exceeds memory admission"))?;
        let mut table = Vec::new();
        table.try_reserve_exact(self.total)
            .map_err(|_| Failure::Budget("the exact evaluation cache allocation failed"))?;
        while table.len() < self.total {
            self.charge(0)?;
            table.resize(table.len().saturating_add(512).min(self.total), UNSET);
        }
        self.charge(0)?;
        self.tables[item] = Some(table);
        self.cache_leases.push(lease);
        Ok(())
    }

    fn value(&mut self, item: usize, idx: usize) -> Result<f64, Failure> {
        // Fast cache hits do not allocate a continuation stack.
        self.tick()?;
        if item >= self.tables.len() || item >= self.items.len() || idx >= self.total {
            return Err(Failure::Invalid("a derivation references an invalid item or cell"));
        }
        if let Some(table) = &self.tables[item] {
            if table[idx] != UNSET { return Ok(table[idx]); }
        }
        let mut stack = ItemStack::new();
        stack.push(ItemFrame::Value { item, idx, depth: 0 }, self.ctx)?;
        let mut returned: Option<f64> = None;
        while let Some(frame) = stack.frames.pop() {
            self.tick()?;
            match frame {
                ItemFrame::Value { item, idx, depth } => {
                    if depth > self.depth_limit { return Err(Failure::Budget("the exact evaluation depth budget is exhausted")); }
                    if item >= self.items.len() || idx >= self.total {
                        return Err(Failure::Invalid("a derivation references an invalid item or cell"));
                    }
                    if let Some(table) = &self.tables[item] {
                        if table[idx] != UNSET { returned = Some(table[idx]); continue; }
                    }
                    self.ensure_item_cache(item)?;
                    let entry = self.items[item];
                    stack.push(ItemFrame::Store { item, idx }, self.ctx)?;
                    let next = depth.checked_add(1).ok_or(Failure::Budget("the exact evaluation depth budget is exhausted"))?;
                    match entry.source {
                        CounterfactualItemSource::Transfer { from } => {
                            stack.push(ItemFrame::Value { item: from, idx, depth: next }, self.ctx)?;
                        }
                        CounterfactualItemSource::Marginal { from } => {
                            let dropped = self.items[from].variables & !entry.variables;
                            let dims: Vec<_> = (0..self.cards.len()).filter(|v| (dropped >> *v) & 1 == 1).collect();
                            let counters = vec![0; dims.len()];
                            stack.push(ItemFrame::Sum(ItemSum { item: from, base: idx, dims, counters, total: 0.0, sentinel: None, depth: next }), self.ctx)?;
                        }
                        CounterfactualItemSource::Extract { from } => {
                            stack.push(ItemFrame::Extract(ItemExtract {
                                from, idx, variables: entry.variables, all: self.items[from].variables,
                                axis: 0, level: 0, actual: 0, later: Vec::new(), masses: Vec::new(),
                                masses_lease: None, product: 1.0, sentinel: None, zero: false, depth: next,
                            }), self.ctx)?;
                        }
                        CounterfactualItemSource::Regime { intervened, .. } => {
                            let key: Vec<usize> = (0..self.cards.len())
                                .filter(|v| (intervened >> *v) & 1 == 1)
                                .map(|v| self.level_at(idx, v)).collect();
                            let levels: Vec<usize> = (0..self.cards.len()).map(|v| self.level_at(idx, v)).collect();
                            if let Some(view) = self.laws[item].get(&key) {
                                let mut flat = 0usize;
                                for (axis, &variable) in view.axes.iter().enumerate() {
                                    flat += view.positions[axis][levels[variable]] * view.strides[axis];
                                }
                                self.reads += 1;
                                returned = Some(view.probabilities[flat]);
                            } else { returned = Some(MISSING); }
                        }
                    }
                }
                ItemFrame::Store { item, idx } => {
                    let value = returned.ok_or(Failure::Invalid("invalid item continuation"))?;
                    self.tables[item].as_mut().expect("admitted cache")[idx] = value;
                }
                ItemFrame::Sum(sum) => {
                    let mut cursor = sum.base;
                    for (k, &d) in sum.dims.iter().enumerate() { cursor = self.with_level(cursor, d, sum.counters[k]); }
                    let item = sum.item;
                    let depth = sum.depth;
                    stack.push(ItemFrame::SumAfter(sum), self.ctx)?;
                    stack.push(ItemFrame::Value { item, idx: cursor, depth }, self.ctx)?;
                }
                ItemFrame::SumAfter(mut sum) => {
                    let value = returned.take().ok_or(Failure::Invalid("invalid sum continuation"))?;
                    if value < 0.0 { sum.sentinel = Some(sum.sentinel.map_or(value, |s| s.min(value))); }
                    else { sum.total += value; }
                    let mut k = sum.dims.len();
                    loop {
                        if k == 0 { returned = Some(sum.sentinel.unwrap_or(sum.total)); break; }
                        k -= 1;
                        sum.counters[k] += 1;
                        if sum.counters[k] < self.cards[sum.dims[k]] {
                            stack.push(ItemFrame::Sum(sum), self.ctx)?;
                            break;
                        }
                        sum.counters[k] = 0;
                    }
                }
                ItemFrame::Extract(mut state) => {
                    while state.axis < self.cards.len() && (state.variables >> state.axis) & 1 == 0 { state.axis += 1; }
                    if state.axis == self.cards.len() {
                        returned = Some(if state.zero { 0.0 } else { state.sentinel.unwrap_or(state.product) });
                        continue;
                    }
                    state.later = (0..self.cards.len()).filter(|&k| (state.all >> k) & 1 == 1 && self.rank[k] > self.rank[state.axis]).collect();
                    state.actual = self.level_at(state.idx, state.axis);
                    state.level = 0;
                    let count = self.cards[state.axis];
                    self.charge(count)?;
                    let bytes = count.checked_mul(std::mem::size_of::<f64>()).and_then(|n| u64::try_from(n).ok())
                        .ok_or(Failure::Budget("the exact evaluation scratch size overflows"))?;
                    let lease = self.ctx.acquire_admission(0, bytes).map_err(|_| Failure::Budget("the exact evaluation scratch exceeds memory admission"))?;
                    state.masses = Vec::new();
                    state.masses.try_reserve_exact(count).map_err(|_| Failure::Budget("the exact evaluation scratch allocation failed"))?;
                    state.masses_lease = Some(lease);
                    stack.push(ItemFrame::ExtractLevel(state), self.ctx)?;
                }
                ItemFrame::ExtractLevel(mut state) => {
                    if state.level == self.cards[state.axis] {
                        if let Some(&bad) = state.masses.iter().find(|m| **m < 0.0) {
                            let worst = state.masses.iter().copied().fold(bad, f64::min);
                            state.sentinel = Some(state.sentinel.map_or(worst, |s| s.min(worst)));
                        } else {
                            let denominator: f64 = state.masses.iter().sum();
                            if denominator > 0.0 {
                                let factor = state.masses[state.actual] / denominator;
                                if factor == 0.0 { state.zero = true; }
                                state.product *= factor;
                            } else {
                                state.sentinel = Some(state.sentinel.map_or(ZERO_MASS, |s| s.min(ZERO_MASS)));
                            }
                        }
                        state.masses = Vec::new();
                        state.masses_lease = None;
                        state.axis += 1;
                        stack.push(ItemFrame::Extract(state), self.ctx)?;
                    } else {
                        let at = self.with_level(state.idx, state.axis, state.level);
                        let dims = state.later.clone();
                        let counters = vec![0; dims.len()];
                        let from = state.from;
                        let depth = state.depth;
                        stack.push(ItemFrame::ExtractAfter(state), self.ctx)?;
                        stack.push(ItemFrame::Sum(ItemSum { item: from, base: at, dims, counters, total: 0.0, sentinel: None, depth }), self.ctx)?;
                    }
                }
                ItemFrame::ExtractAfter(mut state) => {
                    state.masses.push(returned.take().ok_or(Failure::Invalid("invalid extraction continuation"))?);
                    state.level += 1;
                    stack.push(ItemFrame::ExtractLevel(state), self.ctx)?;
                }
            }
        }
        returned.ok_or(Failure::Invalid("invalid item continuation"))
    }

    /// The district item and the law it extracts from, following transfers.
    fn extract_of(&mut self, mut item: usize) -> Result<Option<(usize, usize)>, Failure> {
        loop {
            self.tick()?;
            match self.items[item].source {
                CounterfactualItemSource::Transfer { from } => item = from,
                CounterfactualItemSource::Extract { from } => return Ok(Some((item, from))),
                _ => return Ok(None),
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
        let Some((extract, from)) = self.extract_of(factor_item)? else {
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
            self.tick()?;
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
        self.tick()?;
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
            let Some(card) = self.cards.get(variable.get()) else {
                return Err(Failure::Invalid("a factor references an invalid variable"));
            };
            if level >= *card {
                return Err(Failure::Invalid("a factor references an invalid level"));
            }
            base[variable.get()] = level;
        }
        let key = (factor.item, base.clone());
        if let Some(&cached) = self.factor_cache.get(&key) {
            return Ok(cached);
        }
        if self.policy == FreeCoordinatePolicy::MassWeightedAverage {
            if let Some(value) = self.averaged_factor(factor.item, &base)? {
                self.retain_factor(key, value)?;
                return Ok(value);
            }
        }
        let free: Vec<usize> = (0..n).filter(|&v| base[v] == usize::MAX).collect();
        let mut counters = vec![0usize; free.len()];
        let mut first: Option<f64> = None;
        let mut sentinel: Option<f64> = None;
        loop {
            self.tick()?;
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
                    self.retain_factor(key, result)?;
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
            self.tick()?;
            let mut assignment: BTreeMap<usize, usize> = BTreeMap::new();
            for (k, symbol) in sum.symbols.iter().enumerate() {
                self.tick()?;
                assignment.insert(symbol.id.get(), counters[k]);
            }
            let mut product = 1.0;
            let mut zero = false;
            let mut sentinel: Option<f64> = None;
            for factor in &sum.factors {
                self.tick()?;
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
                self.tick()?;
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
        enum Frame<'a> {
            Visit(&'a CounterfactualFunctional, usize),
            Combine,
        }
        let mut pending = vec![Frame::Visit(functional, 0)];
        let mut values = Vec::new();
        while let Some(frame) = pending.pop() {
            self.tick()?;
            match frame {
                Frame::Visit(node, depth) => {
                    if depth > self.depth_limit {
                        return Err(Failure::Budget("the exact evaluation depth budget is exhausted"));
                    }
                    match node {
                        CounterfactualFunctional::Zero => {
                            values.try_reserve(1).map_err(|_| Failure::Budget("the exact evaluation stack allocation failed"))?;
                            values.push(0.0);
                        }
                        CounterfactualFunctional::Sum(sum) => {
                            let value = self.sum_value(sum)?;
                            values.try_reserve(1).map_err(|_| Failure::Budget("the exact evaluation stack allocation failed"))?;
                            values.push(value);
                        }
                        CounterfactualFunctional::Complement { whole, flipped, .. } => {
                            let next = depth.checked_add(1).ok_or(Failure::Budget("the exact evaluation depth budget is exhausted"))?;
                            pending.try_reserve(3).map_err(|_| Failure::Budget("the exact evaluation stack allocation failed"))?;
                            // LIFO: whole runs before flipped, exactly as in the recursive evaluation.
                            pending.push(Frame::Combine);
                            pending.push(Frame::Visit(flipped, next));
                            pending.push(Frame::Visit(whole, next));
                        }
                    }
                }
                Frame::Combine => {
                    let b = values.pop().ok_or(Failure::Invalid("invalid functional stack"))?;
                    let a = values.pop().ok_or(Failure::Invalid("invalid functional stack"))?;
                    let result = if a < 0.0 {
                        a
                    } else if b < 0.0 {
                        b
                    } else {
                        let difference = a - b;
                        if difference < -AGREEMENT * (1.0 + a.abs())
                            && self.policy == FreeCoordinatePolicy::MassWeightedAverage
                        {
                            self.repairs.complements_clamped += 1;
                            0.0
                        } else if difference < -AGREEMENT * (1.0 + a.abs()) {
                            return Err(Failure::Incompatible(format!(
                                "the consistency complement is {difference}: the supplied laws are not \
                                 compatible with the declared graph"
                            )));
                        } else {
                            difference.max(0.0)
                        }
                    };
                    values.push(result);
                }
            }
        }
        values.pop().ok_or(Failure::Invalid("invalid functional stack"))
    }

}

// Check and charge bounded UTF-8 work without changing canonical text or order.
fn checked_evidence_text_cmp(
    a: &str,
    b: &str,
    check: &mut impl FnMut() -> Result<(), EstimationError>,
) -> Result<std::cmp::Ordering, EstimationError> {
    check()?;
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let common = a.len().min(b.len());
    for start in (0..common).step_by(512) {
        check()?;
        let end = start.saturating_add(512).min(common);
        let order = a[start..end].cmp(&b[start..end]);
        if order != std::cmp::Ordering::Equal { return Ok(order); }
    }
    Ok(a.len().cmp(&b.len()))
}

fn checked_evidence_text_copy(
    value: &str,
    check: &mut impl FnMut() -> Result<(), EstimationError>,
) -> Result<String, EstimationError> {
    check()?;
    let mut out = String::with_capacity(value.len());
    let mut start = 0;
    while start < value.len() {
        check()?;
        let mut end = start.saturating_add(512).min(value.len());
        while !value.is_char_boundary(end) { end -= 1; }
        out.push_str(&value[start..end]);
        start = end;
    }
    check()?;
    Ok(out)
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
    antecedent_core::numerical_admission::with_context(ctx, || {
        evaluate_with_policy(derivation, data, limits, ctx, FreeCoordinatePolicy::RequireAgreement, true)
            .map(|(point, _)| point)
    })
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
    evaluate_with_policy(derivation, data, limits, ctx, FreeCoordinatePolicy::MassWeightedAverage, false)
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
    require_exact: bool,
) -> Result<(CounterfactualTransportPoint, EvaluationRepairs), EstimationError> {
    let problem = &derivation.problem;
    let n = problem.variable_count();
    let mut setup = SetupBudget {
        work: 0,
        limit: u64::try_from(limits.operations).unwrap_or(u64::MAX),
        ctx,
        leases: Vec::new(),
        reserved: 0,
        used: 0,
    };
    if require_exact {
        for law in data.laws() {
            setup.charge(1)?;
            if law.empirical_counts().is_some() || !matches!(law.origin(), LawOrigin::SuppliedExact) {
                return Err(refused(
                    antecedent_core::reason_code!("cell_not_licensed"), DETAIL_COUNTED,
                    "counted, empirical or posterior laws are not licensed here; the route evaluates \
                     supplied exact laws only and publishes no sampling inference",
                ));
            }
        }
        charge_catalog_validation(&derivation.catalog, data, &mut setup)?;
    }
    // The sampled caller prepays this validation in its provider quote; the
    // public exact entry charges it above using this same setup allowance.
    crate::transport::validate_exact_laws(&derivation.catalog, data)
        .map_err(|e| crate::transport::refuse_eval(&e))?;
    let mut cards = Vec::with_capacity(n);
    for v in 0..n {
        setup.charge(1)?;
        cards.push(problem.level_count(v).expect("problem variable index"));
    }
    validate_execution_shape(derivation, &cards, limits.depth, &mut setup)?;
    let total = cards.iter().try_fold(1usize, |total, card| total.checked_mul(*card))
        .ok_or_else(|| Failure::Budget("the exact evaluation table size overflows").into_error())?;
    let mut strides = vec![1usize; n];
    for v in (0..n.saturating_sub(1)).rev() {
        strides[v] = strides[v + 1] * cards[v + 1];
    }
    let mut rank = vec![0usize; n];
    for (position, &v) in problem.topological_order().iter().enumerate() {
        rank[v] = position;
    }
    // Per regime item, the laws by intervention levels.
    let metadata_bytes = derivation.items.len()
        .checked_mul(std::mem::size_of::<HashMap<Vec<usize>, LawView<'_>>>()
            + std::mem::size_of::<Option<Vec<f64>>>() + 64)
        .ok_or_else(|| Failure::Budget("the exact evaluation metadata size overflows").into_error())?;
    setup.reserve(metadata_bytes)?;
    let mut laws: Vec<HashMap<Vec<usize>, LawView<'_>>> =
        Vec::with_capacity(derivation.items.len());
    for item in &derivation.items {
        setup.charge(1)?;
        let mut views: HashMap<Vec<usize>, LawView<'_>> = HashMap::new();
        if let CounterfactualItemSource::Regime { regime, intervened } = item.source {
            let population = derivation.populations[item.population].as_ref();
            for law in data.laws() {
                setup.charge(1)?;
                if law.regime() != regime || checked_evidence_text_cmp(
                    law.population(), population, &mut || setup.charge(1),
                )? != std::cmp::Ordering::Equal {
                    continue;
                }
                // Cover the hash-table bucket and bounded key allocation before insertion.
                setup.reserve(512)?;
                let (key, view) = view_of(derivation, law, item, intervened, &cards, &mut setup)?;
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
        cache_leases: Vec::new(),
        laws,
        factor_cache: HashMap::new(),
        factor_cache_leases: Vec::new(),
        _setup_leases: std::mem::take(&mut setup.leases),
        reads: 0,
        work: setup.work,
        limit: setup.limit,
        depth_limit: limits.depth,
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
    let mut evidence_leases: Vec<Arc<antecedent_core::execution::AdmissionLease>> = Vec::new();
    let mut check = || {
        engine.tick().map_err(Failure::into_error)?;
        if ctx.cancellation.is_cancelled() {
            return Err(Failure::Budget("the exact evaluation was cancelled").into_error());
        }
        Ok(())
    };
    for cited in &derivation.cited {
        check()?;
        for (index, law) in data.laws().iter().enumerate() {
            check()?;
            if law.regime() == cited.regime
                && checked_evidence_text_cmp(law.population(), &cited.population, &mut check)? == std::cmp::Ordering::Equal
            {
                let bytes = law.interventions().len()
                    .checked_mul(std::mem::size_of::<(VariableId, f64)>())
                    .and_then(|n| n.checked_add(law.snapshot_identity().len()))
                    .and_then(|n| n.checked_add(512))
                    .and_then(|n| u64::try_from(n).ok())
                    .ok_or_else(|| Failure::Budget("the exact evidence size overflows").into_error())?;
                let lease = ctx.acquire_admission(0, bytes)
                    .map_err(|_| Failure::Budget("the exact evidence exceeds memory admission").into_error())?;
                evidence_leases.push(Arc::new(lease));
                let mut interventions = Vec::with_capacity(law.interventions().len());
                for assignment in law.interventions() {
                    check()?;
                    interventions.push((assignment.variable, assignment.value.as_f64().unwrap_or(f64::NAN)));
                }
                evidence.push(CounterfactualTransportEvidence {
                    population: Arc::clone(&cited.population),
                    regime: cited.regime,
                    interventions,
                    snapshot_identity: checked_evidence_text_copy(law.snapshot_identity(), &mut check)?,
                    // The provider already computed this identity with cancellation checks.
                    content_identity: data.law_identities()[index].to_hex(),
                });
            }
        }
    }
    // Stable, fallible ordering: comparisons and swaps spend the same operation
    // allowance as the evaluation and observe cancellation at bounded chunks.
    for i in 1..evidence.len() {
        let mut j = i;
        while j > 0 {
            check()?;
            let a = &evidence[j - 1];
            let b = &evidence[j];
            let order = checked_evidence_text_cmp(&a.population, &b.population, &mut check)?
                .then_with(|| a.regime.cmp(&b.regime))
                .then_with(|| a.content_identity.cmp(&b.content_identity));
            if order != std::cmp::Ordering::Greater { break; }
            evidence.swap(j - 1, j);
            j -= 1;
        }
    }
    drop(check);
    engine.repairs.operations = engine.work;
    let repairs = engine.repairs;
    Ok((
        CounterfactualTransportPoint {
            numerator,
            denominator,
            probability,
            conditioning_mass,
            conditioning_mass_checked: derivation.conditioning_mass_checked,
            evidence: RetainedSlice::new(evidence, evidence_leases),
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
    setup: &mut SetupBudget<'_>,
) -> Result<(Vec<usize>, LawView<'a>), EstimationError> {
    let problem = &derivation.problem;
    let n = problem.variable_count();
    // Intervention world.
    let mut levels: BTreeMap<usize, usize> = BTreeMap::new();
    for assignment in law.interventions() {
        setup.charge(1)?;
        let Some(variable) = problem.index_of(assignment.variable) else {
            return Err(invalid("a law intervenes on a variable outside the problem"));
        };
        let level = if let Some(value) = assignment.value.as_f64() {
            problem.try_level_index(variable, value, |amount| setup.charge(amount))?
        } else { None }
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
    let axis_bytes = law.axes().len().checked_mul(128)
        .ok_or_else(|| Failure::Budget("the exact evaluation view size overflows").into_error())?;
    setup.reserve(axis_bytes)?;
    let mut axes: Vec<usize> = Vec::with_capacity(law.axes().len());
    let mut positions: Vec<Vec<usize>> = Vec::with_capacity(law.axes().len());
    let mut mask = 0u32;
    for axis in law.axes() {
        setup.charge(1)?;
        let Some(variable) = problem.index_of(axis.variable) else {
            return Err(invalid("a law measures a variable outside the problem"));
        };
        if axis.values.len() != cards[variable] {
            return Err(invalid("a law's levels do not match the variable's declared levels"));
        }
        setup.charge(cards[variable])?;
        let bytes = cards[variable].checked_mul(std::mem::size_of::<usize>())
            .ok_or_else(|| Failure::Budget("the exact evaluation view size overflows").into_error())?;
        setup.reserve(bytes)?;
        let mut position = vec![usize::MAX; cards[variable]];
        for (p, value) in axis.values.iter().enumerate() {
            let level = if let Some(value) = value.as_f64() {
                problem.try_level_index(variable, value, |amount| setup.charge(amount))?
            } else { None }
            .ok_or_else(|| invalid("a law's level is not a declared level"))?;
            position[level] = p;
        }
        for chunk in position.chunks(512) {
            setup.charge(0)?;
            if chunk.contains(&usize::MAX) {
                return Err(invalid("a law does not cover every declared level of a variable"));
            }
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

#[cfg(test)]
mod evidence_budget_tests {
    use super::*;

    #[test]
    fn cached_cells_and_summation_steps_spend_the_evaluation_budget() {
        use antecedent_identify::{CounterfactualSymbol, CounterfactualSymbolId, CounterfactualVariableIndex};
        let ctx = ExecutionContext::for_tests(7);
        let item = CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Regime { regime: RegimeId::from_raw(0), intervened: 0 } };
        let mut engine = Engine {
            items: std::slice::from_ref(&item), cards: vec![2], strides: vec![1], rank: vec![0], total: 2,
            tables: vec![Some(vec![0.4, 0.6])], cache_leases: Vec::new(), laws: Vec::new(), factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(),
            reads: 0, work: 0, limit: 1, depth_limit: 256, ctx: &ctx, policy: FreeCoordinatePolicy::MassWeightedAverage,
            repairs: EvaluationRepairs::default(),
        };
        assert!(matches!(engine.value(0, 0), Ok(v) if v == 0.4));
        assert!(matches!(engine.value(0, 0), Err(Failure::Budget(_))));
        let sum = CounterfactualSum {
            symbols: vec![CounterfactualSymbol { id: CounterfactualSymbolId::new(0), variable: CounterfactualVariableIndex::new(0) }],
            factors: Vec::new(),
        };
        engine.work = 0;
        engine.limit = 1;
        assert!(matches!(engine.sum_value(&sum), Err(Failure::Budget(_))));
        engine.work = 0;
        engine.limit = 100;
        assert!(matches!(engine.sum_value(&sum), Ok(v) if v == 2.0));
        ctx.cancellation.cancel();
        assert!(matches!(engine.sum_value(&sum), Err(Failure::Budget(_))));
    }

    #[test]
    fn nested_complements_use_depth_and_work_without_native_recursion() {
        use antecedent_identify::CounterfactualVariableIndex;
        let ctx = ExecutionContext::for_tests(7);
        let mut engine = Engine {
            items: &[], cards: Vec::new(), strides: Vec::new(), rank: Vec::new(), total: 1,
            tables: Vec::new(), cache_leases: Vec::new(), laws: Vec::new(), factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(),
            reads: 0, work: 0, limit: 10000, depth_limit: 128, ctx: &ctx,
            policy: FreeCoordinatePolicy::RequireAgreement, repairs: EvaluationRepairs::default(),
        };
        let mut functional = CounterfactualFunctional::Sum(CounterfactualSum { symbols: Vec::new(), factors: Vec::new() });
        for _ in 0..128 {
            functional = CounterfactualFunctional::Complement {
                variable: CounterfactualVariableIndex::new(0), whole: Box::new(functional),
                flipped: Box::new(CounterfactualFunctional::Zero),
            };
        }
        assert!(matches!(engine.functional(&functional), Ok(value) if value == 1.0));
        engine.work = 0;
        engine.depth_limit = 127;
        assert!(matches!(engine.functional(&functional), Err(Failure::Budget(_))));
        engine.work = 0;
        engine.depth_limit = 128;
        engine.limit = 5;
        assert!(matches!(engine.functional(&functional), Err(Failure::Budget(_))));
        engine.work = 0;
        engine.limit = 10000;
        let negative = CounterfactualFunctional::Complement {
            variable: CounterfactualVariableIndex::new(0),
            whole: Box::new(CounterfactualFunctional::Zero),
            flipped: Box::new(CounterfactualFunctional::Sum(CounterfactualSum { symbols: Vec::new(), factors: Vec::new() })),
        };
        assert!(matches!(engine.functional(&negative), Err(Failure::Incompatible(_))));
        engine.policy = FreeCoordinatePolicy::MassWeightedAverage;
        assert!(matches!(engine.functional(&negative), Ok(value) if value == 0.0));
        assert_eq!(engine.repairs.complements_clamped, 1);
        ctx.cancellation.cancel();
        assert!(matches!(engine.functional(&functional), Err(Failure::Budget(_))));
    }

    #[test]
    fn continuations_preserve_marginal_extraction_and_lazy_cells() {
        let items = [
            CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Regime { regime: RegimeId::from_raw(0), intervened: 0 } },
            CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Extract { from: 0 } },
            CounterfactualItem { population: 0, variables: 0, source: CounterfactualItemSource::Marginal { from: 1 } },
            CounterfactualItem { population: 0, variables: 0, source: CounterfactualItemSource::Transfer { from: 2 } },
            CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Regime { regime: RegimeId::from_raw(1), intervened: 0 } },
        ];
        let probabilities = [0.25, 0.75];
        let mut laws: Vec<HashMap<Vec<usize>, LawView<'_>>> = (0..items.len()).map(|_| HashMap::new()).collect();
        laws[0].insert(Vec::new(), LawView { probabilities: &probabilities, axes: vec![0], positions: vec![vec![0, 1]], strides: vec![1] });
        let ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(1 << 20));
        let mut engine = Engine {
            items: &items, cards: vec![2], strides: vec![1], rank: vec![0], total: 2,
            tables: (0..items.len()).map(|_| None).collect(), cache_leases: Vec::new(), laws,
            factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(), reads: 0, work: 0, limit: 10000, depth_limit: 16,
            ctx: &ctx, policy: FreeCoordinatePolicy::MassWeightedAverage, repairs: EvaluationRepairs::default(),
        };
        assert!(matches!(engine.value(3, 0), Ok(value) if value == 1.0));
        assert!(matches!(engine.value(1, 0), Ok(value) if value == 0.25));
        assert!(matches!(engine.value(1, 1), Ok(value) if value == 0.75));
        assert_eq!(engine.reads, 2);
        assert!(engine.tables[4].is_none(), "unrelated item must not be evaluated eagerly");
    }

    #[test]
    fn long_acyclic_item_chain_uses_admitted_continuations() {
        let mut items = vec![CounterfactualItem {
            population: 0, variables: 1,
            source: CounterfactualItemSource::Regime { regime: RegimeId::from_raw(0), intervened: 0 },
        }];
        for _ in 0..1024 {
            let from = items.len() - 1;
            items.push(CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Transfer { from } });
            let from = items.len() - 1;
            items.push(CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Marginal { from } });
            let from = items.len() - 1;
            items.push(CounterfactualItem { population: 0, variables: 1, source: CounterfactualItemSource::Extract { from } });
        }
        let probabilities = [1.0];
        let mut laws: Vec<HashMap<Vec<usize>, LawView<'_>>> = (0..items.len()).map(|_| HashMap::new()).collect();
        laws[0].insert(Vec::new(), LawView { probabilities: &probabilities, axes: vec![0], positions: vec![vec![0]], strides: vec![1] });
        let ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(32 << 20));
        let mut engine = Engine {
            items: &items, cards: vec![1], strides: vec![1], rank: vec![0], total: 1,
            tables: (0..items.len()).map(|_| None).collect(), cache_leases: Vec::new(), laws,
            factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(), reads: 0, work: 0, limit: 200_000,
            depth_limit: items.len(), ctx: &ctx, policy: FreeCoordinatePolicy::MassWeightedAverage,
            repairs: EvaluationRepairs::default(),
        };
        assert!(matches!(engine.value(items.len() - 1, 0), Ok(value) if value == 1.0));
        assert_eq!(engine.reads, 1);
        // A cached repeat is still charged; a fresh bounded run cannot silently
        // process the entire chain under a tiny operation allowance.
        engine.work = 0;
        engine.limit = 0;
        assert!(matches!(engine.value(items.len() - 1, 0), Err(Failure::Budget(_))));
        drop(engine);
        let small_ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(64));
        let mut engine = Engine {
            items: &items, cards: vec![1], strides: vec![1], rank: vec![0], total: 1,
            tables: (0..items.len()).map(|_| None).collect(), cache_leases: Vec::new(),
            laws: (0..items.len()).map(|_| HashMap::new()).collect(), factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(),
            reads: 0, work: 0, limit: 200_000, depth_limit: items.len(), ctx: &small_ctx,
            policy: FreeCoordinatePolicy::MassWeightedAverage, repairs: EvaluationRepairs::default(),
        };
        assert!(matches!(engine.value(items.len() - 1, 0), Err(Failure::Budget(_))));
        assert!(engine.tables.iter().all(Option::is_none));
    }

    #[test]
    fn cartesian_memo_requires_budget_and_retains_memory_admission() {
        // Two independent extra-variable axes make this memo huge even when a
        // particular supplied law has only a single cell.
        let item = CounterfactualItem {
            population: 0, variables: 0,
            source: CounterfactualItemSource::Regime { regime: RegimeId::from_raw(0), intervened: 0 },
        };
        let ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(1 << 20));
        let mut large = Engine {
            items: std::slice::from_ref(&item), cards: vec![1, 32768, 32768],
            strides: vec![1 << 30, 32768, 1], rank: vec![0, 1, 2], total: 1 << 30,
            tables: vec![None], cache_leases: Vec::new(), laws: vec![HashMap::new()],
            factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(), reads: 0, work: 0, limit: u64::MAX - 1,
            depth_limit: 256, ctx: &ctx, policy: FreeCoordinatePolicy::MassWeightedAverage,
            repairs: EvaluationRepairs::default(),
        };
        assert!(matches!(large.value(0, 0), Err(Failure::Budget(_))));
        assert!(large.tables[0].is_none());
        assert!(large.cache_leases.is_empty());
        // A small memo fits, produces the same missing-law sentinel and remains
        // leased until the engine/cache is dropped.
        let mut small = Engine {
            items: std::slice::from_ref(&item), cards: vec![2], strides: vec![1], rank: vec![0], total: 2,
            tables: vec![None], cache_leases: Vec::new(), laws: vec![HashMap::new()],
            factor_cache: HashMap::new(), factor_cache_leases: Vec::new(), _setup_leases: Vec::new(), reads: 0, work: 0, limit: 100,
            depth_limit: 256, ctx: &ctx, policy: FreeCoordinatePolicy::MassWeightedAverage,
            repairs: EvaluationRepairs::default(),
        };
        assert!(matches!(small.value(0, 0), Ok(value) if value == MISSING));
        assert_eq!(small.cache_leases.len(), 1);
        assert_eq!(small.tables[0].as_ref().unwrap()[0], MISSING);
        assert!(ctx.acquire_admission(0, (1 << 20) - 15).is_err());
        let mut tight = Engine { limit: 1, ..small };
        tight.tables[0] = None;
        tight.cache_leases.clear();
        tight.work = 0;
        assert!(matches!(tight.value(0, 0), Err(Failure::Budget(_))));
        assert!(tight.tables[0].is_none());
        drop(tight);
        assert!(ctx.acquire_admission(0, (1 << 20) - 15).is_ok());
    }

    #[test]
    fn escaped_evidence_slice_keeps_its_admission_and_value_semantics() {
        let ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(1024));
        let pool = ctx.admission();
        let lease = Arc::new(ctx.acquire_admission(0, 512).unwrap());
        let evidence = RetainedSlice::new(vec![CounterfactualTransportEvidence {
            population: Arc::from("pop"), regime: RegimeId::from_raw(1),
            interventions: vec![(VariableId::from_raw(0), 1.0)],
            snapshot_identity: "snapshot".into(), content_identity: "digest".into(),
        }], vec![lease]);
        let copy = evidence.clone();
        assert_eq!(evidence, copy);
        assert_eq!(copy.as_ref()[0].snapshot_identity, "snapshot");
        drop(evidence);
        assert_eq!(pool.available().bytes(), 512);
        drop(copy);
        assert_eq!(pool.available().bytes(), 1024);
    }

    #[test]
    fn setup_and_factor_cache_keep_shared_memory_reservations() {
        let ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(512));
        {
            let mut setup = SetupBudget { work: 0, limit: 100, ctx: &ctx, leases: Vec::new(), reserved: 0, used: 0 };
            setup.reserve(400).unwrap();
            assert_eq!(setup.reserve(200).unwrap_err().reason_code(), Some("transport_budget_cancel"));
        }
        let mut engine = Engine {
            items: &[], cards: Vec::new(), strides: Vec::new(), rank: Vec::new(), total: 1,
            tables: Vec::new(), cache_leases: Vec::new(), laws: Vec::new(), factor_cache: HashMap::new(),
            factor_cache_leases: Vec::new(), _setup_leases: Vec::new(), reads: 0, work: 0, limit: 100,
            depth_limit: 10, ctx: &ctx, policy: FreeCoordinatePolicy::MassWeightedAverage,
            repairs: EvaluationRepairs::default(),
        };
        assert!(engine.retain_factor((0, vec![1, 2]), 0.5).is_ok());
        assert_eq!(engine.factor_cache.get(&(0, vec![1, 2])), Some(&0.5));
        assert!(matches!(engine.retain_factor((1, vec![1, 2]), 0.75), Err(Failure::Budget(_))));
        assert!(!engine.factor_cache.contains_key(&(1, vec![1, 2])));
        drop(engine);
        assert!(ctx.acquire_admission(0, 512).is_ok());
    }

    #[test]
    fn evidence_text_copy_and_order_poll_without_changing_utf8() {
        let text = format!("{}é{}", "a".repeat(511), "z".repeat(1100));
        let mut calls = 0;
        let copied = checked_evidence_text_copy(&text, &mut || { calls += 1; Ok(()) }).unwrap();
        assert_eq!(copied, text);
        assert!(calls >= 4);
        let mut calls = 0;
        let error = checked_evidence_text_copy(&text, &mut || {
            calls += 1;
            if calls == 3 { Err(Failure::Budget("stop").into_error()) } else { Ok(()) }
        }).unwrap_err();
        assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
        let other = format!("{}é{}", "a".repeat(511), "z".repeat(1099));
        assert_eq!(checked_evidence_text_cmp(&text, &other, &mut || Ok(())).unwrap(), text.cmp(&other));
        let mut calls = 0;
        assert!(checked_evidence_text_cmp(&text, &other, &mut || {
            calls += 1;
            if calls == 2 { Err(Failure::Budget("stop").into_error()) } else { Ok(()) }
        }).is_err());
    }
}

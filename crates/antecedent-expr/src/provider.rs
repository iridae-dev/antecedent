//! Distribution providers for compiled expression evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, PoisonError, RwLock};

use antecedent_core::{RegimeId, Value, VariableId};

use crate::{DomainRef, FactorRequirement, InterventionAssignment};

/// Weighted quadrature nodes: assignment rows paired with integration weights.
pub type QuadratureNodes = Arc<[(Arc<[Value]>, f64)]>;

/// Shared cartesian support rows, as returned by [`DistributionProvider::support`].
type SupportRows = Arc<[Arc<[Value]>]>;

/// Evaluation context (optional posterior draw index).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EvalContext {
    /// Posterior draw index when evaluating against a draw-indexed provider.
    pub draw: Option<usize>,
}

/// Variable → value binding for density / outcome lookup.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    /// Sorted by variable id.
    entries: Vec<(VariableId, Value)>,
}

impl Assignment {
    /// Empty assignment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from unsorted pairs (sorted + last-wins on duplicate vars).
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (VariableId, Value)>) -> Self {
        let mut entries: Vec<(VariableId, Value)> = pairs.into_iter().collect();
        entries.sort_by_key(|(v, _)| v.raw());
        entries.dedup_by_key(|(v, _)| *v);
        Self { entries }
    }

    /// Insert or replace a binding.
    pub fn set(&mut self, var: VariableId, value: Value) {
        match self.entries.binary_search_by_key(&var.raw(), |(v, _)| v.raw()) {
            Ok(i) => self.entries[i].1 = value,
            Err(i) => self.entries.insert(i, (var, value)),
        }
    }

    /// Borrow value for `var`, if present.
    #[must_use]
    pub fn get(&self, var: VariableId) -> Option<&Value> {
        self.entries
            .binary_search_by_key(&var.raw(), |(v, _)| v.raw())
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// All bindings, sorted.
    #[must_use]
    pub fn entries(&self) -> &[(VariableId, Value)] {
        &self.entries
    }

    /// Extend with another assignment (other wins on conflict).
    pub fn extend_from(&mut self, other: &Assignment) {
        for (v, val) in &other.entries {
            self.set(*v, val.clone());
        }
    }

    /// Remove the binding for `var`, returning the previous value (if any).
    ///
    /// Lets evaluators treat one shared assignment as a binding stack: scoped
    /// bindings are `set` on entry and removed (or restored) on exit instead
    /// of cloning the whole assignment per scope.
    pub fn remove(&mut self, var: VariableId) -> Option<Value> {
        match self.entries.binary_search_by_key(&var.raw(), |(v, _)| v.raw()) {
            Ok(i) => Some(self.entries.remove(i).1),
            Err(_) => None,
        }
    }

    /// Restrict to the given variables (order of `vars` preserved in returned values).
    pub fn values_for(&self, vars: &[VariableId]) -> Result<Vec<Value>, EvalError> {
        let mut out = Vec::with_capacity(vars.len());
        for &v in vars {
            let Some(val) = self.get(v) else {
                return Err(EvalError::MissingBinding(v));
            };
            out.push(val.clone());
        }
        Ok(out)
    }
}

/// Resolved distribution factor identity (no string keys).
#[derive(Clone, Debug)]
pub struct FactorSpec<'a> {
    /// Factor variables.
    pub variables: &'a [VariableId],
    /// Conditioning variables.
    pub conditioned_on: &'a [VariableId],
    /// Hard intervention assignments.
    pub intervention: &'a [InterventionAssignment],
    /// Observational vs interventional domain.
    pub domain: DomainRef,
    /// Population label. Empty is the default single-study factor.
    pub population: &'a str,
    /// Catalog regime this factor cites.
    pub regime: Option<RegimeId>,
}

impl<'a> FactorSpec<'a> {
    /// Untagged single-study factor (existing estimators).
    #[must_use]
    pub const fn new(
        variables: &'a [VariableId],
        conditioned_on: &'a [VariableId],
        intervention: &'a [InterventionAssignment],
        domain: DomainRef,
    ) -> Self {
        Self { variables, conditioned_on, intervention, domain, population: "", regime: None }
    }
}

/// Errors from compiling or evaluating causal expressions.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EvalError {
    /// Located exact-law coverage or support failure.
    ExactLaw(Box<crate::exact::ExactLawError>),
    /// Continuous `IntegralOut` without quadrature nodes and without discrete support.
    UnsupportedIntegralOut,
    /// Provider has no entry for the requested factor / assignment.
    MissingTableEntry,
    /// Required variable binding absent from the assignment.
    MissingBinding(VariableId),
    /// Provider reported empty support for a summed variable.
    EmptySupport(VariableId),
    /// Division by zero while evaluating a ratio.
    DivisionByZero,
    /// Located zero denominator in an exact expression.
    ExactRatioSupport {
        /// Original ratio expression, not its compiled slot.
        expression: crate::ExprId,
        /// Assignment at the failing ratio, including scoped variables.
        assignment: Arc<[(VariableId, Value)]>,
        /// Population/regime dependencies of the ratio.
        bindings: Arc<[crate::LeafBinding]>,
    },
    /// Posterior draw index out of range.
    DrawOutOfRange {
        /// Requested draw.
        draw: usize,
        /// Number of draws available.
        n_draws: usize,
    },
    /// Support row length does not match requested variable count.
    SupportShape {
        /// Expected arity.
        expected: usize,
        /// Actual arity.
        actual: usize,
    },
    /// Empirical provider used where posterior draws are required (or vice versa).
    ProviderKind(&'static str),
    /// Provider cannot answer a conditional query (non-empty `conditioned_on`) —
    /// e.g. an independent-factor provider that only models unconditional marginals.
    UnsupportedConditioning(&'static str),
    /// A provider parameter is outside its valid range (for example a non-positive variance).
    InvalidParameter(&'static str),
    /// A ratio evaluated to a non-finite value: the denominator is nonzero but so small (or a
    /// term so large) that the quotient overflows.
    NonFiniteRatio,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactLaw(error) => error.fmt(f),
            Self::ExactRatioSupport { expression, assignment, bindings } => write!(
                f,
                "zero denominator at expression {} with assignment {:?}, providers {:?}",
                expression.raw(),
                assignment,
                bindings
            ),
            Self::UnsupportedIntegralOut => {
                write!(f, "IntegralOut requires provider quadrature nodes or discrete support")
            }
            Self::MissingTableEntry => write!(f, "missing probability table entry"),
            Self::MissingBinding(v) => write!(f, "missing binding for V{}", v.raw()),
            Self::EmptySupport(v) => write!(f, "empty support for V{}", v.raw()),
            Self::DivisionByZero => write!(f, "division by zero in ratio"),
            Self::DrawOutOfRange { draw, n_draws } => {
                write!(f, "draw {draw} out of range (n_draws={n_draws})")
            }
            Self::SupportShape { expected, actual } => {
                write!(f, "support row arity {actual} != expected {expected}")
            }
            Self::NonFiniteRatio => write!(f, "ratio evaluated to a non-finite value"),
            Self::ProviderKind(msg)
            | Self::UnsupportedConditioning(msg)
            | Self::InvalidParameter(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for EvalError {}

/// Provides densities, discrete supports, and outcome values for evaluation.
pub trait DistributionProvider {
    /// Probability / density mass for a factor under an assignment.
    ///
    /// # Errors
    ///
    /// Missing table entries, bad draw index, or shape errors.
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError>;

    /// Discrete support for variables (cartesian rows of values aligned to `vars`).
    ///
    /// # Errors
    ///
    /// Empty domains or unsupported queries.
    fn support(
        &self,
        vars: &[VariableId],
        ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError>;

    /// Optional continuous quadrature nodes `(assignment_row, Lebesgue weight)` for
    /// [`crate::ExprNode::IntegralOut`].
    ///
    /// Returning `Ok(None)` asks the evaluator to fall back to discrete [`Self::support`].
    ///
    /// # Errors
    ///
    /// Provider-specific continuous-integration failures.
    fn quadrature(
        &self,
        _vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Option<QuadratureNodes>, EvalError> {
        Ok(None)
    }

    /// Outcome function value (identity: the bound value of `var`).
    ///
    /// # Errors
    ///
    /// Missing binding or non-numeric value.
    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError>;

    /// Locate a zero denominator. Legacy providers preserve their historical error.
    fn zero_denominator(
        &self,
        _arena: &crate::CausalExprArena,
        _expression: crate::ExprId,
        _assignment: &Assignment,
    ) -> EvalError {
        EvalError::DivisionByZero
    }

    /// Number of posterior draws, or `None` for a single empirical world.
    fn n_draws(&self) -> Option<usize>;
}

/// Canonical key for a factor table row.
///
/// `Hash`/`PartialEq` are defined via [`FactorKeyView`] (not derived) so the
/// owned key and the borrowed lookup view hash and compare identically — the
/// `Borrow` contract the allocation-free `probability` lookup relies on.
#[derive(Clone, Debug, Eq)]
struct FactorKey {
    variables: Arc<[VariableId]>,
    conditioned_on: Arc<[VariableId]>,
    intervention: Arc<[InterventionAssignment]>,
    domain: DomainRef,
    /// A factor cites its population and regime: `P^source(y|x)` and `P^target(y|x)` are
    /// different table rows even when every other component agrees.
    population: Arc<str>,
    regime: Option<RegimeId>,
    /// Concatenation of values for `variables` then `conditioned_on`.
    values: Arc<[Value]>,
}

/// Borrowed view of a [`FactorKey`], usable as a `HashMap` lookup key.
///
/// `probability` is the innermost evaluator call; building an owned
/// `FactorKey` there costs several `Arc` allocations per lookup. The view
/// borrows the spec slices directly and only needs the caller to assemble the
/// value row, so a lookup allocates that single row and nothing else.
#[derive(Clone, Copy)]
struct FactorKeyView<'a> {
    variables: &'a [VariableId],
    conditioned_on: &'a [VariableId],
    intervention: &'a [InterventionAssignment],
    domain: DomainRef,
    population: &'a str,
    regime: Option<RegimeId>,
    /// Concatenation of values for `variables` then `conditioned_on`.
    values: &'a [Value],
}

/// Unifies owned and borrowed factor keys behind one hash/equality identity,
/// so `HashMap<FactorKey, f64>::get` accepts a [`FactorKeyView`] through
/// `Borrow<dyn FactorKeyLookup>` without constructing a `FactorKey`.
trait FactorKeyLookup {
    fn view(&self) -> FactorKeyView<'_>;
}

impl FactorKeyLookup for FactorKey {
    fn view(&self) -> FactorKeyView<'_> {
        FactorKeyView {
            variables: &self.variables,
            conditioned_on: &self.conditioned_on,
            intervention: &self.intervention,
            domain: self.domain,
            population: &self.population,
            regime: self.regime,
            values: &self.values,
        }
    }
}

impl FactorKeyLookup for FactorKeyView<'_> {
    fn view(&self) -> FactorKeyView<'_> {
        *self
    }
}

impl<'a> Borrow<dyn FactorKeyLookup + 'a> for FactorKey {
    fn borrow(&self) -> &(dyn FactorKeyLookup + 'a) {
        self
    }
}

/// Single hashing routine for both key forms; hashing slices (rather than the
/// `Arc` wrappers) keeps the streams identical for owned and borrowed keys.
fn hash_factor_view<H: Hasher>(v: &FactorKeyView<'_>, state: &mut H) {
    v.variables.hash(state);
    v.conditioned_on.hash(state);
    v.intervention.hash(state);
    v.domain.hash(state);
    v.population.hash(state);
    v.regime.hash(state);
    v.values.hash(state);
}

impl Hash for FactorKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_factor_view(&self.view(), state);
    }
}

impl PartialEq for FactorKey {
    fn eq(&self, other: &Self) -> bool {
        factor_views_eq(&self.view(), &other.view())
    }
}

impl Hash for dyn FactorKeyLookup + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_factor_view(&self.view(), state);
    }
}

impl PartialEq for dyn FactorKeyLookup + '_ {
    fn eq(&self, other: &Self) -> bool {
        factor_views_eq(&self.view(), &other.view())
    }
}

impl Eq for dyn FactorKeyLookup + '_ {}

fn factor_views_eq(a: &FactorKeyView<'_>, b: &FactorKeyView<'_>) -> bool {
    a.variables == b.variables
        && a.conditioned_on == b.conditioned_on
        && a.intervention == b.intervention
        && a.domain == b.domain
        && a.population == b.population
        && a.regime == b.regime
        && a.values == b.values
}

/// Value row for a factor key: `variables` then `conditioned_on`, cloned out
/// of the assignment. This is the only per-lookup allocation on the hot path.
fn factor_values(spec: &FactorSpec<'_>, assignment: &Assignment) -> Result<Vec<Value>, EvalError> {
    let mut values = Vec::with_capacity(spec.variables.len() + spec.conditioned_on.len());
    for &v in spec.variables.iter().chain(spec.conditioned_on.iter()) {
        let Some(val) = assignment.get(v) else {
            return Err(EvalError::MissingBinding(v));
        };
        values.push(canonical_level(val));
    }
    Ok(values)
}

/// `0.0` and `-0.0` are one numeric level, but `Value` compares floats bitwise, so a table keyed
/// by both spellings would hold two rows for one cell. Tables and lookups use `0.0`.
fn canonical_level(value: &Value) -> Value {
    match value {
        Value::Float64(x) if *x == 0.0 => Value::Float64(0.0),
        other => other.clone(),
    }
}

/// Intervention levels with `-0.0` spelled `0.0`; borrowed unchanged in the common case.
fn canonical_interventions(
    assignments: &[InterventionAssignment],
) -> std::borrow::Cow<'_, [InterventionAssignment]> {
    let negative_zero = |a: &InterventionAssignment| matches!(a.value, Value::Float64(x) if x == 0.0 && x.is_sign_negative());
    if assignments.iter().any(negative_zero) {
        std::borrow::Cow::Owned(
            assignments
                .iter()
                .map(|a| InterventionAssignment {
                    variable: a.variable,
                    value: canonical_level(&a.value),
                })
                .collect(),
        )
    } else {
        std::borrow::Cow::Borrowed(assignments)
    }
}

/// Owned key for table inserts (cold path; lookups use [`FactorKeyView`]).
fn factor_key(spec: &FactorSpec<'_>, assignment: &Assignment) -> Result<FactorKey, EvalError> {
    let values = factor_values(spec, assignment)?;
    Ok(FactorKey {
        variables: Arc::from(spec.variables),
        conditioned_on: Arc::from(spec.conditioned_on),
        intervention: Arc::from(canonical_interventions(spec.intervention).into_owned()),
        domain: spec.domain,
        population: Arc::from(spec.population),
        regime: spec.regime,
        values: Arc::from(values),
    })
}

/// Tabular empirical distribution provider (discrete factors + domains).
#[derive(Debug, Default)]
pub struct EmpiricalTableProvider {
    domains: HashMap<VariableId, Arc<[Value]>>,
    tables: HashMap<FactorKey, f64>,
    /// Memoized cartesian supports keyed by the queried variable list.
    ///
    /// `support` takes `&self` (trait signature), so interior mutability is
    /// required; `RwLock` rather than `RefCell` keeps the provider `Sync` for
    /// callers that share it across threads. A cache hit is a cheap `Arc`
    /// clone. Only `set_domain` changes what `support` would return, so it is
    /// the one invalidation point.
    support_cache: RwLock<HashMap<Vec<VariableId>, SupportRows>>,
}

/// One declared finite domain needed by a portable empirical factor snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscreteDomainSnapshot {
    /// Semantic variable identity.
    pub variable: VariableId,
    /// Declared levels in provider order.
    pub values: Arc<[Value]>,
}

/// One complete row of an empirical factor table.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalFactorRow {
    /// Values ordered as factor variables, then conditioning variables.
    pub values: Arc<[Value]>,
    /// Probability mass for this assignment.
    pub probability: f64,
}

/// Complete empirical probability table for one checked program factor.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalFactorTableSnapshot {
    /// Positions of equivalent requirements in the checked program.
    pub requirement_indices: Arc<[usize]>,
    /// Variables whose joint law is tabulated.
    pub variables: Arc<[VariableId]>,
    /// Conditioning variables.
    pub conditioned_on: Arc<[VariableId]>,
    /// Hard intervention values.
    pub intervention: Arc<[InterventionAssignment]>,
    /// Observational or interventional law.
    pub domain: DomainRef,
    /// Population key; empty is the single-study population.
    pub population: Arc<str>,
    /// Regime key when the program factor cites one.
    pub regime: Option<RegimeId>,
    /// Complete cartesian table, including zero probability cells.
    pub rows: Arc<[EmpiricalFactorRow]>,
}

/// Portable finite discrete tables and all domains they reference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmpiricalProviderSnapshot {
    /// Domains referenced by factors, sorted by semantic variable id.
    pub domains: Arc<[DiscreteDomainSnapshot]>,
    /// Deduplicated factor laws in first program-requirement order.
    pub factors: Arc<[EmpiricalFactorTableSnapshot]>,
}

/// Failure to create or validate a complete factor-law snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProviderSnapshotError {
    /// The factor's value domain is absent or empty.
    MissingDomain(VariableId),
    /// The requested factor cell is absent from the provider.
    MissingCell {
        /// Position of the factor requirement.
        factor: usize,
    },
    /// Snapshot expansion exceeded its explicit cell limit.
    CellLimit {
        /// Number of entries required to include the next factor.
        requested: usize,
        /// Maximum entries allowed for this snapshot.
        limit: usize,
    },
    /// A probability was negative, non-finite, or greater than one.
    InvalidProbability {
        /// Position of the factor requirement.
        factor: usize,
    },
    /// A conditional table does not sum to one for one conditioning assignment.
    NotNormalized {
        /// Position of the factor requirement.
        factor: usize,
    },
    /// Snapshot contains duplicate, malformed, or unreferenced fields.
    InvalidSnapshot(&'static str),
}

impl fmt::Display for ProviderSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDomain(variable) => write!(f, "missing domain for V{}", variable.raw()),
            Self::MissingCell { factor } => write!(f, "factor {factor} has a missing table cell"),
            Self::CellLimit { requested, limit } => {
                write!(f, "snapshot requires {requested} cells; limit is {limit}")
            }
            Self::InvalidProbability { factor } => {
                write!(f, "factor {factor} contains invalid probability mass")
            }
            Self::NotNormalized { factor } => {
                write!(f, "factor {factor} is not normalized for a conditioning assignment")
            }
            Self::InvalidSnapshot(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ProviderSnapshotError {}

/// Default maximum number of probability entries materialized per snapshot.
pub const DEFAULT_FACTOR_SNAPSHOT_CELL_LIMIT: usize = 1_000_000;

impl Clone for EmpiricalTableProvider {
    fn clone(&self) -> Self {
        Self {
            domains: self.domains.clone(),
            tables: self.tables.clone(),
            // Carrying the memoized supports over is safe: they are a pure
            // function of `domains`, which is cloned alongside.
            support_cache: RwLock::new(
                self.support_cache.read().unwrap_or_else(PoisonError::into_inner).clone(),
            ),
        }
    }
}

impl EmpiricalTableProvider {
    /// Empty provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare discrete domain for a variable.
    pub fn set_domain(&mut self, var: VariableId, values: impl IntoIterator<Item = Value>) {
        let mut v: Vec<Value> = values.into_iter().collect();
        // Stable unique per numeric level: `-0.0` and `0.0` are one level.
        for x in &mut v {
            *x = canonical_level(x);
        }
        let mut seen = std::collections::HashSet::new();
        v.retain(|x| seen.insert(x.clone()));
        self.domains.insert(var, Arc::from(v));
        // Memoized supports are cartesian products of the domains; any domain
        // change invalidates every cached row set.
        self.support_cache.write().unwrap_or_else(PoisonError::into_inner).clear();
    }

    /// Insert a factor probability for the given spec + assignment.
    ///
    /// # Errors
    ///
    /// Missing bindings for factor variables / conditions.
    pub fn insert_probability(
        &mut self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        probability: f64,
    ) -> Result<(), EvalError> {
        let key = factor_key(spec, assignment)?;
        self.tables.insert(key, probability);
        Ok(())
    }

    /// Snapshot every cell required by `requirements` in portable finite-discrete form.
    /// Equivalent full factor keys share one table and retain all requirement indices.
    pub fn snapshot_factors(
        &self,
        requirements: &[FactorRequirement],
    ) -> Result<EmpiricalProviderSnapshot, ProviderSnapshotError> {
        self.snapshot_factors_with_limit(requirements, DEFAULT_FACTOR_SNAPSHOT_CELL_LIMIT)
    }

    /// As [`Self::snapshot_factors`], with an explicit cap on expanded cells.
    pub fn snapshot_factors_with_limit(
        &self,
        requirements: &[FactorRequirement],
        cell_limit: usize,
    ) -> Result<EmpiricalProviderSnapshot, ProviderSnapshotError> {
        let mut domains_by_id = std::collections::BTreeMap::<VariableId, Arc<[Value]>>::new();
        let mut tables: Vec<EmpiricalFactorTableSnapshot> = Vec::new();
        let mut total_cells = 0usize;
        for (requirement_index, requirement) in requirements.iter().enumerate() {
            let mut ids = requirement.variables.to_vec();
            ids.extend(requirement.conditioned_on.iter().copied());
            let mut unique = std::collections::HashSet::new();
            if ids.iter().any(|id| !unique.insert(*id)) {
                return Err(ProviderSnapshotError::InvalidSnapshot(
                    "factor variables and conditioners must be unique and disjoint",
                ));
            }
            let mut level_sets = Vec::with_capacity(ids.len());
            for variable in &ids {
                let values = self
                    .domains
                    .get(variable)
                    .filter(|values| !values.is_empty())
                    .ok_or(ProviderSnapshotError::MissingDomain(*variable))?;
                domains_by_id.insert(*variable, Arc::clone(values));
                level_sets.push(Arc::clone(values));
            }
            if let Some(existing) = tables.iter_mut().find(|table| {
                table.variables.as_ref() == requirement.variables.as_ref()
                    && table.conditioned_on.as_ref() == requirement.conditioned_on.as_ref()
                    && table.intervention.as_ref() == requirement.intervention.as_ref()
                    && table.domain == requirement.domain
                    && table.population.as_ref() == requirement.population.as_ref()
                    && table.regime == requirement.regime
            }) {
                let mut indices = existing.requirement_indices.to_vec();
                indices.push(requirement_index);
                existing.requirement_indices = Arc::from(indices);
                continue;
            }
            let cell_count = level_sets
                .iter()
                .try_fold(1usize, |count, levels| count.checked_mul(levels.len()))
                .ok_or(ProviderSnapshotError::CellLimit {
                    requested: usize::MAX,
                    limit: cell_limit,
                })?;
            total_cells =
                total_cells.checked_add(cell_count).ok_or(ProviderSnapshotError::CellLimit {
                    requested: usize::MAX,
                    limit: cell_limit,
                })?;
            if total_cells > cell_limit {
                return Err(ProviderSnapshotError::CellLimit {
                    requested: total_cells,
                    limit: cell_limit,
                });
            }

            let spec = FactorSpec {
                variables: &requirement.variables,
                conditioned_on: &requirement.conditioned_on,
                intervention: &requirement.intervention,
                domain: requirement.domain,
                population: &requirement.population,
                regime: requirement.regime,
            };
            let value_rows = cartesian_values(&level_sets);
            let mut rows = Vec::with_capacity(value_rows.len());
            let mut conditional_sums = HashMap::<Vec<Value>, f64>::new();
            for values in value_rows {
                let assignment =
                    Assignment::from_pairs(ids.iter().copied().zip(values.iter().cloned()));
                let probability =
                    self.probability(&spec, &assignment, &EvalContext::default()).map_err(
                        |_| ProviderSnapshotError::MissingCell { factor: requirement_index },
                    )?;
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(ProviderSnapshotError::InvalidProbability {
                        factor: requirement_index,
                    });
                }
                let split = requirement.variables.len();
                *conditional_sums.entry(values[split..].to_vec()).or_default() += probability;
                rows.push(EmpiricalFactorRow { values: Arc::from(values), probability });
            }
            if conditional_sums.values().any(|sum| (sum - 1.0).abs() > 1e-9) {
                return Err(ProviderSnapshotError::NotNormalized { factor: requirement_index });
            }
            tables.push(EmpiricalFactorTableSnapshot {
                requirement_indices: Arc::from([requirement_index]),
                variables: Arc::clone(&requirement.variables),
                conditioned_on: Arc::clone(&requirement.conditioned_on),
                intervention: Arc::clone(&requirement.intervention),
                domain: requirement.domain,
                population: Arc::clone(&requirement.population),
                regime: requirement.regime,
                rows: Arc::from(rows),
            });
        }
        Ok(EmpiricalProviderSnapshot {
            domains: Arc::from(
                domains_by_id
                    .into_iter()
                    .map(|(variable, values)| DiscreteDomainSnapshot { variable, values })
                    .collect::<Vec<_>>(),
            ),
            factors: Arc::from(tables),
        })
    }
}

fn cartesian_values(level_sets: &[Arc<[Value]>]) -> Vec<Vec<Value>> {
    let mut rows = vec![Vec::new()];
    for levels in level_sets {
        let mut next = Vec::with_capacity(rows.len().saturating_mul(levels.len()));
        for prefix in &rows {
            for value in levels.iter() {
                let mut row = prefix.clone();
                row.push(value.clone());
                next.push(row);
            }
        }
        rows = next;
    }
    rows
}

impl DistributionProvider for EmpiricalTableProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        // Borrowed-key lookup: no owned `FactorKey` (and its per-field `Arc`
        // allocations) on the hot path — only the value row is assembled.
        let values = factor_values(spec, assignment)?;
        let intervention = canonical_interventions(spec.intervention);
        let key = FactorKeyView {
            variables: spec.variables,
            conditioned_on: spec.conditioned_on,
            intervention: &intervention,
            domain: spec.domain,
            population: spec.population,
            regime: spec.regime,
            values: &values,
        };
        self.tables.get(&key as &dyn FactorKeyLookup).copied().ok_or(EvalError::MissingTableEntry)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        // Callers (SumOut / IntegralOut / Expectation) request the same
        // variable sets on every draw, replicate, and nesting level; the
        // cartesian product is a pure function of `domains`, so memoize it and
        // hand back a shared `Arc` on hits. Errors (missing / empty domain)
        // are not cached — that path aborts evaluation anyway.
        if let Some(hit) =
            self.support_cache.read().unwrap_or_else(PoisonError::into_inner).get(vars)
        {
            return Ok(Arc::clone(hit));
        }
        // `vars.is_empty()` needs no special case: the fold below yields the
        // single empty row, matching the pre-memoization behavior.
        let mut rows: Vec<Vec<Value>> = vec![Vec::new()];
        for &v in vars {
            let domain = self.domains.get(&v).ok_or(EvalError::EmptySupport(v))?;
            if domain.is_empty() {
                return Err(EvalError::EmptySupport(v));
            }
            let mut next = Vec::with_capacity(rows.len() * domain.len());
            for prefix in &rows {
                for val in domain.iter() {
                    let mut row = prefix.clone();
                    row.push(val.clone());
                    next.push(row);
                }
            }
            rows = next;
        }
        let out: Arc<[Arc<[Value]>]> = rows.into_iter().map(Arc::from).collect();
        self.support_cache
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(vars.to_vec(), Arc::clone(&out));
        Ok(out)
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let value = assignment.get(var).ok_or(EvalError::MissingBinding(var))?;
        value.as_f64().ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

/// Draw-indexed posterior provider: one [`EmpiricalTableProvider`] per draw.
#[derive(Clone, Debug, Default)]
pub struct PosteriorDrawProvider {
    draws: Vec<EmpiricalTableProvider>,
}

impl PosteriorDrawProvider {
    /// Empty posterior provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from per-draw empirical tables.
    #[must_use]
    pub fn from_draws(draws: Vec<EmpiricalTableProvider>) -> Self {
        Self { draws }
    }

    /// Number of draws.
    #[must_use]
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether there are no draws.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    fn table(&self, ctx: &EvalContext) -> Result<&EmpiricalTableProvider, EvalError> {
        let draw = ctx
            .draw
            .ok_or(EvalError::ProviderKind("PosteriorDrawProvider requires EvalContext.draw"))?;
        self.draws.get(draw).ok_or(EvalError::DrawOutOfRange { draw, n_draws: self.draws.len() })
    }
}

impl DistributionProvider for PosteriorDrawProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.table(ctx)?.probability(spec, assignment, ctx)
    }

    fn support(
        &self,
        vars: &[VariableId],
        ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        self.table(ctx)?.support(vars, ctx)
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.table(ctx)?.outcome(var, assignment, ctx)
    }

    fn n_draws(&self) -> Option<usize> {
        Some(self.draws.len())
    }
}

/// Independent Gaussian density provider with Gauss–Hermite quadrature.
///
/// `probability` returns the product of univariate N(μ, σ²) densities for factor
/// variables. [`Self::quadrature`] returns product Gauss–Hermite nodes in
/// Lebesgue measure (suitable for ∫ body(x) dx near each Gaussian mode).
#[derive(Clone, Debug, Default)]
pub struct GaussianDensityProvider {
    /// Per-variable (mean, variance).
    params: HashMap<VariableId, (f64, f64)>,
}

impl GaussianDensityProvider {
    /// Empty provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an independent Gaussian for `var` with mean `mean` and variance `variance`.
    ///
    /// # Errors
    ///
    /// [`EvalError::InvalidParameter`] unless `mean` is finite and `variance` is finite and
    /// positive; nothing is stored then.
    pub fn set_gaussian(
        &mut self,
        var: VariableId,
        mean: f64,
        variance: f64,
    ) -> Result<(), EvalError> {
        if !(variance > 0.0 && variance.is_finite() && mean.is_finite()) {
            return Err(EvalError::InvalidParameter(
                "Gaussian mean must be finite and variance finite and positive",
            ));
        }
        self.params.insert(var, (mean, variance));
        Ok(())
    }
}

/// Physicists' Gauss–Hermite nodes/weights for ∫ e^{-t²} g(t) dt (n = 5).
const GH5_NODES: [f64; 5] = [
    -2.020_182_870_456_085_6,
    -0.958_572_464_613_818_5,
    0.0,
    0.958_572_464_613_818_5,
    2.020_182_870_456_085_6,
];
const GH5_WEIGHTS: [f64; 5] = [
    0.019_953_242_059_045_913,
    0.393_619_323_152_241_35,
    0.945_308_720_482_941_9,
    0.393_619_323_152_241_35,
    0.019_953_242_059_045_913,
];

impl DistributionProvider for GaussianDensityProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        // Independent Gaussians only model unconditional marginals; a non-empty
        // `conditioned_on` would silently return P(variables) instead of the
        // requested P(variables | conditioned_on), so reject rather than guess.
        if !spec.conditioned_on.is_empty() {
            return Err(EvalError::UnsupportedConditioning(
                "GaussianDensityProvider models independent Gaussians and cannot answer \
                 conditional queries; conditioned_on must be empty",
            ));
        }
        // Nor can it answer a factor under `do(.)`, or one that a population / regime tag asks a
        // specific study for: the marginal of y is not `P(y | do(t))`.
        if !spec.intervention.is_empty() || spec.domain == DomainRef::Interventional {
            return Err(EvalError::ProviderKind(
                "GaussianDensityProvider models observational marginals and cannot answer \
                 interventional factors",
            ));
        }
        if !spec.population.is_empty() || spec.regime.is_some() {
            return Err(EvalError::ProviderKind(
                "GaussianDensityProvider is untagged and cannot serve population- or \
                 regime-tagged factors",
            ));
        }
        let mut dens = 1.0;
        for &v in spec.variables {
            let (mean, var) = self.params.get(&v).copied().ok_or(EvalError::EmptySupport(v))?;
            let x =
                assignment.get(v).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(v))?;
            let inv_sqrt = (2.0 * std::f64::consts::PI * var).sqrt().recip();
            let z = (x - mean) / var.sqrt();
            dens *= inv_sqrt * (-0.5 * z * z).exp();
        }
        Ok(dens)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        if vars.is_empty() {
            return Ok(Arc::from(vec![Arc::from(Vec::<Value>::new())]));
        }
        Err(EvalError::EmptySupport(vars[0]))
    }

    fn quadrature(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Option<QuadratureNodes>, EvalError> {
        if vars.is_empty() {
            return Ok(Some(Arc::from([(Arc::from(Vec::<Value>::new()), 1.0)])));
        }
        // Product GH: start with empty prefix.
        let mut nodes: Vec<(Vec<Value>, f64)> = vec![(Vec::new(), 1.0)];
        for &v in vars {
            let (mean, variance) =
                self.params.get(&v).copied().ok_or(EvalError::EmptySupport(v))?;
            let sigma = variance.sqrt();
            let scale = sigma * std::f64::consts::SQRT_2;
            let mut next = Vec::with_capacity(nodes.len() * GH5_NODES.len());
            for (prefix, w0) in &nodes {
                for (i, &t) in GH5_NODES.iter().enumerate() {
                    let x = mean + scale * t;
                    // GH computes ∫ e^{-t²} g(t) dt. For Lebesgue ∫ h(x) dx with
                    // x = μ + σ√2 t we need g(t) = h(x) σ√2 e^{t²}, so the node
                    // weight applied to h(x) is w_i · σ√2 · e^{t²}.
                    let w = w0 * GH5_WEIGHTS[i] * scale * (t * t).exp();
                    let mut row = prefix.clone();
                    row.push(Value::f64(x));
                    next.push((row, w));
                }
            }
            nodes = next;
        }
        let out: Vec<(Arc<[Value]>, f64)> =
            nodes.into_iter().map(|(row, w)| (Arc::from(row), w)).collect();
        Ok(Some(Arc::from(out)))
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let value = assignment.get(var).ok_or(EvalError::MissingBinding(var))?;
        value.as_f64().ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(id: u32) -> VariableId {
        VariableId::from_raw(id)
    }

    fn f(x: f64) -> Value {
        Value::f64(x)
    }

    fn requirement(variables: &[VariableId], conditioned_on: &[VariableId]) -> FactorRequirement {
        FactorRequirement {
            variables: Arc::from(variables),
            conditioned_on: Arc::from(conditioned_on),
            intervention: Arc::from([]),
            domain: DomainRef::Observational,
            population: Arc::from(""),
            regime: None,
        }
    }

    #[test]
    fn factor_snapshot_exports_complete_law_and_deduplicates_aliases() {
        let mut provider = EmpiricalTableProvider::new();
        let y = v(0);
        provider.set_domain(y, [f(0.0), f(1.0)]);
        let y_vars = [y];
        let spec = FactorSpec::new(&y_vars, &[], &[], DomainRef::Observational);
        for (value, probability) in [(0.0, 0.25), (1.0, 0.75)] {
            provider
                .insert_probability(&spec, &Assignment::from_pairs([(y, f(value))]), probability)
                .unwrap();
        }

        let snapshot =
            provider.snapshot_factors(&[requirement(&[y], &[]), requirement(&[y], &[])]).unwrap();
        assert_eq!(snapshot.domains.len(), 1);
        assert_eq!(snapshot.factors.len(), 1);
        assert_eq!(snapshot.factors[0].requirement_indices.as_ref(), &[0, 1]);
        assert_eq!(snapshot.factors[0].rows.len(), 2);
        assert_eq!(snapshot.factors[0].rows[0].values.as_ref(), &[f(0.0)]);
        assert_eq!(snapshot.factors[0].rows[0].probability, 0.25);
        assert_eq!(snapshot.factors[0].rows[1].probability, 0.75);
    }

    #[test]
    fn factor_snapshot_refuses_missing_conditional_cells_and_resource_overflow() {
        let mut provider = EmpiricalTableProvider::new();
        let x = v(0);
        let y = v(1);
        provider.set_domain(x, [f(0.0), f(1.0)]);
        provider.set_domain(y, [f(0.0), f(1.0)]);
        let y_vars = [y];
        let x_conditioners = [x];
        let spec = FactorSpec::new(&y_vars, &x_conditioners, &[], DomainRef::Observational);
        for (x_value, y_value, probability) in [(0.0, 0.0, 0.5), (0.0, 1.0, 0.5), (1.0, 0.0, 0.2)] {
            provider
                .insert_probability(
                    &spec,
                    &Assignment::from_pairs([(x, f(x_value)), (y, f(y_value))]),
                    probability,
                )
                .unwrap();
        }
        assert!(matches!(
            provider.snapshot_factors(&[requirement(&[y], &[x])]),
            Err(ProviderSnapshotError::MissingCell { factor: 0 })
        ));

        let complete = {
            let mut p = EmpiricalTableProvider::new();
            p.set_domain(y, [f(0.0), f(1.0)]);
            let y_vars = [y];
            let law = FactorSpec::new(&y_vars, &[], &[], DomainRef::Observational);
            for value in [0.0, 1.0] {
                p.insert_probability(&law, &Assignment::from_pairs([(y, f(value))]), 0.5).unwrap();
            }
            p
        };
        assert!(matches!(
            complete.snapshot_factors_with_limit(&[requirement(&[y], &[])], 1),
            Err(ProviderSnapshotError::CellLimit { requested: 2, limit: 1 })
        ));
    }

    #[test]
    fn empirical_table_missing_entry_errors() {
        // Domain declared but no `insert_probability` call for this cell: must
        // surface `MissingTableEntry` rather than silently yielding 0.0.
        let mut p = EmpiricalTableProvider::new();
        let y = v(0);
        p.set_domain(y, [f(0.0), f(1.0)]);
        let spec = FactorSpec {
            variables: &[y],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        let assignment = Assignment::from_pairs([(y, f(0.0))]);
        let err = p.probability(&spec, &assignment, &EvalContext::default()).unwrap_err();
        assert_eq!(err, EvalError::MissingTableEntry);
    }

    #[test]
    fn support_memoizes_and_invalidates_on_set_domain() {
        // A repeated `support` query must be a cheap clone of the same shared
        // rows (memoization), and `set_domain` — the only mutation that can
        // change what `support` returns — must invalidate the cache.
        let mut p = EmpiricalTableProvider::new();
        let a = v(0);
        let b = v(1);
        p.set_domain(a, [f(0.0), f(1.0)]);
        p.set_domain(b, [f(0.0), f(1.0), f(2.0)]);
        let ctx = EvalContext::default();
        let first = p.support(&[a, b], &ctx).unwrap();
        assert_eq!(first.len(), 6);
        let second = p.support(&[a, b], &ctx).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "cache hit must return the shared rows");
        p.set_domain(b, [f(0.0), f(1.0)]);
        let third = p.support(&[a, b], &ctx).unwrap();
        assert!(!Arc::ptr_eq(&first, &third), "set_domain must invalidate the cache");
        // Cartesian order (outer var major, domain order minor) is unchanged.
        let rows: Vec<Vec<f64>> =
            third.iter().map(|r| r.iter().map(|x| x.as_f64().unwrap()).collect()).collect();
        assert_eq!(rows, vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]]);
    }

    #[test]
    fn empty_support_query_yields_single_empty_row() {
        // The empty query has one row (the empty assignment); the memoized
        // path must preserve that vacuous-product convention.
        let p = EmpiricalTableProvider::new();
        let rows = p.support(&[], &EvalContext::default()).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].is_empty());
    }

    #[test]
    fn borrowed_key_lookup_matches_owned_insert() {
        // `probability` looks up via the borrowed `FactorKeyView`; it must hit
        // entries inserted via the owned `FactorKey` (same hash/equality
        // identity across every component) and still miss when any component
        // — value row or domain — differs.
        let mut p = EmpiricalTableProvider::new();
        let y = v(0);
        let z = v(1);
        let t = v(2);
        let interv = [InterventionAssignment { variable: t, value: f(1.0) }];
        let spec = FactorSpec {
            variables: &[y],
            conditioned_on: &[z],
            intervention: &interv,
            domain: DomainRef::Interventional,
            population: "",
            regime: None,
        };
        let assign = Assignment::from_pairs([(y, f(1.0)), (z, f(0.0))]);
        p.insert_probability(&spec, &assign, 0.25).unwrap();
        let ctx = EvalContext::default();
        assert!((p.probability(&spec, &assign, &ctx).unwrap() - 0.25).abs() < 1e-15);
        let other = Assignment::from_pairs([(y, f(1.0)), (z, f(1.0))]);
        assert_eq!(p.probability(&spec, &other, &ctx).unwrap_err(), EvalError::MissingTableEntry);
        let obs = FactorSpec { domain: DomainRef::Observational, ..spec.clone() };
        assert_eq!(p.probability(&obs, &assign, &ctx).unwrap_err(), EvalError::MissingTableEntry);
    }

    #[test]
    fn gaussian_provider_rejects_conditional_query() {
        // `GaussianDensityProvider` models independent Gaussians; it must error on a
        // conditional query (non-empty `conditioned_on`) rather than silently
        // returning the unconditional marginal P(variables).
        let mut p = GaussianDensityProvider::new();
        let y = v(0);
        let z = v(1);
        p.set_gaussian(y, 0.0, 1.0).unwrap();
        p.set_gaussian(z, 0.0, 1.0).unwrap();
        let spec = FactorSpec {
            variables: &[y],
            conditioned_on: &[z],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        let assignment = Assignment::from_pairs([(y, f(0.5)), (z, f(0.2))]);
        let err = p.probability(&spec, &assignment, &EvalContext::default()).unwrap_err();
        assert!(matches!(err, EvalError::UnsupportedConditioning(_)));
    }

    #[test]
    fn table_rows_are_keyed_by_population_and_regime() {
        // P^source(y) and P^target(y) share variables, domain and level; the table must keep
        // them apart instead of letting the last insert answer both.
        let mut p = EmpiricalTableProvider::new();
        let y = v(0);
        p.set_domain(y, [f(0.0), f(1.0)]);
        let ys = [y];
        let tagged = |population, regime| FactorSpec {
            variables: &ys,
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
            population,
            regime,
        };
        let assign = Assignment::from_pairs([(y, f(1.0))]);
        let ctx = EvalContext::default();
        let (r0, r1) = (Some(RegimeId::from_raw(0)), Some(RegimeId::from_raw(1)));
        p.insert_probability(&tagged("source", r0), &assign, 0.2).unwrap();
        p.insert_probability(&tagged("target", r0), &assign, 0.9).unwrap();
        assert_eq!(p.probability(&tagged("source", r0), &assign, &ctx), Ok(0.2));
        assert_eq!(p.probability(&tagged("target", r0), &assign, &ctx), Ok(0.9));
        assert_eq!(
            p.probability(&tagged("source", r1), &assign, &ctx),
            Err(EvalError::MissingTableEntry)
        );
        assert_eq!(
            p.probability(&tagged("", None), &assign, &ctx),
            Err(EvalError::MissingTableEntry)
        );
    }

    #[test]
    fn negative_zero_is_the_same_level_as_zero() {
        let mut p = EmpiricalTableProvider::new();
        let y = v(0);
        p.set_domain(y, [f(0.0), f(-0.0), f(1.0)]);
        let rows = p.support(&[y], &EvalContext::default()).unwrap();
        assert_eq!(rows.len(), 2, "0.0 and -0.0 are one level");
        let ys = [y];
        let spec = FactorSpec::new(&ys, &[], &[], DomainRef::Observational);
        p.insert_probability(&spec, &Assignment::from_pairs([(y, f(-0.0))]), 0.5).unwrap();
        let ctx = EvalContext::default();
        assert_eq!(p.probability(&spec, &Assignment::from_pairs([(y, f(0.0))]), &ctx), Ok(0.5));
        assert_eq!(p.probability(&spec, &Assignment::from_pairs([(y, f(-0.0))]), &ctx), Ok(0.5));
        // The same holds for an intervention level.
        let t = v(1);
        let neg = [InterventionAssignment { variable: t, value: f(-0.0) }];
        let pos = [InterventionAssignment { variable: t, value: f(0.0) }];
        let spec_neg = FactorSpec {
            variables: &[],
            conditioned_on: &[],
            intervention: &neg,
            domain: DomainRef::Interventional,
            population: "",
            regime: None,
        };
        let spec_pos = FactorSpec { intervention: &pos, ..spec_neg.clone() };
        p.insert_probability(&spec_neg, &Assignment::new(), 0.25).unwrap();
        assert_eq!(p.probability(&spec_pos, &Assignment::new(), &ctx), Ok(0.25));
    }

    #[test]
    fn gaussian_provider_refuses_interventional_and_tagged_factors() {
        let mut p = GaussianDensityProvider::new();
        let (y, t) = (v(0), v(1));
        p.set_gaussian(y, 0.0, 1.0).unwrap();
        let assign = Assignment::from_pairs([(y, f(0.0))]);
        let ctx = EvalContext::default();
        let ys = [y];
        let interv = [InterventionAssignment { variable: t, value: f(1.0) }];
        let under_do = FactorSpec {
            variables: &ys,
            conditioned_on: &[],
            intervention: &interv,
            domain: DomainRef::Interventional,
            population: "",
            regime: None,
        };
        assert!(matches!(p.probability(&under_do, &assign, &ctx), Err(EvalError::ProviderKind(_))));
        let tagged = FactorSpec {
            population: "target",
            ..FactorSpec::new(&ys, &[], &[], DomainRef::Observational)
        };
        assert!(matches!(p.probability(&tagged, &assign, &ctx), Err(EvalError::ProviderKind(_))));
        // The untagged observational marginal is still the standard normal density at 0.
        let plain = FactorSpec::new(&ys, &[], &[], DomainRef::Observational);
        let density = p.probability(&plain, &assign, &ctx).unwrap();
        assert!((density - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn set_gaussian_rejects_invalid_parameters_and_stores_nothing() {
        let mut p = GaussianDensityProvider::new();
        let y = v(0);
        for (mean, variance) in
            [(0.0, 0.0), (0.0, -1.0), (0.0, f64::NAN), (f64::NAN, 1.0), (0.0, f64::INFINITY)]
        {
            assert!(matches!(
                p.set_gaussian(y, mean, variance),
                Err(EvalError::InvalidParameter(_))
            ));
        }
        let ys = [y];
        let spec = FactorSpec::new(&ys, &[], &[], DomainRef::Observational);
        let assign = Assignment::from_pairs([(y, f(0.0))]);
        assert_eq!(
            p.probability(&spec, &assign, &EvalContext::default()),
            Err(EvalError::EmptySupport(y))
        );
    }
}

//! Validated finite-discrete laws. These inputs are laws, not sample frequencies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use antecedent_core::{RegimeId, Value, VariableId};

use crate::{
    Assignment, DistributionProvider, DomainRef, EvalContext, EvalError, FactorSpec,
    InterventionAssignment,
};

/// How a dense joint was obtained. Structural zeros are licensed only for supplied laws.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LawOrigin {
    /// Caller-supplied complete population law. Zero cells may be structural.
    SuppliedExact,
    /// Frequency plug-in from a finite sample. Zero cells are unobserved, not structural.
    EmpiricalPlugin,
    /// Coherent learned finite law with retained empirical counts.
    LearnedPlugin,
}

impl LawOrigin {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SuppliedExact => "supplied_exact",
            Self::EmpiricalPlugin => "empirical_plugin",
            Self::LearnedPlugin => "learned_plugin",
        }
    }

    /// Parse a stable origin name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "supplied_exact" | "" => Some(Self::SuppliedExact),
            "empirical_plugin" => Some(Self::EmpiricalPlugin),
            "learned_plugin" => Some(Self::LearnedPlugin),
            _ => None,
        }
    }
}

/// Explicit floating-point validation policy; no normalization is performed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LawTolerance {
    /// Absolute error allowance.
    pub absolute: f64,
    /// Relative error allowance.
    pub relative: f64,
}

impl Default for LawTolerance {
    fn default() -> Self {
        Self { absolute: 1e-12, relative: 1e-10 }
    }
}

impl LawTolerance {
    fn valid(self) -> bool {
        self.absolute.is_finite()
            && self.relative.is_finite()
            && self.absolute >= 0.0
            && self.relative >= 0.0
            && self.absolute + self.relative < 1.0
    }
    fn unit_mass(self, mass: f64) -> bool {
        mass.is_finite() && (mass - 1.0).abs() <= self.absolute + self.relative
    }
}

/// A named finite coordinate. Axis order is the dense table's physical order.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteAxis {
    /// Original graph coordinate.
    pub variable: VariableId,
    /// Ordered, distinct, finite levels.
    pub values: Arc<[Value]>,
}

/// Located failure to validate or use an exact law.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactLawError {
    /// Stable machine-readable failure kind.
    pub kind: &'static str,
    /// Population, when a law has been selected.
    pub population: Arc<str>,
    /// Catalog regime.
    pub regime: Option<RegimeId>,
    /// Requested factor variables.
    pub variables: Arc<[VariableId]>,
    /// Located conditioning assignment.
    pub conditioning: Arc<[(VariableId, Value)]>,
    /// Concrete intervention world.
    pub interventions: Arc<[InterventionAssignment]>,
}

impl std::fmt::Display for ExactLawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} in population {} regime {:?}; factor {:?}, conditioning {:?}, interventions {:?}",
            self.kind,
            self.population,
            self.regime,
            self.variables,
            self.conditioning,
            self.interventions
        )
    }
}
impl std::error::Error for ExactLawError {}

/// Immutable dense joint law for one population and concrete intervention world.
/// Last axis varies fastest. Every assignment is present, including zero masses.
#[derive(Clone, Debug)]
pub struct ExactDiscreteLaw {
    population: Arc<str>,
    regime: RegimeId,
    interventions: Arc<[InterventionAssignment]>,
    axes: Arc<[DiscreteAxis]>,
    probabilities: Arc<[f64]>,
    strides: Arc<[usize]>,
    axis_index: Arc<BTreeMap<VariableId, usize>>,
    level_index: Arc<[HashMap<Value, usize>]>,
    snapshot_identity: Arc<str>,
    tolerance: LawTolerance,
    origin: LawOrigin,
    empirical_counts: Option<Arc<[u64]>>,
}

impl ExactDiscreteLaw {
    /// Validate a complete observational or hard-intervention law.
    ///
    /// # Errors
    /// Invalid identities, domains, shape, interventions, or probability mass.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        population: impl Into<Arc<str>>,
        regime: RegimeId,
        interventions: impl Into<Arc<[InterventionAssignment]>>,
        axes: impl Into<Arc<[DiscreteAxis]>>,
        probabilities: impl Into<Arc<[f64]>>,
        snapshot_identity: impl Into<Arc<str>>,
        tolerance: LawTolerance,
    ) -> Result<Self, ExactLawError> {
        let mut law = Self {
            population: population.into(),
            regime,
            interventions: interventions.into(),
            axes: axes.into(),
            probabilities: probabilities.into(),
            strides: Arc::from([]),
            axis_index: Arc::new(BTreeMap::new()),
            level_index: Arc::from([]),
            snapshot_identity: snapshot_identity.into(),
            tolerance,
            origin: LawOrigin::SuppliedExact,
            empirical_counts: None,
        };
        if law.population.is_empty() || law.snapshot_identity.is_empty() || !tolerance.valid() {
            return Err(law.error("invalid_law_metadata"));
        }
        let mut intervened = HashSet::new();
        for assignment in law.interventions.iter() {
            if !finite(&assignment.value) || !intervened.insert(assignment.variable) {
                return Err(law.error("invalid_law_intervention"));
            }
        }
        let mut size = 1usize;
        let mut strides = vec![0; law.axes.len()];
        let mut axis_index = BTreeMap::new();
        let mut level_index = vec![HashMap::new(); law.axes.len()];
        for (i, axis) in law.axes.iter().enumerate().rev() {
            let levels = &mut level_index[i];
            if axis.values.is_empty()
                || axis_index.insert(axis.variable, i).is_some()
                || axis
                    .values
                    .iter()
                    .enumerate()
                    .any(|(index, v)| !finite(v) || levels.insert(value_key(v), index).is_some())
                || intervened.contains(&axis.variable)
            {
                return Err(law.error("invalid_law_axis"));
            }
            strides[i] = size;
            size = size
                .checked_mul(axis.values.len())
                .ok_or_else(|| law.error("law_size_overflow"))?;
        }
        if size != law.probabilities.len() {
            return Err(law.error("incomplete_law_domain"));
        }
        if law.probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
            return Err(law.error("invalid_law_probability"));
        }
        if !tolerance.unit_mass(sum(law.probabilities.iter().copied())) {
            return Err(law.error("unnormalized_law"));
        }
        law.strides = strides.into();
        law.axis_index = Arc::new(axis_index);
        law.level_index = level_index.into();
        Ok(law)
    }

    /// Frequency plug-in joint. Empty cells are sampling zeros, not structural zeros.
    ///
    /// # Errors
    /// Same validation as [`Self::try_new`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_empirical(
        population: impl Into<Arc<str>>,
        regime: RegimeId,
        interventions: impl Into<Arc<[InterventionAssignment]>>,
        axes: impl Into<Arc<[DiscreteAxis]>>,
        probabilities: impl Into<Arc<[f64]>>,
        snapshot_identity: impl Into<Arc<str>>,
        tolerance: LawTolerance,
    ) -> Result<Self, ExactLawError> {
        let mut law = Self::try_new(
            population,
            regime,
            interventions,
            axes,
            probabilities,
            snapshot_identity,
            tolerance,
        )?;
        law.origin = LawOrigin::EmpiricalPlugin;
        Ok(law)
    }
    /// Mark a fitted law as model-based and retain its observed support.
    /// # Errors
    /// Counts with wrong shape, no observations, or overflow.
    pub fn with_empirical_counts(mut self, counts: Vec<u64>) -> Result<Self, ExactLawError> {
        if counts.len() != self.probabilities.len()
            || counts.iter().try_fold(0u64, |n, k| n.checked_add(*k)).is_none_or(|n| n == 0)
        {
            return Err(self.error("invalid_empirical_support"));
        }
        self.origin = LawOrigin::LearnedPlugin;
        self.empirical_counts = Some(counts.into());
        Ok(self)
    }
    /// Observed cell counts for a model-based law; fitted mass never certifies support.
    #[must_use]
    pub fn empirical_counts(&self) -> Option<&[u64]> {
        self.empirical_counts.as_deref()
    }

    /// Population identity.
    #[must_use]
    pub fn population(&self) -> &str {
        &self.population
    }
    /// Catalog regime identity.
    #[must_use]
    pub const fn regime(&self) -> RegimeId {
        self.regime
    }
    /// Physical snapshot identity (not identification authority).
    #[must_use]
    pub fn snapshot_identity(&self) -> &str {
        &self.snapshot_identity
    }
    /// Ordered dense axes.
    #[must_use]
    pub fn axes(&self) -> &[DiscreteAxis] {
        &self.axes
    }
    /// Immutable dense probabilities.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    /// Concrete intervention assignment.
    #[must_use]
    pub fn interventions(&self) -> &[InterventionAssignment] {
        &self.interventions
    }
    /// Declared validation tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> LawTolerance {
        self.tolerance
    }
    /// Whether this joint is a supplied law or an empirical plug-in.
    #[must_use]
    pub const fn origin(&self) -> LawOrigin {
        self.origin
    }

    fn error(&self, kind: &'static str) -> ExactLawError {
        ExactLawError {
            kind,
            population: self.population.clone(),
            regime: Some(self.regime),
            variables: Arc::from([]),
            conditioning: Arc::from([]),
            interventions: self.interventions.clone(),
        }
    }
    fn covers(&self, spec: &FactorSpec<'_>) -> bool {
        self.population.as_ref() == spec.population
            && spec.regime == Some(self.regime)
            && self.interventions.len() == spec.intervention.len()
            && spec.intervention.iter().all(|a| {
                self.interventions
                    .iter()
                    .any(|b| a.variable == b.variable && value_key(&a.value) == value_key(&b.value))
            })
            && (spec.domain == DomainRef::Observational) == self.interventions.is_empty()
            && spec
                .variables
                .iter()
                .chain(spec.conditioned_on)
                .all(|v| self.axis_index.contains_key(v))
    }
    fn positions(
        &self,
        vars: &[VariableId],
        values: &Assignment,
    ) -> Result<Vec<(usize, usize)>, ExactLawError> {
        vars.iter()
            .map(|v| {
                let axis = *self.axis_index.get(v).ok_or_else(|| self.error("missing_law_axis"))?;
                let value = values.get(*v).ok_or_else(|| self.error("missing_assignment"))?;
                let level = *self.level_index[axis]
                    .get(&value_key(value))
                    .ok_or_else(|| self.error("assignment_outside_domain"))?;
                Ok((axis, level))
            })
            .collect()
    }
    fn mass(&self, constraints: &[(usize, usize)]) -> f64 {
        if constraints.len() == self.axes.len()
            && constraints
                .iter()
                .map(|(axis, _)| *axis)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == self.axes.len()
        {
            let index =
                constraints.iter().map(|(axis, level)| self.strides[*axis] * level).sum::<usize>();
            return self.probabilities[index];
        }
        sum(self.probabilities.iter().enumerate().filter_map(|(i, p)| {
            constraints
                .iter()
                .all(|(a, level)| (i / self.strides[*a]) % self.axes[*a].values.len() == *level)
                .then_some(*p)
        }))
    }
}

fn finite(v: &Value) -> bool {
    !matches!(v, Value::Float64(x) if !x.is_finite())
}

// Compensated summation avoids normalization drift on large, valid domains.
pub(crate) fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut correction) = (0.0, 0.0);
    for value in values {
        let next = total + value;
        correction += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    total + correction
}

type WorldIndex = HashMap<Arc<str>, HashMap<RegimeId, HashMap<Vec<InterventionAssignment>, usize>>>;

// Normalize exactly represented integral numeric levels locally, without changing
// core Value equality or the original table/serialized coordinate representation.
#[allow(clippy::cast_possible_truncation)] // Finite integral f64, explicitly within the exact i64 range.
pub(crate) fn value_key(value: &Value) -> Value {
    match value {
        Value::Float64(x)
            if x.is_finite() && x.fract() == 0.0 && x.abs() <= 9_007_199_254_740_992.0 =>
        {
            Value::Int64(*x as i64)
        }
        other => other.clone(),
    }
}
fn world_key(assignments: &[InterventionAssignment]) -> Vec<InterventionAssignment> {
    let mut key: Vec<_> = assignments
        .iter()
        .map(|a| InterventionAssignment { variable: a.variable, value: value_key(&a.value) })
        .collect();
    key.sort_by_key(|a| a.variable);
    key
}

fn lookup_world_key(assignments: &[InterventionAssignment]) -> Cow<'_, [InterventionAssignment]> {
    let ordered = assignments.windows(2).all(|pair| pair[0].variable < pair[1].variable);
    let normalized = assignments.iter().all(|a| value_key(&a.value) == a.value);
    if ordered && normalized {
        Cow::Borrowed(assignments)
    } else {
        Cow::Owned(world_key(assignments))
    }
}

type JointQueryKey = (usize, Vec<(usize, usize)>, Vec<(usize, usize)>);
#[derive(Debug)]
struct SharedFactorCache {
    capacity: usize,
    values: Mutex<HashMap<JointQueryKey, f64>>,
}

/// Immutable collection of exact laws with bounded support materialization.
#[derive(Clone, Debug)]
pub struct ExactTransportData {
    laws: Arc<[ExactDiscreteLaw]>,
    index: Arc<WorldIndex>,
    domains: Arc<BTreeMap<VariableId, Arc<[Value]>>>,
    max_support_rows: usize,
    factor_cache: Option<Arc<SharedFactorCache>>,
}

impl ExactTransportData {
    /// Validate compatible domains and unambiguous population/regime/world bindings.
    ///
    /// # Errors
    /// Duplicate world bindings, incompatible domains, or an empty resource budget.
    pub fn try_new(
        laws: impl Into<Arc<[ExactDiscreteLaw]>>,
        max_support_rows: usize,
    ) -> Result<Self, ExactLawError> {
        let laws = laws.into();
        let error = |kind| ExactLawError {
            kind,
            population: Arc::from(""),
            regime: None,
            variables: Arc::from([]),
            conditioning: Arc::from([]),
            interventions: Arc::from([]),
        };
        if max_support_rows == 0 {
            return Err(error("support_budget"));
        }
        let mut index = WorldIndex::new();
        let mut domains = BTreeMap::<VariableId, Arc<[Value]>>::new();
        let mut domain_keys = HashMap::<VariableId, HashSet<Value>>::new();
        for (i, law) in laws.iter().enumerate() {
            if index
                .entry(law.population.clone())
                .or_default()
                .entry(law.regime)
                .or_default()
                .insert(world_key(&law.interventions), i)
                .is_some()
            {
                return Err(law.error("duplicate_law_binding"));
            }
            for axis in law.axes.iter() {
                if let Some(other) = domains.get(&axis.variable) {
                    if axis.values.len() != other.len()
                        || !axis
                            .values
                            .iter()
                            .all(|v| domain_keys[&axis.variable].contains(&value_key(v)))
                    {
                        return Err(law.error("incompatible_law_domains"));
                    }
                } else {
                    domains.insert(axis.variable, axis.values.clone());
                    domain_keys.insert(axis.variable, axis.values.iter().map(value_key).collect());
                }
            }
        }
        Ok(Self {
            laws,
            index: Arc::new(index),
            domains: Arc::new(domains),
            max_support_rows,
            factor_cache: None,
        })
    }
    /// Share a bounded factor-value cache across plans bound to this immutable provider.
    /// Replacing data creates a new cache; cached values never cross snapshots.
    #[must_use]
    pub fn with_shared_factor_cache(mut self, capacity: usize) -> Self {
        if capacity == 0 {
            self.factor_cache = None;
            return self;
        }
        self.factor_cache =
            Some(Arc::new(SharedFactorCache { capacity, values: Mutex::new(HashMap::new()) }));
        self
    }
    pub(crate) fn factor_cache_bytes(&self) -> Option<usize> {
        let width = self.laws.iter().map(|law| law.axes.len()).max().unwrap_or(0);
        self.factor_cache.as_ref().map_or(Some(0), |cache| {
            width.checked_mul(64)?.checked_add(256)?.checked_mul(cache.capacity)
        })
    }
    /// Declared finite support of a coordinate, without materializing a Cartesian table.
    #[must_use]
    pub fn domain(&self, variable: VariableId) -> Option<&[Value]> {
        self.domains.get(&variable).map(AsRef::as_ref)
    }

    /// Validated laws, without callbacks or data access.
    #[must_use]
    pub fn laws(&self) -> &[ExactDiscreteLaw] {
        &self.laws
    }

    /// Resolve a requested factor before execution. Regime binding must be explicit.
    ///
    /// # Errors
    /// Missing or incompatible population, regime, intervention world, or axes.
    pub fn require_factor(
        &self,
        spec: &FactorSpec<'_>,
    ) -> Result<&ExactDiscreteLaw, ExactLawError> {
        self.index
            .get(spec.population)
            .and_then(|regimes| spec.regime.and_then(|regime| regimes.get(&regime)))
            .and_then(|worlds| worlds.get(lookup_world_key(spec.intervention).as_ref()))
            .map(|index| &self.laws[*index])
            .filter(|law| law.covers(spec))
            .ok_or_else(|| ExactLawError {
                kind: "missing_exact_provider",
                population: Arc::from(spec.population),
                regime: spec.regime,
                variables: Arc::from(spec.variables),
                conditioning: Arc::from([]),
                interventions: Arc::from(spec.intervention),
            })
    }

    /// Evaluate a marginal or conditional from the selected joint law.
    ///
    /// # Errors
    /// Missing coverage, an out-of-domain assignment, or a zero denominator.
    pub fn probability_checked(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
    ) -> Result<f64, ExactLawError> {
        let law = self.require_factor(spec)?;
        let locate = |mut error: ExactLawError| {
            error.variables = Arc::from(spec.variables);
            error.conditioning = spec
                .conditioned_on
                .iter()
                .filter_map(|v| assignment.get(*v).map(|x| (*v, x.clone())))
                .collect();
            error
        };
        let mut conditions = law.positions(spec.conditioned_on, assignment).map_err(locate)?;
        let outputs = law.positions(spec.variables, assignment).map_err(locate)?;
        let key = self.factor_cache.as_ref().map(|_| {
            let index = self.index[spec.population][&law.regime]
                [lookup_world_key(spec.intervention).as_ref()];
            (index, outputs.clone(), conditions.clone())
        });
        if let (Some(cache), Some(key)) = (&self.factor_cache, &key) {
            if let Some(value) =
                cache.values.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(key)
            {
                return Ok(*value);
            }
        }
        if let Some(counts) = &law.empirical_counts {
            let observed = counts.iter().enumerate().any(|(i, count)| {
                *count > 0
                    && conditions.iter().all(|(axis, level)| {
                        (i / law.strides[*axis]) % law.axes[*axis].values.len() == *level
                    })
            });
            if !observed {
                return Err(locate(law.error("sampling_zero")));
            }
        }
        let denominator = if conditions.is_empty() { 1.0 } else { law.mass(&conditions) };
        if denominator == 0.0 {
            return Err(locate(law.error(if law.origin != LawOrigin::SuppliedExact {
                "sampling_zero"
            } else {
                "zero_conditioning_mass"
            })));
        }
        conditions.extend(outputs);
        let value = law.mass(&conditions) / denominator;
        if let (Some(cache), Some(key)) = (&self.factor_cache, key) {
            let mut values = cache.values.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if values.len() < cache.capacity {
                values.insert(key, value);
            }
        }
        Ok(value)
    }
}

impl DistributionProvider for ExactTransportData {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.probability_checked(spec, assignment).map_err(|e| EvalError::ExactLaw(Box::new(e)))
    }
    fn support(
        &self,
        vars: &[VariableId],
        _: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        let mut domains = Vec::with_capacity(vars.len());
        let mut size = 1usize;
        for (i, variable) in vars.iter().enumerate() {
            if vars[..i].contains(variable) {
                return Err(EvalError::ProviderKind("duplicate support coordinate"));
            }
            let domain = self.domains.get(variable).ok_or(EvalError::EmptySupport(*variable))?;
            size = size
                .checked_mul(domain.len())
                .filter(|n| *n <= self.max_support_rows)
                .ok_or(EvalError::ProviderKind("exact support budget exceeded"))?;
            domains.push(domain);
        }
        let mut rows = Vec::with_capacity(size);
        for index in 0..size {
            let mut remainder = index;
            let mut row = vec![Value::Bool(false); domains.len()];
            for (i, domain) in domains.iter().enumerate().rev() {
                row[i] = domain[remainder % domain.len()].clone();
                remainder /= domain.len();
            }
            rows.push(Arc::from(row));
        }
        Ok(rows.into())
    }
    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        assignment.get(var).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(var))
    }
    fn n_draws(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }
    fn axis(i: u32) -> DiscreteAxis {
        DiscreteAxis { variable: v(i), values: Arc::from([Value::Int64(0), Value::Int64(1)]) }
    }
    fn law(pop: &str, probabilities: &[f64]) -> Result<ExactDiscreteLaw, ExactLawError> {
        ExactDiscreteLaw::try_new(
            pop,
            RegimeId::from_raw(0),
            [],
            [axis(0), axis(1)],
            probabilities,
            "snapshot",
            LawTolerance::default(),
        )
    }
    #[test]
    fn validates_dense_law_without_repair() {
        for probabilities in [vec![1.0], vec![0.0; 4], vec![f64::NAN; 4], vec![-0.1, 0.3, 0.3, 0.5]]
        {
            assert!(law("source", &probabilities).is_err());
        }
        let probabilities = [0.1, 0.2, 0.3, 0.400_000_000_001];
        let table = law("source", &probabilities).unwrap();
        assert_eq!(table.probabilities(), &probabilities);
        assert!(
            ExactDiscreteLaw::try_new(
                "p",
                RegimeId::from_raw(0),
                [],
                [axis(0), axis(0)],
                [0.25; 4],
                "s",
                LawTolerance::default()
            )
            .is_err()
        );
    }
    #[test]
    fn conditions_and_marginalizes_by_axis_coordinate() {
        let data = ExactTransportData::try_new([law("target", &[0.1, 0.2, 0.3, 0.4]).unwrap()], 16)
            .unwrap();
        let outcomes = [v(1)];
        let conditions = [v(0)];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(0)),
            ..FactorSpec::new(&outcomes, &conditions, &[], DomainRef::Observational)
        };
        let assignment = Assignment::from_pairs([(v(0), Value::Int64(1)), (v(1), Value::Int64(0))]);
        assert!((data.probability_checked(&spec, &assignment).unwrap() - 3.0 / 7.0).abs() < 1e-14);
        let wrong_population = FactorSpec { population: "source", ..spec.clone() };
        assert_eq!(
            data.require_factor(&wrong_population).unwrap_err().kind,
            "missing_exact_provider"
        );
        let missing_regime = FactorSpec { regime: None, ..spec };
        assert!(data.require_factor(&missing_regime).is_err());
    }
    #[test]
    fn empirical_empty_conditioner_is_sampling_zero_not_structural() {
        let table = ExactDiscreteLaw::try_empirical(
            "target",
            RegimeId::from_raw(0),
            [],
            [axis(0), axis(1)],
            [0.0, 0.0, 0.3, 0.7],
            "snapshot",
            LawTolerance::default(),
        )
        .unwrap();
        assert_eq!(table.origin(), LawOrigin::EmpiricalPlugin);
        let data = ExactTransportData::try_new([table], 16).unwrap();
        let outcomes = [v(1)];
        let conditions = [v(0)];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(0)),
            ..FactorSpec::new(&outcomes, &conditions, &[], DomainRef::Observational)
        };
        let assignment = Assignment::from_pairs([(v(0), Value::Int64(0)), (v(1), Value::Int64(1))]);
        assert_eq!(data.probability_checked(&spec, &assignment).unwrap_err().kind, "sampling_zero");
    }
    #[test]
    fn positive_learned_probabilities_do_not_authorize_empty_empirical_conditioners() {
        let predicted = law("target", &[0.25, 0.25, 0.25, 0.25])
            .unwrap()
            .with_empirical_counts(vec![0, 0, 3, 7])
            .unwrap();
        let data = ExactTransportData::try_new([predicted], 16).unwrap();
        let outcomes = [v(1)];
        let conditions = [v(0)];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(0)),
            ..FactorSpec::new(&outcomes, &conditions, &[], DomainRef::Observational)
        };
        let assignment = Assignment::from_pairs([(v(0), Value::Int64(0)), (v(1), Value::Int64(1))]);
        assert_eq!(data.probability_checked(&spec, &assignment).unwrap_err().kind, "sampling_zero");
    }
    #[test]
    fn locates_zero_condition_and_preserves_structural_zeros() {
        let data = ExactTransportData::try_new([law("target", &[0.0, 0.0, 0.3, 0.7]).unwrap()], 16)
            .unwrap();
        let outcomes = [v(1)];
        let conditions = [v(0)];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(0)),
            ..FactorSpec::new(&outcomes, &conditions, &[], DomainRef::Observational)
        };
        let assignment = Assignment::from_pairs([(v(0), Value::Int64(0)), (v(1), Value::Int64(1))]);
        let error = data.probability_checked(&spec, &assignment).unwrap_err();
        assert_eq!(error.kind, "zero_conditioning_mass");
        assert_eq!(&*error.conditioning, &[(v(0), Value::Int64(0))]);
        let marginal = FactorSpec { variables: &[v(0)], conditioned_on: &[], ..spec };
        assert!(data.probability_checked(&marginal, &assignment).unwrap().abs() < f64::EPSILON);
    }
    #[test]
    fn requires_exact_intervention_world_and_bounds_support() {
        let table = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(2),
            [InterventionAssignment { variable: v(0), value: Value::Int64(1) }],
            [axis(1)],
            [0.4, 0.6],
            "s",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([table], 1).unwrap();
        let intervention = [InterventionAssignment { variable: v(0), value: Value::Int64(0) }];
        let outcomes = [v(1)];
        let conditions = [v(0)];
        let spec = FactorSpec {
            population: "source",
            regime: Some(RegimeId::from_raw(2)),
            ..FactorSpec::new(&outcomes, &conditions[..0], &intervention, DomainRef::Interventional)
        };
        assert!(data.require_factor(&spec).is_err());
        assert!(data.support(&[v(1)], &EvalContext::default()).is_err());
    }
}

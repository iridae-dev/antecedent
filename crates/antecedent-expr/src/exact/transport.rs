//! Bounded exact-law transport and factor evaluation.

use super::{
    AxisCursor, ExactDiscreteLaw, ExactLawError, LawOrigin, WorldBinding, WorldIndex,
    ZERO_CONDITIONING_MASS, checked_world_eq, checked_world_key, poll_world, value_key,
    world_digest,
};
use crate::{Assignment, DistributionProvider, DomainRef, EvalContext, EvalError, FactorSpec};
use antecedent_core::{RegimeId, Value, VariableId};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

type JointQueryKey =
    (antecedent_core::recalc::StageIdentity, Vec<(usize, usize)>, Vec<(usize, usize)>);
#[derive(Debug)]
struct SharedFactorCache {
    capacity: usize,
    values: Mutex<HashMap<JointQueryKey, f64>>,
}

/// Immutable collection of exact laws with bounded support materialization.
#[derive(Clone, Debug)]
pub struct ExactTransportData {
    laws: Arc<[ExactDiscreteLaw]>,
    law_identities: Arc<[antecedent_core::recalc::StageIdentity]>,
    pub(super) index: Arc<WorldIndex>,
    domains: Arc<BTreeMap<VariableId, Arc<[Value]>>>,
    max_support_rows: usize,
    factor_cache: Option<Arc<SharedFactorCache>>,
    /// The cited regimes among which a leaf that names no regime selects its
    /// law by intervention world; `None` keeps every leaf regime-bound.
    world_bound_regimes: Option<Arc<[RegimeId]>>,
}

impl ExactTransportData {
    /// Maximum number of support rows retained for factor evaluation.
    #[must_use]
    pub const fn max_support_rows(&self) -> usize {
        self.max_support_rows
    }

    /// Validate compatible domains and unambiguous population/regime/world bindings.
    ///
    /// # Errors
    /// Duplicate world bindings, incompatible domains, or an empty resource budget.
    pub fn try_new(
        laws: impl Into<Arc<[ExactDiscreteLaw]>>,
        max_support_rows: usize,
    ) -> Result<Self, ExactLawError> {
        let error = |kind| ExactLawError {
            kind,
            population: Arc::from(""),
            regime: None,
            variables: Arc::from([]),
            conditioning: Arc::from([]),
            interventions: Arc::from([]),
        };
        if antecedent_core::numerical_admission::is_cancelled() {
            return Err(error("cancelled"));
        }
        let laws = laws.into();
        if antecedent_core::numerical_admission::is_cancelled() {
            return Err(error("cancelled"));
        }
        if max_support_rows == 0 {
            return Err(error("support_budget"));
        }
        let mut index = WorldIndex::new();
        let mut domains = BTreeMap::<VariableId, Arc<[Value]>>::new();
        let mut domain_keys = HashMap::<VariableId, HashSet<Value>>::new();
        for (i, law) in laws.iter().enumerate() {
            law.check_cancelled()?;
            let key = checked_world_key(&law.interventions)?;
            let digest = world_digest(&law.population, law.regime, &key)?;
            let bucket = index.entry(digest).or_default();
            for (binding, _) in bucket.iter() {
                if checked_world_eq(binding, &law.population, law.regime, &key)? {
                    return Err(law.error("duplicate_law_binding"));
                }
            }
            bucket.push((
                WorldBinding {
                    population: Arc::clone(&law.population),
                    regime: law.regime,
                    interventions: key,
                },
                i,
            ));
            for axis in law.axes.iter() {
                law.check_cancelled()?;
                if let Some(other) = domains.get(&axis.variable) {
                    if axis.values.len() != other.len() {
                        return Err(law.error("incompatible_law_domains"));
                    }
                    for chunk in axis.values.chunks(1024) {
                        law.check_cancelled()?;
                        if !chunk
                            .iter()
                            .all(|v| domain_keys[&axis.variable].contains(&value_key(v)))
                        {
                            return Err(law.error("incompatible_law_domains"));
                        }
                    }
                } else {
                    let mut keys = HashSet::new();
                    for chunk in axis.values.chunks(1024) {
                        law.check_cancelled()?;
                        keys.extend(chunk.iter().map(value_key));
                    }
                    domains.insert(axis.variable, axis.values.clone());
                    domain_keys.insert(axis.variable, keys);
                }
            }
        }
        let law_identities = laws
            .iter()
            .map(ExactDiscreteLaw::content_identity_checked)
            .collect::<Result<Vec<_>, _>>()?
            .into();
        crate::execution_counts::note_static_work(
            crate::execution_counts::StaticWork::ProviderBinding,
        );
        Ok(Self {
            laws,
            law_identities,
            index: Arc::new(index),
            domains: Arc::new(domains),
            max_support_rows,
            factor_cache: None,
            world_bound_regimes: None,
        })
    }
    /// Validate the current laws and reuse only content-bound cached values from a prior
    /// provider. Changed source laws cannot alias unchanged source values, even if labels,
    /// row ordering or caller-supplied snapshot names coincide.
    /// # Errors
    /// Incompatible laws, domains or support budget, as for [`Self::try_new`].
    pub fn rebind_reusing_factor_cache(
        &self,
        previous: Option<&Self>,
        capacity: usize,
    ) -> Result<Self, ExactLawError> {
        let mut rebound = Self::try_new(self.laws.clone(), self.max_support_rows)?;
        rebound.world_bound_regimes.clone_from(&self.world_bound_regimes);
        rebound.factor_cache = previous.and_then(|old| old.factor_cache.clone());
        if rebound.factor_cache.is_none() {
            rebound = rebound.with_shared_factor_cache(capacity);
        }
        Ok(rebound)
    }
    /// Content identities of actual population/regime/world/snapshot laws.
    #[must_use]
    pub fn law_identities(&self) -> &[antecedent_core::recalc::StageIdentity] {
        &self.law_identities
    }

    /// Let a leaf that names no regime select its law by intervention world.
    ///
    /// A z-transport factor whose exchanged coordinate is bound at evaluation
    /// (`do(x = 0, z)` with `z` the enclosing summation variable) cites one
    /// regime per world rather than one regime for the whole family, so the
    /// bound leaf carries no regime and the concrete world selects the law.
    /// The law is selected among the `cited` regimes only, so laws of regimes
    /// the binding did not choose never compete; when several cited regimes
    /// still supply the world, the lowest regime id is used, the binding's own
    /// rule. Every other provider keeps requiring an explicit regime.
    #[must_use]
    pub fn with_world_bound_leaves(mut self, cited: impl Into<Arc<[RegimeId]>>) -> Self {
        self.world_bound_regimes = Some(cited.into());
        self
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
        self.require_factor_index(spec).map(|index| &self.laws[index])
    }

    fn require_factor_index(&self, spec: &FactorSpec<'_>) -> Result<usize, ExactLawError> {
        let wanted = checked_world_key(spec.intervention)?;
        let mut selected: Option<usize> = None;
        let mut consider = |binding: &WorldBinding, index: usize| -> Result<(), ExactLawError> {
            if !checked_world_eq(
                binding,
                spec.population,
                spec.regime.unwrap_or(binding.regime),
                &wanted,
            )? {
                return Ok(());
            }
            let law = &self.laws[index];
            if (spec.domain == DomainRef::Observational) != law.interventions.is_empty() {
                return Ok(());
            }
            for variable in spec.variables.iter().chain(spec.conditioned_on) {
                poll_world()?;
                if !law.axis_index.contains_key(variable) {
                    return Ok(());
                }
            }
            if selected.is_none_or(|old| law.regime.raw() < self.laws[old].regime.raw()) {
                selected = Some(index);
            }
            Ok(())
        };
        if let Some(regime) = spec.regime {
            let digest = world_digest(spec.population, regime, &wanted)?;
            if let Some(bucket) = self.index.get(&digest) {
                for (binding, index) in bucket {
                    consider(binding, *index)?;
                }
            }
        } else if let Some(cited) = &self.world_bound_regimes {
            for bucket in self.index.values() {
                for (binding, index) in bucket {
                    poll_world()?;
                    let mut allowed = false;
                    for chunk in cited.chunks(512) {
                        poll_world()?;
                        if chunk.contains(&binding.regime) {
                            allowed = true;
                            break;
                        }
                    }
                    if allowed {
                        consider(binding, *index)?;
                    }
                }
            }
        }
        poll_world()?;
        if let Some(index) = selected {
            Ok(index)
        } else {
            let population = Arc::from(spec.population);
            poll_world()?;
            let variables = Arc::from(spec.variables);
            poll_world()?;
            let interventions = Arc::from(spec.intervention);
            poll_world()?;
            Err(ExactLawError {
                kind: "missing_exact_provider",
                population,
                regime: spec.regime,
                variables,
                conditioning: Arc::from([]),
                interventions,
            })
        }
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
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::ProviderInvocation,
            || {
                crate::execution_counts::note_static_work(
                    crate::execution_counts::StaticWork::ProviderCall,
                );
                let law_index = self.require_factor_index(spec)?;
                let law = &self.laws[law_index];
                let locate = |mut error: ExactLawError| {
                    error.variables = Arc::from(spec.variables);
                    error.conditioning = spec
                        .conditioned_on
                        .iter()
                        .filter_map(|v| assignment.get(*v).map(|x| (*v, x.clone())))
                        .collect();
                    error
                };
                let mut conditions =
                    law.positions(spec.conditioned_on, assignment).map_err(locate)?;
                let outputs = law.positions(spec.variables, assignment).map_err(locate)?;
                let key = self
                    .factor_cache
                    .as_ref()
                    .map(|_| (self.law_identities[law_index], outputs.clone(), conditions.clone()));
                if let (Some(cache), Some(key)) = (&self.factor_cache, &key)
                    && let Some(value) = cache
                        .values
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(key)
                {
                    return Ok(*value);
                }
                let factor_attempt = antecedent_core::execution_attempt::OperationGuard::begin(
                    antecedent_core::execution_attempt::Operation::FactorEvaluation,
                );
                crate::execution_counts::note_static_work(
                    crate::execution_counts::StaticWork::FactorEvaluation,
                );
                if let Some(counts) = &law.empirical_counts {
                    let mut cursors: Vec<AxisCursor> =
                        conditions.iter().map(|(axis, _)| AxisCursor::new(law, *axis)).collect();
                    let mut observed = false;
                    for count in counts.iter() {
                        if *count > 0
                            && cursors
                                .iter()
                                .zip(&conditions)
                                .all(|(c, (_, level))| c.level == *level)
                        {
                            observed = true;
                            break;
                        }
                        for cursor in &mut cursors {
                            cursor.advance();
                        }
                    }
                    if !observed {
                        return Err(locate(law.error("sampling_zero")));
                    }
                }
                let denominator = if conditions.is_empty() { 1.0 } else { law.mass(&conditions) };
                if denominator == 0.0 {
                    return Err(locate(law.error(if law.origin == LawOrigin::SuppliedExact {
                        ZERO_CONDITIONING_MASS
                    } else {
                        "sampling_zero"
                    })));
                }
                conditions.extend(outputs);
                let value = law.mass(&conditions) / denominator;
                if let (Some(cache), Some(key)) = (&self.factor_cache, key) {
                    let mut values =
                        cache.values.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    if values.len() < cache.capacity {
                        values.insert(key, value);
                    }
                }
                factor_attempt.complete();
                Ok(value)
            },
        )
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

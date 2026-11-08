//! Validated finite-discrete laws. These inputs are laws, not sample frequencies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod transport;

pub use transport::ExactTransportData;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use antecedent_core::{RegimeId, Value, VariableId};

pub(crate) use antecedent_kernels::numeric::CompensatedSum;

/// Preserve `Debug`'s bytes while polling even when a formatter writes a long string at once.
struct CheckedDescriptor<F> {
    text: String,
    check: F,
}

impl<F: FnMut() -> bool> std::fmt::Write for CheckedDescriptor<F> {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let mut start = 0;
        while start < text.len() {
            if !(self.check)() {
                return Err(std::fmt::Error);
            }
            let mut end = start.saturating_add(4096).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            self.text.push_str(&text[start..end]);
            start = end;
        }
        Ok(())
    }
}

use crate::{Assignment, InterventionAssignment};
#[cfg(test)]
use crate::{DistributionProvider, DomainRef, EvalContext, EvalError, FactorSpec};

/// How a dense joint was obtained. Structural zeros are supported only for supplied laws.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LawOrigin {
    /// Caller-supplied complete population law. Zero cells may be structural.
    SuppliedExact,
    /// Frequency plug-in from a finite sample. Zero cells are unobserved, not structural.
    EmpiricalPlugin,
    /// Coherent learned finite law with retained empirical counts.
    LearnedPlugin,
    /// Draw from a declared posterior law; zero cells are posterior outcomes, not structural zeros.
    BayesianPosterior,
}

impl LawOrigin {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SuppliedExact => "supplied_exact",
            Self::EmpiricalPlugin => "empirical_plugin",
            Self::LearnedPlugin => "learned_plugin",
            Self::BayesianPosterior => "bayesian_posterior",
        }
    }

    /// Parse a stable origin name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "supplied_exact" | "" => Some(Self::SuppliedExact),
            "empirical_plugin" => Some(Self::EmpiricalPlugin),
            "learned_plugin" => Some(Self::LearnedPlugin),
            "bayesian_posterior" => Some(Self::BayesianPosterior),
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

/// Failure kind of a conditional on an event the law gives zero mass. For a supplied law this
/// is a structural null event, which an exact evaluation may extend across; for a fitted or
/// empirical law the same situation is reported as `sampling_zero` instead.
pub(crate) const ZERO_CONDITIONING_MASS: &str = "zero_conditioning_mass";

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

impl ExactLawError {
    /// Whether the requested population/regime/world has no covering law.
    /// This is provider availability, not an unsupported conditional cell.
    #[must_use]
    pub const fn is_missing_provider(&self) -> bool {
        let expected = b"missing_exact_provider";
        let actual = self.kind.as_bytes();
        if actual.len() != expected.len() {
            return false;
        }
        let mut index = 0;
        while index < expected.len() {
            if actual[index] != expected[index] {
                return false;
            }
            index += 1;
        }
        true
    }
    /// Whether provider setup or lookup was cancelled.
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        let actual = self.kind.as_bytes();
        let expected = b"cancelled";
        if actual.len() != expected.len() {
            return false;
        }
        let mut i = 0;
        while i < expected.len() {
            if actual[i] != expected[i] {
                return false;
            }
            i += 1;
        }
        true
    }
    /// Whether this is a conditional on a structurally null event (see
    /// `ZERO_CONDITIONING_MASS`), the only failure an exact evaluation may extend across.
    #[must_use]
    pub fn is_null_conditioning_event(&self) -> bool {
        self.kind == ZERO_CONDITIONING_MASS
    }
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
    /// Marginal mass tables per sorted axis subset, built on first use. The probabilities are
    /// immutable, so a table never goes stale and is shared by clones of the law.
    marginals: Arc<MarginalTables>,
}

/// Most axis subsets whose marginal table one law retains. A full shard evicts its
/// oldest table, so a later subset can still be cached.
const MAX_MARGINAL_TABLES: usize = 256;
const MARGINAL_SHARDS: usize = 8;

/// One shard of the marginal-table cache, keyed by the strictly ascending axis list itself so
/// any number of axes is cacheable and a hit allocates nothing. `order` maps each entry's last
/// use tick to its key, so the least recently used entry is the first of the map.
type MarginalCacheEntries = HashMap<Arc<[usize]>, (u64, Arc<[f64]>)>;

#[derive(Debug)]
struct MarginalShard {
    entries: MarginalCacheEntries,
    order: BTreeMap<u64, Arc<[usize]>>,
    clock: u64,
}

impl MarginalShard {
    fn new() -> Self {
        Self { entries: HashMap::new(), order: BTreeMap::new(), clock: 0 }
    }

    /// The cached table for `axes`, marking it most recently used.
    fn touch(&mut self, axes: &[usize]) -> Option<Arc<[f64]>> {
        let next = self.clock.wrapping_add(1);
        let (previous, table) = {
            let (tick, table) = self.entries.get_mut(axes)?;
            let previous = std::mem::replace(tick, next);
            (previous, Arc::clone(table))
        };
        self.clock = next;
        if let Some(key) = self.order.remove(&previous) {
            self.order.insert(next, key);
        }
        Some(table)
    }

    /// Retain `table` for `axes`, evicting the least recently used entry at `capacity`.
    fn insert(&mut self, axes: &[usize], table: &Arc<[f64]>, capacity: usize) {
        if self.entries.contains_key(axes) {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        let tick = self.clock;
        if self.entries.len() >= capacity
            && let Some((_, oldest)) = self.order.pop_first()
        {
            self.entries.remove(&*oldest);
        }
        let key: Arc<[usize]> = Arc::from(axes);
        self.order.insert(tick, Arc::clone(&key));
        self.entries.insert(key, (tick, Arc::clone(table)));
    }
}

#[derive(Debug)]
struct MarginalTables {
    shards: [Mutex<MarginalShard>; MARGINAL_SHARDS],
}

impl Default for MarginalTables {
    fn default() -> Self {
        Self { shards: std::array::from_fn(|_| Mutex::new(MarginalShard::new())) }
    }
}

impl ExactDiscreteLaw {
    /// Validate a complete observational or hard-intervention law.
    ///
    /// # Errors
    /// Invalid identities, domains, shape, interventions, or probability mass.
    #[allow(
        clippy::too_many_lines,
        reason = "Construction validates the whole discrete law before admitting it."
    )]
    pub fn try_new(
        population: impl Into<Arc<str>>,
        regime: RegimeId,
        interventions: impl Into<Arc<[InterventionAssignment]>>,
        axes: impl Into<Arc<[DiscreteAxis]>>,
        probabilities: impl Into<Arc<[f64]>>,
        snapshot_identity: impl Into<Arc<str>>,
        tolerance: LawTolerance,
    ) -> Result<Self, ExactLawError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::ProviderBinding,
            || {
                let check = || {
                    if antecedent_core::numerical_admission::is_cancelled() {
                        Err(ExactLawError {
                            kind: "cancelled",
                            population: Arc::from(""),
                            regime: None,
                            variables: Arc::from([]),
                            conditioning: Arc::from([]),
                            interventions: Arc::from([]),
                        })
                    } else {
                        Ok(())
                    }
                };
                check()?;
                let population = population.into();
                check()?;
                let interventions = interventions.into();
                check()?;
                let axes = axes.into();
                check()?;
                let probabilities = probabilities.into();
                check()?;
                let snapshot_identity = snapshot_identity.into();
                check()?;
                let mut law = Self {
                    population,
                    regime,
                    interventions,
                    axes,
                    probabilities,
                    strides: Arc::from([]),
                    axis_index: Arc::new(BTreeMap::new()),
                    level_index: Arc::from([]),
                    snapshot_identity,
                    tolerance,
                    origin: LawOrigin::SuppliedExact,
                    empirical_counts: None,
                    marginals: Arc::default(),
                };
                law.check_cancelled()?;
                if law.population.is_empty()
                    || law.snapshot_identity.is_empty()
                    || !tolerance.valid()
                {
                    return Err(law.error("invalid_law_metadata"));
                }
                let mut intervened = HashSet::new();
                for assignment in law.interventions.iter() {
                    law.check_cancelled()?;
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
                    if axis.values.is_empty() || axis_index.insert(axis.variable, i).is_some() {
                        return Err(law.error("invalid_law_axis"));
                    }
                    for (index, value) in axis.values.iter().enumerate() {
                        if index % 1024 == 0 {
                            law.check_cancelled()?;
                        }
                        if !finite(value) || levels.insert(value_key(value), index).is_some() {
                            return Err(law.error("invalid_law_axis"));
                        }
                    }
                    if intervened.contains(&axis.variable) {
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
                for chunk in law.probabilities.chunks(1024) {
                    law.check_cancelled()?;
                    if chunk.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
                        return Err(law.error("invalid_law_probability"));
                    }
                }
                let mut mass = CompensatedSum::default();
                for chunk in law.probabilities.chunks(1024) {
                    law.check_cancelled()?;
                    for &probability in chunk {
                        mass.add(probability);
                    }
                }
                if !tolerance.unit_mass(mass.total()) {
                    return Err(law.error("unnormalized_law"));
                }
                law.strides = strides.into();
                law.axis_index = Arc::new(axis_index);
                law.level_index = level_index.into();
                Ok(law)
            },
        )
    }

    /// Frequency plug-in joint. Empty cells are sampling zeros, not structural zeros.
    ///
    /// # Errors
    /// Same validation as [`Self::try_new`].
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
    /// Validated draw from an explicitly specified posterior law.
    /// # Errors
    /// Same validation as [`Self::try_new`].
    pub fn try_bayesian_posterior(
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
        law.origin = LawOrigin::BayesianPosterior;
        Ok(law)
    }
    /// Attach the observed cell counts that this frequency table was computed
    /// from. Every probability must equal its count's share of the total within
    /// the law's tolerance, so a point published from the table and an
    /// interval resampled from the counts describe one law. A supplied exact
    /// law becomes an empirical plug-in; any other origin is kept.
    /// # Errors
    /// Counts with wrong shape, no observations, overflow, or a cell whose
    /// probability is not its count's share (`unreconciled_empirical_counts`).
    pub fn with_empirical_counts(self, counts: Vec<u64>) -> Result<Self, ExactLawError> {
        let total = self.checked_count_total(&counts)?;
        let tolerance = self.tolerance;
        for (probabilities, counts) in self.probabilities.chunks(1024).zip(counts.chunks(1024)) {
            self.check_cancelled()?;
            if !probabilities.iter().zip(counts).all(|(p, count)| {
                let share = *count as f64 / total;
                (p - share).abs() <= tolerance.absolute + tolerance.relative * p.max(share)
            }) {
                return Err(self.error("unreconciled_empirical_counts"));
            }
        }
        self.check_cancelled()?;
        let origin = match self.origin {
            LawOrigin::SuppliedExact => LawOrigin::EmpiricalPlugin,
            other => other,
        };
        Ok(self.with_counts(counts, origin))
    }
    /// Mark a fitted law as model-based and retain its observed support. The
    /// model's cell probabilities are not frequencies and are not reconciled
    /// with the counts; the counts only decide which conditioners were observed.
    /// # Errors
    /// Counts with wrong shape, no observations, or overflow.
    pub fn with_learned_support(self, counts: Vec<u64>) -> Result<Self, ExactLawError> {
        self.checked_count_total(&counts)?;
        Ok(self.with_counts(counts, LawOrigin::LearnedPlugin))
    }
    fn checked_count_total(&self, counts: &[u64]) -> Result<f64, ExactLawError> {
        let mut total = Some(0u64);
        for chunk in counts.chunks(1024) {
            self.check_cancelled()?;
            total = total.and_then(|n| chunk.iter().try_fold(n, |n, k| n.checked_add(*k)));
        }
        self.check_cancelled()?;
        match total {
            Some(total) if counts.len() == self.probabilities.len() && total > 0 => {
                Ok(total as f64)
            }
            _ => Err(self.error("invalid_empirical_support")),
        }
    }
    fn with_counts(mut self, counts: Vec<u64>, origin: LawOrigin) -> Self {
        self.origin = origin;
        self.empirical_counts = Some(counts.into());
        self
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
    /// Stable identity of the complete numerical provider contract.
    #[must_use]
    pub fn content_identity(&self) -> antecedent_core::recalc::StageIdentity {
        let descriptor = format!(
            "{}:{:?}:{:?}:{:?}:{}:{:?}:{:?}:{:?}",
            self.population,
            self.regime,
            self.interventions,
            self.axes,
            self.snapshot_identity,
            self.tolerance,
            self.origin,
            self.empirical_counts
        );
        let probabilities: Vec<u8> =
            self.probabilities.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect();
        antecedent_core::recalc::StageIdentity::of(
            "exact.law.content.v1",
            &[descriptor.as_bytes(), &probabilities],
        )
    }
    /// The fallible constructor path checks cancellation while encoding the same bytes as
    /// `content_identity`; the public infallible identity method remains unchanged.
    fn content_identity_checked(
        &self,
    ) -> Result<antecedent_core::recalc::StageIdentity, ExactLawError> {
        self.check_cancelled()?;
        let mut descriptor = CheckedDescriptor {
            text: String::new(),
            check: || !antecedent_core::numerical_admission::is_cancelled(),
        };
        write!(
            descriptor,
            "{}:{:?}:{:?}:{:?}:{}:{:?}:{:?}:{:?}",
            self.population,
            self.regime,
            self.interventions,
            self.axes,
            self.snapshot_identity,
            self.tolerance,
            self.origin,
            self.empirical_counts
        )
        .map_err(|_| self.error("cancelled"))?;
        let mut probabilities = Vec::new();
        for chunk in self.probabilities.chunks(1024) {
            self.check_cancelled()?;
            probabilities.extend(chunk.iter().flat_map(|value| value.to_bits().to_le_bytes()));
        }
        self.check_cancelled()?;
        antecedent_core::recalc::StageIdentity::try_of(
            "exact.law.content.v1",
            &[descriptor.text.as_bytes(), &probabilities],
            || self.check_cancelled(),
        )
    }

    fn check_cancelled(&self) -> Result<(), ExactLawError> {
        if antecedent_core::numerical_admission::is_cancelled() {
            return Err(self.error("cancelled"));
        }
        Ok(())
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
        if let Some(index) = self.full_joint_index(constraints) {
            return self.probabilities[index];
        }
        if constraints.len() <= STACK_CONSTRAINTS {
            let mut buffer = [(0usize, 0usize); STACK_CONSTRAINTS];
            let by_axis = &mut buffer[..constraints.len()];
            by_axis.copy_from_slice(constraints);
            self.mass_by_axis(by_axis)
        } else {
            let mut owned = constraints.to_vec();
            self.mass_by_axis(&mut owned)
        }
    }

    /// Mass of the cells matching `(axis, level)` constraints, sorted here by axis.
    ///
    /// One axis constrained twice keeps a single constraint when the levels agree and selects
    /// no cell when they differ.
    #[allow(
        clippy::needless_range_loop,
        reason = "in-place compaction reads ahead of the write cursor"
    )]
    fn mass_by_axis(&self, by_axis: &mut [(usize, usize)]) -> f64 {
        by_axis.sort_unstable();
        let mut unique = 0;
        for i in 0..by_axis.len() {
            if unique > 0 && by_axis[unique - 1].0 == by_axis[i].0 {
                if by_axis[unique - 1].1 != by_axis[i].1 {
                    return 0.0;
                }
            } else {
                by_axis[unique] = by_axis[i];
                unique += 1;
            }
        }
        let by_axis = &by_axis[..unique];
        let mut stack_axes = [0usize; STACK_CONSTRAINTS];
        let heap_axes: Vec<usize>;
        let axes: &[usize] = if unique <= STACK_CONSTRAINTS {
            for (slot, (axis, _)) in stack_axes.iter_mut().zip(by_axis) {
                *slot = *axis;
            }
            &stack_axes[..unique]
        } else {
            heap_axes = by_axis.iter().map(|(axis, _)| *axis).collect();
            &heap_axes
        };
        let table = self.marginal_table(axes);
        let index = by_axis
            .iter()
            .fold(0usize, |index, (axis, level)| index * self.axes[*axis].values.len() + level);
        table[index]
    }

    /// Index of a complete assignment of every axis, without allocating for up to 64 axes.
    ///
    /// `None` when an axis is repeated or missing: the caller then falls back to a marginal
    /// table, which is what a repeated constraint (two levels of one axis) requires.
    fn full_joint_index(&self, constraints: &[(usize, usize)]) -> Option<usize> {
        let n_axes = self.axes.len();
        if constraints.len() != n_axes {
            return None;
        }
        // One bit per axis; a single word covers the common case, wider laws spill to a slice.
        let mut narrow = 0u64;
        let mut wide = vec![0u64; if n_axes > 64 { n_axes.div_ceil(64) } else { 0 }];
        let mut index = 0usize;
        for &(axis, level) in constraints {
            if axis >= n_axes {
                return None;
            }
            let seen = if n_axes > 64 { &mut wide[axis / 64] } else { &mut narrow };
            let bit = 1u64 << (axis % 64);
            if *seen & bit != 0 {
                return None;
            }
            *seen |= bit;
            index += self.strides[axis] * level;
        }
        // `n_axes` constraints, each on a distinct in-range axis, cover every axis.
        Some(index)
    }

    /// Mass of every level combination of the distinct, ascending `axes`, summed in table order
    /// exactly as a filtered scan would, so a lookup equals the scan it replaces.
    fn marginal_table(&self, axes: &[usize]) -> Arc<[f64]> {
        if !axes.windows(2).all(|pair| pair[0] < pair[1]) {
            return compute_marginal(self, axes);
        }
        let shard_index =
            axes.iter().fold(0usize, |h, axis| h.wrapping_mul(31).wrapping_add(*axis))
                % MARGINAL_SHARDS;
        let shard = &self.marginals.shards[shard_index];
        {
            let mut cache = shard.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(table) = cache.touch(axes) {
                return table;
            }
        }
        let table = compute_marginal(self, axes);
        let mut cache = shard.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.insert(axes, &table, MAX_MARGINAL_TABLES / MARGINAL_SHARDS);
        table
    }
}

/// Walks the cells of a dense law in table order, tracking the level of one axis without a
/// division per cell: the level advances every `stride` cells and wraps at `len`.
struct AxisCursor {
    stride: usize,
    len: usize,
    level: usize,
    left: usize,
}

impl AxisCursor {
    fn new(law: &ExactDiscreteLaw, axis: usize) -> Self {
        let stride = law.strides[axis];
        Self { stride, len: law.axes[axis].values.len(), level: 0, left: stride }
    }

    fn advance(&mut self) {
        self.left -= 1;
        if self.left == 0 {
            self.left = self.stride;
            self.level += 1;
            if self.level == self.len {
                self.level = 0;
            }
        }
    }
}

/// Most constraints a mass lookup keeps on the stack.
const STACK_CONSTRAINTS: usize = 16;

fn compute_marginal(law: &ExactDiscreteLaw, axes: &[usize]) -> Arc<[f64]> {
    let cells: usize = axes.iter().map(|a| law.axes[*a].values.len()).product();
    let mut sums = vec![CompensatedSum::default(); cells];
    let mut cursors: Vec<AxisCursor> = axes.iter().map(|a| AxisCursor::new(law, *a)).collect();
    for p in law.probabilities.iter() {
        let cell = cursors.iter().fold(0usize, |cell, c| cell * c.len + c.level);
        sums[cell].add(*p);
        for cursor in &mut cursors {
            cursor.advance();
        }
    }
    sums.into_iter().map(CompensatedSum::total).collect()
}

fn finite(v: &Value) -> bool {
    !matches!(v, Value::Float64(x) if !x.is_finite())
}

/// Sum in iteration order with [`CompensatedSum`].
pub(crate) fn sum(values: impl Iterator<Item = f64>) -> f64 {
    CompensatedSum::of(values)
}

// Hash only fixed-size digests in the map; each bucket is checked for exact
// equality, so a digest collision cannot identify or merge different worlds.
#[derive(Clone, Debug)]
struct WorldBinding {
    population: Arc<str>,
    regime: RegimeId,
    interventions: Vec<InterventionAssignment>,
}
type WorldIndex = HashMap<u64, Vec<(WorldBinding, usize)>>;

fn cancelled_error() -> ExactLawError {
    ExactLawError {
        kind: "cancelled",
        population: Arc::from(""),
        regime: None,
        variables: Arc::from([]),
        conditioning: Arc::from([]),
        interventions: Arc::from([]),
    }
}
fn poll_world() -> Result<(), ExactLawError> {
    if antecedent_core::numerical_admission::is_cancelled() {
        Err(cancelled_error())
    } else {
        Ok(())
    }
}
struct PollHasher {
    inner: std::collections::hash_map::DefaultHasher,
    cancelled: bool,
}
impl Hasher for PollHasher {
    fn finish(&self) -> u64 {
        self.inner.finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(512) {
            if poll_world().is_err() {
                self.cancelled = true;
                return;
            }
            self.inner.write(chunk);
        }
    }
}
fn world_digest(
    population: &str,
    regime: RegimeId,
    assignments: &[InterventionAssignment],
) -> Result<u64, ExactLawError> {
    poll_world()?;
    let mut h = PollHasher {
        inner: std::collections::hash_map::DefaultHasher::default(),
        cancelled: false,
    };
    population.hash(&mut h);
    regime.hash(&mut h);
    for assignment in assignments {
        poll_world()?;
        assignment.hash(&mut h);
    }
    if h.cancelled {
        return Err(cancelled_error());
    }
    poll_world()?;
    Ok(h.finish())
}
fn checked_world_key(
    assignments: &[InterventionAssignment],
) -> Result<Vec<InterventionAssignment>, ExactLawError> {
    let mut ordered: BTreeMap<VariableId, Vec<InterventionAssignment>> = BTreeMap::new();
    for a in assignments {
        poll_world()?;
        ordered
            .entry(a.variable)
            .or_default()
            .push(InterventionAssignment { variable: a.variable, value: value_key(&a.value) });
    }
    let mut key = Vec::with_capacity(assignments.len());
    for values in ordered.into_values() {
        for a in values {
            poll_world()?;
            key.push(a);
        }
    }
    poll_world()?;
    Ok(key)
}
fn checked_world_eq(
    binding: &WorldBinding,
    population: &str,
    regime: RegimeId,
    key: &[InterventionAssignment],
) -> Result<bool, ExactLawError> {
    poll_world()?;
    if binding.regime != regime
        || binding.population.len() != population.len()
        || binding.interventions.len() != key.len()
    {
        return Ok(false);
    }
    for (a, b) in binding.population.as_bytes().chunks(512).zip(population.as_bytes().chunks(512)) {
        poll_world()?;
        if a != b {
            return Ok(false);
        }
    }
    for (a, b) in binding.interventions.iter().zip(key) {
        poll_world()?;
        if a.variable != b.variable {
            return Ok(false);
        }
        match (&a.value, &b.value) {
            (Value::Label(a), Value::Label(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                for (a, b) in a.as_bytes().chunks(512).zip(b.as_bytes().chunks(512)) {
                    poll_world()?;
                    if a != b {
                        return Ok(false);
                    }
                }
            }
            (a, b) if !same_value_key(a, b) => return Ok(false),
            _ => {}
        }
    }
    Ok(true)
}
fn integral_level(value: f64) -> Option<i64> {
    if value.is_finite() && value.fract() == 0.0 && value.abs() <= 9_007_199_254_740_992.0 {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "finite integral f64, explicitly within the exact i64 range"
        )]
        Some(value as i64)
    } else {
        None
    }
}

/// `value_key(left) == value_key(right)` without building either key.
///
/// A non-integral float, including NaN, compares as `Value` does.
pub(crate) fn same_value_key(left: &Value, right: &Value) -> bool {
    match (integral_or_self(left), integral_or_self(right)) {
        (Ok(a), Ok(b)) => a == b,
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

fn integral_or_self(value: &Value) -> Result<i64, &Value> {
    match value {
        Value::Float64(level) => integral_level(*level).ok_or(value),
        Value::Int64(level) => Ok(*level),
        other => Err(other),
    }
}

// Normalize exactly represented integral numeric levels locally, without changing
// core Value equality or the original table/serialized coordinate representation.
pub(crate) fn value_key(value: &Value) -> Value {
    match value {
        Value::Float64(level) => match integral_level(*level) {
            Some(level) => Value::Int64(level),
            None => Value::Float64(*level),
        },
        other => other.clone(),
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "These fixtures assert exact representable values or bit-preserving round trips."
)]
mod tests {
    #[test]
    fn empirical_counts_must_reconcile_with_the_table() {
        use super::*;
        let law = |probabilities: &[f64]| {
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                [DiscreteAxis {
                    variable: VariableId::from_raw(0),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                }],
                probabilities.to_vec(),
                "s",
                LawTolerance::default(),
            )
            .unwrap()
        };
        let counted = law(&[0.3, 0.7]).with_empirical_counts(vec![3, 7]).unwrap();
        assert_eq!(counted.origin(), LawOrigin::EmpiricalPlugin);
        let reversed = law(&[0.3, 0.7]).with_empirical_counts(vec![7, 3]).unwrap_err();
        assert_eq!(reversed.kind, "unreconciled_empirical_counts");
        let model = law(&[0.3, 0.7]).with_learned_support(vec![7, 3]).unwrap();
        assert_eq!(model.origin(), LawOrigin::LearnedPlugin);
        assert_eq!(
            law(&[0.3, 0.7]).with_empirical_counts(vec![0, 0]).unwrap_err().kind,
            "invalid_empirical_support"
        );
    }

    use super::*;
    use crate::test_variable as v;
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
    fn inherited_cancellation_interrupts_law_validation_and_identity_encoding() {
        let table = law("source", &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(table.content_identity_checked().unwrap(), table.content_identity());
        for counted in [
            table.clone().with_empirical_counts(vec![1, 2, 3, 4]).unwrap(),
            table.clone().with_learned_support(vec![4, 3, 2, 1]).unwrap(),
        ] {
            assert_eq!(counted.content_identity_checked().unwrap(), counted.content_identity());
        }
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        for model in [false, true] {
            let error = antecedent_core::numerical_admission::with_context(&ctx, || {
                if model {
                    table.clone().with_learned_support(vec![1, 2, 3, 4])
                } else {
                    table.clone().with_empirical_counts(vec![1, 2, 3, 4])
                }
                .unwrap_err()
            });
            assert_eq!(error.kind, "cancelled");
        }
        let error = antecedent_core::numerical_admission::with_context(&ctx, || {
            law("source", &[0.1, 0.2, 0.3, 0.4])
        })
        .unwrap_err();
        assert_eq!(error.kind, "cancelled");
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let error = antecedent_core::numerical_admission::with_context(&ctx, || {
            table.content_identity_checked()
        })
        .unwrap_err();
        assert_eq!(error.kind, "cancelled");
    }

    #[test]
    fn descriptor_formatting_polls_long_utf8_writes_and_preserves_bytes() {
        use std::fmt::Write as _;
        let text = format!("{}é{}", "a".repeat(4095), "z".repeat(5000));
        let mut calls = 0;
        let mut writer = CheckedDescriptor {
            text: String::new(),
            check: || {
                calls += 1;
                true
            },
        };
        write!(writer, "{text:?}").unwrap();
        assert_eq!(writer.text, format!("{text:?}"));
        drop(writer);
        assert!(calls >= 3);

        let mut calls = 0;
        let mut writer = CheckedDescriptor {
            text: String::new(),
            check: || {
                calls += 1;
                calls < 2
            },
        };
        assert!(std::fmt::Write::write_str(&mut writer, &text).is_err());
        assert!(!writer.text.is_empty() && writer.text.len() < text.len());
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
    fn world_digest_buckets_check_collisions_and_cancellation() {
        use super::*;
        let target = law("target", &[0.1, 0.2, 0.3, 0.4]).unwrap();
        let other = law("other", &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(
            ExactTransportData::try_new([target.clone(), target.clone()], 16).unwrap_err().kind,
            "duplicate_law_binding"
        );
        let mut data = ExactTransportData::try_new([target, other], 16).unwrap();
        let target_digest = world_digest("target", RegimeId::from_raw(0), &[]).unwrap();
        let other_digest = world_digest("other", RegimeId::from_raw(0), &[]).unwrap();
        let index = Arc::make_mut(&mut data.index);
        let other = index.remove(&other_digest).unwrap().pop().unwrap();
        index.get_mut(&target_digest).unwrap().insert(0, other);
        let outcomes = [v(1)];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(0)),
            ..FactorSpec::new(&outcomes, &[], &[], DomainRef::Observational)
        };
        assert_eq!(data.require_factor(&spec).unwrap().population(), "target");
        let left = [
            InterventionAssignment::concrete(v(2), Value::Float64(1.0)),
            InterventionAssignment::concrete(v(1), Value::Bool(true)),
        ];
        let right = [
            InterventionAssignment::concrete(v(1), Value::Bool(true)),
            InterventionAssignment::concrete(v(2), Value::Int64(1)),
        ];
        let left = checked_world_key(&left).unwrap();
        let right = checked_world_key(&right).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            world_digest("target", RegimeId::from_raw(0), &left).unwrap(),
            world_digest("target", RegimeId::from_raw(0), &right).unwrap()
        );
        let ctx = antecedent_core::ExecutionContext::for_tests(7);
        ctx.cancellation.cancel();
        let error =
            antecedent_core::numerical_admission::with_context(&ctx, || data.require_factor(&spec))
                .unwrap_err();
        assert_eq!(error.kind, "cancelled");
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
    fn shared_factor_cache_is_content_bound_across_rebinds() {
        let axis =
            DiscreteAxis { variable: v(0), values: Arc::from([Value::Int64(0), Value::Int64(1)]) };
        let make_law = |probabilities| {
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                [axis.clone()],
                probabilities,
                // Deliberately keep caller-supplied snapshot metadata the same.
                "same-snapshot-label",
                LawTolerance::default(),
            )
            .unwrap()
        };
        let old = ExactTransportData::try_new([make_law([0.25, 0.75])], 4)
            .unwrap()
            .with_shared_factor_cache(4);
        let replacement = ExactTransportData::try_new([make_law([0.75, 0.25])], 4).unwrap();
        let outputs = [v(0)];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(0)),
            ..FactorSpec::new(&outputs, &[], &[], DomainRef::Observational)
        };
        let assignment = Assignment::from_pairs([(v(0), Value::Int64(1))]);

        // Warm the shared cache with the first law, then rebind a different immutable
        // probability table that deliberately reuses its external snapshot label.
        assert_eq!(
            old.probability_checked(&spec, &assignment).unwrap().to_bits(),
            0.75_f64.to_bits()
        );
        let rebound = replacement.rebind_reusing_factor_cache(Some(&old), 4).unwrap();
        let warm_rebind_value = rebound.probability_checked(&spec, &assignment).unwrap();
        let fresh_value = replacement.probability_checked(&spec, &assignment).unwrap();
        assert_eq!(warm_rebind_value.to_bits(), 0.25_f64.to_bits());
        assert_eq!(warm_rebind_value.to_bits(), fresh_value.to_bits());
        assert_eq!(
            old.probability_checked(&spec, &assignment).unwrap().to_bits(),
            0.75_f64.to_bits()
        );
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
            .with_learned_support(vec![0, 0, 3, 7])
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
    fn ratio_form_conditional(arena: &mut crate::CausalExprArena) -> crate::ExprId {
        let variables = arena.intern_var_set([v(0), v(1)]);
        let conditioned_on = arena.empty_var_set();
        let intervention = arena.intern_intervention_set([]);
        let population = arena.intern_population("target");
        let joint = arena.intern(crate::ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            population,
            domain: DomainRef::Observational,
            regime: Some(RegimeId::from_raw(0)),
        });
        let summed = arena.intern_var_set([v(1)]);
        let marginal = arena.intern(crate::ExprNode::SumOut { variables: summed, expr: joint });
        arena.intern(crate::ExprNode::Ratio { numerator: joint, denominator: marginal })
    }
    #[test]
    fn ratio_form_empirical_empty_conditioner_is_sampling_zero() {
        // X=0 never appears in the sample, but the query binds it — a DGP that
        // cannot produce the empty cell would not earn this claim.
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
        let data = ExactTransportData::try_new([table], 16).unwrap();
        let mut arena = crate::CausalExprArena::new();
        let root = ratio_form_conditional(&mut arena);
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let plan = crate::ExactEvaluationPlan::compile(
            crate::ExactEvaluationSpec {
                arena: &arena,
                root,
                outcomes: [v(1)].into(),
                request: Assignment::from_pairs([(v(0), Value::Int64(0))]),
                limits: crate::ExactEvaluationLimits::default(),
                tolerance: LawTolerance::default(),
            },
            data,
            &ctx,
        )
        .unwrap();
        let EvalError::ExactLaw(error) = plan.evaluate(&ctx).unwrap_err() else {
            panic!("ratio-form empty empirical conditioner must be sampling_zero");
        };
        assert_eq!(error.kind, "sampling_zero");
    }
    #[test]
    fn ratio_form_positive_learned_probabilities_do_not_authorize_empty_empirical_conditioners() {
        let predicted = law("target", &[0.25, 0.25, 0.25, 0.25])
            .unwrap()
            .with_learned_support(vec![0, 0, 3, 7])
            .unwrap();
        let data = ExactTransportData::try_new([predicted], 16).unwrap();
        let mut arena = crate::CausalExprArena::new();
        let root = ratio_form_conditional(&mut arena);
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let plan = crate::ExactEvaluationPlan::compile(
            crate::ExactEvaluationSpec {
                arena: &arena,
                root,
                outcomes: [v(1)].into(),
                request: Assignment::from_pairs([(v(0), Value::Int64(0))]),
                limits: crate::ExactEvaluationLimits::default(),
                tolerance: LawTolerance::default(),
            },
            data,
            &ctx,
        )
        .unwrap();
        let EvalError::ExactLaw(error) = plan.evaluate(&ctx).unwrap_err() else {
            panic!("ratio-form empty empirical conditioner must be sampling_zero");
        };
        assert_eq!(error.kind, "sampling_zero");
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

    #[test]
    fn marginal_tables_agree_with_brute_force_enumeration() {
        // Three binary axes, an asymmetric joint. Every subset of axes and every level
        // combination must equal the sum of the joint cells that satisfy it, enumerated
        // independently of the strides/table machinery.
        let axes: Vec<DiscreteAxis> = [0, 1, 2].iter().map(|i| axis(*i)).collect();
        let probabilities = [0.01, 0.04, 0.05, 0.10, 0.20, 0.09, 0.26, 0.25];
        let table = ExactDiscreteLaw::try_new(
            "p",
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            "s",
            LawTolerance::default(),
        )
        .unwrap();
        // Cell index = 4*a0 + 2*a1 + a2.
        let brute = |constraints: &[(usize, usize)]| -> f64 {
            (0..8usize)
                .filter(|cell| {
                    constraints.iter().all(|(axis, level)| (cell >> (2 - axis)) & 1 == *level)
                })
                .map(|cell| probabilities[cell])
                .sum()
        };
        for subset in 0..8usize {
            let axes: Vec<usize> = (0..3).filter(|a| (subset >> a) & 1 == 1).collect();
            for levels in 0..(1usize << axes.len()) {
                let constraints: Vec<(usize, usize)> =
                    axes.iter().enumerate().map(|(k, a)| (*a, (levels >> k) & 1)).collect();
                let want = brute(&constraints);
                assert!((table.mass(&constraints) - want).abs() < 1e-15, "{constraints:?}");
                // Order of the constraints does not matter, and repeated queries hit the cache.
                let mut reversed = constraints.clone();
                reversed.reverse();
                assert_eq!(table.mass(&reversed).to_bits(), table.mass(&constraints).to_bits());
            }
        }
        // One axis pinned to two different levels is the empty event.
        assert_eq!(table.mass(&[(0, 0), (0, 1)]).abs(), 0.0);
    }

    #[test]
    fn mass_lookups_work_beyond_the_word_width_of_the_axis_masks() {
        // 140 single-level axes: one cell of mass 1. Both the complete-assignment fast path
        // (more than 64 axes) and the cached marginal tables (more than 128 axes) must work.
        let axes: Vec<DiscreteAxis> = (0..140u32)
            .map(|i| DiscreteAxis { variable: v(i), values: Arc::from([Value::Int64(0)]) })
            .collect();
        let law = ExactDiscreteLaw::try_new(
            "p",
            RegimeId::from_raw(0),
            [],
            axes,
            [1.0],
            "s",
            LawTolerance::default(),
        )
        .unwrap();
        let full: Vec<(usize, usize)> = (0..140).rev().map(|a| (a, 0)).collect();
        assert_eq!(law.mass(&full), 1.0);
        let mut repeated = full.clone();
        repeated[0] = repeated[1];
        assert_eq!(law.mass(&repeated), 1.0);
        let high: Vec<(usize, usize)> = (100..140).map(|a| (a, 0)).collect();
        assert_eq!(law.mass(&high), 1.0);
        assert_eq!(law.mass(&high), 1.0, "second lookup is served from the cache");
    }

    #[test]
    fn marginal_cache_evicts_the_least_recently_used_table() {
        let mut shard = MarginalShard::new();
        let table: Arc<[f64]> = Arc::from([1.0]);
        shard.insert(&[0], &table, 2);
        shard.insert(&[1], &table, 2);
        assert!(shard.touch(&[0]).is_some(), "refreshes the older entry");
        shard.insert(&[2], &table, 2);
        assert!(shard.touch(&[1]).is_none(), "the untouched entry was evicted");
        assert!(shard.touch(&[0]).is_some());
        assert!(shard.touch(&[2]).is_some());
        assert_eq!(shard.entries.len(), shard.order.len());
    }

    #[test]
    fn integral_float_selects_the_int_intervention_world() {
        assert!(same_value_key(&Value::Float64(1.0), &Value::Int64(1)));
        assert!(same_value_key(&Value::Float64(-0.0), &Value::Int64(0)));
        assert!(!same_value_key(&Value::Float64(1.5), &Value::Int64(1)));
        assert!(same_value_key(&Value::Float64(f64::NAN), &Value::Float64(f64::NAN)));
        let law = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(1),
            [InterventionAssignment { variable: v(0), value: Value::Int64(1) }],
            [axis(1)],
            [0.25, 0.75],
            "s",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 1).unwrap();
        let intervention = [InterventionAssignment { variable: v(0), value: Value::Float64(1.0) }];
        let spec = FactorSpec {
            population: "target",
            regime: Some(RegimeId::from_raw(1)),
            domain: DomainRef::Interventional,
            variables: &[v(1)],
            conditioned_on: &[],
            intervention: &intervention,
        };
        let assignment = Assignment::from_pairs([(v(1), Value::Int64(1))]);
        let p = data.probability_checked(&spec, &assignment).unwrap();
        assert!((p - 0.75).abs() < 1e-15);
    }
}

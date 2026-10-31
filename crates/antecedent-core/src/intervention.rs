//! Interventions on causal variables.
//!
//! enables hard, shift, stochastic, soft, and sequenced interventions.
//! Estimators that only support hard `Set` continue to reject other variants.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::callback::Portability;
use crate::canonical_encode::{CanonicalEncode, count, i32_bits};
use crate::identity::CanonicalWriter;
use crate::ids::{DynamicRuleId, VariableId};
use crate::refusal::NextAttempt;
use crate::value::Value;
use crate::value_expr::{ExpressionFault, PolicyReads, ValueExpr};

/// Caller-authorized finite ceilings for temporal materialization.
/// These logical limits remain in force even if the execution pool is unbounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalMaterializationLimits {
    /// Maximum total active offsets expanded by one operation.
    pub max_offsets: u64,
    /// Maximum total logical work across one operation.
    pub max_work: u64,
    /// Maximum bytes charged to an individual admitted materialization.
    pub max_bytes: u64,
    /// Maximum dense nodes in one unrolled graph.
    pub max_graph_nodes: u64,
    /// Maximum edge insertions and induced-pair candidates in one unroll.
    pub max_graph_edges: u64,
}

impl Default for TemporalMaterializationLimits {
    fn default() -> Self {
        Self {
            max_offsets: 100_000,
            max_work: 100_000_000,
            max_bytes: 256 * 1024 * 1024,
            max_graph_nodes: 100_000,
            max_graph_edges: 1_000_000,
        }
    }
}

/// Cumulative logical allowance shared by all stages of one temporal operation.
/// Clones share the same counters, including across concurrent workers.
/// Resident bytes remain charged separately to the leases owning allocations.
#[derive(Clone, Debug)]
pub struct TemporalBudget {
    limits: TemporalMaterializationLimits,
    remaining_offsets: Arc<std::sync::atomic::AtomicU64>,
    remaining_work: Arc<std::sync::atomic::AtomicU64>,
}
impl TemporalBudget {
    /// Start a fresh top-level operation with finite caller-selected limits.
    #[must_use]
    pub fn new(limits: TemporalMaterializationLimits) -> Self {
        Self {
            remaining_offsets: Arc::new(std::sync::atomic::AtomicU64::new(limits.max_offsets)),
            remaining_work: Arc::new(std::sync::atomic::AtomicU64::new(limits.max_work)),
            limits,
        }
    }
    /// Per-materialization byte and per-graph shape limits.
    #[must_use]
    pub const fn limits(&self) -> TemporalMaterializationLimits {
        self.limits
    }
    /// Charge logical expansion before traversing or allocating it.
    pub fn charge_offsets(&mut self, count: u64) -> Result<(), TemporalMaterializationError> {
        self.remaining_offsets
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |remaining| remaining.checked_sub(count),
            )
            .map_err(|_| TemporalMaterializationError::Limit { dimension: "offsets" })?;
        Ok(())
    }
    /// Charge work before starting the bounded operation.
    pub fn charge_work(&mut self, count: u64) -> Result<(), TemporalMaterializationError> {
        self.remaining_work
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |remaining| remaining.checked_sub(count),
            )
            .map_err(|_| TemporalMaterializationError::Limit { dimension: "work" })?;
        Ok(())
    }
}

/// Admission retained while generic intervention validation constructs temporary read/target sets.
/// The visitor is allocation-free and stops before recursive validation at 64 levels.
pub fn admit_temporal_intervention_validation<'a>(
    interventions: impl IntoIterator<Item = &'a Intervention>,
    budget: &mut TemporalBudget,
    ctx: &crate::ExecutionContext,
) -> Result<crate::execution::AdmissionLease, TemporalMaterializationError> {
    struct Visitor<'a> {
        budget: &'a mut TemporalBudget,
        ctx: &'a crate::ExecutionContext,
        bytes: u64,
        expr_nodes: u64,
        intervention_nodes: u64,
    }
    impl Visitor<'_> {
        fn add(&mut self, work: u64, bytes: u64) -> Result<(), TemporalMaterializationError> {
            if self.ctx.cancellation.is_cancelled() {
                return Err(TemporalMaterializationError::Cancelled);
            }
            self.budget.charge_work(work)?;
            self.bytes = self
                .bytes
                .checked_add(bytes)
                .ok_or(TemporalMaterializationError::Limit { dimension: "bytes" })?;
            if self.bytes > self.budget.limits().max_bytes {
                return Err(TemporalMaterializationError::Limit { dimension: "bytes" });
            }
            Ok(())
        }
        fn expression(
            &mut self,
            expr: &ValueExpr,
            depth: usize,
        ) -> Result<(), TemporalMaterializationError> {
            if depth > crate::value_expr::MAX_EXPR_DEPTH {
                return Err(TemporalMaterializationError::Limit { dimension: "expression depth" });
            }
            self.add(8, 256)?;
            self.expr_nodes = self
                .expr_nodes
                .checked_add(1)
                .ok_or(TemporalMaterializationError::Limit { dimension: "work" })?;
            match expr {
                ValueExpr::Neg(a) | ValueExpr::Not(a) | ValueExpr::Indicator(a) => {
                    self.expression(a, depth + 1)?
                }
                ValueExpr::Arith { lhs, rhs, .. }
                | ValueExpr::Compare { lhs, rhs, .. }
                | ValueExpr::Logic { lhs, rhs, .. } => {
                    self.expression(lhs, depth + 1)?;
                    self.expression(rhs, depth + 1)?;
                }
                ValueExpr::Clip { value, lower, upper } => {
                    self.expression(value, depth + 1)?;
                    self.expression(lower, depth + 1)?;
                    self.expression(upper, depth + 1)?;
                }
                ValueExpr::If { condition, then_branch, else_branch } => {
                    self.expression(condition, depth + 1)?;
                    self.expression(then_branch, depth + 1)?;
                    self.expression(else_branch, depth + 1)?;
                }
                ValueExpr::Callback(call) => {
                    let spec = call.spec();
                    let n = spec.inputs.len() as u64;
                    self.add(n, 0)?;
                    let mut longest = 0u64;
                    for label in spec.inputs.iter() {
                        if self.ctx.cancellation.is_cancelled() {
                            return Err(TemporalMaterializationError::Cancelled);
                        }
                        longest = longest.max(label.len() as u64);
                    }
                    let comparisons = n
                        .checked_mul(n)
                        .and_then(|v| v.checked_mul(longest.max(1)))
                        .ok_or(TemporalMaterializationError::Limit { dimension: "work" })?;
                    self.add(comparisons, 0)?;
                    if let crate::callback::CallbackOutput::Levels(levels) = &spec.output {
                        self.add(levels.len() as u64, 0)?;
                    }
                    self.add(call.args().len() as u64, 0)?;
                    for arg in call.args() {
                        self.expression(arg, depth + 1)?;
                    }
                }
                ValueExpr::Const(_)
                | ValueExpr::Generated
                | ValueExpr::Var(_)
                | ValueExpr::Lag { .. } => {}
            }
            Ok(())
        }
        fn intervention(
            &mut self,
            item: &Intervention,
            depth: usize,
        ) -> Result<(), TemporalMaterializationError> {
            if depth > crate::value_expr::MAX_EXPR_DEPTH {
                return Err(TemporalMaterializationError::Limit {
                    dimension: "intervention depth",
                });
            }
            self.add(4, 1024)?;
            self.intervention_nodes = self
                .intervention_nodes
                .checked_add(1)
                .ok_or(TemporalMaterializationError::Limit { dimension: "work" })?;
            match item {
                Intervention::Set { .. } | Intervention::Shift { .. } => {}
                Intervention::Soft { mechanism, .. } => {
                    self.add(mechanism.parameters.len() as u64, 0)?
                }
                Intervention::Stochastic { policy, .. } => {
                    if let StochasticPolicy::Categorical { probs } = policy {
                        self.add((probs.len() as u64).saturating_mul(2), 0)?;
                    }
                }
                Intervention::Transform { expr, reads, .. }
                | Intervention::Assign { expr, reads, .. } => {
                    self.add(
                        (reads.current.len() as u64).saturating_add(reads.history.len() as u64),
                        0,
                    )?;
                    self.expression(expr, 1)?;
                }
                Intervention::Sequence(sequence) => {
                    let n = sequence.steps.len() as u64;
                    self.add(
                        n.checked_mul(n)
                            .and_then(|x| x.checked_add(n))
                            .ok_or(TemporalMaterializationError::Limit { dimension: "work" })?,
                        n.checked_mul(128)
                            .ok_or(TemporalMaterializationError::Limit { dimension: "bytes" })?,
                    )?;
                    for step in sequence.steps.iter() {
                        self.intervention(&step.intervention, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
    }
    let mut visitor = Visitor { budget, ctx, bytes: 4096, expr_nodes: 0, intervention_nodes: 0 };
    for item in interventions {
        visitor.intervention(item, 1)?;
    }
    // Read-set construction sorts collected variables. Charge a conservative
    // log factor for all expression nodes, in addition to the visits above.
    let interventions = visitor.intervention_nodes;
    visitor.add(
        interventions
            .checked_mul(interventions)
            .and_then(|n| n.checked_mul(64))
            .ok_or(TemporalMaterializationError::Limit { dimension: "work" })?,
        0,
    )?;
    let n = visitor.expr_nodes;
    let log = u64::from(64 - n.max(1).leading_zeros());
    visitor.add(
        n.checked_mul(log)
            .and_then(|x| x.checked_mul(8))
            .ok_or(TemporalMaterializationError::Limit { dimension: "work" })?,
        0,
    )?;
    let bytes = visitor.bytes;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        crate::execution::AdmissionError::Cancelled => TemporalMaterializationError::Cancelled,
        other => TemporalMaterializationError::Admission(other),
    })
}

/// An invalid policy or an operational refusal to materialize it.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TemporalMaterializationError {
    /// Policy shape is invalid.
    #[error(transparent)]
    Policy(#[from] InterventionError),
    /// Caller-selected logical or size ceiling was exceeded.
    #[error("temporal materialization limit exceeded: {dimension}")]
    Limit {
        /// Which materialization bound was exceeded.
        dimension: &'static str,
    },
    /// Admission was refused.
    #[error(transparent)]
    Admission(#[from] crate::execution::AdmissionError),
    /// Cancellation was observed.
    #[error("temporal materialization cancelled")]
    Cancelled,
    /// The allocator refused the bounded allocation.
    #[error("temporal allocation failed")]
    Allocation,
}

/// Allocation-free ordered policy traversal. Callers doing long work must poll
/// cancellation and enforce a logical-work limit while consuming the iterator.
pub enum TemporalOffsetIter<'a> {
    /// One offset.
    Pulse(std::option::IntoIter<i32>),
    /// Inclusive sustained range, including ranges reaching i32::MAX.
    Sustained(std::ops::RangeInclusive<i32>),
    /// Borrowed dynamic schedule.
    Dynamic(std::iter::Copied<std::slice::Iter<'a, i32>>),
}
impl Iterator for TemporalOffsetIter<'_> {
    type Item = i32;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Pulse(iter) => iter.next(),
            Self::Sustained(iter) => iter.next(),
            Self::Dynamic(iter) => iter.next(),
        }
    }
}

/// An admitted offset buffer; clones share both its data and its lease.
#[derive(Clone, Debug)]
pub struct AdmittedTemporalOffsets(Arc<AdmittedTemporalOffsetsInner>);
#[derive(Debug)]
struct AdmittedTemporalOffsetsInner {
    values: Box<[i32]>,
    _lease: crate::execution::AdmissionLease,
}
impl AdmittedTemporalOffsets {
    /// Borrow offsets. A separately copied slice is owned by its caller.
    #[must_use]
    pub fn as_slice(&self) -> &[i32] {
        &self.0.values
    }
}
impl AsRef<[i32]> for AdmittedTemporalOffsets {
    fn as_ref(&self) -> &[i32] {
        self.as_slice()
    }
}
impl std::ops::Deref for AdmittedTemporalOffsets {
    type Target = [i32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Temporal intervention policy over discrete time steps.
///
/// Horizons and offsets are **time steps** relative to the series indexer, not
/// wall-clock durations.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TemporalPolicy {
    /// Instantaneous intervention at a single offset.
    Pulse {
        /// Time offset (steps) of the pulse relative to the analysis window origin.
        at: i32,
    },
    /// Intervention held from `from` through `until` inclusive (step offsets).
    Sustained {
        /// First intervened step.
        from: i32,
        /// Last intervened step (inclusive).
        until: i32,
    },
    /// Rule-tagged schedule with explicit active time steps.
    ///
    /// `active_at` is the evaluated schedule (sorted unique offsets). `rule` is
    /// an opaque provenance handle for wire / caller registries.
    Dynamic {
        /// Opaque rule handle.
        rule: DynamicRuleId,
        /// Non-empty sorted unique step offsets where the intervention is active.
        active_at: Arc<[i32]>,
    },
}

impl CanonicalEncode for TemporalPolicy {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Pulse { at } => {
                writer.word(1).word(i32_bits(*at));
            }
            Self::Sustained { from, until } => {
                writer.word(2).word(i32_bits(*from)).word(i32_bits(*until));
            }
            Self::Dynamic { rule, active_at } => {
                writer.word(3).word(u64::from(rule.raw())).word(count(active_at.len()));
                for step in active_at.as_ref() {
                    writer.word(i32_bits(*step));
                }
            }
        }
    }
}

impl TemporalPolicy {
    /// One-step pulse at offset `at`.
    #[must_use]
    pub const fn pulse(at: i32) -> Self {
        Self::Pulse { at }
    }

    /// Sustained intervention on `[from, until]`.
    #[must_use]
    pub const fn sustained(from: i32, until: i32) -> Self {
        Self::Sustained { from, until }
    }

    /// Dynamic schedule with explicit active offsets (deduped + sorted).
    ///
    /// # Panics
    ///
    /// Never panics; empty `active_at` fails [`Self::validate`].
    #[must_use]
    pub fn dynamic(rule: DynamicRuleId, active_at: impl Into<Arc<[i32]>>) -> Self {
        let mut steps: Vec<i32> = active_at.into().as_ref().to_vec();
        steps.sort_unstable();
        steps.dedup();
        Self::Dynamic { rule, active_at: Arc::from(steps) }
    }

    /// Whether this is a pulse or a sustained window rather than a dynamic
    /// schedule.
    #[must_use]
    pub const fn is_pulse_or_sustained(&self) -> bool {
        matches!(self, Self::Pulse { .. } | Self::Sustained { .. })
    }

    /// Whether this policy intervenes at exactly one step: a pulse, or a
    /// sustained window whose `from` and `until` coincide.
    #[must_use]
    pub const fn is_single_step(&self) -> bool {
        match self {
            Self::Pulse { .. } => true,
            Self::Sustained { from, until } => *from == *until,
            Self::Dynamic { .. } => false,
        }
    }

    /// Whether step `t` is an active intervention time under this policy.
    #[must_use]
    pub fn is_active_at(&self, t: i32) -> bool {
        match self {
            Self::Pulse { at } => t == *at,
            Self::Sustained { from, until } => t >= *from && t <= *until,
            Self::Dynamic { active_at, .. } => active_at.binary_search(&t).is_ok(),
        }
    }

    /// Number of active offsets without expanding the policy.
    /// # Errors
    /// An inverted sustained window or empty dynamic schedule.
    pub fn offset_count(&self) -> Result<u64, InterventionError> {
        self.validate()?;
        Ok(match self {
            Self::Pulse { .. } => 1,
            Self::Sustained { from, until } => (i64::from(*until) - i64::from(*from) + 1) as u64,
            Self::Dynamic { active_at, .. } => active_at.len() as u64,
        })
    }

    /// Traverse offsets without allocating the expanded schedule.
    /// # Errors
    /// An inverted sustained window or empty dynamic schedule.
    pub fn offsets(&self) -> Result<TemporalOffsetIter<'_>, InterventionError> {
        self.validate()?;
        Ok(match self {
            Self::Pulse { at } => TemporalOffsetIter::Pulse(Some(*at).into_iter()),
            Self::Sustained { from, until } => TemporalOffsetIter::Sustained(*from..=*until),
            Self::Dynamic { active_at, .. } => {
                TemporalOffsetIter::Dynamic(active_at.iter().copied())
            }
        })
    }

    /// Materialize a schedule only after checking logical limits and reserving
    /// bytes. The returned handle and its clones retain the memory lease.
    /// # Errors
    /// Invalid policy, caller limit, cancellation, admission or allocation failure.
    pub fn active_offsets_within(
        &self,
        budget: &mut TemporalBudget,
        ctx: &crate::ExecutionContext,
    ) -> Result<AdmittedTemporalOffsets, TemporalMaterializationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(TemporalMaterializationError::Cancelled);
        }
        let count = self.offset_count()?;
        budget.charge_offsets(count)?;
        budget.charge_work(count)?;
        let limits = budget.limits();
        // Vec-to-box may transiently move the allocation. Include both buffers
        // plus the returned Arc/lease bookkeeping in the admission.
        let bytes = count
            .checked_mul(8)
            .and_then(|n| n.checked_add(256))
            .ok_or(TemporalMaterializationError::Limit { dimension: "bytes" })?;
        if bytes > limits.max_bytes {
            return Err(TemporalMaterializationError::Limit { dimension: "bytes" });
        }
        let len = usize::try_from(count)
            .map_err(|_| TemporalMaterializationError::Limit { dimension: "offsets" })?;
        let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            crate::execution::AdmissionError::Cancelled => TemporalMaterializationError::Cancelled,
            other => TemporalMaterializationError::Admission(other),
        })?;
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(|_| TemporalMaterializationError::Allocation)?;
        for (index, offset) in self.offsets()?.enumerate() {
            if index % 1024 == 0 && ctx.cancellation.is_cancelled() {
                return Err(TemporalMaterializationError::Cancelled);
            }
            values.push(offset);
        }
        if ctx.cancellation.is_cancelled() {
            return Err(TemporalMaterializationError::Cancelled);
        }
        Ok(AdmittedTemporalOffsets(Arc::new(AdmittedTemporalOffsetsInner {
            values: values.into_boxed_slice(),
            _lease: lease,
        })))
    }

    /// Active offsets under finite default limits. Use `active_offsets_within`
    /// to supply the caller's shared operation budget and cancellation context.
    /// # Errors
    /// Invalid policy, default logical limits, admission, cancellation, or allocation failure.
    pub fn active_offsets(&self) -> Result<AdmittedTemporalOffsets, TemporalMaterializationError> {
        let ctx = crate::ExecutionContext::production_default(0);
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        self.active_offsets_within(&mut budget, &ctx)
    }

    /// Validate policy bounds.
    ///
    /// # Errors
    ///
    /// Empty/inverted sustained window or empty dynamic schedule.
    pub fn validate(&self) -> Result<(), InterventionError> {
        match self {
            Self::Pulse { .. } => Ok(()),
            Self::Sustained { from, until } => {
                if *until < *from {
                    return Err(InterventionError::InvalidTemporalWindow {
                        from: *from,
                        until: *until,
                    });
                }
                Ok(())
            }
            Self::Dynamic { active_at, .. } => {
                if active_at.is_empty() {
                    return Err(InterventionError::EmptyDynamicSchedule);
                }
                Ok(())
            }
        }
    }
}

/// Opaque mechanism replacement used by soft interventions.
///
/// The model layer resolves `family_id` against its registry; `parameters` are a
/// packed coefficient / noise vector interpreted by that family. Equality and
/// hashing are by parameter bits, like [`Value`], so `Eq` and `Hash` agree and a
/// NaN parameter equals itself.
#[derive(Clone, Debug)]
pub struct MechanismOverride {
    /// Registry family identifier (e.g. `"linear_gaussian"`, `"constant"`).
    pub family_id: Arc<str>,
    /// Packed parameters for the override family.
    pub parameters: Arc<[f64]>,
}

impl MechanismOverride {
    /// Named family with packed parameters.
    #[must_use]
    pub fn named(family_id: impl Into<Arc<str>>, parameters: impl Into<Arc<[f64]>>) -> Self {
        Self { family_id: family_id.into(), parameters: parameters.into() }
    }

    /// Constant structural assignment (soft form of a hard set).
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self::named("constant", Arc::<[f64]>::from(vec![value]))
    }

    /// Multiply a temporal structural assignment (including innovation) by `factor`.
    /// Supported by the unfolded sequential mean-mechanism engine.
    #[must_use]
    pub fn multiplicative(factor: f64) -> Self {
        Self::named("multiplicative", vec![factor])
    }

    /// Shift a temporal assignment to target a bounded population mean.
    ///
    /// Defines `f_policy = f + clamp(mu + delta, lower, upper) - mu`, where
    /// `mu = E[f]` under preceding interventions. Parent effects and innovations
    /// remain; realized outcomes need not lie within the bounds. Parameters are
    /// `[delta, lower, upper]`. This is not stochastic outcome clipping.
    #[must_use]
    pub fn truncated_shift(delta: f64, lower: f64, upper: f64) -> Self {
        Self::named("truncated_shift", vec![delta, lower, upper])
    }

    /// Additive shift applied to the structural assignment.
    #[must_use]
    pub fn additive_shift(delta: f64) -> Self {
        Self::named("additive_shift", Arc::<[f64]>::from(vec![delta]))
    }
}

impl PartialEq for MechanismOverride {
    fn eq(&self, other: &Self) -> bool {
        self.family_id == other.family_id && bits_eq(&self.parameters, &other.parameters)
    }
}

impl Eq for MechanismOverride {}

/// Elementwise bit equality: the relation `Hash` (over `to_bits`) is consistent with.
fn bits_eq(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

impl core::hash::Hash for MechanismOverride {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.family_id.hash(state);
        for p in self.parameters.iter() {
            p.to_bits().hash(state);
        }
    }
}

/// Stochastic assignment policy for an intervened variable.
///
/// Equality and hashing are by parameter bits, like [`Value`], so `Eq` and `Hash`
/// agree; [`Self::validate`] rejects the NaN parameters that would make the
/// bit relation differ from numeric equality.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum StochasticPolicy {
    /// Bernoulli draw with success probability `p` in `[0, 1]`.
    Bernoulli {
        /// Success probability.
        p: f64,
    },
    /// Independent Gaussian draws.
    Gaussian {
        /// Mean.
        mean: f64,
        /// Variance (must be positive).
        variance: f64,
    },
    /// Categorical over `probs` (non-negative; normalized at sample time).
    Categorical {
        /// Category probabilities (length = support size).
        probs: Arc<[f64]>,
    },
}

impl CanonicalEncode for StochasticPolicy {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Bernoulli { p } => {
                writer.word(1).real(*p);
            }
            Self::Gaussian { mean, variance } => {
                writer.word(2).real(*mean).real(*variance);
            }
            Self::Categorical { probs } => {
                writer.word(3);
                writer.reals(probs);
            }
        }
    }
}

impl StochasticPolicy {
    /// Bernoulli policy.
    #[must_use]
    pub const fn bernoulli(p: f64) -> Self {
        Self::Bernoulli { p }
    }

    /// Gaussian policy.
    #[must_use]
    pub const fn gaussian(mean: f64, variance: f64) -> Self {
        Self::Gaussian { mean, variance }
    }

    /// Categorical policy.
    #[must_use]
    pub fn categorical(probs: impl Into<Arc<[f64]>>) -> Self {
        Self::Categorical { probs: probs.into() }
    }

    /// Validate policy parameters.
    ///
    /// # Errors
    ///
    /// Invalid probability, non-positive variance, or empty categorical support.
    pub fn validate(&self) -> Result<(), InterventionError> {
        match self {
            Self::Bernoulli { p } => {
                if !(0.0..=1.0).contains(p) || !p.is_finite() {
                    return Err(InterventionError::InvalidStochasticPolicy(
                        StochasticPolicyFault::BernoulliProbability,
                    ));
                }
            }
            Self::Gaussian { mean, variance } => {
                if !mean.is_finite() {
                    return Err(InterventionError::InvalidStochasticPolicy(
                        StochasticPolicyFault::GaussianMean,
                    ));
                }
                if !(variance.is_finite() && *variance > 0.0) {
                    return Err(InterventionError::InvalidStochasticPolicy(
                        StochasticPolicyFault::GaussianVariance,
                    ));
                }
            }
            Self::Categorical { probs } => {
                if probs.is_empty() {
                    return Err(InterventionError::InvalidStochasticPolicy(
                        StochasticPolicyFault::CategoricalEmpty,
                    ));
                }
                if probs.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
                    return Err(InterventionError::InvalidStochasticPolicy(
                        StochasticPolicyFault::CategoricalEntries,
                    ));
                }
                let sum: f64 = probs.iter().sum();
                if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return Err(InterventionError::InvalidStochasticPolicy(
                        StochasticPolicyFault::CategoricalTotal,
                    ));
                }
            }
        }
        Ok(())
    }
}

impl PartialEq for StochasticPolicy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bernoulli { p: a }, Self::Bernoulli { p: b }) => a.to_bits() == b.to_bits(),
            (
                Self::Gaussian { mean: ma, variance: va },
                Self::Gaussian { mean: mb, variance: vb },
            ) => ma.to_bits() == mb.to_bits() && va.to_bits() == vb.to_bits(),
            (Self::Categorical { probs: a }, Self::Categorical { probs: b }) => bits_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for StochasticPolicy {}

impl core::hash::Hash for StochasticPolicy {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Bernoulli { p } => p.to_bits().hash(state),
            Self::Gaussian { mean, variance } => {
                mean.to_bits().hash(state);
                variance.to_bits().hash(state);
            }
            Self::Categorical { probs } => {
                for p in probs.iter() {
                    p.to_bits().hash(state);
                }
            }
        }
    }
}

/// One step in an intervention sequence with temporal policy.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SequencedIntervention {
    /// Intervention applied at this step.
    pub intervention: Intervention,
    /// When the intervention is active.
    pub temporal: TemporalPolicy,
}

impl SequencedIntervention {
    /// Construct a sequenced step.
    #[must_use]
    pub fn new(intervention: Intervention, temporal: TemporalPolicy) -> Self {
        Self { intervention, temporal }
    }
}

/// Ordered list of temporally scoped interventions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InterventionSequence {
    /// Steps in application order.
    pub steps: Arc<[SequencedIntervention]>,
}

impl InterventionSequence {
    /// Construct from steps.
    #[must_use]
    pub fn new(steps: impl Into<Arc<[SequencedIntervention]>>) -> Self {
        Self { steps: steps.into() }
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }
}

/// An intervention applied to one or more variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Intervention {
    /// Hard assignment `do(variable := value)`.
    Set {
        /// Target variable.
        variable: VariableId,
        /// Assigned value.
        value: Value,
    },
    /// Additive shift `do(variable := variable + delta)` on the factual scale.
    Shift {
        /// Target variable.
        variable: VariableId,
        /// Delta added to the structural assignment.
        delta: Value,
    },
    /// Stochastic assignment from a policy.
    Stochastic {
        /// Target variable.
        variable: VariableId,
        /// Sampling policy.
        policy: StochasticPolicy,
    },
    /// Soft intervention replacing the structural mechanism.
    Soft {
        /// Target variable.
        variable: VariableId,
        /// Replacement mechanism description.
        mechanism: MechanismOverride,
    },
    /// Transform of the node's generated value: the mechanism runs as usual (parents and
    /// noise flow through), then the expression replaces the generated value, so the
    /// effect propagates to descendants while the node keeps its own noise. The
    /// expression may read [`ValueExpr::Generated`].
    Transform {
        /// Target variable.
        variable: VariableId,
        /// Expression of the generated value and of other variables' current values.
        expr: ValueExpr,
        /// Declared reads; must equal the variables `expr` references.
        reads: PolicyReads,
    },
    /// Replacement of the node's assignment by an expression of other variables: the
    /// mechanism is discarded, so incoming edges are severed (a computed `do`). The
    /// expression cannot read [`ValueExpr::Generated`].
    Assign {
        /// Target variable.
        variable: VariableId,
        /// Expression of other variables' current values.
        expr: ValueExpr,
        /// Declared reads; must equal the variables `expr` references.
        reads: PolicyReads,
    },
    /// Ordered temporal sequence of interventions.
    Sequence(InterventionSequence),
}

impl CanonicalEncode for Intervention {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Set { variable, value } => {
                writer.word(1);
                variable.encode(writer);
                value.encode(writer);
            }
            Self::Shift { variable, delta } => {
                writer.word(2);
                variable.encode(writer);
                delta.encode(writer);
            }
            Self::Stochastic { variable, policy } => {
                writer.word(3);
                variable.encode(writer);
                policy.encode(writer);
            }
            Self::Soft { variable, mechanism } => {
                writer.word(4);
                variable.encode(writer);
                writer.text(&mechanism.family_id);
                writer.reals(&mechanism.parameters);
            }
            Self::Transform { variable, expr, reads } => {
                writer.word(5);
                variable.encode(writer);
                expr.encode(writer);
                reads.encode(writer);
            }
            Self::Assign { variable, expr, reads } => {
                writer.word(6);
                variable.encode(writer);
                expr.encode(writer);
                reads.encode(writer);
            }
            Self::Sequence(sequence) => {
                writer.word(7).word(count(sequence.steps.len()));
                for step in sequence.steps.as_ref() {
                    step.intervention.encode(writer);
                    step.temporal.encode(writer);
                }
            }
        }
    }
}

impl Intervention {
    /// Transform of the generated value; the declared reads are the expression's own.
    #[must_use]
    pub fn transform(variable: VariableId, expr: ValueExpr) -> Self {
        let reads = expr.reads();
        Self::Transform { variable, expr, reads }
    }

    /// Computed replacement of the assignment; the declared reads are the expression's own.
    #[must_use]
    pub fn assign(variable: VariableId, expr: ValueExpr) -> Self {
        let reads = expr.reads();
        Self::Assign { variable, expr, reads }
    }

    /// Hard set intervention.
    #[must_use]
    pub const fn set(variable: VariableId, value: Value) -> Self {
        Self::Set { variable, value }
    }

    /// Additive shift intervention.
    #[must_use]
    pub const fn shift(variable: VariableId, delta: Value) -> Self {
        Self::Shift { variable, delta }
    }

    /// Stochastic intervention.
    #[must_use]
    pub const fn stochastic(variable: VariableId, policy: StochasticPolicy) -> Self {
        Self::Stochastic { variable, policy }
    }

    /// Soft mechanism override.
    #[must_use]
    pub fn soft(variable: VariableId, mechanism: MechanismOverride) -> Self {
        Self::Soft { variable, mechanism }
    }

    /// Temporal sequence.
    #[must_use]
    pub fn sequence(seq: InterventionSequence) -> Self {
        Self::Sequence(seq)
    }

    /// Every variable this intervention assigns, including Sequence steps.
    #[must_use]
    pub fn target_variables(&self) -> Vec<VariableId> {
        match self {
            Self::Set { variable, .. }
            | Self::Shift { variable, .. }
            | Self::Stochastic { variable, .. }
            | Self::Soft { variable, .. }
            | Self::Transform { variable, .. }
            | Self::Assign { variable, .. } => vec![*variable],
            Self::Sequence(seq) => {
                let mut variables = Vec::new();
                for step in seq.steps.iter() {
                    for variable in step.intervention.target_variables() {
                        if !variables.contains(&variable) {
                            variables.push(variable);
                        }
                    }
                }
                variables
            }
        }
    }

    /// Variable targeted by this intervention, when unique (not a multi-target sequence).
    #[must_use]
    pub fn primary_variable(&self) -> Option<VariableId> {
        match self {
            Self::Set { variable, .. }
            | Self::Shift { variable, .. }
            | Self::Stochastic { variable, .. }
            | Self::Soft { variable, .. }
            | Self::Transform { variable, .. }
            | Self::Assign { variable, .. } => Some(*variable),
            Self::Sequence(seq) => {
                if seq.steps.is_empty() {
                    return None;
                }
                let first = seq.steps[0].intervention.primary_variable()?;
                // The first child was already checked above. Rechecking it here
                // makes a chain of singleton sequences exponential in depth.
                if seq
                    .steps
                    .iter()
                    .skip(1)
                    .all(|s| s.intervention.primary_variable() == Some(first))
                {
                    Some(first)
                } else {
                    None
                }
            }
        }
    }

    /// Whether the intervention can leave the process: closed-form interventions are
    /// portable, a callback program only as a verifiable recovery record, and a program
    /// with a runtime-bound callback not at all. Encoders for artifacts, identities and
    /// the wire call this and refuse on [`Portability::RuntimeBound`].
    #[must_use]
    pub fn portability(&self) -> Portability {
        match self {
            Self::Transform { expr, .. } | Self::Assign { expr, .. } => expr.portability(),
            Self::Sequence(seq) => seq
                .steps
                .iter()
                .fold(Portability::Portable, |acc, step| acc.and(step.intervention.portability())),
            Self::Set { .. } | Self::Shift { .. } | Self::Stochastic { .. } | Self::Soft { .. } => {
                Portability::Portable
            }
        }
    }

    /// Validate nested policies and sequences.
    ///
    /// # Errors
    ///
    /// Invalid stochastic policy or empty sequence.
    pub fn validate(&self) -> Result<(), InterventionError> {
        match self {
            // The symbolic marker is a label, not a level; only a genuine
            // non-finite float is a malformed assignment.
            Self::Set { value: level, .. } | Self::Shift { delta: level, .. } => {
                if matches!(level, Value::Float64(v) if !v.is_finite()) {
                    return Err(InterventionError::NonFiniteLevel);
                }
                Ok(())
            }
            Self::Soft { mechanism, .. } => {
                if mechanism.parameters.iter().any(|p| !p.is_finite()) {
                    return Err(InterventionError::NonFiniteLevel);
                }
                Ok(())
            }
            Self::Stochastic { policy, .. } => policy.validate(),
            Self::Transform { expr, reads, .. } => expr
                .validate_program(true, reads)
                .map_err(|fault| InterventionError::Expression { fault }),
            Self::Assign { expr, reads, .. } => expr
                .validate_program(false, reads)
                .map_err(|fault| InterventionError::Expression { fault }),
            Self::Sequence(seq) => {
                if seq.is_empty() {
                    return Err(InterventionError::EmptySequence);
                }
                for step in seq.steps.iter() {
                    step.temporal.validate()?;
                    step.intervention.validate()?;
                }
                Ok(())
            }
        }
    }
}

/// Which parameter of a [`StochasticPolicy`] is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StochasticPolicyFault {
    /// Refused: Bernoulli p must be finite and in [0, 1]
    #[error("Bernoulli p must be finite and in [0, 1]")]
    BernoulliProbability,
    /// Refused: Gaussian mean must be finite
    #[error("Gaussian mean must be finite")]
    GaussianMean,
    /// Refused: Gaussian variance must be finite and > 0
    #[error("Gaussian variance must be finite and > 0")]
    GaussianVariance,
    /// Refused: categorical probs must be non-empty
    #[error("categorical probs must be non-empty")]
    CategoricalEmpty,
    /// Refused: categorical probs must be finite and >= 0
    #[error("categorical probs must be finite and >= 0")]
    CategoricalEntries,
    /// Refused: categorical probs must sum to a positive value
    #[error("categorical probs must sum to a positive value")]
    CategoricalTotal,
}

/// Errors from intervention construction or validation.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum InterventionError {
    /// Stochastic policy parameters are invalid.
    #[error("invalid stochastic policy: {0}")]
    InvalidStochasticPolicy(StochasticPolicyFault),
    /// A hard level, shift, or soft-mechanism parameter is NaN or infinite.
    #[error("intervention level must be finite")]
    NonFiniteLevel,
    /// Sequence has no steps.
    #[error("intervention sequence is empty")]
    EmptySequence,
    /// Sustained window has `until < from`.
    #[error("invalid temporal window [{from}, {until}]")]
    InvalidTemporalWindow {
        /// Window start.
        from: i32,
        /// Window end.
        until: i32,
    },
    /// Dynamic schedule has no active offsets.
    #[error("TemporalPolicy::Dynamic requires a non-empty active_at schedule")]
    EmptyDynamicSchedule,
    /// A transform or assignment expression is not a valid node program.
    #[error("invalid intervention expression: {fault}")]
    Expression {
        /// What is wrong with the expression.
        fault: ExpressionFault,
    },
    /// A sequence already names a policy on each step.
    #[error("a sequence carries its own temporal policy and cannot take another")]
    TimedSequence,
    /// An initial-state law was not named.
    #[error("initial-state law name is empty")]
    EmptyInitialLaw,
}

/// Which model surface is being asked to execute an intervention.
///
/// The operation comes from [`Intervention`] and, when the action is temporal,
/// [`TemporalPolicy`]. A surface that cannot perform that operation refuses it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExecutionSurface {
    /// One decision time. No pulse, sustained window, or initial-state law.
    Static,
    /// Finite-horizon delayed feedback. Pulse and sustained sets only.
    Temporal,
    /// Counterfactual worlds on a static structural model.
    Counterfactual,
    /// Cyclic model. Without a named family this surface admits the operation
    /// every cyclic family performs: an untimed [`Intervention::Set`].
    /// [`CyclicFamily`] admits the further operations of one family.
    Cyclic,
}

impl ExecutionSurface {
    /// Every surface, in a stable order.
    pub const ALL: [Self; 4] = [Self::Static, Self::Temporal, Self::Counterfactual, Self::Cyclic];

    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Temporal => "temporal",
            Self::Counterfactual => "counterfactual",
            Self::Cyclic => "cyclic",
        }
    }

    /// Classify `intervention` under `temporal` and accept it on this surface.
    ///
    /// `temporal` of `None` is the untimed intervention. A pulse and a sustained
    /// window stay those policies, including a sustained window of one step.
    /// The returned operation is the intervention's own meaning.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] when this surface cannot perform the operation.
    /// [`InterventionError`] when the intervention or the policy is malformed.
    pub fn admit(
        self,
        intervention: &Intervention,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<InterventionOperation, AdmitError> {
        let operation = intervention.operation(temporal)?;
        self.admit_operation(&operation)?;
        Ok(operation)
    }

    /// Accept `operation` on this surface.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] naming the operation this surface cannot perform.
    pub fn admit_operation(
        self,
        operation: &InterventionOperation,
    ) -> Result<(), CapabilityRefusal> {
        if self.supports(operation) { Ok(()) } else { Err(self.refusal(operation)) }
    }

    /// Whether this surface performs `operation` without changing it.
    #[must_use]
    pub fn supports(self, operation: &InterventionOperation) -> bool {
        match self {
            Self::Static | Self::Counterfactual => matches!(
                operation,
                InterventionOperation::Set { .. }
                    | InterventionOperation::Shift { .. }
                    | InterventionOperation::Stochastic { .. }
                    | InterventionOperation::Soft { .. }
                    | InterventionOperation::Transform { .. }
                    | InterventionOperation::Assign { .. }
                    | InterventionOperation::Sequence(_)
            ),
            Self::Cyclic => matches!(operation, InterventionOperation::Set { .. }),
            Self::Temporal => match operation {
                InterventionOperation::Pulse { action, .. }
                | InterventionOperation::Sustained { action, .. } => {
                    matches!(action.as_ref(), InterventionOperation::Set { .. })
                }
                _ => false,
            },
        }
    }

    fn refusal(self, operation: &InterventionOperation) -> CapabilityRefusal {
        let name = refusal_name(operation);
        let reason = match (self, name) {
            (Self::Temporal, "shift") => {
                "a shift is an additive change on the factual scale and is not executed as a set"
            }
            (Self::Temporal, "soft") => {
                "a soft intervention replaces a mechanism and is not executed as a set"
            }
            (Self::Temporal, "set") => {
                "an untimed set is a hard clamp and a finite-horizon rollout does not hard-clamp"
            }
            (_, "shift") => {
                "a shift is an additive change on the factual scale and is not a hard clamp"
            }
            (_, "soft") => "a soft intervention replaces a mechanism and is not a hard clamp",
            (_, "pulse") => {
                "a pulse is one intervention time and is not a hard clamp and is not a sustained \
                 action"
            }
            (_, "sustained") => {
                "a sustained action holds its level across its window and is not a pulse and is \
                 not a hard clamp"
            }
            (_, "initial_state_change") => {
                "an initial-state change replaces the initial-state law and is not a hard clamp"
            }
            (_, "stochastic") => "a stochastic assignment is not a hard clamp",
            (_, "transform") => "a transform runs the mechanism and is not a hard clamp",
            (_, "assign") => "an assignment discards the mechanism and is not a hard clamp",
            (_, "sequence") => "a sequence is not a hard clamp",
            (_, "dynamic") => {
                "a dynamic schedule is not a pulse or a sustained action and is not a hard clamp"
            }
            (_, "set") => "an untimed set is a hard clamp and this surface does not hard-clamp",
            _ => "this surface does not perform that operation",
        };
        CapabilityRefusal { surface: self, operation: name, reason }
    }
}

impl std::fmt::Display for ExecutionSurface {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Model family behind [`ExecutionSurface::Cyclic`].
///
/// Each family admits its own operations. A refusal is a [`CapabilityRefusal`]
/// on the cyclic surface, and [`Self::next_attempt`] names what to declare
/// instead. No operation is rewritten into another one:
///
/// - Both equilibrium families admit an untimed hard clamp
///   ([`InterventionOperation::Set`]), an additive shock on the structural
///   assignment ([`InterventionOperation::Shift`]) and a mechanism replacement
///   ([`InterventionOperation::Soft`]).
/// - The continuous-time family admits an untimed hard clamp held from the
///   initial time, a drift-mechanism replacement, and a change of an
///   initial-state law ([`InterventionOperation::InitialState`]).
///
/// The bare [`ExecutionSurface::Cyclic`] keeps admitting only the untimed set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CyclicFamily {
    /// Linear equilibrium `X = KX + c + ε` with a zero structural diagonal.
    LinearEquilibrium,
    /// Nonlinear additive-noise equilibrium.
    NonlinearEquilibrium,
    /// Deterministic continuous-time initial-value dynamics.
    ContinuousTime,
}

/// Declare the system on the continuous-time family, or run the schedule on
/// the temporal surface.
pub const CYCLIC_CONTINUOUS_TIME_ATTEMPT: NextAttempt = NextAttempt::new(
    "continuous_time_family",
    "declare the system on the continuous-time family, where an untimed set clamps from the \
     initial time and the initial state is its own law, or run the step schedule on the \
     temporal surface",
);

/// Evaluate each level of a stochastic assignment as its own hard-clamp world.
pub const CYCLIC_PER_LEVEL_CLAMP_ATTEMPT: NextAttempt = NextAttempt::new(
    "per_level_hard_clamp_worlds",
    "evaluate each level as its own hard-clamp world and mix the world results under the \
     declared assignment law",
);

/// Declare an expression intervention as a soft mechanism replacement.
pub const CYCLIC_SOFT_REPLACEMENT_ATTEMPT: NextAttempt = NextAttempt::new(
    "soft_mechanism_replacement",
    "declare the replacement equation as a soft intervention of family `constant`, \
     `linear_mechanism`, or `additive_shift`",
);

/// Declare a drift input or an initial-state change instead of a trajectory shift.
pub const CYCLIC_DRIFT_INPUT_ATTEMPT: NextAttempt = NextAttempt::new(
    "drift_input_or_initial_state",
    "declare a soft `additive_shift` drift input, or an initial-state change of the shifted \
     variable",
);

/// Declare an untimed clamp, or run the step schedule on the temporal surface.
pub const CYCLIC_UNTIMED_CLAMP_ATTEMPT: NextAttempt = NextAttempt::new(
    "untimed_clamp_or_temporal_surface",
    "declare an untimed set, which clamps from the initial time over the whole horizon, or run \
     the step schedule on the temporal surface",
);

/// Declare one of the operations the family performs.
pub const CYCLIC_SUPPORTED_OPERATION_ATTEMPT: NextAttempt = NextAttempt::new(
    "supported_cyclic_operation",
    "declare a hard clamp, a soft mechanism replacement, an additive shock, or on the \
     continuous-time family an initial-state change",
);

impl CyclicFamily {
    /// Every family, in a stable order.
    pub const ALL: [Self; 3] =
        [Self::LinearEquilibrium, Self::NonlinearEquilibrium, Self::ContinuousTime];

    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearEquilibrium => "linear_equilibrium",
            Self::NonlinearEquilibrium => "nonlinear_equilibrium",
            Self::ContinuousTime => "continuous_time",
        }
    }

    /// Whether this family performs `operation` without changing it.
    #[must_use]
    pub fn supports(self, operation: &InterventionOperation) -> bool {
        match self {
            Self::LinearEquilibrium | Self::NonlinearEquilibrium => matches!(
                operation,
                InterventionOperation::Set { .. }
                    | InterventionOperation::Shift { .. }
                    | InterventionOperation::Soft { .. }
            ),
            Self::ContinuousTime => matches!(
                operation,
                InterventionOperation::Set { .. }
                    | InterventionOperation::Soft { .. }
                    | InterventionOperation::InitialState { .. }
            ),
        }
    }

    /// Classify `intervention` under `temporal` and accept it on this family.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] on [`ExecutionSurface::Cyclic`] when this family
    /// cannot perform the operation; [`InterventionError`] when the intervention
    /// or the policy is malformed.
    pub fn admit(
        self,
        intervention: &Intervention,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<InterventionOperation, AdmitError> {
        let operation = intervention.operation(temporal)?;
        self.admit_operation(&operation)?;
        Ok(operation)
    }

    /// Accept `operation` on this family.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] on [`ExecutionSurface::Cyclic`] naming the operation.
    pub fn admit_operation(
        self,
        operation: &InterventionOperation,
    ) -> Result<(), CapabilityRefusal> {
        if self.supports(operation) {
            return Ok(());
        }
        let name = self.refused_name(operation);
        Err(CapabilityRefusal {
            surface: ExecutionSurface::Cyclic,
            operation: name,
            reason: self.refusal_parts(name).0,
        })
    }

    /// What to declare instead of the refused operation `operation`.
    ///
    /// `operation` is the name a [`CapabilityRefusal`] from this family carries.
    #[must_use]
    pub fn next_attempt(self, operation: &str) -> NextAttempt {
        self.refusal_parts(operation).1
    }

    /// The timed wrapper is named when its action is one this family performs
    /// untimed; otherwise the refused inner action is named.
    fn refused_name(self, operation: &InterventionOperation) -> &'static str {
        match operation {
            InterventionOperation::Pulse { action, .. }
            | InterventionOperation::Sustained { action, .. }
            | InterventionOperation::Dynamic { action, .. } => {
                if self.supports(action) {
                    operation.name()
                } else {
                    self.refused_name(action)
                }
            }
            other => other.name(),
        }
    }

    fn refusal_parts(self, operation: &str) -> (&'static str, NextAttempt) {
        let equilibrium = !matches!(self, Self::ContinuousTime);
        match (equilibrium, operation) {
            (true, "pulse" | "sustained" | "dynamic" | "sequence") => (
                "a static equilibrium has no time axis, so a timed action is not an equilibrium \
                 intervention",
                CYCLIC_CONTINUOUS_TIME_ATTEMPT,
            ),
            (true, "initial_state_change") => (
                "an equilibrium has no initial-state law; solver initialization is numerical and \
                 is not a causal initial state",
                CYCLIC_CONTINUOUS_TIME_ATTEMPT,
            ),
            (false, "shift") => (
                "a shift on a trajectory could be a state jump or a drift input, and the \
                 continuous-time family does not choose between them",
                CYCLIC_DRIFT_INPUT_ATTEMPT,
            ),
            (false, "pulse" | "sustained" | "dynamic" | "sequence") => (
                "a step-indexed schedule is not a real-time action of the continuous-time \
                 family, whose clamps hold from the initial time",
                CYCLIC_UNTIMED_CLAMP_ATTEMPT,
            ),
            (_, "stochastic") => (
                "a stochastic assignment needs its law joined with the model's disturbance, \
                 which this family does not declare",
                CYCLIC_PER_LEVEL_CLAMP_ATTEMPT,
            ),
            (_, "transform" | "assign") => (
                "an expression intervention inside a feedback loop is a new simultaneous \
                 equation that this family does not compile from an expression",
                CYCLIC_SOFT_REPLACEMENT_ATTEMPT,
            ),
            _ => (
                "this cyclic family does not perform that operation",
                CYCLIC_SUPPORTED_OPERATION_ATTEMPT,
            ),
        }
    }
}

impl std::fmt::Display for CyclicFamily {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal to perform an intervention, or a malformed intervention at admission.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AdmitError {
    /// The intervention or its policy is malformed.
    #[error(transparent)]
    Invalid(#[from] InterventionError),
    /// The surface cannot perform the operation.
    #[error(transparent)]
    Refused(#[from] CapabilityRefusal),
}

/// A surface cannot perform an operation and did not rewrite it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{surface} refuses {operation}: {reason}")]
pub struct CapabilityRefusal {
    /// Surface that refused.
    pub surface: ExecutionSurface,
    /// Operation that was asked for.
    pub operation: &'static str,
    /// Why that operation was not rewritten into one this surface supports.
    pub reason: &'static str,
}

impl CapabilityRefusal {
    /// Operation name.
    #[must_use]
    pub const fn operation(self) -> &'static str {
        self.operation
    }

    /// Why the operation was refused.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        self.reason
    }
}

/// Meaning of an [`Intervention`], timed by [`TemporalPolicy`] when the caller supplied one.
///
/// Equality is the execution identity. A shift is not a set. A soft replacement
/// is not a set. A pulse is not a sustained window. An initial-state change is
/// not a set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InterventionOperation {
    /// Hard assignment `do(variable := value)`.
    Set {
        /// Target variable.
        variable: VariableId,
        /// Assigned value.
        value: Value,
    },
    /// Additive `do(variable := variable + delta)` on the factual scale.
    Shift {
        /// Target variable.
        variable: VariableId,
        /// Delta added to the factual assignment.
        delta: Value,
    },
    /// Stochastic assignment.
    Stochastic {
        /// Target variable.
        variable: VariableId,
        /// Sampling policy.
        policy: StochasticPolicy,
    },
    /// Mechanism replacement.
    Soft {
        /// Target variable.
        variable: VariableId,
        /// Replacement mechanism.
        mechanism: MechanismOverride,
    },
    /// Mechanism runs, then the expression replaces the generated value.
    Transform {
        /// Target variable.
        variable: VariableId,
        /// Expression of the generated value.
        expr: ValueExpr,
        /// Declared reads.
        reads: PolicyReads,
    },
    /// Mechanism discarded; the expression assigns the variable.
    Assign {
        /// Target variable.
        variable: VariableId,
        /// Expression of other variables.
        expr: ValueExpr,
        /// Declared reads.
        reads: PolicyReads,
    },
    /// One intervention time. Not a sustained window and not an untimed set.
    Pulse {
        /// Time offset of the pulse.
        at: i32,
        /// Operation applied at that time.
        action: Box<InterventionOperation>,
    },
    /// Level held on an inclusive window. A window of one step is still sustained.
    Sustained {
        /// First intervened step.
        from: i32,
        /// Last intervened step, inclusive.
        until: i32,
        /// Operation held on the window.
        action: Box<InterventionOperation>,
    },
    /// Rule-tagged schedule. Not a pulse and not a sustained window.
    Dynamic {
        /// Opaque rule handle.
        rule: DynamicRuleId,
        /// Active offsets.
        active_at: Arc<[i32]>,
        /// Operation applied at those offsets.
        action: Box<InterventionOperation>,
    },
    /// Ordered steps. Each step keeps the policy it was declared with.
    Sequence(Arc<[InterventionOperation]>),
    /// Replacement of an initial-state law. Not a clamp of an equilibrium variable.
    InitialState {
        /// Name of the law being replaced.
        law_name: Arc<str>,
        /// Replacement level.
        level: Value,
    },
}

impl InterventionOperation {
    /// Replace the initial-state law `law_name`.
    ///
    /// This is not [`Intervention::Set`].
    ///
    /// # Errors
    ///
    /// [`InterventionError::EmptyInitialLaw`] when `law_name` is empty.
    /// [`InterventionError::NonFiniteLevel`] when `level` is a non-finite float.
    #[must_use = "a refused initial-state change is not an operation"]
    pub fn initial_state(
        law_name: impl Into<Arc<str>>,
        level: Value,
    ) -> Result<Self, InterventionError> {
        let law_name = law_name.into();
        if law_name.is_empty() {
            return Err(InterventionError::EmptyInitialLaw);
        }
        if matches!(level, Value::Float64(value) if !value.is_finite()) {
            return Err(InterventionError::NonFiniteLevel);
        }
        Ok(Self::InitialState { law_name, level })
    }

    /// Stable name of this operation.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Set { .. } => "set",
            Self::Shift { .. } => "shift",
            Self::Stochastic { .. } => "stochastic",
            Self::Soft { .. } => "soft",
            Self::Transform { .. } => "transform",
            Self::Assign { .. } => "assign",
            Self::Pulse { .. } => "pulse",
            Self::Sustained { .. } => "sustained",
            Self::Dynamic { .. } => "dynamic",
            Self::Sequence(_) => "sequence",
            Self::InitialState { .. } => "initial_state_change",
        }
    }
}

impl Intervention {
    /// Operation this intervention performs.
    ///
    /// `temporal` of `None` leaves the intervention untimed. A pulse and a
    /// sustained window wrap the intervention and do not rewrite it: a shift
    /// stays a shift, and a sustained window of one step stays sustained.
    ///
    /// # Errors
    ///
    /// [`InterventionError::TimedSequence`] when `temporal` is supplied for a
    /// sequence. Nested policy and expression errors of [`Self::validate`].
    pub fn operation(
        &self,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<InterventionOperation, InterventionError> {
        if let Some(policy) = temporal {
            policy.validate()?;
            if matches!(self, Self::Sequence(_)) {
                return Err(InterventionError::TimedSequence);
            }
        }
        let untimed = match self {
            Self::Set { variable, value } => {
                InterventionOperation::Set { variable: *variable, value: value.clone() }
            }
            Self::Shift { variable, delta } => {
                InterventionOperation::Shift { variable: *variable, delta: delta.clone() }
            }
            Self::Stochastic { variable, policy } => {
                InterventionOperation::Stochastic { variable: *variable, policy: policy.clone() }
            }
            Self::Soft { variable, mechanism } => {
                InterventionOperation::Soft { variable: *variable, mechanism: mechanism.clone() }
            }
            Self::Transform { variable, expr, reads } => InterventionOperation::Transform {
                variable: *variable,
                expr: expr.clone(),
                reads: reads.clone(),
            },
            Self::Assign { variable, expr, reads } => InterventionOperation::Assign {
                variable: *variable,
                expr: expr.clone(),
                reads: reads.clone(),
            },
            Self::Sequence(seq) => {
                let mut steps = Vec::with_capacity(seq.steps.len());
                for step in seq.steps.iter() {
                    steps.push(step.intervention.operation(Some(&step.temporal))?);
                }
                InterventionOperation::Sequence(Arc::from(steps))
            }
        };
        Ok(match temporal {
            None => untimed,
            Some(TemporalPolicy::Pulse { at }) => {
                InterventionOperation::Pulse { at: *at, action: Box::new(untimed) }
            }
            Some(TemporalPolicy::Sustained { from, until }) => InterventionOperation::Sustained {
                from: *from,
                until: *until,
                action: Box::new(untimed),
            },
            Some(TemporalPolicy::Dynamic { rule, active_at }) => InterventionOperation::Dynamic {
                rule: *rule,
                active_at: Arc::clone(active_at),
                action: Box::new(untimed),
            },
        })
    }
}

/// Name the inner operation when a timed wrapper would hide a shift, a soft
/// replacement, or another action that is not a set.
fn refusal_name(operation: &InterventionOperation) -> &'static str {
    match operation {
        InterventionOperation::Pulse { action, .. }
        | InterventionOperation::Sustained { action, .. }
        | InterventionOperation::Dynamic { action, .. } => match action.as_ref() {
            InterventionOperation::Set { .. } => operation.name(),
            inner => inner.name(),
        },
        other => other.name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_set_primary_variable() {
        let v = VariableId::from_raw(3);
        let i = Intervention::set(v, Value::f64(1.0));
        assert_eq!(i.primary_variable(), Some(v));
        assert!(matches!(i, Intervention::Set { .. }));
        i.validate().unwrap();
    }

    #[test]
    fn stochastic_bernoulli_validates() {
        let v = VariableId::from_raw(0);
        let ok = Intervention::stochastic(v, StochasticPolicy::bernoulli(0.4));
        ok.validate().unwrap();
        let bad = Intervention::stochastic(v, StochasticPolicy::bernoulli(1.5));
        assert!(bad.validate().is_err());
    }

    #[test]
    fn non_finite_levels_fail_validation() {
        let v = VariableId::from_raw(0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                Intervention::set(v, Value::f64(bad)).validate(),
                Err(InterventionError::NonFiniteLevel)
            );
            assert_eq!(
                Intervention::shift(v, Value::f64(bad)).validate(),
                Err(InterventionError::NonFiniteLevel)
            );
            assert_eq!(
                Intervention::soft(v, MechanismOverride::additive_shift(bad)).validate(),
                Err(InterventionError::NonFiniteLevel)
            );
        }
        assert!(
            Intervention::stochastic(v, StochasticPolicy::gaussian(f64::NAN, 1.0))
                .validate()
                .is_err()
        );
        // The symbolic marker is not a numeric level and stays valid.
        assert!(Intervention::set(v, Value::symbolic_intervention()).validate().is_ok());
        assert!(Intervention::set(v, Value::f64(-0.0)).validate().is_ok());
    }

    #[test]
    fn policy_eq_and_hash_agree_on_signed_zero_and_nan() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash = |x: &dyn Fn(&mut DefaultHasher)| {
            let mut h = DefaultHasher::new();
            x(&mut h);
            h.finish()
        };
        let pos = StochasticPolicy::gaussian(0.0, 1.0);
        let neg = StochasticPolicy::gaussian(-0.0, 1.0);
        // Whatever Eq decides, Hash must decide the same way.
        assert_eq!(pos == neg, hash(&|h| pos.hash(h)) == hash(&|h| neg.hash(h)));
        let nan = StochasticPolicy::bernoulli(f64::NAN);
        assert_eq!(nan, nan.clone());
        assert_eq!(hash(&|h| nan.hash(h)), hash(&|h| nan.clone().hash(h)));
        let soft = MechanismOverride::additive_shift(-0.0);
        let soft_pos = MechanismOverride::additive_shift(0.0);
        assert_eq!(soft == soft_pos, hash(&|h| soft.hash(h)) == hash(&|h| soft_pos.hash(h)));
        let nan_soft = MechanismOverride::additive_shift(f64::NAN);
        assert_eq!(nan_soft, nan_soft.clone());
    }

    #[test]
    fn sequence_rejects_empty() {
        let seq = Intervention::sequence(InterventionSequence::new(Vec::new()));
        assert!(matches!(seq.validate(), Err(InterventionError::EmptySequence)));
    }

    #[test]
    fn sequence_uniform_primary() {
        let v = VariableId::from_raw(1);
        let seq = Intervention::sequence(InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(v, Value::f64(1.0)),
                TemporalPolicy::pulse(0),
            ),
            SequencedIntervention::new(
                Intervention::shift(v, Value::f64(0.1)),
                TemporalPolicy::sustained(1, 3),
            ),
        ]));
        seq.validate().unwrap();
        assert_eq!(seq.primary_variable(), Some(v));
    }

    #[test]
    fn soft_override_helpers() {
        let m = MechanismOverride::constant(2.0);
        assert_eq!(&*m.family_id, "constant");
        assert_eq!(m.parameters.as_ref(), &[2.0]);
    }

    #[test]
    fn transform_and_assign_validate_their_programs() {
        let x = VariableId::from_raw(0);
        let z = VariableId::from_raw(1);
        let floor =
            ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
        let transform = Intervention::transform(x, floor.clone());
        transform.validate().unwrap();
        assert_eq!(transform.primary_variable(), Some(x));
        // An assignment replaces the generated value, so it cannot read it.
        assert_eq!(
            Intervention::assign(x, floor).validate(),
            Err(InterventionError::Expression { fault: ExpressionFault::GeneratedInAssignment })
        );
        let from_z = ValueExpr::constant(2.0).mul(ValueExpr::var(z));
        Intervention::assign(x, from_z.clone()).validate().unwrap();
        let undeclared =
            Intervention::Assign { variable: x, expr: from_z, reads: PolicyReads::default() };
        assert_eq!(
            undeclared.validate(),
            Err(InterventionError::Expression { fault: ExpressionFault::DeclaredReadsMismatch })
        );
        let boolean = Intervention::transform(x, ValueExpr::var(z).gt(ValueExpr::constant(0.0)));
        assert!(matches!(
            boolean.validate(),
            Err(InterventionError::Expression { fault: ExpressionFault::OutputType { .. } })
        ));
    }

    #[test]
    fn dynamic_policy_validates() {
        use crate::ids::DynamicRuleId;
        let p = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [0, 2, 5]);
        p.validate().unwrap();
        assert!(p.is_active_at(2));
        assert!(!p.is_active_at(1));
        assert_eq!(p.active_offsets().unwrap().as_ref(), &[0, 2, 5]);
        let empty = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), []);
        assert!(matches!(empty.validate(), Err(InterventionError::EmptyDynamicSchedule)));
    }

    fn refusal_operation(error: AdmitError) -> &'static str {
        match error {
            AdmitError::Refused(refusal) => refusal.operation(),
            AdmitError::Invalid(error) => panic!("malformed intervention: {error}"),
        }
    }

    #[test]
    fn the_same_set_is_one_operation_on_every_surface_that_supports_it() {
        let set = Intervention::set(VariableId::from_raw(0), Value::f64(1.0));
        let static_op = ExecutionSurface::Static.admit(&set, None).unwrap();
        let counterfactual_op = ExecutionSurface::Counterfactual.admit(&set, None).unwrap();
        let cyclic_op = ExecutionSurface::Cyclic.admit(&set, None).unwrap();
        assert_eq!(static_op, counterfactual_op);
        assert_eq!(static_op, cyclic_op);
        assert!(matches!(static_op, InterventionOperation::Set { .. }));
        let refused = ExecutionSurface::Temporal.admit(&set, None).unwrap_err();
        let AdmitError::Refused(refusal) = refused else {
            panic!("an untimed set is not a finite-horizon action");
        };
        assert_eq!(refusal.operation(), "set");
    }

    #[test]
    fn shift_soft_pulse_sustained_and_initial_state_are_not_a_set() {
        let variable = VariableId::from_raw(0);
        let set = Intervention::set(variable, Value::f64(1.0));
        let shift = Intervention::shift(variable, Value::f64(1.0));
        let soft = Intervention::soft(variable, MechanismOverride::constant(1.0));
        let shift_op = ExecutionSurface::Static.admit(&shift, None).unwrap();
        let soft_op = ExecutionSurface::Static.admit(&soft, None).unwrap();
        assert!(matches!(shift_op, InterventionOperation::Shift { .. }));
        assert!(matches!(soft_op, InterventionOperation::Soft { .. }));
        assert_ne!(shift_op, ExecutionSurface::Static.admit(&set, None).unwrap());
        assert_ne!(soft_op, ExecutionSurface::Static.admit(&set, None).unwrap());

        let pulse =
            ExecutionSurface::Temporal.admit(&set, Some(&TemporalPolicy::pulse(0))).unwrap();
        let sustained =
            ExecutionSurface::Temporal.admit(&set, Some(&TemporalPolicy::sustained(0, 0))).unwrap();
        assert!(matches!(pulse, InterventionOperation::Pulse { .. }));
        assert!(matches!(sustained, InterventionOperation::Sustained { .. }));
        assert_ne!(pulse, sustained);
        assert_ne!(pulse.name(), "set");
        assert_ne!(sustained.name(), "set");
        assert_ne!(pulse.name(), sustained.name());

        for surface in
            [ExecutionSurface::Static, ExecutionSurface::Counterfactual, ExecutionSurface::Cyclic]
        {
            let admitted_shift = surface.supports(&shift_op);
            let admitted_soft = surface.supports(&soft_op);
            assert_eq!(admitted_shift, surface != ExecutionSurface::Cyclic);
            assert_eq!(admitted_soft, surface != ExecutionSurface::Cyclic);
            if surface == ExecutionSurface::Cyclic {
                assert_eq!(refusal_operation(surface.admit(&shift, None).unwrap_err()), "shift");
                assert_eq!(refusal_operation(surface.admit(&soft, None).unwrap_err()), "soft");
            }
            assert_eq!(
                refusal_operation(
                    surface.admit(&set, Some(&TemporalPolicy::pulse(0))).unwrap_err()
                ),
                "pulse"
            );
            assert_eq!(
                refusal_operation(
                    surface.admit(&set, Some(&TemporalPolicy::sustained(0, 0))).unwrap_err()
                ),
                "sustained"
            );
        }

        let initial = InterventionOperation::initial_state("x0", Value::f64(2.0)).unwrap();
        assert_eq!(initial.name(), "initial_state_change");
        for surface in ExecutionSurface::ALL {
            let refusal = surface.admit_operation(&initial).unwrap_err();
            assert_eq!(refusal.operation(), "initial_state_change");
            assert!(refusal.reason().contains("not a hard clamp"));
        }
        assert_eq!(
            InterventionOperation::initial_state("", Value::f64(2.0)).unwrap_err(),
            InterventionError::EmptyInitialLaw
        );
    }

    #[test]
    fn cyclic_families_admit_their_own_operations_and_keep_the_bare_surface() {
        let variable = VariableId::from_raw(0);
        let set = Intervention::set(variable, Value::f64(1.0));
        let shift = Intervention::shift(variable, Value::f64(1.0));
        let soft = Intervention::soft(variable, MechanismOverride::constant(1.0));
        let initial = InterventionOperation::initial_state("x0", Value::f64(2.0)).unwrap();
        for family in [CyclicFamily::LinearEquilibrium, CyclicFamily::NonlinearEquilibrium] {
            for intervention in [&set, &shift, &soft] {
                let operation = family.admit(intervention, None).unwrap();
                assert_eq!(operation, intervention.operation(None).unwrap());
            }
            let refusal = family.admit_operation(&initial).unwrap_err();
            assert_eq!(refusal.surface, ExecutionSurface::Cyclic);
            assert_eq!(refusal.operation(), "initial_state_change");
            assert_eq!(family.next_attempt(refusal.operation()), CYCLIC_CONTINUOUS_TIME_ATTEMPT);
        }
        let continuous = CyclicFamily::ContinuousTime;
        continuous.admit(&set, None).unwrap();
        continuous.admit(&soft, None).unwrap();
        continuous.admit_operation(&initial).unwrap();
        let refusal = refusal_operation(continuous.admit(&shift, None).unwrap_err());
        assert_eq!(refusal, "shift");
        assert_eq!(continuous.next_attempt(refusal), CYCLIC_DRIFT_INPUT_ATTEMPT);

        // The bare surface is unchanged: only the untimed set.
        assert!(ExecutionSurface::Cyclic.admit(&set, None).is_ok());
        assert_eq!(
            refusal_operation(ExecutionSurface::Cyclic.admit(&shift, None).unwrap_err()),
            "shift"
        );
        assert_eq!(
            refusal_operation(ExecutionSurface::Cyclic.admit(&soft, None).unwrap_err()),
            "soft"
        );
    }

    #[test]
    fn every_cyclic_family_refusal_names_a_next_attempt() {
        let variable = VariableId::from_raw(0);
        let other = VariableId::from_raw(1);
        let set = Intervention::set(variable, Value::f64(1.0));
        let shift = Intervention::shift(variable, Value::f64(1.0));
        let refused = [
            (Intervention::stochastic(variable, StochasticPolicy::bernoulli(0.5)), None),
            (Intervention::transform(variable, ValueExpr::generated()), None),
            (Intervention::assign(variable, ValueExpr::var(other)), None),
            (set.clone(), Some(TemporalPolicy::pulse(0))),
            (set.clone(), Some(TemporalPolicy::sustained(0, 2))),
            (shift.clone(), Some(TemporalPolicy::pulse(0))),
            (
                Intervention::sequence(InterventionSequence::new(vec![
                    SequencedIntervention::new(set, TemporalPolicy::pulse(0)),
                ])),
                None,
            ),
        ];
        for family in CyclicFamily::ALL {
            for (intervention, temporal) in &refused {
                let name =
                    refusal_operation(family.admit(intervention, temporal.as_ref()).unwrap_err());
                let attempt = family.next_attempt(name);
                assert!(!attempt.name().is_empty() && !attempt.sentence().is_empty());
                assert_ne!(attempt, CYCLIC_SUPPORTED_OPERATION_ATTEMPT, "{family} {name}");
            }
        }
        // A timed shift on an equilibrium names the timing, not the shift it performs untimed.
        let timed_shift = CyclicFamily::LinearEquilibrium
            .admit(&shift, Some(&TemporalPolicy::pulse(0)))
            .unwrap_err();
        assert_eq!(refusal_operation(timed_shift), "pulse");
        // On the continuous-time family the shift itself is what is refused.
        let timed_shift = CyclicFamily::ContinuousTime
            .admit(&shift, Some(&TemporalPolicy::pulse(0)))
            .unwrap_err();
        assert_eq!(refusal_operation(timed_shift), "shift");
    }

    #[test]
    fn intervention_encodings_are_unchanged_by_cyclic_admission() {
        let variable = VariableId::from_raw(7);
        let cases = [
            Intervention::set(variable, Value::f64(1.5)),
            Intervention::shift(variable, Value::f64(-0.25)),
            Intervention::soft(
                variable,
                MechanismOverride::named("linear_mechanism", vec![0.1, 0.0, 0.4]),
            ),
        ];
        for (tag, intervention) in [1_u64, 2, 4].into_iter().zip(&cases) {
            let mut encoded = CanonicalWriter::new("intervention-encoding");
            intervention.encode(&mut encoded);
            let mut expected = CanonicalWriter::new("intervention-encoding");
            expected.word(tag).word(7);
            match intervention {
                Intervention::Set { value, .. } | Intervention::Shift { delta: value, .. } => {
                    value.encode(&mut expected);
                }
                Intervention::Soft { mechanism, .. } => {
                    expected.text(&mechanism.family_id).reals(&mechanism.parameters);
                }
                _ => unreachable!("only the three cyclic action kinds are encoded here"),
            }
            assert_eq!(encoded.finish(), expected.finish());
            let before = encoded.finish();
            let admitted = CyclicFamily::NonlinearEquilibrium.admit(intervention, None).unwrap();
            assert_eq!(admitted, intervention.operation(None).unwrap());
            let mut after = CanonicalWriter::new("intervention-encoding");
            intervention.encode(&mut after);
            assert_eq!(after.finish(), before);
        }
    }
}

#[cfg(test)]
mod bounded_temporal_offsets_tests {
    use super::*;
    use crate::execution::AdmissionPool;
    fn limits() -> TemporalMaterializationLimits {
        TemporalMaterializationLimits {
            max_offsets: 4,
            max_work: 4,
            max_bytes: 1024,
            max_graph_nodes: 4,
            max_graph_edges: 4,
        }
    }
    #[test]
    fn enormous_ranges_are_lazy_and_refuse_before_allocation() {
        let policy = TemporalPolicy::sustained(i32::MIN, i32::MAX);
        assert_eq!(policy.offset_count().unwrap(), 1_u64 << 32);
        assert!(matches!(
            policy.active_offsets(),
            Err(TemporalMaterializationError::Limit { dimension: "offsets" })
        ));
        assert_eq!(
            policy.offsets().unwrap().take(3).collect::<Vec<_>>(),
            [i32::MIN, i32::MIN + 1, i32::MIN + 2]
        );
        let ctx = crate::ExecutionContext::for_tests(1);
        assert!(matches!(
            policy.active_offsets_within(&mut TemporalBudget::new(limits()), &ctx),
            Err(TemporalMaterializationError::Limit { dimension: "offsets" })
        ));
    }
    #[test]
    fn offsets_preserve_values_and_clones_hold_the_admission() {
        let pool = AdmissionPool::fixed(1, 1024);
        let ctx = crate::ExecutionContext::for_tests(1).with_admission(pool.clone());
        let policy = TemporalPolicy::sustained(-1, 1);
        let offsets =
            policy.active_offsets_within(&mut TemporalBudget::new(limits()), &ctx).unwrap();
        assert_eq!(offsets.as_slice(), [-1, 0, 1]);
        let free = pool.available().bytes();
        assert!(free < 1024);
        let clone = offsets.clone();
        drop(offsets);
        assert_eq!(pool.available().bytes(), free);
        drop(clone);
        assert_eq!(pool.available().bytes(), 1024);
        ctx.cancellation.cancel();
        assert!(matches!(
            policy.active_offsets_within(&mut TemporalBudget::new(limits()), &ctx),
            Err(TemporalMaterializationError::Cancelled)
        ));
    }
    #[test]
    fn one_budget_is_consumed_across_nested_materializations() {
        let ctx = crate::ExecutionContext::for_tests(1);
        let mut budget = TemporalBudget::new(limits());
        let mut sibling = budget.clone();
        let first =
            TemporalPolicy::sustained(0, 2).active_offsets_within(&mut budget, &ctx).unwrap();
        assert_eq!(first.as_slice(), &[0, 1, 2]);
        assert!(matches!(
            TemporalPolicy::sustained(0, 1).active_offsets_within(&mut sibling, &ctx),
            Err(TemporalMaterializationError::Limit { dimension: "offsets" })
        ));
    }

    #[test]
    fn validation_preflight_bounds_nested_interventions_and_holds_memory() {
        let v = VariableId::from_raw(0);
        let mut item = Intervention::set(v, Value::f64(1.0));
        for _ in 0..20 {
            item = Intervention::Sequence(InterventionSequence::new([SequencedIntervention {
                intervention: item,
                temporal: TemporalPolicy::pulse(0),
            }]));
        }
        let pool = AdmissionPool::fixed(1, 131_072);
        let ctx = crate::ExecutionContext::for_tests(1).with_admission(pool.clone());
        let mut bounds = limits();
        bounds.max_work = 1_000_000;
        bounds.max_bytes = 131_072;
        let mut budget = TemporalBudget::new(bounds);
        let before = pool.available().bytes();
        let lease = admit_temporal_intervention_validation([&item], &mut budget, &ctx).unwrap();
        assert!(pool.available().bytes() < before);
        assert_eq!(item.primary_variable(), Some(v));
        item.validate().unwrap();
        drop(lease);
        assert_eq!(pool.available().bytes(), before);
        ctx.cancellation.cancel();
        assert!(matches!(
            admit_temporal_intervention_validation([&item], &mut budget, &ctx),
            Err(TemporalMaterializationError::Cancelled)
        ));
    }
}

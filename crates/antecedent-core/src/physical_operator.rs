//! Physical operator metadata published before R, D, N, Y, and G integration.
//!
//! Each value records the physical plan for one [`PhysicalEffect`]. Those names
//! mirror [`crate::operator::OperatorEffect`]. The edge's graph stays on
//! [`crate::operator::OperatorGraph`]: causal, provenance, and execution are not
//! chosen here, and this module does not import [`crate::operator`].
//!
//! Source and design dependencies are ids, not rows or a design matrix. Scratch,
//! retained, and output sizes are byte estimates. An absent estimate is not
//! zero: bounded-memory admission refuses it rather than treating the operator
//! as free. Numerical policy is strict, or a tolerance policy that alone carries
//! a declared tolerance. Cancellation points are scan, solve, resample, search,
//! chain, and provider. Capability is the CPU, or a device named explicitly.
//! Device work without that device capability is refused. Reuse is legal only
//! when the dependencies and the numerical policy match; a changed dependency
//! refuses reuse.
//!
//! Nothing here reads a column, allocates scratch, or launches a device.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use crate::text::is_blank;

// Mirrors `OperatorEffect`. The parent may later unify the two enums.
/// Effect of one physical operator.
///
/// The names match [`crate::operator::OperatorEffect`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PhysicalEffect {
    /// Read recorded observations.
    Read,
    /// Select units.
    Select,
    /// Fit a nuisance model.
    Fit,
    /// Use a prior.
    Prior,
    /// Condition on declared variables.
    Condition,
    /// Intervene.
    Intervene,
    /// Transport a law.
    Transport,
    /// Reuse a frozen fit.
    ReuseFrozenFit,
    /// Solve a simultaneous block. Publishes [`CancellationPoint::Solve`].
    Solve,
    /// Scan a temporal recurrence. Publishes [`CancellationPoint::Scan`].
    Scan,
}

impl PhysicalEffect {
    /// Stable `snake_case` name, matching `OperatorEffect`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Select => "select",
            Self::Fit => "fit",
            Self::Prior => "prior",
            Self::Condition => "condition",
            Self::Intervene => "intervene",
            Self::Transport => "transport",
            Self::ReuseFrozenFit => "reuse_frozen_fit",
            Self::Solve => "solve",
            Self::Scan => "scan",
        }
    }

    /// Cancellation point this effect must publish.
    ///
    /// Solve publishes [`CancellationPoint::Solve`] and scan publishes
    /// [`CancellationPoint::Scan`]. Every other effect publishes none of its
    /// own; resample, search, chain, and provider are still declarable.
    #[must_use = "solve and scan publish a cancellation point; other effects publish none"]
    pub const fn required_cancellation_point(self) -> Option<CancellationPoint> {
        match self {
            Self::Solve => Some(CancellationPoint::Solve),
            Self::Scan => Some(CancellationPoint::Scan),
            Self::Read
            | Self::Select
            | Self::Fit
            | Self::Prior
            | Self::Condition
            | Self::Intervene
            | Self::Transport
            | Self::ReuseFrozenFit => None,
        }
    }
}

impl fmt::Display for PhysicalEffect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Id of a source the operator depends on.
///
/// The id is not the source data.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SourceDependencyId(u64);

impl SourceDependencyId {
    /// Id from its raw value.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw id.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SourceDependencyId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "source:{}", self.0)
    }
}

/// Id of a design the operator depends on.
///
/// The id is not the design matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DesignDependencyId(u64);

impl DesignDependencyId {
    /// Id from its raw value.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw id.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for DesignDependencyId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "design:{}", self.0)
    }
}

/// Source and design dependency ids, compared as sets.
///
/// Declaration order and duplicate ids do not change the set. A different id
/// is a changed dependency and refuses reuse.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PhysicalDependencies {
    /// Source ids, sorted and de-duplicated.
    sources: Arc<[SourceDependencyId]>,
    /// Design ids, sorted and de-duplicated.
    designs: Arc<[DesignDependencyId]>,
}

impl PhysicalDependencies {
    /// Source ids, in ascending order, without duplicates.
    #[must_use]
    pub fn sources(&self) -> &[SourceDependencyId] {
        &self.sources
    }

    /// Design ids, in ascending order, without duplicates.
    #[must_use]
    pub fn designs(&self) -> &[DesignDependencyId] {
        &self.designs
    }
}

/// Which byte estimate was absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ByteEstimateKind {
    /// Per-operator scratch.
    Scratch,
    /// Bytes retained after the operator returns.
    Retained,
    /// Bytes written as the operator's output.
    Output,
}

impl ByteEstimateKind {
    /// Stable name used in admission errors.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scratch => "scratch",
            Self::Retained => "retained",
            Self::Output => "output",
        }
    }
}

impl fmt::Display for ByteEstimateKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Scratch, retained, and output byte estimates.
///
/// `None` means the estimate is absent. Absence is not zero: zero is a
/// declared estimate of no bytes, and only a declared estimate can pass
/// bounded-memory admission. [`Default`] is absence, not zeros.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ByteEstimates {
    /// Scratch bytes, or absence.
    scratch: Option<u64>,
    /// Retained bytes, or absence.
    retained: Option<u64>,
    /// Output bytes, or absence.
    output: Option<u64>,
}

impl ByteEstimates {
    /// No scratch, retained, or output estimate.
    #[must_use]
    pub const fn absent() -> Self {
        Self { scratch: None, retained: None, output: None }
    }

    /// Three declared estimates. Zero is a declared zero, not an absent field.
    #[must_use]
    pub const fn declared(scratch: u64, retained: u64, output: u64) -> Self {
        Self { scratch: Some(scratch), retained: Some(retained), output: Some(output) }
    }

    /// Replace the scratch estimate with a declared byte count.
    #[must_use]
    pub const fn with_scratch(mut self, bytes: u64) -> Self {
        self.scratch = Some(bytes);
        self
    }

    /// Replace the retained estimate with a declared byte count.
    #[must_use]
    pub const fn with_retained(mut self, bytes: u64) -> Self {
        self.retained = Some(bytes);
        self
    }

    /// Replace the output estimate with a declared byte count.
    #[must_use]
    pub const fn with_output(mut self, bytes: u64) -> Self {
        self.output = Some(bytes);
        self
    }

    /// Scratch estimate, if one was declared.
    #[must_use = "an absent scratch estimate is not zero bytes"]
    pub const fn scratch(self) -> Option<u64> {
        self.scratch
    }

    /// Retained estimate, if one was declared.
    #[must_use = "an absent retained estimate is not zero bytes"]
    pub const fn retained(self) -> Option<u64> {
        self.retained
    }

    /// Output estimate, if one was declared.
    #[must_use = "an absent output estimate is not zero bytes"]
    pub const fn output(self) -> Option<u64> {
        self.output
    }

    /// Admit the three estimates under a bounded-memory gate.
    ///
    /// # Errors
    ///
    /// [`PhysicalOperatorError::AbsentByteEstimate`] for the first absent field
    /// in scratch, retained, output order. A declared zero is admitted.
    #[must_use = "a refused admission is not permission to allocate"]
    pub const fn bounded_memory_admission(self) -> Result<AdmittedBytes, PhysicalOperatorError> {
        let Some(scratch) = self.scratch else {
            return Err(PhysicalOperatorError::AbsentByteEstimate {
                kind: ByteEstimateKind::Scratch,
            });
        };
        let Some(retained) = self.retained else {
            return Err(PhysicalOperatorError::AbsentByteEstimate {
                kind: ByteEstimateKind::Retained,
            });
        };
        let Some(output) = self.output else {
            return Err(PhysicalOperatorError::AbsentByteEstimate {
                kind: ByteEstimateKind::Output,
            });
        };
        Ok(AdmittedBytes { scratch, retained, output })
    }
}

/// Bytes admitted by [`ByteEstimates::bounded_memory_admission`].
///
/// Every field was declared. A zero field is a declared zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmittedBytes {
    /// Declared scratch bytes.
    scratch: u64,
    /// Declared retained bytes.
    retained: u64,
    /// Declared output bytes.
    output: u64,
}

impl AdmittedBytes {
    /// Declared scratch bytes.
    #[must_use]
    pub const fn scratch(self) -> u64 {
        self.scratch
    }

    /// Declared retained bytes.
    #[must_use]
    pub const fn retained(self) -> u64 {
        self.retained
    }

    /// Declared output bytes.
    #[must_use]
    pub const fn output(self) -> u64 {
        self.output
    }

    /// Scratch plus retained plus output, saturating at [`u64::MAX`].
    #[must_use]
    pub const fn total(self) -> u64 {
        self.scratch.saturating_add(self.retained).saturating_add(self.output)
    }
}

/// Bit pattern of IEEE-754 negative zero.
const NEGATIVE_ZERO_BITS: u64 = 0x8000_0000_0000_0000;

/// Finite, non-negative `value` as a canonical `f64` bit pattern.
///
/// Negative zero becomes positive zero. Any other sign-negative value, and
/// every non-finite value, is rejected.
const fn finite_nonnegative_bits(value: f64) -> Option<u64> {
    if !value.is_finite() {
        None
    } else if value.to_bits() == NEGATIVE_ZERO_BITS {
        Some(0)
    } else if value.is_sign_negative() {
        None
    } else {
        Some(value.to_bits())
    }
}

/// Absolute and relative tolerance carried only by [`NumericalPolicy::Tolerance`].
///
/// Equality is bitwise after negative zero is canonicalized to positive zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DeclaredTolerance {
    /// Canonical bits of the absolute tolerance.
    absolute: u64,
    /// Canonical bits of the relative tolerance.
    relative: u64,
}

impl DeclaredTolerance {
    /// Finite, non-negative absolute and relative tolerances.
    ///
    /// Negative zero is accepted and stored as positive zero. A negative,
    /// infinite, or NaN component is rejected.
    #[must_use = "a rejected tolerance is not a strict numerical policy"]
    pub const fn new(absolute: f64, relative: f64) -> Option<Self> {
        let Some(absolute) = finite_nonnegative_bits(absolute) else {
            return None;
        };
        let Some(relative) = finite_nonnegative_bits(relative) else {
            return None;
        };
        Some(Self { absolute, relative })
    }

    /// Absolute tolerance.
    #[must_use]
    pub const fn absolute(self) -> f64 {
        f64::from_bits(self.absolute)
    }

    /// Relative tolerance.
    #[must_use]
    pub const fn relative(self) -> f64 {
        f64::from_bits(self.relative)
    }
}

/// Numerical policy for one physical operator.
///
/// [`Self::Strict`] carries no tolerance. [`Self::Tolerance`] carries a
/// [`DeclaredTolerance`] and nothing else does.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NumericalPolicy {
    /// Exact policy. There is no tolerance to read.
    Strict,
    /// Comparison or convergence within `tolerance`.
    Tolerance(DeclaredTolerance),
}

impl NumericalPolicy {
    /// Strict policy, with no declared tolerance.
    #[must_use]
    pub const fn strict() -> Self {
        Self::Strict
    }

    /// Tolerance policy. `tolerance` is the declared tolerance.
    #[must_use]
    pub const fn tolerance(tolerance: DeclaredTolerance) -> Self {
        Self::Tolerance(tolerance)
    }

    /// Declared tolerance. Strict policy returns `None`, which is not a zero
    /// tolerance.
    #[must_use = "strict policy has no tolerance; absence is not a zero tolerance"]
    pub const fn declared_tolerance(self) -> Option<DeclaredTolerance> {
        match self {
            Self::Strict => None,
            Self::Tolerance(tolerance) => Some(tolerance),
        }
    }
}

impl fmt::Display for NumericalPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strict => formatter.write_str("strict"),
            Self::Tolerance(tolerance) => write!(
                formatter,
                "tolerance(absolute={}, relative={})",
                tolerance.absolute(),
                tolerance.relative()
            ),
        }
    }
}

/// Where a running operator observes cancellation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CancellationPoint {
    /// Temporal or table scan.
    Scan,
    /// Simultaneous solve.
    Solve,
    /// Resample, including bootstrap replicates.
    Resample,
    /// Bounded search.
    Search,
    /// Markov chain.
    Chain,
    /// External provider call.
    Provider,
}

impl CancellationPoint {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Solve => "solve",
            Self::Resample => "resample",
            Self::Search => "search",
            Self::Chain => "chain",
            Self::Provider => "provider",
        }
    }

    const fn mask(self) -> u8 {
        match self {
            Self::Scan => 0b0000_0001,
            Self::Solve => 0b0000_0010,
            Self::Resample => 0b0000_0100,
            Self::Search => 0b0000_1000,
            Self::Chain => 0b0001_0000,
            Self::Provider => 0b0010_0000,
        }
    }
}

impl fmt::Display for CancellationPoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

const CANCELLATION_POINTS: [CancellationPoint; 6] = [
    CancellationPoint::Scan,
    CancellationPoint::Solve,
    CancellationPoint::Resample,
    CancellationPoint::Search,
    CancellationPoint::Chain,
    CancellationPoint::Provider,
];

/// Set of [`CancellationPoint`] values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CancellationPoints(u8);

impl CancellationPoints {
    /// No cancellation points.
    pub const EMPTY: Self = Self(0);

    /// Whether `point` was declared.
    #[must_use]
    pub const fn contains(self, point: CancellationPoint) -> bool {
        self.0 & point.mask() != 0
    }

    /// Set with `point` included.
    #[must_use]
    pub const fn with(self, point: CancellationPoint) -> Self {
        Self(self.0 | point.mask())
    }

    /// Declared points in scan, solve, resample, search, chain, provider order.
    #[must_use = "the cancellation points are the declared interrupt boundaries"]
    pub fn iter(self) -> impl Iterator<Item = CancellationPoint> {
        CANCELLATION_POINTS.into_iter().filter(move |point| self.contains(*point))
    }
}

/// Explicit name of a device.
///
/// A blank name is not a name: [`Self::new`] rejects it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DeviceName(Arc<str>);

impl DeviceName {
    /// Name that contains a non-whitespace character.
    #[must_use = "a blank name is not an explicit device"]
    pub fn new(name: &str) -> Option<Self> {
        if is_blank(name) {
            return None;
        }
        Some(Self(Arc::from(name)))
    }

    /// Declared name, unchanged apart from the blank-name rejection.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// CPU, or one explicitly named device.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ComputeCapability {
    /// Host CPU. This is not a device capability.
    Cpu,
    /// Named device. The name is non-empty.
    Device(DeviceName),
}

impl ComputeCapability {
    /// Host CPU.
    #[must_use]
    pub const fn cpu() -> Self {
        Self::Cpu
    }

    /// Named device. A blank `name` is rejected.
    #[must_use = "a blank device name is not an explicit device capability"]
    pub fn device(name: &str) -> Option<Self> {
        DeviceName::new(name).map(Self::Device)
    }

    /// Whether this is [`Self::Device`].
    #[must_use]
    pub const fn is_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }

    /// Device name when this capability is a device.
    #[must_use = "cpu capability has no device name"]
    pub const fn device_name(&self) -> Option<&DeviceName> {
        match self {
            Self::Cpu => None,
            Self::Device(name) => Some(name),
        }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Device(name) => write!(formatter, "device:{name}"),
        }
    }
}

/// Refusal from physical-operator construction, admission, or reuse.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PhysicalOperatorError {
    /// Solve or scan was built without its cancellation point.
    #[error("{effect} requires cancellation point {point}")]
    MissingCancellationPoint {
        /// Solve or scan.
        effect: PhysicalEffect,
        /// Point the effect must publish.
        point: CancellationPoint,
    },
    /// Device work was requested with CPU capability.
    #[error("device work requires a declared device capability")]
    DeviceCapabilityMissing,
    /// Bounded-memory admission saw an absent estimate.
    #[error(
        "bounded-memory admission requires a {kind} byte estimate; an absent estimate is not zero"
    )]
    AbsentByteEstimate {
        /// Scratch, retained, or output.
        kind: ByteEstimateKind,
    },
    /// Source or design dependency ids differ.
    #[error("reuse refused: dependencies differ")]
    ReuseDependencyMismatch,
    /// Strict versus tolerance, or two different declared tolerances.
    #[error("reuse refused: numerical policy differs")]
    ReuseNumericalPolicyMismatch,
}

/// Physical metadata for one operator.
///
/// Constructed with [`Self::declare`]. Byte estimates may stay absent until
/// admission; admission then refuses them. Reuse compares dependencies and
/// numerical policy only.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PhysicalOperator {
    /// Effect. Names mirror `OperatorEffect`.
    effect: PhysicalEffect,
    /// Source and design ids.
    dependencies: PhysicalDependencies,
    /// Scratch, retained, and output estimates.
    estimates: ByteEstimates,
    /// Strict policy, or a tolerance with its declared tolerance.
    numerical_policy: NumericalPolicy,
    /// Points where a run observes cancellation.
    cancellation_points: CancellationPoints,
    /// CPU, or an explicitly named device.
    capability: ComputeCapability,
    /// Whether the operator runs device work.
    device_work: bool,
}

impl PhysicalOperator {
    /// Start metadata for `effect`.
    ///
    /// Estimates start absent, numerical policy starts strict, capability starts
    /// at the CPU, and device work starts off. Dependency ids start empty.
    #[must_use]
    pub const fn declare(effect: PhysicalEffect) -> PhysicalOperatorBuilder {
        PhysicalOperatorBuilder {
            effect,
            sources: Vec::new(),
            designs: Vec::new(),
            estimates: ByteEstimates::absent(),
            numerical_policy: NumericalPolicy::strict(),
            cancellation_points: CancellationPoints::EMPTY,
            capability: ComputeCapability::cpu(),
            device_work: false,
        }
    }

    /// Effect. Names mirror [`crate::operator::OperatorEffect`].
    #[must_use]
    pub const fn effect(&self) -> PhysicalEffect {
        self.effect
    }

    /// Source and design dependency ids.
    #[must_use]
    pub fn dependencies(&self) -> &PhysicalDependencies {
        &self.dependencies
    }

    /// Source ids. These are not source rows.
    #[must_use]
    pub fn sources(&self) -> &[SourceDependencyId] {
        self.dependencies.sources()
    }

    /// Design ids. These are not the design matrix.
    #[must_use]
    pub fn designs(&self) -> &[DesignDependencyId] {
        self.dependencies.designs()
    }

    /// Scratch, retained, and output estimates. Absent fields are not zeros.
    #[must_use]
    pub const fn estimates(&self) -> ByteEstimates {
        self.estimates
    }

    /// Strict policy, or a tolerance policy with its declared tolerance.
    #[must_use]
    pub const fn numerical_policy(&self) -> NumericalPolicy {
        self.numerical_policy
    }

    /// Declared cancellation points, including the point solve or scan requires.
    #[must_use]
    pub const fn cancellation_points(&self) -> CancellationPoints {
        self.cancellation_points
    }

    /// CPU, or an explicitly named device.
    #[must_use]
    pub fn capability(&self) -> &ComputeCapability {
        &self.capability
    }

    /// Whether this operator runs device work.
    #[must_use]
    pub const fn device_work(&self) -> bool {
        self.device_work
    }

    /// Bounded-memory admission of the three byte estimates.
    ///
    /// # Errors
    ///
    /// [`PhysicalOperatorError::AbsentByteEstimate`] when scratch, retained, or
    /// output was not declared. Declared zeros are admitted and are not treated
    /// as free work that skipped an estimate.
    #[must_use = "a refused admission is not permission to allocate"]
    pub const fn bounded_memory_admission(&self) -> Result<AdmittedBytes, PhysicalOperatorError> {
        self.estimates.bounded_memory_admission()
    }

    /// Whether `cached` may be reused for this operator.
    ///
    /// Legal only when the source ids, the design ids, and the numerical policy
    /// are equal. Effect, byte estimates, cancellation points, and capability
    /// are not the reuse key. Dependencies are checked first.
    ///
    /// # Errors
    ///
    /// [`PhysicalOperatorError::ReuseDependencyMismatch`] when a source or
    /// design id differs. [`PhysicalOperatorError::ReuseNumericalPolicyMismatch`]
    /// when the numerical policy differs.
    #[must_use = "refused reuse must not be ignored"]
    pub fn legal_reuse(&self, cached: &Self) -> Result<(), PhysicalOperatorError> {
        if self.dependencies != cached.dependencies {
            return Err(PhysicalOperatorError::ReuseDependencyMismatch);
        }
        if self.numerical_policy != cached.numerical_policy {
            return Err(PhysicalOperatorError::ReuseNumericalPolicyMismatch);
        }
        Ok(())
    }
}

/// Builder returned by [`PhysicalOperator::declare`].
#[derive(Clone, Debug)]
pub struct PhysicalOperatorBuilder {
    /// Effect being declared.
    effect: PhysicalEffect,
    /// Source ids, not yet canonicalized.
    sources: Vec<SourceDependencyId>,
    /// Design ids, not yet canonicalized.
    designs: Vec<DesignDependencyId>,
    /// Scratch, retained, and output estimates.
    estimates: ByteEstimates,
    /// Strict policy, or a tolerance with its declared tolerance.
    numerical_policy: NumericalPolicy,
    /// Declared cancellation points.
    cancellation_points: CancellationPoints,
    /// CPU, or an explicitly named device.
    capability: ComputeCapability,
    /// Whether the operator runs device work.
    device_work: bool,
}

impl PhysicalOperatorBuilder {
    /// Record a source dependency id. This does not attach the source data.
    #[must_use]
    pub fn source(mut self, id: SourceDependencyId) -> Self {
        self.sources.push(id);
        self
    }

    /// Record a design dependency id. This does not attach the design.
    #[must_use]
    pub fn design(mut self, id: DesignDependencyId) -> Self {
        self.designs.push(id);
        self
    }

    /// Replace the byte estimates.
    #[must_use]
    pub const fn estimates(mut self, estimates: ByteEstimates) -> Self {
        self.estimates = estimates;
        self
    }

    /// Replace the numerical policy.
    #[must_use]
    pub const fn numerical_policy(mut self, numerical_policy: NumericalPolicy) -> Self {
        self.numerical_policy = numerical_policy;
        self
    }

    /// Declare a cancellation point.
    #[must_use]
    pub const fn cancellation_point(mut self, point: CancellationPoint) -> Self {
        self.cancellation_points = self.cancellation_points.with(point);
        self
    }

    /// Replace the CPU or device capability.
    #[must_use]
    pub fn capability(mut self, capability: ComputeCapability) -> Self {
        self.capability = capability;
        self
    }

    /// Set whether the operator runs device work.
    ///
    /// Device work on [`ComputeCapability::Cpu`] is refused by [`Self::build`].
    #[must_use]
    pub const fn device_work(mut self, device_work: bool) -> Self {
        self.device_work = device_work;
        self
    }

    /// Finish the metadata.
    ///
    /// # Errors
    ///
    /// [`PhysicalOperatorError::MissingCancellationPoint`] when the effect is
    /// solve or scan and that point was not declared.
    /// [`PhysicalOperatorError::DeviceCapabilityMissing`] when device work is
    /// set and the capability is the CPU.
    #[must_use = "a refused operator is not metadata that can be scheduled"]
    pub fn build(self) -> Result<PhysicalOperator, PhysicalOperatorError> {
        if let Some(point) = self.effect.required_cancellation_point() {
            if !self.cancellation_points.contains(point) {
                return Err(PhysicalOperatorError::MissingCancellationPoint {
                    effect: self.effect,
                    point,
                });
            }
        }
        if self.device_work && !self.capability.is_device() {
            return Err(PhysicalOperatorError::DeviceCapabilityMissing);
        }
        Ok(PhysicalOperator {
            effect: self.effect,
            dependencies: PhysicalDependencies {
                sources: canonical(self.sources),
                designs: canonical(self.designs),
            },
            estimates: self.estimates,
            numerical_policy: self.numerical_policy,
            cancellation_points: self.cancellation_points,
            capability: self.capability,
            device_work: self.device_work,
        })
    }
}

fn canonical<T: Ord>(mut ids: Vec<T>) -> Arc<[T]> {
    ids.sort_unstable();
    ids.dedup();
    Arc::from(ids)
}

#[cfg(test)]
mod tests {
    use super::{
        ByteEstimateKind, ByteEstimates, CancellationPoint, ComputeCapability, DeclaredTolerance,
        DesignDependencyId, DeviceName, NumericalPolicy, PhysicalEffect, PhysicalOperator,
        PhysicalOperatorError, SourceDependencyId,
    };

    fn operator(
        effect: PhysicalEffect,
        sources: &[u64],
        designs: &[u64],
        estimates: ByteEstimates,
        policy: NumericalPolicy,
    ) -> PhysicalOperator {
        let mut builder =
            PhysicalOperator::declare(effect).estimates(estimates).numerical_policy(policy);
        for raw in sources {
            builder = builder.source(SourceDependencyId::from_raw(*raw));
        }
        for raw in designs {
            builder = builder.design(DesignDependencyId::from_raw(*raw));
        }
        if let Some(point) = effect.required_cancellation_point() {
            builder = builder.cancellation_point(point);
        }
        builder.build().expect("physical operator")
    }

    #[test]
    fn physical_effect_names_mirror_operator_effect() {
        let names = [
            (PhysicalEffect::Read, "read"),
            (PhysicalEffect::Select, "select"),
            (PhysicalEffect::Fit, "fit"),
            (PhysicalEffect::Prior, "prior"),
            (PhysicalEffect::Condition, "condition"),
            (PhysicalEffect::Intervene, "intervene"),
            (PhysicalEffect::Transport, "transport"),
            (PhysicalEffect::ReuseFrozenFit, "reuse_frozen_fit"),
            (PhysicalEffect::Solve, "solve"),
            (PhysicalEffect::Scan, "scan"),
        ];
        for (effect, name) in names {
            assert_eq!(effect.as_str(), name);
            assert_eq!(effect.to_string(), name);
        }
    }

    #[test]
    fn changed_dependency_refuses_reuse() {
        let policy = NumericalPolicy::strict();
        let estimates = ByteEstimates::declared(8, 0, 4);
        let cached = operator(PhysicalEffect::Fit, &[1], &[2], estimates, policy);
        let changed_source = operator(PhysicalEffect::Fit, &[9], &[2], estimates, policy);
        let changed_design = operator(PhysicalEffect::Fit, &[1], &[9], estimates, policy);
        let swapped_roles = operator(PhysicalEffect::Fit, &[2], &[1], estimates, policy);
        assert_eq!(
            changed_source.legal_reuse(&cached),
            Err(PhysicalOperatorError::ReuseDependencyMismatch)
        );
        assert_eq!(
            changed_design.legal_reuse(&cached),
            Err(PhysicalOperatorError::ReuseDependencyMismatch)
        );
        assert_eq!(
            swapped_roles.legal_reuse(&cached),
            Err(PhysicalOperatorError::ReuseDependencyMismatch)
        );
    }

    #[test]
    fn numerical_policy_mismatch_refuses_reuse() {
        let estimates = ByteEstimates::declared(8, 0, 4);
        let strict =
            operator(PhysicalEffect::Read, &[1], &[2], estimates, NumericalPolicy::strict());
        let tolerance = DeclaredTolerance::new(1.0, 0.0).expect("finite tolerance");
        let loose = operator(
            PhysicalEffect::Read,
            &[1],
            &[2],
            estimates,
            NumericalPolicy::tolerance(tolerance),
        );
        let other = DeclaredTolerance::new(2.0, 0.0).expect("finite tolerance");
        let looser = operator(
            PhysicalEffect::Read,
            &[1],
            &[2],
            estimates,
            NumericalPolicy::tolerance(other),
        );
        assert_eq!(
            loose.legal_reuse(&strict),
            Err(PhysicalOperatorError::ReuseNumericalPolicyMismatch)
        );
        assert_eq!(
            looser.legal_reuse(&loose),
            Err(PhysicalOperatorError::ReuseNumericalPolicyMismatch)
        );
        assert!(loose.legal_reuse(&loose).is_ok());
    }

    #[test]
    fn matching_dependencies_and_policy_allow_reuse() {
        let policy = NumericalPolicy::strict();
        let cached = operator(
            PhysicalEffect::Fit,
            &[1, 1, 2],
            &[4],
            ByteEstimates::declared(10, 20, 30),
            policy,
        );
        let reordered = operator(
            PhysicalEffect::Read,
            &[2, 1],
            &[4],
            ByteEstimates::declared(99, 0, 1),
            policy,
        );
        assert!(reordered.legal_reuse(&cached).is_ok());
        assert_eq!(reordered.sources(), cached.sources());
        assert_eq!(reordered.designs(), cached.designs());
        assert_ne!(reordered.estimates(), cached.estimates());
        assert_ne!(reordered.effect(), cached.effect());
    }

    #[test]
    fn absent_byte_estimate_is_not_zero() {
        let absent = operator(
            PhysicalEffect::Read,
            &[],
            &[],
            ByteEstimates::absent(),
            NumericalPolicy::strict(),
        );
        assert_eq!(absent.estimates().scratch(), None);
        assert_eq!(
            absent.bounded_memory_admission(),
            Err(PhysicalOperatorError::AbsentByteEstimate { kind: ByteEstimateKind::Scratch })
        );

        let scratch_zero_retained_absent = operator(
            PhysicalEffect::Read,
            &[],
            &[],
            ByteEstimates::absent().with_scratch(0),
            NumericalPolicy::strict(),
        );
        assert_eq!(scratch_zero_retained_absent.estimates().scratch(), Some(0));
        assert_eq!(
            scratch_zero_retained_absent.bounded_memory_admission(),
            Err(PhysicalOperatorError::AbsentByteEstimate { kind: ByteEstimateKind::Retained })
        );

        let output_absent = operator(
            PhysicalEffect::Read,
            &[],
            &[],
            ByteEstimates::absent().with_scratch(0).with_retained(0),
            NumericalPolicy::strict(),
        );
        assert_eq!(
            output_absent.bounded_memory_admission(),
            Err(PhysicalOperatorError::AbsentByteEstimate { kind: ByteEstimateKind::Output })
        );

        let zeros = operator(
            PhysicalEffect::Read,
            &[],
            &[],
            ByteEstimates::declared(0, 0, 0),
            NumericalPolicy::strict(),
        );
        let admitted = zeros.bounded_memory_admission().expect("declared zeros");
        assert_eq!(admitted.scratch(), 0);
        assert_eq!(admitted.retained(), 0);
        assert_eq!(admitted.output(), 0);
        assert_eq!(admitted.total(), 0);

        let mixed = operator(
            PhysicalEffect::Read,
            &[],
            &[],
            ByteEstimates::declared(4, 0, 8),
            NumericalPolicy::strict(),
        );
        let admitted = mixed.bounded_memory_admission().expect("declared estimates");
        assert_eq!(admitted.scratch(), 4);
        assert_eq!(admitted.retained(), 0);
        assert_eq!(admitted.output(), 8);
        assert_eq!(admitted.total(), 12);
    }

    #[test]
    fn solve_and_scan_publish_cancellation_points() {
        assert_eq!(
            PhysicalEffect::Solve.required_cancellation_point(),
            Some(CancellationPoint::Solve)
        );
        assert_eq!(
            PhysicalEffect::Scan.required_cancellation_point(),
            Some(CancellationPoint::Scan)
        );
        assert_eq!(PhysicalEffect::Read.required_cancellation_point(), None);
        assert_eq!(PhysicalEffect::Solve.as_str(), "solve");
        assert_eq!(PhysicalEffect::Scan.as_str(), "scan");

        let solve = operator(
            PhysicalEffect::Solve,
            &[],
            &[],
            ByteEstimates::declared(1, 0, 0),
            NumericalPolicy::strict(),
        );
        let scan = operator(
            PhysicalEffect::Scan,
            &[],
            &[],
            ByteEstimates::declared(1, 0, 0),
            NumericalPolicy::strict(),
        );
        assert!(solve.cancellation_points().contains(CancellationPoint::Solve));
        assert!(scan.cancellation_points().contains(CancellationPoint::Scan));
        assert!(!solve.cancellation_points().contains(CancellationPoint::Scan));
        assert!(!scan.cancellation_points().contains(CancellationPoint::Solve));

        let missing_solve = PhysicalOperator::declare(PhysicalEffect::Solve).build();
        assert_eq!(
            missing_solve,
            Err(PhysicalOperatorError::MissingCancellationPoint {
                effect: PhysicalEffect::Solve,
                point: CancellationPoint::Solve,
            })
        );
        let missing_scan = PhysicalOperator::declare(PhysicalEffect::Scan).build();
        assert_eq!(
            missing_scan,
            Err(PhysicalOperatorError::MissingCancellationPoint {
                effect: PhysicalEffect::Scan,
                point: CancellationPoint::Scan,
            })
        );
    }

    #[test]
    fn device_work_without_declared_device_is_refused() {
        let refused = PhysicalOperator::declare(PhysicalEffect::Fit).device_work(true).build();
        assert_eq!(refused, Err(PhysicalOperatorError::DeviceCapabilityMissing));

        let capability = ComputeCapability::device("gpu0").expect("explicit device");
        let admitted = PhysicalOperator::declare(PhysicalEffect::Fit)
            .capability(capability)
            .device_work(true)
            .build()
            .expect("named device");
        assert!(admitted.device_work());
        assert!(admitted.capability().is_device());
        assert_eq!(admitted.capability().device_name().map(DeviceName::as_str), Some("gpu0"));
    }

    #[test]
    fn blank_device_name_is_not_explicit() {
        assert!(ComputeCapability::device("").is_none());
        assert!(ComputeCapability::device("   ").is_none());
        assert!(DeviceName::new("").is_none());
        let named = ComputeCapability::device("gpu0").expect("explicit device");
        assert_eq!(named.device_name().map(DeviceName::as_str), Some("gpu0"));
    }

    #[test]
    fn strict_policy_carries_no_tolerance() {
        assert!(NumericalPolicy::strict().declared_tolerance().is_none());
        assert!(DeclaredTolerance::new(f64::NAN, 0.0).is_none());
        assert!(DeclaredTolerance::new(f64::INFINITY, 0.0).is_none());
        assert!(DeclaredTolerance::new(-1.0, 0.0).is_none());
        let zero = DeclaredTolerance::new(0.0, 0.0).expect("zero tolerance");
        let negative_zero = DeclaredTolerance::new(-0.0, 0.0).expect("canonical zero");
        assert_eq!(negative_zero, zero);
        let policy = NumericalPolicy::tolerance(zero);
        assert_eq!(policy.declared_tolerance(), Some(zero));
        assert_eq!(zero.absolute().to_bits(), 0);
        assert_eq!(zero.relative().to_bits(), 0);
    }
}

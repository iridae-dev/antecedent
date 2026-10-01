//! Joint mechanism deviations for the registered surrogate z-transport formula
//! (2.2B work package B3, workstream X3).
//!
//! The formula `psi = sum_w P_W(w) [m(w, 1) - m(w, 0)]`, with
//! `m(w, x) = sum_y y P_Y(y | w, x, do(z))`, cites two source factors: the
//! outcome kernel and the shared parent marginal. Each declared factor is
//! box-independently contaminated, `Q_i = (1 - eps_i) P_i + eps_i R_i` with
//! `R_i` free in its simplex and `eps_i <= e_i`. The derivation in
//! `docs/guides/joint-mechanism-sensitivity.md` shows:
//!
//! * the box's extremal range is attained at `eps = e` (containment), so no
//!   continuous fraction optimization exists;
//! * `psi` is linear in each replacement given the others, so the extrema sit
//!   at product-polytope vertices and have the closed form
//!   `U = (1 - e_W) sum_w P_W(w) D+(w) + e_W max_w D+(w)` with
//!   `D+(w) = (1 - e_Y) Delta(w) + e_Y (y_max - y_min)`, `L` symmetric;
//! * nested boxes give nested ranges, and `U` (`L`) is monotone in each
//!   fraction, so a tipping frontier is bracketed by bisection.
//!
//! The kernel stage reuses the 2.1 [`DiscreteKernelSensitivity`] evaluator,
//! so a single kernel factor is bit-equal to [`crate::z_transport_mechanism_sensitivity`].
//! The law's shape is checked against the declared bounds before its cells
//! are read; the read itself is charged one operation per parent level to one
//! [`antecedent_core::SearchBudget`], which the closed-form range and then the
//! frontier's certified brackets (the only iterative work) share. The range is
//! an assumption range, never a confidence interval. Sampling uncertainty is
//! not offered by this record in 2.2: its conservative composition below is
//! compiled only under `calibration-internal` and its route is closed (record
//! `2.2B.X3.joint_sensitivity_uncertainty`, carried forward).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::fmt;

use antecedent_core::{
    ExecutionContext, RegimeId, SearchBudget, SearchLimits, SearchReceipt, SearchStop, reason_code,
};
use antecedent_expr::ExactTransportData;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::BoundZTransportFunctional;

use crate::mechanism_sensitivity::{
    DiscreteKernelSensitivity, DiscreteKernelSensitivityError, ScanStop, ZSurrogateKernels,
    ZTransportSensitivityError, z_surrogate_kernels_metered,
};

/// Hard cap on declared factors: the two factors of the surrogate formula (a
/// third declaration is always a duplicate or an out-of-scope factor, so it
/// refuses by count).
pub const JOINT_SENSITIVITY_MAX_FACTORS: usize = 2;
/// Hard cap on shared-parent levels.
pub const JOINT_SENSITIVITY_MAX_PARENT_LEVELS: usize = 64;
/// Hard cap on outcome categories.
pub const JOINT_SENSITIVITY_MAX_OUTCOME_CATEGORIES: usize = 32;
/// Hard cap on frontier grid points.
pub const JOINT_SENSITIVITY_MAX_FRONTIER_POINTS: usize = 33;
/// Hard cap on the operation limit of the shared budget.
pub const JOINT_SENSITIVITY_MAX_OPERATIONS: usize = 100_000;
/// Hard cap on the depth limit (bisection iterations per bracket).
pub const JOINT_SENSITIVITY_MAX_DEPTH: usize = 64;
/// Default memory cap of the shared budget.
pub const JOINT_SENSITIVITY_DEFAULT_MEMORY_BYTES: u64 = 64 * 1024 * 1024;
/// Hard ceiling on a declared memory cap (the shared search default).
pub const JOINT_SENSITIVITY_MAX_MEMORY_BYTES: u64 = antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES;
/// Hard cap on replicates of the (closed) endpoint bootstrap.
pub const JOINT_SENSITIVITY_MAX_BOOTSTRAP: u32 = 2000;
/// Smallest admitted bracketing tolerance.
pub const JOINT_SENSITIVITY_MIN_TOLERANCE: f64 = 1e-12;
/// Largest admitted bracketing tolerance.
pub const JOINT_SENSITIVITY_MAX_TOLERANCE: f64 = 1e-2;
/// Largest admitted `|sum - 1|` of the shared parent marginal (the 2.1
/// evaluator's kernel-row tolerance).
const JOINT_SENSITIVITY_UNIT_MASS_TOLERANCE: f64 = 1e-10;
/// Optimization method recorded in every receipt.
pub const JOINT_SENSITIVITY_METHOD: &str = "box-independent epsilon contamination of the outcome kernel and the shared parent marginal; closed-form product-polytope vertex extrema (2.1 kernel stage, then the parent-simplex vertex); certified bisection brackets for the tipping frontier";
/// Interpretation of the reported range.
pub const JOINT_SENSITIVITY_INTERPRETATION: &str = "assumption range over the declared joint contamination box; not a confidence interval and not a sampling interval";
/// The only inference claim the range carries.
pub const ASSUMPTION_RANGE_CLAIM: &str = "assumption_range";
/// The one sampling composition declared for this cell (closed route).
pub const JOINT_SENSITIVITY_SAMPLING_METHOD: &str = "conservative_endpoint_percentile_bootstrap";
/// The coverage target that composition is measured against.
pub const JOINT_SENSITIVITY_COVERAGE_TARGET: &str = "one_sided: coverage of psi(delta0) at least nominal for any fixed true deviation delta0 inside the box; exactly the percentile bootstrap at the zero box";
/// Detail of a budget stop, before the range (a refusal) or during bracketing
/// (the frontier's unresolved reason).
pub const JOINT_SENSITIVITY_BUDGET_DETAIL: &str = "joint_sensitivity.budget";
/// Detail of the withheld sampling interval.
pub const JOINT_SENSITIVITY_INTERVAL_WITHHELD: &str = "joint_sensitivity.interval_withheld";

/// A mechanism factor a caller may ask to perturb.
///
/// Only [`Self::OutcomeKernel`] and [`Self::SharedParentMarginal`] are factors
/// of the registered surrogate formula; the others are named so that their
/// requests refuse with a typed detail instead of being silently ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JointFactor {
    /// `P(Y | w, x, do(z))`, contaminated per `(w, x)` stratum.
    OutcomeKernel,
    /// `P(w | do(z))`, the shared parent marginal.
    SharedParentMarginal,
    /// The treatment mechanism; the treatment is set by intervention.
    TreatmentMechanism,
    /// A parent mechanism on the fixed-graph route (out of scope).
    FixedGraphParentMechanism,
    /// The fixed-graph conditional-mechanism joint problem (out of scope).
    FixedGraphConditionalMechanism,
    /// Source-target discrepancy diagnostics (out of scope, 2.3B).
    SourceTargetDiscrepancy,
}

impl JointFactor {
    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OutcomeKernel => "outcome_kernel",
            Self::SharedParentMarginal => "shared_parent_marginal",
            Self::TreatmentMechanism => "treatment_mechanism",
            Self::FixedGraphParentMechanism => "fixed_graph_parent",
            Self::FixedGraphConditionalMechanism => "fixed_graph_conditional",
            Self::SourceTargetDiscrepancy => "source_target_discrepancy",
        }
    }

    /// Parse a wire name; `None` for a name outside the vocabulary.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::OutcomeKernel,
            Self::SharedParentMarginal,
            Self::TreatmentMechanism,
            Self::FixedGraphParentMechanism,
            Self::FixedGraphConditionalMechanism,
            Self::SourceTargetDiscrepancy,
        ]
        .into_iter()
        .find(|factor| factor.name() == name)
    }
}

/// One declared factor and its contamination bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointFactorBound {
    /// The perturbed factor.
    pub factor: JointFactor,
    /// Largest contamination fraction, in `[0, 1]`.
    pub max_fraction: f64,
}

/// Limits of the one shared budget: every range and bracketing step is charged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JointSensitivityLimits {
    /// Operation limit (hard cap [`JOINT_SENSITIVITY_MAX_OPERATIONS`]).
    pub operations: usize,
    /// Depth limit: bisection iterations per bracket (hard cap [`JOINT_SENSITIVITY_MAX_DEPTH`]).
    pub depth: usize,
    /// Memory cap (hard ceiling [`JOINT_SENSITIVITY_MAX_MEMORY_BYTES`]); the
    /// effective cap is also bounded by the context's hard limit.
    pub memory_bytes: u64,
}

impl Default for JointSensitivityLimits {
    fn default() -> Self {
        Self {
            operations: JOINT_SENSITIVITY_MAX_OPERATIONS,
            depth: JOINT_SENSITIVITY_MAX_DEPTH,
            memory_bytes: JOINT_SENSITIVITY_DEFAULT_MEMORY_BYTES,
        }
    }
}

/// The declared joint deviation set and its analysis settings.
#[derive(Clone, Debug, PartialEq)]
pub struct JointDeviationSpec {
    /// Perturbed factors with their bounds (any order; reported canonically).
    pub factors: Vec<JointFactorBound>,
    /// A total budget coupling the fractions (`sum eps <= B`); refused.
    pub total_budget: Option<f64>,
    /// Optional decision threshold for the tipping frontier.
    pub decision_threshold: Option<f64>,
    /// Width at which a bisection bracket is accepted.
    pub tolerance: f64,
    /// Parent-fraction grid size of the frontier (1 to 33).
    pub frontier_points: usize,
    /// Limits of the one shared budget.
    pub limits: JointSensitivityLimits,
}

impl JointDeviationSpec {
    /// A spec over `factors` with no threshold, tolerance `1e-9`, a 17-point
    /// frontier grid and the default limits.
    #[must_use]
    pub fn new(factors: Vec<JointFactorBound>) -> Self {
        Self {
            factors,
            total_budget: None,
            decision_threshold: None,
            tolerance: 1e-9,
            frontier_points: 17,
            limits: JointSensitivityLimits::default(),
        }
    }

    /// The same spec with a decision threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.decision_threshold = Some(threshold);
        self
    }

    /// Declared bound of `factor`, or zero when it is not declared.
    #[must_use]
    pub fn fraction(&self, factor: JointFactor) -> f64 {
        self.factors.iter().find(|bound| bound.factor == factor).map_or(0.0, |b| b.max_fraction)
    }
}

/// Typed refusal of the joint sensitivity contract.
#[derive(Clone, Debug, PartialEq)]
pub enum JointSensitivityError {
    /// A reason-coded refusal with its `joint_sensitivity.*` detail.
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// Namespaced detail.
        detail: &'static str,
        /// Human-readable context.
        message: String,
    },
    /// The shared budget stopped before the range was computed.
    Budget(SearchReceipt),
}

impl JointSensitivityError {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Refused { code, .. } => code,
            Self::Budget(_) => reason_code!("transport_budget_cancel"),
        }
    }

    /// Namespaced `joint_sensitivity.*` detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        match self {
            Self::Refused { detail, .. } => detail,
            Self::Budget(_) => JOINT_SENSITIVITY_BUDGET_DETAIL,
        }
    }
}

impl fmt::Display for JointSensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
            Self::Budget(receipt) => write!(
                f,
                "{JOINT_SENSITIVITY_BUDGET_DETAIL}: the shared budget stopped before the range ({})",
                receipt.stop.code()
            ),
        }
    }
}

impl std::error::Error for JointSensitivityError {}

fn refuse(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
) -> JointSensitivityError {
    JointSensitivityError::Refused { code, detail, message: message.into() }
}

/// Exact extrema over the declared box and their vertex witnesses.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSensitivityRange {
    /// Minimum response over the box.
    pub minimum: f64,
    /// Maximum response over the box.
    pub maximum: f64,
    /// Outcome vertex per `(w, x)` stratum at the minimum (2.1 kernel witness).
    pub minimizing_outcome_by_stratum: Vec<usize>,
    /// Outcome vertex per `(w, x)` stratum at the maximum.
    pub maximizing_outcome_by_stratum: Vec<usize>,
    /// Parent-simplex vertex at the minimum; `None` when the parent is not perturbed.
    pub minimizing_parent_level: Option<usize>,
    /// Parent-simplex vertex at the maximum.
    pub maximizing_parent_level: Option<usize>,
}

/// Where a tipping search along one line ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TippingStatus {
    /// The threshold is reached with no deviation along the searched coordinate.
    ReachedAtOrigin,
    /// The first crossing lies inside the certified bracket.
    Bracketed,
    /// The threshold is not reached inside the box.
    NotReachedInBox,
    /// The shared budget stopped before this line was resolved.
    Unevaluated,
}

impl TippingStatus {
    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ReachedAtOrigin => "reached_at_origin",
            Self::Bracketed => "bracketed",
            Self::NotReachedInBox => "not_reached_in_box",
            Self::Unevaluated => "unevaluated",
        }
    }
}

/// Certified bracket `[lower, upper]` of the smallest crossing fraction: the
/// response does not reach the threshold at `lower` (unless `lower` is zero and
/// the status is `ReachedAtOrigin`) and does at `upper`. The bracket is the
/// unresolved region. It is certified against the floating-point evaluation of
/// the closed-form range, not in exact arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TippingBracket {
    /// Largest evaluated fraction that does not reach the threshold.
    pub lower: f64,
    /// Smallest evaluated fraction that reaches the threshold.
    pub upper: f64,
    /// Bisection iterations spent.
    pub iterations: usize,
}

/// One frontier line: the parent fraction is held and the kernel fraction searched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierPoint {
    /// Parent-marginal fraction of this line.
    pub parent_fraction: f64,
    /// How the search ended.
    pub status: TippingStatus,
    /// Kernel-fraction bracket when resolved.
    pub bracket: Option<TippingBracket>,
}

/// Tipping point along one axis (the other factor unperturbed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisTipping {
    /// The factor whose fraction is searched.
    pub factor: JointFactor,
    /// The 2.1 one-factor analytic tipping fraction.
    pub analytic: Option<f64>,
    /// How the bracket search ended.
    pub status: TippingStatus,
    /// Certified bracket when resolved.
    pub bracket: Option<TippingBracket>,
}

/// Declared sampling composition and why its interval is withheld.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointSensitivityUncertainty {
    /// The one declared method.
    pub method: &'static str,
    /// Its coverage target.
    pub coverage_target: &'static str,
    /// Registered reason code of the withheld interval.
    pub reason_code: &'static str,
    /// `joint_sensitivity.interval_withheld`.
    pub detail: &'static str,
}

impl JointSensitivityUncertainty {
    /// The withheld status every public result carries.
    #[must_use]
    pub const fn withheld() -> Self {
        Self {
            method: JOINT_SENSITIVITY_SAMPLING_METHOD,
            coverage_target: JOINT_SENSITIVITY_COVERAGE_TARGET,
            reason_code: reason_code!("cell_not_licensed"),
            detail: JOINT_SENSITIVITY_INTERVAL_WITHHELD,
        }
    }
}

/// Optimization and bracketing receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSensitivityReceipt {
    /// Optimization method.
    pub method: &'static str,
    /// Declared box: `[kernel fraction, parent fraction]`.
    pub fraction_box: [f64; 2],
    /// Bracketing tolerance.
    pub tolerance: f64,
    /// Limits in force.
    pub limits: JointSensitivityLimits,
    /// Effective memory cap (never absent).
    pub memory_limit_bytes: u64,
    /// Operations charged.
    pub operations_consumed: usize,
    /// Deepest bisection level charged.
    pub depth_reached: usize,
    /// Largest cumulative live-state estimate charged, in bytes.
    pub live_state_bytes: u64,
    /// Why bracketing stopped, when the budget stopped it.
    pub stop: Option<SearchStop>,
    /// Stages fully evaluated, in order.
    pub explored: Vec<String>,
    /// Stages left unevaluated by a stop.
    pub unevaluated: Vec<String>,
}

/// Auditable joint mechanism sensitivity of the registered surrogate formula.
#[derive(Clone, Debug, PartialEq)]
pub struct JointMechanismSensitivityResult {
    /// Stable identity of the checked derivation's inputs.
    pub query_binding: String,
    /// Provider snapshot used by the formula.
    pub provider_snapshot: String,
    /// Source law regime used by the formula.
    pub source_regime: RegimeId,
    /// Declared factors in canonical order.
    pub factors: Vec<JointFactorBound>,
    /// Decision threshold.
    pub decision_threshold: Option<f64>,
    /// Response under the unmodified source factors (the 2.1 baseline).
    pub baseline: f64,
    /// Exact range over the box.
    pub range: JointSensitivityRange,
    /// Axis tipping points, one per declared factor (empty without a threshold).
    pub axis_tipping: Vec<AxisTipping>,
    /// Frontier lines over the parent-fraction grid (empty without a threshold
    /// or without the kernel factor).
    pub frontier: Vec<FrontierPoint>,
    /// `Some(joint_sensitivity.budget)` when the budget left lines unresolved.
    pub unresolved_detail: Option<&'static str>,
    /// Receipt.
    pub receipt: JointSensitivityReceipt,
    /// Interpretation statement.
    pub interpretation: &'static str,
    /// Always [`ASSUMPTION_RANGE_CLAIM`].
    pub inference_claim: &'static str,
    /// Sampling uncertainty status (withheld).
    pub uncertainty: JointSensitivityUncertainty,
}

/// Validated box.
#[derive(Clone, Copy, Debug)]
struct JointBox {
    kernel: Option<f64>,
    parent: Option<f64>,
}

impl JointBox {
    fn kernel(self) -> f64 {
        self.kernel.unwrap_or(0.0)
    }

    fn parent(self) -> f64 {
        self.parent.unwrap_or(0.0)
    }
}

/// Validate the declared deviation set, in the order a caller would fix it.
#[allow(
    clippy::too_many_lines,
    reason = "one flat list of ordered refusals, one per declared detail"
)]
fn validate_spec(spec: &JointDeviationSpec) -> Result<JointBox, JointSensitivityError> {
    if spec.total_budget.is_some() {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "joint_sensitivity.budget_coupling",
            "a total budget coupling the fractions needs a polynomial optimizer; only box-independent fractions are supported",
        ));
    }
    if spec.factors.is_empty() || spec.factors.len() > JOINT_SENSITIVITY_MAX_FACTORS {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "joint_sensitivity.factor_count",
            format!("{} factors declared; one or two are admitted", spec.factors.len()),
        ));
    }
    for bound in &spec.factors {
        match bound.factor {
            JointFactor::OutcomeKernel | JointFactor::SharedParentMarginal => {}
            JointFactor::TreatmentMechanism => {
                return Err(refuse(
                    reason_code!("invalid_argument"),
                    "joint_sensitivity.treatment_factor",
                    "the treatment is set by intervention; its mechanism is not in the target response",
                ));
            }
            JointFactor::FixedGraphParentMechanism => {
                return Err(refuse(
                    reason_code!("route_not_supported"),
                    "joint_sensitivity.fixed_graph_parent",
                    "a fixed-graph parent-mechanism perturbation contradicts the supplied target parent law",
                ));
            }
            JointFactor::FixedGraphConditionalMechanism => {
                return Err(refuse(
                    reason_code!("route_not_supported"),
                    "joint_sensitivity.fixed_graph_conditional",
                    "the fixed-graph conditional-mechanism joint problem is not supported",
                ));
            }
            JointFactor::SourceTargetDiscrepancy => {
                return Err(refuse(
                    reason_code!("route_not_supported"),
                    "joint_sensitivity.source_target_discrepancy",
                    "source-target discrepancy diagnostics are not part of this route",
                ));
            }
        }
    }
    let mut kernel = None;
    let mut parent = None;
    for bound in &spec.factors {
        let slot =
            if bound.factor == JointFactor::OutcomeKernel { &mut kernel } else { &mut parent };
        if slot.is_some() {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "joint_sensitivity.duplicate_factor",
                format!("factor {} is declared twice", bound.factor.name()),
            ));
        }
        if !bound.max_fraction.is_finite() || !(0.0..=1.0).contains(&bound.max_fraction) {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "joint_sensitivity.invalid_fraction",
                format!(
                    "fraction bound {} of {} is outside [0, 1]",
                    bound.max_fraction,
                    bound.factor.name()
                ),
            ));
        }
        *slot = Some(bound.max_fraction);
    }
    if spec.decision_threshold.is_some_and(|t| !t.is_finite()) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "joint_sensitivity.invalid_threshold",
            "the decision threshold must be finite",
        ));
    }
    if !spec.tolerance.is_finite()
        || !(JOINT_SENSITIVITY_MIN_TOLERANCE..=JOINT_SENSITIVITY_MAX_TOLERANCE)
            .contains(&spec.tolerance)
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "joint_sensitivity.invalid_tolerance",
            format!("tolerance {} is outside [1e-12, 1e-2]", spec.tolerance),
        ));
    }
    if spec.frontier_points == 0
        || spec.frontier_points > JOINT_SENSITIVITY_MAX_FRONTIER_POINTS
        || spec.limits.operations > JOINT_SENSITIVITY_MAX_OPERATIONS
        || spec.limits.depth > JOINT_SENSITIVITY_MAX_DEPTH
        || spec.limits.memory_bytes > JOINT_SENSITIVITY_MAX_MEMORY_BYTES
    {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "joint_sensitivity.bounds_exceeded",
            format!(
                "frontier grid {} (1-33), operations {} (<= 100000), depth {} (<= 64) or memory cap {} (<= 512 MiB) is outside the declared bounds",
                spec.frontier_points,
                spec.limits.operations,
                spec.limits.depth,
                spec.limits.memory_bytes
            ),
        ));
    }
    Ok(JointBox { kernel, parent })
}

/// Refuse a law larger than the declared bounds; runs on the law's shape,
/// before any of its cells is read.
fn check_shape(levels: usize, categories: usize) -> Result<(), JointSensitivityError> {
    if levels > JOINT_SENSITIVITY_MAX_PARENT_LEVELS
        || categories > JOINT_SENSITIVITY_MAX_OUTCOME_CATEGORIES
    {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "joint_sensitivity.bounds_exceeded",
            format!("{levels} parent levels (<= 64) or {categories} outcome categories (<= 32)"),
        ));
    }
    Ok(())
}

/// Live-state bytes of the factors the evaluator holds for a law of this
/// shape: two kernel rows per level, their weights, three per-level
/// quantities (`Delta`, the parent marginal and its normalization) and the
/// outcome values. Known from the shape, so the scan is charged with it.
fn factor_bytes(levels: usize, categories: usize) -> u64 {
    let cells = 2 * levels * categories + 2 * levels + 3 * levels + categories;
    (cells * std::mem::size_of::<f64>()) as u64
}

/// Read the surrogate factors with the bounds checked on the law's shape and
/// the read charged one operation per parent level (at the factors' byte
/// estimate) to `budget`, when one is given.
fn read_factors(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    ctx: &ExecutionContext,
    mut budget: Option<&mut SearchBudget<'_>>,
) -> Result<JointFactors, JointSensitivityError> {
    let bytes = std::cell::Cell::new(0);
    let read = z_surrogate_kernels_metered(
        diagram,
        functional,
        data,
        ctx,
        &mut |levels, categories| {
            check_shape(levels, categories)?;
            bytes.set(factor_bytes(levels, categories));
            Ok(())
        },
        &mut |_| match budget.as_deref_mut() {
            Some(budget) => budget.charge(0, bytes.get()).map_err(|stop| {
                JointSensitivityError::Budget(budget.receipt(
                    stop,
                    Vec::new(),
                    vec!["range".into()],
                ))
            }),
            None => Ok(()),
        },
    );
    match read {
        Ok(kernels) => JointFactors::new(kernels),
        Err(ScanStop::Hook(error)) => Err(error),
        Err(ScanStop::Kernel(ZTransportSensitivityError::Cancelled)) => match budget {
            Some(budget) => Err(JointSensitivityError::Budget(budget.receipt(
                SearchStop::Cancelled,
                Vec::new(),
                vec!["range".into()],
            ))),
            None => Err(kernel_error(&ZTransportSensitivityError::Cancelled)),
        },
        Err(ScanStop::Kernel(error)) => Err(kernel_error(&error)),
    }
}

fn kernel_error(error: &ZTransportSensitivityError) -> JointSensitivityError {
    match error {
        ZTransportSensitivityError::InvalidProof
        | ZTransportSensitivityError::IncompatibleFormula => refuse(
            reason_code!("transport_not_certified"),
            "joint_sensitivity.formula_incompatible",
            error.to_string(),
        ),
        ZTransportSensitivityError::ProviderMismatch => refuse(
            reason_code!("transport_missing_provider"),
            "joint_sensitivity.provider_mismatch",
            error.to_string(),
        ),
        ZTransportSensitivityError::UnsupportedDomain
        | ZTransportSensitivityError::InvalidSensitivity(
            DiscreteKernelSensitivityError::InvalidContrast,
        ) => refuse(
            reason_code!("route_not_supported"),
            "joint_sensitivity.unsupported_domain",
            error.to_string(),
        ),
        ZTransportSensitivityError::InvalidSensitivity(
            DiscreteKernelSensitivityError::InvalidFraction,
        ) => refuse(
            reason_code!("invalid_argument"),
            "joint_sensitivity.invalid_fraction",
            error.to_string(),
        ),
        ZTransportSensitivityError::IncompleteKernel
        | ZTransportSensitivityError::Cancelled
        | ZTransportSensitivityError::InvalidSensitivity(_) => refuse(
            reason_code!("transport_support_failure"),
            "joint_sensitivity.incomplete_kernel",
            error.to_string(),
        ),
    }
}

/// The surrogate formula's factors in the form the joint evaluator reads.
#[derive(Clone, Debug)]
pub(crate) struct JointFactors {
    kernels: ZSurrogateKernels,
    /// `Delta(w) = m(w, 1) - m(w, 0)` under the source kernel.
    deltas: Vec<f64>,
    /// Shared parent marginal as read (unit mass checked, not renormalized).
    parent: Vec<f64>,
    /// `y_max - y_min` over the numeric values of the law's `Y` axis (the
    /// listed outcome levels, not a declared outcome domain).
    spread: f64,
}

impl JointFactors {
    fn new(kernels: ZSurrogateKernels) -> Result<Self, JointSensitivityError> {
        let levels = kernels.parent_marginal.len();
        if kernels.outcome_values.iter().any(|y| !y.is_finite()) {
            return Err(refuse(
                reason_code!("route_not_supported"),
                "joint_sensitivity.unsupported_domain",
                "outcome values must be finite numbers",
            ));
        }
        let mean =
            |row: &[f64]| row.iter().zip(&kernels.outcome_values).map(|(p, y)| p * y).sum::<f64>();
        let deltas = (0..levels)
            .map(|w| mean(&kernels.kernels[2 * w + 1]) - mean(&kernels.kernels[2 * w]))
            .collect();
        // Factor normalization is enforced, not repaired. Kernel rows are the
        // law's conditionals and are re-checked by the 2.1 evaluator; the
        // shared parent marginal is the law's own mass, which a caller's loose
        // `LawTolerance` can leave far from one. The contamination class is
        // defined only on distributions, so the marginal must have unit mass
        // within the 2.1 evaluator's row tolerance (the same sum it checks). It
        // is then used as read, never renormalized: the kernel stage weights it
        // as read, so the range, the frontier and both axis values share one
        // scale.
        let total = kernels.parent_marginal.iter().sum::<f64>();
        if !(total.is_finite() && (total - 1.0).abs() <= JOINT_SENSITIVITY_UNIT_MASS_TOLERANCE) {
            return Err(refuse(
                reason_code!("transport_support_failure"),
                "joint_sensitivity.incomplete_kernel",
                format!(
                    "the shared parent marginal has mass {total}, not 1 within {JOINT_SENSITIVITY_UNIT_MASS_TOLERANCE:e}"
                ),
            ));
        }
        let parent = kernels.parent_marginal.clone();
        let high = kernels.outcome_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low = kernels.outcome_values.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(Self { kernels, deltas, parent, spread: high - low })
    }

    /// Live-state bytes of the factors held by the evaluator.
    fn bytes(&self) -> u64 {
        factor_bytes(self.deltas.len(), self.kernels.outcome_values.len())
    }

    /// The 2.1 kernel stage at `fraction`.
    fn kernel_stage(
        &self,
        fraction: f64,
        threshold: Option<f64>,
    ) -> Result<crate::mechanism_sensitivity::DiscreteKernelSensitivityResult, JointSensitivityError>
    {
        DiscreteKernelSensitivity {
            source_kernel: self.kernels.kernels.clone(),
            outcome_values: self.kernels.outcome_values.clone(),
            stratum_contrast_weights: self.kernels.weights.clone(),
            max_fraction: fraction,
            decision_threshold: threshold,
        }
        .evaluate()
        .map_err(|error| kernel_error(&ZTransportSensitivityError::InvalidSensitivity(error)))
    }

    /// The 2.1 one-factor root-mechanism stage for the parent marginal alone.
    fn parent_stage(
        &self,
        fraction: f64,
        threshold: Option<f64>,
    ) -> Result<crate::mechanism_sensitivity::DiscreteKernelSensitivityResult, JointSensitivityError>
    {
        DiscreteKernelSensitivity {
            source_kernel: vec![self.parent.clone()],
            outcome_values: self.deltas.clone(),
            stratum_contrast_weights: vec![1.0],
            max_fraction: fraction,
            decision_threshold: threshold,
        }
        .evaluate()
        .map_err(|error| kernel_error(&ZTransportSensitivityError::InvalidSensitivity(error)))
    }

    /// Exact `(L, U)` at the box corner `(kernel, parent)` with its witnesses.
    ///
    /// `U = A + e_W max(max_w D+(w) - A, 0)` where `A` is the 2.1 kernel-stage
    /// maximum (`sum_w P_W(w) D+(w)`), which equals the closed form
    /// `(1 - e_W) A + e_W max_w D+(w)`; `max_w D+ >= A` holds exactly, so the
    /// clamp only removes rounding. At `e_W = 0` the kernel stage is returned
    /// unchanged, which makes a single kernel factor bit-equal to 2.1.
    fn bounds_at(
        &self,
        kernel: f64,
        parent: f64,
    ) -> Result<JointSensitivityRange, JointSensitivityError> {
        let stage = self.kernel_stage(kernel, None)?;
        let mut range = JointSensitivityRange {
            minimum: stage.minimum,
            maximum: stage.maximum,
            minimizing_outcome_by_stratum: stage.receipt.minimizing_outcome_by_stratum,
            maximizing_outcome_by_stratum: stage.receipt.maximizing_outcome_by_stratum,
            minimizing_parent_level: None,
            maximizing_parent_level: None,
        };
        if parent <= 0.0 {
            return Ok(range);
        }
        let (mut high_level, mut high) = (0, f64::NEG_INFINITY);
        let (mut low_level, mut low) = (0, f64::INFINITY);
        for (level, delta) in self.deltas.iter().enumerate() {
            let up = delta + kernel * (self.spread - delta);
            let down = delta - kernel * (self.spread + delta);
            if up > high {
                (high_level, high) = (level, up);
            }
            if down < low {
                (low_level, low) = (level, down);
            }
        }
        range.maximum = stage.maximum + parent * (high - stage.maximum).max(0.0);
        range.minimum = stage.minimum + parent * (low - stage.minimum).min(0.0);
        range.maximizing_parent_level = Some(high_level);
        range.minimizing_parent_level = Some(low_level);
        Ok(range)
    }
}

/// Which coordinate a bracket searches.
#[derive(Clone, Copy)]
enum Line {
    /// Kernel fraction searched at a fixed parent fraction.
    Kernel { parent: f64 },
    /// Parent fraction searched with the kernel unperturbed.
    Parent,
}

struct Bracketing<'b, 'c> {
    factors: &'b JointFactors,
    budget: &'b mut SearchBudget<'c>,
    threshold: f64,
    upward: bool,
    tolerance: f64,
    bytes: u64,
}

impl Bracketing<'_, '_> {
    fn reached(&self, line: Line, t: f64) -> Result<bool, JointSensitivityError> {
        let range = match line {
            Line::Kernel { parent } => self.factors.bounds_at(t, parent)?,
            Line::Parent => self.factors.bounds_at(0.0, t)?,
        };
        Ok(if self.upward {
            range.maximum >= self.threshold
        } else {
            range.minimum <= self.threshold
        })
    }

    /// Certified bisection of the first crossing on `[0, extent]`, charging
    /// every iteration (depth = iteration) with the cumulative live state.
    fn bracket(
        &mut self,
        line: Line,
        extent: f64,
    ) -> Result<Result<(TippingStatus, Option<TippingBracket>), SearchStop>, JointSensitivityError>
    {
        if let Err(stop) = self.budget.charge(0, self.bytes) {
            return Ok(Err(stop));
        }
        if self.reached(line, 0.0)? {
            return Ok(Ok((
                TippingStatus::ReachedAtOrigin,
                Some(TippingBracket { lower: 0.0, upper: 0.0, iterations: 0 }),
            )));
        }
        if extent <= 0.0 || !self.reached(line, extent)? {
            return Ok(Ok((TippingStatus::NotReachedInBox, None)));
        }
        let (mut lower, mut upper) = (0.0_f64, extent);
        let mut iterations = 0;
        while upper - lower > self.tolerance {
            iterations += 1;
            if let Err(stop) = self.budget.charge(iterations, self.bytes) {
                return Ok(Err(stop));
            }
            let middle = lower + (upper - lower) / 2.0;
            if middle <= lower || middle >= upper {
                break;
            }
            if self.reached(line, middle)? {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        Ok(Ok((TippingStatus::Bracketed, Some(TippingBracket { lower, upper, iterations }))))
    }
}

/// Byte estimate of one resolved frontier or axis entry.
const LINE_BYTES: u64 = std::mem::size_of::<FrontierPoint>() as u64;

/// Evaluate the registered surrogate z formula under a box-independent joint
/// deviation of its outcome kernel and shared parent marginal.
///
/// The declared spec is validated and the law's shape checked against the
/// bounds before any cell of the law is read; the read (three passes over the
/// law per parent level) is charged one operation per parent level. The exact
/// range is then computed in closed form (uncharged, O(levels x categories));
/// with a threshold, the axis tipping points and the frontier over the
/// parent-fraction grid are bracketed under the same budget. A budget stop
/// during the read refuses with its receipt; a stop during bracketing keeps
/// the exact range and reports the remaining lines unevaluated with the
/// receipt.
///
/// # Errors
///
/// A typed [`JointSensitivityError`] carrying its reason code and
/// `joint_sensitivity.*` detail.
///
/// # Panics
///
/// Never: one bracketing outcome is recorded per stage before stages are read back.
#[allow(clippy::too_many_lines)]
pub fn z_transport_joint_mechanism_sensitivity(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    spec: &JointDeviationSpec,
    ctx: &ExecutionContext,
) -> Result<JointMechanismSensitivityResult, JointSensitivityError> {
    let joint_box = validate_spec(spec)?;
    let mut budget = SearchBudget::with_memory(
        SearchLimits { operations: spec.limits.operations, depth: spec.limits.depth },
        spec.limits.memory_bytes,
        ctx,
    )
    .map_err(JointSensitivityError::Budget)?;
    let factors = read_factors(diagram, functional, data, ctx, Some(&mut budget))?;
    let (kernel, parent) = (joint_box.kernel(), joint_box.parent());
    let bytes = factors.bytes();
    let baseline = factors.kernel_stage(0.0, None)?.baseline;
    let range = factors.bounds_at(kernel, parent)?;
    let mut explored = vec!["range".to_owned()];
    let mut unevaluated = Vec::new();
    let mut axis_tipping = Vec::new();
    let mut frontier = Vec::new();
    let mut stop = None;
    let mut peak_bytes = bytes;
    if let Some(threshold) = spec.decision_threshold {
        let upward = threshold >= baseline;
        let grid = if joint_box.parent.is_some() && spec.frontier_points > 1 {
            (0..spec.frontier_points)
                .map(|j| parent * j as f64 / (spec.frontier_points - 1) as f64)
                .collect::<Vec<_>>()
        } else {
            vec![0.0]
        };
        // Stages in charge order: the kernel axis (frontier line 0), the
        // remaining frontier lines, then the parent axis.
        let mut stages: Vec<(String, Line, f64)> = Vec::new();
        if joint_box.kernel.is_some() {
            for fraction in &grid {
                stages.push((
                    format!("frontier[parent_fraction={fraction}]"),
                    Line::Kernel { parent: *fraction },
                    kernel,
                ));
            }
        }
        if joint_box.parent.is_some() {
            stages.push(("axis[shared_parent_marginal]".into(), Line::Parent, parent));
        }
        let mut line_bytes = bytes;
        let mut outcomes = Vec::with_capacity(stages.len());
        let mut bracketing = Bracketing {
            factors: &factors,
            budget: &mut budget,
            threshold,
            upward,
            tolerance: spec.tolerance,
            bytes: line_bytes,
        };
        for (index, (name, line, extent)) in stages.iter().enumerate() {
            if stop.is_some() {
                unevaluated.push(name.clone());
                outcomes.push((TippingStatus::Unevaluated, None));
                continue;
            }
            line_bytes += LINE_BYTES;
            peak_bytes = line_bytes;
            bracketing.bytes = line_bytes;
            match bracketing.bracket(*line, *extent)? {
                Ok(outcome) => {
                    explored.push(name.clone());
                    outcomes.push(outcome);
                    if let Some(progress) = &ctx.progress {
                        progress.report(
                            (index + 1) as f64 / stages.len() as f64,
                            "joint sensitivity frontier",
                        );
                    }
                }
                Err(halt) => {
                    stop = Some(halt);
                    unevaluated.extend(stages[index..].iter().map(|(name, ..)| name.clone()));
                    outcomes.extend(std::iter::repeat_n(
                        (TippingStatus::Unevaluated, None),
                        stages.len() - index,
                    ));
                    break;
                }
            }
        }
        let mut outcomes = outcomes.into_iter();
        if joint_box.kernel.is_some() {
            for fraction in &grid {
                let (status, bracket) = outcomes.next().expect("one outcome per stage");
                frontier.push(FrontierPoint { parent_fraction: *fraction, status, bracket });
            }
            let first = frontier[0];
            axis_tipping.push(AxisTipping {
                factor: JointFactor::OutcomeKernel,
                analytic: factors.kernel_stage(kernel, Some(threshold))?.tipping_fraction,
                status: first.status,
                bracket: first.bracket,
            });
        }
        if joint_box.parent.is_some() {
            let (status, bracket) = outcomes.next().expect("one outcome per stage");
            axis_tipping.push(AxisTipping {
                factor: JointFactor::SharedParentMarginal,
                analytic: factors.parent_stage(parent, Some(threshold))?.tipping_fraction,
                status,
                bracket,
            });
        }
    }
    let mut canonical = spec.factors.clone();
    canonical.sort_by_key(|bound| bound.factor);
    let receipt = JointSensitivityReceipt {
        method: JOINT_SENSITIVITY_METHOD,
        fraction_box: [kernel, parent],
        tolerance: spec.tolerance,
        limits: spec.limits,
        memory_limit_bytes: budget.memory_limit_bytes(),
        operations_consumed: budget.operations(),
        depth_reached: budget.depth_reached(),
        live_state_bytes: peak_bytes,
        stop,
        explored,
        unevaluated,
    };
    Ok(JointMechanismSensitivityResult {
        query_binding: factors.kernels.query_binding.clone(),
        provider_snapshot: factors.kernels.provider_snapshot.clone(),
        source_regime: factors.kernels.source_regime,
        factors: canonical,
        decision_threshold: spec.decision_threshold,
        baseline,
        range,
        axis_tipping,
        frontier,
        unresolved_detail: stop.map(|_| JOINT_SENSITIVITY_BUDGET_DETAIL),
        receipt,
        interpretation: JOINT_SENSITIVITY_INTERPRETATION,
        inference_claim: ASSUMPTION_RANGE_CLAIM,
        uncertainty: JointSensitivityUncertainty::withheld(),
    })
}

/// Conservative endpoint percentile-bootstrap interval (calibration-internal).
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct JointSensitivityInterval {
    /// `q_{(1-level)/2}` of the replicate minima.
    pub lower: f64,
    /// `q_{(1+level)/2}` of the replicate maxima.
    pub upper: f64,
    /// Ordinary percentile interval of the replicate baselines (zero deviation).
    pub pointwise_at_zero: (f64, f64),
    /// Nominal level.
    pub level: f64,
    /// [`JOINT_SENSITIVITY_SAMPLING_METHOD`].
    pub method: &'static str,
    /// [`JOINT_SENSITIVITY_COVERAGE_TARGET`].
    pub coverage_target: &'static str,
    /// Requested replicates.
    pub replicates_requested: u32,
    /// Replicates whose range evaluated.
    pub replicates_ok: u32,
    /// Replicates whose draw or range failed.
    pub replicates_failed: u32,
}

/// The closed conservative endpoint bootstrap: each iid row-bootstrap draw of
/// the cited count table is evaluated by the exact joint range, and the
/// interval is `[q_{(1-level)/2}(L*), q_{(1+level)/2}(U*)]`.
///
/// For any fixed true deviation inside the box, `L*_b <= psi*_b <= U*_b`, so
/// the interval contains the pointwise percentile interval of `psi(delta0)`.
/// Compiled only under `calibration-internal`; no public route reaches it.
///
/// # Errors
///
/// An invalid spec, a formula or provider refusal on the observed data, fewer
/// than two replicates or cancellation.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn joint_sensitivity_bootstrap_interval_internal(
    diagram: &SelectionDiagram,
    functional: &BoundZTransportFunctional,
    data: &ExactTransportData,
    spec: &JointDeviationSpec,
    replicates: u32,
    level: f64,
    ctx: &ExecutionContext,
) -> Result<Result<JointSensitivityInterval, &'static str>, JointSensitivityError> {
    let joint_box = validate_spec(spec)?;
    if !(2..=JOINT_SENSITIVITY_MAX_BOOTSTRAP).contains(&replicates)
        || !level.is_finite()
        || level <= 0.0
        || level >= 1.0
    {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "joint_sensitivity.bounds_exceeded",
            format!("{replicates} replicates (2-2000) or level {level} (strictly inside (0, 1))"),
        ));
    }
    read_factors(diagram, functional, data, ctx, None)?;
    let draws =
        antecedent_estimate::z_transport_bootstrap_law_draws(functional, data, replicates, ctx)
            .map_err(|error| {
                if ctx.cancellation.is_cancelled() {
                    refuse(
                        reason_code!("transport_budget_cancel"),
                        "joint_sensitivity.budget",
                        error.to_string(),
                    )
                } else {
                    refuse(
                        reason_code!("transport_support_failure"),
                        "joint_sensitivity.incomplete_kernel",
                        error.to_string(),
                    )
                }
            })?;
    let draws = match draws {
        Ok(draws) => draws,
        Err(reason) => return Ok(Err(reason)),
    };
    let (mut lows, mut highs, mut zeros) = (Vec::new(), Vec::new(), Vec::new());
    let mut failed = 0u32;
    for draw in draws {
        if ctx.cancellation.is_cancelled() {
            return Err(refuse(
                reason_code!("transport_budget_cancel"),
                "joint_sensitivity.budget",
                "endpoint bootstrap cancelled",
            ));
        }
        let evaluated = draw.and_then(|draw| {
            let factors = read_factors(diagram, functional, &draw, ctx, None).ok()?;
            let range = factors.bounds_at(joint_box.kernel(), joint_box.parent()).ok()?;
            let zero = factors.kernel_stage(0.0, None).ok()?.baseline;
            Some((range.minimum, range.maximum, zero))
        });
        match evaluated {
            Some((low, high, zero)) => {
                lows.push(low);
                highs.push(high);
                zeros.push(zero);
            }
            None => failed += 1,
        }
    }
    let ok = u32::try_from(lows.len()).unwrap_or(u32::MAX);
    if let Err(reason) =
        antecedent_estimate::ReplicatePolicy::BOOTSTRAP.decide(replicates, ok, failed)
    {
        return Ok(Err(reason));
    }
    let (Some((lower, _)), Some((_, upper)), Some(pointwise_at_zero)) = (
        antecedent_estimate::percentile_interval(&lows, level),
        antecedent_estimate::percentile_interval(&highs, level),
        antecedent_estimate::percentile_interval(&zeros, level),
    ) else {
        return Ok(Err(antecedent_estimate::INTERVAL_NUMERICAL_FAILURE));
    };
    Ok(Ok(JointSensitivityInterval {
        lower,
        upper,
        pointwise_at_zero,
        level,
        method: JOINT_SENSITIVITY_SAMPLING_METHOD,
        coverage_target: JOINT_SENSITIVITY_COVERAGE_TARGET,
        replicates_requested: replicates,
        replicates_ok: ok,
        replicates_failed: failed,
    }))
}

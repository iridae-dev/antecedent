//! Finite-action inverse-outcome query over a licensed forward response.
//!
//! The question is "which of these enumerated interventions reach a target
//! mean?": `E[Y^do(a)] >= threshold` (or `<=`) for a caller-supplied finite
//! action grid, each action with a declared cost and named constraints. The
//! module owns no estimator. The forward response `a -> E[Y^do(a)]` is
//! evaluated by an existing, licensed response route and handed in as a
//! [`ForwardEvaluation`]; this module classifies every enumerated action
//! against it and refuses what it cannot answer:
//!
//! - an action off the evaluated grid is `unevaluated`, never interpolated;
//! - an action the forward route reports outside empirical support (or whose
//!   support evidence is missing) is `unsupported`;
//! - an action that violates a declared constraint or budget is `infeasible`;
//! - the remaining actions are `feasible` or `infeasible` by their point
//!   estimate against the threshold, within a declared numerical tolerance.
//!
//! Classification is point-based. When the forward route carries a pointwise
//! or simultaneous interval, the report additionally flags an action
//! `robustly_feasible` when that interval, used exactly as published, lies on
//! the feasible side of the threshold. That flag adds no coverage claim: it is
//! not a joint statement over the action grid for a pointwise band, and it
//! does not account for the selection of an action by its estimate.
//!
//! "Feasible", "infeasible", "necessary" and "sufficient" are relative to the
//! enumerated action set and the forward response's declared assumptions;
//! an action that was not enumerated is unresolved, never excluded. A mean
//! threshold is not a chance constraint `P(Y^do(a) >= y) >= q`, a quantile
//! target needs the interventional distribution (2.3B), and an observational
//! `P(X | Y)` is not an action: all three are typed refusals.
//!
//! The report carries an order-invariant identity digest of every input and
//! is a pure function of them, so an independent consumer recomputes it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use antecedent_core::{CancellationToken, IntervalInterpretation, SupportStatus, reason_code};

/// Most actions one query may enumerate.
pub const MAX_INVERSE_ACTIONS: usize = 4096;

/// Most rows of the forward evaluation the classifier reads.
pub const MAX_INVERSE_FORWARD_POINTS: usize = 65_536;

/// What a result is and is not, carried on every report.
pub const INVERSE_SCOPE_NOTE: &str = "Classification is relative to the enumerated action set and the forward response's declared assumptions; an action that was not enumerated is unresolved, not excluded. Feasibility is point-based; a robustly-feasible flag uses the forward route's own interval as published and adds no coverage claim.";

/// A refusal of an inverse-outcome query, with its registered reason code.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct InverseOutcomeError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `inverse.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

fn invalid(detail: &'static str, message: impl Into<String>) -> InverseOutcomeError {
    InverseOutcomeError { code: reason_code!("invalid_argument"), detail, message: message.into() }
}

fn not_licensed(detail: &'static str, message: impl Into<String>) -> InverseOutcomeError {
    InverseOutcomeError { code: reason_code!("cell_not_licensed"), detail, message: message.into() }
}

fn not_supported(detail: &'static str, message: impl Into<String>) -> InverseOutcomeError {
    InverseOutcomeError {
        code: reason_code!("route_not_supported"),
        detail,
        message: message.into(),
    }
}

/// Which side of the threshold is the goal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetDirection {
    /// `E[Y^do(a)] >= threshold`.
    AtLeast,
    /// `E[Y^do(a)] <= threshold`.
    AtMost,
}

impl TargetDirection {
    /// Stable `snake_case` spelling used on the Python wire.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AtLeast => "at_least",
            Self::AtMost => "at_most",
        }
    }

    /// Parse the wire spelling.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "at_least" => Some(Self::AtLeast),
            "at_most" => Some(Self::AtMost),
            _ => None,
        }
    }
}

/// The question asked. Only the target-mean query executes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InverseQuery {
    /// A threshold on the interventional mean.
    TargetMean {
        /// Threshold on `E[Y^do(a)]`.
        threshold: f64,
        /// Which side of the threshold is the goal.
        direction: TargetDirection,
    },
    /// `P(Y^do(a) >= threshold) >= probability`: refused.
    ChanceConstraint {
        /// Outcome threshold.
        threshold: f64,
        /// Required probability.
        probability: f64,
    },
    /// A target quantile of `Y^do(a)`: refused (2.3B).
    Quantile {
        /// Quantile level.
        level: f64,
        /// Outcome threshold on that quantile.
        threshold: f64,
    },
    /// Observational scenarios compatible with an outcome: refused.
    ObservationalScenarios,
}

/// Whether the forward evaluation carries per-point or one surface-wide support label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportBasis {
    /// One label per evaluated point (a temporal dose-by-horizon surface).
    PerPoint,
    /// The worst label over the whole surface, copied to every point (a static curve).
    SurfaceWorstCase,
}

impl SupportBasis {
    /// Stable `snake_case` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PerPoint => "per_point",
            Self::SurfaceWorstCase => "surface_worst_case",
        }
    }

    /// Parse the wire spelling.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "per_point" => Some(Self::PerPoint),
            "surface_worst_case" => Some(Self::SurfaceWorstCase),
            _ => None,
        }
    }
}

/// Coverage scope of the forward interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalScope {
    /// One interval per grid point, no joint coverage.
    Pointwise,
    /// One band calibrated over the whole evaluated grid.
    Simultaneous,
}

impl IntervalScope {
    /// Stable `snake_case` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pointwise => "pointwise",
            Self::Simultaneous => "simultaneous",
        }
    }

    /// Parse the wire spelling.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "pointwise" => Some(Self::Pointwise),
            "simultaneous" => Some(Self::Simultaneous),
            _ => None,
        }
    }
}

/// An interval published by the forward route, row-aligned with its points.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardInterval {
    /// Pointwise or simultaneous.
    pub scope: IntervalScope,
    /// Confidence or credible.
    pub interpretation: IntervalInterpretation,
    /// The forward route's own level.
    pub level: f64,
    /// Lower endpoint per evaluated point.
    pub lower: Vec<f64>,
    /// Upper endpoint per evaluated point.
    pub upper: Vec<f64>,
}

/// The forward response `a -> E[Y^do(a)]` as the licensed route evaluated it.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardEvaluation {
    /// Outcome name.
    pub outcome: String,
    /// Declared population label (the forward route's target population).
    pub population: String,
    /// Identity of the forward execution claim, when the route supplied one.
    pub claim_id: Option<String>,
    /// Identity of the prepared forward program, when available.
    pub program_id: Option<String>,
    /// Identity of the forward data snapshot, when available.
    pub data_snapshot_id: Option<String>,
    /// The forward estimand is a mean response (not a derivative or a contrast).
    pub mean_response: bool,
    /// The forward response is point identified (not an identified set or a class mixture).
    pub point_identified: bool,
    /// Coordinates per evaluated point (dose, or dose and horizon).
    pub dimension: usize,
    /// Evaluated points, row-major by point.
    pub points: Vec<Vec<f64>>,
    /// Mean response per evaluated point.
    pub mean: Vec<f64>,
    /// Support label per evaluated point.
    pub support: Vec<SupportStatus>,
    /// Whether `support` is per point or one surface-wide worst case.
    pub support_basis: SupportBasis,
    /// The forward route's interval, if it published one.
    pub interval: Option<ForwardInterval>,
    /// The forward route's declared assumptions, verbatim.
    pub assumptions: Vec<String>,
}

/// One caller-declared constraint on an action; the caller evaluates it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionConstraint {
    /// Constraint name.
    pub name: String,
    /// Whether the action satisfies it.
    pub satisfied: bool,
}

/// One enumerated action: a point on the forward grid with a cost and constraints.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionSpec {
    /// Unique action label.
    pub label: String,
    /// Intervention coordinates, in the forward evaluation's coordinate order.
    pub point: Vec<f64>,
    /// Declared cost (caller units).
    pub cost: f64,
    /// Declared constraints.
    pub constraints: Vec<ActionConstraint>,
}

/// Classification of one action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionStatus {
    /// Evaluated, supported, within constraints, and the target is met.
    Feasible,
    /// A constraint or budget is violated, or the evaluated target is missed.
    Infeasible,
    /// The forward route reports no empirical support (or no support evidence) here.
    Unsupported,
    /// The forward route did not produce a finite estimate for this action.
    Unevaluated,
}

impl ActionStatus {
    /// Stable `snake_case` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Feasible => "feasible",
            Self::Infeasible => "infeasible",
            Self::Unsupported => "unsupported",
            Self::Unevaluated => "unevaluated",
        }
    }
}

/// One action's result.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionOutcome {
    /// Action label.
    pub label: String,
    /// Intervention coordinates.
    pub point: Vec<f64>,
    /// Declared cost.
    pub cost: f64,
    /// Classification.
    pub status: ActionStatus,
    /// Stable reason for the classification.
    pub reason: &'static str,
    /// Names of the violated constraints, sorted.
    pub violated_constraints: Vec<String>,
    /// Forward point estimate, when finite and on the grid.
    pub estimate: Option<f64>,
    /// Signed distance to the goal side of the threshold (positive meets it).
    pub margin: Option<f64>,
    /// The margin is within the numerical tolerance of zero.
    pub within_tolerance: bool,
    /// Support label the forward route gave this point.
    pub support_status: Option<SupportStatus>,
    /// The forward interval at this point, as published.
    pub interval: Option<(f64, f64)>,
    /// The published interval lies on the feasible side; `None` without an
    /// interval or for an action that is not point-feasible.
    pub robustly_feasible: Option<bool>,
}

/// What the enumeration as a whole established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnumeratedStatus {
    /// At least one enumerated action is feasible.
    Reachable,
    /// Every enumerated action is infeasible: the target is unreachable within the set.
    UnreachableWithinSet,
    /// No action is feasible but some are unsupported or unevaluated: unresolved.
    Undetermined,
}

impl EnumeratedStatus {
    /// Stable `snake_case` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reachable => "reachable",
            Self::UnreachableWithinSet => "unreachable_within_set",
            Self::Undetermined => "undetermined",
        }
    }
}

/// The forward interval's metadata, carried once on the report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalMeta {
    /// Pointwise or simultaneous.
    pub scope: IntervalScope,
    /// Confidence or credible.
    pub interpretation: IntervalInterpretation,
    /// The forward route's own level.
    pub level: f64,
}

/// The classified enumeration.
#[derive(Clone, Debug, PartialEq)]
pub struct InverseOutcomeReport {
    /// Threshold on `E[Y^do(a)]`.
    pub threshold: f64,
    /// Which side of the threshold is the goal.
    pub direction: TargetDirection,
    /// Numerical tolerance on the margin.
    pub tolerance: f64,
    /// Budget on action cost, if declared.
    pub budget: Option<f64>,
    /// Outcome name.
    pub outcome: String,
    /// Declared population label.
    pub population: String,
    /// Forward execution claim identity, when available.
    pub forward_claim_id: Option<String>,
    /// Forward prepared-program identity, when available.
    pub forward_program_id: Option<String>,
    /// Forward data-snapshot identity, when available.
    pub forward_data_snapshot_id: Option<String>,
    /// The forward route's assumptions, verbatim.
    pub assumptions: Vec<String>,
    /// Whether support labels are per point or a surface-wide worst case.
    pub support_basis: SupportBasis,
    /// The forward interval's metadata when it published one; `None` is point-only.
    pub interval: Option<IntervalMeta>,
    /// Every enumerated action, in the caller's order.
    pub outcomes: Vec<ActionOutcome>,
    /// Labels of feasible actions, sorted.
    pub feasible: Vec<String>,
    /// Labels of infeasible actions, sorted.
    pub infeasible: Vec<String>,
    /// Labels of unsupported actions, sorted.
    pub unsupported: Vec<String>,
    /// Labels of unevaluated actions, sorted.
    pub unevaluated: Vec<String>,
    /// Labels of the cheapest feasible actions (all ties), sorted.
    pub cheapest_feasible: Vec<String>,
    /// Labels of feasible actions whose published interval also clears the threshold, sorted.
    pub robustly_feasible: Vec<String>,
    /// What the enumeration established.
    pub enumerated: EnumeratedStatus,
    /// Order-invariant digest of every input.
    pub identity: String,
}

fn canonical_bits(value: f64) -> u64 {
    if value.to_bits() == (-0.0_f64).to_bits() { 0 } else { value.to_bits() }
}

fn point_key(point: &[f64]) -> Vec<u64> {
    point.iter().map(|value| canonical_bits(*value)).collect()
}

#[derive(Default)]
struct Canon(Vec<u8>);

impl Canon {
    fn word(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn float(&mut self, value: f64) {
        self.word(canonical_bits(value));
    }

    fn text(&mut self, value: &str) {
        self.word(value.len() as u64);
        self.0.extend_from_slice(value.as_bytes());
    }

    fn flag(&mut self, value: bool) {
        self.0.push(u8::from(value));
    }

    fn optional(&mut self, value: Option<f64>) {
        self.flag(value.is_some());
        self.float(value.unwrap_or(0.0));
    }
}

fn identity(
    threshold: f64,
    direction: TargetDirection,
    forward: &ForwardEvaluation,
    actions: &[ActionSpec],
    budget: Option<f64>,
    tolerance: f64,
) -> String {
    let mut canon = Canon::default();
    canon.text("inverse_outcome.target_mean.v2");
    canon.float(threshold);
    canon.text(direction.as_str());
    canon.float(tolerance);
    canon.optional(budget);
    canon.text(&forward.outcome);
    canon.text(&forward.population);
    for value in [&forward.claim_id, &forward.program_id, &forward.data_snapshot_id] {
        canon.flag(value.is_some());
        if let Some(value) = value {
            canon.text(value);
        }
    }
    canon.flag(forward.mean_response);
    canon.flag(forward.point_identified);
    canon.word(forward.dimension as u64);
    canon.text(forward.support_basis.as_str());
    let mut rows: Vec<usize> = (0..forward.points.len()).collect();
    rows.sort_by_key(|row| point_key(&forward.points[*row]));
    canon.word(rows.len() as u64);
    for row in &rows {
        for value in &forward.points[*row] {
            canon.float(*value);
        }
        canon.float(forward.mean[*row]);
        canon.text(forward.support[*row].as_str());
        match &forward.interval {
            Some(interval) => {
                canon.flag(true);
                canon.float(interval.lower[*row]);
                canon.float(interval.upper[*row]);
            }
            None => canon.flag(false),
        }
    }
    match &forward.interval {
        Some(interval) => {
            canon.flag(true);
            canon.text(interval.scope.as_str());
            canon.text(interval.interpretation.as_str());
            canon.float(interval.level);
        }
        None => canon.flag(false),
    }
    canon.word(forward.assumptions.len() as u64);
    for assumption in &forward.assumptions {
        canon.text(assumption);
    }
    let mut ordered: Vec<&ActionSpec> = actions.iter().collect();
    ordered.sort_by(|a, b| a.label.cmp(&b.label));
    canon.word(ordered.len() as u64);
    for action in ordered {
        canon.text(&action.label);
        canon.word(action.point.len() as u64);
        for value in &action.point {
            canon.float(*value);
        }
        canon.float(action.cost);
        let mut constraints: Vec<&ActionConstraint> = action.constraints.iter().collect();
        constraints.sort_by(|a, b| a.name.cmp(&b.name));
        canon.word(constraints.len() as u64);
        for constraint in constraints {
            canon.text(&constraint.name);
            canon.flag(constraint.satisfied);
        }
    }
    let digest = antecedent_io::identity::payload_digest("inverse_outcome_report", &canon.0);
    digest.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

fn validate_forward(
    forward: &ForwardEvaluation,
) -> Result<BTreeMap<Vec<u64>, usize>, InverseOutcomeError> {
    if !forward.mean_response {
        return Err(not_supported(
            "inverse.forward_not_mean_response",
            "the forward route's estimand is not a mean response a -> E[Y^do(a)] (a derivative, \
             contrast or other functional carries no target-mean reading)",
        ));
    }
    if !forward.point_identified {
        return Err(not_supported(
            "inverse.forward_not_point_identified",
            "the forward response is not point identified (an identified set, a class mixture or \
             an unidentified response has no single mean to compare with a threshold)",
        ));
    }
    let rows = forward.points.len();
    if !(1..=2).contains(&forward.dimension)
        || rows == 0
        || forward.mean.len() != rows
        || forward.support.len() != rows
    {
        return Err(invalid(
            "inverse.invalid_forward",
            "the forward evaluation needs at least one point, one dose coordinate with an \
             optional horizon coordinate, and one mean and support label per point",
        ));
    }
    if rows > MAX_INVERSE_FORWARD_POINTS {
        return Err(not_supported(
            "inverse.bounds_exceeded",
            format!(
                "the forward evaluation has {rows} points; the limit is {MAX_INVERSE_FORWARD_POINTS}"
            ),
        ));
    }
    let mut index = BTreeMap::new();
    for (row, point) in forward.points.iter().enumerate() {
        if point.len() != forward.dimension || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "inverse.invalid_forward",
                "every forward point needs `dimension` finite coordinates",
            ));
        }
        if index.insert(point_key(point), row).is_some() {
            return Err(invalid(
                "inverse.invalid_forward",
                "the forward evaluation repeats a point, so its mean is not a function of the action",
            ));
        }
    }
    if let Some(interval) = &forward.interval {
        let bounded = interval.lower.len() == rows
            && interval.upper.len() == rows
            && interval.level > 0.0
            && interval.level < 1.0
            && interval.lower.iter().zip(&interval.upper).all(|(lo, hi)| lo <= hi);
        if !bounded {
            return Err(invalid(
                "inverse.invalid_forward",
                "the forward interval needs ordered, non-NaN endpoints per point and a level in (0, 1)",
            ));
        }
    }
    Ok(index)
}

fn validate_actions(actions: &[ActionSpec], dimension: usize) -> Result<(), InverseOutcomeError> {
    if actions.is_empty() {
        return Err(invalid("inverse.invalid_action", "the action grid is empty"));
    }
    if actions.len() > MAX_INVERSE_ACTIONS {
        return Err(not_supported(
            "inverse.bounds_exceeded",
            format!("{} actions enumerated; the limit is {MAX_INVERSE_ACTIONS}", actions.len()),
        ));
    }
    let mut labels = BTreeSet::new();
    let mut horizon = None;
    for action in actions {
        if action.label.trim().is_empty() || !labels.insert(action.label.as_str()) {
            return Err(invalid(
                "inverse.invalid_action",
                "action labels must be non-empty and unique",
            ));
        }
        if action.point.len() != dimension || action.point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "inverse.invalid_action",
                format!("action `{}` needs {dimension} finite coordinates", action.label),
            ));
        }
        if dimension == 2 {
            let current = canonical_bits(action.point[1]);
            if horizon.replace(current).is_some_and(|first| first != current) {
                return Err(invalid(
                    "inverse.invalid_action",
                    "one inverse-outcome enumeration must use one fixed outcome horizon",
                ));
            }
        }
        if !action.cost.is_finite() || action.cost < 0.0 {
            return Err(invalid(
                "inverse.invalid_action",
                format!("action `{}` needs a finite non-negative cost", action.label),
            ));
        }
        let mut names = BTreeSet::new();
        for constraint in &action.constraints {
            if constraint.name.trim().is_empty() || !names.insert(constraint.name.as_str()) {
                return Err(invalid(
                    "inverse.invalid_action",
                    format!("action `{}` needs non-empty, distinct constraint names", action.label),
                ));
            }
        }
    }
    Ok(())
}

fn classify_action(
    action: &ActionSpec,
    row: Option<usize>,
    forward: &ForwardEvaluation,
    threshold: f64,
    direction: TargetDirection,
    budget: Option<f64>,
    tolerance: f64,
) -> ActionOutcome {
    let mut violated: Vec<String> = action
        .constraints
        .iter()
        .filter(|constraint| !constraint.satisfied)
        .map(|constraint| constraint.name.clone())
        .collect();
    violated.sort();
    let estimate = row.map(|r| forward.mean[r]).filter(|value| value.is_finite());
    let support_status = row.map(|r| forward.support[r]);
    let interval = match (&forward.interval, row) {
        (Some(published), Some(r)) => Some((published.lower[r], published.upper[r])),
        _ => None,
    };
    let signed = |value: f64| match direction {
        TargetDirection::AtLeast => value - threshold,
        TargetDirection::AtMost => threshold - value,
    };
    let margin = estimate.and_then(|value| {
        let gap = signed(value);
        gap.is_finite().then_some(gap)
    });
    let (status, reason) = if !violated.is_empty() {
        (ActionStatus::Infeasible, "constraint_violated")
    } else if budget.is_some_and(|limit| action.cost > limit) {
        (ActionStatus::Infeasible, "over_budget")
    } else if row.is_none() {
        (ActionStatus::Unevaluated, "not_on_evaluated_grid")
    } else if let Some(
        label @ (SupportStatus::OutsideEmpiricalSupport | SupportStatus::MissingEvidence),
    ) = support_status
    {
        (ActionStatus::Unsupported, label.as_str())
    } else if let Some(gap) = margin {
        if gap >= -tolerance {
            (ActionStatus::Feasible, "meets_target")
        } else {
            (ActionStatus::Infeasible, "misses_target")
        }
    } else if estimate.is_some() {
        (ActionStatus::Unevaluated, "numerical_margin_overflow")
    } else {
        (ActionStatus::Unevaluated, "non_finite_estimate")
    };
    let robustly_feasible = match (status, interval) {
        (ActionStatus::Feasible, Some((lower, upper))) => {
            let worst = match direction {
                TargetDirection::AtLeast => lower,
                TargetDirection::AtMost => upper,
            };
            Some(signed(worst) >= -tolerance)
        }
        _ => None,
    };
    ActionOutcome {
        label: action.label.clone(),
        point: action.point.clone(),
        cost: action.cost,
        status,
        reason,
        violated_constraints: violated,
        estimate,
        margin,
        within_tolerance: margin.is_some_and(|gap| gap.abs() <= tolerance),
        support_status,
        interval,
        robustly_feasible,
    }
}

fn labels_where(outcomes: &[ActionOutcome], keep: impl Fn(&ActionOutcome) -> bool) -> Vec<String> {
    let mut labels: Vec<String> =
        outcomes.iter().filter(|outcome| keep(outcome)).map(|o| o.label.clone()).collect();
    labels.sort();
    labels
}

/// Classify every enumerated action against a licensed forward response.
///
/// `query` must be a target-mean query; the probability, quantile and
/// observational-scenario queries are refused before any input is read.
/// Cancellation is observed once per action and is never a verdict.
///
/// # Errors
/// A typed refusal for an unsupported query kind, an invalid target, action or
/// forward summary, an exceeded bound, or cancellation.
pub fn classify_inverse_outcome(
    query: &InverseQuery,
    forward: &ForwardEvaluation,
    actions: &[ActionSpec],
    budget: Option<f64>,
    tolerance: f64,
    cancel: &CancellationToken,
) -> Result<InverseOutcomeReport, InverseOutcomeError> {
    let (threshold, direction) = match *query {
        InverseQuery::TargetMean { threshold, direction } => (threshold, direction),
        InverseQuery::ChanceConstraint { .. } => {
            return Err(not_licensed(
                "inverse.probability_target",
                "a mean response cannot establish a chance constraint P(Y^do(a) >= y) >= q; that \
                 needs a separately identified and calibrated interventional-distribution cell",
            ));
        }
        InverseQuery::Quantile { .. } => {
            return Err(not_licensed(
                "inverse.quantile_target",
                "a target-quantile inverse query needs the interventional distribution and is \
                 scheduled for 2.3B",
            ));
        }
        InverseQuery::ObservationalScenarios => {
            return Err(not_licensed(
                "inverse.observational_scenarios",
                "observational scenarios compatible with an outcome (P(X | Y)) are not actions \
                 and are closed in 2.2; ask for feasible interventions instead",
            ));
        }
    };
    if !threshold.is_finite() {
        return Err(invalid("inverse.invalid_target", "the threshold must be finite"));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid(
            "inverse.invalid_target",
            "the tolerance must be finite and non-negative",
        ));
    }
    if budget.is_some_and(|limit| !limit.is_finite() || limit < 0.0) {
        return Err(invalid(
            "inverse.invalid_target",
            "the budget must be finite and non-negative",
        ));
    }
    let index = validate_forward(forward)?;
    validate_actions(actions, forward.dimension)?;
    let mut outcomes = Vec::with_capacity(actions.len());
    for action in actions {
        if cancel.is_cancelled() {
            return Err(InverseOutcomeError {
                code: reason_code!("transport_budget_cancel"),
                detail: "inverse.cancelled",
                message: "the enumeration was cancelled before every action was classified; no \
                          verdict is reported"
                    .to_owned(),
            });
        }
        let row = index.get(&point_key(&action.point)).copied();
        outcomes
            .push(classify_action(action, row, forward, threshold, direction, budget, tolerance));
    }
    let feasible = labels_where(&outcomes, |o| o.status == ActionStatus::Feasible);
    let infeasible = labels_where(&outcomes, |o| o.status == ActionStatus::Infeasible);
    let unsupported = labels_where(&outcomes, |o| o.status == ActionStatus::Unsupported);
    let unevaluated = labels_where(&outcomes, |o| o.status == ActionStatus::Unevaluated);
    let cheapest = outcomes
        .iter()
        .filter(|o| o.status == ActionStatus::Feasible)
        .map(|o| o.cost)
        .min_by(f64::total_cmp);
    let cheapest_feasible = labels_where(&outcomes, |o| {
        o.status == ActionStatus::Feasible
            && cheapest.is_some_and(|cost| o.cost.total_cmp(&cost).is_eq())
    });
    let robustly_feasible = labels_where(&outcomes, |o| o.robustly_feasible == Some(true));
    let enumerated = if !feasible.is_empty() {
        EnumeratedStatus::Reachable
    } else if unsupported.is_empty() && unevaluated.is_empty() {
        EnumeratedStatus::UnreachableWithinSet
    } else {
        EnumeratedStatus::Undetermined
    };
    Ok(InverseOutcomeReport {
        threshold,
        direction,
        tolerance,
        budget,
        outcome: forward.outcome.clone(),
        population: forward.population.clone(),
        forward_claim_id: forward.claim_id.clone(),
        forward_program_id: forward.program_id.clone(),
        forward_data_snapshot_id: forward.data_snapshot_id.clone(),
        assumptions: forward.assumptions.clone(),
        support_basis: forward.support_basis,
        interval: forward.interval.as_ref().map(|interval| IntervalMeta {
            scope: interval.scope,
            interpretation: interval.interpretation,
            level: interval.level,
        }),
        outcomes,
        feasible,
        infeasible,
        unsupported,
        unevaluated,
        cheapest_feasible,
        robustly_feasible,
        enumerated,
        identity: identity(threshold, direction, forward, actions, budget, tolerance),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known structural law `E[Y | do(a)] = 1 + 2a` evaluated on a grid.
    fn linear_forward(grid: &[f64]) -> ForwardEvaluation {
        ForwardEvaluation {
            outcome: "y".into(),
            population: "source".into(),
            claim_id: None,
            program_id: None,
            data_snapshot_id: None,
            mean_response: true,
            point_identified: true,
            dimension: 1,
            points: grid.iter().map(|a| vec![*a]).collect(),
            mean: grid.iter().map(|a| 1.0 + 2.0 * a).collect(),
            support: vec![SupportStatus::Supported; grid.len()],
            support_basis: SupportBasis::SurfaceWorstCase,
            interval: None,
            assumptions: vec!["backdoor adjustment on z".into()],
        }
    }

    fn action(label: &str, dose: f64, cost: f64) -> ActionSpec {
        ActionSpec { label: label.into(), point: vec![dose], cost, constraints: Vec::new() }
    }

    fn grid_actions(grid: &[f64]) -> Vec<ActionSpec> {
        grid.iter().map(|a| action(&format!("dose_{a}"), *a, a.abs())).collect()
    }

    fn at_least(threshold: f64) -> InverseQuery {
        InverseQuery::TargetMean { threshold, direction: TargetDirection::AtLeast }
    }

    fn run(
        query: &InverseQuery,
        forward: &ForwardEvaluation,
        actions: &[ActionSpec],
    ) -> Result<InverseOutcomeReport, InverseOutcomeError> {
        classify_inverse_outcome(query, forward, actions, None, 1e-12, &CancellationToken::new())
    }

    fn status_of(report: &InverseOutcomeReport, label: &str) -> ActionStatus {
        report.outcomes.iter().find(|o| o.label == label).map(|o| o.status).unwrap()
    }

    const GRID: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

    #[test]
    fn forward_inverse_round_trip_matches_the_analytic_inverse() {
        // E[Y | do(a)] = 1 + 2a >= 3 holds exactly when a >= (3 - 1) / 2 = 1.
        let report = run(&at_least(3.0), &linear_forward(&GRID), &grid_actions(&GRID)).unwrap();
        for dose in GRID {
            let expected = if dose >= (3.0 - 1.0) / 2.0 {
                ActionStatus::Feasible
            } else {
                ActionStatus::Infeasible
            };
            assert_eq!(status_of(&report, &format!("dose_{dose}")), expected, "dose {dose}");
        }
        assert_eq!(report.enumerated, EnumeratedStatus::Reachable);
        // The boundary action sits exactly on the threshold and is within tolerance.
        let boundary = report.outcomes.iter().find(|o| o.label == "dose_1").unwrap();
        assert!(boundary.within_tolerance);
        assert_eq!(boundary.reason, "meets_target");
        // An upper-side goal inverts the set: 1 + 2a <= 3 holds for a <= 1.
        let upper = InverseQuery::TargetMean { threshold: 3.0, direction: TargetDirection::AtMost };
        let report = run(&upper, &linear_forward(&GRID), &grid_actions(&GRID)).unwrap();
        assert_eq!(report.feasible, vec!["dose_0", "dose_0.5", "dose_1"]);
    }

    #[test]
    fn several_feasible_actions_report_every_one_and_every_cost_tie() {
        let mut actions = grid_actions(&GRID);
        actions[3].cost = 1.0; // dose 1.5 now costs the same as dose 1.0
        let report = run(&at_least(3.0), &linear_forward(&GRID), &actions).unwrap();
        assert_eq!(report.feasible, vec!["dose_1", "dose_1.5", "dose_2"]);
        assert_eq!(report.cheapest_feasible, vec!["dose_1", "dose_1.5"]);
        assert_eq!(report.infeasible, vec!["dose_0", "dose_0.5"]);
    }

    #[test]
    fn an_unreachable_target_is_unreachable_only_within_the_enumerated_set() {
        let report = run(&at_least(100.0), &linear_forward(&GRID), &grid_actions(&GRID)).unwrap();
        assert!(report.feasible.is_empty());
        assert_eq!(report.enumerated, EnumeratedStatus::UnreachableWithinSet);
        assert!(report.cheapest_feasible.is_empty());
        assert!(report.outcomes.iter().all(|o| o.reason == "misses_target"));
    }

    #[test]
    fn unsupported_actions_are_not_classified_by_their_estimate() {
        let mut forward = linear_forward(&GRID);
        forward.support_basis = SupportBasis::PerPoint;
        forward.support[4] = SupportStatus::OutsideEmpiricalSupport;
        forward.support[3] = SupportStatus::MissingEvidence;
        forward.support[2] = SupportStatus::Extrapolative;
        let report = run(&at_least(3.0), &forward, &grid_actions(&GRID)).unwrap();
        assert_eq!(status_of(&report, "dose_2"), ActionStatus::Unsupported);
        assert_eq!(status_of(&report, "dose_1.5"), ActionStatus::Unsupported);
        // Extrapolative support is reported, not hidden, and does not block a verdict.
        let extrapolated = report.outcomes.iter().find(|o| o.label == "dose_1").unwrap();
        assert_eq!(extrapolated.status, ActionStatus::Feasible);
        assert_eq!(extrapolated.support_status, Some(SupportStatus::Extrapolative));
        assert_eq!(report.unsupported, vec!["dose_1.5", "dose_2"]);
        // With nothing feasible, unsupported actions leave the answer undetermined.
        let report = run(&at_least(3.0), &forward, &grid_actions(&[1.5, 2.0])).unwrap();
        assert_eq!(report.enumerated, EnumeratedStatus::Undetermined);
    }

    #[test]
    fn an_action_off_the_evaluated_grid_is_unevaluated_not_interpolated() {
        let forward = linear_forward(&[0.0, 1.0, 2.0]);
        let mut actions = grid_actions(&[0.0, 1.0, 2.0]);
        actions.push(action("between", 1.25, 1.25));
        actions.push(action("beyond", 9.0, 9.0));
        let report = run(&at_least(3.0), &forward, &actions).unwrap();
        assert_eq!(report.unevaluated, vec!["between", "beyond"]);
        assert_eq!(report.feasible, vec!["dose_1", "dose_2"]);
        assert_eq!(report.enumerated, EnumeratedStatus::Reachable);
        let missed = report.outcomes.iter().find(|o| o.label == "between").unwrap();
        assert_eq!((missed.reason, missed.estimate), ("not_on_evaluated_grid", None));
        // Without any feasible action the missing evaluations keep the answer open.
        let report = run(&at_least(100.0), &forward, &actions).unwrap();
        assert_eq!(report.enumerated, EnumeratedStatus::Undetermined);
        // A non-finite forward estimate is unevaluated too.
        let mut forward = linear_forward(&[0.0, 1.0]);
        forward.mean[1] = f64::NAN;
        let report = run(&at_least(0.0), &forward, &grid_actions(&[0.0, 1.0])).unwrap();
        assert_eq!(report.unevaluated, vec!["dose_1"]);
    }

    #[test]
    fn constraints_and_budget_make_an_action_infeasible_before_its_estimate_matters() {
        let mut actions = grid_actions(&GRID);
        actions[4].constraints = vec![
            ActionConstraint { name: "regulatory_cap".into(), satisfied: false },
            ActionConstraint { name: "available".into(), satisfied: true },
        ];
        let report = classify_inverse_outcome(
            &at_least(3.0),
            &linear_forward(&GRID),
            &actions,
            Some(1.2),
            1e-12,
            &CancellationToken::new(),
        )
        .unwrap();
        let capped = report.outcomes.iter().find(|o| o.label == "dose_2").unwrap();
        assert_eq!(capped.reason, "constraint_violated");
        assert_eq!(capped.violated_constraints, vec!["regulatory_cap"]);
        // Dose 1.5 meets the target (1 + 3 = 4) but costs 1.5 > the 1.2 budget.
        let costly = report.outcomes.iter().find(|o| o.label == "dose_1.5").unwrap();
        assert_eq!((costly.status, costly.reason), (ActionStatus::Infeasible, "over_budget"));
        assert_eq!(report.feasible, vec!["dose_1"]);
    }

    #[test]
    fn a_changed_assumption_or_forward_response_changes_the_identity_and_the_verdict() {
        let base = run(&at_least(3.0), &linear_forward(&GRID), &grid_actions(&GRID)).unwrap();
        let mut other = linear_forward(&GRID);
        other.assumptions.push("no unmeasured confounding of t and y".into());
        let reassumed = run(&at_least(3.0), &other, &grid_actions(&GRID)).unwrap();
        assert_eq!(base.feasible, reassumed.feasible);
        assert_ne!(base.identity, reassumed.identity);
        // A different adjustment gives a different forward law (here 0.5 + 2a); the
        // boundary action leaves the feasible set.
        let mut shifted = linear_forward(&GRID);
        shifted.mean = GRID.iter().map(|a| 0.5 + 2.0 * a).collect();
        let shifted = run(&at_least(3.0), &shifted, &grid_actions(&GRID)).unwrap();
        assert_eq!(shifted.feasible, vec!["dose_1.5", "dose_2"]);
        assert_ne!(base.identity, shifted.identity);
        let moved = run(&at_least(3.5), &linear_forward(&GRID), &grid_actions(&GRID)).unwrap();
        assert_ne!(base.identity, moved.identity);
    }

    #[test]
    fn action_and_forward_row_order_never_change_a_result_or_the_identity() {
        let forward = linear_forward(&GRID);
        let actions = grid_actions(&GRID);
        let base = run(&at_least(3.0), &forward, &actions).unwrap();
        let mut reversed_actions = actions.clone();
        reversed_actions.reverse();
        let mut reversed_forward = forward.clone();
        reversed_forward.points.reverse();
        reversed_forward.mean.reverse();
        reversed_forward.support.reverse();
        let permuted = run(&at_least(3.0), &reversed_forward, &reversed_actions).unwrap();
        assert_eq!(base.identity, permuted.identity);
        assert_eq!(
            (&base.feasible, &base.infeasible, &base.cheapest_feasible, base.enumerated),
            (
                &permuted.feasible,
                &permuted.infeasible,
                &permuted.cheapest_feasible,
                permuted.enumerated
            )
        );
        for outcome in &base.outcomes {
            let twin = permuted.outcomes.iter().find(|o| o.label == outcome.label).unwrap();
            assert_eq!(outcome, twin);
        }
        // Signed zero is one coordinate: -0.0 and 0.0 name the same action.
        let mut zero = actions;
        zero[0].point = vec![-0.0];
        let signed = run(&at_least(3.0), &forward, &zero).unwrap();
        assert_eq!(base.identity, signed.identity);
        assert_eq!(status_of(&signed, "dose_0"), ActionStatus::Infeasible);
    }

    #[test]
    fn a_published_interval_adds_only_a_robustly_feasible_flag() {
        let mut forward = linear_forward(&GRID);
        // A pointwise band of half-width 0.5 around the mean.
        forward.interval = Some(ForwardInterval {
            scope: IntervalScope::Pointwise,
            interpretation: IntervalInterpretation::Confidence,
            level: 0.9,
            lower: forward.mean.iter().map(|m| m - 0.5).collect(),
            upper: forward.mean.iter().map(|m| m + 0.5).collect(),
        });
        let report = run(&at_least(3.0), &forward, &grid_actions(&GRID)).unwrap();
        // Point classification is unchanged by the interval.
        let point_only = run(&at_least(3.0), &linear_forward(&GRID), &grid_actions(&GRID)).unwrap();
        assert_eq!(report.feasible, point_only.feasible);
        // Robust means the lower endpoint 1 + 2a - 0.5 clears 3, i.e. a >= 1.25: doses 1.5 and 2.
        assert_eq!(report.robustly_feasible, vec!["dose_1.5", "dose_2"]);
        let boundary = report.outcomes.iter().find(|o| o.label == "dose_1").unwrap();
        assert_eq!(boundary.robustly_feasible, Some(false));
        assert_eq!(boundary.interval, Some((2.5, 3.5)));
        let infeasible = report.outcomes.iter().find(|o| o.label == "dose_0").unwrap();
        assert_eq!(infeasible.robustly_feasible, None);
        let meta = report.interval.unwrap();
        assert_eq!(
            (meta.scope, meta.interpretation, meta.level),
            (IntervalScope::Pointwise, IntervalInterpretation::Confidence, 0.9)
        );
        // With no interval the flag is absent everywhere, never false.
        assert!(point_only.interval.is_none());
        assert!(point_only.outcomes.iter().all(|o| o.robustly_feasible.is_none()));
        // An at-most goal reads the upper endpoint.
        let upper = InverseQuery::TargetMean { threshold: 3.0, direction: TargetDirection::AtMost };
        let report = run(&upper, &forward, &grid_actions(&GRID)).unwrap();
        assert_eq!(report.robustly_feasible, vec!["dose_0", "dose_0.5"]);
    }

    #[test]
    fn probability_quantile_and_observational_queries_are_typed_refusals() {
        let forward = linear_forward(&GRID);
        let actions = grid_actions(&GRID);
        for (query, detail, code) in [
            (
                InverseQuery::ChanceConstraint { threshold: 3.0, probability: 0.9 },
                "inverse.probability_target",
                "cell_not_licensed",
            ),
            (
                InverseQuery::Quantile { level: 0.9, threshold: 3.0 },
                "inverse.quantile_target",
                "cell_not_licensed",
            ),
            (
                InverseQuery::ObservationalScenarios,
                "inverse.observational_scenarios",
                "cell_not_licensed",
            ),
        ] {
            let refusal = run(&query, &forward, &actions).unwrap_err();
            assert_eq!((refusal.detail, refusal.code), (detail, code));
        }
        // They refuse before reading inputs, so a malformed forward does not mask them.
        let mut broken = forward;
        broken.points.clear();
        let refusal =
            run(&InverseQuery::ChanceConstraint { threshold: 0.0, probability: 0.5 }, &broken, &[])
                .unwrap_err();
        assert_eq!(refusal.detail, "inverse.probability_target");
    }

    #[test]
    fn a_forward_route_that_is_not_a_point_identified_mean_is_refused() {
        let mut derivative = linear_forward(&GRID);
        derivative.mean_response = false;
        let refusal = run(&at_least(3.0), &derivative, &grid_actions(&GRID)).unwrap_err();
        assert_eq!(
            (refusal.detail, refusal.code),
            ("inverse.forward_not_mean_response", "route_not_supported")
        );
        let mut set = linear_forward(&GRID);
        set.point_identified = false;
        let refusal = run(&at_least(3.0), &set, &grid_actions(&GRID)).unwrap_err();
        assert_eq!(refusal.detail, "inverse.forward_not_point_identified");
    }

    #[test]
    fn malformed_inputs_are_invalid_arguments_not_classifications() {
        let forward = linear_forward(&GRID);
        let actions = grid_actions(&GRID);
        let detail = |result: Result<InverseOutcomeReport, InverseOutcomeError>| {
            let error = result.unwrap_err();
            assert_eq!(error.code, "invalid_argument", "{error}");
            error.detail
        };
        assert_eq!(detail(run(&at_least(f64::NAN), &forward, &actions)), "inverse.invalid_target");
        let negative = classify_inverse_outcome(
            &at_least(1.0),
            &forward,
            &actions,
            None,
            -1.0,
            &CancellationToken::new(),
        );
        assert_eq!(detail(negative), "inverse.invalid_target");
        assert_eq!(detail(run(&at_least(1.0), &forward, &[])), "inverse.invalid_action");
        let mut duplicate = actions.clone();
        duplicate[1].label = duplicate[0].label.clone();
        assert_eq!(detail(run(&at_least(1.0), &forward, &duplicate)), "inverse.invalid_action");
        let mut wrong_dimension = actions.clone();
        wrong_dimension[0].point = vec![0.0, 1.0];
        assert_eq!(
            detail(run(&at_least(1.0), &forward, &wrong_dimension)),
            "inverse.invalid_action"
        );
        let mut priced = actions.clone();
        priced[0].cost = f64::NAN;
        assert_eq!(detail(run(&at_least(1.0), &forward, &priced)), "inverse.invalid_action");
        let mut repeated = forward.clone();
        repeated.points[1] = repeated.points[0].clone();
        assert_eq!(detail(run(&at_least(1.0), &repeated, &actions)), "inverse.invalid_forward");
        let mut backwards = forward;
        backwards.interval = Some(ForwardInterval {
            scope: IntervalScope::Pointwise,
            interpretation: IntervalInterpretation::Credible,
            level: 0.95,
            lower: vec![1.0; GRID.len()],
            upper: vec![0.0; GRID.len()],
        });
        assert_eq!(detail(run(&at_least(1.0), &backwards, &actions)), "inverse.invalid_forward");
    }

    #[test]
    fn the_action_grid_is_bounded_and_cancellation_is_never_a_verdict() {
        let forward = linear_forward(&GRID);
        let many: Vec<ActionSpec> =
            (0..=MAX_INVERSE_ACTIONS).map(|i| action(&format!("a{i}"), 0.0, 0.0)).collect();
        let refusal = run(&at_least(1.0), &forward, &many).unwrap_err();
        assert_eq!(
            (refusal.detail, refusal.code),
            ("inverse.bounds_exceeded", "route_not_supported")
        );
        let at_cap: Vec<ActionSpec> =
            (0..MAX_INVERSE_ACTIONS).map(|i| action(&format!("a{i}"), 0.0, 0.0)).collect();
        assert!(run(&at_least(1.0), &forward, &at_cap).is_ok());
        let token = CancellationToken::cancel_after_checks(2);
        let refusal = classify_inverse_outcome(
            &at_least(3.0),
            &forward,
            &grid_actions(&GRID),
            None,
            0.0,
            &token,
        )
        .unwrap_err();
        assert_eq!(
            (refusal.detail, refusal.code),
            ("inverse.cancelled", "transport_budget_cancel")
        );
    }

    #[test]
    fn a_report_is_a_pure_function_of_its_inputs() {
        let forward = linear_forward(&GRID);
        let actions = grid_actions(&GRID);
        let first = run(&at_least(3.0), &forward, &actions).unwrap();
        let again = run(&at_least(3.0), &forward, &actions).unwrap();
        assert_eq!(first, again);
        assert_eq!(first.identity.len(), 64);
        // Editing one cost, one constraint or one support label is a different input.
        let mut costlier = actions.clone();
        costlier[2].cost += 1.0;
        assert_ne!(first.identity, run(&at_least(3.0), &forward, &costlier).unwrap().identity);
        let mut constrained = actions;
        constrained[2].constraints.push(ActionConstraint { name: "c".into(), satisfied: true });
        assert_ne!(first.identity, run(&at_least(3.0), &forward, &constrained).unwrap().identity);
        let mut weaker = forward;
        weaker.support[0] = SupportStatus::WeakOverlap;
        assert_ne!(
            first.identity,
            run(&at_least(3.0), &weaker, &grid_actions(&GRID)).unwrap().identity
        );
    }

    #[test]
    fn forward_execution_identity_is_bound_to_the_inverse_report() {
        let mut forward = linear_forward(&GRID);
        forward.claim_id = Some("claim-a".into());
        forward.program_id = Some("program-a".into());
        forward.data_snapshot_id = Some("snapshot-a".into());
        let actions = grid_actions(&GRID);
        let report = run(&at_least(3.0), &forward, &actions).unwrap();
        assert_eq!(report.forward_claim_id.as_deref(), Some("claim-a"));
        assert_eq!(report.forward_program_id.as_deref(), Some("program-a"));
        assert_eq!(report.forward_data_snapshot_id.as_deref(), Some("snapshot-a"));
        for changed in ["claim", "program", "snapshot"] {
            let mut other = forward.clone();
            match changed {
                "claim" => other.claim_id = Some("claim-b".into()),
                "program" => other.program_id = Some("program-b".into()),
                _ => other.data_snapshot_id = Some("snapshot-b".into()),
            }
            assert_ne!(report.identity, run(&at_least(3.0), &other, &actions).unwrap().identity);
        }
    }

    #[test]
    fn finite_inputs_with_an_unrepresentable_margin_are_not_classified() {
        let mut forward = linear_forward(&[0.0]);
        forward.mean[0] = f64::MAX;
        let report = run(&at_least(-f64::MAX), &forward, &[action("a", 0.0, 0.0)]).unwrap();
        let outcome = &report.outcomes[0];
        assert_eq!(outcome.status, ActionStatus::Unevaluated);
        assert_eq!(outcome.reason, "numerical_margin_overflow");
        assert_eq!(outcome.estimate, Some(f64::MAX));
        assert_eq!(outcome.margin, None);
        assert_eq!(report.enumerated, EnumeratedStatus::Undetermined);
    }

    #[test]
    fn one_enumeration_cannot_choose_between_outcome_horizons() {
        let mut forward = linear_forward(&[0.0, 1.0]);
        forward.dimension = 2;
        forward.points = vec![vec![0.0, 1.0], vec![1.0, 2.0]];
        let actions = [
            ActionSpec { point: vec![0.0, 1.0], ..action("early", 0.0, 0.0) },
            ActionSpec { point: vec![1.0, 2.0], ..action("late", 1.0, 0.0) },
        ];
        let error = run(&at_least(2.0), &forward, &actions).unwrap_err();
        assert_eq!((error.code, error.detail), ("invalid_argument", "inverse.invalid_action"));
        assert!(error.message.contains("one fixed outcome horizon"));
    }
}

//! Marginal sensitivity model (MSM) bounds with a tipping point (2.3 work package B3).
//!
//! Family: Tan (2006) and Zhao, Small and Bhattacharya (2019) marginal sensitivity
//! model for inverse-propensity-weighted effects, specialized to a binary treatment
//! `T`, a discrete adjustment set (finite strata `x`) and exact population-level
//! inputs. There is no sampling in this module: the stratum masses, propensities
//! and outcome laws are supplied as exact quantities, so every number returned is an
//! assumption range or an identified value, never an estimate with an interval.
//!
//! # Definition
//!
//! * **Perturbation scale.** `Lambda >= 1`. The odds of treatment given `(x, Y(t))`
//!   differ from the odds given `x` by a factor in `[1/Lambda, Lambda]`. `Lambda = 1`
//!   is no unmeasured confounding and reproduces the stratified identified value.
//! * **Feasible set.** Write `e(x) = P(T = 1 | x)`, `f1(. | x)` for the outcome law
//!   of the treated and `rho(x, y) = odds(x) / odds(x, y)`. The inverse propensity
//!   weight is `1 / e(x, y) = 1 + (1 / e(x) - 1) rho(x, y)`, so it lies in
//!   `[1 + (1/e - 1)/Lambda, 1 + (1/e - 1) Lambda]`. The identity
//!   `E[1 / e(x, Y(1)) | T = 1, x] = 1 / e(x)` forces `E_{f1}[rho | x] = 1`. The
//!   feasible set per stratum and arm is therefore
//!   `{ rho : 1/Lambda <= rho <= Lambda, E_f[rho] = 1 }`. Equivalently `rho f1` is the
//!   law of `Y(1)` among the untreated, a law whose density ratio to the treated law
//!   lies in `[1/Lambda, Lambda]`. The same set, built on `1 - e(x)` and the control
//!   law `f0`, bounds `Y(0)` among the treated.
//! * **Factor compatibility.** One `Lambda` is shared by every stratum and both
//!   arms. The unobserved laws of the two arms are separate free parameters, so the
//!   bounds of the two arms combine independently.
//! * **Normalization.** The constraint `E_f[rho] = 1` is imposed exactly. This is the
//!   population-exact version; it is not the Hajek (ratio) renormalization of the
//!   sample program, which drops the constraint, enlarges the feasible set and gives
//!   weakly wider, non-sharp population bounds.
//! * **Target.** The average treatment effect `ATE = sum_x P(x) (E[Y(1)|x] - E[Y(0)|x])`
//!   over the supplied strata, which sum to mass one.
//! * **Tipping point.** The smallest `Lambda` in `[1, lambda_max]` at which the
//!   identified-side bound of the ATE reaches a declared threshold: the lower bound
//!   falls to the threshold when the identified ATE is above it, the upper bound rises
//!   to it when the identified ATE is below it. It is found by bisection on a monotone
//!   predicate (the sets nest in `Lambda`); the result reports the tolerance, the
//!   certified bracket `[lower, upper]` (the unresolved region) and `bracketed`.
//!
//! # Sharp bound
//!
//! Fix a stratum and the treated arm: `E[Y(1) | x] = e m + (1 - e) E_{rho f1}[y]` with
//! `m = E_{f1}[y]`, because the treated contribute `e f1` and the untreated
//! `(1 - e) rho f1`. So the bound is a linear program in `rho`:
//! maximize `sum_j q_j rho_j y_j` over `1/Lambda <= rho_j <= Lambda`,
//! `sum_j q_j rho_j = 1`. Write `rho_j = 1/Lambda + s_j` with
//! `0 <= s_j <= Lambda - 1/Lambda` and `sum_j q_j s_j = 1 - 1/Lambda`: a fractional
//! knapsack in which atom `j` has value `y_j` per unit of `q_j s_j`, capacity
//! `q_j (Lambda - 1/Lambda)` and a common budget. The greedy rule (fill the largest
//! `y` first, splitting one atom if needed) is optimal for a fractional knapsack, so
//! the optimum gives `rho = Lambda` on the top probability mass
//! `tau = (1 - 1/Lambda) / (Lambda - 1/Lambda) = 1 / (Lambda + 1)` and `rho = 1/Lambda`
//! elsewhere. With `T_top` the sum of `y q` over that top mass,
//!
//! `U = m + (1 - e)(Lambda - 1/Lambda)(T_top - tau m)`,
//!
//! and the lower bound `L` is the same expression with the bottom mass. Both are
//! attained, hence sharp for the exact constraint set, and nested in `Lambda` because
//! the feasible sets nest (so the bounds widen monotonically and `Lambda = 1` gives
//! `m`). The control arm uses `e` in place of `1 - e`. ATE bounds are
//! `sum_x P(x)(L1 - U0)` and `sum_x P(x)(U1 - L0)`: sharp because the arm and stratum
//! parameters are free independently.
//!
//! Assumption ranges, identified values and sampling intervals stay distinct: the
//! result's `identified` is the `Lambda = 1` value, its grid bounds are an assumption
//! range, and `uncertainty` withholds any sampling interval
//! (`sampling interval: not reported`). A requested sampling composition refuses with
//! `msm_sensitivity.composition_not_licensed`. Calibration is unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::fmt;

use antecedent_core::reason_code;

use crate::joint_mechanism_sensitivity::{TippingBracket, TippingStatus};

/// Hard cap on strata.
pub const MSM_MAX_STRATA: usize = 4096;
/// Hard cap on atoms of one outcome law.
pub const MSM_MAX_OUTCOME_ATOMS: usize = 256;
/// Hard cap on grid points.
pub const MSM_MAX_GRID_POINTS: usize = 1024;
/// Largest admitted `Lambda`.
pub const MSM_MAX_LAMBDA: f64 = 1.0e3;
/// Smallest admitted bisection tolerance.
pub const MSM_MIN_TOLERANCE: f64 = 1e-12;
/// Largest admitted bisection tolerance.
pub const MSM_MAX_TOLERANCE: f64 = 1e-3;
/// Largest admitted `|sum - 1|` of a stratum-mass vector or an outcome law.
const MSM_UNIT_MASS_TOLERANCE: f64 = 1e-9;
/// Safety cap on bisection iterations (the tolerance bound makes it unreachable).
const MSM_MAX_BISECTIONS: usize = 200;
/// Family identity.
pub const MSM_FAMILY: &str = "marginal_sensitivity_model_ate";
/// Perturbation scale.
pub const MSM_PERTURBATION_SCALE: &str = "lambda_odds_ratio_bound";
/// Normalization statement.
pub const MSM_NORMALIZATION: &str =
    "exact population constraint E[rho]=1 per stratum and arm; not Hajek renormalization";
/// Target statement.
pub const MSM_TARGET: &str = "average_treatment_effect_over_supplied_strata";
/// Optimization method recorded in every result.
pub const MSM_METHOD: &str = "closed-form sharp bounds of the Tan/Zhao-Small-Bhattacharya marginal sensitivity model on finite strata (fractional-knapsack extremum over the top/bottom 1/(Lambda+1) outcome mass); bisection tipping point";
/// Interpretation of the reported range.
pub const MSM_INTERPRETATION: &str = "assumption range over the declared marginal sensitivity set; not a confidence interval and not a sampling interval";
/// The only inference claim the range carries.
pub const MSM_ASSUMPTION_RANGE_CLAIM: &str = "assumption_range";
/// The statement of the withheld sampling interval.
pub const MSM_SAMPLING_STATEMENT: &str = "sampling interval: not reported";
/// Detail of the withheld sampling interval.
pub const MSM_INTERVAL_WITHHELD: &str = "msm_sensitivity.interval_withheld";

/// A finite-support outcome law (values and probabilities, any order).
#[derive(Clone, Debug, PartialEq)]
pub struct MsmOutcomeLaw {
    /// Finite outcome values.
    pub values: Vec<f64>,
    /// Probabilities, one per value, summing to one.
    pub probabilities: Vec<f64>,
}

/// One stratum of the adjustment set.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmStratum {
    /// Population mass `P(x)` of the stratum.
    pub mass: f64,
    /// Propensity `e(x) = P(T = 1 | x)`, strictly inside `(0, 1)`.
    pub propensity: f64,
    /// Outcome law of the treated in the stratum.
    pub treated: MsmOutcomeLaw,
    /// Outcome law of the untreated in the stratum.
    pub control: MsmOutcomeLaw,
}

/// Analysis settings.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmSensitivitySpec {
    /// Largest `Lambda` of the grid (and of the tipping search); above one.
    pub lambda_max: f64,
    /// Grid points from `1` to `lambda_max`, equally spaced (2 to 1024).
    pub grid_points: usize,
    /// Optional decision threshold of the ATE for the tipping point.
    pub decision_threshold: Option<f64>,
    /// Width at which the tipping bracket is accepted.
    pub tolerance: f64,
    /// A requested sampling composition; always refused.
    pub sampling_composition: Option<String>,
}

impl MsmSensitivitySpec {
    /// A spec up to `lambda_max` with a 17-point grid, no threshold and tolerance `1e-10`.
    #[must_use]
    pub fn new(lambda_max: f64) -> Self {
        Self {
            lambda_max,
            grid_points: 17,
            decision_threshold: None,
            tolerance: 1e-10,
            sampling_composition: None,
        }
    }

    /// The same spec with a decision threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.decision_threshold = Some(threshold);
        self
    }
}

/// Typed refusal of the MSM contract.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmSensitivityError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced `msm_sensitivity.*` detail.
    pub detail: &'static str,
    /// Human-readable context.
    pub message: String,
}

impl MsmSensitivityError {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for MsmSensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for MsmSensitivityError {}

fn refuse(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
) -> MsmSensitivityError {
    MsmSensitivityError { code, detail, message: message.into() }
}

/// ATE bounds at one `Lambda`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsmPoint {
    /// The perturbation `Lambda`.
    pub lambda: f64,
    /// Sharp lower bound of the ATE.
    pub lower: f64,
    /// Sharp upper bound of the ATE.
    pub upper: f64,
}

/// Which bound the tipping search follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MsmTippingDirection {
    /// The identified ATE is at or above the threshold; the lower bound falls to it.
    LowerBoundFalls,
    /// The identified ATE is below the threshold; the upper bound rises to it.
    UpperBoundRises,
}

impl MsmTippingDirection {
    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LowerBoundFalls => "lower_bound_falls",
            Self::UpperBoundRises => "upper_bound_rises",
        }
    }
}

/// The tipping point of the ATE bound against a threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsmTipping {
    /// The declared threshold.
    pub threshold: f64,
    /// Which bound is followed.
    pub direction: MsmTippingDirection,
    /// How the search ended (`NotReachedInBox` when `lambda_max` does not reach it).
    pub status: TippingStatus,
    /// Whether the first crossing lies inside the certified bracket (the unresolved
    /// region is `bracket`); false at the origin or when not reached.
    pub bracketed: bool,
    /// Certified bracket in `Lambda`: not reached at `lower`, reached at `upper`.
    pub bracket: Option<TippingBracket>,
    /// Bisection tolerance in `Lambda`.
    pub tolerance: f64,
}

/// Withheld sampling status, distinct from the assumption range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmSensitivityUncertainty {
    /// [`MSM_SAMPLING_STATEMENT`].
    pub sampling_interval: &'static str,
    /// Registered reason code of the withheld interval.
    pub reason_code: &'static str,
    /// [`MSM_INTERVAL_WITHHELD`].
    pub detail: &'static str,
}

impl MsmSensitivityUncertainty {
    /// The withheld status every result carries.
    #[must_use]
    pub const fn withheld() -> Self {
        Self {
            sampling_interval: MSM_SAMPLING_STATEMENT,
            reason_code: reason_code!("cell_not_licensed"),
            detail: MSM_INTERVAL_WITHHELD,
        }
    }
}

/// Auditable MSM sensitivity of the stratified ATE.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmSensitivityResult {
    /// [`MSM_FAMILY`].
    pub family: &'static str,
    /// [`MSM_PERTURBATION_SCALE`].
    pub perturbation_scale: &'static str,
    /// [`MSM_NORMALIZATION`].
    pub normalization: &'static str,
    /// [`MSM_TARGET`].
    pub target: &'static str,
    /// Identified stratified ATE (the `Lambda = 1` value).
    pub identified: f64,
    /// Largest `Lambda`.
    pub lambda_max: f64,
    /// Bounds on the `Lambda` grid, ascending, first `1`, last `lambda_max`.
    pub grid: Vec<MsmPoint>,
    /// Threshold used for the tipping point, when declared.
    pub decision_threshold: Option<f64>,
    /// Tipping point, when a threshold was declared.
    pub tipping: Option<MsmTipping>,
    /// Bisection tolerance.
    pub tolerance: f64,
    /// Number of strata.
    pub strata: usize,
    /// [`MSM_METHOD`].
    pub method: &'static str,
    /// [`MSM_INTERPRETATION`].
    pub interpretation: &'static str,
    /// Always [`MSM_ASSUMPTION_RANGE_CLAIM`].
    pub inference_claim: &'static str,
    /// Sampling status (withheld).
    pub uncertainty: MsmSensitivityUncertainty,
}

/// One arm of one stratum, prepared: atoms ascending by value.
struct Arm {
    atoms: Vec<(f64, f64)>,
    mean: f64,
    /// Fraction of the stratum whose potential outcome in this arm is unobserved.
    hidden: f64,
}

struct Prepared {
    mass: f64,
    treated: Arm,
    control: Arm,
}

fn invalid_law(message: &str) -> MsmSensitivityError {
    refuse(reason_code!("invalid_argument"), "msm_sensitivity.outcome_law", message)
}

fn prepare_arm(law: &MsmOutcomeLaw, hidden: f64) -> Result<Arm, MsmSensitivityError> {
    let n = law.values.len();
    if n == 0 || n > MSM_MAX_OUTCOME_ATOMS || law.probabilities.len() != n {
        return Err(invalid_law("an outcome law needs 1 to 256 values with one probability each"));
    }
    if law.values.iter().any(|v| !v.is_finite())
        || law.probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err(invalid_law("outcome values must be finite and probabilities nonnegative"));
    }
    let total = law.probabilities.iter().sum::<f64>();
    if (total - 1.0).abs() > MSM_UNIT_MASS_TOLERANCE {
        return Err(invalid_law(&format!("outcome probabilities sum to {total}, not 1")));
    }
    let mut atoms: Vec<(f64, f64)> =
        law.values.iter().copied().zip(law.probabilities.iter().copied()).collect();
    atoms.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mean = atoms.iter().map(|(y, q)| y * q).sum::<f64>();
    Ok(Arm { atoms, mean, hidden })
}

fn prepare(strata: &[MsmStratum]) -> Result<Vec<Prepared>, MsmSensitivityError> {
    if strata.is_empty() || strata.len() > MSM_MAX_STRATA {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "msm_sensitivity.bounds_exceeded",
            format!("{} strata declared; 1 to {MSM_MAX_STRATA} are admitted", strata.len()),
        ));
    }
    let mut total = 0.0;
    let mut prepared = Vec::with_capacity(strata.len());
    for stratum in strata {
        let e = stratum.propensity;
        if !e.is_finite() || e <= 0.0 || e >= 1.0 {
            return Err(refuse(
                reason_code!("route_not_supported"),
                "msm_sensitivity.positivity",
                format!("propensity {e} is not strictly inside (0, 1); the weights are undefined"),
            ));
        }
        if !stratum.mass.is_finite() || stratum.mass <= 0.0 {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "msm_sensitivity.stratum_mass",
                format!("stratum mass {} must be positive and finite", stratum.mass),
            ));
        }
        total += stratum.mass;
        prepared.push(Prepared {
            mass: stratum.mass,
            treated: prepare_arm(&stratum.treated, 1.0 - e)?,
            control: prepare_arm(&stratum.control, e)?,
        });
    }
    if (total - 1.0).abs() > MSM_UNIT_MASS_TOLERANCE {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "msm_sensitivity.stratum_mass",
            format!("stratum masses sum to {total}, not 1"),
        ));
    }
    Ok(prepared)
}

/// Sum of `y q` over the first `tau` probability mass of `atoms` (one atom split).
fn mass_sum<'a>(atoms: impl Iterator<Item = &'a (f64, f64)>, tau: f64) -> f64 {
    let mut remaining = tau;
    let mut sum = 0.0;
    for &(value, probability) in atoms {
        if remaining <= 0.0 {
            break;
        }
        let take = probability.min(remaining);
        sum += take * value;
        remaining -= take;
    }
    sum
}

/// Sharp `(lower, upper)` of `E[Y(arm) | x]` at `lambda`.
fn arm_bounds(arm: &Arm, lambda: f64) -> (f64, f64) {
    let spread = lambda - 1.0 / lambda;
    let tau = 1.0 / (lambda + 1.0);
    let top = mass_sum(arm.atoms.iter().rev(), tau);
    let bottom = mass_sum(arm.atoms.iter(), tau);
    let scale = arm.hidden * spread;
    (arm.mean + scale * (bottom - tau * arm.mean), arm.mean + scale * (top - tau * arm.mean))
}

/// Sharp ATE `(lower, upper)` at `lambda`.
fn ate_bounds(prepared: &[Prepared], lambda: f64) -> (f64, f64) {
    let (mut lower, mut upper) = (0.0, 0.0);
    for stratum in prepared {
        let (low1, up1) = arm_bounds(&stratum.treated, lambda);
        let (low0, up0) = arm_bounds(&stratum.control, lambda);
        lower += stratum.mass * (low1 - up0);
        upper += stratum.mass * (up1 - low0);
    }
    (lower, upper)
}

fn check_lambda(lambda: f64) -> Result<(), MsmSensitivityError> {
    if !lambda.is_finite() || lambda < 1.0 {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "msm_sensitivity.lambda_below_one",
            format!("Lambda {lambda} must be finite and at least 1"),
        ));
    }
    if lambda > MSM_MAX_LAMBDA {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "msm_sensitivity.bounds_exceeded",
            format!("Lambda {lambda} exceeds {MSM_MAX_LAMBDA}"),
        ));
    }
    Ok(())
}

fn validate_spec(spec: &MsmSensitivitySpec) -> Result<(), MsmSensitivityError> {
    if let Some(method) = &spec.sampling_composition {
        return Err(refuse(
            reason_code!("cell_not_licensed"),
            "msm_sensitivity.composition_not_licensed",
            format!(
                "composing a sampling interval with the assumption range by `{method}` is not licensed"
            ),
        ));
    }
    check_lambda(spec.lambda_max)?;
    if spec.lambda_max <= 1.0 {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "msm_sensitivity.lambda_range_empty",
            "lambda_max must exceed 1 so the grid spans an assumption range",
        ));
    }
    if !(2..=MSM_MAX_GRID_POINTS).contains(&spec.grid_points) {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "msm_sensitivity.bounds_exceeded",
            format!("{} grid points; 2 to {MSM_MAX_GRID_POINTS} are admitted", spec.grid_points),
        ));
    }
    if spec.decision_threshold.is_some_and(|t| !t.is_finite()) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "msm_sensitivity.invalid_threshold",
            "the decision threshold must be finite",
        ));
    }
    if !spec.tolerance.is_finite()
        || !(MSM_MIN_TOLERANCE..=MSM_MAX_TOLERANCE).contains(&spec.tolerance)
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "msm_sensitivity.invalid_tolerance",
            format!("tolerance {} is outside [1e-12, 1e-3]", spec.tolerance),
        ));
    }
    Ok(())
}

/// Sharp ATE bounds of the stratified MSM at one `Lambda`.
///
/// # Errors
///
/// A typed [`MsmSensitivityError`]: `Lambda` below one or not finite
/// (`msm_sensitivity.lambda_below_one`), a propensity outside `(0, 1)`
/// (`msm_sensitivity.positivity`), malformed masses or outcome laws, or exceeded bounds.
pub fn msm_ate_bounds_at(
    strata: &[MsmStratum],
    lambda: f64,
) -> Result<MsmPoint, MsmSensitivityError> {
    check_lambda(lambda)?;
    let prepared = prepare(strata)?;
    let (lower, upper) = ate_bounds(&prepared, lambda);
    Ok(MsmPoint { lambda, lower, upper })
}

fn grid_of(spec: &MsmSensitivitySpec) -> Vec<f64> {
    let last = spec.grid_points - 1;
    (0..spec.grid_points)
        .map(|j| match j {
            0 => 1.0,
            _ if j == last => spec.lambda_max,
            _ => 1.0 + (spec.lambda_max - 1.0) * j as f64 / last as f64,
        })
        .collect()
}

fn tipping_of(
    prepared: &[Prepared],
    identified: f64,
    threshold: f64,
    spec: &MsmSensitivitySpec,
) -> MsmTipping {
    let downward = identified >= threshold;
    let reached = |lambda: f64| {
        let (lower, upper) = ate_bounds(prepared, lambda);
        if downward { lower <= threshold } else { upper >= threshold }
    };
    let direction = if downward {
        MsmTippingDirection::LowerBoundFalls
    } else {
        MsmTippingDirection::UpperBoundRises
    };
    let done = |status: TippingStatus, bracket: Option<TippingBracket>| MsmTipping {
        threshold,
        direction,
        status,
        bracketed: status == TippingStatus::Bracketed,
        bracket,
        tolerance: spec.tolerance,
    };
    if reached(1.0) {
        return done(
            TippingStatus::ReachedAtOrigin,
            Some(TippingBracket { lower: 1.0, upper: 1.0, iterations: 0 }),
        );
    }
    if !reached(spec.lambda_max) {
        return done(TippingStatus::NotReachedInBox, None);
    }
    let (mut lower, mut upper) = (1.0_f64, spec.lambda_max);
    let mut iterations = 0;
    while upper - lower > spec.tolerance && iterations < MSM_MAX_BISECTIONS {
        iterations += 1;
        let middle = lower + (upper - lower) / 2.0;
        if middle <= lower || middle >= upper {
            break;
        }
        if reached(middle) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    done(TippingStatus::Bracketed, Some(TippingBracket { lower, upper, iterations }))
}

/// Marginal sensitivity model bounds of the stratified ATE over a `Lambda` grid, with
/// an optional tipping point against a declared threshold.
///
/// The strata (exact masses, propensities and outcome laws) are validated first;
/// nothing is sampled. Every grid point is the sharp assumption range at that
/// `Lambda`; the ranges nest, so the bounds widen with `Lambda`. A declared sampling
/// composition refuses.
///
/// # Errors
///
/// A typed [`MsmSensitivityError`] carrying its reason code and `msm_sensitivity.*`
/// detail: `composition_not_licensed`, `lambda_below_one`, `lambda_range_empty`,
/// `positivity`, `stratum_mass`, `outcome_law`, `invalid_threshold`,
/// `invalid_tolerance` or `bounds_exceeded`.
pub fn msm_ate_sensitivity(
    strata: &[MsmStratum],
    spec: &MsmSensitivitySpec,
) -> Result<MsmSensitivityResult, MsmSensitivityError> {
    validate_spec(spec)?;
    let prepared = prepare(strata)?;
    let (identified, _) = ate_bounds(&prepared, 1.0);
    let grid = grid_of(spec)
        .into_iter()
        .map(|lambda| {
            let (lower, upper) = ate_bounds(&prepared, lambda);
            MsmPoint { lambda, lower, upper }
        })
        .collect();
    let tipping =
        spec.decision_threshold.map(|threshold| tipping_of(&prepared, identified, threshold, spec));
    Ok(MsmSensitivityResult {
        family: MSM_FAMILY,
        perturbation_scale: MSM_PERTURBATION_SCALE,
        normalization: MSM_NORMALIZATION,
        target: MSM_TARGET,
        identified,
        lambda_max: spec.lambda_max,
        grid,
        decision_threshold: spec.decision_threshold,
        tipping,
        tolerance: spec.tolerance,
        strata: strata.len(),
        method: MSM_METHOD,
        interpretation: MSM_INTERPRETATION,
        inference_claim: MSM_ASSUMPTION_RANGE_CLAIM,
        uncertainty: MsmSensitivityUncertainty::withheld(),
    })
}

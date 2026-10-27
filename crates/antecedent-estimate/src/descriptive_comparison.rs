//! Descriptive raw-versus-adjusted comparison and declared reporting-scale transforms.
//!
//! # What this is, and is not
//!
//! Two small, separate computations that report on estimates the caller already holds:
//!
//! * **Raw versus adjusted.** The unadjusted contrast `mean(Y | T = 1) - mean(Y | T = 0)`
//!   over the complete rows is set beside an adjusted estimate of the *same* estimand
//!   coding (active level 1, control level 0, difference of means, all-observed
//!   population), and the descriptive gap `raw - adjusted` is reported. The gap is
//!   **descriptive, not a causal decomposition**: it mixes confounding, the adjustment
//!   model's functional form, estimator differences and sampling noise. It is never
//!   split over adjustment columns; a leave-one-out or other per-column attribution is
//!   refused ([`refuse_column_attribution`]) because the adjusted estimate is not an
//!   additive function of the columns.
//! * **Reporting-scale transform.** From a mean-outcome pair `(mu1, mu0)` (a licensed
//!   result's arm means) a declared transform to the risk difference, log risk ratio or
//!   log odds ratio, by the delta method. For `theta = g(mu)` the first-order
//!   covariance of a family `g_1..g_k` is `J Sigma J'`, with `J` the `k x 2` Jacobian and
//!   `Sigma` the `2 x 2` covariance of the pair. It exists **only** when the source
//!   carries the joint covariance of the pair; without it the transformed point is
//!   returned and the covariance is typed unavailable. A covariance of independent
//!   marginals is never assumed for arms that share data, so a missing covariance is
//!   never filled with zero. The only covariance this module forms itself is the
//!   independent-groups one of the raw arm means ([`RawContrast::mean_pair`]).
//!
//! # Inference claim: point only
//!
//! The delta-method covariance is exact algebra of a first-order approximation. A Wald
//! interval built from it would rest on asymptotic normality of the pair and on the
//! linearization being adequate near the boundary (a risk ratio of a rare event, an
//! odds ratio near 0 or 1) that no coverage record measures, so an interval is refused
//! ([`refuse_transform_interval`]); the standard error is reported only as the square
//! root of the covariance diagonal. The raw-minus-adjusted gap carries no interval
//! either: the covariance between the raw and the adjusted estimate is not part of what
//! an estimate carries, and neither is it assumed zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
#[cfg(test)]
use antecedent_core::device_reservation::{
    ComputeSelection, CpuAdmissionError, DeviceReservationPool, FoldAdmission, LearnerAdmission,
    admit_fold, nested_thread_request, retain_scientific_f64, scientific_tolerance, select_compute,
    within_declared_tolerance,
};

use crate::error::EstimationError;
use crate::joint_if::JointCovariance;
use crate::util::refuse;

/// Tolerance for treatment codes and covariance symmetry, matching the propensity
/// preparer's exact-binary check.
const CODE_TOLERANCE: f64 = 1e-12;

/// Rows between cancellation polls in the raw-contrast pass, so a long input stops at a
/// bounded granularity without a poll per row.
const POLL_EVERY_ROWS: usize = 4096;

/// Scratch bytes one fold and one learner each draw from the shared admission pool.
#[cfg(test)]
const FOLD_SCRATCH_BYTES: u64 = 8;

/// A diagnostic or covariance the route does not carry, with the registered code and the
/// detail naming why. Absent is explicit; nothing here is a zero or a fabricated value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Unavailable {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail literal (`<namespace>.<snake_case>`).
    pub detail: &'static str,
    /// What is missing and what the caller can supply or do instead.
    pub reason: &'static str,
}

/// A quantity that is either computed or typed unavailable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Availability<T> {
    /// The quantity, computed from what the source carries.
    Available(T),
    /// The source does not carry what the quantity needs.
    Unavailable(Unavailable),
}

impl<T> Availability<T> {
    /// The computed value, if there is one.
    #[must_use]
    pub const fn available(&self) -> Option<&T> {
        match self {
            Self::Available(value) => Some(value),
            Self::Unavailable(_) => None,
        }
    }

    /// The typed reason, if the quantity is unavailable.
    #[must_use]
    pub const fn unavailable(&self) -> Option<&Unavailable> {
        match self {
            Self::Available(_) => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }
}

fn invalid_data(message: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "descriptive_comparison.invalid_data",
        message,
    )
}

fn cancelled() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "descriptive_comparison.cancelled",
        "the comparison was cancelled before every row was read; no contrast is reported and \
         the stop is not a verdict on the data",
    )
}

fn arm_too_small() -> Unavailable {
    Unavailable {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "descriptive_comparison.arm_too_small",
        reason: "an arm with fewer than two rows has no sample variance, so the raw standard \
                 error and the independent-groups covariance of the arm means are unavailable",
    }
}

/// The reporting scales a mean-outcome pair can be put on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportingScale {
    /// `mu1 - mu0` for any outcome mean.
    MeanDifference,
    /// `mu1 - mu0` with both means probabilities in `[0, 1]` (a binary outcome).
    RiskDifference,
    /// `ln(mu1 / mu0)` with both means strictly positive.
    LogRiskRatio,
    /// `ln(mu1 / (1 - mu1)) - ln(mu0 / (1 - mu0))` with both means in `(0, 1)`.
    LogOddsRatio,
}

impl ReportingScale {
    /// Wire name of the scale.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MeanDifference => "mean_difference",
            Self::RiskDifference => "risk_difference",
            Self::LogRiskRatio => "log_risk_ratio",
            Self::LogOddsRatio => "log_odds_ratio",
        }
    }

    /// Parse a wire name.
    ///
    /// # Errors
    /// `route_not_supported` (`descriptive_comparison.scale_not_supported`) for any other name.
    pub fn parse(name: &str) -> Result<Self, EstimationError> {
        [Self::MeanDifference, Self::RiskDifference, Self::LogRiskRatio, Self::LogOddsRatio]
            .into_iter()
            .find(|scale| scale.name() == name)
            .ok_or_else(|| {
                refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "descriptive_comparison.scale_not_supported",
                    &format!(
                        "reporting scale '{name}' is not declared; the scales are mean_difference, \
                         risk_difference, log_risk_ratio and log_odds_ratio"
                    ),
                )
            })
    }

    /// Value and gradient of the scale at `(mu1, mu0)`.
    fn evaluate(self, active: f64, control: f64) -> Result<(f64, [f64; 2]), EstimationError> {
        let domain = |ok: bool, needs: &str| {
            if ok {
                Ok(())
            } else {
                Err(refuse(
                    antecedent_core::reason_code!("invalid_argument"),
                    "descriptive_comparison.transform_domain",
                    &format!(
                        "{} is undefined at the means ({active}, {control}): it needs {needs}",
                        self.name()
                    ),
                ))
            }
        };
        match self {
            Self::MeanDifference => Ok((active - control, [1.0, -1.0])),
            Self::RiskDifference => {
                domain(
                    (0.0..=1.0).contains(&active) && (0.0..=1.0).contains(&control),
                    "both means to be probabilities in [0, 1]",
                )?;
                Ok((active - control, [1.0, -1.0]))
            }
            Self::LogRiskRatio => {
                domain(active > 0.0 && control > 0.0, "both means strictly positive")?;
                Ok((active.ln() - control.ln(), [1.0 / active, -1.0 / control]))
            }
            Self::LogOddsRatio => {
                domain(
                    active > 0.0 && active < 1.0 && control > 0.0 && control < 1.0,
                    "both means strictly inside (0, 1)",
                )?;
                let logit = |p: f64| (p / (1.0 - p)).ln();
                Ok((
                    logit(active) - logit(control),
                    [1.0 / (active * (1.0 - active)), -1.0 / (control * (1.0 - control))],
                ))
            }
        }
    }
}

/// A pair of arm means with the covariance the source carries, ordered
/// `[var(active), cov(active, control), var(control)]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MeanPair {
    /// Mean outcome at the active level.
    pub active: f64,
    /// Mean outcome at the control level.
    pub control: f64,
    /// Joint covariance of the pair, or the typed reason the source does not carry it.
    pub covariance: Availability<[f64; 3]>,
}

impl MeanPair {
    /// A pair with the covariance as the source published it; `None` is typed unavailable.
    #[must_use]
    pub fn new(active: f64, control: f64, covariance: Option<[f64; 3]>) -> Self {
        let covariance = covariance.map_or_else(
            || {
                Availability::Unavailable(Unavailable {
                    code: antecedent_core::reason_code!("required_option_missing"),
                    detail: "descriptive_comparison.joint_covariance_unavailable",
                    reason:
                        "the source estimate does not carry the joint covariance of the two arm \
                             means; the transformed points are reported and the delta-method \
                             covariance is unavailable (independence is never assumed for arms \
                             that share data)",
                })
            },
            Availability::Available,
        );
        Self { active, control, covariance }
    }

    /// The pair at indices `(i_active, i_control)` of a joint covariance the source carries.
    ///
    /// # Errors
    /// `invalid_argument` (`descriptive_comparison.invalid_covariance`) for equal or
    /// out-of-range indices or a matrix that is not symmetric.
    pub fn from_joint_covariance(
        active: f64,
        control: f64,
        joint: &JointCovariance,
        i_active: usize,
        i_control: usize,
    ) -> Result<Self, EstimationError> {
        if i_active == i_control || i_active >= joint.dim || i_control >= joint.dim {
            return Err(invalid_covariance(
                "the two arm indices must be distinct entries of the joint covariance",
            ));
        }
        let cross = joint.get(i_active, i_control);
        let scale = joint.get(i_active, i_active).abs().max(joint.get(i_control, i_control).abs());
        if (cross - joint.get(i_control, i_active)).abs() > 1e-9 * scale.max(1e-300) {
            return Err(invalid_covariance("the joint covariance is not symmetric"));
        }
        Ok(Self::new(
            active,
            control,
            Some([joint.get(i_active, i_active), cross, joint.get(i_control, i_control)]),
        ))
    }
}

fn invalid_covariance(message: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "descriptive_comparison.invalid_covariance",
        message,
    )
}

/// One transformed point with its gradient with respect to `(mu1, mu0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReportingContrast {
    /// The scale the value is on.
    pub scale: ReportingScale,
    /// The transformed point `g(mu1, mu0)`.
    pub value: f64,
    /// `(dg/dmu1, dg/dmu0)`, the row of the Jacobian.
    pub gradient: [f64; 2],
}

/// A family of transformed points and their first-order joint covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportingTransform {
    /// The declared transforms, in the order requested.
    pub contrasts: Vec<ReportingContrast>,
    /// Column-major `k x k` delta-method covariance `J Sigma J'`, or the typed reason the
    /// source carries no joint covariance of the pair.
    pub covariance: Availability<Vec<f64>>,
}

impl ReportingTransform {
    /// Delta-method standard error of contrast `i`: the root of the covariance diagonal.
    #[must_use]
    pub fn se(&self, i: usize) -> Option<f64> {
        let k = self.contrasts.len();
        self.covariance.available().filter(|_| i < k).map(|c| c[i * k + i].max(0.0).sqrt())
    }
}

/// Put a mean-outcome pair on each declared reporting scale, with the delta-method
/// covariance of the family when the pair carries its joint covariance.
///
/// # Errors
/// `invalid_argument`: no scale requested, a non-finite mean, a mean outside a scale's
/// domain (`descriptive_comparison.transform_domain`), or a covariance that is not finite,
/// has a negative variance or is not positive semidefinite
/// (`descriptive_comparison.invalid_covariance`).
pub fn transform_mean_pair(
    pair: &MeanPair,
    scales: &[ReportingScale],
) -> Result<ReportingTransform, EstimationError> {
    if scales.is_empty() {
        return Err(invalid_data("at least one reporting scale must be declared"));
    }
    if !(pair.active.is_finite() && pair.control.is_finite()) {
        return Err(invalid_data("the arm means must be finite"));
    }
    let mut contrasts = Vec::with_capacity(scales.len());
    for &scale in scales {
        let (value, gradient) = scale.evaluate(pair.active, pair.control)?;
        contrasts.push(ReportingContrast { scale, value, gradient });
    }
    let covariance = match &pair.covariance {
        Availability::Unavailable(reason) => Availability::Unavailable(*reason),
        Availability::Available([var_a, cross, var_c]) => {
            let finite = var_a.is_finite() && var_c.is_finite() && cross.is_finite();
            if !finite || *var_a < 0.0 || *var_c < 0.0 {
                return Err(invalid_covariance(
                    "the covariance must be finite with non-negative variances",
                ));
            }
            if cross * cross > var_a * var_c * (1.0 + 1e-9) {
                return Err(invalid_covariance("the covariance is not positive semidefinite"));
            }
            let k = contrasts.len();
            let mut values = vec![0.0; k * k];
            for (i, gi) in contrasts.iter().enumerate() {
                for (j, gj) in contrasts.iter().enumerate() {
                    values[j * k + i] = gi.gradient[0] * gj.gradient[0] * var_a
                        + (gi.gradient[0] * gj.gradient[1] + gi.gradient[1] * gj.gradient[0])
                            * cross
                        + gi.gradient[1] * gj.gradient[1] * var_c;
                }
            }
            Availability::Available(values)
        }
    };
    Ok(ReportingTransform { contrasts, covariance })
}

/// The refusal of the closed interval route: points and covariance are kept.
#[must_use]
pub fn refuse_transform_interval() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cell_not_licensed"),
        "descriptive_comparison.interval_withheld",
        "no interval is reported on a transformed scale: a delta-method Wald interval rests on \
         asymptotic normality of the arm means and an adequate linearization near the boundary \
         of the scale, which no coverage record measures; the transformed points and the \
         first-order covariance are retained",
    )
}

/// The refusal of any attribution of the raw-versus-adjusted gap to adjustment columns.
#[must_use]
pub fn refuse_column_attribution() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("effect_not_identified"),
        "descriptive_comparison.column_attribution_not_identified",
        "the raw-minus-adjusted gap is descriptive and is not attributed to individual \
         adjustment columns: the adjusted estimate is not an additive function of the columns \
         (a leave-one-out difference depends on the order, the remaining columns and the \
         functional form), so a per-column share is not a causal quantity",
    )
}

/// Count, mean and sum of squared deviations of one arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmSummary {
    /// Complete rows in the arm.
    pub n: usize,
    /// Arm mean of the outcome.
    pub mean: f64,
    /// Sum of squared deviations from the arm mean.
    pub m2: f64,
}

impl ArmSummary {
    /// `s^2 / n`, the variance of the arm mean under independent sampling; `None` below two rows.
    fn variance_of_mean(&self) -> Option<f64> {
        (self.n >= 2).then(|| self.m2 / (self.n - 1) as f64 / self.n as f64)
    }

    fn of(values: &[f64]) -> Self {
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let m2 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
        Self { n, mean, m2 }
    }
}

/// The unadjusted contrast of the two treatment arms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawContrast {
    /// The active (treatment = 1) arm.
    pub active: ArmSummary,
    /// The control (treatment = 0) arm.
    pub control: ArmSummary,
    /// `mean(active) - mean(control)`.
    pub difference: f64,
    /// Welch standard error `sqrt(s1^2/n1 + s0^2/n0)` (independent-groups, descriptive), or
    /// the typed reason an arm is too small.
    pub se: Availability<f64>,
}

impl RawContrast {
    /// The raw arm means as a pair, with the independent-groups covariance
    /// `diag(s1^2/n1, s0^2/n0)` (zero cross term: two disjoint groups of one sample).
    #[must_use]
    pub fn mean_pair(&self) -> MeanPair {
        let covariance = match (self.active.variance_of_mean(), self.control.variance_of_mean()) {
            (Some(va), Some(vc)) => Availability::Available([va, 0.0, vc]),
            _ => Availability::Unavailable(arm_too_small()),
        };
        MeanPair { active: self.active.mean, control: self.control.mean, covariance }
    }
}

/// The unadjusted contrast `mean(Y | T = 1) - mean(Y | T = 0)` of complete rows.
///
/// # Errors
/// `invalid_argument` (`descriptive_comparison.invalid_data`) for empty, unequal-length,
/// non-finite or non-0/1 treatment input; `arm_not_populated`
/// (`descriptive_comparison.arm_not_populated`) when an arm has no rows;
/// `cancelled_no_claim` (`descriptive_comparison.cancelled`) when `ctx` is cancelled (observed
/// at the first row, every 4096 rows and once after the pass), with no contrast reported.
pub fn raw_contrast(
    outcome: &[f64],
    treatment: &[f64],
    ctx: &ExecutionContext,
) -> Result<RawContrast, EstimationError> {
    if outcome.is_empty() || outcome.len() != treatment.len() {
        return Err(invalid_data("outcome and treatment must be non-empty and of equal length"));
    }
    let (mut active, mut control) = (Vec::new(), Vec::new());
    for (i, (&y, &t)) in outcome.iter().zip(treatment).enumerate() {
        if i % POLL_EVERY_ROWS == 0 && ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        if !y.is_finite() {
            return Err(invalid_data("the outcome has a missing or non-finite value"));
        }
        if t.abs() <= CODE_TOLERANCE {
            control.push(y);
        } else if (t - 1.0).abs() <= CODE_TOLERANCE {
            active.push(y);
        } else {
            return Err(invalid_data("the treatment must be coded exactly 0 or 1"));
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    if active.is_empty() || control.is_empty() {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "descriptive_comparison.arm_not_populated",
            &format!(
                "a treatment arm has no rows ({} active, {} control), so there is no raw contrast",
                active.len(),
                control.len()
            ),
        ));
    }
    let (active, control) = (ArmSummary::of(&active), ArmSummary::of(&control));
    let se = match (active.variance_of_mean(), control.variance_of_mean()) {
        (Some(va), Some(vc)) => Availability::Available((va + vc).sqrt()),
        _ => Availability::Unavailable(arm_too_small()),
    };
    Ok(RawContrast { active, control, difference: active.mean - control.mean, se })
}

/// Accounted width of one `f64` outcome and one `f64` treatment.
#[cfg(test)]
const SAMPLE_WIDTH: u64 = 16;

/// A small complete sample on the CPU `f64` path, with its uncertainty.
///
/// The contrast is [`raw_contrast`]. No device bytes are reserved. The fold
/// lease and one learner lease come from `ctx`'s admission pool, and their
/// scratch is returned on success, refusal, and cancellation. The difference
/// stays `f64` and must sit within [`scientific_tolerance`] of that `f64`
/// value. The neural `f32` baseline is not used.
///
/// # Errors
///
/// The errors of [`raw_contrast`]. Also `invalid_argument`
/// (`descriptive_comparison.fold_exceeds_lease`) when the sample does not fit
/// the shared CPU and RAM lease, `descriptive_comparison.learner_exceeds_lease`
/// when the learner does not fit that fold, and
/// `descriptive_comparison.admission_refused` when scratch or the byte account
/// is refused. A cancelled `ctx` is `cancelled_no_claim`.
#[must_use = "a refused contrast is not an estimate"]
#[cfg(test)]
pub fn small_cpu_contrast(
    outcome: &[f64],
    treatment: &[f64],
    ctx: &ExecutionContext,
    device: &DeviceReservationPool,
) -> Result<RawContrast, EstimationError> {
    let Ok(cpu) = select_compute(ComputeSelection::Cpu) else {
        return Err(invalid_data("accelerator selection is refused because no device is promoted"));
    };
    let remainder = cpu.reserve_nothing(device);
    if remainder.device_bytes() != device.device_budget()
        || remainder.spill_bytes() != device.spill_budget()
        || !device_is_untouched(device)
    {
        return Err(invalid_data("a small serial job reserved device bytes"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let bytes = sample_bytes(outcome.len())
        .ok_or_else(|| invalid_data("the sample byte account overflowed"))?;
    let threads = nested_thread_request(ctx.admission().available().threads());
    let mut fold =
        admit_fold(ctx, threads, bytes, FOLD_SCRATCH_BYTES).map_err(cpu_admission_error)?;
    let learner_threads = nested_thread_request(fold.held_threads());
    let mut learner = fold
        .admit_learner(learner_threads, bytes, FOLD_SCRATCH_BYTES)
        .map_err(cpu_admission_error)?;
    if release_if_cancelled(&mut learner, &mut fold, ctx) {
        return Err(cancelled());
    }
    let mut contrast = raw_contrast(outcome, treatment, ctx)?;
    if release_if_cancelled(&mut learner, &mut fold, ctx) {
        return Err(cancelled());
    }
    let exact = contrast.difference;
    contrast.difference = retain_scientific_f64(exact);
    let Some(tolerance) = scientific_tolerance() else {
        return Err(invalid_data("the scientific tolerance was refused"));
    };
    if !within_declared_tolerance(contrast.difference, exact, tolerance) {
        return Err(invalid_data(
            "the contrast left f64 and missed the declared numerical tolerance",
        ));
    }
    if !device_is_untouched(device) {
        return Err(invalid_data("a small serial job reserved device bytes"));
    }
    Ok(contrast)
}

#[cfg(test)]
fn device_is_untouched(device: &DeviceReservationPool) -> bool {
    let available = device.available();
    available.device_bytes() == device.device_budget()
        && available.spill_bytes() == device.spill_budget()
}

#[cfg(test)]
fn sample_bytes(rows: usize) -> Option<u64> {
    u64::try_from(rows).ok()?.checked_mul(SAMPLE_WIDTH)
}

#[cfg(test)]
fn release_if_cancelled(
    learner: &mut LearnerAdmission,
    fold: &mut FoldAdmission,
    ctx: &ExecutionContext,
) -> bool {
    let learner_stopped = learner.observe_cancel(&ctx.cancellation);
    let fold_stopped = fold.observe_cancel(&ctx.cancellation);
    learner_stopped || fold_stopped
}

#[cfg(test)]
fn cpu_admission_error(error: CpuAdmissionError) -> EstimationError {
    match error {
        CpuAdmissionError::Cancelled => cancelled(),
        CpuAdmissionError::FoldExceedsLease { .. } => refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "descriptive_comparison.fold_exceeds_lease",
            "the fold exceeds the shared CPU and RAM lease",
        ),
        CpuAdmissionError::LearnerExceedsLease { .. } => refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "descriptive_comparison.learner_exceeds_lease",
            "the learner exceeds the fold lease",
        ),
        _ => refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "descriptive_comparison.admission_refused",
            "the CPU admission or its scratch was refused",
        ),
    }
}

/// An adjusted estimate and the estimand coding it was computed under.
///
/// The comparison is licensed only for the coding the raw contrast has: active level 1,
/// control level 0, difference of means, all-observed population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjustedEstimate {
    /// The adjusted point estimate.
    pub estimate: f64,
    /// Its standard error as the source published it, if any.
    pub se: Option<f64>,
    /// Active intervention level.
    pub active: f64,
    /// Control intervention level.
    pub control: f64,
    /// Scale of the estimate.
    pub scale: ReportingScale,
    /// Whether the estimate targets the all-observed population (not a subpopulation,
    /// a treated-only effect or a reweighted population).
    pub all_observed_population: bool,
}

impl AdjustedEstimate {
    /// An adjusted average effect on the all-observed population, difference of means,
    /// binary 1-versus-0 coding.
    #[must_use]
    pub const fn mean_difference(estimate: f64, se: Option<f64>) -> Self {
        Self {
            estimate,
            se,
            active: 1.0,
            control: 0.0,
            scale: ReportingScale::MeanDifference,
            all_observed_population: true,
        }
    }
}

/// The raw contrast beside the adjusted estimate, and their descriptive gap.
///
/// Descriptive only: the gap is not a causal decomposition and carries no interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DescriptiveComparison {
    /// The unadjusted contrast.
    pub raw: RawContrast,
    /// The adjusted estimate as supplied.
    pub adjusted: AdjustedEstimate,
    /// `raw.difference - adjusted.estimate`.
    pub gap: f64,
    /// Why the gap has no interval (the covariance of the two estimates is not carried).
    pub gap_interval: Unavailable,
}

/// Set the unadjusted contrast beside an adjusted estimate of the same estimand coding.
///
/// # Errors
/// `route_not_supported` (`descriptive_comparison.estimand_coding_mismatch`) when the
/// adjusted estimate is not a 1-versus-0 difference of means; `population_not_estimable`
/// (`descriptive_comparison.population_not_all_observed`) when it targets another
/// population; `invalid_argument` for a non-finite estimate or standard error; and the
/// [`raw_contrast`] errors (cancellation included).
pub fn compare_raw_adjusted(
    outcome: &[f64],
    treatment: &[f64],
    adjusted: AdjustedEstimate,
    ctx: &ExecutionContext,
) -> Result<DescriptiveComparison, EstimationError> {
    if adjusted.scale != ReportingScale::MeanDifference
        || (adjusted.active - 1.0).abs() > CODE_TOLERANCE
        || adjusted.control.abs() > CODE_TOLERANCE
    {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "descriptive_comparison.estimand_coding_mismatch",
            &format!(
                "the raw contrast is a difference of means at active 1 versus control 0; the \
                 adjusted estimate is on scale {} at active {} versus control {}, so the two do \
                 not share an estimand coding",
                adjusted.scale.name(),
                adjusted.active,
                adjusted.control
            ),
        ));
    }
    if !adjusted.all_observed_population {
        return Err(refuse(
            antecedent_core::reason_code!("population_not_estimable"),
            "descriptive_comparison.population_not_all_observed",
            "the raw contrast averages over all observed rows; an adjusted estimate for another \
             target population answers a different question and is not compared with it",
        ));
    }
    if !adjusted.estimate.is_finite() || adjusted.se.is_some_and(|se| !se.is_finite() || se < 0.0) {
        return Err(invalid_data("the adjusted estimate and its standard error must be finite"));
    }
    let raw = raw_contrast(outcome, treatment, ctx)?;
    Ok(DescriptiveComparison {
        raw,
        adjusted,
        gap: raw.difference - adjusted.estimate,
        gap_interval: Unavailable {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            detail: "descriptive_comparison.gap_interval_unavailable",
            reason: "the covariance between the raw and the adjusted estimate is not carried by \
                     either and is not assumed zero (they use the same rows), so the gap has no \
                     standard error or interval",
        },
    })
}

#[cfg(test)]
mod tests {
    use antecedent_core::device_reservation::DeviceReservationRemainder;
    use antecedent_core::execution::AdmissionPool;

    use super::*;

    fn mean(values: &[f64]) -> f64 {
        values.iter().copied().sum::<f64>() / values.len() as f64
    }

    #[test]
    fn small_cpu_contrast_stays_f64_with_uncertainty() {
        let outcome = [1.0 / 3.0, 0.25, 2.0 / 3.0, 0.5];
        let treatment = [0.0, 0.0, 1.0, 1.0];
        let expected = mean(&[2.0 / 3.0, 0.5]) - mean(&[1.0 / 3.0, 0.25]);
        let device = DeviceReservationPool::fixed(128, 64);
        let ctx = ExecutionContext::for_tests(7);
        let pool = ctx.admission();
        let contrast = small_cpu_contrast(&outcome, &treatment, &ctx, &device).unwrap();
        assert_eq!(contrast.difference.to_bits(), expected.to_bits());
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the f32 round trip is the drop a non-neural contrast must not take"
        )]
        let narrowed = f64::from(expected as f32);
        assert_ne!(contrast.difference.to_bits(), narrowed.to_bits());
        let tolerance = scientific_tolerance().unwrap();
        assert!(within_declared_tolerance(contrast.difference, expected, tolerance));
        assert!(!within_declared_tolerance(narrowed, expected, tolerance));
        let standard_error =
            contrast.se.available().copied().expect("the contrast published no standard error");
        assert!(standard_error.is_finite() && standard_error > 0.0);
        assert_eq!(device.available(), DeviceReservationRemainder::new(128, 64));
        assert_eq!(pool.available().bytes(), pool.byte_budget());
        assert_eq!(pool.available().threads(), pool.thread_budget());
    }

    #[test]
    fn small_cpu_contrast_refuses_a_fold_past_the_lease() {
        let outcome = [1.0 / 3.0, 0.25, 2.0 / 3.0, 0.5];
        let treatment = [0.0, 0.0, 1.0, 1.0];
        let device = DeviceReservationPool::fixed(128, 64);
        let pool = AdmissionPool::fixed(1, 8);
        let ctx = ExecutionContext::for_tests(7).with_admission(pool.clone());
        let error = small_cpu_contrast(&outcome, &treatment, &ctx, &device).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("fold_exceeds_lease"), "{text}");
        assert_eq!(pool.available().bytes(), 8);
        assert_eq!(pool.available().threads(), 1);
        assert_eq!(device.available(), DeviceReservationRemainder::new(128, 64));
    }

    #[test]
    fn cancelled_small_cpu_contrast_reserves_nothing() {
        let outcome = [1.0 / 3.0, 0.25, 2.0 / 3.0, 0.5];
        let treatment = [0.0, 0.0, 1.0, 1.0];
        let device = DeviceReservationPool::fixed(128, 64);
        let pool = AdmissionPool::fixed(1, 1_000);
        let ctx = ExecutionContext::for_tests(7).with_admission(pool.clone());
        ctx.cancellation.cancel();
        let error = small_cpu_contrast(&outcome, &treatment, &ctx, &device).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("cancelled"), "{text}");
        assert_eq!(pool.available().bytes(), 1_000);
        assert_eq!(pool.available().threads(), 1);
        assert_eq!(device.available(), DeviceReservationRemainder::new(128, 64));
    }
}

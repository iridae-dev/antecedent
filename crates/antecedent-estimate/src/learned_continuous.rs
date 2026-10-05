//! Learned continuous-outcome trial-to-target transport (2.2A cell X4).
//!
//! # The one cell
//!
//! A randomized binary source treatment `A` with known probabilities, a continuous
//! outcome `Y`, complete baseline covariates `X` equal to the certified standardizers,
//! and an overlap-supported target population. The estimand is
//!
//! ```text
//! psi = E_target[ E(Y | X, A=1, S=1) - E(Y | X, A=0, S=1) ]
//! ```
//!
//! read through a direct or baseline-standardization certificate (the existing
//! `prepare_trial` graph contract). The target is the nonparticipants of one IID cohort
//! ([`TrialSampling::NestedCohort`]) or a separately sampled representative IID target
//! ([`TrialSampling::IndependentSamples`]); the two designs are distinct and never
//! interchangeable. Sampling is IID only.
//!
//! The estimator is the cross-fitted augmented inverse-odds score of
//! [`estimate_trial_aipw`], with the outcome regressions and the source-membership model
//! learned through `antecedent-learn` `LearnerSpec`s. Every nuisance is cross-fitted on
//! one shared stratified fold assignment and no preprocessing precedes the folds: each
//! learner is fitted on its training rows alone.
//!
//! # Claimed theorem
//!
//! Model double robustness of the point estimate under mean exchangeability over the
//! certified standardizers and positivity of participation, with known randomization
//! probabilities: it is consistent when either the outcome regressions or the
//! participation model is consistent. No efficiency, rate or CATE claim is made.
//!
//! # Inference
//!
//! The original joint outer refit percentile bootstrap, resampled per design with
//! a floor of [`LEARNED_CONTINUOUS_MIN_BOOTSTRAP`], failed its sample-size coverage
//! grid. It remains available only to calibration-internal code and to the older
//! learned-trial route. A design-specific analytic influence interval is now
//! measured by the calibration harness. The public 2.2A route remains closed
//! (`cell_not_licensed`, interval withheld) pending promotion; its estimate is
//! the point with the interval status.
//!
//! Two replicate behaviors are documented rather than changed (their effect on coverage is
//! measured in the rejected bootstrap diagnostic): a replicate reuses the point
//! run's fold label of every resampled row, so fold sizes are not rebalanced (a replicate
//! whose fold lacks a role fails and is counted; any failure withholds the interval), and
//! only the point run is gated on membership overlap ([`check_membership_overlap`]) - a
//! replicate whose out-of-fold membership falls below the threshold still contributes.
//! The shared bootstrap also serves the 2.1 learned-trial cells, whose seeded results must
//! not change. `fit_point` applies no preprocessing beyond an intercept column, so
//! "preprocessing inside folds" holds trivially today.
use crate::learned_trial::{
    TrialAipwInput, TrialAipwOptions, TrialNuisanceDiagnostics, TrialSampling, estimate_trial_aipw,
    trial_fold_assignment, validate_trial_aipw,
};
use crate::{EstimationError, TransportOverlapReport};
use antecedent_core::ExecutionContext;
use antecedent_identify::TransportIdentification;
use antecedent_learn::{LearnerProvenance, LearnerSpec};
use serde::{Deserialize, Serialize};

/// Fewest bootstrap replicates accepted by the closed legacy interval request.
pub const LEARNED_CONTINUOUS_MIN_BOOTSTRAP: u32 = 199;
/// Most bootstrap replicates one request may ask for.
pub const LEARNED_CONTINUOUS_MAX_BOOTSTRAP: u32 = 2000;
/// Most cross-fit folds one request may ask for.
pub const LEARNED_CONTINUOUS_MAX_FOLDS: usize = 20;
/// `(max folds, min bootstrap, max bootstrap)`: the frozen bounds an artifact's premises bind.
pub const LEARNED_CONTINUOUS_BOUNDS: (usize, u32, u32) = (
    LEARNED_CONTINUOUS_MAX_FOLDS,
    LEARNED_CONTINUOUS_MIN_BOOTSTRAP,
    LEARNED_CONTINUOUS_MAX_BOOTSTRAP,
);
/// The fold scheme recorded on every estimate.
pub const LEARNED_CONTINUOUS_FOLD_SCHEME: &str = "stratified_round_robin_source_arm_and_target";
/// The one supported target of a request.
pub const LEARNED_CONTINUOUS_TARGET: &str = "target_mean_contrast";

/// Point-only status: no interval was requested.
pub const CONTINUOUS_POINT_ONLY: &str = "point_only";
/// Withheld status: an interval was requested and is withheld with a reason.
pub const CONTINUOUS_WITHHELD: &str = "withheld";
/// Reason for a point-only estimate.
pub const CONTINUOUS_NO_INTERVAL_REQUESTED: &str = "no_interval_requested";

/// Overlap thresholds, learner specs, folds and bootstrap request of one estimate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedContinuousOptions {
    /// Source outcome nuisance, fitted separately in each randomized arm.
    pub outcome: LearnerSpec,
    /// Source-membership nuisance in the observed sampling design.
    pub membership: LearnerSpec,
    /// Shared cross-fit fold count, `2..=`[`LEARNED_CONTINUOUS_MAX_FOLDS`].
    pub folds: usize,
    /// Smallest out-of-fold membership probability the estimate accepts, in `(0, 0.5)`.
    pub min_membership_probability: f64,
    /// Smallest arm probability of the known randomization, in `(0, 0.5)`.
    pub min_treatment_probability: f64,
    /// Requested bootstrap replicates, at most [`LEARNED_CONTINUOUS_MAX_BOOTSTRAP`].
    /// The public route runs none: the interval route is closed.
    pub bootstrap: u32,
    /// Nominal coverage of the internal interval.
    pub coverage_level: f64,
}

impl Default for LearnedContinuousOptions {
    fn default() -> Self {
        Self {
            outcome: LearnerSpec::Ridge(crate::RidgeSpec::default()),
            membership: LearnerSpec::Logistic(crate::LogisticSpec::default()),
            folds: 5,
            min_membership_probability: 0.05,
            min_treatment_probability: 0.05,
            bootstrap: 0,
            coverage_level: 0.95,
        }
    }
}

impl LearnedContinuousOptions {
    /// The trial-AIPW settings these options run under, with `bootstrap` replicates.
    #[must_use]
    pub const fn trial_options(&self, bootstrap: u32) -> TrialAipwOptions {
        TrialAipwOptions {
            outcome: self.outcome,
            membership: self.membership,
            folds: self.folds,
            bootstrap,
            coverage_level: self.coverage_level,
        }
    }
}

/// Interval status of a public estimate. No interval is ever attached while the
/// interval route is closed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedContinuousUncertainty {
    /// [`CONTINUOUS_POINT_ONLY`] or [`CONTINUOUS_WITHHELD`].
    pub status: String,
    /// [`CONTINUOUS_NO_INTERVAL_REQUESTED`], `estimator_inference_mismatch` or
    /// `cell_not_licensed`.
    pub reason: String,
    /// The namespaced detail when the request was below the replicate floor.
    pub detail: Option<String>,
    /// Bootstrap replicates the request asked for; none ran.
    pub replicates_requested: u32,
}

impl LearnedContinuousUncertainty {
    /// The status of a request for `bootstrap` replicates on the closed route.
    #[must_use]
    pub fn for_request(bootstrap: u32) -> Self {
        let (status, reason, detail) = if bootstrap == 0 {
            (CONTINUOUS_POINT_ONLY, CONTINUOUS_NO_INTERVAL_REQUESTED, None)
        } else if bootstrap < LEARNED_CONTINUOUS_MIN_BOOTSTRAP {
            (
                CONTINUOUS_WITHHELD,
                antecedent_core::reason_code!("estimator_inference_mismatch"),
                Some("learned_transport.bootstrap_below_floor"),
            )
        } else {
            (CONTINUOUS_WITHHELD, antecedent_core::reason_code!("cell_not_licensed"), None)
        };
        Self {
            status: status.into(),
            reason: reason.into(),
            detail: detail.map(Into::into),
            replicates_requested: bootstrap,
        }
    }

    /// Whether an interval is available. Never true while the route is closed.
    #[must_use]
    pub fn available(&self) -> bool {
        false
    }
}

/// How the shared cross-fit folds were assigned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldProvenance {
    /// Fold count.
    pub count: usize,
    /// [`LEARNED_CONTINUOUS_FOLD_SCHEME`].
    pub scheme: String,
    /// Fold of every row, recomputable from the input alone.
    pub assignment: Vec<u16>,
}

/// One executed point estimate with the recomputable evidence behind it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedContinuousEstimate {
    /// Target mean contrast `psi`.
    pub estimate: f64,
    /// Interval status; the interval route is closed.
    pub uncertainty: LearnedContinuousUncertainty,
    /// Out-of-fold source-membership probabilities.
    pub membership: Vec<f64>,
    /// Out-of-fold source outcome mean under treatment zero.
    pub mu0: Vec<f64>,
    /// Out-of-fold source outcome mean under treatment one.
    pub mu1: Vec<f64>,
    /// Provider implementation and version of every fitted nuisance, in role/fold order:
    /// membership, treatment-zero outcome, treatment-one outcome.
    pub provenance: Vec<LearnerProvenance>,
    /// Fold count, scheme and assignment.
    pub folds: FoldProvenance,
    /// Held-out losses, separate from overlap and identification.
    pub diagnostics: TrialNuisanceDiagnostics,
    /// Source-membership (`selection`) and treatment overlap, kept distinct.
    pub overlap: TransportOverlapReport,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn support_refusal(detail: &str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("transport_support_failure"), detail, message)
}

/// The refusal of the closed interval route: the point is retained while the
/// measured analytic candidate awaits a separate API and artifact promotion.
#[must_use]
#[doc(hidden)]
pub fn refuse_learned_continuous_interval() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cell_not_licensed"),
        "learned_transport.interval_withheld",
        "the learned continuous interval route is closed pending promotion of the \
         measured analytic candidate; the point estimate is retained",
    )
}

/// Refuse a sampling design other than the two declared IID designs.
///
/// # Errors
/// `sampling_dependence_unknown` (`learned_transport.non_iid_design`) for clustered,
/// linked or undeclared sampling.
#[doc(hidden)]
pub fn parse_learned_continuous_sampling(name: &str) -> Result<TrialSampling, EstimationError> {
    match name {
        "nested_cohort" => Ok(TrialSampling::NestedCohort),
        "independent_samples" => Ok(TrialSampling::IndependentSamples),
        other => Err(refuse(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            "learned_transport.non_iid_design",
            &format!(
                "sampling '{other}' is not a declared IID design; only nested_cohort and \
                 independent_samples are licensed"
            ),
        )),
    }
}

/// Refuse a heterogeneous or simultaneous target.
///
/// # Errors
/// `route_not_supported` (`learned_transport.cate_requested`) for any target but
/// [`LEARNED_CONTINUOUS_TARGET`].
#[doc(hidden)]
pub fn parse_learned_continuous_target(name: &str) -> Result<(), EstimationError> {
    if name == LEARNED_CONTINUOUS_TARGET {
        return Ok(());
    }
    Err(refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "learned_transport.cate_requested",
        &format!(
            "target '{name}' is not the population mean contrast; conditional, heterogeneous \
             and simultaneous targets are not licensed"
        ),
    ))
}

/// Validate a request before any nuisance is fitted.
///
/// # Errors
/// Declared bounds above the frozen caps (`learned_transport.bounds_exceeded`),
/// randomization probabilities outside the declared bound
/// (`learned_transport.treatment_overlap`), or any [`validate_trial_aipw`] failure.
#[doc(hidden)]
pub fn validate_learned_continuous(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &LearnedContinuousOptions,
) -> Result<(), EstimationError> {
    if options.folds > LEARNED_CONTINUOUS_MAX_FOLDS
        || options.bootstrap > LEARNED_CONTINUOUS_MAX_BOOTSTRAP
    {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "learned_transport.bounds_exceeded",
            &format!(
                "at most {LEARNED_CONTINUOUS_MAX_FOLDS} folds and \
                 {LEARNED_CONTINUOUS_MAX_BOOTSTRAP} bootstrap replicates"
            ),
        ));
    }
    for threshold in [options.min_membership_probability, options.min_treatment_probability] {
        if !(threshold > 0.0 && threshold < 0.5) {
            return Err(EstimationError::data_msg(
                "overlap thresholds must lie strictly between 0 and 0.5",
            ));
        }
    }
    let bound = options.min_treatment_probability;
    for (i, p) in input.randomization.iter().enumerate() {
        if input.source.get(i).copied().unwrap_or(false)
            && p.is_finite()
            && (*p < bound || *p > 1.0 - bound)
        {
            return Err(support_refusal(
                "learned_transport.treatment_overlap",
                &format!("randomization probability {p} leaves [{bound}, {}]", 1.0 - bound),
            ));
        }
    }
    validate_trial_aipw(id, input, &options.trial_options(0))
}

/// Refuse out-of-fold membership probabilities below the declared threshold.
///
/// Every row counts: a trial row below it carries an unbounded odds weight, and a target
/// row below it lies where no trial unit resembles it, so the contrast would extrapolate.
///
/// # Errors
/// `transport_support_failure` (`learned_transport.membership_overlap`).
#[doc(hidden)]
pub fn check_membership_overlap(
    membership: &[f64],
    options: &LearnedContinuousOptions,
) -> Result<(), EstimationError> {
    let floor = options.min_membership_probability;
    let lowest = membership.iter().copied().fold(f64::INFINITY, f64::min);
    if lowest < floor || membership.iter().any(|p| !p.is_finite()) {
        return Err(support_refusal(
            "learned_transport.membership_overlap",
            &format!("out-of-fold membership probability {lowest} is below the declared {floor}"),
        ));
    }
    Ok(())
}

/// Execute the point estimate: cross-fit every nuisance, refuse insufficient overlap,
/// and report the interval status of the closed interval route.
///
/// # Errors
/// Any [`validate_learned_continuous`] refusal, cancellation, a fit failure, or
/// membership overlap below the declared threshold.
pub fn estimate_learned_continuous(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &LearnedContinuousOptions,
    ctx: &ExecutionContext,
) -> Result<LearnedContinuousEstimate, EstimationError> {
    validate_learned_continuous(id, input, options)?;
    let point = estimate_trial_aipw(id, input, &options.trial_options(0), ctx)?;
    check_membership_overlap(&point.membership, options)?;
    Ok(LearnedContinuousEstimate {
        estimate: point.estimate,
        uncertainty: LearnedContinuousUncertainty::for_request(options.bootstrap),
        membership: point.membership,
        mu0: point.mu0,
        mu1: point.mu1,
        provenance: point.provenance,
        folds: FoldProvenance {
            count: options.folds,
            scheme: LEARNED_CONTINUOUS_FOLD_SCHEME.into(),
            assignment: trial_fold_assignment(input, options.folds),
        },
        diagnostics: point.diagnostics,
        overlap: point.overlap,
    })
}

/// The internal estimator the calibration harness measures: the joint outer refit
/// percentile bootstrap of the whole cross-fitted estimator, grouped per design.
///
/// The public route never calls this; it stays closed until the coverage records exist.
/// It is compiled only with the `calibration-internal` feature, which only the facade's
/// dev-dependencies enable (calibration harness and lifecycle tests): an ordinary
/// dependent cannot reach the interval around the `cell_not_licensed` refusal.
///
/// # Errors
/// As [`estimate_learned_continuous`]; and `estimator_inference_mismatch`
/// (`learned_transport.bootstrap_below_floor`) below the replicate floor.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn learned_continuous_interval_internal(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &LearnedContinuousOptions,
    ctx: &ExecutionContext,
) -> Result<crate::learned_trial::TrialAipwEstimate, EstimationError> {
    validate_learned_continuous(id, input, options)?;
    if options.bootstrap < LEARNED_CONTINUOUS_MIN_BOOTSTRAP {
        return Err(refuse(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            "learned_transport.bootstrap_below_floor",
            &format!(
                "{} replicates is below the floor of {LEARNED_CONTINUOUS_MIN_BOOTSTRAP}",
                options.bootstrap
            ),
        ));
    }
    let run = estimate_trial_aipw(id, input, &options.trial_options(options.bootstrap), ctx)?;
    check_membership_overlap(&run.membership, options)?;
    Ok(run)
}

/// Calibration-only Wald interval from the cross-fitted AIPW score. For a
/// nested cohort the nonparticipant denominator is random, so the target
/// contribution is centred at the full estimate. For independent fixed-size
/// samples, the target and trial contributions are centred within their own
/// samples and their variances are added. Nuisance estimation is treated as
/// second order under the usual cross-fitting rate conditions.
#[cfg(feature = "calibration-internal")]
fn learned_continuous_analytic_interval_internal(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &LearnedContinuousOptions,
    ctx: &ExecutionContext,
) -> Result<(f64, (f64, f64)), EstimationError> {
    use crate::learned_trial::TrialSampling;

    validate_learned_continuous(id, input, options)?;
    let run = estimate_trial_aipw(id, input, &options.trial_options(0), ctx)?;
    check_membership_overlap(&run.membership, options)?;
    let target_n = input.source.iter().filter(|s| !**s).count() as f64;
    let trial_n = input.source.len() as f64 - target_n;
    if target_n <= 1.0 || trial_n <= 1.0 {
        return Err(EstimationError::data_msg("analytic trial interval needs two rows per sample"));
    }
    let mut target = Vec::new();
    let mut trial = Vec::new();
    for i in 0..input.source.len() {
        if input.source[i] {
            let p = run.membership[i];
            let e = input.randomization[i];
            let residual = if input.treatment[i] {
                (input.outcome[i] - run.mu1[i]) / e
            } else {
                -(input.outcome[i] - run.mu0[i]) / (1.0 - e)
            };
            trial.push((1.0 - p) / p * residual);
        } else {
            target.push(run.mu1[i] - run.mu0[i]);
        }
    }
    let variance = match input.sampling {
        TrialSampling::NestedCohort => {
            let sum = target.iter().map(|x| (x - run.estimate).powi(2)).sum::<f64>()
                + trial.iter().map(|x| x * x).sum::<f64>();
            sum / (target_n * target_n)
        }
        TrialSampling::IndependentSamples => {
            let target_mean = target.iter().sum::<f64>() / target_n;
            let trial_mean = trial.iter().sum::<f64>() / trial_n;
            target.iter().map(|x| (x - target_mean).powi(2)).sum::<f64>()
                / (target_n * (target_n - 1.0))
                + trial.iter().map(|x| (x - trial_mean).powi(2)).sum::<f64>() * trial_n
                    / ((trial_n - 1.0) * target_n * target_n)
        }
    };
    if !variance.is_finite() || variance <= 0.0 {
        return Err(EstimationError::data_msg(
            "analytic trial variance is nonpositive or nonfinite",
        ));
    }
    let z = antecedent_stats::normal_ppf(0.5 + options.coverage_level / 2.0);
    let width = z * variance.sqrt();
    Ok((run.estimate, (run.estimate - width, run.estimate + width)))
}

#[cfg(feature = "calibration-internal")]
impl LearnedContinuousOptions {
    /// Internal candidate for coverage measurement; no public interval is licensed.
    #[doc(hidden)]
    pub fn analytic_interval_internal(
        &self,
        id: &TransportIdentification,
        input: &TrialAipwInput,
        ctx: &ExecutionContext,
    ) -> Result<(f64, (f64, f64)), EstimationError> {
        learned_continuous_analytic_interval_internal(id, input, self, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_report_their_interval_status() {
        let none = LearnedContinuousUncertainty::for_request(0);
        assert_eq!(
            (none.status.as_str(), none.reason.as_str()),
            ("point_only", "no_interval_requested")
        );
        let low = LearnedContinuousUncertainty::for_request(50);
        assert_eq!(low.reason, "estimator_inference_mismatch");
        assert_eq!(low.detail.as_deref(), Some("learned_transport.bootstrap_below_floor"));
        let closed = LearnedContinuousUncertainty::for_request(199);
        assert_eq!(
            (closed.status.as_str(), closed.reason.as_str()),
            ("withheld", "cell_not_licensed")
        );
        assert!(closed.detail.is_none() && !closed.available());
    }

    #[test]
    fn undeclared_sampling_and_targets_are_refused() {
        assert!(parse_learned_continuous_sampling("nested_cohort").is_ok());
        assert!(matches!(
            parse_learned_continuous_sampling("clustered"),
            Err(EstimationError::Refused { code: "sampling_dependence_unknown", .. })
        ));
        assert!(parse_learned_continuous_target("target_mean_contrast").is_ok());
        assert!(matches!(
            parse_learned_continuous_target("cate"),
            Err(EstimationError::Refused { code: "route_not_supported", .. })
        ));
    }
}

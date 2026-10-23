//! Target-native refutation: refuters that test the quantity a claim actually reports.
//!
//! The scalar refuters of [`crate::suite`] perturb the data and re-estimate one average
//! treatment effect. A claim about a conditional effect, a response derivative, a temporal
//! effect, a mediation decomposition, a vector of intervention-world values or a joint
//! contrast is not certified by that number. This module states what each target family is
//! refuted by and runs those refuters on the **whole target vector**:
//!
//! * [`RefutationTarget`] names the target a check speaks to: its family, its quadrant
//!   (static or temporal, frequentist or Bayesian), its dimension and the identity of the
//!   compiled program or query that defines it.
//! * [`TargetRefutationPlan::for_target`] is the catalogue of target-native refuters for one
//!   target and one suite.
//! * [`run_target_plan`] executes a plan against a [`TargetRefit`], the whole-procedure refit
//!   the route supplies, and returns one [`TargetRefutationReport`] per refuter with a joint
//!   statistic over the full vector (never a set of marginal scalars).
//! * [`certify_target_refutations`] binds reports to a claim and refuses, with a typed
//!   [`TargetCertificationError`] naming the next attempt, any report whose target is not the
//!   claim's target, including a scalar average-effect report offered for another target
//!   ([`refuse_scalar_reports_for_target`]).
//!
//! Every report is a diagnostic. Its pass threshold has no measured error rate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod bayesian;
mod derivative;
mod engine;
mod joint;

use std::fmt;
use std::sync::Arc;

use antecedent_core::CanonicalWriter;

use crate::common::RefutationReport;

pub use bayesian::{
    PosteriorPredictiveTarget, PriorSensitivityTarget, posterior_predictive_target_check,
    prior_sensitivity_target_check,
};
pub use derivative::{
    DerivativeCheckPoint, DerivativeConsistency, LocalLinearity, finite_difference_consistency,
    local_linearity,
};
pub use engine::{
    TargetPerturbation, TargetRefit, TargetRefitError, TargetRefutationConfig, TargetRefutationRun,
    TargetRefuterRefusal, run_target_plan, target_refuter_stream,
};
pub use joint::{JointStatistic, JointStatisticKind, joint_mahalanobis};

/// Domain tag of a refutation-target identity.
const TARGET_IDENTITY_TAG: &str = "antecedent.refutation_target";

/// Family of the quantity a claim reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TargetFamily {
    /// A single average treatment effect: the target the scalar suites were built for.
    ScalarAverageEffect,
    /// Effects conditional on covariate values or subgroups (one coordinate per stratum).
    ConditionalEffect,
    /// Derivative, average derivative, directional derivative or Jacobian of a response.
    ResponseDerivative,
    /// An effect indexed by time or horizon on series or panel data.
    TemporalEffect,
    /// A mediation decomposition (total, direct and indirect components).
    Mediation,
    /// Outcome functionals under several intervention worlds, reported as one vector.
    WorldValued,
    /// A joint contrast of several outcomes or several treatment levels.
    JointContrast,
    /// Per-component attribution (anomaly noise contributions or change contributions) of a
    /// fitted structural model; it has no treatment, so the treatment-keyed refuters do not
    /// denote and the stability refuters of its whole-procedure refit apply.
    Attribution,
}

impl TargetFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScalarAverageEffect => "scalar_average_effect",
            Self::ConditionalEffect => "conditional_effect",
            Self::ResponseDerivative => "response_derivative",
            Self::TemporalEffect => "temporal_effect",
            Self::Mediation => "mediation",
            Self::WorldValued => "world_valued",
            Self::JointContrast => "joint_contrast",
            Self::Attribution => "attribution",
        }
    }

    /// Parse a [`Self::as_str`] name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "scalar_average_effect" => Self::ScalarAverageEffect,
            "conditional_effect" => Self::ConditionalEffect,
            "response_derivative" => Self::ResponseDerivative,
            "temporal_effect" => Self::TemporalEffect,
            "mediation" => Self::Mediation,
            "world_valued" => Self::WorldValued,
            "joint_contrast" => Self::JointContrast,
            "attribution" => Self::Attribution,
            _ => return None,
        })
    }
}

/// Static or temporal data, frequentist or Bayesian inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TargetQuadrant {
    /// Series, panel or event data rather than independent rows.
    pub temporal: bool,
    /// Posterior inference rather than a sampling law.
    pub bayesian: bool,
}

impl TargetQuadrant {
    /// Static frequentist.
    pub const STATIC_FREQUENTIST: Self = Self { temporal: false, bayesian: false };
    /// Static Bayesian.
    pub const STATIC_BAYESIAN: Self = Self { temporal: false, bayesian: true };
    /// Temporal frequentist.
    pub const TEMPORAL_FREQUENTIST: Self = Self { temporal: true, bayesian: false };
    /// Temporal Bayesian.
    pub const TEMPORAL_BAYESIAN: Self = Self { temporal: true, bayesian: true };

    /// Stable name, `static.frequentist` and so on.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match (self.temporal, self.bayesian) {
            (false, false) => "static.frequentist",
            (false, true) => "static.bayesian",
            (true, false) => "temporal.frequentist",
            (true, true) => "temporal.bayesian",
        }
    }
}

/// The target a refutation or a sensitivity range speaks to.
///
/// Two targets are the same exactly when their [`Self::identity`] digests are equal; the
/// digest covers the family, the quadrant, the dimension, the coordinate labels and the
/// caller's defining identity (a lowered-program identity or a query digest).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RefutationTarget {
    family: TargetFamily,
    quadrant: TargetQuadrant,
    labels: Arc<[Arc<str>]>,
    source: [u8; 32],
    identity: [u8; 32],
}

impl RefutationTarget {
    /// A target of `family` in `quadrant` with one label per coordinate, defined by `source`
    /// (the identity of the compiled program or of the query that defines it).
    ///
    /// # Errors
    ///
    /// No coordinates, a [`TargetFamily::ScalarAverageEffect`] with more than one, or an
    /// empty label.
    pub fn new(
        family: TargetFamily,
        quadrant: TargetQuadrant,
        labels: impl IntoIterator<Item = impl Into<Arc<str>>>,
        source: [u8; 32],
    ) -> Result<Self, crate::ValidationError> {
        let labels: Arc<[Arc<str>]> =
            labels.into_iter().map(|label| -> Arc<str> { label.into() }).collect();
        if labels.is_empty() {
            return Err(crate::ValidationError::NotApplicable {
                message: "a refutation target needs at least one coordinate",
            });
        }
        if family == TargetFamily::ScalarAverageEffect && labels.len() != 1 {
            return Err(crate::ValidationError::NotApplicable {
                message: "a scalar average-effect target has exactly one coordinate",
            });
        }
        if labels.iter().any(|label| label.is_empty()) {
            return Err(crate::ValidationError::NotApplicable {
                message: "a refutation-target coordinate label is empty",
            });
        }
        let mut writer = CanonicalWriter::new(TARGET_IDENTITY_TAG);
        writer.text(family.as_str()).text(quadrant.as_str()).texts(&labels[..]).bytes(&source);
        let identity = writer.finish();
        Ok(Self { family, quadrant, labels, source, identity })
    }

    /// The scalar average-effect target of `source`, the one the scalar suites refute.
    #[must_use]
    pub fn scalar_average_effect(quadrant: TargetQuadrant, source: [u8; 32]) -> Self {
        // One non-empty label and the scalar family: `new` cannot refuse.
        match Self::new(TargetFamily::ScalarAverageEffect, quadrant, ["ate"], source) {
            Ok(target) => target,
            Err(_) => unreachable!("one non-empty label is a valid scalar target"),
        }
    }

    /// Target family.
    #[must_use]
    pub const fn family(&self) -> TargetFamily {
        self.family
    }

    /// Quadrant.
    #[must_use]
    pub const fn quadrant(&self) -> TargetQuadrant {
        self.quadrant
    }

    /// Number of coordinates.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.labels.len()
    }

    /// Coordinate labels, in vector order.
    #[must_use]
    pub fn labels(&self) -> &[Arc<str>] {
        &self.labels
    }

    /// Defining identity the caller supplied.
    #[must_use]
    pub const fn source(&self) -> &[u8; 32] {
        &self.source
    }

    /// Identity of the target.
    #[must_use]
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Lower-case hex of [`Self::identity`].
    #[must_use]
    pub fn identity_hex(&self) -> String {
        hex(&self.identity)
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// What a target-native refuter does.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TargetRefuterKind {
    /// Replace the treatment by an independent permutation (a circular shift on series); the
    /// whole target vector should be centred at zero.
    PlaceboTreatment,
    /// Replace the outcome by an independent permutation (a circular shift on series).
    PlaceboOutcome,
    /// Add an independent simulated common cause to the adjustment; the target should not move.
    RandomCommonCause,
    /// Re-estimate on random subsets (contiguous windows on series, whole units on panels).
    SubsetStability,
    /// Re-estimate the full target under the declared resampling design.
    BootstrapStability,
    /// Swap the exposure for a declared negative-control exposure; the target should be zero.
    NegativeControlExposure,
    /// Swap the outcome for a declared negative-control outcome; the target should be zero.
    NegativeControlOutcome,
    /// Temporal targets: shift the treatment to a future period (a lead); a past-outcome
    /// effect of a future treatment is a pre-trend.
    PreTrend,
    /// Temporal targets: the outcome before treatment onset should not respond to the
    /// treatment that follows.
    Anticipation,
    /// Mediation: the range of mediator-outcome error correlation under which the indirect
    /// effect keeps its sign (sequential ignorability, which in the linear model also carries
    /// the cross-world independence the natural effects need).
    SequentialIgnorability,
    /// Derivatives: the reported derivative against a central finite difference of the
    /// fitted response, with a Richardson truncation-error bound.
    FiniteDifferenceConsistency,
    /// Derivatives: second-difference curvature relative to the slope over the step.
    LocalLinearity,
    /// Bayesian: posterior of the full target under alternative priors.
    PriorSensitivity,
    /// Bayesian: posterior-predictive check of the target quantity itself.
    PosteriorPredictive,
}

impl TargetRefuterKind {
    /// Stable refuter id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlaceboTreatment => "target.placebo_treatment",
            Self::PlaceboOutcome => "target.placebo_outcome",
            Self::RandomCommonCause => "target.random_common_cause",
            Self::SubsetStability => "target.subset_stability",
            Self::BootstrapStability => "target.bootstrap_stability",
            Self::NegativeControlExposure => "target.negative_control_exposure",
            Self::NegativeControlOutcome => "target.negative_control_outcome",
            Self::PreTrend => "target.pre_trend",
            Self::Anticipation => "target.anticipation",
            Self::SequentialIgnorability => "target.sequential_ignorability",
            Self::FiniteDifferenceConsistency => "target.finite_difference_consistency",
            Self::LocalLinearity => "target.local_linearity",
            Self::PriorSensitivity => "target.prior_sensitivity",
            Self::PosteriorPredictive => "target.posterior_predictive",
        }
    }

    /// Parse a [`Self::as_str`] id.
    #[must_use]
    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|kind| kind.as_str() == id)
    }

    /// Every refuter kind.
    pub const ALL: [Self; 14] = [
        Self::PlaceboTreatment,
        Self::PlaceboOutcome,
        Self::RandomCommonCause,
        Self::SubsetStability,
        Self::BootstrapStability,
        Self::NegativeControlExposure,
        Self::NegativeControlOutcome,
        Self::PreTrend,
        Self::Anticipation,
        Self::SequentialIgnorability,
        Self::FiniteDifferenceConsistency,
        Self::LocalLinearity,
        Self::PriorSensitivity,
        Self::PosteriorPredictive,
    ];

    /// What the refuted target is compared with.
    #[must_use]
    pub const fn expectation(self) -> RefuterExpectation {
        match self {
            Self::PlaceboTreatment
            | Self::PlaceboOutcome
            | Self::NegativeControlExposure
            | Self::NegativeControlOutcome
            | Self::PreTrend
            | Self::Anticipation => RefuterExpectation::Null,
            Self::RandomCommonCause
            | Self::SubsetStability
            | Self::BootstrapStability
            | Self::PriorSensitivity
            | Self::PosteriorPredictive => RefuterExpectation::Original,
            Self::SequentialIgnorability
            | Self::FiniteDifferenceConsistency
            | Self::LocalLinearity => RefuterExpectation::Diagnostic,
        }
    }
}

/// What a refuter compares the perturbed target with.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuterExpectation {
    /// The perturbation removes the causal signal: the target should be the zero vector.
    Null,
    /// The perturbation should leave the target where the original estimate put it.
    Original,
    /// A route-specific diagnostic with its own statistic.
    Diagnostic,
}

/// Which target-native suite a plan runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TargetSuite {
    /// Cheapest informative checks: one null refuter and one stability refuter (Bayesian:
    /// prior sensitivity), plus the family's native diagnostic.
    Cheap,
    /// Placebo treatment and outcome, plus the temporal pre-trend on temporal targets.
    Placebo,
    /// Every refuter the target family and quadrant admit.
    Full,
}

impl TargetSuite {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cheap => "cheap",
            Self::Placebo => "placebo",
            Self::Full => "full",
        }
    }

    /// Parse a [`Self::as_str`] name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "cheap" => Self::Cheap,
            "placebo" => Self::Placebo,
            "full" => Self::Full,
            _ => return None,
        })
    }
}

/// The target-native refuters one target is checked by, in execution order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetRefutationPlan {
    target: RefutationTarget,
    suite: TargetSuite,
    refuters: Arc<[TargetRefuterKind]>,
}

impl TargetRefutationPlan {
    /// The catalogue for `target` and `suite`.
    ///
    /// * Frequentist quadrants: placebo treatment and outcome, random common cause, subset and
    ///   bootstrap stability of the whole vector, negative-control exposure and outcome
    ///   (executed only where the route declares the controls).
    /// * Temporal quadrants add the pre-trend and anticipation checks.
    /// * Mediation adds the sequential-ignorability range; derivatives add the
    ///   finite-difference consistency and local-linearity diagnostics.
    /// * Bayesian quadrants replace resampling stability by prior sensitivity and add the
    ///   posterior-predictive check of the target quantity; the placebo and native
    ///   diagnostics are kept.
    #[must_use]
    pub fn for_target(target: RefutationTarget, suite: TargetSuite) -> Self {
        use TargetRefuterKind as K;
        let quadrant = target.quadrant();
        let family = target.family();
        let mut refuters: Vec<TargetRefuterKind> = Vec::new();
        if family == TargetFamily::Attribution {
            // No treatment or outcome to permute or swap: the whole contribution vector is
            // refuted by refitting the attribution program on perturbed data.
            refuters.push(K::SubsetStability);
            if suite != TargetSuite::Cheap {
                refuters.push(K::BootstrapStability);
            }
            return Self { target, suite, refuters: refuters.into() };
        }
        let native: &[TargetRefuterKind] = match family {
            TargetFamily::Mediation => &[K::SequentialIgnorability],
            TargetFamily::ResponseDerivative => {
                &[K::FiniteDifferenceConsistency, K::LocalLinearity]
            }
            _ => &[],
        };
        match suite {
            TargetSuite::Cheap => {
                refuters.push(K::PlaceboTreatment);
                refuters.push(if quadrant.bayesian {
                    K::PriorSensitivity
                } else {
                    K::SubsetStability
                });
                refuters.extend_from_slice(native);
            }
            TargetSuite::Placebo => {
                refuters.extend_from_slice(&[K::PlaceboTreatment, K::PlaceboOutcome]);
                if quadrant.temporal {
                    refuters.push(K::PreTrend);
                }
            }
            TargetSuite::Full => {
                refuters.extend_from_slice(&[K::PlaceboTreatment, K::PlaceboOutcome]);
                if quadrant.bayesian {
                    refuters.extend_from_slice(&[K::PriorSensitivity, K::PosteriorPredictive]);
                } else {
                    refuters.extend_from_slice(&[
                        K::RandomCommonCause,
                        K::SubsetStability,
                        K::BootstrapStability,
                    ]);
                }
                refuters
                    .extend_from_slice(&[K::NegativeControlExposure, K::NegativeControlOutcome]);
                if quadrant.temporal {
                    refuters.extend_from_slice(&[K::PreTrend, K::Anticipation]);
                }
                refuters.extend_from_slice(native);
            }
        }
        Self { target, suite, refuters: refuters.into() }
    }

    /// Target this plan refutes.
    #[must_use]
    pub const fn target(&self) -> &RefutationTarget {
        &self.target
    }

    /// Suite the plan was built for.
    #[must_use]
    pub const fn suite(&self) -> TargetSuite {
        self.suite
    }

    /// Refuters in execution order.
    #[must_use]
    pub fn refuters(&self) -> &[TargetRefuterKind] {
        &self.refuters
    }
}

/// One target-native refutation on the whole target vector.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefutationReport {
    /// Target the refuter ran on.
    pub target: RefutationTarget,
    /// Refuter.
    pub kind: TargetRefuterKind,
    /// Original target vector.
    pub original: Arc<[f64]>,
    /// Mean of the refuted target vectors (the refuted vector for a single refit).
    pub refuted: Arc<[f64]>,
    /// Joint statistic over the whole vector.
    pub statistic: JointStatistic,
    /// Whether the check passed its configured threshold.
    pub passed: bool,
    /// Whether the check could speak to the target (enough replicates, a usable covariance).
    pub informative: bool,
    /// Successful refits.
    pub replicates: u32,
    /// Why the check did not pass, when it did not.
    pub failure: Option<Arc<str>>,
}

impl TargetRefutationReport {
    /// Refuter id.
    #[must_use]
    pub const fn refuter(&self) -> &'static str {
        self.kind.as_str()
    }
}

/// Why a claim refuses a set of refutations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetCertificationError {
    /// A report was computed on another target.
    TargetMismatch {
        /// Refuter id of the offending report.
        refuter: Arc<str>,
        /// Identity of the claim's target (hex).
        claim_target: String,
        /// Identity of the report's target (hex).
        report_target: String,
        /// What to run instead.
        next_attempt: &'static str,
    },
    /// A scalar average-effect refutation was offered for a target that is not that scalar.
    ScalarRefuterForOtherTarget {
        /// Refuter id of the scalar report.
        refuter: Arc<str>,
        /// Family of the claim's target.
        claim_family: TargetFamily,
        /// What to run instead.
        next_attempt: &'static str,
    },
}

/// The next attempt named by every target refusal.
pub const TARGET_NATIVE_NEXT_ATTEMPT: &str = "run the target-native suite for this claim's \
     target (TargetRefutationPlan::for_target with the claim's RefutationTarget, executed by \
     run_target_plan on the route's whole-procedure refit)";

impl fmt::Display for TargetCertificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetMismatch { refuter, claim_target, report_target, next_attempt } => write!(
                f,
                "refutation `{refuter}` was computed on target {report_target}, not on the \
                 claim's target {claim_target}; next attempt: {next_attempt}"
            ),
            Self::ScalarRefuterForOtherTarget { refuter, claim_family, next_attempt } => write!(
                f,
                "scalar average-effect refutation `{refuter}` cannot certify a {} target; \
                 next attempt: {next_attempt}",
                claim_family.as_str()
            ),
        }
    }
}

impl std::error::Error for TargetCertificationError {}

/// Refutations a claim accepted, bound to the claim's target.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetCertificate {
    /// The claim's target.
    pub target: RefutationTarget,
    /// Accepted reports, all on [`Self::target`].
    pub reports: Arc<[TargetRefutationReport]>,
    /// Identity binding the target and every report's refuter, statistic and verdict.
    pub identity: [u8; 32],
}

impl TargetCertificate {
    /// Whether every informative report passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.reports.iter().all(|report| !report.informative || report.passed)
    }
}

/// Accept `reports` for a claim on `claim`.
///
/// # Errors
///
/// [`TargetCertificationError::TargetMismatch`] for the first report whose target identity is
/// not the claim's.
pub fn certify_target_refutations(
    claim: &RefutationTarget,
    reports: &[TargetRefutationReport],
) -> Result<TargetCertificate, TargetCertificationError> {
    for report in reports {
        if report.target.identity() != claim.identity() {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from(report.refuter()),
                claim_target: claim.identity_hex(),
                report_target: report.target.identity_hex(),
                next_attempt: TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
    }
    let mut writer = CanonicalWriter::new("antecedent.target_certificate");
    writer.bytes(claim.identity());
    for report in reports {
        writer
            .text(report.refuter())
            .text(report.statistic.kind.as_str())
            .real(report.statistic.value)
            .opt_real(report.statistic.p_value)
            .word(u64::from(report.passed))
            .word(u64::from(report.informative))
            .word(u64::from(report.replicates));
    }
    Ok(TargetCertificate {
        target: claim.clone(),
        reports: reports.to_vec().into(),
        identity: writer.finish(),
    })
}

/// Refuse scalar average-effect reports for a claim whose target is anything else.
///
/// Scalar refuters ([`RefutationReport`]) carry no target identity: they always refute the
/// scalar average effect. A claim on that scalar may cite them; a claim on any other target
/// may not, whatever their verdict.
///
/// # Errors
///
/// [`TargetCertificationError::ScalarRefuterForOtherTarget`] for the first scalar report
/// offered for a non-scalar target.
pub fn refuse_scalar_reports_for_target(
    claim: &RefutationTarget,
    scalar: &[RefutationReport],
) -> Result<(), TargetCertificationError> {
    if claim.family() == TargetFamily::ScalarAverageEffect {
        return Ok(());
    }
    match scalar.first() {
        None => Ok(()),
        Some(report) => Err(TargetCertificationError::ScalarRefuterForOtherTarget {
            refuter: Arc::clone(&report.refuter),
            claim_family: claim.family(),
            next_attempt: TARGET_NATIVE_NEXT_ATTEMPT,
        }),
    }
}

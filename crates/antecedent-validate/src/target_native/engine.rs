//! Execution of a [`TargetRefutationPlan`] against a route's whole-procedure refit.
//!
//! The route supplies a [`TargetRefit`]: given a perturbation and a keyed random stream, it
//! perturbs its own data, reruns its own complete procedure (identification reuse,
//! nuisance fits, the estimator and the target functional) and returns the full target
//! vector. The engine owns the replicate loop, the random streams, admission of the
//! replicate buffer, cancellation and the joint statistic, so every route's refuters share
//! one definition of what each check means.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_core::{
    CausalRng, ChainId, ExecutionContext, FoldId, ReductionPolicy, ReplicateId, RngKey,
    SemanticOperation,
};

use super::joint::joint_mahalanobis_with_bias;
use super::{
    RefutationTarget, RefuterExpectation, TargetFamily, TargetRefutationPlan,
    TargetRefutationReport, TargetRefuterKind,
};
use crate::ValidationError;
use crate::common::{admit_cells, check_cancelled};

/// One perturbation the route applies before rerunning its procedure.
///
/// `replicate` indexes the refit inside its refuter; the random stream handed to
/// [`TargetRefit::refit`] is keyed by refuter and replicate, never by worker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetPerturbation {
    /// Permute the treatment (static) or circularly shift it by a random offset of at least
    /// one dependence block (temporal).
    PlaceboTreatment {
        /// Replicate index.
        replicate: u32,
    },
    /// Permute (static) or circularly shift (temporal) the outcome.
    PlaceboOutcome {
        /// Replicate index.
        replicate: u32,
    },
    /// Add an independent standard-normal common cause to the adjustment.
    RandomCommonCause {
        /// Replicate index.
        replicate: u32,
    },
    /// Keep a random `fraction` of the sampling units (a contiguous window on series).
    Subset {
        /// Replicate index.
        replicate: u32,
        /// Fraction of units kept, in `(0, 1)`.
        fraction: f64,
    },
    /// Resample under the declared sampling design (whole units, blocks or clusters).
    Bootstrap {
        /// Replicate index.
        replicate: u32,
    },
    /// Swap the exposure for the declared negative-control exposure, on a design-respecting
    /// resample of the data (replicate `0` is the data itself).
    NegativeControlExposure {
        /// Replicate index.
        replicate: u32,
    },
    /// Swap the outcome for the declared negative-control outcome, on a resample as above.
    NegativeControlOutcome {
        /// Replicate index.
        replicate: u32,
    },
    /// Replace the treatment by its value `lead` periods later, on a block resample.
    PreTrend {
        /// Replicate index.
        replicate: u32,
        /// Lead in periods (at least one).
        lead: u32,
    },
    /// Restrict the outcome to periods before treatment onset, on a block resample.
    Anticipation {
        /// Replicate index.
        replicate: u32,
    },
}

/// Why a route could not run one refuter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetRefitError {
    /// The route has no construction for this refuter (for example no declared negative
    /// control); the refuter is recorded as refused with the next attempt.
    Unsupported {
        /// Why.
        reason: Arc<str>,
        /// What would make it runnable.
        next_attempt: Arc<str>,
    },
    /// The refit failed.
    Failed(ValidationError),
}

impl fmt::Display for TargetRefitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { reason, next_attempt } => {
                write!(f, "{reason}; next attempt: {next_attempt}")
            }
            Self::Failed(error) => write!(f, "{error}"),
        }
    }
}

impl From<ValidationError> for TargetRefitError {
    fn from(error: ValidationError) -> Self {
        Self::Failed(error)
    }
}

/// A route's whole-procedure refit of its own target.
pub trait TargetRefit {
    /// The target the route reports.
    fn target(&self) -> &RefutationTarget;

    /// The reported target vector, in [`RefutationTarget::labels`] order.
    fn original(&self) -> &[f64];

    /// Perturb, rerun the whole procedure and return the full target vector.
    ///
    /// # Errors
    ///
    /// [`TargetRefitError::Unsupported`] when the route has no construction for this
    /// perturbation; [`TargetRefitError::Failed`] when the refit fails.
    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError>;

    /// A refuter with a statistic of its own (mediation sensitivity, derivative diagnostics,
    /// Bayesian prior sensitivity and posterior-predictive checks).
    ///
    /// # Errors
    ///
    /// [`TargetRefitError::Unsupported`] by default.
    fn native_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let _ = ctx;
        Err(TargetRefitError::Unsupported {
            reason: Arc::from(format!("this route has no `{}` check", kind.as_str())),
            next_attempt: Arc::from(
                "supply the route's native check through TargetRefit::native_check",
            ),
        })
    }
}

/// Settings of [`run_target_plan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetRefutationConfig {
    /// Refits per replicate-based refuter.
    pub replicates: u32,
    /// Tolerance level: a replicate refuter passes when the reference tail of its joint
    /// discrepancy is at least this. It sets a fixed yardstick on a diagnostic, not the size of
    /// a hypothesis test (see [`super::joint`]).
    pub alpha: f64,
    /// Fraction of units each subset refit keeps.
    pub subset_fraction: f64,
    /// Lead of the pre-trend placebo, in periods.
    pub pre_trend_lead: u32,
}

impl Default for TargetRefutationConfig {
    fn default() -> Self {
        Self { replicates: 30, alpha: 0.05, subset_fraction: 0.8, pre_trend_lead: 1 }
    }
}

/// A refuter the route could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetRefuterRefusal {
    /// Refuter.
    pub kind: TargetRefuterKind,
    /// Why.
    pub reason: Arc<str>,
    /// What would make it runnable.
    pub next_attempt: Arc<str>,
}

/// Reports and refusals of one plan.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefutationRun {
    /// One report per refuter that ran.
    pub reports: Vec<TargetRefutationReport>,
    /// One refusal per refuter the route could not run.
    pub refused: Vec<TargetRefuterRefusal>,
}

/// Stream tag of one refuter kind: a fixed word per kind, so adding a kind never moves the
/// streams of another.
const fn operation_tag(kind: TargetRefuterKind) -> u64 {
    0x7A52_4700_0000_0000
        | match kind {
            TargetRefuterKind::PlaceboTreatment => 1,
            TargetRefuterKind::PlaceboOutcome => 2,
            TargetRefuterKind::RandomCommonCause => 3,
            TargetRefuterKind::SubsetStability => 4,
            TargetRefuterKind::BootstrapStability => 5,
            TargetRefuterKind::NegativeControlExposure => 6,
            TargetRefuterKind::NegativeControlOutcome => 7,
            TargetRefuterKind::PreTrend => 8,
            TargetRefuterKind::Anticipation => 9,
            TargetRefuterKind::SequentialIgnorability => 10,
            TargetRefuterKind::FiniteDifferenceConsistency => 11,
            TargetRefuterKind::LocalLinearity => 12,
            TargetRefuterKind::PriorSensitivity => 13,
            TargetRefuterKind::PosteriorPredictive => 14,
        }
}

/// The keyed stream of replicate `replicate` of `kind` under the context seed.
#[must_use]
pub fn target_refuter_stream(
    kind: TargetRefuterKind,
    replicate: u32,
    ctx: &ExecutionContext,
) -> CausalRng {
    RngKey::new(
        SemanticOperation::new(operation_tag(kind)),
        FoldId::new(0),
        ChainId::new(0),
        ReplicateId::new(u64::from(replicate)),
        ReductionPolicy::Strict,
    )
    .open(ctx.rng.master_seed())
}

fn perturbation(
    kind: TargetRefuterKind,
    replicate: u32,
    config: &TargetRefutationConfig,
) -> Option<TargetPerturbation> {
    Some(match kind {
        TargetRefuterKind::PlaceboTreatment => TargetPerturbation::PlaceboTreatment { replicate },
        TargetRefuterKind::PlaceboOutcome => TargetPerturbation::PlaceboOutcome { replicate },
        TargetRefuterKind::RandomCommonCause => TargetPerturbation::RandomCommonCause { replicate },
        TargetRefuterKind::SubsetStability => {
            TargetPerturbation::Subset { replicate, fraction: config.subset_fraction }
        }
        TargetRefuterKind::BootstrapStability => TargetPerturbation::Bootstrap { replicate },
        TargetRefuterKind::NegativeControlExposure => {
            TargetPerturbation::NegativeControlExposure { replicate }
        }
        TargetRefuterKind::NegativeControlOutcome => {
            TargetPerturbation::NegativeControlOutcome { replicate }
        }
        TargetRefuterKind::PreTrend => {
            TargetPerturbation::PreTrend { replicate, lead: config.pre_trend_lead }
        }
        TargetRefuterKind::Anticipation => TargetPerturbation::Anticipation { replicate },
        TargetRefuterKind::SequentialIgnorability
        | TargetRefuterKind::FiniteDifferenceConsistency
        | TargetRefuterKind::LocalLinearity
        | TargetRefuterKind::PriorSensitivity
        | TargetRefuterKind::PosteriorPredictive => return None,
    })
}

/// Run every refuter of `plan` on `refit`.
///
/// Replicate-based refuters lease their `replicates × dimension` buffer before the first
/// refit, poll cancellation before each refit and key each refit's stream by refuter and
/// replicate, so the result does not depend on the thread count. A refuter the route cannot
/// construct is recorded in [`TargetRefutationRun::refused`] with its next attempt; any
/// other failure aborts the run.
///
/// # Errors
///
/// The plan's target is not the refit's target, an invalid configuration, cancellation, an
/// admission refusal, or a failed refit.
pub fn run_target_plan(
    plan: &TargetRefutationPlan,
    refit: &dyn TargetRefit,
    config: &TargetRefutationConfig,
    ctx: &ExecutionContext,
) -> Result<TargetRefutationRun, ValidationError> {
    if plan.target().identity() != refit.target().identity() {
        return Err(ValidationError::estimation_msg(format!(
            "target refutation plan is for target {}, the refit reports target {}",
            plan.target().identity_hex(),
            refit.target().identity_hex()
        )));
    }
    let k = refit.target().dimension();
    if refit.original().len() != k {
        return Err(ValidationError::NotApplicable {
            message: "the refit's original vector does not match its target dimension",
        });
    }
    if !(config.alpha > 0.0
        && config.alpha < 1.0
        && config.subset_fraction > 0.0
        && config.subset_fraction < 1.0)
        || config.pre_trend_lead == 0
    {
        return Err(ValidationError::NotApplicable {
            message: "target refutation: alpha and the subset fraction must lie in (0, 1) and \
                      the pre-trend lead must be at least one period",
        });
    }
    let mut run = TargetRefutationRun { reports: Vec::new(), refused: Vec::new() };
    for &kind in plan.refuters() {
        check_cancelled(ctx)?;
        let native = kind.expectation() == RefuterExpectation::Diagnostic
            || matches!(
                kind,
                TargetRefuterKind::PriorSensitivity | TargetRefuterKind::PosteriorPredictive
            );
        let outcome = if native {
            refit.native_check(kind, ctx)
        } else {
            replicate_check(kind, kind.expectation(), refit, config, ctx)
        };
        match outcome {
            Ok(report) => {
                if report.target.identity() != refit.target().identity() || report.kind != kind {
                    return Err(ValidationError::estimation_msg(format!(
                        "native check `{}` returned a report for another target or refuter",
                        kind.as_str()
                    )));
                }
                run.reports.push(report);
            }
            Err(TargetRefitError::Unsupported { reason, next_attempt }) => {
                run.refused.push(TargetRefuterRefusal { kind, reason, next_attempt });
            }
            Err(TargetRefitError::Failed(error)) => return Err(error),
        }
    }
    Ok(run)
}

fn replicate_check(
    kind: TargetRefuterKind,
    expectation: RefuterExpectation,
    refit: &dyn TargetRefit,
    config: &TargetRefutationConfig,
    ctx: &ExecutionContext,
) -> Result<TargetRefutationReport, TargetRefitError> {
    let target = refit.target();
    let k = target.dimension();
    let replicates = config.replicates;
    if (replicates as usize) < k + 2 {
        return Err(TargetRefitError::Unsupported {
            reason: Arc::from(format!(
                "{replicates} replicates cannot estimate the covariance of a {k}-coordinate target"
            )),
            next_attempt: Arc::from(format!("raise the replicate count to at least {}", k + 2)),
        });
    }
    let _lease = admit_cells(Some(ctx), replicates as usize, k)?;
    let mut rows = Vec::with_capacity(replicates as usize * k);
    for replicate in 0..replicates {
        check_cancelled(ctx)?;
        let Some(perturbation) = perturbation(kind, replicate, config) else {
            return Err(TargetRefitError::Failed(ValidationError::NotApplicable {
                message: "this refuter has no data perturbation",
            }));
        };
        let mut rng = target_refuter_stream(kind, replicate, ctx);
        let vector = refit.refit(perturbation, &mut rng, ctx)?;
        if vector.len() != k {
            return Err(TargetRefitError::Failed(ValidationError::estimation_msg(format!(
                "refit for `{}` returned {} coordinates, the target has {k}",
                kind.as_str(),
                vector.len()
            ))));
        }
        rows.extend_from_slice(&vector);
    }
    let (statistic, standardised_bias) = match joint_statistic(expectation, target, refit.original(), &rows, k) {
        Ok(pair) => pair,
        // No usable replicate law (a coordinate the perturbation never moves, too few
        // replicates): the refuter cannot speak to this target; record it, do not abort.
        Err(ValidationError::NotApplicable { message }) => {
            return Err(TargetRefitError::Unsupported {
                reason: Arc::from(message),
                next_attempt: Arc::from(
                    "raise the replicate count, or check that the perturbation moves the target",
                ),
            });
        }
        Err(error) => return Err(TargetRefitError::Failed(error)),
    };
    let mut refuted = vec![0.0; k];
    for row in rows.chunks_exact(k) {
        for (m, v) in refuted.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut refuted {
        *m /= f64::from(replicates);
    }
    let p = statistic.p_value.unwrap_or(0.0);
    let passed = p >= config.alpha;
    let failure = (!passed).then(|| {
        Arc::from(format!(
            "`{}`: the {} vector is further from the replicate centre than the tolerance \
             (standardised discrepancy {:.4} on {} df, reference tail {p:.4} < {}); a resampling \
             diagnostic, not a hypothesis test",
            kind.as_str(),
            match expectation {
                RefuterExpectation::Null => "zero",
                _ => "original target",
            },
            statistic.value,
            statistic.df,
            config.alpha
        ))
    });
    Ok(TargetRefutationReport {
        target: target.clone(),
        kind,
        original: refit.original().to_vec().into(),
        refuted: refuted.into(),
        standardised_bias_norm: Some(statistic.value.sqrt()),
        standardised_bias: standardised_bias.into(),
        statistic,
        passed,
        informative: true,
        replicates,
        failure,
    })
}

/// The joint statistic of `rows` against the refuter's expectation.
///
/// World values are levels, not effects: a null refuter removes every contrast between
/// worlds, so its joint test is on the `k − 1` contrasts with the first world.
fn joint_statistic(
    expectation: RefuterExpectation,
    target: &RefutationTarget,
    original: &[f64],
    rows: &[f64],
    k: usize,
) -> Result<(super::JointStatistic, Vec<f64>), ValidationError> {
    match expectation {
        RefuterExpectation::Null if target.family() == TargetFamily::WorldValued => {
            if k < 2 {
                return Err(ValidationError::NotApplicable {
                    message: "a single intervention world has no null contrast to test; declare \
                              at least two worlds, or a contrast target",
                });
            }
            let contrasts: Vec<f64> = rows
                .chunks_exact(k)
                .flat_map(|row| row[1..].iter().map(move |v| v - row[0]))
                .collect();
            joint_mahalanobis_with_bias(&vec![0.0; k - 1], &contrasts, k - 1)
        }
        RefuterExpectation::Null => joint_mahalanobis_with_bias(&vec![0.0; k], rows, k),
        _ => joint_mahalanobis_with_bias(original, rows, k),
    }
}

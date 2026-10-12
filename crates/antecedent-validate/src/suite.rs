//! Validation suite orchestration.
//!
//! Runs requested validators, returning explicit [`ValidationOutcome::NotApplicable`] when a
//! check is incompatible with the estimator/estimand (rather than failing the whole suite).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationWorkspace;

use crate::bayesian_checks::{
    McmcDiagnosticsCheck, PosteriorPredictiveCheck, PriorPredictiveCheck, PriorSensitivity,
};
use crate::bootstrap_refute::BootstrapRefute;
use crate::common::{RefutationProblem, RefutationReport};
use crate::custom::CustomEffectValidator;
use crate::data_subset::DataSubsetRefuter;
use crate::dummy_outcome::DummyOutcome;
use crate::error::ValidationError;
use crate::evalue::EValue;
use crate::graph_refute::GraphRefuter;
use crate::overlap::OverlapRefuter;
use crate::overlap_rule::OverlapRuleRefuter;
use crate::placebo::PlaceboTreatment;
use crate::rcc::RandomCommonCause;
use crate::riesz::RieszSensitivity;
use crate::sensitivity::{LinearSensitivity, NonparametricSensitivity, PartialLinearSensitivity};
use crate::unobserved_common_cause::UnobservedCommonCause;
use crate::validator::run_validator;

use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, PreparedBayesianProblem,
};
use antecedent_identify::IdentificationStatus;

/// Context required to run Bayesian PPC / prior-sensitivity validators.
pub struct BayesianSuiteContext<'a> {
    /// Fitted Bayesian estimator configuration.
    pub estimator: &'a BayesianGComputationAte,
    /// Prepared design used for the primary fit.
    pub prepared: &'a PreparedBayesianProblem,
    /// Primary posterior (used for posterior predictive).
    pub posterior: &'a CausalPosterior,
    /// Identification status passed to sensitivity refits.
    pub identification: IdentificationStatus,
    /// Workspace for sensitivity refits.
    pub workspace: &'a mut BayesianGCompWorkspace,
    /// Original effect estimate (ATE) for report comparison.
    pub original_ate: f64,
    /// Two-sided α for predictive-check pass/fail (default 0.05).
    pub ppc_alpha: f64,
}

impl<'a> BayesianSuiteContext<'a> {
    /// Build with default PPC α = 0.05.
    #[must_use]
    pub fn new(
        estimator: &'a BayesianGComputationAte,
        prepared: &'a PreparedBayesianProblem,
        posterior: &'a CausalPosterior,
        identification: IdentificationStatus,
        workspace: &'a mut BayesianGCompWorkspace,
        original_ate: f64,
    ) -> Self {
        Self {
            estimator,
            prepared,
            posterior,
            identification,
            workspace,
            original_ate,
            ppc_alpha: 0.05,
        }
    }
}

/// Named validators that can be attached to a [`ValidationSuite`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ValidatorId {
    /// Placebo treatment.
    Placebo,
    /// Random common cause.
    RandomCommonCause,
    /// Bootstrap CI coverage of the point estimate (not placebo falsification).
    Bootstrap,
    /// Unobserved common cause.
    UnobservedCommonCause,
    /// Overlap / positivity assessment.
    Overlap,
    /// Overlap-rule / trimming assessment.
    OverlapRule,
    /// Data subset.
    DataSubset,
    /// Dummy outcome.
    DummyOutcome,
    /// E-value.
    EValue,
    /// Leave-one-out adjustment-set sensitivity (drop covariates; not DAG edits).
    Graph,
    /// Linear sensitivity.
    LinearSensitivity,
    /// Partial-linear sensitivity.
    PartialLinearSensitivity,
    /// Nonparametric sensitivity.
    NonparametricSensitivity,
    /// Riesz-representer sensitivity.
    Riesz,
    /// Prior predictive check (Bayesian).
    PriorPredictive,
    /// Posterior predictive check (Bayesian).
    PosteriorPredictive,
    /// Prior sensitivity grid (Bayesian).
    PriorSensitivity,
    /// MCMC ESS / R-hat / divergence diagnostics (Bayesian HMC/SMC).
    McmcDiagnostics,
}

impl ValidatorId {
    /// Stable identifier string, for diagnostics and logs. Not necessarily identical to
    /// the `refuter` name on the [`RefutationReport`] this validator would have produced
    /// (a few validators — e.g. `LinearSensitivity` — dispatch to more than one report
    /// name depending on the estimand); this is a validator id, not a report id.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Placebo => "placebo",
            Self::RandomCommonCause => "random_common_cause",
            Self::Bootstrap => "bootstrap",
            Self::UnobservedCommonCause => "unobserved_common_cause",
            Self::Overlap => "overlap",
            Self::OverlapRule => "overlap_rule",
            Self::DataSubset => "data_subset",
            Self::DummyOutcome => "dummy_outcome",
            Self::EValue => "evalue",
            Self::Graph => "graph",
            Self::LinearSensitivity => "linear_sensitivity",
            Self::PartialLinearSensitivity => "partial_linear_sensitivity",
            Self::NonparametricSensitivity => "nonparametric_sensitivity",
            Self::Riesz => "riesz",
            Self::PriorPredictive => "prior_predictive",
            Self::PosteriorPredictive => "posterior_predictive",
            Self::PriorSensitivity => "prior_sensitivity",
            Self::McmcDiagnostics => "mcmc_diagnostics",
        }
    }
}

impl std::fmt::Display for ValidatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome of one validator in a suite.
#[derive(Clone, Debug)]
pub enum ValidationOutcome {
    /// Validator ran and produced a report.
    Report(RefutationReport),
    /// Validator could not compute a scientific verdict.
    Failed {
        /// Stable validator name (built-in or custom).
        validator: Arc<str>,
        /// Operational or numerical failure (not evidence against the claim).
        reason: Arc<str>,
    },
    /// Validator was requested but is incompatible with this problem.
    NotApplicable {
        /// Validator id.
        validator: ValidatorId,
        /// Why it was skipped.
        reason: Arc<str>,
    },
    /// A custom validator declined this problem. The rest of the suite still runs.
    CustomNotApplicable {
        /// [`CustomEffectValidator::name`].
        name: Arc<str>,
        /// Why it was skipped.
        reason: Arc<str>,
    },
}

/// Ordered suite of validators .
#[derive(Clone, Default)]
pub struct ValidationSuite {
    validators: Vec<ValidatorId>,
    custom: Vec<Arc<dyn CustomEffectValidator>>,
}

impl std::fmt::Debug for ValidationSuite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationSuite")
            .field("validators", &self.validators)
            .field("custom", &self.custom.len())
            .finish()
    }
}

impl ValidationSuite {
    /// Empty suite.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a validator (order preserved).
    #[must_use]
    pub fn with(mut self, id: ValidatorId) -> Self {
        self.validators.push(id);
        self
    }

    /// Append a custom (dyn) effect validator; runs after built-ins.
    #[must_use]
    pub fn with_custom(mut self, validator: Arc<dyn CustomEffectValidator>) -> Self {
        self.custom.push(validator);
        self
    }

    /// Placebo + RCC (legacy default).
    #[must_use]
    pub fn placebo_and_rcc() -> Self {
        Self::new().with(ValidatorId::Placebo).with(ValidatorId::RandomCommonCause)
    }

    /// Cheap interactive validators: overlap / positivity + E-value only.
    #[must_use]
    pub fn overlap_and_evalue() -> Self {
        Self::new().with(ValidatorId::Overlap).with(ValidatorId::EValue)
    }

    /// Overlap only. Used when an E-value would be applied to a level, not a contrast.
    #[must_use]
    pub fn overlap_only() -> Self {
        Self::new().with(ValidatorId::Overlap)
    }

    /// Falsifiers of the causal claim under the estimators they are gated to
    /// (UCC, overlap, E-value, sensitivity, Riesz).
    ///
    /// Placebo, dummy-outcome, and random-common-cause are omitted: under the OLS /
    /// linear-adjustment path they are gated to they have no power against a wrong
    /// claim (`informative: false`) and live in [`Self::stability_effect`] instead.
    #[must_use]
    pub fn falsification_effect() -> Self {
        Self::new()
            .with(ValidatorId::UnobservedCommonCause)
            .with(ValidatorId::Overlap)
            .with(ValidatorId::OverlapRule)
            .with(ValidatorId::EValue)
            .with(ValidatorId::LinearSensitivity)
            .with(ValidatorId::PartialLinearSensitivity)
            .with(ValidatorId::NonparametricSensitivity)
            .with(ValidatorId::Riesz)
    }

    /// Stability / sampling-variability diagnostics (not claim falsifiers).
    ///
    /// Includes placebo, dummy-outcome, RCC, and data-subset: under OLS these are
    /// orthogonal/near-null by construction and report `informative: false`.
    #[must_use]
    pub fn stability_effect() -> Self {
        Self::new()
            .with(ValidatorId::Placebo)
            .with(ValidatorId::RandomCommonCause)
            .with(ValidatorId::DummyOutcome)
            .with(ValidatorId::Bootstrap)
            .with(ValidatorId::DataSubset)
            .with(ValidatorId::Graph)
    }

    /// Full effect-validation set: [`Self::falsification_effect`] then [`Self::stability_effect`].
    #[must_use]
    pub fn full_effect() -> Self {
        let mut s = Self::falsification_effect();
        s.validators.extend(Self::stability_effect().validators);
        s
    }

    /// Overlap + sampling-stability of a reported *level* (not a treatment contrast).
    ///
    /// Omits placebo, dummy outcome, RCC/UCC, E-value, sensitivity, and Riesz.
    #[must_use]
    pub fn plugin_level_full() -> Self {
        Self::new()
            .with(ValidatorId::Overlap)
            .with(ValidatorId::OverlapRule)
            .with(ValidatorId::Bootstrap)
            .with(ValidatorId::DataSubset)
            .with(ValidatorId::Graph)
    }

    /// Run all configured validators.
    ///
    /// # Errors
    ///
    /// Propagates hard failures from applicable validators (not `NotApplicable` skips).
    pub fn run(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Vec<ValidationOutcome>, ValidationError> {
        let mut out = Vec::with_capacity(self.validators.len() + self.custom.len());
        for &id in &self.validators {
            out.push(computation_outcome(id, Self::run_one(id, problem, workspace, ctx))?);
        }
        out.extend(run_custom(&self.custom, |custom| custom.validate(problem, ctx))?);
        Ok(out)
    }

    /// Like [`Self::run`], reusing one warmed propensity workspace across every validator
    /// that fits a diagnostic propensity (Overlap, `OverlapRule`, Riesz), so the suite warms
    /// the GLM buffers once instead of fitting cold per refuter. Workspace reuse only:
    /// each refuter still runs its own fit over its own mask, so scores are never shared.
    ///
    /// # Errors
    ///
    /// Propagates hard failures from applicable validators.
    pub fn run_with_propensity(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        propensity: &mut antecedent_stats::PropensityWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Vec<ValidationOutcome>, ValidationError> {
        let mut out = Vec::with_capacity(self.validators.len() + self.custom.len());
        for &id in &self.validators {
            // Temporal designs fall through to `run_one`, which returns the same
            // `NotApplicable` outcomes as the plain `run` path for these validators.
            let outcome = (|| -> Result<ValidationOutcome, ValidationError> {
                Ok(match id {
                    ValidatorId::Overlap if problem.temporal.is_none() => overlap_as_outcome(
                        crate::overlap::OverlapRefuter::new()
                            .refute_with_propensity(problem, propensity),
                    )?,
                    ValidatorId::OverlapRule if problem.temporal.is_none() => {
                        ValidationOutcome::Report(
                            OverlapRuleRefuter::new()
                                .refute_with_propensity(problem, propensity)?,
                        )
                    }
                    ValidatorId::Riesz
                        if problem.temporal.is_none()
                            && crate::common::binary_treatment(problem)? =>
                    {
                        ValidationOutcome::Report(
                            RieszSensitivity::new().refute_with_propensity(problem, propensity)?,
                        )
                    }
                    _ => Self::run_one(id, problem, workspace, ctx)?,
                })
            })();
            out.push(computation_outcome(id, outcome)?);
        }
        out.extend(run_custom(&self.custom, |custom| custom.validate(problem, ctx))?);
        Ok(out)
    }

    /// Collect only successful [`RefutationReport`]s (drops `NotApplicable`).
    #[must_use]
    pub fn reports_only(outcomes: &[ValidationOutcome]) -> Vec<RefutationReport> {
        outcomes
            .iter()
            .filter_map(|o| match o {
                ValidationOutcome::Report(r) => Some(r.clone()),
                ValidationOutcome::NotApplicable { .. }
                | ValidationOutcome::CustomNotApplicable { .. }
                | ValidationOutcome::Failed { .. } => None,
            })
            .collect()
    }

    /// Collect only the skipped ([`ValidationOutcome::NotApplicable`]) outcomes, as
    /// `(validator, reason)` pairs.
    ///
    /// Pairs with [`Self::reports_only`]: together the two partition every outcome in
    /// `outcomes`. Callers that build a caller-visible refutation list from
    /// `reports_only` should also surface this list — e.g. as one diagnostic per
    /// skipped validator — so a per-run skip is visible rather than silently dropped.
    /// This is a *per-run, data-dependent* skip (this validator did not fit this run's
    /// problem), not the support matrix's permanent typed impossibility
    /// (`SupportRefusal::NotApplicable` / wire `not_applicable`); the same validator can
    /// run cleanly on a different dataset against the same licensed cell.
    #[must_use]
    pub fn not_applicable_only(outcomes: &[ValidationOutcome]) -> Vec<(ValidatorId, Arc<str>)> {
        outcomes
            .iter()
            .filter_map(|o| match o {
                ValidationOutcome::Report(_)
                | ValidationOutcome::Failed { .. }
                | ValidationOutcome::CustomNotApplicable { .. } => None,
                ValidationOutcome::NotApplicable { validator, reason } => {
                    Some((*validator, Arc::clone(reason)))
                }
            })
            .collect()
    }

    /// Run Bayesian validators that need a fitted posterior / prepared design.
    ///
    /// Frequentist `run` leaves Prior/Posterior predictive and `PriorSensitivity` as
    /// [`ValidationOutcome::NotApplicable`]; call this path from Bayesian execute.
    ///
    /// # Errors
    ///
    /// Propagates hard failures from applicable Bayesian validators.
    pub fn run_bayesian(
        &self,
        bayes: &mut BayesianSuiteContext<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<ValidationOutcome>, ValidationError> {
        // A custom validator needs a refutation problem this context does not carry.
        // Returning the Bayesian reports and omitting the custom check would hide a
        // requested validator, so the suite refuses instead.
        if !self.custom.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "custom validators need a refutation problem; the Bayesian suite refuses to drop them",
            });
        }
        let mut out = Vec::with_capacity(self.validators.len());
        for &id in &self.validators {
            out.push(computation_outcome(id, Self::run_one_bayesian(id, bayes, ctx))?);
        }
        Ok(out)
    }

    fn run_one(
        id: ValidatorId,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<ValidationOutcome, ValidationError> {
        if problem.effect_refit.is_some()
            && matches!(
                id,
                ValidatorId::LinearSensitivity
                    | ValidatorId::PartialLinearSensitivity
                    | ValidatorId::NonparametricSensitivity
                    | ValidatorId::Riesz
            )
        {
            return Ok(na(
                id,
                "single-regression sensitivity does not describe a composed sequential effect; use the sequential unobserved-common-cause perturbation",
            ));
        }
        let Some(entry) = ENTRIES.iter().find(|entry| entry.id == id) else {
            return Ok(na(
                id,
                "Bayesian PPC/prior-sensitivity/MCMC diagnostics require ValidationSuite::run_bayesian with a fitted posterior",
            ));
        };
        if let Some(reason) = (entry.gate)(problem)? {
            return Ok(na(id, reason));
        }
        let report = (entry.run)(problem, workspace, ctx);
        if entry.overlap_failures_are_outcomes {
            overlap_as_outcome(report)
        } else {
            Ok(ValidationOutcome::Report(report?))
        }
    }

    fn run_one_bayesian(
        id: ValidatorId,
        bayes: &mut BayesianSuiteContext<'_>,
        ctx: &ExecutionContext,
    ) -> Result<ValidationOutcome, ValidationError> {
        match id {
            ValidatorId::PriorPredictive => {
                let check = PriorPredictiveCheck::for_estimator(bayes.estimator, ctx);
                let prior = bayes.estimator.prior_in_force(bayes.prepared.design.ncols);
                let rep = check.check_with_prior(bayes.prepared, &prior, ctx)?;
                Ok(ValidationOutcome::Report(
                    rep.to_refutation_report(bayes.original_ate, bayes.ppc_alpha),
                ))
            }
            ValidatorId::PosteriorPredictive => {
                let check = PosteriorPredictiveCheck::for_estimator(bayes.estimator, ctx);
                let rep = check.check_with_context(bayes.prepared, bayes.posterior, ctx)?;
                Ok(ValidationOutcome::Report(
                    rep.to_refutation_report(bayes.original_ate, bayes.ppc_alpha),
                ))
            }
            ValidatorId::PriorSensitivity => {
                // The grid perturbs the prior the estimator fits with (a supplied or transferred
                // prior gets its variance-multiplier grid, none the isotropic scale grid).
                let sens = PriorSensitivity::for_estimator(bayes.estimator);
                let (summary, _posts) = sens.evaluate_in_force(
                    bayes.estimator,
                    bayes.prepared,
                    bayes.identification,
                    bayes.workspace,
                    ctx,
                )?;
                Ok(ValidationOutcome::Report(sens.to_report(&summary, bayes.original_ate)))
            }
            ValidatorId::McmcDiagnostics => {
                match McmcDiagnosticsCheck::new().check(bayes.posterior) {
                    Some(rep) => Ok(ValidationOutcome::Report(rep)),
                    None => Ok(na(
                        ValidatorId::McmcDiagnostics,
                        "MCMC diagnostics require an HMC/SMC posterior (Laplace/conjugate NotApplicable)",
                    )),
                }
            }
            other => {
                Ok(na(other, "validator is not a Bayesian diagnostic; use ValidationSuite::run"))
            }
        }
    }

    /// Prior predictive check only (cheap; no fitted posterior required beyond prepare).
    #[must_use]
    pub fn prior_predictive() -> Self {
        Self::new().with(ValidatorId::PriorPredictive)
    }
}

/// One built-in suite arm: a gate, a default `::new()` runner, and whether a
/// propensity failure is an outcome rather than a suite abort.
struct SuiteEntry {
    id: ValidatorId,
    gate: fn(&RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError>,
    run: fn(
        &RefutationProblem<'_>,
        &mut EstimationWorkspace,
        &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError>,
    overlap_failures_are_outcomes: bool,
}

fn linear_paths(problem: &RefutationProblem<'_>) -> (bool, bool) {
    let method = problem.estimand.method_kind().ok();
    let static_linear = problem.estimand.is_adjustment_shaped()
        && problem.estimator.is_none_or(|e| {
            // Conditional's licensed number is the interaction-model scalar at Ē[W].
            // That is an effect, so the effect suite runs; skipping it because the
            // estimator id is not `linear.adjustment.ate` is a demotion.
            matches!(e, "linear.adjustment.ate" | "conditional.linear.adjustment")
        });
    let temporal_linear = method == Some(antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded)
        && problem.temporal.is_some()
        && problem
            .estimator
            .is_none_or(|e| matches!(e, "temporal.linear.adjustment" | "bayesian.temporal.gcomp"));
    let linear_ok = static_linear || temporal_linear || problem.effect_refit.is_some();
    (static_linear, linear_ok)
}

fn allow(_problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    Ok(None)
}

fn gate_linear(
    problem: &RefutationProblem<'_>,
    reason: &'static str,
) -> Result<Option<&'static str>, ValidationError> {
    Ok((!linear_paths(problem).1).then_some(reason))
}

fn gate_static_linear(
    problem: &RefutationProblem<'_>,
    reason: &'static str,
) -> Result<Option<&'static str>, ValidationError> {
    Ok((!linear_paths(problem).0).then_some(reason))
}

fn gate_not_temporal(
    problem: &RefutationProblem<'_>,
    reason: &'static str,
) -> Result<Option<&'static str>, ValidationError> {
    Ok(problem.temporal.is_some().then_some(reason))
}

fn run_placebo(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&PlaceboTreatment::new(), problem, workspace, ctx)
}
fn run_rcc(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&RandomCommonCause::new(), problem, workspace, ctx)
}
fn run_bootstrap(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&BootstrapRefute::new(), problem, workspace, ctx)
}
fn run_ucc(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&UnobservedCommonCause::new(), problem, workspace, ctx)
}
fn run_overlap(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&OverlapRefuter::new(), problem, workspace, ctx)
}
fn run_overlap_rule(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&OverlapRuleRefuter::new(), problem, workspace, ctx)
}
fn run_subset(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&DataSubsetRefuter::new(), problem, workspace, ctx)
}
fn run_dummy(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&DummyOutcome::new(), problem, workspace, ctx)
}
fn run_evalue(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&EValue::new(), problem, workspace, ctx)
}
fn run_graph(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&GraphRefuter::new(), problem, workspace, ctx)
}
fn run_linear(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&LinearSensitivity::new(), problem, workspace, ctx)
}
fn run_partial(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&PartialLinearSensitivity::new(), problem, workspace, ctx)
}
fn run_nonparametric(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&NonparametricSensitivity::new(), problem, workspace, ctx)
}
fn run_riesz(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    run_validator(&RieszSensitivity::new(), problem, workspace, ctx)
}

fn gate_placebo(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "PlaceboTreatment requires backdoor.adjustment + linear path \
         (or temporal.backdoor.unfolded + temporal linear path)",
    )
}
fn gate_rcc(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "RandomCommonCause requires backdoor.adjustment + linear path \
         (or temporal.backdoor.unfolded + temporal linear path)",
    )
}
fn gate_bootstrap(
    problem: &RefutationProblem<'_>,
) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "BootstrapCiCoverage requires backdoor.adjustment + linear path \
         (or temporal.backdoor.unfolded + temporal linear path)",
    )
}
fn gate_ucc(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "UnobservedCommonCause requires backdoor.adjustment or temporal.backdoor.unfolded",
    )
}
fn gate_overlap(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_not_temporal(
        problem,
        "OverlapRefuter not applicable to temporal unfolded designs \
         (propensity uses schema adjustment columns)",
    )
}
fn gate_overlap_rule(
    problem: &RefutationProblem<'_>,
) -> Result<Option<&'static str>, ValidationError> {
    gate_not_temporal(problem, "OverlapRuleRefuter not applicable to temporal unfolded designs")
}
fn gate_subset(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "DataSubsetRefuter requires backdoor.adjustment + linear path \
         (or temporal.backdoor.unfolded + temporal linear path)",
    )
}
fn gate_dummy(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "DummyOutcome requires backdoor.adjustment + linear path \
         (or temporal.backdoor.unfolded + temporal linear path)",
    )
}
fn gate_graph(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_static_linear(
        problem,
        "DropAdjustmentCovariate requires static backdoor.adjustment + linear path \
         (not applicable to temporal unfolded designs)",
    )
}
fn gate_linear_sens(
    problem: &RefutationProblem<'_>,
) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "LinearSensitivity requires backdoor.adjustment or temporal.backdoor.unfolded",
    )
}
fn gate_partial(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    gate_linear(
        problem,
        "PartialLinearSensitivity requires backdoor.adjustment or temporal.backdoor.unfolded",
    )
}
fn gate_riesz(problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    if problem.temporal.is_some() {
        return Ok(Some("RieszSensitivity not applicable to temporal unfolded designs"));
    }
    if !crate::common::binary_treatment(problem)? {
        return Ok(Some(
            "RieszSensitivity requires binary treatment; continuous support is assessed by the overlap validators",
        ));
    }
    Ok(None)
}

const ENTRIES: &[SuiteEntry] = &[
    SuiteEntry {
        id: ValidatorId::Placebo,
        gate: gate_placebo,
        run: run_placebo,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::RandomCommonCause,
        gate: gate_rcc,
        run: run_rcc,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::Bootstrap,
        gate: gate_bootstrap,
        run: run_bootstrap,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::UnobservedCommonCause,
        gate: gate_ucc,
        run: run_ucc,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::Overlap,
        gate: gate_overlap,
        run: run_overlap,
        overlap_failures_are_outcomes: true,
    },
    SuiteEntry {
        id: ValidatorId::OverlapRule,
        gate: gate_overlap_rule,
        run: run_overlap_rule,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::DataSubset,
        gate: gate_subset,
        run: run_subset,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::DummyOutcome,
        gate: gate_dummy,
        run: run_dummy,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::EValue,
        gate: allow,
        run: run_evalue,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::Graph,
        gate: gate_graph,
        run: run_graph,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::LinearSensitivity,
        gate: gate_linear_sens,
        run: run_linear,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::PartialLinearSensitivity,
        gate: gate_partial,
        run: run_partial,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::NonparametricSensitivity,
        gate: allow,
        run: run_nonparametric,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::Riesz,
        gate: gate_riesz,
        run: run_riesz,
        overlap_failures_are_outcomes: false,
    },
];

fn run_custom(
    custom: &[Arc<dyn CustomEffectValidator>],
    mut validate: impl FnMut(&dyn CustomEffectValidator) -> Result<RefutationReport, ValidationError>,
) -> Result<Vec<ValidationOutcome>, ValidationError> {
    custom.iter().map(|custom| custom_outcome(custom.as_ref(), validate(custom.as_ref()))).collect()
}

fn custom_outcome(
    custom: &dyn CustomEffectValidator,
    result: Result<RefutationReport, ValidationError>,
) -> Result<ValidationOutcome, ValidationError> {
    match result {
        Ok(report) => Ok(ValidationOutcome::Report(report)),
        Err(ValidationError::Cancelled) => Err(ValidationError::Cancelled),
        Err(ValidationError::NotApplicable { message }) => {
            Ok(ValidationOutcome::CustomNotApplicable {
                name: Arc::from(custom.name()),
                reason: Arc::from(message),
            })
        }
        Err(error) => Ok(ValidationOutcome::Failed {
            validator: Arc::from(custom.name()),
            reason: Arc::from(error.to_string()),
        }),
    }
}

fn na(id: ValidatorId, reason: &str) -> ValidationOutcome {
    ValidationOutcome::NotApplicable { validator: id, reason: Arc::from(reason) }
}

fn computation_outcome(
    id: ValidatorId,
    result: Result<ValidationOutcome, ValidationError>,
) -> Result<ValidationOutcome, ValidationError> {
    match result {
        Ok(outcome) => Ok(outcome),
        Err(ValidationError::Cancelled) => Err(ValidationError::Cancelled),
        Err(ValidationError::NotApplicable { message }) => Ok(na(id, message)),
        Err(error) => Ok(ValidationOutcome::Failed {
            validator: Arc::from(id.as_str()),
            reason: Arc::from(error.to_string()),
        }),
    }
}

/// Overlap propensity failures are typed per-validator results, not claim aborts.
/// Cancellation is not a failure of the validator and still aborts the run.
fn overlap_as_outcome(
    result: Result<RefutationReport, ValidationError>,
) -> Result<ValidationOutcome, ValidationError> {
    match result {
        Ok(report) => Ok(ValidationOutcome::Report(report)),
        Err(ValidationError::Cancelled) => Err(ValidationError::Cancelled),
        Err(err) => Ok(ValidationOutcome::Failed {
            validator: Arc::from(ValidatorId::Overlap.as_str()),
            reason: Arc::from(err.to_string()),
        }),
    }
}

#[cfg(test)]
mod failure_tests {
    use super::*;

    #[test]
    fn overlap_failures_are_typed_but_cancellation_propagates() {
        assert!(matches!(
            overlap_as_outcome(Err(ValidationError::Cancelled)),
            Err(ValidationError::Cancelled)
        ));
        assert!(matches!(
            overlap_as_outcome(Err(ValidationError::estimation_msg("singular fit"))),
            Ok(ValidationOutcome::Failed { .. })
        ));
    }

    #[test]
    fn computation_failures_keep_cancellation_and_incompatibility_distinct() {
        assert!(matches!(
            computation_outcome(ValidatorId::Overlap, Err(ValidationError::Cancelled)),
            Err(ValidationError::Cancelled)
        ));
        assert!(matches!(
            computation_outcome(
                ValidatorId::EValue,
                Err(ValidationError::NotApplicable { message: "constant outcome" })
            ),
            Ok(ValidationOutcome::NotApplicable { .. })
        ));
        assert!(matches!(
            computation_outcome(
                ValidatorId::Overlap,
                Err(ValidationError::estimation_msg("singular fit"))
            ),
            Ok(ValidationOutcome::Failed { .. })
        ));
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{
        EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy,
    };
    use antecedent_expr::ExprId;
    use antecedent_identify::IdentifiedEstimand;

    use super::*;
    use crate::common::RefutationProblem;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        let n = 120usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn full_suite_runs_applicable_validators() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let outcomes = ValidationSuite::full_effect().run(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(outcomes.len(), 14);
        // Every outcome is exactly one of report / skip / operational failure.
        let reports = ValidationSuite::reports_only(&outcomes);
        let skipped = ValidationSuite::not_applicable_only(&outcomes);
        let failed =
            outcomes.iter().filter(|o| matches!(o, ValidationOutcome::Failed { .. })).count();
        assert_eq!(reports.len() + skipped.len() + failed, 14);
        // No validator may fail operationally on this well-posed toy.
        assert_eq!(failed, 0, "{outcomes:?}");
        // The noise-free toy leaves no residual outcome variation, so no finite confounder
        // strength can explain the effect away: the E-value is infinite, not a finite artefact of
        // the outcome's marginal spread.
        let evalue = reports
            .iter()
            .find(|r| r.refuter.as_ref() == "sensitivity.evalue")
            .expect("e-value report");
        assert!(evalue.comparison.is_infinite(), "e-value={}", evalue.comparison);
    }

    #[test]
    fn custom_not_applicable_does_not_abort_the_suite() {
        struct Skip;
        impl CustomEffectValidator for Skip {
            fn name(&self) -> &'static str {
                "skip"
            }
            fn validate(
                &self,
                _: &RefutationProblem<'_>,
                _: &ExecutionContext,
            ) -> Result<RefutationReport, ValidationError> {
                Err(ValidationError::NotApplicable { message: "does not apply" })
            }
        }
        struct OkReport;
        impl CustomEffectValidator for OkReport {
            fn name(&self) -> &'static str {
                "ok"
            }
            fn validate(
                &self,
                _: &RefutationProblem<'_>,
                _: &ExecutionContext,
            ) -> Result<RefutationReport, ValidationError> {
                Ok(RefutationReport::builder("ok")
                    .original_ate(0.0)
                    .refuted_ate(0.0)
                    .comparison(0.0)
                    .informative(false)
                    .passed(true)
                    .failure_condition(None)
                    .replicates(0)
                    .build())
            }
        }
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let original =
            EffectEstimate::new(0.0, 1.0, AssumptionSet::new(), OverlapPolicy::ExplicitOverride);
        let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);
        let ctx = ExecutionContext::for_tests(1);
        let mut ws = EstimationWorkspace::default();
        let outcomes = ValidationSuite::new()
            .with_custom(Arc::new(Skip))
            .with_custom(Arc::new(OkReport))
            .run(&problem, &mut ws, &ctx)
            .unwrap();
        assert!(matches!(
            &outcomes[0],
            ValidationOutcome::CustomNotApplicable { name, .. } if name.as_ref() == "skip"
        ));
        assert!(matches!(&outcomes[1], ValidationOutcome::Report(_)));
    }

    #[test]
    fn plugin_level_full_omits_contrast_shaped_validators() {
        let ids: Vec<_> = ValidationSuite::plugin_level_full().validators;
        assert!(ids.contains(&ValidatorId::Overlap));
        assert!(ids.contains(&ValidatorId::Bootstrap));
        for banned in [
            ValidatorId::Placebo,
            ValidatorId::RandomCommonCause,
            ValidatorId::UnobservedCommonCause,
            ValidatorId::DummyOutcome,
            ValidatorId::EValue,
            ValidatorId::LinearSensitivity,
            ValidatorId::PartialLinearSensitivity,
            ValidatorId::NonparametricSensitivity,
            ValidatorId::Riesz,
        ] {
            assert!(!ids.contains(&banned), "plugin_level_full must omit {banned:?}");
        }
    }

    #[test]
    fn falsification_effect_omits_ols_powerless_refuters() {
        let ids = ValidationSuite::falsification_effect().validators;
        for banned in [
            ValidatorId::Placebo,
            ValidatorId::RandomCommonCause,
            ValidatorId::DummyOutcome,
            ValidatorId::DataSubset,
        ] {
            assert!(
                !ids.contains(&banned),
                "falsification_effect must not include OLS-powerless {banned:?}"
            );
        }
        let stability = ValidationSuite::stability_effect().validators;
        for required in [
            ValidatorId::Placebo,
            ValidatorId::RandomCommonCause,
            ValidatorId::DummyOutcome,
            ValidatorId::DataSubset,
        ] {
            assert!(
                stability.contains(&required),
                "stability_effect must house OLS-powerless {required:?}"
            );
        }
        // Real falsifiers stay in the falsification set.
        for required in [
            ValidatorId::UnobservedCommonCause,
            ValidatorId::Overlap,
            ValidatorId::EValue,
            ValidatorId::Riesz,
        ] {
            assert!(ids.contains(&required), "falsification_effect must keep {required:?}");
        }
    }
}

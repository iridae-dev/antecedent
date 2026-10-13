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
use crate::validator::{PreparedRefutation, Validator, run_validator};

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

/// A built-in validator the caller has already built.
///
/// [`ValidationSuite::with_configured`] stores this value and does not construct
/// another arm. [`ValidationSuite::with`] stores that arm's `::new()` default
/// when the slot is empty. Bayesian arms stay `for_estimator` until a check or
/// grid is stored here.
#[derive(Clone, Debug)]
pub enum ConfiguredValidator {
    /// Placebo treatment.
    Placebo(PlaceboTreatment),
    /// Random common cause.
    RandomCommonCause(RandomCommonCause),
    /// Bootstrap CI coverage.
    Bootstrap(BootstrapRefute),
    /// Unobserved common cause.
    UnobservedCommonCause(UnobservedCommonCause),
    /// Overlap / positivity.
    Overlap(OverlapRefuter),
    /// Overlap-rule / trimming.
    OverlapRule(OverlapRuleRefuter),
    /// Data subset.
    DataSubset(DataSubsetRefuter),
    /// Dummy outcome.
    DummyOutcome(DummyOutcome),
    /// E-value.
    EValue(EValue),
    /// Leave-one-out adjustment set.
    Graph(GraphRefuter),
    /// Linear sensitivity.
    LinearSensitivity(LinearSensitivity),
    /// Partial-linear sensitivity.
    PartialLinearSensitivity(PartialLinearSensitivity),
    /// Nonparametric sensitivity.
    NonparametricSensitivity(NonparametricSensitivity),
    /// Riesz-representer sensitivity.
    Riesz(RieszSensitivity),
    /// Prior predictive check. Replaces `for_estimator`, including family and seed.
    PriorPredictive(PriorPredictiveCheck),
    /// Posterior predictive check. Replaces `for_estimator`, including family and seed.
    PosteriorPredictive(PosteriorPredictiveCheck),
    /// Prior-sensitivity grid. Replaces `for_estimator`.
    PriorSensitivity(PriorSensitivity),
}

impl ConfiguredValidator {
    fn id(&self) -> ValidatorId {
        match self {
            Self::Placebo(_) => ValidatorId::Placebo,
            Self::RandomCommonCause(_) => ValidatorId::RandomCommonCause,
            Self::Bootstrap(_) => ValidatorId::Bootstrap,
            Self::UnobservedCommonCause(_) => ValidatorId::UnobservedCommonCause,
            Self::Overlap(_) => ValidatorId::Overlap,
            Self::OverlapRule(_) => ValidatorId::OverlapRule,
            Self::DataSubset(_) => ValidatorId::DataSubset,
            Self::DummyOutcome(_) => ValidatorId::DummyOutcome,
            Self::EValue(_) => ValidatorId::EValue,
            Self::Graph(_) => ValidatorId::Graph,
            Self::LinearSensitivity(_) => ValidatorId::LinearSensitivity,
            Self::PartialLinearSensitivity(_) => ValidatorId::PartialLinearSensitivity,
            Self::NonparametricSensitivity(_) => ValidatorId::NonparametricSensitivity,
            Self::Riesz(_) => ValidatorId::Riesz,
            Self::PriorPredictive(_) => ValidatorId::PriorPredictive,
            Self::PosteriorPredictive(_) => ValidatorId::PosteriorPredictive,
            Self::PriorSensitivity(_) => ValidatorId::PriorSensitivity,
        }
    }

    fn store(self, arms: &mut SuiteArms) {
        match self {
            Self::Placebo(v) => arms.placebo = Some(v),
            Self::RandomCommonCause(v) => arms.random_common_cause = Some(v),
            Self::Bootstrap(v) => arms.bootstrap = Some(v),
            Self::UnobservedCommonCause(v) => arms.unobserved_common_cause = Some(v),
            Self::Overlap(v) => arms.overlap = Some(v),
            Self::OverlapRule(v) => arms.overlap_rule = Some(v),
            Self::DataSubset(v) => arms.data_subset = Some(v),
            Self::DummyOutcome(v) => arms.dummy_outcome = Some(v),
            Self::EValue(v) => arms.evalue = Some(v),
            Self::Graph(v) => arms.graph = Some(v),
            Self::LinearSensitivity(v) => arms.linear = Some(v),
            Self::PartialLinearSensitivity(v) => arms.partial = Some(v),
            Self::NonparametricSensitivity(v) => arms.nonparametric = Some(v),
            Self::Riesz(v) => arms.riesz = Some(v),
            Self::PriorPredictive(v) => arms.prior_predictive = Some(v),
            Self::PosteriorPredictive(v) => arms.posterior_predictive = Some(v),
            Self::PriorSensitivity(v) => arms.prior_sensitivity = Some(v),
        }
    }
}

/// Stored suite arms. A frequentist slot holds either the caller's value or the
/// `::new()` default installed by [`ValidationSuite::with`]. A Bayesian slot
/// stays empty until the caller stores a check or grid.
#[derive(Clone, Debug, Default)]
struct SuiteArms {
    placebo: Option<PlaceboTreatment>,
    random_common_cause: Option<RandomCommonCause>,
    bootstrap: Option<BootstrapRefute>,
    unobserved_common_cause: Option<UnobservedCommonCause>,
    overlap: Option<OverlapRefuter>,
    overlap_rule: Option<OverlapRuleRefuter>,
    data_subset: Option<DataSubsetRefuter>,
    dummy_outcome: Option<DummyOutcome>,
    evalue: Option<EValue>,
    graph: Option<GraphRefuter>,
    linear: Option<LinearSensitivity>,
    partial: Option<PartialLinearSensitivity>,
    nonparametric: Option<NonparametricSensitivity>,
    riesz: Option<RieszSensitivity>,
    prior_predictive: Option<PriorPredictiveCheck>,
    posterior_predictive: Option<PosteriorPredictiveCheck>,
    prior_sensitivity: Option<PriorSensitivity>,
}

/// Ordered suite of validators.
#[derive(Clone, Default)]
pub struct ValidationSuite {
    validators: Vec<ValidatorId>,
    custom: Vec<Arc<dyn CustomEffectValidator>>,
    arms: SuiteArms,
}

impl std::fmt::Debug for ValidationSuite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationSuite")
            .field("validators", &self.validators)
            .field("custom", &self.custom.len())
            .field("configured", &self.arms)
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
    ///
    /// When this arm has no stored value, a frequentist id stores the same
    /// `::new()` value the runner used to build for an empty slot. An id that
    /// already holds a caller-built value is left alone. Bayesian ids are not
    /// built here.
    #[must_use]
    pub fn with(mut self, id: ValidatorId) -> Self {
        self.validators.push(id);
        self.store_default_if_absent(id);
        self
    }

    /// Store `validator` and run that value.
    ///
    /// The suite does not build a replacement. Appends the arm when [`Self::with`]
    /// has not already listed it. A later call for the same arm replaces the
    /// stored value. [`Self::with`] still appends a duplicate id, and every copy
    /// of that id uses this value.
    #[must_use]
    pub fn with_configured(mut self, validator: ConfiguredValidator) -> Self {
        let id = validator.id();
        if !self.validators.contains(&id) {
            self.validators.push(id);
        }
        validator.store(&mut self.arms);
        self
    }

    /// Install `id`'s `::new()` arm when the caller has not already stored one.
    ///
    /// Bayesian checks need the fitted estimator, so those slots stay empty and
    /// [`Self::run_one_bayesian`] still builds `for_estimator`.
    fn store_default_if_absent(&mut self, id: ValidatorId) {
        let default = match id {
            ValidatorId::Placebo => self
                .arms
                .placebo
                .is_none()
                .then(|| ConfiguredValidator::Placebo(PlaceboTreatment::new())),
            ValidatorId::RandomCommonCause => self
                .arms
                .random_common_cause
                .is_none()
                .then(|| ConfiguredValidator::RandomCommonCause(RandomCommonCause::new())),
            ValidatorId::Bootstrap => self
                .arms
                .bootstrap
                .is_none()
                .then(|| ConfiguredValidator::Bootstrap(BootstrapRefute::new())),
            ValidatorId::UnobservedCommonCause => {
                self.arms.unobserved_common_cause.is_none().then(|| {
                    ConfiguredValidator::UnobservedCommonCause(UnobservedCommonCause::new())
                })
            }
            ValidatorId::Overlap => self
                .arms
                .overlap
                .is_none()
                .then(|| ConfiguredValidator::Overlap(OverlapRefuter::new())),
            ValidatorId::OverlapRule => self
                .arms
                .overlap_rule
                .is_none()
                .then(|| ConfiguredValidator::OverlapRule(OverlapRuleRefuter::new())),
            ValidatorId::DataSubset => self
                .arms
                .data_subset
                .is_none()
                .then(|| ConfiguredValidator::DataSubset(DataSubsetRefuter::new())),
            ValidatorId::DummyOutcome => self
                .arms
                .dummy_outcome
                .is_none()
                .then(|| ConfiguredValidator::DummyOutcome(DummyOutcome::new())),
            ValidatorId::EValue => {
                self.arms.evalue.is_none().then(|| ConfiguredValidator::EValue(EValue::new()))
            }
            ValidatorId::Graph => {
                self.arms.graph.is_none().then(|| ConfiguredValidator::Graph(GraphRefuter::new()))
            }
            ValidatorId::LinearSensitivity => self
                .arms
                .linear
                .is_none()
                .then(|| ConfiguredValidator::LinearSensitivity(LinearSensitivity::new())),
            ValidatorId::PartialLinearSensitivity => self.arms.partial.is_none().then(|| {
                ConfiguredValidator::PartialLinearSensitivity(PartialLinearSensitivity::new())
            }),
            ValidatorId::NonparametricSensitivity => self.arms.nonparametric.is_none().then(|| {
                ConfiguredValidator::NonparametricSensitivity(NonparametricSensitivity::new())
            }),
            ValidatorId::Riesz => self
                .arms
                .riesz
                .is_none()
                .then(|| ConfiguredValidator::Riesz(RieszSensitivity::new())),
            ValidatorId::PriorPredictive
            | ValidatorId::PosteriorPredictive
            | ValidatorId::PriorSensitivity
            | ValidatorId::McmcDiagnostics => None,
        };
        if let Some(default) = default {
            default.store(&mut self.arms);
        }
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
            out.push(computation_outcome(id, self.run_one(id, problem, workspace, ctx))?);
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
                    ValidatorId::Overlap if problem.temporal.is_none() => {
                        let result = match &self.arms.overlap {
                            Some(overlap) => overlap.refute_with_propensity(problem, propensity),
                            None => {
                                OverlapRefuter::new().refute_with_propensity(problem, propensity)
                            }
                        };
                        overlap_as_outcome(result)?
                    }
                    ValidatorId::OverlapRule if problem.temporal.is_none() => {
                        let report = match &self.arms.overlap_rule {
                            Some(rule) => rule.refute_with_propensity(problem, propensity)?,
                            None => OverlapRuleRefuter::new()
                                .refute_with_propensity(problem, propensity)?,
                        };
                        ValidationOutcome::Report(report)
                    }
                    ValidatorId::Riesz
                        if problem.temporal.is_none()
                            && crate::common::binary_treatment(problem)? =>
                    {
                        let report = match &self.arms.riesz {
                            Some(riesz) => riesz.refute_with_propensity(problem, propensity)?,
                            None => RieszSensitivity::new()
                                .refute_with_propensity(problem, propensity)?,
                        };
                        ValidationOutcome::Report(report)
                    }
                    _ => self.run_one(id, problem, workspace, ctx)?,
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
            out.push(computation_outcome(id, self.run_one_bayesian(id, bayes, ctx))?);
        }
        Ok(out)
    }

    fn run_one(
        &self,
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
        let report = self.execute(id, problem, workspace, ctx);
        if entry.overlap_failures_are_outcomes {
            overlap_as_outcome(report)
        } else {
            Ok(ValidationOutcome::Report(report?))
        }
    }

    fn run_one_bayesian(
        &self,
        id: ValidatorId,
        bayes: &mut BayesianSuiteContext<'_>,
        ctx: &ExecutionContext,
    ) -> Result<ValidationOutcome, ValidationError> {
        match id {
            ValidatorId::PriorPredictive => {
                let owned;
                let check = if let Some(check) = &self.arms.prior_predictive {
                    check
                } else {
                    owned = PriorPredictiveCheck::for_estimator(bayes.estimator, ctx);
                    &owned
                };
                let prior = bayes.estimator.prior_in_force(bayes.prepared.design.ncols);
                let rep = check.check_with_prior(bayes.prepared, &prior, ctx)?;
                Ok(ValidationOutcome::Report(
                    rep.to_refutation_report(bayes.original_ate, bayes.ppc_alpha),
                ))
            }
            ValidatorId::PosteriorPredictive => {
                let owned;
                let check = if let Some(check) = &self.arms.posterior_predictive {
                    check
                } else {
                    owned = PosteriorPredictiveCheck::for_estimator(bayes.estimator, ctx);
                    &owned
                };
                let rep = check.check_with_context(bayes.prepared, bayes.posterior, ctx)?;
                Ok(ValidationOutcome::Report(
                    rep.to_refutation_report(bayes.original_ate, bayes.ppc_alpha),
                ))
            }
            ValidatorId::PriorSensitivity => {
                // The grid perturbs the prior the estimator fits with (a supplied or transferred
                // prior gets its variance-multiplier grid, none the isotropic scale grid).
                // A stored grid replaces that choice, including which axis it perturbs.
                // An empty slot still builds `for_estimator`.
                let owned;
                let sens = if let Some(sens) = &self.arms.prior_sensitivity {
                    sens
                } else {
                    owned = PriorSensitivity::for_estimator(bayes.estimator);
                    &owned
                };
                let (summary, _posts) = sens.evaluate_in_force(
                    bayes.estimator,
                    bayes.prepared,
                    bayes.identification,
                    bayes.workspace,
                    ctx,
                )?;
                Ok(ValidationOutcome::Report(sens.to_report(&summary, bayes.original_ate)))
            }
            ValidatorId::McmcDiagnostics => match McmcDiagnosticsCheck::check(bayes.posterior) {
                Some(rep) => Ok(ValidationOutcome::Report(rep)),
                None => Ok(na(
                    ValidatorId::McmcDiagnostics,
                    "MCMC diagnostics require an HMC/SMC posterior (Laplace/conjugate NotApplicable)",
                )),
            },
            other => {
                Ok(na(other, "validator is not a Bayesian diagnostic; use ValidationSuite::run"))
            }
        }
    }

    fn execute(
        &self,
        id: ValidatorId,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        match id {
            ValidatorId::Placebo => run_stored(
                self.arms.placebo.as_ref(),
                PlaceboTreatment::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::RandomCommonCause => run_stored(
                self.arms.random_common_cause.as_ref(),
                RandomCommonCause::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::Bootstrap => run_stored(
                self.arms.bootstrap.as_ref(),
                BootstrapRefute::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::UnobservedCommonCause => run_stored(
                self.arms.unobserved_common_cause.as_ref(),
                UnobservedCommonCause::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::Overlap => {
                run_stored(self.arms.overlap.as_ref(), OverlapRefuter::new, problem, workspace, ctx)
            }
            ValidatorId::OverlapRule => run_stored(
                self.arms.overlap_rule.as_ref(),
                OverlapRuleRefuter::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::DataSubset => run_stored(
                self.arms.data_subset.as_ref(),
                DataSubsetRefuter::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::DummyOutcome => run_stored(
                self.arms.dummy_outcome.as_ref(),
                DummyOutcome::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::EValue => {
                run_stored(self.arms.evalue.as_ref(), EValue::new, problem, workspace, ctx)
            }
            ValidatorId::Graph => {
                run_stored(self.arms.graph.as_ref(), GraphRefuter::new, problem, workspace, ctx)
            }
            ValidatorId::LinearSensitivity => run_stored(
                self.arms.linear.as_ref(),
                LinearSensitivity::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::PartialLinearSensitivity => run_stored(
                self.arms.partial.as_ref(),
                PartialLinearSensitivity::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::NonparametricSensitivity => run_stored(
                self.arms.nonparametric.as_ref(),
                NonparametricSensitivity::new,
                problem,
                workspace,
                ctx,
            ),
            ValidatorId::Riesz => {
                run_stored(self.arms.riesz.as_ref(), RieszSensitivity::new, problem, workspace, ctx)
            }
            ValidatorId::PriorPredictive
            | ValidatorId::PosteriorPredictive
            | ValidatorId::PriorSensitivity
            | ValidatorId::McmcDiagnostics => Err(ValidationError::NotApplicable {
                message: "Bayesian PPC/prior-sensitivity/MCMC diagnostics require ValidationSuite::run_bayesian with a fitted posterior",
            }),
        }
    }

    /// Prior predictive check only (cheap; no fitted posterior required beyond prepare).
    #[must_use]
    pub fn prior_predictive() -> Self {
        Self::new().with(ValidatorId::PriorPredictive)
    }
}

/// Run the stored arm. `default` is the same constructor an empty slot has
/// always used, and it runs only when the caller did not store a value.
fn run_stored<'a, T, F>(
    stored: Option<&T>,
    default: F,
    problem: &RefutationProblem<'a>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError>
where
    T: Validator<
            RefutationProblem<'a>,
            Prepared = PreparedRefutation<'a>,
            Report = RefutationReport,
        >,
    F: FnOnce() -> T,
{
    match stored {
        Some(validator) => run_validator(validator, problem, workspace, ctx),
        None => run_validator(&default(), problem, workspace, ctx),
    }
}

/// One built-in suite arm: a gate, and whether a propensity failure is an
/// outcome rather than a suite abort. The runner is [`ValidationSuite::execute`],
/// which uses the stored arm. An empty frequentist slot still runs `::new()`.
struct SuiteEntry {
    id: ValidatorId,
    gate: fn(&RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError>,
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

#[allow(clippy::unnecessary_wraps, reason = "suite gate callbacks share one fallible signature")]
fn allow(_problem: &RefutationProblem<'_>) -> Result<Option<&'static str>, ValidationError> {
    Ok(None)
}

#[allow(clippy::unnecessary_wraps, reason = "suite gate callbacks share one fallible signature")]
fn gate_linear(
    problem: &RefutationProblem<'_>,
    reason: &'static str,
) -> Result<Option<&'static str>, ValidationError> {
    Ok((!linear_paths(problem).1).then_some(reason))
}

#[allow(clippy::unnecessary_wraps, reason = "suite gate callbacks share one fallible signature")]
fn gate_static_linear(
    problem: &RefutationProblem<'_>,
    reason: &'static str,
) -> Result<Option<&'static str>, ValidationError> {
    Ok((!linear_paths(problem).0).then_some(reason))
}

#[allow(clippy::unnecessary_wraps, reason = "suite gate callbacks share one fallible signature")]
fn gate_not_temporal(
    problem: &RefutationProblem<'_>,
    reason: &'static str,
) -> Result<Option<&'static str>, ValidationError> {
    Ok(problem.temporal.is_some().then_some(reason))
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
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::RandomCommonCause,
        gate: gate_rcc,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::Bootstrap,
        gate: gate_bootstrap,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::UnobservedCommonCause,
        gate: gate_ucc,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::Overlap,
        gate: gate_overlap,
        overlap_failures_are_outcomes: true,
    },
    SuiteEntry {
        id: ValidatorId::OverlapRule,
        gate: gate_overlap_rule,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::DataSubset,
        gate: gate_subset,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::DummyOutcome,
        gate: gate_dummy,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry { id: ValidatorId::EValue, gate: allow, overlap_failures_are_outcomes: false },
    SuiteEntry { id: ValidatorId::Graph, gate: gate_graph, overlap_failures_are_outcomes: false },
    SuiteEntry {
        id: ValidatorId::LinearSensitivity,
        gate: gate_linear_sens,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::PartialLinearSensitivity,
        gate: gate_partial,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry {
        id: ValidatorId::NonparametricSensitivity,
        gate: allow,
        overlap_failures_are_outcomes: false,
    },
    SuiteEntry { id: ValidatorId::Riesz, gate: gate_riesz, overlap_failures_are_outcomes: false },
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
    use crate::placebo::PlaceboTreatment;

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

    #[test]
    fn configured_placebo_replaces_the_default_replicate_count() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let original =
            EffectEstimate::new(0.0, 1.0, AssumptionSet::new(), OverlapPolicy::ExplicitOverride);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let ctx = ExecutionContext::for_tests(1);
        let mut ws = EstimationWorkspace::default();
        let placebo = PlaceboTreatment { replicates: 1, ..PlaceboTreatment::new() };
        let outcomes = ValidationSuite::new()
            .with_configured(ConfiguredValidator::Placebo(placebo))
            .run(&problem, &mut ws, &ctx)
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        match &outcomes[0] {
            ValidationOutcome::NotApplicable { validator, reason } => {
                assert_eq!(*validator, ValidatorId::Placebo);
                assert!(
                    reason.contains("replicates >= 2"),
                    "configured replicate count was ignored: {reason}"
                );
            }
            other => panic!("expected the configured bound, got {other:?}"),
        }
    }

    #[test]
    fn with_stores_the_new_default_and_keeps_a_caller_built_arm() {
        let installed = ValidationSuite::new().with(ValidatorId::Placebo);
        let stored = installed.arms.placebo.as_ref().expect("default arm");
        let fresh = PlaceboTreatment::new();
        assert_eq!(stored.replicates, fresh.replicates);
        assert_eq!(stored.alpha.to_bits(), fresh.alpha.to_bits());
        assert_eq!(stored.mode, fresh.mode);
        assert_eq!(stored.estimator.bootstrap_replicates, fresh.estimator.bootstrap_replicates);

        let custom = PlaceboTreatment { replicates: 7, ..PlaceboTreatment::new() };
        let kept = ValidationSuite::new()
            .with_configured(ConfiguredValidator::Placebo(custom))
            .with(ValidatorId::Placebo);
        assert_eq!(kept.arms.placebo.as_ref().expect("caller arm").replicates, 7);
        assert_eq!(kept.validators, vec![ValidatorId::Placebo, ValidatorId::Placebo]);

        let replaced = ValidationSuite::new().with(ValidatorId::Placebo).with_configured(
            ConfiguredValidator::Placebo(PlaceboTreatment {
                replicates: 3,
                ..PlaceboTreatment::new()
            }),
        );
        assert_eq!(replaced.arms.placebo.as_ref().expect("replaced arm").replicates, 3);
        assert_eq!(replaced.validators, vec![ValidatorId::Placebo]);

        let bayes = ValidationSuite::new().with(ValidatorId::PriorPredictive);
        assert!(bayes.arms.prior_predictive.is_none());
        assert!(bayes.arms.posterior_predictive.is_none());
        assert!(bayes.arms.prior_sensitivity.is_none());
    }
}

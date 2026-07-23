//! Validation suite orchestration.
//!
//! Runs requested validators, returning explicit [`ValidationOutcome::NotApplicable`] when a
//! check is incompatible with the estimator/estimand (rather than failing the whole suite).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::unused_self,
    clippy::too_many_lines
)]

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
use crate::reisz::ReiszSensitivity;
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
    /// Reisz-representer sensitivity.
    Reisz,
    /// Prior predictive check (Bayesian).
    PriorPredictive,
    /// Posterior predictive check (Bayesian).
    PosteriorPredictive,
    /// Prior sensitivity grid (Bayesian).
    PriorSensitivity,
    /// MCMC ESS / R-hat / divergence diagnostics (Bayesian HMC/SMC).
    McmcDiagnostics,
}

/// Outcome of one validator in a suite.
#[derive(Clone, Debug)]
pub enum ValidationOutcome {
    /// Validator ran and produced a report.
    Report(RefutationReport),
    /// Validator was requested but is incompatible with this problem.
    NotApplicable {
        /// Validator id.
        validator: ValidatorId,
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

    /// Full effect-validation set.
    #[must_use]
    pub fn full_effect() -> Self {
        Self::new()
            .with(ValidatorId::Placebo)
            .with(ValidatorId::RandomCommonCause)
            .with(ValidatorId::Bootstrap)
            .with(ValidatorId::UnobservedCommonCause)
            .with(ValidatorId::Overlap)
            .with(ValidatorId::OverlapRule)
            .with(ValidatorId::DataSubset)
            .with(ValidatorId::DummyOutcome)
            .with(ValidatorId::EValue)
            .with(ValidatorId::Graph)
            .with(ValidatorId::LinearSensitivity)
            .with(ValidatorId::PartialLinearSensitivity)
            .with(ValidatorId::NonparametricSensitivity)
            .with(ValidatorId::Reisz)
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
            out.push(self.run_one(id, problem, workspace, ctx)?);
        }
        for custom in &self.custom {
            out.push(ValidationOutcome::Report(custom.validate(problem, ctx)?));
        }
        Ok(out)
    }

    /// Like [`Self::run`], reusing a warmed propensity workspace for overlap diagnostics.
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
            if id == ValidatorId::Overlap {
                out.push(ValidationOutcome::Report(
                    crate::overlap::OverlapRefuter::new()
                        .refute_with_propensity(problem, propensity)?,
                ));
            } else {
                out.push(self.run_one(id, problem, workspace, ctx)?);
            }
        }
        for custom in &self.custom {
            out.push(ValidationOutcome::Report(custom.validate(problem, ctx)?));
        }
        Ok(out)
    }

    /// Collect only successful [`RefutationReport`]s (drops `NotApplicable`).
    #[must_use]
    pub fn reports_only(outcomes: &[ValidationOutcome]) -> Vec<RefutationReport> {
        outcomes
            .iter()
            .filter_map(|o| match o {
                ValidationOutcome::Report(r) => Some(r.clone()),
                ValidationOutcome::NotApplicable { .. } => None,
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
        let mut out = Vec::with_capacity(self.validators.len() + self.custom.len());
        for &id in &self.validators {
            out.push(self.run_one_bayesian(id, bayes, ctx)?);
        }
        // Custom validators need a RefutationProblem; Bayesian path leaves them unused here.
        let _ = &self.custom;
        Ok(out)
    }

    fn run_one(
        &self,
        id: ValidatorId,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<ValidationOutcome, ValidationError> {
        let method = problem.estimand.method_kind().ok();
        let static_linear = method == Some(antecedent_expr::EstimandMethod::BackdoorAdjustment)
            && problem.estimator.is_none_or(|e| e == "linear.adjustment.ate");
        let temporal_linear = method
            == Some(antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded)
            && problem.temporal.is_some()
            && problem.estimator.is_none_or(|e| {
                matches!(e, "temporal.linear.adjustment" | "bayesian.temporal.gcomp")
            });
        let linear_ok = static_linear || temporal_linear;
        match id {
            ValidatorId::Placebo => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "PlaceboTreatment requires backdoor.adjustment + linear path \
                         (or temporal.backdoor.unfolded + temporal linear path)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &PlaceboTreatment::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::RandomCommonCause => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "RandomCommonCause requires backdoor.adjustment + linear path \
                         (or temporal.backdoor.unfolded + temporal linear path)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &RandomCommonCause::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::Bootstrap => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "BootstrapCiCoverage requires backdoor.adjustment + linear path \
                         (or temporal.backdoor.unfolded + temporal linear path)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &BootstrapRefute::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::UnobservedCommonCause => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "UnobservedCommonCause requires backdoor.adjustment or temporal.backdoor.unfolded",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &UnobservedCommonCause::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::Overlap => {
                if problem.temporal.is_some() {
                    return Ok(na(
                        id,
                        "OverlapRefuter not applicable to temporal unfolded designs \
                         (propensity uses schema adjustment columns)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &OverlapRefuter::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::OverlapRule => {
                if problem.temporal.is_some() {
                    return Ok(na(
                        id,
                        "OverlapRuleRefuter not applicable to temporal unfolded designs",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &OverlapRuleRefuter::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::DataSubset => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "DataSubsetRefuter requires backdoor.adjustment + linear path \
                         (or temporal.backdoor.unfolded + temporal linear path)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &DataSubsetRefuter::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::DummyOutcome => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "DummyOutcome requires backdoor.adjustment + linear path \
                         (or temporal.backdoor.unfolded + temporal linear path)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &DummyOutcome::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::EValue => Ok(ValidationOutcome::Report(run_validator(
                &EValue::new(),
                problem,
                workspace,
                ctx,
            )?)),
            ValidatorId::Graph => {
                // Temporal unfolded adjustment ids are not schema drop-covariate targets.
                if !static_linear {
                    return Ok(na(
                        id,
                        "DropAdjustmentCovariate requires static backdoor.adjustment + linear path \
                         (not applicable to temporal unfolded designs)",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &GraphRefuter::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::LinearSensitivity => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "LinearSensitivity requires backdoor.adjustment or temporal.backdoor.unfolded",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &LinearSensitivity::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::PartialLinearSensitivity => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "PartialLinearSensitivity requires backdoor.adjustment or temporal.backdoor.unfolded",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &PartialLinearSensitivity::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::NonparametricSensitivity => Ok(ValidationOutcome::Report(run_validator(
                &NonparametricSensitivity::new(),
                problem,
                workspace,
                ctx,
            )?)),
            ValidatorId::Reisz => {
                if problem.temporal.is_some() {
                    return Ok(na(
                        id,
                        "ReiszSensitivity not applicable to temporal unfolded designs",
                    ));
                }
                Ok(ValidationOutcome::Report(run_validator(
                    &ReiszSensitivity::new(),
                    problem,
                    workspace,
                    ctx,
                )?))
            }
            ValidatorId::PriorPredictive
            | ValidatorId::PosteriorPredictive
            | ValidatorId::PriorSensitivity
            | ValidatorId::McmcDiagnostics => Ok(na(
                id,
                "Bayesian PPC/prior-sensitivity/MCMC diagnostics require ValidationSuite::run_bayesian with a fitted posterior",
            )),
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
                let check = PriorPredictiveCheck {
                    n_sims: 200,
                    seed: ctx.rng.master_seed(),
                    ..PriorPredictiveCheck::new()
                };
                let rep = check.check(bayes.prepared, ctx)?;
                Ok(ValidationOutcome::Report(
                    rep.to_refutation_report(bayes.original_ate, bayes.ppc_alpha),
                ))
            }
            ValidatorId::PosteriorPredictive => {
                let check = PosteriorPredictiveCheck::new();
                let rep = check.check(bayes.prepared, bayes.posterior)?;
                Ok(ValidationOutcome::Report(
                    rep.to_refutation_report(bayes.original_ate, bayes.ppc_alpha),
                ))
            }
            ValidatorId::PriorSensitivity => {
                let sens = PriorSensitivity::standard_grid();
                let (summary, _posts) = sens.evaluate(
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

    /// Bayesian diagnostics suite identifiers.
    #[must_use]
    pub fn bayesian_diagnostics() -> Self {
        Self::new()
            .with(ValidatorId::PriorPredictive)
            .with(ValidatorId::PosteriorPredictive)
            .with(ValidatorId::PriorSensitivity)
            .with(ValidatorId::McmcDiagnostics)
    }

    /// Prior predictive check only (cheap; no fitted posterior required beyond prepare).
    #[must_use]
    pub fn prior_predictive() -> Self {
        Self::new().with(ValidatorId::PriorPredictive)
    }
}

fn na(id: ValidatorId, reason: &str) -> ValidationOutcome {
    ValidationOutcome::NotApplicable { validator: id, reason: Arc::from(reason) }
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
    use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
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
        let problem = RefutationProblem {
            data: &data,
            estimand: &estimand,
            query: &query,
            original: &original,
            estimator: Some("linear.adjustment.ate"),
            temporal: None,
        };
        let outcomes = ValidationSuite::full_effect().run(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(outcomes.len(), 14);
        let reports = ValidationSuite::reports_only(&outcomes);
        assert!(reports.len() >= 10, "reports={}", reports.len());
    }
}

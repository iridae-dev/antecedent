//! Caller-supplied nuisance learners declared on a [`crate::StudyBuilder`].
//!
//! [`crate::StudyBuilder::learner_factories`] declares an outcome-regression and a
//! propensity factory for the study. The compiled static average-effect route over a
//! declared DAG (`StaticAteDag`) then fits the study's configured [`DmlAte`], [`AipwAte`],
//! [`DrLearner`] or [`CausalForest`] with those factories instead of its built-in nuisance
//! learners: it prepares the estimator on the identified adjustment estimand exactly as it
//! does for the built-in learners and calls the estimator's `fit_with_factories` with the
//! study's execution context, so the result is the one that method returns for the same
//! inputs. The estimator records the factories as unverified assumptions and keeps their fold
//! provenance; the declaration's identity extends the estimator identity of the program, and
//! a digest of it extends the calibration key's functional label (`calibration_label`), so
//! coverage measured with the built-in learners never annotates a factory fit.
//!
//! The declaration is admitted at build only where that route is the one that runs: a static
//! average effect on a tabular table under a declared DAG, a configured estimator of one of
//! the four kinds (which owns its replicate count, so the facade bootstrap never refits it),
//! frequentist inference without refuters, custom validators, a tier background, a discovery
//! split, a weighted target or a shared batch design. Anything else is refused at build, so no
//! other route can silently fit its built-in learners instead.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, ExecutionContext};
use antecedent_estimate::{AipwAte, CausalForest, DmlAte, DrLearner, EffectEstimate};
use antecedent_io::EstimatorSpecWire;
use antecedent_learn::{LearnerFactory, PredictionTask};

use crate::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::strategy_table::StaticEstimateInput;

/// Longest declared identity, in bytes.
const MAX_IDENTITY_BYTES: usize = 4096;

/// The outcome and propensity learners a study fits its cross-fitted estimator with.
///
/// The outcome factory predicts the outcome regression (`PredictionTask::Regression`) and the
/// treatment factory the propensity (`PredictionTask::BinaryProbability`). `identity` is the
/// caller's stable name for the pair (for example the provider names and their specification
/// digests): it enters the program's estimator identity, so two studies that differ only in
/// their learners have different identities. The factories are external code and the library
/// does not verify them; the estimate records them as unverified assumptions.
#[derive(Clone)]
pub struct LearnerFactories {
    outcome: Arc<dyn LearnerFactory>,
    treatment: Arc<dyn LearnerFactory>,
    identity: Arc<str>,
    /// `Some(true)` when both factories' `predict` is a pure function of the
    /// artifact bytes and the rows. `Some(false)` when a factory needs live
    /// process state. `None` when the caller did not declare it (native
    /// factories, which export a portable predictor themselves).
    portability: Option<bool>,
}

impl std::fmt::Debug for LearnerFactories {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnerFactories")
            .field("identity", &self.identity)
            .field("outcome_task", &self.outcome.task())
            .field("treatment_task", &self.treatment.task())
            .finish_non_exhaustive()
    }
}

impl LearnerFactories {
    /// Declare the outcome-regression and propensity factories under a stable identity.
    ///
    /// # Errors
    ///
    /// `invalid_argument` when the outcome factory does not predict a regression,
    /// the treatment factory does not predict a binary probability, or the identity is empty
    /// or longer than 4096 bytes.
    pub fn new(
        outcome: Arc<dyn LearnerFactory>,
        treatment: Arc<dyn LearnerFactory>,
        identity: impl Into<String>,
    ) -> Result<Self, CausalError> {
        let identity: String = identity.into();
        if identity.trim().is_empty() || identity.len() > MAX_IDENTITY_BYTES {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "learner factories need a non-empty identity of at most 4096 bytes"
            ));
        }
        if outcome.task() != PredictionTask::Regression {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "the outcome learner factory must predict a regression"
            ));
        }
        if treatment.task() != PredictionTask::BinaryProbability {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "the treatment learner factory must predict a binary probability"
            ));
        }
        Ok(Self { outcome, treatment, identity: Arc::from(identity), portability: None })
    }

    /// Record whether `predict` is a pure function of the artifact and the rows.
    #[must_use]
    pub fn declared_portability(mut self, portable: bool) -> Self {
        self.portability = Some(portable);
        self
    }

    /// Refuse a non-portable declaration on a route that evaluates from the artifact.
    ///
    /// An undeclared portability (`None`) stays admitted: those factories are the
    /// native ones, which export a portable predictor. A provider that needs live
    /// process state is refused with `dose_response.learner_not_portable`.
    ///
    /// # Errors
    ///
    /// `route_not_supported` (`dose_response.learner_not_portable`) when portability
    /// was declared false.
    pub fn admit_portable(&self) -> Result<(), CausalError> {
        if self.portability == Some(false) {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                detail = antecedent_core::reason_details::dose_response::LEARNER_NOT_PORTABLE,
                "a learner whose predict needs live process state is refused on a route that \
                 evaluates from the artifact bytes and the rows. Next attempt: declare a learner \
                 whose predict is a pure function of the artifact and the rows"
            ));
        }
        Ok(())
    }

    /// The outcome-regression factory.
    #[must_use]
    pub fn outcome(&self) -> &dyn LearnerFactory {
        &*self.outcome
    }

    /// The propensity factory.
    #[must_use]
    pub fn treatment(&self) -> &dyn LearnerFactory {
        &*self.treatment
    }

    /// The caller's identity for the pair.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
}

/// Refuse at build a study whose declared learner factories would not reach the factory fit.
///
/// # Errors
///
/// `option_not_applicable` when the study declares learner factories outside the compiled
/// static average-effect route over a DAG with a configured cross-fitted estimator.
pub(crate) fn admit(study: &crate::Study) -> Result<(), CausalError> {
    if study.declared.learner_factories.is_none() {
        return Ok(());
    }
    if !matches!(study.query, antecedent_core::CausalQuery::AverageEffect(_))
        || !matches!(study.data, super::builder::DataInput::Tabular(_))
        || study.graph.class() != crate::GraphClass::Dag
        || study.graph_posterior.is_some()
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "learner factories apply to a static average effect on a tabular table under a \
             declared DAG; declare the DAG and an average-effect query, or drop the factories"
        ));
    }
    if !matches!(
        study.declared.estimator_spec.as_deref(),
        Some(
            EstimatorSpec::Dml(_)
                | EstimatorSpec::Aipw(_)
                | EstimatorSpec::DrLearner(_)
                | EstimatorSpec::CausalForest(_)
        )
    ) {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "learner factories replace the nuisance learners of a configured DmlAte, AipwAte, \
             DrLearner or CausalForest; declare one of them with StudyBuilder::estimator"
        ));
    }
    if !matches!(study.settings.inference, crate::InferenceMode::Frequentist)
        || study.settings.refute != crate::RefuteSuite::None
        || !study.declared.custom_validators.is_empty()
        || study.declared.tiered.is_some()
        || study.declared.split.is_some()
        || study.declared.weighted_target.is_some()
        || study.shared_batch_design.is_some()
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "learner factories are fitted once on the analysed rows under frequentist inference; \
             refuters, custom validators, a tier background, a discovery split, a weighted \
             target or a shared batch design would refit the built-in learners instead, so \
             declare none of them with the factories"
        ));
    }
    Ok(())
}

/// The estimator identity of `spec`, extended by the declared factories.
///
/// Without factories this is exactly the identity the study writes today. With them, the
/// outcome and treatment learner fields of a cross-fitted estimator name the caller factory
/// and its declared identity, replacing the built-in learner names the factories do not use.
pub(crate) fn spec_identity(
    spec: &EstimatorSpec,
    factories: Option<&LearnerFactories>,
) -> EstimatorSpecWire {
    let mut wire = super::contract_identity::estimator_spec_identity(spec);
    // `admit` refuses factories on every other estimator.
    if let (
        Some(factories),
        EstimatorSpecWire::Dml(config)
        | EstimatorSpecWire::Aipw(config)
        | EstimatorSpecWire::DrLearner(config)
        | EstimatorSpecWire::CausalForest(config),
    ) = (factories, &mut wire)
    {
        let declared = format!("caller_factory;{}", factories.identity());
        config.outcome = Some("caller_factory".into());
        config.outcome_config = Some(declared.clone());
        config.treatment = Some("caller_factory".into());
        config.treatment_config = Some(declared);
    }
    wire
}

/// The calibration key's functional suffix of declared learner factories.
///
/// Every coverage record of the cross-fitted estimators was measured with their built-in
/// nuisance learners, so a fit with caller-supplied factories is a different interval
/// construction. `learners=caller_factory:<blake3 hex of the declared identity>` keeps its interval
/// from binding to those records, and to a record measured under other factories; it reports
/// coverage not measured until a record is measured under the same declared learners. `None`
/// without factories, so every other study keeps its key.
#[must_use]
pub(crate) fn calibration_label(factories: Option<&LearnerFactories>) -> Option<String> {
    factories.map(|factories| {
        format!(
            "learners=caller_factory:{}",
            blake3::hash(factories.identity().as_bytes()).to_hex()
        )
    })
}

/// Fit the study's configured cross-fitted estimator with the declared factories.
///
/// The estimator is prepared on the identified estimand and fitted with
/// `fit_with_factories` under the caller's execution context, as the static route prepares
/// and fits it with its built-in learners.
///
/// # Errors
///
/// `option_not_applicable` when no configured cross-fitted estimator is declared, and the
/// estimator's own preparation and factory-fit refusals.
pub(crate) fn fit_static_effect(
    spec: Option<&EstimatorSpec>,
    factories: &LearnerFactories,
    input: StaticEstimateInput<'_>,
    assumptions: AssumptionSet,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let StaticEstimateInput { data, estimand, query, .. } = input;
    if let (Some(spec), Some(design)) = (spec, input.sampling_design) {
        crate::estimator_spec::ensure_dependence_honoured(spec, design)?;
    }
    let (outcome, treatment) = (factories.outcome(), factories.treatment());
    let estimate = match spec {
        Some(EstimatorSpec::Dml(config)) => {
            fit_dml(config, data, estimand, query, ctx, assumptions, outcome, treatment)
        }
        Some(EstimatorSpec::Aipw(config)) => {
            fit_aipw(config, data, estimand, query, ctx, assumptions, outcome, treatment)
        }
        Some(EstimatorSpec::DrLearner(config)) => {
            fit_dr(config, data, estimand, query, ctx, assumptions, outcome, treatment)
        }
        Some(EstimatorSpec::CausalForest(config)) => {
            fit_forest(config, data, estimand, query, ctx, assumptions, outcome, treatment)
        }
        _ => {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "learner factories replace the nuisance learners of a configured DmlAte, \
                 AipwAte, DrLearner or CausalForest; declare one of them with \
                 StudyBuilder::estimator"
            ));
        }
    };
    estimate.map_err(CausalError::from)
}

macro_rules! factory_fit {
    ($name:ident, $estimator:ty) => {
        #[allow(clippy::too_many_arguments, reason = "one prepare-and-fit step per estimator")]
        fn $name(
            estimator: &$estimator,
            data: &antecedent_data::TabularData,
            estimand: &antecedent_expr::IdentifiedEstimand,
            query: &antecedent_core::AverageEffectQuery,
            ctx: &ExecutionContext,
            assumptions: AssumptionSet,
            outcome: &dyn LearnerFactory,
            treatment: &dyn LearnerFactory,
        ) -> Result<EffectEstimate, antecedent_estimate::EstimationError> {
            let problem = estimator.prepare(data, estimand, query)?;
            estimator.fit_with_factories(&problem, ctx, assumptions, outcome, treatment)
        }
    };
}

factory_fit!(fit_dml, DmlAte);
factory_fit!(fit_aipw, AipwAte);
factory_fit!(fit_dr, DrLearner);
factory_fit!(fit_forest, CausalForest);

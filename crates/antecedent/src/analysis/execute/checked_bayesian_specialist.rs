//! Builder independent execution for the model-scoped Bayesian IV and sharp-RD estimators.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_specialist_path::{
    BayesianSpecialistInputs, check_bayesian_iv_scope, check_bayesian_rd_scope,
};
use super::*;
use crate::RdConfig;
use crate::strategy_table::{
    EstimatorId, IdentifierId, identify_static_query, require_identified, select_estimand,
};

/// Which conjugate specialist model the retained operation executes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BayesianSpecialistFamily {
    /// `iv.bayesian_joint_linear`: joint linear Gaussian structural model.
    IvJointLinear,
    /// `rd.bayesian_local_linear`: fixed-bandwidth local-linear sharp design.
    RdLocalLinear(RdConfig),
}

/// Complete target, identification proof, model configuration, and physical
/// plan for one Bayesian IV or sharp-RD average effect. Execution reads only
/// these retained fields and the click data.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianSpecialistOperation {
    query: AverageEffectQuery,
    family: BayesianSpecialistFamily,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    config: BayesianConfig,
    physical: PhysicalExecutionPlan,
}

/// Read-only target and procedure retained for a checked Bayesian IV or sharp-RD route.
#[derive(Clone, Debug)]
pub struct CheckedBayesianSpecialistInfo {
    /// Frozen average-effect target.
    pub query: AverageEffectQuery,
    /// Identifier fixed during preparation.
    pub identifier: IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: EstimatorId,
    /// Posterior draw count fixed during preparation.
    pub n_draws: usize,
    /// Isotropic prior scale fixed during preparation.
    pub prior_scale: f64,
    /// Sharp-RD design retained for `rd.bayesian_local_linear`.
    pub rd: Option<RdConfig>,
    /// The prepared physical plan identity.
    pub plan_id: Arc<str>,
}

impl CheckedBayesianSpecialistOperation {
    /// Seal the route when the study, plan, and model scope all agree; other
    /// shapes return `None` and keep their existing dispatch.
    ///
    /// # Errors
    ///
    /// Identification failure or a model-scope violation the estimator itself
    /// would refuse at execution.
    #[inline(never)]
    pub(crate) fn checked(
        study: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<Self>>, CausalError> {
        let CausalQuery::AverageEffect(query) = &study.query else {
            return Ok(None);
        };
        let (Some(graph), InferenceMode::Bayesian(config), DataInput::Tabular(_)) =
            (study.graph.as_dag(), &study.inference, &study.data)
        else {
            return Ok(None);
        };
        if study.graph_posterior.is_some()
            || study.tiered.is_some()
            || study.split.is_some()
            || !matches!(
                study.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || plan.logical.query != study.query
        {
            return Ok(None);
        }
        let estimator = plan.logical.record.estimator.as_deref().and_then(|id| id.parse().ok());
        let identifier = plan.logical.record.identifier.as_deref().and_then(|id| id.parse().ok());
        let (family, identifier, estimator) = match (estimator, identifier) {
            (Some(EstimatorId::BayesianIvJointLinear), Some(IdentifierId::Iv)) => (
                BayesianSpecialistFamily::IvJointLinear,
                IdentifierId::Iv,
                EstimatorId::BayesianIvJointLinear,
            ),
            (Some(EstimatorId::BayesianRdLocalLinear), Some(IdentifierId::RdSharp)) => {
                let Some(rd) = study.rd else {
                    return Ok(None);
                };
                (
                    BayesianSpecialistFamily::RdLocalLinear(rd),
                    IdentifierId::RdSharp,
                    EstimatorId::BayesianRdLocalLinear,
                )
            }
            _ => return Ok(None),
        };
        // A sharp design binds its target to the cutoff population at build;
        // the joint IV model estimates the all-observed contrast only.
        let population_matches = match (&family, &query.target_population) {
            (_, antecedent_core::TargetPopulation::AllObserved) => true,
            (
                BayesianSpecialistFamily::RdLocalLinear(rd),
                antecedent_core::TargetPopulation::LocalAtCutoff { running, .. },
            ) => *running == rd.running_variable,
            _ => false,
        };
        if !population_matches {
            return Ok(None);
        }
        let target = CausalQuery::AverageEffect(query.clone());
        let (identification, estimand) = match family {
            BayesianSpecialistFamily::IvJointLinear => {
                match study.identification_cache.as_deref() {
                    Some(cache) if cache.identification.query == target => {
                        (cache.identification.clone(), cache.estimand.clone())
                    }
                    _ => {
                        let identification = identify_static_query(identifier, graph, &target)?;
                        require_identified(&identification)?;
                        let estimand = select_estimand(&identification, estimator)?;
                        (identification, estimand)
                    }
                }
            }
            BayesianSpecialistFamily::RdLocalLinear(rd) => {
                let identification = SharpRdIdentifier::new(SharpRdConfig::new(
                    rd.running_variable,
                    rd.cutoff,
                    rd.bandwidth,
                ))
                .identify_on(graph, target)
                .map_err(CausalError::from)?;
                require_identified(&identification)?;
                let estimand = select_estimand(&identification, estimator)?;
                (identification, estimand)
            }
        };
        match family {
            BayesianSpecialistFamily::IvJointLinear => check_bayesian_iv_scope(&estimand, config)?,
            BayesianSpecialistFamily::RdLocalLinear(_) => check_bayesian_rd_scope(query, config)?,
        }
        Ok(Some(Box::new(Self {
            query: query.clone(),
            family,
            identification,
            estimand,
            config: config.clone(),
            physical: plan.clone(),
        })))
    }

    pub(crate) fn info(&self) -> CheckedBayesianSpecialistInfo {
        let (identifier, estimator, rd) = match self.family {
            BayesianSpecialistFamily::IvJointLinear => {
                (IdentifierId::Iv, EstimatorId::BayesianIvJointLinear, None)
            }
            BayesianSpecialistFamily::RdLocalLinear(rd) => {
                (IdentifierId::RdSharp, EstimatorId::BayesianRdLocalLinear, Some(rd))
            }
        };
        CheckedBayesianSpecialistInfo {
            query: self.query.clone(),
            identifier,
            estimator,
            n_draws: self.config.n_draws,
            prior_scale: self.config.prior_scale,
            rd,
            plan_id: Arc::clone(&self.physical.record.plan_id),
        }
    }

    /// Execute the retained model on `data` through the study's result context.
    ///
    /// # Errors
    ///
    /// Estimation failure or a retained plan that no longer binds its target.
    pub(crate) fn execute(
        &self,
        study: &Study,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.identification.query != CausalQuery::AverageEffect(self.query.clone())
            || self.physical.logical.query != CausalQuery::AverageEffect(self.query.clone())
        {
            return Err(CausalError::Compile {
                message: "checked Bayesian specialist operation lost its target binding".into(),
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let inputs = BayesianSpecialistInputs {
            identification: &self.identification,
            estimand: &self.estimand,
            config: &self.config,
            identify_cached: true,
        };
        match self.family {
            BayesianSpecialistFamily::IvJointLinear => {
                study.execute_bayesian_iv_identified(data, &self.query, &self.physical, inputs, ctx)
            }
            BayesianSpecialistFamily::RdLocalLinear(rd) => study.execute_bayesian_rd_identified(
                data,
                &self.query,
                &self.physical,
                rd,
                inputs,
                ctx,
            ),
        }
    }
}

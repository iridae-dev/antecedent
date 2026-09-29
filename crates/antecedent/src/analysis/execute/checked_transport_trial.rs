//! Builder independent execution for the certified trial-to-target transport routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::transport_interference_path::{
    TransportTrialInputs, refuse_transport_bayesian_priors, refuse_unestimable_transport,
    transport_primary_pair, transport_sid_identification,
};
use super::*;
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_identify::TransportIdentification;

/// Inference procedure fixed at preparation for one transported contrast.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransportTrialProcedure {
    /// Delta-method IPW under known selection and treatment probabilities.
    KnownProbabilityIpw,
    /// Dirichlet trial-row-law bootstrap of the same IPW contrast.
    BayesianBootstrap {
        /// Posterior draws fixed at preparation.
        draws: usize,
    },
}

/// Complete transport target, certified sID proof, trial design columns,
/// inference procedure, and physical plan for one trial-to-target contrast.
#[derive(Clone, Debug)]
pub(crate) struct CheckedTransportTrialOperation {
    query: antecedent_core::TransportQuery,
    source_schema: antecedent_core::CausalSchema,
    trial: super::super::builder::TransportTrialSpec,
    transport: TransportIdentification,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    procedure: TransportTrialProcedure,
    estimator: EstimatorId,
    inference: InferenceMode,
    physical: PhysicalExecutionPlan,
}

/// Read-only target, proof shape, and procedure retained for a checked transport route.
#[derive(Clone, Debug)]
pub struct CheckedTransportTrialInfo {
    /// Frozen transport target.
    pub query: antecedent_core::TransportQuery,
    /// Identifier fixed during preparation.
    pub identifier: IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: EstimatorId,
    /// Transport formula rule certified at preparation.
    pub rule: Arc<str>,
    /// Posterior draws for the Bayesian bootstrap; `None` for the IPW route.
    pub posterior_draws: Option<usize>,
    /// The prepared physical plan identity.
    pub plan_id: Arc<str>,
}

impl CheckedTransportTrialOperation {
    /// Seal the route when the study, proof cache, design columns, and plan
    /// all agree; other shapes return `None` and keep their existing dispatch.
    ///
    /// # Errors
    ///
    /// An uncertified transport formula, a Bayesian configuration the route
    /// refuses, or a plan that disagrees with the study.
    #[inline(never)]
    pub(crate) fn checked(
        study: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<Box<Self>>, CausalError> {
        let (CausalQuery::Transport(query), DataInput::Tabular(data)) = (&study.query, &study.data)
        else {
            return Ok(None);
        };
        let (Some(trial), Some(transport)) =
            (study.transport_trial.as_ref(), study.transport_identification_cache.as_deref())
        else {
            return Ok(None);
        };
        if study.graph.class() != GraphClass::Admg
            || study.selection_diagram.is_none()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
            || study.split.is_some()
            || !matches!(
                study.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
        {
            return Ok(None);
        }
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        refuse_unestimable_transport(transport)?;
        let (procedure, estimator) = match &study.inference {
            InferenceMode::Frequentist => {
                (TransportTrialProcedure::KnownProbabilityIpw, EstimatorId::TransportTrialIpw)
            }
            InferenceMode::Bayesian(config) => {
                refuse_transport_bayesian_priors(config)?;
                (
                    TransportTrialProcedure::BayesianBootstrap { draws: config.n_draws },
                    EstimatorId::TransportTrialBayesianBootstrap,
                )
            }
        };
        if plan.logical.query != study.query
            || plan.logical.record.identifier.as_deref()
                != Some(IdentifierId::TransportSid.as_str())
            || plan.logical.record.estimator.as_deref() != Some(estimator.as_str())
        {
            return Err(CausalError::Compile {
                message: "transport target, certified proof, or selected estimator differs from its checked plan".into(),
            });
        }
        let (treatment, outcome) = transport_primary_pair(query)?;
        for column in [
            treatment,
            outcome,
            trial.trial,
            trial.selection_probability,
            trial.treatment_probability,
        ] {
            if data.schema().get(column).is_err() {
                return Err(CausalError::Unsupported {
                    message: "transport trial design names a column absent from the bound table",
                });
            }
        }
        let (identification, estimand) = transport_sid_identification(
            CausalQuery::Transport(query.clone()),
            treatment,
            outcome,
            transport,
        );
        Ok(Some(Box::new(Self {
            query: query.clone(),
            source_schema: data.schema().clone(),
            trial: trial.clone(),
            transport: transport.clone(),
            identification,
            estimand,
            procedure,
            estimator,
            inference: study.inference.clone(),
            physical: plan.clone(),
        })))
    }

    pub(crate) fn info(&self) -> CheckedTransportTrialInfo {
        let rule = match &self.transport {
            TransportIdentification::Transportable { certificate, .. } => {
                Arc::from(certificate.rule.as_ref())
            }
            TransportIdentification::NotCertified(_)
            | TransportIdentification::MissingEvidence(_) => Arc::from("uncertified"),
        };
        CheckedTransportTrialInfo {
            query: self.query.clone(),
            identifier: IdentifierId::TransportSid,
            estimator: self.estimator,
            rule,
            posterior_draws: match self.procedure {
                TransportTrialProcedure::KnownProbabilityIpw => None,
                TransportTrialProcedure::BayesianBootstrap { draws } => Some(draws),
            },
            plan_id: Arc::clone(&self.physical.record.plan_id),
        }
    }

    /// Execute the retained transport contrast on `data` through the study's
    /// result context.
    ///
    /// # Errors
    ///
    /// A table whose schema differs from preparation, or estimation failure.
    pub(crate) fn execute(
        &self,
        study: &Study,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.source_schema {
            return Err(CausalError::Unsupported {
                message: "transport execution requires the prepared table schema",
            });
        }
        let target = CausalQuery::Transport(self.query.clone());
        if self.identification.query != target || self.physical.logical.query != target {
            return Err(CausalError::Compile {
                message: "checked transport operation lost its target binding".into(),
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        study.execute_transport_identified(
            data,
            &self.query,
            &self.physical,
            TransportTrialInputs {
                trial: &self.trial,
                transport: &self.transport,
                identification: &self.identification,
                estimand: &self.estimand,
                inference: &self.inference,
                identify_cached: true,
            },
            ctx,
        )
    }
}

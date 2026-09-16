// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
};
use antecedent_data::TableView;
use antecedent_estimate::{
    EffectEstimate, OverlapPolicy, estimate_interference, trial_to_target_effect,
    trial_to_target_ipw_se,
};
use antecedent_identify::{TransportIdentification, TransportIdentifier};

use super::*;
use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

impl super::Study {
    pub(super) fn execute_transport(
        &self,
        data: &TabularData,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
            message: "TransportQuery execute requires a selection diagram",
        })?;
        let trial_spec = self.transport_trial.as_ref().ok_or(CausalError::Unsupported {
            message: "TransportQuery execute requires transport_trial columns",
        })?;
        let (treatment, outcome) =
            query.response.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                message: "TransportQuery inner response has no treatment/outcome".into(),
            })?;
        let (transport_id, identify_cached) =
            if let Some(cached) = self.transport_identification_cache.as_deref() {
                (cached.clone(), true)
            } else {
                report_identify_compute(ctx);
                (live_transport_identification(diagram, query)?, false)
            };
        refuse_unestimable_transport(&transport_id)?;
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::Transport(query.clone()),
            treatment,
            outcome,
        );
        let outcomes = data
            .float64_values(outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let treatment_col = data
            .float64_values(treatment)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let trial_col = data
            .float64_values(trial_spec.trial)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let selection = data
            .float64_values(trial_spec.selection_probability)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let propensity = data
            .float64_values(trial_spec.treatment_probability)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let treatment_bool: Vec<bool> = treatment_col.iter().map(|v| *v != 0.0).collect();
        let trial_bool: Vec<bool> = trial_col.iter().map(|v| *v != 0.0).collect();
        let transported = trial_to_target_effect(
            &transport_id,
            &outcomes,
            &treatment_bool,
            &trial_bool,
            &selection,
            &propensity,
            None,
        )
        .map_err(CausalError::from)?;
        // Known selection and treatment probabilities: the delta-method SE of
        // the ratio-of-means IPW contrast over iid rows.
        let se = trial_to_target_ipw_se(
            &outcomes,
            &treatment_bool,
            &trial_bool,
            &selection,
            &propensity,
            transported.ipw,
        )
        .map_err(CausalError::from)?;
        let estimate = EffectEstimate::new(
            transported.ipw,
            se,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut result = self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TransportSid,
            estimator_id: EstimatorId::TransportTrialIpw,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: vec![Diagnostic::new(
                "estimate.transport.trial_ipw",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "binary trial-to-target IPW; inner ResponseCurve names treatment/outcome only",
            )],
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        });
        result.transport = Some(transported);
        Ok(result)
    }

    pub(super) fn execute_interference(
        &self,
        query: &antecedent_core::InterferenceQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let spec = self.interference.as_ref().ok_or(CausalError::Unsupported {
            message: "InterferenceQuery execute requires StudyBuilder::interference",
        })?;
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            query.functional;
        let seed = ctx.rng.stream(0x1F7E).next_u64();
        let estimated = estimate_interference(query, &spec.network, &spec.assignment, seed)
            .map_err(CausalError::from)?;
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::Interference(query.clone()),
            outcome,
            outcome,
        );
        let se = estimated.contrast.conservative_variance.sqrt();
        let estimate = EffectEstimate::new(
            estimated.contrast.horvitz_thompson,
            se,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut result = self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::InterferenceDesign,
            estimator_id: EstimatorId::InterferenceHtHajek,
            treatment: outcome,
            outcome,
            identify_cached: false,
            extra_diagnostics: vec![Diagnostic::new(
                "estimate.interference.young_bound",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "conservative Young variance bound; not the Aronow–Samii joint-exposure variance",
            )],
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        });
        result.interference = Some(estimated);
        Ok(result)
    }
}

pub(crate) fn live_transport_identification(
    diagram: &antecedent_graph::SelectionDiagram,
    query: &antecedent_core::TransportQuery,
) -> Result<TransportIdentification, CausalError> {
    let identified = TransportIdentifier::new()
        .identify(diagram, query)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    refuse_unestimable_transport(&identified)?;
    Ok(identified)
}

fn refuse_unestimable_transport(identified: &TransportIdentification) -> Result<(), CausalError> {
    match identified {
        TransportIdentification::NotCertified(certificate) => Err(CausalError::Compile {
            message: format!(
                "transport not certified: {} ({})",
                certificate.reason, certificate.message
            ),
        }),
        TransportIdentification::Transportable {
            formula: antecedent_identify::TransportFormula::RecursiveFactorization { .. },
            ..
        } => Err(CausalError::Unsupported {
            message: "RecursiveFactorization is identified but not estimable on the \
                      licensed trial-to-target IPW path",
        }),
        TransportIdentification::Transportable { .. } => Ok(()),
    }
}

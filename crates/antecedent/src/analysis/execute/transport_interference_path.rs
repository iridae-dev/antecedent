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
        let (identification, estimand) = transport_sid_identification(
            CausalQuery::Transport(query.clone()),
            treatment,
            outcome,
            &transport_id,
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
        data: &TabularData,
        query: &antecedent_core::InterferenceQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        // The executed outcomes are `data`, under the frozen network and assignment.
        let spec = self
            .interference
            .as_ref()
            .ok_or(CausalError::Unsupported {
                message: "InterferenceQuery execute requires StudyBuilder::interference",
            })?
            .bound_to(data)?;
        let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
            query.functional;
        let seed = ctx.rng.stream(0x1F7E).next_u64();
        let estimated = estimate_interference(query, &spec.network, &spec.assignment, seed)
            .map_err(CausalError::from)?;
        let (identification, estimand) =
            interference_design_identification(CausalQuery::Interference(query.clone()), outcome);
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

/// Inspectable do-expectation leaf. Same shape as `parametric_scm_identification`,
/// but the rule and assumptions name the design-based identifier, not GCM.
fn inspectable_do_expectation(
    treatment: VariableId,
    outcome: VariableId,
    rule: &str,
    note: &str,
) -> (CausalExprArena, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let y = arena.intern_var_set([outcome]);
    let do_t = arena.intern_intervention_set([treatment]);
    let empty = arena.empty_var_set();
    let distribution = arena.intern(ExprNode::Distribution {
        variables: y,
        conditioned_on: empty,
        intervention: do_t,
        domain: DomainRef::Interventional,
    });
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution });
    arena.set_derivation(
        functional,
        DerivationMeta { rule: Arc::from(rule), note: Some(Arc::from(note)) },
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    (arena, estimand)
}

fn transport_sid_identification(
    query: CausalQuery,
    treatment: VariableId,
    outcome: VariableId,
    identified: &TransportIdentification,
) -> (IdentificationResult, IdentifiedEstimand) {
    let TransportIdentification::Transportable { certificate, .. } = identified else {
        unreachable!("execute already refused an uncertified transport formula");
    };
    let premises = certificate.premises.iter().map(AsRef::as_ref).collect::<Vec<_>>().join("; ");
    let (arena, estimand) = inspectable_do_expectation(
        treatment,
        outcome,
        certificate.rule.as_ref(),
        &format!("sID: treatment={treatment:?} outcome={outcome:?}; {premises}"),
    );
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(certificate.rule.as_ref()),
            description: Arc::from(if premises.is_empty() {
                "Structural transportability under the implemented sID subset (Direct or S-admissible standardize)."
                    .to_string()
            } else {
                premises.clone()
            }),
        },
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("transport.sid"),
        },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let mut derivation = DerivationTrace::default();
    derivation.push(certificate.rule.as_ref(), premises);
    let identification = IdentificationResult::identified(
        query,
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}

fn interference_design_identification(
    query: CausalQuery,
    outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let (arena, estimand) = inspectable_do_expectation(
        outcome,
        outcome,
        "interference.design",
        "The Dag binds schema and outcome; it does not identify the exposure contrast. \
         Identification is the known assignment mechanism and exposure mapping \
         (Horvitz–Thompson / Hájek).",
    );
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from("interference.design"),
            description: Arc::from(
                "The Dag binds schema and outcome; it does not identify the exposure contrast. \
                 Identification is the known assignment mechanism and exposure mapping.",
            ),
        },
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("interference.design"),
        },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let mut derivation = DerivationTrace::default();
    derivation.push("interference.design", "known assignment mechanism; graph is schema-only");
    let identification = IdentificationResult::identified(
        query,
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
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
        } => Err(crate::support_reason!(
            "construction_not_licensed",
            "RecursiveFactorization is identified but not estimable on the licensed \
             trial-to-target IPW path"
        )),
        TransportIdentification::Transportable { .. } => Ok(()),
    }
}

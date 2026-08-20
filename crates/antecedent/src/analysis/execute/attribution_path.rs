// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn execute_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::CounterfactualQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "counterfactual query missing outcome".into(),
        })?;
        let (treatment, active, control) = binary_cf_interventions(query)?;
        let ite = counterfactual_ite(fitted.model, data, treatment, outcome, active, control, ctx)?;
        let estimate = EffectEstimate::new(
            ite.mean_ite,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        let diagnostics = vec![Diagnostic::new(
            "gcm.counterfactual",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("noise_inference={:?}", ite.noise_inference),
        )];
        Ok(self.finish_gcm(
            physical,
            CausalQuery::Counterfactual(query.clone()),
            treatment,
            outcome,
            estimate,
            started,
            GcmSlot::Counterfactual(ite),
            diagnostics,
        ))
    }

    pub(super) fn execute_anomaly(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::AnomalyAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let _ = ctx;
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let scores = anomaly_attribution(
            &fitted.model,
            data,
            query.targets.iter().copied(),
            query.max_units,
        )?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        Ok(self.finish_gcm(
            physical,
            CausalQuery::AnomalyAttribution(query.clone()),
            outcome,
            outcome,
            nan_effect(),
            started,
            GcmSlot::Anomaly(scores),
            Vec::new(),
        ))
    }

    pub(super) fn execute_change_attribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::ChangeAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_distribution_change(
            &fitted.model,
            data,
            query,
            &antecedent_attribution::DistributionChangeOptions::default(),
            ctx,
        )?;
        let estimate = EffectEstimate::new(
            result.total_change,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        Ok(self.finish_gcm(
            physical,
            CausalQuery::ChangeAttribution(query.clone()),
            query.outcome,
            query.outcome,
            estimate,
            started,
            GcmSlot::Change(result),
            Vec::new(),
        ))
    }

    pub(super) fn execute_mechanism_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MechanismChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let detections = mechanism_change_detection(
            &fitted.model,
            data,
            query,
            antecedent_attribution::MechanismChangeMethod::LikelihoodRatio,
            ctx,
        )?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        Ok(self.finish_gcm(
            physical,
            CausalQuery::MechanismChange(query.clone()),
            outcome,
            outcome,
            nan_effect(),
            started,
            GcmSlot::Mechanism(detections),
            Vec::new(),
        ))
    }

    pub(super) fn execute_unit_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::UnitChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_unit_change(&fitted.model, data, query, ctx)?;
        Ok(self.finish_gcm(
            physical,
            CausalQuery::UnitChange(query.clone()),
            query.outcome,
            query.outcome,
            nan_effect(),
            started,
            GcmSlot::Unit(result),
            Vec::new(),
        ))
    }

    fn finish_gcm(
        &self,
        physical: &PhysicalExecutionPlan,
        query: CausalQuery,
        treatment: VariableId,
        outcome: VariableId,
        estimate: EffectEstimate,
        started: Instant,
        slot: GcmSlot,
        diagnostics: Vec<Diagnostic>,
    ) -> StudyResult {
        let (identification, estimand) = parametric_scm_identification(query, treatment, outcome);
        self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id: EstimatorId::LinearAdjustmentAte,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                gcm: Some(slot),
                empty_provenance: true,
                ..Default::default()
            },
        })
    }
}

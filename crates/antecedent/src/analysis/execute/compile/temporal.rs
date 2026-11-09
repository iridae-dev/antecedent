//! Logical compilation for temporal and panel routes.

use super::{
    Arc, CausalError, CausalQuery, DataClassification, DataInput, EstimatorId, ExecutionContext,
    IdentifierId, InferenceMode, LogicalAnalysisPlan, MultiEnvironmentData, PanelData, PlanId,
    ResponseQuery, TableView, TemporalAdmg, TemporalDag, TemporalEffectQuery, TimeSeriesData,
    VariableId, compile_logical_temporal_effect_classified, compile_logical_temporal_response,
    panel_compile_series, wire_pair,
};

impl super::super::Study {
    /// A plan compiled over a series records event data as `Event`, not `Temporal`.
    pub(in crate::analysis::execute) fn classified_as_series_input(
        &self,
        mut plan: LogicalAnalysisPlan,
    ) -> LogicalAnalysisPlan {
        if let DataInput::Event(_) = &self.data {
            plan.record.data_classification = DataClassification::Event;
        }
        plan
    }

    pub(super) fn compile_logical_temporal_effect_dag(
        &self,
        data: &TimeSeriesData,
        q: &TemporalEffectQuery,
        graph: &TemporalDag,
        classification: DataClassification,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_effect_classified(
            data,
            graph,
            q,
            self.declared.split,
            false,
            classification,
        )?;
        if let Some(
            instrument @ (EstimatorId::TemporalLaggedInstrument
            | EstimatorId::BayesianTemporalLaggedInstrument),
        ) = self.declared.estimator
        {
            plan.record.identifier = Some(Arc::from(IdentifierId::TemporalIvUnfolded.as_str()));
            plan.record.estimator = Some(Arc::from(instrument.as_str()));
        } else if q.is_multi_step_sustained() {
            plan.record.estimator = Some(Arc::from(EstimatorId::TemporalSequentialGcomp.as_str()));
        } else if self.settings.inference.is_bayesian() && q.effect_modifiers.is_empty() {
            // A declared white-innovation premise changes the posterior, so it changes the
            // estimator the plan names; builder validation has already refused it elsewhere.
            let estimator = match &self.settings.inference {
                InferenceMode::Bayesian(config) if config.white_innovation.is_some() => {
                    EstimatorId::BayesianTemporalWhiteInnovation
                }
                _ => EstimatorId::BayesianTemporalGcomp,
            };
            plan.record.estimator = Some(Arc::from(estimator.as_str()));
        }
        Ok(plan)
    }

    /// Logical plan of a pulse, sustained or conditional contrast on a temporal ADMG.
    ///
    /// The contrast is identified by the complete ID algorithm (IDC with modifiers) on the
    /// unrolled window of the template and evaluated by the functional plug-in on the lag-aligned
    /// window table, so the plan names `general.id` and `functional.effect`; a different pair
    /// would answer another question.
    pub(super) fn compile_logical_temporal_effect_admg(
        &self,
        data: &TimeSeriesData,
        q: &TemporalEffectQuery,
        graph: &TemporalAdmg,
        ctx: &ExecutionContext,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.compile_logical_temporal_effect_admg_within(data, q, graph, &mut budget, ctx)
    }

    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the operation keeps its checked inputs and execution context together"
    )]
    pub(super) fn compile_logical_temporal_effect_admg_within(
        &self,
        data: &TimeSeriesData,
        q: &TemporalEffectQuery,
        graph: &TemporalAdmg,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let window = antecedent_identify::temporal_admg::plan_temporal_admg_window_within(
            graph, q, budget, ctx,
        )?;
        let count = (graph.nodes().len() as u64)
            .checked_add(q.effect_modifiers.len() as u64)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| CausalError::Resource {
                message: "temporal planning count overflow".into(),
            })?;
        budget
            .charge_work(count.checked_mul(128).ok_or_else(|| CausalError::Resource {
                message: "temporal planning work overflow".into(),
            })?)
            .map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = count.checked_mul(256).and_then(|n| n.checked_add(4096)).ok_or_else(|| {
            CausalError::Resource { message: "temporal planning bytes overflow".into() }
        })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal planning byte limit".into() });
        }
        let _lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal planning" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut template_variables = std::collections::BTreeSet::<VariableId>::new();
        for node in graph.nodes() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal planning" });
            }
            if let antecedent_core::NodeRef::Lagged { variable, .. } = node {
                template_variables.insert(*variable);
            }
        }
        for variable in
            [q.treatment, q.outcome].into_iter().chain(q.effect_modifiers.iter().copied())
        {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal planning" });
            }
            if !template_variables.contains(&variable) {
                return Err(crate::compile_reason!(
                    "unknown_variable",
                    "variable {} is not a variable of the temporal ADMG template; name a \
                     variable that appears in the template at some lag",
                    variable.raw()
                ));
            }
        }
        let schema_len = data.schema().len();
        if let Some(variable) = template_variables.iter().find(|v| v.as_usize() >= schema_len) {
            return Err(crate::compile_reason!(
                "unknown_variable",
                "template variable {} is outside the series schema of {schema_len} variables; \
                 the template names variables by the schema's dense ids",
                variable.raw()
            ));
        }
        if data.storage().weights().is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "row sampling weights do not apply to a temporal ADMG contrast: the window table \
                 is read from the unweighted series. Next attempt: drop the weights, or use a \
                 TemporalDag whose linear fit reads them"
            ));
        }
        if q.max_history_lag.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a declared history cap does not apply to a temporal ADMG contrast: the window \
                 holds the template's own lags, and a shorter history would silently turn their \
                 causes into latent ones. Next attempt: drop the cap, or edit the template"
            ));
        }
        if self.declared.split.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a supplied TemporalAdmg needs no discovery/estimation split: the window table \
                 is built from the whole series. Next attempt: drop the split, or discover the \
                 structure first and analyse the accepted graph"
            ));
        }
        let (identifier_id, estimator_id) = self.resolve_admg_ids();
        if identifier_id != IdentifierId::GeneralId || estimator_id != EstimatorId::FunctionalEffect
        {
            return Err(crate::compile_reason!(
                "strategy_incompatible",
                "a contrast on a TemporalAdmg is identified by {:?} on the unrolled window and \
                 evaluated by {:?}; got identifier {:?} and estimator {:?}",
                IdentifierId::GeneralId.as_str(),
                EstimatorId::FunctionalEffect.as_str(),
                identifier_id.as_str(),
                estimator_id.as_str()
            ));
        }
        let (identifier, estimator) = wire_pair((identifier_id, estimator_id));
        let classification = match &self.data {
            DataInput::Event(_) => DataClassification::Event,
            _ => DataClassification::Temporal,
        };
        let record = antecedent_contract::LogicalAnalysisPlanRecord {
            plan_id: Arc::from(PlanId::TemporalAdmgContrast.as_str()),
            data_classification: classification,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(identifier),
            estimator: Some(estimator),
            validation_suite: self.validation_suite_id(),
            query_variables: Arc::from([q.treatment, q.outcome]),
        };
        let mut logical = LogicalAnalysisPlan::unsplit(
            record,
            CausalQuery::TemporalEffect(q.clone()),
            data.row_count() as u64,
        );
        logical.design_columns_hint = u64::from(window.periods)
            .checked_mul(template_variables.len() as u64)
            .ok_or_else(|| CausalError::Resource {
                message: "temporal design column count overflow".into(),
            })?
            .max(1);
        logical.validate()?;
        Ok(logical)
    }

    pub(super) fn compile_logical_temporal_response_dag(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_response(data, graph, q, false)?;
        if self.settings.inference.is_bayesian() {
            plan.record.estimator = Some(Arc::from(EstimatorId::TemporalResponseBayesian.as_str()));
        }
        Ok(plan)
    }

    pub(super) fn compile_logical_temporal_response_class(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_response(data, &TemporalDag::empty(), q, false)?;
        plan.record.identifier = Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
        plan.record.estimator =
            Some(Arc::from(EstimatorId::temporal_response_for(&self.settings.inference).as_str()));
        Ok(plan)
    }

    /// Logical plan of a derivative at one horizon on a temporal ADMG.
    ///
    /// The derivative of a continuous treatment is the slope of the lag-aligned regression on an
    /// adjustment set of the unrolled window of the template, so the plan names the temporal
    /// response estimator of the inference mode; the window-ADMG adjustment is recorded in the
    /// identification derivation. Curves and intervention levels need a plug-in the
    /// latent-confounded window does not evaluate, so they are refused here.
    pub(super) fn compile_logical_temporal_response_admg(
        &self,
        data: &TimeSeriesData,
        q: &ResponseQuery,
        graph: &TemporalAdmg,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        if !super::super::response_functional_is_derivative(&q.functional) {
            return Err(CausalError::Unsupported {
                message: "a temporal response on a TemporalAdmg is supported for derivatives, \
                          elasticities, directional derivatives and Jacobians; a curve or an \
                          intervention level of a latent-confounded window has no \
                          continuous-treatment functional estimator",
            });
        }
        let template_variables: std::collections::BTreeSet<VariableId> = graph
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
                _ => None,
            })
            .collect();
        for variable in q.functional.treatment_ids().into_iter().chain(q.functional.outcome_ids()) {
            if !template_variables.contains(&variable) {
                return Err(crate::compile_reason!(
                    "unknown_variable",
                    "variable {} is not a variable of the temporal ADMG template; name a \
                     variable that appears in the template at some lag",
                    variable.raw()
                ));
            }
        }
        let schema_len = data.schema().len();
        if let Some(variable) = template_variables.iter().find(|v| v.as_usize() >= schema_len) {
            return Err(crate::compile_reason!(
                "unknown_variable",
                "template variable {} is outside the series schema of {schema_len} variables; \
                 the template names variables by the schema's dense ids",
                variable.raw()
            ));
        }
        if data.storage().weights().is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "row sampling weights do not apply to a temporal ADMG derivative: the slope \
                 regression reads the unweighted series. Next attempt: drop the weights, or use \
                 a TemporalDag whose linear fit reads them"
            ));
        }
        if q.temporal.as_ref().is_some_and(|temporal| temporal.max_history_lag.is_some()) {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a declared history cap does not apply to a temporal ADMG derivative: the window \
                 holds the template's own lags, and a shorter history would silently turn their \
                 causes into latent ones. Next attempt: drop the cap, or edit the template"
            ));
        }
        if self.declared.split.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a supplied TemporalAdmg needs no discovery/estimation split: the slope \
                 regression reads the whole series. Next attempt: drop the split, or discover \
                 the structure first and analyse the accepted graph"
            ));
        }
        let mut plan = compile_logical_temporal_response(data, &TemporalDag::empty(), q, false)?;
        plan.record.identifier = Some(Arc::from(IdentifierId::GeneralizedAdjustment.as_str()));
        plan.record.estimator =
            Some(Arc::from(EstimatorId::temporal_response_for(&self.settings.inference).as_str()));
        Ok(plan)
    }

    pub(super) fn compile_logical_multi_env_temporal_effect(
        &self,
        multi: &MultiEnvironmentData,
        q: &TemporalEffectQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let data = multi
            .environment(0)
            .map_err(|e| CausalError::Compile { message: format!("multi-env: {e}") })?;
        compile_logical_temporal_effect_classified(
            data,
            graph,
            q,
            self.declared.split,
            false,
            DataClassification::MultiEnvironment,
        )
    }

    pub(super) fn compile_logical_panel_response_dag(
        &self,
        panel: &PanelData,
        q: &ResponseQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan =
            compile_logical_temporal_response(panel_compile_series(panel)?, graph, q, false)?;
        plan.record.data_classification = DataClassification::Panel;
        plan.record.estimator =
            Some(Arc::from(EstimatorId::temporal_response_for(&self.settings.inference).as_str()));
        Ok(plan)
    }

    pub(super) fn compile_logical_panel_effect_dag(
        &self,
        panel: &PanelData,
        q: &TemporalEffectQuery,
        graph: &TemporalDag,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        let mut plan = compile_logical_temporal_effect_classified(
            panel_compile_series(panel)?,
            graph,
            q,
            self.declared.split,
            false,
            DataClassification::Panel,
        )?;
        if let Some(
            link @ (EstimatorId::TemporalGlmLinkContrast
            | EstimatorId::BayesianTemporalGlmLinkContrast),
        ) = self.declared.estimator
        {
            plan.record.estimator = Some(Arc::from(link.as_str()));
        } else if self.settings.inference.is_bayesian() {
            plan.record.estimator = Some(Arc::from(EstimatorId::BayesianTemporalGcomp.as_str()));
        }
        Ok(plan)
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    /// Identify and estimate a continuous response, including licensed observation correction.
    pub(super) fn execute_response(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        _ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_RESPONSE_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_RESPONSE_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let _: EstimatorId = estimator.parse()?;

        let identify_cached = self.identification_cache.is_some();
        let (identification, estimand) = if let Some(cache) = self.identification_cache.as_deref() {
            (cache.identification.clone(), cache.estimand.clone())
        } else {
            let identification =
                identify_static_query(identifier_id, graph, &CausalQuery::Response(query.clone()))?;
            let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                CausalError::Compile { message: "response identifier returned no estimand".into() }
            })?;
            (identification, estimand)
        };
        if identification
            .estimands
            .iter()
            .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Unsupported {
                message: "multi-pair response estimation requires one common adjustment set",
            });
        }

        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            response_estimator.options = options.clone();
        }
        let response = if query.observation == ObservationSpec::Complete {
            response_estimator.estimate_identified(
                data,
                query,
                identification.status,
                identification.required_assumptions.clone(),
            )
        } else {
            ObservationMechanismEstimator::default().estimate_mean_curve(
                &response_estimator,
                data,
                query,
                None,
                identification.status,
                identification.required_assumptions.clone(),
            )
        }
        .map_err(CausalError::from)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(Diagnostic::new(
                "exec.identify.cached",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "identification reused from the prepare-time cache".to_string(),
            ));
        }
        diagnostics.push(Diagnostic::new(
            "refute.response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued response",
        ));
        if !scalar.is_finite() {
            // `result.effect` exists for every plan, but a curve, vector, Jacobian or
            // identified set has no scalar reading. The NaN is a "not applicable", not a
            // failed computation, and the caller must read `result.response` instead.
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let (id_artifact, id_op) = identify_provenance_step(identifier_id);
        let (est_artifact, est_op) = if query.observation == ObservationSpec::Complete {
            (response.provenance_id.as_ref(), response.provenance_id.as_ref())
        } else {
            ("estimate.response.observation_adjusted", "estimate.response.observation_adjusted")
        };
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &response.assumptions),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        let mut result = assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: None,
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        });
        result.response = Some(response);
        Ok(result)
    }
}

fn response_scalar_summary(response: &antecedent_core::CausalResponse) -> (f64, f64) {
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => value,
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            return (f64::NAN, f64::NAN);
        }
    };
    let scalar = match value {
        ResponseValue::Scalar(value) => *value,
        ResponseValue::Surface { .. }
        | ResponseValue::Vector(_)
        | ResponseValue::Jacobian { .. }
        | ResponseValue::Envelope(_) => f64::NAN,
    };
    let se = match response.uncertainty {
        ResponseUncertainty::Scalar { standard_error, .. } => standard_error,
        _ => f64::NAN,
    };
    (scalar, se)
}

fn response_primary_pair(
    functional: &ResponseFunctional,
) -> Result<(VariableId, VariableId), CausalError> {
    match functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => Ok((treatment.variable, *outcome)),
        ResponseFunctional::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
            Ok((*treatment, *outcome))
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
            treatments.first().zip(outcomes.first()).map(|(t, y)| (*t, *y)).ok_or_else(|| {
                CausalError::Compile {
                    message: "response query has no treatment/outcome pair".into(),
                }
            })
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => interventions
            .iter()
            .find_map(antecedent_core::Intervention::primary_variable)
            .map(|treatment| (treatment, *outcome))
            .ok_or_else(|| CausalError::Compile {
                message: "intervention response has no primary treatment variable".into(),
            }),
    }
}

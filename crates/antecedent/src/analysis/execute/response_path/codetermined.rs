//! Codetermined joint-response execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, ContinuousResponseEstimator, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecuteRun, ExecutionContext, IdentifiedClaim,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    OverlapPolicy, PhysicalExecutionPlan, ResponseQuery, RouteDefaults, Strategy, StudyResult,
    TableView, TabularData, elapsed_ns, identification_from_cache_or, identify_cached_diagnostic,
    provenance_ids, response_primary_pair, response_scalar_summary,
};

impl super::super::Study {
    pub(in crate::analysis::execute) fn execute_codetermined_joint_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let background = self.declared.tiered.as_ref().ok_or_else(|| CausalError::Compile {
            message: "CoDetermined joint execute requires a tiered background".into(),
        })?;
        let bayesian = self.settings.inference.is_bayesian();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults {
            identifier: IdentifierId::GeneralizedAdjustment,
            estimator: if bayesian { EstimatorId::ResponseBayesian } else { EstimatorId::CellAipw },
        })?;
        let expected = if bayesian { EstimatorId::ResponseBayesian } else { EstimatorId::CellAipw };
        if estimator_id != expected {
            return Err(CausalError::Unsupported {
                message: "CoDetermined joint cells require estimator cell.aipw (Frequentist) or \
                          response.bayesian (Bayesian)",
            });
        }
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let identification = match self.graph.as_admg() {
                    Some(admg) => {
                        antecedent_identify::identify_tiered_joint_on(background, admg, query)?
                    }
                    None => antecedent_identify::identify_tiered_joint(
                        background,
                        data.schema(),
                        query,
                    )?,
                };
                let estimand =
                    identification.estimands.first().cloned().ok_or(CausalError::Unsupported {
                        message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                    })?;
                Ok((identification, estimand))
            })?;
        if bayesian {
            return self.execute_codetermined_bayesian_joint(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
            );
        }
        self.execute_cell_aipw_response(
            data,
            query,
            ExecuteRun { physical, ctx, started, identify_cached },
            IdentifiedClaim { identification, estimand },
            Strategy::new(identifier_id, estimator_id),
            None,
        )
    }

    /// Bayesian joint level on a `CoDetermined` tier closure: response.bayesian over the
    /// closure's generalized adjustment set, one posterior over every coordinate of the
    /// joint arm. The outcome regression is additive in the treatments, so the level is
    /// that model's joint level, not the saturated cell mean cell.aipw estimates. The
    /// route executes and is unsupported.
    fn execute_codetermined_bayesian_joint(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let InferenceMode::Bayesian(config) = &self.settings.inference else {
            return Err(CausalError::unsupported_inference(
                "the CoDetermined Bayesian joint level requires inference=Bayesian",
            ));
        };
        let mut fitter = ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            fitter.options = options.clone();
        }
        let estimator =
            super::super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                &fitter, data, query, config, ctx,
            )?;
        let response = fitter
            .estimate_bayesian(
                data,
                query,
                identification.status,
                identification.required_assumptions.clone(),
                &estimator,
                ctx,
            )
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
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.codetermined.bayesian_joint",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Bayesian joint level on the CoDetermined tier closure: response.bayesian over the \
             closure's generalized adjustment set, one posterior over every coordinate of the \
             joint arm. The outcome regression is additive in the treatments, so this is that \
             model's joint level, not the saturated cell mean of cell.aipw. It executes \
             unsupported: no coverage or evidence record names it.",
        ));
        diagnostics.extend(response.support.warnings.iter().cloned());
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                strategy,
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }
}

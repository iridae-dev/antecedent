// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn execute_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) = if let Some(cache) =
            self.temporal_identification_cache.as_deref()
        {
            let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                message: "prepared panel identification cache missing query horizon".into(),
            })?;
            (entry.identification.clone(), entry.estimand.clone(), entry.indexer.clone(), true)
        } else {
            let id_res = TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, query)
                .map_err(CausalError::from)?;
            report_identify_compute(ctx);
            let identification = id_res.result;
            require_identified(&identification)?;
            let estimand = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)?;
            (identification, estimand, id_res.indexer, false)
        };
        require_identified(&identification)?;

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let (prep, cluster_ids, panel_times) = estimator
            .prepare_panel(
                panel,
                &estimand,
                query,
                &indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            )
            .map_err(CausalError::from)?;
        let max_lag = query.max_history_lag.unwrap_or(1).max(1) as usize;

        let (estimate, mut posterior, estimate_artifact, estimate_op) = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
                let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                bprep.unit_ids = Some(cluster_ids.clone());
                let (resolved_prior, conflict_summary) =
                    resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
                bayes.inner.prior = resolved_prior;
                let mut ws = BayesianGCompWorkspace::default();
                let mut posterior = bayes
                    .fit(&bprep, identification.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                if let Some(summary) = conflict_summary {
                    posterior = with_conflict_summary(posterior, summary);
                }
                let estimate = effect_from_posterior(&posterior)?;
                (
                    estimate,
                    Some(posterior),
                    "estimate.bayesian_temporal_gcomp.panel",
                    "estimate.bayesian.temporal.gcomp.panel",
                )
            }
            InferenceMode::Frequentist => {
                estimator.inner.cluster_ids = Some(cluster_ids.clone());
                estimator.inner.panel_times = Some(panel_times);
                estimator.inner.se_kind = AnalyticSeKind::PanelClusterHac { lag: max_lag };
                let mut workspace = EstimationWorkspace::default();
                let estimate = estimator
                    .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
                    .map_err(CausalError::from)?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment.panel",
                    "estimate.temporal.linear.adjustment.panel",
                )
            }
        };

        let mut diagnostics = Vec::new();
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let temporal_ctx = TemporalRefitContext {
            indexer: &indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: None,
            panel: Some(panel),
        };
        let (mut refutations, na_diagnostics) = run_refuters(
            &stacked,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            if posterior.is_some() {
                EstimatorId::BayesianTemporalGcomp.as_str()
            } else {
                EstimatorId::TemporalLinearAdjustment.as_str()
            },
            &self.custom_validators,
            Some(temporal_ctx),
        )?;
        diagnostics.extend(na_diagnostics);

        // Panel Bayesian: α-grid under Full when external compose is present (mirror temporal).
        if matches!(self.refute, RefuteSuite::Full) {
            if let (InferenceMode::Bayesian(cfg), Some(post)) = (&self.inference, &posterior) {
                let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                bprep.unit_ids = Some(cluster_ids.clone());
                posterior = Some(apply_temporal_prior_sensitivity(
                    cfg,
                    &bprep,
                    identification.status,
                    post,
                    estimate.ate,
                    ctx,
                    &mut refutations,
                )?);
            }
        }

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: if posterior.is_some() {
                EstimatorId::BayesianTemporalGcomp
            } else {
                EstimatorId::TemporalLinearAdjustment
            },
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(estimate_artifact, estimate_op)),
                posterior,
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_panel_class(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::builder::refuse_unlicensed_panel_effect(
            query,
            self.graph.class(),
            &self.inference,
        )?;
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_PAG_IDENTIFIER_ID.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let (bundle, identify_cached) =
            if let Some(cache) = self.temporal_class_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, query)?, false)
            };
        if query.is_multi_step_sustained() {
            return self.execute_panel_class_sequential(panel, query, physical, ctx);
        }
        let envelope = &bundle.envelope.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::Compile {
                message: "panel class-aware effect not identified (no identified mass in envelope)"
                    .into(),
            });
        }
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut fitted = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let bayesian = matches!(self.inference, InferenceMode::Bayesian(_));
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let estimate = if bayesian {
                fit_panel_pulse_atom_bayesian(
                    panel,
                    &estimand,
                    query,
                    indexer,
                    self.split.as_ref(),
                    &self.inference,
                    ctx,
                    case.result.status,
                    case.result.required_assumptions.clone(),
                )?
            } else {
                fit_panel_pulse_atom(
                    panel,
                    &estimand,
                    query,
                    indexer,
                    self.split.as_ref(),
                    ctx,
                    case.result.required_assumptions.clone(),
                )?
            };
            let w = case.weight.0;
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            total_w += w;
            atom_values[i] = Some(estimate.ate);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            fitted.push((w, estimand, indexer.clone()));
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "panel class-aware envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware envelope missing estimand".into(),
        })?;
        let ate = weighted_ate / total_w;
        let se_analytic = mix_weighted_analytic_se(se_items);
        let (se_bootstrap, bootstrap_ok, bootstrap_failed, cancelled) =
            if self.bootstrap_replicates > 0 && panel.unit_count() >= 2 {
                cluster_bootstrap_panel_class(
                    panel,
                    query,
                    &fitted,
                    self.split.as_ref(),
                    self.bootstrap_replicates,
                    ctx,
                )
            } else {
                (None, None, None, ctx.cancellation.is_cancelled())
            };
        if se_bootstrap.is_none() && fitted.len() > 1 {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_effect.panel.class.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel class Pulse fits {} identified completions on {} units; \
                 mixture weights are completion masses, not stacked-row circular blocks",
                fitted.len(),
                panel.unit_count()
            ),
        ));
        if bayesian {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_effect.panel.class.bayesian_units",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "each completion uses the panel hierarchical GLS likelihood with unit ids; \
                 the series class-posterior is not reused",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let estimate = EffectEstimate::from_parts(
            ate,
            se_analytic,
            se_bootstrap,
            bootstrap_ok,
            bootstrap_failed,
            cancelled,
            false,
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        );
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &atom_values,
        );
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: if bayesian {
                EstimatorId::BayesianTemporalGcomp
            } else {
                EstimatorId::TemporalLinearAdjustment
            },
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: bundle.envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                bootstrap_replicates_requested: (self.bootstrap_replicates > 0)
                    .then_some(Some(self.bootstrap_replicates)),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_panel_response(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::builder::refuse_unlicensed_panel_response(
            query,
            self.graph.class(),
            &self.inference,
        )?;
        let started = Instant::now();
        if query.observation != ObservationSpec::Complete {
            return Err(CausalError::Unsupported {
                message: "panel response is licensed for complete observations only",
            });
        }
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "panel response route requires TemporalResponseSpec".into(),
            });
        };
        if antecedent_estimate::plan_from_response_query(query)
            .map_err(CausalError::from)?
            .and_then(|plan| plan.mechanism_overlays())
            .is_some()
        {
            return Err(CausalError::Unsupported {
                message: "panel multi-step Sequence overlays are not licensed; keep the \
                          series sequential owner",
            });
        }
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                report_identify_compute(ctx);
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_response_horizons(
                        graph,
                        treatment,
                        outcome,
                        temporal,
                        &query.target_population,
                        EstimatorId::TemporalResponseGcomp,
                        None,
                    )?),
                    false,
                )
            };
        let mut aligned = Vec::with_capacity(temporal.horizons.len());
        for &horizon in temporal.horizons.iter() {
            let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
                message: format!("temporal identification missing horizon {horizon}"),
            })?;
            require_identified(&entry.identification)?;
            aligned.push(entry);
        }
        let first = aligned[0];
        let (aggregate_status, aggregate_assumptions) =
            super::temporal_path::aggregate_temporal_horizon_evidence(
                aligned.iter().map(|entry| &entry.identification),
            )?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

        let unit_responses = fit_unit_panel_responses(
            panel,
            &identifications,
            query,
            aggregate_status,
            aggregate_assumptions.clone(),
            &self.inference,
            ctx,
        )?;
        let mut response = average_unit_panel_responses(&unit_responses)?;
        if self.bootstrap_replicates > 0 && unit_responses.len() >= 2 {
            cluster_bootstrap_panel_response(
                &mut response,
                &unit_responses,
                self.bootstrap_replicates,
                ctx,
            );
        }
        if let Some(per_horizon) = response.horizon_identification.as_mut() {
            for (record, entry) in Arc::make_mut(per_horizon).iter_mut().zip(&aligned) {
                record.status = entry.identification.status;
            }
        }

        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = Vec::new();
        for entry in &aligned {
            diagnostics.extend(entry.identification.diagnostics.iter().cloned());
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_response.panel.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel response averages {} unit surfaces; pointwise bands use between-unit \
                 cluster variation, not stacked-row circular blocks",
                unit_responses.len()
            ),
        ));
        diagnostics.push(Diagnostic::new(
            "response.simultaneous_band_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "panel response withholds a simultaneous band: series circular-block simultaneous \
             coverage does not transfer to a unit-cluster contract",
        ));
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            },
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_panel_class_response(
        &self,
        panel: &PanelData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::builder::refuse_unlicensed_panel_response(
            query,
            self.graph.class(),
            &self.inference,
        )?;
        let started = Instant::now();
        if query.observation != ObservationSpec::Complete {
            return Err(CausalError::Unsupported {
                message: "panel response is licensed for complete observations only",
            });
        }
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "panel response route requires TemporalResponseSpec".into(),
            });
        };
        if antecedent_estimate::plan_from_response_query(query)
            .map_err(CausalError::from)?
            .and_then(|plan| plan.mechanism_overlays())
            .is_some()
        {
            return Err(CausalError::Unsupported {
                message: "panel multi-step Sequence overlays are not licensed; keep the \
                          series sequential owner",
            });
        }
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(IdentifierId::GeneralizedAdjustment.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let effect_query = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
            active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
            horizon_steps: temporal.horizons.first().copied().unwrap_or(1),
            max_history_lag: temporal.max_history_lag,
            target_population: query.target_population.clone(),
        };
        let (bundle, identify_cached) =
            if let Some(cache) = self.temporal_class_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, &effect_query)?, false)
            };
        let envelope = &bundle.envelope.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::Compile {
                message:
                    "panel class-aware response not identified (no identified mass in envelope)"
                        .into(),
            });
        }
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        let mut weighted = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                primary_identification = Some(case.result.clone());
                assumptions = case.result.required_assumptions.clone();
            }
            let identifications: Vec<_> =
                temporal.horizons.iter().map(|_| (&estimand, indexer)).collect();
            let unit_responses = fit_unit_panel_responses(
                panel,
                &identifications,
                query,
                case.result.status,
                case.result.required_assumptions.clone(),
                &self.inference,
                ctx,
            )?;
            let response = average_unit_panel_responses(&unit_responses)?;
            if let Some(mean) = surface_mean(&response) {
                atom_values[i] = mean.first().copied();
            }
            weighted.push((case.weight.0, response));
        }
        if weighted.is_empty() {
            return Err(CausalError::Compile {
                message: "panel class-aware response had no estimable identified cases".into(),
            });
        }
        let response = mix_weighted_panel_responses(&weighted)?;
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            assumptions,
            OverlapPolicy::ExplicitOverride,
        );
        let identification = primary_identification.unwrap_or_else(|| {
            envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()))
        });
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware response missing estimand".into(),
        })?;
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_response.panel.class.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel class response averages unit surfaces on {} identified completions \
                 and mixes by completion mass; units are not stacked",
                weighted.len()
            ),
        ));
        diagnostics.push(Diagnostic::new(
            "response.simultaneous_band_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "panel response withholds a simultaneous band: series circular-block simultaneous \
             coverage does not transfer to a unit-cluster contract",
        ));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &atom_values,
        );
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            },
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: bundle.envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    fn execute_panel_class_sequential(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_PAG_IDENTIFIER_ID.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let (bundle, identify_cached) =
            if let Some(cache) = self.temporal_class_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, query)?, false)
            };
        let envelope = &bundle.envelope.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::Compile {
                message: "panel class-aware effect not identified (no identified mass in envelope)"
                    .into(),
            });
        }
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.sustained_window",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each completion's sequential contrast is fit per panel unit and averaged; \
             units are not stacked onto the series sequential class owner",
        ));
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let Some(dag) = case.graph.sequential_dag() else {
                continue;
            };
            let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
                .or_else(|_| {
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
                })?;
            let estimate = fit_panel_sequential_atom(
                panel,
                &dag,
                indexer,
                &estimand,
                query,
                case.result.status,
                case.result.required_assumptions.clone(),
                bayes.as_ref(),
                ctx,
            )?;
            let w = case.weight.0;
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            total_w += w;
            atom_values[i] = Some(estimate.ate);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
                assumptions = estimate.assumptions.clone();
            }
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "panel class-aware sequential envelope had no estimable identified cases"
                    .into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware sequential envelope missing estimand".into(),
        })?;
        let ate = weighted_ate / total_w;
        let se_analytic = mix_weighted_analytic_se(se_items);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let estimate = EffectEstimate::from_parts(
            ate,
            se_analytic,
            None,
            None,
            None,
            ctx.cancellation.is_cancelled(),
            false,
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        );
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &atom_values,
        );
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: EstimatorId::TemporalSequentialGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: bundle.envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }
}

const NORMAL_Z_95: f64 = 1.959_963_984_540_054;

fn surface_mean(response: &CausalResponse) -> Option<&[f64]> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => Some(mean),
        _ => None,
    }
}

fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>()
        / (values.len() - 1) as f64;
    Some(var.sqrt())
}

fn normal_pointwise_band(centers: &[f64], ses: impl Iterator<Item = f64>) -> ResponseUncertainty {
    let mut lower = Vec::with_capacity(centers.len());
    let mut upper = Vec::with_capacity(centers.len());
    for (center, se) in centers.iter().zip(ses) {
        lower.push(center - NORMAL_Z_95 * se);
        upper.push(center + NORMAL_Z_95 * se);
    }
    ResponseUncertainty::PointwiseBand { level: 0.95, lower: lower.into(), upper: upper.into() }
}

fn sample_unit_index(rng: &mut antecedent_core::CausalRng, n: usize) -> usize {
    (rng.next_u64() as usize) % n
}

fn resample_panel_units(
    panel: &PanelData,
    rng: &mut antecedent_core::CausalRng,
) -> Result<PanelData, CausalError> {
    let n = panel.unit_count();
    let sampled: Vec<_> =
        (0..n).map(|_| panel.units()[sample_unit_index(rng, n)].clone()).collect();
    PanelData::try_new(sampled)
        .map_err(|e| CausalError::Compile { message: format!("panel resample: {e}") })
}

fn average_unit_panel_responses(units: &[CausalResponse]) -> Result<CausalResponse, CausalError> {
    let first = units.first().ok_or_else(|| CausalError::Compile {
        message: "panel response requires at least one unit".into(),
    })?;
    let (grid, dimension, n_cells) = match &first.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid,
            dimension,
            mean,
        }) => (Arc::clone(grid), *dimension, mean.len()),
        _ => {
            return Err(CausalError::Compile {
                message: "panel response requires a point-identified surface per unit".into(),
            });
        }
    };
    let n = units.len() as f64;
    let mut acc = vec![0.0; n_cells];
    let mut unit_means = Vec::with_capacity(units.len());
    for unit in units {
        match &unit.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: unit_grid,
                dimension: unit_dim,
                mean,
            }) if unit_grid.as_ref() == grid.as_ref()
                && *unit_dim == dimension
                && mean.len() == n_cells =>
            {
                unit_means.push(mean.clone());
                for (total, value) in acc.iter_mut().zip(mean.iter()) {
                    *total += *value;
                }
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "panel unit surfaces disagree on grid or identification".into(),
                });
            }
        }
    }
    for value in &mut acc {
        *value /= n;
    }
    let uncertainty = panel_cluster_pointwise_band(&acc, &unit_means);
    let mut response = first.clone();
    response.estimate = ResponseIdentification::PointIdentified(ResponseValue::Surface {
        grid,
        dimension,
        mean: acc.into(),
    });
    response.uncertainty = uncertainty;
    response.provenance_id = Arc::from("estimate.temporal_response.gcomp.panel");
    Ok(response)
}

fn panel_cluster_pointwise_band(mean: &[f64], unit_means: &[Arc<[f64]>]) -> ResponseUncertainty {
    let n = unit_means.len();
    if n < 2 {
        return ResponseUncertainty::None;
    }
    let df = (n - 1) as f64;
    normal_pointwise_band(
        mean,
        (0..mean.len()).map(|j| {
            let var = unit_means
                .iter()
                .map(|row| {
                    let delta = row[j] - mean[j];
                    delta * delta
                })
                .sum::<f64>()
                / df;
            (var / n as f64).sqrt()
        }),
    )
}

fn cluster_bootstrap_panel_response(
    response: &mut CausalResponse,
    units: &[CausalResponse],
    replicates: u32,
    ctx: &ExecutionContext,
) {
    let Some(mean) = surface_mean(response) else {
        return;
    };
    let n_cells = mean.len();
    let n_units = units.len();
    let mut rng = ctx.rng.stream(0x50A4_u64);
    let mut draws = vec![Vec::with_capacity(replicates as usize); n_cells];
    for _ in 0..replicates {
        let mut acc = vec![0.0; n_cells];
        for _ in 0..n_units {
            if let Some(unit_mean) =
                units.get(sample_unit_index(&mut rng, n_units)).and_then(surface_mean)
            {
                for (total, value) in acc.iter_mut().zip(unit_mean.iter()) {
                    *total += *value;
                }
            }
        }
        let scale = n_units as f64;
        for (cell, value) in draws.iter_mut().zip(acc.iter()) {
            cell.push(*value / scale);
        }
    }
    let mut lower = Vec::with_capacity(n_cells);
    let mut upper = Vec::with_capacity(n_cells);
    for cell_draws in draws {
        let m = cell_draws.iter().sum::<f64>() / cell_draws.len() as f64;
        let se = sample_sd(&cell_draws).unwrap_or(0.0);
        lower.push(m - NORMAL_Z_95 * se);
        upper.push(m + NORMAL_Z_95 * se);
    }
    response.uncertainty = ResponseUncertainty::PointwiseBand {
        level: 0.95,
        lower: lower.into(),
        upper: upper.into(),
    };
}

fn fit_unit_panel_responses(
    panel: &PanelData,
    identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
    query: &ResponseQuery,
    status: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    inference: &InferenceMode,
    ctx: &ExecutionContext,
) -> Result<Vec<CausalResponse>, CausalError> {
    let mut estimator = TemporalResponseEstimator::new();
    estimator.inner.bootstrap_replicates = 0;
    let mut unit_responses = Vec::with_capacity(panel.unit_count());
    match inference {
        InferenceMode::Bayesian(cfg) => {
            let bayes = bayesian_gcomp(cfg, ctx);
            for unit in panel.units() {
                unit_responses.push(
                    estimator
                        .estimate_bayesian(
                            &unit.series,
                            identifications,
                            query,
                            status,
                            assumptions.clone(),
                            &bayes,
                            ctx,
                        )
                        .map_err(CausalError::from)?,
                );
            }
        }
        InferenceMode::Frequentist => {
            for unit in panel.units() {
                unit_responses.push(
                    estimator
                        .estimate(
                            &unit.series,
                            identifications,
                            query,
                            status,
                            assumptions.clone(),
                            ctx,
                        )
                        .map_err(CausalError::from)?,
                );
            }
        }
    }
    Ok(unit_responses)
}

fn mix_weighted_panel_responses(
    weighted: &[(f64, CausalResponse)],
) -> Result<CausalResponse, CausalError> {
    let first = weighted.first().ok_or_else(|| CausalError::Compile {
        message: "panel class response requires at least one identified completion".into(),
    })?;
    let (grid, dimension, n_cells) = match &first.1.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid,
            dimension,
            mean,
        }) => (Arc::clone(grid), *dimension, mean.len()),
        _ => {
            return Err(CausalError::Compile {
                message: "panel class response requires a point-identified surface per completion"
                    .into(),
            });
        }
    };
    let mut acc = vec![0.0; n_cells];
    let mut total_w = 0.0;
    for (weight, response) in weighted {
        match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: unit_grid,
                dimension: unit_dim,
                mean,
            }) if unit_grid.as_ref() == grid.as_ref()
                && *unit_dim == dimension
                && mean.len() == n_cells =>
            {
                total_w += *weight;
                for (total, value) in acc.iter_mut().zip(mean.iter()) {
                    *total += *weight * *value;
                }
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "panel class completion surfaces disagree on grid or identification"
                        .into(),
                });
            }
        }
    }
    if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return Err(CausalError::Compile {
            message: "panel class response had no positive completion mass".into(),
        });
    }
    for value in &mut acc {
        *value /= total_w;
    }
    let mut response = first.1.clone();
    response.estimate = ResponseIdentification::PointIdentified(ResponseValue::Surface {
        grid,
        dimension,
        mean: acc.into(),
    });
    response.provenance_id = Arc::from("estimate.temporal_response.gcomp.panel.class");
    Ok(response)
}

fn fit_panel_pulse_atom_bayesian(
    panel: &PanelData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    split: Option<&DiscoveryEstimationSplit>,
    inference: &InferenceMode,
    ctx: &ExecutionContext,
    status: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
) -> Result<EffectEstimate, CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Err(CausalError::Compile {
            message: "panel Bayesian atom requires Bayesian inference".into(),
        });
    };
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    let (prep, cluster_ids, _) = estimator
        .prepare_panel(panel, estimand, query, indexer, split, &ctx.kernel_policy)
        .map_err(CausalError::from)?;
    let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
    let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
    bprep.unit_ids = Some(cluster_ids);
    let (resolved_prior, _) = resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
    bayes.inner.prior = resolved_prior;
    let mut ws = BayesianGCompWorkspace::default();
    let posterior = bayes.fit(&bprep, status, &mut ws, ctx).map_err(CausalError::from)?;
    let mut estimate = effect_from_posterior(&posterior)?;
    estimate.assumptions = assumptions;
    Ok(estimate)
}

fn fit_panel_sequential_atom(
    panel: &PanelData,
    dag: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    status: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    bayes: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let mut ates = Vec::with_capacity(panel.unit_count());
    let mut last_assumptions = assumptions.clone();
    for unit in panel.units() {
        let (estimate, _) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                &unit.series,
                dag,
                indexer,
                estimand,
                query,
                status,
                assumptions.clone(),
                0,
                bayes,
                ctx,
                None,
            )
            .map_err(CausalError::from)?;
        if estimate.ate.is_finite() {
            ates.push(estimate.ate);
            last_assumptions = estimate.assumptions;
        }
    }
    if ates.is_empty() {
        return Err(CausalError::Compile {
            message: "panel sequential atom had no finite unit estimates".into(),
        });
    }
    let n = ates.len() as f64;
    let ate = ates.iter().sum::<f64>() / n;
    let se = sample_sd(&ates).map(|sd| sd / n.sqrt()).unwrap_or(0.0);
    Ok(EffectEstimate::new(ate, se, last_assumptions, OverlapPolicy::ExplicitOverride))
}

fn fit_panel_pulse_atom(
    panel: &PanelData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    split: Option<&DiscoveryEstimationSplit>,
    ctx: &ExecutionContext,
    assumptions: antecedent_core::AssumptionSet,
) -> Result<EffectEstimate, CausalError> {
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    let (prep, cluster_ids, panel_times) = estimator
        .prepare_panel(panel, estimand, query, indexer, split, &ctx.kernel_policy)
        .map_err(CausalError::from)?;
    let max_lag = query.max_history_lag.unwrap_or(1).max(1) as usize;
    estimator.inner.cluster_ids = Some(cluster_ids);
    estimator.inner.panel_times = Some(panel_times);
    estimator.inner.se_kind = AnalyticSeKind::PanelClusterHac { lag: max_lag };
    let mut workspace = EstimationWorkspace::default();
    estimator.fit(&prep, &mut workspace, ctx, assumptions).map_err(CausalError::from)
}

fn cluster_bootstrap_panel_class(
    panel: &PanelData,
    query: &TemporalEffectQuery,
    atoms: &[(f64, IdentifiedEstimand, TemporalIndexer)],
    split: Option<&DiscoveryEstimationSplit>,
    replicates: u32,
    ctx: &ExecutionContext,
) -> (Option<f64>, Option<u32>, Option<u32>, bool) {
    let n_units = panel.unit_count();
    if n_units < 2 || atoms.is_empty() {
        return (None, None, None, ctx.cancellation.is_cancelled());
    }
    let mut rng = ctx.rng.stream(0xC1A5_5E11);
    let mut draws = Vec::with_capacity(replicates as usize);
    let mut completed = 0u32;
    let mut failed = 0u32;
    for _ in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            break;
        }
        let Ok(resampled) = resample_panel_units(panel, &mut rng) else {
            failed += 1;
            continue;
        };
        let mut weighted = 0.0;
        let mut total_w = 0.0;
        let mut ok = true;
        for (weight, estimand, indexer) in atoms {
            match fit_panel_pulse_atom(
                &resampled,
                estimand,
                query,
                indexer,
                split,
                ctx,
                antecedent_core::AssumptionSet::default(),
            ) {
                Ok(estimate) if estimate.ate.is_finite() => {
                    weighted += *weight * estimate.ate;
                    total_w += *weight;
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && total_w > 0.0 {
            draws.push(weighted / total_w);
            completed += 1;
        } else {
            failed += 1;
        }
    }
    if draws.len() < 2 {
        return (None, Some(completed), Some(failed), ctx.cancellation.is_cancelled());
    }
    (sample_sd(&draws), Some(completed), Some(failed), ctx.cancellation.is_cancelled())
}

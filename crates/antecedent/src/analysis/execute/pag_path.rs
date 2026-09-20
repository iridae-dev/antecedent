// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    /// ADMG ATE via general ID + functional plug-in (bidirected case).
    pub(super) fn execute_admg(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER);
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!("ADMG ATE requires estimator functional.effect; got {estimator}"),
            });
        }

        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_admg(identifier_id, admg, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;
        let (estimate, posterior) = self.estimate_functional_effect(
            data,
            &estimand,
            &identification,
            &[query.treatment, query.outcome],
            ctx,
        )?;

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra_diagnostics) = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
        )?;

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
    }

    /// ADMG response via general ID + functional plug-in (bidirected case).
    ///
    /// Bidirected edges stay bidirected. Each grid / Set level is the identified
    /// `P(Y | do(X=x))` mean, not an adjustment g-formula.
    pub(super) fn execute_admg_response(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let is_intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        if self.refute != RefuteSuite::None && !is_intervention {
            return Err(CausalError::Unsupported {
                message: "Admg ResponseCurve has no native non-ATE refuter suite; \
                          ATE refuters do not apply to a function-valued surface",
            });
        }
        match &query.functional {
            ResponseFunctional::InterventionResponse { .. } => self.finish_admg_response_levels(
                data,
                admg,
                query,
                &[admg_response_single_level(query)?],
                physical,
                ctx,
                started,
                false,
            ),
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                let levels = treatment
                    .grid
                    .values()
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let level_queries = levels
                    .iter()
                    .map(|level| {
                        admg_response_at_level(query, treatment.variable, *outcome, *level)
                    })
                    .collect::<Vec<_>>();
                self.finish_admg_response_levels(
                    data,
                    admg,
                    query,
                    &level_queries,
                    physical,
                    ctx,
                    started,
                    true,
                )
            }
            _ => Err(CausalError::Unsupported {
                message: "Admg response is licensed for InterventionResponse and ResponseCurve",
            }),
        }
    }

    fn finish_admg_response_levels(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &ResponseQuery,
        level_queries: &[ResponseQuery],
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        started: Instant,
        curve: bool,
    ) -> Result<StudyResult, CausalError> {
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER);
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "ADMG response requires estimator functional.effect; got {estimator}"
                ),
            });
        }
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let bayesian = matches!(self.inference, InferenceMode::Bayesian(_));
        let mut means = Vec::with_capacity(level_queries.len());
        let mut grid = Vec::with_capacity(level_queries.len());
        let mut identification = None;
        let mut estimand = None;
        let mut identify_cached = false;
        let mut extras_posterior = None;
        let mut n_draws = None;
        let mut support = None;
        let mut extra_diagnostics = Vec::new();
        for (i, level_query) in level_queries.iter().enumerate() {
            let (level_id, level_est, cached) = if i == 0 {
                identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                    let identification = identify_admg_query(
                        identifier_id,
                        admg,
                        &CausalQuery::Response(level_query.clone()),
                    )?;
                    let estimand = select_estimand(&identification, estimator_id)?;
                    Ok((identification, estimand))
                })?
            } else {
                let identification = identify_admg_query(
                    identifier_id,
                    admg,
                    &CausalQuery::Response(level_query.clone()),
                )?;
                let estimand = select_estimand(&identification, estimator_id)?;
                (identification, estimand, false)
            };
            if i == 0 {
                identify_cached = cached;
            }
            let (mean, posterior, level_support) = if bayesian {
                let (estimate, posterior) = self.estimate_functional_effect(
                    data,
                    &level_est,
                    &level_id,
                    &[treatment, outcome],
                    ctx,
                )?;
                (
                    estimate.ate,
                    posterior,
                    support_from_functional_eval(None).map_err(|e| {
                        CausalError::from(antecedent_estimate::EstimationError::data_msg(
                            e.to_string(),
                        ))
                    })?,
                )
            } else {
                let (response, _) = super::response_path::estimate_general_id_response(
                    data,
                    level_query,
                    &level_id,
                    &level_est,
                    ctx,
                )?;
                extra_diagnostics.extend(response.support.warnings.iter().cloned());
                let (scalar, _) = super::response_path::response_scalar_summary(&response);
                (scalar, None, response.support)
            };
            if let ResponseFunctional::InterventionResponse { interventions, .. } =
                &level_query.functional
            {
                if let Some(Intervention::Set { value, .. }) = interventions.first() {
                    if let Some(level) = value.as_f64() {
                        grid.push(level);
                    }
                }
            }
            means.push(mean);
            if extras_posterior.is_none() {
                extras_posterior.clone_from(&posterior);
                n_draws =
                    posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX));
            }
            if support.is_none() {
                support = Some(level_support);
            }
            if identification.is_none() {
                identification = Some(level_id);
                estimand = Some(level_est);
            }
        }
        let identification = identification.expect("ADMG response identified at least one level");
        let estimand = estimand.expect("ADMG response selected at least one estimand");
        let mut response_support = support.unwrap_or_else(|| {
            support_from_functional_eval(None).expect("empty functional eval publishes support")
        });
        if !grid.is_empty() {
            let (minima, maxima) = if curve {
                let min = grid.iter().copied().fold(f64::INFINITY, f64::min);
                let max = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (vec![min], vec![max])
            } else {
                (grid.clone(), grid.clone())
            };
            response_support.query_region = antecedent_core::SupportRegion {
                minima: Arc::from(minima),
                maxima: Arc::from(maxima),
            };
        }
        let estimate_payload = if curve {
            ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension: 1,
                mean: Arc::from(means),
            })
        } else {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(
                means.first().copied().unwrap_or(f64::NAN),
            ))
        };
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate: estimate_payload,
            uncertainty: ResponseUncertainty::None,
            support: response_support,
            assumptions: identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "identify.response.general_id",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "intervention mean identified by Shpitser–Pearl ID; levels are the \
             discrete functional.effect plug-in, not an adjustment g-formula",
        ));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.extend(extra_diagnostics);
        let (refutations, refute_diags) = if !curve
            && scalar.is_finite()
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; \
                     cheap runs overlap only and full runs overlap plus sampling-stability of the \
                     identified intervention mean",
                ));
            }
            run_plugin_level_refuters(
                data,
                &estimand,
                &ate_query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
            )?
        } else {
            (Vec::new(), Vec::new())
        };
        diagnostics.extend(refute_diags);
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.general_id",
                    "estimate.response.general_id",
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                n_draws,
                posterior: extras_posterior,
                ..Default::default()
            },
        }))
    }

    /// PAG ATE via generalized-adjustment envelope + mass-weighted estimates.
    pub(super) fn execute_pag(
        &self,
        data: &TabularData,
        pag: &Pag,
        query: &AverageEffectQuery,
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
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(DEFAULT_PAG_ESTIMATOR_ID.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        let (envelope, identification, identify_cached) =
            if let Some(cache) = self.pag_identification_cache.as_deref() {
                (cache.envelope.clone(), cache.identification.clone(), true)
            } else {
                report_identify_compute(ctx);
                let envelope = identify_pag(identifier_id, pag, query)?;
                let identification = envelope_to_identification_result(&envelope, query);
                (envelope, identification, false)
            };
        let certificate = crate::Identification::Envelope {
            envelope: envelope.clone(),
            strategy: identifier_id,
            structure_version: self.graph.version(),
        };
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if matches!(self.inference, InferenceMode::Bayesian(_))
                || matches!(estimator_id, EstimatorId::BayesianGcomp)
            {
                return self
                    .execute_pag_nonidentified_prior(
                        query,
                        physical,
                        ctx,
                        &envelope,
                        identification,
                        identify_cached,
                        started,
                        pag_envelope_diagnostic(&envelope),
                        "pag",
                    )
                    .map(|result| self.attach_certificate(result, certificate));
            }
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "PAG effect not identified (no identified mass in envelope)",
            ));
        }

        let mut diagnostics = vec![pag_envelope_diagnostic(&envelope)];

        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self
                .execute_pag_bayesian(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identification,
                    identify_cached,
                    started,
                    pag_envelope_diagnostic(&envelope),
                    "pag",
                )
                .map(|result| self.attach_certificate(result, certificate));
        }

        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut atom_ifs = Vec::new();
        let mut atom_weights = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let mut outcomes = vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()];
        for (i, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, estimator_id)?;
            let mut case_ws = StaticEstimateWorkspaces::default();
            // Honour a caller-configured estimator across every equivalence-class case,
            // falling back to id-only selection when none was supplied.
            let case_spec = self
                .estimator_spec
                .clone()
                .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id));
            let estimate = estimate_static_effect(
                &case_spec,
                data,
                &estimand,
                query,
                case.result.required_assumptions.clone(),
                self.bootstrap_replicates,
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut case_ws,
            )?;
            let w = case.weight.0;
            outcomes[i] = ClassAtomOutcome::Evaluated(estimate.ate);
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, &estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: None,
                original: estimate,
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "PAG envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "PAG envelope missing estimand".into(),
        })?;
        let n_contributing = se_items.len();
        let se = if atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        };
        let mut estimate = EffectEstimate::new(
            weighted_ate / total_w,
            se,
            assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
                estimate.influence = Some(inf);
            }
        }

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            data,
            query,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
            self.split.as_ref(),
            None,
        )?;
        diagnostics.extend(na_diagnostics);

        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic));
        // A partially identified envelope carries its identified set and the
        // per-completion values it spans; a point-identified one stays a point.
        let structural_response = static_class_structural_mixture(&envelope, &outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(static_class_identified_set_diagnostic(mixture, "pag"));
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
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
                diagnostics: Some(diagnostics),
                structural_response,
                ..Default::default()
            },
        }))
        .map(|result| self.attach_certificate(result, certificate))
    }

    /// CPDAG ATE via MEC-completion envelope + mass-weighted estimates.
    pub(super) fn execute_cpdag(
        &self,
        data: &TabularData,
        cpdag: &antecedent_graph::Cpdag,
        query: &AverageEffectQuery,
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
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(DEFAULT_PAG_ESTIMATOR_ID.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        let (envelope, identification, identify_cached) =
            if let Some(cache) = self.cpdag_identification_cache.as_deref() {
                (cache.envelope.clone(), cache.identification.clone(), true)
            } else {
                report_identify_compute(ctx);
                let envelope = identify_cpdag(identifier_id, cpdag, query)?;
                let identification = envelope_to_identification_result(&envelope, query);
                (envelope, identification, false)
            };
        let certificate = crate::Identification::CpdagEnvelope {
            envelope: envelope.clone(),
            strategy: identifier_id,
            structure_version: self.graph.version(),
        };
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if matches!(self.inference, InferenceMode::Bayesian(_))
                || matches!(estimator_id, EstimatorId::BayesianGcomp)
            {
                return self
                    .execute_pag_nonidentified_prior(
                        query,
                        physical,
                        ctx,
                        &envelope,
                        identification,
                        identify_cached,
                        started,
                        cpdag_envelope_diagnostic(&envelope),
                        "cpdag",
                    )
                    .map(|result| self.attach_certificate(result, certificate));
            }
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "CPDAG effect not identified (no identified mass in envelope)",
            ));
        }

        let mut diagnostics = vec![cpdag_envelope_diagnostic(&envelope)];

        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self
                .execute_pag_bayesian(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identification,
                    identify_cached,
                    started,
                    cpdag_envelope_diagnostic(&envelope),
                    "cpdag",
                )
                .map(|result| self.attach_certificate(result, certificate));
        }

        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut atom_ifs = Vec::new();
        let mut atom_weights = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let mut outcomes = vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()];
        for (i, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, estimator_id)?;
            let mut case_ws = StaticEstimateWorkspaces::default();
            let case_spec = self
                .estimator_spec
                .clone()
                .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id));
            let estimate = estimate_static_effect(
                &case_spec,
                data,
                &estimand,
                query,
                case.result.required_assumptions.clone(),
                self.bootstrap_replicates,
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut case_ws,
            )?;
            let w = case.weight.0;
            outcomes[i] = ClassAtomOutcome::Evaluated(estimate.ate);
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, &estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: None,
                original: estimate,
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "CPDAG envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "CPDAG envelope missing estimand".into(),
        })?;
        let n_contributing = se_items.len();
        let se = if atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        };
        let mut estimate = EffectEstimate::new(
            weighted_ate / total_w,
            se,
            assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
                estimate.influence = Some(inf);
            }
        }

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            data,
            query,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
            self.split.as_ref(),
            None,
        )?;
        diagnostics.extend(na_diagnostics);

        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic));
        // A partially identified envelope carries its identified set and the
        // per-completion values it spans; a point-identified one stays a point.
        let structural_response = static_class_structural_mixture(&envelope, &outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(static_class_identified_set_diagnostic(mixture, "cpdag"));
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
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
                diagnostics: Some(diagnostics),
                structural_response,
                ..Default::default()
            },
        }))
        .map(|result| self.attach_certificate(result, certificate))
    }

    /// Non-identified PAG with Bayesian inference: prior-predictive draws, no invented ID.
    pub(super) fn execute_pag_nonidentified_prior<G>(
        &self,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<G>,
        identification: IdentificationResult,
        identify_cached: bool,
        started: Instant,
        envelope_diagnostic: Diagnostic,
        class_tag: &str,
    ) -> Result<StudyResult, CausalError> {
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let scale = cfg.prior_scale.max(1e-6);
        let mut prior = PriorSet::weakly_informative(1);
        if let Some(g) = prior.specs.iter_mut().find_map(|s| match s {
            antecedent_prob::PriorSpec::GaussianCoefficients(p) => Some(p),
            _ => None,
        }) {
            *g = antecedent_prob::GaussianCoefficientPrior::isotropic(1, scale);
        }
        let posterior = nonidentified_with_prior(
            &prior,
            InferenceDiagnostics::analytic(format!("{class_tag}_nonidentified_prior")),
            cfg.n_draws.max(1),
            ctx.rng.master_seed(),
        );
        let estimate = effect_from_posterior(&posterior)?;
        let estimand = envelope.invariant.clone().unwrap_or_else(|| {
            IdentifiedEstimand::backdoor(
                format!("{class_tag}.nonidentified"),
                Arc::from([]),
                antecedent_expr::ExprId::from_raw(0),
            )
        });
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(envelope_diagnostic);
        diagnostics.push(
            Diagnostic::new(
                format!("estimate.{class_tag}.nonidentified_prior"),
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "{class_tag} not identified; returning prior-predictive draws (unidentified_mass={})",
                    posterior.unidentified_mass
                ),
            )
            .with_fields(super::mass_fields(
                None,
                posterior.unidentified_mass,
            )),
        );
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian_gcomp",
                    "estimate.nonidentified_with_prior",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }
}

/// Structural uncertainty of a static class (CPDAG/PAG) envelope, keyed by
/// case index.
///
/// The mixture itself is [`super::class_structural_mixture`],
/// shared with the temporal class routes. This arm adds only the static
/// class's publishing rule: nothing to publish when the envelope carried no
/// mass or no completion produced a finite value, and `None` for a
/// point-identified envelope — one identified completion, or several that
/// agree, with no unidentified, unevaluable or subsampled-out mass. Such a
/// result is a point, not an identified set, and attaching a mixture would
/// restate it as bounds.
///
/// Static class envelopes publish the frozen-weight mixture's sampling
/// interval (the construction `v19_static_envelope_calibration` scores); no
/// interval for the identified set is constructed on this path.
pub(super) fn static_class_structural_mixture<G>(
    envelope: &IdentificationEnvelope<G>,
    outcomes: &[ClassAtomOutcome],
) -> Option<crate::result::StructuralResponseMixture> {
    let (mixture, facts) = super::class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        |index, _| u64::try_from(index).unwrap_or(u64::MAX),
        None,
        outcomes,
    );
    if facts.total_weight <= 0.0 || mixture.identified_set.is_none() || facts.point_identified {
        return None;
    }
    Some(mixture)
}

/// The identified-set diagnostic a partially identified static class effect
/// reports next to its mixture summary.
pub(super) fn static_class_identified_set_diagnostic(
    mixture: &crate::result::StructuralResponseMixture,
    class_tag: &str,
) -> Option<Diagnostic> {
    let set = mixture.identified_set.as_ref()?;
    Some(
        Diagnostic::new(
            format!("estimate.{class_tag}.identified_set"),
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "identified set over identified completions: [{}, {}]; identified_mass={}; \
             unidentified_mass={}; unevaluable_mass={}; incomplete_search_mass={}; the \
             reported scalar is the frozen-weight mixture over identified completions, \
             and completion weights are enumeration weights, not posterior probabilities",
                set.lower[0],
                set.upper[0],
                mixture.identified_mass,
                mixture.unidentified_mass,
                mixture.unevaluable_mass,
                mixture.subsampled_out_mass,
            ),
        )
        .with_fields(super::mass_fields(Some(mixture.identified_mass), mixture.unidentified_mass)),
    )
}

/// Completion-mass summary every PAG arm (Frequentist, Bayesian, non-identified
/// prior) reports, so the envelope a fixture pins is observable on each.
pub(super) fn pag_envelope_diagnostic<G>(envelope: &IdentificationEnvelope<G>) -> Diagnostic {
    super::class_envelope_diagnostic(
        "identify.pag.envelope",
        "generalized.adjustment envelope",
        envelope,
        &[],
    )
}

/// Completion-mass summary for the CPDAG MEC envelope.
pub(super) fn cpdag_envelope_diagnostic<G>(envelope: &IdentificationEnvelope<G>) -> Diagnostic {
    super::class_envelope_diagnostic("identify.cpdag.envelope", "cpdag.mec envelope", envelope, &[])
}

fn admg_response_at_level(
    query: &ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    level: f64,
) -> ResponseQuery {
    let mut level_query = query.clone();
    level_query.functional = ResponseFunctional::InterventionResponse {
        outcome,
        interventions: Arc::from([Intervention::set(
            treatment,
            antecedent_core::Value::f64(level),
        )]),
    };
    level_query
}

fn admg_response_single_level(query: &ResponseQuery) -> Result<ResponseQuery, CausalError> {
    match &query.functional {
        ResponseFunctional::InterventionResponse { .. } => Ok(query.clone()),
        _ => Err(CausalError::Compile {
            message: "ADMG intervention response requires InterventionResponse".into(),
        }),
    }
}

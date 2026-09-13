// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn execute_temporal_cpdag_mediation(
        &self,
        data: &TimeSeriesData,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let estimator = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let mut slices = Vec::with_capacity(query.horizons.len());
        let mut primary = None;
        let mut primary_envelope = None;
        let mut refutations = Vec::new();
        let mut aggregate_graph_dependent = false;
        let mut structural_atoms = Vec::new();
        let mut full_mass_scope = true;
        let mut truncated_atoms = 0;
        let mut class_weights = TemporalClassWeights::new(self.class_prior.as_ref());
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal_mediation.class_identified_set",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "completion-specific mediation effects are returned as per-horizon identified \
             sets; completion enumeration is not averaged",
        )];
        for (horizon_index, &horizon) in query.horizons.iter().enumerate() {
            let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
            witness.horizon_steps = horizon;
            let bundle =
                self.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &witness)?;
            let envelope = &bundle.envelope.envelope;
            let hit_cap = envelope.truncated_completions > 0;
            full_mass_scope &= !hit_cap;
            truncated_atoms += envelope.truncated_completions;
            aggregate_graph_dependent |= hit_cap;
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            let completion_count = envelope.cases.len();
            let mut effects = Vec::new();
            let mut local_diagnostics = Vec::new();
            for (completion_idx, (case, indexer)) in
                envelope.cases.iter().zip(&bundle.envelope.indexers).enumerate()
            {
                let identification = &case.result;
                structural_atoms.push(crate::result::StructuralResponseAtom {
                    graph_key: ((horizon_index as u64) << 32) | completion_idx as u64,
                    weight: weights[completion_idx],
                    status: identification.status,
                    value: None,
                    posterior: None,
                    response: None,
                });
                if !identification_status_ok_for_case(identification.status) {
                    aggregate_graph_dependent = true;
                    continue;
                }
                let estimand = select_estimand(identification, EstimatorId::TemporalMediation)?;
                let mut horizon_query = query.clone();
                horizon_query.horizons = Arc::from([horizon]);
                let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
                let adjustment: Arc<[antecedent_data::LaggedColumn]> = estimand
                    .adjustment_set
                    .iter()
                    .filter_map(|dense| {
                        let key = indexer.key_of(dense.raw()).ok()?;
                        lagged_column_relative_to_outcome(key, outcome_offset)
                    })
                    .collect::<Vec<_>>()
                    .into();
                let frequentist_estimate = match &self.inference {
                    InferenceMode::Bayesian(cfg) => {
                        let bayes = bayesian_gcomp(cfg, ctx);
                        let preparations =
                            antecedent_estimate::bayesian_mediation::prepare_temporal_mediation_adjusted(
                                data,
                                &estimand,
                                &horizon_query,
                                &adjustment,
                                ctx,
                            )
                            .map_err(CausalError::from)?;
                        let mechanisms = preparations
                            .iter()
                            .enumerate()
                            .map(|(index, prep)| {
                                let mut mechanism = bayes.clone();
                                mechanism.seed = mechanism.seed.wrapping_add(if index == 0 {
                                    0
                                } else {
                                    0xBA71
                                });
                                let (prior, conflict) =
                                    resolve_envelope_prior_anchor(cfg, prep, ctx)?;
                                if let Some(summary) = conflict.as_ref() {
                                    push_conflict_diagnostics(&mut diagnostics, summary);
                                }
                                mechanism.prior = prior;
                                mechanism
                                    .fit(
                                        prep,
                                        identification.status,
                                        &mut BayesianGCompWorkspace::default(),
                                        ctx,
                                    )
                                    .map_err(CausalError::from)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let composed =
                            antecedent_estimate::bayesian_mediation::compose_temporal_mediation(
                                &mechanisms[0],
                                &mechanisms[1],
                                &horizon_query,
                                identification.status,
                            )
                            .map_err(CausalError::from)?;
                        let mean = composed.summaries.mean.first().copied().unwrap_or(f64::NAN);
                        effects.push(mean);
                        structural_atoms.last_mut().expect("mediation atom").posterior =
                            Some(composed.clone());
                        Some(antecedent_estimate::TemporalMediationEstimate {
                            effect: effect_from_posterior(&composed)?,
                            total: None,
                            direct: None,
                            mediated: None,
                        })
                    }
                    InferenceMode::Frequentist => {
                        let estimate = estimator
                            .estimate_with_adjustment(
                                data,
                                &estimand,
                                &horizon_query,
                                &adjustment,
                                &[],
                                ctx,
                            )
                            .map_err(CausalError::from)?;
                        effects.push(estimate.effect.ate);
                        Some(estimate)
                    }
                };
                if let Some(estimate) = frequentist_estimate.as_ref() {
                    if self.refute != RefuteSuite::None {
                        let mut reports =
                            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                                data,
                                &estimand,
                                &horizon_query,
                                estimate,
                                self.refute == RefuteSuite::Full,
                                &adjustment,
                                ctx,
                            )
                            .map_err(CausalError::from)?;
                        for report in &mut reports {
                            report.refuter = Arc::from(format!(
                                "horizon.{horizon}.completion.{completion_idx}.{}",
                                report.refuter
                            ));
                        }
                        refutations.extend(reports);
                    }
                }
                structural_atoms.last_mut().expect("mediation atom").value =
                    effects.last().copied().map(ResponseValue::Scalar);
                if primary.is_none() {
                    primary = Some((identification.clone(), estimand));
                    primary_envelope = Some(bundle.envelope.clone());
                }
            }
            if effects.is_empty() {
                return Err(CausalError::Compile {
                    message: format!(
                        "TemporalCpdag mediation has no identified completion at horizon {horizon}"
                    ),
                });
            }
            let lower = effects.iter().copied().fold(f64::INFINITY, f64::min);
            let upper = effects.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            local_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.cpdag_envelope",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "horizon={horizon}, identified_completions={}, examined_completions={}, \
                     capped={hit_cap}",
                    effects.len(),
                    completion_count
                ),
            ));
            slices.push(antecedent_estimate::TemporalMediationSlice {
                horizon,
                identification_status: if effects.len() == completion_count && !hit_cap {
                    IdentificationStatus::PartiallyIdentified
                } else {
                    IdentificationStatus::GraphDependent
                },
                method: Arc::from("temporal_mediation.cpdag_completion_envelope"),
                adjustment: Arc::from([]),
                estimate: TemporalMediationEstimate {
                    effect: nan_effect(),
                    total: None,
                    direct: None,
                    mediated: None,
                },
                uncertainty: antecedent_estimate::TemporalMediationUncertainty::Unavailable,
                identified_set: Some(antecedent_estimate::TemporalMediationIdentifiedSet {
                    lower,
                    upper,
                }),
                diagnostics: local_diagnostics,
            });
        }
        let (mut identification, estimand) = primary.ok_or_else(|| CausalError::Compile {
            message: "TemporalCpdag mediation missing primary completion".into(),
        })?;
        identification.status = if aggregate_graph_dependent {
            IdentificationStatus::GraphDependent
        } else {
            IdentificationStatus::PartiallyIdentified
        };
        let grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from(slices),
            joint_posterior: false,
        };
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: query.horizons.iter().map(|h| f64::from(*h)).collect::<Vec<_>>().into(),
            dimension: 1,
            lower: grid
                .slices
                .iter()
                .map(|slice| slice.identified_set.as_ref().expect("set").lower)
                .collect::<Vec<_>>()
                .into(),
            upper: grid
                .slices
                .iter()
                .map(|slice| slice.identified_set.as_ref().expect("set").upper)
                .collect::<Vec<_>>()
                .into(),
        };
        let conditional = self.class_prior.as_ref().and_then(|_| {
            temporal_class_response_mean(
                &structural_atoms,
                &response_envelope,
                query.horizons.len(),
                full_mass_scope,
            )
        });
        let identified_mass = structural_atoms
            .iter()
            .filter(|atom| atom.value.is_some())
            .map(|atom| atom.weight)
            .sum::<f64>()
            / query.horizons.len() as f64;
        let structural = crate::result::StructuralResponseMixture {
            weight_basis: if self.class_prior.is_some() {
                crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
            } else {
                crate::result::StructuralWeightBasis::CompletionEnumeration
            },
            atoms: structural_atoms,
            identified_mass,
            unidentified_mass: (1.0 - identified_mass).max(0.0),
            unevaluable_mass: 0.0,
            identified_set: Some(response_envelope),
            conditional_on_identified: conditional,
            full_mass_scope,
            truncated_atoms,
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate: nan_effect(),
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id: if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::BayesianTemporalMediation
            } else {
                EstimatorId::TemporalMediation
            },
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: self.temporal_class_cache_covers(&query.horizons),
            extra_diagnostics: diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: primary_envelope.map(|envelope| {
                    crate::Identification::TemporalEnvelope {
                        envelope,
                        strategy: IdentifierId::GeneralizedAdjustment,
                        structure_version: self.graph.version(),
                    }
                }),
                mediation_grid: Some(grid),
                structural_response: Some(structural),
                ..Default::default()
            },
        }))
    }

    pub(crate) fn mediation_adjustment(
        &self,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        horizon: u32,
    ) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError> {
        if let Some(entry) =
            self.temporal_identification_cache.as_deref().and_then(|cache| cache.get(horizon))
        {
            return Ok(lagged_adjustment_from_entry(entry));
        }
        let ider = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        };
        let (_, temporal) =
            ider.identify_with_horizon(graph, query, horizon).map_err(CausalError::from)?;
        Ok(lagged_adjustment_from_temporal(&temporal))
    }

    pub(super) fn execute_temporal(
        &self,
        data: &TimeSeriesData,
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
                message: format!(
                    "prepared temporal identification missing horizon {}",
                    query.horizon_steps
                ),
            })?;
            (entry.identification.clone(), entry.estimand.clone(), entry.indexer.clone(), true)
        } else {
            let id_res = TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, query)
                .map_err(CausalError::from)?;
            let estimand = select_estimand(
                &id_res.result,
                if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
                {
                    EstimatorId::TemporalSequentialGcomp
                } else {
                    EstimatorId::TemporalLinearAdjustment
                },
            )?;
            (id_res.result, estimand, id_res.indexer, false)
        };
        require_identified(&identification)?;
        if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
        {
            if self.split.is_some() {
                return Err(CausalError::Unsupported {
                    message: "multi-step sustained g-computation currently requires no discovery-estimation split",
                });
            }
            let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
                if cfg.prior.is_some()
                    || cfg.prior_artifact.is_some()
                    || cfg.external_compose.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "multi-step sustained inference requires isotropic per-mechanism priors",
                    });
                }
                Some(bayesian_gcomp(cfg, ctx))
            } else {
                None
            };
            let mut assumptions = identification.required_assumptions.clone();
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.sequential.linear_sem"), description: Arc::from("linear additive mechanisms on the identified unfolded DAG; every sustained time is intervened on; frequentist intervals use a shared moving-block row bootstrap, Bayesian intervals share each stationary mechanism posterior across time copies, fitting its unique complete observed rows once"),
                }),
                source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("temporal.sequential.gcomp") },
                scope: antecedent_core::AssumptionScope::Estimation, status: antecedent_core::AssumptionStatus::Declared,
            });
            let mut mechanisms = Vec::new();
            let (estimate, mut posterior) =
                antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                    data,
                    graph,
                    &indexer,
                    &estimand,
                    query,
                    identification.status,
                    assumptions,
                    self.bootstrap_replicates,
                    bayes.as_ref(),
                    ctx,
                    Some(&mut mechanisms),
                )
                .map_err(CausalError::from)?;
            let atom = super::sequential_validation::SequentialValidationAtom {
                weight: 1.0,
                graph: graph.clone(),
                indexer: indexer.clone(),
                estimand: estimand.clone(),
                status: identification.status,
                estimate: estimate.clone(),
                mechanisms,
            };
            let (refutations, diagnostics, predictive_checks) =
                super::sequential_validation::validate_sequential(
                    data,
                    query,
                    &[atom],
                    self.refute,
                    &self.custom_validators,
                    bayes.as_ref(),
                    posterior.as_mut(),
                    estimate.ate,
                    ctx,
                )?;
            let bootstrap_replicates_ok = estimate.bootstrap_replicates_ok;
            let cancelled = estimate.bootstrap_cancelled;
            return Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
                physical, identification, estimand, estimate,
                identifier_id: IdentifierId::TemporalBackdoorUnfolded, estimator_id: EstimatorId::TemporalSequentialGcomp,
                treatment: query.treatment, outcome: query.outcome, identify_cached,
                extra_diagnostics: vec![Diagnostic::new("estimate.temporal.sustained_window", DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "the contrast propagates through all intervened times in topological order; no one-node collapse; no analytic SE is asserted for the frequentist sequential fit")],
                refutations, distribution: None, mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok, cancelled, early_stopped: false,
                extras: IdentifiedExecuteExtras { posterior, diagnostics: Some(diagnostics), predictive_checks, ..Default::default() },
            }));
        }

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prep = estimator
            .prepare(data, &estimand, query, &indexer, self.split.as_ref(), &ctx.kernel_policy)
            .map_err(CausalError::from)?;

        let (estimate, posterior, estimate_artifact, estimate_op) = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
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
                    "estimate.bayesian_temporal_gcomp",
                    "estimate.bayesian.temporal.gcomp",
                )
            }
            InferenceMode::Frequentist => {
                let mut workspace = EstimationWorkspace::default();
                let estimate = estimator
                    .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
                    .map_err(CausalError::from)?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment",
                    "estimate.temporal.linear.adjustment",
                )
            }
        };

        let mut diagnostics = Vec::new();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        if physical
            .logical
            .record
            .discovery_algorithm
            .as_deref()
            .is_some_and(|a| a.contains("pag_completed_to_dag") || a.contains("completed_to_dag"))
        {
            diagnostics.push(Diagnostic::new(
                "temporal.pag.completed_to_dag",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "TemporalPag completed to TemporalDag before temporal.backdoor \
                 (completion path; not class-aware temporal PAG identification)",
            ));
        }
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let temporal_ctx = TemporalRefitContext {
            indexer: &indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: Some(data.time_index()),
            panel: None,
        };
        let (mut refutations, na_diagnostics) = run_refuters(
            &tabular,
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

        // Bayesian temporal: prior/posterior PPC + prior sensitivity on Full (mirror static).
        let mut posterior = posterior;
        let mut predictive_checks = Vec::new();
        if matches!(&self.inference, InferenceMode::Bayesian(_))
            && !matches!(self.refute, RefuteSuite::None)
        {
            if let Some(ref post) = posterior {
                const PPC_ALPHA: f64 = 0.05;
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let prior_rep = PriorPredictiveCheck {
                    n_sims: 200,
                    seed: ctx.rng.master_seed(),
                    ..PriorPredictiveCheck::new()
                }
                .check(&bprep, ctx)
                .map_err(CausalError::from)?;
                refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
                predictive_checks.push(prior_rep);

                let post_rep = PosteriorPredictiveCheck::new()
                    .check(&bprep, post)
                    .map_err(CausalError::from)?;
                refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
                predictive_checks.push(post_rep);

                if matches!(self.refute, RefuteSuite::Full) {
                    let InferenceMode::Bayesian(cfg) = &self.inference else { unreachable!() };
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
        }

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let certificate = crate::Identification::Point {
            result: identification.clone(),
            temporal_indexer: Some(indexer.clone()),
            strategy: IdentifierId::TemporalBackdoorUnfolded,
            structure_version: self.graph.version(),
        };
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
                certificate: Some(certificate),
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(estimate_artifact, estimate_op)),
                posterior,
                diagnostics: Some(diagnostics),
                predictive_checks,
                ..Default::default()
            },
        }))
    }

    /// Panel temporal effect: identify on the shared graph, estimate on stacked units
    /// with [`AnalyticSeKind::PanelClusterHac`] and per-unit `cluster_ids`.
    ///
    /// Bayesian mode fits [`BayesianTemporalGcomp`] on the stacked lag-aligned design
    /// (no hierarchical unit random effects; cluster-HAC is frequentist-only).
    pub(super) fn execute_temporal_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            return self.execute_bayesian_mediation(data, graph, query, physical, cfg, ctx);
        }
        let started = Instant::now();
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                report_identify_compute(ctx);
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_mediation_horizons(
                        graph,
                        query,
                        EstimatorId::TemporalMediation,
                    )?),
                    false,
                )
            };
        let clicks = mediation_horizon_clicks(&cache, query, |horizon| {
            self.mediation_adjustment(graph, query, horizon)
        })?;
        let mut extra_diagnostics = Vec::new();
        if mediation_horizon_z_differs(&clicks) {
            extra_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each contrast uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        let est = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let single_horizon = clicks.len() == 1;
        let mut slices = Vec::with_capacity(clicks.len());
        let mut refutations = Vec::new();
        for click in &clicks {
            require_identified(&click.identification)?;
            let mut qh = query.clone();
            qh.horizons = Arc::from([click.horizon]);
            let mediation = est
                .estimate_with_adjustment(data, &click.estimand, &qh, &click.adjustment, &[], ctx)
                .map_err(CausalError::from)?;
            if self.refute != RefuteSuite::None {
                let mut reports =
                    antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                        data,
                        &click.estimand,
                        &qh,
                        &mediation,
                        self.refute == RefuteSuite::Full,
                        &click.adjustment,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                for report in &mut reports {
                    if !single_horizon {
                        report.refuter =
                            Arc::from(format!("horizon.{}.{}", click.horizon, report.refuter));
                    }
                }
                refutations.extend(reports);
            }
            let standard_error =
                mediation.effect.se_analytic.is_finite().then_some(mediation.effect.se_analytic);
            slices.push(antecedent_estimate::TemporalMediationSlice {
                horizon: click.horizon,
                identification_status: click.identification.status,
                method: Arc::clone(&click.estimand.method),
                adjustment: mediation_click_adjustment_keys(click),
                estimate: mediation,
                uncertainty:
                    antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
                        standard_error,
                    },
                identified_set: None,
                diagnostics: click.identification.diagnostics.clone(),
            });
        }
        let first = slices.first().ok_or_else(|| CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        })?;
        let estimate = if single_horizon {
            first.estimate.effect.clone()
        } else {
            extra_diagnostics.push(Diagnostic::new(
                "estimate.temporal_mediation.multi_horizon",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "all requested horizons are retained in mediation_grid; the scalar estimate and \
                 scalar mediation fields are intentionally absent because no horizon is \
                 authoritative",
            ));
            nan_effect()
        };
        let mediation = single_horizon.then(|| first.estimate.clone());
        let click = &clicks[0];
        let mut identification = click.identification.clone();
        if !single_horizon {
            identification.status = most_conservative_identification_status(
                slices.iter().map(|slice| slice.identification_status),
            );
            extra_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.multi_horizon_status",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "parent identification.status is the most conservative requested-horizon \
                 status; mediation_grid is authoritative per horizon",
            ));
        }
        let mediation_grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from(slices),
            joint_posterior: false,
        };
        let certificate = single_horizon.then(|| crate::Identification::Point {
            result: click.identification.clone(),
            temporal_indexer: Some(click.indexer.clone()),
            strategy: IdentifierId::Frontdoor,
            structure_version: self.graph.version(),
        });
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: click.estimand.clone(),
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::TemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate,
                mediation_grid: Some(mediation_grid),
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_mediation",
                    "identify.temporal_mediation",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_mediation",
                    "estimate.temporal_mediation",
                )),
                ..Default::default()
            },
        }))
    }

    fn execute_bayesian_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        cfg: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use antecedent_estimate::bayesian_mediation::{
            compose_temporal_mediation, prepare_temporal_mediation_adjusted,
            require_gaussian_mediation,
        };
        let started = Instant::now();
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            return Err(CausalError::Unsupported {
                message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
            });
        }
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                report_identify_compute(ctx);
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_mediation_horizons(
                        graph,
                        query,
                        EstimatorId::BayesianTemporalMediation,
                    )?),
                    false,
                )
            };
        let clicks = mediation_horizon_clicks(&cache, query, |horizon| {
            self.mediation_adjustment(graph, query, horizon)
        })?;
        if clicks.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal mediation requires at least one horizon".into(),
            });
        }
        let mut extra_diagnostics = Vec::new();
        if mediation_horizon_z_differs(&clicks) {
            extra_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each contrast uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        let estimator = bayesian_gcomp(cfg, ctx);
        require_gaussian_mediation(&estimator).map_err(CausalError::from)?;
        let single_horizon = clicks.len() == 1;
        let mut slices = Vec::with_capacity(clicks.len());
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        let mut first_posterior = None;
        for click in &clicks {
            require_identified(&click.identification)?;
            let mut qh = query.clone();
            qh.horizons = Arc::from([click.horizon]);
            let preparations = prepare_temporal_mediation_adjusted(
                data,
                &click.estimand,
                &qh,
                &click.adjustment,
                ctx,
            )
            .map_err(CausalError::from)?;
            let fit = |scale: f64| -> Result<Vec<CausalPosterior>, CausalError> {
                preparations
                    .iter()
                    .enumerate()
                    .map(|(i, prep)| {
                        let mut est = estimator.clone();
                        est.prior_scale = scale;
                        est.seed = est.seed.wrapping_add(if i == 0 { 0 } else { 0xBA71_u64 });
                        est.fit(
                            prep,
                            click.identification.status,
                            &mut BayesianGCompWorkspace::default(),
                            ctx,
                        )
                        .map_err(CausalError::from)
                    })
                    .collect()
            };
            let mechanisms = fit(cfg.prior_scale)?;
            let mut posterior = compose_temporal_mediation(
                &mechanisms[0],
                &mechanisms[1],
                &qh,
                click.identification.status,
            )
            .map_err(CausalError::from)?;
            let horizon_estimate = effect_from_posterior(&posterior)?;
            let mediation = TemporalMediationEstimate {
                effect: horizon_estimate.clone(),
                total: Some(posterior.summaries.mean[1]),
                direct: Some(posterior.summaries.mean[2]),
                mediated: Some(posterior.summaries.mean[3]),
            };
            if self.refute != RefuteSuite::None {
                let mut reports =
                    antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                        data,
                        &click.estimand,
                        &qh,
                        &mediation,
                        self.refute == RefuteSuite::Full,
                        &click.adjustment,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                for report in &mut reports {
                    if !single_horizon {
                        report.refuter =
                            Arc::from(format!("horizon.{}.{}", click.horizon, report.refuter));
                    }
                }
                refutations.extend(reports);

                for (i, (prep, post)) in preparations.iter().zip(&mechanisms).enumerate() {
                    let prior = estimator.prior.clone().unwrap_or_else(|| {
                        let mut prior = PriorSet::weakly_informative(prep.design.ncols);
                        prior.specs = vec![antecedent_prob::PriorSpec::GaussianCoefficients(
                            antecedent_prob::GaussianCoefficientPrior::isotropic(
                                prep.design.ncols,
                                cfg.prior_scale,
                            ),
                        )];
                        prior
                    });
                    let prior_rep = PriorPredictiveCheck::new()
                        .check_with_prior(prep, &prior, ctx)
                        .map_err(CausalError::from)?;
                    let post_rep = PosteriorPredictiveCheck::new()
                        .check(prep, post)
                        .map_err(CausalError::from)?;
                    for report in [prior_rep, post_rep] {
                        let mut r = report.to_refutation_report(horizon_estimate.ate, 0.05);
                        r.refuter = if single_horizon {
                            Arc::from(format!("mediation.mechanism_{i}.{}", r.refuter))
                        } else {
                            Arc::from(format!(
                                "horizon.{}.mediation.mechanism_{i}.{}",
                                click.horizon, r.refuter
                            ))
                        };
                        refutations.push(r);
                        predictive_checks.push(report);
                    }
                }
            }
            if self.refute == RefuteSuite::Full {
                let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
                let mut means = Vec::new();
                let mut sds = Vec::new();
                for &scale in sensitivity.scales.iter() {
                    let posts = fit(scale)?;
                    let post = compose_temporal_mediation(
                        &posts[0],
                        &posts[1],
                        &qh,
                        click.identification.status,
                    )
                    .map_err(CausalError::from)?;
                    means.push(post.summaries.mean[0]);
                    sds.push(post.summaries.sd[0]);
                }
                let summary = antecedent_prob::PriorSensitivitySummary {
                    prior_scales: sensitivity.scales.clone(),
                    alphas: Arc::from([]),
                    effect_means: Arc::from(means),
                    effect_sds: Arc::from(sds),
                };
                let mut report = sensitivity.to_report(&summary, horizon_estimate.ate);
                if !single_horizon {
                    report.refuter =
                        Arc::from(format!("horizon.{}.{}", click.horizon, report.refuter));
                }
                refutations.push(report);
                posterior = with_prior_sensitivity(posterior, summary);
            }
            let summary = |index: usize| antecedent_estimate::MediationPosteriorSummary {
                mean: posterior.summaries.mean[index],
                standard_deviation: posterior.summaries.sd[index],
                q025: posterior.summaries.q025[index],
                q975: posterior.summaries.q975[index],
            };
            slices.push(antecedent_estimate::TemporalMediationSlice {
                horizon: click.horizon,
                identification_status: click.identification.status,
                method: Arc::clone(&click.estimand.method),
                adjustment: mediation_click_adjustment_keys(click),
                estimate: mediation,
                uncertainty: antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                    requested: summary(0),
                    total: summary(1),
                    direct: summary(2),
                    mediated: summary(3),
                    n_draws: posterior.draws.n_draws,
                    backend: Arc::clone(&posterior.diagnostics.backend_id),
                },
                identified_set: None,
                diagnostics: click.identification.diagnostics.clone(),
            });
            if first_posterior.is_none() {
                first_posterior = Some(posterior);
            }
        }
        let first = slices.first().ok_or_else(|| CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        })?;
        let estimate = if single_horizon {
            first.estimate.effect.clone()
        } else {
            extra_diagnostics.push(Diagnostic::new(
                "estimate.temporal_mediation.multi_horizon",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "each horizon retains an independent pointwise posterior summary in \
                 mediation_grid; no scalar or joint cross-horizon posterior is asserted",
            ));
            nan_effect()
        };
        let mediation = single_horizon.then(|| first.estimate.clone());
        let posterior = single_horizon.then(|| first_posterior.clone()).flatten();
        let click = &clicks[0];
        let mut identification = click.identification.clone();
        if !single_horizon {
            identification.status = most_conservative_identification_status(
                slices.iter().map(|slice| slice.identification_status),
            );
            extra_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.multi_horizon_status",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "parent identification.status is the most conservative requested-horizon \
                 status; mediation_grid is authoritative per horizon",
            ));
        }
        let mediation_grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from(slices),
            joint_posterior: false,
        };
        let estimand = click.estimand.clone();
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical, identification: identification.clone(), estimand, estimate,
            identifier_id: IdentifierId::Frontdoor, estimator_id: EstimatorId::BayesianTemporalMediation,
            treatment: query.treatment, outcome: query.outcome, identify_cached,
            extra_diagnostics: {
                extra_diagnostics.push(Diagnostic::new("estimate.mediation.bayesian", DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; natural effects use the linear no-interaction alias"));
                extra_diagnostics
            },
            refutations, distribution: None, mediation,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None, cancelled: false, early_stopped: false,
            extras: IdentifiedExecuteExtras { posterior, predictive_checks,
                mediation_grid: Some(mediation_grid),
                certificate: single_horizon.then(|| crate::Identification::Point {
                    result: identification.clone(),
                    temporal_indexer: Some(click.indexer.clone()),
                    strategy: IdentifierId::Frontdoor,
                    structure_version: self.graph.version(),
                }),
                identify_provenance: Some(provenance_ids("identify.temporal_mediation", "identify.temporal_mediation")),
                ..Default::default() },
        }))
    }

    pub(super) fn execute_temporal_response(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal response route requires TemporalResponseSpec".into(),
            });
        };
        enforce_temporal_response_memory_budget(query, temporal, self.bootstrap_replicates, ctx)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let schedule = match antecedent_estimate::plan_from_response_query(query) {
            Ok(Some(plan)) if plan.mechanism_overlays().is_some() => {
                Some(plan.identification_schedule(temporal))
            }
            Ok(_) => None,
            Err(error) => return Err(CausalError::from(error)),
        };
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_response_horizons(
                        graph,
                        treatment,
                        outcome,
                        temporal,
                        &query.target_population,
                        if matches!(self.inference, InferenceMode::Bayesian(_)) {
                            EstimatorId::TemporalResponseBayesian
                        } else {
                            EstimatorId::TemporalResponseGcomp
                        },
                        schedule.as_deref(),
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
        let (aggregate_status, mut aggregate_assumptions) =
            aggregate_temporal_horizon_evidence(aligned.iter().map(|entry| &entry.identification))?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

        query
            .require_licensed_temporal_observation()
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut working_query = query.clone();
        let mut observation_adjusted = None;
        let observed_bayes = matches!(self.inference, InferenceMode::Bayesian(_))
            && query.observation != ObservationSpec::Complete;
        let series_owned = if query.observation == ObservationSpec::Complete {
            None
        } else if observed_bayes {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not licensed for temporal observation",
                });
            }
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, &mut aggregate_assumptions);
            None
        } else {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not licensed for temporal observation",
                });
            }
            let adjustment = contemporaneous_adjustment_variables(&aligned);
            let (series, adjusted) = ObservationMechanismEstimator::new(self.observation_options)
                .adjust_temporal_series(data, query, &adjustment)
                .map_err(CausalError::from)?;
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, &mut aggregate_assumptions);
            working_query.observation = ObservationSpec::Complete;
            working_query.observation_assumptions = Arc::from([]);
            observation_adjusted = Some(adjusted);
            Some(series)
        };
        let source_data = data;
        let data = series_owned.as_ref().unwrap_or(data);

        if let Some(overlays) = antecedent_estimate::plan_from_response_query(query)
            .map_err(CausalError::from)?
            .and_then(|plan| plan.mechanism_overlays())
            .filter(|_| !observed_bayes)
        {
            return self.execute_temporal_sequence_response(
                data,
                source_data,
                graph,
                query,
                temporal,
                &overlays,
                &aligned,
                identification,
                estimand,
                treatment,
                outcome,
                aggregate_status,
                aggregate_assumptions.clone(),
                identify_cached,
                physical,
                started,
                ctx,
                observation_adjusted.as_ref(),
            );
        }

        // Surface SEs follow the Study bootstrap / replicate contract so Pulse,
        // single-step Sustained, and the dose × horizon surface report comparable
        // uncertainty. Replicates = 0 keeps the analytic OLS linear-functional SE.
        let mut estimator = TemporalResponseEstimator::new();
        estimator.inner.bootstrap_replicates =
            if observation_adjusted.is_some() { 0 } else { self.bootstrap_replicates };
        let mut conflict_summary = None;
        let mut observed_posterior = None;
        let mut response = if observed_bayes {
            let InferenceMode::Bayesian(cfg) = &self.inference else {unreachable!()};
            if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
                return Err(CausalError::Unsupported {message: "observed temporal mechanisms need independent per-mechanism priors; a transferred response coefficient prior has no such mapping"});
            }
            let (response, posterior) = antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(
                data, graph, &identifications, query, aggregate_status,
                aggregate_assumptions.clone(), &bayesian_gcomp(cfg, ctx), ctx,
            ).map_err(CausalError::from)?;
            observed_posterior = Some(posterior);
            Ok(response)
        } else if let InferenceMode::Bayesian(cfg) = &self.inference {
            let mut bayes = bayesian_gcomp(cfg, ctx);
            let (resolved, conflict) = resolve_temporal_response_prior(
                cfg, data, temporal, &aligned, treatment, outcome, ctx,
            )?;
            bayes.prior = resolved;
            conflict_summary = conflict;
            estimator.estimate_bayesian(
                data,
                &identifications,
                &working_query,
                aggregate_status,
                aggregate_assumptions.clone(),
                &bayes,
                ctx,
            )
        } else {
            estimator.estimate(
                data,
                &identifications,
                &working_query,
                aggregate_status,
                aggregate_assumptions.clone(),
                ctx,
            )
        }
        .map_err(CausalError::from)?;
        if let Some(adjusted) = observation_adjusted.as_ref() {
            apply_temporal_observation_result(&mut response, adjusted);
            if self.bootstrap_replicates > 0 {
                if let Some(bootstrap) = bootstrap_observation_adjusted_temporal_response(
                    source_data,
                    query,
                    &aligned,
                    aggregate_status,
                    &aggregate_assumptions,
                    self.observation_options,
                    self.bootstrap_replicates,
                    ctx,
                )? {
                    apply_observation_bootstrap(
                        &mut response,
                        &bootstrap,
                        self.bootstrap_replicates,
                    );
                }
            }
        }
        // The estimator's public API accepts one aggregate status for callers
        // that have a homogeneous set of horizon witnesses. This execution path
        // has the richer per-horizon records, so retain their actual statuses
        // instead of repeating the first/aggregate label across the surface.
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
        if horizon_adjustment_sets_differ(&aligned) {
            diagnostics.push(Diagnostic::new(
                "identify.temporal_response.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each cell uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        if let Some(summary) = conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, summary);
        }
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued; the scalar effect summary is not applicable \
                 and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

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
            identify_cached: false,
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
                posterior: observed_posterior,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    fn execute_temporal_sequence_response(
        &self,
        data: &TimeSeriesData,
        source_data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        overlays: &[antecedent_estimate::SequentialMechanismOverlay],
        aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        treatment: VariableId,
        outcome: VariableId,
        aggregate_status: IdentificationStatus,
        mut assumptions: antecedent_core::AssumptionSet,
        identify_cached: bool,
        physical: &PhysicalExecutionPlan,
        started: Instant,
        ctx: &ExecutionContext,
        observation_adjusted: Option<&antecedent_estimate::ObservationAdjustedOutcome>,
    ) -> Result<StudyResult, CausalError> {
        if self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step Sequence overlays require no discovery-estimation split",
            });
        }
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence inference requires isotropic per-mechanism priors",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.sequential.linear_sem"),
                    description: Arc::from(
                        "linear additive mechanisms on the identified unfolded DAG; each Sequence \
                         step transforms deterministic propagated means: fixed level, mean + shift, \
                         factor * mean, or clip(mean + shift, lower, upper); clipping is not \
                         integration over stochastic realizations; no last-step collapse",
                    ),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal.sequential.gcomp"),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        if overlays.iter().any(|overlay| overlay.bounds.is_some()) {
            assumptions
                .push(antecedent_estimate::SequentialMechanismOverlay::mean_target_assumption());
        }
        let mut mean = Vec::with_capacity(temporal.horizons.len());
        let mut lower = Vec::with_capacity(temporal.horizons.len());
        let mut upper = Vec::with_capacity(temporal.horizons.len());
        let mut horizons = Vec::with_capacity(temporal.horizons.len());
        let mut last_posterior = None;
        let attach_scalar_posterior = temporal.horizons.len() == 1;
        let mut uncertainty_complete = true;
        let mut bootstrap_cancelled = false;
        let z = 1.959_963_984_540_054;
        for (horizon_steps, entry) in temporal.horizons.iter().copied().zip(aligned.iter()) {
            require_identified(&entry.identification)?;
            let outcome_offset = i32::try_from(horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX);
            let (effect, posterior) = antecedent_estimate::estimate_sequence_mechanisms(
                data,
                graph,
                &entry.indexer,
                &entry.estimand,
                outcome,
                outcome_offset,
                overlays,
                entry.identification.status,
                assumptions.clone(),
                if observation_adjusted.is_some() { 0 } else { self.bootstrap_replicates },
                bayes.as_ref(),
                ctx,
            )
            .map_err(CausalError::from)?;
            bootstrap_cancelled |= effect.bootstrap_cancelled;
            let (point, lo, hi) = if let Some(post) = posterior.as_ref() {
                (post.summaries.mean[0], post.summaries.q025[0], post.summaries.q975[0])
            } else {
                let se = effect.se_bootstrap.unwrap_or(effect.se_analytic);
                (effect.ate, effect.ate - z * se, effect.ate + z * se)
            };
            if attach_scalar_posterior {
                last_posterior = posterior;
            }
            mean.push(point);
            uncertainty_complete &= lo.is_finite() && hi.is_finite();
            lower.push(lo);
            upper.push(hi);
            horizons.push(antecedent_core::HorizonIdentification {
                horizon: horizon_steps,
                status: entry.identification.status,
                method: Arc::clone(&entry.estimand.method),
                adjustment: Arc::from(named_adjustment_keys(entry)),
            });
        }
        let first_horizon = f64::from(temporal.horizons.first().copied().unwrap_or(1));
        let last_horizon = f64::from(temporal.horizons.last().copied().unwrap_or(1));
        let mut support = antecedent_core::SupportReport {
            // Joint longitudinal support is not established by univariate
            // ranges. Until a sequential-positivity diagnostic exists, do not
            // label an arbitrary overlay schedule empirically supported.
            status: antecedent_core::SupportStatus::Extrapolative,
            query_region: antecedent_core::SupportRegion {
                minima: Arc::from([first_horizon]),
                maxima: Arc::from([last_horizon]),
            },
            diagnostics: vec![antecedent_core::SupportDiagnostic {
                id: Arc::from("response.temporal.sequence_overlay"),
                values: Arc::from(
                    overlays
                        .iter()
                        .flat_map(|overlay| {
                            [
                                f64::from(overlay.node.variable.raw()),
                                f64::from(overlay.node.offset),
                                if overlay.node.level.is_some() { 1.0 } else { 0.0 },
                                overlay.node.level.unwrap_or(0.0),
                                overlay.node.shift,
                                overlay.multiplier,
                                if overlay.bounds.is_some() { 1.0 } else { 0.0 },
                                overlay.bounds.map_or(0.0, |bounds| bounds.0),
                                overlay.bounds.map_or(0.0, |bounds| bounds.1),
                            ]
                        })
                        .collect::<Vec<_>>(),
                ),
                detail: Arc::from(
                    "sequential Sequence overlays as \
                     [variable, offset, is_fixed_level, level_or_zero, shift, multiplier, \
                     has_bounds, lower_or_zero, upper_or_zero, …]; \
                     transforms apply to deterministic propagated means, not stochastic draws",
                ),
            }],
            warnings: vec![Diagnostic::new(
                "response.temporal.sequence_joint_support_unassessed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "joint longitudinal positivity for the requested Sequence overlay is not \
                 estimated; support is conservatively marked extrapolative",
            )],
            point_status: Some(Arc::from(vec![
                antecedent_core::SupportStatus::Extrapolative;
                temporal.horizons.len()
            ])),
        };
        if !uncertainty_complete {
            support.warnings.push(Diagnostic::new(
                "response.temporal.sequence_uncertainty_unavailable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "a complete pointwise Sequence band was unavailable; uncertainty is omitted \
                 rather than represented by non-finite endpoints",
            ));
        }
        let mut response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: aggregate_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(
                    temporal.horizons.iter().map(|&h| f64::from(h)).collect::<Vec<_>>(),
                ),
                dimension: 1,
                mean: Arc::from(mean.clone()),
            }),
            uncertainty: if uncertainty_complete {
                ResponseUncertainty::PointwiseBand {
                    level: 0.95,
                    lower: Arc::from(lower),
                    upper: Arc::from(upper),
                }
            } else {
                ResponseUncertainty::None
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        };
        if let Some(adjusted) = observation_adjusted {
            apply_temporal_observation_result(&mut response, adjusted);
            if self.bootstrap_replicates > 0 {
                if let Some(bootstrap) = bootstrap_observation_adjusted_sequence_response(
                    source_data,
                    graph,
                    query,
                    overlays,
                    aligned,
                    outcome,
                    &response.assumptions,
                    self.observation_options,
                    self.bootstrap_replicates,
                    ctx,
                )? {
                    apply_observation_bootstrap(
                        &mut response,
                        &bootstrap,
                        self.bootstrap_replicates,
                    );
                }
            }
        }
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.sequence_overlay",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "multi-step / joint Sequence runs as overlays on unfolded sequential g-computation; \
             identifier remains temporal.backdoor.unfolded; no last-step collapse",
        )];
        diagnostics.extend(response.support.warnings.iter().cloned());
        if !attach_scalar_posterior && matches!(self.inference, InferenceMode::Bayesian(_)) {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.sequence_posterior_not_attached",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the response retains per-horizon posterior summaries and pointwise intervals; \
                 no scalar posterior artifact is attached to a multi-horizon surface",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
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
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: bootstrap_cancelled || ctx.cancellation.is_cancelled(),
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
                posterior: last_posterior,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_temporal_class_response(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query
            .require_licensed_temporal_observation()
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if matches!(
            query.observation,
            ObservationSpec::IntervalCensored { .. } | ObservationSpec::Truncated { .. }
        ) || self.observation_delayed_entry.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "class-aware temporal response refuses delayed entry, interval, and \
                          truncation observation mechanisms",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal response requires TemporalResponseSpec".into(),
        })?;
        if matches!(
            antecedent_estimate::plan_from_response_query(query),
            Ok(Some(
                antecedent_estimate::TemporalInterventionPlan::Sequential { .. }
                    | antecedent_estimate::TemporalInterventionPlan::Mechanisms { .. }
            ))
        ) {
            return self.execute_temporal_class_sequence_response(data, query, physical, ctx);
        }
        enforce_temporal_response_memory_budget(query, temporal, self.bootstrap_replicates, ctx)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let cells_per_horizon = match &query.functional {
            ResponseFunctional::MeanCurve { treatment, .. } => treatment
                .grid
                .values()
                .map_err(|error| CausalError::Compile { message: error.to_string() })?
                .len(),
            ResponseFunctional::InterventionResponse { .. } => 1,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal response supports curves and intervention responses",
                });
            }
        };
        let n_horizons = temporal.horizons.len();
        let mut lower = vec![f64::NAN; cells_per_horizon * n_horizons];
        let mut upper = vec![f64::NAN; cells_per_horizon * n_horizons];
        let mut structural_atoms = Vec::new();
        let mut supports = Vec::new();
        let mut horizon_supports = Vec::new();
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut primary_envelope = None;
        let mut assumptions = antecedent_core::AssumptionSet::new();
        let mut full_mass_scope = true;
        let mut truncated_atoms = 0usize;
        let mut diagnostics = Vec::new();
        let mut class_weights = TemporalClassWeights::new(self.class_prior.as_ref());
        let mut horizon_fingerprints: Vec<Vec<u64>> = Vec::new();
        let mut observation_atoms = Vec::new();
        let mut class_observation_band_applied = false;
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let effect_query = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
                active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
                horizon_steps: horizon,
                max_history_lag: temporal.max_history_lag,
                target_population: query.target_population.clone(),
            };
            let bundle =
                self.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &effect_query)?;
            let envelope = &bundle.envelope.envelope;
            diagnostics.push(temporal_class_envelope_diagnostic(envelope, self.graph.class()));
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            horizon_fingerprints
                .push(envelope.cases.iter().map(|case| case.graph.fingerprint()).collect());
            full_mass_scope &= envelope.truncated_completions == 0;
            truncated_atoms += envelope.truncated_completions;
            let mut horizon_values = Vec::new();
            let support_start = supports.len();
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
            {
                if !identification_status_ok_for_case(case.result.status)
                    || case.result.estimands.is_empty()
                {
                    structural_atoms.push(crate::result::StructuralResponseAtom {
                        posterior: None,
                        response: None,
                        graph_key: ((horizon_index as u64) << 32) | case_index as u64,
                        weight: weights[case_index],
                        status: case.result.status,
                        value: None,
                    });
                    continue;
                }
                let estimand =
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
                let mut qh = query.clone();
                let mut temporal_h = temporal.clone();
                temporal_h.horizons = Arc::from([horizon]);
                qh.temporal = Some(temporal_h);
                let observation_query = qh.clone();
                let adjustment = class_atom_adjustment(&estimand, indexer);
                let series_owned = if qh.observation == ObservationSpec::Complete
                    || matches!(self.inference, InferenceMode::Bayesian(_))
                {
                    None
                } else {
                    if self.observation_delayed_entry.is_some() {
                        return Err(CausalError::Unsupported {
                            message: "delayed entry is not licensed for temporal observation",
                        });
                    }
                    let (series, _) = ObservationMechanismEstimator::new(self.observation_options)
                        .adjust_temporal_series(data, &qh, &adjustment)
                        .map_err(CausalError::from)?;
                    qh.observation = ObservationSpec::Complete;
                    qh.observation_assumptions = Arc::from([]);
                    Some(series)
                };
                let series = series_owned.as_ref().unwrap_or(data);
                let mut response = match &self.inference {
                    InferenceMode::Bayesian(cfg)
                        if query.observation != ObservationSpec::Complete =>
                    {
                        let Some(dag) = case.graph.sequential_dag() else {
                            structural_atoms.push(crate::result::StructuralResponseAtom {
                                posterior: None,
                                response: None,
                                graph_key: ((horizon_index as u64) << 32) | case_index as u64,
                                weight: weights[case_index],
                                status: case.result.status,
                                value: None,
                            });
                            continue;
                        };
                        if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
                            return Err(CausalError::Unsupported {
                                message: "prior_artifact stays refused on observed temporal Bayes",
                            });
                        }
                        let bayes = bayesian_gcomp(cfg, ctx);
                        let (response, _) =
                            antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(
                                data,
                                &dag,
                                &[(&estimand, indexer)],
                                &qh,
                                case.result.status,
                                case.result.required_assumptions.clone(),
                                &bayes,
                                ctx,
                            )
                            .map_err(CausalError::from)?;
                        response
                    }
                    InferenceMode::Bayesian(cfg) => {
                        let mut bayes = bayesian_gcomp(cfg, ctx);
                        let entry =
                            crate::analysis::prepared::CachedTemporalHorizonIdentification {
                                horizon,
                                identification: case.result.clone(),
                                estimand: estimand.clone(),
                                indexer: indexer.clone(),
                            };
                        let (prior, conflict) = resolve_temporal_response_prior(
                            cfg,
                            series,
                            qh.temporal.as_ref().expect("temporal horizon"),
                            &[&entry],
                            treatment,
                            outcome,
                            ctx,
                        )?;
                        if let Some(summary) = conflict.as_ref() {
                            push_conflict_diagnostics(&mut diagnostics, summary);
                        }
                        bayes.prior = prior;
                        TemporalResponseEstimator::new()
                            .estimate_bayesian(
                                series,
                                &[(&estimand, indexer)],
                                &qh,
                                case.result.status,
                                case.result.required_assumptions.clone(),
                                &bayes,
                                ctx,
                            )
                            .map_err(CausalError::from)?
                    }
                    InferenceMode::Frequentist => {
                        // Requested replicates ride each completion atom. An
                        // observation-adjusted series keeps the analytic fit: a
                        // naive resample would not refit the nuisance mechanism.
                        let mut estimator = TemporalResponseEstimator::new();
                        estimator.inner.bootstrap_replicates =
                            if series_owned.is_some() { 0 } else { self.bootstrap_replicates };
                        estimator
                            .estimate(
                                series,
                                &[(&estimand, indexer)],
                                &qh,
                                case.result.status,
                                case.result.required_assumptions.clone(),
                                ctx,
                            )
                            .map_err(CausalError::from)?
                    }
                };
                if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
                    response.estimate
                {
                    response.estimate =
                        ResponseIdentification::PointIdentified(ResponseValue::Surface {
                            grid: Arc::from([f64::from(horizon)]),
                            dimension: 1,
                            mean: Arc::from([value]),
                        });
                }
                let values = match &response.estimate {
                    ResponseIdentification::PointIdentified(ResponseValue::Surface {
                        mean,
                        ..
                    }) => mean.to_vec(),
                    _ => {
                        return Err(CausalError::Compile {
                            message: "temporal class response atom did not produce a surface"
                                .into(),
                        });
                    }
                };
                if values.len() != cells_per_horizon {
                    return Err(CausalError::Compile {
                        message: "temporal class response atom shape mismatch".into(),
                    });
                }
                horizon_values.push(values.clone());
                structural_atoms.push(crate::result::StructuralResponseAtom {
                    posterior: None,
                    response: None,
                    graph_key: ((horizon_index as u64) << 32) | case_index as u64,
                    weight: weights[case_index],
                    status: case.result.status,
                    value: match &response.estimate {
                        ResponseIdentification::PointIdentified(value)
                        | ResponseIdentification::PartiallyIdentified(value) => Some(value.clone()),
                        _ => None,
                    },
                });
                structural_atoms.last_mut().expect("response atom").response =
                    Some(response.clone());
                supports.push(response.support);
                if series_owned.is_some() && matches!(self.inference, InferenceMode::Frequentist) {
                    observation_atoms.push(ClassObservationAtom {
                        atom_index: structural_atoms.len() - 1,
                        horizon_index,
                        estimand: estimand.clone(),
                        indexer: indexer.clone(),
                        status: case.result.status,
                        assumptions: case.result.required_assumptions.clone(),
                        weight: weights[case_index],
                        adjustment,
                        horizon,
                        observation_query,
                        kind: ClassObservationKind::Curve,
                    });
                }
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand);
                    primary_identification = Some(case.result.clone());
                    primary_envelope = Some(bundle.envelope.clone());
                    assumptions = response.assumptions;
                }
            }
            if horizon_values.is_empty() {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal class response has no evaluable atom at horizon {horizon}"
                    ),
                });
            }
            horizon_supports.push(super::response_path::mix_support_reports(
                &supports[support_start..].iter().collect::<Vec<_>>(),
            ));
            for cell in 0..cells_per_horizon {
                let destination = cell * n_horizons + horizon_index;
                lower[destination] =
                    horizon_values.iter().map(|values| values[cell]).fold(f64::INFINITY, f64::min);
                upper[destination] = horizon_values
                    .iter()
                    .map(|values| values[cell])
                    .fold(f64::NEG_INFINITY, f64::max);
            }
        }
        let (grid, dimension) = match &query.functional {
            ResponseFunctional::MeanCurve { treatment, .. } => {
                let doses = treatment
                    .grid
                    .values()
                    .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                let grid: Vec<f64> = doses
                    .iter()
                    .flat_map(|dose| {
                        temporal.horizons.iter().flat_map(|horizon| [*dose, f64::from(*horizon)])
                    })
                    .collect();
                (grid, 2)
            }
            ResponseFunctional::InterventionResponse { .. } => {
                (temporal.horizons.iter().map(|horizon| f64::from(*horizon)).collect(), 1)
            }
            _ => unreachable!(),
        };
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: Arc::from(grid),
            dimension,
            lower: Arc::from(lower),
            upper: Arc::from(upper),
        };
        let support_refs = supports.iter().collect::<Vec<_>>();
        let graph_values = temporal_class_complete_values(
            &structural_atoms,
            &horizon_fingerprints,
            &response_envelope,
            n_horizons,
        );
        let (identified_mass, unidentified_mass, unevaluable_mass) =
            temporal_class_response_masses(&structural_atoms);
        let incomplete = unidentified_mass > 0.0 || unevaluable_mass > 0.0 || !full_mass_scope;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(query, &mut assumptions);
        }
        let mut support = super::response_path::mix_support_reports(&support_refs);
        support.point_status = (0..cells_per_horizon)
            .flat_map(|cell| (0..n_horizons).map(move |horizon| (cell, horizon)))
            .map(|(cell, horizon)| {
                horizon_supports[horizon].point_status.as_ref()?.get(cell).copied()
            })
            .collect::<Option<Vec<_>>>()
            .map(Arc::from);
        let mut response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: if incomplete {
                IdentificationStatus::GraphDependent
            } else {
                IdentificationStatus::PartiallyIdentified
            },
            estimate: if incomplete {
                ResponseIdentification::GraphDependent(graph_values)
            } else {
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
                    response_envelope.clone(),
                ))
            },
            uncertainty: ResponseUncertainty::None,
            support,
            assumptions: assumptions.clone(),
            provenance_id: Arc::from("estimate.temporal_response.class_envelope"),
            horizon_identification: Some(
                temporal
                    .horizons
                    .iter()
                    .map(|&horizon| antecedent_core::HorizonIdentification {
                        horizon,
                        status: if incomplete {
                            IdentificationStatus::GraphDependent
                        } else {
                            IdentificationStatus::PartiallyIdentified
                        },
                        method: Arc::from("temporal_class.completion_envelope"),
                        adjustment: Arc::from([]),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            ),
            interaction_structurally_zero: false,
        };
        if query.observation != ObservationSpec::Complete
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.bootstrap_replicates > 0
        {
            let identified_cases = structural_atoms
                .iter()
                .filter(|atom| atom.value.is_some())
                .map(|atom| atom.graph_key & 0xFFFF_FFFF)
                .collect::<std::collections::BTreeSet<_>>();
            let apply_class_band = identified_cases.len() == 1
                && full_mass_scope
                && unidentified_mass == 0.0
                && unevaluable_mass == 0.0;
            class_observation_band_applied = apply_class_observation_bootstrap(
                data,
                &observation_atoms,
                &mut structural_atoms,
                &mut response,
                &mut diagnostics,
                cells_per_horizon,
                n_horizons,
                apply_class_band,
                self.observation_options,
                self.bootstrap_replicates,
                ctx,
            )?;
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class response missing estimand".into(),
        })?;
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "temporal class response missing identification".into(),
        })?;
        identification.status = response.identification_status;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
        }
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_response.class_identified_set",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "bounds are pointwise ranges over completion-specific point surfaces; completion \
             weights are enumeration weights, not posterior probabilities or sampling uncertainty",
        ));
        if matches!(self.inference, InferenceMode::Bayesian(_)) && self.class_prior.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_posterior_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian completion posteriors are retained as atoms; enumeration weights \
                 are not mixed into a posterior",
            ));
        }
        if query.observation != ObservationSpec::Complete {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.observation_no_complete_band",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "observation is adjusted per completion; complete-data bands are not reused \
                 and joint observation/curve bands stay unavailable",
            ));
        }
        let conditional = self.class_prior.as_ref().and_then(|_| {
            temporal_class_response_mean(
                &structural_atoms,
                &response_envelope,
                temporal.horizons.len(),
                full_mass_scope,
            )
        });
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate: nan_effect(),
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id: if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            },
            treatment,
            outcome,
            identify_cached: self.temporal_class_cache_covers(&temporal.horizons),
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: primary_envelope.map(|envelope| {
                    crate::Identification::TemporalEnvelope {
                        envelope,
                        strategy: IdentifierId::GeneralizedAdjustment,
                        structure_version: self.graph.version(),
                    }
                }),
                response: Some(response),
                structural_response: Some(crate::result::StructuralResponseMixture {
                    weight_basis: if self.class_prior.is_some() {
                        crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
                    } else {
                        crate::result::StructuralWeightBasis::CompletionEnumeration
                    },
                    atoms: structural_atoms,
                    identified_mass,
                    unidentified_mass,
                    unevaluable_mass,
                    identified_set: Some(response_envelope),
                    conditional_on_identified: conditional,
                    full_mass_scope,
                    truncated_atoms,
                }),
                diagnostics: Some(diagnostics),
                // Observation circular-block bands ride the class response when a
                // single completion identifies; otherwise requested replicates
                // stay on the atoms and the class band is withheld.
                bootstrap_replicates_requested: Some(if class_observation_band_applied {
                    Some(self.bootstrap_replicates)
                } else {
                    None
                }),
                ..Default::default()
            },
        }))
    }

    fn execute_temporal_class_sequence_response(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware Sequence overlays require no discovery-estimation split",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence requires TemporalResponseSpec".into(),
        })?;
        let plan = antecedent_estimate::plan_from_response_query(query)
            .map_err(CausalError::from)?
            .ok_or_else(|| CausalError::Compile {
                message: "class-aware Sequence missing intervention plan".into(),
            })?;
        let overlays = plan.mechanism_overlays().ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence missing mechanism overlays".into(),
        })?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence transfer stays refused on incomplete classes",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut structural_atoms = Vec::new();
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut primary_envelope = None;
        let mut assumptions = antecedent_core::AssumptionSet::new();
        let mut full_mass_scope = true;
        let mut truncated_atoms = 0usize;
        let mut class_weights = TemporalClassWeights::new(self.class_prior.as_ref());
        let mut horizon_fingerprints: Vec<Vec<u64>> = Vec::new();
        let mut observation_atoms = Vec::new();
        let mut class_observation_band_applied = false;
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.sequence_overlay",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "class-aware Sequence runs per directed completion; a joint-coordinate miss \
             unidentifies that completion, not the class; no last-step collapse",
        )];
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let effect_query = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
                active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
                horizon_steps: horizon,
                max_history_lag: temporal.max_history_lag,
                target_population: query.target_population.clone(),
            };
            let bundle =
                self.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &effect_query)?;
            let envelope = &bundle.envelope.envelope;
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            horizon_fingerprints
                .push(envelope.cases.iter().map(|case| case.graph.fingerprint()).collect());
            full_mass_scope &= envelope.truncated_completions == 0;
            truncated_atoms += envelope.truncated_completions;
            let mut horizon_values = Vec::new();
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
            {
                if !identification_status_ok_for_case(case.result.status)
                    || case.result.estimands.is_empty()
                {
                    structural_atoms.push(crate::result::StructuralResponseAtom {
                        posterior: None,
                        response: None,
                        graph_key: ((horizon_index as u64) << 32) | case_index as u64,
                        weight: weights[case_index],
                        status: case.result.status,
                        value: None,
                    });
                    continue;
                }
                let Some(dag) = case.graph.sequential_dag() else {
                    structural_atoms.push(crate::result::StructuralResponseAtom {
                        posterior: None,
                        response: None,
                        graph_key: ((horizon_index as u64) << 32) | case_index as u64,
                        weight: weights[case_index],
                        status: case.result.status,
                        value: None,
                    });
                    continue;
                };
                let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
                let estimand =
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
                let mut qh = query.clone();
                qh.temporal.as_mut().expect("temporal query").horizons = Arc::from([horizon]);
                let observation_query = qh.clone();
                let adjustment = class_atom_adjustment(&estimand, indexer);
                let (effect, posterior) = if query.observation != ObservationSpec::Complete
                    && bayes.is_some()
                {
                    let (response, posterior) = antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(
                        data, &dag, &[(&estimand, indexer)], &qh, case.result.status,
                        case.result.required_assumptions.clone(), bayes.as_ref().expect("Bayesian"), ctx,
                    ).map_err(CausalError::from)?;
                    // The observed-data posterior summarizes the sequence response as
                    // its first quantity when no effect column is declared; never index
                    // an absent summary.
                    let column = posterior.effect_column().unwrap_or(0);
                    let effect = EffectEstimate::new(
                        posterior.summaries.mean.get(column).copied().unwrap_or(f64::NAN),
                        posterior.summaries.sd.get(column).copied().unwrap_or(f64::NAN),
                        response.assumptions,
                        OverlapPolicy::ExplicitOverride,
                    );
                    (effect, Some(posterior))
                } else {
                    let adjusted = if query.observation == ObservationSpec::Complete {
                        None
                    } else {
                        Some(
                            ObservationMechanismEstimator::new(self.observation_options)
                                .adjust_temporal_series(data, &qh, &adjustment)
                                .map_err(CausalError::from)?
                                .0,
                        )
                    };
                    antecedent_estimate::estimate_sequence_mechanisms(
                        adjusted.as_ref().unwrap_or(data),
                        &dag,
                        indexer,
                        &estimand,
                        outcome,
                        outcome_offset,
                        &overlays,
                        case.result.status,
                        case.result.required_assumptions.clone(),
                        0,
                        bayes.as_ref(),
                        ctx,
                    )
                    .map_err(CausalError::from)?
                };
                horizon_values.push(effect.ate);
                structural_atoms.push(crate::result::StructuralResponseAtom {
                    posterior,
                    response: None,
                    graph_key: ((horizon_index as u64) << 32) | case_index as u64,
                    weight: weights[case_index],
                    status: case.result.status,
                    value: Some(ResponseValue::Scalar(effect.ate)),
                });
                if query.observation != ObservationSpec::Complete && bayes.is_none() {
                    observation_atoms.push(ClassObservationAtom {
                        atom_index: structural_atoms.len() - 1,
                        horizon_index,
                        estimand: estimand.clone(),
                        indexer: indexer.clone(),
                        status: case.result.status,
                        assumptions: case.result.required_assumptions.clone(),
                        weight: weights[case_index],
                        adjustment,
                        horizon,
                        observation_query,
                        kind: ClassObservationKind::Sequence {
                            dag: Box::new(dag.clone()),
                            overlays: overlays.clone(),
                            outcome,
                            outcome_offset,
                        },
                    });
                }
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand);
                    primary_identification = Some(case.result.clone());
                    primary_envelope = Some(bundle.envelope.clone());
                    assumptions = effect.assumptions;
                }
            }
            if horizon_values.is_empty() {
                return Err(CausalError::Compile {
                    message: format!(
                        "class-aware Sequence has no evaluable directed completion at horizon {horizon}"
                    ),
                });
            }
            lower.push(horizon_values.iter().copied().fold(f64::INFINITY, f64::min));
            upper.push(horizon_values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: Arc::from(temporal.horizons.iter().map(|&h| f64::from(h)).collect::<Vec<_>>()),
            dimension: 1,
            lower: Arc::from(lower),
            upper: Arc::from(upper),
        };
        let (identified_mass, unidentified_mass, unevaluable_mass) =
            temporal_class_response_masses(&structural_atoms);
        let incomplete = unidentified_mass > 0.0 || unevaluable_mass > 0.0 || !full_mass_scope;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(query, &mut assumptions);
        }
        let mut response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: if incomplete {
                IdentificationStatus::GraphDependent
            } else {
                IdentificationStatus::PartiallyIdentified
            },
            estimate: if incomplete {
                ResponseIdentification::GraphDependent(temporal_class_complete_values(
                    &structural_atoms,
                    &horizon_fingerprints,
                    &response_envelope,
                    temporal.horizons.len(),
                ))
            } else {
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
                    response_envelope.clone(),
                ))
            },
            uncertainty: ResponseUncertainty::None,
            support: antecedent_core::SupportReport {
                status: antecedent_core::SupportStatus::Extrapolative,
                query_region: antecedent_core::SupportRegion {
                    minima: Arc::from([f64::from(temporal.horizons.first().copied().unwrap_or(1))]),
                    maxima: Arc::from([f64::from(temporal.horizons.last().copied().unwrap_or(1))]),
                },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                point_status: None,
            },
            assumptions: assumptions.clone(),
            provenance_id: Arc::from("estimate.temporal_response.class_sequence"),
            horizon_identification: Some(
                temporal
                    .horizons
                    .iter()
                    .map(|&horizon| antecedent_core::HorizonIdentification {
                        horizon,
                        status: if incomplete {
                            IdentificationStatus::GraphDependent
                        } else {
                            IdentificationStatus::PartiallyIdentified
                        },
                        method: Arc::from("temporal_class.completion_envelope"),
                        adjustment: Arc::from([]),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            ),
            interaction_structurally_zero: false,
        };
        if query.observation != ObservationSpec::Complete
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.bootstrap_replicates > 0
        {
            let identified_cases = structural_atoms
                .iter()
                .filter(|atom| atom.value.is_some())
                .map(|atom| atom.graph_key & 0xFFFF_FFFF)
                .collect::<std::collections::BTreeSet<_>>();
            let apply_class_band = identified_cases.len() == 1
                && full_mass_scope
                && unidentified_mass == 0.0
                && unevaluable_mass == 0.0;
            class_observation_band_applied = apply_class_observation_bootstrap(
                data,
                &observation_atoms,
                &mut structural_atoms,
                &mut response,
                &mut diagnostics,
                1,
                temporal.horizons.len(),
                apply_class_band,
                self.observation_options,
                self.bootstrap_replicates,
                ctx,
            )?;
        }
        if query.observation != ObservationSpec::Complete {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.observation_no_complete_band",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "observation is adjusted per completion; complete-data bands are not reused \
                 and joint observation/curve bands stay unavailable",
            ));
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence missing estimand".into(),
        })?;
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence missing identification".into(),
        })?;
        identification.status = response.identification_status;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
        }
        let conditional = self.class_prior.as_ref().and_then(|_| {
            temporal_class_response_mean(
                &structural_atoms,
                &response_envelope,
                temporal.horizons.len(),
                full_mass_scope,
            )
        });
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate: nan_effect(),
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id: if bayes.is_some() {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            },
            treatment,
            outcome,
            identify_cached: self.temporal_class_cache_covers(&temporal.horizons),
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: primary_envelope.map(|envelope| {
                    crate::Identification::TemporalEnvelope {
                        envelope,
                        strategy: IdentifierId::GeneralizedAdjustment,
                        structure_version: self.graph.version(),
                    }
                }),
                response: Some(response),
                structural_response: Some(crate::result::StructuralResponseMixture {
                    weight_basis: if self.class_prior.is_some() {
                        crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
                    } else {
                        crate::result::StructuralWeightBasis::CompletionEnumeration
                    },
                    atoms: structural_atoms,
                    identified_mass,
                    unidentified_mass,
                    unevaluable_mass,
                    identified_set: Some(response_envelope),
                    conditional_on_identified: conditional,
                    full_mass_scope,
                    truncated_atoms,
                }),
                diagnostics: Some(diagnostics),
                bootstrap_replicates_requested: Some(if class_observation_band_applied {
                    Some(self.bootstrap_replicates)
                } else {
                    None
                }),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_temporal_class(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
        {
            return self.execute_temporal_class_sequential(data, query, physical, ctx);
        }
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return self.execute_temporal_class_bayesian(data, query, physical, ctx);
        }
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
                message:
                    "temporal class-aware effect not identified (no identified mass in envelope)"
                        .into(),
            });
        }
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, self.graph.class())];
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let mut fitted_atoms = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let estimate = fit_frequentist_class_pulse_atom(
                data,
                query,
                &estimand,
                indexer,
                case.result.required_assumptions.clone(),
                self.split.as_ref(),
                ctx,
            )?;
            let w = case.weight.0;
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            total_w += w;
            atom_values[i] = Some(estimate.ate);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            fitted_atoms.push(ClassPulseAtom {
                weight: w,
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                assumptions: case.result.required_assumptions.clone(),
            });
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: Some(indexer.clone()),
                original: estimate,
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "temporal class-aware envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class-aware envelope missing estimand".into(),
        })?;
        let block = shared_circular_block_mixture_se(
            data,
            temporal_class_block_span(fitted_atoms.iter().map(|atom| &atom.indexer)),
            self.bootstrap_replicates,
            0xC1A5_5E00,
            ctx,
            |sampled| {
                let mut sum = 0.0;
                for atom in &fitted_atoms {
                    let estimate = fit_frequentist_class_pulse_atom(
                        sampled,
                        query,
                        &atom.estimand,
                        &atom.indexer,
                        atom.assumptions.clone(),
                        self.split.as_ref(),
                        ctx,
                    )
                    .ok()?;
                    sum += atom.weight * estimate.ate;
                }
                Some(sum / total_w)
            },
        );
        let se_analytic = mix_weighted_analytic_se(se_items);
        let se_bootstrap = block.se.is_finite().then_some(block.se);
        let estimate = EffectEstimate::from_parts(
            weighted_ate / total_w,
            se_analytic,
            se_bootstrap,
            (self.bootstrap_replicates > 0).then_some(block.completed),
            (self.bootstrap_replicates > 0)
                .then_some(block.attempted.saturating_sub(block.completed)),
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
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            &tabular,
            &ate_q,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            "temporal.linear.adjustment",
            &self.custom_validators,
            Some(query),
            self.split.as_ref(),
            Some(data.time_index()),
        )?;
        diagnostics.extend(na_diagnostics);
        if se_bootstrap.is_some() {
            diagnostics.push(envelope_shared_block_diagnostic(
                total_w,
                envelope.unidentified_weight.0,
                block.completed,
                block.attempted,
            ));
        } else if fitted_atoms.len() > 1 {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: EstimatorId::TemporalLinearAdjustment,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: (self.bootstrap_replicates > 0).then_some(block.completed),
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: bundle.envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(temporal_class_structural_mixture(
                    envelope,
                    crate::result::StructuralWeightBasis::CompletionEnumeration,
                    None,
                    &atom_values,
                )),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_temporal_class_sequential(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware multi-step sustained requires no discovery-estimation split",
            });
        }
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
                message:
                    "temporal class-aware effect not identified (no identified mass in envelope)"
                        .into(),
            });
        }
        let class_masses = self
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence transfer stays refused on incomplete classes",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, self.graph.class())];
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.sustained_window",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the contrast propagates through all intervened times on each directed completion; \
             no last-step collapse; bidirected MAG completions stay unevaluable",
        ));
        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut per_graph = Vec::new();
        let mut seq_atoms = Vec::new();
        let mut seq_atom_keys = Vec::new();
        let mut atom_posteriors = std::collections::HashMap::new();
        let mut primary_estimand = None;
        let mut unevaluable_weight = 0.0;
        let mut weighted_ate = 0.0;
        let mut total_w = 0.0;
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            let key = case.graph.fingerprint();
            keys.push(key);
            let weight = class_masses.as_ref().map_or(case.weight.0, |masses| masses[i]);
            weights.push(weight);
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            let Some(dag) = case.graph.sequential_dag() else {
                flags.push(GraphIdentFlag::Unidentified);
                unevaluable_weight += weight;
                diagnostics.push(Diagnostic::new(
                    "estimate.temporal_class.mag_sequential_unevaluable",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "completion {key} has bidirected edges; sequential g-comp is unevaluable"
                    ),
                ));
                continue;
            };
            let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
                .or_else(|_| {
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
                })?;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
            }
            flags.push(GraphIdentFlag::Identified);
            let mut assumptions = case.result.required_assumptions.clone();
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::ParametricRestriction(
                    antecedent_core::ParametricAssumption {
                        id: Arc::from("temporal.sequential.linear_sem"),
                        description: Arc::from(
                            "linear additive mechanisms on the identified unfolded DAG",
                        ),
                    },
                ),
                source: antecedent_core::AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("temporal.sequential.gcomp"),
                },
                scope: antecedent_core::AssumptionScope::Estimation,
                status: antecedent_core::AssumptionStatus::Declared,
            });
            let mut mechanisms = Vec::new();
            let (estimate, posterior) =
                antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                    data,
                    &dag,
                    indexer,
                    &estimand,
                    query,
                    case.result.status,
                    assumptions,
                    if bayes.is_some() { self.bootstrap_replicates } else { 0 },
                    bayes.as_ref(),
                    ctx,
                    Some(&mut mechanisms),
                )
                .map_err(CausalError::from)?;
            atom_values[i] = Some(estimate.ate);
            weighted_ate += weight * estimate.ate;
            total_w += weight;
            if let Some(posterior) = posterior.as_ref() {
                atom_posteriors.insert(key, posterior.clone());
                if let Some(col) = posterior.effect_column() {
                    if let Ok(draws) = posterior.draws.column(col) {
                        per_graph.push(GraphEffectDraws {
                            graph_key: key,
                            effect_draws: Arc::from(draws.to_vec()),
                        });
                    }
                }
            }
            seq_atom_keys.push(key);
            seq_atoms.push(super::sequential_validation::SequentialValidationAtom {
                weight,
                graph: dag,
                indexer: indexer.clone(),
                estimand,
                status: case.result.status,
                estimate,
                mechanisms,
            });
        }
        if seq_atoms.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal class-aware multi-step envelope had no evaluable directed \
                          completions"
                    .into(),
            });
        }
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class sequential envelope missing estimand".into(),
        })?;
        let weight_basis = if class_masses.is_some() {
            crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
        } else {
            crate::result::StructuralWeightBasis::CompletionEnumeration
        };
        let (estimate, mut posterior) = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            if class_masses.is_some()
                && envelope.truncated_completions == 0
                && seq_atoms.iter().any(|atom| atom.weight > 0.0)
            {
                let graphs = WeightedGraphSamples::new(weights, flags, keys)
                    .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                let mixed = aggregate_effect_envelope(
                    &graphs,
                    &per_graph,
                    InferenceDiagnostics::analytic("temporal_class_sequential"),
                    EnvelopeOptions::default(),
                )
                .map_err(CausalError::from)?;
                (effect_from_posterior(&mixed)?, Some(mixed))
            } else {
                diagnostics.push(Diagnostic::new(
                    "estimate.envelope.response_posterior_not_mixed",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "completion posteriors are retained as atoms; mixing requires caller-supplied \
                     class mass, positive evaluable mass, and uncapped class scope",
                ));
                (nan_effect(), None)
            }
        } else if total_w > 0.0 {
            let block = shared_circular_block_mixture_se(
                data,
                temporal_class_block_span(seq_atoms.iter().map(|atom| &atom.indexer)),
                self.bootstrap_replicates,
                0x5E0C_1A55,
                ctx,
                |sampled| {
                    let mut sum = 0.0;
                    for atom in &seq_atoms {
                        let (estimate, _) =
                            antecedent_estimate::temporal_sequential::estimate_sustained_window(
                                sampled,
                                &atom.graph,
                                &atom.indexer,
                                &atom.estimand,
                                query,
                                atom.status,
                                atom.estimate.assumptions.clone(),
                                0,
                                None,
                                ctx,
                            )
                            .ok()?;
                        sum += atom.weight * estimate.ate;
                    }
                    Some(sum / total_w)
                },
            );
            let se_analytic = mix_weighted_analytic_se(
                seq_atoms.iter().map(|atom| (atom.weight, atom.estimate.se_analytic)),
            );
            let se_bootstrap = block.se.is_finite().then_some(block.se);
            if se_bootstrap.is_some() {
                diagnostics.push(envelope_shared_block_diagnostic(
                    total_w,
                    envelope.unidentified_weight.0,
                    block.completed,
                    block.attempted,
                ));
            } else if seq_atoms.len() > 1 {
                diagnostics.push(envelope_se_omits_between_atom_variance());
            }
            (
                EffectEstimate::from_parts(
                    weighted_ate / total_w,
                    se_analytic,
                    se_bootstrap,
                    (self.bootstrap_replicates > 0).then_some(block.completed),
                    (self.bootstrap_replicates > 0)
                        .then_some(block.attempted.saturating_sub(block.completed)),
                    ctx.cancellation.is_cancelled(),
                    false,
                    seq_atoms[0].estimate.assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                    None,
                    None,
                ),
                None,
            )
        } else {
            (nan_effect(), None)
        };
        let (refutations, extra_diagnostics, predictive_checks) = if estimate.ate.is_finite() {
            super::sequential_validation::validate_sequential(
                data,
                query,
                &seq_atoms,
                self.refute,
                &self.custom_validators,
                bayes.as_ref(),
                posterior.as_mut(),
                estimate.ate,
                ctx,
            )?
        } else {
            let mut reports = Vec::new();
            let mut extra = Vec::new();
            let mut checks = Vec::new();
            for (atom, key) in seq_atoms.iter().zip(&seq_atom_keys) {
                let (mut local_reports, local_extra, local_checks) =
                    super::sequential_validation::validate_sequential(
                        data,
                        query,
                        std::slice::from_ref(atom),
                        self.refute,
                        &self.custom_validators,
                        bayes.as_ref(),
                        atom_posteriors.get_mut(key),
                        atom.estimate.ate,
                        ctx,
                    )?;
                for report in &mut local_reports {
                    report.refuter = Arc::from(format!("completion.{key}.{}", report.refuter));
                }
                reports.extend(local_reports);
                extra.extend(local_extra);
                checks.extend(local_checks);
            }
            (reports, extra, checks)
        };
        diagnostics.extend(extra_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let mut structural = temporal_class_structural_mixture(
            envelope,
            weight_basis,
            class_masses.as_deref(),
            &atom_values,
        );
        for atom in &mut structural.atoms {
            atom.posterior = atom_posteriors.remove(&atom.graph_key);
        }
        let total: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        structural.unevaluable_mass = unevaluable_weight / total;
        structural.unidentified_mass =
            (structural.unidentified_mass - structural.unevaluable_mass).max(0.0);
        let bootstrap_replicates_ok = estimate.bootstrap_replicates_ok;
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
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: bundle.envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                posterior,
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                predictive_checks,
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal.sequential.gcomp",
                    "estimate.temporal_class.envelope",
                )),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_temporal_class_bayesian(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let InferenceMode::Bayesian(cfg) = &self.inference else {
            return Err(CausalError::Compile {
                message: "temporal class Bayesian execute requires Bayesian inference".into(),
            });
        };
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
                message:
                    "temporal class-aware effect not identified (no identified mass in envelope)"
                        .into(),
            });
        }
        let class_masses = self
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, self.graph.class())];
        if class_masses.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.enumeration_not_probability",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completion-enumeration weights are not a class prior; no blended posterior \
                 is published",
            ));
        }
        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut fit_atoms = Vec::new();
        let mut primary_estimand = None;
        let mut envelope_conflict = None;
        let mut atom_means = Vec::new();
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            let key = case.graph.fingerprint();
            keys.push(key);
            let weight = class_masses.as_ref().map_or(case.weight.0, |masses| masses[i]);
            weights.push(weight);
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
            }
            flags.push(GraphIdentFlag::Identified);
            fit_atoms.push((key, estimand, case.result.status, indexer.clone(), weight));
        }
        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = 0;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
        let mut atoms = Vec::new();
        let mut per_graph = Vec::new();
        let mut ws = BayesianGCompWorkspace::default();
        for (key, estimand, status, indexer, weight) in &fit_atoms {
            let prep = estimator
                .prepare(data, estimand, query, indexer, self.split.as_ref(), &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
            let (resolved, conflict) = resolve_envelope_prior_anchor(cfg, &bprep, ctx)?;
            bayes.inner.prior = resolved;
            let mut posterior =
                bayes.fit(&bprep, *status, &mut ws, ctx).map_err(CausalError::from)?;
            if let Some(summary) = conflict.as_ref() {
                // A transfer conflict is a per-completion diagnostic. It rides the
                // atom posterior and the result diagnostics whether or not a class
                // prior licenses a mixture, and it never changes identification.
                push_conflict_diagnostics(&mut diagnostics, summary);
                posterior = with_conflict_summary(posterior, summary.clone());
            }
            envelope_conflict = envelope_conflict.or(conflict);
            let mean = posterior.summaries.mean.first().copied().unwrap_or_else(|| {
                posterior
                    .effect_column()
                    .and_then(|col| {
                        posterior.draws.column(col).ok().map(|draws| {
                            draws.iter().copied().sum::<f64>() / draws.len().max(1) as f64
                        })
                    })
                    .unwrap_or(f64::NAN)
            });
            atom_means.push(mean);
            let draws = posterior
                .effect_column()
                .and_then(|col| posterior.draws.column(col).ok().map(|d| Arc::from(d.to_vec())));
            if let Some(effect_draws) = draws {
                per_graph.push(GraphEffectDraws { graph_key: *key, effect_draws });
            }
            atoms.push(EnvelopeAtomFit {
                key: *key,
                prep: bprep,
                posterior,
                status: *status,
                weight: *weight,
                estimand: estimand.clone(),
                indexer: Some(indexer.clone()),
            });
        }
        if atoms.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal class-aware envelope had no estimable identified cases".into(),
            });
        }
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class Bayesian envelope missing estimand".into(),
        })?;
        let mut case_means: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut mean_idx = 0usize;
        for (i, case) in envelope.cases.iter().enumerate() {
            if identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                case_means[i] = atom_means.get(mean_idx).copied();
                mean_idx += 1;
            }
        }
        let (estimate, posterior, weight_basis) = if class_masses.is_some()
            && envelope.truncated_completions == 0
            && atoms.iter().any(|atom| atom.weight > 0.0)
        {
            let graphs = WeightedGraphSamples::new(weights, flags, keys)
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
            let mut mixed = aggregate_effect_envelope(
                &graphs,
                &per_graph,
                InferenceDiagnostics::analytic("temporal_class_envelope"),
                EnvelopeOptions::default(),
            )
            .map_err(CausalError::from)?;
            if let Some(summary) = envelope_conflict {
                mixed = with_conflict_summary(mixed, summary);
            }
            let estimate = effect_from_posterior(&mixed)?;
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.envelope",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!("unidentified_mass={}", mixed.unidentified_mass),
            ));
            (estimate, Some(mixed), crate::result::StructuralWeightBasis::CallerSuppliedClassPrior)
        } else {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_posterior_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completion posteriors are retained as atoms; mixing requires caller-supplied \
                 class mass, positive evaluable mass, and uncapped class scope",
            ));
            (
                nan_effect(),
                None,
                if class_masses.is_some() {
                    crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
                } else {
                    crate::result::StructuralWeightBasis::CompletionEnumeration
                },
            )
        };
        let mut structural_response = temporal_class_structural_mixture(
            envelope,
            weight_basis,
            class_masses.as_deref(),
            &case_means,
        );
        for atom in &mut structural_response.atoms {
            atom.posterior =
                atoms.iter().find(|fit| fit.key == atom.graph_key).map(|fit| fit.posterior.clone());
        }
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        for atom in &atoms {
            let refute_atom = EnvelopeRefuteAtom::from_fit(atom)?;
            let atom_ate = refute_atom.original.ate;
            let start = refutations.len();
            let (reports, na_diagnostics) = run_envelope_effect_refuters(
                &tabular,
                &ate_q,
                std::slice::from_ref(&refute_atom),
                &mut refute_ws,
                ctx,
                self.refute,
                EstimatorId::BayesianTemporalGcomp.as_str(),
                &self.custom_validators,
                Some(query),
                self.split.as_ref(),
                Some(data.time_index()),
            )?;
            refutations.extend(reports);
            diagnostics.extend(na_diagnostics);
            let mut atom_posterior = atom.posterior.clone();
            let mut atom_estimator = bayes.inner.clone();
            atom_estimator.prior = resolve_envelope_prior_anchor(cfg, &atom.prep, ctx)?.0;
            predictive_checks.extend(run_envelope_bayesian_full_validation(
                self.refute,
                cfg,
                &atom_estimator,
                std::slice::from_ref(atom),
                &mut atom_posterior,
                atom_ate,
                ctx,
                &mut refutations,
                &mut diagnostics,
            )?);
            for report in &mut refutations[start..] {
                report.refuter = Arc::from(format!("completion.{}.{}", atom.key, report.refuter));
            }
            if let Some(structural) =
                structural_response.atoms.iter_mut().find(|value| value.graph_key == atom.key)
            {
                structural.posterior = Some(atom_posterior);
            }
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: EstimatorId::BayesianTemporalGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
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
                posterior,
                structural_response: Some(structural_response),
                diagnostics: Some(diagnostics),
                predictive_checks,
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian.temporal.gcomp",
                    "estimate.temporal_class.envelope",
                )),
                ..Default::default()
            },
        }))
    }

    /// Whether the prepared class cache holds a certificate for every horizon.
    fn temporal_class_cache_covers(&self, horizons: &[u32]) -> bool {
        self.temporal_class_identification_cache.as_deref().is_some_and(|cache| {
            horizons
                .iter()
                .all(|horizon| cache.by_horizon.iter().any(|(cached, _)| cached == horizon))
        })
    }

    pub(crate) fn identify_temporal_class(
        &self,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
    ) -> Result<crate::analysis::prepared::CachedTemporalClassIdentification, CausalError> {
        if let Some(cache) = self.temporal_class_identification_cache.as_ref() {
            if let Some((_, envelope)) =
                cache.by_horizon.iter().find(|(horizon, _)| *horizon == query.horizon_steps)
            {
                return Ok(crate::analysis::prepared::CachedTemporalClassIdentification {
                    envelope: envelope.clone(),
                    by_horizon: Vec::new(),
                });
            }
        }
        let mut config = antecedent_identify::GeneralizedAdjustmentConfig::default();
        if let Some(max) = self.max_completions {
            config.max_completions = max;
        }
        let mut envelope = match self.graph.class() {
            GraphClass::TemporalCpdag => {
                let cpdag = self.graph.as_temporal_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "TemporalCpdag execute missing supplied graph".into(),
                })?;
                identify_temporal_cpdag_configured(identifier_id, cpdag, query, config)?
            }
            GraphClass::TemporalPag => {
                let pag = self.graph.as_temporal_pag().ok_or_else(|| CausalError::Compile {
                    message: "TemporalPag execute missing supplied graph".into(),
                })?;
                identify_temporal_pag_configured(identifier_id, pag, query, config)?
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal execute requires TemporalCpdag or TemporalPag",
                });
            }
        };
        crate::identify_api::refine_temporal_class_identification(
            &mut envelope,
            &self.query,
            query.horizon_steps,
        )?;
        Ok(crate::analysis::prepared::CachedTemporalClassIdentification {
            envelope,
            by_horizon: Vec::new(),
        })
    }
}

fn enforce_temporal_response_memory_budget(
    query: &ResponseQuery,
    temporal: &antecedent_core::TemporalResponseSpec,
    bootstrap_replicates: u32,
    ctx: &ExecutionContext,
) -> Result<(), CausalError> {
    const MIN_BYTES_PER_CELL: u64 = 40;

    let doses = match &query.functional {
        antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } => match &treatment.grid {
            antecedent_core::GridSpec::Values(values) => values.len(),
            antecedent_core::GridSpec::Linspace { points, .. } => *points,
        },
        antecedent_core::ResponseFunctional::InterventionResponse { .. } => 1,
        _ => return Ok(()),
    };
    let cells = doses.checked_mul(temporal.horizons.len()).ok_or_else(|| {
        CausalError::Resource { message: "temporal response output cell count overflow".into() }
    })?;
    // Observation bands retain one surface per outer replicate. Account for
    // that buffer as well as the result before any models or draws are fitted.
    let bytes_per_cell = if query.observation == ObservationSpec::Complete {
        MIN_BYTES_PER_CELL
    } else {
        MIN_BYTES_PER_CELL + u64::from(bootstrap_replicates) * 8
    };
    let minimum_bytes = u64::try_from(cells)
        .ok()
        .and_then(|cells| cells.checked_mul(bytes_per_cell))
        .ok_or_else(|| CausalError::Resource {
            message: "temporal response output byte estimate overflow".into(),
        })?;
    let limit =
        [ctx.memory.soft_limit_bytes, ctx.memory.hard_limit_bytes].into_iter().flatten().min();
    if let Some(limit) = limit {
        if minimum_bytes > limit {
            return Err(CausalError::Resource {
                message: format!(
                    "temporal response needs at least {minimum_bytes} output bytes for {cells} \
                     cells; memory limit is {limit}; coarsen the dose grid or request fewer \
                     horizons"
                ),
            });
        }
    }
    Ok(())
}

/// Per-horizon completion weights for a class-aware analysis.
///
/// Caller-supplied class mass binds once, at the first horizon, and is then
/// read by completion fingerprint. Without a prior the enumeration weights are
/// normalized per horizon; they are not probabilities.
struct TemporalClassWeights<'a> {
    prior: Option<&'a crate::ClassPrior>,
    binding: Option<crate::class_prior::ClassPriorBinding>,
}

impl<'a> TemporalClassWeights<'a> {
    const fn new(prior: Option<&'a crate::ClassPrior>) -> Self {
        Self { prior, binding: None }
    }

    fn for_envelope(
        &mut self,
        envelope: &antecedent_identify::TemporalClassEnvelope,
    ) -> Result<Vec<f64>, CausalError> {
        if let Some(prior) = self.prior {
            if self.binding.is_none() {
                self.binding = Some(crate::class_prior::ClassPriorBinding::bind(prior, envelope)?);
            }
            return self.binding.as_ref().expect("bound class prior").masses_for_envelope(envelope);
        }
        let raw: Vec<f64> = envelope.envelope.cases.iter().map(|case| case.weight.0).collect();
        let total: f64 = raw.iter().sum();
        Ok(if total > 0.0 { raw.iter().map(|weight| weight / total).collect() } else { raw })
    }
}

/// Completion curves stitched across horizons by completion fingerprint.
///
/// Each horizon identifies its own envelope, and a PAG window can retain a
/// different completion set or order per horizon, so the positional case index
/// is not a stable identity. Keys are completion fingerprints, the same keys
/// [`crate::ClassPrior::from_pairs`] accepts.
fn temporal_class_complete_values(
    atoms: &[crate::result::StructuralResponseAtom],
    horizon_fingerprints: &[Vec<u64>],
    envelope: &antecedent_core::ResponseEnvelope,
    n_horizons: usize,
) -> Vec<(u64, ResponseValue)> {
    let mut curves = std::collections::BTreeMap::<u64, Vec<Option<f64>>>::new();
    for atom in atoms {
        let values: &[f64] = match atom.value.as_ref() {
            Some(ResponseValue::Scalar(value)) => std::slice::from_ref(value),
            Some(ResponseValue::Surface { mean, .. }) => mean,
            _ => continue,
        };
        let horizon = (atom.graph_key >> 32) as usize;
        let case_index = (atom.graph_key & 0xffff_ffff) as usize;
        let Some(fingerprint) =
            horizon_fingerprints.get(horizon).and_then(|keys| keys.get(case_index)).copied()
        else {
            continue;
        };
        let curve = curves.entry(fingerprint).or_insert_with(|| vec![None; envelope.lower.len()]);
        for (cell, value) in values.iter().enumerate() {
            if let Some(target) = curve.get_mut(cell * n_horizons + horizon) {
                *target = Some(*value);
            }
        }
    }
    curves
        .into_iter()
        .filter_map(|(key, values)| {
            let values = values.into_iter().collect::<Option<Vec<_>>>()?;
            Some((
                key,
                ResponseValue::Surface {
                    grid: envelope.grid.clone(),
                    dimension: envelope.dimension,
                    mean: values.into(),
                },
            ))
        })
        .collect()
}

fn temporal_class_response_masses(
    atoms: &[crate::result::StructuralResponseAtom],
) -> (f64, f64, f64) {
    let total: f64 = atoms.iter().map(|atom| atom.weight).sum();
    let mut identified = 0.0;
    let mut unidentified = 0.0;
    let mut unevaluable = 0.0;
    for atom in atoms {
        if !identification_status_ok_for_case(atom.status) {
            unidentified += atom.weight;
        } else if atom.value.is_some() {
            identified += atom.weight;
        } else {
            unevaluable += atom.weight;
        }
    }
    if total > 0.0 {
        (identified / total, unidentified / total, unevaluable / total)
    } else {
        (0.0, 0.0, 0.0)
    }
}

fn temporal_class_response_mean(
    atoms: &[crate::result::StructuralResponseAtom],
    envelope: &antecedent_core::ResponseEnvelope,
    n_horizons: usize,
    full_mass_scope: bool,
) -> Option<ResponseValue> {
    if !full_mass_scope || n_horizons == 0 {
        return None;
    }
    // Licensed contract: `conditional_on_identified` is published only when
    // every identified atom is aligned *and* unidentified/unevaluable mass is
    // zero. Renormalizing over identified atoms would hide retained class mass.
    let (_, unidentified, unevaluable) = temporal_class_response_masses(atoms);
    if unidentified > 0.0 || unevaluable > 0.0 {
        return None;
    }
    let n_cells = envelope.lower.len() / n_horizons;
    let mut mean = vec![0.0; envelope.lower.len()];
    let mut totals = vec![0.0; n_horizons];
    for atom in atoms {
        if atom.weight == 0.0 {
            continue;
        }
        if !identification_status_ok_for_case(atom.status) {
            continue;
        }
        let values: &[f64] = match atom.value.as_ref()? {
            ResponseValue::Scalar(value) => std::slice::from_ref(value),
            ResponseValue::Surface { mean, .. } => mean,
            _ => return None,
        };
        let horizon = usize::try_from(atom.graph_key >> 32).ok()?;
        if horizon >= n_horizons || values.len() != n_cells {
            return None;
        }
        totals[horizon] += atom.weight;
        for (cell, value) in values.iter().enumerate() {
            mean[cell * n_horizons + horizon] += atom.weight * value;
        }
    }
    for (index, value) in mean.iter_mut().enumerate() {
        let total = totals[index % n_horizons];
        if total <= 0.0 {
            return None;
        }
        *value /= total;
    }
    Some(ResponseValue::Surface {
        grid: envelope.grid.clone(),
        dimension: envelope.dimension,
        mean: Arc::from(mean),
    })
}

fn temporal_class_structural_mixture(
    envelope: &IdentificationEnvelope<antecedent_identify::TemporalCompletionGraph>,
    weight_basis: crate::result::StructuralWeightBasis,
    masses: Option<&[f64]>,
    values: &[Option<f64>],
) -> crate::result::StructuralResponseMixture {
    let mut atoms = Vec::with_capacity(envelope.cases.len());
    let mut identified_weight = 0.0;
    let mut unidentified_weight = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (i, case) in envelope.cases.iter().enumerate() {
        let weight = masses.and_then(|m| m.get(i).copied()).unwrap_or(case.weight.0);
        let value = values.get(i).copied().flatten().filter(|v| v.is_finite());
        if identification_status_ok_for_case(case.result.status) && value.is_some() {
            identified_weight += weight;
            if let Some(v) = value {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        } else {
            unidentified_weight += weight;
        }
        atoms.push(crate::result::StructuralResponseAtom {
            posterior: None,
            response: None,
            graph_key: case.graph.fingerprint(),
            weight,
            status: case.result.status,
            value: value.map(ResponseValue::Scalar),
        });
    }
    let total = identified_weight + unidentified_weight;
    let identified_set = if lo.is_finite() && hi.is_finite() {
        Some(antecedent_core::ResponseEnvelope {
            grid: Arc::from([0.0]),
            dimension: 1,
            lower: Arc::from([lo]),
            upper: Arc::from([hi]),
        })
    } else {
        None
    };
    crate::result::StructuralResponseMixture {
        weight_basis,
        atoms,
        identified_mass: identified_weight / total,
        unidentified_mass: unidentified_weight / total,
        unevaluable_mass: 0.0,
        identified_set,
        conditional_on_identified: None,
        full_mass_scope: envelope.truncated_completions == 0,
        truncated_atoms: envelope.truncated_completions,
    }
}

fn temporal_class_envelope_diagnostic<G>(
    envelope: &IdentificationEnvelope<G>,
    class: GraphClass,
) -> Diagnostic {
    let code = if class == GraphClass::TemporalCpdag {
        "identify.temporal_cpdag.envelope"
    } else {
        "identify.temporal_pag.envelope"
    };
    Diagnostic::new(
        code,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "generalized.adjustment envelope: identified_mass={}, unidentified_mass={}, cases={}, limitations={:?}",
            envelope.identified_weight.0,
            envelope.unidentified_weight.0,
            envelope.cases.len(),
            envelope.critical_graph_features,
        ),
    )
}

fn aggregate_temporal_horizon_evidence<'a>(
    identifications: impl IntoIterator<Item = &'a IdentificationResult>,
) -> Result<(IdentificationStatus, antecedent_core::AssumptionSet), CausalError> {
    let mut aggregate_status = IdentificationStatus::NonparametricallyIdentified;
    let mut assumptions = antecedent_core::AssumptionSet::new();
    let mut saw_any = false;
    for identification in identifications {
        saw_any = true;
        match identification.status {
            IdentificationStatus::NonparametricallyIdentified => {}
            IdentificationStatus::IdentifiedUnderParametricRestrictions => {
                aggregate_status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
            }
            status => {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal response requires point identification at every horizon; got {status:?}"
                    ),
                });
            }
        }
        for record in &identification.required_assumptions.entries {
            if !assumptions.entries.contains(record) {
                assumptions.push(record.clone());
            }
        }
    }
    if !saw_any {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon identification".into(),
        });
    }
    Ok((aggregate_status, assumptions))
}

pub(super) struct MediationHorizonClick {
    pub(super) horizon: u32,
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) indexer: TemporalIndexer,
    pub(super) adjustment: Arc<[antecedent_data::LaggedColumn]>,
}

pub(super) fn mediation_horizon_clicks(
    cache: &crate::analysis::prepared::CachedTemporalIdentification,
    query: &antecedent_core::MediationQuery,
    mut adjustment_for: impl FnMut(u32) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError>,
) -> Result<Vec<MediationHorizonClick>, CausalError> {
    let mut clicks = Vec::with_capacity(query.horizons.len());
    for &horizon in query.horizons.iter() {
        let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
            message: format!(
                "prepared temporal mediation identification missing horizon {horizon}"
            ),
        })?;
        clicks.push(MediationHorizonClick {
            horizon,
            identification: entry.identification.clone(),
            estimand: entry.estimand.clone(),
            indexer: entry.indexer.clone(),
            adjustment: adjustment_for(horizon)?,
        });
    }
    if clicks.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        });
    }
    Ok(clicks)
}

pub(super) fn lagged_adjustment_from_entry(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Arc<[antecedent_data::LaggedColumn]> {
    let outcome_offset = i32::try_from(entry.horizon.saturating_sub(1)).unwrap_or(0);
    entry
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| {
            let key = entry.indexer.key_of(dense.raw()).ok()?;
            lagged_column_relative_to_outcome(key, outcome_offset)
        })
        .collect::<Vec<_>>()
        .into()
}

fn lagged_column_relative_to_outcome(
    key: antecedent_core::TemporalNodeKey,
    outcome_offset: i32,
) -> Option<antecedent_data::LaggedColumn> {
    let lag = outcome_offset.checked_sub(key.offset)?;
    if lag < 0 {
        return None;
    }
    Some(antecedent_data::LaggedColumn {
        variable: key.variable,
        lag: antecedent_core::Lag::from_raw(u32::try_from(lag).ok()?),
    })
}

pub(super) fn mediation_horizon_z_differs(clicks: &[MediationHorizonClick]) -> bool {
    let Some(first) = clicks.first() else {
        return false;
    };
    clicks.iter().skip(1).any(|click| click.adjustment.as_ref() != first.adjustment.as_ref())
}

fn mediation_click_adjustment_keys(
    click: &MediationHorizonClick,
) -> Arc<[antecedent_core::TemporalNodeKey]> {
    let mut keys = click
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|variable| click.indexer.key_of(variable.raw()).ok())
        .collect::<Vec<_>>();
    keys.sort();
    Arc::from(keys)
}

fn lagged_adjustment_from_temporal(
    temporal: &antecedent_identify::TemporalIdentificationResult,
) -> Arc<[antecedent_data::LaggedColumn]> {
    let outcome_offset = match &temporal.result.query {
        CausalQuery::TemporalEffect(q) => q.outcome_offset(),
        CausalQuery::Mediation(q) => {
            i32::try_from(q.horizons.first().copied().unwrap_or(1).saturating_sub(1)).unwrap_or(0)
        }
        _ => 0,
    };
    temporal.result.estimands.first().map_or_else(
        || Arc::from([]),
        |estimand| {
            estimand
                .adjustment_set
                .iter()
                .filter_map(|&dense| {
                    let key = temporal.indexer.key_of(dense.raw()).ok()?;
                    lagged_column_relative_to_outcome(key, outcome_offset)
                })
                .collect::<Vec<_>>()
                .into()
        },
    )
}

fn horizon_adjustment_sets_differ(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> bool {
    let Some(first) = entries.first() else {
        return false;
    };
    let first_z = named_adjustment_keys(first);
    entries.iter().skip(1).any(|entry| named_adjustment_keys(entry) != first_z)
}

fn contemporaneous_adjustment_variables(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> Vec<VariableId> {
    let mut adjustment = Vec::new();
    for entry in entries {
        for &dense in entry.estimand.adjustment_set.iter() {
            let Ok(key) = entry.indexer.key_of(dense.raw()) else {
                continue;
            };
            if !adjustment.contains(&key.variable) {
                adjustment.push(key.variable);
            }
        }
    }
    adjustment
}

fn named_adjustment_keys(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Vec<antecedent_core::TemporalNodeKey> {
    let mut keys: Vec<_> = entry
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| entry.indexer.key_of(dense.raw()).ok())
        .collect();
    keys.sort();
    keys
}

fn append_temporal_observation_assumptions(
    query: &ResponseQuery,
    assumptions: &mut antecedent_core::AssumptionSet,
) {
    for claim in query.observation_assumptions.iter() {
        let (id, description) = match claim {
            ObservationAssumption::IndependentGiven(vars) => (
                "observation.independent_given",
                format!("observation/censoring independent given {vars:?}"),
            ),
            ObservationAssumption::OutcomeIndependentGiven(vars) => (
                "observation.outcome_independent_given",
                format!("observation independent of latent outcome given {vars:?}"),
            ),
            ObservationAssumption::Structural(model) => {
                ("observation.structural", format!("structural observation model {model}"))
            }
        };
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: id.into(),
                description: description.into(),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Untestable,
        });
    }
}

fn apply_temporal_observation_result(
    response: &mut CausalResponse,
    adjusted: &antecedent_estimate::ObservationAdjustedOutcome,
) {
    let (minimum_weight, maximum_weight, effective_sample_size) = {
        let minimum = adjusted
            .weights
            .iter()
            .copied()
            .filter(|weight| *weight > 0.0)
            .fold(f64::INFINITY, f64::min);
        let maximum = adjusted.weights.iter().copied().fold(0.0_f64, f64::max);
        let sum = adjusted.weights.iter().sum::<f64>();
        let sum_squares = adjusted.weights.iter().map(|weight| weight * weight).sum::<f64>();
        let ess = if sum_squares > 0.0 { sum * sum / sum_squares } else { 0.0 };
        (if minimum.is_finite() { minimum } else { 0.0 }, maximum, ess)
    };
    response.uncertainty = ResponseUncertainty::None;
    response.provenance_id = Arc::from("estimate.temporal_response.observation_adjusted");
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from("response.observation_adjustment_weights"),
        values: Arc::from([minimum_weight, maximum_weight, effective_sample_size]),
        detail: Arc::from(
            "minimum positive weight, maximum weight, and Kish effective sample size; weights are diagnostic only and were already incorporated into the pseudo-outcome",
        ),
    });
    response.support.warnings.push(Diagnostic::new(
        "response.observation_joint_uncertainty_unavailable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "point estimate includes observation correction; uncertainty is omitted because complete-data curve bands do not account for the estimated observation mechanism",
    ));
    response.support.warnings.push(Diagnostic::new(
        "response.observation_adjustment_method",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        adjusted.method.clone(),
    ));
}

struct ObservationBootstrapBand {
    lower: Vec<f64>,
    upper: Vec<f64>,
    completed: u32,
    attempted: u32,
    cancelled: bool,
}

// Match the estimator bootstrap policy: fewer than two successes or more
// than half failed attempts cannot justify a reported interval.
fn summarize_observation_bootstrap(
    draws: &[Vec<f64>],
    attempted: u32,
    cancelled: bool,
) -> ObservationBootstrapBand {
    let completed = u32::try_from(draws.len()).unwrap_or(u32::MAX);
    let mut band = ObservationBootstrapBand {
        lower: Vec::new(),
        upper: Vec::new(),
        completed,
        attempted,
        cancelled,
    };
    if !bootstrap_has_enough_successes(completed as usize, attempted as usize) {
        return band;
    }
    let mut values = Vec::with_capacity(draws.len());
    for cell in 0..draws[0].len() {
        values.clear();
        values.extend(draws.iter().map(|draw| draw[cell]));
        values.sort_by(f64::total_cmp);
        band.lower.push(empirical_quantile(&values, 0.025));
        band.upper.push(empirical_quantile(&values, 0.975));
    }
    band
}

#[derive(Clone)]
enum ClassObservationKind {
    Curve,
    Sequence {
        dag: Box<antecedent_graph::TemporalDag>,
        overlays: Vec<antecedent_estimate::SequentialMechanismOverlay>,
        outcome: VariableId,
        outcome_offset: i32,
    },
}

struct ClassPulseAtom {
    weight: f64,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    assumptions: antecedent_core::AssumptionSet,
}

fn fit_frequentist_class_pulse_atom(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    assumptions: antecedent_core::AssumptionSet,
    split: Option<&DiscoveryEstimationSplit>,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    let prep = estimator
        .prepare(data, estimand, query, indexer, split, &ctx.kernel_policy)
        .map_err(CausalError::from)?;
    estimator
        .fit(&prep, &mut EstimationWorkspace::default(), ctx, assumptions)
        .map_err(CausalError::from)
}

struct ClassObservationAtom {
    atom_index: usize,
    horizon_index: usize,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    status: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    weight: f64,
    adjustment: Vec<VariableId>,
    #[allow(dead_code)]
    horizon: u32,
    observation_query: ResponseQuery,
    kind: ClassObservationKind,
}

fn class_atom_adjustment(
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
) -> Vec<VariableId> {
    estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| indexer.key_of(dense.raw()).ok().map(|key| key.variable))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn apply_class_observation_bootstrap(
    source: &TimeSeriesData,
    atoms: &[ClassObservationAtom],
    structural_atoms: &mut [crate::result::StructuralResponseAtom],
    response: &mut CausalResponse,
    diagnostics: &mut Vec<Diagnostic>,
    cells_per_horizon: usize,
    n_horizons: usize,
    apply_class_band: bool,
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    if replicates == 0 || atoms.is_empty() {
        return Ok(false);
    }
    let n = source.row_count();
    let structural_span = atoms
        .iter()
        .map(|atom| atom.indexer.history() as usize + atom.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    let block_length = structural_span.max(integer_cube_root_ceil(n)).min(n);
    let plan = antecedent_data::ResamplingPlan::CircularBlock { length: block_length };
    let mut index_scratch = Vec::with_capacity(n);
    let mut atom_draws: Vec<Vec<Vec<f64>>> = vec![Vec::new(); atoms.len()];
    let mut class_draws: Vec<Vec<f64>> = Vec::new();
    let mut attempted = 0u32;
    let class_len = cells_per_horizon.saturating_mul(n_horizons);

    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            break;
        }
        attempted += 1;
        let mut rng = ctx.rng.stream(0x0C1A_5500 + u64::from(replicate));
        let sampled =
            antecedent_data::resample_timeseries(source, plan, &mut rng, &mut index_scratch)
                .map_err(CausalError::from)?;
        let mut replicate_atom_values: Vec<Option<Vec<f64>>> = vec![None; atoms.len()];
        for (atom_i, atom) in atoms.iter().enumerate() {
            let Ok((adjusted, _)) = ObservationMechanismEstimator::new(options)
                .adjust_temporal_series(&sampled, &atom.observation_query, &atom.adjustment)
            else {
                continue;
            };
            let values = match &atom.kind {
                ClassObservationKind::Curve => {
                    let mut working = atom.observation_query.clone();
                    working.observation = ObservationSpec::Complete;
                    working.observation_assumptions = Arc::from([]);
                    let estimator = TemporalResponseEstimator::new();
                    let Ok(fitted) = estimator.estimate(
                        &adjusted,
                        &[(&atom.estimand, &atom.indexer)],
                        &working,
                        atom.status,
                        atom.assumptions.clone(),
                        ctx,
                    ) else {
                        continue;
                    };
                    match fitted.estimate {
                        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => {
                            vec![value]
                        }
                        ResponseIdentification::PointIdentified(ResponseValue::Surface {
                            mean,
                            ..
                        }) => mean.to_vec(),
                        _ => continue,
                    }
                }
                ClassObservationKind::Sequence { dag, overlays, outcome, outcome_offset } => {
                    let Ok((effect, _)) = antecedent_estimate::estimate_sequence_mechanisms(
                        &adjusted,
                        dag,
                        &atom.indexer,
                        &atom.estimand,
                        *outcome,
                        *outcome_offset,
                        overlays,
                        atom.status,
                        atom.assumptions.clone(),
                        0,
                        None,
                        ctx,
                    ) else {
                        continue;
                    };
                    vec![effect.ate]
                }
            };
            if values.iter().copied().all(f64::is_finite)
                && atom_draws[atom_i].first().is_none_or(|first| first.len() == values.len())
            {
                atom_draws[atom_i].push(values.clone());
                replicate_atom_values[atom_i] = Some(values);
            }
        }
        if apply_class_band && class_len > 0 {
            let mut mixed = vec![0.0; class_len];
            let mut totals = vec![0.0; n_horizons];
            let mut complete = true;
            for (atom_i, atom) in atoms.iter().enumerate() {
                let Some(values) = replicate_atom_values[atom_i].as_ref() else {
                    complete = false;
                    break;
                };
                if atom.horizon_index >= n_horizons || values.len() != cells_per_horizon {
                    complete = false;
                    break;
                }
                totals[atom.horizon_index] += atom.weight;
                for (cell, value) in values.iter().enumerate() {
                    mixed[cell * n_horizons + atom.horizon_index] += atom.weight * value;
                }
            }
            if complete && totals.iter().all(|&total| total > 0.0) {
                for (index, value) in mixed.iter_mut().enumerate() {
                    *value /= totals[index % n_horizons];
                }
                class_draws.push(mixed);
            }
        }
    }

    let cancelled = ctx.cancellation.is_cancelled();
    for (atom_i, atom) in atoms.iter().enumerate() {
        let band = summarize_observation_bootstrap(&atom_draws[atom_i], attempted, cancelled);
        if let Some(atom_response) =
            structural_atoms.get_mut(atom.atom_index).and_then(|item| item.response.as_mut())
        {
            apply_observation_bootstrap(atom_response, &band, replicates);
        } else if let Some(structural) = structural_atoms.get_mut(atom.atom_index) {
            if let Some(value) = structural.value.clone() {
                let mut stub = CausalResponse {
                    estimand: response.estimand.clone(),
                    identification_status: structural.status,
                    estimate: ResponseIdentification::PointIdentified(value),
                    uncertainty: ResponseUncertainty::None,
                    support: response.support.clone(),
                    assumptions: atom.assumptions.clone(),
                    provenance_id: Arc::from("estimate.temporal_response.class_atom"),
                    horizon_identification: None,
                    interaction_structurally_zero: false,
                };
                apply_observation_bootstrap(&mut stub, &band, replicates);
                structural.response = Some(stub);
            }
        }
    }

    if apply_class_band {
        let band = summarize_observation_bootstrap(&class_draws, attempted, cancelled);
        apply_observation_bootstrap(response, &band, replicates);
        Ok(!band.lower.is_empty())
    } else {
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_class.observation_class_band_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "class-level observation band is withheld on a multi-completion identified set; \
             each completion atom retains its outer circular-block band; complete-data bands \
             are not reused",
        ));
        response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
            id: Arc::from("response.observation_block_bootstrap"),
            values: Arc::from([
                f64::from(replicates),
                0.0,
                f64::from(u32::from(cancelled)),
                f64::from(attempted),
            ]),
            detail: Arc::from(
                "requested, completed class-band, cancellation, and attempted counts; the \
                 class band is withheld; atoms retain per-completion circular-block bands",
            ),
        });
        Ok(false)
    }
}

fn apply_observation_bootstrap(
    response: &mut CausalResponse,
    bootstrap: &ObservationBootstrapBand,
    requested: u32,
) {
    if !bootstrap.lower.is_empty() {
        response.uncertainty = ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: Arc::from(bootstrap.lower.clone()),
            upper: Arc::from(bootstrap.upper.clone()),
        };
        response.support.warnings.retain(|warning| {
            !matches!(
                warning.code.as_ref(),
                "response.observation_joint_uncertainty_unavailable"
                    | "response.temporal.sequence_uncertainty_unavailable"
            )
        });
    }
    if bootstrap.lower.is_empty() {
        response.support.warnings.push(Diagnostic::new(
            "response.observation_bootstrap_insufficient",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "observation bootstrap band withheld: {} of {} attempted replicates succeeded; \
                     at least two successes and at most half failed attempts are required",
                bootstrap.completed, bootstrap.attempted
            ),
        ));
    }
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from("response.observation_block_bootstrap"),
        values: Arc::from([
            f64::from(requested),
            f64::from(bootstrap.completed),
            f64::from(bootstrap.cancelled),
            f64::from(bootstrap.attempted),
        ]),
        detail: Arc::from(
            "requested, completed, cancellation, and attempted counts for the outer \
             circular-block bootstrap; each attempt refits observation nuisances and all \
             horizon models; bands require two successes and at most half failed attempts",
        ),
    });
}

#[allow(clippy::too_many_arguments)]
fn bootstrap_observation_adjusted_temporal_response(
    source: &TimeSeriesData,
    query: &ResponseQuery,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    status: IdentificationStatus,
    assumptions: &antecedent_core::AssumptionSet,
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<Option<ObservationBootstrapBand>, CausalError> {
    if replicates == 0 {
        return Ok(None);
    }
    let adjustment = contemporaneous_adjustment_variables(aligned);
    let identifications =
        aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect::<Vec<_>>();
    let n = source.row_count();
    let structural_span = aligned
        .iter()
        .map(|entry| entry.indexer.history() as usize + entry.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    let block_length = structural_span.max(integer_cube_root_ceil(n)).min(n);
    let plan = antecedent_data::ResamplingPlan::CircularBlock { length: block_length };
    let mut index_scratch = Vec::with_capacity(n);
    let mut draws: Vec<Vec<f64>> = Vec::new();
    let mut attempted = 0;
    let mut working_query = query.clone();
    working_query.observation = ObservationSpec::Complete;
    working_query.observation_assumptions = Arc::from([]);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            break;
        }
        attempted += 1;
        let mut rng = ctx.rng.stream(0x0B5E_0000 + u64::from(replicate));
        let sampled =
            antecedent_data::resample_timeseries(source, plan, &mut rng, &mut index_scratch)
                .map_err(CausalError::from)?;
        let Ok((adjusted, _)) = ObservationMechanismEstimator::new(options).adjust_temporal_series(
            &sampled,
            query,
            &adjustment,
        ) else {
            continue;
        };
        let estimator = TemporalResponseEstimator::new();
        let Ok(response) = estimator.estimate(
            &adjusted,
            &identifications,
            &working_query,
            status,
            assumptions.clone(),
            ctx,
        ) else {
            continue;
        };
        let values = match response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![value],
            ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
                mean.to_vec()
            }
            _ => continue,
        };
        if !values.is_empty()
            && values.iter().all(|value| value.is_finite())
            && draws.first().is_none_or(|first| first.len() == values.len())
        {
            draws.push(values);
        }
    }
    Ok(Some(summarize_observation_bootstrap(&draws, attempted, ctx.cancellation.is_cancelled())))
}

#[allow(clippy::too_many_arguments)]
fn bootstrap_observation_adjusted_sequence_response(
    source: &TimeSeriesData,
    graph: &antecedent_graph::TemporalDag,
    query: &ResponseQuery,
    overlays: &[antecedent_estimate::SequentialMechanismOverlay],
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    outcome: VariableId,
    assumptions: &antecedent_core::AssumptionSet,
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<Option<ObservationBootstrapBand>, CausalError> {
    if replicates == 0 {
        return Ok(None);
    }
    let adjustment = contemporaneous_adjustment_variables(aligned);
    let n = source.row_count();
    let structural_span = aligned
        .iter()
        .map(|entry| entry.indexer.history() as usize + entry.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    let block_length = structural_span.max(integer_cube_root_ceil(n)).min(n);
    let plan = antecedent_data::ResamplingPlan::CircularBlock { length: block_length };
    let mut index_scratch = Vec::with_capacity(n);
    let mut draws: Vec<Vec<f64>> = Vec::new();
    let mut attempted = 0;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            break;
        }
        attempted += 1;
        let mut rng = ctx.rng.stream(0x5E0B_0000 + u64::from(replicate));
        let sampled =
            antecedent_data::resample_timeseries(source, plan, &mut rng, &mut index_scratch)
                .map_err(CausalError::from)?;
        let Ok((adjusted, _)) = ObservationMechanismEstimator::new(options).adjust_temporal_series(
            &sampled,
            query,
            &adjustment,
        ) else {
            continue;
        };
        let mut values = Vec::with_capacity(aligned.len());
        let mut failed = false;
        for (horizon_steps, entry) in query
            .temporal
            .as_ref()
            .map(|temporal| temporal.horizons.iter().copied())
            .into_iter()
            .flatten()
            .zip(aligned.iter())
        {
            let outcome_offset = i32::try_from(horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX);
            let Ok((effect, _)) = antecedent_estimate::estimate_sequence_mechanisms(
                &adjusted,
                graph,
                &entry.indexer,
                &entry.estimand,
                outcome,
                outcome_offset,
                overlays,
                entry.identification.status,
                assumptions.clone(),
                0,
                None,
                ctx,
            ) else {
                failed = true;
                break;
            };
            values.push(effect.ate);
        }
        if !failed
            && !values.is_empty()
            && values.iter().all(|value| value.is_finite())
            && draws.first().is_none_or(|first| first.len() == values.len())
        {
            draws.push(values);
        }
    }
    Ok(Some(summarize_observation_bootstrap(&draws, attempted, ctx.cancellation.is_cancelled())))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn empirical_quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability.clamp(0.0, 1.0) * (sorted.len().saturating_sub(1)) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

/// Hydrate a shared coefficient prior into the unfolded design at every horizon.
///
/// Same-design and mapped transfer use the existing
/// [`resolve_bayesian_prior_with_conflict`] filter. Horizons whose coefficient
/// semantics differ fail closed even when their column counts happen to match.
/// Conflict-sensitive composition is also horizon-specific and therefore
/// refuses a multi-horizon surface until the response estimator accepts one
/// resolved prior per horizon.
fn resolve_temporal_response_prior(
    cfg: &BayesianConfig,
    data: &TimeSeriesData,
    temporal: &antecedent_core::TemporalResponseSpec,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
) -> Result<(Option<PriorSet>, Option<antecedent_prob::ConflictSummary>), CausalError> {
    if cfg.prior.is_none() && cfg.prior_artifact.is_none() && cfg.external_compose.is_none() {
        return Ok((None, None));
    }
    if aligned.len() > 1
        && cfg.external_compose.as_ref().is_some_and(|compose| compose.conflict_policy.is_some())
    {
        return Err(CausalError::Unsupported {
            message: "conflict-sensitive Bayesian temporal response prior transfer requires \
                      horizon-specific conflict evaluation",
        });
    }
    let mut resolved = None;
    let mut conflict = None;
    let mut expected_design = None;
    for entry in aligned {
        let pulse = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
            active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
            horizon_steps: entry.horizon,
            max_history_lag: temporal.max_history_lag,
            target_population: antecedent_core::TargetPopulation::AllObserved,
        };
        let prep = TemporalLinearAdjustment::new()
            .prepare(data, &entry.estimand, &pulse, &entry.indexer, None, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
        let adjustment = named_adjustment_keys(entry);
        match &expected_design {
            None => {
                expected_design = Some((bprep.design.ncols, adjustment));
                let (prior, summary) =
                    resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
                resolved = prior;
                conflict = summary;
            }
            Some((ncols, expected_adjustment))
                if temporal_prior_designs_match(
                    *ncols,
                    expected_adjustment,
                    bprep.design.ncols,
                    &adjustment,
                ) => {}
            Some(_) => {
                return Err(CausalError::Unsupported {
                    message: "Bayesian temporal response prior transfer requires a \
                              horizon-specific mapping",
                });
            }
        }
    }
    Ok((resolved, conflict))
}

fn temporal_prior_designs_match(
    expected_ncols: usize,
    expected_adjustment: &[antecedent_core::TemporalNodeKey],
    ncols: usize,
    adjustment: &[antecedent_core::TemporalNodeKey],
) -> bool {
    expected_ncols == ncols && expected_adjustment == adjustment
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus,
    };

    use super::*;

    fn horizon_result(
        status: IdentificationStatus,
        evidence: &'static str,
    ) -> IdentificationResult {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(evidence),
                description: Arc::from("horizon-specific evidence"),
            },
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("test.temporal_horizon"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        IdentificationResult::from_parts(
            status,
            CausalQuery::AverageEffect(query),
            Vec::new(),
            CausalExprArena::new(),
            DerivationTrace::default(),
            assumptions,
            Vec::new(),
            IdentificationPerformanceRecord::default(),
            None,
        )
    }

    #[test]
    fn multi_horizon_parent_status_is_the_most_conservative_slice() {
        assert_eq!(
            most_conservative_identification_status([
                IdentificationStatus::NonparametricallyIdentified,
                IdentificationStatus::GraphDependent,
            ]),
            IdentificationStatus::GraphDependent
        );
        assert_eq!(
            most_conservative_identification_status([
                IdentificationStatus::NonparametricallyIdentified,
                IdentificationStatus::NotIdentified,
            ]),
            IdentificationStatus::NotIdentified
        );
        assert_eq!(
            most_conservative_identification_status([
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                IdentificationStatus::NonparametricallyIdentified,
            ]),
            IdentificationStatus::IdentifiedUnderParametricRestrictions
        );
        assert_eq!(
            most_conservative_identification_status([] as [IdentificationStatus; 0]),
            IdentificationStatus::NotIdentified
        );
    }

    #[test]
    fn temporal_horizon_evidence_is_unioned_and_never_upgraded() {
        let first =
            horizon_result(IdentificationStatus::NonparametricallyIdentified, "horizon.one");
        let second = horizon_result(
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            "horizon.two",
        );
        let (status, assumptions) = aggregate_temporal_horizon_evidence([&first, &second]).unwrap();
        assert_eq!(status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        for evidence in ["horizon.one", "horizon.two"] {
            assert!(assumptions.entries.iter().any(|record| {
                matches!(&record.assumption, Assumption::Custom { id, .. } if id.as_ref() == evidence)
            }));
        }

        let partial = horizon_result(IdentificationStatus::PartiallyIdentified, "horizon.partial");
        let error = aggregate_temporal_horizon_evidence([&first, &partial]).unwrap_err();
        assert!(error.to_string().contains("at every horizon"));
    }

    #[test]
    fn equal_width_temporal_prior_designs_still_compare_adjustment_semantics() {
        let z = antecedent_core::TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 };
        let w = antecedent_core::TemporalNodeKey { variable: VariableId::from_raw(3), offset: -1 };
        assert!(temporal_prior_designs_match(3, &[z], 3, &[z]));
        assert!(!temporal_prior_designs_match(3, &[z], 3, &[w]));
    }
}

#[cfg(test)]
mod observation_bootstrap_tests {
    use super::*;

    #[test]
    fn excessive_failures_withhold_band_but_retain_counts() {
        let band = summarize_observation_bootstrap(&[vec![1.0], vec![3.0]], 5, false);
        assert!(band.lower.is_empty());
        assert!(band.upper.is_empty());
        assert_eq!(band.completed, 2);
        assert_eq!(band.attempted, 5);
    }

    #[test]
    fn cancelled_bootstrap_keeps_attempted_denominator_and_status() {
        let band = summarize_observation_bootstrap(&[vec![1.0], vec![3.0]], 2, true);
        assert!(band.cancelled);
        assert_eq!(band.completed, 2);
        assert!((band.lower[0] - 1.05).abs() < 1e-12);
        assert!((band.upper[0] - 2.95).abs() < 1e-12);
        let empty = summarize_observation_bootstrap(&[], 0, true);
        assert!(empty.cancelled && empty.lower.is_empty());
    }
}

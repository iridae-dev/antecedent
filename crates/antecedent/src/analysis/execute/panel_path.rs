// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    /// The panel license owner, consulted on the data each execute receives
    /// (a prepared refresh hands in new units).
    fn refuse_unlicensed_panel(&self, panel: &PanelData) -> Result<(), CausalError> {
        super::super::builder::refuse_unlicensed_panel_route(
            &self.query,
            self.graph.class(),
            &self.inference,
            panel,
            self.split.as_ref(),
        )
    }

    /// Multi-environment Pulse / single-step Sustained: the environments are the
    /// clusters of one pooled panel regression.
    ///
    /// Reading only `environment(0)` would answer the query on one environment and
    /// publish it as the multi-environment claim, and the answer would depend on the
    /// argument order.
    pub(super) fn execute_multi_env_temporal(
        &self,
        multi: &antecedent_data::MultiEnvironmentData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let units: Vec<antecedent_data::PanelUnit> = multi
            .environments()
            .iter()
            .enumerate()
            .map(|(index, series)| antecedent_data::PanelUnit {
                unit_id: u32::try_from(index).unwrap_or(u32::MAX),
                series: series.clone(),
            })
            .collect();
        let panel = PanelData::try_new(units)
            .map_err(|error| CausalError::Compile { message: format!("multi-env: {error}") })?;
        self.execute_pooled_panel(&panel, graph, query, physical, ctx, PanelClusters::Environments)
    }

    pub(super) fn execute_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_pooled_panel(panel, graph, query, physical, ctx, PanelClusters::Units)
    }

    fn execute_pooled_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        clusters: PanelClusters,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unlicensed_panel(panel)?;
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

        let unit_ids = panel_unit_ids(panel);
        let atom = PanelPulseAtom::prepare(
            panel,
            &estimand,
            query,
            &indexer,
            self.split.as_ref(),
            ctx,
            1.0,
        )?;
        let (prep, cluster_ids, _) = atom.stacked(&unit_ids)?;
        let mut diagnostics = vec![pooled_panel_estimand_diagnostic(clusters, unit_ids.len())];

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
                let mut estimate =
                    atom.fit(&unit_ids, ctx, identification.required_assumptions.clone())?;
                estimate.assumptions.push(panel_estimand_assumption(clusters, unit_ids.len()));
                let boot = panel_unit_bootstrap(
                    std::slice::from_ref(&atom),
                    self.bootstrap_replicates,
                    ctx,
                );
                boot.attach(&mut estimate, self.bootstrap_replicates);
                diagnostics.push(panel_cluster_se_diagnostic(clusters, unit_ids.len()));
                diagnostics.extend(boot.diagnostic(clusters, self.bootstrap_replicates));
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment.panel",
                    "estimate.temporal.linear.adjustment.panel",
                )
            }
        };

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

        let (bootstrap_replicates_ok, cancelled) =
            (estimate.bootstrap_replicates_ok, estimate.bootstrap_cancelled);
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
            bootstrap_replicates_ok,
            cancelled,
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

    /// Identify the class once for a panel class effect (or read the prepared
    /// certificate) and refuse an envelope without identified mass.
    fn panel_class_bundle(
        &self,
        physical: &PhysicalExecutionPlan,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<
        (IdentifierId, crate::analysis::prepared::CachedTemporalClassIdentification, bool),
        CausalError,
    > {
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
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "panel class-aware effect not identified (no identified mass in envelope)",
            ));
        }
        Ok((identifier_id, bundle, identify_cached))
    }

    pub(super) fn execute_panel_class(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unlicensed_panel(panel)?;
        if query.is_multi_step_sustained() {
            return self.execute_panel_class_sequential(panel, query, physical, ctx);
        }
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return self.execute_panel_class_bayesian(panel, query, physical, ctx);
        }
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.panel_class_bundle(physical, query, ctx)?;
        let envelope = &bundle.envelope.envelope;
        let mut diagnostics = vec![
            super::temporal_path::temporal_class_envelope_diagnostic(envelope, self.graph.class()),
            pooled_panel_estimand_diagnostic(PanelClusters::Units, panel.unit_count()),
        ];
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut fitted = Vec::new();
        let mut refute_atoms = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let unit_ids = panel_unit_ids(panel);
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let w = case.weight.0;
            let atom = PanelPulseAtom::prepare(
                panel,
                &estimand,
                query,
                indexer,
                self.split.as_ref(),
                ctx,
                w,
            )?;
            let mut estimate =
                atom.fit(&unit_ids, ctx, case.result.required_assumptions.clone())?;
            estimate
                .assumptions
                .push(panel_estimand_assumption(PanelClusters::Units, unit_ids.len()));
            refute_atoms.push(PanelRefuteAtom {
                key: case.graph.fingerprint(),
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                estimate: estimate.clone(),
            });
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            total_w += w;
            atom_values[i] = Some(estimate.ate);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            fitted.push(atom);
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
        let boot = panel_unit_bootstrap(&fitted, self.bootstrap_replicates, ctx);
        if boot.se.is_none() && fitted.len() > 1 {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        if fitted.len() == 1 {
            diagnostics.push(panel_cluster_se_diagnostic(PanelClusters::Units, unit_ids.len()));
        }
        diagnostics.extend(boot.diagnostic(PanelClusters::Units, self.bootstrap_replicates));
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
        let (refutations, na_diagnostics) = self.panel_class_pulse_refutations(
            panel,
            query,
            &refute_atoms,
            EstimatorId::TemporalLinearAdjustment.as_str(),
            ctx,
        )?;
        diagnostics.extend(na_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let mut estimate = EffectEstimate::from_parts(
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
        boot.attach(&mut estimate, self.bootstrap_replicates);
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let cancelled = estimate.bootstrap_cancelled;
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
            estimator_id: EstimatorId::TemporalLinearAdjustment,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
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

    /// Bayesian panel class Pulse / single-step Sustained under the series class-prior
    /// contract: each identified completion is the panel hierarchical GLS posterior
    /// (unit random intercepts) of its pooled design; with a caller class prior the
    /// completion draws are mixed by class mass, otherwise the effect is NaN and the
    /// completion posteriors stay atoms. No frequentist bootstrap runs.
    fn execute_panel_class_bayesian(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let InferenceMode::Bayesian(cfg) = &self.inference else {
            return Err(CausalError::Compile {
                message: "panel class Bayesian execute requires Bayesian inference".into(),
            });
        };
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.panel_class_bundle(physical, query, ctx)?;
        let envelope = &bundle.envelope.envelope;
        let class_masses = self
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        if class_masses.is_none() {
            diagnostics.push(super::temporal_path::enumeration_not_probability_diagnostic());
        }
        diagnostics
            .push(pooled_panel_estimand_diagnostic(PanelClusters::Units, panel.unit_count()));
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_effect.panel.class.bayesian_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each completion is the panel hierarchical GLS posterior (unit random intercepts) \
             of its pooled design; completion posteriors are mixed only under a caller class \
             prior",
        ));
        let unit_ids = panel_unit_ids(panel);
        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut per_graph = Vec::new();
        let mut fits: Vec<(
            u64,
            antecedent_estimate::PreparedBayesianProblem,
            Option<PriorSet>,
            u64,
        )> = Vec::new();
        let mut posteriors: Vec<(u64, CausalPosterior)> = Vec::new();
        let mut case_means: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut primary_estimand = None;
        let mut envelope_conflict = None;
        let mut refute_atoms = Vec::new();
        let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
        let mut ws = BayesianGCompWorkspace::default();
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            let key = case.graph.fingerprint();
            keys.push(key);
            weights.push(class_masses.as_ref().map_or(case.weight.0, |masses| masses[i]));
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            flags.push(GraphIdentFlag::Identified);
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let atom = PanelPulseAtom::prepare(
                panel,
                &estimand,
                query,
                indexer,
                self.split.as_ref(),
                ctx,
                weights[i],
            )?;
            let (prep, cluster_ids, _) = atom.stacked(&unit_ids)?;
            let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
            bprep.unit_ids = Some(cluster_ids);
            let (resolved, conflict) =
                resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
            // Each completion draws from its own seed stream; completions fitting the
            // same problem under the same prior share a seed, so one model carries one
            // posterior.
            let seed = fits
                .iter()
                .find(|(_, fitted, prior, _)| {
                    *prior == resolved && same_fitted_problem(fitted, &bprep)
                })
                .map_or_else(|| completion_fit_seed(ctx, key), |(_, _, _, seed)| *seed);
            bayes.inner.prior.clone_from(&resolved);
            bayes.inner.seed = seed;
            let mut posterior =
                bayes.fit(&bprep, case.result.status, &mut ws, ctx).map_err(CausalError::from)?;
            if let Some(summary) = conflict.as_ref() {
                push_conflict_diagnostics(&mut diagnostics, summary);
                posterior = with_conflict_summary(posterior, summary.clone());
            }
            envelope_conflict = envelope_conflict.or(conflict);
            let mut atom_estimate = effect_from_posterior(&posterior)?;
            atom_estimate.assumptions = case.result.required_assumptions.clone();
            case_means[i] = Some(atom_estimate.ate);
            refute_atoms.push(PanelRefuteAtom {
                key,
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                estimate: atom_estimate,
            });
            if let Some(draws) = posterior
                .effect_column()
                .and_then(|col| posterior.draws.column(col).ok().map(|d| Arc::from(d.to_vec())))
            {
                per_graph.push(GraphEffectDraws { graph_key: key, effect_draws: draws });
            }
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
            }
            fits.push((key, bprep, resolved, seed));
            posteriors.push((key, posterior));
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware envelope had no estimable identified cases".into(),
        })?;
        let evaluable_mass_positive = fits
            .iter()
            .any(|(key, ..)| keys.iter().zip(&weights).any(|(k, w)| k == key && *w > 0.0));
        let (estimate, posterior, weight_basis) =
            super::temporal_path::mix_class_effect_posteriors(
                super::temporal_path::ClassPosteriorMix {
                    class_prior_supplied: class_masses.is_some(),
                    truncated_completions: envelope.truncated_completions,
                    evaluable_mass_positive,
                    weights,
                    flags,
                    keys,
                    per_graph: &per_graph,
                    conflict: envelope_conflict,
                    label: "panel_temporal_class_envelope",
                },
                posteriors.iter().map(|(_, posterior)| posterior),
                &mut diagnostics,
            )?;
        let mut structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            weight_basis,
            class_masses.as_deref(),
            &case_means,
        );
        structural.identified_set_interval = posterior_identified_set_interval(
            posteriors.iter().map(|(_, posterior)| posterior),
            panel.total_rows(),
            envelope.truncated_completions > 0,
        );
        diagnostics.extend(
            structural
                .identified_set_interval
                .as_ref()
                .into_iter()
                .flat_map(identified_set_interval_diagnostics),
        );
        for atom in &mut structural.atoms {
            atom.posterior = posteriors
                .iter()
                .find(|(key, _)| *key == atom.graph_key)
                .map(|(_, posterior)| posterior.clone());
        }
        if self.bootstrap_replicates > 0 {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_effect.panel.class.bootstrap_not_used",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian panel class effects carry posterior uncertainty; requested bootstrap \
                 replicates are not used",
            ));
        }
        let (refutations, na_diagnostics) = self.panel_class_pulse_refutations(
            panel,
            query,
            &refute_atoms,
            EstimatorId::BayesianTemporalGcomp.as_str(),
            ctx,
        )?;
        diagnostics.extend(na_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
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
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian.temporal.gcomp.panel",
                    "estimate.temporal_class.envelope",
                )),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    /// The panel Pulse refuters of every fitted completion, as the DAG panel route
    /// runs them (stacked panel table, per-unit refits), each report prefixed with its
    /// completion key.
    fn panel_class_pulse_refutations(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        atoms: &[PanelRefuteAtom],
        estimator: &str,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let mut reports = Vec::new();
        let mut diagnostics = Vec::new();
        if matches!(self.refute, RefuteSuite::None) && self.custom_validators.is_empty() {
            return Ok((reports, diagnostics));
        }
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut workspace = EstimationWorkspace::default();
        for atom in atoms {
            let temporal_ctx = TemporalRefitContext {
                indexer: &atom.indexer,
                temporal_query: query,
                split: self.split.as_ref(),
                kernel_policy: &ctx.kernel_policy,
                time_index: None,
                panel: Some(panel),
            };
            let (mut local, local_diagnostics) = run_refuters(
                &stacked,
                &atom.estimand,
                &ate_q,
                &atom.estimate,
                &mut workspace,
                None,
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
                Some(temporal_ctx),
            )?;
            for report in &mut local {
                report.refuter = Arc::from(format!("completion.{}.{}", atom.key, report.refuter));
            }
            reports.extend(local);
            diagnostics.extend(local_diagnostics);
        }
        Ok((reports, diagnostics))
    }

    pub(super) fn execute_panel_response(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unlicensed_panel(panel)?;
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

        let mut diagnostics = Vec::new();
        let unit_responses = fit_unit_panel_responses(
            panel,
            &aligned,
            query,
            aggregate_status,
            &aggregate_assumptions,
            &self.inference,
            ctx,
            &mut diagnostics,
        )?;
        let mut response = average_unit_panel_responses(
            &unit_responses,
            matches!(self.inference, InferenceMode::Bayesian(_)),
        )?;
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
        diagnostics.push(unit_average_estimand_diagnostic(
            "estimate.temporal_response.panel.estimand",
            unit_responses.len(),
        ));
        diagnostics.push(panel_between_unit_band_diagnostic(unit_responses.len()));
        diagnostics.extend(panel_unit_average_bootstrap_note(self.bootstrap_replicates));
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_response.panel.bayesian_cluster_band",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian panel response mixes per-unit posterior means; the published band is \
                 frequentist between-unit cluster variation, not a posterior interval",
            ));
        }
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
        diagnostics.extend(response.support.warnings.iter().cloned());

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
        self.refuse_unlicensed_panel(panel)?;
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
        let identify_cached = self.temporal_class_cache_covers(&temporal.horizons);
        if self.temporal_class_identification_cache.is_some() && !identify_cached {
            return Err(CausalError::Compile {
                message: "prepared panel class identification cache missing a requested horizon"
                    .into(),
            });
        }
        let mut diagnostics = Vec::new();
        let mut assembly = super::temporal_path::ClassResponseAssembly::new(query, temporal)?;
        let mut class_weights =
            super::temporal_path::TemporalClassWeights::new(self.class_prior.as_ref());
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut primary_envelope = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut n_fitted = 0usize;
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let horizon_query = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
                active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
                horizon_steps: horizon,
                max_history_lag: temporal.max_history_lag,
                target_population: query.target_population.clone(),
            };
            if !identify_cached {
                report_identify_compute(ctx);
            }
            let bundle = self.identify_temporal_class(identifier_id, &horizon_query)?;
            let envelope = &bundle.envelope.envelope;
            diagnostics.push(super::temporal_path::temporal_class_envelope_diagnostic(
                envelope,
                self.graph.class(),
            ));
            if matches!(envelope.status, IdentificationStatus::NotIdentified)
                || envelope.identified_weight.0 <= 0.0
            {
                return Err(CausalError::not_identified(
                    envelope.status,
                    envelope.truncated_completions > 0,
                    "panel class-aware response not identified (no identified mass in envelope)",
                ));
            }
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            assembly.begin_horizon(envelope);
            let mut qh = query.clone();
            let mut temporal_h = temporal.clone();
            temporal_h.horizons = Arc::from([horizon]);
            qh.temporal = Some(temporal_h);
            n_fitted = 0;
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
            {
                if !identification_status_ok_for_case(case.result.status)
                    || case.result.estimands.is_empty()
                {
                    assembly.push_unevaluated(
                        horizon_index,
                        case_index,
                        weights[case_index],
                        case.result.status,
                    );
                    continue;
                }
                let estimand =
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
                let entry = crate::analysis::prepared::CachedTemporalHorizonIdentification {
                    horizon,
                    identification: case.result.clone(),
                    estimand: estimand.clone(),
                    indexer: indexer.clone(),
                };
                let unit_responses = fit_unit_panel_responses(
                    panel,
                    &[&entry],
                    &qh,
                    case.result.status,
                    &case.result.required_assumptions,
                    &self.inference,
                    ctx,
                    &mut diagnostics,
                )?;
                let response = average_unit_panel_responses(
                    &unit_responses,
                    matches!(self.inference, InferenceMode::Bayesian(_)),
                )?;
                let atom_assumptions = response.assumptions.clone();
                assembly.push_atom(
                    horizon_index,
                    case_index,
                    weights[case_index],
                    case.result.status,
                    horizon,
                    response,
                )?;
                n_fitted += 1;
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand);
                    primary_identification = Some(case.result.clone());
                    primary_envelope = Some(bundle.envelope.clone());
                    assumptions = atom_assumptions;
                }
            }
            assembly.end_horizon(horizon_index, horizon)?;
        }
        let super::temporal_path::ClassResponseParts {
            response,
            response_envelope,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
        } = assembly.finish(
            query,
            temporal,
            assumptions.clone(),
            "estimate.temporal_response.gcomp.panel.class",
        )?;
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            assumptions,
            OverlapPolicy::ExplicitOverride,
        );
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware response missing identification".into(),
        })?;
        identification.status = response.identification_status;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware response missing estimand".into(),
        })?;
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_response.panel.class.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel class response identifies each horizon and averages unit surfaces on \
                 {n_fitted} identified completions per horizon; the identified set is the \
                 pointwise range over those completion surfaces, and units are not stacked"
            ),
        ));
        diagnostics.extend(super::temporal_path::class_response_identified_set_diagnostics(
            matches!(self.inference, InferenceMode::Bayesian(_)),
            self.class_prior.is_some(),
        ));
        diagnostics.push(Diagnostic::new(
            "response.simultaneous_band_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "panel response withholds a simultaneous band: series circular-block simultaneous \
             coverage does not transfer to a unit-cluster contract",
        ));
        diagnostics.push(unit_average_estimand_diagnostic(
            "estimate.temporal_response.panel.estimand",
            panel.unit_count(),
        ));
        diagnostics.push(panel_between_unit_band_diagnostic(panel.unit_count()));
        diagnostics.extend(panel_unit_average_bootstrap_note(self.bootstrap_replicates));
        diagnostics.extend(response.support.warnings.iter().cloned());
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let conditional = self.class_prior.as_ref().and_then(|_| {
            super::temporal_path::temporal_class_response_mean(
                &assembly.structural_atoms,
                &response_envelope,
                temporal.horizons.len(),
                assembly.full_mass_scope,
            )
        });
        let structural = crate::result::StructuralResponseMixture {
            weight_basis: if self.class_prior.is_some() {
                crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
            } else {
                crate::result::StructuralWeightBasis::CompletionEnumeration
            },
            atoms: assembly.structural_atoms,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
            subsampled_out_mass: 0.0,
            identified_set: Some(response_envelope),
            identified_set_interval: None,
            conditional_on_identified: conditional,
            full_mass_scope: assembly.full_mass_scope,
            truncated_atoms: assembly.truncated_atoms,
        };
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
                certificate: primary_envelope.map(|envelope| {
                    crate::Identification::TemporalEnvelope {
                        envelope,
                        strategy: identifier_id,
                        structure_version: self.graph.version(),
                    }
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    /// Panel class multi-step Sustained: every directed completion's sequential
    /// contrast is fit on each unit's own series and averaged with equal unit
    /// weight. Frequentist completions mix by completion mass, with the between-unit
    /// SE of the per-unit mixture; Bayesian completions follow the series class-prior
    /// contract. Bidirected completions are unevaluable and reported as such.
    fn execute_panel_class_sequential(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.panel_class_bundle(physical, query, ctx)?;
        let envelope = &bundle.envelope.envelope;
        let class_masses = self
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        // The panel license refuses caller priors here; the per-unit fits use the
        // default prior.
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        diagnostics.push(unit_average_estimand_diagnostic(
            "estimate.temporal_effect.panel.estimand",
            panel.unit_count(),
        ));
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.sustained_window",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each completion's sequential contrast propagates through all intervened times and \
             is fit per panel unit, then averaged with equal unit weight; bidirected MAG \
             completions stay unevaluable",
        ));
        if bayes.is_some() && class_masses.is_none() {
            diagnostics.push(super::temporal_path::enumeration_not_probability_diagnostic());
        }
        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut per_graph = Vec::new();
        let mut atoms: Vec<(f64, u64, PanelSequentialAtom)> = Vec::new();
        let mut primary_estimand = None;
        let mut unevaluable_weight = 0.0;
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
            // Each completion's unit posteriors draw from that completion's seed; a
            // completion whose fitted unit mechanisms coincide with an earlier one's
            // is refit on that completion's seed, so one model carries one posterior.
            let mut seed = completion_fit_seed(ctx, key);
            let atom = loop {
                let atom = fit_panel_sequential_atom(
                    panel,
                    &dag,
                    indexer,
                    &estimand,
                    query,
                    case.result.status,
                    &assumptions,
                    bayes.as_ref(),
                    seed,
                    ctx,
                )?;
                let shared = atoms
                    .iter()
                    .find(|(_, _, earlier)| {
                        bayes.is_some() && earlier.seed != seed && earlier.same_mechanisms(&atom)
                    })
                    .map(|(_, _, earlier)| earlier.seed);
                match shared {
                    Some(shared) => seed = shared,
                    None => break atom,
                }
            };
            atom_values[i] = Some(atom.mean());
            if let Some(draws) = atom.mean_draws() {
                per_graph.push(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws) });
            }
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
            }
            atoms.push((weight, key, atom));
        }
        if atoms.is_empty() {
            return Err(CausalError::Compile {
                message: "panel class-aware multi-step envelope had no evaluable directed \
                          completions"
                    .into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware sequential envelope missing estimand".into(),
        })?;
        let total_weight: f64 = weights.iter().sum();
        let unevaluable_share =
            if total_weight > 0.0 { unevaluable_weight / total_weight } else { 0.0 };
        let evaluable_weight: f64 = atoms.iter().map(|(weight, ..)| *weight).sum();
        let (estimate, posterior, weight_basis) = if bayes.is_some() {
            let (estimate, mut posterior, basis) =
                super::temporal_path::mix_class_effect_posteriors(
                    super::temporal_path::ClassPosteriorMix {
                        class_prior_supplied: class_masses.is_some(),
                        truncated_completions: envelope.truncated_completions,
                        evaluable_mass_positive: evaluable_weight > 0.0,
                        weights,
                        flags,
                        keys,
                        per_graph: &per_graph,
                        conflict: None,
                        label: "panel_temporal_class_sequential",
                    },
                    std::iter::empty(),
                    &mut diagnostics,
                )?;
            // Unevaluable (bidirected) completions are flagged unidentified for the
            // mixture, which never draws from them; their share is unevaluable mass.
            if let Some(mixed) = posterior.as_mut() {
                mixed.unidentified_mass = (mixed.unidentified_mass - unevaluable_share).max(0.0);
            }
            (estimate, posterior, basis)
        } else if evaluable_weight > 0.0 {
            // Frozen completion weights: each unit's mixture contrast, averaged over
            // units, with the between-unit SE of that per-unit mixture (it carries the
            // completions' covariance through the shared units).
            let units = panel.unit_count();
            let unit_mixture: Vec<f64> = (0..units)
                .map(|u| {
                    atoms.iter().map(|(w, _, atom)| w / evaluable_weight * atom.unit_ates[u]).sum()
                })
                .collect();
            let ate = unit_mixture.iter().sum::<f64>() / units as f64;
            let se = antecedent_stats::sample_std(&unit_mixture) / (units as f64).sqrt()
                * antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL);
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_effect.panel.between_unit_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "SE is sd/sqrt(N) of the {} completions' frozen-weight mixture contrast over \
                     N={units} units, with N-1 degrees of freedom, scaled by t_(N-1)/z = {:.4} \
                     so estimate ± 1.96·SE is the 95% t_(N-1) interval",
                    atoms.len(),
                    antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL),
                ),
            ));
            let mut assumptions = atoms[0].2.assumptions.clone();
            assumptions.push(panel_estimand_assumption(PanelClusters::UnitAverage, units));
            (
                EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride),
                None,
                crate::result::StructuralWeightBasis::CompletionEnumeration,
            )
        } else {
            (nan_effect(), None, crate::result::StructuralWeightBasis::CompletionEnumeration)
        };
        let (refutations, validation_diagnostics, predictive_checks) =
            self.panel_sequential_validation(panel, query, &atoms, bayes.as_ref(), ctx)?;
        diagnostics.extend(validation_diagnostics);
        if self.bootstrap_replicates > 0 {
            diagnostics.push(
                panel_unit_average_bootstrap_note(self.bootstrap_replicates)
                    .expect("a positive replicate request always yields the note"),
            );
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let mut structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            weight_basis,
            class_masses.as_deref(),
            &atom_values,
        );
        let total: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        if total > 0.0 {
            structural.unevaluable_mass = unevaluable_weight / total;
            structural.unidentified_mass =
                (structural.unidentified_mass - structural.unevaluable_mass).max(0.0);
        }
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
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                predictive_checks,
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal.sequential.gcomp.panel",
                    "estimate.temporal_class.envelope",
                )),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    /// Sequential validation of every completion on every unit's own series (the
    /// series validator; the units are fit separately), each report prefixed with
    /// its unit and completion.
    fn panel_sequential_validation(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        atoms: &[(f64, u64, PanelSequentialAtom)],
        bayes: Option<&BayesianGComputationAte>,
        ctx: &ExecutionContext,
    ) -> Result<PanelSequentialValidation, CausalError> {
        let mut reports = Vec::new();
        let mut diagnostics = Vec::new();
        let mut checks = Vec::new();
        for (unit_index, unit) in panel.units().iter().enumerate() {
            for (weight, key, atom) in atoms {
                let fit = &atom.units[unit_index];
                let validation_atom = super::sequential_validation::SequentialValidationAtom {
                    weight: *weight,
                    graph: atom.graph.clone(),
                    indexer: atom.indexer.clone(),
                    estimand: atom.estimand.clone(),
                    status: atom.status,
                    estimate: fit.estimate.clone(),
                    mechanisms: fit.mechanisms.clone(),
                };
                let mut posterior = fit.posterior.clone();
                let unit_bayes = bayes.map(|estimator| BayesianGComputationAte {
                    seed: fit.seed,
                    ..estimator.clone()
                });
                let (mut local, local_diagnostics, local_checks) =
                    super::sequential_validation::validate_sequential(
                        &unit.series,
                        query,
                        std::slice::from_ref(&validation_atom),
                        self.refute,
                        &self.custom_validators,
                        unit_bayes.as_ref(),
                        posterior.as_mut(),
                        fit.estimate.ate,
                        ctx,
                        super::super::latency::predictive_check_sims(self.latency_mode),
                    )?;
                for report in &mut local {
                    report.refuter = Arc::from(format!(
                        "unit.{}.completion.{key}.{}",
                        unit.unit_id, report.refuter
                    ));
                }
                reports.extend(local);
                diagnostics.extend(local_diagnostics);
                checks.extend(local_checks);
            }
        }
        Ok((reports, diagnostics, checks))
    }
}

/// Equal-weight panel average of per-unit surfaces.
///
/// The published response is the panel's own: the mean surface, the between-unit
/// t band ([`panel_between_unit_band`]), the support merged over every unit
/// ([`merge_panel_unit_supports`]), and one panel band record in place of the
/// unit surfaces' own band assumptions.
fn average_unit_panel_responses(
    units: &[CausalResponse],
    bayesian: bool,
) -> Result<CausalResponse, CausalError> {
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
    let uncertainty = panel_between_unit_band(&acc, &unit_means);
    let mut response = first.clone();
    response.estimate = ResponseIdentification::PointIdentified(ResponseValue::Surface {
        grid,
        dimension,
        mean: acc.into(),
    });
    response.uncertainty = uncertainty;
    response.support = merge_panel_unit_supports(units);
    response.assumptions =
        panel_unit_average_assumptions(&first.assumptions, units.len(), bayesian);
    response.provenance_id = Arc::from("estimate.temporal_response.gcomp.panel");
    Ok(response)
}

/// Assumption records a unit surface carries about its own band, which the panel
/// does not publish.
const UNIT_BAND_ASSUMPTIONS: [&str; 3] = [
    "ols.homoskedastic.pointwise",
    "bayesian.temporal_response.linear_additive",
    antecedent_estimate::DEPENDENCE_ASSUMPTION_ID,
];

/// Support warnings that describe a unit surface's own (withheld) band.
const UNIT_BAND_WARNINGS: [&str; 2] = [
    antecedent_estimate::TEMPORAL_RESPONSE_BAND_WITHHELD,
    "estimate.temporal_response.bootstrap_degraded",
];

fn panel_assumption(id: &str, description: String) -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from(id),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp.panel"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// The unit surfaces' shared assumptions without their own band records, plus the
/// panel's estimand-and-band record (and, for Bayesian units, what each unit
/// surface is).
fn panel_unit_average_assumptions(
    unit: &antecedent_core::AssumptionSet,
    units: usize,
    bayesian: bool,
) -> antecedent_core::AssumptionSet {
    let mut assumptions = unit.clone();
    assumptions.entries.retain(|record| {
        !matches!(
            &record.assumption,
            antecedent_core::Assumption::ParametricRestriction(p)
                if UNIT_BAND_ASSUMPTIONS.contains(&p.id.as_ref())
        )
    });
    if bayesian {
        assumptions.push(panel_assumption(
            "bayesian.temporal_response.panel_unit_posterior_means",
            "each unit surface is the posterior mean of that unit's Gaussian linear-additive \
             unfolded outcome model at each horizon under its own long-run tempering (support \
             diagnostic response.temporal_bayesian.tempering reports the largest factor over \
             units); no posterior interval or credible band is published for the panel average"
                .to_owned(),
        ));
    }
    assumptions.push(panel_estimand_assumption(PanelClusters::UnitAverage, units));
    assumptions.push(panel_assumption(
        "panel.response.between_unit_band",
        format!(
            "the surface is the equal-weight average of the {units} unit-specific surfaces (every \
             unit weighs 1/{units} whatever its length); the pointwise 95% band is \
             mean ± t_(N-1)(0.975)·sd/sqrt(N) over the unit surfaces, which treats units as \
             independent draws from the unit population; no simultaneous band is published"
        ),
    ));
    assumptions
}

/// One support report for the panel average: the most conservative status and
/// per-cell status over units, every unit's warnings (minus the unit surfaces' own
/// band warnings), the union of the query regions, the per-horizon treatment range
/// shared by every unit (the average needs every unit to support a cell), and the
/// largest Bayesian tempering factor. No unit's simultaneous band survives.
fn merge_panel_unit_supports(units: &[CausalResponse]) -> antecedent_core::SupportReport {
    let refs: Vec<&antecedent_core::SupportReport> =
        units.iter().map(|unit| &unit.support).collect();
    let mut support = super::response_path::mix_support_reports(&refs);
    support.warnings.retain(|warning| !UNIT_BAND_WARNINGS.contains(&warning.code.as_ref()));
    let dims = support.query_region.minima.len();
    if refs
        .iter()
        .all(|r| r.query_region.minima.len() == dims && r.query_region.maxima.len() == dims)
    {
        support.query_region.minima = (0..dims)
            .map(|d| refs.iter().map(|r| r.query_region.minima[d]).fold(f64::INFINITY, f64::min))
            .collect();
        support.query_region.maxima = (0..dims)
            .map(|d| {
                refs.iter().map(|r| r.query_region.maxima[d]).fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
    }
    for diagnostic in &mut support.diagnostics {
        let id = diagnostic.id.clone();
        let per_unit: Option<Vec<&[f64]>> = refs
            .iter()
            .map(|r| {
                r.diagnostics
                    .iter()
                    .find(|d| d.id == id && d.values.len() == diagnostic.values.len())
                    .map(|d| d.values.as_ref())
            })
            .collect();
        let Some(per_unit) = per_unit else { continue };
        match id.as_ref() {
            "response.temporal.horizon_treatment_range"
            | "response.temporal.shifted_treatment_range" => {
                diagnostic.values = (0..diagnostic.values.len())
                    .map(|k| {
                        let values = per_unit.iter().map(|v| v[k]);
                        if k % 2 == 0 {
                            values.fold(f64::NEG_INFINITY, f64::max)
                        } else {
                            values.fold(f64::INFINITY, f64::min)
                        }
                    })
                    .collect();
                diagnostic.detail = Arc::from(format!(
                    "{}; intersection over the {} panel units (the largest minimum and the \
                     smallest maximum)",
                    diagnostic.detail,
                    units.len()
                ));
            }
            antecedent_estimate::TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC => {
                diagnostic.values = (0..diagnostic.values.len())
                    .map(|k| per_unit.iter().map(|v| v[k]).fold(f64::NEG_INFINITY, f64::max))
                    .collect();
                diagnostic.detail = Arc::from(format!(
                    "{}; largest factor over the {} panel units",
                    diagnostic.detail,
                    units.len()
                ));
            }
            _ => {}
        }
    }
    support
}

/// Between-unit pointwise band of an equal-weight average of `N` unit surfaces:
/// `mean ± t_{N−1}(0.975) · sd_{N−1} / √N` per cell.
///
/// The unit surfaces are treated as `N` independent draws from the unit population,
/// so the band is for the equal-weight super-population mean and carries each
/// unit's own estimation error through their dispersion. With `N − 1` degrees of
/// freedom the critical value is Student-t, not normal (at `N = 3` a normal band
/// covered about 0.82 of nominal 0.95). Below two units there is no dispersion and
/// no band.
fn panel_between_unit_band(mean: &[f64], unit_means: &[Arc<[f64]>]) -> ResponseUncertainty {
    let n = unit_means.len();
    if n < 2 {
        return ResponseUncertainty::None;
    }
    let critical =
        antecedent_stats::student_t_ppf(0.5 * (1.0 + PANEL_INTERVAL_LEVEL), (n - 1) as f64);
    let root_n = (n as f64).sqrt();
    let (lower, upper): (Vec<f64>, Vec<f64>) = mean
        .iter()
        .enumerate()
        .map(|(cell, center)| {
            let column: Vec<f64> = unit_means.iter().map(|row| row[cell]).collect();
            let half = critical * antecedent_stats::sample_std(&column) / root_n;
            (center - half, center + half)
        })
        .unzip();
    ResponseUncertainty::PointwiseBand {
        level: PANEL_INTERVAL_LEVEL,
        lower: lower.into(),
        upper: upper.into(),
    }
}

/// Diagnostic naming the between-unit band's construction and degrees of freedom,
/// or saying why no band is published.
fn panel_between_unit_band_diagnostic(units: usize) -> Diagnostic {
    if units < 2 {
        return Diagnostic::new(
            "estimate.temporal_response.panel.band_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "a one-unit panel has no between-unit dispersion, so no pointwise band is \
             published",
        );
    }
    Diagnostic::new(
        "estimate.temporal_response.panel.between_unit_band",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "pointwise 95% band = mean ± t_(N-1)(0.975)·sd/sqrt(N) over N={units} unit \
             surfaces ({} degrees of freedom, critical value {:.4}); units are independent \
             draws from the unit population",
            units - 1,
            antecedent_stats::student_t_ppf(0.5 * (1.0 + PANEL_INTERVAL_LEVEL), (units - 1) as f64),
        ),
    )
}

/// Requested bootstrap replicates are not used by the per-unit panel routes.
fn panel_unit_average_bootstrap_note(requested: u32) -> Option<Diagnostic> {
    (requested > 0).then(|| {
        Diagnostic::new(
            "estimate.temporal_response.panel.bootstrap_not_used",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "{requested} requested bootstrap replicates are not used: resampling the fitted \
                 unit surfaces would not refit them and adds nothing to the closed-form \
                 between-unit t band, which already carries each unit's estimation error"
            ),
        )
    })
}

/// Fit every unit's own temporal response surface.
///
/// Lags never cross a unit boundary: each unit's series is fit on its own. Under
/// Bayesian inference a caller prior (explicit, artifact or external composition) is
/// resolved against each unit's design with the series resolver, and its conflict
/// diagnostics are reported once per distinct message.
#[allow(clippy::too_many_arguments)]
fn fit_unit_panel_responses(
    panel: &PanelData,
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    query: &ResponseQuery,
    status: IdentificationStatus,
    assumptions: &antecedent_core::AssumptionSet,
    inference: &InferenceMode,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<Vec<CausalResponse>, CausalError> {
    let span = entries.iter().map(|entry| indexer_span(&entry.indexer)).max().unwrap_or(0);
    refuse_short_panel_units(panel, span)?;
    let identifications: Vec<_> =
        entries.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();
    let mut estimator = TemporalResponseEstimator::new();
    estimator.inner.bootstrap_replicates = 0;
    let mut unit_responses = Vec::with_capacity(panel.unit_count());
    match inference {
        InferenceMode::Bayesian(cfg) => {
            let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                message: "panel response route requires TemporalResponseSpec".into(),
            })?;
            let (treatment, outcome) =
                super::response_path::response_primary_pair(&query.functional)?;
            let mut conflicts = Vec::new();
            for unit in panel.units() {
                let mut bayes = bayesian_gcomp(cfg, ctx);
                let (prior, conflict) = super::temporal_path::resolve_temporal_response_prior(
                    cfg,
                    &unit.series,
                    temporal,
                    entries,
                    treatment,
                    outcome,
                    ctx,
                )?;
                if let Some(summary) = conflict.as_ref() {
                    push_conflict_diagnostics(&mut conflicts, summary);
                }
                bayes.prior = prior;
                unit_responses.push(
                    estimator
                        .estimate_bayesian(
                            &unit.series,
                            &identifications,
                            query,
                            status,
                            assumptions.clone(),
                            &bayes,
                            ctx,
                        )
                        .map_err(CausalError::from)?,
                );
            }
            for diagnostic in conflicts {
                if !diagnostics
                    .iter()
                    .any(|seen| seen.code == diagnostic.code && seen.message == diagnostic.message)
                {
                    diagnostics.push(diagnostic);
                }
            }
        }
        InferenceMode::Frequentist => {
            for unit in panel.units() {
                unit_responses.push(
                    estimator
                        .estimate(
                            &unit.series,
                            &identifications,
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

/// Refutation reports, diagnostics and predictive checks of a panel sequential run.
type PanelSequentialValidation =
    (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>, Vec<PredictiveCheckReport>);

/// One unit's sequential fit of one completion.
struct PanelSequentialUnitFit {
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    mechanisms: Vec<antecedent_estimate::temporal_sequential::SequentialBayesianMechanism>,
    seed: u64,
}

/// One directed completion's sequential contrast fit on every panel unit.
struct PanelSequentialAtom {
    graph: TemporalDag,
    indexer: TemporalIndexer,
    estimand: IdentifiedEstimand,
    status: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    /// Seed the completion's unit seeds derive from.
    seed: u64,
    units: Vec<PanelSequentialUnitFit>,
    unit_ates: Vec<f64>,
}

impl PanelSequentialAtom {
    /// Equal-weight mean of the unit contrasts.
    fn mean(&self) -> f64 {
        self.unit_ates.iter().sum::<f64>() / self.unit_ates.len() as f64
    }

    /// Draws of the equal-weight unit mean: the units' independent posteriors
    /// paired by draw index (their product posterior), when every unit has draws.
    fn mean_draws(&self) -> Option<Vec<f64>> {
        let columns: Option<Vec<&[f64]>> = self
            .units
            .iter()
            .map(|fit| {
                let posterior = fit.posterior.as_ref()?;
                posterior.draws.column(posterior.effect_column()?).ok()
            })
            .collect();
        let columns = columns?;
        let n = columns.iter().map(|c| c.len()).min()?;
        (n > 0).then(|| {
            (0..n)
                .map(|k| columns.iter().map(|c| c[k]).sum::<f64>() / columns.len() as f64)
                .collect()
        })
    }

    /// Whether every unit fitted the same mechanisms as in `other`.
    fn same_mechanisms(&self, other: &Self) -> bool {
        self.units.len() == other.units.len()
            && self
                .units
                .iter()
                .zip(&other.units)
                .all(|(a, b)| same_fitted_mechanisms(&a.mechanisms, &b.mechanisms))
    }
}

/// Fit one completion's sequential contrast on every unit's own series.
///
/// A unit that yields a non-finite contrast refuses the completion: the survivor
/// mean is not the panel effect. Under Bayesian inference each unit draws from its
/// own stream of `seed`.
#[allow(clippy::too_many_arguments)]
fn fit_panel_sequential_atom(
    panel: &PanelData,
    dag: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    status: IdentificationStatus,
    assumptions: &antecedent_core::AssumptionSet,
    bayes: Option<&BayesianGComputationAte>,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<PanelSequentialAtom, CausalError> {
    refuse_short_panel_units(panel, indexer_span(indexer))?;
    let mut units = Vec::with_capacity(panel.unit_count());
    let mut unit_ates = Vec::with_capacity(panel.unit_count());
    let mut last_assumptions = assumptions.clone();
    for (index, unit) in panel.units().iter().enumerate() {
        let unit_seed =
            completion_fit_seed(ctx, seed ^ (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let unit_bayes =
            bayes.map(|estimator| BayesianGComputationAte { seed: unit_seed, ..estimator.clone() });
        let mut mechanisms = Vec::new();
        let (estimate, posterior) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                &unit.series,
                dag,
                indexer,
                estimand,
                query,
                status,
                assumptions.clone(),
                0,
                unit_bayes.as_ref(),
                ctx,
                Some(&mut mechanisms),
            )
            .map_err(CausalError::from)?;
        if !estimate.ate.is_finite() {
            return Err(CausalError::Compile {
                message: format!(
                    "panel sequential completion has no finite contrast on unit {}; the \
                     survivor mean is not the panel effect",
                    unit.unit_id
                ),
            });
        }
        unit_ates.push(estimate.ate);
        last_assumptions = estimate.assumptions.clone();
        units.push(PanelSequentialUnitFit { estimate, posterior, mechanisms, seed: unit_seed });
    }
    Ok(PanelSequentialAtom {
        graph: dag.clone(),
        indexer: indexer.clone(),
        estimand: estimand.clone(),
        status,
        assumptions: last_assumptions,
        seed,
        units,
        unit_ates,
    })
}

/// One fitted completion of a panel class Pulse, kept for its refuters.
struct PanelRefuteAtom {
    key: u64,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    estimate: EffectEstimate,
}

/// Fewest lag-aligned rows a unit may contribute to a per-unit panel fit.
///
/// A per-unit route weighs every unit 1/N whatever its length, so a unit too short
/// for its own fit would carry the same weight as a long one. The floor is the
/// effective-row threshold below which a temporal response band is not calibrated
/// ([`antecedent_estimate::RESPONSE_SHORT_SERIES_ROWS`]).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const PANEL_MIN_UNIT_ROWS: usize = antecedent_estimate::RESPONSE_SHORT_SERIES_ROWS as usize;

/// Refuse a panel whose unit has fewer than [`PANEL_MIN_UNIT_ROWS`] rows left after
/// the structural window `span` (history plus horizon) is aligned.
fn refuse_short_panel_units(panel: &PanelData, span: usize) -> Result<(), CausalError> {
    for unit in panel.units() {
        let rows = unit.series.row_count().saturating_sub(span);
        if rows < PANEL_MIN_UNIT_ROWS {
            return Err(CausalError::Compile {
                message: format!(
                    "panel unit {} has {rows} lag-aligned rows, below the {PANEL_MIN_UNIT_ROWS} \
                     a per-unit panel fit requires: every unit carries weight 1/N in the \
                     equal-weight average whatever its length",
                    unit.unit_id
                ),
            });
        }
    }
    Ok(())
}

/// Structural window of an indexer: history plus horizon.
fn indexer_span(indexer: &TemporalIndexer) -> usize {
    indexer.history() as usize + indexer.horizon() as usize
}

/// What a pooled panel regression clusters on, and (for the record only) the
/// equal-weight average of per-cluster fits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PanelClusters {
    /// Panel units.
    Units,
    /// Environments of a multi-environment study.
    Environments,
    /// Not a pooled fit: the equal-weight average of per-unit fits.
    UnitAverage,
}

impl PanelClusters {
    const fn label(self) -> &'static str {
        match self {
            Self::Units | Self::UnitAverage => "unit",
            Self::Environments => "environment",
        }
    }
}

/// The estimand a pooled panel regression answers.
fn pooled_panel_estimand_diagnostic(clusters: PanelClusters, units: usize) -> Diagnostic {
    let label = clusters.label();
    Diagnostic::new(
        "estimate.temporal_effect.panel.estimand",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "pooled common-coefficient panel effect over {units} {label}s: one coefficient is \
             fit on the stacked {label} rows, so a {label}'s influence grows with its length \
             and its within-{label} treatment variance. This is not the equal-weight average \
             of {label}-specific effects that the panel response and multi-step Sustained \
             routes report; under {label} heterogeneity the two differ"
        ),
    )
}

/// The estimand an equal-weight average of per-unit fits answers.
fn unit_average_estimand_diagnostic(code: &'static str, units: usize) -> Diagnostic {
    Diagnostic::new(
        code,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "equal-weight average of {units} unit-specific fits: every unit weighs 1/{units} \
             whatever its length, targeting the mean over the unit population. This is not the \
             pooled common-coefficient effect that panel Pulse reports; under unit \
             heterogeneity the two differ"
        ),
    )
}

/// The estimand record every panel route carries on its estimate.
fn panel_estimand_assumption(
    clusters: PanelClusters,
    units: usize,
) -> antecedent_core::AssumptionRecord {
    let label = clusters.label();
    let (id, description) = if clusters == PanelClusters::UnitAverage {
        (
            "panel.estimand.equal_weight_unit_average",
            format!(
                "the reported effect is the equal-weight average of {units} unit-specific fits \
                 (each unit weighs 1/{units} whatever its length), not the pooled \
                 common-coefficient panel effect"
            ),
        )
    } else {
        (
            "panel.estimand.pooled_common_coefficient",
            format!(
                "the reported effect is the pooled common-coefficient panel effect over {units} \
                 {label}s (one coefficient on the stacked {label} rows; a {label}'s influence \
                 grows with its length and within-{label} treatment variance), not the \
                 equal-weight average of {label}-specific effects"
            ),
        )
    };
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from(id),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal.panel"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// Two-sided level of every published panel interval (the result's interval
/// binding): the level the facade publishes, not a panel-specific one.
const PANEL_INTERVAL_LEVEL: f64 = crate::result::REPORTED_SE_INTERVAL_LEVEL;

/// Estimator for the pooled panel regression: no row bootstrap (the unit cluster
/// bootstrap below resamples whole units) and the declared overlap policy.
fn pooled_panel_estimator() -> TemporalLinearAdjustment {
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    estimator
}

/// One pooled Pulse design (the DAG, or one identified completion): every unit's own
/// lag-aligned block, prepared once and stacked per fit.
struct PanelPulseAtom {
    weight: f64,
    units: Vec<antecedent_estimate::PreparedEstimationProblem>,
}

impl PanelPulseAtom {
    fn prepare(
        panel: &PanelData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        ctx: &ExecutionContext,
        weight: f64,
    ) -> Result<Self, CausalError> {
        let units = pooled_panel_estimator()
            .prepare_panel_units(panel, estimand, query, indexer, split, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        Ok(Self { weight, units })
    }

    /// Stacked design with the panel's own unit ids.
    fn stacked(
        &self,
        unit_ids: &[u32],
    ) -> Result<(antecedent_estimate::PreparedEstimationProblem, Vec<u32>, Vec<i64>), CausalError>
    {
        pooled_panel_estimator()
            .stack_panel_units(&self.units.iter().collect::<Vec<_>>(), unit_ids)
            .map_err(CausalError::from)
    }

    /// Pooled common-coefficient fit with the cluster-by-unit SE
    /// ([`panel_cluster_se_fit`]).
    fn fit(
        &self,
        unit_ids: &[u32],
        ctx: &ExecutionContext,
        assumptions: antecedent_core::AssumptionSet,
    ) -> Result<EffectEstimate, CausalError> {
        let (prep, cluster_ids, panel_times) = self.stacked(unit_ids)?;
        panel_cluster_se_fit(&prep, cluster_ids, panel_times, ctx, assumptions)
    }
}

/// Pooled fit whose analytic SE is the Arellano cluster-by-unit variance at `G − 1`
/// degrees of freedom.
///
/// `PanelClusterHac { lag: 0 }` is the full within-unit meat `Σ_g s_g s_g'`, robust to
/// any within-unit dependence as the number of units grows, with the
/// `G/(G−1)·(n−1)/(n−p)` finite-sample factor. A positive lag would truncate the
/// within-unit covariance to a Bartlett window and under-cover under persistent
/// scores. The SE is then scaled by `t_{G−1}/z` at the 0.95 interval level
/// ([`antecedent_stats::few_cluster_t_ratio`]) so `estimate ± 1.96·se` is the
/// `t_{G−1}` interval the few-cluster reference distribution requires.
fn panel_cluster_se_fit(
    prep: &antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: Vec<u32>,
    panel_times: Vec<i64>,
    ctx: &ExecutionContext,
    assumptions: antecedent_core::AssumptionSet,
) -> Result<EffectEstimate, CausalError> {
    let clusters = distinct_clusters(&cluster_ids);
    let mut estimator = pooled_panel_estimator();
    estimator.inner.cluster_ids = Some(cluster_ids);
    estimator.inner.panel_times = Some(panel_times);
    estimator.inner.se_kind = AnalyticSeKind::PanelClusterHac { lag: 0 };
    let mut workspace = EstimationWorkspace::default();
    let mut estimate =
        estimator.fit(prep, &mut workspace, ctx, assumptions).map_err(CausalError::from)?;
    estimate.se_analytic *= antecedent_stats::few_cluster_t_ratio(clusters, PANEL_INTERVAL_LEVEL);
    Ok(estimate)
}

fn distinct_clusters(cluster_ids: &[u32]) -> usize {
    cluster_ids.iter().collect::<std::collections::BTreeSet<_>>().len()
}

/// Panel unit ids in panel order.
fn panel_unit_ids(panel: &PanelData) -> Vec<u32> {
    panel.units().iter().map(|unit| unit.unit_id).collect()
}

/// Unit cluster bootstrap of a frozen-weight mixture of pooled Pulse atoms.
struct PanelUnitBootstrap {
    se: Option<f64>,
    completed: u32,
    failed: u32,
    cancelled: bool,
    units: usize,
}

impl PanelUnitBootstrap {
    /// Record the bootstrap on `estimate` (SE, replicate counts, cancellation).
    fn attach(&self, estimate: &mut EffectEstimate, requested: u32) {
        if requested == 0 {
            return;
        }
        estimate.se_bootstrap = self.se;
        estimate.bootstrap_replicates_ok = Some(self.completed);
        estimate.bootstrap_replicates_failed = Some(self.failed);
        estimate.bootstrap_cancelled = self.cancelled;
    }

    fn diagnostic(&self, clusters: PanelClusters, requested: u32) -> Option<Diagnostic> {
        let label = clusters.label();
        (requested > 0).then(|| {
            Diagnostic::new(
                "estimate.temporal_effect.panel.unit_bootstrap",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{label} cluster bootstrap: {} of {} {label}s resampled with replacement \
                     per replicate, each draw its own cluster, the pooled regression refit on \
                     the restacked {label} blocks (lag windows stay inside their {label}); \
                     {} replicates succeeded, {} failed{}; replicate SD scaled by \
                     sqrt(G/(G-1)) and t_(G-1)/z at the 0.95 level",
                    self.units,
                    self.units,
                    self.completed,
                    self.failed,
                    if self.cancelled { ", cancelled" } else { "" },
                ),
            )
        })
    }
}

/// Resample whole units (with replacement, each draw a fresh cluster), restack each
/// atom's prepared unit blocks, and refit the frozen-weight mixture per replicate.
///
/// Units are prepared once; a replicate only restacks and refits. The replicate SD
/// is scaled by `sqrt(G/(G−1))` (the Arellano finite-sample factor the analytic SE
/// carries) and by `t_{G−1}/z`, so both SEs read on the same `t_{G−1}` scale. The SE is
/// withheld unless at least two replicates succeeded and successes outnumber
/// failures.
fn panel_unit_bootstrap(
    atoms: &[PanelPulseAtom],
    replicates: u32,
    ctx: &ExecutionContext,
) -> PanelUnitBootstrap {
    let units = atoms.first().map_or(0, |atom| atom.units.len());
    let total_weight: f64 = atoms.iter().map(|atom| atom.weight).sum();
    let mut out = PanelUnitBootstrap { se: None, completed: 0, failed: 0, cancelled: false, units };
    if replicates == 0
        || units < 2
        || !matches!(total_weight.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
        || atoms.iter().any(|atom| atom.units.len() != units)
    {
        out.cancelled = ctx.cancellation.is_cancelled();
        return out;
    }
    let estimator = pooled_panel_estimator();
    let fresh_ids: Vec<u32> = (0..u32::try_from(units).unwrap_or(u32::MAX)).collect();
    let mut rng = ctx.rng.stream(0xC1A5_5E11);
    let mut workspace = EstimationWorkspace::default();
    let mut draws = Vec::with_capacity(replicates as usize);
    let mut attempted = 0usize;
    for _ in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            out.cancelled = true;
            break;
        }
        attempted += 1;
        let drawn: Vec<usize> = (0..units)
            .map(|_| usize::try_from(rng.next_u64() % units as u64).unwrap_or(0))
            .collect();
        let mut mixture = 0.0;
        let mut ok = true;
        for atom in atoms {
            let blocks: Vec<_> = drawn.iter().map(|&k| &atom.units[k]).collect();
            let fit = estimator.stack_panel_units(&blocks, &fresh_ids).and_then(|(prep, _, _)| {
                estimator.inner.fit_point(
                    &prep,
                    &mut workspace,
                    antecedent_core::AssumptionSet::default(),
                )
            });
            match fit {
                Ok(estimate) if estimate.ate.is_finite() => {
                    mixture += atom.weight / total_weight * estimate.ate;
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            draws.push(mixture);
        } else {
            out.failed += 1;
        }
    }
    out.completed = u32::try_from(draws.len()).unwrap_or(u32::MAX);
    if bootstrap_has_enough_successes(draws.len(), attempted) {
        let g = units as f64;
        let sd = antecedent_stats::sample_std(&draws);
        let se = sd
            * (g / (g - 1.0)).sqrt()
            * antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL);
        out.se = se.is_finite().then_some(se);
    }
    out
}

/// Diagnostic naming the analytic panel cluster SE and its degrees of freedom.
fn panel_cluster_se_diagnostic(clusters: PanelClusters, units: usize) -> Diagnostic {
    let label = clusters.label();
    Diagnostic::new(
        "estimate.temporal_effect.panel.cluster_se",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "analytic SE is the Arellano cluster-by-{label} variance over G={units} {label}s \
             (full \
             within-{label} score covariance, G/(G-1)(n-1)/(n-p) factor) with G-1={} degrees of \
             freedom: it is scaled by t_(G-1)/z = {:.4} so estimate ± 1.96·SE is the 95% \
             t_(G-1) interval",
            units.saturating_sub(1),
            antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL),
        ),
    )
}

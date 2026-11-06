// SPDX-License-Identifier: MIT OR Apache-2.0

use super::completion_fit_memo::{CompletionFitMemo, ResponseFitKey};
use super::sequential_validation::SequentialValidation;
use super::{
    AnalyticSeKind, Arc, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte,
    BootstrapAccounting, CausalError, CausalPosterior, CausalQuery, CausalResponse, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DiscoveryEstimationSplit, EffectEstimate,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphEffectDraws, GraphIdentFlag,
    IdentificationBasis, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    ObservationSpec, OverlapPolicy, PanelData, PhysicalExecutionPlan, PredictiveCheckReport,
    PriorSet, RefutationRun, RefuteSuite, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, Strategy, StudyResult, TableView,
    TemporalBackdoorIdentifier, TemporalDag, TemporalEffectQuery, TemporalIndexer,
    TemporalLinearAdjustment, TemporalRefitContext, TemporalResponseEstimator, UNFOLDED_LINEAR_SEM,
    apply_temporal_prior_sensitivity, bayesian_gcomp, bayesian_temporal_gcomp,
    bootstrap_has_enough_successes, completion_fit_seed, effect_from_posterior, elapsed_ns,
    envelope_se_omits_between_atom_variance, envelope_to_identification_result_for,
    identification_status_ok_for_case, identified_set_interval_diagnostics,
    identify_cached_diagnostic, mag_sequential_unevaluable_diagnostic, mix_weighted_analytic_se,
    nan_effect, posterior_identified_set_interval, provenance_ids, push_conflict_diagnostics,
    report_identify_compute, require_identified, resolve_bayesian_prior_with_conflict,
    same_fit_inputs, same_fitted_mechanisms, same_fitted_problem, select_estimand,
    sequential_linear_sem_assumption, stack_panel_tabular, temporal_refuter_average,
    with_conflict_summary,
};
use crate::planner::RouteDefaults;
use antecedent_core::StreamDomain;

mod response;
mod sequential;

use response::panel_unit_average_bootstrap_note;

impl super::Study {
    /// The panel support owner, consulted on the data each execute receives
    /// (a prepared refresh hands in new units).
    pub(super) fn refuse_unsupported_panel(&self, panel: &PanelData) -> Result<(), CausalError> {
        super::super::builder::refuse_unsupported_panel_route(
            &self.query,
            self.graph.class(),
            &self.settings.inference,
            panel,
            self.declared.split.as_ref(),
        )
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

    pub(super) fn execute_pooled_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        clusters: PanelClusters,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unsupported_panel(panel)?;
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) =
            self.identify_pooled_panel(graph, query, Some(physical), ctx)?;
        require_identified(&identification)?;

        let unit_ids = panel_unit_ids(panel);
        let atom = PanelPulseAtom::prepare(
            panel,
            &estimand,
            query,
            &indexer,
            self.declared.split.as_ref(),
            ctx,
            1.0,
        )?;
        let (prep, cluster_ids, _) = atom.stacked(&unit_ids)?;
        let mut diagnostics = vec![pooled_panel_estimand_diagnostic(clusters, unit_ids.len())];

        let design = PooledPanelDesign {
            atom: &atom,
            prep: &prep,
            cluster_ids: &cluster_ids,
            unit_ids: &unit_ids,
        };
        let (estimate, mut posterior, estimate_artifact, estimate_op) = self
            .fit_pooled_panel_estimate(&design, &identification, clusters, &mut diagnostics, ctx)?;

        let temporal_ctx = TemporalRefitContext {
            indexer: &indexer,
            temporal_query: query,
            split: self.declared.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: None,
            panel: Some(panel),
            gaps: false,
        };
        let (mut refutations, na_diagnostics) = self.refute_pooled_panel(
            panel,
            temporal_ctx,
            &estimand,
            &estimate,
            if posterior.is_some() {
                EstimatorId::BayesianTemporalGcomp.as_str()
            } else {
                EstimatorId::TemporalLinearAdjustment.as_str()
            },
            ctx,
        )?;
        diagnostics.extend(na_diagnostics);

        posterior = self.pooled_panel_prior_sensitivity(
            &design,
            identification.status,
            estimate.ate,
            posterior,
            &mut refutations,
            ctx,
        )?;

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let (bootstrap_replicates_ok, cancelled) = (
            estimate.uncertainty.bootstrap.replicates_ok,
            estimate.uncertainty.bootstrap.cancelled,
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    if posterior.is_some() {
                        EstimatorId::BayesianTemporalGcomp
                    } else {
                        EstimatorId::TemporalLinearAdjustment
                    },
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_replicates_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(estimate_artifact, estimate_op)),
                posterior,
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }

    /// Pooled panel estimate under the study's inference mode, with its provenance ids.
    fn fit_pooled_panel_estimate(
        &self,
        design: &PooledPanelDesign<'_>,
        identification: &IdentificationResult,
        clusters: PanelClusters,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Option<CausalPosterior>, &'static str, &'static str), CausalError>
    {
        Ok(match &self.settings.inference {
            InferenceMode::Bayesian(cfg) => {
                let (estimate, posterior) = fit_pooled_panel_bayesian(
                    cfg,
                    design.prep,
                    design.cluster_ids,
                    identification.status,
                    ctx,
                )?;
                (
                    estimate,
                    Some(posterior),
                    "estimate.bayesian_temporal_gcomp.panel",
                    "estimate.bayesian.temporal.gcomp.panel",
                )
            }
            InferenceMode::Frequentist => {
                let estimate = self.fit_pooled_panel_frequentist(
                    design.atom,
                    design.unit_ids,
                    identification,
                    clusters,
                    diagnostics,
                    ctx,
                )?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment.panel",
                    "estimate.temporal.linear.adjustment.panel",
                )
            }
        })
    }

    /// Prior-sensitivity grid of a Bayesian panel under the Full suite (a no-op otherwise).
    fn pooled_panel_prior_sensitivity(
        &self,
        design: &PooledPanelDesign<'_>,
        status: IdentificationStatus,
        ate: f64,
        posterior: Option<CausalPosterior>,
        refutations: &mut Vec<antecedent_validate::RefutationReport>,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        // Panel Bayesian: α-grid under Full when external compose is present (mirror temporal).
        if matches!(self.settings.refute, RefuteSuite::Full) {
            if let (InferenceMode::Bayesian(cfg), Some(post)) =
                (&self.settings.inference, &posterior)
            {
                let mut bprep = BayesianGComputationAte::from_prepared_estimation(design.prep);
                bprep.unit_ids = Some(design.cluster_ids.to_vec());
                return Ok(Some(apply_temporal_prior_sensitivity(
                    cfg,
                    &bprep,
                    status,
                    post,
                    ate,
                    ctx,
                    refutations,
                )?));
            }
        }
        Ok(posterior)
    }

    /// Identifies the panel's pooled Pulse estimand, or reads the horizon entry the
    /// prepared cache or, given the executing `physical` plan, its lowered program
    /// retained.
    pub(super) fn identify_pooled_panel(
        &self,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: Option<&PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<(IdentificationResult, IdentifiedEstimand, TemporalIndexer, bool), CausalError>
    {
        let reused = match physical {
            Some(physical) => self.reuse_temporal_identification(physical),
            None => self.temporal_cache().cloned().map(|cache| (cache, true)),
        };
        if let Some((cache, prepared)) = reused {
            let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                message: "prepared panel identification cache missing query horizon".into(),
            })?;
            Ok((
                entry.identification.clone(),
                entry.estimand.clone(),
                entry.indexer.clone(),
                prepared,
            ))
        } else {
            let id_res = TemporalBackdoorIdentifier::new()
                .with_parent_adjustment_fallback()
                .identify_temporal(graph, query)
                .map_err(CausalError::from)?;
            report_identify_compute(ctx);
            let identification = id_res.result;
            require_identified(&identification)?;
            let estimand = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)?;
            Ok((identification, estimand, id_res.indexer, false))
        }
    }

    /// Pooled frequentist panel fit with its unit bootstrap and cluster-SE disclosures.
    fn fit_pooled_panel_frequentist(
        &self,
        atom: &PanelPulseAtom,
        unit_ids: &[u32],
        identification: &IdentificationResult,
        clusters: PanelClusters,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, CausalError> {
        let mut estimate = atom.fit(unit_ids, ctx, identification.required_assumptions.clone())?;
        estimate.provenance.assumptions.push(panel_estimand_assumption(clusters, unit_ids.len()));
        let boot = panel_unit_bootstrap(
            std::slice::from_ref(atom),
            self.settings.bootstrap_replicates,
            ctx,
        );
        boot.attach(&mut estimate, self.settings.bootstrap_replicates);
        diagnostics.push(panel_cluster_se_diagnostic(clusters, unit_ids.len()));
        diagnostics.extend(boot.diagnostic(clusters, self.settings.bootstrap_replicates));
        Ok(estimate)
    }

    /// Runs the refutation suite on the stacked panel table.
    fn refute_pooled_panel(
        &self,
        panel: &PanelData,
        temporal: TemporalRefitContext<'_>,
        estimand: &IdentifiedEstimand,
        estimate: &EffectEstimate,
        estimator: &str,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = temporal_refuter_average(temporal.temporal_query);
        let mut refute_ws = EstimationWorkspace::default();
        RefutationRun {
            data: &stacked,
            estimand,
            query: &ate_q,
            estimate,
            suite: self.settings.refute,
            estimator,
            custom: &self.declared.custom_validators,
            temporal: Some(temporal),
        }
        .reports(&mut refute_ws, None, ctx)
    }

    /// Identify the class once for a panel class effect (or read the prepared
    /// certificate) and refuse an envelope without identified mass.
    fn panel_class_bundle(
        &self,
        physical: &PhysicalExecutionPlan,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<
        (IdentifierId, Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, bool),
        CausalError,
    > {
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let (bundle, identify_cached) =
            if let Some(reused) = self.reuse_temporal_class_identification(physical) {
                reused
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, query, ctx)?, false)
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
        self.refuse_unsupported_panel(panel)?;
        if query.is_multi_step_sustained() {
            return self.execute_panel_class_sequential(panel, query, physical, ctx);
        }
        if self.settings.inference.is_bayesian() {
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
        let unit_ids = panel_unit_ids(panel);
        let PanelClassMixture {
            ate,
            se_analytic,
            estimand,
            assumptions,
            fitted,
            refute_atoms,
            atom_values,
        } = self
            .fit_panel_class_atoms(panel, query, &bundle.envelope, &unit_ids, ctx)?
            .into_mixture()?;
        let boot = panel_unit_bootstrap(&fitted, self.settings.bootstrap_replicates, ctx);
        self.push_panel_class_disclosures(
            &mut diagnostics,
            &boot,
            fitted.len(),
            panel.unit_count(),
        );
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
            BootstrapAccounting::cancelled(ctx.cancellation.is_cancelled()),
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        );
        boot.attach(&mut estimate, self.settings.bootstrap_replicates);
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let cancelled = estimate.uncertainty.bootstrap.cancelled;
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
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, EstimatorId::TemporalLinearAdjustment),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: (*bundle.envelope).clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                bootstrap_replicates_requested: (self.settings.bootstrap_replicates > 0)
                    .then_some(Some(self.settings.bootstrap_replicates)),
                ..Default::default()
            }),
        ))
    }

    /// Fits the pooled Pulse design of every identified completion in the class envelope.
    fn fit_panel_class_atoms(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        envelope: &antecedent_identify::TemporalClassEnvelope,
        unit_ids: &[u32],
        ctx: &ExecutionContext,
    ) -> Result<PanelClassFits, CausalError> {
        let bundle_envelope = &envelope.envelope;
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut fitted = Vec::new();
        let mut refute_atoms = Vec::new();
        let mut atom_values: Vec<Option<f64>> = vec![None; bundle_envelope.cases.len()];
        for (i, (case, indexer)) in
            bundle_envelope.cases.iter().zip(envelope.indexers.iter()).enumerate()
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
                self.declared.split.as_ref(),
                ctx,
                w,
            )?;
            let mut estimate = atom.fit(unit_ids, ctx, case.result.required_assumptions.clone())?;
            estimate
                .provenance
                .assumptions
                .push(panel_estimand_assumption(PanelClusters::Units, unit_ids.len()));
            refute_atoms.push(PanelRefuteAtom {
                key: case.graph.fingerprint(),
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                estimate: estimate.clone(),
            });
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic()));
            total_w += w;
            atom_values[i] = Some(estimate.ate);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.provenance.assumptions.clone();
            }
            fitted.push(atom);
        }
        Ok(PanelClassFits {
            weighted_ate,
            se_items,
            total_w,
            primary_estimand,
            assumptions,
            fitted,
            refute_atoms,
            atom_values,
        })
    }

    /// Discloses the SE route and completion count of a panel class Pulse.
    fn push_panel_class_disclosures(
        &self,
        diagnostics: &mut Vec<Diagnostic>,
        boot: &PanelUnitBootstrap,
        fitted: usize,
        units: usize,
    ) {
        if boot.se.is_none() && fitted > 1 {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        if fitted == 1 {
            diagnostics.push(panel_cluster_se_diagnostic(PanelClusters::Units, units));
        }
        diagnostics
            .extend(boot.diagnostic(PanelClusters::Units, self.settings.bootstrap_replicates));
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_effect.panel.class.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel class Pulse fits {fitted} identified completions on {units} units; \
                 mixture weights are completion masses, not stacked-row circular blocks"
            ),
        ));
    }

    /// Envelope and estimand disclosures that open a Bayesian panel class effect.
    fn panel_class_bayesian_disclosures(
        &self,
        envelope: &antecedent_identify::IdentificationEnvelope<
            antecedent_identify::TemporalCompletionGraph,
        >,
        no_class_prior: bool,
        panel_units: usize,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        if no_class_prior {
            diagnostics.push(super::temporal_path::enumeration_not_probability_diagnostic());
        }
        diagnostics.push(pooled_panel_estimand_diagnostic(PanelClusters::Units, panel_units));
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_effect.panel.class.bayesian_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each completion is the panel hierarchical GLS posterior (unit random intercepts) \
             of its pooled design; completion posteriors are mixed only under a caller class \
             prior",
        ));
        diagnostics
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
        let InferenceMode::Bayesian(cfg) = &self.settings.inference else {
            return Err(CausalError::unsupported_inference(
                "panel class Bayesian execute requires Bayesian inference",
            ));
        };
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.panel_class_bundle(physical, query, ctx)?;
        let envelope = &bundle.envelope.envelope;
        let class_masses = self
            .declared
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        let mut diagnostics = self.panel_class_bayesian_disclosures(
            envelope,
            class_masses.is_none(),
            panel.unit_count(),
        );
        let mut state = PanelClassPosteriors::new(
            cfg,
            panel,
            query,
            ctx,
            self.declared.split.as_ref(),
            envelope.cases.len(),
        );
        state.fit_cases(&bundle.envelope, class_masses.as_deref(), &mut diagnostics)?;
        let PanelClassBayesianMix { estimate, posterior, structural, estimand, refute_atoms } =
            state.mix(
                envelope,
                class_masses.as_deref(),
                self.settings.bootstrap_replicates,
                panel.total_rows(),
                &mut diagnostics,
            )?;
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
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, EstimatorId::BayesianTemporalGcomp),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: (*bundle.envelope).clone(),
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
            }),
        ))
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
        if matches!(self.settings.refute, RefuteSuite::None)
            && self.declared.custom_validators.is_empty()
        {
            return Ok((reports, diagnostics));
        }
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = temporal_refuter_average(query);
        let mut workspace = EstimationWorkspace::default();
        for atom in atoms {
            let temporal_ctx = TemporalRefitContext {
                indexer: &atom.indexer,
                temporal_query: query,
                split: self.declared.split.as_ref(),
                kernel_policy: &ctx.kernel_policy,
                time_index: None,
                panel: Some(panel),
                gaps: false,
            };
            let (mut local, local_diagnostics) = RefutationRun {
                data: &stacked,
                estimand: &atom.estimand,
                query: &ate_q,
                estimate: &atom.estimate,
                suite: self.settings.refute,
                estimator,
                custom: &self.declared.custom_validators,
                temporal: Some(temporal_ctx),
            }
            .reports(&mut workspace, None, ctx)?;
            for report in &mut local {
                report.refuter = Arc::from(format!("completion.{}.{}", atom.key, report.refuter));
            }
            reports.extend(local);
            diagnostics.extend(local_diagnostics);
        }
        Ok((reports, diagnostics))
    }
}
/// Bayesian pooled panel posterior with the resolved prior and its conflict summary.
fn fit_pooled_panel_bayesian(
    cfg: &BayesianConfig,
    prep: &antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: &[u32],
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, CausalPosterior), CausalError> {
    let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
    let mut bprep = BayesianGComputationAte::from_prepared_estimation(prep);
    bprep.unit_ids = Some(cluster_ids.to_vec());
    let (resolved_prior, conflict_summary) =
        resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
    bayes.inner.prior = resolved_prior;
    let mut ws = BayesianGCompWorkspace::default();
    let mut posterior = bayes.fit(&bprep, status, &mut ws, ctx).map_err(CausalError::from)?;
    if let Some(summary) = conflict_summary {
        posterior = with_conflict_summary(posterior, summary);
    }
    let estimate = effect_from_posterior(&posterior)?;
    Ok((estimate, posterior))
}

/// The mass-weighted panel class effect with the completions behind it.
struct PanelClassMixture {
    ate: f64,
    se_analytic: f64,
    estimand: IdentifiedEstimand,
    assumptions: antecedent_core::AssumptionSet,
    fitted: Vec<PanelPulseAtom>,
    refute_atoms: Vec<PanelRefuteAtom>,
    atom_values: Vec<Option<f64>>,
}

/// Fitted completions of a panel class Pulse with the sums its mixture reads.
struct PanelClassFits {
    weighted_ate: f64,
    se_items: Vec<(f64, f64)>,
    total_w: f64,
    primary_estimand: Option<IdentifiedEstimand>,
    assumptions: antecedent_core::AssumptionSet,
    fitted: Vec<PanelPulseAtom>,
    refute_atoms: Vec<PanelRefuteAtom>,
    atom_values: Vec<Option<f64>>,
}

/// Running per-completion fits of a Bayesian panel class effect.
struct PanelClassPosteriors<'a> {
    cfg: &'a BayesianConfig,
    panel: &'a PanelData,
    query: &'a TemporalEffectQuery,
    ctx: &'a ExecutionContext,
    split: Option<&'a DiscoveryEstimationSplit>,
    unit_ids: Vec<u32>,
    bayes: antecedent_estimate::BayesianTemporalGcomp,
    ws: BayesianGCompWorkspace,
    weights: Vec<f64>,
    flags: Vec<GraphIdentFlag>,
    keys: Vec<u64>,
    per_graph: Vec<GraphEffectDraws>,
    fits: Vec<(
        u64,
        antecedent_estimate::PreparedBayesianProblem,
        Option<PriorSet>,
        u64,
        IdentificationStatus,
    )>,
    posteriors: Vec<(u64, CausalPosterior)>,
    case_means: Vec<Option<f64>>,
    primary_estimand: Option<IdentifiedEstimand>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
    refute_atoms: Vec<PanelRefuteAtom>,
}

impl<'a> PanelClassPosteriors<'a> {
    /// Empty running fits for an envelope of `n_cases` completions.
    fn new(
        cfg: &'a BayesianConfig,
        panel: &'a PanelData,
        query: &'a TemporalEffectQuery,
        ctx: &'a ExecutionContext,
        split: Option<&'a DiscoveryEstimationSplit>,
        n_cases: usize,
    ) -> Self {
        Self {
            cfg,
            panel,
            query,
            ctx,
            split,
            unit_ids: panel_unit_ids(panel),
            bayes: bayesian_temporal_gcomp(cfg, ctx),
            ws: BayesianGCompWorkspace::default(),
            weights: Vec::new(),
            flags: Vec::new(),
            keys: Vec::new(),
            per_graph: Vec::new(),
            fits: Vec::new(),
            posteriors: Vec::new(),
            case_means: vec![None; n_cases],
            primary_estimand: None,
            envelope_conflict: None,
            refute_atoms: Vec::new(),
        }
    }
}

/// The Bayesian panel class effect with its structural mixture and refutation atoms.
struct PanelClassBayesianMix {
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    structural: crate::result::StructuralResponseMixture,
    estimand: IdentifiedEstimand,
    refute_atoms: Vec<PanelRefuteAtom>,
}

impl PanelClassPosteriors<'_> {
    /// Fits every identified completion of the envelope, recording masses and flags in order.
    fn fit_cases(
        &mut self,
        envelope: &antecedent_identify::TemporalClassEnvelope,
        class_masses: Option<&[f64]>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(), CausalError> {
        for (i, (case, indexer)) in
            envelope.envelope.cases.iter().zip(envelope.indexers.iter()).enumerate()
        {
            let key = case.graph.fingerprint();
            self.keys.push(key);
            self.weights.push(class_masses.map_or(case.weight.0, |masses| masses[i]));
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                self.flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            self.flags.push(GraphIdentFlag::Identified);
            self.fit_case(i, key, &case.result, indexer, diagnostics)?;
        }
        Ok(())
    }

    /// Mixes the completion posteriors into the class effect and its structural mixture.
    fn mix(
        self,
        envelope: &antecedent_identify::IdentificationEnvelope<
            antecedent_identify::TemporalCompletionGraph,
        >,
        class_masses: Option<&[f64]>,
        bootstrap_replicates: u32,
        total_rows: usize,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<PanelClassBayesianMix, CausalError> {
        let PanelClassPosteriors {
            weights,
            flags,
            keys,
            per_graph,
            fits,
            posteriors,
            case_means,
            primary_estimand,
            envelope_conflict,
            refute_atoms,
            ..
        } = self;
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
                diagnostics,
            )?;
        let structural = panel_class_structural_mixture(
            envelope,
            weight_basis,
            class_masses,
            &case_means,
            &posteriors,
            total_rows,
            diagnostics,
        );
        if bootstrap_replicates > 0 {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_effect.panel.class.bootstrap_not_used",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian panel class effects carry posterior uncertainty; requested bootstrap \
                 replicates are not used",
            ));
        }
        Ok(PanelClassBayesianMix { estimate, posterior, structural, estimand, refute_atoms })
    }
}

impl PanelClassPosteriors<'_> {
    /// Fits one identified completion's panel posterior, reusing an earlier identical fit.
    fn fit_case(
        &mut self,
        i: usize,
        key: u64,
        result: &IdentificationResult,
        indexer: &TemporalIndexer,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(), CausalError> {
        let estimand = select_estimand(result, EstimatorId::TemporalLinearAdjustment)?;
        let atom = PanelPulseAtom::prepare(
            self.panel,
            &estimand,
            self.query,
            indexer,
            self.split,
            self.ctx,
            self.weights[i],
        )?;
        let (prep, cluster_ids, _) = atom.stacked(&self.unit_ids)?;
        let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
        bprep.unit_ids = Some(cluster_ids);
        let (resolved, conflict) =
            resolve_bayesian_prior_with_conflict(self.cfg, &bprep, Some(self.ctx))?;
        // Each completion draws from its own seed stream; completions fitting the
        // same problem under the same prior share a seed, so one model carries one
        // posterior.
        let seed = self
            .fits
            .iter()
            .find(|(_, fitted, prior, _, _)| {
                *prior == resolved && same_fitted_problem(fitted, &bprep)
            })
            .map_or_else(|| completion_fit_seed(self.ctx, key), |(_, _, _, seed, _)| *seed);
        self.bayes.inner.prior.clone_from(&resolved);
        self.bayes.inner.seed = seed;
        // An earlier completion that fitted every input of this fit (the
        // problem down to its unit ids and labels, the resolved prior, the
        // seed it shares, and the identification status) already holds
        // this posterior, conflict summary included: the fit is
        // deterministic in exactly those inputs.
        let same_fit = self.fits.iter().position(|(_, fitted, prior, fitted_seed, status)| {
            *prior == resolved
                && *fitted_seed == seed
                && *status == result.status
                && same_fit_inputs(fitted, &bprep)
        });
        let posterior = if let Some(at) = same_fit {
            if let Some(summary) = conflict.as_ref() {
                push_conflict_diagnostics(diagnostics, summary);
            }
            self.posteriors[at].1.clone()
        } else {
            let mut posterior = self
                .bayes
                .fit(&bprep, result.status, &mut self.ws, self.ctx)
                .map_err(CausalError::from)?;
            if let Some(summary) = conflict.as_ref() {
                push_conflict_diagnostics(diagnostics, summary);
                posterior = with_conflict_summary(posterior, summary.clone());
            }
            posterior
        };
        self.envelope_conflict = self.envelope_conflict.take().or(conflict);
        let mut atom_estimate = effect_from_posterior(&posterior)?;
        atom_estimate.provenance.assumptions = result.required_assumptions.clone();
        self.case_means[i] = Some(atom_estimate.ate);
        self.refute_atoms.push(PanelRefuteAtom {
            key,
            estimand: estimand.clone(),
            indexer: indexer.clone(),
            estimate: atom_estimate,
        });
        if let Some(draws) = posterior
            .effect_column()
            .and_then(|col| posterior.draws.column(col).ok().map(|d| Arc::from(d.to_vec())))
        {
            self.per_graph.push(GraphEffectDraws { graph_key: key, effect_draws: draws });
        }
        self.primary_estimand.get_or_insert(estimand);
        self.fits.push((key, bprep, resolved, seed, result.status));
        self.posteriors.push((key, posterior));

        Ok(())
    }
}

/// Structural mixture of a Bayesian panel class effect, with the identified-set interval attached.
fn panel_class_structural_mixture(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
    weight_basis: crate::result::StructuralWeightBasis,
    class_masses: Option<&[f64]>,
    case_means: &[Option<f64>],
    posteriors: &[(u64, CausalPosterior)],
    total_rows: usize,
    diagnostics: &mut Vec<Diagnostic>,
) -> crate::result::StructuralResponseMixture {
    let mut structural = super::temporal_path::temporal_class_structural_mixture(
        envelope,
        weight_basis,
        class_masses,
        case_means,
    );
    structural.identified_set_interval = posterior_identified_set_interval(
        posteriors.iter().map(|(_, posterior)| posterior),
        total_rows,
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
    structural
}

impl PanelClassFits {
    /// Mass-weighted effect and analytic SE, refusing an envelope with no estimable completion.
    fn into_mixture(self) -> Result<PanelClassMixture, CausalError> {
        let PanelClassFits {
            weighted_ate,
            se_items,
            total_w,
            primary_estimand,
            assumptions,
            fitted,
            refute_atoms,
            atom_values,
        } = self;
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
        Ok(PanelClassMixture {
            ate,
            se_analytic,
            estimand,
            assumptions,
            fitted,
            refute_atoms,
            atom_values,
        })
    }
}

/// Prepared pooled-panel design the estimate and sensitivity stages read.
struct PooledPanelDesign<'a> {
    atom: &'a PanelPulseAtom,
    prep: &'a antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: &'a [u32],
    unit_ids: &'a [u32],
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "RESPONSE_SHORT_SERIES_ROWS is the positive integer-valued constant 15.0, so the cast is exact"
)]
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
pub(super) enum PanelClusters {
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
pub(super) fn pooled_panel_estimator() -> TemporalLinearAdjustment {
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.inference.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    estimator
}

/// One pooled Pulse design (the DAG, or one identified completion): every unit's own
/// lag-aligned block, prepared once and stacked per fit.
pub(super) struct PanelPulseAtom {
    weight: f64,
    units: Vec<antecedent_estimate::PreparedEstimationProblem>,
}

impl PanelPulseAtom {
    pub(super) fn prepare(
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
    pub(super) fn stacked(
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
    estimator.inner.inference.cluster_ids = Some(cluster_ids);
    estimator.inner.inference.panel_times = Some(panel_times);
    estimator.inner.inference.se_kind = AnalyticSeKind::PanelClusterHac { lag: 0 };
    let mut workspace = EstimationWorkspace::default();
    let mut estimate =
        estimator.fit(prep, &mut workspace, ctx, assumptions).map_err(CausalError::from)?;
    estimate.uncertainty.analytic.se *=
        antecedent_stats::few_cluster_t_ratio(clusters, PANEL_INTERVAL_LEVEL);
    Ok(estimate)
}

fn distinct_clusters(cluster_ids: &[u32]) -> usize {
    cluster_ids.iter().collect::<std::collections::BTreeSet<_>>().len()
}

/// Panel unit ids in panel order.
pub(super) fn panel_unit_ids(panel: &PanelData) -> Vec<u32> {
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
        estimate.uncertainty.bootstrap.se = self.se;
        estimate.uncertainty.bootstrap.replicates_ok = Some(self.completed);
        estimate.uncertainty.bootstrap.replicates_failed = Some(self.failed);
        estimate.uncertainty.bootstrap.cancelled = self.cancelled;
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
/// withheld unless at least [`super::PERCENTILE_95_BAND_MIN_SUCCESSES`] replicates
/// succeeded, successes outnumber failures, and the run was not cancelled.
/// Cancellation is not adaptive early-stop: a cancel after the success floor still
/// withholds the SE.
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
    let mut rng = ctx.rng.stream_for(StreamDomain::Panel, 0xC1A5_5E11);
    let mut workspace = EstimationWorkspace::default();
    // One scalar per replicate. A pool that cannot fund them withholds the bootstrap SE, which
    // is how this helper reports any bootstrap that cannot complete.
    let Ok(_admitted) = ctx.acquire_admission_product(0, u64::from(replicates), 8) else {
        out.cancelled = ctx.cancellation.is_cancelled();
        return out;
    };
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
    if !out.cancelled && bootstrap_has_enough_successes(draws.len(), attempted) {
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

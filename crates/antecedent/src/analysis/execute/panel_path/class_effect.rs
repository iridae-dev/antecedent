//! Panel class-effect fitting, mixing, and refutation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte, BootstrapAccounting,
    CausalError, CausalPosterior, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    DiscoveryEstimationSplit, EffectEstimate, EstimationWorkspace, EstimatorId, ExecutionContext,
    GraphEffectDraws, GraphIdentFlag, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceMode, Instant, OverlapPolicy, PanelClusters, PanelData, PanelPulseAtom,
    PanelUnitBootstrap, PhysicalExecutionPlan, PriorSet, RefutationRun, RefuteSuite, RouteDefaults,
    Strategy, StudyResult, TemporalEffectQuery, TemporalIndexer, TemporalRefitContext,
    bayesian_temporal_gcomp, completion_fit_seed, effect_from_posterior, elapsed_ns,
    envelope_se_omits_between_atom_variance, envelope_to_identification_result_for,
    identification_status_ok_for_case, identified_set_interval_diagnostics,
    identify_cached_diagnostic, mix_weighted_analytic_se, panel_cluster_se_diagnostic,
    panel_estimand_assumption, panel_unit_bootstrap, panel_unit_ids,
    pooled_panel_estimand_diagnostic, posterior_identified_set_interval, provenance_ids,
    push_conflict_diagnostics, report_identify_compute, resolve_bayesian_prior_with_conflict,
    same_fit_inputs, same_fitted_problem, select_estimand, stack_panel_tabular,
    temporal_refuter_average, with_conflict_summary,
};

impl super::super::Study {
    /// Identify the class once for a panel class effect (or read the prepared
    /// certificate) and refuse an envelope without identified mass.
    pub(super) fn panel_class_bundle(
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

    pub(in crate::analysis::execute) fn execute_panel_class(
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
            super::super::temporal_path::temporal_class_envelope_diagnostic(
                envelope,
                self.graph.class(),
            ),
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
        let structural = super::super::temporal_path::temporal_class_structural_mixture(
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
        let mut diagnostics =
            vec![super::super::temporal_path::temporal_class_envelope_diagnostic(
                envelope,
                self.graph.class(),
            )];
        if no_class_prior {
            diagnostics.push(super::super::temporal_path::enumeration_not_probability_diagnostic());
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
            super::super::temporal_path::mix_class_effect_posteriors(
                super::super::temporal_path::ClassPosteriorMix {
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
    let mut structural = super::super::temporal_path::temporal_class_structural_mixture(
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
/// One fitted completion of a panel class Pulse, kept for its refuters.
struct PanelRefuteAtom {
    key: u64,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    estimate: EffectEstimate,
}

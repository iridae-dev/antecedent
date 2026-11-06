//! Graph-posterior response execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, CausalError, ContinuousResponseEstimator, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimationWorkspace, EstimatorId, ExecutionContext,
    GraphPosterior, GraphPosteriorPooled, GraphPosteriorResponseFits, GraphPosteriorResponseRun,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant,
    OverlapPolicy, PhysicalExecutionPlan, PolicyResponseMix, PosteriorLevelChecks,
    PosteriorMassSummary, RefutationRun, RefuteSuite, ResponseFunctional, ResponseQuery,
    ResponseUncertainty, ResponseValue, Strategy, StudyResult, TableView, TabularData, elapsed_ns,
    graph_posterior_response_supported, graph_posterior_structural_atoms,
    identify_cached_diagnostic, interactive_subsample_graphs_accounted, mix_responses_under_policy,
    mix_support_reports, mixed_response_scalar_se, resolve_structural_aggregation,
    resolved_bayesian_response_estimator, response_identified_value, response_primary_pair,
    response_scalar_summary,
};

impl super::super::Study {
    #[allow(
        clippy::too_many_lines,
        reason = "one graph-posterior response: every atom's fit, the mixture and the identified finish share the frozen posterior"
    )]
    pub(in crate::analysis::execute) fn execute_graph_posterior_response(
        &self,
        data: &TabularData,
        graph_posterior: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        graph_posterior_response_supported(query)?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let identification_query = AverageEffectQuery::binary_ate(treatment, outcome);
        let (identified, identify_cached) = if let Some(cache) = self.graph_posterior_cache() {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    graph_posterior,
                    &identification_query,
                    ctx,
                )?),
                false,
            )
        };
        let run = GraphPosteriorResponseRun { data, query, treatment, outcome, ctx };
        let (mut fits, identification) =
            self.fit_graph_posterior_response_atoms(&run, &identified)?;
        // The primary atom's own identification, before pooling may mark the
        // mixture graph-dependent: the Bayesian refuter state refits that atom.
        let primary_status = identification.status;
        let primary_assumptions = identification.required_assumptions.clone();
        let pooled = self.pool_graph_posterior_response(&run, &fits, identification)?;
        let estimator_id = if self.settings.inference.is_bayesian() {
            EstimatorId::ResponseBayesian
        } else {
            EstimatorId::default_for_response(&query.functional)
        };
        let mut diagnostics = self.graph_posterior_response_diagnostics(&fits, &pooled);
        let (mut refutations, refute_diags) = self.refute_graph_posterior_response(
            &run,
            estimator_id,
            &fits,
            &pooled,
            &mut diagnostics,
        )?;
        diagnostics.extend(refute_diags);
        let (predictive_checks, posterior_reports) = self.bayesian_graph_posterior_level_state(
            &run,
            &fits,
            (primary_status, &primary_assumptions),
            &mut diagnostics,
        )?;
        refutations.extend(posterior_reports);
        diagnostics.extend(std::mem::take(&mut fits.subsample_notes));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let GraphPosteriorPooled {
            identification,
            conditional,
            response,
            estimate,
            identified_mass,
            total_mass,
            structural_set,
            aggregation_policy,
            ..
        } = pooled;
        let GraphPosteriorResponseFits {
            atoms,
            estimand,
            unidentified_mass,
            failed_mass,
            subsampled_out_mass,
            ..
        } = fits;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::BackdoorAdjustment, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                response: Some(response),
                structural_response: Some(crate::result::StructuralResponseMixture {
                    weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
                    atoms,
                    identified_mass: identified_mass / total_mass,
                    unidentified_mass: unidentified_mass / total_mass,
                    unevaluable_mass: failed_mass / total_mass,
                    subsampled_out_mass: subsampled_out_mass / total_mass,
                    identified_set: structural_set,
                    identified_set_interval: None,
                    conditional_on_identified: conditional,
                    full_mass_scope: true,
                    truncated_atoms: 0,
                    aggregation_policy: Some(aggregation_policy),
                }),
                diagnostics: Some(diagnostics),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Bayesian additions to the plugin-level graph-posterior refutation: the predictive
    /// checks and, under `full`, level-native prior sensitivity and MCMC diagnostics,
    /// evaluated on the primary identified atom (the atom whose estimand the plugin-level
    /// suite reads). Returns the predictive checks and the extra refutation reports;
    /// Frequentist studies and `none` return nothing.
    fn bayesian_graph_posterior_level_state(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        fits: &GraphPosteriorResponseFits,
        primary: (IdentificationStatus, &antecedent_core::AssumptionSet),
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<PosteriorLevelChecks, CausalError> {
        let InferenceMode::Bayesian(config) = &self.settings.inference else {
            return Ok((Vec::new(), Vec::new()));
        };
        if !matches!(self.settings.refute, RefuteSuite::Cheap | RefuteSuite::Full)
            || !matches!(run.query.functional, ResponseFunctional::InterventionResponse { .. })
        {
            return Ok((Vec::new(), Vec::new()));
        }
        let Some((_, _, primary_response)) = fits.weighted.first() else {
            return Ok((Vec::new(), Vec::new()));
        };
        let (level, level_sd) = response_scalar_summary(primary_response);
        let mut fitter =
            ContinuousResponseEstimator::new(Arc::clone(&fits.estimand.adjustment_set));
        fitter.options = fits.options.clone();
        let estimator =
            resolved_bayesian_response_estimator(config, &fitter, run.data, run.query, run.ctx)?;
        let (status, assumptions) = primary;
        let state = super::super::bayesian_level_refutation::refute_bayesian_intervention_level(
            &super::super::bayesian_level_refutation::BayesianLevelRefutationInput {
                data: run.data,
                query: run.query,
                estimand: &fits.estimand,
                fitter: &fitter,
                estimator: &estimator,
                status,
                assumptions,
                level,
                level_sd,
                treatment: run.treatment,
                outcome: run.outcome,
                suite: self.settings.refute,
                latency: self.latency_mode,
                // The plugin-level suite already ran on the pooled level; only the
                // posterior checks are added here.
                plugin_level: false,
                summarize: response_scalar_summary,
                target_label: "intervention-level",
            },
            run.ctx,
        )?;
        diagnostics.extend(state.diagnostics);
        diagnostics.push(Diagnostic::new(
            "refute.response.graph_posterior.bayesian_primary_atom",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the predictive checks and prior sensitivity of this Bayesian graph-posterior \
             intervention level are evaluated on the primary identified atom, the atom whose \
             estimand the plugin-level suite reads; they are not mixed across atoms",
        ));
        Ok((state.predictive_checks, state.refutations))
    }

    /// Fits each identified atom of the graph posterior and collects the weighted responses.
    #[allow(
        clippy::too_many_lines,
        reason = "the compiled route keeps its bounded response execution together"
    )]
    fn fit_graph_posterior_response_atoms(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        identified: &crate::analysis::prepared::CachedGraphPosteriorIdentification,
    ) -> Result<(GraphPosteriorResponseFits, IdentificationResult), CausalError> {
        let GraphPosteriorResponseRun { data, query, ctx, .. } = *run;
        // Same interactive budget as the graph-posterior ATE path: identified
        // atoms left out of the stratified subsample are never fit and never
        // renormalized away; their mass is reported as not evaluated.
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            self.latency_mode,
            identified.graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let mut weighted = Vec::new();
        let mut contributing = Vec::new();
        let mut atoms = graph_posterior_structural_atoms(&identified.graphs);
        let mut primary = None;
        let mut failed_mass = 0.0;
        // Atom influence columns keep their own complete-case row indices; they
        // are aligned on the shared data rows before mixing.
        let mut atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)> = Vec::new();
        let options = self.response_options.clone().unwrap_or_default();
        let key_weights = IdentifiedKeyWeights::new(&graphs);
        let work: Vec<_> =
            identified.atoms.iter().filter(|atom| key_weights.get(atom.key) > 0.0).collect();
        let fitted = ctx.map_indexed(work.len(), |i, inner| {
            let atom = work[i];
            let mut estimator =
                ContinuousResponseEstimator::new(Arc::clone(&atom.estimand.adjustment_set));
            estimator.options = options.clone();
            let scored = if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
                let bayes =
                    resolved_bayesian_response_estimator(cfg, &estimator, data, query, inner)?;
                estimator
                    .estimate_bayesian(
                        data,
                        query,
                        atom.identification.status,
                        atom.identification.required_assumptions.clone(),
                        &bayes,
                        inner,
                    )
                    .map(|response| (response, None))
            } else {
                estimator.estimate_identified_scored(
                    data,
                    query,
                    atom.identification.status,
                    atom.identification.required_assumptions.clone(),
                )
            };
            Ok::<_, CausalError>((atom, scored))
        })?;
        for (atom, response) in fitted {
            let weight = key_weights.get(atom.key);
            let response = match response {
                Err(err @ antecedent_estimate::EstimationError::Refused { .. }) => {
                    return Err(err.into());
                }
                Err(err) => {
                    // Cancellation and resource refusals end the route; only an ordinary
                    // fit failure leaves the atom's weight unevaluable.
                    if matches!(err, antecedent_estimate::EstimationError::Capacity(_)) {
                        return Err(err.into());
                    }
                    let err = CausalError::from(err);
                    if matches!(err, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
                        return Err(err);
                    }
                    failed_mass += weight;
                    for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key)
                    {
                        slot.status = atom.identification.status;
                    }
                    continue;
                }
                Ok((response, scores)) => {
                    if let Some(scores) = scores.filter(|s| !s.columns.is_empty()) {
                        atom_scores.push((weight, scores));
                    }
                    response
                }
            };
            let value =
                response_identified_value(&response).ok_or_else(|| CausalError::Compile {
                    message: "identified graph-posterior response atom had no numerical value"
                        .into(),
                })?;
            for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                slot.status = atom.identification.status;
                slot.value = Some(value.clone());
            }
            if primary.is_none() {
                primary = Some((atom.estimand.clone(), atom.identification.clone()));
            }
            contributing.push((
                atom.estimand.clone(),
                atom.identification.status == IdentificationStatus::PartiallyIdentified
                    || response.identification_status == IdentificationStatus::PartiallyIdentified,
            ));
            weighted.push((atom.key, weight, response));
        }
        let (estimand, identification) = primary.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior response has no evaluable identified atom".into(),
        })?;
        let unidentified_mass = identified.graphs.unidentified_mass();
        Ok((
            GraphPosteriorResponseFits {
                graphs,
                unidentified_mass,
                subsampled_out_mass: subsample_drop.mass,
                subsample_notes,
                atoms,
                weighted,
                contributing,
                estimand,
                failed_mass,
                atom_scores,
                options,
            },
            identification,
        ))
    }

    /// Pools the atom responses into the published response, estimate and mass accounting.
    #[allow(
        clippy::too_many_lines,
        reason = "the compiled route keeps its bounded response execution together"
    )]
    fn pool_graph_posterior_response(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        fits: &GraphPosteriorResponseFits,
        mut identification: IdentificationResult,
    ) -> Result<GraphPosteriorPooled, CausalError> {
        let GraphPosteriorResponseRun { data, query, .. } = *run;
        let GraphPosteriorResponseFits { graphs, weighted, atom_scores, options, .. } = fits;
        let failed_mass = fits.failed_mass;
        let identified_mass = weighted.iter().map(|(_, weight, _)| weight).sum::<f64>();
        let total_mass = graphs.total_weight();
        // The subsampled ensemble flags dropped atoms Unidentified; the full
        // ensemble's unidentified mass is the structural one.
        let unidentified_mass = fits.unidentified_mass;
        let subsampled_out_mass = fits.subsampled_out_mass;
        let conditional_values = weighted
            .iter()
            .filter_map(|(_, weight, response)| {
                response_identified_value(response).map(|value| (*weight, value))
            })
            .collect::<Vec<_>>();
        let first = &weighted[0].2;
        // Unevaluable and subsampled-out mass are as incomplete as unidentified
        // mass: the published value covers only the evaluated atoms.
        let aggregation_policy = resolve_structural_aggregation(
            &fits.contributing.iter().map(|(estimand, _)| estimand).collect::<Vec<_>>(),
            fits.contributing.iter().any(|(_, partial)| *partial),
        );
        let PolicyResponseMix { status, estimate, conditional, structural_set } =
            mix_responses_under_policy(
                weighted,
                &conditional_values,
                aggregation_policy,
                (
                    identification.status,
                    PosteriorMassSummary {
                        unidentified: unidentified_mass,
                        failed: failed_mass,
                        subsampled: subsampled_out_mass,
                        ..PosteriorMassSummary::default()
                    },
                ),
            )?;
        identification.status = status;
        // Only a scalar aggregate takes the joint-IF SE: for a curve the first
        // score column is the influence of the first grid point alone, and atoms that
        // do not share one point-identified estimand have no aggregate to bound.
        let mixed_if_se = (weighted.len() > 1
            && self.settings.inference.is_frequentist()
            && matches!(conditional, Some(ResponseValue::Scalar(_)))
            && atom_scores.len() == weighted.len())
        .then(|| mixed_response_scalar_se(atom_scores, data.row_count()))
        .flatten();
        let uncertainty = if weighted.len() == 1 {
            first.uncertainty.clone()
        } else if let (Some(se), Some(ResponseValue::Scalar(value))) = (mixed_if_se, &conditional) {
            let level = options.confidence_level;
            let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
            ResponseUncertainty::Scalar {
                standard_error: se,
                level,
                lower: value - z * se,
                upper: value + z * se,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            }
        } else {
            ResponseUncertainty::None
        };
        let response = antecedent_core::CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate,
            uncertainty,
            support: mix_support_reports(
                &weighted.iter().map(|(_, _, response)| &response.support).collect::<Vec<_>>(),
            ),
            assumptions: first.assumptions.clone(),
            provenance_id: Arc::from("estimate.response.graph_posterior"),
            horizon_identification: None,
            interaction_structurally_zero: first.interaction_structurally_zero,
        };
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        Ok(GraphPosteriorPooled {
            identification,
            conditional,
            mixed_if_se,
            response,
            estimate,
            scalar,
            standard_error,
            identified_mass,
            total_mass,
            structural_set,
            aggregation_policy,
        })
    }

    /// Mass-accounting and uncertainty diagnostics of a pooled graph-posterior response.
    fn graph_posterior_response_diagnostics(
        &self,
        fits: &GraphPosteriorResponseFits,
        pooled: &GraphPosteriorPooled,
    ) -> Vec<Diagnostic> {
        let GraphPosteriorPooled { conditional, mixed_if_se, identified_mass, total_mass, .. } =
            pooled;
        let (identified_mass, total_mass) = (*identified_mass, *total_mass);
        let (unidentified_mass, failed_mass, subsampled_out_mass) =
            (fits.unidentified_mass, fits.failed_mass, fits.subsampled_out_mass);
        let weighted = &fits.weighted;
        let mut diagnostics = vec![
            Diagnostic::new(
                "estimate.response.graph_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "posterior_probability weights; identified_mass={}, unidentified_mass={}; \
                 unevaluable_mass={}; subsampled_out_mass={}; failed estimation and \
                 atoms the Interactive tier did not evaluate are not mixed into \
                 unidentified mass",
                    identified_mass / total_mass,
                    unidentified_mass / total_mass,
                    failed_mass / total_mass,
                    subsampled_out_mass / total_mass
                ),
            )
            .with_fields(super::super::mass_fields(
                Some(identified_mass / total_mass),
                unidentified_mass / total_mass,
            )),
        ];
        if weighted.len() > 1 && mixed_if_se.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.no_uncertainty_method",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if self.settings.inference.is_bayesian() {
                    "multi-atom Bayesian graph-posterior response withholds an aggregate \
                     credible interval: graph-specific dispersion and uncertainty of a \
                     frozen-weight aggregate are different objects"
                } else if conditional.is_none() {
                    "multi-atom Frequentist graph-posterior response has no aggregate: its \
                     atoms name different or partially identified estimands, so no mean and \
                     no band for one is published"
                } else if !matches!(conditional, Some(ResponseValue::Scalar(_))) {
                    "multi-atom Frequentist graph-posterior curve withholds an aggregate \
                     band: the joint-IF SE is published for scalar aggregates only, and \
                     marginal atom bands are not combined as independent"
                } else {
                    "multi-atom Frequentist graph-posterior response withholds an aggregate \
                     interval because aligned atom influences were unavailable; marginal \
                     atom SEs are not combined as independent"
                },
            ));
        } else if mixed_if_se.is_some() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.joint_if_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "SE of the frozen-weight aggregate from the joint influence-function \
                 covariance of identified atoms on the shared sample; simultaneous bands \
                 are not claimed",
            ));
        }
        diagnostics
    }

    /// Plugin-level refutation of a scalar graph-posterior intervention response.
    ///
    /// A mixed level (one estimand across the atoms) is refuted once. When the mean is
    /// withheld because the atoms name different estimands, each evaluated atom's own
    /// level is refuted against that atom's estimand and the reports are retained
    /// per atom, never mixed.
    fn refute_graph_posterior_response(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        estimator_id: EstimatorId,
        fits: &GraphPosteriorResponseFits,
        pooled: &GraphPosteriorPooled,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let GraphPosteriorResponseRun { data, query, treatment, outcome, ctx, .. } = *run;
        let GraphPosteriorPooled { conditional, scalar, standard_error, response, .. } = pooled;
        let mixed_level = match conditional {
            Some(ResponseValue::Scalar(value)) if value.is_finite() => Some(*value),
            _ => scalar.is_finite().then_some(*scalar),
        };
        let intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let requested = !matches!(self.settings.refute, RefuteSuite::None)
            || (!self.declared.custom_validators.is_empty()
                && self.settings.inference.is_frequentist());
        // `(weight, level, standard error, estimand, assumptions)` per refuted level.
        let mut levels: Vec<(f64, f64, f64, &IdentifiedEstimand, &antecedent_core::AssumptionSet)> =
            Vec::new();
        if let Some(level) = mixed_level {
            levels.push((1.0, level, *standard_error, &fits.estimand, &response.assumptions));
        } else {
            for ((_, weight, atom_response), (estimand, _)) in
                fits.weighted.iter().zip(&fits.contributing)
            {
                let (level, se) = response_scalar_summary(atom_response);
                if level.is_finite() {
                    levels.push((*weight, level, se, estimand, &atom_response.assumptions));
                }
            }
        }
        let per_atom = mixed_level.is_none();
        // A Bayesian pooled level is the posterior-mean level: the plugin-level suite reads
        // it exactly as it reads the Frequentist level; the posterior checks are added by
        // `bayesian_graph_posterior_level_state`.
        if !(intervention && requested && !levels.is_empty()) {
            if !matches!(self.settings.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued \
                     graph-posterior response",
                ));
            }
            return Ok((Vec::new(), Vec::new()));
        }
        let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
        let mut refute_ws = EstimationWorkspace::default();
        if matches!(self.settings.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
            diagnostics.push(Diagnostic::new(
                "refute.evalue.not_a_contrast",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "contrast-shaped refuters are not supported for a plugin intervention level; \
                 cheap runs overlap only and full runs overlap plus sampling-stability of the \
                 g-comp level",
            ));
        }
        let mut batches = Vec::with_capacity(levels.len());
        let mut notes = Vec::new();
        for (weight, level, se, estimand, assumptions) in levels {
            let plugin_estimate = EffectEstimate::new(
                level,
                se,
                assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let (reports, run_notes) = RefutationRun {
                data,
                estimand,
                query: &ate_query,
                estimate: &plugin_estimate,
                suite: self.settings.refute,
                estimator: estimator_id.as_str(),
                custom: &self.declared.custom_validators,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, ctx)?;
            notes.extend(run_notes);
            batches.push((weight, reports));
        }
        if !per_atom {
            let (_, reports) = batches.remove(0);
            return Ok((reports, notes));
        }
        let summary = Diagnostic::new(
            "refute.response.graph_posterior.per_atom",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the atoms name different estimands, so plugin-level refuters ran on each \
             evaluated atom's own level and their reports are kept per atom, not mixed",
        );
        Ok(super::super::atom_mix::finish_atom_refutations(
            batches,
            notes,
            false,
            super::super::atom_mix::ScalarRefuterMix::Effect,
            Some(summary),
            false,
        ))
    }
}

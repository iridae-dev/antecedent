//! Class-aware temporal effect execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGCompWorkspace, BayesianGComputationAte, BayesianValidation, CausalError,
    CausalPosterior, CausalQuery, ClassPosteriorMix, Diagnostic, EffectEstimate, EnvelopeAtomFit,
    EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId, ExecutionContext,
    FitFinishMeta, GraphEffectDraws, GraphIdentFlag, IdentificationStatus, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, RouteDefaults, Strategy, StudyResult, TableView, TabularData,
    TemporalEffectQuery, TemporalLinearAdjustment, TemporalRefutation, TimeSeriesData,
    ValidationSink, bayesian_temporal_gcomp, completion_fit_seed, elapsed_ns,
    enumeration_not_probability_diagnostic, envelope_to_identification_result_for,
    identification_status_ok_for_case, identified_set_interval_diagnostics,
    identify_cached_diagnostic, mix_class_effect_posteriors, plan_class_bayesian_atoms,
    posterior_identified_set_interval, push_conflict_diagnostics, report_identify_compute,
    resolve_envelope_prior_anchor, run_envelope_bayesian_full_validation, same_fit_inputs,
    same_fitted_problem, temporal_class_envelope_diagnostic, temporal_class_structural_mixture,
    temporal_refuter_average, with_conflict_summary,
};

use super::class_fits::{
    ClassBayesianFitAtom, ClassBayesianFits, ClassBayesianInputs, ClassBayesianPlan,
};

impl super::super::Study {
    pub(in crate::analysis::execute) fn execute_temporal_class(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the class-aware temporal effect",
        )?;
        if self.settings.inference.is_bayesian()
            && query.effect_modifiers.is_empty()
            && !query.is_multi_step_sustained()
        {
            return self.execute_temporal_class_bayesian(data, query, physical, ctx);
        }
        self.execute_temporal_class_frozen_weights(data, query, physical, ctx)
    }

    /// The class routes whose fit and assembly the sealed operation shares:
    /// conditional point, linear adjustment and sequential g-computation.
    pub(in crate::analysis::execute) fn execute_temporal_class_frozen_weights(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use super::super::temporal_class_effect_core::{
            TemporalClassEffectInputs, fit_temporal_class_effect,
        };
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.temporal_class_proof(physical, query, ctx)?;
        let fit = fit_temporal_class_effect(
            &TemporalClassEffectInputs {
                data,
                query,
                bundle: &bundle,
                graph_class: self.graph.class(),
                inference: &self.settings.inference,
                class_prior: self.declared.class_prior.as_ref(),
                bootstrap_replicates: self.settings.bootstrap_replicates,
                split: self.declared.split.as_ref(),
                refute: self.settings.refute,
                custom_validators: &self.declared.custom_validators,
                latency_mode: self.latency_mode,
                gaps: self.declares_complete_case(),
            },
            ctx,
        )?;
        let point_only = fit.point_only;
        let missing_data = fit.missing_data.clone();
        let mut identification = envelope_to_identification_result_for(
            &bundle.envelope.envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        identification.status = bundle.envelope.envelope.status;
        let args = fit.finish_args(
            physical,
            identification,
            &bundle,
            identifier_id,
            query,
            FitFinishMeta {
                identify_cached,
                structure_version: self.graph.version(),
                wall_time_ns: elapsed_ns(started),
            },
        );
        let mut result = super::super::temporal_effect_core::seal_point_only(
            point_only,
            self.finish_identified_execute(args),
        );
        result.extras.missing_data = missing_data;
        Ok(result)
    }

    /// The identifier and class proof of a one-shot class effect: the prepared
    /// cache when present, else a fresh identification.
    fn temporal_class_proof(
        &self,
        physical: &PhysicalExecutionPlan,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<
        (IdentifierId, Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, bool),
        CausalError,
    > {
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        if let Some((cache, prepared)) = self.reuse_temporal_class_identification(physical) {
            Ok((identifier_id, cache, prepared))
        } else {
            report_identify_compute(ctx);
            Ok((identifier_id, self.identify_temporal_class(identifier_id, query, ctx)?, false))
        }
    }

    #[allow(clippy::too_many_lines, reason = "one linear execution: admit, fit, aggregate, label")]
    pub(in crate::analysis::execute) fn execute_temporal_class_bayesian(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the class-aware Bayesian temporal effect",
        )?;
        let started = Instant::now();
        let InferenceMode::Bayesian(cfg) = &self.settings.inference else {
            return Err(CausalError::unsupported_inference(
                "temporal class Bayesian execute requires Bayesian inference",
            ));
        };
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
                "temporal class-aware effect not identified (no identified mass in envelope)",
            ));
        }
        let class_masses = self
            .declared
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, self.graph.class())];
        if class_masses.is_none() {
            diagnostics.push(enumeration_not_probability_diagnostic());
        }
        let inputs = ClassBayesianInputs { data, query, cfg, ctx };
        let ClassBayesianPlan { weights, flags, keys, fit_atoms, primary_estimand } =
            plan_class_bayesian_atoms(&bundle.envelope, class_masses.as_deref())?;
        let mut fits = self.fit_class_bayesian_atoms(&inputs, &fit_atoms, &mut diagnostics)?;
        if fits.atoms.is_empty() {
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
        let (estimate, posterior, mut structural_response) = Self::mix_class_bayesian_atoms(
            &inputs,
            (&bundle.envelope, class_masses.as_deref()),
            (weights, flags, keys),
            &mut fits,
            &mut diagnostics,
        )?;
        let (refutations, predictive_checks) = self.refute_class_bayesian_atoms(
            &inputs,
            &fits,
            &mut structural_response,
            &mut diagnostics,
        )?;
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
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
                structural_response: Some(structural_response),
                diagnostics: Some(diagnostics),
                predictive_checks,
                estimate_provenance: Some(
                    super::super::temporal_class_effect_core::class_envelope_provenance(
                        EstimatorId::BayesianTemporalGcomp,
                    ),
                ),
                ..Default::default()
            }),
        ))
    }

    /// Fits each identified completion's Bayesian temporal g-computation,
    /// sharing a posterior (and seed) between completions that fit the same
    /// problem under the same prior and status.
    fn fit_class_bayesian_atoms(
        &self,
        inputs: &ClassBayesianInputs<'_>,
        fit_atoms: &[ClassBayesianFitAtom],
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<ClassBayesianFits, CausalError> {
        let ClassBayesianInputs { data, query, cfg, ctx } = *inputs;
        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.inference.bootstrap_replicates = 0;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
        let mut fits = ClassBayesianFits {
            atoms: Vec::new(),
            atom_seeds: Vec::new(),
            per_graph: Vec::new(),
            atom_means: Vec::new(),
            envelope_conflict: None,
            estimator: bayes.inner.clone(),
        };
        let mut ws = BayesianGCompWorkspace::default();
        for (key, estimand, status, indexer, weight) in fit_atoms {
            let prep = estimator
                .prepare(
                    data,
                    estimand,
                    query,
                    indexer,
                    self.declared.split.as_ref(),
                    &ctx.kernel_policy,
                )
                .map_err(CausalError::from)?;
            let names =
                antecedent_estimate::temporal_coefficient_names(data, estimand, query, indexer)
                    .map_err(CausalError::from)?;
            let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
                .map_err(CausalError::from)?;
            let (resolved, conflict) = resolve_envelope_prior_anchor(cfg, &bprep, ctx)?;
            // Each completion draws from its own seed stream; completions fitting the
            // same problem under the same prior share a seed, so one model carries
            // one posterior.
            let seed = fits
                .atoms
                .iter()
                .zip(&fits.atom_seeds)
                .find(|(fit, _): &(&EnvelopeAtomFit, _)| {
                    fit.prior == resolved && same_fitted_problem(&fit.prep, &bprep)
                })
                .map_or_else(|| completion_fit_seed(ctx, *key), |(_, seed)| *seed);
            fits.atom_seeds.push(seed);
            bayes.inner.prior.clone_from(&resolved);
            bayes.inner.seed = seed;
            // An earlier completion that fitted every input of this fit (the problem,
            // the resolved prior, the seed they share, the identification status)
            // already holds its posterior, conflict summary included.
            let earlier = fits.atoms.iter().zip(&fits.atom_seeds).find(
                |(fit, fitted_seed): &(&EnvelopeAtomFit, _)| {
                    **fitted_seed == seed
                        && fit.status == *status
                        && fit.prior == resolved
                        && same_fit_inputs(&fit.prep, &bprep)
                },
            );
            let posterior = if let Some((fit, _)) = earlier {
                if let Some(summary) = conflict.as_ref() {
                    push_conflict_diagnostics(diagnostics, summary);
                }
                fit.posterior.clone()
            } else {
                let mut posterior =
                    bayes.fit(&bprep, *status, &mut ws, ctx).map_err(CausalError::from)?;
                if let Some(summary) = conflict.as_ref() {
                    // A transfer conflict is a per-completion diagnostic. It rides the
                    // atom posterior and the result diagnostics whether or not a class
                    // prior supports a mixture, and it never changes identification.
                    push_conflict_diagnostics(diagnostics, summary);
                    posterior = with_conflict_summary(posterior, summary.clone());
                }
                posterior
            };
            fits.envelope_conflict = fits.envelope_conflict.take().or(conflict);
            // The completion's effect is the posterior's effect column, which is not
            // necessarily the first reported quantity.
            let mean = posterior
                .effect_column()
                .and_then(|col| posterior.summaries.mean.get(col).copied())
                .unwrap_or(f64::NAN);
            fits.atom_means.push(mean);
            let draws = posterior
                .effect_column()
                .and_then(|col| posterior.draws.column(col).ok().map(|d| Arc::from(d.to_vec())));
            if let Some(effect_draws) = draws {
                fits.per_graph.push(GraphEffectDraws { graph_key: *key, effect_draws });
            }
            fits.atoms.push(EnvelopeAtomFit {
                key: *key,
                prep: bprep,
                posterior,
                status: *status,
                weight: *weight,
                estimand: estimand.clone(),
                indexer: Some(indexer.clone()),
                prior: resolved,
            });
        }
        fits.estimator = bayes.inner.clone();
        Ok(fits)
    }

    /// Mixes the atom posteriors into the class effect and builds the
    /// structural mixture with its posterior identified-set interval.
    fn mix_class_bayesian_atoms(
        inputs: &ClassBayesianInputs<'_>,
        (bundle_envelope, class_masses): (
            &antecedent_identify::TemporalClassEnvelope,
            Option<&[f64]>,
        ),
        (weights, flags, keys): (Vec<f64>, Vec<GraphIdentFlag>, Vec<u64>),
        fits: &mut ClassBayesianFits,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (EffectEstimate, Option<CausalPosterior>, crate::result::StructuralResponseMixture),
        CausalError,
    > {
        let ClassBayesianInputs { data, .. } = *inputs;
        let envelope = &bundle_envelope.envelope;
        let mut case_means: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut mean_idx = 0usize;
        for (i, case) in envelope.cases.iter().enumerate() {
            if identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                case_means[i] = fits.atom_means.get(mean_idx).copied();
                mean_idx += 1;
            }
        }
        let (estimate, posterior, weight_basis) = mix_class_effect_posteriors(
            ClassPosteriorMix {
                class_prior_supplied: class_masses.is_some(),
                truncated_completions: envelope.truncated_completions,
                evaluable_mass_positive: fits.atoms.iter().any(|atom| atom.weight > 0.0),
                weights,
                flags,
                keys,
                per_graph: &fits.per_graph,
                conflict: fits.envelope_conflict.take(),
                label: "temporal_class_envelope",
            },
            fits.atoms.iter().map(|atom| &atom.posterior),
            diagnostics,
        )?;
        let mut structural_response =
            temporal_class_structural_mixture(envelope, weight_basis, class_masses, &case_means);
        for atom in &mut structural_response.atoms {
            atom.posterior = fits
                .atoms
                .iter()
                .find(|fit| fit.key == atom.graph_key)
                .map(|fit| fit.posterior.clone());
        }
        structural_response.identified_set_interval = posterior_identified_set_interval(
            fits.atoms.iter().map(|fit| &fit.posterior),
            data.row_count(),
            envelope.truncated_completions > 0,
        );
        diagnostics.extend(
            structural_response
                .identified_set_interval
                .as_ref()
                .into_iter()
                .flat_map(identified_set_interval_diagnostics),
        );
        Ok((estimate, posterior, structural_response))
    }

    /// Runs the refutation suite and full Bayesian validation on each fitted
    /// completion, labelling reports by completion and attaching each
    /// validated posterior to its structural atom.
    fn refute_class_bayesian_atoms(
        &self,
        inputs: &ClassBayesianInputs<'_>,
        fits: &ClassBayesianFits,
        structural_response: &mut crate::result::StructuralResponseMixture,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (
            Vec<antecedent_validate::RefutationReport>,
            Vec<antecedent_validate::PredictiveCheckReport>,
        ),
        CausalError,
    > {
        let ClassBayesianInputs { data, query, cfg, ctx } = *inputs;
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = temporal_refuter_average(query);
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        for (atom, &seed) in fits.atoms.iter().zip(&fits.atom_seeds) {
            let refute_atom = EnvelopeRefuteAtom::from_fit(atom)?;
            let atom_ate = refute_atom.original.ate;
            let start = refutations.len();
            let (reports, na_diagnostics) = EnvelopeRefutation {
                data: &tabular,
                query: &ate_q,
                suite: self.settings.refute,
                estimator: EstimatorId::BayesianTemporalGcomp.as_str(),
                custom: &self.declared.custom_validators,
                temporal: Some(TemporalRefutation {
                    query,
                    split: self.declared.split.as_ref(),
                    time_index: data.time_index(),
                    gaps: false,
                }),
            }
            .run(std::slice::from_ref(&refute_atom), &mut refute_ws, ctx)?;
            refutations.extend(reports);
            diagnostics.extend(na_diagnostics);
            let mut atom_posterior = atom.posterior.clone();
            let atom_estimator = BayesianGComputationAte {
                prior: atom.prior.clone(),
                seed,
                ..fits.estimator.clone()
            };
            predictive_checks.extend(run_envelope_bayesian_full_validation(
                BayesianValidation { suite: self.settings.refute, cfg, estimator: &atom_estimator },
                std::slice::from_ref(atom),
                &mut atom_posterior,
                atom_ate,
                ctx,
                super::super::super::latency::predictive_check_sims(self.latency_mode),
                ValidationSink { refutations: &mut refutations, diagnostics },
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
        Ok((refutations, predictive_checks))
    }
}

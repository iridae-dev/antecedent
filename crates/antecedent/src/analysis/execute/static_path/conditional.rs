//! Conditional-effect execution and class aggregation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalQuery, ClassAtomOutcome, ClassEnvelopeRun, ConditionalLinearAdjustment,
    Dag, EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId,
    ExecuteRun, ExecutionContext, GraphClass, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, Instant, OverlapPolicy,
    PhysicalExecutionPlan, RefutationRun, RouteDefaults, Strategy, StudyResult, TabularData,
    elapsed_ns, envelope_se_omission_diagnostic, envelope_to_identification_result_for,
    identification_from_cache_or, identification_status_ok_for_case, mix_static_envelope_se,
    mix_weighted_analytic_se, mixed_static_influence, overlap_diagnostic, overlap_limit_diagnostic,
    report_identify_compute, select_claim, select_estimand, static_aligned_influence,
};

impl super::super::Study {
    pub(in crate::analysis::execute) fn execute_conditional(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.declared.estimator == Some(EstimatorId::BayesianBasisGcomp) {
            return self.execute_bayesian_basis(data, graph, &query.inner, physical, ctx);
        }
        if self.settings.inference.is_bayesian() {
            return self.execute_bayesian(data, graph, &query.inner, physical, None, ctx);
        }
        let started = Instant::now();
        let (identifier_id, _) = self.resolve_conditional_ids();
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query.inner), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        // A compiled plan's retained claim is reused when it names this identifier.
        let reused = self.reuse_static_identification_if(physical, |_| {
            physical
                .binding()
                .identifier_or(RouteDefaults::CONDITIONAL)
                .is_ok_and(|identifier| identifier == identifier_id)
        });
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, reused, || {
                let identification = crate::strategy_table::identify_conditional_static_in_context(
                    identifier_id,
                    graph,
                    query,
                    Some(ctx),
                )?;
                select_claim(identification, EstimatorId::ConditionalLinearAdjustment)
            })?;
        let data_est = super::super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let mut est = ConditionalLinearAdjustment::new();
        if query.inner.effect_modifiers.len() > 1 {
            est = est.with_joint_modifiers();
        }
        let mut mean_query = query.clone();
        mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
        let estimate =
            est.estimate(&data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
        let estimate = super::super::super::helpers::attach_conditional_functional_grid(
            estimate, data, query, &estimand, ctx,
        )?;
        let uses_aipw_scores = super::super::super::helpers::conditional_uses_crossfit_aipw(query);
        let estimator_name =
            if uses_aipw_scores { "aipw" } else { "conditional.linear.adjustment" };
        let estimator_id = if uses_aipw_scores {
            EstimatorId::Aipw
        } else {
            EstimatorId::ConditionalLinearAdjustment
        };
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, mut extra_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query: &query.inner,
            estimate: &estimate,
            suite: self.settings.refute,
            estimator: estimator_name,
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;
        if let Some(diagnostic) =
            super::super::super::helpers::conditional_quantile_grid_diagnostic(
                data,
                query,
                estimand.adjustment_set.iter().copied(),
            )?
        {
            extra_diagnostics.push(diagnostic);
        }
        extra_diagnostics
            .extend(super::super::super::helpers::conditional_score_estimator_diagnostic(query));
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.inner.treatment,
                query.inner.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started)),
        ))
    }

    /// Cpdag/Pag `ConditionalEffect` via the same generalized-adjustment envelope as ATE.
    pub(in crate::analysis) fn execute_class_conditional(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = self.resolve_class_conditional_ids();
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG ConditionalEffect execute missing resolved static PAG".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some((cache, prepared)) = self.reuse_pag_identification(physical) {
                        (Arc::clone(&cache.envelope), cache.identification.clone(), prepared)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = crate::strategy_table::identify_pag_in_context(
                            identifier_id,
                            pag,
                            &query.inner,
                            Some(ctx),
                        )?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (Arc::new(envelope), identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: super::super::pag_path::pag_envelope_diagnostic(
                            &envelope,
                        ),
                    },
                    identification,
                    Strategy::new(identifier_id, estimator_id),
                    super::super::pag_path::ClassTag::Pag,
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope: Arc::unwrap_or_clone(envelope),
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            GraphClass::Cpdag => {
                let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "CPDAG ConditionalEffect execute missing supplied graph".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some((cache, prepared)) = self.reuse_cpdag_identification(physical) {
                        (Arc::clone(&cache.envelope), cache.identification.clone(), prepared)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = crate::strategy_table::identify_cpdag_in_context(
                            identifier_id,
                            cpdag,
                            &query.inner,
                            Some(ctx),
                        )?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (Arc::new(envelope), identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: super::super::pag_path::cpdag_envelope_diagnostic(
                            &envelope,
                        ),
                    },
                    identification,
                    Strategy::new(identifier_id, estimator_id),
                    super::super::pag_path::ClassTag::Cpdag,
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope: Arc::unwrap_or_clone(envelope),
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            _ => Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect execute requires a Cpdag or Pag",
            }),
        }
    }

    fn finish_class_conditional<G>(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        strategy: Strategy,
        class_tag: super::super::pag_path::ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let estimator_id = strategy.estimator;
        let envelope = run.envelope;
        if envelope.unidentified_weight.0 > 1e-12
            && (query.inner.outcome_functional.quantile_level().is_some()
                || query.inner.outcome_functional.thresholds().is_some())
        {
            return Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect grid requires zero unidentified completion mass",
            });
        }
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if self.settings.inference.is_bayesian()
                || matches!(estimator_id, EstimatorId::BayesianConditional)
            {
                return self.execute_pag_nonidentified_prior(
                    &query.inner,
                    run,
                    identification,
                    class_tag.as_str(),
                );
            }
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                &format!(
                    "{class_tag} ConditionalEffect not identified (no identified mass in envelope)"
                ),
            ));
        }
        if self.settings.inference.is_bayesian()
            || matches!(estimator_id, EstimatorId::BayesianConditional)
        {
            return self.execute_pag_bayesian(data, &query.inner, run, identification, class_tag);
        }
        self.finish_class_conditional_frequentist(
            data,
            query,
            run,
            identification,
            strategy,
            class_tag,
        )
    }

    /// Frequentist pooled conditional effect over the identified atoms of a class envelope.
    fn finish_class_conditional_frequentist<G>(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        strategy: Strategy,
        class_tag: super::super::pag_path::ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic: envelope_diag,
        } = run;
        let data_est = super::super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let mut diagnostics = vec![envelope_diag];
        let atoms =
            evaluate_conditional_atoms(data, &data_est, query, envelope, estimator_id, ctx)?;
        let (mut estimate, estimand, n_contributing) = atoms.pooled_estimate(class_tag.as_str())?;
        let ConditionalAtoms { refute_atoms, grid_atoms, outcomes, .. } = atoms;
        if query.inner.outcome_functional.thresholds().is_some()
            || query.inner.outcome_functional.quantile_level().is_some()
        {
            estimate = super::super::attach_class_conditional_functional_grid(
                estimate,
                data,
                query,
                &grid_atoms,
                ctx,
            )?;
        }
        if let Some(diagnostic) =
            super::super::super::helpers::conditional_quantile_grid_diagnostic(
                data,
                query,
                grid_atoms.iter().flat_map(|(_, e)| e.adjustment_set.iter().copied()),
            )?
        {
            diagnostics.push(diagnostic);
        }
        diagnostics
            .extend(super::super::super::helpers::conditional_score_estimator_diagnostic(query));
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = EnvelopeRefutation {
            data,
            query: &query.inner,
            suite: self.settings.refute,
            estimator: estimator_id.as_str(),
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .run(&refute_atoms, &mut refute_ws, ctx)?;
        diagnostics.extend(na_diagnostics);
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic()));
        // Completions that disagree publish their identified set, exactly as the
        // class ATE arm does; a point-identified envelope stays a point.
        let structural_response =
            super::super::pag_path::static_class_structural_mixture(envelope, &outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(super::super::pag_path::static_class_identified_set_diagnostic(
                mixture,
                class_tag.as_str(),
            ));
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.inner.treatment,
                query.inner.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                structural_response,
                ..Default::default()
            }),
        ))
    }
}

/// Per-atom results of a class-aware conditional envelope, accumulated in envelope order.
struct ConditionalAtoms {
    weighted_ate: f64,
    se_items: Vec<(f64, f64)>,
    atom_ifs: Vec<Vec<f64>>,
    atom_weights: Vec<f64>,
    total_w: f64,
    primary_estimand: Option<IdentifiedEstimand>,
    assumptions: antecedent_core::AssumptionSet,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
    grid_atoms: Vec<(f64, IdentifiedEstimand)>,
    outcomes: Vec<ClassAtomOutcome>,
    /// What the table's sampling weights did to the first atom's fit, when the table has any.
    sampling_weights: Option<antecedent_estimate::sampling_weights::SamplingWeightRecord>,
}

/// Fits the conditional linear adjustment on each identified atom of the envelope.
fn evaluate_conditional_atoms<G>(
    data: &TabularData,
    data_est: &TabularData,
    query: &antecedent_core::ConditionalEffectQuery,
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
    estimator_id: EstimatorId,
    ctx: &ExecutionContext,
) -> Result<ConditionalAtoms, CausalError> {
    let mut atoms = ConditionalAtoms {
        weighted_ate: 0.0,
        se_items: Vec::new(),
        atom_ifs: Vec::new(),
        atom_weights: Vec::new(),
        total_w: 0.0,
        primary_estimand: None,
        assumptions: antecedent_core::AssumptionSet::default(),
        refute_atoms: Vec::new(),
        grid_atoms: Vec::new(),
        outcomes: vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()],
        sampling_weights: None,
    };
    let est = ConditionalLinearAdjustment::new();
    for (i, case) in envelope.cases.iter().enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, estimator_id)?;
        let mut mean_query = query.clone();
        mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
        let estimate =
            est.estimate(data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
        let w = case.weight.0;
        atoms.outcomes[i] = ClassAtomOutcome::Evaluated(estimate.ate);
        atoms.weighted_ate += w * estimate.ate;
        atoms.se_items.push((w, estimate.se_analytic()));
        if let Some(inf) = estimate
            .scores
            .influence
            .as_deref()
            .and_then(|inf| static_aligned_influence(data, &query.inner, &estimand, inf))
        {
            atoms.atom_ifs.push(inf);
            atoms.atom_weights.push(w);
        }
        atoms.total_w += w;
        if atoms.primary_estimand.is_none() {
            atoms.primary_estimand = Some(estimand.clone());
            atoms.assumptions = estimate.provenance.assumptions.clone();
            atoms.sampling_weights = estimate.diagnostics.sampling_weights;
        }
        atoms.grid_atoms.push((w, estimand.clone()));
        atoms.refute_atoms.push(EnvelopeRefuteAtom {
            key: i as u64,
            weight: w,
            estimand,
            indexer: None,
            original: estimate,
        });
    }
    Ok(atoms)
}

impl ConditionalAtoms {
    /// Weight-pooled point estimate, primary estimand and contributing-atom count.
    fn pooled_estimate(
        &self,
        class_tag: &str,
    ) -> Result<(EffectEstimate, IdentifiedEstimand, usize), CausalError> {
        if !matches!(self.total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: format!("{class_tag} ConditionalEffect envelope had no estimable cases"),
            });
        }
        let estimand = self.primary_estimand.clone().ok_or_else(|| CausalError::Compile {
            message: format!("{class_tag} ConditionalEffect envelope missing estimand"),
        })?;
        let n_contributing = self.se_items.len();
        let se = if self.atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&self.atom_ifs, &self.atom_weights)
        } else {
            mix_weighted_analytic_se(self.se_items.iter().copied())
        };
        let mut estimate = EffectEstimate::new(
            self.weighted_ate / self.total_w,
            se,
            self.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_sampling_weights(self.sampling_weights);
        if self.atom_ifs.len() == n_contributing
            && let Some(inf) = mixed_static_influence(&self.atom_ifs, &self.atom_weights)
        {
            estimate.scores.influence = Some(inf);
        }
        Ok((estimate, estimand, n_contributing))
    }
}

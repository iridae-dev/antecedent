// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
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
            let estimand = select_estimand(&id_res.result, EstimatorId::TemporalLinearAdjustment)?;
            (id_res.result, estimand, id_res.indexer, false)
        };
        require_identified(&identification)?;

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
                "bayesian.temporal.gcomp"
            } else {
                "temporal.linear.adjustment"
            },
            &self.custom_validators,
            Some(temporal_ctx),
        )?;
        diagnostics.extend(na_diagnostics);

        // Bayesian temporal: prior/posterior PPC + prior sensitivity on Full (mirror static).
        let mut posterior = posterior;
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

                let post_rep = PosteriorPredictiveCheck::new()
                    .check(&bprep, post)
                    .map_err(CausalError::from)?;
                refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));

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

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: EstimatorId::TemporalLinearAdjustment,
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
        let started = Instant::now();
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(self.identification_cache.as_deref(), || {
                let identification = TemporalMediationIdentifier {
                    allow_natural_controlled_alias: true,
                    ..TemporalMediationIdentifier::new()
                }
                .identify(graph, query)
                .map_err(CausalError::from)?;
                let estimand = select_estimand(&identification, EstimatorId::TemporalMediation)?;
                Ok((identification, estimand))
            })?;
        require_identified(&identification)?;
        let mut est = TemporalMediationEstimator::new();
        est.allow_natural_controlled_alias = true;
        let mediation = est.estimate(data, &estimand, query, ctx).map_err(CausalError::from)?;
        let estimate = mediation.effect.clone();
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::TemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
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
        if matches!(&self.inference, InferenceMode::Bayesian(_)) {
            return Err(CausalError::Unsupported {
                message: "Bayesian temporal response is not licensed in 0.7.0",
            });
        }
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
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
                        EstimatorId::TemporalResponseGcomp,
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
            aggregate_temporal_horizon_evidence(aligned.iter().map(|entry| &entry.identification))?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

        // Surface SEs are analytic (`TemporalResponseEstimator::new` zeros
        // bootstrap). Do not copy Study bootstrap here — it would look like
        // it affects the curve when FittedHorizon is OLS-only.
        let estimator = TemporalResponseEstimator::new();
        let mut response = estimator
            .estimate(data, &identifications, query, aggregate_status, aggregate_assumptions, ctx)
            .map_err(CausalError::from)?;
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
            estimator_id: EstimatorId::TemporalResponseGcomp,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
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

fn horizon_adjustment_sets_differ(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> bool {
    let Some(first) = entries.first() else {
        return false;
    };
    let first_z = named_adjustment_keys(first);
    entries.iter().skip(1).any(|entry| named_adjustment_keys(entry) != first_z)
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
}

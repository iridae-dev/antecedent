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

        let identification = identify_admg(identifier_id, admg, query)?;
        let estimand = select_estimand(&identification, estimator_id)?;
        let est = FunctionalEffect {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalEffect::new()
        };
        let prepared = est
            .prepare(
                data,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                &[query.treatment, query.outcome],
            )
            .map_err(CausalError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(CausalError::from)?;

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
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
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
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
        let envelope = identify_pag(identifier_id, pag, query)?;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if matches!(self.inference, InferenceMode::Bayesian(_))
                || matches!(estimator_id, EstimatorId::BayesianGcomp)
            {
                return self
                    .execute_pag_nonidentified_prior(query, physical, ctx, &envelope, started);
            }
            return Err(CausalError::Compile {
                message: "PAG effect not identified (no identified mass in envelope)".into(),
            });
        }

        let mut diagnostics = Vec::new();
        diagnostics.push(Diagnostic::new(
            "identify.pag.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "generalized.adjustment envelope: identified_mass={}, unidentified_mass={}, cases={}",
                envelope.identified_weight.0,
                envelope.unidentified_weight.0,
                envelope.cases.len()
            ),
        ));

        let identification = envelope_to_identification_result(&envelope, query);

        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_pag_bayesian(
                data,
                query,
                physical,
                ctx,
                &envelope,
                identification,
                started,
            );
        }

        let mut weighted_ate = 0.0;
        let mut weighted_se2 = 0.0;
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        for (i, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let mut estimand = select_estimand(&case.result, estimator_id)?;
            // Generalized-adjustment estimands are backdoor-shaped; estimators expect
            // the canonical backdoor method tag.
            if estimand.method.as_ref().starts_with("generalized.adjustment") {
                estimand.method = Arc::from("backdoor.adjustment");
            }
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
            weighted_ate += w * estimate.ate;
            if estimate.se_analytic.is_finite() {
                weighted_se2 += w * estimate.se_analytic * estimate.se_analytic;
            }
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
                assumptions = estimate.assumptions.clone();
            }
            let _ = i;
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "PAG envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "PAG envelope missing estimand".into(),
        })?;
        let estimate = EffectEstimate::new(
            weighted_ate / total_w,
            (weighted_se2 / total_w).sqrt(),
            assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
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

        diagnostics.push(overlap_diagnostic(estimate.overlap));
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
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
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    /// Non-identified PAG with Bayesian inference: prior-predictive draws, no invented ID.
    pub(super) fn execute_pag_nonidentified_prior(
        &self,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<Pag>,
        started: Instant,
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
            InferenceDiagnostics::analytic("pag_nonidentified_prior"),
            cfg.n_draws.max(1),
            ctx.rng.master_seed(),
        );
        let estimate = effect_from_posterior(&posterior)?;
        let identification = envelope_to_identification_result(envelope, query);
        let estimand = envelope.invariant.clone().unwrap_or_else(|| {
            IdentifiedEstimand::backdoor(
                "pag.nonidentified",
                Arc::from([]),
                antecedent_expr::ExprId::from_raw(0),
            )
        });
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "estimate.pag.nonidentified_prior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "PAG not identified; returning prior-predictive draws (unidentified_mass={})",
                posterior.unidentified_mass
            ),
        ));
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
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

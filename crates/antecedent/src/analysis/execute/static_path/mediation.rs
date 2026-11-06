//! Static total-mediation execution and estimation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, CausalQuery, Dag, EstimatorId, ExecutionContext, IdentificationStatus,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    PhysicalExecutionPlan, RefuteSuite, Strategy, StudyResult, TabularData, bayesian_gcomp,
    elapsed_ns, identification_from_cache_or, select_claim,
};

impl super::super::Study {
    pub(in crate::analysis::execute) fn execute_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let identification = crate::strategy_table::identify_static_query_in_context(
                    IdentifierId::PathSpecificNatural,
                    graph,
                    &CausalQuery::Mediation(query.clone()),
                    None,
                    Some(ctx),
                )?;
                select_claim(identification, EstimatorId::StaticMediationLinear)
            })?;
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if cfg.prior.is_some() || cfg.external_compose.is_some() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
                });
            }
            if cfg.prior_artifact.is_some() && cfg.prior_mapping.is_none() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
                });
            }
        }
        let (mediation, posterior) = self.estimate_static_mediation_total(
            data,
            graph,
            query,
            identification.required_assumptions.clone(),
            identification.status,
            ctx,
        )?;
        let estimate = mediation.effect.clone();
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let (cancelled, early_stopped) = (
            estimate.uncertainty.bootstrap.cancelled,
            estimate.uncertainty.bootstrap.early_stopped,
        );
        let refutations = if self.settings.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_static_mediation(
                data,
                graph,
                query,
                &mediation,
                self.settings.refute == RefuteSuite::Full,
                ctx,
            )?
        };
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::PathSpecificNatural,
                    EstimatorId::StaticMediationLinear,
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .mediation(Some(mediation))
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(cancelled)
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// Frequentist or Bayesian static mediation estimate for the total-effect route.
    fn estimate_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MediationQuery,
        assumptions: antecedent_core::AssumptionSet,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::TemporalMediationEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        ),
        CausalError,
    > {
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            let est = bayesian_gcomp(cfg, ctx);
            let decoded = crate::inference::decode_prior_hydrate_source(cfg)?;
            let bridge = decoded.as_ref().map(|d| antecedent_estimate::MediationPriorBridge {
                mapping: &d.mapping,
                quantities: &d.quantities,
                mean: &d.mean,
                sd: &d.sd,
                source_contrast: d.source_contrast,
            });
            let (mediation, posterior) = antecedent_estimate::estimate_static_mediation_bayesian(
                &antecedent_estimate::StaticMediationProblem { data, graph, query, extra: &[] },
                assumptions,
                &est,
                status,
                bridge,
                ctx,
            )?;
            Ok((mediation, Some(posterior)))
        } else {
            Ok((
                antecedent_estimate::estimate_static_mediation(
                    data,
                    graph,
                    query,
                    assumptions,
                    ctx.imputation_bootstrap().unwrap_or(self.settings.bootstrap_replicates),
                    &[],
                    ctx,
                )?,
                None,
            ))
        }
    }
}

//! Execution and uncertainty diagnostics for interventional distributions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BootstrapAccounting, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    DistributionGraph, EffectEstimate, EstimatorId, ExecutionContext, FunctionalDistribution,
    FunctionalDistributionWorkspace, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, InferenceMode, Instant,
    Intervention, PhysicalExecutionPlan, RefuteSuite, RouteDefaults, Strategy, StudyResult,
    TabularData, bayesian_draw_count, elapsed_ns, ensure_admg_distribution_supported,
    identification_from_cache_or, select_estimand,
};

impl super::super::Study {
    /// Identify + plug-in estimate for an interventional distribution on a
    /// supplied DAG, or on a finite-discrete ADMG via general ID (bidirected
    /// edges stay; they are not dropped to coerce a DAG).
    #[allow(
        clippy::too_many_lines,
        reason = "one interventional distribution: the identification, the plug-in estimate and the identified finish run in sequence"
    )]
    pub(in crate::analysis::execute) fn execute_distribution(
        &self,
        data: &TabularData,
        graph: DistributionGraph<'_>,
        query: &antecedent_core::InterventionalDistributionQuery,
        physical: &PhysicalExecutionPlan,
        distribution_operation: Option<
            &super::super::super::prepared::CheckedDistributionOperation,
        >,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if matches!(graph, DistributionGraph::Admg(_)) {
            ensure_admg_distribution_supported(query, self.settings.refute)?;
        }
        let started = Instant::now();
        let (identifier_id, estimator_id) =
            physical.binding().resolve(RouteDefaults::DISTRIBUTION)?;
        if !matches!(estimator_id, EstimatorId::FunctionalDistribution) {
            return Err(CausalError::Compile {
                message: format!(
                    "Distribution execute requires estimator functional.distribution; got {estimator_id}"
                ),
            });
        }

        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query), all frozen there, so reuse is exact
        // and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let identification = graph.identify(identifier_id, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let (row_resampling, sampling_design) = self.functional_row_resampling()?;
        let est = FunctionalDistribution {
            bootstrap_replicates: self.settings.bootstrap_replicates,
            ..FunctionalDistribution::new()
        }
        .with_row_resampling(row_resampling, sampling_design);
        let prepared = Self::prepare_distribution_problem(
            &est,
            data,
            query,
            &identification,
            &estimand,
            distribution_operation,
        )?;
        let (dist, posterior) =
            self.estimate_distribution(&est, &prepared, identification.status, ctx)?;

        let estimate = EffectEstimate::from_parts(
            dist.mean,
            dist.se_analytic,
            BootstrapAccounting {
                se: dist.se_bootstrap,
                replicates_ok: dist.bootstrap_replicates_ok,
                replicates_failed: dist.bootstrap_replicates_failed,
                cancelled: dist.bootstrap_cancelled,
                early_stopped: dist.bootstrap_early_stopped,
            },
            dist.assumptions.clone(),
            dist.overlap,
            None,
            dist.retained_memory_bytes,
        );

        let treatment =
            query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let cancelled = estimate.uncertainty.bootstrap.cancelled;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;

        // A distribution on a supplied Dag keeps this query-native suite. On an ADMG the
        // functional refuter state prepares the study at validation none and refutes the whole
        // target afterwards (`functional_validation`), so a suite requested there arrives as
        // those reports, not from this branch.
        let refutations = if self.settings.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_distribution(
                data,
                query,
                &identification,
                &estimand,
                &dist,
                self.settings.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(distribution_interval_diagnostics(&dist))
            .refutations(refutations)
            .distribution(Some(dist))
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

    /// Binds the distribution problem to `data`, through the retained operation when one exists.
    fn prepare_distribution_problem(
        est: &FunctionalDistribution,
        data: &TabularData,
        query: &antecedent_core::InterventionalDistributionQuery,
        identification: &IdentificationResult,
        estimand: &IdentifiedEstimand,
        distribution_operation: Option<
            &super::super::super::prepared::CheckedDistributionOperation,
        >,
    ) -> Result<antecedent_estimate::PreparedFunctionalDistribution, CausalError> {
        if let Some(operation) = distribution_operation {
            let checked_arena =
                antecedent_io::expr_arena_to_wire(operation.prepared().program().arena())
                    .map_err(CausalError::rejected)?;
            let claim_arena = antecedent_io::expr_arena_to_wire(&identification.arena)
                .map_err(CausalError::rejected)?;
            if operation.query() != query
                || operation.prepared().estimand.functional != estimand.functional
                || checked_arena != claim_arena
            {
                return Err(CausalError::Compile {
                    message: "prepared distribution target disagrees with selected identification"
                        .into(),
                });
            }
            operation.rebind(data)
        } else {
            est.prepare(
                data,
                query,
                estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)
        }
    }

    /// Frequentist plug-in or Bayesian posterior estimate of the prepared distribution.
    pub(in crate::analysis::execute) fn estimate_distribution(
        &self,
        est: &FunctionalDistribution,
        prepared: &antecedent_estimate::PreparedFunctionalDistribution,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::InterventionalDistributionEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        ),
        CausalError,
    > {
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if cfg.has_transfer() {
                return Err(CausalError::Unsupported {
                    message: "functional Bayesian prior transfer requires a declared \
                              functional mapping; a backdoor coefficient artifact cannot \
                              be applied as an isotropic CPT prior",
                });
            }
            let (dist, posterior) = est
                .estimate_bayesian(
                    prepared,
                    &[],
                    bayesian_draw_count(&self.settings.inference)?,
                    status,
                    ctx,
                )
                .map_err(CausalError::from)?;
            Ok((dist, Some(posterior)))
        } else {
            let mut ws = FunctionalDistributionWorkspace::default();
            Ok((est.estimate(prepared, &[], &mut ws, ctx).map_err(CausalError::from)?, None))
        }
    }
}

/// Name every interventional probability whose bounded interval could not be
/// formed (a boundary plug-in `p̂ ∈ {0, 1}`, a failed bootstrap, or zero
/// spread), so a missing interval is never mistaken for a computed one.
pub(in crate::analysis::execute) fn distribution_interval_diagnostics(
    dist: &antecedent_estimate::InterventionalDistributionEstimate,
) -> Vec<Diagnostic> {
    let missing: Vec<(usize, &'static str)> = dist
        .atom_uncertainty
        .iter()
        .enumerate()
        .filter_map(|(i, u)| match u.interval {
            antecedent_estimate::ProbabilityInterval::Unavailable(reason) => {
                Some((i, reason.as_str()))
            }
            antecedent_estimate::ProbabilityInterval::Bounded { .. } => None,
        })
        .collect();
    if missing.is_empty() {
        return Vec::new();
    }
    let atoms = missing.iter().map(|(i, r)| format!("{i}:{r}")).collect::<Vec<_>>().join(",");
    let mut diagnostic = Diagnostic::new(
        "estimate.distribution.interval_unavailable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        format!(
            "{} of {} interventional probabilities have no bounded interval (atom:reason \
             {atoms}); a plug-in probability of exactly 0 or 1 has no sampling spread to \
             build one from",
            missing.len(),
            dist.atom_uncertainty.len()
        ),
    );
    diagnostic.fields = Arc::from([(Arc::from("atoms"), Arc::from(atoms.as_str()))]);
    vec![diagnostic]
}

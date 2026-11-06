//! Prepared interventional-distribution execution.

use super::{
    Arc, CausalError, CausalQuery, CheckedDistributionOperation, EffectEstimate, ExecutionContext,
    GraphClass, InferenceMode, Instant, InterventionalDistributionQuery, RefuteSuite, StudyResult,
    TabularData, overlap_diagnostic, overlap_limit_diagnostic,
};
use antecedent_core::Intervention;
use antecedent_estimate::BootstrapAccounting;

impl CheckedDistributionOperation {
    pub(crate) fn query(&self) -> &InterventionalDistributionQuery {
        &self.query
    }

    pub(crate) fn prepared(&self) -> &antecedent_estimate::PreparedFunctionalDistribution {
        &self.prepared
    }

    pub(crate) fn rebind(
        &self,
        data: &TabularData,
    ) -> Result<antecedent_estimate::PreparedFunctionalDistribution, CausalError> {
        self.prepared.rebind_checked(data).map_err(CausalError::from)
    }

    pub(in crate::analysis::prepared) fn sealed_for_direct_execution(&self) -> bool {
        matches!(self.graph_class, GraphClass::Dag | GraphClass::Admg | GraphClass::TemporalAdmg)
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalDistribution
            && self.custom_validator_names.is_empty()
            && match self.graph_class {
                GraphClass::Dag => matches!(
                    self.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                ),
                GraphClass::Admg | GraphClass::TemporalAdmg => {
                    self.refute == RefuteSuite::None && self.query.conditioning.is_empty()
                }
                _ => false,
            }
    }

    /// Execute the already rebound provider; the sealed caller binds it once.
    pub(in crate::analysis::prepared) fn execute_bound(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if matches!(self.graph_class, GraphClass::Admg | GraphClass::TemporalAdmg)
            && !self.query.conditioning.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "ADMG InterventionalDistribution is supported for unconditional finite-discrete tables; IDC conditionals are a follow-up",
            });
        }
        let (distribution, posterior) = self.fit_distribution(&self.prepared, ctx)?;
        let bootstrap_ok = distribution.bootstrap_replicates_ok;
        let cancelled = distribution.bootstrap_cancelled;
        let early_stopped = distribution.bootstrap_early_stopped;
        let estimate = EffectEstimate::from_parts(
            distribution.mean,
            distribution.se_analytic,
            BootstrapAccounting {
                se: distribution.se_bootstrap,
                replicates_ok: distribution.bootstrap_replicates_ok,
                replicates_failed: distribution.bootstrap_replicates_failed,
                cancelled: distribution.bootstrap_cancelled,
                early_stopped: distribution.bootstrap_early_stopped,
            },
            distribution.assumptions.clone(),
            distribution.overlap,
            None,
            distribution.retained_memory_bytes,
        );
        let treatment =
            self.query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *self.query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        let diagnostics = self.result_diagnostics(&estimate, &distribution, posterior.as_ref());
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_distribution(
                data,
                &self.query,
                &self.cache.identification,
                &self.cache.estimand,
                &distribution,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let result = crate::analysis::execute::CheckedResultFinish::new(
            &self.common,
            &self.cache.identification,
            &self.cache.estimand,
            estimate,
            started,
            (treatment, outcome),
            (CausalQuery::Distribution(self.query.clone()), self.graph_class),
        )
        .with_diagnostics(diagnostics)
        .with_refutations(refutations)
        .with_distribution(distribution)
        .with_posterior(posterior)
        .with_bootstrap(Some(self.fitter.bootstrap_replicates), bootstrap_ok)
        .with_stop_flags(cancelled, early_stopped)
        .without_evalue_mirror()
        .with_rebound_interval()
        .finish();
        Ok(result)
    }

    /// Point or Bayesian fit of the prepared interventional distribution, chosen by the
    /// inference mode; the Bayesian arm also returns its joint posterior.
    fn fit_distribution(
        &self,
        prepared: &antecedent_estimate::PreparedFunctionalDistribution,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::InterventionalDistributionEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        ),
        CausalError,
    > {
        match &self.inference {
            InferenceMode::Frequentist => {
                let mut workspace = antecedent_estimate::FunctionalDistributionWorkspace::default();
                Ok((
                    self.fitter
                        .estimate(prepared, &[], &mut workspace, ctx)
                        .map_err(CausalError::from)?,
                    None,
                ))
            }
            InferenceMode::Bayesian(_) => {
                let n_draws = crate::analysis::execute::functional_bayesian_draws(&self.inference)?
                    .ok_or(CausalError::Unsupported {
                        message: "Bayesian inference selects no draw count",
                    })?;
                let (distribution, posterior) = self
                    .fitter
                    .estimate_bayesian(
                        prepared,
                        &[],
                        n_draws,
                        self.cache.identification.status,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                Ok((distribution, Some(posterior)))
            }
        }
    }

    /// Diagnostics for a fitted distribution, in the order the result reports them.
    fn result_diagnostics(
        &self,
        estimate: &EffectEstimate,
        distribution: &antecedent_estimate::InterventionalDistributionEstimate,
        posterior: Option<&antecedent_estimate::CausalPosterior>,
    ) -> Vec<antecedent_core::Diagnostic> {
        let mut diagnostics = self.cache.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.extend(distribution_interval_diagnostics_for_prepared(distribution));
        if let Some(diagnostic) = crate::analysis::execute::free_variables_averaged(
            &self.cache.identification,
            &self.cache.estimand,
        ) {
            diagnostics.push(diagnostic);
        }
        diagnostics.push(crate::analysis::execute::identify_cached_diagnostic());
        if let Some(posterior) = posterior {
            diagnostics.extend(crate::analysis::execute::posterior_note_diagnostics([posterior]));
        }
        diagnostics
    }
}

fn distribution_interval_diagnostics_for_prepared(
    distribution: &antecedent_estimate::InterventionalDistributionEstimate,
) -> Vec<antecedent_core::Diagnostic> {
    let missing: Vec<(usize, &'static str)> = distribution
        .atom_uncertainty
        .iter()
        .enumerate()
        .filter_map(|(index, atom)| match atom.interval {
            antecedent_estimate::ProbabilityInterval::Unavailable(reason) => {
                Some((index, reason.as_str()))
            }
            antecedent_estimate::ProbabilityInterval::Bounded { .. } => None,
        })
        .collect();
    if missing.is_empty() {
        return Vec::new();
    }
    let atoms = missing
        .iter()
        .map(|(index, reason)| format!("{index}:{reason}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut diagnostic = antecedent_core::Diagnostic::new(
        "estimate.distribution.interval_unavailable",
        antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Warning,
        format!(
            "{} of {} interventional probabilities have no bounded interval (atom:reason \
             {atoms}); a plug-in probability of exactly 0 or 1 has no sampling spread to \
             build one from",
            missing.len(),
            distribution.atom_uncertainty.len()
        ),
    );
    diagnostic.fields = Arc::from([(Arc::from("atoms"), Arc::from(atoms.as_str()))]);
    vec![diagnostic]
}

//! Tiered Bayesian average effects and scenario publication.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::execute;
use super::{
    AverageEffectQuery, BayesianGcompPointInput, CausalError, CausalPosterior, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    IdentificationResult, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifierId, InferenceMode, Instant, OverlapPolicy, PhysicalExecutionPlan, RefuteSuite,
    Strategy, StudyResult, TableView, TabularData, elapsed_ns, identification_from_cache_or,
    project_for_ate_estimate,
};

/// Per-scenario posteriors of an unidentified within-tier order: each scenario keeps its own
/// posterior and interval, the scalar stays withheld, and no scenario weight is invented.
fn tiered_unknown_scenarios(
    fits: Vec<(CausalPosterior, EffectEstimate)>,
    identification: &IdentificationResult,
    rows: usize,
) -> Result<(EffectEstimate, crate::result::StructuralResponseMixture), CausalError> {
    let mut values = Vec::with_capacity(fits.len());
    let mut intervals = Vec::with_capacity(fits.len());
    let mut atoms = Vec::with_capacity(fits.len());
    let mut assumptions = identification.required_assumptions.clone();
    for (index, (posterior, estimate)) in fits.into_iter().enumerate() {
        let column = posterior.effect_column().ok_or_else(|| CausalError::Compile {
            message: "tiered Bayesian scenario posterior has no effect column".into(),
        })?;
        let mut draws = posterior
            .draws
            .column(column)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?
            .to_vec();
        draws.sort_by(f64::total_cmp);
        let at = |q: f64| {
            let last = draws.len().saturating_sub(1);
            let position = (last as f64 * q).round();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the rounded position of a probability lies in [0, last]"
            )]
            let position = position as usize;
            draws.get(position.min(last)).copied().unwrap_or(f64::NAN)
        };
        intervals.push((at(0.025), at(0.975)));
        values.push(estimate.ate);
        if index == 0 {
            assumptions = estimate.provenance.assumptions.clone();
        }
        atoms.push(crate::result::StructuralResponseAtom {
            graph_key: u64::try_from(index).unwrap_or(u64::MAX),
            weight: 1.0,
            status: identification.status,
            value: Some(antecedent_core::ResponseValue::Scalar(estimate.ate)),
            posterior: Some(posterior),
            response: None,
        });
    }
    let finite = values.iter().copied().filter(|value| value.is_finite());
    let (mut lo, mut hi) = finite
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| (lo.min(value), hi.max(value)));
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        let span = intervals
            .iter()
            .filter(|(lower, upper)| lower.is_finite() && upper.is_finite() && lower <= upper);
        (lo, hi) = span.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (lower, upper)| {
            (lo.min(*lower), hi.max(*upper))
        });
    }
    let identified_set = (lo.is_finite() && hi.is_finite() && lo <= hi)
        .then(|| execute::scalar_identified_set(lo, hi));
    let identified_set_interval = execute::posterior_identified_set_interval(
        atoms.iter().filter_map(|atom| atom.posterior.as_ref()),
        rows,
        false,
    );
    let mixture = crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
        atoms,
        identified_mass: 1.0,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        subsampled_out_mass: 0.0,
        identified_set,
        identified_set_interval,
        conditional_on_identified: None,
        full_mass_scope: true,
        truncated_atoms: 0,
        aggregation_policy: None,
    };
    let mut estimate =
        EffectEstimate::new(f64::NAN, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
    estimate.scenarios.effects = Some(values.into());
    estimate.scenarios.intervals = Some(intervals.into());
    Ok((estimate, mixture))
}

impl execute::Study {
    /// Bayesian average effect under a declared tier background.
    ///
    /// `CoDetermined` identifies one adjustment set (the tier closure), so the posterior is the
    /// ordinary Bayesian g-computation posterior of that identified functional. `Unknown`
    /// identifies two canonical orientation scenarios that are neither exhaustive nor
    /// weighted: each scenario keeps its own posterior of its own functional, none is mixed
    /// into another, no scenario weight is invented, and the scalar stays withheld. A prior
    /// alone never identifies the within-tier order.
    pub(crate) fn execute_tiered_bayesian_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        background: &antecedent_graph::TieredBackground,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let InferenceMode::Bayesian(cfg) = &self.settings.inference else {
            return Err(CausalError::unsupported_inference(
                "tiered Bayesian average requires inference=Bayesian",
            ));
        };
        self.refuse_unsupported_tiered_average(query, cfg)?;
        let identifier_id = IdentifierId::GeneralizedAdjustment;
        let estimator_id = EstimatorId::BayesianGcomp;
        let (identification, _, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let identification = antecedent_identify::identify_tiered(background, query)?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "tiered identification returned no estimand".into(),
                    }
                })?;
                Ok((identification, estimand))
            })?;
        let unknown = background.within_tier == antecedent_graph::WithinTier::Unknown;
        let mut clock = crate::analysis::stage::StageClock::new();
        let mut fits =
            self.fit_tiered_scenarios(data, query, cfg, &identification, &mut clock, ctx)?;
        let Some(first_estimand) = fits.first().map(|(estimand, _, _)| estimand.clone()) else {
            return Err(CausalError::Compile {
                message: "tiered identification returned no estimand".into(),
            });
        };
        if !unknown {
            let Some((estimand, posterior, estimate)) = fits.pop() else {
                return Err(CausalError::Compile { message: "tiered fit missing".into() });
            };
            let n_draws = u32::try_from(posterior.draws.n_draws).ok();
            return Ok(self.finish_identified_execute(
                IdentifiedExecuteFinish::new(
                    physical,
                    identification,
                    estimand,
                    estimate,
                    Strategy::new(identifier_id, estimator_id),
                    query.treatment,
                    query.outcome,
                )
                .identify_cached(identify_cached)
                .wall_time_ns(elapsed_ns(started))
                .extras(IdentifiedExecuteExtras {
                    posterior: Some(posterior),
                    n_draws,
                    bootstrap_replicates_requested: Some(None),
                    ..Default::default()
                }),
            ));
        }
        let (estimate, mixture) = tiered_unknown_scenarios(
            fits.into_iter().map(|(_, posterior, estimate)| (posterior, estimate)).collect(),
            &identification,
            data.row_count(),
        )?;
        let diagnostic = Diagnostic::new(
            "tiered.unknown.bayesian_scenarios",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "each canonical scenario keeps its own posterior of its own functional (credible \
             intervals per scenario); the scenarios are not exhaustive completions and carry no \
             posterior weight, so none is mixed into another and the scalar stays withheld",
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                first_estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(vec![diagnostic])
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                structural_response: Some(mixture),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Fits the Bayesian g-computation posterior of every identified tiered estimand.
    fn fit_tiered_scenarios(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        cfg: &crate::inference::BayesianConfig,
        identification: &IdentificationResult,
        clock: &mut crate::analysis::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(IdentifiedEstimand, CausalPosterior, EffectEstimate)>, CausalError> {
        let mut fits = Vec::new();
        for estimand in &identification.estimands {
            let (data_est, query_est, estimand_est) =
                project_for_ate_estimate(data, query, estimand)?;
            let point = self.fit_bayesian_gcomp_point(
                &BayesianGcompPointInput {
                    data: &data_est,
                    estimand: &estimand_est,
                    query: &query_est,
                    cfg,
                    status: identification.status,
                    conditional: false,
                },
                clock,
                ctx,
            )?;
            fits.push((estimand.clone(), point.posterior, point.estimate));
        }
        Ok(fits)
    }

    /// Refuses a tiered Bayesian average outside the `AllObserved` mean at validation none.
    fn refuse_unsupported_tiered_average(
        &self,
        query: &AverageEffectQuery,
        cfg: &crate::inference::BayesianConfig,
    ) -> Result<(), CausalError> {
        if self.settings.refute != RefuteSuite::None
            || !self.declared.custom_validators.is_empty()
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
        {
            return Err(CausalError::Unsupported {
                message: "a tiered Bayesian average is the AllObserved mean at validation none",
            });
        }
        if cfg.has_transfer() {
            return Err(CausalError::Unsupported {
                message: "a tiered Bayesian average uses the isotropic coefficient prior",
            });
        }
        Ok(())
    }
}

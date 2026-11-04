//! Frequentist quantile treatment effect of a randomized (unconfounded) binary treatment.
//!
//! The target is the quantile coordinate of the cross-fitted cell and of the Bayesian
//! randomized quantile: the `tau`-quantile of each potential outcome law and their difference.
//! With an empty adjustment set the arm's distribution function is the arm's empirical
//! distribution function on the pooled-outcome threshold grid, inverted by the piecewise-linear
//! inversion every quantile route shares, with the delta-method influence interval. The
//! treatment must be unconfounded and coded exactly 0/1; an adjusting graph is refused with the
//! cross-fitted AIPW quantile as the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AverageEffectQuery, CausalError, CausalQuery, Dag, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, IdentifiedExecuteFinish,
    IdentifierId, Instant, OverlapPolicy, PhysicalExecutionPlan, Strategy, Study, StudyResult,
    TabularData, elapsed_ns, require_identified,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::randomized_quantile::randomized_quantile_effect;

fn refuse(code: &'static str, detail: &str, message: &str) -> CausalError {
    CausalError::Estimate(EstimationError::refused_detail(code, detail, message))
}

impl Study {
    /// Randomized quantile treatment effect: identify the back-door claim, then estimate.
    pub(super) fn execute_randomized_quantile(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let binary = |value: f64| {
            antecedent_core::Intervention::set(query.treatment, antecedent_core::Value::f64(value))
        };
        let Some(tau) = query.outcome_functional.quantile_level() else {
            return Err(CausalError::Unsupported {
                message: "randomized.quantile answers a quantile outcome functional",
            });
        };
        if !query.effect_modifiers.is_empty()
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || query.control != binary(0.0)
            || query.active != binary(1.0)
        {
            return Err(refuse(
                antecedent_core::reason_code!("route_not_supported"),
                "randomized_quantile.scope",
                "the randomized quantile answers the AllObserved quantile treatment effect of a \
                 binary 0/1 treatment without effect modifiers. Next attempt: a quantile outcome \
                 functional on an unmodified binary AverageEffect, or the cross-fitted AIPW \
                 quantile",
            ));
        }
        let (identification, estimand, identify_cached) = self.specialist_identification(
            graph,
            &CausalQuery::AverageEffect(query.clone()),
            (IdentifierId::BackdoorAdjustment, EstimatorId::RandomizedQuantile),
            physical,
            ctx,
        )?;
        require_identified(&identification)?;
        if !estimand.adjustment_set.is_empty() {
            return Err(refuse(
                antecedent_core::reason_code!("route_not_supported"),
                "randomized_quantile.adjustment_not_supported",
                "the empirical arm distribution functions carry no covariates, so they are the \
                 potential-outcome quantiles only when the treatment is unconfounded (an empty \
                 adjustment set). Next attempt: a graph in which the treatment has no \
                 confounding parents, or the cross-fitted AIPW quantile that adjusts",
            ));
        }
        let mask = data
            .complete_case_mask(&[query.treatment, query.outcome])
            .map_err(CausalError::from)?;
        let raw = data.float64_masked(query.treatment, &mask).map_err(CausalError::from)?;
        let outcome = data.float64_masked(query.outcome, &mask).map_err(CausalError::from)?;
        if raw.iter().any(|value| {
            !(value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == 1.0_f64.to_bits())
        }) {
            return Err(CausalError::Unsupported {
                message: "randomized.quantile requires treatment values coded exactly 0/1",
            });
        }
        let treated: Vec<bool> = raw.iter().map(|value| *value > 0.5).collect();
        let effect = randomized_quantile_effect(&outcome, &treated, tau)?;
        let mut estimate = EffectEstimate::new(
            effect.difference,
            effect.se,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(outcome.len() as u64);
        estimate.scores.influence = Some(effect.influence.clone().into());
        let diagnostics = vec![Diagnostic::new(
            "randomized_quantile.model_scope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "QTE at tau={tau}: Q_0={:.6} Q_1={:.6}; densities ({:.6}, {:.6}); empirical arm \
                 distribution functions on a {}-point pooled-outcome grid with piecewise-linear \
                 inversion; interval=normal_95 on the iid influence; treatment unconfounded \
                 (empty adjustment set); grid-selection uncertainty and interpolation bias are \
                 excluded",
                effect.control_quantile,
                effect.treated_quantile,
                effect.densities[0],
                effect.densities[1],
                effect.thresholds.len(),
            ),
        )];
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::BackdoorAdjustment, EstimatorId::RandomizedQuantile),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(diagnostics)
            .wall_time_ns(elapsed_ns(started)),
        ))
    }
}

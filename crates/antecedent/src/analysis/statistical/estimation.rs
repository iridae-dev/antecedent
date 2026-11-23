//! Retained-provider statistical estimation.

use super::{
    Assignment, ExecutionContext, IoError, PreparedStudy, StatisticalPreparedState,
    StatisticalStudyResult, VariableId, apply_posterior_draw_floor, assignments, bayesian_provider,
    err, estimate_err, point_options,
};

impl PreparedStudy<StatisticalPreparedState> {
    pub(super) fn bayesian_estimates(
        &self,
        ctx: &ExecutionContext,
        requests: &[Assignment],
    ) -> Result<Option<Vec<antecedent_estimate::BayesianStatisticalTransportEstimate>>, IoError>
    {
        let Some(provider) = bayesian_provider(self.state.options.estimator) else {
            return Ok(None);
        };
        let mut estimates = antecedent_estimate::evaluate_bayesian_statistical_transport_grid(
            &self.state.functional,
            &self.state.input,
            requests,
            self.state.limits,
            antecedent_estimate::statistical_transport::BayesianStatisticalTransportOptions {
                provider,
                draws: self.state.options.posterior_draws,
                coverage_level: self.state.options.coverage_level,
                max_joint_cells: self.state.options.max_joint_cells,
            },
            ctx,
        )
        .map_err(estimate_err)?;
        for estimate in &mut estimates {
            apply_posterior_draw_floor(estimate);
        }
        Ok(Some(estimates))
    }

    /// Execute without re-identification.
    ///
    /// # Errors
    /// Support failure, cancellation, or resource exhaustion.
    pub fn estimate_retained(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<StatisticalStudyResult, IoError> {
        for reservation in &self.state.provider_retention {
            reservation.observe_current();
        }
        if self.state.loaded_only {
            return Err(err(
                "transport.samples_not_embedded: refresh with source samples before re-estimating",
            ));
        }
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let point_options = point_options(self.state.options);
        let requests = if self.state.grid.is_empty() {
            vec![self.state.request.clone()]
        } else {
            self.state
                .grid
                .iter()
                .map(|request| {
                    Assignment::from_pairs(
                        request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
                    )
                })
                .collect()
        };
        let index = if self.state.grid.is_empty() {
            0
        } else {
            self.state
                .grid
                .iter()
                .position(|request| *request == assignments(&self.state.request))
                .ok_or_else(|| err("retained target missing from grid"))?
        };
        let mut estimate = antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid_with_point_laws(
            &self.state.functional,
            &self.state.input,
            &requests,
            self.state.limits,
            &point_options,
            &frozen_ctx,
            &self.state.data,
        )
        .map_err(estimate_err)?
        .remove(index);
        Self::withhold_unearned_percentile(&mut estimate, &self.state.options);
        let bayesian_estimate =
            self.bayesian_estimates(&frozen_ctx, &requests)?.map(|mut rows| rows.remove(index));
        Ok(StatisticalStudyResult {
            reasoning: self.reasoning(true, Some(&estimate), bayesian_estimate.as_ref()),
            estimate,
            bayesian_estimate,
            identities: self.state.identities.clone(),
        })
    }
}

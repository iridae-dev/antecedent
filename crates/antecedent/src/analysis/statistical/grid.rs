//! Shared-provider grid execution for statistical studies.

use super::{
    Arc, Assignment, ExecutionContext, IdentityDomain, IoError, PreparedStudy, StatisticalPremises,
    StatisticalPreparedState, StatisticalStudyResult, StudyInvariants, ValueWire, assignments,
    digest, err, estimate_err, point_options,
};

impl PreparedStudy<StatisticalPreparedState> {
    pub(super) fn with_grid(mut self, grid: Vec<Vec<(u32, ValueWire)>>) -> Result<Self, IoError> {
        if !grid.is_empty() {
            if !grid.contains(&assignments(&self.state.request)) {
                return Err(err("grid omits the retained target"));
            }
            self.state.identities.inference_binding = digest(
                IdentityDomain::InferenceBinding,
                &(&self.state.identities.inference_binding, &grid),
            )?;
            self.state.identities.execution = digest(
                IdentityDomain::Execution,
                &(
                    &self.state.identities.program,
                    &self.state.identities.snapshot,
                    &self.state.identities.inference_binding,
                ),
            )?;
        }
        self.state.grid = grid;
        Ok(self)
    }

    /// Evaluate a treatment grid with shared outer replicates and retained authority per point.
    /// # Errors
    /// Invalid request, absent source samples, or a failed joint execution.
    pub fn estimate_grid(
        &self,
        requests: &[Assignment],
        ctx: &ExecutionContext,
    ) -> Result<Vec<(Self, StatisticalStudyResult)>, IoError> {
        self.estimate_grid_with(requests, None, ctx)
    }

    /// [`Self::estimate_grid`] for a caller that already evaluated each request's plan on this
    /// study's retained point laws: `evaluated[i]` is the exact evaluation of `requests[i]`,
    /// used as that point's estimate instead of evaluating it again.
    /// # Errors
    /// Misaligned evaluations, absent source samples, or a failed joint execution.
    pub(in crate::analysis) fn estimate_grid_evaluated(
        &self,
        requests: &[Assignment],
        evaluated: Vec<antecedent_expr::ExactDistribution>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(Self, StatisticalStudyResult)>, IoError> {
        self.estimate_grid_with(requests, Some(evaluated), ctx)
    }

    fn estimate_grid_with(
        &self,
        requests: &[Assignment],
        evaluated: Option<Vec<antecedent_expr::ExactDistribution>>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(Self, StatisticalStudyResult)>, IoError> {
        if self.state.loaded_only {
            return Err(err("transport.samples_not_embedded"));
        }
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let point_options = point_options(self.state.options);
        let estimates = match evaluated {
            Some(distributions) => antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid_with_evaluated_point_laws(
                &self.state.functional,
                &self.state.input,
                requests,
                self.state.limits,
                &point_options,
                &frozen_ctx,
                antecedent_estimate::statistical_transport::EvaluatedPointLaws {
                    point_laws: &self.state.data,
                    distributions,
                },
            ),
            None => antecedent_estimate::statistical_transport::evaluate_statistical_transport_grid_with_point_laws(
                &self.state.functional,
                &self.state.input,
                requests,
                self.state.limits,
                &point_options,
                &frozen_ctx,
                &self.state.data,
            ),
        }
        .map_err(estimate_err)?;
        let posteriors = self.bayesian_estimates(&frozen_ctx, requests)?;
        let invariants = StudyInvariants::compute(
            &self.state.functional,
            &self.state.data,
            &self.state.samples,
        )?;
        let grid: Vec<_> = requests.iter().map(assignments).collect();
        requests
            .iter()
            .zip(estimates)
            .enumerate()
            .map(|(index, (request, mut estimate))| {
                let prepared = Self::from_invariants(
                    &invariants,
                    StatisticalPremises {
                        diagram: self.state.diagram.clone(),
                        functional: self.state.functional.clone(),
                        input: Arc::clone(&self.state.input),
                        data: self.state.data.clone(),
                        samples: self.state.samples.clone(),
                        request: request.clone(),
                        limits: self.state.limits,
                        options: self.state.options,
                    },
                    &frozen_ctx,
                    false,
                )?
                .with_grid(grid.clone())?;
                Self::withhold_unearned_percentile(&mut estimate, &prepared.state.options);
                let bayesian_estimate = posteriors.as_ref().map(|rows| rows[index].clone());
                let result = StatisticalStudyResult {
                    reasoning: prepared.reasoning(
                        true,
                        Some(&estimate),
                        bayesian_estimate.as_ref(),
                    ),
                    identities: prepared.state.identities.clone(),
                    estimate,
                    bayesian_estimate,
                };
                Ok((prepared, result))
            })
            .collect()
    }
}

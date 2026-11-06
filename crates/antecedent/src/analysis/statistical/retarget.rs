//! Retargeting and refresh of prepared statistical studies.

use super::{
    Arc, Assignment, DeferredStatisticalTransport, ExecutionContext, IoError, PreparedStudy,
    RetainedStatisticalInput, StatisticalPremises, StatisticalPreparedState,
    StatisticalStudyResult, StatisticalTransportInput, assemble_point_laws, check_deferred_cancel,
    declared_statistical_shape, err, estimate_err, point_options, rebind_raw_snapshots,
    rebind_snapshots,
};

impl PreparedStudy<StatisticalPreparedState> {
    /// Preview using the common transformation/invalidation contract.
    ///
    /// # Errors
    /// Invalid retained identity.
    pub fn preview_transform(
        &self,
        intent: antecedent_contract::TransformIntent,
    ) -> Result<antecedent_contract::TransformationReport, IoError> {
        super::super::exact::preview_transform_report(&self.state.identities, intent, "statistical")
    }

    /// Whether a replacement keeps the observation shape.
    ///
    /// # Errors
    /// Identity encoding failure.
    pub fn preview_snapshot(&self, input: &StatisticalTransportInput) -> Result<bool, IoError> {
        Ok(declared_statistical_shape(input, &self.state.functional, self.state.options)?
            == self.state.shape)
    }

    fn replacement(
        &self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if !self.preview_snapshot(&input)? {
            return Err(err("transport.reprepare_required"));
        }
        let data = assemble_point_laws(
            &input,
            &self.state.functional,
            &point_options(self.state.options),
            ctx,
        )
        .map_err(estimate_err)?;
        let catalog = rebind_snapshots(self.state.functional.catalog(), |regime| {
            data.laws()
                .iter()
                .filter(|law| law.regime() == regime)
                .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                .collect()
        })?;
        let functional = self.state.functional.derivation().bind_catalog(&catalog).map_err(err)?;
        let mut frozen_ctx = ctx.clone();
        frozen_ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        Self::build(
            self.state.diagram.clone(),
            functional,
            input,
            self.state.request.clone(),
            self.state.limits,
            self.state.options,
            &frozen_ctx,
        )?
        .with_grid(self.state.grid.clone())
    }

    /// Recheck declared work under a caller's physical policy, retaining the explicit fitted laws.
    /// # Errors
    /// Raw samples are absent, the declared work is refused, or execution is cancelled.
    pub fn requote_execution(&self, ctx: &ExecutionContext) -> Result<Self, IoError> {
        if self.state.loaded_only {
            return Err(err("transport.samples_not_embedded; bind_snapshot before execution"));
        }
        check_deferred_cancel(ctx)?;
        antecedent_estimate::statistical_transport::check_statistical_resources(
            &self.state.input,
            &self.state.functional,
            &self.state.options,
            ctx,
        )
        .map_err(estimate_err)?;
        Ok(self.clone())
    }

    /// Reconfigure a retained raw execution as an unfitted preparation on a new stream.
    /// # Errors
    /// Raw samples are absent after artifact loading, or resource/cancellation refusal.
    pub fn defer_execution(
        &self,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<DeferredStatisticalTransport, IoError> {
        if self.state.loaded_only {
            return Err(err(
                "transport.samples_not_embedded; bind_snapshot before changing execution",
            ));
        }
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(seed);
        DeferredStatisticalTransport::compile_retained(
            self.state.diagram.clone(),
            self.state.functional.clone(),
            RetainedStatisticalInput {
                input: Arc::clone(&self.state.input),
                retention: self.state.provider_retention.first().cloned(),
            },
            self.state.request.clone(),
            self.state.limits,
            self.state.options,
            &frozen,
        )
    }

    /// Rebind the target over immutable already materialized providers without refitting.
    /// A consumed artifact retains its original sample-replay limitations.
    ///
    /// # Errors
    /// Invalid target, changed limits, cancellation or admission refusal.
    pub fn rebind_assignment(
        &self,
        request: Assignment,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let mut next = Self::build_fitted(
            StatisticalPremises {
                diagram: self.state.diagram.clone(),
                functional: self.state.functional.clone(),
                input: Arc::clone(&self.state.input),
                data: self.state.data.clone(),
                samples: self.state.samples.clone(),
                request,
                limits: self.state.limits,
                options: self.state.options,
            },
            &frozen,
            self.state.loaded_only,
        )?;
        next.state.provider_retention.clone_from(&self.state.provider_retention);
        Ok(next)
    }

    /// Branch a fitted or consumed handle onto compatible raw inputs without fitting them.
    /// Independent artifact consumption remains separate from preparation.
    ///
    /// # Errors
    /// Changed observation shape, binding mismatch, cancellation or resource refusal.
    pub fn defer_snapshot(
        &self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<DeferredStatisticalTransport, IoError> {
        if !self.preview_snapshot(&input)? {
            return Err(err("transport.reprepare_required"));
        }
        let functional =
            rebind_raw_snapshots(&self.state.functional, &input, self.state.limits, ctx)?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        DeferredStatisticalTransport::compile(
            self.state.diagram.clone(),
            functional,
            input,
            self.state.request.clone(),
            self.state.limits,
            self.state.options,
            &frozen,
        )
    }

    /// Replace compatible snapshots and invalidate execution claims.
    ///
    /// # Errors
    /// Changed evidence shape, missing providers, or exceeded budget.
    pub fn replace_snapshot(
        &mut self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        *self = self.replacement(input, ctx)?;
        Ok(())
    }

    /// Atomic refresh: execute a candidate before publishing it.
    ///
    /// # Errors
    /// Failed preparation/evaluation leaves the previous valid state intact.
    pub fn refresh(
        &mut self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<StatisticalStudyResult, IoError> {
        let candidate = self.replacement(input, ctx)?;
        let result = candidate.estimate_retained(ctx)?;
        *self = candidate;
        Ok(result)
    }
}

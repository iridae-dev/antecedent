//! Retargeting deferred statistical providers without materialization.

use super::{
    Arc, Assignment, DeferredStatisticalTransport, EmpiricalTableOptions, ExecutionContext,
    IoError, RetainedStatisticalInput, StatisticalTransportInput, declared_statistical_shape, err,
    point_options, rebind_raw_snapshots,
};

impl DeferredStatisticalTransport {
    /// Rebind target coordinates without materializing a provider or altering this owner.
    /// Unchanged provider options share the immutable law cache, not the old target binding.
    ///
    /// # Errors
    /// Invalid inference settings, cancellation, or admission refusal.
    pub fn rebind(
        &self,
        request: Assignment,
        options: EmpiricalTableOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let mut next = Self::compile_retained(
            self.diagram.clone(),
            self.functional.clone(),
            RetainedStatisticalInput {
                input: Arc::clone(&self.input),
                retention: Some(Arc::clone(&self.retention)),
            },
            request,
            self.limits,
            options,
            &frozen,
        )?;
        if point_options(options) == point_options(self.options) {
            next.cache = Arc::clone(&self.cache);
        }
        Ok(next)
    }

    /// Reconfigure the execution stream without fitting or copying immutable rows.
    /// # Errors
    /// Declared work exceeds the caller's resource policy, or cancellation/proof refusal.
    pub fn with_seed(&self, seed: u64, ctx: &ExecutionContext) -> Result<Self, IoError> {
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(seed);
        let mut branch = Self::compile_retained(
            self.diagram.clone(),
            self.functional.clone(),
            RetainedStatisticalInput {
                input: Arc::clone(&self.input),
                retention: Some(Arc::clone(&self.retention)),
            },
            self.request.clone(),
            self.limits,
            self.options,
            &frozen,
        )?;
        if seed == self.seed {
            branch.cache = Arc::clone(&self.cache);
        }
        Ok(branch)
    }

    /// Bind a compatible raw evidence snapshot, leaving the parent and its claims untouched.
    /// This validates declared shape and catalog binding without fitting a joint law.
    ///
    /// # Errors
    /// Changed observation shape, invalid catalog bindings, cancellation or resource refusal.
    pub fn bind_snapshot(
        &self,
        input: StatisticalTransportInput,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if declared_statistical_shape(&input, &self.functional, self.options)?
            != declared_statistical_shape(&self.input, &self.functional, self.options)?
        {
            return Err(err("transport.reprepare_required"));
        }
        let functional = rebind_raw_snapshots(&self.functional, &input, self.limits, ctx)?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.seed);
        Self::compile(
            self.diagram.clone(),
            functional,
            input,
            self.request.clone(),
            self.limits,
            self.options,
            &frozen,
        )
    }
}

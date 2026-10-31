// SPDX-License-Identifier: MIT OR Apache-2.0
//! Retained admission and cancellation policy without caller execution callbacks.
/// Resource policy for rebinding and encoding; caller callbacks are never retained.
#[derive(Clone, Debug)]
pub(crate) struct CompiledResourcePolicy {
    parallelism: antecedent_core::execution::Parallelism,
    cancellation: antecedent_core::CancellationToken,
    admission: antecedent_core::execution::AdmissionPool,
    kernel: antecedent_core::KernelPolicy,
    temporal_limits: antecedent_core::TemporalMaterializationLimits,
}
impl CompiledResourcePolicy {
    pub(crate) fn retain(ctx: &antecedent_core::ExecutionContext) -> Self {
        Self {
            parallelism: ctx.parallelism,
            cancellation: ctx.cancellation.clone(),
            admission: ctx.admission(),
            kernel: ctx.kernel_policy,
            temporal_limits: ctx.temporal_materialization_limits(),
        }
    }
    pub(crate) fn context(&self) -> antecedent_core::ExecutionContext {
        let mut ctx = antecedent_core::ExecutionContext::for_tests(0);
        ctx.parallelism = self.parallelism;
        ctx.cancellation = self.cancellation.clone();
        ctx.admission = self.admission.clone();
        ctx.kernel_policy = self.kernel;
        ctx.with_temporal_materialization_limits(self.temporal_limits)
    }
}

#[cfg(test)]
mod tests {
    use super::CompiledResourcePolicy;
    #[test]
    fn retained_context_keeps_temporal_limits() {
        let ctx = antecedent_core::ExecutionContext::for_tests(7);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_offsets = 7;
        limits.max_work = 41;
        let ctx = ctx.with_temporal_materialization_limits(limits);
        let restored = CompiledResourcePolicy::retain(&ctx).context();
        assert_eq!(restored.temporal_materialization_limits(), limits);
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Retained admission and cancellation policy without caller execution callbacks.
/// Resource policy for rebinding and encoding; caller callbacks are never retained.
#[derive(Clone, Debug)]
pub(crate) struct CompiledResourcePolicy {
    parallelism: antecedent_core::execution::Parallelism,
    cancellation: antecedent_core::CancellationToken,
    admission: antecedent_core::execution::AdmissionPool,
    kernel: antecedent_core::KernelPolicy,
}
impl CompiledResourcePolicy {
    pub(crate) fn retain(ctx: &antecedent_core::ExecutionContext) -> Self {
        Self {
            parallelism: ctx.parallelism,
            cancellation: ctx.cancellation.clone(),
            admission: ctx.admission(),
            kernel: ctx.kernel_policy,
        }
    }
    pub(crate) fn context(&self) -> antecedent_core::ExecutionContext {
        let mut ctx = antecedent_core::ExecutionContext::for_tests(0);
        ctx.parallelism = self.parallelism;
        ctx.cancellation = self.cancellation.clone();
        ctx.admission = self.admission.clone();
        ctx.kernel_policy = self.kernel;
        ctx
    }
}

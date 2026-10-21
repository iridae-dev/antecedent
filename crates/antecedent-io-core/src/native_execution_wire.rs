// SPDX-License-Identifier: MIT OR Apache-2.0
//! Versioned, non-authoritative native execution observations.
use antecedent_core::KernelPolicy;
use antecedent_core::execution_facts::{
    ExecutionBackend, ExecutionCounts, ExecutionFactStatus, ExecutionFacts, ExecutionResources,
    MeasuredExecution, OutputRetention,
};
use serde::{Deserialize, Serialize};

/// Explicitly supported execution observation format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeExecutionFormat {
    /// Optional measurements, admitted reservations rather than process RSS.
    #[serde(rename = "native_execution_v1")]
    V1,
}
/// Recorded observations are descriptive; importing them never executes or licenses a claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeExecutionWire {
    /// Unknown formats are refused during decoding.
    pub format: NativeExecutionFormat,
    /// Instrumented physical boundaries; omitted metrics remain unknown.
    pub coverage: u64,
    /// Scans, copies, enclosing model fits, conditional-independence tests, cache hits.
    pub counts: [u64; 5],
    /// Peak admitted CPU slots, peak admitted bytes, actual spill bytes. Not process RSS.
    pub resources: [u64; 3],
    /// Actual execution backend label.
    pub backend: String,
    /// Portable optimized, architecture SIMD, forced scalar policy flags.
    pub kernel_policy: [bool; 3],
    /// Completed, cancelled or failed.
    pub status: NativeExecutionStatus,
}
/// Terminal execution status; no scientific authority follows from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeExecutionStatus {
    /// Returned an execution result.
    Completed,
    /// Cooperative cancellation.
    Cancelled,
    /// Refused or failed execution.
    Failed,
}
impl NativeExecutionWire {
    /// Capture native or previously recorded observations; caller declarations are excluded.
    #[must_use]
    pub fn from_facts(facts: &ExecutionFacts) -> Option<Self> {
        let coverage = facts.native_coverage()?;
        let counts = facts.counts();
        let resources = facts.resources();
        let kernel = facts.backend().kernel();
        Some(Self {
            format: NativeExecutionFormat::V1,
            coverage,
            counts: [
                counts.scans(),
                counts.copies(),
                counts.fits(),
                counts.tests(),
                counts.cache_hits(),
            ],
            resources: [resources.threads(), resources.memory_bytes(), resources.spill_bytes()],
            backend: facts.backend().name().into(),
            kernel_policy: [
                kernel.allow_portable_optimized,
                kernel.allow_arch_simd,
                kernel.force_scalar,
            ],
            status: match facts.status() {
                ExecutionFactStatus::Completed => NativeExecutionStatus::Completed,
                ExecutionFactStatus::Cancelled => NativeExecutionStatus::Cancelled,
                _ => NativeExecutionStatus::Failed,
            },
        })
    }
    /// Import observations as an artifact record, never as a new native execution.
    #[must_use]
    pub fn recorded_facts(&self) -> ExecutionFacts {
        let [scans, copies, fits, tests, cache_hits] = self.counts;
        let [threads, bytes, spill] = self.resources;
        let [allow_portable_optimized, allow_arch_simd, force_scalar] = self.kernel_policy;
        let status = match self.status {
            NativeExecutionStatus::Completed => ExecutionFactStatus::Completed,
            NativeExecutionStatus::Cancelled => ExecutionFactStatus::Cancelled,
            NativeExecutionStatus::Failed => ExecutionFactStatus::Failed,
        };
        ExecutionFacts::from_artifact(
            MeasuredExecution::new(
                ExecutionCounts::new(scans, copies, fits, tests, cache_hits),
                OutputRetention::SummaryOnly,
                ExecutionResources::new(threads, bytes, spill),
                ExecutionBackend::new(
                    self.backend.clone(),
                    KernelPolicy { allow_portable_optimized, allow_arch_simd, force_scalar },
                ),
                status,
            ),
            self.coverage,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imported_observations_remain_historical_and_unknown_metrics_remain_absent() {
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            let _lease = ctx.acquire_admission(0, 32)?;
            Ok::<_, antecedent_core::execution::AdmissionError>(())
        });
        let facts = attempt.facts.unwrap();
        assert!(facts.is_native());
        let wire = NativeExecutionWire::from_facts(&facts).unwrap();
        let bytes = crate::to_cbor(&wire).unwrap();
        let restored: NativeExecutionWire = crate::from_cbor(&bytes).unwrap();
        let imported = restored.recorded_facts();
        assert!(!imported.is_native());
        assert_eq!(imported.peak_admitted_bytes(), Some(32));
        assert_eq!(imported.observed_scans(), None);
        assert_eq!(imported.physical_identity(), facts.physical_identity());
    }
}

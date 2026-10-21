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
    /// Adds optional buffer volumes and an explicit measured retention mode.
    #[serde(rename = "native_execution_v2")]
    V2,
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
    /// Copied numeric buffer bytes and retained admitted output reservation bytes.
    /// Retention is cumulative handoff volume, not exact heap allocation or RSS.
    /// Absent means unmeasured.
    /// V1 did not encode these values even if its coverage contained the event bit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buffer_bytes: Option<[Option<u64>; 2]>,
    /// Absent in V1; omission never implies summary-only retention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention: Option<NativeOutputRetention>,
}
/// Measured output storage mode, without reusable authority or storage paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOutputRetention {
    /// Full in-memory output.
    FullReusable,
    /// Only a summary was retained.
    SummaryOnly,
    /// Output was retained on disk.
    DiskBacked,
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
            format: NativeExecutionFormat::V2,
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
            buffer_bytes: Some([facts.observed_copied_bytes(), facts.observed_retained_bytes()]),
            retention: match facts.retention() {
                OutputRetention::Unmeasured => None,
                OutputRetention::FullReusable => Some(NativeOutputRetention::FullReusable),
                OutputRetention::SummaryOnly => Some(NativeOutputRetention::SummaryOnly),
                OutputRetention::DiskBacked => Some(NativeOutputRetention::DiskBacked),
            },
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
        // V1 omitted buffer volumes and never measured retention, regardless of its
        // coverage mask. Do not invent zero volumes or infer a storage mode.
        let v2 = self.format == NativeExecutionFormat::V2;
        let [copied, retained] =
            if v2 { self.buffer_bytes.unwrap_or([None, None]) } else { [None, None] };
        let retention = if v2 { self.retention } else { None };
        let mut coverage = self.coverage;
        if copied.is_none() {
            coverage &= !(1 << 2);
        }
        if retained.is_none() {
            coverage &= !(1 << 4);
        }
        ExecutionFacts::from_artifact(
            MeasuredExecution::new(
                ExecutionCounts::new(scans, copies, fits, tests, cache_hits),
                match retention {
                    None => OutputRetention::Unmeasured,
                    Some(NativeOutputRetention::FullReusable) => OutputRetention::FullReusable,
                    Some(NativeOutputRetention::SummaryOnly) => OutputRetention::SummaryOnly,
                    Some(NativeOutputRetention::DiskBacked) => OutputRetention::DiskBacked,
                },
                ExecutionResources::new(threads, bytes, spill),
                ExecutionBackend::new(
                    self.backend.clone(),
                    KernelPolicy { allow_portable_optimized, allow_arch_simd, force_scalar },
                ),
                status,
            )
            .with_observed_buffer_bytes(copied, retained),
            coverage,
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
    #[test]
    fn v2_preserves_measured_bytes_and_v1_does_not_invent_missing_fields() {
        use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            record_physical(PhysicalEvent::CopiedBytes, 64);
            record_physical(PhysicalEvent::RetainedBytes, 128);
            Ok::<_, ()>(())
        });
        let facts = attempt.facts.unwrap();
        let wire = NativeExecutionWire::from_facts(&facts).unwrap();
        assert_eq!(wire.format, NativeExecutionFormat::V2);
        assert_eq!(wire.buffer_bytes, Some([Some(64), Some(128)]));
        let restored: NativeExecutionWire =
            crate::from_cbor(&crate::to_cbor(&wire).unwrap()).unwrap();
        let imported = restored.recorded_facts();
        assert_eq!(imported.observed_copied_bytes(), Some(64));
        assert_eq!(imported.observed_retained_bytes(), Some(128));
        assert_eq!(imported.retention(), OutputRetention::Unmeasured);
        assert_eq!(facts.physical_identity(), imported.physical_identity());
        let mut legacy = serde_json::to_value(wire).unwrap();
        legacy["format"] = "native_execution_v1".into();
        legacy.as_object_mut().unwrap().remove("buffer_bytes");
        legacy.as_object_mut().unwrap().remove("retention");
        let legacy: NativeExecutionWire = serde_json::from_value(legacy).unwrap();
        let imported = legacy.recorded_facts();
        assert_eq!(imported.observed_copied_bytes(), None);
        assert_eq!(imported.observed_retained_bytes(), None);
        assert_eq!(imported.retention(), OutputRetention::Unmeasured);
        assert!(!imported.is_native());
    }

    #[test]
    fn unknown_formats_are_refused() {
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let attempt =
            antecedent_core::execution_attempt::observe_with_context(&ctx, || Ok::<_, ()>(()));
        let mut wire =
            serde_json::to_value(NativeExecutionWire::from_facts(&attempt.facts.unwrap()).unwrap())
                .unwrap();
        wire["format"] = "native_execution_v3".into();
        assert!(serde_json::from_value::<NativeExecutionWire>(wire).is_err());
    }
}

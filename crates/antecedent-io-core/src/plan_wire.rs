//! Plan and performance wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;

/// Data classification, stored as the historical `snake_case` label.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataClassificationWire {
    /// IID tabular.
    Tabular,
    /// Temporal / time series.
    Temporal,
    /// Panel (unit × time).
    Panel,
    /// Multi-environment.
    MultiEnvironment,
    /// Irregular event data.
    Event,
}

impl From<DataClassification> for DataClassificationWire {
    fn from(value: DataClassification) -> Self {
        match value {
            DataClassification::Tabular => Self::Tabular,
            DataClassification::Temporal => Self::Temporal,
            DataClassification::Panel => Self::Panel,
            DataClassification::MultiEnvironment => Self::MultiEnvironment,
            DataClassification::Event => Self::Event,
        }
    }
}

impl From<DataClassificationWire> for DataClassification {
    fn from(value: DataClassificationWire) -> Self {
        match value {
            DataClassificationWire::Tabular => Self::Tabular,
            DataClassificationWire::Temporal => Self::Temporal,
            DataClassificationWire::Panel => Self::Panel,
            DataClassificationWire::MultiEnvironment => Self::MultiEnvironment,
            DataClassificationWire::Event => Self::Event,
        }
    }
}

/// Buffer layout, stored as the historical `snake_case` label.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BufferMaterializationWire {
    /// Borrowed without copy.
    Borrowed,
    /// Copied to contiguous storage.
    CopiedContiguous,
    /// Transposed layout.
    Transposed,
    /// Chunked / streaming.
    Chunked,
}

impl From<BufferMaterialization> for BufferMaterializationWire {
    fn from(value: BufferMaterialization) -> Self {
        match value {
            BufferMaterialization::Borrowed => Self::Borrowed,
            BufferMaterialization::CopiedContiguous => Self::CopiedContiguous,
            BufferMaterialization::Transposed => Self::Transposed,
            BufferMaterialization::Chunked => Self::Chunked,
        }
    }
}

impl From<BufferMaterializationWire> for BufferMaterialization {
    fn from(value: BufferMaterializationWire) -> Self {
        match value {
            BufferMaterializationWire::Borrowed => Self::Borrowed,
            BufferMaterializationWire::CopiedContiguous => Self::CopiedContiguous,
            BufferMaterializationWire::Transposed => Self::Transposed,
            BufferMaterializationWire::Chunked => Self::Chunked,
        }
    }
}

/// Kernel class, stored as the historical `snake_case` label.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KernelSelectionWire {
    /// Portable scalar reference.
    Scalar,
    /// Portable optimized.
    PortableOptimized,
    /// External dense backend.
    DenseBackend,
}

impl From<KernelSelection> for KernelSelectionWire {
    fn from(value: KernelSelection) -> Self {
        match value {
            KernelSelection::Scalar => Self::Scalar,
            KernelSelection::PortableOptimized => Self::PortableOptimized,
            KernelSelection::DenseBackend => Self::DenseBackend,
        }
    }
}

impl From<KernelSelectionWire> for KernelSelection {
    fn from(value: KernelSelectionWire) -> Self {
        match value {
            KernelSelectionWire::Scalar => Self::Scalar,
            KernelSelectionWire::PortableOptimized => Self::PortableOptimized,
            KernelSelectionWire::DenseBackend => Self::DenseBackend,
        }
    }
}

/// Logical plan wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LogicalAnalysisPlanWire {
    /// Plan id.
    pub plan_id: String,
    /// Data classification.
    pub data_classification: DataClassificationWire,
    /// Discovery algorithm.
    pub discovery_algorithm: Option<String>,
    /// Graph review required.
    pub graph_review_required: bool,
    /// Identifier.
    pub identifier: Option<String>,
    /// Estimator.
    pub estimator: Option<String>,
    /// Validation suite.
    pub validation_suite: Option<String>,
    /// Query variables.
    pub query_variables: Vec<u32>,
}

/// Physical plan wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExecutionPlanWire {
    /// Plan id.
    pub plan_id: String,
    /// Materializations.
    pub materializations: Vec<(String, BufferMaterializationWire)>,
    /// Kernels.
    pub kernels: Vec<(String, KernelSelectionWire)>,
    /// Batch size.
    pub batch_size: Option<u64>,
    /// Workspace bytes.
    pub workspace_bytes: Option<u64>,
    /// Peak memory.
    pub estimated_peak_memory_bytes: Option<u64>,
    /// Copy bytes.
    pub estimated_copy_bytes: Option<u64>,
    /// Task schedule.
    pub task_schedule: Vec<(String, u32)>,
    /// Workers.
    pub worker_threads: u32,
    /// Deterministic reductions.
    pub deterministic_reductions: bool,
    /// Python crossings.
    pub expected_python_crossings: u32,
}

/// Performance wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPerformanceWire {
    /// Optional native execution v1 observations; absent in earlier artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_execution: Option<crate::native_execution_wire::NativeExecutionWire>,
    /// Wall ns.
    pub wall_time_ns: Option<u64>,
    /// Peak RSS.
    pub peak_rss_bytes: Option<u64>,
    /// Copies.
    pub copy_count: u64,
    /// Scalar fallbacks.
    pub scalar_fallback_count: u64,
    /// Latency tier label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_mode: Option<String>,
    /// Stage timings `(stage, ns)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stage_timings_ns: Vec<(String, u64)>,
    /// Bootstrap replicates requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_replicates_requested: Option<u32>,
    /// Bootstrap replicates that succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_replicates_ok: Option<u32>,
    /// Posterior draws.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_draws: Option<u32>,
    /// Cancellation observed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cancelled: bool,
    /// Adaptive early-stop.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub early_stopped: bool,
    /// Arrow CDI bytes borrowed (zero-copy ingest).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_borrowed: Option<u64>,
}

/// Encode logical plan.
#[must_use]
pub fn logical_plan_to_wire(p: &LogicalAnalysisPlanRecord) -> LogicalAnalysisPlanWire {
    LogicalAnalysisPlanWire {
        plan_id: p.plan_id.to_string(),
        data_classification: p.data_classification.into(),
        discovery_algorithm: p.discovery_algorithm.as_ref().map(ToString::to_string),
        graph_review_required: p.graph_review_required,
        identifier: p.identifier.as_ref().map(ToString::to_string),
        estimator: p.estimator.as_ref().map(ToString::to_string),
        validation_suite: p.validation_suite.as_ref().map(ToString::to_string),
        query_variables: vars_to_raw(&p.query_variables),
    }
}

/// Decode logical plan.
///
/// # Errors
///
/// A classification outside the historical labels fails while the wire is decoded,
/// so this conversion does not see it.
pub fn logical_plan_from_wire(
    w: &LogicalAnalysisPlanWire,
) -> Result<LogicalAnalysisPlanRecord, IoError> {
    Ok(LogicalAnalysisPlanRecord {
        plan_id: Arc::from(w.plan_id.as_str()),
        data_classification: w.data_classification.into(),
        discovery_algorithm: w.discovery_algorithm.as_ref().map(|s| Arc::from(s.as_str())),
        graph_review_required: w.graph_review_required,
        identifier: w.identifier.as_ref().map(|s| Arc::from(s.as_str())),
        estimator: w.estimator.as_ref().map(|s| Arc::from(s.as_str())),
        validation_suite: w.validation_suite.as_ref().map(|s| Arc::from(s.as_str())),
        query_variables: vars_from_raw(&w.query_variables),
    })
}

/// Encode physical plan.
#[must_use]
pub fn physical_plan_to_wire(p: &PhysicalExecutionPlanRecord) -> PhysicalExecutionPlanWire {
    PhysicalExecutionPlanWire {
        plan_id: p.plan_id.to_string(),
        materializations: p
            .materializations
            .iter()
            .map(|(n, m)| (n.to_string(), (*m).into()))
            .collect(),
        kernels: p.kernels.iter().map(|(n, k)| (n.to_string(), (*k).into())).collect(),
        batch_size: p.batch_size.map(crate::error::lossless_u64),
        workspace_bytes: p.workspace_bytes,
        estimated_peak_memory_bytes: p.estimated_peak_memory_bytes,
        estimated_copy_bytes: p.estimated_copy_bytes,
        task_schedule: p.task_schedule.iter().map(|t| (t.dimension.to_string(), t.units)).collect(),
        worker_threads: p.worker_threads,
        deterministic_reductions: p.deterministic_reductions,
        expected_python_crossings: p.expected_python_crossings,
    }
}

/// Decode physical plan.
///
/// # Errors
///
/// A tag outside the historical labels fails while the wire is decoded.
/// A batch size that does not fit `usize` is [`IoError::TooLarge`].
pub fn physical_plan_from_wire(
    w: &PhysicalExecutionPlanWire,
) -> Result<PhysicalExecutionPlanRecord, IoError> {
    Ok(PhysicalExecutionPlanRecord {
        plan_id: Arc::from(w.plan_id.as_str()),
        materializations: w
            .materializations
            .iter()
            .map(|(n, m)| (Arc::from(n.as_str()), (*m).into()))
            .collect::<Vec<_>>()
            .into(),
        kernels: w
            .kernels
            .iter()
            .map(|(n, k)| (Arc::from(n.as_str()), (*k).into()))
            .collect::<Vec<_>>()
            .into(),
        batch_size: w
            .batch_size
            .map(|b| usize::try_from(b).map_err(|_| IoError::TooLarge))
            .transpose()?,
        workspace_bytes: w.workspace_bytes,
        estimated_peak_memory_bytes: w.estimated_peak_memory_bytes,
        estimated_copy_bytes: w.estimated_copy_bytes,
        task_schedule: w
            .task_schedule
            .iter()
            .map(|(d, u)| ParallelTaskSpec { dimension: Arc::from(d.as_str()), units: *u })
            .collect::<Vec<_>>()
            .into(),
        worker_threads: w.worker_threads,
        deterministic_reductions: w.deterministic_reductions,
        expected_python_crossings: w.expected_python_crossings,
    })
}

/// Encode performance.
#[must_use]
pub fn performance_to_wire(p: &ExecutionPerformanceRecord) -> ExecutionPerformanceWire {
    ExecutionPerformanceWire {
        native_execution: p
            .native_execution
            .as_ref()
            .and_then(crate::native_execution_wire::NativeExecutionWire::from_facts),
        wall_time_ns: p.wall_time_ns,
        peak_rss_bytes: p.peak_rss_bytes,
        copy_count: p.copy_count,
        scalar_fallback_count: p.scalar_fallback_count,
        latency_mode: p.latency_mode.as_ref().map(std::string::ToString::to_string),
        stage_timings_ns: p.stage_timings_ns.iter().map(|(s, ns)| (s.to_string(), *ns)).collect(),
        bootstrap_replicates_requested: p.bootstrap_replicates_requested,
        bootstrap_replicates_ok: p.bootstrap_replicates_ok,
        n_draws: p.n_draws,
        cancelled: p.cancelled,
        early_stopped: p.early_stopped,
        bytes_borrowed: p.bytes_borrowed,
    }
}

/// Decode performance.
#[must_use]
pub fn performance_from_wire(w: &ExecutionPerformanceWire) -> ExecutionPerformanceRecord {
    ExecutionPerformanceRecord {
        native_execution: w
            .native_execution
            .as_ref()
            .map(crate::native_execution_wire::NativeExecutionWire::recorded_facts),
        wall_time_ns: w.wall_time_ns,
        peak_rss_bytes: w.peak_rss_bytes,
        copy_count: w.copy_count,
        scalar_fallback_count: w.scalar_fallback_count,
        latency_mode: w.latency_mode.as_ref().map(|s| Arc::<str>::from(s.as_str())),
        stage_timings_ns: w
            .stage_timings_ns
            .iter()
            .map(|(s, ns)| (Arc::<str>::from(s.as_str()), *ns))
            .collect(),
        bootstrap_replicates_requested: w.bootstrap_replicates_requested,
        bootstrap_replicates_ok: w.bootstrap_replicates_ok,
        n_draws: w.n_draws,
        cancelled: w.cancelled,
        early_stopped: w.early_stopped,
        bytes_borrowed: w.bytes_borrowed,
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferMaterializationWire, DataClassificationWire, KernelSelectionWire};
    use crate::convert::assert_cbor_text_code;

    #[test]
    fn closed_plan_codes_encode_as_their_historical_strings() {
        for (text, value) in [
            ("tabular", DataClassificationWire::Tabular),
            ("temporal", DataClassificationWire::Temporal),
            ("panel", DataClassificationWire::Panel),
            ("multi_environment", DataClassificationWire::MultiEnvironment),
            ("event", DataClassificationWire::Event),
        ] {
            assert_cbor_text_code(text, &value);
        }
        for (text, value) in [
            ("borrowed", BufferMaterializationWire::Borrowed),
            ("copied_contiguous", BufferMaterializationWire::CopiedContiguous),
            ("transposed", BufferMaterializationWire::Transposed),
            ("chunked", BufferMaterializationWire::Chunked),
        ] {
            assert_cbor_text_code(text, &value);
        }
        for (text, value) in [
            ("scalar", KernelSelectionWire::Scalar),
            ("portable_optimized", KernelSelectionWire::PortableOptimized),
            ("dense_backend", KernelSelectionWire::DenseBackend),
        ] {
            assert_cbor_text_code(text, &value);
        }
    }
}

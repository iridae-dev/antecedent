//! Logical and physical plan *records* (structs only; planner is Phase 3).
//!
//! These types attach to diagnostics and results so execution choices remain
//! inspectable (DESIGN.md §21.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;

/// Data modality classification used by the logical planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DataClassification {
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

/// Record of logical analysis semantics (no execution choices).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalAnalysisPlanRecord {
    /// Stable plan identifier.
    pub plan_id: Arc<str>,
    /// Data classification.
    pub data_classification: DataClassification,
    /// Requested discovery algorithm id, if any.
    pub discovery_algorithm: Option<Arc<str>>,
    /// Whether graph review is required before estimation.
    pub graph_review_required: bool,
    /// Identifier algorithm id, if any.
    pub identifier: Option<Arc<str>>,
    /// Estimator / inference method id, if any.
    pub estimator: Option<Arc<str>>,
    /// Validation suite id, if any.
    pub validation_suite: Option<Arc<str>>,
    /// Variables involved in the primary query.
    pub query_variables: Arc<[VariableId]>,
}

/// How a column or buffer is supplied to a kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferMaterialization {
    /// Borrowed without copy.
    Borrowed,
    /// Copied to contiguous storage.
    CopiedContiguous,
    /// Transposed layout.
    Transposed,
    /// Chunked / streaming.
    Chunked,
}

/// Selected kernel implementation class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum KernelSelection {
    /// Portable scalar reference.
    Scalar,
    /// Portable optimized (e.g. auto-vectorized).
    PortableOptimized,
    /// Architecture-specific SIMD.
    ArchSimd,
    /// External dense backend (e.g. faer).
    DenseBackend,
}

/// Record of physical execution choices derived from a logical plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalExecutionPlanRecord {
    /// Stable plan identifier (may match logical).
    pub plan_id: Arc<str>,
    /// Materialization choices by logical buffer name.
    pub materializations: Arc<[(Arc<str>, BufferMaterialization)]>,
    /// Selected kernels by kernel name.
    pub kernels: Arc<[(Arc<str>, KernelSelection)]>,
    /// Batch size chosen for the primary work unit.
    pub batch_size: Option<usize>,
    /// Declared workspace bytes.
    pub workspace_bytes: Option<u64>,
    /// Estimated peak memory bytes.
    pub estimated_peak_memory_bytes: Option<u64>,
    /// Worker threads assigned (0 = serial).
    pub worker_threads: u32,
    /// Whether reductions must be deterministic.
    pub deterministic_reductions: bool,
    /// Expected Python boundary crossings for the plan.
    pub expected_python_crossings: u32,
}

/// Lightweight performance summary attached to results.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionPerformanceRecord {
    /// Wall time in nanoseconds, if measured.
    pub wall_time_ns: Option<u64>,
    /// Peak resident memory bytes, if measured.
    pub peak_rss_bytes: Option<u64>,
    /// Number of recorded buffer copies.
    pub copy_count: u64,
    /// Number of scalar kernel fallbacks.
    pub scalar_fallback_count: u64,
}

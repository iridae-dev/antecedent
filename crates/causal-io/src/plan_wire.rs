//! Plan and performance wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;

/// Logical plan wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LogicalAnalysisPlanWire {
    /// Plan id.
    pub plan_id: String,
    /// Data classification.
    pub data_classification: String,
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
pub struct PhysicalExecutionPlanWire {
    /// Plan id.
    pub plan_id: String,
    /// Materializations.
    pub materializations: Vec<(String, String)>,
    /// Kernels.
    pub kernels: Vec<(String, String)>,
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
pub struct ExecutionPerformanceWire {
    /// Wall ns.
    pub wall_time_ns: Option<u64>,
    /// Peak RSS.
    pub peak_rss_bytes: Option<u64>,
    /// Copies.
    pub copy_count: u64,
    /// Scalar fallbacks.
    pub scalar_fallback_count: u64,
}

/// Encode logical plan.
#[must_use]
pub fn logical_plan_to_wire(p: &LogicalAnalysisPlanRecord) -> LogicalAnalysisPlanWire {
    LogicalAnalysisPlanWire {
        plan_id: p.plan_id.to_string(),
        data_classification: match p.data_classification {
            DataClassification::Tabular => "tabular",
            DataClassification::Temporal => "temporal",
            DataClassification::Panel => "panel",
            DataClassification::MultiEnvironment => "multi_environment",
            DataClassification::Event => "event",
        }
        .into(),
        discovery_algorithm: p.discovery_algorithm.as_ref().map(|s| s.to_string()),
        graph_review_required: p.graph_review_required,
        identifier: p.identifier.as_ref().map(|s| s.to_string()),
        estimator: p.estimator.as_ref().map(|s| s.to_string()),
        validation_suite: p.validation_suite.as_ref().map(|s| s.to_string()),
        query_variables: vars_to_raw(&p.query_variables),
    }
}

/// Decode logical plan.
///
/// # Errors
///
/// Unknown classification.
pub fn logical_plan_from_wire(w: &LogicalAnalysisPlanWire) -> Result<LogicalAnalysisPlanRecord, IoError> {
    Ok(LogicalAnalysisPlanRecord {
        plan_id: Arc::from(w.plan_id.as_str()),
        data_classification: match w.data_classification.as_str() {
            "tabular" => DataClassification::Tabular,
            "temporal" => DataClassification::Temporal,
            "panel" => DataClassification::Panel,
            "multi_environment" => DataClassification::MultiEnvironment,
            "event" => DataClassification::Event,
            other => {
                return Err(IoError::Convert(format!("unknown DataClassification `{other}`")));
            }
        },
        discovery_algorithm: w.discovery_algorithm.as_ref().map(|s| Arc::from(s.as_str())),
        graph_review_required: w.graph_review_required,
        identifier: w.identifier.as_ref().map(|s| Arc::from(s.as_str())),
        estimator: w.estimator.as_ref().map(|s| Arc::from(s.as_str())),
        validation_suite: w.validation_suite.as_ref().map(|s| Arc::from(s.as_str())),
        query_variables: vars_from_raw(&w.query_variables),
    })
}

fn mat_to_str(m: BufferMaterialization) -> &'static str {
    match m {
        BufferMaterialization::Borrowed => "borrowed",
        BufferMaterialization::CopiedContiguous => "copied_contiguous",
        BufferMaterialization::Transposed => "transposed",
        BufferMaterialization::Chunked => "chunked",
    }
}

fn mat_from_str(s: &str) -> Result<BufferMaterialization, IoError> {
    Ok(match s {
        "borrowed" => BufferMaterialization::Borrowed,
        "copied_contiguous" => BufferMaterialization::CopiedContiguous,
        "transposed" => BufferMaterialization::Transposed,
        "chunked" => BufferMaterialization::Chunked,
        other => {
            return Err(IoError::Convert(format!("unknown BufferMaterialization `{other}`")));
        }
    })
}

fn kernel_to_str(k: KernelSelection) -> &'static str {
    match k {
        KernelSelection::Scalar => "scalar",
        KernelSelection::PortableOptimized => "portable_optimized",
        KernelSelection::ArchSimd => "arch_simd",
        KernelSelection::DenseBackend => "dense_backend",
    }
}

fn kernel_from_str(s: &str) -> Result<KernelSelection, IoError> {
    Ok(match s {
        "scalar" => KernelSelection::Scalar,
        "portable_optimized" => KernelSelection::PortableOptimized,
        "arch_simd" => KernelSelection::ArchSimd,
        "dense_backend" => KernelSelection::DenseBackend,
        other => return Err(IoError::Convert(format!("unknown KernelSelection `{other}`"))),
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
            .map(|(n, m)| (n.to_string(), mat_to_str(*m).into()))
            .collect(),
        kernels: p
            .kernels
            .iter()
            .map(|(n, k)| (n.to_string(), kernel_to_str(*k).into()))
            .collect(),
        batch_size: p.batch_size.map(|b| u64::try_from(b).unwrap_or(u64::MAX)),
        workspace_bytes: p.workspace_bytes,
        estimated_peak_memory_bytes: p.estimated_peak_memory_bytes,
        estimated_copy_bytes: p.estimated_copy_bytes,
        task_schedule: p
            .task_schedule
            .iter()
            .map(|t| (t.dimension.to_string(), t.units))
            .collect(),
        worker_threads: p.worker_threads,
        deterministic_reductions: p.deterministic_reductions,
        expected_python_crossings: p.expected_python_crossings,
    }
}

/// Decode physical plan.
///
/// # Errors
///
/// Unknown materialization/kernel tags.
pub fn physical_plan_from_wire(
    w: &PhysicalExecutionPlanWire,
) -> Result<PhysicalExecutionPlanRecord, IoError> {
    Ok(PhysicalExecutionPlanRecord {
        plan_id: Arc::from(w.plan_id.as_str()),
        materializations: w
            .materializations
            .iter()
            .map(|(n, m)| Ok((Arc::from(n.as_str()), mat_from_str(m)?)))
            .collect::<Result<Vec<_>, IoError>>()?
            .into(),
        kernels: w
            .kernels
            .iter()
            .map(|(n, k)| Ok((Arc::from(n.as_str()), kernel_from_str(k)?)))
            .collect::<Result<Vec<_>, IoError>>()?
            .into(),
        batch_size: w.batch_size.map(|b| usize::try_from(b).unwrap_or(usize::MAX)),
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
        wall_time_ns: p.wall_time_ns,
        peak_rss_bytes: p.peak_rss_bytes,
        copy_count: p.copy_count,
        scalar_fallback_count: p.scalar_fallback_count,
    }
}

/// Decode performance.
#[must_use]
pub fn performance_from_wire(w: &ExecutionPerformanceWire) -> ExecutionPerformanceRecord {
    ExecutionPerformanceRecord {
        wall_time_ns: w.wall_time_ns,
        peak_rss_bytes: w.peak_rss_bytes,
        copy_count: w.copy_count,
        scalar_fallback_count: w.scalar_fallback_count,
    }
}

/// Silence unused import when only re-exported.
#[allow(dead_code)]
fn _keep_variable_id(_: VariableId) {}

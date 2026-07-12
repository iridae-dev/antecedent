//! Core types shared across the causal-library workspace.
//!
//! `causal-core` owns identifiers, schemas, assumptions, provenance,
//! diagnostics, errors, and execution policy. It must not depend on numerical,
//! graph-algorithm, Arrow, or Python crates (DESIGN.md §3.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod assumption;
pub mod diagnostic;
pub mod error;
pub mod execution;
pub mod ids;
pub mod plan;
pub mod provenance;
pub mod schema;

pub use assumption::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ParametricAssumption, PriorAssumption,
};
pub use diagnostic::{Diagnostic, DiagnosticKind, DiagnosticSet, DiagnosticSeverity};
pub use error::SchemaError;
pub use execution::{
    CachePolicy, CancellationToken, CausalRng, Determinism, ExecutionContext, KernelPolicy,
    MemoryBudget, NonZeroThreadCount, Parallelism, ProgressSink, RngFactory,
};
pub use ids::{CategoryDomainId, EnvironmentId, Lag, RegimeId, VariableId};
pub use plan::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord,
};
pub use provenance::{ArtifactId, ProvenanceGraph, ProvenanceNode};
pub use schema::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, RoleHint, ScalarType, SmallRoleSet,
    ValueType, VariableSchema,
};

/// Library crate version string from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_semver_like() {
        assert!(super::VERSION.contains('.'));
    }
}

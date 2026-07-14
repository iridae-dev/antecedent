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
pub mod identification;
pub mod ids;
pub mod intervention;
pub mod plan;
pub mod provenance;
pub mod query;
pub mod schema;
pub mod temporal;
pub mod tolerance;
pub mod value;

pub use assumption::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ParametricAssumption, PriorAssumption,
};
pub use diagnostic::{Diagnostic, DiagnosticKind, DiagnosticSet, DiagnosticSeverity};
pub use error::SchemaError;
pub use execution::{
    CacheBudget, CachePolicy, CancellationToken, CausalRng, Determinism, ExecutionContext,
    KernelPolicy, MemoryBudget, MonteCarloBudget, MonteCarloError, NonZeroThreadCount, Parallelism,
    ProgressSink, RngFactory,
};
pub use identification::IdentificationStatus;
pub use ids::{
    CategoryDomainId, ComponentId, EnvironmentId, Lag, ModelId, QueryId, RegimeId, StateVersion,
    VariableId,
};
pub use intervention::{
    Intervention, InterventionError, InterventionSequence, MechanismOverride,
    SequencedIntervention, StochasticPolicy, TemporalPolicy,
};
pub use plan::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
pub use provenance::{ArtifactId, ProvenanceGraph, ProvenanceNode};
pub use query::{
    AllocationMethod, AnomalyAttributionQuery, AttributionComponents, AverageEffectQuery,
    CausalQuery, ChangeAttributionQuery, ConditionalEffectQuery, CounterfactualQuery,
    MechanismChangeQuery, MediationContrast, MediationQuery, OrderedFloatBits, PopulationSelector,
    QueryError, ShapleyConfig, ShapleyMode, TargetPopulation, TemporalEffectQuery, UnitChangeQuery,
};
pub use schema::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, RoleHint, ScalarType, SmallRoleSet,
    ValueType, VariableSchema,
};
pub use temporal::{TemporalIndexError, TemporalIndexer, TemporalNodeKey};
pub use tolerance::ToleranceClass;
pub use value::Value;

/// Library crate version string from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_semver_like() {
        assert!(super::VERSION.contains('.'));
    }
}

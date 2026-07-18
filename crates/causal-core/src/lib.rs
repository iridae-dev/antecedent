//! Core types shared across the causal-library workspace.
//!
//! `causal-core` owns identifiers, schemas, assumptions, provenance,
//! diagnostics, errors, and execution policy. It must not depend on numerical,
//! graph-algorithm, Arrow, or Python crates (DESIGN.md §3.1).
//!
//! # Names at the boundary, IDs on the hot path
//!
//! Human-readable names live in [`CausalSchema`]. Hot-path APIs take
//! [`VariableId`] values resolved from that schema — never raw strings.
//!
//! ```
//! use causal_core::{
//!     AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
//!     ValueType, VariableId,
//! };
//!
//! let mut b = CausalSchemaBuilder::new();
//! b.add_variable(
//!     "treatment",
//!     ValueType::Continuous,
//!     SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
//!     None,
//!     None,
//!     MeasurementSpec::default(),
//! )
//! .unwrap();
//! b.add_variable(
//!     "outcome",
//!     ValueType::Continuous,
//!     SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
//!     None,
//!     None,
//!     MeasurementSpec::default(),
//! )
//! .unwrap();
//! let schema = b.build().unwrap();
//! let t = schema.id_of("treatment").unwrap();
//! let y = schema.id_of("outcome").unwrap();
//! let query = AverageEffectQuery::binary_ate(t, y);
//! assert_eq!(query.treatment, VariableId::from_raw(0));
//! ```
//!
//! Parallelism, budgets, and RNG seeding are configured via [`ExecutionContext`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod assumption;
pub mod diagnostic;
pub mod error;
pub mod execution;
pub mod identification;
pub mod ids;
pub mod intervention;
pub mod node;
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
pub use node::NodeRef;
pub use plan::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
pub use provenance::{ArtifactId, ProvenanceGraph, ProvenanceNode};
pub use query::{
    AllocationMethod, AnomalyAttributionQuery, AttributionComponents, AverageEffectQuery,
    CausalQuery, ChangeAttributionQuery, ConditionalEffectQuery, CounterfactualQuery,
    InterventionalDistributionQuery, MechanismChangeQuery, MediationContrast, MediationQuery,
    OrderedFloatBits, PathSpecificEffectQuery, PopulationSelector, QueryError, ShapleyConfig,
    ShapleyMode, TargetPopulation, TemporalEffectQuery, UnitChangeQuery,
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

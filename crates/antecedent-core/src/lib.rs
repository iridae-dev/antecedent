//! Core types shared across the Antecedent workspace.
//!
//! `antecedent-core` owns identifiers, schemas, assumptions, provenance,
//! diagnostics, errors, and execution policy. It must not depend on numerical,
//! graph-algorithm, Arrow, or Python crates.
//!
//! # Names at the boundary, IDs on the hot path
//!
//! Human-readable names live in [`CausalSchema`]. Hot-path APIs take
//! [`VariableId`] values resolved from that schema — never raw strings.
//!
//! ```
//! use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, VariableId};
//!
//! let schema = CausalSchemaBuilder::new()
//!     .continuous("treatment")
//!     .treatment()
//!     .continuous("outcome")
//!     .outcome()
//!     .build()
//!     .unwrap();
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
pub mod response;
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
    AdaptiveBootstrapBudget, AdaptiveDrawBudget, CacheBudget, CachePolicy, CancellationToken,
    CausalRng, Determinism, ExecutionContext, KernelPolicy, MemoryBudget, MonteCarloBudget,
    MonteCarloError, NonZeroThreadCount, Parallelism, ProgressSink, RngFactory,
};
pub use identification::IdentificationStatus;
pub use ids::{
    CategoryDomainId, ComponentId, DistributionRef, DynamicRuleId, EnvironmentId, Lag, ModelId,
    QueryId, RegimeId, StateVersion, VariableId,
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
    AllocationMethod, AnomalyAttributionQuery, AssignmentDesign, AttributionComponents,
    AverageEffectQuery, CausalQuery, ChangeAttributionQuery, ConditionalEffectQuery,
    ContinuousDomain, CounterfactualQuery, DerivativeScale, DerivativeWeighting,
    EXPOSURE_LEVEL_TOLERANCE, ExposureLevel, ExposureMapping, GridSpec, InterferenceFunctional,
    InterferenceQuery, InterventionalDistributionQuery, MAX_NONPARAMETRIC_RESPONSE_DIM,
    MAX_TEMPORAL_RESPONSE_HORIZONS, MechanismChangeQuery, MediationContrast, MediationQuery,
    ObservationAssumption, ObservationSpec, OrderedFloatBits, PathSpecificEffectQuery,
    PopulationRegistry, PopulationSelection, PopulationSelector, PredicateExpr, QueryError,
    ResponseFunctional, ResponseQuery, ShapleyConfig, ShapleyMode, TargetPopulation,
    TemporalEffectQuery, TemporalResponseSpec, TransportQuery, UnitChangeQuery,
};
pub use response::{
    CausalResponse, IdentifiedSet, ResponseEnvelope, ResponseIdentification, ResponseUncertainty,
    ResponseValue, SupportDiagnostic, SupportRegion, SupportReport, SupportStatus,
};
pub use schema::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, RoleHint, ScalarType, SmallRoleSet,
    ValueType, VariableInProgress, VariableSchema,
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

//! Declarations a study is assembled from: assumptions, identity primitives, diagnostics, ids and
//! the supporting records of the foundation crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Match key and scope facts of one reported interval.
pub use antecedent_core::CalibrationBasis;
/// Shared identification status vocabulary.
pub use antecedent_core::IdentificationStatus;
/// Schema construction errors for `antecedent-core`.
pub use antecedent_core::SchemaError;
/// `antecedent-core` items at the crate root.
pub use antecedent_core::VERSION;
/// Typed causal assumptions with source and scope.
pub use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, PriorAssumption,
};
/// Domain-separated semantic identities for compiled causal contracts.
pub use antecedent_core::{
    CanonicalWriter, ContractIdentities, IdentityDomain, blake3_hex, digest_from_hex, to_hex,
};
/// `antecedent-core` items at the crate root.
pub use antecedent_core::{
    CategoryDomainId, DeclaredSamplingDesign, DistributionRef, EnvironmentId, ModelId, QueryId,
    RegimeId,
};
/// Compact, copyable identifiers used throughout Antecedent.
pub use antecedent_core::{ComponentId, StateVersion};
/// Structured diagnostics.
pub use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
/// Causal variable schemas and role hints.
pub use antecedent_core::{MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
/// Shared compiler and source-replay contract for one categorical candidate.
pub mod categorical_candidate_contract {
    pub use antecedent_core::categorical_candidate_contract::CategoricalCandidateSource;
}
/// Structured diagnostics.
pub mod diagnostic {
    pub use antecedent_core::diagnostic::RouteNativeDiagnostic;
}
/// Shared compiler and source-replay contract for the graph-selection imputation candidate.
pub mod graph_imputation_contract {
    pub use antecedent_core::graph_imputation_contract::{MAX_VARIABLES, MIN_ROWS};
}
/// Domain-separated semantic identities for compiled causal contracts.
pub mod identity {
    pub use antecedent_core::identity::to_hex;
}
/// Typed measurement models: how an observed indicator relates to the scientific quantity it.
pub mod measurement_model {
    pub use antecedent_core::measurement_model::CorrectedFunctionalRecord;
}
/// Native source for the declared-shift MNAR calibration candidate.
pub mod mnar_fixed_shift {
    pub use antecedent_core::mnar_fixed_shift::{
        FixedShiftSource, SamplingPremise, SourceError, SourceRow,
    };
}

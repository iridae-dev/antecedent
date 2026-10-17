//! Unified static/temporal `Study` facade (identify → estimate → refute).
//!
//! Antecedent is an identification-first causal inference engine: identification
//! is evaluated before estimation, refuters run against the estimate, and
//! uncertainty about causal structure (equivalence classes, graph posteriors) is
//! retained rather than silently resolved.
//!
//! # Quick start
//!
//! ```
//! use antecedent::prelude::*;
//! use antecedent::RefuteSuite;
//!
//! let schema = CausalSchemaBuilder::new()
//!     .continuous("t")
//!     .treatment()
//!     .continuous("y")
//!     .outcome()
//!     .continuous("z")
//!     .context()
//!     .build()
//!     .unwrap();
//! let data = TabularData::from_f64_columns([
//!     ("t", &[0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0][..]),
//!     ("y", &[1.0_f64, 3.1, 1.2, 2.9, 0.9, 3.0, 1.1, 3.2][..]),
//!     ("z", &[0.0_f64, 1.0, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7][..]),
//! ])
//! .unwrap();
//! let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
//! let q = AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
//! let result = Study::tabular(data)
//!     .graph(dag)
//!     .query(q)
//!     .refute(RefuteSuite::None)
//!     .bootstrap_replicates(0)
//!     .build()
//!     .unwrap()
//!     .run(&ExecutionContext::for_tests(1))
//!     .unwrap();
//! assert!(result.effect().is_finite());
//! ```
//!
//! Prefer [`prelude`] for day-1 imports. Component crates remain available for stage-specific work.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod accepted;
pub mod analysis;
pub mod callback_plan;
pub mod class_prior;
pub mod counterfactual_id;
pub mod cross_world;
pub mod design;
pub mod discovery;
pub mod discovery_defaults;
pub mod error;
pub mod estimator_spec;
pub mod gcm;
mod graph_identity;
pub mod inference;
pub mod options;
pub mod planner;
pub mod prelude;
pub mod result;
pub mod review;
pub mod state;
pub mod strategy_table;
pub mod support;
pub use antecedent_io::coverage_records_data;

#[path = "analysis/surface/estimate.rs"]
pub mod estimate;
#[path = "analysis/surface/graph.rs"]
pub mod graph;
#[path = "analysis/surface/identify.rs"]
pub mod identify;
pub mod identify_api;
#[path = "analysis/surface/io.rs"]
pub mod io;
#[path = "analysis/surface/query.rs"]
pub mod query;
#[path = "analysis/surface/scientific.rs"]
pub mod scientific;
#[path = "analysis/surface/validate.rs"]
pub mod validate;

// --- Day-1 crate-root surface (stage depth lives under modules) ---
pub use accepted::{AcceptedGraph, GraphClass, GraphRef, IntoAccepted};
pub use analysis::api::*;
#[doc(hidden)]
pub use analysis::temporal_dependent_interval_candidate;
#[doc(hidden)]
pub use analysis::{max_t_critical_value, simultaneous_band_unpublished};
pub use class_prior::ClassPrior;
pub use error::{CausalError, CompileCause, CompileFailure, ReviewKind};
pub use estimate::{CausalPosterior, EffectEstimate, EstimatorId, IdentifierId};
pub use estimator_spec::EstimatorSpec;
pub use graph::{Dag, DenseNodeId, TemporalDag};
pub use identify_api::{Identification, identify, identify_dag, identify_with};
pub use inference::{BayesianConfig, InferenceMode, PosteriorDrawControls};
pub use options::FdrControl;
pub use query::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExposureLevel, ExposureMapping, GridSpec,
    InterferenceFunctional, InterferenceQuery, InterventionalDistributionQuery,
    LocalPolynomialRatioQuery, LongitudinalRegimeQuery, MechanismChangeQuery, MediationContrast,
    MediationQuery, OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, RandomizationDesign,
    RandomizedEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SyntheticControlQuery, SyntheticPanelMethod,
    TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, TransportQuery, UnitChangeQuery,
};
pub use result::{
    AnalysisIdentification, ContinuousDoseResponseEstimate, ExecutedContract,
    LocalPolynomialRatioEstimate, LongitudinalRegimeEstimate, PERCENTILE_95_MIN_REPLICATES,
    PanelDidEstimate, PolicyValueEstimate, PrimaryEstimate, PublishedScalarUncertainty,
    RandomizedEffectEstimate, ResultPayload, RowWeightsBinding, StructuralAggregationPolicy,
    StructuralWeightBasis, StudyResult, SurvivalEstimate, SyntheticControlEstimate,
    SyntheticDidEstimate,
};
pub use support::{
    CellStatus, IntoGraphInput, StructureSource, SupportCell, SupportGraphClass, SupportInference,
    SupportQuery, SupportRefusal, SupportStructure, SupportValidation, cell_coordinate, classify,
    licensed_neighbors, licensed_support_cells, refused_message, support_cell_from_coordinate,
};

// Strategy helpers and peer APIs: use `antecedent::estimate`, `antecedent::discovery`,
// `antecedent::gcm`, `antecedent::io`, `antecedent::design`, `antecedent::state`, `antecedent::graph`,
// `antecedent::identify`, `antecedent::review`, `antecedent::validate`, `antecedent::planner`.

#[cfg(test)]
mod tests;

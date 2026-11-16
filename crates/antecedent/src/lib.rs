//! Identification-first causal inference: declare what you assume, ask a typed question, and
//! get back an answer that states what it rests on, or a refusal that states why not.
//!
//! `antecedent` is the facade crate of the Antecedent library. It decides whether a causal
//! question is identified under a declared structure before it estimates anything, runs
//! refuters against the estimate, and keeps uncertainty about the structure itself
//! (equivalence classes, graph posteriors) instead of resolving it silently. The component
//! crates (`antecedent-graph`, `antecedent-identify`, `antecedent-estimate` and the others)
//! are reachable through the modules below; most programs need only this crate.
//!
//! # The lifecycle of a study
//!
//! 1. Declare a schema with [`CausalSchemaBuilder`](prelude::CausalSchemaBuilder): each
//!    variable, its type and its role (treatment, outcome, context).
//! 2. Supply data in the shape the question needs: [`TabularData`](prelude::TabularData), a
//!    time series, a panel, events or several environments.
//! 3. State the structure you assume: a [`Dag`], a CPDAG, ADMG or PAG from [`graph`], a
//!    [`TemporalDag`], or a discovered structure accepted through [`AcceptedGraph`].
//! 4. Ask a typed question from [`query`], such as an [`AverageEffectQuery`].
//! 5. Assemble a [`Study`] with [`StudyBuilder`] and choose how to execute it:
//!    [`Study::run`] returns a [`StudyResult`]; [`Study::prepare`] returns a
//!    [`PreparedStudy`] that re-estimates on new rows without repeating identification;
//!    [`Study::compile`] returns the plan without running it; and
//!    [`StudyBuilder::identify_structure`] decides identification alone, from the structure and
//!    the query, without reading data.
//! 6. Read the result in order: the identification (is the question answered by a functional
//!    of the observed distribution, and under which assumptions), the estimate, its
//!    uncertainty and the calibration status of the interval, the refutation reports, and
//!    [`StudyResult::support_status`]. [`StudyResult::method_provenance`] and [`bibliography`]
//!    name the published methods behind the route that ran.
//!
//! # Quick start
//!
//! The example declares that `z` causes both `t` and `y`, asks for the average effect of `t`
//! on `y`, and runs with refutation and the bootstrap switched off, so the result is a point
//! estimate with no interval.
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
//! `use antecedent::prelude::*` brings in the builder, schema, data, graph, query, result and
//! error types used above. The runnable examples in the repository's `examples/rust/`
//! directory cover the other query families; run one with
//! `cargo run -p antecedent --example ate_quickstart`.
//!
//! # Refusals and errors
//!
//! The [`Study`] entry points return [`CausalError`]; routes with their own preparation (in
//! [`analysis`] and the other modules) return a typed refusal of their own. A refusal is a
//! deliberate answer, not a malfunction: the question is not identified under the declared
//! structure, the data cannot support it, or the route is not supported. A refusal carries a
//! registered reason code ([`CausalError::reason_code`]), and
//! [`CausalError::is_identification_refusal`] separates "not identified" from budgets,
//! cancellations and invalid arguments. [`reason`] holds the reason-code registry, [`refusal`]
//! typed constructors for registered refusals, and [`host`] maps an error to a stable family
//! and a machine-readable record. The reason codes are listed in the refusal-code reference of
//! the documentation.
//!
//! Which combinations of structure, query and inference are supported is a public lookup in
//! [`support`]. A release build refuses a `not_supported` cell with a typed error that names
//! the missing evidence; the `development` feature runs it instead and lists the refusal it
//! would have raised in [`StudyResult::would_refuse`]. Before using a number, check that
//! [`StudyResult::support_status`] is [`CellStatus::Supported`](support::CellStatus::Supported).
//!
//! # Modules
//!
//! Start with [`prelude`]; the most-used types are also re-exported at the crate root.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`prelude`] | The imports a first study needs. |
//! | [`analysis`] | The `Study` builder and the analysis routes it compiles to. |
//! | [`schema`] | Declarations a study is assembled from: assumptions, identity primitives, diagnostics and ids. |
//! | [`data`] | Tabular, panel, time-series, event and network data, typed columns, missingness and the Arrow boundary. |
//! | [`graph`] | Causal graph types (DAG, CPDAG, ADMG, PAG and temporal forms), edge marks and d-separation. |
//! | [`accepted`] | [`AcceptedGraph`]: a reviewed graph, with its acceptance recorded in its type. |
//! | [`review`] | Edits that resolve a discovered structure before it is accepted. |
//! | [`discovery`] | Parameters, results and engines of the compiled structure-discovery route. |
//! | [`discovery_defaults`] | Default constraints of the structure searches and conditional-independence test names. |
//! | [`query`] | Typed causal queries. |
//! | [`identify`] | Identification results, estimands, and the transport and recovery identifiers. |
//! | [`identify_api`] | [`Identification`]: identification from structure and query without data. |
//! | [`counterfactual_id`] | Identification of the effect of treatment on the treated on a bounded ADMG. |
//! | [`cross_world`] | Cross-world edge contrasts on a fixed Markovian DAG. |
//! | [`estimate`] | Estimators, estimation results and strategy identifiers. |
//! | [`estimator_spec`] | [`EstimatorSpec`]: the estimator a study uses, by id or fully configured. |
//! | [`strategy_table`] | Identifier and estimator ids, defaults and estimand compatibility rules. |
//! | [`inference`] | Inference mode and the Bayesian configuration of a study. |
//! | [`class_prior`] | Caller-supplied mass over the members of an incomplete temporal class. |
//! | [`design_posterior_model`] | Declared models of the randomized-design posterior routes. |
//! | [`validate`] | Refutation, sensitivity and diagnostic checks of an estimate. |
//! | [`gcm`] | Graphical causal models: fit mechanisms, then sample, compute counterfactuals and attribute. |
//! | [`scientific`] | Scientific declaration types that public routes take and return. |
//! | [`design`] | Experiment and measurement design: which study, measurement or policy to run next. |
//! | [`decision_evidence`] | Experimental evidence for decision-relevant outcomes. |
//! | [`result`] | [`StudyResult`] and the typed estimates of each query family. |
//! | [`report`] | A readable Markdown or HTML report rendered from a contract. |
//! | [`contract`] | Claim, provider and obligation records: what a result may claim and what remains owed. |
//! | [`support`] | The public support-matrix lookup. |
//! | [`error`] | [`CausalError`] and its refusal vocabulary. |
//! | [`reason`] | Registered reason codes and the structured details attached to refusals. |
//! | [`refusal`] | Typed constructors for the registered refusals a host raises itself. |
//! | [`declared`] | Declared names parsed into typed values, with unsupported names refused. |
//! | [`host`] | The contract a host application reads to tell a supported claim from a refusal. |
//! | [`identity`] | Digests and identity helpers for hosts. |
//! | [`io`] | Graph interchange and durable, versioned artifacts with replay. |
//! | [`execution`] | Callbacks, cancellation, budgets, deterministic randomness and admission. |
//! | [`planner`] | The logical and physical plans a study compiles to, and their inspection. |
//! | [`callback_plan`] | Marking of a plan that calls back into a host language. |
//! | [`state`] | Incremental state: keep published results consistent with changing data. |
//! | [`retained_refresh`] | Refresh of the least-squares or scatter summary a prepared study rests on. |
//! | [`period_refresh`] | Refresh of an autoregressive mechanism onto a new period. |
//! | [`bootstrap_retention`] | Retention, extension and spilling of bootstrap replicates. |
//!
//! # Feature flags
//!
//! | Feature | Default | Effect |
//! | --- | --- | --- |
//! | `file-source` | yes | Parquet and Arrow IPC table files. |
//! | `ml-gbdt` | no | Gradient-boosted tree learners (`forust-ml`). |
//! | `ml-forest` | no | Random-forest learners (`smartcore`). |
//! | `ml-full` | no | `ml-gbdt` and `ml-forest`. |
//! | `ml-neural` | no | The neural learner on the CPU backend. |
//! | `ml-neural-wgpu` | no | `ml-neural` with the WGPU accelerator backend. |
//! | `development` | no | Runs `not_supported` routes and lists what they would refuse. Never for release use. |
//! | `calibration-internal` | no | Exposes engines to the calibration suites. Not a user feature. |
//!
//! The learner features add estimators; they grant no new scientific claims.
//!
//! # Further reading
//!
//! Tutorials, guides, the support matrix and the refusal-code reference are at
//! <https://antecedent.readthedocs.io/>.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod accepted;
pub mod analysis;
pub mod bootstrap_retention;
pub mod callback_plan;
pub mod class_prior;
pub mod contract;
pub mod counterfactual_id;
pub mod cross_world;
pub mod data;
pub mod decision_evidence;
pub mod declared;
pub mod design;
pub mod design_posterior_model;
pub mod discovery;
pub mod discovery_defaults;
pub mod error;
pub mod estimator_spec;
pub mod execution;
pub mod gcm;
mod graph_identity;
pub mod host;
pub mod identity;
pub mod inference;
pub mod period_refresh;
pub mod planner;
pub mod prelude;
pub mod reason;
pub mod refusal;
pub mod report;
pub mod result;
pub mod retained_refresh;
pub mod review;
pub mod schema;
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
pub use accepted::{AcceptedGraph, GraphClass, GraphRef, IntoAccepted, StructureProvenance};
pub use analysis::api::*;
#[doc(hidden)]
pub use analysis::max_t_critical_value;
pub use analysis::measured_exposure;
#[doc(hidden)]
pub use analysis::temporal_dependent_interval_candidate;
pub use antecedent_estimate::ScoreDependence;
pub use antecedent_estimate::{IvBoundsEstimate, IvBoundsInterval, IvBoundsPosterior};
pub use class_prior::ClassPrior;
pub use design_posterior_model::DesignPosteriorModel;
pub use error::{
    CausalError, CompileCause, CompileFailure, IDENTIFICATION_REFUSAL_CODES, ReviewKind,
};
pub use estimate::{CausalPosterior, EffectEstimate, EstimatorId, IdentifierId};
pub use estimator_spec::EstimatorSpec;
pub use graph::{Dag, DenseNodeId, TemporalDag};
pub use identify_api::Identification;
pub use inference::{BayesianConfig, InferenceMode, PosteriorDrawControls};
pub use query::IvBoundsQuery;
pub use query::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExposureLevel, ExposureMapping, GridSpec,
    HistoryMatchQuery, InterferenceFunctional, InterferenceQuery, InterventionalDistributionQuery,
    LocalPolynomialRatioQuery, LongitudinalRegimeQuery, MechanismChangeQuery, MediationContrast,
    MediationQuery, OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, RandomizationDesign,
    RandomizedEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SequentialContrastQuery, SyntheticControlQuery,
    SyntheticPanelMethod, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec,
    TransportQuery, UnitChangeQuery,
};
pub use query::{AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery};
pub use query::{AttributeQuery, DiscoverQuery, SimulateQuery};
pub use query::{
    OutcomeMisclassification, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget, TemporalRegimeMethod,
};
pub use result::QuasiRolloutEstimate;
pub use result::TemporalDesignEstimate;
pub use result::bibliography;
pub use result::{
    AnalysisIdentification, ContinuousDoseResponseEstimate, ExecutedContract, HistoryMatchEstimate,
    LocalPolynomialRatioEstimate, LongitudinalRegimeEstimate, ObservedWindowReport,
    PERCENTILE_95_MIN_REPLICATES, PanelDidEstimate, PolicyValueEstimate, PosteriorRowEstimate,
    PrimaryEstimate, PublishedScalarUncertainty, RandomizedEffectEstimate, ResultPayload,
    RowWeightsBinding, SequentialContrastEstimate, StructuralAggregationPolicy,
    StructuralWeightBasis, StudyResult, SurvivalEstimate, SyntheticControlEstimate,
    SyntheticDidEstimate,
};
pub use support::{
    CellStatus, IntoGraphInput, StructureSource, SupportCell, SupportGraphClass, SupportInference,
    SupportQuery, SupportRefusal, SupportStructure, SupportValidation, cell_coordinate, classify,
    refused_message, support_cell_from_coordinate, supported_cells, supported_neighbors,
};

// Strategy helpers and peer APIs: use `antecedent::estimate`, `antecedent::discovery`,
// `antecedent::gcm`, `antecedent::io`, `antecedent::design`, `antecedent::state`, `antecedent::graph`,
// `antecedent::identify`, `antecedent::review`, `antecedent::validate`, `antecedent::planner`.

#[cfg(test)]
mod tests;

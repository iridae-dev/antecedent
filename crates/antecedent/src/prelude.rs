//! Day-1 imports for the `antecedent` facade.
//!
//! ```rust,ignore
//! use antecedent::prelude::*;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use crate::analysis::{
    ComputeBudget, LatencyMode, PreparedStudy, RdConfig, RefuteSuite, Study, StudyBuilder,
};
pub use crate::error::CausalError;
pub use crate::inference::{BayesianConfig, InferenceMode};
pub use crate::options::{DiscoveryAccept, FdrControl};
pub use crate::planner::{CompiledAnalysis, GraphInput};
pub use crate::result::StudyResult;
pub use crate::strategy_table::{EstimatorId, IdentifierId};

pub use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, CausalSchemaBuilder, ExecutionContext,
    Intervention, TemporalEffectQuery, Value, VariableId,
};
pub use antecedent_data::{
    EventData, MultiEnvironmentData, PanelData, PanelUnit, TabularData, TimeSeriesData,
};
pub use antecedent_estimate::{CausalPosterior, EffectEstimate};
pub use antecedent_expr::IdentifiedEstimand;
pub use antecedent_graph::{Dag, DenseNodeId, TemporalDag};
pub use antecedent_identify::IdentificationResult;

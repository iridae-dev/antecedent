//! Day-1 imports for the `antecedent` facade.
//!
//! ```rust,ignore
//! use antecedent::prelude::*;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use crate::analysis::{
    CausalAnalysis, CausalAnalysisBuilder, ComputeBudget, LatencyMode, PreparedAnalysis, RdConfig,
    RefuteSuite,
};
pub use crate::error::CausalError;
pub use crate::inference::{BayesianConfig, InferenceMode};
pub use crate::options::{DiscoveryAccept, FdrControl};
pub use crate::planner::{CompiledAnalysis, GraphInput};
pub use crate::result::CausalAnalysisResult;
pub use crate::strategy_table::{EstimatorId, IdentifierId};

pub use causal_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, CausalSchemaBuilder, ExecutionContext,
    Intervention, TemporalEffectQuery, Value, VariableId,
};
pub use causal_data::{
    EventData, MultiEnvironmentData, PanelData, PanelUnit, TabularData, TimeSeriesData,
};
pub use causal_estimate::{CausalPosterior, EffectEstimate};
pub use causal_expr::IdentifiedEstimand;
pub use causal_graph::{Dag, DenseNodeId, TemporalDag};
pub use causal_identify::IdentificationResult;

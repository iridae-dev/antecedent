//! Day-1 imports for the `causal` facade.
//!
//! ```rust,ignore
//! use causal::prelude::*;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use crate::analysis::{CausalAnalysis, CausalAnalysisBuilder, RdConfig, RefuteSuite};
pub use crate::error::{AnalysisError, CausalError};
pub use crate::inference::{BayesianConfig, InferenceMode};
pub use crate::options::{DiscoveryAccept, FdrControl};
pub use crate::planner::{CompiledAnalysis, GraphInput};
pub use crate::result::CausalAnalysisResult;
pub use crate::strategy_table::{EstimatorId, IdentifierId};

pub use causal_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ExecutionContext, TemporalEffectQuery,
    VariableId,
};
pub use causal_data::{TabularData, TimeSeriesData};
pub use causal_graph::{Dag, TemporalDag};
pub use causal_estimate::CausalPosterior;

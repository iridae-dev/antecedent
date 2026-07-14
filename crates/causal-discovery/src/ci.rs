//! Conditional-independence surface owned by discovery (DESIGN.md §3.1 / §12).
//!
//! Numeric kernels remain in `causal-stats`; this module re-exports the DESIGN
//! trait contract so discovery algorithms depend on a discovery-owned CI API.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use causal_stats::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, GSquared, Gpdc, KnnCmi,
    MixedKnnCmi, MultivariatePartialCorrelation, OracleCi, PartialCorrelation, PreparedCiTest,
    RegressionCi, RobustPartialCorrelation, SignificanceMethod, SymbolicCmi,
    WeightedPartialCorrelation, ci_from_name,
};

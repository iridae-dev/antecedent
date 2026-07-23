//! Conditional-independence surface owned by discovery.
//!
//! Numeric kernels remain in `antecedent-stats`; this module re-exports the DESIGN
//! trait contract so discovery algorithms depend on a discovery-owned CI API.
//! Concrete test constructors live in `antecedent-stats` (use
//! [`ci_from_name`] or import the type from that crate).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

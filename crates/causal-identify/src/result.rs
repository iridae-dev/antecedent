//! Identification result types (DESIGN.md §10.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic};
use causal_expr::CausalExprArena;

pub use causal_expr::IdentifiedEstimand;

/// Status of an identification attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationStatus {
    /// Nonparametrically identified.
    NonparametricallyIdentified,
    /// Identified only under a proper subset of the model class (partial ID).
    PartiallyIdentified,
    /// Identification depends on which graph in an equivalence class / ensemble.
    GraphDependent,
    /// Not identified.
    NotIdentified,
}

/// Step in a derivation trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivationStep {
    /// Rule applied.
    pub rule: Arc<str>,
    /// Detail.
    pub detail: Arc<str>,
}

/// Derivation trace for an identification result.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DerivationTrace {
    /// Ordered steps.
    pub steps: Vec<DerivationStep>,
}

impl DerivationTrace {
    /// Push a step.
    pub fn push(&mut self, rule: impl Into<Arc<str>>, detail: impl Into<Arc<str>>) {
        self.steps.push(DerivationStep { rule: rule.into(), detail: detail.into() });
    }
}

/// Performance record for identification.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IdentificationPerformanceRecord {
    /// Candidate sets examined.
    pub candidates_examined: u64,
    /// Adjustment sets returned.
    pub sets_returned: u64,
}

/// Full identification result.
#[derive(Clone, Debug)]
pub struct IdentificationResult {
    /// Status.
    pub status: IdentificationStatus,
    /// Query.
    pub query: CausalQuery,
    /// Estimands (may be empty if not identified).
    pub estimands: Vec<IdentifiedEstimand>,
    /// Expression arena owning functionals.
    pub arena: CausalExprArena,
    /// Derivation.
    pub derivation: DerivationTrace,
    /// Assumptions required.
    pub required_assumptions: AssumptionSet,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Performance.
    pub performance: IdentificationPerformanceRecord,
}

impl IdentificationResult {
    /// Primary average-effect query, if present.
    #[must_use]
    pub fn average_effect(&self) -> Option<&AverageEffectQuery> {
        match &self.query {
            CausalQuery::AverageEffect(q) => Some(q),
            _ => None,
        }
    }

    /// Nonparametrically identified result with estimands.
    #[must_use]
    pub fn identified(
        query: CausalQuery,
        estimands: Vec<IdentifiedEstimand>,
        arena: CausalExprArena,
        derivation: DerivationTrace,
        required_assumptions: AssumptionSet,
        performance: IdentificationPerformanceRecord,
    ) -> Self {
        Self {
            status: IdentificationStatus::NonparametricallyIdentified,
            query,
            estimands,
            arena,
            derivation,
            required_assumptions,
            diagnostics: Vec::new(),
            performance,
        }
    }

    /// Not-identified result (empty estimands / fresh arena).
    #[must_use]
    pub fn not_identified(
        query: CausalQuery,
        derivation: DerivationTrace,
        required_assumptions: AssumptionSet,
        performance: IdentificationPerformanceRecord,
    ) -> Self {
        Self {
            status: IdentificationStatus::NotIdentified,
            query,
            estimands: Vec::new(),
            arena: CausalExprArena::new(),
            derivation,
            required_assumptions,
            diagnostics: Vec::new(),
            performance,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::VariableId;
    use causal_expr::ExprId;

    use super::*;

    #[test]
    fn backdoor_roles_default_empty() {
        let e = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        assert_eq!(e.adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert!(e.instruments.is_empty());
        assert!(e.mediators.is_empty());
    }

    #[test]
    fn iv_and_frontdoor_constructors() {
        let iv = IdentifiedEstimand::instrumental(
            "iv",
            Arc::from([VariableId::from_raw(3)]),
            ExprId::from_raw(1),
        );
        assert!(iv.adjustment_set.is_empty());
        assert_eq!(iv.instruments.as_ref(), &[VariableId::from_raw(3)]);

        let fd = IdentifiedEstimand::frontdoor(
            "frontdoor",
            Arc::from([VariableId::from_raw(4)]),
            ExprId::from_raw(2),
        );
        assert_eq!(fd.mediators.as_ref(), &[VariableId::from_raw(4)]);
        assert!(fd.instruments.is_empty());
    }
}

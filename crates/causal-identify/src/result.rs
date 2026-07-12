//! Identification result types (DESIGN.md §10.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, VariableId};
use causal_expr::{CausalExprArena, ExprId};

/// Status of an identification attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationStatus {
    /// Nonparametrically identified.
    NonparametricallyIdentified,
    /// Not identified.
    NotIdentified,
}

/// One identified estimand (Phase 1: backdoor adjustment).
#[derive(Clone, Debug)]
pub struct IdentifiedEstimand {
    /// Method tag.
    pub method: Arc<str>,
    /// Adjustment set (dense variable ids).
    pub adjustment_set: Arc<[VariableId]>,
    /// Functional expression id in `arena`.
    pub functional: ExprId,
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
}

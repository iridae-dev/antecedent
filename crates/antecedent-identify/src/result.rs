//! Identification result types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
};
use antecedent_expr::CausalExprArena;

use crate::hedge::HedgeCertificate;

pub use antecedent_core::IdentificationStatus;
pub use antecedent_expr::IdentifiedEstimand;

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

/// Execution diagnostic of a search that stopped at its work budget before it could decide.
///
/// How far a search ran is a fact about the run, not about the graph, so the diagnostic is
/// [`DiagnosticKind::Execution`]: a `NotIdentified` carrying it is undecided, not refuted.
/// `code` must end in `.search_bounded`, the suffix [`crate::envelope::search_truncated`]
/// and the Auto strategy recognise; every bounded search builds its diagnostic here so none
/// can be typed as a scientific negative.
pub(crate) fn search_bounded_diagnostic(
    code: &'static str,
    detail: impl Into<Arc<str>>,
) -> Diagnostic {
    debug_assert!(code.ends_with(".search_bounded"), "bounded-search code `{code}`");
    Diagnostic::new(code, DiagnosticKind::Execution, DiagnosticSeverity::Warning, detail)
}

/// Identification claim of one estimand in a result that lists alternatives.
///
/// Strategies differ in what they rely on: a Wald ratio needs an exclusion restriction
/// and a parametric (or monotonicity) restriction, a back-door functional needs neither.
/// A result that lists estimands from several strategies therefore carries one claim per
/// estimand, and [`IdentificationResult::narrowed_to`] turns the listing into the claim
/// of the estimand that is actually used.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EstimandClaim {
    /// Status this estimand's strategy established.
    pub status: IdentificationStatus,
    /// Assumptions this estimand's functional relies on, caller-declared ones included.
    pub required_assumptions: AssumptionSet,
}

/// Full identification result.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    ///
    /// When [`Self::estimand_claims`] is non-empty this is the union over the listed
    /// alternatives (no estimand needs more than this set); the exact set of one estimand
    /// is its claim, see [`Self::narrowed_to`].
    pub required_assumptions: AssumptionSet,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Performance.
    pub performance: IdentificationPerformanceRecord,
    /// Hedge witness when [`IdentificationStatus::NotIdentified`] via general ID.
    pub hedge: Option<HedgeCertificate>,
    /// Per-estimand claims, index-aligned with [`Self::estimands`].
    ///
    /// Empty when every estimand shares [`Self::status`] and
    /// [`Self::required_assumptions`] (single-strategy results).
    pub estimand_claims: Vec<EstimandClaim>,
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
            hedge: None,
            estimand_claims: Vec::new(),
        }
    }

    /// Identified under parametric restrictions (e.g. IV Wald / LATE).
    #[must_use]
    pub fn identified_under_parametric_restrictions(
        query: CausalQuery,
        estimands: Vec<IdentifiedEstimand>,
        arena: CausalExprArena,
        derivation: DerivationTrace,
        required_assumptions: AssumptionSet,
        performance: IdentificationPerformanceRecord,
    ) -> Self {
        Self {
            status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
            query,
            estimands,
            arena,
            derivation,
            required_assumptions,
            diagnostics: Vec::new(),
            performance,
            hedge: None,
            estimand_claims: Vec::new(),
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
            hedge: None,
            estimand_claims: Vec::new(),
        }
    }

    /// Not-identified result with a hedge certificate.
    #[must_use]
    pub fn not_identified_hedge(
        query: CausalQuery,
        derivation: DerivationTrace,
        required_assumptions: AssumptionSet,
        performance: IdentificationPerformanceRecord,
        hedge: HedgeCertificate,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            status: IdentificationStatus::NotIdentified,
            query,
            estimands: Vec::new(),
            arena: CausalExprArena::new(),
            derivation,
            required_assumptions,
            diagnostics,
            performance,
            hedge: Some(hedge),
            estimand_claims: Vec::new(),
        }
    }

    /// Full constructor (required outside this crate because the type is `#[non_exhaustive]`).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        status: IdentificationStatus,
        query: CausalQuery,
        estimands: Vec<IdentifiedEstimand>,
        arena: CausalExprArena,
        derivation: DerivationTrace,
        required_assumptions: AssumptionSet,
        diagnostics: Vec<Diagnostic>,
        performance: IdentificationPerformanceRecord,
        hedge: Option<HedgeCertificate>,
    ) -> Self {
        Self {
            status,
            query,
            estimands,
            arena,
            derivation,
            required_assumptions,
            diagnostics,
            performance,
            hedge,
            estimand_claims: Vec::new(),
        }
    }

    /// Attach one claim per estimand (index-aligned with [`Self::estimands`]).
    ///
    /// # Panics
    ///
    /// `claims` is non-empty and its length differs from the estimand count.
    #[must_use]
    pub fn with_estimand_claims(mut self, claims: Vec<EstimandClaim>) -> Self {
        assert!(
            claims.is_empty() || claims.len() == self.estimands.len(),
            "estimand claims must be index-aligned with estimands"
        );
        self.estimand_claims = claims;
        self
    }

    /// Claim of the estimand at `index`: its own status and assumption set.
    ///
    /// Falls back to the result-level status and assumptions when the result carries no
    /// per-estimand claims. `None` when `index` is out of range.
    #[must_use]
    pub fn claim(&self, index: usize) -> Option<EstimandClaim> {
        if index >= self.estimands.len() {
            return None;
        }
        Some(self.estimand_claims.get(index).cloned().unwrap_or_else(|| EstimandClaim {
            status: self.status,
            required_assumptions: self.required_assumptions.clone(),
        }))
    }

    /// The result restricted to the estimand at `index`.
    ///
    /// Status and assumptions become exactly that estimand's claim, so a result built on
    /// one strategy never reports another strategy's status or omits its own assumptions.
    /// The arena, derivation, diagnostics, and performance record are kept: they describe
    /// the search that produced the estimand. `None` when `index` is out of range.
    #[must_use]
    pub fn narrowed_to(&self, index: usize) -> Option<Self> {
        let claim = self.claim(index)?;
        let mut out = self.clone();
        out.status = claim.status;
        out.required_assumptions = claim.required_assumptions.clone();
        out.estimands = vec![self.estimands[index].clone()];
        out.estimand_claims = vec![claim];
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::VariableId;
    use antecedent_expr::ExprId;

    use super::*;

    #[test]
    fn search_bounded_diagnostic_is_an_execution_warning_the_envelope_recognises() {
        let diagnostic = search_bounded_diagnostic("identify.example.search_bounded", "stopped");
        assert_eq!(diagnostic.kind, DiagnosticKind::Execution);
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        let mut result = IdentificationResult::not_identified(
            CausalQuery::average_effect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        );
        assert!(!crate::envelope::search_truncated(&result));
        result.diagnostics.push(diagnostic);
        assert!(crate::envelope::search_truncated(&result));
    }

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

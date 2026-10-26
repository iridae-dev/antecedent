//! Identification result types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionRecord, AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity,
};
use antecedent_expr::{CausalExprArena, EstimandMethod};

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

/// Fields of an [`IdentificationResult`], for building one outside this crate
/// (the result type is `#[non_exhaustive]`). Per-estimand claims are attached afterwards with
/// [`IdentificationResult::with_estimand_claims`].
#[derive(Clone, Debug)]
pub struct IdentificationResultParts {
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
    /// Hedge witness when [`IdentificationStatus::NotIdentified`] via general ID.
    pub hedge: Option<HedgeCertificate>,
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
    pub fn from_parts(parts: IdentificationResultParts) -> Self {
        let IdentificationResultParts {
            status,
            query,
            estimands,
            arena,
            derivation,
            required_assumptions,
            diagnostics,
            performance,
            hedge,
        } = parts;
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

    /// Add `restriction` to the claim and weaken a nonparametric status to a parametric one.
    ///
    /// The restriction is a premise on the structural model under which an estimator's target
    /// is the identified functional (for example the constant-effect restriction of a Wald
    /// ratio). It joins the result's assumptions and every listed estimand's own claim; a
    /// status that is already weaker than nonparametric identification is kept.
    pub fn restrict_claim(&mut self, restriction: &AssumptionRecord) {
        let weaken = |status: &mut IdentificationStatus| {
            if *status == IdentificationStatus::NonparametricallyIdentified {
                *status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
            }
        };
        weaken(&mut self.status);
        self.required_assumptions.extend_unique([restriction]);
        for own in &mut self.estimand_claims {
            weaken(&mut own.status);
            own.required_assumptions.extend_unique([restriction]);
        }
    }

    /// Index of the one listed estimand `compatible` accepts, with no silent `.first()`.
    ///
    /// A single listed estimand is the choice. Otherwise exactly one compatible estimand wins;
    /// when several are compatible, a criterion estimand (back-door adjustment, front-door,
    /// path-specific, ...) beats the general-ID functional that is listed beside it as the
    /// fallback for the same query, if exactly one compatible estimand is not general-ID. That
    /// tiebreak does not fire when every compatible estimand is general-ID (for example a mean
    /// curve's one general-ID estimand per grid level): those are distinct functionals, not a
    /// criterion against a fallback.
    ///
    /// # Errors
    ///
    /// [`EstimandSelectionError::NoEstimand`] for an empty listing, and
    /// [`EstimandSelectionError::Ambiguous`] when no unique estimand is compatible.
    pub fn select_estimand_index(
        &self,
        compatible: impl Fn(&IdentifiedEstimand) -> bool,
    ) -> Result<usize, EstimandSelectionError> {
        let estimands = &self.estimands;
        match estimands.len() {
            0 => return Err(EstimandSelectionError::NoEstimand),
            1 => return Ok(0),
            _ => {}
        }
        let matches: Vec<usize> =
            (0..estimands.len()).filter(|&i| compatible(&estimands[i])).collect();
        if let [only] = matches[..] {
            return Ok(only);
        }
        let mut non_general = matches
            .iter()
            .copied()
            .filter(|&i| estimands[i].method_kind() != Ok(EstimandMethod::GeneralId));
        if let (Some(only), None) = (non_general.next(), non_general.next()) {
            return Ok(only);
        }
        Err(EstimandSelectionError::Ambiguous {
            methods: estimands.iter().map(|estimand| Arc::clone(&estimand.method)).collect(),
            matched: matches.len(),
        })
    }
}

/// Why [`IdentificationResult::select_estimand_index`] chose no estimand.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EstimandSelectionError {
    /// The result lists no estimand.
    NoEstimand,
    /// Several estimands are listed and none is the unique compatible one.
    Ambiguous {
        /// The method of every listed estimand, in listing order.
        methods: Vec<Arc<str>>,
        /// How many listed estimands were compatible.
        matched: usize,
    },
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

    fn listing(methods: &[&str]) -> IdentificationResult {
        IdentificationResult::identified(
            CausalQuery::average_effect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )),
            methods
                .iter()
                .map(|method| {
                    IdentifiedEstimand::backdoor(*method, Arc::from([]), ExprId::from_raw(0))
                })
                .collect(),
            CausalExprArena::default(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    #[test]
    fn selection_takes_the_unique_compatible_estimand_and_prefers_a_criterion_to_general_id() {
        let result = listing(&["general.id", "backdoor.adjustment", "frontdoor"]);
        let by = |wanted: &'static [&'static str]| {
            result.select_estimand_index(move |e| wanted.contains(&e.method.as_ref()))
        };
        assert_eq!(by(&["frontdoor"]), Ok(2));
        // A criterion estimand beats the general-ID fallback listed beside it.
        assert_eq!(by(&["general.id", "backdoor.adjustment"]), Ok(1));
        // Two criterion estimands, or only general-ID ones, are ambiguous.
        assert_eq!(
            by(&["backdoor.adjustment", "frontdoor"]),
            Err(EstimandSelectionError::Ambiguous {
                methods: vec!["general.id".into(), "backdoor.adjustment".into(), "frontdoor".into()],
                matched: 2,
            })
        );
        assert!(matches!(
            by(&["nothing"]),
            Err(EstimandSelectionError::Ambiguous { matched: 0, .. })
        ));
        assert_eq!(listing(&[]).select_estimand_index(|_| true), Err(EstimandSelectionError::NoEstimand));
        // A single listed estimand is the choice whatever the predicate says.
        assert_eq!(listing(&["iv"]).select_estimand_index(|_| false), Ok(0));
    }

    #[test]
    fn restricting_a_claim_weakens_only_a_nonparametric_status_and_adds_the_premise_once() {
        use antecedent_core::{Assumption, AssumptionScope, AssumptionSource, AssumptionStatus};
        let restriction = AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("restriction"),
                description: Arc::from("a structural restriction"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        };
        let mut result = listing(&["backdoor.adjustment"]);
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        result.restrict_claim(&restriction);
        result.restrict_claim(&restriction);
        assert_eq!(result.status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        assert_eq!(result.required_assumptions.entries, vec![restriction.clone()]);
        // A status that is already weaker is kept.
        let mut partial = listing(&["backdoor.adjustment"]);
        partial.status = IdentificationStatus::PartiallyIdentified;
        partial.restrict_claim(&restriction);
        assert_eq!(partial.status, IdentificationStatus::PartiallyIdentified);
        assert_eq!(partial.required_assumptions.entries, vec![restriction]);
    }
}

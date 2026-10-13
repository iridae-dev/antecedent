//! A retained fitted effect model decides for a declared cohort.
//!
//! A fitted CATE map predicts conditional treatment contrasts. It does not give two
//! absolute potential-outcome means, a law of the contrast or an interval for a cohort.
//! This entry binds a verified retained model to a declared cohort (strict named feature
//! rows, row identities, finite normalized weights, target population and explicit outcome
//! units), checks every cohort row against the retained model's own support contract and
//! issues exactly one native quantity: the weighted mean treatment contrast, a
//! `mean_difference` coordinate bound to the cohort snapshot. That quantity feeds the
//! existing decision boundary as one mean, so only the expectation of an affine utility is
//! answered. Probability, quantile, nonlinear-utility and joint-law requests stay refused.
//!
//! The flow mirrors the original-result decision path:
//!
//! 1. [`FittedEffectModel::load`] verifies the exported analysis result (contract section,
//!    executed claim, no attested evidence) and retains its portable fitted payload, the
//!    original contrast, the support contract and the parent evidence;
//! 2. [`FittedEffectModel::decision_source`] binds a [`CohortBinding`] and issues a
//!    [`CohortDecisionSource`], exportable under its identity;
//! 3. [`cohort_decision_input`] and [`evaluate_cohort_decision`] hand it to the decision
//!    evaluator under a [`DecisionContract`] that names the exact coordinate.
//!
//! A cohort from a different population is accepted only under a declared covariate shift
//! ([`CovariateShiftDeclaration`]) with every row inside the support contract; the result
//! is point-only evidence under that assumption, labelled `Extrapolative`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
pub use antecedent_design::cohort_decision_input::{
    cohort_decision_input, cohort_mean_source, cohort_trust_evidence,
};
use antecedent_design::composition_boundary::{
    BoundaryError, SupportPolicy, SupportedDecision, TrustRequirement, evaluate_with_support,
};
use antecedent_design::decision_contract::DecisionContract;
pub use antecedent_io::cohort_decision::{
    CATE_FUNCTIONAL_ID, COHORT_DECISION_ARTIFACT_KIND, COHORT_DECISION_VERSION,
    COHORT_FUNCTIONAL_ID, COVARIATE_SHIFT_ASSUMPTION, CohortBinding, CohortColumn,
    CohortDecisionSource, CohortExpectation, CohortOptions, CohortRefusal, ContrastDeclaration,
    CovariateShiftDeclaration, LawRequest, MAX_COHORT_ROWS, ParentIdentity, RetainedEffectModel,
    SupportReceipt, UncalibratedStandardError,
};

use super::compact_export::CompactExport;

/// A verified retained effect model.
///
/// Immutable: a new cohort needs a new [`CohortDecisionSource`], never a refit, and new
/// outcomes need a new fit and a new parent claim.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedEffectModel {
    retained: RetainedEffectModel,
}

impl FittedEffectModel {
    /// Load the portable fitted effect of an exported, verified analysis result.
    ///
    /// # Errors
    /// An artifact whose contract section does not verify, an attested external estimate,
    /// a query that is not one mean-effect contrast of a hard-set treatment, a result with
    /// no portable fitted effect, or a predictor on a probability scale.
    pub fn load(artifact: &[u8]) -> Result<Self, CohortRefusal> {
        RetainedEffectModel::from_analysis_result(artifact, None).map(|retained| Self { retained })
    }

    /// As [`Self::load`], requiring the verified claim identity the caller retained.
    ///
    /// # Errors
    /// As [`Self::load`], and `cohort_decision.wrong_parent_identity` for another claim.
    pub fn load_expecting(artifact: &[u8], expected_claim: &str) -> Result<Self, CohortRefusal> {
        RetainedEffectModel::from_analysis_result(artifact, Some(expected_claim))
            .map(|retained| Self { retained })
    }

    /// Retain a caller-built compact linear model whose response declares a conditional
    /// treatment contrast. The model is a declaration, so its sources are never native, and
    /// it is the one route that carries a coefficient covariance.
    ///
    /// # Errors
    /// A response that is not a declared contrast.
    pub fn from_compact_export(export: &CompactExport) -> Result<Self, CohortRefusal> {
        RetainedEffectModel::from_compact_export(export).map(|retained| Self { retained })
    }

    /// The retained model.
    #[must_use]
    pub fn retained(&self) -> &RetainedEffectModel {
        &self.retained
    }

    /// Identity of the parent claim this model was retained from.
    #[must_use]
    pub fn parent_claim(&self) -> &str {
        &self.retained.parent().claim_id
    }

    /// Feature names a cohort must supply exactly.
    #[must_use]
    pub fn features(&self) -> Vec<String> {
        self.retained.feature_names()
    }

    /// Bind `cohort`, check support and issue the cohort-mean contrast source.
    ///
    /// `outcome_units` label the original outcome scale; nothing is converted.
    ///
    /// # Errors
    /// Missing or outside support, a feature-schema, unit, population or contrast mismatch,
    /// or an undeclared population shift; each refusal names the declaration or evidence
    /// that would license the request.
    pub fn decision_source(
        &self,
        cohort: &CohortBinding,
        outcome_units: &str,
        options: &CohortOptions,
        ctx: &ExecutionContext,
    ) -> Result<CohortDecisionSource, CohortRefusal> {
        CohortDecisionSource::build(&self.retained, cohort, outcome_units, options, ctx)
    }
}

/// Consume an exported source under the identity (and optionally the parent, population,
/// units, contrast and cohort) the caller retained, replaying every number.
///
/// # Errors
/// Any changed field, even when the container was resealed, and any differing expectation.
pub fn consume_cohort_decision_source(
    bytes: &[u8],
    expectation: &CohortExpectation,
) -> Result<CohortDecisionSource, CohortRefusal> {
    CohortDecisionSource::consume(bytes, expectation)
}

/// Evaluate `contract` on one cohort-mean source under `policy`.
///
/// Exact only for an affine utility of the source's coordinate; the existing evaluator
/// refuses everything else.
///
/// # Errors
/// An invalid contract, a refused action under `RequireAllActions`, and any refusal of the
/// underlying evaluator.
pub fn evaluate_cohort_decision(
    contract: &DecisionContract,
    source: &CohortDecisionSource,
    policy: &SupportPolicy,
) -> Result<SupportedDecision, BoundaryError> {
    let input = cohort_decision_input("cohort_decision", source, TrustRequirement::Unrestricted)?;
    evaluate_with_support(contract, &[input], policy)
}

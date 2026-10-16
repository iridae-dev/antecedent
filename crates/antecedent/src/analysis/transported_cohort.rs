//! A retained fitted effect model carried to a cohort of another population under a
//! declared selection-diagram premise.
//!
//! [`super::cohort_decision`] decides for a cohort of the model's own population and,
//! under a bare covariate-shift label, for a cohort of another. This module is the
//! stricter transport route of the same retained model. It binds the library's
//! [`transport_cohort_effect`] (selection-shift declaration, overlap check, point-only
//! value) to a verified analysis result and a declared [`CohortBinding`], and then hands
//! the same cohort to the existing decision surface:
//!
//! 1. [`transport_cohort_effect_from_result`] verifies the exported analysis result
//!    ([`FittedEffectModel::load_expecting`]: contract section, executed claim, no
//!    attested evidence), takes its portable fitted effect, and issues the weighted
//!    cohort-mean contrast only when the [`SelectionShiftDeclaration`] states that the
//!    selection node points only into the declared effect modifiers and those modifiers
//!    are exactly the model's features, and the cohort's overlap check against the
//!    model's own support contract passes (no disjoint feature, no row outside the box).
//! 2. [`TransportedCohort::decision_source`] issues the existing
//!    [`CohortDecisionSource`] for the same cohort, labelled `extrapolative`, and requires
//!    its mean to equal the transported mean, so the decision boundary receives exactly the
//!    value the transport receipt describes.
//! 3. [`TransportedCohort::receipt`] is the typed receipt: route, identity, the model
//!    and parent claim, both populations, the selection node and what it points into, the
//!    cohort digest, the overlap record, the support label and the assumptions.
//!
//! # What is and is not claimed
//!
//! The value is the expectation of one cohort-mean conditional contrast under a declared
//! premise: point-only, never target-population data, never two absolute potential-outcome
//! means, never a law. The overlap check is a fitted-box necessary condition, not a
//! density or positivity guarantee. No standard error, interval or utility variance is
//! inherited; [`TransportedCohort::interval_refusal`] and
//! [`TransportedCohort::law_refusal`] name the evidence that would license either.
//!
//! The retained analysis result carries no population label of its own, so the source
//! population is declared by the caller and bound into the identity. The route is bounded
//! to a verified analysis result whose retained effect is a portable identity-scale CATE
//! map with a support contract; a caller-built compact export is refused here
//! (`transported_effect.model_not_portable`) because its coefficient map is a declaration,
//! not a retained fit.
//!
//! # Export and consumption
//!
//! The transported value has no artifact of its own: it is a pure function of the
//! retained result, the cohort and the declaration, and its identity binds all three.
//! [`verify_cohort_transport`] is the fresh-consumer path: it re-runs the whole route
//! from the retained inputs and refuses unless the recomputed identity equals the one the
//! consumer retained. The decision source it hands to the decision boundary is the
//! existing exportable `cohort_decision_source` artifact, consumed by
//! [`super::cohort_decision::consume_cohort_decision_source`].
//!
//! # Refresh with outcome-only data
//!
//! Nothing refreshes in place. New outcome-only data of the model's population is a refit,
//! hence a new analysis result, parent claim and value. Target-population outcomes cannot
//! move the issued value (the cohort carries covariates and weights, not outcomes).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
pub use antecedent_learn::transported_effect::{
    FeatureOverlap, OverlapAssessment, SELECTION_SHIFT_ASSUMPTION, SelectionShiftDeclaration,
    TRANSPORTED_POINT_SCOPE, TargetCohort, TransportRefusal, TransportedCohortEffect,
    TransportedInference, TransportedRequest, assess_overlap, transport_cohort_effect,
};

use super::cohort_decision::{
    CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions, CohortRefusal,
    CovariateShiftDeclaration, FittedEffectModel,
};
use super::composition_verification::{
    ClaimStanding, CompositionNode, library_transported_node, library_transported_standing,
};

/// Route label of this transport in a [`TransportedCohortReceipt`].
pub const COHORT_TRANSPORT_ROUTE: &str = "cohort_effect";
/// Support label of a transported cohort mean: never `supported`.
pub const COHORT_TRANSPORT_SUPPORT: &str = "extrapolative";

/// A declaration that the retained model is not a portable analysis result.
pub const DETAIL_MODEL_NOT_PORTABLE: &str = "transported_effect.model_not_portable";
/// A cohort feature that is not a finite numeric column.
pub const DETAIL_COHORT_NOT_NUMERIC: &str = "transported_effect.cohort_not_numeric";
/// Outcome units were not declared.
pub const DETAIL_UNITS_REQUIRED: &str = "transported_effect.outcome_units_required";
/// The retained identity is not the recomputed one.
pub const DETAIL_IDENTITY_MISMATCH: &str = "transported_effect.identity_mismatch";
/// The decision source's mean is not the transported mean.
pub const DETAIL_DECISION_MEAN_MISMATCH: &str = "transported_effect.decision_source_mismatch";

fn invalid(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> TransportRefusal {
    TransportRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail,
        subject: subject.into(),
        remedy,
    }
}

fn lift(error: CohortRefusal) -> TransportRefusal {
    TransportRefusal {
        code: error.code,
        detail: error.detail,
        subject: error.subject,
        remedy: error.remedy,
    }
}

/// The typed receipt of one transported cohort mean.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedCohortReceipt {
    /// Always [`COHORT_TRANSPORT_ROUTE`].
    pub route: &'static str,
    /// Identity binding model, declaration, cohort, overlap and value.
    pub identity: String,
    /// Identity of the retained model.
    pub model_identity: String,
    /// Identity of the verified parent claim the model was retained from.
    pub parent_claim: String,
    /// Population the model was fitted for (declared by the caller).
    pub source_population: String,
    /// Population the cohort is drawn from.
    pub target_population: String,
    /// Name of the selection node.
    pub selection_node: String,
    /// Variables the selection node points into.
    pub selection_children: Vec<String>,
    /// Effect modifiers declared sufficient (exactly the model's features).
    pub effect_modifiers: Vec<String>,
    /// Digest of the cohort snapshot.
    pub cohort_digest: String,
    /// The overlap check that passed.
    pub overlap: OverlapAssessment,
    /// Always [`COHORT_TRANSPORT_SUPPORT`].
    pub support: &'static str,
    /// Always [`TransportedInference::PointOnly`].
    pub inference: TransportedInference,
    /// Statements that travel with the value.
    pub assumptions: Vec<String>,
}

/// A transported cohort mean with the retained model and cohort it was issued for.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedCohort {
    effect: TransportedCohortEffect,
    model: FittedEffectModel,
    cohort: CohortBinding,
    source_population: String,
}

impl TransportedCohort {
    /// The library's transported value.
    #[must_use]
    pub const fn effect(&self) -> &TransportedCohortEffect {
        &self.effect
    }

    /// The weighted mean of the conditional contrast over the cohort.
    #[must_use]
    pub const fn mean_contrast(&self) -> f64 {
        self.effect.mean_contrast
    }

    /// The conditional contrast at each cohort row, in row order.
    #[must_use]
    pub fn row_contrasts(&self) -> &[f64] {
        &self.effect.row_contrasts
    }

    /// Identity of the transported value.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.effect.identity()
    }

    /// The retained model.
    #[must_use]
    pub const fn model(&self) -> &FittedEffectModel {
        &self.model
    }

    /// The cohort the value is for.
    #[must_use]
    pub const fn cohort(&self) -> &CohortBinding {
        &self.cohort
    }

    /// The typed receipt.
    #[must_use]
    pub fn receipt(&self) -> TransportedCohortReceipt {
        let declaration = &self.effect.declaration;
        TransportedCohortReceipt {
            route: COHORT_TRANSPORT_ROUTE,
            identity: self.effect.identity().to_owned(),
            model_identity: self.effect.model_identity.clone(),
            parent_claim: self.model.parent_claim().to_owned(),
            source_population: self.source_population.clone(),
            target_population: self.cohort.population_id().to_owned(),
            selection_node: declaration.selection_node.clone(),
            selection_children: declaration.selection_children.clone(),
            effect_modifiers: declaration.effect_modifiers.clone(),
            cohort_digest: self.effect.cohort_digest.clone(),
            overlap: self.effect.overlap.clone(),
            support: COHORT_TRANSPORT_SUPPORT,
            inference: self.effect.inference(),
            assumptions: self.effect.assumptions(),
        }
    }

    /// The standing this value states as a claim in a composition graph: a point
    /// computed by the library under a declared premise, so no provider trust
    /// and no calibration record (an interval, band or posterior is refused), and
    /// support `extrapolative` ([`COHORT_TRANSPORT_SUPPORT`]), which is the
    /// strongest a carried value states.
    #[must_use]
    pub fn claim_standing(&self) -> ClaimStanding {
        library_transported_standing(COHORT_TRANSPORT_SUPPORT, &[])
    }

    /// The composition-graph node of this value under `key`, carrying
    /// [`Self::claim_standing`]. Its identity is the transported value's own
    /// identity, so a changed model, cohort, weight, declaration or population
    /// recomputes it.
    #[must_use]
    pub fn composition_node(&self, key: &str) -> CompositionNode {
        library_transported_node(
            key,
            COHORT_TRANSPORT_ROUTE,
            self.effect.identity(),
            self.cohort.population_id(),
            COHORT_TRANSPORT_SUPPORT,
            self.effect.assumptions(),
        )
    }

    /// The refusal for an interval, band or posterior of the transported mean.
    #[must_use]
    pub fn interval_refusal(&self) -> TransportRefusal {
        self.effect.interval_refusal()
    }

    /// The refusal for a probability, quantile, nonlinear utility or joint law.
    #[must_use]
    pub fn law_refusal(&self) -> TransportRefusal {
        self.effect.law_refusal()
    }

    /// Issue the existing decision source for this cohort under the declared shift.
    ///
    /// The source is `extrapolative` and point-only; the decision boundary refuses it
    /// unless the support policy accepts that label. Its mean must equal the transported
    /// mean to rounding, so the two surfaces cannot disagree about the value.
    ///
    /// # Errors
    /// Missing outcome units, any refusal of the existing source builder (feature schema,
    /// support contract, population), and `transported_effect.decision_source_mismatch`.
    pub fn decision_source(
        &self,
        outcome_units: &str,
        ctx: &ExecutionContext,
    ) -> Result<CohortDecisionSource, TransportRefusal> {
        if outcome_units.trim().is_empty() {
            return Err(invalid(
                DETAIL_UNITS_REQUIRED,
                "outcome_units",
                "declare the outcome units; they are never inferred and never converted",
            ));
        }
        let modifiers: Vec<&str> =
            self.effect.declaration.effect_modifiers.iter().map(String::as_str).collect();
        let options = CohortOptions {
            shift: Some(CovariateShiftDeclaration::new(
                self.model.retained().population_id(),
                self.cohort.population_id(),
                &modifiers,
            )),
            expected_contrast_regime: None,
        };
        let source =
            self.model.decision_source(&self.cohort, outcome_units, &options, ctx).map_err(lift)?;
        let (a, b) = (source.mean_contrast(), self.effect.mean_contrast);
        if (a - b).abs() > 1e-9 * (1.0 + b.abs()) {
            return Err(TransportRefusal {
                code: antecedent_core::reason_code!("transport_not_certified"),
                detail: DETAIL_DECISION_MEAN_MISMATCH,
                subject: format!("decision={a} transported={b}"),
                remedy: "the decision source and the transport must predict the same cohort \
                         mean; refit or report the model",
            });
        }
        Ok(source)
    }
}

fn numeric_cohort(cohort: &CohortBinding) -> Result<TargetCohort, TransportRefusal> {
    let mut columns = Vec::with_capacity(cohort.columns().len());
    for (name, column) in cohort.features().iter().zip(cohort.columns()) {
        match column {
            CohortColumn::Numeric(values) => columns.push(values.clone()),
            CohortColumn::Categorical(_) => {
                return Err(invalid(
                    DETAIL_COHORT_NOT_NUMERIC,
                    name.clone(),
                    "the transported route reads finite numeric effect modifiers only; supply \
                     this feature as a numeric column",
                ));
            }
        }
    }
    Ok(TargetCohort {
        population_id: cohort.population_id().to_owned(),
        row_ids: cohort.row_ids().to_vec(),
        features: cohort.features().to_vec(),
        columns,
        weights: cohort.weights().to_vec(),
    })
}

/// Transport the retained effect of a verified analysis result to `cohort`.
///
/// Order of refusal: the parent result (verification, claim identity, native portable
/// payload), outcome units, cohort type, then the library's order: cohort shape, support
/// contract and identity link, strict feature schema, the shift declaration, overlap
/// (disjoint feature, then rows outside), prediction.
///
/// `source_population` is the population the model was fitted for; the declaration must
/// name it as source and the cohort's population as target.
///
/// # Errors
/// A [`TransportRefusal`] naming the unmet premise; no value is issued.
pub fn transport_cohort_effect_from_result(
    artifact: &[u8],
    expected_claim: Option<&str>,
    source_population: &str,
    cohort: &CohortBinding,
    shift: Option<&SelectionShiftDeclaration>,
    ctx: &ExecutionContext,
) -> Result<TransportedCohort, TransportRefusal> {
    let model = match expected_claim {
        Some(claim) => FittedEffectModel::load_expecting(artifact, claim),
        None => FittedEffectModel::load(artifact),
    }
    .map_err(lift)?;
    let consumed = antecedent_io::consume_analysis_result(artifact).map_err(|_| {
        invalid(
            DETAIL_MODEL_NOT_PORTABLE,
            "analysis_result",
            "supply the exported analysis result of a native DR-learner or causal-forest fit",
        )
    })?;
    let Some(effect) = consumed.body.fitted_effect.clone() else {
        return Err(invalid(
            DETAIL_MODEL_NOT_PORTABLE,
            "fitted_effect",
            "fit with a DR-learner or causal-forest estimator, which retain a portable CATE map",
        ));
    };
    let target = numeric_cohort(cohort)?;
    let issued = transport_cohort_effect(
        &effect,
        &model.features(),
        model.retained().identity(),
        source_population,
        &target,
        shift,
        ctx,
    )?;
    Ok(TransportedCohort {
        effect: issued,
        model,
        cohort: cohort.clone(),
        source_population: source_population.to_owned(),
    })
}

/// Re-run the route from the retained inputs and require the retained identity.
///
/// The fresh-consumer path: nothing is read from a stored value, so a changed row, weight,
/// declaration, population or parent changes the identity and is refused.
///
/// # Errors
/// Whatever [`transport_cohort_effect_from_result`] refuses, and
/// `transported_effect.identity_mismatch` when the recomputed identity is not
/// `expected_identity`.
pub fn verify_cohort_transport(
    artifact: &[u8],
    expected_claim: Option<&str>,
    source_population: &str,
    cohort: &CohortBinding,
    shift: Option<&SelectionShiftDeclaration>,
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<TransportedCohort, TransportRefusal> {
    let transported = transport_cohort_effect_from_result(
        artifact,
        expected_claim,
        source_population,
        cohort,
        shift,
        ctx,
    )?;
    if transported.identity() != expected_identity {
        return Err(TransportRefusal {
            code: antecedent_core::reason_code!("transport_not_certified"),
            detail: DETAIL_IDENTITY_MISMATCH,
            subject: format!("recomputed={} retained={expected_identity}", transported.identity()),
            remedy: "re-derive the transport from the retained result, cohort and declaration",
        });
    }
    Ok(transported)
}

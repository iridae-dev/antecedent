//! Retained effect model validation and stable identity.

use antecedent_core::CanonicalWriter;
use antecedent_learn::{FittedEffect, PredictionMap};

use super::{
    CATE_FUNCTIONAL_ID, CohortRefusal, ContrastDeclaration, MODEL_TAG, ParentIdentity, invalid,
    io_refusal, semantics_mismatch,
};
use crate::compact_export::CompactExport;
use crate::convert::to_cbor;
use crate::error::IoError;
use crate::query_wire::{CausalQueryWire, InterventionWire, TargetPopulationWire, ValueWire};

#[derive(Clone, Debug, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "one value per decoded cohort model, held by value; boxing the compact export would not pay"
)]
pub(super) enum ModelKind {
    Portable { effect: FittedEffect, features: Vec<String> },
    Compact { export: CompactExport, bytes: Vec<u8> },
}

/// A verified, immutable retained effect model with its original contrast and support
/// contract. Predictions are recomputed from it; nothing else supplies a number.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedEffectModel {
    pub(super) kind: ModelKind,
    pub(super) population_id: String,
    pub(super) contrast: ContrastDeclaration,
    pub(super) parent: ParentIdentity,
    pub(super) identity: String,
}

fn parent_refusal(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> CohortRefusal {
    invalid(detail, subject, remedy)
}

fn level_of(intervention: &InterventionWire, treatment: u32) -> Option<f64> {
    match intervention {
        InterventionWire::Set { variable, value } if *variable == treatment => match value {
            ValueWire::Float64(v) if v.is_finite() => Some(*v),
            ValueWire::Int64(v) => i32::try_from(*v).ok().map(f64::from),
            ValueWire::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        },
        _ => None,
    }
}

impl RetainedEffectModel {
    fn finish(
        kind: ModelKind,
        population_id: String,
        contrast: ContrastDeclaration,
        parent: ParentIdentity,
    ) -> Result<Self, CohortRefusal> {
        let identity = model_identity(&kind, &population_id, &contrast, &parent)?;
        Ok(Self { kind, population_id, contrast, parent, identity })
    }

    /// Retain the portable fitted effect of a verified analysis result.
    ///
    /// The parent must be an executed, verified claim of a native analysis: an artifact
    /// without a verified contract, with a caller-attested evidence record, or whose query
    /// is not a static mean-effect contrast of a hard-set treatment is refused. When
    /// `expected_claim` is given the verified claim identity must equal it.
    ///
    /// # Errors
    /// An unreadable, unverified or attested parent, a missing portable model, a
    /// non-identity link, a query that is not one declared contrast, and a wrong claim.
    #[allow(
        clippy::too_many_lines,
        reason = "one flat lowering of an analysis result into the wire record, field by field"
    )]
    pub fn from_analysis_result(
        bytes: &[u8],
        expected_claim: Option<&str>,
    ) -> Result<Self, CohortRefusal> {
        let remedy =
            "supply the exported analysis result of a native DR-learner or causal-forest fit";
        let consumed = crate::replay_analysis_result(bytes).map_err(|e| io_refusal(&e))?;
        if !consumed.acceptance.accepts_as_claim() {
            return Err(parent_refusal(
                antecedent_core::reason_details::cohort_decision::PARENT_NOT_VERIFIED,
                "analysis_result",
                "supply an analysis result whose contract section verifies against its body",
            ));
        }
        let claim =
            consumed.contract.as_ref().and_then(|contract| contract.claim.as_ref()).ok_or_else(
                || {
                    parent_refusal(
                        antecedent_core::reason_details::cohort_decision::PARENT_NOT_VERIFIED,
                        "claim",
                        remedy,
                    )
                },
            )?;
        if !claim.attested.is_empty() {
            return Err(parent_refusal(
                antecedent_core::reason_details::cohort_decision::PARENT_ATTESTED_NOT_NATIVE,
                "attested_evidence",
                "an attested external estimate supplies no native fitted payload; refit natively",
            ));
        }
        let claim_id = antecedent_core::to_hex(&claim.claim_id);
        if expected_claim.is_some_and(|expected| expected != claim_id) {
            return Err(parent_refusal(
                antecedent_core::reason_details::cohort_decision::WRONG_PARENT_IDENTITY,
                claim_id,
                "load the model from the claim the consumer retained",
            ));
        }
        let CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            control,
            active,
            target_population,
            outcome_functional,
            ..
        } = &consumed.body.query
        else {
            return Err(parent_refusal(
                antecedent_core::reason_details::cohort_decision::PARENT_QUERY_UNSUPPORTED,
                "query",
                remedy,
            ));
        };
        if !outcome_functional.is_mean() || *target_population != TargetPopulationWire::AllObserved
        {
            return Err(parent_refusal(
                antecedent_core::reason_details::cohort_decision::PARENT_QUERY_UNSUPPORTED,
                "outcome_functional_or_population",
                "only a mean-effect contrast over all observed units retains a CATE map",
            ));
        }
        let (Some(control), Some(active)) =
            (level_of(control, *treatment), level_of(active, *treatment))
        else {
            return Err(parent_refusal(
                antecedent_core::reason_details::cohort_decision::PARENT_QUERY_UNSUPPORTED,
                "intervention",
                "the contrast must be two hard-set levels of the treatment",
            ));
        };
        let name = |id: u32| {
            consumed.header.variable_names.get(id as usize).cloned().ok_or_else(|| {
                parent_refusal(
                    antecedent_core::reason_details::cohort_decision::PARENT_QUERY_UNSUPPORTED,
                    "variable_names",
                    remedy,
                )
            })
        };
        let treatment_name = name(*treatment)?;
        let outcome_name = name(*outcome)?;
        let effect = consumed.body.fitted_effect.as_deref().cloned().ok_or_else(|| {
            CohortRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                antecedent_core::reason_details::cohort_decision::FITTED_MODEL_ABSENT,
                "fitted_effect",
                "fit with a DR-learner or causal-forest estimator, which retain a portable \
                 CATE map; other estimators retain none",
            )
        })?;
        let features = effect
            .features
            .iter()
            .map(|id| name(*id))
            .collect::<Result<Vec<String>, CohortRefusal>>()?;
        let contrast = ContrastDeclaration {
            outcome: outcome_name,
            regime_id: ContrastDeclaration::regime_of(&treatment_name, active, control),
            treatment: Some(treatment_name),
            control: Some(control),
            active: Some(active),
        };
        let parent = ParentIdentity {
            claim_id,
            artifact_digest: antecedent_core::to_hex(&consumed.artifact_digest),
            diagnostics: consumed.body.diagnostics.clone(),
            assumptions: consumed.body.assumptions.clone(),
        };
        Self::from_portable_parts(effect, features, "target".to_owned(), contrast, parent)
    }

    /// Retain a portable effect from already verified parts.
    ///
    /// Crate-private: the only public routes verify their parent.
    pub(crate) fn from_portable_parts(
        effect: FittedEffect,
        features: Vec<String>,
        population_id: String,
        contrast: ContrastDeclaration,
        parent: ParentIdentity,
    ) -> Result<Self, CohortRefusal> {
        effect.validate().map_err(|_| {
            invalid(
                antecedent_core::reason_details::cohort_decision::FITTED_MODEL_INVALID,
                "fitted_effect",
                "refit; the retained payload does not validate",
            )
        })?;
        if features.len() != effect.features.len() || features.iter().any(|f| f.trim().is_empty()) {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::FITTED_MODEL_INVALID,
                "feature_names",
                "name every retained feature",
            ));
        }
        let identity_link = match &effect.predictor.model {
            PredictionMap::Linear { logistic, .. } => !logistic,
            PredictionMap::Trees { logistic, probability, .. } => !logistic && !probability,
        };
        if !identity_link {
            return Err(semantics_mismatch(
                antecedent_core::reason_details::cohort_decision::LINK_NOT_IDENTITY,
                "predictor",
                "a probability-scale predictor is not a treatment contrast; the cohort mean \
                 needs the identity-scale CATE map",
            ));
        }
        Self::finish(ModelKind::Portable { effect, features }, population_id, contrast, parent)
    }

    /// Retain a caller-built compact linear model.
    ///
    /// The export must declare its response as a conditional treatment contrast: functional
    /// [`CATE_FUNCTIONAL_ID`], regime identity starting `contrast:` and an identity
    /// transform. The coefficients are the caller's declaration, so a source built from it
    /// is never native.
    ///
    /// # Errors
    /// A response that is not a declared contrast, or an oversized export.
    pub fn from_compact_export(export: &CompactExport) -> Result<Self, CohortRefusal> {
        let response = &export.body().response;
        if response.functional_id != CATE_FUNCTIONAL_ID
            || !response.regime_id.starts_with("contrast:")
            || response.role != "outcome"
            || response.transform_id != "identity"
        {
            return Err(semantics_mismatch(
                antecedent_core::reason_details::cohort_decision::CONTRAST_NOT_DECLARED,
                response.variable_id.clone(),
                "build the export with an outcome response whose functional is \
                 conditional_mean_difference and whose regime is contrast:do(t=a)-do(t=b)",
            ));
        }
        let bytes = export.to_bytes("cohort_decision_model").map_err(|e| {
            invalid(
                antecedent_core::reason_details::cohort_decision::FITTED_MODEL_INVALID,
                e.to_string(),
                "rebuild the export",
            )
        })?;
        let contrast = ContrastDeclaration {
            outcome: response.variable_id.clone(),
            regime_id: response.regime_id.clone(),
            treatment: None,
            control: None,
            active: None,
        };
        let parent = ParentIdentity {
            claim_id: export.identity().to_owned(),
            artifact_digest: export.identity().to_owned(),
            diagnostics: Vec::new(),
            assumptions: Vec::new(),
        };
        Self::finish(
            ModelKind::Compact { export: export.clone(), bytes },
            response.population_id.clone(),
            contrast,
            parent,
        )
    }

    /// Identity of the retained model: payload, schema, population, contrast and parent.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The population the model is fitted for.
    #[must_use]
    pub fn population_id(&self) -> &str {
        &self.population_id
    }

    /// The declared treatment contrast.
    #[must_use]
    pub fn contrast(&self) -> &ContrastDeclaration {
        &self.contrast
    }

    /// The parent claim identity and preserved evidence.
    #[must_use]
    pub fn parent(&self) -> &ParentIdentity {
        &self.parent
    }

    /// Feature names the cohort must supply exactly.
    #[must_use]
    pub fn feature_names(&self) -> Vec<String> {
        match &self.kind {
            ModelKind::Portable { features, .. } => features.clone(),
            ModelKind::Compact { export, .. } => export
                .body()
                .inputs
                .iter()
                .map(|input| input.quantity.variable_id.clone())
                .collect(),
        }
    }

    /// Whether the model retains an executable support contract.
    #[must_use]
    pub fn has_support_contract(&self) -> bool {
        match &self.kind {
            ModelKind::Portable { effect, .. } => effect.support.is_some(),
            ModelKind::Compact { .. } => true,
        }
    }

    /// Whether the model is a linear compact model with a coefficient covariance.
    #[must_use]
    pub fn is_linear_compact(&self) -> bool {
        matches!(self.kind, ModelKind::Compact { .. })
    }

    /// Whether the model comes from a verified native analysis.
    #[must_use]
    pub fn is_native(&self) -> bool {
        matches!(self.kind, ModelKind::Portable { .. })
    }
}

fn model_identity(
    kind: &ModelKind,
    population_id: &str,
    contrast: &ContrastDeclaration,
    parent: &ParentIdentity,
) -> Result<String, CohortRefusal> {
    let encode = |error: IoError| io_refusal(&error);
    let mut canon = CanonicalWriter::new(MODEL_TAG);
    match kind {
        ModelKind::Portable { effect, features } => {
            canon.text("portable").bytes(&to_cbor(effect).map_err(encode)?).texts(features);
        }
        ModelKind::Compact { export, .. } => {
            canon.text("compact").text(export.identity());
        }
    }
    canon.text(population_id);
    canon.bytes(&to_cbor(contrast).map_err(encode)?);
    canon.bytes(&to_cbor(parent).map_err(encode)?);
    Ok(canon.finish_hex())
}

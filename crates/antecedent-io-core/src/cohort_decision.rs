//! Cohort-mean treatment-contrast decision source from a retained fitted effect model.
//!
//! A retained effect model predicts a conditional treatment contrast `tau(x)` (a CATE)
//! at covariate rows. This module binds such a model to a declared cohort: strict named
//! feature rows, row identities, finite normalized weights, a target population and an
//! explicit outcome unit. It checks every cohort row against the retained model's own
//! support contract and issues exactly one scalar: the weighted mean
//! `sum_i w_i tau(x_i)`, an affine expectation of one declared cohort-mean treatment
//! contrast.
//!
//! What it is not. CATE points are differences; they do not give two absolute
//! potential-outcome means and they do not give a law of the contrast across units or
//! draws. The source therefore answers the expectation of an affine utility of that one
//! quantity and nothing else. Probability, quantile, nonlinear-utility and joint-law
//! requests are refused with a typed refusal that names the evidence that would support
//! them ([`LawRequest`]). No utility standard error and no interval is inherited.
//!
//! The adjacent uncertainty route is not supported. For a linear compact model that retains its
//! coefficient covariance, the variance of the weighted cohort-mean contrast
//! `phibar' V phibar` is computable and is reported as a labelled
//! [`UncalibratedStandardError`] diagnostic. It conditions on the declared cohort rows
//! and weights and on the fitted structure; it carries no cohort-sampling, support
//! selection, misspecification or extrapolation uncertainty and no calibration has been
//! measured, so it is never an interval, never reaches a decision input and every
//! interval request on it is refused ([`CohortDecisionSource::interval_refusal`]); the
//! interval of a natively fitted linear effect is a separate measured route that retains
//! the fit rows and replays the fit (`antecedent_estimate::cohort_linear_effect`).
//!
//! Population. A cohort drawn from the model's own population is decided as is. A cohort
//! from a different population is accepted only under an explicit
//! [`CovariateShiftDeclaration`]: the CATE function is declared invariant given exactly
//! the model's features (the selection node enters only the covariates), and every cohort
//! row must lie in the support contract. The result is point-only and carries the
//! declaration; it is evidence under an assumption, never target-population data.
//!
//! Trust. Numbers are always recomputed here from the retained model and the cohort rows;
//! no caller-supplied prediction is accepted. A source built from a verified analysis
//! result is native. A source built from a caller-built compact export is a declared
//! model and is never native ([`CohortDecisionSource::is_native`]).
//!
//! The artifact (`cohort_decision_source_v1`) embeds the model payload, the cohort rows
//! and the result. A consumer holding only the bytes and an independently retained
//! identity recomputes every number and refuses a changed row, weight, coefficient,
//! unit, population or parent identity even when the container is resealed.

mod artifact;
mod cohort_binding;
mod prediction;
mod retained_model;

use retained_model::ModelKind;
pub use retained_model::RetainedEffectModel;

use prediction::{compact_predictions, portable_predictions, require_exact_schema, resolve_shift};

#[cfg(test)]
use artifact::{ModelWire, SourceBody};

use cohort_binding::compensated_sum;
pub use cohort_binding::{CohortBinding, CohortColumn};

use antecedent_contract::{ExternalRefusal, QuantityCondition, QuantityRole, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext, SupportStatus};
#[cfg(test)]
use antecedent_learn::{FittedEffect, PredictionMap};
use serde::{Deserialize, Serialize};

use crate::analysis_wire::DiagnosticWire;
#[cfg(test)]
use crate::compact_export::CompactExport;
#[cfg(test)]
use crate::compact_export::TermSpec;
#[cfg(test)]
use crate::container::{
    ArtifactManifest, CompressPolicy, EncodedArtifact, SectionBytes, section_descriptor_with_policy,
};
#[cfg(test)]
use crate::convert::from_cbor;
use crate::convert::to_cbor;
use crate::error::IoError;
#[cfg(test)]
use crate::quantity_wire::ScientificQuantityWire;
#[cfg(test)]
use crate::reader::ArtifactReader;
use crate::trace::AssumptionRecordWire;

/// Body format version this module reads and writes.
pub const COHORT_DECISION_VERSION: u16 = 1;
/// Artifact kind tag in the container manifest.
pub const COHORT_DECISION_ARTIFACT_KIND: &str = crate::artifact_kinds::COHORT_DECISION_SOURCE;
/// Id of the single body section.
pub const COHORT_DECISION_SECTION: &str = "cohort_decision_source_body";
/// Largest artifact, and largest declared uncompressed payload, in bytes.
pub const MAX_COHORT_DECISION_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most cohort rows one source binds.
pub const MAX_COHORT_ROWS: usize = 200_000;

/// Intake bounds for a cohort decision source. The defaults are
/// [`MAX_COHORT_DECISION_ARTIFACT_BYTES`] and [`MAX_COHORT_ROWS`]; a consumer with a larger
/// budget passes its own to [`CohortBinding::new_within`],
/// [`CohortDecisionSource::build_within`] and [`CohortDecisionSource::consume_within`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CohortIntakeLimits {
    /// Most container bytes, and most declared uncompressed payload bytes.
    pub max_bytes: usize,
    /// Most cohort rows one source binds.
    pub max_rows: usize,
}

impl Default for CohortIntakeLimits {
    fn default() -> Self {
        Self { max_bytes: MAX_COHORT_DECISION_ARTIFACT_BYTES, max_rows: MAX_COHORT_ROWS }
    }
}

/// Largest allowed deviation of the weight sum from one.
pub const WEIGHT_SUM_TOLERANCE: f64 = 1e-9;
/// Functional identity of the model's conditional contrast.
pub const CATE_FUNCTIONAL_ID: &str = "conditional_mean_difference";
/// Functional identity of the issued cohort-mean contrast.
pub const COHORT_FUNCTIONAL_ID: &str = "mean_difference";
/// Conditioning variable carrying the cohort snapshot digest.
pub const COHORT_CONDITIONING_VARIABLE: &str = "cohort_snapshot";
/// Support basis of a retained feature box.
pub const SUPPORT_BASIS_FITTED_BOX: &str = "fitted_rows_feature_box";
/// Support basis of a declared range and refusal mask.
pub const SUPPORT_BASIS_DECLARED: &str = "declared_range_and_refusal_mask";
/// Stored statement of what the issued value means.
pub const POINT_SCOPE: &str = "weighted mean of the retained model's conditional treatment-contrast predictions over the declared cohort rows: an affine expectation of one cohort-mean contrast; not two absolute potential-outcome means, not a law of the contrast, point-only with no interval";
/// Label of the variance diagnostic.
pub const STANDARD_ERROR_BASIS: &str = "model_based_sqrt_phibar_v_phibar";
/// Stored statement of what the variance diagnostic covers.
pub const STANDARD_ERROR_SCOPE: &str = "conditions on the declared cohort rows and weights and on the fitted structure; excludes cohort sampling variation, support selection, misspecification and extrapolation";
/// Stored calibration status of the variance diagnostic.
pub const STANDARD_ERROR_CALIBRATION: &str = "calibration not_measured";
/// The invariance assumption a covariate-shift declaration states.
pub const COVARIATE_SHIFT_ASSUMPTION: &str =
    "conditional_contrast_invariant_given_declared_modifiers_under_covariate_shift";

const IDENTITY_TAG: &str = "antecedent.cohort_decision.source.v1";
const COHORT_TAG: &str = "antecedent.cohort_decision.cohort.v1";
const MODEL_TAG: &str = "antecedent.cohort_decision.model.v1";
const NEGATIVE_VARIANCE_TOLERANCE: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// A typed refusal. `detail` is a stable `cohort_decision.<snake_case>` literal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CohortRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Stable refusal detail under the `cohort_decision` namespace.
    pub detail: &'static str,
    /// The field, row, feature or identity the refusal is about.
    pub subject: String,
    /// The evidence or declaration that would support the request.
    pub remedy: &'static str,
}

impl std::fmt::Display for CohortRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {} ({})",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.subject
        )
    }
}

impl std::error::Error for CohortRefusal {}

impl CohortRefusal {
    fn new(
        code: &'static str,
        detail: &'static str,
        subject: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { code, detail, subject: subject.into(), remedy }
    }

    /// The structured external refusal form, stage `bind`.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        ExternalRefusal::new(self.code, "bind", self.detail)
            .with_offending(self.subject.clone())
            .with_remedy(self.remedy)
    }
}

fn invalid(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> CohortRefusal {
    CohortRefusal::new(antecedent_core::reason_code!("invalid_argument"), detail, subject, remedy)
}

fn schema_mismatch(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> CohortRefusal {
    CohortRefusal::new(antecedent_core::reason_code!("schema_mismatch"), detail, subject, remedy)
}

fn semantics_mismatch(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> CohortRefusal {
    CohortRefusal::new(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        detail,
        subject,
        remedy,
    )
}

fn invalid_input(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> CohortRefusal {
    CohortRefusal::new(antecedent_core::reason_code!("invalid_argument"), detail, subject, remedy)
}

fn io_refusal(error: &IoError) -> CohortRefusal {
    let detail = match error {
        IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. } => {
            antecedent_core::reason_details::cohort_decision::UNSUPPORTED_VERSION
        }
        IoError::TooLarge => antecedent_core::reason_details::cohort_decision::OVERSIZED,
        IoError::ChecksumMismatch { .. } => {
            antecedent_core::reason_details::cohort_decision::CONTAINER_CHECKSUM_MISMATCH
        }
        _ => antecedent_core::reason_details::cohort_decision::CONTAINER_INVALID,
    };
    invalid(detail, error.to_string(), "supply an artifact written by this library version")
}

/// A request the cohort-mean source cannot answer, with the evidence that would support it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LawRequest {
    /// A probability or threshold-exceedance of the outcome or of the contrast.
    Probability,
    /// A quantile or tail expectation.
    Quantile,
    /// A nonlinear utility of the contrast.
    NonlinearUtility,
    /// A joint law with another quantity, or draws of the contrast.
    JointLaw,
    /// An interval, band or posterior for the cohort mean.
    Interval,
}

impl LawRequest {
    /// The typed refusal for this request.
    #[must_use]
    pub fn refusal(self) -> CohortRefusal {
        match self {
            Self::Probability | Self::Quantile | Self::NonlinearUtility => CohortRefusal::new(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                antecedent_core::reason_details::cohort_decision::LAW_NOT_SUPPLIED,
                match self {
                    Self::Probability => "probability",
                    Self::Quantile => "quantile",
                    _ => "nonlinear_utility",
                },
                "not defined: a cohort mean of conditional contrasts is one expectation, \
                 and one expectation determines no law (a probability, a quantile or a \
                 nonlinear utility of the contrast is not a function of its mean); supply \
                 per-draw cohort utilities from a declared shared law as \
                 `DecisionDraws` (producer `cohort_decision`) and ask the decision \
                 reliability routes",
            ),
            Self::JointLaw => CohortRefusal::new(
                antecedent_core::reason_code!("joint_law_required"),
                antecedent_core::reason_details::cohort_decision::JOINT_LAW_REQUIRED,
                "joint_law",
                "not defined: conditional contrasts supply no pairing across units or with \
                 other quantities, and a coupling is a premise no mean determines; supply \
                 aligned joint draws as `DecisionDraws` (a declared shared simulation) or a \
                 declared dependence",
            ),
            Self::Interval => CohortRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                antecedent_core::reason_details::cohort_decision::INTERVAL_NOT_SUPPORTED,
                "interval",
                "a shifted cohort and a portable model are point-only: the point value (and for \
                 a linear compact model an uncalibrated model-based standard error) is the \
                 result; an interval needs a native fit over a cohort of the model's own \
                 population and measured whole-method coverage",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Cohort binding
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Retained model
// ---------------------------------------------------------------------------

/// The treatment contrast a retained model estimates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastDeclaration {
    /// Outcome variable identity.
    pub outcome: String,
    /// Regime identity, `contrast:do(t=active)-do(t=control)`.
    pub regime_id: String,
    /// Treatment variable, when the producer retained the original query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment: Option<String>,
    /// Control level, when the producer retained the original query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<f64>,
    /// Active level, when the producer retained the original query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<f64>,
}

impl ContrastDeclaration {
    /// The canonical regime identity of an `active` versus `control` contrast.
    #[must_use]
    pub fn regime_of(treatment: &str, active: f64, control: f64) -> String {
        format!("contrast:do({treatment}={active})-do({treatment}={control})")
    }
}

/// Identity and evidence of the claim the model was retained from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentIdentity {
    /// Claim identity (hex), or the compact export identity for a declared model.
    pub claim_id: String,
    /// Digest (hex) of the parent artifact bytes, or the export identity.
    pub artifact_digest: String,
    /// The parent's original diagnostics, preserved without interpretation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<DiagnosticWire>,
    /// The parent's original assumption records, preserved without interpretation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<AssumptionRecordWire>,
}

// ---------------------------------------------------------------------------
// Declared shift, receipts and expectations
// ---------------------------------------------------------------------------

/// A declared covariate shift between the model's population and the cohort's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovariateShiftDeclaration {
    /// The population the model was fitted for.
    pub source_population: String,
    /// The population the cohort is drawn from.
    pub target_population: String,
    /// The effect modifiers declared sufficient: the conditional contrast is invariant
    /// across the two populations given exactly these, which must be the model's features.
    pub invariant_modifiers: Vec<String>,
    /// Always [`COVARIATE_SHIFT_ASSUMPTION`].
    pub assumption: String,
}

impl CovariateShiftDeclaration {
    /// Declare an invariance of the conditional contrast across two populations.
    #[must_use]
    pub fn new(source_population: &str, target_population: &str, modifiers: &[&str]) -> Self {
        let mut invariant_modifiers: Vec<String> =
            modifiers.iter().map(|m| (*m).to_owned()).collect();
        invariant_modifiers.sort();
        Self {
            source_population: source_population.to_owned(),
            target_population: target_population.to_owned(),
            invariant_modifiers,
            assumption: COVARIATE_SHIFT_ASSUMPTION.to_owned(),
        }
    }
}

/// How the support contract was applied.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportReceipt {
    /// [`SUPPORT_BASIS_FITTED_BOX`] or [`SUPPORT_BASIS_DECLARED`].
    pub basis: String,
    /// Cohort rows checked.
    pub rows_checked: u64,
    /// Rows outside the contract; always zero in an issued source.
    pub rows_outside: u64,
    /// Rows the model was fitted on, when the contract records it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_rows: Option<u64>,
}

/// The model-based variance diagnostic of a linear compact model. Never an interval.
#[derive(Clone, Debug, PartialEq)]
pub struct UncalibratedStandardError {
    /// `sqrt(phibar' V phibar)`.
    pub value: f64,
    /// Always [`STANDARD_ERROR_BASIS`].
    pub basis: &'static str,
    /// Always [`STANDARD_ERROR_SCOPE`].
    pub scope: &'static str,
    /// Always [`STANDARD_ERROR_CALIBRATION`].
    pub calibration: &'static str,
}

/// Options of one build.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CohortOptions {
    /// A declared shift, required exactly when the cohort's population differs.
    pub shift: Option<CovariateShiftDeclaration>,
    /// The contrast regime the caller expects the model to estimate.
    pub expected_contrast_regime: Option<String>,
}

/// What a consumer retained out of band.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CohortExpectation {
    /// The source identity.
    pub identity: String,
    /// The parent claim identity, when retained.
    pub parent_claim: Option<String>,
    /// The cohort population, when retained.
    pub population_id: Option<String>,
    /// The declared outcome units, when retained.
    pub outcome_units: Option<String>,
    /// The contrast regime, when retained.
    pub contrast_regime: Option<String>,
    /// The cohort digest, when retained.
    pub cohort_digest: Option<String>,
}

impl CohortExpectation {
    /// Expect only the source identity.
    #[must_use]
    pub fn for_identity(identity: impl Into<String>) -> Self {
        Self { identity: identity.into(), ..Self::default() }
    }
}

// ---------------------------------------------------------------------------
// The source
// ---------------------------------------------------------------------------

/// One native, source-bound weighted-mean contrast with its provenance and support.
#[derive(Clone, Debug, PartialEq)]
pub struct CohortDecisionSource {
    model: RetainedEffectModel,
    cohort: CohortBinding,
    outcome_units: String,
    shift: Option<CovariateShiftDeclaration>,
    quantity: ScientificQuantity,
    mean_contrast: f64,
    row_contrasts: Vec<f64>,
    support: SupportReceipt,
    standard_error: Option<f64>,
    cohort_digest: String,
    identity: String,
}

fn quantity_of(
    model: &RetainedEffectModel,
    cohort: &CohortBinding,
    outcome_units: &str,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: model.contrast.outcome.clone(),
        variable_name: model.contrast.outcome.clone(),
        role: QuantityRole::Outcome,
        units: outcome_units.to_owned(),
        population_id: cohort.population_id.clone(),
        regime_id: model.contrast.regime_id.clone(),
        horizon: 0,
        functional_id: COHORT_FUNCTIONAL_ID.to_owned(),
        conditioning: vec![QuantityCondition {
            variable_id: COHORT_CONDITIONING_VARIABLE.to_owned(),
            value_id: cohort.digest(),
        }],
        transform_id: "identity".to_owned(),
    }
}

impl CohortDecisionSource {
    /// Bind `model` to `cohort`, check support and issue the weighted mean contrast.
    ///
    /// Order of refusal: expected contrast, declared units, strict feature schema,
    /// population and shift declaration, support contract and rows, prediction.
    ///
    /// # Errors
    /// Any mismatch or missing premise above; every refusal is typed and names the
    /// declaration or evidence that would support the request.
    pub fn build(
        model: &RetainedEffectModel,
        cohort: &CohortBinding,
        outcome_units: &str,
        options: &CohortOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, CohortRefusal> {
        Self::build_within(
            model,
            cohort,
            outcome_units,
            options,
            ctx,
            &CohortIntakeLimits::default(),
        )
    }

    /// [`Self::build`] under the caller's intake bounds instead of [`MAX_COHORT_ROWS`].
    ///
    /// # Errors
    /// As [`Self::build`].
    pub fn build_within(
        model: &RetainedEffectModel,
        cohort: &CohortBinding,
        outcome_units: &str,
        options: &CohortOptions,
        ctx: &ExecutionContext,
        limits: &CohortIntakeLimits,
    ) -> Result<Self, CohortRefusal> {
        cohort.validate_within(limits)?;
        if outcome_units.trim().is_empty() {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::OUTCOME_UNITS_REQUIRED,
                "outcome_units",
                "declare the outcome units; they are never inferred and never converted",
            ));
        }
        if let Some(expected) = &options.expected_contrast_regime {
            if *expected != model.contrast.regime_id {
                return Err(semantics_mismatch(
                    antecedent_core::reason_details::cohort_decision::CONTRAST_MISMATCH,
                    format!("expected={expected} model={}", model.contrast.regime_id),
                    "load the model fitted for the contrast the decision names",
                ));
            }
        }
        if let ModelKind::Compact { export, .. } = &model.kind {
            if export.body().response.units != outcome_units {
                return Err(semantics_mismatch(
                    antecedent_core::reason_details::cohort_decision::UNITS_MISMATCH,
                    format!("model={} declared={outcome_units}", export.body().response.units),
                    "declare the units the model's response is expressed in; none are converted",
                ));
            }
        }
        require_exact_schema(model, cohort)?;
        let shifted = resolve_shift(model, cohort, options.shift.as_ref())?;
        let predictions = match &model.kind {
            ModelKind::Portable { effect, features } => {
                portable_predictions(effect, features, cohort, shifted, ctx)?
            }
            ModelKind::Compact { export, .. } => compact_predictions(export, cohort, shifted, ctx)?,
        };
        let mean_contrast = compensated_sum(
            predictions.row_contrasts.iter().zip(&cohort.weights).map(|(v, w)| v * w),
        );
        if !mean_contrast.is_finite() {
            return Err(invalid_input(
                antecedent_core::reason_details::cohort_decision::NON_FINITE_RESULT,
                "mean_contrast",
                "the cohort mean is not finite; check the model and the cohort",
            ));
        }
        let rows = cohort.rows() as u64;
        let mut source = Self {
            quantity: quantity_of(model, cohort, outcome_units),
            model: model.clone(),
            cohort: cohort.clone(),
            outcome_units: outcome_units.to_owned(),
            shift: options.shift.clone(),
            mean_contrast,
            row_contrasts: predictions.row_contrasts,
            support: SupportReceipt {
                basis: predictions.basis.to_owned(),
                rows_checked: rows,
                rows_outside: 0,
                fitted_rows: predictions.fitted_rows,
            },
            standard_error: predictions.standard_error,
            cohort_digest: cohort.digest(),
            identity: String::new(),
        };
        source.identity = source.recomputed_identity()?;
        Ok(source)
    }

    fn recomputed_identity(&self) -> Result<String, CohortRefusal> {
        let mut canon = CanonicalWriter::new(IDENTITY_TAG);
        canon.word(u64::from(COHORT_DECISION_VERSION));
        canon.text(&self.model.identity).text(&self.model.parent.claim_id);
        canon.text(&self.cohort_digest).text(&self.outcome_units);
        self.quantity.write_canonical(&mut canon);
        canon.real(self.mean_contrast).opt_real(self.standard_error);
        canon.text(&self.support.basis).word(self.support.rows_checked);
        canon.word(self.support.rows_outside).opt_word(self.support.fitted_rows);
        canon.word(u64::from(self.is_native()));
        match &self.shift {
            None => {
                canon.word(0);
            }
            Some(shift) => {
                canon.word(1).bytes(&to_cbor(shift).map_err(|e| io_refusal(&e))?);
            }
        }
        Ok(canon.finish_hex())
    }

    /// The identity a consumer retains to verify the source and its artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The retained model.
    #[must_use]
    pub fn model(&self) -> &RetainedEffectModel {
        &self.model
    }

    /// The bound cohort.
    #[must_use]
    pub fn cohort(&self) -> &CohortBinding {
        &self.cohort
    }

    /// The declared outcome units.
    #[must_use]
    pub fn outcome_units(&self) -> &str {
        &self.outcome_units
    }

    /// The declared covariate shift, when the cohort's population differs.
    #[must_use]
    pub fn shift(&self) -> Option<&CovariateShiftDeclaration> {
        self.shift.as_ref()
    }

    /// The scientific coordinate of the issued value: the declared outcome, the model's
    /// contrast regime, the cohort's population, functional `mean_difference` and the
    /// cohort snapshot digest as conditioning.
    #[must_use]
    pub fn quantity(&self) -> &ScientificQuantity {
        &self.quantity
    }

    /// The weighted mean of the conditional contrast over the cohort.
    #[must_use]
    pub fn mean_contrast(&self) -> f64 {
        self.mean_contrast
    }

    /// The conditional contrast at each cohort row, in row order.
    #[must_use]
    pub fn row_contrasts(&self) -> &[f64] {
        &self.row_contrasts
    }

    /// How the support contract was applied.
    #[must_use]
    pub fn support(&self) -> &SupportReceipt {
        &self.support
    }

    /// Support label of the issued coordinate: `Supported` inside the model's own
    /// population, `Extrapolative` under a declared covariate shift.
    #[must_use]
    pub fn support_status(&self) -> SupportStatus {
        if self.shift.is_some() { SupportStatus::Extrapolative } else { SupportStatus::Supported }
    }

    /// Whether the numbers come from a verified native analysis.
    #[must_use]
    pub fn is_native(&self) -> bool {
        self.model.is_native()
    }

    /// Provider identity bound into the decision input.
    #[must_use]
    pub fn provider_id(&self) -> String {
        let route = if self.shift.is_some() {
            "cohort_decision:declared_covariate_shift"
        } else {
            "cohort_decision"
        };
        let origin = if self.is_native() { "native" } else { "declared_model" };
        format!("{origin}:{route}")
    }

    /// The model-based variance diagnostic, present for a linear compact model only. It
    /// is not an interval, is not inherited by any utility and has no measured calibration.
    #[must_use]
    pub fn uncalibrated_standard_error(&self) -> Option<UncalibratedStandardError> {
        self.standard_error.map(|value| UncalibratedStandardError {
            value,
            basis: STANDARD_ERROR_BASIS,
            scope: STANDARD_ERROR_SCOPE,
            calibration: STANDARD_ERROR_CALIBRATION,
        })
    }

    /// The refusal for any interval, band or posterior request.
    #[must_use]
    pub fn interval_refusal(&self) -> CohortRefusal {
        if self.shift.is_none() && self.model.is_linear_compact() {
            return CohortRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                antecedent_core::reason_details::cohort_decision::INTERVAL_REQUIRES_NATIVE_FIT,
                "interval",
                "an interval is issued only through the measured native-fit route, which \
                 retains the fit rows and replays the fit: fit the effect natively (outcome on \
                 treatment-by-basis with a sandwich covariance) and request the measured \
                 interval; coefficients and covariance declared by the caller stay point-only",
            );
        }
        LawRequest::Interval.refusal()
    }

    /// The refusal for a law-valued request.
    #[must_use]
    pub fn law_refusal(&self, request: LawRequest) -> CohortRefusal {
        request.refusal()
    }

    /// Assumptions that travel with the value.
    #[must_use]
    pub fn assumptions(&self) -> Vec<String> {
        let mut statements = vec![
            "the value is the weighted mean of the retained model's conditional contrast at \
             the declared cohort rows; it is not two absolute potential-outcome means"
                .to_owned(),
            "no utility standard error and no interval is inherited; the route is point-only"
                .to_owned(),
            format!("support: {} over {} rows", self.support.basis, self.support.rows_checked),
        ];
        if let Some(shift) = &self.shift {
            statements.push(format!(
                "declared covariate shift from {} to {}: {}",
                shift.source_population, shift.target_population, shift.assumption
            ));
        }
        statements
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests;

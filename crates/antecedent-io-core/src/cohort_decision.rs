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
//! requests are refused with a typed refusal that names the evidence that would license
//! them ([`LawRequest`]). No utility standard error and no interval is inherited.
//!
//! The adjacent uncertainty route is closed. For a linear compact model that retains its
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

use std::collections::BTreeSet;

use antecedent_contract::{ExternalRefusal, QuantityCondition, QuantityRole, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext, SupportStatus, VariableId};
use antecedent_kernels::numeric::CompensatedSum;
use antecedent_learn::{FittedEffect, PredictionMap};
use serde::{Deserialize, Serialize};

use crate::analysis_wire::DiagnosticWire;
use crate::compact_export::{CompactExport, ExportLimits, ExportQuery, ExportRefusal, TermSpec};
use crate::container::{
    ArtifactManifest, CompressPolicy, EncodedArtifact, SectionBytes, section_descriptor_with_policy,
};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::quantity_wire::ScientificQuantityWire;
use crate::query_wire::{CausalQueryWire, InterventionWire, TargetPopulationWire, ValueWire};
use crate::reader::ArtifactReader;
use crate::trace::AssumptionRecordWire;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

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
/// Most features one cohort binds.
pub const MAX_COHORT_FEATURES: usize = 64;
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
pub const STANDARD_ERROR_CALIBRATION: &str = "calibration unmeasured";
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
    /// The evidence or declaration that would license the request.
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

fn unlicensed(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> CohortRefusal {
    CohortRefusal::new(antecedent_core::reason_code!("cell_not_licensed"), detail, subject, remedy)
}

fn io_refusal(error: &IoError) -> CohortRefusal {
    let detail = match error {
        IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. } => {
            "cohort_decision.unsupported_version"
        }
        IoError::TooLarge => "cohort_decision.oversized",
        IoError::ChecksumMismatch { .. } => "cohort_decision.container_checksum_mismatch",
        _ => "cohort_decision.container_invalid",
    };
    invalid(detail, error.to_string(), "supply an artifact written by this library version")
}

/// A request the cohort-mean source cannot answer, with the evidence that would license it.
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
                "cohort_decision.law_not_supplied",
                match self {
                    Self::Probability => "probability",
                    Self::Quantile => "quantile",
                    _ => "nonlinear_utility",
                },
                "closed by logic: a cohort mean of conditional contrasts is one expectation, \
                 and one expectation determines no law (a probability, a quantile or a \
                 nonlinear utility of the contrast is not a function of its mean); supply \
                 per-draw cohort utilities from a declared shared law as \
                 `DecisionDraws` (producer `cohort_decision`) and ask the decision \
                 reliability routes",
            ),
            Self::JointLaw => CohortRefusal::new(
                antecedent_core::reason_code!("joint_law_required"),
                "cohort_decision.joint_law_required",
                "joint_law",
                "closed by logic: conditional contrasts supply no pairing across units or with \
                 other quantities, and a coupling is a premise no mean determines; supply \
                 aligned joint draws as `DecisionDraws` (a declared shared simulation) or a \
                 declared dependence",
            ),
            Self::Interval => CohortRefusal::new(
                antecedent_core::reason_code!("cell_not_licensed"),
                "cohort_decision.interval_not_licensed",
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

/// One named feature column of a cohort.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohortColumn {
    /// Finite numeric values, one per row.
    Numeric(Vec<f64>),
    /// Categorical levels, one per row.
    Categorical(Vec<String>),
}

impl CohortColumn {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Categorical(values) => values.len(),
        }
    }
}

/// A declared cohort: strict named feature rows with identities and normalized weights.
///
/// The cohort digest binds every field, so a changed row, weight, identity, feature name
/// or population is a different cohort.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortBinding {
    snapshot_id: String,
    population_id: String,
    row_ids: Vec<String>,
    features: Vec<String>,
    columns: Vec<CohortColumn>,
    weights: Vec<f64>,
}

/// Sum with [`CompensatedSum`], so the result is nearly independent of row order.
fn compensated_sum(terms: impl Iterator<Item = f64>) -> f64 {
    CompensatedSum::of(terms)
}

impl CohortBinding {
    /// Bind and validate a cohort.
    ///
    /// # Errors
    /// A blank snapshot or population, an empty or oversized cohort, duplicate or blank
    /// row identities or feature names, ragged or nonfinite columns, and weights that are
    /// negative, nonfinite or do not sum to one within [`WEIGHT_SUM_TOLERANCE`].
    pub fn new(
        snapshot_id: &str,
        population_id: &str,
        row_ids: Vec<String>,
        features: Vec<String>,
        columns: Vec<CohortColumn>,
        weights: Vec<f64>,
    ) -> Result<Self, CohortRefusal> {
        let cohort = Self {
            snapshot_id: snapshot_id.to_owned(),
            population_id: population_id.to_owned(),
            row_ids,
            features,
            columns,
            weights,
        };
        cohort.validate()?;
        Ok(cohort)
    }

    /// Weights `raw / sum(raw)`: an explicit normalization the caller asks for.
    ///
    /// # Errors
    /// An empty vector, a nonfinite or negative weight, or a zero total.
    pub fn normalize_weights(raw: &[f64]) -> Result<Vec<f64>, CohortRefusal> {
        let remedy = "supply finite nonnegative weights with a positive total";
        if raw.is_empty() || raw.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid("cohort_decision.weights_invalid", "weights", remedy));
        }
        let total = compensated_sum(raw.iter().copied());
        if !(total.is_finite() && total > 0.0) {
            return Err(invalid("cohort_decision.weights_invalid", "weights", remedy));
        }
        Ok(raw.iter().map(|w| w / total).collect())
    }

    /// Equal weights over `rows` rows.
    ///
    /// # Errors
    /// Zero rows.
    pub fn uniform_weights(rows: usize) -> Result<Vec<f64>, CohortRefusal> {
        if rows == 0 {
            return Err(invalid(
                "cohort_decision.cohort_empty",
                "rows",
                "bind at least one cohort row",
            ));
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "the row count is bounded by MAX_COHORT_ROWS"
        )]
        let weight = 1.0 / rows as f64;
        Ok(vec![weight; rows])
    }

    /// Re-run the construction checks, for a cohort decoded from bytes.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn validate(&self) -> Result<(), CohortRefusal> {
        let remedy = "declare a cohort with named finite feature rows and normalized weights";
        if self.snapshot_id.trim().is_empty() || self.population_id.trim().is_empty() {
            return Err(invalid(
                "cohort_decision.cohort_invalid",
                "snapshot_or_population",
                remedy,
            ));
        }
        let rows = self.weights.len();
        if rows == 0 {
            return Err(invalid("cohort_decision.cohort_empty", "rows", remedy));
        }
        if rows > MAX_COHORT_ROWS || self.features.len() > MAX_COHORT_FEATURES {
            return Err(invalid("cohort_decision.cohort_too_large", rows.to_string(), remedy));
        }
        if self.features.is_empty()
            || self.features.iter().any(|name| name.trim().is_empty())
            || self.features.iter().collect::<BTreeSet<_>>().len() != self.features.len()
            || self.columns.len() != self.features.len()
        {
            return Err(invalid("cohort_decision.cohort_invalid", "features", remedy));
        }
        if self.row_ids.len() != rows
            || self.row_ids.iter().any(|id| id.trim().is_empty())
            || self.row_ids.iter().collect::<BTreeSet<_>>().len() != rows
        {
            return Err(invalid("cohort_decision.cohort_invalid", "row_ids", remedy));
        }
        for (name, column) in self.features.iter().zip(&self.columns) {
            let finite = match column {
                CohortColumn::Numeric(values) => values.iter().all(|v| v.is_finite()),
                CohortColumn::Categorical(values) => values.iter().all(|v| !v.trim().is_empty()),
            };
            if column.len() != rows || !finite {
                return Err(invalid("cohort_decision.cohort_invalid", name.clone(), remedy));
            }
        }
        if self.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid("cohort_decision.weights_invalid", "weights", remedy));
        }
        let total = compensated_sum(self.weights.iter().copied());
        if (total - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(invalid(
                "cohort_decision.weights_not_normalized",
                format!("sum={total}"),
                "normalize the weights to sum to one (CohortBinding::normalize_weights)",
            ));
        }
        Ok(())
    }

    /// Caller-declared name of the cohort snapshot.
    #[must_use]
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    /// Population the rows are declared to come from.
    #[must_use]
    pub fn population_id(&self) -> &str {
        &self.population_id
    }

    /// Row identities, in row order.
    #[must_use]
    pub fn row_ids(&self) -> &[String] {
        &self.row_ids
    }

    /// Feature names, in the order supplied.
    #[must_use]
    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// Feature columns, aligned with [`Self::features`].
    #[must_use]
    pub fn columns(&self) -> &[CohortColumn] {
        &self.columns
    }

    /// Normalized weights, in row order.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.weights.len()
    }

    /// Canonical digest of every field, as lowercase hex. This is the cohort snapshot
    /// identity a decision contract names in its input coordinate.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut canon = CanonicalWriter::new(COHORT_TAG);
        canon.text(&self.snapshot_id).text(&self.population_id);
        canon.texts(&self.row_ids).texts(&self.features);
        canon.word(self.columns.len() as u64);
        for column in &self.columns {
            match column {
                CohortColumn::Numeric(values) => {
                    canon.text("numeric").reals(values);
                }
                CohortColumn::Categorical(values) => {
                    canon.text("categorical").texts(values);
                }
            }
        }
        canon.reals(&self.weights);
        canon.finish_hex()
    }

    fn numeric(&self, name: &str) -> Option<&[f64]> {
        let index = self.features.iter().position(|f| f == name)?;
        match &self.columns[index] {
            CohortColumn::Numeric(values) => Some(values.as_slice()),
            CohortColumn::Categorical(_) => None,
        }
    }

    fn level(&self, name: &str, row: usize) -> Option<&str> {
        let index = self.features.iter().position(|f| f == name)?;
        match &self.columns[index] {
            CohortColumn::Categorical(values) => values.get(row).map(String::as_str),
            CohortColumn::Numeric(_) => None,
        }
    }
}

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

#[derive(Clone, Debug, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "one value per decoded cohort model, held by value; boxing the compact export would not pay"
)]
enum ModelKind {
    Portable { effect: FittedEffect, features: Vec<String> },
    Compact { export: CompactExport, bytes: Vec<u8> },
}

/// A verified, immutable retained effect model with its original contrast and support
/// contract. Predictions are recomputed from it; nothing else supplies a number.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedEffectModel {
    kind: ModelKind,
    population_id: String,
    contrast: ContrastDeclaration,
    parent: ParentIdentity,
    identity: String,
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
        let consumed = crate::consume_analysis_result(bytes).map_err(|e| io_refusal(&e))?;
        if !consumed.acceptance.accepts_as_claim() {
            return Err(parent_refusal(
                "cohort_decision.parent_not_verified",
                "analysis_result",
                "supply an analysis result whose contract section verifies against its body",
            ));
        }
        let claim =
            consumed.contract.as_ref().and_then(|contract| contract.claim.as_ref()).ok_or_else(
                || parent_refusal("cohort_decision.parent_not_verified", "claim", remedy),
            )?;
        if !claim.attested.is_empty() {
            return Err(parent_refusal(
                "cohort_decision.parent_attested_not_native",
                "attested_evidence",
                "an attested external estimate supplies no native fitted payload; refit natively",
            ));
        }
        let claim_id = antecedent_core::to_hex(&claim.claim_id);
        if expected_claim.is_some_and(|expected| expected != claim_id) {
            return Err(parent_refusal(
                "cohort_decision.wrong_parent_identity",
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
                "cohort_decision.parent_query_unsupported",
                "query",
                remedy,
            ));
        };
        if !outcome_functional.is_mean() || *target_population != TargetPopulationWire::AllObserved
        {
            return Err(parent_refusal(
                "cohort_decision.parent_query_unsupported",
                "outcome_functional_or_population",
                "only a mean-effect contrast over all observed units retains a CATE map",
            ));
        }
        let (Some(control), Some(active)) =
            (level_of(control, *treatment), level_of(active, *treatment))
        else {
            return Err(parent_refusal(
                "cohort_decision.parent_query_unsupported",
                "intervention",
                "the contrast must be two hard-set levels of the treatment",
            ));
        };
        let name = |id: u32| {
            consumed.header.variable_names.get(id as usize).cloned().ok_or_else(|| {
                parent_refusal("cohort_decision.parent_query_unsupported", "variable_names", remedy)
            })
        };
        let treatment_name = name(*treatment)?;
        let outcome_name = name(*outcome)?;
        let effect = consumed.body.fitted_effect.clone().ok_or_else(|| {
            CohortRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                "cohort_decision.fitted_model_absent",
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
                "cohort_decision.fitted_model_invalid",
                "fitted_effect",
                "refit; the retained payload does not validate",
            )
        })?;
        if features.len() != effect.features.len() || features.iter().any(|f| f.trim().is_empty()) {
            return Err(invalid(
                "cohort_decision.fitted_model_invalid",
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
                "cohort_decision.link_not_identity",
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
                "cohort_decision.contrast_not_declared",
                response.variable_id.clone(),
                "build the export with an outcome response whose functional is \
                 conditional_mean_difference and whose regime is contrast:do(t=a)-do(t=b)",
            ));
        }
        let bytes = export.to_bytes("cohort_decision_model").map_err(|e| {
            invalid("cohort_decision.fitted_model_invalid", e.to_string(), "rebuild the export")
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

fn map_export_refusal(refusal: &ExportRefusal, row_id: &str, shifted: bool) -> CohortRefusal {
    match refusal.detail {
        "compact_export.out_of_support" | "compact_export.masked_region" => {
            outside_refusal(shifted, format!("row={row_id} feature_or_region={}", refusal.subject))
        }
        _ => schema_mismatch(
            "cohort_decision.feature_schema_mismatch",
            format!("{}: {}", refusal.detail, refusal.subject),
            "supply exactly the model's named features with their declared types",
        ),
    }
}

fn outside_refusal(shifted: bool, subject: String) -> CohortRefusal {
    if shifted {
        CohortRefusal::new(
            antecedent_core::reason_code!("transport_support_failure"),
            "cohort_decision.shift_support_failure",
            subject,
            "restrict the cohort to rows inside the fitted support, or supply a model fitted \
             on rows that cover the target population",
        )
    } else {
        unlicensed(
            "cohort_decision.outside_support",
            subject,
            "restrict the cohort to rows inside the retained support contract (building a \
             restricted cohort is a new, separately identified cohort)",
        )
    }
}

fn resolve_shift(
    model: &RetainedEffectModel,
    cohort: &CohortBinding,
    shift: Option<&CovariateShiftDeclaration>,
) -> Result<bool, CohortRefusal> {
    let same = cohort.population_id == model.population_id;
    match (same, shift) {
        (true, None) => Ok(false),
        (true, Some(_)) => Err(semantics_mismatch(
            "cohort_decision.shift_declaration_unneeded",
            cohort.population_id.clone(),
            "drop the shift declaration: the cohort is declared to be the model's population",
        )),
        (false, None) => Err(CohortRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "cohort_decision.shift_declaration_required",
            format!("model={} cohort={}", model.population_id, cohort.population_id),
            "declare a covariate shift: source and target population and the effect \
             modifiers (the model's features) given which the conditional contrast is \
             invariant, with every cohort row inside the support contract",
        )),
        (false, Some(declared)) => {
            let mut features = model.feature_names();
            features.sort();
            let matches_declared = declared.source_population == model.population_id
                && declared.target_population == cohort.population_id
                && declared.invariant_modifiers == features
                && declared.assumption == COVARIATE_SHIFT_ASSUMPTION;
            if matches_declared {
                Ok(true)
            } else {
                Err(CohortRefusal::new(
                    antecedent_core::reason_code!("transport_not_certified"),
                    "cohort_decision.shift_declaration_mismatch",
                    format!("model={} cohort={}", model.population_id, cohort.population_id),
                    "the declaration must name the model's population as source, the \
                     cohort's population as target and exactly the model's features as the \
                     invariant effect modifiers",
                ))
            }
        }
    }
}

fn require_exact_schema(
    model: &RetainedEffectModel,
    cohort: &CohortBinding,
) -> Result<(), CohortRefusal> {
    let expected: BTreeSet<String> = model.feature_names().into_iter().collect();
    let supplied: BTreeSet<String> = cohort.features.iter().cloned().collect();
    if expected != supplied {
        let difference: Vec<String> = expected.symmetric_difference(&supplied).cloned().collect();
        return Err(schema_mismatch(
            "cohort_decision.feature_schema_mismatch",
            difference.join(","),
            "supply exactly the model's named features: none missing and none extra",
        ));
    }
    Ok(())
}

struct Predictions {
    row_contrasts: Vec<f64>,
    basis: &'static str,
    fitted_rows: Option<u64>,
    standard_error: Option<f64>,
}

fn portable_predictions(
    effect: &FittedEffect,
    names: &[String],
    cohort: &CohortBinding,
    shifted: bool,
    ctx: &ExecutionContext,
) -> Result<Predictions, CohortRefusal> {
    let Some(contract) = &effect.support else {
        return Err(unlicensed(
            "cohort_decision.support_contract_absent",
            "fitted_effect.support",
            "refit with a producer that retains the fitted-row support contract; row-level \
             support cannot be checked without it",
        ));
    };
    let mut columns: Vec<&[f64]> = Vec::with_capacity(names.len());
    for name in names {
        let column = cohort.numeric(name).ok_or_else(|| {
            schema_mismatch(
                "cohort_decision.feature_type_mismatch",
                name.clone(),
                "supply this feature as a finite numeric column",
            )
        })?;
        columns.push(column);
    }
    let report = effect
        .check_support(&columns, cohort.rows())
        .map_err(|e| {
            schema_mismatch(
                "cohort_decision.feature_schema_mismatch",
                e.to_string(),
                "supply the model's features",
            )
        })?
        .ok_or_else(|| {
            unlicensed(
                "cohort_decision.support_contract_absent",
                "fitted_effect.support",
                "refit with a producer that retains the fitted-row support contract",
            )
        })?;
    if let (Some(row), Some(feature)) = (report.first_row, report.first_feature) {
        return Err(outside_refusal(
            shifted,
            format!(
                "row={} feature={} rows_outside={}",
                cohort.row_ids[row], names[feature], report.rows_outside
            ),
        ));
    }
    let ids: Vec<VariableId> = effect.features.iter().map(|v| VariableId::from_raw(*v)).collect();
    let row_contrasts = effect.predict(&ids, &columns, cohort.rows(), ctx).map_err(|e| {
        unlicensed(
            "cohort_decision.prediction_refused",
            e.to_string(),
            "the retained model could not predict at these rows; check the cohort features",
        )
    })?;
    Ok(Predictions {
        row_contrasts,
        basis: SUPPORT_BASIS_FITTED_BOX,
        fitted_rows: Some(contract.rows),
        standard_error: None,
    })
}

fn basis_row(
    terms: &[TermSpec],
    cohort: &CohortBinding,
    row: usize,
) -> Result<Vec<f64>, CohortRefusal> {
    let mismatch = |quantity: &str| {
        schema_mismatch(
            "cohort_decision.feature_type_mismatch",
            quantity.to_owned(),
            "supply each feature with the type its declared support has",
        )
    };
    terms
        .iter()
        .map(|term| match term {
            TermSpec::Intercept => Ok(1.0),
            TermSpec::Linear { quantity } => {
                cohort.numeric(quantity).map(|values| values[row]).ok_or_else(|| mismatch(quantity))
            }
            TermSpec::Power { quantity, degree } => cohort
                .numeric(quantity)
                .map(|values| values[row].powi(i32::try_from(*degree).unwrap_or(i32::MAX)))
                .ok_or_else(|| mismatch(quantity)),
            TermSpec::OneHot { quantity, level } => cohort
                .level(quantity, row)
                .map(|chosen| if chosen == level { 1.0 } else { 0.0 })
                .ok_or_else(|| mismatch(quantity)),
        })
        .collect()
}

fn compact_predictions(
    export: &CompactExport,
    cohort: &CohortBinding,
    shifted: bool,
    ctx: &ExecutionContext,
) -> Result<Predictions, CohortRefusal> {
    let rows = cohort.rows();
    let body = export.body();
    let width = body.terms.len();
    let mut row_contrasts = Vec::with_capacity(rows);
    let mut weighted_basis = vec![Vec::with_capacity(rows); width];
    for row in 0..rows {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid(
                "cohort_decision.cancelled",
                "cohort",
                "re-run without cancellation; no value was issued",
            ));
        }
        let mut query = ExportQuery::new();
        for (name, column) in cohort.features.iter().zip(&cohort.columns) {
            query = match column {
                CohortColumn::Numeric(values) => query.with_number(name, values[row]),
                CohortColumn::Categorical(values) => query.with_level(name, &values[row]),
            };
        }
        let point = export
            .evaluate(&query)
            .map_err(|refusal| map_export_refusal(&refusal, &cohort.row_ids[row], shifted))?;
        row_contrasts.push(point.point);
        let phi = basis_row(&body.terms, cohort, row)?;
        for (slot, value) in weighted_basis.iter_mut().zip(phi) {
            slot.push(value * cohort.weights[row]);
        }
    }
    let phibar: Vec<f64> =
        weighted_basis.into_iter().map(|terms| compensated_sum(terms.into_iter())).collect();
    let variance = compensated_sum(phibar.iter().enumerate().map(|(j, pj)| {
        let row = &body.covariance[j * width..(j + 1) * width];
        pj * compensated_sum(row.iter().zip(&phibar).map(|(v, pk)| v * pk))
    }));
    let scale: f64 =
        body.covariance.iter().step_by(width + 1).zip(&phibar).map(|(v, p)| v * p * p).sum();
    if variance < -NEGATIVE_VARIANCE_TOLERANCE * (1.0 + scale.abs()) || !variance.is_finite() {
        return Err(unlicensed(
            "cohort_decision.negative_variance",
            "phibar_v_phibar",
            "the stored covariance is not positive semidefinite on this cohort; refit",
        ));
    }
    Ok(Predictions {
        row_contrasts,
        basis: SUPPORT_BASIS_DECLARED,
        fitted_rows: None,
        standard_error: Some(variance.max(0.0).sqrt()),
    })
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
    /// declaration or evidence that would license the request.
    pub fn build(
        model: &RetainedEffectModel,
        cohort: &CohortBinding,
        outcome_units: &str,
        options: &CohortOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, CohortRefusal> {
        cohort.validate()?;
        if outcome_units.trim().is_empty() {
            return Err(invalid(
                "cohort_decision.outcome_units_required",
                "outcome_units",
                "declare the outcome units; they are never inferred and never converted",
            ));
        }
        if let Some(expected) = &options.expected_contrast_regime {
            if *expected != model.contrast.regime_id {
                return Err(semantics_mismatch(
                    "cohort_decision.contrast_mismatch",
                    format!("expected={expected} model={}", model.contrast.regime_id),
                    "load the model fitted for the contrast the decision names",
                ));
            }
        }
        if let ModelKind::Compact { export, .. } = &model.kind {
            if export.body().response.units != outcome_units {
                return Err(semantics_mismatch(
                    "cohort_decision.units_mismatch",
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
            return Err(unlicensed(
                "cohort_decision.non_finite_result",
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
                antecedent_core::reason_code!("cell_not_licensed"),
                "cohort_decision.interval_requires_native_fit",
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
// Artifact
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "wire shape: boxing would change the serialized form and the public variant shape"
)]
enum ModelWire {
    Portable {
        effect: FittedEffect,
        features: Vec<String>,
        population_id: String,
        contrast: ContrastDeclaration,
        parent: ParentIdentity,
    },
    Compact {
        export: Vec<u8>,
        identity: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBody {
    version: u16,
    model: ModelWire,
    model_identity: String,
    cohort: CohortBinding,
    cohort_digest: String,
    outcome_units: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shift: Option<CovariateShiftDeclaration>,
    quantity: ScientificQuantityWire,
    mean_contrast: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    standard_error: Option<f64>,
    support: SupportReceipt,
    scope: String,
    identity: String,
}

impl CohortDecisionSource {
    fn to_body(&self) -> SourceBody {
        let model = match &self.model.kind {
            ModelKind::Portable { effect, features } => ModelWire::Portable {
                effect: effect.clone(),
                features: features.clone(),
                population_id: self.model.population_id.clone(),
                contrast: self.model.contrast.clone(),
                parent: self.model.parent.clone(),
            },
            ModelKind::Compact { export, bytes } => {
                ModelWire::Compact { export: bytes.clone(), identity: export.identity().to_owned() }
            }
        };
        SourceBody {
            version: COHORT_DECISION_VERSION,
            model,
            model_identity: self.model.identity.clone(),
            cohort: self.cohort.clone(),
            cohort_digest: self.cohort_digest.clone(),
            outcome_units: self.outcome_units.clone(),
            shift: self.shift.clone(),
            quantity: ScientificQuantityWire::from(&self.quantity),
            mean_contrast: self.mean_contrast,
            standard_error: self.standard_error,
            support: self.support.clone(),
            scope: POINT_SCOPE.to_owned(),
            identity: self.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// A blank artifact id, an oversized payload or an encoding failure.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, CohortRefusal> {
        if artifact_id.trim().is_empty() {
            return Err(invalid(
                "cohort_decision.missing_artifact_id",
                "artifact_id",
                "name the artifact",
            ));
        }
        let payload = to_cbor(&self.to_body()).map_err(|e| io_refusal(&e))?;
        let descriptor = section_descriptor_with_policy(
            COHORT_DECISION_SECTION,
            "application/cbor",
            &payload,
            CompressPolicy::Never,
        );
        let library_version = SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
            .map_err(|e| io_refusal(&e))?;
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: crate::migrate::STABLE_FORMAT,
                minimum_reader_version: crate::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(COHORT_DECISION_ARTIFACT_KIND.into()),
                library_version,
                artifact_id: artifact_id.into(),
                sections: vec![descriptor],
                provenance: ProvenanceWire { note: "cohort_decision_source".into() },
            },
            sections: vec![SectionBytes::new(COHORT_DECISION_SECTION, payload)],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).map_err(|e| io_refusal(&e))?;
        if bytes.len() > MAX_COHORT_DECISION_ARTIFACT_BYTES {
            return Err(invalid(
                "cohort_decision.oversized",
                bytes.len().to_string(),
                "bind a smaller cohort",
            ));
        }
        Ok(bytes)
    }

    /// Independent consumer: decode with bounded sizes, rebuild the model and the cohort
    /// from the stored fields, recompute support, every prediction, the mean and the
    /// diagnostic, and require the retained expectations and identity.
    ///
    /// # Errors
    /// An oversized or malformed artifact, an unknown version, a changed field (even when
    /// the container was resealed), and any retained expectation that differs.
    pub fn consume(bytes: &[u8], expectation: &CohortExpectation) -> Result<Self, CohortRefusal> {
        Self::consume_with(bytes, expectation, &ExecutionContext::production_default(0))
    }

    /// As [`Self::consume`] under a caller's execution context (cancellation, budgets).
    ///
    /// # Errors
    /// As [`Self::consume`].
    pub fn consume_with(
        bytes: &[u8],
        expectation: &CohortExpectation,
        ctx: &ExecutionContext,
    ) -> Result<Self, CohortRefusal> {
        let body = decode_body(bytes)?;
        let model = rebuild_model(&body)?;
        let options = CohortOptions { shift: body.shift.clone(), expected_contrast_regime: None };
        let recomputed = Self::build(&model, &body.cohort, &body.outcome_units, &options, ctx)?;
        check_expectation(&recomputed, expectation)?;
        if recomputed.to_body() != body
            || recomputed.mean_contrast.to_bits() != body.mean_contrast.to_bits()
        {
            return Err(invalid(
                "cohort_decision.replay_mismatch",
                "body",
                "the stored value, support or quantity differs from its recomputation; \
                 re-export from the producer",
            ));
        }
        if recomputed.identity != expectation.identity {
            return Err(invalid(
                "cohort_decision.identity_unexpected",
                expectation.identity.clone(),
                "the artifact is not the source whose identity was retained",
            ));
        }
        Ok(recomputed)
    }
}

fn decode_body(bytes: &[u8]) -> Result<SourceBody, CohortRefusal> {
    if bytes.len() > MAX_COHORT_DECISION_ARTIFACT_BYTES {
        return Err(invalid(
            "cohort_decision.oversized",
            bytes.len().to_string(),
            "bind a smaller cohort",
        ));
    }
    let mut reader =
        ArtifactReader::open_seek(std::io::Cursor::new(bytes)).map_err(|e| io_refusal(&e))?;
    let (layout_ok, declared) = {
        let manifest = reader.manifest();
        let layout_ok = manifest.artifact_kind
            == ArtifactKind::Other(COHORT_DECISION_ARTIFACT_KIND.into())
            && manifest.sections.len() == 1
            && manifest.sections[0].id == COHORT_DECISION_SECTION;
        let declared = manifest
            .sections
            .iter()
            .try_fold(0u64, |total, section| total.checked_add(section.uncompressed_size));
        (layout_ok, declared)
    };
    if !layout_ok {
        return Err(invalid(
            "cohort_decision.layout_invalid",
            "manifest",
            "supply a cohort decision source",
        ));
    }
    if crate::error::declared_exceeds_usize_limit(declared, MAX_COHORT_DECISION_ARTIFACT_BYTES) {
        return Err(invalid(
            "cohort_decision.oversized",
            "declared_uncompressed_size",
            "bind a smaller cohort",
        ));
    }
    let section = reader.load_section(COHORT_DECISION_SECTION).map_err(|e| io_refusal(&e))?;
    if section.as_bytes().len() > MAX_COHORT_DECISION_ARTIFACT_BYTES {
        return Err(invalid("cohort_decision.oversized", "section", "bind a smaller cohort"));
    }
    let body: SourceBody = from_cbor(section.as_bytes()).map_err(|e| io_refusal(&e))?;
    if body.version != COHORT_DECISION_VERSION {
        return Err(invalid(
            "cohort_decision.unsupported_version",
            body.version.to_string(),
            "supply an artifact written by this library version",
        ));
    }
    Ok(body)
}

fn rebuild_model(body: &SourceBody) -> Result<RetainedEffectModel, CohortRefusal> {
    let model = match &body.model {
        ModelWire::Portable { effect, features, population_id, contrast, parent } => {
            RetainedEffectModel::from_portable_parts(
                effect.clone(),
                features.clone(),
                population_id.clone(),
                contrast.clone(),
                parent.clone(),
            )?
        }
        ModelWire::Compact { export: original_bytes, identity } => {
            let export = CompactExport::consume(original_bytes, &ExportLimits::default(), identity)
                .map_err(|refusal| {
                    invalid(
                        "cohort_decision.fitted_model_invalid",
                        format!("{}: {}", refusal.detail, refusal.subject),
                        "re-export the model from the producer",
                    )
                })?;
            let mut model = RetainedEffectModel::from_compact_export(&export)?;
            // Replaying a verified model must retain its original container. Re-exporting
            // it with this reader's producer version changes nested bytes even though the
            // scientific model identity, predictions and support are identical.
            model.kind = ModelKind::Compact { export, bytes: original_bytes.clone() };
            model
        }
    };
    if model.identity != body.model_identity {
        return Err(invalid(
            "cohort_decision.model_identity_mismatch",
            body.model_identity.clone(),
            "the stored model identity differs from the model's recomputed identity",
        ));
    }
    Ok(model)
}

fn check_expectation(
    source: &CohortDecisionSource,
    expectation: &CohortExpectation,
) -> Result<(), CohortRefusal> {
    if expectation.parent_claim.as_deref().is_some_and(|c| c != source.model.parent.claim_id) {
        return Err(invalid(
            "cohort_decision.wrong_parent_identity",
            source.model.parent.claim_id.clone(),
            "the artifact was issued from a different parent claim than the one retained",
        ));
    }
    if expectation.population_id.as_deref().is_some_and(|p| p != source.cohort.population_id) {
        return Err(semantics_mismatch(
            "cohort_decision.population_mismatch",
            source.cohort.population_id.clone(),
            "the cohort population differs from the retained population",
        ));
    }
    if expectation.outcome_units.as_deref().is_some_and(|u| u != source.outcome_units) {
        return Err(semantics_mismatch(
            "cohort_decision.units_mismatch",
            source.outcome_units.clone(),
            "the declared outcome units differ from the retained units; none are converted",
        ));
    }
    if expectation.contrast_regime.as_deref().is_some_and(|c| c != source.model.contrast.regime_id)
    {
        return Err(semantics_mismatch(
            "cohort_decision.contrast_mismatch",
            source.model.contrast.regime_id.clone(),
            "the model estimates a different contrast than the one retained",
        ));
    }
    if expectation.cohort_digest.as_deref().is_some_and(|d| d != source.cohort_digest) {
        return Err(invalid(
            "cohort_decision.cohort_changed",
            source.cohort_digest.clone(),
            "the rows, weights or identities differ from the retained cohort snapshot",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

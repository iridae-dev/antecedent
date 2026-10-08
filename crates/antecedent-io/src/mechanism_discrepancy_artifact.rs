//! B3 mechanism discrepancy artifact (`mechanism_discrepancy_v1`).
//!
//! A bounded, checksummed, sectioned container holding one source-target mechanism discrepancy
//! diagnostic:
//!
//! * `mechanism_discrepancy_meta` (CBOR): version, feature marker, the frozen null, the
//!   asymptotic-Wald claim, the **unmeasured** calibration coordinate, the level and power, the
//!   compared-intercept declaration, the declared dependence, the population / selection
//!   alignment text, each population's label, measurement contract (node, units, parents,
//!   protocol id) and row count, the identity digests, the stored result and the standing
//!   caveats;
//! * `mechanism_discrepancy_numbers` (little-endian `f64`): for the source and then the target,
//!   the summary statistics `X'X` (row-major), `X'y` and `y'y`, in the declared design order
//!   (intercept, then the declared parents).
//!
//! A consumer trusts none of the stored results. It rebuilds the request from the stored
//! declarations and summary statistics, reruns the core diagnostic (both OLS fits, the Wald
//! statistic, the p-value, the per-coefficient breakdown with Holm adjustment and the minimal
//! detectable differences) and refuses unless every stored value is bit-identical. It then
//! recomputes the identity digests and compares them with the stored ones and, when the caller
//! retained an identity of its own, with that: a changed measurement contract, summary statistic,
//! level, alignment or null is refused even when the artifact was resealed consistently.
//!
//! What this does **not** say: non-rejection never certifies invariance
//! (`non_rejection_certifies_invariance` is `false` and a stored `true` is refused), the
//! diagnostic informs only a selection node on the compared node, Type I error and power are
//! unmeasured (`calibration = "unmeasured"`), and no causal or interval license is granted.
//! Unknown major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_estimate::EstimationError;
use antecedent_estimate::mechanism_discrepancy::{
    DiscrepancyConclusion, DiscrepancyOptions, MECHANISM_DISCREPANCY_ALIGNMENT,
    MECHANISM_DISCREPANCY_CAVEAT, MECHANISM_DISCREPANCY_DEPENDENCE,
    MECHANISM_DISCREPANCY_INFERENCE_CLAIM, MECHANISM_DISCREPANCY_MAX_PARENTS,
    MECHANISM_DISCREPANCY_NULL, MECHANISM_DISCREPANCY_POWER_CAVEAT, MechanismDiscrepancyResult,
    MechanismMeasurement, MechanismSummary, ParentSpec, PopulationFit, SampleDependence,
    test_mechanism_discrepancy_from_summaries,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const MECHANISM_DISCREPANCY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const MECHANISM_DISCREPANCY_ARTIFACT_FEATURE: &str = "mechanism_discrepancy_v1";
/// Calibration coordinate of this test: Type I error and power were not measured.
pub const MECHANISM_DISCREPANCY_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_MECHANISM_DISCREPANCY_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

const ARTIFACT_KIND: &str = "mechanism_discrepancy_v1";
const META_SECTION: &str = "mechanism_discrepancy_meta";
const NUMBERS_SECTION: &str = "mechanism_discrepancy_numbers";
const DEPENDENCE_INDEPENDENT: &str = "independent";
const DEPENDENCE_SHARED_UNITS: &str = "shared_units";
const DEPENDENCE_UNKNOWN: &str = "unknown";

/// Why a mechanism discrepancy artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MechanismDiscrepancyArtifactError {
    /// The bytes do not decode as this format.
    #[error("mechanism discrepancy artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported mechanism discrepancy artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker, claim, calibration coordinate, caveat or invariance flag that is not
    /// this format's.
    #[error("unsupported mechanism discrepancy semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("mechanism discrepancy consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed mechanism discrepancy artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("mechanism_discrepancy.wrong_contract: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored result differs from the recomputed one.
    #[error("stored mechanism discrepancy result does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("mechanism discrepancy artifact does not encode: {0}")]
    Encode(String),
    /// The core diagnostic refused (registered reason code; the message begins with its detail).
    #[error("{message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// `detail: explanation`.
        message: String,
    },
}

impl MechanismDiscrepancyArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a core refusal and for a changed identity field; absent for corruption,
    /// unsupported versions and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, message } => {
                let (detail, text) = message.split_once(": ").unwrap_or((message.as_str(), ""));
                Some((*code, detail.to_owned(), text.to_owned()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "mechanism_discrepancy.wrong_contract".to_owned(),
                format!("{field} changed"),
            )),
            _ => None,
        }
    }
}

impl From<EstimationError> for MechanismDiscrepancyArtifactError {
    fn from(error: EstimationError) -> Self {
        match error {
            EstimationError::Refused { code, message }
            | EstimationError::RefusedWithFields { code, message, .. } => {
                Self::Refused { code, message }
            }
            other => Self::Malformed(other.to_string()),
        }
    }
}

impl From<crate::IoError> for MechanismDiscrepancyArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn invalid_request(message: &str) -> MechanismDiscrepancyArtifactError {
    MechanismDiscrepancyArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("mechanism_discrepancy.invalid_request: {message}"),
    }
}

/// One parent on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentWire {
    /// Parent variable name.
    pub name: String,
    /// Unit / coordinate of the parent.
    pub unit: String,
}

/// The measurement contract of one population on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementWire {
    /// The node whose mechanism is compared.
    pub node: String,
    /// Unit / coordinate of the node.
    pub node_unit: String,
    /// The parents, in design order.
    pub parents: Vec<ParentWire>,
    /// Declared measurement protocol id.
    pub protocol_id: String,
}

impl MeasurementWire {
    fn measurement(&self) -> MechanismMeasurement {
        MechanismMeasurement {
            node: self.node.clone(),
            node_unit: self.node_unit.clone(),
            parents: self
                .parents
                .iter()
                .map(|p| ParentSpec { name: p.name.clone(), unit: p.unit.clone() })
                .collect(),
            protocol_id: self.protocol_id.clone(),
        }
    }

    fn from_measurement(m: &MechanismMeasurement) -> Self {
        Self {
            node: m.node.clone(),
            node_unit: m.node_unit.clone(),
            parents: m
                .parents
                .iter()
                .map(|p| ParentWire { name: p.name.clone(), unit: p.unit.clone() })
                .collect(),
            protocol_id: m.protocol_id.clone(),
        }
    }
}

/// One population's declaration and summary statistics, as supplied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationWire {
    /// Population label.
    pub label: String,
    /// The measurement contract.
    pub measurement: MeasurementWire,
    /// Row count.
    pub n: u64,
    /// Row-major `p x p` `X'X` in design order (intercept, then the parents).
    pub xtx: Vec<f64>,
    /// `X'y`.
    pub xty: Vec<f64>,
    /// `y'y`.
    pub yty: f64,
}

/// A complete diagnostic request: what the producer declares and the core consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismDiscrepancyRequestWire {
    /// The source population.
    pub source: PopulationWire,
    /// The target population.
    pub target: PopulationWire,
    /// Whether the intercept enters the compared coefficient vector.
    pub compare_intercept: bool,
    /// Level of the decisions.
    pub alpha: f64,
    /// Power of the minimal detectable differences.
    pub power: f64,
    /// `independent`, `shared_units` or `unknown`; only `independent` runs.
    pub dependence: String,
}

/// Identity digests of one diagnostic: BLAKE3 over a canonical byte encoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismDiscrepancyIdentity {
    /// Both measurement contracts (node, units, parents, protocol ids).
    pub measurement_id: String,
    /// The source label, row count and summary statistics.
    pub source_evidence_id: String,
    /// The target label, row count and summary statistics.
    pub target_evidence_id: String,
    /// The frozen null.
    pub null: String,
    /// Compared-intercept declaration, level, power, dependence and the alignment text.
    pub design_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// One fit on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitWire {
    /// Population label.
    pub label: String,
    /// Row count.
    pub n: u64,
    /// Residual degrees of freedom.
    pub residual_df: u32,
    /// Residual variance.
    pub residual_variance: f64,
    /// OLS coefficients in canonical order (intercept, then parents by name).
    pub coefficients: Vec<f64>,
    /// Their standard errors.
    pub standard_errors: Vec<f64>,
}

impl FitWire {
    fn from_fit(fit: &PopulationFit) -> Self {
        Self {
            label: fit.label.clone(),
            n: fit.n as u64,
            residual_df: u32::try_from(fit.residual_df).unwrap_or(u32::MAX),
            residual_variance: fit.residual_variance,
            coefficients: fit.coefficients.clone(),
            standard_errors: fit.standard_errors.clone(),
        }
    }
}

/// One compared coefficient on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoefficientWire {
    /// Coefficient name.
    pub name: String,
    /// Source coefficient.
    pub source: f64,
    /// Target coefficient.
    pub target: f64,
    /// `target - source`.
    pub difference: f64,
    /// Standard error of the difference.
    pub standard_error: f64,
    /// Standardized difference.
    pub z: f64,
    /// Two-sided normal p-value, unadjusted.
    pub p_value: f64,
    /// Holm-adjusted p-value.
    pub p_holm: f64,
    /// Whether the Holm-adjusted p-value is below the level.
    pub rejected: bool,
    /// Minimal detectable difference at the declared level and power.
    pub minimal_detectable_difference: f64,
}

/// The stored result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultWire {
    /// The Wald statistic.
    pub statistic: f64,
    /// Degrees of freedom.
    pub degrees_of_freedom: u32,
    /// Chi-square tail probability.
    pub p_value: f64,
    /// `not_rejected` or `rejected`; non-rejection never certifies invariance.
    pub conclusion: String,
    /// Always `false`.
    pub non_rejection_certifies_invariance: bool,
    /// `z_{1-alpha/2} + z_{power}`.
    pub detectability_factor: f64,
    /// What differences were detectable.
    pub power_statement: String,
    /// The variables whose selection node the diagnostic informs.
    pub informs_selection_on: Vec<String>,
    /// Canonical coefficient names of the fits.
    pub coefficient_names: Vec<String>,
    /// Source fit.
    pub source: FitWire,
    /// Target fit.
    pub target: FitWire,
    /// Compared coefficients with Holm adjustment and detectability.
    pub coefficients: Vec<CoefficientWire>,
}

impl ResultWire {
    fn from_result(result: &MechanismDiscrepancyResult) -> Self {
        Self {
            statistic: result.test.statistic,
            degrees_of_freedom: u32::try_from(result.test.degrees_of_freedom).unwrap_or(u32::MAX),
            p_value: result.test.p_value,
            conclusion: match result.conclusion {
                DiscrepancyConclusion::NotRejected => "not_rejected",
                DiscrepancyConclusion::Rejected => "rejected",
            }
            .to_owned(),
            non_rejection_certifies_invariance: result.non_rejection_certifies_invariance,
            detectability_factor: result.detectability_factor,
            power_statement: result.power_statement.clone(),
            informs_selection_on: result.informs_selection_on.clone(),
            coefficient_names: result.coefficient_names.clone(),
            source: FitWire::from_fit(&result.source),
            target: FitWire::from_fit(&result.target),
            coefficients: result
                .coefficients
                .iter()
                .map(|c| CoefficientWire {
                    name: c.name.clone(),
                    source: c.source,
                    target: c.target,
                    difference: c.difference,
                    standard_error: c.standard_error,
                    z: c.z,
                    p_value: c.p_value,
                    p_holm: c.p_holm,
                    rejected: c.rejected,
                    minimal_detectable_difference: c.minimal_detectable_difference,
                })
                .collect(),
        }
    }
}

/// A population's declaration without its numbers (those live in the numbers section).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationMeta {
    /// Population label.
    pub label: String,
    /// The measurement contract.
    pub measurement: MeasurementWire,
    /// Row count.
    pub n: u64,
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismDiscrepancyMeta {
    /// Major version ([`MECHANISM_DISCREPANCY_ARTIFACT_VERSION`]).
    pub version: u16,
    /// Feature marker ([`MECHANISM_DISCREPANCY_ARTIFACT_FEATURE`]).
    pub feature: String,
    /// The frozen null.
    pub null: String,
    /// Always the asymptotic-Wald claim with unmeasured calibration.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Level of the decisions.
    pub alpha: f64,
    /// Power of the minimal detectable differences.
    pub power: f64,
    /// Whether the intercept entered the comparison.
    pub compare_intercept: bool,
    /// `independent`, `shared_units` or `unknown`.
    pub dependence: String,
    /// The population / selection alignment text.
    pub alignment: String,
    /// The dependence assumption text.
    pub dependence_assumption: String,
    /// Source declaration.
    pub source: PopulationMeta,
    /// Target declaration.
    pub target: PopulationMeta,
    /// Identity digests.
    pub identity: MechanismDiscrepancyIdentity,
    /// The stored result.
    pub result: ResultWire,
    /// Standing caveats: non-rejection is not invariance, error rates unmeasured.
    pub caveats: Vec<String>,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismDiscrepancyReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The frozen null.
    pub null: String,
    /// Inference claim.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Level of the decisions.
    pub alpha: f64,
    /// Power of the minimal detectable differences.
    pub power: f64,
    /// Whether the intercept entered the comparison.
    pub compare_intercept: bool,
    /// Declared dependence.
    pub dependence: String,
    /// The population / selection alignment text.
    pub alignment: String,
    /// The dependence assumption text.
    pub dependence_assumption: String,
    /// The shared measurement contract (parents in name order).
    pub measurement: MeasurementWire,
    /// The diagnostic result.
    pub result: ResultWire,
    /// Standing caveats.
    pub caveats: Vec<String>,
    /// Identity digests.
    pub identity: MechanismDiscrepancyIdentity,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

fn dependence_of(kind: &str) -> Result<SampleDependence, MechanismDiscrepancyArtifactError> {
    match kind {
        DEPENDENCE_INDEPENDENT => Ok(SampleDependence::Independent),
        DEPENDENCE_SHARED_UNITS => Ok(SampleDependence::SharedUnits),
        DEPENDENCE_UNKNOWN => Ok(SampleDependence::Unknown),
        _ => Err(invalid_request("unknown dependence kind")),
    }
}

fn summary_of(
    wire: &PopulationWire,
) -> Result<MechanismSummary, MechanismDiscrepancyArtifactError> {
    if wire.measurement.parents.len() > MECHANISM_DISCREPANCY_MAX_PARENTS {
        return Err(MechanismDiscrepancyArtifactError::LimitsExceeded("parents"));
    }
    Ok(MechanismSummary {
        label: wire.label.clone(),
        measurement: wire.measurement.measurement(),
        n: usize::try_from(wire.n)
            .map_err(|_| invalid_request("row count is not representable"))?,
        xtx: wire.xtx.clone(),
        xty: wire.xty.clone(),
        yty: wire.yty,
    })
}

/// Length-prefixed canonical byte builder hashed with BLAKE3.
struct Canon(Vec<u8>);

impl Canon {
    fn new(tag: &str) -> Self {
        let mut canon = Self(Vec::new());
        canon.text(tag);
        canon
    }

    fn word(&mut self, word: u64) {
        self.0.extend_from_slice(&word.to_le_bytes());
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        self.0.extend_from_slice(text.as_bytes());
    }

    fn real(&mut self, value: f64) {
        self.word(value.to_bits());
    }

    fn finish(&self) -> String {
        blake3::hash(&self.0).to_hex().to_string()
    }
}

fn measurement_digest(canon: &mut Canon, m: &MechanismMeasurement) {
    canon.text(&m.node);
    canon.text(&m.node_unit);
    canon.text(&m.protocol_id);
    canon.word(m.parents.len() as u64);
    for parent in &m.parents {
        canon.text(&parent.name);
        canon.text(&parent.unit);
    }
}

fn evidence_digest(tag: &str, s: &MechanismSummary) -> String {
    let mut canon = Canon::new(tag);
    canon.text(&s.label);
    canon.word(s.n as u64);
    for v in s.xtx.iter().chain(&s.xty) {
        canon.real(*v);
    }
    canon.real(s.yty);
    canon.finish()
}

fn identity_of(
    source: &MechanismSummary,
    target: &MechanismSummary,
    options: &DiscrepancyOptions,
    dependence_kind: &str,
    result: &MechanismDiscrepancyResult,
) -> MechanismDiscrepancyIdentity {
    let mut measurement = Canon::new("mechanism_discrepancy_v1.measurement");
    measurement_digest(&mut measurement, &source.measurement);
    measurement_digest(&mut measurement, &target.measurement);

    let mut design = Canon::new("mechanism_discrepancy_v1.design");
    design.word(u64::from(options.compare_intercept));
    design.real(options.alpha);
    design.real(options.power);
    design.text(dependence_kind);
    design.text(result.alignment);
    design.text(result.dependence_assumption);
    for node in &result.informs_selection_on {
        design.text(node);
    }

    let measurement_id = measurement.finish();
    let source_evidence_id = evidence_digest("mechanism_discrepancy_v1.source", source);
    let target_evidence_id = evidence_digest("mechanism_discrepancy_v1.target", target);
    let design_id = design.finish();
    let mut whole = Canon::new("mechanism_discrepancy_v1.identity");
    for part in [
        measurement_id.as_str(),
        source_evidence_id.as_str(),
        target_evidence_id.as_str(),
        MECHANISM_DISCREPANCY_NULL,
        design_id.as_str(),
    ] {
        whole.text(part);
    }
    MechanismDiscrepancyIdentity {
        measurement_id,
        source_evidence_id,
        target_evidence_id,
        null: MECHANISM_DISCREPANCY_NULL.to_owned(),
        design_id,
        digest: whole.finish(),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &MechanismDiscrepancyIdentity,
    other: &MechanismDiscrepancyIdentity,
) -> Option<&'static str> {
    if stored.measurement_id != other.measurement_id {
        Some("measurement")
    } else if stored.source_evidence_id != other.source_evidence_id {
        Some("source_evidence")
    } else if stored.target_evidence_id != other.target_evidence_id {
        Some("target_evidence")
    } else if stored.null != other.null {
        Some("null")
    } else if stored.design_id != other.design_id {
        Some("design")
    } else if stored.digest != other.digest {
        Some("identity")
    } else {
        None
    }
}

fn numbers_of(request: &MechanismDiscrepancyRequestWire) -> Vec<u8> {
    let mut bytes = Vec::new();
    for population in [&request.source, &request.target] {
        let reals = population
            .xtx
            .iter()
            .chain(&population.xty)
            .copied()
            .chain(std::iter::once(population.yty));
        for value in reals {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

struct Computed {
    result: MechanismDiscrepancyResult,
    identity: MechanismDiscrepancyIdentity,
}

fn options_of(
    request: &MechanismDiscrepancyRequestWire,
) -> Result<DiscrepancyOptions, MechanismDiscrepancyArtifactError> {
    Ok(DiscrepancyOptions {
        compare_intercept: request.compare_intercept,
        alpha: request.alpha,
        power: request.power,
        dependence: dependence_of(&request.dependence)?,
    })
}

fn compute(
    request: &MechanismDiscrepancyRequestWire,
) -> Result<Computed, MechanismDiscrepancyArtifactError> {
    let source = summary_of(&request.source)?;
    let target = summary_of(&request.target)?;
    let options = options_of(request)?;
    let result = test_mechanism_discrepancy_from_summaries(&source, &target, &options)?;
    let identity = identity_of(&source, &target, &options, &request.dependence, &result);
    Ok(Computed { result, identity })
}

fn population_of(meta: &PopulationMeta, values: &[f64]) -> PopulationWire {
    let p = meta.measurement.parents.len() + 1;
    PopulationWire {
        label: meta.label.clone(),
        measurement: meta.measurement.clone(),
        n: meta.n,
        xtx: values[..p * p].to_vec(),
        xty: values[p * p..p * p + p].to_vec(),
        yty: values[p * p + p],
    }
}

fn request_of(
    meta: &MechanismDiscrepancyMeta,
    numbers: &[u8],
) -> Result<MechanismDiscrepancyRequestWire, MechanismDiscrepancyArtifactError> {
    let (ks, kt) = (meta.source.measurement.parents.len(), meta.target.measurement.parents.len());
    if ks > MECHANISM_DISCREPANCY_MAX_PARENTS || kt > MECHANISM_DISCREPANCY_MAX_PARENTS {
        return Err(MechanismDiscrepancyArtifactError::LimitsExceeded("parents"));
    }
    let width = |k: usize| (k + 1) * (k + 1) + (k + 1) + 1;
    let (ws, wt) = (width(ks), width(kt));
    if numbers.len() != (ws + wt) * 8 {
        return Err(MechanismDiscrepancyArtifactError::Malformed("numbers section length".into()));
    }
    let values: Vec<f64> = numbers
        .chunks_exact(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            f64::from_le_bytes(word)
        })
        .collect();
    Ok(MechanismDiscrepancyRequestWire {
        source: population_of(&meta.source, &values[..ws]),
        target: population_of(&meta.target, &values[ws..]),
        compare_intercept: meta.compare_intercept,
        alpha: meta.alpha,
        power: meta.power,
        dependence: meta.dependence.clone(),
    })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`MechanismDiscrepancyArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &MechanismDiscrepancyMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, MechanismDiscrepancyArtifactError> {
    let encode = |e: crate::IoError| MechanismDiscrepancyArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(MechanismDiscrepancyArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(encode)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .map_err(encode)?,
            artifact_id: artifact_id.into(),
            sections: vec![
                section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                section_descriptor(NUMBERS_SECTION, "application/octet-stream", numbers),
            ],
            provenance: ProvenanceWire {
                note: "mechanism_discrepancy_wald_unmeasured_never_certifies_invariance".into(),
            },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(NUMBERS_SECTION, numbers.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_MECHANISM_DISCREPANCY_ARTIFACT_BYTES {
        return Err(MechanismDiscrepancyArtifactError::LimitsExceeded("artifact bytes"));
    }
    Ok(bytes)
}

/// Decode the two sections of a container, refusing another version before the metadata is
/// interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(MechanismDiscrepancyMeta, Vec<u8>), MechanismDiscrepancyArtifactError> {
    if bytes.len() > MAX_MECHANISM_DISCREPANCY_ARTIFACT_BYTES {
        return Err(MechanismDiscrepancyArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != NUMBERS_SECTION
    {
        return Err(MechanismDiscrepancyArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > MAX_MECHANISM_DISCREPANCY_ARTIFACT_BYTES as u64) {
        return Err(MechanismDiscrepancyArtifactError::LimitsExceeded("artifact bytes"));
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != MECHANISM_DISCREPANCY_ARTIFACT_VERSION {
        return Err(MechanismDiscrepancyArtifactError::UnsupportedVersion {
            version: peek.version,
        });
    }
    let meta: MechanismDiscrepancyMeta = from_cbor(meta_section.as_bytes())?;
    let numbers = reader.load_section(NUMBERS_SECTION)?;
    Ok((meta, numbers.as_bytes().to_vec()))
}

fn check_semantics(
    meta: &MechanismDiscrepancyMeta,
) -> Result<(), MechanismDiscrepancyArtifactError> {
    use MechanismDiscrepancyArtifactError as E;
    if meta.feature != MECHANISM_DISCREPANCY_ARTIFACT_FEATURE {
        return Err(E::UnsupportedSemantics("feature marker"));
    }
    if meta.null != MECHANISM_DISCREPANCY_NULL {
        return Err(E::IdentityMismatch { field: "null" });
    }
    if meta.inference_claim != MECHANISM_DISCREPANCY_INFERENCE_CLAIM {
        return Err(E::UnsupportedSemantics("inference claim"));
    }
    if meta.calibration != MECHANISM_DISCREPANCY_CALIBRATION {
        return Err(E::UnsupportedSemantics("calibration"));
    }
    if meta.caveats != [MECHANISM_DISCREPANCY_CAVEAT, MECHANISM_DISCREPANCY_POWER_CAVEAT] {
        return Err(E::UnsupportedSemantics("caveats"));
    }
    if meta.alignment != MECHANISM_DISCREPANCY_ALIGNMENT {
        return Err(E::UnsupportedSemantics("alignment"));
    }
    if meta.dependence_assumption != MECHANISM_DISCREPANCY_DEPENDENCE {
        return Err(E::UnsupportedSemantics("dependence assumption"));
    }
    if meta.result.non_rejection_certifies_invariance {
        return Err(E::UnsupportedSemantics("non-rejection certifies invariance"));
    }
    Ok(())
}

/// A produced or consumed discrepancy artifact: the declared request, the recomputed result and
/// the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismDiscrepancyArtifact {
    meta: MechanismDiscrepancyMeta,
    request: MechanismDiscrepancyRequestWire,
    result: MechanismDiscrepancyResult,
}

impl MechanismDiscrepancyArtifact {
    /// Run the core diagnostic on `request` and seal the answer with its identity.
    ///
    /// # Errors
    /// The core diagnostic's refusals (`route_not_supported` for incomparable measurements or
    /// unknown dependence, `invalid_argument` for bad inputs) and the format's bounds.
    pub fn seal(
        request: &MechanismDiscrepancyRequestWire,
    ) -> Result<Self, MechanismDiscrepancyArtifactError> {
        let computed = compute(request)?;
        let meta = MechanismDiscrepancyMeta {
            version: MECHANISM_DISCREPANCY_ARTIFACT_VERSION,
            feature: MECHANISM_DISCREPANCY_ARTIFACT_FEATURE.to_owned(),
            null: MECHANISM_DISCREPANCY_NULL.to_owned(),
            inference_claim: MECHANISM_DISCREPANCY_INFERENCE_CLAIM.to_owned(),
            calibration: MECHANISM_DISCREPANCY_CALIBRATION.to_owned(),
            alpha: request.alpha,
            power: request.power,
            compare_intercept: request.compare_intercept,
            dependence: request.dependence.clone(),
            alignment: MECHANISM_DISCREPANCY_ALIGNMENT.to_owned(),
            dependence_assumption: MECHANISM_DISCREPANCY_DEPENDENCE.to_owned(),
            source: PopulationMeta {
                label: request.source.label.clone(),
                measurement: request.source.measurement.clone(),
                n: request.source.n,
            },
            target: PopulationMeta {
                label: request.target.label.clone(),
                measurement: request.target.measurement.clone(),
                n: request.target.n,
            },
            identity: computed.identity,
            result: ResultWire::from_result(&computed.result),
            caveats: vec![
                MECHANISM_DISCREPANCY_CAVEAT.to_owned(),
                MECHANISM_DISCREPANCY_POWER_CAVEAT.to_owned(),
            ],
        };
        Ok(Self { meta, request: request.clone(), result: computed.result })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &MechanismDiscrepancyMeta {
        &self.meta
    }

    /// The request the diagnostic was run on.
    #[must_use]
    pub fn request(&self) -> &MechanismDiscrepancyRequestWire {
        &self.request
    }

    /// The core diagnostic result.
    #[must_use]
    pub fn result(&self) -> &MechanismDiscrepancyResult {
        &self.result
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &MechanismDiscrepancyIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> MechanismDiscrepancyReportWire {
        MechanismDiscrepancyReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            null: self.meta.null.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            calibration: self.meta.calibration.clone(),
            alpha: self.meta.alpha,
            power: self.meta.power,
            compare_intercept: self.meta.compare_intercept,
            dependence: self.meta.dependence.clone(),
            alignment: self.meta.alignment.clone(),
            dependence_assumption: self.meta.dependence_assumption.clone(),
            measurement: MeasurementWire::from_measurement(&self.result.measurement),
            result: self.meta.result.clone(),
            caveats: self.meta.caveats.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, MechanismDiscrepancyArtifactError> {
        encode_parts(&self.meta, &numbers_of(&self.request), artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected` is an identity the consumer retained independently (for example from the
    /// producer's result); when given, every identity field must match it, so a resealed change
    /// of measurement contract, summary statistics, level, alignment or null is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics (including a stored claim that
    /// non-rejection certifies invariance), a changed identity
    /// ([`MechanismDiscrepancyArtifactError::IdentityMismatch`]), a stored result that does not
    /// replay, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&MechanismDiscrepancyIdentity>,
    ) -> Result<Self, MechanismDiscrepancyArtifactError> {
        let (meta, numbers) = decode_parts(bytes)?;
        check_semantics(&meta)?;
        let request = request_of(&meta, &numbers)?;
        let computed = compute(&request)?;
        if let Some(field) = identity_diff(&meta.identity, &computed.identity) {
            return Err(MechanismDiscrepancyArtifactError::IdentityMismatch { field });
        }
        if meta.result != ResultWire::from_result(&computed.result) {
            return Err(MechanismDiscrepancyArtifactError::ResultMismatch(
                "statistic, p-value, coefficients or detectability",
            ));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &computed.identity) {
                return Err(MechanismDiscrepancyArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta, request, result: computed.result })
    }
}

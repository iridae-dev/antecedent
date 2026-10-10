//! F18 `EffectConstancy` artifact (`effect_constancy_v1`).
//!
//! A bounded, checksummed, sectioned container holding one global effect-constancy test:
//!
//! * `effect_constancy.meta` (CBOR): version, feature marker, the frozen null, the point-only
//!   claim, the **unmeasured** calibration coordinate, the level, the declared partitions
//!   (label, coordinate, support, estimand), the dependence kind, the multiplicity family,
//!   the identity digests, the stored test result and the standing caveats;
//! * `effect_constancy.numbers` (little-endian `f64`): the per-partition effects, their
//!   standard errors and, for a dependent partition, the full row-major covariance, in the
//!   order the partitions are declared.
//!
//! A consumer trusts none of the stored results. It rebuilds the request from the stored
//! declarations and numbers, reruns the core test (heterogeneity statistic, p-value, per-pair
//! contrasts and their Holm adjustment) and refuses unless every stored value is bit-identical.
//! It then recomputes the identity digests from the same inputs and compares them with the
//! stored ones and, when the caller retained an identity of its own, with that: a changed
//! partition identity, estimand, covariance, null, multiplicity family or evidence is refused
//! even when the artifact was resealed consistently.
//!
//! What this does **not** say: non-rejection does not prove constancy, the test's Type I error
//! and power are unmeasured (`calibration = "unmeasured"`), and the artifact grants no causal
//! or interval license (`inference_claim = "point_only"`). Unknown major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_estimate::EstimationError;
use antecedent_estimate::effect_constancy::{
    ConstancyConclusion, ContrastFamily, EFFECT_CONSTANCY_INFERENCE_CLAIM,
    EFFECT_CONSTANCY_MAX_PARTITIONS, EFFECT_CONSTANCY_NULL, EffectConstancyResult,
    EffectEstimandIdentity, HeterogeneityStatistic, NON_REJECTION_CAVEAT, POWER_CAVEAT,
    PartitionDependence, PartitionEstimate, PartitionSupport, test_effect_constancy,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const EFFECT_CONSTANCY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const EFFECT_CONSTANCY_ARTIFACT_FEATURE: &str = "effect_constancy_v1";
/// Calibration coordinate of this test: Type I error and power were not measured.
pub const EFFECT_CONSTANCY_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_EFFECT_CONSTANCY_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

const ARTIFACT_KIND: &str = "effect_constancy_v1";
const META_SECTION: &str = "effect_constancy_meta";
const NUMBERS_SECTION: &str = "effect_constancy_numbers";
const DEPENDENCE_INDEPENDENT: &str = "independent";
const DEPENDENCE_COVARIANCE: &str = "covariance";
const FAMILY_ALL_PAIRS: &str = "all_pairs";
const FAMILY_AGAINST_REFERENCE: &str = "against_reference";

/// Why an effect-constancy artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum EffectConstancyArtifactError {
    /// The bytes do not decode as this format.
    #[error("effect constancy artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported effect constancy artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker, claim, calibration coordinate or caveat that is not this format's.
    #[error("unsupported effect constancy semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("effect constancy consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed effect constancy artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("effect_constancy.wrong_contract: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored result differs from the recomputed one.
    #[error("stored effect constancy result does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("effect constancy artifact does not encode: {0}")]
    Encode(String),
    /// The core test refused (registered reason code; the message begins with its detail).
    #[error("{message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// `detail: explanation`.
        message: String,
    },
}

impl EffectConstancyArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a core-test refusal and for a changed identity field; absent for corruption,
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
                "effect_constancy.wrong_contract".to_owned(),
                format!("{field} changed"),
            )),
            _ => None,
        }
    }
}

impl From<EstimationError> for EffectConstancyArtifactError {
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

impl From<crate::IoError> for EffectConstancyArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn invalid_request(message: &str) -> EffectConstancyArtifactError {
    EffectConstancyArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("effect_constancy.invalid_request: {message}"),
    }
}

/// The shared effect estimand on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectEstimandWire {
    /// The effect definition.
    pub estimand: String,
    /// Units / scale of the effect.
    pub units: String,
    /// Regime or contrast the effect is defined under.
    pub regime: String,
    /// Population (and horizon) the effect is defined for.
    pub population: String,
}

impl EffectEstimandWire {
    fn identity(&self) -> EffectEstimandIdentity {
        EffectEstimandIdentity {
            estimand: self.estimand.clone(),
            units: self.units.clone(),
            regime: self.regime.clone(),
            population: self.population.clone(),
        }
    }

    fn from_identity(identity: &EffectEstimandIdentity) -> Self {
        Self {
            estimand: identity.estimand.clone(),
            units: identity.units.clone(),
            regime: identity.regime.clone(),
            population: identity.population.clone(),
        }
    }
}

/// One partition's declaration and estimate, as supplied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionWire {
    /// Unique partition label.
    pub label: String,
    /// Typed coordinate of the partition.
    pub coordinate: String,
    /// `supported`, `partial` or `unsupported`.
    pub support: String,
    /// The estimand the effect estimates.
    pub estimand: EffectEstimandWire,
    /// The effect estimate.
    pub effect: f64,
    /// Its standard error.
    pub standard_error: f64,
}

/// Declared dependence between partition estimates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceWire {
    /// `independent` or `covariance`.
    pub kind: String,
    /// Row-major `k x k` covariance in declaration order; empty when independent.
    #[serde(default)]
    pub covariance: Vec<f64>,
}

/// Declared multiplicity family of the contrasts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyWire {
    /// `all_pairs` or `against_reference`.
    pub kind: String,
    /// The reference partition label for `against_reference`.
    #[serde(default)]
    pub reference: Option<String>,
}

/// A complete constancy request: what the producer declares and the core test consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectConstancyRequestWire {
    /// Partitions in declaration order.
    pub partitions: Vec<PartitionWire>,
    /// Dependence between the partition estimates.
    pub dependence: DependenceWire,
    /// Multiplicity family.
    pub family: FamilyWire,
    /// Level of the decisions.
    pub alpha: f64,
}

/// Identity digests of one constancy test. Each is BLAKE3 over a canonical byte encoding in
/// label order, so permuting the declared partitions leaves every digest unchanged.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectConstancyIdentity {
    /// Labels, coordinates and support of the partitions.
    pub partition_id: String,
    /// The shared estimand.
    pub estimand_id: String,
    /// Dependence kind and the full covariance.
    pub covariance_id: String,
    /// The frozen null.
    pub null: String,
    /// The multiplicity family and level.
    pub family_id: String,
    /// The per-partition effects and standard errors.
    pub evidence_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// One multiplicity-adjusted contrast on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastWire {
    /// Minuend partition label.
    pub left: String,
    /// Subtrahend partition label.
    pub right: String,
    /// `effect(left) - effect(right)`.
    pub difference: f64,
    /// Standard error of the difference.
    pub standard_error: f64,
    /// Standardized difference.
    pub z: f64,
    /// Unadjusted two-sided normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value over the declared family.
    pub p_holm: f64,
    /// Whether the Holm-adjusted p-value is below the level.
    pub rejected: bool,
}

/// The stored test result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultWire {
    /// `cochran_q` or `wald_chi_square`.
    pub statistic_kind: String,
    /// The heterogeneity statistic.
    pub statistic: f64,
    /// Degrees of freedom.
    pub degrees_of_freedom: u32,
    /// Chi-square tail probability.
    pub p_value: f64,
    /// `not_rejected` or `rejected`; non-rejection never proves constancy.
    pub conclusion: String,
    /// Inverse-variance pooled effect; present only for independent partitions.
    pub pooled_effect: Option<f64>,
    /// Contrasts of the declared family with Holm adjustment.
    pub contrasts: Vec<ContrastWire>,
}

impl ResultWire {
    fn from_result(result: &EffectConstancyResult) -> Self {
        Self {
            statistic_kind: match result.test.statistic_kind {
                HeterogeneityStatistic::CochranQ => "cochran_q",
                HeterogeneityStatistic::WaldChiSquare => "wald_chi_square",
            }
            .to_owned(),
            statistic: result.test.statistic,
            degrees_of_freedom: u32::try_from(result.test.degrees_of_freedom).unwrap_or(u32::MAX),
            p_value: result.test.p_value,
            conclusion: match result.conclusion {
                ConstancyConclusion::NotRejected => "not_rejected",
                ConstancyConclusion::Rejected => "rejected",
            }
            .to_owned(),
            pooled_effect: result.pooled_effect,
            contrasts: result
                .contrasts
                .iter()
                .map(|c| ContrastWire {
                    left: c.left.clone(),
                    right: c.right.clone(),
                    difference: c.difference,
                    standard_error: c.standard_error,
                    z: c.z,
                    p_value: c.p_value,
                    p_holm: c.p_holm,
                    rejected: c.rejected,
                })
                .collect(),
        }
    }
}

/// A partition's declaration without its numbers (those live in the numbers section).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionMeta {
    /// Unique partition label.
    pub label: String,
    /// Typed coordinate.
    pub coordinate: String,
    /// `supported`, `partial` or `unsupported`.
    pub support: String,
    /// The estimand the effect estimates.
    pub estimand: EffectEstimandWire,
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectConstancyMeta {
    /// Major version ([`EFFECT_CONSTANCY_ARTIFACT_VERSION`]).
    pub version: u16,
    /// Feature marker ([`EFFECT_CONSTANCY_ARTIFACT_FEATURE`]).
    pub feature: String,
    /// The frozen null.
    pub null: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Level of the decisions.
    pub alpha: f64,
    /// Partitions in declaration order.
    pub partitions: Vec<PartitionMeta>,
    /// `independent` or `covariance`.
    pub dependence: String,
    /// The multiplicity family.
    pub family: FamilyWire,
    /// Identity digests.
    pub identity: EffectConstancyIdentity,
    /// The stored result.
    pub result: ResultWire,
    /// Standing caveats: non-rejection is not proof, error rates unmeasured.
    pub caveats: Vec<String>,
}

/// One partition as reported, in canonical (label) order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionReportWire {
    /// Partition label.
    pub label: String,
    /// Typed coordinate.
    pub coordinate: String,
    /// Coordinate support.
    pub support: String,
    /// Effect estimate.
    pub effect: f64,
    /// Standard error.
    pub standard_error: f64,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectConstancyReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The frozen null.
    pub null: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Level of the decisions.
    pub alpha: f64,
    /// The shared estimand.
    pub estimand: EffectEstimandWire,
    /// Partitions in canonical (label) order with their effects.
    pub partitions: Vec<PartitionReportWire>,
    /// `independent` or `covariance`.
    pub dependence: String,
    /// The multiplicity family.
    pub family: FamilyWire,
    /// The test result.
    pub result: ResultWire,
    /// Standing caveats.
    pub caveats: Vec<String>,
    /// Identity digests.
    pub identity: EffectConstancyIdentity,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

fn support_of(tag: &str) -> Result<PartitionSupport, EffectConstancyArtifactError> {
    match tag {
        "supported" => Ok(PartitionSupport::Supported),
        "partial" => Ok(PartitionSupport::Partial),
        "unsupported" => Ok(PartitionSupport::Unsupported),
        _ => Err(invalid_request("unknown coordinate support tag")),
    }
}

const fn support_tag(support: PartitionSupport) -> &'static str {
    match support {
        PartitionSupport::Supported => "supported",
        PartitionSupport::Partial => "partial",
        PartitionSupport::Unsupported => "unsupported",
    }
}

fn dependence_of(
    wire: &DependenceWire,
) -> Result<PartitionDependence, EffectConstancyArtifactError> {
    match wire.kind.as_str() {
        DEPENDENCE_INDEPENDENT if wire.covariance.is_empty() => {
            Ok(PartitionDependence::Independent)
        }
        DEPENDENCE_INDEPENDENT => {
            Err(invalid_request("an independent partition set carries no covariance"))
        }
        DEPENDENCE_COVARIANCE => {
            if wire.covariance.len()
                > EFFECT_CONSTANCY_MAX_PARTITIONS * EFFECT_CONSTANCY_MAX_PARTITIONS
            {
                return Err(EffectConstancyArtifactError::LimitsExceeded("covariance"));
            }
            Ok(PartitionDependence::Covariance(wire.covariance.clone()))
        }
        _ => Err(invalid_request("unknown dependence kind")),
    }
}

fn family_of(wire: &FamilyWire) -> Result<ContrastFamily, EffectConstancyArtifactError> {
    match (wire.kind.as_str(), &wire.reference) {
        (FAMILY_ALL_PAIRS, None) => Ok(ContrastFamily::AllPairs),
        (FAMILY_AGAINST_REFERENCE, Some(reference)) => {
            Ok(ContrastFamily::AgainstReference(reference.clone()))
        }
        _ => Err(invalid_request("unknown or incomplete multiplicity family")),
    }
}

fn estimate_of(wire: &PartitionWire) -> Result<PartitionEstimate, EffectConstancyArtifactError> {
    Ok(PartitionEstimate {
        label: wire.label.clone(),
        coordinate: wire.coordinate.clone(),
        support: support_of(&wire.support)?,
        estimand: wire.estimand.identity(),
        effect: wire.effect,
        standard_error: wire.standard_error,
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

fn identity_of(
    estimates: &[PartitionEstimate],
    dependence: &PartitionDependence,
    family: &ContrastFamily,
    alpha: f64,
) -> EffectConstancyIdentity {
    let k = estimates.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| estimates[a].label.cmp(&estimates[b].label).then(a.cmp(&b)));

    let mut partitions = Canon::new("effect_constancy_v1.partitions");
    partitions.word(k as u64);
    for &i in &order {
        partitions.text(&estimates[i].label);
        partitions.text(&estimates[i].coordinate);
        partitions.text(support_tag(estimates[i].support));
    }

    let mut estimand = Canon::new("effect_constancy_v1.estimand");
    let shared = &estimates[0].estimand;
    estimand.text(&shared.estimand);
    estimand.text(&shared.units);
    estimand.text(&shared.regime);
    estimand.text(&shared.population);

    let mut covariance = Canon::new("effect_constancy_v1.covariance");
    match dependence {
        PartitionDependence::Independent => covariance.text(DEPENDENCE_INDEPENDENT),
        PartitionDependence::Covariance(matrix) => {
            covariance.text(DEPENDENCE_COVARIANCE);
            for &i in &order {
                for &j in &order {
                    covariance.real(matrix[i * k + j]);
                }
            }
        }
    }

    let mut family_id = Canon::new("effect_constancy_v1.family");
    match family {
        ContrastFamily::AllPairs => family_id.text(FAMILY_ALL_PAIRS),
        ContrastFamily::AgainstReference(reference) => {
            family_id.text(FAMILY_AGAINST_REFERENCE);
            family_id.text(reference);
        }
    }
    family_id.real(alpha);

    let mut evidence = Canon::new("effect_constancy_v1.evidence");
    for &i in &order {
        evidence.real(estimates[i].effect);
        evidence.real(estimates[i].standard_error);
    }

    let partition_id = partitions.finish();
    let estimand_id = estimand.finish();
    let covariance_id = covariance.finish();
    let family_digest = family_id.finish();
    let evidence_id = evidence.finish();
    let mut whole = Canon::new("effect_constancy_v1.identity");
    for part in [
        partition_id.as_str(),
        estimand_id.as_str(),
        covariance_id.as_str(),
        EFFECT_CONSTANCY_NULL,
        family_digest.as_str(),
        evidence_id.as_str(),
    ] {
        whole.text(part);
    }
    EffectConstancyIdentity {
        partition_id,
        estimand_id,
        covariance_id,
        null: EFFECT_CONSTANCY_NULL.to_owned(),
        family_id: family_digest,
        evidence_id,
        digest: whole.finish(),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &EffectConstancyIdentity,
    other: &EffectConstancyIdentity,
) -> Option<&'static str> {
    if stored.partition_id != other.partition_id {
        Some("partition_identity")
    } else if stored.estimand_id != other.estimand_id {
        Some("estimand")
    } else if stored.covariance_id != other.covariance_id {
        Some("covariance")
    } else if stored.null != other.null {
        Some("null")
    } else if stored.family_id != other.family_id {
        Some("multiplicity_family")
    } else if stored.evidence_id != other.evidence_id {
        Some("evidence")
    } else if stored.digest != other.digest {
        Some("identity")
    } else {
        None
    }
}

fn numbers_of(request: &EffectConstancyRequestWire) -> Vec<u8> {
    let mut bytes = Vec::new();
    let reals = request
        .partitions
        .iter()
        .map(|p| p.effect)
        .chain(request.partitions.iter().map(|p| p.standard_error))
        .chain(request.dependence.covariance.iter().copied());
    for value in reals {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

struct Computed {
    result: EffectConstancyResult,
    identity: EffectConstancyIdentity,
}

fn compute(request: &EffectConstancyRequestWire) -> Result<Computed, EffectConstancyArtifactError> {
    if request.partitions.len() > EFFECT_CONSTANCY_MAX_PARTITIONS {
        return Err(EffectConstancyArtifactError::LimitsExceeded("partitions"));
    }
    let estimates: Vec<PartitionEstimate> =
        request.partitions.iter().map(estimate_of).collect::<Result<_, _>>()?;
    let dependence = dependence_of(&request.dependence)?;
    let family = family_of(&request.family)?;
    let result = test_effect_constancy(&estimates, &dependence, &family, request.alpha)?;
    let identity = identity_of(&estimates, &dependence, &family, request.alpha);
    Ok(Computed { result, identity })
}

fn request_of(
    meta: &EffectConstancyMeta,
    numbers: &[u8],
) -> Result<EffectConstancyRequestWire, EffectConstancyArtifactError> {
    let k = meta.partitions.len();
    if k > EFFECT_CONSTANCY_MAX_PARTITIONS {
        return Err(EffectConstancyArtifactError::LimitsExceeded("partitions"));
    }
    let dependent = match meta.dependence.as_str() {
        DEPENDENCE_INDEPENDENT => false,
        DEPENDENCE_COVARIANCE => true,
        _ => return Err(EffectConstancyArtifactError::Malformed("dependence kind".into())),
    };
    let count = 2 * k + if dependent { k * k } else { 0 };
    if numbers.len() != count * 8 {
        return Err(EffectConstancyArtifactError::Malformed("numbers section length".into()));
    }
    let values: Vec<f64> = numbers
        .chunks_exact(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            f64::from_le_bytes(word)
        })
        .collect();
    let partitions = meta
        .partitions
        .iter()
        .enumerate()
        .map(|(i, p)| PartitionWire {
            label: p.label.clone(),
            coordinate: p.coordinate.clone(),
            support: p.support.clone(),
            estimand: p.estimand.clone(),
            effect: values[i],
            standard_error: values[k + i],
        })
        .collect();
    Ok(EffectConstancyRequestWire {
        partitions,
        dependence: DependenceWire {
            kind: meta.dependence.clone(),
            covariance: values[2 * k..].to_vec(),
        },
        family: meta.family.clone(),
        alpha: meta.alpha,
    })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`EffectConstancyArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &EffectConstancyMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, EffectConstancyArtifactError> {
    let encode = |e: crate::IoError| EffectConstancyArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(EffectConstancyArtifactError::Encode("missing artifact id".into()));
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
            provenance: ProvenanceWire { note: "effect_constancy_point_only_unmeasured".into() },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(NUMBERS_SECTION, numbers.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_EFFECT_CONSTANCY_ARTIFACT_BYTES {
        return Err(EffectConstancyArtifactError::LimitsExceeded("artifact bytes"));
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
) -> Result<(EffectConstancyMeta, Vec<u8>), EffectConstancyArtifactError> {
    if bytes.len() > MAX_EFFECT_CONSTANCY_ARTIFACT_BYTES {
        return Err(EffectConstancyArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != NUMBERS_SECTION
    {
        return Err(EffectConstancyArtifactError::Malformed("unsupported container layout".into()));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > MAX_EFFECT_CONSTANCY_ARTIFACT_BYTES as u64) {
        return Err(EffectConstancyArtifactError::LimitsExceeded("artifact bytes"));
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != EFFECT_CONSTANCY_ARTIFACT_VERSION {
        return Err(EffectConstancyArtifactError::UnsupportedVersion { version: peek.version });
    }
    let meta: EffectConstancyMeta = from_cbor(meta_section.as_bytes())?;
    let numbers = reader.load_section(NUMBERS_SECTION)?;
    Ok((meta, numbers.as_bytes().to_vec()))
}

/// A produced or consumed constancy artifact: the declared request, the recomputed result and
/// the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectConstancyArtifact {
    meta: EffectConstancyMeta,
    request: EffectConstancyRequestWire,
    result: EffectConstancyResult,
}

impl EffectConstancyArtifact {
    /// Run the core test on `request` and seal the answer with its identity.
    ///
    /// # Errors
    /// The core test's refusals (`route_not_supported` for incompatible partitions,
    /// `invalid_argument` for bad inputs) and the format's bounds.
    pub fn seal(
        request: &EffectConstancyRequestWire,
    ) -> Result<Self, EffectConstancyArtifactError> {
        let computed = compute(request)?;
        let meta = EffectConstancyMeta {
            version: EFFECT_CONSTANCY_ARTIFACT_VERSION,
            feature: EFFECT_CONSTANCY_ARTIFACT_FEATURE.to_owned(),
            null: EFFECT_CONSTANCY_NULL.to_owned(),
            inference_claim: EFFECT_CONSTANCY_INFERENCE_CLAIM.to_owned(),
            calibration: EFFECT_CONSTANCY_CALIBRATION.to_owned(),
            alpha: request.alpha,
            partitions: request
                .partitions
                .iter()
                .map(|p| PartitionMeta {
                    label: p.label.clone(),
                    coordinate: p.coordinate.clone(),
                    support: p.support.clone(),
                    estimand: p.estimand.clone(),
                })
                .collect(),
            dependence: request.dependence.kind.clone(),
            family: request.family.clone(),
            identity: computed.identity,
            result: ResultWire::from_result(&computed.result),
            caveats: vec![NON_REJECTION_CAVEAT.to_owned(), POWER_CAVEAT.to_owned()],
        };
        Ok(Self { meta, request: request.clone(), result: computed.result })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &EffectConstancyMeta {
        &self.meta
    }

    /// The request the test was run on, in declaration order.
    #[must_use]
    pub fn request(&self) -> &EffectConstancyRequestWire {
        &self.request
    }

    /// The core test result.
    #[must_use]
    pub fn result(&self) -> &EffectConstancyResult {
        &self.result
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &EffectConstancyIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> EffectConstancyReportWire {
        EffectConstancyReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            null: self.meta.null.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            calibration: self.meta.calibration.clone(),
            alpha: self.meta.alpha,
            estimand: EffectEstimandWire::from_identity(&self.result.estimand),
            partitions: self
                .result
                .partitions
                .iter()
                .map(|p| PartitionReportWire {
                    label: p.label.clone(),
                    coordinate: p.coordinate.clone(),
                    support: support_tag(p.support).to_owned(),
                    effect: p.effect,
                    standard_error: p.standard_error,
                })
                .collect(),
            dependence: self.meta.dependence.clone(),
            family: self.meta.family.clone(),
            result: self.meta.result.clone(),
            caveats: self.meta.caveats.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, EffectConstancyArtifactError> {
        encode_parts(&self.meta, &numbers_of(&self.request), artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected` is an identity the consumer retained independently (for example from the
    /// producer's result); when given, every identity field must match it, so a resealed
    /// change of partitions, estimand, covariance, null or multiplicity family is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity
    /// ([`EffectConstancyArtifactError::IdentityMismatch`]), a stored result that does not
    /// replay, or a core-test refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&EffectConstancyIdentity>,
    ) -> Result<Self, EffectConstancyArtifactError> {
        let (meta, numbers) = decode_parts(bytes)?;
        if meta.feature != EFFECT_CONSTANCY_ARTIFACT_FEATURE {
            return Err(EffectConstancyArtifactError::UnsupportedSemantics("feature marker"));
        }
        if meta.null != EFFECT_CONSTANCY_NULL {
            return Err(EffectConstancyArtifactError::IdentityMismatch { field: "null" });
        }
        if meta.inference_claim != EFFECT_CONSTANCY_INFERENCE_CLAIM {
            return Err(EffectConstancyArtifactError::UnsupportedSemantics("inference claim"));
        }
        if meta.calibration != EFFECT_CONSTANCY_CALIBRATION {
            return Err(EffectConstancyArtifactError::UnsupportedSemantics("calibration"));
        }
        if meta.caveats != [NON_REJECTION_CAVEAT, POWER_CAVEAT] {
            return Err(EffectConstancyArtifactError::UnsupportedSemantics("caveats"));
        }
        let request = request_of(&meta, &numbers)?;
        let computed = compute(&request)?;
        if let Some(field) = identity_diff(&meta.identity, &computed.identity) {
            return Err(EffectConstancyArtifactError::IdentityMismatch { field });
        }
        if meta.result != ResultWire::from_result(&computed.result) {
            return Err(EffectConstancyArtifactError::ResultMismatch(
                "statistic, p-value or contrasts",
            ));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &computed.identity) {
                return Err(EffectConstancyArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta, request, result: computed.result })
    }
}

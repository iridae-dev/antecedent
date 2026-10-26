//! Portable evidence ledger (`evidence_ledger_v1`).
//!
//! A bounded, checksummed container holding the full operation history of one
//! evidence ledger and, optionally, the summary-data syntheses computed from
//! its results:
//!
//! * every ledger operation (add, update, replace, withdraw, supersede,
//!   revalidate-downstream, combine) with its declaration, its chained
//!   identity and its downstream effect;
//! * the head operation identity and the ledger state digest;
//! * per synthesis, the study summaries, the declared dependence or prior and
//!   the computed values, bound to the ledger result they are about.
//!
//! A consumer trusts none of the stored table. It replays every operation
//! through the ledger (which recomputes the source identities, the overlap
//! policy, the refusals, the mass accounting and the downstream
//! classification), requires each stored operation identity, effect, head and
//! state digest to match, requires each synthesis to match its ledger result
//! (same sources, same declared dependence, the covariance bound by digest)
//! and recomputes its values with the numerical routines. A resealed edit of
//! any stored value is refused, and a consumer that retained the identity
//! independently refuses a consistently resealed ledger of another history.
//!
//! The artifact decides and records nothing beyond the ledger's own rules:
//! it carries no interval, posterior or coverage statement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{
    AppliedOperation, ClaimClass, CombinationRequest, DetectedOverlap, EvidenceLedger, LedgerError,
    LedgerOperation, Lineage, ObservationBasis, ObservationId, OperationEffect, OperationId,
    PoolingDependence, ResultId, ResultRevalidation, Revalidation, SourceDeclaration, SourceId,
    SourceKind, SynthesisKind, SynthesisOperation, TransportIdentity, parse_digest_hex,
};
use antecedent_core::{CanonicalWriter, SemanticDigest};
use antecedent_stats::synthesis::{
    MeanPrior, TauPrior, equal_study_gaussian, fixed_effect_gls, fixed_effect_inverse_variance,
    hierarchical_normal_borrowing, pooled_dependence_envelope, random_effects_dersimonian_laird,
    random_effects_reml,
};
use serde::{Deserialize, Serialize};

use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};

/// The artifact major version this reader writes and accepts.
pub const EVIDENCE_LEDGER_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const EVIDENCE_LEDGER_ARTIFACT_FEATURE: &str = "evidence_ledger_v1";
/// Most bytes an evidence ledger artifact may occupy.
pub const MAX_EVIDENCE_LEDGER_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;
/// Most syntheses an artifact may carry.
pub const MAX_LEDGER_SYNTHESES: usize = 1024;

/// Intake bounds a consumer applies to a stored evidence ledger artifact. The defaults are
/// [`MAX_EVIDENCE_LEDGER_ARTIFACT_BYTES`], the contract's default operation bound and
/// [`MAX_LEDGER_SYNTHESES`]; a caller with a larger budget passes its own to
/// [`EvidenceLedgerArtifact::from_bytes_within`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvidenceLedgerIntakeLimits {
    /// Most container bytes.
    pub max_bytes: usize,
    /// Most ledger operations replayed.
    pub max_operations: usize,
    /// Most syntheses recomputed.
    pub max_syntheses: usize,
}

impl Default for EvidenceLedgerIntakeLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_EVIDENCE_LEDGER_ARTIFACT_BYTES,
            max_operations: antecedent_contract::evidence_ledger::MAX_OPERATIONS,
            max_syntheses: MAX_LEDGER_SYNTHESES,
        }
    }
}

/// Relative tolerance a consumer applies when comparing a recomputed
/// synthesis value with the stored one (transcendental functions may differ
/// in the last bits across platforms).
pub const SYNTHESIS_REPLAY_TOLERANCE: f64 = 1e-12;

const ARTIFACT_KIND: &str = crate::artifact_kinds::EVIDENCE_LEDGER;
const META_SECTION: &str = "evidence_ledger_meta";

crate::sectioned_artifact_enum! {
    /// Why an evidence ledger artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    EvidenceLedgerArtifactError {
        noun: "evidence ledger",
        identity: "evidence_ledger.identity_mismatch",
        variants {
            /// The ledger refused a stored operation on replay.
            Refused {
                /// Registered reason code.
                code: &'static str,
                /// Namespaced detail.
                detail: &'static str,
                /// Explanation.
                message: String,
            },
            /// A stored operation identity, effect, head or state digest differs from the replay.
            ReplayMismatch(&'static str),
            /// A stored synthesis differs from its ledger result or from its recomputation.
            SynthesisMismatch(&'static str),
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
            Self::ReplayMismatch(what) => {
                write!(f, "evidence_ledger.replay_mismatch: stored {what} does not replay")
            }
            Self::SynthesisMismatch(what) => {
                write!(f, "evidence_ledger.synthesis_mismatch: stored synthesis {what} does not replay")
            }
        }
    }
}

impl EvidenceLedgerArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a ledger refusal on replay, a changed identity and a
    /// stored value that does not replay; absent for corruption, unsupported
    /// versions, malformed values, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, &'static str, String)> {
        match self {
            Self::Refused { code, detail, message } => Some((*code, *detail, message.clone())),
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "evidence_ledger.identity_mismatch",
                format!("{field} changed"),
            )),
            Self::ReplayMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "evidence_ledger.replay_mismatch",
                format!("the stored {what} does not replay"),
            )),
            Self::SynthesisMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "evidence_ledger.synthesis_mismatch",
                format!("the stored synthesis {what} does not replay"),
            )),
            _ => None,
        }
    }
}

fn malformed(text: impl Into<String>) -> EvidenceLedgerArtifactError {
    EvidenceLedgerArtifactError::Malformed(text.into())
}

impl From<LedgerError> for EvidenceLedgerArtifactError {
    fn from(error: LedgerError) -> Self {
        Self::Refused { code: error.code, detail: error.detail, message: error.message }
    }
}

// -- wire types ---------------------------------------------------------------------------

/// Transport provenance on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportWire {
    /// Source populations.
    pub source_populations: Vec<String>,
    /// Target population.
    pub target_population: String,
    /// Selection diagram identity (hex).
    pub selection_diagram: String,
    /// Declared assumptions.
    pub assumptions: Vec<String>,
}

/// Lineage on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageWire {
    /// `original`, `duplicate`, `transform`, `subset`, `prior_plus_likelihood`
    /// or `derived_from_used_data`.
    pub kind: String,
    /// Parent source (hex) for duplicate, transform and subset.
    pub of: Option<String>,
    /// Transform identity (hex).
    pub transform: Option<String>,
    /// Prior source (hex).
    pub prior: Option<String>,
    /// Likelihood source (hex).
    pub likelihood: Option<String>,
    /// Input sources (hex) for derived-from-used-data.
    pub inputs: Vec<String>,
}

/// One source declaration on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceWire {
    /// Label.
    pub label: String,
    /// Kind name.
    pub kind: String,
    /// Declared observation identities (hex); absent when undeclared.
    pub observations: Option<Vec<String>>,
    /// Lineage.
    pub lineage: LineageWire,
    /// Transport provenance.
    pub transport: Option<TransportWire>,
}

/// Declared dependence of a pooling on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceWire {
    /// `independent`, `declared_covariance` or `assumption_range`.
    pub kind: String,
    /// Covariance identity (hex).
    pub covariance: Option<String>,
    /// Whether the covariance has cross terms.
    pub has_cross_covariance: bool,
    /// Lower correlation bound, as `f64` bits.
    pub rho_lo_bits: Option<u64>,
    /// Upper correlation bound, as `f64` bits.
    pub rho_hi_bits: Option<u64>,
}

/// One combination request on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombinationWire {
    /// Label.
    pub label: String,
    /// Combination kind name.
    pub kind: String,
    /// Pooled or repeatedly used sources (hex).
    pub sources: Vec<String>,
    /// Borrowed priors (hex).
    pub priors: Vec<String>,
    /// Likelihood-bearing sources (hex).
    pub likelihood: Vec<String>,
    /// Transported claims (hex).
    pub transported: Vec<String>,
    /// Native sources (hex).
    pub native: Vec<String>,
    /// Result being cited again (hex).
    pub of: Option<String>,
    /// Stage name of a repeated use.
    pub stage: Option<String>,
    /// Pooling dependence.
    pub dependence: Option<DependenceWire>,
    /// Result this one recomputes (hex).
    pub replaces: Option<String>,
}

/// One detected overlap on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapWire {
    /// Earlier source (hex).
    pub existing: String,
    /// Shared observations.
    pub shared: u64,
}

/// One result classification on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevalidationWire {
    /// Result (hex).
    pub result: String,
    /// `still_valid`, `recompute` or `withdrawn`.
    pub class: String,
}

/// The downstream effect of one operation on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectWire {
    /// Source created (hex).
    pub created_source: Option<String>,
    /// Source retired (hex).
    pub retired_source: Option<String>,
    /// Result created (hex).
    pub created_result: Option<String>,
    /// Result retired by a recomputation (hex).
    pub replaced_result: Option<String>,
    /// Overlaps detected at entry.
    pub detected_overlaps: Vec<OverlapWire>,
    /// Classification of every result.
    pub revalidation: Vec<RevalidationWire>,
}

/// One applied ledger operation on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationWire {
    /// `add`, `update`, `replace`, `withdraw`, `supersede`,
    /// `revalidate_downstream` or `combine`.
    pub kind: String,
    /// Operation identity (hex).
    pub id: String,
    /// Target source (hex) for update, replace, withdraw and supersede.
    pub target: Option<String>,
    /// Declared source for add, update, replace and supersede.
    pub source: Option<SourceWire>,
    /// Withdrawal reason.
    pub reason: Option<String>,
    /// Combination request.
    pub request: Option<CombinationWire>,
    /// Stored downstream effect.
    pub effect: EffectWire,
}

/// One summary-data synthesis on the wire. Every real is stored as its
/// `f64` bit pattern.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesisWire {
    /// Ledger result (hex) the synthesis is about.
    pub result: String,
    /// Method name.
    pub method: String,
    /// Source (hex) of each study, in the order of the summaries.
    pub sources: Vec<String>,
    /// Study estimates.
    pub estimates: Vec<u64>,
    /// Study variances (or standard errors for the envelope).
    pub variances: Vec<u64>,
    /// Row-major covariance for generalized least squares.
    pub covariance: Vec<u64>,
    /// Lower correlation bound.
    pub rho_lo: Option<u64>,
    /// Upper correlation bound.
    pub rho_hi: Option<u64>,
    /// Normal prior mean (absent for a flat prior on the mean).
    pub prior_mean: Option<u64>,
    /// Normal prior variance.
    pub prior_variance: Option<u64>,
    /// Between-study standard deviation support.
    pub tau_support: Vec<u64>,
    /// Between-study prior probabilities.
    pub tau_probabilities: Vec<u64>,
    /// Computed values, in the method's fixed order.
    pub values: Vec<u64>,
}

/// Stored metadata of an evidence ledger artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceLedgerMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Applied operations, in order.
    pub operations: Vec<OperationWire>,
    /// Syntheses bound to ledger results.
    pub syntheses: Vec<SynthesisWire>,
    /// Head operation identity (hex).
    pub head: String,
    /// Ledger state digest (hex).
    pub state_digest: String,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

// -- producer-side synthesis description -----------------------------------------------------

/// A summary-data synthesis a producer binds to a ledger result.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SynthesisMethod {
    /// Equal-study mean of independent Gaussian study estimates.
    EqualStudyGaussian {
        /// Study estimates, retaining their original ledger source order.
        estimates: Vec<f64>,
    },
    /// Inverse-variance fixed-effect pooling of independent summaries.
    FixedEffectInverseVariance {
        /// Study estimates.
        estimates: Vec<f64>,
        /// Study variances.
        variances: Vec<f64>,
    },
    /// Generalized least squares with a full covariance.
    FixedEffectGls {
        /// Study estimates.
        estimates: Vec<f64>,
        /// Row-major covariance.
        covariance: Vec<f64>,
    },
    /// Random effects, DerSimonian-Laird.
    RandomEffectsDerSimonianLaird {
        /// Study estimates.
        estimates: Vec<f64>,
        /// Study variances.
        variances: Vec<f64>,
    },
    /// Random effects, restricted maximum likelihood.
    RandomEffectsReml {
        /// Study estimates.
        estimates: Vec<f64>,
        /// Study variances.
        variances: Vec<f64>,
    },
    /// Hierarchical normal-normal borrowing with a declared tau prior.
    HierarchicalBorrowing {
        /// Study estimates.
        estimates: Vec<f64>,
        /// Study variances.
        variances: Vec<f64>,
        /// Mean prior.
        mean_prior: MeanPrior,
        /// Tau prior.
        tau_prior: TauPrior,
    },
    /// Pooled estimate over a declared correlation range, as an envelope.
    DependenceEnvelope {
        /// Study estimates.
        estimates: Vec<f64>,
        /// Study standard errors.
        standard_errors: Vec<f64>,
        /// Lower correlation bound.
        rho_lo: f64,
        /// Upper correlation bound.
        rho_hi: f64,
    },
}

/// A synthesis bound to the ledger result it is about.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisRecord {
    /// The ledger result.
    pub result: ResultId,
    /// The study sources, in the order of the summaries.
    pub sources: Vec<SourceId>,
    /// The method and its inputs.
    pub method: SynthesisMethod,
}

/// Identity of a covariance matrix as bound into a ledger pooling declaration.
#[must_use]
pub fn covariance_digest(studies: usize, covariance: &[f64]) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.covariance");
    writer.word(1).word(studies as u64).reals(covariance);
    SemanticDigest::from_bytes(writer.finish())
}

/// Whether a row-major covariance has any nonzero off-diagonal entry.
#[must_use]
pub fn covariance_has_cross_terms(studies: usize, covariance: &[f64]) -> bool {
    (0..studies).any(|i| {
        (0..studies).any(|j| i != j && covariance.get(i * studies + j).is_some_and(|v| *v != 0.0))
    })
}

// -- hex and bits helpers --------------------------------------------------------------------

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reals(values: &[u64]) -> Vec<f64> {
    values.iter().map(|v| f64::from_bits(*v)).collect()
}

fn digest_of(
    text: &str,
    what: &'static str,
) -> Result<SemanticDigest, EvidenceLedgerArtifactError> {
    parse_digest_hex(text).ok_or_else(|| malformed(format!("{what} is not 64 hex digits")))
}

fn source_id(text: &str) -> Result<SourceId, EvidenceLedgerArtifactError> {
    Ok(SourceId::from_digest(digest_of(text, "source identity")?))
}

fn result_id(text: &str) -> Result<ResultId, EvidenceLedgerArtifactError> {
    Ok(ResultId::from_digest(digest_of(text, "result identity")?))
}

fn source_ids(values: &[String]) -> Result<Vec<SourceId>, EvidenceLedgerArtifactError> {
    values.iter().map(|v| source_id(v)).collect()
}

// -- ledger -> wire ----------------------------------------------------------------------------

fn lineage_to_wire(lineage: &Lineage) -> LineageWire {
    let empty = || LineageWire {
        kind: lineage.as_str().to_owned(),
        of: None,
        transform: None,
        prior: None,
        likelihood: None,
        inputs: Vec::new(),
    };
    match lineage {
        Lineage::Original => empty(),
        Lineage::Duplicate { of } | Lineage::Subset { of } => {
            LineageWire { of: Some(of.to_hex()), ..empty() }
        }
        Lineage::Transform { of, transform } => {
            LineageWire { of: Some(of.to_hex()), transform: Some(transform.to_hex()), ..empty() }
        }
        Lineage::PriorPlusLikelihood { prior, likelihood } => LineageWire {
            prior: Some(prior.to_hex()),
            likelihood: Some(likelihood.to_hex()),
            ..empty()
        },
        Lineage::DerivedFromUsedData { inputs } => {
            LineageWire { inputs: inputs.iter().map(SourceId::to_hex).collect(), ..empty() }
        }
    }
}

fn source_to_wire(source: &SourceDeclaration) -> SourceWire {
    SourceWire {
        label: source.label.clone(),
        kind: source.kind.as_str().to_owned(),
        observations: match &source.observations {
            ObservationBasis::Declared(rows) => {
                Some(rows.iter().map(ObservationId::to_hex).collect())
            }
            ObservationBasis::Undeclared => None,
        },
        lineage: lineage_to_wire(&source.lineage),
        transport: source.transport.as_ref().map(|t| TransportWire {
            source_populations: t.source_populations.clone(),
            target_population: t.target_population.clone(),
            selection_diagram: t.selection_diagram.to_hex(),
            assumptions: t.assumptions.clone(),
        }),
    }
}

fn ids_hex(ids: &[SourceId]) -> Vec<String> {
    ids.iter().map(SourceId::to_hex).collect()
}

fn request_to_wire(request: &CombinationRequest) -> CombinationWire {
    let mut wire = CombinationWire {
        label: request.label.clone(),
        kind: request.operation.kind().as_str().to_owned(),
        sources: Vec::new(),
        priors: Vec::new(),
        likelihood: Vec::new(),
        transported: Vec::new(),
        native: Vec::new(),
        of: None,
        stage: None,
        dependence: None,
        replaces: request.replaces.map(|r| r.to_hex()),
    };
    match &request.operation {
        SynthesisOperation::StatisticalPooling { sources, dependence } => {
            wire.sources = ids_hex(sources);
            wire.dependence = Some(match *dependence {
                PoolingDependence::Independent => DependenceWire {
                    kind: "independent".to_owned(),
                    covariance: None,
                    has_cross_covariance: false,
                    rho_lo_bits: None,
                    rho_hi_bits: None,
                },
                PoolingDependence::DeclaredCovariance { covariance, has_cross_covariance } => {
                    DependenceWire {
                        kind: "declared_covariance".to_owned(),
                        covariance: Some(covariance.to_hex()),
                        has_cross_covariance,
                        rho_lo_bits: None,
                        rho_hi_bits: None,
                    }
                }
                PoolingDependence::AssumptionRange { rho_lo, rho_hi } => DependenceWire {
                    kind: "assumption_range".to_owned(),
                    covariance: None,
                    has_cross_covariance: false,
                    rho_lo_bits: Some(rho_lo.to_bits()),
                    rho_hi_bits: Some(rho_hi.to_bits()),
                },
            });
        }
        SynthesisOperation::BayesianBorrowing { priors, likelihood } => {
            wire.priors = ids_hex(priors);
            wire.likelihood = ids_hex(likelihood);
        }
        SynthesisOperation::CausalTransport { transported, native } => {
            wire.transported = ids_hex(transported);
            wire.native = ids_hex(native);
        }
        SynthesisOperation::EvidenceReuse { of } => wire.of = Some(of.to_hex()),
        SynthesisOperation::RepeatedObservationUse { sources, stage } => {
            wire.sources = ids_hex(sources);
            wire.stage = Some(stage.clone());
        }
    }
    wire
}

fn effect_to_wire(effect: &OperationEffect) -> EffectWire {
    EffectWire {
        created_source: effect.created_source.map(|s| s.to_hex()),
        retired_source: effect.retired_source.map(|s| s.to_hex()),
        created_result: effect.created_result.map(|s| s.to_hex()),
        replaced_result: effect.replaced_result.map(|s| s.to_hex()),
        detected_overlaps: effect
            .detected_overlaps
            .iter()
            .map(|o| OverlapWire { existing: o.existing.to_hex(), shared: o.shared })
            .collect(),
        revalidation: effect
            .revalidation
            .iter()
            .map(|r| RevalidationWire {
                result: r.result.to_hex(),
                class: r.class.as_str().to_owned(),
            })
            .collect(),
    }
}

fn operation_to_wire(applied: &AppliedOperation) -> OperationWire {
    let mut wire = OperationWire {
        kind: applied.operation.name().to_owned(),
        id: applied.id.to_hex(),
        target: None,
        source: None,
        reason: None,
        request: None,
        effect: effect_to_wire(&applied.effect),
    };
    match &applied.operation {
        LedgerOperation::Add { source } => wire.source = Some(source_to_wire(source)),
        LedgerOperation::Update { target, replacement }
        | LedgerOperation::Replace { target, replacement }
        | LedgerOperation::Supersede { target, replacement } => {
            wire.target = Some(target.to_hex());
            wire.source = Some(source_to_wire(replacement));
        }
        LedgerOperation::Withdraw { target, reason } => {
            wire.target = Some(target.to_hex());
            wire.reason = Some(reason.clone());
        }
        LedgerOperation::RevalidateDownstream => {}
        LedgerOperation::Combine { request } => wire.request = Some(request_to_wire(request)),
    }
    wire
}

// -- wire -> ledger ----------------------------------------------------------------------------

fn need<'a>(
    value: Option<&'a String>,
    what: &'static str,
) -> Result<&'a str, EvidenceLedgerArtifactError> {
    value.map(String::as_str).ok_or_else(|| malformed(format!("lineage {what} is missing")))
}

fn lineage_from_wire(wire: &LineageWire) -> Result<Lineage, EvidenceLedgerArtifactError> {
    Ok(match wire.kind.as_str() {
        "original" => Lineage::Original,
        "duplicate" => Lineage::Duplicate { of: source_id(need(wire.of.as_ref(), "parent")?)? },
        "subset" => Lineage::Subset { of: source_id(need(wire.of.as_ref(), "parent")?)? },
        "transform" => Lineage::Transform {
            of: source_id(need(wire.of.as_ref(), "parent")?)?,
            transform: digest_of(
                need(wire.transform.as_ref(), "transform identity")?,
                "transform identity",
            )?,
        },
        "prior_plus_likelihood" => Lineage::PriorPlusLikelihood {
            prior: source_id(need(wire.prior.as_ref(), "prior")?)?,
            likelihood: source_id(need(wire.likelihood.as_ref(), "likelihood")?)?,
        },
        "derived_from_used_data" => {
            Lineage::DerivedFromUsedData { inputs: source_ids(&wire.inputs)? }
        }
        _ => return Err(EvidenceLedgerArtifactError::UnsupportedSemantics("lineage kind")),
    })
}

fn source_from_wire(wire: &SourceWire) -> Result<SourceDeclaration, EvidenceLedgerArtifactError> {
    let kind = SourceKind::from_name(&wire.kind)
        .ok_or(EvidenceLedgerArtifactError::UnsupportedSemantics("source kind"))?;
    let observations = match &wire.observations {
        Some(rows) => ObservationBasis::Declared(
            rows.iter()
                .map(|r| digest_of(r, "observation identity").map(ObservationId::from_digest))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        None => ObservationBasis::Undeclared,
    };
    let transport = match &wire.transport {
        Some(t) => Some(TransportIdentity {
            source_populations: t.source_populations.clone(),
            target_population: t.target_population.clone(),
            selection_diagram: digest_of(&t.selection_diagram, "selection diagram identity")?,
            assumptions: t.assumptions.clone(),
        }),
        None => None,
    };
    Ok(SourceDeclaration {
        label: wire.label.clone(),
        kind,
        observations,
        lineage: lineage_from_wire(&wire.lineage)?,
        transport,
    })
}

fn request_from_wire(
    wire: &CombinationWire,
) -> Result<CombinationRequest, EvidenceLedgerArtifactError> {
    let kind = SynthesisKind::from_name(&wire.kind)
        .ok_or(EvidenceLedgerArtifactError::UnsupportedSemantics("combination kind"))?;
    let operation = match kind {
        SynthesisKind::StatisticalPooling => {
            let dependence = wire
                .dependence
                .as_ref()
                .ok_or_else(|| malformed("a pooling declares its dependence"))?;
            let dependence =
                match dependence.kind.as_str() {
                    "independent" => PoolingDependence::Independent,
                    "declared_covariance" => PoolingDependence::DeclaredCovariance {
                        covariance: digest_of(
                            dependence.covariance.as_deref().ok_or_else(|| {
                                malformed("a declared covariance carries its identity")
                            })?,
                            "covariance identity",
                        )?,
                        has_cross_covariance: dependence.has_cross_covariance,
                    },
                    "assumption_range" => {
                        PoolingDependence::AssumptionRange {
                            rho_lo: f64::from_bits(dependence.rho_lo_bits.ok_or_else(|| {
                                malformed("an assumption range carries its bounds")
                            })?),
                            rho_hi: f64::from_bits(dependence.rho_hi_bits.ok_or_else(|| {
                                malformed("an assumption range carries its bounds")
                            })?),
                        }
                    }
                    _ => {
                        return Err(EvidenceLedgerArtifactError::UnsupportedSemantics(
                            "pooling dependence kind",
                        ));
                    }
                };
            SynthesisOperation::StatisticalPooling {
                sources: source_ids(&wire.sources)?,
                dependence,
            }
        }
        SynthesisKind::BayesianBorrowing => SynthesisOperation::BayesianBorrowing {
            priors: source_ids(&wire.priors)?,
            likelihood: source_ids(&wire.likelihood)?,
        },
        SynthesisKind::CausalTransport => SynthesisOperation::CausalTransport {
            transported: source_ids(&wire.transported)?,
            native: source_ids(&wire.native)?,
        },
        SynthesisKind::EvidenceReuse => SynthesisOperation::EvidenceReuse {
            of: result_id(
                wire.of.as_deref().ok_or_else(|| malformed("a reuse names its result"))?,
            )?,
        },
        SynthesisKind::RepeatedObservationUse => SynthesisOperation::RepeatedObservationUse {
            sources: source_ids(&wire.sources)?,
            stage: wire.stage.clone().ok_or_else(|| malformed("a repeated use names its stage"))?,
        },
        _ => return Err(EvidenceLedgerArtifactError::UnsupportedSemantics("combination kind")),
    };
    Ok(CombinationRequest {
        label: wire.label.clone(),
        operation,
        replaces: wire.replaces.as_deref().map(result_id).transpose()?,
    })
}

fn operation_from_wire(
    wire: &OperationWire,
) -> Result<LedgerOperation, EvidenceLedgerArtifactError> {
    let target = || {
        wire.target
            .as_deref()
            .ok_or_else(|| malformed("operation target is missing"))
            .and_then(source_id)
    };
    let source = || {
        wire.source
            .as_ref()
            .ok_or_else(|| malformed("operation source is missing"))
            .and_then(source_from_wire)
    };
    Ok(match wire.kind.as_str() {
        "add" => LedgerOperation::Add { source: source()? },
        "update" => LedgerOperation::Update { target: target()?, replacement: source()? },
        "replace" => LedgerOperation::Replace { target: target()?, replacement: source()? },
        "supersede" => LedgerOperation::Supersede { target: target()?, replacement: source()? },
        "withdraw" => LedgerOperation::Withdraw {
            target: target()?,
            reason: wire
                .reason
                .clone()
                .ok_or_else(|| malformed("a withdrawal states its reason"))?,
        },
        "revalidate_downstream" => LedgerOperation::RevalidateDownstream,
        "combine" => LedgerOperation::Combine {
            request: request_from_wire(
                wire.request.as_ref().ok_or_else(|| malformed("combination request is missing"))?,
            )?,
        },
        _ => return Err(EvidenceLedgerArtifactError::UnsupportedSemantics("operation kind")),
    })
}

fn effect_from_wire(wire: &EffectWire) -> Result<OperationEffect, EvidenceLedgerArtifactError> {
    let opt_source = |v: &Option<String>| v.as_deref().map(source_id).transpose();
    let opt_result = |v: &Option<String>| v.as_deref().map(result_id).transpose();
    Ok(OperationEffect {
        created_source: opt_source(&wire.created_source)?,
        retired_source: opt_source(&wire.retired_source)?,
        created_result: opt_result(&wire.created_result)?,
        replaced_result: opt_result(&wire.replaced_result)?,
        detected_overlaps: wire
            .detected_overlaps
            .iter()
            .map(|o| Ok(DetectedOverlap { existing: source_id(&o.existing)?, shared: o.shared }))
            .collect::<Result<Vec<_>, EvidenceLedgerArtifactError>>()?,
        revalidation: wire
            .revalidation
            .iter()
            .map(|r| {
                Ok(ResultRevalidation {
                    result: result_id(&r.result)?,
                    class: Revalidation::from_name(&r.class).ok_or(
                        EvidenceLedgerArtifactError::UnsupportedSemantics("revalidation class"),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, EvidenceLedgerArtifactError>>()?,
    })
}

// -- syntheses ---------------------------------------------------------------------------------

/// A synthesis recomputed by a consumer.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputedSynthesis {
    /// The ledger result.
    pub result: ResultId,
    /// The method name.
    pub method: String,
    /// The computed values, in the method's fixed order.
    pub values: Vec<f64>,
}

fn compute_values(
    method: &SynthesisMethod,
) -> Result<(&'static str, Vec<f64>), EvidenceLedgerArtifactError> {
    let refuse = |error: antecedent_stats::synthesis::SynthesisError| {
        malformed(format!("synthesis input: {error}"))
    };
    Ok(match method {
        SynthesisMethod::EqualStudyGaussian { estimates } => {
            let fit = equal_study_gaussian(estimates).map_err(refuse)?;
            ("equal_study_gaussian", vec![fit.estimate, fit.variance])
        }
        SynthesisMethod::FixedEffectInverseVariance { estimates, variances } => {
            let fit = fixed_effect_inverse_variance(estimates, variances).map_err(refuse)?;
            let mut values = vec![fit.estimate, fit.variance, fit.heterogeneity_q];
            values.extend(&fit.weights);
            ("fixed_effect_inverse_variance", values)
        }
        SynthesisMethod::FixedEffectGls { estimates, covariance } => {
            let fit = fixed_effect_gls(estimates, covariance).map_err(refuse)?;
            let mut values = vec![fit.estimate, fit.variance, fit.heterogeneity_q];
            values.extend(&fit.weights);
            ("fixed_effect_gls", values)
        }
        SynthesisMethod::RandomEffectsDerSimonianLaird { estimates, variances } => {
            let fit = random_effects_dersimonian_laird(estimates, variances).map_err(refuse)?;
            let mut values =
                vec![fit.estimate, fit.variance, fit.tau_squared, fit.q, fit.i_squared];
            values.extend(&fit.weights);
            ("random_effects_dersimonian_laird", values)
        }
        SynthesisMethod::RandomEffectsReml { estimates, variances } => {
            let fit = random_effects_reml(estimates, variances).map_err(refuse)?;
            let mut values =
                vec![fit.estimate, fit.variance, fit.tau_squared, fit.q, fit.i_squared];
            values.extend(&fit.weights);
            ("random_effects_reml", values)
        }
        SynthesisMethod::HierarchicalBorrowing { estimates, variances, mean_prior, tau_prior } => {
            let fit = hierarchical_normal_borrowing(estimates, variances, *mean_prior, tau_prior)
                .map_err(refuse)?;
            let mut values = vec![fit.mean, fit.variance];
            values.extend(&fit.study_means);
            values.extend(&fit.study_variances);
            values.extend(&fit.tau_posterior);
            ("hierarchical_normal_borrowing", values)
        }
        SynthesisMethod::DependenceEnvelope { estimates, standard_errors, rho_lo, rho_hi } => {
            let env = pooled_dependence_envelope(estimates, standard_errors, *rho_lo, *rho_hi)
                .map_err(refuse)?;
            (
                "dependence_envelope",
                vec![
                    env.estimate_min,
                    env.rho_at_estimate_min,
                    env.estimate_max,
                    env.rho_at_estimate_max,
                    env.variance_min,
                    env.rho_at_variance_min,
                    env.variance_max,
                    env.rho_at_variance_max,
                ],
            )
        }
    })
}

fn synthesis_to_wire(
    record: &SynthesisRecord,
) -> Result<SynthesisWire, EvidenceLedgerArtifactError> {
    let (name, values) = compute_values(&record.method)?;
    let mut wire = SynthesisWire {
        result: record.result.to_hex(),
        method: name.to_owned(),
        sources: ids_hex(&record.sources),
        estimates: Vec::new(),
        variances: Vec::new(),
        covariance: Vec::new(),
        rho_lo: None,
        rho_hi: None,
        prior_mean: None,
        prior_variance: None,
        tau_support: Vec::new(),
        tau_probabilities: Vec::new(),
        values: bits(&values),
    };
    match &record.method {
        SynthesisMethod::EqualStudyGaussian { estimates } => {
            wire.estimates = bits(estimates);
        }
        SynthesisMethod::FixedEffectInverseVariance { estimates, variances }
        | SynthesisMethod::RandomEffectsDerSimonianLaird { estimates, variances }
        | SynthesisMethod::RandomEffectsReml { estimates, variances } => {
            wire.estimates = bits(estimates);
            wire.variances = bits(variances);
        }
        SynthesisMethod::FixedEffectGls { estimates, covariance } => {
            wire.estimates = bits(estimates);
            wire.covariance = bits(covariance);
        }
        SynthesisMethod::HierarchicalBorrowing { estimates, variances, mean_prior, tau_prior } => {
            wire.estimates = bits(estimates);
            wire.variances = bits(variances);
            if let MeanPrior::Normal { mean, variance } = mean_prior {
                wire.prior_mean = Some(mean.to_bits());
                wire.prior_variance = Some(variance.to_bits());
            }
            wire.tau_support = bits(&tau_prior.support);
            wire.tau_probabilities = bits(&tau_prior.probabilities);
        }
        SynthesisMethod::DependenceEnvelope { estimates, standard_errors, rho_lo, rho_hi } => {
            wire.estimates = bits(estimates);
            wire.variances = bits(standard_errors);
            wire.rho_lo = Some(rho_lo.to_bits());
            wire.rho_hi = Some(rho_hi.to_bits());
        }
    }
    Ok(wire)
}

fn method_from_wire(wire: &SynthesisWire) -> Result<SynthesisMethod, EvidenceLedgerArtifactError> {
    let estimates = reals(&wire.estimates);
    Ok(match wire.method.as_str() {
        "equal_study_gaussian" => SynthesisMethod::EqualStudyGaussian { estimates },
        "fixed_effect_inverse_variance" => SynthesisMethod::FixedEffectInverseVariance {
            estimates,
            variances: reals(&wire.variances),
        },
        "fixed_effect_gls" => {
            SynthesisMethod::FixedEffectGls { estimates, covariance: reals(&wire.covariance) }
        }
        "random_effects_dersimonian_laird" => SynthesisMethod::RandomEffectsDerSimonianLaird {
            estimates,
            variances: reals(&wire.variances),
        },
        "random_effects_reml" => {
            SynthesisMethod::RandomEffectsReml { estimates, variances: reals(&wire.variances) }
        }
        "hierarchical_normal_borrowing" => SynthesisMethod::HierarchicalBorrowing {
            estimates,
            variances: reals(&wire.variances),
            mean_prior: match (wire.prior_mean, wire.prior_variance) {
                (Some(mean), Some(variance)) => MeanPrior::Normal {
                    mean: f64::from_bits(mean),
                    variance: f64::from_bits(variance),
                },
                (None, None) => MeanPrior::Flat,
                _ => return Err(malformed("a normal mean prior carries mean and variance")),
            },
            tau_prior: TauPrior {
                support: reals(&wire.tau_support),
                probabilities: reals(&wire.tau_probabilities),
            },
        },
        "dependence_envelope" => SynthesisMethod::DependenceEnvelope {
            estimates,
            standard_errors: reals(&wire.variances),
            rho_lo: f64::from_bits(
                wire.rho_lo.ok_or_else(|| malformed("an envelope carries its range"))?,
            ),
            rho_hi: f64::from_bits(
                wire.rho_hi.ok_or_else(|| malformed("an envelope carries its range"))?,
            ),
        },
        _ => return Err(EvidenceLedgerArtifactError::UnsupportedSemantics("synthesis method")),
    })
}

/// Check that a synthesis is about the ledger result it names and states the
/// dependence the ledger recorded for it.
fn bind_to_result(
    ledger: &EvidenceLedger,
    result: ResultId,
    sources: &[SourceId],
    method: &SynthesisMethod,
) -> Result<(), EvidenceLedgerArtifactError> {
    let mismatch = EvidenceLedgerArtifactError::SynthesisMismatch;
    let record = ledger.result(result).ok_or(mismatch("result"))?;
    let mut declared = sources.to_vec();
    declared.sort_unstable();
    let mut in_result = ledger.result_sources(result);
    in_result.sort_unstable();
    if declared != in_result {
        return Err(mismatch("sources"));
    }
    let studies = sources.len();
    let (count, operation) = match method {
        SynthesisMethod::EqualStudyGaussian { estimates }
        | SynthesisMethod::FixedEffectInverseVariance { estimates, .. }
        | SynthesisMethod::FixedEffectGls { estimates, .. }
        | SynthesisMethod::RandomEffectsDerSimonianLaird { estimates, .. }
        | SynthesisMethod::RandomEffectsReml { estimates, .. }
        | SynthesisMethod::HierarchicalBorrowing { estimates, .. }
        | SynthesisMethod::DependenceEnvelope { estimates, .. } => {
            (estimates.len(), &record.operation)
        }
    };
    if count != studies {
        return Err(mismatch("study count"));
    }
    match (method, operation) {
        (
            SynthesisMethod::EqualStudyGaussian { .. }
            | SynthesisMethod::FixedEffectInverseVariance { .. }
            | SynthesisMethod::RandomEffectsDerSimonianLaird { .. }
            | SynthesisMethod::RandomEffectsReml { .. },
            SynthesisOperation::StatisticalPooling {
                dependence: PoolingDependence::Independent,
                ..
            },
        )
        | (
            SynthesisMethod::HierarchicalBorrowing { .. },
            SynthesisOperation::BayesianBorrowing { .. },
        ) => Ok(()),
        (
            SynthesisMethod::FixedEffectGls { covariance, .. },
            SynthesisOperation::StatisticalPooling {
                dependence:
                    PoolingDependence::DeclaredCovariance { covariance: bound, has_cross_covariance },
                ..
            },
        ) => {
            if *bound != covariance_digest(studies, covariance)
                || *has_cross_covariance != covariance_has_cross_terms(studies, covariance)
            {
                return Err(mismatch("covariance binding"));
            }
            Ok(())
        }
        (
            SynthesisMethod::DependenceEnvelope { rho_lo, rho_hi, .. },
            SynthesisOperation::StatisticalPooling {
                dependence: PoolingDependence::AssumptionRange { rho_lo: lo, rho_hi: hi },
                ..
            },
        ) => {
            if rho_lo.to_bits() != lo.to_bits() || rho_hi.to_bits() != hi.to_bits() {
                return Err(mismatch("assumption range"));
            }
            if record.claim_class != ClaimClass::AssumptionRange {
                return Err(mismatch("claim class"));
            }
            Ok(())
        }
        _ => Err(mismatch("method for the ledger operation")),
    }
}

// -- identity ----------------------------------------------------------------------------------

fn compute_identity(meta: &EvidenceLedgerMeta) -> String {
    let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.artifact");
    writer.word(u64::from(meta.version));
    writer.text(&meta.feature);
    writer.text(&meta.head);
    writer.text(&meta.state_digest);
    writer.word(meta.operations.len() as u64);
    for operation in &meta.operations {
        writer.text(&operation.kind);
        writer.text(&operation.id);
    }
    writer.word(meta.syntheses.len() as u64);
    for synthesis in &meta.syntheses {
        writer.text(&synthesis.result);
        writer.text(&synthesis.method);
        writer.texts(&synthesis.sources);
        for list in [
            &synthesis.estimates,
            &synthesis.variances,
            &synthesis.covariance,
            &synthesis.tau_support,
            &synthesis.tau_probabilities,
            &synthesis.values,
        ] {
            writer.word(list.len() as u64);
            for value in list {
                writer.word(*value);
            }
        }
        writer.opt_word(synthesis.rho_lo);
        writer.opt_word(synthesis.rho_hi);
        writer.opt_word(synthesis.prior_mean);
        writer.opt_word(synthesis.prior_variance);
    }
    writer.finish_hex()
}

// -- consumption ---------------------------------------------------------------------------------

/// A synthesis value is relative to the recomputed one; an undefined (NaN) value must be
/// undefined in both.
const SYNTHESIS_REPLAY: ReplayTolerance = ReplayTolerance::new(
    SYNTHESIS_REPLAY_TOLERANCE,
    ReplayScale::ReplayedOrOne,
    ReplayNonFinite::NanMatchesNan,
);

fn within(stored: f64, recomputed: f64) -> bool {
    SYNTHESIS_REPLAY.close(stored, recomputed)
}

fn check(
    meta: &EvidenceLedgerMeta,
    expected_identity: Option<&str>,
    limits: &EvidenceLedgerIntakeLimits,
) -> Result<EvidenceLedgerArtifact, EvidenceLedgerArtifactError> {
    if meta.version != EVIDENCE_LEDGER_ARTIFACT_VERSION {
        return Err(EvidenceLedgerArtifactError::UnsupportedVersion { version: meta.version });
    }
    crate::artifact_codec::require_semantics::<EvidenceLedgerArtifactError>(
        meta.feature == EVIDENCE_LEDGER_ARTIFACT_FEATURE,
        "feature marker",
    )?;
    if meta.operations.len() > limits.max_operations {
        return Err(EvidenceLedgerArtifactError::LimitsExceeded("operations"));
    }
    if meta.syntheses.len() > limits.max_syntheses {
        return Err(EvidenceLedgerArtifactError::LimitsExceeded("syntheses"));
    }
    // Replay: the ledger recomputes identities, policy, refusals and effects.
    let mut ledger = EvidenceLedger::new();
    for stored in &meta.operations {
        let operation = operation_from_wire(stored)?;
        let stored_effect = effect_from_wire(&stored.effect)?;
        let stored_id = OperationId::from_digest(digest_of(&stored.id, "operation identity")?);
        let applied = ledger.apply(operation)?;
        if applied.id != stored_id {
            return Err(EvidenceLedgerArtifactError::ReplayMismatch("operation identity"));
        }
        if applied.effect != stored_effect {
            return Err(EvidenceLedgerArtifactError::ReplayMismatch("operation effect"));
        }
    }
    if ledger.head().to_hex() != meta.head {
        return Err(EvidenceLedgerArtifactError::ReplayMismatch("head identity"));
    }
    if ledger.state_digest().to_hex() != meta.state_digest {
        return Err(EvidenceLedgerArtifactError::ReplayMismatch("state digest"));
    }
    // Syntheses: bound to ledger results and recomputed.
    let mut computed = Vec::with_capacity(meta.syntheses.len());
    for stored in &meta.syntheses {
        let result = result_id(&stored.result)?;
        let sources = source_ids(&stored.sources)?;
        let method = method_from_wire(stored)?;
        bind_to_result(&ledger, result, &sources, &method)?;
        let (name, values) = compute_values(&method)
            .map_err(|_| EvidenceLedgerArtifactError::SynthesisMismatch("inputs"))?;
        if name != stored.method {
            return Err(EvidenceLedgerArtifactError::SynthesisMismatch("method"));
        }
        let stored_values = reals(&stored.values);
        if stored_values.len() != values.len()
            || stored_values.iter().zip(&values).any(|(s, r)| !within(*s, *r))
        {
            return Err(EvidenceLedgerArtifactError::SynthesisMismatch("values"));
        }
        computed.push(ComputedSynthesis { result, method: name.to_owned(), values });
    }
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(EvidenceLedgerArtifactError::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(EvidenceLedgerArtifactError::IdentityMismatch { field: "retained_identity" });
    }
    Ok(EvidenceLedgerArtifact { ledger, syntheses: computed, identity })
}

fn encode_meta(
    meta: &EvidenceLedgerMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, EvidenceLedgerArtifactError> {
    crate::artifact_codec::encode_sectioned(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "evidence_ledger",
            max_bytes: MAX_EVIDENCE_LEDGER_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[],
    )
}

fn decode_meta(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<EvidenceLedgerMeta, EvidenceLedgerArtifactError> {
    crate::artifact_codec::decode_sectioned(
        ARTIFACT_KIND,
        max_bytes,
        META_SECTION,
        &[],
        EVIDENCE_LEDGER_ARTIFACT_VERSION,
        bytes,
    )
    .map(|(meta, _)| meta)
}

// -- public type ---------------------------------------------------------------------------------

/// A validated evidence ledger: the replayed ledger, the recomputed syntheses
/// and the identity that binds them.
#[derive(Clone, Debug)]
pub struct EvidenceLedgerArtifact {
    ledger: EvidenceLedger,
    syntheses: Vec<ComputedSynthesis>,
    identity: String,
}

impl EvidenceLedgerArtifact {
    /// Seal a ledger and the syntheses computed from its results. The result
    /// passes the same checks a consumer applies; this returns the sealed
    /// metadata alongside the validated artifact.
    ///
    /// # Errors
    /// An unusable synthesis input, a synthesis that does not match its
    /// ledger result, or a ledger that does not replay.
    pub fn seal(
        ledger: &EvidenceLedger,
        syntheses: &[SynthesisRecord],
    ) -> Result<(Self, EvidenceLedgerMeta), EvidenceLedgerArtifactError> {
        let mut meta = EvidenceLedgerMeta {
            version: EVIDENCE_LEDGER_ARTIFACT_VERSION,
            feature: EVIDENCE_LEDGER_ARTIFACT_FEATURE.to_owned(),
            operations: ledger.operations().iter().map(operation_to_wire).collect(),
            syntheses: syntheses.iter().map(synthesis_to_wire).collect::<Result<Vec<_>, _>>()?,
            head: ledger.head().to_hex(),
            state_digest: ledger.state_digest().to_hex(),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta);
        let artifact = check(&meta, None, &EvidenceLedgerIntakeLimits::default())?;
        Ok((artifact, meta))
    }

    /// The replayed ledger, for inspection.
    #[must_use]
    pub const fn ledger(&self) -> &EvidenceLedger {
        &self.ledger
    }

    /// The recomputed syntheses.
    #[must_use]
    pub fn syntheses(&self) -> &[ComputedSynthesis] {
        &self.syntheses
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize a sealed ledger through the checksummed container.
    ///
    /// # Errors
    /// A ledger or synthesis that does not seal, an encoding failure or an
    /// oversized payload.
    pub fn to_bytes(
        ledger: &EvidenceLedger,
        syntheses: &[SynthesisRecord],
        artifact_id: &str,
    ) -> Result<Vec<u8>, EvidenceLedgerArtifactError> {
        let (_, meta) = Self::seal(ledger, syntheses)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact by replay and recomputation.
    ///
    /// `expected_identity` is an identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// history is refused.
    ///
    /// # Errors
    /// Corruption, another major version, a bound exceeded, an operation the
    /// ledger refuses on replay, a stored identity, effect or value that does
    /// not replay, or an identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, EvidenceLedgerArtifactError> {
        Self::from_bytes_within(bytes, expected_identity, &EvidenceLedgerIntakeLimits::default())
    }

    /// [`Self::from_bytes`] under the caller's intake bounds.
    ///
    /// # Errors
    /// As [`Self::from_bytes`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn from_bytes_within(
        bytes: &[u8],
        expected_identity: Option<&str>,
        limits: &EvidenceLedgerIntakeLimits,
    ) -> Result<Self, EvidenceLedgerArtifactError> {
        check(&decode_meta(bytes, limits.max_bytes)?, expected_identity, limits)
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &EvidenceLedgerMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, EvidenceLedgerArtifactError> {
        check(meta, expected_identity, &EvidenceLedgerIntakeLimits::default())
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut EvidenceLedgerMeta) {
        meta.identity = compute_identity(meta);
    }
}

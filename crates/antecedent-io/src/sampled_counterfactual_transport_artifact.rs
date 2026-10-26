//! Independent replay of sampled counterfactual transport artifacts (kind
//! `sampled_counterfactual_transport_v1`, format version 1).
//!
//! The artifact binds the counts the estimate was computed from, the stored
//! derivation (with the model class, query and evidence catalog it was decided
//! from), the unseen-cell rule and limits, the declared overlap of the units
//! behind the tables, the uncertainty method, and the estimate with its four
//! claim labels. It carries no interval: statistical inference is withheld with
//! `cell_not_licensed`, and a standard error is present only as
//! `uncalibrated_standard_error`.
//!
//! A consumer holding only the bytes and an expected identity
//!
//! 1. refuses another version or required feature before reading the payload;
//! 2. refuses stored limits above its own before any work;
//! 3. recomputes the premises digest (graph, query, search limits, derivation,
//!    overlap declaration, unseen-cell rule, limits and the stored estimate) and
//!    the data digest (the whole catalog and every count table) and compares
//!    them with the stored ones and with the identity the caller holds
//!    independently of the bytes;
//! 4. re-decides the stored model class, query and catalog under the stored
//!    limits and accepts only the identical derivation;
//! 5. rebuilds the count tables and re-runs the estimator on them, with the
//!    stored overlap declaration and rule, and requires every stored number
//!    bit for bit together with every label and report.
//!
//! The recomputation is what protects a resealed artifact: editing counts,
//! flipping the overlap declaration, or swapping the derivation and recomputing
//! both digests still disagrees with what the stored premises and counts
//! produce. It does not protect against a producer that seals a wrong model or
//! false counts on purpose; the digests bind what was sealed, not what is true.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap, SourceId};
use antecedent_core::{ExecutionContext, IdentityDomain, RegimeId, SearchLimits, VariableId};
use antecedent_estimate::counterfactual_transport_sampled::{
    DatasetOverlap, OverlapDeclaration, SAMPLED_TRANSPORT_INTERVAL_LEVEL,
    SAMPLED_TRANSPORT_MAX_TABLE_CELLS, SAMPLED_TRANSPORT_MAX_TABLES, SampledAxis,
    SampledCounterfactualTransportEstimate, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTableKey, SampledTransportLimits, SampledTransportRequest, SamplingDesign,
    UnseenCellRule, estimate_sampled_counterfactual_transport, sampled_transport_interval,
};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_MAX_LIMITS, COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
    CounterfactualTransportDecision, CounterfactualTransportDerivation, IdentificationError,
    counterfactual_transport_refusal, replay_counterfactual_transport,
};
use antecedent_io_core::counterfactual_transport_wire::{
    CounterfactualTransportDerivationWire, CounterfactualTransportProblemWire,
    CounterfactualTransportQueryWire, CounterfactualTransportSearchLimitsWire,
};
use antecedent_io_core::transport_catalog_wire::EvidenceCatalogWire;
use serde::{Deserialize, Serialize};

use crate::{IoError, decode_with_version, digest_hex, search_bounds, to_cbor};

/// The artifact format this reader writes and accepts.
pub const SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION: u32 = 1;
/// The artifact kind and required feature marker of the accepted format.
pub const SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE: &str =
    crate::artifact_kinds::SAMPLED_COUNTERFACTUAL_TRANSPORT;

const DETAIL_INVALID_ARTIFACT: &str = "counterfactual_transport_sampled.invalid_artifact";
const DETAIL_BOUNDS: &str = "counterfactual_transport_sampled.bounds_exceeded";
const DETAIL_DERIVATION: &str = "counterfactual_transport.invalid_derivation";

/// Why a sampled counterfactual transport artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SampledCounterfactualTransportArtifactError {
    /// The bytes do not decode as this format.
    #[error("sampled counterfactual transport artifact does not decode: {0}")]
    Decode(String),
    /// The artifact could not be encoded.
    #[error("sampled counterfactual transport artifact does not encode: {0}")]
    Encode(String),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A required feature tag is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored collection or limit exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// A stored slot is not a well-formed premise.
    #[error("malformed {slot}: {message}")]
    Malformed {
        /// The artifact slot.
        slot: &'static str,
        /// What is wrong.
        message: String,
    },
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The catalog or counts do not match the data digest.
    #[error("data identity mismatch")]
    DataIdentityMismatch,
    /// The artifact is self-consistent but is not the one the caller expected.
    #[error("the artifact's {slot} identity is not the expected one")]
    ExpectedIdentityMismatch {
        /// `premises` or `data`.
        slot: &'static str,
    },
    /// The re-decision refused or differs from the stored derivation.
    #[error("derivation does not replay: {0}")]
    DerivationMismatch(String),
    /// The replayed identification or estimation refused.
    #[error("replay refused ({code}): {message}")]
    Replay {
        /// Registered reason code.
        code: &'static str,
        /// Detailed message.
        message: String,
    },
    /// A stored estimate field differs from the recomputed one.
    #[error("the stored estimate differs from the recomputed one at {field}")]
    EstimateMismatch {
        /// The first differing field.
        field: &'static str,
    },
}

type Error = SampledCounterfactualTransportArtifactError;

fn malformed(slot: &'static str, message: impl Into<String>) -> Error {
    Error::Malformed { slot, message: message.into() }
}

impl From<IoError> for Error {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

impl SampledCounterfactualTransportArtifactError {
    /// The artifact slot the failure concerns.
    #[must_use]
    pub const fn slot(&self) -> &'static str {
        match self {
            Self::Decode(_) | Self::Encode(_) => "artifact",
            Self::UnsupportedVersion { .. } => "version",
            Self::UnsupportedSemantics(slot) | Self::Malformed { slot, .. } => slot,
            Self::LimitsExceeded(_) => "limits",
            Self::PremisesMismatch => "premises_digest",
            Self::DataIdentityMismatch => "data_digest",
            Self::ExpectedIdentityMismatch { .. } => "expected_identity",
            Self::DerivationMismatch(_) => "derivation",
            Self::Replay { .. } => "replay",
            Self::EstimateMismatch { .. } => "estimate",
        }
    }

    /// The registered `(reason code, message)` the failure surfaces as.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, String) {
        let invalid = |message: String| {
            (
                antecedent_core::reason_code!("invalid_argument"),
                format!("{DETAIL_INVALID_ARTIFACT}: {}: {message}", self.slot()),
            )
        };
        match self {
            Self::LimitsExceeded(what) => (
                antecedent_core::reason_code!("route_not_supported"),
                format!("{DETAIL_BOUNDS}: limits: {what}"),
            ),
            Self::DerivationMismatch(message) => (
                antecedent_core::reason_code!("transport_not_certified"),
                format!("{DETAIL_DERIVATION}: derivation: {message}"),
            ),
            Self::Replay { code, message } => (*code, message.clone()),
            other => invalid(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire.
// ---------------------------------------------------------------------------

/// A table's identity on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledTableKeyWire {
    /// Population.
    pub population: String,
    /// Regime id.
    pub regime: u32,
    /// `(intervened variable, level bits)`.
    pub interventions: Vec<(u32, u64)>,
}

impl SampledTableKeyWire {
    fn from_key(key: &SampledTableKey) -> Self {
        Self {
            population: key.population.to_string(),
            regime: key.regime.raw(),
            interventions: key.interventions.iter().map(|(v, b)| (v.raw(), *b)).collect(),
        }
    }

    fn to_key(&self) -> SampledTableKey {
        SampledTableKey {
            population: Arc::from(self.population.as_str()),
            regime: RegimeId::from_raw(self.regime),
            interventions: self
                .interventions
                .iter()
                .map(|(v, b)| (VariableId::from_raw(*v), *b))
                .collect(),
        }
    }
}

/// One count table on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledCountTableWire {
    /// Population.
    pub population: String,
    /// Regime id.
    pub regime: u32,
    /// `(intervened variable, level bits)`, ascending by variable.
    pub interventions: Vec<(u32, u64)>,
    /// `(variable, level bits)` in table order.
    pub axes: Vec<(u32, Vec<u64>)>,
    /// Dense counts, last axis fastest.
    pub counts: Vec<u64>,
    /// Snapshot identity.
    pub snapshot_identity: String,
    /// Stratification variables, when the design is stratified.
    pub stratified_on: Option<Vec<u32>>,
    /// Declared structural zero cells, ascending.
    pub structural_zeros: Vec<u64>,
}

impl SampledCountTableWire {
    /// Encode a validated table.
    #[must_use]
    pub fn from_table(table: &SampledRegimeCounts) -> Self {
        Self {
            population: table.population().to_owned(),
            regime: table.regime().raw(),
            interventions: table
                .interventions()
                .iter()
                .map(|(v, l)| (v.raw(), (l + 0.0).to_bits()))
                .collect(),
            axes: table
                .axes()
                .iter()
                .map(|a| (a.variable.raw(), a.levels.iter().map(|l| (l + 0.0).to_bits()).collect()))
                .collect(),
            counts: table.counts().to_vec(),
            snapshot_identity: table.snapshot_identity().to_owned(),
            stratified_on: match table.design() {
                SamplingDesign::IndependentUnits => None,
                SamplingDesign::Stratified { variables } => {
                    Some(variables.iter().map(|v| v.raw()).collect())
                }
            },
            structural_zeros: table.structural_zeros().iter().map(|c| *c as u64).collect(),
        }
    }

    /// Decode and validate; a canonical table re-encodes to itself.
    ///
    /// # Errors
    /// `Malformed` for a table that does not validate or is not canonical.
    pub fn to_table(&self) -> Result<SampledRegimeCounts, Error> {
        let zeros = self
            .structural_zeros
            .iter()
            .map(|c| usize::try_from(*c))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| malformed("tables", "a structural zero index does not fit"))?;
        let table = SampledRegimeCounts::try_new(SampledRegimeCountsParts {
            population: Arc::from(self.population.as_str()),
            regime: RegimeId::from_raw(self.regime),
            interventions: self
                .interventions
                .iter()
                .map(|(v, b)| (VariableId::from_raw(*v), f64::from_bits(*b)))
                .collect(),
            axes: self
                .axes
                .iter()
                .map(|(v, levels)| SampledAxis {
                    variable: VariableId::from_raw(*v),
                    levels: levels.iter().map(|b| f64::from_bits(*b)).collect(),
                })
                .collect(),
            counts: self.counts.clone(),
            snapshot_identity: Arc::from(self.snapshot_identity.as_str()),
            design: match &self.stratified_on {
                None => SamplingDesign::IndependentUnits,
                Some(variables) => SamplingDesign::Stratified {
                    variables: variables.iter().copied().map(VariableId::from_raw).collect(),
                },
            },
            structural_zeros: zeros,
        })
        .map_err(|error| malformed("tables", error.to_string()))?;
        if Self::from_table(&table) != *self {
            return Err(malformed("tables", "a stored count table is not canonical"));
        }
        Ok(table)
    }
}

/// The overlap declaration on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledOverlapWire {
    /// Relation of undeclared pairs: `independent`, `shared_units` or `unknown`.
    pub default: String,
    /// Declared pairs, canonical order.
    pub pairs: Vec<(SampledTableKeyWire, SampledTableKeyWire, String)>,
}

impl SampledOverlapWire {
    fn from_declaration(declaration: &OverlapDeclaration) -> Self {
        Self {
            default: declaration.default_relation().name().to_owned(),
            pairs: declaration
                .pairs()
                .iter()
                .map(|p| {
                    (
                        SampledTableKeyWire::from_key(&p.a),
                        SampledTableKeyWire::from_key(&p.b),
                        p.relation.name().to_owned(),
                    )
                })
                .collect(),
        }
    }

    fn to_declaration(&self) -> Result<OverlapDeclaration, Error> {
        let parse = |name: &str| {
            DatasetOverlap::parse(name)
                .ok_or_else(|| malformed("overlap", format!("unknown relation `{name}`")))
        };
        let mut declaration = OverlapDeclaration::with_default(parse(&self.default)?);
        for (a, b, relation) in &self.pairs {
            declaration = declaration
                .with_pair(a.to_key(), b.to_key(), parse(relation)?)
                .map_err(|error| malformed("overlap", error.to_string()))?;
        }
        if Self::from_declaration(&declaration) != *self {
            return Err(malformed("overlap", "the stored overlap declaration is not canonical"));
        }
        Ok(declaration)
    }
}

/// The unseen-cell rule on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledUnseenRuleWire {
    /// `refuse`, `additive_smoothing` or `unseen_mass_envelope`.
    pub name: String,
    /// The rule's parameter as IEEE bits.
    pub parameter: Option<u64>,
}

impl SampledUnseenRuleWire {
    fn from_rule(rule: UnseenCellRule) -> Self {
        Self { name: rule.name().to_owned(), parameter: rule.parameter().map(f64::to_bits) }
    }

    fn to_rule(&self) -> Result<UnseenCellRule, Error> {
        let parameter = self.parameter.map(f64::from_bits);
        let rule = match (self.name.as_str(), parameter) {
            ("refuse", None) => UnseenCellRule::Refuse,
            ("additive_smoothing", Some(pseudo_count)) => {
                UnseenCellRule::AdditiveSmoothing { pseudo_count }
            }
            ("unseen_mass_envelope", Some(unseen_mass)) => {
                UnseenCellRule::UnseenMassEnvelope { unseen_mass }
            }
            _ => return Err(malformed("unseen_rule", "an unknown rule or a misplaced parameter")),
        };
        Ok(rule)
    }
}

/// Sampled limits on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledLimitsWire {
    /// Fewest observations per table.
    pub min_table_total: u64,
    /// Largest unseen share, IEEE bits.
    pub max_unseen_fraction: u64,
    /// Envelope family limit.
    pub max_envelope_evaluations: u32,
    /// Evaluation operation limit.
    pub operations: usize,
    /// Evaluation depth limit.
    pub depth: usize,
}

impl SampledLimitsWire {
    fn from_limits(limits: &SampledTransportLimits) -> Self {
        Self {
            min_table_total: limits.min_table_total,
            max_unseen_fraction: limits.max_unseen_fraction.to_bits(),
            max_envelope_evaluations: limits.max_envelope_evaluations,
            operations: limits.evaluation.operations,
            depth: limits.evaluation.depth,
        }
    }

    fn to_limits(&self) -> SampledTransportLimits {
        SampledTransportLimits {
            min_table_total: self.min_table_total,
            max_unseen_fraction: f64::from_bits(self.max_unseen_fraction),
            max_envelope_evaluations: self.max_envelope_evaluations,
            evaluation: ExactEvaluationLimits { operations: self.operations, depth: self.depth },
        }
    }
}

/// A stored plug-in value, as IEEE bits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledPointWire {
    /// Probability.
    pub probability: u64,
    /// Numerator.
    pub numerator: u64,
    /// Denominator.
    pub denominator: u64,
    /// Original conditioning mass.
    pub conditioning_mass: Option<u64>,
    /// Whether the conditioning mass was checked.
    pub conditioning_mass_checked: bool,
    /// Whether the tables were smoothed.
    pub smoothed: bool,
}

/// A stored envelope, as IEEE bits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledEnvelopeWire {
    /// Lower end.
    pub lower: u64,
    /// Upper end.
    pub upper: u64,
    /// Declared unseen mass.
    pub unseen_mass: u64,
    /// Members of the family.
    pub members: u64,
    /// Defined members.
    pub defined_members: u64,
    /// Undefined members.
    pub undefined_members: u64,
    /// Tables with unseen cells.
    pub tables_with_unseen_cells: u64,
    /// The envelope label.
    pub label: String,
}

/// A stored per-table report.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledTableReportWire {
    /// The table.
    pub key: SampledTableKeyWire,
    /// Content identity of the counts.
    pub content_identity: String,
    /// Observations.
    pub total: u64,
    /// Cells.
    pub cells: u64,
    /// Cells with observations.
    pub observed_cells: u64,
    /// Unseen non-structural cells.
    pub unseen_cells: u64,
    /// Declared structural zeros.
    pub structural_zeros: u64,
    /// Whether the table's law was smoothed.
    pub smoothed: bool,
    /// Pseudo-count, as bits, when smoothed.
    pub pseudo_count: Option<u64>,
    /// Dataset index.
    pub dataset: u64,
}

/// A stored standard error.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledStandardErrorWire {
    /// The value as bits.
    pub value: u64,
    /// Its label (`uncalibrated_standard_error`).
    pub label: String,
    /// Its method.
    pub method: String,
    /// Relative gradient step as bits.
    pub gradient_relative_step: u64,
    /// Gradient evaluations.
    pub gradient_evaluations: u64,
    /// Independent datasets.
    pub datasets: u64,
}

/// The whole estimate on the wire: what the consumer recomputes and compares.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledEstimateWire {
    /// Structural identification label.
    pub structural_identification: String,
    /// Exact finite-law evaluation label.
    pub exact_finite_law_evaluation: String,
    /// Point estimate label.
    pub point_estimate: String,
    /// Statistical inference label.
    pub statistical_inference: String,
    /// Digest of the derivation's canonical text.
    pub derivation_identity: String,
    /// Digest of counts, overlap, rule and limits.
    pub evidence_identity: String,
    /// The plug-in value.
    pub point: Option<SampledPointWire>,
    /// Why no point is reported.
    pub point_withheld_reason: Option<String>,
    /// The structural envelope.
    pub envelope: Option<SampledEnvelopeWire>,
    /// Per-table reports.
    pub tables: Vec<SampledTableReportWire>,
    /// Dataset groups.
    pub datasets: Vec<Vec<SampledTableKeyWire>>,
    /// Factors averaged over unread coordinates.
    pub averaged_factors: u64,
    /// Negative complements clamped.
    pub complements_clamped: u64,
    /// Whether the probability was clamped.
    pub probability_clamped: bool,
    /// Regime-law cells read.
    pub cells_read: u64,
    /// Uncertainty method name.
    pub uncertainty_method: String,
    /// The standard error, when computed.
    pub standard_error: Option<SampledStandardErrorWire>,
    /// Why no standard error is reported.
    pub standard_error_withheld_reason: Option<String>,
    /// Interval route reason code (`cell_not_licensed`).
    pub interval_code: String,
    /// Interval route detail.
    pub interval_detail: String,
    /// Calibration reason.
    pub interval_reason: String,
}

impl SampledEstimateWire {
    /// Encode an estimate.
    #[must_use]
    pub fn from_estimate(estimate: &SampledCounterfactualTransportEstimate) -> Self {
        let bits = f64::to_bits;
        Self {
            structural_identification: estimate.labels.structural_identification.to_owned(),
            exact_finite_law_evaluation: estimate.labels.exact_finite_law_evaluation.to_owned(),
            point_estimate: estimate.labels.point_estimate.to_owned(),
            statistical_inference: estimate.labels.statistical_inference.to_owned(),
            derivation_identity: estimate.derivation_identity.clone(),
            evidence_identity: estimate.evidence_identity.clone(),
            point: estimate.point.as_ref().map(|p| SampledPointWire {
                probability: bits(p.probability),
                numerator: bits(p.numerator),
                denominator: bits(p.denominator),
                conditioning_mass: p.conditioning_mass.map(bits),
                conditioning_mass_checked: p.conditioning_mass_checked,
                smoothed: p.smoothed,
            }),
            point_withheld_reason: estimate.point_withheld_reason.map(str::to_owned),
            envelope: estimate.envelope.as_ref().map(|e| SampledEnvelopeWire {
                lower: bits(e.lower),
                upper: bits(e.upper),
                unseen_mass: bits(e.unseen_mass),
                members: e.members,
                defined_members: e.defined_members,
                undefined_members: e.undefined_members,
                tables_with_unseen_cells: e.tables_with_unseen_cells as u64,
                label: e.label.to_owned(),
            }),
            tables: estimate
                .tables
                .iter()
                .map(|t| SampledTableReportWire {
                    key: SampledTableKeyWire::from_key(&t.key),
                    content_identity: t.content_identity.clone(),
                    total: t.total,
                    cells: t.cells as u64,
                    observed_cells: t.observed_cells as u64,
                    unseen_cells: t.unseen_cells as u64,
                    structural_zeros: t.structural_zeros as u64,
                    smoothed: t.smoothed,
                    pseudo_count: t.pseudo_count.map(bits),
                    dataset: t.dataset as u64,
                })
                .collect(),
            datasets: estimate
                .datasets
                .iter()
                .map(|group| group.iter().map(SampledTableKeyWire::from_key).collect())
                .collect(),
            averaged_factors: estimate.repairs.averaged_factors,
            complements_clamped: estimate.repairs.complements_clamped,
            probability_clamped: estimate.repairs.probability_clamped,
            cells_read: estimate.cells_read,
            uncertainty_method: estimate.inference.method.to_owned(),
            standard_error: estimate.inference.standard_error.as_ref().map(|s| {
                SampledStandardErrorWire {
                    value: bits(s.value),
                    label: s.label.to_owned(),
                    method: s.method.to_owned(),
                    gradient_relative_step: bits(s.gradient_relative_step),
                    gradient_evaluations: s.gradient_evaluations,
                    datasets: s.datasets as u64,
                }
            }),
            standard_error_withheld_reason: estimate
                .inference
                .standard_error_withheld_reason
                .map(str::to_owned),
            interval_code: estimate.inference.interval.code.to_owned(),
            interval_detail: estimate.inference.interval.detail.to_owned(),
            interval_reason: estimate.inference.interval.reason.to_owned(),
        }
    }

    /// The first field that differs from `other`.
    fn first_difference(&self, other: &Self) -> Option<&'static str> {
        macro_rules! check {
            ($($field:ident),* $(,)?) => {
                $(if self.$field != other.$field { return Some(stringify!($field)); })*
            };
        }
        check!(
            structural_identification,
            exact_finite_law_evaluation,
            point_estimate,
            statistical_inference,
            derivation_identity,
            evidence_identity,
            point,
            point_withheld_reason,
            envelope,
            tables,
            datasets,
            averaged_factors,
            complements_clamped,
            probability_clamped,
            cells_read,
            uncertainty_method,
            standard_error,
            standard_error_withheld_reason,
            interval_code,
            interval_detail,
            interval_reason,
        );
        None
    }
}

/// The identity a consumer holds independently of the bytes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SampledCounterfactualTransportExpectedIdentity {
    /// Expected premises digest, when held.
    pub premises_digest: Option<String>,
    /// Expected data digest, when held.
    pub data_digest: Option<String>,
}

/// The versioned sampled artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledCounterfactualTransportArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The model class.
    pub problem: CounterfactualTransportProblemWire,
    /// The query with its selection diagram.
    pub query: CounterfactualTransportQueryWire,
    /// Variable names in ascending id order, or empty.
    pub variable_names: Vec<String>,
    /// Search limits the derivation was decided under.
    pub search: CounterfactualTransportSearchLimitsWire,
    /// The complete evidence catalog.
    pub catalog: EvidenceCatalogWire,
    /// The stored derivation.
    pub derivation: CounterfactualTransportDerivationWire,
    /// Count tables in canonical order.
    pub tables: Vec<SampledCountTableWire>,
    /// The declared overlap.
    pub overlap: SampledOverlapWire,
    /// The unseen-cell rule.
    pub unseen_rule: SampledUnseenRuleWire,
    /// The sampled limits.
    pub limits: SampledLimitsWire,
    /// The estimate and its claim labels.
    pub estimate: SampledEstimateWire,
    /// Digest of the premises and the stored estimate.
    pub premises_digest: String,
    /// Digest of the catalog and counts.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    problem: &'a CounterfactualTransportProblemWire,
    query: &'a CounterfactualTransportQueryWire,
    variable_names: &'a [String],
    search: &'a CounterfactualTransportSearchLimitsWire,
    derivation: &'a CounterfactualTransportDerivationWire,
    overlap: &'a SampledOverlapWire,
    unseen_rule: &'a SampledUnseenRuleWire,
    limits: &'a SampledLimitsWire,
    estimate: &'a SampledEstimateWire,
}

fn sorted_cbor<T: Serialize>(items: &[T]) -> Result<Vec<Vec<u8>>, IoError> {
    let mut out = items.iter().map(to_cbor).collect::<Result<Vec<_>, _>>()?;
    out.sort();
    Ok(out)
}

impl SampledCounterfactualTransportArtifactWire {
    /// The premises digest the stored premises should carry.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        digest_hex(
            IdentityDomain::TransportCertificate,
            &PremisesView {
                tag: SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE,
                problem: &self.problem,
                query: &self.query,
                variable_names: &self.variable_names,
                search: &self.search,
                derivation: &self.derivation,
                overlap: &self.overlap,
                unseen_rule: &self.unseen_rule,
                limits: &self.limits,
                estimate: &self.estimate,
            },
        )
    }

    /// The data digest the stored catalog and counts should carry.
    ///
    /// # Errors
    /// The catalog or counts do not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let catalog = (
            sorted_cbor(&self.catalog.environments)?,
            sorted_cbor(&self.catalog.regimes)?,
            sorted_cbor(&self.catalog.bindings)?,
            &self.catalog.target_sampling,
        );
        digest_hex(
            IdentityDomain::TransportCertificate,
            &("sampled_counterfactual_transport_data_v1", catalog, sorted_cbor(&self.tables)?),
        )
    }

    /// Check the shape the format guarantees.
    ///
    /// # Errors
    /// `UnsupportedSemantics` for another feature tag, `Malformed` otherwise.
    pub fn validate_shape(&self) -> Result<(), Error> {
        if self.required_features != [SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE] {
            return Err(Error::UnsupportedSemantics("required_features"));
        }
        if self.derivation.contract != antecedent_identify::COUNTERFACTUAL_TRANSPORT_CONTRACT {
            return Err(Error::UnsupportedSemantics("contract"));
        }
        if self.tables.is_empty() {
            return Err(malformed("tables", "a sampled artifact carries count tables"));
        }
        if self.estimate.statistical_inference
            != antecedent_estimate::counterfactual_transport_sampled::LABEL_INFERENCE_WITHHELD
            || self.estimate.interval_code != "cell_not_licensed"
        {
            return Err(malformed(
                "estimate",
                "a sampled artifact carries no interval: statistical inference is withheld",
            ));
        }
        Ok(())
    }

    /// Seal: check the shape, then fill both digests.
    ///
    /// # Errors
    /// A shape failure or premises that do not encode.
    pub fn sealed(mut self) -> Result<Self, Error> {
        self.validate_shape()?;
        self.data_digest = self.expected_data_digest()?;
        self.premises_digest = self.expected_premises_digest()?;
        Ok(self)
    }

    /// The identity a consumer can hold: both digests.
    #[must_use]
    pub fn identity(&self) -> SampledCounterfactualTransportExpectedIdentity {
        SampledCounterfactualTransportExpectedIdentity {
            premises_digest: Some(self.premises_digest.clone()),
            data_digest: Some(self.data_digest.clone()),
        }
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A shape or encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, Error> {
        self.validate_shape()?;
        to_cbor(self).map_err(|e| Error::Encode(e.to_string()))
    }

    /// Decode, refusing another version before the payload is interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or feature, or a shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let wire: Self = decode_with_version(bytes, |version: u32| {
            if version == SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(Error::UnsupportedVersion { version })
            }
        })?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Recompute both digests and compare with the stored ones and the identity
    /// the caller holds.
    ///
    /// # Errors
    /// `PremisesMismatch`, `DataIdentityMismatch` or `ExpectedIdentityMismatch`.
    pub fn check_digests(
        &self,
        expected: &SampledCounterfactualTransportExpectedIdentity,
    ) -> Result<(), Error> {
        let premises = self.expected_premises_digest()?;
        if premises != self.premises_digest {
            return Err(Error::PremisesMismatch);
        }
        let data = self.expected_data_digest()?;
        if data != self.data_digest {
            return Err(Error::DataIdentityMismatch);
        }
        if expected.premises_digest.as_deref().is_some_and(|e| e != premises) {
            return Err(Error::ExpectedIdentityMismatch { slot: "premises" });
        }
        if expected.data_digest.as_deref().is_some_and(|e| e != data) {
            return Err(Error::ExpectedIdentityMismatch { slot: "data" });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Producer.
// ---------------------------------------------------------------------------

fn estimation_refusal(error: &antecedent_estimate::EstimationError) -> Error {
    use antecedent_estimate::EstimationError as E;
    match error {
        E::Refused { code, message } | E::RefusedWithFields { code, message, .. } => {
            Error::Replay { code, message: message.to_string() }
        }
        other => Error::Malformed { slot: "tables", message: other.to_string() },
    }
}

/// Estimate and seal: runs the sampled estimator on the stored derivation and
/// binds counts, derivation, estimate, uncertainty method and overlap.
///
/// # Errors
/// The estimator's typed refusal (as [`Error::Replay`]) or a shape failure.
pub fn sampled_counterfactual_transport_artifact(
    derivation: &CounterfactualTransportDerivation,
    request: &SampledTransportRequest,
    variable_names: &[String],
    ctx: &ExecutionContext,
) -> Result<
    (SampledCounterfactualTransportArtifactWire, SampledCounterfactualTransportEstimate),
    Error,
> {
    let estimate = estimate_sampled_counterfactual_transport(derivation, request, ctx)
        .map_err(|error| estimation_refusal(&error))?;
    let mut tables: Vec<&SampledRegimeCounts> = request.tables.iter().collect();
    tables.sort_by_key(|t| (t.key(), t.content_identity()));
    let wire = SampledCounterfactualTransportArtifactWire {
        version: SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION,
        required_features: vec![SAMPLED_COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE.into()],
        problem: CounterfactualTransportProblemWire::from_problem(&derivation.problem),
        query: CounterfactualTransportQueryWire::from_query(&derivation.query),
        variable_names: variable_names.to_vec(),
        search: CounterfactualTransportSearchLimitsWire {
            operations: derivation.search.operations_limit,
            depth: derivation.search.depth_limit,
            memory_limit_bytes: derivation.search.memory_limit_bytes,
        },
        catalog: EvidenceCatalogWire::from_catalog(&derivation.catalog),
        derivation: CounterfactualTransportDerivationWire::from_derivation(derivation),
        tables: tables.into_iter().map(SampledCountTableWire::from_table).collect(),
        overlap: SampledOverlapWire::from_declaration(&request.overlap),
        unseen_rule: SampledUnseenRuleWire::from_rule(request.unseen_rule),
        limits: SampledLimitsWire::from_limits(&request.limits),
        estimate: SampledEstimateWire::from_estimate(&estimate),
        premises_digest: String::new(),
        data_digest: String::new(),
    }
    .sealed()?;
    Ok((wire, estimate))
}

// ---------------------------------------------------------------------------
// Consumer.
// ---------------------------------------------------------------------------

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct SampledCounterfactualTransportConsumeLimits {
    /// Largest search limits a stored decision may have run under.
    pub search: SearchLimits,
    /// Largest search memory cap (bytes) a stored decision may have run under.
    pub search_memory_bytes: u64,
    /// Largest exact-evaluation limits the replayed estimator may run under.
    pub evaluation: ExactEvaluationLimits,
    /// Most count tables an artifact may carry.
    pub max_tables: usize,
    /// Most cells one stored table may carry.
    pub max_table_cells: usize,
}

impl Default for SampledCounterfactualTransportConsumeLimits {
    fn default() -> Self {
        Self {
            search: COUNTERFACTUAL_TRANSPORT_MAX_LIMITS,
            search_memory_bytes: COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
            evaluation: ExactEvaluationLimits::default(),
            max_tables: SAMPLED_TRANSPORT_MAX_TABLES,
            max_table_cells: SAMPLED_TRANSPORT_MAX_TABLE_CELLS,
        }
    }
}

/// Everything a replay reconstructed.
#[derive(Debug)]
pub struct ConsumedSampledCounterfactualTransport {
    /// The re-decided derivation.
    pub derivation: CounterfactualTransportDerivation,
    /// The rebuilt request (counts, overlap, rule, limits).
    pub request: SampledTransportRequest,
    /// The recomputed estimate.
    pub estimate: SampledCounterfactualTransportEstimate,
    /// The decoded artifact.
    pub wire: SampledCounterfactualTransportArtifactWire,
}

fn replay_refusal(error: &IdentificationError) -> Error {
    match counterfactual_transport_refusal(error) {
        Some((code, detail)) => Error::Replay { code, message: format!("{detail}: {error}") },
        None => Error::DerivationMismatch(error.to_string()),
    }
}

fn check_limits(
    wire: &SampledCounterfactualTransportArtifactWire,
    limits: &SampledCounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<(), Error> {
    if let Some(what) = search_bounds(wire.search.operations, wire.search.depth, limits.search) {
        return Err(Error::LimitsExceeded(what));
    }
    let memory_cap = ctx
        .memory_limit_bytes()
        .map_or(limits.search_memory_bytes, |hard| hard.min(limits.search_memory_bytes));
    if wire.search.memory_limit_bytes > memory_cap {
        return Err(Error::LimitsExceeded("search memory limit"));
    }
    if wire.limits.operations > limits.evaluation.operations
        || wire.limits.depth > limits.evaluation.depth
    {
        return Err(Error::LimitsExceeded("evaluation limits"));
    }
    if wire.tables.len() > limits.max_tables {
        return Err(Error::LimitsExceeded("count tables"));
    }
    if wire.tables.iter().any(|t| t.counts.len() > limits.max_table_cells) {
        return Err(Error::LimitsExceeded("table cells"));
    }
    Ok(())
}

/// Decode and recheck everything under the consumer's limits, re-decide the
/// derivation, rebuild the counts and recompute the estimate bit for bit.
///
/// # Errors
/// Any [`SampledCounterfactualTransportArtifactError`]; its
/// [`slot`](SampledCounterfactualTransportArtifactError::slot) names the failed
/// check and [`refusal`](SampledCounterfactualTransportArtifactError::refusal)
/// its registered reason code.
pub fn consume_sampled_counterfactual_transport_with_limits(
    bytes: &[u8],
    expected: &SampledCounterfactualTransportExpectedIdentity,
    limits: SampledCounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ConsumedSampledCounterfactualTransport, Error> {
    let wire = SampledCounterfactualTransportArtifactWire::decode(bytes)?;
    check_limits(&wire, &limits, ctx)?;
    wire.check_digests(expected)?;
    let problem = wire.problem.to_problem().map_err(|e| malformed("problem", e.to_string()))?;
    let query = wire.query.to_query().map_err(|e| malformed("query", e.to_string()))?;
    let catalog = wire.catalog.to_catalog().map_err(|e| malformed("catalog", e.to_string()))?;
    let decision =
        replay_counterfactual_transport(&problem, &query, &catalog, &wire.derivation.search, ctx)
            .map_err(|error| replay_refusal(&error))?;
    let derivation = match decision {
        CounterfactualTransportDecision::Identified(derivation) => *derivation,
        CounterfactualTransportDecision::Exhausted(receipt) => {
            return Err(Error::Replay {
                code: antecedent_core::reason_code!("transport_budget_cancel"),
                message: format!(
                    "counterfactual_transport.budget: the replayed decision stopped under the \
                     stored limits: {}",
                    receipt.summary()
                ),
            });
        }
        other => {
            return Err(Error::DerivationMismatch(format!(
                "the stored query no longer decides as identified under the stored limits: {}",
                other.detail_code().unwrap_or("not_identified")
            )));
        }
    };
    if CounterfactualTransportDerivationWire::from_derivation(&derivation) != wire.derivation {
        return Err(Error::DerivationMismatch(
            "the re-decided derivation, search receipt or cited regimes differ from the stored \
             ones"
                .into(),
        ));
    }
    let tables =
        wire.tables.iter().map(SampledCountTableWire::to_table).collect::<Result<Vec<_>, _>>()?;
    let request = SampledTransportRequest {
        tables,
        overlap: wire.overlap.to_declaration()?,
        unseen_rule: wire.unseen_rule.to_rule()?,
        limits: wire.limits.to_limits(),
    };
    let estimate = estimate_sampled_counterfactual_transport(&derivation, &request, ctx)
        .map_err(|error| estimation_refusal(&error))?;
    let recomputed = SampledEstimateWire::from_estimate(&estimate);
    if let Some(field) = wire.estimate.first_difference(&recomputed) {
        return Err(Error::EstimateMismatch { field });
    }
    Ok(ConsumedSampledCounterfactualTransport { derivation, request, estimate, wire })
}

/// [`consume_sampled_counterfactual_transport_with_limits`] as the io error a
/// facade returns.
///
/// # Errors
/// [`IoError::Refused`] carrying the failure's registered code and message.
pub fn consume_sampled_counterfactual_transport(
    bytes: &[u8],
    expected: &SampledCounterfactualTransportExpectedIdentity,
    limits: SampledCounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ConsumedSampledCounterfactualTransport, IoError> {
    consume_sampled_counterfactual_transport_with_limits(bytes, expected, limits, ctx).map_err(
        |error| {
            let (code, message) = error.refusal();
            IoError::Refused { code, message }
        },
    )
}

// ---------------------------------------------------------------------------
// Measured interval.
// ---------------------------------------------------------------------------

fn interval_not_measured(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("measured_interval.protocol_not_measured: {message}"),
    }
}

impl SampledCounterfactualTransportArtifactWire {
    /// Replay the original sampled artifact (re-deciding the derivation and
    /// recomputing the estimate from its counts), build the delta Wald interval
    /// and bind it to the currently attesting coverage record of exactly this
    /// construction. The original artifact is unchanged and never carries an
    /// interval.
    ///
    /// # Errors
    /// Any replay refusal of the original artifact, an estimate outside the
    /// measured design (level, smoothing, unseen cells, conditioning mass,
    /// sampling design, shared datasets, unequal table totals), or absent,
    /// stale or boundary evidence for the stored query and sample size.
    pub fn measure(
        bytes: &[u8],
        expected: &SampledCounterfactualTransportExpectedIdentity,
        limits: SampledCounterfactualTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<crate::measured_inference::MeasuredInference, IoError> {
        let consumed = consume_sampled_counterfactual_transport(bytes, expected, limits, ctx)?;
        if consumed
            .request
            .tables
            .iter()
            .any(|table| !matches!(table.design(), SamplingDesign::IndependentUnits))
        {
            return Err(interval_not_measured(
                "only independently drawn units were measured; stratified designs are not",
            ));
        }
        if consumed.estimate.datasets.len() != consumed.request.tables.len() {
            return Err(interval_not_measured(
                "tables sharing one dataset were not part of the measured construction",
            ));
        }
        let interval = sampled_transport_interval(
            &consumed.derivation,
            &consumed.estimate,
            SAMPLED_TRANSPORT_INTERVAL_LEVEL,
        )
        .map_err(interval_not_measured)?;
        let scope = serde_json::json!({
            "validation_design": "finite_scm_discrete_counterfactual_transport",
            "evidence_interpretation": "finite validation grid for the checked query and sample sizes; other queries, smoothed tables, unseen cells and sparse conditioning are not licensed",
            "declared_model_assumptions": ["independent units within each table", "tables of distinct independent datasets", "stored derivation identifies the query"],
            "sampling_assumptions_authenticated_from_rows": false,
            "units_per_table": interval.basis.row_count,
            "tables": consumed.request.tables.len(),
            "unseen_rule": consumed.request.unseen_rule.name(),
            "method": interval.method,
            "level": interval.level,
            "derivation_identity": consumed.estimate.derivation_identity,
            "reported_scalar": "target_probability",
        });
        crate::measured_inference::authorize(
            "sampled_counterfactual_transport",
            bytes.to_vec(),
            &consumed.wire.premises_digest,
            &consumed.wire.data_digest,
            scope,
            vec![crate::measured_inference::ScalarExecution {
                name: "target_probability".into(),
                point: interval.point,
                interval: (interval.lower, interval.upper),
                basis: interval.basis.clone(),
            }],
        )
    }

    /// Consume a measured envelope by replaying its original sampled artifact
    /// and re-resolving the interval against the current coverage records.
    ///
    /// # Errors
    /// Any source, expected identity, protocol, evidence or scalar mismatch.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &crate::measured_inference::MeasuredExpectation,
        limits: SampledCounterfactualTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<crate::measured_inference::MeasuredInference, IoError> {
        let wire = crate::measured_inference::decode(bytes, expected)?;
        let measured = Self::measure(
            &wire.source,
            &SampledCounterfactualTransportExpectedIdentity {
                premises_digest: Some(expected.premises_digest.clone()),
                data_digest: Some(expected.data_digest.clone()),
            },
            limits,
            ctx,
        )?;
        crate::measured_inference::compare_replay(&wire, &measured)?;
        Ok(measured)
    }
}

// ---------------------------------------------------------------------------
// Overlap from the evidence ledger.
// ---------------------------------------------------------------------------

/// Derive the overlap declaration of count tables from an evidence ledger.
///
/// Each table is bound to the ledger source that holds its observations. A pair
/// of bound sources whose observations the ledger knows to be disjoint is
/// `independent`; sources that share observations are `shared_units`; sources
/// whose observations are not computable are `unknown`, and so is every pair
/// that involves a table not bound to a source: independence is never inferred.
///
/// # Errors
/// `invalid_argument` when a bound source is not in the ledger or a table is
/// bound twice.
pub fn overlap_declaration_from_ledger(
    ledger: &EvidenceLedger,
    bindings: &[(SampledTableKey, SourceId)],
) -> Result<OverlapDeclaration, IoError> {
    let refuse = |message: String| IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("{DETAIL_INVALID_ARTIFACT}: overlap: {message}"),
    };
    let mut declaration = OverlapDeclaration::undeclared();
    for (i, (a, source_a)) in bindings.iter().enumerate() {
        for (b, source_b) in &bindings[i + 1..] {
            if a == b {
                return Err(refuse("a table is bound to two sources".into()));
            }
            let relation = match ledger
                .overlap(*source_a, *source_b)
                .map_err(|error| refuse(error.to_string()))?
            {
                Overlap::Independent => DatasetOverlap::Independent,
                Overlap::SharedObservations { .. } => DatasetOverlap::SharedUnits,
                _ => DatasetOverlap::Unknown,
            };
            declaration = declaration
                .with_pair(a.clone(), b.clone(), relation)
                .map_err(|error| refuse(error.to_string()))?;
        }
    }
    Ok(declaration)
}

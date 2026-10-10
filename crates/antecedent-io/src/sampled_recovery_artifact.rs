//! Independent artifact of sampled observation recovery (2.3A A6, record
//! `2.3A.X10.sampled_observation_recovery`), format version 2. Internal and non-routed:
//! the public interval route `antecedent.transport.sampled_observation_recovery` stays
//! closed (`cell_not_licensed`, `sampled_recovery.route_frozen`) until the whole-method
//! calibration is measured, so the interval stored here carries calibration `unmeasured`.
//!
//! Version 1 of the recovery artifact ([`crate::recovery_artifact`], 2.2 X10) is the
//! exact-law point artifact and is unchanged: this reader refuses its bytes by version,
//! and the version-1 reader refuses these.
//!
//! The artifact carries the derivation identity (the m-graph, roles, effect query, checked
//! derivation record and both expression arenas, with the evidence catalog binding the
//! observed snapshot), the observation rows (id and pattern, in order: the whole-row
//! bootstrap depends on the order), the pattern-count summary, the request configuration
//! with its seed, the recovered law, the recovered effect point, the `BCa` interval
//! (`calibration = "unmeasured"`), the diagnostics, the bootstrap covariance of the
//! recovered cells and the replicate receipt (every replicate's selection digest and
//! effect, failed ones included, and the receipt digest).
//!
//! A consumer trusts none of the result. Under its own bounds it re-verifies the stored
//! derivation against the m-graph and catalog, then reruns the whole composed method
//! through [`antecedent_estimate::replay_sampled_recovery`], which refuses unless the
//! rerun reproduces the stored receipt digest bit for bit, and accepts only a stored
//! result identical to the rerun: law, point, interval, diagnostics, pattern counts,
//! covariance and replicates. A changed pattern, snapshot, derivation, seed or interval
//! receipt refuses. Two digests are separate: the premises digest (derivation, request
//! configuration) and the data-identity digest (the catalog's snapshot bindings and every
//! row). A consumer holding an expected identity passes a
//! [`SampledRecoveryExpectation`].
//!
//! What replay does not protect against: a producer that supplies fabricated rows
//! consistently, and the m-graph's untestable premises, which are declared and bound into
//! the premises digest, never verified from data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    query_wire::ValueWire,
    recovery_artifact::{RecoveredEffectWire, RecoveryQueryWire, RecoveryTableWire},
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{EvidenceCatalog, ExecutionContext, IdentityDomain, VariableId};
use antecedent_estimate::recovery_sampled::{
    SampledBcaReceipt, SampledIntervalMethod, SampledJackknifeRecord, SampledReplicateRecord,
};
use antecedent_estimate::{
    ObservationPattern, ObservationRow, SAMPLED_RECOVERY_CALIBRATION,
    SAMPLED_RECOVERY_MAX_REPLICATES, SAMPLED_RECOVERY_MAX_ROWS, SampledEffectInterval,
    SampledObservationInput, SampledRecoveryConfig, SampledRecoveryDetail, SampledRecoveryError,
    SampledRecoveryReceipt, SampledRecoveryResult, replay_sampled_recovery,
};
use antecedent_graph::{Admg, NodeRef};
use antecedent_identify::{
    RECOVERY_DEFAULT_LIMITS, RecoveredEffectQuery, RecoveryDerivation, RecoveryDerivationRecord,
    RecoveryError, RecoveryLimits, verify_observation_recovery,
};
use serde::{Deserialize, Serialize};

/// Supported `BCa` artifact version.
pub const SAMPLED_RECOVERY_BCA_ARTIFACT_VERSION: u32 = 3;
/// `BCa` feature, including exact midrank ties and full delete-one-row jackknife.
pub const SAMPLED_RECOVERY_BCA_ARTIFACT_FEATURE: &str = "sampled_observation_recovery_bca_v3";

/// Why a sampled-recovery artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SampledRecoveryArtifactError {
    /// The feature marker, a tag or a claim is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored count or size exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The re-decided derivation does not reproduce the stored one.
    #[error("derivation does not verify: {0}")]
    ProofMismatch(RecoveryError),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot and rows.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The stored identity is not the one the consumer expects.
    #[error("identity differs from the consumer's expectation: {0}")]
    ExpectationMismatch(&'static str),
    /// The caller's variable names are not the verified mapping.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
    /// The rerun result differs from the stored result.
    #[error("sampled recovery result does not replay")]
    ResultMismatch,
}

impl From<SampledRecoveryArtifactError> for IoError {
    fn from(error: SampledRecoveryArtifactError) -> Self {
        match &error {
            SampledRecoveryArtifactError::ProofMismatch(inner) => {
                crate::recovery_artifact::recovery_io_error(inner)
            }
            SampledRecoveryArtifactError::ResultMismatch => Self::Refused {
                code: SampledRecoveryDetail::ReceiptMismatch.reason_code(),
                message: format!(
                    "{}: sampled recovery artifact: {error}",
                    SampledRecoveryDetail::ReceiptMismatch.detail()
                ),
            },
            _ => Self::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                message: format!("sampled recovery artifact: {error}"),
            },
        }
    }
}

fn sampled_refusal(error: &SampledRecoveryError) -> IoError {
    IoError::Refused { code: error.reason_code(), message: error.to_string() }
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct SampledRecoveryConsumeLimits {
    /// Maxima of the stored decision limits.
    pub decision: RecoveryLimits,
    /// Most observation rows (the rerun costs `replicates * rows`).
    pub max_rows: usize,
    /// Most bootstrap replicates.
    pub max_replicates: usize,
}

impl Default for SampledRecoveryConsumeLimits {
    fn default() -> Self {
        Self {
            decision: RECOVERY_DEFAULT_LIMITS,
            max_rows: 100_000_usize.min(SAMPLED_RECOVERY_MAX_ROWS),
            max_replicates: SAMPLED_RECOVERY_MAX_REPLICATES,
        }
    }
}

/// An identity a consumer expects the artifact to carry; `None` skips that check.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SampledRecoveryExpectation {
    /// Expected premises digest (derivation and request configuration).
    pub premises_digest: Option<String>,
    /// Expected data-identity digest (snapshot bindings and rows).
    pub data_digest: Option<String>,
}

/// One stored observation row: `(id, responses, proxies, fully)`.
pub type SampledRowWire = (u64, u8, u8, u8);

/// The stored request configuration, seed included.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledConfigWire {
    /// Exactly `bootstrap_bca` in the supported v3 format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_method: Option<String>,
    /// Bootstrap replicates.
    pub replicates: usize,
    /// Bootstrap seed.
    pub seed: u64,
    /// Largest fraction of replicates that may fail jointly.
    pub max_failed_fraction: f64,
    /// Absolute tolerance on the unit mass of the laws.
    pub normalization_tolerance: f64,
    /// Expected-count threshold of a small cell.
    pub small_cell_count: f64,
    /// Treatment level of the contrast.
    pub treated_level: f64,
    /// Reference level of the contrast.
    pub control_level: f64,
}

impl SampledConfigWire {
    fn from_config(config: &SampledRecoveryConfig) -> Self {
        Self {
            interval_method: Some("bootstrap_bca".into()),
            replicates: config.replicates,
            seed: config.seed,
            max_failed_fraction: config.max_failed_fraction,
            normalization_tolerance: config.normalization_tolerance,
            small_cell_count: config.small_cell_count,
            treated_level: config.treated_level,
            control_level: config.control_level,
        }
    }

    fn to_config(&self) -> SampledRecoveryConfig {
        SampledRecoveryConfig {
            interval_method: SampledIntervalMethod::Bca,
            replicates: self.replicates,
            seed: self.seed,
            max_failed_fraction: self.max_failed_fraction,
            normalization_tolerance: self.normalization_tolerance,
            small_cell_count: self.small_cell_count,
            treated_level: self.treated_level,
            control_level: self.control_level,
        }
    }
}

/// The stored `BCa` interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledIntervalWire {
    /// Nominal level.
    pub level: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Always `unmeasured`.
    pub calibration: String,
}

impl SampledIntervalWire {
    fn from_interval(interval: &SampledEffectInterval) -> Self {
        Self {
            level: interval.level,
            lower: interval.lower,
            upper: interval.upper,
            calibration: interval.calibration.into(),
        }
    }

    fn to_interval(&self) -> Result<SampledEffectInterval, SampledRecoveryArtifactError> {
        if self.calibration != SAMPLED_RECOVERY_CALIBRATION {
            return Err(SampledRecoveryArtifactError::UnsupportedSemantics(
                "interval calibration is unmeasured",
            ));
        }
        Ok(SampledEffectInterval {
            level: self.level,
            lower: self.lower,
            upper: self.upper,
            calibration: SAMPLED_RECOVERY_CALIBRATION,
        })
    }
}

/// One stored bootstrap replicate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledReplicateWire {
    /// Replicate id.
    pub id: u32,
    /// Digest of the drawn row ids.
    pub selection_digest: String,
    /// Recovered effect, absent when the replicate failed.
    pub effect: Option<f64>,
    /// Why it failed.
    pub failure: Option<String>,
    /// Zero complete-case patterns of a failed replicate.
    pub zero_patterns: Vec<(u8, u8, u8)>,
}

/// Count-compressed delete-one-row record. Multiplicity represents every row.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledJackknifeWire {
    /// Observed pattern bits.
    pub pattern: (u8, u8, u8),
    /// Number of original rows of this pattern.
    pub multiplicity: u64,
    /// Delete-one effect.
    pub effect: f64,
}

/// The exact `BCa` arithmetic; no values are trusted without independent replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledBcaWire {
    /// Frozen algorithm identity, including ties, quantiles and jackknife scope.
    pub convention: String,
    /// Inverse-normal midrank bias correction.
    pub bias_correction: f64,
    /// Multiplicity-weighted jackknife acceleration.
    pub acceleration: f64,
    /// Type-7 quantile probabilities after `BCa` transformation.
    pub adjusted_probabilities: [f64; 2],
    /// All distinct positive-count patterns.
    pub jackknife: Vec<SampledJackknifeWire>,
}

impl SampledBcaWire {
    fn from_receipt(bca: &SampledBcaReceipt) -> Self {
        Self {
            convention: "midrank_exact_ties:type7:delete_one_row".into(),
            bias_correction: bca.bias_correction,
            acceleration: bca.acceleration,
            adjusted_probabilities: bca.adjusted_probabilities,
            jackknife: bca
                .jackknife
                .iter()
                .map(|r| SampledJackknifeWire {
                    pattern: (r.pattern.responses, r.pattern.proxies, r.pattern.fully),
                    multiplicity: r.multiplicity,
                    effect: r.effect,
                })
                .collect(),
        }
    }
    fn to_receipt(&self) -> SampledBcaReceipt {
        SampledBcaReceipt {
            bias_correction: self.bias_correction,
            acceleration: self.acceleration,
            adjusted_probabilities: self.adjusted_probabilities,
            jackknife: self
                .jackknife
                .iter()
                .map(|r| SampledJackknifeRecord {
                    pattern: ObservationPattern {
                        responses: r.pattern.0,
                        proxies: r.pattern.1,
                        fully: r.pattern.2,
                    },
                    multiplicity: r.multiplicity,
                    effect: r.effect,
                })
                .collect(),
        }
    }
}

/// The stored replicate receipt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledReceiptWire {
    /// Required for the supported `BCa` receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bca: Option<SampledBcaWire>,
    /// Snapshot identity of the rows.
    pub snapshot_id: String,
    /// Number of rows.
    pub rows: usize,
    /// Identity of the recovery derivation.
    pub derivation_identity: String,
    /// Digest of the snapshot, ids and patterns of every row.
    pub input_digest: String,
    /// Request configuration.
    pub config: SampledConfigWire,
    /// Recovered effect point.
    pub point_effect: f64,
    /// `BCa` interval.
    pub interval: SampledIntervalWire,
    /// Every replicate in id order.
    pub replicates: Vec<SampledReplicateWire>,
    /// Digest of everything above.
    pub receipt_digest: String,
}

impl SampledReceiptWire {
    fn from_receipt(receipt: &SampledRecoveryReceipt) -> Self {
        Self {
            bca: receipt.bca.as_ref().map(SampledBcaWire::from_receipt),
            snapshot_id: receipt.snapshot_id.clone(),
            rows: receipt.rows,
            derivation_identity: receipt.derivation_identity.clone(),
            input_digest: receipt.input_digest.clone(),
            config: SampledConfigWire::from_config(&receipt.config),
            point_effect: receipt.point_effect,
            interval: SampledIntervalWire::from_interval(&receipt.interval),
            replicates: receipt
                .replicates
                .iter()
                .map(|r| SampledReplicateWire {
                    id: r.id,
                    selection_digest: r.selection_digest.clone(),
                    effect: r.effect,
                    failure: r.failure.clone(),
                    zero_patterns: r
                        .zero_patterns
                        .iter()
                        .map(|p| (p.responses, p.proxies, p.fully))
                        .collect(),
                })
                .collect(),
            receipt_digest: receipt.receipt_digest.clone(),
        }
    }

    fn to_receipt(&self) -> Result<SampledRecoveryReceipt, SampledRecoveryArtifactError> {
        Ok(SampledRecoveryReceipt {
            bca: self.bca.as_ref().map(SampledBcaWire::to_receipt),
            snapshot_id: self.snapshot_id.clone(),
            rows: self.rows,
            derivation_identity: self.derivation_identity.clone(),
            input_digest: self.input_digest.clone(),
            config: self.config.to_config(),
            point_effect: self.point_effect,
            interval: self.interval.to_interval()?,
            replicates: self
                .replicates
                .iter()
                .map(|r| SampledReplicateRecord {
                    id: r.id,
                    selection_digest: r.selection_digest.clone(),
                    effect: r.effect,
                    failure: r.failure.clone(),
                    zero_patterns: r
                        .zero_patterns
                        .iter()
                        .map(|&(responses, proxies, fully)| ObservationPattern {
                            responses,
                            proxies,
                            fully,
                        })
                        .collect(),
                })
                .collect(),
            receipt_digest: self.receipt_digest.clone(),
        })
    }
}

/// The stored diagnostics (pattern-count summary included).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledDiagnosticsWire {
    /// Number of rows.
    pub rows: usize,
    /// `(responses, proxies, fully, count)` of every nonempty pattern, ascending.
    pub pattern_counts: Vec<(u8, u8, u8, u64)>,
    /// Smallest count over the complete-case patterns.
    pub min_complete_case_count: u64,
    /// Smallest recovered cell probability.
    pub min_recovered_cell: f64,
    /// Smallest recovered cell expected count.
    pub min_recovered_cell_count: f64,
    /// Cells below the small-cell count.
    pub small_recovered_cells: Vec<usize>,
    /// Total mass of the recovered law.
    pub recovered_total: f64,
    /// `|recovered_total - 1|`.
    pub normalization_defect: f64,
    /// The tolerance the defect was bounded by.
    pub normalization_tolerance: f64,
}

/// The stored result of one sampled recovery.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampledResultWire {
    /// The whole recovered law.
    pub recovered: RecoveryTableWire,
    /// Canonical identity of the recovered law's catalog descriptor.
    pub recovered_identity: String,
    /// Recovered effect point.
    pub effect: f64,
    /// Bootstrap standard error of the effect.
    pub effect_standard_error: f64,
    /// Replicates that failed and were dropped jointly.
    pub failed_replicates: usize,
    /// `BCa` interval.
    pub interval: SampledIntervalWire,
    /// Diagnostics.
    pub diagnostics: SampledDiagnosticsWire,
    /// Number of recovered cells (the side of the covariance matrix).
    pub recovered_cells: usize,
    /// Bootstrap covariance of the recovered cells, row-major.
    pub recovered_cell_covariance: Vec<f64>,
    /// Effects of the replicates that succeeded, in replicate order.
    pub replicate_effects: Vec<f64>,
}

impl SampledResultWire {
    fn from_result(result: &SampledRecoveryResult) -> Self {
        let law = result.recovered_law.law();
        let d = &result.diagnostics;
        Self {
            recovered: RecoveryTableWire {
                axes: law
                    .axes()
                    .iter()
                    .map(|a| {
                        (a.variable.raw(), a.values.iter().map(ValueWire::from_value).collect())
                    })
                    .collect(),
                probabilities: law.probabilities().to_vec(),
            },
            recovered_identity: result.recovered_law.descriptor().canonical_identity(),
            effect: result.effect,
            effect_standard_error: result.effect_standard_error,
            failed_replicates: result.failed_replicates,
            interval: SampledIntervalWire::from_interval(&result.interval),
            diagnostics: SampledDiagnosticsWire {
                rows: d.rows,
                pattern_counts: d
                    .pattern_counts
                    .iter()
                    .map(|(p, n)| (p.responses, p.proxies, p.fully, *n))
                    .collect(),
                min_complete_case_count: d.min_complete_case_count,
                min_recovered_cell: d.min_recovered_cell,
                min_recovered_cell_count: d.min_recovered_cell_count,
                small_recovered_cells: d.small_recovered_cells.clone(),
                recovered_total: d.recovered_total,
                normalization_defect: d.normalization_defect,
                normalization_tolerance: d.normalization_tolerance,
            },
            recovered_cells: result.recovered_cells,
            recovered_cell_covariance: result.recovered_cell_covariance.clone(),
            replicate_effects: result.replicate_effects.clone(),
        }
    }
}

/// Versioned sampled-recovery result with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledRecoveryArtifactWire {
    /// Format version ([`SAMPLED_RECOVERY_BCA_ARTIFACT_VERSION`]).
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The m-graph (static nodes `0..n`).
    pub graph: AdmgWire,
    /// Canonical roles.
    pub query: RecoveryQueryWire,
    /// Downstream effect query.
    pub effect: RecoveredEffectWire,
    /// Checked derivation record, including the search receipt and stored limits.
    pub derivation: RecoveryDerivationRecord,
    /// Recovery formula arena.
    pub expression: ExprArenaWire,
    /// Downstream effect arena.
    pub effect_expression: ExprArenaWire,
    /// Evidence catalog binding the observed snapshot.
    pub catalog: EvidenceCatalogWire,
    /// Variable name of every m-graph node, or empty.
    pub variable_names: Vec<String>,
    /// Snapshot identity of the rows.
    pub snapshot_id: String,
    /// The observation rows in order.
    pub rows: Vec<SampledRowWire>,
    /// The stored result.
    pub result: SampledResultWire,
    /// The replicate receipt (carries the request configuration and seed).
    pub receipt: SampledReceiptWire,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Digest of the scientific premises.
    pub premises_digest: String,
    /// Digest of the snapshot bindings and rows.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    query: &'a RecoveryQueryWire,
    effect: &'a RecoveredEffectWire,
    derivation: &'a RecoveryDerivationRecord,
    expression: &'a ExprArenaWire,
    effect_expression: &'a ExprArenaWire,
    variable_names: &'a [String],
    config: &'a SampledConfigWire,
}

#[derive(Serialize)]
struct DataView<'a> {
    tag: &'static str,
    bindings: Vec<(u32, String, Option<String>)>,
    catalog: &'a EvidenceCatalogWire,
    snapshot_id: &'a str,
    rows: &'a [SampledRowWire],
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// The checked premises and result a sampled-recovery artifact is built from.
pub struct SampledRecoveryArtifactInput<'a> {
    /// The m-graph the derivation was decided on (static nodes `0..n`).
    pub graph: &'a Admg,
    /// The downstream effect query.
    pub effect: &'a RecoveredEffectQuery,
    /// The checked derivation.
    pub derivation: &'a RecoveryDerivation,
    /// The catalog binding the observed snapshot.
    pub catalog: &'a EvidenceCatalog,
    /// The observation rows.
    pub input: &'a SampledObservationInput,
    /// The sampled recovery result.
    pub result: &'a SampledRecoveryResult,
    /// Variable names of every m-graph node, or empty.
    pub variable_names: &'a [String],
}

/// Everything a replay reconstructed and recomputed.
#[derive(Debug)]
pub struct ConsumedSampledRecovery {
    /// The re-verified derivation.
    pub derivation: RecoveryDerivation,
    /// The rerun result.
    pub result: SampledRecoveryResult,
    /// The decoded artifact.
    pub wire: SampledRecoveryArtifactWire,
}

impl SampledRecoveryArtifactWire {
    /// Build an artifact from checked premises and a result.
    ///
    /// # Errors
    /// Premises that do not encode, an m-graph whose nodes are not `0..n`, an effect
    /// without an expression arena, or a size above the default consumer bounds.
    pub fn checked(input: &SampledRecoveryArtifactInput<'_>) -> Result<Self, IoError> {
        for (i, node) in input.graph.nodes().iter().enumerate() {
            let expected = NodeRef::Static(VariableId::from_raw(
                u32::try_from(i).map_err(|_| IoError::TooLarge)?,
            ));
            if *node != expected {
                return Err(IoError::Convert("the m-graph must have static nodes 0..n".into()));
            }
        }
        let derivation = input.derivation;
        let Some(effect_checked) = derivation.effect() else {
            return Err(IoError::Convert("the derivation identified no downstream effect".into()));
        };
        let mut wire = Self {
            version: SAMPLED_RECOVERY_BCA_ARTIFACT_VERSION,
            required_features: vec![SAMPLED_RECOVERY_BCA_ARTIFACT_FEATURE.into()],
            graph: admg_to_wire(input.graph)?,
            query: RecoveryQueryWire::from_query(derivation.query()),
            effect: RecoveredEffectWire::from_query(input.effect)?,
            derivation: derivation.record().clone(),
            expression: expr_arena_to_wire(derivation.arena())?,
            effect_expression: expr_arena_to_wire(effect_checked.arena())?,
            catalog: EvidenceCatalogWire::from_catalog(input.catalog),
            variable_names: input.variable_names.to_vec(),
            snapshot_id: input.input.snapshot_id.clone(),
            rows: input
                .input
                .rows
                .iter()
                .map(|r| (r.id, r.pattern.responses, r.pattern.proxies, r.pattern.fully))
                .collect(),
            result: SampledResultWire::from_result(input.result),
            receipt: SampledReceiptWire::from_receipt(&input.result.receipt),
            calibration: SAMPLED_RECOVERY_CALIBRATION.into(),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.validate_shape()?;
        wire.check_producer_bounds()?;
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.data_digest = wire.expected_data_digest()?;
        Ok(wire)
    }

    fn check_producer_bounds(&self) -> Result<(), SampledRecoveryArtifactError> {
        let defaults = SampledRecoveryConsumeLimits::default();
        if self.rows.len() > defaults.max_rows
            || self.receipt.config.replicates > defaults.max_replicates
        {
            return Err(SampledRecoveryArtifactError::LimitsExceeded("export bounds"));
        }
        Ok(())
    }

    /// The premises digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-verifies and reruns everything.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        let view = PremisesView {
            tag: "sampled_observation_recovery_bca_premises_v3",
            graph,
            query: &self.query,
            effect: &self.effect,
            derivation: &self.derivation,
            expression: &self.expression,
            effect_expression: &self.effect_expression,
            variable_names: &self.variable_names,
            config: &self.receipt.config,
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &view)?.to_hex())
    }

    /// The data-identity digest the stored snapshot bindings and rows should carry.
    ///
    /// # Errors
    /// The catalog does not decode or the identity does not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let catalog = self.catalog.to_catalog()?;
        let mut bindings: Vec<(u32, String, Option<String>)> = catalog
            .bindings
            .iter()
            .map(|b| {
                (
                    b.regime.raw(),
                    b.snapshot_identity.to_string(),
                    b.dataset_identity.as_deref().map(str::to_owned),
                )
            })
            .collect();
        bindings.sort();
        let view = DataView {
            tag: "sampled_observation_recovery_data_v2",
            bindings,
            catalog: &self.catalog,
            snapshot_id: &self.snapshot_id,
            rows: &self.rows,
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &view)?.to_hex())
    }

    fn validate_shape(&self) -> Result<(), SampledRecoveryArtifactError> {
        let unsupported = SampledRecoveryArtifactError::UnsupportedSemantics;
        match self.version {
            SAMPLED_RECOVERY_BCA_ARTIFACT_VERSION
                if self.required_features == [SAMPLED_RECOVERY_BCA_ARTIFACT_FEATURE]
                    && self.receipt.config.interval_method.as_deref() == Some("bootstrap_bca")
                    && self.receipt.config.replicates == 2000
                    && self.receipt.config.max_failed_fraction == 0.0
                    && self.receipt.bca.as_ref().is_some_and(|b| {
                        b.convention == "midrank_exact_ties:type7:delete_one_row"
                            && b.jackknife.len() <= 729
                    }) => {}
            _ => {
                return Err(unsupported(
                    "version, interval method, BCa scope or required features",
                ));
            }
        }
        if self.calibration != SAMPLED_RECOVERY_CALIBRATION
            || self.result.interval.calibration != SAMPLED_RECOVERY_CALIBRATION
            || self.receipt.interval.calibration != SAMPLED_RECOVERY_CALIBRATION
        {
            return Err(unsupported("interval calibration is unmeasured"));
        }
        let nodes = self.graph.node_count;
        if !self.variable_names.is_empty()
            && (u32::try_from(self.variable_names.len()).ok() != Some(nodes)
                || self.variable_names.iter().any(|name| name.trim().is_empty())
                || self.variable_names.iter().collect::<std::collections::BTreeSet<_>>().len()
                    != self.variable_names.len())
        {
            return Err(unsupported("variable names"));
        }
        Ok(())
    }

    /// Check a caller's variable names against the verified name mapping.
    ///
    /// # Errors
    /// [`SampledRecoveryArtifactError::NamesMismatch`] unless `names` equals the mapping.
    pub fn check_variable_names(
        &self,
        names: &[String],
    ) -> Result<(), SampledRecoveryArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(SampledRecoveryArtifactError::NamesMismatch)
        }
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version (the 2.2 point artifact is version 1) before
    /// the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != SAMPLED_RECOVERY_BCA_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &SampledRecoveryConsumeLimits,
    ) -> Result<(), SampledRecoveryArtifactError> {
        let exceeded = SampledRecoveryArtifactError::LimitsExceeded;
        let receipt = &self.derivation.receipt;
        if receipt.operations_limit > limits.decision.search.operations {
            return Err(exceeded("search operation limit"));
        }
        if receipt.depth_limit > limits.decision.search.depth {
            return Err(exceeded("search depth limit"));
        }
        if receipt.memory_bytes > limits.decision.memory_bytes {
            return Err(exceeded("search memory cap"));
        }
        if self.rows.len() > limits.max_rows || self.rows.len() > SAMPLED_RECOVERY_MAX_ROWS {
            return Err(exceeded("row count"));
        }
        if self.receipt.config.replicates > limits.max_replicates
            || self.receipt.config.replicates > SAMPLED_RECOVERY_MAX_REPLICATES
            || self.receipt.replicates.len() > limits.max_replicates
        {
            return Err(exceeded("replicate count"));
        }
        // At most 2^6 recovered cells, so at most 64 * 64 covariance entries.
        if self.result.recovered_cell_covariance.len() > 64 * 64 {
            return Err(exceeded("covariance size"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then rerun the whole
    /// composed method and accept only an identical result. No external provider is
    /// accessed.
    ///
    /// # Errors
    /// Any limit, digest, derivation, replay or result mismatch, with the route's own
    /// reason code and detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: SampledRecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedSampledRecovery, IoError> {
        Self::consume_expecting(bytes, &SampledRecoveryExpectation::default(), limits, ctx)
    }

    /// [`Self::consume_with_limits`] that additionally requires the stored premises and
    /// data digests to equal the consumer's expected identity.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`], and an expectation mismatch.
    pub fn consume_expecting(
        bytes: &[u8],
        expected: &SampledRecoveryExpectation,
        limits: SampledRecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedSampledRecovery, IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(SampledRecoveryArtifactError::PremisesMismatch.into());
        }
        if wire.expected_data_digest()? != wire.data_digest {
            return Err(SampledRecoveryArtifactError::DataIdentityMismatch.into());
        }
        if expected.premises_digest.as_ref().is_some_and(|d| *d != wire.premises_digest) {
            return Err(SampledRecoveryArtifactError::ExpectationMismatch("premises").into());
        }
        if expected.data_digest.as_ref().is_some_and(|d| *d != wire.data_digest) {
            return Err(SampledRecoveryArtifactError::ExpectationMismatch("data identity").into());
        }
        let catalog = wire.catalog.to_catalog()?;
        let graph = admg_from_wire(&wire.graph)?;
        let query = wire.query.to_query();
        let effect = wire.effect.to_query()?;
        let arena = expr_arena_from_wire(&wire.expression)?;
        let effect_arena = expr_arena_from_wire(&wire.effect_expression)?;
        let derivation = verify_observation_recovery(
            &graph,
            &query,
            &catalog,
            Some(&effect),
            &wire.derivation,
            &arena,
            Some(&effect_arena),
            limits.decision,
            ctx,
        )
        .map_err(SampledRecoveryArtifactError::ProofMismatch)?;
        let observations = SampledObservationInput {
            snapshot_id: wire.snapshot_id.clone(),
            rows: wire
                .rows
                .iter()
                .map(|&(id, responses, proxies, fully)| ObservationRow {
                    id,
                    pattern: ObservationPattern { responses, proxies, fully },
                })
                .collect(),
        };
        let receipt = wire.receipt.to_receipt()?;
        let result = replay_sampled_recovery(&derivation, &observations, &receipt, ctx)
            .map_err(|error| sampled_refusal(&error))?;
        let replayed = SampledResultWire::from_result(&result);
        let stored_receipt = SampledReceiptWire::from_receipt(&result.receipt);
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.result)?
            || crate::to_cbor(&stored_receipt)? != crate::to_cbor(&wire.receipt)?
        {
            return Err(SampledRecoveryArtifactError::ResultMismatch.into());
        }
        Ok(ConsumedSampledRecovery { derivation, result, wire })
    }
}

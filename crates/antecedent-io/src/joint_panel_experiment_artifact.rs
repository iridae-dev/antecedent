//! Historical frozen-panel experimental conditioning through shared numerical owners.
// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::bridge_artifact::FullBridgeContractWire;
use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionMetadata, DistributionTrust,
};
use crate::evidence_ledger_artifact::{EvidenceLedgerArtifact, EvidenceLedgerMeta};
use crate::joint_panel_artifact::{
    MAX_JOINT_PANEL_ARTIFACT_BYTES, PanelArtifactError, PanelExecutionProgram,
};
use antecedent_contract::bridge::BridgeContract;
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap, SourceId, SourceStatus};
use antecedent_core::execution::AdmissionLease;
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Exact ledger origins/revisions used by a compiled experimental update.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelExperimentOrigins {
    /// Original fitted model source identity.
    pub model: String,
    /// New experimental likelihood identity.
    pub experiment: String,
    /// Original uncertain mapping source identity.
    pub mapping: String,
    /// Data-derived prior origins, when declared.
    pub priors: Vec<String>,
}
impl PanelExperimentOrigins {
    fn ordered(&self) -> Vec<String> {
        let mut ids = vec![self.model.clone(), self.experiment.clone(), self.mapping.clone()];
        ids.extend(self.priors.iter().cloned());
        ids
    }
}
/// Versioned full frozen program, declared experiment and historical ledger authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelExperimentExecutionProgram {
    version: u16,
    panel: PanelExecutionProgram,
    bridge: FullBridgeContractWire,
    ledger: EvidenceLedgerMeta,
    origins: PanelExperimentOrigins,
    projection: Vec<Vec<u64>>,
    prior_metadata: DistributionMetadata,
    observed: u64,
    variance: u64,
    sample_size: u64,
    intercepts: Vec<u64>,
    slopes: Vec<u64>,
    mapping_dependency: [u8; 32],
    policy: String,
}
/// Every field of the program that its identity covers: all but the ledger history, with the
/// frozen panel program stood for by its own identity.
#[derive(Serialize)]
struct PanelExperimentIdentityView<'a> {
    version: u16,
    panel: [u8; 32],
    bridge: &'a FullBridgeContractWire,
    origins: &'a PanelExperimentOrigins,
    projection: &'a [Vec<u64>],
    prior_metadata: &'a DistributionMetadata,
    observed: u64,
    variance: u64,
    sample_size: u64,
    intercepts: &'a [u64],
    slopes: &'a [u64],
    mapping_dependency: [u8; 32],
    policy: &'a str,
}
/// Exact native inputs, borrowed from an already checked original and experimental result.
pub struct PanelExperimentExecutionSource<'a> {
    /// Actual full compiled/frozen original panel.
    pub panel: PanelExecutionProgram,
    /// Actual fully licensed estimand bridge, including uncertain mapping and dependence.
    pub bridge: &'a BridgeContract,
    /// Current ledger whose selected source identities authorized the native update.
    pub ledger: &'a EvidenceLedger,
    /// Exact selected model/prior/experiment/mapping source identities.
    pub origins: PanelExperimentOrigins,
    /// Actual compiled per-world time-functional weights.
    pub projection: Vec<Vec<f64>>,
    /// Metadata of the original complete posterior law before conditioning.
    pub prior_metadata: DistributionMetadata,
    /// Declared experimental observed sample mean.
    pub observed_mean: f64,
    /// Known variance of one independent experimental observation.
    pub observation_variance: f64,
    /// Actual experiment sample size.
    pub sample_size: u64,
    /// Paired mapping intercepts in original shared parameter-slot order.
    pub intercepts: &'a [f64],
    /// Paired mapping slopes in the same order.
    pub slopes: &'a [f64],
    /// Dependency to which the original mapping slots were bound.
    pub mapping_dependency: SemanticDigest,
}
impl PanelExperimentExecutionProgram {
    /// Preserve the actual model/bridge/experiment/ledger inputs; performs no fitting.
    /// # Errors
    /// Existing bridge/ledger conversion errors or changed source/quantity binding.
    pub fn from_source(
        source: PanelExperimentExecutionSource<'_>,
    ) -> Result<Self, PanelArtifactError> {
        let (_, ledger) = EvidenceLedgerArtifact::seal(source.ledger, &[]).map_err(invalid)?;
        let program = Self {
            version: 1,
            panel: source.panel,
            bridge: FullBridgeContractWire::from_domain(source.bridge).map_err(invalid)?,
            ledger,
            origins: source.origins,
            projection: source.projection.iter().map(|weights| bits(weights)).collect(),
            prior_metadata: source.prior_metadata,
            observed: source.observed_mean.to_bits(),
            variance: source.observation_variance.to_bits(),
            sample_size: source.sample_size,
            intercepts: bits(source.intercepts),
            slopes: bits(source.slopes),
            mapping_dependency: *source.mapping_dependency.as_bytes(),
            policy: "frozen_joint_importance".into(),
        };
        program.validate()?;
        Ok(program)
    }
    /// Full numerical/selected-authority identity; unrelated ledger history is not a dependency.
    /// # Errors
    /// Serialization errors.
    pub fn identity(&self) -> Result<[u8; 32], PanelArtifactError> {
        // The frozen panel program (dominated by its retained draws) is represented by its own
        // complete identity, which it digests once; serializing it again here would repeat
        // that work for every export and consumption of the same immutable program.
        let mut value = serde_json::to_value(PanelExperimentIdentityView {
            version: self.version,
            panel: self.panel.identity()?,
            bridge: &self.bridge,
            origins: &self.origins,
            projection: &self.projection,
            prior_metadata: &self.prior_metadata,
            observed: self.observed,
            variance: self.variance,
            sample_size: self.sample_size,
            intercepts: &self.intercepts,
            slopes: &self.slopes,
            mapping_dependency: self.mapping_dependency,
            policy: &self.policy,
        })
        .map_err(invalid)?;
        value.as_object_mut().ok_or_else(|| invalid("experiment request shape"))?;
        Ok(CanonicalWriter::digest_bytes(
            "antecedent.joint_panel.experimental_program.v2",
            &serde_json::to_vec(&value).map_err(invalid)?,
        ))
    }
    fn validate(&self) -> Result<(), PanelArtifactError> {
        if self.version != 1
            || self.policy != "frozen_joint_importance"
            || self.mapping_dependency != *self.panel.dependency_identity().as_bytes()
        {
            return Err(invalid("unknown experiment version/policy or source dependency mismatch"));
        }
        let bridge = self.bridge.to_domain().map_err(invalid)?;
        if bridge.estimand_bridge.is_none() {
            return Err(invalid("experiment has no complete estimand translation"));
        }
        let ledger = EvidenceLedgerArtifact::from_meta(&self.ledger, None).map_err(invalid)?;
        validate_origins(
            ledger.ledger(),
            &self.origins,
            &bridge,
            self.panel.dependency_identity(),
        )?;
        let count = self.prior_metadata.shape[0];
        if count == 0
            || self.intercepts.len() != count
            || self.slopes.len() != count
            || self.projection.len() != self.panel.world_ids().len()
            || self.prior_metadata.shape[1] != self.panel.world_ids().len()
            || self.sample_size == 0
            || !f64::from_bits(self.observed).is_finite()
            || !f64::from_bits(self.variance).is_finite()
            || f64::from_bits(self.variance) <= 0.0
        {
            return Err(invalid("invalid original experimental axes or Gaussian sampling model"));
        }
        let target = self.panel.target()?;
        if self.prior_metadata.identity.causal_contract_id != target.identity().to_hex()
            || self.prior_metadata.identity.snapshot_id != self.panel.dependency_identity().to_hex()
            || !matches!(
                self.prior_metadata.identity.semantic,
                crate::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior
            )
        {
            return Err(invalid(
                "experimental prior is not the exact original compiled posterior mean-functional law",
            ));
        }
        Ok(())
    }
    /// Independently evaluate the stored frozen panel and condition via the same probability engine.
    /// # Errors
    /// All original replay failures, failed slots, changed projection, bridge/ledger or likelihood.
    pub fn replay(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<HistoricalPanelExperiment, PanelArtifactError> {
        cancelled(ctx)?;
        self.validate()?;
        let original = self.panel.replay(ctx)?;
        let weights = self.projection.iter().map(|row| floats(row)).collect::<Vec<_>>();
        let count = original.draws().len();
        let width = weights.len();
        if count != self.prior_metadata.shape[0] {
            return Err(invalid("frozen posterior slot count differs"));
        }
        let mut quote = Counter(0);
        serde_json::to_writer(&mut quote, self).map_err(invalid)?;
        let bytes = count
            .checked_mul(width + 8)
            .and_then(|n| n.checked_mul(24))
            .and_then(|n| n.checked_add(quote.0.checked_mul(4)?))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("experiment replay quote overflow"))?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let mut rows = Vec::with_capacity(count * width);
        for slot in original.draws() {
            cancelled(ctx)?;
            let row = slot.as_ref().ok_or_else(|| {
                invalid("failed original draw slots cannot authorize an experimental posterior")
            })?;
            rows.extend(antecedent_estimate::joint_panel::project_panel_responses(row, &weights)?);
        }
        let prior = DistributionArtifact::new(self.prior_metadata.clone(), rows)?;
        let coefficients = contrast_coefficients(&self.panel)?;
        let values =
            antecedent_prob::finite_belief::linear_functional_rows(prior.draws(), &coefficients)
                .map_err(invalid)?;
        let intercepts = floats(&self.intercepts);
        let slopes = floats(&self.slopes);
        let bridge = self.bridge.to_domain().map_err(invalid)?;
        bridge
            .validate_finite_affine_mapping(&intercepts, &slopes, &values)
            .map_err(|error| invalid(format!("{error:?}")))?;
        let prior_weights =
            prior.metadata().weights.clone().unwrap_or_else(|| vec![1.0 / count as f64; count]);
        let posterior = antecedent_prob::finite_belief::AffineGaussianExperiment {
            observed_mean: f64::from_bits(self.observed),
            observation_variance: f64::from_bits(self.variance),
            sample_size: self.sample_size,
            target_values: &values,
            intercepts: &intercepts,
            slopes: &slopes,
        }
        .condition(&prior_weights)
        .map_err(invalid)?;
        let mut metadata = prior.metadata().clone();
        metadata.weights = Some(posterior.clone());
        let bridge = self.bridge.to_domain().map_err(invalid)?;
        let dependency = experiment_update_identity(
            &metadata.identity.causal_contract_id,
            self.panel.dependency_identity(),
            bridge.identity(),
            f64::from_bits(self.observed),
            f64::from_bits(self.variance),
            self.sample_size,
            &intercepts,
            &slopes,
            &posterior,
            &self.origins.ordered(),
        );
        metadata.identity.causal_contract_id = antecedent_core::to_hex(&dependency);
        metadata.identity.provider_id =
            "antecedent.compiled_joint_panel.conditional_experiment".into();
        metadata.calibration = DistributionCalibration::Unmeasured;
        metadata.trust = DistributionTrust::Unverified;
        let law = DistributionArtifact::new(metadata, prior.draws().to_vec())?;
        Ok(HistoricalPanelExperiment {
            law,
            program: self.clone(),
            dependency: SemanticDigest::from_bytes(dependency),
            retention,
        })
    }
}

/// Recomputed conditional numerical law and historical authority, without current permission.
#[derive(Clone, Debug)]
pub struct HistoricalPanelExperiment {
    law: DistributionArtifact,
    program: PanelExperimentExecutionProgram,
    dependency: SemanticDigest,
    retention: Arc<AdmissionLease>,
}
impl HistoricalPanelExperiment {
    /// Explicit numerical component; this carries no current scientific permission.
    #[must_use]
    pub fn numerical_law(&self) -> &DistributionArtifact {
        &self.law
    }
    /// Exact full stored experiment/model/bridge/selected-source program.
    #[must_use]
    pub fn program(&self) -> &PanelExperimentExecutionProgram {
        &self.program
    }
    /// Exact model, bridge, data update and selected authority dependency.
    #[must_use]
    pub const fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// Historical ledger replay cannot substitute for caller-supplied current authority.
    #[must_use]
    pub const fn grants_current_authority(&self) -> bool {
        false
    }
    /// Conservative admitted numerical output reservation, not process RSS.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u16,
    identity: [u8; 32],
    program: PanelExperimentExecutionProgram,
    result: Vec<u8>,
    result_identity: crate::distribution_artifact::DistributionIdentity,
}

/// Export the already checked actual updated joint law with exact original frozen inputs.
/// # Errors
/// Bridge/ledger failure, serialization, unsupported limits, resource or cancellation.
pub fn encode_panel_experiment(
    program: PanelExperimentExecutionProgram,
    law: &DistributionArtifact,
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, PanelArtifactError> {
    cancelled(ctx)?;
    program.validate()?;
    let record = Record {
        version: 1,
        identity: program.identity()?,
        program,
        result: law.to_bytes("joint_panel_experiment_law")?,
        result_identity: law.metadata().identity.clone(),
    };
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, &record).map_err(invalid)?;
    if counter.0 > MAX_JOINT_PANEL_ARTIFACT_BYTES {
        return Err(invalid("experiment payload exceeds byte ceiling"));
    }
    let _lease = admit(
        ctx,
        (counter.0 as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("experiment encoding quote overflow"))?,
    )?;
    let payload = serde_json::to_vec(&record).map_err(invalid)?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: "joint_panel_experiment",
            note: "historical_conditional_experiment",
            max_bytes: MAX_JOINT_PANEL_ARTIFACT_BYTES,
            artifact_id: "joint_panel_experiment",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: "joint_panel_experiment_v1",
            content_type: crate::CONTENT_BYTES,
            bytes: &payload,
        }],
    )
    .map_err(|error| PanelArtifactError::Io(error.into_io("experiment encoding")))
}
/// Recompute historical frozen response projection and the declared experimental update.
/// # Errors
/// Unknown versions, tampering, semantic/bridge/ledger failure, changed result or resources.
pub fn consume_panel_experiment(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<HistoricalPanelExperiment, PanelArtifactError> {
    let (_lease, payload) = decode_payload(bytes, ctx)?;
    let record: Record = serde_json::from_slice(&payload).map_err(invalid)?;
    if record.version != 1 || record.identity != record.program.identity()? {
        return Err(PanelArtifactError::Mismatch);
    }
    let expected = DistributionArtifact::from_bytes(&record.result, &record.result_identity)?;
    let replay = record.program.replay(ctx)?;
    if expected.metadata() != replay.law.metadata()
        || bits(expected.draws()) != bits(replay.law.draws())
    {
        return Err(PanelArtifactError::Mismatch);
    }
    Ok(replay)
}

/// Canonical dependency shared by native execution and independent experimental replay.
/// Fields include selected evidence identities; unrelated ledger operations do not invalidate it.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "one canonical scientific update identity binds every declared numerical and origin dependency"
)]
pub fn experiment_update_identity(
    target: &str,
    source: SemanticDigest,
    bridge: SemanticDigest,
    observed: f64,
    variance: f64,
    n: u64,
    intercepts: &[f64],
    slopes: &[f64],
    weights: &[f64],
    origins: &[String],
) -> [u8; 32] {
    let mut identity = CanonicalWriter::new("antecedent.joint_panel.experimental_update.v1");
    identity
        .word(1)
        .text(target)
        .text(&source.to_hex())
        .text(&bridge.to_hex())
        .word(observed.to_bits())
        .word(variance.to_bits())
        .word(n);
    for ((a, b), weight) in intercepts.iter().zip(slopes).zip(weights) {
        identity.word(a.to_bits()).word(b.to_bits()).word(weight.to_bits());
    }
    for origin in origins {
        identity.text(origin);
    }
    identity.finish()
}
fn contrast_coefficients(program: &PanelExecutionProgram) -> Result<Vec<f64>, PanelArtifactError> {
    let target = program.target()?;
    let antecedent_core::TargetForm::Contrast(contrast) = target.form() else {
        return Err(invalid("experiment requires an assembled checked contrast"));
    };
    let mut coefficients = vec![0.0; program.world_ids().len()];
    for term in contrast.terms() {
        let index = program
            .world_ids()
            .iter()
            .position(|world| *world == term.world.raw())
            .ok_or_else(|| invalid("missing experimental contrast world"))?;
        coefficients[index] = term.coefficient;
    }
    Ok(coefficients)
}
fn validate_origins(
    ledger: &EvidenceLedger,
    origins: &PanelExperimentOrigins,
    bridge: &BridgeContract,
    dependency: SemanticDigest,
) -> Result<(), PanelArtifactError> {
    let parse = |value: &str| {
        SourceId::from_hex(value).ok_or_else(|| invalid("malformed experimental source identity"))
    };
    let model = parse(&origins.model)?;
    let experiment = parse(&origins.experiment)?;
    let mapping = parse(&origins.mapping)?;
    for id in origins.ordered() {
        let id = parse(&id)?;
        if !ledger.source(id).is_some_and(|source| source.status == SourceStatus::Active) {
            return Err(invalid("experimental source is not active in historical authority"));
        }
    }
    if ledger.source(model).is_none_or(|source| source.declaration.label != dependency.to_hex()) {
        return Err(invalid("model source does not bind actual fit dependency"));
    }
    for origin in [model, mapping]
        .into_iter()
        .chain(origins.priors.iter().map(|id| parse(id)).collect::<Result<Vec<_>, _>>()?)
    {
        if !matches!(ledger.overlap(origin, experiment).map_err(invalid)?, Overlap::Independent) {
            return Err(invalid("experimental evidence overlaps or has unknown origin"));
        }
    }
    if matches!(
        bridge.source_dependence,
        Some(antecedent_contract::bridge::SourceMappingDependence::Independent { .. })
    ) && !matches!(ledger.overlap(model, mapping).map_err(invalid)?, Overlap::Independent)
    {
        return Err(invalid("mapping independence contradicts source origins"));
    }
    Ok(())
}
fn decode_payload(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(AdmissionLease, Vec<u8>), PanelArtifactError> {
    cancelled(ctx)?;
    if bytes.len() > MAX_JOINT_PANEL_ARTIFACT_BYTES {
        return Err(invalid("experiment input exceeds ceiling"));
    }
    let reader = crate::ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    let logical = reader.manifest().sections.iter().try_fold(0u64, |total, section| {
        total
            .checked_add(section.uncompressed_size)
            .ok_or_else(|| invalid("experiment decoded quote overflow"))
    })?;
    if logical > MAX_JOINT_PANEL_ARTIFACT_BYTES as u64 {
        return Err(invalid("experiment decoded payload exceeds ceiling"));
    }
    let lease = admit(
        ctx,
        logical.checked_mul(8).ok_or_else(|| invalid("experiment decoded quote overflow"))?,
    )?;
    let mut payloads = crate::decode_container(
        "joint_panel_experiment",
        MAX_JOINT_PANEL_ARTIFACT_BYTES,
        &["joint_panel_experiment_v1"],
        bytes,
    )
    .map_err(|error| PanelArtifactError::Io(error.into_io("experiment decoding")))?;
    Ok((lease, payloads.pop().ok_or_else(|| invalid("missing experiment feature"))?))
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}
fn floats(values: &[u64]) -> Vec<f64> {
    values.iter().copied().map(f64::from_bits).collect()
}
fn invalid(error: impl std::fmt::Display) -> PanelArtifactError {
    PanelArtifactError::Invalid(error.to_string())
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), PanelArtifactError> {
    if ctx.cancellation.is_cancelled() { Err(PanelArtifactError::Cancelled) } else { Ok(()) }
}
fn admit(ctx: &ExecutionContext, bytes: u64) -> Result<AdmissionLease, PanelArtifactError> {
    let limit = ctx.memory_limit_bytes();
    if limit.is_some_and(|limit| bytes > limit) {
        return Err(PanelArtifactError::Resource("configured memory limit".into()));
    }
    ctx.acquire_admission(0, bytes).map_err(|error| PanelArtifactError::Resource(error.to_string()))
}
struct Counter(usize);
impl std::io::Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("byte count overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

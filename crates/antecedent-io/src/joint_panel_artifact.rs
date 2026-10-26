//! Exact frozen joint-panel execution, replayed by the existing model engine.
//! Historical numerical replay is not current scientific authorization.
// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::compiled_identification_wire::CompiledIdentificationWire;
use crate::scientific_target_wire::ScientificTargetWire;
use antecedent_core::execution::AdmissionLease;
use antecedent_core::{
    Determinism, ExecutionContext, KernelPolicy, RngFactory, ScientificTarget, SemanticDigest,
};
use antecedent_estimate::joint_panel::{
    self as engine, PanelDraw, PanelFit, PanelFitDiagnostics, PanelFitOptions, PanelInference,
    PanelProgram, PanelResponse,
};
use antecedent_identify::CompiledIdentificationProduct;
use engine::time_varying::{
    TemporalModelChoice, TemporalPanelAttempt, TemporalPanelDraw, TemporalPanelFit,
    TemporalPanelSpec,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Maximum logical payload, also enforced before decompressing a stored section.
pub const MAX_JOINT_PANEL_ARTIFACT_BYTES: usize = 256_000_000;
/// Feature version of exact frozen generic-channel execution.
pub const JOINT_PANEL_ARTIFACT_VERSION: u16 = 1;
/// Feature version of frozen execution that also carries a broader mechanism family
/// (geometric adstock, Hill saturation, group-specific saturation or spatial residuals).
/// A reader that predates the family refuses it instead of misreading the mechanism.
pub const JOINT_PANEL_FAMILY_ARTIFACT_VERSION: u16 = 2;

/// The feature version a program needs: reviewed-family programs keep version one, so their
/// stored form and identity are unchanged.
fn artifact_version(data: &engine::PanelData, options: &PanelFitOptions) -> u16 {
    let family_prior = matches!(
        &options.inference,
        PanelInference::Bayesian { prior, .. } if !prior.family.is_default()
    );
    if data.family.is_reviewed() && !family_prior {
        JOINT_PANEL_ARTIFACT_VERSION
    } else {
        JOINT_PANEL_FAMILY_ARTIFACT_VERSION
    }
}

/// Actual original source partition/cadence and encoded schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelSourceMetadata {
    /// Original full source schema, retaining units and role metadata.
    pub schema: crate::SchemaWire,
    /// Original group order, source snapshot, cadence and row counts.
    pub units: Vec<PanelUnitMetadata>,
}
/// One actual source unit, including the original pre-period row count.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelUnitMetadata {
    /// Original panel unit identity.
    pub id: u32,
    /// Actual regular cadence; no calendar duration is invented.
    pub interval_ns: u64,
    /// Original source rows, including observed carryover history.
    pub rows: usize,
    /// Full original storage snapshot identity.
    pub content_digest: [u8; 32],
}

/// Full frozen numerical fit, including original diagnostics, selected model and draw failures.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPanelFit {
    law: FrozenLaw,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case", deny_unknown_fields)]
enum FrozenLaw {
    Invariant {
        point: PanelDraw,
        draws: Vec<Option<PanelDraw>>,
        diagnostics: PanelFitDiagnostics,
        source: [u8; 32],
        inference: PanelInference,
    },
    Varying {
        point: TemporalPanelDraw,
        draws: Vec<Option<TemporalPanelDraw>>,
        attempts: Vec<TemporalPanelAttempt>,
        smooth_fallback: bool,
        selection_counts: Vec<(TemporalModelChoice, usize)>,
        posterior_diagnostics: Option<[f64; 3]>,
        posterior_converged: Option<bool>,
        uncertainty: String,
        claim: String,
        calibration: String,
        source: [u8; 32],
    },
}
impl FrozenPanelFit {
    /// Snapshot the actual invariant fit; numerical draw slots and diagnostics are unchanged.
    #[must_use]
    pub fn invariant(fit: &PanelFit) -> Self {
        Self {
            law: FrozenLaw::Invariant {
                point: fit.point.clone(),
                draws: fit.draws.clone(),
                diagnostics: fit.diagnostics.clone(),
                source: *fit.source_identity.as_bytes(),
                inference: fit.inference.clone(),
            },
        }
    }
    /// Snapshot the complete selected temporal procedure, including selection uncertainty.
    #[must_use]
    pub fn varying(fit: &TemporalPanelFit) -> Self {
        Self {
            law: FrozenLaw::Varying {
                point: fit.point.clone(),
                draws: fit.draws.clone(),
                attempts: fit.attempts.clone(),
                smooth_fallback: fit.smooth_fallback,
                selection_counts: fit.selection_counts.clone(),
                posterior_diagnostics: fit.posterior_diagnostics,
                posterior_converged: fit.posterior_converged,
                uncertainty: fit.uncertainty.clone(),
                claim: fit.claim.clone(),
                calibration: fit.calibration.clone(),
                source: *fit.source_identity.as_bytes(),
            },
        }
    }
}

/// Complete compiled panel product, without a public builder or current authorization.
#[derive(Clone, Debug)]
pub struct PanelExecutionProgram {
    inner: Arc<ProgramWire>,
    retention: Option<Arc<AdmissionLease>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramWire {
    version: u16,
    source: PanelSourceMetadata,
    data: engine::PanelData,
    options: PanelFitOptions,
    temporal: Option<TemporalPanelSpec>,
    programs: Vec<PanelProgram>,
    world_ids: Vec<u32>,
    target: ScientificTargetWire,
    proof: CompiledIdentificationWire,
    source_identity: [u8; 32],
    dependency: [u8; 32],
    seed: u64,
    kernel: [bool; 3],
    strict: bool,
    frozen: FrozenPanelFit,
    /// Canonical digest and serialized length, computed once for this immutable program.
    #[serde(skip)]
    canonical: std::sync::OnceLock<([u8; 32], usize)>,
}
impl Serialize for PanelExecutionProgram {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.inner.as_ref().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for PanelExecutionProgram {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self { inner: Arc::new(ProgramWire::deserialize(deserializer)?), retention: None })
    }
}
/// Original execution inputs borrowed from the actual compiled owner.
pub struct PanelExecutionSource<'a> {
    /// Original source schema/partition/cadence.
    pub source: PanelSourceMetadata,
    /// Exact numerical source, including actual histories, keyed channels and expressions.
    pub data: &'a engine::PanelData,
    /// Actual numerical inference, priors and optimizer policy.
    pub options: &'a PanelFitOptions,
    /// Full prespecified change/smoothing procedure, when present.
    pub temporal: Option<&'a TemporalPanelSpec>,
    /// Exact ordered evaluation programs.
    pub programs: Vec<PanelProgram>,
    /// Exact scientific world key for each ordered numerical program.
    pub world_ids: Vec<u32>,
    /// Actual full scientific target.
    pub target: &'a ScientificTarget,
    /// Actual retained target-bound proof and unresolved support.
    pub proof: &'a CompiledIdentificationProduct,
    /// Original source-adaptation dependency.
    pub source_identity: SemanticDigest,
    /// Actual fit/execution dependency, including policy and random stream.
    pub dependency: SemanticDigest,
    /// Complete frozen fitted numerical law.
    pub frozen: FrozenPanelFit,
    /// Original execution context; only numerical seed/kernel/determinism are persisted.
    pub context: &'a ExecutionContext,
}

impl PanelExecutionProgram {
    /// Reconstruct the controlled point-response expression from the actual frozen
    /// panel law. Historical numerical replay grants no current causal permission.
    /// # Errors
    /// Missing world, invalid controls/time functional, numerical or resource failure.
    pub fn controlled_response_expression(
        &self,
        world: u32,
        controls: &[engine::expression::PanelExpressionControl],
        time_weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<engine::expression::CompiledPanelExpression, PanelArtifactError> {
        self.validate()?;
        let index = self
            .inner
            .world_ids
            .iter()
            .position(|id| *id == world)
            .ok_or_else(|| invalid("missing certificate base world"))?;
        let path = match &self.inner.frozen.law {
            FrozenLaw::Invariant { point, .. } => vec![point.clone()],
            FrozenLaw::Varying { point, .. } => point
                .parameters
                .iter()
                .map(|parameters| engine::PanelDraw {
                    parameters: parameters.clone(),
                    group_intercepts: point.baseline.group_intercepts.clone(),
                })
                .collect(),
        };
        Ok(engine::expression::compile_panel_expression(
            &self.inner.data,
            &path,
            &self.inner.programs[index],
            controls,
            time_weights,
            ctx,
        )?)
    }
    /// Evaluate a reference decomposition from this exact frozen shared law.
    ///
    /// Mechanism maps refer to parameter periods in the original joint fit;
    /// separately fitted marginal laws are never zipped together. This historical
    /// numerical projection grants no current causal authorization.
    ///
    /// # Errors
    /// Invalid program, parameter map, endpoints, resources or cancellation.
    pub fn decompose_reference(
        &self,
        request: &engine::decomposition::PanelDecompositionRequest,
        periods: &engine::decomposition::MechanismPeriodMap,
        ctx: &ExecutionContext,
    ) -> Result<engine::decomposition::PanelDecomposition, PanelArtifactError> {
        self.validate()?;
        Ok(match &self.inner.frozen.law {
            FrozenLaw::Invariant { point, draws, .. } => {
                engine::decomposition::decompose_invariant_law(
                    &self.inner.data,
                    request,
                    periods,
                    point,
                    draws,
                    ctx,
                )?
            }
            FrozenLaw::Varying { point, draws, .. } => {
                engine::decomposition::decompose_temporal_law(
                    &self.inner.data,
                    request,
                    periods,
                    point,
                    draws,
                    ctx,
                )?
            }
        })
    }
    /// Retain exact compiled inputs and the actual fitted law; performs no fit or identification.
    /// # Errors
    /// Incoherent target/proof, source metadata or numerical policy.
    pub fn from_source(source: PanelExecutionSource<'_>) -> Result<Self, PanelArtifactError> {
        let mut count = Counter(0);
        serde_json::to_writer(
            &mut count,
            &(source.data, source.options, source.temporal, &source.programs, &source.frozen),
        )
        .map_err(invalid)?;
        if count.0 > MAX_JOINT_PANEL_ARTIFACT_BYTES {
            return Err(invalid("panel program exceeds byte ceiling"));
        }
        let retention = Arc::new(admit(
            source.context,
            (count.0 as u64)
                .checked_mul(4)
                .ok_or_else(|| invalid("compiled panel quote overflow"))?,
        )?);
        let kernel = source.context.kernel_policy;
        let program = Self {
            inner: Arc::new(ProgramWire {
                version: artifact_version(source.data, source.options),
                source: source.source,
                data: source.data.clone(),
                options: source.options.clone(),
                temporal: source.temporal.cloned(),
                programs: source.programs,
                world_ids: source.world_ids,
                target: ScientificTargetWire::from_domain(source.target)?,
                proof: CompiledIdentificationWire::from_domain(source.proof, source.target)?,
                source_identity: *source.source_identity.as_bytes(),
                dependency: *source.dependency.as_bytes(),
                seed: source.context.rng.master_seed(),
                kernel: [
                    kernel.allow_portable_optimized,
                    kernel.allow_arch_simd,
                    kernel.force_scalar,
                ],
                strict: matches!(source.context.determinism, Determinism::Strict),
                frozen: source.frozen,
                canonical: std::sync::OnceLock::new(),
            }),
            retention: Some(retention),
        };
        program.validate()?;
        Ok(program)
    }
    /// Complete compiled request/frozen-law identity for matching current caller authority.
    /// # Errors
    /// Encoding or size overflow.
    pub fn identity(&self) -> Result<[u8; 32], PanelArtifactError> {
        Ok(self.canonical()?.0)
    }
    /// The program is immutable, so its serialization (dominated by the frozen draws) is
    /// digested and measured once however many consumers ask for identity or a quote.
    fn canonical(&self) -> Result<([u8; 32], usize), PanelArtifactError> {
        if let Some(cached) = self.inner.canonical.get() {
            return Ok(*cached);
        }
        let bytes = serde_json::to_vec(self).map_err(invalid)?;
        let value = (
            antecedent_core::CanonicalWriter::digest_bytes(
                "antecedent.joint_panel.frozen_program.v1",
                &bytes,
            ),
            bytes.len(),
        );
        Ok(*self.inner.canonical.get_or_init(|| value))
    }
    /// Complete stored scientific target, without current permission.
    /// # Errors
    /// The same strict target-wire conversion failures as independent replay.
    pub fn target(&self) -> Result<ScientificTarget, PanelArtifactError> {
        Ok(self.inner.target.to_domain()?)
    }
    /// Original fit/execution dependency, not proof of current source standing.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        SemanticDigest::from_bytes(self.inner.dependency)
    }
    /// Ordered scientific world identifiers matching the numerical program axis.
    #[must_use]
    pub fn world_ids(&self) -> &[u32] {
        &self.inner.world_ids
    }
    fn validate(&self) -> Result<(), PanelArtifactError> {
        if self.inner.version != artifact_version(&self.inner.data, &self.inner.options) {
            return Err(invalid("unsupported panel artifact feature version"));
        }
        self.inner.data.validate()?;
        engine::validate_fit_options(&self.inner.data, &self.inner.options)?;
        engine::validate_panel_programs(&self.inner.data, &self.inner.programs)?;
        if let Some(spec) = &self.inner.temporal {
            engine::time_varying::validate_temporal_spec(&self.inner.data, spec)?;
        }
        let target = self.inner.target.to_domain()?;
        let _historical_proof = self.inner.proof.to_domain(&target)?;
        crate::convert::schema_from_wire(&self.inner.source.schema)?;
        if self.inner.source.units.len() != self.inner.data.groups.len()
            || self.inner.programs.is_empty()
            || self.inner.world_ids.len() != self.inner.programs.len()
            || self.inner.world_ids.iter().collect::<std::collections::BTreeSet<_>>().len()
                != self.inner.world_ids.len()
        {
            return Err(invalid("panel source partition or program axis mismatch"));
        }
        let interval = self.inner.source.units.first().map_or(0, |unit| unit.interval_ns);
        for (unit, group) in self.inner.source.units.iter().zip(&self.inner.data.groups) {
            if interval == 0
                || unit.interval_ns != interval
                || unit.id.to_string() != group.id
                || unit.rows != group.outcome.len() + self.inner.data.lag_order
            {
                return Err(invalid("source cadence/history/partition binding mismatch"));
            }
        }
        let source = self.inner.data.identity()?;
        match &self.inner.frozen.law {
            FrozenLaw::Invariant { source: stored, inference, .. } => {
                if *stored != *source.as_bytes()
                    || self.inner.temporal.is_some()
                    || *inference != self.inner.options.inference
                {
                    return Err(invalid("frozen invariant fit/source/policy mismatch"));
                }
            }
            FrozenLaw::Varying { source: stored, .. } => {
                if *stored != *source.as_bytes() || self.inner.temporal.is_none() {
                    return Err(invalid("frozen temporal fit/source mismatch"));
                }
            }
        }
        Ok(())
    }
    /// Replay the frozen model through the same numerical execution engines, preserving slots.
    /// # Errors
    /// Corrupt semantics, unsupported numerical parameters, admission or cancellation.
    pub fn replay(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<HistoricalPanelExecution, PanelArtifactError> {
        cancelled(ctx)?;
        self.validate()?;
        if let Some(retention) = &self.retention {
            retention.observe_current();
        }
        // The numerical engine admits its own retained output. This quote covers
        // only the replay's retained serialized program storage.
        let bytes = self
            .canonical()?
            .1
            .checked_mul(4)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("panel replay quote overflow"))?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let mut context = ctx.clone();
        context.rng = RngFactory::from_seed(self.inner.seed);
        context.kernel_policy = KernelPolicy {
            allow_portable_optimized: self.inner.kernel[0],
            allow_arch_simd: self.inner.kernel[1],
            force_scalar: self.inner.kernel[2],
        };
        context.determinism =
            if self.inner.strict { Determinism::Strict } else { Determinism::PreferFast };
        let (point, draws, output_retention) = self.evaluate(&context)?.into_parts();
        Ok(HistoricalPanelExecution {
            target: self.inner.target.to_domain()?,
            program: self.clone(),
            point,
            draws,
            retention,
            output_retention,
        })
    }
    fn evaluate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<engine::PanelLawResponses, PanelArtifactError> {
        match &self.inner.frozen.law {
            FrozenLaw::Invariant { point, draws, .. } => Ok(engine::evaluate_panel_law(
                &self.inner.data,
                point,
                draws,
                &self.inner.programs,
                ctx,
            )?),
            FrozenLaw::Varying { point, draws, .. } => {
                Ok(engine::time_varying::evaluate_temporal_law(
                    &self.inner.data,
                    point,
                    draws,
                    &self.inner.programs,
                    ctx,
                )?)
            }
        }
    }
}

/// A historical numerical replay; never current scientific permission.
#[derive(Clone, Debug)]
pub struct HistoricalPanelExecution {
    program: PanelExecutionProgram,
    target: ScientificTarget,
    point: Vec<PanelResponse>,
    draws: Vec<Option<Vec<PanelResponse>>>,
    retention: Arc<AdmissionLease>,
    output_retention: Arc<AdmissionLease>,
}
impl HistoricalPanelExecution {
    /// Exact historical scientific question, including original assumptions.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Recomputed means in original program order.
    #[must_use]
    pub fn point(&self) -> &[PanelResponse] {
        &self.point
    }
    /// Recomputed aligned draw rows; failed original slots remain None.
    #[must_use]
    pub fn draws(&self) -> &[Option<Vec<PanelResponse>>] {
        &self.draws
    }
    /// Complete validated original compiled program and frozen law.
    #[must_use]
    pub fn program(&self) -> &PanelExecutionProgram {
        &self.program
    }
    /// Replay and reproducibility do not establish current causal authorization.
    #[must_use]
    pub const fn grants_current_authority(&self) -> bool {
        false
    }
    /// Admitted retained result reservation; not process RSS.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.output_retention.observe_current();
        self.retention.remainder().bytes().saturating_add(self.output_retention.remainder().bytes())
    }
}

/// Container, proof, engine, identity, resource or cancellation failure.
#[derive(Debug, thiserror::Error)]
pub enum PanelArtifactError {
    /// Existing container or semantic wire error.
    #[error(transparent)]
    Io(#[from] crate::IoError),
    /// Existing panel numerical engine error.
    #[error(transparent)]
    Engine(#[from] engine::PanelError),
    /// Unknown feature or malformed semantic input.
    #[error("joint panel artifact: {0}")]
    Invalid(String),
    /// The stored result/request does not match replay or caller authority.
    #[error("joint panel artifact mismatch")]
    Mismatch,
    /// Refused before the bounded allocation/decompression.
    #[error("joint panel artifact resource refusal: {0}")]
    Resource(String),
    /// Cancelled before a complete replay or artifact.
    #[error("joint panel artifact cancelled")]
    Cancelled,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u16,
    identity: [u8; 32],
    program: PanelExecutionProgram,
    point: Vec<PanelResponse>,
    draws: Vec<Option<Vec<PanelResponse>>>,
}

/// Encode exact compiled/fitted input and the original already executed result.
/// # Errors
/// Invalid proof/program, bounded serialization, resource or cancellation failure.
pub fn encode_panel_execution(
    program: PanelExecutionProgram,
    point: &[PanelResponse],
    draws: &[Option<Vec<PanelResponse>>],
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, PanelArtifactError> {
    cancelled(ctx)?;
    program.validate()?;
    let identity = program.identity()?;
    encode_record(
        &EncodingRecord { version: program.inner.version, identity, program, point, draws },
        ctx,
    )
}
/// Own the consumed program while borrowing its executed outputs. The wire
/// layout matches `Record` without copying the retained draw law.
#[derive(Serialize)]
struct EncodingRecord<'a> {
    version: u16,
    identity: [u8; 32],
    program: PanelExecutionProgram,
    point: &'a [PanelResponse],
    draws: &'a [Option<Vec<PanelResponse>>],
}
fn encode_record(
    record: &EncodingRecord<'_>,
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, PanelArtifactError> {
    // One serialization pass: the quote is 4x the payload, so a payload larger than a
    // quarter of the memory limit is refused as soon as the writer reaches it.
    let limit = ctx.memory_limit_bytes().map_or(MAX_JOINT_PANEL_ARTIFACT_BYTES, |bytes| {
        usize::try_from(bytes / 4).unwrap_or(usize::MAX).min(MAX_JOINT_PANEL_ARTIFACT_BYTES)
    });
    let mut payload = Vec::new();
    serde_json::to_writer(&mut Limited { bytes: &mut payload, limit }, record).map_err(
        |error| {
            if error.is_io() {
                if limit < MAX_JOINT_PANEL_ARTIFACT_BYTES {
                    PanelArtifactError::Resource("configured memory limit".into())
                } else {
                    invalid("panel payload exceeds byte ceiling")
                }
            } else {
                invalid(error)
            }
        },
    )?;
    let _lease = admit(
        ctx,
        (payload.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("panel serialization quote overflow"))?,
    )?;
    cancelled(ctx)?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: "joint_panel_execution",
            note: "historical_frozen_joint_panel",
            max_bytes: MAX_JOINT_PANEL_ARTIFACT_BYTES,
            artifact_id: "joint_panel_execution",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: "joint_panel_execution_v1",
            content_type: crate::CONTENT_BYTES,
            bytes: &payload,
        }],
    )
    .map_err(|error| PanelArtifactError::Io(error.into_io("panel encoding")))
}
struct Limited<'a> {
    bytes: &'a mut Vec<u8>,
    limit: usize,
}
impl std::io::Write for Limited<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.bytes.len().saturating_add(buf.len()) > self.limit {
            return Err(std::io::Error::other("payload exceeds limit"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
/// Independently replay the same engine and verify every original response/draw slot.
/// # Errors
/// Corrupt/unsupported request, changed result, admission or cancellation failure.
pub fn consume_panel_execution(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<HistoricalPanelExecution, PanelArtifactError> {
    let (_lease, payload) = decode_payload(bytes, ctx)?;
    let record: Record = serde_json::from_slice(&payload).map_err(invalid)?;
    if record.version != record.program.inner.version {
        return Err(invalid("unsupported panel feature version"));
    }
    if record.identity != record.program.identity()? {
        return Err(PanelArtifactError::Mismatch);
    }
    let replay = record.program.replay(ctx)?;
    if replay.point != record.point || replay.draws != record.draws {
        return Err(PanelArtifactError::Mismatch);
    }
    Ok(replay)
}
fn decode_payload(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(AdmissionLease, Vec<u8>), PanelArtifactError> {
    cancelled(ctx)?;
    if bytes.len() > MAX_JOINT_PANEL_ARTIFACT_BYTES {
        return Err(invalid("panel input exceeds byte ceiling"));
    }
    let reader = crate::ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    let logical = reader.manifest().sections.iter().try_fold(0u64, |sum, section| {
        sum.checked_add(section.uncompressed_size)
            .ok_or_else(|| invalid("panel decode quote overflow"))
    })?;
    if logical > MAX_JOINT_PANEL_ARTIFACT_BYTES as u64 {
        return Err(invalid("panel decoded payload exceeds ceiling"));
    }
    let lease =
        admit(ctx, logical.checked_mul(8).ok_or_else(|| invalid("panel decode quote overflow"))?)?;
    let mut sections = crate::decode_container(
        "joint_panel_execution",
        MAX_JOINT_PANEL_ARTIFACT_BYTES,
        &["joint_panel_execution_v1"],
        bytes,
    )
    .map_err(|error| PanelArtifactError::Io(error.into_io("panel decoding")))?;
    Ok((lease, sections.pop().ok_or_else(|| invalid("missing panel feature"))?))
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

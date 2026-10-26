// SPDX-License-Identifier: MIT OR Apache-2.0
//! Retained nonlinear equilibrium execution and checked independent replay.
//! This family uses the existing bounded/checksummed artifact container. An
//! independently retained request digest prevents a consistently resealed edit
//! from being mistaken for the originally requested scientific computation.

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::cyclic_likelihood::{
    CyclicFit, CyclicFitOptions, CyclicParameter, fit_cyclic_likelihood,
};
use antecedent_model::nonlinear_equilibrium::{
    Clamp, EquilibriumModel, EquilibriumResult, RegimeRow, SolvePolicy,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const KIND: &str = "cyclic_equilibrium";
const SECTION: &str = "cyclic_equilibrium_v1";
/// Maximum logical/container bytes accepted by this family.
pub const MAX_CYCLIC_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;

/// Optional model-fitting stage; all rows are independent equilibrium replicates
/// conditional on known intervention targets. This is not a time-series license.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicFitRequest {
    /// Coordinates to estimate jointly.
    pub parameters: Vec<CyclicParameter>,
    /// Observations and their known hard-clamp regimes.
    pub rows: Vec<RegimeRow>,
    /// Deterministic numerical policy.
    pub options: CyclicFitOptions,
}

/// Complete retained target, specified model and optional statistical program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicExecutionRequest {
    /// Model template, stable equation identities, and joint disturbance law.
    pub model: EquilibriumModel,
    /// Optional joint fit, run before evaluation.
    pub fit: Option<CyclicFitRequest>,
    /// Prespecified hard-clamp worlds, in requested order.
    pub worlds: Vec<Vec<Clamp>>,
    /// Explicit aligned noise realizations shared across all worlds.
    pub noise: Vec<Vec<f64>>,
    /// Numerical starting state; not an equilibrium selection law.
    pub initial: Vec<f64>,
    /// Solve policy; part of retained execution identity.
    pub policy: SolvePolicy,
    /// Optional fully observed factual state for shared-noise counterfactuals.
    pub factual: Option<Vec<f64>>,
}

/// Recomputed answers; no nonparametric identification or calibration is implied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicExecutionReport {
    #[serde(skip)]
    retention: crate::retained_allocation::RetainedAllocation,
    #[serde(skip)]
    lowered: crate::cyclic_family_artifact::LoweredStamp,
    /// Stable request identity; retain independently before artifact consumption.
    pub identity: String,
    /// Optional joint fit and its numerical convergence facts.
    pub fit: Option<CyclicFit>,
    /// Results indexed by world, then aligned disturbance row.
    pub worlds: Vec<Vec<EquilibriumResult>>,
    /// Shared-noise abduction/action/prediction result for each requested world.
    pub counterfactuals: Option<Vec<EquilibriumResult>>,
    /// Always `specified_or_fitted_model_conditional` in this execution family.
    pub claim: String,
    /// Always `unmeasured`; no nominal interval/coverage claim is produced.
    pub calibration: String,
}

impl CyclicExecutionReport {
    /// Admitted bytes retained until this report and its clones are dropped.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.bytes()
    }

    /// Share the existing output admissions with a native family bridge.
    pub(crate) fn shared_retention(&self) -> crate::retained_allocation::RetainedAllocation {
        self.retention.clone()
    }

    /// Identity of the lowered program whose execution produced this report.
    ///
    /// `None` for a report no prepared handle stamped, such as an independently
    /// replayed artifact. The stamp is not part of the artifact or of equality.
    #[must_use]
    pub const fn lowered_program_identity(&self) -> Option<antecedent_core::SemanticDigest> {
        self.lowered.identity()
    }

    /// Stamp the identity of the lowered program that produced this report.
    #[must_use]
    pub fn with_lowered_program_identity(
        mut self,
        identity: antecedent_core::SemanticDigest,
    ) -> Self {
        self.lowered = crate::cyclic_family_artifact::LoweredStamp::of(identity);
        self
    }
}

/// Serialization, integrity, identity, scientific or execution failure.
#[derive(Debug, Error)]
pub enum CyclicArtifactError {
    /// Family size, shape or version limit.
    #[error("cyclic artifact invalid: {0}")]
    Invalid(String),
    /// Native admission rejected the requested memory or CPU reservation.
    #[error(transparent)]
    Admission(#[from] antecedent_core::execution::AdmissionError),
    /// Execution was cancelled by its bound execution policy.
    #[error("cyclic execution cancelled")]
    Cancelled,
    /// Model or statistical computation failed.
    #[error(transparent)]
    Execution(#[from] antecedent_model::nonlinear_equilibrium::EquilibriumError),
    /// Expected request does not match, or replay differs from stored results.
    #[error(
        "cyclic artifact identity/replay mismatch; next attempt: consume the artifact against the \
         identity its producer retained, or export it again from its prepared family"
    )]
    Mismatch,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    request: CyclicExecutionRequest,
    report: CyclicExecutionReport,
}

fn invalid(e: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(e.to_string())
}

#[cfg(test)]
std::thread_local! {
    static SYSTEM_ENGINE_EXECUTIONS: std::cell::Cell<usize> = const {std::cell::Cell::new(0)};
}

impl CyclicExecutionRequest {
    /// Identity of the complete retained scientific/numerical request.
    pub fn identity(&self) -> Result<String, CyclicArtifactError> {
        self.check_limits()?;
        let bytes = serde_json::to_vec(self).map_err(invalid)?;
        if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("request byte limit"));
        }
        Ok(CanonicalWriter::digest_bytes_hex("antecedent.cyclic_equilibrium.request.v1", &bytes))
    }

    fn check_limits(&self) -> Result<(), CyclicArtifactError> {
        self.model.validate()?;
        let n = self.model.order();
        if self.worlds.is_empty()
            || self.worlds.len() > 256
            || self.noise.is_empty()
            || self.noise.len() > 10_000
            || self.worlds.len().saturating_mul(self.noise.len()).saturating_mul(n) > 100_000
            || self.initial.len() != n
            || self.initial.iter().any(|v| !v.is_finite())
            || self.noise.iter().any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
            || self
                .factual
                .as_ref()
                .is_some_and(|v| v.len() != n || v.iter().any(|v| !v.is_finite()))
            || self.fit.as_ref().is_some_and(|f| f.rows.len() > 10_000 || f.parameters.len() > 1024)
        {
            return Err(invalid("retained request shape or size limit"));
        }
        Ok(())
    }

    /// Execute the retained fit and aligned intervention program without rebuilding
    /// a public analysis builder. Utility consumers can reuse the returned worlds.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        let prepared = PreparedCyclicExecution::prepare(self.clone(), ctx)?;
        prepared.estimate(ctx)
    }

    pub(crate) fn execute_bound(
        &self,
        fit: Option<&CyclicFit>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        #[cfg(test)]
        SYSTEM_ENGINE_EXECUTIONS.with(|count| count.set(count.get() + 1));
        let identity = self.identity()?;
        let n = self.model.order();
        let per_row = n
            .checked_mul(n)
            .and_then(|n| n.checked_mul(8))
            .and_then(|bytes| bytes.checked_add(n.checked_mul(128)?))
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or_else(|| invalid("output size overflow"))?;
        let retained = self
            .worlds
            .len()
            .checked_mul(
                self.noise
                    .len()
                    .checked_add(usize::from(self.factual.is_some()))
                    .ok_or_else(|| invalid("output size overflow"))?,
            )
            .and_then(|count| count.checked_mul(per_row))
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| invalid("output size overflow"))?;
        let retained = u64::try_from(retained)
            .map_err(invalid)?
            .checked_add(
                fit.map_or(Ok(0), crate::retained_allocation::allocation_bytes).map_err(invalid)?,
            )
            .ok_or_else(|| invalid("output size overflow"))?;
        let retention =
            ctx.acquire_admission(0, retained).map_err(CyclicArtifactError::Admission)?;
        if fit.as_ref().is_some_and(|f| !f.converged) {
            return Err(invalid("fit did not converge"));
        }
        let model = fit.as_ref().map_or(&self.model, |f| &f.model);
        let mut worlds = Vec::with_capacity(self.worlds.len());
        for clamps in &self.worlds {
            let mut rows = Vec::with_capacity(self.noise.len());
            for noise in &self.noise {
                rows.push(model.solve(noise, clamps, &self.initial, self.policy, ctx)?);
            }
            worlds.push(rows);
        }
        let counterfactuals = self
            .factual
            .as_ref()
            .map(|factual| {
                self.worlds
                    .iter()
                    .map(|clamps| model.counterfactual(factual, clamps, self.policy, ctx))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::RetainedBytes,
            retained,
        );
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::RetainedFull,
            1,
        );
        Ok(CyclicExecutionReport {
            retention: crate::retained_allocation::RetainedAllocation(vec![std::sync::Arc::new(
                retention,
            )]),
            identity,
            fit: fit.cloned(),
            worlds,
            counterfactuals,
            claim: "specified_or_fitted_model_conditional".into(),
            calibration: "unmeasured".into(),
            lowered: crate::cyclic_family_artifact::LoweredStamp::default(),
        })
    }

    /// Execute and encode in the existing bounded, checksummed section container.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicExecutionReport, Vec<u8>), CyclicArtifactError> {
        if artifact_id.trim().is_empty() || artifact_id.len() > 256 {
            return Err(invalid("artifact id"));
        }
        let report = self.execute(ctx)?;
        let container = encode_report(self, &report, artifact_id, ctx)?;
        Ok((report, container))
    }
}

fn encode_report(
    request: &CyclicExecutionRequest,
    report: &CyclicExecutionReport,
    artifact_id: &str,
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, CyclicArtifactError> {
    #[derive(Serialize)]
    struct BorrowedRecord<'a> {
        version: u32,
        request: &'a CyclicExecutionRequest,
        report: &'a CyclicExecutionReport,
    }
    let record = BorrowedRecord { version: 1, request, report };
    let logical = cyclic_serialized_size(&record, ctx)?;
    let bytes_quote = u64::try_from(logical)
        .map_err(invalid)?
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(65536))
        .ok_or_else(|| invalid("artifact encoding size overflow"))?;
    let _encoding =
        ctx.acquire_admission(0, bytes_quote).map_err(CyclicArtifactError::Admission)?;
    let bytes = serde_json::to_vec(&record).map_err(invalid)?;
    if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(invalid("logical section byte limit"));
    }
    let container = crate::encode_container(
        &crate::ArtifactContainer {
            kind: KIND,
            note: "model_conditional_calibration_unmeasured",
            max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection { id: SECTION, content_type: crate::CONTENT_BYTES, bytes: &bytes }],
    )
    .map_err(|e| invalid(e.into_io("cyclic artifact layout")))?;
    if ctx.cancellation.is_cancelled() {
        return Err(CyclicArtifactError::Cancelled);
    }
    Ok(container)
}

#[derive(Debug)]
struct RetainedCyclicFit {
    fit: CyclicFit,
    retention: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for RetainedCyclicFit {
    type Target = CyclicFit;
    fn deref(&self) -> &CyclicFit {
        &self.fit
    }
}
type SharedCyclicFit = std::sync::Arc<
    std::sync::Mutex<Option<(antecedent_core::KernelPolicy, std::sync::Arc<RetainedCyclicFit>)>>,
>;

/// Immutable compiled equilibrium product, sharing its fit across retargeted worlds.
#[derive(Clone, Debug)]
pub struct PreparedCyclicExecution {
    request: std::sync::Arc<CyclicExecutionRequest>,
    fit: SharedCyclicFit,
    fits_performed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
    resource_context: ExecutionContext,
}
impl PreparedCyclicExecution {
    /// Borrow the frozen scientific request without copying observations.
    #[must_use]
    pub fn request(&self) -> &CyclicExecutionRequest {
        &self.request
    }

    /// Compile/retain the complete request; fitting starts only on execution.
    pub fn prepare(
        request: CyclicExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        request.check_limits()?;
        let retention = retain_request(&request, ctx)?;
        request.identity()?;
        Ok(Self {
            request: std::sync::Arc::new(request),
            fit: std::sync::Arc::default(),
            fits_performed: std::sync::Arc::default(),
            retention: std::sync::Arc::new(retention),
            resource_context: ctx.clone(),
        })
    }
    /// Branch the frozen request, sharing fits only when their inputs are unchanged.
    pub fn rebind(
        &self,
        request: CyclicExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let mut next = Self::prepare(request, ctx)?;
        if next.request.model == self.request.model && next.request.fit == self.request.fit {
            next.fit = self.fit.clone();
        }
        Ok(next)
    }

    /// Number of causal fits performed by execution on this handle (zero at compilation).
    #[must_use]
    pub fn fits_performed(&self) -> usize {
        self.fits_performed.load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Request identity, including source data, target worlds and numerical policy.
    pub fn identity(&self) -> Result<String, CyclicArtifactError> {
        self.request.identity()
    }
    /// Compile/inspect the actual post-intervention SCC solve schedules.
    pub fn blocks(&self) -> Result<Vec<Vec<Vec<usize>>>, CyclicArtifactError> {
        let fit = self.fit.lock().map_err(|_| invalid("frozen fit lock poisoned"))?;
        let model = fit.as_ref().map_or(&self.request.model, |(_, fit)| &fit.model);
        self.request
            .worlds
            .iter()
            .map(|world| model.blocks(world).map_err(CyclicArtifactError::from))
            .collect()
    }
    /// Inspect the compiler's typed mechanism/simultaneous-equation schedules.
    pub fn execution_plans(
        &self,
    ) -> Result<Vec<antecedent_model::compile::CompiledSystemPlan>, CyclicArtifactError> {
        let fit = self.fit.lock().map_err(|_| invalid("frozen fit lock poisoned"))?;
        let model = fit.as_ref().map_or(&self.request.model, |(_, fit)| &fit.model);
        self.request
            .worlds
            .iter()
            .map(|world| {
                antecedent_model::compile::compile_equilibrium(model, world)
                    .map_err(CyclicArtifactError::from)
            })
            .collect()
    }
    /// Execute the retained program, fitting once per active kernel policy and reusing that fit.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        self.retention.observe_current();
        antecedent_core::numerical_admission::with_context(ctx, || {
            let mut retained = self.fit.lock().map_err(|_| invalid("frozen fit lock poisoned"))?;
            let fit = if let Some(recipe) = &self.request.fit {
                if let Some((_, fit)) =
                    retained.as_ref().filter(|(kernel, _)| *kernel == ctx.kernel_policy)
                {
                    antecedent_core::execution_attempt::record_physical(
                        antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
                        1,
                    );
                    antecedent_core::execution_attempt::record_physical(
                        antecedent_core::execution_attempt::PhysicalEvent::ModelFit,
                        0,
                    );
                    fit.retention.observe_current();
                    Some(std::sync::Arc::clone(fit))
                } else {
                    let model_bytes =
                        crate::retained_allocation::allocation_bytes(&self.request.model)
                            .map_err(invalid)?;
                    let order = u64::try_from(self.request.model.order()).map_err(invalid)?;
                    let parameters = u64::try_from(recipe.parameters.len()).map_err(invalid)?;
                    let workspace = model_bytes
                        .checked_mul(4)
                        .and_then(|bytes| {
                            bytes.checked_add(order.checked_mul(order)?.checked_mul(128)?)
                        })
                        .and_then(|bytes| bytes.checked_add(parameters.checked_mul(64)?))
                        .and_then(|bytes| bytes.checked_add(65536))
                        .ok_or_else(|| invalid("fit workspace size overflow"))?;
                    let workspace = std::sync::Arc::new(
                        ctx.acquire_admission(0, workspace)
                            .map_err(CyclicArtifactError::Admission)?,
                    );
                    let fit = antecedent_core::numerical_admission::with_reservation(
                        std::sync::Arc::clone(&workspace),
                        || {
                            fit_cyclic_likelihood(
                                &self.request.model,
                                &recipe.parameters,
                                &recipe.rows,
                                recipe.options,
                                ctx,
                            )
                        },
                    )?;
                    if !fit.converged {
                        return Err(invalid("fit did not converge"));
                    }
                    let bytes =
                        crate::retained_allocation::allocation_bytes(&fit).map_err(invalid)?;
                    let retention = workspace
                        .acquire_child(0, bytes)
                        .map_err(CyclicArtifactError::Admission)?;
                    antecedent_core::execution_attempt::record_physical(
                        antecedent_core::execution_attempt::PhysicalEvent::RetainedBytes,
                        bytes,
                    );
                    let fit = std::sync::Arc::new(RetainedCyclicFit { fit, retention });
                    *retained = Some((ctx.kernel_policy, std::sync::Arc::clone(&fit)));
                    self.fits_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Some(fit)
                }
            } else {
                antecedent_core::execution_attempt::record_physical(
                    antecedent_core::execution_attempt::PhysicalEvent::ModelFit,
                    0,
                );
                None
            };
            drop(retained);
            self.request.execute_bound(fit.as_deref().map(|fit| &fit.fit), ctx)
        })
    }
    /// Change only worlds/disturbance evaluation and numerical initial state.
    /// The model, training rows, fitting policy, and optional factual evidence stay frozen.
    pub fn retarget(
        &self,
        worlds: Vec<Vec<Clamp>>,
        noise: Vec<Vec<f64>>,
        initial: Vec<f64>,
    ) -> Result<Self, CyclicArtifactError> {
        let mut request = self.request.as_ref().clone();
        request.worlds = worlds;
        request.noise = noise;
        request.initial = initial;
        request.identity()?;
        let retention = retain_request(&request, &self.resource_context)?;
        Ok(Self {
            request: std::sync::Arc::new(request),
            fit: self.fit.clone(),
            fits_performed: std::sync::Arc::default(),
            retention: std::sync::Arc::new(retention),
            resource_context: self.resource_context.clone(),
        })
    }
    /// Export without refitting. Independent consumption still recomputes the fit.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicExecutionReport, Vec<u8>), CyclicArtifactError> {
        if artifact_id.trim().is_empty() || artifact_id.len() > 256 {
            return Err(invalid("artifact id"));
        }
        let report = self.estimate(ctx)?;
        let bytes = self.encode_execution(&report, artifact_id, ctx)?;
        Ok((report, bytes))
    }

    /// Encode an already executed report without fitting or solving again.
    ///
    /// This writer checks the source binding; independent consumption still
    /// verifies the numerical answer by replaying the encoded request.
    ///
    /// # Errors
    /// Invalid artifact id, a report from another source, cancellation or encoding failure.
    pub fn encode_execution(
        &self,
        report: &CyclicExecutionReport,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicArtifactError> {
        if artifact_id.trim().is_empty() || artifact_id.len() > 256 {
            return Err(invalid("artifact id"));
        }
        if report.identity != self.identity()? {
            return Err(invalid("cyclic execution source identity mismatch"));
        }
        encode_report(&self.request, report, artifact_id, ctx)
    }
}

/// Count a bounded logical payload without allocating its serialized representation.
/// # Errors
/// Cancellation, serialization failure, or the cyclic logical byte limit.
pub fn cyclic_serialized_size<T: Serialize>(
    value: &T,
    ctx: &ExecutionContext,
) -> Result<usize, CyclicArtifactError> {
    struct Counter<'a>(usize, &'a ExecutionContext);
    impl std::io::Write for Counter<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.1.cancellation.is_cancelled() {
                return Err(std::io::Error::other("cancelled"));
            }
            self.0 = self.0.saturating_add(bytes.len());
            if self.0 > MAX_CYCLIC_ARTIFACT_BYTES {
                return Err(std::io::Error::other("cyclic logical byte limit"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0, ctx);
    serde_json::to_writer(&mut counter, value).map_err(invalid)?;
    Ok(counter.0)
}

/// Reserve logical decoding and heap expansion before inflating a cyclic container.
/// The manifest is admitted separately before its metadata is allocated.
/// # Errors
/// Malformed framing, excessive declared logical sizes, cancellation or resources.
pub fn cyclic_decode_admission(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, CyclicArtifactError> {
    if bytes.len() < 16 || bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(invalid("cyclic container byte limit or framing"));
    }
    let manifest_len = u32::from_le_bytes(bytes[12..16].try_into().map_err(invalid)?) as usize;
    if manifest_len > 16 * 1024 * 1024 || manifest_len > bytes.len() - 16 {
        return Err(invalid("cyclic manifest byte limit"));
    }
    let metadata = ctx
        .acquire_admission(0, (manifest_len as u64) * 8 + 4096)
        .map_err(CyclicArtifactError::Admission)?;
    let reader = crate::ArtifactReader::open_seek(std::io::Cursor::new(bytes)).map_err(invalid)?;
    let logical = reader.manifest().sections.iter().try_fold(0u64, |sum, section| {
        sum.checked_add(section.uncompressed_size).ok_or_else(|| invalid("logical size overflow"))
    })?;
    if logical > MAX_CYCLIC_ARTIFACT_BYTES as u64 {
        return Err(invalid("cyclic logical byte limit"));
    }
    drop(reader);
    drop(metadata);
    ctx.acquire_admission(0, (bytes.len() as u64 + logical + manifest_len as u64) * 8 + 4096)
        .map_err(CyclicArtifactError::Admission)
}

// Covers row/vector headers, retained models and the fitted copy, even for zeros.
fn retain_request(
    request: &CyclicExecutionRequest,
    ctx: &ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, CyclicArtifactError> {
    let bytes =
        cyclic_serialized_size(request, ctx)? * 8 + request.model.order().pow(2) * 32 + 4096;
    ctx.acquire_admission(0, bytes as u64).map_err(CyclicArtifactError::Admission)
}

/// Consume in a fresh process, checking the independently retained identity and
/// recomputing the fit/answers. Integrity, semantic binding and replay are distinct.
pub fn consume_cyclic_artifact(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<CyclicExecutionReport, CyclicArtifactError> {
    consume_cyclic_artifact_with_request(bytes, expected_identity, ctx).map(|(_, report)| report)
}

/// Independently verify a system once and retain its original source declaration.
///
/// # Errors
/// The same framing, identity, numerical replay, cancellation and resource failures
/// as [`consume_cyclic_artifact`]. No prepared study is reconstructed.
pub fn consume_cyclic_artifact_with_request(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<(CyclicExecutionRequest, CyclicExecutionReport), CyclicArtifactError> {
    if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES || expected_identity.len() != 64 {
        return Err(invalid("byte limit or expected identity"));
    }
    let _retention = cyclic_decode_admission(bytes, ctx)?;
    let parts = crate::decode_container(KIND, MAX_CYCLIC_ARTIFACT_BYTES, &[SECTION], bytes)
        .map_err(|e| invalid(e.into_io("cyclic artifact layout")))?;
    let payload = parts.first().ok_or_else(|| invalid("missing section"))?;
    let record: Record = serde_json::from_slice(payload).map_err(invalid)?;
    if record.version != 1 {
        return Err(invalid("unsupported family version"));
    }
    if record.request.identity()? != expected_identity
        || record.report.identity != expected_identity
    {
        return Err(CyclicArtifactError::Mismatch);
    }
    let report = record.request.execute(ctx)?;
    if report != record.report {
        return Err(CyclicArtifactError::Mismatch);
    }
    Ok((record.request, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Equation};
    fn request() -> CyclicExecutionRequest {
        CyclicExecutionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation {
                        id: "a".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.4],
                        activation: Activation::Tanh,
                        expression: None,
                        parameters: vec![],
                    },
                    Equation {
                        id: "b".into(),
                        intercept: 0.0,
                        weights: vec![0.5, 0.0],
                        activation: Activation::Tanh,
                        expression: None,
                        parameters: vec![],
                    },
                ],
                noise_covariance: vec![1.0, 0.3, 0.3, 1.0],
            },
            fit: None,
            worlds: vec![
                vec![],
                vec![Clamp { variable: 0, value: 1.0 }],
                vec![Clamp { variable: 0, value: 2.0 }, Clamp { variable: 1, value: 3.0 }],
            ],
            noise: vec![vec![0.1, 0.2], vec![-0.1, -0.2]],
            initial: vec![0.0; 2],
            policy: SolvePolicy::default(),
            factual: Some(vec![0.3, 0.7]),
        }
    }
    #[test]
    fn system_family_bridge_shares_output_and_never_dispatches_the_system_engine() {
        use crate::cyclic_family_artifact::{
            CompiledCyclicFamilyProgram, CyclicFamilyAnswer, CyclicFamilyRequest,
            consume_cyclic_family,
        };
        use std::sync::Arc;
        let ctx = ExecutionContext::for_tests(8);
        let request = request();
        let report = Arc::new(request.execute(&ctx).unwrap());
        let family = CyclicFamilyRequest::System { request: request.clone() };
        let expected = family.export(&ctx).unwrap().1;
        let count = SYSTEM_ENGINE_EXECUTIONS.with(std::cell::Cell::get);
        let program = CompiledCyclicFamilyProgram::compile(&family, &ctx).unwrap();
        let bridged = program.bind_system_execution(Arc::clone(&report), &ctx).unwrap();
        let CyclicFamilyAnswer::System(shared) = &bridged.answer else {
            panic!("system bridge");
        };
        assert!(Arc::ptr_eq(shared, &report));
        assert_eq!(shared.identity, request.identity().unwrap());
        assert_eq!(bridged.identity, program.identity());
        assert_ne!(bridged.identity, shared.identity);
        let bytes = program.encode_execution(&bridged, &ctx).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(SYSTEM_ENGINE_EXECUTIONS.with(std::cell::Cell::get), count);
        let mut other = request;
        other.noise[0][0] += 1.0;
        let other = CompiledCyclicFamilyProgram::compile(
            &CyclicFamilyRequest::System { request: other },
            &ctx,
        )
        .unwrap();
        assert!(other.bind_system_execution(Arc::clone(&report), &ctx).is_err());
        assert_eq!(SYSTEM_ENGINE_EXECUTIONS.with(std::cell::Cell::get), count);
        assert_eq!(consume_cyclic_family(&bytes, &bridged.identity, &ctx).unwrap(), bridged);
    }

    #[test]
    fn cyclic_compressed_payload_requires_logical_memory_before_inflation() {
        let payload = vec![b'0'; 1024 * 1024];
        let bytes = crate::encode_container(
            &crate::ArtifactContainer {
                kind: KIND,
                note: "compression-budget",
                max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
                artifact_id: "budget",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[crate::OutSection {
                id: SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: &payload,
            }],
        )
        .unwrap();
        assert!(bytes.len() < payload.len() / 8);
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1024 * 1024);
        let ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
        assert!(cyclic_decode_admission(&bytes, &ctx).is_err());
        assert_eq!(pool.available().bytes(), 1024 * 1024);
        let lease = cyclic_decode_admission(&bytes, &ExecutionContext::for_tests(0)).unwrap();
        drop(lease);
        let cancelled = ExecutionContext::for_tests(0);
        cancelled.cancellation.cancel();
        assert!(cyclic_serialized_size(&payload, &cancelled).is_err());
    }
    #[test]
    fn retained_cyclic_products_hold_and_release_shared_admission() {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(7).with_admission(pool.clone());
        let program = PreparedCyclicExecution::prepare(request(), &ctx).unwrap();
        let held = pool.available().bytes();
        assert!(held < 1_000_000);
        let shared = program.clone();
        assert_eq!(pool.available().bytes(), held);
        let retargeted = program.retarget(vec![vec![]], vec![vec![0.0; 2]], vec![0.0; 2]).unwrap();
        assert!(pool.available().bytes() < held);
        drop(retargeted);
        assert_eq!(pool.available().bytes(), held);
        drop(program);
        assert_eq!(pool.available().bytes(), held);
        shared.estimate(&ctx).unwrap();
        ctx.cancellation.cancel();
        assert!(shared.estimate(&ctx).is_err());
        drop(shared);
        assert_eq!(pool.available().bytes(), 1_000_000);
    }
    #[test]
    fn cyclic_artifact_replays_worlds_and_counterfactuals_and_rejects_wrong_identity() {
        let r = request();
        let ctx = ExecutionContext::for_tests(0);
        let (report, bytes) = r.export("independent", &ctx).unwrap();
        let loaded =
            consume_cyclic_artifact(&bytes, &report.identity, &ExecutionContext::for_tests(123))
                .unwrap();
        assert_eq!(loaded, report);
        assert_eq!(loaded.worlds[2][0].values, vec![2.0, 3.0]);
        assert!(loaded.counterfactuals.is_some());
        assert_eq!(loaded.claim, "specified_or_fitted_model_conditional");
        let mut changed = r.clone();
        changed.model.equations[0].intercept = 0.01;
        assert!(matches!(
            consume_cyclic_artifact(&bytes, &changed.identity().unwrap(), &ctx),
            Err(CyclicArtifactError::Mismatch)
        ));
        let mut corrupt = bytes;
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        assert!(consume_cyclic_artifact(&corrupt, &report.identity, &ctx).is_err());
    }
    #[test]
    fn cyclic_artifact_fit_is_recomputed_and_resealed_edit_does_not_pass() {
        let mut r = request();
        let rows = vec![
            RegimeRow { values: vec![1.0, 0.6], clamps: vec![Clamp { variable: 0, value: 1.0 }] },
            RegimeRow {
                values: vec![-1.0, -0.6],
                clamps: vec![Clamp { variable: 0, value: -1.0 }],
            },
        ];
        r.fit = Some(CyclicFitRequest {
            parameters: vec![CyclicParameter::Weight { equation: 1, parent: 0 }],
            rows,
            options: CyclicFitOptions::default(),
        });
        let ctx = ExecutionContext::for_tests(0);
        let (report, bytes) = r.export("fit", &ctx).unwrap();
        assert!(report.fit.as_ref().unwrap().converged);
        assert_eq!(consume_cyclic_artifact(&bytes, &report.identity, &ctx).unwrap(), report);
        let mut record = Record { version: 1, request: r, report: report.clone() };
        record.report.worlds[0][0].values[0] += 1.0;
        let data = serde_json::to_vec(&record).unwrap();
        let resealed = crate::encode_container(
            &crate::ArtifactContainer {
                kind: KIND,
                note: "tampered",
                max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
                artifact_id: "fit",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[crate::OutSection { id: SECTION, content_type: crate::CONTENT_BYTES, bytes: &data }],
        )
        .unwrap();
        assert!(matches!(
            consume_cyclic_artifact(&resealed, &report.identity, &ctx),
            Err(CyclicArtifactError::Mismatch)
        ));
    }
    #[test]
    fn cyclic_output_reservation_survives_prepared_owner_and_cancellation() {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(3).with_admission(pool.clone());
        let prepared = PreparedCyclicExecution::prepare(request(), &ctx).unwrap();
        let report = prepared.estimate(&ctx).unwrap();
        drop(prepared);
        assert!(report.retained_admitted_bytes() > 0);
        assert_eq!(pool.byte_budget() - pool.available().bytes(), report.retained_admitted_bytes());
        ctx.cancellation.cancel();
        assert!(pool.available().bytes() < pool.byte_budget());
        drop(report);
        assert_eq!(pool.available().bytes(), pool.byte_budget());
    }
}

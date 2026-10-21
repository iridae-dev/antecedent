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

/// Serialization, integrity, identity, scientific or execution failure.
#[derive(Debug, Error)]
pub enum CyclicArtifactError {
    /// Family size, shape or version limit.
    #[error("cyclic artifact invalid: {0}")]
    Invalid(String),
    /// Model or statistical computation failed.
    #[error(transparent)]
    Execution(#[from] antecedent_model::nonlinear_equilibrium::EquilibriumError),
    /// Expected request does not match, or replay differs from stored results.
    #[error("cyclic artifact identity/replay mismatch")]
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

    fn execute_bound(
        &self,
        fit: Option<CyclicFit>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        let identity = self.identity()?;
        let n = self.model.order();
        let retained = self.worlds.len() * self.noise.len() * (n * n * 8 + n * 128 + 256);
        let _retention =
            ctx.acquire_admission(0, u64::try_from(retained).map_err(invalid)?).map_err(invalid)?;
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
        Ok(CyclicExecutionReport {
            identity,
            fit,
            worlds,
            counterfactuals,
            claim: "specified_or_fitted_model_conditional".into(),
            calibration: "unmeasured".into(),
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
        let container = encode_report(self, &report, artifact_id)?;
        Ok((report, container))
    }
}

fn encode_report(
    request: &CyclicExecutionRequest,
    report: &CyclicExecutionReport,
    artifact_id: &str,
) -> Result<Vec<u8>, CyclicArtifactError> {
    #[derive(Serialize)]
    struct BorrowedRecord<'a> {
        version: u32,
        request: &'a CyclicExecutionRequest,
        report: &'a CyclicExecutionReport,
    }
    let record = BorrowedRecord { version: 1, request, report };
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
    Ok(container)
}

/// Immutable compiled equilibrium product, sharing its fit across retargeted worlds.
#[derive(Clone, Debug)]
pub struct PreparedCyclicExecution {
    request: std::sync::Arc<CyclicExecutionRequest>,
    fit: Option<std::sync::Arc<CyclicFit>>,
    fits_performed: usize,
    _retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
    resource_context: ExecutionContext,
}
impl PreparedCyclicExecution {
    /// Borrow the frozen scientific request without copying observations.
    #[must_use]
    pub fn request(&self) -> &CyclicExecutionRequest {
        &self.request
    }

    /// Validate the complete request and perform its optional statistical fit once.
    pub fn prepare(
        request: CyclicExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        request.check_limits()?;
        let retention = retain_request(&request, ctx)?;
        request.identity()?;
        let fit = request
            .fit
            .as_ref()
            .map(|f| fit_cyclic_likelihood(&request.model, &f.parameters, &f.rows, f.options, ctx))
            .transpose()?;
        if fit.as_ref().is_some_and(|f| !f.converged) {
            return Err(invalid("fit did not converge"));
        }
        let fits_performed = usize::from(fit.is_some());
        Ok(Self {
            request: std::sync::Arc::new(request),
            fit: fit.map(std::sync::Arc::new),
            fits_performed,
            _retention: std::sync::Arc::new(retention),
            resource_context: ctx.clone(),
        })
    }
    /// Number of causal fits performed while constructing this particular handle.
    #[must_use]
    pub fn fits_performed(&self) -> usize {
        self.fits_performed
    }
    /// Request identity, including source data, target worlds and numerical policy.
    pub fn identity(&self) -> Result<String, CyclicArtifactError> {
        self.request.identity()
    }
    /// Compile/inspect the actual post-intervention SCC solve schedules.
    pub fn blocks(&self) -> Result<Vec<Vec<Vec<usize>>>, CyclicArtifactError> {
        let model = self.fit.as_ref().map_or(&self.request.model, |f| &f.model);
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
        let model = self.fit.as_ref().map_or(&self.request.model, |fit| &fit.model);
        self.request
            .worlds
            .iter()
            .map(|world| {
                antecedent_model::compile::compile_equilibrium(model, world)
                    .map_err(CyclicArtifactError::from)
            })
            .collect()
    }
    /// Execute the retained program using its frozen fitted model; never refit.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        self.request.execute_bound(self.fit.as_ref().map(|fit| fit.as_ref().clone()), ctx)
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
            fits_performed: 0,
            _retention: std::sync::Arc::new(retention),
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
        let bytes = encode_report(&self.request, &report, artifact_id)?;
        Ok((report, bytes))
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
    let metadata = ctx.acquire_admission(0, (manifest_len as u64) * 8 + 4096).map_err(invalid)?;
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
        .map_err(invalid)
}

// Covers row/vector headers, retained models and the fitted copy, even for zeros.
fn retain_request(
    request: &CyclicExecutionRequest,
    ctx: &ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, CyclicArtifactError> {
    let bytes =
        cyclic_serialized_size(request, ctx)? * 8 + request.model.order().pow(2) * 32 + 4096;
    ctx.acquire_admission(0, bytes as u64).map_err(invalid)
}

/// Consume in a fresh process, checking the independently retained identity and
/// recomputing the fit/answers. Integrity, semantic binding and replay are distinct.
pub fn consume_cyclic_artifact(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<CyclicExecutionReport, CyclicArtifactError> {
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
    Ok(report)
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
}

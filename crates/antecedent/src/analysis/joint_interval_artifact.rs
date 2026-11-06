//! Separately versioned replay of the existing endpoint bootstrap. The embedded
//! version-3 artifact continues to describe an assumption range, never an interval.
use super::{JointSensitivityArtifactWire, JointSensitivityConsumeLimits};
use antecedent_core::execution::{AdmissionLease, RngFactory};
use antecedent_core::{CanonicalWriter, ExecutionContext, reason_code};
use antecedent_io::z_transport_artifact::ZTransportArtifactWire;
use antecedent_io::{IoError, decode_with_version, to_cbor};
use antecedent_io_core::contract_section::CalibrationSlotWire;
use antecedent_validate::{JointDeviationSpec, JointSensitivityInterval};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Interval-artifact format, independent of the existing range-artifact version.
pub const JOINT_INTERVAL_ARTIFACT_VERSION: u32 = 1;
const MAGIC: &[u8] = b"ANTE-JOINT-INTERVAL-1";
const PROCEDURE: &str = "iid_count_endpoint_percentile_v1";
const RNG: &str = "antecedent_rng_streams_v1";
const DOMAIN: &str = "antecedent.joint_sensitivity.interval.v1";

fn refused(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: if detail == "joint_sensitivity.consumer_limits" {
            reason_code!("route_not_supported")
        } else {
            reason_code!("transport_not_certified")
        },
        message: format!("{detail}: {message}"),
    }
}

fn admission_error(error: antecedent_core::execution::AdmissionError) -> IoError {
    IoError::Refused {
        code: reason_code!("transport_budget_cancel"),
        message: format!("joint_sensitivity.budget: {error}"),
    }
}

/// Numerical interval output. The full replicate vectors make replay and failures explicit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointIntervalNumerics {
    /// Conservative lower and upper endpoint quantiles.
    pub endpoints: [f64; 2],
    /// Percentile interval of the zero-deviation baseline.
    pub pointwise_at_zero: [f64; 2],
    /// Requested nominal level.
    pub level: f64,
    /// Original engine method identifier.
    pub method: String,
    /// Conditional coverage target; never general simultaneous coverage.
    pub coverage_target: String,
    /// Requested, successful and failed replicate counts.
    pub counts: [u32; 3],
    /// Replicate baselines, in draw order.
    pub baselines: Vec<f64>,
    /// Replicate lower range endpoints, in draw order.
    pub minima: Vec<f64>,
    /// Replicate upper range endpoints, in draw order.
    pub maxima: Vec<f64>,
}
impl From<&JointSensitivityInterval> for JointIntervalNumerics {
    fn from(value: &JointSensitivityInterval) -> Self {
        Self {
            endpoints: [value.lower, value.upper],
            pointwise_at_zero: [value.pointwise_at_zero.0, value.pointwise_at_zero.1],
            level: value.level,
            method: value.method.into(),
            coverage_target: value.coverage_target.into(),
            counts: [value.replicates_requested, value.replicates_ok, value.replicates_failed],
            baselines: value.replicate_baselines.clone(),
            minima: value.replicate_minima.clone(),
            maxima: value.replicate_maxima.clone(),
        }
    }
}

/// Replayable endpoint-bootstrap interval, with an embedded unchanged range artifact.
/// Stored calibration is a historical producer annotation, not acceptance authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSensitivityIntervalArtifact {
    /// This interval format's version.
    pub version: u32,
    /// Existing checked v3 range, binding baseline laws, deviation box and limits.
    pub range: JointSensitivityArtifactWire,
    /// Independently versioned procedure.
    pub procedure: String,
    /// Seed derivation/stream identity.
    pub rng: String,
    /// Master seed for deterministic bootstrap replay.
    pub seed: u64,
    /// Portable-optimized, architecture-SIMD and force-scalar kernel policy flags.
    pub kernels: [bool; 3],
    /// Recomputed numerical output.
    pub numerical: JointIntervalNumerics,
    /// Historical annotation supplied at creation. Consumers resolve current standing separately.
    pub recorded_calibration: CalibrationSlotWire,
    /// Content identity, including range, procedure, seed and output.
    pub digest: String,
}

/// Hard consumer maxima, checked before replay. Stored settings cannot raise them.
#[derive(Clone, Copy, Debug)]
pub struct JointIntervalConsumeLimits {
    /// Limits of the embedded range and baseline.
    pub range: JointSensitivityConsumeLimits,
    /// Maximum requested bootstrap replicates, at most the engine's 2000.
    pub max_replicates: u32,
    /// Maximum encoded interval bytes accepted before decoding.
    pub max_bytes: usize,
}
impl Default for JointIntervalConsumeLimits {
    fn default() -> Self {
        Self {
            range: JointSensitivityConsumeLimits::default(),
            max_replicates: 2000,
            max_bytes: 67_108_864,
        }
    }
}

/// Independently replayed interval with current calibration, kept distinct from
/// the historical annotation embedded in the wire.
#[derive(Clone, Debug)]
pub struct JointIntervalConsumption {
    /// Verified stored artifact.
    pub artifact: JointSensitivityIntervalArtifact,
    /// Current calibration of the actual replayed construction, never trusted from bytes.
    pub calibration: CalibrationSlotWire,
    _retention: Arc<AdmissionLease>,
}

impl JointSensitivityIntervalArtifact {
    /// Produce an interval from a replayed baseline and the existing endpoint procedure.
    /// No new sampling estimator or coverage claim is introduced.
    ///
    /// # Errors
    /// Baseline/range refusal, unsupported bootstrap settings, cancellation or resource limit.
    pub fn checked(
        baseline: Vec<u8>,
        spec: &JointDeviationSpec,
        seed: u64,
        replicates: u32,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if !(2..=2000).contains(&replicates) || !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(IoError::Refused {code:reason_code!("route_not_supported"),message:"joint_sensitivity.bounds_exceeded: interval requires 2..=2000 replicates and a level strictly inside (0,1)".into()});
        }
        let range = JointSensitivityArtifactWire::checked(baseline, spec, ctx)?;
        let mut seeded = ctx.clone();
        seeded.rng = RngFactory::from_seed(seed);
        let interval = compute_interval(
            &range,
            replicates,
            level,
            JointSensitivityConsumeLimits::default(),
            &seeded,
        )?;
        let mut artifact = Self {
            version: JOINT_INTERVAL_ARTIFACT_VERSION,
            range,
            procedure: PROCEDURE.into(),
            rng: RNG.into(),
            seed,
            kernels: [
                ctx.kernel_policy.allow_portable_optimized,
                ctx.kernel_policy.allow_arch_simd,
                ctx.kernel_policy.force_scalar,
            ],
            numerical: JointIntervalNumerics::from(&interval),
            recorded_calibration: interval.calibration,
            digest: String::new(),
        };
        artifact.digest = artifact.computed_digest()?;
        Ok(artifact)
    }
    /// Content identity of the complete scientific request and numerical result.
    /// # Errors
    /// CBOR serialization failure.
    pub fn computed_digest(&self) -> Result<String, IoError> {
        Ok(CanonicalWriter::digest_bytes_hex(
            DOMAIN,
            &to_cbor(&(
                self.version,
                &self.range,
                &self.procedure,
                &self.rng,
                self.seed,
                self.kernels,
                &self.numerical,
                &self.recorded_calibration,
            ))?,
        ))
    }
    /// Serialize this wire; independent consumption is required before accepting its numbers.
    /// # Errors
    /// Serialization failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(self)?;
        let mut bytes = Vec::with_capacity(MAGIC.len() + body.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }
    /// Replay under default hard maxima, deriving current calibration independently.
    /// # Errors
    /// As `consume_with_limits`.
    pub fn consume(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<JointIntervalConsumption, IoError> {
        Self::consume_with_limits(bytes, JointIntervalConsumeLimits::default(), ctx)
    }
    /// Validate identity and versions, replay the embedded range and then rerun the
    /// original endpoint bootstrap with the stored seed and kernel policy. Changed
    /// calibration records affect the returned current annotation, not historical bytes.
    /// # Errors
    /// Excess limits, malformed/stale identity, changed numbers, unsupported procedure or cancellation.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: JointIntervalConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<JointIntervalConsumption, IoError> {
        if bytes.len() > limits.max_bytes {
            return Err(refused(
                "joint_sensitivity.consumer_limits",
                "interval exceeds maximum encoded bytes",
            ));
        }
        let retention =
            ctx.acquire_admission_product(0, bytes.len() as u64, 3).map_err(admission_error)?;
        let body = bytes.strip_prefix(MAGIC).ok_or_else(|| {
            refused(
                "joint_sensitivity.replay_mismatch",
                "not a separately framed interval artifact",
            )
        })?;
        let artifact: Self = decode_with_version(body, |version| {
            if version == JOINT_INTERVAL_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        })?;
        if artifact.procedure != PROCEDURE || artifact.rng != RNG {
            return Err(refused(
                "joint_sensitivity.replay_mismatch",
                "unsupported interval procedure or RNG version",
            ));
        }
        if artifact.numerical.counts[0] > limits.max_replicates
            || artifact.numerical.counts[0] > 2000
        {
            return Err(refused(
                "joint_sensitivity.consumer_limits",
                "stored bootstrap count exceeds consumer maximum",
            ));
        }
        if artifact.computed_digest()? != artifact.digest {
            return Err(refused(
                "joint_sensitivity.premises_mismatch",
                "interval content digest differs",
            ));
        }
        let mut seeded = ctx.clone();
        seeded.rng = RngFactory::from_seed(artifact.seed);
        seeded.kernel_policy.allow_portable_optimized = artifact.kernels[0];
        seeded.kernel_policy.allow_arch_simd = artifact.kernels[1];
        seeded.kernel_policy.force_scalar = artifact.kernels[2];
        let range = JointSensitivityArtifactWire::consume_with_limits(
            &artifact.range.export()?,
            limits.range,
            &seeded,
        )?;
        let interval = compute_interval(
            &range,
            artifact.numerical.counts[0],
            artifact.numerical.level,
            limits.range,
            &seeded,
        )?;
        if to_cbor(&JointIntervalNumerics::from(&interval))? != to_cbor(&artifact.numerical)? {
            return Err(refused(
                "joint_sensitivity.replay_mismatch",
                "endpoint bootstrap numbers or replicate counts differ",
            ));
        }
        Ok(JointIntervalConsumption {
            artifact,
            calibration: interval.calibration,
            _retention: Arc::new(retention),
        })
    }
}

fn compute_interval(
    range: &JointSensitivityArtifactWire,
    replicates: u32,
    level: f64,
    limits: JointSensitivityConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<JointSensitivityInterval, IoError> {
    let baseline = ZTransportArtifactWire::consume_with_limits(
        &range.baseline_artifact,
        limits.baseline,
        ctx,
    )?;
    let spec = range.replay_spec()?;
    // The interval itself has no frontier search. Keep the caller's shared
    // admission pool; changing its cap here would create an independent pool.
    // The original drawer retains all count-law replicas at once, so reserve
    // a conservative envelope before it allocates them (plus endpoint vectors).
    let source_bytes =
        u64::try_from(range.baseline_artifact.len()).map_err(|_| IoError::TooLarge)?;
    let bytes = source_bytes
        .checked_mul(u64::from(replicates))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(u64::from(replicates) * 3 * 8))
        .ok_or(IoError::TooLarge)?;
    let _workspace = ctx.acquire_admission(0, bytes).map_err(admission_error)?;
    antecedent_validate::joint_sensitivity_bootstrap_interval_internal(
        &baseline.diagram,
        &baseline.functional,
        &baseline.data,
        &spec,
        replicates,
        level,
        ctx,
    )
    .map_err(|e| IoError::Refused { code: e.reason_code(), message: e.to_string() })?
    .map_err(|e| IoError::Refused {
        code: reason_code!("transport_numerical_failure"),
        message: e.into(),
    })
}

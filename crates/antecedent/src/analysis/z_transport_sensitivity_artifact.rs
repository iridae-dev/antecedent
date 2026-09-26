//! Portable sensitivity analysis tied to a checked z-transport point artifact.
//!
//! Format version 2. The embedded baseline is replayed through the point
//! artifact's own consumer, so a forged baseline point with a valid digest is
//! refused before any sensitivity range is compared.
use antecedent_core::{ExecutionContext, RegimeId};
use antecedent_io::z_transport_artifact::{ZTransportArtifactWire, ZTransportConsumeLimits};
use antecedent_io::{IoError, from_cbor, to_cbor};
use antecedent_validate::z_transport_mechanism_sensitivity;
use serde::{Deserialize, Serialize};

/// The sensitivity artifact format this reader writes and accepts.
pub const Z_TRANSPORT_SENSITIVITY_ARTIFACT_VERSION: u32 = 2;

/// Versioned replay artifact for discrete outcome-kernel contamination.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportSensitivityArtifactWire {
    /// Independent wire version.
    pub version: u32,
    /// Complete checked point artifact that supplies the baseline premises and laws.
    pub baseline_artifact: Vec<u8>,
    /// Digest binding this result to the exact baseline artifact bytes.
    pub baseline_artifact_digest: String,
    /// The baseline artifact's semantic premises digest, stable across
    /// re-encodings of the same premises.
    pub baseline_premises_digest: String,
    /// Provider snapshots retained by the baseline artifact.
    pub provider_snapshots: Vec<String>,
    /// Source regime used by the checked formula.
    pub source_regime: u32,
    /// Declared maximum contamination fraction.
    pub max_fraction: f64,
    /// Optional tipping threshold.
    pub decision_threshold: Option<f64>,
    /// Recomputed baseline contrast.
    pub baseline: f64,
    /// Exact extrema over the declared contamination domain.
    pub assumption_range: [f64; 2],
    /// Minimum contamination fraction that reaches the threshold.
    pub tipping_fraction: Option<f64>,
    /// Vertex outcome index for each minimizing source stratum.
    pub minimizing_outcome_by_stratum: Vec<usize>,
    /// Vertex outcome index for each maximizing source stratum.
    pub maximizing_outcome_by_stratum: Vec<usize>,
    /// Interpretation statement.
    pub interval_interpretation: String,
    /// Optimization receipt method.
    pub method: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl ZTransportSensitivityArtifactWire {
    /// Recompute a typed sensitivity result and bind it to an exported baseline
    /// artifact, replaying the baseline point under the default consumer limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the baseline artifact does not replay or the
    /// sensitivity optimization fails.
    pub fn checked(
        baseline_artifact: Vec<u8>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        Self::checked_with_limits(
            baseline_artifact,
            max_fraction,
            decision_threshold,
            ZTransportConsumeLimits::default(),
            ctx,
        )
    }

    /// [`Self::checked`] under explicit consumer limits for the baseline replay.
    ///
    /// # Errors
    ///
    /// As [`Self::checked`].
    pub fn checked_with_limits(
        baseline_artifact: Vec<u8>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
        limits: ZTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let consumed =
            ZTransportArtifactWire::consume_with_limits(&baseline_artifact, limits, ctx)?;
        let result = z_transport_mechanism_sensitivity(
            &consumed.diagram,
            &consumed.functional,
            &consumed.data,
            max_fraction,
            decision_threshold,
            ctx,
        )
        .map_err(|error| match error {
            antecedent_validate::ZTransportSensitivityError::Cancelled => IoError::Refused {
                code: antecedent_core::reason_code!("transport_budget_cancel"),
                message: error.to_string(),
            },
            other => IoError::Convert(other.to_string()),
        })?;
        let digest = digest_baseline(&baseline_artifact);
        Ok(Self {
            version: Z_TRANSPORT_SENSITIVITY_ARTIFACT_VERSION,
            baseline_artifact,
            baseline_artifact_digest: digest,
            baseline_premises_digest: consumed.wire.premises_digest,
            provider_snapshots: consumed
                .data
                .laws()
                .iter()
                .map(|law| law.snapshot_identity().to_owned())
                .collect(),
            source_regime: result.source_regime.raw(),
            max_fraction,
            decision_threshold,
            baseline: result.response.baseline,
            assumption_range: [result.response.minimum, result.response.maximum],
            tipping_fraction: result.response.tipping_fraction,
            minimizing_outcome_by_stratum: result.response.receipt.minimizing_outcome_by_stratum,
            maximizing_outcome_by_stratum: result.response.receipt.maximizing_outcome_by_stratum,
            interval_interpretation: result.response.interval_interpretation.into(),
            method: result.response.receipt.method.into(),
        })
    }

    /// Export as a portable CBOR item.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        to_cbor(self)
    }

    /// Validate bindings and independently recompute the response range under
    /// the default consumer limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire is malformed, stale, or its result differs on replay.
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<Self, IoError> {
        Self::consume_with_limits(bytes, ZTransportConsumeLimits::default(), ctx)
    }

    /// [`Self::consume`] under explicit consumer limits for the baseline replay.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedVersion`] for another version, a baseline digest
    /// or premises digest that does not match, a baseline that does not replay,
    /// or a recomputed result that differs from the stored one.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: ZTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let peek: VersionPeek = from_cbor(bytes)?;
        if peek.version != Z_TRANSPORT_SENSITIVITY_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = from_cbor(bytes)?;
        if digest_baseline(&wire.baseline_artifact) != wire.baseline_artifact_digest {
            return Err(IoError::Convert(
                "z-transport sensitivity baseline digest mismatch".into(),
            ));
        }
        let recomputed = Self::checked_with_limits(
            wire.baseline_artifact.clone(),
            wire.max_fraction,
            wire.decision_threshold,
            limits,
            ctx,
        )?;
        if recomputed.baseline_premises_digest != wire.baseline_premises_digest {
            return Err(IoError::Convert(
                "z-transport sensitivity baseline premises digest mismatch".into(),
            ));
        }
        if !wire.replays(&recomputed) {
            return Err(IoError::Convert(
                "z-transport sensitivity result or provider binding mismatch".into(),
            ));
        }
        Ok(recomputed)
    }

    /// Whether a recomputed artifact replays this stored one.
    ///
    /// Equality policy: the replay is the same deterministic optimization on
    /// the same laws, so every floating-point field is compared bit for bit and
    /// a non-finite stored value fails closed.
    fn replays(&self, recomputed: &Self) -> bool {
        let same_f64 =
            |a: f64, b: f64| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits();
        let same_opt = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => same_f64(a, b),
            _ => false,
        };
        self.provider_snapshots == recomputed.provider_snapshots
            && self.source_regime == recomputed.source_regime
            && same_f64(self.max_fraction, recomputed.max_fraction)
            && same_opt(self.decision_threshold, recomputed.decision_threshold)
            && same_f64(self.baseline, recomputed.baseline)
            && same_f64(self.assumption_range[0], recomputed.assumption_range[0])
            && same_f64(self.assumption_range[1], recomputed.assumption_range[1])
            && same_opt(self.tipping_fraction, recomputed.tipping_fraction)
            && self.minimizing_outcome_by_stratum == recomputed.minimizing_outcome_by_stratum
            && self.maximizing_outcome_by_stratum == recomputed.maximizing_outcome_by_stratum
            && self.interval_interpretation == recomputed.interval_interpretation
            && self.method == recomputed.method
    }

    /// Source regime named by this replay artifact.
    #[must_use]
    pub const fn regime(&self) -> RegimeId {
        RegimeId::from_raw(self.source_regime)
    }
}

/// Hex digest of the baseline artifact bytes.
#[must_use]
pub fn digest_baseline(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = antecedent_io::identity::payload_digest("z_transport_sensitivity_baseline", bytes);
    let mut hexadecimal = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hexadecimal, "{byte:02x}")
            .expect("writing hexadecimal bytes to a String cannot fail");
    }
    hexadecimal
}

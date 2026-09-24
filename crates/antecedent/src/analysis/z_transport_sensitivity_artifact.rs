//! Portable sensitivity analysis tied to a checked z-transport point artifact.
use antecedent_core::{ExecutionContext, RegimeId};
use antecedent_io::{IoError, from_cbor, to_cbor};
use antecedent_validate::z_transport_mechanism_sensitivity;
use serde::{Deserialize, Serialize};

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

impl ZTransportSensitivityArtifactWire {
    /// Recompute a typed sensitivity result and bind it to an exported baseline artifact.
    pub fn checked(
        baseline_artifact: Vec<u8>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let (diagram, functional, data, _request, _limits, _point) =
            antecedent_io::z_transport_artifact::ZTransportArtifactWire::reconstruct(
                &baseline_artifact,
            )?;
        let result = z_transport_mechanism_sensitivity(
            &diagram,
            &functional,
            &data,
            max_fraction,
            decision_threshold,
            ctx,
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
        let digest = digest_baseline(&baseline_artifact);
        Ok(Self {
            version: 1,
            baseline_artifact,
            baseline_artifact_digest: digest,
            provider_snapshots: data
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
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        to_cbor(self)
    }

    /// Validate bindings and independently recompute the response range.
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<Self, IoError> {
        let wire: Self = from_cbor(bytes)?;
        if wire.version != 1
            || digest_baseline(&wire.baseline_artifact) != wire.baseline_artifact_digest
        {
            return Err(IoError::Convert(
                "z-transport sensitivity baseline digest mismatch".into(),
            ));
        }
        let recomputed = Self::checked(
            wire.baseline_artifact.clone(),
            wire.max_fraction,
            wire.decision_threshold,
            ctx,
        )?;
        if wire != recomputed {
            return Err(IoError::Convert(
                "z-transport sensitivity result or provider binding mismatch".into(),
            ));
        }
        Ok(recomputed)
    }

    /// Source regime named by this replay artifact.
    #[must_use]
    pub const fn regime(&self) -> RegimeId {
        RegimeId::from_raw(self.source_regime)
    }
}

fn digest_baseline(bytes: &[u8]) -> String {
    antecedent_io::identity::payload_digest("z_transport_sensitivity_baseline", bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

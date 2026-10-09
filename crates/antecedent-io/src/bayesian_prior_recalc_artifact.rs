//! Bounded source-data envelope for independent native Bayesian prior replay.
//! This envelope alone grants no posterior authority: the facade re-executes the
//! checked source Study and compares its original posterior artifact.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::IoError;
use serde::{Deserialize, Serialize};
/// Checked source envelope format marker.
pub const BOUND_PRIOR_MAGIC: &[u8] = b"ANTBOUNDPRIOR1";
/// Maximum source envelope bytes before decoding or replay.
pub const BOUND_PRIOR_MAX_BYTES: usize = 16_000_000;
/// Bounded source Gaussian prior declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BoundPriorSpecWire {
    /// Finite native coefficient prior.
    Gaussian {
        /// Means in original coefficient order.
        mean: Vec<f64>,
        /// Prior scale variances.
        variance: Vec<f64>,
    },
    /// Fixed source residual variance.
    KnownVariance(f64),
    /// Original source inverse-gamma variance prior.
    InvGamma {
        /// Shape.
        shape: f64,
        /// Scale.
        scale: f64,
    },
    /// Original bounded coefficient correlation matrix.
    Correlation {
        /// Dimension.
        dim: usize,
        /// Row-major matrix.
        matrix: Vec<f64>,
    },
}
/// Original source inputs and native issued posterior used for fresh-process replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundBayesianPriorWire {
    /// Original graph-order IEEE-754 value bits, preserving missing payloads exactly.
    pub columns: Vec<(String, Vec<u64>)>,
    /// Original directed graph edges.
    pub edges: Vec<(u32, u32)>,
    /// Original treatment id.
    pub treatment: u32,
    /// Original outcome id.
    pub outcome: u32,
    /// Actual source sampling count.
    pub draws: usize,
    /// Original explicit-count declaration.
    pub draws_explicit: bool,
    /// Original native source stream seed.
    pub seed: u64,
    /// Original isotropic prior scale.
    pub prior_scale: f64,
    /// Bounded explicit source coefficient/variance priors, when present.
    pub prior: Option<Vec<BoundPriorSpecWire>>,
    /// Independently recomputable source physical content digest.
    pub source_snapshot: [u8; 32],
    /// Original native posterior artifact, not caller-projected draw JSON.
    pub posterior: Vec<u8>,
}
impl BoundBayesianPriorWire {
    /// Encode a bounded original source declaration. The consumer still replays it.
    /// # Errors
    /// Oversize or invalid encoding.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        let mut bytes = BOUND_PRIOR_MAGIC.to_vec();
        ciborium::into_writer(self, &mut bytes).map_err(|e| IoError::Convert(e.to_string()))?;
        if bytes.len() > BOUND_PRIOR_MAX_BYTES {
            return Err(IoError::Convert("bound prior envelope exceeds byte limit".into()));
        }
        Ok(bytes)
    }
    /// Decode under a byte ceiling; no scientific authority is granted by decoding.
    /// # Errors
    /// Missing marker, corrupt body, or oversize input.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() > BOUND_PRIOR_MAX_BYTES || !bytes.starts_with(BOUND_PRIOR_MAGIC) {
            return Err(IoError::Convert("unverified Bayesian prior source envelope".into()));
        }
        ciborium::from_reader(&bytes[BOUND_PRIOR_MAGIC.len()..])
            .map_err(|e| IoError::Convert(e.to_string()))
    }
}

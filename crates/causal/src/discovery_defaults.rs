//! Shared discovery constraint defaults and CI resolution.
//!
//! Single source for `TemporalConstraints` / `DiscoveryConstraints` used by
//! [`crate::analysis`] and the Python bindings — do not duplicate these blocks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::Lag;
use causal_discovery::{DiscoveryConstraints, MultiDatasetConstraints, TemporalConstraints};
use causal_stats::{ConditionalIndependence, WeightedPartialCorrelation, ci_from_name};

use crate::error::AnalysisError;

/// Default max conditioning-set size for PCMCI-family runs in the facade / Python path.
pub const DEFAULT_MAX_COND_SIZE: usize = 2;

/// Default significance level when callers omit an override.
pub const DEFAULT_ALPHA: f64 = 0.05;

/// Default minimum regime length for the RPCMCI Python/facade path.
pub const DEFAULT_RPCMCI_MIN_REGIME_LEN: usize = 40;

/// Lagged-only PCMCI constraints (`min_lag = 1`).
#[must_use]
pub fn pcmci_constraints(max_lag: u32, alpha: f64) -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::from_raw(1),
        },
        alpha,
        max_cond_size: DEFAULT_MAX_COND_SIZE,
        ..DiscoveryConstraints::default()
    }
}

/// Contemporaneous-allowed constraints (`min_lag = 0`) for PCMCI+, LPCMCI, RPCMCI.
#[must_use]
pub fn contemporaneous_constraints(max_lag: u32, alpha: f64) -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::CONTEMPORANEOUS,
        },
        alpha,
        max_cond_size: DEFAULT_MAX_COND_SIZE,
        ..DiscoveryConstraints::default()
    }
}

/// Static PC constraints (no lag search; contemp-only temporal fields).
#[must_use]
pub fn static_pc_constraints(alpha: f64, max_cond_size: usize) -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
        },
        alpha,
        max_cond_size,
        ..DiscoveryConstraints::default()
    }
}

/// J-PCMCI+ constraints: contemporaneous search plus multi-dataset / context settings.
#[must_use]
pub fn jpcmci_constraints(
    max_lag: u32,
    alpha: f64,
    multi_dataset: MultiDatasetConstraints,
) -> DiscoveryConstraints {
    let mut c = contemporaneous_constraints(max_lag, alpha);
    c.multi_dataset = multi_dataset;
    c
}

/// Resolve a CI test by name, with optional observation weights for `weighted_parcorr`.
///
/// # Errors
///
/// Unknown CI name, or weights supplied for a non-weighted test / missing when required.
pub fn resolve_ci(
    name: &str,
    weights: Option<Vec<f64>>,
) -> Result<Arc<dyn ConditionalIndependence + Send + Sync>, AnalysisError> {
    let key = name.trim().to_ascii_lowercase();
    if matches!(key.as_str(), "weighted_parcorr" | "weighted_partial_corr") {
        let Some(w) = weights else {
            return Err(AnalysisError::Compile {
                message: "weights required when ci='weighted_parcorr'".into(),
            });
        };
        return Ok(Arc::new(WeightedPartialCorrelation::new(w)));
    }
    if weights.is_some() {
        return Err(AnalysisError::Compile {
            message: "observation weights are only supported when ci='weighted_parcorr'".into(),
        });
    }
    ci_from_name(name).map_err(|e| AnalysisError::Compile { message: e.to_string() })
}

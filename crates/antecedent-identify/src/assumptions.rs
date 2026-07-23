//! Shared assumption-record helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};

/// Declared Causal Markov assumption attributed to `algorithm`.
#[must_use]
pub(crate) fn causal_markov(algorithm: impl Into<Arc<str>>) -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::CausalMarkov,
        source: AssumptionSource::AlgorithmDefault { algorithm: algorithm.into() },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}

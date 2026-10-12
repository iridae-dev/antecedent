//! Exact point-distribution wire shared by the transport artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_expr::ExactDistribution;
use serde::{Deserialize, Serialize};

use crate::query_wire::ValueWire;

/// Point distribution in canonical atom order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzTransportPointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl MzTransportPointWire {
    /// Encode a distribution.
    #[must_use]
    pub fn from_distribution(distribution: &ExactDistribution) -> Self {
        Self {
            outcomes: distribution.outcomes.iter().map(|v| v.raw()).collect(),
            atoms: distribution
                .atoms
                .iter()
                .map(|atom| atom.iter().map(ValueWire::from_value).collect())
                .collect(),
            probabilities: distribution.probabilities.to_vec(),
        }
    }

    /// Bit-exact replay; non-finite probabilities fail closed.
    #[must_use]
    // doc(hidden): antecedent-io artifacts replay stored points with it.
    #[doc(hidden)]
    pub fn replays(&self, recomputed: &Self) -> bool {
        self.outcomes == recomputed.outcomes
            && self.atoms == recomputed.atoms
            && self.probabilities.len() == recomputed.probabilities.len()
            && self.probabilities.iter().zip(&recomputed.probabilities).all(|(stored, fresh)| {
                stored.is_finite() && fresh.is_finite() && stored.to_bits() == fresh.to_bits()
            })
    }
}

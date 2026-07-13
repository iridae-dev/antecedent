//! Discovery constraints (DESIGN.md §13.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{Lag, VariableId};

use crate::result::LaggedLink;

/// Temporal constraint knobs for PCMCI-style search.
#[derive(Clone, Debug)]
pub struct TemporalConstraints {
    /// Maximum lag.
    pub max_lag: Lag,
    /// Minimum lag (usually 1 for lagged-only PCMCI).
    pub min_lag: Lag,
}

impl Default for TemporalConstraints {
    fn default() -> Self {
        Self { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) }
    }
}

/// Compiled discovery constraints.
#[derive(Clone, Debug)]
pub struct DiscoveryConstraints {
    /// Forbidden lagged links.
    pub forbidden: Arc<[LaggedLink]>,
    /// Max parents per target.
    pub max_parents: Option<usize>,
    /// Temporal settings.
    pub temporal: TemporalConstraints,
    /// Max conditioning-set size in PC phase.
    pub max_cond_size: usize,
    /// Significance level.
    pub alpha: f64,
}

impl Default for DiscoveryConstraints {
    fn default() -> Self {
        Self {
            forbidden: Arc::from([]),
            max_parents: None,
            temporal: TemporalConstraints::default(),
            max_cond_size: 3,
            alpha: 0.05,
        }
    }
}

impl DiscoveryConstraints {
    /// Whether a link is forbidden.
    #[must_use]
    pub fn is_forbidden(&self, link: LaggedLink) -> bool {
        self.forbidden.iter().any(|f| *f == link)
    }

    /// Variables that may appear as sources toward `target` (all except self at lag 0).
    #[must_use]
    pub fn candidate_sources(
        &self,
        variables: &[VariableId],
        target: VariableId,
    ) -> Vec<(VariableId, Lag)> {
        let min_l = self.temporal.min_lag.raw();
        let max_l = self.temporal.max_lag.raw();
        let mut out = Vec::new();
        for &v in variables {
            for lag in min_l..=max_l {
                let link = LaggedLink {
                    source: v,
                    source_lag: Lag::from_raw(lag),
                    target,
                    target_lag: Lag::CONTEMPORANEOUS,
                };
                if self.is_forbidden(link) {
                    continue;
                }
                if v == target && lag == 0 {
                    continue;
                }
                out.push((v, Lag::from_raw(lag)));
            }
        }
        out
    }
}

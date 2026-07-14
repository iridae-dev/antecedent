//! Discovery constraints (DESIGN.md §13.2) with compiled masks (§13.4 / §13.8).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{Lag, VariableId};
use causal_graph::{BitSet, DenseNodeId};
use causal_stats::SignificanceMethod;

use crate::error::DiscoveryError;
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

/// How contemporaneous links may relate across environments (J-PCMCI+).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CrossEnvLinkAssumption {
    /// No cross-environment contemporaneous assumption (default).
    #[default]
    Independent,
    /// Require the same contemporaneous skeleton across environments.
    SharedContemporaneousSkeleton,
    /// Allow environment-specific contemporaneous links (still shared lagged search space).
    EnvironmentSpecificContemporaneous,
}

/// Multi-dataset / context-aware discovery constraints (Phase 9 / J-PCMCI+).
#[derive(Clone, Debug, Default)]
pub struct MultiDatasetConstraints {
    /// System variables that act as context (appear as [`NodeRef::Context`] in output graphs).
    pub context_variables: Arc<[VariableId]>,
    /// Cross-environment link assumptions for pooled CI.
    pub cross_env: CrossEnvLinkAssumption,
    /// When true, pool CI evidence across environments for lagged system links.
    pub pool_lagged_ci: bool,
}

impl MultiDatasetConstraints {
    /// Whether `v` is marked as a context variable.
    #[must_use]
    pub fn is_context(&self, v: VariableId) -> bool {
        self.context_variables.iter().any(|&x| x == v)
    }
}

/// Compiled discovery constraints.
#[derive(Clone, Debug)]
pub struct DiscoveryConstraints {
    /// Required lagged links (must survive PC/MCI as candidates).
    pub required: Arc<[LaggedLink]>,
    /// Forbidden lagged links.
    pub forbidden: Arc<[LaggedLink]>,
    /// Variable tiers (earlier tiers cannot be caused by later tiers).
    pub tiers: Arc<[Arc<[VariableId]>]>,
    /// Max parents per target.
    pub max_parents: Option<usize>,
    /// Temporal settings.
    pub temporal: TemporalConstraints,
    /// Max conditioning-set size in PC phase.
    pub max_cond_size: usize,
    /// Significance level.
    pub alpha: f64,
    /// CI significance method (analytic or block-shuffle).
    pub significance: SignificanceMethod,
    /// Optional multi-dataset / context constraints (ignored by single-series PCMCI).
    pub multi_dataset: MultiDatasetConstraints,
}

impl Default for DiscoveryConstraints {
    fn default() -> Self {
        Self {
            required: Arc::from([]),
            forbidden: Arc::from([]),
            tiers: Arc::from([]),
            max_parents: None,
            temporal: TemporalConstraints::default(),
            max_cond_size: 3,
            alpha: 0.05,
            significance: SignificanceMethod::Analytic,
            multi_dataset: MultiDatasetConstraints::default(),
        }
    }
}

impl DiscoveryConstraints {
    /// Whether a link is forbidden by the explicit forbidden list.
    #[must_use]
    pub fn is_forbidden(&self, link: LaggedLink) -> bool {
        self.forbidden.iter().any(|f| *f == link)
    }

    /// Whether a link is required.
    #[must_use]
    pub fn is_required(&self, link: LaggedLink) -> bool {
        self.required.iter().any(|r| *r == link)
    }

    /// Tier index of `v`, or `None` if tiers are unused / variable absent.
    #[must_use]
    pub fn tier_of(&self, v: VariableId) -> Option<usize> {
        if self.tiers.is_empty() {
            return None;
        }
        self.tiers.iter().position(|tier| tier.iter().any(|x| *x == v))
    }

    /// Tier rule: edge `src → target` is forbidden when `tier(src) > tier(target)`.
    #[must_use]
    pub fn tier_forbids(&self, src: VariableId, target: VariableId) -> bool {
        match (self.tier_of(src), self.tier_of(target)) {
            (Some(ts), Some(tt)) => ts > tt,
            _ => false,
        }
    }

    /// Validate required vs forbidden conflicts and multi-dataset sanity.
    ///
    /// # Errors
    ///
    /// Conflicting required/forbidden edges, or context variables listed as
    /// both context and ordinary without a clear role.
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        for r in self.required.iter() {
            if self.is_forbidden(*r) || self.tier_forbids(r.source, r.target) {
                return Err(DiscoveryError::Unsupported {
                    message: "required edge conflicts with forbidden or tier constraints",
                });
            }
        }
        // Context → context lagged links are out of scope for J-PCMCI+ system search.
        for r in self.required.iter() {
            if self.multi_dataset.is_context(r.source) && self.multi_dataset.is_context(r.target) {
                return Err(DiscoveryError::Unsupported {
                    message: "required context→context lagged links are unsupported",
                });
            }
        }
        Ok(())
    }

    /// Variables that may appear as sources toward `target`.
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
                if self.is_forbidden(link) || self.tier_forbids(v, target) {
                    continue;
                }
                if v == target && lag == 0 {
                    continue;
                }
                out.push((v, Lag::from_raw(lag)));
            }
        }
        // Ensure required parents for this target are present.
        for r in self.required.iter() {
            if r.target == target
                && r.target_lag == Lag::CONTEMPORANEOUS
                && !out.contains(&(r.source, r.source_lag))
            {
                out.push((r.source, r.source_lag));
            }
        }
        out
    }

    /// Compile dense candidate / forbidden / required masks over the full
    /// candidate link space for `variables`.
    ///
    /// # Errors
    ///
    /// Validation failures or overflow.
    pub fn compile(&self, variables: &[VariableId]) -> Result<CompiledConstraints, DiscoveryError> {
        self.validate()?;
        let catalog = CandidateCatalog::build(variables, &self.temporal)?;
        let mut forbidden = BitSet::with_len(catalog.len());
        let mut required = BitSet::with_len(catalog.len());
        let mut candidates = BitSet::with_len(catalog.len());
        for idx in 0..catalog.len() {
            let link = catalog.link_at(idx);
            let banned = self.is_forbidden(link) || self.tier_forbids(link.source, link.target);
            if banned {
                forbidden.insert(DenseNodeId::from_raw(idx as u32));
            } else {
                candidates.insert(DenseNodeId::from_raw(idx as u32));
            }
            if self.is_required(link) {
                if banned {
                    return Err(DiscoveryError::Unsupported {
                        message: "required edge is forbidden after compilation",
                    });
                }
                required.insert(DenseNodeId::from_raw(idx as u32));
            }
        }
        Ok(CompiledConstraints { catalog, forbidden, required, candidates })
    }
}

/// Dense enumeration of candidate lagged links for a variable set.
#[derive(Clone, Debug)]
pub struct CandidateCatalog {
    variables: Arc<[VariableId]>,
    min_lag: u32,
    max_lag: u32,
    n_lags: usize,
    /// `n_vars * n_lags * n_vars` (`source_slot`, `lag_slot`, `target_slot`).
    len: usize,
}

impl CandidateCatalog {
    fn build(
        variables: &[VariableId],
        temporal: &TemporalConstraints,
    ) -> Result<Self, DiscoveryError> {
        let min_lag = temporal.min_lag.raw();
        let max_lag = temporal.max_lag.raw();
        if min_lag > max_lag {
            return Err(DiscoveryError::Unsupported { message: "min_lag must be ≤ max_lag" });
        }
        let n_vars = variables.len();
        let n_lags = (max_lag - min_lag) as usize + 1;
        let len = n_vars
            .checked_mul(n_lags)
            .and_then(|x| x.checked_mul(n_vars))
            .ok_or(DiscoveryError::Unsupported { message: "candidate catalog overflow" })?;
        if len > u32::MAX as usize {
            return Err(DiscoveryError::Unsupported {
                message: "candidate catalog exceeds u32 index space",
            });
        }
        Ok(Self { variables: Arc::from(variables), min_lag, max_lag, n_lags, len })
    }

    /// Number of candidate slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Dense index for a link, if it lies in the catalog.
    #[must_use]
    pub fn index_of(&self, link: LaggedLink) -> Option<usize> {
        if link.target_lag != Lag::CONTEMPORANEOUS {
            return None;
        }
        let lag = link.source_lag.raw();
        if lag < self.min_lag || lag > self.max_lag {
            return None;
        }
        let src_slot = self.variables.iter().position(|&v| v == link.source)?;
        let tgt_slot = self.variables.iter().position(|&v| v == link.target)?;
        let lag_slot = (lag - self.min_lag) as usize;
        Some((src_slot * self.n_lags + lag_slot) * self.variables.len() + tgt_slot)
    }

    /// Link at dense index.
    #[must_use]
    pub fn link_at(&self, idx: usize) -> LaggedLink {
        debug_assert!(idx < self.len);
        let n_vars = self.variables.len();
        let tgt_slot = idx % n_vars;
        let rest = idx / n_vars;
        let lag_slot = rest % self.n_lags;
        let src_slot = rest / self.n_lags;
        LaggedLink {
            source: self.variables[src_slot],
            source_lag: Lag::from_raw(self.min_lag + lag_slot as u32),
            target: self.variables[tgt_slot],
            target_lag: Lag::CONTEMPORANEOUS,
        }
    }
}

/// Compiled dense masks over [`CandidateCatalog`].
#[derive(Clone, Debug)]
pub struct CompiledConstraints {
    /// Candidate link catalog.
    pub catalog: CandidateCatalog,
    /// Forbidden link bits.
    pub forbidden: BitSet,
    /// Required link bits.
    pub required: BitSet,
    /// Allowed candidate bits (not forbidden).
    pub candidates: BitSet,
}

impl CompiledConstraints {
    /// Whether the link is an allowed candidate.
    #[must_use]
    pub fn allows(&self, link: LaggedLink) -> bool {
        self.catalog
            .index_of(link)
            .is_some_and(|i| self.candidates.contains(DenseNodeId::from_raw(i as u32)))
    }

    /// Whether the link is required.
    #[must_use]
    pub fn requires(&self, link: LaggedLink) -> bool {
        self.catalog
            .index_of(link)
            .is_some_and(|i| self.required.contains(DenseNodeId::from_raw(i as u32)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_marks_forbidden_and_required() {
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let link = LaggedLink {
            source: vars[0],
            source_lag: Lag::from_raw(1),
            target: vars[1],
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let c = DiscoveryConstraints {
            forbidden: Arc::from([link]),
            required: Arc::from([]),
            temporal: TemporalConstraints { max_lag: Lag::from_raw(2), min_lag: Lag::from_raw(1) },
            ..DiscoveryConstraints::default()
        };
        let compiled = c.compile(&vars).unwrap();
        assert!(!compiled.allows(link));
    }

    #[test]
    fn tiers_forbid_backward_edges() {
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let c = DiscoveryConstraints {
            tiers: Arc::from([Arc::from([vars[0]]), Arc::from([vars[1]])]),
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
            ..DiscoveryConstraints::default()
        };
        // Edge from tier1 → tier0 forbidden.
        let bad = LaggedLink {
            source: vars[1],
            source_lag: Lag::from_raw(1),
            target: vars[0],
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let compiled = c.compile(&vars).unwrap();
        assert!(!compiled.allows(bad));
    }

    #[test]
    fn multi_dataset_context_flags() {
        let ctx = VariableId::from_raw(2);
        let c = DiscoveryConstraints {
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from([ctx]),
                cross_env: CrossEnvLinkAssumption::SharedContemporaneousSkeleton,
                pool_lagged_ci: true,
            },
            ..DiscoveryConstraints::default()
        };
        assert!(c.multi_dataset.is_context(ctx));
        assert!(!c.multi_dataset.is_context(VariableId::from_raw(0)));
        c.validate().unwrap();
    }
}

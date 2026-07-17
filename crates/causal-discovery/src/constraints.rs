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
///
/// Retained for API compatibility. Günther pooled search does not use this enum
/// to drive pooling; link assumptions come from node roles instead.
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

/// Observed context kind for J-PCMCI+ (Günther et al.).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ContextKind {
    /// Constant within an environment; varies across datasets (spatial context).
    #[default]
    Space,
    /// Shared across environments; may vary in time (temporal context).
    Time,
}

/// J-PCMCI+ node role used for Günther link assumptions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum JpcmciNodeRole {
    /// Ordinary system variable.
    System,
    /// Observed spatial context.
    SpaceContext,
    /// Observed temporal context.
    TimeContext,
    /// Synthetic space (dataset) dummy column.
    SpaceDummy,
    /// Synthetic time dummy column.
    TimeDummy,
}

impl JpcmciNodeRole {
    /// Whether this role is any observed context.
    #[must_use]
    pub const fn is_observed_context(self) -> bool {
        matches!(self, Self::SpaceContext | Self::TimeContext)
    }

    /// Whether this role is any dummy.
    #[must_use]
    pub const fn is_dummy(self) -> bool {
        matches!(self, Self::SpaceDummy | Self::TimeDummy)
    }

    /// Whether this role is exogenous to the system (context or dummy).
    #[must_use]
    pub const fn is_exogenous(self) -> bool {
        self.is_observed_context() || self.is_dummy()
    }

    /// Whether lagged parents of this node are allowed (only time context among exogenous).
    #[must_use]
    pub const fn allows_lagged_as_source(self) -> bool {
        matches!(self, Self::System | Self::TimeContext)
    }
}

/// How space (dataset) dummies enter CI tests in J-PCMCI+.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SpaceDummyCiMode {
    /// Production default: `M−1` one-hot columns, each a scalar ParCorr variable.
    #[default]
    ScalarOneHot,
    /// Tigramite-style: one logical space-dummy node; CI expands to the full one-hot block
    /// via [`causal_stats::PairwiseMultivariateCi`].
    MultivariateBlock,
}

/// Multi-dataset / context-aware discovery constraints (J-PCMCI+).
#[derive(Clone, Debug)]
pub struct MultiDatasetConstraints {
    /// Observed context variables (appear as [`causal_graph::NodeRef::Context`] in output).
    pub context_variables: Arc<[VariableId]>,
    /// Optional kind per context variable; missing entries default to [`ContextKind::Space`].
    pub context_kinds: Arc<[(VariableId, ContextKind)]>,
    /// Synthetic space-dummy variable ids (filled by the J-PCMCI+ runner).
    pub space_dummy_variables: Arc<[VariableId]>,
    /// Synthetic time-dummy variable ids (filled by the J-PCMCI+ runner).
    pub time_dummy_variables: Arc<[VariableId]>,
    /// When true (and ≥2 envs), synthesize a space dummy.
    pub include_space_dummy: bool,
    /// When true, synthesize a time-index dummy.
    pub include_time_dummy: bool,
    /// How space dummies enter CI (scalar one-hot vs multivariate block).
    pub space_dummy_ci: SpaceDummyCiMode,
    /// Cross-environment link assumptions (API compatibility; unused by Günther path).
    pub cross_env: CrossEnvLinkAssumption,
    /// Legacy intersection-pool flag; ignored by Günther pooled search.
    pub pool_lagged_ci: bool,
}

impl Default for MultiDatasetConstraints {
    fn default() -> Self {
        Self {
            context_variables: Arc::from([]),
            context_kinds: Arc::from([]),
            space_dummy_variables: Arc::from([]),
            time_dummy_variables: Arc::from([]),
            include_space_dummy: true,
            include_time_dummy: false,
            space_dummy_ci: SpaceDummyCiMode::ScalarOneHot,
            cross_env: CrossEnvLinkAssumption::default(),
            pool_lagged_ci: true,
        }
    }
}

impl MultiDatasetConstraints {
    /// Whether `v` is marked as an observed context variable.
    #[must_use]
    pub fn is_context(&self, v: VariableId) -> bool {
        self.context_variables.iter().any(|&x| x == v)
    }

    /// Whether `v` is a synthetic dummy.
    #[must_use]
    pub fn is_dummy(&self, v: VariableId) -> bool {
        self.space_dummy_variables.iter().any(|&x| x == v)
            || self.time_dummy_variables.iter().any(|&x| x == v)
    }

    /// Kind of an observed context variable (defaults to space).
    #[must_use]
    pub fn context_kind(&self, v: VariableId) -> ContextKind {
        self.context_kinds
            .iter()
            .find(|(id, _)| *id == v)
            .map(|(_, k)| *k)
            .unwrap_or(ContextKind::Space)
    }

    /// Resolve the J-PCMCI+ role of `v` (system if unmarked).
    #[must_use]
    pub fn role_of(&self, v: VariableId) -> JpcmciNodeRole {
        if self.space_dummy_variables.iter().any(|&x| x == v) {
            return JpcmciNodeRole::SpaceDummy;
        }
        if self.time_dummy_variables.iter().any(|&x| x == v) {
            return JpcmciNodeRole::TimeDummy;
        }
        if self.is_context(v) {
            return match self.context_kind(v) {
                ContextKind::Space => JpcmciNodeRole::SpaceContext,
                ContextKind::Time => JpcmciNodeRole::TimeContext,
            };
        }
        JpcmciNodeRole::System
    }

    /// Günther / tigramite link-assumption: whether `link` is forbidden.
    ///
    /// Rules (exogenous context/dummy → system only; no context↔context / dummy children;
    /// space context/dummy and time dummy are contemporaneous-only sources).
    #[must_use]
    pub fn gunther_forbids(&self, link: LaggedLink) -> bool {
        let src = self.role_of(link.source);
        let tgt = self.role_of(link.target);
        let lag = link.source_lag.raw();

        // No edges into exogenous nodes.
        if tgt.is_exogenous() {
            return true;
        }
        // No exogenous ↔ exogenous (context/dummy among themselves).
        if src.is_exogenous() && tgt.is_exogenous() {
            return true;
        }
        // Exogenous → system only; already covered if tgt is system.
        if src.is_exogenous() {
            // Lagged sources only allowed for time context.
            if lag > 0 && !src.allows_lagged_as_source() {
                return true;
            }
        }
        false
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
    /// Whether a link is forbidden by the explicit forbidden list, tiers, or
    /// Günther multi-dataset link assumptions.
    #[must_use]
    pub fn is_forbidden(&self, link: LaggedLink) -> bool {
        self.forbidden.iter().any(|f| *f == link)
            || self.tier_forbids(link.source, link.target)
            || self.multi_dataset.gunther_forbids(link)
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
                if self.is_forbidden(link) {
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
            let banned = self.is_forbidden(link);
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
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        };
        assert!(c.multi_dataset.is_context(ctx));
        assert!(!c.multi_dataset.is_context(VariableId::from_raw(0)));
        assert_eq!(c.multi_dataset.role_of(ctx), JpcmciNodeRole::SpaceContext);
        c.validate().unwrap();
    }

    #[test]
    fn gunther_forbids_system_to_context_and_lagged_space_dummy() {
        let sys = VariableId::from_raw(0);
        let ctx = VariableId::from_raw(1);
        let dummy = VariableId::from_raw(2);
        let md = MultiDatasetConstraints {
            context_variables: Arc::from([ctx]),
            space_dummy_variables: Arc::from([dummy]),
            ..MultiDatasetConstraints::default()
        };
        let into_ctx = LaggedLink {
            source: sys,
            source_lag: Lag::CONTEMPORANEOUS,
            target: ctx,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        assert!(md.gunther_forbids(into_ctx));
        let lagged_dummy = LaggedLink {
            source: dummy,
            source_lag: Lag::from_raw(1),
            target: sys,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        assert!(md.gunther_forbids(lagged_dummy));
        let ok = LaggedLink {
            source: dummy,
            source_lag: Lag::CONTEMPORANEOUS,
            target: sys,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        assert!(!md.gunther_forbids(ok));
    }
}

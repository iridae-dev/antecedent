//! Discovery constraints with compiled masks (§13.4 / §13.8).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use antecedent_core::{Lag, VariableId};
use antecedent_data::VectorVariableGroups;
use antecedent_graph::{BitSet, DenseNodeId};
use antecedent_stats::SignificanceMethod;

use crate::error::DiscoveryError;
use crate::result::{LaggedLink, ScoredLink};

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

/// Which of `(y, x, z)` participate when building masked CI complete-case keep masks.
///
/// Matches the pinned baseline `mask_type` taxonomy (`"y"`, `"x"`, `"z"`, combinations).
/// Default [`Self::Yxz`] intersects masks on the outcome, cause, and conditioning set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum CiMaskType {
    /// Outcome (`y`) only.
    Y,
    /// Cause (`x`) only.
    X,
    /// Conditioning set (`z`) only.
    Z,
    /// `y` and `x`.
    Yx,
    /// `y` and `z`.
    Yz,
    /// `x` and `z`.
    Xz,
    /// `y`, `x`, and `z` (default; full complete-case).
    #[default]
    Yxz,
}

impl CiMaskType {
    /// Parse a baseline-style tag (`"y"`, `"yx"`, `"yxz"`, …). Case-insensitive.
    ///
    /// # Errors
    ///
    /// Unrecognized tag.
    pub fn parse(tag: &str) -> Result<Self, DiscoveryError> {
        match tag.trim().to_ascii_lowercase().as_str() {
            "y" => Ok(Self::Y),
            "x" => Ok(Self::X),
            "z" => Ok(Self::Z),
            "yx" | "xy" => Ok(Self::Yx),
            "yz" | "zy" => Ok(Self::Yz),
            "xz" | "zx" => Ok(Self::Xz),
            "yxz" | "xyz" | "xzy" | "yzx" | "zxy" | "zyx" => Ok(Self::Yxz),
            other => Err(DiscoveryError::data_msg(format!(
                "unknown CiMaskType tag `{other}` (expected y/x/z combinations)"
            ))),
        }
    }

    /// Whether the outcome column participates in the keep-mask.
    #[must_use]
    pub const fn includes_y(self) -> bool {
        matches!(self, Self::Y | Self::Yx | Self::Yz | Self::Yxz)
    }

    /// Whether the cause column participates in the keep-mask.
    #[must_use]
    pub const fn includes_x(self) -> bool {
        matches!(self, Self::X | Self::Yx | Self::Xz | Self::Yxz)
    }

    /// Whether conditioning columns participate in the keep-mask.
    #[must_use]
    pub const fn includes_z(self) -> bool {
        matches!(self, Self::Z | Self::Yz | Self::Xz | Self::Yxz)
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

/// Observed context kind for J-PCMCI+ (Günther et al. 2023).
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
    /// Production default: `M−1` one-hot columns, each a scalar `ParCorr` variable.
    #[default]
    ScalarOneHot,
    /// pinned baseline-style: one logical space-dummy node; CI expands to the full one-hot block
    /// via [`antecedent_stats::PairwiseMultivariateCi`].
    MultivariateBlock,
}

/// How time one-hot dummies enter CI tests in J-PCMCI+ (ignored for integer-index encoding).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum TimeDummyCiMode {
    /// Production default: each one-hot column is a scalar `ParCorr` variable.
    #[default]
    ScalarOneHot,
    /// One logical time-dummy node; CI expands to the full one-hot block
    /// via [`antecedent_stats::PairwiseMultivariateCi`] (paper-faithful single `D_time`).
    MultivariateBlock,
}

/// Multi-dataset / context-aware discovery constraints (J-PCMCI+).
#[derive(Clone, Debug)]
pub struct MultiDatasetConstraints {
    /// Observed context variables (appear as [`antecedent_graph::NodeRef::Context`] in output).
    pub context_variables: Arc<[VariableId]>,
    /// Optional kind per context variable; missing entries default to [`ContextKind::Space`].
    pub context_kinds: Arc<[(VariableId, ContextKind)]>,
    /// Synthetic space-dummy variable ids (filled by the J-PCMCI+ runner).
    pub space_dummy_variables: Arc<[VariableId]>,
    /// Synthetic time-dummy variable ids (filled by the J-PCMCI+ runner).
    pub time_dummy_variables: Arc<[VariableId]>,
    /// When true (and ≥2 envs), synthesize a space dummy.
    pub include_space_dummy: bool,
    /// When true, synthesize a time dummy under [`Self::time_dummy_encoding`].
    pub include_time_dummy: bool,
    /// Integer index vs one-hot of `T` for the time dummy.
    pub time_dummy_encoding: antecedent_data::TimeDummyEncoding,
    /// Cap on distinct time levels for one-hot encoding (fail-closed).
    pub max_time_one_hot_levels: usize,
    /// How space dummies enter CI (scalar one-hot vs multivariate block).
    pub space_dummy_ci: SpaceDummyCiMode,
    /// How time one-hot dummies enter CI (scalar vs multivariate block).
    pub time_dummy_ci: TimeDummyCiMode,
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
            time_dummy_encoding: antecedent_data::TimeDummyEncoding::IntegerIndex,
            max_time_one_hot_levels: antecedent_data::DEFAULT_MAX_TIME_ONE_HOT_LEVELS,
            space_dummy_ci: SpaceDummyCiMode::ScalarOneHot,
            time_dummy_ci: TimeDummyCiMode::ScalarOneHot,
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
        self.context_kinds.iter().find(|(id, _)| *id == v).map_or(ContextKind::Space, |(_, k)| *k)
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

    /// Günther / pinned baseline link-assumption: whether `link` is forbidden.
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

/// The kind of search background knowledge is applied to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnowledgeScope {
    /// Constraint-based search over a table that returns an equivalence class (PC, FCI, RFCI).
    EquivalenceClass,
    /// Score-based search over a table (GES).
    ScoreSearch,
    /// Lagged search over a series (PCMCI, PCMCI+).
    Lagged,
}

/// Why background knowledge cannot be applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeFault {
    /// The statements cannot all hold.
    Contradictory(String),
    /// A statement names a variable the search does not cover.
    UnknownVariable(String),
    /// A statement has no meaning for the algorithm, so it is refused rather than ignored.
    NotApplicable(String),
}

impl KnowledgeFault {
    /// The precise reason, naming the offending statement.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Contradictory(m) | Self::UnknownVariable(m) | Self::NotApplicable(m) => m,
        }
    }

    /// The registered reason code a refusal of this fault carries.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::Contradictory(_) => antecedent_core::reason_code!("invalid_argument"),
            Self::UnknownVariable(_) => antecedent_core::reason_code!("unknown_variable"),
            Self::NotApplicable(_) => antecedent_core::reason_code!("option_not_applicable"),
        }
    }
}

impl std::fmt::Display for KnowledgeFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

fn describe_link(link: LaggedLink) -> String {
    format!(
        "variable {} at lag {} -> variable {} at lag {}",
        link.source.raw(),
        link.source_lag.raw(),
        link.target.raw(),
        link.target_lag.raw()
    )
}

/// Whether the directed graph with these `(from, to)` edges has a cycle (a self-loop counts).
fn has_directed_cycle(edges: &[(u32, u32)]) -> bool {
    let mut in_degree: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for &(from, to) in edges {
        in_degree.entry(from).or_insert(0);
        *in_degree.entry(to).or_insert(0) += 1;
    }
    let mut ready: Vec<u32> = in_degree.iter().filter(|(_, d)| **d == 0).map(|(n, _)| *n).collect();
    let mut removed = 0usize;
    while let Some(node) = ready.pop() {
        removed += 1;
        for &(from, to) in edges {
            if from == node {
                if let Some(d) = in_degree.get_mut(&to) {
                    *d -= 1;
                    if *d == 0 {
                        ready.push(to);
                    }
                }
            }
        }
    }
    removed < in_degree.len()
}

/// The background knowledge of a discovery request: what the analyst asserts about the graph
/// before the data are consulted.
///
/// This is the part of [`DiscoveryConstraints`] the ordinary entry points and the selective
/// contracts share, so one wire parser and one identity cover both.
#[derive(Clone, Debug)]
pub struct BackgroundKnowledge {
    /// Links that must be in the graph (adjacency, and the orientation they state).
    pub required: Arc<[LaggedLink]>,
    /// Links that must not be in the graph.
    pub forbidden: Arc<[LaggedLink]>,
    /// Variable tiers: a later tier cannot cause an earlier one.
    pub tiers: Arc<[Arc<[VariableId]>]>,
    /// Upper bound on the parents of each variable.
    pub max_parents: Option<usize>,
}

impl Default for BackgroundKnowledge {
    fn default() -> Self {
        Self {
            required: Arc::from([]),
            forbidden: Arc::from([]),
            tiers: Arc::from([]),
            max_parents: None,
        }
    }
}

impl BackgroundKnowledge {
    /// Whether nothing is asserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.required.is_empty()
            && self.forbidden.is_empty()
            && self.tiers.is_empty()
            && self.max_parents.is_none()
    }

    /// Set these statements on `constraints`, replacing its own.
    pub fn apply_to(&self, constraints: &mut DiscoveryConstraints) {
        constraints.required = Arc::clone(&self.required);
        constraints.forbidden = Arc::clone(&self.forbidden);
        constraints.tiers = Arc::clone(&self.tiers);
        constraints.max_parents = self.max_parents;
    }

    /// Canonical identity of the statements: the same digest the selective contracts declare as
    /// their `background_knowledge` input, independent of the order the links are listed in.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        let mut constraints = DiscoveryConstraints::default();
        self.apply_to(&mut constraints);
        crate::selective::background_knowledge_identity(&constraints)
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
    /// pinned baseline-style vector-variable groups (logical node = first component).
    pub vector_groups: VectorVariableGroups,
    /// Which of `(y, x, z)` drive masked CI keep-masks (baseline `mask_type`).
    pub mask_type: CiMaskType,
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
            vector_groups: VectorVariableGroups::empty(),
            mask_type: CiMaskType::Yxz,
        }
    }
}

/// A direction background knowledge fixes on one contemporaneous pair (see
/// [`DiscoveryConstraints::static_direction`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FixedDirection {
    /// The cause end.
    pub(crate) from: VariableId,
    /// The effect end.
    pub(crate) to: VariableId,
    /// A required link (a direct cause), as against the reverse of a forbidden link.
    pub(crate) required: bool,
}

/// Contemporaneous link `source → target`.
pub(crate) fn contemp_link(source: VariableId, target: VariableId) -> LaggedLink {
    LaggedLink {
        source,
        source_lag: Lag::CONTEMPORANEOUS,
        target,
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

/// Whether the contemporaneous edge `variables[par] → variables[child]` is
/// forbidden by the constraints. Out-of-range indices are treated as forbidden.
#[cfg(test)]
pub(crate) fn forbidden_edge(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    par: usize,
    child: usize,
) -> bool {
    let Some(&src) = variables.get(par) else {
        return true;
    };
    let Some(&tgt) = variables.get(child) else {
        return true;
    };
    // `is_forbidden` already applies `tier_forbids`.
    constraints.is_forbidden(contemp_link(src, tgt))
}

/// First-tier index and forbidden-link set, built once for a scan of many edges.
///
/// Empty tiers stay a `None` index so a variable is absent rather than tier 0.
/// A variable listed in several tiers keeps the earliest tier, matching
/// [`DiscoveryConstraints::tier_of`].
/// Tier index and forbidden-link set compiled for one constraint scan.
pub(crate) struct ConstraintIndex {
    tiers: Option<HashMap<VariableId, usize>>,
    forbidden: HashSet<LaggedLink>,
}

impl ConstraintIndex {
    /// Compile tiers and the forbidden list. Günther rules stay on `constraints`.
    pub(crate) fn compile(constraints: &DiscoveryConstraints) -> Self {
        let tiers = if constraints.tiers.is_empty() {
            None
        } else {
            let mut index = HashMap::new();
            for (tier_index, tier) in constraints.tiers.iter().enumerate() {
                for &variable in tier.iter() {
                    index.entry(variable).or_insert(tier_index);
                }
            }
            Some(index)
        };
        Self { tiers, forbidden: constraints.forbidden.iter().copied().collect() }
    }

    fn tier_of(&self, variable: VariableId) -> Option<usize> {
        self.tiers.as_ref()?.get(&variable).copied()
    }

    fn tier_forbids(&self, src: VariableId, target: VariableId) -> bool {
        match (self.tier_of(src), self.tier_of(target)) {
            (Some(src_tier), Some(target_tier)) => src_tier > target_tier,
            _ => false,
        }
    }

    /// Same boolean as [`DiscoveryConstraints::is_forbidden`].
    pub(crate) fn is_forbidden(
        &self,
        constraints: &DiscoveryConstraints,
        link: LaggedLink,
    ) -> bool {
        self.forbidden.contains(&link)
            || self.tier_forbids(link.source, link.target)
            || constraints.multi_dataset.gunther_forbids(link)
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

    /// Whether both directions of the contemporaneous pair `{a, b}` are forbidden, i.e. a static
    /// (non-temporal) edge between them cannot exist.
    #[must_use]
    pub fn static_forbidden(&self, a: VariableId, b: VariableId) -> bool {
        self.is_forbidden(contemp_link(a, b)) && self.is_forbidden(contemp_link(b, a))
    }

    /// Whether either direction of the contemporaneous pair `{a, b}` is required.
    #[must_use]
    pub fn static_required(&self, a: VariableId, b: VariableId) -> bool {
        self.is_required(contemp_link(a, b)) || self.is_required(contemp_link(b, a))
    }

    /// The suffix an algorithm appends to its configuration record: empty without background
    /// knowledge, else `,background_knowledge=<digest>`. A result's algorithm record, and every
    /// identity built on it, then changes exactly when the knowledge does.
    pub(crate) fn knowledge_config(&self) -> String {
        let none = self.required.is_empty()
            && self.forbidden.is_empty()
            && self.tiers.is_empty()
            && self.max_parents.is_none()
            && self.vector_groups.groups().is_empty();
        if none {
            return String::new();
        }
        let digest = crate::selective::background_knowledge_identity(self);
        format!(",background_knowledge={}", crate::selective::digest_hex(&digest))
    }

    /// `kept` plus the required links the threshold dropped, with their own scores.
    ///
    /// A required link is an edge by statement. Its test is still run and its p-value is
    /// reported as it is; the threshold only decides which *other* links survive. A required
    /// link that was never scored cannot be reported and is left out.
    pub(crate) fn with_required_links(
        &self,
        mut kept: Vec<ScoredLink>,
        scored: &[ScoredLink],
    ) -> Vec<ScoredLink> {
        for required in self.required.iter().filter(|r| r.target_lag.is_contemporaneous()) {
            if kept.iter().any(|s| s.link == *required) {
                continue;
            }
            if let Some(found) = scored.iter().find(|s| s.link == *required) {
                kept.push(*found);
            }
        }
        kept
    }

    /// Give each contemporaneous link whose reverse cannot be a candidate (the reverse is
    /// forbidden, or the link itself is required) a mirror image, so that the symmetrization
    /// that keeps an edge only when both directions survive does not discard an edge because
    /// of a direction knowledge has already ruled out.
    pub(crate) fn with_mirrored_contemporaneous(
        &self,
        mut scored: Vec<ScoredLink>,
    ) -> Vec<ScoredLink> {
        let mut mirrors = Vec::new();
        for s in &scored {
            let l = s.link;
            if !(l.source_lag.is_contemporaneous() && l.target_lag.is_contemporaneous()) {
                continue;
            }
            let reverse = contemp_link(l.target, l.source);
            if (self.is_forbidden(reverse) || self.is_required(l))
                && !scored.iter().any(|o| o.link == reverse)
            {
                mirrors.push(ScoredLink { link: reverse, ..*s });
            }
        }
        scored.extend(mirrors);
        scored
    }

    /// Whether any statement can fix the direction of a contemporaneous edge: a required
    /// contemporaneous link, a forbidden link, or a tier.
    pub(crate) fn has_directional_knowledge(&self) -> bool {
        !self.required.is_empty() || !self.forbidden.is_empty() || !self.tiers.is_empty()
    }

    /// The direction background knowledge fixes on the contemporaneous pair `{a, b}`, if any.
    ///
    /// A required link `a → b` fixes that direction and states a direct cause. Failing that, a
    /// direction is fixed when exactly the opposite link is forbidden, directly or by a tier:
    /// the pair may still be non-adjacent, but if it is adjacent the edge runs the other way.
    pub(crate) fn static_direction(&self, a: VariableId, b: VariableId) -> Option<FixedDirection> {
        let (ab, ba) = (contemp_link(a, b), contemp_link(b, a));
        if self.is_required(ab) {
            return Some(FixedDirection { from: a, to: b, required: true });
        }
        if self.is_required(ba) {
            return Some(FixedDirection { from: b, to: a, required: true });
        }
        match (self.is_forbidden(ab), self.is_forbidden(ba)) {
            (false, true) => Some(FixedDirection { from: a, to: b, required: false }),
            (true, false) => Some(FixedDirection { from: b, to: a, required: false }),
            _ => None,
        }
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
    /// Conflicting required/forbidden edges, required contemporaneous links that form a cycle,
    /// or context variables listed as both context and ordinary without a clear role.
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        self.check_consistency().map_err(DiscoveryError::Knowledge)?;
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

    /// Whether the background knowledge can hold at all, whatever the variables and the search.
    fn check_consistency(&self) -> Result<(), KnowledgeFault> {
        for r in self.required.iter() {
            if self.is_forbidden(*r) {
                return Err(KnowledgeFault::Contradictory(format!(
                    "required link {} is also forbidden, directly or by the tiers",
                    describe_link(*r)
                )));
            }
        }
        let contemporaneous: Vec<(u32, u32)> = self
            .required
            .iter()
            .filter(|r| r.source_lag.is_contemporaneous() && r.target_lag.is_contemporaneous())
            .map(|r| (r.source.raw(), r.target.raw()))
            .collect();
        if has_directed_cycle(&contemporaneous) {
            return Err(KnowledgeFault::Contradictory(
                "the required contemporaneous links form a directed cycle; DAG discovery refused \
                 this attempt. Next attempt: hold those links as a CyclicDigraph and read \
                 condensation. This search does not drop the cycle"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Check that the background knowledge can be applied by a search of the given kind over
    /// `variables`, so that no statement is silently dropped.
    ///
    /// The statements are checked for consistency ([`Self::validate`] does the same), every
    /// variable they name must be searched, and each statement must have a meaning for the
    /// search: a static search has no lags, a lagged search cannot honour a required link
    /// outside its lag window, and a parent limit is refused where the algorithm returns an
    /// equivalence class (see [`KnowledgeScope`]).
    ///
    /// # Errors
    ///
    /// A [`KnowledgeFault`] naming the offending statement.
    pub fn check_background_knowledge(
        &self,
        scope: KnowledgeScope,
        variables: &[VariableId],
    ) -> Result<(), KnowledgeFault> {
        self.check_consistency()?;
        let known: HashSet<VariableId> = variables.iter().copied().collect();
        for (what, links) in [("required", &self.required), ("forbidden", &self.forbidden)] {
            for link in links.iter() {
                for v in [link.source, link.target] {
                    if !known.contains(&v) {
                        return Err(KnowledgeFault::UnknownVariable(format!(
                            "the {what} link {} names variable {}, which is not searched",
                            describe_link(*link),
                            v.raw()
                        )));
                    }
                }
            }
        }
        for tier in self.tiers.iter() {
            if let Some(v) = tier.iter().find(|v| !known.contains(v)) {
                return Err(KnowledgeFault::UnknownVariable(format!(
                    "a tier names variable {}, which is not searched",
                    v.raw()
                )));
            }
        }
        match scope {
            KnowledgeScope::EquivalenceClass | KnowledgeScope::ScoreSearch => {
                for (what, links) in [("required", &self.required), ("forbidden", &self.forbidden)]
                {
                    if let Some(link) = links.iter().find(|l| {
                        !l.source_lag.is_contemporaneous() || !l.target_lag.is_contemporaneous()
                    }) {
                        return Err(KnowledgeFault::NotApplicable(format!(
                            "the {what} link {} is lagged, but a search over a table has no lags",
                            describe_link(*link)
                        )));
                    }
                }
            }
            KnowledgeScope::Lagged => {
                let (min, max) = (self.temporal.min_lag.raw(), self.temporal.max_lag.raw());
                if let Some(link) = self.required.iter().find(|l| {
                    !l.target_lag.is_contemporaneous()
                        || l.source_lag.raw() < min
                        || l.source_lag.raw() > max
                }) {
                    return Err(KnowledgeFault::NotApplicable(format!(
                        "the required link {} lies outside the searched lag window {min}..={max} \
                         at the target's own time, so it could never be honoured; widen the window \
                         or drop the link",
                        describe_link(*link)
                    )));
                }
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
        let index = ConstraintIndex::compile(self);
        let mut out = Vec::new();
        for &v in variables {
            for lag in min_l..=max_l {
                let link = LaggedLink {
                    source: v,
                    source_lag: Lag::from_raw(lag),
                    target,
                    target_lag: Lag::CONTEMPORANEOUS,
                };
                if index.is_forbidden(self, link) {
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
        let index = ConstraintIndex::compile(self);
        let mut forbidden = BitSet::with_len(catalog.len());
        let mut required = BitSet::with_len(catalog.len());
        let mut candidates = BitSet::with_len(catalog.len());
        for idx in 0..catalog.len() {
            let link = catalog.link_at(idx);
            let banned = index.is_forbidden(self, link);
            if banned {
                forbidden.insert(DenseNodeId::from_raw(crate::indexing::dense_u32(idx)));
            } else {
                candidates.insert(DenseNodeId::from_raw(crate::indexing::dense_u32(idx)));
            }
            if self.is_required(link) {
                if banned {
                    return Err(DiscoveryError::Unsupported {
                        message: "required edge is forbidden after compilation",
                    });
                }
                required.insert(DenseNodeId::from_raw(crate::indexing::dense_u32(idx)));
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
            source_lag: Lag::from_raw(self.min_lag + crate::indexing::dense_u32(lag_slot)),
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
        self.catalog.index_of(link).is_some_and(|i| {
            self.candidates.contains(DenseNodeId::from_raw(crate::indexing::dense_u32(i)))
        })
    }

    /// Whether the link is required.
    #[must_use]
    pub fn requires(&self, link: LaggedLink) -> bool {
        self.catalog.index_of(link).is_some_and(|i| {
            self.required.contains(DenseNodeId::from_raw(crate::indexing::dense_u32(i)))
        })
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
    fn a_required_two_cycle_names_condensation_and_a_chain_is_accepted() {
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let forward = contemp_link(vars[0], vars[1]);
        let back = contemp_link(vars[1], vars[0]);
        let cyclic = DiscoveryConstraints {
            required: Arc::from([forward, back]),
            ..DiscoveryConstraints::default()
        };
        let err = cyclic.validate().unwrap_err();
        let text = err.to_string();
        assert!(text.contains("directed cycle"), "{text}");
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("CyclicDigraph"), "{text}");
        let chain = DiscoveryConstraints {
            required: Arc::from([forward]),
            ..DiscoveryConstraints::default()
        };
        assert!(chain.validate().is_ok());
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

    #[test]
    fn ci_mask_type_parses_baseline_tags() {
        assert_eq!(CiMaskType::parse("yxz").unwrap(), CiMaskType::Yxz);
        assert_eq!(CiMaskType::parse("YX").unwrap(), CiMaskType::Yx);
        assert_eq!(CiMaskType::parse("z").unwrap(), CiMaskType::Z);
        assert!(CiMaskType::parse("nope").is_err());
        assert!(CiMaskType::Yxz.includes_x());
        assert!(CiMaskType::Y.includes_y());
        assert!(!CiMaskType::Y.includes_x());
    }

    fn scan_tier(tiers: &[Arc<[VariableId]>], variable: VariableId) -> Option<usize> {
        if tiers.is_empty() {
            return None;
        }
        tiers.iter().position(|tier| tier.iter().any(|id| *id == variable))
    }

    fn scan_forbidden(constraints: &DiscoveryConstraints, link: LaggedLink) -> bool {
        let tier_forbids = match (
            scan_tier(&constraints.tiers, link.source),
            scan_tier(&constraints.tiers, link.target),
        ) {
            (Some(src), Some(target)) => src > target,
            _ => false,
        };
        constraints.forbidden.iter().any(|forbidden| *forbidden == link)
            || tier_forbids
            || constraints.multi_dataset.gunther_forbids(link)
    }

    #[test]
    fn tiered_forbidden_edge_matches_the_tier_scan() {
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let absent = VariableId::from_raw(99);
        let constraints = DiscoveryConstraints {
            tiers: Arc::from([Arc::from([vars[0], vars[1]]), Arc::from([vars[1], vars[2]])]),
            forbidden: Arc::from([contemp_link(vars[0], vars[2])]),
            required: Arc::from([
                LaggedLink {
                    source: vars[0],
                    source_lag: Lag::from_raw(1),
                    target: vars[2],
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                LaggedLink {
                    source: absent,
                    source_lag: Lag::from_raw(1),
                    target: vars[2],
                    target_lag: Lag::CONTEMPORANEOUS,
                },
            ]),
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(0) },
            ..DiscoveryConstraints::default()
        };
        let probed = [vars[0], vars[1], vars[2], absent];
        for &src in &probed {
            assert_eq!(constraints.tier_of(src), scan_tier(&constraints.tiers, src));
            for &target in &probed {
                let link = contemp_link(src, target);
                assert_eq!(constraints.tier_forbids(src, target), {
                    match (
                        scan_tier(&constraints.tiers, src),
                        scan_tier(&constraints.tiers, target),
                    ) {
                        (Some(src_tier), Some(target_tier)) => src_tier > target_tier,
                        _ => false,
                    }
                });
                assert_eq!(constraints.is_forbidden(link), scan_forbidden(&constraints, link));
            }
        }
        for par in 0..vars.len() {
            for child in 0..vars.len() {
                let link = contemp_link(vars[par], vars[child]);
                assert_eq!(
                    forbidden_edge(&constraints, &vars, par, child),
                    scan_forbidden(&constraints, link)
                );
            }
        }
        assert!(forbidden_edge(&constraints, &vars, vars.len(), 0));

        let mut expected = Vec::new();
        for &source in &vars {
            for lag in 0..=1 {
                let link = LaggedLink {
                    source,
                    source_lag: Lag::from_raw(lag),
                    target: vars[2],
                    target_lag: Lag::CONTEMPORANEOUS,
                };
                if scan_forbidden(&constraints, link) || (source == vars[2] && lag == 0) {
                    continue;
                }
                expected.push((source, Lag::from_raw(lag)));
            }
        }
        for required in constraints.required.iter() {
            if required.target == vars[2]
                && required.target_lag == Lag::CONTEMPORANEOUS
                && !expected.contains(&(required.source, required.source_lag))
            {
                expected.push((required.source, required.source_lag));
            }
        }
        assert_eq!(constraints.candidate_sources(&vars, vars[2]), expected);

        let empty = DiscoveryConstraints::default();
        assert_eq!(empty.tier_of(vars[0]), None);
        assert!(!empty.tier_forbids(vars[1], vars[0]));
        assert!(!empty.is_forbidden(contemp_link(vars[1], vars[0])));
        assert_eq!(
            empty.candidate_sources(&vars[..1], vars[0]),
            vec![(vars[0], empty.temporal.min_lag)]
        );
    }

    fn lagged(source: u32, lag: u32, target: u32) -> LaggedLink {
        LaggedLink {
            source: VariableId::from_raw(source),
            source_lag: Lag::from_raw(lag),
            target: VariableId::from_raw(target),
            target_lag: Lag::CONTEMPORANEOUS,
        }
    }

    fn ids(n: u32) -> Vec<VariableId> {
        (0..n).map(VariableId::from_raw).collect()
    }

    fn fault(
        constraints: &DiscoveryConstraints,
        scope: KnowledgeScope,
        n: u32,
    ) -> Option<KnowledgeFault> {
        constraints.check_background_knowledge(scope, &ids(n)).err()
    }

    #[test]
    fn contradictory_knowledge_is_named_as_such() {
        let tier = |ids: &[u32]| -> Arc<[VariableId]> {
            ids.iter().copied().map(VariableId::from_raw).collect()
        };
        let both = DiscoveryConstraints {
            required: Arc::from([lagged(0, 0, 1)]),
            forbidden: Arc::from([lagged(0, 0, 1)]),
            ..DiscoveryConstraints::default()
        };
        // The tier order makes a later variable unable to cause an earlier one.
        let against_tier = DiscoveryConstraints {
            required: Arc::from([lagged(1, 0, 0)]),
            tiers: Arc::from([tier(&[0]), tier(&[1])]),
            ..DiscoveryConstraints::default()
        };
        let cycle = DiscoveryConstraints {
            required: Arc::from([lagged(0, 0, 1), lagged(1, 0, 2), lagged(2, 0, 0)]),
            ..DiscoveryConstraints::default()
        };
        let reciprocal = DiscoveryConstraints {
            required: Arc::from([lagged(0, 0, 1), lagged(1, 0, 0)]),
            ..DiscoveryConstraints::default()
        };
        for (name, c) in
            [("both", both), ("tier", against_tier), ("cycle", cycle), ("reciprocal", reciprocal)]
        {
            for scope in [
                KnowledgeScope::EquivalenceClass,
                KnowledgeScope::ScoreSearch,
                KnowledgeScope::Lagged,
            ] {
                assert!(
                    matches!(fault(&c, scope, 3), Some(KnowledgeFault::Contradictory(_))),
                    "{name} {scope:?}"
                );
            }
            assert!(c.validate().is_err(), "{name}");
        }
        // A lagged self-link is not a cycle.
        let lagged_self = DiscoveryConstraints {
            required: Arc::from([lagged(0, 1, 0)]),
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
            ..DiscoveryConstraints::default()
        };
        assert_eq!(fault(&lagged_self, KnowledgeScope::Lagged, 1), None);
    }

    #[test]
    fn unsearched_variables_are_refused_wherever_they_appear() {
        let tier: Arc<[VariableId]> = Arc::from([VariableId::from_raw(5)]);
        let in_required = DiscoveryConstraints {
            required: Arc::from([lagged(0, 0, 4)]),
            ..DiscoveryConstraints::default()
        };
        let in_forbidden = DiscoveryConstraints {
            forbidden: Arc::from([lagged(7, 0, 0)]),
            ..DiscoveryConstraints::default()
        };
        let in_tier =
            DiscoveryConstraints { tiers: Arc::from([tier]), ..DiscoveryConstraints::default() };
        for c in [in_required, in_forbidden, in_tier] {
            assert!(matches!(
                fault(&c, KnowledgeScope::EquivalenceClass, 3),
                Some(KnowledgeFault::UnknownVariable(_))
            ));
        }
    }

    #[test]
    fn a_statement_with_no_meaning_for_the_search_is_refused_not_dropped() {
        let capped =
            DiscoveryConstraints { max_parents: Some(2), ..DiscoveryConstraints::default() };
        // Every search enforces the parent limit, so none refuses it.
        assert_eq!(fault(&capped, KnowledgeScope::EquivalenceClass, 3), None);
        assert_eq!(fault(&capped, KnowledgeScope::ScoreSearch, 3), None);
        assert_eq!(fault(&capped, KnowledgeScope::Lagged, 3), None);

        let lagged_in_table = DiscoveryConstraints {
            forbidden: Arc::from([lagged(0, 1, 1)]),
            ..DiscoveryConstraints::default()
        };
        assert!(matches!(
            fault(&lagged_in_table, KnowledgeScope::ScoreSearch, 3),
            Some(KnowledgeFault::NotApplicable(_))
        ));

        // Window 1..=2: a required lag-0 or lag-3 link could never be honoured.
        let window = TemporalConstraints { max_lag: Lag::from_raw(2), min_lag: Lag::from_raw(1) };
        for lag in [0, 3] {
            let c = DiscoveryConstraints {
                required: Arc::from([lagged(0, lag, 1)]),
                temporal: window.clone(),
                ..DiscoveryConstraints::default()
            };
            assert!(
                matches!(
                    fault(&c, KnowledgeScope::Lagged, 3),
                    Some(KnowledgeFault::NotApplicable(_))
                ),
                "lag {lag}"
            );
        }
        let inside = DiscoveryConstraints {
            required: Arc::from([lagged(0, 2, 1)]),
            temporal: window,
            ..DiscoveryConstraints::default()
        };
        assert_eq!(fault(&inside, KnowledgeScope::Lagged, 3), None);
    }

    #[test]
    fn faults_carry_registered_reason_codes() {
        assert_eq!(KnowledgeFault::Contradictory(String::new()).reason_code(), "invalid_argument");
        assert_eq!(
            KnowledgeFault::UnknownVariable(String::new()).reason_code(),
            "unknown_variable"
        );
        assert_eq!(
            KnowledgeFault::NotApplicable(String::new()).reason_code(),
            "option_not_applicable"
        );
    }

    #[test]
    fn a_direction_is_fixed_by_a_required_link_or_exactly_one_forbidden_direction() {
        let (a, b) = (VariableId::from_raw(0), VariableId::from_raw(1));
        let none = DiscoveryConstraints::default();
        assert_eq!(none.static_direction(a, b), None);

        let required =
            DiscoveryConstraints { required: Arc::from([lagged(1, 0, 0)]), ..none.clone() };
        assert_eq!(
            required.static_direction(a, b),
            Some(FixedDirection { from: b, to: a, required: true })
        );

        let one_way =
            DiscoveryConstraints { forbidden: Arc::from([lagged(1, 0, 0)]), ..none.clone() };
        assert_eq!(
            one_way.static_direction(a, b),
            Some(FixedDirection { from: a, to: b, required: false })
        );

        // Both directions forbidden: no edge at all, so no direction to fix.
        let both = DiscoveryConstraints {
            forbidden: Arc::from([lagged(0, 0, 1), lagged(1, 0, 0)]),
            ..none
        };
        assert_eq!(both.static_direction(a, b), None);
        assert!(both.static_forbidden(a, b));
    }

    #[test]
    fn the_knowledge_identity_ignores_link_order_and_changes_with_any_statement() {
        let base = BackgroundKnowledge {
            required: Arc::from([lagged(0, 1, 1), lagged(1, 1, 2)]),
            forbidden: Arc::from([lagged(2, 0, 0)]),
            tiers: Arc::from([Arc::from([VariableId::from_raw(0)])]),
            max_parents: Some(2),
        };
        let reordered = BackgroundKnowledge {
            required: Arc::from([lagged(1, 1, 2), lagged(0, 1, 1)]),
            ..base.clone()
        };
        assert_eq!(base.identity(), reordered.identity());
        assert_ne!(base.identity(), BackgroundKnowledge::default().identity());
        let edits = [
            BackgroundKnowledge { required: Arc::from([lagged(0, 1, 1)]), ..base.clone() },
            BackgroundKnowledge { forbidden: Arc::from([]), ..base.clone() },
            BackgroundKnowledge { tiers: Arc::from([]), ..base.clone() },
            BackgroundKnowledge { max_parents: Some(3), ..base.clone() },
            BackgroundKnowledge { max_parents: None, ..base.clone() },
        ];
        for edit in edits {
            assert_ne!(edit.identity(), base.identity());
        }
    }
}

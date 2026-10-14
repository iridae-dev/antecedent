//! Discovery stages, their canonical dependency identities and the reuse plan between two
//! requests.
//!
//! The stages are distinct objects with distinct inputs:
//!
//! 1. `statistics`: conditional-independence statistics at identical input coordinates. Its
//!    inputs are the data content, the variable order, the test and its settings, the
//!    random seed, the kernel policy, the mask type and the lag-frame depth. New data
//!    changes this identity and nothing retained for the old identity is used again.
//! 2. `search`: the adjacency / parent search that turns statistics into a skeleton under
//!    thresholds, multiplicity control and background knowledge. A threshold change reruns
//!    the decisions and the adaptive test scheduling and draws the statistics from stage 1.
//! 3. `orientation`: collider, Meek or discriminating-path orientation under its own
//!    settings.
//! 4. `graph_output`: the discovered structure.
//!
//! The identity of a stage covers the stage's own declared inputs and the identity of the
//! stage it depends on, so a change anywhere upstream is visible downstream. The plan names
//! the changed inputs. It decides and explains; it computes nothing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::recalc::StageIdentity;

use super::identity::Digest;

/// Components of the statistics stage, in canonical order.
pub const STATISTIC_COMPONENTS: [&str; 9] = [
    "data",
    "variables",
    "population",
    "ci_test",
    "significance",
    "rng",
    "kernel_policy",
    "mask",
    "frame_depth",
];
/// Components added by the search stage.
pub const SEARCH_COMPONENTS: [&str; 5] =
    ["algorithm", "thresholds", "multiplicity", "background_knowledge", "search_limits"];
/// Components added by the orientation stage.
pub const ORIENTATION_COMPONENTS: [&str; 1] = ["orientation_settings"];

/// One discovery stage.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DiscoveryStage {
    /// Cached tests.
    Statistics,
    /// Skeleton / parent search state.
    Search,
    /// Orientations.
    Orientation,
    /// The discovered structure.
    GraphOutput,
}

impl DiscoveryStage {
    /// Every stage in dependency order.
    pub const ALL: [Self; 4] =
        [Self::Statistics, Self::Search, Self::Orientation, Self::GraphOutput];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Statistics => "statistics",
            Self::Search => "search",
            Self::Orientation => "orientation",
            Self::GraphOutput => "graph_output",
        }
    }

    /// The stage this one reads, if any.
    #[must_use]
    pub const fn upstream(self) -> Option<Self> {
        match self {
            Self::Statistics => None,
            Self::Search => Some(Self::Statistics),
            Self::Orientation => Some(Self::Search),
            Self::GraphOutput => Some(Self::Orientation),
        }
    }

    /// Names of the inputs this stage declares itself (not inherited from upstream).
    #[must_use]
    pub const fn own_components(self) -> &'static [&'static str] {
        match self {
            Self::Statistics => &STATISTIC_COMPONENTS,
            Self::Search => &SEARCH_COMPONENTS,
            Self::Orientation => &ORIENTATION_COMPONENTS,
            Self::GraphOutput => &[],
        }
    }
}

/// The declared inputs of one request, as named content digests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageInputs {
    components: BTreeMap<&'static str, Digest>,
    route: &'static str,
}

impl StageInputs {
    /// Inputs of `route` (`static_pc`, `temporal_pcmci`, ...) with the given components.
    ///
    /// Components not named by a stage are rejected by returning `None`, and every
    /// component a stage names must be present.
    #[must_use]
    pub fn new(route: &'static str, components: BTreeMap<&'static str, Digest>) -> Option<Self> {
        let named: Vec<&str> = DiscoveryStage::ALL
            .into_iter()
            .flat_map(|stage| stage.own_components().iter().copied())
            .collect();
        if components.len() != named.len() || named.iter().any(|n| !components.contains_key(n)) {
            return None;
        }
        Some(Self { components, route })
    }

    /// The route label.
    #[must_use]
    pub const fn route(&self) -> &'static str {
        self.route
    }

    /// The digest of one named component.
    #[must_use]
    pub fn component(&self, name: &str) -> Option<&Digest> {
        self.components.get(name)
    }

    /// Every component in canonical (name) order.
    pub fn components(&self) -> impl Iterator<Item = (&'static str, &Digest)> + '_ {
        self.components.iter().map(|(name, digest)| (*name, digest))
    }

    /// The identity of `stage`: its own declared inputs and the identity of its upstream.
    #[must_use]
    pub fn identity(&self, stage: DiscoveryStage) -> StageIdentity {
        let upstream = stage.upstream().map(|up| self.identity(up));
        let mut parts: Vec<&[u8]> = Vec::new();
        parts.push(self.route.as_bytes());
        if let Some(up) = upstream.as_ref() {
            parts.push(up.as_bytes());
        }
        for name in stage.own_components() {
            parts.push(name.as_bytes());
            if let Some(digest) = self.components.get(name) {
                parts.push(digest);
            }
        }
        StageIdentity::of(&format!("discovery_stage.{}", stage.as_str()), &parts)
    }

    /// Names of the components that differ from `other`, in canonical order, restricted to
    /// those `stage` depends on (its own and every upstream stage's).
    #[must_use]
    pub fn changed_for(&self, stage: DiscoveryStage, other: &Self) -> Vec<&'static str> {
        let mut changed = Vec::new();
        let mut current = Some(stage);
        let mut chain = Vec::new();
        while let Some(s) = current {
            chain.push(s);
            current = s.upstream();
        }
        chain.reverse();
        for s in chain {
            for name in s.own_components() {
                if self.components.get(name) != other.components.get(name) {
                    changed.push(*name);
                }
            }
        }
        if self.route != other.route {
            changed.insert(0, "route");
        }
        changed
    }
}

/// What the plan says about one stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageDecision {
    /// The retained artifact for this identity is valid and is used as is.
    Reused,
    /// The stage is recomputed.
    Recomputed {
        /// The declared inputs that changed (`no_retained_state` when nothing was retained).
        because: Vec<&'static str>,
    },
}

impl StageDecision {
    /// Whether the stage is reused.
    #[must_use]
    pub const fn is_reused(&self) -> bool {
        matches!(self, Self::Reused)
    }
}

/// One stage's entry in a plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStage {
    /// The stage.
    pub stage: DiscoveryStage,
    /// Reused or recomputed, and why.
    pub decision: StageDecision,
    /// Identity the stage has under the requested inputs.
    pub identity: StageIdentity,
}

/// Per-stage reuse decisions for a request against the retained state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryPlan {
    entries: Vec<PlannedStage>,
}

impl DiscoveryPlan {
    /// Compare `requested` with the inputs of the retained run, if any.
    #[must_use]
    pub fn between(previous: Option<&StageInputs>, requested: &StageInputs) -> Self {
        let entries = DiscoveryStage::ALL
            .into_iter()
            .map(|stage| {
                let identity = requested.identity(stage);
                let decision = match previous {
                    None => StageDecision::Recomputed { because: vec!["no_retained_state"] },
                    Some(previous) if previous.identity(stage) == identity => StageDecision::Reused,
                    Some(previous) => StageDecision::Recomputed {
                        because: requested.changed_for(stage, previous),
                    },
                };
                PlannedStage { stage, decision, identity }
            })
            .collect();
        Self { entries }
    }

    /// Every stage in dependency order.
    #[must_use]
    pub fn entries(&self) -> &[PlannedStage] {
        &self.entries
    }

    /// The entry of `stage`.
    ///
    /// # Panics
    ///
    /// Never: a plan is built with an entry for every stage.
    #[must_use]
    pub fn entry(&self, stage: DiscoveryStage) -> &PlannedStage {
        self.entries.iter().find(|entry| entry.stage == stage).expect("a plan holds every stage")
    }

    /// The decision for `stage`.
    #[must_use]
    pub fn decision(&self, stage: DiscoveryStage) -> &StageDecision {
        &self.entry(stage).decision
    }

    /// Whether every stage is reused (nothing to execute).
    #[must_use]
    pub fn is_pure_reuse(&self) -> bool {
        self.entries.iter().all(|entry| entry.decision.is_reused())
    }

    /// Names of the recomputed stages.
    #[must_use]
    pub fn recomputed(&self) -> Vec<DiscoveryStage> {
        self.entries.iter().filter(|e| !e.decision.is_reused()).map(|e| e.stage).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(tweak: impl Fn(&'static str) -> u8) -> StageInputs {
        let mut components = BTreeMap::new();
        for stage in DiscoveryStage::ALL {
            for name in stage.own_components() {
                components.insert(*name, [tweak(name); 32]);
            }
        }
        StageInputs::new("static_pc", components).unwrap()
    }

    #[test]
    fn a_threshold_change_keeps_statistics_and_reruns_search_onward() {
        let base = inputs(|_| 1);
        let other = inputs(|name| if name == "thresholds" { 2 } else { 1 });
        let plan = DiscoveryPlan::between(Some(&base), &other);
        assert!(plan.decision(DiscoveryStage::Statistics).is_reused());
        assert_eq!(
            plan.decision(DiscoveryStage::Search),
            &StageDecision::Recomputed { because: vec!["thresholds"] }
        );
        assert!(!plan.decision(DiscoveryStage::Orientation).is_reused());
        assert!(!plan.decision(DiscoveryStage::GraphOutput).is_reused());
    }

    #[test]
    fn new_data_changes_every_stage_identity() {
        let base = inputs(|_| 1);
        let other = inputs(|name| if name == "data" { 9 } else { 1 });
        let plan = DiscoveryPlan::between(Some(&base), &other);
        assert_eq!(plan.recomputed(), DiscoveryStage::ALL.to_vec());
        assert_eq!(
            plan.decision(DiscoveryStage::GraphOutput),
            &StageDecision::Recomputed { because: vec!["data"] }
        );
    }

    #[test]
    fn identical_inputs_are_pure_reuse_and_no_state_is_a_recompute() {
        let base = inputs(|_| 1);
        assert!(DiscoveryPlan::between(Some(&base), &base).is_pure_reuse());
        let first = DiscoveryPlan::between(None, &base);
        assert_eq!(first.recomputed().len(), 4);
        assert_eq!(
            first.decision(DiscoveryStage::Statistics),
            &StageDecision::Recomputed { because: vec!["no_retained_state"] }
        );
    }

    #[test]
    fn orientation_settings_leave_statistics_and_search_reused() {
        let base = inputs(|_| 1);
        let other = inputs(|name| if name == "orientation_settings" { 3 } else { 1 });
        let plan = DiscoveryPlan::between(Some(&base), &other);
        assert!(plan.decision(DiscoveryStage::Statistics).is_reused());
        assert!(plan.decision(DiscoveryStage::Search).is_reused());
        assert!(!plan.decision(DiscoveryStage::Orientation).is_reused());
    }

    #[test]
    fn missing_or_unknown_components_are_refused() {
        let mut components = BTreeMap::new();
        components.insert("data", [0u8; 32]);
        assert!(StageInputs::new("static_pc", components).is_none());
    }
}

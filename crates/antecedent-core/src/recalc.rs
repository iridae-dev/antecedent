//! Stage model for selective recalculation (2.3 C2).
//!
//! A causal workflow is a small directed graph of stages: declared inputs (graph, query,
//! regime, evidence, populations, data snapshot, row design, treatment grid,
//! learner/folds/RNG, external studies, utility) and derived stages (identification,
//! provider request, prior, score or fit artifact, law, decision). Every stage carries one
//! [`StageIdentity`], the BLAKE3 digest of its declared inputs and of the identities of the
//! stages it depends on. [`RecalcPlan::plan`] compares the identities of a previous and a
//! requested workflow and says, per stage, whether it can be [`StageStatus::Reused`], must be
//! [`StageStatus::Recomputed`], or is [`StageStatus::Refused`], always naming the dependency
//! that determined the status.
//!
//! This module decides and explains; it computes nothing. It adds no route: identification
//! and estimation stay in the prepared-study and causal-contract architecture, and a stage is
//! only ever reused when an artifact for it exists on the declared side of the process
//! boundary (see [`Boundary`] and [`ResumeContext`]).
//!
//! Scope notes that matter to readers of a plan:
//!
//! - "Reused" never means "persistently cached". Within one process it names an artifact the
//!   caller still holds; across a process boundary it needs a portable fit, score, data
//!   snapshot or provider named in [`ResumeContext`]. An ordinary loaded result supplies none
//!   of them, so every derived stage is then recomputed or refused, never reused.
//! - Retargeting reuses frozen scores only when the estimator's retarget route is declared
//!   licensed ([`RetargetSupport::Licensed`]); otherwise a target change recomputes the
//!   score artifact, and an incompatible target is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fmt;

/// Number of external-study branches a workflow may carry.
pub const MAX_EXTERNAL_BRANCHES: u8 = 8;

const STAGE_IDENTITY_KEY: &str = "antecedent.recalc.stage_identity.v1";

/// BLAKE3 digest of a stage's declared dependency inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StageIdentity([u8; 32]);

impl StageIdentity {
    /// Identity of a stage that is not part of the workflow.
    pub const ABSENT: Self = Self([0; 32]);

    /// Digest `parts` under `label`. Each part is length-prefixed, so `["ab", "c"]` and
    /// `["a", "bc"]` differ.
    #[must_use]
    pub fn of(label: &str, parts: &[&[u8]]) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key(STAGE_IDENTITY_KEY);
        put(&mut hasher, label.as_bytes());
        hasher.update(&(parts.len() as u64).to_le_bytes());
        for part in parts {
            put(&mut hasher, part);
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// Raw digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// An identity from raw digest bytes, for a consumer that retained one.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Parse the 64-character lowercase hex encoding written by [`Self::to_hex`].
    #[must_use]
    pub fn from_hex(text: &str) -> Option<Self> {
        fn nibble(byte: u8) -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            }
        }
        let raw = text.as_bytes();
        if raw.len() != 64 {
            return None;
        }
        let mut out = [0_u8; 32];
        for (slot, pair) in out.iter_mut().zip(raw.chunks_exact(2)) {
            *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Some(Self(out))
    }

    /// Whether this is [`Self::ABSENT`].
    #[must_use]
    pub fn is_absent(&self) -> bool {
        self.0 == [0; 32]
    }

    /// Lowercase hex encoding (64 characters).
    #[must_use]
    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push(HEX[usize::from(byte >> 4)] as char);
            out.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        out
    }
}

impl fmt::Display for StageIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

fn put(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Index of one external-study branch, below [`MAX_EXTERNAL_BRANCHES`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Branch(u8);

impl Branch {
    /// Branch `index`, or `None` when it is not below [`MAX_EXTERNAL_BRANCHES`].
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if index < MAX_EXTERNAL_BRANCHES { Some(Self(index)) } else { None }
    }

    /// The branch index.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// One stage of the workflow. The declaration order is a topological order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Stage {
    /// Input: the accepted graph.
    Graph,
    /// Input: the causal query.
    Query,
    /// Input: the intervention regime.
    Regime,
    /// Input: the evidence and observation contract.
    Evidence,
    /// Input: the source population.
    SourcePopulation,
    /// Input: the target population and its row weights.
    TargetPopulation,
    /// Input: the data snapshot.
    DataSnapshot,
    /// Input: the row design (complete-case rows, weights, partitions).
    RowDesign,
    /// Input: the treatment grid or levels the answer is requested on.
    TreatmentGrid,
    /// Input: learner, folds, and RNG.
    LearnerFoldsRng,
    /// Input: the utility declaration.
    Utility,
    /// Input: one external study.
    ExternalStudy(Branch),
    /// Derived: the provider request and version for one external study.
    ProviderRequest(Branch),
    /// Derived: the prior built from one external study and the evidence.
    Prior(Branch),
    /// Derived: prepared identification and compilation.
    Identification,
    /// Derived: score or fit artifact (frozen cross-fitted scores, or a portable fit).
    ScoreArtifact,
    /// Derived: the law (retargeted estimate) of the requested quantity.
    Law,
    /// Derived: the decision.
    Decision,
}

impl Stage {
    /// Every possible stage in topological order (all [`MAX_EXTERNAL_BRANCHES`] branches).
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut stages = vec![
            Self::Graph,
            Self::Query,
            Self::Regime,
            Self::Evidence,
            Self::SourcePopulation,
            Self::TargetPopulation,
            Self::DataSnapshot,
            Self::RowDesign,
            Self::TreatmentGrid,
            Self::LearnerFoldsRng,
            Self::Utility,
        ];
        let branches = || (0..MAX_EXTERNAL_BRANCHES).map(Branch);
        stages.extend(branches().map(Self::ExternalStudy));
        stages.extend(branches().map(Self::ProviderRequest));
        stages.extend(branches().map(Self::Prior));
        stages.extend([Self::Identification, Self::ScoreArtifact, Self::Law, Self::Decision]);
        stages
    }

    /// Stable label, for example `external_study.1`.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Graph => "graph".to_string(),
            Self::Query => "query".to_string(),
            Self::Regime => "regime".to_string(),
            Self::Evidence => "evidence".to_string(),
            Self::SourcePopulation => "source_population".to_string(),
            Self::TargetPopulation => "target_population".to_string(),
            Self::DataSnapshot => "data_snapshot".to_string(),
            Self::RowDesign => "row_design".to_string(),
            Self::TreatmentGrid => "treatment_grid".to_string(),
            Self::LearnerFoldsRng => "learner_folds_rng".to_string(),
            Self::Utility => "utility".to_string(),
            Self::ExternalStudy(b) => format!("external_study.{}", b.0),
            Self::ProviderRequest(b) => format!("provider_request.{}", b.0),
            Self::Prior(b) => format!("prior.{}", b.0),
            Self::Identification => "identification".to_string(),
            Self::ScoreArtifact => "score_artifact".to_string(),
            Self::Law => "law".to_string(),
            Self::Decision => "decision".to_string(),
        }
    }

    /// Whether this stage is a declared input rather than a computation.
    #[must_use]
    pub const fn is_input(self) -> bool {
        !matches!(
            self,
            Self::ProviderRequest(_)
                | Self::Prior(_)
                | Self::Identification
                | Self::ScoreArtifact
                | Self::Law
                | Self::Decision
        )
    }

    /// Declared dependency edges, in order. The first edge is the stage's primary dependency.
    ///
    /// The score artifact depends on the target population only when the estimator's
    /// retarget route is not declared licensed: with a licensed retarget the frozen scores
    /// are target-independent and only the law reads the target.
    #[must_use]
    pub fn dependencies(self, retarget: RetargetSupport) -> Vec<Self> {
        match self {
            Self::Identification => vec![Self::Graph, Self::Query, Self::Regime, Self::Evidence],
            Self::ScoreArtifact => {
                let mut deps = vec![
                    Self::Identification,
                    Self::DataSnapshot,
                    Self::RowDesign,
                    Self::TreatmentGrid,
                    Self::LearnerFoldsRng,
                    Self::SourcePopulation,
                ];
                if retarget == RetargetSupport::NotDeclared {
                    deps.push(Self::TargetPopulation);
                }
                deps
            }
            Self::Law => vec![Self::ScoreArtifact, Self::TargetPopulation],
            Self::ProviderRequest(b) => {
                vec![
                    Self::ExternalStudy(b),
                    Self::Graph,
                    Self::Query,
                    Self::Regime,
                    Self::Evidence,
                    Self::SourcePopulation,
                    Self::TargetPopulation,
                    Self::TreatmentGrid,
                ]
            }
            Self::Prior(b) => {
                vec![
                    Self::ExternalStudy(b),
                    Self::Evidence,
                    Self::Graph,
                    Self::Query,
                    Self::Regime,
                    Self::SourcePopulation,
                    Self::TargetPopulation,
                    Self::TreatmentGrid,
                ]
            }
            Self::Decision => {
                let mut deps = vec![Self::Law, Self::Utility];
                for i in 0..MAX_EXTERNAL_BRANCHES {
                    deps.push(Self::Prior(Branch(i)));
                    deps.push(Self::ProviderRequest(Branch(i)));
                }
                deps
            }
            _ => Vec::new(),
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// The declared own-input digest of every stage that is part of a workflow.
///
/// A stage is part of a workflow exactly when an own digest was set for it. A stage's
/// effective identity also covers the effective identities of its dependencies (see
/// [`Self::effective`]).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StageIdentities {
    own: BTreeMap<Stage, StageIdentity>,
}

impl StageIdentities {
    /// An empty workflow.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `stage` with the digest of its own inputs.
    #[must_use]
    pub fn with(mut self, stage: Stage, own: StageIdentity) -> Self {
        self.own.insert(stage, own);
        self
    }

    /// Declare `stage` in place.
    pub fn set(&mut self, stage: Stage, own: StageIdentity) {
        self.own.insert(stage, own);
    }

    /// Remove `stage` from the workflow.
    pub fn remove(&mut self, stage: Stage) {
        self.own.remove(&stage);
    }

    /// Whether `stage` is part of the workflow.
    #[must_use]
    pub fn contains(&self, stage: Stage) -> bool {
        self.own.contains_key(&stage)
    }

    /// The own-input digest of `stage` ([`StageIdentity::ABSENT`] when it is not declared).
    #[must_use]
    pub fn own(&self, stage: Stage) -> StageIdentity {
        self.own.get(&stage).copied().unwrap_or(StageIdentity::ABSENT)
    }

    /// Effective identity of every declared stage under `retarget`: the digest of the stage's
    /// label, its own digest, and the label and effective identity of each dependency edge
    /// (an undeclared dependency contributes [`StageIdentity::ABSENT`]).
    #[must_use]
    pub fn effective(&self, retarget: RetargetSupport) -> BTreeMap<Stage, StageIdentity> {
        let mut out: BTreeMap<Stage, StageIdentity> = BTreeMap::new();
        for stage in Stage::all() {
            let Some(own) = self.own.get(&stage) else {
                continue;
            };
            let mut parts: Vec<Vec<u8>> = vec![own.0.to_vec()];
            for dep in stage.dependencies(retarget) {
                let id = out.get(&dep).copied().unwrap_or(StageIdentity::ABSENT);
                parts.push(dep.label().into_bytes());
                parts.push(id.0.to_vec());
            }
            let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
            out.insert(stage, StageIdentity::of(&stage.label(), &refs));
        }
        out
    }
}

/// Whether the estimator's frozen-score row-weight retarget route is licensed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetargetSupport {
    /// Licensed: a compatible target-weight change reweights frozen scores without a refit.
    Licensed,
    /// Not declared licensed: a target change recomputes the score artifact.
    NotDeclared,
    /// The requested weights are not a licensed function of the adjustment set: a target
    /// change is refused.
    Incompatible,
}

/// Whether the requested treatment grid or operation is inside the declared route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestSupport {
    /// On the declared grid and operation set.
    OnGrid,
    /// Off the declared grid.
    OffGrid {
        /// A separately licensed route that serves the request, when one exists.
        licensed_route: Option<&'static str>,
    },
    /// An operation the declared route does not support.
    Unsupported {
        /// A separately licensed route that serves the request, when one exists.
        licensed_route: Option<&'static str>,
    },
}

/// What a fresh process was handed to resume with.
///
/// An ordinary loaded result supplies none of these (all `false`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
// Independent availability flags; a state enum would not model their combinations.
#[allow(clippy::struct_excessive_bools)]
pub struct ResumeContext {
    /// A portable fitted predictor is available.
    pub portable_fit: bool,
    /// Portable frozen scores are available.
    pub portable_scores: bool,
    /// The portable scores bind the unchanged snapshot identity, without supplying its data.
    pub scores_snapshot_bound: bool,
    /// A compatible data snapshot was supplied.
    pub supplied_data: bool,
    /// A compatible provider (callback) was supplied.
    pub supplied_provider: bool,
}

/// Which side of a process boundary the plan is made on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Boundary {
    /// The previous run's artifacts are still held in this process.
    InProcess,
    /// A fresh process holding only what [`ResumeContext`] names.
    FreshProcess(ResumeContext),
}

/// The declared capabilities a plan is made under.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecalcCapabilities {
    /// Retarget license of the estimator.
    pub retarget: RetargetSupport,
    /// Whether the request is inside the declared route.
    pub request: RequestSupport,
    /// Process boundary.
    pub boundary: Boundary,
}

impl RecalcCapabilities {
    /// In-process, on-grid, with the given retarget license.
    #[must_use]
    pub const fn in_process(retarget: RetargetSupport) -> Self {
        Self { retarget, request: RequestSupport::OnGrid, boundary: Boundary::InProcess }
    }

    /// Fresh process with `resume`, on-grid, with the given retarget license.
    #[must_use]
    pub const fn fresh_process(retarget: RetargetSupport, resume: ResumeContext) -> Self {
        Self { retarget, request: RequestSupport::OnGrid, boundary: Boundary::FreshProcess(resume) }
    }
}

/// How a stage's own input or a dependency changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChangeKind {
    /// Present now, absent before.
    Added,
    /// Present before, absent now.
    Removed,
    /// Present in both with a different identity.
    Modified,
}

impl ChangeKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Removed => "removed",
            Self::Modified => "modified",
        }
    }
}

/// The dependency that forced a recomputation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChangedDependency {
    /// The stage's own declared input changed.
    Own {
        /// The stage (this stage).
        stage: Stage,
        /// How it changed.
        change: ChangeKind,
    },
    /// An upstream stage changed.
    Upstream {
        /// The direct dependency that carried the change (first changed edge).
        via: Stage,
        /// The stage whose own input actually changed.
        origin: Stage,
        /// How the origin changed.
        change: ChangeKind,
    },
    /// Nothing changed, but the artifact does not exist in this fresh process.
    FreshProcess,
}

/// A dependency a fresh process cannot supply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingDependency {
    /// A portable fitted predictor (and no scores or data to rebuild one).
    Fit,
    /// A compatible data snapshot.
    Data,
    /// A compatible provider (the external callback).
    Provider,
}

/// Why a stage was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefusalReason {
    /// A dependency the fresh process was not given.
    Unavailable {
        /// What is missing.
        missing: MissingDependency,
    },
    /// The treatment request is off the declared grid.
    OffGrid {
        /// A separately licensed route that serves the request, when one exists.
        licensed_route: Option<&'static str>,
    },
    /// The requested operation is outside the declared route.
    Unsupported {
        /// A separately licensed route that serves the request, when one exists.
        licensed_route: Option<&'static str>,
    },
    /// The target weights are not a licensed function of the adjustment set.
    RetargetIncompatible,
    /// A dependency was refused.
    Blocked {
        /// The first refused dependency.
        by: Stage,
    },
}

impl RefusalReason {
    /// Plain refusal detail literal, `<namespace>.<snake_case>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        match self {
            Self::Unavailable { missing: MissingDependency::Fit } => "recalc.unavailable_fit",
            Self::Unavailable { missing: MissingDependency::Data } => "recalc.unavailable_data",
            Self::Unavailable { missing: MissingDependency::Provider } => {
                "recalc.unavailable_provider"
            }
            Self::OffGrid { .. } => "recalc.off_grid_request",
            Self::Unsupported { .. } => "recalc.unsupported_request",
            Self::RetargetIncompatible => "recalc.retarget_incompatible",
            Self::Blocked { .. } => "recalc.blocked_by_refused_dependency",
        }
    }

    /// Registered reason code the refusal maps to.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Unavailable { missing: MissingDependency::Provider } => {
                "external_capability_missing"
            }
            Self::Unavailable { .. } => "score_table_unavailable",
            Self::OffGrid { .. } | Self::Unsupported { .. } => "route_not_supported",
            Self::RetargetIncompatible => "construction_not_licensed",
            Self::Blocked { .. } => "not_executed",
        }
    }
}

/// Per-stage outcome of a plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageStatus {
    /// The stage's artifact is valid as it is.
    Reused {
        /// The dependency whose unchanged identity licensed reuse: the stage itself for an
        /// input, otherwise its primary (first declared) dependency.
        dependency: Stage,
    },
    /// The stage must be recomputed.
    Recomputed {
        /// What forced it.
        because: ChangedDependency,
    },
    /// The stage cannot be served.
    Refused {
        /// Why.
        reason: RefusalReason,
    },
}

impl StageStatus {
    /// `reused`, `recomputed`, or `refused`.
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::Reused { .. } => "reused",
            Self::Recomputed { .. } => "recomputed",
            Self::Refused { .. } => "refused",
        }
    }
}

impl fmt::Display for ChangedDependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Own { stage, change } => write!(f, "own:{stage}:{}", change.as_str()),
            Self::Upstream { via, origin, change } => {
                write!(f, "upstream:{via}<-{origin}:{}", change.as_str())
            }
            Self::FreshProcess => f.write_str("fresh_process"),
        }
    }
}

impl fmt::Display for RefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.detail())?;
        match self {
            Self::OffGrid { licensed_route: Some(route) }
            | Self::Unsupported { licensed_route: Some(route) } => write!(f, "[{route}]"),
            Self::Blocked { by } => write!(f, "[{by}]"),
            _ => Ok(()),
        }
    }
}

impl fmt::Display for StageStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reused { dependency } => write!(f, "reused({dependency})"),
            Self::Recomputed { because } => write!(f, "recomputed({because})"),
            Self::Refused { reason } => write!(f, "refused({reason})"),
        }
    }
}

/// One row of a plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageEntry {
    /// The stage.
    pub stage: Stage,
    /// Its status.
    pub status: StageStatus,
    /// Its effective identity in the requested workflow.
    pub identity: StageIdentity,
}

/// Per-stage reuse, recompute, or refuse table for a requested workflow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecalcPlan {
    entries: Vec<StageEntry>,
}

impl RecalcPlan {
    /// Plan `requested` against `previous` under `capabilities`.
    ///
    /// Stages are visited in topological order; only stages declared in `requested` appear.
    /// See the module docs and [`StageStatus`] for the rules; in a fresh process
    /// ([`Boundary::FreshProcess`]) no derived stage is ever reused except a score artifact
    /// backed by portable scores, a stage whose artifact is missing is refused with
    /// [`RefusalReason::Unavailable`] when its inputs were not supplied, and recomputed
    /// otherwise.
    #[must_use]
    pub fn plan(
        previous: &StageIdentities,
        requested: &StageIdentities,
        capabilities: &RecalcCapabilities,
    ) -> Self {
        let mut planner = Planner {
            previous,
            requested,
            caps: capabilities,
            statuses: BTreeMap::new(),
            changed: BTreeMap::new(),
        };
        let identities = requested.effective(capabilities.retarget);
        let mut entries = Vec::new();
        for stage in Stage::all() {
            if !requested.contains(stage) {
                continue;
            }
            let status = planner.decide(stage);
            planner.record(stage, status);
            let identity = identities.get(&stage).copied().unwrap_or(StageIdentity::ABSENT);
            entries.push(StageEntry { stage, status, identity });
        }
        Self { entries }
    }

    /// Rows in topological order.
    #[must_use]
    pub fn entries(&self) -> &[StageEntry] {
        &self.entries
    }

    /// Status of `stage`, when it is part of the requested workflow.
    #[must_use]
    pub fn status(&self, stage: Stage) -> Option<&StageStatus> {
        self.entries.iter().find(|e| e.stage == stage).map(|e| &e.status)
    }

    /// Effective identity of `stage` in the requested workflow.
    #[must_use]
    pub fn identity(&self, stage: Stage) -> Option<StageIdentity> {
        self.entries.iter().find(|e| e.stage == stage).map(|e| e.identity)
    }

    /// Stages with status `reused`.
    #[must_use]
    pub fn reused(&self) -> Vec<Stage> {
        self.with_tag("reused")
    }

    /// Stages with status `recomputed`.
    #[must_use]
    pub fn recomputed(&self) -> Vec<Stage> {
        self.with_tag("recomputed")
    }

    /// Stages with status `refused`.
    #[must_use]
    pub fn refused(&self) -> Vec<Stage> {
        self.with_tag("refused")
    }

    /// Recomputed stages that are computations, not declared inputs.
    #[must_use]
    pub fn recomputed_computations(&self) -> Vec<Stage> {
        self.recomputed().into_iter().filter(|s| !s.is_input()).collect()
    }

    /// The first refused stage and its reason.
    #[must_use]
    pub fn first_refusal(&self) -> Option<(Stage, RefusalReason)> {
        self.entries.iter().find_map(|e| match e.status {
            StageStatus::Refused { reason } => Some((e.stage, reason)),
            _ => None,
        })
    }

    /// Whether the plan holds no refused stage.
    #[must_use]
    pub fn is_executable(&self) -> bool {
        self.first_refusal().is_none()
    }

    /// Canonical digest of the whole table (stage, status, identity).
    #[must_use]
    pub fn canonical_identity(&self) -> StageIdentity {
        let parts: Vec<Vec<u8>> = self
            .entries
            .iter()
            .flat_map(|e| {
                [
                    e.stage.label().into_bytes(),
                    e.status.to_string().into_bytes(),
                    e.identity.0.to_vec(),
                ]
            })
            .collect();
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        StageIdentity::of("recalc_plan", &refs)
    }

    fn with_tag(&self, tag: &str) -> Vec<Stage> {
        self.entries.iter().filter(|e| e.status.tag() == tag).map(|e| e.stage).collect()
    }
}

struct Planner<'a> {
    previous: &'a StageIdentities,
    requested: &'a StageIdentities,
    caps: &'a RecalcCapabilities,
    statuses: BTreeMap<Stage, StageStatus>,
    /// Stage -> (origin stage whose own input changed, how it changed), for stages whose
    /// identity changed. Recomputations caused only by a fresh process are not recorded.
    changed: BTreeMap<Stage, (Stage, ChangeKind)>,
}

impl Planner<'_> {
    fn record(&mut self, stage: Stage, status: StageStatus) {
        if let StageStatus::Recomputed { because } = status {
            match because {
                ChangedDependency::Own { stage: own, change } => {
                    self.changed.insert(stage, (own, change));
                }
                ChangedDependency::Upstream { origin, change, .. } => {
                    self.changed.insert(stage, (origin, change));
                }
                ChangedDependency::FreshProcess => {}
            }
        }
        self.statuses.insert(stage, status);
    }

    fn decide(&self, stage: Stage) -> StageStatus {
        let deps = stage.dependencies(self.caps.retarget);
        if let Some(reason) = self.request_refusal(stage) {
            return StageStatus::Refused { reason };
        }
        let resume = match self.caps.boundary {
            Boundary::FreshProcess(resume) => Some(resume),
            Boundary::InProcess => None,
        };
        if let Some(resume) = resume {
            if let Some(status) = self.fresh_status(stage, &deps, resume) {
                return status;
            }
        }
        if let Some(by) = self.refused_dependency(&deps, None) {
            return StageStatus::Refused { reason: RefusalReason::Blocked { by } };
        }
        if stage == Stage::Law
            && self.caps.retarget == RetargetSupport::Incompatible
            && self.changed.contains_key(&Stage::TargetPopulation)
        {
            return StageStatus::Refused { reason: RefusalReason::RetargetIncompatible };
        }
        let status = self.by_change(stage, &deps);
        if resume.is_some() && !stage.is_input() && matches!(status, StageStatus::Reused { .. }) {
            return StageStatus::Recomputed { because: ChangedDependency::FreshProcess };
        }
        status
    }

    fn request_refusal(&self, stage: Stage) -> Option<RefusalReason> {
        if stage != Stage::TreatmentGrid {
            return None;
        }
        match self.caps.request {
            RequestSupport::OnGrid => None,
            RequestSupport::OffGrid { licensed_route } => {
                Some(RefusalReason::OffGrid { licensed_route })
            }
            RequestSupport::Unsupported { licensed_route } => {
                Some(RefusalReason::Unsupported { licensed_route })
            }
        }
    }

    fn own_change(&self, stage: Stage) -> Option<ChangeKind> {
        match (self.previous.contains(stage), self.requested.contains(stage)) {
            (false, true) => Some(ChangeKind::Added),
            (true, true) if self.previous.own(stage) != self.requested.own(stage) => {
                Some(ChangeKind::Modified)
            }
            _ => None,
        }
    }

    fn by_change(&self, stage: Stage, deps: &[Stage]) -> StageStatus {
        if let Some(change) = self.own_change(stage) {
            return StageStatus::Recomputed { because: ChangedDependency::Own { stage, change } };
        }
        for &dep in deps {
            if let Some(&(origin, change)) = self.changed.get(&dep) {
                return StageStatus::Recomputed {
                    because: ChangedDependency::Upstream { via: dep, origin, change },
                };
            }
            if self.previous.contains(dep) && !self.requested.contains(dep) {
                return StageStatus::Recomputed {
                    because: ChangedDependency::Upstream {
                        via: dep,
                        origin: dep,
                        change: ChangeKind::Removed,
                    },
                };
            }
        }
        StageStatus::Reused { dependency: deps.first().copied().unwrap_or(stage) }
    }

    fn refused_dependency(&self, deps: &[Stage], skip: Option<Stage>) -> Option<Stage> {
        deps.iter().copied().find(|dep| {
            Some(*dep) != skip
                && matches!(self.statuses.get(dep), Some(StageStatus::Refused { .. }))
        })
    }

    fn fresh_status(
        &self,
        stage: Stage,
        deps: &[Stage],
        resume: ResumeContext,
    ) -> Option<StageStatus> {
        let unavailable =
            |missing| StageStatus::Refused { reason: RefusalReason::Unavailable { missing } };
        match stage {
            Stage::DataSnapshot
                if !(resume.supplied_data
                    || resume.portable_scores
                        && resume.scores_snapshot_bound
                        && self.previous.contains(Stage::DataSnapshot)
                        && self.own_change(Stage::DataSnapshot).is_none()) =>
            {
                Some(unavailable(MissingDependency::Data))
            }
            Stage::ProviderRequest(_) if !resume.supplied_provider => {
                Some(unavailable(MissingDependency::Provider))
            }
            Stage::ScoreArtifact => Some(self.fresh_scores(deps, resume)),
            _ => None,
        }
    }

    /// Score artifact in a fresh process: reusable only as portable scores whose identity is
    /// unchanged; rebuilt from supplied data otherwise; refused when neither exists.
    fn fresh_scores(&self, deps: &[Stage], resume: ResumeContext) -> StageStatus {
        if let Some(by) = self.refused_dependency(deps, Some(Stage::DataSnapshot)) {
            return StageStatus::Refused { reason: RefusalReason::Blocked { by } };
        }
        // A data snapshot that differs from the one the scores were frozen on is a change
        // even when the snapshot stage itself is refused (so it never entered the changed
        // set): the frozen scores do not describe that data and cannot be claimed reused.
        if !resume.supplied_data
            && self.own_change(Stage::DataSnapshot) == Some(ChangeKind::Modified)
        {
            return StageStatus::Refused {
                reason: RefusalReason::Unavailable { missing: MissingDependency::Data },
            };
        }
        let by_change = self.by_change(Stage::ScoreArtifact, deps);
        let unchanged = matches!(by_change, StageStatus::Reused { .. });
        if unchanged && resume.portable_scores {
            return by_change;
        }
        if resume.supplied_data {
            return if unchanged {
                StageStatus::Recomputed { because: ChangedDependency::FreshProcess }
            } else {
                by_change
            };
        }
        let missing = if resume.portable_fit || resume.portable_scores {
            MissingDependency::Data
        } else {
            MissingDependency::Fit
        };
        StageStatus::Refused { reason: RefusalReason::Unavailable { missing } }
    }
}

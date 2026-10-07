//! Selective recalculation with a visible receipt (2.3 C2).
//!
//! [`antecedent_core::recalc`] decides, per stage, whether a workflow can reuse, must
//! recompute, or must refuse. This module executes one minimal real route under that plan
//! and counts what actually ran, so reuse is proven by counts and not by the plan alone.
//!
//! The route is the cross-fitted AIPW average effect over a binary treatment, driven through
//! the existing prepared-study architecture:
//!
//! - `Study::prepare` identifies the query and builds the frozen score table (stages
//!   `Identification` and `ScoreArtifact`);
//! - [`PreparedStudy::refresh`] rebuilds only the score table on new rows or a new fold seed,
//!   re-using the retained physical plan (stage `ScoreArtifact` without `Identification`);
//! - [`PreparedStudy::retarget`] reweights the frozen scores (stage `Law`), with no refit;
//! - a small net-benefit rule `benefit_per_unit * ate - cost` is the `Decision`.
//!
//! What is instrumented:
//!
//! - **Fold fits.** Each prepare or refresh runs inside one
//!   [`antecedent_estimate::CrossfitNuisanceCache`] scope owned by the [`ReceiptRecorder`].
//!   The cache counts every cross-fitted nuisance set it actually fit (`propensity_fits`,
//!   `outcome_fits`; reuse inside the call is counted separately and not as a fit). One fit
//!   set is `DEFAULT_AIPW_FOLDS` fold fits, so a first run records `5 folds x 2 nuisance
//!   sets = 10` fold fits. The cache lives only as long as the recorder (one call) and is
//!   not a persistent fit cache; reuse across calls is the retained [`PreparedStudy`] and its
//!   frozen score table, and nothing else.
//! - **Identifications, score computations, reweights, decisions.** Counted at the call sites
//!   of `Study::prepare` (identification), the score-table build (`prepare` or `refresh`),
//!   `PreparedStudy::retarget`, and the decision rule. These are call-level counts: they say
//!   the wrapper invoked the operation, and the fold-fit counter, which is read from the
//!   estimator's own cache, independently confirms that a refit really happened.
//!
//! [`RecalcReceipt`] refuses to build when a count contradicts a status: a reused stage that
//! did work, or a recomputed computation that did none.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, CrossfitNuisanceCache, DEFAULT_AIPW_FOLDS};

use crate::error::CausalError;
use crate::graph::{Dag, DenseNodeId};

use super::builder::RefuteSuite;
use super::execute::Study;
use super::prepared::PreparedStudy;

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// One kind of counted work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Counter {
    /// A query was identified (`Study::prepare`).
    Identification,
    /// One cross-fitted nuisance fold fit.
    FoldFit,
    /// A frozen score table was built.
    ScoreComputation,
    /// Frozen scores were reweighted into a law.
    Reweight,
    /// The decision rule was evaluated.
    Decision,
}

impl Counter {
    const ALL: [Self; 5] = [
        Self::Identification,
        Self::FoldFit,
        Self::ScoreComputation,
        Self::Reweight,
        Self::Decision,
    ];

    /// The one stage this work belongs to.
    #[must_use]
    pub const fn stage(self) -> Stage {
        match self {
            Self::Identification => Stage::Identification,
            Self::FoldFit | Self::ScoreComputation => Stage::ScoreArtifact,
            Self::Reweight => Stage::Law,
            Self::Decision => Stage::Decision,
        }
    }
}

/// Work counted for one stage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StageCounts {
    /// Identifications performed.
    pub identifications: u64,
    /// Cross-fitted nuisance fold fits performed.
    pub fold_fits: u64,
    /// Score tables built.
    pub score_computations: u64,
    /// Reweights of frozen scores performed.
    pub reweights: u64,
    /// Decision evaluations performed.
    pub decisions: u64,
}

impl StageCounts {
    /// The count of one kind of work.
    #[must_use]
    pub const fn get(&self, counter: Counter) -> u64 {
        match counter {
            Counter::Identification => self.identifications,
            Counter::FoldFit => self.fold_fits,
            Counter::ScoreComputation => self.score_computations,
            Counter::Reweight => self.reweights,
            Counter::Decision => self.decisions,
        }
    }

    /// Sum over every kind of work.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.identifications
            .saturating_add(self.fold_fits)
            .saturating_add(self.score_computations)
            .saturating_add(self.reweights)
            .saturating_add(self.decisions)
    }

    fn add(&mut self, counter: Counter, n: u64) {
        let slot = match counter {
            Counter::Identification => &mut self.identifications,
            Counter::FoldFit => &mut self.fold_fits,
            Counter::ScoreComputation => &mut self.score_computations,
            Counter::Reweight => &mut self.reweights,
            Counter::Decision => &mut self.decisions,
        };
        *slot = slot.saturating_add(n);
    }

    fn merged(mut self, other: &Self) -> Self {
        for counter in Counter::ALL {
            self.add(counter, other.get(counter));
        }
        self
    }
}

/// Counts actual work per stage during one call.
#[derive(Debug)]
pub struct ReceiptRecorder {
    counts: BTreeMap<Stage, StageCounts>,
    nuisances: Arc<CrossfitNuisanceCache>,
}

impl Default for ReceiptRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReceiptRecorder {
    /// A recorder with no counts and a fresh per-call nuisance cache.
    #[must_use]
    pub fn new() -> Self {
        Self { counts: BTreeMap::new(), nuisances: CrossfitNuisanceCache::new() }
    }

    /// Count `n` units of `counter` against its stage.
    pub fn record(&mut self, counter: Counter, n: u64) {
        self.counts.entry(counter.stage()).or_default().add(counter, n);
    }

    /// Counts recorded against `stage`.
    #[must_use]
    pub fn counts(&self, stage: Stage) -> StageCounts {
        self.counts.get(&stage).copied().unwrap_or_default()
    }

    /// Run `work` with this recorder's nuisance cache active on the calling thread and record
    /// every nuisance fit set the cache actually fit as `DEFAULT_AIPW_FOLDS` fold fits.
    pub fn measured<R>(&mut self, work: impl FnOnce() -> R) -> R {
        let before = self.nuisances.stats();
        let (out, _used) = self.nuisances.scope(work);
        let after = self.nuisances.stats();
        let sets = after
            .propensity_fits
            .saturating_sub(before.propensity_fits)
            .saturating_add(after.outcome_fits.saturating_sub(before.outcome_fits));
        let folds = to_u64(DEFAULT_AIPW_FOLDS);
        self.record(Counter::FoldFit, to_u64(sets).saturating_mul(folds));
        out
    }

    /// Combine the counts with `plan` into a verified receipt.
    ///
    /// # Errors
    ///
    /// [`ReceiptError`] when a count contradicts the plan's status for a stage.
    pub fn finish(self, plan: &RecalcPlan) -> Result<RecalcReceipt, ReceiptError> {
        let entries = plan
            .entries()
            .iter()
            .map(|e| ReceiptEntry {
                stage: e.stage,
                status: e.status,
                identity: e.identity,
                counts: self.counts(e.stage),
            })
            .collect();
        RecalcReceipt::new(entries)
    }
}

/// One stage of a receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReceiptEntry {
    /// The stage.
    pub stage: Stage,
    /// Its planned status.
    pub status: StageStatus,
    /// Its effective identity in the requested workflow.
    pub identity: StageIdentity,
    /// The work counted for it.
    pub counts: StageCounts,
}

/// Why a receipt could not be built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReceiptError {
    /// Two entries named the same stage.
    DuplicateStage(Stage),
    /// A reused or refused stage, or one that is not a computation, counted work.
    UnexpectedWork {
        /// The stage.
        stage: Stage,
    },
    /// A recomputed computation counted none of its required work.
    MissingWork {
        /// The stage.
        stage: Stage,
        /// The work that should have been counted.
        counter: Counter,
    },
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStage(stage) => write!(f, "receipt names stage {stage} twice"),
            Self::UnexpectedWork { stage } => {
                write!(f, "stage {stage} counted work its status does not allow")
            }
            Self::MissingWork { stage, counter } => {
                write!(f, "recomputed stage {stage} counted no {counter:?} work")
            }
        }
    }
}

impl std::error::Error for ReceiptError {}

/// The work a recomputed stage must have counted.
const fn required_work(stage: Stage) -> &'static [Counter] {
    match stage {
        Stage::Identification => &[Counter::Identification],
        Stage::ScoreArtifact => &[Counter::FoldFit, Counter::ScoreComputation],
        Stage::Law => &[Counter::Reweight],
        Stage::Decision => &[Counter::Decision],
        _ => &[],
    }
}

/// Canonical, order-independent record of what a call reused, recomputed, refused and did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecalcReceipt {
    entries: Vec<ReceiptEntry>,
    totals: StageCounts,
}

impl RecalcReceipt {
    /// Build a receipt from entries in any order: entries are sorted into topological order,
    /// duplicates are refused, and every count is checked against its status.
    ///
    /// # Errors
    ///
    /// [`ReceiptError`] for a duplicate stage or a count that contradicts a status.
    pub fn new(mut entries: Vec<ReceiptEntry>) -> Result<Self, ReceiptError> {
        entries.sort_by_key(|e| e.stage);
        if let Some(pair) = entries.windows(2).find(|w| w[0].stage == w[1].stage) {
            return Err(ReceiptError::DuplicateStage(pair[0].stage));
        }
        let mut totals = StageCounts::default();
        for entry in &entries {
            verify_entry(entry)?;
            totals = totals.merged(&entry.counts);
        }
        Ok(Self { entries, totals })
    }

    /// Entries in topological order.
    #[must_use]
    pub fn entries(&self) -> &[ReceiptEntry] {
        &self.entries
    }

    /// The entry of `stage`.
    #[must_use]
    pub fn entry(&self, stage: Stage) -> Option<&ReceiptEntry> {
        self.entries.iter().find(|e| e.stage == stage)
    }

    /// Totals over every stage.
    #[must_use]
    pub const fn totals(&self) -> &StageCounts {
        &self.totals
    }

    /// `(stage, status)` rows in topological order.
    #[must_use]
    pub fn status_table(&self) -> Vec<(Stage, StageStatus)> {
        self.entries.iter().map(|e| (e.stage, e.status)).collect()
    }

    /// Canonical identity over stage, status, stage identity and counts. It depends only on
    /// the set of entries, not on the order they were supplied in.
    #[must_use]
    pub fn identity(&self) -> StageIdentity {
        let parts: Vec<Vec<u8>> = self
            .entries
            .iter()
            .flat_map(|e| {
                let mut counts = Vec::with_capacity(40);
                for counter in Counter::ALL {
                    counts.extend_from_slice(&e.counts.get(counter).to_le_bytes());
                }
                [
                    e.stage.label().into_bytes(),
                    e.status.to_string().into_bytes(),
                    e.identity.as_bytes().to_vec(),
                    counts,
                ]
            })
            .collect();
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        StageIdentity::of("recalc_receipt", &refs)
    }
}

fn verify_entry(entry: &ReceiptEntry) -> Result<(), ReceiptError> {
    let stage = entry.stage;
    let required = match entry.status {
        StageStatus::Recomputed { .. } => required_work(stage),
        StageStatus::Reused { .. } | StageStatus::Refused { .. } => &[],
    };
    for counter in Counter::ALL {
        let n = entry.counts.get(counter);
        let allowed =
            matches!(entry.status, StageStatus::Recomputed { .. }) && counter.stage() == stage;
        if n > 0 && !allowed {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
    }
    for &counter in required {
        if entry.counts.get(counter) == 0 {
            return Err(ReceiptError::MissingWork { stage, counter });
        }
    }
    Ok(())
}

/// Row weights declaring a target population, with their declared parents.
#[derive(Clone, Debug)]
pub struct TargetWeights {
    /// One non-negative weight per complete-case row, in row order.
    pub weights: Vec<f64>,
    /// Declared parents of the weights (must lie in the adjustment set).
    pub depends_on: Vec<VariableId>,
}

/// A net-benefit rule: `benefit_per_unit * ate - cost`, treat when positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UtilitySpec {
    /// Benefit per unit of average effect.
    pub benefit_per_unit: f64,
    /// Fixed cost of acting.
    pub cost: f64,
}

/// The law of the requested quantity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LawValue {
    /// Weighted mean of the frozen contrast scores.
    pub ate: f64,
    /// Influence-function standard error.
    pub std_error: f64,
}

/// The decision under a utility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionValue {
    /// `benefit_per_unit * ate - cost`.
    pub net_benefit: f64,
    /// Whether the net benefit is positive.
    pub treat: bool,
}

/// One cross-fitted AIPW average-effect request over raw columns and declared edges.
#[derive(Clone, Debug)]
pub struct RecalcRequest {
    /// Named data columns; the variable id of a column is its position.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Directed edges `(from, to)` over column positions.
    pub edges: Vec<(u32, u32)>,
    /// Treatment column position.
    pub treatment: u32,
    /// Outcome column position.
    pub outcome: u32,
    /// Estimator configuration. Bootstrap replicates should be zero.
    pub estimator: AipwAte,
    /// Target population weights; `None` is the observed population.
    pub target: Option<TargetWeights>,
    /// Utility rule.
    pub utility: UtilitySpec,
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}

fn digest(label: &str, parts: &[Vec<u8>]) -> StageIdentity {
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    StageIdentity::of(label, &refs)
}

fn literal(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}

impl RecalcRequest {
    /// Declared own-input identity of every stage of this request under `ctx`.
    #[must_use]
    pub fn identities(&self, ctx: &ExecutionContext) -> StageIdentities {
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        let edge_bytes: Vec<u8> =
            edges.iter().flat_map(|(a, b)| [a.to_le_bytes(), b.to_le_bytes()]).flatten().collect();
        let mut data_parts = Vec::new();
        for (name, values) in &self.columns {
            data_parts.push(name.as_bytes().to_vec());
            data_parts.push(f64_bytes(values));
        }
        let rows = self.columns.first().map_or(0, |(_, v)| v.len());
        let folds = to_u64(DEFAULT_AIPW_FOLDS);
        let query = AverageEffectQuery::binary_ate(
            VariableId::from_raw(self.treatment),
            VariableId::from_raw(self.outcome),
        );
        let target = self.target.as_ref().map_or_else(
            || literal("target_population", "all_observed"),
            |t| {
                let parents: Vec<u8> =
                    t.depends_on.iter().flat_map(|v| v.raw().to_le_bytes()).collect();
                digest("target_population", &[f64_bytes(&t.weights), parents])
            },
        );
        let mut ids = StageIdentities::new();
        ids.set(
            Stage::Graph,
            digest("graph", &[to_u64(self.columns.len()).to_le_bytes().to_vec(), edge_bytes]),
        );
        ids.set(Stage::Query, literal("query", &format!("{query:?}")));
        ids.set(Stage::Regime, literal("regime", "static.binary"));
        ids.set(Stage::Evidence, literal("evidence", "observational.no_external"));
        ids.set(Stage::SourcePopulation, literal("source_population", "all_observed"));
        ids.set(Stage::TargetPopulation, target);
        ids.set(Stage::DataSnapshot, digest("data_snapshot", &data_parts));
        ids.set(Stage::RowDesign, literal("row_design", &format!("complete_case.rows={rows}")));
        ids.set(Stage::TreatmentGrid, literal("treatment_grid", "binary.0_1"));
        ids.set(
            Stage::LearnerFoldsRng,
            digest(
                "learner_folds_rng",
                &[
                    format!("{:?}", self.estimator).into_bytes(),
                    folds.to_le_bytes().to_vec(),
                    ctx.rng.master_seed().to_le_bytes().to_vec(),
                ],
            ),
        );
        ids.set(
            Stage::Utility,
            digest("utility", &[f64_bytes(&[self.utility.benefit_per_unit, self.utility.cost])]),
        );
        ids.set(Stage::Identification, literal("identification", "backdoor.prepared"));
        ids.set(Stage::ScoreArtifact, literal("score_artifact", "crossfit_aipw.scores"));
        ids.set(Stage::Law, literal("law", "weighted_mean_of_scores"));
        ids.set(Stage::Decision, literal("decision", "net_benefit"));
        ids
    }

    fn graph(&self) -> Result<Dag, RecalcRunError> {
        let n = u32::try_from(self.columns.len())
            .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
        let mut graph = Dag::with_variables(n);
        for &(from, to) in &self.edges {
            graph
                .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
        }
        Ok(graph)
    }

    fn data(&self) -> Result<TabularData, RecalcRunError> {
        let borrowed: Vec<(&str, &[f64])> =
            self.columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        TabularData::from_f64_columns(borrowed)
            .map_err(|_| RecalcRunError::Request("recalc.invalid_data"))
    }

    fn study(&self, data: TabularData) -> Result<Study, RecalcRunError> {
        let query = AverageEffectQuery::binary_ate(
            VariableId::from_raw(self.treatment),
            VariableId::from_raw(self.outcome),
        );
        Ok(Study::tabular(data)
            .graph(self.graph()?)
            .query(query)
            .estimator(self.estimator.clone())
            .refute(RefuteSuite::None)
            .build()?)
    }
}

/// Why a recalculation did not run.
#[derive(Debug)]
pub enum RecalcRunError {
    /// The plan contains a refused stage; nothing was executed.
    Refused(Box<RecalcPlan>),
    /// The request itself is malformed (plain `recalc.<snake_case>` detail).
    Request(&'static str),
    /// The session holds no live artifact for a stage the plan says to reuse.
    NoLiveState(Stage),
    /// A study operation failed.
    Execution(CausalError),
    /// The receipt contradicted the plan.
    Receipt(ReceiptError),
}

impl fmt::Display for RecalcRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(plan) => match plan.first_refusal() {
                Some((stage, reason)) => write!(f, "recalculation refused at {stage}: {reason}"),
                None => f.write_str("recalculation refused"),
            },
            Self::Request(detail) => f.write_str(detail),
            Self::NoLiveState(stage) => write!(f, "no live artifact to reuse for stage {stage}"),
            Self::Execution(error) => write!(f, "{error}"),
            Self::Receipt(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RecalcRunError {}

impl RecalcRunError {
    /// The registered reason code of a malformed request (`invalid_argument`); other
    /// failures carry their own codes through the plan, the study or the receipt.
    #[must_use]
    pub fn request_reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Request(_) => Some(antecedent_core::reason_code!("invalid_argument")),
            _ => None,
        }
    }
}

impl From<CausalError> for RecalcRunError {
    fn from(error: CausalError) -> Self {
        Self::Execution(error)
    }
}

impl From<ReceiptError> for RecalcRunError {
    fn from(error: ReceiptError) -> Self {
        Self::Receipt(error)
    }
}

struct Live {
    prepared: PreparedStudy,
    law: LawValue,
    decision: DecisionValue,
}

/// What a caller holds between calls: the retained prepared study and the last identities.
///
/// A session built with [`Self::resume`] models a fresh process: it knows the previous
/// identities (what an ordinary loaded result keeps) but holds no live artifact, so it can
/// never reuse a derived stage.
pub struct RecalcSession {
    live: Option<Live>,
    previous: StageIdentities,
    resume: Option<ResumeContext>,
    retarget: RetargetSupport,
    request: RequestSupport,
}

impl Default for RecalcSession {
    fn default() -> Self {
        Self::new()
    }
}

impl RecalcSession {
    /// An empty in-process session with the retarget route declared licensed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            live: None,
            previous: StageIdentities::new(),
            resume: None,
            retarget: RetargetSupport::Licensed,
            request: RequestSupport::OnGrid,
        }
    }

    /// A fresh-process session that knows only `previous` identities and what `resume` names.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        Self { previous, resume: Some(resume), ..Self::new() }
    }

    /// Declare the estimator's retarget license for later plans.
    pub fn set_retarget_support(&mut self, retarget: RetargetSupport) {
        self.retarget = retarget;
    }

    /// Declare whether the next request is inside the route's declared grid.
    pub fn set_request_support(&mut self, request: RequestSupport) {
        self.request = request;
    }

    /// Identities of the last successful run (or the resumed ones).
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }

    /// Whether the session holds a live prepared study.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }

    /// The frozen score table of the live prepared study, for independent checks.
    #[must_use]
    pub fn score_table(&self) -> Option<&antecedent_estimate::ScoreTable> {
        self.live.as_ref().and_then(|live| live.prepared.score_table())
    }

    /// The plan `request` would run under, without running it.
    #[must_use]
    pub fn plan(&self, request: &RecalcRequest, ctx: &ExecutionContext) -> RecalcPlan {
        let boundary = match self.resume {
            Some(resume) if self.live.is_none() => Boundary::FreshProcess(resume),
            _ => Boundary::InProcess,
        };
        let caps = RecalcCapabilities { retarget: self.retarget, request: self.request, boundary };
        RecalcPlan::plan(&self.previous, &request.identities(ctx), &caps)
    }
}

/// Result of one recalculation.
#[derive(Clone, Debug)]
pub struct RecalcOutcome {
    /// The plan the call ran under.
    pub plan: RecalcPlan,
    /// What actually ran.
    pub receipt: RecalcReceipt,
    /// The law of the requested quantity.
    pub law: LawValue,
    /// The decision.
    pub decision: DecisionValue,
}

fn is_recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}

/// Identification and score table: `Study::prepare` when identification must be redone,
/// `refresh` (retained plan, new score table) when only the scores must be rebuilt, nothing
/// when both are reused.
fn score_stages(
    live: Option<Live>,
    plan: &RecalcPlan,
    request: &RecalcRequest,
    recorder: &mut ReceiptRecorder,
    ctx: &ExecutionContext,
) -> Result<(PreparedStudy, Option<(LawValue, DecisionValue)>), RecalcRunError> {
    if is_recomputed(plan, Stage::Identification) {
        let study = request.study(request.data()?)?;
        recorder.record(Counter::Identification, 1);
        recorder.record(Counter::ScoreComputation, 1);
        let prepared = recorder.measured(|| study.prepare(ctx))?;
        return Ok((prepared, None));
    }
    let Some(live) = live else {
        return Err(RecalcRunError::NoLiveState(Stage::Identification));
    };
    let Live { mut prepared, law, decision } = live;
    if is_recomputed(plan, Stage::ScoreArtifact) {
        let data = request.data()?;
        recorder.record(Counter::ScoreComputation, 1);
        recorder.measured(|| prepared.refresh(data, ctx))?;
        return Ok((prepared, None));
    }
    Ok((prepared, Some((law, decision))))
}

fn law_stage(
    prepared: &PreparedStudy,
    request: &RecalcRequest,
    recorder: &mut ReceiptRecorder,
    ctx: &ExecutionContext,
) -> Result<LawValue, RecalcRunError> {
    let (weights, depends_on) = if let Some(target) = &request.target {
        (target.weights.clone(), target.depends_on.clone())
    } else {
        let rows = prepared
            .score_table()
            .map(|t| t.n_rows)
            .ok_or(RecalcRunError::Request("recalc.score_table_unavailable"))?;
        (vec![1.0; rows], Vec::new())
    };
    let result = prepared.retarget(&weights, &depends_on, ctx)?;
    recorder.record(Counter::Reweight, 1);
    Ok(LawValue { ate: result.estimate.ate, std_error: result.estimate.se_analytic })
}

fn decide(law: LawValue, utility: UtilitySpec, recorder: &mut ReceiptRecorder) -> DecisionValue {
    recorder.record(Counter::Decision, 1);
    let net_benefit = utility.benefit_per_unit.mul_add(law.ate, -utility.cost);
    DecisionValue { net_benefit, treat: net_benefit > 0.0 }
}

fn run_stages(
    live: Option<Live>,
    plan: &RecalcPlan,
    request: &RecalcRequest,
    recorder: &mut ReceiptRecorder,
    ctx: &ExecutionContext,
) -> Result<Live, RecalcRunError> {
    let (prepared, held) = score_stages(live, plan, request, recorder, ctx)?;
    let law = if is_recomputed(plan, Stage::Law) {
        law_stage(&prepared, request, recorder, ctx)?
    } else {
        held.map(|(law, _)| law).ok_or(RecalcRunError::NoLiveState(Stage::Law))?
    };
    let decision = if is_recomputed(plan, Stage::Decision) {
        decide(law, request.utility, recorder)
    } else {
        held.map(|(_, decision)| decision).ok_or(RecalcRunError::NoLiveState(Stage::Decision))?
    };
    Ok(Live { prepared, law, decision })
}

/// Plan `request` against the session, run only what the plan says must run, count the work,
/// and return the verified receipt with the law and decision.
///
/// A refused plan (off-grid or unsupported request, incompatible target, an unavailable
/// dependency in a fresh process) returns [`RecalcRunError::Refused`] before any work, and
/// leaves the session unchanged. A failure while running clears the session's identities, so
/// the next call recomputes every stage.
///
/// # Errors
///
/// [`RecalcRunError`] for a refused plan, a malformed request, a failed study operation, or a
/// receipt that contradicts its plan.
pub fn execute_with_receipt(
    session: &mut RecalcSession,
    request: &RecalcRequest,
    ctx: &ExecutionContext,
) -> Result<RecalcOutcome, RecalcRunError> {
    let plan = session.plan(request, ctx);
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    // A derived stage can only be reused from a live artifact. This wrapper has no loader for
    // portable scores or fits, so a plan that reuses a derived stage without one cannot be
    // honored; refuse before any work rather than recompute a stage the plan calls reused.
    if session.live.is_none() {
        let unbacked = plan
            .entries()
            .iter()
            .find(|e| !e.stage.is_input() && matches!(e.status, StageStatus::Reused { .. }));
        if let Some(entry) = unbacked {
            return Err(RecalcRunError::NoLiveState(entry.stage));
        }
    }
    let mut recorder = ReceiptRecorder::new();
    let live = session.live.take();
    let result = run_stages(live, &plan, request, &mut recorder, ctx);
    let live = match result {
        Ok(live) => live,
        Err(error) => {
            session.previous = StageIdentities::new();
            return Err(error);
        }
    };
    let receipt = match recorder.finish(&plan) {
        Ok(receipt) => receipt,
        Err(error) => {
            session.previous = StageIdentities::new();
            return Err(error.into());
        }
    };
    let (law, decision) = (live.law, live.decision);
    session.previous = request.identities(ctx);
    session.live = Some(live);
    Ok(RecalcOutcome { plan, receipt, law, decision })
}

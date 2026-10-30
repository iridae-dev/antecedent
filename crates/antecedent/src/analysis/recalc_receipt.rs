//! Selective recalculation with a visible receipt.
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
//!   Its completed-fold counter observes each fitted outer-fold propensity model and outcome
//!   model pair, including unshared penalized fits and completed folds before a successful
//!   fallback rerun. Cache hits and failed model-fit attempts do not count; inner penalty
//!   selection fits are not this counter's unit. A first run records `5 propensity folds + 5
//!   outcome folds = 10` fold fits. The scope lives only for one call; across calls, reuse is
//!   the retained [`PreparedStudy`] and frozen score table.
//! - **Identifications, score computations, reweights, decisions.** Counted at the call sites
//!   of `Study::prepare` (identification), the score-table build (`prepare` or `refresh`),
//!   `PreparedStudy::retarget`, and the decision rule. These are call-level counts: they say
//!   the wrapper invoked the operation, while the fold-fit counter records completed nuisance
//!   model fits observed inside the estimator's scope.
//!
//! [`RecalcReceipt`] refuses to build when a count contradicts a status: a reused stage that
//! did work, or a recomputed computation that did none.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
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
use super::recalc_common::{float_bytes, identity, literal};

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
    /// One successful retained or final-stage model fit.
    ModelFit,
    /// A frozen score table was built.
    ScoreComputation,
    /// Frozen scores were reweighted into a law.
    Reweight,
    /// The decision rule was evaluated.
    Decision,
    /// Successful empirical factor table constructions.
    FactorBuild,
    /// Successful expression program compilations.
    ProgramCompilation,
    /// Successful provider construction or rebinding.
    ProviderBinding,
    /// Uncached numerical factor evaluations.
    FactorEvaluation,
    /// Successful full functional integrations.
    Integration,
    /// Factor provider invocations, including cache hits.
    ProviderCall,
    /// Checked fit or retained law was reduced to a scalar summary.
    LawSummary,
    /// Actual aligned posterior rows emitted by a numerical inference engine.
    PosteriorDraw,
    /// Actual external callback invocations on one declared branch.
    ExternalInvocation(antecedent_core::recalc::Branch),
}

impl Counter {
    const ALL: [Self; 5] = [
        Self::Identification,
        Self::FoldFit,
        Self::ScoreComputation,
        Self::Reweight,
        Self::Decision,
    ];

    const EXTENDED: [Self; 9] = [
        Self::FactorBuild,
        Self::ProgramCompilation,
        Self::ProviderBinding,
        Self::FactorEvaluation,
        Self::Integration,
        Self::ProviderCall,
        Self::LawSummary,
        Self::PosteriorDraw,
        Self::ExternalInvocation(antecedent_core::recalc::Branch::ZERO),
    ];

    /// The one stage this work belongs to.
    #[must_use]
    pub const fn stage(self) -> Stage {
        match self {
            Self::ExternalInvocation(branch) => Stage::ProviderRequest(branch),
            Self::Identification => Stage::Identification,
            Self::FoldFit
            | Self::ModelFit
            | Self::PosteriorDraw
            | Self::ScoreComputation
            | Self::FactorBuild
            | Self::ProgramCompilation
            | Self::ProviderBinding => Stage::ScoreArtifact,
            Self::Reweight
            | Self::FactorEvaluation
            | Self::Integration
            | Self::ProviderCall
            | Self::LawSummary => Stage::Law,
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
    /// Successful retained or final-stage model fits.
    pub model_fits: u64,
    /// Score tables built.
    pub score_computations: u64,
    /// Reweights of frozen scores performed.
    pub reweights: u64,
    /// Decision evaluations performed.
    pub decisions: u64,
    /// Successful empirical factor table constructions.
    pub factor_builds: u64,
    /// Successful expression program compilations.
    pub program_compilations: u64,
    /// Successful provider construction or rebinding.
    pub provider_bindings: u64,
    /// Uncached numerical factor evaluations.
    pub factor_evaluations: u64,
    /// Successful full functional integrations.
    pub integrations: u64,
    /// Factor provider invocations, including cache hits.
    pub provider_calls: u64,
    /// Checked scalar summary reductions.
    pub law_summaries: u64,
    /// Actual emitted aligned posterior rows.
    pub posterior_draws: u64,
    /// Actual external callback invocations, summed across branches.
    pub external_invocations: u64,
}

impl StageCounts {
    /// The count of one kind of work.
    #[must_use]
    pub const fn get(&self, counter: Counter) -> u64 {
        match counter {
            Counter::Identification => self.identifications,
            Counter::FoldFit => self.fold_fits,
            Counter::ModelFit => self.model_fits,
            Counter::ScoreComputation => self.score_computations,
            Counter::Reweight => self.reweights,
            Counter::Decision => self.decisions,
            Counter::FactorBuild => self.factor_builds,
            Counter::ProgramCompilation => self.program_compilations,
            Counter::ProviderBinding => self.provider_bindings,
            Counter::FactorEvaluation => self.factor_evaluations,
            Counter::Integration => self.integrations,
            Counter::ProviderCall => self.provider_calls,
            Counter::LawSummary => self.law_summaries,
            Counter::PosteriorDraw => self.posterior_draws,
            Counter::ExternalInvocation(_) => self.external_invocations,
        }
    }

    /// Sum over every kind of work.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.identifications
            .saturating_add(self.fold_fits)
            .saturating_add(self.model_fits)
            .saturating_add(self.score_computations)
            .saturating_add(self.reweights)
            .saturating_add(self.decisions)
            .saturating_add(self.factor_builds)
            .saturating_add(self.program_compilations)
            .saturating_add(self.provider_bindings)
            .saturating_add(self.factor_evaluations)
            .saturating_add(self.integrations)
            .saturating_add(self.provider_calls)
            .saturating_add(self.law_summaries)
            .saturating_add(self.posterior_draws)
            .saturating_add(self.external_invocations)
    }

    fn add(&mut self, counter: Counter, n: u64) {
        let slot = match counter {
            Counter::Identification => &mut self.identifications,
            Counter::FoldFit => &mut self.fold_fits,
            Counter::ModelFit => &mut self.model_fits,
            Counter::ScoreComputation => &mut self.score_computations,
            Counter::Reweight => &mut self.reweights,
            Counter::Decision => &mut self.decisions,
            Counter::FactorBuild => &mut self.factor_builds,
            Counter::ProgramCompilation => &mut self.program_compilations,
            Counter::ProviderBinding => &mut self.provider_bindings,
            Counter::FactorEvaluation => &mut self.factor_evaluations,
            Counter::Integration => &mut self.integrations,
            Counter::ProviderCall => &mut self.provider_calls,
            Counter::LawSummary => &mut self.law_summaries,
            Counter::PosteriorDraw => &mut self.posterior_draws,
            Counter::ExternalInvocation(_) => &mut self.external_invocations,
        };
        *slot = slot.saturating_add(n);
    }

    fn merged(mut self, other: &Self) -> Self {
        for counter in Counter::ALL {
            self.add(counter, other.get(counter));
        }
        self.add(Counter::ModelFit, other.model_fits);
        for counter in Counter::EXTENDED {
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

    /// A recorder whose nuisance cache is charged to the admission pool of `ctx`.
    #[must_use]
    pub fn charged_to(ctx: &antecedent_core::ExecutionContext) -> Self {
        Self { counts: BTreeMap::new(), nuisances: CrossfitNuisanceCache::charged_to(ctx) }
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

    /// Run `work` with this recorder's nuisance cache active and record completed outer-fold
    /// propensity and outcome model fits observed by the AIPW score builder.
    pub fn measured<R>(&mut self, work: impl FnOnce() -> R) -> R {
        let before = self.nuisances.completed_fold_fits();
        let (out, _used) = self.nuisances.scope(work);
        let after = self.nuisances.completed_fold_fits();
        self.record(Counter::FoldFit, to_u64(after.saturating_sub(before)));
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
                status: e.status.clone(),
                identity: e.identity,
                counts: self.counts(e.stage),
            })
            .collect();
        RecalcReceipt::new(entries)
    }
}

/// One stage of a receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
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
        self.entries.iter().map(|e| (e.stage, e.status.clone())).collect()
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
                if e.counts.model_fits > 0 {
                    counts.extend_from_slice(&e.counts.model_fits.to_le_bytes());
                }
                for (index, counter) in Counter::EXTENDED.iter().enumerate() {
                    let n = e.counts.get(*counter);
                    if n > 0 {
                        counts.push(u8::try_from(index + 1).unwrap_or(u8::MAX));
                        counts.extend_from_slice(&n.to_le_bytes());
                    }
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

fn entry_recomputed(entry: &ReceiptEntry) -> bool {
    matches!(entry.status, StageStatus::Recomputed { .. })
}

/// Work counted at a stage that does not own it: only a recomputed stage may carry work, and
/// only of its own counters (a provider request also carries its branch's invocations).
fn reject_foreign_work(entry: &ReceiptEntry) -> Result<(), ReceiptError> {
    let stage = entry.stage;
    for counter in Counter::ALL.into_iter().chain([Counter::ModelFit]).chain(Counter::EXTENDED) {
        let allowed = entry_recomputed(entry)
            && (counter.stage() == stage
                || matches!(
                    (counter, stage),
                    (Counter::ExternalInvocation(_), Stage::ProviderRequest(_))
                ));
        if entry.counts.get(counter) > 0 && !allowed {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
    }
    Ok(())
}

/// The work a recomputed score-artifact stage must show, which depends on which route ran.
fn score_artifact_required(entry: &ReceiptEntry) -> Result<Vec<Counter>, ReceiptError> {
    let stage = entry.stage;
    let c = &entry.counts;
    let unexpected = Err(ReceiptError::UnexpectedWork { stage });
    if c.posterior_draws > 0 {
        if c.fold_fits > 0
            || c.score_computations > 0
            || c.provider_bindings > 0
            || c.factor_builds > 0
            || c.program_compilations > 0
        {
            return unexpected;
        }
        Ok(vec![Counter::ModelFit, Counter::PosteriorDraw])
    } else if c.provider_bindings > 0 {
        if c.fold_fits > 0 || c.model_fits > 0 || c.score_computations > 0 {
            return unexpected;
        }
        Ok(vec![Counter::ProviderBinding])
    } else if c.factor_builds > 0 || c.program_compilations > 0 {
        Err(ReceiptError::MissingWork { stage, counter: Counter::ProviderBinding })
    } else if c.score_computations == 0 && c.model_fits > 0 {
        if c.fold_fits > 0 {
            return unexpected;
        }
        Ok(vec![Counter::ModelFit])
    } else {
        Ok(vec![Counter::FoldFit, Counter::ScoreComputation])
    }
}

/// The work a recomputed law stage must show: a summary, an integration, or a reweight.
fn law_required(entry: &ReceiptEntry) -> Result<Vec<Counter>, ReceiptError> {
    let stage = entry.stage;
    let c = &entry.counts;
    if c.law_summaries > 0 {
        if c.reweights > 0 || c.integrations > 0 || c.factor_evaluations > 0 || c.provider_calls > 0
        {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
        Ok(vec![Counter::LawSummary])
    } else if c.integrations > 0 || c.factor_evaluations > 0 || c.provider_calls > 0 {
        if c.reweights > 0 {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
        Ok(vec![Counter::Integration])
    } else {
        Ok(required_work(stage).to_vec())
    }
}

/// The counters a stage's entry must show, given its status and the route its counts reveal.
fn required_counters(entry: &ReceiptEntry) -> Result<Vec<Counter>, ReceiptError> {
    if !entry_recomputed(entry) {
        return Ok(Vec::new());
    }
    match entry.stage {
        Stage::ProviderRequest(branch) => Ok(vec![Counter::ExternalInvocation(branch)]),
        Stage::ScoreArtifact => score_artifact_required(entry),
        Stage::Law => law_required(entry),
        stage => Ok(required_work(stage).to_vec()),
    }
}

fn verify_entry(entry: &ReceiptEntry) -> Result<(), ReceiptError> {
    reject_foreign_work(entry)?;
    for counter in required_counters(entry)? {
        if entry.counts.get(counter) == 0 {
            return Err(ReceiptError::MissingWork { stage: entry.stage, counter });
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

impl TargetWeights {
    /// Whether the weights can declare a population: every weight finite and non-negative,
    /// positive finite total mass, and one weight per row when `rows` is given.
    #[must_use]
    pub fn are_valid(&self, rows: Option<usize>) -> bool {
        let mass = self.weights.iter().sum::<f64>();
        rows.is_none_or(|n| self.weights.len() == n)
            && self.weights.iter().all(|w| w.is_finite() && *w >= 0.)
            && mass.is_finite()
            && mass > 0.
    }
}

/// How a [`Columns`] table treats non-finite entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NonFinite {
    /// Every entry must be finite.
    Reject,
    /// `NaN` marks a missing entry (complete-case routes drop such rows); infinities are
    /// still refused.
    MissingAsNan,
}

/// Named numeric columns of a request; the variable id of a column is its position.
///
/// The table is not validated on construction: [`Columns::is_well_formed`] and
/// [`Columns::are_distinct_roles`] are the shared checks every route's validator applies, and
/// [`Columns::column`] is the checked access by role id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Columns(Vec<(String, Vec<f64>)>);

impl Columns {
    /// Wrap named columns in stable id order.
    #[must_use]
    pub const fn new(columns: Vec<(String, Vec<f64>)>) -> Self {
        Self(columns)
    }

    /// Number of rows of the first column; zero for an empty table.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.0.first().map_or(0, |(_, values)| values.len())
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.0.len()
    }

    /// The `(name, values)` pair of role `id`, or `None` when no such column exists.
    #[must_use]
    pub fn column(&self, id: u32) -> Option<&(String, Vec<f64>)> {
        self.0.get(usize::try_from(id).ok()?)
    }

    /// The values of role `id`.
    #[must_use]
    pub fn values(&self, id: u32) -> Option<&[f64]> {
        self.column(id).map(|(_, values)| values.as_slice())
    }

    /// The name of role `id`.
    #[must_use]
    pub fn name(&self, id: u32) -> Option<&str> {
        self.column(id).map(|(name, _)| name.as_str())
    }

    /// The columns as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[(String, Vec<f64>)] {
        &self.0
    }

    /// Append a column; its id is the previous column count.
    pub fn push(&mut self, column: (String, Vec<f64>)) {
        self.0.push(column);
    }

    /// The underlying columns.
    #[must_use]
    pub fn into_inner(self) -> Vec<(String, Vec<f64>)> {
        self.0
    }

    /// At least one row, unique non-empty names, equal column lengths, and entries finite
    /// according to `policy`.
    #[must_use]
    pub fn is_well_formed(&self, policy: NonFinite) -> bool {
        let rows = self.n_rows();
        let names: BTreeSet<&str> = self.0.iter().map(|(name, _)| name.as_str()).collect();
        rows > 0
            && names.len() == self.0.len()
            && self.0.iter().all(|(name, values)| {
                !name.is_empty()
                    && values.len() == rows
                    && values.iter().all(|x| match policy {
                        NonFinite::Reject => x.is_finite(),
                        NonFinite::MissingAsNan => !x.is_infinite(),
                    })
            })
    }

    /// Whether every role names a column and no two roles coincide.
    #[must_use]
    pub fn are_distinct_roles(&self, roles: &[u32]) -> bool {
        let distinct: BTreeSet<u32> = roles.iter().copied().collect();
        distinct.len() == roles.len() && roles.iter().all(|&id| self.column(id).is_some())
    }
}

impl std::ops::Deref for Columns {
    type Target = [(String, Vec<f64>)];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Columns {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for &'a Columns {
    type Item = &'a (String, Vec<f64>);
    type IntoIter = std::slice::Iter<'a, (String, Vec<f64>)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl From<Vec<(String, Vec<f64>)>> for Columns {
    fn from(columns: Vec<(String, Vec<f64>)>) -> Self {
        Self(columns)
    }
}

impl FromIterator<(String, Vec<f64>)> for Columns {
    fn from_iter<I: IntoIterator<Item = (String, Vec<f64>)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A net-benefit rule: `benefit_per_unit * ate - cost`, treat when positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UtilitySpec {
    /// Benefit per unit of average effect.
    pub benefit_per_unit: f64,
    /// Fixed cost of acting.
    pub cost: f64,
}

impl UtilitySpec {
    pub(crate) fn validate(self) -> Result<(), RecalcRunError> {
        if !self.benefit_per_unit.is_finite() || !self.cost.is_finite() {
            return Err(RecalcRunError::Request(RecalcRequestCode::InvalidRequest));
        }
        Ok(())
    }
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
    pub columns: Columns,
    /// Directed edges `(from, to)` over column positions.
    pub edges: Vec<(u32, u32)>,
    /// Treatment column position.
    pub treatment: u32,
    /// Outcome column position.
    pub outcome: u32,
    /// Estimator configuration. This score-retarget route requires zero bootstrap replicates,
    /// homoskedastic analytic SEs, and
    /// [`OverlapPolicy::RequireDiagnostics`](antecedent_estimate::OverlapPolicy::RequireDiagnostics)
    /// with `trim: None`; the optional clip and other AIPW nuisance and GLM options are fit
    /// and keyed normally.
    pub estimator: AipwAte,
    /// Target population weights; `None` is the observed population.
    pub target: Option<TargetWeights>,
    /// Net-benefit rule.
    pub utility: UtilitySpec,
}

/// Append-only builder of an estimator's identity bytes.
///
/// Integers are little endian at their declared width, floats are bit patterns, every sum
/// type starts with a one-byte tag and every sequence with its length, so unequal
/// configurations cannot share bytes. The layout is part of the stage identities that resume
/// and reuse key on: changing it changes every retained identity.
struct IdentityBytes(Vec<u8>);

impl IdentityBytes {
    fn new(domain: &[u8]) -> Self {
        Self(domain.to_vec())
    }

    fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.extend_from_slice(bytes);
        self
    }

    fn tag(&mut self, tag: u8) -> &mut Self {
        self.0.push(tag);
        self
    }

    fn u32(&mut self, value: u32) -> &mut Self {
        self.raw(&value.to_le_bytes())
    }

    fn usize(&mut self, value: usize) -> &mut Self {
        self.raw(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes())
    }

    fn f64(&mut self, value: f64) -> &mut Self {
        self.raw(&value.to_bits().to_le_bytes())
    }

    fn opt_f64(&mut self, value: Option<f64>) -> &mut Self {
        match value {
            None => self.tag(0),
            Some(value) => self.tag(1).f64(value),
        }
    }

    fn f64s(&mut self, values: &[f64]) -> &mut Self {
        self.usize(values.len());
        for value in values {
            self.f64(*value);
        }
        self
    }

    fn u32s(&mut self, values: &[u32]) -> &mut Self {
        self.usize(values.len());
        for value in values {
            self.u32(*value);
        }
        self
    }

    fn opt_u32s(&mut self, values: Option<&[u32]>) -> &mut Self {
        match values {
            None => self.tag(0),
            Some(values) => self.tag(1).u32s(values),
        }
    }

    fn opt_panel(&mut self, values: Option<&[i64]>) -> &mut Self {
        match values {
            None => self.tag(0),
            Some(values) => {
                self.tag(1).usize(values.len());
                for value in values {
                    self.raw(&value.to_le_bytes());
                }
                self
            }
        }
    }

    fn write(&mut self, part: &impl CanonicalIdentity) -> &mut Self {
        part.write_identity(self);
        self
    }
}

/// A part of an estimator specification with a canonical identity encoding.
trait CanonicalIdentity {
    fn write_identity(&self, out: &mut IdentityBytes);
}

impl CanonicalIdentity for antecedent_estimate::OverlapPolicy {
    fn write_identity(&self, out: &mut IdentityBytes) {
        match self.bounds() {
            None => {
                out.tag(0);
            }
            Some((clip, trim)) => {
                out.tag(1).opt_f64(clip).opt_f64(trim);
            }
        }
    }
}

impl CanonicalIdentity for antecedent_stats::NbAlphaPolicy {
    fn write_identity(&self, out: &mut IdentityBytes) {
        match *self {
            Self::Fixed(alpha) => {
                out.tag(0).f64(alpha);
            }
            Self::MethodOfMoments => {
                out.tag(1);
            }
            Self::NestedMle { max_outer, tol_alpha } => {
                out.tag(2).u32(max_outer).f64(tol_alpha);
            }
        }
    }
}

impl CanonicalIdentity for antecedent_stats::GlmOptions {
    fn write_identity(&self, out: &mut IdentityBytes) {
        out.u32(self.max_iter)
            .f64(self.tol)
            .write(&self.nb_alpha)
            .opt_f64(self.ridge_on_separation);
    }
}

impl CanonicalIdentity for antecedent_estimate::AnalyticSeKind {
    fn write_identity(&self, out: &mut IdentityBytes) {
        use antecedent_estimate::AnalyticSeKind as Kind;
        match *self {
            Kind::Homoskedastic => {
                out.tag(0);
            }
            Kind::Hc0 => {
                out.tag(1);
            }
            Kind::Hc1 => {
                out.tag(2);
            }
            Kind::Hc2 => {
                out.tag(3);
            }
            Kind::Hc3 => {
                out.tag(4);
            }
            Kind::Cluster => {
                out.tag(5);
            }
            Kind::Multiway => {
                out.tag(6);
            }
            Kind::NeweyWest { lag } => {
                out.tag(7).usize(lag);
            }
            Kind::PanelClusterHac { lag } => {
                out.tag(8).usize(lag);
            }
        }
    }
}

impl CanonicalIdentity for antecedent_core::PopulationRegistry {
    fn write_identity(&self, out: &mut IdentityBytes) {
        for (name, rows) in self.predicates() {
            out.raw(name.as_bytes()).tag(0).usize(rows.len());
            for row in rows {
                out.usize(*row);
            }
        }
        out.tag(0xff);
        for (id, weights, parents) in self.distributions() {
            out.u32(id.raw()).f64s(weights).usize(parents.len());
            for parent in parents {
                out.u32(parent.raw());
            }
        }
    }
}

fn estimator_identity_bytes(estimator: &AipwAte) -> Vec<u8> {
    let mut out = IdentityBytes::new(b"aipw.v1");
    out.raw(&estimator.inference.bootstrap_replicates.to_le_bytes());
    out.write(&estimator.overlap).write(&estimator.glm_options).write(&estimator.inference.se_kind);
    out.opt_u32s(estimator.inference.cluster_ids.as_deref())
        .opt_u32s(estimator.cluster_ids_second.as_deref())
        .opt_panel(estimator.inference.panel_times.as_deref());
    match estimator.inference.multiway_ids.as_deref() {
        None => {
            out.tag(0);
        }
        Some(groups) => {
            out.tag(1).usize(groups.len());
            for group in groups {
                out.u32s(group);
            }
        }
    }
    out.raw(&estimator.propensity.identity_digest());
    match estimator.cluster_dml.as_ref() {
        None => {
            out.tag(0);
        }
        Some(cluster) => {
            out.tag(1).raw(cluster.canonical_key().as_bytes());
        }
    }
    match estimator.population_registry.as_ref() {
        None => {
            out.tag(0);
        }
        Some(registry) => {
            out.tag(1).write(registry);
        }
    }
    // Labels per source row are appended only when declared, so a configuration without
    // them keeps its earlier encoding.
    if let Some(labels) = estimator.inference.source_cluster_ids.as_deref() {
        out.tag(2).u32s(labels);
    }
    if let Some(labels) = estimator.source_cluster_ids_second.as_deref() {
        out.tag(3).u32s(labels);
    }
    if let Some(weights) = estimator.sampling_weights.as_ref() {
        out.tag(4).f64s(weights.source());
    }
    out.0
}

impl RecalcRequest {
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        if self.estimator.inference.bootstrap_replicates != 0 {
            return Err(RecalcRunError::Request(RecalcRequestCode::BootstrapReplicatesUnsupported));
        }
        if self.estimator.overlap == antecedent_estimate::OverlapPolicy::ExplicitOverride {
            return Err(RecalcRunError::Request(RecalcRequestCode::AipwRequiresOverlapDiagnostics));
        }
        if self.estimator.inference.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
            || matches!(
                self.estimator.overlap,
                antecedent_estimate::OverlapPolicy::RequireDiagnostics { trim: Some(_), .. }
            )
        {
            return Err(RecalcRunError::Request(
                RecalcRequestCode::UnsupportedAipwScoreConfiguration,
            ));
        }
        Ok(())
    }

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
            data_parts.push(float_bytes(values));
        }
        let rows = self.columns.n_rows();
        let folds = to_u64(DEFAULT_AIPW_FOLDS);
        let target = self.target.as_ref().map_or_else(
            || literal("target_population", "all_observed"),
            |t| {
                let parents: Vec<u8> =
                    t.depends_on.iter().flat_map(|v| v.raw().to_le_bytes()).collect();
                identity("target_population", &[float_bytes(&t.weights), parents])
            },
        );
        let mut ids = StageIdentities::new();
        ids.set(
            Stage::Graph,
            identity("graph", &[to_u64(self.columns.len()).to_le_bytes().to_vec(), edge_bytes]),
        );
        ids.set(
            Stage::Query,
            literal("query", &format!("binary_ate:{}:{}", self.treatment, self.outcome)),
        );
        ids.set(Stage::Regime, literal("regime", "static.binary"));
        ids.set(Stage::Evidence, literal("evidence", "observational.no_external"));
        ids.set(Stage::SourcePopulation, literal("source_population", "all_observed"));
        ids.set(Stage::TargetPopulation, target);
        ids.set(Stage::DataSnapshot, identity("data_snapshot", &data_parts));
        ids.set(Stage::RowDesign, literal("row_design", &format!("complete_case.rows={rows}")));
        ids.set(Stage::TreatmentGrid, literal("treatment_grid", "binary.0_1"));
        ids.set(
            Stage::LearnerFoldsRng,
            identity(
                "learner_folds_rng",
                &[
                    estimator_identity_bytes(&self.estimator),
                    folds.to_le_bytes().to_vec(),
                    ctx.rng.master_seed().to_le_bytes().to_vec(),
                ],
            ),
        );
        ids.set(
            Stage::Utility,
            identity(
                "utility",
                &[float_bytes(&[self.utility.benefit_per_unit, self.utility.cost])],
            ),
        );
        ids.set(Stage::Identification, literal("identification", "backdoor.prepared"));
        ids.set(Stage::ScoreArtifact, literal("score_artifact", "crossfit_aipw.scores"));
        ids.set(Stage::Law, literal("law", "weighted_mean_of_scores"));
        ids.set(Stage::Decision, literal("decision", "net_benefit"));
        ids
    }

    fn graph(&self) -> Result<Dag, RecalcRunError> {
        let n = u32::try_from(self.columns.len())
            .map_err(|_| RecalcRunError::Request(RecalcRequestCode::InvalidGraph))?;
        let mut graph = Dag::with_variables(n);
        for &(from, to) in &self.edges {
            graph
                .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                .map_err(|_| RecalcRunError::Request(RecalcRequestCode::InvalidGraph))?;
        }
        Ok(graph)
    }

    fn data(&self) -> Result<TabularData, RecalcRunError> {
        let borrowed: Vec<(&str, &[f64])> =
            self.columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        TabularData::from_f64_columns(borrowed)
            .map_err(|_| RecalcRunError::Request(RecalcRequestCode::InvalidData))
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

/// Why a recalculation request was refused, as a closed set.
///
/// The wire spelling (`recalc.<snake_case>`) is [`as_str`](Self::as_str); matching on the
/// variant keeps every producer and consumer of a code checked at compile time.
macro_rules! recalc_request_codes {
    ($($variant:ident => $text:literal,)*) => {
        /// A refused-request code of the recalculation routes.
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        pub enum RecalcRequestCode {
            $(
                #[doc = $text]
                $variant,
            )*
        }

        impl RecalcRequestCode {
            /// Every code, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant,)*];

            /// The `recalc.<snake_case>` spelling of the code.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)*
                }
            }
        }
    };
}

recalc_request_codes! {
    AdjustedPredictionOutOfSupport => "recalc.adjusted_prediction_out_of_support",
    AdjustedPredictionSchemaMismatch => "recalc.adjusted_prediction_schema_mismatch",
    AdjustmentNotIdentified => "recalc.adjustment_not_identified",
    AipwRequiresOverlapDiagnostics => "recalc.aipw_requires_overlap_diagnostics",
    BayesianBasisPriorUnsupported => "recalc.bayesian_basis_prior_unsupported",
    BayesianEffectUnavailable => "recalc.bayesian_effect_unavailable",
    BayesianExternalComposeUnsupported => "recalc.bayesian_external_compose_unsupported",
    BayesianInferenceUnsupported => "recalc.bayesian_inference_unsupported",
    BayesianInvalidData => "recalc.bayesian_invalid_data",
    BayesianInvalidGraph => "recalc.bayesian_invalid_graph",
    BayesianInvalidQuery => "recalc.bayesian_invalid_query",
    BayesianInvalidSummary => "recalc.bayesian_invalid_summary",
    BayesianPriorInvalid => "recalc.bayesian_prior_invalid",
    BayesianPriorLikelihoodDoubleUse => "recalc.bayesian_prior_likelihood_double_use",
    BayesianPriorSourceConflict => "recalc.bayesian_prior_source_conflict",
    BayesianPriorSourceUnsupported => "recalc.bayesian_prior_source_unsupported",
    BayesianPriorSourceUnverified => "recalc.bayesian_prior_source_unverified",
    BayesianWorkspaceLimit => "recalc.bayesian_workspace_limit",
    BootstrapReplicatesUnsupported => "recalc.bootstrap_replicates_unsupported",
    Cancelled => "recalc.cancelled",
    CategoricalRoleMismatch => "recalc.categorical_role_mismatch",
    ContrastOutOfSupport => "recalc.contrast_out_of_support",
    ContrastUnavailable => "recalc.contrast_unavailable",
    DesignArtifactMismatch => "recalc.design_artifact_mismatch",
    DesignArtifactUnverified => "recalc.design_artifact_unverified",
    DesignDataUnavailable => "recalc.design_data_unavailable",
    DesignEffectUnavailable => "recalc.design_effect_unavailable",
    DesignNotChecked => "recalc.design_not_checked",
    DesignPointUnavailable => "recalc.design_point_unavailable",
    DrEffectUnavailable => "recalc.dr_effect_unavailable",
    DrFitCounterMismatch => "recalc.dr_fit_counter_mismatch",
    DrPredictionOutOfSupport => "recalc.dr_prediction_out_of_support",
    DrPredictionSchemaMismatch => "recalc.dr_prediction_schema_mismatch",
    DrPredictorUnavailable => "recalc.dr_predictor_unavailable",
    DrScoresUnavailable => "recalc.dr_scores_unavailable",
    FrozenScoresRejected => "recalc.frozen_scores_rejected",
    IncompatibleContrast => "recalc.incompatible_contrast",
    InvalidAdjustmentSet => "recalc.invalid_adjustment_set",
    InvalidCategoricalContrast => "recalc.invalid_categorical_contrast",
    InvalidChangedInput => "recalc.invalid_changed_input",
    InvalidContrast => "recalc.invalid_contrast",
    InvalidData => "recalc.invalid_data",
    InvalidDesignArtifact => "recalc.invalid_design_artifact",
    InvalidDesignRoles => "recalc.invalid_design_roles",
    InvalidEvidence => "recalc.invalid_evidence",
    InvalidGraph => "recalc.invalid_graph",
    InvalidQuery => "recalc.invalid_query",
    InvalidRdWindow => "recalc.invalid_rd_window",
    InvalidRequest => "recalc.invalid_request",
    InvalidTargetWeights => "recalc.invalid_target_weights",
    InvalidTemporalProvider => "recalc.invalid_temporal_provider",
    InvalidTemporalWindow => "recalc.invalid_temporal_window",
    IvDiagnosticsUnavailable => "recalc.iv_diagnostics_unavailable",
    JointBayesianGraphUnsupported => "recalc.joint_bayesian_graph_unsupported",
    JointBayesianInvalidData => "recalc.joint_bayesian_invalid_data",
    MemoryBudgetExceeded => "recalc.memory_budget_exceeded",
    MzInvalidRequest => "recalc.mz_invalid_request",
    PrimaryResultIsNotAnEffect => "recalc.primary_result_is_not_an_effect",
    RetargetNotLicensed => "recalc.retarget_not_licensed",
    RowCountMismatch => "recalc.row_count_mismatch",
    RowIdsMismatch => "recalc.row_ids_mismatch",
    ScoreStageUndeclared => "recalc.score_stage_undeclared",
    ScoreTableUnavailable => "recalc.score_table_unavailable",
    StaticActionOutOfSupport => "recalc.static_action_out_of_support",
    StaticGraphUnsupported => "recalc.static_graph_unsupported",
    StaticInvalidContrast => "recalc.static_invalid_contrast",
    StaticInvalidData => "recalc.static_invalid_data",
    StaticInvalidQuery => "recalc.static_invalid_query",
    StaticRequiredFactorMissing => "recalc.static_required_factor_missing",
    TemporalArtifactMismatch => "recalc.temporal_artifact_mismatch",
    TemporalArtifactSeedMismatch => "recalc.temporal_artifact_seed_mismatch",
    UnsupportedAipwScoreConfiguration => "recalc.unsupported_aipw_score_configuration",
    WeightedOverlapFailed => "recalc.weighted_overlap_failed",
}

impl fmt::Display for RecalcRequestCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<RecalcRequestCode> for &'static str {
    fn from(code: RecalcRequestCode) -> Self {
        code.as_str()
    }
}

/// Why a recalculation did not run.
#[derive(Debug)]
pub enum RecalcRunError {
    /// The plan contains a refused stage; nothing was executed.
    Refused(Box<RecalcPlan>),
    /// The request itself is malformed.
    Request(RecalcRequestCode),
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
            Self::Request(detail) => f.write_str(detail.as_str()),
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
    prepared: Arc<PreparedStudy>,
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
        let caps =
            RecalcCapabilities { retarget: self.retarget, request: self.request.clone(), boundary };
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
type RetainedScoreStage = (Arc<PreparedStudy>, Option<(LawValue, DecisionValue)>);

fn score_stages(
    live: Option<&Live>,
    plan: &RecalcPlan,
    request: &RecalcRequest,
    recorder: &mut ReceiptRecorder,
    ctx: &ExecutionContext,
) -> Result<RetainedScoreStage, RecalcRunError> {
    if is_recomputed(plan, Stage::Identification) {
        let study = request.study(request.data()?)?;
        recorder.record(Counter::Identification, 1);
        recorder.record(Counter::ScoreComputation, 1);
        let prepared = recorder.measured(|| study.prepare(ctx))?;
        return Ok((Arc::new(prepared), None));
    }
    let Some(live) = live else {
        return Err(RecalcRunError::NoLiveState(Stage::Identification));
    };
    if is_recomputed(plan, Stage::ScoreArtifact) {
        let mut prepared = live.prepared.as_ref().clone();
        let data = request.data()?;
        recorder.record(Counter::ScoreComputation, 1);
        recorder.measured(|| {
            prepared.refresh_aipw_configuration(data, request.estimator.clone(), ctx)
        })?;
        return Ok((Arc::new(prepared), None));
    }
    Ok((Arc::clone(&live.prepared), Some((live.law, live.decision))))
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
            .ok_or(RecalcRunError::Request(RecalcRequestCode::ScoreTableUnavailable))?;
        (vec![1.0; rows], Vec::new())
    };
    let result = prepared.retarget(&weights, &depends_on, ctx)?;
    recorder.record(Counter::Reweight, 1);
    let effect = result
        .estimate
        .as_effect()
        .ok_or(RecalcRunError::Request(RecalcRequestCode::PrimaryResultIsNotAnEffect))?;
    Ok(LawValue { ate: effect.ate, std_error: effect.se_analytic() })
}

pub(crate) fn decide(
    law: LawValue,
    utility: UtilitySpec,
    recorder: &mut ReceiptRecorder,
) -> Result<DecisionValue, RecalcRunError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::Decision,
        || {
            utility.validate()?;
            let net_benefit = utility.benefit_per_unit.mul_add(law.ate, -utility.cost);
            if !net_benefit.is_finite() {
                return Err(RecalcRunError::Request(RecalcRequestCode::InvalidRequest));
            }
            recorder.record(Counter::Decision, 1);
            Ok(DecisionValue { net_benefit, treat: net_benefit > 0.0 })
        },
    )
}

fn run_stages(
    live: Option<&Live>,
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
        decide(law, request.utility, recorder)?
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
/// leaves the session unchanged. Running and receipt validation are transactional: a failed
/// refresh, law or decision retains the last successful prepared study and identities.
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
    request.validate()?;
    // A cancelled run stops before any stage runs and leaves the session as it was.
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request(RecalcRequestCode::Cancelled));
    }
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
    let live = run_stages(session.live.as_ref(), &plan, request, &mut recorder, ctx)?;
    let receipt = recorder.finish(&plan)?;
    let (law, decision) = (live.law, live.decision);
    session.previous = request.identities(ctx);
    session.live = Some(live);
    Ok(RecalcOutcome { plan, receipt, law, decision })
}

// -- standing inputs ------------------------------------------------------------------------

/// A recalculation session family, for the standing inputs its stage identities declare.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum RecalcSessionKind {
    /// Cross-fitted AIPW average effect with retargeting ([`RecalcRequest`]).
    Receipt,
    /// Adjusted mean contrast.
    Adjusted,
    /// Binary joint-cell AIPW.
    Cell,
    /// Design-based (instrumental-variable and front-door) estimators.
    Design,
    /// Doubly robust CATE.
    DoublyRobust,
    /// Finite static ADMG responses and multi-source transport.
    Static,
    /// Checked temporal response.
    Temporal,
    /// Conjugate Bayesian posterior.
    Bayesian,
    /// Joint multi-source Bayesian posterior.
    JointBayesian,
    /// Attested external mean callbacks.
    External,
    /// Native response with attested callbacks under one plan.
    Composite,
}

/// How a session states one standing input in its stage identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StandingInputDeclaration {
    /// The input is declared; a change in it moves the law and the decision, and nothing
    /// that reads only data.
    Declared,
    /// The session's claim does not rest on the input, so none is declared and the
    /// identities are exactly those of a session that predates standing inputs.
    NotApplicable {
        /// Why the claim does not rest on it.
        reason: &'static str,
    },
    /// The session does not hold the information the input names, or declaring it would
    /// reseal the identity of a retained artifact, so none is declared. A claim that needs
    /// it is not licensed by this session's identities alone.
    Undeclared {
        /// What is missing, as a namespaced finding.
        finding: &'static str,
        /// What would make the input declarable.
        remedy: &'static str,
    },
}

/// The three standing inputs of a session's stage identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionStanding {
    /// The calibration record behind an interval or posterior statement.
    pub calibration: StandingInputDeclaration,
    /// The representation (model family and version) the score artifact was built with.
    pub representation: StandingInputDeclaration,
    /// The coupling between streams a joint statement relies on.
    pub coupling: StandingInputDeclaration,
}

const POINT_ONLY: StandingInputDeclaration = StandingInputDeclaration::NotApplicable {
    reason: "the law is a point value; its standard error is an uncalibrated diagnostic",
};
const NO_JOINT: StandingInputDeclaration = StandingInputDeclaration::NotApplicable {
    reason: "the session forms no statement across more than one sampled stream",
};
const REPRESENTATION_UNAVAILABLE: StandingInputDeclaration = StandingInputDeclaration::Undeclared {
    finding: "recalc.standing_undeclared.representation_version_not_held",
    remedy: "declare the representation when the session consumes a measurement model; the \
             score artifact identity then moves, so each retained score artifact of the earlier \
             identity is recomputed once, visibly, and none is resealed in place",
};

impl RecalcSessionKind {
    /// The standing inputs this session family declares, and why each other is not.
    ///
    /// Declared inputs enter only the effective identities of the law and the decision.
    /// The representation enters the score artifact: it is declared by no session, because
    /// no session consumes a measurement model and declaring a label with no model behind
    /// it would reseal the identity of every retained score artifact for no information.
    #[must_use]
    pub const fn standing(self) -> SessionStanding {
        match self {
            Self::Bayesian | Self::JointBayesian => SessionStanding {
                calibration: StandingInputDeclaration::Declared,
                representation: REPRESENTATION_UNAVAILABLE,
                coupling: StandingInputDeclaration::Declared,
            },
            Self::External | Self::Composite => SessionStanding {
                calibration: StandingInputDeclaration::Undeclared {
                    finding: "recalc.standing_undeclared.calibration_folded_into_decision",
                    remedy: "declare the calibration of the provider's output where the \
                             provider standing is declared; an existing previous state would \
                             recompute its law once, visibly",
                },
                representation: REPRESENTATION_UNAVAILABLE,
                coupling: StandingInputDeclaration::Undeclared {
                    finding: "recalc.standing_undeclared.callback_metadata_establishes_no_coupling",
                    remedy: "declare the coupling of the callback stream to the native stream \
                             when a joint statement over both is requested",
                },
            },
            Self::Receipt
            | Self::Adjusted
            | Self::Cell
            | Self::Design
            | Self::DoublyRobust
            | Self::Static
            | Self::Temporal => SessionStanding {
                calibration: POINT_ONLY,
                representation: REPRESENTATION_UNAVAILABLE,
                coupling: NO_JOINT,
            },
        }
    }
}

#[cfg(test)]
mod estimator_identity_tests {
    use super::{RecalcRequest, UtilitySpec};
    use antecedent_core::ExecutionContext;
    use antecedent_core::recalc::Stage;
    use antecedent_estimate::{AipwAte, OverlapPolicy};

    fn request(estimator: AipwAte) -> RecalcRequest {
        RecalcRequest {
            columns: vec![("t".into(), vec![0.0, 1.0]), ("y".into(), vec![0.0, 2.0])].into(),
            edges: vec![(0, 1)],
            treatment: 0,
            outcome: 1,
            estimator,
            target: None,
            utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
        }
    }

    #[test]
    fn aipw_fold_identity_follows_estimator_structure() {
        let ctx = ExecutionContext::for_tests(1);
        let stage = Stage::LearnerFoldsRng;
        let base = request(AipwAte::new()).identities(&ctx).own(stage);
        let again = request(AipwAte::new()).identities(&ctx).own(stage);
        assert_eq!(base, again);
        let mut clipped = AipwAte::new();
        clipped.overlap = OverlapPolicy::RequireDiagnostics { clip: Some(1e-13), trim: None };
        let changed = request(clipped).identities(&ctx).own(stage);
        assert_ne!(base, changed);
    }
}

#[cfg(test)]
mod request_code_tests {
    use super::{Columns, NonFinite, RecalcRequestCode, TargetWeights};
    use std::collections::BTreeSet;

    #[test]
    fn request_codes_are_unique_recalc_spellings() {
        let spellings: BTreeSet<_> = RecalcRequestCode::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(spellings.len(), RecalcRequestCode::ALL.len());
        assert!(spellings.iter().all(|s| s.starts_with("recalc.")));
        assert_eq!(RecalcRequestCode::Cancelled.to_string(), "recalc.cancelled");
    }

    fn table(columns: &[(&str, &[f64])]) -> Columns {
        columns.iter().map(|(name, values)| ((*name).to_owned(), values.to_vec())).collect()
    }

    #[test]
    fn columns_validation_is_shared_and_policy_aware() {
        let ok = table(&[("a", &[1., 2.]), ("b", &[3., 4.])]);
        assert!(ok.is_well_formed(NonFinite::Reject));
        assert_eq!((ok.n_rows(), ok.n_cols()), (2, 2));
        assert_eq!(ok.name(1), Some("b"));
        assert!(ok.column(2).is_none());
        let missing = table(&[("a", &[1., f64::NAN])]);
        assert!(!missing.is_well_formed(NonFinite::Reject));
        assert!(missing.is_well_formed(NonFinite::MissingAsNan));
        let infinite = table(&[("a", &[1., f64::INFINITY])]);
        assert!(!infinite.is_well_formed(NonFinite::MissingAsNan));
        assert!(!table(&[("a", &[1.]), ("a", &[2.])]).is_well_formed(NonFinite::Reject));
        assert!(!table(&[("a", &[1.]), ("b", &[2., 3.])]).is_well_formed(NonFinite::Reject));
        assert!(!table(&[("", &[1.])]).is_well_formed(NonFinite::Reject));
        assert!(!Columns::default().is_well_formed(NonFinite::Reject));
        assert!(ok.are_distinct_roles(&[0, 1]));
        assert!(!ok.are_distinct_roles(&[0, 0]));
        assert!(!ok.are_distinct_roles(&[0, 2]));
    }

    #[test]
    fn target_weights_need_positive_finite_mass_and_matching_rows() {
        let weights = |w: &[f64]| TargetWeights { weights: w.to_vec(), depends_on: Vec::new() };
        assert!(weights(&[1., 0.]).are_valid(Some(2)));
        assert!(!weights(&[1., 0.]).are_valid(Some(3)));
        assert!(weights(&[1., 0.]).are_valid(None));
        assert!(!weights(&[0., 0.]).are_valid(None));
        assert!(!weights(&[1., -1.]).are_valid(None));
        assert!(!weights(&[1., f64::NAN]).are_valid(None));
        assert!(!weights(&[]).are_valid(None));
    }
}

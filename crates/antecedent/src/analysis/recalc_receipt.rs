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
        Self::ExternalInvocation(
            antecedent_core::recalc::Branch::new(0).expect("branch zero is valid"),
        ),
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

fn verify_entry(entry: &ReceiptEntry) -> Result<(), ReceiptError> {
    let stage = entry.stage;
    let required = match entry.status {
        StageStatus::Recomputed { .. } => required_work(stage),
        StageStatus::Reused { .. } | StageStatus::Refused { .. } => &[],
    };
    for counter in Counter::ALL.into_iter().chain([Counter::ModelFit]).chain(Counter::EXTENDED) {
        let n = entry.counts.get(counter);
        let allowed = matches!(entry.status, StageStatus::Recomputed { .. })
            && (counter.stage() == stage
                || matches!(
                    (counter, stage),
                    (Counter::ExternalInvocation(_), Stage::ProviderRequest(_))
                ));
        if n > 0 && !allowed {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
    }
    let callback_required = [Counter::ExternalInvocation(match stage {
        Stage::ProviderRequest(branch) => branch,
        _ => antecedent_core::recalc::Branch::new(0).expect("branch zero is valid"),
    })];
    let required = if matches!(stage, Stage::ProviderRequest(_))
        && matches!(entry.status, StageStatus::Recomputed { .. })
    {
        &callback_required[..]
    } else if stage == Stage::ScoreArtifact
        && matches!(entry.status, StageStatus::Recomputed { .. })
    {
        if entry.counts.posterior_draws > 0 {
            if entry.counts.fold_fits > 0
                || entry.counts.score_computations > 0
                || entry.counts.provider_bindings > 0
                || entry.counts.factor_builds > 0
                || entry.counts.program_compilations > 0
            {
                return Err(ReceiptError::UnexpectedWork { stage });
            }
            &[Counter::ModelFit, Counter::PosteriorDraw][..]
        } else if entry.counts.provider_bindings > 0 {
            if entry.counts.fold_fits > 0
                || entry.counts.model_fits > 0
                || entry.counts.score_computations > 0
            {
                return Err(ReceiptError::UnexpectedWork { stage });
            }
            &[Counter::ProviderBinding][..]
        } else if entry.counts.factor_builds > 0 || entry.counts.program_compilations > 0 {
            return Err(ReceiptError::MissingWork { stage, counter: Counter::ProviderBinding });
        } else if entry.counts.score_computations == 0 && entry.counts.model_fits > 0 {
            if entry.counts.fold_fits > 0 {
                return Err(ReceiptError::UnexpectedWork { stage });
            }
            &[Counter::ModelFit][..]
        } else {
            &[Counter::FoldFit, Counter::ScoreComputation][..]
        }
    } else if stage == Stage::Law
        && matches!(entry.status, StageStatus::Recomputed { .. })
        && entry.counts.law_summaries > 0
    {
        if entry.counts.reweights > 0
            || entry.counts.integrations > 0
            || entry.counts.factor_evaluations > 0
            || entry.counts.provider_calls > 0
        {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
        &[Counter::LawSummary][..]
    } else if stage == Stage::Law
        && matches!(entry.status, StageStatus::Recomputed { .. })
        && (entry.counts.integrations > 0
            || entry.counts.factor_evaluations > 0
            || entry.counts.provider_calls > 0)
    {
        if entry.counts.reweights > 0 {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
        &[Counter::Integration][..]
    } else {
        required
    };
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

impl UtilitySpec {
    pub(crate) fn validate(self) -> Result<(), RecalcRunError> {
        if !self.benefit_per_unit.is_finite() || !self.cost.is_finite() {
            return Err(RecalcRunError::Request("recalc.invalid_request"));
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
    pub columns: Vec<(String, Vec<f64>)>,
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

fn estimator_identity_bytes(estimator: &AipwAte) -> Vec<u8> {
    let mut out = b"aipw.v1".to_vec();
    out.extend_from_slice(&estimator.bootstrap_replicates.to_le_bytes());
    match estimator.overlap {
        antecedent_estimate::OverlapPolicy::ExplicitOverride => out.push(0),
        antecedent_estimate::OverlapPolicy::RequireDiagnostics { clip, trim } => {
            out.push(1);
            push_opt_f64(&mut out, clip);
            push_opt_f64(&mut out, trim);
        }
    }
    push_u32(&mut out, estimator.glm_options.max_iter);
    push_f64(&mut out, estimator.glm_options.tol);
    match estimator.glm_options.nb_alpha {
        antecedent_stats::NbAlphaPolicy::Fixed(alpha) => {
            out.push(0);
            push_f64(&mut out, alpha);
        }
        antecedent_stats::NbAlphaPolicy::MethodOfMoments => out.push(1),
        antecedent_stats::NbAlphaPolicy::NestedMle { max_outer, tol_alpha } => {
            out.push(2);
            push_u32(&mut out, max_outer);
            push_f64(&mut out, tol_alpha);
        }
    }
    push_opt_f64(&mut out, estimator.glm_options.ridge_on_separation);
    match estimator.se_kind {
        antecedent_estimate::AnalyticSeKind::Homoskedastic => out.push(0),
        antecedent_estimate::AnalyticSeKind::Hc0 => out.push(1),
        antecedent_estimate::AnalyticSeKind::Hc1 => out.push(2),
        antecedent_estimate::AnalyticSeKind::Hc2 => out.push(3),
        antecedent_estimate::AnalyticSeKind::Hc3 => out.push(4),
        antecedent_estimate::AnalyticSeKind::Cluster => out.push(5),
        antecedent_estimate::AnalyticSeKind::Multiway => out.push(6),
        antecedent_estimate::AnalyticSeKind::NeweyWest { lag } => {
            out.push(7);
            push_usize(&mut out, lag);
        }
        antecedent_estimate::AnalyticSeKind::PanelClusterHac { lag } => {
            out.push(8);
            push_usize(&mut out, lag);
        }
    }
    push_opt_u32s(&mut out, estimator.cluster_ids.as_deref());
    push_opt_u32s(&mut out, estimator.cluster_ids_second.as_deref());
    push_opt_panel(&mut out, estimator.panel_times.as_deref());
    match estimator.multiway_ids.as_deref() {
        None => out.push(0),
        Some(groups) => {
            out.push(1);
            push_usize(&mut out, groups.len());
            for group in groups {
                push_u32s(&mut out, group);
            }
        }
    }
    out.extend_from_slice(&estimator.propensity.identity_digest());
    match estimator.cluster_dml.as_ref() {
        None => out.push(0),
        Some(cluster) => {
            out.push(1);
            out.extend_from_slice(cluster.canonical_key().as_bytes());
        }
    }
    match estimator.population_registry.as_ref() {
        None => out.push(0),
        Some(registry) => {
            out.push(1);
            for (name, rows) in registry.predicates() {
                out.extend_from_slice(name.as_bytes());
                out.push(0);
                push_usize(&mut out, rows.len());
                for row in rows {
                    push_usize(&mut out, *row);
                }
            }
            out.push(0xff);
            for (id, weights, parents) in registry.distributions() {
                push_u32(&mut out, id.raw());
                push_f64s(&mut out, weights);
                push_usize(&mut out, parents.len());
                for parent in parents {
                    push_u32(&mut out, parent.raw());
                }
            }
        }
    }
    // Labels per source row are appended only when declared, so a configuration without
    // them keeps its earlier encoding.
    if let Some(labels) = estimator.source_cluster_ids.as_deref() {
        out.push(2);
        push_u32s(&mut out, labels);
    }
    if let Some(labels) = estimator.source_cluster_ids_second.as_deref() {
        out.push(3);
        push_u32s(&mut out, labels);
    }
    if let Some(weights) = estimator.sampling_weights.as_ref() {
        out.push(4);
        push_f64s(&mut out, weights.source());
    }
    out
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_usize(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn push_opt_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            push_f64(out, value);
        }
    }
}

fn push_u32s(out: &mut Vec<u8>, values: &[u32]) {
    push_usize(out, values.len());
    for value in values {
        push_u32(out, *value);
    }
}

fn push_opt_u32s(out: &mut Vec<u8>, values: Option<&[u32]>) {
    match values {
        None => out.push(0),
        Some(values) => {
            out.push(1);
            push_u32s(out, values);
        }
    }
}

fn push_opt_panel(out: &mut Vec<u8>, values: Option<&[i64]>) {
    match values {
        None => out.push(0),
        Some(values) => {
            out.push(1);
            push_usize(out, values.len());
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
}

fn push_f64s(out: &mut Vec<u8>, values: &[f64]) {
    push_usize(out, values.len());
    for value in values {
        push_f64(out, *value);
    }
}

impl RecalcRequest {
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        if self.estimator.bootstrap_replicates != 0 {
            return Err(RecalcRunError::Request("recalc.bootstrap_replicates_unsupported"));
        }
        if self.estimator.overlap == antecedent_estimate::OverlapPolicy::ExplicitOverride {
            return Err(RecalcRunError::Request("recalc.aipw_requires_overlap_diagnostics"));
        }
        if self.estimator.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
            || matches!(
                self.estimator.overlap,
                antecedent_estimate::OverlapPolicy::RequireDiagnostics { trim: Some(_), .. }
            )
        {
            return Err(RecalcRunError::Request("recalc.unsupported_aipw_score_configuration"));
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
        let rows = self.columns.first().map_or(0, |(_, v)| v.len());
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
            .ok_or(RecalcRunError::Request("recalc.score_table_unavailable"))?;
        (vec![1.0; rows], Vec::new())
    };
    let result = prepared.retarget(&weights, &depends_on, ctx)?;
    recorder.record(Counter::Reweight, 1);
    let effect = result
        .estimate
        .as_effect()
        .ok_or(RecalcRunError::Request("recalc.primary_result_is_not_an_effect"))?;
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
                return Err(RecalcRunError::Request("recalc.invalid_request"));
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
        return Err(RecalcRunError::Request("recalc.cancelled"));
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
            columns: vec![("t".into(), vec![0.0, 1.0]), ("y".into(), vec![0.0, 2.0])],
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

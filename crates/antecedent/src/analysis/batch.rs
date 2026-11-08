//! Batch multi-query: one table, N average-effect estimates.
//!
//! [`BatchStudy::prepare`] and [`BatchStudy::prepare_cells`] freeze a
//! [`SharedBatchDesign`]: one fold seed/count every query cross-fits with and, when
//! every query shares a certified adjustment set, one compiled `[1 | Z…]` covariate
//! design. Each query draws its own `crossfit_fold_plan` from that shared seed.
//! Cross-fitted nuisances (AIPW, learner-backed DML and doubly robust) are shared through one
//! [`antecedent_estimate::CrossfitNuisanceCache`] per estimate: a propensity or
//! outcome fit is reused only when every input it reads is bit-identical (rows,
//! treatment coding, adjustment design, sampling weights, fold plan, learner options and, for
//! the outcome, the outcome column), so a shared result equals the per-query fit bit
//! for bit. The cache is charged to the caller's [`ExecutionContext`] admission pool: its
//! keys and predictions evict under a small budget, and a fit that cannot be retained runs
//! unshared. See [`SharedBatchDesign`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, ExecutionContext, ResponseQuery, VariableId};

/// Query frozen on a [`PreparedBatch`] handle.
#[derive(Clone, Debug)]
pub enum BatchQuery {
    /// Average-effect claim.
    Average(AverageEffectQuery),
    /// Discrete joint / cell `InterventionResponse`.
    Response(ResponseQuery),
}

super::string_enum! {
    /// Contrast used for family p-values / FDR on a joint-cell batch.
    ///
    /// `estimate.ate` on `cell.aipw` is the requested interventional **level**, not
    /// a contrast. Family tests use a declared score-difference contrast instead.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum CellFamilyContrast, lookup = from_name {
        /// Requested cell minus the all-zero control (same contrast as cell.aipw refuters).
        CellMinusControl => "cell_minus_control",
        /// 2×2 non-additivity `+μ00 −μ10 −μ01 +μ11`.
        Interaction => "interaction",
    }
}
use antecedent_data::{TableView, TabularData, ValidityBitmap};
use antecedent_estimate::score_posterior::ScorePosteriorOptions;
use antecedent_estimate::{
    CrossfitNuisanceCache, DEFAULT_AIPW_FOLDS, NuisanceScopeUse, PreparedPropensityProblem,
};
use antecedent_graph::{Dag, TieredBackground};

use crate::error::CausalError;
use crate::result::StudyResult;

use super::batch_retarget::{
    BatchRetargetError, BatchRetargetReport, BatchRetargetRequest, BatchScores, ScoreSource,
    retarget_family,
};
use super::builder::{DataInput, RefuteSuite, StudyBuilder};
use super::execute::Study;
use super::latency::LatencyMode;
use super::prepared::PreparedStudy;
use crate::estimator_spec::EstimatorSpec;
use crate::strategy_table::{EstimatorId, IdentifierId};

/// How a batch family picked its surfaced candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateProcedure {
    /// Largest `|estimate|/SE` under the shared-row max-t family.
    MaxT,
    /// Smallest Benjamini–Hochberg q-value.
    BenjaminiHochberg,
    /// Smallest Benjamini–Yekutieli q-value.
    BenjaminiYekutieli,
    /// No screen was recorded; simultaneous inference covers only the fixed family.
    Unrecorded,
}

impl CandidateProcedure {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxT => "max_t",
            Self::BenjaminiHochberg => "bh",
            Self::BenjaminiYekutieli => "by",
            Self::Unrecorded => "unrecorded",
        }
    }
}

/// Declared screen / estimate split for a batch family.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScreen {
    /// Caller-stable id for the screen run.
    pub screen_id: std::sync::Arc<str>,
    /// Procedure used to pick the winner after the screen.
    pub procedure: CandidateProcedure,
    /// Row indexes used to screen candidates.
    pub screen_rows: std::sync::Arc<[antecedent_data::GlobalId]>,
    /// Row indexes used to estimate the surfaced family.
    pub estimate_rows: std::sync::Arc<[antecedent_data::GlobalId]>,
}

/// Provenance recorded on each batch result.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSelection {
    /// Screen run id, or `"unrecorded"`.
    pub screen_id: std::sync::Arc<str>,
    /// Selection rule.
    pub procedure: CandidateProcedure,
    /// Winning query index in the supplied family, or `None` when ranking is unavailable.
    pub winner_index: Option<usize>,
    /// Family size.
    pub family_size: usize,
    /// Screen half.
    pub screen_rows: std::sync::Arc<[antecedent_data::GlobalId]>,
    /// Estimate half.
    pub estimate_rows: std::sync::Arc<[antecedent_data::GlobalId]>,
    /// Whether the two halves are disjoint.
    pub disjoint: bool,
}

/// Compiled `[1 | Z…]` reused by every query that shares an adjustment set.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedCovariateDesign {
    /// Certified adjustment variables in design-column order (after the intercept).
    pub adjustment_set: Arc<[VariableId]>,
    /// Column-major `[1 | Z…]` on [`Self::row_index`] rows.
    pub design_matrix: Arc<[f64]>,
    /// `1 + adjustment_set.len()`.
    pub design_ncols: usize,
    /// Original-row index of each design row.
    pub row_index: Arc<[antecedent_data::GlobalId]>,
}

/// Shared fold seed/count and optional shared covariate design for a batch.
///
/// What is shared
///
/// - **Fold count and seed:** every query in the batch cross-fits with the same
///   `n_folds` and the same `fold_seed` (the run's master seed). That is enough for a
///   batch query and the identical query run solo to draw the *same* fold plan,
///   because both call the same `crossfit_fold_plan` with the same rows, the same
///   stratification (treatment arm for AIPW, joint cell for cell.aipw) and the same
///   seed — see [`Self::apply_to_propensity`]. [`Self::fold_ids`] is a seeded balanced
///   shuffle used only as an opaque identity fingerprint (batch-share digests,
///   introspection); it does not drive any query's actual cross-fit, precisely because
///   different queries in one batch can stratify differently (different treatments,
///   different cells) and no single row-level assignment could match all of them.
/// - **Covariates:** when every query's certified adjustment set is the same,
///   `[1 | Z…]` is compiled once and gathered into each propensity/cell design.
///
/// What is not shared
///
/// - **Folds actually used for cross-fitting:** each query draws its own plan (see
///   above); queries that share the same stratification (same treatment / same joint
///   cells) end up with the identical plan as a consequence, not because one is copied
///   into the other.
/// - **Propensity and outcome residualization** are not part of this object. A
///   batch estimate shares binary AIPW cross-fits through a
///   [`CrossfitNuisanceCache`] instead, keyed on every input the fit reads (rows,
///   treatment coding, adjustment design, fold plan, GLM options and, for the
///   outcome, the outcome column), so only identical fits are reused. Joint-cell
///   (`cell.aipw`) nuisances stay per query.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedBatchDesign {
    /// Opaque per-row identity fingerprint (not the fold plan any query actually
    /// cross-fits with; see the struct docs).
    pub fold_ids: Arc<[u32]>,
    /// Fold count every query in the batch cross-fits with.
    pub n_folds: u32,
    /// Master seed every query in the batch cross-fits with.
    pub fold_seed: u64,
    /// Shared covariate design, when adjustment sets agree.
    pub covariate: Option<SharedCovariateDesign>,
}

impl SharedBatchDesign {
    /// Compile a shared assignment for `data` and the identified adjustment sets.
    ///
    /// # Errors
    ///
    /// Missing adjustment columns or fold count `< 2`.
    pub fn compile(
        data: &TabularData,
        adjustment_sets: &[Arc<[VariableId]>],
        n_folds: u32,
        fold_seed: u64,
    ) -> Result<Self, CausalError> {
        if n_folds < 2 {
            return Err(CausalError::Compile {
                message: "shared batch design requires at least two folds".into(),
            });
        }
        let n = data.row_count();
        let folds = usize::try_from(n_folds).unwrap_or(usize::MAX);
        // A seeded balanced shuffle: fold membership does not follow the file order (a
        // periodic `row % k` assignment aliases with any periodic structure in the data).
        let mut rng = antecedent_core::CausalRng::from_seed(fold_seed ^ 0xF01D_5EED_BA7C_4001);
        let assignment = antecedent_kernels::shuffled_fold_assignment(&mut rng, n, folds, None)
            .map_err(|error| CausalError::Compile {
                message: match error {
                    antecedent_kernels::ViewError::InvalidShape { message } => message.to_owned(),
                    other => other.to_string(),
                },
            })?;
        let fold_ids: Arc<[u32]> = assignment
            .into_iter()
            .map(|f| u32::try_from(f).unwrap_or(u32::MAX))
            .collect::<Vec<_>>()
            .into();
        let covariate = match common_adjustment(adjustment_sets) {
            Some(set) if !set.is_empty() => Some(compile_shared_covariates(data, set)?),
            Some(_) | None => None,
        };
        Ok(Self { fold_ids, n_folds, fold_seed, covariate })
    }

    /// Rebuild data-dependent covariates and row folds on a new estimate table.
    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let sets: Vec<_> = self
            .covariate
            .as_ref()
            .map(|c| vec![Arc::clone(&c.adjustment_set)])
            .unwrap_or_default();
        Self::compile(data, &sets, self.n_folds, self.fold_seed)
    }

    /// Restrict [`Self::fold_ids`] to complete-case rows.
    ///
    /// # Errors
    ///
    /// A row index outside the prepared table.
    pub fn folds_for(
        &self,
        row_index: &[antecedent_data::GlobalId],
    ) -> Result<Vec<u32>, CausalError> {
        let mut out = Vec::with_capacity(row_index.len());
        for &row in row_index {
            let id = self
                .fold_ids
                .get(usize::try_from(row.get()).unwrap_or(usize::MAX))
                .copied()
                .ok_or_else(|| CausalError::Compile {
                message: "shared fold assignment does not cover a complete-case row".into(),
            })?;
            out.push(id);
        }
        Ok(out)
    }

    /// Gather the shared `[1 | Z…]` into `row_index` order when the set matches.
    ///
    /// # Errors
    ///
    /// A requested row missing from the shared design.
    pub fn design_for(
        &self,
        adjustment: &[VariableId],
        row_index: &[antecedent_data::GlobalId],
    ) -> Result<Option<Vec<f64>>, CausalError> {
        let Some(cov) = self.covariate.as_ref() else {
            return Ok(None);
        };
        if cov.adjustment_set.as_ref() != adjustment {
            return Ok(None);
        }
        let mut slot = vec![u32::MAX; self.fold_ids.len()];
        for (i, &row) in cov.row_index.iter().enumerate() {
            if let Some(entry) = slot.get_mut(usize::try_from(row.get()).unwrap_or(usize::MAX)) {
                *entry = u32::try_from(i).unwrap_or(u32::MAX);
            }
        }
        let n = row_index.len();
        let ncols = cov.design_ncols;
        let mut out = vec![0.0; n * ncols];
        for (i, &row) in row_index.iter().enumerate() {
            let src = slot
                .get(usize::try_from(row.get()).unwrap_or(usize::MAX))
                .copied()
                .unwrap_or(u32::MAX);
            if src == u32::MAX {
                return Err(CausalError::Compile {
                    message: "shared covariate design is missing a complete-case row".into(),
                });
            }
            let src = src as usize;
            for c in 0..ncols {
                out[c * n + i] = cov.design_matrix[c * cov.row_index.len() + src];
            }
        }
        Ok(Some(out))
    }

    /// Apply the shared design matrix, when the query's adjustment set matches it.
    ///
    /// Folds are deliberately *not* overridden here: `problem.fold_assignment` stays
    /// `None` so the propensity/outcome cross-fit falls back to its own
    /// `crossfit_fold_plan` call, stratified by this query's treatment arm and keyed by
    /// `problem.fold_seed` (the batch's master seed). That is the same call a solo
    /// analysis makes for the same query, seed, and rows, so it reproduces the solo
    /// fold plan bit-for-bit instead of drawing a private, unstratified shuffle that a
    /// solo run would never draw. See the batch design documentation.
    ///
    /// # Errors
    ///
    /// Row or shape mismatch.
    pub fn apply_to_propensity(
        &self,
        problem: &mut PreparedPropensityProblem,
    ) -> Result<bool, CausalError> {
        if let Some(design) = self.design_for(&problem.adjustment_set, &problem.row_index)? {
            if design.len() != problem.design_matrix.len() {
                return Err(CausalError::Compile {
                    message: "shared covariate design shape does not match the propensity problem"
                        .into(),
                });
            }
            // The shared design replaces the query's own prepared design, so it must carry the
            // same complete-case rows with the same values in every column.
            let ncols = problem.design_ncols;
            let own = design_frame(&problem.row_index, &problem.design_matrix, ncols)?;
            let shared = design_frame(&problem.row_index, &design, ncols)?;
            antecedent_contract::rewrite_legality::check_executed(
                antecedent_contract::rewrite_legality::Rewrite::SHARED_ROW_DESIGN_PREPARATION,
                &own,
                &[shared],
            )
            .map_err(|refusal| CausalError::Compile { message: refusal.to_string() })?;
            problem.design_matrix = design.into();
            problem.shared_design = true;
            return Ok(true);
        }
        Ok(false)
    }
}

/// Executed frame of a column-major design: each complete-case row with its values.
fn design_frame(
    row_index: &[antecedent_data::GlobalId],
    design: &[f64],
    ncols: usize,
) -> Result<antecedent_contract::rewrite_legality::FrameFingerprint, CausalError> {
    use antecedent_contract::rewrite_legality::{FrameLedger, value_row_id};
    let n = row_index.len();
    let refused = |refusal: antecedent_contract::rewrite_legality::RewriteRefusal| {
        CausalError::Compile { message: refusal.to_string() }
    };
    let mut ledger =
        FrameLedger::try_new("complete_case_design", "static_cross_section", "unweighted")
            .map_err(refused)?;
    for (i, row) in row_index.iter().enumerate() {
        let id = value_row_id(
            row.get(),
            (0..ncols).map(|c| design.get(c * n + i).copied().unwrap_or(f64::NAN)),
        );
        ledger.record(&id, "0", &row.get().to_string(), "unweighted");
    }
    Ok(ledger.fingerprint())
}

fn common_adjustment(sets: &[Arc<[VariableId]>]) -> Option<&[VariableId]> {
    let first = sets.first()?.as_ref();
    sets.iter().all(|s| s.as_ref() == first).then_some(first)
}

fn compile_shared_covariates(
    data: &TabularData,
    adjustment: &[VariableId],
) -> Result<SharedCovariateDesign, CausalError> {
    let mask = data.complete_case_mask(adjustment).map_err(CausalError::from)?;
    let n = mask.iter().filter(|keep| **keep).count();
    if n < 2 {
        return Err(CausalError::Compile {
            message: "shared covariate design needs at least two complete adjustment rows".into(),
        });
    }
    let ncols = 1 + adjustment.len();
    let mut design = vec![0.0; n * ncols];
    for slot in design.iter_mut().take(n) {
        *slot = 1.0;
    }
    for (j, &z) in adjustment.iter().enumerate() {
        let col = data.float64_masked(z, &mask).map_err(CausalError::from)?;
        design[(1 + j) * n..(1 + j) * n + n].copy_from_slice(&col);
    }
    let mut row_index = Vec::with_capacity(n);
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            row_index.push(antecedent_data::GlobalId::new(u64::try_from(i).expect("row id")));
        }
    }
    Ok(SharedCovariateDesign {
        adjustment_set: Arc::from(adjustment.to_vec()),
        design_matrix: design.into(),
        design_ncols: ncols,
        row_index: row_index.into(),
    })
}

#[derive(Clone, Debug)]
enum BatchGraph {
    Dag(Dag),
    Tiered(TieredBackground),
}

/// Shared-table batch of static average-effect queries.
///
/// Binds data once. Prepared plans share one [`SharedBatchDesign`] (folds and,
/// when supported, the covariate matrix). Estimation still runs per-query plans
/// in parallel chunks.
#[derive(Clone, Debug)]
pub struct BatchStudy {
    data_diagnostics: Option<crate::DataDiagnostics>,
    data: TabularData,
    graph: BatchGraph,
    /// `None` leaves the per-query builder's omitted table and latency tier in force.
    bootstrap_replicates: Option<u32>,
    /// `None` leaves the per-query builder's omitted suite (and its downgrade) in force.
    refute: Option<RefuteSuite>,
    latency_mode: Option<LatencyMode>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    /// A fully configured estimator (the typed configuration single-query `analyze`
    /// accepts). Supersedes [`Self::estimator`]'s bare id when set.
    estimator_spec: Option<EstimatorSpec>,
    screen: Option<CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
    family_posterior: Option<ScorePosteriorOptions>,
    /// What the batch's nuisance cache may keep ([`Self::retention`]).
    retention: antecedent_state::retention::RetentionMode,
}

/// The nuisance cache of one batch call: charged to the caller's admission pool, and empty by
/// construction when `retention` does not allow fitted state to be kept.
pub(crate) fn batch_nuisance_cache(
    ctx: &ExecutionContext,
    retention: antecedent_state::retention::RetentionMode,
) -> Arc<CrossfitNuisanceCache> {
    if retention.retains_fitted_state() {
        CrossfitNuisanceCache::charged_to(ctx)
    } else {
        CrossfitNuisanceCache::bounded(Some(ctx), 0, 0)
    }
}

impl BatchStudy {
    /// Configure automatic fit-free diagnostics for every member; `None` disables them.
    #[must_use]
    pub fn data_diagnostics(mut self, policy: Option<crate::DataDiagnostics>) -> Self {
        self.data_diagnostics = policy;
        self
    }

    /// A batch over `data` and `graph` with every option at its default.
    fn with_graph(data: TabularData, graph: BatchGraph) -> Self {
        Self {
            data,
            graph,
            data_diagnostics: Some(crate::DataDiagnostics::default()),
            bootstrap_replicates: None,
            refute: None,
            latency_mode: None,
            identifier: None,
            estimator: None,
            estimator_spec: None,
            screen: None,
            family_contrast: Some(CellFamilyContrast::CellMinusControl),
            family_posterior: None,
            retention: antecedent_state::retention::RetentionMode::FullReusable,
        }
    }

    /// Start a batch over `data` and a static DAG.
    #[must_use]
    pub fn new(data: TabularData, graph: Dag) -> Self {
        Self::with_graph(data, BatchGraph::Dag(graph))
    }

    /// Start a batch over a declared [`TieredBackground`].
    ///
    /// `CoDetermined` materializes as an ADMG; Unknown as a PAG. Identification
    /// uses the same [`crate::StudyBuilder::tiered_background`] path as a single-query study.
    #[must_use]
    pub fn tiered(data: TabularData, background: TieredBackground) -> Self {
        Self::with_graph(data, BatchGraph::Tiered(background))
    }

    /// Record a screen/estimate split for candidate-selection provenance.
    #[must_use]
    pub fn candidate_screen(mut self, screen: CandidateScreen) -> Self {
        self.screen = Some(screen);
        self
    }

    /// How results are retained. Only [`RetentionMode::FullReusable`] lets the batch keep
    /// fitted nuisances for reuse; a reduced mode builds an empty nuisance cache, so every
    /// fit runs unshared and no hidden cache keeps what the mode dropped.
    ///
    /// [`RetentionMode::FullReusable`]: antecedent_state::retention::RetentionMode::FullReusable
    #[must_use]
    pub fn retention(mut self, mode: antecedent_state::retention::RetentionMode) -> Self {
        self.retention = mode;
        self
    }

    /// Bootstrap replicates for every query.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = Some(n);
        self
    }

    /// Refute suite for every query.
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = Some(suite);
        self
    }

    /// Optional latency tier applied to every query.
    #[must_use]
    pub fn latency_mode(mut self, mode: LatencyMode) -> Self {
        self.latency_mode = Some(mode);
        self
    }

    /// Optional identification strategy applied to every query.
    ///
    /// Parse a wire name with `"backdoor.adjustment".parse::<IdentifierId>()?`.
    #[must_use]
    pub const fn identifier(mut self, id: IdentifierId) -> Self {
        self.identifier = Some(id);
        self
    }

    /// Optional estimator applied to every query.
    ///
    /// Parse a wire name with `"propensity.weighting".parse::<EstimatorId>()?`. Replaces a
    /// configuration declared earlier with [`Self::estimator_spec`].
    #[must_use]
    pub fn estimator(mut self, id: EstimatorId) -> Self {
        self.estimator = Some(id);
        self.estimator_spec = None;
        self
    }

    /// A fully configured estimator applied to every query, the typed configuration
    /// single-query [`StudyBuilder::estimator`] takes (for example an
    /// [`antecedent_estimate::AipwAte`] with a ridge propensity).
    ///
    /// One configuration governs the whole batch, so every nuisance fit a batch shares was
    /// declared by it: a different penalty, learner option, outcome, treatment coding,
    /// adjustment set, fold plan or overlap rule changes the fit's inputs and is never served
    /// from another fit (a penalized propensity is never shared at all). As for a single
    /// query, combining a configured estimator with [`Self::bootstrap_replicates`] is refused
    /// when the study is built. [`Self::estimator_fingerprint`] names the configuration.
    #[must_use]
    pub fn estimator_spec(mut self, spec: impl Into<EstimatorSpec>) -> Self {
        let spec = spec.into();
        self.estimator = Some(spec.id());
        self.estimator_spec = Some(spec);
        self
    }

    /// Canonical fingerprint of the estimator configuration every query of this batch runs
    /// under: the structured estimator-spec identity (the same one a study's contract
    /// carries, including a non-default propensity nuisance), or the bare estimator id, or
    /// `"default"` when none was chosen. Equal fingerprints mean one configuration.
    ///
    /// # Errors
    ///
    /// The estimator-spec identity cannot be encoded; a failure never becomes a fingerprint
    /// string that could be compared or stored.
    pub fn estimator_fingerprint(&self) -> Result<String, CausalError> {
        match (&self.estimator_spec, self.estimator) {
            (Some(spec), _) => Ok(antecedent_io::digest_hex(
                antecedent_core::IdentityDomain::InferenceBinding,
                &super::contract_identity::estimator_spec_identity(spec),
            )?),
            (None, Some(id)) => Ok(id.as_str().to_string()),
            (None, None) => Ok("default".to_string()),
        }
    }

    /// Contrast used for family p-values on [`Self::prepare_cells`].
    ///
    /// Default is [`CellFamilyContrast::CellMinusControl`]. `None` publishes
    /// simultaneous intervals on cell levels and omits p-values.
    #[must_use]
    pub const fn family_contrast(mut self, contrast: Option<CellFamilyContrast>) -> Self {
        self.family_contrast = contrast;
        self
    }

    /// Also publish the Bayesian-bootstrap posterior of the declared family contrast on
    /// [`Self::prepare_cells`] results: the frozen cross-fit scores are reweighted by
    /// Dirichlet(1, ..., 1) row weights shared by every member of the family, so the draws
    /// are one aligned joint posterior of the contrast (for the interaction, the complete
    /// non-additivity, not a difference of marginal intervals). Conditional on the cross-fit
    /// nuisances; calibration is `not_measured`. `None` publishes no posterior.
    #[must_use]
    pub const fn family_posterior(mut self, posterior: Option<ScorePosteriorOptions>) -> Self {
        self.family_posterior = posterior;
        self
    }

    /// Estimate each query against the shared table.
    ///
    /// Shared fold assignment and covariate design when the batch uses AIPW /
    /// score tables, and shared cross-fitted AIPW nuisances wherever two queries
    /// would fit the identical propensity or outcome model (same rows, treatment
    /// coding, adjustment set, fold plan and learner options). Each result is
    /// bit-identical to the query's own per-query fit. Queries run in parallel
    /// under `ctx.parallelism.max_threads`.
    /// After the batch, joint IF covariance across claims that share rows
    /// supplies max-t and BH/BY diagnostics. An overlap validator failure
    /// does not abort a claim.
    ///
    /// # Errors
    ///
    /// Empty query list, or any per-query analysis failure.
    pub fn estimate_many(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyResult>, CausalError> {
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch estimate_many requires at least one query".into(),
            });
        }
        let fingerprint = self.estimator_fingerprint()?;
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let shared = self.compile_shared_ate(&data, queries, ctx)?;
        let nuisances = batch_nuisance_cache(ctx, self.retention);
        let completed = ctx
            .try_map_indexed(queries.len(), |index, inner| {
                if inner.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                    });
                }
                let (result, used) = nuisances
                    .scope(|| self.run_one_on(&data, &queries[index], inner, shared.as_ref()));
                result.map(|result| (result, used))
            })
            .map_err(batch_map_error)?;
        let (mut out, uses): (Vec<_>, Vec<_>) = completed.into_iter().unzip();
        attach_batch_joint_inference(&mut out, &data, queries, self.screen.as_ref());
        attach_shared_design_diagnostics(
            &mut out,
            shared.as_ref(),
            &nuisances,
            &uses,
            &fingerprint,
        );
        Ok(out)
    }

    /// Compile each query once into a reusable prepared batch.
    ///
    /// Identification and score tables are frozen per query. A later
    /// [`PreparedBatch::estimate`] reuses those plans on new rows under the
    /// same schema.
    ///
    /// # Errors
    ///
    /// Empty query list or any per-query prepare failure.
    pub fn prepare(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        self.prepare_mode(queries, ctx, true)
    }

    /// Identify, lower and quote a batch without fitting operations or score tables.
    ///
    /// # Errors
    /// Empty query list, unsupported declarations, or preparation admission failures.
    pub fn prepare_deferred(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        self.prepare_mode(queries, ctx, false)
    }

    fn prepare_mode(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
        materialized: bool,
    ) -> Result<PreparedBatch, CausalError> {
        let _diagnostic_profiles = super::data_diagnostics::batch_profiles();
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch prepare requires at least one query".into(),
            });
        }
        let fingerprint = self.estimator_fingerprint()?;
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let shared = self.compile_shared_ate(&data, queries, ctx)?;
        let nuisances = batch_nuisance_cache(ctx, self.retention);
        let mut plans = Vec::with_capacity(queries.len());
        for query in queries {
            let mut study = self.study_for_data(&data, query)?;
            study.shared_batch_design.clone_from(&shared);
            plans.push(
                nuisances
                    .scope(|| {
                        if materialized { study.prepare(ctx) } else { study.prepare_deferred(ctx) }
                    })
                    .0?,
            );
        }
        Ok(PreparedBatch {
            retention: self.retention,
            plans,
            queries: queries.iter().cloned().map(BatchQuery::Average).collect(),
            screen: self.screen.clone(),
            shared_design: shared,
            family_contrast: None,
            family_posterior: None,
            estimator_fingerprint: Arc::from(fingerprint),
            execution_owner: materialized.then(|| BatchExecutionOwner::capture(ctx)).transpose()?,
        })
    }

    /// Compile discrete joint `InterventionResponse` cells into a reusable prepared batch.
    ///
    /// Defaults to [`EstimatorId::CellAipw`] when the batch has no estimator override.
    /// Supported on a DAG and on `CoDetermined` via joint ADMG adjustment on the
    /// known closure graph. Unknown has no single ADMG. Bare ADMG
    /// `InterventionResponse` already refuses.
    ///
    /// Family p-values use [`Self::family_contrast`] (default
    /// [`CellFamilyContrast::CellMinusControl`]), never the cell level stored
    /// in `estimate.ate`. Max-t and `family_contrast_interval` are formed on
    /// that contrast family. `family_contrast(None)` keeps intervals on levels
    /// and publishes no p-values.
    ///
    /// # Errors
    ///
    /// Empty query list; a query outside the covered family (not a static
    /// `InterventionResponse` setting two or more variables), refused with
    /// `route_not_supported` naming `analyze` as the per-query route; Unknown-tier
    /// joint (no single ADMG); or any per-query prepare failure.
    pub fn prepare_cells(
        &self,
        queries: &[ResponseQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        self.prepare_cells_mode(queries, ctx, true)
    }

    /// Prepare supported joint-cell declarations without fitting operations or scores.
    ///
    /// # Errors
    /// Empty or unsupported query family, or preparation admission failures.
    pub fn prepare_cells_deferred(
        &self,
        queries: &[ResponseQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        self.prepare_cells_mode(queries, ctx, false)
    }

    fn prepare_cells_mode(
        &self,
        queries: &[ResponseQuery],
        ctx: &ExecutionContext,
        materialized: bool,
    ) -> Result<PreparedBatch, CausalError> {
        let _diagnostic_profiles = super::data_diagnostics::batch_profiles();
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch prepare_cells requires at least one query".into(),
            });
        }
        if !queries.iter().all(is_discrete_joint_cell) {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "batch prepare_cells covers static discrete joint InterventionResponse cells \
                 (two or more Set interventions) only; estimate any other query one at a time \
                 with analyze (Study::run)"
            ));
        }
        let fingerprint = self.estimator_fingerprint()?;
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let shared = self.compile_shared_cells(&data, queries, ctx)?;
        let mut plans = Vec::with_capacity(queries.len());
        let mut frozen = Vec::with_capacity(queries.len());
        for query in queries {
            let mut study = self.study_for_response_on(&data, query)?;
            study.shared_batch_design.clone_from(&shared);
            plans.push(if materialized {
                study.prepare(ctx)?
            } else {
                study.prepare_deferred(ctx)?
            });
            frozen.push(BatchQuery::Response(query.clone()));
        }
        Ok(PreparedBatch {
            retention: self.retention,
            plans,
            queries: frozen,
            screen: self.screen.clone(),
            shared_design: shared,
            family_contrast: self.family_contrast,
            family_posterior: self.family_posterior,
            estimator_fingerprint: Arc::from(fingerprint),
            execution_owner: materialized.then(|| BatchExecutionOwner::capture(ctx)).transpose()?,
        })
    }

    /// The per-query builder with this batch's refutation, bootstrap, latency, identifier and
    /// estimator settings applied. `default_estimator` stands in when the batch names none.
    fn configure(
        &self,
        mut builder: StudyBuilder,
        default_estimator: Option<EstimatorId>,
    ) -> StudyBuilder {
        builder = builder.data_diagnostics(self.data_diagnostics.clone());
        if let Some(suite) = self.refute {
            builder = builder.refute(suite);
        }
        if let Some(replicates) = self.bootstrap_replicates {
            builder = builder.bootstrap_replicates(replicates);
        }
        if let Some(mode) = self.latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(id) = self.identifier {
            builder = builder.identifier(id);
        }
        if let Some(spec) = &self.estimator_spec {
            builder = builder.estimator(spec.clone());
        } else if let Some(est) = self.estimator.or(default_estimator) {
            builder = builder.estimator(est);
        }
        builder
    }

    fn study_for_response_on(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
    ) -> Result<Study, CausalError> {
        let builder = self.bind_graph(Study::tabular(data.clone()))?.query(query.clone());
        self.configure(builder, Some(EstimatorId::CellAipw)).build()
    }

    fn compile_shared_ate(
        &self,
        data: &TabularData,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<Option<Arc<SharedBatchDesign>>, CausalError> {
        self.compile_shared(data, queries, ctx, Self::ate_adjustment_set)
    }

    fn compile_shared_cells(
        &self,
        data: &TabularData,
        queries: &[ResponseQuery],
        ctx: &ExecutionContext,
    ) -> Result<Option<Arc<SharedBatchDesign>>, CausalError> {
        self.compile_shared(data, queries, ctx, Self::cell_adjustment_set)
    }

    /// The shared cross-fit design over the adjustment sets `adjustment_set` selects for each
    /// of `queries`. Each set is read from the compiled identification of that query's study
    /// under the caller's context, so the shared work is admitted by, and cancelled with, the
    /// call that asked for it; the fold seed is the context's master seed.
    #[allow(
        clippy::type_complexity,
        reason = "the signature describes the existing typed operation"
    )]
    fn compile_shared<Q>(
        &self,
        data: &TabularData,
        queries: &[Q],
        ctx: &ExecutionContext,
        adjustment_set: fn(
            &Self,
            &TabularData,
            &Q,
            &ExecutionContext,
        ) -> Result<Arc<[VariableId]>, CausalError>,
    ) -> Result<Option<Arc<SharedBatchDesign>>, CausalError> {
        let mut sets = Vec::with_capacity(queries.len());
        for query in queries {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_IDENTIFY,
                });
            }
            sets.push(adjustment_set(self, data, query, ctx)?);
        }
        Ok(Some(Arc::new(SharedBatchDesign::compile(
            data,
            &sets,
            u32::try_from(DEFAULT_AIPW_FOLDS).unwrap_or(5),
            ctx.rng.master_seed(),
        )?)))
    }

    fn run_one_on(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
        shared: Option<&Arc<SharedBatchDesign>>,
    ) -> Result<StudyResult, CausalError> {
        let mut study = self.study_for_data(data, query)?;
        study.shared_batch_design = shared.cloned();
        study.run(ctx)
    }

    fn study_for_data(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
    ) -> Result<Study, CausalError> {
        let builder = self.bind_graph(Study::tabular(data.clone()))?.query(query.clone());
        self.configure(builder, None).build()
    }

    fn bind_graph(&self, builder: StudyBuilder) -> Result<StudyBuilder, CausalError> {
        match &self.graph {
            BatchGraph::Dag(graph) => Ok(builder.graph(graph.clone())),
            BatchGraph::Tiered(background) => builder.tiered_background(background.clone()),
        }
    }

    /// The adjustment set of the study this batch would build for `query`, read from that
    /// study's compiled identification under the caller's context.
    fn ate_adjustment_set(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<Arc<[VariableId]>, CausalError> {
        if let BatchGraph::Tiered(background) = &self.graph
            && background.within_tier == antecedent_graph::WithinTier::Unknown
        {
            return Ok(Arc::from([]));
        }
        let identification = self.study_for_data(data, query)?.identify_structure_within(ctx)?;
        let crate::Identification::Point { result, .. } = &identification else {
            return Err(CausalError::Compile {
                message: "batch identification is not a single-graph product".into(),
            });
        };
        match &self.graph {
            BatchGraph::Dag(_) => {
                let estimator = self.estimator.unwrap_or(EstimatorId::Aipw);
                Ok(crate::strategy_table::select_estimand(result, estimator)?.adjustment_set)
            }
            BatchGraph::Tiered(_) => {
                result.estimands.first().map(|e| Arc::clone(&e.adjustment_set)).ok_or_else(|| {
                    CausalError::Compile {
                        message: "tiered batch identifier returned no estimand".into(),
                    }
                })
            }
        }
    }

    fn cell_adjustment_set(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<Arc<[VariableId]>, CausalError> {
        let identification =
            self.study_for_response_on(data, query)?.identify_structure_within(ctx)?;
        let crate::Identification::Point { result, .. } = &identification else {
            return Err(CausalError::Compile {
                message: "batch prepare_cells identification is not a single-graph product".into(),
            });
        };
        let first = result.estimands.first().map(|e| Arc::clone(&e.adjustment_set));
        match &self.graph {
            BatchGraph::Dag(_) => first.ok_or_else(|| CausalError::Compile {
                message: "batch prepare_cells identifier returned no estimand".into(),
            }),
            BatchGraph::Tiered(_) => first.ok_or(CausalError::Unsupported {
                message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
            }),
        }
    }
}

/// Whether `query` is a cell a [`BatchStudy::prepare_cells`] family covers: a static
/// `InterventionResponse` that sets two or more variables.
fn is_discrete_joint_cell(query: &ResponseQuery) -> bool {
    query.temporal.is_none()
        && matches!(
            &query.functional,
            antecedent_core::ResponseFunctional::InterventionResponse { interventions, .. }
                if interventions.len() >= 2
                    && interventions
                        .iter()
                        .all(|iv| matches!(iv, antecedent_core::Intervention::Set { .. }))
        )
}

/// Ownership and numerical policies under which retained batch state was materialized.
///
/// The zero-sized lease is only a pool-identity witness; it does not claim to reserve fit
/// bytes. Actual numerical allocations keep their own leases in that same admission pool.
#[derive(Clone, Debug)]
struct BatchExecutionOwner {
    pool_identity: Arc<antecedent_core::execution::AdmissionLease>,
    cache_policy: antecedent_core::execution::CachePolicy,
    kernel_policy: antecedent_core::execution::KernelPolicy,
    determinism: antecedent_core::execution::Determinism,
}

impl BatchExecutionOwner {
    fn capture(ctx: &ExecutionContext) -> Result<Self, CausalError> {
        let lease = ctx.acquire_admission(0, 0).map_err(|error| {
            crate::error::resource_admission_error(
                error,
                crate::analysis::stage::STAGE_ESTIMATE_POINT,
            )
        })?;
        Ok(Self {
            pool_identity: Arc::new(lease),
            cache_policy: ctx.cache_policy,
            kernel_policy: ctx.kernel_policy,
            determinism: ctx.determinism,
        })
    }

    fn compatible(&self, ctx: &ExecutionContext) -> bool {
        self.pool_identity.belongs_to(&ctx.admission())
            && self.cache_policy == ctx.cache_policy
            && self.kernel_policy == ctx.kernel_policy
            && self.determinism == ctx.determinism
    }
}

/// Frozen batch of prepared average-effect (and joint-cell) plans.
#[derive(Clone, Debug)]
pub struct PreparedBatch {
    retention: antecedent_state::retention::RetentionMode,
    plans: Vec<PreparedStudy>,
    queries: Vec<BatchQuery>,
    screen: Option<CandidateScreen>,
    shared_design: Option<Arc<SharedBatchDesign>>,
    family_contrast: Option<CellFamilyContrast>,
    family_posterior: Option<ScorePosteriorOptions>,
    /// [`BatchStudy::estimator_fingerprint`] of the batch this was prepared from.
    estimator_fingerprint: Arc<str>,
    execution_owner: Option<BatchExecutionOwner>,
}

/// Immutable executed family, retaining the exact materialized plans and estimated scores.
///
/// Results include the final family covariance, contrasts and sharing diagnostics. Retargets
/// read this execution's score snapshot; another execution cannot replace those scores.
#[derive(Clone, Debug)]
pub struct CompletedBatch {
    prepared: PreparedBatch,
    results: Vec<Arc<StudyResult>>,
    scores: BatchScores,
}

impl CompletedBatch {
    /// The exact executed observations, shared without copying their column storage.
    ///
    /// # Errors
    /// A malformed family with no bound tabular plan.
    pub fn bound_table(&self) -> Result<&TabularData, CausalError> {
        self.prepared.bound_data()
    }

    /// Materialized plans bound to the observations this execution used.
    #[must_use]
    pub fn prepared(&self) -> &PreparedBatch {
        &self.prepared
    }

    /// Final scientific results in declared query order.
    #[must_use]
    pub fn results(&self) -> &[Arc<StudyResult>] {
        &self.results
    }

    /// The study's bibliography: the sources of every result, de-duplicated across results,
    /// each with the caveats of the records citing it (see [`crate::bibliography`]).
    #[must_use = "the bibliography is computed on each call"]
    pub fn bibliography(&self) -> Vec<antecedent_io::method_provenance::CitationWire> {
        crate::bibliography(self.results.iter().map(AsRef::as_ref))
    }

    /// Estimated score tables, including empty slots where retention was unavailable.
    #[must_use]
    pub fn scores(&self) -> &BatchScores {
        &self.scores
    }

    /// Consume the execution without copying its retained state.
    #[must_use]
    pub fn into_parts(self) -> (PreparedBatch, Vec<Arc<StudyResult>>, BatchScores) {
        (self.prepared, self.results, self.scores)
    }

    /// Retarget this execution's retained scores under an explicit population declaration.
    ///
    /// # Errors
    /// Invalid families, unavailable scores, incompatible row snapshots, or cancellation.
    pub fn retarget(
        &self,
        request: &BatchRetargetRequest,
        ctx: &ExecutionContext,
    ) -> Result<BatchRetargetReport, BatchRetargetError> {
        self.prepared.retarget(&self.scores, request, ctx)
    }
}

impl PreparedBatch {
    /// Queries frozen on this handle.
    #[must_use]
    pub fn queries(&self) -> &[BatchQuery] {
        &self.queries
    }

    /// Prepared plans, one per query, sharing the batch schema.
    #[must_use]
    pub fn plans(&self) -> &[PreparedStudy] {
        &self.plans
    }

    /// Shared fold assignment and covariate design frozen at prepare.
    #[must_use]
    pub fn shared_design(&self) -> Option<&SharedBatchDesign> {
        self.shared_design.as_deref()
    }

    /// Install per-member observers, allowing adapters to attach query identity to events.
    ///
    /// The sinks are execution controls and are stripped from completed native plans.
    ///
    /// # Errors
    /// The observer count differs from the family size, or a member cannot stream stages.
    pub fn with_stage_sinks(
        &self,
        sinks: Vec<Arc<dyn crate::analysis::stage::StageResultSink>>,
    ) -> Result<Self, CausalError> {
        if sinks.len() != self.plans.len() {
            return Err(CausalError::Compile {
                message: "batch stage observers require one sink per query".into(),
            });
        }
        if self.plans.iter().any(|plan| !plan.streams_stages()) {
            return Err(crate::unsupported_reason!(
                "stage_stream_unavailable",
                "batch stage observers require every member to support progressive scalar stages"
            ));
        }
        let plans = self
            .plans
            .iter()
            .zip(sinks)
            .map(|(plan, sink)| {
                let mut private = plan.clone();
                private.set_stage_sink(Some(sink));
                private
            })
            .collect();
        Ok(self.with_plans(plans))
    }

    /// Branch onto new observations without fitting or changing this handle.
    ///
    /// The declared candidate-screen estimate subset is applied once, before binding plans.
    /// Identification can be retained; row-bound fits and scores are invalidated.
    ///
    /// # Errors
    /// Incompatible schema or screen rows, preparation admission failures, or cancellation.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the operation accepts owned input as part of its interface"
    )]
    pub fn rebind_data(
        &self,
        data: TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let data = subset_estimate_rows(&data, self.screen.as_ref())?;
        self.branch_bound(data, ctx)
    }

    /// Requote the bound observations and shared design under execution controls, fit-free.
    ///
    /// A new seed rebuilds the shared fold fingerprint and all seed-bound statistical plans.
    ///
    /// # Errors
    /// Preparation admission failures or cancellation.
    pub fn reconfigure_execution(&self, ctx: &ExecutionContext) -> Result<Self, CausalError> {
        if self
            .shared_design
            .as_ref()
            .is_some_and(|design| design.fold_seed != ctx.rng.master_seed())
            || self.execution_owner.as_ref().is_some_and(|owner| !owner.compatible(ctx))
        {
            return self.branch_bound(self.bound_data()?.clone(), ctx);
        }
        let plans = self
            .plans
            .iter()
            .map(|plan| plan.requote_batch_execution(ctx))
            .collect::<Result<Vec<_>, _>>()?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        Ok(self.with_plans(plans))
    }

    /// Count of materialized member operations retained by this immutable branch.
    #[must_use]
    pub fn materialized_plan_count(&self) -> usize {
        self.plans.iter().filter(|plan| plan.is_materialized()).count()
    }

    /// Count of retained per-member score tables; a deferred branch has none.
    #[must_use]
    pub fn retained_score_table_count(&self) -> usize {
        self.plans.iter().filter(|plan| plan.score_table().is_some()).count()
    }

    /// Copy family declarations without cloning plans that the caller is replacing.
    fn with_plans(&self, plans: Vec<PreparedStudy>) -> Self {
        Self {
            retention: self.retention,
            plans,
            queries: self.queries.clone(),
            screen: self.screen.clone(),
            shared_design: self.shared_design.clone(),
            family_contrast: self.family_contrast,
            family_posterior: self.family_posterior,
            estimator_fingerprint: Arc::clone(&self.estimator_fingerprint),
            execution_owner: self.execution_owner.clone(),
        }
    }

    fn bound_data(&self) -> Result<&TabularData, CausalError> {
        match self.plans.first().map(|plan| &plan.study().data) {
            Some(DataInput::Tabular(data)) => Ok(data),
            _ => Err(CausalError::Compile {
                message: "a prepared batch requires at least one bound tabular plan".into(),
            }),
        }
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "the operation accepts owned input as part of its interface"
    )]
    fn branch_bound(&self, data: TabularData, ctx: &ExecutionContext) -> Result<Self, CausalError> {
        let shared = self
            .shared_design
            .as_ref()
            .map(|design| {
                let adjustment_sets = design
                    .covariate
                    .as_ref()
                    .map(|covariate| vec![Arc::clone(&covariate.adjustment_set)])
                    .unwrap_or_default();
                SharedBatchDesign::compile(
                    &data,
                    &adjustment_sets,
                    design.n_folds,
                    ctx.rng.master_seed(),
                )
                .map(Arc::new)
            })
            .transpose()?;
        let plans = self
            .plans
            .iter()
            .map(|plan| plan.rebind_batch_data(data.clone(), shared.clone(), ctx))
            .collect::<Result<Vec<_>, _>>()?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        let mut branch = self.with_plans(plans);
        branch.shared_design = shared;
        branch.execution_owner = None;
        Ok(branch)
    }

    /// Explicitly execute the bound family, retaining the exact materialized native plans.
    ///
    /// Materialization and execution share one nuisance scope per member. Family inference
    /// is attached before any result is frozen. Failure or cancellation publishes no state.
    ///
    /// # Errors
    /// Statistical, resource or cancellation failures from any family member.
    pub fn execute_retained(&self, ctx: &ExecutionContext) -> Result<CompletedBatch, CausalError> {
        // Requote admission even when the seed agrees. Retained state may only cross an
        // execution boundary under its original admission ownership and numerical policies.
        let bound = self.reconfigure_execution(ctx)?;
        let data = bound.bound_data()?;
        let shared = bound.shared_design.clone();
        let nuisances = batch_nuisance_cache(ctx, bound.retention);
        let completed = ctx
            .try_map_indexed(bound.plans.len(), |index, inner| {
                if inner.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                    });
                }
                let (result, used) = nuisances.scope(|| {
                    let (mut ready, scores_owed) =
                        bound.plans[index].materialize_for_estimate(inner)?;
                    let result = ready.estimate_with_shared(data, shared.clone(), inner)?;
                    if let Some(from_shared_design) = scores_owed {
                        ready.retain_estimated_scores(&result, from_shared_design, inner)?;
                    }
                    ready.set_stage_sink(None);
                    Ok::<_, CausalError>((ready, result))
                });
                result.map(|(ready, result)| (ready, result, used))
            })
            .map_err(batch_map_error)?;
        let mut plans = Vec::with_capacity(completed.len());
        let mut results = Vec::with_capacity(completed.len());
        let mut uses = Vec::with_capacity(completed.len());
        for (plan, result, used) in completed {
            plans.push(plan);
            results.push(result);
            uses.push(used);
        }
        attach_prepared_batch_joint_inference(
            &mut results,
            data,
            &bound.queries,
            bound.screen.as_ref(),
            bound.family_contrast,
            bound.family_posterior.map(|options| (options, ctx)),
        );
        attach_shared_design_diagnostics(
            &mut results,
            shared.as_ref(),
            &nuisances,
            &uses,
            &bound.estimator_fingerprint,
        );
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let scores = BatchScores::new(
            ScoreSource::Estimated,
            results
                .iter()
                .map(|result| {
                    result.estimate.as_effect().and_then(|effect| effect.scores.table.clone())
                })
                .collect(),
        );
        Ok(CompletedBatch {
            prepared: Self {
                plans,
                execution_owner: Some(BatchExecutionOwner::capture(ctx)?),
                ..bound
            },
            results: results.into_iter().map(Arc::new).collect(),
            scores,
        })
    }

    /// Re-estimate every frozen plan on `data` and attach joint-family inference.
    ///
    /// # Errors
    ///
    /// Schema mismatch or any per-plan estimate failure.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyResult>, CausalError> {
        let data = subset_estimate_rows(data, self.screen.as_ref())?;
        let shared =
            self.shared_design.as_ref().map(|s| s.rebind(&data).map(Arc::new)).transpose()?;
        let nuisances = batch_nuisance_cache(ctx, self.retention);
        let completed = ctx
            .try_map_indexed(self.plans.len(), |index, inner| {
                if inner.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                    });
                }
                let (result, used) = nuisances
                    .scope(|| self.plans[index].estimate_with_shared(&data, shared.clone(), inner));
                result.map(|result| (result, used))
            })
            .map_err(batch_map_error)?;
        let (mut out, uses): (Vec<_>, Vec<_>) = completed.into_iter().unzip();
        attach_prepared_batch_joint_inference(
            &mut out,
            &data,
            &self.queries,
            self.screen.as_ref(),
            self.family_contrast,
            self.family_posterior.map(|options| (options, ctx)),
        );
        attach_shared_design_diagnostics(
            &mut out,
            shared.as_ref(),
            &nuisances,
            &uses,
            &self.estimator_fingerprint,
        );
        Ok(out)
    }

    /// Canonical fingerprint of the estimator configuration this batch was prepared under
    /// (see [`BatchStudy::estimator_fingerprint`]).
    #[must_use]
    pub fn estimator_fingerprint(&self) -> &str {
        &self.estimator_fingerprint
    }

    /// The score tables frozen at prepare, one slot per plan (empty where a plan prepared
    /// none). A retarget of these reweights the prepare-time rows.
    #[must_use]
    pub fn prepared_scores(&self) -> BatchScores {
        BatchScores::new(
            ScoreSource::Prepared,
            self.plans.iter().map(|plan| plan.score_table().cloned()).collect(),
        )
    }

    /// [`Self::estimate`], plus the score tables that estimate produced.
    ///
    /// A retarget after an estimate must reweight the rows of that estimate, not the
    /// prepare-time rows, so the tables travel with the results: retain the returned
    /// [`BatchScores`] and pass it to [`Self::retarget`]. A plan whose estimate carries no
    /// score table (a trimmed or non-iid AIPW, or a non-AIPW estimator) leaves its slot empty
    /// and a retarget of it refuses with `scores_unavailable_after_estimate`; it never falls
    /// back to the prepare-time rows.
    ///
    /// # Errors
    ///
    /// As [`Self::estimate`].
    pub fn estimate_scored(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<StudyResult>, BatchScores), CausalError> {
        let results = self.estimate(data, ctx)?;
        let scores = BatchScores::new(
            ScoreSource::Estimated,
            results
                .iter()
                .map(|r| r.estimate.as_effect().and_then(|effect| effect.scores.table.clone()))
                .collect(),
        );
        Ok((results, scores))
    }

    /// Retarget a declared family of claims over `scores`.
    ///
    /// Every claim names a plan (query index) and row-aligned target weights; the report
    /// carries each point, the joint score covariance of the retargeted claims, the named
    /// linear contrasts, the failed members, and a closed simultaneous interval. See
    /// [`BatchRetargetRequest`] for the declaration and [`BatchRetargetReport`] for the
    /// formulas.
    ///
    /// # Errors
    ///
    /// A malformed family (empty, duplicate or unknown names, scores that do not match the
    /// plans), scores that do not share one row snapshot, or a cancelled `ctx`
    /// (`cancelled_no_claim`, never a verdict). A member that cannot be retargeted is
    /// reported as failed inside the report, never dropped.
    pub fn retarget(
        &self,
        scores: &BatchScores,
        request: &BatchRetargetRequest,
        ctx: &ExecutionContext,
    ) -> Result<BatchRetargetReport, BatchRetargetError> {
        retarget_family(
            &self.plans,
            &self.queries,
            scores,
            request,
            &self.estimator_fingerprint,
            ctx,
        )
    }
}

mod family_inference;

use family_inference::{
    attach_batch_joint_inference, attach_prepared_batch_joint_inference,
    attach_shared_design_diagnostics,
};
#[cfg(test)]
use family_inference::{embed_centered_influence, two_sided_p};

mod selection;

#[cfg(test)]
use selection::{argmin_finite, screen_row_bitmap};
use selection::{attach_candidate_selection, resolve_max_t_critical, subset_estimate_rows};

fn batch_map_error(
    error: antecedent_core::execution::MapExecutionError<CausalError>,
) -> CausalError {
    match error {
        antecedent_core::execution::MapExecutionError::Work(error) => error,
        antecedent_core::execution::MapExecutionError::Admission(
            antecedent_core::execution::AdmissionError::Cancelled,
        ) => CausalError::Cancelled { stage: crate::analysis::stage::STAGE_ESTIMATE_POINT },
        antecedent_core::execution::MapExecutionError::Admission(error) => {
            CausalError::Resource { message: error.to_string() }
        }
    }
}

#[cfg(test)]
mod fail_open_regression_tests;

#[cfg(test)]
mod max_t_diagnostic_tests;

#[cfg(test)]
mod selection_review_tests;

#[cfg(test)]
mod shared_fold_tests;

#[cfg(test)]
mod shared_preparation_legality_tests;

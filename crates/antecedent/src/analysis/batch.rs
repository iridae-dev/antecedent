//! Batch multi-query: one table, N average-effect estimates.
//!
//! [`BatchStudy::prepare`] and [`BatchStudy::prepare_cells`] freeze a
//! [`SharedBatchDesign`]: one fold-assignment object and, when every query
//! shares a certified adjustment set, one compiled `[1 | Z…]` covariate
//! design. Propensity and outcome residualization are still fit per query on
//! that shared design. See [`SharedBatchDesign`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, ResponseFunctional,
    ResponseQuery, VariableId,
};

/// Query frozen on a [`PreparedBatch`] handle.
#[derive(Clone, Debug)]
pub enum BatchQuery {
    /// Average-effect claim.
    Average(AverageEffectQuery),
    /// Discrete joint / cell `InterventionResponse`.
    Response(ResponseQuery),
}

/// Contrast used for family p-values / FDR on a joint-cell batch.
///
/// `estimate.ate` on `cell.aipw` is the requested interventional **level**, not
/// a contrast. Family tests use a declared score-difference contrast instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CellFamilyContrast {
    /// Requested cell minus the all-zero control (same contrast as cell.aipw refuters).
    CellMinusControl,
    /// 2×2 non-additivity `+μ00 −μ10 −μ01 +μ11`.
    Interaction,
}

impl CellFamilyContrast {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CellMinusControl => "cell_minus_control",
            Self::Interaction => "interaction",
        }
    }
}

impl std::str::FromStr for CellFamilyContrast {
    type Err = CausalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cell_minus_control" => Ok(Self::CellMinusControl),
            "interaction" => Ok(Self::Interaction),
            other => Err(CausalError::Compile {
                message: format!(
                    "unknown family_contrast={other:?}; use cell_minus_control|interaction"
                ),
            }),
        }
    }
}
use antecedent_data::{TableView, TabularData, ValidityBitmap};
use antecedent_estimate::{DEFAULT_AIPW_FOLDS, PreparedPropensityProblem};
use antecedent_graph::{Dag, TieredBackground};

use crate::error::CausalError;
use crate::result::StudyResult;

use super::builder::{RefuteSuite, StudyBuilder};
use super::execute::Study;
use super::latency::LatencyMode;
use super::prepared::PreparedStudy;
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
    pub screen_rows: std::sync::Arc<[u32]>,
    /// Row indexes used to estimate the surfaced family.
    pub estimate_rows: std::sync::Arc<[u32]>,
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
    pub screen_rows: std::sync::Arc<[u32]>,
    /// Estimate half.
    pub estimate_rows: std::sync::Arc<[u32]>,
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
    pub row_index: Arc<[u32]>,
}

/// Shared fold assignment and optional shared covariate design for a batch.
///
/// What is shared
///
/// - **Folds:** one original-row assignment (`row % n_folds`). Every query
///   restricts this object to its complete-case rows; it does not draw a
///   private `i % 5` assignment.
/// - **Covariates:** when every query's certified adjustment set is the same,
///   `[1 | Z…]` is compiled once and gathered into each propensity/cell design.
///
/// What is not shared
///
/// - **Propensity:** still fit per query (and per treatment / cell coding) on
///   the shared folds and design.
/// - **Outcome residualization:** still fit per outcome / threshold / cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedBatchDesign {
    /// Fold id for every original row of the prepared table.
    pub fold_ids: Arc<[u32]>,
    /// Fold count used to build [`Self::fold_ids`].
    pub n_folds: u32,
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
    ) -> Result<Self, CausalError> {
        if n_folds < 2 {
            return Err(CausalError::Compile {
                message: "shared batch design requires at least two folds".into(),
            });
        }
        let n = data.row_count();
        let folds = usize::try_from(n_folds).unwrap_or(usize::MAX);
        let fold_ids: Arc<[u32]> =
            (0..n).map(|i| u32::try_from(i % folds).unwrap_or(u32::MAX)).collect::<Vec<_>>().into();
        let covariate = match common_adjustment(adjustment_sets) {
            Some(set) if !set.is_empty() => Some(compile_shared_covariates(data, set)?),
            Some(_) | None => None,
        };
        Ok(Self { fold_ids, n_folds, covariate })
    }

    /// Rebuild data-dependent covariates and row folds on a new estimate table.
    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let sets: Vec<_> = self
            .covariate
            .as_ref()
            .map(|c| vec![Arc::clone(&c.adjustment_set)])
            .unwrap_or_default();
        Self::compile(data, &sets, self.n_folds)
    }

    /// Restrict [`Self::fold_ids`] to complete-case rows.
    ///
    /// # Errors
    ///
    /// A row index outside the prepared table.
    pub fn folds_for(&self, row_index: &[u32]) -> Result<Vec<u32>, CausalError> {
        let mut out = Vec::with_capacity(row_index.len());
        for &row in row_index {
            let id =
                self.fold_ids.get(row as usize).copied().ok_or_else(|| CausalError::Compile {
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
        row_index: &[u32],
    ) -> Result<Option<Vec<f64>>, CausalError> {
        let Some(cov) = self.covariate.as_ref() else {
            return Ok(None);
        };
        if cov.adjustment_set.as_ref() != adjustment {
            return Ok(None);
        }
        let mut slot = vec![u32::MAX; self.fold_ids.len()];
        for (i, &row) in cov.row_index.iter().enumerate() {
            if let Some(entry) = slot.get_mut(row as usize) {
                *entry = u32::try_from(i).unwrap_or(u32::MAX);
            }
        }
        let n = row_index.len();
        let ncols = cov.design_ncols;
        let mut out = vec![0.0; n * ncols];
        for (i, &row) in row_index.iter().enumerate() {
            let src = slot.get(row as usize).copied().unwrap_or(u32::MAX);
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

    /// Apply shared folds and, when the set matches, the shared design matrix.
    ///
    /// # Errors
    ///
    /// Row or shape mismatch.
    pub fn apply_to_propensity(
        &self,
        problem: &mut PreparedPropensityProblem,
    ) -> Result<bool, CausalError> {
        problem.fold_assignment = Some(self.folds_for(&problem.row_index)?.into());
        if let Some(design) = self.design_for(&problem.adjustment_set, &problem.row_index)? {
            if design.len() != problem.design_matrix.len() {
                return Err(CausalError::Compile {
                    message: "shared covariate design shape does not match the propensity problem"
                        .into(),
                });
            }
            problem.design_matrix = design.into();
            return Ok(true);
        }
        Ok(false)
    }
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
            row_index.push(u32::try_from(i).unwrap_or(u32::MAX));
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
/// when licensed, the covariate matrix). Estimation still runs per-query plans
/// in parallel chunks.
#[derive(Clone, Debug)]
pub struct BatchStudy {
    data: TabularData,
    graph: BatchGraph,
    bootstrap_replicates: u32,
    refute: RefuteSuite,
    latency_mode: Option<LatencyMode>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    screen: Option<CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
}

impl BatchStudy {
    /// Start a batch over `data` and a static DAG.
    #[must_use]
    pub fn new(data: TabularData, graph: Dag) -> Self {
        Self {
            data,
            graph: BatchGraph::Dag(graph),
            bootstrap_replicates: 50,
            refute: RefuteSuite::PlaceboAndRcc,
            latency_mode: None,
            identifier: None,
            estimator: None,
            screen: None,
            family_contrast: Some(CellFamilyContrast::CellMinusControl),
        }
    }

    /// Start a batch over a declared [`TieredBackground`].
    ///
    /// CoDetermined materializes as an ADMG; Unknown as a PAG. Identification
    /// uses the same [`crate::StudyBuilder::tiered_background`] path as a single-query study.
    #[must_use]
    pub fn tiered(data: TabularData, background: TieredBackground) -> Self {
        Self {
            data,
            graph: BatchGraph::Tiered(background),
            bootstrap_replicates: 50,
            refute: RefuteSuite::PlaceboAndRcc,
            latency_mode: None,
            identifier: None,
            estimator: None,
            screen: None,
            family_contrast: Some(CellFamilyContrast::CellMinusControl),
        }
    }

    /// Record a screen/estimate split for candidate-selection provenance.
    #[must_use]
    pub fn candidate_screen(mut self, screen: CandidateScreen) -> Self {
        self.screen = Some(screen);
        self
    }

    /// Bootstrap replicates for every query.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = n;
        self
    }

    /// Refute suite for every query.
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = suite;
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
    /// Parse a wire name with `"propensity.weighting".parse::<EstimatorId>()?`.
    #[must_use]
    pub const fn estimator(mut self, id: EstimatorId) -> Self {
        self.estimator = Some(id);
        self
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

    /// Estimate each query against the shared table.
    ///
    /// Shared fold assignment and covariate design when the batch uses AIPW /
    /// score tables. Queries run in parallel under `ctx.parallelism.max_threads`.
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
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let shared = self.compile_shared_ate(&data, queries)?;
        let threads = ctx.parallelism.max_threads.get().max(1) as usize;
        let mut out = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(threads) {
            std::thread::scope(|scope| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|q| scope.spawn(|| self.run_one_on(&data, q, ctx, shared.as_ref())))
                    .collect();
                for handle in handles {
                    match handle.join() {
                        Ok(Ok(result)) => out.push(result),
                        Ok(Err(err)) => return Err(err),
                        Err(_) => {
                            return Err(CausalError::Compile {
                                message: "batch worker panicked".into(),
                            });
                        }
                    }
                }
                Ok(())
            })?;
        }
        attach_batch_joint_inference(&mut out, &data, queries, self.screen.as_ref());
        attach_shared_design_diagnostics(&mut out, shared.as_ref());
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
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch prepare requires at least one query".into(),
            });
        }
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let shared = self.compile_shared_ate(&data, queries)?;
        let mut plans = Vec::with_capacity(queries.len());
        for query in queries {
            let mut study = self.study_for_data(&data, query)?;
            study.shared_batch_design.clone_from(&shared);
            plans.push(study.prepare(ctx)?);
        }
        Ok(PreparedBatch {
            plans,
            queries: queries.iter().cloned().map(BatchQuery::Average).collect(),
            screen: self.screen.clone(),
            shared_design: shared,
            family_contrast: None,
        })
    }

    /// Compile discrete joint `InterventionResponse` cells into a reusable prepared batch.
    ///
    /// Defaults to [`EstimatorId::CellAipw`] when the batch has no estimator override.
    /// Licensed on a DAG and on CoDetermined via joint ADMG adjustment on the
    /// known closure graph. Unknown has no single ADMG. Bare ADMG
    /// InterventionResponse already refuses.
    ///
    /// Family p-values use [`Self::family_contrast`] (default
    /// [`CellFamilyContrast::CellMinusControl`]), never the cell level stored
    /// in `estimate.ate`. Max-t and `family_contrast_interval` are formed on
    /// that contrast family. `family_contrast(None)` keeps intervals on levels
    /// and publishes no p-values.
    ///
    /// # Errors
    ///
    /// Empty query list, Unknown-tier joint (no single ADMG), a non-Set
    /// intervention bundle, or any per-query prepare failure.
    pub fn prepare_cells(
        &self,
        queries: &[ResponseQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch prepare_cells requires at least one query".into(),
            });
        }
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let shared = self.compile_shared_cells(&data, queries)?;
        let mut plans = Vec::with_capacity(queries.len());
        let mut frozen = Vec::with_capacity(queries.len());
        for query in queries {
            let mut study = self.study_for_response_on(&data, query)?;
            study.shared_batch_design.clone_from(&shared);
            plans.push(study.prepare(ctx)?);
            frozen.push(BatchQuery::Response(query.clone()));
        }
        Ok(PreparedBatch {
            plans,
            queries: frozen,
            screen: self.screen.clone(),
            shared_design: shared,
            family_contrast: self.family_contrast,
        })
    }

    fn study_for_response_on(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
    ) -> Result<Study, CausalError> {
        let mut builder = self
            .bind_graph(Study::tabular(data.clone()))?
            .query(query.clone())
            .refute(self.refute)
            .bootstrap_replicates(self.bootstrap_replicates);
        if let Some(mode) = self.latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(id) = self.identifier {
            builder = builder.identifier(id);
        }
        builder = builder.estimator(self.estimator.unwrap_or(EstimatorId::CellAipw));
        builder.build()
    }

    fn compile_shared_ate(
        &self,
        data: &TabularData,
        queries: &[AverageEffectQuery],
    ) -> Result<Option<Arc<SharedBatchDesign>>, CausalError> {
        let mut sets = Vec::with_capacity(queries.len());
        for query in queries {
            sets.push(self.ate_adjustment_set(query)?);
        }
        Ok(Some(Arc::new(SharedBatchDesign::compile(
            data,
            &sets,
            u32::try_from(DEFAULT_AIPW_FOLDS).unwrap_or(5),
        )?)))
    }

    fn compile_shared_cells(
        &self,
        data: &TabularData,
        queries: &[ResponseQuery],
    ) -> Result<Option<Arc<SharedBatchDesign>>, CausalError> {
        let mut sets = Vec::with_capacity(queries.len());
        for query in queries {
            sets.push(self.cell_adjustment_set(query)?);
        }
        Ok(Some(Arc::new(SharedBatchDesign::compile(
            data,
            &sets,
            u32::try_from(DEFAULT_AIPW_FOLDS).unwrap_or(5),
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
        let mut builder = self
            .bind_graph(Study::tabular(data.clone()))?
            .query(query.clone())
            .refute(self.refute)
            .bootstrap_replicates(self.bootstrap_replicates);
        if let Some(mode) = self.latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(id) = self.identifier {
            builder = builder.identifier(id);
        }
        if let Some(est) = self.estimator {
            builder = builder.estimator(est);
        }
        builder.build()
    }

    fn bind_graph(&self, builder: StudyBuilder) -> Result<StudyBuilder, CausalError> {
        match &self.graph {
            BatchGraph::Dag(graph) => Ok(builder.graph(graph.clone())),
            BatchGraph::Tiered(background) => builder.tiered_background(background.clone()),
        }
    }

    fn ate_adjustment_set(
        &self,
        query: &AverageEffectQuery,
    ) -> Result<Arc<[VariableId]>, CausalError> {
        match &self.graph {
            BatchGraph::Dag(graph) => {
                ate_adjustment_set(graph, query, self.identifier, self.estimator)
            }
            BatchGraph::Tiered(background) => {
                if background.within_tier == antecedent_graph::WithinTier::Unknown {
                    return Ok(Arc::from([]));
                }
                let identification = antecedent_identify::identify_tiered(background, query)?;
                identification.estimands.first().map(|e| Arc::clone(&e.adjustment_set)).ok_or_else(
                    || CausalError::Compile {
                        message: "tiered batch identifier returned no estimand".into(),
                    },
                )
            }
        }
    }

    fn cell_adjustment_set(&self, query: &ResponseQuery) -> Result<Arc<[VariableId]>, CausalError> {
        match &self.graph {
            BatchGraph::Dag(graph) => cell_adjustment_set(graph, query, self.identifier),
            BatchGraph::Tiered(background) => {
                let identification = antecedent_identify::identify_tiered_joint(
                    background,
                    self.data.schema(),
                    query,
                )?;
                identification.estimands.first().map(|e| Arc::clone(&e.adjustment_set)).ok_or(
                    CausalError::Unsupported {
                        message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                    },
                )
            }
        }
    }
}

/// Frozen batch of prepared average-effect (and joint-cell) plans.
#[derive(Clone, Debug)]
pub struct PreparedBatch {
    plans: Vec<PreparedStudy>,
    queries: Vec<BatchQuery>,
    screen: Option<CandidateScreen>,
    shared_design: Option<Arc<SharedBatchDesign>>,
    family_contrast: Option<CellFamilyContrast>,
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
        let mut out = Vec::with_capacity(self.plans.len());
        let threads = ctx.parallelism.max_threads.get().max(1) as usize;
        for chunk in self.plans.chunks(threads) {
            std::thread::scope(|scope| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|plan| {
                        scope.spawn(|| plan.estimate_with_shared(&data, shared.clone(), ctx))
                    })
                    .collect();
                for handle in handles {
                    out.push(handle.join().map_err(|_| CausalError::Compile {
                        message: "prepared batch worker panicked".into(),
                    })??);
                }
                Ok::<(), CausalError>(())
            })?;
        }
        attach_prepared_batch_joint_inference(
            &mut out,
            &data,
            &self.queries,
            self.screen.as_ref(),
            self.family_contrast,
        );
        attach_shared_design_diagnostics(&mut out, shared.as_ref());
        Ok(out)
    }
}

fn ate_adjustment_set(
    graph: &Dag,
    query: &AverageEffectQuery,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
) -> Result<Arc<[VariableId]>, CausalError> {
    let identifier = identifier.unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER_ID);
    let estimator = estimator.unwrap_or(EstimatorId::Aipw);
    let identification = crate::strategy_table::identify_static(identifier, graph, query)?;
    let estimand = crate::strategy_table::select_estimand(&identification, estimator)?;
    Ok(estimand.adjustment_set)
}

fn cell_adjustment_set(
    graph: &Dag,
    query: &ResponseQuery,
    identifier: Option<IdentifierId>,
) -> Result<Arc<[VariableId]>, CausalError> {
    let identifier = identifier.unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID);
    let identification = crate::strategy_table::identify_static_query(
        identifier,
        graph,
        &CausalQuery::Response(query.clone()),
    )?;
    identification.estimands.first().map(|e| Arc::clone(&e.adjustment_set)).ok_or_else(|| {
        CausalError::Compile {
            message: "batch prepare_cells identifier returned no estimand".into(),
        }
    })
}

fn attach_shared_design_diagnostics(
    results: &mut [StudyResult],
    shared: Option<&Arc<SharedBatchDesign>>,
) {
    let Some(shared) = shared else {
        return;
    };
    let shares_covariates = shared.covariate.is_some();
    for result in results.iter_mut() {
        if result.diagnostics.iter().any(|d| d.code.as_ref() == "batch.shared_design") {
            continue;
        }
        let mut d = antecedent_core::Diagnostic::new(
            "batch.shared_design",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            format!(
                "shared fold assignment (n_folds={}); covariates={}; propensity and outcome residualization remain per-query fits on that design",
                shared.n_folds,
                if shares_covariates { "shared" } else { "per-query" }
            ),
        );
        d.fields = Arc::from([
            (Arc::from("n_folds"), Arc::from(shared.n_folds.to_string())),
            (
                Arc::from("shares_covariates"),
                Arc::from(if shares_covariates { "true" } else { "false" }),
            ),
            (Arc::from("shares_propensity"), Arc::from("false")),
            (Arc::from("shares_outcome_residualization"), Arc::from("false")),
        ]);
        result.diagnostics.push(d);
    }
}

fn attach_prepared_batch_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
    screen: Option<&CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
) {
    attach_batch_family_joint_inference(results, data, queries, screen, family_contrast);
}

fn attach_batch_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[AverageEffectQuery],
    screen: Option<&CandidateScreen>,
) {
    let family: Vec<BatchQuery> = queries.iter().cloned().map(BatchQuery::Average).collect();
    attach_batch_family_joint_inference(results, data, &family, screen, None);
}

fn batch_claim_alignment(
    query: &BatchQuery,
) -> (antecedent_core::TargetPopulation, Vec<VariableId>) {
    match query {
        BatchQuery::Average(q) => {
            let mut ids = vec![q.treatment, q.outcome];
            ids.extend(q.effect_modifiers.iter().copied());
            (q.target_population.clone(), ids)
        }
        BatchQuery::Response(q) => {
            let mut ids = q.functional.treatment_ids();
            ids.extend(q.functional.outcome_ids());
            (q.target_population.clone(), ids)
        }
    }
}

fn attach_batch_family_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
    screen: Option<&CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
) {
    if results.len() < 2 {
        attach_candidate_selection(results, screen, &[], &[], None);
        return;
    }
    let mut level_aligned = Vec::new();
    let mut row_sets = Vec::new();
    let full_n = data.row_count();
    for (result, query) in results.iter().zip(queries) {
        match align_influence_to_rows(data, result, query, result.estimate.influence.as_deref()) {
            Ok((col, rows)) => {
                level_aligned.push(col);
                row_sets.push(rows);
            }
            Err(reason) => {
                attach_batch_inference_unavailable(results, screen, reason);
                return;
            }
        }
    }
    let cols: Vec<&[f64]> = level_aligned.iter().map(Vec::as_slice).collect();
    let Ok(level_cov) = antecedent_estimate::joint_influence_covariance(&cols, None) else {
        attach_batch_inference_unavailable(
            results,
            screen,
            "batch joint IF covariance could not be formed",
        );
        return;
    };
    let crit = antecedent_estimate::max_t_critical(&level_cov, 0.95, 4096, 1).ok();
    for (i, result) in results.iter_mut().enumerate() {
        result.estimate.joint_covariance = Some(level_cov.clone());
        if let Some(c) = crit {
            result.estimate.simultaneous_interval = Some((
                result.estimate.ate - c * level_cov.se(i),
                result.estimate.ate + c * level_cov.se(i),
                0.95,
            ));
        }
    }

    let cell_family = queries.iter().any(|q| matches!(q, BatchQuery::Response(_)));
    if cell_family {
        match family_contrast {
            None => {
                for result in results.iter_mut() {
                    result.estimate.adjusted_p_values = None;
                    result.estimate.family_contrast = None;
                    result.estimate.family_contrast_interval = None;
                    result.diagnostics.push(antecedent_core::Diagnostic::new(
                        "batch.joint_if",
                        antecedent_core::DiagnosticKind::Scientific,
                        antecedent_core::DiagnosticSeverity::Info,
                        "shared-row joint IF on cell levels; no family contrast declared, p-values omitted",
                    ));
                }
                attach_candidate_selection(results, screen, &[], &[], None);
                return;
            }
            Some(kind) => {
                let mut contrast_aligned = Vec::new();
                let mut contrast_values = Vec::new();
                for ((result, query), rows) in results.iter().zip(queries).zip(&row_sets) {
                    let Some((value, scores)) = cell_family_contrast_claim(result, query, kind)
                    else {
                        for result in results.iter_mut() {
                            result.estimate.adjusted_p_values = None;
                            result.estimate.family_contrast = None;
                            result.estimate.family_contrast_interval = None;
                            result.diagnostics.push(antecedent_core::Diagnostic::new(
                                "batch.joint_if",
                                antecedent_core::DiagnosticKind::Scientific,
                                antecedent_core::DiagnosticSeverity::Info,
                                format!(
                                    "shared-row joint IF on cell levels; {} contrast could not be formed, p-values omitted",
                                    kind.as_str()
                                ),
                            ));
                        }
                        attach_candidate_selection(results, screen, &[], &[], None);
                        return;
                    };
                    if scores.len() != rows.len() {
                        attach_batch_inference_unavailable(
                            results,
                            screen,
                            "batch family contrast influence length did not match the complete-case row universe",
                        );
                        return;
                    }
                    contrast_aligned.push(embed_centered_influence(&scores, rows, full_n));
                    contrast_values.push(value);
                }
                let contrast_cols: Vec<&[f64]> =
                    contrast_aligned.iter().map(Vec::as_slice).collect();
                let Ok(contrast_cov) =
                    antecedent_estimate::joint_influence_covariance(&contrast_cols, None)
                else {
                    for result in results.iter_mut() {
                        result.estimate.adjusted_p_values = None;
                        result.estimate.family_contrast = None;
                        result.estimate.family_contrast_interval = None;
                        result.diagnostics.push(antecedent_core::Diagnostic::new(
                            "batch.joint_if",
                            antecedent_core::DiagnosticKind::Scientific,
                            antecedent_core::DiagnosticSeverity::Info,
                            "shared-row joint IF on cell levels; family contrast covariance could not be formed, p-values omitted",
                        ));
                    }
                    attach_candidate_selection(results, screen, &[], &[], None);
                    return;
                };
                let contrast_crit =
                    antecedent_estimate::max_t_critical(&contrast_cov, 0.95, 4096, 1).ok();
                attach_family_p_values(
                    results,
                    &contrast_values,
                    &contrast_cov,
                    contrast_crit,
                    Some(kind),
                    screen,
                );
                return;
            }
        }
    }

    let values: Vec<f64> = results.iter().map(|r| r.estimate.ate).collect();
    attach_family_p_values(results, &values, &level_cov, crit, None, screen);
}

fn align_influence_to_rows(
    data: &TabularData,
    result: &StudyResult,
    query: &BatchQuery,
    inf: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<usize>), &'static str> {
    let (target_population, mut ids) = batch_claim_alignment(query);
    if !matches!(target_population, antecedent_core::TargetPopulation::AllObserved) {
        return Err("batch joint IF requires AllObserved");
    }
    let Some(inf) = inf else {
        return Err("batch joint IF requires per-claim influence");
    };
    ids.extend(result.estimand.adjustment_set.iter().copied());
    ids.extend(result.estimand.instruments.iter().copied());
    ids.extend(result.estimand.mediators.iter().copied());
    if let Some(table) = result.estimate.score_table.as_ref() {
        ids.push(table.treatment);
        ids.extend(table.intervened.iter().copied());
    }
    let Ok(mask) = data.complete_case_mask(&ids) else {
        return Err("batch joint IF complete-case masks did not align");
    };
    let rows: Vec<_> = mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
    if rows.len() != inf.len() || rows.len() < 2 {
        return Err("batch joint IF influence length did not match the complete-case row universe");
    }
    Ok((embed_centered_influence(inf, &rows, data.row_count()), rows))
}

fn embed_centered_influence(inf: &[f64], rows: &[usize], full_n: usize) -> Vec<f64> {
    let mean = inf.iter().sum::<f64>() / inf.len() as f64;
    let mut col = vec![0.0; full_n];
    for (&r, &v) in rows.iter().zip(inf) {
        col[r] = (v - mean) * full_n as f64 / inf.len() as f64;
    }
    col
}

fn response_requested_arm(query: &ResponseQuery) -> Option<u32> {
    let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional else {
        return None;
    };
    let mut arm = 0u32;
    for (j, iv) in interventions.iter().enumerate() {
        let Intervention::Set { value, .. } = iv else {
            return None;
        };
        let level = value.as_f64()?;
        if level != 0.0 && level != 1.0 {
            return None;
        }
        if j >= antecedent_estimate::cell_aipw::MAX_JOINT_BINARY {
            return None;
        }
        arm |= u32::from(level == 1.0) << j;
    }
    Some(arm)
}

fn cell_family_contrast_claim(
    result: &StudyResult,
    query: &BatchQuery,
    kind: CellFamilyContrast,
) -> Option<(f64, Vec<f64>)> {
    let BatchQuery::Response(query) = query else {
        return None;
    };
    if query.outcome_functional.quantile_level().is_some() {
        return None;
    }
    let table = result.estimate.score_table.as_ref()?;
    let mut thresholds: Vec<f64> = table.columns.iter().filter_map(|c| c.threshold).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    if thresholds.len() > 1 {
        return None;
    }
    let arm = response_requested_arm(query)?;
    let (contrast, scores) =
        antecedent_estimate::family_cell_contrast(table, arm, kind.as_str()).ok()?;
    Some((contrast.value, scores))
}

fn two_sided_p(value: f64, se: f64) -> f64 {
    let z = if se > 0.0 {
        value.abs() / se
    } else if value == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    2.0 * antecedent_stats::student_t_sf(z, 1.0e8)
}

fn attach_family_p_values(
    results: &mut [StudyResult],
    values: &[f64],
    cov: &antecedent_estimate::JointCovariance,
    crit: Option<f64>,
    contrast: Option<CellFamilyContrast>,
    screen: Option<&CandidateScreen>,
) {
    let mut p_values = Vec::with_capacity(results.len());
    let mut rank_stats = Vec::with_capacity(results.len());
    for (i, value) in values.iter().copied().enumerate() {
        let se = cov.se(i);
        p_values.push(two_sided_p(value, se));
        rank_stats.push((value, se));
    }
    let bh = antecedent_stats::benjamini_hochberg(&p_values);
    let by = antecedent_stats::benjamini_yekutieli(&p_values);
    for (i, result) in results.iter_mut().enumerate() {
        result.estimate.adjusted_p_values = Some((bh[i], by[i]));
        if contrast.is_some() {
            let se = cov.se(i);
            result.estimate.family_contrast = Some((values[i], se));
            result.estimate.family_contrast_interval =
                crit.map(|c| (values[i] - c * se, values[i] + c * se, 0.95));
        }
        let mut d = antecedent_core::Diagnostic::new(
            "batch.joint_if",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            match contrast {
                Some(kind) => format!(
                    "shared-row joint IF; family contrast {}={:.4} (se={:.4}); BH={:.4} BY={:.4}{}",
                    kind.as_str(),
                    values[i],
                    cov.se(i),
                    bh.get(i).copied().unwrap_or(f64::NAN),
                    by.get(i).copied().unwrap_or(f64::NAN),
                    crit.map(|c| format!(" max-t_0.95={c:.3}")).unwrap_or_default()
                ),
                None => format!(
                    "shared-row joint IF; BH={:.4} BY={:.4}{}",
                    bh.get(i).copied().unwrap_or(f64::NAN),
                    by.get(i).copied().unwrap_or(f64::NAN),
                    crit.map(|c| format!(" max-t_0.95={c:.3}")).unwrap_or_default()
                ),
            },
        );
        let mut fields = vec![
            (
                std::sync::Arc::from("bh_q"),
                std::sync::Arc::from(format!("{}", bh.get(i).copied().unwrap_or(f64::NAN))),
            ),
            (
                std::sync::Arc::from("by_q"),
                std::sync::Arc::from(format!("{}", by.get(i).copied().unwrap_or(f64::NAN))),
            ),
        ];
        if let Some(kind) = contrast {
            fields.push((
                std::sync::Arc::from("family_contrast"),
                std::sync::Arc::from(kind.as_str()),
            ));
            fields.push((
                std::sync::Arc::from("contrast_value"),
                std::sync::Arc::from(format!("{}", values[i])),
            ));
        }
        d.fields = std::sync::Arc::from(fields);
        result.diagnostics.push(d);
    }
    attach_candidate_selection(results, screen, &bh, &by, Some(&rank_stats));
}

fn subset_estimate_rows(
    data: &TabularData,
    screen: Option<&CandidateScreen>,
) -> Result<TabularData, CausalError> {
    let Some(screen) = screen else {
        return Ok(data.clone());
    };
    if screen.screen_id.is_empty()
        || screen.estimate_rows.is_empty()
        || screen.screen_rows.is_empty()
    {
        return Err(CausalError::Compile {
            message:
                "recorded candidate screens require an id and nonempty screen/estimate row sets"
                    .into(),
        });
    }
    let n = data.row_count();
    for rows in [&screen.screen_rows, &screen.estimate_rows] {
        let mut seen = std::collections::HashSet::new();
        for &r in rows.iter() {
            if r as usize >= n || !seen.insert(r) {
                return Err(CausalError::Compile {
                    message: "candidate-selection row indexes must be unique and within the table"
                        .into(),
                });
            }
        }
    }
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for &r in screen.estimate_rows.iter() {
        let i = r as usize;
        if i >= n {
            return Err(CausalError::Compile {
                message: "candidate-selection estimate_rows exceed the table".into(),
            });
        }
        bytes[i / 8] |= 1 << (i % 8);
    }
    let mask = ValidityBitmap::from_bytes(bytes, n).map_err(CausalError::from)?;
    data.with_analysis_mask(mask).map_err(CausalError::from)
}

fn attach_batch_inference_unavailable(
    results: &mut [StudyResult],
    screen: Option<&CandidateScreen>,
    reason: &str,
) {
    for result in results.iter_mut() {
        result.diagnostics.push(antecedent_core::Diagnostic::new(
            "batch.joint_if.unavailable",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Warning,
            reason,
        ));
    }
    attach_candidate_selection(results, screen, &[], &[], None);
}

fn rows_disjoint(a: &[u32], b: &[u32]) -> bool {
    let mut left: Vec<u32> = a.to_vec();
    let mut right: Vec<u32> = b.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left.dedup();
    right.dedup();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => return false,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    true
}

fn attach_candidate_selection(
    results: &mut [StudyResult],
    screen: Option<&CandidateScreen>,
    bh: &[f64],
    by: &[f64],
    rank_stats: Option<&[(f64, f64)]>,
) {
    let family_size = results.len();
    let (selection, recorded) = match screen {
        None => (
            CandidateSelection {
                screen_id: std::sync::Arc::from("unrecorded"),
                procedure: CandidateProcedure::Unrecorded,
                winner_index: None,
                family_size,
                screen_rows: std::sync::Arc::from([]),
                estimate_rows: std::sync::Arc::from([]),
                disjoint: false,
            },
            false,
        ),
        Some(screen) => {
            let disjoint = rows_disjoint(&screen.screen_rows, &screen.estimate_rows);
            let winner_index = match screen.procedure {
                CandidateProcedure::BenjaminiHochberg => bh
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i),
                CandidateProcedure::BenjaminiYekutieli => by
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i),
                CandidateProcedure::Unrecorded => None,
                CandidateProcedure::MaxT => rank_stats
                    .map(|stats| {
                        stats
                            .iter()
                            .enumerate()
                            .filter(|(_, (value, se))| {
                                value.is_finite() && se.is_finite() && *se >= 0.0
                            })
                            .max_by(|a, b| {
                                let sa = a.1.1.abs().max(1e-12);
                                let sb = b.1.1.abs().max(1e-12);
                                (a.1.0.abs() / sa)
                                    .partial_cmp(&(b.1.0.abs() / sb))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                    })
                    .unwrap_or_else(|| {
                        results
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| {
                                r.estimate.ate.is_finite()
                                    && r.estimate.se_analytic.is_finite()
                                    && r.estimate.se_analytic >= 0.0
                            })
                            .max_by(|a, b| {
                                let sa = a.1.estimate.se_analytic.abs().max(1e-12);
                                let sb = b.1.estimate.se_analytic.abs().max(1e-12);
                                (a.1.estimate.ate.abs() / sa)
                                    .partial_cmp(&(b.1.estimate.ate.abs() / sb))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                    }),
            };
            (
                CandidateSelection {
                    screen_id: std::sync::Arc::clone(&screen.screen_id),
                    procedure: screen.procedure,
                    winner_index,
                    family_size,
                    screen_rows: std::sync::Arc::clone(&screen.screen_rows),
                    estimate_rows: std::sync::Arc::clone(&screen.estimate_rows),
                    disjoint,
                },
                true,
            )
        }
    };
    for result in results.iter_mut() {
        result.candidate_selection = Some(selection.clone());
        result.estimate.candidate_selection = Some(antecedent_estimate::CandidateSelectionRecord {
            screen_id: std::sync::Arc::clone(&selection.screen_id),
            procedure: std::sync::Arc::from(selection.procedure.as_str()),
            winner_index: selection.winner_index,
            family_size: selection.family_size,
            screen_rows: std::sync::Arc::clone(&selection.screen_rows),
            estimate_rows: std::sync::Arc::clone(&selection.estimate_rows),
            disjoint: selection.disjoint,
        });
        if recorded {
            let mut d = antecedent_core::Diagnostic::new(
                "batch.candidate_selection",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                format!(
                    "screen={} procedure={} winner={:?} disjoint={}",
                    selection.screen_id,
                    selection.procedure.as_str(),
                    selection.winner_index,
                    selection.disjoint
                ),
            );
            d.fields = std::sync::Arc::from([
                (std::sync::Arc::from("screen_id"), std::sync::Arc::clone(&selection.screen_id)),
                (
                    std::sync::Arc::from("procedure"),
                    std::sync::Arc::from(selection.procedure.as_str()),
                ),
                (
                    std::sync::Arc::from("winner_index"),
                    std::sync::Arc::from(
                        selection
                            .winner_index
                            .map_or_else(|| "unavailable".into(), |i| i.to_string()),
                    ),
                ),
                (
                    std::sync::Arc::from("disjoint"),
                    std::sync::Arc::from(if selection.disjoint { "true" } else { "false" }),
                ),
            ]);
            result.diagnostics.push(d);
            if !selection.disjoint {
                result.diagnostics.push(antecedent_core::Diagnostic::new(
                    "batch.candidate_selection.overlap",
                    antecedent_core::DiagnosticKind::Scientific,
                    antecedent_core::DiagnosticSeverity::Warning,
                    "screen and estimate row sets are not disjoint",
                ));
            }
        } else {
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                "batch.candidate_selection.unrecorded",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "no candidate-selection split has been recorded; simultaneous inference covers only the fixed supplied family and excludes screening uncertainty",
            ));
        }
        if recorded {
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                "batch.candidate_selection.ranked_on_reported_family",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "winner is ranked on the reported estimate-sample family; screen_rows are recorded provenance and were not used to select",
            ));
        }
    }
}

#[cfg(test)]
mod selection_review_tests {
    use super::*;
    #[test]
    fn recorded_splits_reject_empty_duplicate_and_out_of_range_rows() {
        let values = [0.0, 1.0, 2.0];
        let data = TabularData::from_f64_columns([("x", values.as_slice())]).unwrap();
        for (screen_rows, estimate_rows) in [
            (vec![], vec![1]),
            (vec![0], vec![]),
            (vec![0, 0], vec![1]),
            (vec![3], vec![1]),
            (vec![0], vec![1, 1]),
            (vec![0], vec![3]),
        ] {
            let screen = CandidateScreen {
                screen_id: "review".into(),
                procedure: CandidateProcedure::MaxT,
                screen_rows: screen_rows.into(),
                estimate_rows: estimate_rows.into(),
            };
            assert!(subset_estimate_rows(&data, Some(&screen)).is_err());
        }
    }
}

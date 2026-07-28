//! Front-door two-stage (product-of-coefficients) regression estimator .
//!
//! Requires a `"frontdoor"` estimand with a non-empty [`IdentifiedEstimand::mediators`] set
//! (see `antecedent_identify::frontdoor`). Supports one or more mediators `M₁…Mₖ`; the front-door
//! criterion for a valid mediator set guarantees:
//!
//! 1. The mediators intercept every directed path from `T` to `Y`.
//! 2. No unblocked backdoor path from `T` to the mediators.
//! 3. Every backdoor path from the mediators to `Y` is blocked by conditioning on `T`.
//!
//! For a linear SEM the nonparametric front-door formula collapses to a **path-sum** of
//! product-of-coefficients terms:
//!
//! - **Stage 1** (per mediator): OLS of `Mⱼ` on `[1, T]` → `β_{T→Mⱼ}`.
//! - **Stage 2**: OLS of `Y` on `[1, T, M₁…Mₖ]` → `β_{Mⱼ→Y}` (holding `T` and other mediators).
//!
//! `ATE = (Σⱼ β_{T→Mⱼ} · β_{Mⱼ→Y}) · (active − control)`.
//!
//! This assumes no direct `T → Y` edge (all of the treatment effect flows through the mediators,
//! as the front-door criterion requires) and linear structural equations.
//!
//! The analytic standard error is a stacked M-estimator sandwich: every stage-1 mediator
//! regression and the stage-2 outcome regression share one score vector per row so that
//! same-sample cross-stage (and cross-mediator) covariances enter the delta-method variance
//! of the path sum. Bootstrap remains available for resampling checks.
//!
//! Positivity is not meaningful here — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, form_xtx, invert_square,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::se::{AnalyticSeKind, require_clusters};
use crate::util::{BootstrapSeResult, bootstrap_se, stats_err};

/// Stage-1 design column count: `[1, T]`.
const STAGE1_NCOLS: usize = 2;
/// Stage-1 column index of the treatment coefficient (`β_{T→M}`).
const STAGE1_TREATMENT_COL: usize = 1;
/// Stage-2 column index of treatment (after intercept).
const STAGE2_TREATMENT_COL: usize = 1;
/// Stage-2 first mediator column index (`[1, T, M…]`).
const STAGE2_FIRST_MEDIATOR_COL: usize = 2;

/// Prepared front-door problem: treatment, mediator columns, and outcome after complete-case
/// filtering.
#[derive(Clone, Debug)]
pub struct PreparedFrontDoorProblem {
    /// Treatment, length `nrows`.
    pub treatment: Arc<[f64]>,
    /// Mediator columns (each length `nrows`), in estimand order.
    pub mediators: Arc<[Arc<[f64]>]>,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Complete-case row count.
    pub nrows: usize,
    /// Estimand method tag (always `"frontdoor"`).
    pub method: Arc<str>,
    /// Mediator variable ids (aligned with [`Self::mediators`]).
    pub mediator_ids: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active − control treatment contrast used for the ATE scaling.
    pub treatment_delta: f64,
}

fn prepare_frontdoor_problem(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    overlap: OverlapPolicy,
) -> Result<PreparedFrontDoorProblem, EstimationError> {
    crate::util::require_explicit_override(
        overlap,
        "FrontDoorTwoStage requires ExplicitOverride overlap policy (not propensity-based)",
    )?;
    if estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::FrontDoor) {
        return Err(EstimationError::IncompatibleEstimand {
            message: "FrontDoorTwoStage expects a \"frontdoor\" estimand",
        });
    }
    if estimand.mediators.is_empty() {
        return Err(EstimationError::IncompatibleEstimand {
            message: "FrontDoorTwoStage requires a non-empty mediator set",
        });
    }
    query.validate()?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported(
            "FrontDoorTwoStage does not support effect modifiers",
        ));
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::unsupported(
            "FrontDoorTwoStage only supports TargetPopulation::AllObserved",
        ));
    }
    let treatment = query.treatment;
    let outcome = query.outcome;
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let treatment_delta = active - control;
    if treatment_delta == 0.0 {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }

    let mut ids = Vec::with_capacity(2 + estimand.mediators.len());
    ids.push(treatment);
    ids.push(outcome);
    ids.extend_from_slice(&estimand.mediators);
    let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
    let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
    let mut mediators = Vec::with_capacity(estimand.mediators.len());
    for &mid in estimand.mediators.iter() {
        let m = data.float64_masked(mid, &row_mask).map_err(EstimationError::from)?;
        mediators.push(Arc::<[f64]>::from(m));
    }
    let nrows = t.len();

    Ok(PreparedFrontDoorProblem {
        treatment: Arc::from(t),
        mediators: Arc::from(mediators),
        outcome: Arc::from(y),
        nrows,
        method: Arc::clone(&estimand.method),
        mediator_ids: Arc::clone(&estimand.mediators),
        overlap,
        treatment_delta,
    })
}

/// Estimation workspace, reused for both regression stages and across bootstrap replicates.
#[derive(Clone, Debug, Default)]
pub struct FrontDoorWorkspace {
    /// OLS scratch shared by the `M ~ T` and `Y ~ T, M…` stages.
    pub ols: LeastSquaresWorkspace,
}

/// Front-door two-stage (path-sum product-of-coefficients) regression estimator.
///
/// See the module docs for the estimator definition. Supports one or more mediators.
#[derive(Clone, Debug)]
pub struct FrontDoorTwoStage {
    /// Dense linear-algebra backend used by both regression stages.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Analytic SE policy for the stacked path-sum sandwich (default HC0).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids (`length = nrows`) for [`AnalyticSeKind::Cluster`].
    pub cluster_ids: Option<Arc<[u32]>>,
}

impl Default for FrontDoorTwoStage {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontDoorTwoStage {
    /// Default: 200 bootstrap replicates, explicit overlap override, HC0 stacked SE.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Hc0,
            cluster_ids: None,
        }
    }

    /// Set the dense linear-algebra backend used by both regression stages.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic
    /// stacked-sandwich SE.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — front-door
    /// is not a propensity-based method.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the analytic SE policy for the stacked path-sum sandwich (default
    /// [`AnalyticSeKind::Hc0`]). Supports Homoskedastic/Hc0/Hc1/Cluster only.
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids (length = prepared `nrows`) for [`AnalyticSeKind::Cluster`].
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Arc<[u32]>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Prepare the treatment/mediator/outcome design.
    ///
    /// # Errors
    ///
    /// See [`prepare_frontdoor_problem`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedFrontDoorProblem, EstimationError> {
        prepare_frontdoor_problem(data, estimand, query, self.overlap)
    }

    /// Fit the path-sum product-of-coefficients estimator, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// Backend/rank failure in either stage.
    pub fn fit(
        &self,
        problem: &PreparedFrontDoorProblem,
        workspace: &mut FrontDoorWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let (ate, se_analytic) = self.point_estimate(problem, workspace)?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        }
        .with_bootstrap(boot))
    }

    fn point_estimate(
        &self,
        problem: &PreparedFrontDoorProblem,
        workspace: &mut FrontDoorWorkspace,
    ) -> Result<(f64, f64), EstimationError> {
        let k = problem.mediators.len();
        let n = problem.nrows;
        let x1 = stage1_matrix(&problem.treatment);
        let mut stage1_coefs = Vec::with_capacity(k);
        let mut stage1_resid = Vec::with_capacity(k);
        for m in problem.mediators.iter() {
            let stage1 = self.fit_stage1(&problem.treatment, m, workspace)?;
            let mut resid = vec![0.0; n];
            for r in 0..n {
                resid[r] = m[r]
                    - stage1.coefficients[0]
                    - stage1.coefficients[STAGE1_TREATMENT_COL] * problem.treatment[r];
            }
            stage1_coefs.push(stage1.coefficients);
            stage1_resid.push(resid);
        }
        let stage2 =
            self.fit_stage2(&problem.treatment, &problem.mediators, &problem.outcome, workspace)?;
        let x2 = stage2_matrix(&problem.treatment, &problem.mediators);
        let stage2_ncols = 2 + k;
        let mut stage2_resid = vec![0.0; n];
        for r in 0..n {
            let mut pred = 0.0;
            for c in 0..stage2_ncols {
                pred += x2[c * n + r] * stage2.coefficients[c];
            }
            stage2_resid[r] = problem.outcome[r] - pred;
        }
        let mut path_sum = 0.0;
        for (j, coefs) in stage1_coefs.iter().enumerate() {
            path_sum +=
                coefs[STAGE1_TREATMENT_COL] * stage2.coefficients[STAGE2_FIRST_MEDIATOR_COL + j];
        }
        let ate = path_sum * problem.treatment_delta;
        let stages = StackedStageViews {
            x1: &x1,
            stage1_resid: &stage1_resid,
            stage1_coefs: &stage1_coefs,
            x2: &x2,
            stage2_ncols,
            stage2_resid: &stage2_resid,
            stage2_coefs: &stage2.coefficients,
            treatment_delta: problem.treatment_delta,
        };
        let se_analytic = stacked_path_sum_se(
            &stages,
            self.se_kind,
            self.cluster_ids.as_deref(),
            CrossStagePolicy::Empirical,
        )?;
        Ok((ate, se_analytic))
    }

    fn fit_stage1(
        &self,
        treatment: &[f64],
        mediator: &[f64],
        workspace: &mut FrontDoorWorkspace,
    ) -> Result<antecedent_stats::LeastSquaresFit, EstimationError> {
        let n = treatment.len();
        let x = stage1_matrix(treatment);
        self.backend
            .least_squares(&x, n, STAGE1_NCOLS, mediator, &mut workspace.ols)
            .map_err(stats_err)
    }

    fn fit_stage2(
        &self,
        treatment: &[f64],
        mediators: &[Arc<[f64]>],
        outcome: &[f64],
        workspace: &mut FrontDoorWorkspace,
    ) -> Result<antecedent_stats::LeastSquaresFit, EstimationError> {
        let n = treatment.len();
        let ncols = 2 + mediators.len();
        let x = stage2_matrix(treatment, mediators);
        self.backend.least_squares(&x, n, ncols, outcome, &mut workspace.ols).map_err(stats_err)
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedFrontDoorProblem,
        workspace: &mut FrontDoorWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.nrows;
        let k = problem.mediators.len();
        let mut t_boot = vec![0.0; n];
        let mut m_boot: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; n]).collect();
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0xF80D_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                t_boot[r] = problem.treatment[src];
                y_boot[r] = problem.outcome[src];
                for (j, mcol) in problem.mediators.iter().enumerate() {
                    m_boot[j][r] = mcol[src];
                }
            }
            let mediators: Vec<Arc<[f64]>> =
                m_boot.iter().map(|m| Arc::<[f64]>::from(m.as_slice())).collect();
            let Ok(stage2) = self.fit_stage2(&t_boot, &mediators, &y_boot, workspace) else {
                return Ok(None);
            };
            let mut path_sum = 0.0;
            for (j, m_j) in mediators.iter().enumerate() {
                let Ok(s1) = self.fit_stage1(&t_boot, m_j, workspace) else {
                    return Ok(None);
                };
                path_sum += s1.coefficients[STAGE1_TREATMENT_COL]
                    * stage2.coefficients[STAGE2_FIRST_MEDIATOR_COL + j];
            }
            Ok(Some(path_sum * problem.treatment_delta))
        })
    }
}

/// Build the column-major `[1, T]` stage-1 design.
fn stage1_matrix(treatment: &[f64]) -> Vec<f64> {
    let n = treatment.len();
    let mut x = vec![0.0; n * STAGE1_NCOLS];
    x[..n].fill(1.0);
    x[n..2 * n].copy_from_slice(treatment);
    x
}

/// Build the column-major `[1, T, M₁…Mₖ]` stage-2 design.
fn stage2_matrix(treatment: &[f64], mediators: &[Arc<[f64]>]) -> Vec<f64> {
    let n = treatment.len();
    let ncols = 2 + mediators.len();
    let mut x = vec![0.0; n * ncols];
    x[..n].fill(1.0);
    x[n..2 * n].copy_from_slice(treatment);
    for (j, m) in mediators.iter().enumerate() {
        let base = (STAGE2_FIRST_MEDIATOR_COL + j) * n;
        x[base..base + n].copy_from_slice(m);
    }
    let _ = STAGE2_TREATMENT_COL;
    x
}

/// Whether the stacked meat keeps empirical cross-stage blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CrossStagePolicy {
    /// Full stacked HC meat (includes Cov across stages/mediators).
    Empirical,
    /// Zero cross-stage / cross-mediator blocks (independent-path Sobel delta).
    IndependentPaths,
}

/// Shared stage-1 / stage-2 views for stacked front-door sandwich SE.
#[derive(Clone, Copy, Debug)]
struct StackedStageViews<'a> {
    x1: &'a [f64],
    stage1_resid: &'a [Vec<f64>],
    stage1_coefs: &'a [Vec<f64>],
    x2: &'a [f64],
    stage2_ncols: usize,
    stage2_resid: &'a [f64],
    stage2_coefs: &'a [f64],
    treatment_delta: f64,
}

fn stacked_path_sum_se(
    stages: &StackedStageViews<'_>,
    se_kind: AnalyticSeKind,
    cluster_ids: Option<&[u32]>,
    cross_stage: CrossStagePolicy,
) -> Result<f64, EstimationError> {
    let (cov, grad) = stacked_theta_cov_and_grad(stages, se_kind, cluster_ids, cross_stage)?;
    let n_params = grad.len();
    let mut tmp = vec![0.0; n_params];
    for row in 0..n_params {
        let mut sum = 0.0;
        for col in 0..n_params {
            sum += cov[row * n_params + col] * grad[col];
        }
        tmp[row] = sum;
    }
    let mut var = 0.0;
    for (row, &g) in grad.iter().enumerate() {
        var += g * tmp[row];
    }
    if !var.is_finite() {
        return Ok(f64::NAN);
    }
    Ok(var.max(0.0).sqrt())
}

fn stacked_theta_cov_and_grad(
    stages: &StackedStageViews<'_>,
    se_kind: AnalyticSeKind,
    cluster_ids: Option<&[u32]>,
    cross_stage: CrossStagePolicy,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let n_mediators = stages.stage1_resid.len();
    if n_mediators == 0 || stages.stage1_coefs.len() != n_mediators {
        return Err(EstimationError::unsupported("stacked front-door SE needs ≥1 mediator"));
    }
    let n_rows = stages.stage2_resid.len();
    if stages.x1.len() < n_rows * STAGE1_NCOLS || stages.x2.len() < n_rows * stages.stage2_ncols {
        return Err(EstimationError::data_msg("stacked front-door design buffer too short"));
    }
    for resid in stages.stage1_resid {
        if resid.len() != n_rows {
            return Err(EstimationError::data_msg("stage-1 residual length mismatch"));
        }
    }
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;

    // Block-diagonal bread A = diag(X1'X1, …, X1'X1, X2'X2).
    let mut bread = vec![0.0; n_params * n_params];
    let mut xtx1 = vec![0.0; STAGE1_NCOLS * STAGE1_NCOLS];
    form_xtx(stages.x1, n_rows, STAGE1_NCOLS, &mut xtx1);
    for mediator_idx in 0..n_mediators {
        let off = mediator_idx * STAGE1_NCOLS;
        for row in 0..STAGE1_NCOLS {
            for col in 0..STAGE1_NCOLS {
                bread[(off + row) * n_params + (off + col)] = xtx1[row * STAGE1_NCOLS + col];
            }
        }
    }
    let mut xtx2 = vec![0.0; stages.stage2_ncols * stages.stage2_ncols];
    form_xtx(stages.x2, n_rows, stages.stage2_ncols, &mut xtx2);
    let stage2_off = n_mediators * STAGE1_NCOLS;
    for row in 0..stages.stage2_ncols {
        for col in 0..stages.stage2_ncols {
            bread[(stage2_off + row) * n_params + (stage2_off + col)] =
                xtx2[row * stages.stage2_ncols + col];
        }
    }
    let bread_inv = invert_square(&bread, n_params)
        .ok_or_else(|| EstimationError::stats_msg("singular stacked front-door bread"))?;

    let meat = match se_kind {
        AnalyticSeKind::Homoskedastic | AnalyticSeKind::Hc0 | AnalyticSeKind::Hc1 => {
            let mut meat = stacked_hc_meat(stages, cross_stage);
            if matches!(se_kind, AnalyticSeKind::Hc1) {
                if n_rows <= n_params {
                    return Err(EstimationError::stats_msg("non-positive residual df for HC1"));
                }
                let scale = n_rows as f64 / (n_rows as f64 - n_params as f64);
                for v in &mut meat {
                    *v *= scale;
                }
            }
            meat
        }
        AnalyticSeKind::Cluster => {
            let groups = require_clusters(cluster_ids, n_rows)?;
            stacked_cluster_meat(stages, groups, cross_stage)?
        }
        AnalyticSeKind::Hc2
        | AnalyticSeKind::Hc3
        | AnalyticSeKind::Multiway
        | AnalyticSeKind::NeweyWest { .. }
        | AnalyticSeKind::PanelClusterHac { .. } => {
            return Err(EstimationError::unsupported(
                "FrontDoorTwoStage stacked SE supports Homoskedastic/Hc0/Hc1/Cluster only",
            ));
        }
    };

    // Σ = A⁻¹ B A⁻¹
    let mut tmp = vec![0.0; n_params * n_params];
    for row in 0..n_params {
        for col in 0..n_params {
            let mut sum = 0.0;
            for mid in 0..n_params {
                sum += bread_inv[row * n_params + mid] * meat[mid * n_params + col];
            }
            tmp[row * n_params + col] = sum;
        }
    }
    let mut cov = vec![0.0; n_params * n_params];
    for row in 0..n_params {
        for col in 0..n_params {
            let mut sum = 0.0;
            for mid in 0..n_params {
                sum += tmp[row * n_params + mid] * bread_inv[mid * n_params + col];
            }
            cov[row * n_params + col] = sum;
        }
    }

    // ∇g for g = Δ Σⱼ β_{T→Mⱼ} β_{Mⱼ→Y}
    let mut grad = vec![0.0; n_params];
    for mediator_idx in 0..n_mediators {
        let beta_t_to_m = stages.stage1_coefs[mediator_idx][STAGE1_TREATMENT_COL];
        let beta_m_to_y = stages.stage2_coefs[STAGE2_FIRST_MEDIATOR_COL + mediator_idx];
        grad[mediator_idx * STAGE1_NCOLS + STAGE1_TREATMENT_COL] =
            stages.treatment_delta * beta_m_to_y;
        grad[stage2_off + STAGE2_FIRST_MEDIATOR_COL + mediator_idx] =
            stages.treatment_delta * beta_t_to_m;
    }
    Ok((cov, grad))
}

fn fill_row_score(score: &mut [f64], stages: &StackedStageViews<'_>, row: usize, n_rows: usize) {
    let n_mediators = stages.stage1_resid.len();
    score.fill(0.0);
    for (mediator_idx, resid) in stages.stage1_resid.iter().enumerate() {
        let e1 = resid[row];
        let off = mediator_idx * STAGE1_NCOLS;
        for col in 0..STAGE1_NCOLS {
            score[off + col] = e1 * stages.x1[col * n_rows + row];
        }
    }
    let e2 = stages.stage2_resid[row];
    let stage2_off = n_mediators * STAGE1_NCOLS;
    for col in 0..stages.stage2_ncols {
        score[stage2_off + col] = e2 * stages.x2[col * n_rows + row];
    }
}

fn zero_cross_blocks(meat: &mut [f64], n_params: usize, n_mediators: usize, stage2_ncols: usize) {
    // Zero every off-diagonal stage block (independent-path Sobel).
    let blocks: Vec<(usize, usize)> = {
        let mut block_specs = Vec::with_capacity(n_mediators + 1);
        for mediator_idx in 0..n_mediators {
            block_specs.push((mediator_idx * STAGE1_NCOLS, STAGE1_NCOLS));
        }
        block_specs.push((n_mediators * STAGE1_NCOLS, stage2_ncols));
        block_specs
    };
    for (bi, &(o1, n1)) in blocks.iter().enumerate() {
        for (bj, &(o2, n2)) in blocks.iter().enumerate() {
            if bi == bj {
                continue;
            }
            for row in 0..n1 {
                for col in 0..n2 {
                    meat[(o1 + row) * n_params + (o2 + col)] = 0.0;
                }
            }
        }
    }
}

fn stacked_hc_meat(stages: &StackedStageViews<'_>, cross_stage: CrossStagePolicy) -> Vec<f64> {
    let n_mediators = stages.stage1_resid.len();
    let n_rows = stages.stage2_resid.len();
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;
    let mut meat = vec![0.0; n_params * n_params];
    let mut score = vec![0.0; n_params];
    for row in 0..n_rows {
        fill_row_score(&mut score, stages, row, n_rows);
        for i in 0..n_params {
            for j in 0..n_params {
                meat[i * n_params + j] += score[i] * score[j];
            }
        }
    }
    if cross_stage == CrossStagePolicy::IndependentPaths {
        zero_cross_blocks(&mut meat, n_params, n_mediators, stages.stage2_ncols);
    }
    meat
}

fn stacked_cluster_meat(
    stages: &StackedStageViews<'_>,
    groups: &[u32],
    cross_stage: CrossStagePolicy,
) -> Result<Vec<f64>, EstimationError> {
    let n_mediators = stages.stage1_resid.len();
    let n_rows = stages.stage2_resid.len();
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;
    let mut totals: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    let mut score = vec![0.0; n_params];
    for (row, &group_id) in groups.iter().enumerate().take(n_rows) {
        fill_row_score(&mut score, stages, row, n_rows);
        let entry = totals.entry(group_id).or_insert_with(|| vec![0.0; n_params]);
        for (col, &s) in score.iter().enumerate() {
            entry[col] += s;
        }
    }
    let n_clusters = totals.len();
    if n_clusters < 2 {
        return Err(EstimationError::stats_msg(
            "cluster-robust variance requires at least 2 clusters",
        ));
    }
    if n_rows <= n_params {
        return Err(EstimationError::stats_msg("non-positive residual df for cluster SE"));
    }
    let mut meat = vec![0.0; n_params * n_params];
    for score_g in totals.values() {
        for i in 0..n_params {
            for j in 0..n_params {
                meat[i * n_params + j] += score_g[i] * score_g[j];
            }
        }
    }
    if cross_stage == CrossStagePolicy::IndependentPaths {
        zero_cross_blocks(&mut meat, n_params, n_mediators, stages.stage2_ncols);
    }
    let scale = (n_clusters as f64 / (n_clusters as f64 - 1.0))
        * ((n_rows as f64 - 1.0) / (n_rows as f64 - n_params as f64));
    for v in &mut meat {
        *v *= scale;
    }
    Ok(meat)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::ExprId;
    use antecedent_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;
    use antecedent_kernels::standard_normal;

    /// `U -> T -> M -> Y` with `U -> Y` directly (no `T -> Y` edge): `T = U + noise`,
    /// `M = 2T + noise`, `Y = 3M + U + noise`. The `U` confounder makes the `T-Y` backdoor
    /// unblockable directly, but `M` satisfies the front-door criterion (mirrors the SCM in
    /// `antecedent_identify::frontdoor::tests::classic_frontdoor_with_unmeasured_confounder`).
    /// True effect through the mediator path = `2 * 3 = 6`.
    fn frontdoor_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xF400_u64);
        let mut t = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let u = standard_normal(&mut rng);
            let ti = u + 0.1 * standard_normal(&mut rng);
            let mi = 2.0 * ti + 0.1 * standard_normal(&mut rng);
            let yi = 3.0 * mi + u + 0.1 * standard_normal(&mut rng);
            t[i] = ti;
            m[i] = mi;
            y[i] = yi;
        }
        (build_frontdoor_data(n, t, y, m), frontdoor_estimand())
    }

    fn frontdoor_estimand() -> IdentifiedEstimand {
        IdentifiedEstimand::frontdoor(
            "frontdoor",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        )
    }

    fn build_frontdoor_data(n: usize, t: Vec<f64>, y: Vec<f64>, m: Vec<f64>) -> TabularData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "m",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(m),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TabularData::new(storage)
    }

    fn query() -> AverageEffectQuery {
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(41)
    }

    #[test]
    fn frontdoor_two_stage_recovers_effect_six() {
        let (data, estimand) = frontdoor_scm(4000, 1);
        let est = FrontDoorTwoStage { bootstrap_replicates: 30, ..FrontDoorTwoStage::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = FrontDoorWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 6.0).abs() < 0.5, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.se_analytic.is_finite());
    }

    #[test]
    fn frontdoor_two_stage_rejects_explicit_override_violation() {
        let (data, estimand) = frontdoor_scm(200, 2);
        let est = FrontDoorTwoStage {
            overlap: OverlapPolicy::require_diagnostics(),
            ..FrontDoorTwoStage::new()
        };
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn frontdoor_two_stage_rejects_non_frontdoor_estimand() {
        let (data, mut estimand) = frontdoor_scm(200, 3);
        estimand.method = Arc::from("backdoor.adjustment");
        let est = FrontDoorTwoStage::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn frontdoor_two_stage_rejects_empty_mediators() {
        let (data, mut estimand) = frontdoor_scm(200, 4);
        estimand.mediators = Arc::from([]);
        let est = FrontDoorTwoStage::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn frontdoor_two_stage_recovers_two_mediator_path_sum() {
        // T → M1 → Y and T → M2 → Y: M1=1·T, M2=2·T, Y=3·M1+4·M2 (+ noise, no U).
        // Path sum = 1·3 + 2·4 = 11.
        let n = 3000usize;
        let mut rng = ExecutionContext::for_tests(9).rng.stream(0xF401_u64);
        let mut t = vec![0.0; n];
        let mut m1 = vec![0.0; n];
        let mut m2 = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ti = standard_normal(&mut rng);
            let a = ti + 0.05 * standard_normal(&mut rng);
            let b = 2.0 * ti + 0.05 * standard_normal(&mut rng);
            let yi = 3.0 * a + 4.0 * b + 0.05 * standard_normal(&mut rng);
            t[i] = ti;
            m1[i] = a;
            m2[i] = b;
            y[i] = yi;
        }
        let mut b = CausalSchemaBuilder::new();
        for (name, role) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("m1", RoleHint::Context),
            ("m2", RoleHint::Context),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(m1),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(3),
                    Arc::from(m2),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let estimand = IdentifiedEstimand::frontdoor(
            "frontdoor",
            Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            ExprId::from_raw(0),
        );
        let est = FrontDoorTwoStage { bootstrap_replicates: 0, ..FrontDoorTwoStage::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        assert_eq!(prep.mediators.len(), 2);
        let mut ws = FrontDoorWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 11.0).abs() < 0.5, "ate={}", effect.ate);
        assert!(
            effect.se_analytic.is_finite() && effect.se_analytic > 0.0,
            "se={}",
            effect.se_analytic
        );
    }

    #[test]
    fn stacked_se_captures_cross_stage_covariance_and_matches_bootstrap() {
        // Correlated stage residuals → nonzero Cov(a,b); analytic SE ≈ paired bootstrap.
        let n = 2500usize;
        let mut rng = ExecutionContext::for_tests(11).rng.stream(0xF402_u64);
        let mut t = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ti = standard_normal(&mut rng);
            let e1 = standard_normal(&mut rng);
            let e2 = 0.85 * e1 + 0.15 * standard_normal(&mut rng);
            let mi = 1.5 * ti + e1;
            let yi = 2.0 * mi + e2;
            t[i] = ti;
            m[i] = mi;
            y[i] = yi;
        }
        let data = build_frontdoor_data(n, t, y, m);
        let estimand = frontdoor_estimand();
        let est = FrontDoorTwoStage { bootstrap_replicates: 400, ..FrontDoorTwoStage::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = FrontDoorWorkspace::default();

        let x1 = stage1_matrix(&prep.treatment);
        let s1 = est.fit_stage1(&prep.treatment, &prep.mediators[0], &mut ws).unwrap();
        let mut r1 = vec![0.0; n];
        for (row, residual) in r1.iter_mut().enumerate() {
            *residual = prep.mediators[0][row]
                - s1.coefficients[0]
                - s1.coefficients[STAGE1_TREATMENT_COL] * prep.treatment[row];
        }
        let s2 = est.fit_stage2(&prep.treatment, &prep.mediators, &prep.outcome, &mut ws).unwrap();
        let x2 = stage2_matrix(&prep.treatment, &prep.mediators);
        let mut r2 = vec![0.0; n];
        for (row, residual) in r2.iter_mut().enumerate() {
            let mut pred = 0.0;
            for c in 0..(2 + prep.mediators.len()) {
                pred += x2[c * n + row] * s2.coefficients[c];
            }
            *residual = prep.outcome[row] - pred;
        }
        let stage1_resid = [r1];
        let stage1_coefs = [s1.coefficients.clone()];
        let stages = StackedStageViews {
            x1: &x1,
            stage1_resid: &stage1_resid,
            stage1_coefs: &stage1_coefs,
            x2: &x2,
            stage2_ncols: 3,
            stage2_resid: &r2,
            stage2_coefs: &s2.coefficients,
            treatment_delta: prep.treatment_delta,
        };
        let (cov, _) = stacked_theta_cov_and_grad(
            &stages,
            AnalyticSeKind::Hc0,
            None,
            CrossStagePolicy::Empirical,
        )
        .unwrap();
        // Indices: a at stage1 col 1 (index 1), b at stage2 mediator col (index 4).
        let n_theta = 5usize;
        let idx_a = STAGE1_TREATMENT_COL;
        let idx_b = STAGE1_NCOLS + STAGE2_FIRST_MEDIATOR_COL;
        let cov_ab = cov[idx_a * n_theta + idx_b];
        let var_a = cov[idx_a * n_theta + idx_a];
        let var_b = cov[idx_b * n_theta + idx_b];
        assert!(
            cov_ab.is_finite() && cov_ab.abs() > 0.0,
            "expected nonzero cross-stage Cov(a,b), got {cov_ab}"
        );
        assert!(var_a > 0.0 && var_b > 0.0);

        let se_full =
            stacked_path_sum_se(&stages, AnalyticSeKind::Hc0, None, CrossStagePolicy::Empirical)
                .unwrap();
        let se_indep = stacked_path_sum_se(
            &stages,
            AnalyticSeKind::Hc0,
            None,
            CrossStagePolicy::IndependentPaths,
        )
        .unwrap();
        assert!(
            (se_full - se_indep).abs() / se_full.max(1e-8) > 0.001,
            "cross-stage term should move SE: full={se_full} indep={se_indep}"
        );

        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let boot = effect.se_bootstrap.expect("bootstrap SE");
        let rel = (effect.se_analytic - boot).abs() / boot.max(1e-8);
        assert!(rel < 0.25, "analytic={} boot={} rel={}", effect.se_analytic, boot, rel);
    }

    #[test]
    fn two_mediator_stacked_se_matches_bootstrap() {
        let n = 3000usize;
        let mut rng = ExecutionContext::for_tests(12).rng.stream(0xF403_u64);
        let mut t = vec![0.0; n];
        let mut m1 = vec![0.0; n];
        let mut m2 = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ti = standard_normal(&mut rng);
            let e = standard_normal(&mut rng);
            let a = ti + 0.5 * e + 0.1 * standard_normal(&mut rng);
            let b = 1.5 * ti + 0.5 * e + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * a + 1.5 * b + 0.4 * e + 0.1 * standard_normal(&mut rng);
            t[i] = ti;
            m1[i] = a;
            m2[i] = b;
            y[i] = yi;
        }
        let mut bldr = CausalSchemaBuilder::new();
        for (name, role) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("m1", RoleHint::Context),
            ("m2", RoleHint::Context),
        ] {
            bldr.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = bldr.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(m1),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(3),
                    Arc::from(m2),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let estimand = IdentifiedEstimand::frontdoor(
            "frontdoor",
            Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            ExprId::from_raw(0),
        );
        let est = FrontDoorTwoStage { bootstrap_replicates: 350, ..FrontDoorTwoStage::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = FrontDoorWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0);
        let boot = effect.se_bootstrap.expect("bootstrap SE");
        let rel = (effect.se_analytic - boot).abs() / boot.max(1e-8);
        assert!(rel < 0.3, "analytic={} boot={} rel={}", effect.se_analytic, boot, rel);
    }

    #[test]
    fn zeroing_cross_stage_meat_recovers_independent_path_delta() {
        let (data, estimand) = frontdoor_scm(2000, 21);
        let est = FrontDoorTwoStage { bootstrap_replicates: 0, ..FrontDoorTwoStage::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = FrontDoorWorkspace::default();
        let n = prep.nrows;
        let x1 = stage1_matrix(&prep.treatment);
        let s1 = est.fit_stage1(&prep.treatment, &prep.mediators[0], &mut ws).unwrap();
        let mut r1 = vec![0.0; n];
        for (row, residual) in r1.iter_mut().enumerate() {
            *residual = prep.mediators[0][row]
                - s1.coefficients[0]
                - s1.coefficients[STAGE1_TREATMENT_COL] * prep.treatment[row];
        }
        let s2 = est.fit_stage2(&prep.treatment, &prep.mediators, &prep.outcome, &mut ws).unwrap();
        let x2 = stage2_matrix(&prep.treatment, &prep.mediators);
        let mut r2 = vec![0.0; n];
        for (row, residual) in r2.iter_mut().enumerate() {
            let mut pred = 0.0;
            for c in 0..3 {
                pred += x2[c * n + row] * s2.coefficients[c];
            }
            *residual = prep.outcome[row] - pred;
        }
        let a = s1.coefficients[STAGE1_TREATMENT_COL];
        let b = s2.coefficients[STAGE2_FIRST_MEDIATOR_COL];
        let stage1_resid = [r1];
        let stage1_coefs = [s1.coefficients.clone()];
        let stages = StackedStageViews {
            x1: &x1,
            stage1_resid: &stage1_resid,
            stage1_coefs: &stage1_coefs,
            x2: &x2,
            stage2_ncols: 3,
            stage2_resid: &r2,
            stage2_coefs: &s2.coefficients,
            treatment_delta: prep.treatment_delta,
        };
        let se_indep = stacked_path_sum_se(
            &stages,
            AnalyticSeKind::Hc0,
            None,
            CrossStagePolicy::IndependentPaths,
        )
        .unwrap();
        // Ordinary independent-path delta on HC0 marginal variances of a and b.
        let (cov, _) = stacked_theta_cov_and_grad(
            &stages,
            AnalyticSeKind::Hc0,
            None,
            CrossStagePolicy::IndependentPaths,
        )
        .unwrap();
        let n_theta = 5usize;
        let idx_a = STAGE1_TREATMENT_COL;
        let idx_b = STAGE1_NCOLS + STAGE2_FIRST_MEDIATOR_COL;
        let var_a = cov[idx_a * n_theta + idx_a];
        let var_b = cov[idx_b * n_theta + idx_b];
        let cov_ab = cov[idx_a * n_theta + idx_b];
        assert!(cov_ab.abs() < 1e-12, "cross-stage block should be zeroed, got {cov_ab}");
        let d = prep.treatment_delta;
        let var_ref = (b * b * var_a + a * a * var_b) * d * d;
        assert!((se_indep - var_ref.max(0.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn clustered_stacked_se_invariant_to_cluster_relabel() {
        let (data, estimand) = frontdoor_scm(800, 31);
        let prep = FrontDoorTwoStage::new().prepare(&data, &estimand, &query()).unwrap();
        let n = prep.nrows;
        let mut groups: Vec<u32> = (0..n as u32).map(|i| i % 40).collect();
        let est = FrontDoorTwoStage {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Cluster,
            cluster_ids: Some(Arc::from(groups.as_slice())),
            ..FrontDoorTwoStage::new()
        };
        let mut ws = FrontDoorWorkspace::default();
        let se1 = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap().se_analytic;

        // Relabel clusters by a fixed permutation of ids.
        for g in &mut groups {
            *g = (*g * 7 + 3) % 40;
        }
        let est2 = FrontDoorTwoStage {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Cluster,
            cluster_ids: Some(Arc::from(groups.as_slice())),
            ..FrontDoorTwoStage::new()
        };
        let se2 = est2.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap().se_analytic;
        assert!((se1 - se2).abs() < 1e-12, "relabel changed SE: {se1} vs {se2}");
        assert!(se1.is_finite() && se1 > 0.0);
    }

    #[test]
    fn frontdoor_two_stage_rejects_unsupported_target_population() {
        let (data, estimand) = frontdoor_scm(200, 6);
        let est = FrontDoorTwoStage::new();
        let query = query().with_target_population(TargetPopulation::Treated);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }));
    }
}

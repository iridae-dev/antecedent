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
//! The analytic standard error uses a Sobel-style delta-method for a single mediator; for
//! `|M| > 1` the analytic SE is left as NaN and bootstrap is recommended.
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
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
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
}

impl Default for FrontDoorTwoStage {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontDoorTwoStage {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
        }
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
        let mut path_sum = 0.0;
        let mut beta_tm = Vec::with_capacity(k);
        for m in problem.mediators.iter() {
            let stage1 = self.fit_stage1(&problem.treatment, m, workspace)?;
            let b = stage1.coefficients[STAGE1_TREATMENT_COL];
            beta_tm.push((b, stage1.rss));
        }
        let stage2 =
            self.fit_stage2(&problem.treatment, &problem.mediators, &problem.outcome, workspace)?;
        for (j, &(b_tm, _)) in beta_tm.iter().enumerate() {
            let b_my = stage2.coefficients[STAGE2_FIRST_MEDIATOR_COL + j];
            path_sum += b_tm * b_my;
        }
        let ate = path_sum * problem.treatment_delta;

        // Single-mediator Sobel SE; multi-mediator analytic SE left as NaN (use bootstrap).
        let se_analytic = if k == 1 {
            let n = problem.nrows as f64;
            let (beta_m_t, rss1) = beta_tm[0];
            let beta_y_m = stage2.coefficients[STAGE2_FIRST_MEDIATOR_COL];
            let var1 = crate::util::coefficient_variance(
                &stage1_matrix(&problem.treatment),
                problem.nrows,
                STAGE1_NCOLS,
                STAGE1_TREATMENT_COL,
                rss1 / (n - STAGE1_NCOLS as f64).max(1.0),
            );
            let stage2_ncols = 2 + k;
            let var2 = crate::util::coefficient_variance(
                &stage2_matrix(&problem.treatment, &problem.mediators),
                problem.nrows,
                stage2_ncols,
                STAGE2_FIRST_MEDIATOR_COL,
                stage2.rss / (n - stage2_ncols as f64).max(1.0),
            );
            let var_ate = (beta_y_m * beta_y_m * var1 + beta_m_t * beta_m_t * var2)
                * problem.treatment_delta
                * problem.treatment_delta;
            var_ate.max(0.0).sqrt()
        } else {
            f64::NAN
        };
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

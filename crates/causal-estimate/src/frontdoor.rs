//! Front-door two-stage (product-of-coefficients) regression estimator .
//!
//! Requires a `"frontdoor"` estimand with a non-empty [`IdentifiedEstimand::mediators`] set
//! (see `causal_identify::frontdoor`). supports exactly one mediator `M`; the front-door
//! criterion for a valid `M` guarantees:
//!
//! 1. `M` intercepts every directed path from `T` to `Y`.
//! 2. No unblocked backdoor path from `T` to `M`.
//! 3. Every backdoor path from `M` to `Y` is blocked by conditioning on `T`.
//!
//! Pearl's front-door adjustment formula is
//!
//! ```text
//! P(y | do(t)) = Σ_m P(m | t) Σ_t' P(y | m, t') P(t')
//! ```
//!
//! For a linear structural model this collapses to the classic **product-of-coefficients**
//! (a.k.a. two-stage / indirect-effect) estimator used throughout the mediation-analysis
//! literature:
//!
//! - **Stage 1**: OLS of `M` on `[1, T]` → `β_{M←T}`, the effect of `T` on `M`.
//! - **Stage 2**: OLS of `Y` on `[1, T, M]` → `β_{Y←M·T}`, the effect of `M` on `Y` holding `T`
//! fixed. Conditioning on `T` here is what blocks the `M-Y` backdoor path guaranteed clear by
//! front-door criterion (3) above (e.g. `T <- U -> Y` confounding routed back through `M`);
//! dropping `T` from stage 2 would reintroduce that confounding.
//!
//! `ATE = β_{M←T} · β_{Y←M·T} · (active − control)`.
//!
//! This assumes no direct `T → Y` edge (all of the treatment effect flows through `M`, as the
//! front-door criterion requires) and linear structural equations; it is a practical
//! approximation rather than the fully nonparametric Pearl plug-in, matching the "practical
//! continuous approximation" used by comparable libraries.
//!
//! The analytic standard error uses the Sobel delta-method approximation
//! `SE(ab) ≈ sqrt(b² Var(a) + a² Var(b))`, treating the two stages as independent (the standard
//! simplification in mediation analysis). The bootstrap SE below refits both stages on every
//! resample and is the recommended uncertainty estimate.
//!
//! Positivity is not meaningful here — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::{bootstrap_se, BootstrapSeResult, stats_err};

/// Stage-1 design column count: `[1, T]`.
const STAGE1_NCOLS: usize = 2;
/// Stage-1 column index of the treatment coefficient (`β_{M←T}`).
const STAGE1_TREATMENT_COL: usize = 1;
/// Stage-2 design column count: `[1, T, M]`.
const STAGE2_NCOLS: usize = 3;
/// Stage-2 column index of the mediator coefficient (`β_{Y←M·T}`).
const STAGE2_MEDIATOR_COL: usize = 2;

/// Prepared front-door problem: treatment, single mediator, and outcome columns after
/// complete-case filtering.
#[derive(Clone, Debug)]
pub struct PreparedFrontDoorProblem {
    /// Treatment, length `nrows`.
    pub treatment: Arc<[f64]>,
    /// Mediator, length `nrows`.
    pub mediator: Arc<[f64]>,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Complete-case row count.
    pub nrows: usize,
    /// Estimand method tag (always `"frontdoor"`).
    pub method: Arc<str>,
    /// The single mediator variable .
    pub mediator_id: VariableId,
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
    if estimand.method_kind().ok() != Some(causal_expr::EstimandMethod::FrontDoor) {
        return Err(EstimationError::IncompatibleEstimand {
            message: "FrontDoorTwoStage expects a \"frontdoor\" estimand",
        });
    }
    if estimand.mediators.is_empty() {
        return Err(EstimationError::IncompatibleEstimand {
            message: "FrontDoorTwoStage requires a non-empty mediator set",
        });
    }
    if estimand.mediators.len() != 1 {
        return Err(EstimationError::UnsupportedQuery(
            "FrontDoorTwoStage supports exactly one mediator; multi-mediator front-door \
             sets are unsupported"
                .into(),
        ));
    }
    query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::UnsupportedQuery(
            "FrontDoorTwoStage does not support effect modifiers".into(),
        ));
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::UnsupportedQuery(
            "FrontDoorTwoStage only supports TargetPopulation::AllObserved".into(),
        ));
    }
    let treatment = query.treatment;
    let outcome = query.outcome;
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let treatment_delta = active - control;
    if treatment_delta == 0.0 {
        return Err(EstimationError::UnsupportedQuery(
            "active and control treatment levels must differ".into(),
        ));
    }
    let mediator_id = estimand.mediators[0];

    let ids = [treatment, outcome, mediator_id];
    let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
    let m = data.float64_masked(mediator_id, &row_mask).map_err(EstimationError::from)?;
    let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
    let nrows = t.len();

    Ok(PreparedFrontDoorProblem {
        treatment: Arc::from(t),
        mediator: Arc::from(m),
        outcome: Arc::from(y),
        nrows,
        method: Arc::clone(&estimand.method),
        mediator_id,
        overlap,
        treatment_delta,
    })
}

/// Estimation workspace, reused for both regression stages and across bootstrap replicates.
#[derive(Clone, Debug, Default)]
pub struct FrontDoorWorkspace {
    /// OLS scratch shared by the `M ~ T` and `Y ~ T, M` stages.
    pub ols: LeastSquaresWorkspace,
}

/// Front-door two-stage (product-of-coefficients) regression estimator.
///
/// See the module docs for the estimator definition. Supports exactly one mediator.
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

    /// Fit the two-stage product-of-coefficients estimator, with optional bootstrap.
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
        let stage1 = self.fit_stage1(&problem.treatment, &problem.mediator, workspace)?;
        let stage2 =
            self.fit_stage2(&problem.treatment, &problem.mediator, &problem.outcome, workspace)?;

        let beta_m_t = stage1.coefficients[STAGE1_TREATMENT_COL];
        let beta_y_m = stage2.coefficients[STAGE2_MEDIATOR_COL];
        let ate = beta_m_t * beta_y_m * problem.treatment_delta;

        let n = problem.nrows as f64;
        let var1 = crate::util::coefficient_variance(
            &stage1_matrix(&problem.treatment),
            problem.nrows,
            STAGE1_NCOLS,
            STAGE1_TREATMENT_COL,
            stage1.rss / (n - STAGE1_NCOLS as f64).max(1.0),
        );
        let var2 = crate::util::coefficient_variance(
            &stage2_matrix(&problem.treatment, &problem.mediator),
            problem.nrows,
            STAGE2_NCOLS,
            STAGE2_MEDIATOR_COL,
            stage2.rss / (n - STAGE2_NCOLS as f64).max(1.0),
        );
        // Sobel delta-method SE for the product β_{M←T}·β_{Y←M·T}, treating the two stages as
        // independent estimates (module docs).
        let var_ate = (beta_y_m * beta_y_m * var1 + beta_m_t * beta_m_t * var2)
            * problem.treatment_delta
            * problem.treatment_delta;
        let se_analytic = var_ate.max(0.0).sqrt();

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
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        }
        .with_bootstrap(boot))
    }

    fn fit_stage1(
        &self,
        treatment: &[f64],
        mediator: &[f64],
        workspace: &mut FrontDoorWorkspace,
    ) -> Result<causal_stats::LeastSquaresFit, EstimationError> {
        let n = treatment.len();
        let x = stage1_matrix(treatment);
        self.backend
            .least_squares(&x, n, STAGE1_NCOLS, mediator, &mut workspace.ols)
            .map_err(stats_err)
    }

    fn fit_stage2(
        &self,
        treatment: &[f64],
        mediator: &[f64],
        outcome: &[f64],
        workspace: &mut FrontDoorWorkspace,
    ) -> Result<causal_stats::LeastSquaresFit, EstimationError> {
        let n = treatment.len();
        let x = stage2_matrix(treatment, mediator);
        self.backend
            .least_squares(&x, n, STAGE2_NCOLS, outcome, &mut workspace.ols)
            .map_err(stats_err)
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedFrontDoorProblem,
        workspace: &mut FrontDoorWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let mut rng = ctx.rng.stream(0xF80D_u64);
        let n = problem.nrows;
        let mut t_boot = vec![0.0; n];
        let mut m_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, &mut rng, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                t_boot[r] = problem.treatment[src];
                m_boot[r] = problem.mediator[src];
                y_boot[r] = problem.outcome[src];
            }
            let Ok(stage1) = self.fit_stage1(&t_boot, &m_boot, workspace) else {
                return Ok(None);
            };
            let Ok(stage2) = self.fit_stage2(&t_boot, &m_boot, &y_boot, workspace) else {
                return Ok(None);
            };
            let ate = stage1.coefficients[STAGE1_TREATMENT_COL]
                * stage2.coefficients[STAGE2_MEDIATOR_COL]
                * problem.treatment_delta;
            Ok(Some(ate))
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

/// Build the column-major `[1, T, M]` stage-2 design.
fn stage2_matrix(treatment: &[f64], mediator: &[f64]) -> Vec<f64> {
    let n = treatment.len();
    let mut x = vec![0.0; n * STAGE2_NCOLS];
    x[..n].fill(1.0);
    x[n..2 * n].copy_from_slice(treatment);
    x[2 * n..3 * n].copy_from_slice(mediator);
    x
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;

    fn standard_normal(rng: &mut CausalRng) -> f64 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// `U -> T -> M -> Y` with `U -> Y` directly (no `T -> Y` edge): `T = U + noise`,
    /// `M = 2T + noise`, `Y = 3M + U + noise`. The `U` confounder makes the `T-Y` backdoor
    /// unblockable directly, but `M` satisfies the front-door criterion (mirrors the SCM in
    /// `causal_identify::frontdoor::tests::classic_frontdoor_with_unmeasured_confounder`).
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
    fn frontdoor_two_stage_rejects_multiple_mediators() {
        let (data, mut estimand) = frontdoor_scm(200, 5);
        estimand.mediators = Arc::from([VariableId::from_raw(2), VariableId::from_raw(0)]);
        let est = FrontDoorTwoStage::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }

    #[test]
    fn frontdoor_two_stage_rejects_unsupported_target_population() {
        let (data, estimand) = frontdoor_scm(200, 6);
        let est = FrontDoorTwoStage::new();
        let query = query().with_target_population(TargetPopulation::Treated);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }
}

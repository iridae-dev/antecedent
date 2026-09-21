//! Doubly robust CATE learner (`DrLearner`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_learn::{LearnerSpec, LinearSpec, LogisticSpec, PredictionTask, RidgeSpec};
use antecedent_stats::{SandwichKind, coefficient_covariance};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::learn_nuisance::{
    aipw_scores, cached_aipw_nuisances, clip_propensity, design_for_spec, learn_err,
    resolve_nuisance,
};
use crate::overlap::OverlapPolicy;
use crate::prepare::{require_adjustment_shaped, validate_ate_query_with_targets};
use crate::propensity::{
    PreparedPropensityProblem, clip_of, default_propensity_overlap,
    prepare_propensity_problem_with_registry, trim_of, trim_retained_rows,
};

/// Cross-fitted DR-Learner: CATE from a final-stage regression on the DR score.
#[derive(Clone, Debug, PartialEq)]
pub struct DrLearner {
    /// Cross-fit folds.
    pub folds: usize,
    /// Outcome nuisance.
    pub outcome: LearnerSpec,
    /// Treatment nuisance.
    pub treatment: LearnerSpec,
    /// Final-stage regression of the DR pseudo-outcome.
    pub final_learner: LearnerSpec,
    /// Overlap policy.
    pub overlap: OverlapPolicy,
}

impl Default for DrLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl DrLearner {
    /// Ridge / logistic nuisances and a linear final stage.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            folds: 5,
            outcome: LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 }),
            treatment: LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 }),
            final_learner: LearnerSpec::Linear(LinearSpec {}),
            overlap: default_propensity_overlap(),
        }
    }

    /// Adapt one shared spec to each first-stage nuisance task.
    #[must_use]
    pub const fn with_learner(mut self, spec: LearnerSpec) -> Self {
        self.outcome = spec.for_task(PredictionTask::Regression);
        self.treatment = spec.for_task(PredictionTask::BinaryProbability);
        self
    }

    /// Outcome nuisance.
    #[must_use]
    pub const fn with_outcome(mut self, spec: LearnerSpec) -> Self {
        self.outcome = spec;
        self
    }

    /// Treatment nuisance.
    #[must_use]
    pub const fn with_treatment(mut self, spec: LearnerSpec) -> Self {
        self.treatment = spec;
        self
    }

    /// Final-stage CATE regression.
    #[must_use]
    pub const fn with_final_learner(mut self, spec: LearnerSpec) -> Self {
        self.final_learner = spec;
        self
    }

    /// Fold count.
    #[must_use]
    pub const fn with_folds(mut self, folds: usize) -> Self {
        self.folds = folds;
        self
    }

    /// Overlap policy.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Prepare the adjustment design.
    ///
    /// # Errors
    ///
    /// Non-adjustment estimand or propensity prepare failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        require_adjustment_shaped(estimand, "DRLearner requires an adjustment-shaped estimand")?;
        validate_ate_query_with_targets(query)?;
        prepare_propensity_problem_with_registry(data, estimand, query, self.overlap, None)
    }

    /// Fit OOF nuisances, the DR pseudo-outcome, and the final-stage CATE.
    ///
    /// # Errors
    ///
    /// Learner failure or a target other than `AllObserved`.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::TargetPopulation);
        }
        if self.folds < 2 {
            return Err(EstimationError::data_msg("DRLearner requires at least two folds"));
        }
        let nuisance =
            cached_aipw_nuisances(problem, self.outcome, self.treatment, self.folds, ctx)?;
        let (mu0, mu1, raw_e, treat) = nuisance.as_ref();
        let mut ehat = raw_e.clone();
        clip_propensity(&mut ehat, clip_of(problem.overlap));
        let phi =
            aipw_scores(problem.treatment.as_ref(), problem.outcome.as_ref(), &ehat, &mu0, &mu1);
        let (factory, _) = resolve_nuisance(
            self.final_learner,
            PredictionTask::Regression,
            problem.design_matrix.as_ref(),
            problem.nrows,
            problem.design_ncols,
            &phi,
            ctx,
        )?;
        let view = design_for_spec(
            self.final_learner,
            problem.design_matrix.as_ref(),
            problem.nrows,
            problem.design_ncols,
        )?;
        // The reported ATE averages the common-support rows only, so the CATE regression is
        // fitted on those rows too; trimmed extreme-weight rows would otherwise dominate the
        // final stage while being excluded from the estimand.
        let retained = trim_retained_rows(raw_e, trim_of(problem.overlap))?;
        let retained_u32: Option<Vec<u32>> = retained
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|&i| {
                        u32::try_from(i).map_err(|_| {
                            EstimationError::data_msg("row index exceeds the u32 row-id capacity")
                        })
                    })
                    .collect::<Result<Vec<u32>, _>>()
            })
            .transpose()?;
        let fit_view = match &retained_u32 {
            Some(rows) => {
                view.with_rows(antecedent_learn::RowSelection::new(rows)).map_err(learn_err)?
            }
            None => view,
        };
        let fitted = factory
            .fit(fit_view, antecedent_learn::TargetView::new(&phi), None, ctx)
            .map_err(learn_err)?;
        let mut cate = vec![0.0; problem.nrows];
        fitted.predict(view, &mut cate, ctx).map_err(learn_err)?;
        let yhat = problem
            .treatment
            .iter()
            .zip(mu0)
            .zip(mu1)
            .map(|((&t, &m0), &m1)| if t > 0.5 { m1 } else { m0 })
            .collect::<Vec<_>>();
        let outcome_diag =
            antecedent_learn::diagnose(PredictionTask::Regression, problem.outcome.as_ref(), &yhat);
        let mut effect = crate::dml::finish_dml(
            &phi,
            problem,
            assumptions,
            treat.validation.logloss,
            outcome_diag.r2,
            raw_e.clone(),
        )?
        .with_cate(Some(Arc::from(cate.clone())))
        .with_cate_se(linear_cate_pointwise_se(
            self.final_learner,
            view,
            &phi,
            &cate,
            retained.as_deref(),
        ));
        effect.crossfit_folds = Some(self.folds);
        effect.crossfit_seed = Some(ctx.rng.master_seed());
        effect.learner_provenance = treat.model_provenance.clone();
        effect.learner_provenance.push(fitted.provenance());
        match fitted.portable() {
            Ok(predictor) => {
                let model = crate::FittedEffect {
                    version: 1,
                    features: problem.adjustment_set.iter().map(|v| v.raw()).collect(),
                    intercept: view.ncols() == problem.design_ncols,
                    predictor,
                };
                model.validate()?;
                effect.fitted_effect = Some(Arc::new(model));
            }
            Err(antecedent_learn::LearnError::Unsupported { .. }) => {}
            Err(error) => return Err(learn_err(error)),
        }
        Ok(effect)
    }
}

/// HC0 sandwich SEs for a linear CATE regression of orthogonal scores.
///
/// The regression uses `rows` (the retained common-support rows; all rows when `None`) and the
/// covariance is the stats crate's HC0 `(XᵀX)⁻¹ Σ e² x xᵀ (XᵀX)⁻¹`, evaluated at every row's
/// features. Cross-fitting plus Neyman orthogonality of φ makes first-stage estimation
/// second-order. Penalized or nonlinear finals are not this OLS map, so they return `None`
/// instead of an invented interval.
fn linear_cate_pointwise_se(
    spec: LearnerSpec,
    x: antecedent_learn::DesignView<'_>,
    phi: &[f64],
    cate: &[f64],
    rows: Option<&[usize]>,
) -> Option<Arc<[f64]>> {
    if !matches!(spec, LearnerSpec::Linear(_)) {
        return None;
    }
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || phi.len() != n || cate.len() != n {
        return None;
    }
    let mut design = vec![0.0; n.saturating_mul(p)];
    for c in 0..p {
        for r in 0..n {
            design[c * n + r] = x.get(r, c).ok()?;
        }
    }
    let all_rows: Vec<usize>;
    let used: &[usize] = match rows {
        Some(rows) => rows,
        None => {
            all_rows = (0..n).collect();
            &all_rows
        }
    };
    let m = used.len();
    let mut sub = vec![0.0; m * p];
    for c in 0..p {
        for (k, &r) in used.iter().enumerate() {
            sub[c * m + k] = design[c * n + r];
        }
    }
    let residuals: Vec<f64> = used.iter().map(|&r| phi[r] - cate[r]).collect();
    let cov = coefficient_covariance(&sub, m, p, &residuals, SandwichKind::Hc0).ok()?;
    let mut se = vec![0.0; n];
    for i in 0..n {
        let mut v = 0.0;
        for j in 0..p {
            let mut cj = 0.0;
            for k in 0..p {
                cj += cov[j * p + k] * design[k * n + i];
            }
            v += design[j * n + i] * cj;
        }
        if !(v.is_finite() && v >= 0.0) {
            return None;
        }
        se[i] = v.sqrt();
    }
    Some(Arc::from(se))
}

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;

    use super::*;

    fn interaction_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand, Vec<f64>) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x51u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let p = 1.0 / (1.0 + (-(-0.3 + 0.6 * zi)).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = (1.0 + zi) * ti + 0.4 * zi + standard_normal(&mut rng) * 0.4;
        }
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
            "z",
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
                    Arc::from(z.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (
            TabularData::new(storage),
            IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from([VariableId::from_raw(2)]),
                ExprId::from_raw(0),
            ),
            z,
        )
    }

    #[test]
    fn interaction_recovers_monotone_cate() {
        let (data, estimand, z) = interaction_scm(900, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = DrLearner::new();
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ExecutionContext::for_tests(1), AssumptionSet::new()).unwrap();
        let cate = effect.cate.as_ref().expect("cate");
        let mut pairs: Vec<(f64, f64)> =
            z.iter().zip(cate.iter()).map(|(&zi, &c)| (zi, c)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let lo =
            pairs[..pairs.len() / 4].iter().map(|p| p.1).sum::<f64>() / (pairs.len() / 4) as f64;
        let hi = pairs[3 * pairs.len() / 4..].iter().map(|p| p.1).sum::<f64>()
            / (pairs.len() - 3 * pairs.len() / 4) as f64;
        assert!(hi > lo + 0.2, "cate should rise with z: lo={lo} hi={hi}");
        let mean_phi = effect.influence.as_ref().map(|p| p.iter().sum::<f64>() / p.len() as f64);
        if let Some(m) = mean_phi {
            assert!(m.abs() < 1e-10, "influence must be centered: {m}");
        }
        let se = effect.cate_se.as_ref().expect("linear final stage licenses CATE SEs");
        assert_eq!(se.len(), cate.len());
        assert!(se.iter().all(|&s| s.is_finite() && s > 0.0));
    }

    #[test]
    fn penalized_final_stage_withholds_cate_se() {
        let (data, estimand, _) = interaction_scm(200, 5);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est =
            DrLearner::new().with_final_learner(LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 }));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ExecutionContext::for_tests(1), AssumptionSet::new()).unwrap();
        assert!(effect.cate.is_some());
        assert!(effect.cate_se.is_none());
    }

    #[test]
    fn linear_cate_se_is_hc0_on_the_retained_rows() {
        // Design [1 | z] on 6 rows; the regression uses rows 1..=4 only. With the hand-coded
        // 2×2 sandwich `(XᵀX)⁻¹ Σ e² x xᵀ (XᵀX)⁻¹` on those rows, SE(x) = √(xᵀ V x).
        let z = [9.0, 0.0, 1.0, 2.0, 3.0, -7.0];
        let mut design = vec![1.0; 6];
        design.extend_from_slice(&z);
        let view = antecedent_learn::DesignView::from_column_major(&design, 6, 2).unwrap();
        let phi = [50.0, 1.0, 2.5, 2.0, 4.5, -40.0];
        // OLS of φ on [1, z] over rows 1..=4: z̄ = 1.5, φ̄ = 2.5, Sxx = 5, Sxy = 5.
        let slope = 5.0 / 5.0;
        let intercept = 2.5 - slope * 1.5;
        let cate: Vec<f64> = z.iter().map(|zi| intercept + slope * zi).collect();
        let rows = [1usize, 2, 3, 4];
        let se = linear_cate_pointwise_se(
            LearnerSpec::Linear(LinearSpec::default()),
            view,
            &phi,
            &cate,
            Some(&rows),
        )
        .unwrap();
        // XᵀX = [[4, 6], [6, 14]], inverse = 1/20 · [[14, −6], [−6, 4]].
        let inv = [[0.7, -0.3], [-0.3, 0.2]];
        let mut meat = [[0.0; 2]; 2];
        for &r in &rows {
            let e2 = (phi[r] - cate[r]).powi(2);
            let x = [1.0, z[r]];
            for a in 0..2 {
                for b in 0..2 {
                    meat[a][b] += e2 * x[a] * x[b];
                }
            }
        }
        let mut tmp = [[0.0; 2]; 2];
        let mut v = [[0.0; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                tmp[a][b] = (0..2).map(|k| inv[a][k] * meat[k][b]).sum();
            }
        }
        for a in 0..2 {
            for b in 0..2 {
                v[a][b] = (0..2).map(|k| tmp[a][k] * inv[k][b]).sum();
            }
        }
        for (i, &zi) in z.iter().enumerate() {
            let x = [1.0, zi];
            let var: f64 = (0..2).map(|a| (0..2).map(|b| x[a] * v[a][b] * x[b]).sum::<f64>()).sum();
            assert!((se[i] - var.sqrt()).abs() < 1e-10, "row {i}: {} vs {}", se[i], var.sqrt());
        }
    }
}

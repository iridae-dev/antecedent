//! Doubly robust CATE learner (`DrLearner`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names, clippy::similar_names)]

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_learn::{LearnerSpec, LinearSpec, LogisticSpec, PredictionTask, RidgeSpec};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::learn_nuisance::{
    aipw_scores, clip_propensity, cross_fit_aipw_nuisances, design_for_spec, learn_err,
    resolve_nuisance,
};
use crate::overlap::OverlapPolicy;
use crate::prepare::{require_adjustment_shaped, validate_ate_query_with_targets};
use crate::propensity::{
    PreparedPropensityProblem, clip_of, default_propensity_overlap,
    prepare_propensity_problem_with_registry,
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
            treatment: LearnerSpec::Logistic(LogisticSpec {}),
            final_learner: LearnerSpec::Linear(LinearSpec {}),
            overlap: default_propensity_overlap(),
        }
    }

    /// Copy one spec onto both first-stage nuisances.
    #[must_use]
    pub const fn with_learner(mut self, spec: LearnerSpec) -> Self {
        self.outcome = spec;
        self.treatment = spec;
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
        let (mu0, mu1, mut ehat, treat) = cross_fit_aipw_nuisances(
            self.outcome,
            self.treatment,
            problem.design_matrix.as_ref(),
            problem.nrows,
            problem.design_ncols,
            problem.outcome.as_ref(),
            problem.treatment.as_ref(),
            self.folds,
            ctx,
        )?;
        let raw_e = ehat.clone();
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
        let fitted = factory
            .fit(view, antecedent_learn::TargetView::new(&phi), None, ctx)
            .map_err(learn_err)?;
        let mut cate = vec![0.0; problem.nrows];
        fitted.predict(view, &mut cate, ctx).map_err(learn_err)?;
        let mut effect = crate::dml::finish_dml(&phi, problem, assumptions, None, None, raw_e)?
            .with_cate(Some(Arc::from(cate)));
        effect.learner_provenance = treat.model_provenance;
        effect.learner_provenance.push(fitted.provenance());
        Ok(effect)
    }
}

#[cfg(test)]
mod tests {
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
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x51u64);
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
    }
}

//! Logical compilation for specialized study designs.

use super::{
    Arc, CausalError, ContinuousDoseResponseQuery, DataClassification, EstimatorId,
    HistoryMatchQuery, IdentifierId, LocalPolynomialRatioQuery, LogicalAnalysisPlan,
    LongitudinalRegimeQuery, PanelDidQuery, PlanId, PolicyValueQuery, RandomizedEffectQuery,
    SequentialContrastQuery, SurvivalQuery, SyntheticControlQuery, TableView, TabularData,
};

impl super::super::Study {
    pub(super) fn compile_logical_randomized(
        &self,
        data: &TabularData,
        q: &RandomizedEffectQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.realized_assignment.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "randomized assignment metadata must align with the table rows".into(),
            });
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            PlanId::RandomizedTrialItt.as_str(),
            "randomized.design",
            if self.settings.inference.is_bayesian() {
                super::super::checked_randomized::bayesian_randomized_estimator(
                    self.declared.estimator,
                    q,
                )
                .as_str()
            } else {
                match &q.design {
                    antecedent_core::RandomizationDesign::Bernoulli => {
                        if q.received_treatment.is_some() {
                            "randomized.wald_cace_late"
                        } else if q.fixed_cuped.is_some() {
                            "randomized.fixed_cuped_ht_itt"
                        } else if !q.ancova_covariates.is_empty() {
                            "randomized.ancova_itt"
                        } else {
                            "randomized.ht_itt"
                        }
                    }
                    antecedent_core::RandomizationDesign::Cluster { .. }
                        if q.fixed_cuped.is_some() =>
                    {
                        "randomized.cluster_fixed_cuped_itt"
                    }
                    antecedent_core::RandomizationDesign::Complete { .. }
                    | antecedent_core::RandomizationDesign::Stratified { .. }
                    | antecedent_core::RandomizationDesign::Factorial2x2 { .. }
                    | antecedent_core::RandomizationDesign::Cluster { .. } => {
                        "randomized.neyman_itt"
                    }
                    antecedent_core::RandomizationDesign::Switchback { .. } => {
                        "randomized.switchback_ht_itt"
                    }
                    antecedent_core::RandomizationDesign::MultiArm { .. } => "randomized.ht_itt",
                }
            },
            Arc::from([q.outcome]),
        ))
    }

    pub(super) fn compile_logical_policy_value(
        &self,
        data: &TabularData,
        q: &PolicyValueQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.multi_action.as_ref().map_or(q.assignment.len(), |multi| multi.assignment.len())
            != data.row_count()
        {
            return Err(CausalError::Compile {
                message: "policy inputs must align with the table rows".into(),
            });
        }
        if self.settings.inference.is_bayesian() {
            return Ok(self.study_query_plan(
                data.row_count() as u64,
                DataClassification::Tabular,
                PlanId::BayesianPolicyValue.as_str(),
                "randomized.design",
                EstimatorId::BayesianBinaryPolicyValue.as_str(),
                Arc::from([q.outcome]),
            ));
        }
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            if q.multi_action.is_some() {
                PlanId::RandomizedPolicyValueMultiActionIpw.as_str()
            } else if q.mu0.is_empty() {
                PlanId::RandomizedPolicyValueIpw.as_str()
            } else {
                PlanId::RandomizedPolicyValueDr.as_str()
            },
            "randomized.design",
            if q.multi_action.is_some() {
                "randomized.multi_action_ipw_policy"
            } else if q.mu0.is_empty() {
                "randomized.ipw_policy"
            } else {
                "randomized.dr_policy"
            },
            Arc::from([q.outcome]),
        ))
    }

    pub(super) fn compile_logical_panel_did(
        &self,
        data: &TabularData,
        q: &PanelDidQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.treated.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "panel DiD metadata must align with table rows".into(),
            });
        }
        let (plan_id, identifier, estimator) = if self.settings.inference.is_bayesian() {
            super::super::checked_panel_did::bayesian_panel_did_plan_ids(q.design)
        } else {
            super::super::checked_panel_did::panel_did_plan_ids(q.design)
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan_id,
            identifier,
            estimator,
            Arc::from([q.outcome]),
        ))
    }

    pub(super) fn compile_logical_synthetic_control(
        &self,
        data: &TabularData,
        q: &SyntheticControlQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.units.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "synthetic-control metadata must align with table rows".into(),
            });
        }
        let (plan_id, identifier, estimator) = if self.settings.inference.is_bayesian() {
            super::super::checked_synthetic_control::bayesian_synthetic_panel_plan_ids()
        } else {
            super::super::checked_synthetic_control::synthetic_query_plan_ids(q)
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan_id,
            identifier,
            estimator,
            Arc::from([q.outcome]),
        ))
    }

    pub(super) fn compile_logical_continuous_dose(
        &self,
        data: &TabularData,
        q: &ContinuousDoseResponseQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.baseline_groups.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "continuous-dose baseline groups must align with table rows".into(),
            });
        }
        let (plan, estimator) = if self.settings.inference.is_bayesian() {
            (PlanId::BayesianDosePolicy, EstimatorId::BayesianDosePolicy)
        } else {
            (PlanId::PolicyContinuousDoseResponse, EstimatorId::ContinuousDoseKernel)
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan.as_str(),
            IdentifierId::ContinuousDoseExchangeability.as_str(),
            estimator.as_str(),
            Arc::from([q.outcome, q.dose, q.dose_density]),
        ))
    }

    pub(super) fn compile_logical_local_polynomial_ratio(
        &self,
        data: &TabularData,
        q: &LocalPolynomialRatioQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for variable in [q.outcome, q.treatment, q.running] {
            data.schema().get(variable).map_err(CausalError::rejected)?;
        }
        let (plan_id, identifier, estimator) = if self.settings.inference.is_bayesian() {
            super::super::checked_local_polynomial_ratio::bayesian_local_polynomial_ratio_plan_ids(
                q.kink,
            )
        } else {
            super::super::checked_local_polynomial_ratio::local_ratio_query_plan_ids(q)
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            &plan_id,
            &identifier,
            &estimator,
            Arc::from([q.outcome, q.treatment, q.running]),
        ))
    }

    pub(super) fn compile_logical_survival(
        &self,
        data: &TabularData,
        q: &SurvivalQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in [Some(q.duration), Some(q.event), Some(q.treatment), q.delayed_entry]
            .into_iter()
            .flatten()
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let (plan, estimator) = if self.settings.inference.is_bayesian() {
            (
                PlanId::BayesianPiecewiseSurvival.as_str(),
                EstimatorId::BayesianPiecewiseSurvival.as_str(),
            )
        } else {
            (
                PlanId::RandomizedSurvivalProductLimit.as_str(),
                PlanId::RandomizedSurvivalProductLimit.as_str(),
            )
        };
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            estimator,
            Arc::from([q.duration, q.event, q.treatment]),
        ))
    }

    pub(super) fn compile_logical_history_match(
        &self,
        data: &TabularData,
        q: &HistoryMatchQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in [q.outcome, q.treated]
            .into_iter()
            .chain(q.history.iter().copied())
            .chain(q.period)
            .chain(q.weight)
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        // The matching contrast has no posterior; a Bayesian request reaches the outcome-model
        // estimator only by selecting it, and otherwise keeps the matching plan so the typed
        // boundary refuses it at build with the estimator named as the next attempt.
        let plan = if self.settings.inference.is_bayesian()
            && self.declared.estimator == Some(EstimatorId::BayesianPeriodHistoryOutcomeModel)
        {
            PlanId::BayesianPeriodHistoryOutcomeModel
        } else {
            PlanId::PeriodHistoryMatching
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.treated]),
        ))
    }

    pub(super) fn compile_logical_iv_bounds(
        &self,
        data: &TabularData,
        q: &antecedent_core::IvBoundsQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in super::super::checked_iv_bounds::query_columns(q) {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = if self.settings.inference.is_bayesian() {
            PlanId::BayesianIvBounds
        } else {
            PlanId::IvBoundsSharp
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.treatment, q.instrument]),
        ))
    }

    /// Logical plan of a noncausal association. The plan names no causal identifier or
    /// estimator: the analysis is picked from the outcome scale by the declared selection rule
    /// and recorded on the lowered program.
    pub(super) fn compile_logical_association(
        &self,
        data: &impl TableView,
        q: &antecedent_core::AssociationQuery,
        classification: DataClassification,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        for id in q.columns() {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = if self.settings.inference.is_bayesian() {
            PlanId::BayesianAssociation
        } else {
            PlanId::Association
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            classification,
            plan,
            super::super::checked_association::ASSOCIATION_IDENTIFIER,
            plan,
            Arc::from(q.columns()),
        ))
    }

    pub(super) fn compile_logical_temporal_design(
        &self,
        data: &TabularData,
        q: &antecedent_core::TemporalDesignQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.rows() != data.row_count() {
            return Err(CausalError::Compile {
                message: "temporal-design unit and period labels must align with the table rows"
                    .into(),
            });
        }
        for id in q.columns() {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let survival =
            matches!(q.assignment, antecedent_core::TemporalAssignment::RegimeSurvival { .. });
        let plan = match (survival, self.settings.inference.is_bayesian()) {
            (false, false) => PlanId::TemporalDesignExcursion,
            (false, true) => PlanId::BayesianTemporalDesignExcursion,
            (true, false) => PlanId::TemporalRegimeSurvival,
            (true, true) => PlanId::BayesianTemporalRegimeSurvival,
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from(q.columns()),
        ))
    }

    pub(super) fn compile_logical_sequential_contrast(
        &self,
        data: &TabularData,
        q: &SequentialContrastQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.fold_of_row.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "sequential contrast folds must align with the table rows".into(),
            });
        }
        for id in [q.outcome, q.first_treatment, q.second_treatment]
            .into_iter()
            .chain(q.baseline.iter().copied())
            .chain(q.intermediate.iter().copied())
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let plan = if self.settings.inference.is_bayesian() {
            PlanId::BayesianSequentialCrossfitContrast
        } else {
            PlanId::SequentialCrossfitContrastWald
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            plan,
            "randomized.design",
            plan,
            Arc::from([q.outcome, q.first_treatment, q.second_treatment]),
        ))
    }

    pub(super) fn compile_logical_longitudinal_regime(
        &self,
        data: &TabularData,
        q: &LongitudinalRegimeQuery,
    ) -> Result<LogicalAnalysisPlan, CausalError> {
        q.validate()?;
        if q.subject_ids.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "longitudinal subject histories must align with outcome rows".into(),
            });
        }
        data.schema().get(q.outcome).map_err(CausalError::rejected)?;
        let bayesian = self.settings.inference.is_bayesian();
        let method_id = match q.method {
            _ if bayesian => PlanId::BayesianSequentialRegimeValue,
            antecedent_core::LongitudinalRegimeMethod::Ipw => PlanId::LongitudinalIpwRegime,
            antecedent_core::LongitudinalRegimeMethod::GFormula => {
                PlanId::LongitudinalGFormulaRegime
            }
            antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => {
                PlanId::LongitudinalSequentialDrRegime
            }
            antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                PlanId::LongitudinalMarginalStructuralModel
            }
        }
        .as_str();
        Ok(self.study_query_plan(
            data.row_count() as u64,
            DataClassification::Tabular,
            method_id,
            "randomized.design",
            method_id,
            Arc::from([q.outcome]),
        ))
    }
}

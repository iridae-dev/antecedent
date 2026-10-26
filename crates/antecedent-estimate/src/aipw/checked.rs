//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Licensed procedure retained by the checked frequentist AIPW ATE route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckedAipwProcedure {
    /// Logistic propensity and arm-specific OLS outcome nuisances with cross-fitted scores.
    CrossFittedLogisticOls,
    /// Logistic propensity and arm-specific OLS outcome nuisances refit on the
    /// common-support rows a trim rule retains; full-sample residualized, no score table.
    TrimmedLogisticOls,
}

impl CheckedAipwProcedure {
    /// The procedure an overlap policy selects: the trim rule decides between the
    /// cross-fitted score table and the full-sample common-support refit.
    #[must_use]
    pub fn for_overlap(overlap: OverlapPolicy) -> Self {
        if trim_of(overlap).is_some() {
            Self::TrimmedLogisticOls
        } else {
            Self::CrossFittedLogisticOls
        }
    }

    /// Cross-fit fold count the procedure runs; zero for the full-sample refit.
    #[must_use]
    pub const fn folds(self) -> usize {
        match self {
            Self::CrossFittedLogisticOls => crate::crossfit_aipw::DEFAULT_AIPW_FOLDS,
            Self::TrimmedLogisticOls => 0,
        }
    }

    /// Stable wire spelling of the procedure.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CrossFittedLogisticOls => "cross_fitted_logistic_ols",
            Self::TrimmedLogisticOls => "trimmed_logistic_ols",
        }
    }
}

/// Selected target roles and numerical choices for checked AIPW execution.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedAipwLowering {
    /// Source causal target root.
    pub functional: antecedent_expr::ExprId,
    /// Treatment role.
    pub treatment: antecedent_core::VariableId,
    /// Outcome role.
    pub outcome: antecedent_core::VariableId,
    /// Adjustment roles in selected-estimand order.
    pub adjustment: Arc<[antecedent_core::VariableId]>,
    /// Population whose mean contrast is estimated.
    pub population: TargetPopulation,
    /// Estimator procedure retained for execution.
    pub procedure: CheckedAipwProcedure,
    /// Overlap policy the procedure clips and trims with; its trim rule selects
    /// `procedure`.
    pub overlap: OverlapPolicy,
    /// Cross-fit fold count (zero for the trimmed full-sample procedure).
    pub folds: usize,
    /// Analytic uncertainty method.
    pub se_kind: AnalyticSeKind,
    /// Bootstrap replicate count (zero disables bootstrap uncertainty).
    pub bootstrap_replicates: u32,
    /// Declared binary-propensity nuisance. A penalized one keeps the cross-fitted
    /// `procedure` but withholds every interval; the configuration is part of the receipt.
    pub propensity: PropensityNuisance,
    /// Declared cluster-DML independence unit, when the cross-fit owns whole clusters.
    pub cluster_dml: Option<ClusterDml>,
    /// Complete-case source row identities bound during preparation.
    pub rows: Arc<[antecedent_data::GlobalId]>,
}

/// Checked AIPW preparation sealed to one identified ATE claim and data binding.
#[derive(Clone, Debug)]
pub struct CheckedAipwPreparation {
    pub(super) target: IdentifiedEstimand,
    pub(super) program: FunctionalProgram,
    pub(super) factor_requirements: Arc<[FactorRequirement]>,
    pub(super) lowering: CheckedAipwLowering,
    pub(super) query: AverageEffectQuery,
    pub(super) required_assumptions: AssumptionSet,
    pub(super) problem: PreparedPropensityProblem,
}

impl CheckedAipwPreparation {
    /// Selected identified target.
    #[must_use]
    pub fn target(&self) -> &IdentifiedEstimand {
        &self.target
    }
    /// Checked owner of the selected causal functional.
    #[must_use]
    pub fn program(&self) -> &FunctionalProgram {
        &self.program
    }
    /// Checked provider requirements of the retained target program.
    #[must_use]
    pub fn factor_requirements(&self) -> &[FactorRequirement] {
        &self.factor_requirements
    }
    /// Retained typed procedure and role bindings.
    #[must_use]
    pub fn lowering(&self) -> &CheckedAipwLowering {
        &self.lowering
    }
    /// Selected identification assumptions.
    #[must_use]
    pub fn required_assumptions(&self) -> &AssumptionSet {
        &self.required_assumptions
    }
    /// Bound numerical design, including complete-case row identities.
    #[must_use]
    pub fn problem(&self) -> &PreparedPropensityProblem {
        &self.problem
    }
}

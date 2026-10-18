//! Row accounting and the explicit-policy check of tabular adjustment routes.
//!
//! These routes select complete cases inside their estimators; this module reports what that
//! selection dropped, over the variables the result reads, and refuses a table with missing
//! values when the caller declared `error`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{AnalysisRoute, CausalError, Study, StudyResult, TabularData};
use crate::analysis::missing_data::{self, MissingRoute};
use antecedent_data::{MissingPolicy, RowAccounting, complete_rows, first_missing_cell};

impl Study {
    /// Whether the caller declared complete-case handling, which lets gap-aware estimators
    /// fit the intact lag windows of a series with missing time points.
    pub(super) fn declares_complete_case(&self) -> bool {
        self.missing
            .as_ref()
            .is_some_and(|declaration| declaration.policy == MissingPolicy::CompleteCase)
    }

    /// Attach the missing-data report of a static adjustment result.
    ///
    /// Routes that account for their own rows (randomized, panel) already carry one.
    pub(super) fn report_missing_data(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        result: &mut StudyResult,
    ) -> Result<(), CausalError> {
        if !matches!(
            route,
            AnalysisRoute::StaticAte
                | AnalysisRoute::Conditional
                | AnalysisRoute::StaticMediation
                | AnalysisRoute::Response
                | AnalysisRoute::Distribution
        ) {
            return Ok(());
        }
        self.report_static_missing_data(data, result)
    }

    /// The missing-data report of a static adjustment result, attached once.
    pub(crate) fn report_static_missing_data(
        &self,
        data: &TabularData,
        result: &mut StudyResult,
    ) -> Result<(), CausalError> {
        if result.missing_data.is_some() {
            return Ok(());
        }
        let mut variables: Vec<antecedent_core::VariableId> = Vec::new();
        for id in result
            .treatment
            .into_iter()
            .chain(std::iter::once(result.outcome))
            .chain(result.estimand.adjustment_set.iter().copied())
            .chain(result.estimand.instruments.iter().copied())
            .chain(result.estimand.mediators.iter().copied())
        {
            if !variables.contains(&id) {
                variables.push(id);
            }
        }
        let (policy, declared) =
            missing_data::effective_policy(MissingRoute::Static, self.missing.as_ref());
        if policy == MissingPolicy::ErrorOnMissing {
            if let Some(cell) = first_missing_cell(data, &variables)? {
                return Err(CausalError::Estimate(
                    antecedent_estimate::missing_policy::policy_required(
                        &cell,
                        MissingRoute::Static.label(),
                        &[MissingPolicy::CompleteCase],
                    ),
                ));
            }
        }
        let mask = data.storage().analysis_mask();
        let keep = complete_rows(data, mask, &variables)?;
        let accounting = RowAccounting::from_selection(data, mask, policy, &variables, &keep)?;
        result.missing_data = Some(missing_data::report(
            MissingRoute::Static,
            declared,
            std::sync::Arc::from([]),
            accounting,
        ));
        Ok(())
    }
}

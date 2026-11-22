//! A power-prior discount selected from the study's own table.
//!
//! [`OutcomePriorSelection`] declares external sources for the outcome-nuisance regression of a
//! cross-fitted AIPW or DML estimate together with the rule that chooses their common discount
//! `alpha` from the target data. It is resolved when the study is built, against the table the
//! study analyses: a study built on each completed table of an imputation therefore selects its
//! own discount on that table. The selection (criterion, selected `alpha`, level and interval
//! semantics) is a restriction of the composed prior, so it is recorded in the assumptions of
//! every result that used the prior and in the prior's identity.
//!
//! The selection runs on the design the nuisance regression itself uses: one column per named
//! coefficient (`intercept`, and `coef_<column>` for a column of the table), the study's outcome
//! as the response, every analysed row, and the mean of the design as the functional whose
//! posterior is scored. The prior shapes the efficiency of the estimate and never its
//! identification.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::OutcomePriorSource;

use crate::error::CausalError;
use crate::inference::{
    AlphaSelectionOptions, ComposedPrior, ExternalPriorSource, LinearTarget, PriorSet,
    SelectedAlphaReport, outcome_prior_source_from_composed, select_linear_prior,
};

/// External sources and the rule that selects their common discount from the study's table.
#[derive(Clone, Debug)]
pub struct OutcomePriorSelection {
    sources: Arc<[ExternalPriorSource]>,
    baseline: PriorSet,
    coefficient_names: Arc<[Arc<str>]>,
    options: AlphaSelectionOptions,
}

/// The prior a selection resolved to on one table, with the selection report.
#[derive(Clone, Debug)]
pub struct ResolvedOutcomePrior {
    /// The outcome-nuisance prior the estimator takes.
    pub source: OutcomePriorSource,
    /// The composed prior at the selected discount, named by coefficient.
    pub composed: ComposedPrior,
    /// The criterion, selected discount, intervals and sensitivity table.
    pub report: SelectedAlphaReport,
}

impl OutcomePriorSelection {
    /// A selection over `sources` composed on `baseline`, whose coefficients are `names`
    /// (`intercept`, then `coef_<column>` for each adjusted column), under `options`.
    ///
    /// # Errors
    ///
    /// No source, or names that are empty, repeated or not led by `intercept`.
    pub fn new(
        sources: Vec<ExternalPriorSource>,
        baseline: PriorSet,
        coefficient_names: Vec<Arc<str>>,
        options: AlphaSelectionOptions,
    ) -> Result<Self, CausalError> {
        if sources.is_empty() {
            return Err(crate::compile_reason!(
                "required_option_missing",
                "an alpha selection needs at least one external source"
            ));
        }
        if coefficient_names.first().map(AsRef::as_ref) != Some("intercept") {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "the selected prior names its coefficients `intercept`, then `coef_<column>` for \
                 each adjusted column"
            ));
        }
        for (index, name) in coefficient_names.iter().enumerate() {
            if name.is_empty() || coefficient_names[..index].iter().any(|other| other == name) {
                return Err(crate::compile_reason!(
                    "invalid_argument",
                    "coefficient names must be non-empty and unique"
                ));
            }
        }
        Ok(Self {
            sources: Arc::from(sources),
            baseline,
            coefficient_names: Arc::from(coefficient_names),
            options,
        })
    }

    /// The selection a declared [`antecedent_estimate::AlphaSelectionRequest`] states.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn from_request(
        request: &antecedent_estimate::AlphaSelectionRequest,
    ) -> Result<Self, CausalError> {
        Self::new(
            request.sources.clone(),
            request.baseline.clone(),
            request.coefficient_names.clone(),
            request.options.clone(),
        )
    }

    /// The coefficient names of the selected prior.
    #[must_use]
    pub fn coefficient_names(&self) -> &[Arc<str>] {
        &self.coefficient_names
    }

    /// Select the discount from `table` (the outcome regressed on the named columns over every
    /// row with all of them observed) and compose the sources at it.
    ///
    /// # Errors
    ///
    /// A coefficient that names no column of the table, fewer complete rows than coefficients,
    /// or the refusals of the selection and composition.
    pub fn resolve(
        &self,
        table: &TabularData,
        outcome: VariableId,
    ) -> Result<ResolvedOutcomePrior, CausalError> {
        let schema = table.schema();
        let response = table.float64_values(outcome)?;
        let mut columns: Vec<Option<Vec<f64>>> = Vec::with_capacity(self.coefficient_names.len());
        for name in self.coefficient_names.iter() {
            if &**name == "intercept" {
                columns.push(None);
                continue;
            }
            let column = name.strip_prefix("coef_").and_then(|column| schema.id_of(column).ok());
            let Some(id) = column else {
                return Err(crate::compile_reason!(
                    "invalid_argument",
                    "the coefficient `{}` names no column of the study's table; name each \
                     adjusted column `coef_<column>`",
                    name
                ));
            };
            columns.push(Some(table.float64_values(id)?));
        }
        let rows: Vec<usize> = (0..response.len())
            .filter(|&row| {
                response[row].is_finite()
                    && columns.iter().flatten().all(|column| column[row].is_finite())
            })
            .collect();
        let width = columns.len();
        let mut design = Vec::with_capacity(rows.len() * width);
        let mut y = Vec::with_capacity(rows.len());
        for &row in &rows {
            for column in &columns {
                design.push(column.as_ref().map_or(1.0, |values| values[row]));
            }
            y.push(response[row]);
        }
        let count = rows.len();
        let functional: Vec<f64> = (0..width)
            .map(|j| (0..count).map(|i| design[i * width + j]).sum::<f64>() / count.max(1) as f64)
            .collect();
        let target = LinearTarget {
            x: &design,
            y: &y,
            n_rows: count,
            n_cols: width,
            functional: &functional,
        };
        let (composed, report) =
            select_linear_prior(&self.sources, &self.baseline, target, &self.options)?;
        let composed = composed.with_coefficient_names(self.coefficient_names.to_vec())?;
        let source = outcome_prior_source_from_composed(&composed)?;
        Ok(ResolvedOutcomePrior { source, composed, report })
    }
}

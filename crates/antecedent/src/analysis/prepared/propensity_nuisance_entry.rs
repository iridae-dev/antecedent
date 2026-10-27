//! Prepared-lifecycle entry of the treatment-propensity nuisance product.
//!
//! A prepared tabular average-effect study knows its treatment and its certified adjustment set.
//! The product is the Laplace posterior of the logistic treatment model on exactly those
//! columns, under a stated Gaussian coefficient prior. It is a posterior of a nuisance: it is
//! never an effect result, and a weighting-effect posterior built from it is refused with the
//! posteriors that do answer that effect. The retained product embeds the design it was fitted
//! on, so a loader refits it and refuses any change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{PreparedModality, PreparedStudy};
use crate::analysis::array_fits::{
    ArrayFit, ArrayFitRequestWire, PenalizedScoreWire, PropensityDesignWire,
    PropensityPosteriorWire, PropensityScorePosteriorWire, TemporalScoreModel,
    TemporalTreatmentScoreSpec, TreatmentScoreSpec,
};
use crate::analysis::builder::DataInput;
use crate::error::CausalError;
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_data::{MissingPolicy, TableView, TabularData, first_missing_cell};

impl PreparedStudy {
    /// The posterior of the treatment propensity on this study's adjustment set.
    ///
    /// Fitting uses the analysis mask and complete cases of treatment, outcome and the
    /// certified adjustment set. The outcome selects the study cohort but is not a predictor.
    /// `summary_rows` are physical input row positions, not positions in the selected cohort:
    /// they may name masked or held-out rows, provided all adjustment predictors are observed.
    /// Their treatment and outcome need not be observed. Sampling weights and nonunit storage
    /// weights are refused because this product does not implement a weighted likelihood.
    ///
    /// # Errors
    /// A handle that is not a prepared tabular study, a schema change, a query that is not a
    /// binary average effect with a retained identification, a treatment not coded 0 or 1, a
    /// summary row outside the data, or a refused Laplace fit.
    pub fn treatment_score_posterior(
        &self,
        data: &TabularData,
        spec: &TreatmentScoreSpec,
    ) -> Result<ArrayFit, CausalError> {
        self.treatment_score_on(data, spec)
    }

    /// [`Self::treatment_score_posterior`] on the data the study was prepared with.
    ///
    /// # Errors
    /// As [`Self::treatment_score_posterior`]; a handle that is not tabular is refused.
    pub fn treatment_score_posterior_retained(
        &self,
        spec: &TreatmentScoreSpec,
    ) -> Result<ArrayFit, CausalError> {
        match &self.analysis.data {
            DataInput::Tabular(data) => self.treatment_score_on(data, spec),
            _ => Err(CausalError::Compile {
                message: "a treatment score product needs a prepared tabular study".into(),
            }),
        }
    }

    fn treatment_score_on(
        &self,
        data: &TabularData,
        spec: &TreatmentScoreSpec,
    ) -> Result<ArrayFit, CausalError> {
        if self.modality != PreparedModality::Tabular {
            return Err(CausalError::Compile {
                message: "a treatment score product needs a prepared tabular study".into(),
            });
        }
        self.ensure_schema_compatible(data)?;
        let CausalQuery::AverageEffect(query) = &self.analysis.query else {
            return Err(CausalError::Unsupported {
                message: "a treatment score product needs a prepared AverageEffect study",
            });
        };
        let Some(cache) = self.analysis.static_cache().map(|cache| &**cache) else {
            return Err(CausalError::Unsupported {
                message: "a treatment score product needs the retained adjustment set of a \
                          static average effect",
            });
        };
        if self
            .analysis
            .declared
            .estimator_spec
            .as_deref()
            .is_some_and(|estimator| estimator.sampling_weights_calibration_label().is_some())
            || data.storage().weights().is_some_and(|weights| weights.iter().any(|&w| w != 1.0))
        {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "the treatment score posterior does not implement a sampling-weighted \
                     likelihood; next attempt: prepare an explicitly unweighted population",
                ),
            ));
        }
        let predictors: Vec<_> = cache
            .estimand
            .adjustment_set
            .iter()
            .copied()
            .filter(|&id| id != query.treatment && id != query.outcome)
            .collect();
        let mut variables = vec![query.treatment, query.outcome];
        variables.extend_from_slice(&predictors);
        use crate::analysis::missing_data::{MissingRoute, effective_policy};
        let (policy, _) =
            effective_policy(MissingRoute::Static, self.analysis.declared.missing.as_ref());
        match policy {
            MissingPolicy::CompleteCase => {}
            MissingPolicy::ErrorOnMissing => {
                if let Some(cell) = first_missing_cell(data, &variables)? {
                    return Err(CausalError::Estimate(
                        antecedent_estimate::missing_policy::policy_required(
                            &cell,
                            "the treatment score posterior",
                            &[MissingPolicy::CompleteCase],
                        ),
                    ));
                }
            }
            _ => {
                return Err(CausalError::Estimate(
                    antecedent_estimate::missing_policy::policy_not_applicable(
                        policy,
                        "the treatment score posterior",
                        "this nuisance product fits complete cases, not imputed or weighted fits",
                        &[MissingPolicy::ErrorOnMissing, MissingPolicy::CompleteCase],
                    ),
                ));
            }
        }
        let keep = data.complete_case_mask(&variables)?;
        let nrows = keep.iter().filter(|&&selected| selected).count();
        let treatment_column = data.float64_cow(query.treatment)?;
        let treatment = treatment_column
            .iter()
            .zip(&keep)
            .filter_map(|(&value, &selected)| selected.then_some(value))
            .collect();
        let mut x_colmajor = vec![1.0_f64; nrows];
        let mut rows = vec![vec![1.0]; spec.summary_rows.len()];
        for &row in &spec.summary_rows {
            if row >= data.row_count() {
                return Err(CausalError::Unsupported {
                    message: "a summary row is outside the prepared data",
                });
            }
        }
        for variable in predictors {
            let column = data.float64_cow(variable)?;
            x_colmajor.extend(
                column
                    .iter()
                    .zip(&keep)
                    .filter_map(|(&value, &selected)| selected.then_some(value)),
            );
            let validity = data.column(variable)?.validity();
            for (&row, summary) in spec.summary_rows.iter().zip(&mut rows) {
                if !validity.is_valid(row) || !column[row].is_finite() {
                    return Err(CausalError::Unsupported {
                        message: "a summary row has a missing or non-finite adjustment predictor",
                    });
                }
                summary.push(column[row]);
            }
        }
        let request = PropensityPosteriorWire {
            x_colmajor,
            treatment,
            nrows,
            prior_sd: spec.prior_sd,
            draws: spec.draws,
            seed: spec.seed,
            rows,
            clip: spec.clip,
        };
        Ok(ArrayFit::fit(ArrayFitRequestWire::PropensityPosterior(request))?)
    }
}

impl PreparedStudy {
    /// The temporal treatment propensity on the independent-subject rows of this panel study.
    ///
    /// Each unit contributes its earliest row with a full lag window, so rows are independent
    /// subjects; the treatment of that row is regressed on the identified adjustment lags. The
    /// product is a nuisance score, never an effect, and it is unlicensed with calibration
    /// unmeasured.
    ///
    /// # Errors
    /// A single dependent series (`route_not_supported`, naming a panel of independent subjects
    /// as the next attempt), a handle that is not a prepared panel temporal-effect study, an
    /// empty adjustment set (no lag to condition on), a treatment not coded 0 or 1, or a refused
    /// fit.
    pub fn temporal_treatment_score(
        &self,
        spec: &TemporalTreatmentScoreSpec,
    ) -> Result<ArrayFit, CausalError> {
        self.temporal_treatment_score_with_context(spec, &ExecutionContext::production(0, 1))
    }

    /// Fit temporal nuisance scores under the caller's execution authority.
    ///
    /// # Errors
    /// As [`Self::temporal_treatment_score`], including caller cancellation or admission refusal.
    pub fn temporal_treatment_score_with_context(
        &self,
        spec: &TemporalTreatmentScoreSpec,
        ctx: &ExecutionContext,
    ) -> Result<ArrayFit, CausalError> {
        self.temporal_treatment_score_owned_with_context(spec.clone(), ctx)
    }

    /// Transfer owned histories once into the independently replayable nuisance request.
    ///
    /// # Errors
    /// As [`Self::temporal_treatment_score_with_context`].
    pub fn temporal_treatment_score_owned_with_context(
        &self,
        spec: TemporalTreatmentScoreSpec,
        ctx: &ExecutionContext,
    ) -> Result<ArrayFit, CausalError> {
        if self.modality == PreparedModality::Series {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "a single dependent series is not an independent-trajectory propensity sample, so \
                 no temporal treatment score is fitted as if its rows were iid. Next attempt: \
                 prepare a panel of independent subjects (one trajectory per subject)"
            ));
        }
        let (DataInput::Panel(panel), CausalQuery::TemporalEffect(query)) =
            (&self.analysis.data, &self.analysis.query)
        else {
            return Err(CausalError::Compile {
                message: "a temporal treatment score needs a prepared panel temporal-effect study"
                    .into(),
            });
        };
        let graph = self.plan.temporal_graph().ok_or(CausalError::Compile {
            message: "the prepared panel plan has no resolved temporal graph".into(),
        })?;
        let mut subjects = self.analysis.panel_subject_columns(panel, graph, query, ctx)?;
        if subjects.columns.len() <= 2 {
            return Err(CausalError::Unsupported {
                message: "a temporal treatment score needs at least one identified adjustment lag \
                          to condition on; the adjustment set of this pulse is empty",
            });
        }
        let treatment = std::mem::take(&mut subjects.columns[1]);
        let count = subjects.nrows.checked_mul(subjects.columns.len() - 1).ok_or_else(|| {
            CausalError::Resource { message: "temporal nuisance design size overflow".into() }
        })?;
        let mut x_colmajor = Vec::with_capacity(count);
        x_colmajor.resize(subjects.nrows, 1.0_f64);
        for column in subjects.columns.into_iter().skip(2) {
            x_colmajor.extend(column);
        }
        let design = PropensityDesignWire {
            x_colmajor,
            treatment,
            nrows: subjects.nrows,
            histories: spec.histories,
        };
        let request = match spec.model {
            TemporalScoreModel::Penalized { penalty } => {
                ArrayFitRequestWire::TemporalPropensity(PenalizedScoreWire { design, penalty })
            }
            TemporalScoreModel::Posterior { intercept_sd, slope_sd, draws, seed } => {
                ArrayFitRequestWire::TemporalPropensityPosterior(PropensityScorePosteriorWire {
                    design,
                    intercept_sd,
                    slope_sd,
                    draws,
                    seed,
                })
            }
        };
        antecedent_core::numerical_admission::with_context(ctx, || Ok(ArrayFit::fit(request)?))
    }
}

//! Result section of the associational route.
//!
//! The description is computed by the compiled `Association` study (`Study.compile` and
//! `prepare_association`); this module only carries the quantities and the evidence behind them
//! to Python. An association is never the effect of an intervention and ranks no action: its
//! standard errors, spreads and posteriors are unlicensed and their calibration is unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::association::AssociationPayload;
use pyo3::prelude::*;

/// One reported quantity: a stratum value, a coefficient, a correlation.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct AssociationTermSection {
    pub label: String,
    pub scale: String,
    pub estimate: f64,
    pub standard_error: Option<f64>,
    pub standard_error_method: Option<String>,
    pub spread_lower: Option<f64>,
    pub spread_upper: Option<f64>,
    pub posterior_mean: Option<f64>,
    pub posterior_sd: Option<f64>,
    pub posterior_lower: Option<f64>,
    pub posterior_upper: Option<f64>,
    pub rows: usize,
}

/// Resampling behind the standard errors and spreads of a frequentist result.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct AssociationResamplingSection {
    pub method: String,
    pub unit: String,
    pub replicates_requested: u32,
    pub replicates_ok: u32,
    pub block_length: Option<usize>,
    pub spread_level: f64,
}

/// Posterior behind the summaries of a Bayesian result.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct AssociationPosteriorSection {
    pub method: String,
    pub draws: usize,
    pub level: f64,
}

/// Out-of-fold scores of a predictive model.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct AssociationPredictionSection {
    pub folds: u32,
    pub fold_design: String,
    pub cases: usize,
    pub score: String,
    pub mean_score: f64,
    pub baseline_score: f64,
    pub mean_crps: Option<f64>,
    pub calibration: String,
}

/// A noncausal association of the observed data.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct AssociationSection {
    pub analysis: String,
    pub rule: String,
    pub requested: String,
    pub outcome_kind: String,
    pub modality: String,
    pub terms: Vec<AssociationTermSection>,
    pub rows: usize,
    pub rows_dropped: usize,
    pub resampling: Option<AssociationResamplingSection>,
    pub posterior: Option<AssociationPosteriorSection>,
    pub prediction: Option<AssociationPredictionSection>,
    pub assumptions: Vec<String>,
    pub refused_after: Option<String>,
    pub uncertainty: String,
    pub calibration: String,
    pub licensed: bool,
    pub ranks_actions: bool,
    pub is_intervention_effect: bool,
}

impl From<&AssociationPayload> for AssociationSection {
    fn from(payload: &AssociationPayload) -> Self {
        let estimate = &payload.estimate;
        let calibration = antecedent::analysis::association::ASSOCIATION_CALIBRATION.to_owned();
        Self {
            analysis: estimate.selection.analysis.as_str().to_owned(),
            rule: estimate.selection.rule.to_owned(),
            requested: estimate.selection.requested.to_owned(),
            outcome_kind: estimate.selection.outcome_kind.as_str().to_owned(),
            modality: estimate.selection.modality.as_str().to_owned(),
            terms: estimate
                .terms
                .iter()
                .map(|term| AssociationTermSection {
                    label: term.label.clone(),
                    scale: term.scale.to_owned(),
                    estimate: term.estimate,
                    standard_error: term.standard_error,
                    standard_error_method: term.standard_error_method.map(str::to_owned),
                    spread_lower: term.spread.map(|(lower, _)| lower),
                    spread_upper: term.spread.map(|(_, upper)| upper),
                    posterior_mean: term.posterior.map(|p| p.mean),
                    posterior_sd: term.posterior.map(|p| p.sd),
                    posterior_lower: term.posterior.map(|p| p.lower),
                    posterior_upper: term.posterior.map(|p| p.upper),
                    rows: term.rows,
                })
                .collect(),
            rows: estimate.rows,
            rows_dropped: estimate.rows_dropped,
            resampling: estimate.resampling.map(|record| AssociationResamplingSection {
                method: record.method.to_owned(),
                unit: record.unit.to_owned(),
                replicates_requested: record.replicates_requested,
                replicates_ok: record.replicates_ok,
                block_length: record.block_length,
                spread_level: antecedent_estimate::SPREAD_LEVEL,
            }),
            posterior: estimate.posterior.map(|record| AssociationPosteriorSection {
                method: record.method.to_owned(),
                draws: record.draws,
                level: antecedent_estimate::SPREAD_LEVEL,
            }),
            prediction: payload.scores.as_ref().zip(estimate.prediction.as_ref()).map(
                |(scores, fit)| AssociationPredictionSection {
                    folds: fit.folds,
                    fold_design: fit.fold_design.to_owned(),
                    cases: scores.cases,
                    score: scores.score.to_owned(),
                    mean_score: scores.mean_score,
                    baseline_score: scores.baseline_score,
                    mean_crps: scores.mean_crps,
                    calibration: calibration.clone(),
                },
            ),
            assumptions: estimate.assumptions.iter().map(|premise| (*premise).to_owned()).collect(),
            refused_after: payload.refused_after.as_deref().map(str::to_owned),
            uncertainty: payload.uncertainty_label().to_owned(),
            calibration,
            licensed: false,
            ranks_actions: false,
            is_intervention_effect: false,
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AssociationTermSection>()?;
    m.add_class::<AssociationResamplingSection>()?;
    m.add_class::<AssociationPosteriorSection>()?;
    m.add_class::<AssociationPredictionSection>()?;
    m.add_class::<AssociationSection>()
}

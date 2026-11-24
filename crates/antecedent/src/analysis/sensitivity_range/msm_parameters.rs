//! The period parameters of the marginal structural model a longitudinal regime study fitted.
//!
//! The longitudinal regime route with [`LongitudinalRegimeMethod::MarginalStructuralModel`]
//! fits the additive model `outcome ~ 1 + A_1 + ... + A_T` on the observed subjects by weighted
//! least squares with the stabilized sequential weights its query declares
//! (`prod_t num_t(A_t) / (p_t(A_t) c_t)`, from the stabilizing numerators, the known treatment
//! probabilities and the censoring probabilities). This module rebuilds that design and those
//! weights from the study's own query and retained outcome column, in the route's arithmetic,
//! and reads the parameters' sensitivity from them:
//!
//! * [`SensitivityRangeRequest::MsmParameters`]: [`msm_parameter_fit`] (weights held fixed) and
//!   [`component_ovb_range`] under sequential exchangeability;
//! * [`SensitivityRangeRequest::MsmParameterEValue`]: [`component_evalue_range`] on each
//!   parameter over the weighted residual standard deviation `sqrt(sum w r^2 / sum w)` of the
//!   same fit;
//! * [`SensitivityRangeRequest::MsmParametersRefittingWeights`]:
//!   [`msm_parameter_fit_refitting_weights`] with the declared numerator and denominator
//!   covariate columns of the table, and [`component_ovb_range`] on the refitted summaries. The
//!   refitted weight models cover treatment only; censoring is not modelled in the refit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, LongitudinalRegimeMethod, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_validate::target_native::{RefutationTarget, TargetFamily};
use antecedent_validate::{
    ComponentModel, EffectInterval, EffectScale, LinearAdjustmentSummary, MsmWeightPeriod,
    component_evalue_range, component_ovb_range, msm_parameter_fit,
    msm_parameter_fit_refitting_weights,
};

use super::super::prepared::PreparedStudy;
use super::super::target_validation::result_quadrant;
use super::{SensitivityRangeOutcome, SensitivityRangeRequest};
use crate::error::CausalError;
use crate::result::StudyResult;

/// The weighted design of the route's marginal structural model, on its observed subjects.
struct MsmDesign {
    /// Observed subjects' rows in the table.
    rows: Vec<usize>,
    /// Terminal outcome of each observed subject.
    outcome: Vec<f64>,
    /// One 0/1 treatment column per period.
    exposures: Vec<Vec<f64>>,
    /// Stabilized sequential weight of each observed subject.
    weights: Vec<f64>,
}

impl PreparedStudy {
    /// The marginal structural model analyses, or `None` when `request` is another analysis.
    pub(super) fn msm_parameter_range(
        &self,
        data: &TabularData,
        result: &StudyResult,
        program: [u8; 32],
        request: &SensitivityRangeRequest,
    ) -> Result<Option<SensitivityRangeOutcome>, CausalError> {
        if !matches!(
            request,
            SensitivityRangeRequest::MsmParameters { .. }
                | SensitivityRangeRequest::MsmParameterEValue
                | SensitivityRangeRequest::MsmParametersRefittingWeights(_)
        ) {
            return Ok(None);
        }
        let design = self.msm_design(data)?;
        let periods = design.exposures.len();
        let labels: Vec<String> = (0..periods).map(|t| format!("period[{t}]")).collect();
        let family = if periods == 1 {
            TargetFamily::ScalarAverageEffect
        } else {
            TargetFamily::JointContrast
        };
        let target = RefutationTarget::new(family, result_quadrant(result), labels, program)
            .map_err(CausalError::from)?;
        let model = ComponentModel::MarginalStructural;
        let outcome = match request {
            SensitivityRangeRequest::MsmParameters { q, alpha, r2yz_extreme } => {
                let summaries =
                    msm_parameter_fit(&design.outcome, &design.exposures, &design.weights)?;
                let range =
                    component_ovb_range(target, &summaries, *q, *alpha, *r2yz_extreme, model)?;
                SensitivityRangeOutcome::MsmParameters { summaries, weights: design.weights, range }
            }
            SensitivityRangeRequest::MsmParameterEValue => {
                let summaries =
                    msm_parameter_fit(&design.outcome, &design.exposures, &design.weights)?;
                let residual_sd = weighted_residual_sd(&design, &summaries)?;
                let intervals: Vec<EffectInterval> = summaries
                    .iter()
                    .map(|s| EffectInterval::Standardized {
                        estimate: s.estimate / residual_sd,
                        standard_error: Some(s.standard_error / residual_sd),
                    })
                    .collect();
                let (bounds, range) = component_evalue_range(
                    target,
                    &intervals,
                    EffectScale::StandardizedMeanDifference,
                    model,
                )?;
                SensitivityRangeOutcome::MsmParameterEValue {
                    summaries,
                    residual_sd,
                    bounds,
                    range,
                }
            }
            SensitivityRangeRequest::MsmParametersRefittingWeights(refit) => {
                if refit.numerator.len() != periods || refit.denominator.len() != periods {
                    return Err(CausalError::Unsupported {
                        message: "the weight refit declares one numerator and one denominator \
                                  covariate list per treatment period",
                    });
                }
                let column = |id: VariableId| -> Result<Vec<f64>, CausalError> {
                    let values = data.float64_values(id).map_err(|_| CausalError::Unsupported {
                        message: "a weight-model covariate is not a numeric column",
                    })?;
                    let selected: Vec<f64> = design.rows.iter().map(|&r| values[r]).collect();
                    if selected.iter().any(|v| !v.is_finite()) {
                        return Err(CausalError::Unsupported {
                            message: "a weight-model covariate is missing for an observed subject",
                        });
                    }
                    Ok(selected)
                };
                let mut weight_periods = Vec::with_capacity(periods);
                for (t, treatment) in design.exposures.iter().enumerate() {
                    weight_periods.push(MsmWeightPeriod {
                        treatment: treatment.clone(),
                        numerator: refit.numerator[t]
                            .iter()
                            .map(|&id| column(id))
                            .collect::<Result<_, _>>()?,
                        denominator: refit.denominator[t]
                            .iter()
                            .map(|&id| column(id))
                            .collect::<Result<_, _>>()?,
                    });
                }
                let fit = msm_parameter_fit_refitting_weights(
                    &design.outcome,
                    &design.exposures,
                    &weight_periods,
                    refit.replicates,
                    refit.seed,
                )?;
                let range = component_ovb_range(
                    target,
                    &fit.refit,
                    refit.q,
                    refit.alpha,
                    refit.r2yz_extreme,
                    model,
                )?;
                SensitivityRangeOutcome::MsmParametersRefittingWeights { fit, range }
            }
            _ => return Ok(None),
        };
        Ok(Some(outcome))
    }

    /// The route's weighted design, rebuilt from the study's query and retained outcome column.
    fn msm_design(&self, data: &TabularData) -> Result<MsmDesign, CausalError> {
        let CausalQuery::LongitudinalRegime(query) = &self.study().query else {
            return Err(CausalError::Unsupported {
                message: "the marginal structural model parameters are read from a longitudinal \
                          regime study; next attempt: a LongitudinalRegimeQuery with the marginal \
                          structural model method",
            });
        };
        if !matches!(query.method, LongitudinalRegimeMethod::MarginalStructuralModel) {
            return Err(CausalError::Unsupported {
                message: "the longitudinal regime study was not fitted with the marginal \
                          structural model method",
            });
        }
        let numerator = query.stabilizing_numerator_probabilities.as_deref().ok_or(
            CausalError::Unsupported {
                message: "marginal structural model requires stabilizing numerators",
            },
        )?;
        let periods = query.periods;
        let y = data.float64_values(query.outcome).map_err(|_| CausalError::Unsupported {
            message: "the longitudinal endpoint outcome is not a numeric column",
        })?;
        let n = y.len();
        let a = &query.treatment_history;
        let p = &query.treatment_probabilities;
        let censor = &query.censoring_probabilities;
        let observed = &query.outcome_observed;
        let cells = n.checked_mul(periods);
        if periods == 0
            || cells != Some(a.len())
            || cells != Some(p.len())
            || cells != Some(censor.len())
            || observed.len() != n
            || numerator.len() != periods
        {
            return Err(CausalError::Unsupported {
                message: "the longitudinal histories do not match the subjects and periods",
            });
        }
        let mut design = MsmDesign {
            rows: Vec::new(),
            outcome: Vec::new(),
            exposures: vec![Vec::new(); periods],
            weights: Vec::new(),
        };
        for i in 0..n {
            if !observed[i] {
                continue;
            }
            if !y[i].is_finite() {
                return Err(CausalError::Unsupported {
                    message: "observed terminal outcomes must be finite",
                });
            }
            // The route's stabilized weight, multiplied in the same order.
            let mut weight = 1.0;
            for t in 0..periods {
                let j = i * periods + t;
                let top = if a[j] { numerator[t] } else { 1.0 - numerator[t] };
                let bottom = if a[j] { p[j] } else { 1.0 - p[j] };
                weight *= top / (bottom * censor[j]);
                design.exposures[t].push(f64::from(a[j]));
            }
            if !weight.is_finite() {
                return Err(CausalError::Unsupported {
                    message: "the stabilized sequential weight is not finite",
                });
            }
            design.rows.push(i);
            design.outcome.push(y[i]);
            design.weights.push(weight);
        }
        Ok(design)
    }
}

/// `sqrt(sum w r^2 / sum w)` of the weighted fit, the intercept from the weighted normal
/// equation `sum w (y - b0 - sum b_j x_j) = 0`.
fn weighted_residual_sd(
    design: &MsmDesign,
    summaries: &[LinearAdjustmentSummary],
) -> Result<f64, CausalError> {
    let total: f64 = design.weights.iter().sum();
    let weighted = |column: &[f64]| -> f64 {
        column.iter().zip(&design.weights).map(|(v, w)| v * w).sum::<f64>() / total
    };
    let intercept = weighted(&design.outcome)
        - design
            .exposures
            .iter()
            .zip(summaries)
            .map(|(x, s)| s.estimate * weighted(x))
            .sum::<f64>();
    let mut sum = 0.0;
    for (i, &w) in design.weights.iter().enumerate() {
        let fitted = intercept
            + design.exposures.iter().zip(summaries).map(|(x, s)| s.estimate * x[i]).sum::<f64>();
        let residual = design.outcome[i] - fitted;
        sum += w * residual * residual;
    }
    let sd = (sum / total).sqrt();
    if !(sd.is_finite() && sd > 0.0) {
        return Err(CausalError::Unsupported {
            message: "the weighted marginal structural model leaves no residual variation to \
                      standardize by",
        });
    }
    Ok(sd)
}

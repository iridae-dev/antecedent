//! Evaluation and support for temporal derivative responses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ExecutionContext, IdentificationResult, IdentifiedEstimand, InferenceMode, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, SlopeSource,
    TableView, TemporalIndexer, TimeSeriesData, VariableId, point_uncertainty,
};

/// The fitted slope of a point derivative on the requested scale, evaluated at
/// the treatment point `at`.
fn scaled_point_slope(
    scale: antecedent_core::DerivativeScale,
    at: f64,
    (intercept, slope): (f64, f64),
) -> Result<f64, CausalError> {
    use antecedent_core::DerivativeScale;
    let mean = intercept + slope * at;
    match scale {
        DerivativeScale::Identity => Ok(slope),
        DerivativeScale::LogTreatment if at > 0.0 => Ok(at * slope),
        DerivativeScale::LogOutcome if mean.is_finite() && mean != 0.0 => Ok(slope / mean),
        DerivativeScale::LogLog if at > 0.0 && mean.is_finite() && mean != 0.0 => {
            Ok(at * slope / mean)
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal derivative scale needs a positive treatment point and a non-zero fitted mean",
        }),
    }
}

/// The directional derivative of each outcome along `direction`, from the
/// fitted treatment slopes.
fn directional_derivative_value(
    fit: &impl Fn(VariableId, &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    outcomes: &[VariableId],
    treatments: &[VariableId],
    direction: &[f64],
) -> Result<ResponseValue, CausalError> {
    let mut values = Vec::with_capacity(outcomes.len());
    for &outcome in outcomes {
        let (_, slopes) = fit(outcome, treatments)?;
        values.push(slopes.iter().zip(direction.iter()).map(|(slope, step)| slope * step).sum());
    }
    Ok(ResponseValue::Vector(std::sync::Arc::from(values)))
}

/// The Jacobian of the outcomes with respect to the treatments at `at`, with
/// the requested derivative scale applied to each fitted slope.
fn jacobian_derivative_value(
    fit: &impl Fn(VariableId, &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    (outcomes, treatments): (&[VariableId], &[VariableId]),
    at: &[f64],
    scale: antecedent_core::DerivativeScale,
) -> Result<ResponseValue, CausalError> {
    use antecedent_core::DerivativeScale;
    let mut values = Vec::with_capacity(outcomes.len() * treatments.len());
    for &outcome in outcomes {
        let (intercept, slopes) = fit(outcome, treatments)?;
        let mean = intercept
            + slopes.iter().zip(at.iter()).map(|(slope, point)| slope * point).sum::<f64>();
        for (&slope, &point) in slopes.iter().zip(at.iter()) {
            let scaled = match scale {
                DerivativeScale::Identity => slope,
                DerivativeScale::LogTreatment if point > 0.0 => point * slope,
                DerivativeScale::LogOutcome if mean.is_finite() && mean != 0.0 => slope / mean,
                DerivativeScale::LogLog if point > 0.0 && mean.is_finite() && mean != 0.0 => {
                    point * slope / mean
                }
                _ => {
                    return Err(CausalError::Unsupported {
                        message: "temporal Jacobian scale needs a positive treatment point and a non-zero fitted mean",
                    });
                }
            };
            values.push(scaled);
        }
    }
    Ok(ResponseValue::Jacobian {
        outcomes: outcomes.len(),
        treatments: treatments.len(),
        values: std::sync::Arc::from(values),
    })
}

pub(in crate::analysis::execute) fn temporal_linear_derivative_response(
    data: &TimeSeriesData,
    query: &ResponseQuery,
    identification: &IdentificationResult,
    source: &dyn SlopeSource,
) -> Result<CausalResponse, CausalError> {
    if query.target_population != antecedent_core::TargetPopulation::AllObserved {
        return Err(CausalError::Unsupported {
            message: "temporal derivative requires the all-observed population",
        });
    }
    let value = temporal_linear_derivative_value(&query.functional, source)?;
    let (support_treatments, support_at, average_over_observed) =
        derivative_support_inputs(&query.functional);
    let support =
        derivative_support_report(data, &support_treatments, &support_at, average_over_observed);
    Ok(CausalResponse {
        estimand: query.functional.clone(),
        identification_status: identification.status,
        estimate: ResponseIdentification::PointIdentified(value),
        uncertainty: ResponseUncertainty::None,
        support,
        assumptions: identification.required_assumptions.clone(),
        provenance_id: std::sync::Arc::from("estimate.temporal.linear_derivative"),
        horizon_identification: None,
        interaction_structurally_zero: false,
    })
}

/// One response per horizon of a point or average derivative, as one value per horizon in
/// horizon order with a pointwise band per horizon (horizons are not mixed, so the band is
/// marginal, not joint).
pub(in crate::analysis::execute) fn combine_horizon_derivatives(
    functional: &ResponseFunctional,
    responses: &[CausalResponse],
) -> Result<CausalResponse, CausalError> {
    let first = responses.first().ok_or_else(|| CausalError::Compile {
        message: "a multi-horizon derivative has no horizon responses".into(),
    })?;
    let mut values = Vec::with_capacity(responses.len());
    for response in responses {
        match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => {
                values.push(*value);
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "a multi-horizon derivative combines point-identified scalars",
                });
            }
        }
    }
    let mut lower = Vec::with_capacity(responses.len());
    let mut upper = Vec::with_capacity(responses.len());
    let mut level_interpretation = None;
    for response in responses {
        if let ResponseUncertainty::Scalar {
            lower: low, upper: high, level, interpretation, ..
        } = &response.uncertainty
        {
            lower.push(*low);
            upper.push(*high);
            level_interpretation = Some((*level, *interpretation));
        } else {
            lower.clear();
            break;
        }
    }
    let uncertainty = match level_interpretation {
        Some((level, interpretation)) if lower.len() == responses.len() => {
            ResponseUncertainty::PointwiseBand {
                level,
                lower: lower.into(),
                upper: upper.into(),
                interpretation,
                draws: None,
            }
        }
        _ => ResponseUncertainty::None,
    };
    let mut assumptions = first.assumptions.clone();
    let mut warnings = first.support.warnings.clone();
    for response in responses.iter().skip(1) {
        assumptions.extend_unique(&response.assumptions.entries);
        for warning in &response.support.warnings {
            if !warnings
                .iter()
                .any(|seen| seen.code == warning.code && seen.message == warning.message)
            {
                warnings.push(warning.clone());
            }
        }
    }
    let mut support = first.support.clone();
    support.warnings = warnings;
    support.point_status = None;
    Ok(CausalResponse {
        estimand: functional.clone(),
        identification_status: first.identification_status,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Vector(values.into())),
        uncertainty,
        support,
        assumptions,
        provenance_id: Arc::clone(&first.provenance_id),
        horizon_identification: None,
        interaction_structurally_zero: false,
    })
}

/// One `TemporalDag` derivative fit: the horizon's back-door estimand, its lag
/// indexer and proof, and the probe's treatment and outcome offsets.
#[derive(Clone, Copy)]
pub(in crate::analysis::execute) struct DagDerivativeFit<'a> {
    pub(in crate::analysis::execute) estimand: &'a IdentifiedEstimand,
    pub(in crate::analysis::execute) indexer: &'a TemporalIndexer,
    pub(in crate::analysis::execute) identification: &'a IdentificationResult,
    pub(in crate::analysis::execute) offsets: (i32, i32),
}

/// A `TemporalDag` derivative response with its whole-procedure uncertainty.
///
/// The point is the linear slope functional at the full-series fit (the
/// posterior-mean coefficients under Bayesian inference). Frequentist: every
/// outcome regression is refit on one shared circular-block bootstrap of
/// series times and the band (with a sup-t simultaneous band for several
/// coordinates) is attached when replicates were requested. Bayesian: the
/// functional is evaluated on joint draws of the tempered slope posterior and
/// its credible band is attached. Callers refuse a mapped prior before calling.
pub(in crate::analysis::execute) fn temporal_dag_derivative_response(
    data: &TimeSeriesData,
    query: &ResponseQuery,
    fit: DagDerivativeFit<'_>,
    settings: (&InferenceMode, u32),
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let no_path = fit.identification.diagnostics.iter().any(|diagnostic| {
        diagnostic.code.as_ref() == crate::analysis::temporal_admg_contrast::NO_DIRECTED_PATH_CODE
    });
    let mut response = temporal_dag_derivative_fit(data, query, fit, settings, ctx)?;
    if no_path {
        // The window holds no directed path from the treatment to the outcome: the do-effect
        // is exactly zero and the fitted slope is an association, so the point and its band
        // collapse to the structural zero (nothing is estimated, hence no sampling band).
        response.estimate = match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(_)) => {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(0.0))
            }
            other => other.clone(),
        };
        response.uncertainty = match &response.uncertainty {
            ResponseUncertainty::Scalar { level, interpretation, .. } => {
                ResponseUncertainty::Scalar {
                    standard_error: 0.0,
                    level: *level,
                    lower: 0.0,
                    upper: 0.0,
                    interpretation: *interpretation,
                    draws: None,
                }
            }
            other => other.clone(),
        };
        if let Some(note) = fit.identification.diagnostics.iter().find(|diagnostic| {
            diagnostic.code.as_ref()
                == crate::analysis::temporal_admg_contrast::NO_DIRECTED_PATH_CODE
        }) {
            response.support.warnings.push(note.clone());
        }
    }
    Ok(response)
}

fn temporal_dag_derivative_fit(
    data: &TimeSeriesData,
    query: &ResponseQuery,
    fit: DagDerivativeFit<'_>,
    (inference, bootstrap_replicates): (&InferenceMode, u32),
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let prior_scale = match inference {
        InferenceMode::Bayesian(cfg) => Some(cfg.prior_scale),
        InferenceMode::Frequentist => None,
    };
    let atom = point_uncertainty::SlopeAtom::new(
        data,
        fit.estimand,
        fit.indexer,
        fit.offsets,
        ctx.kernel_policy,
        prior_scale,
    );
    let mut response = temporal_linear_derivative_response(data, query, fit.identification, &atom)?;
    let center = match &response.estimate {
        ResponseIdentification::PointIdentified(value) => point_uncertainty::response_values(value),
        _ => None,
    }
    .ok_or_else(|| CausalError::Compile {
        message: "temporal derivative point has no numeric value".into(),
    })?;
    let frozen = [atom.freeze()];
    let functional = &query.functional;
    let evaluate = |_: usize, source: &dyn SlopeSource| {
        let value = temporal_linear_derivative_value(functional, source)?;
        point_uncertainty::response_values(&value).ok_or_else(|| CausalError::Compile {
            message: "temporal derivative replicate has no numeric value".into(),
        })
    };
    let structural_span = super::super::temporal_class_block_span(std::iter::once(fit.indexer));
    if let InferenceMode::Bayesian(cfg) = inference {
        let draws = point_uncertainty::point_posterior_draws(
            &frozen,
            point_uncertainty::PointPosteriorPlan {
                prior_scale: cfg.prior_scale,
                draws: cfg.n_draws,
                structural_span,
                stream_base: point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
            },
            ctx,
            evaluate,
        )?;
        point_uncertainty::attach_posterior_response_band(
            &mut response,
            &draws.atom_draws[0],
            &center,
        );
        response
            .support
            .warnings
            .push(point_uncertainty::posterior_diagnostic(&draws, "temporal linear derivative"));
        return Ok(response);
    }
    let replicates = point_uncertainty::point_block_replicates(
        &frozen,
        point_uncertainty::PointBlockPlan {
            replicates: bootstrap_replicates,
            stream_base: point_uncertainty::DERIVATIVE_BLOCK_STREAM,
            structural_span,
        },
        false,
        ctx,
        evaluate,
        |values| values.first().cloned(),
    )?;
    if let Some(replicates) = &replicates {
        point_uncertainty::attach_block_response_band(&mut response, replicates, &center);
    }
    response.support.warnings.extend(point_uncertainty::block_diagnostics(
        replicates.as_ref(),
        bootstrap_replicates,
        antecedent_estimate::CircularBlockFamily::SingleWindow,
        "every outcome regression of the derivative is refit on each replicate",
    ));
    Ok(response)
}

/// The treatments, evaluation point and averaging of a derivative's support report.
fn derivative_support_inputs(functional: &ResponseFunctional) -> (Vec<VariableId>, Vec<f64>, bool) {
    use antecedent_core::ResponseFunctional as F;
    match functional {
        F::AverageDerivative { treatment, .. } => (vec![*treatment], vec![0.0], true),
        F::PointDerivative { treatment, at, .. } => (vec![*treatment], vec![*at], false),
        F::DirectionalDerivative { treatments, at, .. } | F::Jacobian { treatments, at, .. } => {
            (treatments.to_vec(), at.to_vec(), false)
        }
        _ => (Vec::new(), vec![0.0], false),
    }
}

/// The linear derivative of `functional` from the slopes `source` fits: the
/// point routes call it on the full series, the shared block bootstrap on every
/// replicate, and the slope posterior on every draw.
pub(in crate::analysis::execute) fn temporal_linear_derivative_value(
    functional: &ResponseFunctional,
    source: &dyn SlopeSource,
) -> Result<ResponseValue, CausalError> {
    use antecedent_core::{DerivativeWeighting, ResponseFunctional as F};
    let fit = |outcome, treatments: &[VariableId]| source.slopes(outcome, treatments, &[]);
    Ok(match functional {
        F::AverageDerivative { outcome, treatment, weighting } => {
            if !matches!(weighting, DerivativeWeighting::Observed) {
                return Err(CausalError::Unsupported {
                    message: "temporal average derivative is supported for the observed weighting",
                });
            }
            let (_, slopes) = fit(*outcome, &[*treatment])?;
            ResponseValue::Scalar(slopes[0])
        }
        F::PointDerivative { outcome, treatment, at, order, scale } => {
            if *order != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal point derivative is the first derivative of the linear fit",
                });
            }
            let (intercept, slopes) = fit(*outcome, &[*treatment])?;
            ResponseValue::Scalar(scaled_point_slope(*scale, *at, (intercept, slopes[0]))?)
        }
        F::DirectionalDerivative { outcomes, treatments, at, direction } => {
            if at.len() != treatments.len() || direction.len() != treatments.len() {
                return Err(CausalError::Unsupported {
                    message: "temporal directional derivative point and direction follow the treatments",
                });
            }
            directional_derivative_value(&fit, outcomes, treatments, direction)?
        }
        F::Jacobian { outcomes, treatments, at, scale } => {
            if at.len() != treatments.len() {
                return Err(CausalError::Unsupported {
                    message: "temporal Jacobian is evaluated at one point per treatment",
                });
            }
            jacobian_derivative_value(&fit, (outcomes, treatments), at, *scale)?
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "temporal linear derivative does not estimate this functional",
            });
        }
    })
}

/// Average derivatives are supported on the observed treatment range. A requested
/// evaluation point is outside the data when it leaves that range.
fn derivative_support_report(
    data: &TimeSeriesData,
    treatments: &[VariableId],
    requested: &[f64],
    average_over_observed: bool,
) -> antecedent_core::SupportReport {
    use antecedent_core::{SupportRegion, SupportReport, SupportStatus};
    let range = |variable: VariableId| -> Option<(f64, f64)> {
        let values = data.float64_values(variable).ok()?;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in values {
            if value.is_finite() {
                min = min.min(value);
                max = max.max(value);
            }
        }
        (min.is_finite() && max.is_finite()).then_some((min, max))
    };
    if average_over_observed {
        let (minima, maxima, status) = match treatments.first().and_then(|id| range(*id)) {
            Some((min, max)) => (vec![min], vec![max], SupportStatus::Supported),
            None => (vec![f64::NAN], vec![f64::NAN], SupportStatus::OutsideEmpiricalSupport),
        };
        let mut warnings = Vec::new();
        if status != SupportStatus::Supported {
            warnings.push(Diagnostic::new(
                "estimate.temporal.linear_derivative.support",
                DiagnosticKind::Support,
                DiagnosticSeverity::Warning,
                "the average derivative has no finite observed treatment values",
            ));
        }
        return SupportReport {
            status,
            query_region: SupportRegion {
                minima: std::sync::Arc::from(minima),
                maxima: std::sync::Arc::from(maxima),
            },
            diagnostics: Vec::new(),
            warnings,
            point_status: None,
        };
    }
    let inside = treatments.iter().zip(requested).all(|(variable, point)| {
        range(*variable)
            .is_some_and(|(min, max)| point.is_finite() && *point >= min && *point <= max)
    });
    let status = if inside && treatments.len() == requested.len() && !requested.is_empty() {
        SupportStatus::Supported
    } else {
        SupportStatus::OutsideEmpiricalSupport
    };
    let mut warnings = Vec::new();
    if status != SupportStatus::Supported {
        warnings.push(Diagnostic::new(
            "estimate.temporal.linear_derivative.support",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "the requested evaluation point is outside the observed treatment series",
        ));
    }
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: std::sync::Arc::from(requested),
            maxima: std::sync::Arc::from(requested),
        },
        diagnostics: Vec::new(),
        warnings,
        point_status: None,
    }
}

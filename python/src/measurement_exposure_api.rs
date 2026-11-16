//! Python bridge for the exposure-error corrections.
//!
//! Python states the counts, probabilities, codings and declarations as typed values and passes
//! them as JSON. Every check, correction and refusal is Rust's
//! (`antecedent::estimate::measurement::exposure_routes`): a refusal comes back as structured
//! JSON for the Python layer to raise as its own typed exception, and a malformed declaration
//! that never reached the checks is an input error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::estimate::measurement::{
    DeclaredEstimate, MeasurementError, SoftExposureCombination,
    correct_detection_dependent_prevalence, correct_exposure_misclassification,
    correct_soft_exposure_effect, fit_repeat_codings, soft_exposure_reliability_report,
};
use antecedent::inference::measurement_error::{
    BinaryErrorRates, DetectionAdjusted, DetectionAnchor, DetectionData, DetectionDeclaration,
    DetectionDirection, DetectionFunction, ExposureErrorStructure, ExposureOutcomeCounts,
    FalsePositiveRate, LatentClassConstraints, LatentClassData, LatentClassSettings,
    SoftExposureData, exposure_rates_from_validation,
};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    index, list, member, object, opt_flag, parse, present, real, reals, text, unsigned,
};
use crate::transport_common::execution_context;

type Outcome = (Option<String>, Option<String>);

fn refusal(error: &MeasurementError) -> String {
    crate::bridge::compact_refusal_json(
        error.code,
        "measurement_model",
        error.detail,
        &error.message,
        None,
    )
}

fn finish(result: Result<Json, MeasurementError>) -> Outcome {
    match result {
        Ok(value) => (Some(value.to_string()), None),
        Err(error) => (None, Some(refusal(&error))),
    }
}

fn level_of(map: &Map<String, Json>) -> PyResult<f64> {
    match present(map, "level") {
        None => Ok(0.95),
        Some(_) => real(map, "level"),
    }
}

fn estimate_json(estimate: &DeclaredEstimate, level: f64) -> Result<Json, MeasurementError> {
    let interval = estimate.wald_interval(level)?;
    Ok(json!({
        "functional_id": estimate.functional_id,
        "point": estimate.point,
        "naive_point": estimate.naive_point,
        "standard_error": estimate.standard_error,
        "variance_components": estimate
            .variance_components
            .iter()
            .map(|c| json!({"name": c.name, "variance": c.variance}))
            .collect::<Vec<_>>(),
        "rows": estimate.rows,
        "data_identity": estimate.data_identity.to_hex(),
        "parameter_uncertainty": estimate.parameter_uncertainty.as_str(),
        "inference": estimate.inference.as_str(),
        "interval": {"level": level, "lower": interval.0, "upper": interval.1, "calibration": "not_measured"},
    }))
}

fn rates_of(map: &Map<String, Json>) -> PyResult<BinaryErrorRates> {
    BinaryErrorRates::new(
        real(map, "sensitivity")?,
        real(map, "specificity")?,
        real(map, "sensitivity_variance")?,
        real(map, "specificity_variance")?,
    )
    .map_err(|error| crate::value_err(error.to_string()))
}

fn quad(map: &Map<String, Json>, key: &str) -> PyResult<(u64, u64, u64, u64)> {
    let items = list(map, key)?;
    let counts: Vec<u64> = items
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| crate::value_err(format!("`{key}` holds a non-count"))))
        .collect::<PyResult<_>>()?;
    match counts.as_slice() {
        [a, b, c, d] => Ok((*a, *b, *c, *d)),
        _ => Err(crate::value_err(format!(
            "`{key}` must be (true_positive, false_negative, true_negative, false_positive)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Exposure misclassification
// ---------------------------------------------------------------------------

fn structure_of(map: &Map<String, Json>) -> PyResult<ExposureErrorStructure> {
    let kind = text(map, "kind")?;
    match kind.as_str() {
        "shared" => {
            Ok(ExposureErrorStructure::Shared(rates_of(object(member(map, "rates")?, "rates")?)?))
        }
        "shared_validation" => {
            let c = quad(map, "counts")?;
            BinaryErrorRates::from_validation_counts(c.0, c.1, c.2, c.3)
                .map(ExposureErrorStructure::Shared)
                .map_err(|error| crate::value_err(error.to_string()))
        }
        "by_outcome" => Ok(ExposureErrorStructure::ByOutcome {
            cases: rates_of(object(member(map, "cases")?, "cases")?)?,
            controls: rates_of(object(member(map, "controls")?, "controls")?)?,
        }),
        "by_outcome_validation" => {
            exposure_rates_from_validation(quad(map, "controls")?, quad(map, "cases")?)
                .map_err(|error| crate::value_err(error.to_string()))
        }
        other => Err(crate::value_err(format!("unknown exposure error structure `{other}`"))),
    }
}

/// Correct a binary exposure's misclassification, including error that differs by outcome.
#[pyfunction]
fn measurement_exposure_misclassification(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "exposure misclassification")?;
    let map = object(&value, "exposure misclassification")?;
    let counts_map = object(member(map, "counts")?, "counts")?;
    let counts = ExposureOutcomeCounts {
        exposed_cases: unsigned(counts_map, "exposed_cases")?,
        cases: unsigned(counts_map, "cases")?,
        exposed_controls: unsigned(counts_map, "exposed_controls")?,
        controls: unsigned(counts_map, "controls")?,
    };
    let structure = structure_of(object(member(map, "structure")?, "structure")?)?;
    let min_youden = real(map, "min_youden")?;
    let level = level_of(map)?;
    let ctx = execution_context(0, None, None);
    Ok(finish(correct_exposure_misclassification(&counts, &structure, min_youden, &ctx).and_then(
        |fit| {
            let a = &fit.association;
            Ok(json!({
                "observed_prevalence_cases": a.observed_prevalence_cases,
                "observed_prevalence_controls": a.observed_prevalence_controls,
                "prevalence_cases": a.prevalence_cases,
                "prevalence_controls": a.prevalence_controls,
                "naive_log_odds_ratio": a.naive_log_odds_ratio,
                "log_odds_ratio": a.log_odds_ratio,
                "odds_ratio": a.odds_ratio,
                "variance": a.variance,
                "variance_sampling": a.variance_sampling,
                "variance_rates": a.variance_rates,
                "youden_cases": a.youden_cases,
                "youden_controls": a.youden_controls,
                "differential": a.differential,
                "estimate": estimate_json(&fit.estimate, level)?,
            }))
        },
    )))
}

// ---------------------------------------------------------------------------
// Soft exposures
// ---------------------------------------------------------------------------

/// Fit a linear outcome model on per-unit exposure probabilities.
#[pyfunction]
fn measurement_soft_exposure(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "soft exposure")?;
    let map = object(&value, "soft exposure")?;
    let outcome = reals(map, "outcome")?;
    let probability = reals(map, "probability")?;
    let covariates = reals(map, "covariates")?;
    let covariate_count = index(map, "covariate_count")?;
    let level = level_of(map)?;
    let method = object(member(map, "method")?, "method")?;
    let kind = text(method, "kind")?;
    let supplied;
    let combination = match kind.as_str() {
        "posterior_mean" => SoftExposureCombination::PosteriorMean,
        "multiple_imputation" => SoftExposureCombination::MultipleImputation {
            draw_count: index(method, "draw_count")?,
            seed: unsigned(method, "seed")?,
        },
        "supplied_draws" => {
            supplied = reals(method, "draws")?;
            SoftExposureCombination::SuppliedDraws {
                draws: &supplied,
                draw_count: index(method, "draw_count")?,
            }
        }
        other => return Err(crate::value_err(format!("unknown soft-exposure method `{other}`"))),
    };
    let data = SoftExposureData {
        outcome: &outcome,
        probability: &probability,
        covariates: &covariates,
        covariate_count,
    };
    let ctx = execution_context(0, None, None);
    Ok(finish(correct_soft_exposure_effect(&data, &combination, &ctx).and_then(|effect| {
        let substitution = effect.substitution.as_ref().map(|s| {
            json!({
                "mean_posterior_variance": s.mean_posterior_variance,
                "variance": s.variance,
                "variance_exposure_uncertainty": s.variance_exposure_uncertainty,
                "variance_sampling": s.variance_sampling,
            })
        });
        let imputations = effect.imputations.as_ref().map(|c| {
            let d = c.dim;
            json!({
                "draws": c.draws,
                "within": c.within[d + 1],
                "between": c.between[d + 1],
                "draw_coefficients": c.draw_coefficients.iter().map(|b| b[1]).collect::<Vec<_>>(),
            })
        });
        Ok(json!({
            "coefficients": effect.coefficients,
            "covariance": effect.covariance,
            "substitution": substitution,
            "imputations": imputations,
            "estimate": estimate_json(&effect.estimate, level)?,
        }))
    })))
}

/// Compare stated exposure probabilities with repeat codings in bins of the probability.
#[pyfunction]
fn measurement_soft_exposure_reliability(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "soft exposure reliability")?;
    let map = object(&value, "soft exposure reliability")?;
    let probability = reals(map, "probability")?;
    let codings: Vec<u8> = list(map, "codings")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u8::try_from(n).ok())
                .ok_or_else(|| crate::value_err("`codings` holds a value that is not 0 or 1"))
        })
        .collect::<PyResult<_>>()?;
    let sizes: Vec<u32> = list(map, "sizes")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| crate::value_err("`sizes` holds a value that is not a count"))
        })
        .collect::<PyResult<_>>()?;
    let bins = index(map, "bins")?;
    let reference = match present(map, "reference") {
        None => None,
        Some(reference) => {
            let r = object(reference, "reference")?;
            Some(
                BinaryErrorRates::new(real(r, "sensitivity")?, real(r, "specificity")?, 0.0, 0.0)
                    .map_err(|error| crate::value_err(error.to_string()))?,
            )
        }
    };
    let ctx = execution_context(0, None, None);
    Ok(finish(
        soft_exposure_reliability_report(
            &probability,
            &codings,
            &sizes,
            bins,
            reference.as_ref(),
            &ctx,
        )
        .map(|report| {
            json!({
                "units": report.units,
                "max_abs_gap": report.max_abs_gap,
                "mean_abs_gap": report.mean_abs_gap,
                "chi_square": report.chi_square,
                "chi_square_bins": report.chi_square_bins,
                "gates": false,
                "bins": report.bins.iter().map(|b| json!({
                    "lower": b.lower,
                    "upper": b.upper,
                    "units": b.units,
                    "mean_stated": b.mean_stated,
                    "mean_expected": b.mean_expected,
                    "mean_observed": b.mean_observed,
                    "gap": b.gap,
                    "standard_error": b.standard_error,
                })).collect::<Vec<_>>(),
            })
        }),
    ))
}

// ---------------------------------------------------------------------------
// Detection that depends on a covariate
// ---------------------------------------------------------------------------

fn declaration_of(map: &Map<String, Json>) -> PyResult<DetectionDeclaration> {
    let kind = text(map, "kind")?;
    match kind.as_str() {
        "logistic" => Ok(DetectionDeclaration::Full(DetectionFunction::Logistic {
            intercept: real(map, "intercept")?,
            slope: real(map, "slope")?,
            intercept_variance: real(map, "intercept_variance")?,
            covariance: real(map, "covariance")?,
            slope_variance: real(map, "slope_variance")?,
        })),
        "table" => Ok(DetectionDeclaration::Full(DetectionFunction::Table {
            knots: reals(map, "knots")?,
            values: reals(map, "values")?,
        })),
        "envelope" => {
            let direction = match text(map, "direction")?.as_str() {
                "nondecreasing" => DetectionDirection::Nondecreasing,
                "nonincreasing" => DetectionDirection::Nonincreasing,
                other => {
                    return Err(crate::value_err(format!("unknown detection direction `{other}`")));
                }
            };
            let anchors = list(map, "anchors")?
                .iter()
                .map(|a| {
                    let triple: Vec<f64> = a
                        .as_array()
                        .ok_or_else(|| {
                            crate::value_err("an anchor must be (covariate, lower, upper)")
                        })?
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| crate::value_err("an anchor holds a non-number"))
                        })
                        .collect::<PyResult<_>>()?;
                    match triple.as_slice() {
                        [c, lo, hi] => {
                            Ok(DetectionAnchor { covariate: *c, lower: *lo, upper: *hi })
                        }
                        _ => Err(crate::value_err("an anchor must be (covariate, lower, upper)")),
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(DetectionDeclaration::Partial { direction, anchors })
        }
        other => Err(crate::value_err(format!("unknown detection declaration `{other}`"))),
    }
}

/// Prevalence of a condition detected with a probability that depends on a covariate.
#[pyfunction]
fn measurement_detection_dependent(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "detection")?;
    let map = object(&value, "detection")?;
    let detected: Vec<bool> = list(map, "detected")?
        .iter()
        .map(|v| v.as_bool().ok_or_else(|| crate::value_err("`detected` holds a non-boolean")))
        .collect::<PyResult<_>>()?;
    let covariate = reals(map, "covariate")?;
    let declaration = declaration_of(object(member(map, "declaration")?, "declaration")?)?;
    let fp = match present(map, "false_positive") {
        None => FalsePositiveRate::NONE,
        Some(fp) => {
            let fp = object(fp, "false_positive")?;
            FalsePositiveRate { rate: real(fp, "rate")?, variance: real(fp, "variance")? }
        }
    };
    let floor = real(map, "min_net_detection")?;
    let level = level_of(map)?;
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let ctx = execution_context(0, None, None);
    Ok(finish(
        correct_detection_dependent_prevalence(&data, &declaration, &fp, floor, &ctx).and_then(
            |result| match result.adjusted {
                DetectionAdjusted::Point(p) => Ok(json!({
                    "kind": "point",
                    "units": p.units,
                    "observed_rate": p.observed_rate,
                    "mean_detection": p.mean_detection,
                    "prevalence": p.prevalence,
                    "variance": p.variance,
                    "variance_sampling": p.variance_sampling,
                    "variance_parameters": p.variance_parameters,
                    "outside_unit_interval": p.outside_unit_interval,
                    "estimate": match &result.estimate {
                        Some(estimate) => estimate_json(estimate, level)?,
                        None => Json::Null,
                    },
                })),
                DetectionAdjusted::Bounds(b) => Ok(json!({
                    "kind": "bounds",
                    "units": b.units,
                    "observed_rate": b.observed_rate,
                    "mean_detection_lower": b.mean_detection_lower,
                    "mean_detection_upper": b.mean_detection_upper,
                    "lower": b.lower,
                    "upper": b.upper,
                    "lower_standard_error": b.lower_standard_error,
                    "upper_standard_error": b.upper_standard_error,
                })),
            },
        ),
    ))
}

// ---------------------------------------------------------------------------
// Latent-class repeat codings
// ---------------------------------------------------------------------------

/// Fit repeat codings of a binary or ordinal facet as replicates of one latent level.
#[pyfunction]
fn measurement_repeat_codings(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "repeat codings")?;
    let map = object(&value, "repeat codings")?;
    let codings: Vec<u8> = list(map, "codings")?
        .iter()
        .map(|v| {
            v.as_u64().and_then(|n| u8::try_from(n).ok()).ok_or_else(|| {
                crate::value_err("`codings` holds a value that is not a category index")
            })
        })
        .collect::<PyResult<_>>()?;
    let units = index(map, "units")?;
    let coding_count = index(map, "coding_count")?;
    let categories = index(map, "categories")?;
    let latent_levels = index(map, "latent_levels")?;
    let exchangeable = opt_flag(map, "exchangeable")?.unwrap_or(false);
    let lowest_level = match present(map, "lowest_level") {
        None => None,
        Some(_) => Some(reals(map, "lowest_level")?),
    };
    let defaults = LatentClassSettings::default();
    let settings = LatentClassSettings {
        starts: if present(map, "starts").is_some() {
            index(map, "starts")?
        } else {
            defaults.starts
        },
        max_iterations: if present(map, "max_iterations").is_some() {
            index(map, "max_iterations")?
        } else {
            defaults.max_iterations
        },
        tolerance: if present(map, "tolerance").is_some() {
            real(map, "tolerance")?
        } else {
            defaults.tolerance
        },
        min_separation: if present(map, "min_separation").is_some() {
            real(map, "min_separation")?
        } else {
            defaults.min_separation
        },
        seed: if present(map, "seed").is_some() { unsigned(map, "seed")? } else { defaults.seed },
    };
    let data =
        LatentClassData { codings: &codings, units, coding_count, categories, latent_levels };
    let constraints = LatentClassConstraints { exchangeable, lowest_level };
    let ctx = execution_context(0, None, None);
    Ok(finish(fit_repeat_codings(&data, &constraints, &settings, &ctx).map(|fit| {
        let errors = fit.standard_errors.as_ref();
        json!({
            "latent_levels": fit.latent_levels,
            "categories": fit.categories,
            "coding_count": fit.coding_count,
            "units": fit.units,
            "prevalence": fit.prevalence,
            "response": fit.response,
            "response_slots": fit.response_slots,
            "posterior": fit.posterior,
            "log_likelihood": fit.log_likelihood,
            "iterations": fit.iterations,
            "converged": fit.converged,
            "starts": fit.starts,
            "starts_agreeing": fit.starts_agreeing,
            "prevalence_standard_error": errors.map(|e| e.prevalence.clone()),
            "response_standard_error": errors.map(|e| e.response.clone()),
            "identification": {
                "kruskal": fit.identification.kruskal,
                "constrained": fit.identification.constrained,
                "informative_codings": fit.identification.informative_codings,
            },
        })
    })))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(measurement_exposure_misclassification, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_soft_exposure, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_soft_exposure_reliability, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_detection_dependent, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_repeat_codings, module)?)?;
    Ok(())
}

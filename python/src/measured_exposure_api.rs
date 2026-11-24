//! Python bridge for the measured-exposure route.
//!
//! The declaration, the imputation of the latent exposure, Rubin pooling and the draws
//! artifact are Rust's (`antecedent::estimate::measurement::measured_exposure` and
//! `antecedent::measured_exposure`). Python states the declaration as JSON, runs its own
//! per-draw study lifecycle on the completed columns and hands the per-draw estimates back
//! for pooling; a refusal comes back as structured JSON. The imputation and the pooling run
//! through their compiled program families
//! (`antecedent::analysis::kernel_programs::measured_exposure_effect`), which pass the support
//! gate of their unit and then make the same kernel call.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::kernel_programs::measured_exposure_effect::ImputationColumns;
use antecedent::estimate::measurement::{
    DetectionFunctionDeclaration, ExposureDraws, ExposureErrorDeclaration, ExposureMeasurement,
    MeasurementError, RatePair,
};
use antecedent::measured_exposure::{consume_exposure_draws, export_exposure_draws};
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

fn rate_pair(map: &Map<String, Json>) -> PyResult<Result<RatePair, MeasurementError>> {
    if present(map, "validation").is_some() {
        let counts: Vec<u64> = list(map, "validation")?
            .iter()
            .map(|v| v.as_u64().ok_or_else(|| crate::value_err("`validation` holds a non-count")))
            .collect::<PyResult<_>>()?;
        let [a, b, c, d] = counts.as_slice() else {
            return Err(crate::value_err("`validation` must hold four counts"));
        };
        return Ok(RatePair::validated(*a, *b, *c, *d));
    }
    Ok(RatePair::declared(
        real(map, "sensitivity")?,
        real(map, "specificity")?,
        real(map, "sensitivity_variance")?,
        real(map, "specificity_variance")?,
    ))
}

fn flags(map: &Map<String, Json>, key: &str) -> PyResult<Vec<bool>> {
    list(map, key)?
        .iter()
        .map(|v| match v {
            Json::Bool(b) => Ok(*b),
            other => other
                .as_u64()
                .filter(|n| *n <= 1)
                .map(|n| n == 1)
                .ok_or_else(|| crate::value_err(format!("`{key}` must hold 0/1 values"))),
        })
        .collect()
}

/// Parse the declaration JSON; a declaration refused by construction is the inner error.
fn declaration_of(
    map: &Map<String, Json>,
) -> PyResult<Result<ExposureMeasurement, MeasurementError>> {
    let kind = text(map, "kind")?;
    match kind.as_str() {
        "misclassified" => {
            let recorded = flags(map, "recorded")?;
            let min_youden =
                if present(map, "min_youden").is_some() { real(map, "min_youden")? } else { 0.1 };
            let error = object(member(map, "error")?, "error")?;
            let structure = match text(error, "kind")?.as_str() {
                "shared" => rate_pair(object(member(error, "rates")?, "rates")?)?
                    .map(ExposureErrorDeclaration::Shared),
                "by_outcome" => {
                    let cases = rate_pair(object(member(error, "cases")?, "cases")?)?;
                    let controls = rate_pair(object(member(error, "controls")?, "controls")?)?;
                    cases.and_then(|cases| {
                        controls
                            .map(|controls| ExposureErrorDeclaration::ByOutcome { cases, controls })
                    })
                }
                other => {
                    return Err(crate::value_err(format!("unknown error structure `{other}`")));
                }
            };
            Ok(structure.map(|error| ExposureMeasurement::Misclassified {
                recorded,
                error,
                min_youden,
            }))
        }
        "soft" => Ok(Ok(ExposureMeasurement::Soft { probability: reals(map, "probability")? })),
        "detected" => {
            let function = object(member(map, "function")?, "function")?;
            let function = match text(function, "kind")?.as_str() {
                "logistic" => DetectionFunctionDeclaration::Logistic {
                    intercept: real(function, "intercept")?,
                    slope: real(function, "slope")?,
                    intercept_variance: real(function, "intercept_variance")?,
                    covariance: real(function, "covariance")?,
                    slope_variance: real(function, "slope_variance")?,
                },
                "table" => DetectionFunctionDeclaration::Table {
                    knots: reals(function, "knots")?,
                    values: reals(function, "values")?,
                },
                other => {
                    return Err(crate::value_err(format!("unknown detection function `{other}`")));
                }
            };
            Ok(Ok(ExposureMeasurement::Detected {
                detected: flags(map, "detected")?,
                covariate: reals(map, "covariate")?,
                function,
                false_positive_rate: real(map, "false_positive_rate")?,
                false_positive_variance: real(map, "false_positive_variance")?,
                min_net_detection: real(map, "min_net_detection")?,
            }))
        }
        "repeat_coded" => {
            let codings: Vec<u8> = list(map, "codings")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| crate::value_err("`codings` holds a non-category"))
                })
                .collect::<PyResult<_>>()?;
            Ok(Ok(ExposureMeasurement::RepeatCoded {
                codings,
                coding_count: index(map, "coding_count")?,
                categories: index(map, "categories")?,
                latent_levels: index(map, "latent_levels")?,
                exchangeable: opt_flag(map, "exchangeable")?.unwrap_or(false),
                lowest_level: present(map, "lowest_level")
                    .map(|_| reals(map, "lowest_level"))
                    .transpose()?,
                starts: index(map, "starts")?,
            }))
        }
        other => Err(crate::value_err(format!("unknown exposure measurement `{other}`"))),
    }
}

fn draws_json(draws: &ExposureDraws) -> Json {
    json!({
        "kind": draws.kind,
        "levels": draws.levels,
        "rows": draws.rows,
        "draw_count": draws.draw_count,
        "probability": draws.probability,
        "draws": draws.draws,
        "declaration_identity": draws.declaration_identity,
        "data_identity": draws.data_identity,
        "seed": draws.seed,
    })
}

fn kind_of(name: &str) -> PyResult<&'static str> {
    match name {
        "misclassified_exposure" => Ok("misclassified_exposure"),
        "soft_exposure" => Ok("soft_exposure"),
        "detected_exposure" => Ok("detected_exposure"),
        "repeat_coded_exposure" => Ok("repeat_coded_exposure"),
        other => Err(crate::value_err(format!("unknown exposure kind `{other}`"))),
    }
}

/// Impute the latent exposure of a declaration.
#[pyfunction]
fn measured_exposure_impute(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "measured exposure")?;
    let map = object(&value, "measured exposure")?;
    let declaration = match declaration_of(object(member(map, "declaration")?, "declaration")?)? {
        Ok(declaration) => declaration,
        Err(error) => return Ok((None, Some(refusal(&error)))),
    };
    let outcome = reals(map, "outcome")?;
    let covariates = reals(map, "covariates")?;
    let covariate_count = index(map, "covariate_count")?;
    let draws = index(map, "draws")?;
    let seed = unsigned(map, "seed")?;
    let ctx = execution_context(seed, None, None);
    let columns = ImputationColumns { outcome, covariates, covariate_count };
    let compiled = Study::compile_exposure_imputation(declaration, columns, draws, seed, &ctx)
        .map_err(crate::py_err)?;
    Ok(finish(compiled.impute(&ctx).map(|draws| draws_json(&draws))))
}

/// The identity of a declaration, without imputing.
#[pyfunction]
fn measured_exposure_identity(declaration_json: &str) -> PyResult<Outcome> {
    let value = parse(declaration_json, "measured exposure")?;
    let map = object(&value, "measured exposure")?;
    Ok(match declaration_of(map)? {
        Ok(declaration) => (Some(declaration.identity()), None),
        Err(error) => (None, Some(refusal(&error))),
    })
}

/// Pool per-draw estimates and variances with Rubin's rules.
#[pyfunction]
fn measured_exposure_pool(pool_json: &str) -> PyResult<Outcome> {
    let value = parse(pool_json, "measured exposure pooling")?;
    let map = object(&value, "measured exposure pooling")?;
    let points = reals(map, "points")?;
    let variances = reals(map, "variances")?;
    let level = real(map, "level")?;
    if !(level > 0.0 && level < 1.0) {
        return Err(crate::value_err("level must lie strictly between 0 and 1"));
    }
    let ctx = execution_context(0, None, None);
    let compiled = Study::compile_measured_effect_pool(points, variances, None, &ctx)
        .map_err(crate::py_err)?;
    Ok(finish(compiled.pool(&ctx).map(|effect| {
        let (lower, upper) = effect.interval(level);
        json!({
            "estimate": effect.pooled.point,
            "within": effect.pooled.within,
            "between": effect.pooled.between,
            "total": effect.pooled.total,
            "degrees_of_freedom": effect.pooled.df,
            "draws": effect.pooled.imputations,
            "standard_error": effect.standard_error(),
            "interval": {"level": level, "lower": lower, "upper": upper, "calibration": "not_measured"},
            "variance_components": effect
                .variance_components()
                .iter()
                .map(|(name, variance)| json!({"name": name, "variance": variance}))
                .collect::<Vec<_>>(),
        })
    })))
}

fn draws_of(map: &Map<String, Json>) -> PyResult<ExposureDraws> {
    Ok(ExposureDraws {
        kind: kind_of(&text(map, "kind")?)?,
        levels: index(map, "levels")?,
        rows: index(map, "rows")?,
        draw_count: index(map, "draw_count")?,
        probability: reals(map, "probability")?,
        draws: reals(map, "draws")?,
        declaration_identity: text(map, "declaration_identity")?,
        data_identity: text(map, "data_identity")?,
        seed: unsigned(map, "seed")?,
    })
}

/// Write exposure draws as the library's posterior artifact.
#[pyfunction]
fn measured_exposure_export(draws_json: &str, exposure: &str) -> PyResult<Vec<u8>> {
    let value = parse(draws_json, "exposure draws")?;
    let draws = draws_of(object(&value, "exposure draws")?)?;
    if draws.draws.len() != draws.rows * draws.draw_count {
        return Err(crate::value_err("draws must hold draw_count x rows values"));
    }
    export_exposure_draws(exposure, &draws).map_err(|error| crate::value_err(error.to_string()))
}

/// Read exposure draws back from their posterior artifact.
#[pyfunction]
#[pyo3(signature = (artifact, exposure, expected_declaration_identity=None))]
fn measured_exposure_consume(
    artifact: Vec<u8>,
    exposure: &str,
    expected_declaration_identity: Option<&str>,
) -> PyResult<String> {
    let draws = consume_exposure_draws(&artifact, exposure, expected_declaration_identity)
        .map_err(|error| crate::value_err(error.to_string()))?;
    Ok(draws_json(&draws).to_string())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(measured_exposure_impute, module)?)?;
    module.add_function(wrap_pyfunction!(measured_exposure_identity, module)?)?;
    module.add_function(wrap_pyfunction!(measured_exposure_pool, module)?)?;
    module.add_function(wrap_pyfunction!(measured_exposure_export, module)?)?;
    module.add_function(wrap_pyfunction!(measured_exposure_consume, module)?)?;
    Ok(())
}

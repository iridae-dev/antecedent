//! Python bindings of the compiled sensitivity ranges of supplied evidence: Rosenbaum bounds of
//! matched designs and E-values of estimates computed outside the library.
//!
//! Each binding compiles the program of its family
//! ([`antecedent::analysis::supplied_sensitivity_programs`]) and executes it; Python receives JSON
//! keyed as the direct sensitivity function of the same analysis (`antecedent.validation`), with
//! the compiled program's identity under `program`. A kernel refusal raises exactly the
//! exception the direct function raises.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::supplied_sensitivity_programs::{
    MatchedDesign, MatchedSensitivityOutcome, SuppliedEValueOutcome, SuppliedEstimate,
    TerminalError, compile_matched_design_sensitivity, compile_supplied_estimate_evalue,
};
use antecedent::scientific::SemanticDigest;
use antecedent::validate::rosenbaum::{DEFAULT_GAMMA_GRID, MatchedSet, MatchedSetScore};
use antecedent::validate::{EffectInterval, EffectScale, ValidationError};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::target_validation_api::{hex, model_of, range_json, scalar_target, scale_of, target_of};
use crate::{detach_catch, py_err, py_execution_context, resolve_user_threads, value_err};

fn finite(value: f64) -> Value {
    if value.is_finite() { json!(value) } else { Value::Null }
}

/// The kernel's refusal as the direct function raises it, or the compiler's.
fn refusal(error: TerminalError<ValidationError>) -> PyErr {
    match error {
        TerminalError::Kernel(error) => py_err(error),
        TerminalError::Refused(error) => py_err(error),
    }
}

fn with_program(mut wire: Value, program: &SemanticDigest) -> String {
    if let Value::Object(map) = &mut wire {
        map.insert("program".into(), json!(hex(program.as_bytes())));
    }
    wire.to_string()
}

/// Compile and run the matched-design program of `design`.
fn run_matched(
    design: MatchedDesign,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<(MatchedSensitivityOutcome, SemanticDigest)> {
    let ctx = py_execution_context(seed, resolve_user_threads(threads));
    let program = compile_matched_design_sensitivity(design, &ctx).map_err(py_err)?;
    let outcome = program.execute(&ctx).map_err(refusal)?;
    Ok((outcome, program.identity()))
}

/// Compile and run the supplied-estimate E-value program of `estimate`.
fn run_evalue(
    estimate: SuppliedEstimate,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<(SuppliedEValueOutcome, SemanticDigest)> {
    let ctx = py_execution_context(seed, resolve_user_threads(threads));
    let program = compile_supplied_estimate_evalue(estimate, &ctx).map_err(py_err)?;
    let outcome = program.execute(&ctx).map_err(refusal)?;
    Ok((outcome, program.identity()))
}

/// Rosenbaum signed-rank bounds of matched-pair differences, through the compiled program.
#[pyfunction]
#[pyo3(signature = (differences, *, gammas=None, alpha=0.05, target_source=None, seed=1,
    threads=None))]
fn sensitivity_matched_pairs_compiled(
    py: Python<'_>,
    differences: Vec<f64>,
    gammas: Option<Vec<f64>>,
    alpha: f64,
    target_source: Option<String>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let gammas = gammas.unwrap_or_else(|| DEFAULT_GAMMA_GRID.to_vec());
    let target = scalar_target(target_source.as_deref(), &["matched_effect"])?;
    detach_catch(py, move || {
        let design = MatchedDesign::Pairs { target, differences, gammas, alpha };
        let (outcome, program) = run_matched(design, seed, threads)?;
        let MatchedSensitivityOutcome::Pairs { analysis, range } = outcome else {
            return Err(value_err("the matched-pair program returned another analysis"));
        };
        Ok(with_program(
            json!({
                "pairs": analysis.pairs,
                "statistic": analysis.statistic,
                "exact": analysis.exact,
                "alpha": alpha,
                "grid": analysis.grid.iter().map(|(g, p)| json!({"gamma": g, "p_value": p})).collect::<Vec<_>>(),
                "gamma_break_even": analysis.gamma_break_even.map(finite),
                "range": range_json(&range),
            }),
            &program,
        ))
    })
}

/// Rosenbaum bounds of a vector of matched designs, read at the weakest coordinate, through the
/// compiled program.
#[pyfunction]
#[pyo3(signature = (differences, labels, *, gammas=None, alpha=0.05,
    family="conditional_effect", temporal=false, target_source=None, seed=1, threads=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_matched_pair_vector_compiled(
    py: Python<'_>,
    differences: Vec<Vec<f64>>,
    labels: Vec<String>,
    gammas: Option<Vec<f64>>,
    alpha: f64,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let gammas = gammas.unwrap_or_else(|| DEFAULT_GAMMA_GRID.to_vec());
    let target = target_of(family, temporal, false, labels, target_source.as_deref())?;
    detach_catch(py, move || {
        let design = MatchedDesign::PairVector { target, differences, gammas, alpha };
        let (outcome, program) = run_matched(design, seed, threads)?;
        let MatchedSensitivityOutcome::PairVector { analyses, range } = outcome else {
            return Err(value_err("the matched-pair vector program returned another analysis"));
        };
        Ok(with_program(
            json!({
                "alpha": alpha,
                "pairs": analyses.iter().map(|a| a.pairs).collect::<Vec<_>>(),
                "gamma_break_even": analyses.iter().map(|a| a.gamma_break_even.map(finite)).collect::<Vec<_>>(),
                "range": range_json(&range),
            }),
            &program,
        ))
    })
}

/// Rosenbaum bounds for matched sets with several controls (separable approximation), through
/// the compiled program.
#[pyfunction]
#[pyo3(signature = (sets, *, score="aligned_ranks", gammas=None, alpha=0.05, seed=1,
    threads=None))]
fn sensitivity_matched_sets_compiled(
    py: Python<'_>,
    sets: Vec<(Vec<f64>, usize)>,
    score: &str,
    gammas: Option<Vec<f64>>,
    alpha: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let score = match score {
        "aligned_ranks" => MatchedSetScore::AlignedRanks,
        "outcomes" => MatchedSetScore::Outcomes,
        other => {
            return Err(value_err(format!(
                "unknown matched-set score {other:?}; expected aligned_ranks or outcomes"
            )));
        }
    };
    let sets: Vec<MatchedSet> =
        sets.into_iter().map(|(outcomes, treated)| MatchedSet { outcomes, treated }).collect();
    let gammas = gammas.unwrap_or_else(|| DEFAULT_GAMMA_GRID.to_vec());
    let count = sets.len();
    detach_catch(py, move || {
        let design = MatchedDesign::Sets { sets, score, gammas, alpha };
        let (outcome, program) = run_matched(design, seed, threads)?;
        let MatchedSensitivityOutcome::Sets { grid, gamma_break_even } = outcome else {
            return Err(value_err("the matched-set program returned another analysis"));
        };
        Ok(with_program(
            json!({
                "sets": count,
                "alpha": alpha,
                "grid": grid.iter().map(|bound| json!({
                    "gamma": bound.gamma,
                    "statistic": bound.statistic,
                    "expectation": bound.expectation,
                    "variance": bound.variance,
                    "p_value": bound.p_value,
                })).collect::<Vec<_>>(),
                "gamma_break_even": gamma_break_even.map(finite),
                "calibration": "not_measured",
            }),
            &program,
        ))
    })
}

/// The interval of one estimate on `scale`: limits on a ratio scale, a standard error on the
/// standardized scale.
fn interval_of(
    scale: EffectScale,
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    standard_error: Option<f64>,
) -> EffectInterval {
    if scale == EffectScale::StandardizedMeanDifference {
        EffectInterval::Standardized { estimate, standard_error }
    } else {
        EffectInterval::Ratio { estimate, lower, upper }
    }
}

/// E-values of a supplied estimate, through the compiled program.
#[pyfunction]
#[pyo3(signature = (estimate, *, lower=None, upper=None, standard_error=None,
    scale="risk_ratio", target_source=None, seed=1, threads=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_supplied_evalue_compiled(
    py: Python<'_>,
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    standard_error: Option<f64>,
    scale: &str,
    target_source: Option<String>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let scale = scale_of(scale)?;
    let interval = interval_of(scale, estimate, lower, upper, standard_error);
    let target = scalar_target(target_source.as_deref(), &["effect"])?;
    detach_catch(py, move || {
        let (outcome, program) =
            run_evalue(SuppliedEstimate::Scalar { target, interval, scale }, seed, threads)?;
        let SuppliedEValueOutcome::Scalar { bounds, range } = outcome else {
            return Err(value_err("the E-value program returned another analysis"));
        };
        Ok(with_program(
            json!({
                "scale": scale.as_str(),
                "risk_ratio": finite(bounds.risk_ratio),
                "risk_ratio_limit": bounds.risk_ratio_limit.map(finite),
                "evalue": finite(bounds.point),
                "evalue_limit": bounds.limit.map(finite),
                "range": range_json(&range),
            }),
            &program,
        ))
    })
}

/// E-values of every coordinate of a supplied vector of estimates, through the compiled
/// program.
#[pyfunction]
#[pyo3(signature = (estimates, labels, *, lowers=None, uppers=None, standard_errors=None,
    scale="risk_ratio", family="conditional_effect", temporal=false, target_source=None,
    model=None, seed=1, threads=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_supplied_evalue_vector_compiled(
    py: Python<'_>,
    estimates: Vec<f64>,
    labels: Vec<String>,
    lowers: Option<Vec<f64>>,
    uppers: Option<Vec<f64>>,
    standard_errors: Option<Vec<f64>>,
    scale: &str,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
    model: Option<&str>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let scale = scale_of(scale)?;
    let model = model_of(model)?;
    let n = estimates.len();
    for given in [&lowers, &uppers, &standard_errors].into_iter().flatten() {
        if given.len() != n {
            return Err(value_err(
                "limits and standard errors need one entry per coordinate label",
            ));
        }
    }
    let coordinates: Vec<EffectInterval> = (0..n)
        .map(|k| {
            interval_of(
                scale,
                estimates[k],
                lowers.as_ref().map(|v| v[k]),
                uppers.as_ref().map(|v| v[k]),
                standard_errors.as_ref().map(|v| v[k]),
            )
        })
        .collect();
    let target = target_of(family, temporal, false, labels, target_source.as_deref())?;
    detach_catch(py, move || {
        let estimate = SuppliedEstimate::Vector { target, coordinates, scale, model };
        let (outcome, program) = run_evalue(estimate, seed, threads)?;
        let SuppliedEValueOutcome::Vector { bounds, range } = outcome else {
            return Err(value_err("the E-value program returned another analysis"));
        };
        Ok(with_program(
            json!({
                "scale": scale.as_str(),
                "risk_ratio": bounds.iter().map(|b| finite(b.risk_ratio)).collect::<Vec<_>>(),
                "risk_ratio_limit": bounds.iter().map(|b| b.risk_ratio_limit.map(finite)).collect::<Vec<_>>(),
                "evalue": bounds.iter().map(|b| finite(b.point)).collect::<Vec<_>>(),
                "evalue_limit": bounds.iter().map(|b| b.limit.map(finite)).collect::<Vec<_>>(),
                "range": range_json(&range),
            }),
            &program,
        ))
    })
}

/// E-values of supplied posterior draws over a vector target, through the compiled program.
#[pyfunction]
#[pyo3(signature = (draws, labels, *, scale="risk_ratio", level=0.95,
    family="conditional_effect", temporal=false, target_source=None, model=None, seed=1,
    threads=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_supplied_evalue_posterior_compiled(
    py: Python<'_>,
    draws: Vec<Vec<f64>>,
    labels: Vec<String>,
    scale: &str,
    level: f64,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
    model: Option<&str>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let scale = scale_of(scale)?;
    let model = model_of(model)?;
    let target = target_of(family, temporal, true, labels, target_source.as_deref())?;
    detach_catch(py, move || {
        let estimate = SuppliedEstimate::Posterior { target, draws, scale, level, model };
        let (outcome, program) = run_evalue(estimate, seed, threads)?;
        let SuppliedEValueOutcome::Posterior { summary, range } = outcome else {
            return Err(value_err("the E-value program returned another analysis"));
        };
        Ok(with_program(
            json!({
                "scale": scale.as_str(),
                "level": level,
                "median_risk_ratio": summary.median_risk_ratio.iter().copied().map(finite).collect::<Vec<_>>(),
                "credible_interval": summary.credible_interval.iter()
                    .map(|(low, high)| json!([finite(*low), finite(*high)])).collect::<Vec<_>>(),
                "evalue": summary.bounds.iter().map(|b| finite(b.point)).collect::<Vec<_>>(),
                "evalue_limit": summary.bounds.iter().map(|b| b.limit.map(finite)).collect::<Vec<_>>(),
                "retention": summary.retention.iter()
                    .map(|(strength, probability)| json!({"strength": strength, "probability": probability}))
                    .collect::<Vec<_>>(),
                "range": range_json(&range),
            }),
            &program,
        ))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sensitivity_matched_pairs_compiled, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_matched_pair_vector_compiled, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_matched_sets_compiled, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_supplied_evalue_compiled, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_supplied_evalue_vector_compiled, module)?)?;
    module
        .add_function(wrap_pyfunction!(sensitivity_supplied_evalue_posterior_compiled, module)?)?;
    Ok(())
}

//! Python bindings for target-native refutation and the sensitivity ranges.
//!
//! Every statistic, catalogue, range and identity is computed in Rust
//! ([`antecedent_validate`] for the numerics, [`antecedent::analysis::target_validation`] for
//! the refits through a prepared program). Python receives JSON and builds typed result
//! objects from it (`antecedent.validation`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::target_validation::{
    PreparedTargetData, RefutationTarget, TargetFamily, TargetQuadrant, TargetRefitOptions,
    TargetRefutationConfig, TargetRefutationPlan, TargetRefutationReport, TargetSuite,
    refute_prepared_target,
};
use antecedent_core::VariableId;
use antecedent_validate::rosenbaum::{DEFAULT_GAMMA_GRID, MatchedSet, MatchedSetScore};
use antecedent_validate::target_native::{
    DerivativeCheckPoint, PosteriorPredictiveTarget, PriorSensitivityTarget,
    finite_difference_consistency, local_linearity, posterior_predictive_target_check,
    prior_sensitivity_target_check,
};
use antecedent_validate::{
    AssumptionRange, EffectInterval, EffectScale, LinearAdjustmentSummary, ValidationError,
    amplify, benchmark_bound, evalue_range, linear_adjustment_fit, matched_sets_break_even,
    matched_sets_upper_p, mediation_range, mediation_sensitivity_fit, ovb_range, rosenbaum_range,
};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::prepared_api::PyPreparedAnalysis;
use crate::{
    detach_catch, py_err, py_execution_context, resolve_user_threads, tabular_from_py_columns,
    value_err,
};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn finite(value: f64) -> Value {
    if value.is_finite() { json!(value) } else { Value::Null }
}

fn family_of(name: &str) -> PyResult<TargetFamily> {
    TargetFamily::parse(name).ok_or_else(|| {
        value_err(format!(
            "unknown target family {name:?}; expected scalar_average_effect, conditional_effect, \
             response_derivative, temporal_effect, mediation, world_valued or joint_contrast"
        ))
    })
}

fn suite_of(name: &str) -> PyResult<TargetSuite> {
    TargetSuite::parse(name).ok_or_else(|| {
        value_err(format!("unknown suite {name:?}; expected cheap, placebo or full"))
    })
}

/// A target defined by `source` (hex of 32 bytes, or the digest of the label list when absent).
fn target_of(
    family: &str,
    temporal: bool,
    bayesian: bool,
    labels: Vec<String>,
    source: Option<&str>,
) -> PyResult<RefutationTarget> {
    let family = family_of(family)?;
    let source = if let Some(text) = source {
        parse_source(text)?
    } else {
        let mut writer = antecedent_core::CanonicalWriter::new("antecedent.python_target");
        writer.texts(&labels[..]);
        writer.finish()
    };
    RefutationTarget::new(family, TargetQuadrant { temporal, bayesian }, labels, source)
        .map_err(py_err)
}

fn parse_source(text: &str) -> PyResult<[u8; 32]> {
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(value_err("a target source is 64 hexadecimal characters"));
    }
    let mut out = [0_u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
            .map_err(|_| value_err("a target source is 64 hexadecimal characters"))?;
    }
    Ok(out)
}

fn target_json(target: &RefutationTarget) -> Value {
    json!({
        "family": target.family().as_str(),
        "quadrant": target.quadrant().as_str(),
        "labels": target.labels().iter().map(|label| &**label).collect::<Vec<&str>>(),
        "identity": target.identity_hex(),
        "source": hex(target.source()),
    })
}

fn range_json(range: &AssumptionRange) -> Value {
    json!({
        "target": target_json(&range.target),
        "assumption": range.assumption.as_str(),
        "statement": range.assumption.statement(),
        "parameter": range.parameter.as_ref(),
        "null_parameter": range.null_parameter,
        "tested": range.tested.iter().map(|p| json!({
            "parameter": finite(p.parameter),
            "target": finite(p.target),
            "test": p.test.map(finite),
        })).collect::<Vec<_>>(),
        "break_even": range.break_even.map(finite),
        "break_even_significance": range.break_even_significance.map(finite),
        "identity": range.identity_hex(),
        "calibration": "unmeasured",
    })
}

fn report_json(report: &TargetRefutationReport) -> Value {
    json!({
        "refuter": report.refuter(),
        "target": report.target.identity_hex(),
        "original": report.original.iter().copied().map(finite).collect::<Vec<_>>(),
        "refuted": report.refuted.iter().copied().map(finite).collect::<Vec<_>>(),
        "statistic": report.statistic.kind.as_str(),
        "value": finite(report.statistic.value),
        "df": report.statistic.df,
        "p_value": report.statistic.p_value.map(finite),
        "passed": report.passed,
        "informative": report.informative,
        "replicates": report.replicates,
        "failure": report.failure.as_deref(),
    })
}

fn scalar_target(source: Option<&str>, labels: &[&str]) -> PyResult<RefutationTarget> {
    target_of(
        "scalar_average_effect",
        false,
        false,
        labels.iter().map(|s| (*s).to_string()).collect(),
        source,
    )
}

/// E-values of a ratio estimate (or a standardized mean difference) as an assumption range.
#[pyfunction]
#[pyo3(signature = (estimate, *, lower=None, upper=None, standard_error=None, scale="risk_ratio", target_source=None))]
fn sensitivity_evalue(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    standard_error: Option<f64>,
    scale: &str,
    target_source: Option<String>,
) -> PyResult<String> {
    let scale = EffectScale::parse(scale).ok_or_else(|| {
        value_err(format!(
            "unknown effect scale {scale:?}; expected risk_ratio, odds_ratio_rare, \
             odds_ratio_common, hazard_ratio_rare, hazard_ratio_common or \
             standardized_mean_difference"
        ))
    })?;
    let interval = if scale == EffectScale::StandardizedMeanDifference {
        EffectInterval::Standardized { estimate, standard_error }
    } else {
        EffectInterval::Ratio { estimate, lower, upper }
    };
    let target = scalar_target(target_source.as_deref(), &["effect"])?;
    let (bounds, range) = evalue_range(target, interval, scale).map_err(py_err)?;
    Ok(json!({
        "scale": scale.as_str(),
        "risk_ratio": finite(bounds.risk_ratio),
        "risk_ratio_limit": bounds.risk_ratio_limit.map(finite),
        "evalue": finite(bounds.point),
        "evalue_limit": bounds.limit.map(finite),
        "range": range_json(&range),
    })
    .to_string())
}

/// Omitted-variable-bias sensitivity of a linear-adjustment coefficient.
#[pyfunction]
#[pyo3(signature = (estimate, standard_error, dof, *, q=1.0, alpha=0.05, r2yz_extreme=1.0, benchmarks=None, kd=1.0, ky=None, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_omitted_variable_bias(
    estimate: f64,
    standard_error: f64,
    dof: f64,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    benchmarks: Option<Vec<(String, f64, f64)>>,
    kd: f64,
    ky: Option<f64>,
    target_source: Option<String>,
) -> PyResult<String> {
    let summary = LinearAdjustmentSummary::new(estimate, standard_error, dof).map_err(py_err)?;
    ovb_json(
        &summary,
        q,
        alpha,
        r2yz_extreme,
        benchmarks.unwrap_or_default(),
        kd,
        ky,
        target_source,
    )
}

#[allow(clippy::too_many_arguments)]
fn ovb_json(
    summary: &LinearAdjustmentSummary,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    benchmarks: Vec<(String, f64, f64)>,
    kd: f64,
    ky: Option<f64>,
    target_source: Option<String>,
) -> PyResult<String> {
    let target = scalar_target(target_source.as_deref(), &["coefficient"])?;
    let range = ovb_range(target, summary, q, alpha, r2yz_extreme).map_err(py_err)?;
    let ky = ky.unwrap_or(kd);
    let mut bounds = Vec::with_capacity(benchmarks.len());
    for (name, r2dxj, r2yxj) in benchmarks {
        let bound = benchmark_bound(r2dxj, r2yxj, kd, ky).map_err(py_err)?;
        bounds.push(json!({
            "covariate": name,
            "kd": bound.kd,
            "ky": bound.ky,
            "r2dz": bound.r2dz,
            "r2yz": bound.r2yz,
            "adjusted_estimate": finite(summary.adjusted_estimate(bound.r2dz, bound.r2yz)),
            "adjusted_standard_error":
                finite(summary.adjusted_standard_error(bound.r2dz, bound.r2yz)),
        }));
    }
    Ok(json!({
        "estimate": summary.estimate,
        "standard_error": summary.standard_error,
        "dof": summary.dof,
        "partial_r2_treatment": summary.partial_r2_treatment(),
        "robustness_value": summary.robustness_value(q, None),
        "robustness_value_alpha": summary.robustness_value(q, Some(alpha)),
        "q": q,
        "alpha": alpha,
        "benchmarks": bounds,
        "range": range_json(&range),
    })
    .to_string())
}

/// Omitted-variable-bias sensitivity of `outcome ~ treatment + covariates` fitted on columns,
/// benchmarked against every covariate.
#[pyfunction]
#[pyo3(signature = (outcome, treatment, covariates, covariate_names, *, q=1.0, alpha=0.05, r2yz_extreme=1.0, kd=1.0, ky=None, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_omitted_variable_bias_fit(
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    kd: f64,
    ky: Option<f64>,
    target_source: Option<String>,
) -> PyResult<String> {
    if covariate_names.len() != covariates.len() {
        return Err(value_err("one name per covariate column"));
    }
    let fit = linear_adjustment_fit(&outcome, &treatment, &covariates).map_err(py_err)?;
    let benchmarks = covariate_names
        .into_iter()
        .zip(fit.r2_treatment.iter().zip(&fit.r2_outcome))
        .map(|(name, (d, y))| (name, *d, *y))
        .collect();
    ovb_json(&fit.summary, q, alpha, r2yz_extreme, benchmarks, kd, ky, target_source)
}

/// Rosenbaum signed-rank bounds of matched-pair differences.
#[pyfunction]
#[pyo3(signature = (differences, *, gammas=None, alpha=0.05, target_source=None))]
fn sensitivity_rosenbaum(
    py: Python<'_>,
    differences: Vec<f64>,
    gammas: Option<Vec<f64>>,
    alpha: f64,
    target_source: Option<String>,
) -> PyResult<String> {
    let gammas = gammas.unwrap_or_else(|| DEFAULT_GAMMA_GRID.to_vec());
    let target = scalar_target(target_source.as_deref(), &["matched_effect"])?;
    detach_catch(py, move || {
        let (analysis, range) =
            rosenbaum_range(target, &differences, &gammas, alpha, None).map_err(py_err)?;
        Ok(json!({
            "pairs": analysis.pairs,
            "statistic": analysis.statistic,
            "exact": analysis.exact,
            "alpha": alpha,
            "grid": analysis.grid.iter().map(|(g, p)| json!({"gamma": g, "p_value": p})).collect::<Vec<_>>(),
            "gamma_break_even": analysis.gamma_break_even.map(finite),
            "range": range_json(&range),
        })
        .to_string())
    })
}

/// Rosenbaum bounds for matched sets with several controls (separable approximation).
#[pyfunction]
#[pyo3(signature = (sets, *, score="aligned_ranks", gammas=None, alpha=0.05))]
fn sensitivity_matched_sets(
    sets: Vec<(Vec<f64>, usize)>,
    score: &str,
    gammas: Option<Vec<f64>>,
    alpha: f64,
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
    let mut grid = Vec::with_capacity(gammas.len());
    for gamma in gammas {
        let bound = matched_sets_upper_p(&sets, score, gamma).map_err(py_err)?;
        grid.push(json!({
            "gamma": gamma,
            "statistic": bound.statistic,
            "expectation": bound.expectation,
            "variance": bound.variance,
            "p_value": bound.p_value,
        }));
    }
    let gamma_break_even = matched_sets_break_even(&sets, score, alpha).map_err(py_err)?;
    Ok(json!({
        "sets": sets.len(),
        "alpha": alpha,
        "grid": grid,
        "gamma_break_even": gamma_break_even.map(finite),
        "calibration": "unmeasured",
    })
    .to_string())
}

/// The outcome effect an unobserved covariate with treatment effect `lam` needs to produce
/// hidden bias `gamma`, or `None` when `lam <= gamma`.
#[pyfunction]
#[pyo3(signature = (gamma, lam))]
fn sensitivity_amplify(gamma: f64, lam: f64) -> Option<f64> {
    amplify(gamma, lam)
}

/// Mediation sensitivity to mediator-outcome confounding, fitted on columns.
#[pyfunction]
#[pyo3(signature = (treatment, mediator, outcome, covariates, *, target_source=None))]
fn sensitivity_mediation(
    treatment: Vec<f64>,
    mediator: Vec<f64>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    target_source: Option<String>,
) -> PyResult<String> {
    let summary =
        mediation_sensitivity_fit(&treatment, &mediator, &outcome, &covariates).map_err(py_err)?;
    let target = target_of(
        "mediation",
        false,
        false,
        vec!["total".into(), "direct".into(), "indirect".into()],
        target_source.as_deref(),
    )?;
    let range = mediation_range(target, &summary).map_err(py_err)?;
    Ok(json!({
        "beta_mediator": summary.beta_mediator,
        "sigma_total": summary.sigma_total,
        "sigma_mediator": summary.sigma_mediator,
        "rho_tilde": summary.rho_tilde,
        "indirect_at_zero": finite(summary.acme(0.0).map_err(py_err)?),
        "range": range_json(&range),
    })
    .to_string())
}

/// The target-native refuter catalogue of a target family, quadrant and suite.
#[pyfunction]
#[pyo3(signature = (family, *, temporal=false, bayesian=false, suite="full", labels=None))]
fn target_refutation_plan(
    family: &str,
    temporal: bool,
    bayesian: bool,
    suite: &str,
    labels: Option<Vec<String>>,
) -> PyResult<String> {
    let labels = labels.unwrap_or_else(|| vec!["target".to_string()]);
    let target = target_of(family, temporal, bayesian, labels, None)?;
    let plan = TargetRefutationPlan::for_target(target, suite_of(suite)?);
    Ok(json!({
        "target": target_json(plan.target()),
        "suite": plan.suite().as_str(),
        "refuters": plan.refuters().iter().map(|k| k.as_str()).collect::<Vec<_>>(),
    })
    .to_string())
}

fn call_response(response: &Bound<'_, PyAny>, x: f64) -> Result<f64, ValidationError> {
    let value = response.call1((x,)).map_err(|error| {
        ValidationError::estimation_msg(format!("response callable failed: {error}"))
    })?;
    value.extract::<f64>().map_err(|error| {
        ValidationError::estimation_msg(format!(
            "response callable did not return a float: {error}"
        ))
    })
}

/// Reported derivatives against Richardson-extrapolated central differences of a response.
#[pyfunction]
#[pyo3(signature = (response, points, reported, *, step=1e-2, tolerance=1e-4))]
fn target_derivative_consistency(
    response: Bound<'_, PyAny>,
    points: Vec<f64>,
    reported: Vec<f64>,
    step: f64,
    tolerance: f64,
) -> PyResult<String> {
    if points.len() != reported.len() {
        return Err(value_err("one reported derivative per evaluation point"));
    }
    let checks: Vec<DerivativeCheckPoint> = points
        .iter()
        .zip(&reported)
        .map(|(&x, &reported)| DerivativeCheckPoint { x, reported })
        .collect();
    let result =
        finite_difference_consistency(|x| call_response(&response, x), &checks, step, tolerance)
            .map_err(py_err)?;
    let target = derivative_target(&points)?;
    Ok(json!({
        "finite_difference": result.finite_difference,
        "truncation_error": result.truncation_error,
        "standardized": result.standardized,
        "report": report_json(&result.to_report(&target).map_err(py_err)?),
    })
    .to_string())
}

fn derivative_target(points: &[f64]) -> PyResult<RefutationTarget> {
    target_of(
        "response_derivative",
        false,
        false,
        points.iter().map(|x| format!("d/dx at {x}")).collect(),
        None,
    )
}

/// Second-difference curvature of a response relative to its slope over `step`.
#[pyfunction]
#[pyo3(signature = (response, points, *, step=1e-1, tolerance=0.1, slope_floor=1e-12))]
fn target_local_linearity(
    response: Bound<'_, PyAny>,
    points: Vec<f64>,
    step: f64,
    tolerance: f64,
    slope_floor: f64,
) -> PyResult<String> {
    let result =
        local_linearity(|x| call_response(&response, x), &points, step, tolerance, slope_floor)
            .map_err(py_err)?;
    let target = derivative_target(&points)?;
    Ok(json!({
        "slope": result.slope,
        "curvature": result.curvature,
        "nonlinearity": result.nonlinearity,
        "report": report_json(&result.to_report(&target).map_err(py_err)?),
    })
    .to_string())
}

fn rows_of(draws: &[Vec<f64>], k: usize, what: &str) -> PyResult<Vec<f64>> {
    if draws.iter().any(|row| row.len() != k) {
        return Err(value_err(format!("every {what} row has one value per target coordinate")));
    }
    Ok(draws.iter().flatten().copied().collect())
}

/// Prior sensitivity of the posterior of a whole target vector.
#[pyfunction]
#[pyo3(signature = (labels, baseline, alternatives, *, tolerance=0.5, family="world_valued", temporal=false))]
fn target_prior_sensitivity(
    labels: Vec<String>,
    baseline: Vec<Vec<f64>>,
    alternatives: Vec<(String, Vec<Vec<f64>>)>,
    tolerance: f64,
    family: &str,
    temporal: bool,
) -> PyResult<String> {
    let k = labels.len();
    let target = target_of(family, temporal, true, labels, None)?;
    let baseline = rows_of(&baseline, k, "baseline draw")?;
    let alternatives: Vec<(Arc<str>, Vec<f64>)> = alternatives
        .into_iter()
        .map(|(name, draws)| Ok((Arc::from(name), rows_of(&draws, k, "alternative draw")?)))
        .collect::<PyResult<_>>()?;
    let report = prior_sensitivity_target_check(&PriorSensitivityTarget {
        target: &target,
        baseline: &baseline,
        alternatives: &alternatives,
        tolerance,
    })
    .map_err(py_err)?;
    Ok(json!({"target": target_json(&target), "report": report_json(&report)}).to_string())
}

/// Posterior-predictive check of a target quantity.
#[pyfunction]
#[pyo3(signature = (labels, observed, draws, replicated, *, alpha=0.05, family="world_valued", temporal=false))]
fn target_posterior_predictive(
    labels: Vec<String>,
    observed: Vec<f64>,
    draws: Vec<Vec<f64>>,
    replicated: Vec<Vec<f64>>,
    alpha: f64,
    family: &str,
    temporal: bool,
) -> PyResult<String> {
    let k = labels.len();
    let target = target_of(family, temporal, true, labels, None)?;
    let draws = rows_of(&draws, k, "posterior draw")?;
    let replicated = rows_of(&replicated, k, "replicated target")?;
    let report = posterior_predictive_target_check(&PosteriorPredictiveTarget {
        target: &target,
        observed: &observed,
        draws: &draws,
        replicated: &replicated,
        alpha,
    })
    .map_err(py_err)?;
    Ok(json!({"target": target_json(&target), "report": report_json(&report)}).to_string())
}

fn column_id(names: &[String], name: Option<&str>, what: &str) -> PyResult<Option<VariableId>> {
    let Some(name) = name else { return Ok(None) };
    let index = names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| value_err(format!("{what} {name:?} is not a column of the data")))?;
    let raw = u32::try_from(index).map_err(|_| value_err("too many columns"))?;
    Ok(Some(VariableId::from_raw(raw)))
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Estimate on the columns, then refute the result's own target by refits of this
    /// prepared program; the outcome as JSON.
    #[pyo3(signature = (names, columns, *, suite="full", replicates=30, alpha=0.05, subset_fraction=0.8, negative_control_exposure=None, negative_control_outcome=None, mediator=None, mediation_rho_bar=0.1, common_cause_strength=0.0, prior_scale_factors=None, derivative_step=None, seed=1, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn refute_target(
        &self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        suite: &str,
        replicates: u32,
        alpha: f64,
        subset_fraction: f64,
        negative_control_exposure: Option<String>,
        negative_control_outcome: Option<String>,
        mediator: Option<String>,
        mediation_rho_bar: f64,
        common_cause_strength: f64,
        prior_scale_factors: Option<Vec<f64>>,
        derivative_step: Option<f64>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let suite = suite_of(suite)?;
        let options = TargetRefitOptions {
            negative_control_exposure: column_id(
                &names,
                negative_control_exposure.as_deref(),
                "negative_control_exposure",
            )?,
            negative_control_outcome: column_id(
                &names,
                negative_control_outcome.as_deref(),
                "negative_control_outcome",
            )?,
            mediator: column_id(&names, mediator.as_deref(), "mediator")?,
            mediation_rho_bar,
            common_cause_strength,
            prior_scale_factors: prior_scale_factors.unwrap_or_else(|| vec![0.25, 4.0]),
            derivative_step,
            ..TargetRefitOptions::default()
        };
        let config = TargetRefutationConfig {
            replicates,
            alpha,
            subset_fraction,
            ..TargetRefutationConfig::default()
        };
        let (data, _) = tabular_from_py_columns(names, columns)?;
        let study = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let result = study.estimate(&data, &ctx).map_err(py_err)?;
            let outcome = refute_prepared_target(
                &study,
                &result,
                PreparedTargetData::Tabular(&data),
                suite,
                &config,
                &options,
                &ctx,
            )
            .map_err(py_err)?;
            Ok(json!({
                "target": target_json(outcome.plan.target()),
                "suite": outcome.plan.suite().as_str(),
                "plan": outcome.plan.refuters().iter().map(|k| k.as_str()).collect::<Vec<_>>(),
                "reports": outcome.run.reports.iter().map(report_json).collect::<Vec<_>>(),
                "refused": outcome.run.refused.iter().map(|r| json!({
                    "refuter": r.kind.as_str(),
                    "reason": r.reason.as_ref(),
                    "next_attempt": r.next_attempt.as_ref(),
                })).collect::<Vec<_>>(),
                "certificate": hex(&outcome.certificate.identity),
                "calibration": "unmeasured",
            })
            .to_string())
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sensitivity_evalue, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_omitted_variable_bias, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_omitted_variable_bias_fit, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_rosenbaum, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_matched_sets, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_amplify, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_mediation, module)?)?;
    module.add_function(wrap_pyfunction!(target_refutation_plan, module)?)?;
    module.add_function(wrap_pyfunction!(target_derivative_consistency, module)?)?;
    module.add_function(wrap_pyfunction!(target_local_linearity, module)?)?;
    module.add_function(wrap_pyfunction!(target_prior_sensitivity, module)?)?;
    module.add_function(wrap_pyfunction!(target_posterior_predictive, module)?)?;
    Ok(())
}

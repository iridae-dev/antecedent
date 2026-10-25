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
    AssumptionRange, ComponentModel, EffectInterval, EffectScale, LinearAdjustmentSummary,
    MediationComponent, MediationComponents, MediationPath, MediationPosterior, MsmWeightPeriod,
    OvbPosterior, ValidationError,
    amplify, benchmark_bound, component_evalue_posterior_range, component_evalue_range,
    component_ovb_posterior_range, component_ovb_range, evalue_posterior_range, evalue_range,
    evalue_vector_range, linear_adjustment_fit, matched_sets_break_even, matched_sets_upper_p,
    mediation_components_fit, mediation_path_ovb_range, mediation_paths_fit, mediation_posterior_range, mediation_range,
    mediation_sensitivity_fit, mediation_vector_range, msm_parameter_fit, msm_parameter_fit_refitting_weights, ovb_posterior_range,
    ovb_range, ovb_vector_range, rosenbaum_range, rosenbaum_vector_range,
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
             response_derivative, temporal_effect, mediation, world_valued, joint_contrast or attribution"
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

/// The family of a marginal structural model's exposure parameters: one parameter is a scalar
/// average effect, several are a joint contrast.
fn msm_family(labels: &[String]) -> &'static str {
    if labels.len() == 1 { "scalar_average_effect" } else { "joint_contrast" }
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

fn scale_of(scale: &str) -> PyResult<EffectScale> {
    EffectScale::parse(scale).ok_or_else(|| {
        value_err(format!(
            "unknown effect scale {scale:?}; expected risk_ratio, odds_ratio_rare, \
             odds_ratio_common, hazard_ratio_rare, hazard_ratio_common or \
             standardized_mean_difference"
        ))
    })
}

/// The setting a vector of coordinates is read in: absent for a plain effect vector.
fn model_of(name: Option<&str>) -> PyResult<Option<ComponentModel>> {
    match name {
        None => Ok(None),
        Some("mediation") => Ok(Some(ComponentModel::Mediation)),
        Some("marginal_structural") => Ok(Some(ComponentModel::MarginalStructural)),
        Some(other) => Err(value_err(format!(
            "unknown model {other:?}; expected mediation or marginal_structural"
        ))),
    }
}

fn component_of(name: &str) -> PyResult<MediationComponent> {
    match name {
        "indirect" => Ok(MediationComponent::Indirect),
        "direct" => Ok(MediationComponent::Direct),
        other => Err(value_err(format!("unknown component {other:?}; expected indirect or direct"))),
    }
}

fn same_length(what: &str, lengths: &[usize]) -> PyResult<()> {
    if lengths.windows(2).all(|w| w[0] == w[1]) {
        Ok(())
    } else {
        Err(value_err(format!("{what} need one entry per coordinate label")))
    }
}

/// E-values of every coordinate of a vector target (conditional strata, horizons, contrasts),
/// read at the weakest coordinate.
#[pyfunction]
#[pyo3(signature = (estimates, labels, *, lowers=None, uppers=None, standard_errors=None,
    scale="risk_ratio", family="conditional_effect", temporal=false, target_source=None,
    model=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_evalue_vector(
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
) -> PyResult<String> {
    let scale = scale_of(scale)?;
    let model = model_of(model)?;
    let n = estimates.len();
    for given in [&lowers, &uppers, &standard_errors].into_iter().flatten() {
        same_length("limits and standard errors", &[n, given.len()])?;
    }
    let coordinates: Vec<EffectInterval> = (0..n)
        .map(|k| {
            if scale == EffectScale::StandardizedMeanDifference {
                EffectInterval::Standardized {
                    estimate: estimates[k],
                    standard_error: standard_errors.as_ref().map(|v| v[k]),
                }
            } else {
                EffectInterval::Ratio {
                    estimate: estimates[k],
                    lower: lowers.as_ref().map(|v| v[k]),
                    upper: uppers.as_ref().map(|v| v[k]),
                }
            }
        })
        .collect();
    let target = target_of(family, temporal, false, labels, target_source.as_deref())?;
    let (bounds, range) = match model {
        Some(model) => component_evalue_range(target, &coordinates, scale, model),
        None => evalue_vector_range(target, &coordinates, scale),
    }
    .map_err(py_err)?;
    Ok(json!({
        "scale": scale.as_str(),
        "risk_ratio": bounds.iter().map(|b| finite(b.risk_ratio)).collect::<Vec<_>>(),
        "risk_ratio_limit": bounds.iter().map(|b| b.risk_ratio_limit.map(finite)).collect::<Vec<_>>(),
        "evalue": bounds.iter().map(|b| finite(b.point)).collect::<Vec<_>>(),
        "evalue_limit": bounds.iter().map(|b| b.limit.map(finite)).collect::<Vec<_>>(),
        "range": range_json(&range),
    })
    .to_string())
}

/// E-values of a posterior over a vector target, with the joint posterior probability that
/// every coordinate keeps its reported side of the null at each confounder strength.
#[pyfunction]
#[pyo3(signature = (draws, labels, *, scale="risk_ratio", level=0.95,
    family="conditional_effect", temporal=false, target_source=None, model=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_evalue_posterior(
    draws: Vec<Vec<f64>>,
    labels: Vec<String>,
    scale: &str,
    level: f64,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
    model: Option<&str>,
) -> PyResult<String> {
    let scale = scale_of(scale)?;
    let model = model_of(model)?;
    let target = target_of(family, temporal, true, labels, target_source.as_deref())?;
    let (summary, range) = match model {
        Some(model) => component_evalue_posterior_range(target, &draws, scale, level, model),
        None => evalue_posterior_range(target, &draws, scale, level),
    }
    .map_err(py_err)?;
    Ok(json!({
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
    })
    .to_string())
}

/// Partial-R2 sensitivity of a vector of linear-adjustment coefficients.
#[pyfunction]
#[pyo3(signature = (estimates, standard_errors, dofs, labels, *, q=1.0, alpha=0.05,
    r2yz_extreme=1.0, family="conditional_effect", temporal=false, target_source=None,
    model=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_ovb_vector(
    estimates: Vec<f64>,
    standard_errors: Vec<f64>,
    dofs: Vec<f64>,
    labels: Vec<String>,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
    model: Option<&str>,
) -> PyResult<String> {
    let model = model_of(model)?;
    same_length(
        "estimates, standard errors and degrees of freedom",
        &[estimates.len(), standard_errors.len(), dofs.len()],
    )?;
    let summaries: Vec<LinearAdjustmentSummary> = estimates
        .iter()
        .zip(&standard_errors)
        .zip(&dofs)
        .map(|((e, s), d)| LinearAdjustmentSummary::new(*e, *s, *d))
        .collect::<Result<_, _>>()
        .map_err(py_err)?;
    let target = target_of(family, temporal, false, labels, target_source.as_deref())?;
    let range = match model {
        Some(model) => component_ovb_range(target, &summaries, q, alpha, r2yz_extreme, model),
        None => ovb_vector_range(target, &summaries, q, alpha, r2yz_extreme),
    }
    .map_err(py_err)?;
    Ok(json!({
        "partial_r2_treatment": summaries.iter().map(|s| s.partial_r2_treatment()).collect::<Vec<_>>(),
        "robustness_value": summaries.iter().map(|s| s.robustness_value(q, None)).collect::<Vec<_>>(),
        "robustness_value_alpha":
            summaries.iter().map(|s| s.robustness_value(q, Some(alpha))).collect::<Vec<_>>(),
        "q": q,
        "alpha": alpha,
        "range": range_json(&range),
    })
    .to_string())
}

/// Partial-R2 sensitivity of a posterior over a vector of linear-adjustment coefficients.
#[pyfunction]
#[pyo3(signature = (coefficients, bias_scale, labels, *, level=0.95, r2yz_extreme=1.0,
    family="conditional_effect", temporal=false, target_source=None, model=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_ovb_posterior(
    coefficients: Vec<Vec<f64>>,
    bias_scale: Vec<Vec<f64>>,
    labels: Vec<String>,
    level: f64,
    r2yz_extreme: f64,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
    model: Option<&str>,
) -> PyResult<String> {
    let model = model_of(model)?;
    let target = target_of(family, temporal, true, labels, target_source.as_deref())?;
    let posterior = OvbPosterior { coefficients: &coefficients, bias_scale: &bias_scale };
    let range = match model {
        Some(model) => {
            component_ovb_posterior_range(target, &posterior, level, r2yz_extreme, model)
        }
        None => ovb_posterior_range(target, &posterior, level, r2yz_extreme),
    }
    .map_err(py_err)?;
    Ok(json!({ "level": level, "range": range_json(&range) }).to_string())
}

/// Rosenbaum bounds of a vector of matched designs, read at the weakest coordinate.
#[pyfunction]
#[pyo3(signature = (differences, labels, *, gammas=None, alpha=0.05,
    family="conditional_effect", temporal=false, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_rosenbaum_vector(
    py: Python<'_>,
    differences: Vec<Vec<f64>>,
    labels: Vec<String>,
    gammas: Option<Vec<f64>>,
    alpha: f64,
    family: &str,
    temporal: bool,
    target_source: Option<String>,
) -> PyResult<String> {
    let gammas = gammas.unwrap_or_else(|| DEFAULT_GAMMA_GRID.to_vec());
    let target = target_of(family, temporal, false, labels, target_source.as_deref())?;
    detach_catch(py, move || {
        let (analyses, range) =
            rosenbaum_vector_range(target, &differences, &gammas, alpha, None).map_err(py_err)?;
        Ok(json!({
            "alpha": alpha,
            "pairs": analyses.iter().map(|a| a.pairs).collect::<Vec<_>>(),
            "gamma_break_even": analyses.iter().map(|a| a.gamma_break_even.map(finite)).collect::<Vec<_>>(),
            "range": range_json(&range),
        })
        .to_string())
    })
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

fn components_json(coordinate: &MediationComponents, component: MediationComponent) -> Value {
    json!({
        "beta_mediator": coordinate.mediation.beta_mediator,
        "sigma_total": coordinate.mediation.sigma_total,
        "sigma_mediator": coordinate.mediation.sigma_mediator,
        "rho_tilde": coordinate.mediation.rho_tilde,
        "total_effect": coordinate.total_effect,
        "indirect_at_zero": coordinate.effect_at(MediationComponent::Indirect, 0.0).ok().map(finite),
        "direct_at_zero": coordinate.effect_at(MediationComponent::Direct, 0.0).ok().map(finite),
        "break_even": coordinate.break_even(component).map(finite),
    })
}

/// Mediation sensitivity of one mediator to several outcomes (or horizons): the indirect or
/// direct effect of every outcome under mediator-outcome error correlation `rho`, read at the
/// weakest outcome.
#[pyfunction]
#[pyo3(signature = (treatment, mediator, outcomes, covariates, labels, *, component="indirect",
    temporal=false, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_mediation_vector(
    treatment: Vec<f64>,
    mediator: Vec<f64>,
    outcomes: Vec<Vec<f64>>,
    covariates: Vec<Vec<f64>>,
    labels: Vec<String>,
    component: &str,
    temporal: bool,
    target_source: Option<String>,
) -> PyResult<String> {
    let component = component_of(component)?;
    let coordinates: Vec<MediationComponents> = outcomes
        .iter()
        .map(|outcome| mediation_components_fit(&treatment, &mediator, outcome, &covariates))
        .collect::<Result<_, _>>()
        .map_err(py_err)?;
    let target = target_of("mediation", temporal, false, labels, target_source.as_deref())?;
    let range = mediation_vector_range(target, &coordinates, component).map_err(py_err)?;
    Ok(json!({
        "coordinates": coordinates.iter().map(|c| components_json(c, component)).collect::<Vec<_>>(),
        "range": range_json(&range),
    })
    .to_string())
}

/// Partial-R2 sensitivity of one path of a mediation analysis (treatment-mediator,
/// mediator-outcome, treatment-outcome or the indirect product), per outcome, read at the
/// weakest outcome.
#[pyfunction]
#[pyo3(signature = (treatment, mediator, outcomes, covariates, labels, *, path="indirect",
    q=1.0, alpha=0.05, r2yz_extreme=1.0, temporal=false, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_mediation_paths(
    treatment: Vec<f64>,
    mediator: Vec<f64>,
    outcomes: Vec<Vec<f64>>,
    covariates: Vec<Vec<f64>>,
    labels: Vec<String>,
    path: &str,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    temporal: bool,
    target_source: Option<String>,
) -> PyResult<String> {
    let path = match path {
        "treatment_mediator" => MediationPath::TreatmentMediator,
        "mediator_outcome" => MediationPath::MediatorOutcome,
        "treatment_outcome" => MediationPath::TreatmentOutcome,
        "indirect" => MediationPath::Indirect,
        other => {
            return Err(value_err(format!(
                "unknown path {other:?}; expected treatment_mediator, mediator_outcome, \
                 treatment_outcome or indirect"
            )));
        }
    };
    let coordinates: Vec<_> = outcomes
        .iter()
        .map(|outcome| mediation_paths_fit(&treatment, &mediator, outcome, &covariates))
        .collect::<Result<_, _>>()
        .map_err(py_err)?;
    let target = target_of("mediation", temporal, false, labels, target_source.as_deref())?;
    let range = mediation_path_ovb_range(target, &coordinates, path, q, alpha, r2yz_extreme)
        .map_err(py_err)?;
    let summary = |s: &LinearAdjustmentSummary| {
        json!({
            "estimate": s.estimate,
            "standard_error": s.standard_error,
            "dof": s.dof,
            "robustness_value": s.robustness_value(q, None),
            "robustness_value_alpha": s.robustness_value(q, Some(alpha)),
        })
    };
    Ok(json!({
        "coordinates": coordinates.iter().map(|c| json!({
            "treatment_mediator": summary(&c.treatment_mediator),
            "mediator_outcome": summary(&c.mediator_outcome),
            "treatment_outcome": summary(&c.treatment_outcome),
        })).collect::<Vec<_>>(),
        "q": q,
        "alpha": alpha,
        "range": range_json(&range),
    })
    .to_string())
}

/// Mediation sensitivity of a posterior over the regression summaries of several outcomes.
#[pyfunction]
#[pyo3(signature = (beta_mediator, sigma_total, sigma_mediator, rho_tilde, total_effect, labels,
    *, component="indirect", level=0.95, temporal=false, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_mediation_posterior(
    beta_mediator: Vec<Vec<f64>>,
    sigma_total: Vec<Vec<f64>>,
    sigma_mediator: Vec<Vec<f64>>,
    rho_tilde: Vec<Vec<f64>>,
    total_effect: Vec<Vec<f64>>,
    labels: Vec<String>,
    component: &str,
    level: f64,
    temporal: bool,
    target_source: Option<String>,
) -> PyResult<String> {
    let component = component_of(component)?;
    let target = target_of("mediation", temporal, true, labels, target_source.as_deref())?;
    let posterior = MediationPosterior {
        beta_mediator: &beta_mediator,
        sigma_total: &sigma_total,
        sigma_mediator: &sigma_mediator,
        rho_tilde: &rho_tilde,
        total_effect: &total_effect,
    };
    let range =
        mediation_posterior_range(target, &posterior, component, level).map_err(py_err)?;
    Ok(json!({ "level": level, "range": range_json(&range) }).to_string())
}

/// Partial-R2 sensitivity of the exposure parameters of a marginal structural model, fitted by
/// weighted least squares with a robust standard error.
#[pyfunction]
#[pyo3(signature = (outcome, exposures, weights, labels, *, q=1.0, alpha=0.05,
    r2yz_extreme=1.0, temporal=false, target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_msm_parameters(
    outcome: Vec<f64>,
    exposures: Vec<Vec<f64>>,
    weights: Vec<f64>,
    labels: Vec<String>,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    temporal: bool,
    target_source: Option<String>,
) -> PyResult<String> {
    let summaries = msm_parameter_fit(&outcome, &exposures, &weights).map_err(py_err)?;
    let target = target_of(msm_family(&labels), temporal, false, labels, target_source.as_deref())?;
    let range = component_ovb_range(
        target,
        &summaries,
        q,
        alpha,
        r2yz_extreme,
        ComponentModel::MarginalStructural,
    )
    .map_err(py_err)?;
    Ok(json!({
        "estimate": summaries.iter().map(|s| s.estimate).collect::<Vec<_>>(),
        "standard_error": summaries.iter().map(|s| s.standard_error).collect::<Vec<_>>(),
        "dof": summaries.iter().map(|s| s.dof).collect::<Vec<_>>(),
        "partial_r2_treatment": summaries.iter().map(|s| s.partial_r2_treatment()).collect::<Vec<_>>(),
        "robustness_value": summaries.iter().map(|s| s.robustness_value(q, None)).collect::<Vec<_>>(),
        "robustness_value_alpha":
            summaries.iter().map(|s| s.robustness_value(q, Some(alpha))).collect::<Vec<_>>(),
        "q": q,
        "alpha": alpha,
        "range": range_json(&range),
    })
    .to_string())
}

/// Partial-R2 sensitivity of the exposure parameters of a marginal structural model whose
/// stabilized weights are estimated: every treatment model is refitted in each unit-bootstrap
/// replicate, so the standard error carries the cost of estimating the weights.
#[pyfunction]
#[pyo3(signature = (outcome, exposures, treatments, numerators, denominators, labels, *,
    replicates=200, seed=1, q=1.0, alpha=0.05, r2yz_extreme=1.0, temporal=false,
    target_source=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_msm_parameters_refit(
    outcome: Vec<f64>,
    exposures: Vec<Vec<f64>>,
    treatments: Vec<Vec<f64>>,
    numerators: Vec<Vec<Vec<f64>>>,
    denominators: Vec<Vec<Vec<f64>>>,
    labels: Vec<String>,
    replicates: usize,
    seed: u64,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    temporal: bool,
    target_source: Option<String>,
) -> PyResult<String> {
    same_length(
        "treatments and their numerator and denominator covariates",
        &[treatments.len(), numerators.len(), denominators.len()],
    )?;
    let periods: Vec<MsmWeightPeriod> = treatments
        .into_iter()
        .zip(numerators)
        .zip(denominators)
        .map(|((treatment, numerator), denominator)| MsmWeightPeriod {
            treatment,
            numerator,
            denominator,
        })
        .collect();
    let fit = msm_parameter_fit_refitting_weights(&outcome, &exposures, &periods, replicates, seed)
        .map_err(py_err)?;
    let target = target_of(msm_family(&labels), temporal, false, labels, target_source.as_deref())?;
    let range = component_ovb_range(
        target,
        &fit.refit,
        q,
        alpha,
        r2yz_extreme,
        ComponentModel::MarginalStructural,
    )
    .map_err(py_err)?;
    Ok(json!({
        "estimate": fit.refit.iter().map(|s| s.estimate).collect::<Vec<_>>(),
        "standard_error": fit.refit.iter().map(|s| s.standard_error).collect::<Vec<_>>(),
        "fixed_weight_standard_error":
            fit.fixed_weights.iter().map(|s| s.standard_error).collect::<Vec<_>>(),
        "dof": fit.refit.iter().map(|s| s.dof).collect::<Vec<_>>(),
        "replicates": [fit.replicates.0, fit.replicates.1],
        "robustness_value": fit.refit.iter().map(|s| s.robustness_value(q, None)).collect::<Vec<_>>(),
        "q": q,
        "alpha": alpha,
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
    module.add_function(wrap_pyfunction!(sensitivity_evalue_vector, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_evalue_posterior, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_ovb_vector, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_ovb_posterior, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_rosenbaum_vector, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_omitted_variable_bias, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_omitted_variable_bias_fit, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_rosenbaum, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_matched_sets, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_amplify, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_mediation, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_mediation_vector, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_mediation_posterior, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_mediation_paths, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_msm_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(sensitivity_msm_parameters_refit, module)?)?;
    module.add_function(wrap_pyfunction!(target_refutation_plan, module)?)?;
    module.add_function(wrap_pyfunction!(target_derivative_consistency, module)?)?;
    module.add_function(wrap_pyfunction!(target_local_linearity, module)?)?;
    module.add_function(wrap_pyfunction!(target_prior_sensitivity, module)?)?;
    module.add_function(wrap_pyfunction!(target_posterior_predictive, module)?)?;
    Ok(())
}

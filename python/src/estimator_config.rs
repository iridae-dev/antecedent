//! Shared `estimator_config` dict parser for the analyze entry points.
//!
//! Before this module, the only estimator tuning reachable from Python was the three loose
//! `rd.sharp` kwargs (`running_variable` / `cutoff` / `bandwidth`, still supported unchanged
//! for backward compatibility). This module adds one table-driven parser behind a single
//! `estimator_config: dict | None` kwarg, covering the ten estimators the facade's
//! [`antecedent::EstimatorSpec`] can carry fully configured
//! (`crates/antecedent/src/estimator_spec.rs`), plus the `rd.sharp` triple.
//!
//! Two invariants drive the design:
//!
//! - **`estimator_config=None` is a strict no-op.** [`parse_estimator_config`] returns
//!   [`ParsedEstimatorConfig::default()`] immediately, so every existing call site is
//!   byte-identical to today.
//! - **Unknown or misplaced keys are hard errors, never silent no-ops.** A key that doesn't
//!   exist for any estimator, or that belongs to a *different* estimator than the one
//!   resolved for this call, is rejected with a message naming the key and (when applicable)
//!   the estimator(s) it does belong to. Dropping a config key silently is exactly the bug
//!   class this module exists to prevent.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::EstimatorSpec;
use antecedent_estimate::{
    AipwAte, AnalyticSeKind, CaliperScale, DistanceMatching, FrontDoorTwoStage, GlmAdjustmentAte,
    LinearAdjustmentAte, LinearFitKind, PropensityMatching, PropensityStratification,
    PropensityWeighting, TwoStageLeastSquares, WaldIv,
};
use antecedent_stats::{GlmFamily, GlmOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// Estimator id `StudyBuilder` resolves to when `.estimator(...)` is never called (mirrors
/// `StudyBuilder::estimator`'s documented default; see
/// `crates/antecedent/src/analysis/builder.rs`). Used only to pick which key table applies
/// when `estimator_config` is supplied without an explicit `estimator=` kwarg.
const DEFAULT_ESTIMATOR_ID: &str = "linear.adjustment.ate";

/// One `(estimator_id, valid_keys)` row. Table-driven: adding a configurable estimator or a
/// key to an existing one is a data change here, not a new branch scattered through the
/// parser. The unknown-key / wrong-estimator error messages are both derived from this table.
const ESTIMATOR_KEYS: &[(&str, &[&str])] = &[
    (
        "linear.adjustment.ate",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "fit_kind",
            "fit_lambda",
            "fit_c",
        ],
    ),
    ("propensity.weighting", &["bootstrap_replicates", "glm_options"]),
    (
        "propensity.matching",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "caliper",
            "caliper_scale",
        ],
    ),
    ("propensity.stratification", &["bootstrap_replicates", "glm_options", "n_strata"]),
    (
        "distance.matching",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "caliper",
        ],
    ),
    (
        "aipw",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
        ],
    ),
    (
        "glm.adjustment",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "family",
        ],
    ),
    ("frontdoor.two_stage", &["bootstrap_replicates", "se_kind", "se_lag", "cluster_ids"]),
    (
        "iv.wald",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
        ],
    ),
    (
        "iv.2sls",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
        ],
    ),
    // `rd.sharp` has no `EstimatorSpec` variant (see estimator_spec.rs doc comment) — its
    // triple is applied via the pre-existing `StudyBuilder::rd_config` path instead, merged
    // with the loose `running_variable`/`cutoff`/`bandwidth` kwargs by `merge_rd_triple`.
    ("rd.sharp", &["running_variable", "cutoff", "bandwidth"]),
];

/// Valid `glm_options` sub-dict keys (see [`build_glm_options`]).
const GLM_OPTION_KEYS: &[&str] = &["max_iter", "tol", "ridge_on_separation"];

fn valid_keys_for(estimator_id: &str) -> Option<&'static [&'static str]> {
    ESTIMATOR_KEYS.iter().find(|(id, _)| *id == estimator_id).map(|(_, keys)| *keys)
}

fn estimators_accepting(key: &str) -> Vec<&'static str> {
    ESTIMATOR_KEYS.iter().filter(|(_, keys)| keys.contains(&key)).map(|(id, _)| *id).collect()
}

fn supported_estimator_ids() -> Vec<&'static str> {
    ESTIMATOR_KEYS.iter().map(|(id, _)| *id).collect()
}

fn format_id_list(ids: &[&str]) -> String {
    let mut sorted: Vec<&str> = ids.to_vec();
    sorted.sort_unstable();
    sorted.join(", ")
}

/// Outcome of parsing `estimator_config`.
///
/// `spec`, when `Some`, is a fully-built [`EstimatorSpec`] ready for
/// `StudyBuilder::estimator` (one of the ten estimators `EstimatorSpec` covers). The RD
/// fields carry the `rd.sharp` triple parsed out of `estimator_config`, to be merged with the
/// existing loose `running_variable`/`cutoff`/`bandwidth` kwargs via [`merge_rd_triple`] —
/// `rd.sharp` has no `EstimatorSpec` variant, so it never sets `spec`.
#[derive(Default)]
pub(crate) struct ParsedEstimatorConfig {
    pub(crate) spec: Option<EstimatorSpec>,
    pub(crate) rd_running_variable: Option<String>,
    pub(crate) rd_cutoff: Option<f64>,
    pub(crate) rd_bandwidth: Option<f64>,
}

/// Parse `estimator_config` into a [`ParsedEstimatorConfig`].
///
/// `estimator` is the Python `estimator=` kwarg exactly as given; when `None`, the resolved
/// id mirrors `StudyBuilder`'s own default ([`DEFAULT_ESTIMATOR_ID`]). `ambient_bootstrap` is
/// the analyze function's own `bootstrap=` kwarg, used as the configured estimator's
/// bootstrap-replicate count whenever `estimator_config` doesn't override it — so a
/// configured estimator still honors the top-level `bootstrap=` kwarg exactly as an
/// unconfigured one would (the caller must then skip its own
/// `StudyBuilder::bootstrap_replicates` call whenever `spec` comes back `Some`; see
/// `ate_api.rs`, since combining both is refused at `StudyBuilder::build()` time).
///
/// `estimator_config=None` (the kwarg omitted or explicitly `None`) always returns
/// [`ParsedEstimatorConfig::default()`] — a strict no-op, byte-identical to today's behavior.
/// An empty dict (`{}`) is treated the same way, regardless of estimator.
///
/// # Errors
///
/// Returns [`PyValueError`] when: a key is unknown for the resolved estimator (message lists
/// the valid keys for that estimator); a key is valid but for a *different* estimator
/// (message names which one); a value has the wrong Python type or shape; or the resolved
/// estimator has no configuration surface at all (the temporal / conditional / mediation /
/// functional-plug-in families, and `bayesian.gcomp`, are selectable only via
/// [`antecedent::EstimatorSpec::Default`] — see `estimator_spec.rs`).
pub(crate) fn parse_estimator_config(
    estimator_config: Option<&Bound<'_, PyDict>>,
    estimator: Option<&str>,
    ambient_bootstrap: u32,
) -> PyResult<ParsedEstimatorConfig> {
    let Some(dict) = estimator_config else {
        return Ok(ParsedEstimatorConfig::default());
    };
    if dict.is_empty() {
        return Ok(ParsedEstimatorConfig::default());
    }
    let resolved_id = estimator.unwrap_or(DEFAULT_ESTIMATOR_ID);
    let Some(valid_keys) = valid_keys_for(resolved_id) else {
        return Err(PyValueError::new_err(format!(
            "estimator_config is not supported for estimator {resolved_id:?}; no per-estimator \
             configuration surface exists for it yet. estimator_config currently supports: {}",
            format_id_list(&supported_estimator_ids()),
        )));
    };

    // Validate every key up front, so the error always names the actual offending key
    // rather than whichever field extraction happened to run first.
    for (key_obj, _val) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyValueError::new_err("estimator_config keys must be str"))?;
        if !valid_keys.contains(&key.as_str()) {
            let owners = estimators_accepting(&key);
            if owners.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "unknown estimator_config key {key:?} for estimator {resolved_id:?}; valid \
                     keys for {resolved_id:?} are: {}",
                    format_id_list(valid_keys),
                )));
            }
            return Err(PyValueError::new_err(format!(
                "estimator_config key {key:?} is not valid for estimator {resolved_id:?}; it \
                 belongs to estimator(s) {}. valid keys for {resolved_id:?} are: {}",
                format_id_list(&owners),
                format_id_list(valid_keys),
            )));
        }
    }

    if resolved_id == "rd.sharp" {
        return Ok(ParsedEstimatorConfig {
            spec: None,
            rd_running_variable: get_string(dict, "running_variable")?,
            rd_cutoff: get_f64(dict, "cutoff")?,
            rd_bandwidth: get_f64(dict, "bandwidth")?,
        });
    }

    let spec = build_configured_spec(resolved_id, dict, ambient_bootstrap)?;
    Ok(ParsedEstimatorConfig { spec: Some(spec), ..ParsedEstimatorConfig::default() })
}

/// Merge the `rd.sharp` triple parsed out of `estimator_config` with the pre-existing loose
/// `running_variable`/`cutoff`/`bandwidth` kwargs.
///
/// Supplying a field in only one place (either, not both) passes it through unchanged — this
/// is what keeps behavior byte-identical when `estimator_config` isn't used. Supplying the
/// same field in both places with agreeing values also passes through. Disagreement is a hard
/// error naming the conflicting field and both values, rather than silently picking a winner.
///
/// # Errors
///
/// Returns [`PyValueError`] when the same field is set to different values by the loose kwarg
/// and by `estimator_config`.
pub(crate) fn merge_rd_triple(
    loose_running_variable: Option<String>,
    loose_cutoff: Option<f64>,
    loose_bandwidth: Option<f64>,
    configured_running_variable: Option<String>,
    configured_cutoff: Option<f64>,
    configured_bandwidth: Option<f64>,
) -> PyResult<(Option<String>, Option<f64>, Option<f64>)> {
    let running_variable = merge_rd_field(
        "running_variable",
        loose_running_variable,
        configured_running_variable,
        |a, b| a == b,
    )?;
    // Exact equality is deliberate: this asks "did the caller pass the same literal
    // twice", not "are these numerically close". A tolerance here would silently accept
    // two genuinely different cutoffs.
    #[allow(clippy::float_cmp)]
    let cutoff =
        merge_rd_field("cutoff", loose_cutoff, configured_cutoff, |a: &f64, b: &f64| a == b)?;
    #[allow(clippy::float_cmp)]
    let bandwidth =
        merge_rd_field("bandwidth", loose_bandwidth, configured_bandwidth, |a: &f64, b: &f64| {
            a == b
        })?;
    Ok((running_variable, cutoff, bandwidth))
}

fn merge_rd_field<T: std::fmt::Display>(
    name: &str,
    loose: Option<T>,
    configured: Option<T>,
    eq: impl Fn(&T, &T) -> bool,
) -> PyResult<Option<T>> {
    match (loose, configured) {
        (Some(a), Some(b)) => {
            if eq(&a, &b) {
                Ok(Some(a))
            } else {
                Err(PyValueError::new_err(format!(
                    "conflicting rd.sharp {name}: the loose {name}={a} kwarg disagrees with \
                     estimator_config[{name:?}]={b}; supply it in only one place",
                )))
            }
        }
        (Some(a), None) => Ok(Some(a)),
        (None, Some(b)) => Ok(Some(b)),
        (None, None) => Ok(None),
    }
}

// --- Value extraction helpers -----------------------------------------------------------

fn type_name(v: &Bound<'_, PyAny>) -> String {
    v.get_type().name().map_or_else(|_| "?".to_string(), |n| n.to_string())
}

fn get_u32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u32>> {
    let Some(v) = dict.get_item(key)? else { return Ok(None) };
    v.extract::<u32>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config[{key:?}] must be a non-negative int, got {}",
            type_name(&v)
        ))
    })
}

fn get_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    let Some(v) = dict.get_item(key)? else { return Ok(None) };
    v.extract::<f64>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config[{key:?}] must be a float, got {}",
            type_name(&v)
        ))
    })
}

fn get_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(v) = dict.get_item(key)? else { return Ok(None) };
    v.extract::<String>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config[{key:?}] must be a str, got {}",
            type_name(&v)
        ))
    })
}

fn get_u32_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<u32>>> {
    let Some(v) = dict.get_item(key)? else { return Ok(None) };
    v.extract::<Vec<u32>>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config[{key:?}] must be a list of non-negative ints, got {}",
            type_name(&v)
        ))
    })
}

fn get_u32_list_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<Vec<u32>>>> {
    let Some(v) = dict.get_item(key)? else { return Ok(None) };
    v.extract::<Vec<Vec<u32>>>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config[{key:?}] must be a list of lists of non-negative ints, got {}",
            type_name(&v)
        ))
    })
}

fn get_i64_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<i64>>> {
    let Some(v) = dict.get_item(key)? else { return Ok(None) };
    v.extract::<Vec<i64>>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config[{key:?}] must be a list of ints, got {}",
            type_name(&v)
        ))
    })
}

/// Build [`AnalyticSeKind`] from `se_kind` (+ `se_lag` for the two lag-parameterized tags).
fn build_se_kind(dict: &Bound<'_, PyDict>) -> PyResult<Option<AnalyticSeKind>> {
    let Some(tag) = get_string(dict, "se_kind")? else { return Ok(None) };
    let lag = get_u32(dict, "se_lag")?;
    let needs_lag = matches!(tag.as_str(), "newey_west" | "panel_cluster_hac");
    if !needs_lag && lag.is_some() {
        return Err(PyValueError::new_err(
            "estimator_config['se_lag'] is only valid when se_kind is \"newey_west\" or \
             \"panel_cluster_hac\"",
        ));
    }
    let kind = match tag.as_str() {
        "homoskedastic" => AnalyticSeKind::Homoskedastic,
        "hc0" => AnalyticSeKind::Hc0,
        "hc1" => AnalyticSeKind::Hc1,
        "hc2" => AnalyticSeKind::Hc2,
        "hc3" => AnalyticSeKind::Hc3,
        "cluster" => AnalyticSeKind::Cluster,
        "multiway" => AnalyticSeKind::Multiway,
        "newey_west" => {
            let lag = lag.ok_or_else(|| {
                PyValueError::new_err(
                    "estimator_config['se_lag'] is required when se_kind is \"newey_west\"",
                )
            })?;
            AnalyticSeKind::NeweyWest { lag: lag as usize }
        }
        "panel_cluster_hac" => {
            let lag = lag.ok_or_else(|| {
                PyValueError::new_err(
                    "estimator_config['se_lag'] is required when se_kind is \
                     \"panel_cluster_hac\"",
                )
            })?;
            AnalyticSeKind::PanelClusterHac { lag: lag as usize }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "estimator_config['se_kind'] {other:?} is not recognized; use homoskedastic|\
                 hc0|hc1|hc2|hc3|cluster|multiway|newey_west|panel_cluster_hac",
            )));
        }
    };
    Ok(Some(kind))
}

/// Build [`LinearFitKind`] from `fit_kind` (+ `fit_lambda`/`fit_c`), `linear.adjustment.ate` only.
fn build_fit_kind(dict: &Bound<'_, PyDict>) -> PyResult<Option<LinearFitKind>> {
    let Some(tag) = get_string(dict, "fit_kind")? else { return Ok(None) };
    let lambda = get_f64(dict, "fit_lambda")?;
    let c = get_f64(dict, "fit_c")?;
    let kind = match tag.as_str() {
        "ols" => LinearFitKind::Ols,
        "ridge" => LinearFitKind::Ridge {
            lambda: lambda.ok_or_else(|| {
                PyValueError::new_err(
                    "estimator_config['fit_lambda'] is required when fit_kind is \"ridge\"",
                )
            })?,
        },
        "lasso" => LinearFitKind::Lasso {
            lambda: lambda.ok_or_else(|| {
                PyValueError::new_err(
                    "estimator_config['fit_lambda'] is required when fit_kind is \"lasso\"",
                )
            })?,
        },
        "huber" => LinearFitKind::Huber {
            c: c.ok_or_else(|| {
                PyValueError::new_err(
                    "estimator_config['fit_c'] is required when fit_kind is \"huber\"",
                )
            })?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "estimator_config['fit_kind'] {other:?} is not recognized; use ols|ridge|lasso|\
                 huber",
            )));
        }
    };
    Ok(Some(kind))
}

/// Build [`GlmFamily`] from `family`, `glm.adjustment` only.
fn build_family(dict: &Bound<'_, PyDict>) -> PyResult<Option<GlmFamily>> {
    let Some(tag) = get_string(dict, "family")? else { return Ok(None) };
    Ok(Some(match tag.as_str() {
        "binomial_logit" => GlmFamily::BinomialLogit,
        "binomial_probit" => GlmFamily::BinomialProbit,
        "gaussian_identity" => GlmFamily::GaussianIdentity,
        "poisson_log" => GlmFamily::PoissonLog,
        "negative_binomial" => GlmFamily::NegativeBinomial,
        other => {
            return Err(PyValueError::new_err(format!(
                "estimator_config['family'] {other:?} is not recognized; use binomial_logit|\
                 binomial_probit|gaussian_identity|poisson_log|negative_binomial",
            )));
        }
    }))
}

/// Build [`GlmOptions`] from the `glm_options` nested dict. `nb_alpha` (NB2 dispersion policy)
/// is intentionally not exposed yet — see the P5b report's gap list — so it always keeps
/// [`GlmOptions::default`]'s `MethodOfMoments` policy.
fn build_glm_options(dict: &Bound<'_, PyDict>) -> PyResult<Option<GlmOptions>> {
    let Some(v) = dict.get_item("glm_options")? else { return Ok(None) };
    let sub = v.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "estimator_config['glm_options'] must be a dict, got {}",
            type_name(&v)
        ))
    })?;
    for (key_obj, _val) in sub.iter() {
        let key: String = key_obj.extract().map_err(|_| {
            PyValueError::new_err("estimator_config['glm_options'] keys must be str")
        })?;
        if !GLM_OPTION_KEYS.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown estimator_config['glm_options'] key {key:?}; valid keys are: {}",
                GLM_OPTION_KEYS.join(", "),
            )));
        }
    }
    let mut opts = GlmOptions::default();
    if let Some(max_iter) = get_u32(sub, "max_iter")? {
        opts.max_iter = max_iter;
    }
    if let Some(tol) = get_f64(sub, "tol")? {
        opts.tol = tol;
    }
    if let Some(raw) = sub.get_item("ridge_on_separation")? {
        opts.ridge_on_separation = if raw.is_none() {
            None
        } else {
            Some(raw.extract::<f64>().map_err(|_| {
                PyValueError::new_err(
                    "estimator_config['glm_options']['ridge_on_separation'] must be a float or \
                     None",
                )
            })?)
        };
    }
    Ok(Some(opts))
}

/// Build the caller-configured [`EstimatorSpec`] for one of the ten estimators
/// [`ESTIMATOR_KEYS`] covers (excluding `rd.sharp`, handled separately in
/// [`parse_estimator_config`]). By the time this runs, every key in `dict` has already been
/// validated against `resolved_id`'s row in [`ESTIMATOR_KEYS`], so extracting a field this
/// estimator doesn't support is impossible — the field simply won't be present.
fn build_configured_spec(
    resolved_id: &str,
    dict: &Bound<'_, PyDict>,
    ambient_bootstrap: u32,
) -> PyResult<EstimatorSpec> {
    let bootstrap = get_u32(dict, "bootstrap_replicates")?.unwrap_or(ambient_bootstrap);
    let se_kind = build_se_kind(dict)?;
    let cluster_ids = get_u32_list(dict, "cluster_ids")?;
    let multiway_ids = get_u32_list_list(dict, "multiway_ids")?;
    let panel_times = get_i64_list(dict, "panel_times")?;
    let glm_options = build_glm_options(dict)?;

    Ok(match resolved_id {
        "linear.adjustment.ate" => {
            let mut est = LinearAdjustmentAte::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            if let Some(fit) = build_fit_kind(dict)? {
                est = est.with_fit_kind(fit);
            }
            est.into()
        }
        "propensity.weighting" => {
            let mut est = PropensityWeighting::new().with_bootstrap_replicates(bootstrap);
            if let Some(opts) = glm_options {
                est = est.with_glm_options(opts);
            }
            est.into()
        }
        "propensity.matching" => {
            let mut est = PropensityMatching::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            if let Some(opts) = glm_options {
                est = est.with_glm_options(opts);
            }
            if let Some(cal) = get_f64(dict, "caliper")? {
                est = est.with_caliper(cal);
            }
            if let Some(scale) = get_string(dict, "caliper_scale")? {
                // The caliper is a distance on whichever scale matching runs on; the 0.2
                // rule of thumb (Rosenbaum-Rubin 1985, Austin 2011) is a logit-scale
                // quantity, which is why `Logit` is the default.
                let parsed = match scale.as_str() {
                    "logit" => CaliperScale::Logit,
                    "raw" => CaliperScale::Raw,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "caliper_scale must be \"logit\" or \"raw\", got {other:?}"
                        )));
                    }
                };
                est = est.with_caliper_scale(parsed);
            }
            est.into()
        }
        "propensity.stratification" => {
            let mut est = PropensityStratification::new().with_bootstrap_replicates(bootstrap);
            if let Some(opts) = glm_options {
                est = est.with_glm_options(opts);
            }
            if let Some(n) = get_u32(dict, "n_strata")? {
                est = est.with_n_strata(n);
            }
            est.into()
        }
        "distance.matching" => {
            let mut est = DistanceMatching::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            if let Some(opts) = glm_options {
                est = est.with_glm_options(opts);
            }
            if let Some(cal) = get_f64(dict, "caliper")? {
                est = est.with_caliper(cal);
            }
            est.into()
        }
        "aipw" => {
            let mut est = AipwAte::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            if let Some(opts) = glm_options {
                est = est.with_glm_options(opts);
            }
            est.into()
        }
        "glm.adjustment" => {
            let mut est = GlmAdjustmentAte::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            if let Some(opts) = glm_options {
                est = est.with_glm_options(opts);
            }
            if let Some(fam) = build_family(dict)? {
                est = est.with_family(fam);
            }
            est.into()
        }
        "frontdoor.two_stage" => {
            let mut est = FrontDoorTwoStage::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(Arc::from(ids));
            }
            est.into()
        }
        "iv.wald" => {
            let mut est = WaldIv::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            est.into()
        }
        "iv.2sls" => {
            let mut est = TwoStageLeastSquares::new().with_bootstrap_replicates(bootstrap);
            if let Some(k) = se_kind {
                est = est.with_se_kind(k);
            }
            if let Some(ids) = cluster_ids {
                est = est.with_cluster_ids(ids);
            }
            if let Some(mw) = multiway_ids {
                est = est.with_multiway_ids(mw);
            }
            if let Some(pt) = panel_times {
                est = est.with_panel_times(pt);
            }
            est.into()
        }
        other => {
            // Unreachable in practice: `resolved_id` already passed `valid_keys_for` above,
            // and every non-`rd.sharp` row in `ESTIMATOR_KEYS` has a matching arm here.
            return Err(PyValueError::new_err(format!(
                "internal error: estimator_config resolved id {other:?} is not table-driven"
            )));
        }
    })
}

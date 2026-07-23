//! Prior-bank catalog bindings (P4A).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use causal_io::{
    CompatibilityReport, DesignVariableRole, DesignVariableSummary, EstimandFingerprint,
    PriorCatalog, PriorMapping, PriorSourceMeta, PriorSourceRef, TargetDesign,
};
use causal_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
    compose_external_priors, compose_external_priors_with_alphas,
};
use causal_validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::py_err;

fn role_from_str(s: &str) -> PyResult<DesignVariableRole> {
    match s {
        "treatment" => Ok(DesignVariableRole::Treatment),
        "outcome" => Ok(DesignVariableRole::Outcome),
        "covariate" => Ok(DesignVariableRole::Covariate),
        "other" => Ok(DesignVariableRole::Other),
        other => Err(PyValueError::new_err(format!("unknown design role `{other}`"))),
    }
}

fn role_to_str(r: DesignVariableRole) -> &'static str {
    match r {
        DesignVariableRole::Treatment => "treatment",
        DesignVariableRole::Outcome => "outcome",
        DesignVariableRole::Covariate => "covariate",
        DesignVariableRole::Other => "other",
    }
}

pub(crate) fn mapping_from_dict(d: &Bound<'_, PyDict>) -> PyResult<PriorMapping> {
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("mapping requires kind"))?
        .extract()?;
    match kind.as_str() {
        "identical_coefficient_subspace" | "IdenticalCoefficientSubspace" => {
            Ok(PriorMapping::IdenticalCoefficientSubspace)
        }
        "effect_functional" | "EffectFunctional" => {
            let q: String = d
                .get_item("source_quantity")?
                .ok_or_else(|| PyValueError::new_err("EffectFunctional requires source_quantity"))?
                .extract()?;
            Ok(PriorMapping::EffectFunctional { source_quantity: q })
        }
        "named_parameters" | "NamedParameters" => {
            let pairs_obj = d
                .get_item("pairs")?
                .ok_or_else(|| PyValueError::new_err("NamedParameters requires pairs"))?;
            let pairs: Vec<(String, String)> = pairs_obj.extract()?;
            Ok(PriorMapping::NamedParameters { pairs })
        }
        other => Err(PyValueError::new_err(format!("unknown PriorMapping kind `{other}`"))),
    }
}

fn meta_from_dict(d: &Bound<'_, PyDict>) -> PyResult<PriorSourceMeta> {
    let artifact_id: String = d
        .get_item("artifact_id")?
        .ok_or_else(|| PyValueError::new_err("meta.artifact_id required"))?
        .extract()?;
    let estimand_obj =
        d.get_item("estimand")?.ok_or_else(|| PyValueError::new_err("meta.estimand required"))?;
    let estimand_d = estimand_obj.cast::<PyDict>()?;
    let estimand = EstimandFingerprint::new(
        estimand_d
            .get_item("query_kind")?
            .ok_or_else(|| PyValueError::new_err("estimand.query_kind required"))?
            .extract::<String>()?,
        estimand_d
            .get_item("treatment")?
            .ok_or_else(|| PyValueError::new_err("estimand.treatment required"))?
            .extract::<String>()?,
        estimand_d
            .get_item("outcome")?
            .ok_or_else(|| PyValueError::new_err("estimand.outcome required"))?
            .extract::<String>()?,
    );
    let identification: String = d
        .get_item("identification")?
        .ok_or_else(|| PyValueError::new_err("meta.identification required"))?
        .extract()?;
    let mut meta = PriorSourceMeta::new(artifact_id, estimand, identification);

    if let Some(tags_obj) = d.get_item("tags")? {
        let tags: BTreeMap<String, String> = tags_obj.extract()?;
        meta = meta.with_tags(tags);
    }
    if let Some(design_obj) = d.get_item("design")? {
        let rows: Vec<Bound<'_, PyAny>> = design_obj.extract()?;
        let mut design = Vec::with_capacity(rows.len());
        for row in rows {
            let rd = row.cast::<PyDict>()?;
            let name: String = rd
                .get_item("name")?
                .ok_or_else(|| PyValueError::new_err("design entry needs name"))?
                .extract()?;
            let role_s: String = rd
                .get_item("role")?
                .ok_or_else(|| PyValueError::new_err("design entry needs role"))?
                .extract()?;
            design.push(DesignVariableSummary::new(name, role_from_str(&role_s)?));
        }
        meta = meta.with_design(design);
    }
    if let Some(c) = d.get_item("contrast")? {
        if !c.is_none() {
            meta.contrast = Some(c.extract()?);
        }
    }
    if let Some(p) = d.get_item("provenance")? {
        meta.provenance = p.extract()?;
    }
    if let Some(m) = d.get_item("declared_mapping")? {
        if !m.is_none() {
            meta = meta.with_mapping(mapping_from_dict(m.cast::<PyDict>()?)?);
        }
    }
    Ok(meta)
}

fn report_to_dict<'py>(
    py: Python<'py>,
    report: &CompatibilityReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match report {
        CompatibilityReport::Compatible { artifact_id } => {
            d.set_item("status", "compatible")?;
            d.set_item("artifact_id", artifact_id.clone())?;
        }
        CompatibilityReport::Partial { artifact_id, missing, mappable } => {
            d.set_item("status", "partial")?;
            d.set_item("artifact_id", artifact_id.clone())?;
            d.set_item("missing", missing.clone())?;
            d.set_item("mappable", mappable.clone())?;
        }
        CompatibilityReport::Rejected { artifact_id, reason } => {
            d.set_item("status", "rejected")?;
            d.set_item("artifact_id", artifact_id.clone())?;
            let reason_s =
                serde_json::to_string(reason).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let json = py.import("json")?;
            let reason_d = json.call_method1("loads", (reason_s,))?;
            d.set_item("reason", reason_d)?;
        }
    }
    Ok(d)
}

/// Filter prior-bank sources for compatibility with a target design.
///
/// `sources` is a list of dicts `{meta: {...}, artifact: optional bytes}`.
/// `target` is a dict with estimand / variables / tags / allow_unidentified.
#[pyfunction]
fn prior_catalog_filter<'py>(
    py: Python<'py>,
    sources: &Bound<'_, PyList>,
    target: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let mut refs = Vec::with_capacity(sources.len());
    for item in sources.iter() {
        let d = item.cast::<PyDict>()?;
        let meta_obj = d
            .get_item("meta")?
            .ok_or_else(|| PyValueError::new_err("source requires meta dict"))?;
        let meta = meta_from_dict(meta_obj.cast::<PyDict>()?)?;
        let artifact = match d.get_item("artifact")? {
            Some(a) if !a.is_none() => {
                let bytes: Vec<u8> = a.extract()?;
                Some(bytes)
            }
            _ => None,
        };
        refs.push(match artifact {
            Some(b) => PriorSourceRef::with_bytes(meta, b),
            None => PriorSourceRef::from_meta(meta),
        });
    }

    let estimand_obj = target
        .get_item("estimand")?
        .ok_or_else(|| PyValueError::new_err("target.estimand required"))?;
    let ed = estimand_obj.cast::<PyDict>()?;
    let estimand = EstimandFingerprint::new(
        ed.get_item("query_kind")?
            .ok_or_else(|| PyValueError::new_err("estimand.query_kind"))?
            .extract::<String>()?,
        ed.get_item("treatment")?
            .ok_or_else(|| PyValueError::new_err("estimand.treatment"))?
            .extract::<String>()?,
        ed.get_item("outcome")?
            .ok_or_else(|| PyValueError::new_err("estimand.outcome"))?
            .extract::<String>()?,
    );
    let variables: Vec<String> = match target.get_item("variables")? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    let mut td = TargetDesign::new(estimand, variables);
    if let Some(tags) = target.get_item("tags")? {
        td = td.with_tags(tags.extract()?);
    }
    if let Some(allow) = target.get_item("allow_unidentified")? {
        if allow.extract::<bool>()? {
            td = td.allow_unidentified();
        }
    }

    let catalog = PriorCatalog::from_sources(refs);
    let reports = catalog.filter_compatible(&td);
    let out = PyList::empty(py);
    for r in &reports {
        out.append(report_to_dict(py, r)?)?;
    }
    Ok(out)
}

/// Rank usable compatibility reports by similarity scores `{id: score}`.
#[pyfunction]
fn prior_catalog_rank<'py>(
    py: Python<'py>,
    reports: &Bound<'_, PyList>,
    scores: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let mut parsed = Vec::with_capacity(reports.len());
    for item in reports.iter() {
        let d = item.cast::<PyDict>()?;
        let status: String = d
            .get_item("status")?
            .ok_or_else(|| PyValueError::new_err("report.status required"))?
            .extract()?;
        let artifact_id: String = d
            .get_item("artifact_id")?
            .ok_or_else(|| PyValueError::new_err("report.artifact_id required"))?
            .extract()?;
        let report = match status.as_str() {
            "compatible" => CompatibilityReport::Compatible { artifact_id },
            "partial" => CompatibilityReport::Partial {
                artifact_id,
                missing: d
                    .get_item("missing")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default(),
                mappable: d
                    .get_item("mappable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default(),
            },
            "rejected" => CompatibilityReport::Rejected {
                artifact_id,
                reason: causal_io::CompatibilityRejectReason::ArtifactUnreadable {
                    message: "rank ignores rejected detail".into(),
                },
            },
            other => {
                return Err(PyValueError::new_err(format!("unknown report status `{other}`")));
            }
        };
        parsed.push(report);
    }
    let score_map: BTreeMap<String, f64> = scores.extract()?;
    let score_vec: Vec<(String, f64)> = score_map.into_iter().collect();
    let catalog = PriorCatalog::new();
    let ranked = catalog.rank(&parsed, &score_vec);
    let out = PyList::empty(py);
    for r in &ranked {
        out.append(report_to_dict(py, r)?)?;
    }
    Ok(out)
}

/// Encode prior-source meta dict → CBOR bytes.
#[pyfunction]
fn encode_prior_source_meta(meta: &Bound<'_, PyDict>) -> PyResult<Vec<u8>> {
    let m = meta_from_dict(meta)?;
    causal_io::encode_prior_source_meta(&m).map_err(py_err)
}

/// Decode prior-source meta CBOR → dict.
#[pyfunction]
fn decode_prior_source_meta(py: Python<'_>, bytes: Vec<u8>) -> PyResult<Bound<'_, PyDict>> {
    let meta = causal_io::decode_prior_source_meta(&bytes).map_err(py_err)?;
    let d = PyDict::new(py);
    d.set_item("artifact_id", meta.artifact_id)?;
    let est = PyDict::new(py);
    est.set_item("query_kind", meta.estimand.query_kind)?;
    est.set_item("treatment", meta.estimand.treatment)?;
    est.set_item("outcome", meta.estimand.outcome)?;
    d.set_item("estimand", est)?;
    d.set_item("identification", meta.identification)?;
    d.set_item("tags", meta.tags)?;
    let design = PyList::empty(py);
    for row in meta.design {
        let rd = PyDict::new(py);
        rd.set_item("name", row.name)?;
        rd.set_item("role", role_to_str(row.role))?;
        design.append(rd)?;
    }
    d.set_item("design", design)?;
    if let Some(c) = meta.contrast {
        d.set_item("contrast", c)?;
    }
    if !meta.provenance.is_empty() {
        d.set_item("provenance", meta.provenance)?;
    }
    if let Some(m) = meta.declared_mapping {
        let md = PyDict::new(py);
        match m {
            PriorMapping::IdenticalCoefficientSubspace => {
                md.set_item("kind", "identical_coefficient_subspace")?;
            }
            PriorMapping::EffectFunctional { source_quantity } => {
                md.set_item("kind", "effect_functional")?;
                md.set_item("source_quantity", source_quantity)?;
            }
            PriorMapping::NamedParameters { pairs } => {
                md.set_item("kind", "named_parameters")?;
                md.set_item("pairs", pairs)?;
            }
        }
        d.set_item("declared_mapping", md)?;
    }
    Ok(d)
}

fn prior_set_from_moments(mean: &[f64], variance: &[f64]) -> PyResult<PriorSet> {
    if mean.len() != variance.len() || mean.is_empty() {
        return Err(PyValueError::new_err(
            "prior mean/variance must be non-empty and equal length",
        ));
    }
    let coef = GaussianCoefficientPrior {
        mean: std::sync::Arc::from(mean.to_vec()),
        variance: std::sync::Arc::from(variance.to_vec()),
    };
    coef.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(coef));
    Ok(prior)
}

fn source_from_dict(d: &Bound<'_, PyDict>) -> PyResult<ExternalPriorSource> {
    let id: String =
        d.get_item("id")?.ok_or_else(|| PyValueError::new_err("source.id required"))?.extract()?;
    let mean: Vec<f64> = d
        .get_item("mean")?
        .ok_or_else(|| PyValueError::new_err("source.mean required"))?
        .extract()?;
    let variance: Vec<f64> = d
        .get_item("variance")?
        .ok_or_else(|| PyValueError::new_err("source.variance required"))?
        .extract()?;
    let alpha: f64 = d
        .get_item("alpha")?
        .ok_or_else(|| PyValueError::new_err("source.alpha required"))?
        .extract()?;
    let mixture_weight: Option<f64> = match d.get_item("mixture_weight")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(v.extract()?),
    };
    let weight = ExternalPriorWeight::new(alpha, mixture_weight)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(ExternalPriorSource {
        id: std::sync::Arc::from(id),
        prior: prior_set_from_moments(&mean, &variance)?,
        weight,
    })
}

/// Parse external prior sources from a Python list of dicts (shared with analyze).
pub(crate) fn sources_from_list(list: &Bound<'_, PyList>) -> PyResult<Vec<ExternalPriorSource>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d = item.cast::<PyDict>()?;
        out.push(source_from_dict(d)?);
    }
    Ok(out)
}

/// Optional conflict policy from a dict `{p_min, kl_scale}`.
pub(crate) fn conflict_policy_from_dict(
    d: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<ConflictPolicy>> {
    let Some(d) = d else {
        return Ok(None);
    };
    let p_min: f64 = d.get_item("p_min")?.map(|v| v.extract()).transpose()?.unwrap_or(0.05);
    let kl_scale: f64 = d.get_item("kl_scale")?.map(|v| v.extract()).transpose()?.unwrap_or(1.0);
    Ok(Some(
        ConflictPolicy::try_new(p_min, kl_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    ))
}

fn composed_to_dict<'py>(
    py: Python<'py>,
    composed: &causal_prob::ComposedPrior,
) -> PyResult<Bound<'py, PyDict>> {
    let coef = composed
        .prior
        .gaussian_coefficients()
        .ok_or_else(|| PyValueError::new_err("composed prior missing coefficients"))?;
    let d = PyDict::new(py);
    d.set_item("mean", coef.mean.as_ref())?;
    d.set_item("variance", coef.variance.as_ref())?;
    d.set_item(
        "source_ids",
        composed.source_ids.iter().map(|s| s.as_ref().to_string()).collect::<Vec<_>>(),
    )?;
    d.set_item("alphas_requested", composed.alphas_requested.as_ref())?;
    d.set_item("alphas_applied", composed.alphas_applied.as_ref())?;
    let weights: Vec<Option<f64>> = composed.mixture_weights.iter().copied().collect();
    d.set_item("mixture_weights", weights)?;
    let assumption_ids: Vec<String> =
        composed.prior.restrictions.iter().map(|r| r.id.as_ref().to_string()).collect();
    d.set_item("assumption_ids", assumption_ids)?;
    Ok(d)
}

fn transport_policy_from_str(s: &str) -> PyResult<causal_prob::TransportPolicy> {
    causal_prob::TransportPolicy::parse(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn conflict_signals_from_list(
    list: Option<&Bound<'_, PyList>>,
    n: usize,
) -> PyResult<Vec<ConflictSignals>> {
    let Some(list) = list else {
        return Ok(vec![ConflictSignals::default(); n]);
    };
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d = item.cast::<PyDict>()?;
        let p_value: Option<f64> = match d.get_item("p_value")? {
            None => None,
            Some(v) if v.is_none() => None,
            Some(v) => Some(v.extract()?),
        };
        let kl: Option<f64> = match d.get_item("kl")? {
            None => None,
            Some(v) if v.is_none() => None,
            Some(v) => Some(v.extract()?),
        };
        out.push(ConflictSignals { p_value, kl });
    }
    Ok(out)
}

fn sources_to_py_list<'py>(
    py: Python<'py>,
    sources: &[ExternalPriorSource],
) -> PyResult<Bound<'py, PyList>> {
    let src_list = PyList::empty(py);
    for s in sources {
        let sd = PyDict::new(py);
        sd.set_item("id", s.id.as_ref())?;
        let coef = s
            .prior
            .gaussian_coefficients()
            .ok_or_else(|| PyValueError::new_err("source missing coefficients"))?;
        sd.set_item("mean", coef.mean.as_ref())?;
        sd.set_item("variance", coef.variance.as_ref())?;
        sd.set_item("alpha", s.weight.alpha)?;
        sd.set_item("mixture_weight", s.weight.mixture_weight)?;
        src_list.append(sd)?;
    }
    Ok(src_list)
}

fn with_conflict_fields(
    d: &Bound<'_, PyDict>,
    summary: &causal_prob::ConflictSummary,
) -> PyResult<()> {
    d.set_item("alphas_applied", summary.alphas_applied.as_ref())?;
    d.set_item(
        "conflict_p_values",
        summary.p_values.iter().copied().collect::<Vec<Option<f64>>>(),
    )?;
    d.set_item(
        "conflict_kl_values",
        summary.kl_values.iter().copied().collect::<Vec<Option<f64>>>(),
    )?;
    Ok(())
}

fn compose_transport_path<'py>(
    py: Python<'py>,
    srcs: &[ExternalPriorSource],
    baseline: &PriorSet,
    conflict: Option<&Bound<'py, PyDict>>,
    conflict_signals: Option<&Bound<'py, PyList>>,
    transport: Option<&str>,
    target_population: Option<&str>,
    source_populations: Option<Vec<Option<String>>>,
    unit_effects: Option<Vec<f64>>,
    transport_weights: Option<Vec<f64>>,
    coef_index: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use causal_prob::{
        TransportAdjustment, TransportContext, TransportPolicy, compose_with_transport,
    };

    let pop_owned: Vec<Option<String>> = match source_populations {
        Some(v) => {
            if v.len() != srcs.len() {
                return Err(PyValueError::new_err("source_populations length must match sources"));
            }
            v
        }
        None => vec![None; srcs.len()],
    };
    let pop_refs: Vec<Option<&str>> = pop_owned.iter().map(|o| o.as_deref()).collect();
    let policy: Option<TransportPolicy> = match transport {
        Some(s) => Some(transport_policy_from_str(s)?),
        None => None,
    };
    let adjustment = match (unit_effects, transport_weights) {
        (None, None) => None,
        (Some(e), Some(w)) => Some(
            TransportAdjustment::new(e, w).map_err(|err| PyValueError::new_err(err.to_string()))?,
        ),
        _ => {
            return Err(PyValueError::new_err(
                "unit_effects and transport_weights must be provided together",
            ));
        }
    };
    let ctx = TransportContext {
        source_populations: &pop_refs,
        target_population,
        policy,
        adjustment: adjustment.as_ref(),
        coef_index,
    };
    let (composed, _) = compose_with_transport(srcs, baseline, &ctx)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (prepared, _) = causal_prob::apply_transport(srcs, &ctx)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let prepared: Vec<ExternalPriorSource> = prepared
        .into_iter()
        .zip(composed.alphas_applied.iter())
        .map(|(mut s, &a)| {
            s.weight.alpha = a;
            s
        })
        .collect();
    let src_list = sources_to_py_list(py, &prepared)?;

    if let Some(c) = conflict {
        let policy_c = conflict_policy_from_dict(Some(c))?
            .ok_or_else(|| PyValueError::new_err("conflict dict invalid"))?;
        let signals = conflict_signals_from_list(conflict_signals, prepared.len())?;
        let (composed2, summary) =
            apply_conflict_and_compose(&prepared, baseline, &policy_c, &signals)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let d2 = composed_to_dict(py, &composed2)?;
        with_conflict_fields(&d2, &summary)?;
        d2.set_item("sources", src_list)?;
        d2.set_item("conflict", c)?;
        return Ok(d2);
    }
    let d = composed_to_dict(py, &composed)?;
    d.set_item("sources", src_list)?;
    Ok(d)
}

#[pyfunction(name = "compose_external_priors")]
#[pyo3(signature = (
    sources,
    baseline_mean,
    baseline_variance,
    *,
    conflict=None,
    conflict_signals=None,
    transport=None,
    target_population=None,
    source_populations=None,
    unit_effects=None,
    transport_weights=None,
    coef_index=None,
))]
#[allow(clippy::too_many_arguments)]
fn compose_external_priors_py<'py>(
    py: Python<'py>,
    sources: &Bound<'py, PyList>,
    baseline_mean: Vec<f64>,
    baseline_variance: Vec<f64>,
    conflict: Option<&Bound<'py, PyDict>>,
    conflict_signals: Option<&Bound<'py, PyList>>,
    transport: Option<&str>,
    target_population: Option<&str>,
    source_populations: Option<Vec<Option<String>>>,
    unit_effects: Option<Vec<f64>>,
    transport_weights: Option<Vec<f64>>,
    coef_index: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let srcs = sources_from_list(sources)?;
    let baseline = prior_set_from_moments(&baseline_mean, &baseline_variance)?;

    let needs_transport = transport.is_some()
        || target_population.is_some()
        || source_populations.is_some()
        || unit_effects.is_some()
        || transport_weights.is_some();
    if needs_transport {
        return compose_transport_path(
            py,
            &srcs,
            &baseline,
            conflict,
            conflict_signals,
            transport,
            target_population,
            source_populations,
            unit_effects,
            transport_weights,
            coef_index,
        );
    }

    let policy = conflict_policy_from_dict(conflict)?;
    if let Some(policy) = policy {
        let signals = conflict_signals_from_list(conflict_signals, srcs.len())?;
        let (composed, summary) = apply_conflict_and_compose(&srcs, &baseline, &policy, &signals)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let d = composed_to_dict(py, &composed)?;
        with_conflict_fields(&d, &summary)?;
        d.set_item("sources", sources)?;
        if let Some(c) = conflict {
            d.set_item("conflict", c)?;
        }
        return Ok(d);
    }
    let composed = compose_external_priors(&srcs, &baseline)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = composed_to_dict(py, &composed)?;
    d.set_item("sources", sources)?;
    Ok(d)
}

#[pyfunction(name = "conflict_shrink_alpha")]
#[pyo3(signature = (alpha, *, p_value=None, kl=None, p_min=0.05, kl_scale=1.0))]
fn shrink_alpha_py(
    alpha: f64,
    p_value: Option<f64>,
    kl: Option<f64>,
    p_min: f64,
    kl_scale: f64,
) -> PyResult<f64> {
    let policy = ConflictPolicy::try_new(p_min, kl_scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(policy.shrink_alpha(alpha, p_value, kl))
}

/// Owned payload for applying a composed external prior after GIL release.
pub(crate) struct OwnedComposedPrior {
    /// Hydrated external sources.
    pub sources: Vec<ExternalPriorSource>,
    /// Requested alphas.
    pub alphas_requested: Vec<f64>,
    /// Applied alphas (after optional offline conflict shrink).
    pub alphas_applied: Vec<f64>,
    /// Optional conflict policy for data-bound re-shrink.
    pub conflict: Option<ConflictPolicy>,
    /// Restriction ids to preserve (e.g. transport assumptions).
    pub assumption_ids: Vec<String>,
}

/// Parse a composed-prior dict into an owned payload.
pub(crate) fn owned_composed_prior_from_dict(
    composed: &Bound<'_, PyDict>,
) -> PyResult<OwnedComposedPrior> {
    let sources_obj = composed
        .get_item("sources")?
        .ok_or_else(|| PyValueError::new_err("composed_prior.sources required"))?;
    let sources_list = sources_obj.cast::<PyList>()?;
    let sources = sources_from_list(sources_list)?;
    let alphas_requested: Vec<f64> = match composed.get_item("alphas_requested")? {
        Some(v) => v.extract()?,
        None => sources.iter().map(|s| s.weight.alpha).collect(),
    };
    let alphas_applied: Vec<f64> = match composed.get_item("alphas_applied")? {
        Some(v) => v.extract()?,
        None => alphas_requested.clone(),
    };
    let conflict = match composed.get_item("conflict")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => conflict_policy_from_dict(Some(v.cast::<PyDict>()?))?,
    };
    let assumption_ids: Vec<String> = match composed.get_item("assumption_ids")? {
        None => Vec::new(),
        Some(v) if v.is_none() => Vec::new(),
        Some(v) => v.extract()?,
    };
    Ok(OwnedComposedPrior { sources, alphas_requested, alphas_applied, conflict, assumption_ids })
}

/// Build [`BayesianConfig`] external compose from owned composed-prior data.
pub(crate) fn apply_owned_composed_prior(
    cfg: antecedent::BayesianConfig,
    owned: OwnedComposedPrior,
) -> PyResult<antecedent::BayesianConfig> {
    use causal_core::PriorAssumption;
    use std::sync::Arc;

    let baseline_n = owned
        .sources
        .first()
        .and_then(|s| s.prior.gaussian_coefficients().map(GaussianCoefficientPrior::len))
        .ok_or_else(|| PyValueError::new_err("composed prior sources empty"))?;
    let baseline = PriorSet::weakly_informative(baseline_n);
    let mut prior_composed = compose_external_priors_with_alphas(
        &owned.sources,
        &owned.alphas_requested,
        &owned.alphas_applied,
        &baseline,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    for id in &owned.assumption_ids {
        if prior_composed.prior.restrictions.iter().any(|r| r.id.as_ref() == id.as_str()) {
            continue;
        }
        prior_composed.prior.restrictions.push(PriorAssumption {
            id: Arc::from(id.as_str()),
            description: Arc::from(format!("Preserved prior-bank assumption `{id}`")),
        });
    }
    Ok(cfg.prior_from_composed(owned.sources, prior_composed, owned.conflict))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prior_catalog_filter, m)?)?;
    m.add_function(wrap_pyfunction!(prior_catalog_rank, m)?)?;
    m.add_function(wrap_pyfunction!(encode_prior_source_meta, m)?)?;
    m.add_function(wrap_pyfunction!(decode_prior_source_meta, m)?)?;
    m.add_function(wrap_pyfunction!(compose_external_priors_py, m)?)?;
    m.add_function(wrap_pyfunction!(shrink_alpha_py, m)?)?;
    Ok(())
}

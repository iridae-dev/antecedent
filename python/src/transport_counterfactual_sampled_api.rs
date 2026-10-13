//! Python bindings for sampled counterfactual transport.
//!
//! A stored counterfactual transport derivation is evaluated, without searching
//! again, on laws estimated from sampled count tables under a declared overlap
//! of the units behind the tables and a declared unseen-cell rule. The result
//! keeps four claims apart (structural identification, exact evaluation on the
//! estimated tables, the point estimate, statistical inference) and publishes
//! no interval: the delta-method standard error is labelled uncalibrated.
use crate::graphs::Admg;
use crate::transport_common::{error, execution_context, resolve};
use crate::transport_counterfactual_api::{
    PreparedCounterfactualTransportNative, io_refusal, regime_label, variable_name,
};
use crate::transport_z_api::to_py_json;
use antecedent_core::{EvidenceCatalog, SearchLimits};
use antecedent_estimate::counterfactual_transport_sampled::{
    DatasetOverlap, OverlapDeclaration, SampledAxis, SampledCounterfactualTransportEstimate,
    SampledRegimeCounts, SampledRegimeCountsParts, SampledTableKey, SampledTransportLimits,
    SampledTransportRequest, SamplingDesign, UnseenCellRule,
};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_io::sampled_counterfactual_transport_artifact::{
    SampledCounterfactualTransportArtifactWire, SampledCounterfactualTransportConsumeLimits,
    SampledCounterfactualTransportExpectedIdentity, consume_sampled_counterfactual_transport,
    sampled_counterfactual_transport_artifact,
};
use pyo3::prelude::*;
use std::sync::Arc;

/// Sampled count tables: a plug-in point with an uncalibrated standard error.
const SAMPLED_SCOPE: &str = "counterfactual_transport_sampled_counts_point_only";

/// One count-table key: `(population, regime label, concrete interventions)`.
type TableKeyTuple = (String, String, Vec<(String, f64)>);

fn level_bits(level: f64) -> u64 {
    (level + 0.0).to_bits()
}

fn estimate_json(
    estimate: &SampledCounterfactualTransportEstimate,
    names: &[String],
    catalog: &EvidenceCatalog,
) -> serde_json::Value {
    let key = |k: &SampledTableKey| {
        serde_json::json!({
            "population": k.population.as_ref(),
            "regime": regime_label(catalog, k.regime),
            "interventions": k.interventions.iter()
                .map(|(v, bits)| (variable_name(names, *v), serde_json::json!(f64::from_bits(*bits))))
                .collect::<serde_json::Map<_, _>>(),
        })
    };
    let point = estimate.point.as_ref().map(|p| {
        serde_json::json!({
            "probability": p.probability,
            "numerator": p.numerator,
            "denominator": p.denominator,
            "conditioning_mass": p.conditioning_mass,
            "conditioning_mass_checked": p.conditioning_mass_checked,
            "smoothed": p.smoothed,
        })
    });
    let envelope = estimate.envelope.as_ref().map(|e| {
        serde_json::json!({
            "lower": e.lower,
            "upper": e.upper,
            "unseen_mass": e.unseen_mass,
            "members": e.members,
            "defined_members": e.defined_members,
            "undefined_members": e.undefined_members,
            "tables_with_unseen_cells": e.tables_with_unseen_cells,
            "label": e.label,
        })
    });
    let standard_error = estimate.inference.standard_error.as_ref().map(|s| {
        serde_json::json!({
            "value": s.value,
            "label": s.label,
            "method": s.method,
            "gradient_relative_step": s.gradient_relative_step,
            "gradient_evaluations": s.gradient_evaluations,
            "datasets": s.datasets,
        })
    });
    serde_json::json!({
        "scope": SAMPLED_SCOPE,
        "labels": {
            "structural_identification": estimate.labels.structural_identification,
            "exact_finite_law_evaluation": estimate.labels.exact_finite_law_evaluation,
            "point_estimate": estimate.labels.point_estimate,
            "statistical_inference": estimate.labels.statistical_inference,
        },
        "derivation_identity": estimate.derivation_identity,
        "evidence_identity": estimate.evidence_identity,
        "point": point,
        "point_withheld_reason": estimate.point_withheld_reason,
        "envelope": envelope,
        "tables": estimate.tables.iter().map(|t| {
            let mut entry = key(&t.key);
            entry["content_identity"] = t.content_identity.clone().into();
            entry["total"] = t.total.into();
            entry["cells"] = t.cells.into();
            entry["observed_cells"] = t.observed_cells.into();
            entry["unseen_cells"] = t.unseen_cells.into();
            entry["structural_zeros"] = t.structural_zeros.into();
            entry["smoothed"] = t.smoothed.into();
            entry["pseudo_count"] = t.pseudo_count.into();
            entry["dataset"] = t.dataset.into();
            entry
        }).collect::<Vec<_>>(),
        "datasets": estimate.datasets.iter()
            .map(|group| group.iter().map(&key).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "unseen_rule": {
            "name": estimate.unseen_rule.name(),
            "parameter": estimate.unseen_rule.parameter(),
        },
        "repairs": {
            "averaged_factors": estimate.repairs.averaged_factors,
            "complements_clamped": estimate.repairs.complements_clamped,
            "probability_clamped": estimate.repairs.probability_clamped,
        },
        "cells_read": estimate.cells_read,
        "inference": {
            "method": estimate.inference.method,
            "standard_error": standard_error,
            "standard_error_withheld_reason": estimate.inference.standard_error_withheld_reason,
            "interval": {
                "available": false,
                "status": "withheld",
                "reason": estimate.inference.interval.code,
                "detail": estimate.inference.interval.detail,
                "calibration_reason": estimate.inference.interval.reason,
            },
        },
    })
}

/// A sampled estimate with the artifact that seals it.
#[pyclass(skip_from_py_object)]
struct SampledCounterfactualTransportResultNative {
    wire: SampledCounterfactualTransportArtifactWire,
    estimate: SampledCounterfactualTransportEstimate,
    names: Vec<String>,
    catalog: EvidenceCatalog,
}

#[pymethods]
impl SampledCounterfactualTransportResultNative {
    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(py, &estimate_json(&self.estimate, &self.names, &self.catalog))
    }

    /// The estimate as a `sampled_counterfactual_transport_v1` artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let bytes = self.wire.export().map_err(|e| {
            let (code, message) = e.refusal();
            io_refusal(code, message)
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    /// `(premises digest, data digest)`, independent of the bytes.
    fn identity(&self) -> (Option<String>, Option<String>) {
        let identity = self.wire.identity();
        (identity.premises_digest, identity.data_digest)
    }
}

// ---------------------------------------------------------------------------
// Parsing.
// ---------------------------------------------------------------------------

fn regime_of<'c>(
    catalog: &'c EvidenceCatalog,
    population: &str,
    label: &str,
) -> PyResult<&'c antecedent_core::EvidenceRegime> {
    catalog
        .regimes
        .iter()
        .find(|r| r.label.as_deref() == Some(label) && r.population.as_ref() == population)
        .ok_or_else(|| {
            crate::value_err(format!(
                "count table names an unknown population/regime ({population}, {label})"
            ))
        })
}

fn parse_count_table(
    table: &Bound<'_, PyAny>,
    catalog: &EvidenceCatalog,
    graph: &Admg,
) -> PyResult<SampledRegimeCounts> {
    let population: String = table.getattr("population")?.extract()?;
    let label: String = table.getattr("regime")?.extract()?;
    let regime = regime_of(catalog, &population, &label)?;
    let axes: Vec<(String, Vec<f64>)> = table.getattr("axes")?.extract()?;
    let axes = axes
        .into_iter()
        .map(|(name, levels)| Ok(SampledAxis { variable: resolve(&graph.names, &name)?, levels }))
        .collect::<PyResult<Vec<_>>>()?;
    let interventions: Vec<(String, f64)> = table.getattr("interventions")?.extract()?;
    let interventions = interventions
        .into_iter()
        .map(|(name, level)| Ok((resolve(&graph.names, &name)?, level)))
        .collect::<PyResult<Vec<_>>>()?;
    if interventions.len() != regime.interventions.len()
        || !interventions.iter().all(|(v, _)| regime.interventions.contains(v))
        || axes.iter().any(|a| !regime.measured.contains(&a.variable))
    {
        return Err(crate::value_err("count table disagrees with its evidence regime"));
    }
    let counts: Vec<i64> = table.getattr("counts")?.extract()?;
    let counts = counts
        .into_iter()
        .map(|count| {
            u64::try_from(count).map_err(|_| crate::value_err("counts must be non-negative"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let stratified_on: Vec<String> = table.getattr("stratified_on")?.extract()?;
    let design = if stratified_on.is_empty() {
        SamplingDesign::IndependentUnits
    } else {
        SamplingDesign::Stratified {
            variables: stratified_on
                .iter()
                .map(|name| resolve(&graph.names, name))
                .collect::<PyResult<Vec<_>>>()?,
        }
    };
    let zeros: Vec<i64> = table.getattr("structural_zeros")?.extract()?;
    let structural_zeros = zeros
        .into_iter()
        .map(|cell| {
            usize::try_from(cell).map_err(|_| crate::value_err("structural zeros are cell indices"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let snapshot: String = table.getattr("snapshot_identity")?.extract()?;
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(population.as_str()),
        regime: regime.id,
        interventions,
        axes,
        counts,
        snapshot_identity: Arc::from(snapshot.as_str()),
        design,
        structural_zeros,
    })
    .map_err(error)
}

fn table_key(
    key: &TableKeyTuple,
    catalog: &EvidenceCatalog,
    graph: &Admg,
) -> PyResult<SampledTableKey> {
    let (population, label, interventions) = key;
    let regime = regime_of(catalog, population, label)?;
    let mut levels = interventions
        .iter()
        .map(|(name, level)| Ok((resolve(&graph.names, name)?, level_bits(*level))))
        .collect::<PyResult<Vec<_>>>()?;
    levels.sort_by_key(|(v, _)| *v);
    Ok(SampledTableKey {
        population: Arc::from(population.as_str()),
        regime: regime.id,
        interventions: levels,
    })
}

fn parse_overlap(
    overlap: &(String, Vec<(TableKeyTuple, TableKeyTuple, String)>),
    catalog: &EvidenceCatalog,
    graph: &Admg,
) -> PyResult<OverlapDeclaration> {
    let relation = |name: &str| {
        DatasetOverlap::parse(name).ok_or_else(|| {
            crate::value_err(format!(
                "unknown overlap relation {name:?}; use independent, shared_units or unknown"
            ))
        })
    };
    let mut declaration = OverlapDeclaration::with_default(relation(&overlap.0)?);
    for (a, b, name) in &overlap.1 {
        declaration = declaration
            .with_pair(
                table_key(a, catalog, graph)?,
                table_key(b, catalog, graph)?,
                relation(name)?,
            )
            .map_err(error)?;
    }
    Ok(declaration)
}

fn parse_unseen_rule(rule: &(String, Option<f64>)) -> PyResult<UnseenCellRule> {
    let parameter = |what: &str| {
        rule.1.ok_or_else(|| crate::value_err(format!("the {what} rule needs a parameter")))
    };
    match rule.0.as_str() {
        "refuse" => Ok(UnseenCellRule::Refuse),
        "additive_smoothing" => {
            Ok(UnseenCellRule::AdditiveSmoothing { pseudo_count: parameter("additive_smoothing")? })
        }
        "unseen_mass_envelope" => Ok(UnseenCellRule::UnseenMassEnvelope {
            unseen_mass: parameter("unseen_mass_envelope")?,
        }),
        other => Err(crate::value_err(format!(
            "unknown unseen-cell rule {other:?}; use refuse, additive_smoothing or \
             unseen_mass_envelope"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Estimate the stored derivation of a prepared counterfactual transport from
/// sampled count tables.
///
/// `overlap` is `(default relation, [(table key, table key, relation)])`;
/// `unseen_rule` is `(name, parameter)`; limits are the declared sample-size,
/// sparsity and envelope bounds (`None` keeps the route's default).
#[pyfunction]
#[pyo3(signature=(prepared, tables, overlap, unseen_rule, *, min_table_total=None, max_unseen_fraction=None, max_envelope_evaluations=None, max_operations=None, max_depth=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
fn estimate_sampled_counterfactual_transport_native(
    py: Python<'_>,
    prepared: PyRef<'_, PreparedCounterfactualTransportNative>,
    tables: &Bound<'_, PyAny>,
    overlap: (String, Vec<(TableKeyTuple, TableKeyTuple, String)>),
    unseen_rule: (String, Option<f64>),
    min_table_total: Option<u64>,
    max_unseen_fraction: Option<f64>,
    max_envelope_evaluations: Option<u32>,
    max_operations: Option<usize>,
    max_depth: Option<usize>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<SampledCounterfactualTransportResultNative> {
    let mut parsed = Vec::new();
    for table in tables.try_iter()? {
        parsed.push(parse_count_table(&table?, &prepared.catalog, &prepared.graph)?);
    }
    let declaration = parse_overlap(&overlap, &prepared.catalog, &prepared.graph)?;
    let rule = parse_unseen_rule(&unseen_rule)?;
    let defaults = SampledTransportLimits::default();
    let evaluation = ExactEvaluationLimits::default();
    let limits = SampledTransportLimits {
        min_table_total: min_table_total.unwrap_or(defaults.min_table_total),
        max_unseen_fraction: max_unseen_fraction.unwrap_or(defaults.max_unseen_fraction),
        max_envelope_evaluations: max_envelope_evaluations
            .unwrap_or(defaults.max_envelope_evaluations),
        evaluation: ExactEvaluationLimits {
            operations: max_operations.unwrap_or(evaluation.operations),
            depth: max_depth.unwrap_or(evaluation.depth),
        },
    };
    let request =
        SampledTransportRequest { tables: parsed, overlap: declaration, unseen_rule: rule, limits };
    let memory = memory_bytes.or(prepared.memory_bytes);
    let inner = prepared.inner.clone();
    let names = prepared.graph.names.clone();
    let catalog = inner.derivation().catalog.clone();
    let (wire, estimate) = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory, cancel);
        sampled_counterfactual_transport_artifact(inner.derivation(), &request, &names, &ctx)
            .map_err(|e| {
                let (code, message) = e.refusal();
                io_refusal(code, message)
            })
    })?;
    Ok(SampledCounterfactualTransportResultNative {
        wire,
        estimate,
        names: prepared.graph.names.clone(),
        catalog,
    })
}

/// Independently recheck a sampled artifact under the consumer's limits:
/// re-decide, rebuild the counts and recompute every stored number bit for bit.
#[pyfunction]
#[pyo3(signature=(artifact, *, expected_premises_digest=None, expected_data_digest=None, max_search_operations=None, max_search_depth=None, max_operations=None, max_depth=None, max_tables=None, max_table_cells=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
fn consume_sampled_counterfactual_transport_artifact_native(
    py: Python<'_>,
    artifact: &[u8],
    expected_premises_digest: Option<String>,
    expected_data_digest: Option<String>,
    max_search_operations: Option<usize>,
    max_search_depth: Option<usize>,
    max_operations: Option<usize>,
    max_depth: Option<usize>,
    max_tables: Option<usize>,
    max_table_cells: Option<usize>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<SampledCounterfactualTransportResultNative> {
    let defaults = SampledCounterfactualTransportConsumeLimits::default();
    let limits = SampledCounterfactualTransportConsumeLimits {
        search: SearchLimits {
            operations: max_search_operations.unwrap_or(defaults.search.operations),
            depth: max_search_depth.unwrap_or(defaults.search.depth),
        },
        evaluation: ExactEvaluationLimits {
            operations: max_operations.unwrap_or(defaults.evaluation.operations),
            depth: max_depth.unwrap_or(defaults.evaluation.depth),
        },
        max_tables: max_tables.unwrap_or(defaults.max_tables),
        max_table_cells: max_table_cells.unwrap_or(defaults.max_table_cells),
        ..defaults
    };
    let expected = SampledCounterfactualTransportExpectedIdentity {
        premises_digest: expected_premises_digest,
        data_digest: expected_data_digest,
    };
    let bytes = artifact.to_vec();
    let consumed = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        consume_sampled_counterfactual_transport(&bytes, &expected, limits, &ctx).map_err(error)
    })?;
    let catalog = consumed.derivation.catalog.clone();
    let names = consumed.wire.variable_names.clone();
    Ok(SampledCounterfactualTransportResultNative {
        wire: consumed.wire,
        estimate: consumed.estimate,
        names,
        catalog,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SampledCounterfactualTransportResultNative>()?;
    module.add_function(wrap_pyfunction!(
        estimate_sampled_counterfactual_transport_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        consume_sampled_counterfactual_transport_artifact_native,
        module
    )?)?;
    Ok(())
}

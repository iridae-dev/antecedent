//! Original F18 artifact consumption feeding the original bank and point decision engines.
use std::collections::BTreeMap;

use antecedent::analysis::effect_constancy::EffectConstancyIdentity;
use antecedent::analysis::effect_constancy_consumers::{ConstancyConsumers, PriorPartitionBinding};
use antecedent_core::ScientificQuantity;
use antecedent_design::decision_artifact::{contract_from_json, mean_result_to_json};
use antecedent_io::prior_bank::{
    EstimandFingerprint, PriorCatalog, PriorSourceMeta, PriorSourceRef, TargetDesign,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;

use crate::{detach_catch, value_err, with_reason_code};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceWire {
    meta: PriorSourceMeta,
    artifact: Option<Vec<u8>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetWire {
    estimand: EstimandFingerprint,
    variables: Vec<String>,
    tags: BTreeMap<String, String>,
    allow_unidentified: bool,
}
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum ReviewWire {
    Transport {
        left: String,
        right: String,
    },
    Prior {
        sources: Vec<SourceWire>,
        target: TargetWire,
        target_partition: String,
        bindings: Vec<PriorPartitionBinding>,
    },
    Policy {
        contract: serde_json::Value,
        effect: ScientificQuantityWire,
    },
}
fn invalid(message: impl std::fmt::Display) -> PyErr {
    with_reason_code(
        value_err(format!("effect_constancy_consumer.invalid_request: {message}")),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

/// Recompute source evidence and invoke its explicitly declared downstream consumer.
#[pyfunction]
fn review_effect_constancy(
    py: Python<'_>,
    artifact: &Bound<'_, PyBytes>,
    identity_json: &str,
    request_json: &str,
) -> PyResult<String> {
    if artifact.as_bytes().len() > 64 * 1024 * 1024
        || request_json.len() > 64 * 1024 * 1024
        || identity_json.len() > 4096
    {
        return Err(invalid("byte limit exceeded"));
    }
    let expected: EffectConstancyIdentity = serde_json::from_str(identity_json).map_err(invalid)?;
    let request: ReviewWire = serde_json::from_str(request_json).map_err(invalid)?;
    let bytes = artifact.as_bytes().to_vec();
    detach_catch(py, move || {
        let consumer = ConstancyConsumers::consume(&bytes, &expected)
            .map_err(|error| with_reason_code(value_err(error.to_string()), error.code))?;
        let report = match request {
            ReviewWire::Transport { left, right } => {
                let out = consumer
                    .transport_diagnostic(&left, &right)
                    .map_err(|e| with_reason_code(value_err(e.to_string()), e.code))?;
                let c = out.contrast;
                serde_json::json!({"evidence": out.evidence, "separate_transport_identification_required": out.separate_transport_identification_required, "contrast": {"left": c.left, "right": c.right, "difference": c.difference, "standard_error": c.standard_error, "z": c.z, "p_value": c.p_value, "p_holm": c.p_holm, "rejected": c.rejected}})
            }
            ReviewWire::Prior { sources, target, target_partition, bindings } => {
                if sources.len() > 1024
                    || bindings.len() > 1024
                    || sources
                        .iter()
                        .any(|s| s.artifact.as_ref().is_some_and(|a| a.len() > 16 * 1024 * 1024))
                {
                    return Err(invalid("prior catalog exceeds bounds"));
                }
                let catalog = PriorCatalog::from_sources(
                    sources
                        .into_iter()
                        .map(|s| match s.artifact {
                            Some(bytes) => PriorSourceRef::with_bytes(s.meta, bytes),
                            None => PriorSourceRef::from_meta(s.meta),
                        })
                        .collect(),
                );
                let target = TargetDesign {
                    estimand: target.estimand,
                    variables: target.variables.into_iter().collect(),
                    tags: target.tags,
                    allow_unidentified: target.allow_unidentified,
                };
                let out = consumer
                    .rank_prior_sources(&catalog, &target, &target_partition, &bindings)
                    .map_err(|e| with_reason_code(value_err(e.to_string()), e.code))?;
                serde_json::to_value(out).map_err(invalid)?
            }
            ReviewWire::Policy { contract, effect } => {
                let effect =
                    ScientificQuantity::try_from(effect).map_err(|e| invalid(format!("{e:?}")))?;
                let contract = contract_from_json(&contract.to_string()).map_err(invalid)?;
                let out = consumer
                    .policy_review(&contract, &effect)
                    .map_err(|e| with_reason_code(value_err(e.to_string()), e.code))?;
                let partitions = out.partitions.into_iter().map(|p| {
                    let result = mean_result_to_json(&p.result, &p.source).map_err(invalid)?;
                    Ok(serde_json::json!({"label": p.label, "coordinate": p.coordinate, "mean": p.source.means[0], "result": serde_json::from_str::<serde_json::Value>(&result).map_err(invalid)?}))
                }).collect::<PyResult<Vec<_>>>()?;
                serde_json::json!({"evidence": out.evidence, "partitions": partitions, "common_leaders": out.common_leaders, "generalization_guarantee": out.generalization_guarantee})
            }
        };
        serde_json::to_string(&report).map_err(invalid)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(review_effect_constancy, module)?)
}

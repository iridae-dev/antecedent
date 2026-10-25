//! Execution-contract maps and transform previews.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::PreparedStudy;
use antecedent_core::ExecutionContext;
use pyo3::prelude::*;

use crate::response_api::{ResponseAnalysisResult, attach_study_response_meta, response_result};

pub(crate) fn response_from_study(
    names: &[String],
    result: &antecedent::StudyResult,
    prepared: &PreparedStudy,
    ctx: &ExecutionContext,
) -> PyResult<ResponseAnalysisResult> {
    response_from_retained(names, result, prepared, ctx)
}

pub(crate) fn response_from_retained(
    names: &[String],
    result: &antecedent::StudyResult,
    prepared: &antecedent::PreparedStudy,
    ctx: &ExecutionContext,
) -> PyResult<ResponseAnalysisResult> {
    let response = result.response().cloned().ok_or_else(|| {
        crate::value_err("prepared response estimate did not carry a response payload")
    })?;
    let name_of = |id: antecedent_core::VariableId| {
        names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
    };
    let treatments = result.treatment.map(name_of).into_iter().collect();
    let outcomes = vec![name_of(result.outcome)];
    let adjustment_set = result.estimand.adjustment_set.iter().copied().map(name_of).collect();
    Ok(attach_study_response_meta(
        response_result(
            response,
            result.structural_response.as_ref(),
            treatments,
            outcomes,
            adjustment_set,
            names,
            result.support_status,
        )?,
        crate::identification_details::analysis_to_json(result, names)?,
        result.logical_plan.identifier.as_deref().map(str::to_owned),
        result.diagnostics.iter().map(|d| format!("{}: {}", d.code, d.message)).collect(),
        crate::missing_report::report_json(result.missing_data.as_ref()),
    )
    .with_authority(prepared, prepared.schema(), result, ctx))
}

pub(crate) fn parse_transform_intent(
    intent: &str,
) -> PyResult<antecedent_contract::TransformIntent> {
    let intent = match intent {
        "display_precision" => antecedent_contract::TransformIntent::DisplayPrecision,
        "compatible_data_replace" => antecedent_contract::TransformIntent::CompatibleDataReplace,
        "retarget" => antecedent_contract::TransformIntent::Retarget,
        "filter_display" => antecedent_contract::TransformIntent::FilterDisplay,
        "filter_population" => antecedent_contract::TransformIntent::FilterPopulation,
        "new_conditional_query" => antecedent_contract::TransformIntent::NewConditionalQuery,
        "change_graph" => antecedent_contract::TransformIntent::ChangeGraph,
        "change_prior" => antecedent_contract::TransformIntent::ChangePrior,
        "change_physical_policy" => antecedent_contract::TransformIntent::ChangePhysicalPolicy,
        "average_unweighted_class" => antecedent_contract::TransformIntent::AverageUnweightedClass,
        other => {
            return Err(crate::value_err(format!("unknown transform intent {other:?}")));
        }
    };
    Ok(intent)
}
pub(crate) fn transform_report_map(
    report: &antecedent_contract::TransformationReport,
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    out.insert("intent".into(), report.intent.as_str().to_string());
    for identity in report.input_identities.iter() {
        out.insert(format!("input_{}", identity.domain.as_str()), identity.digest.to_hex());
    }
    out.insert("refused".into(), report.refused.to_string());
    if let Some(refusal) = &report.refusal {
        if let Some((code, _)) = antecedent_core::reason_code::split_prefix(refusal) {
            out.insert("refusal_code".into(), code.to_string());
        }
        out.insert("refusal".into(), refusal.to_string());
    }
    out.insert(
        "obligations".into(),
        report.obligations.iter().map(|o| o.id.to_string()).collect::<Vec<_>>().join(","),
    );
    out
}

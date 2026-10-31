//! The opt-in descriptive fallback, run by Rust inside the native prepare lifecycle.
//!
//! A causal query the declared structure does not identify is refused while the study is
//! prepared. When the caller declared `on_refusal="describe"`, the refused `Study` is not
//! dropped: [`Study::analyze_with_fallback`] runs on it, with a [`FallbackRequest`] built from
//! the declared candidate studies, and the refusal is raised unchanged with a [`FallbackStage`]
//! attached as its `fallback` attribute. The stage holds Rust's [`AnalysisWithFallback`]: the
//! compiled associational study and result, the verified obligations, the next actions, the
//! candidate templates and the ranked plan. Python reads those products; it composes none of them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::evidence_plan::EvidencePlanRequest;
use antecedent::analysis::fallback::{
    AnalysisWithFallback, FallbackOutcome, FallbackRequest, FallbackUnavailable, RankedPlan,
};
use antecedent::report::{ReportFormat, render_fallback_report};
use antecedent::{CausalError, Study};
use antecedent_core::{ExecutionContext, SearchLimits, VariableId};
use antecedent_design::{DurableStudyCandidate, RepairLimits, RepairObjective};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde_json::{Value, json};

use crate::evidence_plan_api::{EvidencePlanStage, generation_json};
use crate::repair_api::{obligation_json, parse_candidates};

/// What the caller declared for the fallback: the candidate studies and the plan bounds.
pub(crate) struct FallbackDeclaration {
    /// The declared candidate studies, spelled in variable names, as JSON; `None` asks for no
    /// ranked plan.
    candidates_json: Option<String>,
    objective: RepairObjective,
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
}

impl FallbackDeclaration {
    /// Parse the `fallback` entry of the prepare options.
    pub(crate) fn parse(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
        };
        let required = |key: &str| -> PyResult<Bound<'_, PyAny>> {
            item(key)?.ok_or_else(|| crate::value_err(format!("fallback option needs {key:?}")))
        };
        let objective = match required("objective")?.extract::<String>()?.as_str() {
            "minimize_cost" => RepairObjective::MinimizeCost,
            "minimize_sample_budget" => RepairObjective::MinimizeSampleBudget,
            other => {
                return Err(crate::value_err(format!(
                    "unknown objective {other:?}; use minimize_cost or minimize_sample_budget"
                )));
            }
        };
        Ok(Self {
            candidates_json: item("candidates")?.map(|value| value.extract()).transpose()?,
            objective,
            max_operations: required("max_operations")?.extract()?,
            max_depth: required("max_depth")?.extract()?,
            memory_bytes: item("memory_bytes")?.map(|value| value.extract()).transpose()?,
        })
    }

    /// The request Rust runs the fallback with. Candidates are spelled in the names of the
    /// failure (the study's variables, then the unmeasured common causes of the declared graph).
    fn request(&self, study: &Study) -> PyResult<(FallbackRequest, Vec<DurableStudyCandidate>)> {
        let (Some(json), Some(names)) = (&self.candidates_json, study.fallback_candidate_names())
        else {
            return Ok((FallbackRequest::default(), Vec::new()));
        };
        let candidates = parse_candidates(&names, json)?;
        let retained = candidates.clone();
        let limits = RepairLimits {
            search: SearchLimits { operations: self.max_operations, depth: self.max_depth },
            memory_limit_bytes: self
                .memory_bytes
                .unwrap_or_else(|| RepairLimits::default().memory_limit_bytes),
        };
        let evidence = EvidencePlanRequest {
            candidates,
            objective: self.objective,
            limits,
            sensitivity: None,
            value: None,
        };
        Ok((FallbackRequest { evidence: Some(evidence) }, retained))
    }
}

/// Rust's descriptive fallback of a refused study, read by Python.
#[pyclass(skip_from_py_object, name = "FallbackStage")]
pub(crate) struct FallbackStage {
    fallback: AnalysisWithFallback,
    names: Vec<String>,
    candidates: Vec<DurableStudyCandidate>,
    seed: u64,
    threads: u32,
    metadata_context: ExecutionContext,
}

fn unavailable(reason: &FallbackUnavailable) -> (String, String, String, String) {
    (
        reason.part.as_str().to_owned(),
        reason.code.clone(),
        reason.detail.clone(),
        reason.message.clone(),
    )
}

fn name_of(names: &[String], variable: VariableId) -> String {
    names.get(variable.as_usize()).cloned().unwrap_or_else(|| format!("v{}", variable.raw()))
}

#[pymethods]
impl FallbackStage {
    /// The roles of the description: outcome, treatment, held-fixed columns and the rule's name;
    /// `None` when the description is unavailable.
    fn roles(&self) -> Option<(String, String, Vec<String>, String)> {
        let part = self.fallback.associational.as_ref().ok()?;
        Some((
            name_of(&self.names, part.roles.outcome),
            name_of(&self.names, part.roles.treatment),
            part.roles.adjustment.iter().map(|v| name_of(&self.names, *v)).collect(),
            part.roles.rule.to_owned(),
        ))
    }

    /// The typed reason there is no associational description, when there is none.
    fn associational_unavailable(&self) -> Option<(String, String, String, String)> {
        self.fallback.associational.as_ref().err().map(unavailable)
    }

    /// The compiled associational study, completed, and its result.
    fn associational(
        &self,
    ) -> PyResult<Option<(crate::prepared_api::PyPreparedAnalysis, crate::AteAnalysisResult)>> {
        let Ok(part) = &self.fallback.associational else {
            return Ok(None);
        };
        let raw = crate::ate_result_from_analysis(
            &self.names,
            &part.result,
            false,
            &self.metadata_context,
        )?;
        let handle = crate::prepared_api::PyPreparedAnalysis::from_completed(
            Arc::new(part.prepared.clone()),
            Arc::new(part.result.clone()),
            self.names.clone(),
            self.seed,
            self.threads,
        );
        Ok(Some((handle, raw)))
    }

    /// The typed reason there are no evidence needs, when there are none.
    fn evidence_unavailable(&self) -> Option<(String, String, String, String)> {
        self.fallback.evidence.as_ref().err().map(unavailable)
    }

    /// The verified obligations, each with its proof step.
    fn obligations(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Ok(needs) = &self.fallback.evidence else {
            return Ok(None);
        };
        let values: Vec<Value> =
            needs.obligations.iter().map(|o| obligation_json(o, &needs.names)).collect();
        crate::transport_z_api::to_py_json(py, &Value::Array(values)).map(Some)
    }

    /// The evidence-gathering next actions the obligations imply.
    fn next_actions(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Ok(needs) = &self.fallback.evidence else {
            return Ok(None);
        };
        let values: Vec<Value> = needs
            .next_actions
            .iter()
            .map(|a| json!({ "id": a.id.as_ref(), "description": a.description.as_ref() }))
            .collect();
        crate::transport_z_api::to_py_json(py, &Value::Array(values)).map(Some)
    }

    /// The unfinished studies the obligations determine.
    fn templates(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Ok(needs) = &self.fallback.evidence else {
            return Ok(None);
        };
        crate::transport_z_api::to_py_json(
            py,
            &generation_json(&needs.candidate_templates, &needs.names),
        )
        .map(Some)
    }

    /// `not_requested`, `ranked` or `unavailable`; `None` without evidence needs.
    fn ranking_kind(&self) -> Option<&'static str> {
        let needs = self.fallback.evidence.as_ref().ok()?;
        Some(match &needs.ranking {
            RankedPlan::NotRequested => "not_requested",
            RankedPlan::Ranked(_) => "ranked",
            RankedPlan::Unavailable(_) => "unavailable",
        })
    }

    /// The ranked plan, as the stage a plan declared on a failure contract is.
    fn ranked_plan(&self) -> Option<EvidencePlanStage> {
        let needs = self.fallback.evidence.as_ref().ok()?;
        match &needs.ranking {
            RankedPlan::Ranked(result) => {
                Some(EvidencePlanStage::of_result(result, &self.candidates, &needs.observed))
            }
            _ => None,
        }
    }

    /// The report of the refusal beside its description and evidence plan, as `markdown` or
    /// `html`: Rust reads its own `AnalysisWithFallback` and writes the document.
    fn report(&self, format: &str) -> PyResult<String> {
        let format = ReportFormat::from_name(format).ok_or_else(|| {
            crate::value_err(format!("unknown report format {format:?}; use markdown or html"))
        })?;
        Ok(render_fallback_report(&self.fallback, format))
    }

    /// The typed reason the declared candidates could not be ranked.
    fn ranking_unavailable(&self) -> Option<(String, String, String, String)> {
        let needs = self.fallback.evidence.as_ref().ok()?;
        match &needs.ranking {
            RankedPlan::Unavailable(reason) => Some(unavailable(reason)),
            _ => None,
        }
    }
}

/// The error a refused preparation raises: the refusal unchanged and, when the caller declared a
/// description and the refusal is an identification refusal, Rust's fallback beside it.
pub(crate) fn describe_refusal(
    study: &Study,
    declaration: &FallbackDeclaration,
    names: &[String],
    seed: u64,
    threads: u32,
    ctx: &ExecutionContext,
    refusal: CausalError,
) -> PyErr {
    if !refusal.is_identification_refusal() {
        return crate::py_err(refusal);
    }
    let (request, candidates) = match declaration.request(study) {
        Ok(parts) => parts,
        Err(error) => return error,
    };
    match study.analyze_with_fallback(&request, ctx) {
        Ok(FallbackOutcome::Described(fallback)) => {
            let error = crate::py_err(refusal);
            let stage = FallbackStage {
                fallback: *fallback,
                names: names.to_vec(),
                candidates,
                seed,
                threads,
                metadata_context: ctx.clone(),
            };
            Python::attach(|py| {
                if let Ok(object) = Py::new(py, stage) {
                    let _ = error.value(py).setattr("fallback", object);
                }
            });
            error
        }
        Ok(_) => crate::py_err(refusal),
        Err(other) => crate::py_err(other),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FallbackStage>()?;
    Ok(())
}

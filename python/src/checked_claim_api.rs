//! Checked synthesis of conditional study claims over retained prepared studies.
//!
//! Each parent is a completed estimate: the retained prepared study and its exact retained
//! result, from which the parent's claim is minted here (never supplied by the caller). Python
//! declares the linear coefficients and the output weights; the output contrast is assembled
//! from the parents' own contrast worlds, and Rust verifies every program, target and
//! support identity, the coefficients against the declared target, and the arithmetic.
// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use crate::prepared_api::PyPreparedAnalysis;
use antecedent::analysis::PreparedStudy;
use antecedent::analysis::checked_claim_composition::{
    CheckedStudyClaim, PreparedClaimParent, compose_checked_study_claims,
};
use antecedent::contract::{
    CheckedWorldPointError, CheckedWorldPointLimits, ClaimEnvelope, ClaimOperation,
};
use antecedent::planner::lowered::ProblemTarget;
use antecedent::scientific::{
    InterventionContrast, InterventionWorld, ScientificTarget, ScientificWorldId, TargetForm,
    WorldCollection,
};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::json;

fn invalid(error: impl std::fmt::Display) -> PyErr {
    crate::value_err(error.to_string())
}

/// The caller's work and memory budgets (see `CheckedClaimLimits` in Python).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_field_names,
    reason = "the field names are the wire keys of CheckedClaimLimits"
)]
struct LimitsWire {
    max_target_work: usize,
    max_target_payload_bytes: usize,
    max_target_depth: usize,
    max_memory_bytes: u64,
    max_terms: usize,
    max_comparisons: usize,
    max_world_actions: usize,
    max_assumption_records: usize,
    max_evidence_references: usize,
    max_evidence_bytes: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    operation: String,
    /// The linear coefficient applied to each parent's point value.
    coefficients: Vec<f64>,
    /// The declared output weights: the output target is `sum(weight_i * (active_i - control))`.
    weights: Vec<f64>,
    limits: LimitsWire,
}

fn operation(name: &str) -> PyResult<ClaimOperation> {
    match name {
        "contrast" => Ok(ClaimOperation::Contrast),
        "aggregate" => Ok(ClaimOperation::Aggregate),
        other => {
            Err(invalid(format!("operation must be \"contrast\" or \"aggregate\", got {other:?}")))
        }
    }
}

fn composition_error(error: CheckedWorldPointError) -> PyErr {
    match error {
        CheckedWorldPointError::Cancelled => {
            crate::CausalCancelledError::new_err(error.to_string())
        }
        CheckedWorldPointError::Admission
        | CheckedWorldPointError::TermLimit
        | CheckedWorldPointError::EvidenceLimit => {
            crate::CausalResourceError::new_err(error.to_string())
        }
        _ => crate::value_err(error.to_string()),
    }
}

/// A parent's retained study, its minted claim and the study's own scientific target.
struct Parent {
    study: Arc<PreparedStudy>,
    claim: ClaimEnvelope,
    target: ScientificTarget,
}

fn scientific_target(study: &PreparedStudy) -> PyResult<ScientificTarget> {
    let lowered = study
        .plan()
        .lowered()
        .ok_or_else(|| invalid("a parent study carries no lowered scientific target"))?;
    match lowered.scientific().target() {
        ProblemTarget::Target(target) => Ok(target.clone()),
        _ => Err(invalid("a parent study does not declare one scalar scientific target")),
    }
}

/// Mint a parent from a completed estimate: the exact retained study and result.
fn parent(
    prepared: &PyPreparedAnalysis,
    ctx: &antecedent::prelude::ExecutionContext,
) -> PyResult<Parent> {
    let (study, retained) = prepared.target_refutation_source();
    let result = retained.ok_or_else(|| {
        invalid("checked composition requires a completed estimate of every parent study")
    })?;
    let target = scientific_target(&study)?;
    // A target that is no average-effect contrast refuses before any claim is minted.
    active_and_control(&target)?;
    let contract = study.contract_for_result(&result).map_err(crate::py_err)?;
    let claim = result.claim(&contract, ctx).map_err(crate::py_err)?;
    Ok(Parent { study, claim, target })
}

/// The `(active, control)` worlds of an average-effect contrast target.
fn active_and_control(
    target: &ScientificTarget,
) -> PyResult<(InterventionWorld, InterventionWorld)> {
    let TargetForm::Contrast(contrast) = target.form() else {
        return Err(invalid(
            "checked composition needs parents whose target is a world contrast (average effect)",
        ));
    };
    let world = |coefficient: f64| -> PyResult<InterventionWorld> {
        contrast
            .terms()
            .into_iter()
            .find(|term| term.coefficient.to_bits() == coefficient.to_bits())
            .and_then(|term| contrast.worlds().get(term.world))
            .cloned()
            .ok_or_else(|| {
                invalid("a parent target is not a unit-coefficient active-versus-control contrast")
            })
    };
    Ok((world(1.0)?, world(-1.0)?))
}

/// The declared output: `sum(weight_i * (active_i - control))` over the first parent's control.
fn output_target(parents: &[Parent], weights: &[f64]) -> PyResult<ScientificTarget> {
    let pairs =
        parents.iter().map(|p| active_and_control(&p.target)).collect::<PyResult<Vec<_>>>()?;
    let total: f64 = weights.iter().sum();
    let control = pairs[0].1.clone();
    let mut raw = vec![(control, -total)];
    raw.extend(pairs.into_iter().zip(weights).map(|((active, _), weight)| (active, *weight)));
    // A zero coefficient names no world of the contrast.
    raw.retain(|(_, coefficient)| coefficient.abs() > 0.0);
    let mut terms = Vec::new();
    let worlds = raw
        .into_iter()
        .enumerate()
        .map(|(index, (world, coefficient))| {
            let id = ScientificWorldId::from_raw(
                u32::try_from(index).map_err(|_| invalid("too many contrast worlds"))?,
            );
            terms.push((id, coefficient));
            InterventionWorld::new(
                id,
                world.action().clone(),
                index == 0,
                world.binding().clone(),
                world.time().clone(),
            )
            .map_err(invalid)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let contrast = InterventionContrast::linear(
        WorldCollection::new(worlds).map_err(invalid)?,
        terms,
        ScientificWorldId::from_raw(0),
    )
    .map_err(invalid)?;
    let first = &parents[0].target;
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        first.assumptions().clone(),
        first.inference(),
    ))
}

/// A checked, in-process conditional claim derived from prepared-study claims.
#[pyclass(name = "_CheckedStudyClaim", frozen)]
pub(crate) struct PyCheckedStudyClaim {
    inner: CheckedStudyClaim,
}

#[pymethods]
impl PyCheckedStudyClaim {
    #[getter]
    fn point(&self) -> f64 {
        self.inner.point()
    }

    /// Facts of the derived claim, its parents, proof, receipt and provenance as JSON.
    fn report_json(&self) -> String {
        let claim = self.inner.claim();
        let receipt = self.inner.receipt();
        let strings =
            |items: &[Arc<str>]| items.iter().map(ToString::to_string).collect::<Vec<_>>();
        let provenance = self.inner.provenance().validate();
        json!({
            "point": self.inner.point(),
            "claim_id": claim.claim_id.to_hex(),
            "kind": format!("{:?}", claim.kind),
            "parents": self.inner.parents().iter().map(antecedent::scientific::SemanticDigest::to_hex).collect::<Vec<_>>(),
            "proof_identity": self.inner.proof_identity().to_hex(),
            "target_identity": self.inner.target().identity().to_hex(),
            "domains": {
                "identification": format!("{:?}", claim.domains.identification),
                "support": format!("{:?}", claim.domains.support),
                "evaluated": format!("{:?}", claim.domains.evaluated),
            },
            "calibration": claim.calibration.status.to_string(),
            "receipt": {
                "rule": receipt.rule.to_string(),
                "consumer": receipt.consumer.to_string(),
                "output": receipt.output.map(|id| id.to_hex()),
                "retained": strings(&receipt.retained),
                "omitted": strings(&receipt.omitted),
                "unresolved": strings(&receipt.unresolved),
                "unavailable_operations": strings(&receipt.unavailable_operations),
            },
            "provenance": {
                "valid": provenance.is_ok(),
                "unresolved_parents": provenance
                    .as_ref()
                    .map(|ids| ids.iter().map(ToString::to_string).collect::<Vec<_>>())
                    .unwrap_or_default(),
            },
        })
        .to_string()
    }
}

/// Mint each completed parent's claim and compose them under the declared operation.
///
/// Refuses (a typed error) a parent without a completed estimate, targets that are not
/// average-effect contrasts, coefficients or output weights the parents do not realize, and
/// any claim whose kind, identification, support or evaluation does not support the point.
#[pyfunction]
#[pyo3(
    name = "compose_checked_study_claims",
    signature=(parents,declaration_json,*,max_bytes=None,cancel=None)
)]
fn compose_claims(
    py: Python<'_>,
    parents: Vec<PyRef<'_, PyPreparedAnalysis>>,
    declaration_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCheckedStudyClaim> {
    crate::bridge::check_declared_size(
        declaration_json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "checked claim composition declaration",
    )?;
    let declaration: Declaration = serde_json::from_str(declaration_json).map_err(invalid)?;
    let operation = operation(&declaration.operation)?;
    if parents.is_empty()
        || declaration.coefficients.len() != parents.len()
        || declaration.weights.len() != parents.len()
    {
        return Err(invalid(
            "coefficients and weights must each declare one value per parent (at least one parent)",
        ));
    }
    if declaration.coefficients.iter().chain(&declaration.weights).any(|v| !v.is_finite()) {
        return Err(invalid("coefficients and weights must be finite"));
    }
    let limits = CheckedWorldPointLimits {
        max_target_work: declaration.limits.max_target_work,
        max_target_payload_bytes: declaration.limits.max_target_payload_bytes,
        max_target_depth: declaration.limits.max_target_depth,
        max_memory_bytes: declaration.limits.max_memory_bytes,
        max_terms: declaration.limits.max_terms,
        max_comparisons: declaration.limits.max_comparisons,
        max_world_actions: declaration.limits.max_world_actions,
        max_assumption_records: declaration.limits.max_assumption_records,
        max_evidence_references: declaration.limits.max_evidence_references,
        max_evidence_bytes: declaration.limits.max_evidence_bytes,
    };
    let ctx = crate::transport_common::execution_context(0, max_bytes, cancel);
    if ctx.cancellation.is_cancelled() {
        return Err(composition_error(CheckedWorldPointError::Cancelled));
    }
    let minted =
        parents.iter().map(|prepared| parent(prepared, &ctx)).collect::<PyResult<Vec<_>>>()?;
    let target = output_target(&minted, &declaration.weights)?;
    let inner = crate::detach_catch(py, || {
        let claims: Vec<_> = minted
            .iter()
            .zip(&declaration.coefficients)
            .map(|(parent, coefficient)| PreparedClaimParent {
                prepared: &parent.study,
                claim: &parent.claim,
                coefficient: *coefficient,
            })
            .collect();
        compose_checked_study_claims(&target, &claims, operation, &limits, &ctx)
            .map_err(composition_error)
    })?;
    Ok(PyCheckedStudyClaim { inner })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCheckedStudyClaim>()?;
    module.add_function(wrap_pyfunction!(compose_claims, module)?)
}

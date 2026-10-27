//! Explicit next attempts when a finer panel explanation is not identified.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{DecompositionDeclaration, PyJointPanelResult, context, facade, panel_error};
use antecedent_core::ScientificWorldId;
use antecedent_estimate::joint_panel::time_varying::PanelExplanation;
use pyo3::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fallback {
    explanation: PanelExplanation,
    world: u32,
    reference_world: u32,
    before: (usize, usize),
    after: (usize, usize),
}
impl Fallback {
    fn native(self) -> facade::JointPanelExplanationRequest {
        facade::JointPanelExplanationRequest {
            explanation: self.explanation,
            world: ScientificWorldId::from_raw(self.world),
            reference_world: ScientificWorldId::from_raw(self.reference_world),
            before: self.before,
            after: self.after,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Attempt {
    reference: DecompositionDeclaration,
    fallback: Option<Fallback>,
}
fn explanation_json(attempt: &facade::JointPanelExplanationAttempt) -> serde_json::Value {
    serde_json::json!({
        "explanation": attempt.request.explanation,
        "world": attempt.request.world.raw(),
        "reference_world": attempt.request.reference_world.raw(),
        "before": attempt.request.before,
        "after": attempt.request.after,
        "point": attempt.point,
        "draws": attempt.draws,
        "parent_target": attempt.parent_target.to_hex(),
        "target": attempt.compiled_response().request().target.identity().to_hex(),
        "dependency": attempt.compiled_response().dependency_identity().to_hex(),
        "unique_refusal": attempt.unique_refusal,
        "unique_attribution_refusal":
            attempt.require_unique_attribution().err().map(|error| error.to_string()),
        "scope": "model_conditional_coarser_contrast",
        "calibration": "unmeasured",
        "grants_unique_causal_attribution": false,
    })
}
fn checked_json(checked: &facade::CheckedReferenceDecomposition) -> serde_json::Value {
    let result = &checked.decomposition;
    serde_json::json!({
        "point": result.point,
        "draws": result.draws,
        "weights": result.weights,
        "identified_reference": checked.is_identified_reference(),
        "identification_identity": checked.identification_identity.to_hex(),
        "hybrid_support": checked.hybrids.iter().map(|h| serde_json::json!({
            "mask": h.mask,
            "source_world": h.source_world.map(ScientificWorldId::raw),
            "unresolved": h.unresolved,
            "obligations": h.obligations.iter().map(|o| format!("{o:?}")).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "scope": "model_conditional_reference_decomposition",
        "calibration": "unmeasured",
        "grants_unique_causal_attribution": false,
    })
}
impl PyJointPanelResult {
    fn decomposition_request(
        &self,
        declaration: &DecompositionDeclaration,
    ) -> PyResult<facade::PanelDecompositionRequest> {
        let endpoint = |id: u32, weights: &Vec<f64>| {
            self.inner
                .request()
                .worlds
                .iter()
                .find(|world| world.world.raw() == id)
                .map(|world| facade::DecompositionEndpoint {
                    program: world.program.clone(),
                    population_weights: weights.clone(),
                })
                .ok_or_else(|| crate::value_err("decomposition names an absent compiled world"))
        };
        Ok(facade::PanelDecompositionRequest {
            before: endpoint(declaration.before_world, &declaration.before_population_weights)?,
            after: endpoint(declaration.after_world, &declaration.after_population_weights)?,
            time_weights: declaration.time_weights.clone(),
            order: declaration.order,
        })
    }
}
#[pymethods]
impl PyJointPanelResult {
    /// Execute the coarser total response contrast the finer explanation falls back to.
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn attempt_explanation(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel coarser explanation",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let fallback: Fallback = serde_json::from_str(declaration_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let request = fallback.native();
        crate::detach_catch(py, || {
            let attempt = self.inner.attempt_explanation(request, &ctx).map_err(panel_error)?;
            Ok(explanation_json(&attempt).to_string())
        })
    }
    /// The identified reference decomposition, or the supplied coarser contrast when not identified.
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn attempt_identified_reference(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel identified reference attempt",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let attempt: Attempt = serde_json::from_str(declaration_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let request = self.decomposition_request(&attempt.reference)?;
        let fallback = attempt
            .fallback
            .ok_or_else(|| crate::value_err("an identified reference attempt needs a fallback"))?
            .native();
        crate::detach_catch(py, || {
            let executed = self
                .inner
                .attempt_identified_reference_decomposition(
                    &request,
                    &attempt.reference.periods,
                    fallback,
                    &ctx,
                )
                .map_err(panel_error)?;
            Ok(match executed {
                facade::IdentifiedReferenceAttempt::Supported(checked) => {
                    serde_json::json!({"route":"supported","requested":checked_json(&checked)})
                }
                facade::IdentifiedReferenceAttempt::Coarser { requested, executed } => {
                    serde_json::json!({"route":"coarser","requested":checked_json(&requested),
                        "executed":explanation_json(&executed)})
                }
            }
            .to_string())
        })
    }
    /// The finest decomposition the design identifies, and the design that would earn the rest.
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn attempt_design_adaptive_reference(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel design-adaptive reference attempt",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let attempt: Attempt = serde_json::from_str(declaration_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let request = self.decomposition_request(&attempt.reference)?;
        let fallback = attempt.fallback.map(Fallback::native);
        crate::detach_catch(py, || {
            let adaptive = self
                .inner
                .attempt_design_adaptive_reference(
                    &request,
                    &attempt.reference.periods,
                    fallback,
                    &ctx,
                )
                .map_err(panel_error)?;
            Ok(serde_json::json!({
                "complete": adaptive.is_complete(),
                "requested": checked_json(&adaptive.requested),
                "identified_steps": adaptive.identified_steps.iter().map(|step| serde_json::json!({
                    "component": step.component, "context_mask": step.context_mask,
                    "point": step.point, "draws": step.draws,
                })).collect::<Vec<_>>(),
                "unidentified_components": adaptive.unidentified_components,
                "next_designs": adaptive.next_designs.iter().map(|step| serde_json::json!({
                    "obligation": format!("{:?}", step.obligation),
                    "components": step.components, "hybrids": step.hybrids, "design": step.design,
                })).collect::<Vec<_>>(),
                "selection": adaptive.selection,
                "coarser": adaptive.coarser.as_ref().map(explanation_json),
            })
            .to_string())
        })
    }
}

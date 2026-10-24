//! Certified search with an executable best-found fallback for unsupported expressions.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{CertificateDeclaration, PyJointPanelResult, context, facade, panel_error};
use antecedent_core::ScientificWorldId;
use antecedent_design::{
    decision_contract::{ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr},
    program_search::{
        FiniteProgramDomain, ProgramControl, ProgramRestriction,
        continuous::ContinuousSearchOptions,
    },
};
use pyo3::prelude::*;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PollOptions {
    max_evaluations: usize,
    max_iterations: usize,
    initial_step: f64,
    step_tolerance: f64,
    objective_tolerance: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    certificate: CertificateDeclaration,
    poll_options: PollOptions,
    #[serde(default)]
    uncertainty: SearchUncertainty,
}
#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SearchUncertainty {
    #[default]
    PointMeans,
    AlignedLaw,
}
#[pymethods]
impl PyJointPanelResult {
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    #[expect(
        clippy::too_many_lines,
        reason = "assembles one checked search declaration and serializes its two typed execution routes"
    )]
    fn search_program(
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
            "panel program search",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let _input = ctx
            .acquire_admission(0, (declaration_json.len() as u64).saturating_mul(8))
            .map_err(|e| match e {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    crate::CausalCancelledError::new_err(e.to_string())
                }
                _ => crate::CausalResourceError::new_err(e.to_string()),
            })?;
        let d: Declaration =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        let c = d.certificate;
        let options = c.options;
        let controls = c
            .controls
            .into_iter()
            .map(|c| {
                Ok(ProgramControl {
                    variable: self
                        .inner
                        .prepared()
                        .source_data()
                        .schema()
                        .id_of(&c.channel)
                        .map_err(|e| crate::value_err(e.to_string()))?,
                    offset: c.offset,
                    lower: c.lower,
                    upper: c.upper,
                    unit_cost: c.unit_cost,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let request = facade::JointPanelContinuousSearchRequest {
            base_world: ScientificWorldId::from_raw(c.base_world),
            scope: facade::JointPanelScientificScope::ModelConditional,
            functional: c.functional.native(),
            uncertainty: match d.uncertainty {
                SearchUncertainty::PointMeans => facade::JointPanelSearchUncertainty::PointMeans,
                SearchUncertainty::AlignedLaw => facade::JointPanelSearchUncertainty::AlignedLaw,
            },
            contract: ProgramSearchContract {
                space: ProgramSearchSpace {
                    id: "compiled_panel_response".into(),
                    variables: vec!["response".into()],
                },
                objective: ProgramObjective::ExpectedUtility(UtilityExpr::sum(
                    UtilityExpr::Const(c.utility_offset),
                    UtilityExpr::product(
                        UtilityExpr::Const(c.utility_scale),
                        UtilityExpr::Input(0),
                    ),
                )),
                constraints: vec![],
            },
            domain: FiniteProgramDomain {
                controls,
                budget: c.budget,
                restrictions: c
                    .restrictions
                    .into_iter()
                    .map(|r| ProgramRestriction {
                        id: r.id,
                        coefficients: r.coefficients,
                        bound: r.bound,
                    })
                    .collect(),
            },
            utility_units: "declared_utility".into(),
            initial: c.initial,
            options: ContinuousSearchOptions {
                max_evaluations: d.poll_options.max_evaluations,
                max_iterations: d.poll_options.max_iterations,
                initial_step: d.poll_options.initial_step,
                step_tolerance: d.poll_options.step_tolerance,
                objective_tolerance: d.poll_options.objective_tolerance,
            },
        };
        crate::detach_catch(py, || {
            let result = self.inner.search_program(request, options, &ctx).map_err(panel_error)?;
            Ok(match result {
                facade::JointPanelProgramSearchResult::Certified(r)=>serde_json::json!({
                    "route":"certified","controls":r.certificate.controls,"incumbent":r.certificate.incumbent,
                    "domain_identity":r.certificate.identity.to_hex(),
                    "global_certificate":matches!(r.certificate.stop,antecedent_design::program_search::certified::CertificateStop::Certified),
                    "assumptions":["retained fitted point response","declared bounded controls and hard linear restrictions","outward-rounded supported arithmetic"],
                    "global_upper":r.certificate.global_upper,"gap":r.certificate.gap,"stop":r.certificate.stop,
                    "splits":r.certificate.splits,"frontier":r.certificate.frontier,
                    "monotone_reductions":r.certificate.monotone_reductions,"dual_bound":r.certificate.dual_bound,
                    "separable_bound":r.certificate.separable_bound,"fitted_dependency":r.fitted_dependency.to_hex(),
                    "scope":"model_conditional","selected_value_reliability":"unmeasured" }),
                facade::JointPanelProgramSearchResult::BestFound { result,certificate_refusal }=>serde_json::json!({
                    "route":"best_found","certificate_refusal":certificate_refusal,"best":result.search.best.as_ref().map(|b|
                        serde_json::json!({"id":b.id,"controls":b.controls,"cost":b.cost,"value":b.value,
                            "feasibility":format!("{:?}",b.feasibility)})),
                    "stop":format!("{:?}",result.search.stop),"gap":result.search.gap,
                    "global_certificate":result.search.global_certificate,"scope":"model_conditional",
                    "selected_action_law":result.best_source.as_ref().and_then(|source|match source.as_numerical_source(){
                        antecedent_design::scientific_source::ScientificSourcePayload::Law(law)=>Some(serde_json::json!({
                            "metadata":law.metadata(),"draws":law.draws(),"frequentist_selected_value_coverage":false})),
                        antecedent_design::scientific_source::ScientificSourcePayload::Means(_)=>None,
                    }),
                    "selected_value_reliability":result.search.reliability }),
            }.to_string())
        })
    }
}

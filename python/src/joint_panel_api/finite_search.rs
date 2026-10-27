//! Exhaustive search over the compiled programs of a retained panel fit.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{PyJointPanelResult, context, facade, panel_error};
use antecedent_design::{
    decision_contract::{ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr},
    program_search::{FiniteProgramDomain, ProgramControl, ProgramFeasibility, ProgramRestriction},
};
use pyo3::prelude::*;
use serde::Deserialize;

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Uncertainty {
    #[default]
    PointMeans,
    AlignedLaw,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    controls: Vec<super::CertificateControlDeclaration>,
    budget: Option<f64>,
    restrictions: Vec<super::CertificateRestrictionDeclaration>,
    functional: super::ContrastFunctional,
    utility_offset: f64,
    utility_scale: f64,
    #[serde(default)]
    uncertainty: Uncertainty,
    max_candidates: usize,
}
fn feasibility(value: &ProgramFeasibility) -> serde_json::Value {
    match value {
        ProgramFeasibility::Feasible => serde_json::json!({"kind":"feasible"}),
        ProgramFeasibility::Operational(reason) => {
            serde_json::json!({"kind":"operational","reason":reason.as_str()})
        }
        ProgramFeasibility::Risk(constraints) => {
            serde_json::json!({"kind":"risk","constraints":constraints})
        }
    }
}
#[pymethods]
impl PyJointPanelResult {
    /// Exhaustively rank the compiled worlds as candidate programs under a declared
    /// expected-utility objective, scheduled controls, budget and hard linear restrictions.
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn search_declared_programs(
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
            "panel program enumeration",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let d: Declaration =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        let controls = d
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
        let request = facade::JointPanelSearchRequest {
            scope: facade::JointPanelScientificScope::ModelConditional,
            functional: d.functional.native(),
            uncertainty: match d.uncertainty {
                Uncertainty::PointMeans => facade::JointPanelSearchUncertainty::PointMeans,
                Uncertainty::AlignedLaw => facade::JointPanelSearchUncertainty::AlignedLaw,
            },
            contract: ProgramSearchContract {
                space: ProgramSearchSpace {
                    id: "compiled_panel_response".into(),
                    variables: vec!["response".into()],
                },
                objective: ProgramObjective::ExpectedUtility(UtilityExpr::sum(
                    UtilityExpr::Const(d.utility_offset),
                    UtilityExpr::product(
                        UtilityExpr::Const(d.utility_scale),
                        UtilityExpr::Input(0),
                    ),
                )),
                constraints: vec![],
            },
            domain: FiniteProgramDomain {
                controls,
                budget: d.budget,
                restrictions: d
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
            max_candidates: d.max_candidates,
        };
        crate::detach_catch(py, || {
            let search = self.inner.compile_search(&request, &ctx).map_err(panel_error)?;
            let scoped = search.execute(&ctx).map_err(panel_error)?;
            let result = &scoped.result;
            Ok(serde_json::json!({
                "rows": result.rows.iter().map(|row| serde_json::json!({
                    "id": row.id, "controls": row.controls, "cost": row.cost,
                    "feasibility": feasibility(&row.feasibility), "value": row.value,
                })).collect::<Vec<_>>(),
                "best": result.best,
                "exhaustive": result.exhaustive,
                "exact_finite_law": result.exact_finite_law,
                "continuous_gap": result.continuous_gap,
                "global_certificate": false,
                "candidate_set_identity": antecedent_core::to_hex(&result.candidate_set_identity),
                "search_identity": antecedent_core::to_hex(&result.search_identity),
                "dependency_identity": scoped.dependency_identity.to_hex(),
                "scope": "model_conditional",
                "selected_value_reliability": result.reliability,
            })
            .to_string())
        })
    }
}

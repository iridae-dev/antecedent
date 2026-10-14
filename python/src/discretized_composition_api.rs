//! Python bridge for the discretization of a linear-Gaussian composition into an executable
//! finite model.
//!
//! Python declares the plan (a method and a level count for every variable) and the contrasts
//! of discretized interventions; the cells, the conditional tables, the discretized value, its
//! closed-form comparison and computed mean-error bound, the artifact and every refusal are
//! Rust's. A refusal comes back as structured JSON for the Python layer to raise as a typed
//! exception.
//!
//! A discretized value is a point with a computed mean-error bound: it carries no interval, no
//! posterior and no calibration claim, and a parameter uncertainty declared on the source
//! composition is not propagated into it.

use std::collections::BTreeMap;

use antecedent::analysis::discretized_composition::{
    BRIDGE_CALIBRATION, BRIDGE_ERROR_SCOPE, BRIDGE_INFERENCE, DiscreteArm, DiscreteContrast,
    DiscretizationMethod, DiscretizationPlan, DiscretizationRefusal, DiscretizedLinearGaussian,
    InferenceParadigm, PreparedDiscretizedComposition, consume_discretized_composition,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_core::ExecutionContext;
use antecedent_model::discretized_scm_artifact::export_discretized_scm;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{RefusalJson, check_declared_size};
use crate::mechanism_composition_api::PreparedCompositionHandle;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;
const STAGE: &str = "discretized_composition";

fn check_size(text: &str, what: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, what)
}

fn refusal_json(error: &DiscretizationRefusal) -> String {
    let base = RefusalJson::new(error.code(), STAGE, &error.detail(), &error.to_string());
    let base = match error {
        DiscretizationRefusal::Bridge(e) => base.offending(Some(&e.subject)),
        DiscretizationRefusal::Artifact(e) => base.offending(Some(&e.subject)),
        _ => base,
    };
    base.remedy(error.remedy()).json()
}

/// The finite model a plan produced, held as prepared or as consumed from an artifact.
enum Held {
    Prepared(Box<PreparedDiscretizedComposition>),
    Consumed { bridge: Box<DiscretizedLinearGaussian> },
}

/// A discretized composition.
#[pyclass(name = "DiscretizedCompositionHandle", skip_from_py_object)]
pub(crate) struct DiscretizedCompositionHandle {
    held: Held,
}

impl DiscretizedCompositionHandle {
    fn bridge(&self) -> &DiscretizedLinearGaussian {
        match &self.held {
            Held::Prepared(prepared) => prepared.bridge(),
            Held::Consumed { bridge } => bridge,
        }
    }

    fn plan(&self) -> DiscretizationPlan {
        match &self.held {
            Held::Prepared(prepared) => prepared.plan().clone(),
            Held::Consumed { bridge } => DiscretizationPlan {
                method: bridge.method(),
                levels: bridge
                    .variables()
                    .into_iter()
                    .map(|name| {
                        (name.to_owned(), bridge.level_names(name).map_or(0, <[String]>::len))
                    })
                    .collect::<BTreeMap<_, _>>(),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanIn {
    method: String,
    levels: Vec<(String, usize)>,
}

fn plan_of(text: &str) -> PyResult<DiscretizationPlan> {
    check_size(text, "discretization plan")?;
    let wire: PlanIn = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid discretization plan: {e}")))?;
    let method = match wire.method.as_str() {
        "equal_probability_quantile" => DiscretizationMethod::EqualProbabilityQuantile,
        other => {
            return Err(crate::value_err(format!("unknown discretization method `{other}`")));
        }
    };
    Ok(DiscretizationPlan { method, levels: wire.levels.into_iter().collect() })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmIn {
    weight: f64,
    interventions: Vec<(String, usize)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContrastIn {
    outcome: String,
    arms: Vec<ArmIn>,
}

fn contrast_of(text: &str) -> PyResult<DiscreteContrast> {
    check_size(text, "discrete contrast")?;
    let wire: ContrastIn = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid discrete contrast: {e}")))?;
    Ok(DiscreteContrast {
        outcome: wire.outcome,
        arms: wire
            .arms
            .into_iter()
            .map(|arm| DiscreteArm { weight: arm.weight, interventions: arm.interventions })
            .collect(),
    })
}

fn inspection_json(handle: &DiscretizedCompositionHandle) -> Json {
    let bridge = handle.bridge();
    let variables = bridge
        .variables()
        .into_iter()
        .map(|name| {
            json!({
                "variable": name,
                "levels": bridge.level_names(name).map_or(0, <[String]>::len),
                "level_names": bridge.level_names(name),
                "level_values": bridge.level_values(name),
                "cell_edges": bridge.cell_edges(name),
                "mean_error": bridge.mechanism_mean_error(name),
            })
        })
        .collect::<Vec<_>>();
    let source = match &handle.held {
        Held::Prepared(prepared) => Some(prepared.source_plan_identity().to_owned()),
        Held::Consumed { .. } => None,
    };
    json!({
        "inference": BRIDGE_INFERENCE,
        "calibration": BRIDGE_CALIBRATION,
        "error_scope": BRIDGE_ERROR_SCOPE,
        "method": bridge.method().as_str(),
        "variables": variables,
        "identity": bridge.identity_hex(),
        "source_plan_identity": source,
    })
}

#[pymethods]
impl DiscretizedCompositionHandle {
    /// Identity of the discretization (hex): the source mechanisms, the method and every level
    /// count.
    fn identity(&self) -> String {
        self.bridge().identity_hex()
    }

    /// The plan, labels, per-variable cells and computed mechanism errors, as JSON.
    fn inspect_json(&self) -> String {
        inspection_json(self).to_string()
    }

    /// A contrast of discretized interventions beside its closed-form Gaussian value and the
    /// computed bound.
    #[pyo3(signature = (contrast_json, *, cancel=None))]
    fn mean_contrast(
        &self,
        contrast_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let contrast = contrast_of(contrast_json)?;
        let ctx: ExecutionContext = execution_context(0, None, cancel);
        let outcome = match &self.held {
            Held::Prepared(prepared) => prepared.mean_contrast(&contrast, &ctx),
            Held::Consumed { bridge } => {
                bridge.mean_contrast(&contrast, &ctx).map_err(DiscretizationRefusal::from)
            }
        };
        Ok(match outcome {
            Ok(value) => (
                Some(
                    json!({
                        "value": value.value,
                        "closed_form": value.closed_form,
                        "absolute_error": value.absolute_error,
                        "error_bound": value.error_bound,
                        "arms": value.arms.iter().map(|arm| json!({
                            "weight": arm.weight,
                            "discretized_mean": arm.discretized_mean,
                            "closed_form_mean": arm.closed_form_mean,
                            "mean_error_bound": arm.mean_error_bound,
                        })).collect::<Vec<_>>(),
                        "method": value.method,
                        "inference": value.inference,
                        "calibration": value.calibration,
                        "error_scope": value.error_scope,
                        "parameter_uncertainty_not_propagated":
                            value.parameter_uncertainty_not_propagated,
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(error) => (None, Some(refusal_json(&error))),
        })
    }

    /// An interval, band, posterior or law-valued reading: always the typed refusal.
    fn uncertainty(&self, paradigm: &str) -> PyResult<String> {
        let paradigm = match paradigm {
            "frequentist" => InferenceParadigm::Frequentist,
            "bayesian" => InferenceParadigm::Bayesian,
            other => return Err(crate::value_err(format!("unknown paradigm `{other}`"))),
        };
        let refusal = DiscretizationRefusal::from(self.bridge().uncertainty_refusal(paradigm));
        Ok(refusal_json(&refusal))
    }

    /// Discretize the refit composition again under this plan. A refusal leaves this handle as
    /// it was.
    #[pyo3(signature = (refit, *, cancel=None))]
    fn refresh(
        &self,
        py: Python<'_>,
        refit: PyRef<'_, PreparedCompositionHandle>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<DiscretizedCompositionHandle>> {
        let plan = self.plan();
        let refit = refit.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(refit.discretize(&plan, &ctx))
        })?;
        Ok(match outcome {
            Ok(prepared) => (Some(Self { held: Held::Prepared(Box::new(prepared)) }), None),
            Err(error) => (None, Some(refusal_json(&error))),
        })
    }

    /// Serialize the source premises, the plan and the claimed numbers.
    fn export(&self, py: Python<'_>) -> Refused<Py<PyBytes>> {
        let written = match &self.held {
            Held::Prepared(prepared) => prepared.export(),
            Held::Consumed { bridge } => {
                export_discretized_scm(bridge).map_err(DiscretizationRefusal::Artifact)
            }
        };
        match written {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(error) => (None, Some(refusal_json(&error))),
        }
    }
}

/// Discretize a linear-Gaussian composition under a declared plan.
#[pyfunction]
#[pyo3(signature = (handle, plan_json, *, cancel=None))]
fn discretized_composition_prepare(
    py: Python<'_>,
    handle: PyRef<'_, PreparedCompositionHandle>,
    plan_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<DiscretizedCompositionHandle>> {
    let plan = plan_of(plan_json)?;
    let inner = handle.inner.clone();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(inner.discretize(&plan, &ctx))
    })?;
    Ok(match outcome {
        Ok(prepared) => {
            (Some(DiscretizedCompositionHandle { held: Held::Prepared(Box::new(prepared)) }), None)
        }
        Err(error) => (None, Some(refusal_json(&error))),
    })
}

/// Consume an exported discretization: the model is rebuilt from the carried premises and
/// every claimed number must equal the re-derived one bit for bit.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, *, cancel=None))]
fn discretized_composition_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<DiscretizedCompositionHandle>> {
    let expected = match expected_identity {
        None => None,
        Some(text) => Some(
            *parse_digest_hex(text)
                .ok_or_else(|| {
                    crate::value_err("expected_identity must be 64 hexadecimal characters")
                })?
                .as_bytes(),
        ),
    };
    let bytes = artifact.to_vec();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(consume_discretized_composition(&bytes, expected, &ctx))
    })?;
    Ok(match outcome {
        Ok(bridge) => (
            Some(DiscretizedCompositionHandle {
                held: Held::Consumed { bridge: Box::new(bridge) },
            }),
            None,
        ),
        Err(error) => (None, Some(refusal_json(&error))),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DiscretizedCompositionHandle>()?;
    module.add_function(wrap_pyfunction!(discretized_composition_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(discretized_composition_consume, module)?)?;
    Ok(())
}

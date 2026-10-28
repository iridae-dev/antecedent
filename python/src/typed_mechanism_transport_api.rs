//! Python bridge for the transport of fitted nonlinear and non-Gaussian mechanisms.
//!
//! Python declares each mechanism's specification (parents, conditional-mean form,
//! response family), hands over the retained data of one population to fit, and
//! declares the selection diagram, invariances and parameter dependence of the
//! transport; the fits, the admission of each mechanism as carried or refitted,
//! the Wald test of a declared invariance, the Monte Carlo interventional means,
//! the delta-method parameter variance and every refusal are Rust's. A refusal
//! comes back as structured JSON (every defect found, in checker order) for the
//! Python layer to raise as a typed exception.

use std::collections::BTreeMap;

use antecedent::analysis::mechanism_composition::MechanismCompositionRefusal;
use antecedent::analysis::typed_transported_mechanism::{
    check_typed_invariance, fit_typed_mechanism, refit_typed_mechanism_in_target,
    transport_typed_mechanisms, verify_typed_transport, ContrastEstimate, CrossBlock,
    FittedMechanism, InvarianceCheck, MeanSpec, MechanismRole, MechanismSpec, ParameterDependence,
    ResponseFamily, TypedDisposition, TypedFunctionalUncertainty, TypedTransportReceipt,
    TypedTransportRequest, TypedTransportedComposition,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ContextCoordinate, ContrastArm, InvarianceDeclaration, MeanContrast, MechanismContext,
    MechanismOrigin, MechanismProvenance, SelectionDiagramDeclaration, SelectionNodeDeclaration,
};
use antecedent_model::mechanism_expr::MechanismExpression;
use pyo3::prelude::*;
use serde_json::{json, Map, Value as Json};

use crate::measurement_model_api::{
    list, member, object, parse, present, real, reals, text, texts,
};
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

fn refusal_json(refusal: &MechanismCompositionRefusal) -> String {
    let all: Vec<Json> = refusal
        .refusals
        .iter()
        .map(|r| {
            json!({
                "code": r.code,
                "stage": r.stage,
                "detail": r.detail,
                "offending": r.offending,
                "expected": r.expected,
                "supplied": r.supplied,
                "remedy": r.remedy,
            })
        })
        .collect();
    let mut first = all.first().cloned().unwrap_or_else(|| {
        json!({
            "code": refusal.code(),
            "stage": "compose",
            "detail": "mechanism_composition.request_unsupported",
            "offending": Json::Null,
            "expected": Json::Null,
            "supplied": Json::Null,
            "remedy": Json::Null,
        })
    });
    first["refusals"] = Json::Array(all);
    first.to_string()
}

fn refused<T>(refusal: &MechanismCompositionRefusal) -> Refused<T> {
    (None, Some(refusal_json(refusal)))
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn parse_spec(value: &Json) -> PyResult<MechanismSpec> {
    let map = object(value, "mechanism specification")?;
    let family_name = text(map, "family")?;
    let family = ResponseFamily::from_name(&family_name)
        .ok_or_else(|| crate::value_err(format!("unknown response family `{family_name}`")))?;
    let mean_map = object(member(map, "mean")?, "mean specification")?;
    let mean = match text(mean_map, "kind")?.as_str() {
        "polynomial" => {
            let degree = member(mean_map, "degree")?
                .as_u64()
                .and_then(|d| u8::try_from(d).ok())
                .ok_or_else(|| crate::value_err("polynomial degree must be a small integer"))?;
            let interactions = member(mean_map, "interactions")?
                .as_bool()
                .ok_or_else(|| crate::value_err("`interactions` must be a boolean"))?;
            MeanSpec::Polynomial { degree, interactions }
        }
        "expression" => {
            let expression: MechanismExpression = serde_json::from_value(
                member(mean_map, "expression")?.clone(),
            )
            .map_err(|error| crate::value_err(format!("invalid mechanism expression: {error}")))?;
            MeanSpec::Expression { expression, start: reals(mean_map, "start")? }
        }
        other => return Err(crate::value_err(format!("unknown mean kind `{other}`"))),
    };
    Ok(MechanismSpec {
        variable: text(map, "variable")?,
        parents: texts(map, "parents")?,
        mean,
        family,
    })
}

fn spec_json(spec: &MechanismSpec) -> Json {
    let mean = match &spec.mean {
        MeanSpec::Polynomial { degree, interactions } => {
            json!({"kind": "polynomial", "degree": degree, "interactions": interactions})
        }
        MeanSpec::Expression { expression, start } => json!({
            "kind": "expression",
            "expression": serde_json::to_value(expression).unwrap_or(Json::Null),
            "start": start,
        }),
    };
    json!({
        "variable": spec.variable,
        "parents": spec.parents,
        "family": spec.family.as_str(),
        "mean": mean,
    })
}

struct FitContext {
    context: MechanismContext,
    provenance: MechanismProvenance,
}

fn parse_fit_context(value: &Json) -> PyResult<FitContext> {
    let map = object(value, "fit context")?;
    Ok(FitContext {
        context: MechanismContext {
            population: text(map, "population")?,
            regime: text(map, "regime")?,
        },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: text(map, "artifact_id")?,
            evidence_id: text(map, "evidence_id")?,
        },
    })
}

fn parse_dependence(value: &Json) -> PyResult<ParameterDependence> {
    let map = object(value, "dependence")?;
    match text(map, "kind")?.as_str() {
        "unknown" => Ok(ParameterDependence::Unknown),
        "independent_by_design" => Ok(ParameterDependence::IndependentByDesign {
            justification: text(map, "justification")?,
        }),
        "joint" => Ok(ParameterDependence::Joint {
            variables: texts(map, "variables")?,
            covariance: reals(map, "covariance")?,
        }),
        "cross_covariance" => {
            let mut blocks = Vec::new();
            for entry in list(map, "blocks")? {
                let block = object(entry, "cross-covariance block")?;
                blocks.push(CrossBlock {
                    first: text(block, "first")?,
                    second: text(block, "second")?,
                    values: reals(block, "values")?,
                });
            }
            Ok(ParameterDependence::CrossCovariance { blocks })
        }
        other => Err(crate::value_err(format!("unknown dependence kind `{other}`"))),
    }
}

fn parse_coordinate(name: &str) -> PyResult<ContextCoordinate> {
    match name {
        "population" => Ok(ContextCoordinate::Population),
        "regime" => Ok(ContextCoordinate::Regime),
        other => Err(crate::value_err(format!("unknown invariance coordinate `{other}`"))),
    }
}

fn parse_request(
    value: &Json,
    sources: Vec<FittedMechanism>,
    targets: Vec<FittedMechanism>,
) -> PyResult<TypedTransportRequest> {
    let map = object(value, "typed transport")?;
    let target = object(member(map, "target")?, "target context")?;
    let diagram = object(member(map, "diagram")?, "selection diagram")?;
    let mut selection_nodes = Vec::new();
    for entry in list(diagram, "selection_nodes")? {
        let node = object(entry, "selection node")?;
        selection_nodes.push(SelectionNodeDeclaration {
            name: text(node, "name")?,
            points_into: texts(node, "points_into")?,
        });
    }
    let mut invariances = Vec::new();
    for entry in list(map, "invariances")? {
        let i = object(entry, "invariance declaration")?;
        invariances.push(InvarianceDeclaration {
            variable: text(i, "variable")?,
            coordinate: parse_coordinate(&text(i, "coordinate")?)?,
            source: text(i, "source")?,
            target: text(i, "target")?,
            assumption_id: text(i, "assumption_id")?,
        });
    }
    Ok(TypedTransportRequest {
        target: MechanismContext {
            population: text(target, "population")?,
            regime: text(target, "regime")?,
        },
        diagram: SelectionDiagramDeclaration {
            source_population: text(diagram, "source_population")?,
            target_population: text(diagram, "target_population")?,
            selection_nodes,
        },
        source_mechanisms: sources,
        target_mechanisms: targets,
        invariances,
        dependence: parse_dependence(member(map, "dependence")?)?,
        invariance_level: real(map, "invariance_level")?,
    })
}

fn pairs(map: &Map<String, Json>, key: &str) -> PyResult<Vec<(String, f64)>> {
    list(map, key)?
        .iter()
        .map(|entry| {
            let pair = entry
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-pair")))?;
            let name = pair[0]
                .as_str()
                .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-string name")))?;
            let value = pair[1]
                .as_f64()
                .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-number")))?;
            Ok((name.to_owned(), value))
        })
        .collect()
}

fn parse_contrast(value: &Json) -> PyResult<MeanContrast> {
    let map = object(value, "contrast")?;
    let mut arms = Vec::new();
    for entry in list(map, "arms")? {
        let arm = object(entry, "contrast arm")?;
        arms.push(ContrastArm {
            weight: real(arm, "weight")?,
            interventions: pairs(arm, "interventions")?,
        });
    }
    Ok(MeanContrast { outcome: text(map, "outcome")?, arms })
}

fn draw_count(draws: u64) -> PyResult<usize> {
    usize::try_from(draws).map_err(|_| crate::value_err("draw count does not fit"))
}

// ---------------------------------------------------------------------------
// Results as data.
// ---------------------------------------------------------------------------

fn check_json(check: &InvarianceCheck) -> Json {
    json!({
        "statistic": check.statistic,
        "degrees_of_freedom": check.degrees_of_freedom,
        "p_value": check.p_value,
        "level": check.level,
        "rejected": check.rejected,
    })
}

fn disposition_json(d: &TypedDisposition) -> Json {
    json!({
        "variable": d.variable,
        "role": match d.role {
            MechanismRole::Transported => "transported",
            MechanismRole::NativeInTarget => "native_in_target",
        },
        "family": d.family.as_str(),
        "fitted_population": d.fitted_population,
        "artifact_id": d.artifact_id,
        "evidence_id": d.evidence_id,
        "covariance_source": d.covariance_source,
        "consistency": d.consistency.as_ref().map(check_json),
    })
}

fn receipt_json(receipt: &TypedTransportReceipt) -> Json {
    json!({
        "route": receipt.route,
        "identity": receipt.identity,
        "source_population": receipt.source_population,
        "target_population": receipt.target_population,
        "target_regime": receipt.target_regime,
        "selection_nodes": receipt.selection_nodes.iter().map(|(name, into)| json!({
            "name": name,
            "points_into": into,
        })).collect::<Vec<_>>(),
        "selection_targets": receipt.selection_targets,
        "dispositions": receipt.dispositions.iter().map(disposition_json).collect::<Vec<_>>(),
        "invariance_level": receipt.invariance_level,
        "uncertainty_scope": receipt.uncertainty_scope,
        "support": receipt.support,
        "inference": receipt.inference,
        "assumptions": receipt.assumptions,
    })
}

fn estimate_json(e: &ContrastEstimate) -> Json {
    json!({"value": e.value, "monte_carlo_se": e.monte_carlo_se, "draws": e.draws})
}

fn uncertainty_json(u: &TypedFunctionalUncertainty) -> Json {
    json!({
        "estimate": estimate_json(&u.estimate),
        "variance": u.variance,
        "standard_error": u.standard_error,
        "dependence": u.dependence,
        "scope": u.scope,
        "gradient": u.gradient.iter().map(|(mechanism, label, g)| json!({
            "mechanism": mechanism,
            "parameter": label,
            "gradient": g,
        })).collect::<Vec<_>>(),
    })
}

/// A fitted typed mechanism: parameters, covariance and the population it came from.
#[pyclass(name = "TypedMechanismFitHandle", skip_from_py_object)]
pub(crate) struct TypedMechanismFitHandle {
    inner: FittedMechanism,
}

#[pymethods]
impl TypedMechanismFitHandle {
    /// The variable the mechanism produces.
    fn variable(&self) -> String {
        self.inner.variable().to_owned()
    }

    /// Specification, parameters, labels, covariance, scaling and fit provenance as JSON.
    fn summary_json(&self) -> String {
        let m = &self.inner;
        json!({
            "specification": spec_json(m.spec()),
            "n_obs": m.n_obs(),
            "parameters": m.parameters(),
            "parameter_labels": m.parameter_labels(),
            "mean_dimension": m.mean_dimension(),
            "covariance": m.covariance(),
            "scaling": m.scaling().iter().map(|(c, s)| json!([c, s])).collect::<Vec<_>>(),
            "population": m.context().population,
            "regime": m.context().regime,
            "artifact_id": m.provenance().artifact_id,
            "evidence_id": m.provenance().evidence_id,
        })
        .to_string()
    }

    /// `E[variable | parents]` at raw parent values in specification order.
    fn conditional_mean(&self, parents: Vec<f64>) -> PyResult<f64> {
        self.inner.conditional_mean(&parents).map_err(|e| crate::value_err(e.to_string()))
    }
}

/// A transported typed composition: one executable target-population model and its receipt.
#[pyclass(name = "TypedTransportedHandle", skip_from_py_object)]
pub(crate) struct TypedTransportedHandle {
    inner: TypedTransportedComposition,
}

#[pymethods]
impl TypedTransportedHandle {
    /// Identity of the transported composition (lowercase hex).
    fn identity(&self) -> String {
        self.inner.identity_hex()
    }

    /// The receipt as JSON.
    fn receipt_json(&self) -> String {
        receipt_json(self.inner.receipt()).to_string()
    }

    /// `sum_k weight_k * E[outcome | do(arm_k)]` by Monte Carlo over `draws` paired units.
    #[pyo3(signature = (contrast_json, draws, seed, *, cancel=None))]
    fn mean_contrast(
        &self,
        py: Python<'_>,
        contrast_json: &str,
        draws: u64,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let contrast = parse_contrast(&parse(contrast_json, "contrast")?)?;
        let draws = draw_count(draws)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx: ExecutionContext = execution_context(seed, None, cancel);
            Ok(inner.mean_contrast(&contrast, draws, &ctx))
        })?;
        Ok(match outcome {
            Ok(done) => (Some(estimate_json(&done).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// First-order parameter variance of a contrast with its scope label.
    #[pyo3(signature = (contrast_json, draws, seed, *, cancel=None))]
    fn mean_contrast_uncertainty(
        &self,
        py: Python<'_>,
        contrast_json: &str,
        draws: u64,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let contrast = parse_contrast(&parse(contrast_json, "contrast")?)?;
        let draws = draw_count(draws)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx: ExecutionContext = execution_context(seed, None, cancel);
            Ok(inner.mean_contrast_uncertainty(&contrast, draws, &ctx))
        })?;
        Ok(match outcome {
            Ok(done) => (Some(uncertainty_json(&done).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Always the typed refusal: no coverage includes the transport assumption.
    fn interval(&self) -> String {
        match self.inner.interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Fit one typed mechanism from one population's data.
#[pyfunction]
#[pyo3(signature = (spec_json, columns, context_json, design_from=None, *, cancel=None))]
fn typed_mechanism_fit(
    py: Python<'_>,
    spec_json: &str,
    columns: BTreeMap<String, Vec<f64>>,
    context_json: &str,
    design_from: Option<PyRef<'_, TypedMechanismFitHandle>>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<TypedMechanismFitHandle>> {
    let spec = parse_spec(&parse(spec_json, "mechanism specification")?)?;
    let fit_context = parse_fit_context(&parse(context_json, "fit context")?)?;
    let design = design_from.map(|d| d.inner.clone());
    let outcome = crate::detach_catch(py, move || {
        let _ = execution_context(0, None, cancel);
        Ok(fit_typed_mechanism(
            &spec,
            &columns,
            fit_context.context,
            fit_context.provenance,
            design.as_ref(),
        ))
    })?;
    Ok(match outcome {
        Ok(inner) => (Some(TypedMechanismFitHandle { inner }), None),
        Err(refusal) => refused(&refusal),
    })
}

/// Refit a source mechanism's specification on target data, on the source's design.
#[pyfunction]
#[pyo3(signature = (source, columns, context_json, *, cancel=None))]
fn typed_mechanism_refit(
    py: Python<'_>,
    source: PyRef<'_, TypedMechanismFitHandle>,
    columns: BTreeMap<String, Vec<f64>>,
    context_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<TypedMechanismFitHandle>> {
    let fit_context = parse_fit_context(&parse(context_json, "fit context")?)?;
    let source = source.inner.clone();
    let outcome = crate::detach_catch(py, move || {
        let _ = execution_context(0, None, cancel);
        Ok(refit_typed_mechanism_in_target(
            &source,
            &columns,
            fit_context.context,
            fit_context.provenance,
        ))
    })?;
    Ok(match outcome {
        Ok(inner) => (Some(TypedMechanismFitHandle { inner }), None),
        Err(refusal) => refused(&refusal),
    })
}

/// Wald comparison of a source and a target fit of one mechanism.
#[pyfunction]
fn typed_mechanism_invariance_check(
    source: PyRef<'_, TypedMechanismFitHandle>,
    target: PyRef<'_, TypedMechanismFitHandle>,
    level: f64,
) -> Refused<String> {
    match check_typed_invariance(&source.inner, &target.inner, level) {
        Ok(check) => (Some(check_json(&check).to_string()), None),
        Err(refusal) => refused(&refusal),
    }
}

/// Carry invariant mechanisms and use target refits for shifted ones: `(handle, receipt
/// json)` or the refusal.
#[pyfunction]
#[pyo3(signature = (transport_json, source_fits, target_fits, *, cancel=None))]
fn typed_mechanism_transport(
    py: Python<'_>,
    transport_json: &str,
    source_fits: Vec<PyRef<'_, TypedMechanismFitHandle>>,
    target_fits: Vec<PyRef<'_, TypedMechanismFitHandle>>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<(TypedTransportedHandle, String)>> {
    let sources = source_fits.iter().map(|h| h.inner.clone()).collect();
    let targets = target_fits.iter().map(|h| h.inner.clone()).collect();
    let request = parse_request(&parse(transport_json, "typed transport")?, sources, targets)?;
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(transport_typed_mechanisms(&request, &ctx))
    })?;
    Ok(match outcome {
        Ok(done) => {
            let receipt = receipt_json(done.receipt()).to_string();
            (Some((TypedTransportedHandle { inner: done }, receipt)), None)
        }
        Err(refusal) => refused(&refusal),
    })
}

/// Re-derive the transport and require the retained identity: `receipt json` or the refusal.
#[pyfunction]
#[pyo3(signature = (transport_json, source_fits, target_fits, expected_identity, *, cancel=None))]
fn typed_mechanism_verify_transport(
    py: Python<'_>,
    transport_json: &str,
    source_fits: Vec<PyRef<'_, TypedMechanismFitHandle>>,
    target_fits: Vec<PyRef<'_, TypedMechanismFitHandle>>,
    expected_identity: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let sources = source_fits.iter().map(|h| h.inner.clone()).collect();
    let targets = target_fits.iter().map(|h| h.inner.clone()).collect();
    let request = parse_request(&parse(transport_json, "typed transport")?, sources, targets)?;
    let expected = expected_identity.to_owned();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(verify_typed_transport(&request, &expected, &ctx))
    })?;
    Ok(match outcome {
        Ok(done) => (Some(receipt_json(done.receipt()).to_string()), None),
        Err(refusal) => refused(&refusal),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TypedMechanismFitHandle>()?;
    module.add_class::<TypedTransportedHandle>()?;
    module.add_function(wrap_pyfunction!(typed_mechanism_fit, module)?)?;
    module.add_function(wrap_pyfunction!(typed_mechanism_refit, module)?)?;
    module.add_function(wrap_pyfunction!(typed_mechanism_invariance_check, module)?)?;
    module.add_function(wrap_pyfunction!(typed_mechanism_transport, module)?)?;
    module.add_function(wrap_pyfunction!(typed_mechanism_verify_transport, module)?)?;
    Ok(())
}

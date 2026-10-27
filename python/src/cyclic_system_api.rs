// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nonlinear cyclic systems: bounded Rust execution, fitting and artifact replay.
use antecedent_core::ExecutionContext;
use antecedent_io::cyclic_system_artifact::{CyclicExecutionRequest, MAX_CYCLIC_ARTIFACT_BYTES};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type NativeFacts = Py<crate::inspect_receipt_api::PyExecutionFacts>;
type CyclicExecution = (String, Py<PyBytes>, NativeFacts);

fn execution_context(
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = crate::py_execution_context_cancel(0, 1, cancel);
    if max_bytes.is_some() {
        ctx.set_memory_limit(max_bytes);
    }
    ctx
}

fn operation_error(error: impl std::error::Error + 'static, ctx: &ExecutionContext) -> PyErr {
    // A Python callback that raised inside the operation is found by its structured
    // failure, so the run boundary can restore the original exception as the cause.
    let failure = antecedent_core::CallbackFailure::in_chain(&error).cloned();
    let converted = classified_operation_error(error, ctx);
    match &failure {
        Some(failure) => crate::interrupt::with_callback_failure(converted, failure),
        None => converted,
    }
}

fn classified_operation_error(
    error: impl std::error::Error + 'static,
    ctx: &ExecutionContext,
) -> PyErr {
    use antecedent_io::cyclic_system_artifact::CyclicArtifactError;
    if ctx.cancellation.is_cancelled() {
        return crate::CausalCancelledError::new_err(error.to_string());
    }
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(CyclicArtifactError::RecoveryFit(failure)) =
            cause.downcast_ref::<CyclicArtifactError>()
        {
            use antecedent_estimate::experimental_fit::ExperimentalFitError;
            return match failure {
                ExperimentalFitError::BudgetExceeded { .. }
                | ExperimentalFitError::ResourceOverflow => {
                    crate::CausalResourceError::new_err(failure.to_string())
                }
                _ => crate::value_err(failure.to_string()),
            };
        }
        if let Some(CyclicArtifactError::WorldRefusal(refusal)) =
            cause.downcast_ref::<CyclicArtifactError>()
        {
            return world_refusal_error(refusal);
        }
        if let Some(refusal) =
            cause.downcast_ref::<antecedent_model::cyclic_world::CyclicInterventionRefusal>()
        {
            return world_refusal_error(refusal);
        }
        if let Some(CyclicArtifactError::ScientificRefusal {
            code,
            detail,
            message,
            next_attempt,
        }) = cause.downcast_ref::<CyclicArtifactError>()
        {
            return scientific_error(code, detail, message, next_attempt);
        }
        if matches!(
            cause.downcast_ref::<CyclicArtifactError>(),
            Some(CyclicArtifactError::Cancelled)
        ) {
            return crate::CausalCancelledError::new_err(error.to_string());
        }
        if cause.is::<antecedent_core::execution::AdmissionError>()
            || matches!(
                cause.downcast_ref::<CyclicArtifactError>(),
                Some(
                    CyclicArtifactError::Admission(_)
                        | CyclicArtifactError::Execution(
                            antecedent_model::nonlinear_equilibrium::EquilibriumError::Resource(_)
                        )
                )
            )
            || matches!(
                cause.downcast_ref::<antecedent_model::nonlinear_equilibrium::EquilibriumError>(),
                Some(antecedent_model::nonlinear_equilibrium::EquilibriumError::Resource(_))
            )
        {
            return crate::CausalResourceError::new_err(error.to_string());
        }
        match cause.source() {
            Some(source) => cause = source,
            None => return crate::value_err(error.to_string()),
        }
    }
}

fn world_refusal_error(
    refusal: &antecedent_model::cyclic_world::CyclicInterventionRefusal,
) -> PyErr {
    use antecedent_model::cyclic_world::CyclicRefusalKind;
    match &refusal.kind {
        CyclicRefusalKind::Capability { .. } | CyclicRefusalKind::Evidence { .. } => {
            scientific_error(
                antecedent_core::reason_code!("route_not_supported"),
                &format!("cyclic_world.{}", refusal.operation),
                &refusal.to_string(),
                refusal.next_attempt.sentence(),
            )
        }
        CyclicRefusalKind::Execution { .. } => {
            crate::CausalResourceError::new_err(refusal.to_string())
        }
        _ => crate::value_err(refusal.to_string()),
    }
}

fn scientific_error(code: &str, detail: &str, message: &str, next_attempt: &str) -> PyErr {
    Python::attach(|py| {
        let build = || -> PyResult<PyErr> {
            let fields = pyo3::types::PyDict::new(py);
            fields.set_item("code", code)?;
            fields.set_item("detail", detail)?;
            fields.set_item("message", message)?;
            fields.set_item("remedy", next_attempt)?;
            fields.set_item("next_attempt", next_attempt)?;
            let class = pyo3::types::PyModule::import(py, "antecedent.errors")?
                .getattr("StructuredRefusal")?;
            Ok(PyErr::from_value(class.call1((fields,))?))
        };
        build().unwrap_or_else(|error| error)
    })
}

fn replay_error(error: impl std::error::Error + 'static, ctx: &ExecutionContext) -> PyErr {
    let message = error.to_string();
    let classified = operation_error(error, ctx);
    Python::attach(|py| {
        if classified.is_instance_of::<crate::CausalResourceError>(py)
            || classified.is_instance_of::<crate::CausalCancelledError>(py)
        {
            classified
        } else {
            crate::serialization_err(message)
        }
    })
}

fn input_admission(
    json: &str,
    ctx: &ExecutionContext,
) -> PyResult<antecedent_core::execution::AdmissionLease> {
    if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic input byte limit"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("cancelled"));
    }
    ctx.acquire_admission(0, (json.len() as u64).saturating_mul(4))
        .map_err(|e| operation_error(e, &ctx))
}

fn parse_request(json: &str) -> PyResult<CyclicExecutionRequest> {
    if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic request exceeds byte limit"));
    }
    serde_json::from_str(json).map_err(|e| crate::value_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn execute_cyclic_system(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CyclicExecution> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
        let _input = input_admission(request_json, &ctx)?;
        let request = parse_request(request_json)?;
        let (report, bytes) = py
            .detach(|| request.export("cyclic_equilibrium", &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        let json =
            serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?;
        Ok((json, bytes))
    });
    let ((json, bytes), facts) =
        crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)?;
    Ok((json, PyBytes::new(py, &bytes).unbind(), facts))
}

#[pyfunction]
#[pyo3(signature = (bytes, expected_identity, *, max_bytes=None, cancel=None))]
fn consume_cyclic_system(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(PyCyclicExecution, NativeFacts)> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = py.detach(|| {
        antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            antecedent::analysis::cyclic_evidence::ConsumedCyclicSystem::consume(
                bytes,
                expected_identity,
                &ctx,
            )
            .map(CyclicExecutionOwner::ConsumedSystem)
        })
    });
    let (owner, facts) = crate::inspect_receipt_api::finish_observed(
        py,
        attempt.result.map_err(|error| replay_error(error, &ctx)),
        attempt.facts,
    )?;
    Ok((
        PyCyclicExecution {
            owner: std::sync::Arc::new(CyclicExecutionState::new(owner, facts.clone_ref(py))),
        },
        facts,
    ))
}

#[pyfunction]
#[pyo3(signature = (request_json, prior_sd, chains, warmup, draws, proposal_scale, seed, *, max_bytes=None, cancel=None))]
fn cyclic_parameter_posterior(
    py: Python<'_>,
    request_json: &str,
    prior_sd: f64,
    chains: usize,
    warmup: usize,
    draws: usize,
    proposal_scale: f64,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let mut ctx = execution_context(max_bytes, cancel);
    ctx.rng = antecedent_core::RngFactory::from_seed(seed);
    let _input = input_admission(request_json, &ctx)?;
    let request = parse_request(request_json)?;
    let fit = request.fit.as_ref().ok_or_else(|| {
        crate::value_err("posterior requires retained data/parameter coordinates")
    })?;
    let report = py
        .detach(|| {
            antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior(
                &request.model,
                &fit.parameters,
                &fit.rows,
                prior_sd,
                antecedent_prob::metropolis::MetropolisOptions {
                    chains,
                    warmup,
                    draws,
                    proposal_scale,
                },
                &ctx,
            )
        })
        .map_err(|e| operation_error(e, &ctx))?;
    serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCyclicEvidence>()?;
    m.add_class::<PyCyclicExecution>()?;
    m.add_class::<PyCyclicNumericView>()?;
    m.add_function(wrap_pyfunction!(consume_cyclic_evidence, m)?)?;
    m.add_function(wrap_pyfunction!(execute_local_cyclic, m)?)?;
    m.add_class::<PyPreparedCyclicSystem>()?;
    m.add_class::<PyPreparedCyclicFamily>()?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_experimental_recovery, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_system_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(execute_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(execute_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(consume_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(consume_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_parameter_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(interface_contract_check_composition_with_cyclic, m)?)?;
    Ok(())
}

/// The contract of a prepared cyclic handle as JSON: layered identities and the
/// cyclic layers as lowercase hex, and the typed layer payloads the family export
/// writes. Deriving it does not fit, sample or solve.
fn contract_json(
    contract: &antecedent::analysis::cyclic::CyclicFamilyContract,
) -> PyResult<String> {
    let hex = antecedent_core::SemanticDigest::to_hex;
    let identities = &contract.identities;
    let section = &contract.section;
    let layer = |value: Result<serde_json::Value, serde_json::Error>| {
        value.map_err(|e| crate::serialization_err(e.to_string()))
    };
    Ok(serde_json::json!({
        "family": section.family,
        "request_identity": section.request_identity,
        "identities": {
            "target": hex(&identities.target),
            "identification": hex(&identities.identification),
            "identification_product": identities.identification_product.as_ref().map(hex),
            "program": identities.program.as_ref().map(hex),
            "inference_binding": hex(&identities.inference_binding),
            "observation": hex(&identities.observation),
            "data_snapshot": hex(&identities.data_snapshot),
        },
        "cyclic_solution": hex(&contract.cyclic_solution),
        "recovery": contract.recovery.as_ref().map(hex),
        "lowered_program_identity": hex(&contract.lowered_program_identity),
        "identification_kind": section.identification.kind(),
        "layers": {
            "target": layer(serde_json::to_value(&section.target))?,
            "identification": layer(serde_json::to_value(&section.identification))?,
            "program": layer(serde_json::to_value(&section.program))?,
            "inference_binding": layer(serde_json::to_value(&section.inference_binding))?,
            "observation": layer(serde_json::to_value(&section.observation))?,
            "data_snapshot": layer(serde_json::to_value(&section.data_snapshot))?,
            "solution": layer(serde_json::to_value(&section.solution))?,
        },
    })
    .to_string())
}

/// The contract of a prepared cyclic family or system handle.
fn cyclic_contract_of(
    handle: &Bound<'_, PyAny>,
) -> PyResult<antecedent::analysis::cyclic::CyclicFamilyContract> {
    let contract = if let Ok(family) = handle.extract::<PyRef<'_, PyPreparedCyclicFamily>>() {
        family.inner.contract()
    } else if let Ok(system) = handle.extract::<PyRef<'_, PyPreparedCyclicSystem>>() {
        system.inner.contract()
    } else {
        return Err(crate::value_err(
            "a cyclic component must be a prepared cyclic family or system handle",
        ));
    };
    contract.map_err(|e| crate::value_err(e.to_string()))
}

/// Check an interface composition in which each prepared cyclic product enters as one
/// declared component (`check_interface_composition_with_cyclic`).
///
/// `cyclic` pairs a component name with a prepared cyclic family or system handle; the
/// declaration is that handle's contract (`CyclicFamilyContract::as_component`), derived
/// without fitting. Feedback inside a product stays inside its component; feedback among
/// components is refused with its path. Refusals are returned as a JSON array.
#[pyfunction]
#[pyo3(signature = (components_json, connections_json, cyclic, ledger=None))]
fn interface_contract_check_composition_with_cyclic(
    components_json: &str,
    connections_json: &str,
    cyclic: Vec<(String, Bound<'_, PyAny>)>,
    ledger: Option<PyRef<'_, crate::evidence_ledger_api::EvidenceLedgerHandle>>,
) -> PyResult<crate::interface_contract_api::Checked> {
    let components = crate::interface_contract_api::components_of(components_json)?;
    let connections = crate::interface_contract_api::connections_of(connections_json)?;
    let declarations = cyclic
        .iter()
        .map(|(component, handle)| {
            cyclic_contract_of(handle).map(|contract| contract.as_component(component.clone()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(crate::interface_contract_api::composed_checked(
        antecedent::analysis::interface_contract::check_interface_composition_with_cyclic(
            &components,
            &connections,
            &declarations,
            ledger.as_deref().map(crate::evidence_ledger_api::EvidenceLedgerHandle::ledger),
        ),
    ))
}

#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn execute_cyclic_family(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CyclicExecution> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
        let _input = input_admission(request_json, &ctx)?;
        let request: antecedent_io::cyclic_family_artifact::CyclicFamilyRequest =
            serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
        let (report, bytes) =
            py.detach(|| request.export(&ctx)).map_err(|e| operation_error(e, &ctx))?;
        let json =
            serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?;
        Ok((json, bytes))
    });
    let ((json, bytes), facts) =
        crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)?;
    Ok((json, PyBytes::new(py, &bytes).unbind(), facts))
}

#[pyfunction]
#[pyo3(signature = (bytes, expected_identity, *, max_bytes=None, cancel=None))]
fn consume_cyclic_family(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(PyCyclicExecution, NativeFacts)> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = py.detach(|| {
        antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            antecedent::analysis::cyclic_evidence::ConsumedCyclicFamily::consume(
                bytes,
                expected_identity,
                &ctx,
            )
        })
    });
    let (owner, facts) = crate::inspect_receipt_api::finish_observed(
        py,
        attempt.result.map_err(|error| replay_error(error, &ctx)),
        attempt.facts,
    )?;
    Ok((
        PyCyclicExecution {
            owner: std::sync::Arc::new(CyclicExecutionState::new(
                CyclicExecutionOwner::ConsumedFamily(owner),
                facts.clone_ref(py),
            )),
        },
        facts,
    ))
}

enum CyclicExecutionOwner {
    System(antecedent::analysis::cyclic::ExecutedCyclicSystem),
    Family(antecedent::analysis::cyclic::ExecutedCyclicFamily),
    ConsumedSystem(antecedent::analysis::cyclic_evidence::ConsumedCyclicSystem),
    ConsumedFamily(antecedent::analysis::cyclic_evidence::ConsumedCyclicFamily),
}

struct CyclicExecutionState {
    source: CyclicExecutionOwner,
    _facts: NativeFacts,
    recovery_summary: Option<serde_json::Value>,
}

impl CyclicExecutionState {
    fn new(source: CyclicExecutionOwner, facts: NativeFacts) -> Self {
        let recovery_summary = source.recovery_summary();
        Self { source, _facts: facts, recovery_summary }
    }
}

impl std::ops::Deref for CyclicExecutionState {
    type Target = CyclicExecutionOwner;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl CyclicExecutionOwner {
    fn system(&self) -> Option<&antecedent_io::cyclic_system_artifact::CyclicExecutionReport> {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        match self {
            Self::System(owner) => Some(owner.report()),
            Self::ConsumedSystem(owner) => Some(owner.report()),
            Self::Family(owner) => match &owner.report().answer {
                A::System(report) => Some(report),
                _ => None,
            },
            Self::ConsumedFamily(owner) => match &owner.report().answer {
                A::System(report) => Some(report),
                _ => None,
            },
        }
    }

    fn family(&self) -> Option<&antecedent_io::cyclic_family_artifact::CyclicFamilyReport> {
        match self {
            Self::Family(owner) => Some(owner.report()),
            Self::ConsumedFamily(owner) => Some(owner.report()),
            _ => None,
        }
    }

    fn family_request(
        &self,
    ) -> Option<&antecedent_io::cyclic_family_artifact::CyclicFamilyRequest> {
        match self {
            Self::Family(owner) => Some(owner.request()),
            Self::ConsumedFamily(owner) => Some(owner.request()),
            _ => None,
        }
    }

    fn source_identity(&self) -> &str {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        match self.family().map(|report| &report.answer) {
            Some(A::GaussianFunctional(report)) => &report.source_identity,
            Some(A::Functional(report)) => &report.source_identity,
            _ => self.identity(),
        }
    }

    fn recovered_origin(
        &self,
    ) -> Option<&antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire> {
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer as A, CyclicFamilyRequest as R,
        };
        match self.family().map(|report| &report.answer) {
            Some(A::ExperimentalRecovery(report)) => Some(&report.law.origin),
            Some(A::RecoveredLinear(_)) => match self.family_request() {
                Some(R::RecoveredLinear { uncertainty, .. }) => Some(uncertainty),
                _ => None,
            },
            _ => None,
        }
    }

    fn identity(&self) -> &str {
        match self {
            Self::System(owner) => owner.identity(),
            Self::Family(owner) => owner.identity(),
            Self::ConsumedSystem(owner) => owner.identity(),
            Self::ConsumedFamily(owner) => owner.identity(),
        }
    }

    fn family_name(&self) -> &'static str {
        self.family_request().map_or("system", |request| request.family_name())
    }

    fn scope(&self) -> &'static str {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        match self.family().map(|report| &report.answer) {
            Some(A::GaussianFunctional(_)) => {
                "theorem_identified_conditional_on_declared_Gaussian_observational_density"
            }
            Some(A::Functional(_)) => {
                "theorem_identified_conditional_on_declared_observational_law"
            }
            Some(A::Graph(_)) => "theorem_scoped_graphical_evidence",
            Some(A::ExperimentalRecovery(_)) => {
                "recovered_from_declared_clamp_experiments_conditional_on_the_recipe"
            }
            _ => "specified_or_fitted_model_conditional",
        }
    }

    fn world_count(&self) -> usize {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        if let Some(system) = self.system() {
            return system.worlds.len();
        }
        match self.family().map(|report| &report.answer) {
            Some(A::Worlds(worlds)) => worlds.len(),
            Some(A::RecoveredLinear(report)) => report.worlds.len(),
            Some(A::ExperimentalRecovery(report)) => report.worlds.len(),
            Some(A::EquilibriumActions(worlds)) => worlds.len(),
            Some(A::ParameterPosterior { worlds, .. }) => worlds.len(),
            Some(A::FrequentistBootstrap { worlds, .. }) => worlds.len(),
            Some(
                A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. },
            ) => worlds.len(),
            Some(A::OuMechanisms { laws, .. }) => laws.len(),
            Some(A::Ode(_) | A::OuStationary(_) | A::OuTransient(_)) => 1,
            _ => 0,
        }
    }

    fn world_id(&self, position: usize) -> PyResult<u32> {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        match self.family().map(|report| &report.answer) {
            Some(A::Worlds(worlds)) => worlds.get(position).map(|world| world.world),
            Some(A::ExperimentalRecovery(report)) => {
                report.worlds.get(position).map(|world| world.world)
            }
            _ => u32::try_from(position).ok().filter(|_| position < self.world_count()),
        }
        .ok_or_else(|| crate::value_err("cyclic world position out of range"))
    }

    fn position(&self, world: u32) -> PyResult<usize> {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        match self.family().map(|report| &report.answer) {
            Some(A::Worlds(worlds)) => worlds.iter().position(|item| item.world == world),
            Some(A::ExperimentalRecovery(report)) => {
                report.worlds.iter().position(|item| item.world == world)
            }
            _ => usize::try_from(world).ok().filter(|position| *position < self.world_count()),
        }
        .ok_or_else(|| crate::value_err("unknown cyclic world id"))
    }

    fn values(&self, coordinate: &CyclicValueCoordinate) -> PyResult<Option<&[f64]>> {
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer as A, CyclicWorldOutcomeAnswer as W,
        };
        let position = self.position(coordinate.world)?;
        let row = coordinate.row;
        let draw = coordinate.draw;
        let field = coordinate.field.as_str();
        let invalid = || crate::value_err("cyclic numeric coordinate/field out of range");
        if !matches!(
            field,
            "values"
                | "mean"
                | "covariance"
                | "disturbance_mean"
                | "disturbance_covariance"
                | "times"
        ) {
            return Err(invalid());
        }
        if let Some(system) = self.system() {
            if field != "values" || draw != 0 {
                return Err(invalid());
            }
            let solution = if coordinate.counterfactual {
                if row != 0 {
                    return Err(invalid());
                }
                system.counterfactuals.as_ref().and_then(|worlds| worlds.get(position))
            } else {
                system.worlds.get(position).and_then(|rows| rows.get(row))
            }
            .ok_or_else(invalid)?;
            return Ok(Some(&solution.values));
        }
        let report = self.family().ok_or_else(invalid)?;
        if coordinate.counterfactual && !matches!(&report.answer, A::EquilibriumActions(_)) {
            return Err(invalid());
        }
        match &report.answer {
            A::Worlds(worlds) if draw == 0 => match &worlds[position].outcome {
                W::GaussianLaw { mean, covariance, .. } if row == 0 => match field {
                    "mean" => Ok(Some(mean)),
                    "covariance" => Ok(Some(covariance)),
                    _ => Err(invalid()),
                },
                W::EquilibriumDraws { outcomes } => {
                    equilibrium_values(outcomes.get(row).ok_or_else(invalid)?, field)
                }
                W::Trajectory { trajectory } => match field {
                    "values" => Ok(Some(trajectory.states.get(row).ok_or_else(invalid)?)),
                    "times" if row == 0 => Ok(Some(&trajectory.times)),
                    _ => Err(invalid()),
                },
                W::NoSolution { .. } | W::Withheld => Ok(None),
                _ => Err(invalid()),
            },
            A::EquilibriumActions(worlds) if draw == 0 => {
                if coordinate.counterfactual {
                    if row != 0 {
                        return Err(invalid());
                    }
                    match &worlds[position].counterfactual {
                        Some(value) => equilibrium_values(value, field),
                        None => Err(invalid()),
                    }
                } else {
                    equilibrium_values(worlds[position].rows.get(row).ok_or_else(invalid)?, field)
                }
            }
            A::ParameterPosterior { worlds, .. } => equilibrium_values(
                worlds[position].get(draw).and_then(|rows| rows.get(row)).ok_or_else(invalid)?,
                field,
            ),
            A::FrequentistBootstrap { worlds, .. } => {
                match worlds[position].get(draw).ok_or_else(invalid)? {
                    Some(rows) => equilibrium_values(rows.get(row).ok_or_else(invalid)?, field),
                    None => Ok(None),
                }
            }
            A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. }
                if row == 0 =>
            {
                equilibrium_values(worlds[position].get(draw).ok_or_else(invalid)?, field)
            }
            A::RecoveredLinear(report) => {
                recovered_values(&report.worlds[position], draw, row, field)
            }
            A::ExperimentalRecovery(report) => {
                recovered_values(&report.worlds[position].answer, draw, row, field)
            }
            A::Ode(trajectory) if draw == 0 => match field {
                "values" => Ok(Some(trajectory.states.get(row).ok_or_else(invalid)?)),
                "times" if row == 0 => Ok(Some(&trajectory.times)),
                _ => Err(invalid()),
            },
            A::OuMechanisms { laws, .. } if draw == 0 && row == 0 => match &laws[position] {
                antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) => {
                    match field {
                        "mean" => Ok(Some(&law.mean)),
                        "covariance" => Ok(Some(&law.covariance)),
                        _ => Err(invalid()),
                    }
                }
                antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Failed { .. } => Ok(None),
            },
            A::OuStationary(covariance) if draw == 0 && row == 0 && field == "covariance" => {
                Ok(Some(covariance))
            }
            A::OuTransient(moments) if draw == 0 => match field {
                "mean" => Ok(Some(moments.means.get(row).ok_or_else(invalid)?)),
                "covariance" => Ok(Some(moments.covariances.get(row).ok_or_else(invalid)?)),
                "times" if row == 0 => Ok(Some(&moments.times)),
                _ => Err(invalid()),
            },
            _ => Err(invalid()),
        }
    }
}

fn equilibrium_values<'a>(
    outcome: &'a antecedent_model::nonlinear_equilibrium::EquilibriumOutcome,
    field: &str,
) -> PyResult<Option<&'a [f64]>> {
    use antecedent_model::nonlinear_equilibrium::EquilibriumOutcome as E;
    match outcome {
        E::Solved(value) if field == "values" => Ok(Some(&value.values)),
        E::Failed(_) => Ok(None),
        _ => Err(crate::value_err("unknown equilibrium numeric field")),
    }
}

fn recovered_values<'a>(
    world: &'a antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    draw: usize,
    row: usize,
    field: &str,
) -> PyResult<Option<&'a [f64]>> {
    use antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer as D;
    if row != 0 {
        return Err(crate::value_err("recovered world has no noise-row axis"));
    }
    let slot = world.draws.get(draw).ok_or_else(|| crate::value_err("cyclic draw out of range"))?;
    match &slot.outcome {
        D::Equilibrium { mean, covariance, disturbance_mean, disturbance_covariance, .. } => {
            match field {
                "mean" => Ok(Some(mean)),
                "covariance" => Ok(Some(covariance)),
                "disturbance_mean" => Ok(Some(disturbance_mean)),
                "disturbance_covariance" => Ok(Some(disturbance_covariance)),
                _ => Err(crate::value_err("unknown recovered world numeric field")),
            }
        }
        D::Inadmissible { .. } | D::NotCertified { .. } => Ok(None),
    }
}

struct CyclicValueCoordinate {
    world: u32,
    row: usize,
    draw: usize,
    counterfactual: bool,
    field: String,
}

#[pyclass(name = "_CyclicNumericView", frozen)]
pub(crate) struct PyCyclicNumericView {
    owner: std::sync::Arc<CyclicExecutionState>,
    coordinate: CyclicValueCoordinate,
}

impl PyCyclicNumericView {
    pub(crate) fn values(&self) -> PyResult<&[f64]> {
        self.owner
            .values(&self.coordinate)?
            .ok_or_else(|| crate::value_err("cyclic failed slot has no numeric values"))
    }
}

#[pymethods]
impl PyCyclicNumericView {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.values()?.len())
    }
}

#[pyclass(name = "_CyclicExecution", frozen)]
struct PyCyclicExecution {
    owner: std::sync::Arc<CyclicExecutionState>,
}

#[pymethods]
impl PyCyclicExecution {
    fn identity(&self) -> &str {
        self.owner.identity()
    }

    fn metadata_json(&self) -> PyResult<String> {
        let system = self.owner.system();
        let retained = match self.owner.family() {
            Some(report) => report.retained_admitted_bytes(),
            None => system.map_or(0, |report| report.retained_admitted_bytes()),
        };
        Ok(serde_json::json!({
            "identity":self.owner.identity(), "family":self.owner.family_name(),
            "claim":system.map(|report|report.claim.as_str()).unwrap_or(self.owner.scope()),
            "calibration":system.map(|report|report.calibration.as_str()).unwrap_or("unmeasured"),
            "source_identity":self.owner.source_identity(), "scope":self.owner.scope(),
            "draw_origin":self.owner.recovered_origin().map(origin_metadata),
            "law_meaning":self.owner.recovered_origin().and_then(|origin|origin.law_meaning()),
            "world_count":self.owner.world_count(),
            "contract":self.owner.contract_summary(),
            "recovery":&self.owner.recovery_summary,
            "diagnostics":{"retained_admitted_bytes":retained,"graph":self.owner.graph_metadata()},
            "loaded":matches!(&self.owner.source,CyclicExecutionOwner::ConsumedSystem(_)|CyclicExecutionOwner::ConsumedFamily(_)),
        }).to_string())
    }

    /// Retained unit ids, preserving the original source namespace (including gaps).
    fn recovery_nodes(&self) -> PyResult<Vec<u32>> {
        Ok(self
            .owner
            .recovery()
            .ok_or_else(|| crate::value_err("expected an experimental recovered law"))?
            .law
            .nodes
            .clone())
    }

    #[pyo3(signature=(worlds_json,outcome,*,max_bytes=None,cancel=None))]
    fn prepare_recovered_worlds(
        &self,
        py: Python<'_>,
        worlds_json: &str,
        outcome: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyPreparedCyclicFamily> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(worlds_json, &ctx)?;
        let worlds =
            serde_json::from_str(worlds_json).map_err(|error| operation_error(error, &ctx))?;
        let inner = match &self.owner.source {
            CyclicExecutionOwner::Family(owner) => {
                py.detach(|| owner.prepare_recovered_worlds(worlds, outcome, &ctx))
            }
            CyclicExecutionOwner::ConsumedFamily(owner) => {
                py.detach(|| owner.prepare_recovered_worlds(worlds, outcome, &ctx))
            }
            _ => return Err(crate::value_err("expected an experimental recovered law")),
        }
        .map_err(|error| operation_error(error, &ctx))?;
        Ok(PyPreparedCyclicFamily { inner })
    }

    fn recovery_law_report_json(&self) -> PyResult<String> {
        let law = &self
            .owner
            .recovery()
            .ok_or_else(|| crate::value_err("expected an experimental recovery result"))?
            .law;
        serde_json::to_string(law.as_ref())
            .map_err(|error| crate::serialization_err(error.to_string()))
    }

    fn recovery_summary_json(&self) -> String {
        self.owner
            .recovery_summary
            .as_ref()
            .map_or_else(|| "null".into(), serde_json::Value::to_string)
    }

    #[pyo3(signature=(*,offset=0,limit=64))]
    fn recovery_draw_status_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        if limit > 256 {
            return Err(crate::value_err("cyclic recovery page limit exceeds256"));
        }
        let law = &self
            .owner
            .recovery()
            .ok_or_else(|| crate::value_err("expected an experimental recovery result"))?
            .law;
        let end = offset.saturating_add(limit).min(law.draw_status.len());
        let page = law.draw_status.get(offset..end).unwrap_or(&[]);
        Ok(serde_json::json!({"count":law.draw_status.len(),"offset":offset,"draw_status":page,"next_offset":(end<law.draw_status.len()).then_some(end)}).to_string())
    }

    #[pyo3(signature=(draw,*,offset=0,limit=64))]
    fn recovery_unresolved_json(
        &self,
        draw: usize,
        offset: usize,
        limit: usize,
    ) -> PyResult<String> {
        if limit > 256 {
            return Err(crate::value_err("cyclic recovery page limit exceeds256"));
        }
        let law = &self
            .owner
            .recovery()
            .ok_or_else(|| crate::value_err("expected an experimental recovery result"))?
            .law;
        let directions = &law
            .draws
            .get(draw)
            .ok_or_else(|| crate::value_err("cyclic recovery draw out of range"))?
            .unresolved;
        let end = offset.saturating_add(limit).min(directions.len());
        let page = directions.get(offset..end).unwrap_or(&[]);
        Ok(serde_json::json!({"draw":draw,"count":directions.len(),"offset":offset,"unresolved_directions":page,"next_offset":(end<directions.len()).then_some(end)}).to_string())
    }

    fn contract_metadata_json(&self) -> String {
        self.owner.contract_summary().to_string()
    }

    #[pyo3(signature=(*,offset=0,limit=64))]
    fn assumptions_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        use antecedent_io::contract_section::cyclic::CyclicTargetWire as T;
        if limit > 256 {
            return Err(crate::value_err("cyclic assumption page limit exceeds256"));
        }
        let Some(contract) = self.owner.contract_metadata() else {
            return Ok(serde_json::json!({"available":false,"reason":"not_present_in_source_artifact","assumptions":[],"count":0}).to_string());
        };
        let count = assumption_count(&contract.target);
        let mut page = Vec::new();
        let mut position = 0;
        let mut append =
            |target_index: usize,
             target: &antecedent_io::scientific_target_wire::ScientificTargetWire| {
                for record in target.assumptions() {
                    if position >= offset && page.len() < limit {
                        page.push(serde_json::json!({"target_index":target_index,"record":record}));
                    }
                    position += 1;
                }
            };
        match &contract.target {
            T::Worlds { target } => append(0, target),
            T::JointWorlds { targets } => {
                for (index, target) in targets.iter().enumerate() {
                    append(index, target);
                }
            }
            T::Functional { .. } => {}
        }
        let end = offset.saturating_add(page.len());
        Ok(serde_json::json!({"available":true,"count":count,"offset":offset,"assumptions":page,"next_offset":(end<count).then_some(end)}).to_string())
    }

    fn report_json(&self) -> PyResult<String> {
        if let Some(report) = self.owner.family() {
            serde_json::to_string(report)
        } else {
            let report = self
                .owner
                .system()
                .ok_or_else(|| crate::value_err("missing cyclic execution report"))?;
            serde_json::to_string(report)
        }
        .map_err(|error| crate::serialization_err(error.to_string()))
    }

    #[pyo3(signature=(*,offset=0,limit=64))]
    fn world_ids(&self, offset: usize, limit: usize) -> PyResult<Vec<u32>> {
        if limit > 256 {
            return Err(crate::value_err("cyclic metadata page limit exceeds256"));
        }
        (offset..offset.saturating_add(limit).min(self.owner.world_count()))
            .map(|position| self.owner.world_id(position))
            .collect()
    }

    fn variable_names(&self) -> Option<Vec<String>> {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyRequest as R;
        let request = match &self.owner.source {
            CyclicExecutionOwner::System(owner) => Some(owner.prepared().request()),
            CyclicExecutionOwner::ConsumedSystem(owner) => Some(owner.request()),
            _ => match self.owner.family_request() {
                Some(R::System { request }) => Some(request),
                _ => None,
            },
        }?;
        Some(request.model.equations.iter().map(|equation| equation.id.clone()).collect())
    }

    fn require_intervention_worlds(&self) -> PyResult<()> {
        if self.world_view_kind().is_some() {
            return Ok(());
        }
        Err(scientific_error(
            antecedent_core::reason_code!("route_not_supported"),
            "cyclic_family.intervention_worlds_unavailable",
            &format!("family `{}` has no declared intervention worlds", self.owner.family_name()),
            "declare the worlds as typed interventions with `linear_worlds`, `equilibrium_worlds`, `ode_worlds`, `recovered_linear` or `experimental_recovery`",
        ))
    }

    fn world_view_kind(&self) -> Option<&'static str> {
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer as A, CyclicFamilyRequest as R,
        };
        match self.owner.family().map(|report| &report.answer) {
            Some(A::Worlds(_)) => Some("specified"),
            Some(A::ExperimentalRecovery(_)) => Some("recovered"),
            Some(A::RecoveredLinear(_))
                if matches!(self.owner.family_request(), Some(R::RecoveredLinear { .. })) =>
            {
                Some("recovered")
            }
            _ => None,
        }
    }

    fn world_metadata_json(&self, world: u32) -> PyResult<String> {
        self.owner.world_metadata(world).map(|value| value.to_string())
    }

    fn world_report_json(&self, world: u32) -> PyResult<String> {
        use antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer as A;
        let position = self.owner.position(world)?;
        let value = match self.owner.family().map(|report| &report.answer) {
            Some(A::Worlds(worlds)) => {
                serde_json::json!({"source":"specified","answer":worlds[position]})
            }
            Some(A::RecoveredLinear(report)) => {
                serde_json::json!({"source":"recovered","world":world,"origin":self.owner.recovered_origin(),"answer":report.worlds[position]})
            }
            Some(A::ExperimentalRecovery(report)) => {
                serde_json::json!({"source":"recovered","world":world,"origin":report.law.origin,"answer":report.worlds[position].answer})
            }
            _ => {
                let system = self.owner.system().ok_or_else(|| {
                    crate::value_err("this cyclic family has no typed intervention-world report")
                })?;
                serde_json::json!({"world":world,"rows":system.worlds[position],"counterfactual":system.counterfactuals.as_ref().and_then(|rows|rows.get(position))})
            }
        };
        Ok(value.to_string())
    }

    #[pyo3(signature=(world,*,row=0,draw=0,counterfactual=false))]
    fn world_slot_metadata_json(
        &self,
        world: u32,
        row: usize,
        draw: usize,
        counterfactual: bool,
    ) -> PyResult<String> {
        self.owner.slot_metadata(world, row, draw, counterfactual).map(|value| value.to_string())
    }

    #[pyo3(signature=(world,*,row=0,draw=0,counterfactual=false,field="values"))]
    fn world_values(
        &self,
        world: u32,
        row: usize,
        draw: usize,
        counterfactual: bool,
        field: &str,
    ) -> PyResult<Option<PyCyclicNumericView>> {
        let coordinate =
            CyclicValueCoordinate { world, row, draw, counterfactual, field: field.into() };
        if self.owner.values(&coordinate)?.is_none() {
            return Ok(None);
        }
        Ok(Some(PyCyclicNumericView { owner: std::sync::Arc::clone(&self.owner), coordinate }))
    }

    #[pyo3(signature=(*,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = execution_context(max_bytes, cancel);
        if ctx.cancellation.is_cancelled() {
            return Err(crate::CausalCancelledError::new_err("cancelled"));
        }
        match &self.owner.source {
            CyclicExecutionOwner::System(owner) => {
                let bytes = py
                    .detach(|| owner.export("cyclic_equilibrium", &ctx))
                    .map_err(|error| operation_error(error, &ctx))?;
                Ok(PyBytes::new(py, &bytes).unbind())
            }
            CyclicExecutionOwner::Family(owner) => {
                let bytes = py
                    .detach(|| owner.export(&ctx))
                    .map_err(|error| operation_error(error, &ctx))?;
                Ok(PyBytes::new(py, &bytes).unbind())
            }
            CyclicExecutionOwner::ConsumedSystem(owner) => {
                let bytes = owner.artifact();
                let _output = ctx
                    .acquire_admission(0, bytes.len() as u64)
                    .map_err(|error| operation_error(error, &ctx))?;
                Ok(PyBytes::new(py, bytes).unbind())
            }
            CyclicExecutionOwner::ConsumedFamily(owner) => {
                let bytes = owner.artifact();
                let _output = ctx
                    .acquire_admission(0, bytes.len() as u64)
                    .map_err(|error| operation_error(error, &ctx))?;
                Ok(PyBytes::new(py, bytes).unbind())
            }
        }
    }

    #[pyo3(signature=(schema_json,selectors_json,*,max_bytes=None,cancel=None))]
    fn evidence(
        &self,
        py: Python<'_>,
        schema_json: &str,
        selectors_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCyclicEvidence> {
        let ctx = execution_context(max_bytes, cancel);
        let _schema = input_admission(schema_json, &ctx)?;
        let _selectors = input_admission(selectors_json, &ctx)?;
        let schema =
            serde_json::from_str(schema_json).map_err(|error| operation_error(error, &ctx))?;
        let selectors =
            serde_json::from_str(selectors_json).map_err(|error| operation_error(error, &ctx))?;
        let inner = match &self.owner.source {
            CyclicExecutionOwner::Family(owner) => {
                py.detach(|| owner.evidence(schema, selectors, &ctx))
            }
            CyclicExecutionOwner::ConsumedFamily(owner) => {
                py.detach(|| owner.evidence(schema, selectors, &ctx))
            }
            CyclicExecutionOwner::System(owner) => {
                py.detach(|| owner.evidence(schema, selectors, &ctx))
            }
            CyclicExecutionOwner::ConsumedSystem(owner) => {
                py.detach(|| owner.evidence(schema, selectors, &ctx))
            }
        }
        .map_err(|error| operation_error(error, &ctx))?;
        Ok(PyCyclicEvidence { inner })
    }
}

impl CyclicExecutionOwner {
    fn world_metadata(&self, world: u32) -> PyResult<serde_json::Value> {
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer as A, CyclicWorldOutcomeAnswer as W,
        };
        let position = self.position(world)?;
        if let Some(system) = self.system() {
            return Ok(
                serde_json::json!({"world":world,"family":"system","status":"equilibrium_draws","row_count":system.worlds[position].len(),"counterfactual":system.counterfactuals.is_some()}),
            );
        }
        let report = self.family().ok_or_else(|| crate::value_err("expected cyclic family"))?;
        match &report.answer {
            A::Worlds(worlds) => {
                let value = &worlds[position];
                let outcome = match &value.outcome {
                    W::GaussianLaw { mean, .. } => {
                        serde_json::json!({"status":"gaussian_law","dimension":mean.len()})
                    }
                    W::EquilibriumDraws { outcomes } => {
                        serde_json::json!({"status":"equilibrium_draws","row_count":outcomes.len()})
                    }
                    W::Trajectory { trajectory } => {
                        serde_json::json!({"status":"trajectory","row_count":trajectory.states.len(),"dimension":trajectory.states.first().map(Vec::len)})
                    }
                    W::NoSolution { cause, next_attempt } => {
                        serde_json::json!({"status":"no_solution","cause":cause,"next_attempt":next_attempt})
                    }
                    W::Withheld => serde_json::json!({"status":"withheld"}),
                };
                Ok(
                    serde_json::json!({"source":"specified","world":value.world,"family":value.family,"variables":value.variables,"actions":value.actions,"outcome":outcome,"certificate":value.certificate,"solved_by":value.solved_by,"failed_mass":value.failed_mass,"withheld_mass":value.withheld_mass,"physical_stability":value.physical_stability}),
                )
            }
            A::RecoveredLinear(report) => {
                Ok(recovered_metadata(world, &report.worlds[position], self.recovered_origin()))
            }
            A::ExperimentalRecovery(report) => Ok(recovered_metadata(
                world,
                &report.worlds[position].answer,
                Some(&report.law.origin),
            )),
            A::ParameterPosterior { worlds, .. } => Ok(
                serde_json::json!({"world":world,"family":self.family_name(),"draw_count":worlds[position].len(),"row_count":worlds[position].first().map(Vec::len),"law_meaning":"parameter_posterior"}),
            ),
            A::FrequentistBootstrap { worlds, .. } => Ok(
                serde_json::json!({"world":world,"family":self.family_name(),"draw_count":worlds[position].len(),"law_meaning":"sampling_law"}),
            ),
            A::JointCounterfactual { worlds, .. } => Ok(
                serde_json::json!({"world":world,"family":self.family_name(),"draw_count":worlds[position].len(),"law_meaning":"joint_parameter_factual_posterior"}),
            ),
            A::PartialCounterfactual { worlds, .. } => Ok(
                serde_json::json!({"world":world,"family":self.family_name(),"draw_count":worlds[position].len(),"law_meaning":"factual_abduction_posterior"}),
            ),
            A::EquilibriumActions(worlds) => Ok(
                serde_json::json!({"world":world,"family":self.family_name(),"row_count":worlds[position].rows.len(),"counterfactual":worlds[position].counterfactual.is_some()}),
            ),
            A::Ode(trajectory) => Ok(
                serde_json::json!({"world":world,"family":"ode","status":"trajectory","row_count":trajectory.states.len(),"accepted_steps":trajectory.accepted_steps,"rejected_steps":trajectory.rejected_steps}),
            ),
            A::OuTransient(moments) => Ok(
                serde_json::json!({"world":world,"family":"ou_transient","row_count":moments.times.len()}),
            ),
            A::OuStationary(covariance) => Ok(
                serde_json::json!({"world":world,"family":"ou_stationary","covariance_entries":covariance.len()}),
            ),
            A::OuMechanisms { laws, .. } => match &laws[position] {
                antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) => Ok(
                    serde_json::json!({"world":world,"family":"ou_mechanisms","status":"gaussian_law","dimension":law.mean.len()}),
                ),
                antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Failed {
                    failure,
                    detail,
                } => Ok(
                    serde_json::json!({"world":world,"family":"ou_mechanisms","status":"failed","failure":failure,"detail":detail}),
                ),
            },
            _ => Err(crate::value_err("this cyclic execution has no world view")),
        }
    }

    fn graph_metadata(&self) -> serde_json::Value {
        use antecedent_io::{
            cyclic_family_artifact::CyclicFamilyAnswer as A, cyclic_graph::CyclicGraphReport as G,
        };
        let bounded = |values: &[String]| {
            values
                .iter()
                .take(16)
                .map(|value| value.chars().take(512).collect::<String>())
                .collect::<Vec<_>>()
        };
        match self.family().map(|report| &report.answer) {
            Some(A::GaussianFunctional(report)) => {
                serde_json::json!({"source_identity":report.source_identity,"derivation":bounded(&report.derivation),"derivation_count":report.derivation.len(),"fitted_rows":report.fitted_rows,"covariance_condition_estimate":report.covariance_condition_estimate,"operations":report.operations})
            }
            Some(A::Functional(report)) => {
                serde_json::json!({"source_identity":report.source_identity,"derivation":bounded(&report.derivation),"derivation_count":report.derivation.len()})
            }
            Some(A::Graph(G::Identify { status, program, derivation, .. })) => {
                serde_json::json!({"status":status,"has_functional":program.is_some(),"derivation":bounded(derivation),"derivation_count":derivation.len()})
            }
            Some(A::Graph(G::Calculus {
                status,
                fully_observational,
                program,
                sigma_checks,
                derivation,
                ..
            })) => {
                serde_json::json!({"status":status,"fully_observational":fully_observational,"has_functional":program.is_some(),"sigma_checks":sigma_checks,"derivation":bounded(derivation),"derivation_count":derivation.len()})
            }
            Some(A::Graph(G::Bipartite { separated, assumptions, .. })) => {
                serde_json::json!({"separated":separated,"assumptions":assumptions.chars().take(2048).collect::<String>()})
            }
            Some(A::Graph(G::DomainInvariance { invariant, assumptions })) => {
                serde_json::json!({"invariant":invariant,"assumptions":assumptions.chars().take(2048).collect::<String>()})
            }
            Some(A::Graph(G::Backdoor { valid })) => serde_json::json!({"valid":valid}),
            Some(A::Graph(G::Discover {
                conclusions,
                edges,
                diagnostics,
                interpretation,
                ci_source,
            })) => {
                serde_json::json!({"conclusion_count":conclusions.len(),"edge_count":edges.len(),"diagnostics":bounded(diagnostics),"diagnostic_count":diagnostics.len(),"interpretation":interpretation.chars().take(2048).collect::<String>(),"ci_source":ci_source})
            }
            _ => serde_json::Value::Null,
        }
    }
}

fn recovered_metadata(
    world: u32,
    value: &antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    origin: Option<&antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire>,
) -> serde_json::Value {
    serde_json::json!({"source":"recovered","world":world,"origin":origin.map(origin_metadata),"draw_count":value.draws.len(),"outcome":value.outcome,"total_mass":value.total_mass,"withheld_mass":value.withheld_mass,"mixture_mean":value.mixture_mean,"mixture_variance":value.mixture_variance,"between_draw_variance":value.between_draw_variance})
}

#[pyclass(name = "_PreparedCyclicSystem")]
pub(crate) struct PyPreparedCyclicSystem {
    inner: antecedent::analysis::PreparedStudy<antecedent::analysis::cyclic::CyclicPreparedState>,
}

#[pymethods]
impl PyPreparedCyclicSystem {
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(PyCyclicExecution, NativeFacts)> {
        let ctx = execution_context(max_bytes, cancel);
        let attempt = py.detach(|| {
            antecedent_core::execution_attempt::observe_with_context(&ctx, || {
                self.inner.execute_retained(&ctx)
            })
        });
        let (owner, facts) = crate::inspect_receipt_api::finish_observed(
            py,
            attempt.result.map_err(|e| operation_error(e, &ctx)),
            attempt.facts,
        )?;
        Ok((
            PyCyclicExecution {
                owner: std::sync::Arc::new(CyclicExecutionState::new(
                    CyclicExecutionOwner::System(owner),
                    facts.clone_ref(py),
                )),
            },
            facts,
        ))
    }
    fn request_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.request())
            .map_err(|e| crate::serialization_err(e.to_string()))
    }

    /// Contract of this handle, derived through the `system` family it is carried as
    /// (`PreparedStudy::contract`). Deriving it does not fit.
    fn contract(&self, py: Python<'_>) -> PyResult<String> {
        let contract =
            py.detach(|| self.inner.contract()).map_err(|e| crate::value_err(e.to_string()))?;
        contract_json(&contract)
    }

    fn inspect(&self) -> PyResult<String> {
        let inspected = self.inner.inspect().map_err(|e| crate::value_err(e.to_string()))?;
        Ok(serde_json::json!({"identity":inspected.identity,"fits_performed":inspected.fits_performed,"solve_blocks":inspected.solve_blocks,"execution_plans":inspected.execution_plans}).to_string())
    }
    #[pyo3(signature=(rows_json, observations, *, max_bytes=None, cancel=None))]
    fn refresh_buffers(
        &self,
        py: Python<'_>,
        rows_json: &str,
        observations: PyReadonlyArray2<'_, f64>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        let ctx = execution_context(max_bytes, cancel);
        let _metadata = input_admission(rows_json, &ctx)?;
        let mut rows: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow> =
            serde_json::from_str(rows_json).map_err(|e| operation_error(e, &ctx))?;
        let values = observations.as_array();
        let base = self.inner.request();
        if values.nrows() != rows.len()
            || values.ncols() != base.model.order()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(crate::value_err("cyclic observations shape or values"));
        }
        let bytes = (values.len() as u64).saturating_mul(32);
        if bytes > MAX_CYCLIC_ARTIFACT_BYTES as u64 {
            return Err(crate::value_err("cyclic input byte limit"));
        }
        let _input = ctx.acquire_admission(0, bytes).map_err(|e| operation_error(e, &ctx))?;
        let fit = base
            .fit
            .as_ref()
            .ok_or_else(|| crate::value_err("refresh requires a fitted retained program"))?;
        for (row, data) in rows.iter_mut().zip(values.rows()) {
            row.values = data.iter().copied().collect();
        }
        let request = CyclicExecutionRequest {
            model: base.model.clone(),
            fit: Some(antecedent_io::cyclic_system_artifact::CyclicFitRequest {
                parameters: fit.parameters.clone(),
                rows,
                options: fit.options,
            }),
            worlds: base.worlds.clone(),
            noise: base.noise.clone(),
            initial: base.initial.clone(),
            policy: base.policy,
            factual: base.factual.clone(),
        };
        let inner = py
            .detach(|| antecedent::analysis::Study::prepare_cyclic(request, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(request_json, *, max_bytes=None, cancel=None))]
    fn rebind(
        &self,
        py: Python<'_>,
        request_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(request_json, &ctx)?;
        let request = parse_request(request_json)?;
        let inner =
            py.detach(|| self.inner.rebind(request, &ctx)).map_err(|e| operation_error(e, &ctx))?;
        Ok(Self { inner })
    }

    fn initial_state(&self) -> Vec<f64> {
        self.inner.request().initial.clone()
    }

    /// The same frozen request as a family program (`PreparedStudy::to_family`).
    /// Compilation does not fit; the family keeps its own fit cache.
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn to_family(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyPreparedCyclicFamily> {
        let ctx = execution_context(max_bytes, cancel);
        let attempt = py.detach(|| {
            antecedent_core::execution_attempt::observe_with_context(&ctx, || {
                self.inner.to_family(&ctx)
            })
        });
        crate::inspect_receipt_api::finish_observed(
            py,
            attempt.result.map_err(|e| operation_error(e, &ctx)),
            attempt.facts,
        )
        .map(|(inner, _)| PyPreparedCyclicFamily { inner })
    }

    fn retarget_buffers(
        &self,
        py: Python<'_>,
        worlds_json: &str,
        noise: PyReadonlyArray2<'_, f64>,
        initial: Vec<f64>,
    ) -> PyResult<Self> {
        if worlds_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("cyclic retarget input limit"));
        }
        let worlds =
            serde_json::from_str(worlds_json).map_err(|e| crate::value_err(e.to_string()))?;
        let noise = noise.as_array();
        if noise.ncols() != self.inner.request().model.order()
            || noise.len().saturating_mul(32) > MAX_CYCLIC_ARTIFACT_BYTES
            || noise.iter().any(|x| !x.is_finite())
        {
            return Err(crate::value_err("cyclic noise shape or values"));
        }
        let noise = noise.rows().into_iter().map(|r| r.iter().copied().collect()).collect();
        let inner = py
            .detach(|| self.inner.retarget_worlds(worlds, noise, initial))
            .map_err(|e| crate::value_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn retarget(&self, worlds_json: &str, noise_json: &str, initial_json: &str) -> PyResult<Self> {
        for json in [worlds_json, noise_json, initial_json] {
            if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
                return Err(crate::value_err("cyclic retarget input limit"));
            }
        }
        let worlds =
            serde_json::from_str(worlds_json).map_err(|e| crate::value_err(e.to_string()))?;
        let noise =
            serde_json::from_str(noise_json).map_err(|e| crate::value_err(e.to_string()))?;
        let initial =
            serde_json::from_str(initial_json).map_err(|e| crate::value_err(e.to_string()))?;
        let inner = self
            .inner
            .retarget_worlds(worlds, noise, initial)
            .map_err(|e| crate::value_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn prepare_cyclic_system(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedCyclicSystem> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
        let _input = input_admission(request_json, &ctx)?;
        let request = parse_request(request_json)?;
        let inner = py
            .detach(|| antecedent::analysis::Study::prepare_cyclic(request, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(PyPreparedCyclicSystem { inner })
    });
    crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)
        .map(|(handle, _)| handle)
}

#[pyfunction]
#[pyo3(signature=(metadata, noise, observations, *, max_bytes=None, cancel=None))]
fn prepare_cyclic_system_buffers(
    py: Python<'_>,
    metadata: &str,
    noise: PyReadonlyArray2<'_, f64>,
    observations: PyReadonlyArray2<'_, f64>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedCyclicSystem> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
        let _metadata = input_admission(metadata, &ctx)?;
        let mut request = parse_request(metadata)?;
        let noise = noise.as_array();
        let observations = observations.as_array();
        let n = request.model.order();
        let rows = request.fit.as_ref().map_or(0, |fit| fit.rows.len());
        if noise.ncols() != n || observations.ncols() != n || observations.nrows() != rows {
            return Err(crate::value_err("cyclic numeric buffer shape"));
        }
        let count = noise
            .len()
            .checked_add(observations.len())
            .ok_or_else(|| crate::value_err("cyclic buffer overflow"))?;
        let bytes = (count as u64).saturating_mul(32);
        if bytes > MAX_CYCLIC_ARTIFACT_BYTES as u64 {
            return Err(crate::value_err("cyclic input byte limit"));
        }
        let _input = ctx.acquire_admission(0, bytes).map_err(|e| operation_error(e, &ctx))?;
        if noise.iter().chain(observations.iter()).any(|v| !v.is_finite()) {
            return Err(crate::value_err("cyclic buffers must be finite"));
        }
        request.noise = noise.rows().into_iter().map(|r| r.iter().copied().collect()).collect();
        if let Some(fit) = &mut request.fit {
            for (row, values) in fit.rows.iter_mut().zip(observations.rows()) {
                row.values = values.iter().copied().collect();
            }
        }
        let inner = py
            .detach(|| antecedent::analysis::Study::prepare_cyclic(request, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(PyPreparedCyclicSystem { inner })
    });
    crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)
        .map(|(handle, _)| handle)
}

#[pyclass(name = "_PreparedCyclicFamily")]
pub(crate) struct PyPreparedCyclicFamily {
    inner: antecedent::analysis::PreparedStudy<
        antecedent::analysis::cyclic::CyclicFamilyPreparedState,
    >,
}
#[pymethods]
impl PyPreparedCyclicFamily {
    #[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
    fn rebind(
        &self,
        py: Python<'_>,
        request_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(request_json, &ctx)?;
        let request = serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
        let inner = py
            .detach(|| self.inner.rebind(&request, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(Self { inner })
    }
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(PyCyclicExecution, NativeFacts)> {
        let ctx = execution_context(max_bytes, cancel);
        let attempt = py.detach(|| {
            antecedent_core::execution_attempt::observe_with_context(&ctx, || {
                self.inner.execute_retained(&ctx)
            })
        });
        let (owner, facts) = crate::inspect_receipt_api::finish_observed(
            py,
            attempt.result.map_err(|e| operation_error(e, &ctx)),
            attempt.facts,
        )?;
        Ok((
            PyCyclicExecution {
                owner: std::sync::Arc::new(CyclicExecutionState::new(
                    CyclicExecutionOwner::Family(owner),
                    facts.clone_ref(py),
                )),
            },
            facts,
        ))
    }
    #[pyo3(signature=(schema_json,selectors_json,*,max_bytes=None,cancel=None))]
    fn evidence(
        &self,
        py: Python<'_>,
        schema_json: &str,
        selectors_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCyclicEvidence> {
        let ctx = execution_context(max_bytes, cancel);
        let _schema_input = input_admission(schema_json, &ctx)?;
        let _selectors_input = input_admission(selectors_json, &ctx)?;
        if schema_json.len() > MAX_CYCLIC_ARTIFACT_BYTES
            || selectors_json.len() > MAX_CYCLIC_ARTIFACT_BYTES
        {
            return Err(crate::value_err("cyclic evidence input byte limit"));
        }
        let schema = serde_json::from_str(schema_json).map_err(|e| operation_error(e, &ctx))?;
        let selectors =
            serde_json::from_str(selectors_json).map_err(|e| operation_error(e, &ctx))?;
        let inner = py
            .detach(|| self.inner.cyclic_evidence(schema, selectors, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(PyCyclicEvidence { inner })
    }
    /// Execute and return the intervention worlds keyed by scientific world id
    /// (`PreparedStudy::interventions`), with certificates, solve routes and
    /// failed or withheld mass. A family without worlds is refused with its next attempt.
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn interventions(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(String, NativeFacts)> {
        let ctx = execution_context(max_bytes, cancel);
        let attempt = py.detach(|| {
            antecedent_core::execution_attempt::observe_with_context(&ctx, || {
                self.inner.interventions(&ctx)
            })
        });
        let (worlds, facts) = crate::inspect_receipt_api::finish_observed(
            py,
            attempt.result.map_err(|e| operation_error(e, &ctx)),
            attempt.facts,
        )?;
        Ok((
            serde_json::to_string(&worlds).map_err(|e| crate::serialization_err(e.to_string()))?,
            facts,
        ))
    }
    /// Cheap inspection. `design_unresolved` lists the coefficient rows the clamp
    /// design of an experimental recovery leaves unresolved, known before any fit
    /// (`PreparedStudy::design_unresolved`); it is null for every other family.
    fn inspect(&self) -> String {
        let inspection = self.inner.inspect();
        serde_json::json!({"identity":inspection.identity,"family":inspection.family,"preparation_evaluations":inspection.preparation_evaluations,"retained_bytes":inspection.retained_bytes,"execution_plans":inspection.execution_plans,"design_unresolved":self.inner.design_unresolved()}).to_string()
    }
    /// Contract of this handle (`PreparedStudy::contract`): layered identities, the
    /// `cyclic.solution.v1` tag, the recovery layer of an experimental recovery and
    /// the layer payloads. Deriving it does not fit, sample or solve.
    fn contract(&self) -> PyResult<String> {
        let contract = self.inner.contract().map_err(|e| crate::value_err(e.to_string()))?;
        contract_json(&contract)
    }
}

/// Compile a linear law to be recovered from clamp experiments
/// (`Study::prepare_experimental_recovery`). Any other family is refused.
/// Compilation admits the worlds on the retained units and does not fit.
#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn prepare_cyclic_experimental_recovery(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedCyclicFamily> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
        let _input = input_admission(request_json, &ctx)?;
        let request = serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
        let inner = py
            .detach(|| antecedent::analysis::Study::prepare_experimental_recovery(&request, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(PyPreparedCyclicFamily { inner })
    });
    crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)
        .map(|(handle, _)| handle)
}
#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn prepare_cyclic_family(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedCyclicFamily> {
    let ctx = execution_context(max_bytes, cancel);
    let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
        let _input = input_admission(request_json, &ctx)?;
        if request_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("cyclic family request limit"));
        }
        let request = serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
        let inner = py
            .detach(|| antecedent::analysis::Study::prepare_cyclic_family(&request, &ctx))
            .map_err(|e| operation_error(e, &ctx))?;
        Ok(PyPreparedCyclicFamily { inner })
    });
    crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)
        .map(|(handle, _)| handle)
}

struct PythonLocalMechanism {
    id: String,
    reads: Vec<usize>,
    callback: Py<PyAny>,
}
impl antecedent_model::local_cyclic::LocalCyclicMechanism for PythonLocalMechanism {
    fn id(&self) -> &str {
        &self.id
    }
    fn reads(&self) -> &[usize] {
        &self.reads
    }
    fn evaluate(
        &self,
        inputs: &[f64],
    ) -> Result<(f64, Vec<f64>), antecedent_model::nonlinear_equilibrium::EquilibriumError> {
        Python::attach(|py| {
            self.callback.bind(py).call1((inputs.to_vec(),))?.extract::<(f64, Vec<f64>)>()
        })
        .map_err(|error| {
            antecedent_model::nonlinear_equilibrium::EquilibriumError::CallbackFailed {
                mechanism: self.id.clone(),
                failure: crate::interrupt::callback_failure("cyclic callback raised", error),
            }
        })
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalRequest {
    mechanisms: Vec<(String, Vec<usize>)>,
    noise: Vec<f64>,
    clamps: Vec<antecedent_model::nonlinear_equilibrium::Clamp>,
    initial: Vec<f64>,
    factual: Option<Vec<f64>>,
    policy: antecedent_model::local_cyclic::LocalCyclicPolicy,
}
/// Process-local execution retains no executable artifact representation.
#[pyfunction]
#[pyo3(signature = (request_json, callbacks, *, max_bytes=None, cancel=None))]
fn execute_local_cyclic(
    py: Python<'_>,
    request_json: &str,
    callbacks: Vec<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    if request_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("local cyclic request limit"));
    }
    let request: LocalRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    if callbacks.len() != request.mechanisms.len() {
        return Err(crate::value_err("local callback count"));
    }
    let model = antecedent_model::local_cyclic::LocalCyclicModel {
        mechanisms: request
            .mechanisms
            .into_iter()
            .zip(callbacks)
            .map(|((id, reads), callback)| {
                Box::new(PythonLocalMechanism { id, reads, callback })
                    as Box<dyn antecedent_model::local_cyclic::LocalCyclicMechanism>
            })
            .collect(),
    };
    crate::detach_catch(py, move || {
        let execute = || {
            let inspection = model.inspect(&request.clamps)?;
            let result = model.solve(
                &request.noise,
                &request.clamps,
                &request.initial,
                request.policy,
                &ctx,
            )?;
            let counterfactual = request
                .factual
                .as_ref()
                .map(|factual| {
                    let shared = model.abduce(factual, request.policy, &ctx)?;
                    model.solve(&shared, &request.clamps, factual, request.policy, &ctx)
                })
                .transpose()?;
            serde_json::to_string(&serde_json::json!({"inspection":inspection,"result":result,"counterfactual":counterfactual}))
                .map_err(|e| antecedent_model::nonlinear_equilibrium::EquilibriumError::Invalid(e.to_string()))
        };
        execute().map_err(|e| operation_error(e, &ctx))
    })
}

#[pyclass(name = "_CyclicEvidence")]
struct PyCyclicEvidence {
    inner: antecedent::analysis::cyclic_evidence::CyclicEvidence,
}
#[pymethods]
impl PyCyclicEvidence {
    fn identity(&self) -> &str {
        self.inner.identity()
    }

    #[pyo3(signature=(*,offset=0,limit=64))]
    fn metadata_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        if limit > 256 {
            return Err(crate::value_err("cyclic evidence metadata page limit exceeds256"));
        }
        let columns = self.inner.columns();
        let end = offset.saturating_add(limit).min(columns.len());
        let page = columns
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .map(|column| {
                serde_json::json!({
                    "quantity":column.quantity,"envelope_identity":column.envelope_identity,
                    "port":column.port,"uncertainty_axes":column.uncertainty_axes,
                    "world":column.world,"time":column.time,"sample_count":column.samples.len(),
                    "enclosure_count":column.numerical_enclosures.len(),
                    "mass_bound_count":column.mass_bounds.as_ref().map_or(0,Vec::len),
                })
            })
            .collect::<Vec<_>>();
        Ok(serde_json::json!({"identity":self.inner.identity(),"source_identity":self.inner.source_identity(),"scope":self.inner.scope(),"source_bridge":self.inner.system_bridge(),"recovery_origins":self.inner.report().recovery_origins.as_ref(),"column_count":columns.len(),"offset":offset,"columns":page,"next_offset":(end<columns.len()).then_some(end)}).to_string())
    }

    fn report(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.report())
            .map_err(|e| crate::serialization_err(e.to_string()))
    }
    #[pyo3(signature=(*,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = execution_context(max_bytes, cancel);
        let bytes = py.detach(|| self.inner.export(&ctx)).map_err(|e| operation_error(e, &ctx))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(*,others=None,dependence=None,max_bytes=None,cancel=None))]
    fn joint_law(
        &self,
        others: Option<Vec<PyRef<'_, PyCyclicEvidence>>>,
        dependence: Option<&str>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<crate::distribution_api::PyJointDistributionArtifact> {
        let ctx = execution_context(max_bytes, cancel);
        if ctx.cancellation.is_cancelled() {
            return Err(crate::CausalCancelledError::new_err("cancelled"));
        }
        if dependence.is_some_and(|value| !matches!(value, "aligned" | "independent")) {
            return Err(crate::value_err("dependence must be aligned or independent"));
        }
        let others = others.unwrap_or_default();
        if others.len() > 256 {
            return Err(crate::value_err("at most 256 additional evidence products"));
        }
        let references = others.iter().map(|item| &item.inner).collect::<Vec<_>>();
        let samples = std::iter::once(&self.inner)
            .chain(references.iter().copied())
            .flat_map(|item| item.report().columns.iter())
            .try_fold(0_u64, |sum, column| sum.checked_add(column.samples.len() as u64))
            .ok_or_else(|| {
                crate::CausalResourceError::new_err("joint law sample count overflow")
            })?;
        let _lease = ctx
            .acquire_admission(
                0,
                samples.checked_mul(32).ok_or_else(|| {
                    crate::CausalResourceError::new_err("joint law allocation overflow")
                })?,
            )
            .map_err(|error| operation_error(error, &ctx))?;
        let artifact = self
            .inner
            .joint_law_with(&references, dependence)
            .map_err(|error| operation_error(error, &ctx))?;
        Ok(crate::distribution_api::PyJointDistributionArtifact { artifact, source_claim: None })
    }

    #[pyo3(signature=(contract_json, *, max_bytes=None, cancel=None))]
    fn decide_enclosures(
        &self,
        contract_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(contract_json, &ctx)?;
        let sample_slots = self
            .inner
            .report()
            .columns
            .iter()
            .map(|column| column.samples.len() as u64)
            .sum::<u64>();
        let _workspace = ctx
            .acquire_admission(
                0,
                sample_slots
                    .saturating_mul(32)
                    .saturating_add((contract_json.len() as u64).saturating_mul(8)),
            )
            .map_err(|e| operation_error(e, &ctx))?;
        if contract_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("decision contract byte limit"));
        }
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| operation_error(e, &ctx))?;
        antecedent_core::numerical_admission::with_context(&ctx, || {
            self.inner.decide_enclosures(&contract)
        })
        .map(|v| v.to_string())
        .map_err(|e| operation_error(e, &ctx))
    }
    #[pyo3(signature=(contract_json, *, max_bytes=None, cancel=None))]
    fn decide_means(
        &self,
        contract_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(contract_json, &ctx)?;
        let sample_slots = self
            .inner
            .report()
            .columns
            .iter()
            .map(|column| column.samples.len() as u64)
            .sum::<u64>();
        let _workspace = ctx
            .acquire_admission(
                0,
                sample_slots
                    .saturating_mul(32)
                    .saturating_add((contract_json.len() as u64).saturating_mul(8)),
            )
            .map_err(|e| operation_error(e, &ctx))?;
        if contract_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("decision contract byte limit"));
        }
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| operation_error(e, &ctx))?;
        antecedent_core::numerical_admission::with_context(&ctx, || {
            self.inner.decide_means(&contract)
        })
        .map(|v| v.to_string())
        .map_err(|e| operation_error(e, &ctx))
    }
}
#[pyfunction]
#[pyo3(signature=(bytes,expected_identity,*,max_bytes=None,cancel=None))]
fn consume_cyclic_evidence(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCyclicEvidence> {
    let ctx = execution_context(max_bytes, cancel);
    let inner = py
        .detach(|| {
            antecedent::analysis::cyclic_evidence::CyclicEvidence::consume(
                bytes,
                expected_identity,
                &ctx,
            )
        })
        .map_err(|e| replay_error(e, &ctx))?;
    Ok(PyCyclicEvidence { inner })
}

fn origin_metadata(
    origin: &antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire,
) -> serde_json::Value {
    use antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire as O;
    match origin {
        O::Supplied => serde_json::json!({"origin":"supplied"}),
        O::AsymptoticNormal { draws, seed } => {
            serde_json::json!({"origin":"asymptotic_normal","draws":draws,"seed":seed})
        }
        O::Bootstrap { replicates, seed, design, rule } => {
            serde_json::json!({"origin":"bootstrap","replicates":replicates,"seed":seed,"design":design,"rule":rule})
        }
        O::Posterior { draws, burn_in, seed, .. } => {
            serde_json::json!({"origin":"posterior","draws":draws,"burn_in":burn_in,"seed":seed,"prior":"retained_in_source_declaration"})
        }
    }
}

impl CyclicExecutionOwner {
    fn slot_metadata(
        &self,
        world: u32,
        row: usize,
        draw: usize,
        counterfactual: bool,
    ) -> PyResult<serde_json::Value> {
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer as A, CyclicWorldOutcomeAnswer as W,
        };
        let position = self.position(world)?;
        let invalid = || crate::value_err("cyclic slot coordinate out of range");
        if let Some(system) = self.system() {
            if draw != 0 {
                return Err(invalid());
            }
            let value = if counterfactual {
                if row != 0 {
                    return Err(invalid());
                }
                system.counterfactuals.as_ref().and_then(|rows| rows.get(position))
            } else {
                system.worlds[position].get(row)
            }
            .ok_or_else(invalid)?;
            return Ok(solution_metadata(value));
        }
        let answer = &self.family().ok_or_else(invalid)?.answer;
        if counterfactual && !matches!(answer, A::EquilibriumActions(_)) {
            return Err(invalid());
        }
        match answer {
            A::Worlds(worlds) if draw == 0 => match &worlds[position].outcome {
                W::EquilibriumDraws { outcomes } => {
                    outcomes.get(row).map(equilibrium_metadata).ok_or_else(invalid)
                }
                W::GaussianLaw { residual_inf, .. } if row == 0 => {
                    Ok(serde_json::json!({"status":"gaussian_law","residual_inf":residual_inf}))
                }
                W::Trajectory { trajectory } => trajectory
                    .times
                    .get(row)
                    .map(|time| serde_json::json!({"status":"trajectory","time":time}))
                    .ok_or_else(invalid),
                W::NoSolution { cause, next_attempt } if row == 0 => Ok(
                    serde_json::json!({"status":"no_solution","cause":cause,"next_attempt":next_attempt}),
                ),
                W::Withheld if row == 0 => Ok(serde_json::json!({"status":"withheld"})),
                _ => Err(invalid()),
            },
            A::EquilibriumActions(worlds) if draw == 0 => {
                let value = if counterfactual {
                    if row != 0 {
                        return Err(invalid());
                    }
                    worlds[position].counterfactual.as_ref()
                } else {
                    worlds[position].rows.get(row)
                }
                .ok_or_else(invalid)?;
                Ok(equilibrium_metadata(value))
            }
            A::ParameterPosterior { worlds, .. } => worlds[position]
                .get(draw)
                .and_then(|rows| rows.get(row))
                .map(equilibrium_metadata)
                .ok_or_else(invalid),
            A::FrequentistBootstrap { worlds, .. } => {
                match worlds[position].get(draw).ok_or_else(invalid)? {
                    Some(rows) => rows.get(row).map(equilibrium_metadata).ok_or_else(invalid),
                    None => Ok(serde_json::json!({"status":"failed_fit"})),
                }
            }
            A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. }
                if row == 0 =>
            {
                worlds[position].get(draw).map(equilibrium_metadata).ok_or_else(invalid)
            }
            A::RecoveredLinear(report) if row == 0 => {
                recovered_slot_metadata(&report.worlds[position], draw)
            }
            A::ExperimentalRecovery(report) if row == 0 => {
                recovered_slot_metadata(&report.worlds[position].answer, draw)
            }
            A::Ode(trajectory) if draw == 0 => trajectory
                .times
                .get(row)
                .map(|time| serde_json::json!({"status":"trajectory","time":time}))
                .ok_or_else(invalid),
            A::OuTransient(moments) if draw == 0 => moments
                .times
                .get(row)
                .map(|time| serde_json::json!({"status":"gaussian_law","time":time}))
                .ok_or_else(invalid),
            A::OuStationary(_) if draw == 0 && row == 0 => {
                Ok(serde_json::json!({"status":"stationary_covariance"}))
            }
            A::OuMechanisms { laws, .. } if draw == 0 && row == 0 => match &laws[position] {
                antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Stationary(_) => {
                    Ok(serde_json::json!({"status":"gaussian_law"}))
                }
                antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Failed {
                    failure,
                    detail,
                } => Ok(serde_json::json!({"status":"failed","failure":failure,"detail":detail})),
            },
            _ => Err(invalid()),
        }
    }
}

fn solution_metadata(
    value: &antecedent_model::nonlinear_equilibrium::EquilibriumResult,
) -> serde_json::Value {
    serde_json::json!({"status":"solved","dimension":value.values.len(),"residual":value.residual,"iterations":value.iterations,"globally_unique":value.globally_unique,"existence_certified":value.existence_certified,"numerical_converged":value.numerical_converged,"dynamic_stability":value.dynamic_stability,"jacobian_condition_estimate":value.jacobian_condition_estimate,"blocks":value.blocks})
}

fn equilibrium_metadata(
    value: &antecedent_model::nonlinear_equilibrium::EquilibriumOutcome,
) -> serde_json::Value {
    match value {
        antecedent_model::nonlinear_equilibrium::EquilibriumOutcome::Solved(value) => {
            solution_metadata(value)
        }
        antecedent_model::nonlinear_equilibrium::EquilibriumOutcome::Failed(value) => {
            serde_json::json!({"status":"failed","failure":value.failure,"detail":value.detail})
        }
    }
}

fn recovered_slot_metadata(
    world: &antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    draw: usize,
) -> PyResult<serde_json::Value> {
    use antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer as D;
    let value = world
        .draws
        .get(draw)
        .ok_or_else(|| crate::value_err("cyclic recovered draw out of range"))?;
    let outcome = match &value.outcome {
        D::Equilibrium { mean, solved_by, .. } => {
            serde_json::json!({"status":"equilibrium","dimension":mean.len(),"solved_by":solved_by})
        }
        D::Inadmissible { reason } => serde_json::json!({"status":"inadmissible","reason":reason}),
        D::NotCertified { fault } => serde_json::json!({"status":"not_certified","fault":fault}),
    };
    Ok(serde_json::json!({"weight":value.weight,"outcome":outcome}))
}

impl CyclicExecutionOwner {
    fn contract_metadata(
        &self,
    ) -> Option<&antecedent_io::contract_section::cyclic::CyclicContractWire> {
        match self {
            Self::System(owner) => Some(&owner.contract_metadata().section),
            Self::Family(owner) => Some(&owner.contract_metadata().section),
            Self::ConsumedFamily(owner) => owner.contract_metadata(),
            Self::ConsumedSystem(_) => None,
        }
    }

    fn contract_summary(&self) -> serde_json::Value {
        use antecedent_io::contract_section::cyclic::{
            CyclicIdentificationWire as I, CyclicTargetWire as T,
        };
        let Some(contract) = self.contract_metadata() else {
            return serde_json::Value::Null;
        };
        let hex = |bytes: [u8; 32]| antecedent_core::SemanticDigest::from_bytes(bytes).to_hex();
        let ids = &contract.identities;
        let identification = match &contract.identification {
            I::SpecifiedModelConditional { model } => {
                serde_json::json!({"kind":contract.identification.kind(),"model":hex(*model)})
            }
            I::ExperimentalRecovery { design, unresolved } => {
                serde_json::json!({"kind":contract.identification.kind(),"design":hex(*design),"unresolved_count":unresolved.len()})
            }
            I::CyclicIoScm { source } => {
                serde_json::json!({"kind":contract.identification.kind(),"source":hex(*source)})
            }
        };
        let count = assumption_count(&contract.target);
        let assumptions =
            serde_json::json!({"count":count,"operation":"assumptions_json","paged":true});
        let target = match &contract.target {
            T::Worlds { .. } => {
                serde_json::json!({"kind":"worlds","target_count":1,"assumptions":assumptions})
            }
            T::JointWorlds { targets } => {
                serde_json::json!({"kind":"joint_worlds","target_count":targets.len(),"assumptions":assumptions})
            }
            T::Functional { kind, outcome, query } => {
                serde_json::json!({"kind":kind,"outcome":outcome,"query":hex(*query)})
            }
        };
        let lowered = match self {
            Self::System(owner) => {
                Some(owner.contract_metadata().lowered_program_identity.to_hex())
            }
            Self::Family(owner) => {
                Some(owner.contract_metadata().lowered_program_identity.to_hex())
            }
            _ => None,
        };
        serde_json::json!({
            "family":contract.family,"request_identity":contract.request_identity,
            "identification_kind":contract.identification.kind(),
            "identities":{"target":hex(ids.target),"identification":hex(ids.identification),"identification_product":ids.identification_product.map(hex),"program":hex(ids.program),"inference_binding":hex(ids.inference_binding),"observation":hex(ids.observation),"data_snapshot":hex(ids.data_snapshot)},
            "cyclic_solution":hex(contract.cyclic_solution),"recovery":contract.recovery.map(hex),
            "lowered_program_identity":lowered,
            "layers":{"identification":identification,"target":target,"inference_binding":{"method":contract.inference_binding.method,"settings":hex(contract.inference_binding.settings)},"program":{"family":contract.program.family,"realization":contract.program.realization,"fit_dependencies":hex(contract.program.fit_dependencies)},"observation":{"modality":contract.observation.modality,"variable_count":contract.observation.variables.len(),"unit_count":contract.observation.units.len()}},
        })
    }
}

fn assumption_count(target: &antecedent_io::contract_section::cyclic::CyclicTargetWire) -> usize {
    use antecedent_io::contract_section::cyclic::CyclicTargetWire as T;
    match target {
        T::Worlds { target } => target.assumptions().len(),
        T::JointWorlds { targets } => targets.iter().map(|target| target.assumptions().len()).sum(),
        T::Functional { .. } => 0,
    }
}

impl CyclicExecutionOwner {
    fn recovery(
        &self,
    ) -> Option<&antecedent_io::cyclic_family_artifact::ExperimentalRecoveryReport> {
        match self.family().map(|report| &report.answer) {
            Some(
                antecedent_io::cyclic_family_artifact::CyclicFamilyAnswer::ExperimentalRecovery(
                    report,
                ),
            ) => Some(report),
            _ => None,
        }
    }

    fn recovery_summary(&self) -> Option<serde_json::Value> {
        use antecedent_io::cyclic_family_artifact::RecoveredDrawStatusWire as S;
        let report = self.recovery()?;
        let (mut complete, mut unresolved, mut failed, mut marked) =
            (0usize, 0usize, 0usize, 0usize);
        for status in &report.law.draw_status {
            match status {
                S::Complete { .. } => complete += 1,
                S::Unresolved { .. } => unresolved += 1,
                S::Failed { .. } => failed += 1,
                S::Marked => marked += 1,
            }
        }
        Some(
            serde_json::json!({"draw_count":report.law.draws.len(),"node_count":report.law.nodes.len(),"failed_mass":report.law.failed_mass,"draw_status_counts":{"complete":complete,"unresolved":unresolved,"failed":failed,"marked":marked},"unresolved_direction_count":report.law.draws.iter().map(|draw|draw.unresolved.len()).sum::<usize>(),"draw_origin":origin_metadata(&report.law.origin),"law_meaning":report.law.origin.law_meaning()}),
        )
    }
}

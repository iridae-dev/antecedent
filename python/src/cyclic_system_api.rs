// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nonlinear cyclic systems: bounded Rust execution, fitting and artifact replay.

mod errors;
mod evidence;
mod local;
mod metadata;

use antecedent::analysis::CyclicExecutionRequest;
use antecedent::io::cyclic_system_artifact::MAX_CYCLIC_ARTIFACT_BYTES;
use antecedent::prelude::ExecutionContext;
use errors::{operation_error, replay_error, scientific_error};
use evidence::PyCyclicEvidence;
use metadata::{assumption_count, origin_metadata};
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

fn input_admission(
    json: &str,
    ctx: &ExecutionContext,
) -> PyResult<antecedent::execution::Retention> {
    if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic input byte limit"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("cancelled"));
    }
    antecedent::execution::retain(ctx, (json.len() as u64).saturating_mul(4))
        .map_err(|e| operation_error(e, ctx))
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
    let attempt = antecedent::execution::observe(&ctx, || {
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
        antecedent::execution::observe(&ctx, || {
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
    ctx.rng = antecedent::execution::RngFactory::from_seed(seed);
    let _input = input_admission(request_json, &ctx)?;
    let request = parse_request(request_json)?;
    let fit = request.fit.as_ref().ok_or_else(|| {
        crate::value_err("posterior requires retained data/parameter coordinates")
    })?;
    let report = py
        .detach(|| {
            antecedent::estimate::cyclic_likelihood::sample_cyclic_posterior(
                &request.model,
                &fit.parameters,
                &fit.rows,
                prior_sd,
                antecedent::inference::metropolis::MetropolisOptions {
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
    evidence::register_consumer(m)?;
    local::register(m)?;
    m.add_class::<PyPreparedCyclicSystem>()?;
    m.add_class::<PyPreparedCyclicFamily>()?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_experimental_recovery, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_system_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(execute_cyclic_system, m)?)?;
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
    let hex = antecedent::scientific::SemanticDigest::to_hex;
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
        antecedent::execution::observe(&ctx, || {
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
    fn system(&self) -> Option<&antecedent::analysis::CyclicExecutionReport> {
        use antecedent::analysis::CyclicFamilyAnswer as A;
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

    fn family(&self) -> Option<&antecedent::analysis::CyclicFamilyReport> {
        match self {
            Self::Family(owner) => Some(owner.report()),
            Self::ConsumedFamily(owner) => Some(owner.report()),
            _ => None,
        }
    }

    fn family_request(&self) -> Option<&antecedent::analysis::CyclicFamilyRequest> {
        match self {
            Self::Family(owner) => Some(owner.request()),
            Self::ConsumedFamily(owner) => Some(owner.request()),
            _ => None,
        }
    }

    fn source_identity(&self) -> &str {
        use antecedent::analysis::CyclicFamilyAnswer as A;
        match self.family().map(|report| &report.answer) {
            Some(A::GaussianFunctional(report)) => &report.source_identity,
            Some(A::Functional(report)) => &report.source_identity,
            _ => self.identity(),
        }
    }

    fn recovered_origin(&self) -> Option<&antecedent::analysis::RecoveredDrawOriginWire> {
        use antecedent::analysis::{CyclicFamilyAnswer as A, CyclicFamilyRequest as R};
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
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
            Some(A::SdeAnalysis(report)) => report.simulation.worlds.len(),
            Some(A::Ode(_) | A::OuStationary(_) | A::OuTransient(_)) => 1,
            _ => 0,
        }
    }

    fn world_id(&self, position: usize) -> PyResult<u32> {
        use antecedent::analysis::CyclicFamilyAnswer as A;
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
        use antecedent::io::cyclic_family_artifact::CyclicWorldOutcomeAnswer as W;
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
                W::GaussianLaw { .. } => Err(invalid()),
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
            A::SdeAnalysis(report) if draw == 0 => {
                let world = report.simulation.worlds.get(position).ok_or_else(invalid)?;
                match field {
                    "mean" => Ok(Some(world.mean.get(row).ok_or_else(invalid)?)),
                    "variance" => Ok(Some(world.variance.get(row).ok_or_else(invalid)?)),
                    "covariance" => Ok(Some(world.covariance.get(row).ok_or_else(invalid)?)),
                    "monte_carlo_standard_error" => {
                        Ok(Some(world.monte_carlo_standard_error.get(row).ok_or_else(invalid)?))
                    }
                    "times" if row == 0 => Ok(Some(&report.simulation.times)),
                    _ => Err(invalid()),
                }
            }
            A::Ode(trajectory) if draw == 0 => match field {
                "values" => Ok(Some(trajectory.states.get(row).ok_or_else(invalid)?)),
                "times" if row == 0 => Ok(Some(&trajectory.times)),
                _ => Err(invalid()),
            },
            A::OuMechanisms { laws, .. } if draw == 0 && row == 0 => match &laws[position] {
                antecedent::estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) => {
                    match field {
                        "mean" => Ok(Some(&law.mean)),
                        "covariance" => Ok(Some(&law.covariance)),
                        _ => Err(invalid()),
                    }
                }
                antecedent::estimate::ou_mechanisms::OuMechanismOutcome::Failed { .. } => Ok(None),
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
    outcome: &'a antecedent::gcm::nonlinear_equilibrium::EquilibriumOutcome,
    field: &str,
) -> PyResult<Option<&'a [f64]>> {
    use antecedent::gcm::nonlinear_equilibrium::EquilibriumOutcome as E;
    match outcome {
        E::Solved(value) if field == "values" => Ok(Some(&value.values)),
        E::Failed(_) => Ok(None),
        E::Solved(_) => Err(crate::value_err("unknown equilibrium numeric field")),
    }
}

fn recovered_values<'a>(
    world: &'a antecedent::io::cyclic_family_artifact::RecoveredWorldAnswer,
    draw: usize,
    row: usize,
    field: &str,
) -> PyResult<Option<&'a [f64]>> {
    use antecedent::io::cyclic_family_artifact::RecoveredDrawAnswer as D;
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

    #[allow(
        clippy::unnecessary_wraps,
        reason = "the PyO3 method retains the binding's fallible return convention"
    )]
    fn metadata_json(&self) -> PyResult<String> {
        let system = self.owner.system();
        let retained = match self.owner.family() {
            Some(report) => report.retained_admitted_bytes(),
            None => system
                .map_or(0, antecedent::analysis::CyclicExecutionReport::retained_admitted_bytes),
        };
        Ok(serde_json::json!({
            "identity":self.owner.identity(), "family":self.owner.family_name(),
            "claim":system.map_or(self.owner.scope(), |report|report.claim.as_str()),
            "calibration":system.map_or("not_measured", |report|report.calibration.as_str()),
            "source_identity":self.owner.source_identity(), "scope":self.owner.scope(),
            "draw_origin":self.owner.recovered_origin().map(origin_metadata),
            "law_meaning":self.owner.recovered_origin().and_then(antecedent::analysis::RecoveredDrawOriginWire::law_meaning),
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
        use antecedent::io::contract_section::cyclic::CyclicTargetWire as T;
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
             target: &antecedent::io::scientific_target_wire::ScientificTargetWire| {
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
        use antecedent::analysis::CyclicFamilyRequest as R;
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
            antecedent::reason::reason_code!("route_not_supported"),
            "cyclic_family.intervention_worlds_unavailable",
            &format!("family `{}` has no declared intervention worlds", self.owner.family_name()),
            "declare the worlds as typed interventions with `linear_worlds`, `equilibrium_worlds`, `ode_worlds`, `recovered_linear` or `experimental_recovery`",
        ))
    }

    fn world_view_kind(&self) -> Option<&'static str> {
        use antecedent::analysis::{CyclicFamilyAnswer as A, CyclicFamilyRequest as R};
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
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
                let _output = antecedent::execution::retain(&ctx, bytes.len() as u64)
                    .map_err(|error| operation_error(error, &ctx))?;
                Ok(PyBytes::new(py, bytes).unbind())
            }
            CyclicExecutionOwner::ConsumedFamily(owner) => {
                let bytes = owner.artifact();
                let _output = antecedent::execution::retain(&ctx, bytes.len() as u64)
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
        use antecedent::io::cyclic_family_artifact::CyclicWorldOutcomeAnswer as W;
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
                    W::GaussianLaw { mean, block_diagnostics, .. } => {
                        serde_json::json!({"status":"gaussian_law","dimension":mean.len(),"block_diagnostics":block_diagnostics})
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
            A::SdeAnalysis(report) => {
                let ensemble = &report.simulation.worlds[position];
                Ok(serde_json::json!({
                    "world": world,
                    "family": "sde_analysis",
                    "status": "ensemble_moments",
                    "world_id": ensemble.id,
                    "row_count": report.simulation.times.len(),
                    "dimension": ensemble.mean.first().map(Vec::len),
                    "paths": report.simulation.paths,
                    "scheme": report.simulation.scheme,
                    "step": report.simulation.step,
                    "seed": report.simulation.seed,
                    "step_converged": ensemble.numerical.as_ref().map(|n| n.step_converged),
                    "law_meaning": "monte_carlo_ensemble_of_a_specified_sde",
                }))
            }
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
                antecedent::estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) => Ok(
                    serde_json::json!({"world":world,"family":"ou_mechanisms","status":"gaussian_law","dimension":law.mean.len()}),
                ),
                antecedent::estimate::ou_mechanisms::OuMechanismOutcome::Failed {
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
        use antecedent::analysis::CyclicFamilyAnswer as A;
        use antecedent::io::cyclic_graph::CyclicGraphReport as G;
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
    value: &antecedent::io::cyclic_family_artifact::RecoveredWorldAnswer,
    origin: Option<&antecedent::analysis::RecoveredDrawOriginWire>,
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
        let attempt = py
            .detach(|| antecedent::execution::observe(&ctx, || self.inner.execute_retained(&ctx)));
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

    fn inspect(&self, py: Python<'_>) -> PyResult<String> {
        let inspected = crate::detach_catch(py, || {
            self.inner.inspect().map_err(|e| crate::value_err(e.to_string()))
        })?;
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
        let mut rows: Vec<antecedent::gcm::nonlinear_equilibrium::RegimeRow> =
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
        let _input =
            antecedent::execution::retain(&ctx, bytes).map_err(|e| operation_error(e, &ctx))?;
        let fit = base
            .fit
            .as_ref()
            .ok_or_else(|| crate::value_err("refresh requires a fitted retained program"))?;
        for (row, data) in rows.iter_mut().zip(values.rows()) {
            row.values = data.iter().copied().collect();
        }
        let request = CyclicExecutionRequest {
            model: base.model.clone(),
            fit: Some(antecedent::analysis::CyclicFitRequest {
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
        let attempt =
            py.detach(|| antecedent::execution::observe(&ctx, || self.inner.to_family(&ctx)));
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
    let attempt = antecedent::execution::observe(&ctx, || {
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
    let attempt = antecedent::execution::observe(&ctx, || {
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
        let _input =
            antecedent::execution::retain(&ctx, bytes).map_err(|e| operation_error(e, &ctx))?;
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
        let attempt = py
            .detach(|| antecedent::execution::observe(&ctx, || self.inner.execute_retained(&ctx)));
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
        let attempt =
            py.detach(|| antecedent::execution::observe(&ctx, || self.inner.interventions(&ctx)));
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
    let attempt = antecedent::execution::observe(&ctx, || {
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
    let attempt = antecedent::execution::observe(&ctx, || {
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

impl CyclicExecutionOwner {
    fn recovery(
        &self,
    ) -> Option<&antecedent::io::cyclic_family_artifact::ExperimentalRecoveryReport> {
        match self.family().map(|report| &report.answer) {
            Some(antecedent::analysis::CyclicFamilyAnswer::ExperimentalRecovery(report)) => {
                Some(report)
            }
            _ => None,
        }
    }

    fn recovery_summary(&self) -> Option<serde_json::Value> {
        use antecedent::io::cyclic_family_artifact::RecoveredDrawStatusWire as S;
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

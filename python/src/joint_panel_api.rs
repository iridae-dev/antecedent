// SPDX-License-Identifier: MIT OR Apache-2.0
//! Python declarations delegate to the public retained joint-panel compiler.
use antecedent::analysis::{
    PreparedStudy,
    joint_panel::{
        self as facade, JointPanelColumns, JointPanelFit, JointPanelFunctional, JointPanelRequest,
        JointPanelState, JointPanelWorld,
    },
};
use antecedent_core::{ExecutionContext, InferenceSelection, ScientificWorldId, VariableId};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_estimate::joint_panel::{
    self as engine, PanelChannelHistory, PanelChannelProgram, PanelFitOptions, PanelProgram,
    PanelScenario,
};
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::Arc};
mod designs;
mod experiment;
mod search;
mod selection;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceGroup {
    id: u32,
    columns: Vec<Vec<Option<f64>>>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    group: u32,
    history: BTreeMap<String, [f64; 2]>,
    confounder: Vec<f64>,
    season: Vec<usize>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct World {
    id: String,
    reference: bool,
    channels: BTreeMap<String, Vec<f64>>,
    scenarios: Vec<Scenario>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Feature {
    id: String,
    expression: serde_json::Value,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Functional {
    terms: Vec<(u32, f64)>,
    reference: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecompositionDeclaration {
    #[serde(default)]
    require_identified_reference: bool,
    before_world: u32,
    after_world: u32,
    before_population_weights: Vec<f64>,
    after_population_weights: Vec<f64>,
    time_weights: Vec<f64>,
    order: [facade::ChangeComponent; 4],
    periods: facade::MechanismPeriodMap,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CertificateControlDeclaration {
    channel: String,
    offset: i32,
    lower: f64,
    upper: f64,
    unit_cost: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CertificateRestrictionDeclaration {
    id: String,
    coefficients: Vec<f64>,
    bound: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CertificateDeclaration {
    base_world: u32,
    controls: Vec<CertificateControlDeclaration>,
    initial: Vec<f64>,
    budget: Option<f64>,
    restrictions: Vec<CertificateRestrictionDeclaration>,
    functional: ContrastFunctional,
    utility_offset: f64,
    utility_scale: f64,
    options: antecedent_design::program_search::certified::CertificateOptions,
}
/// Time functional applied identically to every scheduled program before contrasting.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum ContrastFunctional {
    TerminalMean,
    WindowAverage { from: usize, until: usize },
    WindowChange { before: (usize, usize), after: (usize, usize) },
    WeightedSum { coefficients: Vec<f64> },
}
impl ContrastFunctional {
    fn native(self) -> facade::JointPanelResponseFunctional {
        match self {
            Self::TerminalMean => facade::JointPanelResponseFunctional::TerminalMean,
            Self::WindowAverage { from, until } => {
                facade::JointPanelResponseFunctional::WindowAverage { from, until }
            }
            Self::WindowChange { before, after } => {
                facade::JointPanelResponseFunctional::WindowChange { before, after }
            }
            Self::WeightedSum { coefficients } => {
                facade::JointPanelResponseFunctional::WeightedSum { coefficients }
            }
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    names: Vec<String>,
    channels: Vec<String>,
    outcome: String,
    confounder: String,
    season: String,
    seasons: usize,
    lag_order: usize,
    interval_ns: u64,
    groups: Vec<SourceGroup>,
    worlds: Vec<World>,
    interactions: Vec<Feature>,
    options: PanelFitOptions,
    temporal: Option<engine::time_varying::TemporalPanelSpec>,
    functional: Option<Functional>,
    assignment: Option<antecedent_io_core::query_wire::RandomizedEffectQueryWire>,
    #[serde(default)]
    exogenous_context: bool,
}
fn context(
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ExecutionContext> {
    if threads == 0 {
        return Err(crate::value_err("threads must be positive"));
    }
    let mut ctx = crate::py_execution_context_cancel(seed, threads, cancel);
    ctx.set_memory_limit(max_bytes);
    Ok(ctx)
}
fn panel_error(error: facade::JointPanelError) -> PyErr {
    match error {
        facade::JointPanelError::Engine(engine::PanelError::Cancelled) => {
            crate::CausalCancelledError::new_err(error.to_string())
        }
        facade::JointPanelError::Engine(engine::PanelError::Resource(_)) => {
            crate::CausalResourceError::new_err(error.to_string())
        }
        _ => crate::value_err(error.to_string()),
    }
}
fn variable(names: &[String], name: &str) -> PyResult<VariableId> {
    names
        .iter()
        .position(|candidate| candidate == name)
        .and_then(|index| u32::try_from(index).ok())
        .map(VariableId::from_raw)
        .ok_or_else(|| crate::value_err(format!("unknown source variable {name:?}")))
}
fn worlds(names: &[String], declarations: Vec<World>) -> PyResult<Vec<JointPanelWorld>> {
    declarations
        .into_iter()
        .enumerate()
        .map(|(index, world)| {
            let channels = world
                .channels
                .into_iter()
                .map(|(name, values)| {
                    Ok(PanelChannelProgram {
                        id: variable(names, &name)?.raw().to_string(),
                        values,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let scenarios = world
                .scenarios
                .into_iter()
                .map(|scenario| {
                    let history = scenario
                        .history
                        .into_iter()
                        .map(|(name, values)| {
                            Ok(PanelChannelHistory {
                                id: variable(names, &name)?.raw().to_string(),
                                values,
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PanelScenario {
                        group: scenario.group.to_string(),
                        history,
                        confounder: scenario.confounder,
                        season: scenario.season,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(JointPanelWorld {
                world: ScientificWorldId::from_raw(
                    u32::try_from(index).map_err(|_| crate::value_err("too many worlds"))?,
                ),
                reference: world.reference,
                program: PanelProgram { id: world.id, channels, scenarios },
            })
        })
        .collect()
}
fn functional(declaration: Option<&Functional>) -> JointPanelFunctional {
    declaration.map_or(JointPanelFunctional::Worlds, |value| JointPanelFunctional::Linear {
        terms: value
            .terms
            .iter()
            .map(|(id, coefficient)| (ScientificWorldId::from_raw(*id), *coefficient))
            .collect(),
        reference: ScientificWorldId::from_raw(value.reference),
    })
}
fn prepare(declaration: Declaration, ctx: &ExecutionContext) -> PyResult<PyPreparedJointPanel> {
    let mut unique = std::collections::BTreeSet::new();
    if declaration.names.iter().any(|name| !unique.insert(name)) {
        return Err(crate::value_err("duplicate source column name"));
    }
    let columns = JointPanelColumns {
        channels: declaration
            .channels
            .iter()
            .map(|name| variable(&declaration.names, name))
            .collect::<PyResult<Vec<_>>>()?,
        outcome: variable(&declaration.names, &declaration.outcome)?,
        confounder: variable(&declaration.names, &declaration.confounder)?,
        season: variable(&declaration.names, &declaration.season)?,
    };
    let data = source_data(&declaration.names, declaration.groups, declaration.interval_ns)?;
    let interactions =
        features(&declaration.names, &declaration.outcome, declaration.interactions)?;
    let worlds = worlds(&declaration.names, declaration.worlds)?;
    let selection =
        if matches!(&declaration.options.inference, engine::PanelInference::Bayesian { .. }) {
            InferenceSelection::Bayesian
        } else {
            InferenceSelection::Frequentist
        };
    let mut target = facade::joint_panel_target(
        data.schema(),
        &columns,
        declaration.lag_order,
        &worlds,
        functional(declaration.functional.as_ref()),
        selection,
    )
    .map_err(panel_error)?;
    if declaration.exogenous_context {
        target = facade::declare_joint_panel_exogenous_context(&target);
    }
    let assignment = declaration
        .assignment
        .map(|wire| {
            match antecedent_io_core::query_wire::causal_query_from_wire(
                &antecedent_io_core::query_wire::CausalQueryWire::RandomizedEffect(wire),
            )
            .map_err(crate::intervention_program_api::io_err)?
            {
                antecedent_core::CausalQuery::RandomizedEffect(query) => Ok(query),
                _ => Err(crate::value_err("expected randomized assignment")),
            }
        })
        .transpose()?;
    let request = JointPanelRequest {
        columns,
        interactions,
        seasons: declaration.seasons,
        lag_order: declaration.lag_order,
        options: declaration.options,
        worlds,
        target,
        assignment,
        temporal: declaration.temporal,
    };
    let inner = antecedent::Study::prepare_joint_panel(data, request, ctx).map_err(panel_error)?;
    Ok(PyPreparedJointPanel {
        inner,
        names: declaration.names,
        functional: declaration.functional,
        exogenous_context: declaration.exogenous_context,
    })
}

#[pyclass(name = "_PreparedJointPanel", frozen)]
pub(crate) struct PyPreparedJointPanel {
    inner: PreparedStudy<JointPanelState>,
    names: Vec<String>,
    functional: Option<Functional>,
    exogenous_context: bool,
}
#[pymethods]
impl PyPreparedJointPanel {
    fn inspect(&self) -> String {
        let inspection = self.inner.inspect();
        serde_json::json!({"fits_performed":inspection.fits_performed,"identified":inspection.identification.is_identified_claim(),"source_identity":inspection.source_identity.to_hex(),"attempts":inspection.design_attempts.iter().map(|attempt|serde_json::json!({"id":attempt.id,"outcome":format!("{:?}",attempt.outcome),"detail":attempt.detail,"next_attempt":attempt.next_attempt})).collect::<Vec<_>>()}).to_string()
    }
    #[pyo3(signature=(*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn estimate(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyJointPanelResult> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let attempt = crate::detach_catch(py, || {
            Ok(antecedent_core::execution_attempt::observe_with_context(&ctx, || {
                self.inner.estimate(&ctx).map_err(panel_error)
            }))
        })?;
        let (inner, facts) =
            crate::inspect_receipt_api::finish_observed(py, attempt.result, attempt.facts)?;
        Ok(PyJointPanelResult { inner, facts })
    }
    #[pyo3(signature=(worlds_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn retarget(
        &self,
        py: Python<'_>,
        worlds_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        crate::bridge::check_declared_size(
            worlds_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel worlds",
        )?;
        let declarations = serde_json::from_str(worlds_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let worlds = worlds(&self.names, declarations)?;
        let request = self.inner.request();
        let selection =
            if matches!(&request.options.inference, engine::PanelInference::Bayesian { .. }) {
                InferenceSelection::Bayesian
            } else {
                InferenceSelection::Frequentist
            };
        let mut target = facade::joint_panel_target(
            self.inner.source_data().schema(),
            &request.columns,
            request.lag_order,
            &worlds,
            functional(self.functional.as_ref()),
            selection,
        )
        .map_err(panel_error)?;
        if self.exogenous_context {
            target = facade::declare_joint_panel_exogenous_context(&target);
        }
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner.retarget(worlds, target, &ctx).map_err(panel_error)
        })?;
        Ok(Self {
            inner,
            names: self.names.clone(),
            functional: self.functional.clone(),
            exogenous_context: self.exogenous_context,
        })
    }
}
#[pyclass(name = "_JointPanelResult", frozen)]
pub(crate) struct PyJointPanelResult {
    inner: facade::JointPanelResult,
    facts: Py<crate::inspect_receipt_api::PyExecutionFacts>,
}
#[pymethods]
impl PyJointPanelResult {
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn certify_program(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyPanelProgramCertificate> {
        use antecedent_design::decision_contract::{
            ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
        };
        use antecedent_design::program_search::{
            FiniteProgramDomain, ProgramControl, ProgramRestriction,
            continuous::ContinuousSearchOptions,
        };
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel program certificate",
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
        let declaration: CertificateDeclaration = serde_json::from_str(declaration_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let controls = declaration
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
            base_world: ScientificWorldId::from_raw(declaration.base_world),
            scope: facade::JointPanelScientificScope::ModelConditional,
            functional: declaration.functional.native(),
            uncertainty: facade::JointPanelSearchUncertainty::PointMeans,
            contract: ProgramSearchContract {
                space: ProgramSearchSpace {
                    id: "compiled_panel_response".into(),
                    variables: vec!["response".into()],
                },
                objective: ProgramObjective::ExpectedUtility(UtilityExpr::sum(
                    UtilityExpr::Const(declaration.utility_offset),
                    UtilityExpr::product(
                        UtilityExpr::Const(declaration.utility_scale),
                        UtilityExpr::Input(0),
                    ),
                )),
                constraints: vec![],
            },
            domain: FiniteProgramDomain {
                controls,
                budget: declaration.budget,
                restrictions: declaration
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
            initial: declaration.initial,
            options: ContinuousSearchOptions {
                max_evaluations: 1,
                max_iterations: 1,
                initial_step: 0.5,
                step_tolerance: 1e-6,
                objective_tolerance: 0.0,
            },
        };
        crate::detach_catch(py, || {
            let plan = self.inner.compile_certified_search(request, &ctx).map_err(panel_error)?;
            let inner = plan.execute(declaration.options, &ctx).map_err(panel_error)?;
            Ok(PyPanelProgramCertificate { plan, inner, options: declaration.options })
        })
    }
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn decompose_reference(
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
            "panel reference decomposition",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let declaration: DecompositionDeclaration = serde_json::from_str(declaration_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let endpoint = |id, weights| {
            self.inner
                .request()
                .worlds
                .iter()
                .find(|world| world.world.raw() == id)
                .map(|world| facade::DecompositionEndpoint {
                    program: world.program.clone(),
                    population_weights: weights,
                })
                .ok_or_else(|| crate::value_err("decomposition names an absent compiled world"))
        };
        let request = facade::PanelDecompositionRequest {
            before: endpoint(declaration.before_world, declaration.before_population_weights)?,
            after: endpoint(declaration.after_world, declaration.after_population_weights)?,
            time_weights: declaration.time_weights,
            order: declaration.order,
        };
        crate::detach_catch(py, || {
            let checked = self
                .inner
                .decompose_reference_checked(&request, &declaration.periods, &ctx)
                .map_err(panel_error)?;
            if declaration.require_identified_reference {
                checked.require_identified_reference().map_err(panel_error)?;
            }
            let result = &checked.decomposition;
            let joint_uncertainty = match result.joint_uncertainty() {
                Some(summary) => serde_json::json!({"available":true,"summary":summary}),
                None => {
                    serde_json::json!({"available":false,"reason":"fewer than two complete retained law slots"})
                }
            };
            Ok(serde_json::json!({"point":result.point,"draws":result.draws,"weights":result.weights,
                "joint_uncertainty":joint_uncertainty,
                "identified_reference":checked.is_identified_reference(),
                "identification_identity":checked.identification_identity.to_hex(),
                "hybrid_support":checked.hybrids.iter().map(|h|serde_json::json!({"mask":h.mask,
                    "source_world":h.source_world.map(|w|w.raw()),"unresolved":h.unresolved})).collect::<Vec<_>>(),
                "scope":"model_conditional_reference_decomposition",
                "calibration":"unmeasured", "grants_unique_causal_attribution":false})
            .to_string())
        })
    }
    #[getter]
    fn execution_facts(&self, py: Python<'_>) -> Py<crate::inspect_receipt_api::PyExecutionFacts> {
        self.facts.clone_ref(py)
    }
    fn report(&self) -> PyResult<String> {
        let diagnostics=match &self.inner.fit {JointPanelFit::Invariant(fit)=>serde_json::to_value(&fit.diagnostics),JointPanelFit::Varying(fit)=>Ok(serde_json::json!({"attempts":fit.attempts,"selection_counts":fit.selection_counts,"posterior_diagnostics":fit.posterior_diagnostics,"posterior_converged":fit.posterior_converged,"uncertainty":fit.uncertainty,"claim":fit.claim,"calibration":fit.calibration}))}.map_err(|error|crate::serialization_err(error.to_string()))?;
        let point_parameters = match &self.inner.fit {
            JointPanelFit::Invariant(fit) => serde_json::to_value(&fit.point),
            JointPanelFit::Varying(fit) => serde_json::to_value(&fit.point),
        }
        .map_err(|error| crate::serialization_err(error.to_string()))?;
        Ok(serde_json::json!({"point_parameters":point_parameters,"point":self.inner.point,"draws":self.inner.draws,"identified":self.inner.identification().is_identified_claim(),"diagnostics":diagnostics,"fits_performed":self.inner.prepared().fits_performed()}).to_string())
    }
    /// The target's one assembled contrast of scheduled programs, evaluated on every shared
    /// draw so the cross-world covariance is carried by the draws.
    #[pyo3(signature=(functional_json,level))]
    fn scheduled_contrast(&self, functional_json: &str, level: f64) -> PyResult<String> {
        crate::bridge::check_declared_size(
            functional_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel contrast functional",
        )?;
        let functional: ContrastFunctional = serde_json::from_str(functional_json)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let contrast =
            self.inner.scheduled_contrast(&functional.native(), level).map_err(panel_error)?;
        Ok(serde_json::json!({
            "point": contrast.point,
            "draws": contrast.draws,
            "standard_error": contrast.standard_error,
            "interval": contrast.interval,
            "level": contrast.level,
            "uncertainty": format!("{:?}", contrast.uncertainty),
        })
        .to_string())
    }
    #[pyo3(signature=(*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let bytes = crate::detach_catch(py, || self.inner.export(&ctx).map_err(artifact_error))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

#[pyclass(name = "_PanelProgramCertificate", frozen)]
pub(crate) struct PyPanelProgramCertificate {
    plan: facade::JointPanelCertifiedSearch,
    inner: facade::JointPanelCertifiedSearchResult,
    options: antecedent_design::program_search::certified::CertificateOptions,
}
#[pymethods]
impl PyPanelProgramCertificate {
    #[pyo3(signature=(*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let bytes =
            crate::detach_catch(py, || self.plan.export(self.options, &ctx).map_err(panel_error))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(bytes,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn verify_current(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        crate::detach_catch(py, || {
            self.plan.consume_current(bytes, &ctx).map(|_| ()).map_err(panel_error)
        })
    }
    fn report(&self) -> String {
        let certificate = &self.inner.certificate;
        serde_json::json!({"controls":certificate.controls,"incumbent":certificate.incumbent,
            "domain_identity":certificate.identity.to_hex(),
            "global_certificate":matches!(certificate.stop, antecedent_design::program_search::certified::CertificateStop::Certified),
            "assumptions":["retained fitted point response","declared bounded controls and hard linear restrictions","outward-rounded supported arithmetic"],
            "global_upper":certificate.global_upper,"gap":certificate.gap,"stop":certificate.stop,
            "splits":certificate.splits,"infeasible_regions":certificate.infeasible_regions,
            "dominated_regions":certificate.dominated_regions,"monotone_reductions":certificate.monotone_reductions,
            "dual_bound":certificate.dual_bound,"separable_bound":certificate.separable_bound,"frontier":certificate.frontier,
            "fitted_dependency":self.inner.fitted_dependency.to_hex(),
            "parent_target":self.inner.parent_target.to_hex(),
            "scope":"model_conditional","selected_value_reliability":"unmeasured"})
        .to_string()
    }
}
#[pyfunction]
#[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn prepare_joint_panel(
    py: Python<'_>,
    declaration_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedJointPanel> {
    crate::bridge::check_declared_size(
        declaration_json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "joint panel",
    )?;
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let _input = ctx
        .acquire_admission(
            0,
            u64::try_from(declaration_json.len()).unwrap_or(u64::MAX).saturating_mul(4),
        )
        .map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                crate::CausalCancelledError::new_err(error.to_string())
            }
            _ => crate::CausalResourceError::new_err(error.to_string()),
        })?;
    let declaration = serde_json::from_str(declaration_json)
        .map_err(|error| crate::value_err(error.to_string()))?;
    crate::detach_catch(py, || prepare(declaration, &ctx))
}
#[pyfunction]
#[pyo3(signature=(bytes,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn consume_joint_panel(
    py: Python<'_>,
    bytes: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        let replay = antecedent_io::joint_panel_artifact::consume_panel_execution(bytes, &ctx)
            .map_err(artifact_error)?;
        Ok(serde_json::json!({"point":replay.point(),"draws":replay.draws(),"grants_current_authority":replay.grants_current_authority()}).to_string())
    })
}
#[pyfunction]
#[pyo3(signature=(bytes,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn consume_panel_program_certificate(
    py: Python<'_>,
    bytes: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        let (panel, certificate) =
            facade::replay_panel_certificate(bytes, &ctx).map_err(panel_error)?;
        Ok(serde_json::json!({"panel_identity":panel.to_hex(),
            "domain_identity":certificate.identity.to_hex(),"controls":certificate.controls,
            "incumbent":certificate.incumbent,"global_upper":certificate.global_upper,
            "gap":certificate.gap,"stop":certificate.stop,"splits":certificate.splits,
            "infeasible_regions":certificate.infeasible_regions,"dominated_regions":certificate.dominated_regions,
            "frontier":certificate.frontier,"monotone_reductions":certificate.monotone_reductions,
            "dual_bound":certificate.dual_bound,"separable_bound":certificate.separable_bound,
            "grants_current_authority":false}).to_string())
    })
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    selection::register(module)?;
    experiment::register(module)?;
    designs::register(module)?;
    module.add_class::<PyPanelProgramCertificate>()?;
    module.add_class::<PyPreparedJointPanel>()?;
    module.add_class::<PyJointPanelResult>()?;
    module.add_function(wrap_pyfunction!(prepare_joint_panel, module)?)?;
    module.add_function(wrap_pyfunction!(consume_joint_panel, module)?)?;
    module.add_function(wrap_pyfunction!(consume_panel_program_certificate, module)?)?;
    Ok(())
}

fn source_data(
    names: &[String],
    groups: Vec<SourceGroup>,
    interval_ns: u64,
) -> PyResult<PanelData> {
    let units = groups
        .into_iter()
        .map(|group| {
            if group.columns.len() != names.len() {
                return Err(crate::value_err(
                    "source columns must match the declared source schema",
                ));
            }
            let values = group
                .columns
                .into_iter()
                .map(|column| {
                    column.into_iter().map(|value| value.unwrap_or(f64::NAN)).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let series = TimeSeriesData::from_f64_columns(
                names.iter().map(String::as_str).zip(values.iter().map(Vec::as_slice)),
                interval_ns,
            )
            .map_err(crate::py_err)?;
            Ok(PanelUnit { unit_id: group.id, series })
        })
        .collect::<PyResult<Vec<_>>>()?;
    PanelData::try_new(Arc::from(units)).map_err(crate::py_err)
}
fn features(
    names: &[String],
    outcome: &str,
    features: Vec<Feature>,
) -> PyResult<Vec<engine::PanelInteractionSpec>> {
    let interactions = features
        .into_iter()
        .map(|feature| {
            // Reuse the existing named Expr decoder; this parses syntax, never performs identification.
            let text =
                serde_json::json!([{"mode":"assign","variable":outcome,"expr":feature.expression}])
                    .to_string();
            let parsed = antecedent_io::policy_program_artifact::interventions_from_named_json(
                &text,
                &|name| variable(names, name).ok().map(VariableId::raw),
            )
            .map_err(crate::intervention_program_api::io_err)?;
            let Some(antecedent_core::Intervention::Assign { expr: expression, .. }) =
                parsed.into_iter().next()
            else {
                return Err(crate::value_err("missing programmable feature expression"));
            };
            engine::PanelInteractionSpec::from_saturated_features(feature.id, &expression)
                .map_err(|error| crate::value_err(error.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(interactions)
}

fn artifact_error(error: antecedent_io::joint_panel_artifact::PanelArtifactError) -> PyErr {
    use antecedent_io::joint_panel_artifact::PanelArtifactError;
    match error {
        PanelArtifactError::Cancelled
        | PanelArtifactError::Engine(engine::PanelError::Cancelled) => {
            crate::CausalCancelledError::new_err(error.to_string())
        }
        PanelArtifactError::Resource(_)
        | PanelArtifactError::Engine(engine::PanelError::Resource(_)) => {
            crate::CausalResourceError::new_err(error.to_string())
        }
        _ => crate::serialization_err(error.to_string()),
    }
}

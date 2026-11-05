// SPDX-License-Identifier: MIT OR Apache-2.0
//! Python declarations delegate to the public retained joint-panel compiler.
use antecedent::analysis::joint_panel::{
    JointPanelColumns, JointPanelFit, JointPanelFunctional, JointPanelRequest, JointPanelState,
    JointPanelWorld,
};
use antecedent::analysis::{PreparedStudy, joint_panel as facade};
use antecedent::data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
    OwnedColumnarStorage, SamplingRegularity, TableView, TimeIndex, UnknownCategoryPolicy,
    ValidityBitmap,
};
use antecedent::estimate::joint_panel as engine;
use antecedent::estimate::joint_panel::{
    PanelChannelHistory, PanelChannelProgram, PanelFitOptions, PanelProgram, PanelScenario,
};
use antecedent::prelude::{
    CausalSchema, CausalSchemaBuilder, ExecutionContext, PanelData, PanelUnit, TimeSeriesData,
    VariableId,
};
use antecedent::schema::{CategoryDomainId, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
use antecedent::scientific::{InferenceSelection, ScientificTarget, ScientificWorldId};
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::Arc};
mod attempts;
mod contrast_target;
mod designs;
mod experiment;
mod finite_search;
mod imputed;
mod replay;
mod search;
mod selection;
mod unrolled;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceGroup {
    id: u32,
    columns: Vec<SourceColumn>,
}
/// One source column: numbers (missing cells `null`), or category labels for a declared
/// categorical channel (missing cells `null`).
#[derive(Deserialize)]
#[serde(untagged)]
enum SourceColumn {
    Numeric(Vec<Option<f64>>),
    Labels(Vec<Option<String>>),
}
/// A categorical assignment channel and its complete declared level set.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CategoricalDeclaration {
    channel: String,
    levels: Vec<String>,
    reference: String,
}
/// An indicator channel the dummy coding produced, by name.
#[derive(Clone)]
struct Indicator {
    name: String,
    source: String,
    level: String,
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
    options: antecedent::design::program_search::certified::CertificateOptions,
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
    #[serde(default)]
    family: engine::PanelFamily,
    temporal: Option<engine::time_varying::TemporalPanelSpec>,
    functional: Option<Functional>,
    assignment: Option<antecedent::io::query_wire::RandomizedEffectQueryWire>,
    #[serde(default)]
    exogenous_context: bool,
    #[serde(default)]
    gaussian_restriction: bool,
    #[serde(default)]
    categorical: Vec<CategoricalDeclaration>,
}
pub(crate) fn context(
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
/// A declaration lowered to the Rust request: the source panel (dummy-coded when categorical
/// channels were declared), its roles and the names the worlds and features refer to.
struct Compiled {
    data: PanelData,
    request: JointPanelRequest,
    names: Vec<String>,
    functional: Option<Functional>,
    exogenous_context: bool,
    gaussian_restriction: bool,
    indicators: Vec<Indicator>,
}
fn selection_of(options: &PanelFitOptions) -> InferenceSelection {
    if matches!(&options.inference, engine::PanelInference::Bayesian { .. }) {
        InferenceSelection::Bayesian
    } else {
        InferenceSelection::Frequentist
    }
}
/// The target choices a handle retains so a declared context or restriction survives
/// replacing the worlds.
struct TargetChoices<'a> {
    functional: Option<&'a Functional>,
    selection: InferenceSelection,
    exogenous_context: bool,
    gaussian_restriction: bool,
}
/// The one assembly of the scientific target, shared by preparation and retargeting.
fn assemble_target(
    schema: &CausalSchema,
    columns: &JointPanelColumns,
    lag_order: usize,
    worlds: &[JointPanelWorld],
    choices: &TargetChoices<'_>,
) -> PyResult<ScientificTarget> {
    let mut target = facade::joint_panel_target(
        schema,
        columns,
        lag_order,
        worlds,
        functional(choices.functional),
        choices.selection,
    )
    .map_err(panel_error)?;
    if choices.exogenous_context {
        target = facade::declare_joint_panel_exogenous_context(&target);
    }
    if choices.gaussian_restriction {
        target = facade::declare_joint_panel_gaussian_restriction(&target);
    }
    Ok(target)
}
fn compile(declaration: Declaration) -> PyResult<Compiled> {
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
    let data = source_data(
        &declaration.names,
        declaration.groups,
        declaration.interval_ns,
        &declaration.categorical,
    )?;
    let (data, columns, names, indicators) = if declaration.categorical.is_empty() {
        (data, columns, declaration.names, Vec::new())
    } else {
        code_categorical(&data, &columns, &declaration.names, &declaration.categorical)?
    };
    let interactions = features(&names, &declaration.outcome, declaration.interactions)?;
    let worlds = worlds(&names, declaration.worlds)?;
    let target = assemble_target(
        data.schema(),
        &columns,
        declaration.lag_order,
        &worlds,
        &TargetChoices {
            functional: declaration.functional.as_ref(),
            selection: selection_of(&declaration.options),
            exogenous_context: declaration.exogenous_context,
            gaussian_restriction: declaration.gaussian_restriction,
        },
    )?;
    let assignment = declaration
        .assignment
        .map(|wire| {
            match antecedent::io::query_wire::causal_query_from_wire(
                &antecedent::io::query_wire::CausalQueryWire::RandomizedEffect(wire),
            )
            .map_err(crate::intervention_program_api::io_err)?
            {
                antecedent::query::CausalQuery::RandomizedEffect(query) => Ok(query),
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
        family: declaration.family,
        worlds,
        target,
        assignment,
        temporal: declaration.temporal,
    };
    Ok(Compiled {
        data,
        request,
        names,
        functional: declaration.functional,
        exogenous_context: declaration.exogenous_context,
        gaussian_restriction: declaration.gaussian_restriction,
        indicators,
    })
}
fn prepare(declaration: Declaration, ctx: &ExecutionContext) -> PyResult<PyPreparedJointPanel> {
    let compiled = compile(declaration)?;
    let inner = antecedent::Study::prepare_joint_panel(compiled.data, compiled.request, ctx)
        .map_err(panel_error)?;
    Ok(PyPreparedJointPanel {
        inner,
        names: compiled.names,
        functional: compiled.functional,
        exogenous_context: compiled.exogenous_context,
        gaussian_restriction: compiled.gaussian_restriction,
        indicators: compiled.indicators,
    })
}

#[pyclass(name = "_PreparedJointPanel", frozen)]
pub(crate) struct PyPreparedJointPanel {
    inner: PreparedStudy<JointPanelState>,
    names: Vec<String>,
    functional: Option<Functional>,
    exogenous_context: bool,
    gaussian_restriction: bool,
    indicators: Vec<Indicator>,
}
#[pymethods]
impl PyPreparedJointPanel {
    fn inspect(&self) -> String {
        let inspection = self.inner.inspect();
        serde_json::json!({"fits_performed":inspection.fits_performed,"identified":inspection.identification.is_identified_claim(),"source_identity":inspection.source_identity.to_hex(),"attempts":inspection.design_attempts.iter().map(|attempt|serde_json::json!({"id":attempt.id,"outcome":format!("{:?}",attempt.outcome),"detail":attempt.detail,"next_attempt":attempt.next_attempt})).collect::<Vec<_>>(),"indicators":self.indicators.iter().map(|indicator|serde_json::json!({"name":indicator.name,"source":indicator.source,"level":indicator.level})).collect::<Vec<_>>(),"gaussian_restriction_declared":self.gaussian_restriction}).to_string()
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
            Ok(antecedent::execution::observe(&ctx, || {
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
        let target = assemble_target(
            self.inner.source_data().schema(),
            &request.columns,
            request.lag_order,
            &worlds,
            &self.choices(selection_of(&request.options)),
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner.retarget(worlds, target, &ctx).map_err(panel_error)
        })?;
        Ok(self.rebound(inner))
    }
    /// Run general ID on a declared unrolled point graph of this panel's program.
    ///
    /// With the Gaussian restriction declared on the target, an identified verdict is retained
    /// as an identification of the fit under that restriction; otherwise it is recorded as a
    /// graph-level statement only.
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn with_unrolled_graph(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "unrolled panel graph",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let (graph, outcome_periods, _lease) = unrolled::graph_from_json(declaration_json, &ctx)?;
        let inner = crate::detach_catch(py, || {
            self.inner
                .with_unrolled_graph_attempt(&graph, &outcome_periods, &ctx)
                .map_err(panel_error)
        })?;
        Ok(self.rebound(inner))
    }
}
impl PyPreparedJointPanel {
    fn choices(&self, selection: InferenceSelection) -> TargetChoices<'_> {
        TargetChoices {
            functional: self.functional.as_ref(),
            selection,
            exogenous_context: self.exogenous_context,
            gaussian_restriction: self.gaussian_restriction,
        }
    }
    /// The same declarations around a derived prepared study.
    fn rebound(&self, inner: PreparedStudy<JointPanelState>) -> Self {
        Self {
            inner,
            names: self.names.clone(),
            functional: self.functional.clone(),
            exogenous_context: self.exogenous_context,
            gaussian_restriction: self.gaussian_restriction,
            indicators: self.indicators.clone(),
        }
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
        use antecedent::design::decision_contract::{
            ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
        };
        use antecedent::design::program_search::continuous::ContinuousSearchOptions;
        use antecedent::design::program_search::{
            FiniteProgramDomain, ProgramControl, ProgramRestriction,
        };
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "panel program certificate",
        )?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let _input =
            antecedent::execution::retain(&ctx, (declaration_json.len() as u64).saturating_mul(8))
                .map_err(|e| match e {
                    antecedent::execution::AdmissionRefusal::Cancelled => {
                        crate::CausalCancelledError::new_err(e.to_string())
                    }
                    antecedent::execution::AdmissionRefusal::Refused(_) => {
                        crate::CausalResourceError::new_err(e.to_string())
                    }
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
                    "source_world":h.source_world.map(antecedent::scientific::ScientificWorldId::raw),"unresolved":h.unresolved})).collect::<Vec<_>>(),
                "scope":"model_conditional_reference_decomposition",
                "calibration":"not_measured", "grants_unique_causal_attribution":false})
            .to_string())
        })
    }
    #[getter]
    fn execution_facts(&self, py: Python<'_>) -> Py<crate::inspect_receipt_api::PyExecutionFacts> {
        self.facts.clone_ref(py)
    }
    fn report(&self, py: Python<'_>) -> PyResult<String> {
        crate::detach_catch(py, || {
            let diagnostics=match &self.inner.fit {JointPanelFit::Invariant(fit)=>serde_json::to_value(&fit.diagnostics),JointPanelFit::Varying(fit)=>Ok(serde_json::json!({"attempts":fit.attempts,"selection_counts":fit.selection_counts,"posterior_diagnostics":fit.posterior_diagnostics,"posterior_converged":fit.posterior_converged,"uncertainty":fit.uncertainty,"claim":fit.claim,"calibration":fit.calibration}))}.map_err(|error|crate::serialization_err(error.to_string()))?;
            let point_parameters = match &self.inner.fit {
                JointPanelFit::Invariant(fit) => serde_json::to_value(&fit.point),
                JointPanelFit::Varying(fit) => serde_json::to_value(&fit.point),
            }
            .map_err(|error| crate::serialization_err(error.to_string()))?;
            Ok(serde_json::json!({"point_parameters":point_parameters,"point":self.inner.point,"draws":self.inner.draws,"identified":self.inner.identification().is_identified_claim(),"diagnostics":diagnostics,"dependency":self.inner.dependency_identity().to_hex(),"fits_performed":self.inner.prepared().fits_performed()}).to_string())
        })
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
    options: antecedent::design::program_search::certified::CertificateOptions,
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
            "global_certificate":matches!(certificate.stop, antecedent::design::program_search::certified::CertificateStop::Certified),
            "assumptions":["retained fitted point response","declared bounded controls and hard linear restrictions","outward-rounded supported arithmetic"],
            "global_upper":certificate.global_upper,"gap":certificate.gap,"stop":certificate.stop,
            "splits":certificate.splits,"infeasible_regions":certificate.infeasible_regions,
            "dominated_regions":certificate.dominated_regions,"monotone_reductions":certificate.monotone_reductions,
            "dual_bound":certificate.dual_bound,"separable_bound":certificate.separable_bound,"frontier":certificate.frontier,
            "fitted_dependency":self.inner.fitted_dependency.to_hex(),
            "parent_target":self.inner.parent_target.to_hex(),
            "scope":"model_conditional","selected_value_reliability":"not_measured"})
        .to_string()
    }
}
/// Charge the declaration's own text to the caller's admission before it is parsed.
fn admit_declaration(
    ctx: &ExecutionContext,
    declaration_json: &str,
) -> PyResult<antecedent::execution::Retention> {
    crate::bridge::check_declared_size(
        declaration_json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "joint panel",
    )?;
    admit_bytes(ctx, u64::try_from(declaration_json.len()).unwrap_or(u64::MAX).saturating_mul(4))
}
/// Charge `bytes` to the caller's admission: a cancelled run or an unfunded request is an
/// operational stop (`CausalCancelledError`, `CausalResourceError`), never a request error.
fn admit_bytes(ctx: &ExecutionContext, bytes: u64) -> PyResult<antecedent::execution::Retention> {
    antecedent::execution::retain(ctx, bytes).map_err(|error| match error {
        antecedent::execution::AdmissionRefusal::Cancelled => {
            crate::CausalCancelledError::new_err(error.to_string())
        }
        antecedent::execution::AdmissionRefusal::Refused(_) => {
            crate::CausalResourceError::new_err(error.to_string())
        }
    })
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
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let _input = admit_declaration(&ctx, declaration_json)?;
    let declaration = serde_json::from_str(declaration_json)
        .map_err(|error| crate::value_err(error.to_string()))?;
    crate::detach_catch(py, || prepare(declaration, &ctx))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CodingDeclaration {
    names: Vec<String>,
    channels: Vec<String>,
    outcome: String,
    confounder: String,
    season: String,
    interval_ns: u64,
    groups: Vec<SourceGroup>,
    categorical: Vec<CategoricalDeclaration>,
}
/// Dummy-code declared categorical channels of a source panel, without fitting anything.
///
/// Returns the rewritten panel as named columns (a missing cell is `null`), the channel names
/// in order, and every indicator with its source and level.
#[pyfunction]
#[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn dummy_code_joint_panel(
    py: Python<'_>,
    declaration_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let _input = admit_declaration(&ctx, declaration_json)?;
    let declaration: CodingDeclaration = serde_json::from_str(declaration_json)
        .map_err(|error| crate::value_err(error.to_string()))?;
    // The coding rewrites the panel: the source copy, the coded copy with one indicator per
    // non-reference level, and the returned document, charged before any of them exists.
    let rows: usize = declaration
        .groups
        .iter()
        .map(|group| {
            group.columns.first().map_or(0, |column| match column {
                SourceColumn::Numeric(values) => values.len(),
                SourceColumn::Labels(labels) => labels.len(),
            })
        })
        .sum();
    let width = declaration
        .names
        .len()
        .saturating_add(declaration.categorical.iter().map(|c| c.levels.len()).sum::<usize>());
    let _coding = admit_bytes(
        &ctx,
        u64::try_from(rows.saturating_mul(width).saturating_mul(64)).unwrap_or(u64::MAX),
    )?;
    crate::detach_catch(py, || {
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
        let data = source_data(
            &declaration.names,
            declaration.groups,
            declaration.interval_ns,
            &declaration.categorical,
        )?;
        let (coded, columns, names, indicators) =
            code_categorical(&data, &columns, &declaration.names, &declaration.categorical)?;
        let name = |id: VariableId| names.get(id.raw() as usize).cloned().unwrap_or_default();
        let groups = coded
            .units()
            .iter()
            .map(|unit| {
                let cells = (0..names.len())
                    .map(|index| {
                        let id = VariableId::from_raw(u32::try_from(index).unwrap_or(u32::MAX));
                        unit.series.float64_cow(id).map_err(crate::py_err).map(|values| {
                            values
                                .iter()
                                .map(|value| (!value.is_nan()).then_some(*value))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(serde_json::json!({"id": unit.unit_id, "columns": cells}))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(serde_json::json!({
            "names": names,
            "channels": columns.channels.iter().map(|id| name(*id)).collect::<Vec<_>>(),
            "outcome": name(columns.outcome),
            "confounder": name(columns.confounder),
            "season": name(columns.season),
            "groups": groups,
            "indicators": indicators.iter().map(|indicator| {
                serde_json::json!({"name": indicator.name, "source": indicator.source,
                    "level": indicator.level})
            }).collect::<Vec<_>>(),
        })
        .to_string())
    })
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
        let replay = antecedent::io::joint_panel_artifact::consume_panel_execution(bytes, &ctx)
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
    replay::register(module)?;
    contrast_target::register(module)?;
    designs::register(module)?;
    module.add_class::<PyPanelProgramCertificate>()?;
    module.add_class::<PyPreparedJointPanel>()?;
    module.add_class::<PyJointPanelResult>()?;
    module.add_function(wrap_pyfunction!(prepare_joint_panel, module)?)?;
    module.add_function(wrap_pyfunction!(dummy_code_joint_panel, module)?)?;
    imputed::register(module)?;
    module.add_function(wrap_pyfunction!(consume_joint_panel, module)?)?;
    module.add_function(wrap_pyfunction!(consume_panel_program_certificate, module)?)?;
    Ok(())
}

fn numeric_cells(column: SourceColumn, name: &str) -> PyResult<Vec<f64>> {
    match column {
        SourceColumn::Numeric(cells) => {
            Ok(cells.into_iter().map(|value| value.unwrap_or(f64::NAN)).collect())
        }
        SourceColumn::Labels(_) => Err(crate::value_err(format!(
            "column {name:?} holds category labels but is not a declared categorical channel"
        ))),
    }
}
/// Labels of a declared categorical channel. A column with no value at all is every cell
/// missing; numbers are refused rather than read as labels.
fn label_cells(column: &SourceColumn, name: &str) -> PyResult<Vec<Option<String>>> {
    match column {
        SourceColumn::Labels(cells) => Ok(cells.clone()),
        SourceColumn::Numeric(cells) if cells.iter().all(Option::is_none) => {
            Ok(vec![None; cells.len()])
        }
        SourceColumn::Numeric(_) => Err(crate::value_err(format!(
            "categorical channel {name:?} must supply category labels, not numbers"
        ))),
    }
}
fn validity_of(present: impl Iterator<Item = bool>, len: usize) -> PyResult<ValidityBitmap> {
    let mut bytes = vec![0_u8; len.div_ceil(8)];
    for (row, present) in present.enumerate() {
        if present {
            bytes[row / 8] |= 1 << (row % 8);
        }
    }
    ValidityBitmap::from_bytes(bytes, len).map_err(crate::py_err)
}
fn source_data(
    names: &[String],
    groups: Vec<SourceGroup>,
    interval_ns: u64,
    categorical: &[CategoricalDeclaration],
) -> PyResult<PanelData> {
    for group in &groups {
        if group.columns.len() != names.len() {
            return Err(crate::value_err("source columns must match the declared source schema"));
        }
    }
    for declared in categorical {
        variable(names, &declared.channel)?;
    }
    // Every categorical column shares one domain: the labels observed in any unit, sorted. The
    // declared level set is then checked against it by the Rust dummy coding, so a declared
    // level nobody assigns and an assigned level nobody declared are both refused by name.
    let mut domains: Vec<Option<std::sync::Arc<CategoryDomain>>> = vec![None; names.len()];
    for (slot, name) in names.iter().enumerate() {
        if !categorical.iter().any(|declared| &declared.channel == name) {
            continue;
        }
        let mut observed = std::collections::BTreeSet::new();
        for group in &groups {
            observed.extend(label_cells(&group.columns[slot], name)?.into_iter().flatten());
        }
        let levels: Vec<CategoryLevel> = observed
            .into_iter()
            .map(|label| CategoryLevel { label: std::sync::Arc::from(label) })
            .collect();
        if levels.is_empty() {
            return Err(crate::value_err(format!(
                "categorical channel {name:?} has no observed level"
            )));
        }
        let id = CategoryDomainId::from_raw(u32::try_from(slot + 1).unwrap_or(u32::MAX));
        domains[slot] = Some(std::sync::Arc::new(
            CategoryDomain::try_new(id, levels, false, None, UnknownCategoryPolicy::Fail)
                .map_err(crate::py_err)?,
        ));
    }
    let units = groups
        .into_iter()
        .map(|group| {
            let series = if categorical.is_empty() {
                let values = group
                    .columns
                    .into_iter()
                    .zip(names)
                    .map(|(column, name)| numeric_cells(column, name))
                    .collect::<PyResult<Vec<_>>>()?;
                TimeSeriesData::from_f64_columns(
                    names.iter().map(String::as_str).zip(values.iter().map(Vec::as_slice)),
                    interval_ns,
                )
                .map_err(crate::py_err)?
            } else {
                mixed_series(names, group.columns, &domains, interval_ns)?
            };
            Ok(PanelUnit { unit_id: group.id, series })
        })
        .collect::<PyResult<Vec<_>>>()?;
    PanelData::try_new(Arc::from(units)).map_err(crate::py_err)
}
/// One unit's series holding numeric columns as `f64` and declared categorical channels as
/// categorical columns over their shared domain.
fn mixed_series(
    names: &[String],
    columns: Vec<SourceColumn>,
    domains: &[Option<std::sync::Arc<CategoryDomain>>],
    interval_ns: u64,
) -> PyResult<TimeSeriesData> {
    let schema_error = |error: antecedent::schema::SchemaError| crate::value_err(error.to_string());
    let mut builder = CausalSchemaBuilder::new();
    let mut built = Vec::with_capacity(names.len());
    let mut length = 0;
    for (slot, ((name, column), domain)) in names.iter().zip(columns).zip(domains).enumerate() {
        let id = VariableId::from_raw(
            u32::try_from(slot).map_err(|_| crate::value_err("too many source columns"))?,
        );
        let roles = SmallRoleSet::from_hint(RoleHint::Context);
        if let Some(domain) = domain {
            let labels = label_cells(&column, name)?;
            length = labels.len();
            let codes: Vec<CategoryCode> = labels
                .iter()
                .map(|label| {
                    let code = label.as_ref().and_then(|label| {
                        domain.levels.iter().position(|level| &*level.label == label.as_str())
                    });
                    CategoryCode::from_raw(u32::try_from(code.unwrap_or(0)).unwrap_or(0))
                })
                .collect();
            builder
                .add_variable(
                    name.as_str(),
                    ValueType::Categorical,
                    roles,
                    None,
                    Some(domain.id),
                    MeasurementSpec::default(),
                )
                .map_err(schema_error)?;
            built.push(OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    id,
                    codes,
                    validity_of(labels.iter().map(Option::is_some), labels.len())?,
                    domain.clone(),
                )
                .map_err(crate::py_err)?,
            ));
        } else {
            let values = numeric_cells(column, name)?;
            length = values.len();
            builder
                .add_variable(
                    name.as_str(),
                    ValueType::Continuous,
                    roles,
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .map_err(schema_error)?;
            let validity = validity_of(values.iter().map(|value| !value.is_nan()), values.len())?;
            built.push(OwnedColumn::Float64(
                Float64Column::new(id, Arc::<[f64]>::from(values), validity)
                    .map_err(crate::py_err)?,
            ));
        }
    }
    let storage =
        OwnedColumnarStorage::try_new(builder.build().map_err(schema_error)?, built, None, None)
            .map_err(crate::py_err)?;
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length },
    )
    .map_err(crate::py_err)
}
/// Replace each declared categorical channel by its indicator channels.
///
/// Returns the rewritten panel, its roles, the names every later declaration refers to
/// (indicators are `"<channel>=<level>"`) and the indicators with their sources.
fn code_categorical(
    data: &PanelData,
    columns: &JointPanelColumns,
    names: &[String],
    categorical: &[CategoricalDeclaration],
) -> PyResult<(PanelData, JointPanelColumns, Vec<String>, Vec<Indicator>)> {
    let declared = categorical
        .iter()
        .map(|declaration| {
            Ok(facade::PanelCategoricalChannel {
                variable: variable(names, &declaration.channel)?,
                levels: declaration.levels.clone(),
                reference: declaration.reference.clone(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let coded = facade::dummy_code_panel(data, columns, &declared).map_err(panel_error)?;
    let coded_names: Vec<String> =
        coded.data.schema().variables().iter().map(|v| v.name.to_string()).collect();
    let indicators = coded
        .indicators
        .iter()
        .map(|indicator| {
            Ok(Indicator {
                name: coded_names
                    .get(indicator.variable.raw() as usize)
                    .cloned()
                    .ok_or_else(|| crate::value_err("indicator outside the coded schema"))?,
                source: names
                    .get(indicator.source.raw() as usize)
                    .cloned()
                    .ok_or_else(|| crate::value_err("indicator source outside the schema"))?,
                level: indicator.level.clone(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((coded.data, coded.columns, coded_names, indicators))
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
            let parsed = antecedent::io::policy_program_artifact::interventions_from_named_json(
                &text,
                &|name| variable(names, name).ok().map(VariableId::raw),
            )
            .map_err(crate::intervention_program_api::io_err)?;
            let Some(antecedent::prelude::Intervention::Assign { expr: expression, .. }) =
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

fn artifact_error(error: antecedent::io::joint_panel_artifact::PanelArtifactError) -> PyErr {
    use antecedent::io::joint_panel_artifact::PanelArtifactError;
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

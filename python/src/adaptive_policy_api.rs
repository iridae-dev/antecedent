//! Python binding for bounded history-dependent policy search.
//!
//! The configuration (families, constraints, search policy) crosses as JSON; the response
//! source is a Python object that simulates decision paths under the shared draws it owns. The
//! native layer holds the search, constraints, standing label, held-out refusal and replayable
//! record; the Python object only supplies paths.
use antecedent::gcm::{ModelPolicySource, ModelRolloutDesign, StageUtility};
use antecedent_core::VariableId;
use antecedent_design::adaptive_policy::{AdaptivePolicyFamily, PolicyFeature, PolicyRule};
use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, ChanceBound, DrawSplit, HeldOutDesign, HeldOutRequirement,
    PolicyConstraints, PolicyObjective, PolicyPath, PolicyRolloutSource, PolicySearchOptions,
    PolicyStanding, PreparedAdaptivePolicySearch, ResponseStanding, SearchStop, StandingRequest,
};
use pyo3::prelude::*;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::PyCancellationToken;

fn bad(message: impl AsRef<str>) -> PyErr {
    crate::value_err(message)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Largest number of simulated rows, steps or cells a rollout may allocate.
const MAX_ROLLOUT_UNITS: u64 = 10_000_000;
/// Largest number of values one rollout may hold across rows and nodes.
const MAX_ROLLOUT_CELLS: u64 = 64_000_000;

fn num(spec: &Value, key: &str) -> PyResult<f64> {
    match spec.get(key) {
        Some(value) => match value.as_f64() {
            Some(number) if number.is_finite() => Ok(number),
            _ => Err(bad(format!("field {key} must be a finite number"))),
        },
        None => Err(bad(format!("field {key} is required"))),
    }
}

/// A number that may be absent; a present value of another type is an error.
fn opt_num(spec: &Value, key: &str) -> PyResult<Option<f64>> {
    match spec.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => num(spec, key).map(Some),
    }
}

/// A count that may be absent; a present value that is not a non-negative integer is an error.
fn opt_count(spec: &Value, key: &str) -> PyResult<Option<u64>> {
    match spec.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| bad(format!("field {key} must be a non-negative integer"))),
    }
}

fn opt_size(spec: &Value, key: &str) -> PyResult<Option<usize>> {
    opt_count(spec, key)?
        .map(|n| usize::try_from(n).map_err(|_| bad(format!("field {key} is too large"))))
        .transpose()
}

fn opt_flag(spec: &Value, key: &str) -> PyResult<Option<bool>> {
    match spec.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(bad(format!("field {key} must be a boolean"))),
    }
}

fn opt_text<'a>(spec: &'a Value, key: &str) -> PyResult<Option<&'a str>> {
    match spec.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.as_str())),
        Some(_) => Err(bad(format!("field {key} must be a string"))),
    }
}

/// A rollout size bounded so a stray magnitude is refused rather than allocated.
fn bounded_size(spec: &Value, key: &str, default: usize) -> PyResult<usize> {
    let value = opt_count(spec, key)?;
    match value {
        None => Ok(default),
        Some(n) if n <= MAX_ROLLOUT_UNITS => Ok(n as usize),
        Some(_) => Err(crate::CausalResourceError::new_err(format!(
            "field {key} exceeds the bound of {MAX_ROLLOUT_UNITS}"
        ))),
    }
}

fn pair(spec: &Value, key: &str) -> PyResult<(f64, f64)> {
    let items = spec.get(key).and_then(Value::as_array);
    match items.map(|a| a.iter().map(Value::as_f64).collect::<Vec<_>>()).as_deref() {
        Some([Some(a), Some(b)]) => Ok((*a, *b)),
        _ => Err(bad(format!("family field {key} must be a [lower, upper] pair"))),
    }
}

fn indices(spec: &Value, key: &str) -> PyResult<Vec<usize>> {
    spec.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| bad(format!("family field {key} must be a list of feature indices")))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| bad(format!("family field {key} holds a non-index")))
        })
        .collect()
}

fn index(spec: &Value, key: &str) -> PyResult<usize> {
    opt_size(spec, key)?.ok_or_else(|| bad(format!("field {key} is required")))
}

fn variable_of(spec: &Value, key: &str, names: &[String]) -> PyResult<VariableId> {
    match spec.get(key) {
        Some(Value::String(name)) => names
            .iter()
            .position(|n| n == name)
            .map(|i| VariableId::from_raw(i as u32))
            .ok_or_else(|| bad(format!("unknown variable {name:?}"))),
        Some(_) => {
            let raw = u32::try_from(index(spec, key)?)
                .map_err(|_| bad(format!("field {key} is not a variable index")))?;
            if !names.is_empty() && raw as usize >= names.len() {
                return Err(bad(format!("field {key} is not a variable index")));
            }
            Ok(VariableId::from_raw(raw))
        }
        None => Err(bad(format!("field {key} is required"))),
    }
}

fn parse_family(spec: &Value, names: &[String]) -> PyResult<AdaptivePolicyFamily> {
    let id = spec.get("id").and_then(Value::as_str).ok_or_else(|| bad("family id is required"))?;
    let kind =
        spec.get("kind").and_then(Value::as_str).ok_or_else(|| bad("family kind is required"))?;
    let mut features = Vec::new();
    let mut ranges = Vec::new();
    for feature in spec.get("features").and_then(Value::as_array).map_or(&[][..], Vec::as_slice) {
        let variable = variable_of(feature, "variable", names)?;
        features.push(match opt_count(feature, "lag")? {
            Some(lag) if lag > 0 => PolicyFeature::Lag {
                variable,
                lag: u32::try_from(lag).map_err(|_| bad("lag too large"))?,
            },
            _ => PolicyFeature::Current(variable),
        });
        ranges.push(pair(feature, "range")?);
    }
    let rule = match kind {
        "schedule" => PolicyRule::Schedule { steps: index(spec, "steps")? },
        "threshold" => PolicyRule::Threshold { feature: index(spec, "feature")? },
        "piecewise" => {
            PolicyRule::Piecewise { feature: index(spec, "feature")?, cuts: index(spec, "cuts")? }
        }
        "linear_feedback" => {
            PolicyRule::LinearFeedback { features: indices(spec, "features_used")? }
        }
        "tree" => PolicyRule::Tree { splits: indices(spec, "splits")? },
        other => {
            return Err(bad(format!(
                "unknown policy family kind {other:?}; use schedule|threshold|piecewise|linear_feedback|tree"
            )));
        }
    };
    let coefficients = if spec.get("coefficient_range").is_some() {
        pair(spec, "coefficient_range")?
    } else {
        (-1.0, 1.0)
    };
    AdaptivePolicyFamily::new(
        id,
        features,
        &ranges,
        rule,
        pair(spec, "action_bounds")?,
        coefficients,
    )
    .map_err(|e| bad(e.to_string()))
}

fn chance(spec: &Value, key: &str) -> PyResult<Option<ChanceBound>> {
    match spec.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => Ok(Some(ChanceBound {
            bound: num(v, "bound")?,
            max_violation_probability: opt_num(v, "max_violation_probability")?.unwrap_or(0.0),
        })),
    }
}

/// The Python response source: it owns the shared draws and returns paths.
///
/// The descriptive members (identity, draw counts, horizon, standing, held-out design and
/// observable variables) are read once when the source is bound, so a method that fails or
/// returns the wrong type is an error naming it rather than a default the search runs on.
struct PySource {
    object: Py<PyAny>,
    identity: String,
    optimization_draws: usize,
    held_out_draws: usize,
    horizon: usize,
    standing: ResponseStanding,
    held_out_design: HeldOutDesign,
    observable: Option<Vec<VariableId>>,
}

impl PySource {
    fn bind(object: Py<PyAny>) -> PyResult<Self> {
        let read = |method: &str, args: Vec<String>| -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                let tuple = pyo3::types::PyTuple::new(py, args)?;
                let value = object.bind(py).call_method1(method, tuple).map_err(|error| {
                    if error.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                        error
                    } else {
                        bad(format!("response source method {method} failed: {error}"))
                    }
                })?;
                Ok(value.unbind())
            })
        };
        let wrong = |method: &str, expected: &str| {
            bad(format!("response source method {method} must return {expected}"))
        };
        let identity = Python::attach(|py| {
            read("identity", Vec::new())?
                .extract::<String>(py)
                .map_err(|_| wrong("identity", "a string"))
        })?;
        let draws = |split: DrawSplit| -> PyResult<usize> {
            Python::attach(|py| {
                read("n_draws", vec![split_name(split).to_owned()])?
                    .extract::<usize>(py)
                    .map_err(|_| wrong("n_draws", "a non-negative integer"))
            })
        };
        let optimization_draws = draws(DrawSplit::Optimization)?;
        let held_out_draws = draws(DrawSplit::HeldOut)?;
        let horizon = Python::attach(|py| {
            read("horizon", Vec::new())?
                .extract::<usize>(py)
                .map_err(|_| wrong("horizon", "a non-negative integer"))
        })?;
        let standing = Python::attach(|py| {
            read("standing", Vec::new())?
                .extract::<(bool, String)>(py)
                .map(|(identified, basis)| ResponseStanding { identified, basis })
                .map_err(|_| wrong("standing", "an (identified, basis) pair"))
        })?;
        let held_out_design = Python::attach(|py| {
            let (kind, reason) = read("heldout_design", Vec::new())?
                .extract::<(String, String)>(py)
                .map_err(|_| wrong("heldout_design", "a (kind, reason) pair"))?;
            Ok::<_, PyErr>(match kind.as_str() {
                "independent_draws" => HeldOutDesign::IndependentDraws,
                "held_out_units" => HeldOutDesign::HeldOutUnits,
                _ => HeldOutDesign::Unavailable(reason),
            })
        })?;
        let observable = Python::attach(|py| {
            read("observable_variables", Vec::new())?
                .extract::<Option<Vec<u32>>>(py)
                .map(|ids| ids.map(|ids| ids.into_iter().map(VariableId::from_raw).collect()))
                .map_err(|_| wrong("observable_variables", "None or a list of variable indices"))
        })?;
        Ok(Self {
            object,
            identity,
            optimization_draws,
            held_out_draws,
            horizon,
            standing,
            held_out_design,
            observable,
        })
    }

    fn call<T>(
        &self,
        method: &str,
        args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>>,
        convert: impl for<'py> FnOnce(Bound<'py, PyAny>) -> PyResult<T>,
    ) -> Result<T, AdaptivePolicySearchError> {
        Python::attach(|py| {
            let tuple = args(py)?;
            let value = self.object.bind(py).call_method1(method, tuple)?;
            convert(value)
        })
        .map_err(|e: PyErr| AdaptivePolicySearchError::Source(format!("{method}: {e}")))
    }
}

fn split_name(split: DrawSplit) -> &'static str {
    match split {
        DrawSplit::Optimization => "optimization",
        DrawSplit::HeldOut => "held_out",
    }
}

type RawPath = (f64, Vec<f64>, Vec<f64>, Vec<f64>);

impl PolicyRolloutSource for PySource {
    fn identity(&self) -> String {
        self.identity.clone()
    }
    fn n_draws(&self, split: DrawSplit) -> usize {
        match split {
            DrawSplit::Optimization => self.optimization_draws,
            DrawSplit::HeldOut => self.held_out_draws,
        }
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn standing(&self) -> ResponseStanding {
        self.standing.clone()
    }
    fn heldout_design(&self) -> HeldOutDesign {
        self.held_out_design.clone()
    }
    fn observable_variables(&self) -> Option<Vec<VariableId>> {
        self.observable.clone()
    }
    fn rollout(
        &self,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        split: DrawSplit,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Vec<PolicyPath>, AdaptivePolicySearchError> {
        if ctx.cancellation.is_cancelled() {
            return Err(AdaptivePolicySearchError::Cancelled);
        }
        let id = family.id().to_owned();
        let raw = self.call(
            "rollout",
            |py| {
                pyo3::types::PyTuple::new(
                    py,
                    [
                        id.into_pyobject(py)?.into_any(),
                        theta.to_vec().into_pyobject(py)?.into_any(),
                        split_name(split).into_pyobject(py)?.into_any(),
                    ],
                )
            },
            |v| v.extract::<Vec<RawPath>>(),
        )?;
        Ok(raw
            .into_iter()
            .map(|(utility, actions, step_costs, step_exposures)| PolicyPath {
                utility,
                actions,
                step_costs,
                step_exposures,
            })
            .collect())
    }
}

fn translate(error: AdaptivePolicySearchError) -> PyErr {
    let message = error.to_string();
    match error {
        AdaptivePolicySearchError::Cancelled => crate::CausalCancelledError::new_err(message),
        AdaptivePolicySearchError::Resource(_) => crate::CausalResourceError::new_err(message),
        other => match other.reason_code() {
            Some(code) => crate::unsupported_py_err(format!("refused: reason={code}: {message}")),
            None => bad(message),
        },
    }
}

fn evaluation_json(e: &antecedent_design::adaptive_policy_search::PolicyEvaluation) -> Value {
    json!({
        "n_draws": e.n_draws,
        "objective": e.objective,
        "mean_utility": e.mean_utility,
        "standard_error": e.standard_error,
        "feasible": e.feasible,
        "constraints": e.constraints.iter().map(|c| json!({
            "id": c.id,
            "violations": c.violations,
            "n_draws": c.n_draws,
            "violation_rate": c.violation_rate,
            "violation_rate_upper_wilson_95": c.violation_rate_upper,
            "tolerance": c.tolerance,
            "worst_observed": c.worst_observed,
            "satisfied": c.satisfied,
            "upper_bound_satisfied": c.upper_bound_satisfied,
        })).collect::<Vec<_>>(),
    })
}

fn stop_name(stop: SearchStop) -> &'static str {
    match stop {
        SearchStop::SpreadConverged => "spread_converged",
        SearchStop::GenerationLimit => "generation_limit",
        SearchStop::EvaluationBudget => "evaluation_budget",
    }
}

/// Run the search described by `config` on a response source and return the result as JSON.
fn run_config(
    config: &Value,
    names: &[String],
    source: &dyn PolicyRolloutSource,
    cancel: Option<PyCancellationToken>,
) -> PyResult<String> {
    let families = config
        .get("families")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("families are required"))?
        .iter()
        .map(|f| parse_family(f, names))
        .collect::<PyResult<Vec<_>>>()?;
    let constraints = config.get("constraints").cloned().unwrap_or(Value::Null);
    let constraints = PolicyConstraints {
        cumulative_cost: chance(&constraints, "cumulative_cost")?,
        cumulative_exposure: chance(&constraints, "cumulative_exposure")?,
        per_step_action: if constraints.get("per_step_action").is_some_and(|v| !v.is_null()) {
            Some(pair(&constraints, "per_step_action")?)
        } else {
            None
        },
    };
    let objective = match opt_text(config, "objective")?.unwrap_or("expected_utility") {
        "expected_utility" => PolicyObjective::ExpectedUtility,
        "lower_tail_mean" => PolicyObjective::LowerTailMean { alpha: num(&config, "tail_alpha")? },
        other => {
            return Err(bad(format!(
                "unknown objective {other:?}; use expected_utility|lower_tail_mean"
            )));
        }
    };
    let defaults = PolicySearchOptions::default();
    let options = PolicySearchOptions {
        seed: opt_count(config, "seed")?.unwrap_or(defaults.seed),
        max_evaluations: opt_size(config, "max_evaluations")?.unwrap_or(defaults.max_evaluations),
        population: opt_size(config, "population")?.unwrap_or(defaults.population),
        elite_fraction: opt_num(config, "elite_fraction")?.unwrap_or(defaults.elite_fraction),
        max_generations: opt_size(config, "max_generations")?.unwrap_or(defaults.max_generations),
        spread_tolerance: opt_num(config, "spread_tolerance")?.unwrap_or(defaults.spread_tolerance),
        polish: opt_flag(config, "polish")?.unwrap_or(defaults.polish),
        standing: match opt_text(config, "standing")?.unwrap_or("fitted_model") {
            "fitted_model" => StandingRequest::FittedModel,
            "identified_policy_effect" => StandingRequest::IdentifiedPolicyEffect,
            other => return Err(bad(format!("unknown standing {other:?}"))),
        },
        held_out: match opt_text(config, "held_out")?.unwrap_or("preferred") {
            "preferred" => HeldOutRequirement::Preferred,
            "required" => HeldOutRequirement::Required,
            other => return Err(bad(format!("unknown held_out requirement {other:?}"))),
        },
    };
    let baseline = opt_size(config, "baseline")?;
    let prepared = PreparedAdaptivePolicySearch::compile(
        families,
        baseline,
        constraints,
        objective,
        options.clone(),
    )
    .map_err(translate)?;
    let mut ctx = antecedent_core::ExecutionContext::production_default(options.seed);
    crate::apply_cancel(&mut ctx, cancel);
    crate::execution_api::apply(&mut ctx);
    let result = prepared.execute(source, &ctx).map_err(translate)?;

    let standing = match &result.standing {
        PolicyStanding::FittedModelOptimization { note } => {
            json!({"label": result.standing.label(), "note": note})
        }
        PolicyStanding::IdentifiedPolicyEffect { basis } => {
            json!({"label": result.standing.label(), "basis": basis})
        }
    };
    let out = json!({
        "standing": standing,
        "complete": result.complete,
        "flags": result.flags,
        "selected": result.selected,
        "selection_optimism": result.selection_optimism,
        "source_identity": result.source_identity,
        "problem_identity": hex(&result.problem_identity),
        "record_identity": hex(&result.record_identity),
        "admitted_bytes": result.admitted_bytes(),
        "families": result.families.iter().map(|f| json!({
            "id": f.family_id,
            "identity": hex(&f.family_identity),
            "theta": f.theta,
            "in_sample": evaluation_json(&f.in_sample),
            "held_out": f.held_out.as_ref().map(evaluation_json),
            "evaluations": f.evaluations,
            "generations": f.generations,
            "stop": stop_name(f.stop),
            "failed_attempts": f.attempts.iter().filter(|a| a.failure.is_some()).count(),
            "attempts": f.attempts.iter().map(|a| json!({
                "theta": a.theta, "objective": a.objective, "feasible": a.feasible,
                "excess_violation": a.excess_violation.is_finite().then_some(a.excess_violation),
                "failure": a.failure,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "contrasts": result.contrasts.iter().map(|c| json!({
            "policy": c.policy, "baseline": c.baseline, "split": split_name(c.split),
            "n_draws": c.n_draws, "mean_difference": c.mean_difference,
            "standard_error": c.standard_error,
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string(&out).map_err(|e| bad(e.to_string()))
}

/// Run a bounded history-dependent policy search against a Python response source and return
/// the result as JSON text.
#[pyfunction]
#[pyo3(signature = (config_json, source, cancel=None))]
fn adaptive_policy_search(
    config_json: &str,
    source: Py<PyAny>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<String> {
    let config: Value = serde_json::from_str(config_json).map_err(|e| bad(e.to_string()))?;
    run_config(&config, &[], &PySource::bind(source)?, cancel)
}

fn scalar(spec: &Value, key: &str, default: f64) -> PyResult<f64> {
    Ok(opt_num(spec, key)?.unwrap_or(default))
}

#[pymethods]
impl crate::gcm_api::PyFittedGcm {
    /// Optimise bounded history-dependent policies against this fitted model. Variables in
    /// the JSON configuration are named. The model's own noise gives the shared draws;
    /// parameter uncertainty of the fit is not propagated, so the result is a fitted-model
    /// optimisation and cannot be labelled an identified policy effect unless the
    /// configuration states an identification basis.
    #[pyo3(signature = (config_json, cancel=None))]
    fn optimize_adaptive_policy(
        &self,
        py: Python<'_>,
        config_json: &str,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<String> {
        let config: Value = serde_json::from_str(config_json).map_err(|e| bad(e.to_string()))?;
        let names: Vec<String> = self.names.to_vec();
        let spec = config.get("model_source").ok_or_else(|| bad("model_source is required"))?;
        let var = |key: &str| variable_of(spec, key, &names);
        let utility = spec.get("utility").cloned().unwrap_or(Value::Null);
        let stage = StageUtility {
            outcome: var("outcome")?,
            target: scalar(&utility, "target", 0.0)?,
            over: scalar(&utility, "over", 0.0)?,
            under: scalar(&utility, "under", 0.0)?,
            quadratic: scalar(&utility, "quadratic", 0.0)?,
            action_linear: scalar(&utility, "action_linear", 0.0)?,
            action_quadratic: scalar(&utility, "action_quadratic", 0.0)?,
            discount: scalar(&utility, "discount", 1.0)?,
        };
        let mut initial_state = Vec::new();
        if let Some(Value::Object(map)) = spec.get("initial_state") {
            for (name, value) in map {
                let id = names
                    .iter()
                    .position(|n| n == name)
                    .ok_or_else(|| bad(format!("unknown variable {name:?}")))?;
                let value =
                    value.as_f64().ok_or_else(|| bad("initial_state values are numbers"))?;
                initial_state.push((VariableId::from_raw(id as u32), value));
            }
        }
        let mut observed = Vec::new();
        for item in spec.get("observed").and_then(Value::as_array).map_or(&[][..], Vec::as_slice) {
            let name = item.as_str().ok_or_else(|| bad("observed lists variable names"))?;
            let id = names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| bad(format!("unknown variable {name:?}")))?;
            observed.push(VariableId::from_raw(id as u32));
        }
        let design = ModelRolloutDesign {
            horizon: bounded_size(spec, "horizon", 0)?,
            rows: bounded_size(spec, "rows", 1000)?,
            held_out_rows: bounded_size(spec, "held_out_rows", 0)?,
            noise_seed: opt_count(spec, "noise_seed")?.unwrap_or(0),
            initial_state,
            unit_cost: scalar(spec, "unit_cost", 0.0)?,
            exposure_weight: scalar(spec, "exposure_weight", 0.0)?,
            decision_interval: bounded_size(spec, "decision_interval", 1)?,
        };
        let cells = (design.rows.max(design.held_out_rows) as u64)
            .saturating_mul(names.len().max(1) as u64);
        if cells > MAX_ROLLOUT_CELLS {
            return Err(crate::CausalResourceError::new_err(format!(
                "rows and held_out_rows over {} variables exceed the bound of \
                 {MAX_ROLLOUT_CELLS} simulated values",
                names.len()
            )));
        }
        let identified = opt_flag(spec, "identified")?.unwrap_or(false);
        let basis = opt_text(spec, "identification_basis")?.unwrap_or(
            "fitted structural model; identification of the policy effect not established",
        );
        let source = ModelPolicySource::new(
            vec![Arc::new(self.inner.model.clone())],
            var("action")?,
            stage,
            design,
            observed,
            ResponseStanding { identified, basis: basis.to_owned() },
        )
        .map_err(translate)?;
        let _ = py;
        run_config(&config, &names, &source, cancel)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(adaptive_policy_search, module)?)?;
    Ok(())
}

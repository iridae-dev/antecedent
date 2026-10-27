//! Python bridge for the mixed-evidence decision workflow.
//!
//! One decision rests on a cluster-randomized trial, a temporal period effect, a Bayesian cost
//! model and an uncertain endpoint-to-margin bridge. Rust owns every fit, the joint combination
//! under the declared dependence, the decision evaluation, the current source authority held in an
//! evidence ledger, the exported decision artifact and its independent loader and replayer, and the
//! claim classification after the trial is withdrawn. Python chooses the dependence, the execution
//! controls and the change to replay, and reads JSON back.
//!
//! A refusal that names the source authority or the artifact comes back as structured JSON for the
//! Python layer to raise as a typed refusal; an unusable request, a corrupt artifact and an
//! implementation failure raise.

use antecedent::analysis::{
    DecisionArtifact, FlagshipDependence, FlagshipError, FlagshipWorkflow, MixedDecision,
    load_decision, replay_decision,
};
use antecedent_contract::mixed_module::{
    CutFeedback, GaussianLaw, MixedModulePolicy, SharedParameterModel,
};
use antecedent_core::ExecutionContext;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::measurement_model_api::{member, object, parse, present, text};

const STAGE: &str = "mixed_decision";

/// Stack the later replays run on: the nested trial, temporal and posterior frames together exceed
/// the stack of a thread the host created.
const REPLAY_STACK_BYTES: usize = 16 * 1024 * 1024;

fn on_big_stack<T: Send>(run: impl FnOnce() -> T + Send) -> PyResult<T> {
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .stack_size(REPLAY_STACK_BYTES)
            .spawn_scoped(scope, run)
            .map_err(|error| crate::py_msg(format!("the route thread could not start: {error}")))?;
        handle.join().map_err(|payload| {
            crate::py_msg(format!(
                "internal Rust panic: {}",
                crate::errors::panic_payload_msg(payload.as_ref())
            ))
        })
    })
}

fn refusal(code: &str, detail: &str, message: &str, remedy: &str) -> String {
    json!({
        "code": code,
        "stage": STAGE,
        "detail": detail,
        "message": message,
        "offending": Json::Null,
        "remedy": remedy,
    })
    .to_string()
}

/// A workflow error as a typed refusal, or the error to raise.
fn classified(error: &FlagshipError) -> Result<String, PyErr> {
    match error {
        FlagshipError::Authority(inner) => Ok(refusal(
            inner.code,
            inner.detail,
            &inner.message,
            "decide again from sources that are still current, or recompute the decision",
        )),
        FlagshipError::Verification(inner) => Ok(refusal(
            inner.code,
            &inner.detail,
            &inner.message,
            inner.remedy.as_deref().unwrap_or("declare a claim graph that validates"),
        )),
        FlagshipError::Margin(inner) => Ok(refusal(
            inner.code,
            &inner.detail,
            &inner.detail,
            inner.remedy.unwrap_or("declare the margin mapping the bridge needs"),
        )),
        FlagshipError::Route(detail)
        | FlagshipError::Dependence(detail)
        | FlagshipError::Policy(detail) => Ok(refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "mixed_decision.invalid_request",
            detail,
            "supply a request the workflow can run: a lawful covariance, a finite positive \
                 scale, an artifact that matches its retained identity",
        )),
        FlagshipError::Serialization(inner) => Err(crate::serialization_err(inner.to_string())),
        other => Err(crate::py_msg(other.to_string())),
    }
}

type Payload<T> = (Option<T>, Option<String>);

fn settle<T>(result: Result<T, FlagshipError>) -> PyResult<Payload<T>> {
    match result {
        Ok(value) => Ok((Some(value), None)),
        Err(error) => Ok((None, Some(classified(&error)?))),
    }
}

fn dependence_of(text_json: &str) -> PyResult<FlagshipDependence> {
    let value = parse(text_json, "mixed decision dependence")?;
    let map = object(&value, "dependence")?;
    let real = |key: &str| -> PyResult<f64> {
        present(map, key)
            .and_then(Json::as_f64)
            .filter(|v| v.is_finite())
            .ok_or_else(|| crate::value_err(format!("field `{key}` must be a finite number")))
    };
    let empty = serde_json::Map::new();
    let model = present(map, "model").and_then(Json::as_object).unwrap_or(&empty);
    Ok(match text(map, "kind")?.as_str() {
        "independent" => FlagshipDependence::Independent,
        "covariance" => FlagshipDependence::Covariance {
            id: text(map, "id")?,
            margin_temporal: real("margin_temporal")?,
            margin_cost: real("margin_cost")?,
            temporal_cost: real("temporal_cost")?,
        },
        "correlated" => FlagshipDependence::Correlated {
            id: text(map, "id")?,
            margin_temporal: real("margin_temporal")?,
            margin_cost_correlation: real("margin_cost_correlation")?,
            temporal_cost: real("temporal_cost")?,
        },
        "shared_parameter" => {
            let law = |key: &str| -> PyResult<GaussianLaw> {
                let map = object(member(model, key)?, key)?;
                let value = |name: &str| -> PyResult<f64> {
                    present(map, name).and_then(Json::as_f64).filter(|v| v.is_finite()).ok_or_else(
                        || crate::value_err(format!("`{key}.{name}` must be a finite number")),
                    )
                };
                Ok(GaussianLaw { mean: value("mean")?, variance: value("variance")? })
            };
            let weight = |key: &str| -> PyResult<f64> {
                present(model, key)
                    .and_then(Json::as_f64)
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| crate::value_err(format!("`{key}` must be a finite number")))
            };
            FlagshipDependence::SharedParameter {
                id: text(map, "id")?,
                model: SharedParameterModel {
                    margin_prior: law("margin_prior")?,
                    cost_prior: law("cost_prior")?,
                    pooling_weight: weight("pooling_weight")?,
                    margin_observation: law("margin_observation")?,
                    cost_observation: law("cost_observation")?,
                    specific_prior: law("specific_prior")?,
                    feedback_weight: weight("feedback_weight")?,
                },
                margin_temporal: real("margin_temporal")?,
                temporal_cost: real("temporal_cost")?,
            }
        }
        other => {
            return Err(crate::value_err(format!(
                "unknown dependence `{other}`; use independent, covariance, correlated or \
                 shared_parameter"
            )));
        }
    })
}

fn policy_of(policy_json: Option<&str>) -> PyResult<MixedModulePolicy> {
    let Some(policy_json) = policy_json else { return Ok(MixedModulePolicy::coherent_joint()) };
    let value = parse(policy_json, "mixed decision policy")?;
    let map = object(&value, "policy")?;
    let modular = |cut_feedback: CutFeedback| -> PyResult<MixedModulePolicy> {
        MixedModulePolicy::declared_modular(&text(map, "reconciliation")?, cut_feedback)
            .map_err(|refusal| crate::value_err(format!("{refusal:?}")))
    };
    match text(map, "kind")?.as_str() {
        "coherent_joint" => Ok(MixedModulePolicy::coherent_joint()),
        "cut" => modular(CutFeedback::Cut),
        "feedback" => modular(CutFeedback::Feedback),
        other => Err(crate::value_err(format!(
            "unknown policy `{other}`; use coherent_joint, cut or feedback"
        ))),
    }
}

fn evaluation_json(evaluation: &antecedent_design::decision_eval::DecisionResult) -> Json {
    use antecedent_design::decision_eval::Verdict;
    let (verdict, chosen): (&str, Vec<&str>) = match &evaluation.verdict {
        Verdict::UniquelyOptimal(action) => ("uniquely_optimal", vec![action.as_str()]),
        Verdict::Indistinguishable(actions) => {
            ("indistinguishable", actions.iter().map(String::as_str).collect())
        }
        Verdict::NoAdmissibleAction => ("no_admissible_action", Vec::new()),
    };
    json!({
        "contract_identity": evaluation.contract_identity,
        "criterion_label": evaluation.criterion_label,
        "verdict": verdict,
        "chosen": chosen,
        "evpi": evaluation.evpi,
        "actions": evaluation.actions.iter().map(|a| json!({
            "id": a.id,
            "admissible": a.admissible,
            "expected_utility": a.expected_utility,
        })).collect::<Vec<_>>(),
    })
}

fn decision_json(decision: &MixedDecision) -> Json {
    let dependence = match &decision.dependence {
        antecedent::analysis::joint_law_propagation::LinearDependence::DeclaredIndependent => {
            json!({"kind": "independent", "id": Json::Null})
        }
        antecedent::analysis::joint_law_propagation::LinearDependence::Covariance { id } => {
            json!({"kind": "covariance", "id": id})
        }
    };
    json!({
        "mean": decision.joint.value("mean"),
        "variance": decision.joint.value("variance"),
        "identity": decision.joint.identity(),
        "operation": decision.joint.operation(),
        "bridge_value": decision.bridge_value,
        "temporal_variance": decision.temporal_variance,
        "cost_variance": decision.cost_variance,
        "margin": {
            "plug_in_mean": decision.margin.plug_in_mean,
            "mean": decision.margin.mean,
            "variance": decision.margin.variance,
        },
        "dependence": dependence,
        "evaluation": evaluation_json(&decision.evaluation),
    })
}

fn artifact_payload(py: Python<'_>, artifact: DecisionArtifact) -> (Py<PyBytes>, String) {
    (PyBytes::new(py, &artifact.bytes).unbind(), artifact.identity)
}

/// The fitted workflow: the three routes, the joint decision and the current source authority.
#[pyclass(name = "MixedDecisionWorkflowHandle", skip_from_py_object)]
pub(crate) struct MixedDecisionWorkflowHandle {
    inner: FlagshipWorkflow,
    seed: u64,
    threads: Option<u32>,
    memory: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
}

impl MixedDecisionWorkflowHandle {
    fn ctx(&self) -> ExecutionContext {
        crate::recalc_api::bounded_context(
            self.seed,
            self.threads,
            self.memory,
            self.cancel.clone(),
        )
    }
}

#[pymethods]
impl MixedDecisionWorkflowHandle {
    /// What the three routes estimated, as JSON.
    fn summary_json(&self) -> String {
        json!({
            "cluster_trials": self.inner.cluster_trials(),
            "cluster_effect": self.inner.cluster_effect(),
            "cluster_variance": self.inner.cluster_variance(),
            "cluster_uncertainty": self.inner.cluster_uncertainty(),
            "cluster_design": self.inner.cluster_design(),
            "temporal_effect": self.inner.temporal_effect(),
            "temporal_means": self.inner.temporal_means(),
            "cost_mean": self.inner.cost_mean(),
            "cost_draws": self.inner.cost_draws(),
        })
        .to_string()
    }

    /// The current decision, or the refusal when its source authority is withdrawn.
    fn decision_json(&self) -> PyResult<Payload<String>> {
        settle(self.inner.decision().map(|d| decision_json(d).to_string()))
    }

    /// Export the decision for an independent consumer: `(bytes, identity)` or the refusal.
    fn export_decision(&self, py: Python<'_>) -> PyResult<Payload<(Py<PyBytes>, String)>> {
        settle(self.inner.export_decision().map(|a| artifact_payload(py, a)))
    }

    /// Export the numerical inputs for a historical replay that grants no current authority.
    fn export_historical_replay(&self, py: Python<'_>) -> PyResult<Payload<(Py<PyBytes>, String)>> {
        settle(self.inner.export_historical_replay().map(|a| artifact_payload(py, a)))
    }

    /// Consume a decision against this workflow's current source authority.
    #[pyo3(signature = (artifact, expected_identity=None))]
    fn load(&self, artifact: &[u8], expected_identity: Option<&str>) -> PyResult<Payload<String>> {
        let ctx = self.ctx();
        let loaded = on_big_stack(|| {
            load_decision(artifact, expected_identity, self.inner.authority(), &ctx)
        })?;
        settle(loaded.map(|l| {
            json!({
                "mean": l.mean,
                "variance": l.variance,
                "evaluation": evaluation_json(&l.evaluation),
            })
            .to_string()
        }))
    }

    /// Recompute the decision at a new benefit and refit no route.
    fn utility_only(&mut self, benefit: f64) -> PyResult<Payload<String>> {
        let ctx = self.ctx();
        let replay = on_big_stack(|| self.inner.utility_only(benefit, &ctx))?;
        settle(replay.map(|r| {
            json!({
                "temporal_factor_builds": r.temporal_factor_builds,
                "temporal_identifications": r.temporal_identifications,
                "bayesian_model_fits": r.bayesian_model_fits,
                "bayesian_posterior_draws": r.bayesian_posterior_draws,
                "bayesian_identifications": r.bayesian_identifications,
            })
            .to_string()
        }))
    }

    /// Replace the cost prior scale and refit only the cost posterior.
    fn prior_change(&mut self, prior_scale: f64) -> PyResult<Payload<String>> {
        let ctx = self.ctx();
        let replay = on_big_stack(|| self.inner.prior_change(prior_scale, &ctx))?;
        settle(replay.map(|r| {
            json!({
                "bayesian_model_fits": r.bayesian_model_fits,
                "bayesian_identifications": r.bayesian_identifications,
                "score_recomputed": r.score_recomputed,
                "identification_reused": r.identification_reused,
            })
            .to_string()
        }))
    }

    /// Replace the temporal observation period and refit only the temporal mechanisms.
    fn period_change(&mut self, start: u64, end: u64) -> PyResult<Payload<String>> {
        let ctx = self.ctx();
        let replay = on_big_stack(|| self.inner.period_change((start, end), &ctx))?;
        settle(replay.map(|r| {
            json!({
                "temporal_factor_builds": r.temporal_factor_builds,
                "temporal_identifications": r.temporal_identifications,
                "score_recomputed": r.score_recomputed,
                "identification_reused": r.identification_reused,
            })
            .to_string()
        }))
    }

    /// Withdraw the cluster trial and classify every claim: the sealed composition verification
    /// as `(artifact bytes, identity)`, for the composition-verification consumer.
    fn withdraw_cluster(&mut self, py: Python<'_>) -> PyResult<Payload<(Py<PyBytes>, String)>> {
        let sealed = on_big_stack(|| {
            let verification = self.inner.withdraw_cluster()?;
            let identity = verification.identity().to_owned();
            let bytes = verification
                .export("mixed-decision-withdrawal")
                .map_err(|error| FlagshipError::Verification(Box::new(error)))?;
            Ok::<_, FlagshipError>((bytes, identity))
        })?;
        settle(sealed.map(|(bytes, identity)| (PyBytes::new(py, &bytes).unbind(), identity)))
    }
}

/// Fit every route and the decision at benefit one under `dependence_json`.
#[pyfunction]
#[pyo3(signature = (dependence_json, seed=1, threads=None, memory_limit_bytes=None, cancel=None, policy_json=None))]
fn mixed_decision_execute(
    py: Python<'_>,
    dependence_json: &str,
    seed: u64,
    threads: Option<u32>,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
    policy_json: Option<&str>,
) -> PyResult<Payload<MixedDecisionWorkflowHandle>> {
    let dependence = dependence_of(dependence_json)?;
    let policy = policy_of(policy_json)?;
    let ctx = crate::recalc_api::bounded_context(seed, threads, memory_limit_bytes, cancel.clone());
    let fitted = crate::detach_catch(py, move || {
        Ok(FlagshipWorkflow::execute_with_policy(&ctx, dependence, &policy))
    })?;
    settle(fitted.map(|inner| MixedDecisionWorkflowHandle {
        inner,
        seed,
        threads,
        memory: memory_limit_bytes,
        cancel,
    }))
}

/// Replay a historical decision numerically without granting current source authority:
/// `(mean, variance)` or the refusal.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, seed=1))]
fn mixed_decision_replay(
    artifact: &[u8],
    expected_identity: Option<&str>,
    seed: u64,
) -> PyResult<Payload<(f64, f64)>> {
    let ctx = crate::recalc_api::context(seed, None);
    let replayed = on_big_stack(|| replay_decision(artifact, expected_identity, &ctx))?;
    settle(replayed)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MixedDecisionWorkflowHandle>()?;
    module.add_function(wrap_pyfunction!(mixed_decision_execute, module)?)?;
    module.add_function(wrap_pyfunction!(mixed_decision_replay, module)?)?;
    Ok(())
}

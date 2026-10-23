//! Experiment planning: power, sample size and minimum detectable effect,
//! compiled through the lowered program and executed natively.
//!
//! The request crosses as canonical JSON (the declarations of
//! `antecedent.design.planning`) and the plan returns as JSON that the Python
//! layer reads into typed results. Every refusal is the native one, with its
//! reason code and next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::planning::compile_planning;
use antecedent::design::{
    ArmComparison, FactorialContrast, FamilyPower, InterferenceExposure, Multiplicity, NormalPrior,
    OutcomeModel, PlannedDesign, PlannedExperiment, PlanningAssumptions, PlanningCriterion,
    PlanningGoal, PlanningRequest, SpendingFunction, StratumSpec, SuccessCriterion, TestSidedness,
};
use pyo3::prelude::*;
use serde_json::{Map, Value, json};

use crate::{PyCancellationToken, detach_catch, py_err, value_err};

type Object = Map<String, Value>;

fn invalid(message: impl AsRef<str>) -> PyErr {
    value_err(format!("reason=invalid_argument: planning request: {}", message.as_ref()))
}

fn object<'a>(value: &'a Value, what: &str) -> PyResult<&'a Object> {
    value.as_object().ok_or_else(|| invalid(format!("{what} must be an object")))
}

fn field<'a>(map: &'a Object, key: &str) -> PyResult<&'a Value> {
    map.get(key).ok_or_else(|| invalid(format!("missing field {key:?}")))
}

fn real(map: &Object, key: &str) -> PyResult<f64> {
    field(map, key)?.as_f64().ok_or_else(|| invalid(format!("{key:?} must be a number")))
}

fn real_or(map: &Object, key: &str, default: f64) -> PyResult<f64> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value.as_f64().ok_or_else(|| invalid(format!("{key:?} must be a number"))),
    }
}

fn optional_real(map: &Object, key: &str) -> PyResult<Option<f64>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            value.as_f64().map(Some).ok_or_else(|| invalid(format!("{key:?} must be a number")))
        }
    }
}

fn count(map: &Object, key: &str) -> PyResult<u64> {
    field(map, key)?
        .as_u64()
        .ok_or_else(|| invalid(format!("{key:?} must be a nonnegative integer")))
}

fn text<'a>(map: &'a Object, key: &str) -> PyResult<&'a str> {
    field(map, key)?.as_str().ok_or_else(|| invalid(format!("{key:?} must be a string")))
}

fn named<T: Copy>(name: &str, all: &[T], as_str: fn(T) -> &'static str, what: &str) -> PyResult<T> {
    all.iter()
        .copied()
        .find(|value| as_str(*value) == name)
        .ok_or_else(|| invalid(format!("unknown {what} {name:?}")))
}

fn prior(value: &Value, what: &str) -> PyResult<NormalPrior> {
    let map = object(value, what)?;
    Ok(NormalPrior { mean: real(map, "mean")?, sd: real(map, "sd")? })
}

fn spending(name: &str) -> PyResult<SpendingFunction> {
    named(
        name,
        &[SpendingFunction::OBrienFlemingType, SpendingFunction::PocockType],
        SpendingFunction::as_str,
        "spending function",
    )
}

fn design(value: &Value) -> PyResult<PlannedDesign> {
    let map = object(value, "design")?;
    Ok(match text(map, "kind")? {
        "bernoulli" => PlannedDesign::Bernoulli,
        "complete" => PlannedDesign::Complete,
        "stratified" => {
            let strata = field(map, "strata")?
                .as_array()
                .ok_or_else(|| invalid("strata must be a list"))?
                .iter()
                .map(|stratum| {
                    let stratum = object(stratum, "stratum")?;
                    Ok(StratumSpec {
                        share: real(stratum, "share")?,
                        within: real(stratum, "within")?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            PlannedDesign::Stratified { strata }
        }
        "cluster" => PlannedDesign::Cluster {
            mean_cluster_size: real(map, "mean_cluster_size")?,
            icc: real(map, "icc")?,
            cluster_size_cv: real_or(map, "cluster_size_cv", 0.0)?,
        },
        "multi_arm" => {
            let treatment_arms = usize::try_from(count(map, "treatment_arms")?)
                .map_err(|_| invalid("treatment_arms is too large"))?;
            let comparison = match text(map, "comparison")? {
                "omnibus" => ArmComparison::Omnibus,
                "many_to_one" => ArmComparison::ManyToOne {
                    multiplicity: named(
                        text(map, "multiplicity")?,
                        &Multiplicity::ALL,
                        Multiplicity::as_str,
                        "multiplicity",
                    )?,
                    power: named(
                        text(map, "family_power")?,
                        &FamilyPower::ALL,
                        FamilyPower::as_str,
                        "family power",
                    )?,
                },
                other => return Err(invalid(format!("unknown comparison {other:?}"))),
            };
            PlannedDesign::MultiArm { treatment_arms, comparison }
        }
        "factorial" => PlannedDesign::Factorial {
            contrast: named(
                text(map, "contrast")?,
                &FactorialContrast::ALL,
                FactorialContrast::as_str,
                "factorial contrast",
            )?,
        },
        "switchback" => PlannedDesign::Switchback {
            sequences: count(map, "sequences")?,
            periods_per_block: count(map, "periods_per_block")?,
            autocorrelation: real_or(map, "autocorrelation", 0.0)?,
            carryover: real_or(map, "carryover", 0.0)?,
            washout_periods: match map.get("washout_periods") {
                None | Some(Value::Null) => 0,
                Some(_) => count(map, "washout_periods")?,
            },
        },
        "group_sequential" => PlannedDesign::GroupSequential {
            information_fractions: field(map, "information_fractions")?
                .as_array()
                .ok_or_else(|| invalid("information_fractions must be a list"))?
                .iter()
                .map(|f| {
                    f.as_f64().ok_or_else(|| invalid("an information fraction is not a number"))
                })
                .collect::<PyResult<Vec<_>>>()?,
            spending: spending(text(map, "spending")?)?,
        },
        "interference" => PlannedDesign::Interference {
            exposure: match map.get("exposure") {
                None | Some(Value::Null) => None,
                Some(exposure) => {
                    let exposure = object(exposure, "exposure")?;
                    Some(InterferenceExposure::ContainedInClusters {
                        mean_cluster_size: real(exposure, "mean_cluster_size")?,
                        icc: real(exposure, "icc")?,
                        cluster_size_cv: real_or(exposure, "cluster_size_cv", 0.0)?,
                    })
                }
            },
        },
        "outcome_adaptive" => PlannedDesign::OutcomeAdaptive {
            rule: match map.get("rule") {
                None | Some(Value::Null) => None,
                Some(rule) => {
                    Some(rule.as_str().ok_or_else(|| invalid("rule must be a string"))?.to_string())
                }
            },
        },
        other => return Err(invalid(format!("unknown design {other:?}"))),
    })
}

fn assumptions(value: &Value) -> PyResult<PlanningAssumptions> {
    let map = object(value, "assumptions")?;
    let outcome = match text(map, "outcome")? {
        "continuous" => OutcomeModel::Continuous {
            sd: real(map, "sd")?,
            control_mean: optional_real(map, "control_mean")?,
        },
        "binary" => OutcomeModel::Binary { control_rate: real(map, "control_rate")? },
        other => return Err(invalid(format!("unknown outcome {other:?}"))),
    };
    let sidedness = match map.get("sidedness") {
        None | Some(Value::Null) => TestSidedness::TwoSided,
        Some(_) => {
            named(text(map, "sidedness")?, &TestSidedness::ALL, TestSidedness::as_str, "sidedness")?
        }
    };
    let covariates = match map.get("covariates") {
        None | Some(Value::Null) => 0,
        Some(_) => u32::try_from(count(map, "covariates")?)
            .map_err(|_| invalid("covariates is too large"))?,
    };
    Ok(PlanningAssumptions {
        outcome,
        effect: optional_real(map, "effect")?,
        alpha: real_or(map, "alpha", 0.05)?,
        sidedness,
        allocation_ratio: real_or(map, "allocation_ratio", 1.0)?,
        attrition: real_or(map, "attrition", 0.0)?,
        covariate_r_squared: real_or(map, "covariate_r_squared", 0.0)?,
        covariates,
    })
}

fn criterion(value: &Value) -> PyResult<PlanningCriterion> {
    let map = object(value, "criterion")?;
    Ok(match text(map, "kind")? {
        "power" => PlanningCriterion::Power { target: real(map, "target")? },
        "assurance" => {
            let success = object(field(map, "success")?, "success")?;
            let success = match text(success, "kind")? {
                "significance" => SuccessCriterion::Significance,
                "posterior" => SuccessCriterion::Posterior {
                    analysis_prior: prior(field(success, "analysis_prior")?, "analysis prior")?,
                    threshold: real(success, "threshold")?,
                    probability: real(success, "probability")?,
                },
                other => return Err(invalid(format!("unknown success criterion {other:?}"))),
            };
            PlanningCriterion::Assurance {
                design_prior: prior(field(map, "design_prior")?, "design prior")?,
                success,
                target: real(map, "target")?,
            }
        }
        "posterior_precision" => PlanningCriterion::PosteriorPrecision {
            analysis_prior: prior(field(map, "analysis_prior")?, "analysis prior")?,
            max_posterior_sd: real(map, "max_posterior_sd")?,
        },
        "correct_decision" => PlanningCriterion::CorrectDecision {
            design_prior: prior(field(map, "design_prior")?, "design prior")?,
            analysis_prior: prior(field(map, "analysis_prior")?, "analysis prior")?,
            threshold: real(map, "threshold")?,
            probability: real(map, "probability")?,
            target: real(map, "target")?,
        },
        other => return Err(invalid(format!("unknown criterion {other:?}"))),
    })
}

fn goal(value: &Value) -> PyResult<PlanningGoal> {
    let map = object(value, "goal")?;
    Ok(match text(map, "kind")? {
        "sample_size" => PlanningGoal::SampleSize { max_units: count(map, "max_units")? },
        "evaluate" => PlanningGoal::Evaluate { units: count(map, "units")? },
        "minimum_detectable_effect" => {
            PlanningGoal::MinimumDetectableEffect { units: count(map, "units")? }
        }
        other => return Err(invalid(format!("unknown goal {other:?}"))),
    })
}

fn request(text_json: &str) -> PyResult<PlanningRequest> {
    let value: Value =
        serde_json::from_str(text_json).map_err(|error| invalid(format!("not JSON: {error}")))?;
    let map = object(&value, "request")?;
    let mut request = PlanningRequest::new(
        design(field(map, "design")?)?,
        assumptions(field(map, "assumptions")?)?,
        criterion(field(map, "criterion")?)?,
        goal(field(map, "goal")?)?,
    );
    if let Some(simulation) = map.get("simulation").filter(|value| !value.is_null()) {
        let simulation = object(simulation, "simulation")?;
        let replicates = u32::try_from(count(simulation, "replicates")?)
            .map_err(|_| invalid("replicates is too large"))?;
        request = request.with_simulation(replicates, count(simulation, "seed")?);
    }
    Ok(request)
}

fn finite(value: f64) -> Value {
    if value.is_finite() { json!(value) } else { Value::Null }
}

fn optional(value: Option<f64>) -> Value {
    value.map_or(Value::Null, finite)
}

fn reals(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(finite).collect())
}

fn encode(executed: &PlannedExperiment) -> Value {
    let result = &executed.result;
    let analytic = &result.analytic;
    let family = analytic.family.as_ref().map_or(Value::Null, |family| {
        json!({
            "any": finite(family.any),
            "all": finite(family.all),
            "per_comparison": finite(family.per_comparison),
            "critical": reals(&family.critical),
            "correlation": finite(family.correlation),
        })
    });
    let sequential = analytic.sequential.as_ref().map_or(Value::Null, |sequential| {
        json!({
            "critical": reals(&sequential.critical),
            "look_units": sequential.look_units,
            "stop_probability_alternative": reals(&sequential.stop_probability_alternative),
            "stop_probability_null": reals(&sequential.stop_probability_null),
            "expected_units_alternative": finite(sequential.expected_units_alternative),
            "expected_units_null": finite(sequential.expected_units_null),
            "fixed_design_units": sequential.fixed_design_units,
        })
    });
    let simulation = result.simulation.as_ref().map_or(Value::Null, |simulated| {
        let family = simulated.family.map_or(Value::Null, |family| {
            json!({
                "any": finite(family.any),
                "all": finite(family.all),
                "per_comparison": finite(family.per_comparison),
            })
        });
        let sequential = simulated.sequential.as_ref().map_or(Value::Null, |sequential| {
            json!({
                "stop_frequency": reals(&sequential.stop_frequency),
                "expected_units": finite(sequential.expected_units),
                "expected_units_se": finite(sequential.expected_units_se),
            })
        });
        json!({
            "value": finite(simulated.value),
            "monte_carlo_se": finite(simulated.monte_carlo_se),
            "replicates": simulated.replicates,
            "seed": simulated.seed,
            "failed_replicates": simulated.failed_replicates,
            "mean_estimate": finite(simulated.mean_estimate),
            "estimate_sd": finite(simulated.estimate_sd),
            "family": family,
            "sequential": sequential,
            "agreement_z": optional(simulated.agreement_z),
        })
    });
    json!({
        "program_identity": executed.program_identity.to_hex(),
        "request_identity": result.request_identity.to_hex(),
        "design": result.design,
        "analysis": result.analysis,
        "criterion": result.criterion,
        "goal": result.goal,
        "unit": result.unit,
        "units": result.units,
        "arm_units": result.arm_units,
        "enrolled_individuals": finite(result.enrolled_individuals),
        "analysed_individuals": finite(result.analysed_individuals),
        "sample_size": result.sample_size(),
        "effect": optional(result.effect),
        "target": finite(result.target),
        "target_met": result.target_met,
        "analytic": {
            "value": finite(analytic.value),
            "estimator_variance": finite(analytic.estimator_variance),
            "noncentrality": optional(analytic.noncentrality),
            "reference_df": optional(analytic.reference_df),
            "critical_value": optional(analytic.critical_value),
            "design_effect": optional(analytic.design_effect),
            "power": optional(analytic.power),
            "family": family,
            "sequential": sequential,
            "method": analytic.method,
        },
        "simulation": simulation,
        "assumptions": result.assumptions,
        "calibration": result.calibration,
    })
}

/// Compile and execute an experiment-planning request given as JSON; return
/// the plan as JSON.
#[pyfunction]
#[pyo3(signature = (request_json, threads=1, cancel=None))]
fn plan_experiment_py(
    py: Python<'_>,
    request_json: &str,
    threads: u32,
    cancel: Option<PyCancellationToken>,
) -> PyResult<String> {
    let request = request(request_json)?;
    let seed = request.simulation.map_or(0, |spec| spec.seed);
    let ctx = crate::py_execution_context_cancel(seed, threads.max(1), cancel);
    let executed = detach_catch(py, move || {
        compile_planning(&request, &ctx).and_then(|plan| plan.execute(&ctx)).map_err(py_err)
    })?;
    Ok(encode(&executed).to_string())
}

/// Identity of the lowered program a planning request compiles to.
#[pyfunction]
fn planning_program_identity_py(request_json: &str) -> PyResult<String> {
    let request = request(request_json)?;
    let ctx = crate::py_execution_context(0, 1);
    compile_planning(&request, &ctx).map(|plan| plan.identity().to_hex()).map_err(py_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(plan_experiment_py, module)?)?;
    module.add_function(wrap_pyfunction!(planning_program_identity_py, module)?)?;
    Ok(())
}

//! Python bridge for the derived uncertainty of an adaptive procedure.
//!
//! Python declares the whole data-collection procedure (assignment rule,
//! looks, outcome delay and loss, exposure failures, stopping rule, final
//! selection), either directly or as the elements the retained analysis
//! contract does not keep, and asks for the exact law of its terminal state, a
//! simulation of the order-dependent weighted estimator, or one paradigm's
//! report. The premise gates, the dynamic program, the simulation, every
//! number and every refusal are Rust's; a refusal comes back as structured
//! JSON for the Python layer to raise as a typed exception.
//!
//! Every result is labelled `derived_unmeasured`: exact (or simulated with a
//! reported Monte Carlo error) under the stated model, with no interval or
//! coverage licence.

use antecedent::analysis::adaptive_procedure_declaration::{
    AdaptiveProcedure, ProcedureElements, Retention,
};
use antecedent::analysis::adaptive_procedure_report::{
    ProcedureReport, ProcedureReportBody, ProcedureReportRequest, RepeatedMeasuresProcedure,
    RepeatedMeasuresReport, RepeatedMeasuresReportBody, RepeatedMeasuresReportRequest,
};
use antecedent_stats::adaptive_inference::group_sequential::{Sidedness, StagewiseLook};
use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, AssignmentRule, Attribution, BetaPrior, Estimand, EstimandStatement,
    ExposureModel, FinalAnalysis, FinalSelection, GaussianMixtureLaw, InversionGrid,
    ObservedTerminal, OutcomeModelPrior, OutcomeObservation, OutcomeStructure,
    ProcedureDeclaration, RepeatedMeasuresDeclaration, RepeatedMeasuresRequest, ResourceReceipt,
    SimulationRequest, StoppingRule, UncertaintyParadigm, UnitAssignment, UnitSummary,
    WeightScheme, WithinUnitCorrelation,
};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::adaptive_inference_api::AdaptiveInferenceHandle;
use crate::measurement_model_api::{
    index, list, object, opt_flag, parse, present, real, reals, small, text, unsigned,
};
use crate::transport_common::execution_context;

use crate::adaptive_inference_api::{Refused, refused, refused_all};

fn prior_of(map: &Map<String, Json>, alpha: &str, beta: &str) -> PyResult<BetaPrior> {
    Ok(BetaPrior { alpha: real(map, alpha)?, beta: real(map, beta)? })
}

fn optional<T>(
    map: &Map<String, Json>,
    key: &str,
    build: impl FnOnce(&Json) -> PyResult<T>,
) -> PyResult<Option<T>> {
    present(map, key).map(build).transpose()
}

fn rule_of(value: &Json) -> PyResult<AssignmentRule> {
    let map = object(value, "assignment rule")?;
    Ok(match text(map, "kind")?.as_str() {
        "fixed" => AssignmentRule::Fixed { probabilities: reals(map, "probabilities")? },
        "epsilon_greedy" => AssignmentRule::EpsilonGreedy {
            epsilon: real(map, "epsilon")?,
            prior: prior_of(map, "prior_alpha", "prior_beta")?,
        },
        "thompson_beta" => {
            AssignmentRule::ThompsonBeta { prior: prior_of(map, "prior_alpha", "prior_beta")? }
        }
        "thompson_beta_floored" => AssignmentRule::ThompsonBetaFloored {
            prior: prior_of(map, "prior_alpha", "prior_beta")?,
            min_probability: real(map, "min_probability")?,
        },
        "upper_confidence_bound" => {
            AssignmentRule::UpperConfidenceBound { exploration: real(map, "exploration")? }
        }
        other => return Err(crate::value_err(format!("unknown assignment rule `{other}`"))),
    })
}

fn observation_of(value: &Json) -> PyResult<OutcomeObservation> {
    let map = object(value, "outcome observation")?;
    Ok(OutcomeObservation {
        lag_looks: index(map, "lag_looks")?,
        retain_success: real(map, "retain_success")?,
        retain_failure: real(map, "retain_failure")?,
    })
}

fn exposure_of(value: &Json) -> PyResult<ExposureModel> {
    let map = object(value, "exposure model")?;
    let attribution = match text(map, "attribution")?.as_str() {
        "assigned" => Attribution::Assigned,
        "delivered" => Attribution::Delivered,
        other => return Err(crate::value_err(format!("unknown attribution `{other}`"))),
    };
    Ok(ExposureModel {
        failure_probability: reals(map, "failure_probability")?,
        fallback_arm: index(map, "fallback_arm")?,
        attribution,
    })
}

fn stopping_of(value: &Json) -> PyResult<StoppingRule> {
    let map = object(value, "stopping rule")?;
    Ok(match text(map, "kind")?.as_str() {
        "no_stopping" => StoppingRule::NoStopping,
        "posterior_probability_best" => StoppingRule::PosteriorProbabilityBest {
            threshold: real(map, "threshold")?,
            prior: prior_of(map, "prior_alpha", "prior_beta")?,
        },
        "posterior_expected_loss" => StoppingRule::PosteriorExpectedLoss {
            loss_threshold: real(map, "loss_threshold")?,
            prior: prior_of(map, "prior_alpha", "prior_beta")?,
        },
        "posterior_futility" => StoppingRule::PosteriorFutility {
            margin: real(map, "margin")?,
            threshold: real(map, "threshold")?,
            prior: prior_of(map, "prior_alpha", "prior_beta")?,
        },
        "wald_boundaries" => StoppingRule::WaldBoundaries {
            critical: reals(map, "critical")?,
            sidedness: match text(map, "sidedness")?.as_str() {
                "one_sided_upper" => Sidedness::OneSidedUpper,
                "two_sided_symmetric" => Sidedness::TwoSidedSymmetric,
                other => return Err(crate::value_err(format!("unknown sidedness `{other}`"))),
            },
        },
        "resource_cap" => StoppingRule::ResourceCap { max_units: small(map, "max_units")? },
        other => return Err(crate::value_err(format!("unknown stopping rule `{other}`"))),
    })
}

fn selection_of(value: &Json) -> PyResult<FinalSelection> {
    let map = object(value, "final selection")?;
    Ok(match text(map, "kind")?.as_str() {
        "no_selection" => FinalSelection::NoSelection,
        "highest_observed_mean" => FinalSelection::HighestObservedMean,
        "highest_posterior_mean" => {
            FinalSelection::HighestPosteriorMean(prior_of(map, "prior_alpha", "prior_beta")?)
        }
        other => return Err(crate::value_err(format!("unknown final selection `{other}`"))),
    })
}

fn unsigned_list(map: &Map<String, Json>, key: &str) -> PyResult<Vec<u64>> {
    list(map, key)?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| crate::value_err(format!("`{key}` holds a non-count"))))
        .collect()
}

fn declaration_of(value: &Json) -> PyResult<ProcedureDeclaration> {
    let map = object(value, "procedure declaration")?;
    Ok(ProcedureDeclaration {
        arms: index(map, "arms")?,
        context_probabilities: optional(map, "context_probabilities", |_| {
            reals(map, "context_probabilities")
        })?,
        assignment: optional(map, "assignment", rule_of)?,
        enrolment: optional(map, "enrolment", |_| {
            unsigned_list(map, "enrolment")?
                .into_iter()
                .map(|n| {
                    u32::try_from(n)
                        .map_err(|_| crate::value_err("an enrolment does not fit in 32 bits"))
                })
                .collect()
        })?,
        observation: optional(map, "observation", observation_of)?,
        exposure: optional(map, "exposure", exposure_of)?,
        stopping: optional(map, "stopping", stopping_of)?,
        selection: optional(map, "selection", selection_of)?,
    })
}

fn elements_of(value: &Json) -> PyResult<ProcedureElements> {
    let map = object(value, "procedure elements")?;
    Ok(ProcedureElements {
        assignment: optional(map, "assignment", rule_of)?,
        context_probabilities: optional(map, "context_probabilities", |_| {
            reals(map, "context_probabilities")
        })?,
        retention: optional(map, "retention", |v| {
            let r = object(v, "retention")?;
            Ok(Retention { success: real(r, "success")?, failure: real(r, "failure")? })
        })?,
        exposure: optional(map, "exposure", exposure_of)?,
        selection: optional(map, "selection", selection_of)?,
        stopping_prior: optional(map, "stopping_prior", |v| {
            let p = object(v, "stopping prior")?;
            prior_of(p, "alpha", "beta")
        })?,
    })
}

fn prior_json(prior: &BetaPrior) -> Json {
    json!({"prior_alpha": prior.alpha, "prior_beta": prior.beta})
}

fn rule_json(rule: &AssignmentRule) -> Json {
    match rule {
        AssignmentRule::Fixed { probabilities } => {
            json!({"kind": "fixed", "probabilities": probabilities})
        }
        AssignmentRule::EpsilonGreedy { epsilon, prior } => {
            let mut v = prior_json(prior);
            v["kind"] = json!("epsilon_greedy");
            v["epsilon"] = json!(epsilon);
            v
        }
        AssignmentRule::ThompsonBeta { prior } => {
            let mut v = prior_json(prior);
            v["kind"] = json!("thompson_beta");
            v
        }
        AssignmentRule::ThompsonBetaFloored { prior, min_probability } => {
            let mut v = prior_json(prior);
            v["kind"] = json!("thompson_beta_floored");
            v["min_probability"] = json!(min_probability);
            v
        }
        AssignmentRule::UpperConfidenceBound { exploration } => {
            json!({"kind": "upper_confidence_bound", "exploration": exploration})
        }
        _ => json!({"kind": "unrecognized"}),
    }
}

fn stopping_json(rule: &StoppingRule) -> Json {
    match rule {
        StoppingRule::NoStopping => json!({"kind": "no_stopping"}),
        StoppingRule::PosteriorProbabilityBest { threshold, prior } => {
            let mut v = prior_json(prior);
            v["kind"] = json!("posterior_probability_best");
            v["threshold"] = json!(threshold);
            v
        }
        StoppingRule::PosteriorExpectedLoss { loss_threshold, prior } => {
            let mut v = prior_json(prior);
            v["kind"] = json!("posterior_expected_loss");
            v["loss_threshold"] = json!(loss_threshold);
            v
        }
        StoppingRule::PosteriorFutility { margin, threshold, prior } => {
            let mut v = prior_json(prior);
            v["kind"] = json!("posterior_futility");
            v["margin"] = json!(margin);
            v["threshold"] = json!(threshold);
            v
        }
        StoppingRule::WaldBoundaries { critical, sidedness } => json!({
            "kind": "wald_boundaries",
            "critical": critical,
            "sidedness": sidedness.as_str(),
        }),
        StoppingRule::ResourceCap { max_units } => {
            json!({"kind": "resource_cap", "max_units": max_units})
        }
        _ => json!({"kind": "unrecognized"}),
    }
}

fn selection_json(selection: &FinalSelection) -> Json {
    match selection {
        FinalSelection::NoSelection => json!({"kind": "no_selection"}),
        FinalSelection::HighestObservedMean => json!({"kind": "highest_observed_mean"}),
        FinalSelection::HighestPosteriorMean(prior) => {
            let mut v = prior_json(prior);
            v["kind"] = json!("highest_posterior_mean");
            v
        }
        _ => json!({"kind": "unrecognized"}),
    }
}

fn terminal_of(value: &Json) -> PyResult<ObservedTerminal> {
    let map = object(value, "observed terminal state")?;
    Ok(ObservedTerminal {
        look: index(map, "look")?,
        rule_stopped: opt_flag(map, "rule_stopped")?.unwrap_or(false),
        decision: present(map, "decision").map(|_| index(map, "decision")).transpose()?,
        estimate: real(map, "estimate")?,
    })
}

fn counts_of(map: &Map<String, Json>, key: &str) -> PyResult<Option<Vec<u16>>> {
    optional(map, key, |_| {
        unsigned_list(map, key)?
            .into_iter()
            .map(|n| {
                u16::try_from(n)
                    .map_err(|_| crate::value_err(format!("`{key}` holds a count above 65535")))
            })
            .collect()
    })
}

fn estimand_of(value: &Json) -> PyResult<Estimand> {
    let map = object(value, "estimand")?;
    Ok(match text(map, "kind")?.as_str() {
        "arm_mean" => Estimand::ArmMean(index(map, "arm")?),
        "contrast" => Estimand::Contrast { plus: index(map, "plus")?, minus: index(map, "minus")? },
        other => return Err(crate::value_err(format!("unknown estimand `{other}`"))),
    })
}

fn simulation_of(map: &Map<String, Json>) -> PyResult<SimulationRequest> {
    let scheme = match text(map, "scheme")?.as_str() {
        "uniform" => WeightScheme::Uniform,
        "variance_stabilizing" => WeightScheme::VarianceStabilizing,
        other => return Err(crate::value_err(format!("unknown weighting scheme `{other}`"))),
    };
    Ok(SimulationRequest {
        replications: small(map, "replications")?,
        seed: unsigned(map, "seed")?,
        estimand: estimand_of(crate::measurement_model_api::member(map, "estimand")?)?,
        scheme,
        outcome_prior: OutcomeModelPrior {
            mean: real(map, "outcome_prior_mean")?,
            strength: real(map, "outcome_prior_strength")?,
        },
        probability_floor: real(map, "probability_floor")?,
    })
}

fn request_of(value: &Json) -> PyResult<ProcedureReportRequest> {
    let map = object(value, "procedure report request")?;
    Ok(match text(map, "kind")?.as_str() {
        "bayesian_posterior" => ProcedureReportRequest::BayesianPosterior {
            prior: prior_of(map, "prior_alpha", "prior_beta")?,
            observed: counts_of(map, "observed")?,
        },
        "decision_theoretic" => {
            ProcedureReportRequest::DecisionTheoretic { truth: reals(map, "truth")? }
        }
        "fixed_sample" => ProcedureReportRequest::FixedSample {
            truth: reals(map, "truth")?,
            plus: index(map, "plus")?,
            minus: index(map, "minus")?,
            observed: optional(map, "observed", terminal_of)?,
        },
        "group_sequential" => ProcedureReportRequest::GroupSequential {
            truth: reals(map, "truth")?,
            level: real(map, "level")?,
            observed: optional(map, "observed", |v| {
                let o = object(v, "observed stopped statistic")?;
                Ok(StagewiseLook { look: index(o, "look")?, statistic: real(o, "statistic")? })
            })?,
        },
        "adaptive_weighting" => ProcedureReportRequest::AdaptiveWeighting {
            truth: reals(map, "truth")?,
            simulation: simulation_of(map)?,
            z: real(map, "z")?,
        },
        "anytime_valid" => ProcedureReportRequest::AnytimeValid {
            alpha: real(map, "alpha")?,
            tuning_samples: real(map, "tuning_samples")?,
            observed: counts_of(map, "observed")?,
            truth: optional(map, "truth", |_| reals(map, "truth"))?,
        },
        "whole_procedure_inversion" => ProcedureReportRequest::WholeProcedureInversion {
            observed: terminal_of(crate::measurement_model_api::member(map, "observed")?)?,
            grid: InversionGrid {
                contrasts: reals(map, "contrasts")?,
                baselines: reals(map, "baselines")?,
            },
            alpha: real(map, "alpha")?,
        },
        other => return Err(crate::value_err(format!("unknown report kind `{other}`"))),
    })
}

/// What a simulation was admitted for and did.
pub(crate) fn receipt_json(receipt: &ResourceReceipt) -> Json {
    json!({
        "replications": receipt.replications,
        "looks": receipt.looks,
        "states": receipt.states,
        "work_planned": receipt.work_planned,
        "work_done": receipt.work_done,
        "peak_units": receipt.peak_units,
        "estimated_peak_bytes": receipt.estimated_peak_bytes,
    })
}

fn law_json(law: &antecedent_stats::adaptive_inference::EstimatorLaw) -> Json {
    json!({
        "support": law.support.iter().map(|(v, p)| json!([v, p])).collect::<Vec<_>>(),
        "undefined_mass": law.undefined_mass,
    })
}

fn report_json(report: &ProcedureReport) -> Json {
    let body = match &report.body {
        ProcedureReportBody::Bayesian { posterior, mean_gaps } => json!({
            "posterior": posterior.as_ref().map(|p| json!({
                "alpha": p.posteriors.iter().map(|b| b.alpha()).collect::<Vec<_>>(),
                "beta": p.posteriors.iter().map(|b| b.beta()).collect::<Vec<_>>(),
                "mean": p.posteriors.iter().map(|b| b.mean()).collect::<Vec<_>>(),
                "probability_best": p.probability_best,
                "expected_loss": p.expected_loss,
            })),
            "mean_gaps": mean_gaps.iter().map(|g| json!({
                "cell": g.cell,
                "prior_mean": g.prior_mean,
                "expected_posterior_mean": g.expected_posterior_mean,
                "gap": g.gap,
            })).collect::<Vec<_>>(),
        }),
        ProcedureReportBody::DecisionTheoretic(d) => json!({
            "stop_probabilities": d.stop_probabilities,
            "horizon_probability": d.horizon_probability,
            "expected_enrolled": d.expected_enrolled,
            "decision_probabilities": d.decision_probabilities,
            "undecided_probability": d.undecided_probability,
            "wrong_decision_probability": d.wrong_decision_probability,
            "expected_regret": d.expected_regret,
        }),
        ProcedureReportBody::FixedSample { law, ordering } => json!({
            "law": law_json(law),
            "mean": law.mean(),
            "variance": law.variance(),
            "ordering": ordering.as_ref().map(|o| json!({
                "upper": o.upper, "lower": o.lower, "undefined_mass": o.undefined_mass,
            })),
        }),
        ProcedureReportBody::GroupSequential {
            critical,
            information_fractions,
            stop_probabilities,
            horizon_probability,
            expected_enrolled,
            stagewise,
        } => json!({
            "critical": critical,
            "information_fractions": information_fractions,
            "stop_probabilities": stop_probabilities,
            "horizon_probability": horizon_probability,
            "expected_enrolled": expected_enrolled,
            "stagewise": stagewise.as_ref().map(|s| json!({
                "median_unbiased": s.median_unbiased,
                "lower": s.lower,
                "upper": s.upper,
                "p_value_at_null": s.p_value_at_null,
            })),
        }),
        ProcedureReportBody::AdaptiveWeighting {
            summary,
            truth_value,
            mean_enrolled,
            rule_stop_fraction,
            replications,
        } => json!({
            "truth_value": truth_value,
            "defined": summary.defined,
            "mean": summary.mean,
            "standard_deviation": summary.standard_deviation,
            "bias": summary.bias,
            "monte_carlo_standard_error": summary.monte_carlo_standard_error,
            "mean_reported_variance": summary.mean_reported_variance,
            "wald_contains_truth": summary.wald_contains_truth,
            "mean_enrolled": mean_enrolled,
            "rule_stop_fraction": rule_stop_fraction,
            "replications": replications,
            "receipt": report.receipt.as_ref().map(receipt_json),
        }),
        ProcedureReportBody::AnytimeValid { report, exclusion_probability } => json!({
            "bounds": report.as_ref().map(|r| json!({
                "alpha_per_cell": r.alpha_per_cell,
                "cells": r.bounds.iter().map(|b| json!({
                    "cell": b.cell, "units": b.units, "mean": b.mean,
                    "lower": b.lower, "upper": b.upper,
                })).collect::<Vec<_>>(),
            })),
            "exclusion_probability": exclusion_probability,
        }),
        ProcedureReportBody::WholeProcedureInversion(i) => json!({
            "accepted": i.accepted,
            "lower": i.lower,
            "upper": i.upper,
            "laws_evaluated": i.laws_evaluated,
        }),
        _ => json!({}),
    };
    json!({
        "paradigm": report.paradigm.as_str(),
        "procedure_identity": report.procedure_identity,
        "contract_identity": report.contract_identity,
        "estimand": match report.estimand {
            EstimandStatement::AssignedArmIntentToTreat => "assigned_arm_intent_to_treat",
            EstimandStatement::DeliveredArmMean => "delivered_arm_mean",
        },
        "calibration": report.calibration,
        "body": body,
    })
}

/// A declared adaptive procedure.
#[pyclass(name = "AdaptiveProcedureHandle", skip_from_py_object)]
pub(crate) struct AdaptiveProcedureHandle {
    inner: AdaptiveProcedure,
}

#[pymethods]
impl AdaptiveProcedureHandle {
    /// Canonical identity (hex) of the declared procedure.
    fn identity(&self) -> String {
        self.inner.identity().to_owned()
    }

    /// The validated procedure's elements and matrix labels, as JSON.
    fn summary_json(&self) -> String {
        let spec = self.inner.spec();
        let exposure = spec.exposure();
        json!({
            "identity": self.inner.identity(),
            "contract_identity": self.inner.contract_identity(),
            "taken_from_contract": self.inner.taken_from_contract(),
            "arms": spec.arms(),
            "cells": spec.cells(),
            "contexts": spec.contexts(),
            "assignment": rule_json(spec.assignment()),
            "enrolment": spec.enrolment(),
            "observation": {
                "lag_looks": spec.observation().lag_looks,
                "retain_success": spec.observation().retain_success,
                "retain_failure": spec.observation().retain_failure,
                "mechanism": spec.observation().mechanism().as_str(),
            },
            "exposure": {
                "failure_probability": exposure.failure_probability,
                "fallback_arm": exposure.fallback_arm,
                "attribution": match exposure.attribution {
                    Attribution::Assigned => "assigned",
                    Attribution::Delivered => "delivered",
                },
            },
            "stopping": stopping_json(spec.stopping()),
            "selection": selection_json(spec.selection()),
            "family": spec.family_name(),
            "monitoring": spec.monitoring_name(),
            "stopping_kind": spec.stopping_name(),
            "calibration": antecedent_stats::adaptive_inference::DERIVED_UNMEASURED,
        })
        .to_string()
    }

    /// The exact law of the terminal state under a truth, as JSON.
    #[pyo3(signature = (truth, cancel=None))]
    fn law(&self, truth: Vec<f64>, cancel: Option<crate::PyCancellationToken>) -> Refused<String> {
        let ctx = execution_context(0, None, cancel);
        match self.inner.law(&truth, &ctx) {
            Ok(report) => {
                let law = &report.law;
                (
                    Some(
                        json!({
                            "procedure_identity": report.procedure_identity,
                            "calibration": report.calibration,
                            "cumulative_enrolment": law.cumulative_enrolment,
                            "work_units": law.work_units,
                            "peak_states": law.peak_states,
                            "expected_enrolled": law.expected_enrolled(),
                            "rule_stop_probabilities": law.rule_stop_probabilities(),
                            "horizon_probability": law.horizon_probability(),
                            "atoms": law.atoms.iter().map(|a| json!({
                                "look": a.look,
                                "rule_stopped": a.rule_stopped,
                                "observed": a.observed,
                                "decision": a.decision,
                                "enrolled": a.enrolled,
                                "probability": a.probability,
                            })).collect::<Vec<_>>(),
                        })
                        .to_string(),
                    ),
                    None,
                )
            }
            Err(refusal) => refused(&refusal),
        }
    }

    /// Simulate the weighted estimator under the whole procedure, as JSON.
    #[pyo3(signature = (truth, request_json, cancel=None))]
    fn simulate(
        &self,
        truth: Vec<f64>,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let value = parse(request_json, "simulation request")?;
        let request = simulation_of(object(&value, "simulation request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.simulate(&truth, &request, &ctx) {
            Ok(report) => (
                Some(
                    json!({
                        "procedure_identity": report.procedure_identity,
                        "calibration": report.calibration,
                        "mean_enrolled": report.law.mean_enrolled(),
                        "rule_stop_fraction": report.law.rule_stop_fraction(),
                        "receipt": receipt_json(&report.law.receipt),
                        "replicates": report.law.replicates.iter().map(|r| json!({
                            "estimate": r.estimate,
                            "variance": r.variance,
                            "look": r.look,
                            "rule_stopped": r.rule_stopped,
                            "decision": r.decision,
                            "enrolled": r.enrolled,
                            "analysed_units": r.analysed_units,
                        })).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// One paradigm's derived report, as JSON.
    #[pyo3(signature = (request_json, cancel=None))]
    fn report(
        &self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = request_of(&parse(request_json, "procedure report request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.report(&request, &ctx) {
            Ok(report) => (Some(report_json(&report).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }
}

/// Declare an adaptive procedure.
#[pyfunction]
fn adaptive_procedure_declare(
    declaration_json: &str,
) -> PyResult<Refused<AdaptiveProcedureHandle>> {
    let declaration = declaration_of(&parse(declaration_json, "procedure declaration")?)?;
    Ok(match AdaptiveProcedure::declare(declaration) {
        Ok(inner) => (Some(AdaptiveProcedureHandle { inner }), None),
        Err(refusals) => refused_all(&refusals),
    })
}

/// Build the procedure of a prepared contract from the elements the contract
/// does not keep.
#[pyfunction]
#[pyo3(signature = (session, elements_json, cancel=None))]
fn adaptive_procedure_from_contract(
    session: PyRef<'_, AdaptiveInferenceHandle>,
    elements_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<AdaptiveProcedureHandle>> {
    let elements = elements_of(&parse(elements_json, "procedure elements")?)?;
    let ctx = execution_context(0, None, cancel);
    Ok(match session.inner.procedure(&elements, &ctx) {
        Ok(inner) => (Some(AdaptiveProcedureHandle { inner }), None),
        Err(refusals) => refused_all(&refusals),
    })
}

// -- Repeated measures and clustered outcomes -----------------------------------------------------

fn unit_summary_of(map: &Map<String, Json>, key: &str) -> PyResult<UnitSummary> {
    Ok(match text(map, key)?.as_str() {
        "mean" => UnitSummary::Mean,
        "gls" => UnitSummary::Gls,
        other => return Err(crate::value_err(format!("unknown unit summary `{other}`"))),
    })
}

fn unit_assignment_of(value: &Json) -> PyResult<UnitAssignment> {
    let map = object(value, "unit assignment rule")?;
    Ok(match text(map, "kind")?.as_str() {
        "fixed" => UnitAssignment::Fixed { probabilities: reals(map, "probabilities")? },
        "epsilon_greedy" => UnitAssignment::EpsilonGreedy {
            epsilon: real(map, "epsilon")?,
            prior: OutcomeModelPrior {
                mean: real(map, "prior_mean")?,
                strength: real(map, "prior_strength")?,
            },
        },
        other => return Err(crate::value_err(format!("unknown unit assignment rule `{other}`"))),
    })
}

fn correlation_of(value: &Json) -> PyResult<WithinUnitCorrelation> {
    let map = object(value, "within-unit correlation")?;
    Ok(match text(map, "kind")?.as_str() {
        "compound_symmetric" => WithinUnitCorrelation::CompoundSymmetric { rho: real(map, "rho")? },
        "autoregressive" => WithinUnitCorrelation::Autoregressive { rho: real(map, "rho")? },
        other => return Err(crate::value_err(format!("unknown correlation `{other}`"))),
    })
}

fn structure_of(value: &Json) -> PyResult<OutcomeStructure> {
    let map = object(value, "outcome structure")?;
    Ok(match text(map, "kind")?.as_str() {
        "repeated_measures" => OutcomeStructure::RepeatedMeasures {
            measurements: small(map, "measurements")?,
            variance: real(map, "variance")?,
            correlation: optional(map, "correlation", correlation_of)?,
        },
        "hierarchical" => OutcomeStructure::Hierarchical {
            members: small(map, "members")?,
            between_variance: optional(map, "between_variance", |_| real(map, "between_variance"))?,
            within_variance: optional(map, "within_variance", |_| real(map, "within_variance"))?,
        },
        other => return Err(crate::value_err(format!("unknown outcome structure `{other}`"))),
    })
}

fn final_analysis_of(value: &Json) -> PyResult<FinalAnalysis> {
    let map = object(value, "final analysis")?;
    let estimator = object(crate::measurement_model_api::member(map, "estimator")?, "estimator")?;
    let estimator = match text(estimator, "kind")?.as_str() {
        "cluster_robust" => {
            AnalysisEstimator::ClusterRobust { summary: unit_summary_of(estimator, "summary")? }
        }
        "pooled_independent" => AnalysisEstimator::PooledIndependent,
        "adaptively_weighted" => AnalysisEstimator::AdaptivelyWeighted {
            summary: unit_summary_of(estimator, "summary")?,
            scheme: match text(estimator, "scheme")?.as_str() {
                "uniform" => WeightScheme::Uniform,
                "variance_stabilizing" => WeightScheme::VarianceStabilizing,
                other => {
                    return Err(crate::value_err(format!("unknown weighting scheme `{other}`")));
                }
            },
            prior: OutcomeModelPrior {
                mean: real(estimator, "prior_mean")?,
                strength: real(estimator, "prior_strength")?,
            },
            probability_floor: real(estimator, "probability_floor")?,
        },
        other => return Err(crate::value_err(format!("unknown estimator `{other}`"))),
    };
    Ok(FinalAnalysis { estimator, plus: index(map, "plus")?, minus: index(map, "minus")? })
}

fn repeated_measures_declaration_of(value: &Json) -> PyResult<RepeatedMeasuresDeclaration> {
    let map = object(value, "repeated-measures declaration")?;
    Ok(RepeatedMeasuresDeclaration {
        arms: index(map, "arms")?,
        assignment: optional(map, "assignment", unit_assignment_of)?,
        enrolment: optional(map, "enrolment", |_| {
            unsigned_list(map, "enrolment")?
                .into_iter()
                .map(|n| {
                    u32::try_from(n)
                        .map_err(|_| crate::value_err("an enrolment does not fit in 32 bits"))
                })
                .collect()
        })?,
        lag_looks: index(map, "lag_looks")?,
        structure: optional(map, "structure", structure_of)?,
        analysis: optional(map, "analysis", final_analysis_of)?,
        stopping: optional(map, "stopping", stopping_of)?,
    })
}

fn paradigm_of(name: &str) -> PyResult<UncertaintyParadigm> {
    Ok(match name {
        "bayesian_posterior" => UncertaintyParadigm::BayesianPosterior,
        "decision_theoretic_stopping" => UncertaintyParadigm::DecisionTheoreticStopping,
        "frequentist_fixed_sample" => UncertaintyParadigm::FrequentistFixedSample,
        "frequentist_group_sequential" => UncertaintyParadigm::FrequentistGroupSequential,
        "frequentist_adaptive_weighting" => UncertaintyParadigm::FrequentistAdaptiveWeighting,
        "anytime_valid_frequentist" => UncertaintyParadigm::AnytimeValidFrequentist,
        "whole_procedure_inversion" => UncertaintyParadigm::WholeProcedureInversion,
        other => return Err(crate::value_err(format!("unknown paradigm `{other}`"))),
    })
}

fn repeated_measures_simulation_of(map: &Map<String, Json>) -> PyResult<RepeatedMeasuresRequest> {
    Ok(RepeatedMeasuresRequest {
        replications: small(map, "replications")?,
        seed: unsigned(map, "seed")?,
    })
}

fn repeated_measures_request_of(value: &Json) -> PyResult<RepeatedMeasuresReportRequest> {
    let map = object(value, "repeated-measures report request")?;
    Ok(match text(map, "kind")?.as_str() {
        "closed_form" => {
            RepeatedMeasuresReportRequest::ClosedForm { arm_means: reals(map, "arm_means")? }
        }
        "derived" => RepeatedMeasuresReportRequest::Derived {
            arm_means: reals(map, "arm_means")?,
            simulation: repeated_measures_simulation_of(map)?,
            z: real(map, "z")?,
        },
        "paradigm" => RepeatedMeasuresReportRequest::Paradigm {
            paradigm: paradigm_of(&text(map, "paradigm")?)?,
            arm_means: reals(map, "arm_means")?,
            simulation: repeated_measures_simulation_of(map)?,
            z: real(map, "z")?,
        },
        "cell_transfer" => RepeatedMeasuresReportRequest::CellTransfer {
            cell_id: text(map, "cell_id")?,
            paradigm: text(map, "paradigm")?,
        },
        other => {
            return Err(crate::value_err(format!("unknown repeated-measures request `{other}`")));
        }
    })
}

fn mixture_json(law: &GaussianMixtureLaw) -> Json {
    let defined = !law.components.is_empty();
    json!({
        "components": law.components.iter().map(|c| json!({
            "weight": c.weight, "mean": c.mean, "variance": c.variance,
        })).collect::<Vec<_>>(),
        "undefined_mass": law.undefined_mass,
        "defined_mass": law.defined_mass(),
        "mean": defined.then(|| law.mean()),
        "variance": defined.then(|| law.variance()),
    })
}

fn repeated_measures_report_json(report: &RepeatedMeasuresReport) -> Json {
    let body = match &report.body {
        RepeatedMeasuresReportBody::ClosedForm {
            marginal_variance,
            unit_summary_variance,
            design_effect,
            measurements,
            revealed_units,
            contrast_value,
            exact_law,
        } => json!({
            "kind": "closed_form",
            "marginal_variance": marginal_variance,
            "unit_summary_variance": unit_summary_variance,
            "design_effect": design_effect,
            "measurements": measurements,
            "revealed_units": revealed_units,
            "contrast_value": contrast_value,
            "exact_law": exact_law.as_ref().map(mixture_json),
        }),
        RepeatedMeasuresReportBody::Derived(derived) => {
            let s = &derived.simulated;
            json!({
                "kind": "derived",
                "paradigm": derived.paradigm.as_str(),
                "contrast_value": derived.contrast_value,
                "unit_summary_variance": derived.unit_summary_variance,
                "design_effect": derived.design_effect,
                "exact_law": derived.exact_law.as_ref().map(mixture_json),
                "simulated": {
                    "defined": s.defined,
                    "with_variance": s.with_variance,
                    "undefined_fraction": s.undefined_fraction,
                    "mean": s.mean,
                    "variance": s.variance,
                    "bias": s.bias,
                    "monte_carlo_standard_error": s.monte_carlo_standard_error,
                    "mean_reported_variance": s.mean_reported_variance,
                    "wald_contains_truth": s.wald_contains_truth,
                },
                "replications": derived.replications,
                "seed": derived.seed,
                "enrolled": derived.enrolled,
                "mean_analysed_units": derived.mean_analysed_units,
                "receipt": receipt_json(&derived.receipt),
            })
        }
        RepeatedMeasuresReportBody::CellTransfer { cell_id } => {
            json!({"kind": "cell_transfer", "cell_id": cell_id})
        }
        _ => json!({"kind": "unrecognized"}),
    };
    json!({
        "procedure_identity": report.procedure_identity,
        "cell": {
            "id": report.cell.id(),
            "family": report.cell.family,
            "outcome": report.cell.outcome,
            "monitoring": report.cell.monitoring,
            "stopping": report.cell.stopping,
            "paradigm": report.cell.paradigm,
        },
        "route": report.route,
        "calibration": report.calibration,
        "body": body,
    })
}

/// A declared repeated-measures or clustered adaptive procedure.
#[pyclass(name = "RepeatedMeasuresProcedureHandle", skip_from_py_object)]
pub(crate) struct RepeatedMeasuresProcedureHandle {
    inner: RepeatedMeasuresProcedure,
    declaration: Json,
}

#[pymethods]
impl RepeatedMeasuresProcedureHandle {
    /// Canonical identity (hex) of the declared procedure.
    fn identity(&self) -> String {
        self.inner.identity().to_owned()
    }

    /// The matrix cell of the declared procedure, as JSON.
    fn cell_json(&self) -> String {
        let cell = self.inner.cell();
        json!({
            "id": cell.id(),
            "family": cell.family,
            "outcome": cell.outcome,
            "monitoring": cell.monitoring,
            "stopping": cell.stopping,
            "paradigm": cell.paradigm,
        })
        .to_string()
    }

    /// One statement about the procedure, as JSON.
    #[pyo3(signature = (request_json, cancel=None))]
    fn report(
        &self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = repeated_measures_request_of(&parse(
            request_json,
            "repeated-measures report request",
        )?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.report(&request, &ctx) {
            Ok(report) => (Some(repeated_measures_report_json(&report).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// The exact posterior of the arm means and of the declared contrast, as the retained
    /// report's JSON.
    fn posterior(&self, request_json: &str) -> PyResult<Refused<String>> {
        crate::bridge::check_declared_size(
            request_json,
            crate::bridge::MAX_DECLARED_ARTIFACT_BYTES,
            "repeated-measures posterior request",
        )?;
        let wire: antecedent::analysis::repeated_measures_posterior_report::RepeatedMeasuresPosteriorRequestWire =
            serde_json::from_str(request_json).map_err(|error| {
                crate::value_err(format!("invalid repeated-measures posterior request: {error}"))
            })?;
        let outcome = wire.into_spec().and_then(|spec| self.inner.posterior(&spec));
        Ok(match outcome {
            Ok(report) => match serde_json::to_string(&report) {
                Ok(text) => (Some(text), None),
                Err(error) => return Err(crate::serialization_err(error.to_string())),
            },
            Err(refusal) => refused(&refusal),
        })
    }

    /// Write a posterior report (JSON) as the retained artifact text, after recomputing it under
    /// this procedure.
    fn posterior_artifact(&self, report_json: &str) -> PyResult<Refused<String>> {
        crate::bridge::check_declared_size(
            report_json,
            crate::bridge::MAX_DECLARED_ARTIFACT_BYTES,
            "repeated-measures posterior report",
        )?;
        let report: antecedent::analysis::repeated_measures_posterior_report::RepeatedMeasuresPosteriorReport =
            serde_json::from_str(report_json).map_err(|error| {
                crate::value_err(format!("invalid repeated-measures posterior report: {error}"))
            })?;
        Ok(
            match antecedent::analysis::repeated_measures_posterior_report::encode_repeated_measures_posterior_report(&report, &self.inner) {
                Ok(text) => (Some(text), None),
                Err(refusal) => refused(&refusal),
            },
        )
    }

    /// Export a self-contained envelope without altering the historical posterior bytes.
    fn export_posterior(&self, report_json: &str) -> PyResult<Refused<(String, String)>> {
        let (posterior, refusal) = self.posterior_artifact(report_json)?;
        let Some(posterior) = posterior else {
            return Ok((None, refusal));
        };
        let identity = posterior_bundle_identity(&self.declaration, &posterior)?;
        let bundle = PosteriorBundle {
            format: POSTERIOR_BUNDLE_FORMAT.to_owned(),
            version: 1,
            declaration: self.declaration.clone(),
            posterior,
            identity: identity.clone(),
        };
        let text = serde_json::to_string(&bundle)
            .map_err(|error| crate::serialization_err(error.to_string()))?;
        crate::bridge::check_declared_size(
            &text,
            crate::bridge::MAX_DECLARED_ARTIFACT_BYTES,
            "repeated-measures posterior bundle",
        )?;
        Ok((Some((text, identity)), None))
    }

    /// Load a retained posterior artifact, recomputing it under this procedure; returns the
    /// report's JSON.
    fn load_posterior_artifact(&self, artifact: &str) -> PyResult<Refused<String>> {
        crate::bridge::check_declared_size(
            artifact,
            crate::bridge::MAX_DECLARED_ARTIFACT_BYTES,
            "repeated-measures posterior artifact",
        )?;
        Ok(
            match antecedent::analysis::repeated_measures_posterior_report::load_repeated_measures_posterior_report(artifact, &self.inner) {
                Ok(report) => match serde_json::to_string(&report) {
                    Ok(text) => (Some(text), None),
                    Err(error) => return Err(crate::serialization_err(error.to_string())),
                },
                Err(refusal) => refused(&refusal),
            },
        )
    }
}

// This additive envelope retains the exact historical posterior text unchanged.
// Its identity binds the declaration needed by the existing scientific replay owner.
const POSTERIOR_BUNDLE_FORMAT: &str = "antecedent.repeated_measures_posterior_bundle";

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PosteriorBundle {
    format: String,
    version: u32,
    declaration: Json,
    posterior: String,
    identity: String,
}

fn posterior_bundle_identity(declaration: &Json, posterior: &str) -> PyResult<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"antecedent.repeated_measures_posterior_bundle.v1\0");
    let premises = serde_json::to_vec(declaration)
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    hasher.update(
        &u64::try_from(premises.len())
            .map_err(|error| crate::serialization_err(error.to_string()))?
            .to_le_bytes(),
    );
    hasher.update(&premises);
    hasher.update(posterior.as_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

/// Independently replay a self-contained posterior and its declared procedure.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn repeated_measures_posterior_consume(
    artifact: &str,
    expected_identity: Option<&str>,
) -> PyResult<Refused<(String, String)>> {
    crate::bridge::check_declared_size(
        artifact,
        crate::bridge::MAX_DECLARED_ARTIFACT_BYTES,
        "repeated-measures posterior bundle",
    )?;
    let bundle: PosteriorBundle = serde_json::from_str(artifact)
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    if bundle.format != POSTERIOR_BUNDLE_FORMAT || bundle.version != 1 {
        return Err(crate::serialization_err("unsupported repeated-measures posterior bundle"));
    }
    let identity = posterior_bundle_identity(&bundle.declaration, &bundle.posterior)?;
    if identity != bundle.identity || expected_identity.is_some_and(|value| value != identity) {
        return Err(crate::serialization_err(
            "repeated-measures posterior bundle identity mismatch",
        ));
    }
    let declaration = repeated_measures_declaration_of(&bundle.declaration)?;
    let procedure = match RepeatedMeasuresProcedure::declare(declaration) {
        Ok(procedure) => procedure,
        Err(refusals) => return Ok(refused_all(&refusals)),
    };
    Ok(match antecedent::analysis::repeated_measures_posterior_report::load_repeated_measures_posterior_report(
        &bundle.posterior, &procedure,
    ) {
        Ok(report) => (Some((serde_json::to_string(&report).map_err(|error| crate::serialization_err(error.to_string()))?, identity)), None),
        Err(refusal) => refused(&refusal),
    })
}

/// Declare a repeated-measures or clustered adaptive procedure.
#[pyfunction]
fn repeated_measures_procedure_declare(
    declaration_json: &str,
) -> PyResult<Refused<RepeatedMeasuresProcedureHandle>> {
    let retained = parse(declaration_json, "repeated-measures declaration")?;
    let declaration = repeated_measures_declaration_of(&retained)?;
    Ok(match RepeatedMeasuresProcedure::declare(declaration) {
        Ok(inner) => (Some(RepeatedMeasuresProcedureHandle { inner, declaration: retained }), None),
        Err(refusals) => refused_all(&refusals),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AdaptiveProcedureHandle>()?;
    module.add_function(wrap_pyfunction!(adaptive_procedure_declare, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_procedure_from_contract, module)?)?;
    module.add_class::<RepeatedMeasuresProcedureHandle>()?;
    module.add_function(wrap_pyfunction!(repeated_measures_procedure_declare, module)?)?;
    module.add_function(wrap_pyfunction!(repeated_measures_posterior_consume, module)?)?;
    Ok(())
}

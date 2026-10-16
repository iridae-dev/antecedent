//! Python bridge for the participant-level adaptive family.
//!
//! Python declares the procedure (arms, binary responses per participant and
//! their correlation, the priors with their historical sources and discounts,
//! the schedule, the allocation floor, the decisions and the
//! confidence-sequence settings) and asks for a simulation of the whole of it,
//! a look-by-look reading of a retained history, or the retained analysis
//! contract of a history. The model, the posterior update, the allocation, the
//! decisions, the confidence sequences, every refusal and every identity are
//! Rust's; a refusal comes back as structured JSON for the Python layer to
//! raise as a typed exception.
//!
//! Every reading keeps the posterior claim and the anytime-valid frequentist
//! claim in separate, labelled fields, and every result is labelled
//! `derived_unmeasured`.

use antecedent::analysis::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession,
};
use antecedent::analysis::adaptive_participant_procedure::ParticipantProcedure;
use antecedent_stats::adaptive_inference::{
    AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, CorrelationPrior, DecisionRules,
    FutilityRule, HistoricalPrior, LookDecision, LookReading, ParticipantDeclaration,
    ParticipantHistory, ParticipantSimulationRequest, ResolvedArmPrior,
};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::adaptive_inference_api::AdaptiveInferenceHandle;
use crate::measurement_model_api::{
    index, list, object, parse, present, real, small, text, unsigned,
};
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

fn refusal_value(refusal: &AdaptiveInferenceRefusal) -> Json {
    json!({
        "code": refusal.code,
        "stage": "adaptive_inference",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
}

fn refused<T>(refusal: &AdaptiveInferenceRefusal) -> Refused<T> {
    (None, Some(refusal_value(refusal).to_string()))
}

fn refused_all<T>(refusals: &[AdaptiveInferenceRefusal]) -> Refused<T> {
    let mut first = refusals.first().map_or_else(|| json!({}), refusal_value);
    first["refusals"] = Json::Array(refusals.iter().map(refusal_value).collect());
    (None, Some(first.to_string()))
}

fn optional<T>(
    map: &Map<String, Json>,
    key: &str,
    build: impl FnOnce(&Json) -> PyResult<T>,
) -> PyResult<Option<T>> {
    present(map, key).map(build).transpose()
}

// -- declaration ----------------------------------------------------------------------------------

fn historical_of(value: &Json) -> PyResult<HistoricalPrior> {
    let map = object(value, "historical source")?;
    Ok(HistoricalPrior {
        source: text(map, "source")?,
        successes: unsigned(map, "successes")?,
        trials: unsigned(map, "trials")?,
        discount: real(map, "discount")?,
    })
}

fn arm_prior_of(value: &Json) -> PyResult<ArmPrior> {
    let map = object(value, "arm prior")?;
    Ok(ArmPrior {
        initial: BetaPrior { alpha: real(map, "initial_alpha")?, beta: real(map, "initial_beta")? },
        historical: list(map, "historical")?
            .iter()
            .map(historical_of)
            .collect::<PyResult<Vec<_>>>()?,
    })
}

fn borrowing_of(value: &Json) -> PyResult<BorrowingPlan> {
    let items = value
        .as_array()
        .ok_or_else(|| crate::value_err("a borrowing plan is a list of arm priors"))?;
    Ok(BorrowingPlan { arms: items.iter().map(arm_prior_of).collect::<PyResult<Vec<_>>>()? })
}

fn decisions_of(value: &Json) -> PyResult<DecisionRules> {
    let map = object(value, "decision rules")?;
    Ok(DecisionRules {
        loss_threshold: real(map, "loss_threshold")?,
        futility: optional(map, "futility", |v| {
            let rule = object(v, "futility rule")?;
            Ok(FutilityRule { margin: real(rule, "margin")?, threshold: real(rule, "threshold")? })
        })?,
        participant_cap: present(map, "participant_cap")
            .map(|_| small(map, "participant_cap"))
            .transpose()?,
    })
}

fn declaration_of(value: &Json) -> PyResult<ParticipantDeclaration> {
    let map = object(value, "participant declaration")?;
    Ok(ParticipantDeclaration {
        arms: index(map, "arms")?,
        responses: present(map, "responses").map(|_| small(map, "responses")).transpose()?,
        correlation: present(map, "correlation").map(|_| real(map, "correlation")).transpose()?,
        correlation_prior: optional(map, "correlation_prior", |v| {
            let prior = object(v, "correlation prior")?;
            Ok(CorrelationPrior { alpha: real(prior, "alpha")?, beta: real(prior, "beta")? })
        })?,
        borrowing: optional(map, "borrowing", borrowing_of)?,
        enrolment: present(map, "enrolment")
            .map(|_| {
                list(map, "enrolment")?
                    .iter()
                    .map(|v| {
                        v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| {
                            crate::value_err("`enrolment` holds a value that is not a 32-bit count")
                        })
                    })
                    .collect::<PyResult<Vec<u32>>>()
            })
            .transpose()?,
        lag_looks: index(map, "lag_looks")?,
        min_exploration: present(map, "min_exploration")
            .map(|_| real(map, "min_exploration"))
            .transpose()?,
        decisions: optional(map, "decisions", decisions_of)?,
        anytime: optional(map, "anytime", |v| {
            let settings = object(v, "anytime settings")?;
            Ok(AnytimeSettings {
                alpha: real(settings, "alpha")?,
                tuning_samples: real(settings, "tuning_samples")?,
            })
        })?,
    })
}

fn history_of(value: &Json) -> PyResult<ParticipantHistory> {
    let map = object(value, "participant history")?;
    let counts = |key: &str| -> PyResult<Vec<u32>> {
        list(map, key)?
            .iter()
            .map(|v| {
                v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| {
                    crate::value_err(format!("`{key}` holds a value that is not a 32-bit count"))
                })
            })
            .collect()
    };
    let successes = list(map, "successes")?
        .iter()
        .map(|v| match v {
            Json::Null => Ok(None),
            other => other
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| crate::value_err("`successes` holds a non-count")),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let assignment = list(map, "assignment")?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| crate::value_err("`assignment` holds a non-list row"))?
                .iter()
                .map(|p| {
                    p.as_f64().ok_or_else(|| crate::value_err("`assignment` holds a non-number"))
                })
                .collect::<PyResult<Vec<f64>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(ParticipantHistory {
        batches: counts("batches")?,
        arms: counts("arms")?.into_iter().map(|a| a as usize).collect(),
        successes,
        assignment,
    })
}

// -- rendering ------------------------------------------------------------------------------------

fn history_json(history: &ParticipantHistory) -> Json {
    json!({
        "batches": history.batches,
        "arms": history.arms,
        "successes": history.successes,
        "assignment": history.assignment,
    })
}

fn decision_json(decision: LookDecision) -> Json {
    match decision {
        LookDecision::Continue => json!({"kind": "continue"}),
        LookDecision::Stop { reason, arm } => {
            json!({"kind": "stop", "reason": reason.as_str(), "arm": arm})
        }
    }
}

/// One look: the posterior claim and the frequentist claim are separate
/// objects, each labelled with its paradigm.
fn reading_json(reading: &LookReading) -> Json {
    let posterior = &reading.posterior;
    let frequentist = &reading.frequentist;
    let comparison = |c: &antecedent_stats::adaptive_inference::ComparisonReading| json!({"arm": c.arm, "lower": c.lower, "upper": c.upper});
    json!({
        "look": reading.look,
        "enrolled": reading.enrolled,
        "completed": reading.completed,
        "assignment_probabilities": reading.assignment_probabilities,
        "decision": decision_json(reading.decision),
        "posterior": {
            "paradigm": posterior.paradigm,
            "mean": posterior.mean,
            "sd": posterior.sd,
            "probability_best": posterior.probability_best,
            "expected_loss": posterior.expected_loss,
            "exceeds_control": posterior.exceeds_control,
            "correlation_mean": posterior.correlation_mean,
            "correlation_sd": posterior.correlation_sd,
            "selected_arm": posterior.selected_arm,
        },
        "frequentist": {
            "paradigm": frequentist.paradigm,
            "calibration": frequentist.calibration,
            "alpha": frequentist.alpha,
            "alpha_per_comparison": frequentist.alpha_per_comparison,
            "alpha_per_arm": frequentist.alpha_per_arm,
            "arms": frequentist.arms.iter().map(|a| json!({
                "count": a.count, "mean": a.mean, "lower": a.lower, "upper": a.upper,
            })).collect::<Vec<_>>(),
            "comparisons": frequentist.comparisons.iter().map(comparison).collect::<Vec<_>>(),
            "best_treatment": frequentist.best_treatment,
            "best_treatment_comparison": comparison(&frequentist.best_treatment_comparison),
        },
    })
}

pub(crate) fn priors_json(priors: &[ResolvedArmPrior]) -> Json {
    Json::Array(
        priors
            .iter()
            .map(|p| {
                json!({
                    "alpha": p.prior.alpha,
                    "beta": p.prior.beta,
                    "effective_sample_size": p.effective_sample_size,
                    "borrowed_sample_size": p.borrowed_sample_size,
                    "sources": p.sources.iter().map(|s| json!({
                        "source": s.source,
                        "successes": s.successes,
                        "trials": s.trials,
                        "discount": s.discount,
                        "borrowed_trials": s.borrowed_trials(),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
}

pub(crate) fn readings_json(readings: &[LookReading]) -> Json {
    Json::Array(readings.iter().map(reading_json).collect())
}

// -- the handle -----------------------------------------------------------------------------------

/// A declared participant-level adaptive procedure.
#[pyclass(name = "ParticipantProcedureHandle", skip_from_py_object)]
pub(crate) struct ParticipantProcedureHandle {
    pub(crate) inner: ParticipantProcedure,
}

#[pymethods]
impl ParticipantProcedureHandle {
    /// Canonical identity (hex) of the declared procedure, borrowing included.
    fn identity(&self) -> String {
        self.inner.identity().to_owned()
    }

    /// The validated elements, the priors with their sources and effective
    /// sample sizes, and the matrix cell, as JSON.
    fn summary_json(&self) -> String {
        let spec = self.inner.spec();
        let cell = self.inner.cell();
        let decisions = spec.decisions();
        json!({
            "identity": self.inner.identity(),
            "arms": spec.arms(),
            "responses": spec.responses(),
            "correlation": spec.correlation(),
            "correlation_prior": spec.correlation_prior().map(|p| json!({
                "alpha": p.alpha, "beta": p.beta,
            })),
            "design_effect": spec.design_effect(),
            "priors": priors_json(spec.priors()),
            "enrolment": spec.enrolment(),
            "total_participants": spec.total_participants(),
            "lag_looks": spec.lag_looks(),
            "min_exploration": spec.min_exploration(),
            "decisions": {
                "loss_threshold": decisions.loss_threshold,
                "futility": decisions.futility.map(|f| json!({
                    "margin": f.margin, "threshold": f.threshold,
                })),
                "participant_cap": decisions.participant_cap,
            },
            "anytime": {
                "alpha": spec.anytime().alpha,
                "tuning_samples": spec.anytime().tuning_samples,
                "alpha_per_arm": spec.alpha_per_arm(),
            },
            "cell": cell.id(),
            "calibration": antecedent_stats::adaptive_inference::DERIVED_UNMEASURED,
        })
        .to_string()
    }

    /// Run the whole procedure under the arm means `truth`, as JSON.
    #[pyo3(signature = (truth, request_json, memory_bytes=None, cancel=None))]
    fn simulate(
        &self,
        truth: Vec<f64>,
        request_json: &str,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let value = parse(request_json, "participant simulation request")?;
        let map = object(&value, "participant simulation request")?;
        let request = ParticipantSimulationRequest {
            replications: small(map, "replications")?,
            seed: unsigned(map, "seed")?,
            true_correlation: present(map, "true_correlation")
                .map(|_| real(map, "true_correlation"))
                .transpose()?,
        };
        let ctx = execution_context(0, memory_bytes, cancel);
        Ok(match self.inner.simulate(&truth, &request, &ctx) {
            Ok(report) => {
                let summary = &report.summary;
                let receipt = &report.simulation.receipt;
                let fractions = &summary.stop_fractions;
                (
                    Some(
                        json!({
                            "procedure_identity": report.procedure_identity,
                            "calibration": report.calibration,
                            "receipt": crate::adaptive_procedure_api::receipt_json(receipt),
                            "summary": {
                                "replications": summary.replications,
                                "mean_enrolled": summary.mean_enrolled,
                                "mean_completed": summary.mean_completed,
                                "stop_fractions": {
                                    "expected_loss": fractions.expected_loss,
                                    "futility": fractions.futility,
                                    "recruitment_cap": fractions.recruitment_cap,
                                    "horizon": fractions.horizon,
                                },
                                "decision_fractions": summary.decision_fractions,
                                "covered_every_look": summary.covered_every_look,
                                "covered_at_stop": summary.covered_at_stop,
                                "contrast_covered_at_stop": summary.contrast_covered_at_stop,
                            },
                            "replicates": report.simulation.replicates.iter().map(|r| json!({
                                "look": r.look,
                                "reason": r.reason.as_str(),
                                "decision": r.decision,
                                "enrolled": r.enrolled,
                                "completed": r.completed,
                                "best_treatment": r.best_treatment,
                                "covered_every_look": r.covered_every_look,
                                "covered_at_stop": r.covered_at_stop,
                                "contrast_covered_at_stop": r.contrast_covered_at_stop,
                            })).collect::<Vec<_>>(),
                        })
                        .to_string(),
                    ),
                    None,
                )
            }
            Err(refusal) => refused(&refusal),
        })
    }

    /// Run the procedure once under `truth` and retain what happened, as
    /// JSON: batches, arms, success counts and allocation probabilities.
    #[pyo3(signature = (truth, seed, true_correlation=None, cancel=None))]
    fn simulate_history(
        &self,
        truth: Vec<f64>,
        seed: u64,
        true_correlation: Option<f64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> Refused<String> {
        let ctx = execution_context(0, None, cancel);
        match self.inner.simulate_history(&truth, true_correlation, seed, &ctx) {
            Ok(history) => (Some(history_json(&history).to_string()), None),
            Err(refusal) => refused(&refusal),
        }
    }

    /// Recompute every look of a retained history, as JSON.
    #[pyo3(signature = (history_json, cancel=None))]
    fn read_history(
        &self,
        history_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let history = history_of(&parse(history_json, "participant history")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.read_history(&history, &ctx) {
            Ok(report) => (
                Some(
                    json!({
                        "procedure_identity": report.procedure_identity,
                        "calibration": report.calibration,
                        "priors": priors_json(&report.priors),
                        "readings": readings_json(&report.readings),
                        "decision": decision_json(report.decision),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Re-declare the procedure with another borrowing plan while borrowing
    /// is still open (no look performed).
    fn amend_borrowing(
        &self,
        borrowing_json: &str,
        looks_performed: usize,
    ) -> PyResult<Refused<ParticipantProcedureHandle>> {
        let plan = borrowing_of(&parse(borrowing_json, "borrowing plan")?)?;
        Ok(match self.inner.amend_borrowing(&plan, looks_performed) {
            Ok(inner) => (Some(Self { inner }), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Prepare the retained analysis contract of a history for analysis.
    #[pyo3(signature = (label, arm_labels, history_json, cancel=None))]
    fn prepare_session(
        &self,
        label: &str,
        arm_labels: Vec<String>,
        history_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<AdaptiveInferenceHandle>> {
        let history = history_of(&parse(history_json, "participant history")?)?;
        let ctx = execution_context(0, None, cancel);
        let declaration = match self.inner.retained_declaration(label, arm_labels, &history, &ctx) {
            Ok(declaration) => declaration,
            Err(refusal) => return Ok(refused(&refusal)),
        };
        Ok(match AdaptiveInferenceSession::prepare(declaration) {
            Ok(inner) => (Some(AdaptiveInferenceHandle { inner }), None),
            Err(refusals) => refused_all(&refusals),
        })
    }
}

/// Declare a participant-level adaptive procedure.
#[pyfunction]
fn participant_procedure_declare(
    declaration_json: &str,
) -> PyResult<Refused<ParticipantProcedureHandle>> {
    let declaration = declaration_of(&parse(declaration_json, "participant declaration")?)?;
    Ok(match ParticipantProcedure::declare(declaration) {
        Ok(inner) => (Some(ParticipantProcedureHandle { inner }), None),
        Err(refusals) => refused_all(&refusals),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ParticipantProcedureHandle>()?;
    module.add_function(wrap_pyfunction!(participant_procedure_declare, module)?)?;
    Ok(())
}

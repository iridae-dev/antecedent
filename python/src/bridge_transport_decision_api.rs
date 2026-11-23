//! Python bridge for the predictive bridge transport decision, the assessment of its transport
//! premise and the composition bundle whose terminal is a predictive bridge.
//!
//! Python declares the bridge contract, the predictive source law, the decision contracts and,
//! when the premise is assessed, the source and target validation trials. The compiler programs,
//! the mapping, the premise diagnostic, the discrepancy envelope, the target-line posterior, the
//! bundle replay, every artifact and every refusal are Rust's. A refusal comes back as structured
//! JSON for the Python layer to raise as a typed exception; corrupt artifact bytes raise a
//! serialization error and a cancelled run raises the cancellation error.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::bridge_transport_decision::{
    AssessedBridgePosteriorDecisionResult, AssessedBridgeRobustDecisionError,
    AssessedBridgeRobustDecisionResult, AssessedBridgeTransportDecisionError,
    AssessedBridgeTransportDecisionResult, BridgeTargetPosteriorPremises,
    BridgeTransportDecisionError, BridgeTransportDecisionResult,
    CompiledAssessedBridgePosteriorDecision, CompiledAssessedBridgeRobustDecision,
    CompiledAssessedBridgeTransportDecision, CompiledBridgeTransportDecision,
    compile_assessed_bridge_posterior_decision, compile_assessed_bridge_robust_decision,
    compile_assessed_bridge_transport_decision, compile_bridge_transport_decision,
    consume_assessed_bridge_posterior_decision, consume_assessed_bridge_robust_decision,
    consume_assessed_bridge_transport_decision, consume_bridge_transport_decision,
};
use antecedent::analysis::bridge_transport_premise_assessment::{
    BridgeTransportPremiseAssessmentError, BridgeTransportPremiseAssessmentRequest,
    BridgeTransportPremiseAssessmentResult, CompiledBridgeTransportPremiseAssessment,
    compile_bridge_transport_premise_assessment, consume_bridge_transport_premise_assessment,
};
use antecedent::analysis::composition::{
    BridgeBundleRecomputationReceipt, BridgeBundleRecomputeError, BundleLimits,
    CompiledBridgeBundleRecomputation, CompositionBundle, compile_bridge_bundle_recomputation,
    consume_portable_bridge_bundle_recomputation,
};
use antecedent::analysis::inverse_query::DecisionContract;
use antecedent::analysis::inverse_query::DistributionArtifact;
use antecedent::analysis::{ChainDecisionError, ChainDecisionResult, ChainRobustDecisionResult};
use antecedent::contract::bridge::{
    BridgeContract, BridgeScope, TransportabilityPremise, TrialEffectPair, TrialLevelValidation,
};
use antecedent::design::decision_artifact::contract_from_json_refusal;
use antecedent::design::decision_eval::{DecisionResult, Verdict};
use antecedent::io::IoError;
use antecedent::prelude::ExecutionContext;
use antecedent::scientific::SemanticDigest;
use antecedent::{ScientificBridgeDistributionResult, ScientificBridgeRefusal};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::bridge::{
    MAX_DECLARED_ARTIFACT_BYTES, MAX_SEALED_ARTIFACT_BYTES, RefusalJson, check_declared_len,
    check_declared_size, external_refusal_json,
};
use crate::distribution_api::PyJointDistributionArtifact;

type Refused<T> = (Option<T>, Option<String>);

const DECISION_STAGE: &str = "bridge_transport_decision";
const ASSESSMENT_STAGE: &str = "bridge_transport_premise_assessment";
const BUNDLE_STAGE: &str = "bridge_bundle_recomputation";
const DECISION_REFUSED: &str = "bridge_transport_decision.refused";
const COORDINATE_REFUSED: &str = "bridge_transport_decision.coordinate_refused";
const EVALUATION_REFUSED: &str = "bridge_transport_decision.evaluation_refused";
const FOREIGN_ARTIFACT: &str = "bridge_transport_decision.foreign_artifact";
const ASSESSMENT_REFUSED: &str = "bridge_transport_premise_assessment.refused";
const BUNDLE_REFUSED: &str = "bridge_bundle_recomputation.refused";

// The features the four decision artifacts declare, read to select their own consumer.
const CONDITIONAL_FEATURE: &str = "bridge_transport_decision_v1";
const ASSESSED_FEATURE: &str = "assessed_bridge_transport_decision_v1";
const ROBUST_FEATURE: &str = "assessed_bridge_discrepancy_robust_decision_v1";
const TARGET_POSTERIOR_FEATURE: &str = "assessed_bridge_target_posterior_decision_v1";

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

/// A failure of one of the compilers, executions, exports or consumers, before it is read.
enum Failure {
    Bridge(BridgeTransportDecisionError),
    Assessed(AssessedBridgeTransportDecisionError),
    Robust(AssessedBridgeRobustDecisionError),
    /// The premise assessment, with the code its declaration-level refusal carries.
    Assessment(BridgeTransportPremiseAssessmentError, &'static str),
    Bundle(BridgeBundleRecomputeError),
    /// An artifact whose declared feature is not a bridge transport decision.
    Foreign,
    /// A result handed to a compiled program that did not produce it.
    Route(&'static str),
}

fn binding_refusal(stage: &str, detail: &str, message: &str) -> String {
    // The support gate carries its registered `not_supported` reason in the message of an
    // invalid-state refusal; it keeps that code instead of a binding mismatch.
    let code = if message.contains("reason=not_supported") {
        antecedent::reason::reason_code!("not_supported")
    } else {
        antecedent::reason::reason_code!("bridge_binding_mismatch")
    };
    RefusalJson::new(code, stage, detail, message).json()
}

fn cancelled() -> PyErr {
    crate::CausalCancelledError::new_err("the bridge transport run was cancelled")
}

fn scientific_refusal(refusal: &ScientificBridgeRefusal) -> String {
    crate::bridge::compact_refusal_json(
        refusal.code,
        "scientific_bridge",
        &refusal.detail,
        &refusal.message,
        refusal.remedy.as_deref(),
    )
}

/// A coded artifact refusal as a refusal; any other codec failure raises.
fn io_refusal(error: &IoError, stage: &str) -> PyResult<String> {
    match error {
        IoError::Refused { code, message } => {
            Ok(RefusalJson::from_coded_message(code, stage, message).json())
        }
        other => Err(crate::serialization_err(other.to_string())),
    }
}

// The evaluator's error type is not reachable through the facade, so it is reported by its
// message, as the chain decision binding reports it.
fn evaluation_refusal(error: &impl std::fmt::Display) -> String {
    RefusalJson::new(
        antecedent::reason::reason_code!("decision_contract_unsatisfied"),
        DECISION_STAGE,
        EVALUATION_REFUSED,
        &error.to_string(),
    )
    .json()
}

fn chain_refusal(error: &ChainDecisionError) -> PyResult<String> {
    match error {
        ChainDecisionError::Chain(refusal) => Ok(json!({
            "code": refusal.code(),
            "stage": DECISION_STAGE,
            "detail": refusal.detail(),
            "message": refusal.to_string(),
            "offending": refusal.stage(),
            "expected": null,
            "supplied": null,
            "next_attempt": refusal.next_attempt().name(),
            "remedy": refusal.next_attempt().sentence(),
        })
        .to_string()),
        ChainDecisionError::Contract(inner) => Ok(external_refusal_json(&inner.to_refusal())),
        ChainDecisionError::Evaluation(inner) => Ok(evaluation_refusal(inner)),
        ChainDecisionError::Io(inner) => io_refusal(inner, DECISION_STAGE),
        ChainDecisionError::Coordinate(message) => {
            Ok(binding_refusal(DECISION_STAGE, COORDINATE_REFUSED, message))
        }
    }
}

fn bridge_refusal(error: &BridgeTransportDecisionError) -> PyResult<String> {
    match error {
        BridgeTransportDecisionError::Invalid(message) => {
            Ok(binding_refusal(DECISION_STAGE, DECISION_REFUSED, message))
        }
        BridgeTransportDecisionError::Bridge(refusal) => Ok(scientific_refusal(refusal)),
        BridgeTransportDecisionError::Contract(inner) => {
            Ok(external_refusal_json(&inner.to_refusal()))
        }
        BridgeTransportDecisionError::Chain(inner) => chain_refusal(inner),
        BridgeTransportDecisionError::Io(inner) => io_refusal(inner, DECISION_STAGE),
    }
}

fn assessment_refusal(
    error: &BridgeTransportPremiseAssessmentError,
    invalid_code: &str,
) -> PyResult<String> {
    match error {
        BridgeTransportPremiseAssessmentError::Invalid(message) => {
            let code = if message.contains("reason=not_supported") {
                antecedent::reason::reason_code!("not_supported")
            } else {
                invalid_code
            };
            Ok(RefusalJson::new(code, ASSESSMENT_STAGE, ASSESSMENT_REFUSED, message).json())
        }
        BridgeTransportPremiseAssessmentError::Io(inner) => io_refusal(inner, ASSESSMENT_STAGE),
    }
}

fn assessed_refusal(error: &AssessedBridgeTransportDecisionError) -> PyResult<String> {
    match error {
        AssessedBridgeTransportDecisionError::Invalid(message) => {
            Ok(binding_refusal(DECISION_STAGE, DECISION_REFUSED, message))
        }
        AssessedBridgeTransportDecisionError::Contract(inner) => {
            Ok(external_refusal_json(&inner.to_refusal()))
        }
        AssessedBridgeTransportDecisionError::Assessment(inner) => {
            assessment_refusal(inner, antecedent::reason::reason_code!("bridge_binding_mismatch"))
        }
        AssessedBridgeTransportDecisionError::Bridge(inner) => bridge_refusal(inner),
        AssessedBridgeTransportDecisionError::Io(inner) => io_refusal(inner, DECISION_STAGE),
    }
}

fn robust_refusal(error: &AssessedBridgeRobustDecisionError) -> PyResult<String> {
    match error {
        AssessedBridgeRobustDecisionError::Invalid(message) => {
            Ok(binding_refusal(DECISION_STAGE, DECISION_REFUSED, message))
        }
        AssessedBridgeRobustDecisionError::Assessed(inner) => assessed_refusal(inner),
        AssessedBridgeRobustDecisionError::Contract(inner) => {
            Ok(external_refusal_json(&inner.to_refusal()))
        }
        AssessedBridgeRobustDecisionError::Chain(inner) => chain_refusal(inner),
        AssessedBridgeRobustDecisionError::Io(inner) => io_refusal(inner, DECISION_STAGE),
    }
}

fn bundle_refusal(error: &BridgeBundleRecomputeError) -> PyResult<String> {
    match error {
        BridgeBundleRecomputeError::Bundle(inner) => {
            Ok(crate::composition_bundle_api::error_refusal(inner))
        }
        BridgeBundleRecomputeError::Bridge(refusal) => Ok(scientific_refusal(refusal)),
        BridgeBundleRecomputeError::Io(inner) => io_refusal(inner, BUNDLE_STAGE),
        BridgeBundleRecomputeError::Invalid(message) => {
            Ok(binding_refusal(BUNDLE_STAGE, BUNDLE_REFUSED, message))
        }
        BridgeBundleRecomputeError::NotSupported(inner) => {
            Ok(RefusalJson::new(inner.code(), BUNDLE_STAGE, inner.detail, &inner.to_string())
                .json())
        }
    }
}

impl Failure {
    fn refusal(&self) -> PyResult<String> {
        match self {
            Self::Bridge(error) => bridge_refusal(error),
            Self::Assessed(error) => assessed_refusal(error),
            Self::Robust(error) => robust_refusal(error),
            Self::Assessment(error, invalid_code) => assessment_refusal(error, invalid_code),
            Self::Bundle(error) => bundle_refusal(error),
            Self::Foreign => Ok(RefusalJson::new(
                antecedent::reason::reason_code!("schema_mismatch"),
                DECISION_STAGE,
                FOREIGN_ARTIFACT,
                "the artifact does not declare a bridge transport decision feature",
            )
            .json()),
            Self::Route(message) => Ok(binding_refusal(DECISION_STAGE, DECISION_REFUSED, message)),
        }
    }
}

/// A premise assessment that refused once compiled: its execution, export or replay disagrees.
fn assessment_failure(error: BridgeTransportPremiseAssessmentError) -> Failure {
    Failure::Assessment(error, antecedent::reason::reason_code!("bridge_binding_mismatch"))
}

/// The value, or the refusal; a run whose token was cancelled raises the cancellation error.
fn settle<T>(outcome: Result<T, Failure>, ctx: &ExecutionContext) -> PyResult<Refused<T>> {
    match outcome {
        Ok(value) => Ok((Some(value), None)),
        Err(_) if ctx.cancellation.is_cancelled() => Err(cancelled()),
        Err(failure) => Ok((None, Some(failure.refusal()?))),
    }
}

fn context(
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
) -> ExecutionContext {
    crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel)
}

fn decision_contract(text: &str, what: &str) -> PyResult<Result<DecisionContract, String>> {
    check_declared_size(text, MAX_DECLARED_ARTIFACT_BYTES, what)?;
    Ok(contract_from_json_refusal(text).map_err(|refusal| external_refusal_json(&refusal)))
}

fn bridge_contract(text: &str) -> PyResult<BridgeContract> {
    crate::scientific_bridge_api::parse_contract(&crate::measurement_model_api::parse(
        text,
        "bridge contract",
    )?)
}

// ---------------------------------------------------------------------------
// Results as JSON.
// ---------------------------------------------------------------------------

// The fit type is not reachable through the facade; a macro reads its fields in place.
macro_rules! fit_json {
    ($fit:expr) => {{
        let fit = &$fit;
        json!({
            "intercept": fit.intercept,
            "slope": fit.slope,
            "r_squared": fit.r_squared,
            "trials": fit.trials,
            "surrogate_min": fit.surrogate_min,
            "surrogate_max": fit.surrogate_max,
        })
    }};
}

fn assessment_json(result: &BridgeTransportPremiseAssessmentResult) -> Json {
    json!({
        "program_identity": result.lowered_program_identity.to_hex(),
        "source_fit": fit_json!(result.source_fit),
        "target_fit": fit_json!(result.target_fit),
        "grid": result.grid.iter().map(|point| json!({
            "surrogate_effect": point.surrogate_effect,
            "difference": point.difference,
            "standard_error": point.standard_error,
            "lower": point.lower,
            "upper": point.upper,
        })).collect::<Vec<_>>(),
        "equivalence_screen_passed": result.equivalence_screen_passed,
        "calibration": result.calibration,
    })
}

fn decision_json(decision: &DecisionResult) -> Json {
    let (verdict, chosen): (&str, Vec<&str>) = match &decision.verdict {
        Verdict::UniquelyOptimal(action) => ("uniquely_optimal", vec![action.as_str()]),
        Verdict::Indistinguishable(actions) => {
            ("indistinguishable", actions.iter().map(String::as_str).collect())
        }
        Verdict::NoAdmissibleAction => ("no_admissible_action", Vec::new()),
    };
    json!({
        "contract_identity": decision.contract_identity,
        "criterion": decision.criterion_label,
        "verdict": verdict,
        "chosen": chosen,
        "evpi": decision.evpi,
        "n_draws": decision.n_draws,
        "effective_draws": decision.effective_draws,
        "assumptions": decision.assumptions,
        "actions": decision.actions.iter().map(|action| json!({
            "id": action.id,
            "admissible": action.admissible,
            "expected_utility": action.expected_utility,
            "value": action.value,
            "standard_error": action.standard_error,
            "expected_regret": action.expected_regret,
            "max_regret": action.max_regret,
            "exclusions": action.exclusions.iter().map(|exclusion| json!({
                "constraint_id": exclusion.constraint_id,
                "probability": exclusion.probability,
                "required": exclusion.required,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

fn chain_decision_json(result: &ChainDecisionResult) -> Json {
    let mut body = decision_json(&result.decision);
    body["program_identity"] = json!(result.lowered_program_identity.to_hex());
    body["chain_identity"] = json!(result.chain.identity());
    body["chain_label"] = json!(result.chain.label());
    body
}

fn robust_json(result: &ChainRobustDecisionResult) -> Json {
    json!({
        "program_identity": result.lowered_program_identity.to_hex(),
        "chain_identity": result.chain.identity(),
        "contract_identity": result.contract_identity,
        "dominant_action": result.dominant_action,
        "actions": result.actions.iter().map(|action| json!({
            "id": action.id,
            "utility_lower": action.utility_lower,
            "utility_upper": action.utility_upper,
            "certainly_admissible": action.certainly_admissible,
            "certainly_excluded": action.certainly_excluded,
            "constraints": action.constraints.iter().map(|constraint| json!({
                "id": constraint.id,
                "probability_lower": constraint.probability_lower,
                "probability_upper": constraint.probability_upper,
                "required": constraint.required,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

fn bridge_json(result: &ScientificBridgeDistributionResult) -> Json {
    let inspection = result.inspect();
    json!({
        "kind": inspection.kind.as_str(),
        "mapping": inspection.mapping.as_str(),
        "value_kind": inspection.value_kind.as_str(),
        "n_draws": inspection.n_draws,
        "column": result.bridged().column(),
        "source_mean": inspection.source_mean,
        "source_variance": inspection.source_variance,
        "mean": inspection.mean,
        "variance": inspection.variance,
        "stream": inspection.stream,
        "alignment_identity": inspection.alignment_identity,
        "draws_digest": inspection.draws_digest,
        "source_claim_id": inspection.source_claim_id,
        "claim_id": inspection.claim_id,
        "contract_identity": inspection.contract_identity,
        "transported": inspection.transported,
        "reading": inspection.reading,
        "unavailable": inspection.unavailable,
    })
}

fn receipt_json(receipt: &BridgeBundleRecomputationReceipt) -> Json {
    json!({
        "bundle_identity": receipt.bundle_identity,
        "terminal_node": receipt.terminal_node,
        "terminal_identity": receipt.terminal_identity,
        "program_identity": SemanticDigest::from_bytes(receipt.lowered_program_identity).to_hex(),
        "dependencies": receipt.dependencies.iter().map(|dependency| json!({
            "node_id": dependency.node_id,
            "artifact_identity": dependency.artifact_identity,
            "chain_digest": dependency.chain_digest,
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// The premise assessment.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScopeIn {
    population_id: String,
    regime_id: String,
}

impl ScopeIn {
    fn into_domain(self) -> BridgeScope {
        BridgeScope { population_id: self.population_id, regime_id: self.regime_id }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PremiseIn {
    id: String,
    statement: String,
    established_in: ScopeIn,
    applies_to: ScopeIn,
    #[serde(default)]
    evidence: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialsIn {
    kind: String,
    trials: Vec<[f64; 2]>,
    min_trials: u32,
    min_r_squared: f64,
}

impl TrialsIn {
    fn into_domain(self, what: &str) -> PyResult<TrialLevelValidation> {
        if self.kind != "trial_level_linear" {
            return Err(crate::value_err(format!(
                "the {what} validation trials must be a trial-level linear mapping"
            )));
        }
        Ok(TrialLevelValidation {
            trials: self
                .trials
                .into_iter()
                .map(|[surrogate_effect, terminal_effect]| TrialEffectPair {
                    surrogate_effect,
                    terminal_effect,
                })
                .collect(),
            min_trials: self.min_trials,
            min_r_squared: self.min_r_squared,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssessmentIn {
    premise: PremiseIn,
    source: TrialsIn,
    target: TrialsIn,
    source_trial_ids: Vec<String>,
    target_trial_ids: Vec<String>,
    surrogate_grid: Vec<f64>,
    equivalence_margin: f64,
}

fn assessment_request(text: &str) -> PyResult<BridgeTransportPremiseAssessmentRequest> {
    check_declared_size(text, MAX_DECLARED_ARTIFACT_BYTES, "premise assessment")?;
    let wire: AssessmentIn = serde_json::from_str(text).map_err(|error| {
        crate::value_err(format!("invalid premise assessment declaration: {error}"))
    })?;
    let premise = wire.premise;
    Ok(BridgeTransportPremiseAssessmentRequest {
        premise: TransportabilityPremise {
            id: premise.id,
            statement: premise.statement,
            established_in: premise.established_in.into_domain(),
            applies_to: premise.applies_to.into_domain(),
            evidence: premise.evidence,
        },
        source: wire.source.into_domain("source")?,
        target: wire.target.into_domain("target")?,
        source_trial_ids: wire.source_trial_ids,
        target_trial_ids: wire.target_trial_ids,
        surrogate_grid: wire.surrogate_grid,
        equivalence_margin: wire.equivalence_margin,
    })
}

/// A compiled premise assessment: both trial fits and the grid, retained behind one program.
#[pyclass(name = "BridgePremiseAssessmentHandle", skip_from_py_object)]
pub(crate) struct BridgePremiseAssessmentHandle {
    inner: Arc<CompiledBridgeTransportPremiseAssessment>,
}

/// An executed premise assessment and the program that produced it.
#[pyclass(name = "BridgePremiseAssessmentResultHandle", skip_from_py_object)]
pub(crate) struct BridgePremiseAssessmentResultHandle {
    plan: Arc<CompiledBridgeTransportPremiseAssessment>,
    result: BridgeTransportPremiseAssessmentResult,
}

#[pymethods]
impl BridgePremiseAssessmentHandle {
    /// The identity of the compiled program (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }

    /// Every layer of the compiled program as JSON; nothing is fitted.
    fn plan_view_json(&self) -> String {
        self.inner.plan_view().to_json().to_string()
    }

    /// The support decision of the unit this route runs under, as JSON (`null` while unbound).
    fn unit_support_json(&self) -> String {
        unit_support_of(&self.inner.plan_view()).to_string()
    }

    /// Fit both trial relations and form the discrepancy bands, or the refusal.
    #[pyo3(signature = (seed=1, threads=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<BridgePremiseAssessmentResultHandle>> {
        let ctx = context(seed, threads, cancel);
        let outcome =
            crate::detach_catch(py, || Ok(self.inner.execute(&ctx).map_err(assessment_failure)))?;
        settle(
            outcome.map(|result| BridgePremiseAssessmentResultHandle {
                plan: Arc::clone(&self.inner),
                result,
            }),
            &ctx,
        )
    }
}

#[pymethods]
impl BridgePremiseAssessmentResultHandle {
    /// The fits, the grid, the screen and the calibration status, as JSON.
    fn summary_json(&self) -> String {
        assessment_json(&self.result).to_string()
    }

    /// Recompute, then seal the validation evidence: the artifact bytes, or the refusal.
    #[pyo3(signature = (seed=1, threads=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let ctx = context(seed, threads, cancel);
        let outcome = crate::detach_catch(py, || {
            Ok(self.plan.export(&self.result, &ctx).map_err(assessment_failure))
        })?;
        settle(outcome.map(|bytes| PyBytes::new(py, &bytes).unbind()), &ctx)
    }
}

/// Compile a source-target premise assessment from its declaration JSON, or the refusal.
#[pyfunction]
fn bridge_premise_assessment_compile(
    request_json: &str,
) -> PyResult<Refused<BridgePremiseAssessmentHandle>> {
    crate::catch_ffi(|| {
        let request = assessment_request(request_json)?;
        Ok(match compile_bridge_transport_premise_assessment(request) {
            Ok(inner) => (Some(BridgePremiseAssessmentHandle { inner: Arc::new(inner) }), None),
            Err(error) => (
                None,
                Some(assessment_refusal(
                    &error,
                    antecedent::reason::reason_code!("scientific_bridge_invalid"),
                )?),
            ),
        })
    })
}

/// Recompile and refit a premise assessment artifact under the retained program identity:
/// the result JSON, or the refusal.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, seed=1, threads=None, cancel=None))]
fn bridge_premise_assessment_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    check_declared_len(artifact.len(), MAX_SEALED_ARTIFACT_BYTES, "premise assessment artifact")?;
    let expected = crate::digest::digest(expected_identity, "expected_identity")?;
    let ctx = context(seed, threads, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(consume_bridge_transport_premise_assessment(artifact, expected, &ctx)
            .map_err(assessment_failure))
    })?;
    settle(outcome.map(|result| assessment_json(&result).to_string()), &ctx)
}

// ---------------------------------------------------------------------------
// The bridge transport decision and its assessed variants.
// ---------------------------------------------------------------------------

/// One compiled route: the bridge decision conditional on its declared premise, or that decision
/// bound to a matched premise assessment, alone, with the discrepancy envelope or with the
/// target-line posterior.
#[allow(clippy::large_enum_variant, reason = "one route is held per handle, behind an Arc")]
enum CompiledRoute {
    Conditional(CompiledBridgeTransportDecision),
    Assessed(CompiledAssessedBridgeTransportDecision),
    Robust(CompiledAssessedBridgeRobustDecision),
    TargetPosterior(CompiledAssessedBridgePosteriorDecision),
}

#[allow(clippy::large_enum_variant, reason = "one result is held per handle")]
enum RouteResult {
    Conditional(BridgeTransportDecisionResult),
    Assessed(AssessedBridgeTransportDecisionResult),
    Robust(AssessedBridgeRobustDecisionResult),
    TargetPosterior(AssessedBridgePosteriorDecisionResult),
}

/// The support decision of the unit a compiled route is bound to, as JSON: `null` while the
/// route is bound to no unit. Reading it changes no program identity.
fn unit_support_of(view: &antecedent::planner::PlanView) -> Json {
    view.unit_support().map_or(Json::Null, |support| {
        json!({
            "unit": support.unit(),
            "status": support.status().as_str(),
            "supported": support.supported(),
            "missing_evidence": support.missing_evidence(),
        })
    })
}

impl CompiledRoute {
    fn unit_support(&self) -> Json {
        match self {
            Self::Conditional(plan) => unit_support_of(&plan.plan_view()),
            Self::Assessed(plan) => unit_support_of(&plan.plan_view()),
            Self::Robust(plan) => unit_support_of(&plan.plan_view()),
            Self::TargetPosterior(plan) => unit_support_of(&plan.plan_view()),
        }
    }

    const fn variant(&self) -> &'static str {
        match self {
            Self::Conditional(_) => "conditional",
            Self::Assessed(_) => "assessed",
            Self::Robust(_) => "robust",
            Self::TargetPosterior(_) => "target_posterior",
        }
    }

    fn identity(&self) -> SemanticDigest {
        match self {
            Self::Conditional(plan) => plan.lowered().identity(),
            Self::Assessed(plan) => plan.lowered().identity(),
            Self::Robust(plan) => plan.lowered().identity(),
            Self::TargetPosterior(plan) => plan.lowered().identity(),
        }
    }

    fn plan_view(&self) -> Json {
        match self {
            Self::Conditional(plan) => plan.plan_view().to_json(),
            Self::Assessed(plan) => plan.plan_view().to_json(),
            Self::Robust(plan) => plan.plan_view().to_json(),
            Self::TargetPosterior(plan) => plan.plan_view().to_json(),
        }
    }

    fn execute(&self, ctx: &ExecutionContext) -> Result<RouteResult, Failure> {
        Ok(match self {
            Self::Conditional(plan) => {
                RouteResult::Conditional(plan.execute(ctx).map_err(Failure::Bridge)?)
            }
            Self::Assessed(plan) => {
                RouteResult::Assessed(plan.execute(ctx).map_err(Failure::Assessed)?)
            }
            Self::Robust(plan) => RouteResult::Robust(plan.execute(ctx).map_err(Failure::Robust)?),
            Self::TargetPosterior(plan) => {
                RouteResult::TargetPosterior(plan.execute(ctx).map_err(Failure::Robust)?)
            }
        })
    }

    fn export(
        &self,
        result: &RouteResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Failure> {
        match (self, result) {
            (Self::Conditional(plan), RouteResult::Conditional(result)) => {
                plan.export(result, artifact_id, ctx).map_err(Failure::Bridge)
            }
            (Self::Assessed(plan), RouteResult::Assessed(result)) => {
                plan.export(result, artifact_id, ctx).map_err(Failure::Assessed)
            }
            (Self::Robust(plan), RouteResult::Robust(result)) => {
                plan.export(result, artifact_id, ctx).map_err(Failure::Robust)
            }
            (Self::TargetPosterior(plan), RouteResult::TargetPosterior(result)) => {
                plan.export(result, artifact_id, ctx).map_err(Failure::Robust)
            }
            _ => Err(Failure::Route("the result was not produced by this compiled program")),
        }
    }
}

impl RouteResult {
    const fn variant(&self) -> &'static str {
        match self {
            Self::Conditional(_) => "conditional",
            Self::Assessed(_) => "assessed",
            Self::Robust(_) => "robust",
            Self::TargetPosterior(_) => "target_posterior",
        }
    }

    /// The bridge decision conditional on the declared transport premise.
    fn bridge(&self) -> &BridgeTransportDecisionResult {
        match self {
            Self::Conditional(result) => result,
            Self::Assessed(result) => &result.bridge,
            Self::Robust(result) => &result.assessed.bridge,
            Self::TargetPosterior(result) => &result.assessed.bridge,
        }
    }

    fn assessment(&self) -> Option<&BridgeTransportPremiseAssessmentResult> {
        match self {
            Self::Conditional(_) => None,
            Self::Assessed(result) => Some(&result.assessment),
            Self::Robust(result) => Some(&result.assessed.assessment),
            Self::TargetPosterior(result) => Some(&result.assessed.assessment),
        }
    }

    fn identity(&self) -> SemanticDigest {
        match self {
            Self::Conditional(result) => result.lowered_program_identity,
            Self::Assessed(result) => result.lowered_program_identity,
            Self::Robust(result) => result.lowered_program_identity,
            Self::TargetPosterior(result) => result.lowered_program_identity,
        }
    }

    fn calibration(&self) -> &'static str {
        match self {
            Self::Conditional(_) => "not_measured",
            Self::Assessed(result) => result.assessment.calibration,
            Self::Robust(result) => result.calibration,
            Self::TargetPosterior(result) => result.calibration,
        }
    }

    fn summary(&self) -> Json {
        let bridge = self.bridge();
        let robust = match self {
            Self::Robust(result) => Some(robust_json(&result.robust)),
            _ => None,
        };
        let posterior = match self {
            Self::TargetPosterior(result) => Some(chain_decision_json(&result.decision)),
            _ => None,
        };
        json!({
            "variant": self.variant(),
            "program_identity": self.identity().to_hex(),
            "calibration": self.calibration(),
            "bridge": bridge_json(&bridge.bridge),
            "bridge_program_identity": bridge.lowered_program_identity.to_hex(),
            "conditional_decision": chain_decision_json(&bridge.decision),
            "assessment": self.assessment().map(assessment_json),
            "robust": robust,
            "target_posterior_decision": posterior,
        })
    }
}

/// A compiled bridge transport decision route.
#[pyclass(name = "BridgeTransportDecisionHandle", skip_from_py_object)]
pub(crate) struct BridgeTransportDecisionHandle {
    inner: Arc<CompiledRoute>,
}

/// An executed or consumed bridge transport decision. A consumed one carries no program.
#[pyclass(name = "BridgeTransportDecisionResultHandle", skip_from_py_object)]
pub(crate) struct BridgeTransportDecisionResultHandle {
    plan: Option<Arc<CompiledRoute>>,
    result: RouteResult,
}

#[pymethods]
impl BridgeTransportDecisionHandle {
    /// The identity of the compiled program (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.inner.identity().to_hex()
    }

    /// `conditional`, `assessed`, `robust` or `target_posterior`.
    #[getter]
    fn variant(&self) -> &'static str {
        self.inner.variant()
    }

    /// Every layer of the compiled program as JSON; nothing is mapped or scored.
    fn plan_view_json(&self) -> String {
        self.inner.plan_view().to_string()
    }

    /// The support decision of the unit this route runs under, as JSON (`null` while unbound).
    fn unit_support_json(&self) -> String {
        self.inner.unit_support().to_string()
    }

    /// Execute the retained program, or the refusal.
    #[pyo3(signature = (seed=1, threads=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<BridgeTransportDecisionResultHandle>> {
        let ctx = context(seed, threads, cancel);
        let outcome = crate::detach_catch(py, || Ok(self.inner.execute(&ctx)))?;
        settle(
            outcome.map(|result| BridgeTransportDecisionResultHandle {
                plan: Some(Arc::clone(&self.inner)),
                result,
            }),
            &ctx,
        )
    }
}

#[pymethods]
impl BridgeTransportDecisionResultHandle {
    /// The variant, identities, bridged law, decisions, assessment and robust bounds, as JSON.
    fn summary_json(&self) -> String {
        self.result.summary().to_string()
    }

    /// The bridged draw table under the declared premise, aligned with the source table.
    fn distribution(&self) -> PyJointDistributionArtifact {
        PyJointDistributionArtifact {
            artifact: self.result.bridge().bridge.distribution().clone(),
            source_claim: None,
        }
    }

    /// The source draw table, unchanged.
    fn source_distribution(&self) -> PyJointDistributionArtifact {
        PyJointDistributionArtifact {
            artifact: self.result.bridge().bridge.source().clone(),
            source_claim: None,
        }
    }

    /// Recompute, then seal the independently replayable artifact: the bytes, or the refusal.
    #[pyo3(signature = (artifact_id, seed=1, threads=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        artifact_id: &str,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let Some(plan) = self.plan.as_ref() else {
            return Err(crate::value_err(
                "a consumed bridge transport decision forwards its own artifact",
            ));
        };
        let ctx = context(seed, threads, cancel);
        let outcome = crate::detach_catch(py, || Ok(plan.export(&self.result, artifact_id, &ctx)))?;
        settle(outcome.map(|bytes| PyBytes::new(py, &bytes).unbind()), &ctx)
    }
}

fn compile_route(
    contract: BridgeContract,
    source: DistributionArtifact,
    stream: u64,
    decision: DecisionContract,
    assessment: Option<CompiledBridgeTransportPremiseAssessment>,
    robust: Option<DecisionContract>,
    premises: Option<BridgeTargetPosteriorPremises>,
    ctx: &ExecutionContext,
) -> Result<CompiledRoute, Failure> {
    let bridge = compile_bridge_transport_decision(contract, source, stream, decision)
        .map_err(Failure::Bridge)?;
    let Some(assessment) = assessment else {
        return Ok(CompiledRoute::Conditional(bridge));
    };
    let assessed = compile_assessed_bridge_transport_decision(assessment, bridge)
        .map_err(Failure::Assessed)?;
    Ok(match (robust, premises) {
        (Some(contract), _) => CompiledRoute::Robust(
            compile_assessed_bridge_robust_decision(assessed, contract).map_err(Failure::Robust)?,
        ),
        (None, Some(premises)) => CompiledRoute::TargetPosterior(
            compile_assessed_bridge_posterior_decision(assessed, premises, ctx)
                .map_err(Failure::Robust)?,
        ),
        (None, None) => CompiledRoute::Assessed(assessed),
    })
}

/// Compile a predictive bridge with its transport premise, the source law and a decision
/// contract as one program; with a compiled premise assessment, bind it to the bridge and, at
/// most one of, a maximin contract over the discrepancy envelope or the target-line posterior
/// premises. Returns the handle, or the refusal.
#[pyfunction]
#[pyo3(signature = (contract_json, source, decision_json, stream, assessment=None, robust_json=None, target_posterior_json=None, seed=1, threads=None, cancel=None))]
fn bridge_transport_decision_compile(
    contract_json: &str,
    source: PyRef<'_, PyJointDistributionArtifact>,
    decision_json: &str,
    stream: u64,
    assessment: Option<PyRef<'_, BridgePremiseAssessmentHandle>>,
    robust_json: Option<&str>,
    target_posterior_json: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<BridgeTransportDecisionHandle>> {
    crate::catch_ffi(|| {
        let contract = bridge_contract(contract_json)?;
        let decision = match decision_contract(decision_json, "decision contract")? {
            Ok(decision) => decision,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
        let robust = match robust_json {
            None => None,
            Some(text) => match decision_contract(text, "robust decision contract")? {
                Ok(robust) => Some(robust),
                Err(refusal) => return Ok((None, Some(refusal))),
            },
        };
        let premises = match target_posterior_json {
            None => None,
            Some(text) => {
                check_declared_size(
                    text,
                    MAX_DECLARED_ARTIFACT_BYTES,
                    "target posterior premises",
                )?;
                Some(serde_json::from_str::<BridgeTargetPosteriorPremises>(text).map_err(
                    |error| crate::value_err(format!("invalid target posterior premises: {error}")),
                )?)
            }
        };
        let assessment =
            assessment.map(|handle| CompiledBridgeTransportPremiseAssessment::clone(&handle.inner));
        if assessment.is_none() && (robust.is_some() || premises.is_some()) {
            return Err(crate::value_err(
                "a robust or target-posterior decision needs a compiled premise assessment",
            ));
        }
        if robust.is_some() && premises.is_some() {
            return Err(crate::value_err(
                "declare a robust decision or target-posterior premises, not both",
            ));
        }
        let ctx = context(seed, threads, cancel);
        let compiled = compile_route(
            contract,
            source.inner().clone(),
            stream,
            decision,
            assessment,
            robust,
            premises,
            &ctx,
        );
        settle(compiled.map(|route| BridgeTransportDecisionHandle { inner: Arc::new(route) }), &ctx)
    })
}

/// The feature a decision artifact declares, read only to select its consumer.
#[derive(Deserialize)]
struct FeatureProbe {
    #[serde(default)]
    required_feature: Option<String>,
    #[serde(default)]
    feature: Option<String>,
}

fn consume_route(
    feature: &str,
    artifact: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<RouteResult, Failure> {
    match feature {
        CONDITIONAL_FEATURE => consume_bridge_transport_decision(artifact, expected, ctx)
            .map(RouteResult::Conditional)
            .map_err(Failure::Bridge),
        ASSESSED_FEATURE => consume_assessed_bridge_transport_decision(artifact, expected, ctx)
            .map(RouteResult::Assessed)
            .map_err(Failure::Assessed),
        ROBUST_FEATURE => consume_assessed_bridge_robust_decision(artifact, expected, ctx)
            .map(RouteResult::Robust)
            .map_err(Failure::Robust),
        TARGET_POSTERIOR_FEATURE => {
            consume_assessed_bridge_posterior_decision(artifact, expected, ctx)
                .map(RouteResult::TargetPosterior)
                .map_err(Failure::Robust)
        }
        _ => Err(Failure::Foreign),
    }
}

/// Independently replay a bridge transport decision artifact of any variant under the retained
/// program identity, through the consumer its declared feature names: the result handle, or the
/// refusal. A target-posterior decision replays only under the seed it was compiled with.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, seed=1, threads=None, cancel=None))]
fn bridge_transport_decision_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<BridgeTransportDecisionResultHandle>> {
    check_declared_len(
        artifact.len(),
        MAX_SEALED_ARTIFACT_BYTES,
        "bridge transport decision artifact",
    )?;
    let expected = crate::digest::digest(expected_identity, "expected_identity")?;
    let probe: FeatureProbe = antecedent::io::from_cbor(artifact)
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    let feature = probe.required_feature.or(probe.feature).unwrap_or_default();
    let ctx = context(seed, threads, cancel);
    let outcome =
        crate::detach_catch(py, || Ok(consume_route(&feature, artifact, expected, &ctx)))?;
    settle(outcome.map(|result| BridgeTransportDecisionResultHandle { plan: None, result }), &ctx)
}

// ---------------------------------------------------------------------------
// A bundle whose terminal is a predictive bridge.
// ---------------------------------------------------------------------------

/// A compiled bundle with a live predictive bridge terminal.
#[pyclass(name = "BridgeBundleRecomputationHandle", skip_from_py_object)]
pub(crate) struct BridgeBundleRecomputationHandle {
    inner: Arc<CompiledBridgeBundleRecomputation>,
}

/// A recomputed bridge bundle receipt and the program that produced it.
#[pyclass(name = "BridgeBundleRecomputationResultHandle", skip_from_py_object)]
pub(crate) struct BridgeBundleRecomputationResultHandle {
    plan: Arc<CompiledBridgeBundleRecomputation>,
    receipt: BridgeBundleRecomputationReceipt,
}

#[pymethods]
impl BridgeBundleRecomputationHandle {
    /// The identity of the compiled program (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }

    /// Every layer of the compiled program as JSON; nothing is replayed.
    fn plan_view_json(&self) -> String {
        self.inner.plan_view().to_json().to_string()
    }

    /// Replay every dependency and derive the terminal bridge: the receipt, or the refusal.
    #[pyo3(signature = (seed=1, threads=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<BridgeBundleRecomputationResultHandle>> {
        let ctx = context(seed, threads, cancel);
        let outcome =
            crate::detach_catch(py, || Ok(self.inner.execute(&ctx).map_err(Failure::Bundle)))?;
        settle(
            outcome.map(|receipt| BridgeBundleRecomputationResultHandle {
                plan: Arc::clone(&self.inner),
                receipt,
            }),
            &ctx,
        )
    }
}

#[pymethods]
impl BridgeBundleRecomputationResultHandle {
    /// The bundle, terminal and program identities and every replayed dependency, as JSON.
    fn summary_json(&self) -> String {
        receipt_json(&self.receipt).to_string()
    }

    /// Recompute, then seal the portable bundle and receipt: the bytes, or the refusal.
    #[pyo3(signature = (seed=1, threads=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let ctx = context(seed, threads, cancel);
        let outcome = crate::detach_catch(py, || {
            Ok(self.plan.export_portable(&self.receipt, &ctx).map_err(Failure::Bundle))
        })?;
        settle(outcome.map(|bytes| PyBytes::new(py, &bytes).unbind()), &ctx)
    }
}

/// Compile a bundle (its exported bytes, checked against `bundle_identity`) whose node
/// `terminal_node` is an embedded predictive scientific bridge, with the bridge contract, its
/// source law and its mapping stream: the handle, or the refusal.
#[pyfunction]
fn bridge_bundle_recomputation_compile(
    bundle: &[u8],
    bundle_identity: &str,
    terminal_node: &str,
    contract_json: &str,
    source: PyRef<'_, PyJointDistributionArtifact>,
    stream: u64,
) -> PyResult<Refused<BridgeBundleRecomputationHandle>> {
    crate::catch_ffi(|| {
        check_declared_len(bundle.len(), MAX_SEALED_ARTIFACT_BYTES, "composition bundle")?;
        let contract = bridge_contract(contract_json)?;
        let bundle = match CompositionBundle::from_bytes(
            bundle,
            bundle_identity,
            &BundleLimits::default(),
        ) {
            Ok(bundle) => bundle,
            Err(error) => {
                return Ok((None, Some(crate::composition_bundle_api::error_refusal(&error))));
            }
        };
        Ok(
            match compile_bridge_bundle_recomputation(
                bundle,
                terminal_node,
                contract,
                source.inner().clone(),
                stream,
            ) {
                Ok(inner) => {
                    (Some(BridgeBundleRecomputationHandle { inner: Arc::new(inner) }), None)
                }
                Err(error) => (None, Some(bundle_refusal(&error)?)),
            },
        )
    })
}

/// Reconstruct the terminal bridge from a portable bundle artifact, recompile and replay it
/// under the retained bundle and program identities: the receipt JSON, or the refusal.
#[pyfunction]
#[pyo3(signature = (artifact, expected_bundle, expected_identity, seed=1, threads=None, cancel=None))]
fn bridge_bundle_recomputation_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_bundle: &str,
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    check_declared_len(artifact.len(), MAX_SEALED_ARTIFACT_BYTES, "bridge bundle artifact")?;
    let expected = crate::digest::digest(expected_identity, "expected_identity")?;
    let ctx = context(seed, threads, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(consume_portable_bridge_bundle_recomputation(artifact, expected_bundle, expected, &ctx)
            .map_err(Failure::Bundle))
    })?;
    settle(outcome.map(|receipt| receipt_json(&receipt).to_string()), &ctx)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BridgePremiseAssessmentHandle>()?;
    module.add_class::<BridgePremiseAssessmentResultHandle>()?;
    module.add_class::<BridgeTransportDecisionHandle>()?;
    module.add_class::<BridgeTransportDecisionResultHandle>()?;
    module.add_class::<BridgeBundleRecomputationHandle>()?;
    module.add_class::<BridgeBundleRecomputationResultHandle>()?;
    module.add_function(wrap_pyfunction!(bridge_premise_assessment_compile, module)?)?;
    module.add_function(wrap_pyfunction!(bridge_premise_assessment_consume, module)?)?;
    module.add_function(wrap_pyfunction!(bridge_transport_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(bridge_transport_decision_consume, module)?)?;
    module.add_function(wrap_pyfunction!(bridge_bundle_recomputation_compile, module)?)?;
    module.add_function(wrap_pyfunction!(bridge_bundle_recomputation_consume, module)?)?;
    Ok(())
}

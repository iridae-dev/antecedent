//! Bounded Python bridge for the 2.3 composition boundary (C1), the decision
//! adapters over real 2.2 scenario claims (B) and the X6 per-proposal receipt.
//!
//! Python builds declarations and hands over library-produced objects; every
//! identity, trust decision, support decision and refusal rule is Rust's. A
//! refusal comes back as structured JSON for the Python layer to raise as its
//! own typed exception.
//!
//! Trust is never asserted from Python. A [`DecisionInput`] built from artifact
//! metadata alone is `Unverified`: the only evidence Python can state is `none`
//! or an external attestation, and the native execution record or the
//! exact-request verification receipt is minted by the library object that
//! produced the input (see [`PyTrustEvidence::native`] and
//! [`PyTrustEvidence::verified`]), never by a caller.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent::analysis::decision_claims::{
    ClaimError, OutcomeBinding, ScenarioClaimBinding, atom_leaders, cpdag_claims, scenario_claims,
};
use antecedent_core::{
    DistributionAvailability, EvidenceKind, EvidenceRegime, ExternalCapability, ExternalRefusal,
    ExternalTrustState, RegimeId, RegimeKind, ScientificQuantity, SupportStatus, VariableId,
};
use antecedent_design::composition_boundary::{
    ActionDisposition, ActionStatus, AtomCombination, AtomPlan, BoundaryError, CalibrationStatus,
    CompositionOperation, CoordinateIssue, DecisionInput, DependenceRoute, EvidenceRelation,
    InputProvenance, InputSource, NativeExecutionRecord, PairRelation, ProviderKind, ReceiptRef,
    RouteKind, ScalarClaim, SupportPolicy, SupportedDecision, SupportedVerdict,
    TrustEvidence as DesignTrust, TrustRequirement, UnsupportedActionPolicy,
    check_atom_combination, check_composition, check_paired_draws, evaluate_functional_on_input,
    evaluate_with_support,
};
use antecedent_design::decision_adapters::evaluate_adapted;
use antecedent_design::decision_artifact::{
    MAX_DECISION_ARTIFACT_BYTES, contract_from_json_refusal,
};
use antecedent_design::decision_contract::{DecisionFunctional, StructuralPolicy, Tail};
use antecedent_design::decision_eval::{ActionOutcome, MeanSource, Verdict};
use antecedent_design::decision_robust_artifact::admissible_contract_from_json_refusal;
use antecedent_design::decision_robustness::{AtomSupport, InputSupport, RobustVerdict};
use antecedent_design::decision_structural::{
    AtomEvidence, AtomStatus, StructuralAtom, StructuralDecisionResult, StructuralVerdict,
};
use antecedent_design::design_ranking_artifact::DesignRankingArtifactWire;
use antecedent_design::proposal_receipt::{
    Arrival, ArrivalVerdict, ArrivedEvidence, BaseFailureBinding, ChangedPremise, CostBinding,
    DecisionReceipt, HypotheticalBinding, LineageBinding, ObservedLaw, ProposalBundle,
    ProposalReceipt, ProposalReceiptError, ValuationBinding, on_arrival, verify as verify_bundle,
};
use antecedent_design::repair_artifact::RepairReportArtifact;
use antecedent_io::distribution_artifact::{DistributionCalibration, DistributionTrust};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cpdag_scenario_api::CpdagScenarioRun;
use crate::distribution_api::PyJointDistributionArtifact;
use crate::repair_api::{REPAIR_PREFIX, RepairContractStage, parse_candidates};
use crate::transport_common::{execution_context, resolve, unframe_named_artifact};
use crate::transport_scenario_api::PreparedTransportScenariosStage;
use crate::transport_z_api::intervention_assignments;
use crate::{CausalSerializationError, value_err};

/// A result JSON paired with a refusal JSON; exactly one is present.
type Pair = (Option<String>, Option<String>);

fn refusal_json(value: &ExternalRefusal) -> String {
    json!({
        "code": value.code,
        "stage": value.stage,
        "detail": value.detail,
        "offending": value.offending,
        "expected": value.expected,
        "supplied": value.supplied,
        "remedy": value.remedy,
    })
    .to_string()
}

fn boundary_refusal(error: &BoundaryError) -> String {
    refusal_json(&error.to_refusal())
}

fn check_size(json: &str) -> PyResult<()> {
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(PyValueError::new_err("composition declaration is too large"));
    }
    Ok(())
}

/// Refusal JSON for a malformed argument that never reached Rust validation.
fn invalid_json(detail: &str, offending: Option<String>) -> String {
    refusal_json(&ExternalRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        stage: "adapt",
        detail: detail.to_owned(),
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

fn parse_quantity(json: &str) -> PyResult<ScientificQuantity> {
    check_size(json)?;
    let wire: ScientificQuantityWire = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid coordinate: {e}")))?;
    ScientificQuantity::try_from(wire).map_err(|e| PyValueError::new_err(e.to_owned()))
}

fn parse_quantities(json: &str) -> PyResult<Vec<ScientificQuantity>> {
    check_size(json)?;
    let wires: Vec<ScientificQuantityWire> = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid coordinates: {e}")))?;
    wires
        .into_iter()
        .map(|wire| {
            ScientificQuantity::try_from(wire).map_err(|e| PyValueError::new_err(e.to_owned()))
        })
        .collect()
}

fn parse_support(name: &str) -> PyResult<SupportStatus> {
    SupportStatus::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown support status {name:?}")))
}

fn parse_requirement(name: &str) -> PyResult<TrustRequirement> {
    match name {
        "unrestricted" => Ok(TrustRequirement::Unrestricted),
        "native" => Ok(TrustRequirement::Native),
        "exact_request_verified" => Ok(TrustRequirement::ExactRequestVerified),
        other => Err(PyValueError::new_err(format!("unknown trust requirement {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Trust evidence and decision inputs.
// ---------------------------------------------------------------------------

/// What backs an input's trust. Python can state `none` or an external
/// attestation only; a native execution record or a verification receipt is
/// minted by the library object that produced the input.
#[pyclass(name = "TrustEvidence", skip_from_py_object)]
pub(crate) struct PyTrustEvidence {
    inner: DesignTrust,
}

impl PyTrustEvidence {
    /// Evidence that Antecedent itself executed the route behind an input; for
    /// library producers only. No Python constructor exists.
    #[allow(dead_code, reason = "minted by native producers as they adopt the boundary")]
    pub(crate) fn native(record: NativeExecutionRecord) -> Self {
        Self { inner: DesignTrust::NativeExecution(record) }
    }

    /// Evidence from an exact-request verification receipt the library issued;
    /// for library producers only. No Python constructor exists.
    #[allow(dead_code, reason = "minted by native producers as they adopt the boundary")]
    pub(crate) fn verified(state: ExternalTrustState) -> Self {
        Self { inner: DesignTrust::External(state) }
    }
}

#[pymethods]
impl PyTrustEvidence {
    /// No evidence beyond the artifact's own labels, which prove nothing.
    #[staticmethod]
    fn none() -> Self {
        Self { inner: DesignTrust::None }
    }

    /// An external party's attestation; never native, never verified.
    #[staticmethod]
    fn attested(attestor: &str) -> PyResult<Self> {
        if attestor.trim().is_empty() {
            return Err(PyValueError::new_err("an attestation names its attestor"));
        }
        Ok(Self {
            inner: DesignTrust::External(ExternalTrustState::ExternallyAttested {
                attestor: attestor.to_owned(),
            }),
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            DesignTrust::None => "none",
            DesignTrust::NativeExecution(_) => "native_execution",
            DesignTrust::External(ExternalTrustState::ExternallyAttested { .. }) => {
                "externally_attested"
            }
            DesignTrust::External(ExternalTrustState::ExactRequestVerified(_)) => {
                "exact_request_verified"
            }
            DesignTrust::External(ExternalTrustState::NativeLicensed) => "native_label",
        }
    }
}

/// One decision input: a source with its provenance and per-coordinate support.
#[pyclass(name = "DecisionInput", skip_from_py_object)]
pub(crate) struct PyDecisionInput {
    inner: DecisionInput,
}

#[pymethods]
impl PyDecisionInput {
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }

    /// Source kind, provenance and support map as JSON.
    fn summary_json(&self) -> String {
        input_json(&self.inner).to_string()
    }
}

const fn provider_kind_name(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Native => "native",
        ProviderKind::ExternalAttested => "external_attested",
        ProviderKind::ExternalExactRequestVerified => "external_exact_request_verified",
    }
}

const fn trust_name(trust: DistributionTrust) -> &'static str {
    match trust {
        DistributionTrust::NativeLicensed => "native_licensed",
        DistributionTrust::ExternalAttested => "external_attested",
        DistributionTrust::VerifiedExtension => "verified_extension",
        DistributionTrust::Unverified => "unverified",
    }
}

const fn calibration_name(status: CalibrationStatus) -> &'static str {
    match status {
        CalibrationStatus::Exact => "exact",
        CalibrationStatus::PointOnly => "point_only",
        CalibrationStatus::Measured => "measured",
        CalibrationStatus::Unmeasured => "unmeasured",
        CalibrationStatus::ExactClaimUnverified => "exact_claim_unverified",
    }
}

const fn capability_name(capability: ExternalCapability) -> &'static str {
    match capability {
        ExternalCapability::Sample => "sample",
        ExternalCapability::Cdf => "cdf",
        ExternalCapability::Quantile => "quantile",
        ExternalCapability::LogProbability => "log_probability",
        ExternalCapability::Mean => "mean",
        ExternalCapability::Covariance => "covariance",
        ExternalCapability::Conditional => "conditional",
        ExternalCapability::Intervention => "intervention",
        ExternalCapability::PosteriorPredictive => "posterior_predictive",
        ExternalCapability::Update => "update",
        ExternalCapability::Factor => "factor",
        ExternalCapability::EvaluateUtility => "evaluate_utility",
    }
}

fn receipt_json(receipt: Option<&ReceiptRef>) -> Value {
    match receipt {
        None => Value::Null,
        Some(ReceiptRef::NativeExecution { execution_id }) => {
            json!({"kind": "native_execution", "execution_id": execution_id})
        }
        Some(ReceiptRef::Attestation { attestor }) => {
            json!({"kind": "attestation", "attestor": attestor})
        }
        Some(ReceiptRef::Verification { object_id, request_id }) => {
            json!({"kind": "verification", "object_id": object_id, "request_id": request_id})
        }
    }
}

fn provenance_json(provenance: &InputProvenance) -> Value {
    json!({
        "provider_kind": provider_kind_name(provenance.provider_kind),
        "trust": trust_name(provenance.trust),
        "receipt": receipt_json(provenance.receipt.as_ref()),
        "calibration": calibration_name(provenance.calibration),
        "capabilities": provenance
            .capabilities
            .iter()
            .map(|c| capability_name(*c))
            .collect::<Vec<_>>(),
        "provider_id": provenance.provider_id,
        "snapshot_id": provenance.snapshot_id,
        "lineage_digest": provenance.lineage_digest,
    })
}

fn quantity_json(quantity: &ScientificQuantity) -> Value {
    serde_json::to_value(ScientificQuantityWire::from(quantity)).unwrap_or(Value::Null)
}

fn input_json(input: &DecisionInput) -> Value {
    let source = match input.source() {
        InputSource::Mean(_) => "mean",
        InputSource::JointLaw(_) => "joint_law",
        InputSource::Scalar(_) => "scalar",
    };
    let support: Vec<Value> = input
        .support()
        .entries()
        .iter()
        .map(|(coordinate, status)| {
            json!({"coordinate": quantity_json(coordinate), "status": status.as_str()})
        })
        .collect();
    json!({
        "id": input.id(),
        "source": source,
        "provenance": provenance_json(input.provenance()),
        "support": support,
    })
}

fn input_pair(
    result: Result<DecisionInput, BoundaryError>,
) -> (Option<PyDecisionInput>, Option<String>) {
    match result {
        Ok(inner) => (Some(PyDecisionInput { inner }), None),
        Err(error) => (None, Some(boundary_refusal(&error))),
    }
}

/// An input from an aligned joint-law artifact. The artifact's own `trust` and
/// `calibration` labels are claims; the provider kind comes from `evidence` only.
#[pyfunction]
fn composition_input_from_distribution(
    input_id: &str,
    artifact: &PyJointDistributionArtifact,
    evidence: &PyTrustEvidence,
    requirement: &str,
) -> PyResult<(Option<PyDecisionInput>, Option<String>)> {
    let requirement = parse_requirement(requirement)?;
    Ok(input_pair(DecisionInput::from_distribution_artifact(
        input_id,
        artifact.inner(),
        &evidence.inner,
        requirement,
    )))
}

/// An input from a mean source: one mean per coordinate, one support status each.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn composition_input_from_means(
    input_id: &str,
    coordinates_json: &str,
    means: Vec<f64>,
    statuses: Vec<String>,
    provider_id: &str,
    snapshot_id: &str,
    causal_contract_id: &str,
    evidence: &PyTrustEvidence,
    requirement: &str,
) -> PyResult<(Option<PyDecisionInput>, Option<String>)> {
    let coordinates = parse_quantities(coordinates_json)?;
    let statuses = statuses.iter().map(|s| parse_support(s)).collect::<PyResult<Vec<_>>>()?;
    let requirement = parse_requirement(requirement)?;
    let source = MeanSource {
        coordinates,
        means,
        provider_id: provider_id.to_owned(),
        snapshot_id: snapshot_id.to_owned(),
        causal_contract_id: causal_contract_id.to_owned(),
        rng_id: "none:mean_grid".to_owned(),
    };
    Ok(input_pair(DecisionInput::from_mean_source(
        input_id,
        source,
        &statuses,
        &evidence.inner,
        requirement,
    )))
}

/// An input from one scalar claim.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn composition_input_from_scalar(
    input_id: &str,
    coordinate_json: &str,
    value: f64,
    status: &str,
    provider_id: &str,
    snapshot_id: &str,
    causal_contract_id: &str,
    evidence: &PyTrustEvidence,
    requirement: &str,
) -> PyResult<(Option<PyDecisionInput>, Option<String>)> {
    let coordinate = parse_quantity(coordinate_json)?;
    let status = parse_support(status)?;
    let requirement = parse_requirement(requirement)?;
    let claim = ScalarClaim {
        coordinate,
        value,
        provider_id: provider_id.to_owned(),
        snapshot_id: snapshot_id.to_owned(),
        causal_contract_id: causal_contract_id.to_owned(),
    };
    Ok(input_pair(DecisionInput::from_scalar_claim(
        input_id,
        claim,
        status,
        &evidence.inner,
        requirement,
    )))
}

// ---------------------------------------------------------------------------
// Support-aware evaluation.
// ---------------------------------------------------------------------------

fn outcome_json(outcome: &ActionOutcome) -> Value {
    json!({
        "id": outcome.id,
        "admissible": outcome.admissible,
        "exclusions": outcome
            .exclusions
            .iter()
            .map(|e| json!({
                "constraint_id": e.constraint_id,
                "probability": e.probability,
                "required": e.required,
            }))
            .collect::<Vec<_>>(),
        "expected_utility": outcome.expected_utility,
        "value": outcome.value,
        "standard_error": outcome.standard_error,
        "expected_regret": outcome.expected_regret,
        "max_regret": outcome.max_regret,
    })
}

fn disposition_json(disposition: &ActionDisposition) -> Value {
    let mut reasons: Vec<Value> = Vec::new();
    let mut unevaluated: Option<&'static str> = None;
    let status = match &disposition.status {
        ActionStatus::Evaluated => "evaluated",
        ActionStatus::Unsupported { reasons: blocking } => {
            for reason in blocking {
                let support = match reason.issue {
                    CoordinateIssue::BelowSupport(status) => Some(status.as_str()),
                    CoordinateIssue::NotInSource => None,
                };
                reasons.push(json!({
                    "input_id": reason.input_id,
                    "coordinate": reason.coordinate,
                    "issue": reason.issue.detail(),
                    "support": support,
                }));
            }
            "unsupported"
        }
        ActionStatus::Unevaluated { reason } => {
            unevaluated = Some(reason.detail());
            "unevaluated"
        }
    };
    json!({
        "id": disposition.id,
        "status": status,
        "input_id": disposition.input_id,
        "reasons": reasons,
        "unevaluated_reason": unevaluated,
    })
}

fn verdict_json(verdict: &SupportedVerdict) -> Value {
    match verdict {
        SupportedVerdict::Compared(Verdict::UniquelyOptimal(id)) => {
            json!({"kind": "uniquely_optimal", "actions": [id]})
        }
        SupportedVerdict::Compared(Verdict::Indistinguishable(ids)) => {
            json!({"kind": "indistinguishable", "actions": ids})
        }
        SupportedVerdict::Compared(Verdict::NoAdmissibleAction) => {
            json!({"kind": "no_admissible_action", "actions": []})
        }
        SupportedVerdict::OnlyOneEvaluated(id) => {
            json!({"kind": "only_one_evaluated", "actions": [id]})
        }
        SupportedVerdict::NoSupportedAction => {
            json!({"kind": "no_supported_action", "actions": []})
        }
    }
}

fn supported_json(decision: &SupportedDecision) -> Value {
    json!({
        "contract_identity": decision.contract_identity,
        "dispositions": decision.dispositions.iter().map(disposition_json).collect::<Vec<_>>(),
        "outcomes": decision.outcomes.iter().map(outcome_json).collect::<Vec<_>>(),
        "verdict": verdict_json(&decision.verdict),
        "evpi": decision.evpi,
        "sources": decision.results.len(),
        "assumptions": decision.assumptions,
    })
}

fn parse_unsupported(name: &str) -> PyResult<UnsupportedActionPolicy> {
    match name {
        "compare_supported" => Ok(UnsupportedActionPolicy::CompareSupported),
        "require_all" => Ok(UnsupportedActionPolicy::RequireAllActions),
        other => Err(PyValueError::new_err(format!("unknown unsupported-action policy {other:?}"))),
    }
}

/// Evaluate a decision contract over several inputs, deciding support per
/// action. Returns the dispositions, outcomes and verdict, or a refusal.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn composition_evaluate(
    contract_json: &str,
    inputs: Vec<PyRef<'_, PyDecisionInput>>,
    unsupported: &str,
    weakest_support: &str,
) -> PyResult<Pair> {
    check_size(contract_json)?;
    let policy = SupportPolicy {
        unsupported: parse_unsupported(unsupported)?,
        weakest_support: parse_support(weakest_support)?,
    };
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let owned: Vec<DecisionInput> = inputs.iter().map(|input| input.inner.clone()).collect();
    Ok(match evaluate_with_support(&contract, &owned, &policy) {
        Ok(decision) => (Some(supported_json(&decision).to_string()), None),
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionalWire {
    kind: String,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    p: Option<f64>,
    #[serde(default)]
    tail: Option<String>,
}

fn tail_of(name: Option<&str>) -> Option<Tail> {
    match name {
        Some("lower") => Some(Tail::Lower),
        Some("upper") => Some(Tail::Upper),
        _ => None,
    }
}

fn functional_from(wire: &FunctionalWire) -> Option<DecisionFunctional> {
    match wire.kind.as_str() {
        "expectation" => Some(DecisionFunctional::Expectation),
        "variance" => Some(DecisionFunctional::Variance),
        "expected_utility" => Some(DecisionFunctional::ExpectedUtility),
        "probability" => Some(DecisionFunctional::Probability {
            threshold: wire.threshold?,
            tail: tail_of(wire.tail.as_deref())?,
        }),
        "quantile" => Some(DecisionFunctional::Quantile { p: wire.p? }),
        "tail_expectation" => Some(DecisionFunctional::TailExpectation {
            p: wire.p?,
            tail: tail_of(wire.tail.as_deref())?,
        }),
        _ => None,
    }
}

/// One functional of one action's utility from one input. A mean or scalar
/// answers only the expectation of an affine utility over non-outcome inputs;
/// every other functional refuses, never approximated from a mean.
#[pyfunction]
fn composition_functional(
    contract_json: &str,
    action_id: &str,
    functional_json: &str,
    input: &PyDecisionInput,
    weakest_support: &str,
) -> PyResult<Pair> {
    check_size(contract_json)?;
    check_size(functional_json)?;
    let weakest = parse_support(weakest_support)?;
    let wire: FunctionalWire = serde_json::from_str(functional_json)
        .map_err(|e| PyValueError::new_err(format!("invalid functional: {e}")))?;
    let functional = functional_from(&wire)
        .ok_or_else(|| PyValueError::new_err("an unknown or incomplete functional"))?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    Ok(
        match evaluate_functional_on_input(&contract, action_id, functional, &input.inner, weakest)
        {
            Ok(value) => (
                Some(
                    json!({
                        "value": value.value,
                        "standard_error": value.standard_error,
                        "source_mode": format!("{:?}", value.source_mode),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(error) => (None, Some(boundary_refusal(&error))),
        },
    )
}

// ---------------------------------------------------------------------------
// Evidence relations, declared operations and atom combination.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationKindWire {
    kind: String,
    #[serde(default)]
    ids: Vec<String>,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteWire {
    kind: String,
    id: String,
    source_input: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationWire {
    left: String,
    right: String,
    relation: RelationKindWire,
    #[serde(default)]
    route: Option<RouteWire>,
}

fn relation_from(wire: RelationKindWire) -> Result<EvidenceRelation, BoundaryError> {
    let RelationKindWire { kind, ids, id } = wire;
    let single = |id: Option<String>| id.ok_or(BoundaryError::InvalidInput("a relation identity"));
    Ok(match kind.as_str() {
        "independent_sources" => EvidenceRelation::IndependentSources,
        "shared_data" => EvidenceRelation::SharedData { ids },
        "shared_prior" => EvidenceRelation::SharedPrior { id: single(id)? },
        "shared_fitted_model" => EvidenceRelation::SharedFittedModel { id: single(id)? },
        "unknown_dependence" => EvidenceRelation::UnknownDependence,
        _ => return Err(BoundaryError::InvalidInput("an evidence relation kind")),
    })
}

fn route_from(wire: RouteWire) -> Result<DependenceRoute, BoundaryError> {
    let kind = match wire.kind.as_str() {
        "covariance" => RouteKind::Covariance,
        "joint_law" => RouteKind::JointLaw,
        _ => return Err(BoundaryError::InvalidInput("a dependence route kind")),
    };
    Ok(DependenceRoute { kind, id: wire.id, source_input: wire.source_input })
}

fn relations_from(json: &str) -> Result<Vec<PairRelation>, BoundaryError> {
    let wires: Vec<RelationWire> = serde_json::from_str(json)
        .map_err(|_| BoundaryError::InvalidInput("a relation declaration"))?;
    wires
        .into_iter()
        .map(|wire| {
            Ok(PairRelation {
                left: wire.left,
                right: wire.right,
                relation: relation_from(wire.relation)?,
                route: wire.route.map(route_from).transpose()?,
            })
        })
        .collect()
}

fn operation_from(name: Option<&str>) -> Result<Option<CompositionOperation>, BoundaryError> {
    Ok(match name {
        None => None,
        Some("statistical_pooling") => Some(CompositionOperation::StatisticalPooling),
        Some("bayesian_borrowing") => Some(CompositionOperation::BayesianBorrowing),
        Some("causal_transport") => Some(CompositionOperation::CausalTransport),
        Some("evidence_reuse") => Some(CompositionOperation::EvidenceReuse),
        Some(_) => return Err(BoundaryError::InvalidInput("a composition operation")),
    })
}

const fn operation_name(operation: CompositionOperation) -> &'static str {
    match operation {
        CompositionOperation::StatisticalPooling => "statistical_pooling",
        CompositionOperation::BayesianBorrowing => "bayesian_borrowing",
        CompositionOperation::CausalTransport => "causal_transport",
        CompositionOperation::EvidenceReuse => "evidence_reuse",
    }
}

fn composition_receipt_json(
    receipt: &antecedent_design::composition_boundary::CompositionReceipt,
) -> String {
    json!({
        "operation": operation_name(receipt.operation),
        "independence_assumed": receipt.independence_assumed,
        "routes": receipt.routes,
        "shared_evidence": receipt.shared_evidence,
    })
    .to_string()
}

/// Check that inputs may be combined under one declared operation.
#[pyfunction]
#[pyo3(signature = (inputs, relations_json, operation=None))]
#[allow(clippy::needless_pass_by_value)]
fn composition_check(
    inputs: Vec<PyRef<'_, PyDecisionInput>>,
    relations_json: &str,
    operation: Option<&str>,
) -> PyResult<Pair> {
    check_size(relations_json)?;
    let owned: Vec<DecisionInput> = inputs.iter().map(|input| input.inner.clone()).collect();
    let checked = relations_from(relations_json).and_then(|relations| {
        let operation = operation_from(operation)?;
        check_composition(&owned, &relations, operation)
    });
    Ok(match checked {
        Ok(receipt) => (Some(composition_receipt_json(&receipt)), None),
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

/// Check that the draws of inputs may be paired row by row.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn composition_check_paired_draws(
    inputs: Vec<PyRef<'_, PyDecisionInput>>,
    relations_json: &str,
) -> PyResult<Pair> {
    check_size(relations_json)?;
    let owned: Vec<DecisionInput> = inputs.iter().map(|input| input.inner.clone()).collect();
    let checked =
        relations_from(relations_json).and_then(|relations| check_paired_draws(&owned, &relations));
    Ok(match checked {
        Ok(receipt) => (Some(composition_receipt_json(&receipt)), None),
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomWire {
    id: String,
    #[serde(default)]
    probability: Option<f64>,
}

/// Check that conflicting structural atoms may be combined as requested.
#[pyfunction]
fn composition_check_atoms(atoms_json: &str, combination: &str) -> PyResult<Pair> {
    check_size(atoms_json)?;
    let wires: Vec<AtomWire> = serde_json::from_str(atoms_json)
        .map_err(|e| PyValueError::new_err(format!("invalid atoms: {e}")))?;
    let combination = match combination {
        "report_each" => AtomCombination::ReportEach,
        "worst_case" => AtomCombination::WorstCase,
        "weighted_by_declared_probabilities" => AtomCombination::WeightedByDeclaredProbabilities,
        other => {
            return Err(PyValueError::new_err(format!("unknown atom combination {other:?}")));
        }
    };
    let atoms: Vec<StructuralAtom> = wires
        .into_iter()
        .map(|wire| StructuralAtom {
            id: wire.id,
            probability: wire.probability,
            evidence: AtomEvidence::Unidentified,
        })
        .collect();
    Ok(match check_atom_combination(&atoms, combination) {
        Ok(AtomPlan::ReportEach) => (Some(json!({"kind": "report_each"}).to_string()), None),
        Ok(AtomPlan::WorstCase) => (Some(json!({"kind": "worst_case"}).to_string()), None),
        Ok(AtomPlan::Weighted(weights)) => {
            (Some(json!({"kind": "weighted", "weights": weights}).to_string()), None)
        }
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

// ---------------------------------------------------------------------------
// Decisions over real 2.2 scenario and CPDAG-completion claims.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSupportWire {
    action_id: String,
    input: usize,
    status: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportWire {
    overall: String,
    #[serde(default)]
    per_input: Vec<InputSupportWire>,
}

fn status_of(name: &str) -> Result<SupportStatus, String> {
    SupportStatus::from_name(name)
        .ok_or_else(|| invalid_json("scenario_decision.unknown_support_status", Some(name.into())))
}

fn support_from(wire: SupportWire) -> Result<AtomSupport, String> {
    Ok(AtomSupport {
        overall: status_of(&wire.overall)?,
        per_input: wire
            .per_input
            .into_iter()
            .map(|p| {
                Ok(InputSupport {
                    action_id: p.action_id,
                    input: p.input,
                    status: status_of(&p.status)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeWire {
    outcome: String,
    quantity: ScientificQuantityWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioSupportWire {
    scenario: String,
    support: SupportWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    outcomes: Vec<OutcomeWire>,
    calibration: String,
    source_id: String,
    provider_id: String,
    causal_contract_id: String,
    premises_digest: String,
    data_digest: String,
    #[serde(default)]
    support: Vec<ScenarioSupportWire>,
    #[serde(default)]
    default_support: Option<SupportWire>,
}

fn binding_from(json: &str, names: &[String]) -> Result<ScenarioClaimBinding, String> {
    let wire: BindingWire = serde_json::from_str(json)
        .map_err(|e| invalid_json("scenario_decision.invalid_binding", Some(e.to_string())))?;
    let mut outcomes = Vec::with_capacity(wire.outcomes.len());
    for item in wire.outcomes {
        let unknown =
            || invalid_json("scenario_decision.unknown_outcome", Some(item.outcome.clone()));
        let position = names.iter().position(|name| *name == item.outcome).ok_or_else(unknown)?;
        let id = u32::try_from(position).map_err(|_| unknown())?;
        let quantity = ScientificQuantity::try_from(item.quantity)
            .map_err(|e| invalid_json("scenario_decision.invalid_quantity", Some(e.to_owned())))?;
        outcomes.push(OutcomeBinding { outcome: VariableId::from_raw(id), quantity });
    }
    let calibration = match wire.calibration.as_str() {
        "exact" => DistributionCalibration::Exact,
        "point_only" => DistributionCalibration::PointOnly,
        "measured" => DistributionCalibration::Measured,
        "unmeasured" => DistributionCalibration::Unmeasured,
        other => {
            return Err(invalid_json("scenario_decision.unknown_calibration", Some(other.into())));
        }
    };
    let mut binding = ScenarioClaimBinding::new(
        outcomes,
        &wire.source_id,
        &wire.provider_id,
        &wire.causal_contract_id,
        &wire.premises_digest,
        &wire.data_digest,
    );
    binding.calibration = calibration;
    if let Some(default) = wire.default_support {
        binding = binding.with_default_support(support_from(default)?);
    }
    for item in wire.support {
        binding = binding.with_support(&item.scenario, support_from(item.support)?);
    }
    Ok(binding)
}

fn policy_from(name: &str) -> Result<StructuralPolicy, String> {
    match name {
        "require_invariant_best_action" => Ok(StructuralPolicy::RequireInvariantBestAction),
        "maximin" => Ok(StructuralPolicy::Maximin),
        "bayes_over_structures" => Ok(StructuralPolicy::BayesOverStructures),
        "report_only" => Ok(StructuralPolicy::ReportOnly),
        other => Err(invalid_json("scenario_decision.unknown_policy", Some(other.into()))),
    }
}

const fn policy_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

fn claim_refusal(error: &ClaimError) -> String {
    refusal_json(&error.to_refusal())
}

/// Declared weights must be the scenario set's own: all of them, equal to the
/// report's. A CPDAG completion has none to declare: counts are never probabilities.
fn check_declared_weights(
    scenarios: Option<&[(String, Option<f64>)]>,
    weights: Option<&BTreeMap<String, f64>>,
) -> Result<(), String> {
    let Some(weights) = weights else {
        return Ok(());
    };
    let Some(scenarios) = scenarios else {
        return Err(claim_refusal(&ClaimError::UnsupportedClaimShape(
            "completion counts are never probabilities; weights cannot be declared for CPDAG completions",
        )));
    };
    if scenarios.iter().all(|(_, weight)| weight.is_none()) {
        return Err(claim_refusal(&ClaimError::UnsupportedClaimShape(
            "scenario weights are declared only through the scenario set",
        )));
    }
    let same_ids = scenarios.len() == weights.len()
        && scenarios.iter().all(|(id, _)| weights.contains_key(id));
    let same_values = scenarios.iter().all(|(id, declared)| {
        matches!((declared, weights.get(id)), (Some(a), Some(b)) if (a - b).abs() <= 1e-12)
    });
    if same_ids && same_values {
        Ok(())
    } else {
        Err(claim_refusal(&ClaimError::IdentityMismatch("weights".to_owned())))
    }
}

fn structural_verdict_json(verdict: &StructuralVerdict) -> Value {
    match verdict {
        StructuralVerdict::InvariantBest(id) => json!({"kind": "invariant_best", "action": id}),
        StructuralVerdict::NoInvariantBest(leaders) => {
            json!({"kind": "no_invariant_best", "leaders": leaders})
        }
        StructuralVerdict::WorstCaseChoice(id) => {
            json!({"kind": "worst_case_choice", "action": id})
        }
        StructuralVerdict::BayesChoice { action, evaluated_mass } => {
            json!({"kind": "bayes_choice", "action": action, "evaluated_mass": evaluated_mass})
        }
        StructuralVerdict::ReportOnly => json!({"kind": "report_only"}),
        StructuralVerdict::InsufficientScience(why) => {
            json!({"kind": "insufficient_science", "reason": why})
        }
        StructuralVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn structural_json(result: &StructuralDecisionResult) -> Value {
    let atoms: Vec<Value> = result
        .atoms
        .iter()
        .map(|atom| {
            let (status, reason, actions) = match &atom.status {
                AtomStatus::Evaluated(inner) => (
                    "evaluated",
                    None,
                    Some(
                        inner
                            .actions
                            .iter()
                            .map(|o| {
                                json!({"id": o.id, "admissible": o.admissible, "value": o.value})
                            })
                            .collect::<Vec<_>>(),
                    ),
                ),
                AtomStatus::Unidentified => ("unidentified", None, None),
                AtomStatus::Unevaluated(why) => ("unevaluated", Some(why.clone()), None),
            };
            json!({
                "id": atom.id,
                "probability": atom.probability,
                "status": status,
                "reason": reason,
                "actions": actions,
            })
        })
        .collect();
    let actions: Vec<Value> = result
        .actions
        .iter()
        .map(|a| {
            json!({
                "id": a.id,
                "per_atom": a.per_atom,
                "range": a.range,
                "weighted_value": a.weighted_value,
                "mass_where_best": a.mass_where_best,
                "excluded_in": a.excluded_in,
            })
        })
        .collect();
    json!({
        "contract_identity": result.contract_identity,
        "policy": policy_name(result.policy),
        "atoms": atoms,
        "actions": actions,
        "unidentified_mass": result.unidentified_mass,
        "unevaluated_mass": result.unevaluated_mass,
        "evaluated_mass": result.evaluated_mass,
        "verdict": structural_verdict_json(&result.verdict),
    })
}

fn robust_verdict_json(verdict: &RobustVerdict) -> Value {
    match verdict {
        RobustVerdict::StructurallyRobust(id) => {
            json!({"kind": "structurally_robust", "action": id})
        }
        RobustVerdict::SupportRobust(id) => json!({"kind": "support_robust", "action": id}),
        RobustVerdict::SupportDependent { supported_choice, unrestricted_choice } => json!({
            "kind": "support_dependent",
            "action": supported_choice,
            "unrestricted_choice": unrestricted_choice,
        }),
        RobustVerdict::GraphDependentChoice(leaders) => {
            json!({"kind": "graph_dependent_choice", "leaders": leaders})
        }
        RobustVerdict::UnsupportedExtrapolation => json!({"kind": "unsupported_extrapolation"}),
        RobustVerdict::InsufficientClaims(why) => {
            json!({"kind": "insufficient_claims", "reason": why})
        }
        RobustVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
        RobustVerdict::WorstCaseChoice(id) => json!({"kind": "worst_case_choice", "action": id}),
        RobustVerdict::BayesChoice { action, evaluated_mass } => {
            json!({"kind": "bayes_choice", "action": action, "evaluated_mass": evaluated_mass})
        }
        RobustVerdict::ReportOnly => json!({"kind": "report_only"}),
    }
}

/// Decide over the claims of a real 2.2 scenario or CPDAG-completion result.
///
/// `kind` is `transport` (a prepared scenario stage after `estimate()`) or
/// `cpdag` (a completion run). `weights_json`, when given, is the
/// `{scenario: weight}` map the caller expects the scenario set to have
/// declared; it is checked against the report and never applied to it.
/// Returns the atom table, masses, leaders and verdicts as JSON, or a refusal.
#[pyfunction]
#[pyo3(signature = (declaration_json, kind, result, binding_json, policy, weights_json=None, expected_causal=None))]
#[allow(clippy::too_many_lines)]
fn decide_scenario_claims(
    declaration_json: &str,
    kind: &str,
    result: &Bound<'_, PyAny>,
    binding_json: &str,
    policy: &str,
    weights_json: Option<&str>,
    expected_causal: Option<&str>,
) -> PyResult<Pair> {
    check_size(declaration_json)?;
    check_size(binding_json)?;
    if let Some(text) = weights_json {
        check_size(text)?;
    }
    let run = || -> Result<String, String> {
        let contract = admissible_contract_from_json_refusal(declaration_json)
            .map_err(|refusal| refusal_json(&refusal))?;
        let policy = policy_from(policy)?;
        let weights: Option<BTreeMap<String, f64>> = weights_json
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| invalid_json("scenario_decision.invalid_weights", Some(e.to_string())))?;
        let wrong =
            |slot: &str| invalid_json("scenario_decision.wrong_result_type", Some(slot.to_owned()));
        let claims = match kind {
            "transport" => {
                let stage = result
                    .extract::<PyRef<'_, PreparedTransportScenariosStage>>()
                    .map_err(|_| wrong("transport"))?;
                let (report, names) = stage.last_report().ok_or_else(|| {
                    invalid_json("scenario_decision.not_estimated", Some("estimate() first".into()))
                })?;
                let scenarios: Vec<(String, Option<f64>)> =
                    report.scenarios.iter().map(|s| (s.name.to_string(), s.weight)).collect();
                check_declared_weights(Some(scenarios.as_slice()), weights.as_ref())?;
                let binding = binding_from(binding_json, names)?;
                check_causal(&binding, expected_causal)?;
                scenario_claims(report, &binding, policy, None).map_err(|e| claim_refusal(&e))?
            }
            "cpdag" => {
                let completions =
                    result.extract::<PyRef<'_, CpdagScenarioRun>>().map_err(|_| wrong("cpdag"))?;
                let (report, names) = completions.report_and_names();
                check_declared_weights(None, weights.as_ref())?;
                let binding = binding_from(binding_json, names)?;
                check_causal(&binding, expected_causal)?;
                cpdag_claims(report, &binding, policy, None).map_err(|e| claim_refusal(&e))?
            }
            other => {
                return Err(invalid_json("scenario_decision.unknown_kind", Some(other.into())));
            }
        };
        let decision = evaluate_adapted(&contract, &claims.adapted)
            .map_err(|e| refusal_json(&e.to_refusal()))?;
        let leaders = atom_leaders(&decision);
        let atoms: Vec<Value> = claims
            .atoms
            .iter()
            .map(|a| {
                json!({
                    "id": a.id,
                    "status": a.status,
                    "weight": a.weight,
                    "support": a.support.as_str(),
                    "digest": a.digest,
                    "detail": a.detail,
                })
            })
            .collect();
        let masses: Vec<Value> = claims
            .masses
            .iter()
            .map(|m| json!({"status": m.status, "count": m.count, "mass": m.mass}))
            .collect();
        let scenarios: Vec<Value> = claims
            .source
            .scenarios
            .iter()
            .map(|s| json!({"id": s.id, "status": s.status, "digest": s.digest}))
            .collect();
        Ok(json!({
            "kind": claims.source.kind,
            "source": {
                "identity": claims.source.identity,
                "cpdag_identity": claims.source.cpdag_identity,
                "premises_digest": claims.source.premises_digest,
                "data_digest": claims.source.data_digest,
                "scenarios": scenarios,
            },
            "declared_weights": claims.declared_weights,
            "atoms": atoms,
            "masses": masses,
            "residual_mass": claims.residual_mass,
            "structural": structural_json(&decision.structural),
            "leaders": leaders,
            "robust": {
                "verdict": robust_verdict_json(&decision.robust.verdict),
                "unsupported_atoms": decision.robust.unsupported_atoms,
                "unsupported_mass": decision.robust.unsupported_mass,
            },
            "completion_counts": decision.completion_counts,
        })
        .to_string())
    };
    Ok(match run() {
        Ok(json) => (Some(json), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

/// The binding must answer the causal identification the caller expects.
fn check_causal(binding: &ScenarioClaimBinding, expected: Option<&str>) -> Result<(), String> {
    match expected {
        Some(identity) if identity != binding.causal_contract_id => {
            Err(claim_refusal(&ClaimError::IdentityMismatch("causal_contract_id".to_owned())))
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// X6 per-proposal receipt.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaseWire {
    family: String,
    contract: String,
    obligation_ids: Vec<String>,
    premises_digest: String,
    data_digest: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HypotheticalWire {
    classification: String,
    delta_digest: String,
    delta_regimes: u64,
    derivation_digest: Option<String>,
    derivation_verified: bool,
    addressed: Vec<String>,
    unmet: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CostWire {
    units: u64,
    unit_label: String,
    sample_budget: u64,
    sample_size: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LineageWire {
    snapshots: Vec<String>,
    source_digest: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValuationWire {
    basis: String,
    signal_request_fingerprint: String,
    signal_identity: String,
    evsi_bits: u64,
    net_value_bits: Option<u64>,
    rank: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionWire {
    contract_identity: String,
    utility_unit: String,
    action_ids_digest: String,
    ranking_identity: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    candidate_id: String,
    base_failure: BaseWire,
    hypothetical: HypotheticalWire,
    cost: CostWire,
    lineage: LineageWire,
    valuation: ValuationWire,
    decision: DecisionWire,
    repair_report_digest: String,
    ranking_digest: String,
    identity: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleWire {
    repair_report_digest: String,
    ranking_digest: String,
    decision_contract_identity: String,
    proposals: Vec<ReceiptWire>,
    identity: String,
}

fn receipt_wire(receipt: &ProposalReceipt) -> ReceiptWire {
    ReceiptWire {
        candidate_id: receipt.candidate_id.clone(),
        base_failure: BaseWire {
            family: receipt.base_failure.family.clone(),
            contract: receipt.base_failure.contract.clone(),
            obligation_ids: receipt.base_failure.obligation_ids.clone(),
            premises_digest: receipt.base_failure.premises_digest.clone(),
            data_digest: receipt.base_failure.data_digest.clone(),
        },
        hypothetical: HypotheticalWire {
            classification: receipt.hypothetical.classification.clone(),
            delta_digest: receipt.hypothetical.delta_digest.clone(),
            delta_regimes: receipt.hypothetical.delta_regimes,
            derivation_digest: receipt.hypothetical.derivation_digest.clone(),
            derivation_verified: receipt.hypothetical.derivation_verified,
            addressed: receipt.hypothetical.addressed.clone(),
            unmet: receipt.hypothetical.unmet.clone(),
        },
        cost: CostWire {
            units: receipt.cost.units,
            unit_label: receipt.cost.unit_label.clone(),
            sample_budget: receipt.cost.sample_budget,
            sample_size: receipt.cost.sample_size,
        },
        lineage: LineageWire {
            snapshots: receipt.lineage.snapshots.clone(),
            source_digest: receipt.lineage.source_digest.clone(),
        },
        valuation: ValuationWire {
            basis: receipt.valuation.basis.clone(),
            signal_request_fingerprint: receipt.valuation.signal_request_fingerprint.clone(),
            signal_identity: receipt.valuation.signal_identity.clone(),
            evsi_bits: receipt.valuation.evsi_bits,
            net_value_bits: receipt.valuation.net_value_bits,
            rank: receipt.valuation.rank,
        },
        decision: DecisionWire {
            contract_identity: receipt.decision.contract_identity.clone(),
            utility_unit: receipt.decision.utility_unit.clone(),
            action_ids_digest: receipt.decision.action_ids_digest.clone(),
            ranking_identity: receipt.decision.ranking_identity.clone(),
        },
        repair_report_digest: receipt.repair_report_digest.clone(),
        ranking_digest: receipt.ranking_digest.clone(),
        identity: receipt.identity.clone(),
    }
}

/// The stored fields exactly as given: a receipt read back is never re-sealed,
/// so an edited field is caught by `verify`, not hidden by it.
fn receipt_from_wire(wire: ReceiptWire) -> ProposalReceipt {
    ProposalReceipt {
        candidate_id: wire.candidate_id,
        base_failure: BaseFailureBinding {
            family: wire.base_failure.family,
            contract: wire.base_failure.contract,
            obligation_ids: wire.base_failure.obligation_ids,
            premises_digest: wire.base_failure.premises_digest,
            data_digest: wire.base_failure.data_digest,
        },
        hypothetical: HypotheticalBinding {
            classification: wire.hypothetical.classification,
            delta_digest: wire.hypothetical.delta_digest,
            delta_regimes: wire.hypothetical.delta_regimes,
            derivation_digest: wire.hypothetical.derivation_digest,
            derivation_verified: wire.hypothetical.derivation_verified,
            addressed: wire.hypothetical.addressed,
            unmet: wire.hypothetical.unmet,
        },
        cost: CostBinding {
            units: wire.cost.units,
            unit_label: wire.cost.unit_label,
            sample_budget: wire.cost.sample_budget,
            sample_size: wire.cost.sample_size,
        },
        lineage: LineageBinding {
            snapshots: wire.lineage.snapshots,
            source_digest: wire.lineage.source_digest,
        },
        valuation: ValuationBinding {
            basis: wire.valuation.basis,
            signal_request_fingerprint: wire.valuation.signal_request_fingerprint,
            signal_identity: wire.valuation.signal_identity,
            evsi_bits: wire.valuation.evsi_bits,
            net_value_bits: wire.valuation.net_value_bits,
            rank: wire.valuation.rank,
        },
        decision: DecisionReceipt {
            contract_identity: wire.decision.contract_identity,
            utility_unit: wire.decision.utility_unit,
            action_ids_digest: wire.decision.action_ids_digest,
            ranking_identity: wire.decision.ranking_identity,
        },
        repair_report_digest: wire.repair_report_digest,
        ranking_digest: wire.ranking_digest,
        identity: wire.identity,
    }
}

fn proposal_refusal_json(error: &ProposalReceiptError) -> String {
    json!({
        "code": error.code,
        "stage": "proposal",
        "detail": error.detail,
        "message": error.message,
    })
    .to_string()
}

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

/// Decode the two retained artifacts: the framed repair report and the design
/// ranking, as the Python repair and ranking modules export them.
fn decode_artifacts(
    repair: &[u8],
    ranking: &[u8],
) -> PyResult<(RepairReportArtifact, DesignRankingArtifactWire)> {
    let (_names, inner) = unframe_named_artifact(REPAIR_PREFIX, repair, "repair")?;
    let repair = RepairReportArtifact::from_bytes(&inner)
        .map_err(|e| crate::refusal(e.code, format!("{}: {}", e.detail, e.message)))?;
    let ranking = DesignRankingArtifactWire::from_bytes(ranking)
        .map_err(|e| serialization(format!("design ranking artifact: {e:?}")))?;
    Ok((repair, ranking))
}

/// A bundle of per-proposal receipts, each binding by identity and digest.
#[pyclass(name = "ProposalBundle", skip_from_py_object)]
pub(crate) struct PyProposalBundle {
    inner: ProposalBundle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegimeWire {
    population: String,
    #[serde(default)]
    interventions: Vec<String>,
    #[serde(default)]
    levels: BTreeMap<String, f64>,
    #[serde(default)]
    conditioned_on: Vec<String>,
    measured: Vec<String>,
    #[serde(default = "default_true")]
    joint: bool,
    #[serde(default = "default_available")]
    evidence_kind: String,
}

fn default_true() -> bool {
    true
}

fn default_available() -> String {
    "available".to_owned()
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ArrivedWire {
    ObservedLaw { population: String, measured: Vec<String>, joint: bool },
    Regimes { regimes: Vec<RegimeWire> },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArrivalWire {
    snapshot_id: String,
    sample_size: u64,
    evidence: ArrivedWire,
}

fn variables(names: &[String], wanted: &[String]) -> PyResult<Arc<[VariableId]>> {
    wanted.iter().map(|name| resolve(names, name)).collect::<PyResult<Vec<_>>>().map(Arc::from)
}

fn arrival_invalid(message: impl Into<String>) -> String {
    proposal_refusal_json(&ProposalReceiptError {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "proposal_receipt.arrival_invalid",
        message: message.into(),
    })
}

fn regime_from(
    names: &[String],
    index: usize,
    wire: RegimeWire,
) -> PyResult<Result<EvidenceRegime, String>> {
    let evidence_kind = match wire.evidence_kind.as_str() {
        "available" => EvidenceKind::Available,
        "manipulable" => EvidenceKind::Manipulable,
        "proposed" => EvidenceKind::Proposed,
        other => {
            return Ok(Err(arrival_invalid(format!("unknown evidence kind {other:?}"))));
        }
    };
    let interventions = variables(names, &wire.interventions)?;
    let measured = variables(names, &wire.measured)?;
    let kind =
        if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental };
    let distribution = if wire.joint {
        DistributionAvailability::Joint
    } else {
        DistributionAvailability::SeparateMarginals { variables: Arc::clone(&measured) }
    };
    let values = intervention_assignments(names, wire.levels)?;
    let id = RegimeId::from_raw(u32::try_from(index).unwrap_or(u32::MAX));
    let built = EvidenceRegime::try_new(
        id,
        kind,
        evidence_kind,
        interventions,
        values,
        measured,
        wire.population,
        distribution,
    );
    let mut regime = match built {
        Ok(regime) => regime,
        Err(error) => return Ok(Err(arrival_invalid(error.to_string()))),
    };
    regime.conditioned_on = variables(names, &wire.conditioned_on)?;
    Ok(Ok(regime))
}

fn arrival_from(names: &[String], json: &str) -> PyResult<Result<Arrival, String>> {
    check_size(json)?;
    let wire: ArrivalWire = serde_json::from_str(json)
        .map_err(|e| value_err(format!("invalid arrival declaration: {e}")))?;
    let evidence = match wire.evidence {
        ArrivedWire::ObservedLaw { population, measured, joint } => {
            ArrivedEvidence::ObservedLaw(ObservedLaw {
                population: Arc::from(population.as_str()),
                measured: variables(names, &measured)?.to_vec(),
                joint,
            })
        }
        ArrivedWire::Regimes { regimes } => {
            let mut built = Vec::with_capacity(regimes.len());
            for (index, regime) in regimes.into_iter().enumerate() {
                match regime_from(names, index, regime)? {
                    Ok(regime) => built.push(regime),
                    Err(refusal) => return Ok(Err(refusal)),
                }
            }
            ArrivedEvidence::CatalogDelta(built)
        }
    };
    Ok(Ok(Arrival { snapshot_id: wire.snapshot_id, sample_size: wire.sample_size, evidence }))
}

fn name_of(names: &[String], id: VariableId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("v{}", id.raw()))
}

fn arrival_verdict_json(verdict: &ArrivalVerdict, names: &[String]) -> Value {
    match verdict {
        ArrivalVerdict::Verified { checker, steps, exact, snapshot_id } => json!({
            "kind": "verified",
            "checker": checker,
            "steps": steps,
            "exact": exact,
            "snapshot_id": snapshot_id,
        }),
        ArrivalVerdict::StillInsufficient { reasons } => {
            json!({"kind": "still_insufficient", "reasons": reasons})
        }
        ArrivalVerdict::Invalidated { changed } => {
            let changed = match changed {
                ChangedPremise::Population { proposed, arrived } => {
                    json!({"kind": "population", "proposed": proposed, "arrived": arrived})
                }
                ChangedPremise::Regime { population, arrived_interventions } => json!({
                    "kind": "regime",
                    "population": population,
                    "arrived_interventions": arrived_interventions
                        .iter()
                        .map(|raw| name_of(names, VariableId::from_raw(*raw)))
                        .collect::<Vec<_>>(),
                }),
            };
            json!({"kind": "invalidated", "changed": changed})
        }
    }
}

#[pymethods]
impl PyProposalBundle {
    /// Read a bundle back from its JSON. The stored identities are kept as
    /// given; nothing is re-sealed.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        check_size(json)?;
        let wire: BundleWire = serde_json::from_str(json)
            .map_err(|e| value_err(format!("invalid proposal bundle: {e}")))?;
        Ok(Self {
            inner: ProposalBundle {
                repair_report_digest: wire.repair_report_digest,
                ranking_digest: wire.ranking_digest,
                decision_contract_identity: wire.decision_contract_identity,
                proposals: wire.proposals.into_iter().map(receipt_from_wire).collect(),
                identity: wire.identity,
            },
        })
    }

    /// The bundle and every receipt as JSON.
    fn to_json(&self) -> PyResult<String> {
        let wire = BundleWire {
            repair_report_digest: self.inner.repair_report_digest.clone(),
            ranking_digest: self.inner.ranking_digest.clone(),
            decision_contract_identity: self.inner.decision_contract_identity.clone(),
            proposals: self.inner.proposals.iter().map(receipt_wire).collect(),
            identity: self.inner.identity.clone(),
        };
        serde_json::to_string(&wire).map_err(serialization)
    }

    #[getter]
    fn identity(&self) -> &str {
        &self.inner.identity
    }

    /// Recompute every receipt from the two retained artifacts: `None` when the
    /// bundle is exactly what they imply, else the refusal. Only the artifacts'
    /// seals are checked; a caller consumes each artifact independently first.
    fn verify(&self, repair: &[u8], ranking: &[u8]) -> PyResult<Option<String>> {
        let (repair, ranking) = decode_artifacts(repair, ranking)?;
        Ok(verify_bundle(&self.inner, &repair, &ranking)
            .err()
            .map(|error| proposal_refusal_json(&error)))
    }

    /// Re-run the failed contract's own identification on evidence that
    /// actually arrived for one proposal. The hypothetical delta and derivation
    /// are never consulted as evidence.
    fn on_arrival(
        &self,
        contract: &RepairContractStage,
        candidate_json: &str,
        arrival_json: &str,
    ) -> PyResult<Pair> {
        check_size(candidate_json)?;
        let names = contract.variable_names();
        let mut candidates = parse_candidates(names, candidate_json)?;
        let (Some(candidate), true) = (candidates.pop(), candidates.is_empty()) else {
            return Err(value_err("on_arrival takes exactly one candidate"));
        };
        let arrival = match arrival_from(names, arrival_json)? {
            Ok(arrival) => arrival,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
        let id = candidate.semantic_id();
        let Some(receipt) = self.inner.proposal(&id) else {
            return Ok((
                None,
                Some(proposal_refusal_json(&ProposalReceiptError {
                    code: antecedent_core::reason_code!("design_signal_invalid"),
                    detail: "proposal_receipt.candidate_not_in_bundle",
                    message: format!("candidate {id} has no receipt in this bundle"),
                })),
            ));
        };
        let ctx = execution_context(0, None, None);
        Ok(match on_arrival(receipt, contract.family_ref(), &candidate, &arrival, &ctx) {
            Ok(verdict) => (Some(arrival_verdict_json(&verdict, names).to_string()), None),
            Err(error) => (None, Some(proposal_refusal_json(&error))),
        })
    }
}

/// Build the bundle of every ranked candidate from the exported repair report
/// and the exported design ranking.
#[pyfunction]
fn proposal_bundle_build(
    repair: &[u8],
    ranking: &[u8],
) -> PyResult<(Option<PyProposalBundle>, Option<String>)> {
    let (repair, ranking) = decode_artifacts(repair, ranking)?;
    Ok(match ProposalBundle::build(&repair, &ranking) {
        Ok(inner) => (Some(PyProposalBundle { inner }), None),
        Err(error) => (None, Some(proposal_refusal_json(&error))),
    })
}

/// Candidate semantic ids of a retained design ranking, sorted.
#[pyfunction]
fn proposal_ranked_candidates(ranking: &[u8]) -> PyResult<Vec<String>> {
    let wire = DesignRankingArtifactWire::from_bytes(ranking)
        .map_err(|e| serialization(format!("design ranking artifact: {e:?}")))?;
    let ids: BTreeSet<String> = wire.candidates.iter().map(|c| c.semantic_id.clone()).collect();
    Ok(ids.into_iter().collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrustEvidence>()?;
    m.add_class::<PyDecisionInput>()?;
    m.add_class::<PyProposalBundle>()?;
    m.add_function(wrap_pyfunction!(composition_input_from_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(composition_input_from_means, m)?)?;
    m.add_function(wrap_pyfunction!(composition_input_from_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(composition_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(composition_functional, m)?)?;
    m.add_function(wrap_pyfunction!(composition_check, m)?)?;
    m.add_function(wrap_pyfunction!(composition_check_paired_draws, m)?)?;
    m.add_function(wrap_pyfunction!(composition_check_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(decide_scenario_claims, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_bundle_build, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_ranked_candidates, m)?)
}

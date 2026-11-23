//! Bounded Python bridge for the composition boundary, the decision
//! adapters over real scenario claims and the per-proposal receipt.
//!
//! Python builds declarations and hands over library-produced objects; every
//! identity, trust decision, support decision and refusal rule is Rust's. A
//! refusal comes back as structured JSON for the Python layer to raise as its
//! own typed exception.
//!
//! Trust is never asserted from Python. A [`DecisionInput`] built from artifact
//! metadata alone is `Unverified`: the only evidence Python can state is `none`
//! or an external attestation. A bound external claim that
//! already retains an exact-request verification receipt is carried into
//! [`PyTrustEvidence::verified`] by [`composition_input_from_external_claim`];
//! a caller still cannot mint that receipt.

mod checks;
mod proposal;
mod scenario;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::bridge::{check_declared_size, external_refusal_json, serialization};
use antecedent::analysis::decision_claims::{
    ClaimError, IntervalReading, OutcomeBinding, ScenarioClaimBinding, ScenarioDecision,
    envelope_identified_utilities, evaluate_cpdag_decision, evaluate_scenario_decision,
    interval_identified_utilities, report_identified_utilities, weighted_identified_utilities,
};
use antecedent::analysis::design_ranking::DesignRankingArtifactWire;
use antecedent::analysis::inverse_query::{
    AtomEvidence, DecisionFunctional, MeanSource, StructuralAtom, StructuralPolicy, Tail,
};
use antecedent::analysis::repair::RepairReportArtifact;
use antecedent::contract::{
    DistributionMeaning, ExternalCapability, ExternalRefusal, ExternalScientificObject,
    ExternalTrustState, LawProviderContract, ProviderObjectIdentity, ProviderTrust,
    VerificationProbe, capability_from_name, probe_from_name, verify_external_object,
};
use antecedent::design::composition_boundary::{
    ActionDisposition, ActionStatus, AtomCombination, AtomPlan, BoundaryError, CalibrationStatus,
    CompositionOperation, CoordinateIssue, DecisionInput, DependenceRoute, EvidenceRelation,
    InputProvenance, InputSource, NativeExecutionRecord, PairRelation, ProviderKind, ReceiptRef,
    RouteKind, ScalarClaim, SupportPolicy, SupportedDecision, SupportedVerdict,
    TrustEvidence as DesignTrust, TrustRequirement, UnsupportedActionPolicy,
    check_atom_combination, check_composition, check_paired_draws, evaluate_functional_on_input,
    evaluate_with_support,
};
use antecedent::design::decision_adapters::IdentifiedUtility;
use antecedent::design::decision_artifact::{
    MAX_DECISION_ARTIFACT_BYTES, contract_from_json_refusal,
};
use antecedent::design::decision_eval::{ActionOutcome, MeanEstimate, Verdict};
use antecedent::design::decision_robust_artifact::admissible_contract_from_json_refusal;
use antecedent::design::decision_robustness::{AtomSupport, InputSupport, RobustVerdict};
use antecedent::design::decision_structural::{
    AtomStatus, StructuralDecisionResult, StructuralVerdict,
};
use antecedent::design::proposal_receipt::{
    Arrival, ArrivalVerdict, ArrivedEvidence, BaseFailureBinding, ChangedPremise, CostBinding,
    DecisionReceipt, HypotheticalBinding, LineageBinding, ObservedLaw, ProposalBundle,
    ProposalReceipt, ProposalReceiptError, ValuationBinding, on_arrival, verify as verify_bundle,
};
use antecedent::estimate::transport_scenarios::ScenarioSetReport;
use antecedent::io::distribution_artifact::{DistributionCalibration, DistributionTrust};
use antecedent::io::external_claim_artifact::ExternalClaimTrust;
use antecedent::io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent::prelude::VariableId;
use antecedent::query::{
    DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeKind, SupportStatus,
};
use antecedent::schema::RegimeId;
use antecedent::scientific::ScientificQuantity;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cpdag_scenario_api::CpdagScenarioRun;
use crate::detach_catch;
use crate::distribution_api::PyJointDistributionArtifact;
use crate::repair_api::{REPAIR_PREFIX, RepairContractStage, parse_candidates};
use crate::transport_common::{execution_context, resolve, unframe_named_artifact};
use crate::transport_scenario_api::PreparedTransportScenariosStage;
use crate::transport_z_api::intervention_assignments;
use crate::value_err;

/// A result JSON paired with a refusal JSON; exactly one is present.
type Pair = (Option<String>, Option<String>);

pub(crate) fn boundary_refusal(error: &BoundaryError) -> String {
    external_refusal_json(&error.to_refusal())
}

pub(crate) fn check_size(json: &str) -> PyResult<()> {
    check_declared_size(json, MAX_DECISION_ARTIFACT_BYTES, "composition")
}

/// Refusal JSON for a malformed argument that never reached Rust validation.
fn invalid_json(detail: &str, offending: Option<String>) -> String {
    external_refusal_json(&ExternalRefusal {
        code: antecedent::reason::reason_code!("invalid_argument"),
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
        .map_err(|e| crate::value_err(format!("invalid coordinate: {e}")))?;
    ScientificQuantity::try_from(wire).map_err(crate::value_err)
}

fn parse_quantities(json: &str) -> PyResult<Vec<ScientificQuantity>> {
    check_size(json)?;
    let wires: Vec<ScientificQuantityWire> = serde_json::from_str(json)
        .map_err(|e| crate::value_err(format!("invalid coordinates: {e}")))?;
    wires
        .into_iter()
        .map(|wire| ScientificQuantity::try_from(wire).map_err(crate::value_err))
        .collect()
}

pub(crate) fn parse_support(name: &str) -> PyResult<SupportStatus> {
    SupportStatus::from_name(name)
        .ok_or_else(|| crate::value_err(format!("unknown support status {name:?}")))
}

fn parse_requirement(name: &str) -> PyResult<TrustRequirement> {
    match name {
        "unrestricted" => Ok(TrustRequirement::Unrestricted),
        "native" => Ok(TrustRequirement::Native),
        "exact_request_verified" => Ok(TrustRequirement::ExactRequestVerified),
        other => Err(crate::value_err(format!("unknown trust requirement {other:?}"))),
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
    /// Evidence from an exact-request verification receipt the library issued.
    /// No Python constructor exists; [`composition_input_from_external_claim`] carries
    /// the receipt a bound claim already retained.
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

    /// Evidence from a Rust-issued provider verification receipt. The input's
    /// provider and snapshot must be ones the receipt verified, or it proves nothing.
    #[staticmethod]
    fn from_verification(receipt: &crate::provider_api::PyProviderVerificationReceipt) -> Self {
        Self::verified(ExternalTrustState::ExactRequestVerified(Box::new(receipt.inner.clone())))
    }

    /// An external party's attestation; never native, never verified.
    #[staticmethod]
    fn attested(attestor: &str) -> PyResult<Self> {
        if attestor.trim().is_empty() {
            return Err(crate::value_err("an attestation names its attestor"));
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
            DesignTrust::External(state) => match ProviderTrust::from_external_state(state) {
                ProviderTrust::NativeSupported => "native_label",
                other => other.as_wire(),
            },
        }
    }
}

/// One decision input: a source with its provenance and per-coordinate support.
#[pyclass(name = "DecisionInput", skip_from_py_object)]
pub(crate) struct PyDecisionInput {
    inner: DecisionInput,
    pub(crate) source_claim: Option<crate::program_claims_api::PyNativeResponseClaim>,
    external_source: Option<antecedent::analysis::source_evidence::SourceEvidence>,
}

impl PyDecisionInput {
    pub(crate) fn inner(&self) -> &DecisionInput {
        &self.inner
    }
}

#[pymethods]
impl PyDecisionInput {
    fn _retain_external_source(
        &mut self,
        claim: &crate::external_api::PyExternalClaimArtifact,
    ) -> PyResult<()> {
        let source = crate::source_evidence_api::PySourceEvidence::from_external(claim)?;
        let antecedent::design::composition_boundary::InputSource::Mean(means) =
            self.inner.source()
        else {
            return Err(crate::recalc_api::invalid(
                antecedent::reason::reason_details::source_evidence::POINT_BINDING_MISMATCH,
                "external original requires a mean input",
            ));
        };
        source
            .inner
            .require_mean_source(means)
            .map_err(|cause| crate::recalc_api::invalid(cause.detail, cause.to_string()))?;
        self.external_source = Some(source.inner);
        Ok(())
    }
    #[getter]
    fn source_evidence(&self) -> PyResult<Option<crate::source_evidence_api::PySourceEvidence>> {
        if let Some(inner) = &self.external_source {
            return Ok(Some(crate::source_evidence_api::PySourceEvidence {
                inner: inner.clone(),
                resolution: None,
            }));
        }
        self.source_claim
            .as_ref()
            .map(crate::source_evidence_api::PySourceEvidence::from_claim)
            .transpose()
    }
    #[getter]
    fn has_source_evidence(&self) -> bool {
        self.source_claim.is_some() || self.external_source.is_some()
    }
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
        ProviderKind::ExternalAttested => ProviderTrust::ExternallyAttested.as_wire(),
        ProviderKind::ExternalExactRequestVerified => ProviderTrust::ExactRequestVerified.as_wire(),
    }
}

const fn trust_name(trust: DistributionTrust) -> &'static str {
    match trust {
        DistributionTrust::NativeSupported => ProviderTrust::NativeSupported.as_wire(),
        DistributionTrust::ExternalAttested => ProviderTrust::ExternallyAttested.as_wire(),
        DistributionTrust::VerifiedExtension => ProviderTrust::ExactRequestVerified.as_wire(),
        DistributionTrust::Unverified => ProviderTrust::Unverified.as_wire(),
    }
}

const fn calibration_name(status: CalibrationStatus) -> &'static str {
    match status {
        CalibrationStatus::Exact => "exact",
        CalibrationStatus::PointOnly => "point_only",
        CalibrationStatus::Measured => "measured",
        CalibrationStatus::NotMeasured => "not_measured",
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
        Ok(inner) => {
            (Some(PyDecisionInput { inner, source_claim: None, external_source: None }), None)
        }
        Err(error) => (None, Some(boundary_refusal(&error))),
    }
}

/// Build native composition evidence only from retained checked execution authority.
#[pyfunction]
fn composition_input_from_native_claim(
    input_id: &str,
    claim: &crate::program_claims_api::PyNativeResponseClaim,
    contract_json: &str,
) -> PyResult<(Option<PyDecisionInput>, Option<String>)> {
    use antecedent::analysis::native_claims::NativeDecisionSource;
    use antecedent::io::external_binding_wire::RefusalWire;
    if contract_json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(crate::value_err("native composition contract exceeds bounds"));
    }
    let refused = |refusal| -> PyResult<(Option<PyDecisionInput>, Option<String>)> {
        Ok((None, Some(serde_json::to_string(&RefusalWire::from(refusal)).map_err(serialization)?)))
    };
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(error) => return refused(error),
    };
    let requirement = match contract.source_requirement() {
        Ok(Some(requirement)) => requirement,
        Ok(None) => {
            return Err(crate::value_err("native composition requires a source criterion"));
        }
        Err(error) => return refused(error.to_refusal()),
    };
    let input = match claim.claim.decision_source(&requirement) {
        Ok(input) => input,
        Err(error) => return refused(error),
    };
    let execution_id = antecedent::io::execution_digest(
        &antecedent::io::execution_identity_from_context(&claim.authority.context),
    )
    .map_err(serialization)?;
    let (provider_id, snapshot_id) = match &input.source {
        NativeDecisionSource::Mean(source) => {
            (source.provider_id.clone(), source.snapshot_id.clone())
        }
        NativeDecisionSource::JointLaw(artifact) => (
            artifact.metadata().identity.provider_id.clone(),
            artifact.metadata().identity.snapshot_id.clone(),
        ),
    };
    let evidence = DesignTrust::NativeExecution(NativeExecutionRecord {
        execution_id: execution_id.to_hex(),
        provider_id,
        snapshot_id,
    });
    let result = match input.source {
        NativeDecisionSource::Mean(source) => DecisionInput::from_mean_source(
            input_id,
            source,
            &input.point_status,
            &evidence,
            TrustRequirement::Native,
        ),
        NativeDecisionSource::JointLaw(artifact) => DecisionInput::from_distribution_artifact(
            input_id,
            &artifact,
            &evidence,
            TrustRequirement::Native,
        ),
    };
    let (mut input, refusal) = input_pair(result);
    if let Some(input) = &mut input {
        input.source_claim = Some(claim.clone());
    }
    Ok((input, refusal))
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
    let native_evidence = if matches!(evidence.inner, DesignTrust::None) {
        artifact
            .source_claim
            .as_ref()
            .map(|claim| {
                let execution = antecedent::io::execution_digest(
                    &antecedent::io::execution_identity_from_context(&claim.authority.context),
                )
                .map_err(serialization)?;
                Ok::<_, PyErr>(DesignTrust::NativeExecution(NativeExecutionRecord {
                    execution_id: execution.to_hex(),
                    provider_id: artifact.inner().metadata().identity.provider_id.clone(),
                    snapshot_id: artifact.inner().metadata().identity.snapshot_id.clone(),
                }))
            })
            .transpose()?
    } else {
        None
    };
    let (mut result, refusal) = input_pair(DecisionInput::from_distribution_artifact(
        input_id,
        artifact.inner(),
        native_evidence.as_ref().unwrap_or(&evidence.inner),
        requirement,
    ));
    if let Some(result) = &mut result {
        result.source_claim.clone_from(&artifact.source_claim);
    }
    Ok((result, refusal))
}

fn evidence_from_external_claim(
    claim: &crate::external_api::PyExternalClaimArtifact,
) -> Result<DesignTrust, String> {
    let identity = &claim.artifact.metadata().identity;
    match identity.trust {
        ExternalClaimTrust::ExternallyAttested => {
            Ok(DesignTrust::External(ExternalTrustState::ExternallyAttested {
                attestor: identity.provider_id.clone(),
            }))
        }
        ExternalClaimTrust::ExactRequestVerified => {
            let probes = identity.verification.as_ref().ok_or_else(|| {
                invalid_json(
                    antecedent::reason::reason_details::external_object_verification::MISSING_PROBE,
                    Some(identity.provider_id.clone()),
                )
            })?;
            let probes = probes
                .iter()
                .map(|probe| {
                    Ok(VerificationProbe {
                        kind: probe_from_name(&probe.kind).ok_or_else(|| {
                            invalid_json(
                                antecedent::reason::reason_details::external_object_verification::MISSING_PROBE,
                                Some(probe.kind.clone()),
                            )
                        })?,
                        observed: probe.observed,
                        expected: probe.expected,
                        tolerance: probe.tolerance,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let capabilities = identity
                .capabilities
                .iter()
                .map(|name| {
                    capability_from_name(name).ok_or_else(|| {
                        invalid_json(
                            antecedent::reason::reason_details::external_response_binding::MALFORMED_CAPABILITY,
                            Some(name.clone()),
                        )
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let quantities = identity
                .quantities
                .iter()
                .cloned()
                .map(ScientificQuantity::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| {
                    invalid_json(
                        antecedent::reason::reason_details::external_response_binding::MALFORMED_QUANTITY,
                        Some(error.to_owned()),
                    )
                })?;
            let meaning = serde_json::from_value::<DistributionMeaningWire>(
                serde_json::Value::String(identity.provider_meaning.clone()),
            )
            .map(DistributionMeaning::from)
            .map_err(|error| {
                invalid_json(
                    antecedent::reason::reason_details::external_response_binding::MALFORMED_MEANING,
                    Some(error.to_string()),
                )
            })?;
            let object = ExternalScientificObject::Law(LawProviderContract {
                identity: ProviderObjectIdentity {
                    provider_id: identity.provider_id.clone(),
                    object_id: identity.object_id.clone(),
                    version_id: identity.version_id.clone(),
                    snapshot_id: identity.snapshot_id.clone(),
                    request_id: identity.request_id.clone(),
                },
                quantities,
                meaning,
                capabilities,
            });
            let receipt = verify_external_object(&object, &probes)
                .map_err(|error| external_refusal_json(&error.to_refusal()))?;
            Ok(PyTrustEvidence::verified(ExternalTrustState::ExactRequestVerified(Box::new(
                receipt,
            )))
            .inner)
        }
    }
}

/// An input from a bound external claim. Attested claims stay attested. A claim
/// that retains a passing exact-request receipt is carried as that receipt.
#[pyfunction]
fn composition_input_from_external_claim(
    input_id: &str,
    claim: &crate::external_api::PyExternalClaimArtifact,
    requirement: &str,
) -> PyResult<(Option<PyDecisionInput>, Option<String>)> {
    let requirement = parse_requirement(requirement)?;
    let evidence = match evidence_from_external_claim(claim) {
        Ok(evidence) => evidence,
        Err(refusal) => return Ok((None, Some(refusal))),
    };
    let identity = &claim.artifact.metadata().identity;
    let coordinates = identity
        .quantities
        .iter()
        .cloned()
        .map(ScientificQuantity::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(crate::value_err)?;
    let statuses = identity
        .point_status
        .iter()
        .map(|status| parse_support(status))
        .collect::<PyResult<Vec<_>>>()?;
    let source = MeanSource {
        coordinates,
        means: claim.artifact.values().to_vec(),
        provider_id: identity.provider_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        causal_contract_id: identity.causal_contract_id.clone(),
        rng_id: "none:mean_grid".to_owned(),
        estimate: MeanEstimate::Frequentist,
    };
    Ok(input_pair(DecisionInput::from_mean_source(
        input_id,
        source,
        &statuses,
        &evidence,
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
    if means.len() != coordinates.len() {
        return Err(crate::value_err(format!(
            "{} means for {} coordinates; give one mean per coordinate",
            means.len(),
            coordinates.len()
        )));
    }
    if means.iter().any(|mean| !mean.is_finite()) {
        return Err(crate::value_err("every mean must be finite"));
    }
    let source = MeanSource {
        coordinates,
        means,
        provider_id: provider_id.to_owned(),
        snapshot_id: snapshot_id.to_owned(),
        causal_contract_id: causal_contract_id.to_owned(),
        rng_id: "none:mean_grid".to_owned(),
        estimate: MeanEstimate::Frequentist,
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

pub(crate) fn supported_json(decision: &SupportedDecision) -> Value {
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

pub(crate) fn parse_unsupported(name: &str) -> PyResult<UnsupportedActionPolicy> {
    match name {
        "compare_supported" => Ok(UnsupportedActionPolicy::CompareSupported),
        "require_all" => Ok(UnsupportedActionPolicy::RequireAllActions),
        other => Err(crate::value_err(format!("unknown unsupported-action policy {other:?}"))),
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
        Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
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

pub(crate) fn parsed_functional(json: &str) -> PyResult<DecisionFunctional> {
    check_size(json)?;
    let wire: FunctionalWire =
        serde_json::from_str(json).map_err(|error| crate::value_err(error.to_string()))?;
    functional_from(&wire).ok_or_else(|| crate::value_err("unknown functional"))
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
        .map_err(|e| crate::value_err(format!("invalid functional: {e}")))?;
    let functional = functional_from(&wire)
        .ok_or_else(|| crate::value_err("an unknown or incomplete functional"))?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrustEvidence>()?;
    m.add_class::<PyDecisionInput>()?;
    proposal::register_class(m)?;
    m.add_function(wrap_pyfunction!(composition_input_from_native_claim, m)?)?;
    m.add_function(wrap_pyfunction!(composition_input_from_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(composition_input_from_means, m)?)?;
    m.add_function(wrap_pyfunction!(composition_input_from_external_claim, m)?)?;
    m.add_function(wrap_pyfunction!(composition_input_from_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(composition_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(composition_functional, m)?)?;
    crate::functional_source_api::register(m)?;
    checks::register(m)?;
    scenario::register(m)?;
    proposal::register_functions(m)
}

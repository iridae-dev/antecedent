//! Proposal bundle, receipts and arrival decisions.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, Arrival, ArrivalVerdict, ArrivedEvidence, BTreeMap, BTreeSet, BaseFailureBinding, Bound,
    ChangedPremise, CostBinding, DecisionReceipt, Deserialize, DesignRankingArtifactWire,
    DistributionAvailability, EvidenceKind, EvidenceRegime, HypotheticalBinding, LineageBinding,
    ObservedLaw, Pair, ProposalBundle, ProposalReceipt, ProposalReceiptError, PyModule,
    PyModuleMethods, PyResult, REPAIR_PREFIX, RegimeId, RegimeKind, RepairContractStage,
    RepairReportArtifact, Serialize, ValuationBinding, Value, VariableId, check_size,
    execution_context, intervention_assignments, json, on_arrival, parse_candidates, pyclass,
    pyfunction, pymethods, resolve, serialization, unframe_named_artifact, value_err,
    verify_bundle, wrap_pyfunction,
};

// ---------------------------------------------------------------------------
// Per-proposal receipt.
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
        code: antecedent::reason::reason_code!("invalid_argument"),
        detail: antecedent::reason::reason_details::proposal_receipt::ARRIVAL_INVALID,
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
                    code: antecedent::reason::reason_code!("design_signal_invalid"),
                    detail:
                        antecedent::reason::reason_details::proposal_receipt::CANDIDATE_NOT_IN_BUNDLE,
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

pub(super) fn register_class(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProposalBundle>()
}

pub(super) fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(proposal_bundle_build, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_ranked_candidates, m)?)
}

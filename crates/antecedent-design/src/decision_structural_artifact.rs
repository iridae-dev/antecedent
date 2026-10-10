//! Portable structural decision result artifact.
//!
//! The artifact stores a [`StructuralDecisionResult`] bound to the decision
//! contract's identity and, for every evaluated structure, the BLAKE3 digest of
//! its draws. It retains each atom's id, probability and status (evaluated,
//! unidentified, or unevaluated with its reason), the declared structural
//! policy, the per-action values across structures, the unidentified,
//! unevaluated and evaluated masses, and the verdict. It loads only under the
//! contract identity and atom digests the consumer retained independently of
//! the bytes, and it can be replayed: the structural evaluation is recomputed
//! and must reproduce the stored wire exactly.

use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::error::IoError;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

use crate::decision_artifact::{MAX_DECISION_ARTIFACT_BYTES, source_digest};
use crate::decision_contract::{DecisionContract, DecisionCriterion, StructuralPolicy};
use crate::decision_eval::{
    ActionOutcome, ConstraintExclusion, DecisionResult, SourceReceipt, Verdict,
};
use crate::decision_structural::{
    ActionStructure, AtomEvidence, AtomStatus, AtomSummary, StructuralAtom,
    StructuralDecisionResult, StructuralVerdict, evaluate_structural,
};

const RESULT_KIND: &str = "decision_structural_result_v1";
const BODY_SECTION: &str = "decision_structural_body";

/// One retained atom identity with the digest of its draws, when evaluated.
type AtomDigest = (String, Option<String>);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CriterionWire {
    PosteriorExpectedUtility,
    PosteriorExpectedLoss,
    ThresholdProbability { threshold: f64 },
    Quantile { p: f64 },
    MinimaxOverIdentifiedSet,
    MaximinOverStructures,
    Regret,
    ExpectedRegret,
}

impl From<DecisionCriterion> for CriterionWire {
    fn from(c: DecisionCriterion) -> Self {
        match c {
            DecisionCriterion::PosteriorExpectedUtility => Self::PosteriorExpectedUtility,
            DecisionCriterion::PosteriorExpectedLoss => Self::PosteriorExpectedLoss,
            DecisionCriterion::ThresholdProbability { threshold } => {
                Self::ThresholdProbability { threshold }
            }
            DecisionCriterion::Quantile { p } => Self::Quantile { p },
            DecisionCriterion::MinimaxOverIdentifiedSet => Self::MinimaxOverIdentifiedSet,
            DecisionCriterion::MaximinOverStructures => Self::MaximinOverStructures,
            DecisionCriterion::Regret => Self::Regret,
            DecisionCriterion::ExpectedRegret => Self::ExpectedRegret,
        }
    }
}

impl From<CriterionWire> for DecisionCriterion {
    fn from(c: CriterionWire) -> Self {
        match c {
            CriterionWire::PosteriorExpectedUtility => Self::PosteriorExpectedUtility,
            CriterionWire::PosteriorExpectedLoss => Self::PosteriorExpectedLoss,
            CriterionWire::ThresholdProbability { threshold } => {
                Self::ThresholdProbability { threshold }
            }
            CriterionWire::Quantile { p } => Self::Quantile { p },
            CriterionWire::MinimaxOverIdentifiedSet => Self::MinimaxOverIdentifiedSet,
            CriterionWire::MaximinOverStructures => Self::MaximinOverStructures,
            CriterionWire::Regret => Self::Regret,
            CriterionWire::ExpectedRegret => Self::ExpectedRegret,
        }
    }
}

fn policy_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

fn policy_from(name: &str) -> Result<StructuralPolicy, IoError> {
    Ok(match name {
        "require_invariant_best_action" => StructuralPolicy::RequireInvariantBestAction,
        "maximin" => StructuralPolicy::Maximin,
        "bayes_over_structures" => StructuralPolicy::BayesOverStructures,
        "report_only" => StructuralPolicy::ReportOnly,
        _ => return Err(IoError::Convert(format!("unknown structural policy `{name}`"))),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExclusionWire {
    constraint_id: String,
    probability: f64,
    required: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeWire {
    id: String,
    admissible: bool,
    exclusions: Vec<ExclusionWire>,
    expected_utility: f64,
    value: f64,
    standard_error: Option<f64>,
    expected_regret: Option<f64>,
    max_regret: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AtomVerdictWire {
    UniquelyOptimal(String),
    Indistinguishable(Vec<String>),
    NoAdmissibleAction,
}

/// The per-structure decision of an evaluated atom.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InnerWire {
    contract_identity: String,
    criterion: CriterionWire,
    actions: Vec<OutcomeWire>,
    verdict: AtomVerdictWire,
    evpi: Option<f64>,
    n_draws: u64,
    effective_draws: f64,
    provider_id: String,
    snapshot_id: String,
    rng_id: String,
    causal_contract_id: String,
    assumptions: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StatusWire {
    Evaluated(Box<InnerWire>),
    Unidentified,
    Unevaluated(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomWire {
    id: String,
    probability: Option<f64>,
    status: StatusWire,
    source_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionStructureWire {
    id: String,
    per_atom: Vec<Option<f64>>,
    range: Option<(f64, f64)>,
    weighted_value: Option<f64>,
    mass_where_best: Option<f64>,
    excluded_in: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerdictWire {
    InvariantBest(String),
    NoInvariantBest(Vec<(String, Vec<String>)>),
    WorstCaseChoice(String),
    BayesChoice { action: String, evaluated_mass: f64 },
    ReportOnly,
    InsufficientScience(String),
    NoAdmissibleAction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralBody {
    version: u16,
    contract_identity: String,
    policy: String,
    atoms: Vec<AtomWire>,
    actions: Vec<ActionStructureWire>,
    unidentified_mass: Option<f64>,
    unevaluated_mass: Option<f64>,
    evaluated_mass: Option<f64>,
    verdict: VerdictWire,
}

fn inner_to_wire(result: &DecisionResult) -> InnerWire {
    InnerWire {
        contract_identity: result.contract_identity.clone(),
        criterion: result.criterion.into(),
        actions: result
            .actions
            .iter()
            .map(|a| OutcomeWire {
                id: a.id.clone(),
                admissible: a.admissible,
                exclusions: a
                    .exclusions
                    .iter()
                    .map(|e| ExclusionWire {
                        constraint_id: e.constraint_id.clone(),
                        probability: e.probability,
                        required: e.required,
                    })
                    .collect(),
                expected_utility: a.expected_utility,
                value: a.value,
                standard_error: a.standard_error,
                expected_regret: a.expected_regret,
                max_regret: a.max_regret,
            })
            .collect(),
        verdict: match &result.verdict {
            Verdict::UniquelyOptimal(id) => AtomVerdictWire::UniquelyOptimal(id.clone()),
            Verdict::Indistinguishable(ids) => AtomVerdictWire::Indistinguishable(ids.clone()),
            Verdict::NoAdmissibleAction => AtomVerdictWire::NoAdmissibleAction,
        },
        evpi: result.evpi,
        n_draws: u64::try_from(result.n_draws).unwrap_or(u64::MAX),
        effective_draws: result.effective_draws,
        provider_id: result.source.provider_id.clone(),
        snapshot_id: result.source.snapshot_id.clone(),
        rng_id: result.source.rng_id.clone(),
        causal_contract_id: result.source.causal_contract_id.clone(),
        assumptions: result.assumptions.clone(),
    }
}

fn inner_from_wire(wire: InnerWire) -> Result<DecisionResult, IoError> {
    Ok(DecisionResult {
        contract_identity: wire.contract_identity,
        criterion: wire.criterion.into(),
        actions: wire
            .actions
            .into_iter()
            .map(|a| ActionOutcome {
                id: a.id,
                admissible: a.admissible,
                exclusions: a
                    .exclusions
                    .into_iter()
                    .map(|e| ConstraintExclusion {
                        constraint_id: e.constraint_id,
                        probability: e.probability,
                        required: e.required,
                    })
                    .collect(),
                expected_utility: a.expected_utility,
                value: a.value,
                standard_error: a.standard_error,
                expected_regret: a.expected_regret,
                max_regret: a.max_regret,
            })
            .collect(),
        verdict: match wire.verdict {
            AtomVerdictWire::UniquelyOptimal(id) => Verdict::UniquelyOptimal(id),
            AtomVerdictWire::Indistinguishable(ids) => Verdict::Indistinguishable(ids),
            AtomVerdictWire::NoAdmissibleAction => Verdict::NoAdmissibleAction,
        },
        evpi: wire.evpi,
        n_draws: usize::try_from(wire.n_draws).map_err(|_| IoError::TooLarge)?,
        effective_draws: wire.effective_draws,
        source: SourceReceipt {
            provider_id: wire.provider_id,
            snapshot_id: wire.snapshot_id,
            rng_id: wire.rng_id,
            causal_contract_id: wire.causal_contract_id,
        },
        assumptions: wire.assumptions,
    })
}

fn atom_to_wire(summary: &AtomSummary, digest: Option<&String>) -> AtomWire {
    AtomWire {
        id: summary.id.clone(),
        probability: summary.probability,
        status: match &summary.status {
            AtomStatus::Evaluated(result) => StatusWire::Evaluated(Box::new(inner_to_wire(result))),
            AtomStatus::Unidentified => StatusWire::Unidentified,
            AtomStatus::Unevaluated(reason) => StatusWire::Unevaluated(reason.clone()),
        },
        source_digest: digest.cloned(),
    }
}

fn verdict_to_wire(verdict: &StructuralVerdict) -> VerdictWire {
    match verdict {
        StructuralVerdict::InvariantBest(id) => VerdictWire::InvariantBest(id.clone()),
        StructuralVerdict::NoInvariantBest(leaders) => {
            VerdictWire::NoInvariantBest(leaders.clone())
        }
        StructuralVerdict::WorstCaseChoice(id) => VerdictWire::WorstCaseChoice(id.clone()),
        StructuralVerdict::BayesChoice { action, evaluated_mass } => {
            VerdictWire::BayesChoice { action: action.clone(), evaluated_mass: *evaluated_mass }
        }
        StructuralVerdict::ReportOnly => VerdictWire::ReportOnly,
        StructuralVerdict::InsufficientScience(reason) => {
            VerdictWire::InsufficientScience(reason.clone())
        }
        StructuralVerdict::NoAdmissibleAction => VerdictWire::NoAdmissibleAction,
    }
}

fn verdict_from_wire(verdict: VerdictWire) -> StructuralVerdict {
    match verdict {
        VerdictWire::InvariantBest(id) => StructuralVerdict::InvariantBest(id),
        VerdictWire::NoInvariantBest(leaders) => StructuralVerdict::NoInvariantBest(leaders),
        VerdictWire::WorstCaseChoice(id) => StructuralVerdict::WorstCaseChoice(id),
        VerdictWire::BayesChoice { action, evaluated_mass } => {
            StructuralVerdict::BayesChoice { action, evaluated_mass }
        }
        VerdictWire::ReportOnly => StructuralVerdict::ReportOnly,
        VerdictWire::InsufficientScience(reason) => StructuralVerdict::InsufficientScience(reason),
        VerdictWire::NoAdmissibleAction => StructuralVerdict::NoAdmissibleAction,
    }
}

fn action_to_wire(action: &ActionStructure) -> ActionStructureWire {
    ActionStructureWire {
        id: action.id.clone(),
        per_atom: action.per_atom.clone(),
        range: action.range,
        weighted_value: action.weighted_value,
        mass_where_best: action.mass_where_best,
        excluded_in: action.excluded_in.clone(),
    }
}

fn action_from_wire(wire: ActionStructureWire) -> ActionStructure {
    ActionStructure {
        id: wire.id,
        per_atom: wire.per_atom,
        range: wire.range,
        weighted_value: wire.weighted_value,
        mass_where_best: wire.mass_where_best,
        excluded_in: wire.excluded_in,
    }
}

fn result_to_body(result: &StructuralDecisionResult, digests: &[AtomDigest]) -> StructuralBody {
    StructuralBody {
        version: 1,
        contract_identity: result.contract_identity.clone(),
        policy: policy_name(result.policy).to_owned(),
        atoms: result
            .atoms
            .iter()
            .enumerate()
            .map(|(k, s)| atom_to_wire(s, digests.get(k).and_then(|d| d.1.as_ref())))
            .collect(),
        actions: result.actions.iter().map(action_to_wire).collect(),
        unidentified_mass: result.unidentified_mass,
        unevaluated_mass: result.unevaluated_mass,
        evaluated_mass: result.evaluated_mass,
        verdict: verdict_to_wire(&result.verdict),
    }
}

fn atoms_from_wire(atoms: Vec<AtomWire>) -> Result<(Vec<AtomSummary>, Vec<AtomDigest>), IoError> {
    let mut summaries = Vec::with_capacity(atoms.len());
    let mut digests = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let status = match atom.status {
            StatusWire::Evaluated(inner) => {
                if atom.source_digest.is_none() {
                    return Err(refused("atom_digest", "an evaluated atom has no source digest"));
                }
                AtomStatus::Evaluated(Box::new(inner_from_wire(*inner)?))
            }
            StatusWire::Unidentified => AtomStatus::Unidentified,
            StatusWire::Unevaluated(reason) => AtomStatus::Unevaluated(reason),
        };
        digests.push((atom.id.clone(), atom.source_digest));
        summaries.push(AtomSummary { id: atom.id, probability: atom.probability, status });
    }
    Ok((summaries, digests))
}

fn body_to_result(
    body: StructuralBody,
) -> Result<(StructuralDecisionResult, Vec<AtomDigest>), IoError> {
    if body.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
    }
    let (atoms, digests) = atoms_from_wire(body.atoms)?;
    let result = StructuralDecisionResult {
        contract_identity: body.contract_identity,
        policy: policy_from(&body.policy)?,
        atoms,
        actions: body.actions.into_iter().map(action_from_wire).collect(),
        unidentified_mass: body.unidentified_mass,
        unevaluated_mass: body.unevaluated_mass,
        evaluated_mass: body.evaluated_mass,
        verdict: verdict_from_wire(body.verdict),
    };
    Ok((result, digests))
}

fn encode(artifact_id: &str, body: &[u8]) -> Result<Vec<u8>, IoError> {
    if artifact_id.trim().is_empty() {
        return Err(IoError::Convert("missing artifact id".into()));
    }
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: antecedent_io::migrate::STABLE_FORMAT,
            minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(RESULT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
            artifact_id: artifact_id.into(),
            sections: vec![section_descriptor(BODY_SECTION, "application/cbor", body)],
            provenance: ProvenanceWire { note: "decision_structural_result".into() },
        },
        sections: vec![SectionBytes::new(BODY_SECTION, body.to_vec())],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes)?;
    if bytes.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<StructuralBody, IoError> {
    if bytes.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(RESULT_KIND.into())
        || manifest.sections.len() != 1
        || manifest.sections[0].id != BODY_SECTION
    {
        return Err(IoError::Convert("unsupported decision artifact layout".into()));
    }
    if manifest.sections[0].uncompressed_size > MAX_DECISION_ARTIFACT_BYTES as u64 {
        return Err(IoError::TooLarge);
    }
    let section = reader.load_section(BODY_SECTION)?;
    from_cbor(section.as_bytes())
}

fn refused(slot: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!("decision_artifact.{slot}: {message}"),
    }
}

/// Each atom's retained id and, for an evaluated atom, the digest of its draws.
fn digests_of(atoms: &[StructuralAtom]) -> Vec<AtomDigest> {
    atoms
        .iter()
        .map(|a| {
            let digest = match &a.evidence {
                AtomEvidence::Evaluated(source) => Some(source_digest(source)),
                AtomEvidence::Unidentified | AtomEvidence::Unevaluated(_) => None,
            };
            (a.id.clone(), digest)
        })
        .collect()
}

/// A structural decision result bound to its contract and atom evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralResultArtifact {
    result: StructuralDecisionResult,
    atom_digests: Vec<AtomDigest>,
}

impl StructuralResultArtifact {
    /// Bind a result to the atoms it was computed from; the digest of each
    /// evaluated atom's draws is taken from its evidence.
    #[must_use]
    pub fn new(result: StructuralDecisionResult, atoms: &[StructuralAtom]) -> Self {
        Self { result, atom_digests: digests_of(atoms) }
    }

    /// The stored result.
    #[must_use]
    pub fn result(&self) -> &StructuralDecisionResult {
        &self.result
    }

    /// Atom ids with the digest of each evaluated atom's draws, in atom order.
    #[must_use]
    pub fn atom_digests(&self) -> &[(String, Option<String>)] {
        &self.atom_digests
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload or a blank artifact id refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(&result_to_body(&self.result, &self.atom_digests))?;
        encode(artifact_id, &body)
    }

    /// Load under the consumer's retained contract identity and atom digests.
    ///
    /// # Errors
    /// A corrupt or oversized artifact, or one bound to a different contract,
    /// different atom ids or different atom draws, refuses.
    pub fn from_bytes(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_atom_digests: &[(String, Option<String>)],
    ) -> Result<Self, IoError> {
        let body = decode(bytes)?;
        if body.contract_identity != expected_contract_identity {
            return Err(refused(
                "contract_expected",
                "result was computed under a different decision contract",
            ));
        }
        let (result, atom_digests) = body_to_result(body)?;
        if atom_digests.as_slice() != expected_atom_digests {
            return Err(refused(
                "atoms_expected",
                "result was computed from different structures or draws",
            ));
        }
        Ok(Self { result, atom_digests })
    }

    /// Recompute the structural evaluation from `contract` and `atoms` and
    /// require the stored result to match exactly.
    ///
    /// # Errors
    /// A different contract or atoms, a refused evaluation or any differing
    /// number or verdict refuses.
    pub fn replay(
        &self,
        contract: &DecisionContract,
        atoms: &[StructuralAtom],
    ) -> Result<(), IoError> {
        let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        if identity != self.result.contract_identity {
            return Err(refused("replay_contract", "contract differs from the stored result's"));
        }
        let digests = digests_of(atoms);
        if digests != self.atom_digests {
            return Err(refused("replay_atoms", "atoms differ from the stored result's"));
        }
        let recomputed = evaluate_structural(contract, atoms)
            .map_err(|e| refused("replay_evaluation", &format!("{e:?}")))?;
        let stored = to_cbor(&result_to_body(&self.result, &self.atom_digests))?;
        let fresh = to_cbor(&result_to_body(&recomputed, &digests))?;
        if stored != fresh {
            return Err(refused(
                "replay_mismatch",
                "recomputation does not reproduce the stored result",
            ));
        }
        Ok(())
    }
}

//! Portable decision contract and decision result artifacts.
//!
//! A contract artifact is identified by the contract's canonical digest; a
//! result artifact additionally binds the digest of the draws it was computed
//! from. Both load only under an identity the consumer retained independently
//! of the bytes. A result can be replayed: given the contract and the source,
//! the decision is recomputed and must reproduce the stored result exactly, so
//! a stored number never needs the producing process to be explained.

use antecedent_core::ScientificQuantity;
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

use crate::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use crate::decision_eval::{
    ActionOutcome, ConstraintExclusion, DecisionResult, SourceReceipt, Verdict, evaluate_contract,
};

/// Maximum artifact bytes accepted before any decode allocation.
pub const MAX_DECISION_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
const CONTRACT_KIND: &str = "decision_contract_v1";
const RESULT_KIND: &str = "decision_result_v1";
const BODY_SECTION: &str = "decision.body";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExprWire {
    Const(f64),
    Input(usize),
    Neg(Box<ExprWire>),
    Add(Box<ExprWire>, Box<ExprWire>),
    Sub(Box<ExprWire>, Box<ExprWire>),
    Mul(Box<ExprWire>, Box<ExprWire>),
    Min(Box<ExprWire>, Box<ExprWire>),
    Max(Box<ExprWire>, Box<ExprWire>),
}

impl From<&UtilityExpr> for ExprWire {
    fn from(expr: &UtilityExpr) -> Self {
        let b = |e: &UtilityExpr| Box::new(Self::from(e));
        match expr {
            UtilityExpr::Const(v) => Self::Const(*v),
            UtilityExpr::Input(k) => Self::Input(*k),
            UtilityExpr::Neg(a) => Self::Neg(b(a)),
            UtilityExpr::Add(x, y) => Self::Add(b(x), b(y)),
            UtilityExpr::Sub(x, y) => Self::Sub(b(x), b(y)),
            UtilityExpr::Mul(x, y) => Self::Mul(b(x), b(y)),
            UtilityExpr::Min(x, y) => Self::Min(b(x), b(y)),
            UtilityExpr::Max(x, y) => Self::Max(b(x), b(y)),
        }
    }
}

impl From<ExprWire> for UtilityExpr {
    fn from(wire: ExprWire) -> Self {
        let b = |e: Box<ExprWire>| Box::new(Self::from(*e));
        match wire {
            ExprWire::Const(v) => Self::Const(v),
            ExprWire::Input(k) => Self::Input(k),
            ExprWire::Neg(a) => Self::Neg(b(a)),
            ExprWire::Add(x, y) => Self::Add(b(x), b(y)),
            ExprWire::Sub(x, y) => Self::Sub(b(x), b(y)),
            ExprWire::Mul(x, y) => Self::Mul(b(x), b(y)),
            ExprWire::Min(x, y) => Self::Min(b(x), b(y)),
            ExprWire::Max(x, y) => Self::Max(b(x), b(y)),
        }
    }
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionWire {
    id: String,
    kind: String,
    inputs: Vec<ScientificQuantityWire>,
    utility: ExprWire,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstraintWire {
    id: String,
    expr: ExprWire,
    bound: f64,
    min_probability: f64,
    units: String,
    applies_to: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractBody {
    version: u16,
    #[serde(default)]
    identity: String,
    actions: Vec<ActionWire>,
    utility_units: String,
    criterion: CriterionWire,
    constraints: Vec<ConstraintWire>,
    target_population: String,
    horizon: u32,
    structural_policy: String,
}

fn kind_name(kind: ActionKind) -> &'static str {
    match kind {
        ActionKind::Intervention => "intervention",
        ActionKind::Policy => "policy",
        ActionKind::Regime => "regime",
        ActionKind::Study => "study",
        ActionKind::External => "external",
    }
}

fn kind_from(name: &str) -> Result<ActionKind, IoError> {
    Ok(match name {
        "intervention" => ActionKind::Intervention,
        "policy" => ActionKind::Policy,
        "regime" => ActionKind::Regime,
        "study" => ActionKind::Study,
        "external" => ActionKind::External,
        _ => return Err(IoError::Convert(format!("unknown action kind `{name}`"))),
    })
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

fn contract_from_body(body: ContractBody) -> Result<DecisionContract, IoError> {
    if body.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
    }
    let actions = body
        .actions
        .into_iter()
        .map(|a| {
            let inputs = a
                .inputs
                .into_iter()
                .map(|q| ScientificQuantity::try_from(q).map_err(|e| IoError::Convert(e.into())))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DecisionAction {
                id: a.id,
                kind: kind_from(&a.kind)?,
                inputs,
                utility: a.utility.into(),
            })
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(DecisionContract {
        actions,
        utility_units: body.utility_units,
        criterion: body.criterion.into(),
        constraints: body
            .constraints
            .into_iter()
            .map(|c| HardConstraint {
                id: c.id,
                expr: c.expr.into(),
                bound: c.bound,
                min_probability: c.min_probability,
                units: c.units,
                applies_to: c.applies_to,
            })
            .collect(),
        target_population: body.target_population,
        horizon: body.horizon,
        structural_policy: policy_from(&body.structural_policy)?,
    })
}

fn body_from_contract(contract: &DecisionContract) -> Result<ContractBody, IoError> {
    let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
    Ok(ContractBody {
        version: 1,
        identity,
        actions: contract
            .actions
            .iter()
            .map(|a| ActionWire {
                id: a.id.clone(),
                kind: kind_name(a.kind).to_owned(),
                inputs: a.inputs.iter().map(ScientificQuantityWire::from).collect(),
                utility: ExprWire::from(&a.utility),
            })
            .collect(),
        utility_units: contract.utility_units.clone(),
        criterion: contract.criterion.into(),
        constraints: contract
            .constraints
            .iter()
            .map(|c| ConstraintWire {
                id: c.id.clone(),
                expr: ExprWire::from(&c.expr),
                bound: c.bound,
                min_probability: c.min_probability,
                units: c.units.clone(),
                applies_to: c.applies_to.clone(),
            })
            .collect(),
        target_population: contract.target_population.clone(),
        horizon: contract.horizon,
        structural_policy: policy_name(contract.structural_policy).to_owned(),
    })
}

fn encode(kind: &str, artifact_id: &str, note: &str, body: &[u8]) -> Result<Vec<u8>, IoError> {
    if artifact_id.trim().is_empty() {
        return Err(IoError::Convert("missing artifact id".into()));
    }
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: antecedent_io::migrate::STABLE_FORMAT,
            minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(kind.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
            artifact_id: artifact_id.into(),
            sections: vec![section_descriptor(BODY_SECTION, "application/cbor", body)],
            provenance: ProvenanceWire { note: note.into() },
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

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8], kind: &str) -> Result<T, IoError> {
    if bytes.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(kind.into())
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

/// A decision contract with its canonical identity.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionContractArtifact {
    contract: DecisionContract,
    identity: String,
}

impl DecisionContractArtifact {
    /// Own a validated contract.
    ///
    /// # Errors
    /// An invalid contract refuses.
    pub fn new(contract: DecisionContract) -> Result<Self, IoError> {
        let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        Ok(Self { contract, identity })
    }

    /// The contract.
    #[must_use]
    pub fn contract(&self) -> &DecisionContract {
        &self.contract
    }

    /// Canonical identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload or invalid contract refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(&body_from_contract(&self.contract)?)?;
        encode(CONTRACT_KIND, artifact_id, "decision_contract", &body)
    }

    /// Load under an independently retained identity.
    ///
    /// # Errors
    /// A corrupt, oversized or semantically changed artifact refuses, including
    /// one whose stored identity was resealed to match an edited contract.
    pub fn from_bytes(bytes: &[u8], expected_identity: &str) -> Result<Self, IoError> {
        let body: ContractBody = decode(bytes, CONTRACT_KIND)?;
        let stored = body.identity.clone();
        let contract = contract_from_body(body)?;
        let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        if identity != stored || identity != expected_identity {
            return Err(refused(
                "identity_expected",
                "contract identity differs from the consumer's retained identity",
            ));
        }
        Ok(Self { contract, identity })
    }
}

/// BLAKE3 of a source's aligned draws and coordinates, retained by a result.
#[must_use]
pub fn source_digest(source: &DistributionArtifact) -> String {
    let mut hasher = blake3::Hasher::new();
    for quantity in source.quantities() {
        hasher.update(format!("{quantity:?}").as_bytes());
    }
    hasher.update(&(source.n_draws() as u64).to_le_bytes());
    for value in source.draws() {
        hasher.update(&value.to_le_bytes());
    }
    if let Some(weights) = &source.metadata().weights {
        for weight in weights {
            hasher.update(&weight.to_le_bytes());
        }
    }
    hasher.finalize().to_hex().to_string()
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
enum VerdictWire {
    UniquelyOptimal(String),
    Indistinguishable(Vec<String>),
    NoAdmissibleAction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultBody {
    version: u16,
    contract_identity: String,
    source_digest: String,
    criterion: CriterionWire,
    actions: Vec<OutcomeWire>,
    verdict: VerdictWire,
    evpi: Option<f64>,
    n_draws: u64,
    effective_draws: f64,
    provider_id: String,
    snapshot_id: String,
    rng_id: String,
    causal_contract_id: String,
    assumptions: Vec<String>,
}

fn result_to_body(result: &DecisionResult, digest: &str) -> ResultBody {
    ResultBody {
        version: 1,
        contract_identity: result.contract_identity.clone(),
        source_digest: digest.to_owned(),
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
            Verdict::UniquelyOptimal(id) => VerdictWire::UniquelyOptimal(id.clone()),
            Verdict::Indistinguishable(ids) => VerdictWire::Indistinguishable(ids.clone()),
            Verdict::NoAdmissibleAction => VerdictWire::NoAdmissibleAction,
        },
        evpi: result.evpi,
        n_draws: result.n_draws as u64,
        effective_draws: result.effective_draws,
        provider_id: result.source.provider_id.clone(),
        snapshot_id: result.source.snapshot_id.clone(),
        rng_id: result.source.rng_id.clone(),
        causal_contract_id: result.source.causal_contract_id.clone(),
        assumptions: result.assumptions.clone(),
    }
}

fn body_to_result(body: ResultBody) -> Result<DecisionResult, IoError> {
    if body.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
    }
    Ok(DecisionResult {
        contract_identity: body.contract_identity,
        criterion: body.criterion.into(),
        actions: body
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
        verdict: match body.verdict {
            VerdictWire::UniquelyOptimal(id) => Verdict::UniquelyOptimal(id),
            VerdictWire::Indistinguishable(ids) => Verdict::Indistinguishable(ids),
            VerdictWire::NoAdmissibleAction => Verdict::NoAdmissibleAction,
        },
        evpi: body.evpi,
        n_draws: usize::try_from(body.n_draws).map_err(|_| IoError::TooLarge)?,
        effective_draws: body.effective_draws,
        source: SourceReceipt {
            provider_id: body.provider_id,
            snapshot_id: body.snapshot_id,
            rng_id: body.rng_id,
            causal_contract_id: body.causal_contract_id,
        },
        assumptions: body.assumptions,
    })
}

/// A decision result bound to its contract and source digest.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResultArtifact {
    result: DecisionResult,
    source_digest: String,
}

impl DecisionResultArtifact {
    /// Bind a result to the source it was computed from.
    #[must_use]
    pub fn new(result: DecisionResult, source: &DistributionArtifact) -> Self {
        Self { result, source_digest: source_digest(source) }
    }

    /// The stored result.
    #[must_use]
    pub fn result(&self) -> &DecisionResult {
        &self.result
    }

    /// Digest of the source draws the result was computed from.
    #[must_use]
    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(&result_to_body(&self.result, &self.source_digest))?;
        encode(RESULT_KIND, artifact_id, "decision_result", &body)
    }

    /// Load under the consumer's retained contract identity and source digest.
    ///
    /// # Errors
    /// A corrupt or changed artifact, or one bound to a different contract or
    /// source, refuses.
    pub fn from_bytes(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_source_digest: &str,
    ) -> Result<Self, IoError> {
        let body: ResultBody = decode(bytes, RESULT_KIND)?;
        if body.contract_identity != expected_contract_identity {
            return Err(refused(
                "contract_expected",
                "result was computed under a different decision contract",
            ));
        }
        if body.source_digest != expected_source_digest {
            return Err(refused("source_expected", "result was computed from different draws"));
        }
        let digest = body.source_digest.clone();
        Ok(Self { result: body_to_result(body)?, source_digest: digest })
    }

    /// Recompute the decision from `contract` and `source` and require the
    /// stored result to match exactly.
    ///
    /// # Errors
    /// A different contract or source, a refused evaluation or any differing
    /// number refuses.
    pub fn replay(
        &self,
        contract: &DecisionContract,
        source: &DistributionArtifact,
    ) -> Result<(), IoError> {
        let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        if identity != self.result.contract_identity {
            return Err(refused("replay_contract", "contract differs from the stored result's"));
        }
        if source_digest(source) != self.source_digest {
            return Err(refused("replay_source", "source differs from the stored result's"));
        }
        let recomputed = evaluate_contract(contract, source)
            .map_err(|e| refused("replay_evaluation", &format!("{e:?}")))?;
        let stored = to_cbor(&result_to_body(&self.result, &self.source_digest))?;
        let fresh = to_cbor(&result_to_body(&recomputed, &self.source_digest))?;
        if stored != fresh {
            return Err(refused(
                "replay_mismatch",
                "recomputation does not reproduce the stored result",
            ));
        }
        Ok(())
    }
}

/// Parse a contract from its JSON declaration (the wire a host language builds).
/// A stored `identity` field is ignored: the identity is always recomputed.
///
/// # Errors
/// Malformed JSON, unknown fields or an invalid declaration refuse.
pub fn contract_from_json(json: &str) -> Result<DecisionContract, IoError> {
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let body: ContractBody =
        serde_json::from_str(json).map_err(|e| IoError::Convert(e.to_string()))?;
    let contract = contract_from_body(body)?;
    contract.validate().map_err(|e| IoError::Convert(format!("{e:?}")))?;
    Ok(contract)
}

/// The JSON declaration of a contract, including its recomputed identity.
///
/// # Errors
/// An invalid contract refuses.
pub fn contract_to_json(contract: &DecisionContract) -> Result<String, IoError> {
    serde_json::to_string(&body_from_contract(contract)?)
        .map_err(|e| IoError::Convert(e.to_string()))
}

/// The JSON form of a result bound to its source digest. Values that are
/// undefined for an excluded action are `null`.
///
/// # Errors
/// Serialization failure.
pub fn result_to_json(
    result: &DecisionResult,
    source: &DistributionArtifact,
) -> Result<String, IoError> {
    serde_json::to_string(&result_to_body(result, &source_digest(source)))
        .map_err(|e| IoError::Convert(e.to_string()))
}
